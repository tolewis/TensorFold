//! Port of wojo's GLM 0110 (PR #124, spill v2 block staging) and 0118 (PR #152, a carried or failed destination's
//! reset): the launcher of zig/kernels/cuda/dsv41_port.cu's one kernel, `tf_ds_port_planes_kernel`, with its CPU
//! reference (the host tests here, and tf-dsv41-port's GPU check on a Spark, port_main.zig).
//!
//! UNTESTED ON A GPU: the .cu is compiled by nvcc on a Spark (`-Dnvcc=...`); run `tf-dsv41-port` there before any
//! feature relies on it. Nothing in the served engine calls it yet (every switch that would is off by default).
//!
//! `Port.planes`: up to 32 planes' rows of n blocks to or from one contiguous staging buffer, or reset to their fill
//! bytes, in one launch. A spill caller goes through `packBlocksDev`, `unpackBlocksDev` and `clearBlocksDev`: each is
//! the spill_state.Packer call of the same name with Packer's own checks (checkBlocks, the pin current, clear's
//! generation bump) and Ticket, its copies one launch into or out of a device pack buffer (PORT-SPILL-V2: "one
//! measured device pack buffer"), the planes built from the extent's base at call time. A pack followed by one
//! cuMemcpyDtoHAsync of the buffer stages exactly the bytes `Packer.packBlocks` stages with 16 copies a block.
//! `spillBlocks` (the plane set alone, no extent or generation checks) is for those three and tf-dsv41-port only.
//!
//! Everything else the GLM port moves is covered by kernels that exist (docs/port/kit.md has the map): the sparse fetch,
//! owner-decode mask and select and the prompt union (dsv41_shard.cu, shard_ops.zig), row gathers/scatters
//! (ops.gatherRows / scatterRows), a strided 2-D copy (ops.copyRows), and the cross-part carry (Comm.sendSpans /
//! recvSpans straight from the planes, no staging).
const std = @import("std");
const cuda = @import("cuda");
const spill = @import("spill_state.zig");

/// Threads a block.
pub const threads = 256;
/// The grid's y/z cap the call checks against (blocks a call).
pub const max_grid_z = 65535;

/// dsv41_port.cu TfDsPortPlane: rows row0 + b * rows .. + rows of the plane at ptr (rows ld bytes apart, row_bytes each)
/// for block b of a call, at stage_off within each staged block; fill: the byte mode `fill` writes.
pub const Plane = extern struct { ptr: u64 = 0, ld: u64 = 0, row_bytes: u64 = 0, row0: u64 = 0, rows: u64 = 0, stage_off: u64 = 0, fill: u32 = 0, pad: u32 = 0 };
pub const max_planes = 32;
/// dsv41_port.cu TfDsPortPlanes, passed by value (1,800 bytes; a launch's arguments fit cuda.Args' 2,048).
pub const Planes = extern struct {
    plane: [max_planes]Plane = @splat(Plane{}),
    n: u32 = 0,
    pad: u32 = 0,

    pub fn add(s: *Planes, p: Plane) !void {
        if (s.n == max_planes) return error.TooManyPlanes;
        s.plane[s.n] = p;
        s.n += 1;
    }

    pub fn slice(s: *Planes) []Plane {
        return s.plane[0..s.n];
    }

    pub fn constSlice(s: *const Planes) []const Plane {
        return s.plane[0..s.n];
    }

    /// Each plane's rows one after another in a staged block, every offset 16-byte aligned (stage_off set here);
    /// returns the block's bytes.
    pub fn layout(s: *Planes) !u64 {
        var off: u64 = 0;
        for (s.slice()) |*p| {
            try checkPlane(p.*);
            off = std.mem.alignForward(u64, off, 16);
            p.stage_off = off;
            off += p.rows * p.row_bytes;
        }
        return std.mem.alignForward(u64, off, 16);
    }

    /// Every plane well formed and inside a staged block of block_bytes; no two planes' staged rows overlap.
    pub fn check(s: *const Planes, block_bytes: u64) !void {
        if (s.n > max_planes) return error.TooManyPlanes;
        const ps = s.constSlice();
        for (ps, 0..) |p, i| {
            try checkPlane(p);
            const end = std.math.add(u64, p.stage_off, p.rows * p.row_bytes) catch return error.BadPlane;
            if (end > block_bytes) return error.StagedPastBlock;
            for (ps[0..i]) |q| {
                const qe = q.stage_off + q.rows * q.row_bytes;
                if (p.rows > 0 and q.rows > 0 and p.stage_off < qe and q.stage_off < end) return error.StagedOverlap;
            }
        }
    }
};

fn checkPlane(p: Plane) !void {
    if (p.row_bytes == 0 or p.ld < p.row_bytes or p.fill > 255) return error.BadPlane;
    _ = std.math.mul(u64, p.rows, p.ld) catch return error.BadPlane;
    _ = std.math.mul(u64, p.rows, p.row_bytes) catch return error.BadPlane;
}

comptime {
    // the CUDA structs' layout (dsv41_port.cu): 8-byte fields, no padding but the explicit one
    std.debug.assert(@sizeOf(Plane) == 56 and @sizeOf(Planes) == max_planes * 56 + 8);
    std.debug.assert(@offsetOf(Planes, "n") == max_planes * 56);
}

/// The kernel's modes (its `mode`).
pub const Mode = enum(c_int) { pack = 0, unpack = 1, fill = 2 };

/// The kernel's grid x: blocks enough for the largest plane's 16-byte words, 1 .. 1024.
pub fn planesGridX(s: *const Planes) u32 {
    var most: u64 = 0;
    for (s.constSlice()) |p| most = @max(most, p.rows * p.row_bytes);
    const blocks = (most / 16 + threads - 1) / threads;
    return @intCast(std.math.clamp(blocks, 1, 1024));
}

/// The plane set of a spill point's blocks [first_block, first_block + count) of the extent at pool position `base`, in
/// spill_state's block layout: block plane i of `l` at stage offset l.blockOffset(i), rows base / ratio +
/// (first_block + b) * (block_tokens / ratio) of the plane whose row 0 is `addr[i]` (Model.SpillDev.blockPlane), its
/// fill byte the plane's sentinel. Staging block b is at b * l.block_bytes: the bytes Packer.packBlocks stages.
/// Refused: owner_sharded (not served: the Packer's copies pack a sharded point; this kit path would need the owner's
/// planes only, with the point's owner, through Packer.checkBlocks' seam), a base off a block boundary (BadView), an
/// address of 0 (a plane this rank does not hold) or more planes than one launch takes. It checks no extent bounds
/// and no generations: serving code calls the *Dev functions below, never this.
pub fn spillBlocks(l: *const spill.StateLayout, addr: []const u64, base: usize, first_block: usize) !Planes {
    if (l.owner_layout != .replicated) return error.OwnerShardedNotPorted;
    const bt = l.geo.block_tokens;
    if (bt == 0 or base % bt != 0) return error.BadView;
    const bp = l.blockPlanes();
    if (addr.len != bp.len) return error.BadPlane;
    if (bp.len > max_planes) return error.TooManyPlanes;
    var s: Planes = .{};
    for (bp, addr, 0..) |p, a, i| {
        if (a == 0) return error.BadPlane;
        const rows: u64 = bt / p.ratio;
        const row0 = std.math.add(u64, base / p.ratio, try std.math.mul(u64, first_block, rows)) catch return error.Overflow;
        try s.add(.{ .ptr = a, .ld = p.row, .row_bytes = p.row, .row0 = row0, .rows = rows, .stage_off = l.blockOffset(i), .fill = p.sentinel });
    }
    try s.check(l.block_bytes);
    return s;
}

/// spillBlocks with the addresses from a spill Dev (Model.SpillDev, or a test's).
fn spillBlocksOf(dev: anytype, l: *const spill.StateLayout, base: usize, first_block: usize) !Planes {
    var addr: [spill.max_planes]u64 = undefined;
    const bp = l.blockPlanes();
    for (bp, 0..) |p, i| addr[i] = dev.blockPlane(p);
    return spillBlocks(l, addr[0..bp.len], base, first_block);
}

pub const Port = struct {
    module: cuda.Module,
    planes_fn: cuda.Function,

    /// cuda.kernels.dsv41_port (or the fatbin's bytes).
    pub fn load(d: *const cuda.Driver, image: []const u8) !Port {
        if (image.len == 0) return error.NoPortKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .planes_fn = try m.function("tf_ds_port_planes_kernel") };
    }

    pub fn unload(p: *Port) void {
        p.module.unload();
        p.* = undefined;
    }

    /// This port on a stream: the launcher packBlocksDev, unpackBlocksDev and clearBlocksDev take.
    pub fn on(p: *const Port, s: cuda.Stream) On {
        return .{ .port = p, .stream = s };
    }

    pub const On = struct {
        port: *const Port,
        stream: cuda.Stream,

        pub fn planes(o: On, set: *const Planes, staging: u64, block_bytes: u64, n_blocks: usize, mode: Mode) !void {
            return o.port.planes(o.stream, set, staging, block_bytes, n_blocks, mode);
        }
    };

    /// n_blocks blocks of every plane of `set` to (pack) or from (unpack) staging, block b at staging + b * block_bytes,
    /// or (fill) the planes' rows set to their fill byte (staging unused).
    pub fn planes(p: *const Port, s: cuda.Stream, set: *const Planes, staging: u64, block_bytes: u64, n_blocks: usize, mode: Mode) !void {
        if (set.n == 0 or n_blocks == 0) return;
        try checkCall(set, block_bytes, n_blocks, mode);
        if (mode != .fill and staging == 0) return error.BadStaging;
        var args: cuda.Args = .{};
        args.add(set.*);
        args.add(staging);
        args.add(block_bytes);
        args.add(@as(c_uint, @intCast(n_blocks)));
        args.add(@intFromEnum(mode));
        try cuda.launch.launch(p.planes_fn, .{ .grid = .{ .x = planesGridX(set), .y = set.n, .z = @intCast(n_blocks) }, .block = .{ .x = threads } }, s, &args);
    }
};

// -- spill_state.Packer's block calls in one launch -----------------------------------------------------------------
// `pk`: a *spill_state.Packer(Dev); `launcher`: Port.on(stream), or ref.On over host memory in the tests; `done`:
// the Dev's event, recorded after the launch (and on an error after the checks, as Packer records it).

/// Packer.packBlocks in one launch: blocks [first_block, first_block + count) of the point pinned as `src` into the
/// device buffer `staging` (staging_bytes = count * block_bytes); one DtoH of it then holds packBlocks' bytes.
/// error.StaleExtent: the rows were given away since the pin.
pub fn packBlocksDev(pk: anytype, launcher: anytype, src: spill.ExtentRef, first_block: usize, count: usize, staging: u64, staging_bytes: usize, done: anytype) !spill.Ticket {
    try pk.checkBlocks(src, first_block, count, staging_bytes);
    if (!pk.gens.current(src)) return error.StaleExtent;
    try blocksCall(pk, launcher, src.lo, first_block, count, staging, .pack, done);
    return .{ .op = .pack_blocks, .extent = src, .first_block = first_block, .blocks = count, .bytes = staging_bytes };
}

/// Packer.unpackBlocks in one launch: the staged blocks in the device buffer `staging` (after one HtoD) back into the
/// reserved destination `dst`. The restore publishes only while Generations.valid(ticket) holds after `done`.
pub fn unpackBlocksDev(pk: anytype, launcher: anytype, dst: spill.ExtentRef, first_block: usize, count: usize, staging: u64, staging_bytes: usize, done: anytype) !spill.Ticket {
    try pk.checkBlocks(dst, first_block, count, staging_bytes);
    if (!pk.gens.current(dst)) return error.StaleExtent;
    try blocksCall(pk, launcher, dst.lo, first_block, count, staging, .unpack, done);
    return .{ .op = .unpack_blocks, .extent = dst, .first_block = first_block, .blocks = count, .bytes = staging_bytes };
}

/// Packer.clearExtent in one launch, for a destination of whole blocks (lo and hi on block boundaries; error.BadView
/// otherwise, where clearExtent's ratio-aligned fills still serve): every block plane's rows at its sentinel. Bumps
/// the rows' generation first, as clearExtent does, so a restore still in flight over them goes stale.
pub fn clearBlocksDev(pk: anytype, launcher: anytype, dst: spill.ExtentRef, done: anytype) !spill.Ticket {
    const bt = pk.layout.geo.block_tokens;
    if (dst.hi <= dst.lo or dst.hi > pk.dev.capacity()) return error.BadRange;
    if (bt == 0 or dst.lo % bt != 0 or dst.hi % bt != 0) return error.BadView;
    const count = (dst.hi - dst.lo) / bt;
    if (count > max_grid_z) return error.BadShape;
    if (!pk.gens.current(dst)) return error.StaleExtent;
    const set = try spillBlocksOf(pk.dev, pk.layout, dst.lo, 0); // a refusal here leaves the generations alone
    pk.gens.bump(dst.lo, dst.hi);
    errdefer pk.dev.record(done) catch {};
    try launcher.planes(&set, 0, 0, count, .fill);
    try pk.dev.record(done);
    return .{ .op = .clear_extent, .extent = try pk.gens.pin(dst.lo, dst.hi), .bytes = 0 };
}

fn blocksCall(pk: anytype, launcher: anytype, base: usize, first_block: usize, count: usize, staging: u64, mode: Mode, done: anytype) !void {
    if (staging == 0) return error.BadStaging;
    if (count > max_grid_z) return error.BadShape;
    errdefer pk.dev.record(done) catch {};
    const set = try spillBlocksOf(pk.dev, pk.layout, base, first_block); // the extent's base now, not when it was pinned
    try launcher.planes(&set, staging, pk.layout.block_bytes, count, mode);
    try pk.dev.record(done);
}

fn checkCall(set: *const Planes, block_bytes: u64, n_blocks: usize, mode: Mode) !void {
    if (n_blocks > max_grid_z) return error.BadShape;
    if (mode == .fill) return set.check(std.math.maxInt(u64));
    try set.check(block_bytes);
    _ = std.math.mul(u64, block_bytes, n_blocks) catch return error.BadStaging;
}

/// The kernel's semantics on the host: `mem` stands for device memory, every address an offset into it. The host tests
/// check them here; tf-dsv41-port compares the GPU's bytes with them on a Spark.
pub const ref = struct {
    fn copy(mem: []u8, dst: u64, src: u64, n: u64) void {
        @memmove(mem[@intCast(dst)..][0..@intCast(n)], mem[@intCast(src)..][0..@intCast(n)]);
    }

    fn set(mem: []u8, dst: u64, v: u8, n: u64) void {
        @memset(mem[@intCast(dst)..][0..@intCast(n)], v);
    }

    /// Port.On over host memory: the launcher the *Dev functions' tests pass.
    pub const On = struct {
        mem: []u8,

        pub fn planes(o: On, s: *const Planes, staging: u64, block_bytes: u64, n_blocks: usize, mode: Mode) !void {
            return ref.planes(o.mem, s, staging, block_bytes, n_blocks, mode);
        }
    };

    /// Port.planes.
    pub fn planes(mem: []u8, s: *const Planes, staging: u64, block_bytes: u64, n_blocks: usize, mode: Mode) !void {
        if (s.n == 0 or n_blocks == 0) return;
        try checkCall(s, block_bytes, n_blocks, mode); // (staging 0 is mem[0] here; the device call refuses null)
        for (0..n_blocks) |b| for (s.constSlice()) |p| for (0..@intCast(p.rows)) |r| {
            const row = p.ptr + (p.row0 + b * p.rows + r) * p.ld;
            const st = staging + b * block_bytes + p.stage_off + r * p.row_bytes;
            switch (mode) {
                .pack => copy(mem, st, row, p.row_bytes),
                .unpack => copy(mem, row, st, p.row_bytes),
                .fill => set(mem, row, @intCast(p.fill), p.row_bytes),
            }
        };
    }
};

// -- host tests ---------------------------------------------------------------------------------------------------

const testing = std.testing;

test "launch arguments fit cuda.Args" {
    var a: cuda.Args = .{};
    const s: Planes = .{};
    a.add(s);
    a.add(@as(u64, 0));
    a.add(@as(u64, 0));
    a.add(@as(c_uint, 0));
    a.add(@as(c_int, 0));
    try testing.expect(a.used <= cuda.Args.max_bytes);
}

test "planes pack, unpack and fill round-trip" {
    var mem: [16384]u8 = @splat(0);
    var prng: std.Random.DefaultPrng = .init(110);
    prng.random().bytes(mem[0..8192]);
    var s: Planes = .{};
    // a packed plane (ld = row_bytes), a strided one, an odd-sized one: 3 blocks from rows 2, 1 and 0
    try s.add(.{ .ptr = 0, .ld = 48, .row_bytes = 48, .row0 = 2 * 4, .rows = 4 });
    try s.add(.{ .ptr = 2048, .ld = 80, .row_bytes = 64, .row0 = 2, .rows = 2, .fill = 127 });
    try s.add(.{ .ptr = 4096, .ld = 13, .row_bytes = 13, .row0 = 0, .rows = 5 });
    const bb = try s.layout();
    for (s.constSlice()) |p| try testing.expect(p.stage_off % 16 == 0);
    try s.check(bb);
    const staging: u64 = 8192;
    const before = mem;
    try ref.planes(&mem, &s, staging, bb, 3, .pack);
    for (0..3) |b| for (s.constSlice()) |p| for (0..@intCast(p.rows)) |r| {
        const row = p.ptr + (p.row0 + b * p.rows + r) * p.ld;
        try testing.expectEqualSlices(u8, before[@intCast(row)..][0..@intCast(p.row_bytes)], mem[@intCast(staging + b * bb + p.stage_off + r * p.row_bytes)..][0..@intCast(p.row_bytes)]);
    };
    // fill the planes' rows, then unpack: every byte back as it was (the ld gaps untouched throughout)
    try ref.planes(&mem, &s, 0, 0, 3, .fill);
    try testing.expect(std.mem.allEqual(u8, mem[2048 + 2 * 80 ..][0..64], 127));
    try testing.expect(!std.mem.eql(u8, before[0..8192], mem[0..8192]));
    try ref.planes(&mem, &s, staging, bb, 3, .unpack);
    try testing.expectEqualSlices(u8, before[0..8192], mem[0..8192]);
}

test "planes checks" {
    var s: Planes = .{};
    try s.add(.{ .ld = 4, .row_bytes = 8 });
    try testing.expectError(error.BadPlane, s.layout());
    s = .{};
    try s.add(.{ .ld = 8, .row_bytes = 8, .rows = 2 });
    try s.add(.{ .ld = 8, .row_bytes = 8, .rows = 2, .stage_off = 8 });
    try testing.expectError(error.StagedOverlap, s.check(64));
    try testing.expectError(error.StagedPastBlock, s.check(16));
    var full: Planes = .{};
    for (0..max_planes) |_| try full.add(.{ .ld = 1, .row_bytes = 1 });
    try testing.expectError(error.TooManyPlanes, full.add(.{ .ld = 1, .row_bytes = 1 }));
    var one: Planes = .{};
    try one.add(.{ .ld = 16, .row_bytes = 16, .rows = 1 });
    try testing.expectError(error.BadShape, checkCall(&one, 16, max_grid_z + 1, .pack));
    try testing.expectError(error.StagedPastBlock, checkCall(&one, 8, 1, .pack));
    try checkCall(&one, 0, 1, .fill);
}

test "grid" {
    var s: Planes = .{};
    try s.add(.{ .ld = 288, .row_bytes = 288, .rows = 2048 });
    try testing.expectEqual(@as(u32, 144), planesGridX(&s));
    s = .{};
    try s.add(.{ .ld = 4, .row_bytes = 4, .rows = 1 });
    try testing.expectEqual(@as(u32, 1), planesGridX(&s));
}

/// A spill Dev over a host array (addresses are offsets into `mem`), copies immediate: enough of Model.SpillDev for
/// spill_state.Packer's block calls.
const HostDev = struct {
    mem: []u8,
    cap: usize,
    addr: []const u64, // block plane i's row 0
    l: *const spill.StateLayout,

    pub const Event = u32;

    pub fn blockPlane(d: HostDev, p: spill.Plane) u64 {
        for (d.l.blockPlanes(), d.addr) |q, a| if (q.kind == p.kind and q.layer == p.layer) return a;
        return 0;
    }
    pub fn slotPlane(_: HostDev, _: spill.Plane, _: usize) u64 {
        return 0;
    }
    pub fn capacity(d: HostDev) usize {
        return d.cap;
    }
    pub fn slots(_: HostDev) usize {
        return 1;
    }
    pub fn d2h(d: HostDev, dst: []u8, src: u64) !void {
        @memcpy(dst, d.mem[@intCast(src)..][0..dst.len]);
    }
    pub fn h2d(d: HostDev, dst: u64, src: []const u8) !void {
        @memcpy(d.mem[@intCast(dst)..][0..src.len], src);
    }
    pub fn fill(d: HostDev, dst: u64, value: u8, bytes: usize) !void {
        @memset(d.mem[@intCast(dst)..][0..bytes], value);
    }
    pub fn record(_: HostDev, _: Event) !void {}
};

test "spillBlocks stages exactly what spill_state.Packer.packBlocks stages" {
    const gpa = testing.allocator;
    // the served checkpoint's block planes (sources at ratios 2, 2, 2, 1, all with index keys; head_dim 512, index
    // head_dim 128) over a 64-block-token geometry so the pool fits a test array
    const shapes = [_]spill.LayerShape{
        .{},
        .{ .ratio = 2, .kv = true, .idx = true, .raw = true },
        .{ .ratio = 2, .kv = true, .idx = true, .raw = true },
        .{},
        .{ .ratio = 2, .kv = true, .idx = true, .raw = true },
        .{ .ratio = 1, .kv = true, .idx = true },
    };
    const g: spill.Geometry = .{ .block_tokens = 64, .window = 128, .ring_extra = 16, .raw_rows = 64, .head_dim = 512, .index_head_dim = 128 };
    const l = try spill.StateLayout.init(&shapes, g, .replicated);
    try testing.expectEqual(@as(usize, 16), l.blockPlanes().len);
    try testing.expectEqual(@as(usize, 890), l.token_bytes);
    const cap: usize = 6 * 64; // six blocks of positions
    // carve each plane's (cap / ratio + 2) rows as Caches.init does, from offset 64 (0 stays "no plane")
    var addr: [16]u64 = undefined;
    var at: u64 = 64;
    for (l.blockPlanes(), 0..) |p, i| {
        addr[i] = at;
        at += (cap / p.ratio + 2) * p.row;
    }
    const pool_end = at;
    const count = 3;
    const staging_dev = std.mem.alignForward(u64, pool_end, 16);
    const mem = try gpa.alloc(u8, @intCast(staging_dev + count * l.block_bytes));
    defer gpa.free(mem);
    var prng: std.Random.DefaultPrng = .init(0x0110);
    prng.random().bytes(mem);
    const dev: HostDev = .{ .mem = mem, .cap = cap, .addr = &addr, .l = &l };
    var gens = try spill.Generations.init(gpa, cap, 64);
    defer gens.deinit(gpa);
    var pk = spill.Packer(HostDev){ .dev = dev, .layout = &l, .gens = &gens };
    // the extent at base 128 (block 2 of the pool), blocks 1 .. 3 of it
    const base: usize = 128;
    const pin = try gens.pin(base, cap);
    const host = try gpa.alloc(u8, count * l.block_bytes);
    defer gpa.free(host);
    _ = try pk.packBlocks(pin, 1, count, host, 0);
    // the same blocks in one launch into a device pack buffer: packBlocks' bytes, packBlocks' ticket
    const on: ref.On = .{ .mem = mem };
    const t = try packBlocksDev(&pk, on, pin, 1, count, staging_dev, host.len, 0);
    try testing.expectEqual(spill.Op.pack_blocks, t.op);
    try testing.expectEqual(@as(usize, count), t.blocks);
    try testing.expectEqualSlices(u8, host, mem[@intCast(staging_dev)..][0..host.len]);
    try testing.expectEqual(@as(u32, 16), (try spillBlocksOf(dev, &l, base, 1)).n);
    // unpack the device staging into a reserved extent (base 0), then Packer's pack of it gives the same bytes again
    const dst = try gens.reserve(0, count * 64);
    const ut = try unpackBlocksDev(&pk, on, dst, 0, count, staging_dev, host.len, 0);
    try testing.expect(gens.valid(ut));
    const again = try gpa.alloc(u8, host.len);
    defer gpa.free(again);
    _ = try pk.packBlocks(try gens.pin(0, cap), 0, count, again, 0);
    try testing.expectEqualSlices(u8, host, again);
    // Packer's checks hold: blocks past the extent's end, a staging size off count blocks, a stale pin, null staging
    try testing.expectError(error.BadRange, packBlocksDev(&pk, on, dst, 1, count, staging_dev, host.len, 0));
    try testing.expectError(error.BadStaging, packBlocksDev(&pk, on, pin, 1, count, staging_dev, host.len - 1, 0));
    try testing.expectError(error.StaleExtent, packBlocksDev(&pk, on, pin, 1, count, staging_dev, host.len, 0)); // dst's reserve took [128, 192)
    try testing.expectError(error.BadStaging, packBlocksDev(&pk, on, try gens.pin(base, cap), 1, count, 0, host.len, 0));
    try testing.expectError(error.BadRange, unpackBlocksDev(&pk, on, .{ .lo = base, .hi = cap + 64, .stamp = 0 }, 0, 1, staging_dev, l.block_bytes, 0));
    // clearBlocksDev is clearExtent: generations bumped first (the restore into dst goes stale), every block plane's
    // rows of those blocks at its sentinel (index scales 127, else 0), a fresh pin in the ticket
    const ct = try clearBlocksDev(&pk, on, dst, 0);
    try testing.expect(!gens.valid(ut));
    try testing.expect(gens.valid(ct));
    try testing.expectError(error.StaleExtent, unpackBlocksDev(&pk, on, dst, 0, count, staging_dev, host.len, 0));
    try testing.expectError(error.StaleExtent, clearBlocksDev(&pk, on, dst, 0));
    for (l.blockPlanes(), 0..) |p, i| {
        const rows = count * 64 / p.ratio;
        try testing.expect(std.mem.allEqual(u8, mem[@intCast(addr[i])..][0 .. rows * p.row], p.sentinel));
    }
    // clearExtent takes ratio-aligned ranges; the one-launch clear only whole blocks, past the pool never
    try testing.expectError(error.BadView, clearBlocksDev(&pk, on, try gens.pin(64, 64 + 32), 0));
    try testing.expectError(error.BadRange, clearBlocksDev(&pk, on, .{ .lo = 0, .hi = cap + 64, .stamp = 0 }, 0));
    // refusals: a base off a block, a missing plane, owner_sharded
    try testing.expectError(error.BadView, spillBlocks(&l, &addr, 32, 0));
    var holed = addr;
    holed[3] = 0;
    try testing.expectError(error.BadPlane, spillBlocks(&l, &holed, 0, 0));
    try testing.expectError(error.BadPlane, spillBlocks(&l, addr[0..15], 0, 0));
    const ls = try spill.StateLayout.init(&shapes, g, .owner_sharded);
    try testing.expectError(error.OwnerShardedNotPorted, spillBlocks(&ls, &addr, 0, 0));
}

//! tf-dsv41-port: dsv41_port.cu's plane staging kernel on this GPU against port_kernels.zig's CPU reference, every byte
//! of a 4 MiB arena compared (the GLM 0110/0118 port's block pack/unpack/fill; docs/port/kit.md). No model, no kit, one
//! GPU; a few seconds. Exit 0: every case equal; 1: a case differs (its first differing byte printed).
//! (The sparse fetch, select and union kernels are dsv41_shard.cu's: their host emulation is tools/dsv41_shard_emulate.)
//!
//! Needs a build with the kernels (`-Dnvcc=...` or `-Dfatbins=...`) and `-Dport_kernels`; otherwise it exits 2.
//!
//! `tf-dsv41-port --gate KIT [ROWS ...]` (default rows 17 .. 32): the MoE gate's chunk sums (tri_norm.rowmmGate, K
//! 5120, N 384, PDL on) of one random round on KIT's Triton set (KIT/aot), every row's bytes compared across launch
//! shapes: the engine's 16-row pieces against each row launched alone (RB 1, as 1-row rounds run it), and, when the
//! kit has the single-launch variant (build_kit.sh EXTRA_ROWS), against one RB = rows launch. Row invariance is what
//! the split relies on (docs/port/kit.md section 4); this shows it or not, before any token-hash comparison. Needs
//! no -Dport_kernels and no model. UNTESTED ON A GPU.
const std = @import("std");
const cuda = @import("cuda");
const pk = @import("port_kernels.zig");
const spill = @import("spill_state.zig");
const tri = @import("tri.zig");
const tri_norm = @import("tri_norm.zig");

const arena_bytes = 4 << 20;

/// One launch, its addresses offsets into the arena plus `base` (0 for the reference, the device buffer for the GPU).
const Call = union(enum) {
    planes: struct { set: pk.Planes, staging: u64, block_bytes: u64, n_blocks: usize, mode: pk.Mode },
};

const Case = struct {
    name: []const u8,
    /// Writes the case's inputs (indices, not addresses) into the arena.
    prepare: *const fn (mem: []u8, rnd: std.Random) void,
    /// The case's call with its addresses at `base`.
    call: *const fn (base: u64) anyerror!Call,
};

fn nothing(mem: []u8, rnd: std.Random) void {
    _ = mem;
    _ = rnd;
}

/// A spill block's planes: comp codes/scales of four sources (ratios 2, 2, 2, 1) and their index planes, packed (ld =
/// row) the way prompt.Caches lays them out, plus one strided odd plane; two blocks from block 3 on.
fn planesCase(comptime mode: pk.Mode) Case {
    const S = struct {
        fn call(base: u64) anyerror!Call {
            const tokens: u64 = 256; // a 2048-token block scaled down 8x so the planes fit the arena
            var set: pk.Planes = .{};
            var at: u64 = 0;
            for ([_]u64{ 2, 2, 2, 1 }) |ratio| for ([_]u64{ 256, 32, 64, 4 }) |rb| {
                const rows = tokens / ratio;
                try set.add(.{ .ptr = base + at, .ld = rb, .row_bytes = rb, .row0 = 3 * rows, .rows = rows, .fill = if (rb == 4) 127 else 0 });
                at += 6 * rows * rb;
            };
            try set.add(.{ .ptr = base + at, .ld = 37, .row_bytes = 29, .row0 = 1, .rows = 7, .fill = 0x5A });
            const bb = try set.layout();
            return .{ .planes = .{ .set = set, .staging = base + (5 << 19), .block_bytes = bb, .n_blocks = 2, .mode = mode } };
        }
    };
    return .{ .name = "planes " ++ @tagName(mode) ++ " (16 spill planes + an odd strided one, 2 blocks)", .prepare = nothing, .call = S.call };
}

/// The served checkpoint's block planes in spill_state's layout (spillBlocks), 256-token blocks, 3 blocks of an extent
/// at pool block 2: the bytes spill_state.Packer would stage with 48 copies, in one launch.
fn spillCase(comptime mode: pk.Mode) Case {
    const S = struct {
        fn call(base: u64) anyerror!Call {
            const shapes = [_]spill.LayerShape{
                .{ .ratio = 2, .kv = true, .idx = true, .raw = true },
                .{ .ratio = 2, .kv = true, .idx = true, .raw = true },
                .{ .ratio = 2, .kv = true, .idx = true, .raw = true },
                .{ .ratio = 1, .kv = true, .idx = true },
            };
            const g: spill.Geometry = .{ .block_tokens = 256, .window = 128, .ring_extra = 16, .raw_rows = 64, .head_dim = 512, .index_head_dim = 128 };
            const l = try spill.StateLayout.init(&shapes, g, .replicated);
            const cap: usize = 8 * 256;
            var addr: [16]u64 = undefined;
            var at: u64 = 0;
            for (l.blockPlanes(), 0..) |p, i| {
                addr[i] = base + at;
                at += (cap / p.ratio + 2) * p.row;
            }
            const set = try pk.spillBlocks(&l, &addr, 2 * 256, 1);
            return .{ .planes = .{ .set = set, .staging = base + (3 << 20), .block_bytes = l.block_bytes, .n_blocks = 3, .mode = mode } };
        }
    };
    return .{ .name = "spill blocks " ++ @tagName(mode) ++ " (spill_state layout, 16 planes, 3 blocks of 256 tokens)", .prepare = nothing, .call = S.call };
}

const cases = [_]Case{
    planesCase(.pack),
    planesCase(.unpack),
    planesCase(.fill),
    spillCase(.pack),
    spillCase(.unpack),
    spillCase(.fill),
};

fn runRef(mem: []u8, c: Call) !void {
    switch (c) {
        .planes => |p| try pk.ref.planes(mem, &p.set, p.staging, p.block_bytes, p.n_blocks, p.mode),
    }
}

fn runGpu(port: *const pk.Port, s: cuda.Stream, c: Call) !void {
    switch (c) {
        .planes => |p| try port.planes(s, &p.set, p.staging, p.block_bytes, p.n_blocks, p.mode),
    }
}

/// --gate: the round's gate at its served shape, every row's chunk sums compared bitwise across launch shapes.
fn gateCheck(gpa: std.mem.Allocator, io: std.Io, kit: []const u8, rows_list: []const usize) !u8 {
    const k: usize = 5120;
    const n: usize = 384;
    const kc = k / 256;
    const most: usize = 48;
    for (rows_list) |r| if (r == 0 or r > most) return error.BadRows;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const aot_dir = try std.fs.path.join(gpa, &.{ kit, "aot" });
    defer gpa.free(aot_dir);
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, aot_dir);
    defer set.deinit();
    // X [48, K] bf16 in [-1, 1), W [N, K] fp16 in [-1/16, 1/16): the gate's magnitudes, no inf or NaN
    const x_bytes = most * k * 2;
    const w_bytes = n * k * 2;
    const s_bytes = most * kc * n * 4;
    const x_off: usize = 0;
    const w_off = std.mem.alignForward(usize, x_off + x_bytes, 256);
    var so: [3]usize = undefined;
    so[0] = std.mem.alignForward(usize, w_off + w_bytes, 256);
    so[1] = std.mem.alignForward(usize, so[0] + s_bytes, 256);
    so[2] = std.mem.alignForward(usize, so[1] + s_bytes, 256);
    var dev = try cuda.DeviceBuffer.alloc(&driver, so[2] + s_bytes);
    defer dev.free();
    var prng: std.Random.DefaultPrng = .init(0x0118_0032);
    const rnd = prng.random();
    const xw = try gpa.alloc(u8, w_off + w_bytes);
    defer gpa.free(xw);
    @memset(xw, 0);
    for (0..most * k) |i| std.mem.writeInt(u16, xw[x_off + 2 * i ..][0..2], @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16), .little);
    for (0..n * k) |i| std.mem.writeInt(u16, xw[w_off + 2 * i ..][0..2], @bitCast(@as(f16, @floatCast((rnd.float(f32) * 2 - 1) / 16))), .little);
    try dev.upload(0, xw);
    const got = [3][]u8{ try gpa.alloc(u8, s_bytes), try gpa.alloc(u8, s_bytes), try gpa.alloc(u8, s_bytes) };
    defer for (got) |g| gpa.free(g);
    const t: tri.Tri = .{ .set = &set, .stream = stream };
    const x = dev.ptr + x_off;
    const w = dev.ptr + w_off;
    const row_bytes = kc * n * 4;
    var failed: usize = 0;
    for (rows_list) |rows| {
        // 0: the engine's 16-row pieces; 1: each row alone; 2: one launch, when the kit resolves it
        const single = rows > tri_norm.rowmm_parts_piece and tri_norm.rowmmPartsPiece(t, x, k, w, dev.ptr + so[2], rows, k, n) == rows;
        const pieces = [3]usize{ tri_norm.rowmm_parts_piece, 1, rows };
        const shapes: usize = if (single) 3 else 2;
        for (0..shapes) |i| {
            try tri_norm.rowmmGatePieces(t, x, k, w, dev.ptr + so[i], rows, k, n, pieces[i]);
        }
        try stream.synchronize();
        for (0..shapes) |i| try dev.download(so[i], got[i][0 .. rows * row_bytes]);
        var bad = false;
        for (1..shapes) |i| if (std.mem.indexOfDiff(u8, got[0][0 .. rows * row_bytes], got[i][0 .. rows * row_bytes])) |at| {
            bad = true;
            std.debug.print("FAIL gate {d} rows: 16-row pieces differ from {s} at row {d} (byte {d} of the row)\n", .{ rows, if (i == 1) "rows alone" else "one launch", at / row_bytes, at % row_bytes });
        };
        if (bad) failed += 1 else std.debug.print("ok   gate {d} rows: 16-row pieces == rows alone{s}\n", .{ rows, if (single) " == one RB-rows launch" else " (kit has no single launch)" });
    }
    std.debug.print("{d} of {d} gate row counts equal\n", .{ rows_list.len - failed, rows_list.len });
    return if (failed == 0) 0 else 1;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--gate")) {
        if (args.len < 3) {
            std.debug.print("usage: tf-dsv41-port --gate KIT [ROWS ...]\n", .{});
            return 2;
        }
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(gpa);
        for (args[3..]) |r| try list.append(gpa, try std.fmt.parseInt(usize, r, 10));
        if (list.items.len == 0) for (17..33) |r| try list.append(gpa, r);
        return gateCheck(gpa, init.io, args[2], list.items);
    }
    if (!cuda.kernels.available) {
        std.debug.print("tf-dsv41-port: built without kernel images (build with -Dnvcc=... or -Dfatbins=...)\n", .{});
        return 2;
    }
    if (cuda.kernels.dsv41_port.len == 0) {
        std.debug.print("tf-dsv41-port: built without the opt-in dsv41_port fatbin (build with -Dport_kernels)\n", .{});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var port = try pk.Port.load(&driver, cuda.kernels.dsv41_port);
    defer port.unload();
    var dev = try cuda.DeviceBuffer.alloc(&driver, arena_bytes);
    defer dev.free();
    const host = try gpa.alloc(u8, arena_bytes);
    defer gpa.free(host);
    const want = try gpa.alloc(u8, arena_bytes);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, arena_bytes);
    defer gpa.free(got);
    var failed: usize = 0;
    for (cases, 0..) |c, ci| {
        var prng: std.Random.DefaultPrng = .init(0x0118_0110 + ci);
        const rnd = prng.random();
        rnd.bytes(host);
        c.prepare(host, rnd);
        const ref_call = try c.call(0);
        const gpu_call = try c.call(dev.ptr);
        @memcpy(want, host);
        try runRef(want, ref_call);
        try dev.upload(0, host);
        try runGpu(&port, stream, gpu_call);
        try stream.synchronize();
        try dev.download(0, got);
        if (std.mem.indexOfDiff(u8, want, got)) |at| {
            failed += 1;
            std.debug.print("FAIL {s}: first difference at byte {d} (want {x:0>2}, got {x:0>2})\n", .{ c.name, at, want[at], got[at] });
        } else {
            std.debug.print("ok   {s}\n", .{c.name});
        }
    }
    std.debug.print("{d} of {d} cases equal\n", .{ cases.len - failed, cases.len });
    return if (failed == 0) 0 else 1;
}

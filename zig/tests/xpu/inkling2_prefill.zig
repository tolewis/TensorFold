//! Inkling 2 Mini's row-batched prompt glue on XPU under the contract (kernels/xpu/inkling2/prefill.cl). Each kernel here
//! is the batched form of a kernel that already holds its own contract (glue.cl, dense.cl) or a pure function of its
//! inputs, and does the same arithmetic in the same order, so its registration `replaces` that kernel: gate 3 holds its
//! bytes equal to the replaced kernel's (or to the host's exact copy) on the same inputs, over launches that span more
//! rows than the replaced kernel takes. Every kernel has a planted twin (one edit of the production source, made by
//! inkling2-control-gen) that must go red at gate 3.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const images = @import("xpu_contract_images");
const host = @import("inkling2_prefill_host");
const glue_host = @import("inkling2_xpu_glue");
const dense_host = @import("inkling2_dense_host");
const loop_host = @import("inkling2_xpu_loop");

const c = core.spec;
const g = core.gates;

const HID: usize = 2304;
const D: usize = 128; // PF_D: values of a K/V head row
const EXPERTS: usize = 128;
const REL: usize = 16;
const EXTENT: usize = 1024;

pub const Kind = enum {
    copy_rows,
    perm,
    kv_copy,
    conv_static,
    router,
    ar,
    dense,
    rel_table,
    messages,

    fn name(k: Kind) []const u8 {
        return @tagName(k);
    }
    /// What the registration says the kernel replaces.
    fn replaces(k: Kind) []const u8 {
        return switch (k) {
            .copy_rows => "the host's strided row copy",
            .perm => "the host's plane reordering",
            .kv_copy => "the host's ring copy",
            .conv_static => "tf_inkling2_conv_static",
            .router => "tf_inkling2_router",
            .ar => "tf_inkling2_ar",
            .dense => "tf_ink2_dense_32",
            .rel_table => "the host's relative-bias sum",
            .messages => "tf_inkling2_loop_messages",
        };
    }
    fn fault(k: Kind) []const u8 {
        return switch (k) {
            .copy_rows => "source column",
            .perm => "parity",
            .kv_copy => "destination ring",
            .conv_static => "tap scale",
            .router => "bias",
            .ar => "scale",
            .dense => "activation half",
            .rel_table => "tap order",
            .messages => "scale",
        };
    }
};

pub fn title(k: Kind) []const u8 {
    return switch (k) {
        inline else => |x| "xpu inkling2 prefill glue tf_ink2_pf_" ++ @tagName(x),
    };
}

fn controlName(k: Kind) []const u8 {
    return switch (k) {
        inline else => |x| "control: xpu inkling2 prefill glue tf_ink2_pf_" ++ @tagName(x) ++ " " ++ comptime x.fault(),
    };
}

/// The argument counts gate 5 checks a loaded kernel against.
pub fn argsOf(symbol: []const u8) ?u32 {
    const table = [_]struct { []const u8, u32 }{
        .{ "copy_rows", 6 },
        .{ "perm", 4 },
        .{ "kv_copy", 7 },
        .{ "conv_static", 7 },
        .{ "router", 6 },
        .{ "ar", 7 },
        .{ "dense", 7 },
        .{ "rel_table", 3 },
        .{ "messages", 6 },
    };
    const prefix = if (std.mem.startsWith(u8, symbol, "tf_ink2_pf_")) "tf_ink2_pf_" else if (std.mem.startsWith(u8, symbol, "tf_ctl_ink2_pf_")) "tf_ctl_ink2_pf_" else return null;
    const rest = symbol[prefix.len..];
    for (table) |e| if (std.mem.eql(u8, rest, e[0])) return e[1];
    return null;
}

fn zero(_: c.Shape) usize {
    return 0;
}

// Call shapes: `rows` is the launch's row count; `n` and `k` carry the kernel's own sizes where it has them. Row counts
// run past the 32 and 64 rows the replaced kernels take.
const copy_served = [_]c.Shape{ .{ .rows = 1, .n = 1000, .k = 1100 }, .{ .rows = 5, .n = 1000, .k = 1100 }, .{ .rows = 70, .n = 1000, .k = 1100 } };
const perm_served = [_]c.Shape{ .{ .rows = 1, .n = HID, .k = HID }, .{ .rows = 5, .n = HID, .k = HID }, .{ .rows = 70, .n = HID, .k = HID } };
const kv_served = [_]c.Shape{ .{ .rows = 1, .n = 8, .k = D }, .{ .rows = 40, .n = 8, .k = D }, .{ .rows = 79, .n = 8, .k = D } };
const conv_served = [_]c.Shape{ .{ .rows = 1, .n = HID, .k = 4 }, .{ .rows = 5, .n = 1024, .k = 4 }, .{ .rows = 70, .n = HID, .k = 4 } };
const router_served = [_]c.Shape{ .{ .rows = 1, .n = 8, .k = HID }, .{ .rows = 7, .n = 8, .k = HID }, .{ .rows = 70, .n = 8, .k = HID } };
const ar_served = [_]c.Shape{ .{ .rows = 2, .n = HID, .k = 1 }, .{ .rows = 5, .n = HID, .k = 3 }, .{ .rows = 70, .n = HID, .k = 2 } };
const dense_served = [_]c.Shape{ .{ .rows = 5, .n = 200, .k = HID }, .{ .rows = 40, .n = 203, .k = HID }, .{ .rows = 70, .n = 200, .k = HID } };
const messages_served = [_]c.Shape{ .{ .rows = 1, .n = HID, .k = 48 }, .{ .rows = 5, .n = HID, .k = 48 }, .{ .rows = 70, .n = HID, .k = 48 } };
const rel_served = [_]c.Shape{ .{ .rows = 8, .n = EXTENT, .k = REL }, .{ .rows = 24, .n = EXTENT, .k = REL }, .{ .rows = 72, .n = EXTENT, .k = REL } };

pub fn spec(k: Kind, control: bool) c.Spec {
    const served: []const c.Shape = switch (k) {
        .copy_rows => &copy_served,
        .perm => &perm_served,
        .kv_copy => &kv_served,
        .conv_static => &conv_served,
        .router => &router_served,
        .ar => &ar_served,
        .dense => &dense_served,
        .rel_table => &rel_served,
        .messages => &messages_served,
    };
    return .{
        .name = if (control) controlName(k) else title(k),
        .accum = .none,
        .depth = zero,
        .out = .raw,
        .replaces = k.replaces(),
        .served = if (control) served[1..2] else served,
    };
}

fn bits(rng: std.Random, scale: f32) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * scale);
}

fn fillBf16(out: []u16, rng: std.Random, scale: f32) void {
    for (out) |*v| v.* = bits(rng, scale);
}

/// Poisons a few values of an input for the non-finite fill: NaN, +inf, -inf.
fn poison(v: []u16) void {
    v[0] = c.toBf16(std.math.nan(f32));
    v[v.len / 2] = c.toBf16(std.math.inf(f32));
    v[v.len - 1] = c.toBf16(-std.math.inf(f32));
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    ctl: bool,
    ks: *const host.Kernels, // the kernels under test (a control: its twin, the rest production)
    glue: *const glue_host.Pipelines,
    dense: *const dense_host.Kernels,
    loop: *const loop_host.Pipelines,
    shape: c.Shape = undefined,
    // the baseline: the replaced kernel's (or the host's) bytes of every output, in the rig's output order
    base: []u8 = &.{},
    // device addresses
    p: [8]?*anyopaque = @splat(null),
    o: [3]?*anyopaque = @splat(null),

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, ks: *const host.Kernels, glue: *const glue_host.Pipelines, dense: *const dense_host.Kernels, loop: *const loop_host.Pipelines, kind: Kind, control: bool) Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .dev = dev }, .kind = kind, .ctl = control, .ks = ks, .glue = glue, .dense = dense, .loop = loop };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
    }

    fn releaseAll(ctx: *anyopaque) void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.rig.reset();
        _ = p.arena.reset(.free_all);
        p.base = &.{};
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.ctl), .ctx = p, .vt = &vt, .control = if (p.ctl) .shapes else null };
    }

    const vt = blk: {
        var t = g.hooks(Probe);
        t.release = releaseAll;
        break :blk t;
    };

    fn accepts(p: *const Probe, s: c.Shape) bool {
        if (s.rows == 0 or s.rows > 128) return false;
        return switch (p.kind) {
            .copy_rows => s.n > 0 and s.n <= s.k,
            .perm => s.k == s.n and s.k % 256 == 0,
            .kv_copy => s.n == 8 and s.k == D,
            .conv_static => s.k == 4 and s.n % 4 == 0 and s.n <= HID,
            .router => s.n == 8 and s.k == HID,
            .ar => s.n == HID and s.k >= 1 and s.k <= 3 and s.rows >= 2 and s.rows <= 128,
            .dense => s.k == HID and s.n > 0,
            .rel_table => s.k == REL and s.n == EXTENT and s.rows % 8 == 0,
            .messages => s.n == HID and s.k == 48,
        };
    }

    fn put(p: *Probe, data: anytype) !?*anyopaque {
        return p.rig.input(std.mem.sliceAsBytes(data));
    }

    /// The baseline's bytes for the poisoned outputs: 0xff where the kernel writes nothing.
    fn poisoned(p: *Probe, len: usize) ![]u8 {
        const b = try p.arena.allocator().alloc(u8, len);
        @memset(b, 0xff);
        return b;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!p.accepts(shape)) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        const a = p.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const nf = fill == .nonfinite;
        switch (p.kind) {
            .copy_rows => {
                const x = try a.alloc(u16, rows * shape.k);
                for (x, 0..) |*v, i| v.* = @truncate((i *% 2654435761) >> 7);
                p.p[0] = try p.put(x);
                const ldy = 1024;
                p.o[0] = try p.rig.output(rows * ldy * 2);
                const want = try a.alloc(u16, rows * ldy);
                @memset(std.mem.sliceAsBytes(want), 0xff);
                for (0..rows) |r| @memcpy(want[r * ldy ..][0..shape.n], x[r * shape.k ..][0..shape.n]);
                p.base = std.mem.sliceAsBytes(want);
            },
            .perm => {
                const x = try a.alloc(u16, rows * shape.k);
                for (x, 0..) |*v, i| v.* = @truncate((i *% 2654435761) >> 7);
                p.p[0] = try p.put(x);
                p.o[0] = try p.rig.output(rows * shape.k * 2);
                const want = try a.alloc(u16, rows * shape.k);
                for (0..rows) |r| for (0..shape.k / 256) |plane| for (0..256) |o| {
                    const s = o >> 4;
                    const l = o & 15;
                    want[r * shape.k + plane * 256 + o] = x[r * shape.k + plane * 256 + (s >> 1) * 32 + 2 * l + (s & 1)];
                };
                p.base = std.mem.sliceAsBytes(want);
            },
            .kv_copy => {
                const src_ring = 96;
                const dst_ring = 80;
                const src = try a.alloc(u16, 8 * src_ring * D);
                for (src, 0..) |*v, i| v.* = @truncate((i *% 2654435761) >> 7);
                p.p[0] = try p.put(src);
                p.o[0] = try p.rig.output(8 * dst_ring * D * 2);
                const want = try a.alloc(u16, 8 * dst_ring * D);
                @memset(std.mem.sliceAsBytes(want), 0xff);
                const from = 70;
                for (0..8) |h| for (0..rows) |i| {
                    const pos = from + i;
                    @memcpy(want[(h * dst_ring + pos % dst_ring) * D ..][0..D], src[(h * src_ring + pos % src_ring) * D ..][0..D]);
                };
                p.base = std.mem.sliceAsBytes(want);
            },
            .conv_static => {
                const ch = shape.n;
                const hist = try a.alloc(u16, 3 * ch);
                const cur = try a.alloc(u16, rows * ch);
                const w = try a.alloc(u16, 4 * ch);
                fillBf16(hist, rng, 1);
                fillBf16(cur, rng, 1);
                fillBf16(w, rng, 1);
                if (nf) poison(cur);
                p.p[0] = try p.put(hist);
                p.p[1] = try p.put(cur);
                p.p[2] = try p.put(w);
                p.o[0] = try p.rig.output((3 + rows) * ch * 2); // taps
                p.o[1] = try p.rig.output(rows * ch * 2); // out
                // the replaced kernel, on its own buffers
                const taps = try p.rig.scratch((3 + rows) * ch * 2);
                const out = try p.rig.scratch(rows * ch * 2);
                try glue_host.convStatic(p.glue, p.rig.dev.s, p.p[0], p.p[1], taps, p.p[2], out, @intCast(rows), @intCast(ch));
                try p.rig.dev.s.synchronize();
                const t = try a.dupe(u8, try p.rig.bytes(taps, (3 + rows) * ch * 2));
                const o = try a.dupe(u8, try p.rig.bytes(out, rows * ch * 2));
                p.base = try std.mem.concat(a, u8, &.{ t, o });
            },
            .router => {
                const x = try a.alloc(u16, rows * HID);
                const w = try a.alloc(u16, EXPERTS * HID);
                const bias = try a.alloc(f32, EXPERTS);
                fillBf16(x, rng, 1);
                fillBf16(w, rng, 0.05);
                for (bias) |*v| v.* = rng.float(f32) - 0.5;
                if (nf) poison(x);
                p.p[0] = try p.put(x);
                p.p[1] = try p.put(w);
                p.p[2] = try p.put(bias);
                p.o[0] = try p.rig.output(rows * 8 * 4); // ids
                p.o[1] = try p.rig.output(rows * 8 * 4); // weights
                const ids = try p.rig.scratch(rows * 8 * 4);
                const weights = try p.rig.scratch(rows * 8 * 4);
                // the replaced kernel takes at most 64 rows a launch (runtime_plan's bound): launch it a row at a time
                for (0..rows) |r| try glue_host.router(p.glue, p.rig.dev.s, at(p.p[0], r * HID * 2), p.p[1], p.p[2], at(ids, r * 32), at(weights, r * 32), 1);
                try p.rig.dev.s.synchronize();
                const i = try a.dupe(u8, try p.rig.bytes(ids, rows * 32));
                const wv = try a.dupe(u8, try p.rig.bytes(weights, rows * 32));
                p.base = try std.mem.concat(a, u8, &.{ i, wv });
            },
            .ar => {
                // sources = shape.k + 1 (blocks and the partial); queries = shape.k
                const sources: usize = shape.k + 1;
                const queries: usize = shape.k;
                const blocks = try a.alloc(u16, rows * 4 * HID);
                const partial = try a.alloc(f32, rows * HID);
                const q = try a.alloc(f32, queries * HID);
                fillBf16(blocks, rng, 1);
                for (partial) |*v| v.* = rng.float(f32) * 2 - 1;
                for (q) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
                if (nf) {
                    poison(blocks);
                    q[0] = std.math.inf(f32);
                }
                p.p[0] = try p.put(blocks);
                p.p[1] = try p.put(partial);
                p.p[2] = try p.put(q);
                p.o[0] = try p.rig.output(queries * rows * HID * 4);
                const out = try p.rig.scratch(queries * rows * HID * 4);
                try glue_host.arRows(p.glue, p.rig.dev.s, p.p[0], p.p[1], p.p[2], out, @intCast(sources), @intCast(queries), @intCast(rows), @intCast(4 * HID));
                try p.rig.dev.s.synchronize();
                p.base = try a.dupe(u8, try p.rig.bytes(out, queries * rows * HID * 4));
            },
            .dense => {
                const n = shape.n;
                const x = try a.alloc(u16, rows * HID);
                const w = try a.alloc(u16, n * HID);
                fillBf16(x, rng, 1);
                fillBf16(w, rng, 0.05);
                if (nf) poison(x);
                p.p[0] = try p.put(x);
                p.p[1] = try p.put(w);
                p.o[0] = try p.rig.output(rows * n * 2);
                const out = try p.rig.scratch(rows * n * 2);
                try dense_host.dense(p.dense, p.rig.dev.s, p.p[1], p.p[0], out, @intCast(rows), @intCast(n), @intCast(HID));
                try p.rig.dev.s.synchronize();
                p.base = try a.dupe(u8, try p.rig.bytes(out, rows * n * 2));
            },
            .messages => {
                const bank0 = try a.alloc(u16, rows * HID);
                const bank1 = try a.alloc(u16, rows * HID);
                const q_mlp = try a.alloc(f32, 24 * HID);
                const kv = try a.alloc(f32, 24 * HID);
                fillBf16(bank0, rng, 1);
                fillBf16(bank1, rng, 1);
                for (q_mlp) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
                for (kv) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
                if (nf) {
                    poison(bank0);
                    q_mlp[0] = std.math.inf(f32);
                }
                p.p[0] = try p.put(bank0);
                p.p[1] = try p.put(bank1);
                p.p[2] = try p.put(q_mlp);
                p.p[3] = try p.put(kv);
                p.o[0] = try p.rig.output(48 * rows * HID * 4); // [entry][row][H]
                // the replaced kernel lays its output [row][entry][H]; one row a launch keeps its scratch under 2 MiB
                const want = try a.alloc(u8, 48 * rows * HID * 4);
                const tmp = try p.rig.scratch(48 * HID * 4);
                for (0..rows) |r| {
                    try loop_host.crossPassMessageAttention(p.loop, p.rig.dev.s, at(p.p[0], r * HID * 2), at(p.p[1], r * HID * 2), p.p[2], p.p[3], tmp, 1);
                    const got = try p.rig.bytes(tmp, 48 * HID * 4);
                    for (0..48) |e| @memcpy(want[((e * rows + r) * HID) * 4 ..][0 .. HID * 4], got[(e * HID) * 4 ..][0 .. HID * 4]);
                }
                p.base = want;
            },
            .rel_table => {
                const rel = try a.alloc(u16, rows * REL);
                const proj = try a.alloc(u16, REL * EXTENT);
                fillBf16(rel, rng, 1);
                fillBf16(proj, rng, 1);
                if (nf) poison(rel);
                p.p[0] = try p.put(rel);
                p.p[1] = try p.put(proj);
                p.o[0] = try p.rig.output(rows * EXTENT * 2);
                // attention.cl's table: sixteen taps added in order, each product and sum rounded in binary32
                const want = try a.alloc(u16, rows * EXTENT);
                for (0..rows) |rh| for (0..EXTENT) |d| {
                    var acc: f32 = 0;
                    for (0..REL) |j| acc += c.fromBf16(rel[rh * REL + j]) * c.fromBf16(proj[j * EXTENT + d]);
                    want[rh * EXTENT + d] = c.toBf16(acc);
                };
                p.base = std.mem.sliceAsBytes(want);
            },
        }
        return true;
    }

    fn at(ptr: ?*anyopaque, off: usize) ?*anyopaque {
        return @ptrFromInt(@intFromPtr(ptr.?) + off);
    }

    pub fn run(p: *Probe) !void {
        const s = p.rig.dev.s;
        const ks = p.ks;
        const sh = p.shape;
        const rows: u32 = @intCast(sh.rows);
        switch (p.kind) {
            .copy_rows => try host.copyRows(ks, s, p.p[0], p.o[0], rows, @intCast(sh.n), @intCast(sh.k), 1024),
            .perm => try host.permute(ks, s, p.p[0], p.o[0], rows, @intCast(sh.k)),
            .kv_copy => try host.kvCopy(ks, s, p.p[0], p.o[0], 70, 96, 80, rows, 8),
            .conv_static => try host.convStatic(ks, s, p.p[0], p.p[1], p.o[0], p.p[2], p.o[1], rows, @intCast(sh.n)),
            .router => try host.router(ks, s, p.p[0], p.p[1], p.p[2], p.o[0], p.o[1], rows),
            .ar => try host.ar(ks, s, p.p[0], p.p[1], p.p[2], p.o[0], @intCast(sh.k + 1), @intCast(sh.k), rows),
            .dense => try host.denseBf16(ks, s, p.p[1], p.p[0], p.o[0], rows, @intCast(sh.n), @intCast(HID)),
            .rel_table => try host.relTable(ks, s, rows, p.p[0], p.p[1], p.o[0]),
            .messages => try host.messages(ks, s, p.p[0], p.p[1], p.p[2], p.p[3], p.o[0], rows),
        }
        try s.synchronize();
    }

    pub fn count(p: *const Probe) usize {
        return p.base.len;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        _ = p;
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = p;
        _ = acc;
        _ = order;
        @memset(out, 0);
    }

    /// The replaced kernel's bytes on these inputs.
    pub fn replaced(p: *Probe) ![]const u8 {
        return p.base;
    }

    /// With non-finite inputs the kernel and what it replaces must give the same bytes, NaN and infinity classes included.
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite) return null;
        const got = try p.rig.outputBytes();
        if (!std.mem.eql(u8, got, p.base)) return "its bytes differ from the replaced kernel's on non-finite inputs";
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    production: host.Kernels,
    glue: glue_host.Pipelines,
    dense: dense_host.Kernels,
    loop: loop_host.Pipelines,
    twins: [9]host.Kernels = undefined,
    twins_made: usize = 0,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Suite {
        try dev.addImage("inkling2_prefill", &images.inkling2_prefill);
        try dev.addImage("inkling2_prefill_ctl", &images.inkling2_prefill_ctl);
        var production = try host.Kernels.load(dev.named("inkling2_prefill").?.*);
        errdefer production.deinit();
        var glue = try glue_host.Pipelines.load((try dev.module(.ink2_glue)).*, false);
        errdefer glue.deinit();
        var dense = try dense_host.Kernels.load(dev.named("inkling2_dense").?.*, false);
        errdefer dense.deinit();
        const loop = try loop_host.Pipelines.load((try dev.module(.inkling2_loop)).*);
        return .{ .a = a, .dev = dev, .production = production, .glue = glue, .dense = dense, .loop = loop };
    }

    /// Every kernel and, after it, its planted twin: the production kernels with the one kernel swapped for the twin.
    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel)) !void {
        const planted = s.dev.named("inkling2_prefill_ctl").?;
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            var ks: *const host.Kernels = &s.production;
            if (control) {
                s.twins[s.twins_made] = try host.Kernels.load(s.dev.named("inkling2_prefill").?.*);
                try s.twins[s.twins_made].swap(planted.*, kind.name());
                ks = &s.twins[s.twins_made];
                s.twins_made += 1;
            }
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = Probe.init(s.a, s.dev, ks, &s.glue, &s.dense, &s.loop, kind, control);
            try s.probes.append(s.a, probe);
            try kernels.append(s.a, probe.kernel());
        };
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
        for (s.twins[0..s.twins_made]) |*t| t.deinit();
        s.loop.deinit();
        s.dense.deinit();
        s.glue.deinit();
        s.production.deinit();
    }
};

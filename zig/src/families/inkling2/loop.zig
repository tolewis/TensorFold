//! Inkling 2 Mini's loop launches (inkling2/loop.cu): cross-pass message mixes and the core inputs' rank-r gates.
//! C2 stand-in: the arithmetic #4337's Metal implementation runs; the kernel contract for these is C2's slice.
const std = @import("std");
const cuda = @import("cuda");

pub const max_blocks = 8;
pub const max_queries = 4;
pub const max_gates = 3;
pub const max_rank = 128;
const threads = 256;
const lin_threads = 128; // loop.cu kLin

/// loop.cu's LoopMixArgs, field for field.
pub const MixArgs = extern struct {
    blocks: [max_blocks]u64 = @splat(0),
    partial: u64,
    query: [max_queries]u64 = @splat(0),
    out: u64,
    out_stride: i64,
    n_blocks: i32,
    partial_f32: i32,
    nq: i32,
    rows: i32,
    h: i32,
    eps: f32,
    scale: f32,
};

/// loop.cu's LoopGateArgs, field for field.
pub const GateArgs = extern struct {
    h: [max_gates]u64 = @splat(0),
    msg: u64,
    msg_stride: i64,
    slot: [max_gates]i32 = @splat(0),
    down_w: [max_gates]u64 = @splat(0),
    forget: [max_gates]u64 = @splat(0),
    gain: [max_gates]u64 = @splat(0),
    norm: [max_gates]u64 = @splat(0),
    down: u64,
    out: u64,
    out_stride: i64,
    rows: i32,
    h_dim: i32,
    rank: i32,
    eps_unit: f32,
    eps: f32,
    ys: u64 = 0, // fp32 [gates][rows, H] scratch: the gated rows before their norm
};

pub const Kernels = struct {
    mod: cuda.Module,
    mix: cuda.Function,
    mix_layers: cuda.Function,
    gate_down: cuda.Function,
    gate_lin: cuda.Function,
    gate_mix: cuda.Function,
    gate_prep: cuda.Function,
    gate_down_rows: cuda.Function,
    gate_lin_rows: cuda.Function,
    tiled: bool,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .mix = try mod.function("tf_ink2_loop_mix"),
            .mix_layers = try mod.function("tf_ink2_loop_mix_layers"),
            .gate_down = try mod.function("tf_ink2_loop_gate_down"),
            .gate_lin = try mod.function("tf_ink2_loop_gate_lin"),
            .gate_mix = try mod.function("tf_ink2_loop_gate_mix"),
            .gate_prep = try mod.function("tf_ink2_loop_gate_prep"),
            .gate_down_rows = try mod.function("tf_ink2_loop_gate_down_rows"),
            .gate_lin_rows = try mod.function("tf_ink2_loop_gate_lin_rows"),
            .tiled = tiledOn(),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// TF_INKLING2_LOOP_TILED=0: the original per-row gate kernels (the tiled ones are bitwise the same).
fn tiledOn() bool {
    const v = std.c.getenv("TF_INKLING2_LOOP_TILED") orelse return true;
    return v[0] != '0';
}

const down_rows = 8; // loop.cu kDR
const down_ranks = 8 * 8; // kWarps * kDK
const lin_rows = 64; // loop.cu kLR
/// Rows from which gate_down's norms are formed once per row (a prep pass) and its weights reused over row tiles.
const tiled_down_min_rows = 16;

/// The shapes the kernels hold: blocks, queries, gates and rank in their arrays; a refusal names the bound.
pub fn mixFits(a: MixArgs) error{Inkling2LoopShape}!void {
    if (a.rows <= 0 or a.h <= 0 or a.nq <= 0 or a.nq > max_queries or a.n_blocks < 0 or a.n_blocks > max_blocks)
        return error.Inkling2LoopShape;
}

pub fn gateFits(a: GateArgs, count: usize) error{Inkling2LoopShape}!void {
    if (a.rows <= 0 or a.h_dim <= 0 or a.rank <= 0 or a.rank > max_rank or count == 0 or count > max_gates or a.ys == 0)
        return error.Inkling2LoopShape;
}

/// fp32 mixes of `nq` queries over a row's blocks and partial (no norm): a block a row.
pub fn mix(k: *const Kernels, s: cuda.Stream, a: MixArgs) !void {
    try mixFits(a);
    var args: cuda.Args = .{};
    args.add(a);
    try cuda.launch.launch(k.mix, .{ .grid = .{ .x = @intCast(a.rows) }, .block = .{ .x = threads } }, s, &args);
}

/// `layers` mixes at once, layer l's queries `query_step` bytes and outputs `out_step` floats past layer 0's.
pub fn mixLayers(k: *const Kernels, s: cuda.Stream, a: MixArgs, layers: usize, query_step: i64, out_step: i64) !void {
    try mixFits(a);
    var args: cuda.Args = .{};
    args.add(a);
    args.add(query_step);
    args.add(out_step);
    try cuda.launch.launch(k.mix_layers, .{ .grid = .{ .x = @intCast(a.rows), .y = @intCast(layers) }, .block = .{ .x = threads } }, s, &args);
}

/// The gates' rank-r down rows, then the gated, normed inputs: a block a (row, gate) each.
pub fn gates(k: *const Kernels, s: cuda.Stream, a: GateArgs, n: usize) !void {
    if (k.tiled and check.on()) return check.run(k, s, a, n);
    return gatesWith(k, k.tiled, s, a, n);
}

/// TF_INKLING2_LOOP_CHECK=1 (ungraphed only): each gates call runs the original kernels, then the tiled ones, and compares
/// down and out bytewise; totals print every 500 calls.
const check = struct {
    var state: enum(u8) { unknown, off, enabled } = .unknown;
    var calls: u64 = 0;
    var bad: u64 = 0;
    var bad_down: u64 = 0;
    var rows_max: i32 = 0;
    fn on() bool {
        if (state == .unknown) {
            state = .off;
            if (std.c.getenv("TF_INKLING2_LOOP_CHECK")) |v| if (v[0] == '1') {
                state = .enabled;
            };
        }
        return state == .enabled;
    }
    fn grab(s: cuda.Stream, gpa: std.mem.Allocator, a: GateArgs, n: usize) ![]u8 {
        const rows: usize = @intCast(a.rows);
        const h: usize = @intCast(a.h_dim);
        const rank: usize = @intCast(a.rank);
        const plane = rows * h * 2;
        const down = n * rows * rank * 2;
        var buf = try gpa.alloc(u8, n * plane + down);
        try s.synchronize();
        for (0..n) |g| try s.d.check(s.d.api.cuMemcpyDtoH_v2(buf[g * plane ..].ptr, a.out + g * @as(u64, @intCast(a.out_stride)) * 2, plane), "dtoh");
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(buf[n * plane ..].ptr, a.down, down), "dtoh");
        return buf;
    }
    fn run(k: *const Kernels, s: cuda.Stream, a: GateArgs, n: usize) !void {
        const gpa = std.heap.page_allocator;
        try gatesWith(k, false, s, a, n);
        const want = try grab(s, gpa, a, n);
        defer gpa.free(want);
        try gatesWith(k, true, s, a, n);
        const got = try grab(s, gpa, a, n);
        defer gpa.free(got);
        const split = want.len - n * @as(usize, @intCast(a.rows)) * @as(usize, @intCast(a.rank)) * 2;
        calls += 1;
        rows_max = @max(rows_max, a.rows);
        if (!std.mem.eql(u8, want[0..split], got[0..split])) bad += 1;
        if (!std.mem.eql(u8, want[split..], got[split..])) bad_down += 1;
        if (calls % 500 == 0) std.debug.print("[loopcheck] {d} calls (rows up to {d}): {d} out mismatches, {d} down mismatches\n", .{ calls, rows_max, bad, bad_down });
    }
};

fn gatesWith(k: *const Kernels, tiled: bool, s: cuda.Stream, a: GateArgs, n: usize) !void {
    try gateFits(a, n);
    var args: cuda.Args = .{};
    args.add(a);
    const grid: cuda.Dim3 = .{ .x = @intCast(a.rows), .y = @intCast(n) };
    const h: u32 = @intCast(a.h_dim);
    // spread over the GPU: rank rows 8 a block, outputs 128 a block; the norm a block a row and gate
    const rows: u32 = @intCast(a.rows);
    const rank: u32 = @intCast(a.rank);
    if (tiled and rows >= tiled_down_min_rows and @as(u32, @intCast(2 * a.h_dim)) % 256 == 0) {
        // xs (bf16 [gates][rows][2H]) borrows ys, which gate_lin writes only after gate_down has read it
        var pa: cuda.Args = .{};
        pa.add(a);
        pa.add(a.ys);
        try cuda.launch.launch(k.gate_prep, .{ .grid = grid, .block = .{ .x = threads } }, s, &pa);
        const dg: cuda.Dim3 = .{ .x = std.math.divCeil(u32, rows, down_rows) catch unreachable, .y = grid.y, .z = std.math.divCeil(u32, rank, down_ranks) catch unreachable };
        try cuda.launch.launch(k.gate_down_rows, .{ .grid = dg, .block = .{ .x = threads } }, s, &pa);
    } else {
        const down: cuda.Dim3 = .{ .x = grid.x, .y = grid.y, .z = @intCast(std.math.divCeil(i32, a.rank, threads / 32) catch unreachable) };
        try cuda.launch.launch(k.gate_down, .{ .grid = down, .block = .{ .x = threads }, .shared = 2 * h * 4 }, s, &args);
    }
    if (tiled and rank == 64 and @as(u32, @intCast(a.h_dim)) % 8 == 0) {
        const lin: cuda.Dim3 = .{ .x = std.math.divCeil(u32, rows, lin_rows) catch unreachable, .y = grid.y, .z = std.math.divCeil(u32, h, lin_threads) catch unreachable };
        try cuda.launch.launch(k.gate_lin_rows, .{ .grid = lin, .block = .{ .x = lin_threads } }, s, &args);
    } else {
        const lin: cuda.Dim3 = .{ .x = grid.x, .y = grid.y, .z = std.math.divCeil(u32, h, lin_threads) catch unreachable };
        try cuda.launch.launch(k.gate_lin, .{ .grid = lin, .block = .{ .x = lin_threads } }, s, &args);
    }
    try cuda.launch.launch(k.gate_mix, .{ .grid = grid, .block = .{ .x = threads } }, s, &args);
}

test "loop: the argument structs keep loop.cu's layouts" {
    try std.testing.expectEqual(@as(usize, 8 * 8 + 8 + 4 * 8 + 8 + 8 + 5 * 4 + 2 * 4 + 4), @sizeOf(MixArgs));
    try std.testing.expectEqual(@as(usize, 3 * 8 + 8 + 8 + 3 * 4 + 4 + 4 * 3 * 8 + 3 * 8 + 3 * 4 + 2 * 4 + 4 + 8), @sizeOf(GateArgs));
}

test "loop: shapes past the kernels' arrays are refused by name" {
    const ok: MixArgs = .{ .partial = 1, .out = 1, .out_stride = 0, .n_blocks = 1, .partial_f32 = 0, .nq = 4, .rows = 1, .h = 2304, .eps = 1e-6, .scale = 1 };
    try mixFits(ok);
    var bad = ok;
    bad.nq = 5;
    try std.testing.expectError(error.Inkling2LoopShape, mixFits(bad));
    bad = ok;
    bad.n_blocks = 9;
    try std.testing.expectError(error.Inkling2LoopShape, mixFits(bad));
    const g: GateArgs = .{ .ys = 1, .msg = 1, .msg_stride = 0, .down = 1, .out = 1, .out_stride = 0, .rows = 1, .h_dim = 2304, .rank = 64, .eps_unit = 1e-6, .eps = 1e-6 };
    try gateFits(g, 3);
    try std.testing.expectError(error.Inkling2LoopShape, gateFits(g, 4));
    var wide = g;
    wide.rank = 129;
    try std.testing.expectError(error.Inkling2LoopShape, gateFits(wide, 1));
}

//! Inkling 2 Mini's prompt-chunk kernels on XPU: the row-batched glue of kernels/xpu/inkling2/prefill.cl and its launches.
//! Buffers are Level Zero device addresses with offsets applied; every launch queues on the stream.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden: u32 = 2304;
pub const message_entries: u32 = 48;
/// A device address; offsets are already applied.
pub const Ref = ?*anyopaque;

pub const Kernels = struct {
    ar: xpu.Kernel,
    messages: xpu.Kernel,
    copy_rows: xpu.Kernel,
    perm: xpu.Kernel,
    q8: xpu.Kernel,
    zeta_norm: xpu.Kernel,
    zeta: xpu.Kernel,
    gates: xpu.Kernel,
    kv_copy: xpu.Kernel,
    attn_window: xpu.Kernel,
    dense: xpu.Kernel,
    router: xpu.Kernel,
    conv_static: xpu.Kernel,
    rel_table: xpu.Kernel,

    pub fn load(module: xpu.Module) !Kernels {
        var k_ar = try module.kernel("tf_ink2_pf_ar", .{ 256, 1, 1 });
        errdefer k_ar.deinit();
        var k_messages = try module.kernel("tf_ink2_pf_messages", .{ 256, 1, 1 });
        errdefer k_messages.deinit();
        var k_copy = try module.kernel("tf_ink2_pf_copy_rows", .{ 256, 1, 1 });
        errdefer k_copy.deinit();
        var k_perm = try module.kernel("tf_ink2_pf_perm", .{ 256, 1, 1 });
        errdefer k_perm.deinit();
        var k_q8 = try module.kernel("tf_ink2_pf_q8", .{ 128, 1, 1 });
        errdefer k_q8.deinit();
        var k_zn = try module.kernel("tf_ink2_pf_zeta_norm", .{ 256, 1, 1 });
        errdefer k_zn.deinit();
        var k_zeta = try module.kernel("tf_ink2_pf_zeta", .{ 64, 1, 1 });
        errdefer k_zeta.deinit();
        var k_gates = try module.kernel("tf_ink2_pf_gates", .{ 16, 1, 1 });
        errdefer k_gates.deinit();
        var k_kv = try module.kernel("tf_ink2_pf_kv_copy", .{ 16, 1, 1 });
        errdefer k_kv.deinit();
        var k_win = try module.kernel("tf_ink2_pf_attn_window", .{ 16, 1, 1 });
        errdefer k_win.deinit();
        var k_dense = try module.kernel("tf_ink2_pf_dense", .{ 128, 1, 1 });
        errdefer k_dense.deinit();
        var k_router = try module.kernel("tf_ink2_pf_router", .{ 256, 1, 1 });
        errdefer k_router.deinit();
        var k_conv = try module.kernel("tf_ink2_pf_conv_static", .{ 256, 1, 1 });
        errdefer k_conv.deinit();
        const k_rel = try module.kernel("tf_ink2_pf_rel_table", .{ 256, 1, 1 });
        return .{ .ar = k_ar, .messages = k_messages, .copy_rows = k_copy, .perm = k_perm, .q8 = k_q8, .zeta_norm = k_zn, .zeta = k_zeta, .gates = k_gates, .kv_copy = k_kv, .attn_window = k_win, .dense = k_dense, .router = k_router, .conv_static = k_conv, .rel_table = k_rel };
    }

    pub fn deinit(k: *Kernels) void {
        k.rel_table.deinit();
        k.conv_static.deinit();
        k.router.deinit();
        k.dense.deinit();
        k.attn_window.deinit();
        k.kv_copy.deinit();
        k.gates.deinit();
        k.zeta.deinit();
        k.zeta_norm.deinit();
        k.q8.deinit();
        k.perm.deinit();
        k.copy_rows.deinit();
        k.messages.deinit();
        k.ar.deinit();
        k.* = undefined;
    }
};

/// The attention residual of `rows` rows for `queries` query rows: row r takes its sources from `blocks` (4 block slots a
/// row) and `partial` (the last source) and writes `out[query][row][H]` as FP32.
pub fn ar(k: *const Kernels, s: xpu.Stream, blocks: Ref, partial: Ref, queries: Ref, out: Ref, sources: u32, count: u32, rows: u32) !void {
    std.debug.assert(sources > 0 and sources <= 5 and count > 0 and count <= 3 and rows > 0);
    try xpu.launch.dispatch(k.ar, s, .{ count, rows, 1 }, .{ blocks, partial, queries, out, sources, count, rows });
}

/// The cross-pass messages of `rows` rows, laid out [entry][row][H] FP32.
pub fn messages(k: *const Kernels, s: xpu.Stream, bank0: Ref, bank1: Ref, q_mlp: Ref, kv: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(k.messages, s, .{ rows * message_entries, 1, 1 }, .{ bank0, bank1, q_mlp, kv, out, rows });
}

/// y[r][0..n] = x[r * ldx ..] for bf16 rows, `ldx` and `ldy` in elements.
pub fn copyRows(k: *const Kernels, s: xpu.Stream, x: Ref, y: Ref, rows: u32, n: u32, ldx: u32, ldy: u32) !void {
    std.debug.assert(rows > 0 and n > 0);
    try xpu.launch.dispatch(k.copy_rows, s, .{ (n + 255) / 256, rows, 1 }, .{ x, y, rows, n, ldx, ldy });
}

/// Weight rows a work-group covers in `q8` (8 sub-groups of PF_WA weight tiles of 8 rows) and rows of x a group covers (PF_TB tiles of 16).
const q8_cols: u32 = 128;
const q8_rows: u32 = 32;

/// out[r][0..n] = bf16(x[r][0..k] . dequant(weight)[n][k]) for the Q8 matrix `weight` (q8.zig's layout), `xp` a scratch of
/// `rows * k` bf16 values for the reordered x. `n` a multiple of 8, `k` a multiple of 256.
pub fn denseQ8(k_: *const Kernels, s: xpu.Stream, weight: Ref, x: Ref, xp: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    std.debug.assert(rows > 0 and n % 8 == 0 and k % 256 == 0);
    try xpu.launch.dispatch(k_.perm, s, .{ k / 256, rows, 1 }, .{ x, xp, rows, k });
    try xpu.launch.dispatch(k_.q8, s, .{ (n + q8_cols - 1) / q8_cols, (rows + q8_rows - 1) / q8_rows, 1 }, .{ xp, weight, out, rows, n, k, n });
}

/// out[r][0..n] = bf16(x[r][0..k] . weight[n][k]) for BF16 weights, bit for bit the rows of dense.zig's 32-row launches, all rows in one launch.
pub fn denseBf16(k_: *const Kernels, s: xpu.Stream, weight: Ref, x: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    std.debug.assert(rows > 0 and n > 0 and k % 8 == 0);
    try xpu.launch.dispatch(k_.dense, s, .{ (n + 7) / 8, (rows + 31) / 32, 1 }, .{ x, weight, out, rows, n, k, n });
}

/// The router of glue.cl for `rows` rows, 4 a work-group; ids and weights are the same bit for bit.
pub fn router(k: *const Kernels, s: xpu.Stream, x: Ref, weight: Ref, bias: Ref, ids: Ref, weights: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(k.router, s, .{ (rows + 3) / 4, 1, 1 }, .{ x, weight, bias, ids, weights, rows });
}

/// glue.zig's static convolution (same arguments), four channels a work-item; `channels` a multiple of 4.
pub fn convStatic(k: *const Kernels, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, weight: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0 and channels % 4 == 0);
    try xpu.launch.dispatch(k.conv_static, s, .{ (channels / 4 + 255) / 256, rows, 1 }, .{ history, current, taps, weight, out, rows, channels });
}

/// attention.zig's relative-bias table for `rows * heads` (row, head) pairs; bit for bit the same.
pub fn relTable(k: *const Kernels, s: xpu.Stream, pairs: u32, rel: Ref, proj: Ref, tab: Ref) !void {
    std.debug.assert(pairs % 8 == 0); // PF_REL_BATCH
    try xpu.launch.dispatch(k.rel_table, s, .{ pairs / 8, 1, 1 }, .{ rel, proj, tab });
}

/// The controller tuple for every row: zn is a scratch of `rows * 2 * hidden` bf16 values, zeta one of `rows * 64`.
/// out = (1 - tanh F) h + tanh Gamma m, h and m [row][hidden] binary32.
pub fn controller(k: *const Kernels, s: xpu.Stream, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, zn: Ref, zeta: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(k.zeta_norm, s, .{ rows, 1, 1 }, .{ h, m, zn, rows });
    try xpu.launch.dispatch(k.zeta, s, .{ (rows + 31) / 32, 1, 1 }, .{ zn, down, zeta, rows });
    try xpu.launch.dispatch(k.gates, s, .{ hidden / 16, (rows + 15) / 16, 1 }, .{ h, m, zeta, forget, gain, out, rows });
}

/// Copies `n` positions from `from` of every K/V head of one ring to the same positions of another ring.
pub fn kvCopy(k: *const Kernels, s: xpu.Stream, src: Ref, dst: Ref, from: u32, src_ring: u32, dst_ring: u32, n: u32, kv_heads: u32) !void {
    if (n == 0) return;
    try xpu.launch.dispatch(k.kv_copy, s, .{ n, kv_heads, 1 }, .{ src, dst, from, src_ring, dst_ring, n, kv_heads });
}

/// Sliding-window attention of one slice of `rows` rows (3 query heads a K/V head) over a staging ring of `ring` positions.
/// `seg` holds the slice's `attention_plan.Seg` (row 0, `rows`, first position, offset 0). One partial a (row, head) goes to
/// `part_o` and `part_ml`; `tab` is the slice's relative-bias table.
pub fn attnWindow(k: *const Kernels, s: xpu.Stream, q: Ref, kc: Ref, vc: Ref, tab: Ref, seg: Ref, part_o: Ref, part_ml: Ref, rows: u32, heads: u32, kv_heads: u32, ring: u32, scale: f32) !void {
    std.debug.assert(rows > 0 and heads == kv_heads * 3);
    try xpu.launch.dispatch(k.attn_window, s, .{ (rows * 3 + 15) / 16, kv_heads, 1 }, .{ q, kc, vc, tab, seg, part_o, part_ml, heads, kv_heads, ring, scale });
}

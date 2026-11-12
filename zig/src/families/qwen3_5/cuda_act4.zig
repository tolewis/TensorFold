//! A dense MLP's activation and its down projection's NVFP4 rows in one kernel (qwen35_rows.cuh, tf_q35_mlp_act_q4):
//! tf_q35_mlp_act's bf16 values, quantized in registers as tf_ck_quant4_rows does, so the bf16 rows are not written.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const modelopt = @import("cuda_modelopt.zig");

const Engine = engine.Engine;

/// The first call's rows that both ways compute (TF_ACT4_CHECK=0: no check).
const check_rows = 256;

var checked = false;
var differ = false; // the check found a byte the two kernels write otherwise: they run from then on

/// SiLU(gate) x up times the down projection `w`, into `out`, where `w` takes the checkpoint's NVFP4 prompt GEMM
/// (qlinear's Linear.prompt, checkpoint4). False: nothing ran, the caller runs tf_q35_mlp_act and the projection.
pub fn down(e: *Engine, w: engine.Proj, rows: usize, out: u64) !bool {
    if (differ or w != .ql) return false;
    const q = switch (w.ql) {
        .nvfp4 => |q| q,
        else => return false,
    };
    const k = if (e.mo) |*m| m else return false;
    const l = k.linear();
    if (!l.ck4 or q.act == 0 or l.ck == null or rows < cuda.ck.prompt_tokens or l.ck_rows == 0) return false;
    if (q.k != e.g.intermediate) return false;
    const c = l.ck.?;
    if (!checked and modelopt.envOn("TF_ACT4_CHECK")) {
        checked = true;
        if (!try same(e, c, q, @min(rows, check_rows), l.ck_rows)) {
            differ = true;
            return false;
        }
    }
    try rowsOf(e, q.k, q.act, rows, l.ck_rows);
    try c.prompt4(e.stream, l.ck_rows, l.ck_rows + rows * q.k / 2, q, rows, out);
    return true;
}

/// e2m1 codes [rows][k / 2] at `codes`, e4m3 block scales [rows][k / 16] after them, from e.gate and e.up.
fn rowsOf(e: *Engine, k: usize, act: f32, rows: usize, codes: u64) !void {
    const blocks = rows * (k / 16);
    const args = .{ e.gate, e.up, @as(c_int, @intCast(rows)), @as(c_int, @intCast(k)), cuda.ck.inverse(act), codes, codes + rows * k / 2 };
    try e.run(e.fns.mlp_act_q4, .{ (blocks + 127) / 128, 1, 1 }, .{ 128, 1, 1 }, args);
}

/// Both ways on the first `rows`: the two kernels' codes and scales into a scratch, the one kernel's into `codes`.
fn same(e: *Engine, c: *const cuda.ck.Ck, q: anytype, rows: usize, codes: u64) !bool {
    const bytes = rows * q.k / 2 + rows * q.k / 16;
    var two = try cuda.DeviceBuffer.alloc(e.d, bytes);
    defer two.free();
    const n: c_longlong = @intCast(rows * q.k);
    const act_args = .{ e.gate, e.up, e.act, n, @as(u64, 0), @as(c_int, @intCast(q.k)), @as(u64, 0) };
    try e.run(e.fns.mlp_act, .{ (rows * q.k + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, act_args);
    try c.quant4Rows(e.stream, e.act, q.k, rows, q.k, q.act, two.ptr);
    try rowsOf(e, q.k, q.act, rows, codes);
    try e.stream.synchronize();
    const a = try e.gpa.alloc(u8, bytes);
    defer e.gpa.free(a);
    const b = try e.gpa.alloc(u8, bytes);
    defer e.gpa.free(b);
    try two.download(0, a);
    const one: cuda.DeviceBuffer = .{ .d = e.d, .ptr = codes, .len = bytes, .borrowed = true };
    try one.download(0, b);
    var bad: usize = 0;
    for (a, b) |x, y| bad += @intFromBool(x != y);
    const then = if (bad == 0) "" else ": the two kernels run from now on";
    std.log.info("qwen3_5: MLP act and NVFP4 rows in one kernel, rows={d} k={d}: {d} of {d} bytes differ from the two kernels{s}", .{ rows, q.k, bad, bytes, then });
    return bad == 0;
}

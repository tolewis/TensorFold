//! The prompt int8 (W4A8) path's arithmetic on the host, for the error a GPU window would show: the same rotated
//! activation plane as exl3_rotate_in_h, the fp16 GEMM of exl3_gemm against the int8 GEMM of exl3_int8.cl, then the
//! rotate-out of exl3_rotate_out_rows. The trellis decode is exact (it yields the stored fp16 values), so only the
//! two int8 quantizations are modeled here. Run: `zig test zig/src/xpu/exl3_int8_ref.zig` prints the figures.
const std = @import("std");

const BITS = 4;
const SG = 16;
const INV_SQRT128: f32 = 0.08838834764831845;

/// The Sylvester Hadamard of a 128 block, as exl3.cl's hadamard128 (unitary, / sqrt(128)).
fn hadamard128(v: *[128]f32) void {
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var i: usize = 0;
        while (i < 128) : (i += 1) {
            if (i & d != 0) continue;
            const a = v[i];
            const b = v[i + d];
            v[i] = a + b;
            v[i + d] = a - b;
        }
    }
}

/// The rotated activation plane: rotate every 128 block of x, then the fp16 rounding exl3_rotate_in_h stores.
fn rotateIn(x: []const f32, k: usize, out: []f16) void {
    var blk: usize = 0;
    while (blk < k) : (blk += 128) {
        var v: [128]f32 = undefined;
        for (0..128) |i| v[i] = x[blk + i];
        hadamard128(&v);
        for (0..128) |i| out[blk + i] = @floatCast(v[i] * INV_SQRT128);
    }
}

/// int8 per 32-dimension group with an fp32 scale, as exl3_int8_quant and exl3_int8_gemm do: q = rint(v / s),
/// s = max |v| / 127, clamped to the 254 symmetric levels.
fn quantize(src: []const f32, dst: []i8, scale: []f32) void {
    var g: usize = 0;
    while (g < src.len) : (g += 32) {
        var m: f32 = 0;
        for (src[g..][0..32]) |v| m = @max(m, @abs(v));
        const s = if (m > 0) m / 127.0 else 1.0;
        const inv = 1.0 / s;
        for (src[g..][0..32], dst[g..][0..32]) |v, *q| q.* = @intFromFloat(@max(-127.0, @min(127.0, @round(v * inv))));
        scale[g / 32] = s;
    }
}

/// out[row][col] = H(sums[row][blk]) / sqrt(128) * svh[col], one 128 block of the rotate-out.
fn rotateOut(sums: []const f32, svh: []const f16, n: usize, out: []f32) void {
    var blk: usize = 0;
    while (blk < n) : (blk += 128) {
        var v: [128]f32 = undefined;
        for (0..128) |i| v[i] = sums[blk + i];
        hadamard128(&v);
        for (0..128) |i| out[blk + i] = v[i] * INV_SQRT128 * @as(f32, svh[blk + i]);
    }
}

/// The error figures: the absolute error against the reference's rms, the relative L2 norm, and the per-element
/// relative error over the outputs whose magnitude is at least a tenth of the reference's rms (a near-zero output
/// from cancellation has no useful relative figure).
const Stats = struct { max_abs: f32, p99_abs: f32, rel_l2: f32, p50: f32, p90: f32, p99: f32, max_rel: f32 };

fn stats(err: []const f32, rel: []const f32, keep: []const bool, rms_y: f32, gpa: std.mem.Allocator) !Stats {
    const e = try gpa.dupe(f32, err);
    defer gpa.free(e);
    std.mem.sort(f32, e, {}, std.sort.asc(f32));
    var kept: usize = 0;
    var sq_err: f64 = 0;
    for (err) |v| sq_err += @as(f64, v) * v;
    for (keep) |k| kept += @intFromBool(k);
    const r = try gpa.alloc(f32, kept);
    defer gpa.free(r);
    var at: usize = 0;
    for (rel, keep) |v, k| {
        if (!k) continue;
        r[at] = v;
        at += 1;
    }
    std.mem.sort(f32, r, {}, std.sort.asc(f32));
    return .{
        .max_abs = e[e.len - 1],
        .p99_abs = e[e.len * 99 / 100] / rms_y,
        .rel_l2 = @floatCast(@sqrt(sq_err / @as(f64, @floatFromInt(err.len))) / @as(f64, rms_y)),
        .p50 = r[r.len / 2],
        .p90 = r[r.len * 9 / 10],
        .p99 = r[r.len * 99 / 100],
        .max_rel = r[r.len - 1],
    };
}

test "the int8 prompt GEMM's error against the fp16 kernel's output (one 27B shape)" {
    const gpa = std.testing.allocator;
    const rows = 8; // a slice of a 1,152-row chunk; the error does not depend on the row count
    const k = 5120; // the 27B's D (gate|up, down, the mixers)
    const n = 1024; // a slice of gate|up's 8,704 columns a rank
    var prng = std.Random.DefaultPrng.init(0x19b7);
    const rnd = prng.random();

    const x = try gpa.alloc(f32, rows * k);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const w = try gpa.alloc(f32, k * n);
    defer gpa.free(w);
    for (w) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const svh = try gpa.alloc(f16, n);
    defer gpa.free(svh);
    for (svh) |*v| v.* = @floatCast(0.5 + rnd.float(f32));

    const a16 = try gpa.alloc(f16, rows * k);
    defer gpa.free(a16);
    for (0..rows) |r| rotateIn(x[r * k ..][0..k], k, a16[r * k ..][0..k]);
    const a8 = try gpa.alloc(i8, rows * k);
    defer gpa.free(a8);
    const sa = try gpa.alloc(f32, rows * (k / 32));
    defer gpa.free(sa);
    for (0..rows) |r| {
        const af = try gpa.alloc(f32, k);
        defer gpa.free(af);
        for (0..k) |i| af[i] = @floatCast(a16[r * k + i]);
        quantize(af, a8[r * k ..][0..k], sa[r * (k / 32) ..][0 .. k / 32]);
    }
    const w8 = try gpa.alloc(i8, k * n);
    defer gpa.free(w8);
    const sw = try gpa.alloc(f32, (k / 32) * n);
    defer gpa.free(sw);
    {
        const col = try gpa.alloc(f32, k);
        defer gpa.free(col);
        const qcol = try gpa.alloc(i8, k);
        defer gpa.free(qcol);
        for (0..n) |c| {
            for (0..k) |i| col[i] = w[i * n + c];
            quantize(col, qcol, sw[c * (k / 32) ..][0 .. k / 32]);
            for (0..k) |i| w8[i * n + c] = qcol[i];
        }
    }

    const sums16 = try gpa.alloc(f32, rows * n);
    defer gpa.free(sums16);
    const sums8 = try gpa.alloc(f32, rows * n);
    defer gpa.free(sums8);
    @memset(sums16, 0);
    @memset(sums8, 0);
    for (0..rows) |r| for (0..n) |c| {
        var acc16: f32 = 0;
        var acc8: f32 = 0;
        var g: usize = 0;
        while (g < k) : (g += 32) {
            var d16: f32 = 0;
            var d8: i32 = 0;
            for (0..32) |i| {
                d16 += @as(f32, a16[r * k + g + i]) * w[(g + i) * n + c];
                d8 += @as(i32, a8[r * k + g + i]) * @as(i32, w8[(g + i) * n + c]);
            }
            acc16 += d16;
            acc8 += @as(f32, @floatFromInt(d8)) * sa[r * (k / 32) + g / 32] * sw[c * (k / 32) + g / 32];
        }
        sums16[r * n + c] = acc16;
        sums8[r * n + c] = acc8;
    };

    const y16 = try gpa.alloc(f32, rows * n);
    defer gpa.free(y16);
    const y8 = try gpa.alloc(f32, rows * n);
    defer gpa.free(y8);
    for (0..rows) |r| {
        rotateOut(sums16[r * n ..][0..n], svh, n, y16[r * n ..][0..n]);
        rotateOut(sums8[r * n ..][0..n], svh, n, y8[r * n ..][0..n]);
    }
    const err = try gpa.alloc(f32, rows * n);
    defer gpa.free(err);
    const rel = try gpa.alloc(f32, rows * n);
    defer gpa.free(rel);
    const keep = try gpa.alloc(bool, rows * n);
    defer gpa.free(keep);
    var sq_y: f64 = 0;
    for (y16) |a| sq_y += @as(f64, a) * a;
    const rms_y: f32 = @floatCast(@sqrt(sq_y / @as(f64, @floatFromInt(rows * n))));
    for (y16, y8, err, rel, keep) |a, b, *e, *rr, *ok| {
        e.* = @abs(b - a);
        rr.* = e.* / @abs(a);
        ok.* = @abs(a) >= 0.1 * rms_y;
    }
    const st = try stats(err, rel, keep, rms_y, gpa);
    std.debug.print("exl3 int8 prompt GEMM, {d} rows x {d} cols x {d} k: |y_fp16| rms {d:.3}, max abs error {e:.3} ({d:.2} % of the rms), abs p99 {d:.2} % of the rms, rel L2 {e:.3}, rel p50 {e:.3} p90 {e:.3} p99 {e:.3} max {e:.3}\n", .{ rows, n, k, rms_y, st.max_abs, 100.0 * st.max_abs / rms_y, 100.0 * st.p99_abs, st.rel_l2, st.p50, st.p90, st.p99, st.max_rel });
    try std.testing.expect(st.rel_l2 < 0.02);
    try std.testing.expect(st.max_abs < 0.1 * rms_y);
}

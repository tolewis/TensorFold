//! NVFP4 blocks for Inkling 2 Mini's dense projections on XPU: e2m1 codes, one e4m3 scale for every 16 values of a
//! row, and one fp32 factor for the matrix. The same layout and the same decode as the expert kernels
//! (kernels/xpu/inkling2/experts.cl): low nibble is the even column, a decoded weight is bf16((e2m1 * e4m3) * factor),
//! and the factor the kernel uses is bf16(amax) / bf16(denominator). Here denominator is 1 and amax is the factor.
//!
//! One buffer: codes (n * k / 2 bytes), scales (n * k / 16 bytes), then the factor and the denominator (two f32).
//! k is a multiple of 128, so the expert down kernel's 64-byte path applies. Pure host code.
const std = @import("std");

pub const group: usize = 16;
pub const e2_max: f32 = 6;
pub const e4_max: f32 = 448;

/// The aux buffer the launches share: 256 zero expert ids, 32 identity members, then a plan for every row count 1 to 32.
pub const aux_ids: usize = 256;
pub const aux_members: usize = 32;
pub const aux_plans: usize = 32;

pub fn bytes(n: usize, k: usize) usize {
    return n * (k / 2) + n * (k / group) + 8;
}
pub fn scaleOffset(n: usize, k: usize) usize {
    return n * (k / 2);
}
pub fn factorOffset(n: usize, k: usize) usize {
    return scaleOffset(n, k) + n * (k / group);
}
pub fn admits(k: usize) bool {
    return k != 0 and k % (group * 8) == 0;
}

pub fn auxBytes() usize {
    return (aux_ids + aux_members + aux_plans * 4) * 4;
}

fn bf16bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}
fn round32(v: f32) f32 {
    return widen(bf16bits(v));
}
fn bf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// The factor the kernel applies: bf16(amax) / bf16(1). `amax` is the stored factor.
pub fn kernelPhi(factor: f32) f32 {
    return round32(factor);
}

pub fn factorOf(amax: f32) f32 {
    if (amax == 0) return 0;
    return amax / (e2_max * e4_max);
}

fn e4m3(b: u8) f32 {
    if (b & 0x7f == 0x7f) return std.math.nan(f32);
    const e: i32 = (b >> 3) & 0xF;
    const m: f32 = @floatFromInt(b & 7);
    const mag = if (e == 0) m / 8.0 * 0x1p-6 else (1 + m / 8.0) * std.math.pow(f32, 2, @as(f32, @floatFromInt(e - 7)));
    return if (b & 0x80 != 0) -mag else mag;
}

fn e2m1(n: u8) f32 {
    const mag = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = mag[n & 7];
    return if (n & 8 != 0) -v else v;
}

/// Nearest e4m3 code, ties to even, saturating at 448 (never the NaN codes).
fn e4Bits(x: f32) u8 {
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a = @abs(x);
    if (a >= e4_max) return sign | 0x7e;
    var best: u8 = 0;
    var best_d = a;
    for (0..127) |i| {
        const b: u8 = @intCast(i);
        const d = @abs(a - e4m3(b));
        const take = d < best_d or (d == best_d and (b & 1) == 0 and (best & 1) != 0);
        if (take) {
            best = b;
            best_d = d;
        }
    }
    return sign | best;
}

/// Nearest e2m1 nibble, ties to even mantissa, saturating at 6.
fn e2Bits(x: f32) u8 {
    const sign: u8 = if (std.math.signbit(x)) 8 else 0;
    const a = @abs(x);
    const mag = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    var best: u8 = 0;
    var best_d = a;
    for (mag, 0..) |m, i| {
        const d = @abs(a - m);
        const b: u8 = @intCast(i);
        const take = d < best_d or (d == best_d and (b & 1) == 0 and (best & 1) != 0);
        if (take) {
            best = b;
            best_d = d;
        }
    }
    if (best == 0) return 0;
    return sign | best;
}

/// The largest magnitude in `src` (bf16 bits).
pub fn amaxOf(src: []const u16) f32 {
    var a: f32 = 0;
    for (src) |h| a = @max(a, @abs(bf16(h)));
    return a;
}

/// Encodes `rows` rows of `k` bf16 values. `phi` is `kernelPhi` of the matrix factor.
pub fn encodeRows(src: []const u16, rows: usize, k: usize, phi: f32, codes: []u8, scales: []u8) void {
    std.debug.assert(admits(k) and src.len == rows * k and codes.len == rows * (k / 2) and scales.len == rows * (k / group));
    const nb = k / group;
    for (0..rows) |j| {
        const row = src[j * k ..][0..k];
        const out = codes[j * (k / 2) ..][0 .. k / 2];
        for (0..nb) |b| {
            const v = row[b * group ..][0..group];
            var gmax: f32 = 0;
            for (v) |h| gmax = @max(gmax, @abs(bf16(h)));
            var scale: u8 = 0;
            var dec: f32 = 0;
            if (phi != 0 and gmax != 0) {
                scale = e4Bits(gmax / (e2_max * phi));
                dec = e4m3(scale) * phi;
            }
            scales[j * nb + b] = scale;
            for (0..group / 2) |i| {
                const c0: u8 = if (dec == 0) 0 else e2Bits(bf16(v[2 * i]) / dec);
                const c1: u8 = if (dec == 0) 0 else e2Bits(bf16(v[2 * i + 1]) / dec);
                out[b * (group / 2) + i] = c0 | (c1 << 4);
            }
        }
    }
}

/// A decoded weight as the expert kernel stores it: bf16((e2m1 * e4m3) * phi).
pub fn decode(codes: []const u8, scales: []const u8, k: usize, phi: f32, j: usize, col: usize) f32 {
    const byte = codes[j * (k / 2) + col / 2];
    const nib: u8 = if (col & 1 == 0) byte & 0xF else byte >> 4;
    const scale = scales[j * (k / group) + col / group];
    return round32(e2m1(nib) * e4m3(scale) * phi);
}

/// Zeros, identity members, and one prompt plan for each row count 1 to 32.
pub fn fillAux(buf: []u8) void {
    std.debug.assert(buf.len == auxBytes());
    @memset(buf, 0);
    const words: []i32 = @alignCast(std.mem.bytesAsSlice(i32, buf));
    for (0..aux_members) |i| words[aux_ids + i] = @intCast(i);
    const base = aux_ids + aux_members;
    for (1..aux_plans + 1) |r| {
        const at = base + (r - 1) * 4;
        words[at] = 0;
        words[at + 1] = 0;
        words[at + 2] = @intCast(r);
        words[at + 3] = 1;
    }
}

test "a block round-trips within half an e2m1 step of its scale" {
    var src: [128]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    var amax: f32 = 0;
    for (&src) |*h| {
        const v = (prng.random().float(f32) - 0.5) * 0.2;
        amax = @max(amax, @abs(v));
        h.* = bf16bits(v);
    }
    const phi = kernelPhi(factorOf(amaxOf(&src)));
    var codes: [64]u8 = undefined;
    var scales: [8]u8 = undefined;
    encodeRows(&src, 1, 128, phi, &codes, &scales);
    for (0..128) |c| {
        const got = decode(&codes, &scales, 128, phi, 0, c);
        const want = bf16(src[c]);
        const scale = e4m3(scales[c / 16]) * phi;
        try std.testing.expect(@abs(got - want) <= scale + 1e-4);
    }
}

test "aux plans name expert 0 and every row" {
    var buf: [auxBytes()]u8 align(4) = undefined;
    fillAux(&buf);
    const words: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, &buf));
    try std.testing.expectEqual(@as(i32, 0), words[0]);
    try std.testing.expectEqual(@as(i32, 7), words[aux_ids + 7]);
    const at = aux_ids + aux_members + 7 * 4;
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 8, 1 }, words[at..][0..4]);
}

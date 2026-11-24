//! Q8_0-style blocks for Inkling 2 Mini's dense weights on XPU: int8 codes with one fp16 scale for every 32 values of a
//! row, re-encoded from the checkpoint's BF16 at load. Pure host code (no device), shared by the loader and the
//! contract's float64 reference.
//!
//! A matrix [n][k] with k a multiple of 256 is one buffer: the code plane (n * k bytes), then the scale plane (n * k / 32
//! fp16 values, row major). Within a row the codes are laid out in 256-byte planes of eight blocks, the order the decode
//! kernel's sub-group reads them: plane p holds blocks 8p to 8p + 7, and in it lane l (0 to 15) owns bytes 16 l to 16 l + 15,
//! which are, for each of the eight blocks in turn, the values 2 l and 2 l + 1 of the block. So one 16-byte load a lane
//! covers a plane in 256 contiguous bytes, and a lane's two values of a block sit side by side.
//!
//! Encoding of a block: amax = max |v|, scale d = fp16(amax / 127) rounded to nearest even, code = clamp(round_to_nearest_even(v / d),
//! -127, 127); a block of zeros (or one whose scale underflows fp16) has d = 0 and all codes 0. A value decodes as code * d.
const std = @import("std");

pub const block: usize = 32;
/// Blocks in a plane, bytes in a plane, and the lanes of the sub-group that reads it.
pub const plane_blocks: usize = 8;
pub const plane_bytes: usize = 256;
pub const lanes: usize = 16;

/// The bytes of a [n][k] matrix: codes then scales.
pub fn bytes(n: usize, k: usize) usize {
    return n * k + n * (k / block) * 2;
}
/// Offset of the scale plane.
pub fn scaleOffset(n: usize, k: usize) usize {
    return n * k;
}
pub fn admits(k: usize) bool {
    return k != 0 and k % (plane_blocks * block) == 0;
}

fn bf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// The byte offset in a row's code plane of element `e` (0 to 31) of block `b`.
pub fn codeOffset(b: usize, e: usize) usize {
    const p = b / plane_blocks;
    const q = b % plane_blocks;
    return p * plane_bytes + (e / 2) * 16 + q * 2 + (e % 2);
}

/// The fp16 scale of a block whose largest magnitude is `amax`, as bits; 0 when it underflows.
pub fn scaleBits(amax: f32) u16 {
    const d: f16 = @floatCast(amax / 127.0);
    return @bitCast(d);
}

/// Encodes `rows` rows of `k` bf16 values (`src` row major) into `codes` (rows * k bytes, the plane layout) and
/// `scales` (rows * k / 32 fp16 values).
pub fn encodeRows(src: []const u16, rows: usize, k: usize, codes: []u8, scales: []u16) void {
    std.debug.assert(admits(k) and src.len == rows * k and codes.len == rows * k and scales.len == rows * (k / block));
    const nb = k / block;
    for (0..rows) |j| {
        const row = src[j * k ..][0..k];
        const out = codes[j * k ..][0..k];
        for (0..nb) |b| {
            const v = row[b * block ..][0..block];
            var amax: f32 = 0;
            for (v) |h| amax = @max(amax, @abs(bf16(h)));
            const sbits = scaleBits(amax);
            scales[j * nb + b] = sbits;
            const d: f32 = @floatCast(@as(f16, @bitCast(sbits)));
            for (0..block) |e| {
                var q: i32 = 0;
                if (d != 0) {
                    // nearest even by the 1.5 * 2^23 trick (|x| is at most about 127.5 here)
                    const x = std.math.clamp(bf16(v[e]) / d, -128, 128);
                    const even = (x + 0x1.8p23) - 0x1.8p23;
                    q = @intFromFloat(std.math.clamp(even, -127, 127));
                }
                out[codeOffset(b, e)] = @bitCast(@as(i8, @intCast(q)));
            }
        }
    }
}

/// The decoded value of element `e` of block `b` of row `j` (float64), for references.
pub fn decode(codes: []const u8, scales: []const u16, k: usize, j: usize, b: usize, e: usize) f64 {
    const nb = k / block;
    const q: i8 = @bitCast(codes[j * k + codeOffset(b, e)]);
    const d: f64 = @floatCast(@as(f16, @bitCast(scales[j * nb + b])));
    return @as(f64, @floatFromInt(q)) * d;
}

test "layout: a lane's bytes in a plane are its two values of each of eight blocks" {
    try std.testing.expectEqual(@as(usize, 0), codeOffset(0, 0));
    try std.testing.expectEqual(@as(usize, 1), codeOffset(0, 1));
    try std.testing.expectEqual(@as(usize, 2), codeOffset(1, 0));
    try std.testing.expectEqual(@as(usize, 16), codeOffset(0, 2));
    try std.testing.expectEqual(@as(usize, 255), codeOffset(7, 31));
    try std.testing.expectEqual(@as(usize, 256), codeOffset(8, 0));
    // a bijection on a row of two planes
    var seen: [512]bool = @splat(false);
    for (0..16) |b| for (0..32) |e| {
        const o = codeOffset(b, e);
        try std.testing.expect(!seen[o]);
        seen[o] = true;
    };
}

test "encode: values come back within half a step, zeros stay zero, the largest is 127" {
    var src: [256]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&src, 0..) |*h, i| {
        const v: f32 = if (i < 32) 0 else (prng.random().float(f32) - 0.5) * (if (i < 64) @as(f32, 1e-3) else 4);
        h.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
    var codes: [256]u8 = undefined;
    var scales: [8]u16 = undefined;
    encodeRows(&src, 1, 256, &codes, &scales);
    try std.testing.expectEqual(@as(u16, 0), scales[0]);
    for (0..8) |b| for (0..32) |e| {
        const want: f64 = bf16(src[b * 32 + e]);
        const got = decode(&codes, &scales, 256, 0, b, e);
        const d: f64 = @floatCast(@as(f16, @bitCast(scales[b])));
        try std.testing.expect(@abs(got - want) <= d * 0.5 + 1e-12);
    };
    var top: i32 = 0;
    for (0..32) |e| top = @max(top, @abs(@as(i32, @as(i8, @bitCast(codes[codeOffset(3, e)])))));
    try std.testing.expect(top >= 126 and top <= 127);
}

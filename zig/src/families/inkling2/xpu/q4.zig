//! Q4 blocks for Inkling 2 Mini's dense projections on XPU: signed 4-bit codes and one fp16 scale for every 16 values
//! of a row, re-encoded from bf16 at load. The code plane is q8.zig's layout with two codes in each byte, so a
//! sub-group reads a plane of eight 32-value blocks in 128 contiguous bytes. The scale is finer than that plane: one
//! fp16 for elements 0 to 15 of a block and one for 16 to 31, because a group of 32 missed the quality bar.
//!
//! A matrix [n][k], k a multiple of 256, is one buffer: codes (n * k / 2 bytes), then scales (n * k / 16 fp16, row
//! major). Lane l's eight bytes of a plane hold, for each block in turn, elements 2l and 2l + 1.
//!
//! A group of 16: amax = max |v|, scale d = fp16(amax / 7), code = clamp(round_to_nearest_even(v / d), -7, 7). A zero
//! group has d = 0 and codes 0. A value decodes as code * d.
const std = @import("std");

pub const group: usize = 16;
pub const code_block: usize = 32;
pub const plane_blocks: usize = 8;
pub const plane_bytes: usize = 128;

pub fn bytes(n: usize, k: usize) usize {
    return n * (k / 2) + n * (k / group) * 2;
}
pub fn scaleOffset(n: usize, k: usize) usize {
    return n * (k / 2);
}
pub fn admits(k: usize) bool {
    return k != 0 and k % (plane_blocks * code_block) == 0;
}

fn bf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
fn scaleBits(amax: f32) u16 {
    const d: f16 = @floatCast(amax / 7.0);
    return @bitCast(d);
}

/// The byte in a row's code plane that holds element `e` of block `b`. `e` may be odd: the high nibble.
pub fn codeByte(b: usize, e: usize) usize {
    const p = b / plane_blocks;
    const q = b % plane_blocks;
    return p * plane_bytes + (e / 2) * 8 + q;
}

pub fn encodeRows(src: []const u16, rows: usize, k: usize, codes: []u8, scales: []u16) void {
    std.debug.assert(admits(k) and src.len == rows * k and codes.len == rows * (k / 2) and scales.len == rows * (k / group));
    const nb = k / code_block;
    @memset(codes, 0);
    for (0..rows) |j| {
        const row = src[j * k ..][0..k];
        const out = codes[j * (k / 2) ..][0 .. k / 2];
        const parts = code_block / group;
        for (0..nb) |b| for (0..parts) |part| {
            const v = row[b * code_block + part * group ..][0..group];
            var amax: f32 = 0;
            for (v) |h| amax = @max(amax, @abs(bf16(h)));
            const sbits = scaleBits(amax);
            scales[j * (k / group) + b * parts + part] = sbits;
            const d: f32 = @floatCast(@as(f16, @bitCast(sbits)));
            if (d == 0) continue;
            for (0..group) |e| {
                const x = std.math.clamp(bf16(v[e]) / d, -8, 8);
                const even = (x + 0x1.8p23) - 0x1.8p23;
                const q: i8 = @intFromFloat(std.math.clamp(even, -7, 7));
                const at = codeByte(b, part * group + e);
                const nib: u8 = @as(u4, @bitCast(@as(i4, @intCast(q))));
                if ((part * group + e) % 2 == 0) out[at] = nib else out[at] |= nib << 4;
            }
        };
    }
}

pub fn decode(codes: []const u8, scales: []const u16, k: usize, j: usize, b: usize, e: usize) f64 {
    const byte = codes[j * (k / 2) + codeByte(b, e)];
    const nib: i4 = @bitCast(@as(u4, if (e % 2 == 0) byte & 0xf else byte >> 4));
    const d: f64 = @floatCast(@as(f16, @bitCast(scales[j * (k / group) + b * (code_block / group) + e / group])));
    return @as(f64, @floatFromInt(nib)) * d;
}

test "a lane's eight bytes are its pair of each of eight blocks" {
    try std.testing.expectEqual(@as(usize, 0), codeByte(0, 0));
    try std.testing.expectEqual(@as(usize, 0), codeByte(0, 1));
    try std.testing.expectEqual(@as(usize, 1), codeByte(1, 0));
    try std.testing.expectEqual(@as(usize, 8), codeByte(0, 2));
    try std.testing.expectEqual(@as(usize, 128), codeByte(8, 0));
}

test "encode comes back within half a step" {
    var src: [256]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(9);
    for (&src, 0..) |*h, i| {
        const v: f32 = if (i < 32) 0 else (prng.random().float(f32) - 0.5) * 3;
        h.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
    var codes: [128]u8 = undefined;
    var scales: [32]u16 = undefined;
    encodeRows(&src, 1, 256, &codes, &scales);
    try std.testing.expectEqual(@as(u16, 0), scales[0]);
    try std.testing.expectEqual(@as(u16, 0), scales[1]);
    for (0..8) |b| for (0..32) |e| {
        const want: f64 = bf16(src[b * 32 + e]);
        const got = decode(&codes, &scales, 256, 0, b, e);
        const d: f64 = @floatCast(@as(f16, @bitCast(scales[b * (32 / group) + e / group])));
        try std.testing.expect(@abs(got - want) <= d * 0.5 + 1e-6);
    };
}

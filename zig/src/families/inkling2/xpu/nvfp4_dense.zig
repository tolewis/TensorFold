//! Host NVFP4 for dense matrices, matching `loop_model.nvfp4ReferenceWeight`.
//! Codes are packed low-nibble-even. One E4M3 scale per 16 values. `amax` is the tensor max, denominator is 2688.
const std = @import("std");

pub const group: usize = 16;
pub const denominator: f32 = 2688;

pub fn bytes(n: usize, k: usize) usize {
    return n * (k / 2) + n * (k / group) + 8;
}

pub fn admits(k: usize) bool {
    return k != 0 and k % group == 0;
}

fn bf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    const rounded = (bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16;
    return @bitCast(rounded << 16);
}

fn e2m1(code: u8) f32 {
    const table = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const mag = table[code & 7];
    return if (code & 8 != 0) -mag else mag;
}

fn e4m3(code: u8) f32 {
    const exp: u8 = (code >> 3) & 0xf;
    const man: u8 = code & 7;
    if (exp == 15 and man == 7) return std.math.nan(f32);
    const mag: f32 = if (exp == 0)
        @as(f32, @floatFromInt(man)) * 0x1p-9
    else
        std.math.ldexp(1 + @as(f32, @floatFromInt(man)) / 8, @as(i32, exp) - 7);
    return if (code & 0x80 != 0) -mag else mag;
}

fn nearestE4m3(x: f32) u8 {
    var best: u8 = 0;
    var err: f32 = std.math.inf(f32);
    for (0..127) |c| {
        const v = e4m3(@intCast(c));
        const d = @abs(v - x);
        if (d < err) {
            err = d;
            best = @intCast(c);
        }
    }
    return best;
}

fn decoded(code: u8, scale: u8, factor: f32) f32 {
    return bf16(bf16(e2m1(code) * e4m3(scale)) * factor);
}

fn nearestCode(value: f32, scale: u8, factor: f32) u8 {
    var best: u8 = 0;
    var err: f32 = std.math.inf(f32);
    for (0..16) |c| {
        const d = @abs(decoded(@intCast(c), scale, factor) - value);
        if (d < err) {
            err = d;
            best = @intCast(c);
        }
    }
    return best;
}

/// Encodes `rows` of `k` bf16 values. `amax` is the tensor max, already rounded to bf16.
pub fn encodeRows(src: []const u16, rows: usize, k: usize, amax: f32, codes: []u8, scales: []u8) void {
    const nb = k / group;
    const factor = bf16(bf16(amax) / bf16(denominator));
    for (0..rows) |j| {
        const row = src[j * k ..][0..k];
        for (0..nb) |b| {
            var local: f32 = 0;
            for (0..group) |e| {
                const h: u32 = row[b * group + e];
                local = @max(local, @abs(@as(f32, @bitCast(h << 16))));
            }
            const ideal: f32 = if (factor == 0) 0 else local / (6 * factor);
            const scale = nearestE4m3(ideal);
            scales[j * nb + b] = scale;
            var packed: [group / 2]u8 = @splat(0);
            for (0..group) |e| {
                const h: u32 = row[b * group + e];
                const value: f32 = @bitCast(h << 16);
                const code = nearestCode(value, scale, factor);
                if (e % 2 == 0) packed[e / 2] = code else packed[e / 2] |= code << 4;
            }
            @memcpy(codes[j * (k / 2) + b * (group / 2) ..][0..group / 2], &packed);
        }
    }
}

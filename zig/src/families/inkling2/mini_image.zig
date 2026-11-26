//! Inkling 2's image processor (Inkling2ImageProcessor): PNG/JPEG/WebP decode, EXIF transpose, 8x8 checkerboard
//! alpha composite, optional aspect-preserving downscale to fit max_tokens, -1.0 pad to 40x40 tiles, [-1, 1] scale,
//! and spatial-merge-block patch packing [gh * gw, 1200] in f32.
const std = @import("std");
const core = @import("core");

const Allocator = std.mem.Allocator;
const picture = core.picture;

pub const patch_size: u32 = 20;
pub const merge_size: u32 = 2;
pub const tile_size: u32 = patch_size * merge_size; // 40 px per language model image token
pub const patch_width: usize = 3 * patch_size * patch_size; // 1200 f32 per native patch

/// One prepared image ready for the vision tower.
pub const Prepared = struct {
    pixels: []f32, // [gh * gw, 1200]
    gh: u32,
    gw: u32,
    tokens: u32,
    hash: u64,
};

pub const Error = picture.Error || error{ImageTooSmall};

/// Oriented RGB with Inkling2's 8x8 checkerboard (204 / 153) for RGBA alpha.
fn orientAndComposite(a: Allocator, s: picture.Samples) Allocator.Error!picture.Rgb {
    const o = if (s.orientation >= 2 and s.orientation <= 8) s.orientation else 1;
    const turned = o >= 5;
    const w, const h = if (turned) .{ s.height, s.width } else .{ s.width, s.height };
    const out = try a.alloc(u8, w * h * 3);
    for (0..h) |y| for (0..w) |x| {
        const sx, const sy = switch (o) {
            2 => .{ s.width - 1 - x, y },
            3 => .{ s.width - 1 - x, s.height - 1 - y },
            4 => .{ x, s.height - 1 - y },
            5 => .{ y, x },
            6 => .{ y, s.height - 1 - x },
            7 => .{ s.width - 1 - y, s.height - 1 - x },
            8 => .{ s.width - 1 - y, x },
            else => .{ x, y },
        };
        const px = s.data[(sy * s.width + sx) * s.channels ..][0..s.channels];
        const dst = out[(y * w + x) * 3 ..][0..3];
        if (s.channels == 4) {
            const alpha: u32 = px[3];
            const bg: u32 = if (((y / 8) + (x / 8)) % 2 == 0) 204 else 153;
            for (dst, px[0..3]) |*c, v| {
                c.* = @intCast((@as(u32, v) * alpha + bg * (255 - alpha) + 127) / 255);
            }
        } else {
            @memcpy(dst, px[0..3]);
        }
    };
    return .{ .width = w, .height = h, .pixels = out };
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// If ceil(h/40) * ceil(w/40) exceeds max_tokens, find the largest aspect-preserving (ch, cw) that fits.
fn fitDimensions(h: usize, w: usize, max_tokens: u32) error{ImageTooSmall}!struct { h: usize, w: usize } {
    if (max_tokens == 0) return error.ImageTooSmall;
    if (ceilDiv(h, tile_size) * ceilDiv(w, tile_size) <= max_tokens) return .{ .h = h, .w = w };
    var low: usize = 1;
    var high: usize = h;
    var best_h: usize = 1;
    var best_w: usize = 1;
    while (low <= high) {
        const mid = low + (high - low) / 2;
        const cw = @max(@as(usize, 1), (w * mid) / h);
        if (ceilDiv(mid, tile_size) * ceilDiv(cw, tile_size) <= max_tokens) {
            best_h = mid;
            best_w = cw;
            low = mid + 1;
        } else {
            if (mid == 1) break;
            high = mid - 1;
        }
    }
    if (ceilDiv(best_h, tile_size) * ceilDiv(best_w, tile_size) > max_tokens) return error.ImageTooSmall;
    return .{ .h = best_h, .w = best_w };
}

/// Pack RGB image into [rows * cols * 4, 1200] f32 patches in (block_row, block_col, my, mx, py, px, ch) order.
/// Pixels past (height, width) are filled with -1.0 on the 0-255 scale before rescaling to [-1, 1].
pub fn packPatches(a: Allocator, rgb: picture.Rgb) Allocator.Error!struct { pixels: []f32, gh: u32, gw: u32, tokens: u32 } {
    const rows = ceilDiv(rgb.height, tile_size);
    const cols = ceilDiv(rgb.width, tile_size);
    const tokens = rows * cols;
    const gh: u32 = @intCast(rows * merge_size);
    const gw: u32 = @intCast(cols * merge_size);
    const patches = tokens * merge_size * merge_size;
    const out = try a.alloc(f32, patches * patch_width);

    const scale: f32 = 2.0 / 255.0;
    const pad_val: f32 = (-1.0) * scale - 1.0;

    var patch_idx: usize = 0;
    for (0..rows) |br| {
        for (0..cols) |bc| {
            for (0..merge_size) |my| {
                for (0..merge_size) |mx| {
                    const y0 = (br * merge_size + my) * patch_size;
                    const x0 = (bc * merge_size + mx) * patch_size;
                    const dst = out[patch_idx * patch_width ..][0..patch_width];
                    var off: usize = 0;
                    for (0..patch_size) |dy| {
                        const y = y0 + dy;
                        for (0..patch_size) |dx| {
                            const x = x0 + dx;
                            if (y < rgb.height and x < rgb.width) {
                                const src = rgb.pixels[(y * rgb.width + x) * 3 ..][0..3];
                                dst[off + 0] = @as(f32, @floatFromInt(src[0])) * scale - 1.0;
                                dst[off + 1] = @as(f32, @floatFromInt(src[1])) * scale - 1.0;
                                dst[off + 2] = @as(f32, @floatFromInt(src[2])) * scale - 1.0;
                            } else {
                                dst[off + 0] = pad_val;
                                dst[off + 1] = pad_val;
                                dst[off + 2] = pad_val;
                            }
                            off += 3;
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }
    return .{ .pixels = out, .gh = gh, .gw = gw, .tokens = @intCast(tokens) };
}

/// Decode image bytes (PNG, JPEG or WebP), composite alpha over checkerboard, fit max_tokens, and pack patches.
pub fn prepare(io: std.Io, a: Allocator, bytes: []const u8, max_tokens: u32) Error!Prepared {
    if (max_tokens == 0) return error.ImageTooSmall;
    const limits: picture.Limits = .{};
    const samples = switch (picture.sniff(bytes)) {
        .png => try picture.png.decodeWithLimits(a, bytes, limits),
        .jpeg => try picture.jpeg.decodeWithLimits(io, a, bytes, limits),
        .webp => try picture.webp.decodeWithLimits(io, a, bytes, limits),
        .unknown => return error.Invalid,
    };
    defer a.free(samples.data);

    var rgb = try orientAndComposite(a, samples);
    defer rgb.deinit(a);

    const fit = try fitDimensions(rgb.height, rgb.width, max_tokens);
    var resized: ?picture.Rgb = null;
    defer if (resized) |*r| r.deinit(a);
    const active = if (fit.h != rgb.height or fit.w != rgb.width) blk: {
        const px = try picture.resample.resize(a, .pillow, picture.resample.View.of(rgb.pixels, rgb.width, rgb.height), fit.w, fit.h);
        resized = .{ .width = fit.w, .height = fit.h, .pixels = px };
        break :blk resized.?;
    } else rgb;

    const packed_ = try packPatches(a, active);
    const hash = picture.identity.withGrid(picture.identity.ofBytes(0x494e4b3256495331, bytes), .{ packed_.gh, packed_.gw });
    return .{
        .pixels = packed_.pixels,
        .gh = packed_.gh,
        .gw = packed_.gw,
        .tokens = packed_.tokens,
        .hash = hash,
    };
}

test "Inkling2ImageProcessor: checkerboard alpha, -1.0 pad, and patch order" {
    const a = std.testing.allocator;
    // 10x10 RGBA: fully transparent (alpha=0) -> checkerboard 204 at (0,0) and 153 at (0,8)
    const rgba = try a.alloc(u8, 10 * 10 * 4);
    defer a.free(rgba);
    @memset(rgba, 0);
    var rgb = try orientAndComposite(a, .{ .width = 10, .height = 10, .channels = 4, .data = rgba });
    defer rgb.deinit(a);
    try std.testing.expectEqual(@as(u8, 204), rgb.pixels[0]);
    try std.testing.expectEqual(@as(u8, 153), rgb.pixels[8 * 3]);

    const p = try packPatches(a, rgb);
    defer a.free(p.pixels);
    try std.testing.expectEqual(@as(u32, 1), p.tokens);
    try std.testing.expectEqual(@as(u32, 2), p.gh);
    try std.testing.expectEqual(@as(u32, 2), p.gw);
    try std.testing.expectEqual(@as(usize, 4 * 1200), p.pixels.len);
    // (0,0) is 204 -> 204*(2/255) - 1 = 0.6
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), p.pixels[0], 1e-5);
    // (0, 20) is padded (-1.0) -> -1.0*(2/255) - 1.0 = -1.0078431
    try std.testing.expectApproxEqAbs(@as(f32, -2.0 / 255.0 - 1.0), p.pixels[1 * 1200], 1e-5);
}

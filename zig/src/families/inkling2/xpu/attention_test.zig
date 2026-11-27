//! Host-only Mini window, ring and dispatch geometry checks for the XPU attention; no Level Zero import.
const std = @import("std");
const p = @import("attention_plan.zig");
const t = std.testing;

test "inkling2 xpu int8 cache refuses a segment offset" {
    const segs = [_]p.Seg{ .{ .row0 = 0, .n = 4, .p0 = 0, .off = 0 }, .{ .row0 = 0, .n = 4, .p0 = 4, .off = 128 } };
    try p.checkSegs(segs[0..1], true);
    try p.checkSegs(&segs, false);
    try t.expectError(error.Inkling2Int8SegmentOffset, p.checkSegs(&segs, true));
}

test "inkling2 xpu ring and window boundaries" {
    try t.expectEqual(@as(usize, 24), @sizeOf(p.Seg));
    try t.expectEqual(@as(u32, 1024), p.ringFor(1024, 1));
    try t.expectEqual(@as(u32, 1088), p.ringFor(1024, 2));
    try t.expectEqual(@as(u32, 3072), p.Layer.sliding(1024, 2048).ring);
    const l = p.Layer.sliding(1024, 1);
    try t.expectEqual(@as(u32, 17), p.slotsFor(l, 1048576));
    try t.expectEqual(@as(?usize, 15), p.partialSlot(l, 15, 1023));
    try t.expectEqual(@as(?usize, null), p.partialSlot(l, 16, 1023));
    try t.expectEqual(@as(?usize, 16), p.partialSlot(l, 16, 1024));
    try t.expectEqual(@as(?usize, 15), p.partialSlot(l, 31, 2047));
}

test "inkling2 xpu decode and prompt dispatch geometry" {
    p.flash_splits = 0;
    const sl = p.Layer.sliding(1024, 8);
    const sr = p.Rows{ .rows = 8, .last_pos = 4095, .streams = 2 };
    const sd = try p.attend(sr, sl, 17);
    try t.expectEqual(p.Path.decode, sd.path);
    try t.expectEqual([3]u32{ 64, 17, 1 }, sd.grid);
    try t.expectEqual([3]u32{ 192, 1, 1 }, sd.merge_grid);
    try t.expectEqual(@as(u32, 16), sd.threads);
    try t.expectEqual(@as(u32, 64), sd.chunk);
    const gl = p.Layer.global(1048576);
    const one = p.Rows{ .rows = 1, .last_pos = 1048575 };
    const gd = try p.attend(one, gl, 4096);
    try t.expectEqual(p.Path.decode, gd.path);
    try t.expectEqual([3]u32{ 8, 4096, 1 }, gd.grid);
    try t.expectError(error.Inkling2Partials, p.attend(one, gl, 4095));
    // R7: the grid holds the live chunks, not the partial stride
    const short = try p.attend(.{ .rows = 1, .last_pos = 1023 }, gl, 4096);
    try t.expectEqual([3]u32{ 8, 4, 1 }, short.grid);
    try t.expectEqual(@as(u32, 1), (try p.attend(.{ .rows = 1, .last_pos = 0 }, gl, 4096)).grid[1]);
    try t.expectEqual(@as(u32, 1), (try p.attend(.{ .rows = 1, .last_pos = 0 }, sl, 17)).grid[1]);
    // a window: positions 1024 (lo 1, chunk 0 .. 16) and 1023 (chunk 0 .. 15) at chunk 64
    try t.expectEqual(@as(u32, 17), (try p.attend(.{ .rows = 2, .last_pos = 1024 }, sl, 17)).grid[1]);
    try t.expectEqual(@as(u32, 16), (try p.attend(.{ .rows = 1, .last_pos = 1023 }, sl, 17)).grid[1]);
    try t.expectEqual(@as(u32, 16), (try p.attend(.{ .rows = 1, .last_pos = 1087 }, sl, 17)).grid[1]);
    // 128 splits: a 40,001-key global visit at chunk 320 (126 chunks), a 1,024-key window at chunk 16 (65 at most)
    p.flash_splits = 128;
    defer p.flash_splits = 0;
    const g40 = try p.attend(.{ .rows = 1, .last_pos = 40000 }, p.Layer.global(40072), 157);
    try t.expectEqual(@as(u32, 320), g40.chunk);
    try t.expectEqual(@as(u32, 126), g40.grid[1]);
    const s40 = try p.attend(.{ .rows = 1, .last_pos = 40000 }, p.Layer.sliding(1024, 1), 157);
    try t.expectEqual(@as(u32, 16), s40.chunk);
    try t.expectEqual(@as(u32, 65), s40.grid[1]);
    for ([_]u32{ 39999, 40000, 40015, 40016 }) |pos| {
        const s1 = try p.attend(.{ .rows = 1, .last_pos = pos }, p.Layer.sliding(1024, 1), 157);
        const lo = pos + 1 - 1024;
        try t.expectEqual(pos / s1.chunk + 1 - lo / s1.chunk, s1.grid[1]);
    }
    try t.expectError(error.Inkling2Heads, p.attend(.{ .rows = 1, .last_pos = 5, .heads = 16 }, gl, 4096));
    const prompt = try p.attend(.{ .rows = 130, .last_pos = 20000 }, gl, 79);
    try t.expectEqual(p.Path.prompt, prompt.path);
    try t.expectEqual([3]u32{ 25, 8, 20 }, prompt.grid); // ceil(130 * 3 / 16) pair tiles, 20 chunks of 1024
    try t.expectEqual(@as(u32, 1024), prompt.chunk);
    try t.expectEqual(@as(u32, 16), prompt.threads);
    try t.expectError(error.Inkling2Path, p.attendAs(.{ .rows = 4, .last_pos = 99, .streams = 2 }, gl, 4096, .prompt));
    try t.expectError(error.Inkling2Path, p.attendAs(.{ .rows = 4, .last_pos = 99 }, sl, 17, .prompt));
}

test "inkling2 xpu int8 K tile bytes: a sliding ring and a global ring" {
    // one head's ring: 128 bytes a position, 2,048 a tile of 16 positions
    try t.expectEqual(@as(usize, 2048), p.kv_tile_bytes);
    for ([_]usize{ 1088, 65536 }) |ring| {
        // head 0, tile 0: dim 0 of position 0 is the first byte, dim 127 of position 15 the last
        try t.expectEqual(@as(usize, 0), p.kCodeOffset(0, 0, 0, ring));
        try t.expectEqual(@as(usize, 2047), p.kCodeOffset(0, 15, 127, ring));
        // dim d of a position sits at (d / 4) * 64 + (p % 16) * 4 + d % 4 inside its tile
        try t.expectEqual(@as(usize, 2048 + 3 * 64 + 5 * 4 + 2), p.kCodeOffset(0, 21, 14, ring));
        try t.expectEqual(@as(usize, 2048 + 7 * 64 + 0 * 4 + 1), p.kCodeOffset(0, 16, 29, ring));
        // head 3 starts after three head rings, and the position wraps at the ring
        try t.expectEqual(@as(usize, 3 * ring * 128 + 2048 + 2 * 64 + 9 * 4 + 3), p.kCodeOffset(3, ring + 25, 11, ring));
        // whole tiles a head: 100 live positions take 112
        try t.expectEqual(@as(usize, 112 * 128), p.kCodeBytes(100));
        try t.expectEqual(@as(usize, 128 * 128), p.kCodeBytes(128));
    }
    // V codes are row-major and scales head-major
    try t.expectEqual(@as(usize, 5 * 1088 * 128 + 7 * 128 + 96), p.vCodeOffset(5, 1088 + 7, 96, 1088));
    try t.expectEqual(@as(usize, (5 * 1088 + 7) * 4), p.kvScaleOffset(5, 1088 + 7, 1088));
}

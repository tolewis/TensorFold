//! Host-only Mini window, ring and dispatch geometry checks for the XPU attention; no Level Zero import.
const std = @import("std");
const p = @import("attention_plan.zig");
const t = std.testing;

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
    p.rows_old = false;
    const sl = p.Layer.sliding(1024, 8);
    const sr = p.Rows{ .rows = 8, .last_pos = 4095, .streams = 2 };
    const sd = try p.attend(sr, sl, 17);
    try t.expectEqual(p.Path.decode_rows, sd.path);
    // every K/V head, and the seventeen chunks the eight rows read from the lowest first key
    try t.expectEqual([3]u32{ 8, 17, 1 }, sd.grid);
    try t.expectEqual(@as(u32, 47), sd.c0);
    try t.expectEqual([3]u32{ 192, 1, 1 }, sd.merge_grid);
    try t.expectEqual(@as(u32, 16), sd.threads);
    try t.expectEqual(@as(u32, 64), sd.chunk);
    const one = p.Rows{ .rows = 1, .last_pos = 4095 };
    const s1 = try p.attend(one, sl, 17);
    try t.expectEqual(p.Path.decode, s1.path);
    try t.expectEqual([3]u32{ 8, 17, 1 }, s1.grid);
    try t.expectError(error.Inkling2Rows, p.attendRows(one, sl, 17));
    const gl = p.Layer.global(1048576);
    const gr = p.Rows{ .rows = 4, .last_pos = 1029 };
    const gd = try p.attend(gr, gl, 4096);
    try t.expectEqual(p.Path.decode_rows, gd.path);
    try t.expectEqual([3]u32{ 8, 5, 1 }, gd.grid); // chunks 0 .. 1029/256
    try t.expectEqual(@as(u32, 0), gd.c0);
    const gone = p.Rows{ .rows = 1, .last_pos = 1048575 };
    const g1 = try p.attend(gone, gl, 4096);
    try t.expectEqual(p.Path.decode, g1.path);
    try t.expectEqual([3]u32{ 8, 4096, 1 }, g1.grid);
    try t.expectError(error.Inkling2Partials, p.attend(gone, gl, 4095));
    try t.expectError(error.Inkling2Heads, p.attend(.{ .rows = 1, .last_pos = 5, .heads = 16 }, gl, 4096));
    const prompt = try p.attend(.{ .rows = 130, .last_pos = 20000 }, gl, 79);
    try t.expectEqual(p.Path.prompt, prompt.path);
    try t.expectEqual([3]u32{ 25, 8, 20 }, prompt.grid); // ceil(130 * 3 / 16) pair tiles, 20 chunks of 1024
    try t.expectEqual(@as(u32, 1024), prompt.chunk);
    try t.expectEqual(@as(u32, 16), prompt.threads);
    try t.expectError(error.Inkling2Path, p.attendAs(.{ .rows = 4, .last_pos = 99, .streams = 2 }, gl, 4096, .prompt));
    try t.expectError(error.Inkling2Path, p.attendAs(.{ .rows = 4, .last_pos = 99 }, sl, 17, .prompt));
    // the A/B: the one-row kernel, one launch a row
    p.rows_old = true;
    defer p.rows_old = null;
    const old = try p.attend(sr, sl, 17);
    try t.expectEqual(p.Path.decode, old.path);
    try t.expectEqual([3]u32{ 64, 17, 1 }, old.grid);
}

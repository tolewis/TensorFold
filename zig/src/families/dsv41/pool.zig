//! The shared pool's size and the RoPE rows it needs. The server's window (--context) is the most positions one request may
//! use; the pool (TF_DS_POOL) is how many positions the streams' extents and kept prompts share, and may exceed the window so
//! that more long prompts stay resident. A stream's RoPE positions run from 0 inside its own extent, so the tables need only
//! the window's rows, however large the pool is (TF_DS_WINDOW names the window to the ranks that are not told it by the server).
const std = @import("std");

/// A positive count from an environment value; null when unset or blank, `bad` when it is anything else.
fn count(raw: ?[]const u8, bad: anyerror) !?usize {
    const t = std.mem.trim(u8, raw orelse return null, " \t\r\n");
    if (t.len == 0) return null;
    const n = std.fmt.parseInt(usize, t, 10) catch return bad;
    if (n == 0) return bad;
    return n;
}

/// The pool's positions: TF_DS_POOL when set (it may not be smaller than the window), else the window itself.
pub fn resolve(context: usize, raw: ?[]const u8) !usize {
    const n = (try count(raw, error.BadPool)) orelse return context;
    if (n < context) return error.BadPool;
    return n;
}

/// The positions one stream can reach: TF_DS_WINDOW when set, else the whole pool; never more than the pool.
pub fn window(pool: usize, raw: ?[]const u8) !usize {
    const n = (try count(raw, error.BadWindow)) orelse return pool;
    return @min(n, pool);
}

/// The rows each RoPE table holds on the device: every position of a stream plus a round's rows and one.
pub fn ropeRows(window_positions: usize, round_rows: usize) usize {
    return window_positions + round_rows + 1;
}

test "pool: unset or blank follows the window; a larger pool is kept; a smaller or malformed one is refused" {
    try std.testing.expectEqual(@as(usize, 1048576), try resolve(1048576, null));
    try std.testing.expectEqual(@as(usize, 1048576), try resolve(1048576, "  "));
    try std.testing.expectEqual(@as(usize, 1048576), try resolve(1048576, "1048576"));
    try std.testing.expectEqual(@as(usize, 2097152), try resolve(1048576, " 2097152\n"));
    for ([_][]const u8{ "1048575", "0", "-1", "2M", "1.5" }) |bad| try std.testing.expectError(error.BadPool, resolve(1048576, bad));
}

test "window: unset is the pool; set is kept up to the pool; zero or malformed is refused" {
    try std.testing.expectEqual(@as(usize, 2097152), try window(2097152, null));
    try std.testing.expectEqual(@as(usize, 1048576), try window(2097152, "1048576"));
    try std.testing.expectEqual(@as(usize, 1048576), try window(1048576, "2097152")); // never past the pool
    for ([_][]const u8{ "0", "x", "-5" }) |bad| try std.testing.expectError(error.BadWindow, window(2097152, bad));
}

test "rope rows: the window's positions plus a round's rows plus one, whatever the pool is" {
    try std.testing.expectEqual(@as(usize, 1048576 + 48 + 1), ropeRows(1048576, 48));
    try std.testing.expectEqual(@as(usize, 1049), ropeRows(1000, 48));
}

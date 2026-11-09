//! Joining consecutive K/V page ids into one copy. Host only: no device.
const std = @import("std");

/// One span of consecutive page ids. `at` is the first index in the id list. `len` pages are one copy.
pub const Run = struct { at: u32, len: u32 };

/// Writes one run per span where each next id is the previous id plus one.
/// Each run is one copy. `out` receives the runs that fit. The return value is the full run count.
pub fn pageRuns(ids: []const u32, out: []Run) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < ids.len) {
        var j = i + 1;
        while (j < ids.len and ids[j - 1] != std.math.maxInt(u32) and ids[j] == ids[j - 1] + 1) : (j += 1) {}
        if (n < out.len) out[n] = .{ .at = @intCast(i), .len = @intCast(j - i) };
        n += 1;
        i = j;
    }
    return n;
}

test "consecutive page ids become one copy" {
    var out: [8]Run = undefined;
    const ids = [_]u32{ 4, 5, 6, 9, 2, 3 };
    const n = pageRuns(&ids, &out);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(Run{ .at = 0, .len = 3 }, out[0]);
    try std.testing.expectEqual(Run{ .at = 3, .len = 1 }, out[1]);
    try std.testing.expectEqual(Run{ .at = 4, .len = 2 }, out[2]);
    const one = [_]u32{8};
    try std.testing.expectEqual(@as(usize, 1), pageRuns(&one, &out));
    try std.testing.expectEqual(@as(u32, 1), out[0].len);
    const gap = [_]u32{ 1, 3 };
    try std.testing.expectEqual(@as(usize, 2), pageRuns(&gap, &out));
    const none: [0]u32 = .{};
    try std.testing.expectEqual(@as(usize, 0), pageRuns(&none, &out));
    const wrap = [_]u32{ std.math.maxInt(u32), 0 };
    try std.testing.expectEqual(@as(usize, 2), pageRuns(&wrap, &out));
}

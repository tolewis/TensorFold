//! Late prompt absorbs leave the window and its rows as eager ones do, and every row they read is still held.
const std = @import("std");
const lazy = @import("cuda_draft_lazy.zig");

const window = 2048; // the 27B DFlash2 drafter's sliding window
const ring_rows = window + 16;
const tap_rows = 6144;
const most_verify = 128;
const limits: lazy.Limits = .{ .keep = @min(window, tap_rows - most_verify), .held = tap_rows - most_verify };
const none = std.math.maxInt(u64);

/// The drafter's absorb on a ring of positions: a read of a row the target's tap ring no longer holds fails.
const Model = struct {
    begin: u64 = 0,
    end: u64 = 0,
    late: lazy.Pending = .{},
    ring: [ring_rows]u64 = @splat(none),
    written: u64 = 0,
    computed: usize = 0,

    pub fn absorb(m: *Model, ctx: *Model, ring: u64, start: u64, n: usize, rows: ?[]const u32, prompt: bool) !void {
        if (m != ctx or ring != 0x7000 or rows != null or !prompt) return error.BadLateStep;
        const r = lazy.open(ctx, start, n, limits.keep);
        for (r.start..r.start + r.n) |p| {
            if (p + tap_rows < m.written) return error.RowNotHeld; // cuda_native.zig's featuresFn rule
            m.ring[p % ring_rows] = p;
        }
        m.computed += r.n;
        m.end = r.start + r.n;
        lazy.close(m, window);
    }
};

fn feed(m: *Model, late: bool, cached: u64, steps: []const usize) !void {
    var at = cached;
    for (steps) |n| {
        m.written = at + n;
        if (late) {
            try lazy.put(m, m, .{ .ring = 0x7000, .start = at, .n = n }, limits);
        } else try m.absorb(m, 0x7000, at, n, null, true);
        at += n;
    }
    m.written += most_verify; // a verify pass of the most rows runs before the window is read
    if (late) try lazy.flush(m, m);
}

/// Eager and late absorbs of the same steps: the same window, the same position in each slot it reads.
fn same(cached: u64, steps: []const usize) !usize {
    var eager: Model = .{};
    var late: Model = .{};
    try feed(&eager, false, cached, steps);
    try feed(&late, true, cached, steps);
    try std.testing.expectEqual(eager.begin, late.begin);
    try std.testing.expectEqual(eager.end, late.end);
    for (eager.begin..eager.end) |p| {
        try std.testing.expectEqual(@as(u64, p), eager.ring[p % ring_rows]);
        try std.testing.expectEqual(eager.ring[p % ring_rows], late.ring[p % ring_rows]);
    }
    try std.testing.expect(late.computed <= eager.computed);
    return late.computed;
}

test "late absorbs: a 40k prompt in 4,096-row steps computes the last step's 2,048 rows and the first token's row" {
    const steps = @as([9]usize, @splat(4096)) ++ [_]usize{ 3135, 1 };
    try std.testing.expectEqual(@as(usize, 2049), try same(0, &steps));
}

test "late absorbs: short steps, steps of exactly the kept rows, a cached prefix" {
    _ = try same(0, &.{ 4096, 1500, 1000 });
    _ = try same(0, &@as([8]usize, @splat(2048)));
    _ = try same(0, &@as([70]usize, @splat(100)));
    _ = try same(0, &@as([12]usize, @splat(1000)));
    _ = try same(30000, &.{ 4096, 700, 1 });
    _ = try same(5, &.{ 1, 1, 1 });
}

test "late absorbs: 2,000 seeded step lists match the eager absorbs" {
    var prng = std.Random.DefaultPrng.init(0x5eed_d4af);
    const r = prng.random();
    var steps: [24]usize = undefined;
    for (0..2000) |_| {
        const count = r.intRangeAtMost(usize, 1, steps.len);
        for (steps[0..count]) |*n| n.* = switch (r.uintLessThan(u8, 4)) {
            0 => r.intRangeAtMost(usize, 1, 64),
            1 => r.intRangeAtMost(usize, 1, 2048),
            2 => r.intRangeAtMost(usize, 2040, 2056),
            else => r.intRangeAtMost(usize, 1, 8192),
        };
        _ = try same(r.uintLessThan(u64, 50000), steps[0..count]);
    }
}

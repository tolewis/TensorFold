//! Shared-page prompt states (XPU_KEEP_PAGES): holding counts each page once, and a state that frees nothing stays.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fake = @import("prompt_cache_fake.zig");
const Store = pc.Store;
const Shared = fake.Shared;
const Fake = fake.Fake;

fn fill(buf: []u32, base: u32) void {
    for (buf, 0..) |*t, i| t.* = base +% @as(u32, @intCast(i));
}

test "keep pages: six conversations of five marks hold the distinct pages" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa };
    defer f.deinit();
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 20);
    defer s.deinit();
    const marks = [_]u32{ 64, 128, 192, 256, 320 };
    var prompt: [321]u32 = undefined;
    var charged: u64 = 0;
    for (0..6) |c| {
        f.resetLive();
        fill(&prompt, @intCast(1000 * c + 1));
        for (marks) |at| {
            try f.grow(at);
            try std.testing.expect(s.keep(prompt[0..321], at, null, &.{}, &.{}));
            charged += at; // bytes(at) is the token count; the prefixes share pages
        }
    }
    // 6 conversations x 5 pages, 64 bytes a page. The sum of the charges counts each prefix again.
    try std.testing.expectEqual(@as(u64, 30 * 64), s.holding());
    try std.testing.expect(s.holding() < charged);
    try std.testing.expectEqual(@as(usize, 30), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    for (0..6) |c| {
        fill(&prompt, @intCast(1000 * c + 1));
        try std.testing.expectEqual(@as(u32, 320), s.find(&prompt, &.{}, &.{}).?.at);
    }
}

test "keep pages: 851 new tokens on a 17152-token state add at most 14 pages" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa };
    defer f.deinit();
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    var prompt: [18004]u32 = undefined;
    fill(&prompt, 1);
    try f.grow(17152);
    try std.testing.expect(s.keep(&prompt, 17152, null, &.{}, &.{}));
    const before = s.holding();
    try f.grow(18003);
    const add = f.snapshots().vtable.adds.?(f.snapshots().ptr, null, 18003);
    try std.testing.expect(add / f.pageBytes() <= 14);
    try std.testing.expect(s.keep(&prompt, 18003, null, &.{}, &.{}));
    try std.testing.expect(s.holding() - before <= 14 * f.pageBytes());
    try std.testing.expectEqual(@as(u64, 14 * 64), s.holding() - before);
}

test "keep pages: a state that frees nothing stays, and the least recently used state that frees bytes leaves" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 1 };
    defer f.deinit();
    var s = Store.init(gpa, f.snapshots(), .{}, 2); // two distinct pages
    defer s.deinit();
    const a = [_]u32{ 10, 1 };
    const b = [_]u32{ 20, 1 };
    const c = [_]u32{ 30, 1 };
    const d = [_]u32{ 40, 1 };
    try f.grow(1); // page 0
    try std.testing.expect(s.keep(&a, 1, null, &.{}, &.{}));
    try std.testing.expect(s.keep(&b, 1, null, &.{}, &.{})); // the same page: this state frees nothing
    f.resetLive();
    try f.grow(1); // page 1, held by one state
    try std.testing.expect(s.keep(&c, 1, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 2), s.holding());
    f.resetLive();
    try f.grow(1); // page 2, one new page, the budget holds two
    try std.testing.expect(s.keep(&d, 1, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}) != null);
    try std.testing.expect(s.find(&b, &.{}, &.{}) != null);
    try std.testing.expect(s.find(&c, &.{}, &.{}) == null); // the only state that freed a page
    try std.testing.expect(s.find(&d, &.{}, &.{}) != null);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evict_budget);
    try std.testing.expectEqual(@as(u64, 2), s.holding());
}

test "keep pages: evictOne is false when no state returns storage, else the least recently used one that does" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 1 };
    defer f.deinit();
    var s = Store.init(gpa, f.snapshots(), .{}, 8);
    defer s.deinit();
    const a = [_]u32{ 10, 1 };
    try f.grow(1);
    try std.testing.expect(s.keep(&a, 1, null, &.{}, &.{}));
    f.hold(0); // the stream holds the page too: dropping the state returns nothing
    try std.testing.expect(!s.evictOne());
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);

    const b = [_]u32{ 20, 1 };
    const c = [_]u32{ 30, 1 };
    f.resetLive();
    try f.grow(1);
    try std.testing.expect(s.keep(&b, 1, null, &.{}, &.{}));
    try std.testing.expect(s.keep(&c, 1, null, &.{}, &.{})); // both hold page 1: neither returns it alone
    // a returns nothing (the stream holds page 0). b and c return nothing (they share page 1).
    try std.testing.expect(!s.evictOne());
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);

    const d = [_]u32{ 40, 1 };
    f.resetLive();
    try f.grow(1); // page 2, one holder
    try std.testing.expect(s.keep(&d, 1, null, &.{}, &.{}));
    try std.testing.expect(s.evictOne());
    try std.testing.expect(s.find(&a, &.{}, &.{}) != null);
    try std.testing.expect(s.find(&b, &.{}, &.{}) != null);
    try std.testing.expect(s.find(&c, &.{}, &.{}) != null);
    try std.testing.expect(s.find(&d, &.{}, &.{}) == null); // the only state that returns a page
    try std.testing.expectEqual(@as(u64, 1), s.counts.evict_stream);
}

test "keep pages: hooks unset, a short budget still drops the least recently used state" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    const snaps = f.snapshots();
    try std.testing.expect(snaps.vtable.held == null);
    try std.testing.expect(snaps.vtable.frees == null);
    try std.testing.expect(snaps.vtable.returns == null);
    var s = Store.init(gpa, snaps, .{ .lookahead = 1, .min_prompt = 0 }, 300);
    defer s.deinit();
    const old = [_]u32{ 7, 8, 9, 1, 2 };
    const mid = [_]u32{ 1, 2, 3, 4, 9 };
    const nxt = [_]u32{ 4, 5, 6, 7, 8 };
    var p = try s.begin(a, &old, 4, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, &old, p);
    p = try s.begin(a, &mid, 4, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, &mid, p);
    p = try s.begin(a, &nxt, 4, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, &nxt, p);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expect(s.find(&old, &.{}, &.{}) == null);
    try std.testing.expect(s.find(&mid, &.{}, &.{}) != null);
    try std.testing.expect(s.find(&nxt, &.{}, &.{}) != null);
}

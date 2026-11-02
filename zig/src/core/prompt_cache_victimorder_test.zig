//! The W5-m2b victim order (item 20 part K): a multi-conversation working set evicts interior cuts before
//! history marks, and a pass's own superseded states before both. A fake family's snapshots are host memory;
//! a state's bytes are 100 + position. Six conversations keep a 10 000-token user message each: its interior
//! cut at 7936 and its history mark at 9984. Twelve states ask for a cap of eight: four interior cuts go, all six
//! history marks survive (the two most recent cuts stay). A miss never evicts another conversation's history mark
//! while an interior cut is free.
//! A live continuation of conversation 0 resumes its history mark (the probe ranks it first), and its own cut keeps
//! with its mark while the other conversations' history marks survive.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Allocator = std.mem.Allocator;
const Fake = @import("prompt_cache_fake.zig").Fake;

/// The index of the entry that `prefix` names, or null.
fn at(s: *const Store, prefix: []const u32) ?usize {
    for (s.entries.items, 0..) |e, i| if (e.at == prefix.len and std.mem.eql(u32, e.tokens[0 .. prefix.len], prefix)) return i;
    return null;
}

/// Conversation `i`'s 10 001-token prompt: 10 000 user tokens plus one trailing token.
fn prompt(a: Allocator, i: u32) ![]u32 {
    const p = try a.alloc(u32, 10001);
    for (p, 0..) |*t, j| t.* = i * 1000000 + @as(u32, @intCast(j)) + 1;
    return p;
}

test "a multi-conversation working set evicts interior cuts before history marks: every history mark survives" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 64 }, 1 << 30);
    defer s.deinit();
    s.most = 8;
    var prompts: std.ArrayList([]u32) = .empty;
    defer prompts.deinit(a);
    for (0..6) |i| try prompts.append(a, try prompt(a, @intCast(i)));
    for (prompts.items) |p| {
        const plan = try s.beginUser(a, p, 10000, 16, 10000, &.{}, &.{}, null, &.{});
        try std.testing.expectEqualSlices(u32, &.{ 7936, 9984 }, plan.marks);
        _ = f.pass(&s, p, plan);
    }
    try std.testing.expectEqual(@as(usize, 8), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 4), s.counts.evicted); // the cuts the cap could not hold
    for (prompts.items) |p| {
        try std.testing.expect(at(&s, p[0..9984]) != null); // every conversation's history mark survived the misses
    }
    // A live continuation of conversation 0: a reply token, then a second 10 000-token user message.
    const p0 = prompts.items[0];
    const p2 = try a.alloc(u32, p0.len + 1 + 10000);
    @memcpy(p2[0..p0.len], p0);
    p2[p0.len] = 6000001;
    for (p2[p0.len + 1 ..], 0..) |*t, j| t.* = 6000002 + @as(u32, @intCast(j));
    const v = s.victimBut(null, p2, @intCast(p2.len)) orelse @panic("a victim exists: the store holds eight states at a cap of eight");
    try std.testing.expectEqual(at(&s, p0[0..9984]).?, v); // its own older state is the first candidate, before another conversation's cut
    const plan = try s.beginUser(a, p2, 20001, 16, 20001, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 9984), plan.from); // it resumes conversation 0's history mark
    try std.testing.expectEqualSlices(u32, &.{ 17920, 19968 }, plan.marks);
    _ = f.pass(&s, p2, plan);
    try std.testing.expectEqual(@as(usize, 8), s.entries.items.len);
    try std.testing.expect(at(&s, p2[0..17920]) != null); // the continuation's own interior cut kept with its history mark
    try std.testing.expect(at(&s, p2[0..19968]) != null);
    for (prompts.items[1..]) |p| {
        try std.testing.expect(at(&s, p[0..9984]) != null); // the other conversations' history marks survived: only interior cuts went
    }
}

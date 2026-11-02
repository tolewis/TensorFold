//! The prompt cache's keep-one rule: a later turn's state takes its own older one's storage, and the least
//! recently used one never wins a kept state's page. A fake family's snapshots are host memory (a position and a
//! running sum); a state's bytes are 100 + position. A budget of 300 holds two of them, so a third keep forces
//! exactly one eviction.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Allocator = std.mem.Allocator;
const Fake = @import("prompt_cache_fake.zig").Fake;

/// The index of the entry that `prefix` names, or null.
fn at(s: *const Store, prefix: []const u32) ?usize {
    for (s.entries.items, 0..) |e, i| if (e.tokens.len >= prefix.len and std.mem.eql(u32, e.tokens[0 .. prefix.len], prefix)) return i;
    return null;
}

test "a later turn's state takes its own older one's storage, not the least recently used state's" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 300); // bytes: 100 + position
    defer s.deinit();
    // An unrelated conversation keeps a state first: the least recently used state when the turn comes.
    const pc_: []const u32 = &[_]u32{ 7, 8, 9, 1, 2 };
    var p = try s.begin(a, pc_, 4, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, pc_, p);
    // Turn 1 of a conversation: history [1 2 3 4], its reply's first token 9 keys the state.
    const p1: []const u32 = &[_]u32{ 1, 2, 3, 4, 9 };
    p = try s.begin(a, p1, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 4 }, p.marks);
    _ = f.pass(&s, p1, p);
    // Turn 2 of the first conversation: history [1 2 3 4 5 6], and its reply's first token differs from the one
    // that keyed turn 1's state, so the prompt extends that state's tokens but not its key.
    const p2: []const u32 = &[_]u32{ 1, 2, 3, 4, 5, 6, 9 };
    p = try s.begin(a, p2, 6, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from); // its own older state is not resumable: the next token differs
    _ = f.pass(&s, p2, p);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(at(&s, &.{ 1, 2, 3, 4 }) == null); // the conversation's older state took its storage
    try std.testing.expectEqual(@as(u32, 6), s.find(p2, &.{}, &.{}).?.at); // its later turn's state stayed
    try std.testing.expectEqual(@as(u32, 4), s.find(pc_, &.{}, &.{}).?.at); // the unrelated state stayed
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted); // exactly one state left; the unrelated one never left
    try std.testing.expect(s.subsumed(p2, 7, 6) == null); // nothing the later turn's state supersedes remains
}

test "a later turn's shorter prompt is a rewind, not a supersede: the longer state is never taken by it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 209 << 20);
    defer s.deinit();
    const p1: []const u32 = &[_]u32{ 1, 2, 3, 4, 9 };
    var p = try s.begin(a, p1, 4, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, p1, p);
    const pc_: []const u32 = &[_]u32{ 7, 8, 9, 1, 2 };
    p = try s.begin(a, pc_, 4, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, pc_, p);
    const ai = at(&s, &.{ 1, 2, 3, 4 }) orelse return;
    const ci = at(&s, &.{ 7, 8, 9, 1 }) orelse return;
    // Turn 2 of the conversation (born 7): its state at 6 supersedes turn 1's state, never the unrelated one's.
    const p2: []const u32 = &[_]u32{ 1, 2, 3, 4, 5, 6, 9 };
    try std.testing.expect(s.subsumes(p2, 7, 6, s.entries.items[ai]));
    try std.testing.expect(!s.subsumes(p2, 7, 6, s.entries.items[ci]));
    try std.testing.expectEqual(@as(usize, ai), s.subsumed(p2, 7, 6));
    try std.testing.expectEqualStrings("superseded by the prompt", s.victimReason(ai, p2, 7));
    try std.testing.expectEqualStrings("least recently used", s.victimReason(ci, p2, 7));
    // A resend of turn 1's history (born 5): a later turn's state, shorter than the one it resumes: supersedes none.
    const p3: []const u32 = &[_]u32{ 1, 2, 3, 4, 7 };
    try std.testing.expect(!s.subsumes(p3, 5, 4, s.entries.items[ai]));
    try std.testing.expect(s.subsumed(p3, 5, 4) == null);
}

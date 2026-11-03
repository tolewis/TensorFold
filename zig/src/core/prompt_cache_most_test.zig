//! The prompt cache's count cap (Store.most), as XPU_KEPT_HOST=1 sets it (lane p43-keptcap): with a cap of 32 and a
//! budget that does not bind, six conversations of five turns keep every turn's state, so each one's latest state is
//! there for its next turn. When the count binds, the least recently used state leaves first. A fake family's
//! snapshots are host memory (bytes 100 + position).
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Fake = @import("prompt_cache_fake.zig").Fake;

const CONVS = 6;
const TURNS = 5;

/// Conversation `c`'s turn `t` (0 based): a history of 4 + 2 t tokens all its own, then its reply's first token.
fn turn(buf: []u32, c: u32, t: u32) []const u32 {
    const n = 4 + 2 * t;
    for (buf[0..n], 0..) |*x, i| x.* = 1000 * (c + 1) + @as(u32, @intCast(i));
    buf[n] = 9;
    return buf[0 .. n + 1];
}

/// Runs conversation `c`'s turn `t` through the store: its pass keeps the mark at the history's end.
fn run(s: *Store, f: *Fake, a: std.mem.Allocator, c: u32, t: u32) !void {
    var buf: [64]u32 = undefined;
    const p = turn(&buf, c, t);
    const plan = try s.begin(a, p, @intCast(p.len - 1), &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{@as(u32, @intCast(p.len - 1))}, plan.marks);
    _ = f.pass(s, p, plan);
}

/// Whether the store holds conversation `c`'s state of turn `t`.
fn holds(s: *Store, c: u32, t: u32) bool {
    var buf: [64]u32 = undefined;
    const p = turn(&buf, c, t);
    const e = s.find(p, &.{}, &.{}) orelse return false;
    return e.at == p.len - 1;
}

test "a cap of 32: six conversations of five marks each keep every state, the latest of each among them" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 40); // the budget never binds
    defer s.deinit();
    s.most = 32;
    for (0..TURNS) |t| for (0..CONVS) |c| try run(&s, &f, a, @intCast(c), @intCast(t)); // turns interleaved, as 6 agents
    try std.testing.expectEqual(@as(usize, CONVS * TURNS), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    for (0..CONVS) |c| try std.testing.expect(holds(&s, @intCast(c), TURNS - 1));
}

test "a cap of 32 binds: the least recently used state leaves first, every conversation's latest stays" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 40);
    defer s.deinit();
    s.most = 32;
    for (0..TURNS) |t| for (0..CONVS) |c| try run(&s, &f, a, @intCast(c), @intCast(t));
    // four new one-turn conversations: 34 states asked of a cap of 32, so the two least recently used leave
    for (CONVS..CONVS + 4) |c| try run(&s, &f, a, @intCast(c), 0);
    try std.testing.expectEqual(@as(usize, 32), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 2), s.counts.evicted);
    try std.testing.expect(!holds(&s, 0, 0)); // the first two states kept, never used since
    try std.testing.expect(!holds(&s, 1, 0));
    try std.testing.expect(holds(&s, 2, 0)); // the next one stays
    for (0..CONVS + 4) |c| try std.testing.expect(holds(&s, @intCast(c), if (c < CONVS) TURNS - 1 else 0));
    // a resume makes a state recently used: conversation 2's first state is used again, so conversation 3's goes next
    {
        var buf: [64]u32 = undefined;
        const p = turn(&buf, 2, 0);
        const plan = try s.begin(a, p, @intCast(p.len - 1), &.{}, &.{}, null, &.{}); // the same prompt again: a hit
        try std.testing.expectEqual(@as(u32, @intCast(p.len - 1)), plan.from); // resumed at its state, its use renewed
        if (plan.entry) |e| s.release(e);
    }
    try run(&s, &f, a, CONVS + 4, 0);
    try std.testing.expectEqual(@as(usize, 32), s.entries.items.len);
    try std.testing.expect(holds(&s, 2, 0));
    try std.testing.expect(!holds(&s, 3, 0));
}

test "today's cap of 8: a cold prompt's five marks leave room for three older states" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 40);
    defer s.deinit();
    s.most = 8;
    for (0..CONVS) |c| try run(&s, &f, a, @intCast(c), 0);
    try std.testing.expectEqual(@as(usize, 6), s.entries.items.len);
    for (0..CONVS) |c| try std.testing.expect(holds(&s, @intCast(c), 0));
    for (0..TURNS) |t| try run(&s, &f, a, CONVS, @intCast(t)); // one long conversation of five turns
    try std.testing.expectEqual(@as(usize, 8), s.entries.items.len);
    try std.testing.expect(holds(&s, CONVS, TURNS - 1));
}

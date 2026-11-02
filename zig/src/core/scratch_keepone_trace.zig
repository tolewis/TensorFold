const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Allocator = std.mem.Allocator;
const Fake = @import("prompt_cache_fake.zig").Fake;

fn dump(label: []const u8, s: *const Store) void {
    std.debug.print("{s}: clock={d} held={d} budget={d}\n", .{ label, s.clock, s.holding(), s.budget });
    for (s.entries.items, 0..) |e, i| std.debug.print("  [{d}] at={d} born={d} used={d} bytes={d} tokens={any}\n", .{ i, e.at, e.born, e.used, e.bytes, e.tokens });
}

test "trace keepone scenario" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 300);
    defer s.deinit();
    const pc_ = [_]u32{ 7, 8, 9, 1, 0 };
    _ = f.pass(&s, &pc_, try s.begin(a, &pc_, 4, &.{}, &.{}, null, &.{}));
    dump("after unrelated", &s);
    const p1 = [_]u32{ 1, 2, 3, 4, 9 };
    _ = f.pass(&s, &p1, try s.begin(a, &p1, 4, &.{}, &.{}, null, &.{}));
    dump("after turn 1", &s);
    const p2 = [_]u32{ 1, 2, 3, 4, 9, 1, 0 };
    const plan = try s.begin(a, &p2, 6, &.{}, &.{}, null, &.{});
    dump("after begin p2 (before pass)", &s);
    _ = f.pass(&s, &p2, plan);
    dump("after turn 2 keep", &s);
    std.debug.print("evicted={d} refused={d} failed={d}\n", .{ s.counts.evicted, s.counts.refused, s.counts.failed });
}

test "trace 136 scenario" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 330);
    defer s.deinit();
    const other = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &other, try s.begin(a, &other, 3, &.{}, &.{}, null, &.{}));
    dump("136 after O", &s);
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{}));
    dump("136 after T1", &s);
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 6, &.{}, &.{}, null, &.{}));
    dump("136 after T2", &s);
    const t3 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    _ = f.pass(&s, &t3, try s.begin(a, &t3, 8, &.{}, &.{}, null, &.{}));
    dump("136 after T3", &s);
    std.debug.print("136 evicted={d} refused={d}\n", .{ s.counts.evicted, s.counts.refused });
}

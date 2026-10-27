//! A family whose kept states share storage (Snapshots.held, adds): a state holds its stream's pages by reference.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Saved = pc.Saved;
const Store = pc.Store;

/// One live stream over a pool of pages, a page a token: a state holds the stream's pages through its position and one
/// page of its own (the fixed part, like a recurrent state), so two states of one conversation share their prefix.
const Paged = struct {
    gpa: std.mem.Allocator,
    refs: [256]u32 = @splat(0), // the kept states that hold each page
    next: u32 = 0, // the next fresh page
    live: [64]u32 = undefined, // the stream's pages, a token each
    at: u32 = 0,

    const State = struct { ids: []u32 }; // the prefix's pages, then the state's own page

    fn snapshots(f: *Paged) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .held = heldFn, .adds = addsFn } };
    }
    fn of(ptr: *anyopaque) *Paged {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return at + 1;
    }
    fn heldFn(ptr: *anyopaque) u64 {
        var n: u64 = 0;
        for (of(ptr).refs) |r| n += @intFromBool(r > 0);
        return n;
    }
    fn addsFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) u64 {
        const f = of(ptr);
        var n: u64 = 1;
        for (f.live[0..at]) |id| n += @intFromBool(f.refs[id] == 0);
        return n;
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (at != f.at) return error.NotAtMark;
        const st = try f.gpa.create(State);
        st.ids = try f.gpa.alloc(u32, at + 1);
        @memcpy(st.ids[0..at], f.live[0..at]);
        st.ids[at] = f.fresh();
        for (st.ids) |id| f.refs[id] += 1;
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: Saved) anyerror!void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        f.at = @intCast(st.ids.len - 1);
        @memcpy(f.live[0..f.at], st.ids[0..f.at]); // the stream maps the state's pages and goes on in fresh ones
    }
    fn dropFn(ptr: *anyopaque, saved: Saved) void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        for (st.ids) |id| f.refs[id] -= 1;
        f.gpa.free(st.ids);
        f.gpa.destroy(st);
    }
    fn fresh(f: *Paged) u32 {
        f.next += 1;
        return f.next - 1;
    }

    /// A prompt pass: resumed at plan.from (begin restored it), else from 0 on fresh pages, keeping at each mark.
    fn pass(f: *Paged, s: *Store, prompt: []const u32, plan: pc.Plan) void {
        if (plan.from == 0) f.at = 0;
        var mi: usize = 0;
        while (f.at < prompt.len) {
            f.live[f.at] = f.fresh();
            f.at += 1;
            if (mi < plan.marks.len and plan.marks[mi] == f.at) {
                _ = s.keep(prompt, f.at, null, &.{}, &.{});
                mi += 1;
            }
        }
    }
};

fn conversation(base: u32, n: usize, out: []u32) []u32 {
    for (out[0..n], 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    return out[0..n];
}

test "a sharing family's next turn costs its new pages: another conversation's state stays" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Paged = .{ .gpa = gpa };
    // two states of 10 tokens hold 22 pages; a next turn's state of 12 adds 2 pages and its own: 25
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 2 }, 25);
    defer s.deinit();
    var ba: [20]u32 = undefined;
    var bb: [20]u32 = undefined;
    const a1 = conversation(1000, 11, &ba);
    f.pass(&s, a1, try s.begin(a, a1, 10, &.{}, &.{}, null, &.{}));
    const b1 = conversation(2000, 11, &bb);
    f.pass(&s, b1, try s.begin(a, b1, 10, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u64, 22), s.holding());
    const a2 = conversation(1000, 13, &ba);
    const p = try s.begin(a, a2, 12, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 10), p.from);
    f.pass(&s, a2, p);
    // begin lets the state it resumed go (superseded, its own page alone): a's 12 pages and own page, b's 11. Charged in
    // full, a's new state (13) beside both 11s would have evicted b's state too.
    try std.testing.expectEqual(@as(u64, 24), s.holding());
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    const b2 = conversation(2000, 13, &bb);
    try std.testing.expectEqual(@as(u32, 10), s.find(b2, &.{}, &.{}).?.at);
}

test "a sharing family evicts until the pages it frees make room" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Paged = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 2 }, 24);
    defer s.deinit();
    var ba: [20]u32 = undefined;
    var bb: [20]u32 = undefined;
    const a1 = conversation(1000, 11, &ba);
    f.pass(&s, a1, try s.begin(a, a1, 10, &.{}, &.{}, null, &.{}));
    const b1 = conversation(2000, 11, &bb);
    f.pass(&s, b1, try s.begin(a, b1, 10, &.{}, &.{}, null, &.{}));
    const a2 = conversation(1000, 13, &ba);
    f.pass(&s, a2, try s.begin(a, a2, 12, &.{}, &.{}, null, &.{})); // 25 does not fit 24: a state goes
    try std.testing.expect(s.holding() <= 24);
    try std.testing.expectEqual(@as(u32, 12), s.find(conversation(1000, 14, &ba), &.{}, &.{}).?.at);
}

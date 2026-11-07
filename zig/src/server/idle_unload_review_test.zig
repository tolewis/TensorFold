//! A request that arrives while the idle timer closes the engine waits for the close, then reloads and is served.
const std = @import("std");
const idle = @import("idle_unload.zig");
const Harness = @import("idle_unload_test.zig").Harness;
const t = std.testing;

const Wake = struct {
    h: *Harness,
    reply: std.heap.ArenaAllocator,
    why: []const u8 = "unset",
    failed: bool = false,
    fn run(w: *Wake) void {
        w.h.man.?.begin(); // the route's count, as routes.post raises it before the wake
        defer w.h.man.?.end();
        w.why = idle.wakeReload(w.h.man.?, w.h.srv.?, w.reply.allocator()) catch {
            w.failed = true;
            return;
        };
    }
};

test "idle review: a wake during the unload waits for the close, then reloads and never returns the retry reason" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.gate_close.store(true, .release);
    h.expire();
    try h.waitEvent(&h.close_entered);
    try t.expectEqual(idle.State.unloading, h.state());
    var w: Wake = .{ .h = &h, .reply = .init(t.allocator) };
    defer w.reply.deinit();
    const thread = try std.Thread.spawn(.{}, Wake.run, .{&w});
    h.sleepMs(100); // the wake is parked on the manager: no reopen while the old engine still holds the card
    try t.expectEqual(@as(u32, 0), h.opens.load(.acquire));
    try t.expectEqual(idle.State.unloading, h.state());
    h.close_release.set(h.io);
    thread.join();
    try t.expect(!w.failed);
    try t.expectEqualStrings("", w.why);
    try t.expectEqual(@as(u32, 1), h.closes.load(.acquire));
    try t.expectEqual(@as(u32, 1), h.opens.load(.acquire));
    try t.expectEqual(idle.State.resident, h.state());
    try t.expect(!h.man.?.isParked());
}

test "idle review: a wake during the unload whose reload fails keeps the reopen's reason, and the next wake reloads" {
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(3600);
    h.gate_close.store(true, .release);
    h.fail_open = true;
    h.expire();
    try h.waitEvent(&h.close_entered);
    var w: Wake = .{ .h = &h, .reply = .init(t.allocator) };
    defer w.reply.deinit();
    const thread = try std.Thread.spawn(.{}, Wake.run, .{&w});
    h.sleepMs(50);
    h.close_release.set(h.io);
    thread.join();
    try t.expect(!w.failed);
    try t.expectEqualStrings("the fake open failed", w.why);
    try t.expectEqual(idle.State.unloaded, h.state());
    h.fail_open = false;
    var problem: []const u8 = "";
    try h.reload(&problem);
    try t.expectEqual(idle.State.resident, h.state());
}

//! A long cold prompt beside a decoding stream and a warm arrival (XPU_PROMPT_SLICE, lane_host.zig `slice_long`):
//! sliced, the decoder gets a round after every slice and the warm request goes in before the long pass ends;
//! whole (the flag off), the long pass runs through and the others wait for it, as before.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

const cold_len = 1200; // more new rows than Share.short (512): a long pass
const slices = 8;

const Seen = struct {
    mutex: std.Io.Mutex = .init,
    decoded: usize = 0, // the decoding stream's tokens delivered so far
    at_slice: [slices]usize = @splat(0), // and when each of the cold pass's slices began
    slices_seen: usize = 0,
    prefilled: [3]Id = @splat(0),
    count: usize = 0,
    done: usize = 0,

    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const o: *Seen = @ptrCast(@alignCast(ctx));
        o.mutex.lockUncancelable(std.testing.io);
        defer o.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| if (id == 1) {
                o.decoded += t.len;
            },
            .prefilled => {
                o.prefilled[o.count] = id;
                o.count += 1;
            },
            .finished => o.done += 1,
            else => {},
        }
    }

    fn read(o: *Seen, comptime field: []const u8) usize {
        o.mutex.lockUncancelable(std.testing.io);
        defer o.mutex.unlock(std.testing.io);
        return @field(o, field);
    }
};

/// The cold pass's slices: each notes the decoder's tokens, and the third sends the warm request.
const Arrive = struct {
    engine: api.Engine,
    warm: *const Request,
    seen: *Seen,
    sent: bool = false,

    fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const a: *Arrive = @ptrCast(@alignCast(ctx));
        if (s.prompt_len != cold_len) return;
        a.seen.mutex.lockUncancelable(std.testing.io);
        if (chunk < slices) a.seen.at_slice[chunk] = a.seen.decoded;
        a.seen.slices_seen = @max(a.seen.slices_seen, chunk + 1);
        a.seen.mutex.unlock(std.testing.io);
        if (chunk == 2 and !a.sent) {
            a.sent = true;
            a.engine.submit(3, a.warm, .{ .ctx = a.seen, .event = Seen.event }) catch {};
        }
    }
};

fn wait(seen: *Seen, comptime field: []const u8, at_least: usize) void {
    while (seen.read(field) < at_least) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
}

/// One decoding stream, then a cold long prompt, then a warm short one during the cold pass's third slice.
fn scene(sliced: bool, seen: *Seen) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = slices };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    // the flag off is today's qwen27 backend: a whole `prefill`, no `prefill_step`
    var core = lanes.Engine.init(gpa, &cfg, if (sliced) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    host.slice_long = sliced;
    var cold_ids: [cold_len]u32 = undefined;
    for (&cold_ids, 0..) |*t, i| t.* = @intCast(20 + i % 50);
    const decoder: Request = .{ .prompt = &.{ 2, 7, 1, 8 }, .max_tokens = 4000 };
    const cold: Request = .{ .prompt = &cold_ids, .max_tokens = 4 };
    const warm: Request = .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_tokens = 4 };
    const e = host.engine();
    var arrive: Arrive = .{ .engine = e, .warm = &warm, .seen = seen };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &decoder, .{ .ctx = seen, .event = Seen.event });
    try host.start();
    defer host.stop();
    wait(seen, "decoded", 1); // the decoder is in its rounds before the cold prompt comes
    try e.submit(2, &cold, .{ .ctx = seen, .event = Seen.event });
    wait(seen, "count", 3);
    e.cancel(1);
    wait(seen, "done", 3);
}

test "prompt slice: the decoder gets a round after every slice and a warm arrival goes in before the long pass ends" {
    var seen: Seen = .{};
    try scene(true, &seen);
    try std.testing.expectEqual(@as(usize, slices), seen.slices_seen);
    for (1..slices) |i| try std.testing.expect(seen.at_slice[i] > seen.at_slice[i - 1]);
    try std.testing.expectEqualSlices(Id, &.{ 1, 3, 2 }, &seen.prefilled);
}

test "prompt slice off: the long pass runs whole, no round and no admission inside it, as before" {
    var seen: Seen = .{};
    try scene(false, &seen);
    try std.testing.expectEqual(@as(usize, slices), seen.slices_seen);
    for (1..slices) |i| try std.testing.expectEqual(seen.at_slice[0], seen.at_slice[i]);
    try std.testing.expectEqualSlices(Id, &.{ 1, 2, 3 }, &seen.prefilled);
}

//! jubiku 2026-11-07 05:43Z: one long conversation (122k tokens, a turn of 239 more) beside short judge calls
//! (3.3k tokens each, each its own conversation), on the XPU store's numbers: a state's bytes are 333 MiB plus
//! 32.5 KiB a token, the budget 7765 MiB, at most 4 states, chunk starts on a 64-token grid. The lane host's
//! order: lookup, the restore reported, then the keeps at the pass's marks.
//! The store keeps the state: the misses on jubiku came from verifyMulti (runner_multi.zig), which made room for
//! the slot's last stream's length on a judge call's first round and dropped the long state "for streams".
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const t = std.testing;

const MiB: u64 = 1 << 20;

const Family = struct {
    live: usize = 0,
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 333 * MiB + @as(u64, at) * 33280;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *Family = @ptrCast(@alignCast(ptr));
        const st = try t.allocator.create(u32);
        st.* = at;
        f.live += 1;
        return st;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *Family = @ptrCast(@alignCast(ptr));
        t.allocator.destroy(@as(*u32, @ptrCast(@alignCast(saved))));
        f.live -= 1;
    }
};

const GEN = 5; // the generation prompt's tokens after the history

/// One request through the store as the lane host runs it; returns where its pass resumed.
fn turn(s: *Store, prompt: []const u32, system_len: u32, rewind_len: u32) !u32 {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const history: u32 = @intCast(prompt.len - GEN);
    var shared: std.ArrayList(u32) = .empty;
    for ([_]i64{ @as(i64, system_len) - 2048, @as(i64, system_len) - 512, system_len }) |cut| if (cut >= 512) try shared.append(a, @intCast(cut));
    const l = try s.lookupUser(a, prompt, @intCast(prompt.len), history, rewind_len, 0, shared.items, &.{}, &.{});
    const from: u32 = if (l.entry) |e| e.at else 0;
    if (l.entry) |e| s.resumedWithReason(e, prompt, null);
    for (l.marks) |m| if (m > from) {
        _ = s.keepBorn(prompt, @intCast(prompt.len), m, null, &.{}, &.{});
    };
    return from;
}

test "long turn: a 122k conversation keeps resuming past half the pool while short calls come between its turns" {
    var f: Family = .{};
    var s = Store.init(t.allocator, f.snapshots(), .{ .planned = true, .grid = 64 }, 7765 * MiB);
    s.most = 4;
    defer s.deinit();
    // the long conversation: a 9180-token system prefix, its last user message at 64330, then tool turns
    var conv: std.ArrayList(u32) = .empty;
    defer conv.deinit(t.allocator);
    var next: u32 = 1;
    while (conv.items.len < 122170 - GEN) : (next += 1) try conv.append(t.allocator, next % 50000 + 10);
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(t.allocator);
    var judge: u32 = 0;
    var resumed: [8]u32 = undefined;
    for (&resumed, 0..) |*r, k| {
        prompt.clearRetainingCapacity();
        try prompt.appendSlice(t.allocator, conv.items);
        try prompt.appendSlice(t.allocator, &.{ 1, 2, 3, 4, 5 });
        r.* = try turn(&s, prompt.items, 9180, 64330);
        // a judge call between the turns: its own 3.3k-token conversation
        var small: [3364]u32 = undefined;
        judge += 1;
        for (&small, 0..) |*x, i| x.* = 60000 + judge * 4000 + @as(u32, @intCast(i));
        _ = try turn(&s, &small, 0, 0);
        // the turn's reply and its tool result: 239 more tokens of history
        for (0..239) |_| {
            try conv.append(t.allocator, next % 50000 + 10);
            next += 1;
        }
        if (k > 0) try t.expect(r.* > 120000); // every turn after the first resumes near its end
    }
}

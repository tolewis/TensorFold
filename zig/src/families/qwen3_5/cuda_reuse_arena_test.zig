//! A kept state saved into gaps of the arena and restored from them, through a mock driver on host memory.
const std = @import("std");
const cuda = @import("cuda");
const reuse = @import("cuda_reuse.zig");
const hp = @import("cuda_host_pool.zig");
const engine = @import("cuda_engine.zig");
const t = std.testing;

/// Device pointers are host addresses here: a copy moves the bytes, and pinning is counted.
const Mock = struct {
    var pinned: usize = 0;
    var runs: usize = 0;

    fn dtoh(dst: ?*anyopaque, src: cuda.abi.DevicePtr, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        runs += 1;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return cuda.abi.success;
    }
    fn htod(dst: cuda.abi.DevicePtr, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        runs += 1;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return cuda.abi.success;
    }
    fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        pinned += 1;
        return cuda.abi.error_not_ready;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

test "Qwen reuse arena: a state no free range holds is kept in the gaps, restored byte for byte, and given back" {
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyDtoHAsync_v2 = Mock.dtoh;
    api.cuMemcpyHtoDAsync_v2 = Mock.htod;
    api.cuMemHostAlloc = Mock.hostAlloc;
    api.cuStreamSynchronize = Mock.sync;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: engine.Engine = undefined;
    e.kit = null;
    e.gpa = t.allocator;
    e.d = &d;
    e.g = undefined;
    e.g.hidden = 5120;
    e.g.k_heads = 16;
    e.g.v_heads = 16;
    e.g.q_heads = 32;
    e.g.kv_heads = 4;
    e.g.layers = 8;
    @memset(&e.g.linear, false);
    for (0..8) |i| e.g.linear[i] = i % 4 != 3;
    e.stream = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |kv8| {
        e.kv8 = kv8;
        const l: reuse.Layout = .{ .g = e.g, .format = reuse.formatOf(&e), .cap = 64, .at = 40 };
        const len: usize = @intCast(l.total());
        const want = (len + hp.unit - 1) / hp.unit; // the units the state takes
        // an arena of 2 * want + 2 units, every other unit held: no two free units touch
        const units = 2 * want + 2;
        const arena = try t.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(hp.unit), units * hp.unit);
        defer t.allocator.free(arena);
        const pool = try hp.Pool.over(t.allocator, arena);
        defer pool.deinit();
        e.pool = pool;
        e.pool_budget = units * hp.unit;
        var held: [64][]align(16) u8 = undefined;
        for (held[0..units]) |*h| h.* = pool.take(hp.unit).?;
        for (held[0..units], 0..) |h, i| if (i % 2 == 0) pool.give(h);
        try t.expect(pool.take(2 * hp.unit) == null);

        const state = try t.allocator.alloc(u8, e.g.stateBytes());
        defer t.allocator.free(state);
        const kv = try t.allocator.alloc(u8, l.kvBytes());
        defer t.allocator.free(kv);
        for (state, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
        for (kv, 0..) |*b, i| b.* = @truncate(i *% 197 +% 3);
        const state0 = try t.allocator.dupe(u8, state);
        defer t.allocator.free(state0);
        const kv0 = try t.allocator.dupe(u8, kv);
        defer t.allocator.free(kv0);
        var cache: engine.Cache = undefined;
        cache.cap = 64;
        cache.len = 40;
        cache.last = null;
        cache.pend = 0;
        cache.delta = 3;
        cache.paged = null;
        cache.kv = .{ .d = &d, .ptr = @intFromPtr(kv.ptr), .len = kv.len };
        cache.state = .{ .d = &d, .ptr = @intFromPtr(state.ptr), .len = state.len };

        Mock.pinned = 0;
        Mock.runs = 0;
        const copy = try reuse.saveCache(&e, &cache, 40, &.{}, false);
        try t.expect(copy.pooled);
        try t.expectEqual(want, copy.parts.n);
        try t.expectEqual(@as(usize, 0), Mock.pinned);
        try t.expect(Mock.runs > l.count()); // runs split where the ranges end
        try t.expectEqual(len, copy.room());

        @memset(state, 0);
        @memset(kv, 0);
        try reuse.restoreCache(&e, &cache, copy, &.{});
        try t.expectEqualSlices(u8, state0, state);
        for (1..l.count()) |i| {
            const p = l.piece(i);
            try t.expectEqualSlices(u8, kv0[p.off..][0..p.len], kv[p.off..][0..p.len]);
        }
        reuse.dropCopy(&e, copy);
        for (held[0..units], 0..) |h, i| if (i % 2 == 1) pool.give(h);
        try t.expectEqual(@as(usize, 1), pool.ranges.items.len);
        try t.expectEqual(units * hp.unit, pool.ranges.items[0].len);
    }
}

/// Each call the mock driver sees, in order, with its stream and event, and for a copy whether it moves piece 0.
const Log = struct {
    const Kind = enum { dtoh, htod, record, wait, sync };
    const Op = struct { kind: Kind, stream: usize, event: usize = 0, state: bool = false };
    var ops: [4096]Op = undefined;
    var n: usize = 0;
    var state_lo: usize = 0;
    var state_hi: usize = 0;

    fn add(op: Op) void {
        ops[n] = op;
        n += 1;
    }
    fn dtoh(dst: ?*anyopaque, src: cuda.abi.DevicePtr, len: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        add(.{ .kind = .dtoh, .stream = @intFromPtr(s), .state = src >= state_lo and src < state_hi });
        return Mock.dtoh(dst, src, len, s);
    }
    fn htod(dst: cuda.abi.DevicePtr, src: ?*const anyopaque, len: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        add(.{ .kind = .htod, .stream = @intFromPtr(s) });
        return Mock.htod(dst, src, len, s);
    }
    fn record(ev: cuda.abi.Event, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        add(.{ .kind = .record, .stream = @intFromPtr(s), .event = @intFromPtr(ev) });
        return cuda.abi.success;
    }
    fn wait(s: cuda.abi.Stream, ev: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        add(.{ .kind = .wait, .stream = @intFromPtr(s), .event = @intFromPtr(ev) });
        return cuda.abi.success;
    }
    fn sync(s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        add(.{ .kind = .sync, .stream = @intFromPtr(s) });
        return cuda.abi.success;
    }
    fn find(from: usize, op: Op) ?usize {
        for (ops[from..n], from..) |o, i| if (o.kind == op.kind and o.stream == op.stream and o.event == op.event) return i;
        return null;
    }
};

test "Qwen reuse arena: a save copies on the copy stream after the mark, and the next pass waits only for the state" {
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyDtoHAsync_v2 = Log.dtoh;
    api.cuMemcpyHtoDAsync_v2 = Log.htod;
    api.cuMemHostAlloc = Mock.hostAlloc;
    api.cuStreamSynchronize = Log.sync;
    api.cuEventRecord = Log.record;
    api.cuStreamWaitEvent = Log.wait;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: engine.Engine = undefined;
    e.kit = null;
    e.gpa = t.allocator;
    e.d = &d;
    e.g = undefined;
    e.g.hidden = 5120;
    e.g.k_heads = 16;
    e.g.v_heads = 16;
    e.g.q_heads = 32;
    e.g.kv_heads = 4;
    e.g.layers = 8;
    @memset(&e.g.linear, false);
    for (0..8) |i| e.g.linear[i] = i % 4 != 3;
    const main: usize = 0x10;
    const side: usize = 0x20;
    const ready: usize, const state_ev: usize, const done: usize = .{ 0x31, 0x32, 0x33 };
    e.stream = .{ .d = &d, .handle = @ptrFromInt(main) };
    for ([_]bool{ false, true }) |kv8| {
        e.kv8 = kv8;
        const l: reuse.Layout = .{ .g = e.g, .format = reuse.formatOf(&e), .cap = 64, .at = 40 };
        const units = (@as(usize, @intCast(l.total())) + hp.unit - 1) / hp.unit;
        const arena = try t.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(hp.unit), units * hp.unit);
        defer t.allocator.free(arena);
        const pool = try hp.Pool.over(t.allocator, arena);
        defer pool.deinit();
        defer pool.side = null; // the mock has no stream or event to destroy
        pool.side = .{
            .stream = .{ .d = &d, .handle = @ptrFromInt(side) },
            .ready = .{ .d = &d, .handle = @ptrFromInt(ready) },
            .state = .{ .d = &d, .handle = @ptrFromInt(state_ev) },
            .done = .{ .d = &d, .handle = @ptrFromInt(done) },
        };
        e.pool = pool;
        e.pool_budget = units * hp.unit;

        const state = try t.allocator.alloc(u8, e.g.stateBytes());
        defer t.allocator.free(state);
        const kv = try t.allocator.alloc(u8, l.kvBytes());
        defer t.allocator.free(kv);
        for (state, 0..) |*b, i| b.* = @truncate(i *% 61 +% 5);
        for (kv, 0..) |*b, i| b.* = @truncate(i *% 89 +% 1);
        const state0 = try t.allocator.dupe(u8, state);
        defer t.allocator.free(state0);
        const kv0 = try t.allocator.dupe(u8, kv);
        defer t.allocator.free(kv0);
        Log.state_lo = @intFromPtr(state.ptr);
        Log.state_hi = Log.state_lo + state.len;
        var cache: engine.Cache = undefined;
        cache.cap = 64;
        cache.len = 40;
        cache.last = null;
        cache.pend = 0;
        cache.delta = 1;
        cache.paged = null;
        cache.kv = .{ .d = &d, .ptr = @intFromPtr(kv.ptr), .len = kv.len };
        cache.state = .{ .d = &d, .ptr = @intFromPtr(state.ptr), .len = state.len };

        Log.n = 0;
        const copy = try reuse.saveCache(&e, &cache, 40, &.{}, false);
        try t.expect(copy.saving and copy.pooled);
        // the copy stream starts at the mark; the engine's stream waits for piece 0 and nothing after it
        try t.expectEqual(@as(?usize, 0), Log.find(0, .{ .kind = .record, .stream = main, .event = ready }));
        try t.expectEqual(@as(?usize, 1), Log.find(0, .{ .kind = .wait, .stream = side, .event = ready }));
        const handed = Log.find(0, .{ .kind = .record, .stream = side, .event = state_ev }).?;
        const waited = Log.find(0, .{ .kind = .wait, .stream = main, .event = state_ev }).?;
        try t.expect(handed < waited);
        var copies: usize = 0;
        for (Log.ops[0..Log.n], 0..) |o, i| if (o.kind == .dtoh) {
            copies += 1;
            try t.expectEqual(side, o.stream);
            try t.expect(if (o.state) i < handed else i > waited);
        };
        try t.expect(copies >= l.count());
        try t.expectEqual(Log.n - 1, Log.find(0, .{ .kind = .record, .stream = side, .event = done }).?);
        for (Log.ops[0..Log.n]) |o| try t.expect(o.kind != .sync); // the host does not wait for the copy

        // a restore's engine stream waits for the queued save before it reads the bytes
        Log.n = 0;
        @memset(state, 0);
        @memset(kv, 0);
        try reuse.settled(&e);
        try t.expectEqual(@as(?usize, 0), Log.find(0, .{ .kind = .wait, .stream = main, .event = done }));
        try reuse.restoreCache(&e, &cache, copy, &.{});
        try t.expectEqual(@as(?usize, 1), Log.find(1, .{ .kind = .wait, .stream = main, .event = done }));
        for (Log.ops[2..Log.n]) |o| if (o.kind == .htod) try t.expectEqual(main, o.stream);
        try t.expectEqualSlices(u8, state0, state);
        for (1..l.count()) |i| {
            const p = l.piece(i);
            try t.expectEqualSlices(u8, kv0[p.off..][0..p.len], kv[p.off..][0..p.len]);
        }

        // a drop waits on the host for the copy stream before the ranges go back to the arena
        Log.n = 0;
        reuse.dropCopy(&e, copy);
        try t.expect(Log.find(0, .{ .kind = .sync, .stream = side }) != null);
        try t.expectEqual(@as(usize, 1), pool.ranges.items.len);
        try t.expectEqual(units * hp.unit, pool.ranges.items[0].len);
    }
}

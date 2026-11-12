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

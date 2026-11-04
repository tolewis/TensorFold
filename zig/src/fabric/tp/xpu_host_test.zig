//! The XPU host collective against the reference at 2-4 ranks, with NaN, desync, a killed peer and planted faults.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const xpu = @import("xpu_host.zig");
const host = @import("host.zig");
const DType = collective.DType;

fn addr(b: []u8) collective.DevicePtr {
    return @intFromPtr(b.ptr);
}

const n = 96;

const Calls = struct {
    fn go(c: collective.Collective, rank: u32, _: void) !void {
        const world = c.world();
        try std.testing.expectEqual(collective.Kind.xpu, c.kind());
        inline for (.{ DType.f32, DType.bf16, DType.i32, DType.f16 }) |t| {
            const bytes = n * t.size();
            var mine: [n * 8]u8 = undefined;
            var out: [n * 8 * xpu.max_ranks]u8 = undefined;
            reference.fill(mine[0..bytes], t, rank, 7);
            // the reduction against every rank's input, in rank order
            var ins: [xpu.max_ranks][n * 8]u8 = undefined;
            var views: [xpu.max_ranks][]const u8 = undefined;
            for (0..world) |r| {
                reference.fill(ins[r][0..bytes], t, @intCast(r), 7);
                views[r] = ins[r][0..bytes];
            }
            var want: [n * 8]u8 = undefined;
            reference.reduce(want[0..bytes], views[0..world], t, .sum);
            try c.allReduce(addr(&mine), addr(&out), n, t, .sum, null);
            try std.testing.expectEqualSlices(u8, want[0..bytes], out[0..bytes]);
            // in place: the sum replaces this rank's input
            try c.allReduce(addr(&mine), addr(&mine), n, t, .sum, null);
            try std.testing.expectEqualSlices(u8, want[0..bytes], mine[0..bytes]);
            // all-gather: rank order, rank 0's bytes first
            reference.fill(mine[0..bytes], t, rank, 7);
            try c.allGather(addr(&mine), addr(&out), n, t, null);
            for (0..world) |r| try std.testing.expectEqualSlices(u8, views[r], out[r * bytes ..][0..bytes]);
        }
        try c.barrier();
    }
};

test "every call on two, three and four XPU ranks equals the reference" {
    try xpu.run(2, {}, Calls.go);
    try xpu.run(3, {}, Calls.go);
    try xpu.run(4, {}, Calls.go);
}

const Pairs = struct {
    fn go(c: collective.Collective, rank: u32, _: void) !void {
        var mine: [64]u8 = @splat(@intCast(rank + 1));
        var got: [64]u8 = undefined;
        // (0, 1), (2, 3), ...: the two ranks of a pair trade, at any even world
        const peer = rank ^ 1;
        try c.exchange(addr(&mine), addr(&got), 64, .u8, peer, null);
        try std.testing.expectEqualSlices(u8, &@as([64]u8, @splat(@intCast(peer + 1))), &got);
        if (rank == 0) {
            try c.send(addr(&mine), 16, .f32, 1, null);
        } else if (rank == 1) {
            var buf: [64]u8 = undefined;
            try c.recv(addr(&buf), 16, .f32, 0, null);
            try std.testing.expectEqualSlices(u8, &@as([64]u8, @splat(1)), &buf);
        }
        try std.testing.expectError(error.Invalid, c.exchange(addr(&mine), addr(&got), 64, .u8, rank, null));
    }
};

test "exchange, send and recv between two XPU ranks, and in pairs of four" {
    try xpu.run(2, {}, Pairs.go);
    try xpu.run(4, {}, Pairs.go);
}

const Nan = struct {
    fn go(c: collective.Collective, rank: u32, _: void) !void {
        const world = c.world();
        const bytes = n * 4;
        var mine: [n * 4]u8 = undefined;
        var out: [n * 4]u8 = undefined;
        reference.fill(mine[0..bytes], .f32, rank, 3);
        if (rank == 0) std.mem.writeInt(u32, mine[8..12], @bitCast(std.math.nan(f32)), .little);
        try c.allReduce(addr(&mine), addr(&out), n, .f32, .sum, null);
        // the reference adds the same partials in the same order, so the bits match and the NaN is still a NaN
        var ins: [xpu.max_ranks][n * 4]u8 = undefined;
        var views: [xpu.max_ranks][]const u8 = undefined;
        for (0..world) |r| {
            reference.fill(ins[r][0..bytes], .f32, @intCast(r), 3);
            if (r == 0) std.mem.writeInt(u32, ins[r][8..12], @bitCast(std.math.nan(f32)), .little);
            views[r] = ins[r][0..bytes];
        }
        var want: [n * 4]u8 = undefined;
        reference.reduce(want[0..bytes], views[0..world], .f32, .sum);
        try std.testing.expectEqualSlices(u8, want[0..bytes], out[0..bytes]);
        try std.testing.expect(std.math.isNan(@as(f32, @bitCast(std.mem.readInt(u32, out[8..12], .little)))));
        // a NaN in every rank's partial stays a NaN, and the ranks agree
        for (0..world) |r| std.mem.writeInt(u32, ins[r][8..12], @bitCast(std.math.nan(f32)), .little);
        reference.reduce(want[0..bytes], views[0..world], .f32, .sum);
        var out2: [n * 4]u8 = undefined;
        try c.allReduce(addr(&ins[rank]), addr(&out2), n, .f32, .sum, null);
        try std.testing.expectEqualSlices(u8, want[0..bytes], out2[0..bytes]);
        try std.testing.expect(std.math.isNan(@as(f32, @bitCast(std.mem.readInt(u32, out2[8..12], .little)))));
    }
};

test "a NaN partial stays a NaN on every XPU rank" {
    try xpu.run(2, {}, Nan.go);
    try xpu.run(3, {}, Nan.go);
}

const Inf = struct {
    fn go(c: collective.Collective, rank: u32, _: void) !void {
        const world = c.world();
        const bytes = n * 4;
        var mine: [n * 4]u8 = undefined;
        var out: [n * 4]u8 = undefined;
        reference.fill(mine[0..bytes], .f32, rank, 11);
        // one rank's +Inf and another rank's -Inf: each stays itself through the sum
        if (rank == 0) std.mem.writeInt(u32, mine[16..20], @bitCast(std.math.inf(f32)), .little);
        if (rank == 1) std.mem.writeInt(u32, mine[20..24], @bitCast(-std.math.inf(f32)), .little);
        try c.allReduce(addr(&mine), addr(&out), n, .f32, .sum, null);
        var ins: [xpu.max_ranks][n * 4]u8 = undefined;
        var views: [xpu.max_ranks][]const u8 = undefined;
        for (0..world) |r| {
            reference.fill(ins[r][0..bytes], .f32, @intCast(r), 11);
            if (r == 0) std.mem.writeInt(u32, ins[r][16..20], @bitCast(std.math.inf(f32)), .little);
            if (r == 1) std.mem.writeInt(u32, ins[r][20..24], @bitCast(-std.math.inf(f32)), .little);
            views[r] = ins[r][0..bytes];
        }
        var want: [n * 4]u8 = undefined;
        reference.reduce(want[0..bytes], views[0..world], .f32, .sum);
        try std.testing.expectEqualSlices(u8, want[0..bytes], out[0..bytes]);
        try std.testing.expect(std.math.isPositiveInf(@as(f32, @bitCast(std.mem.readInt(u32, out[16..20], .little)))));
        try std.testing.expect(std.math.isNegativeInf(@as(f32, @bitCast(std.mem.readInt(u32, out[20..24], .little)))));
    }
};

test "an Inf partial stays Inf with the reference's bits on every XPU rank" {
    try xpu.run(2, {}, Inf.go);
    try xpu.run(3, {}, Inf.go);
}

const Desync = struct {
    fn go(c: collective.Collective, rank: u32, _: void) !void {
        var a: [64]u8 = @splat(0);
        var b: [64]u8 = undefined;
        // the ranks disagree on the count: both see it instead of hanging
        const r = c.allReduce(addr(&a), addr(&b), if (rank == 0) 8 else 16, .f32, .sum, null);
        if (r) |_| return error.ShouldHaveFailed else |e| if (e != error.Invalid and e != error.Aborted) return e;
    }
};

test "a desynchronised call is refused on every XPU rank" {
    try xpu.run(2, {}, Desync.go);
    try xpu.run(4, {}, Desync.go);
}

test "a killed peer is PeerFailed at once, not a timeout" {
    var g = xpu.init(2);
    var r0 = g.rank(0);
    xpu.kill(&g, 1); // rank 1's process is gone: it never calls
    var a: [64]u8 = @splat(0);
    var b: [64]u8 = undefined;
    const t0 = host.now();
    try std.testing.expectError(error.PeerFailed, r0.iface().allReduce(addr(&a), addr(&b), 16, .f32, .sum, null));
    try std.testing.expect((host.now() - t0) / std.time.ns_per_ms < 1000); // the 10 s timeout was never reached
    // check() names a dead peer too, on a group that has not aborted yet
    var g2 = xpu.init(2);
    xpu.kill(&g2, 1);
    var r1 = g2.rank(0);
    try std.testing.expectError(error.PeerFailed, r1.iface().check());
}

const Slow = struct {
    fn go(c: collective.Collective, rank: u32, sleep_ms: u64) !void {
        if (rank == 1) {
            // alive, but it never calls: rank 0 must give up on its own deadline
            @import("sock.zig").sleepNs(sleep_ms * std.time.ns_per_ms);
            return;
        }
        var a: [64]u8 = @splat(0);
        var b: [64]u8 = undefined;
        try std.testing.expectError(error.Timeout, c.allReduce(addr(&a), addr(&b), 16, .f32, .sum, null));
    }
};

const BarrierSlow = struct {
    fn go(c: collective.Collective, rank: u32, sleep_ms: u64) !void {
        if (rank == 1) {
            @import("sock.zig").sleepNs(sleep_ms * std.time.ns_per_ms);
            return;
        }
        const t0 = host.now();
        try std.testing.expectError(error.Timeout, c.barrier());
        const ms = (host.now() - t0) / std.time.ns_per_ms;
        try std.testing.expect(ms >= 20 and ms < 500); // its own 40 ms deadline, not the 10 s default
    }
};

test "a barrier with a silent peer times out at its deadline" {
    try xpu.runWith(2, .{ .timeout_ns = 40 * std.time.ns_per_ms, .peer_deadline_ns = 5 * std.time.ns_per_s }, @as(u64, 120), BarrierSlow.go);
}

test "a silent but alive peer is Timeout" {
    try xpu.runWith(2, .{ .timeout_ns = 40 * std.time.ns_per_ms, .peer_deadline_ns = 5 * std.time.ns_per_s }, @as(u64, 120), Slow.go);
}

const Ctl = struct {
    fn go(c: collective.Collective, rank: u32, _: void) !void {
        const world = c.world();
        // 1.0, 2^-24, -1.0: 1.0 + 2^-24 rounds back to 1.0, but 2^-24 - 1.0 + 1.0 keeps the 2^-24, so the order matters
        var mine: [4]u8 = undefined;
        std.mem.writeInt(u32, &mine, @bitCast(value(rank)), .little);
        var out: [4]u8 = undefined;
        try c.allReduce(addr(&mine), addr(&out), 1, .f32, .sum, null);
        var ins: [xpu.max_ranks][4]u8 = undefined;
        var views: [xpu.max_ranks][]const u8 = undefined;
        for (0..world) |r| {
            std.mem.writeInt(u32, &ins[r], @bitCast(value(@intCast(r))), .little);
            views[r] = &ins[r];
        }
        var want: [4]u8 = undefined;
        reference.reduce(&want, views[0..world], .f32, .sum);
        try std.testing.expect(!std.mem.eql(u8, &want, &out)); // the planted fault must not match the reference
    }

    fn value(r: u32) f32 {
        return switch (r) {
            0 => 1.0,
            1 => 5.960464477539063e-8, // 2^-24, half an ulp of 1.0
            else => -1.0,
        };
    }
};

test "planted faults fail the comparison" {
    for ([_]xpu.Control{ .reverse, .drop_last, .double_first }) |control|
        try xpu.runWith(3, .{ .control = control }, {}, Ctl.go);
}

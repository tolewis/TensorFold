//! LaneHost robustness (round 3, docs/port/robustness.md): restores the tier refuses for capacity (deferred and
//! asked again, bounded; the shorter memory state kept meanwhile), and a broken rank pair (nothing admitted, waiting
//! work failed, streams drained or failed, the collectives stopped once, one exit with the policy's status, also
//! when the engine thread is stuck on the peer), through the fake lane backend, a fake tier and fake abort/exit hooks.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const lh = @import("lane_host.zig");
const tt = @import("lane_host_tier_test.zig");
const LaneHost = lh.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const Rig = tt.Rig;
const Box = tt.Box;
const t2 = tt.t2;
const gpa = std.testing.allocator;
const io = std.testing.io;

// -- restores refused for capacity -------------------------------------------------------------------------------

test "a restore the tier refuses for capacity is asked again shortly, then resumes exactly" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.refuse = 2; // two refusals (TooManyRestores: restores in flight), then the begin goes through
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred); // once a request, however often it asked
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 2), r.tier.refused);
    try std.testing.expectEqual(@as(u64, 3), r.tier.begins);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
}

test "a full tier refuses before any room is made: the request asks again, the refusal counted once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    try shortThenPoint(r);
    r.tier.full = 3; // the restores in flight at the tier's cap for three asks
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_refused); // once a request, not once an ask
    try std.testing.expectEqual(@as(u64, 1), r.tier.begins); // never asked to begin while full
    try std.testing.expectEqual(@as(u64, 0), s.evicted);
}

/// A memory state of t2 at 4 (a short turn's keep), then t2's durable point at 8.
fn shortThenPoint(r: *Rig) !void {
    var a: Box = .{};
    defer a.deinit();
    const short: Request = .{ .prompt = t2[0..6], .max_tokens = 1, .history_len = 4 };
    try r.host.engine().submit(1, &short, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    _ = try r.point(&t2, 8);
}

test "a restore still refused past its deferral bound resumes the shorter memory state, never a prefill from 0" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.host.tier.defer_ns = 40 * std.time.ns_per_ms;
    try shortThenPoint(r);
    r.tier.refuse = 1 << 20; // at capacity throughout
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 4), b.cached); // the memory state, kept while the restore was refused
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred_expired);
    try std.testing.expect(r.tier.refused >= 2); // asked again while deferred
    try std.testing.expectEqual(@as(u64, 0), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
}

test "with the deferral off a refused restore resumes what memory holds at once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.host.tier.defer_ns = 0;
    try shortThenPoint(r);
    r.tier.refuse = 1 << 20;
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 4), b.cached);
    try b.exact(&t2);
    try std.testing.expectEqual(@as(u64, 0), r.stats().restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), r.tier.refused);
}

test "a refusal that waiting cannot cure is not deferred; a deferred request cancelled leaves every list once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.refuse = 1;
    r.tier.refuse_err = error.PointMissing; // not a capacity refusal: admitted at once, prefilled
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 0), a.cached);
    try std.testing.expectEqual(@as(u64, 0), r.stats().restores_deferred);
    // at capacity for long: the request waits deferred (no lane, counted waiting), and its cancel ends it there
    r.host.tier.defer_ns = 10 * std.time.ns_per_s;
    r.tier.refuse = 1 << 20;
    r.tier.refuse_err = error.TooManyRestores;
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    var b: Box = .{};
    defer b.deinit();
    const req_b: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req_b, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1 and s.parked == 1;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 1), st.waiting);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.parked == 0;
        }
    }.ok);
    try std.testing.expectEqual(@as(usize, 0), r.tier.jobs.items.len); // nothing was ever begun for it
}

test "shutdown with a deferred request cancels it once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.host.tier.defer_ns = 10 * std.time.ns_per_s;
    r.tier.refuse = 1 << 20;
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), b.finished());
}

// -- a broken rank pair --------------------------------------------------------------------------------------------

/// The fake backend with failures on demand: a round's verify (and one-token rounds) or a prompt pass fails with
/// `fail`, or a round blocks until the pair's abort (a round stuck on a dead peer).
const Flaky = struct {
    inner: lanes.backend.Backend,
    rounds_fail: std.atomic.Value(bool) = .init(false),
    prefill_fail: ?anyerror = null,
    stuck: std.atomic.Value(bool) = .init(false),
    /// A stuck round returns as if it succeeded once released (an RDMA collect kernel that gave up on its abort
    /// word: the graph completes with rows that are not the model's).
    stuck_ok: bool = false,
    entered: std.atomic.Value(bool) = .init(false), // a round is stuck now
    released: *std.atomic.Value(bool), // the pair's abort ran
    fail: anyerror = error.PairBroken,

    fn backend(f: *Flaky) lanes.backend.Backend {
        return .{ .ptr = f, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }
    fn of(ptr: *anyopaque) *Flaky {
        return @ptrCast(@alignCast(ptr));
    }
    fn round(f: *Flaky) !void {
        if (f.stuck.load(.acquire)) {
            f.entered.store(true, .release);
            while (!f.released.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            if (f.stuck_ok) return;
            return error.CommAborted;
        }
        if (f.rounds_fail.load(.acquire)) return f.fail;
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const f = of(ptr);
        if (f.prefill_fail) |e| return e;
        return f.inner.vtable.prefill(f.inner.ptr, s);
    }
    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const f = of(ptr);
        return f.inner.vtable.first(f.inner.ptr, s, position);
    }
    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
        const f = of(ptr);
        try f.round();
        return f.inner.vtable.queue(f.inner.ptr, s, feed, position);
    }
    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const f = of(ptr);
        return f.inner.vtable.read(f.inner.ptr, handle);
    }
    fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const f = of(ptr);
        try f.round();
        return f.inner.vtable.verify(f.inner.ptr, windows, out);
    }
    fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
        const f = of(ptr);
        return f.inner.vtable.keep(f.inner.ptr, windows, paths);
    }
    fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
        const f = of(ptr);
        return f.inner.vtable.draft(f.inner.ptr, requests);
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const f = of(ptr);
        f.inner.vtable.release(f.inner.ptr, s);
    }
};

/// The policy's abort and exit, recorded (the real exit never returns).
const Ends = struct {
    aborts: std.atomic.Value(u32) = .init(0),
    exits: std.atomic.Value(u32) = .init(0),
    code: std.atomic.Value(u8) = .init(0),
    released: std.atomic.Value(bool) = .init(false),

    fn abort(ctx: *anyopaque) void {
        const e: *Ends = @ptrCast(@alignCast(ctx));
        _ = e.aborts.fetchAdd(1, .acq_rel);
        e.released.store(true, .release);
    }
    fn exit(ctx: *anyopaque, code: u8) void {
        const e: *Ends = @ptrCast(@alignCast(ctx));
        e.code.store(code, .release);
        _ = e.exits.fetchAdd(1, .acq_rel);
    }
    fn waitExit(e: *Ends) !void {
        for (0..5000) |_| {
            if (e.exits.load(.acquire) > 0) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// A request's reply with its final reason and message.
const Reply = struct {
    mutex: std.Io.Mutex = .init,
    tokens: usize = 0,
    done: ?Reason = null,
    message: [256]u8 = undefined,
    message_len: usize = 0,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Reply = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| {
                b.done = f.reason;
                const n = @min(f.message.len, b.message.len);
                @memcpy(b.message[0..n], f.message[0..n]);
                b.message_len = n;
            },
            else => {},
        }
    }
    fn sink(b: *Reply) api.Sink {
        return .{ .ctx = b, .event = event };
    }
    fn read(b: *Reply) struct { tokens: usize, done: ?Reason } {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return .{ .tokens = b.tokens, .done = b.done };
    }
    fn wait(b: *Reply) Reason {
        while (true) {
            if (b.read().done) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    fn says(b: *Reply, text: []const u8) bool {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return std.mem.indexOf(u8, b.message[0..b.message_len], text) != null;
    }
    fn decoding(b: *Reply) void {
        while (b.read().tokens == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
};

/// The fake backend behind Flaky, the lane core and a host with one lane, its pair policy set before start.
const Pair = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    flaky: Flaky,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,
    ends: Ends = .{},
    stopped: bool = false,

    const fatal = [_]LaneHost.Fatal{ .{ .err = error.PairBroken, .drain = false }, .{ .err = error.TransferTimeout, .drain = false } };

    fn create(policy: bool, drain_ns: i96, deadline_ns: i96, lanes_n: u32) !*Pair {
        return createWith(policy, drain_ns, deadline_ns, lanes_n, 0);
    }
    fn createWith(policy: bool, drain_ns: i96, deadline_ns: i96, lanes_n: u32, stall_ns: i96) !*Pair {
        const p = try gpa.create(Pair);
        errdefer gpa.destroy(p);
        p.* = .{ .cfg = undefined, .target = .{ .gpa = gpa }, .flaky = undefined, .core = undefined, .host = undefined };
        p.flaky = .{ .inner = p.target.backend(), .released = &p.ends.released };
        p.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        p.core = lanes.Engine.init(gpa, &p.cfg, p.flaky.backend(), p.clock.clock());
        p.host = LaneHost.init(gpa, io, &p.core, .{ .lanes = lanes_n });
        p.host.interleave_ns = 0;
        if (policy) p.host.pair = .{ .ctx = &p.ends, .abort = Ends.abort, .exit = Ends.exit, .drain_ns = drain_ns, .deadline_ns = deadline_ns, .grace_ns = std.time.ns_per_ms, .stall_ns = stall_ns, .fatal = &fatal };
        try p.host.start();
        return p;
    }
    fn stop(p: *Pair) void {
        if (p.stopped) return;
        p.stopped = true;
        p.host.stop();
    }
    fn destroy(p: *Pair) void {
        p.stop();
        p.core.deinit();
        p.target.deinit();
        p.cfg.deinit(gpa);
        gpa.destroy(p);
    }
};

const prompt_a = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const prompt_b = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };

test "a broken pair: what waits fails at once, a decoding stream drains, then one abort and one exit with the policy's status" {
    const p = try Pair.create(true, 30 * std.time.ns_per_ms, 10 * std.time.ns_per_s, 1);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const long: Request = .{ .prompt = &prompt_a, .max_tokens = 1 << 30 };
    try e.submit(1, &long, a.sink());
    a.decoding();
    var c: Reply = .{};
    const queued: Request = .{ .prompt = &prompt_b, .max_tokens = 4 };
    try e.submit(2, &queued, c.sink()); // (one lane: it waits behind A)
    p.host.pairBroken("rank 1 stopped answering (test)", true);
    try std.testing.expect(p.host.isBroken());
    try std.testing.expectEqual(Reason.failed, c.wait()); // failed right away, A still decoding
    try std.testing.expect(c.says("rank pair broke (rank 1 stopped answering (test))"));
    try std.testing.expectEqual(@as(?Reason, null), a.read().done);
    try std.testing.expectError(error.Closed, e.submit(3, &queued, c.sink())); // nothing new is admitted
    // A cannot finish in 30 ms: it fails at the drain's end, then the collectives stop and the process exits, once
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expect(a.says("the server restarts"));
    try p.ends.waitExit();
    try std.testing.expectEqual(LaneHost.pair_exit_code, p.ends.code.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
    p.host.pairBroken("a second cause", false); // the first cause only
    p.stop();
    try std.testing.expectEqual(@as(u32, 1), p.ends.exits.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
    try std.testing.expectEqualStrings("rank 1 stopped answering (test)", p.host.pair_why[0..p.host.pair_why_len]);
    try std.testing.expectEqual(@as(u64, 2), p.host.pair_failed);
}

test "a broken pair with room to drain lets a short stream finish normally before the exit" {
    const p = try Pair.create(true, 5 * std.time.ns_per_s, 10 * std.time.ns_per_s, 2);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const short: Request = .{ .prompt = &prompt_a, .max_tokens = 2000 };
    try e.submit(1, &short, a.sink());
    a.decoding();
    p.host.pairBroken("a spill command was refused (test)", true);
    try std.testing.expectEqual(Reason.length, a.wait()); // finished within the drain (or before the break)
    try std.testing.expectEqual(@as(usize, 2000), a.read().tokens);
    try p.ends.waitExit();
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), p.host.pair_failed);
}

test "a round failing with a fatal error breaks the pair: every stream fails at once, no drain" {
    const p = try Pair.create(true, 10 * std.time.ns_per_s, 10 * std.time.ns_per_s, 2);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const long: Request = .{ .prompt = &prompt_a, .max_tokens = 1 << 30 };
    try e.submit(1, &long, a.sink());
    a.decoding();
    p.flaky.rounds_fail.store(true, .release); // the next round: error.PairBroken (drain = false in the policy)
    try std.testing.expectEqual(Reason.failed, a.wait());
    try p.ends.waitExit(); // well before the 10 s drain
    try std.testing.expect(p.host.isBroken());
    try std.testing.expect(std.mem.startsWith(u8, p.host.pair_why[0..p.host.pair_why_len], "PairBroken"));
    try std.testing.expect(!p.host.pair_drain);
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
}

test "a prompt pass failing with a fatal error breaks the pair; another error fails its request alone" {
    const p = try Pair.create(true, 10 * std.time.ns_per_s, 10 * std.time.ns_per_s, 2);
    defer p.destroy();
    const e = p.host.engine();
    p.flaky.prefill_fail = error.SomethingElse;
    var a: Reply = .{};
    const req: Request = .{ .prompt = &prompt_a, .max_tokens = 4 };
    try e.submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expect(!p.host.isBroken()); // not the policy's: the host serves on
    p.flaky.prefill_fail = null;
    var b: Reply = .{};
    try e.submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    p.flaky.prefill_fail = error.TransferTimeout; // a KV transfer that did not complete: the stream is wedged
    var c: Reply = .{};
    try e.submit(3, &req, c.sink());
    try std.testing.expectEqual(Reason.failed, c.wait());
    try p.ends.waitExit();
    try std.testing.expect(std.mem.startsWith(u8, p.host.pair_why[0..p.host.pair_why_len], "TransferTimeout"));
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
}

test "a round blocked on the dead peer: the deadline thread aborts, the engine thread then fails its streams, one exit" {
    const p = try Pair.create(true, 10 * std.time.ns_per_s, 40 * std.time.ns_per_ms, 2);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const long: Request = .{ .prompt = &prompt_a, .max_tokens = 1 << 30 };
    try e.submit(1, &long, a.sink());
    a.decoding();
    p.flaky.stuck.store(true, .release); // the next round waits on a peer that never answers, until the abort
    p.host.pairBroken("rank 1's link closed (test)", true);
    try std.testing.expectEqual(Reason.failed, a.wait()); // the abort unblocked the round; its error ended the drain
    try p.ends.waitExit();
    p.stop();
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.ends.exits.load(.acquire)); // the engine thread's or the deadline's, once
}

test "without a pair policy a break is nothing and a fatal-looking error fails its request alone" {
    const p = try Pair.create(false, 0, 0, 2);
    defer p.destroy();
    const e = p.host.engine();
    p.host.pairBroken("ignored", false);
    try std.testing.expect(!p.host.isBroken());
    p.flaky.prefill_fail = error.PairBroken;
    var a: Reply = .{};
    const req: Request = .{ .prompt = &prompt_a, .max_tokens = 4 };
    try e.submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.failed, a.wait());
    p.flaky.prefill_fail = null;
    var b: Reply = .{};
    try e.submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expect(!p.host.isBroken());
    try std.testing.expectEqual(@as(u32, 0), p.ends.exits.load(.acquire));
}

test "a break reported after the host stopped (a tier's teardown) changes nothing and starts no deadline" {
    const p = try Pair.create(true, 0, 0, 1);
    defer p.destroy();
    p.stop();
    p.host.pairBroken("the spill tier's agreement: rank 1 went (teardown)", true);
    try std.testing.expect(!p.host.isBroken());
    try std.testing.expect(p.host.pair_thread == null);
    try std.testing.expectEqual(@as(u32, 0), p.ends.exits.load(.acquire));
}

test "a peer that hangs without erroring: the stall monitor breaks the pair, nothing after the abort is delivered" {
    const p = try Pair.createWith(true, 10 * std.time.ns_per_s, 20 * std.time.ns_per_ms, 2, 60 * std.time.ns_per_ms);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const long: Request = .{ .prompt = &prompt_a, .max_tokens = 1 << 30 };
    try e.submit(1, &long, a.sink());
    a.decoding();
    try std.testing.expect(!p.host.isBroken()); // decoding rounds keep the stall clock fresh
    // the next round never ends and nothing reports an error (a SIGSTOPped follower), then returns as if it worked
    p.flaky.stuck_ok = true;
    p.flaky.stuck.store(true, .release);
    while (!p.flaky.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    const before = a.read().tokens; // (every earlier round delivered: the engine thread is inside this one)
    try std.testing.expectEqual(Reason.failed, a.wait());
    try p.ends.waitExit();
    try std.testing.expectEqual(before, a.read().tokens); // the round that returned after the abort sent nothing
    try std.testing.expect(p.host.isBroken());
    try std.testing.expect(std.mem.startsWith(u8, p.host.pair_why[0..p.host.pair_why_len], "a round or prompt chunk made no progress"));
    try std.testing.expect(!p.host.pair_drain);
    try std.testing.expect(a.says("the server restarts"));
    p.stop();
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), p.ends.exits.load(.acquire));
}

test "status during a hang: past the stall bound it is not healthy (stalled, then broken); no policy, no pair status" {
    const p = try Pair.createWith(true, 10 * std.time.ns_per_s, 20 * std.time.ns_per_ms, 2, 60 * std.time.ns_per_ms);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const long: Request = .{ .prompt = &prompt_a, .max_tokens = 1 << 30 };
    try e.submit(1, &long, a.sink());
    a.decoding();
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expect(st.pair != null);
    try std.testing.expectEqualStrings("ok", st.pair.?.health());
    try std.testing.expectEqual(@as(u64, 60), st.pair.?.stall_ms);
    p.flaky.stuck_ok = true;
    p.flaky.stuck.store(true, .release);
    while (!p.flaky.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    // the monitor ticks every 250 ms: between the bound and its tick the status says stalled, after it broken
    var saw_stalled = false;
    var i: usize = 0;
    while (i < 5000) : (i += 1) {
        e.status(&st, &.{});
        const ps = st.pair.?;
        if (ps.stalled and !ps.broken) {
            saw_stalled = true;
            try std.testing.expectEqualStrings("stalled", ps.health());
            try std.testing.expect(ps.busy_ms >= 60);
        }
        if (ps.broken) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(saw_stalled);
    try std.testing.expectEqualStrings("broken", st.pair.?.health());
    try std.testing.expectEqual(Reason.failed, a.wait());
    try p.ends.waitExit();
    e.status(&st, &.{});
    try std.testing.expect(st.pair.?.pair_failed >= 1);
    p.stop();
    const q = try Pair.create(false, 0, 0, 1);
    defer q.destroy();
    q.host.engine().status(&st, &.{});
    try std.testing.expect(st.pair == null);
}

test "an idle host with the monitor on never breaks; it leaves with the host" {
    const p = try Pair.createWith(true, 0, 0, 1, 30 * std.time.ns_per_ms);
    defer p.destroy();
    var a: Reply = .{};
    const req: Request = .{ .prompt = &prompt_a, .max_tokens = 4 };
    try p.host.engine().submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    std.Io.sleep(io, .fromMilliseconds(120), .awake) catch {};
    try std.testing.expect(!p.host.isBroken());
    try std.testing.expectEqual(@as(i64, 0), p.host.progress_at.load(.acquire));
    p.stop();
    try std.testing.expect(p.host.pair_monitor == null);
}

test "a break during a clean shutdown ends the streams and lets the shutdown finish: no restart exit" {
    const p = try Pair.create(true, 10 * std.time.ns_per_s, 20 * std.time.ns_per_ms, 2);
    defer p.destroy();
    const e = p.host.engine();
    var a: Reply = .{};
    const long: Request = .{ .prompt = &prompt_a, .max_tokens = 1 << 30 };
    try e.submit(1, &long, a.sink());
    a.decoding();
    p.flaky.stuck.store(true, .release); // rank 1 stopped first: the round waits on it
    while (!p.flaky.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    const stopper = try std.Thread.spawn(.{}, Pair.stop, .{p}); // stop.sh stopping rank 0 meanwhile
    while (true) { // (closing set: stop() now waits for the engine thread)
        p.host.mutex.lockUncancelable(io);
        const closing = p.host.closing;
        p.host.mutex.unlock(io);
        if (closing) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    p.host.pairBroken("rank 1's link closed (test)", false);
    stopper.join();
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expectEqual(@as(u32, 1), p.ends.aborts.load(.acquire)); // (the round was stuck: the deadline's)
    try std.testing.expectEqual(@as(u32, 0), p.ends.exits.load(.acquire));
}

/// A prompt pass of slow chunks (15 ms each); at `break_at` the pair breaks mid-pass (null: never).
const SlowChunks = struct {
    host: *LaneHost,
    break_at: ?usize = null,
    fn hook(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
        const c: *SlowChunks = @ptrCast(@alignCast(ctx));
        std.Io.sleep(io, .fromMilliseconds(15), .awake) catch {};
        if (c.break_at) |b| if (chunk == b) c.host.pairBroken("rank 1 went mid-pass (test)", true);
    }
};

test "a long prompt pass that keeps making progress is never a stall: each chunk restarts the clock" {
    const p = try Pair.createWith(true, 0, 10 * std.time.ns_per_s, 1, 60 * std.time.ns_per_ms);
    defer p.destroy();
    var slow: SlowChunks = .{ .host = &p.host };
    p.target.prefill_chunks = 12; // 180 ms, three times the stall bound
    p.target.prefill_hook = SlowChunks.hook;
    p.target.prefill_hook_ctx = &slow;
    var a: Reply = .{};
    const req: Request = .{ .prompt = &prompt_a, .max_tokens = 4 };
    try p.host.engine().submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expect(!p.host.isBroken());
    try std.testing.expectEqual(@as(u32, 0), p.ends.exits.load(.acquire));
}

test "a break during a prompt pass stops it at its next chunk and fails it with the pair's message" {
    const p = try Pair.create(true, 10 * std.time.ns_per_s, 10 * std.time.ns_per_s, 1);
    defer p.destroy();
    var slow: SlowChunks = .{ .host = &p.host, .break_at = 2 };
    p.target.prefill_chunks = 400; // 6 s if it ran whole
    p.target.prefill_hook = SlowChunks.hook;
    p.target.prefill_hook_ctx = &slow;
    var a: Reply = .{};
    const req: Request = .{ .prompt = &prompt_a, .max_tokens = 4 };
    try p.host.engine().submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expect(a.says("rank pair broke"));
    try p.ends.waitExit();
    try std.testing.expect(p.target.prefill_count < 10);
    try std.testing.expectEqual(@as(u64, 1), p.host.pair_failed);
}

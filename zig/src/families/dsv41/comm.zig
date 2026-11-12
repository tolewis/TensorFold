//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
//!
//! Abort (robustness, round 3): a cross-part KV transfer that failed or timed out on one rank (docs/port/shard-lanes.md,
//! "Failure": killing rank 1 during a carry) leaves the other rank's device stream behind an NCCL send or receive that
//! will never be matched, so every later launch on that stream waits forever. `Comm.abort` (ncclCommAbort, looked up
//! from the library the engine already opened) stops the communicator's kernels so the stream drains; the communicator
//! cannot be rebuilt in place (a new one needs every rank's ncclCommInitRank over a fresh unique id), so after an abort
//! every collective here refuses with error.CommAborted and the engine exits for its supervisor to restart the pair
//! (docs/port/robustness.md).
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;
const rdma = @import("rdma.zig");

const nccl = cuda.nccl;

/// ncclCommAbort's signature (NCCL 2.4 and later; every NCCL the serving image carries has it).
pub const AbortFn = *const fn (nccl.Comm) callconv(.c) nccl.Result;

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    /// Set once by `abort` (any thread): every collective after it refuses (error.CommAborted) instead of reaching a
    /// communicator NCCL has freed. Read through the methods' pointer, under the enqueue gate (never from a copy).
    aborted: std.atomic.Value(bool) = .init(false),
    /// ncclCommAbort; null: looked up in `lib` when first needed (tests set a fake).
    abort_fn: ?AbortFn = null,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        c.end();
        c.lib.close();
        c.* = undefined;
    }

    /// The communicator released: destroyed, unless an abort already freed it (NCCL: an aborted communicator is
    /// gone, a destroy of it is a use after free).
    fn end(c: *Comm) void {
        if (!c.aborted.load(.acquire)) _ = c.lib.api.ncclCommDestroy(c.comm);
    }

    /// Stops this rank's communicator now (ncclCommAbort): its kernels in flight return, so a device stream wedged
    /// behind an unmatched send or receive (a peer half that never ran) drains, and nothing waits on it afterwards.
    /// Safe from any thread, including while the engine thread is blocked on that stream (NCCL's documented use).
    /// Once only: true when this call aborted it. The communicator is unusable afterwards (CommAborted): the engine
    /// must exit, since rebuilding it needs every rank. False when it was already aborted, or when the library has no
    /// ncclCommAbort (logged; the communicator is then marked aborted all the same, so nothing new is enqueued).
    ///
    /// Never concurrent with an enqueue here: the gate every collective holds around its NCCL calls is taken first,
    /// within abort_wait_ms. When the engine thread holds it that long (blocked inside NCCL itself, e.g. a group
    /// waiting to connect to a dead peer), the communicator is only marked aborted (nothing new is enqueued) and not
    /// freed under the call: false then, and the caller's exit ends the process.
    pub fn abort(c: *Comm) bool {
        if (c.aborted.load(.acquire)) return false;
        if (!tryEnter(abort_wait_ms)) {
            if (c.aborted.swap(true, .acq_rel)) return false;
            logErr("dsv41: rank {d}'s NCCL communicator is busy in an enqueue for {d} ms: marked aborted, not freed under it", .{ c.rank, abort_wait_ms });
            return false;
        }
        defer leave();
        if (c.aborted.swap(true, .acq_rel)) return false;
        const f = c.abort_fn orelse c.lib.lib.lookup(AbortFn, "ncclCommAbort") orelse {
            logErr("dsv41: libnccl has no ncclCommAbort: the communicator cannot be stopped, only abandoned", .{});
            return false;
        };
        const r = f(c.comm);
        if (r != 0) {
            const s = c.lib.api.ncclGetErrorString(r);
            logErr("dsv41: ncclCommAbort: {s} ({d})", .{ if (s) |p| std.mem.span(p) else "?", r });
            return false;
        }
        logErr("dsv41: rank {d}'s NCCL communicator aborted: its device stream no longer waits on a peer", .{c.rank});
        return true;
    }

    /// Whether `abort` ran (the communicator is gone).
    pub fn isAborted(c: *const Comm) bool {
        return c.aborted.load(.acquire);
    }

    /// The enqueue gate taken and the communicator still live: the caller's NCCL calls, then `leave`.
    fn enter(c: *const Comm) error{CommAborted}!void {
        while (gate.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        if (c.aborted.load(.acquire)) {
            leave();
            return error.CommAborted;
        }
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: *const Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        try c.enter();
        defer leave();
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: *const Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        try c.enter();
        defer leave();
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }

    /// TF_DS_KV_SHARD (GLM 0118's send_rows / recv_rows): device byte spans one way, `spans` to rank `peer` in one NCCL
    /// group (the peer calls recvSpans with spans of the same sizes in the same order; each send meets its receive).
    /// A plane's rows go straight from the sender's plane into the receiver's: no staging.
    pub fn sendSpans(c: *const Comm, spans: []const Span, peer: u32, stream: cuda.Stream) !void {
        try c.p2p(spans, peer, stream, true);
    }

    pub fn recvSpans(c: *const Comm, spans: []const Span, peer: u32, stream: cuda.Stream) !void {
        try c.p2p(spans, peer, stream, false);
    }

    fn p2p(c: *const Comm, spans: []const Span, peer: u32, stream: cuda.Stream, out: bool) !void {
        if (peer == c.rank or peer >= c.world) return error.BadPeer;
        try c.enter();
        defer leave();
        var any = false;
        for (spans) |sp| any = any or sp.bytes > 0;
        if (!any) return;
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        var first: nccl.Result = 0;
        for (spans) |sp| {
            if (sp.bytes == 0) continue;
            const r = if (out)
                c.lib.api.ncclSend(sp.ptr, sp.bytes, .u8, @intCast(peer), c.comm, stream.handle)
            else
                c.lib.api.ncclRecv(sp.ptr, sp.bytes, .u8, @intCast(peer), c.comm, stream.handle);
            if (first == 0) first = r;
        }
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(first, if (out) "ncclSend" else "ncclRecv");
    }

    /// `count` bytes from this rank's slot of `buf` (buf + rank * count) to every rank's slot, in place (NCCL's
    /// in-place all-gather: the send buffer is the receive buffer's own slot).
    pub fn allGatherInPlace(c: *const Comm, buf: u64, count: usize, stream: cuda.Stream) !void {
        try c.allGather(buf + c.rank * count, buf, count, .u8, stream);
    }
};

/// The gate every collective's NCCL calls hold against `Comm.abort` (process-wide: one engine thread enqueues, the
/// abort is the only other taker, so it is never contended in the served path).
var gate: std.atomic.Value(bool) = .init(false);

/// How long an abort waits for an enqueue in progress before it marks the communicator aborted without freeing it.
pub const abort_wait_ms = 200;

fn leave() void {
    gate.store(false, .release);
}

/// The gate within `ms` (1 ms steps); false: held all that time.
fn tryEnter(ms: u32) bool {
    var waited: u32 = 0;
    while (gate.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
        if (waited >= ms) return false;
        const ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
        waited += 1;
    }
    return true;
}

/// Every collective of this rank stopped after a broken pair, any thread, once each: the RDMA rings first (their
/// abort word releases a collect kernel spinning on a dead peer's flags; TP2's decode gathers run there whenever
/// TF_RDMA_DEVICES is set), then NCCL (a stream behind an unmatched send or receive). Returns whether NCCL aborted.
pub fn abortAll(c: *Comm, rings: []const *rdma.Ring, why: []const u8) bool {
    for (rings) |r| r.abort(why);
    return c.abort();
}

/// The engine log's error line; host tests stay quiet (a test that logs an error fails).
fn logErr(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.err(fmt, args);
}

/// A device byte span (Comm.sendSpans / recvSpans).
pub const Span = struct { ptr: u64, bytes: usize };

/// A Comm over a fake NCCL (no library, no device): its calls are counted, ncclCommAbort included.
const FakeNccl = struct {
    var sends: usize = 0;
    var recvs: usize = 0;
    var gathers: usize = 0;
    var aborts: usize = 0;
    var destroys: usize = 0;
    var abort_result: nccl.Result = 0;
    var hold: std.atomic.Value(bool) = .init(false); // a send blocks inside NCCL while set
    var holding: std.atomic.Value(bool) = .init(false);

    fn reset() void {
        sends = 0;
        recvs = 0;
        gathers = 0;
        aborts = 0;
        destroys = 0;
        abort_result = 0;
    }
    fn ok() callconv(.c) nccl.Result {
        return 0;
    }
    fn send(_: u64, _: usize, _: nccl.DataType, _: c_int, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) nccl.Result {
        holding.store(true, .release);
        while (hold.load(.acquire)) std.atomic.spinLoopHint();
        holding.store(false, .release);
        sends += 1;
        return 0;
    }
    fn recv(_: u64, _: usize, _: nccl.DataType, _: c_int, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) nccl.Result {
        recvs += 1;
        return 0;
    }
    fn gather(_: u64, _: u64, _: usize, _: nccl.DataType, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) nccl.Result {
        gathers += 1;
        return 0;
    }
    fn destroy(_: nccl.Comm) callconv(.c) nccl.Result {
        destroys += 1;
        return 0;
    }
    fn abortIt(_: nccl.Comm) callconv(.c) nccl.Result {
        aborts += 1;
        return abort_result;
    }
    fn errString(_: nccl.Result) callconv(.c) ?[*:0]const u8 {
        return "fake";
    }

    fn comm(rank: u32) Comm {
        var a: nccl.Api = undefined;
        a.ncclSend = send;
        a.ncclRecv = recv;
        a.ncclAllGather = gather;
        a.ncclGroupStart = ok;
        a.ncclGroupEnd = ok;
        a.ncclCommDestroy = destroy;
        a.ncclGetErrorString = errString;
        return .{ .lib = .{ .lib = undefined, .api = a }, .comm = @ptrFromInt(0x1000), .rank = rank, .world = 2, .abort_fn = abortIt };
    }
};

test "an aborted communicator stops once, refuses every later collective, and is never destroyed again" {
    FakeNccl.reset();
    var c = FakeNccl.comm(0);
    const s: cuda.Stream = .{ .d = undefined, .handle = null };
    const spans = [_]Span{ .{ .ptr = 0x10, .bytes = 64 }, .{ .ptr = 0x80, .bytes = 0 }, .{ .ptr = 0x100, .bytes = 32 } };
    try c.sendSpans(&spans, 1, s); // a carry's send half: two non-empty spans in one group
    try std.testing.expectEqual(@as(usize, 2), FakeNccl.sends);
    try c.allGather(0x10, 0x20, 4, .f32, s);
    try std.testing.expect(!c.isAborted());
    // the peer's half never ran: rank 0 aborts (the lane host's deadline thread or the transfer's own failure path)
    try std.testing.expect(c.abort());
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.aborts);
    try std.testing.expect(c.isAborted());
    try std.testing.expect(!c.abort()); // once: a second cause does not abort a freed communicator again
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.aborts);
    // nothing reaches NCCL any more (each call reads the flag through its pointer, under the gate)
    try std.testing.expectError(error.CommAborted, c.sendSpans(&spans, 1, s));
    try std.testing.expectError(error.CommAborted, c.recvSpans(&spans, 1, s));
    try std.testing.expectError(error.CommAborted, c.allGather(0x10, 0x20, 4, .f32, s));
    try std.testing.expectError(error.CommAborted, c.exchange(0x10, 4, 0x20, 4, .u8, 1, s));
    try std.testing.expectError(error.CommAborted, c.allGatherInPlace(0x10, 4, s));
    try std.testing.expectEqual(@as(usize, 2), FakeNccl.sends);
    try std.testing.expectEqual(@as(usize, 0), FakeNccl.recvs);
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.gathers);
    c.end(); // (deinit's first half: the library itself is fake here)
    try std.testing.expectEqual(@as(usize, 0), FakeNccl.destroys);
}

test "a live communicator is destroyed at the end; a failed abort still marks it unusable" {
    FakeNccl.reset();
    var c = FakeNccl.comm(1);
    c.end();
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.destroys);
    FakeNccl.reset();
    var d = FakeNccl.comm(1);
    FakeNccl.abort_result = 3; // ncclInternalError
    try std.testing.expect(!d.abort());
    try std.testing.expect(d.isAborted());
    const s: cuda.Stream = .{ .d = undefined, .handle = null };
    try std.testing.expectError(error.CommAborted, d.recvSpans(&.{.{ .ptr = 1, .bytes = 1 }}, 0, s));
    try std.testing.expectEqual(@as(usize, 0), FakeNccl.recvs);
}

test "an abort never frees the communicator under an enqueue in progress: it marks it, and nothing new goes in" {
    FakeNccl.reset();
    var c = FakeNccl.comm(0);
    const s: cuda.Stream = .{ .d = undefined, .handle = null };
    const spans = [_]Span{.{ .ptr = 0x10, .bytes = 64 }};
    FakeNccl.hold.store(true, .release); // the engine thread blocked inside NCCL (a group waiting on a dead peer)
    const T = struct {
        fn run(cc: *Comm, sp: []const Span, st: cuda.Stream) void {
            cc.sendSpans(sp, 1, st) catch unreachable;
        }
    };
    const t = try std.Thread.spawn(.{}, T.run, .{ &c, &spans, s });
    while (!FakeNccl.holding.load(.acquire)) std.atomic.spinLoopHint();
    try std.testing.expect(!c.abort()); // the gate stayed held for abort_wait_ms: marked, ncclCommAbort not called
    try std.testing.expectEqual(@as(usize, 0), FakeNccl.aborts);
    try std.testing.expect(c.isAborted());
    FakeNccl.hold.store(false, .release);
    t.join();
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.sends); // the call in progress finished on a live communicator
    try std.testing.expectError(error.CommAborted, c.sendSpans(&spans, 1, s));
    try std.testing.expect(!c.abort());
    try std.testing.expectEqual(@as(usize, 0), FakeNccl.aborts);
}

/// A ring with only what `abort` touches (its failure flag, reason and metadata block's abort word).
var test_meta: [4096]u8 align(std.heap.page_size_min) = @splat(0);

test "abortAll stops the RDMA rings (their abort word: a collect kernel spinning on a dead peer returns) and NCCL" {
    FakeNccl.reset();
    var c = FakeNccl.comm(0);
    var ring: rdma.Ring = undefined;
    ring.failed = .init(false);
    ring.why = @splat(0);
    ring.s = .{ .slots = 2, .max_bytes = 64 };
    ring.world = 2;
    ring.meta.mem = &test_meta;
    @memset(&test_meta, 0);
    const rings = [_]*rdma.Ring{&ring};
    try std.testing.expect(abortAll(&c, &rings, "rank pair broken"));
    try std.testing.expectEqualStrings("rank pair broken", ring.failure().?);
    const words: [*]const u64 = @ptrCast(@alignCast(&test_meta));
    try std.testing.expectEqual(@as(u64, 1), words[2 * 2 + 1 + 2 + 2]); // flags[slots][world], doorbell, sizes, sent, abort
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.aborts);
    // a second cause: the ring keeps its first reason, NCCL is not aborted again
    try std.testing.expect(!abortAll(&c, &rings, "a later cause"));
    try std.testing.expectEqualStrings("rank pair broken", ring.failure().?);
    try std.testing.expectEqual(@as(usize, 1), FakeNccl.aborts);
}

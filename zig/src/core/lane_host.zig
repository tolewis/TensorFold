//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
//!
//! Pending durable restores and continuation telemetry port wojo's GLM 0110 (PR #124; the scheduler hooks) and 0111
//! (PR #128). A request may be parked before its prompt pass on a restore job reading a longer kept state from the
//! prompt cache's tier (shared by every request that wants the same point). Parked requests hold no lane, count as
//! waiting work and wake back to the front of the queue, where an ordinary lookup runs again. With no tier nothing is
//! parked, nothing is serviced and nothing else changes.
//!
//! Robustness (round 3, docs/port/robustness.md), each off unless its backend or tier asks for it:
//! - A broken rank pair (`pair`, the backend's TF_DS_PAIR_BROKEN=exit): nothing new is admitted, waiting requests
//!   fail at once, decoding streams finish within a drain bound or fail, the backend's collectives stop, and the
//!   process exits with a distinct code for its supervisor to restart both ranks; a deadline thread enforces the exit
//!   even when a round is blocked on a dead peer.
//! - Restores the tier refuses for capacity are deferred briefly and asked again (bounded), keeping the best memory
//!   state meanwhile, instead of a prefill from 0 (`restores_deferred`).
//! - The service fence: parked work's service and the lone driver's pump never run inside a round, inside a prompt
//!   pass except between its chunks (Stream.between), or while the backend reports a CUDA graph capture (`quiet`).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    // the request at the queue's front waits for room in the backend's pool (its prompt pass refused ContextFull while
    // other streams ran): admitted again once fewer than this many streams are active, or at full_retry (a backend
    // may refuse for a while to keep its kept states, waiting for one stream, and then place it)
    full_wait: ?usize = null,
    full_retry: i96 = 0,
    // after a long prompt pass (the whole pass runs inside its admission, no round meanwhile), the streams within
    // finish_left tokens of their max_tokens decode before the next admission, finish_ns at most: such a stream ends
    // and frees its extent (its kept state then resumes in place) instead of waiting out every queued pass, as the
    // Python lane's fills leave decoding its share
    finish_until: ?i96 = null,
    filling: ?*Job = null, // the job whose prompt pass runs in pieces between rounds (core.filling's)
    // prefill interleave (TF_DS_PREFILL_INTERLEAVE_MS, 0/unset: off): between two chunks of a long prompt pass, decode
    // rounds for the other live streams for up to this many ms, their tokens delivered, so a cold long prompt slows a
    // chat instead of freezing it for the whole pass; the other jobs' cancels are taken there too (issue #3). The
    // passing job itself is left alone: its own cancel stays in `cancels` for its pass to see, admissions wait.
    // Off when TF_FILL_CHUNKS runs passes in pieces (the two are alternatives, not meant to combine).
    interleave_ns: i96 = 0,
    passing: ?*Job = null, // the job whose prompt pass runs inside addStream (Stream.between's caller)
    between_err: ?anyerror = null,
    // -- parked requests (0110). The lists are the engine thread's; changes take the lock (status reads their lengths).
    ready: std.ArrayList(*Job) = .empty, // woken: admitted before the queue, in the order they woke
    loading: std.ArrayList(Load) = .empty, // each waits for a restore job (several may share one)
    turn: usize = 0, // which restore a service polls first (one budget, shared fairly)
    lone_commits: u32 = 0, // the lone driver's commits since its last pump
    tier: TierPolicy = .{},
    host_counts: HostCounts = .{},
    /// What status reports under `cache` (published on the engine thread at each noteLive; null until configured).
    tele: ?api.CacheStats = null,
    /// A backend's KV-ownership stats (its parts), read on the engine thread at each noteLive, published as `kv_tele`.
    kv_parts: ?lanes.kv_parts.Source = null,
    kv_tele: ?api.KvParts = null,
    // -- a broken rank pair (the backend's policy, set before start; null: off, nothing below runs) ---------------
    pair: ?PairPolicy = null,
    pair_state: std.atomic.Value(u8) = .init(0), // 0 whole, 1 a cause being recorded, 2 broken (why/drain/since set)
    pair_why: [pair_why_max]u8 = undefined,
    pair_why_len: usize = 0,
    pair_drain: bool = true,
    pair_since: i96 = 0,
    pair_thread: ?std.Thread = null, // the deadline: aborts and exits even when the engine thread is stuck
    pair_done: std.atomic.Value(bool) = .init(false), // the deadline thread may leave (the host stopped or exits)
    pair_aborted: std.atomic.Value(bool) = .init(false),
    pair_exited: std.atomic.Value(bool) = .init(false),
    pair_failed: u64 = 0, // requests failed because the pair broke (engine thread)
    pair_monitor: ?std.Thread = null, // the stall monitor: a round or a prompt chunk that never ends is a break too
    busy_depth: u32 = 0, // engine thread: inside a round, a prompt pass or the lone driver (nested)
    progress_at: std.atomic.Value(i64) = .init(0), // the engine thread's last progress while busy (awake ns); 0: idle
    // -- the service fence (cache-tier's lifetime rules) -------------------------------------------------------
    /// The backend's word that its device is between launches it may interleave with (dsv41: no CUDA graph capture
    /// on its stream); null: always.
    quiet: ?Quiet = null,
    rounds_open: u32 = 0, // engine thread: inside the lane core's step (a round's launches)
    in_between: bool = false, // engine thread: inside Stream.between (the filling pass stands between two chunks)
    /// A service asked for where it must not run panics (debug builds), else is skipped and counted.
    fence_panic: bool = @import("builtin").mode == .debug,
    service_refused: u64 = 0,
    // -- restores the tier refused for capacity, asked again shortly (engine thread's; changes take the lock) --
    deferred: std.ArrayList(Defer) = .empty,

    const Mark = struct { at: i96, tokens: u64 };
    const full_retry_ns: i96 = std.time.ns_per_s;
    const long_pass_ns: i96 = std.time.ns_per_s;
    const finish_ns: i96 = 5 * std.time.ns_per_s;
    const finish_left: i64 = 256;
    const window_ns: i96 = 2 * std.time.ns_per_s;
    /// How often an idle host polls work in flight (restores, staged writes, deferred retries).
    const poll_ns: i64 = 2 * std.time.ns_per_ms;
    /// The lone driver pumps staged writes every this many commits.
    const lone_pump_commits: u32 = 16;

    const Load = struct { job: *Job, restore: *pc.Restore, since: i96 };
    const Defer = struct { job: *Job, retry_at: i96 };
    const pair_why_max = 240;
    const monitor_ms = 250;

    /// The backend's device state for the service fence.
    pub const Quiet = struct { ctx: *anyopaque, check: *const fn (ctx: *anyopaque) bool };

    /// A backend error that means the rank pair broke, wherever it surfaces (a prompt pass, a round, the lone
    /// driver); `drain`: rounds can still run, so decoding streams may finish first.
    pub const Fatal = struct { err: anyerror, drain: bool };

    /// What the host does once the backend's rank pair cannot go on (dsv41: TF_DS_PAIR_BROKEN=exit). The backend
    /// reports a break through `pairBroken` (its hooks) or through an error in `fatal`.
    pub const PairPolicy = struct {
        ctx: *anyopaque,
        /// Stop the backend's collectives so a device stream blocked on the peer drains (dsv41: ncclCommAbort).
        /// Called once, from the engine thread or the deadline thread, possibly while the other is blocked on it.
        abort: *const fn (ctx: *anyopaque) void,
        /// End the process with `code` (the service's supervisor restarts the pair); returns only in tests.
        exit: *const fn (ctx: *anyopaque, code: u8) void,
        code: u8 = pair_exit_code,
        /// Decoding streams may finish for this long after a break that leaves rounds runnable (`drain`).
        drain_ns: i96 = 5 * std.time.ns_per_s,
        /// The deadline thread aborts the collectives this long after the break if the engine thread has not exited
        /// by then (a round blocked on a dead peer), and exits `grace_ns` later.
        deadline_ns: i96 = pair_deadline_ns,
        /// Before the exit: the HTTP threads send the failed requests' last events.
        grace_ns: i96 = 300 * std.time.ns_per_ms,
        /// A round, a prompt chunk or a lone-driver token taking longer than this is a break too (a peer that hangs
        /// without erroring: stopped, a GPU hang, a follower stuck in its own device wait, whose RDMA writes still
        /// complete at the NIC so nothing on this rank fails). Longer than every rank_wire reply deadline (validate:
        /// 90 s) and any one chunk; 0: no monitor.
        stall_ns: i96 = pair_stall_ns,
        fatal: []const Fatal = &.{},
    };

    /// PairPolicy's bounds: the deadline thread's after a break, and the stall monitor's.
    pub const pair_deadline_ns: i96 = 15 * std.time.ns_per_s;
    pub const pair_stall_ns: i96 = 120 * std.time.ns_per_s;

    /// The process exit status after a broken pair (EX_TEMPFAIL: restart and retry).
    pub const pair_exit_code: u8 = 75;

    /// How parked work is serviced (LaneHost.configure; the defaults are off).
    pub const TierPolicy = struct {
        /// One elapsed-time budget per service for every restore together, then staged writes (TF_SPILL_LOAD_MS).
        service_ns: u64 = 40 * std.time.ns_per_ms,
        /// Staging's share of a service while streams decode (GLM pumped 4 blocks a round); idle, all that is left.
        pump_ns: u64 = 2 * std.time.ns_per_ms,
        /// How long a request waits parked on a restore before it is admitted without it (TF_SPILL_LOAD_WAIT_MS).
        load_wait_ns: i96 = 30 * std.time.ns_per_s,
        /// A clean shutdown's bound for draining writes (TF_SPILL_FLUSH_MS).
        flush_ns: u64 = 60 * std.time.ns_per_s,
        /// A restore the tier refused for capacity (Store.deferrable) is asked again every defer_retry_ns, and when a
        /// restore in flight ends, for at most defer_ns (TF_SPILL_DEFER_MS; 0: never, admitted on memory at once),
        /// when it would save at least defer_min tokens over memory (0: the store's grid, or 1).
        defer_ns: i96 = 5 * std.time.ns_per_s,
        defer_retry_ns: i96 = 25 * std.time.ns_per_ms,
        defer_min: u32 = 0,
    };

    pub const HostCounts = struct {
        restore_waits: u64 = 0, // requests parked on a restore
        restore_wait_timeouts: u64 = 0, // of them admitted without it at load_wait_ns
        restore_wait_ns: u64 = 0,
        restore_wait_max_ns: u64 = 0,
        cached_tokens: u64 = 0, // prompt tokens passes resumed (memory or restored)
        computed_tokens: u64 = 0, // and computed
        restores_deferred: u64 = 0, // requests whose restore was refused for capacity and asked again
        restores_deferred_expired: u64 = 0, // of them, admitted on memory at defer_ns
    };

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up
        tried_disk: bool = false, // it waited for a restore once: a later lookup takes what memory has
        deferred_since: ?i96 = null, // its restore was first refused for capacity then (asked again until defer_ns)
        join: pc.Join = .{}, // its conversation (0111)
        parent: ?u64 = null, // the durable point its pass resumes from, and where
        fork: u32 = 0,
        lineage: []pc.Fork = &.{}, // the computation its pass's rows belong to (gpa-owned; Store.passLineage)

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            const h = job.host;
            const store = h.cache orelse return;
            // an image span keyed by its digest (Request.cacheKey)
            _ = store.keepWith(job.request.cacheKey(), at, s, job.request.chunks, .{ .group = job.join.group, .images = job.request.images.len > 0, .parent = job.parent, .fork = job.fork, .lineage = job.lineage });
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
            if (!ok) { // it did not compute from that state's rows: a lineage of its own
                job.parent = null;
                job.fork = 0;
                if (job.lineage.len > 0) {
                    const fresh: []pc.Fork = store.passLineage(null) catch &.{};
                    job.host.gpa.free(job.lineage);
                    job.lineage = fresh;
                }
            }
            if (job.stream.reuse_failed) store.resumed(e, job.request.cacheKey(), false) else if (ok) store.resumed(e, job.request.cacheKey(), true);
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        var ms: i96 = 0;
        if (std.c.getenv("TF_DS_PREFILL_INTERLEAVE_MS")) |v| ms = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced, .interleave_ns = ms * std.time.ns_per_ms };
    }

    /// The tier's switches (before start). Telemetry is published from here on when either is on: a tier on the cache
    /// (Store.setTier), or grouping.
    pub fn configure(h: *LaneHost, sp: api.Spill) void {
        h.tier.service_ns = @as(u64, @max(sp.load_ms, 1)) * std.time.ns_per_ms;
        h.tier.pump_ns = @min(h.tier.pump_ns, h.tier.service_ns);
        h.tier.load_wait_ns = @as(i96, @max(sp.load_wait_ms, 1)) * std.time.ns_per_ms;
        h.tier.flush_ns = @as(u64, sp.flush_ms) * std.time.ns_per_ms;
        h.tier.defer_ns = @as(i96, sp.defer_ms) * std.time.ns_per_ms;
        const store_on = if (h.cache) |s| s.tier != null or s.groups else false;
        h.tele = if (store_on) .{} else null;
    }

    pub fn start(h: *LaneHost) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
        if (h.pair) |pol| if (pol.stall_ns > 0) {
            h.pair_monitor = std.Thread.spawn(.{}, pairMonitor, .{h}) catch |e| blk: {
                noteErr("the rank pair's stall monitor did not start ({s}): a hung peer is caught only by its errors", .{@errorName(e)});
                break :blk null;
            };
        };
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        h.pair_done.store(true, .release);
        if (h.pair_monitor) |t| t.join(); // (first: a break it reported started the deadline thread below)
        h.pair_monitor = null;
        if (h.pair_thread) |t| t.join();
        h.pair_thread = null;
        h.deferred.deinit(h.gpa);
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
        h.ready.deinit(h.gpa);
        h.loading.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn } };
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.isBroken()) { // (a broken pair admits nothing more: the process restarts)
            h.gpa.destroy(job);
            return error.Closed;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
        out.* = .{
            .running = @intCast(h.admitted.items.len),
            .waiting = @intCast(h.queued.items.len + h.waitingCount()),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
            .cache = h.tele,
            .kv_parts = h.kv_tele,
            .pair = h.pairStatus(now),
        };
    }

    /// The broken-pair policy's state and the fence's counter (any thread; the counters are the engine thread's,
    /// read as they stand); null with neither the policy nor a disk tier.
    fn pairStatus(h: *const LaneHost, now: i96) ?api.PairStatus {
        const tier = if (h.cache) |s| s.tier != null else false;
        if (h.pair == null and !tier) return null;
        var p: api.PairStatus = .{ .service_refused = h.service_refused, .pair_failed = h.pair_failed };
        const pol = h.pair orelse return p;
        p.broken = h.isBroken();
        const at = h.progress_at.load(.acquire);
        const idle: i96 = if (at == 0) 0 else @max(now - at, 0);
        p.busy_ms = @intCast(@divTrunc(idle, std.time.ns_per_ms));
        p.stall_ms = @intCast(@divTrunc(@max(pol.stall_ns, 0), std.time.ns_per_ms));
        p.stalled = pol.stall_ns > 0 and at != 0 and idle >= pol.stall_ns;
        return p;
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.progressed(); // (asked once a prompt chunk: the stall monitor's clock restarts)
        if (job.host.isBroken()) return true; // (a broken pair: the pass stops at its next chunk, failed below)
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, "");
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        job.reported();
        h.gpa.free(job.marks);
        h.gpa.free(job.lineage);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *LaneHost, n: usize) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.takeCancelsKeeping(null);
    }

    /// takeCancels, except a cancel of `keep` (the job in its prompt pass) stays in `cancels` for that pass to see.
    fn takeCancelsKeeping(h: *LaneHost, keep: ?Id) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        if (keep) |k| if (std.mem.indexOfScalar(Id, ids, k) != null) h.cancels.append(h.gpa, k) catch {};
        var dropped: std.ArrayList(*Job) = .empty;
        var unloaded: std.ArrayList(*pc.Restore) = .empty; // restores a cancelled job waited for
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
            for (h.ready.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.ready.orderedRemove(i)) catch {};
                break;
            };
            for (h.loading.items, 0..) |l, i| if (l.job.id == id) {
                _ = h.loading.orderedRemove(i);
                dropped.append(h.gpa, l.job) catch {};
                unloaded.append(h.gpa, l.restore) catch {};
                break;
            };
            for (h.deferred.items, 0..) |d, i| if (d.job.id == id) {
                dropped.append(h.gpa, h.deferred.orderedRemove(i).job) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        // a restore no request waits for any more stops (its reads, DMA and reservations go), exactly once
        for (unloaded.items, 0..) |r, k| {
            if (std.mem.indexOfScalar(*pc.Restore, unloaded.items[0..k], r) != null) continue;
            const wanted = for (h.loading.items) |l| {
                if (l.restore == r) break true;
            } else false;
            if (!wanted) if (h.cache) |store| store.restoreCancel(r);
        }
        unloaded.deinit(h.gpa);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            if (keep) |k| if (id == k) continue; // its pass ends it (error.Cancelled) and admitOne finishes it
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                if (h.filling == job) h.filling = null;
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request (while a pass fills, the first that core.cutsIn); false if none or no lane.
    fn admitOne(h: *LaneHost) bool {
        if (h.isBroken()) return false;
        h.lock();
        if ((h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return false;
        }
        var at: usize = 0;
        if (h.filling != null) {
            at = for (h.queued.items, 0..) |j, i| {
                if (h.core.cutsIn(j.request.prompt.len, j.request.images.len > 0)) break i;
            } else {
                h.unlock();
                return false;
            };
        }
        if (h.full_wait) |n| {
            const active = h.core.activeCount();
            if (active > 0 and active >= n and std.Io.Clock.awake.now(h.io).toNanoseconds() < h.full_retry) {
                h.unlock();
                return false;
            }
            h.full_wait = null;
        }
        if (h.finish_until) |t| {
            if (std.Io.Clock.awake.now(h.io).toNanoseconds() < t and h.nearEnd()) {
                h.unlock();
                return false;
            }
            h.finish_until = null;
        }
        // a woken (restored) request goes first, but not while a pass fills: then only a queued one that cuts in
        const job = if (h.filling == null and h.ready.items.len > 0) h.ready.orderedRemove(0) else h.queued.orderedRemove(at);
        if (h.parking()) {
            h.unlock();
            if (h.park(job)) return true; // waits for a restore, holding no lane
            h.lock();
        }
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        if (h.cache) |store| if (store.lookup(h.gpa, r.cacheKey(), r.history_len, r.shared_prefixes, r.chunks)) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
            if (store.groups) job.join = store.join(r.cacheKey(), l.entry, r.images.len == 0 and r.drafts);
            if (l.entry) |e| {
                job.parent = e.point;
                job.fork = e.at;
            }
            // its rows' computation: the resumed state's forked at its position (written or not), else its own
            job.lineage = store.passLineage(l.entry) catch &.{};
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return h.drop(job, "the drafter could not start");
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .call = if (r.call) |c| .{
                .opener = c.opener,
                .think_open = if (c.think_open) |t| t else -1,
                .think_end = if (c.think_end) |t| t else -1,
                .lead = c.lead,
                .names = c.names,
                .tail = c.tail,
                .lex = .{ .ctx = c.lex.ctx, .blank = c.lex.blank, .text = c.lex.text, .encode = c.lex.encode },
            } else null,
            .chunks = r.chunks,
            .reuse = reuse,
            .images = r.images,
        }) catch {
            job.proposer.deinit();
            return h.drop(job, "out of memory");
        };
        job.started = true;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.began = began;
        if (h.loneFits(job)) return h.runLone(job, began);
        // between two prompt chunks: the other streams' rounds (interleave), and parked work's service, which runs
        // with the interleave off too
        if (r.images.len == 0 and (h.interleaving() or h.servicing())) job.stream.between = .{ .ptr = h, .wanted = betweenWanted, .run = between };
        h.passing = job;
        defer h.passing = null;
        h.busyEnter();
        const added = h.core.addStream(&job.stream);
        h.busyLeave();
        added catch |e| {
            if (h.isBroken()) return h.dropBroken(job);
            if (e == error.Cancelled) return h.cancel(job);
            // refused for room: while other streams run or a pass fills (this one cut in), it waits in the queue
            if (e == error.ContextFull and (h.core.activeCount() > 0 or h.filling != null)) return h.requeue(job);
            h.checkFatal(e);
            return h.drop(job, h.words(e));
        };
        if (h.aborted()) return h.dropBroken(job); // (the collectives were stopped under it: its rows are not the model's)
        if (h.core.filling == &job.stream) {
            h.filling = job; // its next pieces run between rounds (fillOne)
            return true;
        }
        h.prefilled(job, began);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        if (now - began >= long_pass_ns) {
            h.lock();
            h.finish_until = now + finish_ns;
            h.unlock();
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    /// The prefill interleave is on: TF_DS_PREFILL_INTERLEAVE_MS set and passes not run in pieces (TF_FILL_CHUNKS).
    fn interleaving(h: *const LaneHost) bool {
        return h.interleave_ns > 0 and h.core.fill_chunks == 0;
    }

    fn betweenWanted(ptr: *anyopaque) bool {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        if (h.core.log != null) return false;
        return (h.interleaving() and h.core.runnableCount() > 0) or h.pending();
    }

    /// Stream.between: decode rounds for the other live streams, interleave_ns at most, between two of the filling
    /// job's prompt chunks; their tokens go out as they land. Then parked work's service. Engine thread only (inside
    /// the filling job's addStream).
    fn between(ptr: *anyopaque) anyerror!void {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        if (h.core.log != null) return; // a step resets the core's arena, which a trail's open events live in
        h.in_between = true; // (the filling pass stands between two chunks: the service fence's one way in)
        defer h.in_between = false;
        h.progressed(); // (a chunk ended: the stall monitor's clock restarts)
        if (h.interleaving()) {
            const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            while (true) {
                h.takeCancelsKeeping(if (h.passing) |f| f.id else null);
                if (h.core.runnableCount() == 0 or std.Io.Clock.awake.now(h.io).toNanoseconds() - t0 >= h.interleave_ns) break;
                h.stepRound() catch |e| {
                    h.between_err = e; // the pass fails with it; run() then fails the streams the round held
                    return e;
                };
                if (h.aborted()) { // (the collectives were stopped under the round: its tokens are not the model's)
                    h.between_err = error.CommAborted;
                    return error.CommAborted;
                }
                var i: usize = 0;
                while (i < h.admitted.items.len) {
                    const job = h.admitted.items[i];
                    if (job == h.passing or job.prefilled == null) {
                        i += 1;
                        continue;
                    }
                    if (h.deliver(job)) {
                        h.lock();
                        _ = h.admitted.orderedRemove(i);
                        h.unlock();
                    } else i += 1;
                }
                h.noteLive();
            }
        }
        if (h.servicing()) {
            if (h.passing) |f| f.reported(); // its restore is done (a chunk ran): a restore's room may evict that entry now
            h.service(h.tier.service_ns);
        }
    }

    /// Whether an admitted stream decodes within finish_left tokens of its max_tokens (the lock held).
    fn nearEnd(h: *LaneHost) bool {
        for (h.admitted.items) |job| if (job.started and !job.stream.finished and job.stream.budgetLeft() <= finish_left) return true;
        return false;
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        h.host_counts.cached_tokens += job.stream.cached;
        h.host_counts.computed_tokens += job.request.prompt.len -| job.stream.cached;
        if (h.cache) |store| {
            store.continued(job.join, job.stream.cached, job.request.prompt.len);
            store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        }
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// An idle backend driver takes a lone drafted request, including sampling when supported.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0 and h.waitingCount() == 0;
    }

    /// Requests parked or woken (the lock held).
    fn waitingCount(h: *const LaneHost) usize {
        return h.ready.items.len + h.loading.items.len + h.deferred.items.len;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                host.progressed();
                if (host.aborted()) return; // (stopped collectives: nothing more of it is sent; yield ends it)
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
                // writes enqueued at keep are staged during a long lone decode too (parked work makes it yield)
                host.lone_commits += 1;
                if (host.lone_commits >= lone_pump_commits and host.servicing()) {
                    host.lone_commits = 0;
                    if (host.cache) |store| if (host.fenced("the lone driver's pump")) store.pump(host.tier.pump_ns);
                }
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing or host.waitingCount() > 0 or host.isBroken();
            }
        };
        job.began = began;
        h.busyEnter();
        const ran = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.busyLeave();
        h.lone_job = null;
        const paused = if (h.aborted()) error.CommAborted else ran;
        const handed = paused catch |e| {
            h.checkFatal(e);
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    /// A prompt pass the broken pair stopped (or ran under an abort) fails with the pair's message.
    fn dropBroken(h: *LaneHost, job: *Job) bool {
        h.pair_failed += 1;
        return h.drop(job, "the engine's rank pair broke; the server restarts");
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A request whose prompt pass the backend's pool cannot hold while other streams run or a pass fills (ContextFull,
    /// before anything of it was placed) goes back to the queue's front and waits until one of them ends, as the
    /// Python lane's window waits, instead of failing, or a second at most (the backend may place it by then). Returns
    /// false: nothing more is admitted now.
    fn requeue(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.core.discard(&job.stream);
        job.entry = null; // nothing was restored: the store keeps the entry for the next lookup
        h.gpa.free(job.marks);
        job.marks = &.{};
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        job.started = false;
        job.began = 0;
        job.prefilled = null;
        job.delivered = 0;
        job.prefill_sent = false;
        job.join = .{};
        job.parent = null;
        job.fork = 0;
        h.gpa.free(job.lineage);
        job.lineage = &.{};
        h.lock();
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.full_wait = h.core.activeCount();
        h.full_retry = std.Io.Clock.awake.now(h.io).toNanoseconds() + full_retry_ns;
        h.unlock();
        return false;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn noteLive(h: *LaneHost) void {
        const tele = h.telemetry();
        var parts: ?api.KvParts = null;
        if (h.kv_parts) |src| {
            parts = .{};
            src.read(src.ctx, &parts.?);
        }
        h.lock();
        defer h.unlock();
        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
        if (tele) |t| h.tele = t;
        if (parts) |p| h.kv_tele = p;
    }

    /// The filling job's next piece, when its turn has come; its first tokens once its prompt pass ends.
    fn fillOne(h: *LaneHost) void {
        const job = h.filling orelse return;
        // a piece is a pass's collectives too: under the stall monitor, and a broken pair ends it as addStream's does
        h.busyEnter();
        const stepped = h.core.fillStep();
        h.busyLeave();
        const done = stepped catch |e| {
            h.filling = null;
            if (h.isBroken()) {
                _ = h.dropBroken(job);
                return;
            }
            if (e == error.Cancelled) {
                _ = h.cancel(job);
                return;
            }
            h.checkFatal(e);
            _ = h.drop(job, h.words(e));
            return;
        };
        if (h.aborted()) { // (the collectives were stopped under it: its rows are not the model's)
            h.filling = null;
            _ = h.dropBroken(job);
            return;
        }
        if (!done) return;
        h.filling = null;
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }

    fn run(h: *LaneHost) void {
        while (true) {
            if (h.isBroken()) {
                if (h.brokenStep()) return;
                continue;
            }
            h.takeCancels();
            while (h.admitOne()) {}
            h.fillOne();
            if (h.servicing()) h.service(h.tier.service_ns); // restores, staged writes and deferred retries, between rounds and at idle
            if (h.between_err) |e| {
                h.between_err = null;
                h.checkFatal(e);
                h.failAll(@errorName(e));
                continue;
            }
            h.noteLive();
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len + h.waitingCount();
                h.unlock();
                if (left == 0) return h.shutdownTier();
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (h.filling != null) {
                    h.unlock();
                    continue;
                }
                if (h.cancels.items.len == 0 and ((h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes)) {
                    const ms: i64 = if (h.pending()) poll_ns / std.time.ns_per_ms else 100; // work in flight: poll it
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(ms), .clock = .awake } }) catch {};
                }
                h.unlock();
                continue;
            }
            h.unlock();
            h.stepRound() catch |e| {
                h.checkFatal(e);
                h.failAll(@errorName(e));
                continue;
            };
            if (h.aborted()) continue; // (stopped under the round: brokenStep fails its streams, nothing is delivered)
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        h.filling = null;
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.ready.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.loading.items) |l| h.cancels.append(h.gpa, l.job.id) catch {};
        for (h.deferred.items) |d| h.cancels.append(h.gpa, d.job.id) catch {};
        h.unlock();
        h.takeCancels();
    }

    /// One round of the lane core, counted open while it runs (the service fence).
    fn stepRound(h: *LaneHost) !void {
        h.rounds_open += 1;
        defer h.rounds_open -= 1;
        h.busyEnter();
        defer h.busyLeave();
        return h.core.step();
    }

    // -- the stall monitor's clock (engine thread; nothing without `pair`) --------------------------------------

    fn busyEnter(h: *LaneHost) void {
        if (h.pair == null) return;
        h.busy_depth += 1;
        h.progressed();
    }

    fn busyLeave(h: *LaneHost) void {
        if (h.pair == null) return;
        h.busy_depth -= 1;
        if (h.busy_depth == 0) h.progress_at.store(0, .release) else h.progressed();
    }

    /// The engine thread moved on (a round or a chunk ended, a lone token landed): the stall clock restarts.
    fn progressed(h: *LaneHost) void {
        if (h.pair == null or h.busy_depth == 0) return;
        h.progress_at.store(@intCast(std.Io.Clock.awake.now(h.io).toNanoseconds()), .release);
    }

    /// Whether the collectives were stopped (an abort ran): what a round returns after it is not the model's.
    fn aborted(h: *const LaneHost) bool {
        return h.pair != null and h.pair_aborted.load(.acquire);
    }

    // -- parked work (0110) ---------------------------------------------------------------------------------------

    /// Whether admission may park a request: the cache has a tier.
    fn parking(h: *const LaneHost) bool {
        const store = h.cache orelse return false;
        return store.tier != null;
    }

    /// Whether the engine thread services parked work (restores, staged writes, deferred retries).
    fn servicing(h: *const LaneHost) bool {
        return h.parking();
    }

    /// Work in flight the host must poll (engine thread): restores, writes to stage, requests parked or woken.
    fn pending(h: *const LaneHost) bool {
        if (h.loading.items.len > 0 or h.ready.items.len > 0 or h.deferred.items.len > 0) return true;
        const store = h.cache orelse return false;
        return store.busy();
    }

    /// Before `job`'s pass (engine thread, no lock): park it on a restore of a longer durable point (once). False:
    /// admit it now.
    fn park(h: *LaneHost, job: *Job) bool {
        const store = h.cache orelse return false;
        const r = job.request;
        if (r.images.len > 0 and r.cache_key.len == 0) return false; // pictures without their digest key: never from disk
        const key = r.cacheKey();
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        const probe = store.probe(key, r.chunks);
        if (!job.tried_disk) if (probe.durable) |p| {
            job.tried_disk = true;
            h.lock();
            const room = h.loading.ensureUnusedCapacity(h.gpa, 1);
            h.unlock();
            if (room) |_| {
                // (the best memory state stays while the budget holds it beside the restore: refused, the pass
                // resumes it; a full tier refuses before any room is made)
                if (store.restoreBeginOver(p, key, .{ .have = probe.memory, .again = job.deferred_since != null })) |rs| {
                    h.lock();
                    h.loading.appendAssumeCapacity(.{ .job = job, .restore = rs, .since = now });
                    h.unlock();
                    h.host_counts.restore_waits += 1;
                    job.deferred_since = null;
                    return true;
                } else |err| {
                    const left = store.find(key, r.chunks); // (the room may have evicted probe.memory: never read it now)
                    if (h.deferRestore(job, p, left, err, now)) return true;
                    if (left) |e| {
                        note("reading {d} kept tokens from disk did not start ({s}); resuming the {d} memory holds", .{ p.at, @errorName(err), e.at });
                    } else note("reading {d} kept tokens from disk did not start ({s}); prefilling", .{ p.at, @errorName(err) });
                }
            } else |_| {}
        };
        return false;
    }

    /// A restore of `p` the tier or the budget refused with `err`: when that may change shortly (Store.deferrable)
    /// and the point saves at least defer_min tokens over `have` (the best memory state), `job` waits on the deferred
    /// list (holding no lane) and asks again every defer_retry_ns or when a restore in flight ends, defer_ns at most
    /// from its first refusal; true then. False: admit it now on what memory holds (`have` is never evicted for the
    /// restore's room, so a usable shorter state still resumes).
    fn deferRestore(h: *LaneHost, job: *Job, p: pc.Point, have: ?*pc.Entry, err: anyerror, now: i96) bool {
        const store = h.cache orelse return false;
        if (h.tier.defer_ns <= 0 or !store.deferrable(err)) return false;
        const at: u32 = if (have) |e| e.at else 0;
        const min = if (h.tier.defer_min > 0) h.tier.defer_min else @max(store.rules.grid, 1);
        if (p.at <= at or p.at - at < min) return false;
        const first = job.deferred_since orelse now;
        if (now - first >= h.tier.defer_ns) {
            h.host_counts.restores_deferred_expired += 1;
            note("reading {d} kept tokens from disk was refused for capacity for {d} ms ({s}); admitted on what memory holds", .{ p.at, @divTrunc(now - first, std.time.ns_per_ms), @errorName(err) });
            return false;
        }
        h.lock();
        defer h.unlock();
        h.deferred.append(h.gpa, .{ .job = job, .retry_at = now + h.tier.defer_retry_ns }) catch return false;
        if (job.deferred_since == null) h.host_counts.restores_deferred += 1;
        job.deferred_since = first;
        job.tried_disk = false; // (it asks for the restore again when it wakes)
        return true;
    }

    /// Deferred restores whose retry time came, or every one when a restore in flight ended (`ended`: its read slot
    /// is free now), go to the ready list to ask again.
    fn retryDeferred(h: *LaneHost, now: i96, ended: bool) void {
        if (h.deferred.items.len == 0) return;
        var moved = false;
        var i: usize = 0;
        while (i < h.deferred.items.len) {
            if (!ended and h.deferred.items[i].retry_at > now) {
                i += 1;
                continue;
            }
            h.lock();
            const d = h.deferred.orderedRemove(i);
            h.toReady(d.job); // (releases the lock)
            moved = true;
        }
        if (!moved) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// A woken job to the ready list (the lock held; released here). Out of memory fails it alone.
    fn toReady(h: *LaneHost, job: *Job) void {
        h.ready.append(h.gpa, job) catch {
            h.unlock();
            return h.finish(job, .failed, "out of memory");
        };
        h.unlock();
    }

    /// Parked work's service (engine thread): restores progress under one elapsed-time budget shared fairly (each gets
    /// an equal share of what is left, a different one first each time); writes are staged with what remains. Restores
    /// that end wake every request waiting for them; deferred restores whose retry came ask again.
    /// Called inside a fill's `between` too, so a family's restore commands there must keep the follower's ordered
    /// stream valid (dsv41 joins Engram read-ahead before `between`, as recipe 0005 requires).
    fn service(h: *LaneHost, budget_ns: u64) void {
        const store = h.cache orelse return;
        if (!h.fenced("parked work's service")) return;
        const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.expireLoads(t0);
        var ended = false;
        const n = store.restoring.items.len;
        if (n > 0) { // at most 64 a service, a different first one each time: every restore is polled in turn
            var buf: [64]*pc.Restore = undefined;
            const m = @min(n, buf.len);
            h.turn = (h.turn + 1) % n;
            for (0..m) |k| buf[k] = store.restoring.items[(h.turn + k) % n];
            for (buf[0..m], 0..) |r, k| {
                const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
                if (spent >= budget_ns and k > 0) break;
                const share = (budget_ns -| spent) / (m - k);
                const st = store.restorePoll(r, share);
                if (st != .pending) {
                    h.restoreEnded(r);
                    ended = true;
                }
            }
        }
        h.retryDeferred(std.Io.Clock.awake.now(h.io).toNanoseconds(), ended);
        const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
        const idle = h.filling == null and h.core.activeCount() == 0; // nothing decodes: staging takes what is left
        store.pump(if (idle) budget_ns -| spent else @min(budget_ns -| spent, h.tier.pump_ns));
    }

    /// Requests parked on a restore longer than load_wait_ns are admitted without it (they look up what memory holds);
    /// a restore no request waits for any more stops, once.
    fn expireLoads(h: *LaneHost, now: i96) void {
        const store = h.cache orelse return;
        var any = false;
        var i: usize = 0;
        while (i < h.loading.items.len) {
            const l = h.loading.items[i];
            if (now - l.since < h.tier.load_wait_ns) {
                i += 1;
                continue;
            }
            any = true;
            h.lock();
            _ = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_timeouts += 1;
            h.host_counts.restore_wait_ns += waited;
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job); // (releases the lock)
            const wanted = for (h.loading.items) |o| {
                if (o.restore == l.restore) break true;
            } else false;
            if (!wanted) store.restoreCancel(l.restore);
        }
        if (!any) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// Restore `r` ended (adopted, dropped or failed; `r` is already freed, only its address is compared): its
    /// waiters wake in arrival order, to look up what memory holds now.
    fn restoreEnded(h: *LaneHost, r: *pc.Restore) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        while (true) {
            h.lock();
            const i = for (h.loading.items, 0..) |l, i| {
                if (l.restore == r) break i;
            } else {
                h.wake.signal(h.io);
                return h.unlock();
            };
            const l = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_ns += waited;
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job);
        }
    }

    // -- the service fence --------------------------------------------------------------------------------------

    /// Whether parked work may be serviced now (cache-tier's lifetime rules): no round open (its launches, or a graph
    /// it captures, in flight), a prompt pass only from Stream.between (between two chunks, never between a chunk's
    /// layers), and the backend not capturing a CUDA graph (`quiet`).
    fn serviceSafe(h: *const LaneHost) bool {
        if (h.rounds_open > 0) return false;
        if (h.filling != null and !h.in_between) return false;
        if (h.quiet) |q| if (!q.check(q.ctx)) return false;
        return true;
    }

    /// serviceSafe, or the refusal: a panic in debug builds (an engine bug), else counted, logged once and skipped.
    fn fenced(h: *LaneHost, comptime what: []const u8) bool {
        if (h.serviceSafe()) return true;
        h.service_refused += 1;
        if (h.fence_panic) @panic("lane host: " ++ what ++ " asked for inside a round, a prompt chunk or a CUDA graph capture");
        if (h.service_refused == 1) noteErr(what ++ " asked for inside a round, a prompt chunk or a CUDA graph capture: skipped (an engine bug)", .{});
        return false;
    }

    // -- a broken rank pair ---------------------------------------------------------------------------------------

    /// Whether the backend's pair broke (any thread).
    pub fn isBroken(h: *const LaneHost) bool {
        return h.pair_state.load(.acquire) == 2;
    }

    /// The backend's rank pair cannot go on (`why`, for the log and the failed requests; `drain`: rounds still run,
    /// so decoding streams may finish within drain_ns). Any thread; the first cause only; nothing without `pair`.
    /// The engine thread then admits nothing, fails what waits, ends the streams, stops the collectives and exits;
    /// a deadline thread does the last two if the engine thread cannot (blocked on the peer).
    pub fn pairBroken(h: *LaneHost, why: []const u8, drain: bool) void {
        _ = h.pair orelse return;
        if (h.pair_done.load(.acquire)) return; // stopped (a tier's teardown may still report): nothing to end
        if (h.pair_state.cmpxchgStrong(0, 1, .acq_rel, .acquire) != null) return;
        const n = @min(why.len, pair_why_max);
        @memcpy(h.pair_why[0..n], why[0..n]);
        h.pair_why_len = n;
        h.pair_drain = drain;
        h.pair_since = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.pair_state.store(2, .release);
        h.lock();
        const closing = h.closing;
        h.unlock();
        if (closing) {
            note("the rank pair broke during the shutdown ({s}): the streams left end, the shutdown goes on", .{why[0..n]});
        } else noteErr("the rank pair broke ({s}): admitting nothing more; {s}", .{ why[0..n], if (drain) "decoding streams may finish first" else "every stream fails now" });
        h.pair_thread = std.Thread.spawn(.{}, pairDeadline, .{h}) catch null;
        h.lock();
        h.wake.broadcast(h.io);
        h.unlock();
    }

    /// pairBroken as a callback (a backend's hook: `ctx` is the LaneHost).
    pub fn pairBrokenFn(ctx: *anyopaque, why: []const u8, drain: bool) void {
        self(ctx).pairBroken(why, drain);
    }

    /// A backend error the policy names as a broken pair (`fatal`): the policy runs, its words as the cause.
    fn checkFatal(h: *LaneHost, e: anyerror) void {
        const pol = h.pair orelse return;
        for (pol.fatal) |f| if (f.err == e) {
            var buf: [pair_why_max]u8 = undefined;
            const said = h.words(e);
            const why = if (std.mem.eql(u8, said, @errorName(e))) said else std.fmt.bufPrint(&buf, "{s}: {s}", .{ @errorName(e), said }) catch @errorName(e);
            return h.pairBroken(why, f.drain);
        };
    }

    /// The engine thread's loop after the break: what waits fails; decoding streams step on within the drain bound;
    /// then the collectives stop, every stream left fails and the process exits. True: the host is stopping (tests).
    fn brokenStep(h: *LaneHost) bool {
        const pol = h.pair.?;
        var mbuf: [pair_why_max + 64]u8 = undefined;
        const msg = std.fmt.bufPrint(&mbuf, "the engine's rank pair broke ({s}); the server restarts", .{h.pair_why[0..h.pair_why_len]}) catch "the engine's rank pair broke; the server restarts";
        h.failWaiting(msg);
        h.lock();
        const closing = h.closing;
        h.unlock();
        if (closing and !h.pair_exited.load(.acquire)) {
            // a break during a clean shutdown (stop.sh stops both ranks, rank 1 often first): the streams end, the
            // communicator is stopped if a transfer may have wedged it, and the shutdown finishes as it would have
            // (the tier's flush, an ordinary exit), not the restart exit
            if (!h.pair_drain) h.abortPair();
            h.failPair(msg, null);
            h.shutdownTier();
            return true;
        }
        if (h.pair_exited.load(.acquire)) { // (only when the exit hook returned: tests) idle until stopped
            h.lock();
            defer h.unlock();
            if (h.closing) return true;
            h.cancels.clearRetainingCapacity(); // (every request already ended)
            h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(20), .clock = .awake } }) catch {};
            return false;
        }
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        if (h.pair_drain and now - h.pair_since < pol.drain_ns and h.core.activeCount() > 0) {
            h.takeCancels();
            h.stepRound() catch |e| {
                h.failPair(msg, e);
                return false;
            };
            if (h.aborted()) { // (the deadline stopped the collectives under it)
                h.failPair(msg, null);
                return false;
            }
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
            h.noteLive();
            return false;
        }
        h.abortPair(); // no round runs any more: a stream blocked on the peer drains
        h.failPair(msg, null);
        h.exitPair();
        return false;
    }

    /// Every admitted stream fails with the pair's message (`e`: a round's error while draining, logged too).
    fn failPair(h: *LaneHost, msg: []const u8, e: ?anyerror) void {
        if (e) |x| noteErr("a round failed while the streams drained ({s}): every stream ends", .{@errorName(x)});
        h.lock();
        const n = h.admitted.items.len;
        h.unlock();
        h.pair_failed += n;
        h.failAll(msg);
    }

    /// Everything that waits fails with `msg` (queued, woken, reading a restore, deferred), each restore it
    /// waited for stopped once.
    fn failWaiting(h: *LaneHost, msg: []const u8) void {
        var jobs: std.ArrayList(*Job) = .empty;
        defer jobs.deinit(h.gpa);
        var restores: std.ArrayList(*pc.Restore) = .empty;
        defer restores.deinit(h.gpa);
        h.lock();
        for ([_]*std.ArrayList(*Job){ &h.queued, &h.ready }) |list| {
            for (list.items) |job| jobs.append(h.gpa, job) catch {};
            list.clearRetainingCapacity();
        }
        for (h.loading.items) |l| {
            jobs.append(h.gpa, l.job) catch {};
            if (std.mem.indexOfScalar(*pc.Restore, restores.items, l.restore) == null) restores.append(h.gpa, l.restore) catch {};
        }
        h.loading.clearRetainingCapacity();
        for (h.deferred.items) |d| jobs.append(h.gpa, d.job) catch {};
        h.deferred.clearRetainingCapacity();
        h.unlock();
        if (h.cache) |store| for (restores.items) |r| store.restoreCancel(r);
        h.pair_failed += jobs.items.len;
        for (jobs.items) |job| h.finish(job, .failed, msg);
    }

    fn abortPair(h: *LaneHost) void {
        const pol = h.pair orelse return;
        if (!h.pair_aborted.swap(true, .acq_rel)) pol.abort(pol.ctx);
    }

    /// The distinct last line, a grace for the failed requests' events to leave, and the exit (once).
    fn exitPair(h: *LaneHost) void {
        const pol = h.pair orelse return;
        if (h.pair_exited.swap(true, .acq_rel)) return;
        noteErr("rank pair broken ({s}); {d} requests failed; exiting with status {d} for a restart of both ranks", .{ h.pair_why[0..h.pair_why_len], h.pair_failed, pol.code });
        std.Io.sleep(h.io, .fromNanoseconds(@intCast(@max(pol.grace_ns, 0))), .awake) catch {};
        h.pair_done.store(true, .release);
        pol.exit(pol.ctx, pol.code);
    }

    /// The deadline thread: if the engine thread has not exited deadline_ns after the break (a round or a prompt
    /// pass blocked on the dead peer), the collectives stop, which should let it fail its streams and exit; grace_ns
    /// after that the process exits from here.
    fn pairDeadline(h: *LaneHost) void {
        const pol = h.pair.?;
        if (!h.sleepUnlessDone(h.pair_since + pol.deadline_ns)) return;
        noteErr("the engine thread did not end its streams within {d} ms of the broken pair: stopping the collectives", .{@divTrunc(pol.deadline_ns, std.time.ns_per_ms)});
        h.abortPair();
        if (!h.sleepUnlessDone(std.Io.Clock.awake.now(h.io).toNanoseconds() + pol.grace_ns + std.time.ns_per_s)) return;
        if (h.pair_exited.swap(true, .acq_rel)) return;
        noteErr("rank pair broken ({s}); the engine thread is stuck; exiting with status {d} for a restart of both ranks", .{ h.pair_why[0..h.pair_why_len], pol.code });
        pol.exit(pol.ctx, pol.code);
    }

    /// The stall monitor: the engine thread busy (a round, a prompt pass, the lone driver) without progress for
    /// stall_ns is a break that drains nothing (the deadline thread then stops the collectives and exits). Leaves at
    /// the first break, or when the host stops.
    fn pairMonitor(h: *LaneHost) void {
        const pol = h.pair.?;
        while (!h.pair_done.load(.acquire) and h.pair_state.load(.acquire) == 0) {
            std.Io.sleep(h.io, .fromMilliseconds(monitor_ms), .awake) catch {};
            const at = h.progress_at.load(.acquire);
            if (at == 0) continue;
            const idle = std.Io.Clock.awake.now(h.io).toNanoseconds() - at;
            if (idle < pol.stall_ns) continue;
            var buf: [pair_why_max]u8 = undefined;
            const why = std.fmt.bufPrint(&buf, "a round or prompt chunk made no progress for {d} s (a peer that stopped answering, or a device hang)", .{@divTrunc(idle, std.time.ns_per_s)}) catch "a round made no progress";
            return h.pairBroken(why, false);
        }
    }

    /// Sleeps until `until` in 10 ms steps; false as soon as pair_done is set.
    fn sleepUnlessDone(h: *LaneHost, until: i96) bool {
        while (std.Io.Clock.awake.now(h.io).toNanoseconds() < until) {
            if (h.pair_done.load(.acquire)) return false;
            std.Io.sleep(h.io, .fromMilliseconds(10), .awake) catch {};
        }
        return !h.pair_done.load(.acquire);
    }

    /// Clean shutdown on the engine thread, nothing left to serve: restores stopped, writes drained within flush_ns.
    fn shutdownTier(h: *LaneHost) void {
        const store = h.cache orelse return;
        if (store.tier == null) return;
        _ = store.flush(h.tier.flush_ns);
        if (h.telemetry()) |t| {
            h.lock();
            h.tele = t;
            h.unlock();
        }
    }

    /// The cache's counters for status (engine thread); null while telemetry is off.
    fn telemetry(h: *LaneHost) ?api.CacheStats {
        if (h.tele == null) return null;
        var t: api.CacheStats = .{};
        if (h.cache) |store| {
            const c = store.counts;
            const s = store.stats;
            t.memory_hits = c.hits;
            t.memory_misses = c.misses;
            t.kept = c.kept;
            t.evicted = c.evicted;
            t.refused = c.refused;
            t.failed = c.failed;
            t.entries = store.entries.items.len;
            t.held_bytes = store.held;
            t.disk_lookups = s.disk_lookups;
            t.disk_hits = s.disk_hits;
            t.disk_hits_committed = s.disk_hits_committed;
            t.restores_begun = s.restores_begun;
            t.restores_shared = s.restores_shared;
            t.restores_failed = s.restores_failed;
            t.restores_cancelled = s.restores_cancelled;
            t.restores_dropped = s.restores_dropped;
            t.restores_refused = s.restores_refused;
            t.restores_active = store.restoring.items.len;
            t.restoring_bytes = store.restoring_bytes;
            for (store.restoring.items) |r| {
                t.restore_blocks_done += r.done;
                t.restore_blocks_total += r.total;
            }
            t.restored_tokens = s.restored_tokens;
            t.writes_enqueued = s.writes_enqueued;
            t.shared_promoted = s.shared_promoted;
            t.flushes = s.flushes;
            t.flush_timeouts = s.flush_timeouts;
            t.flush_unpublished = s.flush_unpublished;
            t.chats_continued = s.chats_continued;
            t.chats_continued_prefill_tokens = s.chats_continued_prefill_tokens;
            t.spill = store.tierCounts();
        }
        const hc = h.host_counts;
        t.restore_waits = hc.restore_waits;
        t.restore_wait_timeouts = hc.restore_wait_timeouts;
        t.restore_wait_ns = hc.restore_wait_ns;
        t.restore_wait_max_ns = hc.restore_wait_max_ns;
        t.restores_deferred = hc.restores_deferred;
        t.restores_deferred_expired = hc.restores_deferred_expired;
        t.cached_tokens = hc.cached_tokens;
        t.computed_tokens = hc.computed_tokens;
        t.parked = h.loading.items.len + h.deferred.items.len;
        return t;
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("lane host: " ++ fmt, args);
}

/// An error line (the broken pair's, the fence's); tests stay quiet too.
fn noteErr(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.err("lane host: " ++ fmt, args);
}

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "prefill interleave: a decoding stream keeps getting tokens and its cancel lands during another's prompt pass" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.interleave_ns = 5 * std.time.ns_per_ms;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn read(b: *@This()) struct { tokens: usize, done: ?Reason } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ .tokens = b.tokens, .done = b.done };
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                if (b.read().done) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    // A: a long decode, admitted and decoding before B arrives
    const prompt_a = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var a: Box = .{};
    const req_a: Request = .{ .prompt = &prompt_a, .max_tokens = 100000 };
    try e.submit(1, &req_a, .{ .ctx = &a, .event = Box.event });
    while (a.read().tokens == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // B: a six-chunk prompt pass; at each of its chunks note A's tokens, and cancel A at chunk 3
    const Watch = struct {
        engine: Engine,
        a: *Box,
        seen: [6]usize = @splat(0),
        done_at_5: ?Reason = null,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            const r = w.a.read();
            if (chunk < w.seen.len) w.seen[chunk] = r.tokens;
            if (chunk == 3) w.engine.cancel(1);
            if (chunk == 5) w.done_at_5 = r.done;
        }
    };
    var watch = Watch{ .engine = e, .a = &a };
    target.prefill_chunks = 6;
    target.prefill_hook = Watch.call;
    target.prefill_hook_ctx = &watch;
    const prompt_b = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var b: Box = .{};
    const req_b: Request = .{ .prompt = &prompt_b, .max_tokens = 1 };
    try e.submit(2, &req_b, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    // A decoded during B's pass (no interleave: seen[0] == seen[2]), and its cancel ended it before B's last chunk
    try std.testing.expect(watch.seen[2] > watch.seen[0]);
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), watch.done_at_5);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

test "the service fence: never inside a round, a prompt pass's chunk or a graph capture; skipped and counted when not panicking" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop(); // (never started: frees the lists)
    host.fence_panic = false;
    try std.testing.expect(host.fenced("service")); // the run loop between rounds, or idle
    host.rounds_open = 1; // inside the lane core's step: a round's launches in flight
    try std.testing.expect(!host.fenced("service"));
    host.rounds_open = 0;
    var job: LaneHost.Job = undefined;
    host.filling = &job; // a prompt pass running its chunk
    try std.testing.expect(!host.fenced("service"));
    host.in_between = true; // ... standing between two chunks (Stream.between)
    try std.testing.expect(host.fenced("service"));
    host.rounds_open = 1; // an interleaved round inside between
    try std.testing.expect(!host.fenced("service"));
    host.rounds_open = 0;
    host.filling = null;
    host.in_between = false;
    const Capture = struct {
        var on = true;
        fn quiet(_: *anyopaque) bool {
            return !on;
        }
    };
    host.quiet = .{ .ctx = &host, .check = Capture.quiet }; // the backend capturing a CUDA graph
    try std.testing.expect(!host.fenced("the lone driver's pump"));
    Capture.on = false;
    try std.testing.expect(host.fenced("the lone driver's pump"));
    try std.testing.expectEqual(@as(u64, 4), host.service_refused);
}

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_host_tier_test.zig");
    _ = @import("lane_host_robust_test.zig");
}

//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
//!
//! Pending durable restores and continuation telemetry port wojo's GLM 0110 (PR #124; the scheduler hooks) and 0111
//! (PR #128). A request may be parked before its prompt pass on a restore job reading a longer kept state from the
//! prompt cache's tier (shared by every request that wants the same point). Parked requests hold no lane, count as
//! waiting work and wake back to the front of the queue, where an ordinary lookup runs again. With no tier nothing is
//! parked, nothing is serviced and nothing else changes.
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
    };

    pub const HostCounts = struct {
        restore_waits: u64 = 0, // requests parked on a restore
        restore_wait_timeouts: u64 = 0, // of them admitted without it at load_wait_ns
        restore_wait_ns: u64 = 0,
        restore_wait_max_ns: u64 = 0,
        cached_tokens: u64 = 0, // prompt tokens passes resumed (memory or restored)
        computed_tokens: u64 = 0, // and computed
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
        const store_on = if (h.cache) |s| s.tier != null or s.groups else false;
        h.tele = if (store_on) .{} else null;
    }

    pub fn start(h: *LaneHost) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
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
        if (h.closing) {
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
            .waiting = @intCast(h.queued.items.len + h.ready.items.len + h.loading.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
            .cache = h.tele,
            .kv_parts = h.kv_tele,
        };
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
        h.core.addStream(&job.stream) catch |e| {
            if (e == error.Cancelled) return h.cancel(job);
            // refused for room: while other streams run or a pass fills (this one cut in), it waits in the queue
            if (e == error.ContextFull and (h.core.activeCount() > 0 or h.filling != null)) return h.requeue(job);
            return h.drop(job, h.words(e));
        };
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
        if (h.interleaving()) {
            const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            while (true) {
                h.takeCancelsKeeping(if (h.passing) |f| f.id else null);
                if (h.core.runnableCount() == 0 or std.Io.Clock.awake.now(h.io).toNanoseconds() - t0 >= h.interleave_ns) break;
                h.core.step() catch |e| {
                    h.between_err = e; // the pass fails with it; run() then fails the streams the round held
                    return e;
                };
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
        return h.ready.items.len + h.loading.items.len;
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
                    if (host.cache) |store| store.pump(host.tier.pump_ns);
                }
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing or host.waitingCount() > 0;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
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
        const done = h.core.fillStep() catch |e| {
            h.filling = null;
            _ = if (e == error.Cancelled) h.cancel(job) else h.drop(job, h.words(e));
            return;
        };
        if (!done) return;
        h.filling = null;
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }

    fn run(h: *LaneHost) void {
        while (true) {
            h.takeCancels();
            while (h.admitOne()) {}
            h.fillOne();
            if (h.servicing()) h.service(h.tier.service_ns); // restores, staged writes and deferred retries, between rounds and at idle
            if (h.between_err) |e| {
                h.between_err = null;
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
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
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
        h.unlock();
        h.takeCancels();
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
        if (h.loading.items.len > 0 or h.ready.items.len > 0) return true;
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
                if (store.restoreBegin(p, key)) |rs| {
                    h.lock();
                    h.loading.appendAssumeCapacity(.{ .job = job, .restore = rs, .since = now });
                    h.unlock();
                    h.host_counts.restore_waits += 1;
                    return true;
                } else |err| note("reading {d} kept tokens from disk did not start ({s}); prefilling", .{ p.at, @errorName(err) });
            } else |_| {}
        };
        return false;
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
        const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.expireLoads(t0);
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
                if (st != .pending) h.restoreEnded(r);
            }
        }
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
        t.cached_tokens = hc.cached_tokens;
        t.computed_tokens = hc.computed_tokens;
        t.parked = h.loading.items.len;
        return t;
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("lane host: " ++ fmt, args);
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

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_host_tier_test.zig");
}

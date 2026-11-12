//! tf-dsv41-lanes MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT KIT_DIR --requests FILE [--parallel N] [--pool P]
//!   [--drafts 0|1] [--arena-gib G] [--engram DIR --token-map FILE] [--rdma DEVICES --rdma-kernels FATBIN]: the lane
//! gate of the Zig port. TensorFold 1.0's lane core (core/lanes: its round loop, depth rule, window planning, copies,
//! acceptance) serves the requests through the port's Backend (lanes.zig) on every rank: rank 0 runs the core and
//! admits up to N requests at a time (the host's SuffixLookup proposer on each, min_match 4), rank 1 follows (WORLD 4:
//! the four nodes of the 2D split, ranks 1-3 following; --rdma: its decode-size exchanges over the RDMA rings, as the
//! layer gate's, their kernels from --rdma-kernels or the embedded image). Each reply's token_sha (sha256 of its comma-joined ids,
//! the served server's) against the served reply's: the drafted, concurrent reply must be the served one. One JSON
//! line a request, then the summary. FILE: zrec_lanereq.py's requests (prompt ids, max_tokens, sampling, expect_sha).
//! KIT_DIR: a gate dir's aot/, cubins/, RoPE tables (as many rows as the pool) and engram.json.
//! TF_DS_PROMPT_WINDOW_ROWS and TF_DS_PROMPT_GEMM_ROWS (64 and 128 unset) are read here as the served process reads them: set them alike on every rank.
//! The spill tier (GPU validation of keep, evict, restore and restart without the HTTP server): every rank takes
//! --spill-dir DIR --spill-gib G [--spill-min-free-gib F] [--spill-min-tokens T] [--spill-quota Q]
//! [--spill-staging-mib M] (else TF_DS_SPILL_DIR, TF_DS_SPILL_GIB, ... and TF_SPILL_TURNS / TF_SPILL_FLUSH_MS; the
//! recipe's TF_DS_ENGINE_ID and TF_DS_RECIPE_PATCHES must be set on every rank). Rank 0 then runs the prompt cache
//! (--prompt-cache-gib G, default 4) with the tier as the server's lane host would: each request probes it, restores
//! a durable point longer than memory, resumes and keeps at its chunk ends (written at keep). A request's
//! "drop_memory": true first waits for the tier's writes, then drops every memory state, so its hit must come from
//! disk; a second run over the same --spill-dir after a restart restores what the first one published.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const lanes = @import("lanes");

const usage = "usage: tf-dsv41-lanes MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT KIT_DIR --requests FILE [--parallel N] [--pool P] [--drafts 0|1] [--arena-gib G] [--engram DIR --token-map FILE] [--rdma DEVICES --rdma-kernels FATBIN] [--graphs 0|1 [--graph-cap N]] [--fix-k K] [--round-rows R] [--streams S] [--spill-dir DIR --spill-gib G [--spill-min-free-gib F] [--spill-min-tokens T] [--spill-quota Q] [--spill-staging-mib M] [--prompt-cache-gib G]]\n";

const pcache = dsv41.spill_tier.prompt_cache;

/// The spill switches given as flags (Settings.follower reads them, else the environment).
var spill_flags: [6]?[]const u8 = @splat(null);
const spill_names = [_][]const u8{ "--spill-dir", "--spill-gib", "--spill-min-free-gib", "--spill-min-tokens", "--spill-quota", "--spill-staging-mib" };

fn spillFlag(name: []const u8) ?[]const u8 {
    for (spill_names, spill_flags) |n, v| if (std.mem.eql(u8, n, name)) return v;
    return null;
}

fn spillEnv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

const Request = struct {
    name: []const u8 = "",
    prompt: []const u32,
    max_tokens: u32,
    seed: ?u64 = null,
    temperature: f64 = 0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    expect_sha: ?[]const u8 = null,
    images: []const ImageRows = &.{}, // a vision request: its spans' rows ready (zrec_v3req.py from a recording)
    drop_memory: bool = false, // with the spill tier: its writes drained, then every memory state dropped first
};

/// An image span of a request's prompt and its rows: bf16 [tokens, hidden] from row `row` of `rows_file`.
const ImageRows = struct { at: u32, tokens: u32, rows_file: []const u8, row: u32 = 0 };

const Job = struct {
    req: *const Request,
    index: usize,
    proposer: lanes.SuffixLookup,
    stream: lanes.Stream,
    began: std.Io.Timestamp,
    done: bool = false,
    // with the prompt cache (the spill tier on): the state it resumed, the pass's lineage, where it keeps
    store: ?*pcache.Store = null,
    entry: ?*pcache.Entry = null,
    lineage: []pcache.Fork = &.{},
    restored: u32 = 0,
    // the ids the cache keys it by (the prompt; with pictures each one's positions made from its rows' SHA-256, as the
    // server makes them from its payload: recipe 0006) and each picture's identity for the tier
    key: []const u32 = &.{},
    media: []const pcache.Media = &.{},
};

/// The prompt cache's keep at a chunk end the pass reached (Stream.reuse.hook), with the lane host's lineage rule.
fn keepHook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
    const j: *Job = @ptrCast(@alignCast(ptr));
    const st = j.store orelse return;
    const e = if (s.reuse_failed) null else j.entry;
    const pics = j.media.len > 0;
    _ = st.keepWith(j.key, at, s, &.{}, .{ .lineage = j.lineage, .parent = if (e) |x| x.point else null, .fork = if (e) |x| x.at else 0, .images = pics, .media = j.media, .raw = if (pics) j.req.prompt else &.{} });
}

/// The tier's writes drained (bounded), then every memory state forgotten: the next hit comes from disk.
fn dropMemory(io: std.Io, st: *pcache.Store) void {
    var k: usize = 0;
    while (st.busy() and k < 60_000) : (k += 1) {
        st.pump(5 * std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    while (st.entries.items.len > 0) {
        const e = st.entries.items[0];
        st.forget(e.tokens, e.at);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: dsv41.model.Options = .{
        .model_dir = args[1],
        .cache_dir = args[2],
        .rank = try std.fmt.parseInt(u32, args[3], 10),
        .world = try std.fmt.parseInt(u32, args[4], 10),
        .master = try dsv41.link.parseIp(args[5]),
        .port = try std.fmt.parseInt(u16, args[6], 10),
        .kit_dir = args[7],
    };
    o.window_rows = try dsv41.prompt_rows.windowFromEnv();
    o.gemm_rows = try dsv41.prompt_rows.gemmFromEnv();
    var requests_file: ?[]const u8 = null;
    var parallel: ?usize = null; // default: every stream of the pool
    var arena_gib: ?usize = null;
    var profile = false;
    // TF_DS_CLOCK_MARKS=1 with --profile: a GPU clock at every round phase mark (in the graphs too; it splits their
    // PDL chains, so the stretch totals run long), else only around the stretches; 4: only a_q, a_idx, a_core
    // and TF_DS_KV_SHARD's parts between them (round.PhaseTimer.shardMark: sh_* per compressed layer)
    const clock_marks: u8 = if (std.c.getenv("TF_DS_CLOCK_MARKS")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0 else 0;
    var phases = false; // --profile 2: also the rounds' GPU time by phase (eager rounds)
    var serial = false; // --serial 1: every stream without drafts (one row a round)
    var fix_k: ?usize = null; // --fix-k K: every drafted stream verifies K drafts a round (a cost measurement)
    var cache_gib: f64 = 4; // --prompt-cache-gib: rank 0's prompt cache with the spill tier
    var ai: usize = 8;
    while (ai + 1 < args.len) : (ai += 2) {
        const key = args[ai];
        const val = args[ai + 1];
        if (std.mem.eql(u8, key, "--requests")) {
            requests_file = val;
        } else if (std.mem.eql(u8, key, "--parallel")) {
            parallel = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--round-rows")) {
            o.round_rows = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--round-ms")) {
            o.round_ms = try dsv41.lanes.parseMs(a, val);
        } else if (std.mem.eql(u8, key, "--streams")) {
            o.streams = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--pool")) {
            o.pool = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--drafts")) {
            o.drafts = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--arena-gib")) {
            arena_gib = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--aio")) {
            o.engram_aio = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--prefetch")) {
            o.prefetch = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--side")) {
            o.side = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--graphs")) {
            o.graphs = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--graph-cap")) {
            o.graph_cap = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--serial")) {
            serial = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--fix-k")) {
            fix_k = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--rdma")) {
            o.rdma_devices = val;
        } else if (std.mem.eql(u8, key, "--profile")) {
            profile = !std.mem.eql(u8, val, "0");
            phases = std.mem.eql(u8, val, "2");
        } else if (std.mem.eql(u8, key, "--engram")) {
            o.engram_dir = val;
        } else if (std.mem.eql(u8, key, "--token-map")) {
            o.token_map = val;
        } else if (std.mem.eql(u8, key, "--rdma-kernels")) {
            o.rdma_kernels = val;
        } else if (std.mem.eql(u8, key, "--vision-bias")) {
            o.vision_bias = val;
        } else if (std.mem.eql(u8, key, "--logits-sha")) {
            dsv41.lanes.logits_sha = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--prompt-cache-gib")) {
            cache_gib = try std.fmt.parseFloat(f64, val);
        } else blk: {
            for (spill_names, &spill_flags) |n, *v| if (std.mem.eql(u8, n, key)) {
                v.* = val;
                break :blk;
            };
            return error.BadArgument;
        }
    }
    const spill = try dsv41.spill_tier.Settings.follower(spillFlag, spillEnv);
    // the spill tier's refusals from the arguments and environment alone, before the weights load
    if (spill != null) {
        const pre: anyerror!void = if (o.world != 2)
            error.SpillNeedsTwoRanks
        else if (o.engram_dir != null and (spillEnv("TF_DS_ENGRAM_REVISION") orelse @as([]const u8, "")).len == 0)
            error.NoEngramId // (--engram)
        else
            dsv41.native.spillPreflight(spillEnv);
        pre catch |e| {
            if (dsv41.native.explain(null, e)) |t| std.log.err("{s}", .{t});
            return e;
        };
    }
    const par = parallel orelse o.streams;
    if (par == 0 or par > o.streams or o.streams > dsv41.model.max_streams) return error.BadArgument;
    o.arena_bytes = (arena_gib orelse 0) << 30; // (0: the arena grows as the buffers ask)

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    // the spill tier's staging, before the pool
    var staging: ?cuda.HostBuffer = if (spill) |sp| try dsv41.native.spillStaging(&ctx, sp) else null;
    defer if (staging) |*b| b.free();
    const t0 = std.Io.Timestamp.now(io, .awake);
    const m = try dsv41.model.Model.open(gpa, io, &ctx, o);
    defer m.close();
    try w.print("{{\"rank\": {d}, \"loaded_s\": {d:.1}, \"pool\": {d}, \"drafts\": {}, \"arena_used\": {d}, \"arena_reserved\": {d}}}\n", .{ o.rank, seconds(t0, io), o.pool, m.drafting(), m.arena.used, m.arena.reserved });
    try w.flush();

    if (o.rank != 0) {
        if (profile) {
            m.prof = true;
            if (m.eh) |*eh| eh.io = io;
            if (phases) try m.usePhaseTimer();
            try m.useClock(clock_marks);
        }
        defer if (profile) {
            const fp = dsv41.lanes.follow_prof;
            const r: f64 = @floatFromInt(@max(fp.rounds, 1));
            w.print("{{\"rank\": {d}, \"follow_profile\": {{\"rounds\": {d}, \"enqueue_ms\": {d:.2}, \"forward_ms\": {d:.2}", .{ o.rank, fp.rounds, @as(f64, @floatFromInt(fp.enqueue)) / r / 1e6, @as(f64, @floatFromInt(fp.forward)) / r / 1e6 }) catch {};
            if (m.eh) |eh| if (eh.p_calls > 0) {
                const pc: f64 = @floatFromInt(eh.p_calls);
                w.print(", \"prefill_engram_ms\": {{\"calls\": {d}, \"rows\": {d:.2}, \"upload\": {d:.2}, \"mm\": {d:.2}, \"exchange\": {d:.2}, \"gate\": {d:.2}}}", .{ eh.p_calls, @as(f64, @floatFromInt(eh.p_rows)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_upload)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_mm)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_x)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_gate)) / pc / 1e6 }) catch {};
            };
            if (m.eh) |eh| w.print(", \"engram_read_ms\": {d:.3}, \"engram_hash_ms\": {d:.3}", .{ @as(f64, @floatFromInt(eh.t_read)) / r / 1e6, @as(f64, @floatFromInt(eh.t_hash)) / r / 1e6 }) catch {};
            w.print("}}}}\n", .{}) catch {};
            printClock(w, o.rank, m) catch {};
            printKvShard(w, o.rank, m) catch {};
            if (m.ptimer) |tm| {
                w.print("{{\"rank\": {d}, \"prefill_phases_ms\": {{", .{o.rank}) catch {};
                for (dsv41.round.PhaseTimer.phases, 0..) |name, i| w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, if (tm.rounds == 0) 0 else tm.ms[i] / @as(f64, @floatFromInt(tm.rounds)) }) catch {};
                w.print("}}, \"chunks\": {d}}}\n", .{tm.rounds}) catch {};
            }
            w.flush() catch {};
        };
        var events: []cuda.Event = &.{};
        defer {
            for (events) |*e| e.deinit();
            if (events.len > 0) gpa.free(events);
        }
        const side: ?*dsv41.lanes.SpillFollower = if (spill) |sp| try dsv41.native.followerSpill(gpa, io, m, o, sp, staging.?, &events) else null;
        defer if (side) |sd| sd.destroy();
        // the pair's one rank_wire Follower: the spill side's fingerprint, or round 2's shard-only hello for
        // TF_DS_KV_SHARD alone (lanes.followPair)
        dsv41.lanes.followWith(gpa, m, side) catch |err| {
            try w.print("{{\"rank\": {d}, \"follow\": \"{s}\"}}\n", .{ o.rank, @errorName(err) });
            try w.flush();
            if (dsv41.native.explain(null, err)) |t| std.log.err("{s}", .{t});
            return 1;
        };
        if (side) |sd| if (sd.side == null) std.log.warn("dsv41 spill: this rank was started with the spill tier but rank 0 never agreed one (rank 0 runs without --spill-dir/--spill-gib?): its staging went unused", .{});
        if (side) |sd| if (sd.side) |s| {
            s.refresh();
            const c = s.cached;
            try w.print("{{\"rank\": {d}, \"spill\": {{\"points\": {d}, \"committed\": {d}, \"cancelled\": {d}, \"failed\": {d}, \"reads\": {d}, \"read_failed\": {d}, \"skipped\": {d}}}}}\n", .{ o.rank, c.points, c.writes_committed, c.writes_cancelled, c.writes_failed, c.reads, c.read_failed, sd.skipped });
        };
        try w.print("{{\"rank\": {d}, \"followed\": true}}\n", .{o.rank});
        try w.flush();
        return 0;
    }

    // rank 0: the requests, the backend and the lane core
    const text = try std.Io.Dir.cwd().readFileAlloc(io, requests_file orelse return error.NoRequests, a, .limited(1 << 30));
    const Requests = struct { requests: []Request };
    const reqs = (try std.json.parseFromSliceLeaky(Requests, a, text, .{ .ignore_unknown_fields = true })).requests;
    var ln = dsv41.lanes.Lanes.init(gpa, m);
    defer ln.deinit();
    // the pair's one rank_wire Leader (the spill tier and TF_DS_KV_SHARD share it), then the parts agreed with the
    // followers before the first request
    try dsv41.native.openPair(gpa, io, m, &ln, o, spill);
    // the spill tier before the parts' planner (native.zig's order: both ranks scan their stores at once), then the
    // prompt cache over it (the server's wiring: native/cuda.zig openWith)
    var events: []cuda.Event = &.{};
    defer {
        for (events) |*e| e.deinit();
        if (events.len > 0) gpa.free(events);
    }
    if (spill) |sp| events = try dsv41.native.openSpill(gpa, io, m, &ln, o, sp, staging.?);
    defer if (ln.spill) |t| { // (after the prompt cache's store, before the events)
        t.destroy();
        ln.spill = null;
    };
    try ln.startKv();
    ln.fix_k = fix_k;
    var snap_vt: pcache.Snapshots.VTable = undefined;
    var store_v: pcache.Store = undefined;
    var store: ?*pcache.Store = null;
    defer if (store) |st| {
        _ = st.flush(@as(u64, spill.?.flush_ms) * std.time.ns_per_ms);
        st.deinit();
    };
    if (spill != null) {
        const L = dsv41.lanes.Lanes;
        snap_vt = .{ .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn, .usable = L.usableFn };
        store_v = pcache.Store.init(gpa, .{ .ptr = &ln, .vtable = &snap_vt }, .{ .grid = dsv41.model.chunk_rows, .window = @intCast(m.cfg.window) }, @intFromFloat(cache_gib * (1 << 30)));
        var seed: [4]u8 = undefined;
        io.random(&seed);
        store_v.setTier(ln.spill.?.tier(), std.mem.readInt(u32, &seed, .little));
        store = &store_v;
    }
    if (profile) {
        ln.prof = .{};
        m.prof = true;
        if (m.eh) |*eh| eh.io = io;
        if (phases) try m.usePhaseTimer();
        try m.useClock(clock_marks);
    }
    var step_ns: u64 = 0;
    const rows: u32 = @intCast(m.eng.round_rows);
    var cfg = try lanes.Config.init(gpa, ln.facts(), rows, rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &cfg, ln.backend(), clock.clock());
    defer core.deinit();

    const eos = [_]u32{@intCast(m.cfg.eos)};
    const jobs = try a.alloc(Job, reqs.len);
    var admitted: usize = 0;
    var finished: usize = 0;
    var all_equal = true;
    var tokens_out: usize = 0;
    const started = std.Io.Timestamp.now(io, .awake);
    while (finished < reqs.len) {
        // admit while a lane is free (the host's admission: one prompt pass at a time, before the rounds)
        while (admitted < reqs.len and core.activeCount() < par) {
            const r = &reqs[admitted];
            const j = &jobs[admitted];
            j.* = .{ .req = r, .index = admitted, .proposer = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 }), .stream = undefined, .began = std.Io.Timestamp.now(io, .awake), .store = store };
            var reuse: lanes.stream.Reuse = .{};
            const images = try imageRows(a, io, r.images, m.cfg.hidden);
            if (store) |st| {
                try cacheKey(a, r.prompt, images, j);
                if (r.drop_memory) dropMemory(io, st);
                // a durable point longer than memory: restored first (the lane host parks the request meanwhile)
                const pr = st.probeMedia(j.key, &.{}, images.len > 0, j.media);
                if (pr.durable) |p| if (st.restoreBegin(p, j.key)) |rs| {
                    var got = st.restorePoll(rs, 40 * std.time.ns_per_ms);
                    while (got == .pending) : (got = st.restorePoll(rs, 40 * std.time.ns_per_ms)) {
                        if (core.live.items.len > 0) try core.step();
                    }
                    if (got == .adopted) j.restored = p.at;
                    try w.print("{{\"rank\": 0, \"request\": {d}, \"restore\": \"{s}\", \"at\": {d}}}\n", .{ admitted, @tagName(got), p.at });
                } else |err| try w.print("{{\"rank\": 0, \"request\": {d}, \"restore_refused\": \"{s}\"}}\n", .{ admitted, @errorName(err) });
                const look = try st.lookup(a, j.key, @intCast(r.prompt.len), &.{}, &.{});
                j.entry = look.entry;
                j.lineage = try st.passLineage(look.entry);
                reuse = .{ .saved = if (look.entry) |e| e.saved else null, .at = if (look.entry) |e| e.at else 0, .marks = look.marks, .hook = .{ .ptr = j, .at = keepHook } };
            }
            j.stream = try lanes.Stream.init(gpa, .{
                .id = r.name,
                .prompt = r.prompt,
                .images = images,
                .max_new = r.max_tokens,
                .eos = &eos,
                .sampling = if (r.temperature > 0) .{ .seed = r.seed orelse 0, .temperature = r.temperature, .top_k = r.top_k, .top_p = r.top_p } else null,
                .drafts = !serial,
                .proposer = j.proposer.proposer(),
                .reuse = reuse,
            });
            admitted += 1;
            core.addStream(&j.stream) catch |err| {
                try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"error\": \"{s}\"}}\n", .{ j.index, r.name, @errorName(err) });
                try w.flush();
                return 1;
            };
            if (store) |st| if (j.entry) |e| st.resumed(e, j.key, !j.stream.reuse_failed);
            try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"prompt_tokens\": {d}, \"cached\": {d}, \"restored\": {d}, \"prefill_s\": {d:.2}}}\n", .{ j.index, r.name, r.prompt.len, j.stream.cached, j.restored, seconds(j.began, io) });
            try w.flush();
        }
        const s0 = m.now();
        if (core.live.items.len > 0) core.step() catch |err| {
            try w.print("{{\"rank\": 0, \"step\": {d}, \"error\": \"{s}\"}}\n", .{ core.steps, @errorName(err) });
            try w.flush();
            return 1;
        };
        step_ns += m.now() - s0;
        if (store) |st| st.pump(2 * std.time.ns_per_ms); // staged writes between rounds, as the lane host's service
        // the finished streams' replies
        for (jobs[0..admitted]) |*j| {
            if (j.done or !j.stream.finished) continue;
            j.done = true;
            finished += 1;
            const reply = j.stream.emitted();
            tokens_out += reply.len;
            const sha = try tokenSha(a, reply);
            const equal = if (j.req.expect_sha) |e| std.mem.eql(u8, e, sha) else true;
            if (!equal) all_equal = false;
            try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"tokens\": {d}, \"reason\": \"{s}\", \"token_sha\": \"{s}\", \"expect_sha\": \"{s}\", \"equal\": {}, \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"min_rows\": {d}, \"seconds\": {d:.2}, \"reply\": [", .{ j.index, j.req.name, reply.len, j.stream.reason.name(), sha, j.req.expect_sha orelse "", equal, j.stream.rounds, j.stream.drafted, j.stream.accepted, j.stream.min_rows, seconds(j.began, io) });
            for (reply, 0..) |t, k| try w.print("{s}{d}", .{ if (k == 0) "" else ", ", t });
            try w.print("]}}\n", .{});
            try w.flush();
            j.stream.deinit(gpa);
            j.proposer.deinit();
            if (j.lineage.len > 0) gpa.free(j.lineage);
        }
    }
    const total_s = seconds(started, io);
    if (ln.prof) |p| {
        const ms = struct {
            fn f(ns: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        try w.print("{{\"rank\": 0, \"profile\": {{\"rounds\": {d}, \"rows_a_round\": {d:.2}, \"step_ms\": {d:.2}, \"send_ms\": {d:.3}, \"enqueue_ms\": {d:.2}, \"forward_ms\": {d:.2}, \"absorb_ms\": {d:.2}, \"logits_ms\": {d:.2}, \"sample_ms\": {d:.2}, \"passes\": {d}, \"streams_a_pass\": {d:.2}, \"pass_ms\": {d:.2}, \"prefills\": {d}, \"prefill_ms\": {d:.1}}}}}\n", .{ p.rounds, ms(p.rows * 1000000, p.rounds), ms(step_ns, core.steps), ms(p.send, p.rounds), ms(p.enqueue, p.rounds), ms(p.forward, p.rounds), ms(p.absorb, p.rounds), ms(p.logits, p.rounds), ms(p.sample, p.rounds), p.passes, ms(p.pass_streams * 1000000, p.passes), ms(p.pass, p.passes), p.prefills, ms(p.prefill, p.prefills) });
        try w.flush();
        try printClock(w, 0, m);
        try printKvShard(w, 0, m);
        if (m.eh) |eh| if (eh.p_calls > 0) {
            const pc: f64 = @floatFromInt(eh.p_calls);
            try w.print("{{\"rank\": 0, \"prefill_engram_ms\": {{\"calls\": {d}, \"rows\": {d:.2}, \"upload\": {d:.2}, \"mm\": {d:.2}, \"exchange\": {d:.2}, \"gate\": {d:.2}}}}}\n", .{ eh.p_calls, @as(f64, @floatFromInt(eh.p_rows)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_upload)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_mm)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_x)) / pc / 1e6, @as(f64, @floatFromInt(eh.p_gate)) / pc / 1e6 });
        };
        if (m.ptimer) |tm| {
            try w.print("{{\"rank\": 0, \"prefill_phases_ms\": {{", .{});
            for (dsv41.round.PhaseTimer.phases, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, if (tm.rounds == 0) 0 else tm.ms[i] / @as(f64, @floatFromInt(tm.rounds)) });
            try w.print("}}, \"chunks\": {d}}}\n", .{tm.rounds});
        }
        if (m.timer) |tm| {
            try w.print("{{\"rank\": 0, \"phases_ms\": {{", .{});
            for (dsv41.round.PhaseTimer.phases, 0..) |name, i| try w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, if (tm.rounds == 0) 0 else tm.ms[i] / @as(f64, @floatFromInt(tm.rounds)) });
            try w.print("}}}}\n", .{});
            try w.flush();
        }
        if (m.eh) |eh| {
            try w.print("{{\"rank\": 0, \"engram\": {{\"calls_a_round\": {d:.2}, \"hash_ms\": {d:.3}, \"read_ms\": {d:.3}, \"decode_ms\": {d:.3}, \"upload_ms\": {d:.3}}}}}\n", .{ ms(eh.calls * 1000000, p.rounds), ms(eh.t_hash, p.rounds), ms(eh.t_read, p.rounds), ms(eh.t_decode, p.rounds), ms(eh.t_upload, p.rounds) });
            try w.flush();
        }
    }
    if (store) |st| {
        const x = st.stats;
        const c = (st.tierCounts() orelse pcache.RankCounts{}).slice();
        const t: pcache.TierCounts = if (c.len > 0) c[0] else .{};
        try w.print("{{\"rank\": 0, \"spill\": {{\"disk_lookups\": {d}, \"disk_hits\": {d}, \"disk_hits_committed\": {d}, \"restores_failed\": {d}, \"restored_tokens\": {d}, \"writes_enqueued\": {d}, \"points\": {d}, \"committed\": {d}, \"cancelled\": {d}, \"failed\": {d}, \"presence_refused\": {d}, \"memory_hits\": {d}, \"pictures_kept\": {d}, \"pictures_refused\": {d}}}}}\n", .{ x.disk_lookups, x.disk_hits, x.disk_hits_committed, x.restores_failed, x.restored_tokens, x.writes_enqueued, t.points, t.writes_committed, t.writes_cancelled, t.writes_failed, t.presence_refused, st.counts.hits, t.pictures_kept, t.pictures_refused });
    }
    try w.print("{{\"rank\": 0, \"requests\": {d}, \"tokens\": {d}, \"steps\": {d}, \"shared_rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"seconds\": {d:.2}, \"all_equal\": {}}}\n", .{ reqs.len, tokens_out, core.steps, core.shared_rounds, core.drafted, core.accepted, total_s, all_equal });
    try w.flush();
    return if (all_equal) 0 else 1;
}

fn seconds(t: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

/// The cache key of a request with pictures (recipe 0006, the server's rule with the rows as the payload): each
/// picture's positions the ids made from the SHA-256 of its rows, and its identity for the tier; the prompt without.
fn cacheKey(a: std.mem.Allocator, prompt: []const u32, images: []const lanes.stream.Image, j: *Job) !void {
    j.key = prompt;
    if (images.len == 0) return;
    const key = try a.dupe(u32, prompt);
    const media = try a.alloc(pcache.Media, images.len);
    for (images, media) |img, *md| {
        if (img.tokens == 0 or @as(usize, img.at) + img.tokens > key.len) return error.BadImageSpan;
        md.* = .{ .start = img.at, .len = img.tokens, .digest = undefined };
        std.crypto.hash.sha2.Sha256.hash(img.bytes, &md.digest, .{});
        for (key[img.at..][0..img.tokens], 0..) |*id, k| id.* = pcache.pictureId(&md.digest, @intCast(k));
    }
    j.key = key;
    j.media = media;
}

/// A request's image spans with their rows read (the files kept for the run).
fn imageRows(a: std.mem.Allocator, io: std.Io, imgs: []const ImageRows, hidden: usize) ![]const lanes.stream.Image {
    const out = try a.alloc(lanes.stream.Image, imgs.len);
    for (imgs, out) |img, *o| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, img.rows_file, a, .limited(1 << 32));
        const row = hidden * 2;
        if ((@as(usize, img.row) + img.tokens) * row > bytes.len) return error.ShortImageRows;
        o.* = .{ .at = img.at, .tokens = img.tokens, .bytes = bytes[img.row * row ..][0 .. img.tokens * row], .rows = true };
    }
    return out;
}

/// The served server's token_sha: sha256 of the reply's ids joined by commas, its first 12 hex digits.
fn tokenSha(a: std.mem.Allocator, ids: []const u32) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    var buf: [16]u8 = undefined;
    for (ids, 0..) |t, k| {
        if (k > 0) try text.append(a, ',');
        try text.appendSlice(a, try std.fmt.bufPrint(&buf, "{d}", .{t}));
    }
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text.items, &h, .{});
    const hex = try std.fmt.allocPrint(a, "{x}", .{&h});
    return hex[0..12];
}

/// TF_DS_KV_SHARD's counters on this rank (a profile): the rounds by decode mode, auto's choices, the rounds whose
/// exchanges rode the RDMA ring, the rows sent, the prompt fetches and carries.
fn printKvShard(w: *std.Io.Writer, rank: u32, m: *const dsv41.model.Model) !void {
    const kv = m.kvs orelse return;
    try w.print("{{\"rank\": {d}, \"kv_shard\": {{\"mode\": \"{s}\", \"ring\": {}, \"overlap\": {}", .{ rank, @tagName(kv.mode), kv.use_ring, kv.overlap });
    inline for (@typeInfo(@TypeOf(kv.stats)).@"struct".field_names) |f| try w.print(", \"{s}\": {d}", .{ f, @field(kv.stats, f) });
    try w.print("}}}}\n", .{});
    try w.flush();
}

/// A profile's round GPU clock gaps (us a round): start to the first stretch, then each stretch and the wait after it.
fn printClock(w: *std.Io.Writer, rank: u32, m: *const dsv41.model.Model) !void {
    if (m.clk_rounds == 0) return;
    const r: f64 = @floatFromInt(m.clk_rounds);
    try w.print("{{\"rank\": {d}, \"round_clock_us\": {{\"rounds\": {d}", .{ rank, m.clk_rounds });
    const names = dsv41.round.PhaseTimer.phases;
    for (0..256) |i| {
        if (m.clk_cnt[i] == 0) continue;
        const name = if (i < names.len) names[i] else switch (i) {
            dsv41.round.Round.clock_wait => "wait",
            dsv41.round.Round.clock_tail => "tail",
            else => "other",
        };
        if (i >= 64 and i < 128) {
            try w.print(", \"L{d}\": [{d:.1}, {d:.1}]", .{ i - 64, @as(f64, @floatFromInt(m.clk_ph[i])) / r / 1e3, @as(f64, @floatFromInt(m.clk_cnt[i])) / r });
            continue;
        }
        try w.print(", \"{s}\": [{d:.1}, {d:.1}]", .{ name, @as(f64, @floatFromInt(m.clk_ph[i])) / r / 1e3, @as(f64, @floatFromInt(m.clk_cnt[i])) / r });
    }
    try w.print("}}}}\n", .{});
    try w.print("{{\"rank\": {d}, \"round_launches\": {{", .{rank});
    var first = true;
    for (m.rd.launches, 0..) |n, i| {
        if (n == 0 or i >= names.len) continue;
        try w.print("{s}\"{s}\": {d:.1}", .{ if (first) "" else ", ", names[i], @as(f64, @floatFromInt(n)) / r });
        first = false;
    }
    try w.print("}}}}\n", .{});
    try w.flush();
}

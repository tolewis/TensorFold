//! DeepSeek-V4.1-Flash behind the native CUDA family interface, tensor parallel over two GPUs: the server's process is
//! rank 0 (the lane core over lanes.zig's Backend on model.zig), rank 1 is `tf-dsv41-lanes` following it on the other
//! node (the same model directory, rank cache, kit, pool and port). The ranks link over TCP and NCCL; the inputs
//! beyond the checkpoint come from the environment:
//!   TF_TP_WORLD (2), TF_TP_MASTER, TF_TP_PORT   the ranks: rank 0's fabric address and port (rank 1 connects)
//!   TF_DS_RANK_CACHE                            the lane's per-rank weight files (the Python lane's TF_DS_RANK_CACHE)
//!   TF_DS_KIT                                   cubins/ (the served extension cubins), the served RoPE tables
//!                                               (rope-*.f32, as many rows as the context) and engram.json
//!   TF_DS_ENGRAM, TF_DS_TOKEN_MAP               the Engram tables and the compressed token map
//!   TF_DS_ARENA_GIB                             a first block of device memory for the buffers (unset: they take what
//!                                               they need, the pool's caches by the context)
//!   TF_DS_ROUND_MS                              the lane core's round costs by rows (ms, comma separated; the
//!                                               served ROUND_MS unless set: a four-node lane's rounds cost less)
//!   TF_DS_ROUND_MS_<n>                          a round of n streams (2 and up) by its rows, in TF_DS_ROUND_MS's form: a row of
//!                                               another stream costs more than a draft row of the same stream, so the draft depth
//!                                               of shared rounds is priced from their own table (unset: TF_DS_ROUND_MS's)
//!   TF_DS_ROUND_ROWS, TF_DS_STREAMS             a round's rows at most (16; the four-node lane's 48) and the pool's
//!                                               streams (4; its 16): --parallel at most the streams
//!   TF_DS_PROMPT_WINDOW_ROWS=<rows>             a prompt chunk under these rows (64 .. 128; 64 unset) takes the MoE decode window
//!                                               in place of the prompt kernels; the scratch grows to these rows. Bits may differ
//!                                               from the prompt kernels' (set alike on every rank: a mismatch ends the start)
//!   TF_DS_PROMPT_GEMM_ROWS=<rows>               a prompt chunk over these rows (16 .. 128; 128 unset) takes the prefill GEMM for the
//!                                               dense linears in place of the glinear group; bits may differ (set alike on every
//!                                               rank: a mismatch ends the start)
//!   TF_DS_PLAIN_ROWS                            the rows a mixed pack's plain (cuBLAS) linear runs over, padded to this
//!                                               count so a token's bits do not change with the round's shape (16; 0: off)
//!   TF_RDMA_DEVICES                             the decode gathers over the RDMA ring on these devices (else NCCL)
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_GRAPH_CAP=N                           (with TF_DS_GRAPHS) the captured stretches kept at most, the least
//!                                               recently replayed destroyed first (0 or unset: all); rank 1's
//!                                               --graph-cap, the same number on both ranks
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//!   TF_DS_SOLO_OVERHEAD=1                       a lone drafted stream's head windows cost the fixed draft allowance
//!   TF_DS_T25_DRAFT_CAL                         a weight in (0, 1]: each stream's draft survival is scaled by its realised over
//!                                               predicted drafts (unset or 0: off; core/lanes/windows.zig)
//!   TF_DS_T25_CAL_LOG                           print every n-th calibrated verify of a stream to stderr (unset or 0: off)
//!   TF_DS_NO_COPIES=1                           a drafted stream proposes no suffix-lookup copy windows
//!   TF_DS_TURN_KEEP=1                           after each prompt pass, keep one more state at the replay cut (even floor of
//!                                               len - window, memory only); the next turn restores it and runs growth + window
//!   TF_DS_SERVE_PROFILE=N                       a measurement: rank 0 prints its backend's times (send, forward, absorb,
//!                                               logits, sample, the drafter pass, the rest of a round) to stderr every N
//!                                               rounds; the forward is synchronized apart, so the rounds are a little slower
//!   TF_DS_PASS_PARTS=1                          with SERVE_PROFILE: split pass ms (host, run, wait, read) and per-part body
//!                                               times (eager body, not the pass graph)
//!   TF_DS_POOL=P, TF_DS_WINDOW=W                (0008, pool.zig) the pool's positions, every stream's extents
//!                                               together (unset: the context), and the window a request may use
//!                                               (rank 0: the context; a follower without it takes rank 0's);
//!                                               rank 1's pool is its --pool. Model.open compares the ranks' pool and
//!                                               window before any request. With TF_DS_KV_SHARD=1 the same pool is cut
//!                                               into one part a rank, each holding the whole window
//!   TF_DS_KV_SHARD=1                            (GLM 0118's port, shard.zig; TP2 only, off by default) each
//!                                               conversation's compressed latents held once, on the rank whose half
//!                                               of the pool its extent lies in; prompts fetch the picked rows from it,
//!                                               rounds too (TF_DS_KV_SHARD_DECODE=fetch, the default) or the owner
//!                                               attends both ranks' heads (=owner). A request then fits a part: the
//!                                               limit is reported (Loaded.window, the log), never silently halved
//! The kernel set (TENSORFOLD_CUDA_KERNELS) is the recorded Triton set (aot_pack.py). The pool holds --context
//! positions for every stream together (each takes an extent: its prompt, its max_tokens and a round's rows).
//! The spill tier (wojo's GLM 0088/0110 port, spill_tier.zig; off unless the server's --spill-dir and --spill-gib):
//! rank 0's switches come through the server (api.Spill: --spill-dir, --spill-gib, --spill-min-free-gib,
//! --spill-min-tokens, --spill-quota, TF_SPILL_TURNS, TF_SPILL_FLUSH_MS), its staging cap from
//! TF_DS_SPILL_STAGING_MIB (128). Each follower runs its own store on its own disk with its own switches
//! (tf-dsv41-lanes --spill-dir/--spill-gib/..., or TF_DS_SPILL_DIR, TF_DS_SPILL_GIB, TF_DS_SPILL_MIN_FREE_GIB,
//! TF_DS_SPILL_MIN_TOKENS, TF_DS_SPILL_QUOTA, TF_DS_SPILL_STAGING_MIB through ENGINE_ENV); the ranks' fingerprints
//! (rank_identity.zig, which needs TF_DS_ENGINE_ID, TF_DS_RECIPE_PATCHES and TF_DS_RANK_CACHE_SHA or
//! TF_DS_MODEL_REVISION on every rank, and TF_DS_ENGRAM_REVISION with Engram) must agree at startup, a difference
//! naming the field. Not with TF_DS_KV_SHARD=1 yet (its owner_sharded layout is round 3); both refusals, and a
//! missing identity, come before the weights load (spillPreflight).
//! TF_TP_WORLD=4: the four nodes of the exact 2D split (prompt2d.zig), ranks 1-3 `tf-dsv41-lanes` following on the
//! other nodes; TF_RDMA_DEVICES (and TF_DS_RDMA_KERNELS: rdma_gather.cu's fatbin, unset: the embedded one) put its
//! decode-size exchanges on the RDMA rings, as the four-node Python lane's TF_RDMA_DEVICES does.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("model.zig");
const rows_mod = @import("prompt_rows.zig");
const pool_mod = @import("pool.zig");
const lanes_mod = @import("lanes.zig");
const round = @import("round.zig");
const link = @import("link.zig");
const rank_cache = @import("rank_cache.zig");
const vision_mod = @import("vision.zig");
const spill_tier = @import("spill_tier.zig");
const rank_wire = @import("rank_wire.zig");
const rank_identity = @import("rank_identity.zig");
const prompt = @import("prompt.zig");
const core = @import("core");
const shard = @import("shard.zig");
const kv_lanes = @import("kv_lanes.zig");
const api = @import("engine_api");
const comm = @import("comm.zig");
const rdma = @import("rdma.zig");

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 2048;
/// The served lane's reasoning effort numbers (text.zig): the server passes each name and the template hears its number.
pub const efforts = @import("text.zig").efforts;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    /// The spill tier (null: off; the served path and its bytes unchanged).
    spill: ?spill_tier.Settings = null,
};

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence.
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    /// Kept prompt states for the host's prompt cache (core/prompt_cache Snapshots: `ptr` and its functions).
    snaps: ?Snaps = null,
    /// Images in requests (the server renders their placeholders; the tower runs on this rank).
    vision: ?lanes.stream.Vision = null,
    /// The single-conversation limit (prompt + reply tokens) when it is not the context asked for: TF_DS_POOL or
    /// TF_DS_KV_SHARD (a part may hold less than the window); null: the context, as ever.
    window: ?usize = null,
    /// The spill tier (the prompt cache's Tier), when Options.spill asked for it.
    tier: ?spill_tier.prompt_cache.Tier = null,
    /// The prompt cache's native rules with the tier (Rules.grid: the chunk; Rules.window: the
    /// replay window, Lanes.usable's).
    grid: u32 = 0,
    replay: u32 = 0,
    /// TF_DS_TURN_KEEP: prompt-cache `Rules.fine_grid` (2); 0 with the switch off.
    fine_grid: u32 = 0,
    /// TF_DS_KV_SHARD: the parts' counters and rows for /health and /metrics (null: the switch off).
    kv_parts: ?lanes.kv_parts.Source = null,
    /// The broken-pair policy's hooks (TF_DS_PAIR_BROKEN, exit by default; null: degrade). `ctx` is `ctx`.
    pair: ?Pair = null,
    /// Whether the model's stream is free of a CUDA graph capture (the lane host's service fence), with `ctx`.
    quiet: ?*const fn (ctx: *anyopaque) bool = null,
};

pub const Snaps = struct {
    ptr: *anyopaque,
    bytes: *const fn (*anyopaque, u32) u64,
    save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
    restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
    drop: *const fn (*anyopaque, *anyopaque) void,
    /// Rows a resumed prompt pass must still run past a kept state (core/prompt_cache Rules.tail).
    tail: u32 = 0,
    /// Whether a kept state still restores (Lanes.usable): given with the spill tier only.
    usable: ?*const fn (*anyopaque, *anyopaque, u32) bool = null,
};

/// The rank's model and the backend over it (what `open` hands the server, freed by `deinit`).
const State = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    lanes: lanes_mod.Lanes,
    vision: ?*vision_mod.Vision = null,
    staging: ?cuda.HostBuffer = null, // the spill tier's pinned staging (allocated before the pool)
    events: []cuda.Event = &.{}, // one a staging slot
    pair_host: ?*anyopaque = null, // the broken-pair policy's report (pairWatch): the lane host and its call
    pair_broken: ?*const fn (host: *anyopaque, why: []const u8, drain: bool) void = null,
    watch: ?link.Watch = null, // the followers' links, watched for a close while the policy is on (pairWatch)
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// An empty variable as none (a container passes TF_RDMA_DEVICES empty when it is unset).
fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// The device bytes this rank's weights take, for the host's memory check: its file in TF_DS_RANK_CACHE (the
/// checkpoint holds both ranks'); 0 when there is none to read (open then says what to set).
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    _ = dir;
    var d = std.Io.Dir.cwd().openDir(io, getenv("TF_DS_RANK_CACHE") orelse return 0, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var buf: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return 0;
    const name = rank_cache.find(fba.allocator(), io, d, 0, world) catch return 0;
    const st = d.statFile(io, name, .{}) catch return 0;
    return st.size;
}

/// Loads rank 0 and links rank 1 (which must start against TF_TP_MASTER:TF_TP_PORT); the pool holds `o.context`
/// positions. Needs `ctx` current.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    if (world != 2 and world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split
    const master = link.parseIp(getenv("TF_TP_MASTER") orelse return error.NoMaster) catch return error.BadMaster;
    const port = std.fmt.parseInt(u16, getenv("TF_TP_PORT") orelse "29620", 10) catch return error.BadPort;
    const arena_gib = if (getenv("TF_DS_ARENA_GIB")) |t| std.fmt.parseInt(usize, t, 10) catch return error.BadArena else 0;
    if (o.spill != null) try spillPreflight(getenvFn); // (before the staging, the weights and the pool)
    const pair_mode = try pairMode(getenvFn); // (a bad value refused before anything loads)
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.gpa = gpa;
    st.staging = null;
    st.events = &.{};
    st.pair_host = null;
    st.pair_broken = null;
    st.watch = null;
    // the spill tier's staging comes out of memory before the pool is sized (PORT-SPILL-V2: budgeted first)
    if (o.spill) |sp| st.staging = try spillStaging(ctx, sp);
    errdefer if (st.staging) |*b| b.free();
    const mo: model.Options = .{
        .model_dir = dir,
        .cache_dir = getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache,
        .kit_dir = getenv("TF_DS_KIT") orelse return error.NoKit,
        .aot_dir = kernels,
        .rank = 0,
        .world = world,
        .master = master,
        .port = port,
        .engram_dir = getenv("TF_DS_ENGRAM"),
        .token_map = getenv("TF_DS_TOKEN_MAP"),
        .rdma_devices = nonEmpty(getenv("TF_RDMA_DEVICES")),
        .rdma_kernels = getenv("TF_DS_RDMA_KERNELS"),
        .graphs = if (getenv("TF_DS_GRAPHS")) |v| !std.mem.eql(u8, v, "0") else false,
        .graph_cap = if (getenv("TF_DS_GRAPH_CAP")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadGraphCap else 0,
        .side = if (getenv("TF_DS_HC_SIDE")) |v| !std.mem.eql(u8, v, "0") else false,
        .prefetch = if (getenv("TF_DS_L2_PREFETCH")) |v| !std.mem.eql(u8, v, "0") else false,
        .engram_aio = if (getenv("TF_DS_ENGRAM_AIO")) |v| !std.mem.eql(u8, v, "0") else false,
        .pool = try pool_mod.resolve(o.context, getenv("TF_DS_POOL")), // may exceed the window: more kept prompts stay resident
        .window = o.context, // (a follower takes TF_DS_WINDOW, else rank 0's: Model.open)
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
        .window_rows = if (getenv(rows_mod.window_env)) |v| rows_mod.parseWindow(v) catch return error.BadPromptWindowRows else rows_mod.default_window,
        .gemm_rows = if (getenv(rows_mod.gemm_env)) |v| rows_mod.parseGemm(v) catch return error.BadPromptGemmRows else rows_mod.default_gemm,
        .plain_rows = if (getenv("TF_DS_PLAIN_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadPlainRows else round.default_rows,
        .round_ms = if (getenv("TF_DS_ROUND_MS")) |v| lanes_mod.parseMs(std.heap.page_allocator, v) catch return error.BadRoundMs else null,
        .streams = if (getenv("TF_DS_STREAMS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadStreams else model.max_streams_default,
    };
    st.m = try model.Model.open(gpa, io, ctx, mo);
    errdefer st.m.close();
    st.lanes = lanes_mod.Lanes.init(gpa, st.m);
    // a start refused past here (a fingerprint mismatch at openPair, a store that does not open) frees the frames the
    // agreement wrote through the lanes (the hello: leaked on the mismatch path before); rank 1 ends on its own
    errdefer st.lanes.wire.deinit(gpa);
    // the pair's one rank_wire Leader, agreed with the full fingerprint when the spill tier or TF_DS_KV_SHARD is on
    // (a follower with another build, kit, model or switches fails the start, named); then the parts' planner over it
    try openPair(gpa, io, st.m, &st.lanes, mo, o.spill);
    errdefer if (st.lanes.rw) |l| {
        gpa.destroy(l);
        st.lanes.rw = null;
    };
    // the spill tier before the parts' planner: rank 0 scans its store while each follower scans its own (opened at
    // the agreement), and startKv's first ask then waits the hello's bound for them (Lanes.startKv)
    if (o.spill) |sp| st.events = try openSpill(gpa, io, st.m, &st.lanes, mo, sp, st.staging.?);
    errdefer if (st.lanes.spill) |t| {
        t.destroy();
        st.lanes.spill = null;
        for (st.events) |*e| e.deinit();
        if (st.events.len > 0) gpa.free(st.events);
        st.events = &.{};
    };
    try st.lanes.startKv();
    errdefer st.lanes.stopKv();
    if (getenv("TF_DS_FIX_K")) |t| st.lanes.fix_k = std.fmt.parseInt(usize, t, 10) catch return error.BadFixK;
    st.lanes.solo_overhead = if (getenv("TF_DS_SOLO_OVERHEAD")) |v| !std.mem.eql(u8, v, "0") else false;
    st.lanes.draft_cal = if (getenv("TF_DS_T25_DRAFT_CAL")) |v| lanes_mod.parseCalWeight(v) catch return error.BadDraftCal else 0.0;
    st.lanes.cal_log = if (getenv("TF_DS_T25_CAL_LOG")) |v| std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return error.BadCalLog else 0;
    st.lanes.no_copies = if (getenv("TF_DS_NO_COPIES")) |v| !std.mem.eql(u8, v, "0") else false;
    st.lanes.turn_keep = if (getenv("TF_DS_TURN_KEEP")) |v| !std.mem.eql(u8, v, "0") else false;
    {
        // TF_DS_ROUND_MS_<n>: a round of n streams by its rows (unset: TF_DS_ROUND_MS's table)
        var n: usize = 2;
        while (n <= model.max_streams) : (n += 1) {
            var name: [32]u8 = undefined;
            const text = std.fmt.bufPrint(name[0 .. name.len - 1], "TF_DS_ROUND_MS_{d}", .{n}) catch unreachable;
            name[text.len] = 0;
            const key: [:0]const u8 = name[0..text.len :0];
            if (nonEmpty(getenv(key))) |v| try st.lanes.setStreamCosts(n, lanes_mod.parseMs(std.heap.page_allocator, v) catch return error.BadRoundMs);
        }
    }
    if (getenv("TF_DS_SERVE_PROFILE")) |v| {
        const every = std.fmt.parseInt(u64, std.mem.trim(u8, v, " "), 10) catch return error.BadServeProfile;
        if (every != 0) {
            st.lanes.prof = .{};
            st.lanes.prof_every = every;
            st.m.prof = true; // the forward and the absorb synchronized apart: a measurement, not a served setting
            if (getenv("TF_DS_PASS_PARTS")) |pp| {
                if (std.mem.eql(u8, std.mem.trim(u8, pp, " "), "1")) st.lanes.pass_parts = true;
            }
        }
    }
    // a vision checkpoint's tower on this rank, from the kit's vision/ (torch's attention cubin, the routing bias)
    st.vision = null;
    if (st.m.cfg.vision) st.vision = try vision_mod.Vision.open(gpa, io, st.m, dir, getenv("TF_DS_KIT").?);
    errdefer if (st.vision) |v| v.close();
    st.lanes.vision = st.vision;
    // the pool is the model's (preallocated): a stream takes no device memory of its own
    const L = lanes_mod.Lanes;
    var snaps: Snaps = .{ .ptr = &st.lanes, .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn, .tail = @intCast(st.m.cfg.window) };
    const window: ?usize = if (st.m.kvs != null or st.m.pool_cap != o.context) st.m.window_cap else null;
    var tier: ?spill_tier.prompt_cache.Tier = null;
    if (o.spill != null) {
        snaps.usable = L.usableFn;
        tier = st.lanes.spill.?.tier();
    }
    const kv_parts: ?lanes.kv_parts.Source = if (st.m.kvs != null) .{ .ctx = &st.lanes, .read = L.kvPartsFn } else null;
    const pair: ?Pair = if (pair_mode == .exit) .{ .watch = pairWatch, .abort = pairAbort } else null;
    if (pair != null) std.log.info("dsv41: a broken rank pair ends this server with status {d} for a restart of both ranks (TF_DS_PAIR_BROKEN=degrade: requests fail one by one instead)", .{api.LaneHost.pair_exit_code});
    if (st.lanes.turn_keep) std.log.info("dsv41: turn keep at the replay cut (TF_DS_TURN_KEEP=1)", .{});
    const fine_grid: u32 = if (st.lanes.turn_keep) 2 else 0;
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .vision = if (st.vision) |v| v.hook() else null, .window = window, .tier = tier, .grid = model.chunk_rows, .replay = @intCast(st.m.cfg.window), .fine_grid = fine_grid, .kv_parts = kv_parts, .pair = pair, .quiet = quietFn };
}

// -- the broken-pair policy (robustness, docs/port/robustness.md) ------------------------------------------------

/// TF_DS_PAIR_BROKEN: what rank 0 does once the pair cannot go on (a follower ended or stopped answering, a KV
/// transfer failed across parts or timed out, the ranks' spill/KV-shard command streams diverged). `exit`, the
/// default (unset or empty): the lane host admits nothing more, fails what waits, lets decoding streams finish within
/// its bound (or fails them when the device stream is wedged), aborts the NCCL communicator and exits with
/// LaneHost.pair_exit_code, so the service's supervisor (systemd) restarts both ranks; the communicator cannot be
/// rebuilt in place, and the stock engine's rank 0 would wait on the dead peer forever. `degrade`: the old behaviour
/// (requests fail one by one, the spill tier stops writing and restoring; the process stays up). Either way nothing
/// acts before a failure: replies are unchanged. A server with one rank has no pair and no policy (dsv41 always runs
/// two or four, TF_TP_WORLD).
pub const PairMode = enum { degrade, exit };

pub fn pairMode(env: *const fn ([:0]const u8) ?[]const u8) !PairMode {
    const v = std.mem.trim(u8, nonEmpty(env("TF_DS_PAIR_BROKEN")) orelse return .exit, " ");
    if (std.mem.eql(u8, v, "exit")) return .exit;
    if (std.mem.eql(u8, v, "degrade")) return .degrade;
    return error.BadPairBroken;
}

/// The errors from a prompt pass or a round that mean the pair broke (rank_wire's PairBroken; kv_lanes' transfers;
/// a follower's StaleExtent/ShardDiverged reach rank 0 as its link going down; an aborted communicator; an RDMA ring
/// that saw a dead peer or was aborted, RingFailed; a decode round's ring order or row lists broken,
/// round.pair_ending: a sized send may be in flight with no receive, the peer spinning in its collect). None of them
/// drains: the next round's collective would wait on the peer.
pub const pair_fatal = [_]api.LaneHost.Fatal{
    .{ .err = error.PairBroken, .drain = false },
    .{ .err = error.TransferTimeout, .drain = false },
    .{ .err = error.TransferFailed, .drain = false },
    .{ .err = error.ShardDiverged, .drain = false },
    .{ .err = error.PeerDown, .drain = false },
    .{ .err = error.CommAborted, .drain = false },
    .{ .err = error.RingFailed, .drain = false },
    .{ .err = error.RingOrder, .drain = false },
    .{ .err = error.ShardRingOrder, .drain = false },
    .{ .err = error.BadRingRows, .drain = false },
    .{ .err = error.EngramRankFault, .drain = false },
};

/// Loaded.pair: the hooks the server wires to its lane host (native/cuda.zig).
pub const Pair = struct {
    /// The family's own breaks reach `broken(host, ...)` from now on: the spill tier's on_broken and the KV
    /// planner's on_end (call it once, after the lane host is set up and before it starts).
    watch: *const fn (ctx: *anyopaque, host: *anyopaque, broken: *const fn (host: *anyopaque, why: []const u8, drain: bool) void) void,
    /// Rank 0's NCCL communicator aborted (comm.zig, once): any stream wedged behind an unmatched call drains.
    abort: *const fn (ctx: *anyopaque) void,
    fatal: []const api.LaneHost.Fatal = &pair_fatal,
};

fn pairWatch(ctx: *anyopaque, host: *anyopaque, broken: *const fn (*anyopaque, []const u8, bool) void) void {
    const st: *State = @ptrCast(@alignCast(ctx));
    st.pair_host = host;
    st.pair_broken = broken;
    if (st.lanes.spill) |t| {
        t.on_broken = spillBroken;
        t.broken_ctx = st;
    }
    if (st.lanes.kv) |k| k.on_end = .{ .ctx = st, .call = kvEnded };
    // a follower that died (killed, crashed) closes its link at once: the pair breaks now, not at the stall bound
    st.watch = link.Watch.init(st.m.followers(), st, linkLost);
    st.watch.?.start() catch |e| {
        std.log.warn("dsv41: no watch over the followers' links ({s}): a dead follower is found by the stall monitor", .{@errorName(e)});
        st.watch = null;
    };
}

/// A follower's link closed (link.Watch, its thread): the process on the other end is gone, so every collective is
/// stopped at once (a round blocked on the peer returns) and the pair is broken; nothing drains.
fn linkLost(ctx: *anyopaque, follower: usize) void {
    pairAbort(ctx);
    const st: *State = @ptrCast(@alignCast(ctx));
    var wbuf: [160]u8 = undefined;
    report(st, linkLostWhy(&wbuf, follower), false);
}

/// The broken pair's cause when follower `i`'s link closed.
pub fn linkLostWhy(buf: []u8, follower: usize) []const u8 {
    return std.fmt.bufPrint(buf, "rank {d}'s link to rank 0 closed (its process ended or the connection reset)", .{follower + 1}) catch "a follower's link closed";
}

/// Every collective of this rank stopped (comm.abortAll): the RDMA rings' abort word (TP2's decode gathers run on
/// the ring whenever TF_RDMA_DEVICES is set, and ncclCommAbort does not release a collect kernel spinning on a dead
/// peer's flags), then NCCL. Any thread, once each.
fn pairAbort(ctx: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ctx));
    var buf: [3]*rdma.Ring = undefined;
    _ = comm.abortAll(&st.m.comm, st.m.rdmaRings(&buf), "rank pair broken");
}

fn report(st: *State, why: []const u8, drain: bool) void {
    if (st.pair_broken) |f| f(st.pair_host.?, why, drain);
}

/// The spill tier's agreement broke (it logged the rank and cause, and stops writing and restoring): rounds do not
/// depend on it, so decoding streams may drain (the lane host's deadline bounds a round stuck on a dead peer).
fn spillBroken(ctx: ?*anyopaque, err: anyerror) void {
    const st: *State = @ptrCast(@alignCast(ctx.?));
    var buf: [192]u8 = undefined;
    var wbuf: [224]u8 = undefined;
    const lead = if (st.lanes.spill) |t| t.lead else null;
    const cause = if (lead) |l| (if (l.broken) |b| rank_wire.describeBroken(b, &buf) else @errorName(err)) else @errorName(err);
    report(st, std.fmt.bufPrint(&wbuf, "the spill tier's agreement: {s}", .{cause}) catch cause, true);
}

/// Rank 0 ended the pair for a KV transfer of its own (kv_lanes.Kv.endPair): its stream may hold an unmatched NCCL
/// half, so the communicator is aborted at once and nothing drains.
fn kvEnded(ctx: *anyopaque, err: anyerror) void {
    pairAbort(ctx);
    const st: *State = @ptrCast(@alignCast(ctx));
    var wbuf: [160]u8 = undefined;
    report(st, std.fmt.bufPrint(&wbuf, "a KV transfer between the ranks: {s}", .{@errorName(err)}) catch @errorName(err), false);
}

/// Loaded.quiet: no CUDA graph capture in progress on the model's stream (a driver error reads as quiet: the
/// fence then rests on the lane host's own round and chunk rules).
fn quietFn(ctx: *anyopaque) bool {
    const st: *State = @ptrCast(@alignCast(ctx));
    const status = cuda.graph.captureStatus(st.m.stream) catch return true;
    return status == .none;
}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    if (st.watch) |*w| w.end(); // (before the links close: rank 1 ending after `done` is no break)
    // the tier after the prompt cache's store (cuda.zig frees it first) and before the family: its store closes
    if (st.lanes.spill) |t| t.destroy();
    st.lanes.spill = null;
    for (st.events) |*e| e.deinit();
    if (st.events.len > 0) st.gpa.free(st.events);
    if (st.staging) |*b| b.free();
    st.lanes.deinit(); // rank 1 hears done
    if (st.vision) |v| v.close();
    st.m.close();
    st.gpa.destroy(st);
}

// -- the spill tier's startup (rank 0 here, the followers in tf-dsv41-lanes) --------------------------------------

/// What refuses the spill tier from the environment alone, checked on every rank before anything loads (minutes on
/// the Sparks): the pair (TF_TP_WORLD=2), a TF_DS_KV_SHARD setting it cannot read, and the identity the fingerprint
/// needs (rank_identity.Extra.env: TF_DS_ENGINE_ID, TF_DS_RECIPE_PATCHES, TF_DS_RANK_CACHE_SHA or
/// TF_DS_MODEL_REVISION; TF_DS_ENGRAM_REVISION with TF_DS_ENGRAM). TF_DS_KV_SHARD=1 runs with it (round 3: the
/// owner_sharded layout, part-salted lineage, restores into the recorded part, one Leader for the pair).
pub fn spillPreflight(env: *const fn ([:0]const u8) ?[]const u8) !void {
    const world = std.fmt.parseInt(u32, env("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    if (world != 2) return error.SpillNeedsTwoRanks;
    _ = shard.Mode.parse(env("TF_DS_KV_SHARD"), env("TF_DS_KV_SHARD_DECODE")) catch return error.BadKvShard;
    _ = try rank_identity.Extra.env(env);
    if (nonEmpty(env("TF_DS_ENGRAM")) != null and (env("TF_DS_ENGRAM_REVISION") orelse @as([]const u8, "")).len == 0) return error.NoEngramId;
}

/// The spill tier's switches from the server's (rank 0; api.Spill's fields) and TF_DS_SPILL_STAGING_MIB.
pub fn spillSettings(dir: []const u8, gib: f64, min_free_gib: f64, quota: f64, min_tokens: u32, turns: u32, flush_ms: u32) !spill_tier.Settings {
    var s: spill_tier.Settings = .{ .dir = dir, .gib = gib, .min_free_gib = min_free_gib, .quota = quota, .min_tokens = min_tokens, .turns = turns, .flush_ms = flush_ms };
    if (getenv("TF_DS_SPILL_STAGING_MIB")) |v| s.staging_mib = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return error.BadSpillSetting;
    if (s.staging_mib < spill_tier.min_staging_mib) return error.BadSpillSetting;
    s.image_cache = spill_tier.Settings.imageCache(getenv("TF_DS_IMAGE_CACHE")); // (pictures' points: the fingerprint's reading)
    return s;
}

/// The pinned staging a rank's spill tier works in (its cap; allocated before the model's pool).
pub fn spillStaging(ctx: *const cuda.Context, s: spill_tier.Settings) !cuda.HostBuffer {
    var b = try cuda.HostBuffer.alloc(ctx.d, @intCast(s.stagingBytes()));
    if (@intFromPtr(b.bytes.ptr) % 4096 != 0) {
        b.free();
        return error.UnalignedStaging;
    }
    return b;
}

fn stagingSlice(b: cuda.HostBuffer) []align(4096) u8 {
    return @alignCast(b.bytes);
}

pub fn getenvFn(name: [:0]const u8) ?[]const u8 {
    return getenv(name);
}

/// This rank's fingerprint exchange (rank_identity's inputs: the model options it opened with, the opened model, the
/// recipe's identity from the environment, its switches and its state layout). Without the spill tier (`s` null:
/// TF_DS_KV_SHARD alone) nothing persists, so a missing recipe identity reads "unset" (Extra.envFor).
pub fn rankExchange(gpa: std.mem.Allocator, io: std.Io, mo: model.Options, m: *model.Model, s: ?spill_tier.Settings, layout: *const model.Model.spill.StateLayout) !@import("fingerprint.zig").Exchange {
    const envp = try rank_identity.Extra.envFor(getenvFn, s != null);
    var fbuf: [256]u8 = undefined;
    const image_cache = spill_tier.Settings.imageCache(getenv("TF_DS_IMAGE_CACHE"));
    // TF_DS_KV_SHARD's switches every rank's planner takes (carry, twins) and its part size: agreed in `features`
    const kvs: kv_lanes.Settings = if (m.kvs != null) try kv_lanes.Settings.fromEnv(getenvFn) else .{ .carry = false, .twins = false };
    const part_rows: u64 = if (m.caches.shard) |x| x.part else 0;
    const features = try rank_identity.featureString(&fbuf, .{ .spill = s != null, .spill_min_tokens = if (s) |x| x.min_tokens else 0, .kv_shard = m.kvs != null, .kv_carry = kvs.carry, .twins = kvs.twins, .part_rows = part_rows, .image_cache = image_cache, .streams = @intCast(m.streams), .round_rows = @intCast(m.eng.round_rows), .pool = m.pool_cap, .drafts = m.drafting() });
    var pbuf: [model.Model.spill.StateLayout.max_encoded]u8 = undefined;
    const lp = try rank_identity.layoutPart(layout, m.world, &pbuf);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const rc_name = try rank_cache.find(arena.allocator(), io, m.cache_dir, m.rank, m.world);
    const x: rank_identity.Extra = .{
        .engine_id = envp.engine_id,
        .recipe_patches = envp.recipe_patches,
        .build_flags = envp.build_flags,
        .model_revision = envp.model_revision,
        .ablation = envp.ablation,
        .rank_cache_sha = envp.rank_cache_sha,
        .engram_revision = envp.engram_revision,
        .features = features,
        // the store's namespace version, not its newest format: version-3 points (pictures) came with no change to a
        // version-2 text point, so a v3 reader shares round 2's namespace and its text points (a v2 reader drops v3
        // manifests as foreign); a format change that breaks old points bumps namespace_version
        .native_format = core.spill.format.namespace_version,
        .chunk_rows = model.chunk_rows,
        .replay_floor = prompt.replay_floor,
        .idx_split = prompt.idx_split,
        .planes = lp.planes,
        .rank_layout = lp.rank_layout,
        .owner_layout = lp.owner_layout,
        .owner_version = lp.owner_version,
        .parts = lp.parts,
        .persist = s != null,
        .gpu_capability = try m.ctx.capability(),
        .cuda_driver = try m.ctx.d.version(),
        .rank_cache_name = rc_name,
    };
    var g = try rank_identity.gather(gpa, io, mo, m, x);
    defer g.deinit();
    const fp = try g.build(gpa);
    return fp.exchange();
}

/// One reusable event a staging slot (spill_tier.eventCount).
pub fn spillEvents(gpa: std.mem.Allocator, m: *model.Model, n: usize) ![]cuda.Event {
    const evs = try gpa.alloc(cuda.Event, n);
    var made: usize = 0;
    errdefer {
        for (evs[0..made]) |*e| e.deinit();
        gpa.free(evs);
    }
    for (evs) |*e| {
        e.* = try cuda.Event.init(m.ctx.d, false);
        made += 1;
    }
    return evs;
}

fn agreeSink(ctx: *anyopaque, sub: []const u8) anyerror!void {
    return lanes_mod.Lanes.spillFrame(ctx, rank_wire.lanes_kind, sub);
}

/// Rank 0: the pair's one rank_wire Leader (Lanes.rw) when the spill tier (`s`) or TF_DS_KV_SHARD is on, agreed with
/// every follower before anything spill- or shard-related crosses. With the spill tier the hello carries this rank's
/// full fingerprint (rankExchange: build, kit, model, tokenizer, geometry, the owner layout with its parts and part
/// size, the switches); TF_DS_KV_SHARD alone keeps round 2's shard-only hello (kv_lanes.shardExchange: nothing it did
/// not need before, no spill layout, no recipe identity). The spill tier (openSpill) and the KV parts
/// (Lanes.startKv) both post through it, so one Tracker orders every frame of the pair. Nothing with both off (no
/// frame, no fingerprint: the served bytes unchanged) or one rank.
pub fn openPair(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, ln: *lanes_mod.Lanes, mo: model.Options, s: ?spill_tier.Settings) !void {
    if (ln.rw != null or m.world < 2 or (s == null and m.kvs == null)) return;
    if (s != null and m.world != 2) return error.SpillNeedsTwoRanks;
    const fx = if (s != null) blk: {
        const layout = try m.spillLayout();
        // the planner's carried bytes a token are the layout's owner-only bytes (spill_state, from the loaded
        // weights), the rows each rank stores of a point in its own part
        if (m.kvs != null and layout.owner_token_bytes != lanes_mod.kvConfig(m).carry_bytes) return error.SpillCarryMismatch;
        break :blk try rankExchange(gpa, io, mo, m, s, &layout);
    } else try shardOnlyExchange(m);
    const lead = try gpa.create(rank_wire.Leader);
    errdefer gpa.destroy(lead);
    lead.* = rank_wire.Leader.init(@intCast(m.world), rank_wire.freshEpoch(io), fx);
    _ = lead.agree(.{ .ctx = ln, .send = agreeSink }, m.followers()) catch |e| {
        var buf: [192]u8 = undefined;
        if (lead.broken) |b| std.log.err("dsv41: {s}", .{rank_wire.describeBroken(b, &buf)});
        // a follower without the tier ends at the hello (SpillNotStarted on its side): rank 0 sees its link go down
        if (s != null and e != error.FingerprintMismatch) std.log.err("dsv41 spill: no agreement with rank 1; if it printed \"follow\": \"SpillNotStarted\", start it with the spill tier too (TF_DS_SPILL_DIR and TF_DS_SPILL_GIB)", .{});
        return if (e == error.FingerprintMismatch and s != null) error.SpillFingerprintMismatch else e;
    };
    ln.rw = lead;
}

/// Rank 0's hello for TF_DS_KV_SHARD alone: round 2's (kv_lanes.shardExchange over the planner's config with this
/// environment's switches, as Kv.init and the follower's kv_lanes.Follower.exchange take them).
fn shardOnlyExchange(m: *model.Model) !@import("fingerprint.zig").Exchange {
    const settings = try kv_lanes.Settings.fromEnv(getenvFn);
    var c = lanes_mod.kvConfig(m);
    c.carry = settings.carry;
    c.twins = settings.twins;
    return kv_lanes.shardExchange(0, m.world, c, settings);
}

/// Rank 0: this rank's store under <spill-dir>/native-spill-<compat>/rank0 and the tier on `ln` (ln.spill), over the
/// pair's agreed Leader (openPair: made here when it has not run). TF_DS_KV_SHARD: an owner_sharded store (its parts'
/// block sizes, part-salted names, restores into the recorded part through the planner). The events it returns are
/// the caller's (freed after the tier).
pub fn openSpill(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, ln: *lanes_mod.Lanes, mo: model.Options, s: spill_tier.Settings, staging: cuda.HostBuffer) ![]cuda.Event {
    if (m.world != 2) return error.SpillNeedsTwoRanks;
    try openPair(gpa, io, m, ln, mo, s);
    const lead = ln.rw orelse return error.PairNotAgreed;
    const a = lead.agreement orelse return error.PairNotAgreed;
    const layout = try m.spillLayout();
    const plan = try spill_tier.stagingPlan(s, &layout);
    const wire = ln.spillWire();
    const events = try spillEvents(gpa, m, spill_tier.eventCount(plan));
    errdefer {
        for (events) |*e| e.deinit();
        gpa.free(events);
    }
    var gens = try m.spillGenerations(gpa);
    const compat = spill_tier.storeCompat(a, 0, &layout);
    const max_tokens: u32 = @intCast(if (m.request_cap > 0) m.request_cap else m.pool_cap);
    const side = lanes_mod.SpillTier.S.create(gpa, io, .{ .m = m }, layout, gens, s, 0, compat, max_tokens, stagingSlice(staging), events, .{}) catch |e| {
        gens.deinit(gpa);
        return e;
    };
    var boot: [16]u8 = undefined;
    io.random(&boot);
    const t = lanes_mod.SpillTier.create(gpa, side, ln.spillFamily(), wire, lead, s, boot) catch |e| {
        side.destroy();
        return e;
    };
    t.own_lead = false; // (Lanes.rw, shared with the KV parts)
    // rank 1 scans its store on its own thread since the agreement (rank 0's scan above ran meanwhile): waited for
    // while its scan makes progress, before any spill command or the KV parts' start
    t.awaitFollowers(null, 50 * std.time.ns_per_ms) catch |e| {
        t.destroy();
        return e;
    };
    ln.spill = t;
    std.log.info("dsv41 spill: rank 0's store under {s} ({d:.1} GiB, staging {d} MiB); the ranks agreed", .{ s.dir, s.gib, s.stagingBytes() >> 20 });
    return events;
}

/// A follower's spill side, made before its model's pool (staging) is sized and opened at rank 0's agreement. The
/// staging and events stay the caller's.
pub fn followerSpill(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, mo: model.Options, s: spill_tier.Settings, staging: cuda.HostBuffer, events: *[]cuda.Event) !*lanes_mod.SpillFollower {
    const layout = try m.spillLayout(); // (TF_DS_KV_SHARD: owner_sharded, this rank's part its own)
    if (m.kvs != null and layout.owner_token_bytes != lanes_mod.kvConfig(m).carry_bytes) return error.SpillCarryMismatch;
    const plan = try spill_tier.stagingPlan(s, &layout);
    const fx = try rankExchange(gpa, io, mo, m, s, &layout);
    events.* = try spillEvents(gpa, m, spill_tier.eventCount(plan));
    errdefer {
        for (events.*) |*e| e.deinit();
        gpa.free(events.*);
        events.* = &.{};
    }
    var gens = try m.spillGenerations(gpa);
    errdefer gens.deinit(gpa);
    const max_tokens: u32 = @intCast(if (m.request_cap > 0) m.request_cap else m.pool_cap);
    // (no family yet: lanes.followWith attaches its kept states)
    return lanes_mod.SpillFollower.create(gpa, io, @intCast(m.rank), fx, null, .{ .dev = .{ .m = m }, .layout = layout, .gens = gens, .settings = s, .max_tokens = max_tokens, .staging = stagingSlice(staging), .events = events.* });
}

/// A request or a start this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    if (rank_wire.explain(err)) |t| return t; // the ranks' agreement (TF_DS_KV_SHARD's frames, the spill tier's)
    return switch (err) {
        error.NotPortedYet => "the native DeepSeek-V4.1 engine runs tensor parallel over two GPUs or the four nodes of the 2D split: set TF_TP_WORLD=2 or 4",
        error.NoMaster, error.BadMaster => "set TF_TP_MASTER to rank 0's fabric address (rank 1: tf-dsv41-lanes MODEL CACHE 1 2 MASTER PORT KIT; ranks 1-3 of four: ... G 4 ...)",
        error.NoRdmaKernels => "TF_RDMA_DEVICES at world 4 needs TF_DS_RDMA_KERNELS (rdma_gather.cu's fatbin) in a build without the kernel images",
        error.NoRankCache => "set TF_DS_RANK_CACHE to the lane's per-rank weight files",
        error.NoKit => "set TF_DS_KIT to the kernel kit (cubins/, the RoPE tables, engram.json)",
        error.NoFreeSlot => "this engine serves TF_DS_STREAMS streams at once (default 4): --parallel at most that",
        error.BadPromptWindowRows => rows_mod.window_hint,
        error.BadPromptGemmRows => rows_mod.gemm_hint,
        error.PromptRowsMismatch => "TF_DS_PROMPT_WINDOW_ROWS / TF_DS_PROMPT_GEMM_ROWS: the ranks run different prompt paths; set them alike on every rank (tf-dsv41-lanes reads them too)",
        error.BadServeProfile => "TF_DS_SERVE_PROFILE: the rounds between two profile lines, a whole number (0: off)",
        error.BadRoundRows => "TF_DS_ROUND_ROWS: a round's rows at most, 16 .. 48",
        error.BadGraphCap => "TF_DS_GRAPH_CAP: the captured CUDA graphs kept at most, a count (0: all)",
        error.BadPlainRows => "TF_DS_PLAIN_ROWS: the rows a mixed pack's plain linear runs over, 0 (the round's own) or a count up to TF_DS_ROUND_ROWS",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 16",
        error.ContextFull => "the streams' prompts and budgets fill the context's pool: lower max_tokens, or retry when a stream ends",
        error.PromptTooLong => "the prompt and its max_tokens do not fit the context window",
        error.NoVisionKit => "this checkpoint reads images: its kit (TF_DS_KIT) needs vision/torch_fmha_sm120.cubin and vision/gate_bias_vl.safetensors",
        error.BadPool => "TF_DS_POOL (rank 1: --pool): the pool's positions, a whole number of at least the context (unset: the context)",
        error.BadWindow => "TF_DS_WINDOW: the positions one request may use, a whole number; on rank 0 it must be --context (start.sh sets both from CONTEXT)",
        error.BadKvShard => "TF_DS_KV_SHARD: 1 (on) or 0 (off)",
        error.BadKvShardDecode => "TF_DS_KV_SHARD_DECODE: fetch (the default), owner or auto (per round by TF_DS_KV_SHARD_COST)",
        error.KvShardNeedsTwoRanks => "TF_DS_KV_SHARD=1 shards the compressed latents over the two TP2 ranks: TF_TP_WORLD=2",
        error.KvShardMismatch => "the ranks disagree on TF_DS_KV_SHARD, the pool (TF_DS_POOL, else rank 1's --pool), TF_DS_ROUND_ROWS or the window, or the other rank could not read one of them: give every rank the same TF_DS_* environment (start.sh's ENGINE_ENV) and settings",
        error.BadPart => "TF_DS_KV_SHARD=1: a round row whose part is not its extent's (a placement bug): restart with TF_DS_KV_SHARD=0",
        error.BadRound => "a round row with a malformed extent or position (an engine bug); with TF_DS_KV_SHARD=1, restart with TF_DS_KV_SHARD=0",
        error.WindowTooWide => "TF_DS_KV_SHARD=1: a round of more rows than the shard's trash rows (TF_DS_ROUND_ROWS above 48): lower TF_DS_ROUND_ROWS or set TF_DS_KV_SHARD=0",
        error.FetchTooLarge => "TF_DS_KV_SHARD=1: a prompt chunk's fetched rows exceed the fetch buffer (an engine bug): restart with TF_DS_KV_SHARD=0",
        error.BadFetch, error.BadSelect, error.NoKvShard => "TF_DS_KV_SHARD=1: the sharded attention's buffers or kernels are not set up as it needs (an engine bug): restart with TF_DS_KV_SHARD=0",
        error.UnevenWoA => "TF_DS_KV_SHARD_DECODE=owner needs every layer's wo_a of one width: use TF_DS_KV_SHARD_DECODE=fetch",
        error.PoolNotWholeParts => "TF_DS_KV_SHARD=1: the pool (TF_DS_POOL, else the context) must be whole 4,096-position parts (two parts of 2,048-position blocks)",
        error.PartTooSmall => "TF_DS_KV_SHARD=1 cuts the pool (TF_DS_POOL) into one part a rank, and a conversation fits one part: each part must hold the window and a round's rows and two, in whole 2,048-position blocks (2,101,248 at the 1M window): raise TF_DS_POOL or lower --context",
        error.CrossesPart => "TF_DS_KV_SHARD=1: an extent that crosses a part (a placement bug): restart with TF_DS_KV_SHARD=0",
        error.PairNotAgreed => "the spill tier or TF_DS_KV_SHARD=1 started without the ranks' agreement (an engine bug): restart with both off",
        error.SpillNoPart => "the spill tier with TF_DS_KV_SHARD=1 found a point without its part (an engine bug, or a store written by another build): restart with --spill-dir unset or TF_DS_KV_SHARD=0",
        error.SpillCarryMismatch => "the spill tier with TF_DS_KV_SHARD=1: the KV planner's carried bytes a token differ from the spill layout's owner-only rows (an engine bug): restart with --spill-dir unset or TF_DS_KV_SHARD=0",
        error.SpillLayoutMismatch => "the spill state layout does not match the loaded model or its allocation (an engine bug)",
        error.SpillSelfCheckSharded => "the spill self-check reads the replicated planes: run it with TF_DS_KV_SHARD=0 (sharded packing is checked by the restored token_sha)",
        error.SpillNeedsTwoRanks => "the spill tier runs on the TP2 pair (TF_TP_WORLD=2)",
        error.SpillFingerprintMismatch => "the ranks' configurations differ (the log names the rank and field): start every rank from the same image, model, kit and TF_DS_* settings, with the spill tier on both (rank 1: TF_DS_SPILL_DIR/TF_DS_SPILL_GIB)",
        error.SpillNotStarted => "rank 0 runs the spill tier (--spill-dir/--spill-gib): start this rank with it too (TF_DS_SPILL_DIR and TF_DS_SPILL_GIB, or tf-dsv41-lanes --spill-dir/--spill-gib)",
        error.BadSpillSetting => "TF_DS_SPILL_STAGING_MIB: the spill staging cap in MiB, at least 16 (default 128)",
        error.StagingTooSmall => "the spill staging cap (TF_DS_SPILL_STAGING_MIB) holds too few slots for this model's blocks and small state: raise it",
        error.BadKvShardSetting => "TF_DS_KV_SHARD_CARRY and TF_DS_KV_SHARD_TWINS: 1 (on, the default) or 0; TF_DS_KV_SHARD_CHECK: decode rounds between the ranks' planner checks (0: at start only); TF_DS_KV_SHARD_STATS_MS: 0 (the default, rank 0's own kv_shard row only) up to 3600000 ms between asks for the followers' rows (only while /health or /metrics read them)",
        error.ShardLayoutMismatch => "TF_DS_KV_SHARD=1: the ranks cut the pool into different parts or plan with other switches (another TF_DS_POOL, --pool, TF_DS_KV_SHARD_CARRY, TF_DS_KV_SHARD_TWINS or build on rank 1): give every rank the same TF_DS_* environment and build",
        error.KvShardSettingsMismatch => "TF_DS_KV_SHARD=1: the ranks disagree on the decode exchanges' transport or cost model (TF_DS_KV_SHARD_RING, TF_DS_KV_SHARD_COST, or one rank has the RDMA ring, or the ring kernels in its dsv41_shard image, and the other not), or the other rank could not read one of them or set its decode buffers up (its log names why): give every rank the same TF_DS_* environment, RDMA setting and image",
        error.BadKvShardRing => "TF_DS_KV_SHARD_RING must be 1 (the decode exchanges ride the RDMA ring when there is one; the default) or 0 (NCCL)",
        error.BadKvShardCost => "TF_DS_KV_SHARD_COST must be nine non-negative integers, fetch_fixed,fetch_row,owner_fixed,owner_row,pass_fixed,pass_row,mask_row,copy_fixed,copy_kib (ns; docs/port/shard-ring.md), or unset",
        error.ShardRingOrder, error.BadRingRows, error.RingOrder => "TF_DS_KV_SHARD=1: a decode round's ring transfers left their fixed order or row lists (an engine bug; the ranks would part, so the pair ends): restart with TF_DS_KV_SHARD_RING=0 (NCCL), or TF_DS_KV_SHARD=0",
        error.ShardDiverged, error.StaleExtent => "TF_DS_KV_SHARD=1: the ranks' KV-part planners disagree (an engine bug; the pair stops rather than read the wrong rows): restart with TF_DS_KV_SHARD=0",
        error.TransferFailed => "TF_DS_KV_SHARD=1: a KV transfer between the ranks failed on one rank only: restart; if it recurs, TF_DS_KV_SHARD_CARRY=0 or TF_DS_KV_SHARD=0",
        error.TransferTimeout => "TF_DS_KV_SHARD=1: a KV transfer did not complete in time (a rank's half never ran, or a device fault; the GPU stream holds the unmatched call): restart; if it recurs, TF_DS_KV_SHARD_CARRY=0 or TF_DS_KV_SHARD=0",
        error.KvShardNotStarted, error.BadPlan => "TF_DS_KV_SHARD=1: the KV-part planner was not set up as the lanes need (an engine bug): restart with TF_DS_KV_SHARD=0",
        error.FollowerScanStalled => "rank 1's spill store scan stopped making progress (its log names the store): check its spill disk (TF_DS_SPILL_DIR), or start it with a fresh directory",
        error.BadPairBroken => "TF_DS_PAIR_BROKEN: exit (the default: a broken rank pair ends the server with status 75 for its supervisor to restart both ranks) or degrade (requests fail one by one, the server stays up)",
        error.CommAborted => "the ranks' NCCL communicator was aborted after a broken pair (a transfer that never completed): the server restarts both ranks",
        error.RingFailed => "the ranks' RDMA ring failed (the peer stopped answering, or the ring was aborted after a broken pair): the server restarts both ranks",
        error.EngramRankFault => "a prompt chunk's Engram rows could not be read on one rank (its disk or memory failed; the other rank waits for it in the layer's collective, so the pair ends): the server restarts both ranks; if it recurs, check the Engram tables' disk",
        else => null,
    };
}

test "refusals name what to set" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoMaster).?, "TF_TP_MASTER") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.PartTooSmall).?, "TF_DS_POOL") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.KvShardMismatch).?, "--pool") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadPromptWindowRows).?, "TF_DS_PROMPT_WINDOW_ROWS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadPromptGemmRows).?, "TF_DS_PROMPT_GEMM_ROWS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.PromptRowsMismatch).?, "TF_DS_PROMPT_GEMM_ROWS") != null);
    for ([_]anyerror{ error.BadPart, error.WindowTooWide, error.FetchTooLarge, error.BadFetch, error.NoKvShard, error.CrossesPart, error.ShardDiverged, error.StaleExtent, error.TransferFailed, error.TransferTimeout, error.KvShardNotStarted, error.BadPlan, error.ShardRingOrder, error.BadRingRows }) |err|
        try std.testing.expect(std.mem.indexOf(u8, explain(null, err).?, "TF_DS_KV_SHARD=0") != null);
    for ([_]anyerror{ error.KvShardSettingsMismatch, error.BadKvShardRing }) |err|
        try std.testing.expect(std.mem.indexOf(u8, explain(null, err).?, "TF_DS_KV_SHARD_RING") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadKvShardCost).?, "TF_DS_KV_SHARD_COST") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadKvShardDecode).?, "auto") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadKvShardDecode).?, "owner") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.RingOrder).?, "TF_DS_KV_SHARD_RING") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
}

test "TF_DS_PAIR_BROKEN: exit by default (spill and KV sharding off), degrade the opt-out, a bad value refused" {
    const E = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_DS_PAIR_BROKEN")) return v;
            if (std.mem.eql(u8, name, "TF_DS_KV_SHARD")) return "0";
            return null; // (no TF_DS_SPILL_DIR; pairMode takes nothing of the spill tier either)
        }
    };
    try std.testing.expectEqual(PairMode.exit, try pairMode(E.get)); // (unset)
    E.v = "";
    try std.testing.expectEqual(PairMode.exit, try pairMode(E.get)); // (empty: unset)
    E.v = " exit ";
    try std.testing.expectEqual(PairMode.exit, try pairMode(E.get));
    E.v = "degrade";
    try std.testing.expectEqual(PairMode.degrade, try pairMode(E.get));
    for ([_][]const u8{ "restart", "Exit", "0", "1" }) |bad| {
        E.v = bad;
        try std.testing.expectError(error.BadPairBroken, pairMode(E.get));
    }
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadPairBroken).?, "degrade") != null);
    for (pair_fatal) |f| try std.testing.expect(!f.drain);
}

test "a round's pair-ending errors end the pair on rank 0 (none drains) and are named" {
    // the ring-order refusals and BadRingRows leave a sized send or a peer's collect pending: a refusal that
    // only failed the request would stop every later sharded round (RowPairing open) with the server still up
    for (round.pair_ending) |err| {
        var found = false;
        for (pair_fatal) |f| if (f.err == err) {
            found = true;
            try std.testing.expect(!f.drain);
        };
        try std.testing.expect(found);
        try std.testing.expect(explain(null, err) != null);
    }
    for ([_]anyerror{ error.RingOrder, error.ShardRingOrder, error.BadRingRows, error.RingFailed }) |err| try std.testing.expect(round.endsPair(err));
    // a prompt chunk's Engram rows that one rank could not read: the peer waits in the layer's allGather
    try std.testing.expect(round.endsPair(error.EngramRankFault));
    try std.testing.expect(!round.endsPair(error.ReadFailed));
    try std.testing.expect(!round.endsPair(error.OutOfMemory));
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.ReadFailed));
    try std.testing.expect(!round.endsPair(error.NoFreeSlot));
    try std.testing.expect(!round.endsPair(error.ContextFull));
}

test "the spill tier's refusals from the environment come before anything loads, and each says what to set" {
    const E = struct {
        var kv: ?[]const u8 = null;
        var world: ?[]const u8 = null;
        var weights: ?[]const u8 = "rev-1";
        var engram: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_DS_ENGINE_ID")) return "engine";
            if (std.mem.eql(u8, name, "TF_DS_RECIPE_PATCHES")) return "patches";
            if (std.mem.eql(u8, name, "TF_DS_MODEL_REVISION")) return weights;
            if (std.mem.eql(u8, name, "TF_DS_KV_SHARD")) return kv;
            if (std.mem.eql(u8, name, "TF_TP_WORLD")) return world;
            if (std.mem.eql(u8, name, "TF_DS_ENGRAM")) return engram;
            return null;
        }
    };
    try spillPreflight(E.get);
    E.kv = "1"; // (round 3: the spill tier runs with TF_DS_KV_SHARD)
    try spillPreflight(E.get);
    E.kv = "maybe";
    try std.testing.expectError(error.BadKvShard, spillPreflight(E.get));
    E.kv = "0";
    E.world = "4";
    try std.testing.expectError(error.SpillNeedsTwoRanks, spillPreflight(E.get));
    E.world = null;
    E.weights = null;
    try std.testing.expectError(error.NoWeightsId, spillPreflight(E.get));
    E.weights = "rev-1";
    E.engram = "/engram";
    try std.testing.expectError(error.NoEngramId, spillPreflight(E.get));
    for ([_]anyerror{ error.SpillNeedsTwoRanks, error.NoWeightsId, error.NoEngramId, error.SpillNotStarted, error.NoEngineId, error.PairNotAgreed, error.SpillNoPart, error.SpillCarryMismatch, error.SpillLayoutMismatch, error.SpillSelfCheckSharded }) |err|
        try std.testing.expect(explain(null, err) != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SpillNotStarted).?, "TF_DS_SPILL_DIR") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoWeightsId).?, "TF_DS_RANK_CACHE_SHA") != null);
}

test "a follower killed mid-round: the link watch stops the collectives and the pair ends within a second" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // a backend whose next round waits on the peer until the collectives are stopped (an RDMA collect spinning on a
    // dead rank's flag), then fails as an aborted communicator does
    const Stuck = struct {
        inner: lanes.backend.Backend,
        stuck: std.atomic.Value(bool) = .init(false),
        entered: std.atomic.Value(bool) = .init(false),
        aborted: std.atomic.Value(bool) = .init(false),
        exits: std.atomic.Value(u32) = .init(0),
        code: std.atomic.Value(u8) = .init(0),

        fn backend(f: *@This()) lanes.backend.Backend {
            return .{ .ptr = f, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
        }
        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn round(f: *@This()) !void {
            if (!f.stuck.load(.acquire)) return;
            f.entered.store(true, .release);
            while (!f.aborted.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            return error.CommAborted;
        }
        fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            return of(ptr).inner.vtable.prefill(of(ptr).inner.ptr, s);
        }
        fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            return of(ptr).inner.vtable.first(of(ptr).inner.ptr, s, position);
        }
        fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
            try of(ptr).round();
            return of(ptr).inner.vtable.queue(of(ptr).inner.ptr, s, feed, position);
        }
        fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
            return of(ptr).inner.vtable.read(of(ptr).inner.ptr, handle);
        }
        fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
            try of(ptr).round();
            return of(ptr).inner.vtable.verify(of(ptr).inner.ptr, windows, out);
        }
        fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
            return of(ptr).inner.vtable.keep(of(ptr).inner.ptr, windows, paths);
        }
        fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
            return of(ptr).inner.vtable.draft(of(ptr).inner.ptr, requests);
        }
        fn release(ptr: *anyopaque, s: *lanes.Stream) void {
            of(ptr).inner.vtable.release(of(ptr).inner.ptr, s);
        }
        fn abort(ctx: *anyopaque) void {
            of(ctx).aborted.store(true, .release);
        }
        fn exit(ctx: *anyopaque, code: u8) void {
            of(ctx).code.store(code, .release);
            _ = of(ctx).exits.fetchAdd(1, .acq_rel);
        }
    };
    // what linkLost does with the State's hooks: stop the collectives, then break the pair with nothing to drain
    const Glue = struct {
        stuck: *Stuck,
        host: *api.LaneHost,
        fn lost(ctx: *anyopaque, follower: usize) void {
            const g: *@This() = @ptrCast(@alignCast(ctx));
            Stuck.abort(g.stuck);
            var buf: [160]u8 = undefined;
            g.host.pairBroken(linkLostWhy(&buf, follower), false);
        }
    };
    const Reply = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?api.Reason = null,
        message: [256]u8 = undefined,
        len: usize = 0,
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.len = @min(f.message.len, b.message.len);
                    @memcpy(b.message[0..b.len], f.message[0..b.len]);
                },
                else => {},
            }
        }
        fn state(b: *@This()) struct { tokens: usize, done: ?api.Reason } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ .tokens = b.tokens, .done = b.done };
        }
    };
    const Follower = struct {
        fn run(p: u16) void {
            var l = link.Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {}; // "die"
            l.close(); // (kill -9: the kernel closes the process's sockets)
        }
    };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var stuck: Stuck = .{ .inner = target.backend() };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core_e = lanes.Engine.init(gpa, &cfg, stuck.backend(), clock.clock());
    defer core_e.deinit();
    var host = api.LaneHost.init(gpa, io, &core_e, .{ .lanes = 1 });
    host.interleave_ns = 0;
    // the deadline (30 s) and the stall monitor (off) cannot be what ends this
    host.pair = .{ .ctx = &stuck, .abort = Stuck.abort, .exit = Stuck.exit, .deadline_ns = 30 * std.time.ns_per_s, .grace_ns = std.time.ns_per_ms, .stall_ns = 0, .fatal = &pair_fatal };
    try host.start();
    defer host.stop();
    const port: u16 = 27100 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const ft = try std.Thread.spawn(.{}, Follower.run, .{port});
    var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var glue: Glue = .{ .stuck = &stuck, .host = &host };
    var w = link.Watch.init(&.{@as(?link.Link, l)}, &glue, Glue.lost);
    try w.start();
    defer w.end();
    var a: Reply = .{};
    const prompt_ids = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const long: api.Request = .{ .prompt = &prompt_ids, .max_tokens = 1 << 30 };
    try host.engine().submit(1, &long, .{ .ctx = &a, .event = Reply.event });
    while (a.state().tokens == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    stuck.stuck.store(true, .release); // the next round waits on rank 1
    while (!stuck.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try l.send("die");
    ft.join();
    while (a.state().done == null or stuck.exits.load(.acquire) == 0) {
        if (std.Io.Clock.awake.now(io).toNanoseconds() - t0 > 5 * std.time.ns_per_s) return error.Timeout;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    const took = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    try std.testing.expect(took < std.time.ns_per_s); // (the watch's 100 ms period and 500 ms grace, then the abort)
    try std.testing.expectEqual(api.Reason.failed, a.state().done.?);
    try std.testing.expect(host.isBroken());
    try std.testing.expect(std.mem.startsWith(u8, host.pair_why[0..host.pair_why_len], "rank 1's link to rank 0 closed"));
    try std.testing.expect(!host.pair_drain);
    try std.testing.expectEqual(api.LaneHost.pair_exit_code, stuck.code.load(.acquire));
}

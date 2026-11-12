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
//!   TF_DS_ROUND_ROWS, TF_DS_STREAMS             a round's rows at most (16; the four-node lane's 48) and the pool's
//!                                               streams (4; its 16): --parallel at most the streams
//!   TF_RDMA_DEVICES                             the decode gathers over the RDMA ring on these devices (else NCCL)
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//!   TF_DS_SOLO_OVERHEAD=1                       a lone drafted stream's head windows cost the fixed draft allowance
//!   TF_DS_NO_COPIES=1                           a drafted stream proposes no suffix-lookup copy windows
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
    /// TF_DS_KV_SHARD: the parts' counters and rows for /health and /metrics (null: the switch off).
    kv_parts: ?lanes.kv_parts.Source = null,
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
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.gpa = gpa;
    st.staging = null;
    st.events = &.{};
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
        .side = if (getenv("TF_DS_HC_SIDE")) |v| !std.mem.eql(u8, v, "0") else false,
        .prefetch = if (getenv("TF_DS_L2_PREFETCH")) |v| !std.mem.eql(u8, v, "0") else false,
        .engram_aio = if (getenv("TF_DS_ENGRAM_AIO")) |v| !std.mem.eql(u8, v, "0") else false,
        .pool = try pool_mod.resolve(o.context, getenv("TF_DS_POOL")), // may exceed the window: more kept prompts stay resident
        .window = o.context, // (a follower takes TF_DS_WINDOW, else rank 0's: Model.open)
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
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
    st.lanes.no_copies = if (getenv("TF_DS_NO_COPIES")) |v| !std.mem.eql(u8, v, "0") else false;
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
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .vision = if (st.vision) |v| v.hook() else null, .window = window, .tier = tier, .grid = model.chunk_rows, .replay = @intCast(st.m.cfg.window), .kv_parts = kv_parts };
}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
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
    const image_cache = if (getenv("TF_DS_IMAGE_CACHE")) |v| !std.mem.eql(u8, v, "0") else true;
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
        .native_format = core.spill.format.version,
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
        error.BadRoundRows => "TF_DS_ROUND_ROWS: a round's rows at most, 16 .. 48",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 16",
        error.ContextFull => "the streams' prompts and budgets fill the context's pool: lower max_tokens, or retry when a stream ends",
        error.PromptTooLong => "the prompt and its max_tokens do not fit the context window",
        error.NoVisionKit => "this checkpoint reads images: its kit (TF_DS_KIT) needs vision/torch_fmha_sm120.cubin and vision/gate_bias_vl.safetensors",
        error.BadPool => "TF_DS_POOL (rank 1: --pool): the pool's positions, a whole number of at least the context (unset: the context)",
        error.BadWindow => "TF_DS_WINDOW: the positions one request may use, a whole number; on rank 0 it must be --context (start.sh sets both from CONTEXT)",
        error.BadKvShard => "TF_DS_KV_SHARD: 1 (on) or 0 (off)",
        error.BadKvShardDecode => "TF_DS_KV_SHARD_DECODE: fetch (the default) or owner",
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
        error.ShardDiverged, error.StaleExtent => "TF_DS_KV_SHARD=1: the ranks' KV-part planners disagree (an engine bug; the pair stops rather than read the wrong rows): restart with TF_DS_KV_SHARD=0",
        error.TransferFailed => "TF_DS_KV_SHARD=1: a KV transfer between the ranks failed on one rank only: restart; if it recurs, TF_DS_KV_SHARD_CARRY=0 or TF_DS_KV_SHARD=0",
        error.TransferTimeout => "TF_DS_KV_SHARD=1: a KV transfer did not complete in time (a rank's half never ran, or a device fault; the GPU stream holds the unmatched call): restart; if it recurs, TF_DS_KV_SHARD_CARRY=0 or TF_DS_KV_SHARD=0",
        error.KvShardNotStarted, error.BadPlan => "TF_DS_KV_SHARD=1: the KV-part planner was not set up as the lanes need (an engine bug): restart with TF_DS_KV_SHARD=0",
        error.FollowerScanStalled => "rank 1's spill store scan stopped making progress (its log names the store): check its spill disk (TF_DS_SPILL_DIR), or start it with a fresh directory",
        else => null,
    };
}

test "refusals name what to set" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoMaster).?, "TF_TP_MASTER") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.PartTooSmall).?, "TF_DS_POOL") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.KvShardMismatch).?, "--pool") != null);
    for ([_]anyerror{ error.BadPart, error.WindowTooWide, error.FetchTooLarge, error.BadFetch, error.NoKvShard, error.CrossesPart, error.ShardDiverged, error.StaleExtent, error.TransferFailed, error.TransferTimeout, error.KvShardNotStarted, error.BadPlan }) |err|
        try std.testing.expect(std.mem.indexOf(u8, explain(null, err).?, "TF_DS_KV_SHARD=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
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

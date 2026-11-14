//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const engram_lead = @import("engram_lead.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");
const rdma_env = @import("rdma_env.zig");
const pool_mod = @import("pool.zig");
const shard = @import("shard.zig");
const shard_ops = @import("shard_ops.zig");
const prompt_rows = @import("prompt_rows.zig");

/// The served prompt chunk (TF_DS_PREFILL_CHUNK): a prompt runs in chunks that start at its multiples.
pub const chunk_rows = 2048;
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    // the positions one conversation may reach (its prompt and reply: the RoPE table's rows, the prompt chunks' and
    // rounds' buckets); null: rank 0's (a follower adopts it at open), or the pool's on rank 0 (today's engine)
    window: ?usize = null,
    // TF_DS_KV_SHARD (shard.zig): owner-only compressed rows over the TP2 ranks; null: read from the environment on
    // every rank alike (TF_DS_KV_SHARD, TF_DS_KV_SHARD_DECODE), as TF_DS_IDX_SPLIT is
    kv_shard: ?shard.Mode = null,
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    graph_cap: usize = 0, // the captured stretches kept at most, least recently replayed out first (TF_DS_GRAPH_CAP; 0: all)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    plain_rows: usize = round.default_rows, // a mixed pack's plain linear runs over at least these rows (TF_DS_PLAIN_ROWS; 0: the round's own)
    window_rows: usize = prompt_rows.default_window, // a chunk under these rows takes the MoE decode window (TF_DS_PROMPT_WINDOW_ROWS; every rank alike)
    gemm_rows: usize = prompt_rows.default_gemm, // a chunk over these rows takes the prefill GEMM (TF_DS_PROMPT_GEMM_ROWS; every rank alike)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
};

/// TF_DS_SERVE_PROFILE: one fillChunk's device buckets (ns).
pub const TurnFillPass = struct {
    upload_ns: u64 = 0,
    replay_ns: u64 = 0,
    layers_ns: u64 = 0,
    drafter_ns: u64 = 0,
    tokens: u32 = 0,

    pub fn deviceNs(p: TurnFillPass) u64 {
        return p.upload_ns + p.replay_ns + p.layers_ns + p.drafter_ns;
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    window_rows: usize, // Options.window_rows
    gemm_rows: usize, // Options.gemm_rows
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    paio2: ?*engram_aio.Aio = null, // TF_DS_ENGRAM_LEAD: the second Engram layer's prompt reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    // TF_DS_ENGRAM_LEAD=1: each Engram layer's prompt rows a chunk ahead on its own reader, in place of `ahead`
    lead: engram_lead.Lead = .{},
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    turn_timing: bool = false,
    turn_pass: TurnFillPass = .{},
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    rdma_rings: ?ring2d.Rings = null,
    // the pool / window split and TF_DS_KV_SHARD (open): the positions one stream's extent may take at most (a part
    // sharded; the request check's bound), the single-conversation limit reported (shard.plan), the shard's kernels
    request_cap: usize = 0,
    window_cap: usize = 0,
    kvs: ?shard_ops.KvShard = null,

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        if (o.world != 2 and o.world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split (prompt2d.zig)
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        const idx_split_env = std.c.getenv("TF_DS_IDX_SPLIT"); // every rank alike
        prompt.idx_split = if (idx_split_env) |v| std.mem.eql(u8, std.mem.span(v), "1") else true;
        prompt.idx_split_min_comp = if (idx_split_env != null) 0 else 65536;
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        if (!prompt_rows.validWindow(o.window_rows)) return error.BadPromptWindowRows;
        if (!prompt_rows.validGemm(o.gemm_rows)) return error.BadPromptGemmRows;
        m.window_rows = o.window_rows;
        m.gemm_rows = o.gemm_rows;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool; // (agree, below, compares it with the other rank's)
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.paio2 = null;
        m.eh = null;
        m.fill_seq = &.{};
        m.ahead = null;
        m.lead = .{};
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.rdma_rings = null;
        m.kvs = null;
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();

        // rank 0's ports for the followers first: a worker that loads its weights sooner waits in its port's backlog
        // (before this, rank 0 listened only after its own weights had loaded, and a worker that was ready more
        // than TF_DS_CONNECT_S earlier exited with ConnectFailed)
        var ports = try prompt2d.Fan.bind(o.master, o.port, o.rank, o.world);
        defer ports.close();

        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer m.cache_dir.close(io);
        const cache_name = try rank_cache.find(a, io, m.cache_dir, o.rank, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        const sp = try prompt2d.split(o.rank, o.world);
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts);
        errdefer m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' link and NCCL (rank 0 takes the connections its ports have waiting; rank 1 connects)
        m.fan = try prompt2d.Fan.openBound(io, &ports, o.master, o.port, o.rank, o.world);
        errdefer m.fan.close();
        m.comm = try m.fan.comm(o.rank, o.world);
        errdefer m.comm.deinit();

        // the pool, the window and the shard: every rank's compared first thing after NCCL, before anything rank-local
        // can fail and before any buffer depends on them (a follower without a window takes rank 0's). A setting this
        // rank cannot parse is refused after the gather, so the other rank refuses with it instead of waiting. Then the
        // plan (shard.plan: the effective single-conversation limit), the same on every rank from the same numbers.
        const want_mode: anyerror!shard.Mode = if (o.kv_shard) |k| k else shard.Mode.parse(envOf("TF_DS_KV_SHARD"), envOf("TF_DS_KV_SHARD_DECODE"));
        // the pool and window are 0008's (pool.zig): rank 0's pool is TF_DS_POOL over --context (native.open), a
        // follower's its --pool; a window is --context on rank 0 and TF_DS_WINDOW on a follower (else rank 0's)
        const want_pool: anyerror!usize = if (o.pool == 0) error.BadPool else o.pool;
        const want_window: anyerror!?usize = windowOf(o);
        const ag = try agree(m, o, want_mode, want_pool, want_window);
        const mode = ag.mode;
        const pool = ag.pool;
        const window = ag.window;
        m.pool_cap = pool;
        if (mode != .off and (o.world != 2 or sp.pair != null)) return error.KvShardNeedsTwoRanks;
        const pl = try shard.plan(pool, window, o.round_rows, o.world, mode != .off);
        // (with the spill tier: spillLayout is owner_sharded with the part size, so its compat digest and block lineage
        // carry the owner layout and the part salt; spill_state packs a sharded plane's part-local rows)
        m.request_cap = pl.extent;
        m.window_cap = pl.window;
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            m.ring = try openRing(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.comm, m.stream);
        }
        errdefer if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer m.set.deinit();
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer m.exk.unload();
        if (!cuda.kernels.available) return error.NoKernelImages;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer m.ops.unload();
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer m.ex.unload();

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position a stream can reach (the window, not the pool)
        const half = c.rope_dim / 2;
        // (0008's ropeRows over the longest extent a stream may take: the window's rows rounded up to whole blocks,
        // and in a part when sharded; the pool's only when the pool is the window)
        const rope_rows = pool_mod.ropeRows(m.request_cap, round.max_rows);
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table of fewer rows than the window
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = try rdma_env.servedGidIndex(), .ack_timeout = try rdma_env.servedAckTimeout() }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        m.eng.window_rows = m.window_rows;
        m.eng.gemm_rows = m.gemm_rows;
        m.eng.plain_rows = o.plain_rows;
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, chunk_rows, m.request_cap);
        errdefer if (m.kvs) |*x| x.k.unload(); // (registered before the load: nothing between them leaks it)
        if (mode != .off) {
            // TF_DS_KV_SHARD: the fetch kernels, the parts (a trash row a round row), owner-only planes
            if (!cuda.kernels.available) return error.NoKernelImages;
            const sh = try shard.Shard.init(o.rank, o.world, pool, round.max_rows);
            m.kvs = .{ .sh = sh, .mode = mode, .k = try shard_ops.Kernels.load(d, cuda.kernels.dsv41_shard) };
            m.eng.kv = &m.kvs.?;
            m.caches = try prompt.Caches.initShard(&m.eng, &m.arena, pool, m.streams, sh);
            try m.ch.useShard(&m.eng, &m.arena, prompt.prompt_fetch_rows, m.request_cap);
        } else m.caches = try prompt.Caches.init(&m.eng, &m.arena, pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.request_cap);
        if (m.kvs != null) try m.rd.useShard(&m.eng, &m.arena, a);
        m.amax = try m.arena.take(round.max_rows * 4);
        if (o.graphs) m.graphs = round.Graphs.init(gpa, o.graph_cap);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

        // whatever the Engram setup below has made goes if the open fails after it (close() only runs for an open model);
        // registered here, not inside the block, so a later failure runs them too
        errdefer if (m.tables) |*t| t.close();
        errdefer if (m.epool) |p| p.deinit(gpa);
        errdefer if (m.aio) |x| x.deinit();
        errdefer if (m.paio) |x| x.deinit();
        errdefer if (m.paio2) |x| x.deinit();
        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, chunk_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
            // TF_DS_ENGRAM_LEAD=1, read on each rank (they need not agree): each Engram layer's prompt reads on a ring
            if (engram_lead.enabled(if (std.c.getenv("TF_DS_ENGRAM_LEAD")) |v| std.mem.span(v) else null)) {
                if (m.paio) |ring| {
                    m.paio2 = try engram_aio.Aio.init(gpa, io, 4096);
                    try m.eh.?.useLead(a, chunk_rows, .{ ring, m.paio2.? });
                    std.log.info("engram-lead: on (rank {d}): each Engram layer's prompt rows read a chunk ahead", .{o.rank});
                } else std.log.warn("engram-lead: off (rank {d}): it needs the Engram AIO (TF_DS_ENGRAM_AIO=1; rank 1: --aio 1)", .{o.rank});
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision) {
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, chunk_rows);
        m.ids64 = try a.alloc(i64, chunk_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        if (mode != .off or window != pool) m.notePlan(mode, pl);
        return m;
    }

    /// The environment variable `name` (null when unset).
    fn envOf(name: [:0]const u8) ?[]const u8 {
        return std.mem.span(std.c.getenv(name) orelse return null);
    }

    /// The window this rank was given: `o.window` (rank 0: --context), else TF_DS_WINDOW (0008: the recipe gives it to
    /// every rank, = CONTEXT; never past the pool), else null (agree takes rank 0's). When both are given they must be one
    /// number, so the RoPE tables and the request bound cannot disagree on a rank.
    fn windowOf(o: Options) !?usize {
        return pickWindow(o.window, o.pool, envOf("TF_DS_WINDOW"));
    }
    fn pickWindow(given: ?usize, pool: usize, env: ?[]const u8) !?usize {
        const set = env != null and std.mem.trim(u8, env.?, " \t\r\n").len != 0;
        const from_env: ?usize = if (set) try pool_mod.window(pool, env) else null;
        if (given) |w| {
            if (from_env) |e| if (e != @min(w, pool)) return error.BadWindow;
            return w;
        }
        return from_env;
    }

    /// What every rank serves once agree has compared them.
    const Agreed = struct { mode: shard.Mode, pool: usize, window: usize };

    /// Every rank's (mode, pool, window, round rows, prompt window rows, prompt GEMM rows) gathered and compared before any buffer depends on them. A rank
    /// whose shard switch or pool did not parse sends a sentinel (bad_setting) instead of failing before the gather,
    /// then refuses with its own error after it; the other rank sees the sentinel and refuses too (KvShardMismatch),
    /// so neither waits on the other. A rank whose switch, pool or round rows differ from rank 0's refuses
    /// (KvShardMismatch, on both at once), a window given on both must match, and a rank without one takes rank 0's
    /// (the follower lane has no --context). Ranks on different prompt rows refuse too (PromptRowsMismatch): they would
    /// launch different kernels for one chunk and sum wrong partials.
    fn agree(m: *Model, o: Options, want_mode: anyerror!shard.Mode, want_pool: anyerror!usize, want_window: anyerror!?usize) !Agreed {
        const n: usize = 6;
        const bad_setting: u64 = 0xff;
        const mine = [n]u64{
            if (want_mode) |x| @backingInt(x) else |_| bad_setting,
            if (want_pool) |x| x else |_| 0,
            if (want_window) |x| x orelse 0 else |_| bad_setting,
            o.round_rows,
            o.window_rows,
            o.gemm_rows,
        };
        var dev = try cuda.DeviceBuffer.alloc(m.ctx.d, (1 + o.world) * n * 8);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try m.comm.allGather(dev.ptr, dev.ptr + n * 8, n, .u64, m.stream);
        try m.stream.synchronize();
        var all: [4][n]u64 = undefined;
        if (o.world > all.len) return error.BadWorld;
        try dev.download(n * 8, std.mem.sliceAsBytes(all[0..o.world]));
        // this rank's own parse errors first (each rank names what it could not read), then any other rank's
        const mode = try want_mode;
        const pool = try want_pool;
        const own_window = try want_window;
        for (all[0..o.world]) |r| {
            if (r[0] == bad_setting or r[1] == 0 or r[2] == bad_setting) return error.KvShardMismatch;
            if (r[0] != all[0][0] or r[1] != all[0][1] or r[3] != all[0][3]) return error.KvShardMismatch;
            if (r[4] != all[0][4] or r[5] != all[0][5]) return error.PromptRowsMismatch;
            if (r[2] != 0 and all[0][2] != 0 and r[2] != all[0][2]) return error.KvShardMismatch;
        }
        const window = if (own_window) |w| w else if (all[0][2] != 0) all[0][2] else pool;
        return .{ .mode = mode, .pool = pool, .window = window };
    }

    /// The start-up's note of what it serves when the pool is not the window or the shard is on: the pool, its parts,
    /// the single-conversation limit (below the window asked: said so, with the numbers), the compressed planes'
    /// bytes a rank (calculated: rows by the formula; the arena's figure is the measured one).
    fn notePlan(m: *const Model, mode: shard.Mode, pl: shard.Plan) void {
        var buf: [16]shard.Source = undefined;
        const src = m.kvSources(&buf);
        const sharded = mode != .off;
        const trash: usize = if (m.kvs) |k| k.sh.trash else 0;
        const planes = shard.poolBytes(src, pl.pool, m.world, sharded, trash);
        const flat = shard.poolBytes(src, pl.pool, m.world, false, 0);
        std.log.info("dsv41: pool {d} positions, {s} {d} a part; a conversation may use up to {d} tokens{s}; compressed planes {d} MiB a rank ({d} MiB replicated); arena {d} MiB used", .{
            pl.pool, @tagName(mode), pl.part, pl.window, if (pl.shrunk) " (less than the window asked: a part holds no more)" else "", planes >> 20, flat >> 20, m.arena.used >> 20,
        });
    }

    /// Every kv-source layer's packed row bytes (shard.Source: the capacity arithmetic's and the carry cost's), the
    /// first `buf.len` of them.
    pub fn kvSources(m: *const Model, buf: *[16]shard.Source) []shard.Source {
        var ns: usize = 0;
        const c = m.eng.c;
        for (m.w.layers) |lay| {
            if (lay.comp_wkv == null or ns == buf.len) continue;
            buf[ns] = .{ .ratio = lay.ratio, .comp_row = c.head_dim / 2 + c.head_dim / 16, .idx_row = if (lay.idx_wk != null) c.index_head_dim / 2 + c.index_head_dim / 32 else 0 };
            ns += 1;
        }
        return buf[0..ns];
    }

    pub fn close(m: *Model) void {
        m.joinAhead();
        m.stream.synchronize() catch {};
        if (m.img_buf) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        m.rd.dropShard();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.paio2) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        if (m.kvs) |*x| x.k.unload();
        m.arena.deinit();
        m.rope_buf.free();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        m.comm.deinit();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        if (comm.world != 2) return error.NotTwoRanks;
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, 2, .{ .max_bytes = ring_bytes, .gid_index = try rdma_env.servedGidIndex(), .ack_timeout = try rdma_env.servedAckTimeout() });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        const mine = [2]rdma.Info{ r.info(0), r.info(1) };
        var dev = try cuda.DeviceBuffer.alloc(d, 3 * 2 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 2 * per, .u8, stream);
        try stream.synchronize();
        var all: [2][2]rdma.Info = undefined;
        try dev.download(2 * per, std.mem.sliceAsBytes(&all));
        try r.connect(&.{ all[0][rank], all[1][rank] });
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| th.join();
        m.ahead = null;
        m.lead.joinAll(); // (TF_DS_ENGRAM_LEAD's readers)
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
    }

    /// This rank's RDMA rings (TP2's decode gathers, or world 4's three), into `buf`: what a broken pair's abort stops.
    pub fn rdmaRings(m: *Model, buf: *[3]*rdma.Ring) []const *rdma.Ring {
        var n: usize = 0;
        if (m.ring) |r| {
            buf[n] = r;
            n += 1;
        }
        if (m.rdma_rings) |rs| for ([_]*rdma.Ring{ rs.all4, rs.col, rs.pair }) |r| {
            if (n == buf.len) break;
            buf[n] = r;
            n += 1;
        };
        return buf[0..n];
    }

    /// A ring that failed (its proxy saw a dead peer, or an abort): a round replayed from a graph does not ask the
    /// ring, its collect kernel just gave up on the abort word, so the round's rows are not the model's. Rounds check
    /// this after reading their tokens back (error.RingFailed: rank pair broken).
    pub fn ringFailed(m: *Model) bool {
        var buf: [3]*rdma.Ring = undefined;
        for (m.rdmaRings(&buf)) |r| if (r.failure() != null) return true;
        return false;
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    /// TF_DS_KV_SHARD: the extents lie in one part each (else CrossesPart, nothing copied); the index keys copy on
    /// every rank as replicated, the compressed rows by shard.Shard.extentMove: within a part its owner copies them
    /// (the other rank holds none), across parts (a carry) the source's owner sends them straight from its plane and the
    /// destination's owner receives them into its own, both ranks in this same order (lanes.zig runs copyExtent on
    /// every rank in its frame order). The source's rows stay; nothing is freed.
    ///
    /// Every refusal comes first (checkExtent: the pool's bounds, the parts, every source's ratio, overlapping extents),
    /// so a refused copy enqueues nothing: never a partial copy that leaves some layers' rows moved and others not.
    /// With every argument valid the copies are the same, in the same order, as before the check moved up front.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        return m.copyExtentPinned(from, to, at, null);
    }

    /// copyExtent whose source rows must still be the ones `pin` recorded (its generation stamp, spill_state's
    /// Generations): a source rewritten since then (a placement staled it, a stream wrote over it) is refused with
    /// error.StaleExtent before anything is enqueued, so a late copy never attaches old bytes to a reused extent.
    pub fn copyExtentPinned(m: *Model, from: usize, to: usize, at: usize, pin: ?ExtentPin) !void {
        const mv = try m.checkExtentPinned(from, to, at, pin);
        const c = m.eng.c;
        const cs = &m.caches;
        var spans: [2 * max_extent_sources]comm_mod.Span = undefined;
        var ns: usize = 0;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            std.debug.assert(from % r == 0 and to % r == 0 and at % r == 0 and ns + 2 <= spans.len); // (checkExtent)
            const n = at / r;
            const cb = c.head_dim / 2;
            const sb = c.head_dim / 16;
            if (mv) |x| switch (x) {
                .none => {},
                .local => |l| {
                    try m.copySpan(cs.comp_codes[i], l.from / r, l.to / r, n, cb);
                    try m.copySpan(cs.comp_scales[i], l.from / r, l.to / r, n, sb);
                },
                .send => |t| {
                    spans[ns] = .{ .ptr = cs.comp_codes[i] + t.from / r * cb, .bytes = n * cb };
                    spans[ns + 1] = .{ .ptr = cs.comp_scales[i] + t.from / r * sb, .bytes = n * sb };
                    ns += 2;
                },
                .recv => |t| {
                    spans[ns] = .{ .ptr = cs.comp_codes[i] + t.to / r * cb, .bytes = n * cb };
                    spans[ns + 1] = .{ .ptr = cs.comp_scales[i] + t.to / r * sb, .bytes = n * sb };
                    ns += 2;
                },
            } else {
                try m.copySpan(cs.comp_codes[i], from / r, to / r, n, cb);
                try m.copySpan(cs.comp_scales[i], from / r, to / r, n, sb);
            }
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
        // a carry or a live move across parts (kv_lanes.zig runs parts.zig's ops in frame order on every rank, rank 0
        // decides the transfer's ok and every rank finishes with it; the generation is published at finish)
        const x = mv orelse return;
        var bytes: usize = 0;
        for (spans[0..ns]) |sp| bytes += sp.bytes;
        switch (x) {
            .send => |t| try m.comm.sendSpans(spans[0..ns], t.peer, m.stream),
            .recv => |t| try m.comm.recvSpans(spans[0..ns], t.peer, m.stream),
            else => return,
        }
        if (m.kvs) |*k| {
            k.stats.carries += 1;
            k.stats.carry_bytes += bytes;
        }
    }

    /// copyExtent's refusals with nothing enqueued: the extents' parts (shard.Shard.extentMove: CrossesPart) and every
    /// kv source's ratio (BadView). Deterministic: every rank refuses the same arguments alike under the agreed
    /// layout (kv_lanes.zig checks before any rank enqueues its side, so a refused carry leaves no NCCL half waiting).
    pub fn checkExtent(m: *const Model, from: usize, to: usize, at: usize) !void {
        _ = try m.checkExtentPinned(from, to, at, null);
    }

    /// The kv-source layers a copyExtent carries at most (its span list: codes and scales of each).
    pub const max_extent_sources = 64;

    /// A copyExtent source's generation, recorded when its rows were the caller's (Generations.pin over
    /// [from, from + at)): copyExtentPinned refuses the copy when they changed since.
    pub const ExtentPin = struct { gens: *const spill.Generations, src: spill.ExtentRef };

    /// checkExtent with an optional source pin; the compressed rows' move on this rank (null: replicated).
    fn checkExtentPinned(m: *const Model, from: usize, to: usize, at: usize, pin: ?ExtentPin) !?shard.Move {
        var ratios: [16]usize = undefined;
        var nr: usize = 0;
        var sources: usize = 0;
        for (m.eng.w.layers) |lay| {
            if (lay.comp_wkv == null) continue;
            sources += 1;
            if (sources > max_extent_sources) return error.BadView; // (copyExtent's spans: two a source)
            const r: usize = lay.ratio;
            if (std.mem.indexOfScalar(usize, ratios[0..nr], r) != null) continue;
            if (nr == ratios.len) return error.BadView;
            ratios[nr] = r;
            nr += 1;
        }
        return extentRefusal(m.caches.cap, ratios[0..nr], m.caches.shard, from, to, at, pin);
    }

    /// copyExtent's refusals as plain data (host-testable): the extents [from, from + at) and [to, to + at) inside the
    /// pool's `cap` positions (ExtentOutOfPool), in one part each (shard.Shard.extentMove: CrossesPart), on every
    /// source's ratio (BadView), not overlapping unless they are the same extent (ExtentsOverlap: a device copy over
    /// its own source is undefined), and the source still at its pin's generation (StaleExtent). Returns the
    /// compressed rows' move on this rank (sharded), else null.
    pub fn extentRefusal(cap: usize, ratios: []const usize, sh: ?shard.Shard, from: usize, to: usize, at: usize, pin: ?ExtentPin) !?shard.Move {
        if (from > cap or to > cap or at > cap - from or at > cap - to) return error.ExtentOutOfPool;
        const mv: ?shard.Move = if (sh) |x| try x.extentMove(from, to, at) else null;
        for (ratios) |r| if (r == 0 or from % r != 0 or to % r != 0 or at % r != 0) return error.BadView;
        if (from != to and at > 0 and from < to + at and to < from + at) return error.ExtentsOverlap;
        if (pin) |p| {
            if (p.src.lo > from or p.src.hi < from + at or !p.gens.current(p.src)) return error.StaleExtent;
        }
        return mv;
    }

    /// Rows [from, from + n) of `plane` (rows of `row` bytes; a sharded plane's rows of this rank's part) to rows
    /// [to, to + n), in the stream's order.
    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(plane + to * row, plane + from * row, n * row, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    // -- spill v2's checkpoint state (port of wojo's GLM 0110, PR #124: spill_state.zig) ------------------------------
    // A kept point's state on this rank as the spill tier stores it: the descriptor derived from the loaded weights and
    // the live allocation, and the planes' addresses and copies for spill.Packer. Nothing here runs unless the tier
    // calls it (round 2's lanes wiring); the served path is unchanged.

    pub const spill = @import("spill_state.zig");

    comptime {
        std.debug.assert(draft.max_streams <= spill.max_slots);
    }

    /// This rank's state planes as spill.Packer reads and writes them: their device addresses (the whole pool's
    /// compressed planes, a slot's rings and raw stores) and copies in the model's stream's order.
    pub const SpillDev = struct {
        m: *Model,

        pub const Event = cuda.Event;

        /// Row 0 of a block plane over the whole pool (the extent at base b starts b / ratio rows in, as view() puts it);
        /// TF_DS_KV_SHARD: an owner-only plane's is row 0 of this rank's part (spill_state's rowAddr adds the local row).
        pub fn blockPlane(dv: SpillDev, p: spill.Plane) u64 {
            const cs = &dv.m.caches;
            return switch (p.kind) {
                .comp_codes => cs.comp_codes[p.layer],
                .comp_scales => cs.comp_scales[p.layer],
                .idx_codes => cs.idx_codes[p.layer],
                .idx_scales => cs.idx_scales[p.layer],
                .ring, .raw_kv, .raw_score => 0,
            };
        }

        /// Slot `slot`'s rows of a small plane (snapCopy's addresses).
        pub fn slotPlane(dv: SpillDev, p: spill.Plane, slot: usize) u64 {
            const m = dv.m;
            const base: u64 = switch (p.kind) {
                .ring => m.rings[p.layer],
                .raw_kv => m.caches.raw_kv[p.layer],
                .raw_score => m.caches.raw_score[p.layer],
                .comp_codes, .comp_scales, .idx_codes, .idx_scales => 0,
            };
            return if (base == 0) 0 else base + slot * p.slotBytes();
        }

        pub fn capacity(dv: SpillDev) usize {
            return dv.m.caches.cap;
        }

        pub fn slots(dv: SpillDev) usize {
            return dv.m.streams;
        }

        pub fn d2h(dv: SpillDev, dst: []u8, src: u64) !void {
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, dv.m.stream.handle), "cuMemcpyDtoHAsync");
        }

        pub fn h2d(dv: SpillDev, dst: u64, src: []const u8) !void {
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, dv.m.stream.handle), "cuMemcpyHtoDAsync");
        }

        pub fn fill(dv: SpillDev, dst: u64, value: u8, bytes: usize) !void {
            const d = dv.m.ctx.d;
            try d.check(d.api.cuMemsetD8Async(dst, value, bytes, dv.m.stream.handle), "cuMemsetD8Async");
        }

        pub fn record(dv: SpillDev, ev: cuda.Event) !void {
            try ev.record(dv.m.stream);
        }

        /// Whether the copies before `ev` completed (spill_tier polls it; a device error reads as done, and the next
        /// copy on the stream reports it).
        pub fn done(_: SpillDev, ev: cuda.Event) bool {
            return ev.done() catch true;
        }

        /// The copies before `ev` completed (a cancel, a drop, a shutdown: bounded by what was enqueued).
        pub fn wait(_: SpillDev, ev: cuda.Event) void {
            ev.synchronize() catch {};
        }
    };

    /// The spill state descriptor of this rank, derived from the loaded layers (Caches.init's reading of them) and
    /// checked against the live allocation: snapCopy's bytes, a ring per layer, every plane present. TF_DS_KV_SHARD:
    /// owner_sharded, cut into the shard's parts (its codes and scales owner-only, at their part-local rows).
    pub fn spillLayout(m: *const Model) !spill.StateLayout {
        const layers = m.eng.w.layers;
        if (layers.len > spill.max_layers) return error.SpillLayoutMismatch;
        var shapes: [spill.max_layers]spill.LayerShape = undefined;
        for (layers, shapes[0..layers.len]) |lay, *s| {
            const kv = lay.comp_wkv != null;
            s.* = .{ .ratio = if (kv) lay.ratio else 0, .kv = kv, .idx = kv and lay.idx_wk != null, .raw = kv and lay.ratio > 1 };
        }
        const sh = m.caches.shard;
        const raw = try spill.StateLayout.init(shapes[0..layers.len], spill.Geometry.of(m.eng.c, chunk_rows, prompt.ring_extra, prompt.raw_rows), if (sh != null) .owner_sharded else .replicated);
        const l = if (sh) |x| try raw.withParts(x.part) else raw;
        if (l.small_bytes != m.snapBytes() or m.rings.len != layers.len) return error.SpillLayoutMismatch;
        const dv: SpillDev = .{ .m = @constCast(m) };
        for (l.blockPlanes()) |p| if (dv.blockPlane(p) == 0) return error.SpillLayoutMismatch;
        for (l.smallPlanes()) |p| {
            if (dv.slotPlane(p, 0) == 0) return error.SpillLayoutMismatch;
            if (p.kind == .ring and p.slotBytes() != m.eng.ringBytes()) return error.SpillLayoutMismatch;
        }
        return l;
    }

    /// The pool's extent and slot generations (one counter a chunk of positions), sized once at setup.
    pub fn spillGenerations(m: *const Model, gpa: std.mem.Allocator) !spill.Generations {
        return spill.Generations.init(gpa, m.caches.cap, chunk_rows);
    }

    /// A packer over this rank's planes, `l` from spillLayout and `g` from spillGenerations (both the caller's, alive
    /// while it is used). Its calls enqueue on the model's stream and record the caller's event: no allocation, no wait.
    pub fn spillPacker(m: *Model, l: *const spill.StateLayout, g: *spill.Generations) spill.Packer(SpillDev) {
        return .{ .dev = .{ .m = m }, .layout = l, .gens = g, .rank = m.rank };
    }

    /// The Engram read-ahead joined and its rows dropped, so the next chunk reads its own rows from its prompt's ids.
    /// Baseline hazard (also without spill): fillChunk takes a joined read-ahead whose (ahead_start, ahead_n) match its
    /// chunk, and a pass that ends between chunks (cancel, error) leaves one standing; a later prompt resuming a kept
    /// point at that position (in memory or restored) would consume rows hashed from the other prompt's ids. The fix
    /// is one call at the start of every pass, in fillBegin after its joinAhead (outside this block: routed to the
    /// coordinator, docs/port/dsv41-state.md); until it lands, a restore calls this right before the resumed pass.
    pub fn resetEngramAhead(m: *Model) void {
        m.joinAhead();
        if (m.eh) |*eh| { // rows read ahead for another prompt are not this one's, even at the same positions
            eh.ahead_start = std.math.maxInt(usize);
            eh.ahead_ready = .{ false, false };
            eh.ahead_n = 0;
            eh.ahead_err = null;
            if (eh.lead_on) m.lead.reset(eh);
        }
    }

    /// After a restored prompt's fill: the drafter's rings were rebuilt by its replay through the prompt's end
    /// (spill.checkReplayRebuilds' rule, here as the pool records it), so the first draft reads no unwritten row.
    pub fn spillCheckResumed(m: *const Model, slot: usize, len: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        if (m.dpool) |*p| if (p.absorbed[slot] != len) return error.DraftNotRebuilt;
    }

    /// A compressed plane as copyExtent copies it: its address, row bytes (copyExtent's own widths from the config)
    /// and ratio.
    const SpillRefPlane = struct { plane: u64, row: usize, ratio: usize };

    /// copyExtent's planes in its order (every kv layer: codes, scales, then the index keys' codes and scales), read
    /// from the loaded layers and the config, not from the spill layout.
    fn spillRefPlanes(m: *const Model, out: *[4 * spill.max_layers]SpillRefPlane) ![]SpillRefPlane {
        const c = m.eng.c;
        const cs = &m.caches;
        var k: usize = 0;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            if (k + 4 > out.len) return error.SpillLayoutMismatch;
            const r: usize = lay.ratio;
            out[k] = .{ .plane = cs.comp_codes[i], .row = c.head_dim / 2, .ratio = r };
            out[k + 1] = .{ .plane = cs.comp_scales[i], .row = c.head_dim / 16, .ratio = r };
            k += 2;
            if (lay.idx_wk != null) {
                out[k] = .{ .plane = cs.idx_codes[i], .row = c.index_head_dim / 2, .ratio = r };
                out[k + 1] = .{ .plane = cs.idx_scales[i], .row = c.index_head_dim / 32, .ratio = r };
                k += 2;
            }
        }
        return out[0..k];
    }

    /// Stage 2's completeness check on one instance (a GPU tool, not a serving path), against the engine's own resume
    /// primitives rather than the layout itself. The point at `at` of slot `from_slot` and the extent at `from`:
    ///   - reference: copyExtent(from, ref, at) into a third extent and snapCopy(from_slot) into a scratch Snap buffer;
    ///   - packed, unpacked into slot `to_slot` and the extent at `to` (both overwritten, as is `ref`), packed again.
    /// Then (round trip) both packs are equal; (blocks) the restored rows of `to` equal copyExtent's rows at `ref`, and
    /// the packed blocks, cut with offsets computed from copyExtent's loop and the config's row widths, equal them;
    /// (small) a snapCopy of the restored slot equals the reference Snap, and the packed small state equals it too.
    /// Every argument is checked before anything is enqueued; its own pinned staging, device scratch and event, none
    /// freed before the stream is idle.
    pub fn spillSelfCheck(m: *Model, gpa: std.mem.Allocator, from_slot: usize, from: usize, to_slot: usize, to: usize, ref: usize, at: usize) !void {
        // (its reference reads the pool-wide planes copyExtent's replicated loop addresses; TF_DS_KV_SHARD's packing is
        // checked by spill_state's two-rank host tests and on the GPU by the restored token_sha)
        if (m.caches.shard != null) return error.SpillSelfCheckSharded;
        if (from_slot >= m.streams or to_slot >= m.streams or from_slot == to_slot) return error.BadSlot;
        if (at == 0) return error.BadRange;
        for ([_]usize{ from, to, ref, at }) |v| if (v % chunk_rows != 0) return error.BadRange;
        const cap = m.caches.cap;
        for ([_]usize{ from, to, ref }) |b| if (b >= cap or at > cap - b) return error.BadRange;
        for ([_][2]usize{ .{ from, to }, .{ from, ref }, .{ to, ref } }) |p| if (p[0] < p[1] + at and p[1] < p[0] + at) return error.BadRange;
        const l = try m.spillLayout();
        var refs_buf: [4 * spill.max_layers]SpillRefPlane = undefined;
        const refs = try m.spillRefPlanes(&refs_buf);
        var ref_bytes: usize = 0;
        for (refs) |rp| ref_bytes += at / rp.ratio * rp.row;
        const nb = try l.blocks(at);
        const blk = try std.math.mul(usize, nb, l.block_bytes);
        if (ref_bytes != blk) return error.SpillLayoutMismatch; // the layout's block bytes are copyExtent's
        const small = l.small_bytes;
        var g = try m.spillGenerations(gpa);
        defer g.deinit(gpa);
        var pk = m.spillPacker(&l, &g);

        var a = try cuda.HostBuffer.alloc(m.ctx.d, blk + small);
        defer a.free();
        var b = try cuda.HostBuffer.alloc(m.ctx.d, blk + small);
        defer b.free();
        var rows = try cuda.HostBuffer.alloc(m.ctx.d, 2 * blk); // copyExtent's rows at ref, then the restored rows at to
        defer rows.free();
        var snaps = try cuda.HostBuffer.alloc(m.ctx.d, 2 * small); // the reference Snap, then the restored slot's
        defer snaps.free();
        var sa = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer sa.free();
        var sb = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer sb.free();
        var ev = try cuda.Event.init(m.ctx.d, false);
        defer ev.deinit();
        defer m.stream.synchronize() catch {}; // (runs first: nothing above is freed under an enqueued copy)

        // the engine's own copies of the point
        try m.snapCopy(from_slot, sa.ptr, true);
        try m.copyExtent(from, ref, at);
        // pack, restore elsewhere, pack again
        _ = try pk.packBlocks(try g.pin(from, from + at), 0, nb, a.bytes[0..blk], ev);
        _ = try pk.packSmall(try g.pinSlot(from_slot), a.bytes[blk..][0..small], ev);
        try ev.synchronize();
        const dst = try g.reserve(to, to + at);
        const ds = try g.reserveSlot(to_slot);
        const ub = try pk.unpackBlocks(dst, 0, nb, a.bytes[0..blk], ev);
        const us = try pk.unpackSmall(ds, a.bytes[blk..][0..small], ev);
        _ = try pk.packBlocks(dst, 0, nb, b.bytes[0..blk], ev);
        _ = try pk.packSmall(ds, b.bytes[blk..][0..small], ev);
        try m.snapCopy(to_slot, sb.ptr, true);
        // plain copies of each plane's rows [ref / r, (ref + at) / r) and [to / r, (to + at) / r), plane after plane
        const d = m.ctx.d;
        var off: usize = 0;
        for ([_]usize{ ref, to }) |base| for (refs) |rp| {
            const n = at / rp.ratio * rp.row;
            try d.check(d.api.cuMemcpyDtoHAsync_v2(rows.bytes[off..].ptr, rp.plane + base / rp.ratio * rp.row, n, m.stream.handle), "cuMemcpyDtoHAsync");
            off += n;
        };
        try d.check(d.api.cuMemcpyDtoHAsync_v2(snaps.bytes.ptr, sa.ptr, small, m.stream.handle), "cuMemcpyDtoHAsync");
        try d.check(d.api.cuMemcpyDtoHAsync_v2(snaps.bytes[small..].ptr, sb.ptr, small, m.stream.handle), "cuMemcpyDtoHAsync");
        try m.stream.synchronize();
        if (!g.valid(ub) or !g.valid(us)) return error.StaleExtent;

        if (!std.mem.eql(u8, a.bytes[0 .. blk + small], b.bytes[0 .. blk + small])) return error.SpillRoundTripMismatch;
        const ref_rows = rows.bytes[0..blk];
        if (!std.mem.eql(u8, ref_rows, rows.bytes[blk..][0..blk])) return error.SpillRestoreMismatch;
        // the packed blocks against copyExtent's rows: block k's plane j is that plane's rows from k * 2048 / ratio
        var in_block: usize = 0; // plane j's offset in a block (copyExtent's order and widths)
        var in_rows: usize = 0; // plane j's offset in ref_rows
        for (refs) |rp| {
            const per = chunk_rows / rp.ratio * rp.row;
            for (0..nb) |k| {
                if (!std.mem.eql(u8, b.bytes[k * l.block_bytes + in_block ..][0..per], ref_rows[in_rows + k * per ..][0..per])) return error.SpillLayoutMismatch;
            }
            in_block += per;
            in_rows += at / rp.ratio * rp.row;
        }
        if (in_block != l.block_bytes) return error.SpillLayoutMismatch;
        const ref_snap = snaps.bytes[0..small];
        if (!std.mem.eql(u8, ref_snap, snaps.bytes[small..][0..small])) return error.SpillRestoreMismatch;
        if (!std.mem.eql(u8, ref_snap, b.bytes[blk..][0..small])) return error.SpillLayoutMismatch;
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try m.ctx.d.check(m.ctx.d.api.cuMemcpyHtoD_v2(dst, tmp.ptr, tmp.len), "cuMemcpyHtoD");
            lay.gate_b_vl = dst;
        }
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(m.img_buf.?.ptr, rows.ptr, rows.len, m.stream.handle), "cuMemcpyHtoDAsync");
            try m.stream.synchronize(); // (the caller's rows may go once this returns)
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        // a read-ahead an earlier pass left (cancelled between chunks) never feeds this one: a resume at that pass's
        // next chunk start would otherwise take its rows, hashed from the other prompt's ids
        m.resetEngramAhead();
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// TF_DS_KV_SHARD: the filling prompt's view (fillBegin's, at `base`) with its borrowed twins (`on`; `local`: this
    /// rank's twin when it does not own the extent; prompt.Caches.twinned), or plain again. Both ranks bind at the
    /// same frame (kv_lanes.zig), between chunks only.
    pub fn bindTwin(m: *Model, base: usize, on: bool, local: ?usize) !void {
        const v = try m.caches.view(&m.eng, m.fill_slot, base);
        m.view = if (on) try v.twinned(&m.caches, &m.eng, base, local) else v;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    fn turnSyncedNow(m: *Model) u64 {
        m.stream.synchronize() catch {};
        return m.now();
    }

    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n == 0 or n > chunk_rows or seq.len != start + n) return error.BadChunk;
        const timing = m.turn_timing;
        var turn: TurnFillPass = .{};
        if (timing) turn.tokens = @intCast(n);
        defer if (timing) {
            m.turn_pass = turn;
        };
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        if (pt) |t| try t.mark(m.stream, .start);
        var layers0: u64 = 0;
        if (timing) {
            _ = turnSyncedNow(m);
            layers0 = m.now();
        }
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        if (timing) {
            const t1 = turnSyncedNow(m);
            turn.upload_ns = t1 - layers0;
            layers0 = t1;
        }
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // Engram rows read ahead on a thread while the GPU runs the layers before them (the served reader pool's
        // overlap): this chunk's both layers' during the chunk before it (m.ahead), else the first layer's from the
        // chunk's start and the next's once the first is applied
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| ref: {
            // TF_DS_ENGRAM_LEAD: each layer's rows come from its own reader (engram_lead.zig), not the read-ahead below
            if (eh.lead_on) {
                m.lead.begin(eh, c, seq, start, n);
                break :ref;
            }
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            // rows read ahead that ended before a round or a piece's end are this chunk's when they cover it, both layers
            const whole = eh.ahead_start == start and eh.ahead_n == n and eh.ahead_err == null and eh.ahead_ready[0] and (c.engram_layers.slice().len < 2 or eh.ahead_ready[1]);
            if (ahead == null and !whole) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                var replay0: u64 = 0;
                if (timing) replay0 = turnSyncedNow(m);
                const past = try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos);
                if (timing) turn.replay_ns += turnSyncedNow(m) - replay0;
                if (!past) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (timing) turn.layers_ns = turnSyncedNow(m) - layers0 - turn.replay_ns;
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                if (eh.lead_on) try m.lead.before(eh, j, start, n); // (TF_DS_ENGRAM_LEAD: this layer's reader)
                try prompt.engramApply(e, ch, eh, li, seq);
                const ne = @min(2, c.engram_layers.slice().len);
                if (eh.lead_on) {
                    // this layer's rows of the next chunk from now, a whole chunk ahead of their layer
                    const more = m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr;
                    const nn = if (more) @min(chunk_rows, m.fill_seq.len - start - n) else 0;
                    m.lead.after(eh, c, seq, start, n, j, if (more) m.fill_seq[0 .. start + n + nn] else null, nn);
                } else if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1])
                        ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next chunk's rows while this one's layers run
                    const nn = @min(chunk_rows, m.fill_seq.len - start - n);
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.pa, ch.ga);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.pm, ch.gm);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (timing) turn.layers_ns = turnSyncedNow(m) - layers0 - turn.replay_ns;
        if (m.dr) |*dr| {
            const draft0 = if (timing) turnSyncedNow(m) else 0;
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
            if (timing) turn.drafter_ns = turnSyncedNow(m) - draft0;
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        return true;
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        m.joinAhead(); // a prompt chunk's Engram read-ahead shares the host buffers a round's Engram reads use
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.request_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try d.check(d.api.cuMemcpyDtoH_v2(&h, m.rd.clk, h.len * 8), "cuMemcpyDtoH");
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        m.rd.clk_shard = marks == 4; // the sharded attention's parts alone (round.PhaseTimer.shardMark)
        m.rd.clk_ratio = if (marks == 3) 2 else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, m.amax);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0 .. R * m.vocab];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0..m.vocab];
    }

    /// The monotonic clock for `PassMeasure` (`ctx` is `*Model`).
    pub fn passNow(ctx: *anyopaque) u64 {
        return @as(*Model, @ptrCast(@alignCast(ctx))).now();
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, measure: ?*@import("draft.zig").PassMeasure) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null, measure);
    }
};

test "copyExtent's refusals come before anything is enqueued: pool bounds, parts, ratios, overlap, a stale source" {
    const R = Model.extentRefusal;
    const ratios = [_]usize{ 1, 2 };
    const cap: usize = 16384;
    // valid: replicated (no move), the same extent (a copy onto itself stays as before), an empty copy
    try std.testing.expectEqual(@as(?shard.Move, null), try R(cap, &ratios, null, 0, 4096, 2048, null));
    try std.testing.expectEqual(@as(?shard.Move, null), try R(cap, &ratios, null, 2048, 2048, 2048, null));
    _ = try R(cap, &ratios, null, 0, 14336, 2048, null); // ends exactly at the pool's end
    // bounds: either extent past the pool (no wrap-around through a huge `at`)
    try std.testing.expectError(error.ExtentOutOfPool, R(cap, &ratios, null, 15360, 0, 2048, null));
    try std.testing.expectError(error.ExtentOutOfPool, R(cap, &ratios, null, 0, 15360, 2048, null));
    try std.testing.expectError(error.ExtentOutOfPool, R(cap, &ratios, null, 20000, 0, 0, null));
    try std.testing.expectError(error.ExtentOutOfPool, R(cap, &ratios, null, 2048, 0, std.math.maxInt(usize) - 1024, null));
    // ratios: every source's rows whole
    try std.testing.expectError(error.BadView, R(cap, &ratios, null, 0, 4096, 2047, null));
    try std.testing.expectError(error.BadView, R(cap, &ratios, null, 1, 4096, 2048, null));
    // overlapping extents (a device copy over its own source): either direction
    try std.testing.expectError(error.ExtentsOverlap, R(cap, &ratios, null, 0, 1024, 2048, null));
    try std.testing.expectError(error.ExtentsOverlap, R(cap, &ratios, null, 1024, 0, 2048, null));
    _ = try R(cap, &ratios, null, 0, 2048, 2048, null); // adjacent: fine
    // sharded (two 8192-position parts): within a part the owner copies, across parts rank 0 sends to rank 1
    const sh = try shard.Shard.init(0, 2, cap, 48);
    try std.testing.expectEqual(shard.Move{ .local = .{ .from = 0, .to = 4096 } }, (try R(cap, &ratios, sh, 0, 4096, 2048, null)).?);
    try std.testing.expectEqual(shard.Move{ .send = .{ .from = 0, .peer = 1 } }, (try R(cap, &ratios, sh, 0, 8192, 2048, null)).?);
    try std.testing.expectError(error.CrossesPart, R(cap, &ratios, sh, 7168, 0, 2048, null));
    // a source pinned at its generation: refused once its rows changed, or when the pin does not cover them
    var gens = try Model.spill.Generations.init(std.testing.allocator, cap, 2048);
    defer gens.deinit(std.testing.allocator);
    const pin: Model.ExtentPin = .{ .gens = &gens, .src = try gens.pin(4096, 8192) };
    _ = try R(cap, &ratios, null, 4096, 0, 4096, pin);
    try std.testing.expectError(error.StaleExtent, R(cap, &ratios, null, 4096, 10240, 6144, pin)); // past the pin
    try std.testing.expectError(error.StaleExtent, R(cap, &ratios, null, 2048, 8192, 2048, pin)); // before it
    gens.bump(6000, 6001); // a stream wrote over one of its rows
    try std.testing.expectError(error.StaleExtent, R(cap, &ratios, null, 4096, 0, 4096, pin));
    // bounds come first, then the parts, then the ratios (each refusal alike on every rank)
    try std.testing.expectError(error.ExtentOutOfPool, R(cap, &ratios, sh, 15360, 7168, 2047, null));
    try std.testing.expectError(error.CrossesPart, R(cap, &ratios, sh, 7168, 0, 2047, null));
}

test "pickWindow: --context wins and must match TF_DS_WINDOW; a follower takes TF_DS_WINDOW, else rank 0's" {
    try std.testing.expectEqual(@as(?usize, 1048576), try Model.pickWindow(1048576, 2101248, null));
    try std.testing.expectEqual(@as(?usize, 1048576), try Model.pickWindow(1048576, 2101248, "1048576"));
    try std.testing.expectEqual(@as(?usize, 1048576), try Model.pickWindow(1048576, 2101248, " "));
    try std.testing.expectError(error.BadWindow, Model.pickWindow(1048576, 2101248, "65536"));
    try std.testing.expectError(error.BadWindow, Model.pickWindow(null, 2101248, "x"));
    try std.testing.expectEqual(@as(?usize, 65536), try Model.pickWindow(null, 2101248, "65536"));
    try std.testing.expectEqual(@as(?usize, 2101248), try Model.pickWindow(null, 2101248, "9999999")); // never past the pool
    try std.testing.expectEqual(@as(?usize, null), try Model.pickWindow(null, 2101248, null));
}

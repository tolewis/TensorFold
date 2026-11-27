//! Inkling 2 Mini's forward on one GPU over cache slots: 48 visit slots a stream (40 sliding rings, 8 whole caches,
//! 48 conv states), prompt chunks of one stream and decode rows of many, embedding to soft-capped logits.
const std = @import("std");
const fused_ = @import("fused.zig");
const cuda = @import("cuda");
const config = @import("config.zig");
const wts = @import("mini_weights.zig");
const attn = @import("kernels.zig");
const ops = @import("ops.zig");
const loop = @import("loop.zig");
const visit = @import("mini_visit.zig");
const mini_vision = @import("mini_vision.zig");

const DeviceBuffer = cuda.DeviceBuffer;

/// Prompt rows a forward takes at most: one stream's chunk.
pub const prompt_rows = 4096;

const WideGraphKey = struct {
    rows: u16,
    streams: u16,
    n_tiles: u16,
    hist_on: bool,
    lazy: bool,
    taps_on: bool,
};
const WideGraph = struct {
    key: WideGraphKey,
    exec: cuda.graph.Exec,
};
const wide_graph_cap = 64;
/// Decode rows a forward takes over every stream.
pub const round_rows = 64;
/// The widest window one stream verifies in a decode forward.
pub const window_rows = 16;
/// Conv sites a visit keeps state for: the key and value convs, after attention, after the MLP.
pub const Site = enum { k, v, attn, mlp };
/// The visits whose input state a drafter reads (DSpark's taps 5, 23, 41), then the final normed row (tap 48).
pub const tap_visits = [_]u32{ 5, 23, 41 };
pub const taps_n = 4;

/// A stream's rows in one forward: its cache slot, first position and tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

pub const Kernels = struct {
    attn: attn.Kernels,
    ops: ops.Kernels,
    loop: loop.Kernels,
    lane: cuda.qmmf.Lane,
    dense: cuda.dense_rows.DenseRows,
    nv: cuda.experts.Nvfp4Experts,
    mx: cuda.experts_mx.Mxfp8Experts,
    glue: cuda.moe_glue.MoeGlue,
    plan_mod: cuda.Module,
    router: cuda.grouped.Router,
    mtp_mod: cuda.Module, // mtp.cu: the taps a drafter reads and the conv backups a rollback restores from
    tap_fn: cuda.Function,
    backup_fn: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var k: Kernels = undefined;
        k.attn = try attn.Kernels.load(d);
        errdefer k.attn.unload();
        k.ops = try ops.Kernels.load(d);
        errdefer k.ops.unload();
        k.loop = try loop.Kernels.load(d);
        errdefer k.loop.unload();
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        // decode projections as dependent launches: their weight prologue overlaps the previous kernel's tail
        // (qmmf_kernel waits on the grid before it reads its inputs); TF_INKLING2_PDL=1 (measured <1% on GB10, so off)
        // qmmf's SM12x bf16 prompt tile for wide prompt projections (+6% prefill on GB10); TF_INKLING2_PROMPT_TILE=0: off
        const tile = if (std.c.getenv("TF_INKLING2_PROMPT_TILE")) |v| v[0] != '0' else true;
        if (tile) try k.lane.loadPrompt(d);
        k.lane.pdl = if (std.c.getenv("TF_INKLING2_PDL")) |v| v[0] == '1' else false;
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        k.nv = try cuda.experts.Nvfp4Experts.load(d, sms);
        errdefer k.nv.unload();
        // TF_INKLING2_EXPERT_SPLIT=0: decode experts one warp a 32-column unit, as before (the split gives the same bits)
        if (if (std.c.getenv("TF_INKLING2_EXPERT_SPLIT")) |v| v[0] != '0' else true) try k.nv.loadSplit(sms);
        k.mx = try cuda.experts_mx.Mxfp8Experts.load(d, sms);
        errdefer k.mx.unload();
        k.glue = try cuda.moe_glue.MoeGlue.load(d);
        errdefer k.glue.unload();
        k.plan_mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.plan_mod.unload();
        k.router = try cuda.grouped.Router.resolve(k.plan_mod);
        errdefer k.plan_mod.unload();
        k.mtp_mod = try cuda.Module.load(d, cuda.kernels.inkling2_mtp);
        errdefer k.mtp_mod.unload();
        k.tap_fn = try k.mtp_mod.function("tf_ink2m_tap");
        k.backup_fn = try k.mtp_mod.function("tf_ink2m_conv_backup");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mtp_mod.unload();
        k.plan_mod.unload();
        k.glue.unload();
        k.mx.unload();
        k.nv.unload();
        k.dense.unload();
        k.lane.unload();
        k.loop.unload();
        k.ops.unload();
        k.attn.unload();
    }

    pub fn experts(k: *const Kernels) cuda.experts.Grouped {
        return .{ .nvfp4 = &k.nv, .mxfp8 = &k.mx };
    }
};

/// The forward's device scratch, sized for `prompt_rows` rows (decode rounds use the first `round_rows`).
pub const Scratch = struct {
    ids: u64, // int32 [R]
    pos: u64, // int32 [R]
    seg_of: u64, // int32 [R]: the row's stream in the call
    segs: [2]u64, // attention Seg [streams]: sliding, then full caches (offsets differ by cache length)
    conv_segs: u64, // int4 [streams]: row0, rows, slot
    tiles: u64, // attention Tile [max tiles]
    h: [2]u64, // bf16 [R, H]: the running state in, the next visit's out
    blocks: u64, // bf16 [max blocks][R, H]
    bank: u64, // bf16 [bank blocks][R, H]: the previous pass's blocks past its entry
    bank_h: u64, // bf16 [R, H]: the previous pass's output
    msg: u64, // fp32 [4][R, H]
    msg_all: u64, // fp32 [core layers][4][rows, H]: a decode pass's messages, all formed at its restart
    mixf: u64, // fp32 [3][R, H]
    gdown: u64, // bf16 [3][R][rank]
    xin: u64, // bf16 [3][R, H]
    q: u64,
    kraw: u64,
    vraw: u64,
    rel: u64,
    gate: u64,
    sink: u64,
    gys: u64, // fp32 [3][rows, H]: the gates' rows before their norm
    fused: u64, // a one-row forward's fused projections, Fused.all bf16 columns (q, gate, rel, sink, k, v)
    dyn: [2]u64, // fp32 [R, kv * 4]
    kc: u64,
    vc: u64,
    qp: u64,
    relp: u64,
    tab: u64,
    part_o: u64,
    part_ml: u64,
    att: u64,
    o: u64,
    h32: u64, // fp32 [R, H]
    xm: u64,
    scores: u64,
    pick: u64,
    rw: u64,
    plan: cuda.grouped.Plan,
    act: u64,
    y: u64,
    x_tc_codes: u64, // pre-quantized FP4/MXFP8 activation codes for TC prefill
    x_tc_scales: u64, // pre-quantized UE4M3/UE8M0 activation scales for TC prefill
    moe: u64,
    xf: u64,
    logits: u64, // fp32 [round_rows, unpadded vocab]
    lane_part: u64,
    entries: usize, // (row, head, partial) slots part_o and part_ml hold
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    c: *const config.Text,
    context: usize, // positions a slot holds
    slots: usize,
    sliding: attn.Layer,
    full: attn.Layer,
    kv: [2][config.max_visits]u64 = undefined, // a visit's K and V: [slots][kv][len][128] bf16 (kv8: e4m3 codes)
    kv8: bool = false, // --kv-cache fp8: codes, and an fp32 scale a (slot, key head, position) in `kvs`
    kvs: [2][config.max_visits]u64 = @splat(@splat(0)), // kv8: a visit's K and V scales [slots][kv][len]
    conv: [config.max_visits][4]u64 = undefined, // a visit's conv states: fp32 [slots][3][C] by Site
    s: Scratch = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated
    /// Decode forwards of n one-row windows as CUDA graphs, captured at first use ([n]); TF_INKLING2_GRAPHS=0: off.
    graphs: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    /// The same forwards with the certified lazy head (`lazy_ok` set by the caller: every row greedy).
    graphs_lazy: [round_rows + 1]?cuda.graph.Exec = @splat(null),
    /// Drafted / multi-row verify rounds as CUDA graphs, keyed by (rows, streams, n_tiles, hist_on, lazy, taps_on).
    wide_graphs: [wide_graph_cap]?WideGraph = @splat(null),
    wide_graph_next: usize = 0,
    graphed: bool = true,
    /// TF_INKLING2_LAZY_HEAD=1: an FP8 proxy of the head with per-row error bounds; exact logits for the candidates.
    lazy: ?ops.Lazy = null,
    /// TF_INKLING2_ATTN_FP8=1 (Track D, lossy, opt-in): each layer's attention projections as per-tensor FP8 copies.
    attn8: ?[]wts.Attn = null,
    /// TF_INKLING2_FUSED: the fused visit's module and buffers (null: switch off, nothing loaded)
    fused: ?fused_.State = null,
    lazy_ok: bool = false,
    lazy_rows: [ops.Lazy.max_rows]ops.Lazy.Row = @splat(.{}), // the head rows' draws, uploaded with the forward
    /// Each conv site's inputs a row, fp32 [round_rows][C] by (visit, site), when `hist_on` (a verify of wide windows).
    hist: [config.max_visits][4]u64 = undefined,
    hist_base: u64 = 0,
    hist_on: bool = false,
    /// Every slot's conv states as a wide window found them: fp32, `backup_slot` bytes a slot.
    backup: u64 = 0,
    backup_slot: usize = 0,
    /// ops.cu's RewindSite for every (visit, site), on the device; `widest`: the largest C.
    sites: u64 = 0,
    n_sites: usize = 0,
    widest: usize = 0,
    /// The taps of the last forward's rows, bf16 [prompt_rows][taps_n][H], written once `taps_on`.
    taps: u64 = 0,
    taps_on: bool = false,
    /// Forwards run so far: a view of `taps` or `hist` is valid only within its forward's generation.
    gen: u64 = 0,
    /// Optional vision tower/image rows manager and the active prompt chunk's slot & starting position.
    vision: ?*mini_vision.Images = null,
    vision_slot: ?usize = null,
    vision_p0: usize = 0,

    fn alloc(m: *Model, bytes: usize) !u64 {
        if (m.planned) |*p| {
            p.* += std.mem.alignForward(usize, @max(bytes, 256), 256);
            return 0;
        }
        var b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    fn zeros(m: *Model, s: cuda.Stream, bytes: usize) !u64 {
        const p = try m.alloc(bytes);
        if (m.planned == null) try m.d.check(m.d.api.cuMemsetD8Async(p, 0, bytes, s.handle), "cuMemsetD8Async");
        return p;
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const wts.Weights, context: usize, slots: usize, kv8: bool) Model {
        return .{
            .kv8 = kv8,
            .gpa = gpa,
            .d = d,
            .w = w,
            .c = &w.config.text,
            .context = context,
            .slots = slots,
            .sliding = attn.Layer.sliding(w.config.text.window, prompt_rows),
            .full = attn.Layer.global(@intCast(context)),
            .graphed = graphsOn(),
        };
    }

    fn graphsOn() bool {
        if (fused_.checking()) return false; // the check synchronizes and reads back inside the forward
        const v = std.c.getenv("TF_INKLING2_GRAPHS") orelse return true;
        return v[0] != '0';
    }

    /// The caches and scratch of `slots` streams at `context` positions.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize, kv8: bool) !Model {
        var m = blank(gpa, d, w, context, slots, kv8);
        errdefer m.deinit();
        try m.build(s, k);
        try s.synchronize();
        if (fused_.read().mode != .off and m.attn8 != null) {
            m.fused = try fused_.State.init(d);
            if (fused_.checking()) try fused_.reportLoad(&m, k);
        }
        return m;
    }

    /// The device bytes init would allocate.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize, slots: usize, kv8: bool) !usize {
        var m = blank(gpa, d, w, context, slots, kv8);
        m.planned = 0;
        try m.build(s, k);
        return m.planned.?;
    }

    /// One stream's caches: 40 sliding rings, 8 whole caches and 48 conv states (Mini).
    pub fn streamBytes(m: *const Model) usize {
        const t = m.c;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        const v = &m.w.plan.visits;
        const caches = v.count(.sliding) * m.sliding.ring + v.count(.full) * m.full.ring;
        const per: usize = if (m.kv8) kvw + 4 * @as(usize, t.local.kv) else kvw * 2; // a position's K (or V) bytes
        return caches * per * 2 + 2 * v.count(.conv) * 3 * (2 * kvw + 2 * @as(usize, t.hidden)) * 4;
    }

    pub fn deinit(m: *Model) void {
        if (m.attn8) |a| m.gpa.free(a);
        for ([_]*[round_rows + 1]?cuda.graph.Exec{ &m.graphs, &m.graphs_lazy }) |set| for (set) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
        for (&m.wide_graphs) |*wg| if (wg.*) |*e| {
            e.exec.deinit();
            wg.* = null;
        };
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    fn build(m: *Model, s: cuda.Stream, k: *const Kernels) !void {
        const t = m.c;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        var widths: usize = 0; // fp32 columns of every site of every visit
        for (m.w.plan.visits.slice(), 0..) |step, i| {
            const len: usize = if (step.of(.full) != null) m.full.ring else m.sliding.ring;
            for (0..2) |side| {
                m.kv[side][i] = try m.alloc(m.slots * len * kvw * @as(usize, if (m.kv8) 1 else 2));
                if (m.kv8) m.kvs[side][i] = try m.alloc(m.slots * len * @as(usize, t.local.kv) * 4);
            }
            for (std.enums.values(Site)) |site| {
                const width: usize = if (site == .k or site == .v) kvw else t.hidden;
                m.conv[i][@intFromEnum(site)] = try m.zeros(s, m.slots * 3 * width * 4);
                widths += width;
            }
        }
        // rollback: each site's inputs a verify row, and each slot's states before a wide verify
        m.hist_base = try m.alloc(widths * round_rows * 4);
        m.backup_slot = widths * 3 * 4;
        m.backup = try m.alloc(m.slots * m.backup_slot);
        m.n_sites = m.w.plan.visits.len * 4;
        m.sites = try m.alloc(m.n_sites * @sizeOf(ops.RewindSite));
        m.taps = try m.alloc(prompt_rows * taps_n * @as(usize, t.hidden) * 2);
        var table: [config.max_visits * 4]ops.RewindSite = undefined;
        var hist_off: usize = 0;
        var backup_off: usize = 0;
        for (0..m.w.plan.visits.len) |i| for (std.enums.values(Site)) |site| {
            const width: usize = if (site == .k or site == .v) kvw else t.hidden;
            const j = i * 4 + @intFromEnum(site);
            m.hist[i][@intFromEnum(site)] = m.hist_base + hist_off;
            table[j] = .{ .state = m.conv[i][@intFromEnum(site)], .hist = @intCast(hist_off), .backup = @intCast(backup_off), .c = @intCast(width) };
            hist_off += width * round_rows * 4;
            backup_off += width * 3 * 4;
            m.widest = @max(m.widest, width);
        };
        if (m.planned == null) try m.upload(s, m.sites, std.mem.sliceAsBytes(table[0..m.n_sites]));
        try m.scratch(k);
        if (lazyOn()) try m.buildLazy(s, k);
        if (attnFp8On()) try m.buildAttn8(s, k);
    }

    fn attnFp8On() bool {
        const v = std.c.getenv("TF_INKLING2_ATTN_FP8") orelse return false;
        return v[0] == '1';
    }

    /// A packed bf16 lane weight as a per-tensor FP8 lane weight (e4m3 at amax / 448; mtp.cu's repack), its slices kept.
    fn fp8Lane(m: *Model, s: cuda.Stream, k: *const Kernels, u: cuda.qmmf.Weight) !cuda.qmmf.Weight {
        const total: usize = @as(usize, u.npad) * u.k;
        const codes = try m.alloc(total);
        const amax = try m.alloc(4);
        if (m.planned != null) return u;
        try m.d.check(m.d.api.cuMemsetD8Async(amax, 0, 4, s.handle), "cuMemsetD8Async");
        var a: cuda.Args = .{};
        a.add(u.codes);
        a.add(@as(i64, @intCast(total)));
        a.add(amax);
        try cuda.launch.launch(try k.mtp_mod.function("tf_ink2m_amax"), .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, s, &a);
        try s.synchronize();
        var bits: u32 = 0;
        try cuda.memory.download(m.d, amax, std.mem.asBytes(&bits));
        const mx: f32 = @bitCast(bits);
        if (!(mx > 0) or !std.math.isFinite(mx)) return error.AttnFp8Range;
        const scale = mx / 448.0;
        var b: cuda.Args = .{};
        b.add(u.codes);
        b.add(codes);
        b.add(@as(i32, @intCast(u.k)));
        b.add(@as(f32, 1.0 / scale));
        b.add(@as(i64, @intCast(total)));
        try cuda.launch.launch(try k.mtp_mod.function("tf_ink2m_fp8_lane"), .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &b);
        var w = cuda.modelopt.fp8Weight(codes, scale, u.n, u.k);
        w.sk = u.sk;
        return w;
    }

    /// Each layer's attention with FP8 projections: the fused rows once (its part views at the same rows, one scale),
    /// and the output projection.
    fn buildAttn8(m: *Model, s: cuda.Stream, k: *const Kernels) !void {
        const t = m.c;
        const f = wts.Fused.of(t);
        const out = if (m.planned == null) try m.gpa.alloc(wts.Attn, m.w.layers.len) else null;
        for (m.w.layers, 0..) |*L, i| {
            const a = L.attn;
            const all8 = try m.fp8Lane(s, k, a.all);
            const wo8 = try m.fp8Lane(s, k, a.wo);
            if (out) |o| {
                var b = a;
                const view = struct {
                    fn at(base: cuda.qmmf.Weight, row: usize, n: u32, part: cuda.qmmf.Weight) cuda.qmmf.Weight {
                        var w = cuda.modelopt.fp8Weight(base.codes + row * base.k, base.scale, n, base.k);
                        w.sk = part.sk;
                        return w;
                    }
                }.at;
                b.all = view(all8, 0, a.all.n, a.all);
                b.qx = view(all8, 0, a.qx.n, a.qx);
                b.wq = view(all8, f.q, a.wq.n, a.wq);
                b.gate = view(all8, f.gate, a.gate.n, a.gate);
                b.wr = view(all8, f.rel, a.wr.n, a.wr);
                b.sink = view(all8, f.sink, a.sink.n, a.sink);
                b.wk = view(all8, f.k, a.wk.n, a.wk);
                b.wv = view(all8, f.v, a.wv.n, a.wv);
                b.wo = wo8;
                o[i] = b;
            }
        }
        if (out) |o| {
            try s.synchronize();
            m.attn8 = o;
        }
    }

    /// TF_INKLING2_LAZY_HEAD=0: every head row through the full bf16 head, as before (the lazy head picks the same).
    fn lazyOn() bool {
        const v = std.c.getenv("TF_INKLING2_LAZY_HEAD") orelse return true;
        return v[0] != '0';
    }

    /// The lazy head's FP8 proxy (per-tensor e4m3 at amax / 448, mtp.cu's repack) and each row's error bound.
    fn buildLazy(m: *Model, s: cuda.Stream, k: *const Kernels) !void {
        const u = m.w.unembed;
        const total: usize = @as(usize, u.npad) * u.k;
        const R = ops.Lazy.max_rows;
        var z: ops.Lazy = undefined;
        const codes = try m.alloc(total);
        z.delta = try m.alloc(@as(usize, u.n) * 4);
        z.xn = try m.alloc(R * 4);
        z.lg = try m.alloc(R * 3 * 4);
        z.info = try m.alloc(R * @sizeOf(ops.Lazy.Row));
        z.count = try m.alloc(R * 4);
        z.cand = try m.alloc(R * ops.Lazy.cands * 4);
        const amax = try m.alloc(4);
        if (m.planned != null) return;
        try m.d.check(m.d.api.cuMemsetD8Async(amax, 0, 4, s.handle), "cuMemsetD8Async");
        var a: cuda.Args = .{};
        a.add(u.codes);
        a.add(@as(i64, @intCast(total)));
        a.add(amax);
        try cuda.launch.launch(try k.mtp_mod.function("tf_ink2m_amax"), .{ .grid = .{ .x = 1024 }, .block = .{ .x = 256 } }, s, &a);
        try s.synchronize();
        var bits: u32 = 0;
        try cuda.memory.download(m.d, amax, std.mem.asBytes(&bits));
        const mx: f32 = @bitCast(bits);
        if (!(mx > 0) or !std.math.isFinite(mx)) return error.LazyHeadRange;
        const scale = mx / 448.0;
        var b: cuda.Args = .{};
        b.add(u.codes);
        b.add(codes);
        b.add(@as(i32, @intCast(u.k)));
        b.add(@as(f32, 1.0 / scale));
        b.add(@as(i64, @intCast(total)));
        try cuda.launch.launch(try k.mtp_mod.function("tf_ink2m_fp8_lane"), .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &b);
        z.proxy = cuda.modelopt.fp8Weight(codes, scale, u.n, u.k);
        try ops.lazyDelta(&k.ops, s, u, 1.0 / scale, scale, z.delta);
        try s.synchronize();
        m.lazy = z;
    }

    /// `slot`'s conv states, every site, into its backup (before a verify whose window may be cut).
    pub fn backupSlot(m: *const Model, k: *const Kernels, s: cuda.Stream, slot: usize) !void {
        var a: cuda.Args = .{};
        a.add(m.sites);
        a.add(m.backup + slot * m.backup_slot);
        a.add(@as(i32, @intCast(slot)));
        try cuda.launch.launch(k.backup_fn, .{ .grid = .{ .x = @intCast((m.widest + 127) / 128), .y = @intCast(m.n_sites) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// `slot`'s last wide window keeps its first `n` rows (rows [row0, row0 + n) of that forward): its conv states.
    pub fn rewindSlot(m: *const Model, k: *const Kernels, s: cuda.Stream, slot: usize, row0: usize, n: usize) !void {
        try ops.rewind(&k.ops, s, m.sites, m.n_sites, m.widest, m.backup + slot * m.backup_slot, m.hist_base, slot, row0, n);
    }

    /// Rows [0, rows) of the forward's state `src` (bf16 [rows][H]) as tap `tap` of `m.taps`.
    pub fn tap(m: *const Model, k: *const Kernels, s: cuda.Stream, src: u64, rows: usize, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(m.taps);
        a.add(@as(i32, @intCast(m.c.hidden)));
        a.add(@as(i32, taps_n));
        a.add(@as(i32, @intCast(t)));
        try cuda.launch.launch(k.tap_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Bytes of one tapped row (every tap of one position).
    pub fn tapRowBytes(m: *const Model) usize {
        return taps_n * @as(usize, m.c.hidden) * 2;
    }

    /// Captured graphs forget their launches (the taps were turned on after them).
    pub fn dropGraphs(m: *Model) void {
        for ([_]*[round_rows + 1]?cuda.graph.Exec{ &m.graphs, &m.graphs_lazy }) |set| for (set) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
        for (&m.wide_graphs) |*wg| if (wg.*) |*e| {
            e.exec.deinit();
            wg.* = null;
        };
    }

    fn scratch(m: *Model, k: *const Kernels) !void {
        const t = m.c;
        const r: usize = prompt_rows;
        const h: usize = t.hidden;
        const q: usize = @as(usize, t.local.q) * t.local.dim;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        const heads: usize = t.local.q;
        const rank: usize = t.loop.?.rank;
        const pairs = r * t.topk;
        const blocks = (t.layers + t.block - 1) / t.block;
        // partials: every prompt row's sliding or full chunks, or a split decode call's sub-chunks
        const stride_full = attn.slotsFor(m.full, m.context);
        const entries = @max(r * heads * @max(attn.slotsFor(m.sliding, m.context), stride_full), attn.split_rows * heads * stride_full * attn.subs);
        var part: usize = 0;
        for (m.w.layers) |l| for ([_]cuda.qmmf.Weight{ l.attn.qx, l.attn.all }) |w| {
            part = @max(part, k.lane.partBytes(1, w));
        };
        for (m.w.layers) |l| for ([_]cuda.qmmf.Weight{ l.attn.wq, l.attn.wk, l.attn.wv, l.attn.wr, l.attn.gate, l.attn.sink, l.attn.wo, l.attn.all, l.attn.qx }) |w| {
            for (1..r + 1) |rows| part = @max(part, k.lane.partBytes(rows, w));
        };
        for (1..round_rows + 1) |rows| part = @max(part, k.lane.partBytes(rows, m.w.unembed));
        const items = cuda.grouped.maxItems(pairs, t.experts, 64);
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .seg_of = try m.alloc(r * 4),
            .segs = .{ try m.alloc(round_rows * @sizeOf(attn.Seg)), try m.alloc(round_rows * @sizeOf(attn.Seg)) },
            .conv_segs = try m.alloc(round_rows * 16),
            .tiles = try m.alloc((attn.tileCount(r, 3) + round_rows) * @sizeOf(attn.Tile)),
            .h = .{ try m.alloc(r * h * 2), try m.alloc(r * h * 2) },
            .blocks = try m.alloc(blocks * r * h * 2),
            .bank = try m.alloc(@max(1, m.w.plan.bank_blocks) * r * h * 2),
            .bank_h = try m.alloc(r * h * 2),
            .msg = try m.alloc(4 * r * h * 4),
            .msg_all = try m.alloc(@as(usize, m.w.plan.visits.loop.end - m.w.plan.visits.loop.start) * 4 * round_rows * h * 4),
            .mixf = try m.alloc(3 * r * h * 4),
            .gdown = try m.alloc(3 * r * rank * 2),
            .xin = try m.alloc(3 * r * h * 2),
            .q = try m.alloc(r * q * 2),
            .kraw = try m.alloc(r * kvw * 2),
            .vraw = try m.alloc(r * kvw * 2),
            .rel = try m.alloc(r * heads * t.d_rel * 2),
            .gate = try m.alloc(r * q * 2),
            .sink = try m.alloc(r * heads * 2),
            .gys = try m.alloc(3 * r * h * 4),
            .fused = try m.alloc(r * wts.Fused.of(t).all * 2), // r rows: the n > 1 fused projections too
            .dyn = .{ try m.alloc(r * t.global.kv * config.conv_taps * 4), try m.alloc(r * t.global.kv * config.conv_taps * 4) },
            .kc = try m.alloc(r * kvw * 2),
            .vc = try m.alloc(r * kvw * 2),
            .qp = try m.alloc(r * q * 2),
            .relp = try m.alloc(r * heads * t.d_rel * 2),
            .tab = try m.alloc(attn.tableBytes(r, @intCast(heads))),
            .part_o = try m.alloc(entries * attn.head_dim * 4),
            .part_ml = try m.alloc(entries * 8),
            .att = try m.alloc(r * q * 2),
            .o = try m.alloc(r * h * 2),
            .h32 = try m.alloc(r * h * 4),
            .xm = try m.alloc(r * h * 2),
            .scores = try m.alloc(r * t.experts * 4),
            .pick = try m.alloc(pairs * 4),
            .rw = try m.alloc(pairs * 4),
            .plan = .{
                .members = try m.alloc(pairs * 4),
                .items = try m.alloc(items * 12),
                .counts = try m.alloc(8),
                .rank = try m.alloc(pairs * 4),
                .hist = try m.alloc(((pairs + 1023) / 1024) * t.experts * 4),
            },
            .act = try m.alloc(pairs * t.expert_inter * 2),
            .y = try m.alloc(pairs * h * 4),
            .x_tc_codes = try m.alloc(@max(r * h, pairs * t.expert_inter)),
            .x_tc_scales = try m.alloc(@max(r * (h / 16), pairs * (t.expert_inter / 16))),
            .moe = try m.alloc(r * h * 2),
            .xf = try m.alloc(r * h * 2),
            .logits = try m.alloc(round_rows * t.unpadded_vocab * 4),
            .lane_part = try m.alloc(part),
            .entries = entries,
        };
    }

    pub fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    pub fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    /// A slot's conv states to zero: a stream starting at position 0 has no inputs before it.
    pub fn clear(m: *Model, s: cuda.Stream, slot: usize) !void {
        const t = m.c;
        const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
        for (0..m.w.plan.visits.len) |v| for (std.enums.values(Site)) |site| {
            const width: usize = if (site == .k or site == .v) kvw else t.hidden;
            const at = m.conv[v][@intFromEnum(site)] + slot * 3 * width * 4;
            try m.d.check(m.d.api.cuMemsetD8Async(at, 0, 3 * width * 4, s.handle), "cuMemsetD8Async");
        };
    }

    /// Every chain's rows (one prompt chunk, or decode windows); rows [head0, head0 + heads)'s fp32 logits at s.logits.
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, chains: []const Chain, prompt: bool, head0: usize, heads: usize) !void {
        var rows: usize = 0;
        for (chains, 0..) |ch, i| {
            if (ch.tokens.len == 0 or ch.slot >= m.slots or ch.p0 + ch.tokens.len > m.context) return error.BadChain;
            if (!prompt and ch.tokens.len > window_rows) return error.WindowTooWide;
            for (chains[0..i]) |o| if (o.slot == ch.slot) return error.BadChain;
            for (ch.tokens) |id| if (id >= m.c.vocab) return error.InvalidToken;
            rows += ch.tokens.len;
        }
        if (rows == 0 or rows > @as(usize, if (prompt) prompt_rows else round_rows) or (prompt and chains.len != 1)) return error.TooManyRows;
        if (chains.len > round_rows or heads > round_rows or head0 + heads > rows) return error.TooManyRows;
        for (chains) |ch| if (ch.p0 == 0) try m.clear(s, ch.slot);
        m.gen += 1;
        var r = try visit.Rows.stage(m, s, chains, rows, prompt);
        if (m.lazyFor(heads)) try m.upload(s, m.lazy.?.info, std.mem.sliceAsBytes(m.lazy_rows[0..heads]));
        // Decode forwards with every row's logits: launches depend only on (rows, streams, n_tiles, hist_on, lazy, taps_on).
        if (prompt or !m.graphed or head0 != 0 or heads != rows) {
            try visit.run(m, k, s, &r, head0, heads);
            if (m.lazyFor(heads)) try m.lazyTrace(s, heads);
            return;
        }
        if (!m.hist_on and r.streams == rows and r.n_tiles == rows) {
            const g = if (m.lazyFor(heads)) &m.graphs_lazy[rows] else &m.graphs[rows];
            if (g.* == null) g.* = try m.capture(k, s, &r);
            try g.*.?.launchOn(s);
            if (m.lazyFor(heads)) try m.lazyTrace(s, heads);
            return;
        }
        if (wideGraphsOn()) {
            const key: WideGraphKey = .{
                .rows = @intCast(rows),
                .streams = @intCast(r.streams),
                .n_tiles = @intCast(r.n_tiles),
                .hist_on = m.hist_on,
                .lazy = m.lazyFor(heads),
                .taps_on = m.taps_on,
            };
            for (&m.wide_graphs) |*wg| {
                if (wg.*) |*e| {
                    if (std.meta.eql(e.key, key)) {
                        try e.exec.launchOn(s);
                        if (m.lazyFor(heads)) try m.lazyTrace(s, heads);
                        return;
                    }
                }
            }
            const exec = try m.capture(k, s, &r);
            const idx = m.wide_graph_next % wide_graph_cap;
            m.wide_graph_next +%= 1;
            if (m.wide_graphs[idx]) |*old| old.exec.deinit();
            m.wide_graphs[idx] = .{ .key = key, .exec = exec };
            try m.wide_graphs[idx].?.exec.launchOn(s);
            if (m.lazyFor(heads)) try m.lazyTrace(s, heads);
            return;
        }
        try visit.run(m, k, s, &r, head0, heads);
        if (m.lazyFor(heads)) try m.lazyTrace(s, heads);
    }

    /// TF_INKLING2_WIDE_GRAPHS=0: disable CUDA graphs on drafted/multi-row verify rounds.
    fn wideGraphsOn() bool {
        const S = struct {
            var v: ?bool = null;
        };
        if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_WIDE_GRAPHS")) |e| e[0] != '0' else true;
        return S.v.?;
    }

    /// TF_INKLING2_LAZY_TRACE=1: the lazy head's candidates a row (synchronizes; debug only), every 256 forwards.
    pub fn lazyTrace(m: *Model, s: cuda.Stream, heads: usize) !void {
        if (std.c.getenv("TF_INKLING2_LAZY_TRACE") == null) return;
        const T = struct {
            var hist: [12]u64 = @splat(0); // <=1, 2, <=4, ..., <=1024 (by powers of two), overflow
            var calls: u64 = 0;
            var most: i32 = 0;
            var race: u64 = 0;
        };
        try s.synchronize();
        var cnt: [ops.Lazy.max_rows]i32 = undefined;
        try cuda.memory.download(m.d, m.lazy.?.count, std.mem.sliceAsBytes(cnt[0..heads]));
        for (cnt[0..heads], 0..) |c, i| {
            const b: usize = if (c > ops.Lazy.cands) 11 else @min(10, std.math.log2_int_ceil(u32, @intCast(@max(c, 1))));
            T.hist[b] += 1;
            T.most = @max(T.most, c);
            T.race += m.lazy_rows[i].mode;
        }
        T.calls += 1;
        if (T.calls % 256 == 0) std.debug.print("[lazy] {d} forwards, rows by candidates (<=1,2,4..1024,over): {any}; most {d}; race rows {d}\n", .{ T.calls, T.hist, T.most, T.race });
    }

    /// TF_INKLING2_LAZY_ROWS: the most head rows a lazy forward takes (default a whole round, ops.Lazy.max_rows; 16 as first measured).
    fn lazyRows() usize {
        const S = struct {
            var v: ?usize = null;
        };
        if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_LAZY_ROWS")) |e| @min(ops.Lazy.max_rows, std.fmt.parseInt(usize, std.mem.span(e), 10) catch ops.Lazy.max_rows) else ops.Lazy.max_rows;
        return S.v.?;
    }

    /// Whether this forward's head is the lazy one: built, asked for, and few enough rows.
    pub fn lazyFor(m: *const Model, heads: usize) bool {
        return m.lazy != null and m.lazy_ok and heads > 0 and heads <= lazyRows();
    }

    /// A decode forward of `r.n` one-row windows, captured as a graph.
    fn capture(m: *Model, k: *const Kernels, s: cuda.Stream, r: *const visit.Rows) !cuda.graph.Exec {
        try s.synchronize();
        try cuda.graph.beginCapture(s, .thread_local);
        visit.run(m, k, s, r, 0, r.n) catch |e| {
            if (cuda.graph.endCapture(s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return e;
        };
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        return g.instantiate();
    }
};

test "the forward's launches type-check against the shared operators" {
    std.testing.refAllDecls(Model);
    std.testing.refAllDecls(Kernels);
}

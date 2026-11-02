//! The checkpoint's own MTP head as the drafter on the XPU 27B path: XPU_MTP=1, XPU_MTP_K drafts (default 3).
//!
//! One dense full-attention layer (mtp.*, EXL3): the head reads the target's final normed hidden state and the token
//! after it, and the chain follows the checkpoint's own head, as cuda_mtp.zig's hold does: the first draft is the
//! newest absorbed row's head, each later draft the previous row's output and the draft before it. Verify runs through
//! the target's own windows (verifyMulti/keepPaths), so the MTP only drafts.
//!
//! The fc is stored at 5 bits, the one width the EXL3 kernels have no image for: it is decoded on the host at load
//! (core.exl3_format, the reference decoder the kernels are qualified against) and runs as fp16 through the runner's
//! dense_rows kernel. Every other MTP matrix is 4 bits and runs through exl3.
//!
//! K/V: the MTP layer's own cache a slot, int8 (or bf16) at the runner's context, written by attn_prep and read by
//! attn_decode / attn_prompt8: the target's kernels with the MTP's buffers and its own page-free layout. The MTP's
//! calls are recorded in the TP step ring (tp.zig) so a follower rank runs the same all-reduces.
const std = @import("std");
const ze = @import("ze");
const core = @import("core");
const xpu = @import("xpu");
const exl3 = xpu.exl3;
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const ex = @import("qwen27_exl3.zig");
const tp = @import("tp.zig");
const fwd = @import("forward.zig");
const exf = core.exl3_format;
const alloc = q.alloc;
const Dev = dev.Dev;
const D = q.D;
const FFR = q.FF_R;
const HQ = q.ATT_Q_HEADS;
const AKV = q.ATT_KV_HEADS;
const ATT_IN = q.ATT_IN;
const OUT = q.GDN_Z;
const EPS = q.EPS;
const off = dev.off;
const copy = dev.copy;

/// XPU_MTP=1: the checkpoint's own MTP head drafts (off by default: DFlash2 through --drafter).
pub var on = false;
/// XPU_MTP_K: the drafts a round holds at most (1 to 15).
pub var most: u32 = 3;
/// XPU_MTP_SYNC=1: drain the list after every launch of a head pass. A device fault then names the launch that
/// caused it, instead of the drain of the whole pass (lane p29-mtpfix: the rank 1 fault at the first absorb).
pub var sync_each = false;

/// Prompt rows one absorb launch (a slice of a prompt chunk).
pub const SLICE = 128;
/// The rows a draft or keep absorb pass holds at most (the target's verify window; XPU_VERIFY_ROWS=32: 32).
pub const ROWS = q.MAX_ROWS;

/// The io of every MTP layer projection. The layer's activation rows are fp16: attn_gate, add_rmsnorm and swiglu all
/// write `half`, so a plan that reads them as bf16 misreads every value (lane p29-mtpfix: the o_proj's plan said bf16
/// and the attention half of every absorbed row was scaled by about 2^-7 before the o projection).
pub const ProjIO = struct { out: exl3.Output = .f16, in: exl3.Input = .f16 };
pub const layer_io: ProjIO = .{};

/// The cache position of a chained draft row: the newest absorbed position plus its step. cuda_mtp.zig's hold runs row
/// k at pos0 + k - 1 with pos0 = ctx.end, and ctx.end is the newest absorbed position plus one, so the first draft row
/// is one past the newest absorbed row: its K/V does not overwrite that row's K/V and its rope is at its own position.
/// Pure, so a host test covers it.
pub fn draftPos(newest: u32, step: u32) u32 {
    return newest + step;
}

test "a draft row's position is one past the newest absorbed row, and its step advances by one" {
    // newest 100: the absorbed rows are at 0..100; draft rows go to 101, 102, 103, never back onto 100
    try std.testing.expectEqual(@as(u32, 101), draftPos(100, 1));
    try std.testing.expectEqual(@as(u32, 102), draftPos(100, 2));
    try std.testing.expectEqual(@as(u32, 103), draftPos(100, 3));
    try std.testing.expectEqual(@as(u32, 1), draftPos(0, 1)); // a fresh slot: the prompt's last row is at 0
    var step: u32 = 1;
    while (step < 4) : (step += 1) try std.testing.expect(draftPos(100, step) > 100);
}

test "every MTP projection reads the layer's fp16 activation rows" {
    try std.testing.expectEqual(exl3.Input.f16, layer_io.in);
    try std.testing.expectEqual(exl3.Output.f16, layer_io.out);
}

/// Reads the two flags at the runner's init, before anything asks whether the head is on. `Mtp.load` reads them again
/// (it is also the path a test drives), so the flags are set from one place.
pub fn readEnv() void {
    const on_env = std.c.getenv("XPU_MTP");
    const k_env = std.c.getenv("XPU_MTP_K");
    if (std.c.getenv("XPU_MTP_SYNC")) |v| sync_each = v[0] != '0';
    applyEnv(
        if (on_env) |v| v[0] != '0' else null,
        if (k_env) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch null else null,
    );
}

/// The flags from an open's environment: `mtp` null when XPU_MTP is unset, `k` null when XPU_MTP_K is unset. Unset
/// turns the head off and restores K = 3, so a second open in one process does not inherit the first one's flags.
pub fn applyEnv(mtp: ?bool, k: ?u32) void {
    on = mtp orelse false;
    most = @min(@max(k orelse 3, 1), q.windowRows() - 1);
}

test "the MTP flags come from the open's environment, and unset turns them off" {
    const wr = q.windowRows();
    defer q.verify_rows = wr;
    applyEnv(true, 5);
    try std.testing.expect(on);
    try std.testing.expectEqual(@as(u32, 5), most);
    applyEnv(null, null); // a second open without XPU_MTP
    try std.testing.expect(!on);
    try std.testing.expectEqual(@as(u32, 3), most);
    applyEnv(false, 2); // XPU_MTP=0
    try std.testing.expect(!on);
    try std.testing.expectEqual(@as(u32, 2), most);
    applyEnv(true, 0); // K below the floor
    try std.testing.expectEqual(@as(u32, 1), most);
    applyEnv(true, 99); // K above the window
    try std.testing.expectEqual(wr - 1, most);
    q.verify_rows = 32; // XPU_VERIFY_ROWS=32 raises the cap with the window
    applyEnv(true, 99);
    try std.testing.expectEqual(@as(u32, 31), most);
    applyEnv(null, null);
}

/// The host record a kept state keeps with the MTP part: the cache's newest absorbed position and the drafts it held.
pub const SnapRecord = struct { pos: u32, chain_n: u32, chain: [ROWS]u32 };

/// One half of a slot's MTP cache (its K or its V) as the kernels index it: Q27_KVROW a position for the values, then
/// a float scale a head and position after `cap` positions.
pub const Layout = struct { heads: usize, cap: usize, int8: bool };

/// Bytes of one half of a slot's MTP cache: the values at every position, then the int8 scales.
pub fn halfBytes(l: Layout) usize {
    return l.cap * l.heads * 256 * @as(usize, if (l.int8) 1 else 2) + (if (l.int8) l.cap * l.heads * 4 else 0);
}

/// The MTP cache layout of this rank: its own K/V heads (half the model's at TP 2), as the kernels index it.
pub fn layout() Layout {
    return .{ .heads = AKV, .cap = q.MAX_CTX, .int8 = q.kv_i8 };
}

/// A copy a kept state's MTP part takes: `len` bytes of a cache half at `src` to the state's buffer at `dst`.
pub const Region = struct { src: usize, dst: usize, len: usize };

/// The copies of a slot's MTP cache at `at` positions: K values, K scales, V values, V scales (no scales at bf16).
pub fn snapRegions(l: Layout, at: usize) [4]Region {
    const row = l.heads * 256 * @as(usize, if (l.int8) 1 else 2);
    const half = halfBytes(l);
    const values = l.cap * l.heads * 256 * @as(usize, if (l.int8) 1 else 2);
    const sc = if (l.int8) at * l.heads * 4 else 0;
    return .{
        .{ .src = 0, .dst = 0, .len = at * row },
        .{ .src = values, .dst = at * row, .len = sc },
        .{ .src = half, .dst = at * row + sc, .len = at * row },
        .{ .src = half + values, .dst = 2 * at * row + sc, .len = sc },
    };
}

/// Bytes a kept state's MTP part takes at `at` positions: both halves' values and scales, and the newest row's state.
/// A state at 0 positions carries none.
pub fn snapBytes(at: u32) u64 {
    if (at == 0) return 0;
    const l = layout();
    const row: u64 = l.heads * 256 * @as(u64, if (l.int8) 1 else 2);
    return 2 * @as(u64, at) * row + (if (l.int8) 2 * @as(u64, at) * l.heads * 4 else 0) + D * 2;
}

/// One stream's kept rows of a verify: the slot, the position of its first kept row, the target's block rows of the
/// kept path (their final normed states are the head's inputs), and the token after each row.
pub const Kept = struct { slot: u32, pos: u32, block: []const u32, follow: []const u32 };
/// A stream's next window: its pending token and the drafts it asks for.
pub const Ask = struct { slot: u32, pending: u32, depth: u32 };
/// One lone round's result: the accepted tokens (the drafts that landed, then the target's token).
pub const Round = struct { tokens: []const u32, drafted: u32, accepted: u32 };

/// One projection's plans: rows 1 to q.windowRows() (decode and draft), and the prompt slice (SLICE rows).
const Plans = struct { dec: [ROWS]exl3.Plan = undefined, dec_n: u32 = 0, chunk: ?exl3.Plan = null };

const PlanSet = struct { qkv: Plans, o: Plans, gu: Plans, down: Plans };

pub const Mtp = struct {
    r: *q.Runner,
    w: *ex.Weights,
    // the EXL3 projections (mtp.layers.0.*) and the shared lm_head bound to the MTP's rows
    qkv: *ex.Proj,
    o: *ex.Proj,
    gu: *ex.Proj,
    down: *ex.Proj,
    head: *ex.Proj,
    plans: PlanSet,
    scratch: [4]*anyopaque,
    // the fc's two input halves (embedding inputs, then hidden inputs) as fp16 dense rows [N, D] on this rank
    fc_e: *anyopaque,
    fc_h: *anyopaque,
    fc_n: u32,
    // norms (bf16, centred as the checkpoint stores them)
    norm_e: *anyopaque,
    norm_h: *anyopaque,
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    q_norm: *anyopaque,
    k_norm: *anyopaque,
    norm: *anyopaque,
    // activations (SLICE rows; a draft uses the first rows)
    x: *anyopaque, // fp32 [SLICE, D]: the embedding, then the residual
    en: *anyopaque, // fp16 [SLICE, D]: normed embedding
    hn: *anyopaque, // fp16 [SLICE, D]: normed target state
    tgt: *anyopaque, // fp16 [SLICE, D]: the target's gathered final normed states
    h: *anyopaque, // fp16 [SLICE, D]: the layer's normed inputs and outputs
    pr: *anyopaque, // fp16 [SLICE, D]: a partial projection
    r2: *anyopaque,
    p: *anyopaque, // fp16 [SLICE, ATT_IN]: q | gate | k | v
    qb: *anyopaque, // bf16 [SLICE, HQ, 256]
    y: *anyopaque, // bf16 [SLICE, OUT]: the attention output
    guv: *anyopaque, // fp16 [SLICE, 2 FFR]
    act: *anyopaque, // fp16 [SLICE, FFR]
    ahi: *anyopaque, // fp16 [SLICE, FF / GS]
    alo: *anyopaque,
    logits: *anyopaque, // fp16 [ROWS, VOCAB]
    hb: *anyopaque, // fp16 [ROWS, D]: the rows the head reads
    ids: *anyopaque, // i32 [SLICE]: the follow tokens
    mt: *anyopaque, // u64 [ROWS, 8]: attn_prep/attn_decode's per-row cache addresses and positions
    mpos: *anyopaque, // i32 [ROWS]
    zero: *anyopaque, // fp32 [SLICE, D], zeros: a draft's residual starts here
    kv: [q.MAX_SLOTS]?*anyopaque = @splat(null), // each slot's MTP K/V cache
    pos: [q.MAX_SLOTS]u32 = @splat(0), // its newest kept row's position
    last: *anyopaque, // fp16 [MAX_SLOTS, D]: each slot's newest absorbed row's final normed state
    chain: [q.MAX_SLOTS][ROWS]u32 = undefined, // its held drafts for the next window
    chain_n: [q.MAX_SLOTS]u32 = @splat(0),
    step_ms: f64 = 0.6, // one chained draft step, timed at open
    quiet: bool = false, // no records (the timed step at open)
    words: [256]u32 = undefined,
    mt_host: [ROWS * 8]u64 = @splat(0),
    pos_host: [ROWS]i32 = @splat(0),
    ids_host: [SLICE]i32 = @splat(0),

    /// Bytes one slot's MTP K/V cache takes on this card: two halves (K and V), each the values of its K/V heads at
    /// every position and then the int8 scales (int8: q.kv_i8, as the live pair runs).
    pub fn slotBytes(m: *const Mtp) u64 {
        _ = m;
        return 2 * halfBytes(layout());
    }

    fn kvHalf(m: *const Mtp) usize {
        _ = m;
        return halfBytes(layout());
    }

    /// Copies `slot`'s MTP cache at `at` positions into `buf` (snapRegions' order), then its newest row's state.
    pub fn saveSnap(m: *Mtp, slot: u32, at: u32, buf: *anyopaque) !void {
        if (slot >= q.MAX_SLOTS) return error.NoSuchSlot;
        const kv = m.kv[slot] orelse return error.NoSlot;
        const d = &m.r.d;
        for (snapRegions(layout(), at)) |r| if (r.len > 0) try copy(d, off(buf, r.dst), off(kv, r.src), r.len);
        try copy(d, off(buf, @intCast(snapBytes(at) - D * 2)), off(m.last, @as(usize, slot) * D * 2), D * 2);
    }

    /// A kept state's MTP part back into `slot`'s cache (allocated when the slot has none), then its newest row's
    /// state. The slot's record (position, held drafts) goes back with applyRecord.
    pub fn restoreSnap(m: *Mtp, slot: u32, at: u32, buf: *anyopaque) !void {
        if (slot >= q.MAX_SLOTS) return error.NoSuchSlot;
        if (m.kv[slot] == null) try m.open(slot);
        const kv = m.kv[slot].?;
        const d = &m.r.d;
        for (snapRegions(layout(), at)) |r| if (r.len > 0) try copy(d, off(kv, r.src), off(buf, r.dst), r.len);
        try copy(d, off(m.last, @as(usize, slot) * D * 2), off(buf, @intCast(snapBytes(at) - D * 2)), D * 2);
    }

    /// The host record a kept state keeps beside its MTP part: where the cache stood and the drafts it held.
    pub fn snapRecord(m: *const Mtp, slot: u32) SnapRecord {
        return .{ .pos = m.pos[slot], .chain_n = m.chain_n[slot], .chain = m.chain[slot] };
    }

    pub fn applyRecord(m: *Mtp, slot: u32, rec: SnapRecord) void {
        if (slot >= q.MAX_SLOTS) return;
        m.pos[slot] = rec.pos;
        m.chain_n[slot] = rec.chain_n;
        m.chain[slot] = rec.chain;
    }

    /// The MTP head of the runner's checkpoint onto the device: weights, plans and buffers. No all-reduce (a follower
    /// rank calls this too, before its own pool is sized); `measure` times a step once the pool exists.
    pub fn load(r: *q.Runner) !*Mtp {
        readEnv();
        const d = &r.d;
        const w = r.model.exl3 orelse return error.NotExl3;
        const c = &r.ckpt;
        const m = try alloc.create(Mtp);
        errdefer alloc.destroy(m);
        m.* = .{ .r = r, .w = w, .qkv = undefined, .o = undefined, .gu = undefined, .down = undefined, .head = undefined, .plans = undefined, .scratch = .{ undefined, undefined, undefined, undefined }, .fc_e = undefined, .fc_h = undefined, .fc_n = 0, .norm_e = undefined, .norm_h = undefined, .in_norm = undefined, .post_norm = undefined, .q_norm = undefined, .k_norm = undefined, .norm = undefined, .x = undefined, .en = undefined, .hn = undefined, .tgt = undefined, .h = undefined, .pr = undefined, .r2 = undefined, .p = undefined, .qb = undefined, .y = undefined, .guv = undefined, .act = undefined, .ahi = undefined, .alo = undefined, .logits = undefined, .hb = undefined, .ids = undefined, .mt = undefined, .mpos = undefined, .zero = undefined, .last = undefined };
        // the checkpoint's own MTP tensors: 39 of them, the dense full-attention layer and the fc
        const R = q.RANK;
        const tp1 = q.TP == 1;
        const qw = HQ * 512;
        const kw = AKV * 256;
        const p = "mtp.layers.0.";
        m.qkv = try w.add(d, c, &.{
            .{ .prefix = "mtp.layers.0.self_attn.q_proj", .cut = if (tp1) .{} else ex.Cut.cols(R * qw, qw, D) },
            .{ .prefix = "mtp.layers.0.self_attn.k_proj", .cut = if (tp1) .{} else ex.Cut.cols(R * kw, kw, D) },
            .{ .prefix = "mtp.layers.0.self_attn.v_proj", .cut = if (tp1) .{} else ex.Cut.cols(R * kw, kw, D) },
        }, ATT_IN);
        m.o = try w.add(d, c, &.{.{ .prefix = "mtp.layers.0.self_attn.o_proj", .cut = if (tp1) .{} else ex.Cut.rows(R * OUT, OUT, D) }}, D);
        m.gu = try w.add(d, c, &.{
            .{ .prefix = "mtp.layers.0.mlp.gate_proj", .cut = if (tp1) .{} else ex.Cut.cols(R * FFR, FFR, D) },
            .{ .prefix = "mtp.layers.0.mlp.up_proj", .cut = if (tp1) .{} else ex.Cut.cols(R * FFR, FFR, D) },
        }, 2 * FFR);
        m.down = try w.add(d, c, &.{.{ .prefix = "mtp.layers.0.mlp.down_proj", .cut = if (tp1) .{} else ex.Cut.rows(R * FFR, FFR, D) }}, D);
        // the shared lm_head, read from the MTP's rows: a copy of the head's projection, its own plans
        const hp = try alloc.create(ex.Proj);
        hp.* = w.head.*;
        hp.head = true;
        m.head = hp;
        m.norm_e = try ex.normTensor(d, c, "mtp.pre_fc_norm_embedding.weight", true);
        m.norm_h = try ex.normTensor(d, c, "mtp.pre_fc_norm_hidden.weight", true);
        m.in_norm = try ex.normTensor(d, c, p ++ "input_layernorm.weight", true);
        m.post_norm = try ex.normTensor(d, c, p ++ "post_attention_layernorm.weight", true);
        m.q_norm = try ex.normTensor(d, c, p ++ "self_attn.q_norm.weight", true);
        m.k_norm = try ex.normTensor(d, c, p ++ "self_attn.k_norm.weight", true);
        m.norm = try ex.normTensor(d, c, "mtp.norm.weight", true);
        // the fc: 5 bits, decoded on the host, the rank's columns as fp16 dense rows
        const fc = try fcRows(d, c);
        m.fc_e = fc.e;
        m.fc_h = fc.h;
        m.fc_n = fc.n;
        try m.bind();
        try m.open(0); // counted in the load: the pool's sizing sets its bytes aside for every slot
        // the prompt absorb, on every rank that runs the prompt (the follower replays it through the same hook)
        r.on_prompt = .{ .ctx = m, .f = promptHook };
        r.mtp = m;
        return m;
    }

    /// The drafts a round holds at most, and one chained step's ms (the lane core's depth rule prices them).
    pub fn mostDrafts(m: *const Mtp) u32 {
        _ = m;
        return most;
    }
    pub fn stepMs(m: *const Mtp) f64 {
        return m.step_ms;
    }

    fn bind(m: *Mtp) !void {
        const d = &m.r.d;
        const rows = q.maxRows();
        _ = rows;
        const sl = @as(usize, SLICE);
        m.x = try d.alloc_(sl * D * 4);
        m.en = try d.alloc_(sl * D * 2);
        m.hn = try d.alloc_(sl * D * 2);
        m.tgt = try d.alloc_(sl * D * 2);
        m.h = try d.alloc_(sl * D * 2);
        m.pr = try d.alloc_(sl * D * 2);
        m.r2 = try d.alloc_(sl * D * 2);
        m.p = try d.alloc_(sl * ATT_IN * 2);
        m.qb = try d.alloc_(sl * HQ * 256 * 2);
        m.y = try d.alloc_(sl * OUT * 2);
        m.guv = try d.alloc_(sl * 2 * FFR * 2);
        m.act = try d.alloc_(sl * FFR * 2);
        // rowsums writes [row, K / gs] with K the kernel's width: D for add_rmsnorm, FF_R for swiglu, OUT for
        // attn_gate. The largest is D, so the buffer is sized from D, not FF_R (a 128-row prompt overran m.alo).
        const ahi_rows = sl * (@max(D, @max(FFR, OUT)) / q.GS + 2) * 2;
        m.ahi = try d.alloc_(ahi_rows);
        m.alo = try d.alloc_(ahi_rows);
        m.logits = try d.alloc_(q.windowRows() * q.VOCAB * 2);
        m.hb = try d.alloc_(q.windowRows() * D * 2);
        m.ids = try d.alloc_(sl * 4);
        m.mt = try d.alloc_(q.windowRows() * 8 * 8);
        m.mpos = try d.alloc_(q.windowRows() * 4);
        m.zero = try d.alloc_(sl * D * 4);
        try d.fill(m.zero, @as(f32, 0.0), sl * D * 4);
        m.last = try d.alloc_(q.MAX_SLOTS * D * 2);
        try d.fill(m.last, @as(u16, 0), q.MAX_SLOTS * D * 2);
        // the plans' scratch: the largest need of a projection's part, decode rows and prompt rows apart
        var need: [4]usize = .{ 0, 0, 0, 0 };
        inline for (.{ m.qkv, m.o, m.gu, m.down }) |pr| {
            for (pr.parts[0..pr.count], 0..) |pt, j| {
                need[j] = @max(need[j], exl3.scratchBytes(pt.m.k, pt.m.n, q.windowRows()));
                need[j] = @max(need[j], exl3.promptScratchBytes(pt.m.k, pt.m.n, SLICE));
            }
        }
        for (&m.scratch, need) |*sc, n| sc.* = try d.alloc_(@max(n, 64));
        const io = layer_io;
        try m.mkPlans(&m.plans.qkv, m.qkv, m.h, m.p, io.out, io.in);
        try m.mkPlans(&m.plans.o, m.o, m.h, m.pr, io.out, io.in);
        try m.mkPlans(&m.plans.gu, m.gu, m.h, m.guv, io.out, io.in);
        try m.mkPlans(&m.plans.down, m.down, m.act, m.pr, io.out, io.in);
        // the head over the MTP's rows (the same lm_head matrix the target's head uses)
        m.head.x = @intFromPtr(m.hb);
        m.head.y = @intFromPtr(m.logits);
        const wr: usize = q.windowRows();
        m.head.dec_n = @intCast(wr);
        for (m.head.dec[0..wr], 0..) |*pl, t| pl.* = try m.head.plan(m.w, m.head.x, m.head.y, @intCast(t + 1));
        try d.sync();
    }

    fn mkPlans(m: *Mtp, out: *Plans, pr: *ex.Proj, x: *anyopaque, y: *anyopaque, ot: exl3.Output, it: exl3.Input) !void {
        const wr: usize = q.windowRows();
        out.dec_n = @intCast(wr);
        for (out.dec[0..wr], 0..) |*pl, t| pl.* = try m.mk(pr, x, y, @intCast(t + 1), ot, it);
        out.chunk = try m.mk(pr, x, y, SLICE, ot, it);
    }

    fn mk(m: *Mtp, pr: *ex.Proj, x: *anyopaque, y: *anyopaque, rows: u32, ot: exl3.Output, it: exl3.Input) !exl3.Plan {
        var gs: [4]exl3.Group = undefined;
        const list = pr.groups(m.w, @intFromPtr(x), @intFromPtr(y), rows, &gs);
        for (list, 0..) |*g, j| {
            g.scratch = @intFromPtr(m.scratch[j]);
            g.output = ot;
            g.input = it;
        }
        return exl3.Plan.init(&m.w.ctx, m.w.stream(), alloc, list);
    }

/// The fc's two input halves (the embedding's inputs, then the hidden state's) as this rank's fp16 dense rows [N, D].
/// The fc is the checkpoint's one 5-bit matrix: the reference decoder (core.exl3_format) turns it into W[K, N] on the
/// host, and the rank's columns are transposed to the dense_rows layout (row = output column, K = inputs).
fn fcRows(d: *Dev, c: *const dev.Checkpoint) !struct { e: *anyopaque, h: *anyopaque, n: u32 } {
    const tr = try c.get("mtp.fc.trellis");
    const suh_t = try c.get("mtp.fc.suh");
    const svh_t = try c.get("mtp.fc.svh");
    // the fc is decoded with the mul1 codebook below; another codebook would decode every state wrong
    const mul1_t = try c.get("mtp.fc.mul1");
    if (mul1_t.bytes.len != 4 or std.mem.readInt(i32, mul1_t.bytes[0..4], .little) == 0) return error.UnsupportedExl3Codebook;
    if (!std.mem.eql(u8, tr.dtype, "I16") or tr.rank != 3 or !std.mem.eql(u8, suh_t.dtype, "F16") or !std.mem.eql(u8, svh_t.dtype, "F16")) return error.UnexpectedShape;
    const tiles_k: usize = @intCast(tr.shape[0]);
    const tiles_n: usize = @intCast(tr.shape[1]);
    const words: usize = @intCast(tr.shape[2]);
    const width: exf.Width = .{ .halves = @intCast(words / 8) };
    if (width.tileWords() != words) return error.UnexpectedShape;
    const k = tiles_k * 16; // 10 240: the fc's inputs
    const n = tiles_n * 16; // 5 120: its output columns
    if (k != 2 * D or n != D) return error.UnexpectedShape;
    // the checkpoint's bytes are read through an unaligned slice: copy them into aligned ones the decoder takes
    const suh = try alloc.alloc(f16, k);
    defer alloc.free(suh);
    const svh = try alloc.alloc(f16, n);
    defer alloc.free(svh);
    const trellis = try alloc.alloc(i16, tiles_k * tiles_n * words);
    defer alloc.free(trellis);
    @memcpy(std.mem.sliceAsBytes(suh), suh_t.bytes);
    @memcpy(std.mem.sliceAsBytes(svh), svh_t.bytes);
    @memcpy(std.mem.sliceAsBytes(trellis), tr.bytes);
    // both halves whole on every rank: the fc's output is the layer's input hidden state, which every rank needs in
    // full for the input norm and the qkv projection (the target's in-projections split their output columns instead,
    // which works there because each rank keeps its own heads)
    const n0: usize = 0;
    const nw: usize = n;
    const half = k / 2; // each half is D inputs
    const tiles_half = tiles_k / 2;
    const out_e = try alloc.alloc(f16, nw * D);
    defer alloc.free(out_e);
    const out_h = try alloc.alloc(f16, nw * D);
    defer alloc.free(out_h);
    const wq = try alloc.alloc(f64, half * n);
    defer alloc.free(wq);
    const scratch = try alloc.alloc(f64, 2 * half * n);
    defer alloc.free(scratch);
    for (0..2) |hi| {
        const at = hi * tiles_half * tiles_n * words;
        const part = trellis[at..][0 .. tiles_half * tiles_n * words];
        exf.dequantize(width, .mul1, part, tiles_half, tiles_n, suh[hi * half ..][0..half], svh, wq, scratch);
        const out = if (hi == 0) out_e else out_h;
        for (0..nw) |j| {
            for (0..half) |kk| out[j * D + kk] = @floatCast(wq[kk * n + n0 + j]);
        }
    }
    return .{ .e = try d.upload(std.mem.sliceAsBytes(out_e)), .h = try d.upload(std.mem.sliceAsBytes(out_h)), .n = @intCast(nw) };
}

/// The card's memory goes with the runner's device (Dev.deinit closes it); the plans' tables go with it.
pub fn deinit(m: *Mtp) void {
    alloc.destroy(m);
}

/// A slot's MTP K/V cache: allocated on its first use (the pool's sizing reserved its bytes), zeroed.
pub fn open(m: *Mtp, slot: u32) !void {
    if (slot >= q.MAX_SLOTS) return error.NoSuchSlot;
    if (m.kv[slot] == null) {
        const kv = try m.r.d.alloc_(m.slotBytes());
        try m.r.d.fill(kv, @as(u32, 0), @intCast(m.slotBytes()));
        m.kv[slot] = kv;
    }
    m.pos[slot] = 0;
    m.chain_n[slot] = 0;
}

test "the MTP snapshot's regions tile its cache and its record round trips" {
    const l: Layout = .{ .heads = 2, .cap = 8, .int8 = true };
    const at: usize = 5;
    const rowv = l.heads * 256;
    const half = halfBytes(l);
    try std.testing.expectEqual(@as(usize, 2 * 8 * 256 + 2 * 8 * 4), half);
    // a cache filled as store_kv8 writes it: values at pos * Q27_KVROW + head * 256, the scale at cap * Q27_KVROW +
    // (pos * heads + head) * 4
    const cache = try std.testing.allocator.alloc(u8, 2 * half);
    defer std.testing.allocator.free(cache);
    @memset(cache, 0);
    for (0..2) |v| for (0..at) |pos| for (0..l.heads) |h| {
        for (0..256) |i| {
            cache[v * half + pos * rowv + h * 256 + i] = @truncate(pos * 7 + h * 3 + i + v * 11);
        }
        std.mem.writeInt(u32, cache[v * half + l.cap * rowv + (pos * l.heads + h) * 4 ..][0..4], @intCast(0x1000 * v + pos * 100 + h), .little);
    };
    const bytes = 2 * at * rowv + 2 * at * l.heads * 4; // the cache part; the newest row's state follows it
    const out = try std.testing.allocator.alloc(u8, bytes);
    defer std.testing.allocator.free(out);
    @memset(out, 0);
    var covered: usize = 0;
    for (snapRegions(l, at)) |r| {
        try std.testing.expect(r.len > 0);
        try std.testing.expect(r.dst >= covered and r.dst + r.len <= bytes); // no two regions overlap
        covered = r.dst + r.len;
        @memcpy(out[r.dst .. r.dst + r.len], cache[r.src .. r.src + r.len]);
    }
    try std.testing.expectEqual(bytes, covered);
    for (0..2) |v| for (0..at) |pos| for (0..l.heads) |h| {
        const o = if (v == 0) 0 else at * rowv + at * l.heads * 4;
        try std.testing.expectEqualSlices(u8, cache[v * half + pos * rowv + h * 256 ..][0..256], out[o + pos * rowv + h * 256 ..][0..256]);
        const so = o + at * rowv;
        try std.testing.expectEqualSlices(u8, cache[v * half + l.cap * rowv + (pos * l.heads + h) * 4 ..][0..4], out[so + (pos * l.heads + h) * 4 ..][0..4]);
    };
    // the record
    var m: Mtp = undefined;
    m.pos = @splat(0);
    m.chain_n = @splat(0);
    for (&m.chain) |*c| for (c) |*x| {
        x.* = 0;
    };
    var rec: SnapRecord = .{ .pos = 39, .chain_n = 3, .chain = @splat(0) };
    rec.chain[0] = 11;
    rec.chain[1] = 12;
    rec.chain[2] = 13;
    m.applyRecord(2, rec);
    const back = m.snapRecord(2);
    try std.testing.expectEqual(rec.pos, back.pos);
    try std.testing.expectEqual(rec.chain_n, back.chain_n);
    try std.testing.expectEqualSlices(u32, &rec.chain, &back.chain);
    try std.testing.expectEqual(@as(u32, 0), m.pos[0]); // another slot's record is untouched
}

pub fn release(m: *Mtp, slot: u32) void {
    if (slot >= q.MAX_SLOTS) return;
    m.pos[slot] = 0;
    m.chain_n[slot] = 0;
}

/// One draft step's cost at 1 row, timed at open on both ranks (the depth rule prices a draft with it).
pub fn measure(m: *Mtp) !void {
    try m.open(0);
    m.quiet = true;
    defer m.quiet = false;
    m.pos[0] = 0;
    m.chain_n[0] = 0;
    const asks = [_]Ask{.{ .slot = 0, .pending = 100, .depth = 2 }};
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const t0 = q.now(); // seconds (qwen27.now)
        m.draft(&asks) catch return;
        try m.r.d.sync();
        best = @min(best, q.now() - t0);
    }
    // one draft step in ms: the ask holds two drafts (one layer pass and two head calls), as cuda_mtp.zig's load prices
    // a chain by its depth. q.now is seconds, so the price is 1e3 * seconds / drafts.
    m.step_ms = best * 1e3 / 2.0;
    m.chain_n[0] = 0;
    m.pos[0] = 0;
    std.debug.print("MTP head: one draft step {d:.3} ms, {d} drafts a round at most, {d:.2} GB of K/V a card at {d} slots\n", .{ m.step_ms, most, @as(f64, @floatFromInt(m.slotBytes())) * @as(f64, q.MAX_SLOTS) / 1e9, q.MAX_SLOTS });
}

/// One MTP layer pass over `n` rows: the layer input is `tgt` (the hidden state rows, fp16) and `ids` (the token after
/// each row). The K/V rows go to each row's cache at `p0` (wide, one slot) or at MT's POS[t] (mt, any slots). The
/// output rows land in `h` (the final normed state) and `x` (the residual).
/// XPU_MTP_SYNC: drain after this launch so a fault names it.
fn afterLaunch(m: *Mtp) !void {
    if (sync_each) try m.r.d.sync();
}

fn layer(m: *Mtp, slot: u32, n: u32, p0: u32, wide: bool, mt: bool) !void {
    const d = &m.r.d;
    const t32: i32 = @intCast(n);
    const gs: i32 = @intCast(q.GS);
    const ms: i32 = @intCast(q.maxRows() + 32);
    const rot: i32 = q.rot_on;
    const cap: i32 = fwd.kvCap();
    const scale: f32 = 1.0 / 16.0;
    const nul: ?*anyopaque = null;
    if (!wide and !mt) return error.MtpNeedsMt;
    if (wide and m.kv[slot] == null) return error.NoSlot;
    const nrm = if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec;
    // the embedding of the follow tokens, its norm; then the hidden state's norm (R = tgt, X = 0: one fp32 copy)
    try d.launch(d.k.embed_dense, .{ .x = n }, .{ m.ids, m.r.model.embed_w, m.x, @as(i32, D) });
    try afterLaunch(m);
    try d.launch(nrm, .{ .x = n }, .{ m.x, m.pr, m.norm_e, m.en, m.ahi, m.alo, @as(i32, D), EPS, @as(i32, 0), gs, nul, nul, ms, rot });
    try copy(d, m.x, m.zero, @as(usize, n) * D * 4);
    try d.launch(nrm, .{ .x = n }, .{ m.x, m.tgt, m.norm_h, m.hn, m.ahi, m.alo, @as(i32, D), EPS, @as(i32, 1), gs, nul, nul, ms, rot });
    // fc: the embedding's inputs through the first half, the hidden state's through the second, summed
    try d.launch(d.k.dense_rows, .{ .x = (n + 15) / 16, .y = m.fc_n / 16 }, .{ m.en, m.fc_e, m.pr, t32, @as(i32, D), @as(i32, 0) });
    try d.launch(d.k.dense_rows, .{ .x = (n + 15) / 16, .y = m.fc_n / 16 }, .{ m.hn, m.fc_h, m.r2, t32, @as(i32, D), @as(i32, 0) });
    try d.launch(d.k.tp_add, .{ .x = (n * D + 255) / 256 }, .{ m.pr, m.pr, m.r2, @as(i32, @intCast(n * D)) });
    try afterLaunch(m);
    try copy(d, m.x, m.zero, @as(usize, n) * D * 4);
    try d.launch(nrm, .{ .x = n }, .{ m.x, m.pr, m.in_norm, m.h, m.ahi, m.alo, @as(i32, D), EPS, @as(i32, 1), gs, nul, nul, ms, rot });
    try m.runPlans(&m.plans.qkv, n, wide);
    try afterLaunch(m);
    // q, k, v with their norms and rope; K and V to each row's cache row
    if (mt)
        try d.launch(d.k.attn_prep, .{ .x = n }, .{ m.p, m.q_norm, m.k_norm, m.r.model.invf, m.qb, nul, nul, @as(i32, 0), EPS, nul, m.mt, m.mpos, @as(i32, q.kv8), cap, nul, nul })
    else
        try d.launch(d.k.attn_prep, .{ .x = n }, .{ m.p, m.q_norm, m.k_norm, m.r.model.invf, m.qb, m.kv[slot].?, off(m.kv[slot].?, m.kvHalf()), @as(i32, @intCast(p0)), EPS, nul, nul, nul, @as(i32, q.kv8), cap, nul, nul });
    // attention: the prompt kernels over a slice of one slot, else the decode kernel over each row's own window
    if (wide) {
        const kv = m.kv[slot].?;
        const vc = off(kv, m.kvHalf());
        if (q.kv_i8)
            try d.launch(d.k.attn_prompt8, .{ .x = (n + 7) / 8, .y = AKV }, .{ m.qb, kv, vc, m.y, @as(i32, @intCast(p0)), t32, scale, cap, nul })
        else
            try d.launch(d.k.attn_prompt, .{ .x = (n + 7) / 8, .y = AKV }, .{ m.qb, kv, vc, m.y, @as(i32, @intCast(p0)), t32, scale, nul });
    } else {
        var far: i32 = 0;
        for (m.pos_host[0..n]) |v| far = @max(far, v);
        const ns: u32 = @intCast((@as(u32, @intCast(far)) + q.SPLIT) / q.SPLIT);
        try d.launch(d.k.attn_decode, .{ .x = ns, .y = AKV, .z = n }, .{ m.qb, nul, nul, m.r.acts.pm, m.r.acts.pl, m.r.acts.po, @as(i32, 0), @as(i32, @intCast(ns)), scale, nul, m.mt, m.mpos, cap, nul });
        try afterLaunch(m);
        try d.launch(d.k.attn_merge, .{ .x = HQ, .y = n }, .{ m.r.acts.pm, m.r.acts.pl, m.r.acts.po, m.y, @as(i32, 0), @as(i32, @intCast(ns)), nul, m.mpos });
    }
    // XPU_KV8 & 1 rotates q, k, v by Hadamard in attn_prep, so the merged output is un-rotated before the gate, as the
    // target's forward does. Off with XPU_KV8 unset (the live pair's setting), where the launch is skipped.
    if (q.kv8 & 1 != 0) try d.launch(d.k.rot_heads, .{ .x = n, .y = HQ }, .{m.y});
    try d.launch(d.k.attn_gate, .{ .x = n }, .{ m.y, m.p, m.h, m.ahi, m.alo, gs, rot, nul, nul, ms });
    try afterLaunch(m);
    try m.runPlans(&m.plans.o, n, wide);
    try afterLaunch(m);
    if (tp.group) |*g| try g.allReduce(d, m.pr, n);
    try d.launch(nrm, .{ .x = n }, .{ m.x, m.pr, m.post_norm, m.h, m.ahi, m.alo, @as(i32, D), EPS, @as(i32, 1), gs, nul, nul, ms, rot });
    try m.runPlans(&m.plans.gu, n, wide);
    try afterLaunch(m);
    try d.launch(d.k.swiglu, .{ .x = n, .y = (FFR + 1023) / 1024 }, .{ m.guv, m.act, m.ahi, m.alo, gs, @as(i32, 0), rot, @as(i32, FFR) });
    try afterLaunch(m);
    try m.runPlans(&m.plans.down, n, wide);
    try afterLaunch(m);
    if (tp.group) |*g| try g.allReduce(d, m.pr, n);
    try d.launch(nrm, .{ .x = n }, .{ m.x, m.pr, m.norm, m.h, m.ahi, m.alo, @as(i32, D), EPS, @as(i32, 1), gs, nul, nul, ms, rot });
}

fn runPlans(m: *Mtp, ps: *Plans, n: u32, wide: bool) !void {
    if (wide) return exl3.encodeRows(&m.w.p, m.w.stream(), &ps.chunk.?, n);
    if (n == 0 or n > ps.dec_n) return error.BadRows;
    return exl3.encode(&m.w.p, m.w.stream(), &ps.dec[n - 1]);
}

/// The head over the rows `m.hb` already holds (n of them): the logits and their argmax, into r.acts.tok.
fn headRows(m: *Mtp, n: u32) !void {
    const d = &m.r.d;
    try ex.project(m.head, m.w, d, m.hb, m.logits, n, false);
    try d.launch(d.k.argmax_part, .{ .x = 256, .y = n }, .{ m.logits, @as(i32, q.VOCAB), m.r.acts.pv, m.r.acts.pi });
    try d.launch(d.k.argmax, .{ .x = n }, .{ m.r.acts.pv, m.r.acts.pi, m.r.acts.tok });
}

/// n tokens out of r.acts.tok, read after a wait (the draft's chain needs them on the host).
fn readTokens(m: *Mtp, out: []u32) !void {
    var host: [ROWS]i32 = undefined;
    try copy(&m.r.d, @ptrCast(&host), m.r.acts.tok, out.len * 4);
    try m.r.d.sync();
    for (out, 0..) |*o, i| o.* = @intCast(host[i]);
}

/// Gather h's rows into hb (the head's rows), one copy a row.
fn gather(m: *Mtp, rows: []const u32) !void {
    for (rows, 0..) |r, i| try copy(&m.r.d, off(m.hb, i * D * 2), off(m.h, @as(usize, r) * D * 2), D * 2);
}

/// A prompt chunk's absorb: every row's K/V into `slot`'s MTP cache, the newest row's state kept for the first draft.
/// `ids` is the chunk's tokens from `p0`, and `last_token` is the token after the chunk's last row (the pending one).
pub fn absorbPrompt(m: *Mtp, slot: u32, ids: []const u32, p0: u32, last_token: u32) !void {
    if (ids.len == 0) return;
    if (m.kv[slot] == null) try m.open(slot);
    var at: usize = 0;
    var last_row: usize = 0;
    while (at < ids.len) {
        const n: usize = @min(SLICE, ids.len - at);
        for (0..n) |i| {
            const j = at + i + 1;
            m.ids_host[i] = @intCast(if (j < ids.len) ids[j] else last_token);
        }
        try m.copyInIds(n);
        try copy(&m.r.d, m.tgt, off(m.r.acts.h, at * D * 2), n * D * 2);
        try m.layer(slot, @intCast(n), p0 + @as(u32, @intCast(at)), true, false);
        last_row = n - 1;
        at += n;
    }
    try copy(&m.r.d, off(m.last, @as(usize, slot) * D * 2), off(m.h, last_row * D * 2), D * 2);
    m.pos[slot] = p0 + @as(u32, @intCast(ids.len)) - 1;
}

/// A verify's kept rows: the MTP layer over the target's final normed states of those rows, their K/V to each stream's
/// cache at the kept positions, the newest row's state kept for the next window's first draft.
pub fn absorbKept(m: *Mtp, items: []const Kept) !void {
    var total: usize = 0;
    const cap: u32 = @intCast(layout().cap);
    for (items) |it| {
        if (it.block.len != it.follow.len or it.block.len == 0) return error.BadKept;
        // a slot that has not run a prompt absorb has no cache yet: open it here (the pool sizing reserved its bytes)
        if (m.kv[it.slot] == null) try m.open(it.slot);
        // every row's cache position and its target row must fit the cache and the target's row buffer: a row past
        // either writes past a card buffer and the next drain returns DEVICE_LOST instead of a named error
        if (it.pos + @as(u32, @intCast(it.block.len)) > cap) return error.MtpPosOverCap;
        for (it.block) |r| if (r >= q.maxRows()) return error.MtpRowOverMax;
        // embed_dense reads E[id * D] with no bound of its own, and argmax writes 0x7fffffff for a row whose logits
        // are all NaN or -infinity. A token at or past the vocabulary is a device read past the table: name it.
        for (it.follow) |t| if (t >= q.VOCAB) return error.MtpBadToken;
        total += it.block.len;
    }
    if (total > @as(usize, q.windowRows())) return error.BadKept;
    if (!m.quiet) if (tp.group) |*g| try g.record(.mtp_absorb, @intCast(items.len), 0, 0, m.keptWords(items));
    var row: usize = 0;
    var ends: [ROWS]struct { slot: u32, row: usize } = undefined;
    for (items, 0..) |it, i| {
        const kv = m.kv[it.slot].?;
        for (0..it.block.len) |j| {
            try copy(&m.r.d, off(m.tgt, row * D * 2), off(m.r.acts.h, @as(usize, it.block[j]) * D * 2), D * 2);
            m.ids_host[row] = @intCast(it.follow[j]);
            m.mt_host[row * 8 + 0] = @intFromPtr(kv);
            m.mt_host[row * 8 + 1] = @intFromPtr(off(kv, m.kvHalf()));
            m.mt_host[row * 8 + 2] = 0; // no page table: the head's cache is its own allocation
            m.mt_host[row * 8 + 6] = 0;
            m.mt_host[row * 8 + 7] = 1;
            m.pos_host[row] = @intCast(it.pos + @as(u32, @intCast(j)));
            row += 1;
        }
        ends[i] = .{ .slot = it.slot, .row = row - 1 };
    }
    try m.copyInIds(total);
    try m.copyInMt(total);
    // drain the two host copies before the layer: a fault in a copy from host memory then names this step, not a kernel
    try m.r.d.sync();
    try m.layer(0, @intCast(total), 0, false, true);
    for (ends[0..items.len]) |e| try copy(&m.r.d, off(m.last, @as(usize, e.slot) * D * 2), off(m.h, e.row * D * 2), D * 2);
    // each stream's newest kept row's position, and its chain reset for the next window
    for (items) |it| {
        m.pos[it.slot] = it.pos + @as(u32, @intCast(it.block.len)) - 1;
        m.chain_n[it.slot] = 0;
    }
}

/// The next window's drafts for each stream: the first draft from the newest absorbed row's head, then each later draft
/// from the previous row's output and the draft before it (cuda_mtp.zig's hold), at positions newest + step - 1.
pub fn draft(m: *Mtp, asks: []const Ask) !void {
    // the head's row buffers (hb, logits, mt, mpos, ids, the decode partials) are sized for windowRows, not for ROWS:
    // a window holds windowRows rows in all, so more streams than that is a bad request, not a card fault
    if (asks.len == 0 or asks.len > @as(usize, q.windowRows())) return error.BadDrafts;
    const cap: u32 = @intCast(layout().cap);
    for (asks) |a| {
        if (m.kv[a.slot] == null) return error.NoSlot;
        if (a.pending >= q.VOCAB) return error.MtpBadToken;
        const d = @min(@max(a.depth, 1), most);
        if (draftPos(m.pos[a.slot], d) >= cap) return error.MtpPosOverCap;
    }
    if (!m.quiet) if (tp.group) |*g| try g.record(.mtp_draft, @intCast(asks.len), 0, 0, m.askWords(asks));
    var n: usize = 0;
    for (asks) |a| {
        try copy(&m.r.d, off(m.hb, n * D * 2), off(m.last, @as(usize, a.slot) * D * 2), D * 2);
        n += 1;
    }
    try m.headRows(@intCast(n));
    var tok: [ROWS]u32 = undefined;
    try m.readTokens(tok[0..n]);
    var depth: [ROWS]u32 = undefined;
    var prev: [ROWS]u32 = @splat(0xffffffff);
    for (asks, 0..) |a, i| {
        depth[i] = @min(@max(a.depth, 1), most);
        m.chain[a.slot][0] = tok[i];
        m.chain_n[a.slot] = 1;
    }
    var step: u32 = 1;
    while (true) : (step += 1) {
        var cnt: usize = 0;
        var which: [ROWS]usize = undefined;
        for (asks, 0..) |a, i| {
            if (depth[i] <= step) continue;
            const dst = off(m.tgt, cnt * D * 2);
            if (prev[i] == 0xffffffff)
                try copy(&m.r.d, dst, off(m.last, @as(usize, a.slot) * D * 2), D * 2)
            else
                try copy(&m.r.d, dst, off(m.h, @as(usize, prev[i]) * D * 2), D * 2);
            const prev_tok = m.chain[a.slot][step - 1];
            if (prev_tok >= q.VOCAB) return error.MtpBadToken;
            m.ids_host[cnt] = @intCast(prev_tok);
            m.mt_host[cnt * 8 + 0] = @intFromPtr(m.kv[a.slot].?);
            m.mt_host[cnt * 8 + 1] = @intFromPtr(off(m.kv[a.slot].?, m.kvHalf()));
            m.mt_host[cnt * 8 + 2] = 0;
            m.mt_host[cnt * 8 + 6] = 0;
            m.mt_host[cnt * 8 + 7] = 1;
            m.pos_host[cnt] = @intCast(draftPos(m.pos[a.slot], step));
            which[cnt] = i;
            prev[i] = @intCast(cnt);
            cnt += 1;
        }
        if (cnt == 0) break;
        try m.copyInIds(cnt);
        try m.copyInMt(cnt);
        try m.layer(0, @intCast(cnt), 0, false, true);
        var rows: [ROWS]u32 = undefined;
        for (0..cnt) |i| rows[i] = @intCast(i);
        try m.gather(rows[0..cnt]);
        try m.headRows(@intCast(cnt));
        try m.readTokens(tok[0..cnt]);
        for (0..cnt) |i| {
            m.chain[asks[which[i]].slot][step] = tok[i];
            m.chain_n[asks[which[i]].slot] = step + 1;
        }
    }
}

/// The drafts a slot holds for the next window, parents -1, 0, 1, … (a chain).
pub fn held(m: *Mtp, slot: u32, out: []u32) []const u32 {
    const n = @min(m.chain_n[slot], @as(u32, @intCast(out.len)));
    @memcpy(out[0..n], m.chain[slot][0..n]);
    return out[0..n];
}

/// One lone stream's round: draft, verify through the target's window, keep the accepted path, absorb its rows for the
/// next round. The tokens are the drafts that landed, then the target's own token (the next pending one).
pub fn round(m: *Mtp, slot: u32, pending: u32, depth: u32, out: *[ROWS]u32) !Round {
    var asks = [_]Ask{.{ .slot = slot, .pending = pending, .depth = depth }};
    try m.draft(&asks);
    const k = m.chain_n[slot];
    if (k == 0 or k + 1 > ROWS) return error.BadDrafts;
    var ids: [ROWS]u32 = undefined;
    ids[0] = pending;
    @memcpy(ids[1..][0..k], m.chain[slot][0..k]);
    const pos = m.r.seqs[slot].pos;
    var wins = [_]q.Runner.Win{.{ .slot = slot, .ids = ids[0 .. k + 1] }};
    try m.r.verifyMulti(&wins);
    var acc: u32 = 0;
    while (acc < k and m.r.block[acc] == m.chain[slot][acc]) : (acc += 1) {}
    const na: usize = acc;
    var path: [ROWS]u32 = undefined;
    var block: [ROWS]u32 = undefined;
    var follow: [ROWS]u32 = undefined;
    for (0..na + 1) |i| {
        path[i] = @intCast(i);
        block[i] = @intCast(i);
        follow[i] = if (i < na) m.chain[slot][i] else m.r.block[acc];
    }
    const kept = [_]Kept{.{ .slot = slot, .pos = pos, .block = block[0 .. na + 1], .follow = follow[0 .. na + 1] }};
    try m.absorbKept(&kept);
    var paths = [_][]const u32{path[0 .. na + 1]};
    try m.r.keepPaths(&paths);
    for (0..na) |i| out[i] = m.chain[slot][i];
    out[na] = m.r.block[acc];
    return .{ .tokens = out[0 .. na + 1], .drafted = k, .accepted = acc };
}

/// The runner's prompt hook: a chunk's rows into the MTP head's cache (the same call on a follower rank).
fn promptHook(ptr: *anyopaque, ids: []const u32, p0: u32, after: u32) anyerror!void {
    const m: *Mtp = @ptrCast(@alignCast(ptr));
    try m.absorbPrompt(m.r.slot, ids, p0, after);
}

fn copyInIds(m: *Mtp, n: usize) !void {
    try ze.check(m.r.d.g.api.zeCommandListAppendMemoryCopy(m.r.d.list, m.ids, @as(*const anyopaque, @ptrCast(&m.ids_host)), n * 4, null, 0, null));
}

fn copyInMt(m: *Mtp, n: usize) !void {
    try ze.check(m.r.d.g.api.zeCommandListAppendMemoryCopy(m.r.d.list, m.mt, @as(*const anyopaque, @ptrCast(&m.mt_host)), n * 8 * 8, null, 0, null));
    try ze.check(m.r.d.g.api.zeCommandListAppendMemoryCopy(m.r.d.list, m.mpos, @as(*const anyopaque, @ptrCast(&m.pos_host)), n * 4, null, 0, null));
}

fn keptWords(m: *Mtp, items: []const Kept) []const u32 {
    var at: usize = 0;
    for (items) |it| {
        m.words[at] = it.slot;
        m.words[at + 1] = it.pos;
        m.words[at + 2] = @intCast(it.block.len);
        at += 3;
        for (it.block) |b| {
            m.words[at] = b;
            at += 1;
        }
        for (it.follow) |f| {
            m.words[at] = f;
            at += 1;
        }
    }
    return m.words[0..at];
}

fn askWords(m: *Mtp, asks: []const Ask) []const u32 {
    var at: usize = 0;
    for (asks) |a| {
        m.words[at] = a.slot;
        m.words[at + 1] = a.pending;
        m.words[at + 2] = a.depth;
        at += 3;
    }
    return m.words[0..at];
}

};

/// The kept rows a follower rank rebuilds from a record (the same slices absorbKept takes).
pub const KeptBuf = struct {
    items: [ROWS]Kept = undefined,
    block: [ROWS][ROWS]u32 = undefined,
    follow: [ROWS][ROWS]u32 = undefined,
    n: u32 = 0,

    pub fn from(words: []const u32) KeptBuf {
        var b: KeptBuf = .{};
        var at: usize = 0;
        while (at + 3 <= words.len and b.n < ROWS) {
            const slot = words[at];
            const pos = words[at + 1];
            const n: usize = words[at + 2];
            at += 3;
            if (at + 2 * n > words.len) break;
            @memcpy(b.block[b.n][0..n], words[at..][0..n]);
            at += n;
            @memcpy(b.follow[b.n][0..n], words[at..][0..n]);
            at += n;
            b.items[b.n] = .{ .slot = slot, .pos = pos, .block = b.block[b.n][0..n], .follow = b.follow[b.n][0..n] };
            b.n += 1;
        }
        return b;
    }
    pub fn list(b: *const KeptBuf) []const Kept {
        return b.items[0..b.n];
    }
};

/// The streams a follower rank rebuilds from a record.
pub const AskBuf = struct {
    items: [ROWS]Ask = undefined,
    n: u32 = 0,

    pub fn from(words: []const u32) AskBuf {
        var b: AskBuf = .{};
        var at: usize = 0;
        while (at + 3 <= words.len and b.n < ROWS) {
            b.items[b.n] = .{ .slot = words[at], .pending = words[at + 1], .depth = words[at + 2] };
            at += 3;
            b.n += 1;
        }
        return b;
    }
    pub fn list(b: *const AskBuf) []const Ask {
        return b.items[0..b.n];
    }
};

test "a kept record round-trips its rows" {
    var m: Mtp = undefined;
    const items = [_]Kept{
        .{ .slot = 2, .pos = 100, .block = &.{ 3, 4 }, .follow = &.{ 7, 8 } },
        .{ .slot = 5, .pos = 40, .block = &.{9}, .follow = &.{11} },
    };
    const words = Mtp.keptWords(&m, &items);
    const b = KeptBuf.from(words);
    try std.testing.expectEqual(@as(u32, 2), b.n);
    const got = b.list();
    try std.testing.expectEqual(@as(u32, 2), got[0].slot);
    try std.testing.expectEqual(@as(u32, 100), got[0].pos);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, got[0].block);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, got[0].follow);
    try std.testing.expectEqualSlices(u32, &.{9}, got[1].block);
    try std.testing.expectEqualSlices(u32, &.{11}, got[1].follow);
}

test "an ask record round-trips its streams" {
    var m: Mtp = undefined;
    const asks = [_]Ask{ .{ .slot = 1, .pending = 42, .depth = 3 }, .{ .slot = 4, .pending = 7, .depth = 2 } };
    const b = AskBuf.from(Mtp.askWords(&m, &asks));
    try std.testing.expectEqual(@as(u32, 2), b.n);
    try std.testing.expectEqual(@as(u32, 42), b.list()[0].pending);
    try std.testing.expectEqual(@as(u32, 3), b.list()[0].depth);
    try std.testing.expectEqual(@as(u32, 4), b.list()[1].slot);
}

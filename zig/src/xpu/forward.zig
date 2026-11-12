//! One Qwen3.8-27B forward on the XPU: activations, prompt GEMMs (fp16, W4A8, W4A4) and the layer loop.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const loader = @import("load.zig");
const exl3 = @import("qwen27_exl3.zig");
const D = q.D;
const Dev = dev.Dev;
const EPS = q.EPS;
const FF = q.FF_R;
const GDN_QKV = q.GDN_QKV;
const GDN_AB = q.GDN_AB;
const HK = q.GDN_K_HEADS;
const HV = q.GDN_V_HEADS;
const HQ = q.ATT_Q_HEADS;
const AKV = q.ATT_KV_HEADS;
const MIX_OUT = q.GDN_Z; // a mixer's output width a rank: 48 value heads of 128, or 24 attention heads of 256, at TP 1
const GDN_IN = q.GDN_IN;
const LAYERS = q.LAYERS;
const MASK_WORDS = q.MASK_WORDS;
const Model = loader.Model;
const Proj = loader.Proj;
const SPLIT = q.SPLIT;
const limits = @import("limits.zig");
const k1mod = @import("k1.zig"); // XPU_K1: the split-K DPAS decode over int8 pool pages
const ST_B = q.ST_B;
const ST_G = q.ST_G;
const ST_K = q.ST_K;
const ST_P = q.ST_P;
const ST_Q = q.ST_Q;
const ST_V = q.ST_V;
const TAP_LAYERS = q.TAP_LAYERS;
const VOCAB = q.VOCAB;
const alloc = q.alloc;
const ranks = @import("tp.zig"); // the TP ranks' all-reduce (null at TP 1)
extern "c" fn close(fd: c_int) c_int;
const copy = dev.copy;
const off = dev.off;

// ---------------------------------------------------------------- activations and the forward

pub const Acts = struct {
    x: *anyopaque, // fp32 [maxRows, D]
    h: *anyopaque, // fp16 [maxRows, FF]
    ahi: *anyopaque, // fp16 [maxRows, FF / 64]
    alo: *anyopaque,
    r: *anyopaque, // fp16 [maxRows, D]
    p: *anyopaque, // fp16 [maxRows, 2 FF]
    q: *anyopaque, // bf16 [maxRows, 16, 128] (GDN) / [maxRows, 24, 256] (attention)
    k: *anyopaque, // bf16 [maxRows, 16, 128]
    v: *anyopaque, // bf16 [maxRows, 48, 128]
    g: *anyopaque, // fp32 [maxRows, 48]
    beta: *anyopaque,
    y: *anyopaque, // bf16 [maxRows, 6144] (GDN out, attention out)
    la: *anyopaque,
    wb: *anyopaque, // wb, hs, vn: hi and lo fp16 planes of an fp32 operand, then its row exponents (float)
    u0: *anyopaque,
    hs: *anyopaque,
    vn: *anyopaque,
    pm: *anyopaque,
    pl: *anyopaque,
    po: *anyopaque,
    spm: ?*anyopaque = null, // XPU_K1_SHADOW: the scratch partials of the shadow launches
    spl: ?*anyopaque = null,
    spo: ?*anyopaque = null,
    nsx: ?*anyopaque = null, // XPU_NORM_SHADOW: the scratch X, H and row sums of the shadow launches
    nsh: ?*anyopaque = null,
    nsa: ?*anyopaque = null,
    nsl: ?*anyopaque = null,
    pscr: *anyopaque, // fp32 [maxRows, D] (slice totals of the narrow prompt GEMM)
    logits: *anyopaque, // fp16 [16, VOCAB]
    taps: *anyopaque, // bf16 [maxRows, 5 D]: hidden states after TAP_LAYERS, when Mode.taps
    tok: *anyopaque, // i32
    pv: *anyopaque, // argmax partials: f32 [256], i32 [256]
    pi: *anyopaque,
    mrows: *anyopaque, // i32 [16]: argmaxMasked's rows
    mbits: *anyopaque, // u32 [16, MASK_WORDS]: and their grammar bits
    mtok: *anyopaque, // i32 [16]: and their tokens
    ids: *anyopaque, // i32 [maxRows]
    zeros: *anyopaque, // padding rows of the GDN chunks: k = v = 0, β = 0 …
    ones: *anyopaque, // … and g = 1
    chain: *anyopaque, // i32 [16, 17]: the tree table of a chain (row t at depth t), for the decode attention
    tree: *anyopaque, // i32 [16, 17]: the verified block's tree table (decode.cl, DEPTH and ANC)
    path: *anyopaque, // i32 [16]: the rows of the kept path
    mpath: *anyopaque, // i32 [16, 2, 16]: keepPaths' per-window paths, rows of the block then rows of the window
    sl: *anyopaque, // fp32 [16, 48, 128, 128]: GDN state after each row of a verified tree
    mt: *anyopaque, // u64 [LAYERS, 16, 8]: stepMulti's per-row state addresses
    mpos: *anyopaque, // i32 [16]: and positions
    a8: *anyopaque, // W4A8 prompt: activations int8 [maxRows, FF] (permuted by blocks of 8) …
    sa: *anyopaque, // … and their scales fp32 [FF / 128, maxRows + 32]
    grp: *anyopaque, // i32 [16, 2]: K1's row groups (first row, rows)
    kmt: *anyopaque, // u64 [16, 8]: the K1 tree verify's MT table (the slot's kc, vc, page table)
    kpos: *anyopaque, // i32 [16]: its K1 POS (every row's last prefix key, p0 - 1)
    kposm: *anyopaque, // i32 [16]: its merge POS (p0 + span - 1, so the draft's slice NS joins)

    pub fn init(d: *Dev) !Acts {
        const rows = q.maxRows();
        const nc = limits.chunks(rows);
        const ns = limits.splits(q.MAX_CTX, SPLIT);
        return .{
            .x = try d.alloc_(rows * D * 4),
            .h = try d.alloc_(rows * FF * 2),
            .ahi = try d.alloc_(rows * (FF / 64) * 2),
            .alo = try d.alloc_(rows * (FF / 64) * 2),
            .r = try d.alloc_(rows * D * 2),
            .p = try d.alloc_(rows * 2 * FF * 2),
            .q = try d.alloc_(rows * HQ * 256 * 2),
            .k = try d.alloc_(rows * HK * 128 * 2),
            .v = try d.alloc_(rows * HV * 128 * 2),
            .g = try d.alloc_(rows * HV * 4),
            .beta = try d.alloc_(rows * HV * 4),
            .y = try d.alloc_(rows * MIX_OUT * 2),
            .la = try d.alloc_(HV * nc * 64 * 4),
            .wb = try d.alloc_(3 * nc * HV * 64 * 128 * 2),
            .u0 = try d.alloc_(nc * HV * 64 * 128 * 4),
            .hs = try d.alloc_(3 * nc * HV * 128 * 128 * 2),
            .vn = try d.alloc_(3 * nc * HV * 128 * 64 * 2),
            // decode attention partials: at most 16 rows (the prompt kernel needs none)
            .pm = try d.alloc_(16 * HQ * ns * 4),
            .pl = try d.alloc_(16 * HQ * ns * 4),
            .po = try d.alloc_(16 * HQ * ns * 256 * 4),
            .spm = if (q.k1_shadow[0] != null) try d.alloc_(16 * HQ * ns * 4) else null,
            .spl = if (q.k1_shadow[0] != null) try d.alloc_(16 * HQ * ns * 4) else null,
            .spo = if (q.k1_shadow[0] != null) try d.alloc_(16 * HQ * ns * 256 * 4) else null,
            .nsx = if (q.norm_shadow[0] != null) try d.alloc_(16 * D * 4) else null,
            .nsh = if (q.norm_shadow[0] != null) try d.alloc_(16 * D * 2) else null,
            .nsa = if (q.norm_shadow[0] != null) try d.alloc_(16 * D * 2) else null,
            .nsl = if (q.norm_shadow[0] != null) try d.alloc_(16 * D * 2) else null,
            .pscr = try d.alloc_(rows * D * 4),
            .logits = try d.alloc_(16 * VOCAB * 2),
            .tok = try d.alloc_(16 * 4),
            .pv = try d.alloc_(16 * 1024),
            .pi = try d.alloc_(16 * 1024),
            .mrows = try d.alloc_(16 * 4),
            .mbits = try d.alloc_(16 * MASK_WORDS * 4),
            .mtok = try d.alloc_(16 * 4),
            .taps = try d.alloc_(rows * 5 * D * 2),
            .ids = try d.alloc_(rows * 4),
            .zeros = try d.alloc_(rows * HV * 128 * 2),
            .ones = try d.alloc_(rows * HV * 4),
            .chain = try d.upload(std.mem.sliceAsBytes(&chainTable())),
            .tree = try d.alloc_(16 * 17 * 4),
            .path = try d.alloc_(16 * 4),
            .mpath = try d.alloc_(16 * 2 * 16 * 4),
            .sl = try d.alloc_(16 * HV * 128 * 128 * 4),
            .mt = try d.alloc_(LAYERS * 16 * 8 * 8),
            .mpos = try d.alloc_(16 * 4),
            .a8 = try d.alloc_(rows * FF),
            .sa = try d.alloc_((FF / 128) * (rows + 32) * 4),
            .grp = try d.alloc_(16 * 2 * 4),
            .kmt = try d.alloc_(16 * 8 * 8),
            .kpos = try d.alloc_(16 * 4),
            .kposm = try d.alloc_(16 * 4),
        };
    }
};

/// Row t at depth t, its ancestors rows 0 .. t: a chain's tree table.
pub fn chainTable() [16 * 17]i32 {
    var t: [16 * 17]i32 = @splat(0);
    for (0..16) |r| {
        t[r * 17] = @intCast(r);
        for (0..r + 1) |dd| t[r * 17 + 1 + dd] = @intCast(dd);
    }
    return t;
}

/// gate|up and SwiGLU: the activation and its row sums; prompt rows via the GEMM epilogue, decode rows via `swiglu`.
pub fn mlpIn(d: *Dev, m: *const Model, a: *Acts, w: Proj, T: u32, gs: i32, pre: bool, wide: bool) !*anyopaque {
    if (w.x3) |e| { // EXL3: [gate | up] (ilv 0), then swiglu; the W4A8 and nGEN epilogues are int4 only
        try exl3.project(e, m.exl3.?, d, a.h, a.p, T, wide);
        try d.launch(d.k.swiglu, .{ .x = T, .y = (FF + 1023) / 1024 }, .{ a.p, a.h, a.ahi, a.alo, gs, @as(i32, 0), q.rot_on, @as(i32, FF) });
        return a.h;
    }
    if (q.prefill_int8 and wide and w.ilv and w.gs == 128) { // W4A8: SwiGLU epilogue; down quantizes its own input
        if (!pre) try quantA8(d, a, a.h, w.k, T, 0);
        try d.launch(d.k.a8_swiglu, a8Groups(w, T), a8Args(a, w, a.p, T));
        return a.p;
    }
    if (wide and w.ilv and gemm_hook == null) {
        const M: i32 = @intCast(T);
        const tiles = (T + 63) / 64;
        const k = if (w.gs == 128) d.k.ngen_swiglu_g128 else d.k.ngen_swiglu;
        try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, .{ a.h, w.w, w.s, w.bg, a.p, M, @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast((T + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) });
        try d.launch(d.k.rowsums_ff, .{ .x = T }, .{ a.p, a.ahi, a.alo, gs, q.rot_on });
        return a.p;
    }
    q.prof.site = gate_site;
    try gemmRows(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T, wide);
    q.prof.site = .none;
    try d.launch(d.k.swiglu, .{ .x = T, .y = (FF + 1023) / 1024 }, .{ a.p, a.h, a.ahi, a.alo, gs, @as(i32, @intFromBool(w.ilv)), q.rot_on, @as(i32, FF) });
    return a.h;
}

/// The rows' embeddings: x fp32 [T, D] (bf16 values) for the token ids at `ids` on the device.
/// `m.embed_w` is device memory by default; with XPU_EMBED_HOST=1 it is pinned host USM and `embed_dense`
/// reads its rows over PCIe, so no gather or staging buffer is needed on the host side.
pub fn embedRows(d: *Dev, m: *const Model, ids: *anyopaque, x: *anyopaque, rows: u32) !void {
    if (m.embed_s) |es|
        try d.launch(d.k.embed, .{ .x = rows }, .{ ids, m.embed_w, es, m.embed_b.?, x, @as(i32, D) })
    else
        try d.launch(d.k.embed_dense, .{ .x = rows }, .{ ids, m.embed_w, x, @as(i32, D) });
}

/// out[M, N] fp16 = h[M, K] · Wᵀ : decode kernels up to 16 rows, the nGEN prompt GEMMs above.
pub fn gemm(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    return gemmRows(d, pscr, h, ahi, alo, w, out, m, m > 16);
}

/// XPU_SITE_SPLIT (lane p4-k-first): the profile site of the gate|up launch of the layer in work. `forward` sets it.
var gate_site: q.Site = .gate_up;

/// XPU_MV cold shadows (lane p4-k-mv): the weights of each call site in the order of their first launch.
var mv_ring: [16][128]Proj = undefined;
var mv_ring_n: [16]u32 = @splat(0);

/// The place of `w` in the ring of its call site (a new W joins the ring). Null when the ring is full.
fn mvRingIndex(site: usize, w: Proj) ?u32 {
    const n = mv_ring_n[site];
    for (mv_ring[site][0..n], 0..) |r, i| if (r.w == w.w) return @as(u32, @intCast(i));
    if (n == mv_ring[site].len) return null;
    mv_ring[site][n] = w;
    mv_ring_n[site] = n + 1;
    return n;
}

/// The shadow launches of `list` on the shape of `w` (lane p4-k-mv). A cold shadow (`@@`) reads the W of another layer
/// of `site`: slot j takes the layer (j + 1) x n / 16 places on in the ring of the site's n layers, so a W is read
/// again after 4 layers at the earliest (64 layers). `j0` moves the slots of a list, so the lists of two places of
/// a layer read different layers.
fn mvShadows(d: *Dev, list: *const [16]?q.MvShadow, site: usize, j0: u32, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, m: u32) !void {
    if (list[0] == null) return;
    const at = mvRingIndex(site, w);
    const n = mv_ring_n[site];
    const step = @max(1, n / 16);
    for (list, 0..) |sh, j| {
        const shd = sh orelse break;
        var ws = w;
        if (shd.cold) {
            const i = at orelse continue;
            const t = (i + step * (j0 + @as(u32, @intCast(j)) + 1)) % n;
            ws = mv_ring[site][t];
            if (t == i or ws.n != w.n or ws.k != w.k or ws.gs != w.gs or ws.wide != w.wide) continue;
        }
        if (w.n % shd.div == 0 and @as(usize, m) * w.n * 2 <= q.mv_scratch_bytes)
            try d.launch(shd.k, .{ .x = 1, .y = w.n / shd.div }, .{ h, ws.w, ws.s, ws.bg, q.mv_scratch.?, @as(i32, @intCast(m)), @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), ahi, alo, @as(i32, @intCast((m + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) });
    }
}

/// XPU_DIRTY (lane p4-k-first): `probe_wr` writes dirty_mb MB of the matvec scratch buffer (256 uint a work-group).
fn dirtyProbe(d: *Dev) !void {
    const k = q.dirty_k orelse return;
    const site = q.prof.site;
    q.prof.site = .none;
    defer q.prof.site = site;
    try d.launch(k, .{ .x = q.dirty_mb * 1024 }, .{q.mv_scratch.?});
}

/// XPU_MV pre1 and pre2 (lane p4-k-first): timing launches on the shape of the layer's gate|up before the launch
/// of that site. Place 0 (pre1): after mixer_out, before its exchange. Place 1 (pre2): after that exchange, before
/// the norm. They read h as it is and write the scratch buffer, so the model keeps its bits. Decode rows only.
fn mvPre(d: *Dev, a: *Acts, place: usize, w: Proj, T: u32) !void {
    if (w.x3 != null or w.gs == 128) return;
    const site = q.prof.site;
    q.prof.site = .none; // a row of its own kernel only
    defer q.prof.site = site;
    try mvShadows(d, &q.mv_pre[place], @intFromEnum(q.Site.gate_up), @intCast(5 * (place + 1)), a.h, a.ahi, a.alo, w, T);
}

/// `gemm` with the path chosen by the caller: `wide`, the nGEN prompt GEMMs whatever the rows (a prompt's chunk).
pub fn gemmRows(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32, wide: bool) !void {
    const M: i32 = @intCast(m);
    const N: i32 = @intCast(w.n);
    const K: i32 = @intCast(w.k);
    const G: i32 = @intCast(w.k / w.gs);
    const g128 = w.gs == 128;
    const k = d.k;
    const dec_wide = if (g128) k.dec_wide_g128 else k.dec_wide;
    const dec_wide64 = if (g128) k.dec_wide64_g128 else k.dec_wide64;
    const dec_narrow = if (g128) k.dec_narrow_g128 else k.dec_narrow;
    const ngen_wide = if (g128) k.ngen_wide_g128 else k.ngen_wide;
    const ngen_narrow = if (g128) k.ngen_narrow_g128 else k.ngen_narrow;
    const mp: i32 = @intCast((m + 7) / 8 * 8);
    // XPU_MV shadows (lane p4-k-mv): timing launches of other kernels on this call's shape, the output to a scratch buffer.
    // With XPU_SITE_SPLIT the gate|up of an attention layer has its own profile site and the shadows and pick of gate_up.
    const site = @intFromEnum(if (q.prof.site == .gate_up_attn) q.Site.gate_up else q.prof.site);
    if (!wide and !g128) try mvShadows(d, &q.mv_shadow[site], site, 0, h, ahi, alo, w, m);
    // XPU_MV (lane p4-k-mv): the pick of this call site, for a shape its tile divides
    const pick: ?q.MvPick = if (!wide and !g128) q.mv_pick[site] else null;
    if (pick != null and pick.?.wide == w.wide and w.n % pick.?.div == 0) {
        try d.launch(pick.?.k, .{ .x = 1, .y = w.n / pick.?.div }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    } else if (!wide) {
        if (w.wide)
            // 64 columns per subgroup for the widest shapes (gate|up, head), 32 below (measured end to end)
            if (w.n >= 30000)
                try d.launch(dec_wide64, .{ .x = 1, .y = w.n / 256 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try d.launch(dec_wide, .{ .x = 1, .y = w.n / 128 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(dec_narrow, .{ .x = 1, .y = w.n / 32 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    } else {
        const tiles = (m + 63) / 64;
        const groups: ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
        if (w.wide)
            try d.launch(ngen_wide, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(ngen_narrow, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G, pscr });
    }
    if (gemm_hook) |f| try f(d, h, w, out, m);
}

/// XPU_FIDELITY: called after every GEMM (its input rows, weights and output); null in production.
pub var gemm_hook: ?*const fn (*Dev, *anyopaque, Proj, *anyopaque, u32) anyerror!void = null;

extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

pub fn dump(d: *Dev, a: *Acts, T: u32, name: []const u8, with_r: bool) !void {
    const dir = q.dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, T * D);
    defer alloc.free(x);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, x.ptr, a.x, T * D * 4, null, 0, null));
    errdefer d.settle(); // x is freed on the way out: the copy into it must be done first
    if (with_r) {
        const r = try alloc.alloc(f16, T * D);
        defer alloc.free(r);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.ptr, a.r, T * D * 2, null, 0, null));
        try d.sync();
        for (x, r) |*xv, rv| xv.* += @floatCast(rv);
    } else try d.sync();
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_x_{s}.f32", .{ dir, name }, 0);
    const fd = creat(path.ptr, 0o644);
    _ = write(fd, @ptrCast(x.ptr), x.len * 4);
    _ = close(fd);
}

/// A device buffer of `n` values (2 bytes: fp16 or bf16, 4: fp32) as fp32 in DIR/xpu_{name}.f32.
pub fn dumpBuf(d: *Dev, name: []const u8, p: *anyopaque, n: usize, kind: enum { f16, bf16, f32 }) !void {
    const dir = q.dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const raw = try alloc.alloc(u16, n * 2);
    defer alloc.free(raw);
    const bytes: usize = if (kind == .f32) n * 4 else n * 2;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, raw.ptr, p, bytes, null, 0, null));
    try d.sync();
    for (x, 0..) |*v, i| v.* = switch (kind) {
        .f16 => @floatCast(@as(f16, @bitCast(raw[i]))),
        .bf16 => @bitCast(@as(u32, raw[i]) << 16),
        .f32 => @bitCast(@as(u32, raw[2 * i]) | (@as(u32, raw[2 * i + 1]) << 16)),
    };
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_{s}.f32", .{ dir, name }, 0);
    const fd = creat(path.ptr, 0o644);
    _ = write(fd, @ptrCast(x.ptr), x.len * 4);
    _ = close(fd);
}

/// What a forward leaves: nothing, the last row's logits and greedy token, or every row's (verification, ≤ 16 rows).
pub const Want = enum { none, last, rows };
/// Bytes of one layer's K (or V) cache on this rank: its own K/V heads (4 at TP 1, 2 at TP 2), not the model's.
pub fn kvBytes() usize {
    return @as(usize, q.MAX_CTX) * @as(usize, q.ATT_KV_HEADS) * @as(usize, if (q.kv_i8) 256 + 4 else 256 * 2);
}
pub fn kvCap() i32 {
    return if (q.kv_i8) @intCast(q.MAX_CTX) else 0;
}
/// add_rmsnorm without the fused W4A8 quantization (its A8 and SA arguments).
pub const no_q8: ?*anyopaque = null;

/// Prompt GEMM: W4A8 when on and the rows are prompt rows, else fp16; `pre`: x already in acts.a8; `qrot`: see quantA8.
pub fn pgemm(d: *Dev, m: *const Model, a: *Acts, x: *anyopaque, w: Proj, out: *anyopaque, T: u32, pre: bool, qrot: i32, int4: bool, wide: bool) !void {
    if (w.x3) |e| return exl3.project(e, m.exl3.?, d, x, out, T, wide); // EXL3: no W4A8 or W4A4 copies
    if (!q.prefill_int8 or !wide or w.gs != 128) return gemmRows(d, a.pscr, x, a.ahi, a.alo, w, out, T, wide);
    if (int4 and q.prefill_int4) {
        try d.launch(d.k.quant_a4, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(w.k)), @as(i32, @intCast(q.maxRows() + 32)), qrot });
        return d.launch(d.k.a4, a8Groups(w, T), a8Args(a, w, out, T));
    }
    if (!pre) try quantA8(d, a, x, w.k, T, qrot);
    try d.launch(d.k.a8, a8Groups(w, T), a8Args(a, w, out, T));
}
pub fn quantA8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, rot: i32) !void {
    try d.launch(d.k.quant_a8, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(q.maxRows() + 32)), rot });
}
pub fn a8Groups(w: Proj, T: u32) ze.GroupCount {
    const tiles = (T + 63) / 64;
    return .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
}
pub fn a8Args(a: *Acts, w: Proj, out: *anyopaque, T: u32) struct { *anyopaque, *anyopaque, *anyopaque, *anyopaque, *anyopaque, i32, i32, i32, *anyopaque, *anyopaque, i32, i32, *anyopaque } {
    return .{ a.a8, w.w, w.s, w.bg, out, @intCast(T), @intCast(w.n), @intCast(w.k), a.ahi, a.alo, @intCast(q.maxRows() + 32), @intCast(w.k / 128), a.sa };
}

pub fn a8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, site: u32) !void {
    if (q.a8_group == 0 or T <= 16 or q.a8_sites & site == 0) return;
    const qg = if (q.a8_group == std.math.maxInt(u32)) K else q.a8_group;
    try d.launch(d.k.fakequant_rows, .{ .x = T }, .{ x, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(qg)), @as(i32, @intCast(q.GS)), q.a8_rot, if (q.a8_sites4 & site != 0) q.a8_levels else @as(i32, 127) });
}
/// XPU_FP64_REF: called with each int4 GEMM's real input (site 0 mixer in … 4 head; layer LAYERS: the head).
pub var gemm_probe: ?*const fn (*Dev, *Acts, Proj, *anyopaque, u32, usize, u32) anyerror!void = null;
/// XPU_FP64_REF: called after a prompt's chunked GDN from position 0, with the state it left (layer i).
pub var gdn_probe: ?*const fn (*Dev, *Acts, *anyopaque, u32, usize) anyerror!void = null;
/// prompt: a prompt chunk, prompt kernels at any rows; tree: rows see their ancestors; multi: one row a slot.
/// multi.win: the windows' (first row, rows) pairs, as verifyMulti's win_host holds them (K1's row groups).
/// pt: the selected slot's page table (pool mode); with multi, each row's table is in MT.
pub const Mode = struct { prompt: bool = false, want: Want = .last, stash: bool = false, taps: bool = false, tree: bool = false, pt: ?*anyopaque = null, multi: ?struct { mt: *anyopaque, pos: *anyopaque, max_pos: u32, win: []const i32 = &.{}, kpos: []const i32 = &.{}, kkind: []const u8 = &.{} } = null };

/// K1: the row groups of the forward's attn_k1 launch; a copy of it is queued, so it lives as long as the process.
var grp_host: [32]i32 = undefined;

/// One forward over `ids` (< q.maxRows() rows) at p0..; the last row's greedy token; ≤ 16 non-prompt rows: GDN serial.
/// XPU_NORM_SHADOW (lane p4-k-norm): the norm probes before an add_rmsnorm_dec launch of T rows (16 or less). A
/// probe reads the X, R and weight of the launch and writes the scratch buffers of Acts.
fn normShadows(d: *Dev, a: *Acts, T: u32, wn: *anyopaque, has_r: i32, gs: i32, ms: i32) !void {
    const xo = a.nsx orelse return;
    const nul: ?*anyopaque = null;
    for (q.norm_shadow) |sh| try d.launch(sh orelse break, .{ .x = T }, .{ a.x, a.r, wn, a.nsh.?, a.nsa.?, a.nsl.?, @as(i32, D), EPS, has_r, gs, nul, nul, ms, @as(i32, 0), xo });
}

pub fn forward(d: *Dev, m: *Model, a: *Acts, ids: []const i32, p0: u32, mode: Mode) !?i32 {
    const want = mode.want != .none;
    if (ids.len > q.maxRows()) return error.TooManyRows; // the scratch holds maxRows, padded to 64
    const T: u32 = @intCast(ids.len);
    const wide = T > 16 or mode.prompt; // the prompt kernels: a prompt chunk cuts the same bits at any length
    const t32: i32 = @intCast(T);
    const tp = (T + 63) / 64 * 64; // GDN chunk rows (padded)
    const nul: ?*anyopaque = null;
    const gs: i32 = @intCast(q.GS);
    const tree: ?*anyopaque = if (mode.tree) a.tree else null;
    // W4A8 prompt: the norms before mixer in and gate|up quantize h themselves (not with the study mode)
    const q8on = q.prefill_int8 and wide and q.GS == 128 and q.a8_group == 0;
    const q8a: ?*anyopaque = if (q8on) a.a8 else null;
    const q8s: ?*anyopaque = if (q8on) a.sa else null;
    const q8ms: i32 = @intCast(q.maxRows() + 32);
    // the mixers' out projection in W4A8: its input quantized by attn_gate / gated_norm (not when it is in int4)
    const q8out = q8on and !(q.prefill_int4 and q.int4_sites & 2 != 0);
    const q8oa: ?*anyopaque = if (q8out) a.a8 else null;
    const q8os: ?*anyopaque = if (q8out) a.sa else null;
    const table = if (mode.tree) a.tree else a.chain;
    const mpos: ?*anyopaque = if (mode.multi) |mm| mm.pos else null;
    var fused = false; // the device-flag exchange ran the residual add and norm itself (TF_TP_DEVFLAG)
    // XPU_FUSE bit 1 (WP2 step 3): the device-direct exchange joins the norm that follows. Decode only, the direct
    // path only, and never when a probe reads the exchange's output ahead of that norm.
    const fuse1 = q.fuse & 1 != 0 and !wide and gemm_probe == null;
    var pend: ?*anyopaque = null; // the peer's partial an exchange left in pbuf, for the fused add of the next norm
    try dev.tableCopy(d, a.ids, ids.ptr, ids.len * 4); // XPU_HRING bit 2: through the ring
    errdefer d.settle(); // the queued copy reads the caller's ids: an error returns only once the queue is idle
    try embedRows(d, m, a.ids, a.x, T);
    if (p0 == 0) try dump(d, a, T, "embed", false);
    for (&m.layers, 0..) |*l, i| {
        q.prof.site = .norm_in; // XPU_PROFILE=2: the call site of the launches up to the next assignment
        if (!wide) try normShadows(d, a, T, l.in_norm, if (i == 0) 0 else 1, gs, q8ms);
        if (pend) |peer| { // XPU_FUSE bit 1: the previous layer's exchange joined into this layer's input norm
            try d.launch(d.k.tp_norm_peer, .{ .x = T }, .{ a.x, a.r, peer, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), gs, q8a, q8s, q8ms, q.rot_on });
            pend = null;
        } else try d.launch(if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, if (i == 0) 0 else 1), gs, q8a, q8s, q8ms, q.rot_on });
        q.prof.site = .none;
        const probe = p0 == 0 and i == 0;
        if (probe) try dumpBuf(d, "h0", a.h, T * D, .f16);
        // W4A8 study: the GEMM reads a quantized copy (in r, free until mixer_out); the dense a/b gates read h as is
        const min_in = if (q.a8_group != 0 and wide and q.a8_sites & 1 != 0) a.r else a.h;
        if (min_in != a.h) try copy(d, a.r, a.h, T * D * 2);
        try a8(d, a, min_in, D, T, 1);
        // XPU_GDN bit 1 (lane p4-k-gdn): a stashed GDN layer's GEMM and gdn_pre write the stash (the layout of the
        // copies they replace), and gdn_step and gated_norm read it there. The same bits, six copies a layer fewer
        const st_in: ?*anyopaque = if (q.gdn_bits & 1 != 0 and mode.stash and !wide and !l.attn) l.stash.? else null;
        const pb = if (st_in) |st| off(st, ST_P) else a.p;
        q.prof.site = if (l.attn) .mixer_in_attn else .mixer_in_gdn;
        try pgemm(d, m, a, min_in, l.mixer_in, pb, T, q8on, 0, false, wide);
        q.prof.site = .none;
        if (gemm_probe) |f| try f(d, a, l.mixer_in, min_in, T, i, 0);
        if (l.ab) |ab| try d.launch(d.k.dense_rows, .{ .x = (T + 15) / 16, .y = 2 * HV / 16 }, .{ a.h, ab, pb, t32, @as(i32, GDN_IN), @as(i32, GDN_AB) });
        if (probe) try dumpBuf(d, "p0", pb, T * GDN_IN, .f16);
        if (l.attn) {
            const ctx = if (mode.multi) |mm| mm.max_pos + 1 else p0 + T; // tree rows see fewer slices
            const mt: ?*anyopaque = if (mode.multi) |mm| off(mm.mt, i * 16 * 8 * 8) else null;
            const ns = (ctx + SPLIT - 1) / SPLIT;
            // the shared pool's page table of the current slot; a shared round carries each row's in MT
            const pt: ?*anyopaque = if (mode.multi != null) null else mode.pt;
            try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, @as(i32, @intCast(p0)), EPS, tree, mt, mpos, q.kv8, kvCap(), @as(?*anyopaque, null), pt });
            if (wide and q.kv_i8) {
                try d.launch(d.k.attn_prompt8, .{ .x = (T + 7) / 8, .y = AKV }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(f32, 1.0 / 16.0), kvCap(), pt });
            } else if (wide) {
                try d.launch(d.k.attn_prompt, .{ .x = (T + 7) / 8, .y = AKV }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(f32, 1.0 / 16.0), pt });
            } else {
                const path = k1mod.attnPath(.{ .on = k1mod.on, .tree = k1mod.tree, .multi_tree = k1mod.multi_tree, .i8 = q.kv_i8, .pool = l.pt != null, .round_tree = mode.tree, .shared = mode.multi != null, .has_k1 = d.k.attn_k1 != null, .has_draft = d.k.attn_k1_draft != null, .has_draft_mt = d.k.attn_k1_draft_mt != null });
                const k1on = path == .chain;
                const k1t = path == .tree;
                const k1mt = path == .multi_tree;
                if (k1t) { // K1 tree (lane p22-k1tree): the prefix keys through the 8-row groups, the draft keys
                           // through attn_k1_draft into slice NS, merged as one row of partials a (row, head)
                    const wf: [1]u32 = .{0};
                    const wn: [1]u32 = .{T};
                    // XPU_K1_R16=2 (lane p4-kernel): a window of 9 to 16 rows is one group through attn_k1_k16
                    const k16 = k1mod.k16t and T > k1mod.ROWS and d.k.attn_k1_k16 != null;
                    const n = if (k16) blk: {
                        grp_host[0] = 0;
                        grp_host[1] = @intCast(T);
                        break :blk @as(u32, 1);
                    } else k1mod.rowGroups(&wf, &wn, grp_host[0..32]);
                    const plan = k1mod.planTile(p0, n, AKV, k1mod.wave, if (k16) k1mod.TILE_K16 else k1mod.TILE);
                    const p0i: i32 = @intCast(p0);
                    const posm: i32 = @as(i32, @intCast(plan.ns)) * plan.span + plan.span - 1; // used = NS + 1
                    for (0..16) |r| {
                        k1mod.mt_host[i][r * 8] = @intFromPtr(l.kc.?);
                        k1mod.mt_host[i][r * 8 + 1] = @intFromPtr(l.vc.?);
                        k1mod.mt_host[i][r * 8 + 2] = @intFromPtr(l.pt.?);
                    }
                    for (0..16) |r| {
                        k1mod.pos_host[r] = p0i - 1;
                        k1mod.posm_host[r] = posm;
                    }
                    try dev.hostCopy(d, a.kmt, &k1mod.mt_host[i], 16 * 8 * 8);
                    try dev.hostCopy(d, a.kpos, &k1mod.pos_host, 16 * 4);
                    try dev.hostCopy(d, a.kposm, &k1mod.posm_host, 16 * 4);
                    try k1mod.queueGroups(d, a.grp, grp_host[0 .. 2 * n], n);
                    // the partials' stride is ns + 1 (k1.cl attn_k1_draft): the prefix launch and the merge take it as NS,
                    // the draft takes ns as its slice index
                    const ns_stride: i32 = @as(i32, @intCast(plan.ns)) + 1;
                    if (a.spo != null) for (q.k1_shadow) |sh| try d.launch(sh orelse break, .{ .x = plan.ns, .y = AKV, .z = n }, .{ a.q, l.kc.?, l.vc.?, a.spm.?, a.spl.?, a.spo.?, @as(i32, p0i), ns_stride, plan.span, @as(f32, 1.0 / 16.0), a.kmt, a.kpos, kvCap(), l.pt.?, a.grp }); // XPU_K1_SHADOW: timing launches, scratch partials
                    try d.launch(if (k16) d.k.attn_k1_k16.? else d.k.attn_k1.?, .{ .x = plan.ns, .y = AKV, .z = n }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, p0i), ns_stride, plan.span, @as(f32, 1.0 / 16.0), a.kmt, a.kpos, kvCap(), l.pt.?, a.grp });
                    try d.launch(d.k.attn_k1_draft.?, .{ .x = 1, .y = AKV, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, p0i), @as(i32, @intCast(plan.ns)), @as(f32, 1.0 / 16.0), a.tree, l.pt.? });
                    try d.launch(d.k.merge_span, .{ .x = HQ, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, p0i), ns_stride, plan.span, a.kposm });
                } else if (k1mt) { // K1 multi-tree (lane p33-multitree, XPU_MULTI_TREE): the prefix keys through the
                                   // decode kernels with per-row POS (a chain row its mpos, a tree row its window base
                                   // - 1), the tree rows' draft keys (their ancestors and themselves) through
                                   // attn_k1_draft_mt into slice NS, one attn_merge_span join. A tree row's empty
                                   // prefix slices hold PM = -inf (k1_body writes them), so its merge reads all
                                   // ns + 1 slices; a chain row skips the draft slice (k1.zig mergePos)
                    const mm2 = mode.multi.?;
                    const rd = k1mod.roundGroups(mm2.win, T, &grp_host);
                    const tile = if (rd.k16) k1mod.TILE16 else k1mod.TILE;
                    const plan = k1mod.planTile(ctx, rd.n, AKV, k1mod.wave, tile);
                    const kern = if (rd.k16) d.k.attn_k1_16.? else d.k.attn_k1.?;
                    // the queued copy reads this buffer when the list runs, after forward returns: a global, not the stack
                    k1mod.mergePos(mm2.kkind, mm2.kpos, plan.ns, plan.span, &k1mod.posm_host);
                    try dev.tableCopy(d, a.kpos, @ptrCast(&mm2.kpos[0]), 16 * 4); // XPU_HRING bit 2: through the ring
                    try dev.tableCopy(d, a.kposm, &k1mod.posm_host, 16 * 4);
                    try k1mod.queueGroups(d, a.grp, grp_host[0 .. 2 * rd.n], rd.n);
                    // the partials' stride is ns + 1 (k1.cl attn_k1_draft_mt): the prefix launch and the merge take
                    // it as NS, the draft takes ns as its slice index
                    const ns_stride: i32 = @as(i32, @intCast(plan.ns)) + 1;
                    if (a.spo != null and !rd.k16) for (q.k1_shadow) |sh| try d.launch(sh orelse break, .{ .x = plan.ns, .y = AKV, .z = rd.n }, .{ a.q, l.kc.?, l.vc.?, a.spm.?, a.spl.?, a.spo.?, @as(i32, @intCast(p0)), ns_stride, plan.span, @as(f32, 1.0 / 16.0), mt, a.kpos, kvCap(), l.pt.?, a.grp }); // XPU_K1_SHADOW: timing launches, scratch partials
                    try d.launch(kern, .{ .x = plan.ns, .y = AKV, .z = rd.n }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), ns_stride, plan.span, @as(f32, 1.0 / 16.0), mt, a.kpos, kvCap(), l.pt.?, a.grp });
                    try d.launch(d.k.attn_k1_draft_mt.?, .{ .x = 1, .y = AKV, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(plan.ns)), @as(f32, 1.0 / 16.0), a.tree, mt, mm2.pos });
                    try d.launch(d.k.merge_span, .{ .x = HQ, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), ns_stride, plan.span, a.kposm });
                } else if (k1on) { // K1: split-K DPAS over the slot's int8 pages, one work-group a row group of a stream
                    const rd = k1mod.roundGroups(if (mode.multi) |mm| mm.win else &.{}, T, &grp_host);
                    const tile = if (rd.k16) k1mod.TILE16 else k1mod.TILE;
                    const plan = k1mod.planTile(ctx, rd.n, AKV, k1mod.wave, tile);
                    const kern = if (rd.k16) d.k.attn_k1_16.? else d.k.attn_k1.?;
                    try k1mod.queueGroups(d, a.grp, grp_host[0 .. 2 * rd.n], rd.n);
                    if (a.spo != null and !rd.k16) for (q.k1_shadow) |sh| try d.launch(sh orelse break, .{ .x = plan.ns, .y = AKV, .z = rd.n }, .{ a.q, l.kc.?, l.vc.?, a.spm.?, a.spl.?, a.spo.?, @as(i32, @intCast(p0)), @as(i32, @intCast(plan.ns)), plan.span, @as(f32, 1.0 / 16.0), mt, mpos, kvCap(), l.pt.?, a.grp }); // XPU_K1_SHADOW: timing launches, scratch partials
                    try d.launch(kern, .{ .x = plan.ns, .y = AKV, .z = rd.n }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(plan.ns)), plan.span, @as(f32, 1.0 / 16.0), mt, mpos, kvCap(), l.pt.?, a.grp });
                    try d.launch(d.k.merge_span, .{ .x = HQ, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), @as(i32, @intCast(plan.ns)), plan.span, mpos });
                } else {
                    try d.launch(d.k.attn_decode, .{ .x = ns, .y = AKV, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), @as(f32, 1.0 / 16.0), if (mode.multi != null) tree else table, mt, mpos, kvCap(), pt });
                    try d.launch(d.k.attn_merge, .{ .x = HQ, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), table, mpos });
                }
            }
            if (q.kv8 & 1 != 0) try d.launch(d.k.rot_heads, .{ .x = T, .y = HQ }, .{a.y});
            try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, q.rot_on, q8oa, q8os, q8ms });
        } else {
            const cin = l.conv[l.conv_cur].?;
            const cout = l.conv[1 - l.conv_cur].?;
            const gq = if (st_in) |st| off(st, ST_Q) else a.q; // gdn_pre's outputs: the stash itself (XPU_GDN bit 1)
            const gk = if (st_in) |st| off(st, ST_K) else a.k;
            const gv = if (st_in) |st| off(st, ST_V) else a.v;
            const gg = if (st_in) |st| off(st, ST_G) else a.g;
            const gb = if (st_in) |st| off(st, ST_B) else a.beta;
            const gmt: ?*anyopaque = if (mode.multi) |mm| off(mm.mt, i * 16 * 8 * 8) else null;
            if (mode.multi == null) l.conv_cur = 1 - l.conv_cur; // multi: each slot's flips after the forward
            if (wide)
                try d.launch(d.k.gdn_pre_rows, .{ .x = (T + 7) / 8, .y = (GDN_QKV / 128 + 15) / 16 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32 })
            else
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = GDN_QKV / 256 }, .{ pb, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, gq, gk, gv, gg, gb, t32, tree, gmt });
            if (wide and tp > T) { // neutral padding rows: the state does not move
                const pad = tp - T;
                const cp = struct {
                    fn f(dd: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
                        try ze.check(dd.g.api.zeCommandListAppendMemoryCopy(dd.list, dst, src, n, null, 0, null));
                    }
                }.f;
                try cp(d, off(a.k, T * HK * 128 * 2), a.zeros, pad * HK * 128 * 2);
                try cp(d, off(a.v, T * HV * 128 * 2), a.zeros, pad * HV * 128 * 2);
                try cp(d, off(a.beta, T * HV * 4), a.zeros, pad * HV * 4);
                try cp(d, off(a.g, T * HV * 4), a.ones, pad * HV * 4);
            }
            if (!wide) {
                if (mode.stash and st_in == null) { // inputs, for Runner.commit's replay
                    const st = l.stash.?;
                    q.prof.site = .copy_stash;
                    try copy(d, off(st, ST_Q), a.q, T * HK * 128 * 2);
                    try copy(d, off(st, ST_K), a.k, T * HK * 128 * 2);
                    try copy(d, off(st, ST_V), a.v, T * HV * 128 * 2);
                    try copy(d, off(st, ST_G), a.g, T * HV * 4);
                    try copy(d, off(st, ST_B), a.beta, T * HV * 4);
                    try copy(d, off(st, ST_P), a.p, T * GDN_IN * 2);
                }
                // verification: post-block state to rec_save, rec keeps the entry; tree: states to acts.sl
                q.prof.site = .gdn_verify;
                defer q.prof.site = .none;
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = HV, .z = if (mode.multi != null) T else 1 }, .{ gq, gk, gv, gg, gb, l.rec.?, if (mode.stash) l.rec_save.? else l.rec.?, a.y, t32, nul, tree, a.sl, gmt });
            } else {
                const nc: i32 = @intCast(tp / 64);
                const ncu: u32 = tp / 64;
                const hk: i32 = HK;
                const hv: i32 = HV;
                try d.launch(d.k.gdn_prep, .{ .x = ncu, .y = HV }, .{ a.k, a.v, a.g, a.beta, a.la, a.wb, a.u0, hk, hv, nc, @as(i32, 0) });
                try d.launch(d.k.gdn_state, .{ .x = HV }, .{ a.k, a.wb, a.u0, a.la, l.rec.?, l.rec.?, a.hs, a.vn, hk, hv, nc });
                try d.launch(d.k.gdn_out, .{ .x = ncu, .y = HV }, .{ a.q, a.k, a.la, a.hs, a.vn, a.y, hk, hv, nc });
                if (gdn_probe) |f| if (p0 == 0) try f(d, a, l.rec.?, T, i);
            }
            if (probe) {
                try dumpBuf(d, "q0", gq, T * HK * 128, .bf16);
                try dumpBuf(d, "k0", gk, T * HK * 128, .bf16);
                try dumpBuf(d, "v0", gv, T * MIX_OUT, .bf16);
                try dumpBuf(d, "g0", gg, T * HV, .f32);
                try dumpBuf(d, "beta0", gb, T * HV, .f32);
                try dumpBuf(d, "y0", a.y, T * MIX_OUT, .bf16);
            }
            try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, pb, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs, q.rot_on, q8oa, q8os, q8ms });
            if (probe) try dumpBuf(d, "o0", a.h, T * MIX_OUT, .f16);
        }
        try a8(d, a, a.h, MIX_OUT, T, 2);
        q.prof.site = if (l.attn) .mixer_out_attn else .mixer_out_gdn;
        try pgemm(d, m, a, a.h, l.mixer_out, a.r, T, q8out, 0, q.int4_sites & 2 != 0, wide);
        q.prof.site = .none;
        if (!wide) try mvPre(d, a, 0, l.gu, T);
        if (ranks.group) |*g| { // TP: the ranks' partial out-projections summed
            if (!wide and g.devflag and g.ex == 0) { // TF_TP_DEVFLAG 1, 2: the add fused into the residual add and norm below
                try g.allReduceNorm(d, a.r, T, .{ .x = a.x, .w = l.post_norm, .h = a.h, .ahi = a.ahi, .alo = a.alo, .a8 = q8a, .sa = q8s, .ms = q8ms, .eps = EPS, .has_r = 1, .gs = gs, .rot = q.rot_on });
                fused = true;
            } else if (fuse1 and g.peerPath()) { // XPU_FUSE bit 1: the same add and norm on the device-direct or EX path
                const peer = try g.allReducePeer(d, a.r, T);
                try mvPre(d, a, 1, l.gu, T);
                try dirtyProbe(d);
                try d.launch(d.k.tp_norm_peer, .{ .x = T }, .{ a.x, a.r, peer, l.post_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), gs, q8a, q8s, q8ms, q.rot_on });
                fused = true;
            } else try g.allReduce(d, a.r, T);
        }
        if (gemm_probe) |f| try f(d, a, l.mixer_out, a.h, T, i, 1);
        if (probe) try dumpBuf(d, "r0", a.r, T * D, .f16);
        if (!fused and !wide) try mvPre(d, a, 1, l.gu, T); // the device-flag exchange holds the norm: it has no place 1
        if (!fused and !wide) try dirtyProbe(d);
        q.prof.site = .norm_post;
        if (!fused and !wide) try normShadows(d, a, T, l.post_norm, 1, gs, q8ms);
        if (!fused) try d.launch(if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), gs, q8a, q8s, q8ms, q.rot_on });
        q.prof.site = .none;
        try a8(d, a, a.h, D, T, 4);
        if (gemm_probe) |f| try f(d, a, l.gu, a.h, T, i, 2);
        gate_site = if (q.site_split and l.attn) .gate_up_attn else .gate_up;
        const act = try mlpIn(d, m, a, l.gu, T, gs, q8on, wide);
        try a8(d, a, act, FF, T, 8);
        q.prof.site = .down;
        try pgemm(d, m, a, act, l.down, a.r, T, false, if (act == a.p) q.rot_on else 0, q.int4_sites & 8 != 0, wide);
        q.prof.site = .none;
        if (ranks.group) |*g| { // TP: the ranks' partial down projections summed, joined into the next norm when bit 1 is set
            var tap_here = false;
            for (TAP_LAYERS) |tl| {
                if (tl == i) tap_here = true;
            }
            // tap and dump read r before that norm, so their layers keep the separate add
            const keep_r = (mode.taps and tap_here) or (q.dump_dir != null and p0 == 0 and i < 4);
            const has_norm = i + 1 < LAYERS or want or q.dump_h != null; // a norm follows this exchange
            if (fuse1 and g.peerPath() and !keep_r and has_norm) {
                pend = try g.allReducePeer(d, a.r, T);
            } else try g.allReduce(d, a.r, T);
        }
        if (gemm_probe) |f| try f(d, a, l.down, act, T, i, 3);
        if (p0 == 0 and i < 4) try dump(d, a, T, try std.fmt.allocPrint(alloc, "{d}", .{i}), true);
        if (mode.taps) for (TAP_LAYERS, 0..) |tl, slot| if (tl == i)
            try d.launch(d.k.tap, .{ .x = D / 256, .y = T }, .{ a.x, a.r, a.taps, @as(i32, D), @as(i32, @intCast(slot)) });
    }
    if (!want and q.dump_h == null) { // x is per token: nothing carries to the next chunk
        try d.sync();
        if (ranks.group) |*g| try g.check(d);
        return null;
    }
    const hgs: i32 = @intCast(m.head.gs);
    if (!wide) try normShadows(d, a, T, m.norm, 1, hgs, 0);
    if (pend) |peer| { // XPU_FUSE bit 1: the last layer's exchange joined into the final norm
        try d.launch(d.k.tp_norm_peer, .{ .x = T }, .{ a.x, a.r, peer, m.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), hgs, no_q8, no_q8, @as(i32, 0), @as(i32, 0) });
        pend = null;
    } else try d.launch(if (wide) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), hgs, no_q8, no_q8, @as(i32, 0), @as(i32, 0) });
    if (q.dump_h) |fd| { // XPU_DUMP_H: every row's final normed hidden state, appended
        const host = try alloc.alloc(u16, T * D);
        defer alloc.free(host);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, host.ptr, a.h, T * D * 2, null, 0, null));
        try d.sync();
        _ = write(fd, @ptrCast(host.ptr), T * D * 2);
        if (!want) return null;
    }
    if (gemm_probe) |f| try f(d, a, m.head, a.h, T, LAYERS, 4);
    const first = if (mode.want == .rows) 0 else T - 1;
    const rows = T - first;
    q.prof.site = .head;
    if (m.head.x3) |e| try exl3.project(e, m.exl3.?, d, off(a.h, first * D * 2), a.logits, rows, false) else try gemm(d, a.pscr, off(a.h, first * D * 2), off(a.ahi, first * (D / m.head.gs) * 2), off(a.alo, first * (D / m.head.gs) * 2), m.head, a.logits, rows);
    q.prof.site = .none;
    try d.launch(d.k.argmax_part, .{ .x = 256, .y = rows }, .{ a.logits, @as(i32, VOCAB), a.pv, a.pi });
    try d.launch(d.k.argmax, .{ .x = rows }, .{ a.pv, a.pi, a.tok });
    var toks: [16]i32 = undefined;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, a.tok, rows * 4, null, 0, null));
    try d.sync();
    if (ranks.group) |*g| try g.check(d); // TF_TP_DEVFLAG: a timed-out exchange (the peer went away) is PeerLost
    return toks[rows - 1];
}

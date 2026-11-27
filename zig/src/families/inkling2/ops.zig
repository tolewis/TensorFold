//! Inkling 2's glue kernels (inkling2/ops.cu): embedding, residual mix, row dots, router, SwiGLU epilogue, convs.
const std = @import("std");
const cuda = @import("cuda");

pub const max_blocks = 8;
pub const max_queries = 3;

/// ops.cu's MixArgs, field for field.
pub const MixArgs = extern struct {
    blocks: [max_blocks]u64 = @splat(0),
    partial: u64,
    query: u64,
    norm: [max_queries]u64 = @splat(0),
    out: u64,
    n_blocks: i32,
    partial_bf16: i32,
    nq: i32,
    rows: i32,
    c: i32,
    eps: f32,
};

/// ops.cu's ConvArgs, field for field.
pub const ConvArgs = extern struct {
    x: u64,
    state: u64,
    w: u64,
    dyn: u64 = 0,
    base: u64 = 0,
    resid: u64 = 0,
    out: u64 = 0,
    segs: u64 = 0, // int4 per stream: row0, rows, state slot (0: one stream)
    hist: u64 = 0, // fp32 [rows][c]: each row's input as the state keeps it (0: none)
    part_stride: i64 = 0, // elements between the ranks' partials of x (parts > 1)
    state_stride: i64 = 0, // elements between the streams' states
    parts: i32 = 1,
    x_bf16: i32 = 0,
    round_in: i32 = 0,
    heads: i32 = 1,
    c: i32,
    rows: i32,
    mode: i32,
};

/// ops.cu's GateUpArgs, field for field.
pub const GateUpArgs = extern struct {
    z: u64,
    pick: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    xd: u64,
    p: i32,
    n: i32,
    sk: i32,
    experts: i32,
    gate_cap: f32,
    up_cap: f32,
};

pub const Kernels = struct {
    mod: cuda.Module,
    rows_mod: cuda.Module,
    rowdot_tiles: cuda.Function,
    prompt_tail: cuda.Function,
    embed: cuda.Function,
    mix: cuda.Function,
    mix_rows: cuda.Function,
    mix_a: cuda.Function,
    mix_b: cuda.Function,
    mix_c: cuda.Function,
    rowdot: cuda.Function,
    topk: cuda.Function,
    topk_warp: cuda.Function,
    gateup: cuda.Function,
    conv: cuda.Function,
    softcap: cuda.Function,
    top1: cuda.Function,
    tobf: cuda.Function,
    put_words: cuda.Function,
    rewind: cuda.Function,
    race1: cuda.Function,
    lazy: [8]cuda.Function, // the certified lazy head: delta, prep, low, pick, fill, exact, full, low2
    split_cols: cuda.Function,
    masked_argmax: cuda.Function,
    masked_argmax16: cuda.Function,
    grammar_guard: cuda.Function,
    grammar_scrub: cuda.Function,
    vis_embed: cuda.Function,
    vis_pack_qkv_rope: cuda.Function,
    vis_ls_add: cuda.Function,
    vis_splice: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_ops);
        errdefer mod.unload();
        var rows_mod = try cuda.Module.load(d, cuda.kernels.inkling2_rows);
        errdefer rows_mod.unload();
        return .{
            .mod = mod,
            .rows_mod = rows_mod,
            .rowdot_tiles = try rows_mod.function("tf_ink2_rowdot_tiles"),
            .prompt_tail = try rows_mod.function("tf_ink2_prompt_tail"),
            .embed = try mod.function("tf_ink2_embed"),
            .mix = try mod.function("tf_ink2_mix"),
            .mix_rows = try mod.function("tf_ink2_mix_rows"),
            .mix_a = try mod.function("tf_ink2_mix_a"),
            .mix_b = try mod.function("tf_ink2_mix_b"),
            .mix_c = try mod.function("tf_ink2_mix_c"),
            .rowdot = try mod.function("tf_ink2_rowdot"),
            .topk = try mod.function("tf_ink2_topk"),
            .topk_warp = try mod.function("tf_ink2_topk_warp"),
            .gateup = try mod.function("tf_ink2_gateup"),
            .conv = try mod.function("tf_ink2_conv"),
            .softcap = try mod.function("tf_ink2_softcap"),
            .top1 = try mod.function("tf_ink2_top1"),
            .tobf = try mod.function("tf_ink2_tobf"),
            .put_words = try mod.function("tf_ink2_put_words"),
            .rewind = try mod.function("tf_ink2_conv_rewind"),
            .race1 = try mod.function("tf_ink2_race1"),
            .split_cols = try mod.function("tf_ink2_split_cols"),
            .masked_argmax = try mod.function("tf_ink2_masked_argmax"),
            .masked_argmax16 = try mod.function("tf_ink2_masked_argmax16"),
            .grammar_guard = try mod.function("tf_ink2_grammar_guard"),
            .grammar_scrub = try mod.function("tf_ink2_grammar_scrub"),
            .vis_embed = try mod.function("tf_ink2_vis_embed"),
            .vis_pack_qkv_rope = try mod.function("tf_ink2_vis_pack_qkv_rope"),
            .vis_ls_add = try mod.function("tf_ink2_vis_ls_add"),
            .vis_splice = try mod.function("tf_ink2_vis_splice"),
            .lazy = .{
                try mod.function("tf_ink2_lazy_delta"), try mod.function("tf_ink2_lazy_prep"),
                try mod.function("tf_ink2_lazy_low"),   try mod.function("tf_ink2_lazy_pick"),
                try mod.function("tf_ink2_lazy_fill"),  try mod.function("tf_ink2_lazy_exact"),
                try mod.function("tf_ink2_lazy_full"),  try mod.function("tf_ink2_lazy_low2"),
            },
        };
    }

    pub fn unload(k: *Kernels) void {
        k.rows_mod.unload();
        k.mod.unload();
        k.* = undefined;
    }
};

fn int(x: anytype) i32 {
    return @intCast(x);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// Embedding rows of `ids` (int32 on the device) through embed_norm into bf16 [rows, c].
pub fn embed(k: *const Kernels, s: cuda.Stream, ids: u64, rows: usize, table: u64, norm: u64, out: u64, c: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ ids, table, norm, out }) |v| a.add(v);
    a.add(int(c));
    a.add(eps);
    try go(k.embed, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// Whether tf_ink2_mix holds a row: ops.cu's kMixCache x 512 columns, 16-byte copies (Small 4,608, Mini 2,304).
pub fn mixFits(c: i32) error{Inkling2MixWidth}!void {
    if (c <= 0 or c > 10 * 512 or @rem(c, 128) != 0) return error.Inkling2MixWidth;
}

/// Whether tf_ink2_topk picks: 8 scores a thread of 512, 32 picks at most (Small 384 experts, Mini 128; top 8).
pub fn routerFits(experts: usize, top: usize) error{Inkling2RouterShape}!void {
    if (experts == 0 or experts > 512 * 8 or top == 0 or top > 32 or top > experts) return error.Inkling2RouterShape;
}

/// Whether dynamic conv taps split the width: column c reads head c / (C / heads) (Small 4 key heads, Mini 8).
pub fn convFits(c: ConvArgs) error{Inkling2ConvHeads}!void {
    if (c.dyn != 0 and (c.heads <= 0 or c.c <= 0 or @rem(c.c, c.heads) != 0)) return error.Inkling2ConvHeads;
}

/// The residual mix of each row and query, normed: a block a row.
pub fn mix(k: *const Kernels, s: cuda.Stream, m: MixArgs) !void {
    try mixFits(m.c);
    var a: cuda.Args = .{};
    a.add(m);
    // split-block mix: the same sums in the same order, the 16 warps of a row spread over B blocks
    if (m.rows <= 8 and m.query != 0) {
        if (std.c.getenv("TF_INKLING2_MIX_SPLIT")) |v| {
            const b = std.fmt.parseInt(i32, std.mem.span(v), 10) catch 0;
            if (b == 2 or b == 4 or b == 8 or b == 16) {
                const grid: cuda.Dim3 = .{ .x = @intCast(m.rows), .y = @intCast(b) };
                const threads: u32 = @intCast(@divTrunc(512, b));
                try go(k.mix_a, s, grid, threads, &a);
                try go(k.mix_b, s, grid, threads, &a);
                try go(k.mix_c, s, grid, threads, &a);
                return;
            }
        }
    }
    // the narrow kernel measured slower on GB10 (38 vs 27 us a decode row); TF_INKLING2_MIX_NARROW=1 restores it
    const narrow = m.rows <= 16 and std.c.getenv("TF_INKLING2_MIX_NARROW") != null;
    try go(if (narrow) k.mix else k.mix_rows, s, .{ .x = @intCast(m.rows) }, 512, &a); // equal bits either way
}

/// out[row, j] = x[row] . w[j] over `c` (bf16 in), `j` outputs a row, bf16 or fp32 out, sigmoid when `sigmoid`.
pub fn rowdot(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, j: usize, c: usize, out_bf16: bool, sigmoid: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, out }) |v| a.add(v);
    if (rows >= 32 and !out_bf16 and c % 128 == 0) { // prompt rows: 32 x 16 tiles, the same sums in the same order
        for ([_]i32{ int(rows), int(j), int(c), @intFromBool(sigmoid) }) |v| a.add(v);
        return go(k.rowdot_tiles, s, .{ .x = @intCast((rows + 31) / 32), .y = @intCast((j + 15) / 16) }, 256, &a);
    }
    for ([_]i32{ int(j), int(c), @intFromBool(out_bf16), @intFromBool(sigmoid) }) |v| a.add(v);
    try go(k.rowdot, s, .{ .x = @intCast(rows), .y = @intCast((j + 7) / 8) }, 256, &a);
}

/// The router's picks and weights from fp32 sigmoid scores [rows, experts] and the selection bias.
pub fn topk(k: *const Kernels, s: cuda.Stream, scores: u64, bias: u64, pick: u64, wts: u64, rows: usize, experts: usize, top: usize, scale: f32) !void {
    try routerFits(experts, top);
    var a: cuda.Args = .{};
    for ([_]u64{ scores, bias, pick, wts }) |v| a.add(v);
    a.add(int(experts));
    a.add(int(top));
    a.add(scale);
    if (topkWarp() and experts <= 256 and top <= 32) {
        a.add(int(rows));
        return go(k.topk_warp, s, .{ .x = @intCast((rows + 3) / 4) }, 128, &a);
    }
    try go(k.topk, s, .{ .x = @intCast(rows) }, 512, &a);
}

/// TF_INKLING2_TOPK_WARP=0: the router's picks a 512-thread block a row, as before (the warp form has the same bits).
fn topkWarp() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_TOPK_WARP")) |e| e[0] != '0' else true;
    return S.v.?;
}

/// The softcapped SwiGLU epilogue over `p` (row, slot) pairs, `n` wide: the grid of exl3_experts' gate/up epilogue.
pub fn gateup(k: *const Kernels, s: cuda.Stream, g: GateUpArgs) !void {
    var a: cuda.Args = .{};
    a.add(g);
    try go(k.gateup, s, .{ .x = @intCast(g.p), .y = @intCast(@divExact(g.n, 128)) }, 32, &a);
}

/// A short convolution over each of `streams` streams' rows, their states carried (ops.cu's tf_ink2_conv).
pub fn conv(k: *const Kernels, s: cuda.Stream, c: ConvArgs, streams: usize) !void {
    try convFits(c);
    var a: cuda.Args = .{};
    a.add(c);
    // a stream's rows in blocks of 32 (blocks past a stream's rows return at once)
    const z: u32 = @intCast(std.math.divCeil(i32, c.rows, 32) catch unreachable);
    try go(k.conv, s, .{ .x = @intCast(std.math.divCeil(i32, c.c, 128) catch unreachable), .y = @intCast(streams), .z = z }, 128, &a);
}

/// ops.cu's RewindSite: a conv site's slot-0 state, byte offsets of its rows in a hist and in a slot's backup, its C.
pub const RewindSite = extern struct { state: u64, hist: i64, backup: i64, c: i32, pad: i32 = 0 };

/// A slot's last window keeps `n` rows: every site's state from its backup (`backup`: the slot's) and `hist` rows.
pub fn rewind(k: *const Kernels, s: cuda.Stream, sites: u64, n_sites: usize, widest: usize, backup: u64, hist: u64, slot: usize, row0: usize, n: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ sites, backup, hist }) |v| a.add(v);
    for ([_]i32{ int(slot), int(row0), int(n) }) |v| a.add(v);
    try go(k.rewind, s, .{ .x = @intCast((widest + 127) / 128), .y = @intCast(n_sites) }, 128, &a);
}

/// tanh(x / cap) * cap over `n` fp32 logits in place.
pub fn softcap(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, cap: f32) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(n)));
    a.add(cap);
    try go(k.softcap, s, .{ .x = @intCast((n + 255) / 256) }, 256, &a);
}

/// Each of `rows` fp32 rows (`stride` apart): its largest value among the first `n` (lowest id on ties), id + `base`.
pub fn top1(k: *const Kernels, s: cuda.Stream, x: u64, rows: usize, stride: usize, n: usize, base: usize, val: u64, id: u64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(stride)));
    a.add(int(n));
    a.add(int(base));
    a.add(val);
    a.add(id);
    try go(k.top1, s, .{ .x = @intCast(rows) }, 1024, &a);
}

/// One fp32 row's keyed Gumbel race at position `pos` (sample.cu's keys, ids `base` + i): best score and id.
pub fn race1(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, base: usize, seed: u64, inv_t: f32, pos: usize, val: u64, id: u64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(n));
    a.add(int(base));
    a.add(seed);
    a.add(inv_t);
    a.add(@as(u32, @intCast(pos)));
    a.add(val);
    a.add(id);
    try go(k.race1, s, .{ .x = 1 }, 1024, &a);
}

/// `rows` fp32 rows into bf16 rows of `out` at column `col0`: `n` columns rounded, the rest of `width` set to -inf.
pub fn tobf(k: *const Kernels, s: cuda.Stream, x: u64, rows: usize, stride: usize, n: usize, out: u64, out_stride: usize, col0: usize, width: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(stride)));
    a.add(int(n));
    a.add(out);
    a.add(@as(i64, @intCast(out_stride)));
    a.add(@as(i64, @intCast(col0)));
    a.add(int(width));
    try go(k.tobf, s, .{ .x = @intCast((width + 255) / 256), .y = @intCast(rows) }, 256, &a);
}

/// ops.cu's Ink2Words: a link step's header and ids, passed by value.
pub const Words = extern struct { w: [96]u32 };

/// The first `n` of `words` to `dst` on the stream (no host buffer to keep alive).
pub fn putWords(k: *const Kernels, s: cuda.Stream, dst: u64, n: usize, words: Words) !void {
    var a: cuda.Args = .{};
    a.add(dst);
    a.add(int(n));
    a.add(words);
    try go(k.put_words, s, .{ .x = 1 }, 128, &a);
}

test "ops: the argument structs keep ops.cu's layouts" {
    try std.testing.expectEqual(@as(usize, 8 * 8 + 8 + 8 + 3 * 8 + 8 + 6 * 4), @sizeOf(MixArgs));
    try std.testing.expectEqual(@as(usize, 9 * 8 + 2 * 8 + 7 * 4 + 4), @sizeOf(ConvArgs));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(RewindSite));
    try std.testing.expectEqual(@as(usize, 6 * 8 + 6 * 4), @sizeOf(GateUpArgs));
}

test {
    _ = @import("ops_test.zig");
}

/// The certified lazy head's device state: an FP8 proxy of the head, each proxy row's error bound, the candidates.
pub const Lazy = struct {
    proxy: cuda.qmmf.Weight,
    delta: u64, // fp32 [V]: || W_v - proxy_v ||_2, rounded up
    xn: u64, // fp32 [rows]
    lg: u64, // u32 [rows][3]: the bounds' ordered bits (floor, race ceiling, race winner's floor)
    info: u64, // Row [rows]: each head row's draw
    count: u64, // i32 [rows]
    cand: u64, // i32 [rows][cands]
    /// A head row's draw (ops.cu ink2z::Info): mode 0 greedy, 1 sample.cu's keyed race (top_p 1, top_k off).
    pub const Row = extern struct { seed: u64 = 0, inv_t: f32 = 1, min_log: f32 = -std.math.inf(f32), pos: u32 = 0, mode: u32 = 0 };
    pub const cands = 256;
    pub const max_rows = 64; // buffers for a whole round (mm.round_rows); TF_INKLING2_LAZY_ROWS picks the limit
    // the bound's slack past delta ||x||: the proxy's and the head's fp32 sums (relative to ||x||), and rounding
    pub const eps: f32 = 4e-3;
    pub const abs: f32 = 0.02;
};

/// delta for every head row: the packed bf16 head against its FP8 copy (codes at e4m3(w inv), dequantized x scale).
pub fn lazyDelta(k: *const Kernels, s: cuda.Stream, head: cuda.qmmf.Weight, inv: f32, scale: f32, delta: u64) !void {
    var a: cuda.Args = .{};
    a.add(head.codes);
    a.add(int(head.k));
    a.add(int(head.n));
    a.add(inv);
    a.add(scale);
    a.add(delta);
    try go(k.lazy[0], s, .{ .x = @intCast((head.n + 7) / 8) }, 256, &a);
}

/// Rows [0, rows) of x (bf16, h wide) through the lazy head into `logits` (fp32 [rows][V], capped): the proxy's
/// candidates exact, every other logit -inf; a row past `cands` candidates gets every logit exact.
pub fn lazyHead(k: *const Kernels, lane: cuda.qmmf.Lane, s: cuda.Stream, z: Lazy, head: cuda.qmmf.Weight, x: u64, h: usize, rows: usize, logits: u64, part: u64, cap: f32) !void {
    const v: usize = head.n;
    {
        var a: cuda.Args = .{};
        for ([_]u64{x}) |p| a.add(p);
        a.add(int(h));
        a.add(int(h));
        for ([_]u64{ z.xn, z.lg, z.count }) |p| a.add(p);
        try go(k.lazy[1], s, .{ .x = @intCast(rows) }, 256, &a);
    }
    try lane.matmulOut(s, x, h, rows, z.proxy, logits, .f32, part);
    {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, z.delta, z.xn }) |p| a.add(p);
        a.add(int(v));
        a.add(Lazy.eps);
        a.add(Lazy.abs);
        a.add(z.lg);
        a.add(z.info);
        a.add(cap);
        try go(k.lazy[2], s, .{ .x = 96, .y = @intCast(rows) }, 256, &a);
        try go(k.lazy[7], s, .{ .x = 96, .y = @intCast(rows) }, 256, &a);
    }
    {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, z.delta, z.xn }) |p| a.add(p);
        a.add(int(v));
        a.add(Lazy.eps);
        a.add(Lazy.abs);
        a.add(z.lg);
        a.add(z.cand);
        a.add(int(Lazy.cands));
        a.add(z.count);
        a.add(z.info);
        a.add(cap);
        try go(k.lazy[3], s, .{ .x = 96, .y = @intCast(rows) }, 256, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i64, @intCast(rows * v)));
        try go(k.lazy[4], s, .{ .x = 384 }, 256, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(h));
        a.add(head.codes);
        a.add(int(h));
        a.add(int(v));
        a.add(head.scale);
        a.add(z.cand);
        a.add(int(Lazy.cands));
        a.add(z.count);
        a.add(logits);
        a.add(cap);
        try go(k.lazy[5], s, .{ .x = Lazy.cands / 32, .y = @intCast(rows) }, 128, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(h));
        a.add(head.codes);
        a.add(int(h));
        a.add(int(v));
        a.add(head.scale);
        a.add(int(Lazy.cands));
        a.add(z.count);
        a.add(int(rows));
        a.add(logits);
        a.add(cap);
        try go(k.lazy[6], s, .{ .x = 192 }, 128, &a);
    }
}

/// Structured output, greedy: each of `rows` fp32 rows' first maximum among the tokens `bits` allows (`words` u32
/// a row), into `out` (u32 a row; 0xFFFFFFFF when no allowed token has a usable score).
pub fn maskedArgmax(k: *const Kernels, s: cuda.Stream, logits: u64, vocab: usize, bits: u64, words: usize, rows: usize, out: u64) !void {
    var args: cuda.Args = .{};
    args.add(logits);
    args.add(@as(u32, @intCast(vocab)));
    args.add(bits);
    args.add(@as(u32, @intCast(words)));
    args.add(out);
    try go(k.masked_argmax, s, .{ .x = @intCast(rows) }, 1024, &args);
}

/// maskedArgmax over bf16 rows (a sampled draw's own logits): the guard for rows with no usable score.
pub fn maskedArgmax16(k: *const Kernels, s: cuda.Stream, logits16: u64, vocab: usize, bits: u64, words: usize, rows: usize, out: u64) !void {
    var args: cuda.Args = .{};
    args.add(logits16);
    args.add(@as(u32, @intCast(vocab)));
    args.add(bits);
    args.add(@as(u32, @intCast(words)));
    args.add(out);
    try go(k.masked_argmax16, s, .{ .x = @intCast(rows) }, 1024, &args);
}

/// Structured output, sampled: each of `rows` bf16 rows' NaN logits to -inf.
pub fn grammarScrub(k: *const Kernels, s: cuda.Stream, logits16: u64, vocab: usize, rows: usize) !void {
    var args: cuda.Args = .{};
    args.add(logits16);
    args.add(@as(u32, @intCast(vocab)));
    try go(k.grammar_scrub, s, .{ .x = @intCast(rows) }, 256, &args);
}

/// Structured output, sampled: rows whose `guard` (maskedArgmax) is 0xFFFFFFFF get one finite bf16 logit before
/// the draw (`after` false); after it (`after` true) a row's token becomes 0xFFFFFFFF when it had nothing usable or
/// its bits do not allow the drawn token.
pub fn grammarGuard(k: *const Kernels, s: cuda.Stream, logits16: u64, vocab: usize, guard: u64, tok: u64, rows: usize, after: bool, bits: u64, words: usize) !void {
    var args: cuda.Args = .{};
    args.add(logits16);
    args.add(@as(u32, @intCast(vocab)));
    args.add(guard);
    args.add(tok);
    args.add(@as(u32, @intCast(rows)));
    args.add(@as(u32, @intFromBool(after)));
    args.add(bits);
    args.add(@as(u32, @intCast(words)));
    try go(k.grammar_guard, s, .{ .x = @intCast((rows + 31) / 32) }, 32, &args);
}

/// ops.cu SplitCols: up to six parts of a fused row, each copied to its own plane.
pub const SplitCols = extern struct { dst: [6]u64 = @splat(0), off: [6]i32 = @splat(0), w: [6]i32 = @splat(0), n: i32 = 0 };

pub fn splitCols(k: *const Kernels, s: cuda.Stream, src: u64, ld: usize, rows: usize, a: SplitCols) !void {
    var args: cuda.Args = .{};
    args.add(src);
    args.add(int(ld));
    args.add(a);
    try go(k.split_cols, s, .{ .x = @intCast(rows) }, 256, &args);
}

pub fn visEmbed(k: *const Kernels, s: cuda.Stream, patch_emb: u64, pos_tab: u64, cls_emb: u64, cls_pos: u64, out: u64, gh: usize, gw: usize, merge: usize, grid: usize, h: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ patch_emb, pos_tab, cls_emb, cls_pos, out }) |p| args.add(p);
    for ([_]usize{ gh, gw, merge, grid, h }) |n| args.add(int(n));
    try go(k.vis_embed, s, .{ .x = @intCast(gh * gw + 1) }, 256, &args);
}

pub fn visPackQkvRope(k: *const Kernels, s: cuda.Stream, q: u64, kk: u64, v: u64, qkv: u64, gh: usize, gw: usize, merge: usize, heads: usize, hd: usize, theta: f32) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ q, kk, v, qkv }) |p| args.add(p);
    for ([_]usize{ gh, gw, merge, heads, hd }) |n| args.add(int(n));
    args.add(theta);
    try go(k.vis_pack_qkv_rope, s, .{ .x = @intCast(gh * gw + 1) }, 256, &args);
}

pub fn visLsAdd(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, gamma: u64, rows: usize, h: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ x, y, gamma }) |p| args.add(p);
    args.add(int(h));
    try go(k.vis_ls_add, s, .{ .x = @intCast(rows) }, 256, &args);
}

pub fn visSplice(k: *const Kernels, s: cuda.Stream, dst: u64, src: u64, map: u64, rows: usize, h: usize) !void {
    if (rows == 0) return;
    var args: cuda.Args = .{};
    for ([_]u64{ dst, src, map }) |p| args.add(p);
    args.add(int(h));
    try go(k.vis_splice, s, .{ .x = @intCast(rows) }, 256, &args);
}


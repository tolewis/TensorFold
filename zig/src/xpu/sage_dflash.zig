//! DFlash (v1) block drafter for the XPU Sage runner: z-lab's Qwen3.6-35B-A3B-DFlash, its forward as in z-lab/dflash
//! (dflash/model.py, model_mlx.py). The target's residual stream after 8 of its layers (the taps) is the context: fc
//! and its norm make a row of it, and each layer keeps that row's K and V. One forward of a block (the pending token,
//! then mask tokens) drafts the tokens after the pending one: row j's output, normed, through the target's head. Six
//! Qwen3 decoder layers: a sliding layer sees the context rows within its window and the block rows up to its own; the
//! full-attention layer sees every block row and the context rows of a ring of `full` positions. The projections are
//! affine 4/64 on the target's GEMM, quantized at load as the 27B's DFlash2 drafter's are (drafter.zig); the embedding
//! and the head are the target's. DFlash2 adds a grouped dynamic convolution and a candidate selector: not here.
const std = @import("std");
const ze = @import("ze");
const kb = @import("xpu_qwen27_kernels");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const loader = @import("load.zig");
const d27 = @import("drafter.zig");
const sage = @import("sage.zig");
const q = @import("qwen27.zig"); // the profiler state (XPU_PROFILE)

const Dev = dev.Dev;
const Proj = loader.Proj;
const off = dev.off;
const alloc = std.heap.c_allocator;

const D = 2048; // the target's hidden size and the drafter's (sage_drafter: -DD)
pub const NTAP = 8; // tapped target layers
pub const TAPS = NTAP * D; // a row of taps (-DTAPS)
const LAYERS = 6;
const NH = 32;
const NKV = 8;
const HD = 128;
const FF = 6144; // (-DFF)
const GS = 64;
const EPS: f32 = 1e-6;
/// Block rows at most: the target verifies a block in one window.
pub const MAXL = sage.WINDOW;
/// Rows of a batched block pass or a batched absorb at most (one verify window's).
const MAXB = sage.WINDOW;
/// Ring positions past a window: a block's rows and a margin (drafter.zig CAP).
const SLACK = MAXL + 17;
/// Context rows the full-attention layer sees by default (XPU_DFLASH_FULL): the checkpoint was trained on sequences of
/// 40k; 32k keeps a ring of 134 MB a slot.
pub const default_full: u32 = 32767;

const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    qkv: Proj, // q | k | v
    kv: Proj, // k | v: a context row's
    qn: *anyopaque,
    kn: *anyopaque,
    o: Proj,
    gu: Proj, // gate | up
    down: Proj,
    win: u32, // context rows a block row sees (sliding_window − 1, or `full`)
    cap: u32, // the ring's positions: win + SLACK
    causal: bool, // a block row sees the block rows up to its own (sliding layers)
};

/// A slot's context: each layer's K, V (bf16 [NKV, cap, HD]) in a ring; target rows `start` .. `end` absorbed.
const Ctx = struct {
    kc: [LAYERS]*anyopaque,
    vc: [LAYERS]*anyopaque,
    start: u32 = 0,
    end: u32 = 0,
};

/// A block of a batched pass: a slot, its pending token at position `at` and its rows (the pending one, then mask rows).
pub const Block = struct { slot: u32, pending: i32, rows: u32, at: u32 };
/// Rows to absorb into a slot's context: `n` rows of taps (bf16 [n, TAPS]) at `taps`, their first at position `at`.
pub const Absorb = struct { slot: u32, taps: *anyopaque, n: u32, at: u32 };

/// The checkpoint's config.json, the fields the forward reads.
const Config = struct {
    architectures: []const []const u8 = &.{},
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    sliding_window: ?u32 = null,
    layer_types: []const []const u8 = &.{},
    rope_theta: ?f64 = null,
    rope_parameters: ?struct { rope_theta: f64 = 1e7 } = null,
    dflash_config: struct { block_size: u32 = 16, mask_token_id: u32, target_layer_ids: []const u32 },
};

pub const Drafter = struct {
    r: *sage.Runner,
    k: struct { dprep: ze.H, dattn: ze.H, rowsums_rows: ze.H, taps_in: ze.H, to_res: ze.H, quantize4: ze.H, dswiglu: ze.H, tap: ze.H },
    layers: [LAYERS]Layer,
    fc: Proj,
    hidden_norm: *anyopaque,
    norm: *anyopaque,
    invf: *anyopaque,
    mask: i32, // the mask token's id
    block: u32, // the checkpoint's block rows (its training block)
    tap_of: [256]i8 = @splat(-1), // a target layer's place among the taps (-1: not tapped)
    ctxs: [sage.MAX_SLOTS]Ctx = undefined,
    rows_host: [MAXB][4]i32 = undefined, // a batched pass's rows: position, context start, block start, block
    ptrs_host: [LAYERS][sage.MAX_SLOTS][2]u64 = undefined, // and each block's K, V rings by layer
    rows_dev: *anyopaque,
    ptrs_dev: *anyopaque,
    ctx_bytes: usize = 0, // the rings of all slots
    weight_bytes: usize = 0,
    // buffers
    taps: *anyopaque, // bf16 [CHUNK, TAPS]: the target's taps of its last forward, a row each (Runner.forward)
    tapsbuf: *anyopaque, // bf16 [MAXB, TAPS]: a batched absorb's rows gathered
    x: *anyopaque, // fp32 [CHUNK, D]
    h: *anyopaque, // fp16 [CHUNK, TAPS]
    ahi: *anyopaque,
    alo: *anyopaque,
    p: *anyopaque, // fp16 [CHUNK, kv.n] or [MAXB, gu.n]
    r16: *anyopaque, // fp16 [CHUNK, D]
    h2: *anyopaque, // fp16 [MAXB, FF]
    qb: *anyopaque, // bf16 [MAXB, NH, HD]
    o: *anyopaque, // fp16 [MAXB, NH HD]
    ids: *anyopaque, // i32 [MAXB]
    pscr: *anyopaque,

    /// bf16 weights [n_i, k] stacked on N, quantized to affine 4/64 on the device, then repacked (drafter.zig).
    fn quantized(dr: *Drafter, c: *const dev.Checkpoint, names: []const []const u8, wide: bool, stage: *anyopaque) !Proj {
        const d = &dr.r.d;
        var parts: [4]d27.Mlx = undefined;
        var k: u32 = 0;
        var at: usize = 0;
        for (names, 0..) |nm, i| {
            const t = try c.get(nm);
            if (t.rank != 2) return error.UnexpectedShape;
            const n: u32 = @intCast(t.shape[0]);
            k = @intCast(t.shape[1]);
            const raw = off(stage, at);
            try d.copyIn(raw, t.bytes);
            at += t.bytes.len;
            const mw = off(stage, at);
            at += @as(usize, n) * (k / 8) * 4;
            const ms = off(stage, at);
            at += @as(usize, n) * (k / GS) * 2;
            const mb = off(stage, at);
            at += @as(usize, n) * (k / GS) * 2;
            at = (at + 4095) / 4096 * 4096;
            if (at > stage_bytes) return error.StageTooSmall;
            try d.launch(dr.k.quantize4, .{ .x = (n + 15) / 16, .y = k / GS }, .{ raw, @as(i32, @intCast(n)), @as(i32, @intCast(k)), mw, ms, mb });
            parts[i] = .{ .w = mw, .s = ms, .b = mb, .n = n };
        }
        try d.sync();
        const before = d.allocated;
        const p = try d27.Drafter.pack(d, parts[0..names.len], k, wide);
        dr.weight_bytes += d.allocated - before;
        return p;
    }

    const stage_bytes: usize = 160 << 20;

    /// The drafter in `dir` (config.json, model.safetensors bf16) on `r`'s card, with every slot's context rings: the
    /// full-attention layer's ring holds `full` context rows. `block` (0: the checkpoint's) is not checked here.
    pub fn init(r: *sage.Runner, dir: []const u8, full: u32) !*Drafter {
        const d = &r.d;
        const bytes = try sage.readFile(try std.fmt.allocPrint(alloc, "{s}/config.json", .{dir}));
        defer alloc.free(bytes);
        const parsed = std.json.parseFromSlice(Config, alloc, bytes, .{ .ignore_unknown_fields = true }) catch return error.BadDrafterConfig;
        defer parsed.deinit();
        const cf = parsed.value;
        for (cf.architectures) |a| if (!std.mem.eql(u8, a, "DFlashDraftModel")) return error.DrafterNotDFlashV1; // DFlash2: conv and selector
        const theta = if (cf.rope_parameters) |rp| rp.rope_theta else cf.rope_theta orelse 1e7;
        const dc = cf.dflash_config;
        if (cf.hidden_size != D or cf.num_hidden_layers != LAYERS or cf.num_attention_heads != NH or cf.num_key_value_heads != NKV or cf.head_dim != HD or cf.intermediate_size != FF or r.z.s.hidden != D) return error.DrafterShapeNotBuilt;
        if (dc.target_layer_ids.len != NTAP or cf.layer_types.len != LAYERS or dc.block_size < 2 or dc.block_size > MAXL) return error.DrafterShapeNotBuilt;
        const dr = try alloc.create(Drafter);
        errdefer alloc.destroy(dr);
        const m = try d.g.module(&kb.sage_drafter);
        const kern = struct {
            fn f(dd: *Dev, mm: ze.H, nm: [*:0]const u8, local: u32) !ze.H {
                return dd.kernel(mm, nm, local);
            }
        }.f;
        const C: usize = sage.CHUNK;
        dr.* = .{
            .r = r,
            .k = .{ .dprep = try kern(d, m, "dprep", 16), .dattn = try kern(d, m, "dattn_win", 16), .rowsums_rows = try kern(d, m, "rowsums_rows", 256), .taps_in = try kern(d, m, "taps_in", 256), .to_res = try kern(d, m, "to_res", 256), .quantize4 = try kern(d, m, "quantize4", 16), .dswiglu = try kern(d, m, "dswiglu", 256), .tap = try kern(d, m, "tap_rows", 256) },
            .layers = undefined,
            .fc = undefined,
            .hidden_norm = undefined,
            .norm = undefined,
            .invf = undefined,
            .mask = @intCast(dc.mask_token_id),
            .block = dc.block_size,
            .rows_dev = try d.alloc_(MAXB * 16),
            .ptrs_dev = try d.alloc_(LAYERS * sage.MAX_SLOTS * 16),
            .taps = try d.alloc_(C * TAPS * 2),
            .tapsbuf = try d.alloc_(MAXB * TAPS * 2),
            .x = try d.alloc_(C * D * 4),
            .h = try d.alloc_(C * TAPS * 2),
            .ahi = try d.alloc_(C * (TAPS / GS) * 2),
            .alo = try d.alloc_(C * (TAPS / GS) * 2),
            .p = try d.alloc_(@as(usize, @max(C * 2 * NKV * HD, MAXB * 2 * FF)) * 2),
            .r16 = try d.alloc_(C * D * 2),
            .h2 = try d.alloc_(MAXB * FF * 2),
            .qb = try d.alloc_(MAXB * NH * HD * 2),
            .o = try d.alloc_(MAXB * NH * HD * 2),
            .ids = try d.alloc_(MAXB * 4),
            .pscr = try d.alloc_(C * D * 4),
        };
        if (q.prof.on) { // XPU_PROFILE: the drafter's kernels by name, after the runner's
            var at: usize = 0;
            while (q.prof.names[at].k != null) at += 1;
            inline for (@typeInfo(@TypeOf(dr.k)).@"struct".field_names, 0..) |n, i| q.prof.names[at + i] = .{ .k = @field(dr.k, n), .name = "df_" ++ n };
        }
        for (dc.target_layer_ids, 0..) |l, i| {
            if (l >= r.z.s.layers or l >= dr.tap_of.len) return error.DrafterShapeNotBuilt;
            dr.tap_of[l] = @intCast(i);
        }
        var c: dev.Checkpoint = .{};
        defer c.deinit();
        try c.addShard(try std.fmt.allocPrintSentinel(alloc, "{s}/model.safetensors", .{dir}, 0));
        const stage = try d.alloc_(stage_bytes);
        defer {
            d.g.free(stage);
            d.allocated -= stage_bytes;
        }
        const bf = struct {
            fn f(dx: *Drafter, cc: *const dev.Checkpoint, nm: []const u8) !*anyopaque {
                const t = try cc.get(nm);
                dx.weight_bytes += t.bytes.len;
                return dx.r.d.upload(t.bytes);
            }
        }.f;
        const nm = struct {
            fn f(l: usize, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "layers.{d}.{s}", .{ l, s }) catch @panic("out of memory");
            }
        }.f;
        dr.fc = try dr.quantized(&c, &.{"fc.weight"}, false, stage);
        if (dr.fc.k != TAPS) return error.DrafterShapeNotBuilt;
        dr.hidden_norm = try bf(dr, &c, "hidden_norm.weight");
        dr.norm = try bf(dr, &c, "norm.weight");
        for (&dr.layers, 0..) |*l, i| {
            const sliding = std.mem.eql(u8, cf.layer_types[i], "sliding_attention");
            if (!sliding and !std.mem.eql(u8, cf.layer_types[i], "full_attention")) return error.DrafterShapeNotBuilt;
            const win: u32 = if (sliding) (cf.sliding_window orelse return error.DrafterShapeNotBuilt) - 1 else full;
            l.* = .{
                .in_norm = try bf(dr, &c, nm(i, "input_layernorm.weight")),
                .post_norm = try bf(dr, &c, nm(i, "post_attention_layernorm.weight")),
                .qkv = try dr.quantized(&c, &.{ nm(i, "self_attn.q_proj.weight"), nm(i, "self_attn.k_proj.weight"), nm(i, "self_attn.v_proj.weight") }, false, stage),
                .kv = try dr.quantized(&c, &.{ nm(i, "self_attn.k_proj.weight"), nm(i, "self_attn.v_proj.weight") }, false, stage),
                .qn = try bf(dr, &c, nm(i, "self_attn.q_norm.weight")),
                .kn = try bf(dr, &c, nm(i, "self_attn.k_norm.weight")),
                .o = try dr.quantized(&c, &.{nm(i, "self_attn.o_proj.weight")}, false, stage),
                .gu = try dr.quantized(&c, &.{ nm(i, "mlp.gate_proj.weight"), nm(i, "mlp.up_proj.weight") }, true, stage),
                .down = try dr.quantized(&c, &.{nm(i, "mlp.down_proj.weight")}, false, stage),
                .win = win,
                .cap = win + SLACK,
                .causal = sliding,
            };
        }
        var invf: [HD / 2]f32 = undefined;
        for (&invf, 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / HD));
        dr.invf = try d.upload(std.mem.sliceAsBytes(&invf));
        const before = d.allocated;
        for (dr.ctxs[0..r.slots]) |*x| {
            for (&x.kc, &x.vc, dr.layers) |*kc, *vc, l| {
                kc.* = try d.alloc_(@as(usize, NKV) * l.cap * HD * 2);
                vc.* = try d.alloc_(@as(usize, NKV) * l.cap * HD * 2);
            }
            x.start = 0;
            x.end = 0;
        }
        dr.ctx_bytes = d.allocated - before;
        try d.sync();
        return dr;
    }

    pub fn deinit(dr: *Drafter) void {
        dr.r.d.settle(); // the host tables may still be in a queued copy
        alloc.destroy(dr);
    }

    /// Context bytes a position of a slot's rings takes: K and V of every layer.
    pub fn posBytes() usize {
        return LAYERS * 2 * NKV * HD * 2;
    }

    /// The full-attention layer's window (context rows).
    pub fn fullWindow(dr: *const Drafter) u32 {
        var w: u32 = 0;
        for (dr.layers) |l| w = @max(w, l.win);
        return w;
    }

    /// Launches the tap of target layer `layer` over T rows when it is tapped: bf16(x + r) into taps.
    pub fn tapLayer(dr: *Drafter, layer: usize, x: *anyopaque, rr: *anyopaque, T: u32) !void {
        const i = dr.tap_of[layer];
        if (i < 0) return;
        try dr.r.d.launch(dr.k.tap, .{ .x = D / 256, .y = T }, .{ x, rr, dr.taps, @as(i32, i) });
    }

    /// A fresh sequence in `slot`: no context rows.
    pub fn reset(dr: *Drafter, slot: u32) void {
        dr.restart(slot, 0);
    }

    /// `slot`'s context from position `at` on (the target rows before it unseen): absorb appends from there.
    pub fn restart(dr: *Drafter, slot: u32, at: u32) void {
        dr.ctxs[slot].start = at;
        dr.ctxs[slot].end = at;
    }

    /// Appends `n` target rows (taps bf16 [n, TAPS] at `taps`) at position `at` to `slot`'s context: fc and its norm,
    /// then each layer's k and v into its ring. A context that does not end at `at` starts there.
    pub fn absorb(dr: *Drafter, slot: u32, taps: *anyopaque, n: u32, at: u32) !void {
        if (n == 0) return;
        const d = &dr.r.d;
        const c = &dr.ctxs[slot];
        if (c.end != at) dr.restart(slot, at);
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        // of more rows than the widest window, the first are never read
        var done: u32 = n -| dr.fullWindow();
        if (done > 0) c.start = @max(c.start, at + done);
        while (done < n) {
            const rows: u32 = @min(n - done, sage.CHUNK);
            try d.launch(dr.k.taps_in, .{ .x = rows }, .{ off(taps, @as(usize, done) * TAPS * 2), dr.h, dr.ahi, dr.alo });
            try fwd.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.fc, dr.r16, rows);
            try d.launch(dr.k.to_res, .{ .x = rows * D / 256 }, .{ dr.r16, dr.x });
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = rows }, .{ dr.x, dr.r16, dr.hidden_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), EPS, zero, @as(i32, GS), nul, nul, zero, zero });
            for (&dr.layers, c.kc, c.vc) |*l, kc, vc| {
                try fwd.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kv, dr.p, rows);
                try d.launch(dr.k.dprep, .{ .x = rows, .y = 2 * NKV }, .{ dr.p, @as(i32, @intCast(l.kv.n)), @as(i32, -1), l.qn, l.kn, dr.invf, dr.qb, kc, vc, @as(i32, @intCast(l.cap)), @as(i32, @intCast(at + done)), EPS, nul, nul });
            }
            done += rows;
        }
        c.end = at + n; // no wait: the next block queues behind on the in-order list
    }

    /// Each item's rows into its slot's context in one pass (rows of a verify window, MAXB at most in all).
    pub fn absorbMulti(dr: *Drafter, items: []const Absorb) !void {
        var live: [sage.MAX_SLOTS]Absorb = undefined;
        var n: usize = 0;
        for (items) |it| if (it.n > 0) {
            live[n] = it;
            n += 1;
        };
        if (n == 0) return;
        if (n == 1) return dr.absorb(live[0].slot, live[0].taps, live[0].n, live[0].at);
        const d = &dr.r.d;
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        var R: u32 = 0;
        for (live[0..n], 0..) |it, b| {
            if (R + it.n > MAXB) return error.TooManyRows;
            const c = &dr.ctxs[it.slot];
            if (c.end != it.at) dr.restart(it.slot, it.at);
            try dev.copy(d, off(dr.tapsbuf, @as(usize, R) * TAPS * 2), it.taps, @as(usize, it.n) * TAPS * 2);
            for (0..it.n) |j| dr.rows_host[R + j] = .{ @intCast(c.end + j), 0, 0, @intCast(b) };
            for (0..LAYERS) |li| dr.ptrs_host[li][b] = .{ @intFromPtr(c.kc[li]), @intFromPtr(c.vc[li]) };
            R += it.n;
        }
        try dr.tables(R);
        try d.launch(dr.k.taps_in, .{ .x = R }, .{ dr.tapsbuf, dr.h, dr.ahi, dr.alo });
        try fwd.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.fc, dr.r16, R);
        try d.launch(dr.k.to_res, .{ .x = R * D / 256 }, .{ dr.r16, dr.x });
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = R }, .{ dr.x, dr.r16, dr.hidden_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), EPS, zero, @as(i32, GS), nul, nul, zero, zero });
        for (&dr.layers, 0..) |*l, li| {
            const kc0: *anyopaque = @ptrFromInt(dr.ptrs_host[li][0][0]);
            const vc0: *anyopaque = @ptrFromInt(dr.ptrs_host[li][0][1]);
            try fwd.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kv, dr.p, R);
            try d.launch(dr.k.dprep, .{ .x = R, .y = 2 * NKV }, .{ dr.p, @as(i32, @intCast(l.kv.n)), @as(i32, -1), l.qn, l.kn, dr.invf, dr.qb, kc0, vc0, @as(i32, @intCast(l.cap)), zero, EPS, dr.rows_dev, off(dr.ptrs_dev, li * sage.MAX_SLOTS * 16) });
        }
        for (live[0..n]) |it| dr.ctxs[it.slot].end += it.n;
        try d.sync(); // the tables are read from the host
    }

    /// The batched pass's row and ring tables to the device.
    fn tables(dr: *Drafter, rows: u32) !void {
        const d = &dr.r.d;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.rows_dev, &dr.rows_host, @as(usize, rows) * 16, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.ptrs_dev, &dr.ptrs_host, LAYERS * sage.MAX_SLOTS * 16, null, 0, null));
    }

    /// One block a slot in one pass: block i's drafts (its rows − 1, greedy, a chain) to out[i]. A slot's context that
    /// does not end at its pending token's position starts there (none of its rows seen).
    pub fn propose(dr: *Drafter, items: []const Block, out: [][sage.WINDOW]i32) !void {
        const r = dr.r;
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const S: u32 = @intCast(items.len);
        if (S == 0 or S > sage.MAX_SLOTS or out.len < S) return error.BadWindow;
        var R: u32 = 0;
        var tokens: [MAXB]i32 = undefined;
        var first: [sage.MAX_SLOTS]u32 = undefined;
        for (items, 0..) |it, b| {
            if (it.rows < 2 or it.rows > MAXL or R + it.rows > MAXB) return error.TooManyRows;
            const c = &dr.ctxs[it.slot];
            if (c.end != it.at) dr.restart(it.slot, it.at);
            first[b] = R;
            tokens[R] = it.pending;
            for (1..it.rows) |i| tokens[R + i] = dr.mask;
            for (0..it.rows) |i| dr.rows_host[R + i] = .{ @intCast(c.end + i), @intCast(c.start), @intCast(c.end), @intCast(b) };
            for (0..LAYERS) |li| dr.ptrs_host[li][b] = .{ @intFromPtr(c.kc[li]), @intFromPtr(c.vc[li]) };
            R += it.rows;
        }
        // one block (S = 1) runs with no tables, the kernels at its rows' scalars
        const one = S == 1;
        const c0 = &dr.ctxs[items[0].slot];
        const L: i32 = @intCast(items[0].rows);
        const end_: i32 = @intCast(c0.end);
        const start_: i32 = @intCast(c0.start);
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        const rows_t: ?*anyopaque = if (one) nul else dr.rows_dev;
        if (!one) try dr.tables(R);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.ids, &tokens, R * 4, null, 0, null));
        errdefer d.settle(); // tokens and the tables live in this frame: an error returns only once the queue is idle
        if (m.embed_h) |e|
            try d.launch(d.k.embed_dense, .{ .x = R }, .{ dr.ids, e, dr.x, @as(i32, D) })
        else
            try d.launch(d.k.embed, .{ .x = R }, .{ dr.ids, m.embed_w, m.embed_s, m.embed_b, dr.x, @as(i32, D) });
        for (&dr.layers, c0.kc, c0.vc, 0..) |*l, kc, vc, li| {
            const ptrs_t: ?*anyopaque = if (one) nul else off(dr.ptrs_dev, li * sage.MAX_SLOTS * 16);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = R }, .{ dr.x, dr.r16, l.in_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), EPS, @as(i32, if (li == 0) 0 else 1), @as(i32, GS), nul, nul, zero, zero });
            try fwd.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.qkv, dr.p, R);
            try d.launch(dr.k.dprep, .{ .x = R, .y = NH + 2 * NKV }, .{ dr.p, @as(i32, @intCast(l.qkv.n)), zero, l.qn, l.kn, dr.invf, dr.qb, kc, vc, @as(i32, @intCast(l.cap)), end_, EPS, rows_t, ptrs_t });
            try d.launch(dr.k.dattn, .{ .x = R, .y = NKV }, .{ dr.qb, kc, vc, @as(i32, @intCast(l.cap)), start_, end_, L, @as(i32, @intCast(l.win)), @as(i32, @intFromBool(l.causal)), @as(f32, 1.0 / @sqrt(@as(f32, HD))), dr.o, rows_t, ptrs_t });
            try d.launch(dr.k.rowsums_rows, .{ .x = R }, .{ dr.o, @as(i32, NH * HD), dr.ahi, dr.alo });
            try fwd.gemm(d, dr.pscr, dr.o, dr.ahi, dr.alo, l.o, dr.r16, R);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = R }, .{ dr.x, dr.r16, l.post_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), EPS, @as(i32, 1), @as(i32, GS), nul, nul, zero, zero });
            try fwd.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.gu, dr.p, R);
            try d.launch(dr.k.dswiglu, .{ .x = R, .y = FF / 1024 }, .{ dr.p, @as(i32, @intCast(l.gu.n)), dr.h2, dr.ahi, dr.alo });
            try fwd.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.down, dr.r16, R);
        }
        // every row's final norm and the target's head (a block's row 0 is read by no draft), greedy tokens
        const hgs: i32 = @intCast(m.head.gs);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = R }, .{ dr.x, dr.r16, dr.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), hgs, nul, nul, zero, zero });
        try r.headRows(a.h, a.ahi, a.alo, R);
        try d.launch(d.k.argmax_part, .{ .x = 256, .y = R }, .{ a.logits, @as(i32, @intCast(r.z.s.vocab)), a.pv, a.pi });
        try d.launch(d.k.argmax, .{ .x = R }, .{ a.pv, a.pi, a.tok });
        var tok: [MAXB]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &tok, a.tok, R * 4, null, 0, null));
        try d.sync();
        for (items, 0..) |it, b| @memcpy(out[b][0 .. it.rows - 1], tok[first[b] + 1 ..][0 .. it.rows - 1]);
    }
};

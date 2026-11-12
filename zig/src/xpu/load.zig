//! Qwen3.8-27B weights onto the device: int4 projections (MLX affine 4/64 or GPTQ 4/128) and the layer table.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const x3 = @import("qwen27_exl3.zig");
const mlx = @import("mlx_tp.zig");
const Checkpoint = dev.Checkpoint;
const D = q.D;
const Dev = dev.Dev;
const GDN_IN = q.GDN_IN;
const HV = q.GDN_V_HEADS;
const LAYERS = q.LAYERS;
const STASH = q.STASH;
const alloc = q.alloc;
const isAttention = q.isAttention;
const kvBytes = fwd.kvBytes;
const off = dev.off;

/// An int4 projection in the GEMM layout (words, fp16 scales, biases); `wide` picks the kernel family, `ilv` gate|up.
/// `x3`: an EXL3 projection (qwen27_exl3.zig); the int4 fields are then unused.
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool, gs: u32 = 64, ilv: bool = false, x3: ?*x3.Proj = null };

pub const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // gdn.in or attn.qkv
    mixer_out: Proj, // out_proj or o_proj
    gu: Proj,
    down: Proj,
    // GDN
    ab: ?*anyopaque = null, // GPTQ: the a and b gates, fp16 [96, 5120] (dense, as stored), into P columns 16 384 …
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [48, 128, 128]
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, 10240], ping-pong
    conv_cur: u1 = 0,
    // a verified block's GDN inputs and its entry state, to replay the kept rows (Runner.commit)
    rec_save: ?*anyopaque = null,
    stash: ?*anyopaque = null, // per-row stash: q, k, v, g, β and the qkv rows
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [MAX_CTX, 4, 256]
    vc: ?*anyopaque = null,
    // rec, conv, conv_cur, kc and vc above are the selected slot's (Runner.select); each slot's own are kept here
    slot_rec: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_save: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_conv: [MAX_SLOTS][2]?*anyopaque = @splat(.{ null, null }),
    slot_cur: [MAX_SLOTS]u1 = @splat(0),
    slot_kc: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_vc: [MAX_SLOTS]?*anyopaque = @splat(null),
    // K1 (XPU_K1) and the pool (lane p21-pool): the layer's K/V page pool as Sage's allocPool ([pool, PAGE, AKV, 256])
    // and each slot's page table (u32 page ids, one an entry). Unused while `pt` is null.
    pool: u32 = 0, // pages the pool holds
    pt: ?*anyopaque = null, // the selected slot's page table
    slot_pt: [MAX_SLOTS]?*anyopaque = @splat(null),
};

/// Sequences a runner holds at most (one a stream of the lane host).
pub const MAX_SLOTS = 16;

pub const Model = struct {
    embed_w: *anyopaque, // MLX: words [V, D/8] ; GPTQ: the fp16 table [V, D]
    embed_s: ?*anyopaque = null, // MLX only
    embed_b: ?*anyopaque = null,
    layers: [LAYERS]Layer,
    norm: *anyopaque,
    head: Proj,
    invf: *anyopaque,
    /// EXL3 checkpoints: the projections' pipelines, regions and plans (bound after Acts.init).
    exl3: ?*x3.Weights = null,
};

pub fn bf16Tensor(d: *Dev, c: *const Checkpoint, name: []const u8) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    return d.upload(t.bytes);
}

/// Fused projection from MLX parts stacked on N, repacked on the GPU through a raw staging area.
pub fn projection(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        n_total += @intCast(w.shape[0]);
        k = @intCast(w.shape[1] * 8);
    }
    // N padded to the GEMM tile: the padding columns have zero scales and biases, so they come out as zeros.
    const n_real = n_total;
    n_total = (n_total + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.s, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        const s = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        const b = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, p }));
        const n: u32 = @intCast(w.shape[0]);
        const sw = stage;
        const ss = off(stage, w.bytes.len);
        const sb = off(ss, s.bytes.len);
        try d.copyIn(sw, w.bytes);
        try d.copyIn(ss, s.bytes);
        try d.copyIn(sb, b.bytes);
        try d.launch(d.k.repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ sw, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// Fused projection from GPTQ symmetric 4-bit parts stacked on N, at least `min_n` columns (the extra ones zero).
pub fn projectionGptq(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, min_n: u32, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        n_total += @intCast(w.shape[1]);
        k = @intCast(w.shape[0] * 8);
    }
    const n_real = n_total;
    n_total = (@max(n_total, min_n) + 255) / 256 * 256;
    const g = q.GS;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / g) * n_total * 2), .bg = try d.alloc_(@as(usize, k / g) * n_total * 2), .n = n_total, .k = k, .wide = wide, .gs = g, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.w, @as(u32, 0), @as(usize, k / 8) * n_total * 4);
        try d.fill(out.s, @as(u16, 0), @as(usize, k / g) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / g) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        const sc = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        if (!std.mem.eql(u8, sc.dtype, "F16") or sc.shape[0] * g != k) return error.UnexpectedGptqLayout;
        const n: u32 = @intCast(w.shape[1]);
        const ss = off(stage, w.bytes.len);
        try d.copyIn(stage, w.bytes);
        try d.copyIn(ss, sc.bytes);
        try d.launch(d.k.repack_gptq, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ stage, ss, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, @intCast(g)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// An fp16 [N, K] weight quantized at load to MLX affine 4/64 in the GEMM layout, in row blocks via staging.
pub fn projectionQuantized(d: *Dev, c: *const Checkpoint, name: []const u8, wide: bool, stage: *anyopaque) !Proj {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n: u32 = @intCast(t.shape[0]);
    const k: u32 = @intCast(t.shape[1]);
    if (n % 256 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n * 4), .s = try d.alloc_(@as(usize, k / 64) * n * 2), .bg = try d.alloc_(@as(usize, k / 64) * n * 2), .n = n, .k = k, .wide = wide };
    const rows: u32 = 32768;
    var n0: u32 = 0;
    while (n0 < n) : (n0 += rows) {
        const r = @min(rows, n - n0);
        const raw = stage;
        const mw = off(raw, @as(usize, r) * k * 2);
        const ms = off(mw, @as(usize, r) * (k / 8) * 4);
        const mb = off(ms, @as(usize, r) * (k / 64) * 2);
        try d.copyIn(raw, t.bytes[@as(usize, n0) * k * 2 ..][0 .. @as(usize, r) * k * 2]);
        try d.launch(d.k.quantize_mlx, .{ .x = (r + 15) / 16, .y = k / 64 }, .{ raw, @as(i32, @intCast(r)), @as(i32, @intCast(k)), mw, ms, mb });
        try d.launch(d.k.repack, .{ .x = (r + 15) / 16, .y = k / 8 }, .{ mw, ms, mb, out.w, out.s, out.bg, @as(i32, @intCast(r)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)), @as(i32, 0) });
        try d.sync();
    }
    return out;
}

/// A small weight to bf16 on the device; `centred`: a GPTQ RMSNorm weight stored as gamma − 1 gets its 1 back.
pub fn vecTensor(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool) !*anyopaque {
    const t = try c.get(name);
    if (std.mem.eql(u8, t.dtype, "BF16")) return d.upload(t.bytes);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n = t.bytes.len / 2;
    const out = try alloc.alloc(u16, n);
    defer alloc.free(out);
    for (out, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
        var v: f32 = h;
        if (centred) v += 1.0;
        o.* = toBf16(v);
    }
    return d.upload(std.mem.sliceAsBytes(out));
}

/// fp32 to bf16 as core's bf16_round.bits rounds (NaN stays NaN); spelled out, the XPU 27B modules have no `core`.
fn toBf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// Qwen3.8-27B from a symmetric 4-bit GPTQ checkpoint (g128): int4 projections, fp16 embedding, head quantized to 4/64.
pub fn loadGptq(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    m.exl3 = null;
    const stage = try d.alloc_(800 << 20);
    const e = try c.get("model.embed_tokens.weight");
    if (!std.mem.eql(u8, e.dtype, "F16")) return error.UnexpectedDtype;
    m.embed_w = try d.alloc_(e.bytes.len);
    try d.copyIn(m.embed_w, e.bytes);
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try vecTensor(d, c, "model.norm.weight", true);
    m.head = try projectionQuantized(d, c, "lm_head.weight", true, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try vecTensor(d, c, f(p, "input_layernorm.weight"), true), .post_norm = try vecTensor(d, c, f(p, "post_attention_layernorm.weight"), true), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projectionGptq(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, 0, true, stage), .down = try projectionGptq(d, c, f(p, "mlp."), &.{"down_proj"}, false, 0, false, stage) };
        if (l.attn) {
            l.mixer_in = try projectionGptq(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, 0, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "self_attn."), &.{"o_proj"}, false, 0, false, stage);
            l.qn = try vecTensor(d, c, f(p, "self_attn.q_norm.weight"), true);
            l.kn = try vecTensor(d, c, f(p, "self_attn.k_norm.weight"), true);
            if (!q.kv_pool) { // the shared pool replaces the slot's own cache
                l.kc = try d.alloc_(kvBytes());
                l.vc = try d.alloc_(kvBytes());
            }
        } else {
            l.mixer_in = try projectionGptq(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z" }, true, GDN_IN, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, 0, false, stage);
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != HV * D * 2 or tb.bytes.len != HV * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(96 * D * 2);
            try d.copyIn(l.ab.?, ta.bytes);
            try d.copyIn(off(l.ab.?, HV * D * 2), tb.bytes);
            l.conv_w = try vecTensor(d, c, f(p, "linear_attn.conv1d.weight"), false);
            l.alog = try vecTensor(d, c, f(p, "linear_attn.A_log"), false);
            l.dt = try vecTensor(d, c, f(p, "linear_attn.dt_bias"), false);
            l.gnorm = try vecTensor(d, c, f(p, "linear_attn.norm.weight"), false);
            l.rec = try d.alloc_(HV * 128 * 128 * 4);
            l.rec_save = try d.alloc_(HV * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), HV * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * q.GDN_QKV * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * q.GDN_QKV * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20; // the staging area is gone
    return m;
}

/// One part of a rank-sliced MLX projection: the tensor (prefix + name) and the rank's cut of it.
pub const PartCut = struct { name: []const u8, cut: mlx.Cut = .{} };

/// `projection` for TP: parts stacked on N, each cut to the rank's rows (a column split) or K range (a row split) on the
/// host, then repacked by the same kernel. Padding rule as `projection`: N padded to 256, padding scales and biases zero.
pub fn projectionCut(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const PartCut, wide: bool, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p.name }));
        const sp = try mlx.span(p.cut, @intCast(w.shape[0]), @intCast(w.shape[1] * 8));
        if (k != 0 and sp.k != k) return error.UnexpectedShape;
        k = sp.k;
        n_total += sp.n;
    }
    const n_real = n_total;
    n_total = (n_total + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.s, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const nm = try std.fmt.allocPrint(alloc, "{s}{s}", .{ prefix, p.name });
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}.weight", .{nm}));
        const s = try c.get(try std.fmt.allocPrint(alloc, "{s}.scales", .{nm}));
        const b = try c.get(try std.fmt.allocPrint(alloc, "{s}.biases", .{nm}));
        const n_all: u32 = @intCast(w.shape[0]);
        const k_all: u32 = @intCast(w.shape[1] * 8);
        const sp = try mlx.span(p.cut, n_all, k_all);
        const wb = mlx.wordBytes(sp.n, sp.k);
        const gb = mlx.groupBytes(sp.n, sp.k);
        const host = try alloc.alloc(u8, wb + 2 * gb);
        defer alloc.free(host);
        try mlx.slice(host[0..wb], host[wb..][0..gb], host[wb + gb ..][0..gb], w.bytes, s.bytes, b.bytes, n_all, k_all, p.cut);
        const ss = off(stage, wb);
        const sb = off(ss, gb);
        try d.copyIn(stage, host[0..wb]);
        try d.copyIn(ss, host[wb..][0..gb]);
        try d.copyIn(sb, host[wb + gb ..][0..gb]);
        try d.launch(d.k.repack, .{ .x = (sp.n + 15) / 16, .y = sp.k / 8 }, .{ stage, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(sp.n)), @as(i32, @intCast(sp.k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += sp.n;
    }
    return out;
}

/// XPU_EMBED_HOST on an MLX checkpoint: the 4/64 table dequantized to the fp16 rows `embed_dense` reads, in pinned host
/// memory. Each value is q * s + b rounded to bf16, as the device `embed` kernel computes it, then held as fp16 (exact).
fn embedHost(d: *Dev, c: *const Checkpoint) !*anyopaque {
    const w = try c.get("model.embed_tokens.weight");
    const s = try c.get("model.embed_tokens.scales");
    const b = try c.get("model.embed_tokens.biases");
    if (w.shape[1] * 8 != D) return error.UnexpectedShape;
    const rows = w.shape[0];
    const out: [*]f16 = @ptrCast(@alignCast(try d.hostAlloc(rows * D * 2)));
    const bf = struct {
        fn f(bytes: []const u8, i: usize) f32 {
            return @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16);
        }
        fn r(v: f32) f32 {
            const u: u32 = @bitCast(v);
            return @bitCast(((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16);
        }
    };
    for (0..rows) |row| {
        for (0..D) |e| {
            const word = std.mem.readInt(u32, w.bytes[(row * (D / 8) + e / 8) * 4 ..][0..4], .little);
            const qv: f32 = @floatFromInt((word >> @intCast(4 * (e % 8))) & 0xF);
            const g = row * (D / 64) + e / 64;
            out[row * D + e] = @floatCast(bf.r(qv * bf.f(s.bytes, g) + bf.f(b.bytes, g)));
        }
    }
    return @ptrCast(out);
}

fn openEmbedHost() void {
    // the same line as the EXL3 loader: the gate looks for it on both ranks
    const E_BYTES = @as(f64, q.VOCAB * D / 2 + 2 * q.VOCAB * (D / 64) * 2) / 1e9; // GB the int4 table took on the device
    std.debug.print("embed on the host: {d:.2} GB of device memory saved ({d} x {d} fp16 rows)\n", .{ E_BYTES, q.VOCAB, D });
}

/// Qwen3.8-27B MLX 4/64 at TP 2 (or any TP): rank R's slices of every projection, head and embedding whole.
/// Column splits take output rows, row splits take K ranges, all on 64-input group boundaries (mlx_tp.zig).
pub fn loadMlxTp(d: *Dev, c: *const Checkpoint) !Model {
    q.GS = 64;
    const R = q.RANK;
    var m: Model = undefined;
    m.exl3 = null;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    if (q.embed_host) {
        m.embed_w = try embedHost(d, c);
        m.embed_s = null;
        m.embed_b = null;
    } else {
        m.embed_w = try d.upload(e_w.bytes);
        m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
        m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    }
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    // XPU_HSPLIT (WP6): each rank holds half of the head's vocab rows (rank 0 the low half, rank 1 the high). Off: whole on every rank
    m.head = if (q.hsplit) try projectionCut(d, c, "lm_head", &.{ .{ .name = "", .cut = mlx.Cut.cols(R * (q.VOCAB / 2), q.VOCAB / 2) } }, true, false, stage) else try projection(d, c, "lm_head", &.{""}, true, false, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    const FF = q.FF_R;
    const hk: u32 = q.GDN_K_HEADS * 128; // a rank's q (and k) rows of in_proj_qkv; stored [q 16 heads | k 16 | v 48]
    const hv: u32 = q.GDN_V_HEADS * 128;
    const Cut = mlx.Cut;
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, s }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")), .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projectionCut(d, c, f(p, "mlp."), &.{ .{ .name = "gate_proj", .cut = Cut.cols(R * FF, FF) }, .{ .name = "up_proj", .cut = Cut.cols(R * FF, FF) } }, true, true, stage), .down = try projectionCut(d, c, f(p, "mlp."), &.{.{ .name = "down_proj", .cut = Cut.rows(R * FF, FF) }}, false, false, stage) };
        if (l.attn) {
            const qw: u32 = q.ATT_Q_HEADS * 512; // a rank's q heads with their gates
            const kw: u32 = q.ATT_KV_HEADS * 256;
            l.mixer_in = try projectionCut(d, c, f(p, "self_attn."), &.{ .{ .name = "q_proj", .cut = Cut.cols(R * qw, qw) }, .{ .name = "k_proj", .cut = Cut.cols(R * kw, kw) }, .{ .name = "v_proj", .cut = Cut.cols(R * kw, kw) } }, true, false, stage);
            l.mixer_out = try projectionCut(d, c, f(p, "self_attn."), &.{.{ .name = "o_proj", .cut = Cut.rows(R * q.GDN_Z, q.GDN_Z) }}, false, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            if (!q.kv_pool) { // the shared pool replaces the slot's own cache
                l.kc = try d.alloc_(kvBytes());
                l.vc = try d.alloc_(kvBytes());
            }
        } else {
            // [q | k | v | z | a | b] of the rank, a and b as quantized parts of the fused projection (as at TP 1)
            l.mixer_in = try projectionCut(d, c, f(p, "linear_attn."), &.{
                .{ .name = "in_proj_qkv", .cut = Cut.cols(R * hk, hk) },
                .{ .name = "in_proj_qkv", .cut = Cut.cols(q.TP * hk + R * hk, hk) },
                .{ .name = "in_proj_qkv", .cut = Cut.cols(2 * q.TP * hk + R * hv, hv) },
                .{ .name = "in_proj_z", .cut = Cut.cols(R * hv, hv) },
                .{ .name = "in_proj_a", .cut = Cut.cols(R * HV, HV) },
                .{ .name = "in_proj_b", .cut = Cut.cols(R * HV, HV) },
            }, true, false, stage);
            if (l.mixer_in.n != GDN_IN) return error.UnexpectedShape;
            l.mixer_out = try projectionCut(d, c, f(p, "linear_attn."), &.{.{ .name = "out_proj", .cut = Cut.rows(R * hv, hv) }}, false, false, stage);
            l.conv_w = try x3.rowsTensor(d, c, f(p, "linear_attn.conv1d.weight"), 4 * 2, &.{ .{ R * hk, hk }, .{ q.TP * hk + R * hk, hk }, .{ 2 * q.TP * hk + R * hv, hv } });
            l.alog = try x3.rowsTensor(d, c, f(p, "linear_attn.A_log"), 2, &.{.{ R * HV, HV }});
            l.dt = try x3.rowsTensor(d, c, f(p, "linear_attn.dt_bias"), 2, &.{.{ R * HV, HV }});
            l.gnorm = try bf16Tensor(d, c, f(p, "linear_attn.norm.weight"));
            l.rec = try d.alloc_(HV * 128 * 128 * 4);
            l.rec_save = try d.alloc_(HV * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), HV * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * q.GDN_QKV * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * q.GDN_QKV * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  MLX TP layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20; // the staging area is gone
    if (q.embed_host) openEmbedHost();
    return m;
}

pub fn load(d: *Dev, c: *const Checkpoint) !Model {
    if (c.has("model.layers.0.linear_attn.in_proj_qkv.trellis")) return x3.loadExl3(d, c);
    if (c.has("in_proj_qkv.qweight")) {
        q.GS = 128;
        return loadGptq(d, c);
    }
    q.GS = 64;
    if (q.TP > 1) return loadMlxTp(d, c);
    var m: Model = undefined;
    m.exl3 = null;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    if (q.embed_host) {
        m.embed_w = try embedHost(d, c);
        m.embed_s = null;
        m.embed_b = null;
    } else {
        m.embed_w = try d.upload(e_w.bytes);
        m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
        m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    }
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    m.head = try projection(d, c, "lm_head", &.{""}, true, false, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, s }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")), .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projection(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, true, stage), .down = try projection(d, c, f(p, "mlp."), &.{"down_proj"}, false, false, stage) };
        if (l.attn) {
            l.mixer_in = try projection(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "self_attn."), &.{"o_proj"}, false, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            if (!q.kv_pool) { // the shared pool replaces the slot's own cache
                l.kc = try d.alloc_(kvBytes());
                l.vc = try d.alloc_(kvBytes());
            }
        } else {
            l.mixer_in = try projection(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, false, stage);
            l.conv_w = try bf16Tensor(d, c, f(p, "linear_attn.conv1d.weight"));
            l.alog = try bf16Tensor(d, c, f(p, "linear_attn.A_log"));
            l.dt = try bf16Tensor(d, c, f(p, "linear_attn.dt_bias"));
            l.gnorm = try bf16Tensor(d, c, f(p, "linear_attn.norm.weight"));
            l.rec = try d.alloc_(HV * 128 * 128 * 4);
            l.rec_save = try d.alloc_(HV * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), HV * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * q.GDN_QKV * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * q.GDN_QKV * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20; // the staging area is gone
    if (q.embed_host) openEmbedHost();
    return m;
}

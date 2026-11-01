//! Qwen3.8-27B on EXL3 weights (trellis, mul1 codebook): every projection through xpu/exl3.zig on the runner's list.
//! One exl3.Group a stored part. Fused projections join at the output by column offset, in the int4 path's layout.
//! The norms, GDN, attention, the dense a and b gates and the embedding stay the runner's (forward.zig).
const std = @import("std");
const ze = @import("ze");
const xpu = @import("xpu");
const exl3 = xpu.exl3;
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const loader = @import("load.zig");
const fwd = @import("forward.zig");
const Dev = dev.Dev;
const Checkpoint = dev.Checkpoint;
const D = q.D;
const FF = q.FF_R;
const HV = q.GDN_V_HEADS;
const alloc = q.alloc;
const off = dev.off;

/// Rows of a decode window (DFlash2 verify windows): a plan a row count, built once.
pub const WINDOW = 16;

/// The slice of a stored matrix a rank holds: rows [k0, k1) and columns [n0, n1), in multiples of 16 (the trellis
/// tile) and of 128 (the rotate block) where the matvec rotates. All zero: the whole matrix (TP 1).
pub const Cut = struct {
    k0: u32 = 0,
    k1: u32 = 0,
    n0: u32 = 0,
    n1: u32 = 0,

    fn whole(c: Cut) bool {
        return c.k1 == 0 and c.n1 == 0;
    }
    /// Columns [n0, n0 + n) of a matrix of `k` rows.
    pub fn cols(n0: u32, n: u32, k: u32) Cut {
        return .{ .k0 = 0, .k1 = k, .n0 = n0, .n1 = n0 + n };
    }
    /// Rows [k0, k0 + k) of a matrix of `n` columns.
    pub fn rows(k0: u32, k: u32, n: u32) Cut {
        return .{ .k0 = k0, .k1 = k0 + k, .n0 = 0, .n1 = n };
    }
};

/// A stored matrix to upload: its tensor prefix and the rank's slice of it.
pub const Spec = struct { prefix: []const u8, cut: Cut = .{} };

/// A stored matrix's place in its layer's device region.
const Mat = struct {
    k: u32,
    n: u32,
    halves: u32,
    at: [3]usize,

    fn on(m: Mat, base: u64) exl3.Matrix {
        return .{ .trellis = base + m.at[0], .suh = base + m.at[1], .svh = base + m.at[2], .k = m.k, .n = m.n, .halves = m.halves };
    }
};

/// One region's host image: each piece starts on a 256-byte boundary.
const Pack = struct {
    host: std.ArrayList(u8) = .empty,

    fn add(p: *Pack, bytes: []const u8) !usize {
        const at = std.mem.alignForward(usize, p.host.items.len, 256);
        try p.host.appendNTimes(alloc, 0, at - p.host.items.len);
        try p.host.appendSlice(alloc, bytes);
        return at;
    }

    fn matrix(p: *Pack, c: *const Checkpoint, prefix: []const u8, cut: Cut) !Mat {
        var buf: [256]u8 = undefined;
        const tr = try c.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}));
        const suh = try c.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{prefix}));
        const svh = try c.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{prefix}));
        const mul1 = try c.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix}));
        if (!std.mem.eql(u8, tr.dtype, "I16") or tr.rank != 3) return error.UnexpectedShape;
        // the mul1 flag of the exllamav3 export: other codebooks have no kernel here
        if (mul1.bytes.len != 4 or std.mem.readInt(i32, mul1.bytes[0..4], .little) == 0) return error.UnsupportedExl3Codebook;
        if (!std.mem.eql(u8, suh.dtype, "F16") or !std.mem.eql(u8, svh.dtype, "F16")) return error.UnexpectedDtype;
        // trellis [K/16, N/16, 16 * bits / 16 words]: halves = 2 * bits = words / 8 (exl3_layout.Group)
        const m: Mat = .{ .k = @intCast(tr.shape[0] * 16), .n = @intCast(tr.shape[1] * 16), .halves = @intCast(tr.shape[2] / 8), .at = undefined };
        if (suh.bytes.len != @as(usize, m.k) * 2 or svh.bytes.len != @as(usize, m.n) * 2) return error.UnexpectedShape;
        try m.on(0).validate();
        if (cut.whole()) {
            var out = m;
            out.at = .{ try p.add(tr.bytes), try p.add(suh.bytes), try p.add(svh.bytes) };
            return out;
        }
        // the rank's slice: tile rows [k0, k1) whole (contiguous), tiles [n0, n1) of each (strided)
        if (cut.k1 > m.k or cut.n1 > m.n or cut.k0 >= cut.k1 or cut.n0 >= cut.n1 or cut.k0 % 128 != 0 or cut.k1 % 128 != 0 or cut.n0 % 128 != 0 or cut.n1 % 128 != 0) {
            std.debug.print("exl3: {s} is {d} x {d}; the rank's cut rows [{d}, {d}) cols [{d}, {d}) is out of it or off a 128 boundary\n", .{ prefix, m.k, m.n, cut.k0, cut.k1, cut.n0, cut.n1 });
            return error.UnexpectedShape;
        }
        const tile = tr.shape[2] * 2; // bytes a 16 x 16 tile
        const n_tiles = m.n / 16;
        var out: Mat = .{ .k = cut.k1 - cut.k0, .n = cut.n1 - cut.n0, .halves = m.halves, .at = undefined };
        const at = std.mem.alignForward(usize, p.host.items.len, 256);
        try p.host.appendNTimes(alloc, 0, at - p.host.items.len);
        var kt = cut.k0 / 16;
        while (kt < cut.k1 / 16) : (kt += 1) {
            const row = (kt * n_tiles + cut.n0 / 16) * tile;
            try p.host.appendSlice(alloc, tr.bytes[row .. row + (out.n / 16) * tile]);
        }
        out.at = .{ at, try p.add(suh.bytes[cut.k0 * 2 .. cut.k1 * 2]), try p.add(svh.bytes[cut.n0 * 2 .. cut.n1 * 2]) };
        try out.on(0).validate();
        return out;
    }
};

/// One part of a fused projection: the matrix and its first output column.
const Part = struct { m: Mat, col: u32 };

/// An EXL3 projection of the runner: its parts in one region, the fused output width, and the plans over the
/// runner's fixed buffers (bind). `loader.Proj.x3` points here.
pub const Proj = struct {
    base: u64,
    parts: [4]Part = undefined,
    count: u32 = 0,
    /// The fused output width: the consumers' row stride.
    width: u32,
    k: u32,
    /// The buffers the plans read and write.
    x: u64 = 0,
    y: u64 = 0,
    dec: [WINDOW]exl3.Plan = undefined,
    chunk: ?exl3.Plan = null,
    /// The head only: rows 1 from row t of x (want = last, t < WINDOW), and from `row` (a prompt chunk's last row).
    last: [WINDOW]exl3.Plan = undefined,
    row: u64 = 0,
    row_plan: ?exl3.Plan = null,
    head: bool = false,

    fn groups(p: *const Proj, w: *const Weights, x: u64, y: u64, rows: u32, out: *[4]exl3.Group) []exl3.Group {
        for (p.parts[0..p.count], out[0..p.count], 0..) |pt, *g, j| g.* = .{
            .m = pt.m.on(p.base),
            .rows = rows,
            .x = x,
            .x_stride = p.k,
            .y = y + @as(u64, pt.col) * 2,
            .y_stride = p.width,
            .output = .f16,
            .input = .f16,
            .scratch = w.scratch[j],
        };
        return out[0..p.count];
    }

    fn plan(p: *const Proj, w: *const Weights, x: u64, y: u64, rows: u32) !exl3.Plan {
        var gs: [4]exl3.Group = undefined;
        return exl3.Plan.init(&w.ctx, w.stream(), alloc, p.groups(w, x, y, rows, &gs));
    }
};

pub const Weights = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    p: exl3.Pipelines,
    list: ze.H,
    projs: std.ArrayList(*Proj) = .empty,
    head: *Proj = undefined,
    scratch: [4]u64 = .{ 0, 0, 0, 0 },

    fn stream(w: *const Weights) xpu.Stream {
        return .{ .ctx = &w.ctx, .list = w.list };
    }

    /// Uploads one projection's parts (each a stored matrix, or the rank's slice of one) into one region, side by
    /// side at the output; `width` at least the parts' columns (the consumers' stride).
    fn add(w: *Weights, d: *Dev, c: *const Checkpoint, specs: []const Spec, width: u32) !*Proj {
        var pk: Pack = .{};
        defer pk.host.deinit(alloc);
        if (specs.len > 4) return error.UnexpectedShape;
        var parts: [4]Part = undefined;
        var col: u32 = 0;
        var k: u32 = 0;
        for (specs, parts[0..specs.len]) |sp, *pt| {
            pt.* = .{ .m = try pk.matrix(c, sp.prefix, sp.cut), .col = col };
            if (k != 0 and pt.m.k != k) return error.UnexpectedShape;
            k = pt.m.k;
            col += pt.m.n;
        }
        if (col > width) return error.UnexpectedShape;
        const p = try alloc.create(Proj);
        p.* = .{ .base = @intFromPtr(try d.upload(pk.host.items)), .parts = parts, .count = @intCast(specs.len), .width = width, .k = k };
        try w.projs.append(alloc, p);
        return p;
    }

    /// The plans over the runner's activations (after Acts.init), and the scratch they share (launches run in order,
    /// and the groups of one launch use scratch[part]).
    pub fn bind(w: *Weights, d: *Dev, m: *loader.Model, a: *fwd.Acts) !void {
        const rows = q.maxRows();
        if (rows > exl3.variants.moe.prompt_rows_max) return error.InvalidExl3Rows;
        var need: [4]usize = .{ 0, 0, 0, 0 };
        for (w.projs.items) |p| for (p.parts[0..p.count], 0..) |pt, j| {
            need[j] = @max(need[j], exl3.scratchBytes(pt.m.k, pt.m.n, WINDOW));
            if (!p.head) need[j] = @max(need[j], exl3.promptScratchBytes(pt.m.k, pt.m.n, rows));
        };
        for (&w.scratch, need) |*sc, n| if (n > 0) {
            sc.* = @intFromPtr(try d.alloc_(n));
        };
        const h: u64 = @intFromPtr(a.h);
        const pp: u64 = @intFromPtr(a.p);
        const r: u64 = @intFromPtr(a.r);
        for (&m.layers) |*l| {
            try w.bindOne(l.mixer_in.x3.?, h, pp, rows);
            try w.bindOne(l.mixer_out.x3.?, h, r, rows);
            try w.bindOne(l.gu.x3.?, h, pp, rows);
            try w.bindOne(l.down.x3.?, h, r, rows); // swiglu with ilv 0 writes the activation to h
        }
        const hp = w.head;
        hp.x = h;
        hp.y = @intFromPtr(a.logits);
        for (&hp.dec, 0..) |*pl, t| pl.* = try hp.plan(w, h, hp.y, @intCast(t + 1));
        for (&hp.last, 0..) |*pl, t| pl.* = try hp.plan(w, h + t * D * 2, hp.y, 1);
        hp.row = @intFromPtr(try d.alloc_(D * 2));
        hp.row_plan = try hp.plan(w, hp.row, hp.y, 1);
        try d.sync();
    }

    fn bindOne(w: *const Weights, p: *Proj, x: u64, y: u64, rows: u32) !void {
        p.x = x;
        p.y = y;
        for (&p.dec, 0..) |*pl, t| pl.* = try p.plan(w, x, y, @intCast(t + 1));
        p.chunk = try p.plan(w, x, y, rows);
    }
};

/// `x` [T, k] fp16 to `out` [T, width] fp16 through the plans `bind` built; `wide`: a prompt chunk (the XMX form).
pub fn project(p: *const Proj, w: *const Weights, d: *Dev, x: *anyopaque, out: *anyopaque, T: u32, wide: bool) !void {
    const xa: u64 = @intFromPtr(x);
    const ya: u64 = @intFromPtr(out);
    if (ya != p.y) return error.Exl3Unbound;
    if (xa == p.x) {
        if (!wide and T <= WINDOW) return exl3.encode(&w.p, w.stream(), &p.dec[T - 1]);
        if (p.chunk) |*c| return exl3.encodeRows(&w.p, w.stream(), c, T);
        if (T <= WINDOW) return exl3.encode(&w.p, w.stream(), &p.dec[T - 1]);
        return error.Exl3Unbound;
    }
    // the head: one row t of h (the forward's last row)
    if (!p.head or T != 1 or xa < p.x or (xa - p.x) % (D * 2) != 0) return error.Exl3Unbound;
    const t = (xa - p.x) / (D * 2);
    if (t < WINDOW) return exl3.encode(&w.p, w.stream(), &p.last[t]);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, @ptrFromInt(p.row), x, D * 2, null, 0, null));
    return exl3.encode(&w.p, w.stream(), &p.row_plan.?);
}

/// A bf16 tensor's rows [start, start + count) for each range, `row_bytes` a row, uploaded as one piece: a rank's
/// heads of a per-head tensor. One range over every row is the whole tensor.
fn rowsTensor(d: *Dev, c: *const Checkpoint, name: []const u8, row_bytes: usize, ranges: []const [2]u32) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    var total: usize = 0;
    for (ranges) |r| total += r[1];
    if (total * row_bytes == t.bytes.len) return d.upload(t.bytes);
    const out = try alloc.alloc(u8, total * row_bytes);
    defer alloc.free(out);
    var at: usize = 0;
    for (ranges) |r| {
        const from = @as(usize, r[0]) * row_bytes;
        const n = @as(usize, r[1]) * row_bytes;
        if (from + n > t.bytes.len) return error.UnexpectedShape;
        @memcpy(out[at .. at + n], t.bytes[from .. from + n]);
        at += n;
    }
    return d.upload(out);
}

/// A BF16 norm weight; `centred`: stored as gamma − 1 (the Hugging Face Qwen3.5 form) gets its 1 back.
fn normTensor(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    if (!centred) return d.upload(t.bytes);
    const n = t.bytes.len / 2;
    const out = try alloc.alloc(u16, n);
    defer alloc.free(out);
    for (out, 0..) |*o, i| {
        const v: f32 = @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16)) + 1.0;
        const u: u32 = @bitCast(v);
        o.* = @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    return d.upload(std.mem.sliceAsBytes(out));
}

/// The BF16 embedding table to the fp16 table embed_dense reads, in row blocks through the host.
fn embedTable(d: *Dev, c: *const Checkpoint) !*anyopaque {
    const t = try c.get("model.embed_tokens.weight");
    if (!std.mem.eql(u8, t.dtype, "BF16") or t.shape[1] != D) return error.UnexpectedDtype;
    const out = try d.alloc_(t.bytes.len);
    const block = 4096 * D;
    const buf = try alloc.alloc(f16, block);
    defer alloc.free(buf);
    var at: usize = 0;
    const n = t.bytes.len / 2;
    while (at < n) {
        const m = @min(block, n - at);
        for (buf[0..m], 0..) |*o, i| o.* = @floatCast(@as(f32, @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * (at + i) ..][0..2], .little)) << 16)));
        try d.copyIn(off(out, at * 2), std.mem.sliceAsBytes(buf[0..m]));
        at += m;
    }
    return out;
}

/// An int4 Proj shell for an EXL3 projection: the int4 fields stay unused, `x3` takes the path.
fn shell(p: *Proj, wide: bool) loader.Proj {
    const none: *anyopaque = @ptrFromInt(@as(usize, 256));
    return .{ .w = none, .s = none, .bg = none, .n = p.width, .k = p.k, .wide = wide, .gs = q.GS, .ilv = false, .x3 = p };
}

/// Qwen3.8-27B from an EXL3 checkpoint (exllamav3, mul1 codebook): EXL3 projections, the fp16 embedding, BF16 norms.
pub fn loadExl3(d: *Dev, c: *const Checkpoint) !loader.Model {
    q.GS = 64; // the norms' activation groups only: no int4 GEMM reads them, and the W4A8 paths stay off (GS 128)
    const w = try alloc.create(Weights);
    w.* = .{ .drv = try xpu.Driver.open(), .ctx = undefined, .p = undefined, .list = d.list };
    w.ctx = .{ .d = &w.drv, .gpu = d.g };
    w.p = try exl3.Pipelines.load(&w.ctx);
    var m: loader.Model = undefined;
    m.exl3 = w;
    m.embed_w = try embedTable(d, c);
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try normTensor(d, c, "model.norm.weight", true);
    w.head = try w.add(d, c, &.{.{ .prefix = "lm_head" }}, q.VOCAB); // whole on every rank: the hidden state is whole
    w.head.head = true;
    m.head = shell(w.head, true);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    var nb: [8][128]u8 = undefined;
    for (&m.layers, 0..) |*l, i| {
        const f = struct {
            fn n(b: *[128]u8, li: usize, sx: []const u8) []const u8 {
                return std.fmt.bufPrint(b, "model.layers.{d}.{s}", .{ li, sx }) catch unreachable;
            }
        }.n;
        const attn = q.isAttention(i);
        // TP: rank R holds columns [R * width, (R + 1) * width) of every in-projection, by head where heads exist, and
        // rows of every out-projection (its output is a partial sum the ranks add). At TP 1 every cut is whole.
        const R = q.RANK;
        const tp1 = q.TP == 1;
        const gu = try w.add(d, c, &.{
            .{ .prefix = f(&nb[0], i, "mlp.gate_proj"), .cut = if (tp1) .{} else Cut.cols(R * FF, FF, D) },
            .{ .prefix = f(&nb[1], i, "mlp.up_proj"), .cut = if (tp1) .{} else Cut.cols(R * FF, FF, D) },
        }, 2 * FF);
        const down = try w.add(d, c, &.{.{ .prefix = f(&nb[0], i, "mlp.down_proj"), .cut = if (tp1) .{} else Cut.rows(R * FF, FF, D) }}, D);
        l.* = .{ .in_norm = try normTensor(d, c, f(&nb[0], i, "input_layernorm.weight"), true), .post_norm = try normTensor(d, c, f(&nb[1], i, "post_attention_layernorm.weight"), true), .attn = attn, .mixer_in = undefined, .mixer_out = undefined, .gu = shell(gu, true), .down = shell(down, false) };
        if (attn) {
            // [q (q | gate per head) 12288 | k 1024 | v 1024] = ATT_IN, as the int4 fused layout
            const qw = q.ATT_Q_HEADS * 512; // a rank's q heads with their gates
            const kw = q.ATT_KV_HEADS * 256;
            const mi = try w.add(d, c, &.{
                .{ .prefix = f(&nb[0], i, "self_attn.q_proj"), .cut = if (tp1) .{} else Cut.cols(R * qw, qw, D) },
                .{ .prefix = f(&nb[1], i, "self_attn.k_proj"), .cut = if (tp1) .{} else Cut.cols(R * kw, kw, D) },
                .{ .prefix = f(&nb[2], i, "self_attn.v_proj"), .cut = if (tp1) .{} else Cut.cols(R * kw, kw, D) },
            }, q.ATT_IN);
            l.mixer_in = shell(mi, true);
            l.mixer_out = shell(try w.add(d, c, &.{.{ .prefix = f(&nb[0], i, "self_attn.o_proj"), .cut = if (tp1) .{} else Cut.rows(R * q.GDN_Z, q.GDN_Z, D) }}, D), false);
            l.qn = try normTensor(d, c, f(&nb[0], i, "self_attn.q_norm.weight"), true);
            l.kn = try normTensor(d, c, f(&nb[0], i, "self_attn.k_norm.weight"), true);
            l.kc = try d.alloc_(fwd.kvBytes());
            l.vc = try d.alloc_(fwd.kvBytes());
        } else {
            // [qkv 10240 | z 6144] in a row of GDN_IN; the dense a and b gates go to columns 16384 on (forward.zig)
            const hk = q.GDN_K_HEADS * 128; // a rank's q (and k) columns of in_proj_qkv: [q 16 heads | k 16 | v 48] whole
            const hv = q.GDN_V_HEADS * 128;
            const qkv = f(&nb[0], i, "linear_attn.in_proj_qkv");
            const mi = if (tp1) try w.add(d, c, &.{ .{ .prefix = qkv }, .{ .prefix = f(&nb[1], i, "linear_attn.in_proj_z") } }, q.GDN_IN) else try w.add(d, c, &.{
                .{ .prefix = qkv, .cut = Cut.cols(R * hk, hk, D) },
                .{ .prefix = qkv, .cut = Cut.cols(q.TP * hk + R * hk, hk, D) },
                .{ .prefix = qkv, .cut = Cut.cols(2 * q.TP * hk + R * hv, hv, D) },
                .{ .prefix = f(&nb[1], i, "linear_attn.in_proj_z"), .cut = Cut.cols(R * hv, hv, D) },
            }, q.GDN_IN);
            if (mi.parts[mi.count - 1].col + mi.parts[mi.count - 1].m.n != q.GDN_AB) return error.UnexpectedShape;
            l.mixer_in = shell(mi, true);
            l.mixer_out = shell(try w.add(d, c, &.{.{ .prefix = f(&nb[0], i, "linear_attn.out_proj"), .cut = if (tp1) .{} else Cut.rows(R * hv, hv, D) }}, D), false);
            const ta = try c.get(f(&nb[0], i, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(&nb[1], i, "linear_attn.in_proj_b.weight"));
            // the dense a and b gates: 48 rows of D (whole), the rank's 24 at TP 2
            if (!std.mem.eql(u8, ta.dtype, "F16") or !std.mem.eql(u8, tb.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(2 * HV * D * 2);
            try d.copyIn(l.ab.?, ta.bytes[R * HV * D * 2 ..][0 .. HV * D * 2]);
            try d.copyIn(off(l.ab.?, HV * D * 2), tb.bytes[R * HV * D * 2 ..][0 .. HV * D * 2]);
            // conv1d [10240 channels, 4 taps] bf16 by channel in qkv order, A_log and dt_bias by value head: the
            // rank's heads of each (whole at TP 1). The GDN norm weight is [128], one row of head dim shared by every
            // head: whole at every TP.
            l.conv_w = try rowsTensor(d, c, f(&nb[0], i, "linear_attn.conv1d.weight"), 4 * 2, &.{ .{ R * hk, hk }, .{ q.TP * hk + R * hk, hk }, .{ 2 * q.TP * hk + R * hv, hv } });
            l.alog = try rowsTensor(d, c, f(&nb[0], i, "linear_attn.A_log"), 2, &.{.{ R * q.GDN_V_HEADS, q.GDN_V_HEADS }});
            l.dt = try rowsTensor(d, c, f(&nb[0], i, "linear_attn.dt_bias"), 2, &.{.{ R * q.GDN_V_HEADS, q.GDN_V_HEADS }});
            l.gnorm = try normTensor(d, c, f(&nb[0], i, "linear_attn.norm.weight"), false);
            l.rec = try d.alloc_(HV * 128 * 128 * 4);
            l.rec_save = try d.alloc_(HV * 128 * 128 * 4);
            l.stash = try d.alloc_(q.STASH);
            try d.fill(l.rec.?, @as(u32, 0), HV * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * q.GDN_QKV * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * q.GDN_QKV * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  EXL3 layers loaded: {d}/{d}\n", .{ i + 1, q.LAYERS });
    }
    try d.sync();
    return m;
}

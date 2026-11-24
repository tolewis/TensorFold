//! Inkling 2 Mini's resident text model on the XPU backend: checkpoint weights validated by name, dtype and shape,
//! then uploaded to device buffers as stored (experts stay NVFP4 or MXFP8; nothing is decoded or repacked here).
const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const st = @import("core").safetensors;
const images = @import("inkling2_xpu_images");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const experts = @import("experts.zig");
const loop = @import("loop.zig");
const runtime = @import("runtime.zig");
const dense = @import("dense.zig");
const dense_q8 = @import("dense_q8.zig");
const dense_q4 = @import("dense_q4.zig");
const q8 = @import("q8.zig");
const q4 = @import("q4.zig");
const nv4 = @import("nv4.zig");

const rt = xpu.rt;
const loader = xpu.loader;

pub const DType = st.DType;

/// One checkpoint tensor resident on the device (or in pinned host memory when `host`). The buffer starts at offset 0.
pub const Tensor = struct {
    buf: rt.Buffer,
    bytes: usize,
    dtype: DType,
    shape: [st.max_rank]usize = @splat(1),
    rank: usize = 0,
    host: bool = false,
    /// A dense matrix re-encoded at load to q8.zig's blocks (`dtype` is u8, `shape` the matrix's [n][k]); `bytes` is the codes then the scales.
    q8: bool = false,
    /// A dense matrix re-encoded at load to nv4.zig's NVFP4 blocks. The head is never this.
    nv4: bool = false,
    /// A dense matrix re-encoded at load to q4.zig's blocks. The head is never this.
    q4: bool = false,

    pub fn count(self: Tensor) usize {
        return self.bytes / self.dtype.size();
    }

    /// The tensor's bytes as host memory; only pinned host tensors have one.
    pub fn hostSlice(self: Tensor, comptime T: type) ?[]const T {
        if (!self.host) return null;
        const base: [*]const T = @ptrCast(@alignCast(self.buf.ptr.?));
        return base[0 .. self.bytes / @sizeOf(T)];
    }
};
pub const DynamicConv = struct { static_w: Tensor, projection: Tensor };
pub const Conv = union(enum) { static: Tensor, dynamic: DynamicConv };
pub const Controller = struct { down: Tensor, forget: Tensor, gain: Tensor };
pub const AttentionWeights = struct {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wr: Tensor,
    wo: Tensor,
    gate: Tensor,
    sink: Tensor,
    q_scale: Tensor,
    rel_scale: Tensor,
    rel_projection: Tensor,
    k_conv: Conv,
    v_conv: Conv,
    q_norm: Tensor,
    k_norm: Tensor,
    v_norm: Tensor,
};
/// NVFP4: `weight` u8 pairs, `scale` FP8 E4M3 per 16, `amax` per expert and the outer-scale `denominator`.
/// MXFP8: `weight` FP8 E4M3, `scale` E8M0 bytes per 32; no `amax`, no `denominator`.
pub const ExpertMatrix = struct {
    weight: Tensor,
    scale: Tensor,
    amax: ?Tensor = null,
    denominator: ?Tensor = null,
};
pub const LayerWeights = struct {
    attn: AttentionWeights,
    attn_conv: Tensor,
    mlp_norm: Tensor,
    mlp_conv: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    gu: ExpertMatrix,
    down: ExpertMatrix,
    controllers: ?[4]Controller,
};
pub const Weights = struct {
    embed: Tensor,
    embed_norm: Tensor,
    norm: Tensor,
    /// The checkpoint's BF16 head; empty (no buffer) when the Q8 head replaced it (`Options.keep_bf16_head` false).
    unembed: Tensor,
    /// The head's first `unpadded_vocab` rows as Q8_0 blocks, when `Options.dense` is `.q8`.
    unembed_q8: ?Tensor,
    output_query: Tensor,
    cross_queries: Tensor,
    cross_kv_queries: Tensor,
    attn_queries: [23]Tensor,
    mlp_queries: [24]Tensor,
    layers: [24]LayerWeights,
};

/// The production kernels of every slice, built from the embedded images (`inkling2_xpu_images`): the modules own the
/// code, each wrapper holds the kernels made from one of them.
pub const Pipelines = struct {
    modules: [image_count]xpu.Module,
    attention: attention.Kernels,
    glue: glue.Pipelines,
    experts: experts.Kernels,
    loop: loop.Pipelines,
    runtime: runtime.Pipelines,
    dense: dense.Kernels,
    dense_q8: dense_q8.Kernels,
    dense_q4: dense_q4.Kernels,

    const image_count = 9;

    fn native(ctx: *const xpu.Context, comptime name: []const u8) !xpu.Module {
        return xpu.Module.loadNative(ctx, &@field(images, name));
    }

    pub fn load(ctx: *const xpu.Context) !Pipelines {
        var p: Pipelines = undefined;
        var loaded: usize = 0;
        errdefer for (p.modules[0..loaded]) |*mod| mod.unload();
        inline for (.{ "attention", "prompt", "glue", "experts", "loop", "runtime", "dense", "dense_q8", "dense_q4" }, 0..) |name, i| {
            p.modules[i] = try native(ctx, name);
            loaded += 1;
        }
        p.attention = try attention.Kernels.init(p.modules[0], p.modules[1]);
        errdefer p.attention.deinit();
        p.glue = try glue.Pipelines.load(p.modules[2], false);
        errdefer p.glue.deinit();
        p.experts = try experts.Kernels.load(&p.modules[3], experts.default_prefix, null);
        errdefer p.experts.deinit();
        p.loop = try loop.Pipelines.load(p.modules[4]);
        errdefer p.loop.deinit();
        p.runtime = try runtime.Pipelines.load(p.modules[5], false);
        errdefer p.runtime.deinit();
        p.dense = try dense.Kernels.load(p.modules[6], false);
        errdefer p.dense.deinit();
        p.dense_q8 = try dense_q8.Kernels.load(p.modules[7], false);
        errdefer p.dense_q8.deinit();
        p.dense_q4 = try dense_q4.Kernels.load(p.modules[8], false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.dense_q4.deinit();
        p.dense_q8.deinit();
        p.dense.deinit();
        p.runtime.deinit();
        p.loop.deinit();
        p.experts.deinit();
        p.glue.deinit();
        p.attention.deinit();
        for (&p.modules) |*mod| mod.unload();
        p.* = undefined;
    }
};

/// How the dense weights (attention projections, the gate, the unembedding head) are held on the device.
pub const DenseFormat = enum {
    /// Re-encoded at load to Q8_0 blocks (q8.zig): int8 codes and one fp16 scale a 32 values, about half the bytes of bf16.
    q8,
    /// As stored: bf16.
    bf16,
    /// Attention projections (wq, wk, wv, wo, gate, wr) re-encoded to NVFP4 (nv4.zig). The head stays Q8_0.
    /// Opt-in: `XPU_DENSE=nv4` or `--dense nv4`. Q8_0 stays the default.
    nv4,
    /// The same projections as Q4 blocks (q4.zig): 4-bit codes, one fp16 scale a 32 values. The head stays Q8_0.
    /// Opt-in: `XPU_DENSE=q4` or `--dense q4`.
    q4,
};

/// Projections `XPU_DENSE_Q8` keeps at Q8_0 when the format is NVFP4 (a quality fallback: `wk,wv,wr`).
pub const Q8Keep = struct {
    wq: bool = false,
    wk: bool = false,
    wv: bool = false,
    wo: bool = false,
    gate: bool = false,
    wr: bool = false,

    pub fn has(self: Q8Keep, role: []const u8) bool {
        if (std.mem.eql(u8, role, "wq")) return self.wq;
        if (std.mem.eql(u8, role, "wk")) return self.wk;
        if (std.mem.eql(u8, role, "wv")) return self.wv;
        if (std.mem.eql(u8, role, "wo")) return self.wo;
        if (std.mem.eql(u8, role, "gate")) return self.gate;
        if (std.mem.eql(u8, role, "wr")) return self.wr;
        return false;
    }

    fn parse(text: []const u8) Q8Keep {
        var out: Q8Keep = .{};
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |role| {
            if (std.mem.eql(u8, role, "wq")) out.wq = true;
            if (std.mem.eql(u8, role, "wk")) out.wk = true;
            if (std.mem.eql(u8, role, "wv")) out.wv = true;
            if (std.mem.eql(u8, role, "wo")) out.wo = true;
            if (std.mem.eql(u8, role, "gate")) out.gate = true;
            if (std.mem.eql(u8, role, "wr")) out.wr = true;
        }
        return out;
    }
};

pub const Options = struct {
    /// Keep the token embedding in pinned host memory instead of VRAM (the plan's S2). Default: on the device.
    embed_host: bool = false,
    /// The dense weights' format on the device. `XPU_DENSE=bf16`, `q8`, `nv4` or `q4` in the environment overrides this at load,
    /// so the format A/B is one flag on any binary.
    dense: DenseFormat = .q8,
    /// With `.q8` or `.nv4`, keep the checkpoint's bf16 head on the device as well. DSpark's drafter reads that head (its base
    /// kernel and its one-time fold), so a drafter needs it; without one, `false` gives back about 0.93 GB.
    keep_bf16_head: bool = true,
    /// Projections left at Q8_0 when `dense` is `.nv4`. `XPU_DENSE_Q8=wk,wv,wr` sets this.
    keep_q8: Q8Keep = .{},
};

/// `options` with the `XPU_DENSE` environment override applied.
fn withEnvironment(options: Options) Options {
    var out = options;
    if (std.c.getenv("XPU_DENSE")) |v| {
        const text = std.mem.span(v);
        if (std.mem.eql(u8, text, "bf16")) out.dense = .bf16 else if (std.mem.eql(u8, text, "q8")) out.dense = .q8 else if (std.mem.eql(u8, text, "nv4")) out.dense = .nv4 else if (std.mem.eql(u8, text, "q4")) out.dense = .q4;
    }
    if (std.c.getenv("XPU_DENSE_Q8")) |v| out.keep_q8 = Q8Keep.parse(std.mem.span(v));
    return out;
}

pub fn admitConfig(c: *const cfg.Config) !void {
    const t = c.text;
    const l = t.loop orelse return error.Inkling2NotMini;
    if (l.passes != 3 or l.core_start != 6 or l.core_end != 18 or l.gate_rank != 64)
        return error.UnsupportedInkling2Loop;
    if (t.hidden != 2304 or t.layers != 24 or t.vocab != 201024 or t.unpadded_vocab != 200058 or
        t.window != 1024 or t.d_rel != 16 or t.rel_extent != 1024 or t.log_floor != 8192 or
        t.block != 6 or t.experts != 128 or t.topk != 8 or t.expert_inter != 1152 or
        t.max_context != 1048576) return error.UnsupportedInkling2Geometry;
    for ([_]cfg.Heads{ t.local, t.global }) |h| if (h.q != 24 or h.kv != 8 or h.dim != 128)
        return error.UnsupportedInkling2Geometry;
    if (t.eps != 1e-6 or t.log_alpha != 0.1 or t.qk_clip != 22.627416997969522 or
        t.rel_clip != 8 or t.route_scale != 8 or t.gate_cap != 4 or t.up_cap != 25 or
        t.logit_cap != 30) return error.UnsupportedInkling2Constants;
}

/// Reads tensors by name, checks each one's dtype, shape and byte count, and (unless `dry`) uploads it.
/// A dry pass walks every name and checks it without allocating, so a wrong checkpoint is refused before any upload.
const Source = struct {
    l: *loader.Loader,
    owned: *std.ArrayList(rt.Buffer),
    gpa: std.mem.Allocator,
    dry: bool,
    embed_host: bool,
    options: Options = .{},
    scratch: ?*Scratch = null,

    /// Host buffers one matrix chunk is read into and encoded in.
    const Scratch = struct {
        raw: []u16,
        codes: []u8,
        scales: []u16,
        const values = 2 << 20;
    };

    fn require(s: *const Source, name: []const u8, dtype: DType, dims: []const usize) !loader.Info {
        const inf = s.l.map.get(name) orelse {
            std.log.err("inkling2: tensor missing: {s}", .{name});
            return error.MissingTensor;
        };
        if (inf.dt != dtype or inf.rank != dims.len) return error.UnexpectedInkling2Tensor;
        var count: usize = 1;
        for (dims, 0..) |d, i| {
            if (inf.shape[i] != d) return error.UnexpectedInkling2Tensor;
            count = std.math.mul(usize, count, d) catch return error.UnexpectedInkling2Tensor;
        }
        const want = std.math.mul(usize, count, dtype.size()) catch return error.UnexpectedInkling2Tensor;
        if (inf.len != want) return error.UnexpectedInkling2Tensor;
        return inf;
    }

    fn get(s: *Source, comptime fmt: []const u8, args: anytype, dtype: DType, dims: []const usize) !Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const inf = try s.require(name, dtype, dims);
        var t: Tensor = .{ .buf = .{ .rt = s.l.r, .ptr = null, .len = 0 }, .bytes = @intCast(inf.len), .dtype = dtype, .shape = inf.shape, .rank = inf.rank };
        if (s.dry) return t;
        try s.owned.ensureUnusedCapacity(s.gpa, 1);
        t.buf = try s.l.load(name);
        s.owned.appendAssumeCapacity(t.buf);
        return t;
    }

    /// A projection: NVFP4 when that format is selected and the role is not in `keep_q8`, else the Q8 or bf16 path.
    fn getProj(s: *Source, role: []const u8, comptime fmt: []const u8, args: anytype, dims: []const usize) !Tensor {
        if (s.options.dense == .nv4 and !s.options.keep_q8.has(role)) return s.getNv4(fmt, args, dims);
        if (s.options.dense == .q4 and !s.options.keep_q8.has(role)) return s.getQ4(fmt, args, dims);
        return s.getDense(fmt, args, dims, null);
    }

    /// A bf16 matrix [n][k] as NVFP4 (nv4.zig): one expert, the factor in amax, denominator 1.
    fn getNv4(s: *Source, comptime fmt: []const u8, args: anytype, dims: []const usize) !Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const inf = try s.require(name, .bf16, dims);
        if (dims.len != 2 or !nv4.admits(dims[1])) return error.UnexpectedInkling2Tensor;
        const n = dims[0];
        const k = dims[1];
        var t: Tensor = .{ .buf = .{ .rt = s.l.r, .ptr = null, .len = 0 }, .bytes = nv4.bytes(n, k), .dtype = .u8, .shape = inf.shape, .rank = inf.rank, .nv4 = true };
        if (s.dry) return t;
        try s.owned.ensureUnusedCapacity(s.gpa, 1);
        t.buf = try s.l.empty(t.bytes);
        errdefer t.buf.free();
        try s.encodeNv4(t.buf, inf, n, k);
        s.owned.appendAssumeCapacity(t.buf);
        return t;
    }

    /// A bf16 matrix [n][k] as Q4 blocks (q4.zig).
    fn getQ4(s: *Source, comptime fmt: []const u8, args: anytype, dims: []const usize) !Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const inf = try s.require(name, .bf16, dims);
        if (dims.len != 2 or !q4.admits(dims[1])) return error.UnexpectedInkling2Tensor;
        const n = dims[0];
        const k = dims[1];
        var t: Tensor = .{ .buf = .{ .rt = s.l.r, .ptr = null, .len = 0 }, .bytes = q4.bytes(n, k), .dtype = .u8, .shape = inf.shape, .rank = inf.rank, .q4 = true };
        if (s.dry) return t;
        try s.owned.ensureUnusedCapacity(s.gpa, 1);
        t.buf = try s.l.empty(t.bytes);
        errdefer t.buf.free();
        try s.encodeQ4(t.buf, inf, n, k);
        s.owned.appendAssumeCapacity(t.buf);
        return t;
    }

    /// A bf16 matrix [n][k] as the options say: as stored, or (`.q8` or `.nv4`) re-encoded to Q8_0 blocks as it streams in.
    /// `rows` limits the encoding to the matrix's first rows (the head's padding rows are never read). NVFP4 projections
    /// go through `getProj`; this path is the head and any projection `keep_q8` holds back.
    fn getDense(s: *Source, comptime fmt: []const u8, args: anytype, dims: []const usize, rows: ?usize) !Tensor {
        if (s.options.dense == .bf16) return s.get(fmt, args, .bf16, dims);
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const inf = try s.require(name, .bf16, dims);
        if (dims.len != 2 or !q8.admits(dims[1])) return error.UnexpectedInkling2Tensor;
        const n = rows orelse dims[0];
        const k = dims[1];
        var t: Tensor = .{ .buf = .{ .rt = s.l.r, .ptr = null, .len = 0 }, .bytes = q8.bytes(n, k), .dtype = .u8, .shape = inf.shape, .rank = inf.rank, .q8 = true };
        t.shape[0] = n;
        if (s.dry) return t;
        try s.owned.ensureUnusedCapacity(s.gpa, 1);
        t.buf = try s.l.empty(t.bytes);
        errdefer t.buf.free();
        try s.encodeInto(t.buf, inf, n, k);
        s.owned.appendAssumeCapacity(t.buf);
        return t;
    }

    fn encodeInto(s: *Source, dst: rt.Buffer, inf: loader.Info, n: usize, k: usize) !void {
        if (s.scratch == null) {
            const sc = try s.gpa.create(Scratch);
            errdefer s.gpa.destroy(sc);
            sc.raw = try s.gpa.alloc(u16, Scratch.values);
            errdefer s.gpa.free(sc.raw);
            sc.codes = try s.gpa.alloc(u8, Scratch.values);
            errdefer s.gpa.free(sc.codes);
            sc.scales = try s.gpa.alloc(u16, Scratch.values / q8.block);
            s.scratch = sc;
        }
        const sc = s.scratch.?;
        const nb = k / q8.block;
        const per = @max(1, Scratch.values / k);
        const base: [*]u8 = @ptrCast(dst.ptr.?);
        var row: usize = 0;
        while (row < n) {
            const m = @min(per, n - row);
            try loader.readExact(inf.fd, std.mem.sliceAsBytes(sc.raw[0 .. m * k]), inf.off + @as(u64, row) * k * 2);
            q8.encodeRows(sc.raw[0 .. m * k], m, k, sc.codes[0 .. m * k], sc.scales[0 .. m * nb]);
            try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + row * k, .len = m * k }, sc.codes[0 .. m * k]);
            try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + q8.scaleOffset(n, k) + row * nb * 2, .len = m * nb * 2 }, std.mem.sliceAsBytes(sc.scales[0 .. m * nb]));
            try s.l.r.sync();
            row += m;
        }
    }

    fn encodeNv4(s: *Source, dst: rt.Buffer, inf: loader.Info, n: usize, k: usize) !void {
        if (s.scratch == null) {
            const sc = try s.gpa.create(Scratch);
            errdefer s.gpa.destroy(sc);
            sc.raw = try s.gpa.alloc(u16, Scratch.values);
            errdefer s.gpa.free(sc.raw);
            sc.codes = try s.gpa.alloc(u8, Scratch.values);
            errdefer s.gpa.free(sc.codes);
            sc.scales = try s.gpa.alloc(u16, Scratch.values / q8.block);
            s.scratch = sc;
        }
        const sc = s.scratch.?;
        const per = @max(1, Scratch.values / k);
        const base: [*]u8 = @ptrCast(dst.ptr.?);
        var amax: f32 = 0;
        var row: usize = 0;
        while (row < n) {
            const m = @min(per, n - row);
            try loader.readExact(inf.fd, std.mem.sliceAsBytes(sc.raw[0 .. m * k]), inf.off + @as(u64, row) * k * 2);
            amax = @max(amax, nv4.amaxOf(sc.raw[0 .. m * k]));
            row += m;
        }
        const factor = nv4.factorOf(amax);
        const phi = nv4.kernelPhi(factor);
        var tail = [_]f32{ factor, 1 };
        try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + nv4.factorOffset(n, k), .len = 8 }, std.mem.sliceAsBytes(&tail));
        row = 0;
        while (row < n) {
            const m = @min(per, n - row);
            const code_bytes = m * (k / 2);
            const scale_bytes = m * (k / nv4.group);
            try loader.readExact(inf.fd, std.mem.sliceAsBytes(sc.raw[0 .. m * k]), inf.off + @as(u64, row) * k * 2);
            nv4.encodeRows(sc.raw[0 .. m * k], m, k, phi, sc.codes[0..code_bytes], sc.codes[code_bytes..][0..scale_bytes]);
            try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + row * (k / 2), .len = code_bytes }, sc.codes[0..code_bytes]);
            try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + nv4.scaleOffset(n, k) + row * (k / nv4.group), .len = scale_bytes }, sc.codes[code_bytes..][0..scale_bytes]);
            try s.l.r.sync();
            row += m;
        }
    }

    fn encodeQ4(s: *Source, dst: rt.Buffer, inf: loader.Info, n: usize, k: usize) !void {
        if (s.scratch == null) {
            const sc = try s.gpa.create(Scratch);
            errdefer s.gpa.destroy(sc);
            sc.raw = try s.gpa.alloc(u16, Scratch.values);
            errdefer s.gpa.free(sc.raw);
            sc.codes = try s.gpa.alloc(u8, Scratch.values);
            errdefer s.gpa.free(sc.codes);
            sc.scales = try s.gpa.alloc(u16, Scratch.values / q8.block);
            s.scratch = sc;
        }
        const sc = s.scratch.?;
        const nb = k / q4.group;
        const per = @max(1, @min(Scratch.values / k, sc.scales.len / nb));
        const base: [*]u8 = @ptrCast(dst.ptr.?);
        var row: usize = 0;
        while (row < n) {
            const m = @min(per, n - row);
            try loader.readExact(inf.fd, std.mem.sliceAsBytes(sc.raw[0 .. m * k]), inf.off + @as(u64, row) * k * 2);
            q4.encodeRows(sc.raw[0 .. m * k], m, k, sc.codes[0 .. m * (k / 2)], sc.scales[0 .. m * nb]);
            try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + row * (k / 2), .len = m * (k / 2) }, sc.codes[0 .. m * (k / 2)]);
            try s.l.r.upload(.{ .rt = dst.rt, .ptr = base + q4.scaleOffset(n, k) + row * nb * 2, .len = m * nb * 2 }, std.mem.sliceAsBytes(sc.scales[0 .. m * nb]));
            try s.l.r.sync();
            row += m;
        }
    }

    fn freeScratch(s: *Source) void {
        if (s.scratch) |sc| {
            s.gpa.free(sc.raw);
            s.gpa.free(sc.codes);
            s.gpa.free(sc.scales);
            s.gpa.destroy(sc);
            s.scratch = null;
        }
    }

    /// As `get`, in pinned host memory when `embed_host` (the table the kernels read a row at a time over PCIe).
    fn getEmbed(s: *Source, comptime name: []const u8, dtype: DType, dims: []const usize) !Tensor {
        if (!s.embed_host) return s.get(name, .{}, dtype, dims);
        const inf = try s.require(name, dtype, dims);
        var t: Tensor = .{ .buf = .{ .rt = s.l.r, .ptr = null, .len = 0 }, .bytes = @intCast(inf.len), .dtype = dtype, .shape = inf.shape, .rank = inf.rank, .host = true };
        if (s.dry) return t;
        try s.owned.ensureUnusedCapacity(s.gpa, 1);
        t.buf = try s.l.emptyHost(@intCast(inf.len));
        errdefer t.buf.free();
        const dst: [*]u8 = @ptrCast(t.buf.ptr.?);
        try loader.readExact(inf.fd, dst[0..@intCast(inf.len)], inf.off);
        s.owned.appendAssumeCapacity(t.buf);
        return t;
    }
};

fn matrix(s: *Source, layer: u32, comptime suffix: []const u8, precision: layer_plan.Precision, rows: usize, k: usize) !ExpertMatrix {
    const base = "model.llm.layers.{d}.mlp.experts." ++ suffix;
    const e = 128;
    switch (precision) {
        .nvfp4 => return .{
            .weight = try s.get(base ++ "_weight", .{layer}, .u8, &.{ e, rows, k / 2 }),
            .scale = try s.get(base ++ "_weight_scale", .{layer}, .f8_e4m3, &.{ e, rows, k / 16 }),
            .amax = try s.get(base ++ "_weight_amax", .{layer}, .f32, &.{e}),
            .denominator = try s.get(base ++ "_weight_outer_scale_denominator", .{layer}, .f32, &.{1}),
        },
        .mxfp8 => return .{
            .weight = try s.get(base ++ "_weight", .{layer}, .f8_e4m3, &.{ e, rows, k }),
            .scale = try s.get(base ++ "_weight_scale", .{layer}, .u8, &.{ e, rows, k / 32 }),
        },
    }
}

fn loadConv(s: *Source, layer: u32, comptime role: []const u8, global: bool) !Conv {
    if (!global) return .{ .static = try s.get("model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight", .{layer}, .bf16, &.{ 1024, 1, 4 }) };
    return .{ .dynamic = .{
        .static_w = try s.get("model.llm.layers.{d}.attn." ++ role ++ "_sconv.static_w", .{layer}, .bf16, &.{ 4, 1024 }),
        .projection = try s.get("model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight_proj", .{layer}, .bf16, &.{ 32, 2304 }),
    } };
}

fn loadWeights(s: *Source, config: *const cfg.Config, wp: *const layer_plan.Plan) !Weights {
    var w: Weights = undefined;
    w.embed = try s.getEmbed("model.llm.embed.weight", .bf16, &.{ 201024, 2304 });
    w.embed_norm = try s.get("model.llm.embed_norm.weight", .{}, .bf16, &.{2304});
    w.norm = try s.get("model.llm.norm.weight", .{}, .bf16, &.{2304});
    w.unembed_q8 = null;
    if (s.options.dense == .q8 or s.options.dense == .nv4 or s.options.dense == .q4) {
        w.unembed_q8 = try s.getDense("model.llm.unembed.weight", .{}, &.{ 201024, 2304 }, config.text.unpadded_vocab);
    }
    if ((s.options.dense == .q8 or s.options.dense == .nv4 or s.options.dense == .q4) and !s.options.keep_bf16_head) {
        // the head's BF16 is validated, not uploaded
        _ = try s.require("model.llm.unembed.weight", .bf16, &.{ 201024, 2304 });
        w.unembed = .{ .buf = .{ .rt = s.l.r, .ptr = null, .len = 0 }, .bytes = 0, .dtype = .bf16 };
    } else w.unembed = try s.get("model.llm.unembed.weight", .{}, .bf16, &.{ 201024, 2304 });
    w.output_query = try s.get("model.llm.attn_res.output_query", .{}, .f32, &.{2304});
    w.cross_queries = try s.get("model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ 24, 2304 });
    w.cross_kv_queries = try s.get("model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ 24, 2304 });
    for (&w.attn_queries, 0..) |*q, i| {
        const global = cfg.isGlobal(@intCast(i + 1));
        const dims: []const usize = if (global) &.{ 3, 2304 } else &.{2304};
        q.* = try s.get("model.llm.attn_res.attn_queries.{d}", .{i}, .f32, dims);
    }
    for (&w.mlp_queries, 0..) |*q, i| q.* = try s.get("model.llm.attn_res.mlp_queries.{d}", .{i}, .f32, &.{2304});
    for (&w.layers, 0..) |*l, i| {
        const layer: u32 = @intCast(i);
        const p = wp.at(layer);
        const global = p.kind == .global;
        const q_width: usize = @as(usize, p.heads.q) * p.heads.dim;
        // a sliding layer's three input norms are one stored tensor: uploaded once, shared by q, k and v
        const local_norm = if (!global) try s.get("model.llm.layers.{d}.attn_norm.weight", .{layer}, .bf16, &.{2304}) else undefined;
        l.attn = .{
            .wq = try s.getProj("wq", "model.llm.layers.{d}.attn.wq_du.weight", .{layer}, &.{ q_width, 2304 }),
            .wk = try s.getProj("wk", "model.llm.layers.{d}.attn.wk_dv.weight", .{layer}, &.{ 1024, 2304 }),
            .wv = try s.getProj("wv", "model.llm.layers.{d}.attn.wv_dv.weight", .{layer}, &.{ 1024, 2304 }),
            .wr = try s.getProj("wr", "model.llm.layers.{d}.attn.wr_du.weight", .{layer}, &.{ 384, 2304 }),
            .wo = try s.getProj("wo", "model.llm.layers.{d}.attn.wo_ud.weight", .{layer}, &.{ 2304, q_width }),
            .gate = try s.getProj("gate", "model.llm.layers.{d}.attn.gate_attn.weight", .{layer}, &.{ q_width, 2304 }),
            .sink = try s.get("model.llm.layers.{d}.attn.attn_sink_proj.weight", .{layer}, .bf16, &.{ 24, 2304 }),
            .q_scale = try s.get("model.llm.layers.{d}.attn.q_l2_clip.scale", .{layer}, .bf16, &.{ 24, 128 }),
            .rel_scale = try s.get("model.llm.layers.{d}.attn.rel_l2_clip.scale", .{layer}, .bf16, &.{ 24, 16 }),
            .rel_projection = try s.get("model.llm.layers.{d}.attn.rel_logits_proj.proj", .{layer}, .bf16, &.{ 16, p.extent }),
            .k_conv = try loadConv(s, layer, "k", global),
            .v_conv = try loadConv(s, layer, "v", global),
            .q_norm = if (global) try s.get("model.llm.layers.{d}.q_input_norm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
            .k_norm = if (global) try s.get("model.llm.layers.{d}.k_input_norm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
            .v_norm = if (global) try s.get("model.llm.layers.{d}.v_input_norm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
        };
        l.attn_conv = try s.get("model.llm.layers.{d}.attn_sconv.weight", .{layer}, .bf16, &.{ 2304, 1, 4 });
        l.mlp_norm = try s.get("model.llm.layers.{d}.mlp_norm.weight", .{layer}, .bf16, &.{2304});
        l.mlp_conv = try s.get("model.llm.layers.{d}.mlp_sconv.weight", .{layer}, .bf16, &.{ 2304, 1, 4 });
        l.router_weight = try s.get("model.llm.layers.{d}.mlp.gate.weight", .{layer}, .bf16, &.{ 128, 2304 });
        l.router_bias = try s.get("model.llm.layers.{d}.mlp.gate.bias", .{layer}, .f32, &.{128});
        l.gu = try matrix(s, layer, "w13", p.precision, 2304, 2304);
        l.down = try matrix(s, layer, "w2", p.precision, 2304, 1152);
        l.controllers = null;
        if (config.text.loop) |lp| if (layer >= lp.core_start and layer < lp.core_end) {
            var controllers: [4]Controller = undefined;
            inline for (.{ "attn", "k", "v", "mlp" }, 0..) |role, j| controllers[j] = .{
                .down = try s.get("model.llm.layers.{d}.controllers." ++ role ++ ".down_weight", .{layer}, .bf16, &.{ 64, 4608 }),
                .forget = try s.get("model.llm.layers.{d}.controllers." ++ role ++ ".forget_weight", .{layer}, .bf16, &.{ 2304, 64 }),
                .gain = try s.get("model.llm.layers.{d}.controllers." ++ role ++ ".gain_weight", .{layer}, .bf16, &.{ 2304, 64 }),
            };
            l.controllers = controllers;
        };
    }
    return w;
}

/// Mini's expert format by layer: NVFP4 for layers 0 to 19, MXFP8 for 20 to 23. `matrix` rechecks every stored tensor.
pub fn precisions() [24]layer_plan.Precision {
    var out: [24]layer_plan.Precision = undefined;
    for (&out, 0..) |*p, i| p.* = if (i < 20) .nvfp4 else .mxfp8;
    return out;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    r: *rt.Runtime,
    config: cfg.Config,
    weights: Weights,
    weight_plan: layer_plan.Plan,
    /// The device context the kernels were made on; set with `pipelines` by `loadKernels`.
    ctx: ?*const xpu.Context = null,
    pipelines: ?Pipelines = null,
    options: Options,
    /// Zeros and the prompt plan for NVFP4 dense launches. Null unless `options.dense` is `.nv4`.
    nv4_aux: ?rt.Buffer = null,
    /// Every device or pinned host buffer the weights own, each once (shared tensors appear once).
    owned: std.ArrayList(rt.Buffer),
    /// Bytes of device memory the weights hold, and bytes of pinned host memory (the embedding with `embed_host`).
    device_bytes: u64,
    host_bytes: u64,

    /// Reads `dir`'s config and checkpoint, admits Mini, validates every tensor, then uploads. `r` must outlive the model.
    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, dir: []const u8, requested: Options) !*Model {
        const options = withEnvironment(requested);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .r = r, .config = undefined, .weights = undefined, .weight_plan = undefined, .options = options, .owned = .empty, .device_bytes = 0, .host_bytes = 0 };
        errdefer m.freeBuffers();
        const text = try loader.readFile(gpa, dir, "config.json");
        defer gpa.free(text);
        var why: cfg.Why = .{};
        m.config = cfg.parse(gpa, text, &why) catch |err| {
            std.log.err("inkling2_loop: {s}", .{why.text()});
            return err;
        };
        try admitConfig(&m.config);
        var l = try loader.Loader.init(gpa, r, dir);
        defer l.deinit();
        var prec = precisions();
        m.weight_plan = try layer_plan.build(&m.config.text, &prec);
        if (m.weight_plan.n_visits != 48) return error.UnsupportedInkling2Loop;
        var src: Source = .{ .l = &l, .owned = &m.owned, .gpa = gpa, .dry = true, .embed_host = options.embed_host, .options = options };
        defer src.freeScratch();
        _ = try loadWeights(&src, &m.config, &m.weight_plan);
        src.dry = false;
        m.weights = try loadWeights(&src, &m.config, &m.weight_plan);
        try r.sync();
        if (options.dense == .nv4) {
            try m.owned.ensureUnusedCapacity(m.gpa, 1);
            var aux = try r.alloc(nv4.auxBytes());
            errdefer aux.free();
            var host_aux: [nv4.auxBytes()]u8 align(4) = undefined;
            nv4.fillAux(&host_aux);
            try r.upload(aux, &host_aux);
            try r.sync();
            m.owned.appendAssumeCapacity(aux);
            m.nv4_aux = aux;
        }
        for (m.owned.items) |b| if (b.host) {
            m.host_bytes += b.len;
        } else {
            m.device_bytes += b.len;
        };
        return m;
    }

    /// Builds the production kernels on `ctx` (the context `r` runs on); the forward needs them.
    pub fn loadKernels(m: *Model, ctx: *const xpu.Context) !void {
        if (m.pipelines != null) return;
        m.pipelines = try Pipelines.load(ctx);
        m.ctx = ctx;
    }

    fn freeBuffers(m: *Model) void {
        for (m.owned.items) |*b| b.free();
        m.owned.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        if (m.pipelines) |*p| p.deinit();
        m.freeBuffers();
        m.gpa.destroy(m);
    }
};

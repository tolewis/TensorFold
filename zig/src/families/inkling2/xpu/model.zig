//! Inkling 2 Mini's resident text model on the XPU backend: checkpoint weights validated by name, dtype and shape,
//! then uploaded to device buffers as stored (experts stay NVFP4 or MXFP8; nothing is decoded or repacked here).
const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const st = @import("core").safetensors;

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
    unembed: Tensor,
    output_query: Tensor,
    cross_queries: Tensor,
    cross_kv_queries: Tensor,
    attn_queries: [23]Tensor,
    mlp_queries: [24]Tensor,
    layers: [24]LayerWeights,
};

/// Where the compiled kernels of the XPU slices go. The lead fills one field per slice when X1 to X4 land, each
/// built from that slice's embedded image the way `metal/model.zig` builds its pipelines. Nothing here yet.
pub const Pipelines = struct {
    /// X1: attention, decode and prompt paths.
    attention: void = {},
    /// X2: glue (embedding, block attention residual, convolutions, router, SwiGLU, pick, final cap).
    glue: void = {},
    /// X3: experts decoded as stored (NVFP4 and MXFP8).
    experts: void = {},
    /// X4: loop controllers and the cross-pass message.
    loop: void = {},
    /// X2: runtime copies, widen, partial sums and the tap capture.
    runtime: void = {},
};

pub const Options = struct {
    /// Keep the token embedding in pinned host memory instead of VRAM (the plan's S2). Default: on the device.
    embed_host: bool = false,
};

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
    w.unembed = try s.get("model.llm.unembed.weight", .{}, .bf16, &.{ 201024, 2304 });
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
            .wq = try s.get("model.llm.layers.{d}.attn.wq_du.weight", .{layer}, .bf16, &.{ q_width, 2304 }),
            .wk = try s.get("model.llm.layers.{d}.attn.wk_dv.weight", .{layer}, .bf16, &.{ 1024, 2304 }),
            .wv = try s.get("model.llm.layers.{d}.attn.wv_dv.weight", .{layer}, .bf16, &.{ 1024, 2304 }),
            .wr = try s.get("model.llm.layers.{d}.attn.wr_du.weight", .{layer}, .bf16, &.{ 384, 2304 }),
            .wo = try s.get("model.llm.layers.{d}.attn.wo_ud.weight", .{layer}, .bf16, &.{ 2304, q_width }),
            .gate = try s.get("model.llm.layers.{d}.attn.gate_attn.weight", .{layer}, .bf16, &.{ q_width, 2304 }),
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
    pipelines: Pipelines = .{},
    options: Options,
    /// Every device or pinned host buffer the weights own, each once (shared tensors appear once).
    owned: std.ArrayList(rt.Buffer),
    /// Bytes of device memory the weights hold, and bytes of pinned host memory (the embedding with `embed_host`).
    device_bytes: u64,
    host_bytes: u64,

    /// Reads `dir`'s config and checkpoint, admits Mini, validates every tensor, then uploads. `r` must outlive the model.
    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, dir: []const u8, options: Options) !*Model {
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
        var src: Source = .{ .l = &l, .owned = &m.owned, .gpa = gpa, .dry = true, .embed_host = options.embed_host };
        _ = try loadWeights(&src, &m.config, &m.weight_plan);
        src.dry = false;
        m.weights = try loadWeights(&src, &m.config, &m.weight_plan);
        try r.sync();
        for (m.owned.items) |b| if (b.host) {
            m.host_bytes += b.len;
        } else {
            m.device_bytes += b.len;
        };
        return m;
    }

    fn freeBuffers(m: *Model) void {
        for (m.owned.items) |*b| b.free();
        m.owned.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        m.freeBuffers();
        m.gpa.destroy(m);
    }
};

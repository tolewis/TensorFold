//! Validated mtp.safetensors weights and the one-time folded Markov output head, in XPU device memory.
//! Mirrors metal/dspark_weights.zig: the same names, shapes and admission rules, the same folded Phi and beta. The
//! target's unembedding is passed in as host BF16 (the engine already reads it once to load the target), so this file
//! needs no model type. Dense projections are stored as the checkpoint has them (BF16, row-major, [out][in]).
const std = @import("std");
const xpu = @import("xpu");
const st = @import("core").safetensors;
const fold = @import("dspark_fold.zig");

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const intermediate = 9216;
pub const rank = 256;
pub const vocabulary = 201024;

/// What the drafter's config must say for these kernels to be the right ones (Mini: taps at visits 5, 23, 41 and 48).
pub const Config = struct {
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    inter: u32,
    window: u32,
    block: u32,
    rank: u32,
    rope_theta: f64,
    taps_n: u32,
    taps: [4]u32,
    local: [3]bool,
};

pub fn admit(m: Config) !void {
    if (m.layers != 3 or m.q_heads != 24 or m.kv_heads != 8 or m.head_dim != 128 or m.inter != intermediate or
        m.window != 1024 or m.block != 8 or m.rank != rank or m.rope_theta != 1e6 or m.taps_n != 4 or
        !m.local[0] or m.local[1] or !m.local[2] or !std.mem.eql(u32, &m.taps, &.{ 5, 23, 41, 48 }))
        return error.UnsupportedDSparkConfig;
}

/// One tensor in device memory: BF16, as stored.
pub const Tensor = struct {
    buf: xpu.DeviceBuffer,
    rank: u8,
    shape: [st.max_rank]usize,

    pub fn bytes(t: Tensor) usize {
        return t.buf.len;
    }
    pub fn ptr(t: Tensor) !?*anyopaque {
        return t.buf.at(0);
    }
};

pub const Layer = struct {
    dflare: Tensor,
    input_norm: Tensor,
    post_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    k_norm_ctx: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_ctx: Tensor,
    v_ctx: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
};

pub const Weights = struct {
    fuse: Tensor,
    mask: Tensor,
    norm: Tensor,
    markov_norm: Tensor,
    hidden_out: Tensor,
    hidden_bias: Tensor,
    markov_w1: Tensor,
    rnn: Tensor,
    rnn_bias: Tensor,
    layers: [3]Layer,
};

pub const Loaded = struct {
    weights: Weights,
    dflare: xpu.DeviceBuffer, // four H-wide rows: three layers, then the Markov head
    phi: xpu.DeviceBuffer, // [201024,256] BF16, folded once in binary64
    beta: xpu.DeviceBuffer, // [201024] BF16
    /// Every device buffer the weights own, so `deinit` frees them once each.
    owned: std.ArrayList(xpu.DeviceBuffer),
    gpa: std.mem.Allocator,

    pub fn deinit(l: *Loaded) void {
        l.beta.free();
        l.phi.free();
        l.dflare.free();
        for (l.owned.items) |*b| b.free();
        l.owned.deinit(l.gpa);
        l.* = undefined;
    }

    /// Device bytes the drafter's weights hold for good.
    pub fn fixedBytes(l: *const Loaded) u64 {
        var n: u64 = l.dflare.len + l.phi.len + l.beta.len;
        for (l.owned.items) |b| n += b.len;
        return n;
    }
};

/// The entry named `name` if it is BF16 with exactly these dims.
fn require(e: st.Entry, dims: []const usize) !st.Entry {
    if (e.dtype != .bf16 or e.rank != dims.len) return error.DSparkTensorShape;
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (e.shape[i] != d) return error.DSparkTensorShape;
        count = std.math.mul(usize, count, d) catch return error.DSparkTensorShape;
    }
    if (e.end - e.begin != count * 2) return error.DSparkTensorShape;
    return e;
}

fn get(h: *const st.Header, name: []const u8, dims: []const usize) !st.Entry {
    const e = h.get(name) orelse {
        std.log.warn("DSpark mtp.safetensors is missing {s}", .{name});
        return error.DSparkMissingTensor;
    };
    return require(e, dims);
}

/// The names and shapes of Mini's drafter, checked against a header before any byte is read.
pub fn validate(h: *const st.Header) !void {
    var b: [128]u8 = undefined;
    inline for (.{
        .{ "fuse_weight", &[_]usize{ 4, 4 } },
        .{ "mask_embed", &[_]usize{hidden} },
        .{ "norm.weight", &[_]usize{hidden} },
        .{ "markov_head.dflare_norm.weight", &[_]usize{hidden} },
        .{ "markov_head.hidden_out.weight", &[_]usize{ hidden, rank } },
        .{ "markov_head.hidden_out.bias", &[_]usize{hidden} },
        .{ "markov_head.markov_w1.weight", &[_]usize{ vocabulary, rank } },
        .{ "markov_head.rnn_proj.weight", &[_]usize{ 3 * rank, 2 * hidden + 2 * rank } },
        .{ "markov_head.rnn_proj.bias", &[_]usize{3 * rank} },
    }) |t| _ = try get(h, t[0], t[1]);
    for (0..3) |i| {
        inline for (layer_tensors) |t| _ = try get(h, try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, t[0] }), t[1]);
    }
}

const layer_tensors = .{
    .{ "dflare_norm.weight", &[_]usize{hidden} },
    .{ "input_layernorm.weight", &[_]usize{hidden} },
    .{ "post_attention_layernorm.weight", &[_]usize{hidden} },
    .{ "self_attn.q_norm.weight", &[_]usize{128} },
    .{ "self_attn.k_norm.weight", &[_]usize{128} },
    .{ "self_attn.k_norm_ctx.weight", &[_]usize{128} },
    .{ "self_attn.q_proj.weight", &[_]usize{ query, hidden } },
    .{ "self_attn.k_proj.weight", &[_]usize{ kv, hidden } },
    .{ "self_attn.v_proj.weight", &[_]usize{ kv, hidden } },
    .{ "self_attn.k_proj_ctx.weight", &[_]usize{ kv, hidden } },
    .{ "self_attn.v_proj_ctx.weight", &[_]usize{ kv, hidden } },
    .{ "self_attn.o_proj.weight", &[_]usize{ hidden, query } },
    .{ "mlp.gate_proj.weight", &[_]usize{ intermediate, hidden } },
    .{ "mlp.up_proj.weight", &[_]usize{ intermediate, hidden } },
    .{ "mlp.down_proj.weight", &[_]usize{ hidden, intermediate } },
};

const Uploader = struct {
    file: *const st.File,
    header: *const st.Header,
    ctx: *const xpu.Context,
    s: xpu.Stream,
    out: *std.ArrayList(xpu.DeviceBuffer),
    gpa: std.mem.Allocator,

    /// The tensor's bytes, in the mapping.
    fn host(u: Uploader, e: st.Entry) []const u16 {
        const raw = u.file.map.memory[u.file.data + e.begin ..][0 .. e.end - e.begin];
        return @alignCast(std.mem.bytesAsSlice(u16, raw));
    }

    fn tensor(u: Uploader, name: []const u8, dims: []const usize) !Tensor {
        const e = try get(u.header, name, dims);
        const raw = u.file.map.memory[u.file.data + e.begin ..][0 .. e.end - e.begin];
        try u.out.ensureUnusedCapacity(u.gpa, 1);
        const buf = try xpu.DeviceBuffer.fromHost(u.ctx, u.s, raw);
        u.out.appendAssumeCapacity(buf);
        return .{ .buf = buf, .rank = e.rank, .shape = e.shape };
    }

    fn layerTensor(u: Uploader, i: usize, suffix: []const u8, dims: []const usize) !Tensor {
        var b: [128]u8 = undefined;
        return u.tensor(try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, suffix }), dims);
    }
};

/// Reads `path` (the file, or a directory holding mtp.safetensors), checks every name and shape, copies the weights to
/// the card, packs the four DFlare rows and folds Phi and beta from `unembed` (the target's [201024][2304] BF16 rows).
pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const xpu.Context, s: xpu.Stream, config: Config, path: []const u8, unembed: []const u16) !Loaded {
    try admit(config);
    if (unembed.len != @as(usize, vocabulary) * hidden) return error.DSparkFoldShape;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const file_path = if (std.mem.endsWith(u8, path, ".safetensors")) path else try std.fmt.bufPrint(&buf, "{s}/mtp.safetensors", .{path});
    var file = st.File.open(gpa, io, file_path) catch |err| {
        std.log.err("DSpark cannot open the drafter file: {s}", .{@errorName(err)});
        return error.DSparkMissingFile;
    };
    defer file.close(io);
    try validate(&file.names);

    var owned: std.ArrayList(xpu.DeviceBuffer) = .empty;
    errdefer {
        for (owned.items) |*b| b.free();
        owned.deinit(gpa);
    }
    const u: Uploader = .{ .file = &file, .header = &file.names, .ctx = ctx, .s = s, .out = &owned, .gpa = gpa };
    var layers: [3]Layer = undefined;
    for (&layers, 0..) |*l, i| {
        inline for (@typeInfo(Layer).@"struct".field_names, layer_tensors) |field, t| @field(l, field) = try u.layerTensor(i, t[0], t[1]);
    }
    const weights: Weights = .{
        .fuse = try u.tensor("fuse_weight", &.{ 4, 4 }),
        .mask = try u.tensor("mask_embed", &.{hidden}),
        .norm = try u.tensor("norm.weight", &.{hidden}),
        .markov_norm = try u.tensor("markov_head.dflare_norm.weight", &.{hidden}),
        .hidden_out = try u.tensor("markov_head.hidden_out.weight", &.{ hidden, rank }),
        .hidden_bias = try u.tensor("markov_head.hidden_out.bias", &.{hidden}),
        .markov_w1 = try u.tensor("markov_head.markov_w1.weight", &.{ vocabulary, rank }),
        .rnn = try u.tensor("markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * hidden + 2 * rank }),
        .rnn_bias = try u.tensor("markov_head.rnn_proj.bias", &.{3 * rank}),
        .layers = layers,
    };

    var packed_rows: [4 * hidden]u16 = undefined;
    for (0..3) |i| {
        var name: [128]u8 = undefined;
        const e = try get(&file.names, try std.fmt.bufPrint(&name, "layers.{d}.dflare_norm.weight", .{i}), &.{hidden});
        @memcpy(packed_rows[i * hidden ..][0..hidden], u.host(e));
    }
    @memcpy(packed_rows[3 * hidden ..][0..hidden], u.host(try get(&file.names, "markov_head.dflare_norm.weight", &.{hidden})));
    var dflare = try xpu.DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(&packed_rows));
    errdefer dflare.free();

    const phi_host = try gpa.alloc(u16, vocabulary * rank);
    defer gpa.free(phi_host);
    const beta_host = try gpa.alloc(u16, vocabulary);
    defer gpa.free(beta_host);
    try fold.fold(gpa, unembed, u.host(try get(&file.names, "markov_head.hidden_out.weight", &.{ hidden, rank })), u.host(try get(&file.names, "markov_head.hidden_out.bias", &.{hidden})), phi_host, beta_host);
    var phi = try xpu.DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(phi_host));
    errdefer phi.free();
    const beta = try xpu.DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(beta_host));
    return .{ .weights = weights, .dflare = dflare, .phi = phi, .beta = beta, .owned = owned, .gpa = gpa };
}

fn entry(dtype: st.DType, dims: []const usize) st.Entry {
    var e: st.Entry = .{ .dtype = dtype, .rank = @intCast(dims.len), .shape = @splat(0), .begin = 0, .end = 2 };
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        e.shape[i] = d;
        count *= d;
    }
    e.end = count * 2;
    return e;
}

test "a misshapen DSpark tensor has a named refusal" {
    _ = try require(entry(.bf16, &.{4}), &.{4});
    try std.testing.expectError(error.DSparkTensorShape, require(entry(.bf16, &.{4}), &.{ 2, 2 }));
    try std.testing.expectError(error.DSparkTensorShape, require(entry(.f32, &.{4}), &.{4}));
    var short = entry(.bf16, &.{4});
    short.end -= 2;
    try std.testing.expectError(error.DSparkTensorShape, require(short, &.{4}));
}

test "the drafter's config is admitted only as Mini's" {
    const mini: Config = .{ .layers = 3, .q_heads = 24, .kv_heads = 8, .head_dim = 128, .inter = intermediate, .window = 1024, .block = 8, .rank = rank, .rope_theta = 1e6, .taps_n = 4, .taps = .{ 5, 23, 41, 48 }, .local = .{ true, false, true } };
    try admit(mini);
    var other = mini;
    other.taps[1] = 22;
    try std.testing.expectError(error.UnsupportedDSparkConfig, admit(other));
    other = mini;
    other.local[1] = true;
    try std.testing.expectError(error.UnsupportedDSparkConfig, admit(other));
}

test "validate names the missing tensor and checks every shape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var h: st.Header = .empty;
    try std.testing.expectError(error.DSparkMissingTensor, validate(&h));
    const a = arena.allocator();
    try h.put(a, "fuse_weight", entry(.bf16, &.{ 4, 4 }));
    try std.testing.expectError(error.DSparkMissingTensor, validate(&h));
    try h.put(a, "fuse_weight", entry(.bf16, &.{ 4, 5 }));
    try std.testing.expectError(error.DSparkTensorShape, validate(&h));
}

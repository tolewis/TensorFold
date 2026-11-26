//! Mini's DFlash 2 drafter checkpoint (incoai/Inkling-2-Mini-DFlash2) on XPU: the config it must carry, every tensor
//! name and shape checked before a byte is read, the weights in device memory (BF16 as stored, or the dense projections
//! re-encoded to q8.zig's Q8_0 blocks with `TF_DFLASH2_Q8=1`), and the two candidate codebooks left in the mapped file
//! (the host reads a few rows of them a round).
const std = @import("std");
const xpu = @import("xpu");
const st = @import("core").safetensors;
const q8 = @import("q8.zig");

pub const hidden = 2304;
pub const query = 4096;
pub const kv = 1024;
pub const intermediate = 6912;
pub const layers = 5;
pub const rank = 256;
pub const groups = 144;
pub const dyn = 576;
/// The target's padded vocabulary (codebook rows); ids below `valid_vocabulary` are tokens.
pub const vocabulary = 201024;
pub const valid_vocabulary = 200058;
pub const tap_list = [4]u32{ 5, 23, 41, 48 };

/// Whether `dir` holds a DFlash 2 draft model (its config names the architecture).
pub fn isDraft(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}) catch return false;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return false;
    defer gpa.free(text);
    return std.mem.indexOf(u8, text, "DFlash2DraftModel") != null;
}

/// What the drafter's config must say for these kernels to be the right ones.
pub fn admit(gpa: std.mem.Allocator, text: []const u8) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return error.UnsupportedDFlash2Config;
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.UnsupportedDFlash2Config,
    };
    const cfg = switch (root.get("dflash_config") orelse return error.UnsupportedDFlash2Config) {
        .object => |o| o,
        else => return error.UnsupportedDFlash2Config,
    };
    try expectInt(root, "hidden_size", hidden);
    try expectInt(root, "num_hidden_layers", layers);
    try expectInt(root, "num_attention_heads", 32);
    try expectInt(root, "num_key_value_heads", 8);
    try expectInt(root, "head_dim", 128);
    try expectInt(root, "intermediate_size", intermediate);
    try expectInt(root, "sliding_window", 2048);
    try expectInt(root, "num_target_layers", 48);
    try expectInt(cfg, "block_size", 8);
    try expectInt(cfg, "conv_group_size", 16);
    try expectInt(cfg, "conv_kernel_size", 2);
    try expectInt(cfg, "selector_rank", rank);
    try expectInt(cfg, "selector_top_k", 16);
    try expectInt(cfg, "mask_token_id", 200056);
    const ids = switch (cfg.get("target_layer_ids") orelse return error.UnsupportedDFlash2Config) {
        .array => |a| a.items,
        else => return error.UnsupportedDFlash2Config,
    };
    if (ids.len != 4) return error.UnsupportedDFlash2Config;
    for (ids, tap_list) |v, want| {
        if (v != .integer or v.integer != want) return error.UnsupportedDFlash2Config;
    }
    const cap = cfg.get("final_logit_softcapping") orelse return error.UnsupportedDFlash2Config;
    if (!(cap == .float and cap.float == 30.0) and !(cap == .integer and cap.integer == 30)) return error.UnsupportedDFlash2Config;
    const mode = cfg.get("selector_softcap_mode") orelse return error.UnsupportedDFlash2Config;
    if (mode != .string or !std.mem.eql(u8, mode.string, "after_correction")) return error.UnsupportedDFlash2Config;
    const rope = switch (root.get("rope_parameters") orelse return error.UnsupportedDFlash2Config) {
        .object => |o| o,
        else => return error.UnsupportedDFlash2Config,
    };
    const theta = rope.get("rope_theta") orelse return error.UnsupportedDFlash2Config;
    if (!(theta == .float and theta.float == 10000.0) and !(theta == .integer and theta.integer == 10000)) return error.UnsupportedDFlash2Config;
}

fn expectInt(o: std.json.ObjectMap, key: []const u8, want: i64) !void {
    const v = o.get(key) orelse return error.UnsupportedDFlash2Config;
    if (v != .integer or v.integer != want) return error.UnsupportedDFlash2Config;
}

/// One tensor in device memory: BF16 as stored, a matrix [n][k] as q8.zig's codes then scales (`q8`), or the same codes
/// repacked to q8.zig's DPAS tiles (`dpas`).
pub const Tensor = struct { buf: xpu.DeviceBuffer, q8: bool = false, dpas: bool = false };

pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    attn_kernel: Tensor, // attention_conv.kernel_projection [576][2304]
    attn_base: Tensor, // attention_conv.base_kernel [2][2][2304]
    mlp_kernel: Tensor,
    mlp_base: Tensor,
};

pub const Weights = struct {
    fc: Tensor, // [2304][9216]
    hidden_norm: Tensor,
    norm: Tensor,
    selector: Tensor, // candidate_selector.hidden_projection [256][2304]
    layers: [layers]Layer,
};

pub const Loaded = struct {
    weights: Weights,
    /// The mapped file: the codebooks below point into it.
    file: st.File,
    pred: []const u16, // candidate_selector.predecessor_codebook [201024][256] BF16
    succ: []const u16,
    owned: std.ArrayList(xpu.DeviceBuffer),
    gpa: std.mem.Allocator,
    io: std.Io,

    pub fn deinit(l: *Loaded) void {
        for (l.owned.items) |*b| b.free();
        l.owned.deinit(l.gpa);
        l.file.close(l.io);
        l.* = undefined;
    }

    /// Device bytes the drafter's weights hold for good.
    pub fn fixedBytes(l: *const Loaded) u64 {
        var n: u64 = 0;
        for (l.owned.items) |b| n += b.len;
        return n;
    }
};

fn require(e: st.Entry, dims: []const usize) !st.Entry {
    if (e.dtype != .bf16 or e.rank != dims.len) return error.DFlash2TensorShape;
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (e.shape[i] != d) return error.DFlash2TensorShape;
        count = std.math.mul(usize, count, d) catch return error.DFlash2TensorShape;
    }
    if (e.end - e.begin != count * 2) return error.DFlash2TensorShape;
    return e;
}

fn get(h: *const st.Header, name: []const u8, dims: []const usize) !st.Entry {
    const e = h.get(name) orelse {
        std.log.warn("DFlash 2 checkpoint is missing {s}", .{name});
        return error.DFlash2MissingTensor;
    };
    return require(e, dims);
}

const layer_tensors = .{
    .{ "input_layernorm.weight", &[_]usize{hidden} },
    .{ "post_attention_layernorm.weight", &[_]usize{hidden} },
    .{ "self_attn.q_norm.weight", &[_]usize{128} },
    .{ "self_attn.k_norm.weight", &[_]usize{128} },
    .{ "self_attn.q_proj.weight", &[_]usize{ query, hidden } },
    .{ "self_attn.k_proj.weight", &[_]usize{ kv, hidden } },
    .{ "self_attn.v_proj.weight", &[_]usize{ kv, hidden } },
    .{ "self_attn.o_proj.weight", &[_]usize{ hidden, query } },
    .{ "mlp.gate_proj.weight", &[_]usize{ intermediate, hidden } },
    .{ "mlp.up_proj.weight", &[_]usize{ intermediate, hidden } },
    .{ "mlp.down_proj.weight", &[_]usize{ hidden, intermediate } },
    .{ "attention_conv.kernel_projection.weight", &[_]usize{ dyn, hidden } },
    .{ "attention_conv.base_kernel", &[_]usize{ 2, 2, hidden } },
    .{ "mlp_conv.kernel_projection.weight", &[_]usize{ dyn, hidden } },
    .{ "mlp_conv.base_kernel", &[_]usize{ 2, 2, hidden } },
};

/// The names and shapes of Mini's DFlash 2 drafter, checked against a header before any byte is read.
pub fn validate(h: *const st.Header) !void {
    var b: [128]u8 = undefined;
    inline for (.{
        .{ "fc.weight", &[_]usize{ hidden, 4 * hidden } },
        .{ "hidden_norm.weight", &[_]usize{hidden} },
        .{ "norm.weight", &[_]usize{hidden} },
        .{ "candidate_selector.hidden_projection.weight", &[_]usize{ rank, hidden } },
        .{ "candidate_selector.predecessor_codebook", &[_]usize{ vocabulary, rank } },
        .{ "candidate_selector.successor_codebook", &[_]usize{ vocabulary, rank } },
    }) |t| _ = try get(h, t[0], t[1]);
    for (0..layers) |i| {
        inline for (layer_tensors) |t| _ = try get(h, try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, t[0] }), t[1]);
    }
}

const Uploader = struct {
    file: *const st.File,
    ctx: *const xpu.Context,
    s: xpu.Stream,
    out: *std.ArrayList(xpu.DeviceBuffer),
    gpa: std.mem.Allocator,
    /// Encode a two-dimensional projection to Q8_0 blocks instead of uploading its bf16 bytes.
    q8: bool,
    /// The same blocks repacked to the DPAS tiles (`dense_q8_dpas`), for the drafter's int8 path.
    dpas: bool,

    fn tensor(u: Uploader, name: []const u8, dims: []const usize) !Tensor {
        if (u.q8 and dims.len == 2 and q8.admits(dims[1])) return if (u.dpas) u.dpasTensor(name, dims) else u.q8Tensor(name, dims);
        const e = try get(&u.file.names, name, dims);
        const raw = u.file.map.memory[u.file.data + e.begin ..][0 .. e.end - e.begin];
        try u.out.ensureUnusedCapacity(u.gpa, 1);
        const buf = try xpu.DeviceBuffer.fromHost(u.ctx, u.s, raw);
        u.out.appendAssumeCapacity(buf);
        return .{ .buf = buf };
    }

    /// A matrix [n][k] of the mapped file encoded to q8.zig's blocks: codes, then the fp16 scales, one buffer.
    fn q8Tensor(u: Uploader, name: []const u8, dims: []const usize) !Tensor {
        const e = try get(&u.file.names, name, dims);
        const n = dims[0];
        const k = dims[1];
        const raw = u.file.map.memory[u.file.data + e.begin ..][0 .. e.end - e.begin];
        const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
        const codes = try u.gpa.alloc(u8, n * k);
        defer u.gpa.free(codes);
        const scales = try u.gpa.alloc(u16, n * (k / q8.block));
        defer u.gpa.free(scales);
        q8.encodeRows(src, n, k, codes, scales);
        try u.out.ensureUnusedCapacity(u.gpa, 1);
        var buf = try xpu.DeviceBuffer.alloc(u.ctx, q8.bytes(n, k));
        errdefer buf.free();
        try buf.upload(u.s, 0, codes);
        try buf.upload(u.s, q8.scaleOffset(n, k), std.mem.sliceAsBytes(scales));
        u.out.appendAssumeCapacity(buf);
        return .{ .buf = buf, .q8 = true };
    }

    /// A matrix [n][k] of the mapped file as q8.zig's DPAS tiles: 16-column tiles of 512 code bytes a block, then their
    /// scales (`q8.repackDpas`).
    fn dpasTensor(u: Uploader, name: []const u8, dims: []const usize) !Tensor {
        const e = try get(&u.file.names, name, dims);
        const n = dims[0];
        const k = dims[1];
        const raw = u.file.map.memory[u.file.data + e.begin ..][0 .. e.end - e.begin];
        const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
        const nb = k / q8.block;
        const tile = try u.gpa.alloc(u8, q8.dpasBytes(16, k));
        defer u.gpa.free(tile);
        try u.out.ensureUnusedCapacity(u.gpa, 1);
        var buf = try xpu.DeviceBuffer.alloc(u.ctx, q8.dpasBytes(n, k));
        errdefer buf.free();
        const sc_base = q8.dpasTiles(n) * nb * 512;
        const codes = try u.gpa.alloc(u8, 16 * k);
        defer u.gpa.free(codes);
        const scales = try u.gpa.alloc(u16, 16 * nb);
        defer u.gpa.free(scales);
        var row: usize = 0;
        var ti: usize = 0;
        while (row < n) {
            const m = @min(16, n - row);
            q8.encodeRows(src[row * k ..][0 .. m * k], m, k, codes[0 .. m * k], scales[0 .. m * nb]);
            q8.repackDpas(codes[0 .. m * k], scales[0 .. m * nb], m, k, tile);
            try buf.upload(u.s, ti * nb * 512, tile[0 .. nb * 512]);
            try buf.upload(u.s, sc_base + ti * nb * 32, tile[nb * 512 ..][0 .. nb * 32]);
            row += m;
            ti += 1;
        }
        u.out.appendAssumeCapacity(buf);
        return .{ .buf = buf, .q8 = true, .dpas = true };
    }

    fn layerTensor(u: Uploader, i: usize, suffix: []const u8, dims: []const usize) !Tensor {
        var b: [128]u8 = undefined;
        return u.tensor(try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, suffix }), dims);
    }
};

/// Reads `dir` (config.json, model.safetensors), checks every name and shape, and copies the weights to the card.
pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const xpu.Context, s: xpu.Stream, dir: []const u8) !Loaded {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir});
    const text = std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20)) catch |err| {
        std.log.err("DFlash 2 cannot read {s}: {s}", .{ cfg_path, @errorName(err) });
        return error.DFlash2MissingFile;
    };
    defer gpa.free(text);
    try admit(gpa, text);
    const file_path = try std.fmt.bufPrint(&buf, "{s}/model.safetensors", .{dir});
    var file = st.File.open(gpa, io, file_path) catch |err| {
        std.log.err("DFlash 2 cannot open the drafter file: {s}", .{@errorName(err)});
        return error.DFlash2MissingFile;
    };
    errdefer file.close(io);
    try validate(&file.names);

    var owned: std.ArrayList(xpu.DeviceBuffer) = .empty;
    errdefer {
        for (owned.items) |*b| b.free();
        owned.deinit(gpa);
    }
    const dense_q8 = if (std.c.getenv("TF_DFLASH2_Q8")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const dpas = dense_q8 and std.c.getenv("TF_DFLASH2_DPAS") != null;
    const u: Uploader = .{ .file = &file, .ctx = ctx, .s = s, .out = &owned, .gpa = gpa, .q8 = dense_q8, .dpas = dpas };
    var ls: [layers]Layer = undefined;
    for (&ls, 0..) |*l, i| {
        inline for (@typeInfo(Layer).@"struct".field_names, layer_tensors) |field, t| @field(l, field) = try u.layerTensor(i, t[0], t[1]);
    }
    const weights: Weights = .{
        .fc = try u.tensor("fc.weight", &.{ hidden, 4 * hidden }),
        .hidden_norm = try u.tensor("hidden_norm.weight", &.{hidden}),
        .norm = try u.tensor("norm.weight", &.{hidden}),
        .selector = try u.tensor("candidate_selector.hidden_projection.weight", &.{ rank, hidden }),
        .layers = ls,
    };
    const pe = try get(&file.names, "candidate_selector.predecessor_codebook", &.{ vocabulary, rank });
    const se = try get(&file.names, "candidate_selector.successor_codebook", &.{ vocabulary, rank });
    const pred: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, file.map.memory[file.data + pe.begin ..][0 .. pe.end - pe.begin]));
    const succ: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, file.map.memory[file.data + se.begin ..][0 .. se.end - se.begin]));
    file.adviseRandom(std.mem.sliceAsBytes(pred));
    file.adviseRandom(std.mem.sliceAsBytes(succ));
    return .{ .weights = weights, .file = file, .pred = pred, .succ = succ, .owned = owned, .gpa = gpa, .io = io };
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

test "a misshapen DFlash 2 tensor has a named refusal" {
    _ = try require(entry(.bf16, &.{4}), &.{4});
    try std.testing.expectError(error.DFlash2TensorShape, require(entry(.bf16, &.{4}), &.{ 2, 2 }));
    try std.testing.expectError(error.DFlash2TensorShape, require(entry(.f32, &.{4}), &.{4}));
}

test "validate names the missing tensor" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var h: st.Header = .empty;
    try std.testing.expectError(error.DFlash2MissingTensor, validate(&h));
    try h.put(arena.allocator(), "fc.weight", entry(.bf16, &.{ hidden, 4 * hidden + 1 }));
    try std.testing.expectError(error.DFlash2TensorShape, validate(&h));
}

test "the shipped config is admitted, and a changed tap is refused" {
    const text =
        \\{"hidden_size":2304,"num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
        \\"intermediate_size":6912,"sliding_window":2048,"num_target_layers":48,"rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
        \\"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"final_logit_softcapping":30.0,"mask_token_id":200056,
        \\"selector_rank":256,"selector_softcap_mode":"after_correction","selector_top_k":16,"target_layer_ids":[5,23,41,48]}}
    ;
    try admit(std.testing.allocator, text);
    var changed = try std.testing.allocator.dupe(u8, text);
    defer std.testing.allocator.free(changed);
    const at = std.mem.indexOf(u8, changed, "[5,23,41,48]").?;
    changed[at + 3] = '4';
    try std.testing.expectError(error.UnsupportedDFlash2Config, admit(std.testing.allocator, changed));
}

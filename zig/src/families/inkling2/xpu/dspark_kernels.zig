//! Mini DSpark's fixed-shape arithmetic on XPU: the eleven OpenCL C kernels of kernels/xpu/inkling2/dspark.cl, with the
//! launch arguments and grids that mirror the Metal host (metal/dspark_kernels.zig). Buffers are device addresses.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden: u32 = 2304;
pub const query_width: u32 = 3072;
pub const kv_width: u32 = 1024;
pub const block_rows: u32 = 8;
pub const rank: u32 = 256;
pub const rnn_width: u32 = 768;
pub const intermediate: u32 = 9216;
pub const vocab: u32 = 200058;
/// The most rows of y the base projection takes at once (the block's seven draft rows, with one spare).
pub const base_rows_max: u32 = 8;

/// A device address (what `xpu.DeviceBuffer.at` and the contract rig hand out).
pub const Ref = ?*anyopaque;

/// Round through binary32 and nearest-even BF16, returning an exactly representable FP64 value.
pub fn roundBf16(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x)) @truncate((bits >> 16) | 0x40) else @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}

/// Rotate-half RoPE partner is d+64 below the midpoint and d-64 above it.
pub fn ropeRotateHalf(x: f32, other: f32, d: u32, angle: f32) f32 {
    return if (d < 64) x * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x * @cos(angle);
}

/// Every recurrent blend operation rounds independently to model precision.
pub fn recurrentBlend(gamma: f32, candidate: f32, previous: f32) f64 {
    const keep: f32 = @floatCast(roundBf16(@as(f32, 1.0 - gamma)));
    const left: f32 = @floatCast(roundBf16(@as(f32, gamma * previous)));
    const right: f32 = @floatCast(roundBf16(@as(f32, keep * candidate)));
    return roundBf16(@as(f32, left + right));
}

/// Folded bias and base each round before the target-style softcap.
pub fn foldedLogit(base_value: f32, beta: f32, dot: f32) f64 {
    const bias: f32 = @floatCast(roundBf16(@as(f32, beta + dot)));
    const logit: f32 = @floatCast(roundBf16(@as(f32, base_value + bias)));
    return roundBf16(@as(f32, 30.0) * std.math.tanh(logit / @as(f32, 30.0)));
}

pub const Kind = enum {
    init_rows,
    fuse_taps,
    block_qkv,
    context_kv,
    attention,
    residual_norm,
    swiglu,
    kappa,
    recurrent,
    base,
    logits,

    /// The kernel's name in the production image.
    pub fn symbol(k: Kind) [:0]const u8 {
        return switch (k) {
            inline else => |v| "tf_ink2_dspark_" ++ @tagName(v),
        };
    }
    /// The same kernel with its one planted arithmetic mistake, in the contract's control image.
    pub fn controlSymbol(k: Kind) [:0]const u8 {
        return switch (k) {
            inline else => |v| "tf_ctl_ink2_dspark_" ++ @tagName(v),
        };
    }
    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("xpu inkling2 dspark {s} {s}", .{ @tagName(v), comptime v.symbol() }),
        };
    }
    /// Work-items a work-group: the kernel's reqd_work_group_size.
    pub fn threads(k: Kind) u32 {
        return switch (k) {
            .block_qkv, .context_kv, .attention => 128,
            else => 256,
        };
    }
    /// Arguments the kernel takes, as its OpenCL C parameter list counts them.
    pub fn args(k: Kind) u32 {
        return switch (k) {
            .init_rows => 3,
            .fuse_taps => 6,
            .block_qkv => 9,
            .context_kv => 8,
            .attention => 11,
            .residual_norm => 6,
            .swiglu => 4,
            .kappa => 6,
            .recurrent => 8,
            .base => 5,
            .logits => 6,
        };
    }
};

/// The kind a kernel symbol of either image names, and the number of arguments it takes.
pub fn argsOf(symbol: []const u8) ?u32 {
    inline for (std.enums.values(Kind)) |k| {
        if (std.mem.eql(u8, symbol, comptime k.symbol()) or std.mem.eql(u8, symbol, comptime k.controlSymbol())) return k.args();
    }
    return null;
}

/// The eleven kernels of one loaded image, each at its own work-group size.
pub const Kernels = struct {
    init_rows: xpu.Kernel,
    fuse_taps: xpu.Kernel,
    block_qkv: xpu.Kernel,
    context_kv: xpu.Kernel,
    attention: xpu.Kernel,
    residual_norm: xpu.Kernel,
    swiglu: xpu.Kernel,
    kappa: xpu.Kernel,
    recurrent: xpu.Kernel,
    base: xpu.Kernel,
    logits: xpu.Kernel,

    /// Creates every kernel of `module`, the production one or the control image's planted copy.
    pub fn load(module: xpu.Module, control: bool) !Kernels {
        var out: Kernels = undefined;
        var made: usize = 0;
        errdefer inline for (@typeInfo(Kernels).@"struct".field_names, 0..) |name, i| {
            if (made > i) @field(out, name).deinit();
        };
        inline for (@typeInfo(Kernels).@"struct".field_names, 0..) |name, i| {
            const kind: Kind = @fromBackingInt(@intCast(i));
            const symbol = if (control) kind.controlSymbol() else kind.symbol();
            @field(out, name) = try module.kernel(symbol, .{ kind.threads(), 1, 1 });
            made += 1;
        }
        return out;
    }

    pub fn deinit(k: *Kernels) void {
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| @field(k, name).deinit();
        k.* = undefined;
    }
};

fn blocks(count: usize, width: usize) u32 {
    return @intCast((count + width - 1) / width);
}

/// Row zero copies the target's normalized anchor; the other seven rows copy the raw mask embedding.
pub fn initializeRows(k: *const Kernels, s: xpu.Stream, anchor: Ref, mask_embed: Ref, out: Ref) !void {
    try xpu.launch.dispatch(k.init_rows, s, .{ block_rows, 1, 1 }, .{ anchor, mask_embed, out });
}

/// Input taps are [row][tap][H], output c is [fusion][row][H], and newest_f is one [H] row.
pub fn fuseTaps(k: *const Kernels, s: xpu.Stream, taps: Ref, fuse_weight: Ref, norm_weight: Ref, c: Ref, newest_f: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(k.fuse_taps, s, .{ 4 * rows, 1, 1 }, .{ taps, fuse_weight, norm_weight, c, newest_f, rows });
}

/// Raw projections become per-head RMS-normalized rotate-half Q/K and an unchanged BF16 V.
pub fn blockQkv(k: *const Kernels, s: xpu.Stream, q_raw: Ref, k_raw: Ref, v_raw: Ref, q_norm: Ref, k_norm: Ref, q: Ref, kk: Ref, v: Ref, base_position: u32) !void {
    try xpu.launch.dispatch(k.block_qkv, s, .{ block_rows * 40, 1, 1 }, .{ q_raw, k_raw, v_raw, q_norm, k_norm, q, kk, v, base_position });
}

/// Context K/V is written at absolute-position modulo ring_capacity for committed rows only.
pub fn commitContext(k: *const Kernels, s: xpu.Stream, k_raw: Ref, v_raw: Ref, k_norm_ctx: Ref, keys: Ref, values: Ref, base_position: u32, rows: u32, ring_capacity: u32) !void {
    std.debug.assert(rows > 0 and ring_capacity > 0);
    try xpu.launch.dispatch(k.context_kv, s, .{ rows * 8, 1, 1 }, .{ k_raw, v_raw, k_norm_ctx, keys, values, base_position, rows, ring_capacity });
}

/// Scores is FP32 scratch sized 8*24*(retained_context+8); global layers retain all context.
pub fn attend(k: *const Kernels, s: xpu.Stream, q: Ref, block_k: Ref, block_v: Ref, context_k: Ref, context_v: Ref, scores: Ref, out: Ref, committed: u32, ring_capacity: u32, local: bool, from: u32) !void {
    std.debug.assert(ring_capacity > 0 and from <= committed);
    try xpu.launch.dispatch(k.attention, s, .{ block_rows * 24, 1, 1 }, .{ q, block_k, block_v, context_k, context_v, scores, out, committed, ring_capacity, @as(u32, @intFromBool(local)), from });
}

/// Normalization consumes the FP32 residual sum, independently of the rounded BF16 residual store.
pub fn residualNorm(k: *const Kernels, s: xpu.Stream, x: Ref, residual: Ref, weight: Ref, new_residual: Ref, normalized: Ref, rows: u32, init: bool) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(k.residual_norm, s, .{ rows, 1, 1 }, .{ x, residual, weight, new_residual, normalized, @as(u32, @intFromBool(init)) });
}

/// Gate and up projections are BF16 rows with 9216 entries; only this SwiGLU product is stored here.
pub fn swiglu(k: *const Kernels, s: xpu.Stream, gate: Ref, up: Ref, out: Ref, count: u32) !void {
    std.debug.assert(count == 8 * intermediate);
    try xpu.launch.dispatch(k.swiglu, s, .{ blocks(count, 256), 1, 1 }, .{ gate, up, out, count });
}

/// Kappa is FP32 [rows][768], using RNN columns 512..2815 for y and 2816..5119 for newest_f.
pub fn kappa(k: *const Kernels, s: xpu.Stream, y: Ref, newest_f: Ref, rnn_weight: Ref, rnn_bias: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0 and rows <= 7);
    try xpu.launch.dispatch(k.kappa, s, .{ blocks(@as(usize, rows) * rnn_width, 256), 1, 1 }, .{ y, newest_f, rnn_weight, rnn_bias, out, rows });
}

/// One recurrent step writes rounded Z, the next BF16 state and BF16 lambda for rows 1..7 in turn.
pub fn recurrentStep(k: *const Kernels, s: xpu.Stream, state: Ref, kappa_row: Ref, rnn_weight: Ref, markov_w1: Ref, z: Ref, next_state: Ref, lambda: Ref, previous_token: u32) !void {
    try xpu.launch.dispatch(k.recurrent, s, .{ 1, 1, 1 }, .{ state, kappa_row, rnn_weight, markov_w1, z, next_state, lambda, previous_token });
}

/// Raw U*y rows round once to BF16, with no target unembedding softcap.
pub fn base(k: *const Kernels, s: xpu.Stream, unembed: Ref, y: Ref, out: Ref, rows: u32, vocabulary_count: u32) !void {
    std.debug.assert(rows > 0 and rows <= base_rows_max and vocabulary_count > 0 and vocabulary_count <= vocab);
    try xpu.launch.dispatch(k.base, s, .{ blocks(vocabulary_count, 256), 1, 1 }, .{ unembed, y, out, rows, vocabulary_count });
}

/// Folded Phi/beta correction and target-style 30 tanh softcap, restricted to Mini's 200058 logits.
pub fn finalLogits(k: *const Kernels, s: xpu.Stream, base_values: Ref, phi: Ref, beta: Ref, lambda: Ref, out: Ref) !void {
    try xpu.launch.dispatch(k.logits, s, .{ blocks(vocab, 256), 1, 1 }, .{ base_values, phi, beta, lambda, out, vocab });
}

test "every kernel has its own symbol in both images, and its argument count is listed" {
    for (std.enums.values(Kind)) |a| {
        try std.testing.expect(argsOf(a.symbol()).? == a.args());
        try std.testing.expect(argsOf(a.controlSymbol()).? == a.args());
        for (std.enums.values(Kind)) |b| if (a != b) {
            try std.testing.expect(!std.mem.eql(u8, a.symbol(), b.symbol()));
            try std.testing.expect(!std.mem.eql(u8, a.controlSymbol(), b.symbol()));
        };
    }
    try std.testing.expect(argsOf("attn_decode") == null);
}

test "the BF16 rounding points keep their ties to even" {
    try std.testing.expectEqual(@as(f64, 1.0), roundBf16(@as(f32, 1.00390625))); // halfway: rounds down to even
    try std.testing.expectEqual(@as(f64, 1.015625), roundBf16(@as(f32, 1.01171875))); // halfway: rounds up to even
    try std.testing.expect(std.math.isNan(roundBf16(std.math.nan(f32))));
}

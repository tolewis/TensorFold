//! Mini DFlash 2's own kernels on XPU: the five OpenCL C kernels of kernels/xpu/inkling2/dflash2.cl and their launches.
//! The projections run on the dense matvec, and the residual norm, swiglu and base projection are the DSpark kernels
//! (same width); this file holds only what they do not cover. Buffers are device addresses.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden: u32 = 2304;
pub const query_width: u32 = 4096;
pub const kv_width: u32 = 1024;
pub const intermediate: u32 = 6912;
pub const dyn_width: u32 = 576;
pub const block_rows: u32 = 8;
pub const chunk: u32 = 128;
pub const window: u32 = 2048;
pub const topk: u32 = 16;
pub const topk_groups: u32 = 256;
pub const q_heads: u32 = 32;
pub const kv_heads: u32 = 8;

/// The most 128-key chunks a block row's attention spans: 2,047 context rows and the eight block rows.
pub const max_chunks: u32 = (window - 1 + block_rows + chunk - 1) / chunk;
/// Floats one (row, key/value head, chunk) partial holds: m[4], l[4] and o[4][128].
pub const part_floats: u32 = 8 + 4 * 128;

pub const Ref = ?*anyopaque;

pub const Kind = enum {
    conv,
    block_qk,
    context_kv,
    attention,
    merge,
    topk,

    pub fn symbol(k: Kind) [:0]const u8 {
        return switch (k) {
            inline else => |v| "tf_ink2_dflash2_" ++ @tagName(v),
        };
    }
    /// The same kernel with its one planted arithmetic mistake, in the contract's control image.
    pub fn controlSymbol(k: Kind) [:0]const u8 {
        return switch (k) {
            inline else => |v| "tf_ctl_ink2_dflash2_" ++ @tagName(v),
        };
    }
    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("xpu inkling2 dflash2 {s} {s}", .{ @tagName(v), comptime v.symbol() }),
        };
    }
    /// Work-items a work-group: the kernel's reqd_work_group_size.
    pub fn threads(k: Kind) u32 {
        return switch (k) {
            .conv => 256,
            .block_qk, .context_kv, .attention, .merge => 128,
            .topk => 16,
        };
    }
    /// Arguments the kernel takes, as its OpenCL C parameter list counts them.
    pub fn args(k: Kind) u32 {
        return switch (k) {
            .conv => 4,
            .block_qk => 7,
            .context_kv => 7,
            .attention => 10,
            .merge => 5,
            .topk => 5,
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

pub const Kernels = struct {
    conv: xpu.Kernel,
    block_qk: xpu.Kernel,
    context_kv: xpu.Kernel,
    attention: xpu.Kernel,
    merge: xpu.Kernel,
    topk: xpu.Kernel,

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

/// Chunks of 128 keys a block at `committed` context rows needs for its first row (the one that sees the most).
pub fn chunksFor(committed: u32, from: u32) u32 {
    const first = @max(from, committed -| (window - 1));
    const keys = (committed - first) + block_rows;
    return (keys + chunk - 1) / chunk;
}

/// y = bf16(sum over the two taps of (base + dyn) * x), x [rows][2304], dyn at the branch's first column of [rows][576],
/// base at the branch's [2][2304] taps. The block's first row has no earlier row.
pub fn conv(k: *const Kernels, s: xpu.Stream, x: Ref, dyn: Ref, base: Ref, y: Ref, rows: u32) !void {
    std.debug.assert(rows > 0 and rows <= block_rows);
    try xpu.launch.dispatch(k.conv, s, .{ rows, (hidden + 255) / 256, 1 }, .{ x, dyn, base, y });
}

/// Per-head norm and RoPE of the block's queries and keys at positions base .. base + 7.
pub fn blockQk(k: *const Kernels, s: xpu.Stream, q_raw: Ref, k_raw: Ref, q_weight: Ref, k_weight: Ref, q: Ref, kk: Ref, base_position: u32) !void {
    try xpu.launch.dispatch(k.block_qk, s, .{ block_rows * 40, 1, 1 }, .{ q_raw, k_raw, q_weight, k_weight, q, kk, base_position });
}

/// `rows` committed rows' keys (norm, RoPE) and values into the ring at absolute position modulo ring.
pub fn commitContext(k: *const Kernels, s: xpu.Stream, k_raw: Ref, v_raw: Ref, k_weight: Ref, keys: Ref, values: Ref, base_position: u32, rows: u32, ring: u32) !void {
    std.debug.assert(rows > 0 and ring > 0);
    try xpu.launch.dispatch(k.context_kv, s, .{ rows * kv_heads, 1, 1 }, .{ k_raw, v_raw, k_weight, keys, values, base_position, ring });
}

/// The block's attention, first pass: one partial per (row, key/value head, chunk of 128 keys).
pub fn attendPartial(k: *const Kernels, s: xpu.Stream, q: Ref, block_k: Ref, block_v: Ref, context_k: Ref, context_v: Ref, part: Ref, committed: u32, ring: u32, from: u32) !void {
    const chunks = chunksFor(committed, from);
    std.debug.assert(ring > 0 and chunks <= max_chunks and from <= committed);
    try xpu.launch.dispatch(k.attention, s, .{ chunks, kv_heads, block_rows }, .{ q, block_k, block_v, context_k, context_v, part, committed, ring, from, chunks });
}

/// Second pass: the partials merge to BF16 [8][4096].
pub fn attendMerge(k: *const Kernels, s: xpu.Stream, part: Ref, out: Ref, committed: u32, from: u32) !void {
    try xpu.launch.dispatch(k.merge, s, .{ block_rows, q_heads, 1 }, .{ part, out, committed, from, chunksFor(committed, from) });
}

/// The block's attention in two passes: partials per 128-key chunk, then the merge to BF16 [8][4096].
pub fn attend(k: *const Kernels, s: xpu.Stream, q: Ref, block_k: Ref, block_v: Ref, context_k: Ref, context_v: Ref, part: Ref, out: Ref, committed: u32, ring: u32, from: u32) !void {
    std.debug.assert(ring >= window);
    try attendPartial(k, s, q, block_k, block_v, context_k, context_v, part, committed, ring, from);
    try attendMerge(k, s, part, out, committed, from);
}

/// The 16 best of the first `valid` logits of each row: per row 256 sorted lists of 16 (values f32, ids i32).
pub fn topKeep(k: *const Kernels, s: xpu.Stream, logits: Ref, stride: u32, valid: u32, rows: u32, values: Ref, ids: Ref) !void {
    std.debug.assert(rows > 0 and rows <= block_rows and valid > 0);
    try xpu.launch.dispatch(k.topk, s, .{ topk_groups, rows, 1 }, .{ logits, stride, valid, values, ids });
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

test "a block's chunk count follows the window and the first retained row" {
    try std.testing.expectEqual(@as(u32, 1), chunksFor(0, 0));
    try std.testing.expectEqual(@as(u32, 1), chunksFor(120, 0));
    try std.testing.expectEqual(@as(u32, 2), chunksFor(121, 0));
    try std.testing.expectEqual(@as(u32, max_chunks), chunksFor(100000, 0));
    try std.testing.expectEqual(@as(u32, 1), chunksFor(5000, 4990));
    try std.testing.expectEqual(@as(u32, 17), max_chunks);
}

//! One planted arithmetic or layout fault for every Inkling 2 Mini XPU kernel, made from the production source.
//!
//! A control image is the production source up to its first kernel, then for each kernel a copy with exactly one edit
//! and the name `tf_ctl_<name>`. The edit's text must appear once in the whole source and once in its kernel, so a
//! control can never drift from the kernel it breaks.
const std = @import("std");

pub const Entry = struct { symbol: []const u8, from: []const u8, to: []const u8 };

pub const glue = [_]Entry{
    .{ .symbol = "tf_inkling2_embed_norm", .from = "bf2f(weight[d]) * (bf2f(x[d]) * inverse)", .to = "bf2f(weight[0]) * (bf2f(x[d]) * inverse)" },
    .{ .symbol = "tf_inkling2_ar", .from = "lane_sum16(dots) * INK_AR_SCALE;", .to = "lane_sum16(dots);" },
    .{ .symbol = "tf_inkling2_conv_static", .from = "bf2f(weight[c * 4 + tap])", .to = "bf2f(weight[c * 4 + 3 - tap])" },
    .{ .symbol = "tf_inkling2_conv_dynamic", .from = "out, r, channels, 0, 0, projected, xs, 0);", .to = "out, r, channels, 3, 0, projected, xs, 0);" },
    .{ .symbol = "tf_inkling2_router", .from = "keys[expert] = scores[expert] + bias[expert];", .to = "keys[expert] = scores[expert];" },
    .{ .symbol = "tf_inkling2_swiglu", .from = "ink_cap(u, 25.0f)", .to = "ink_cap(g, 25.0f)" },
    .{ .symbol = "tf_inkling2_logits", .from = "ink_cap(bf2f(f2bf(dot)), 30.0f)", .to = "ink_cap(dot, 30.0f)" },
    .{ .symbol = "tf_inkling2_expert_pick", .from = "ids[r * 8 + j]", .to = "ids[r * 8 + 7 - j]" },
    .{
        .symbol = "tf_inkling2_expert_combine",
        .from = "acc += weights[r * 8 + j] * bf2f(picked[((ulong)r * 8 + j) * channels + d]);",
        .to = "acc -= weights[r * 8 + j] * bf2f(picked[((ulong)r * 8 + j) * channels + d]);",
    },
};

pub const runtime = [_]Entry{
    .{
        .symbol = "tf_ink2_copy_bf16",
        .from = "tf_ink2_copy_bf16(global const ushort *x, global ushort *y, uint count, uint boundary) {\n  const uint i = get_global_id(0);\n  if (i < count) y[i] = x[i];",
        .to = "tf_ink2_copy_bf16(global const ushort *x, global ushort *y, uint count, uint boundary) {\n  const uint i = get_global_id(0);\n  if (i < count) y[i] = 0;",
    },
    .{
        .symbol = "tf_ink2_copy_f32",
        .from = "tf_ink2_copy_f32(global const float *x, global float *y, uint count, uint boundary) {\n  const uint i = get_global_id(0);\n  if (i < count) y[i] = x[i];",
        .to = "tf_ink2_copy_f32(global const float *x, global float *y, uint count, uint boundary) {\n  const uint i = get_global_id(0);\n  if (i < count) y[i] = 0.0f;",
    },
    .{ .symbol = "tf_ink2_widen", .from = "if (i < count) y[i] = bf2f(x[i]);", .to = "if (i < count) y[i] = 2.0f * bf2f(x[i]);" },
    .{
        .symbol = "tf_ink2_attention_partial",
        .from = "blocks[(ulong)row * 4 * 2304 + (boundary - 1) * 2304 + d] = p[i];",
        .to = "blocks[(ulong)row * 2304 + (boundary - 1) * 2304 + d] = p[i];",
    },
    .{ .symbol = "tf_ink2_mlp_partial", .from = "p[i] = f2bf(t[i] + bf2f(x[i]));", .to = "p[i] = f2bf(t[i] - bf2f(x[i]));" },
    .{
        .symbol = "tf_ink2_capture_tap",
        .from = "y[(ulong)row * 4 * 2304 + boundary * 2304 + d] = x[i];",
        .to = "y[(ulong)row * 4 * 2304 + ((boundary + 1) % 4) * 2304 + d] = x[i];",
    },
};

/// prefill.cl's row-batched kernels. The edits break only the kernel they sit in.
pub const prefill = [_]Entry{
    .{ .symbol = "tf_ink2_pf_copy_rows", .from = "y[(ulong)r * ldy + i] = x[(ulong)r * ldx + i];", .to = "y[(ulong)r * ldy + i] = x[(ulong)r * ldx + (i ^ 1u)];" },
    .{ .symbol = "tf_ink2_pf_perm", .from = "x[(ulong)row * k + plane * 256 + (s >> 1) * 32 + 2 * l + (s & 1)];", .to = "x[(ulong)row * k + plane * 256 + (s >> 1) * 32 + 2 * l + (1u - (s & 1))];" },
    .{ .symbol = "tf_ink2_pf_kv_copy", .from = "vstore8(v, 0, dst + ((ulong)h * dst_ring + p % dst_ring) * PF_D + lane * 8);", .to = "vstore8(v, 0, dst + ((ulong)h * dst_ring + p % src_ring) * PF_D + lane * 8);" },
    .{ .symbol = "tf_ink2_pf_conv_static", .from = "const float4 q = sum / PF_SQRT5;", .to = "const float4 q = sum;" },
    .{ .symbol = "tf_ink2_pf_router", .from = "keys[r][t] = scores[r][t] + bias[t];", .to = "keys[r][t] = scores[r][t];" },
    .{ .symbol = "tf_ink2_pf_ar", .from = "weights[s] = dot * PF_AR_SCALE;", .to = "weights[s] = dot;" },
    .{ .symbol = "tf_ink2_pf_dense", .from = "float part = as_float(wv.x << 16) * as_float(xv.x << 16);", .to = "float part = as_float(wv.x << 16) * as_float(xv.x & 0xffff0000u);" },
    .{ .symbol = "tf_ink2_pf_messages", .from = "\n  const float l0 = pf_group_sum(dot0, red) * PF_AR_SCALE;", .to = "\n  const float l0 = pf_group_sum(dot0, red);" },
    .{ .symbol = "tf_ink2_pf_rel_table", .from = "acc += bf2f(rr[j]) * pj[j];", .to = "acc += bf2f(rr[15 - j]) * pj[j];" },
};

/// The control kernel's symbol: `tf_` becomes `tf_ctl_`.
pub fn controlSymbol(a: std.mem.Allocator, symbol: []const u8) ![]u8 {
    std.debug.assert(std.mem.startsWith(u8, symbol, "tf_"));
    return std.fmt.allocPrint(a, "tf_ctl_{s}", .{symbol["tf_".len..]});
}

/// A kernel's text: from the start of the line holding `void <symbol>(` to its closing brace in column 0.
fn span(production: []const u8, symbol: []const u8) !struct { usize, usize } {
    var buf: [96]u8 = undefined;
    const needle = try std.fmt.bufPrint(&buf, "void {s}(", .{symbol});
    if (std.mem.count(u8, production, needle) != 1) return error.KernelMissing;
    const at = std.mem.indexOf(u8, production, needle).?;
    const start = if (std.mem.lastIndexOfScalar(u8, production[0..at], '\n')) |nl| nl + 1 else 0;
    const close = std.mem.indexOfPos(u8, production, at, "\n}\n") orelse return error.KernelMissing;
    return .{ start, close + 3 };
}

/// The control image's source for `entries`: the helpers, then each kernel with its one edit, renamed.
pub fn source(a: std.mem.Allocator, production: []const u8, entries: []const Entry) ![]u8 {
    var first: usize = production.len;
    for (entries) |e| first = @min(first, (try span(production, e.symbol))[0]);
    if (std.mem.indexOf(u8, production[first..], "\nstatic ") != null) return error.HelperAfterKernel;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, production[0..first]);
    for (entries) |e| {
        const start, const end = try span(production, e.symbol);
        const text = production[start..end];
        if (std.mem.count(u8, production, e.from) != 1 or std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
        const edited = try std.mem.replaceOwned(u8, a, text, e.from, e.to);
        defer a.free(edited);
        const name = try controlSymbol(a, e.symbol);
        defer a.free(name);
        const renamed = try std.mem.replaceOwned(u8, a, edited, e.symbol, name);
        defer a.free(renamed);
        try out.appendSlice(a, "\n");
        try out.appendSlice(a, renamed);
    }
    return out.toOwnedSlice(a);
}

const glue_cl = @embedFile("inkling2_glue_cl");
const runtime_cl = @embedFile("inkling2_runtime_cl");
const prefill_cl = @embedFile("inkling2_prefill_cl");

test "every glue and runtime control applies once, inside its own kernel, and renames only that kernel" {
    const a = std.testing.allocator;
    for ([_]struct { []const u8, []const Entry }{ .{ glue_cl, &glue }, .{ runtime_cl, &runtime }, .{ prefill_cl, &prefill } }) |pair| {
        const production, const entries = pair;
        const text = try source(a, production, entries);
        defer a.free(text);
        for (entries) |e| {
            const name = try controlSymbol(a, e.symbol);
            defer a.free(name);
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, name));
            try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, e.symbol));
            const to = try std.mem.replaceOwned(u8, a, e.to, e.symbol, name);
            defer a.free(to);
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, to));
            try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, text, e.from));
        }
    }
}

test "a missing or repeated anchor refuses to make a control" {
    const a = std.testing.allocator;
    const twice = try std.mem.concat(a, u8, &.{ glue_cl, glue_cl });
    defer a.free(twice);
    try std.testing.expectError(error.KernelMissing, source(a, twice, glue[0..1]));
    const gone = [_]Entry{.{ .symbol = glue[0].symbol, .from = "no such text", .to = "x" }};
    try std.testing.expectError(error.ControlEditMissing, source(a, glue_cl, &gone));
    const ghost = [_]Entry{.{ .symbol = "tf_inkling2_ghost", .from = "x", .to = "y" }};
    try std.testing.expectError(error.KernelMissing, source(a, glue_cl, &ghost));
}

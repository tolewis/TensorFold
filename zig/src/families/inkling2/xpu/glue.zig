//! Inkling 2 Mini glue on XPU: the production OpenCL kernels and their launches, mirroring metal/glue.zig.
//! Buffers are Level Zero device addresses; every launch queues on the stream and the caller synchronizes.
const std = @import("std");
const xpu = @import("xpu");
pub const plan = @import("inkling2_glue_plan");

pub const hidden: u32 = 2304;
pub const experts: u32 = 128;
pub const intermediate: u32 = 1152;
pub const top: u32 = 8;
/// A device address; offsets are already applied.
pub const Ref = ?*anyopaque;
const group: [3]u32 = .{ 256, 1, 1 };

fn blocks(n: u32) u32 {
    return (n + 255) / 256;
}

pub const Pipelines = struct {
    embed_norm: xpu.Kernel,
    ar: xpu.Kernel,
    conv_static: xpu.Kernel,
    conv_dynamic: xpu.Kernel,
    router: xpu.Kernel,
    swiglu: xpu.Kernel,
    logits: xpu.Kernel,
    expert_pick: xpu.Kernel,
    expert_combine: xpu.Kernel,

    /// The kernels of a loaded image. A contract control's image holds the same kernels under `tf_ctl_inkling2_*`.
    pub fn load(module: xpu.Module, control: bool) !Pipelines {
        var made: [9]xpu.Kernel = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (comptime std.enums.values(plan.Kind)) |kind| {
            const name: [:0]const u8 = if (control) "tf_ctl_inkling2_" ++ @tagName(kind) else "tf_inkling2_" ++ @tagName(kind);
            comptime std.debug.assert(std.mem.eql(u8, kind.symbol(), "tf_inkling2_" ++ @tagName(kind)));
            made[@backingInt(kind)] = try module.kernel(name, group);
            n += 1;
        }
        return .{
            .embed_norm = made[@backingInt(plan.Kind.embed_norm)],
            .ar = made[@backingInt(plan.Kind.ar)],
            .conv_static = made[@backingInt(plan.Kind.conv_static)],
            .conv_dynamic = made[@backingInt(plan.Kind.conv_dynamic)],
            .router = made[@backingInt(plan.Kind.router)],
            .swiglu = made[@backingInt(plan.Kind.swiglu)],
            .logits = made[@backingInt(plan.Kind.logits)],
            .expert_pick = made[@backingInt(plan.Kind.expert_pick)],
            .expert_combine = made[@backingInt(plan.Kind.expert_combine)],
        };
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
    }
};

pub fn embedNorm(p: *const Pipelines, s: xpu.Stream, ids: Ref, table: Ref, weight: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(p.embed_norm, s, .{ rows, 1, 1 }, .{ ids, table, weight, out, rows });
}

/// sources = completed BF16 blocks plus one unrounded F32 partial; output is one FP32 row per query.
pub fn ar(p: *const Pipelines, s: xpu.Stream, blocks_: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32) !void {
    std.debug.assert(sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3);
    try xpu.launch.dispatch(p.ar, s, .{ queries_, 1, 1 }, .{ blocks_, partial, queries, out, sources_, queries_ });
}

/// taps contains three old rows followed by `rows` current BF16 inputs.
pub fn convStatic(p: *const Pipelines, s: xpu.Stream, taps: Ref, weight: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0);
    try xpu.launch.dispatch(p.conv_static, s, .{ blocks(channels), rows, 1 }, .{ taps, weight, out, rows, channels });
}

/// Dynamic static_w is [lag][channel], the opposite orientation to static convolution taps.
pub fn convDynamic(p: *const Pipelines, s: xpu.Stream, taps: Ref, input: Ref, static_w: Ref, projection: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0 and channels <= 1024);
    try xpu.launch.dispatch(p.conv_dynamic, s, .{ rows, 1, 1 }, .{ taps, input, static_w, projection, out, rows, channels });
}

/// Router IDs and weights are sorted by ascending expert index for the deterministic expert sum.
pub fn router(p: *const Pipelines, s: xpu.Stream, x: Ref, weight: Ref, bias: Ref, ids: Ref, weights: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(p.router, s, .{ rows, 1, 1 }, .{ x, weight, bias, ids, weights, rows });
}

pub fn swiglu(p: *const Pipelines, s: xpu.Stream, gu: Ref, out: Ref, rows: u32, width: u32) !void {
    std.debug.assert(rows > 0 and width > 0);
    try xpu.launch.dispatch(p.swiglu, s, .{ blocks(width), rows, 1 }, .{ gu, out, rows, width });
}

/// One sub-group of 16 a vocabulary row; a work-group of 256 covers 16 rows.
pub fn logits(p: *const Pipelines, s: xpu.Stream, h: Ref, unembed: Ref, out: Ref, rows: u32, vocab: u32) !void {
    std.debug.assert(rows > 0 and vocab > 0 and vocab <= 200058);
    try xpu.launch.dispatch(p.logits, s, .{ (vocab + 15) / 16, rows, 1 }, .{ h, unembed, out, rows, vocab });
}

pub fn expertPick(p: *const Pipelines, s: xpu.Stream, all: Ref, ids: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0);
    try xpu.launch.dispatch(p.expert_pick, s, .{ blocks(channels), 8, rows }, .{ all, ids, out, rows, channels });
}

pub fn expertCombine(p: *const Pipelines, s: xpu.Stream, picked: Ref, weights: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0);
    try xpu.launch.dispatch(p.expert_combine, s, .{ blocks(channels), rows, 1 }, .{ picked, weights, out, rows, channels });
}

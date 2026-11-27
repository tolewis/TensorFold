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
    /// The K and the V dynamic convolution of a visit in one launch (the production image only).
    conv_dynamic_kv: ?xpu.Kernel = null,

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
            .conv_dynamic_kv = if (control) null else try module.kernel("tf_inkling2_conv_dynamic_kv", group),
        };
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| {
            if (@TypeOf(@field(p, name)) == ?xpu.Kernel) {
                if (@field(p, name)) |*k| k.deinit();
            } else @field(p, name).deinit();
        }
    }
};

pub fn embedNorm(p: *const Pipelines, s: xpu.Stream, ids: Ref, table: Ref, weight: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(p.embed_norm, s, .{ rows, 1, 1 }, .{ ids, table, weight, out, rows });
}

/// sources = completed BF16 blocks plus one unrounded F32 partial; output is one FP32 row per query and token. `rows` tokens:
/// token t's blocks start `blocks_stride` BF16 values after token t - 1's, its partial is the next H floats, and its result
/// for query q is row `q * rows + t` of `out`. One token (`ar1`) is the launch of a single row.
pub fn arRows(p: *const Pipelines, s: xpu.Stream, blocks_: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32, rows: u32, blocks_stride: u32) !void {
    std.debug.assert(sources_ > 0 and sources_ <= 5 and queries_ > 0 and queries_ <= 3 and rows > 0);
    try xpu.launch.dispatch(p.ar, s, .{ queries_, rows, 1 }, .{ blocks_, partial, queries, out, sources_, queries_, rows, blocks_stride });
}

pub fn ar(p: *const Pipelines, s: xpu.Stream, blocks_: Ref, partial: Ref, queries: Ref, out: Ref, sources_: u32, queries_: u32) !void {
    try arRows(p, s, blocks_, partial, queries, out, sources_, queries_, 1, 0);
}

/// The window of a row is three old rows (`history`) and then the current rows; the kernel stages that window in `taps`
/// ([3 + rows][channels]) as it reads it, so the cache needs no separate copies.
pub fn convStatic(p: *const Pipelines, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, weight: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0);
    try xpu.launch.dispatch(p.conv_static, s, .{ blocks(channels), rows, 1 }, .{ history, current, taps, weight, out, rows, channels });
}

/// Dynamic static_w is [lag][channel], the opposite orientation to static convolution taps. `taps` is staged as in `convStatic`.
pub fn convDynamic(p: *const Pipelines, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, input: Ref, static_w: Ref, projection: Ref, out: Ref, rows: u32, channels: u32) !void {
    std.debug.assert(rows > 0 and channels > 0 and channels <= 1024);
    try xpu.launch.dispatch(p.conv_dynamic, s, .{ rows, 1, 1 }, .{ history, current, taps, input, static_w, projection, out, rows, channels });
}

/// The K and the V dynamic convolution of a visit, one launch of (rows, 2) work-groups. Each work-group runs the code
/// of `convDynamic` for its role.
pub fn convDynamicKv(p: *const Pipelines, s: xpu.Stream, k: [7]Ref, v: [7]Ref, rows: u32, channels: u32, commit1: u32) !void {
    std.debug.assert(rows > 0 and channels > 0 and channels <= 1024);
    try xpu.launch.dispatch(p.conv_dynamic_kv.?, s, .{ rows, 2, 1 }, .{ k[0], k[1], k[2], k[3], k[4], k[5], k[6], v[0], v[1], v[2], v[3], v[4], v[5], v[6], rows, channels, commit1 });
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

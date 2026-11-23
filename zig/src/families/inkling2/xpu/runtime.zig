//! Inkling 2 Mini's copy, widening and partial-residual kernels on XPU, mirroring metal/runtime_plan.zig's launches.
//! Buffers are Level Zero device addresses; every launch queues on the stream and the caller synchronizes.
const std = @import("std");
const xpu = @import("xpu");
pub const plan = @import("inkling2_runtime_plan");

pub const hidden: u32 = plan.hidden;
/// A device address; offsets are already applied.
pub const Ref = ?*anyopaque;

pub const Pipelines = struct {
    copy_bf16: xpu.Kernel,
    copy_f32: xpu.Kernel,
    widen: xpu.Kernel,
    attention_partial: xpu.Kernel,
    mlp_partial: xpu.Kernel,
    capture_tap: xpu.Kernel,

    /// The kernels of a loaded image. A contract control's image holds the same kernels under `tf_ctl_ink2_*`.
    pub fn load(module: xpu.Module, control: bool) !Pipelines {
        var made: [6]xpu.Kernel = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (comptime std.enums.values(plan.Kind)) |kind| {
            const name: [:0]const u8 = if (control) "tf_ctl_ink2_" ++ @tagName(kind) else "tf_ink2_" ++ @tagName(kind);
            comptime std.debug.assert(std.mem.eql(u8, kind.symbol(), "tf_ink2_" ++ @tagName(kind)));
            made[@backingInt(kind)] = try module.kernel(name, .{ plan.threads, 1, 1 });
            n += 1;
        }
        return .{
            .copy_bf16 = made[@backingInt(plan.Kind.copy_bf16)],
            .copy_f32 = made[@backingInt(plan.Kind.copy_f32)],
            .widen = made[@backingInt(plan.Kind.widen)],
            .attention_partial = made[@backingInt(plan.Kind.attention_partial)],
            .mlp_partial = made[@backingInt(plan.Kind.mlp_partial)],
            .capture_tap = made[@backingInt(plan.Kind.capture_tap)],
        };
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (@typeInfo(Pipelines).@"struct".field_names) |name| @field(p, name).deinit();
    }

    pub fn of(p: *const Pipelines, kind: plan.Kind) xpu.Kernel {
        return switch (kind) {
            .copy_bf16 => p.copy_bf16,
            .copy_f32 => p.copy_f32,
            .widen => p.widen,
            .attention_partial => p.attention_partial,
            .mlp_partial => p.mlp_partial,
            .capture_tap => p.capture_tap,
        };
    }
};

fn groups(c: plan.Call) u32 {
    return @intCast((plan.geometry(c).count + plan.threads - 1) / plan.threads);
}

/// One pointwise launch over rows * n values; `k` is one, the attention boundary, or the tap channel.
fn launch(p: *const Pipelines, s: xpu.Stream, kind: plan.Kind, c: plan.Call, bufs: anytype) !void {
    const a = plan.args(kind, c);
    try xpu.launch.dispatch(p.of(kind), s, .{ groups(c), 1, 1 }, bufs ++ .{ a.count, a.boundary });
}

pub fn copyBf16(p: *const Pipelines, s: xpu.Stream, x: Ref, y: Ref, rows: usize, n: usize) !void {
    try launch(p, s, .copy_bf16, .{ .rows = rows, .n = n, .k = 1 }, .{ x, y });
}

pub fn copyF32(p: *const Pipelines, s: xpu.Stream, x: Ref, y: Ref, rows: usize, n: usize) !void {
    try launch(p, s, .copy_f32, .{ .rows = rows, .n = n, .k = 1 }, .{ x, y });
}

pub fn widen(p: *const Pipelines, s: xpu.Stream, x: Ref, y: Ref, rows: usize, n: usize) !void {
    try launch(p, s, .widen, .{ .rows = rows, .n = n, .k = 1 }, .{ x, y });
}

/// boundary 0 adds p and x into `t`; 1 to 4 restarts `t` at x and stores p as that completed block of its row.
pub fn attentionPartial(p: *const Pipelines, s: xpu.Stream, prev: Ref, x: Ref, blocks: Ref, t: Ref, rows: usize, boundary: usize) !void {
    try launch(p, s, .attention_partial, .{ .rows = rows, .n = hidden, .k = boundary }, .{ prev, x, blocks, t });
}

pub fn mlpPartial(p: *const Pipelines, s: xpu.Stream, t: Ref, x: Ref, out: Ref, rows: usize) !void {
    try launch(p, s, .mlp_partial, .{ .rows = rows, .n = hidden, .k = 1 }, .{ t, x, out });
}

/// Writes one tap channel of each row's four-tap feature block.
pub fn captureTap(p: *const Pipelines, s: xpu.Stream, x: Ref, y: Ref, rows: usize, channel: usize) !void {
    try launch(p, s, .capture_tap, .{ .rows = rows, .n = hidden, .k = channel }, .{ x, y });
}

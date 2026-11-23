//! Inkling 2 Mini's dense projections and RMS norm on XPU: the kernels of kernels/xpu/inkling2/dense.cl and their launches,
//! with the names and arguments of metal/forward.zig's `dense` and `norm`. Buffers are Level Zero device addresses
//! with offsets applied; every launch queues on the stream and the caller synchronizes.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden: u32 = 2304;
/// A device address; offsets are already applied.
pub const Ref = ?*anyopaque;

/// Work-items a work-group of the matvec and the output columns it covers (dense.cl's DENSE_WG and DENSE_SG).
pub const matvec_threads: u32 = 128;
pub const matvec_columns: u32 = matvec_threads / 16;
pub const norm_threads: u32 = 256;
/// The row counts of the matvec kernels, in the order of `Kernels.dense`.
pub const variants = [_]u32{ 1, 2, 4, 8, 16, 32 };

pub const Kernels = struct {
    dense: [variants.len]xpu.Kernel,
    norm: xpu.Kernel,
    /// Output columns a matvec work-group covers.
    columns: u32 = matvec_columns,

    /// The kernels of a loaded image. A contract control's image holds the same kernels under `tf_ctl_ink2_*`.
    pub fn load(module: xpu.Module, control: bool) !Kernels {
        return loadAt(module, control, matvec_threads);
    }

    /// The same with the matvec's work-group set to `threads` (the image's DENSE_WG), for a timing of another build.
    pub fn loadAt(module: xpu.Module, control: bool, threads: u32) !Kernels {
        var made: [variants.len]xpu.Kernel = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (variants, 0..) |rows, i| {
            made[i] = try module.kernel(symbol(rows, control), .{ threads, 1, 1 });
            n += 1;
        }
        const norm_kernel = try module.kernel(if (control) "tf_ctl_ink2_norm" else "tf_ink2_norm", .{ norm_threads, 1, 1 });
        return .{ .dense = made, .norm = norm_kernel, .columns = threads / 16 };
    }

    pub fn deinit(p: *Kernels) void {
        for (&p.dense) |*k| k.deinit();
        p.norm.deinit();
    }
};

/// The matvec kernel that takes `rows` rows of x.
pub fn symbol(comptime rows: u32, control: bool) [:0]const u8 {
    return if (control) std.fmt.comptimePrint("tf_ctl_ink2_dense_{d}", .{rows}) else std.fmt.comptimePrint("tf_ink2_dense_{d}", .{rows});
}

/// The most rows one launch takes at or below `left`: 32, 16, 8, 4, 2 or 1, as the Metal host splits them.
pub fn take(left: usize) u32 {
    std.debug.assert(left > 0);
    inline for (.{ 32, 16, 8, 4, 2 }) |t| if (left >= t) return t;
    return 1;
}

/// The index in `variants` of a row count `take` returns.
pub fn variantOf(rows: u32) usize {
    return switch (rows) {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        else => unreachable,
    };
}

fn at(r: Ref, bytes: usize) Ref {
    return if (r) |p| @ptrFromInt(@intFromPtr(p) + bytes) else null;
}

/// One launch of the matvec kernel that takes `lead` rows of x: `rows` <= `lead` of them are stored, `ldy` values apart.
/// `input` must hold `lead` rows (the kernel reads them all).
pub fn matvec(p: *const Kernels, s: xpu.Stream, lead: u32, weight: Ref, input: Ref, out: Ref, rows: u32, n: u32, k: u32, ldy: u32) !void {
    std.debug.assert(rows > 0 and rows <= lead and n > 0 and k > 0 and k % 8 == 0 and ldy >= n);
    try xpu.launch.dispatch(p.dense[variantOf(lead)], s, .{ (n + p.columns - 1) / p.columns, 1, 1 }, .{ input, weight, out, rows, n, k, ldy });
}

/// out[rows][n] = bf16(input[rows][k] . weight[n][k]) with FP32 sums, for bf16 input and weight (n rows of k), k a
/// multiple of 8. Rows go 32, 16, 8, 4, 2, 1 at a time; each launch's input and output start at its first row.
pub fn dense(p: *const Kernels, s: xpu.Stream, weight: Ref, input: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    std.debug.assert(rows > 0);
    var base: u32 = 0;
    while (base < rows) {
        const t = take(rows - base);
        try matvec(p, s, t, weight, at(input, @as(usize, base) * k * 2), at(out, @as(usize, base) * n * 2), t, n, k, n);
        base += t;
    }
}

/// out[rows][2304] = bf16(weight * (input * inv)), inv = 1 / sqrt(mean(input^2) + 1e-6), for FP32 input rows.
pub fn norm(p: *const Kernels, s: xpu.Stream, input: Ref, weight: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0);
    try xpu.launch.dispatch(p.norm, s, .{ rows, 1, 1 }, .{ input, weight, out, rows });
}

test "rows split 32, 16, 8, 4, 2, 1 as the Metal host does" {
    var left: usize = 63;
    var got: [8]u32 = undefined;
    var i: usize = 0;
    while (left > 0) : (i += 1) {
        got[i] = take(left);
        left -= got[i];
    }
    try std.testing.expectEqualSlices(u32, &.{ 32, 16, 8, 4, 2, 1 }, got[0..i]);
    try std.testing.expectEqual(@as(u32, 32), take(100));
    try std.testing.expectEqual(@as(u32, 1), take(1));
    try std.testing.expectEqual(@as(u32, 2), take(3));
    try std.testing.expectEqual(@as(usize, 5), variantOf(32));
}

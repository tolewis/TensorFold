//! Inkling 2 Mini's dense projections over Q4 blocks on XPU: the kernels of kernels/xpu/inkling2/dense_q4.cl.
//! The weight format is q4.zig's. Rows of x are bf16, sums are FP32, the store is bf16. The head stays on the Q8 kernels.
const std = @import("std");
const xpu = @import("xpu");

pub const Ref = ?*anyopaque;
pub const group_threads: u32 = 128;
pub const variants = [_]u32{ 1, 2, 4, 8, 16, 32 };

pub const Kernels = struct {
    dense: [variants.len]xpu.Kernel,
    columns: u32 = group_threads / 16,

    pub fn load(module: xpu.Module, control: bool) !Kernels {
        var made: [variants.len]xpu.Kernel = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (variants) |rows| {
            made[n] = try module.kernel(symbol(rows, control), .{ group_threads, 1, 1 });
            n += 1;
        }
        return .{ .dense = made };
    }

    pub fn deinit(p: *Kernels) void {
        for (&p.dense) |*k| k.deinit();
    }
};

pub fn symbol(comptime rows: u32, control: bool) [:0]const u8 {
    return if (control) std.fmt.comptimePrint("tf_ctl_ink2_dense_q4_{d}", .{rows}) else std.fmt.comptimePrint("tf_ink2_dense_q4_{d}", .{rows});
}

pub fn take(left: usize) u32 {
    std.debug.assert(left > 0);
    inline for (.{ 32, 16, 8, 4, 2 }) |t| if (left >= t) return t;
    return 1;
}

fn variantOf(rows: u32) usize {
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

pub fn dense(p: *const Kernels, s: xpu.Stream, weight: Ref, input: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    std.debug.assert(rows > 0 and n > 0 and k % 256 == 0);
    var base: u32 = 0;
    while (base < rows) {
        const t = take(rows - base);
        try xpu.launch.dispatch(p.dense[variantOf(t)], s, .{ (n + p.columns - 1) / p.columns, 1, 1 }, .{ at(input, @as(usize, base) * k * 2), weight, at(out, @as(usize, base) * n * 2), t, n, k, n });
        base += t;
    }
}

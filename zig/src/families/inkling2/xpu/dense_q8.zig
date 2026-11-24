//! Inkling 2 Mini's dense projections and unembedding head over Q8_0-style weights on XPU: the kernels of
//! kernels/xpu/inkling2/dense_q8.cl and their launches. The weight format is q8.zig's; the rows of x are bf16, the sums
//! FP32, the store bf16 (the head stores the soft-capped logit as FP32). Buffers are Level Zero device addresses with
//! offsets applied; every launch queues on the stream and the caller synchronizes.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden: u32 = 2304;
/// A device address; offsets are already applied.
pub const Ref = ?*anyopaque;

/// Work-items a work-group (dense_q8.cl's Q8_WG) and the output columns it covers.
pub const group_threads: u32 = 128;
/// The row counts of the kernels, in the order of `Kernels.dense` and `Kernels.head`.
pub const variants = [_]u32{ 1, 2, 4, 8, 16, 32 };

pub const Kernels = struct {
    dense: [variants.len]xpu.Kernel,
    head: [variants.len]xpu.Kernel,
    /// Output columns a work-group covers.
    columns: u32 = group_threads / 16,

    /// The kernels of a loaded image. A contract control's image holds the same kernels under `tf_ctl_ink2_*`.
    pub fn load(module: xpu.Module, control: bool) !Kernels {
        return loadAt(module, control, group_threads);
    }

    /// The same with the work-group set to `threads` (the image's Q8_WG), for a timing of another build.
    pub fn loadAt(module: xpu.Module, control: bool, threads: u32) !Kernels {
        var made: [2 * variants.len]xpu.Kernel = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (variants) |rows| {
            made[n] = try module.kernel(symbol("dense", rows, control), .{ threads, 1, 1 });
            n += 1;
            made[n] = try module.kernel(symbol("head", rows, control), .{ threads, 1, 1 });
            n += 1;
        }
        var out: Kernels = .{ .dense = undefined, .head = undefined, .columns = threads / 16 };
        for (0..variants.len) |i| {
            out.dense[i] = made[2 * i];
            out.head[i] = made[2 * i + 1];
        }
        return out;
    }

    pub fn deinit(p: *Kernels) void {
        for (&p.dense) |*k| k.deinit();
        for (&p.head) |*k| k.deinit();
    }
};

/// The symbol of the matvec (`kind` "dense") or head (`kind` "head") kernel that takes `rows` rows of x.
pub fn symbol(comptime kind: []const u8, comptime rows: u32, control: bool) [:0]const u8 {
    return if (control) std.fmt.comptimePrint("tf_ctl_ink2_{s}_q8_{d}", .{ kind, rows }) else std.fmt.comptimePrint("tf_ink2_{s}_q8_{d}", .{ kind, rows });
}

/// The most rows one launch takes at or below `left`: 32, 16, 8, 4, 2 or 1.
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
/// `input` must hold `lead` rows (the kernel reads them all). `weight` is the matrix of `n` rows of `k` values in q8.zig's
/// layout (k a multiple of 256).
pub fn matvec(p: *const Kernels, s: xpu.Stream, lead: u32, weight: Ref, input: Ref, out: Ref, rows: u32, n: u32, k: u32, ldy: u32) !void {
    std.debug.assert(rows > 0 and rows <= lead and n > 0 and k > 0 and k % 256 == 0 and ldy >= n);
    try xpu.launch.dispatch(p.dense[variantOf(lead)], s, .{ (n + p.columns - 1) / p.columns, 1, 1 }, .{ input, weight, out, rows, n, k, ldy });
}

/// out[rows][n] = bf16(input[rows][k] . dequant(weight)[n][k]) with FP32 sums. Rows go 32, 16, 8, 4, 2, 1 at a time; each
/// launch's input and output start at its first row.
pub fn dense(p: *const Kernels, s: xpu.Stream, weight: Ref, input: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    std.debug.assert(rows > 0);
    var base: u32 = 0;
    while (base < rows) {
        const t = take(rows - base);
        try matvec(p, s, t, weight, at(input, @as(usize, base) * k * 2), at(out, @as(usize, base) * n * 2), t, n, k, n);
        base += t;
    }
}

/// One launch of the head kernel that takes `lead` rows of h.
pub fn headLaunch(p: *const Kernels, s: xpu.Stream, lead: u32, weight: Ref, h: Ref, out: Ref, rows: u32, vocab: u32) !void {
    std.debug.assert(rows > 0 and rows <= lead and vocab > 0);
    try xpu.launch.dispatch(p.head[variantOf(lead)], s, .{ (vocab + p.columns - 1) / p.columns, 1, 1 }, .{ h, weight, out, rows, vocab });
}

/// out[rows][vocab] = softcap30(bf16(h[rows][2304] . dequant(unembed)[vocab][2304])) in FP32, the logits of the first
/// `vocab` rows of the head; rows go 32, 16, 8, 4, 2, 1 at a time. `weight` holds `vocab` rows.
pub fn head(p: *const Kernels, s: xpu.Stream, weight: Ref, h: Ref, out: Ref, rows: u32, vocab: u32) !void {
    std.debug.assert(rows > 0);
    var base: u32 = 0;
    while (base < rows) {
        const t = take(rows - base);
        try headLaunch(p, s, t, weight, at(h, @as(usize, base) * hidden * 2), at(out, @as(usize, base) * vocab * 4), t, vocab);
        base += t;
    }
}

test "rows split 32, 16, 8, 4, 2, 1" {
    var left: usize = 63;
    var got: [8]u32 = undefined;
    var i: usize = 0;
    while (left > 0) : (i += 1) {
        got[i] = take(left);
        left -= got[i];
    }
    try std.testing.expectEqualSlices(u32, &.{ 32, 16, 8, 4, 2, 1 }, got[0..i]);
}

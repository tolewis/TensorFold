//! Inkling 2 Mini's routed experts for prompt chunks on XPU: the launches of kernels/xpu/inkling2/prefill_experts.cl.
//! The routes of a chunk are sorted by expert; each GEMM runs on the sorted rows with the weights decoded in registers.
//! Buffers are Level Zero device addresses with offsets applied; every launch queues on the stream.
const std = @import("std");
const xpu = @import("xpu");

pub const experts: u32 = 128;
pub const top_k: u32 = 8;
/// The most sorted rows of one item (PF_MT in the source).
pub const item_rows: u32 = 64;
/// The weight rows one work-group covers (PF_NSG sub-groups of 32 rows).
const group_cols: u32 = 256;
pub const Ref = ?*anyopaque;

pub const Format = enum { nvfp4, mxfp8 };

pub const Kernels = struct {
    plan: xpu.Kernel,
    gather: xpu.Kernel,
    nvfp4_gu: xpu.Kernel,
    nvfp4_down: xpu.Kernel,
    mxfp8_gu: xpu.Kernel,
    mxfp8_down: xpu.Kernel,
    combine: xpu.Kernel,

    pub fn load(module: xpu.Module) !Kernels {
        var plan_k = try module.kernel("tf_ink2_pf_plan", .{ 256, 1, 1 });
        errdefer plan_k.deinit();
        var gather_k = try module.kernel("tf_ink2_pf_gather", .{ 256, 1, 1 });
        errdefer gather_k.deinit();
        var nv_gu = try module.kernel("tf_ink2_pf_nvfp4_gu", .{ 128, 1, 1 });
        errdefer nv_gu.deinit();
        var nv_down = try module.kernel("tf_ink2_pf_nvfp4_down", .{ 128, 1, 1 });
        errdefer nv_down.deinit();
        var mx_gu = try module.kernel("tf_ink2_pf_mxfp8_gu", .{ 128, 1, 1 });
        errdefer mx_gu.deinit();
        var mx_down = try module.kernel("tf_ink2_pf_mxfp8_down", .{ 128, 1, 1 });
        errdefer mx_down.deinit();
        const combine_k = try module.kernel("tf_ink2_pf_combine", .{ 64, 1, 1 });
        return .{ .plan = plan_k, .gather = gather_k, .nvfp4_gu = nv_gu, .nvfp4_down = nv_down, .mxfp8_gu = mx_gu, .mxfp8_down = mx_down, .combine = combine_k };
    }

    pub fn deinit(k: *Kernels) void {
        k.combine.deinit();
        k.mxfp8_down.deinit();
        k.mxfp8_gu.deinit();
        k.nvfp4_down.deinit();
        k.nvfp4_gu.deinit();
        k.gather.deinit();
        k.plan.deinit();
        k.* = undefined;
    }
};

/// The most items a plan lists for `rows` rows: every expert's last item is short, the others hold `item_rows`.
pub fn maxItems(rows: u32) u32 {
    const r = rows * top_k;
    return @min(r, experts + r / item_rows);
}

/// Sorts the routes `ids[rows][8]` by expert: order[sorted] = route, inv[route] = sorted, items (expert, first, count).
pub fn plan(k: *const Kernels, s: xpu.Stream, ids: Ref, order: Ref, inv: Ref, items: Ref, count: Ref, rows: u32) !void {
    try xpu.launch.dispatch(k.plan, s, .{ 1, 1, 1 }, .{ ids, order, inv, items, count, rows * top_k, item_rows });
}

/// xs[sorted] = x[order[sorted] / 8], `width` BF16 values a row (a multiple of 8).
pub fn gather(k: *const Kernels, s: xpu.Stream, x: Ref, order: Ref, xs: Ref, rows: u32, width: u32) !void {
    std.debug.assert(width % 8 == 0);
    try xpu.launch.dispatch(k.gather, s, .{ rows * top_k, 1, 1 }, .{ x, order, xs, width, top_k });
}

pub const Weights = struct {
    w: Ref,
    scales: Ref,
    amax: Ref = null,
    den: Ref = null,
};

/// The GU GEMM over sorted rows: out[sorted][n] = the activation of (gate, up) = xs[sorted] . W[expert] (rows 2c, 2c + 1).
pub fn gu(k: *const Kernels, s: xpu.Stream, format: Format, xs: Ref, w: Weights, items: Ref, count: Ref, out: Ref, rows: u32, kdim: u32, n: u32) !void {
    const groups: [3]u32 = .{ (2 * n + group_cols - 1) / group_cols, maxItems(rows), 1 };
    switch (format) {
        .nvfp4 => try xpu.launch.dispatch(k.nvfp4_gu, s, groups, .{ xs, w.w, w.scales, w.amax, w.den, items, count, out, kdim, n }),
        .mxfp8 => try xpu.launch.dispatch(k.mxfp8_gu, s, groups, .{ xs, w.w, w.scales, items, count, out, kdim, n }),
    }
}

/// The down GEMM over sorted rows: out[sorted][n] = act[sorted] . W[expert].
pub fn down(k: *const Kernels, s: xpu.Stream, format: Format, act: Ref, w: Weights, items: Ref, count: Ref, out: Ref, rows: u32, kdim: u32, n: u32) !void {
    const groups: [3]u32 = .{ (n + group_cols - 1) / group_cols, maxItems(rows), 1 };
    switch (format) {
        .nvfp4 => try xpu.launch.dispatch(k.nvfp4_down, s, groups, .{ act, w.w, w.scales, w.amax, w.den, items, count, out, kdim, n }),
        .mxfp8 => try xpu.launch.dispatch(k.mxfp8_down, s, groups, .{ act, w.w, w.scales, items, count, out, kdim, n }),
    }
}

/// y[token][n] = BF16(sum over the token's routes in ascending expert order of weight * ds[inv[route]][n]).
pub fn combine(k: *const Kernels, s: xpu.Stream, ds: Ref, inv: Ref, ids: Ref, weights: Ref, y: Ref, rows: u32, n: u32) !void {
    try xpu.launch.dispatch(k.combine, s, .{ (n + 63) / 64, rows, 1 }, .{ ds, inv, ids, weights, y, n, top_k });
}

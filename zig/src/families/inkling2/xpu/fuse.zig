//! Fused visit glue. Phase A keeps the controller ranks on 32 work-groups and drops the extra launches.
//! The arithmetic is the existing kernels', in the same order. `INK2_FUSE=1` selects it.
const std = @import("std");
const xpu = @import("xpu");

pub const Ref = ?*anyopaque;

pub var force: ?bool = null;
pub var seg: bool = true;
pub var router_wide: bool = false;
pub var conv_fold: bool = false;
pub var zeta_wide: bool = false;

pub fn zetaWide() bool {
    if (zeta_wide) return true;
    if (std.c.getenv("INK2_ZETA_WIDE")) |p| {
        const s = std.mem.span(p);
        return s.len > 0 and s[0] == '1';
    }
    return false;
}

pub fn convFold() bool {
    if (conv_fold) return true;
    if (std.c.getenv("INK2_CONV_FOLD")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn routerWide() bool {
    if (router_wide) return true;
    if (std.c.getenv("INK2_ROUTER_WIDE")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn enabled() bool {
    if (force) |v| return v;
    if (std.c.getenv("INK2_FUSE")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub const Pipelines = struct {
    pre: xpu.Kernel,
    zeta: xpu.Kernel,
    gates: xpu.Kernel,
    norms: xpu.Kernel,
    router_gemv: xpu.Kernel,
    router_pick: xpu.Kernel,
    conv_attn: xpu.Kernel,
    conv_mlp: xpu.Kernel,
    zeta_wide: xpu.Kernel,

    pub fn load(module: xpu.Module) !Pipelines {
        var pre = try module.kernel("tf_ink2_fuse_pre", .{ 256, 1, 1 });
        errdefer pre.deinit();
        var zeta = try module.kernel("tf_ink2_fuse_zeta", .{ 256, 1, 1 });
        errdefer zeta.deinit();
        var gates = try module.kernel("tf_ink2_fuse_gates", .{ 64, 1, 1 });
        errdefer gates.deinit();
        var norms = try module.kernel("tf_ink2_fuse_norms", .{ 256, 1, 1 });
        errdefer norms.deinit();
        var router_gemv = try module.kernel("tf_ink2_router_gemv", .{ 256, 1, 1 });
        errdefer router_gemv.deinit();
        var router_pick = try module.kernel("tf_ink2_router_pick", .{ 128, 1, 1 });
        errdefer router_pick.deinit();
        var conv_attn = try module.kernel("tf_ink2_conv_attn_partial", .{ 256, 1, 1 });
        errdefer conv_attn.deinit();
        var conv_mlp = try module.kernel("tf_ink2_conv_mlp_partial", .{ 256, 1, 1 });
        errdefer conv_mlp.deinit();
        const zeta_wide_k = try module.kernel("tf_ink2_zeta_wide", .{ 256, 1, 1 });
        return .{ .pre = pre, .zeta = zeta, .gates = gates, .norms = norms, .router_gemv = router_gemv, .router_pick = router_pick, .conv_attn = conv_attn, .conv_mlp = conv_mlp, .zeta_wide = zeta_wide_k };
    }

    pub fn deinit(p: *Pipelines) void {
        p.zeta_wide.deinit();
        p.conv_mlp.deinit();
        p.conv_attn.deinit();
        p.router_pick.deinit();
        p.router_gemv.deinit();
        p.norms.deinit();
        p.gates.deinit();
        p.zeta.deinit();
        p.pre.deinit();
    }
};

pub const PhaseA = struct {
    p: Ref,
    partial: Ref,
    blocks: Ref,
    queries: Ref,
    ar: Ref,
    messages: Ref,
    down: [3]Ref,
    forget: [3]Ref,
    gain: [3]Ref,
    zeta: Ref,
    gated: Ref,
    wq: Ref,
    wk: Ref,
    wv: Ref,
    q_norm: Ref,
    k_norm: Ref,
    v_norm: Ref,
    rows: u32,
    blocks_stride: u32,
    sources: u32,
    n_queries: u32,
    roles: u32,
    local_src: u32,
    core: u32,
    msg_entries: u32,
    norm_n: u32,
};

pub fn convAttnPartial(p: *const Pipelines, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, weight: Ref, conv_out: Ref, prev: Ref, blocks: Ref, partial: Ref, rows: u32, channels: u32, boundary: u32) !void {
    const groups = (channels + 255) / 256;
    try xpu.launch.dispatch(p.conv_attn, s, .{ groups, rows, 1 }, .{ history, current, taps, weight, conv_out, prev, blocks, partial, rows, channels, boundary });
}

pub fn convMlpPartial(p: *const Pipelines, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, weight: Ref, conv_out: Ref, t: Ref, out: Ref, rows: u32, channels: u32) !void {
    const groups = (channels + 255) / 256;
    try xpu.launch.dispatch(p.conv_mlp, s, .{ groups, rows, 1 }, .{ history, current, taps, weight, conv_out, t, out, rows, channels });
}

pub fn routerWideLaunch(p: *const Pipelines, s: xpu.Stream, x: Ref, weight: Ref, bias: Ref, scores: Ref, ids: Ref, weights: Ref, rows: u32) !void {
    try xpu.launch.dispatch(p.router_gemv, s, .{ 8, rows, 1 }, .{ x, weight, scores, rows });
    try xpu.launch.dispatch(p.router_pick, s, .{ rows, 1, 1 }, .{ scores, bias, ids, weights, rows });
}

pub fn phaseA(p: *const Pipelines, s: xpu.Stream, a: PhaseA) !void {
    const dummy = a.zeta;
    const norm_here: u32 = if (a.roles == 0) a.norm_n else 0;
    try xpu.launch.dispatch(p.pre, s, .{ a.rows, 1, 1 }, .{
        a.p, a.partial, a.blocks, a.queries orelse dummy, a.ar,
        a.wq, a.wk orelse a.wq, a.wv orelse a.wq, a.q_norm, a.k_norm orelse a.q_norm, a.v_norm orelse a.q_norm,
        a.rows, a.blocks_stride, a.sources, a.n_queries, norm_here,
    });
    if (a.roles == 0) return;
    if (zetaWide()) {
        try xpu.launch.dispatch(p.zeta_wide, s, .{ 4, a.roles, a.rows }, .{
            a.ar, a.messages orelse dummy, a.down[0], a.down[1], a.down[2], a.zeta,
            a.rows, a.roles, a.local_src, a.core, a.msg_entries, @as(u32, 1),
        });
    } else try xpu.launch.dispatch(p.zeta, s, .{ 32, a.roles, a.rows }, .{
        a.ar, a.messages orelse dummy, a.down[0], a.down[1], a.down[2], a.zeta,
        a.rows, a.roles, a.local_src, a.core, a.msg_entries,
    });
    try xpu.launch.dispatch(p.gates, s, .{ 36, a.roles, a.rows }, .{
        a.ar, a.messages orelse dummy,
        a.forget[0], a.forget[1], a.forget[2], a.gain[0], a.gain[1], a.gain[2],
        a.zeta, a.gated, a.rows, a.roles, a.local_src, a.core, a.msg_entries,
    });
    try xpu.launch.dispatch(p.norms, s, .{ a.norm_n, a.rows, 1 }, .{
        a.gated, a.wq, a.wk orelse a.wq, a.wv orelse a.wq, a.q_norm, a.k_norm orelse a.q_norm, a.v_norm orelse a.q_norm,
        a.rows, a.norm_n,
    });
}

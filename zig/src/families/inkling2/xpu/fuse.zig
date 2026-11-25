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
pub var kv_conv: bool = false;
pub var down_epi: bool = false;
pub var norm_router: bool = false;
pub var role_conv: bool = false;
pub var ctl_one: bool = false;

pub fn roleConv() bool {
    if (role_conv) return true;
    if (std.c.getenv("INK2_ROLE")) |p| {
        const s = std.mem.span(p);
        return s.len > 0 and s[0] == '1';
    }
    return false;
}

pub fn ctlOne() bool {
    if (ctl_one) return true;
    if (std.c.getenv("INK2_CTL_ONE")) |p| {
        const s = std.mem.span(p);
        return s.len > 0 and s[0] == '1';
    }
    return false;
}

pub fn normRouter() bool {
    if (norm_router) return true;
    if (std.c.getenv("INK2_NORM_ROUTER")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn downEpi() bool {
    if (down_epi) return true;
    if (std.c.getenv("INK2_DOWN_EPI")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn kvConv() bool {
    if (kv_conv) return true;
    if (std.c.getenv("INK2_KV_CONV")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn zetaWide() bool {
    if (zeta_wide) return true;
    if (std.c.getenv("INK2_ZETA_WIDE")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
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
    conv_kv: xpu.Kernel,
    down_epi: xpu.Kernel,
    router_norm: xpu.Kernel,
    mlp_one: xpu.Kernel,

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
        var zeta_wide_k = try module.kernel("tf_ink2_zeta_wide", .{ 256, 1, 1 });
        errdefer zeta_wide_k.deinit();
        var conv_kv = try module.kernel("tf_ink2_conv_kv", .{ 256, 1, 1 });
        errdefer conv_kv.deinit();
        var down_epi_k = try module.kernel("tf_ink2_down_epilogue", .{ 256, 1, 1 });
        errdefer down_epi_k.deinit();
        var router_norm = try module.kernel("tf_ink2_router_norm_gemv", .{ 256, 1, 1 });
        errdefer router_norm.deinit();
        const mlp_one = try module.kernel("tf_ink2_mlp_one", .{ 64, 1, 1 });
        return .{ .pre = pre, .zeta = zeta, .gates = gates, .norms = norms, .router_gemv = router_gemv, .router_pick = router_pick, .conv_attn = conv_attn, .conv_mlp = conv_mlp, .zeta_wide = zeta_wide_k, .conv_kv = conv_kv, .down_epi = down_epi_k, .router_norm = router_norm, .mlp_one = mlp_one };
    }

    pub fn deinit(p: *Pipelines) void {
        p.mlp_one.deinit();
        p.router_norm.deinit();
        p.down_epi.deinit();
        p.conv_kv.deinit();
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

pub fn convAttnPartial(p: *const Pipelines, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, weight: Ref, conv_out: Ref, prev: Ref, blocks: Ref, partial: Ref, rows: u32, channels: u32, boundary: u32, commit1: u32) !void {
    const groups = (channels + 255) / 256;
    try xpu.launch.dispatch(p.conv_attn, s, .{ groups, rows, 1 }, .{ history, current, taps, weight, conv_out, prev, blocks, partial, rows, channels, boundary, commit1 });
}

pub fn downEpilogue(p: *const Pipelines, s: xpu.Stream, down: Ref, ids: Ref, weights: Ref, history: Ref, taps: Ref, conv_w: Ref, current: Ref, conv_out: Ref, partial: Ref, pout: Ref, n: u32, topk: u32, channels: u32, rows: u32, commit1: u32) !void {
    const groups = (n + 255) / 256;
    try xpu.launch.dispatch(p.down_epi, s, .{ groups, 1, 1 }, .{ down, ids, weights, history, taps, conv_w, current, conv_out, partial, pout, n, topk, channels, rows, commit1 });
}

pub fn convKv(p: *const Pipelines, s: xpu.Stream, hk: Ref, ck: Ref, tk: Ref, wk: Ref, ok: Ref, hv: Ref, cv: Ref, tv: Ref, wv: Ref, ov: Ref, rows: u32, channels: u32, commit1: u32) !void {
    const groups = (channels + 255) / 256;
    try xpu.launch.dispatch(p.conv_kv, s, .{ groups, rows, 1 }, .{ hk, ck, tk, wk, ok, hv, cv, tv, wv, ov, rows, channels, commit1 });
}

pub fn convMlpPartial(p: *const Pipelines, s: xpu.Stream, history: Ref, current: Ref, taps: Ref, weight: Ref, conv_out: Ref, t: Ref, out: Ref, rows: u32, channels: u32, commit1: u32) !void {
    const groups = (channels + 255) / 256;
    try xpu.launch.dispatch(p.conv_mlp, s, .{ groups, rows, 1 }, .{ history, current, taps, weight, conv_out, t, out, rows, channels, commit1 });
}

pub fn mlpOne(p: *const Pipelines, s: xpu.Stream, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, out: Ref, rows: u32, m_stride: u32) !void {
    try xpu.launch.dispatch(p.mlp_one, s, .{ 36, rows, 1 }, .{ h, m, down, forget, gain, out, rows, m_stride });
}

pub fn routerNormLaunch(p: *const Pipelines, s: xpu.Stream, x: Ref, norm_w: Ref, weight: Ref, normed: Ref, bias: Ref, scores: Ref, ids: Ref, weights: Ref, rows: u32) !void {
    try xpu.launch.dispatch(p.router_norm, s, .{ 8, rows, 1 }, .{ x, norm_w, weight, normed, scores, rows });
    try xpu.launch.dispatch(p.router_pick, s, .{ rows, 1, 1 }, .{ scores, bias, ids, weights, rows });
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

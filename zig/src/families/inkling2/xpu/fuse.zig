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
pub var sink_seg: bool = false;

pub fn sinkSeg() bool {
    if (sink_seg) return true;
    if (xpu.launch.envOnce("INK2_SINK_SEG")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}
pub fn roleConv() bool {
    if (role_conv) return true;
    if (xpu.launch.envOnce("INK2_ROLE")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn normRouter() bool {
    if (norm_router) return true;
    if (xpu.launch.envOnce("INK2_NORM_ROUTER")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn downEpi() bool {
    if (down_epi) return true;
    if (xpu.launch.envOnce("INK2_DOWN_EPI")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn kvConv() bool {
    if (kv_conv) return true;
    if (xpu.launch.envOnce("INK2_KV_CONV")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

/// The AR of the prologue over one group a query row. Default; `INK2_PRE_Q=0` selects the looped prologue.
pub var pre_q: bool = false;

pub fn preQ() bool {
    if (pre_q) return true;
    if (xpu.launch.envOnce("INK2_PRE_Q")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

/// The prologue's AR and the message gate projection in one launch. Default; `INK2_PRE_ZETA=0` selects the two launches.
pub var pre_zeta: bool = false;

pub fn preZeta() bool {
    if (pre_zeta) return true;
    if (xpu.launch.envOnce("INK2_PRE_ZETA")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

/// The gates and the norms of the gated rows in one group a row, off until it wins.
pub var gates_norms: bool = false;

pub fn gatesNorms() bool {
    if (gates_norms) return true;
    if (xpu.launch.envOnce("INK2_GATES_NORMS")) |p| {
        const s = std.mem.span(p);
        return s.len > 0 and s[0] == '1';
    }
    return false;
}

/// The zeta over 16 groups of four sub-groups a rank. Default; `INK2_ZETA_SPLIT=0` selects the wide kernel.
pub var zeta_split: bool = false;

pub fn zetaSplit() bool {
    if (zeta_split) return true;
    if (xpu.launch.envOnce("INK2_ZETA_SPLIT")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn zetaWide() bool {
    if (zeta_wide) return true;
    if (xpu.launch.envOnce("INK2_ZETA_WIDE")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn convFold() bool {
    if (conv_fold) return true;
    if (xpu.launch.envOnce("INK2_CONV_FOLD")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn routerWide() bool {
    if (router_wide) return true;
    if (xpu.launch.envOnce("INK2_ROUTER_WIDE")) |p| {
        const s = std.mem.span(p);
        if (s.len > 0 and s[0] == '0') return false;
    }
    return true;
}

pub fn enabled() bool {
    if (force) |v| return v;
    if (xpu.launch.envOnce("INK2_FUSE")) |p| {
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
    zeta_split_k: xpu.Kernel,
    gates_norms_k: xpu.Kernel,
    pre_q_k: xpu.Kernel,
    pre_zeta_k: xpu.Kernel,
    conv_kv: xpu.Kernel,
    down_epi: xpu.Kernel,
    router_norm: xpu.Kernel,
    /// Empty kernels at 64, 128 and 256 work-items, for the launch-cost probe.
    empty: [3]xpu.Kernel,

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
        const zeta_split_k = try module.kernel("tf_ink2_zeta_split", .{ 256, 1, 1 });
        const gates_norms_k = try module.kernel("tf_ink2_gates_norms", .{ 256, 1, 1 });
        const pre_q_k = try module.kernel("tf_ink2_fuse_pre_q", .{ 256, 1, 1 });
        const pre_zeta_k = try module.kernel("tf_ink2_pre_zeta", .{ 256, 1, 1 });
        var conv_kv = try module.kernel("tf_ink2_conv_kv", .{ 256, 1, 1 });
        errdefer conv_kv.deinit();
        var down_epi_k = try module.kernel("tf_ink2_down_epilogue", .{ 256, 1, 1 });
        errdefer down_epi_k.deinit();
        const router_norm = try module.kernel("tf_ink2_router_norm_gemv", .{ 256, 1, 1 });
        const empty: [3]xpu.Kernel = .{
            try module.kernel("tf_ink2_empty64", .{ 64, 1, 1 }),
            try module.kernel("tf_ink2_empty128", .{ 128, 1, 1 }),
            try module.kernel("tf_ink2_empty256", .{ 256, 1, 1 }),
        };
        return .{ .empty = empty, .pre = pre, .zeta = zeta, .gates = gates, .norms = norms, .router_gemv = router_gemv, .router_pick = router_pick, .conv_attn = conv_attn, .conv_mlp = conv_mlp, .zeta_wide = zeta_wide_k, .zeta_split_k = zeta_split_k, .gates_norms_k = gates_norms_k, .pre_q_k = pre_q_k, .pre_zeta_k = pre_zeta_k, .conv_kv = conv_kv, .down_epi = down_epi_k, .router_norm = router_norm };
    }

    pub fn deinit(p: *Pipelines) void {
        for (&p.empty) |*k| k.deinit();
        p.router_norm.deinit();
        p.down_epi.deinit();
        p.conv_kv.deinit();
        p.zeta_wide.deinit();
        p.zeta_split_k.deinit();
        p.gates_norms_k.deinit();
        p.pre_q_k.deinit();
        p.pre_zeta_k.deinit();
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

/// The launch-cost probe: `INK2_EMPTY_PROBE=1` adds one empty launch a (work-group size, grid) pair at the end of a
/// one-row forward, each under its own profile label. The grids are those of the decode kernels' sheet.
pub fn emptyProbe() bool {
    if (xpu.launch.envOnce("INK2_EMPTY_PROBE")) |p| {
        const s = std.mem.span(p);
        return s.len > 0 and s[0] == '1';
    }
    return false;
}

const probe_grids = [_]u32{ 1, 3, 4, 8, 9, 12, 16, 36, 48, 108 };
const probe_labels = [3][probe_grids.len][]const u8{
    .{ "w64g1", "w64g3", "w64g4", "w64g8", "w64g9", "w64g12", "w64g16", "w64g36", "w64g48", "w64g108" },
    .{ "w128g1", "w128g3", "w128g4", "w128g8", "w128g9", "w128g12", "w128g16", "w128g36", "w128g48", "w128g108" },
    .{ "w256g1", "w256g3", "w256g4", "w256g8", "w256g9", "w256g12", "w256g16", "w256g36", "w256g48", "w256g108" },
};

pub fn emptyLaunches(p: *const Pipelines, s: xpu.Stream) !void {
    const saved = xpu.profile.label;
    defer xpu.profile.label = saved;
    for (0..3) |ki| {
        for (probe_grids, 0..) |n, gi| {
            xpu.profile.label = probe_labels[ki][gi];
            try xpu.launch.dispatch(p.empty[ki], s, .{ n, 1, 1 }, .{@as(u32, 0)});
        }
    }
}

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
    if (preZeta() and norm_here == 0 and a.n_queries > 0 and a.queries != null and a.messages != null) {
        try xpu.launch.dispatch(p.pre_zeta_k, s, .{ 16, a.roles, a.rows }, .{
            a.p, a.partial, a.blocks, a.queries orelse dummy, a.ar,
            a.messages orelse dummy, a.down[0], a.down[1], a.down[2], a.zeta,
            a.rows, a.blocks_stride, a.sources, a.roles, a.local_src, a.core, a.msg_entries,
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
        return;
    }
    if (preQ() and a.n_queries > 0 and a.queries != null and (norm_here == 0 or norm_here == a.n_queries)) {
        try xpu.launch.dispatch(p.pre_q_k, s, .{ a.n_queries, a.rows, 1 }, .{
            a.p, a.partial, a.blocks, a.queries orelse dummy, a.ar,
            a.wq, a.wk orelse a.wq, a.wv orelse a.wq,
            a.q_norm, a.k_norm orelse a.q_norm, a.v_norm orelse a.q_norm,
            a.rows, a.blocks_stride, a.sources, a.n_queries, norm_here,
        });
    } else try xpu.launch.dispatch(p.pre, s, .{ a.rows, 1, 1 }, .{
        a.p, a.partial, a.blocks, a.queries orelse dummy, a.ar,
        a.wq, a.wk orelse a.wq, a.wv orelse a.wq, a.q_norm, a.k_norm orelse a.q_norm, a.v_norm orelse a.q_norm,
        a.rows, a.blocks_stride, a.sources, a.n_queries, norm_here,
    });
    if (a.roles == 0) return;
    if (zetaSplit()) {
        try xpu.launch.dispatch(p.zeta_split_k, s, .{ 16, a.roles, a.rows }, .{
            a.ar, a.messages orelse dummy, a.down[0], a.down[1], a.down[2], a.zeta,
            a.rows, a.roles, a.local_src, a.core, a.msg_entries, @as(u32, 1),
        });
    } else if (zetaWide()) {
        try xpu.launch.dispatch(p.zeta_wide, s, .{ 4, a.roles, a.rows }, .{
            a.ar, a.messages orelse dummy, a.down[0], a.down[1], a.down[2], a.zeta,
            a.rows, a.roles, a.local_src, a.core, a.msg_entries, @as(u32, 1),
        });
    } else try xpu.launch.dispatch(p.zeta, s, .{ 32, a.roles, a.rows }, .{
        a.ar, a.messages orelse dummy, a.down[0], a.down[1], a.down[2], a.zeta,
        a.rows, a.roles, a.local_src, a.core, a.msg_entries,
    });
    if (gatesNorms()) {
        try xpu.launch.dispatch(p.gates_norms_k, s, .{ a.roles, a.rows, 1 }, .{
            a.ar, a.messages orelse dummy,
            a.forget[0], a.forget[1], a.forget[2], a.gain[0], a.gain[1], a.gain[2],
            a.zeta, a.gated, a.wq, a.wk orelse a.wq, a.wv orelse a.wq,
            a.q_norm, a.k_norm orelse a.q_norm, a.v_norm orelse a.q_norm,
            a.rows, a.roles, a.local_src, a.core, a.msg_entries, a.norm_n,
        });
    } else {
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
}

//! Fused visit glue. Phase A keeps the controller ranks on 32 work-groups and drops the extra launches.
//! The arithmetic is the existing kernels', in the same order. `INK2_FUSE=1` selects it.
const std = @import("std");
const xpu = @import("xpu");

pub const Ref = ?*anyopaque;

pub var force: ?bool = null;
pub var seg: bool = true;

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

    pub fn load(module: xpu.Module) !Pipelines {
        var pre = try module.kernel("tf_ink2_fuse_pre", .{ 256, 1, 1 });
        errdefer pre.deinit();
        var zeta = try module.kernel("tf_ink2_fuse_zeta", .{ 256, 1, 1 });
        errdefer zeta.deinit();
        var gates = try module.kernel("tf_ink2_fuse_gates", .{ 64, 1, 1 });
        errdefer gates.deinit();
        const norms = try module.kernel("tf_ink2_fuse_norms", .{ 256, 1, 1 });
        return .{ .pre = pre, .zeta = zeta, .gates = gates, .norms = norms };
    }

    pub fn deinit(p: *Pipelines) void {
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

pub fn phaseA(p: *const Pipelines, s: xpu.Stream, a: PhaseA) !void {
    const dummy = a.zeta;
    const norm_here: u32 = if (a.roles == 0) a.norm_n else 0;
    try xpu.launch.dispatch(p.pre, s, .{ a.rows, 1, 1 }, .{
        a.p, a.partial, a.blocks, a.queries orelse dummy, a.ar,
        a.wq, a.wk orelse a.wq, a.wv orelse a.wq, a.q_norm, a.k_norm orelse a.q_norm, a.v_norm orelse a.q_norm,
        a.rows, a.blocks_stride, a.sources, a.n_queries, norm_here,
    });
    if (a.roles == 0) return;
    try xpu.launch.dispatch(p.zeta, s, .{ 32, a.roles, a.rows }, .{
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

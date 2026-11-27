//! TF_INKLING2_FUSED: one cooperative kernel a decode visit (kernels/cuda/inkling2/fused_main.cu, Recipe
//! lanes/inkspeed/fused-cuda/DESIGN.md). Unset or 0: nothing loads, today's kernels run. `check:<v>,<v>,...`: each
//! listed one-row decode visit runs today's kernels, then the fused kernel on the same input, and every buffer the
//! visit writes is compared byte by byte. Today's results stay in place, so the reply is today's reply.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const attn = @import("kernels.zig");
const ops = @import("ops.zig");
const loop = @import("loop.zig");
const mm = @import("mini_model.zig");
const wts = @import("mini_weights.zig");

const Model = mm.Model;
const Kernels = mm.Kernels;

pub const Mode = enum { off, on, check };

const max_list = 48;

pub const Switch = struct {
    mode: Mode = .off,
    list: [max_list]u32 = undefined,
    n: usize = 0,

    pub fn has(w: *const Switch, v: u32) bool {
        return std.mem.indexOfScalar(u32, w.list[0..w.n], v) != null;
    }
};

/// The switch, read once.
pub fn read() Switch {
    const S = struct {
        var v: ?Switch = null;
    };
    if (S.v) |v| return v;
    var w: Switch = .{};
    if (std.c.getenv("TF_INKLING2_FUSED")) |e| {
        const t = std.mem.span(e);
        if (std.mem.startsWith(u8, t, "check:")) {
            w.mode = .check;
            var it = std.mem.tokenizeScalar(u8, t[6..], ',');
            while (it.next()) |x| if (w.n < max_list) {
                w.list[w.n] = std.fmt.parseInt(u32, x, 10) catch continue;
                w.n += 1;
            };
        } else if (t.len > 0 and t[0] != '0') w.mode = .on;
    }
    S.v = w;
    return w;
}

pub fn checking() bool {
    return read().mode == .check;
}

// ---- the table (fused_visit.cuh, field for field) ----

pub const Proj = extern struct { x: u64 = 0, codes: u64 = 0, scales: u64 = 0, out: u64 = 0, part: u64 = 0, scale: f32 = 0, n: i32 = 0, k: i32 = 0, sk: i32 = 0, npad: i32 = 0, mode: i32 = 0 };

pub const Prep = extern struct {
    q_raw: u64,
    k_raw: u64,
    v_raw: u64,
    rel_raw: u64,
    q_scale: u64,
    rel_scale: u64,
    rpos: u64,
    rseg: u64,
    segs: u64,
    q_out: u64,
    rel_out: u64,
    kc: u64,
    vc: u64,
    ksc: u64,
    vsc: u64,
    heads: i32,
    kv_heads: i32,
    ring: i32,
    global: i32,
    qk_clip: f32,
    rel_clip: f32,
    floor_pos: f32,
    alpha: f32,
    eps: f32,
    pad: f32 = 0,
};

pub const Attn = extern struct {
    q: u64,
    kc: u64,
    vc: u64,
    tab: u64,
    segs: u64,
    tiles: u64,
    part_o: u64,
    part_ml: u64,
    ksc: u64,
    vsc: u64,
    sink: u64,
    gate: u64,
    rpos: u64,
    att: u64,
    relp: u64,
    rel_proj: u64,
    rtab: u64,
    heads: i32,
    kv_heads: i32,
    ring: i32,
    window: i32,
    chunk: i32,
    chunks: i32,
    split: i32,
    zgrid: i32,
    slots_rh: i32,
    subs: i32,
    scale: f32,
    db: i32,
};

pub const Moe = extern struct {
    xm: u64,
    router: u64,
    scores: u64,
    bias: u64,
    pick: u64,
    rw: u64,
    members: u64,
    items: u64,
    counts: u64,
    up: u64,
    down: u64,
    up_scale: u64,
    down_scale: u64,
    act: u64,
    y: u64,
    moe: u64,
    e: i32,
    k: i32,
    t: i32,
    h: i32,
    width: i32,
    pad: i32 = 0,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
    pad2: f32 = 0,
};

pub const Taps = extern struct { x: [2]u64 = .{ 0, 0 }, w: [2]u64 = .{ 0, 0 }, ones: u64 = 0, out: [2]u64 = .{ 0, 0 }, n: i32 = 0, k: i32 = 0 };

pub const Visit = extern struct {
    global: i32,
    gated: i32,
    nq_attn: i32,
    n_proj: i32,
    copy_bytes: i32 = 0,
    drop: i32 = 0, // review A2, A3: a skipped arrival, ((group + 1) << 3) | sub (0: none)
    probe: i32 = 0, // the probed phase + 1 (0: none), TF_INKLING2_FUSED_PROBE
    fix: i32 = 0, // the review corrections the kernel runs (fused_visit.cuh fz_fix), TF_INKLING2_FUSED_FIX
    copy_dst: u64 = 0,
    copy_src: u64 = 0,
    mix_attn: u64,
    mix_mlp: u64,
    gate_attn: u64 = 0,
    gate_mlp: u64 = 0,
    conv: [4]u64,
    proj: [3]Proj,
    wo: Proj,
    taps: Taps = .{},
    prep: Prep,
    attn: Attn,
    moe: Moe,
};

pub const StateArgs = extern struct { cnt: u64, tk: u64, fault: u64, part1: u64, part2: u64, times: u64, probe: u64 };

comptime {
    std.debug.assert(@sizeOf(Proj) == 64);
    std.debug.assert(@sizeOf(Prep) == 160);
    std.debug.assert(@sizeOf(Attn) == 184);
    std.debug.assert(@sizeOf(Moe) == 168);
    std.debug.assert(@sizeOf(Taps) == 64);
    std.debug.assert(@sizeOf(Visit) == 944);
    std.debug.assert(@sizeOf(StateArgs) == 56);
}

// a visit's argument structs in device memory: offsets in its 2 KB region
const arg_mix_attn = 0;
const arg_mix_mlp = 256;
const arg_gate_attn = 512;
const arg_gate_mlp = 768;
const arg_conv = 1024; // 4 x 128
const arg_bytes = 2048;
comptime {
    std.debug.assert(@sizeOf(ops.MixArgs) <= 256 and @sizeOf(loop.MixArgs) <= 256 and @sizeOf(loop.GateArgs) <= 256);
    std.debug.assert(@sizeOf(ops.ConvArgs) <= 128);
}

const phases_max = 32;
// the table sizes and the bound of each sender (review A6); fused_visit.cuh and fused_mix.cu use the same numbers
const part1_sources = 9; // n_blocks + 1, n_blocks at most mix_blocks
const part1_warps = 16; // MIXW (LOOPW is 8)
const part1_stride = 4; // floats a warp: the norm and up to 3 query partials
const mix_blocks = 8; // fused_mix.cu NB
const tiles_call = 256; // tickets a qmmf call
const qpart_sk = 8;
const qpart_n = 8704;

/// Whether a sender fits the fused tables; the host refuses the visit before the launch otherwise (review A6).
pub fn bounds(nq: usize, n_blocks: usize, n: usize, sk: usize, npad: usize) !void {
    if (nq + 1 > part1_stride) return error.FusedQueries;
    if (n_blocks > mix_blocks or n_blocks + 1 > part1_sources) return error.FusedBlocks;
    if ((n + 63) / 64 > tiles_call) return error.FusedTickets;
    if (sk > qpart_sk or @max(n, npad) > qpart_n) return error.FusedQpart;
}

const tickets_visit = 1024;
const times_visit = 64;
const visits_max = config.max_visits;
const threads = 128;
const ref_bytes = 32 << 20;
const probe_stamps = 32;

pub const State = struct {
    d: *const cuda.Driver,
    mod: cuda.Module,
    f: cuda.Function,
    grid: u32,
    dyn: u32,
    sw: Switch,
    table: cuda.DeviceBuffer, // Visit [visits]
    args: cuda.DeviceBuffer, // [visits][arg_bytes]
    cnt: cuda.DeviceBuffer,
    tk: cuda.DeviceBuffer,
    fault: cuda.DeviceBuffer,
    part1: cuda.DeviceBuffer,
    part2: cuda.DeviceBuffer,
    times: cuda.DeviceBuffer,
    probe: cuda.DeviceBuffer, // the probed phase of a launch: start, arrival, exit [G] each, 32 unit stamps
    qpart: cuda.DeviceBuffer, // fp32 [4 calls][8][8600] qmmf slice partials
    ref: cuda.DeviceBuffer, // check: today's bytes of each compared buffer
    save: cuda.DeviceBuffer, // check: the conv states before the visit
    ev: [8]cuda.abi.Event = @splat(null),
    graph_ok: bool = true, // the fused launch captures into a graph (review A22); false after a refused capture
    at: []const u8 = "",
    steps: [visits_max]usize = @splat(0),
    bad: [visits_max]usize = @splat(0),
    drop_at: [visits_max]usize = @splat(0),

    pub fn init(d: *const cuda.Driver) !State {
        const sw = read();
        if (sw.mode == .on) { // review A8: no serve path runs the fused kernel yet
            std.debug.print("[fused] TF_INKLING2_FUSED=1: the fused kernel serves no visit yet (check:<visits> checks it); refused at start\n", .{});
            return error.FusedOnNotBuilt;
        }
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_fused);
        errdefer mod.unload();
        const f = try mod.function("tf_ink2_fused_visit");
        const dyn: u32 = @max(@max(attn.db8_shared, cuda.qmmf.sharedBytes(.mxfp8, 16)), @max(cuda.qmmf.sharedBytes(.fp8, 16), 4 * 5760));
        try f.allowDynamicShared(dyn);
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var sms: c_int = 0;
        try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, dev), "cuDeviceGetAttribute");
        var per: c_int = 0;
        try d.check(d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor(&per, f.handle, threads, dyn), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
        if (per < 1) return error.FusedOccupancy;
        const grid: u32 = @intCast(sms); // one block a multiprocessor (DESIGN section 10)
        var s: State = .{
            .d = d,
            .mod = mod,
            .f = f,
            .grid = grid,
            .dyn = dyn,
            .sw = sw,
            .table = try cuda.DeviceBuffer.alloc(d, visits_max * @sizeOf(Visit)),
            .args = try cuda.DeviceBuffer.alloc(d, visits_max * arg_bytes),
            .cnt = try cuda.DeviceBuffer.alloc(d, visits_max * phases_max * 8 * 4),
            .tk = try cuda.DeviceBuffer.alloc(d, visits_max * tickets_visit * 4),
            .fault = try cuda.DeviceBuffer.alloc(d, visits_max * 4),
            .part1 = try cuda.DeviceBuffer.alloc(d, part1_sources * part1_warps * part1_stride * 4),
            .part2 = try cuda.DeviceBuffer.alloc(d, 4 * 16 * 4),
            .times = try cuda.DeviceBuffer.alloc(d, visits_max * times_visit * 8),
            .probe = try cuda.DeviceBuffer.alloc(d, (3 * @as(usize, @intCast(sms)) + probe_stamps) * 8),
            .qpart = try cuda.DeviceBuffer.alloc(d, 4 * qpart_sk * qpart_n * 4),
            .ref = try cuda.DeviceBuffer.alloc(d, @as(usize, if (sw.mode == .check) ref_bytes else 0)),
            .save = try cuda.DeviceBuffer.alloc(d, @as(usize, if (sw.mode == .check) 1 << 20 else 0)),
        };
        for (&s.ev) |*e| try d.check(d.api.cuEventCreate(e, 0), "cuEventCreate");
        std.debug.print("[fused] {s}: grid {d} x {d}, {d} B dynamic shared, {d} a multiprocessor, visits {any}\n", .{ @tagName(sw.mode), grid, threads, dyn, per, sw.list[0..sw.n] });
        return s;
    }

    fn statePtrs(f: *const State) StateArgs {
        return .{ .cnt = f.cnt.ptr, .tk = f.tk.ptr, .fault = f.fault.ptr, .part1 = f.part1.ptr, .part2 = f.part2.ptr, .times = f.times.ptr, .probe = f.probe.ptr };
    }

    /// Visits [first, first + count): counters, tickets and fault words cleared, one cooperative launch.
    pub fn launch(f: *const State, s: cuda.Stream, first: u32, count: u32) !void {
        return f.launchMarked(s, first, count, null, false);
    }

    /// `launch` with event `mid` recorded after the memsets (review A17); `external` when the stream captures.
    pub fn launchMarked(f: *const State, s: cuda.Stream, first: u32, count: u32, mid: cuda.abi.Event, external: bool) !void {
        return f.launchFix(s, first, count, mid, external, true);
    }

    /// `own`: clear the counter groups of `count` visits (the kernel counts groups from 0 in a launch), and the
    /// tickets and fault words of visits [first, first + count) only (review A8, A17). Else every visit's.
    pub fn launchFix(f: *const State, s: cuda.Stream, first: u32, count: u32, mid: cuda.abi.Event, external: bool, own: bool) !void {
        const d = f.d;
        if (first + count > visits_max) return error.FusedVisitRange;
        if (own) {
            const cv: usize = phases_max * 8 * 4;
            try d.check(d.api.cuMemsetD8Async(f.cnt.ptr, 0, @as(usize, count) * cv, s.handle), "cuMemsetD8Async");
            try d.check(d.api.cuMemsetD8Async(f.tk.ptr + @as(u64, first) * tickets_visit * 4, 0, @as(usize, count) * tickets_visit * 4, s.handle), "cuMemsetD8Async");
            try d.check(d.api.cuMemsetD8Async(f.fault.ptr + @as(u64, first) * 4, 0, @as(usize, count) * 4, s.handle), "cuMemsetD8Async");
        } else {
            try d.check(d.api.cuMemsetD8Async(f.cnt.ptr, 0, f.cnt.len, s.handle), "cuMemsetD8Async");
            try d.check(d.api.cuMemsetD8Async(f.tk.ptr, 0, f.tk.len, s.handle), "cuMemsetD8Async");
            try d.check(d.api.cuMemsetD8Async(f.fault.ptr, 0, f.fault.len, s.handle), "cuMemsetD8Async");
        }
        if (mid != null) try d.check(d.api.cuEventRecordWithFlags(mid, s.handle, if (external) 1 else 0), "cuEventRecordWithFlags");
        var a: cuda.Args = .{};
        a.add(f.table.ptr);
        a.add(@as(i32, @intCast(first)));
        a.add(@as(i32, @intCast(count)));
        a.add(f.statePtrs());
        try cuda.launch.launch(f.f, .{ .grid = .{ .x = f.grid, .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = f.dyn, .cooperative = true }, s, &a);
    }
};

/// Whether a forward is the fused path's: one decode row of one stream, no history rows or taps.
pub fn takes(m: *const Model, r: anytype) bool {
    return r.n == 1 and r.streams == 1 and !r.prompt and r.n_tiles == 1 and !m.hist_on and !m.taps_on and m.attn8 != null;
}

fn proj1(m: *const Model, w: cuda.qmmf.Weight, x: u64, out: u64, part: u64) !Proj {
    if (w.mode != .mxfp8 and w.mode != .fp8) return error.FusedWeightMode;
    const p = cuda.qmmf.planFor(1, w, true);
    if (p.fused or p.bm != 16) return error.FusedPlan;
    try bounds(0, 0, w.n, p.sk, w.npad);
    _ = m;
    return .{ .x = x, .codes = w.codes, .scales = w.scales, .out = out, .part = part, .scale = w.scale, .n = @intCast(w.n), .k = @intCast(w.k), .sk = @intCast(p.sk), .npad = @intCast(w.npad), .mode = @intCast(@backingInt(w.mode)) };
}

fn scaleOf(h: usize) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(h))));
}

/// Visit `v`'s table entry and argument structs (the same arguments visitOne gives today's kernels, n = 1).
pub fn fill(f: *const State, m: *Model, r: anytype, v: u32, hin: u64, hout: u64, args_host: *[arg_bytes]u8) !Visit {
    const t = m.c;
    const p = &m.w.plan;
    const i: u32 = p.visits.steps[v].layer;
    const spec = p.at(i);
    const L = &m.w.layers[i];
    const a = if (m.attn8) |x| &x[i] else return error.FusedAttn8;
    const h: usize = t.hidden;
    const plane = h;
    const global = spec.kind == .global;
    const msgs = p.messages(v);
    if (p.restarts(v) or i == 0) return error.FusedVisitKind; // Part 3: restarts and layer 0
    try bounds(spec.attn_queries, spec.blocks_in, 0, 0, 0);
    const base = f.args.ptr + @as(u64, v) * arg_bytes;
    @memset(args_host, 0);
    const put = struct {
        fn put(buf: *[arg_bytes]u8, off: usize, val: anytype) void {
            @memcpy(buf[off..][0..@sizeOf(@TypeOf(val))], std.mem.asBytes(&val));
        }
    }.put;
    const msg_at = m.s.msg_all + @as(u64, p.coreIndex(v)) * 4 * t.hidden * 4;
    // the attention input
    if (msgs) {
        var la: loop.MixArgs = .{ .partial = hin, .out = m.s.mixf, .out_stride = @intCast(plane), .n_blocks = @intCast(spec.blocks_in), .partial_f32 = 0, .nq = @intCast(spec.attn_queries), .rows = 1, .h = @intCast(h), .eps = t.eps, .scale = scaleOf(h) };
        for (0..spec.blocks_in) |b| la.blocks[b] = m.s.blocks + b * plane * 2;
        for (0..spec.attn_queries) |q| la.query[q] = L.attn_query + q * h * 4;
        put(args_host, arg_mix_attn, la);
        var g: loop.GateArgs = .{ .msg = msg_at, .msg_stride = @intCast(plane), .down = m.s.gdown, .ys = m.s.gys, .out = m.s.xin, .out_stride = @intCast(plane), .rows = 1, .h_dim = @intCast(h), .rank = @intCast(t.loop.?.rank), .eps_unit = t.eps, .eps = t.eps };
        const ctl = L.controllers.?;
        for (0..3) |c| {
            g.h[c] = m.s.mixf + (if (spec.attn_queries == 3) c else 0) * plane * 4;
            g.slot[c] = @intCast(c);
            g.down_w[c] = ctl[c].down;
            g.forget[c] = ctl[c].forget;
            g.gain[c] = ctl[c].gain;
            g.norm[c] = a.norms[c];
        }
        put(args_host, arg_gate_attn, g);
    } else {
        var mix: ops.MixArgs = .{ .partial = hin, .query = L.attn_query, .out = m.s.xin, .n_blocks = @intCast(spec.blocks_in), .partial_bf16 = 1, .nq = @intCast(spec.attn_queries), .rows = 1, .c = @intCast(h), .eps = t.eps };
        for (0..spec.blocks_in) |j| mix.blocks[j] = m.s.blocks + j * plane * 2;
        mix.norm = a.norms;
        put(args_host, arg_mix_attn, mix);
    }
    const separate = global or msgs;
    const xq = m.s.xin;
    const xk = if (separate) m.s.xin + plane * 2 else xq;
    const xv = if (separate) m.s.xin + 2 * plane * 2 else xq;
    const fz = m.s.fused;
    const fo = wts.Fused.of(t);
    const col = struct {
        fn col(b: u64, c: usize) u64 {
            return b + c * 2;
        }
    }.col;
    const ok = col(fz, fo.k);
    const ov = col(fz, fo.v);
    var pr: [3]Proj = .{ .{}, .{}, .{} };
    const qp_ = f.qpart.ptr;
    const call_part: u64 = 8 * 8704 * 4;
    pr[0] = try proj1(m, if (separate) a.qx else a.all, xq, fz, qp_);
    if (separate) {
        pr[1] = try proj1(m, a.wk, xk, ok, qp_ + call_part);
        pr[2] = try proj1(m, a.wv, xv, ov, qp_ + 2 * call_part);
    }
    const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
    const kvh: usize = t.local.kv;
    const taps_n = kvh * config.conv_taps;
    var taps: Taps = .{};
    if (global) taps = .{ .x = .{ xk, xv }, .w = .{ a.k_taps, a.v_taps }, .ones = m.w.ones, .out = .{ m.s.dyn[0], m.s.dyn[1] }, .n = @intCast(taps_n), .k = @intCast(h) };
    // the four convs (conv() of mini_visit: state by site, the call's segments, one row)
    const sites = [4]mm.Site{ .k, .v, .attn, .mlp };
    var cv: [4]ops.ConvArgs = .{
        .{ .x = ok, .state = 0, .w = a.k_conv, .dyn = if (global) m.s.dyn[0] else 0, .out = m.s.kc, .x_bf16 = 1, .heads = @intCast(kvh), .c = @intCast(kvw), .rows = 0, .mode = 0 },
        .{ .x = ov, .state = 0, .w = a.v_conv, .dyn = if (global) m.s.dyn[1] else 0, .out = m.s.vc, .x_bf16 = 1, .heads = @intCast(kvh), .c = @intCast(kvw), .rows = 0, .mode = 0 },
        .{ .x = m.s.o, .state = 0, .w = L.attn_conv, .base = if (spec.boundary) 0 else hin, .resid = m.s.h32, .x_bf16 = 1, .c = @intCast(h), .rows = 0, .mode = 1 },
        .{ .x = m.s.moe, .state = 0, .w = L.mlp_conv, .resid = m.s.h32, .out = hout, .x_bf16 = 1, .c = @intCast(h), .rows = 0, .mode = 2 },
    };
    var conv_ptr: [4]u64 = undefined;
    for (&cv, sites, 0..) |*c, site, ci| {
        try ops.convFits(c.*);
        c.state = m.conv[v][@intFromEnum(site)];
        c.segs = m.s.conv_segs;
        c.state_stride = 3 * @as(i64, c.c);
        c.rows = 1;
        put(args_host, arg_conv + 128 * ci, c.*);
        conv_ptr[ci] = base + arg_conv + 128 * ci;
    }
    // prep, rel table, attention, merge
    const l = if (global) m.full else m.sliding;
    const ksc = m.kvs[0][v];
    const vsc = m.kvs[1][v];
    if (ksc == 0) return error.FusedKv8;
    const prep: Prep = .{
        .q_raw = col(fz, fo.q),
        .k_raw = m.s.kc,
        .v_raw = m.s.vc,
        .rel_raw = col(fz, fo.rel),
        .q_scale = a.q_scale,
        .rel_scale = a.rel_scale,
        .rpos = m.s.pos,
        .rseg = m.s.seg_of,
        .segs = m.s.segs[@intFromBool(global)],
        .q_out = m.s.qp,
        .rel_out = m.s.relp,
        .kc = m.kv[0][v],
        .vc = m.kv[1][v],
        .ksc = ksc,
        .vsc = vsc,
        .heads = @intCast(t.local.q),
        .kv_heads = @intCast(t.local.kv),
        .ring = @intCast(l.ring),
        .global = @intFromBool(l.tau),
        .qk_clip = t.qk_clip,
        .rel_clip = t.rel_clip,
        .floor_pos = @floatFromInt(t.log_floor),
        .alpha = t.log_alpha,
        .eps = t.eps,
    };
    const stride = attn.slotsFor(l, m.context);
    const rows: attn.Rows = .{ .rows = 1, .heads = t.local.q, .kv_heads = t.local.kv, .pos = m.s.pos, .seg_of = m.s.seg_of, .segs = m.s.segs[@intFromBool(global)], .tiles = m.s.tiles, .n_tiles = 1, .last_pos = r.last_pos, .streams = 1 };
    if (attn.slotsFor(l, r.last_pos + 1) > stride) return error.Inkling2Partials;
    const db = attn.db8For(global);
    var at: Attn = .{
        .q = m.s.qp,
        .kc = m.kv[0][v],
        .vc = m.kv[1][v],
        .tab = m.s.tab,
        .segs = m.s.segs[@intFromBool(global)],
        .tiles = m.s.tiles,
        .part_o = m.s.part_o,
        .part_ml = m.s.part_ml,
        .ksc = ksc,
        .vsc = vsc,
        .sink = col(fz, fo.sink),
        .gate = col(fz, fo.gate),
        .rpos = m.s.pos,
        .att = m.s.att,
        .relp = m.s.relp,
        .rel_proj = a.rel_proj,
        .rtab = m.s.tab,
        .heads = @intCast(t.local.q),
        .kv_heads = @intCast(t.local.kv),
        .ring = @intCast(l.ring),
        .window = @intCast(l.window),
        .chunk = @intCast(l.chunk),
        .chunks = @intCast(stride),
        .split = 0,
        .zgrid = @intCast(attn.tileChunks(t.local.q / t.local.kv, l)),
        .slots_rh = @intCast(stride),
        .subs = 1,
        .scale = 1.0 / @sqrt(@as(f32, attn.head_dim)),
        .db = @intFromBool(db),
    };
    if (l.window == 0) {
        if (!attn.split(rows, stride, m.s.entries)) return error.FusedSplit;
        if (attn.zBlocksOf(stride * attn.subs, t.local.kv) != stride * attn.subs) return error.FusedZcap;
        at.split = 1;
        at.window = 0;
        at.slots_rh = @intCast(stride * attn.subs);
        at.chunks = at.slots_rh;
        at.zgrid = at.slots_rh;
        at.subs = @intCast(attn.subs);
    }
    // the MLP input
    const nb = spec.blocks_in + @intFromBool(spec.boundary);
    if (msgs) {
        var la: loop.MixArgs = .{ .partial = m.s.h32, .out = m.s.mixf, .out_stride = @intCast(plane), .n_blocks = @intCast(nb), .partial_f32 = 1, .nq = 1, .rows = 1, .h = @intCast(h), .eps = t.eps, .scale = scaleOf(h) };
        for (0..nb) |b| la.blocks[b] = m.s.blocks + b * plane * 2;
        la.query[0] = L.mlp_query;
        put(args_host, arg_mix_mlp, la);
        var g: loop.GateArgs = .{ .msg = msg_at, .msg_stride = @intCast(plane), .down = m.s.gdown, .ys = m.s.gys, .out = m.s.xm, .out_stride = @intCast(plane), .rows = 1, .h_dim = @intCast(h), .rank = @intCast(t.loop.?.rank), .eps_unit = t.eps, .eps = t.eps };
        const ctl = L.controllers.?;
        g.h[0] = m.s.mixf;
        g.slot[0] = 3;
        g.down_w[0] = ctl[3].down;
        g.forget[0] = ctl[3].forget;
        g.gain[0] = ctl[3].gain;
        g.norm[0] = L.mlp_norm;
        put(args_host, arg_gate_mlp, g);
    } else {
        var mix: ops.MixArgs = .{ .partial = m.s.h32, .query = L.mlp_query, .out = m.s.xm, .n_blocks = @intCast(nb), .partial_bf16 = 0, .nq = 1, .rows = 1, .c = @intCast(h), .eps = t.eps };
        for (0..nb) |j| mix.blocks[j] = m.s.blocks + j * plane * 2;
        mix.norm[0] = L.mlp_norm;
        put(args_host, arg_mix_mlp, mix);
    }
    // experts: NVFP4 split kernels only (MXFP8 layers are Part 3)
    const ex = &L.experts;
    if (ex.format != .nvfp4 or ex.caps[0] == 0) return error.FusedExperts;
    const caps = cuda.experts_mx.kernelCaps(ex.*);
    const moe: Moe = .{
        .xm = m.s.xm,
        .router = L.router,
        .scores = m.s.scores,
        .bias = L.router_bias,
        .pick = m.s.pick,
        .rw = m.s.rw,
        .members = m.s.plan.members,
        .items = m.s.plan.items,
        .counts = m.s.plan.counts,
        .up = ex.up,
        .down = ex.down,
        .up_scale = ex.up_scale,
        .down_scale = ex.down_scale,
        .act = m.s.act,
        .y = m.s.y,
        .moe = m.s.moe,
        .e = @intCast(t.experts),
        .k = @intCast(t.topk),
        .t = 16,
        .h = @intCast(h),
        .width = @intCast(t.expert_inter),
        .route_scale = t.route_scale,
        .gate_cap = caps[0],
        .up_cap = caps[1],
    };
    var vis: Visit = .{
        .global = @intFromBool(global),
        .gated = @intFromBool(msgs),
        .nq_attn = @intCast(spec.attn_queries),
        .n_proj = if (separate) 3 else 1,
        .mix_attn = base + arg_mix_attn,
        .mix_mlp = base + arg_mix_mlp,
        .gate_attn = if (msgs) base + arg_gate_attn else 0,
        .gate_mlp = if (msgs) base + arg_gate_mlp else 0,
        .conv = conv_ptr,
        .proj = pr,
        .wo = try proj1(m, a.wo, m.s.att, m.s.o, qp_ + 3 * call_part),
        .taps = taps,
        .prep = prep,
        .attn = at,
        .moe = moe,
    };
    if (spec.boundary) {
        vis.copy_bytes = @intCast(plane * 2);
        vis.copy_dst = m.s.blocks + @as(usize, i / t.block) * plane * 2;
        vis.copy_src = hin;
    }
    return vis;
}

// ---- check mode ----

const Kind = enum { bf16, f32, f16, i32, u8 };
const list_max = 96; // a gated global visit lists 65 buffers
const Buf = struct { name: []const u8, ptr: u64, len: usize, kind: Kind };

fn buffers(m: *Model, r: anytype, v: u32, hout: u64, vis: *const Visit, out: *[list_max]Buf) usize {
    const t = m.c;
    const h: usize = t.hidden;
    var n: usize = 0;
    const add = struct {
        fn add(o: *[list_max]Buf, k: *usize, name: []const u8, ptr: u64, len: usize, kind: Kind) void {
            if (ptr == 0 or len == 0 or k.* == list_max) return;
            o[k.*] = .{ .name = name, .ptr = ptr, .len = len, .kind = kind };
            k.* += 1;
        }
    }.add;
    const fo = wts.Fused.of(t);
    const kvw: usize = @as(usize, t.local.kv) * t.local.dim;
    const heads: usize = t.local.q;
    const nq: usize = @intCast(vis.nq_attn);
    if (vis.gated != 0) {
        add(out, &n, "mixf", m.s.mixf, nq * h * 4, .f32);
        add(out, &n, "gdown", m.s.gdown, 3 * t.loop.?.rank * 2, .bf16);
        add(out, &n, "gys", m.s.gys, 3 * h * 4, .f32);
        add(out, &n, "xin", m.s.xin, 3 * h * 2, .bf16);
    } else add(out, &n, "xin", m.s.xin, nq * h * 2, .bf16);
    add(out, &n, "fused", m.s.fused, fo.all * 2, .bf16);
    if (vis.global != 0) {
        add(out, &n, "dyn_k", m.s.dyn[0], @as(usize, @intCast(vis.taps.n)) * 4, .f32);
        add(out, &n, "dyn_v", m.s.dyn[1], @as(usize, @intCast(vis.taps.n)) * 4, .f32);
    }
    add(out, &n, "kc", m.s.kc, kvw * 2, .bf16);
    add(out, &n, "vc", m.s.vc, kvw * 2, .bf16);
    const names = [4][]const u8{ "conv_k", "conv_v", "conv_attn", "conv_mlp" };
    const slot: usize = r.slot0;
    for (0..4) |ci| {
        const c: usize = if (ci < 2) kvw else h;
        add(out, &n, names[ci], m.conv[v][ci] + slot * 3 * c * 4, 3 * c * 4, .f32);
    }
    add(out, &n, "qp", m.s.qp, heads * attn.head_dim * 2, .bf16);
    add(out, &n, "relp", m.s.relp, heads * t.d_rel * 2, .bf16);
    add(out, &n, "tab", m.s.tab, attn.tableBytes(1, t.local.q), .bf16);
    // the new key and value rows: e4m3 codes [kv][ring][128] after the slot's offset, scales [kv][ring]
    const ring: usize = @intCast(vis.prep.ring);
    const idx: usize = r.last_pos % ring;
    const off: usize = slot * kvw * ring;
    const kv_names = [2][8][]const u8{
        .{ "k8_h0", "k8_h1", "k8_h2", "k8_h3", "k8_h4", "k8_h5", "k8_h6", "k8_h7" },
        .{ "v8_h0", "v8_h1", "v8_h2", "v8_h3", "v8_h4", "v8_h5", "v8_h6", "v8_h7" },
    };
    for (0..2) |side| for (0..@min(t.local.kv, 8)) |kh| {
        add(out, &n, kv_names[side][kh], m.kv[side][v] + off + (kh * ring + idx) * attn.head_dim, attn.head_dim, .u8);
    };
    for (0..2) |side| for (0..@min(t.local.kv, 8)) |kh| {
        add(out, &n, if (side == 0) "k8_scale" else "v8_scale", m.kvs[side][v] + (off / attn.head_dim + kh * ring + idx) * 4, 4, .f32);
    };
    const slots_rh: usize = @intCast(vis.attn.slots_rh);
    add(out, &n, "part_o", m.s.part_o, heads * slots_rh * attn.head_dim * 4, .f32);
    add(out, &n, "part_ml", m.s.part_ml, heads * slots_rh * 8, .f32);
    add(out, &n, "att", m.s.att, heads * attn.head_dim * 2, .bf16);
    add(out, &n, "o", m.s.o, h * 2, .bf16);
    add(out, &n, "h32", m.s.h32, h * 4, .f32);
    if (vis.copy_bytes != 0) add(out, &n, "block", vis.copy_dst, @intCast(vis.copy_bytes), .bf16);
    add(out, &n, "xm", m.s.xm, h * 2, .bf16);
    add(out, &n, "scores", m.s.scores, t.experts * 4, .f32);
    add(out, &n, "pick", m.s.pick, t.topk * 4, .i32);
    add(out, &n, "rw", m.s.rw, t.topk * 4, .f32);
    const items = cuda.grouped.maxItems(t.topk, t.experts, 16);
    add(out, &n, "members", m.s.plan.members, t.topk * 4, .i32);
    add(out, &n, "items", m.s.plan.items, 3 * items * 4, .i32);
    add(out, &n, "counts", m.s.plan.counts, 2 * 4, .i32);
    add(out, &n, "act", m.s.act, t.topk * t.expert_inter * 2, .f16);
    add(out, &n, "y", m.s.y, t.topk * h * 4, .f32);
    add(out, &n, "moe", m.s.moe, h * 2, .bf16);
    add(out, &n, "hout", hout, h * 2, .bf16);
    return n;
}

fn value(kind: Kind, b: []const u8, i: usize) f64 {
    return switch (kind) {
        .bf16 => @floatCast(@as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[2 * i ..][0..2], .little)) << 16))),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[2 * i ..][0..2], .little)))),
        .f32 => @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, b[4 * i ..][0..4], .little)))),
        .i32 => @floatFromInt(std.mem.readInt(i32, b[4 * i ..][0..4], .little)),
        .u8 => @floatFromInt(b[i]),
    };
}

fn width(kind: Kind) usize {
    return switch (kind) {
        .bf16, .f16 => 2,
        .f32, .i32 => 4,
        .u8 => 1,
    };
}

fn every() usize {
    const S = struct {
        var v: ?usize = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_FUSED_EVERY")) |e| std.fmt.parseInt(usize, std.mem.span(e), 10) catch 64 else 64;
    return @max(S.v.?, 1);
}

/// TF_INKLING2_FUSED_POISON (default 1): every compared buffer except the conv states is filled with 0xFF bytes
/// before today's kernels and again before the fused launch (review A20). 0: the old method (no fill).
fn poisonOn() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_FUSED_POISON")) |e| e[0] != '0' else true;
    return S.v.?;
}

/// TF_INKLING2_FUSED_CONTROL=1: at step 1 of each visit, compare once with no fused launch by the old method (no
/// fill) and once by the fill method, then run the check as usual (review A20).
fn controlOn() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_FUSED_CONTROL")) |e| e[0] == '1' else false;
    return S.v.?;
}

/// TF_INKLING2_FUSED_DROP=<visit>:<group>:<sub>,...: on the steps with a position below TF_INKLING2_FUSED_DROP_POS
/// (default 4096), the next entry of the visit makes block 0 (or unit 0 of a mix counter) skip its arrival on counter
/// `sub` of counter group `group` (review A2, A3). The visit table carries it in `drop`: ((group + 1) << 3) | sub.
const Drop = struct { v: u32, code: i32 };
const drops_max = 16;
fn drops() []const Drop {
    const S = struct {
        var list: [drops_max]Drop = undefined;
        var n: usize = 0;
        var done = false;
    };
    if (!S.done) {
        S.done = true;
        if (std.c.getenv("TF_INKLING2_FUSED_DROP")) |e| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(e), ',');
            while (it.next()) |x| if (S.n < drops_max) {
                var f = std.mem.tokenizeScalar(u8, x, ':');
                const v = std.fmt.parseInt(u32, f.next() orelse continue, 10) catch continue;
                const g = std.fmt.parseInt(i32, f.next() orelse continue, 10) catch continue;
                const sub = std.fmt.parseInt(i32, f.next() orelse continue, 10) catch continue;
                S.list[S.n] = .{ .v = v, .code = ((g + 1) << 3) | (sub & 7) };
                S.n += 1;
            };
        }
    }
    return S.list[0..S.n];
}

fn dropPos() u32 {
    const S = struct {
        var v: ?u32 = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_FUSED_DROP_POS")) |e| std.fmt.parseInt(u32, std.mem.span(e), 10) catch 4096 else 4096;
    return S.v.?;
}

/// TF_INKLING2_FUSED_PROBE=<visit>:<phase>,...: on the steps that print no [fused-time] line, step s of a visit
/// probes entry s % n of its n entries: each block stores the globaltimer at the phase start, at its arrival at the
/// phase's last wait and at the wait's exit, and named units store stamps (fused_visit.cuh fz_probe).
const Probe = struct { v: u32, ph: i32 };
fn probes() []const Probe {
    const S = struct {
        var list: [64]Probe = undefined;
        var n: usize = 0;
        var done = false;
    };
    if (!S.done) {
        S.done = true;
        if (std.c.getenv("TF_INKLING2_FUSED_PROBE")) |e| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(e), ',');
            while (it.next()) |x| if (S.n < S.list.len) {
                var f = std.mem.tokenizeScalar(u8, x, ':');
                const v = std.fmt.parseInt(u32, f.next() orelse continue, 10) catch continue;
                const ph = std.fmt.parseInt(i32, f.next() orelse continue, 10) catch continue;
                S.list[S.n] = .{ .v = v, .ph = ph };
                S.n += 1;
            };
        }
    }
    return S.list[0..S.n];
}

fn probeFor(v: u32, step: usize) i32 {
    if (step % every() == 0) return -1;
    var n: usize = 0;
    for (probes()) |p| n += @intFromBool(p.v == v);
    if (n == 0) return -1;
    var k: usize = step % n;
    for (probes()) |p| if (p.v == v) {
        if (k == 0) return p.ph;
        k -= 1;
    };
    return -1;
}

/// TF_INKLING2_FUSED_FIX: the review corrections on (fused_visit.cuh fz_fix: 1 A9, 2 A11, 4 A12, 8 A18, 32 A1; 16 A17, the
/// host clears its own visits only); default all. TF_INKLING2_FUSED_FIX_AB=1: steps s with (s / every) even run none,
/// so one run holds the phase times before and after.
const fix_all: i32 = 63;
const fix_own_clear: i32 = 16;
fn fixFor(step: usize) i32 {
    const S = struct {
        var mask: ?i32 = null;
        var ab = false;
    };
    if (S.mask == null) {
        S.mask = if (std.c.getenv("TF_INKLING2_FUSED_FIX")) |e| std.fmt.parseInt(i32, std.mem.span(e), 0) catch fix_all else fix_all;
        S.ab = if (std.c.getenv("TF_INKLING2_FUSED_FIX_AB")) |e| e[0] == '1' else false;
    }
    if (S.ab and (step / every()) % 2 == 0) return 0;
    return S.mask.? & fix_all;
}

/// TF_INKLING2_FUSED_GRAPH (default 1): today's visit also runs as a captured graph (the live launch method), and
/// the fused launch runs from a captured graph on even steps (review A22). 0: the old method only.
fn graphOn() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_FUSED_GRAPH")) |e| e[0] != '0' else true;
    return S.v.?;
}

/// Today's visit captured into a graph, instantiated and uploaded; the stream is idle at return.
fn todayGraph(m: *Model, k: *const Kernels, s: cuda.Stream, r: anytype, v: u32, hin: u64, hout: u64, today: anytype) !cuda.graph.Exec {
    try s.synchronize();
    try cuda.graph.beginCapture(s, .thread_local);
    today.run(m, k, s, r, v, hin, hout) catch |e| {
        if (cuda.graph.endCapture(s)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return e;
    };
    var g = try cuda.graph.endCapture(s);
    defer g.deinit();
    var ex = try g.instantiate();
    errdefer ex.deinit();
    try ex.upload(s);
    try s.synchronize();
    return ex;
}

/// The fused launch of visit v captured into a graph (memsets, event `mid`, the cooperative launch); null when the
/// driver refuses the capture.
fn fusedGraph(f: *State, s: cuda.Stream, v: u32, own: bool) !?cuda.graph.Exec {
    try s.synchronize();
    try cuda.graph.beginCapture(s, .thread_local);
    f.launchFix(s, v, 1, f.ev[6], true, own) catch |e| {
        if (cuda.graph.endCapture(s)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        std.debug.print("[fused] the fused launch does not capture ({s}): stream launches only\n", .{@errorName(e)});
        return null;
    };
    var g = cuda.graph.endCapture(s) catch |e| {
        std.debug.print("[fused] the fused capture ends with {s}: stream launches only\n", .{@errorName(e)});
        return null;
    };
    defer g.deinit();
    var ex = g.instantiate() catch |e| {
        std.debug.print("[fused] the fused graph does not instantiate ({s}): stream launches only\n", .{@errorName(e)});
        return null;
    };
    errdefer ex.deinit();
    try ex.upload(s);
    try s.synchronize();
    return ex;
}

/// At load in check mode: the query pointers of each layer (review A7) and the resident blocks a multiprocessor of
/// today's fp8 attention kernels at 128 threads against the fused kernel (review A10).
/// At load in check mode: the bounds of review A6 against the senders the review names.
fn reportBounds() void {
    const cases = [_]struct { name: []const u8, nq: usize, nb: usize, n: usize }{
        .{ .name = "loop mix nq 3 (live)", .nq = 3, .nb = 8, .n = 0 },
        .{ .name = "restart message mix nq 4", .nq = 4, .nb = 8, .n = 0 },
        .{ .name = "qmmf all N 8600 (live)", .nq = 0, .nb = 0, .n = 8600 },
        .{ .name = "head N 200064 (3126 tiles)", .nq = 0, .nb = 0, .n = 200064 },
    };
    for (cases) |c| {
        if (bounds(c.nq, c.nb, c.n, 4, c.n)) |_| {
            std.debug.print("[fused-bounds] {s}: fits\n", .{c.name});
        } else |e| std.debug.print("[fused-bounds] {s}: refused ({s})\n", .{ c.name, @errorName(e) });
    }
}

pub fn reportLoad(m: *const Model, k: *const Kernels) !void {
    const d = m.d;
    reportBounds();
    for (m.w.layers, 0..) |L, i| std.debug.print("[fused-load] layer {d} attn_query 0x{x} mlp_query 0x{x}\n", .{ i, L.attn_query, L.mlp_query });
    var per: c_int = 0;
    try d.check(d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor(&per, k.attn.attn8.handle, threads, 0), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    var per_db: c_int = 0;
    try d.check(d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_db, k.attn.attn8_db.handle, threads, attn.db8_shared), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    const f = &m.fused.?;
    var per_f: c_int = 0;
    try d.check(d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_f, f.f.handle, threads, f.dyn), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    std.debug.print("[fused-load] occupancy at 128 threads, blocks a multiprocessor: tf_ink2_attn8 {d}, tf_ink2_attn8_db {d} ({d} B dynamic), tf_ink2_fused_visit {d} ({d} B dynamic), grid {d}\n", .{ per, per_db, attn.db8_shared, per_f, f.dyn, f.grid });
}

/// The fault word as text: (code << 16) | (block + 1), code = ((group + 1) << 3) | sub (fused_visit.cuh fz_sync).
fn faultText(buf: []u8, w: u32) []const u8 {
    if (w == 0) return "none";
    const code = w >> 16;
    return std.fmt.bufPrint(buf, "group {d} sub {d} block {d}", .{ @as(i64, code >> 3) - 1, code & 7, @as(i64, w & 0xffff) - 1 }) catch "?";
}

/// Fill every compared buffer except the conv states (the visit's inputs) with 0xFF bytes.
fn poison(d: *const cuda.Driver, s: cuda.Stream, list: []const Buf) !void {
    for (list) |b| if (!std.mem.startsWith(u8, b.name, "conv_")) try d.check(d.api.cuMemsetD8Async(b.ptr, 0xff, b.len, s.handle), "cuMemsetD8Async");
}

/// Compare each listed buffer with its bytes in `ref`; one line a buffer into `lines`. Returns the differing bytes.
fn compare(f: *State, list: []const Buf, v: u32, step: usize, pos: u32, lines: ?*std.ArrayList(u8)) !usize {
    return compareAt(f, list, 0, v, step, pos, lines);
}

/// `compare` against the bytes of `ref` from offset `base`.
fn compareAt(f: *State, list: []const Buf, base: usize, v: u32, step: usize, pos: u32, lines: ?*std.ArrayList(u8)) !usize {
    const d = f.d;
    const gpa = std.heap.page_allocator;
    var diff_total: usize = 0;
    var at: usize = base;
    for (list) |b| {
        const want = try gpa.alloc(u8, b.len);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, b.len);
        defer gpa.free(got);
        try cuda.memory.download(d, f.ref.ptr + at, want);
        try cuda.memory.download(d, b.ptr, got);
        var bytes: usize = 0;
        for (want, got) |x, y| bytes += @intFromBool(x != y);
        var big: f64 = 0;
        if (bytes != 0) {
            const w = width(b.kind);
            for (0..b.len / w) |e| big = @max(big, @abs(value(b.kind, want, e) - value(b.kind, got, e)));
        }
        diff_total += bytes;
        if (lines) |l| try l.print(gpa, "[fused-check] visit {d} step {d} pos {d} {s} bytes {d} differing {d} largest {e}\n", .{ v, step, pos, b.name, b.len, bytes, big });
        at += b.len;
    }
    return diff_total;
}

/// TF_INKLING2_FUSED_PAIR=<v>: the check of visit v launches visits v and v + 1 in one launch (count 2, reviews A1,
/// A5, A8). 0 or unset: none.
fn pairVisit() u32 {
    const S = struct {
        var v: ?u32 = null;
    };
    if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_FUSED_PAIR")) |e| std.fmt.parseInt(u32, std.mem.span(e), 10) catch 0 else 0;
    return S.v.?;
}

fn overlaps(b: Buf, list: []const Buf) bool {
    for (list) |o| if (b.ptr < o.ptr + o.len and o.ptr < b.ptr + b.len) return true;
    return false;
}

/// Visits v and v + 1 by today's kernels, then by one fused launch of count 2 from the same state (every compared
/// buffer filled first); compare every buffer of v + 1 and the buffers of v that v + 1 does not write again. Then the
/// state of v + 1 goes back to its state before the pair: the forward runs visit v + 1 next as usual.
fn checkPair(f: *State, m: *Model, k: *const Kernels, s: cuda.Stream, r: anytype, v: u32, hin: u64, hout: u64, today: anytype) !bool {
    const d = f.d;
    const gpa = std.heap.page_allocator;
    const w = v + 1;
    f.at = "pair-fill";
    var host_v: [arg_bytes]u8 = undefined;
    var host_w: [arg_bytes]u8 = undefined;
    var vis_v = fill(f, m, r, v, hin, hout, &host_v) catch return false;
    var vis_w = fill(f, m, r, w, hout, hin, &host_w) catch |e| {
        if (f.steps[v] == 0) std.debug.print("[fused-pair] visit {d}: not a Part 2 visit ({s}), single checks\n", .{ w, @errorName(e) });
        return false;
    };
    const step = f.steps[v];
    const fix = fixFor(step);
    vis_v.fix = fix & ~fix_own_clear;
    vis_w.fix = fix & ~fix_own_clear;
    try s.synchronize();
    try f.args.upload(@as(usize, v) * arg_bytes, &host_v);
    try f.args.upload(@as(usize, w) * arg_bytes, &host_w);
    try f.table.upload(@as(usize, v) * @sizeOf(Visit), std.mem.asBytes(&vis_v));
    try f.table.upload(@as(usize, w) * @sizeOf(Visit), std.mem.asBytes(&vis_w));
    var la: [list_max]Buf = undefined;
    var lb: [list_max]Buf = undefined;
    const na = buffers(m, r, v, hout, &vis_v, &la);
    const nb = buffers(m, r, w, hin, &vis_w, &lb);
    // the state before the pair: the conv states of both visits, and hin (visit w's output buffer)
    var saved: usize = 0;
    for ([2][]const Buf{ la[0..na], lb[0..nb] }) |l| for (l) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
        try m.copy(s, f.save.ptr + saved, b.ptr, b.len);
        saved += b.len;
    };
    const h_bytes: usize = m.c.hidden * 2;
    const hin_at = saved;
    try m.copy(s, f.save.ptr + hin_at, hin, h_bytes);
    f.at = "pair-today";
    try today.run(m, k, s, r, v, hin, hout);
    var at: usize = 0;
    for (la[0..na]) |b| {
        try m.copy(s, f.ref.ptr + at, b.ptr, b.len);
        at += b.len;
    }
    const base_b = at;
    try today.run(m, k, s, r, w, hout, hin);
    for (lb[0..nb]) |b| {
        if (at + b.len > f.ref.len) return error.FusedRefFull;
        try m.copy(s, f.ref.ptr + at, b.ptr, b.len);
        at += b.len;
    }
    // fill, then the inputs back: conv states and hin
    for ([2][]const Buf{ la[0..na], lb[0..nb] }) |l| try poison(d, s, l);
    saved = 0;
    for ([2][]const Buf{ la[0..na], lb[0..nb] }) |l| for (l) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
        try m.copy(s, b.ptr, f.save.ptr + saved, b.len);
        saved += b.len;
    };
    try m.copy(s, hin, f.save.ptr + hin_at, h_bytes);
    f.at = "pair-fused";
    try s.synchronize();
    try d.check(d.api.cuEventRecord(f.ev[2], s.handle), "cuEventRecord");
    try f.launchFix(s, v, 2, null, false, (fix & fix_own_clear) != 0);
    try d.check(d.api.cuEventRecord(f.ev[3], s.handle), "cuEventRecord");
    try s.synchronize();
    var ms: f32 = 0;
    try d.check(d.api.cuEventElapsedTime(&ms, f.ev[2], f.ev[3]), "cuEventElapsedTime");
    var faults: [2]u32 = undefined;
    try cuda.memory.download(d, f.fault.ptr + @as(u64, v) * 4, std.mem.sliceAsBytes(&faults));
    // compare: visit w fully, visit v where w does not write again
    var nka: usize = 0;
    var skipped: usize = 0;
    var lines: std.ArrayList(u8) = .empty;
    defer lines.deinit(gpa);
    var diff_v: usize = 0;
    at = 0;
    for (la[0..na]) |b| {
        if (!overlaps(b, lb[0..nb])) {
            nka += 1;
            diff_v += try compareAt(f, &.{b}, at, v, step, r.last_pos, &lines);
        } else skipped += 1;
        at += b.len;
    }
    const diff_w = try compareAt(f, lb[0..nb], base_b, w, step, r.last_pos, &lines);
    // whether the two visits' picks differ (review A1: the plan is written again by visit w)
    var picks_differ = false;
    {
        var pa: usize = 0;
        var pb: usize = base_b;
        var ia: ?usize = null;
        var ib: ?usize = null;
        var len: usize = 0;
        for (la[0..na]) |b| {
            if (std.mem.eql(u8, b.name, "pick")) {
                ia = pa;
                len = b.len;
            }
            pa += b.len;
        }
        for (lb[0..nb]) |b| {
            if (std.mem.eql(u8, b.name, "pick")) ib = pb;
            pb += b.len;
        }
        if (ia != null and ib != null and len <= 64) {
            var x: [64]u8 = undefined;
            var y: [64]u8 = undefined;
            try cuda.memory.download(d, f.ref.ptr + ia.?, x[0..len]);
            try cuda.memory.download(d, f.ref.ptr + ib.?, y[0..len]);
            picks_differ = !std.mem.eql(u8, x[0..len], y[0..len]);
        }
    }
    // today's bytes back (v, then w: a buffer both write ends with w's bytes), then visit w's conv states before the
    // pair (the forward runs w next)
    at = 0;
    for ([2][]const Buf{ la[0..na], lb[0..nb] }) |l| for (l) |b| {
        try m.copy(s, b.ptr, f.ref.ptr + at, b.len);
        at += b.len;
    };
    saved = 0;
    for (la[0..na]) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
        saved += b.len;
    };
    for (lb[0..nb]) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
        try m.copy(s, b.ptr, f.save.ptr + saved, b.len);
        saved += b.len;
    };
    try s.synchronize();
    f.steps[v] += 1;
    const bad = diff_v != 0 or diff_w != 0 or faults[0] != 0 or faults[1] != 0;
    if (bad) f.bad[v] += 1;
    if (bad or step % every() == 0) std.debug.print("{s}", .{lines.items});
    std.debug.print("[fused-pair] visits {d} {d} step {d} pos {d} fused_us {d:.2} fix {d} faults 0x{x} 0x{x} differing {d} (visit {d}, {d} buffers, {d} not compared: visit {d} writes them again) {d} (visit {d}, {d} buffers) picks differ {d}\n", .{ v, w, step, r.last_pos, ms * 1000, fix, faults[0], faults[1], diff_v, v, nka, skipped, w, diff_w, w, nb, @intFromBool(picks_differ) });
    return true;
}

/// Visit `v` by today's kernels (`today`), then by the fused kernel from the same state; compare; keep today's bytes.
pub fn check(f: *State, m: *Model, k: *const Kernels, s: cuda.Stream, r: anytype, v: u32, hin: u64, hout: u64, today: anytype) !void {
    if (pairVisit() == v and v + 1 < visits_max) {
        const done = checkPair(f, m, k, s, r, v, hin, hout, today) catch |e| {
            std.debug.print("[fused-pair] visit {d} pos {d}: {s} at step {s}\n", .{ v, r.last_pos, @errorName(e), f.at });
            return e;
        };
        if (done) return;
    }
    checkSteps(f, m, k, s, r, v, hin, hout, today) catch |e| {
        std.debug.print("[fused-check] visit {d} pos {d}: {s} at step {s}\n", .{ v, r.last_pos, @errorName(e), f.at });
        return e;
    };
}

fn printProbe(f: *State, v: u32, step: usize, pos: u32, ph: i32, ms_fused: f32, fix: i32) !void {
    const gpa = std.heap.page_allocator;
    const n = 3 * @as(usize, f.grid) + probe_stamps;
    const p = try gpa.alloc(u64, n);
    defer gpa.free(p);
    try cuda.memory.download(f.d, f.probe.ptr, std.mem.sliceAsBytes(p));
    var t0: u64 = std.math.maxInt(u64);
    for (p[0..f.grid]) |x| if (x != 0) {
        t0 = @min(t0, x);
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try out.print(gpa, "[fused-probe] visit {d} step {d} pos {d} phase {d} fused_us {d:.2} fix {d} grid {d} ns from the first start:", .{ v, step, pos, ph, ms_fused * 1000, fix, f.grid });
    const names = [_][]const u8{ " start", " arrive", " exit", " stamps" };
    for (0..4) |part| {
        try out.appendSlice(gpa, names[part]);
        const lo = part * f.grid;
        const hi = if (part == 3) n else lo + f.grid;
        for (p[lo..hi]) |x| {
            if (x == 0 or t0 == std.math.maxInt(u64)) try out.appendSlice(gpa, " -") else try out.print(gpa, " {d}", .{@as(i64, @intCast(x)) - @as(i64, @intCast(t0))});
        }
    }
    try out.append(gpa, '\n');
    std.debug.print("{s}", .{out.items});
}

fn checkSteps(f: *State, m: *Model, k: *const Kernels, s: cuda.Stream, r: anytype, v: u32, hin: u64, hout: u64, today: anytype) !void {
    const d = f.d;
    const gpa = std.heap.page_allocator; // not the model's allocator: the check's buffers live for one visit
    f.at = "fill";
    var host: [arg_bytes]u8 = undefined;
    var vis = fill(f, m, r, v, hin, hout, &host) catch |e| {
        if (f.steps[v] == 0) std.debug.print("[fused-check] visit {d}: not a Part 2 visit ({s}), today's kernels\n", .{ v, @errorName(e) });
        f.steps[v] += 1;
        return today.run(m, k, s, r, v, hin, hout);
    };
    // the drop entry of this step (review A2, A3): the table entry carries it
    const step = f.steps[v];
    var drop: i32 = 0;
    if (r.last_pos < dropPos()) {
        var k_: usize = 0;
        for (drops()) |e| if (e.v == v) {
            if (k_ == f.drop_at[v]) {
                drop = e.code;
                break;
            }
            k_ += 1;
        };
        if (drop != 0) f.drop_at[v] += 1;
    }
    vis.drop = drop;
    const probe = probeFor(v, step);
    vis.probe = probe + 1;
    const fix = fixFor(step);
    vis.fix = fix & ~fix_own_clear;
    f.at = "upload";
    try s.synchronize();
    try f.args.upload(@as(usize, v) * arg_bytes, &host);
    try f.table.upload(@as(usize, v) * @sizeOf(Visit), std.mem.asBytes(&vis));
    var list: [list_max]Buf = undefined;
    const nbuf = buffers(m, r, v, hout, &vis, &list);
    const fill_on = poisonOn();
    // the conv states before the visit
    var saved: usize = 0;
    for (list[0..nbuf]) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
        try m.copy(s, f.save.ptr + saved, b.ptr, b.len);
        saved += b.len;
    };
    if (fill_on) try poison(d, s, list[0..nbuf]);
    if (probe >= 0) try d.check(d.api.cuMemsetD8Async(f.probe.ptr, 0, f.probe.len, s.handle), "cuMemsetD8Async");
    // today's kernels: one by one from an idle stream (the method of deb490d58), then as a captured graph (the live
    // launch method, review A22) from the same state. The graph's bytes are the reference.
    f.at = "today";
    try s.synchronize();
    try d.check(d.api.cuEventRecord(f.ev[0], s.handle), "cuEventRecord");
    try today.run(m, k, s, r, v, hin, hout);
    try d.check(d.api.cuEventRecord(f.ev[1], s.handle), "cuEventRecord");
    var ms_graph: f32 = -1;
    if (graphOn()) {
        saved = 0;
        for (list[0..nbuf]) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
            try m.copy(s, b.ptr, f.save.ptr + saved, b.len);
            saved += b.len;
        };
        if (fill_on) try poison(d, s, list[0..nbuf]);
        f.at = "today-graph";
        var ex = try todayGraph(m, k, s, r, v, hin, hout, today);
        defer ex.deinit();
        try d.check(d.api.cuEventRecord(f.ev[4], s.handle), "cuEventRecord");
        try ex.launchOn(s);
        try d.check(d.api.cuEventRecord(f.ev[5], s.handle), "cuEventRecord");
        try s.synchronize();
        try d.check(d.api.cuEventElapsedTime(&ms_graph, f.ev[4], f.ev[5]), "cuEventElapsedTime");
    }
    f.at = "ref";
    var at: usize = 0;
    for (list[0..nbuf]) |b| {
        if (at + b.len > f.ref.len) return error.FusedRefFull;
        try m.copy(s, f.ref.ptr + at, b.ptr, b.len);
        at += b.len;
    }
    saved = 0;
    for (list[0..nbuf]) |b| if (std.mem.startsWith(u8, b.name, "conv_")) {
        try m.copy(s, b.ptr, f.save.ptr + saved, b.len);
        saved += b.len;
    };
    // the control (review A20): no fused launch, compared by the old method, then by the fill method
    if (controlOn() and step == 1) {
        try s.synchronize();
        const old_n = try compare(f, list[0..nbuf], v, step, r.last_pos, null);
        try poison(d, s, list[0..nbuf]);
        try s.synchronize();
        const new_n = try compare(f, list[0..nbuf], v, step, r.last_pos, null);
        std.debug.print("[fused-control] visit {d} step {d} pos {d} no fused launch: differing bytes {d} by the old method, {d} by the fill method, buffers {d}\n", .{ v, step, r.last_pos, old_n, new_n, nbuf });
    }
    if (fill_on) try poison(d, s, list[0..nbuf]);
    // the fused launch: on even steps from a captured graph after an idle stream (review A22); on odd steps as a stream
    // launch behind the queued copies (the method of deb490d58), with an event after the memsets (review A17)
    f.at = "fused";
    var method: []const u8 = "stream-queued";
    var ms_memset: f32 = -1;
    var fx: ?cuda.graph.Exec = null;
    defer if (fx) |*e| e.deinit();
    if (graphOn() and f.graph_ok and step % 2 == 0) {
        fx = try fusedGraph(f, s, v, (fix & fix_own_clear) != 0);
        if (fx == null) f.graph_ok = false;
    }
    if (fx) |e| {
        method = "graph";
        try d.check(d.api.cuEventRecord(f.ev[2], s.handle), "cuEventRecord");
        try e.launchOn(s);
        try d.check(d.api.cuEventRecord(f.ev[3], s.handle), "cuEventRecord");
    } else {
        try d.check(d.api.cuEventRecord(f.ev[2], s.handle), "cuEventRecord");
        try f.launchFix(s, v, 1, f.ev[7], false, (fix & fix_own_clear) != 0);
        try d.check(d.api.cuEventRecord(f.ev[3], s.handle), "cuEventRecord");
    }
    try s.synchronize();
    var ms_today: f32 = 0;
    var ms_fused: f32 = 0;
    try d.check(d.api.cuEventElapsedTime(&ms_today, f.ev[0], f.ev[1]), "cuEventElapsedTime");
    try d.check(d.api.cuEventElapsedTime(&ms_fused, f.ev[2], f.ev[3]), "cuEventElapsedTime");
    if (fx != null) {
        d.check(d.api.cuEventElapsedTime(&ms_memset, f.ev[2], f.ev[6]), "cuEventElapsedTime") catch {
            ms_memset = -1;
        };
    } else try d.check(d.api.cuEventElapsedTime(&ms_memset, f.ev[2], f.ev[7]), "cuEventElapsedTime");
    var fault: u32 = 0;
    try cuda.memory.download(d, f.fault.ptr + @as(u64, v) * 4, std.mem.asBytes(&fault));
    var times: [times_visit]u64 = undefined;
    try cuda.memory.download(d, f.times.ptr + @as(u64, v) * times_visit * 8, std.mem.sliceAsBytes(&times));
    // compare
    f.at = "compare";
    f.steps[v] += 1;
    var lines: std.ArrayList(u8) = .empty;
    defer lines.deinit(gpa);
    const diff_total = try compare(f, list[0..nbuf], v, step, r.last_pos, &lines);
    // today's bytes back (the reply stays today's)
    at = 0;
    for (list[0..nbuf]) |b| {
        try m.copy(s, b.ptr, f.ref.ptr + at, b.len);
        at += b.len;
    }
    try s.synchronize();
    var fbuf: [64]u8 = undefined;
    if (drop != 0) {
        std.debug.print("[fused-drop] visit {d} step {d} pos {d} group {d} sub {d} fused_us {d:.2} fault 0x{x} ({s}) differing {d}\n", .{ v, step, r.last_pos, (drop >> 3) - 1, drop & 7, ms_fused * 1000, fault, faultText(&fbuf, fault), diff_total });
    } else if (diff_total != 0 or fault != 0) f.bad[v] += 1;
    if (probe >= 0) try printProbe(f, v, step, r.last_pos, probe, ms_fused, fix);
    if (drop == 0 and (diff_total != 0 or fault != 0 or step % every() == 0)) {
        std.debug.print("{s}", .{lines.items});
        var tl: std.ArrayList(u8) = .empty;
        defer tl.deinit(gpa);
        var ph: usize = 1;
        while (ph < times_visit and times[ph] != 0 and times[ph] >= times[ph - 1]) : (ph += 1) try tl.print(gpa, " {d}", .{times[ph] - times[ph - 1]});
        std.debug.print("[fused-time] visit {d} step {d} pos {d} today_us {d:.2} fused_us {d:.2} fault 0x{x} fill {d} today_graph_us {d:.2} memset_us {d:.2} method {s} fix {d} phases_ns{s}\n", .{ v, step, r.last_pos, ms_today * 1000, ms_fused * 1000, fault, @intFromBool(fill_on), ms_graph * 1000, ms_memset * 1000, method, fix, tl.items });
        if (fault != 0) std.debug.print("[fused-fault] visit {d} step {d} {s}\n", .{ v, step, faultText(&fbuf, fault) });
        std.debug.print("[fused-sum] visit {d} steps {d} bad {d} buffers {d} differing {d}\n", .{ v, step + 1, f.bad[v], nbuf, diff_total });
    }
    try d.check(d.api.cuMemsetD8Async(f.times.ptr + @as(u64, v) * times_visit * 8, 0, times_visit * 8, s.handle), "cuMemsetD8Async");
}

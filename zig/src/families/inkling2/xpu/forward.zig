//! Mini's whole 48-visit forward on the XPU backend, for prompt chunks and decode: the port of `metal/forward.zig`, with the
//! same visit order, the same rounding points and the same buffers, queued on one in-order stream.
//! K/V rows are bf16 (the attention kernels read and write nothing else); an int8 cache is refused.
const std = @import("std");
const xpu = @import("xpu");
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const experts = @import("experts.zig");
const loop = @import("loop.zig");
const runtime = @import("runtime.zig");
const dense_kernels = @import("dense.zig");
const dense_q8_kernels = @import("dense_q8.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;
const Tensor = model_file.Tensor;

const rt = xpu.rt;

/// A device address, offset already applied.
pub const Ptr = ?*anyopaque;
pub const Segment = struct { cache: *state.Cache, rows: usize };
pub const Options = struct {
    /// Logits for every row (teacher forcing) in `Scratch.logits_all`; otherwise the last row's go to the cache.
    all: bool = false,
    /// Where the DSpark taps go ([row][tap][H] bf16), or none.
    taps: ?Ptr = null,
};

pub const FinalPlan = struct { work_rows: usize, logits_rows: usize, logits_row: usize, capture_rows: usize };

/// Features normalize all rows; without full logits, only the last unembeds. taps=null preserves the original plan.
pub fn finalPlan(rows: usize, all: bool, taps: bool) FinalPlan {
    std.debug.assert(rows > 0);
    return .{
        .work_rows = if (all or taps) rows else 1,
        .logits_rows = if (all) rows else 1,
        .logits_row = if (!all and taps) rows - 1 else 0,
        .capture_rows = if (taps) rows else 0,
    };
}

/// Rows the decode gather experts take in one call; more go through the grouped prompt kernels.
pub const decode_rows = 16;

/// Diagnostic, off by default: when set, every visit copies its routes (rows x 8 int32 expert ids) here, visit-major
/// (`visit * rows * 8` ids), for the measurement of how many distinct experts a window reads.
pub var route_trace: ?rt.Buffer = null;

fn at(b: rt.Buffer, off: usize) Ptr {
    return @ptrFromInt(@intFromPtr(b.ptr.?) + off);
}
fn plus(p: Ptr, off: usize) Ptr {
    return @ptrFromInt(@intFromPtr(p.?) + off);
}
fn row(b: rt.Buffer, i: usize, row_bytes: usize) Ptr {
    return at(b, i * row_bytes);
}
fn tensor(t: Tensor) Ptr {
    return t.buf.ptr;
}
fn aref(b: rt.Buffer, off: usize) attention.Ref {
    return .{ .ptr = b.ptr, .off = off };
}
fn sref(r: state.Ref) attention.Ref {
    return .{ .ptr = r.buf.ptr, .off = r.off };
}
fn arRow(s: *state.Scratch, role: usize, rows: usize, i: usize, gated: bool) Ptr {
    return at(if (gated) s.gated else s.ar, (role * rows + i) * state.hidden * 4);
}
fn messageRow(s: *state.Scratch, core: usize, role: usize, i: usize) Ptr {
    return at(s.messages, ((i * state.message_entries + core * 4 + role) * state.hidden) * 4);
}

/// One forward in flight: the model, its kernels, the stream and the scratch it writes.
const Run = struct {
    m: *const Model,
    p: *const model_file.Pipelines,
    s: xpu.Stream,
    sc: *state.Scratch,

    fn copyBf16(g: Run, from: Ptr, to: Ptr, count: usize) !void {
        try runtime.copyBf16(&g.p.runtime, g.s, from, to, 1, count);
    }
    fn captureTap(g: Run, from: Ptr, to: Ptr, rows: usize, channel: usize) !void {
        std.debug.assert(channel < state.feature_taps);
        try runtime.captureTap(&g.p.runtime, g.s, from, to, rows, channel);
    }
    fn widen(g: Run, from: Ptr, to: Ptr, rows: usize) !void {
        try runtime.widen(&g.p.runtime, g.s, from, to, rows, state.hidden);
    }

    fn norm(g: Run, input: Ptr, weight: Tensor, out: Ptr, rows: usize) !void {
        try dense_kernels.norm(&g.p.dense, g.s, input, tensor(weight), out, @intCast(rows));
    }

    /// Every projection uses the shared matvec's FP32 reduction and one BF16 store: over BF16 weights, or over Q8_0 blocks
    /// (q8.zig) when the model loaded them re-encoded.
    fn dense(g: Run, weight: Tensor, input: Ptr, out: Ptr, rows: usize) !void {
        const k: u32 = @intCast(weight.shape[weight.rank - 1]);
        const n: u32 = @intCast(weight.shape[weight.rank - 2]);
        if (weight.q8) return dense_q8_kernels.dense(&g.p.dense_q8, g.s, tensor(weight), input, out, @intCast(rows), n, k);
        try dense_kernels.dense(&g.p.dense, g.s, tensor(weight), input, out, @intCast(rows), n, k);
    }

    /// One launch for every row: the AR of a row reads that row's blocks and partial and nothing of another row's, so batch
    /// composition never mixes rows (the arithmetic of a row is the single-row launch's, bit for bit).
    fn residual(g: Run, queries: Tensor, sources: usize, rows: usize) !void {
        const s = g.sc;
        try glue.arRows(&g.p.glue, g.s, at(s.blocks, 0), at(s.partial, 0), tensor(queries), at(s.mlp_ar, 0), @intCast(sources), 1, @intCast(rows), state.block_slots * state.hidden);
    }

    fn bankAndReset(g: Run, rows: usize) !void {
        const s = g.sc;
        for (0..rows) |i| {
            // The second completed block of the previous pass is B12; its final p is P18.
            try g.copyBf16(at(s.blocks, (i * state.block_slots + 2) * state.hidden * 2), row(s.bank0, i, state.hidden * 2), state.hidden);
        }
        try g.copyBf16(at(s.p, 0), at(s.bank1, 0), rows * state.hidden);
        try loop.crossPassMessageAttention(&g.p.loop, g.s, at(s.bank0, 0), at(s.bank1, 0), tensor(g.m.weights.cross_queries), tensor(g.m.weights.cross_kv_queries), at(s.messages, 0), @intCast(rows));
        // The prelude p is restored, but every visit's KV and convolution state remains live.
        try g.copyBf16(at(s.prelude, 0), at(s.p, 0), rows * state.hidden);
    }

    fn prepareAttentionInputs(g: Run, v: *const layer_plan.Visit, rows: usize) !void {
        const s = g.sc;
        const m = g.m;
        const lp = m.weight_plan.at(v.layer);
        const w = &m.weights.layers[v.layer];
        if (v.layer == 0) {
            try g.widen(at(s.p, 0), at(s.ar, 0), rows);
        } else {
            const q = m.weights.attn_queries[v.layer - 1];
            // One launch serves every role and every row: result (role, row) is `arRow(role, rows, row)`.
            try glue.arRows(&g.p.glue, g.s, at(s.blocks, 0), at(s.partial, 0), at(q.buf, 0), arRow(s, 0, rows, 0, false), lp.attn_sources, @intCast(lp.attn_queries), @intCast(rows), state.block_slots * state.hidden);
        }
        if (v.hasMessages()) {
            const controllers = w.controllers.?;
            const core = v.layer - 6;
            for (0..3) |role| {
                const c = controllers[role];
                const source_role = if (lp.kind == .local) 0 else role;
                try loop.gates(&g.p.loop, g.s, arRow(s, source_role, rows, 0, false), messageRow(s, core, role, 0), tensor(c.down), tensor(c.forget), tensor(c.gain), at(s.zeta, 0), arRow(s, role, rows, 0, true), @intCast(rows), state.message_entries * state.hidden);
            }
        }
        const gated = v.hasMessages();
        // An ungated local visit has exactly one AR result and one norm result; Q/K/V alias.
        if (!gated and lp.kind == .local) {
            try g.norm(arRow(s, 0, rows, 0, false), w.attn.q_norm, at(s.q_norm, 0), rows);
        } else {
            try g.norm(arRow(s, 0, rows, 0, gated), w.attn.q_norm, at(s.q_norm, 0), rows);
            try g.norm(arRow(s, 1, rows, 0, gated), w.attn.k_norm, at(s.k_norm, 0), rows);
            try g.norm(arRow(s, 2, rows, 0, gated), w.attn.v_norm, at(s.v_norm, 0), rows);
        }
    }

    fn staticPostConv(g: Run, cache: *state.Cache, slot: usize, conv_state: usize, raw: rt.Buffer, weight: Tensor, out: rt.Buffer, base: usize, rows: usize) !void {
        const channels = state.convWidth(conv_state);
        try glue.convStatic(&g.p.glue, g.s, cache.history(slot, conv_state).ptr(), at(raw, base * channels * 2), cache.staged(slot, conv_state, 0).ptr(), tensor(weight), at(out, base * channels * 2), @intCast(rows), @intCast(channels));
    }

    fn convInputForAttention(g: Run, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool) !void {
        const s = g.sc;
        const lp = g.m.weight_plan.at(v.layer);
        const w = &g.m.weights.layers[v.layer];
        const raw = at(if (key) s.k_conv else s.v_conv, base * state.kv_width * 2);
        const projected = at(if (key) s.k_raw else s.v_raw, base * state.kv_width * 2);
        const normalized = if (key) s.k_norm else if (v.hasMessages() or lp.kind == .global) s.v_norm else s.q_norm;
        const conv_state: usize = if (key) 0 else 1;
        const history = cache.history(v.cache_slot, conv_state).ptr();
        const taps = cache.staged(v.cache_slot, conv_state, 0).ptr();
        switch (if (key) w.attn.k_conv else w.attn.v_conv) {
            .static => |weight| try glue.convStatic(&g.p.glue, g.s, history, projected, taps, tensor(weight), raw, @intCast(rows), state.kv_width),
            .dynamic => |d| try glue.convDynamic(&g.p.glue, g.s, history, projected, taps, at(normalized, base * state.hidden * 2), tensor(d.static_w), tensor(d.projection), raw, @intCast(rows), state.kv_width),
        }
    }

    fn attendSegments(g: Run, v: *const layer_plan.Visit, segments: []const Segment) !void {
        const s = g.sc;
        const m = g.m;
        const w = &m.weights.layers[v.layer];
        var base: usize = 0;
        for (segments, 0..) |seg, si| {
            const rows = seg.rows;
            const cache = seg.cache;
            const layer = cache.attentionLayer(&m.weight_plan, v.cache_slot, rows);
            const start = cache.len;
            const rr = ap.Rows{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = @intCast(start + rows - 1), .streams = 1 };
            const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
            const meta = attention.RowMeta{
                .pos = aref(s.pos, base * 4),
                .seg_of = aref(s.seg_of, base * 4),
                .segs = aref(s.segs, si * @sizeOf(ap.Seg)),
            };
            const kv = attention.Cache{ .k = sref(cache.keys(v.cache_slot)), .v = sref(cache.values(v.cache_slot)) };
            try g.convInputForAttention(cache, v, base, rows, true);
            try g.convInputForAttention(cache, v, base, rows, false);
            const k = &g.p.attention;
            const q_at = base * state.query_width * 2;
            const tab_at = base * 24 * ap.extent * 2;
            try attention.prep(k, g.s, rr, layer, prep, meta, .{ aref(s.q_raw, q_at), aref(s.k_conv, base * state.kv_width * 2), aref(s.v_conv, base * state.kv_width * 2), aref(s.rel_raw, base * state.rel_width * 2) }, .{ aref(w.attn.q_scale.buf, 0), aref(w.attn.rel_scale.buf, 0) }, aref(s.q_prep, q_at), aref(s.rel_prep, tab_at), kv);
            try attention.relTable(k, g.s, rr, aref(s.rel_prep, tab_at), aref(w.attn.rel_projection.buf, 0), aref(s.rel_tab, tab_at));
            const part = base * 24 * s.stride * ap.head_dim * 4;
            const part_ml = base * 24 * s.stride * 8;
            try attention.attend(k, g.s, rr, layer, s.stride, meta, aref(s.q_prep, q_at), kv, aref(s.rel_tab, tab_at), aref(s.part_o, part), aref(s.part_ml, part_ml), aref(s.sink, base * 24 * 2), aref(s.gate, q_at), aref(s.att, q_at));
            base += rows;
        }
    }

    fn routedExperts(g: Run, layer: usize, rows: usize) !void {
        const s = g.sc;
        const w = &g.m.weights.layers[layer];
        const ex = &g.p.experts;
        const precision: experts.Format = switch (g.m.weight_plan.at(@intCast(layer)).precision) {
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
        };
        const gu_call = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision };
        const down_call = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision };
        const x = at(s.mlp_normed, 0);
        const ids = at(s.route_ids, 0);
        const act = at(s.gu, 0);
        const down = at(s.down, 0);
        if (rows <= decode_rows) {
            switch (precision) {
                .nvfp4 => try ex.nvfp4Gu(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), tensor(w.gu.amax.?), tensor(w.gu.denominator.?), ids, act, gu_call),
                .mxfp8 => try ex.mxfp8Gu(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), ids, act, gu_call),
            }
            switch (precision) {
                .nvfp4 => try ex.nvfp4Down(g.s, act, tensor(w.down.weight), tensor(w.down.scale), tensor(w.down.amax.?), tensor(w.down.denominator.?), ids, down, down_call),
                .mxfp8 => try ex.mxfp8Down(g.s, act, tensor(w.down.weight), tensor(w.down.scale), ids, down, down_call),
            }
        } else {
            const members = at(s.members, 0);
            const items = at(s.items, 0);
            const count = at(s.count, 0);
            try ex.plan(g.s, ids, members, items, count, gu_call);
            switch (precision) {
                .nvfp4 => try ex.nvfp4GuPrompt(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), tensor(w.gu.amax.?), tensor(w.gu.denominator.?), items, count, members, act, gu_call),
                .mxfp8 => try ex.mxfp8GuPrompt(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), items, count, members, act, gu_call),
            }
            switch (precision) {
                .nvfp4 => try ex.nvfp4DownPrompt(g.s, act, tensor(w.down.weight), tensor(w.down.scale), tensor(w.down.amax.?), tensor(w.down.denominator.?), items, count, members, down, down_call),
                .mxfp8 => try ex.mxfp8DownPrompt(g.s, act, tensor(w.down.weight), tensor(w.down.scale), items, count, members, down, down_call),
            }
        }
        try ex.combine(g.s, down, ids, at(s.route_weights, 0), at(s.mlp_out, 0), down_call);
    }

    fn visit(g: Run, v: *const layer_plan.Visit, segments: []const Segment, rows: usize, taps: ?Ptr) !void {
        const s = g.sc;
        const m = g.m;
        const lp = m.weight_plan.at(v.layer);
        const gated = v.hasMessages();
        if (taps) |to| {
            const channel: ?usize = switch (v.cache_slot) {
                5 => 0,
                23 => 1,
                41 => 2,
                else => null,
            };
            if (channel) |ch| try g.captureTap(at(s.p, 0), to, rows, ch);
        }
        if (v.pass > 1 and v.layer == 6) try g.bankAndReset(rows);
        const w = &m.weights.layers[v.layer];
        // The running BF16 p is the partial AR source before attention replaces it with t.
        try g.widen(at(s.p, 0), at(s.partial, 0), rows);
        try g.prepareAttentionInputs(v, rows);
        const knorm = if (gated or lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
        const vnorm = if (gated or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
        xpu.profile.label = "wq";
        try g.dense(w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
        xpu.profile.label = "wk";
        try g.dense(w.attn.wk, knorm, at(s.k_raw, 0), rows);
        xpu.profile.label = "wv";
        try g.dense(w.attn.wv, vnorm, at(s.v_raw, 0), rows);
        xpu.profile.label = "wr";
        try g.dense(w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
        xpu.profile.label = "sink";
        try g.dense(w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
        xpu.profile.label = "gate";
        try g.dense(w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
        xpu.profile.label = "";
        try g.attendSegments(v, segments);
        xpu.profile.label = "wo";
        try g.dense(w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
        xpu.profile.label = "";
        var base: usize = 0;
        for (segments) |seg| {
            try g.staticPostConv(seg.cache, v.cache_slot, 2, s.attn_dense, w.attn_conv, s.attn_conv, base, seg.rows);
            base += seg.rows;
        }
        try runtime.attentionPartial(&g.p.runtime, g.s, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0), rows, if (lp.boundary) lp.blocks_in + 1 else 0);
        try g.residual(m.weights.mlp_queries[v.layer], lp.mlp_sources, rows);
        if (gated) {
            const c = w.controllers.?[3];
            const core = v.layer - 6;
            try loop.gates(&g.p.loop, g.s, at(s.mlp_ar, 0), messageRow(s, core, 3, 0), tensor(c.down), tensor(c.forget), tensor(c.gain), at(s.zeta, 0), at(s.mlp_gated, 0), @intCast(rows), state.message_entries * state.hidden);
            try g.norm(at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        } else try g.norm(at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        try glue.router(&g.p.glue, g.s, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
        if (route_trace) |tb| try g.s.copy(plus(tb.ptr, v.cache_slot * rows * 32), at(s.route_ids, 0), rows * 32);
        try g.routedExperts(v.layer, rows);
        base = 0;
        for (segments) |seg| {
            try g.staticPostConv(seg.cache, v.cache_slot, 3, s.mlp_out, w.mlp_conv, s.mlp_conv, base, seg.rows);
            base += seg.rows;
        }
        try runtime.mlpPartial(&g.p.runtime, g.s, at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0), rows);
        if (v.cache_slot == 5) try g.copyBf16(at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
    }

    fn commitFullHistories(g: Run, segments: []const Segment) !void {
        for (segments) |seg| for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
            try g.copyBf16(seg.cache.staged(slot, cs, seg.rows).ptr(), seg.cache.history(slot, cs).ptr(), 3 * state.convWidth(cs));
        };
    }
};

pub fn run(m: *const Model, s: *state.Scratch, cache: *state.Cache, ids: []const u32, options: Options) !void {
    return runShared(m, s, &.{.{ .cache = cache, .rows = ids.len }}, ids, options);
}

/// One forward over `ids` (the rows of every segment in order); returns when the stream has finished it. With
/// `options.all` the logits of every row are in `s.logits_all` ([row][200058] f32), otherwise the last row's are in the
/// first segment's `cache.logits`.
pub fn runShared(m: *const Model, s: *state.Scratch, segments: []const Segment, ids: []const u32, options: Options) !void {
    const p: *const model_file.Pipelines = if (m.pipelines) |*x| x else return error.Inkling2NoKernels;
    const ctx = m.ctx orelse return error.Inkling2NoKernels;
    const rows = ids.len;
    if (rows == 0 or rows > s.rows or segments.len == 0 or segments.len > state.max_segments or (!options.all and segments.len != 1)) return error.Inkling2RowsExceeded;
    var pos: [state.max_rows]i32 = undefined;
    var seg_of: [state.max_rows]i32 = undefined;
    var segs: [state.max_segments]ap.Seg = undefined;
    var total: usize = 0;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        if (seg.cache.kv != .bf16) return error.Inkling2KvDtype;
        if (seg.cache.capacity > s.capacity) return error.Inkling2ScratchCapacity;
        if (seg.rows == 0 or seg.rows > state.max_rows or seg.cache.len + seg.rows > seg.cache.capacity) return error.Inkling2ContextFull;
        if (seg.rows > state.window_rows and seg.cache.pending != null) return error.Inkling2CacheTransaction;
        if (seg.cache.pending) |pending| if (pending != seg.rows) return error.Inkling2CacheTransaction;
        total += seg.rows;
        for (0..seg.rows) |i| {
            pos[base + i] = @intCast(seg.cache.len + i);
            seg_of[base + i] = 0;
        }
        segs[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        base += seg.rows;
    }
    if (total != rows) return error.Inkling2Segments;
    for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
    // The host arrays below stay valid until the stream has finished, which this function waits for before it returns.
    const stream = m.r.stream(ctx);
    errdefer stream.synchronize() catch {};
    try m.r.upload(s.ids, std.mem.sliceAsBytes(ids));
    try m.r.upload(s.pos, std.mem.sliceAsBytes(pos[0..rows]));
    try m.r.upload(s.seg_of, std.mem.sliceAsBytes(seg_of[0..rows]));
    try m.r.upload(s.segs, std.mem.sliceAsBytes(segs[0..segments.len]));
    const g = Run{ .m = m, .p = p, .s = stream, .sc = s };
    try glue.embedNorm(&p.glue, stream, at(s.ids, 0), tensor(m.weights.embed), tensor(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    for (m.weight_plan.visits[0..m.weight_plan.n_visits]) |*v| try g.visit(v, segments, rows, options.taps);
    try g.widen(at(s.p, 0), at(s.partial, 0), rows);
    const final = finalPlan(rows, options.all, options.taps != null);
    if (final.work_rows == rows) {
        try glue.arRows(&p.glue, stream, at(s.blocks, 0), at(s.partial, 0), tensor(m.weights.output_query), at(s.mlp_ar, 0), m.weight_plan.final_sources, 1, @intCast(rows), state.block_slots * state.hidden);
    } else {
        // the last row only
        try glue.ar(&p.glue, stream, at(s.blocks, (rows - 1) * state.block_slots * state.hidden * 2), row(s.partial, rows - 1, state.hidden * 4), tensor(m.weights.output_query), at(s.mlp_ar, 0), m.weight_plan.final_sources, 1);
    }
    try g.norm(at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), final.work_rows);
    if (options.taps) |to| try g.captureTap(at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) at(s.logits_all, 0) else at(segments[0].cache.logits, 0);
    const head_input = at(s.mlp_normed, final.logits_row * state.hidden * 2);
    if (m.weights.unembed_q8) |head| {
        try dense_q8_kernels.head(&p.dense_q8, stream, tensor(head), head_input, logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab);
    } else try glue.logits(&p.glue, stream, head_input, tensor(m.weights.unembed), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab);
    try g.commitFullHistories(segments);
    if (xpu.profile.active) |prof| prof.enqueued();
    try stream.synchronize();
}

test "the final plan unembeds one row unless all rows or taps are asked" {
    const t = std.testing;
    try t.expectEqual(FinalPlan{ .work_rows = 1, .logits_rows = 1, .logits_row = 0, .capture_rows = 0 }, finalPlan(5, false, false));
    try t.expectEqual(FinalPlan{ .work_rows = 5, .logits_rows = 5, .logits_row = 0, .capture_rows = 0 }, finalPlan(5, true, false));
    try t.expectEqual(FinalPlan{ .work_rows = 5, .logits_rows = 1, .logits_row = 4, .capture_rows = 5 }, finalPlan(5, false, true));
}

//! Inkling 2 Mini's prompt chunk on the XPU backend: one forward over up to `state.max_prefill_rows` rows of one stream.
//! The visit order, the rounding points and the buffers are forward.zig's; what differs is that every launch that forward
//! makes a row at a time (the attention residual, the message gates, the cross-pass copies) runs once for all rows, the
//! convolution windows stage in the scratch, and attention runs in slices of `state.max_rows` rows. A row's result does
//! not depend on the other rows of the chunk, so a prompt gives the same bits however it is cut into chunks.
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
const pk = @import("prefill_kernels.zig");
const pe = @import("prefill_experts.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const forward = @import("forward.zig");
const Model = model_file.Model;
const Tensor = model_file.Tensor;

const rt = xpu.rt;
const Ptr = forward.Ptr;

pub const Options = struct {
    /// Where the DSpark taps go ([row][tap][H] bf16), or none.
    taps: ?Ptr = null,
    /// Leave the head to `headRows`: the final-normed rows stay in `Scratch.mlp_normed` and no logits are made.
    defer_head: bool = false,
    /// Logits for every row in `Scratch.logits_all` (at most `state.max_rows` rows); otherwise the last row's go to the cache.
    all: bool = false,
    /// The routed experts of the decode path's prompt kernels (the control of an A/B); otherwise the grouped DPAS GEMMs.
    old_experts: bool = false,
    /// The dense projections of the decode path's kernels, 32 rows a launch (the control of an A/B).
    old_dense: bool = false,
    /// The message controllers of the decode path's kernels (the control of an A/B).
    old_loop: bool = false,
    /// The global layers' attention on the decode kernel in 64-row slices (the control of an A/B).
    old_global: bool = false,
    /// The sliding layers' attention on the decode kernel in 64-row slices over the cache ring (the control of an A/B).
    old_sliding: bool = false,
};

fn at(b: rt.Buffer, off: usize) Ptr {
    return @ptrFromInt(@intFromPtr(b.ptr.?) + off);
}
fn plus(p: Ptr, off: usize) Ptr {
    return @ptrFromInt(@intFromPtr(p.?) + off);
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


/// runtime.zig's pointwise launches for a chunk of more than `runtime.plan.max_rows` rows. The plan's `args` asserts at most 64
/// rows, and that assert is `unreachable` in an optimized build, which lets the compiler assume a chunk has at most 64 rows.
fn pointwise(p: *const runtime.Pipelines, s: xpu.Stream, kind: runtime.plan.Kind, rows: usize, boundary: u32, bufs: anytype) !void {
    const count: u32 = @intCast(rows * state.hidden);
    try xpu.launch.dispatch(p.of(kind), s, .{ (count + runtime.plan.threads - 1) / runtime.plan.threads, 1, 1 }, bufs ++ .{ count, boundary });
}

const Run = struct {
    m: *const Model,
    p: *const model_file.Pipelines,
    s: xpu.Stream,
    sc: *state.Scratch,
    rows: usize,
    old_experts: bool = false,
    old_dense: bool = false,
    old_loop: bool = false,
    /// Rows a global layer's attention takes a launch (at most `state.max_rows` selects the decode kernel), and a sliding
    /// layer's; the first segment slot of each kind of slice in `Scratch.segs`.
    old_sliding: bool = false,
    gl_slice: usize = state.max_rows,
    sl_slice: usize = state.max_rows,
    sl_seg0: usize = 0,

    fn copyBf16(g: Run, from: Ptr, to: Ptr, count: usize) !void {
        try runtime.copyBf16(&g.p.runtime, g.s, from, to, 1, count);
    }
    fn widen(g: Run, from: Ptr, to: Ptr, rows: usize) !void {
        try pointwise(&g.p.runtime, g.s, .widen, rows, 0, .{ from, to });
    }
    fn norm(g: Run, input: Ptr, weight: Tensor, out: Ptr, rows: usize) !void {
        try dense_kernels.norm(&g.p.dense, g.s, input, tensor(weight), out, @intCast(rows));
    }

    fn dense(g: Run, weight: Tensor, input: Ptr, out: Ptr, rows: usize) !void {
        const k: u32 = @intCast(weight.shape[weight.rank - 1]);
        const n: u32 = @intCast(weight.shape[weight.rank - 2]);
        if (weight.nvfp4) {
            const ids = g.m.dense_ids orelse return error.Inkling2NoKernels;
            var i: usize = 0;
            while (i < rows) : (i += 1) {
                const in: experts.Ptr = @ptrFromInt(@intFromPtr(input) + i * @as(usize, k) * 2);
                const ou: experts.Ptr = @ptrFromInt(@intFromPtr(out) + i * @as(usize, n) * 2);
                try g.p.experts.denseRow(g.s, weight.buf.ptr, n, k, in, ou, ids.ptr);
            }
            return;
        }
        if (weight.q8 and !g.old_dense and n % 8 == 0 and k % 256 == 0) return pk.denseQ8(&g.p.prefill, g.s, tensor(weight), input, at(g.sc.xp, 0), out, @intCast(rows), n, k);
        if (weight.q8) return dense_q8_kernels.dense(&g.p.dense_q8, g.s, tensor(weight), input, out, @intCast(rows), n, k);
        if (!g.old_dense and k % 8 == 0) return pk.denseBf16(&g.p.prefill, g.s, tensor(weight), input, out, @intCast(rows), n, k);
        try dense_kernels.dense(&g.p.dense, g.s, tensor(weight), input, out, @intCast(rows), n, k);
    }

    /// Out of the gate pair of `h` and the messages of one entry: out = (1 - f) h + g m, every row at once.
    fn gates(g: Run, h: Ptr, m: Ptr, c: model_file.Controller, out: Ptr) !void {
        const rows: u32 = @intCast(g.rows);
        if (!g.old_loop) return pk.controller(&g.p.prefill, g.s, h, m, tensor(c.down), tensor(c.forget), tensor(c.gain), at(g.sc.xp, 0), at(g.sc.zeta, 0), out, rows);
        const lp = &g.p.loop;
        try xpu.launch.dispatch(lp.zeta, g.s, .{ loop.rank / loop.zeta_ranks, rows, 1 }, .{ h, m, tensor(c.down), at(g.sc.zeta, 0), rows });
        try xpu.launch.dispatch(lp.gates, g.s, .{ loop.hidden / loop.gate_threads, rows, 1 }, .{ h, m, tensor(c.forget), tensor(c.gain), at(g.sc.zeta, 0), out, rows });
    }

    fn messageBlock(g: Run, core: usize, role: usize) Ptr {
        return at(g.sc.messages, (core * 4 + role) * g.rows * state.hidden * 4);
    }

    fn roleBlock(g: Run, buf: rt.Buffer, role: usize) Ptr {
        return at(buf, role * g.rows * state.hidden * 4);
    }

    fn bankAndReset(g: Run) !void {
        const s = g.sc;
        const rows = g.rows;
        // The second completed block of the previous pass is B12; its final p is P18.
        try pk.copyRows(&g.p.prefill, g.s, at(s.blocks, 2 * state.hidden * 2), at(s.bank0, 0), @intCast(rows), state.hidden, state.block_slots * state.hidden, state.hidden);
        try g.copyBf16(at(s.p, 0), at(s.bank1, 0), rows * state.hidden);
        try pk.messages(&g.p.prefill, g.s, at(s.bank0, 0), at(s.bank1, 0), tensor(g.m.weights.cross_queries), tensor(g.m.weights.cross_kv_queries), at(s.messages, 0), @intCast(rows));
        // The prelude p is restored, but every visit's KV and convolution state remains live.
        try g.copyBf16(at(s.prelude, 0), at(s.p, 0), rows * state.hidden);
    }

    fn prepareAttentionInputs(g: Run, v: *const layer_plan.Visit) !void {
        const s = g.sc;
        const m = g.m;
        const rows = g.rows;
        const lp = m.weight_plan.at(v.layer);
        const w = &m.weights.layers[v.layer];
        if (v.layer == 0) {
            try g.widen(at(s.p, 0), at(s.ar, 0), rows);
        } else {
            const q = m.weights.attn_queries[v.layer - 1];
            try pk.ar(&g.p.prefill, g.s, at(s.blocks, 0), at(s.partial, 0), at(q.buf, 0), at(s.ar, 0), @intCast(lp.attn_sources), @intCast(lp.attn_queries), @intCast(rows));
        }
        if (v.hasMessages()) {
            const controllers = w.controllers.?;
            const core = v.layer - 6;
            for (0..3) |role| {
                const source_role = if (lp.kind == .local) 0 else role;
                try g.gates(g.roleBlock(s.ar, source_role), g.messageBlock(core, role), controllers[role], g.roleBlock(s.gated, role));
            }
        }
        const gated = v.hasMessages();
        const roles = if (gated) s.gated else s.ar;
        // An ungated local visit has exactly one AR result and one norm result; Q/K/V alias.
        if (!gated and lp.kind == .local) {
            try g.norm(at(roles, 0), w.attn.q_norm, at(s.q_norm, 0), rows);
        } else {
            try g.norm(g.roleBlock(roles, 0), w.attn.q_norm, at(s.q_norm, 0), rows);
            try g.norm(g.roleBlock(roles, 1), w.attn.k_norm, at(s.k_norm, 0), rows);
            try g.norm(g.roleBlock(roles, 2), w.attn.v_norm, at(s.v_norm, 0), rows);
        }
    }

    /// Runs one of the visit's four convolutions over the whole chunk and commits its three-row history.
    fn commitHistory(g: Run, cache: *state.Cache, slot: usize, conv_state: usize) !void {
        const channels = state.convWidth(conv_state);
        try g.copyBf16(at(g.sc.staging, g.rows * channels * 2), cache.history(slot, conv_state).ptr(), 3 * channels);
    }

    fn staticPostConv(g: Run, cache: *state.Cache, slot: usize, conv_state: usize, raw: rt.Buffer, weight: Tensor, out: rt.Buffer) !void {
        const channels = state.convWidth(conv_state);
        try pk.convStatic(&g.p.prefill, g.s, cache.history(slot, conv_state).ptr(), at(raw, 0), at(g.sc.staging, 0), tensor(weight), at(out, 0), @intCast(g.rows), @intCast(channels));
        try g.commitHistory(cache, slot, conv_state);
    }

    fn convInputForAttention(g: Run, cache: *state.Cache, v: *const layer_plan.Visit, key: bool) !void {
        const s = g.sc;
        const lp = g.m.weight_plan.at(v.layer);
        const w = &g.m.weights.layers[v.layer];
        const raw = at(if (key) s.k_conv else s.v_conv, 0);
        const projected = at(if (key) s.k_raw else s.v_raw, 0);
        const normalized = if (key) s.k_norm else if (v.hasMessages() or lp.kind == .global) s.v_norm else s.q_norm;
        const conv_state: usize = if (key) 0 else 1;
        const history = cache.history(v.cache_slot, conv_state).ptr();
        const taps = at(s.staging, 0);
        const rows: u32 = @intCast(g.rows);
        switch (if (key) w.attn.k_conv else w.attn.v_conv) {
            .static => |weight| try pk.convStatic(&g.p.prefill, g.s, history, projected, taps, tensor(weight), raw, rows, state.kv_width),
            .dynamic => |d| try glue.convDynamic(&g.p.glue, g.s, history, projected, taps, at(normalized, 0), tensor(d.static_w), tensor(d.projection), raw, rows, state.kv_width),
        }
        try g.commitHistory(cache, v.cache_slot, conv_state);
    }

    /// Attention over the chunk. Every row's K and V enter the cache (global layers) or a staging ring (sliding layers, whose
    /// cache ring is too short to hold the window and a long chunk) in one launch; the rows then attend in slices that the
    /// partial buffers hold. A sliding layer's chunk reaches the cache ring afterwards.
    fn attend(g: Run, cache: *state.Cache, v: *const layer_plan.Visit) !void {
        const s = g.sc;
        const m = g.m;
        const w = &m.weights.layers[v.layer];
        try g.convInputForAttention(cache, v, true);
        try g.convInputForAttention(cache, v, false);
        const kv = attention.Cache{ .k = sref(cache.keys(v.cache_slot)), .v = sref(cache.values(v.cache_slot)) };
        const k = &g.p.attention;
        const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
        const layer = cache.attentionLayer(&m.weight_plan, v.cache_slot, state.max_rows);
        const sliding = layer.window != 0;
        if (sliding and g.old_sliding) return g.attendOld(cache, v);
        const start: u32 = @intCast(cache.len);
        const rows: u32 = @intCast(g.rows);
        const stage: u32 = @intCast(state.stageRows(g.rows));
        const all = ap.Rows{ .rows = rows, .heads = 24, .kv_heads = 8, .last_pos = start + rows - 1, .streams = 1 };
        const meta_all = attention.RowMeta{ .pos = aref(s.pos, 0), .seg_of = aref(s.seg_of, 0), .segs = aref(s.segs, 0) };
        var into = kv;
        var prep_layer = layer;
        if (sliding) {
            into = .{ .k = aref(s.kstage, 0), .v = aref(s.vstage, 0) };
            prep_layer.ring = stage;
            const history = @min(start, layer.window - 1);
            try pk.kvCopy(&g.p.prefill, g.s, kv.k.at(), into.k.at(), start - history, layer.ring, stage, history, 8);
            try pk.kvCopy(&g.p.prefill, g.s, kv.v.at(), into.v.at(), start - history, layer.ring, stage, history, 8);
        }
        try attention.prep(k, g.s, all, prep_layer, prep, meta_all, .{ aref(s.q_raw, 0), aref(s.k_conv, 0), aref(s.v_conv, 0), aref(s.rel_raw, 0) }, .{ aref(w.attn.q_scale.buf, 0), aref(w.attn.rel_scale.buf, 0) }, aref(s.q_prep, 0), aref(s.rel_prep, 0), into);
        try pk.relTable(&g.p.prefill, g.s, all.rows * all.heads, aref(s.rel_prep, 0).at(), aref(w.attn.rel_projection.buf, 0).at(), aref(s.rel_tab, 0).at());
        const step: usize = if (sliding) g.sl_slice else g.gl_slice;
        var base: usize = 0;
        var si: usize = if (sliding) g.sl_seg0 else 0;
        while (base < g.rows) : (si += 1) {
            const n = @min(step, g.rows - base);
            const rr = ap.Rows{ .rows = @intCast(n), .heads = 24, .kv_heads = 8, .last_pos = @intCast(start + base + n - 1), .streams = 1 };
            const meta = attention.RowMeta{ .pos = aref(s.pos, base * 4), .seg_of = aref(s.seg_of, base * 4), .segs = aref(s.segs, si * @sizeOf(ap.Seg)) };
            const q_at = base * state.query_width * 2;
            const tab_at = base * 24 * ap.extent * 2;
            const sink = aref(s.sink, base * 24 * 2);
            if (sliding) {
                const scale: f32 = 1.0 / @sqrt(@as(f32, ap.head_dim));
                try pk.attnWindow(&g.p.prefill, g.s, at(s.q_prep, q_at), into.k.at(), into.v.at(), at(s.rel_tab, tab_at), meta.segs.at(), at(s.part_o, 0), at(s.part_ml, 0), @intCast(n), 24, 8, stage, scale);
                // one partial a (row, head) and a key chunk no window spans: the merge folds it and adds the sink and gate
                try xpu.launch.dispatch(k.merge, g.s, .{ @as(u32, @intCast(n)) * 24, 1, 1 }, .{ aref(s.part_o, 0).at(), aref(s.part_ml, 0).at(), sink.at(), aref(s.gate, q_at).at(), meta.pos.at(), aref(s.att, q_at).at(), @as(u32, 24), layer.window, @as(u32, 1 << 30), @as(u32, 0), @as(u32, 1), @as(u32, 1) });
            } else if (g.gl_slice > state.max_rows) {
                var lg = layer;
                lg.chunk = ap.prompt_chunk;
                try attention.attendAs(k, g.s, .prompt, rr, lg, ap.chunks(rr.last_pos, ap.prompt_chunk), meta, aref(s.q_prep, q_at), kv, aref(s.rel_tab, tab_at), aref(s.part_o, 0), aref(s.part_ml, 0), sink, aref(s.gate, q_at), aref(s.att, q_at));
            } else {
                try attention.attend(k, g.s, rr, layer, s.stride, meta, aref(s.q_prep, q_at), kv, aref(s.rel_tab, tab_at), aref(s.part_o, 0), aref(s.part_ml, 0), sink, aref(s.gate, q_at), aref(s.att, q_at));
            }
            base += n;
        }
        if (sliding) {
            const n: u32 = @min(rows, layer.ring);
            try pk.kvCopy(&g.p.prefill, g.s, into.k.at(), kv.k.at(), start + rows - n, stage, layer.ring, n, 8);
            try pk.kvCopy(&g.p.prefill, g.s, into.v.at(), kv.v.at(), start + rows - n, stage, layer.ring, n, 8);
        }
    }

    fn attendOld(g: Run, cache: *state.Cache, v: *const layer_plan.Visit) !void {
        const s = g.sc;
        const m = g.m;
        const w = &m.weights.layers[v.layer];
        const kv = attention.Cache{ .k = sref(cache.keys(v.cache_slot)), .v = sref(cache.values(v.cache_slot)) };
        const k = &g.p.attention;
        const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
        var base: usize = 0;
        var si: usize = 0;
        while (base < g.rows) : (si += 1) {
            const rows = @min(state.max_rows, g.rows - base);
            const layer = cache.attentionLayer(&m.weight_plan, v.cache_slot, state.max_rows);
            const start = cache.len + base;
            const rr = ap.Rows{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = @intCast(start + rows - 1), .streams = 1 };
            const meta = attention.RowMeta{ .pos = aref(s.pos, base * 4), .seg_of = aref(s.seg_of, base * 4), .segs = aref(s.segs, si * @sizeOf(ap.Seg)) };
            const q_at = base * state.query_width * 2;
            const tab_at = base * 24 * ap.extent * 2;
            try attention.prep(k, g.s, rr, layer, prep, meta, .{ aref(s.q_raw, q_at), aref(s.k_conv, base * state.kv_width * 2), aref(s.v_conv, base * state.kv_width * 2), aref(s.rel_raw, base * state.rel_width * 2) }, .{ aref(w.attn.q_scale.buf, 0), aref(w.attn.rel_scale.buf, 0) }, aref(s.q_prep, q_at), aref(s.rel_prep, base * state.rel_width * 2), kv);
            try attention.relTable(k, g.s, rr, aref(s.rel_prep, base * state.rel_width * 2), aref(w.attn.rel_projection.buf, 0), aref(s.rel_tab, tab_at));
            try attention.attend(k, g.s, rr, layer, s.stride, meta, aref(s.q_prep, q_at), kv, aref(s.rel_tab, tab_at), aref(s.part_o, 0), aref(s.part_ml, 0), aref(s.sink, base * 24 * 2), aref(s.gate, q_at), aref(s.att, q_at));
            base += rows;
        }
    }

    fn routedExperts(g: Run, layer: usize) !void {
        if (g.old_experts) return g.routedExpertsOld(layer);
        const s = g.sc;
        const rows: u32 = @intCast(g.rows);
        const w = &g.m.weights.layers[layer];
        const k = &g.p.prefill_experts;
        const format: pe.Format = switch (g.m.weight_plan.at(@intCast(layer)).precision) {
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
        };
        const order = at(s.members, 0);
        const inv = at(s.inv, 0);
        const items = at(s.items, 0);
        const count = at(s.count, 0);
        const xs = at(s.xs, 0);
        const gu_w: pe.Weights = .{ .w = tensor(w.gu.weight), .scales = tensor(w.gu.scale), .amax = if (w.gu.amax) |a| tensor(a) else null, .den = if (w.gu.denominator) |d| tensor(d) else null };
        const down_w: pe.Weights = .{ .w = tensor(w.down.weight), .scales = tensor(w.down.scale), .amax = if (w.down.amax) |a| tensor(a) else null, .den = if (w.down.denominator) |d| tensor(d) else null };
        try pe.plan(k, g.s, at(s.route_ids, 0), order, inv, items, count, rows);
        try pe.gather(k, g.s, at(s.mlp_normed, 0), order, xs, rows, state.hidden);
        try pe.gu(k, g.s, format, xs, gu_w, items, count, at(s.gu, 0), rows, state.hidden, state.intermediate);
        try pe.down(k, g.s, format, at(s.gu, 0), down_w, items, count, at(s.down, 0), rows, state.intermediate, state.hidden);
        try pe.combine(k, g.s, at(s.down, 0), inv, at(s.route_ids, 0), at(s.route_weights, 0), at(s.mlp_out, 0), rows, state.hidden);
    }

    fn routedExpertsOld(g: Run, layer: usize) !void {
        const s = g.sc;
        const rows = g.rows;
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
        try ex.combine(g.s, down, ids, at(s.route_weights, 0), at(s.mlp_out, 0), down_call);
    }

    fn visit(g: Run, cache: *state.Cache, v: *const layer_plan.Visit, taps: ?Ptr) !void {
        const s = g.sc;
        const m = g.m;
        const rows = g.rows;
        const lp = m.weight_plan.at(v.layer);
        const gated = v.hasMessages();
        if (taps) |to| {
            const channel: ?usize = switch (v.cache_slot) {
                5 => 0,
                23 => 1,
                41 => 2,
                else => null,
            };
            if (channel) |ch| try pointwise(&g.p.runtime, g.s, .capture_tap, rows, @intCast(ch), .{ at(s.p, 0), to });
        }
        if (v.pass > 1 and v.layer == 6) try g.bankAndReset();
        const w = &m.weights.layers[v.layer];
        // The running BF16 p is the partial AR source before attention replaces it with t.
        try g.widen(at(s.p, 0), at(s.partial, 0), rows);
        try g.prepareAttentionInputs(v);
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
        try g.attend(cache, v);
        xpu.profile.label = "wo";
        try g.dense(w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
        xpu.profile.label = "";
        try g.staticPostConv(cache, v.cache_slot, 2, s.attn_dense, w.attn_conv, s.attn_conv);
        try pointwise(&g.p.runtime, g.s, .attention_partial, rows, if (lp.boundary) @intCast(lp.blocks_in + 1) else 0, .{ at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0) });
        try pk.ar(&g.p.prefill, g.s, at(s.blocks, 0), at(s.partial, 0), tensor(m.weights.mlp_queries[v.layer]), at(s.mlp_ar, 0), @intCast(lp.mlp_sources), 1, @intCast(rows));
        if (gated) {
            try g.gates(at(s.mlp_ar, 0), g.messageBlock(v.layer - 6, 3), w.controllers.?[3], at(s.mlp_gated, 0));
            try g.norm(at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        } else try g.norm(at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        if (g.old_experts) try glue.router(&g.p.glue, g.s, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows)) else try pk.router16(&g.p.prefill, g.s, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows), at(s.logits_all, 0));
        try g.routedExperts(v.layer);
        try g.staticPostConv(cache, v.cache_slot, 3, s.mlp_out, w.mlp_conv, s.mlp_conv);
        try pointwise(&g.p.runtime, g.s, .mlp_partial, rows, 0, .{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) });
        if (v.cache_slot == 5) try g.copyBf16(at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
    }
};

/// One forward over the prompt rows `ids` of `cache`; returns when the stream has finished it. The last row's logits go to
/// `cache.logits` (with `options.all`, every row's to `s.logits_all`, at most `state.max_rows` rows).
pub fn run(m: *const Model, s: *state.Scratch, cache: *state.Cache, ids: []const u32, options: Options) !void {
    const p: *const model_file.Pipelines = if (m.pipelines) |*x| x else return error.Inkling2NoKernels;
    const ctx = m.ctx orelse return error.Inkling2NoKernels;
    const rows = ids.len;
    if (rows == 0 or rows > s.rows or rows > state.max_prefill_rows) return error.Inkling2RowsExceeded;
    if (options.all and rows > state.max_rows) return error.Inkling2RowsExceeded;
    if (cache.kv != .bf16) return error.Inkling2KvDtype;
    if (cache.capacity > s.capacity) return error.Inkling2ScratchCapacity;
    if (cache.len + rows > cache.capacity) return error.Inkling2ContextFull;
    if (cache.pending != null) return error.Inkling2CacheTransaction;
    for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
    var pos_buf: [state.max_prefill_rows]i32 = undefined;
    var zero_buf: [state.max_prefill_rows]i32 = @splat(0);
    var segs: [state.prefill_segments]ap.Seg = undefined;
    // Attention slices. A global layer takes `gl_slice` rows a launch: the prompt kernel's partials (a slot per 1,024 keys)
    // must fit the buffer the decode kernel's 64 rows fill, and a slice of 64 rows or fewer takes the decode kernel. A
    // sliding layer leaves one partial a (row, head).
    const slots = (cache.len + rows - 1) / ap.prompt_chunk + 1;
    const limit = state.max_rows * @as(usize, s.stride) / slots;
    const gl_slice: usize = if (!options.old_global and limit > state.max_rows) @min(rows, limit) else state.max_rows;
    const sl_slice: usize = @min(rows, state.max_rows * @as(usize, s.stride));
    const gl_n = (rows + gl_slice - 1) / gl_slice;
    const sl_n = (rows + sl_slice - 1) / sl_slice;
    if (gl_n + sl_n > state.prefill_segments) return error.Inkling2RowsExceeded;
    for (0..rows) |i| pos_buf[i] = @intCast(cache.len + i);
    for (0..gl_n) |j| segs[j] = .{ .row0 = 0, .n = @intCast(@min(gl_slice, rows - j * gl_slice)), .p0 = @intCast(cache.len + j * gl_slice), .off = 0 };
    for (0..sl_n) |j| segs[gl_n + j] = .{ .row0 = 0, .n = @intCast(@min(sl_slice, rows - j * sl_slice)), .p0 = @intCast(cache.len + j * sl_slice), .off = 0 };
    const slices = gl_n + sl_n;
    // The host arrays stay valid until the stream has finished, which this function waits for before it returns.
    const stream = m.r.stream(ctx);
    errdefer stream.synchronize() catch {};
    try m.r.upload(s.ids, std.mem.sliceAsBytes(ids));
    try m.r.upload(s.pos, std.mem.sliceAsBytes(pos_buf[0..rows]));
    try m.r.upload(s.seg_of, std.mem.sliceAsBytes(zero_buf[0..rows]));
    try m.r.upload(s.segs, std.mem.sliceAsBytes(segs[0..slices]));
    const g = Run{ .m = m, .p = p, .s = stream, .sc = s, .rows = rows, .old_experts = options.old_experts, .old_dense = options.old_dense, .old_loop = options.old_loop, .old_sliding = options.old_sliding, .gl_slice = gl_slice, .sl_slice = sl_slice, .sl_seg0 = gl_n };
    try glue.embedNorm(&p.glue, stream, at(s.ids, 0), tensor(m.weights.embed), tensor(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    for (m.weight_plan.visits[0..m.weight_plan.n_visits]) |*v| try g.visit(cache, v, options.taps);
    try g.widen(at(s.p, 0), at(s.partial, 0), rows);
    const all = options.all or options.defer_head;
    const final = forward.finalPlan(rows, all, options.taps != null);
    if (final.work_rows == rows) {
        try pk.ar(&p.prefill, stream, at(s.blocks, 0), at(s.partial, 0), tensor(m.weights.output_query), at(s.mlp_ar, 0), @intCast(m.weight_plan.final_sources), 1, @intCast(rows));
    } else {
        const i = rows - 1;
        try pk.ar(&p.prefill, stream, at(s.blocks, i * state.block_slots * state.hidden * 2), at(s.partial, i * state.hidden * 4), tensor(m.weights.output_query), at(s.mlp_ar, 0), @intCast(m.weight_plan.final_sources), 1, 1);
    }
    try g.norm(at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), final.work_rows);
    // runtime.captureTap takes at most 64 rows; a prompt chunk has up to 2048, so launch the kernel over all of them.
    if (options.taps) |to| try pointwise(&p.runtime, stream, .capture_tap, final.capture_rows, 3, .{ at(s.mlp_normed, 0), to });
    if (!options.defer_head) {
        const logits = if (options.all) at(s.logits_all, 0) else at(cache.logits, 0);
        try head(m, p, stream, at(s.mlp_normed, final.logits_row * state.hidden * 2), logits, final.logits_rows);
    }
    if (xpu.profile.active) |prof| prof.enqueued();
    try stream.synchronize();
}

fn head(m: *const Model, p: *const model_file.Pipelines, stream: xpu.Stream, input: Ptr, logits: Ptr, rows: usize) !void {
    if (m.weights.unembed_q8) |h| {
        try dense_q8_kernels.head(&p.dense_q8, stream, tensor(h), input, logits, @intCast(rows), m.config.text.unpadded_vocab);
    } else try glue.logits(&p.glue, stream, input, tensor(m.weights.unembed), logits, @intCast(rows), m.config.text.unpadded_vocab);
}

/// The logits of rows `from .. from + n` of the last `run` made with `defer_head` (n at most `state.max_rows`), in
/// `s.logits_all`. Returns when the stream has finished.
pub fn headRows(m: *const Model, s: *state.Scratch, from: usize, n: usize) !void {
    const p: *const model_file.Pipelines = if (m.pipelines) |*x| x else return error.Inkling2NoKernels;
    const ctx = m.ctx orelse return error.Inkling2NoKernels;
    if (n == 0 or n > state.max_rows or from + n > s.rows) return error.Inkling2RowsExceeded;
    const stream = m.r.stream(ctx);
    try head(m, p, stream, at(s.mlp_normed, from * state.hidden * 2), at(s.logits_all, 0), n);
    try stream.synchronize();
}

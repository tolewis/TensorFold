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
const fuse = @import("fuse.zig");
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
    wrote: ?*[state.visit_slots]u8 = null,

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
    fn projN(g: Run, weight: Tensor) u32 {
        _ = g;
        return @intCast(weight.shape[weight.rank - 2]);
    }

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

    /// Widen, the attention residual, the three controller gates and the norms, in one launch.
    fn phaseA(g: Run, v: *const layer_plan.Visit, rows: usize) !void {
        const s = g.sc;
        const m = g.m;
        const lp = m.weight_plan.at(v.layer);
        const w = &m.weights.layers[v.layer];
        const gated = v.hasMessages();
        const nq: u32 = if (v.layer == 0) 0 else lp.attn_queries;
        var down: [3]fuse.Ref = .{ null, null, null };
        var forget: [3]fuse.Ref = .{ null, null, null };
        var gain: [3]fuse.Ref = .{ null, null, null };
        if (gated) {
            const controllers = w.controllers orelse return error.Inkling2NoKernels;
            for (0..3) |role| {
                down[role] = tensor(controllers[role].down);
                forget[role] = tensor(controllers[role].forget);
                gain[role] = tensor(controllers[role].gain);
            }
        }
        try fuse.phaseA(&g.p.fuse, g.s, .{
            .p = at(s.p, 0),
            .partial = at(s.partial, 0),
            .blocks = at(s.blocks, 0),
            .queries = if (v.layer == 0) null else at(m.weights.attn_queries[v.layer - 1].buf, 0),
            .ar = at(s.ar, 0),
            .messages = if (gated) at(s.messages, 0) else null,
            .down = down,
            .forget = forget,
            .gain = gain,
            .zeta = at(s.zeta, 0),
            .gated = at(s.gated, 0),
            .wq = tensor(w.attn.q_norm),
            .wk = tensor(w.attn.k_norm),
            .wv = tensor(w.attn.v_norm),
            .q_norm = at(s.q_norm, 0),
            .k_norm = at(s.k_norm, 0),
            .v_norm = at(s.v_norm, 0),
            .rows = @intCast(rows),
            .blocks_stride = state.block_slots * state.hidden,
            .sources = lp.attn_sources,
            .n_queries = nq,
            .roles = if (gated) 3 else 0,
            .local_src = if (lp.kind == .local) 1 else 0,
            .core = if (gated) v.layer - 6 else 0,
            .msg_entries = state.message_entries,
            .norm_n = if (!gated and lp.kind == .local) 1 else 3,
        });
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
            const both_static = switch (w.attn.k_conv) {
                .static => switch (w.attn.v_conv) {
                    .static => true,
                    .dynamic => false,
                },
                .dynamic => false,
            };
            if (fuse.kvConv() and both_static) {
                const channels: u32 = state.kv_width;
                const wk = switch (w.attn.k_conv) {
                    .static => |weight| weight,
                    .dynamic => unreachable,
                };
                const wv = switch (w.attn.v_conv) {
                    .static => |weight| weight,
                    .dynamic => unreachable,
                };
                const commit1: u32 = if (fuse.roleConv() and rows == 1) 1 else 0;
                if (commit1 == 1) if (g.wrote) |mask| {
                    mask[v.cache_slot] = mask[v.cache_slot] | 0x3;
                };
                try fuse.convKv(&g.p.fuse, g.s, cache.history(v.cache_slot, 0).ptr(), at(s.k_raw, base * channels * 2), cache.staged(v.cache_slot, 0, 0).ptr(), tensor(wk), at(s.k_conv, base * channels * 2), cache.history(v.cache_slot, 1).ptr(), at(s.v_raw, base * channels * 2), cache.staged(v.cache_slot, 1, 0).ptr(), tensor(wv), at(s.v_conv, base * channels * 2), @intCast(rows), channels, commit1);
            } else {
                try g.convInputForAttention(cache, v, base, rows, true);
                try g.convInputForAttention(cache, v, base, rows, false);
            }
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
        if (!fuse.downEpi()) try ex.combine(g.s, down, ids, at(s.route_weights, 0), at(s.mlp_out, 0), down_call);
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
        if (fuse.enabled()) try g.phaseA(v, rows) else {
            try g.widen(at(s.p, 0), at(s.partial, 0), rows);
            try g.prepareAttentionInputs(v, rows);
        }
        const knorm = if (gated or lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
        const vnorm = if (gated or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
        if (fuse.seg and rows == 1 and w.attn.wq.q8) {
            xpu.profile.label = "wqkv";
            const k: u32 = @intCast(w.attn.wq.shape[w.attn.wq.rank - 1]);
            if (fuse.sinkSeg()) {
                try dense_q8_kernels.seg1Sink(&g.p.dense_q8, g.s, .{
                    at(s.q_norm, 0), knorm, vnorm, at(s.q_norm, 0), at(s.q_norm, 0),
                }, .{
                    tensor(w.attn.wq), tensor(w.attn.wk), tensor(w.attn.wv), tensor(w.attn.wr), tensor(w.attn.gate),
                }, .{
                    at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), at(s.rel_raw, 0), at(s.gate, 0),
                }, .{
                    g.projN(w.attn.wq), g.projN(w.attn.wk), g.projN(w.attn.wv), g.projN(w.attn.wr), g.projN(w.attn.gate),
                }, k, tensor(w.attn.sink), at(s.sink, 0), 24, 1);
            } else try dense_q8_kernels.seg1(&g.p.dense_q8, g.s, .{
                at(s.q_norm, 0), knorm, vnorm, at(s.q_norm, 0), at(s.q_norm, 0),
            }, .{
                tensor(w.attn.wq), tensor(w.attn.wk), tensor(w.attn.wv), tensor(w.attn.wr), tensor(w.attn.gate),
            }, .{
                at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), at(s.rel_raw, 0), at(s.gate, 0),
            }, .{
                g.projN(w.attn.wq), g.projN(w.attn.wk), g.projN(w.attn.wv), g.projN(w.attn.wr), g.projN(w.attn.gate),
            }, k);
        } else {
            xpu.profile.label = "wq";
            try g.dense(w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
            xpu.profile.label = "wk";
            try g.dense(w.attn.wk, knorm, at(s.k_raw, 0), rows);
            xpu.profile.label = "wv";
            try g.dense(w.attn.wv, vnorm, at(s.v_raw, 0), rows);
            xpu.profile.label = "wr";
            try g.dense(w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
            xpu.profile.label = "gate";
            try g.dense(w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
        }
        // the sink is inside the segmented launch only for one row; every other row count needs the standalone launch
        if (!(fuse.sinkSeg() and rows == 1 and w.attn.wq.q8)) {
            xpu.profile.label = "sink";
            try g.dense(w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
        }
        xpu.profile.label = "";
        if (dump_dir_len > 0) {
            dumpRow(g.m, g.s, s.q_raw, state.query_width * 2, "qraw");
            dumpRow(g.m, g.s, s.k_raw, state.kv_width * 2, "kraw");
            dumpRow(g.m, g.s, s.v_raw, state.kv_width * 2, "vraw");
            dumpRow(g.m, g.s, s.rel_raw, state.rel_width * 2, "relraw");
            dumpRow(g.m, g.s, s.gate, state.query_width * 2, "gate");
            dumpRow(g.m, g.s, s.sink, 24 * 2, "sink");
        }
        if (dump_dir_len > 0) {
            // The window's own last row's K/V as the cache holds it right now, before attention reads it.
            const seg0 = segments[0];
            const ring = state.sliding_ring;
            const kref = seg0.cache.keys(v.cache_slot);
            const p0 = seg0.cache.len;
            dumpRowRef(g.m, g.s, kref, (p0 % ring) * ap.head_dim * 2, ap.head_dim * 2, "kvp0");
            dumpRowRef(g.m, g.s, kref, ((p0 + 1) % ring) * ap.head_dim * 2, ap.head_dim * 2, "kvp1");
            dumpRow(g.m, g.s, s.rel_tab, state.query_width * 2, "reltab");
            std.debug.print("ATT ARGS v{d}: rows {d} layer {d} extent {d}\n", .{ dump_visit, rows, v.layer, ap.extent });
        }
        if (dump_dir_len > 0) {
            dump_layer = v.layer;
            const seg0 = segments[0];
            const lay = seg0.cache.attentionLayer(&g.m.weight_plan, v.cache_slot, seg0.rows);
            const glp = seg0.cache.len + seg0.rows - 1;
            std.debug.print("GEOM v{d}: rows {d} last_pos {d} chunk {d} chunks {d} window {d} ring {d}\n", .{ dump_visit, seg0.rows, glp, lay.chunk, ap.chunks(glp, lay.chunk), lay.window, lay.ring });
        }
        try g.attendSegments(v, segments);
        if (dump_dir_len > 0) {
            dumpBufAt(g.m, g.s, s.rel_tab, 0, 24 * ap.extent * 2, "reltab0");
            dumpBufAt(g.m, g.s, s.rel_tab, 24 * ap.extent * 2, 24 * ap.extent * 2, "reltab1");
            const stride = s.stride;
            const slot = 24 * stride;
            dumpBufAt(g.m, g.s, s.part_o, 0, 8 * 128 * 4, "po0");
            dumpBufAt(g.m, g.s, s.part_o, slot * 128 * 4, 8 * 128 * 4, "po1");
            dumpBufAt(g.m, g.s, s.part_ml, 0, 8 * 8, "pm0");
            dumpBufAt(g.m, g.s, s.part_ml, slot * 8, 8 * 8, "pm1");
            std.debug.print("PART stride {d} slot {d}\n", .{ stride, slot });
            dumpBufAt(g.m, g.s, s.pos, 0, 8 * 4, "pos");
            dumpBufAt(g.m, g.s, s.seg_of, 0, 8 * 4, "segof");
            dumpBufAt(g.m, g.s, s.segs, 0, @sizeOf(ap.Seg) * 2, "segs");
            var pr: usize = 500;
            while (pr <= 507) : (pr += 1) {
                var tb: [16]u8 = undefined;
                const tt = std.fmt.bufPrint(&tb, "pr{d}", .{pr}) catch "pr";
                dumpRowRef(g.m, g.s, segments[0].cache.keys(v.cache_slot), (pr % state.sliding_ring) * ap.head_dim * 2, ap.head_dim * 2, tt);
            }
            var kr: usize = 0;
            while (kr < 8) : (kr += 1) {
                var tb2: [16]u8 = undefined;
                const t2 = std.fmt.bufPrint(&tb2, "kc{d}", .{kr}) catch "kc";
                dumpBufAt(g.m, g.s, s.k_conv, kr * state.kv_width * 2, state.kv_width * 2, t2);
            }
            dumpBufAt(g.m, g.s, s.k_conv, 0, state.kv_width * 2, "kc0");
            dumpBufAt(g.m, g.s, s.k_conv, state.kv_width * 2, state.kv_width * 2, "kc1");
            dumpRowRef(g.m, g.s, segments[0].cache.keys(v.cache_slot), (501 % state.sliding_ring) * ap.head_dim * 2, ap.head_dim * 2, "kp501");
            const segk = segments[0];
            const ring = state.sliding_ring;
            const kref2 = segk.cache.keys(v.cache_slot);
            const vref2 = segk.cache.values(v.cache_slot);
            var pk: usize = 500;
            while (pk <= 507) : (pk += 1) {
                var kb: [16]u8 = undefined;
                var vb: [16]u8 = undefined;
                const kt = std.fmt.bufPrint(&kb, "k{d}", .{pk}) catch "k";
                const vt = std.fmt.bufPrint(&vb, "v{d}", .{pk}) catch "v";
                dumpRowRef(g.m, g.s, kref2, (pk % ring) * ap.head_dim * 2, ap.head_dim * 2, kt);
                dumpRowRef(g.m, g.s, vref2, (pk % ring) * ap.head_dim * 2, ap.head_dim * 2, vt);
            }
        }
        xpu.profile.label = "wo";
        try g.dense(w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
        xpu.profile.label = "";
        var base: usize = 0;
        if (fuse.convFold() and segments.len == 1) {
            const seg = segments[0];
            const channels: u32 = @intCast(state.convWidth(2));
            const commit1: u32 = if (fuse.roleConv() and rows == 1) 1 else 0;
            if (commit1 == 1) if (g.wrote) |mask| {
                mask[v.cache_slot] = mask[v.cache_slot] | 0x4;
            };
            try fuse.convAttnPartial(&g.p.fuse, g.s, seg.cache.history(v.cache_slot, 2).ptr(), at(s.attn_dense, 0), seg.cache.staged(v.cache_slot, 2, 0).ptr(), tensor(w.attn_conv), at(s.attn_conv, 0), at(s.p, 0), at(s.blocks, 0), at(s.partial, 0), @intCast(rows), channels, if (lp.boundary) lp.blocks_in + 1 else 0, commit1);
        } else {
            for (segments) |seg| {
                try g.staticPostConv(seg.cache, v.cache_slot, 2, s.attn_dense, w.attn_conv, s.attn_conv, base, seg.rows);
                base += seg.rows;
            }
            try runtime.attentionPartial(&g.p.runtime, g.s, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0), rows, if (lp.boundary) lp.blocks_in + 1 else 0);
        }
        try g.residual(m.weights.mlp_queries[v.layer], lp.mlp_sources, rows);
        if (gated) {
            const c = w.controllers.?[3];
            const core = v.layer - 6;
            if (fuse.zetaWide()) {
                const stride: u32 = state.message_entries * state.hidden;
                try xpu.launch.dispatch(g.p.fuse.zeta_wide, g.s, .{ 4, 1, @as(u32, @intCast(rows)) }, .{
                    at(s.mlp_ar, 0), messageRow(s, core, 3, 0), tensor(c.down), tensor(c.down), tensor(c.down), at(s.zeta, 0),
                    @as(u32, @intCast(rows)), @as(u32, 1), @as(u32, 1), @as(u32, 0), stride / state.hidden, @as(u32, 0),
                });
                try loop.gatesOnly(&g.p.loop, g.s, at(s.mlp_ar, 0), messageRow(s, core, 3, 0), tensor(c.forget), tensor(c.gain), at(s.zeta, 0), at(s.mlp_gated, 0), @intCast(rows), stride);
            } else try loop.gates(&g.p.loop, g.s, at(s.mlp_ar, 0), messageRow(s, core, 3, 0), tensor(c.down), tensor(c.forget), tensor(c.gain), at(s.zeta, 0), at(s.mlp_gated, 0), @intCast(rows), state.message_entries * state.hidden);
            if (!fuse.normRouter()) try g.norm(at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        } else if (!fuse.normRouter()) try g.norm(at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        if (fuse.normRouter() and fuse.routerWide()) {
            const src = if (gated) at(s.mlp_gated, 0) else at(s.mlp_ar, 0);
            try fuse.routerNormLaunch(&g.p.fuse, g.s, src, tensor(w.mlp_norm), tensor(w.router_weight), at(s.mlp_normed, 0), tensor(w.router_bias), at(s.logits_all, 0), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
        } else if (fuse.routerWide()) {
            try fuse.routerWideLaunch(&g.p.fuse, g.s, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.logits_all, 0), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
        } else try glue.router(&g.p.glue, g.s, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
        if (dump_dir_len > 0) {
            dumpRow(g.m, g.s, s.route_ids, state.experts_per_token * 4, "rids");
            dumpRow(g.m, g.s, s.route_weights, state.experts_per_token * 4, "rw");
        }
        if (route_trace) |tb| try g.s.copy(plus(tb.ptr, v.cache_slot * rows * 32), at(s.route_ids, 0), rows * 32);
        try g.routedExperts(v.layer, rows);
        base = 0;
        if (fuse.downEpi() and segments.len == 1) {
            const seg = segments[0];
            const channels: u32 = @intCast(state.convWidth(3));
            const commit1: u32 = if (fuse.roleConv() and rows == 1) 1 else 0;
            if (commit1 == 1) if (g.wrote) |mask| {
                mask[v.cache_slot] = mask[v.cache_slot] | 0x8;
            };
            try fuse.downEpilogue(&g.p.fuse, g.s, at(s.down, 0), at(s.route_ids, 0), at(s.route_weights, 0), seg.cache.history(v.cache_slot, 3).ptr(), seg.cache.staged(v.cache_slot, 3, 0).ptr(), tensor(w.mlp_conv), at(s.mlp_out, 0), at(s.mlp_conv, 0), at(s.partial, 0), at(s.p, 0), state.hidden, 8, channels, @intCast(rows), commit1);
        } else if (fuse.convFold() and segments.len == 1) {
            const seg = segments[0];
            const channels: u32 = @intCast(state.convWidth(3));
            const commit1: u32 = if (fuse.roleConv() and rows == 1) 1 else 0;
            if (commit1 == 1) if (g.wrote) |mask| {
                mask[v.cache_slot] = mask[v.cache_slot] | 0x8;
            };
            try fuse.convMlpPartial(&g.p.fuse, g.s, seg.cache.history(v.cache_slot, 3).ptr(), at(s.mlp_out, 0), seg.cache.staged(v.cache_slot, 3, 0).ptr(), tensor(w.mlp_conv), at(s.mlp_conv, 0), at(s.partial, 0), at(s.p, 0), @intCast(rows), channels, commit1);
        } else {
            for (segments) |seg| {
                try g.staticPostConv(seg.cache, v.cache_slot, 3, s.mlp_out, w.mlp_conv, s.mlp_conv, base, seg.rows);
                base += seg.rows;
            }
            try runtime.mlpPartial(&g.p.runtime, g.s, at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0), rows);
        }
        if (v.cache_slot == 5) try g.copyBf16(at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
        if (dump_dir_len > 0) {
            dump2(g.m, g.s, "p", s.p, state.hidden * 2, state.hidden * 2);
            dump2(g.m, g.s, "att", s.att, state.query_width * 2, state.query_width * 2);
            dump2(g.m, g.s, "pres", s.attn_dense, state.hidden * 2, state.hidden * 2);
            dump2(g.m, g.s, "mlpnorm", s.mlp_normed, state.hidden * 2, state.hidden * 2);
            dump2(g.m, g.s, "mlpout", s.mlp_out, state.hidden * 2, state.hidden * 2);
            dump_visit += 1;
        }
        // A window stages every layer's conv rows into one shared buffer, so committing once at the end would copy the last
        // layer's rows into every slot's history. Each visit commits its own slot's last three rows instead.
        if (rows > 1) {
            for (segments) |seg| for (0..state.conv_states) |cs| {
                try g.copyBf16(seg.cache.staged(v.cache_slot, cs, seg.rows).ptr(), seg.cache.history(v.cache_slot, cs).ptr(), 3 * state.convWidth(cs));
            };
        }
    }

    fn commitFullHistories(g: Run, segments: []const Segment) !void {
        for (segments) |seg| for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
            // A window already committed each slot at its own visit.
            if (seg.rows > 1) continue;
            if (g.wrote) |mask| {
                if ((mask[slot] & (@as(u8, 1) << @intCast(cs))) != 0) continue;
            }
            try g.copyBf16(seg.cache.staged(slot, cs, seg.rows).ptr(), seg.cache.history(slot, cs).ptr(), 3 * state.convWidth(cs));
        };
    }
};


// Per-visit row dumps for the bit check (TF_STAGE_DUMP=dir). One file a visit and tensor, row `TF_STAGE_ROW`.
var dump_dir_buf: [256]u8 = undefined;
var dump_dir_len: usize = 0;
var dump_row: usize = 1;
var dump_visit: usize = 0;
var dump_layer: u32 = 0;

pub fn dumpEnable(dir: []const u8, at_row: usize) void {
    dump_dir_len = @min(dir.len, dump_dir_buf.len);
    @memcpy(dump_dir_buf[0..dump_dir_len], dir[0..dump_dir_len]);
    dump_row = at_row;
    dump_visit = 0;
}

fn dump2(m: *const Model, s: xpu.Stream, tag: []const u8, buf: rt.Buffer, stride: usize, len: usize) void {
    var a: [64]u8 = undefined;
    var b: [64]u8 = undefined;
    const t0 = std.fmt.bufPrint(&a, "{s}0", .{tag}) catch return;
    const t1 = std.fmt.bufPrint(&b, "{s}1", .{tag}) catch return;
    dumpBufAt(m, s, buf, 0, len, t0);
    dumpBufAt(m, s, buf, stride, len, t1);
}

fn dumpBufAt(m: *const Model, s: xpu.Stream, buf: rt.Buffer, off: usize, len: usize, tag: []const u8) void {
    if (dump_dir_len == 0 or dump_visit != 0 or dump_layer != 0) return;
    var stack: [65536]u8 = undefined;
    const n = @min(len, stack.len);
    const b = rt.Buffer{ .rt = m.r, .ptr = @ptrFromInt(@intFromPtr(buf.ptr.?) + off), .len = n };
    s.synchronize() catch return;
    m.r.download(stack[0..n], b) catch return;
    var path_buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/v{d}-{s}.bin", .{ dump_dir_buf[0..dump_dir_len], dump_visit, tag }) catch return;
    path_buf[path.len] = 0;
    const fd = std.c.open(path_buf[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    _ = std.c.write(fd, stack[0..n].ptr, n);
    _ = std.c.close(fd);
}

fn dumpRowRef(m: *const Model, s: xpu.Stream, ref: state.Ref, off: usize, n: usize, tag: []const u8) void {
    if (dump_dir_len == 0 or dump_visit != 0 or dump_layer != 0) return;
    var stack: [state.hidden * 2]u8 = undefined;
    const len = @min(n, stack.len);
    const b = rt.Buffer{ .rt = m.r, .ptr = @ptrFromInt(@intFromPtr(ref.buf.ptr.?) + ref.off + off), .len = len };
    s.synchronize() catch return;
    m.r.download(stack[0..len], b) catch return;
    var path_buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/v{d}-{s}.bin", .{ dump_dir_buf[0..dump_dir_len], dump_visit, tag }) catch return;
    path_buf[path.len] = 0;
    const fd = std.c.open(path_buf[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    _ = std.c.write(fd, stack[0..len].ptr, len);
    _ = std.c.close(fd);
}

fn dumpRow(m: *const Model, s: xpu.Stream, buf: rt.Buffer, row_bytes: usize, tag: []const u8) void {
    if (dump_dir_len == 0) return;
    var stack: [state.hidden * 2]u8 = undefined;
    const n = @min(row_bytes, stack.len);
    const b = rt.Buffer{ .rt = m.r, .ptr = @ptrFromInt(@intFromPtr(buf.ptr.?) + dump_row * row_bytes), .len = n };
    s.synchronize() catch return;
    m.r.download(stack[0..n], b) catch return;
    var path_buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/v{d}-{s}.bin", .{ dump_dir_buf[0..dump_dir_len], dump_visit, tag }) catch return;
    path_buf[path.len] = 0;
    const fd = std.c.open(path_buf[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    _ = std.c.write(fd, stack[0..n].ptr, n);
    _ = std.c.close(fd);
}

fn bf16bits(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

/// One row, three visit kinds, fused phase A against the unfused launches. Prints the first mismatch. Returns the mismatch count.
pub fn checkPhaseA(m: *const Model, s: *state.Scratch, r: *xpu.rt.Runtime) !usize {
    if (m.pipelines == null or m.ctx == null) return error.Inkling2NoKernels;
    const pipes = &m.pipelines.?;
    const stream = m.r.stream(m.ctx.?);
    const H = state.hidden;
    var p_bits: [H]u16 = undefined;
    var blk: [state.block_slots * H]u16 = undefined;
    var msg: [state.message_entries * H]f32 = undefined;
    for (0..H) |d| {
        p_bits[d] = bf16bits(0.01 * @as(f32, @floatFromInt((d * 3) % 97)) - 0.2);
        msg[d] = 0.02 * @as(f32, @floatFromInt((d * 5) % 53)) - 0.4;
    }
    for (0..blk.len) |d| blk[d] = bf16bits(0.005 * @as(f32, @floatFromInt(d % 113)) - 0.1);
    for (H..msg.len) |d| msg[d] = msg[d % H] * 0.5;
    try r.upload(s.p, std.mem.sliceAsBytes(&p_bits));
    try r.upload(s.blocks, std.mem.sliceAsBytes(&blk));
    try r.upload(s.messages, std.mem.sliceAsBytes(&msg));
    const g = Run{ .m = m, .p = pipes, .s = stream, .sc = s };
    var bad: usize = 0;
    var part_a: [H]f32 = undefined;
    var part_b: [H]f32 = undefined;
    var ar_a: [3 * H]f32 = undefined;
    var ar_b: [3 * H]f32 = undefined;
    var gate_a: [3 * H]f32 = undefined;
    var gate_b: [3 * H]f32 = undefined;
    var nq_a: [H]u16 = undefined;
    var nq_b: [H]u16 = undefined;
    var nk_a: [H]u16 = undefined;
    var nk_b: [H]u16 = undefined;
    var nv_a: [H]u16 = undefined;
    var nv_b: [H]u16 = undefined;
    for (m.weight_plan.visits[0..m.weight_plan.n_visits]) |*v| {
        const kind = m.weight_plan.at(v.layer).kind;
        const want = (v.layer == 0) or (v.layer == 5 and !v.hasMessages()) or (v.hasMessages() and v.layer == 6);
        if (!want) continue;
        fuse.force = false;
        try g.widen(at(s.p, 0), at(s.partial, 0), 1);
        try g.prepareAttentionInputs(v, 1);
        try stream.synchronize();
        try r.download(std.mem.sliceAsBytes(&part_a), s.partial);
        try r.download(std.mem.sliceAsBytes(&ar_a), s.ar);
        try r.download(std.mem.sliceAsBytes(&gate_a), s.gated);
        try r.download(std.mem.sliceAsBytes(&nq_a), s.q_norm);
        try r.download(std.mem.sliceAsBytes(&nk_a), s.k_norm);
        try r.download(std.mem.sliceAsBytes(&nv_a), s.v_norm);
        try r.sync();
        fuse.force = true;
        try g.phaseA(v, 1);
        try stream.synchronize();
        try r.download(std.mem.sliceAsBytes(&part_b), s.partial);
        try r.download(std.mem.sliceAsBytes(&ar_b), s.ar);
        try r.download(std.mem.sliceAsBytes(&gate_b), s.gated);
        try r.download(std.mem.sliceAsBytes(&nq_b), s.q_norm);
        try r.download(std.mem.sliceAsBytes(&nk_b), s.k_norm);
        try r.download(std.mem.sliceAsBytes(&nv_b), s.v_norm);
        try r.sync();
        var row_bad: usize = 0;
        for (part_a, part_b) |x, y| if (x != y) {
            row_bad += 1;
        };
        for (ar_a, ar_b) |x, y| if (x != y) {
            row_bad += 1;
        };
        for (gate_a, gate_b) |x, y| if (x != y) {
            row_bad += 1;
        };
        for (nq_a, nq_b) |x, y| if (x != y) {
            row_bad += 1;
        };
        if (kind == .global or v.hasMessages()) {
            for (nk_a, nk_b) |x, y| if (x != y) {
                row_bad += 1;
            };
            for (nv_a, nv_b) |x, y| if (x != y) {
                row_bad += 1;
            };
        }
        std.debug.print("fuse-a layer {d} pass {d} gated {} mismatches {d}\n", .{ v.layer, v.pass, v.hasMessages(), row_bad });
        bad += row_bad;
    }
    fuse.force = null;
    return bad;
}

/// One row, layer 0 router weights. The wide path's picks against the old kernel.
pub fn checkRouter(m: *const Model, s: *state.Scratch, r: *xpu.rt.Runtime) !usize {
    if (m.pipelines == null or m.ctx == null) return error.Inkling2NoKernels;
    const pipes = &m.pipelines.?;
    const stream = m.r.stream(m.ctx.?);
    const H = state.hidden;
    var xbits: [H]u16 = undefined;
    for (0..H) |d| xbits[d] = bf16bits(0.02 * @as(f32, @floatFromInt((d * 13) % 80)) - 0.4);
    try r.upload(s.mlp_normed, std.mem.sliceAsBytes(&xbits));
    const w = &m.weights.layers[0];
    const g = Run{ .m = m, .p = pipes, .s = stream, .sc = s };
    _ = g;
    try glue.router(&pipes.glue, stream, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), 1);
    try stream.synchronize();
    var ids_a: [8]u32 = undefined;
    var w_a: [8]f32 = undefined;
    var ids_b: [8]u32 = undefined;
    var w_b: [8]f32 = undefined;
    try r.download(std.mem.sliceAsBytes(&ids_a), s.route_ids);
    try r.download(std.mem.sliceAsBytes(&w_a), s.route_weights);
    try r.sync();
    try fuse.routerWideLaunch(&pipes.fuse, stream, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.logits_all, 0), at(s.route_ids, 0), at(s.route_weights, 0), 1);
    try stream.synchronize();
    try r.download(std.mem.sliceAsBytes(&ids_b), s.route_ids);
    try r.download(std.mem.sliceAsBytes(&w_b), s.route_weights);
    try r.sync();
    var bad: usize = 0;
    for (ids_a, ids_b) |a, b| if (a != b) {
        bad += 1;
    };
    for (w_a, w_b) |a, b| if (a != b) {
        bad += 1;
    };
    std.debug.print("router-check ids {any} vs {any} weight mismatches in the {d} bad slots\n", .{ ids_a, ids_b, bad });
    return bad;
}

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
    if (rows == 0 or rows > s.rows or rows > state.max_rows or segments.len == 0 or segments.len > state.max_segments or (!options.all and segments.len != 1)) return error.Inkling2RowsExceeded;
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
    var wrote: [state.visit_slots]u8 = @splat(0);
    const g = Run{ .m = m, .p = p, .s = stream, .sc = s, .wrote = if (fuse.roleConv() and rows == 1) &wrote else null };
    try glue.embedNorm(&p.glue, stream, at(s.ids, 0), tensor(m.weights.embed), tensor(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    dumpRow(m, stream, s.p, state.hidden * 2, "p0");
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

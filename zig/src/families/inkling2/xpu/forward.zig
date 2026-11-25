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
const place_mod = @import("place.zig");
const group = @import("group.zig");

var dump_fwd: u32 = 0;

fn dumpWanted() bool {
    const z = std.c.getenv("TF_TP_DUMP_FWD") orelse return dump_fwd == 0;
    const want = std.fmt.parseInt(u32, std.mem.span(z), 10) catch return dump_fwd == 0;
    return dump_fwd == want;
}
var slice_own_dumped = false;
var slice_pulled_dumped = false;

fn dumpF32(r: *xpu.rt.Runtime, rank: u32, name: []const u8, src: ?*anyopaque, n: u32) !void {
    const dir_z = std.c.getenv("TF_TP_DUMP") orelse return;
    const ptr = src orelse return;
    const bytes: usize = @as(usize, n) * 4;
    const host = try std.heap.page_allocator.alloc(u8, bytes);
    defer std.heap.page_allocator.free(host);
    try r.download(host, .{ .rt = r, .ptr = ptr, .len = bytes });
    try r.sync();
    var path_buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/{s}-r{d}.bin", .{ std.mem.span(dir_z), name, rank }) catch return;
    path_buf[path.len] = 0;
    const fd = open(path_buf[0..path.len :0], 1 | 64 | 512, 0o644);
    if (fd < 0) return;
    _ = write(fd, host.ptr, host.len);
    _ = close(fd);
}

fn dumpNamed(g: *const Run, row_tag: []const u8, what: []const u8, src: Ptr, bytes: usize) !void {
    const dir_z = std.c.getenv("TF_TP_DUMP") orelse return;
    const ptr = src orelse return;
    try g.m.r.sync();
    const host = try g.m.gpa.alloc(u8, bytes);
    defer g.m.gpa.free(host);
    try g.m.r.download(host, .{ .rt = g.m.r, .ptr = ptr, .len = bytes });
    try g.m.r.sync();
    var path_buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/{s}-{s}-r{d}.bin", .{ std.mem.span(dir_z), row_tag, what, g.m.options.place.rank }) catch return;
    path_buf[path.len] = 0;
    const fd = open(path_buf[0..path.len :0], 1 | 64 | 512, 0o644);
    if (fd < 0) return;
    _ = write(fd, host.ptr, host.len);
    _ = close(fd);
}

fn dumpBytes(g: *const Run, visit_i: usize, tag: []const u8, src: ?*anyopaque, bytes: usize) !void {
    if (!dumpWanted()) return;
    const dir_z = std.c.getenv("TF_TP_DUMP") orelse return;
    const ptr = src orelse return;
    const host = try g.m.gpa.alloc(u8, bytes);
    defer g.m.gpa.free(host);
    try g.m.r.download(host, .{ .rt = g.m.r, .ptr = ptr, .len = bytes });
    try g.m.r.sync();
    var path_buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/v{d}-{s}-r{d}.bin", .{ std.mem.span(dir_z), visit_i, tag, g.m.options.place.rank }) catch return;
    path_buf[path.len] = 0;
    const fd = open(path_buf[0..path.len :0], 1 | 64 | 512, 0o644);
    if (fd < 0) return;
    _ = write(fd, host.ptr, host.len);
    _ = close(fd);
}

fn copyF32(r: *xpu.rt.Runtime, dst: Ptr, src: Ptr, n: u32) !void {
    const bytes: usize = @as(usize, n) * 4;
    try r.copy(.{ .rt = r, .ptr = dst, .len = bytes }, .{ .rt = r, .ptr = src, .len = bytes }, bytes);
}

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;

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

    /// One row because M2's AR ABI is [sources,H] -> [queries,H]. Batch composition never mixes rows.
    fn residual(g: Run, queries: Tensor, sources: usize, rows: usize) !void {
        const s = g.sc;
        for (0..rows) |i| try glue.ar(&g.p.glue, g.s, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), tensor(queries), row(s.mlp_ar, i, state.hidden * 4), @intCast(sources), 1);
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
            // One row: the queries and their results are contiguous, so one launch serves every role.
            if (rows == 1) {
                try glue.ar(&g.p.glue, g.s, at(s.blocks, 0), row(s.partial, 0, state.hidden * 4), at(q.buf, 0), arRow(s, 0, 1, 0, false), lp.attn_sources, @intCast(lp.attn_queries));
            } else for (0..lp.attn_queries) |role| {
                const query = at(q.buf, role * state.hidden * 4);
                for (0..rows) |i| try glue.ar(&g.p.glue, g.s, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), query, arRow(s, role, rows, i, false), lp.attn_sources, 1);
            }
        }
        if (v.hasMessages()) {
            const controllers = w.controllers.?;
            const core = v.layer - 6;
            for (0..3) |role| for (0..rows) |i| {
                const c = controllers[role];
                const source_role = if (lp.kind == .local) 0 else role;
                try loop.gates(&g.p.loop, g.s, arRow(s, source_role, rows, i, false), messageRow(s, core, role, i), tensor(c.down), tensor(c.forget), tensor(c.gain), row(s.zeta, i, 64 * 4), arRow(s, role, rows, i, true), 1);
            };
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
        const kv_stride = g.m.options.place.kvWidth() * 2;
        const raw = at(if (key) s.k_conv else s.v_conv, base * kv_stride);
        const projected = at(if (key) s.k_raw else s.v_raw, base * kv_stride);
        const normalized = if (key) s.k_norm else if (v.hasMessages() or lp.kind == .global) s.v_norm else s.q_norm;
        const conv_state: usize = if (key) 0 else 1;
        const history = cache.history(v.cache_slot, conv_state).ptr();
        const taps = cache.staged(v.cache_slot, conv_state, 0).ptr();
        switch (if (key) w.attn.k_conv else w.attn.v_conv) {
            .static => |weight| try glue.convStatic(&g.p.glue, g.s, history, projected, taps, tensor(weight), raw, @intCast(rows), g.m.options.place.kvWidth()),
            .dynamic => |d| try glue.convDynamic(&g.p.glue, g.s, history, projected, taps, at(normalized, base * state.hidden * 2), tensor(d.static_w), tensor(d.projection), raw, @intCast(rows), g.m.options.place.kvWidth()),
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
            const pl = g.m.options.place;
            const rr = ap.Rows{ .rows = @intCast(rows), .heads = pl.qHeads(), .kv_heads = pl.kvHeads(), .last_pos = @intCast(start + rows - 1), .streams = 1 };
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
            const q_at = base * pl.qWidth() * 2;
            const tab_at = base * pl.qHeads() * ap.extent * 2;
            try attention.prep(k, g.s, rr, layer, prep, meta, .{ aref(s.q_raw, q_at), aref(s.k_conv, base * state.kv_width * 2), aref(s.v_conv, base * state.kv_width * 2), aref(s.rel_raw, base * pl.relWidth() * 2) }, .{ aref(w.attn.q_scale.buf, 0), aref(w.attn.rel_scale.buf, 0) }, aref(s.q_prep, q_at), aref(s.rel_prep, tab_at), kv);
            try attention.relTable(k, g.s, rr, aref(s.rel_prep, tab_at), aref(w.attn.rel_projection.buf, 0), aref(s.rel_tab, tab_at));
            const part = base * pl.qHeads() * s.stride * ap.head_dim * 4;
            const part_ml = base * pl.qHeads() * s.stride * 8;
            try attention.attend(k, g.s, rr, layer, s.stride, meta, aref(s.q_prep, q_at), kv, aref(s.rel_tab, tab_at), aref(s.part_o, part), aref(s.part_ml, part_ml), aref(s.sink, base * pl.qHeads() * 2), aref(s.gate, q_at), aref(s.att, q_at));
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
        const pl = g.m.options.place;
        if (pl.split()) {
            const local_ids = at(s.members, 0);
            if (group.live) |gp| {
                try xpu.launch.dispatch(gp.zero, g.s, .{ @intCast((rows * 8 + 255) / 256), 1, 1 }, .{ ids, at(s.route_weights, 0), local_ids, @as(u32, @intCast(rows * 8)), pl.expertBase(), pl.expertCount() });
            }
            if (rows <= decode_rows) {
                const local = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision, .experts = pl.expertCount() };
                const down_local = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision, .experts = pl.expertCount() };
                switch (precision) {
                    .nvfp4 => try ex.nvfp4Gu(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), tensor(w.gu.amax.?), tensor(w.gu.denominator.?), local_ids, act, local),
                    .mxfp8 => try ex.mxfp8Gu(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), local_ids, act, local),
                }
                switch (precision) {
                    .nvfp4 => try ex.nvfp4Down(g.s, act, tensor(w.down.weight), tensor(w.down.scale), tensor(w.down.amax.?), tensor(w.down.denominator.?), local_ids, down, down_local),
                    .mxfp8 => try ex.mxfp8Down(g.s, act, tensor(w.down.weight), tensor(w.down.scale), local_ids, down, down_local),
                }
            } else {
                const items = at(s.items, 0);
                const count = at(s.count, 0);
                var call = gu_call;
                call.experts = 128;
                try ex.plan(g.s, ids, local_ids, items, count, call);
                if (group.live) |gp| try xpu.launch.dispatch(gp.filter, g.s, .{ 1, 1, 1 }, .{ items, count, pl.expertBase(), pl.expertCount() });
                const local = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision, .experts = pl.expertCount() };
                const down_local = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision, .experts = pl.expertCount() };
                switch (precision) {
                    .nvfp4 => try ex.nvfp4GuPrompt(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), tensor(w.gu.amax.?), tensor(w.gu.denominator.?), items, count, local_ids, act, local),
                    .mxfp8 => try ex.mxfp8GuPrompt(g.s, x, tensor(w.gu.weight), tensor(w.gu.scale), items, count, local_ids, act, local),
                }
                switch (precision) {
                    .nvfp4 => try ex.nvfp4DownPrompt(g.s, act, tensor(w.down.weight), tensor(w.down.scale), tensor(w.down.amax.?), tensor(w.down.denominator.?), items, count, local_ids, down, down_local),
                    .mxfp8 => try ex.mxfp8DownPrompt(g.s, act, tensor(w.down.weight), tensor(w.down.scale), items, count, local_ids, down, down_local),
                }
            }
            try finishExperts(g, down, ids, at(s.route_weights, 0), down_call);
            return;
        }
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
        try finishExperts(g, down, ids, at(s.route_weights, 0), down_call);
    }

fn finishExperts(g: Run, down: Ptr, ids: Ptr, weights: Ptr, call: experts.Call) !void {
    try experts.Kernels.combine(&g.p.experts, g.s, down, ids, weights, at(g.sc.mlp_out, 0), call);
}

    fn visit(g: Run, v: *const layer_plan.Visit, segments: []const Segment, rows: usize, taps: ?Ptr, visit_i: usize) !void {
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
        try g.dense(w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
        try g.dense(w.attn.wk, knorm, at(s.k_raw, 0), rows);
        try g.dense(w.attn.wv, vnorm, at(s.v_raw, 0), rows);
        try g.dense(w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
        try g.dense(w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
        try g.dense(w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
        try g.attendSegments(v, segments);
        if (visit_i == 0 and std.c.getenv("TF_TP_HEAD") != null and (dump_fwd == 0 or dump_fwd == 15)) {
            const local: usize = if (dump_fwd == 0) 10 else 0;
            const tag = if (dump_fwd == 0) "r10" else "r960";
            const pl = g.m.options.place;
            try dumpNamed(&g, tag, "att", plus(at(s.att, 0), local * pl.qWidth() * 2), pl.qWidth() * 2);
            try dumpNamed(&g, tag, "q", plus(at(s.q_prep, 0), local * pl.qWidth() * 2), pl.qWidth() * 2);
            try dumpNamed(&g, tag, "rel", plus(at(s.rel_tab, 0), local * pl.qHeads() * ap.extent * 2), pl.qHeads() * ap.extent * 2);
            try dumpNamed(&g, tag, "sink", plus(at(s.sink, 0), local * pl.qHeads() * 2), pl.qHeads() * 2);
            try dumpNamed(&g, tag, "k", plus(at(s.k_conv, 0), local * pl.kvWidth() * 2), pl.kvWidth() * 2);
        }
        try g.dense(w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
        if (g.m.options.place.split()) if (group.live) |gp| try gp.allReduce(at(s.attn_dense, 0).?, @intCast(rows));
        try dumpBytes(&g, visit_i, "wo", at(s.attn_dense, 0), 2 * state.hidden * 2);
        try dumpBytes(&g, visit_i, "attn-red", at(s.attn_dense, 0), 2 * state.hidden * 2);
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
            for (0..rows) |i| try loop.gates(&g.p.loop, g.s, row(s.mlp_ar, i, state.hidden * 4), messageRow(s, core, 3, i), tensor(c.down), tensor(c.forget), tensor(c.gain), row(s.zeta, i, 64 * 4), row(s.mlp_gated, i, state.hidden * 4), 1);
            try g.norm(at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        } else try g.norm(at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
        try glue.router(&g.p.glue, g.s, at(s.mlp_normed, 0), tensor(w.router_weight), tensor(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
        try dumpBytes(&g, visit_i, "router", at(s.mlp_normed, 0), 2 * state.hidden * 2);
        try dumpBytes(&g, visit_i, "route-ids", at(s.route_ids, 0), 2 * 8 * 4);
        try g.routedExperts(v.layer, rows);
        if (g.m.options.place.split()) if (group.live) |gp| try gp.allReduce(at(s.mlp_out, 0).?, @intCast(rows));
        try dumpBytes(&g, visit_i, "exp-red", at(s.mlp_out, 0), 2 * state.hidden * 2);
        base = 0;
        for (segments) |seg| {
            try g.staticPostConv(seg.cache, v.cache_slot, 3, s.mlp_out, w.mlp_conv, s.mlp_conv, base, seg.rows);
            base += seg.rows;
        }
        try runtime.mlpPartial(&g.p.runtime, g.s, at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0), rows);
        if (v.cache_slot == 5) try g.copyBf16(at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
    }

    fn commitFullHistories(g: Run, segments: []const Segment) !void {
        const pl = g.m.options.place;
        for (segments) |seg| for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
            // KV convolution writes taps at the rank width. The slot stride is the full width, so a full-width
            // copy reads the wrong rows after the first chunk.
            const kv = cs < 2;
            const channels: usize = if (kv) pl.kvWidth() else state.convWidth(cs);
            const src = if (kv) plus(seg.cache.staged(slot, cs, 0).ptr(), seg.rows * channels * 2) else seg.cache.staged(slot, cs, seg.rows).ptr();
            try g.copyBf16(src, seg.cache.history(slot, cs).ptr(), 3 * channels);
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
    if (m.options.place.rank == 0) if (group.live) |gp| try gp.note(.forward, ids, if (options.all) 1 else 0, @intCast(segments[0].cache.len));
    try glue.embedNorm(&p.glue, stream, at(s.ids, 0), tensor(m.weights.embed), tensor(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    for (m.weight_plan.visits[0..m.weight_plan.n_visits], 0..) |*v, vi| try g.visit(v, segments, rows, options.taps, vi);
    if (std.c.getenv("TF_TP_DUMP") != null) dump_fwd += 1;
    try g.widen(at(s.p, 0), at(s.partial, 0), rows);
    const final = finalPlan(rows, options.all, options.taps != null);
    for (0..final.work_rows) |j| {
        const i = if (final.work_rows == rows) j else rows - 1;
        try glue.ar(&p.glue, stream, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), tensor(m.weights.output_query), row(s.mlp_ar, j, state.hidden * 4), m.weight_plan.final_sources, 1);
    }
    try g.norm(at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), final.work_rows);
    if (options.taps) |to| try g.captureTap(at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) at(s.logits_all, 0) else at(segments[0].cache.logits, 0);
    const head_input = at(s.mlp_normed, final.logits_row * state.hidden * 2);
    if (!m.options.place.split()) {
        const vocab_n: u32 = m.config.text.unpadded_vocab;
        if (m.weights.unembed_q8) |head| {
            try dense_q8_kernels.head(&p.dense_q8, stream, tensor(head), head_input, logits, @intCast(final.logits_rows), vocab_n);
        } else try glue.logits(&p.glue, stream, head_input, tensor(m.weights.unembed), logits, @intCast(final.logits_rows), vocab_n);
    } else if (group.live) |gp| {
        // The head kernel stores [row][slice]. The logit buffer is [row][full vocab]. Write the slice aside, pull the
        // other rank's slice, then place both at the full stride. The kernel itself is unchanged.
        const vocab_n = m.options.place.vocabCount();
        const n_rows: usize = final.logits_rows;
        const tight_bytes = n_rows * @as(usize, vocab_n) * 4;
        var tight = try m.r.alloc(tight_bytes);
        defer tight.free();
        var peer = try m.r.alloc(tight_bytes);
        defer peer.free();
        if (m.weights.unembed_q8) |head| {
            try dense_q8_kernels.head(&p.dense_q8, stream, tensor(head), head_input, tight.ptr, @intCast(n_rows), vocab_n);
        } else try glue.logits(&p.glue, stream, head_input, tensor(m.weights.unembed), tight.ptr, @intCast(n_rows), vocab_n);
        try stream.synchronize();
        if (!slice_own_dumped) {
            try dumpF32(m.r, m.options.place.rank, "slice-all", tight.ptr, vocab_n * @as(u32, @intCast(n_rows)));
            slice_own_dumped = true;
        }
        const full: usize = m.config.text.unpadded_vocab;
        const own: usize = m.options.place.vocabBase();
        const other: usize = (full + 1) / 2;
        for (0..n_rows) |r| {
            const off = r * @as(usize, vocab_n) * 4;
            try gp.headPull(plus(tight.ptr, off).?, plus(peer.ptr, off).?, vocab_n);
            if (r + 1 == n_rows and m.options.place.rank == 0 and !slice_pulled_dumped) {
                try stream.synchronize();
                try dumpF32(m.r, 0, "slice-pulled-all", peer.ptr, vocab_n * @as(u32, @intCast(n_rows)));
                slice_pulled_dumped = true;
            }
            if (m.options.place.rank == 0) {
                try copyF32(m.r, plus(logits, (r * full + own) * 4), plus(tight.ptr, off), vocab_n);
                try copyF32(m.r, plus(logits, (r * full + other) * 4), plus(peer.ptr, off), vocab_n);
            }
        }
    }
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

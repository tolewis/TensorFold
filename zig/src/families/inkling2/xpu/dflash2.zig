//! Inkling 2 Mini DFlash 2 on the XPU backend: the block drafter of incoai/Inkling-2-Mini-DFlash2 (Apache 2.0) for the
//! shared lane rounds, beside DSpark (`dspark.zig`), which stays the fallback. Committed target features feed a
//! five-layer block of eight rows (the pending token and seven masks) in one pass; the 16 best logits of each row and a
//! rank-256 pair score trace one chain, drawn with the target's own keyed rule so a draft and the target's later draw
//! meet on the same noise. Every step queues on the target's stream, which the drafter waits for before it reads.
const std = @import("std");
const xpu = @import("xpu");
const lanes = @import("lanes");
const images = @import("inkling2_xpu_images");
const gpu_rule = lanes.gpu_rule;
const Model = @import("model.zig").Model;
const dense_kernels = @import("dense.zig");
const glue = @import("glue.zig");
const ds = @import("dspark_kernels.zig");
const k = @import("dflash2_kernels.zig");
const weight_loader = @import("dflash2_weights.zig");
const state = @import("dflash2_state.zig");
pub const loader = weight_loader;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const hidden = state.hidden;
const topk = k.topk;
const mask_token: u32 = 200056;

fn at(buf: xpu.DeviceBuffer, off: usize) Ref {
    return if (buf.ptr) |p| @ptrFromInt(@intFromPtr(p) + off) else null;
}
fn offset(p: Ref, off: usize) Ref {
    return @ptrFromInt(@intFromPtr(p.?) + off);
}
fn r(t: weight_loader.Tensor) Ref {
    return t.buf.ptr;
}
fn bf(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Where a drafted round's drafter time goes (`TF_DFLASH2_TIMES=1` logs a line every 50 holds).
const Times = struct {
    on: bool = false,
    holds: u64 = 0,
    absorbs: u64 = 0,
    absorb: u64 = 0,
    block: u64 = 0,
    download: u64 = 0,
    chain: u64 = 0,
};

/// The noise rows of the block: the target's raw embedding table rows (the reference), or the table rows through the
/// target's embedding norm (`TF_DFLASH2_EMBED=normed`).
const EmbedMode = enum { raw, normed };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    ctx: *const xpu.Context,
    stream: xpu.Stream,
    weights: weight_loader.Loaded,
    module: xpu.Module,
    kernels: k.Kernels,
    spark_module: xpu.Module,
    spark: ds.Kernels,
    scratch: state.Scratch,
    capacity: usize = 1,
    contexts: std.AutoHashMapUnmanaged(*Stream, state.Lane) = .empty,
    host_ids: [2]u32 = .{ 0, mask_token },
    top_v: []f32,
    top_i: []i32,
    sel: []u16,
    embed_mode: EmbedMode = .raw,
    /// `TF_DFLASH2_FULL=1`: draw all seven positions of every block (the rest are not offered), and log how often each matches.
    full: bool = false,
    times: Times = .{},

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *const Model, drafter_dir: []const u8) !*Drafter {
        const ctx = target.ctx orelse return error.Inkling2NoKernels;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        const stream = target.r.stream(ctx);
        var weights = try weight_loader.load(gpa, io, ctx, stream, drafter_dir);
        errdefer weights.deinit();
        var module = try xpu.Module.loadNative(ctx, &images.dflash2);
        errdefer module.unload();
        var kernels = try k.Kernels.load(module, false);
        errdefer kernels.deinit();
        var spark_module = try xpu.Module.loadNative(ctx, &images.dspark);
        errdefer spark_module.unload();
        var spark = try ds.Kernels.load(spark_module, false);
        errdefer spark.deinit();
        var scratch = try state.Scratch.init(ctx);
        errdefer scratch.deinit();
        const top_v = try gpa.alloc(f32, 7 * state.lists);
        errdefer gpa.free(top_v);
        const top_i = try gpa.alloc(i32, 7 * state.lists);
        errdefer gpa.free(top_i);
        const sel = try gpa.alloc(u16, 7 * state.rank);
        errdefer gpa.free(sel);
        d.* = .{ .gpa = gpa, .target = target, .ctx = ctx, .stream = stream, .weights = weights, .module = module, .kernels = kernels, .spark_module = spark_module, .spark = spark, .scratch = scratch, .top_v = top_v, .top_i = top_i, .sel = sel };
        d.times = .{ .on = std.c.getenv("TF_DFLASH2_TIMES") != null };
        d.full = std.c.getenv("TF_DFLASH2_FULL") != null;
        if (std.c.getenv("TF_DFLASH2_EMBED")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "normed")) d.embed_mode = .normed;
        }
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.stream.synchronize() catch {};
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        d.gpa.free(d.sel);
        d.gpa.free(d.top_i);
        d.gpa.free(d.top_v);
        d.scratch.deinit();
        d.spark.deinit();
        d.spark_module.unload();
        d.kernels.deinit();
        d.module.unload();
        d.weights.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.weights.fixedBytes() + state.scratchBytes();
    }

    /// The attention partials live in the fixed scratch; nothing grows with the window.
    pub fn scoreBytes(_: usize) u64 {
        return state.scratchBytes();
    }

    /// A stream's rings are the window's, whatever the target's context.
    pub fn streamBytes(_: usize) u64 {
        return state.laneBytes();
    }

    pub fn setContext(d: *Drafter, window: u32) !void {
        if (window == 0 or window > 1048576) return error.DFlash2LaneGeometry;
        if (d.contexts.count() != 0) return error.DFlash2ContextActive;
        d.capacity = window;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash 2", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &weight_loader.tap_list;
    }

    fn factsFn(_: *anyopaque) dr.Facts {
        return .{ .fixed_block = true, .depth = 7, .step_ms = 0, .plain_guard = true, .stream_bytes = @intCast(state.laneBytes()) };
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        const lane = try state.Lane.init(d.ctx);
        errdefer {
            var owned = lane;
            owned.deinit();
        }
        try d.contexts.put(d.gpa, stream, lane);
    }

    fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
        const d = cast(ptr);
        if (d.contexts.fetchRemove(stream)) |entry| {
            var lane = entry.value;
            d.stream.synchronize() catch {};
            lane.deinit();
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |stream, tokens| {
            const lane = d.contexts.getPtr(stream) orelse return error.DraftStreamUnknown;
            if (tokens.len > lane.held_len) return error.DraftHeldFewer;
            @memcpy(tokens, lane.held[0..tokens.len]);
        }
    }

    /// out[rows][n] = bf16(input[rows][width] . weight[n][width]) on the target's dense matvec.
    fn project(d: *Drafter, weight: weight_loader.Tensor, input: Ref, out: Ref, rows: usize, n: u32, width: u32) !void {
        const p = if (d.target.pipelines) |*x| x else return error.Inkling2NoKernels;
        try dense_kernels.dense(&p.dense, d.stream, r(weight), input, out, @intCast(rows), n, width);
    }

    /// Committed rows enter every layer's key and value ring: fc over the four taps, its norm, then each layer's k and v.
    fn absorbRows(d: *Drafter, lane: *const state.Lane, taps: Ref, start: u64, rows: usize) !void {
        const s = d.stream;
        const sc = &d.scratch;
        const w = &d.weights.weights;
        const n: u32 = @intCast(rows);
        try d.project(w.fc, taps, at(sc.fc, 0), rows, hidden, 4 * hidden);
        try ds.residualNorm(&d.spark, s, at(sc.fc, 0), at(sc.fc, 0), r(w.hidden_norm), at(sc.junk, 0), at(sc.hn, 0), n, true);
        for (w.layers, 0..) |layer, i| {
            try d.project(layer.k, at(sc.hn, 0), at(sc.ck, 0), rows, state.kv, hidden);
            try d.project(layer.v, at(sc.hn, 0), at(sc.cv, 0), rows, state.kv, hidden);
            try k.commitContext(&d.kernels, s, at(sc.ck, 0), at(sc.cv, 0), r(layer.k_norm), at(lane.k[i], 0), at(lane.v[i], 0), @intCast(start), n, state.ring);
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        errdefer d.stream.synchronize() catch {};
        const t_absorb = nowNs();
        defer if (d.times.on) {
            d.times.absorb += nowNs() - t_absorb;
            d.times.absorbs += 1;
        };
        const row_bytes = 4 * hidden * 2;
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const features = item.features;
            if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            if (item.start < lane.end) return error.DraftContextGap;
            if (item.start > lane.end) lane.from = item.start;
            const end = item.start + features.rows;
            if (end > 1048576) return error.DraftContextGap;
            const source: Ref = @ptrFromInt(features.buffer);
            var position = item.start;
            // Only the last window of a long prompt can ever be read; the earlier rows are skipped.
            if (end - position > state.ring) {
                position = end - state.ring;
                lane.from = position;
            }
            while (position < end) {
                const rows: usize = @intCast(@min(state.absorb_rows, end - position));
                const from = features.offset + @as(usize, @intCast(position - item.start)) * row_bytes;
                try d.absorbRows(lane, offset(source, from), position, rows);
                position += rows;
            }
            lane.end = end;
            lane.held_len = 0;
        }
        try d.stream.synchronize();
    }

    /// One block: the pending token at `anchor` and seven masks through the five layers, then the candidates of `rows`
    /// draft rows. The device work is queued; the candidate lists and the selector rows are queued to the host.
    fn encodeBlock(d: *Drafter, lane: *const state.Lane, pending: u32, anchor: u64, rows: u32) !void {
        const s = d.stream;
        const sc = &d.scratch;
        const w = &d.weights.weights;
        const target = d.target;
        const p = if (target.pipelines) |*x| x else return error.Inkling2NoKernels;
        const table = target.weights.embed.buf.ptr.?;
        const row_bytes: usize = hidden * 2;
        const committed: u32 = @intCast(lane.end);
        const from: u32 = @intCast(lane.from);
        switch (d.embed_mode) {
            .raw => try ds.initializeRows(&d.spark, s, offset(table, @as(usize, pending) * row_bytes), offset(table, @as(usize, mask_token) * row_bytes), at(sc.x, 0)),
            .normed => {
                d.host_ids[0] = pending;
                try sc.ids.upload(s, 0, std.mem.sliceAsBytes(&d.host_ids));
                try glue.embedNorm(&p.glue, s, at(sc.ids, 0), table, target.weights.embed_norm.buf.ptr, at(sc.y, 0), 2);
                try ds.initializeRows(&d.spark, s, at(sc.y, 0), at(sc.y, row_bytes), at(sc.x, 0));
            },
        }
        try ds.residualNorm(&d.spark, s, at(sc.x, 0), at(sc.resid, 0), r(w.layers[0].input_norm), at(sc.resid, 0), at(sc.y, 0), 8, true);
        for (w.layers, 0..) |layer, i| {
            // attention: dynamic conv in, q k v, block attention, o, dynamic conv out
            try d.project(layer.attn_kernel, at(sc.y, 0), at(sc.dyn, 0), 8, state.dyn_width, hidden);
            try k.conv(&d.kernels, s, at(sc.y, 0), at(sc.dyn, 0), r(layer.attn_base), at(sc.conv, 0), 8);
            try d.project(layer.q, at(sc.conv, 0), at(sc.q_raw, 0), 8, state.query, hidden);
            try d.project(layer.k, at(sc.conv, 0), at(sc.k_raw, 0), 8, state.kv, hidden);
            try d.project(layer.v, at(sc.conv, 0), at(sc.v_raw, 0), 8, state.kv, hidden);
            try k.blockQk(&d.kernels, s, at(sc.q_raw, 0), at(sc.k_raw, 0), r(layer.q_norm), r(layer.k_norm), at(sc.q, 0), at(sc.kb, 0), @intCast(anchor));
            try k.attend(&d.kernels, s, at(sc.q, 0), at(sc.kb, 0), at(sc.v_raw, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(sc.part, 0), at(sc.attn, 0), committed, state.ring, from);
            try d.project(layer.o, at(sc.attn, 0), at(sc.out, 0), 8, hidden, state.query);
            try k.conv(&d.kernels, s, at(sc.out, 0), at(sc.dyn, state.dyn_width / 2 * 2), offset(r(layer.attn_base), 2 * hidden * 2), at(sc.conv, 0), 8);
            try ds.residualNorm(&d.spark, s, at(sc.conv, 0), at(sc.resid, 0), r(layer.post_norm), at(sc.next_resid, 0), at(sc.y, 0), 8, false);
            // MLP
            try d.project(layer.mlp_kernel, at(sc.y, 0), at(sc.dyn, 0), 8, state.dyn_width, hidden);
            try k.conv(&d.kernels, s, at(sc.y, 0), at(sc.dyn, 0), r(layer.mlp_base), at(sc.conv, 0), 8);
            try d.project(layer.gate, at(sc.conv, 0), at(sc.gate, 0), 8, state.intermediate, hidden);
            try d.project(layer.up, at(sc.conv, 0), at(sc.up, 0), 8, state.intermediate, hidden);
            try xpu.launch.dispatch(d.spark.swiglu, s, .{ (8 * state.intermediate + 255) / 256, 1, 1 }, .{ at(sc.gate, 0), at(sc.up, 0), at(sc.act, 0), @as(u32, 8 * state.intermediate) });
            try d.project(layer.down, at(sc.act, 0), at(sc.out, 0), 8, hidden, state.intermediate);
            try k.conv(&d.kernels, s, at(sc.out, 0), at(sc.dyn, state.dyn_width / 2 * 2), offset(r(layer.mlp_base), 2 * hidden * 2), at(sc.conv, 0), 8);
            const last = i + 1 == w.layers.len;
            const next_norm = if (last) w.norm else w.layers[i + 1].input_norm;
            try ds.residualNorm(&d.spark, s, at(sc.conv, 0), at(sc.next_resid, 0), r(next_norm), at(sc.resid, 0), if (last) at(sc.final, 0) else at(sc.y, 0), 8, false);
        }
        // rows 1 .. rows of the final norm: the unembedding once for all of them, the 16 best of each, the selector rows
        const draft_rows = at(sc.final, row_bytes);
        try ds.base(&d.spark, s, target.weights.unembed.buf.ptr, draft_rows, at(sc.base, 0), rows, state.vocabulary);
        try k.topKeep(&d.kernels, s, at(sc.base, 0), state.vocabulary, state.vocabulary, rows, at(sc.top_v, 0), at(sc.top_i, 0));
        try d.project(w.selector, draft_rows, at(sc.sel, 0), rows, state.rank, hidden);
        try sc.top_v.downloadAsync(s, 0, std.mem.sliceAsBytes(d.top_v[0 .. rows * state.lists]));
        try sc.top_i.downloadAsync(s, 0, std.mem.sliceAsBytes(d.top_i[0 .. rows * state.lists]));
        try sc.sel.downloadAsync(s, 0, std.mem.sliceAsBytes(d.sel[0 .. rows * state.rank]));
    }

    /// Row `row`'s 16 best (value, id) from its 256 lists, value descending then id ascending.
    fn merge(d: *const Drafter, row: usize, ids: *[topk]u32, vals: *[topk]f32) void {
        var bv: [topk]f32 = @splat(-std.math.inf(f32));
        var bi: [topk]i32 = @splat(std.math.maxInt(i32));
        const base = row * state.lists;
        for (0..state.lists) |j| {
            const v = d.top_v[base + j];
            const ix = d.top_i[base + j];
            if (ix == std.math.maxInt(i32)) continue;
            if (v < bv[topk - 1] or (v == bv[topk - 1] and ix > bi[topk - 1])) continue;
            var at_: usize = topk - 1;
            while (at_ > 0 and (v > bv[at_ - 1] or (v == bv[at_ - 1] and ix < bi[at_ - 1]))) : (at_ -= 1) {
                bv[at_] = bv[at_ - 1];
                bi[at_] = bi[at_ - 1];
            }
            bv[at_] = v;
            bi[at_] = ix;
        }
        for (0..topk) |j| {
            ids[j] = if (bi[j] == std.math.maxInt(i32)) 0 else @intCast(bi[j]);
            vals[j] = bv[j];
        }
    }

    /// One draft: the 16 candidates of row `row`, scored by the logit plus the pair term with the previous token (the
    /// selector row, the predecessor's codebook row, the candidate's), capped after the sum, then drawn by the target's rule.
    fn draw(d: *Drafter, row: usize, previous: u32, sampling: ?lanes.sampling.Sampling, position: u32) !u32 {
        var ids: [topk]u32 = undefined;
        var vals: [topk]f32 = undefined;
        d.merge(row, &ids, &vals);
        var left: [state.rank]f32 = undefined;
        const pred_row = d.weights.pred[@as(usize, previous) * state.rank ..][0..state.rank];
        const sel_row = d.sel[row * state.rank ..][0..state.rank];
        for (&left, pred_row, sel_row) |*o, p, h| o.* = bf(p) * bf(h);
        var scores: [topk]f32 = undefined;
        for (ids, vals, &scores) |id, v, *out| {
            const succ_row = d.weights.succ[@as(usize, id) * state.rank ..][0..state.rank];
            var e: f32 = 0;
            for (left, succ_row) |a, b| e += a * bf(b);
            out.* = 30.0 * std.math.tanh((v + e) / 30.0);
        }
        const sm = sampling orelse {
            var best: usize = 0;
            for (1..topk) |j| {
                if (scores[j] > scores[best] or (scores[j] == scores[best] and ids[j] < ids[best])) best = j;
            }
            return ids[best];
        };
        const inv_t = gpu_rule.rule(sm).inv_t;
        var cands: [topk]gpu_rule.Cand = undefined;
        var m: f32 = -std.math.inf(f32);
        for (&cands, scores, 0..) |*c, sc_, j| {
            c.* = .{ .v = sc_ * inv_t, .i = @intCast(j) };
            m = @max(m, c.v);
        }
        std.mem.sort(gpu_rule.Cand, &cands, {}, gpu_rule.better);
        var z: f32 = 0;
        for (cands) |c| z += @exp(c.v - m);
        return gpu_rule.pick(&cands, m, z, sm, position, &ids);
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        errdefer d.stream.synchronize() catch {};
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (lane.end != anchor) return error.DraftContextGap;
            const offered: u32 = @min(item.depth, 7);
            const depth: u32 = if (d.full) 7 else offered;
            const t0 = nowNs();
            try d.encodeBlock(lane, item.pending, anchor, depth);
            try d.stream.synchronize();
            const t1 = nowNs();
            var previous = item.pending;
            const base_position = std.math.cast(u32, item.position) orelse return error.DraftDepth;
            for (0..depth) |step| {
                const position = std.math.add(u32, base_position, @intCast(step)) catch return error.DraftDepth;
                previous = try d.draw(step, previous, item.sampling, position);
                if (previous >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = previous;
            }
            lane.held_len = offered;
            if (d.times.on) {
                d.times.holds += 1;
                d.times.block += t1 - t0;
                d.times.chain += nowNs() - t1;
                if (d.times.holds % 50 == 0) {
                    const t = d.times;
                    const ms = struct {
                        fn of(ns: u64, n: u64) f64 {
                            return @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(@max(n, 1)));
                        }
                    }.of;
                    std.log.info("dflash2 times: holds {d} | per hold: block + download {d:.2} ms, host chain {d:.2} ms | absorb {d:.2} ms each ({d})", .{ t.holds, ms(t.block, t.holds), ms(t.chain, t.holds), ms(t.absorb, t.absorbs), t.absorbs });
                }
            }
        }
    }
};

test "the block's noise rows and mask id are the checkpoint's" {
    try std.testing.expectEqual(@as(u32, 200056), mask_token);
    try std.testing.expectEqual(@as(usize, 4096), state.lists);
}

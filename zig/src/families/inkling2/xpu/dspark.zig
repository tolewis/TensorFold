//! Inkling 2 Mini DSpark on the XPU backend: committed target features, three-layer proposal blocks, folded Markov logits.
//! The drafter of `metal/dspark.zig` on the device runtime: every step is queued on the target's stream, which the
//! drafter waits for before it reads a result.
const std = @import("std");
const xpu = @import("xpu");
const lanes = @import("lanes");
const images = @import("inkling2_xpu_images");
const gpu_rule = lanes.gpu_rule;
const Model = @import("model.zig").Model;
const dense_kernels = @import("dense.zig");
const glue = @import("glue.zig");
const k = @import("dspark_kernels.zig");
const weight_loader = @import("dspark_weights.zig");
const state = @import("dspark_state.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;
const tap_list = [_]u32{ 5, 23, 41, 48 };

fn at(buf: xpu.DeviceBuffer, off: usize) Ref {
    return if (buf.ptr) |p| @ptrFromInt(@intFromPtr(p) + off) else null;
}
fn offset(p: Ref, off: usize) Ref {
    return @ptrFromInt(@intFromPtr(p.?) + off);
}
fn r(t: weight_loader.Tensor) Ref {
    return t.buf.ptr;
}

extern "c" fn fflush(stream: ?*std.c.FILE) c_int;

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    target: *const Model,
    ctx: *const xpu.Context,
    stream: xpu.Stream,
    weights: weight_loader.Loaded,
    module: xpu.Module,
    kernels: k.Kernels,
    scratch: state.Scratch,
    scores: ?xpu.DeviceBuffer = null,
    capacity: usize = 1,
    contexts: std.AutoHashMapUnmanaged(*Stream, state.Lane) = .empty,
    /// The pending token's host copy, and one row of bf16 logits.
    host_id: [1]u32 = .{0},
    host_logits: []u16,
    /// Diagnostic trace (`TF_DSPARK_TRACE=file`): every absorb and the first blocks' intermediates, for replay against a reference.
    trace: ?*std.c.FILE = null,
    trace_blocks: u32 = 0,
    trace_absorbs: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *const Model, drafter_dir: []const u8) !*Drafter {
        const ctx = target.ctx orelse return error.Inkling2NoKernels;
        const mtp = target.config.mtp orelse return error.DSparkNotConfigured;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.stream = target.r.stream(ctx);
        // The fold reads the target's unembedding once, as host bf16.
        const unembed = target.weights.unembed;
        const host = try gpa.alloc(u16, unembed.bytes / 2);
        defer gpa.free(host);
        try target.r.download(std.mem.sliceAsBytes(host), unembed.buf);
        try target.r.sync();
        var config: weight_loader.Config = .{
            .layers = mtp.layers,
            .q_heads = mtp.q_heads,
            .kv_heads = mtp.kv_heads,
            .head_dim = mtp.head_dim,
            .inter = mtp.inter,
            .window = mtp.window,
            .block = mtp.block,
            .rank = mtp.rank,
            .rope_theta = mtp.rope_theta,
            .taps_n = mtp.taps_n,
            .taps = @splat(0),
            .local = @splat(false),
        };
        for (0..@min(4, mtp.taps_n)) |i| config.taps[i] = mtp.taps[i];
        for (0..@min(3, mtp.layers)) |i| config.local[i] = mtp.local[i];
        const weights = try weight_loader.load(gpa, io, ctx, d.stream, config, drafter_dir, host);
        d.* = .{ .gpa = gpa, .target = target, .ctx = ctx, .stream = d.stream, .weights = weights, .module = undefined, .kernels = undefined, .scratch = undefined, .host_logits = undefined };
        errdefer d.weights.deinit();
        d.module = try xpu.Module.loadNative(ctx, &images.dspark);
        errdefer d.module.unload();
        d.kernels = try k.Kernels.load(d.module, false);
        errdefer d.kernels.deinit();
        d.scratch = try state.Scratch.init(ctx);
        errdefer d.scratch.deinit();
        d.host_logits = try gpa.alloc(u16, state.vocabulary);
        d.trace = null;
        d.trace_blocks = 0;
        if (std.c.getenv("TF_DSPARK_TRACE")) |path| d.trace = std.c.fopen(path, "wb");
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.stream.synchronize() catch {};
        if (d.trace) |f| _ = std.c.fclose(f);
        var it = d.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        d.contexts.deinit(d.gpa);
        if (d.scores) |*scores| scores.free();
        d.gpa.free(d.host_logits);
        d.scratch.deinit();
        d.kernels.deinit();
        d.module.unload();
        d.weights.deinit();
        d.gpa.destroy(d);
    }

    fn traceRecord(d: *Drafter, tag: u8, parts: []const []const u8) void {
        const f = d.trace orelse return;
        const n: u32 = @intCast(parts.len);
        _ = std.c.fwrite(@ptrCast(&tag), 1, 1, f);
        _ = std.c.fwrite(@ptrCast(&n), 4, 1, f);
        for (parts) |part| {
            const len: u64 = part.len;
            _ = std.c.fwrite(@ptrCast(&len), 8, 1, f);
            if (part.len > 0) _ = std.c.fwrite(part.ptr, 1, part.len, f);
        }
        _ = fflush(f);
    }

    /// Copies `n` device bytes to the host (waits), then writes them as the record's second part after `head`.
    fn traceDevice(d: *Drafter, tag: u8, head: []const u8, src: Ref, n: usize) void {
        if (d.trace == null) return;
        const host = d.gpa.alloc(u8, n) catch return;
        defer d.gpa.free(host);
        d.stream.copy(host.ptr, src, n) catch return;
        d.stream.synchronize() catch return;
        d.traceRecord(tag, &.{ head, host });
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.weights.fixedBytes() + state.scratchBytes();
    }

    pub fn scoreBytes(capacity: usize) u64 {
        return state.scoreBytes(capacity);
    }

    pub fn streamBytes(capacity: usize) u64 {
        return state.laneBytes(capacity);
    }

    /// The selected target window fixes every new lane's global ring and the shared attention score allocation.
    pub fn setContext(d: *Drafter, window: u32) !void {
        const capacity: usize = @intCast(window);
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        if (d.contexts.count() != 0) return error.DSparkContextActive;
        const scores = try xpu.DeviceBuffer.alloc(d.ctx, @intCast(state.scoreBytes(capacity)));
        if (d.scores) |*old| old.free();
        d.scores = scores;
        d.capacity = capacity;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &tap_list;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = 7, .step_ms = 0, .plain_guard = true, .stream_bytes = @intCast(state.laneBytes(d.capacity)) };
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.contexts.contains(stream)) return error.DraftStreamOpen;
        const lane = try state.Lane.init(d.ctx, d.capacity);
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

    /// A projection of the drafter's own weights: `weight` is [out][in] bf16.
    fn project(d: *Drafter, weight: weight_loader.Tensor, input: Ref, out: Ref, rows: usize) !void {
        const p = if (d.target.pipelines) |*x| x else return error.Inkling2NoKernels;
        const n: u32 = @intCast(weight.shape[weight.rank - 2]);
        const width: u32 = @intCast(weight.shape[weight.rank - 1]);
        try dense_kernels.dense(&p.dense, d.stream, r(weight), input, out, @intCast(rows), n, width);
    }

    fn absorbRows(d: *Drafter, lane: *const state.Lane, taps: Ref, start: u64, rows: usize) !void {
        const s = d.stream;
        const n: u32 = @intCast(rows);
        try k.fuseTaps(&d.kernels, s, taps, r(d.weights.weights.fuse), at(d.weights.dflare, 0), at(d.scratch.fused, 0), at(lane.newest, 0), n);
        if (d.trace != null and d.trace_absorbs < 2) d.traceDevice('M', std.mem.asBytes(&start), at(d.scratch.fused, 0), 4 * rows * state.hidden * 2);
        for (d.weights.weights.layers, 0..) |layer, i| {
            const c = at(d.scratch.fused, i * rows * state.hidden * 2);
            try d.project(layer.k_ctx, c, at(d.scratch.ck, 0), rows);
            try d.project(layer.v_ctx, c, at(d.scratch.cv, 0), rows);
            if (d.trace != null and d.trace_absorbs < 2 and i == 0) {
                d.traceDevice('C', std.mem.asBytes(&start), at(d.scratch.ck, 0), rows * state.kv * 2);
                d.traceDevice('D', std.mem.asBytes(&start), at(d.scratch.cv, 0), rows * state.kv * 2);
            }
            try k.commitContext(&d.kernels, s, at(d.scratch.ck, 0), at(d.scratch.cv, 0), r(layer.k_norm_ctx), at(lane.k[i], 0), at(lane.v[i], 0), @intCast(start), n, @intCast(lane.rings[i]));
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        if (items.len == 0) return;
        errdefer d.stream.synchronize() catch {};
        const row_bytes = 4 * state.hidden * 2;
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
            if (d.trace != null) {
                const meta = [_]u64{ item.start, features.rows, lane.from, lane.end };
                d.traceRecord('F', &.{ std.mem.sliceAsBytes(&meta), std.mem.sliceAsBytes(item.follow) });
                d.traceDevice('G', std.mem.asBytes(&item.start), offset(source, features.offset), @as(usize, features.rows) * row_bytes);
            }
            var position = item.start;
            while (position < end) {
                const rows: usize = @intCast(@min(state.absorb_rows, end - position));
                const from = features.offset + @as(usize, @intCast(position - item.start)) * row_bytes;
                try d.absorbRows(lane, offset(source, from), position, rows);
                position += rows;
            }
            lane.end = end;
            lane.held_len = 0;
            d.trace_absorbs += 1;
        }
        try d.stream.synchronize();
    }

    fn encodeBlock(d: *Drafter, lane: *const state.Lane, pending: u32, anchor: u64, debug: bool) !void {
        const s = d.stream;
        const sc = &d.scratch;
        const target = d.target;
        const p = if (target.pipelines) |*x| x else return error.Inkling2NoKernels;
        d.host_id[0] = pending;
        try sc.ids.upload(s, 0, std.mem.sliceAsBytes(&d.host_id));
        try glue.embedNorm(&p.glue, s, at(sc.ids, 0), target.weights.embed.buf.ptr, target.weights.embed_norm.buf.ptr, at(sc.x, 0), 1);
        try k.initializeRows(&d.kernels, s, at(sc.x, 0), r(d.weights.weights.mask), at(sc.x, 0));
        try k.residualNorm(&d.kernels, s, at(sc.x, 0), at(sc.resid, 0), r(d.weights.weights.layers[0].input_norm), at(sc.resid, 0), at(sc.x, 0), 8, true);
        if (debug) d.traceDevice('X', &.{0xff}, at(sc.x, 0), 8 * state.hidden * 2);
        for (d.weights.weights.layers, 0..) |layer, i| {
            try d.project(layer.q, at(sc.x, 0), at(sc.q_raw, 0), 8);
            try d.project(layer.k, at(sc.x, 0), at(sc.k_raw, 0), 8);
            try d.project(layer.v, at(sc.x, 0), at(sc.v_raw, 0), 8);
            try k.blockQkv(&d.kernels, s, at(sc.q_raw, 0), at(sc.k_raw, 0), at(sc.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(sc.q, 0), at(sc.kb, 0), at(sc.vb, 0), @intCast(anchor));
            try k.attend(&d.kernels, s, at(sc.q, 0), at(sc.kb, 0), at(sc.vb, 0), at(lane.k[i], 0), at(lane.v[i], 0), at(d.scores.?, 0), at(sc.attn, 0), @intCast(lane.end), @intCast(lane.rings[i]), i != 1, @intCast(lane.from));
            if (debug) {
                const tag: [2]u8 = .{ @intCast(i), 0 };
                d.traceDevice('Q', &tag, at(sc.q, 0), 8 * state.query * 2);
                d.traceDevice('K', &tag, at(sc.kb, 0), 8 * state.kv * 2);
                d.traceDevice('V', &tag, at(sc.vb, 0), 8 * state.kv * 2);
                d.traceDevice('T', &tag, at(sc.attn, 0), 8 * state.query * 2);
            }
            try d.project(layer.o, at(sc.attn, 0), at(sc.y, 0), 8);
            try k.residualNorm(&d.kernels, s, at(sc.y, 0), at(sc.resid, 0), r(layer.post_norm), at(sc.next_resid, 0), at(sc.x, 0), 8, false);
            try d.project(layer.gate, at(sc.x, 0), at(sc.gate, 0), 8);
            try d.project(layer.up, at(sc.x, 0), at(sc.up, 0), 8);
            try k.swiglu(&d.kernels, s, at(sc.gate, 0), at(sc.up, 0), at(sc.act, 0), 8 * state.intermediate);
            try d.project(layer.down, at(sc.act, 0), at(sc.y, 0), 8);
            const next_norm = if (i + 1 < d.weights.weights.layers.len) d.weights.weights.layers[i + 1].input_norm else d.weights.weights.norm;
            try k.residualNorm(&d.kernels, s, at(sc.y, 0), at(sc.next_resid, 0), r(next_norm), at(sc.resid, 0), at(sc.x, 0), 8, false);
            if (debug) {
                const tag: [2]u8 = .{ @intCast(i), 0 };
                d.traceDevice('R', &tag, at(sc.resid, 0), 8 * state.hidden * 2);
                d.traceDevice('N', &tag, at(sc.x, 0), 8 * state.hidden * 2);
            }
        }
        try k.base(&d.kernels, s, target.weights.unembed.buf.ptr, at(sc.x, state.hidden * 2), at(sc.base, 0), 7, state.vocabulary);
        try k.kappa(&d.kernels, s, at(sc.x, state.hidden * 2), at(lane.newest, 0), r(d.weights.weights.rnn), r(d.weights.weights.rnn_bias), at(sc.kappa, 0), 7);
    }

    fn recurrent(d: *Drafter, step: usize, previous: u32, second_bank: bool) !void {
        const sc = &d.scratch;
        const current = if (second_bank) sc.next_state else sc.state;
        const next = if (second_bank) sc.state else sc.next_state;
        try k.recurrentStep(&d.kernels, d.stream, at(current, 0), at(sc.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(sc.z, 0), at(next, 0), at(sc.lambda, 0), previous);
        try k.finalLogits(&d.kernels, d.stream, at(sc.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(sc.lambda, 0), at(sc.logits, 0));
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        errdefer d.stream.synchronize() catch {};
        for (items) |item| {
            const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (lane.end != anchor) return error.DraftContextGap;
            const depth: usize = @min(item.depth, 7);
            try d.scratch.state.fill8(d.stream, 0);
            const debug = d.trace != null and d.trace_blocks < 3;
            if (debug) {
                for (0..3) |i| {
                    const tag: [2]u8 = .{ @intCast(i), 0 };
                    const bytes = lane.rings[i] * state.kv * 2;
                    d.traceDevice('k', &tag, at(lane.k[i], 0), bytes);
                    d.traceDevice('v', &tag, at(lane.v[i], 0), bytes);
                }
                d.traceDevice('f', &.{0}, at(lane.newest, 0), state.hidden * 2);
            }
            try d.encodeBlock(lane, item.pending, anchor, debug);
            try d.recurrent(0, item.pending, false);
            try d.stream.synchronize();
            if (debug) d.traceDevice('a', &.{0}, at(d.scratch.kappa, 0), 7 * 3 * state.rank * 4);
            const values = try d.gpa.alloc(f32, state.vocabulary);
            defer d.gpa.free(values);
            var previous = item.pending;
            var second_bank = false;
            for (0..depth) |step| {
                if (step != 0) {
                    second_bank = !second_bank;
                    try d.recurrent(step, previous, second_bank);
                    try d.stream.synchronize();
                }
                try d.scratch.logits.download(d.stream, 0, std.mem.sliceAsBytes(d.host_logits[0..state.vocabulary]));
                for (d.host_logits[0..state.vocabulary], values) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
                const base_position = std.math.cast(u32, item.position) orelse return error.DraftDepth;
                const position = std.math.add(u32, base_position, @intCast(step)) catch return error.DraftDepth;
                previous = try gpu_rule.sample(d.gpa, values, item.sampling, position, null);
                if (previous >= state.vocabulary) return error.DraftNonfinite;
                lane.held[step] = previous;
                if (d.trace != null) {
                    var top: [8]u32 = @splat(0);
                    var topv: [8]f32 = @splat(-std.math.inf(f32));
                    for (values, 0..) |value, id| {
                        if (value <= topv[7]) continue;
                        var at_: usize = 7;
                        while (at_ > 0 and topv[at_ - 1] < value) : (at_ -= 1) {
                            topv[at_] = topv[at_ - 1];
                            top[at_] = top[at_ - 1];
                        }
                        topv[at_] = value;
                        top[at_] = @intCast(id);
                    }
                    const head = [_]u64{ item.position, item.pending, step, previous };
                    d.traceRecord('S', &.{ std.mem.sliceAsBytes(&head), std.mem.sliceAsBytes(&top), std.mem.sliceAsBytes(&topv) });
                    if (debug and step == 0) d.traceRecord('L', &.{std.mem.sliceAsBytes(d.host_logits[0..state.vocabulary])});
                }
            }
            lane.held_len = @intCast(depth);
            if (d.trace != null) {
                const head = [_]u64{ item.position, item.pending, depth, lane.end, lane.from };
                d.traceRecord('B', &.{ std.mem.sliceAsBytes(&head), std.mem.sliceAsBytes(lane.held[0..depth]) });
                d.trace_blocks += 1;
            }
        }
    }
};

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &tap_list);
    try std.testing.expectEqual(@as(u32, 48), tap_list[3]);
    try std.testing.expectEqual(@as(u64, 768 * 1032), Drafter.scoreBytes(1024));
}

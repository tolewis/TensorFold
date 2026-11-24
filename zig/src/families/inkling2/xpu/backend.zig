//! Inkling 2 Mini behind the lane engine on the XPU backend: independent visit state and exact copy-window rollback.
//! The adapter of `metal/backend.zig` on `forward.zig`'s public API: one stream of the device runs every forward, which
//! returns when the stream has finished it, so a draw reads the logits straight after.
const std = @import("std");
const xpu = @import("xpu");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const snapshot = @import("snapshot.zig");
const sampler = @import("sampler.zig");
const cfg = @import("../config.zig");
const be = lanes.backend;

pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64 };
const ring = 4096;
/// The forward serves the K/V rows as bf16 (an int8 cache is refused there), so every cache is bf16.
const kv: st.KvDtype = .bf16;

/// The prompt cache gets the bytes left by live streams and never evicts the state being restored.
pub const Room = struct { ctx: *anyopaque, fit: *const fn (ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void };

/// Mini's DSpark ABI is fixed: four visit taps in checkpoint order.
pub fn validateTaps(c: *const cfg.Config, taps: []const u32) !void {
    const mtp = c.mtp orelse return error.UnsupportedInkling2Taps;
    const expected = [_]u32{ 5, 23, 41, 48 };
    if (mtp.taps_n != expected.len or !std.mem.eql(u32, mtp.taps[0..mtp.taps_n], &expected) or !std.mem.eql(u32, taps, &expected))
        return error.UnsupportedInkling2Taps;
}

/// Device bytes of one stream's state for `positions` tokens (prompt plus reply).
pub fn streamBytes(positions: usize) u64 {
    return st.cacheBytes(positions, kv);
}

/// Device bytes the backend's scratch holds for good, for a context of `capacity` positions.
pub fn scratchBytes(capacity: usize) u64 {
    return st.scratchBytes(st.max_rows, capacity);
}

pub const Xpu = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    held: u64 = 0,
    room: ?Room = null,
    tokens: []u32,
    next: u64 = 0,
    /// Host copies of device logits: one row for a draw, the rows of a window for a verify, and the masked row.
    values: []f32,
    row: []f32,
    window: []f32,
    /// Selects a sampled row's candidates on the device, so a draw copies back the candidates and not the row.
    sampler: sampler.Sampler,
    prepared: bool = false,
    sink: ?be.FeatureSink = null,
    tap_gen: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Xpu {
        const out = try gpa.create(Xpu);
        errdefer gpa.destroy(out);
        var scratch = try st.Scratch.init(model.r, st.max_rows, options.capacity);
        errdefer scratch.deinit();
        const tokens = try gpa.alloc(u32, ring);
        errdefer gpa.free(tokens);
        const vocab_size = model.config.text.unpadded_vocab;
        const values = try gpa.alloc(f32, vocab_size);
        errdefer gpa.free(values);
        const row = try gpa.alloc(f32, vocab_size);
        errdefer gpa.free(row);
        const window = try gpa.alloc(f32, st.batch_rows * vocab_size);
        errdefer gpa.free(window);
        const ctx = model.ctx orelse return error.Inkling2NoKernels;
        var select = try sampler.Sampler.init(gpa, model.r, ctx, st.batch_rows);
        errdefer select.deinit();
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .tokens = tokens, .values = values, .row = row, .window = window, .sampler = select };
        return out;
    }

    pub fn deinit(self: *Xpu) void {
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        self.sampler.deinit();
        self.scratch.deinit();
        self.gpa.free(self.tokens);
        self.gpa.free(self.values);
        self.gpa.free(self.row);
        self.gpa.free(self.window);
        self.gpa.destroy(self);
    }

    pub fn createCache(self: *Xpu, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.r, capacity, &self.model.weight_plan, kv);
        cache.kv_channels = self.model.options.place.kvWidth();
        return cache;
    }

    pub fn facts(self: *const Xpu) lanes.Model {
        return .{ .exact_width = st.window_rows, .gpu_tokens = false, .hidden_rows = true, .batch_rows = fwd.decode_rows, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 1, .max_streams = self.options.streams, .streams_exact = true };
    }

    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (self.prepared) return;
        if (self.scratch.taps == null) self.scratch.taps = try self.model.r.alloc(st.captureBytes(st.max_rows));
        self.prepared = true;
    }

    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (!self.prepared) return error.UnpreparedInkling2Taps;
        const tp = (try self.cacheOf(stream)).tap;
        const buf = tp.buf orelse return error.Inkling2TapsNotHeld;
        const off = try st.featureOffset(tp, start, count, self.tap_gen);
        return .{ .buffer = @intFromPtr(buf.ptr.?), .offset = off, .rows = count, .row_bytes = @intCast(st.feature_row_bytes), .space = .device, .dtype = .bf16 };
    }

    fn prefillFeaturesFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) !void {
        const self = cast(ptr);
        try validateTaps(&self.model.config, taps);
        if (!self.prepared) return error.UnpreparedInkling2Taps;
        self.sink = sink;
        defer self.sink = null;
        return prefillFn(ptr, stream);
    }

    fn nextTapGeneration(self: *Xpu) u64 {
        self.tap_gen +%= 1;
        return self.tap_gen;
    }

    pub fn backend(self: *Xpu) be.Backend {
        return .{ .keep_full_lone = true, .ptr = self, .vtable = &.{ .prefill = prefillFn, .prepare_features = prepareFn, .features = featuresFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
    }

    fn cast(ptr: *anyopaque) *Xpu {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Xpu, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownInklingStream;
    }

    fn vocab(self: *const Xpu) usize {
        return self.model.config.text.unpadded_vocab;
    }

    /// The last row the forward wrote for `cache`, read back to the host.
    fn logits(self: *Xpu, cache: *st.Cache) ![]const f32 {
        try cache.readLogits(self.row);
        return self.row;
    }

    fn draw(self: *Xpu, l: []const f32, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        const t = sampled(sampling) orelse {
            var best: ?usize = null;
            for (l, 0..) |w, id| {
                if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
                if (best == null or w > l[best.?]) best = id;
            }
            return @intCast(best orelse return error.InklingEmptyTokenMask);
        };
        const values = if (mask.len == 0) l else blk: {
            for (l, self.values, 0..) |w, *v, id| v.* = if (lanes.grammar.allows(mask, id)) w else -std.math.inf(f32);
            break :blk self.values;
        };
        return lanes.gpu_full.draw(self.gpa, values, t, @intCast(position), null);
    }

    /// The next token of `cache`'s last row. A sampled row's candidates are selected on the device and only they come back,
    /// with the whole row too where the draw sums every token (top_k off); greedy and masked rows are drawn from the row.
    fn drawLast(self: *Xpu, cache: *st.Cache, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        const t = sampled(sampling) orelse return self.draw(try self.logits(cache), sampling, position, mask);
        if (mask.len > 0 or !sampler.Row.supported(t)) return self.draw(try self.logits(cache), sampling, position, mask);
        const row = sampler.Row.of(t);
        const set = try self.sampler.select(cache.logits.ptr, @intCast(self.vocab()), &.{row});
        var buf: [sampler.top_max]lanes.gpu_rule.Cand = undefined;
        const top = set.list(0, row.n, &buf);
        const whole: ?[]const f32 = if (lanes.gpu_full.needsRow(t)) try self.logits(cache) else null;
        return lanes.gpu_full.drawTop(whole, top, t, @intCast(position));
    }

    fn sampled(sampling: ?lanes.Sampling) ?lanes.Sampling {
        const t = sampling orelse return null;
        return if (t.temperature > 0) t else null;
    }

    fn take(self: *Xpu) u64 {
        const h = self.next;
        self.next += 1;
        return h;
    }

    fn put(self: *Xpu, token: u32) u64 {
        const h = self.take();
        self.tokens[h % ring] = token;
        return h;
    }

    fn tapPtr(self: *Xpu) ?fwd.Ptr {
        return if (self.prepared) self.scratch.taps.?.ptr else null;
    }

    fn feed(self: *Xpu, cache: *st.Cache, ids: []const u32, stream: ?*lanes.Stream) !void {
        const r = self.model.r;
        const base = cache.len;
        var at: usize = 0;
        while (at < ids.len) {
            var end = @min(ids.len, at + st.prompt_rows);
            if (stream) |sm| {
                if (sm.isCancelled()) return error.Cancelled;
                for (sm.chunks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
                for (sm.reuse.marks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
            }
            const capture = self.prepared and stream != null;
            if (capture) _ = self.nextTapGeneration();
            try fwd.run(self.model, &self.scratch, cache, ids[at..end], .{ .taps = if (capture) self.scratch.taps.?.ptr else null });
            if (capture) {
                const rows = end - at;
                const taps = self.scratch.taps.?;
                if (self.sink) |k| try k.take(k.ptr, stream.?, .{
                    .buffer = @intFromPtr(taps.ptr.?),
                    .offset = 0,
                    .rows = @intCast(rows),
                    .row_bytes = @intCast(st.feature_row_bytes),
                    .space = .device,
                    .dtype = .bf16,
                }, @intCast(cache.len));
                if (base + end == stream.?.prompt().len) {
                    if (cache.last_tap == null) cache.last_tap = try r.alloc(st.feature_row_bytes);
                    try r.copy(cache.last_tap.?, st.view(taps, (rows - 1) * st.feature_row_bytes, st.feature_row_bytes), st.feature_row_bytes);
                    try r.sync();
                    cache.tap = .{ .buf = cache.last_tap, .start = @intCast(cache.len + rows - 1), .rows = 1 };
                }
            }
            cache.len += end - at;
            if (stream) |sm| lanes.prefill_widths.completed(self.gpa, sm, end - at);
            at = end;
            if (stream) |sm| if (std.mem.indexOfScalar(u32, sm.reuse.marks, @intCast(cache.len)) != null) if (sm.reuse.hook) |k| k.at(k.ptr, sm, @intCast(cache.len));
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidInklingPrompt;
        if (stream.images.len > 0) return error.InklingHasNoImageTower;
        const need = streamBytes(ids.len + stream.max_new);
        if (self.held + need > self.options.budget) {
            std.log.warn("inkling2_loop: a stream needs {d:.1} GiB of state; {d:.1} of {d:.1} GiB are held", .{ gib(need), gib(self.held), gib(self.options.budget) });
            return error.InklingStateFull;
        }
        self.held += need;
        errdefer self.held -= need;
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
        const cache = try self.createCache(ids.len + stream.max_new);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache);
        errdefer _ = self.caches.remove(stream);
        stream.cached = 0;
        if (stream.reuse.saved) |saved| {
            const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
            if (snap.at < ids.len) {
                if (snapshot.restore(cache, snap)) stream.cached = @intCast(snap.at) else |_| stream.reuse_failed = true;
            } else stream.reuse_failed = true;
        }
        try self.feed(cache, ids[cache.len..], stream);
    }

    /// One prompt chunk and one window through every kernel, so the first request pays for neither.
    pub fn warm(self: *Xpu) !void {
        var cache = try st.Cache.init(self.model.r, st.prompt_rows + st.window_rows, &self.model.weight_plan, kv);
        cache.kv_channels = self.model.options.place.kvWidth();
        defer cache.deinit();
        var ids: [st.prompt_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
        try self.feed(&cache, &ids, null);
        try fwd.run(self.model, &self.scratch, &cache, ids[0..st.window_rows], .{ .all = true });
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        return cast(ptr).firstSlot(stream, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        return cast(ptr).firstSlot(stream, position, mask);
    }

    fn firstSlot(self: *Xpu, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const cache = try self.cacheOf(stream);
        if (position != cache.len or cache.pending != null) return error.InvalidInklingPosition;
        return self.put(try self.drawLast(cache, stream.sampling, position, mask));
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, what: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1 or cache.pending != null) return error.InvalidInklingPosition;
        const token = switch (what) {
            .handle => |h| try self.read(h),
            .value => |v| v,
        };
        if (token >= self.vocab()) return error.InvalidInklingToken;
        const generation = if (self.prepared) self.nextTapGeneration() else 0;
        try fwd.run(self.model, &self.scratch, cache, &.{token}, .{ .taps = self.tapPtr() });
        if (self.prepared) cache.tap = .{ .buf = self.scratch.taps, .start = @intCast(cache.len), .rows = 1, .gen = generation };
        cache.len += 1;
        return self.put(try self.drawLast(cache, stream.sampling, position, &.{}));
    }

    fn read(self: *Xpu, handle: u64) !u32 {
        if (handle >= self.next) return error.InvalidInklingTokenHandle;
        const token = self.tokens[handle % ring];
        if (token >= self.vocab()) return error.InvalidInklingToken;
        return token;
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        return cast(ptr).read(handle);
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = try self.logits(try self.cacheOf(stream));
        return firstLogprobRow(l, self.tokens, self.next, stream.logprobs orelse 0);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const r = self.model.r;
        const words = (self.vocab() + 31) / 32;
        if (windows.len != out.len or windows.len > st.batch_rows) return error.InvalidInklingWindows;
        var segments: [st.batch_rows]fwd.Segment = undefined;
        var ids: [st.batch_rows]u32 = undefined;
        var total: usize = 0;
        var begun: usize = 0;
        errdefer for (segments[0..begun]) |seg| seg.cache.rollback() catch {};
        for (windows, out, 0..) |w, o, i| {
            const rows = w.rows();
            if (rows == 0 or rows > st.window_rows or total + rows > st.batch_rows or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len != rows - 1 or w.positions.len != rows) return error.UnsupportedInklingWindow;
            const cache = try self.cacheOf(w.stream);
            if (cache.pending != null) return error.InklingWindowAlreadyOpen;
            for (w.positions, 0..) |p, k| if (p != cache.len + k + 1) return error.InvalidInklingPosition;
            segments[i] = .{ .cache = cache, .rows = rows };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += rows;
        }
        for (segments[0..windows.len]) |seg| {
            try seg.cache.begin(seg.rows);
            begun += 1;
        }
        const generation = if (self.prepared) self.nextTapGeneration() else 0;
        try fwd.runShared(self.model, &self.scratch, segments[0..windows.len], ids[0..total], .{ .all = true, .taps = self.tapPtr() });
        const v = self.vocab();
        const all = self.window[0 .. total * v];
        // A sampled row without a mask or logprobs is selected on the device; the rest of the window is drawn from the host rows.
        var picks: [st.batch_rows]sampler.Row = undefined;
        var device: [st.batch_rows]bool = undefined;
        var whole = false;
        var any = false;
        var at: usize = 0;
        for (windows, out) |w, o| {
            const t = sampled(w.stream.sampling);
            for (0..w.rows()) |k| {
                const masked = w.masks.len >= (k + 1) * words;
                const ok = t != null and !masked and o.rows.len != w.rows() and sampler.Row.supported(t.?);
                device[at + k] = ok;
                any = any or ok;
                picks[at + k] = if (ok) sampler.Row.of(t.?) else .{ .inv_t = 1.0, .n = 1 };
                if (!ok or lanes.gpu_full.needsRow(t.?)) whole = true;
            }
            at += w.rows();
        }
        const set = if (any) try self.sampler.select(self.scratch.logits_all.ptr, @intCast(v), picks[0..total]) else undefined;
        if (whole) {
            try r.download(std.mem.sliceAsBytes(all), self.scratch.logits_all);
            try r.sync();
        }
        var base: usize = 0;
        for (windows, out, segments[0..windows.len]) |w, o, seg| {
            if (self.prepared) seg.cache.tap = .{ .buf = self.scratch.taps, .off = base * st.feature_row_bytes, .start = @intCast(seg.cache.len), .rows = seg.rows, .gen = generation };
            @memcpy(o.drafts, w.tokens);
            for (0..seg.rows) |k| {
                const l = all[(base + k) * v ..][0..v];
                const mask = if (w.masks.len >= (k + 1) * words) w.masks[k * words ..][0..words] else &.{};
                if (device[base + k]) {
                    const t = sampled(w.stream.sampling).?;
                    var buf: [sampler.top_max]lanes.gpu_rule.Cand = undefined;
                    const top = set.list(base + k, picks[base + k].n, &buf);
                    o.sampled[k] = lanes.gpu_full.drawTop(if (lanes.gpu_full.needsRow(t)) l else null, top, t, @intCast(w.positions[k]));
                } else {
                    o.sampled[k] = try self.draw(l, w.stream.sampling, w.positions[k], mask);
                }
                if (o.rows.len == seg.rows) o.rows[k] = logprobRow(l, o.sampled[k], w.stream.logprobs orelse 0);
            }
            try r.copy(seg.cache.logits, st.view(self.scratch.logits_all, (base + seg.rows - 1) * v * 4, v * 4), v * 4);
            base += seg.rows;
        }
        try r.sync();
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        if (windows.len != paths.len) return error.InvalidInklingWindows;
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedInklingTree;
            if (path.len == 0 or path.len > w.rows() or cache.pending == null or cache.pending.? != w.rows()) return error.UnsupportedInklingWindow;
            try cache.commit(path.len);
        }
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        _ = ptr;
        for (requests) |r| if (r.depth != 0 or r.lanes != null or r.plan != null) return error.InklingHasNoDraftHead;
    }

    fn heldFn(_: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        if (out.len > 0) return error.InklingDraftsNotHeld;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.caches.fetchRemove(stream)) |entry| {
            const cache = entry.value;
            if (cache.pending != null) cache.rollback() catch |err| std.log.err("inkling2_loop rollback at release: {s}", .{@errorName(err)});
            self.held -= streamBytes(cache.capacity);
            cache.deinit();
            self.gpa.destroy(cache);
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
    }

    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        _ = ptr;
        return snapshot.bytes(at, kv);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        if (cache.len != at or cache.pending != null) return error.SnapshotOutOfStep;
        const snap = try self.gpa.create(snapshot.Snap);
        errdefer self.gpa.destroy(snap);
        snap.* = try snapshot.save(self.model.r, cache);
        return snap;
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores;
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
        snap.free();
        cast(ptr).gpa.destroy(snap);
    }
};

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn firstLogprobRow(l: []const f32, tokens: []const u32, next: usize, count: u8) lanes.LogprobRow {
    std.debug.assert(tokens.len == ring and next > 0);
    return logprobRow(l, tokens[(next - 1) % ring], count);
}

fn logprobRow(l: []const f32, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, w);
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, w) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = l[pick] - lse, .count = 0 };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        if (filled == top and w <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and w > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = w - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

test "sampled first-token logprob uses the last written handle, not argmax" {
    const logits = [_]f32{ 0, -1, -2, -3 };
    var tokens: [ring]u32 = @splat(0);
    tokens[2] = 3;
    const row = firstLogprobRow(&logits, &tokens, 3, 0);
    try std.testing.expectEqual(@as(u32, 3), row.token);
    try std.testing.expect(std.math.isFinite(row.logprob));
    const lse = @log(@as(f64, 1) + @exp(-@as(f64, 1)) + @exp(-@as(f64, 2)) + @exp(-@as(f64, 3)));
    try std.testing.expectApproxEqAbs(@as(f32, -3) - @as(f32, @floatCast(lse)), row.logprob, 1e-6);
}

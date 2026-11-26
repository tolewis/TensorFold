//! Inkling 2 Mini behind lanes.Backend: a cache slot a stream (48 visit slots), a prompt chunk a step, one forward
//! a round over every stream's window. A window wider than one row (drafts) backs its conv states up first and
//! records each row's conv inputs, so `keep` rolls a cut window back; a drafter reads the tapped rows (features).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const mm = @import("mini_model.zig");
const ops = @import("ops.zig");
const snap = @import("mini_snap.zig");
const mini_vision = @import("mini_vision.zig");

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes.
pub const prompt_rows = mm.prompt_rows;
/// Window rows a round takes over every stream.
pub const round_rows = mm.round_rows;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;
/// The widest window a stream verifies: the pending row and seven drafts (DSpark's block).
pub const verify_rows = 8;
/// The taps a drafter may name: DSpark's, in its order.
pub const taps_wanted = [_]u32{ 5, 23, 41, 48 };

/// The serving launches beside the forward: argmax, bf16 logits and sample.cu's keyed draws.
pub const Draws = struct {
    mods: [3]cuda.Module,
    argmax_fn: cuda.Function,
    to_bf16_fn: cuda.Function,
    grammar_mask_fn: cuda.Function, // sample.cu's tf_grammar_mask: a sampled row's disallowed bf16 logits to -inf
    draws: cuda.sample.Draws,

    pub fn load(d: *const cuda.Driver) !Draws {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const kk = cuda.kernels;
        var x: Draws = undefined;
        var loaded: usize = 0;
        errdefer for (x.mods[0..loaded]) |*m| m.unload();
        for ([_][]const u8{ kk.sample, kk.torch_argmax, kk.torch_pointwise }, 0..) |img, i| {
            x.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        x.draws = try cuda.sample.Draws.resolve(x.mods[0]);
        x.argmax_fn = try x.mods[1].function("tf_argmax_rows_i32_kernel");
        x.to_bf16_fn = try x.mods[2].function("tf_f32_to_bf16_kernel");
        x.grammar_mask_fn = try x.mods[0].function("tf_grammar_mask");
        return x;
    }

    pub fn deinit(x: *Draws) void {
        for (&x.mods) |*m| m.unload();
    }

    /// Each row's argmax over `v` fp32 logits, the first maximum on ties, into `out` (int32 a row).
    fn argmax(x: *const Draws, s: cuda.Stream, logits: u64, rows: usize, v: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, v, v }) |n| a.add(@as(i64, @intCast(n)));
        a.add(@as(i32, 1)); // dtype 1: fp32
        try cuda.launch.launch(x.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// Rows of bf16 logits (`v` apart) with their disallowed tokens at -inf (`words` u32 of bits a row).
    fn grammarMask(x: *const Draws, s: cuda.Stream, logits: u64, rows: usize, v: usize, bits: u64, words: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(v)));
        a.add(bits);
        a.add(@as(u32, @intCast(words)));
        try cuda.launch.launch(x.grammar_mask_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    fn toBf16(x: *const Draws, s: cuda.Stream, src: u64, dst: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(u64, n));
        const grid: u32 = @intCast(@max(1, @min((n + 255) / 256, 65535)));
        try cuda.launch.launch(x.to_bf16_fn, .{ .grid = .{ .x = grid }, .block = .{ .x = 256 } }, s, &a);
    }
};

const Lane = struct {
    slot: usize,
    len: u64 = 0, // positions [0, len) hold this stream's keys and its conv states are at len
    base: u64 = 0, // the length the last verify started from
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's fp32 logits (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
    rule_host: cuda.sample.Rule = undefined, // the same, on the host (the lazy head's race rows)
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    row0: usize = 0, // the last verify's first row in its forward
    cut_gen: ?u64 = null, // the wide verify whose hist and backup a cut window rolls back with (its forward's gen)
    held_at: u64 = 0, // the slot's own tapped rows: positions [held_at, held_at + held_rows)
    held_rows: usize = 0,
    chunk_at: u64 = 0, // the last prompt chunk's rows in m.taps, valid while m.gen == chunk_gen
    chunk_rows: usize = 0,
    chunk_gen: u64 = 0,
};

/// The positions a slot holds for a served window, in whole 1,024s, with room for a full window's last verify.
pub fn capacity(window: usize) !usize {
    if (window == 0) return error.BadContext;
    return (try std.math.add(usize, window, 1023 + verify_rows)) / 1024 * 1024;
}

/// A prompt and its reply fit `physical` positions.
pub fn fits(physical: usize, prompt: usize, reply: usize) bool {
    return prompt <= physical and reply <= physical - prompt;
}

pub const Mini = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *mm.Model,
    k: *const mm.Kernels,
    x: *const Draws,
    s: cuda.Stream,
    image_token: u32,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    logits16: cuda.DeviceBuffer, // a round's logits as bf16 for keyed draws
    tok: cuda.DeviceBuffer, // drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    slot_bufs: []cuda.DeviceBuffer, // a slot's last-row logits, then its rule: two a slot
    bits: cuda.DeviceBuffer, // structured output: a round's rows of allowed-token bits, bits_words apart
    bits_words: usize, // u32 a row's bits hold: the grammar's padded vocabulary (vocab_size), exactly
    guard: cuda.DeviceBuffer, // a sampled grammar round's masked argmax a row (0xFFFFFFFF: nothing usable)
    next: u64 = 0,
    held: ?cuda.DeviceBuffer = null, // every slot's own tapped rows, verify_rows a slot (once prepare_features)
    vision: ?*mini_vision.Images = null,

    fn vocab(x: *const Mini) usize {
        return x.m.c.unpadded_vocab;
    }

    /// The device bytes `init` allocates: bf16 logits, the token ring, metas, grammar bits, each slot's last row and
    /// rule (`padded`: the config's vocab_size, which the grammar's bits cover).
    pub fn openBytes(vocab_: usize, padded: usize, slots: usize) usize {
        return round_rows * vocab_ * 2 + (ring + round_rows) * 4 + round_rows * 16 + round_rows * bitsWords(padded) * 4 + round_rows * 4 + slots * (vocab_ * 4 + @sizeOf(cuda.sample.Rule));
    }

    fn bitsWords(padded: usize) usize {
        return (padded + 31) / 32;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *mm.Model, k: *const mm.Kernels, x: *const Draws, s: cuda.Stream, image_token: u32) !Mini {
        const v = m.c.unpadded_vocab;
        const free = try gpa.alloc(bool, m.slots);
        errdefer gpa.free(free);
        @memset(free, true);
        var logits16 = try cuda.DeviceBuffer.alloc(d, round_rows * v * 2);
        errdefer logits16.free();
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        var meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16);
        errdefer meta.free();
        const bw = bitsWords(m.c.vocab);
        if (bw * 32 < v) return error.InvalidGrammarMasks;
        var bits = try cuda.DeviceBuffer.alloc(d, round_rows * bw * 4);
        errdefer bits.free();
        var guard = try cuda.DeviceBuffer.alloc(d, round_rows * 4);
        errdefer guard.free();
        const bufs = try gpa.alloc(cuda.DeviceBuffer, 2 * m.slots);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (bufs, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(d, if (i % 2 == 0) v * 4 else @sizeOf(cuda.sample.Rule));
            made += 1;
        }
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .x = x, .s = s, .image_token = image_token, .free = free, .logits16 = logits16, .tok = tok, .meta = meta, .bits = bits, .bits_words = bw, .guard = guard, .slot_bufs = bufs };
    }

    pub fn deinit(x: *Mini) void {
        if (x.held) |*h| h.free();
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs) |*b| b.free();
        x.gpa.free(x.slot_bufs);
        x.gpa.free(x.free);
        x.logits16.free();
        x.tok.free();
        x.meta.free();
        x.bits.free();
        x.guard.free();
    }

    pub fn backend(x: *Mini) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true, .prepare_features = prepareFeatures, .features = features, .prefill_features = prefillFeatures };

    /// The round loop's facts: no draft head of its own; windows of up to verify_rows rows (a drafter's, or copies
    /// with TF_INKLING2_COPY_DRAFTS=1), streams sharing a forward.
    pub fn facts(x: *const Mini) lanes.Model {
        const copies = copyDrafts();
        return .{ .copy_drafts = copies, .exact_width = verify_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .plain_share = true, .batch_rows = round_rows, .max_streams = @intCast(@min(x.m.slots, round_rows)), .shared_costs = if (sharedCostsOn()) &shared_costs else &.{}, .window_costs = if (loneCostsOn()) &lone_costs else &.{}, .window_costs_lone = true };
    }

    /// A shared verify round's ms by total rows on one GB10 (TF_INKLING2_VERIFY_TRACE=1, 2026-11-23, greedy, short
    /// contexts, 2 to 4 streams). Each MoE row routes to its own experts, so a wide round costs far more than a plain
    /// one; without these the depth rule drafts every shared stream a full block. Lone streams keep their own rule.
    const shared_costs = [_]lanes.config.Cost{
        .{ .width = 1, .ms = 30.1 },  .{ .width = 2, .ms = 38.3 },  .{ .width = 3, .ms = 45.8 },  .{ .width = 4, .ms = 50.6 },
        .{ .width = 8, .ms = 60.5 },  .{ .width = 12, .ms = 72.0 }, .{ .width = 16, .ms = 76.0 }, .{ .width = 20, .ms = 81.0 },
        .{ .width = 24, .ms = 88.7 }, .{ .width = 32, .ms = 102.8 },
    };

    /// A lone stream's verify round ms by rows on the same GB10 (one stream: its conv backup, no graph past one row).
    const lone_costs = [_]lanes.config.Cost{
        .{ .width = 1, .ms = 30.1 }, .{ .width = 2, .ms = 36.1 }, .{ .width = 3, .ms = 39.9 }, .{ .width = 4, .ms = 44.9 },
        .{ .width = 5, .ms = 47.8 }, .{ .width = 6, .ms = 50.6 }, .{ .width = 8, .ms = 58.7 }, .{ .width = 9, .ms = 61.0 },
    };

    /// TF_INKLING2_LONE_COSTS=1: lone streams priced by lone_costs (default off: their prior rule drafts every round).
    fn loneCostsOn() bool {
        const v = std.c.getenv("TF_INKLING2_LONE_COSTS") orelse return false;
        return v[0] == '1';
    }

    /// TF_INKLING2_SHARED_COSTS=0: shared rounds priced without forward costs, as before.
    fn sharedCostsOn() bool {
        const v = std.c.getenv("TF_INKLING2_SHARED_COSTS") orelse return true;
        return v[0] != '0';
    }

    /// TF_INKLING2_VERIFY_TRACE=1: each verify round's rows, streams and wall ms (forward through the draws).
    fn verifyTrace() bool {
        const S = struct {
            var v: ?bool = null;
        };
        if (S.v == null) S.v = if (std.c.getenv("TF_INKLING2_VERIFY_TRACE")) |e| e[0] == '1' else false;
        return S.v.?;
    }

    fn clockNs() u64 {
        var t: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &t);
        return @as(u64, @intCast(t.sec)) * 1_000_000_000 + @as(u64, @intCast(t.nsec));
    }

    fn copyDrafts() bool {
        const v = std.mem.span(std.c.getenv("TF_INKLING2_COPY_DRAFTS") orelse return false);
        return std.mem.eql(u8, v, "1");
    }

    fn self(ptr: *anyopaque) *Mini {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(x: *Mini, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(x.d, dst, bytes, x.s);
    }

    /// A grammar's bits a row for `rows` rows: their stride, refused unless it covers the logits and fits the buffer.
    fn maskWords(x: *const Mini, masks: []const u32, rows: usize) !usize {
        if (rows == 0 or masks.len != rows * x.bits_words) return error.InvalidGrammarMasks;
        return x.bits_words;
    }

    /// Rows [row0, row0 + rows) of fp32 `logits` drawn into tok[at..], row r at `position0 + r` (argmax when greedy).
    /// `masks` (empty: none) holds each row's allowed tokens; they go to the bits from row `bits_row`. A greedy row takes the
    /// first maximum among its allowed tokens (0xFFFFFFFF when none has a usable score; read and verify refuse it), a
    /// sampled row draws with the rest at -inf. The logits are never written, so a prompt's last row stays as it was.
    fn drawRows(x: *Mini, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize, masks: []const u32, bits_row: usize) !void {
        const v = x.vocab();
        const src = logits + row0 * v * 4;
        var bits: u64 = 0;
        var words: usize = 0;
        if (masks.len != 0) {
            words = try x.maskWords(masks, rows);
            if (bits_row + rows > round_rows) return error.InvalidGrammarMasks;
            bits = x.bits.ptr + bits_row * x.bits_words * 4;
            try x.upload(bits, std.mem.sliceAsBytes(masks));
        }
        if (!l.sampled) {
            if (masks.len != 0) return ops.maskedArgmax(&x.k.ops, x.s, src, v, bits, words, rows, x.tok.ptr + at * 4);
            return x.x.argmax(x.s, src, rows, v, x.tok.ptr + at * 4);
        }
        try x.x.toBf16(x.s, src, x.logits16.ptr, rows * v);
        if (masks.len != 0) {
            // a row with no usable allowed score draws from one finite logit, then its token becomes the sentinel
            try x.x.grammarMask(x.s, x.logits16.ptr, rows, v, bits, words);
            try ops.grammarScrub(&x.k.ops, x.s, x.logits16.ptr, v, rows);
            try ops.maskedArgmax16(&x.k.ops, x.s, x.logits16.ptr, v, bits, words, rows, x.guard.ptr);
            try ops.grammarGuard(&x.k.ops, x.s, x.logits16.ptr, v, x.guard.ptr, 0, rows, false, bits, words);
        }
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.x.draws.draw(x.s, x.logits16.ptr, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
        if (masks.len != 0) try ops.grammarGuard(&x.k.ops, x.s, x.logits16.ptr, v, x.guard.ptr, x.tok.ptr + at * 4, rows, true, bits, words);
    }

    fn freeSlot(x: *const Mini) ?usize {
        return std.mem.indexOfScalar(bool, x.free, true);
    }

    /// Whether a request could start now: a free slot; one that never fits passes (its open refuses it).
    pub fn admits(x: *const Mini, prompt: usize, max_new: usize) bool {
        if (!fits(x.m.context, prompt, max_new)) return true;
        return x.freeSlot() != null;
    }

    /// A prompt pass begins: a slot and the stream's rule; images encode through the vision tower when --vision is on.
    fn opening(x: *Mini, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (s.images.len > 0 and x.vision == null) return error.Inkling2MiniImages;
        if (s.images.len == 0 and std.mem.indexOfScalar(u32, p, x.image_token) != null) return error.Inkling2MiniImages;
        if (!fits(x.m.context, p.len, @as(usize, s.max_new))) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            const slot = x.freeSlot() orelse {
                x.lanes.removeByPtr(got.key_ptr);
                return error.NoFreeSlot;
            };
            got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr };
            x.free[slot] = false;
        }
        const l = got.value_ptr;
        l.sampled = false;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
            l.rule_host = r;
        }
        s.cached = 0;
        l.fill = x.resumeAt(s, l, p.len);
        if (x.vision) |v| {
            if (s.images.len > 0) {
                try v.begin(&x.k.ops, x.s, l.slot, s.images, l.fill.?);
            } else {
                v.clear(l.slot);
            }
        }
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    /// The kept state's position, restored into the slot (0: none, not this prompt's, or another plan's).
    fn resumeAt(x: *Mini, s: *Stream, l: *const Lane, len: usize) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len) {
            s.reuse_failed = true;
            return 0;
        }
        snap.restore(x.m, x.s, l.slot, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    /// The prompt pass stands at `at` (a chunk start or a mark): copy the stream's state there.
    pub fn saveState(x: *Mini, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at) return error.NotAtMark; // a state is copied only where the pass stands
        return snap.save(x.gpa, x.m, x.s, l.slot, at);
    }

    pub fn restoreState(x: *Mini, owner: ?*anyopaque, k: *const snap.Saved) !void {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        try snap.restore(x.m, x.s, l.slot, k);
        l.len = k.at;
    }

    pub fn dropState(x: *Mini, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    /// One chunk from `l.fill`; true once the prompt's last row is in.
    fn chunk(x: *Mini, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        x.m.vision_slot = if (s.images.len > 0) l.slot else null;
        x.m.vision_p0 = at;
        defer x.m.vision_slot = null;
        try x.m.forward(x.k, x.s, &.{.{ .slot = l.slot, .p0 = at, .tokens = p[at..end] }}, true, end - at - 1, @intFromBool(done));
        if (done) try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(l.last, x.m.s.logits, x.vocab() * 4, x.s.handle), "cuMemcpyDtoDAsync");
        if (x.m.taps_on) {
            l.chunk_at = at;
            l.chunk_rows = end - at;
            l.chunk_gen = x.m.gen;
            if (done) try x.holdRows(l, end - at - 1, end - 1, 1); // the prompt's last row, for the first draft
        }
        s.fill_at.store(@intCast(end), .monotonic);
        l.fill = if (done) null else end;
        if (done) l.len = p.len;
        if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        return x.chunk(s, l);
    }

    fn handle(x: *Mini) usize {
        const h = x.next;
        x.next += 1;
        return @intCast(h % ring);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = x.next;
        try x.drawRows(l, l.last, 0, 1, position, x.handle(), 0, &.{}, 0);
        return h;
    }

    /// The first token among the grammar's allowed ones (`mask`: one row of bits).
    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        _ = try x.maskWords(mask, 1);
        const h = x.next;
        try x.drawRows(l, l.last, 0, 1, position, x.handle(), 0, mask, 0);
        return h;
    }

    fn value(x: *Mini, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        const t = try x.value(feed);
        if (position != l.len + 1) return error.PositionMismatch;
        x.m.lazy_ok = lazyRow(x, l, position, 0);
        defer x.m.lazy_ok = false;
        try x.m.forward(x.k, x.s, &.{.{ .slot = l.slot, .p0 = l.len, .tokens = &.{t} }}, false, 0, 1);
        if (x.m.taps_on) try x.holdRows(l, 0, l.len, 1);
        l.cut_gen = null;
        l.len += 1;
        const h = x.next;
        try x.drawRows(l, x.m.s.logits, 0, 1, position, x.handle(), 0, &.{}, 0);
        return h;
    }

    /// Head row `at`'s draw for the lazy head (the draw's position): false when its rule needs every logit.
    fn lazyRow(x: *Mini, l: *const Lane, position: u64, at: usize) bool {
        if (at >= ops.Lazy.max_rows) return false;
        if (!l.sampled) {
            x.m.lazy_rows[at] = .{ .mode = 0 };
            return true;
        }
        const r = l.rule_host;
        const whole = r.top_k == 0 or r.top_k >= x.vocab();
        const cut = r.top_p > 0 and r.top_p < 1;
        if (!whole or cut or !(r.inv_t > 0) or !std.math.isFinite(r.inv_t) or r.min_log > -24.0) return false;
        x.m.lazy_rows[at] = .{ .seed = r.seed, .inv_t = r.inv_t, .min_log = r.min_log, .pos = @intCast(position), .mode = 1 };
        return true;
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try x.s.synchronize();
        try x.tok.download(@intCast((h % ring) * 4), std.mem.sliceAsBytes(&out));
        if (out[0] >= x.vocab()) return error.NoAllowedToken; // a grammar row with no usable score
        return out[0];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        const t0: u64 = if (verifyTrace()) clockNs() else 0;
        const chains = try x.gpa.alloc(mm.Chain, windows.len);
        defer x.gpa.free(chains);
        var total: usize = 0;
        var wide = false;
        var masked = false;
        for (windows, chains) |w, *c| {
            if (w.held != 0) return error.NoHeldDrafts; // drafts reach this target as host tokens (Drafted)
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) _ = try x.maskWords(w.masks, w.rows());
            masked = masked or w.masks.len != 0;
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.rows() > verify_rows) return error.WindowTooWide;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, w.tokens);
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            c.* = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items };
            l.row0 = total;
            total += l.rows.items.len;
            wide = wide or l.rows.items.len > 1;
        }
        if (total > round_rows) return error.RoundTooWide;
        // a window that may be cut: its slot's conv states before it, and every row's conv inputs (no graph)
        if (wide) for (chains) |c| if (c.tokens.len > 1) try x.m.backupSlot(x.k, x.s, c.slot);
        x.m.hist_on = wide;
        defer x.m.hist_on = false;
        // a grammar's rows need every logit: its mask may refuse every candidate the lazy head keeps
        x.m.lazy_ok = !masked and total <= ops.Lazy.max_rows and for (windows) |w| {
            const l = x.lanes.getPtr(w.stream).?;
            var ok = true;
            for (0..l.rows.items.len) |r| ok = ok and lazyRow(x, l, w.positions[0] + r, l.row0 + r);
            if (!ok) break false;
        } else true;
        defer x.m.lazy_ok = false;
        try x.m.forward(x.k, x.s, chains, false, 0, total);
        for (windows) |w| {
            const l = x.lanes.getPtr(w.stream).?;
            l.cut_gen = if (l.rows.items.len > 1) x.m.gen else null;
            if (x.m.taps_on) try x.holdRows(l, l.row0, l.base, l.rows.items.len);
        }
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
            try x.drawRows(l, x.m.s.logits, at, l.rows.items.len, w.positions[0], ring + at, i, w.masks, at);
            at += l.rows.items.len;
        }
        var got: [round_rows]u32 = undefined;
        try x.s.synchronize();
        if (verifyTrace()) std.log.info("verify rows={d} streams={d} wide={} ms={d:.3}", .{ total, windows.len, wide, @as(f64, @floatFromInt(clockNs() - t0)) / 1e6 });
        try x.tok.download(ring * 4, std.mem.sliceAsBytes(got[0..total]));
        if (masked) for (got[0..total]) |t| if (t >= x.vocab()) return error.NoAllowedToken; // a grammar row with no usable score
        at = 0;
        for (windows, out) |w, o| {
            const l = x.lanes.getPtr(w.stream).?;
            const n = l.rows.items.len;
            @memcpy(o.sampled[0..n], got[at..][0..n]);
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..]);
            l.len = l.base + n;
            at += n;
        }
    }

    /// Each window keeps a prefix of its rows: a cut one's length and conv states roll back to that prefix (the
    /// backup and the row inputs its verify recorded); the caches' rows past it are written over by later rows.
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            const n = l.rows.items.len;
            if (path.len == 0 or path.len > n) return error.Inkling2KeepPath;
            for (path, 0..) |r, i| if (r != i) return error.ChainsOnly;
            if (path.len == n) continue;
            if (l.cut_gen != x.m.gen) return error.Inkling2MiniNoRollback; // another forward ran since its verify
            try x.m.rewindSlot(x.k, x.s, l.slot, l.row0, path.len);
            l.len = l.base + path.len;
            if (l.held_at == l.base) l.held_rows = @min(l.held_rows, path.len);
        }
    }

    /// Rows [row, row + n) of the last forward's taps as the slot's own rows at positions [at, at + n).
    fn holdRows(x: *Mini, l: *Lane, row: usize, at: u64, n: usize) !void {
        if (n > verify_rows) return error.WindowTooWide;
        const rb = x.m.tapRowBytes();
        const h = x.held orelse return error.UnpreparedInkling2Taps;
        try x.d.check(x.d.api.cuMemcpyDtoDAsync_v2(h.ptr + l.slot * verify_rows * rb, x.m.taps + row * rb, n * rb, x.s.handle), "cuMemcpyDtoDAsync");
        l.held_at = at;
        l.held_rows = n;
    }

    /// DSpark's taps only: the three visit inputs and the final normed row.
    fn validTaps(taps: []const u32) !void {
        if (!std.mem.eql(u32, taps, &taps_wanted)) return error.UnsupportedInkling2Taps;
    }

    /// From now on every forward taps its rows (before any stream opens: captured graphs are dropped).
    fn prepareFeatures(ptr: *anyopaque, taps: []const u32) anyerror!void {
        const x = self(ptr);
        try validTaps(taps);
        if (x.lanes.count() != 0) return error.TapsAfterStreams;
        if (x.held == null) x.held = try cuda.DeviceBuffer.alloc(x.d, x.m.slots * verify_rows * x.m.tapRowBytes());
        x.m.dropGraphs();
        x.m.taps_on = true;
    }

    /// Cache rows [start, start + count) of the stream's taps: its own held rows, or its last prompt chunk's while
    /// no other forward has run since; anything else is refused by name.
    fn features(ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const x = self(ptr);
        try validTaps(taps);
        if (!x.m.taps_on) return error.UnpreparedInkling2Taps;
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        const rb = x.m.tapRowBytes();
        const f: be.Features = .{ .buffer = 0, .rows = count, .row_bytes = @intCast(rb), .space = .device, .dtype = .bf16 };
        if (count == 0) return f;
        if (start >= l.held_at and start + count <= l.held_at + l.held_rows) {
            var out = f;
            out.buffer = x.held.?.ptr + l.slot * verify_rows * rb;
            out.offset = (start - l.held_at) * rb;
            return out;
        }
        if (l.chunk_gen == x.m.gen and start >= l.chunk_at and start + count <= l.chunk_at + l.chunk_rows) {
            var out = f;
            out.buffer = x.m.taps;
            out.offset = (start - l.chunk_at) * rb;
            return out;
        }
        return error.Inkling2TapsNotHeld;
    }

    /// The prompt pass whole, `sink` taking each chunk's tapped rows as the chunk is made.
    fn prefillFeatures(ptr: *anyopaque, s: *Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
        const x = self(ptr);
        try validTaps(taps);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (true) {
            const at = l.fill.?;
            const done = try x.chunk(s, l);
            const end: u64 = if (done) s.prompt().len else l.fill.?;
            try sink.take(sink.ptr, s, try features(ptr, s, taps, at, @intCast(end - at)), at);
            if (done) break;
        }
    }

    /// No draft head: absorbing is a no-op, and a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    fn dropLane(x: *Mini, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            var l = entry.value;
            if (x.vision) |v| v.clear(l.slot);
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

test "a slot holds the window in whole 1,024s; a reply past it does not fit" {
    try std.testing.expectEqual(@as(usize, 33792), try capacity(32768));
    try std.testing.expectEqual(@as(usize, 1024), try capacity(1000));
    try std.testing.expectError(error.BadContext, capacity(0));
    try std.testing.expect(fits(32768, 30000, 2768));
    try std.testing.expect(!fits(32768, 30000, 2769));
    try std.testing.expect(!fits(32768, 32769, 0));
}

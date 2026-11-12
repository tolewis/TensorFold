//! The XPU 27B on the lane host: a slot per stream, shared rounds; Drafted drafts from its taps, lone streams on trees.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const be = lanes.backend;
const TREE_MIN = @import("xpu_dflash2.zig").TREE_MIN;
const Drafter = xq.drafter.Drafter;
const Allocator = std.mem.Allocator;

const ring = 64;
const TAP_ROW = 5 * xq.D * 2; // bf16 [5 D]

/// A kept prompt state (prompt cache): the slot's states after `at` tokens and its DFlash2 context.
pub const Kept = struct { snap: xq.Runner.Snap, ctx: ?*anyopaque = null, range: [2]u32 = .{ 0, 0 } };

/// XPU_ROUND_STATS=N (rank 0 only): host time per round, one line every N rounds. Off: `on` is false, one branch per call.
/// A round opens at its verify and closes at the next verify, so its time runs to the last backend call it made.
pub const RoundStats = struct {
    on: bool = false,
    every: u32 = 0,
    open: bool = false,
    t_start: f64 = 0, // the round's verify entry (seconds, monotonic)
    t_last: f64 = 0, // the exit of the round's last backend call
    t_verify: f64 = 0, // the round's verify call
    t_draft: f64 = 0, // the round's drafter calls, summed
    streams: u32 = 0, // the round's windows
    rows: u32 = 0, // the round's rows, pending tokens included
    win_acc: [16]u32 = @splat(0), // each window's accepted drafts: rows - 1 until a keep gives its kept path
    rounds: u32 = 0, // closed rounds in this window
    sum_streams: u64 = 0,
    sum_rows: u64 = 0,
    sum_acc: u64 = 0,
    sum_round: f64 = 0,
    sum_verify: f64 = 0,
    sum_draft: f64 = 0,
};

/// A slot's prompt pass: the next prompt row, and the rows the drafter still skips.
pub const Fill = struct { on: bool = false, at: usize = 0, skip: u32 = 0 };

pub const Lanes = struct {
    gpa: Allocator,
    runner: *xq.Runner,
    drafter: ?*Drafter,
    pick: *const fn (ctx: *anyopaque, row: []const f32, position: u64, s: api.Sampling) anyerror!u32,
    pick_ctx: *anyopaque,
    slots: u32,
    block: u32 = 8,
    nodes: u32 = 15,
    owners: [xq.MAX_SLOTS]?*lanes.Stream = @splat(null),
    first_logits: [xq.MAX_SLOTS]?[]f32 = @splat(null), // a sampled or structured stream's prompt-row logits, for `first`
    mask_rows: [16]u32 = undefined, // a verify's greedy structured rows (argmaxMasked on the device)
    mask_bits: ?[]u32 = null, // and their grammar bits, gathered
    masked_vals: ?[]f64 = null, // a structured sampled row's allowed candidates
    masked_ids: ?[]u64 = null,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    ids: [16]u32 = undefined, // the last verify's rows, window after window
    widths: [16]u32 = undefined, // and each window's rows
    count: usize = 0,
    wslots: [16]u32 = undefined, // and each window's slot
    wfirst: [16]u32 = undefined, // and its first row
    absorbed: [16]bool = @splat(false), // the window's kept rows already in its slot's drafter context
    chain: [xq.MAX_SLOTS][16]u32 = undefined, // each slot's DFlash2 drafts for its next window (a tree)
    chain_par: [xq.MAX_SLOTS][16]i32 = undefined, // and each one's parent (−1: the pending token)
    chain_n: [xq.MAX_SLOTS]u32 = @splat(0),
    tapped: usize = 0, // the last verify's windows whose taps acts.taps still holds (features)
    tail: ?Tail = null, // the prompt chunk whose taps acts.taps still holds (features)
    sink: ?struct { s: *lanes.Stream, take: be.FeatureSink } = null, // a drafting prompt pass hands chunks here
    direct: bool = false, // the lone driver's prompt pass: taps straight into the slot's drafter context
    wrap: ?*lanes.drafted.Drafted = null, // the drafter in front of this backend for lane rounds
    prompt_skip: u32 = 0, // a prefill's first rows the drafter never reads (further than WIN from the prompt's end)
    fill: [xq.MAX_SLOTS]Fill = @splat(.{}), // each slot's prompt pass in progress (a sliced pass stands between calls)
    slice: bool = false, // XPU_PROMPT_SLICE: prompt passes a chunk a call (prefill_step), decode rounds between
    costs: [16]lanes.config.Cost = undefined, // a window's ms by rows (verify and keep), timed by measure()
    timed: usize = 0,
    rs: RoundStats = .{}, // XPU_ROUND_STATS, set by the opener

    pub const max_slots = xq.MAX_SLOTS;
    pub const tap_row = TAP_ROW;
    pub const draft_step_ms = 0.6; // a DFlash2 draft pass's ms, as the round loop prices it
    pub const tap_layers = blk: {
        var t: [xq.TAP_LAYERS.len]u32 = undefined;
        for (xq.TAP_LAYERS, &t) |l, *x| x.* = l;
        break :blk t;
    };
    /// Rows [start, start + rows) of a prompt pass in `slot`, their taps in acts.taps.
    pub const Tail = struct { slot: u32, start: u64, rows: u32 };

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn backend(b: *Lanes) be.Backend {
        return .{ .ptr = b, .vtable = vtableOf(b.slice) };
    }

    /// The backend's vtable: whole prompt passes, or (XPU_PROMPT_SLICE) a chunk a call; static (xpu_slice_vtable.zig).
    pub fn vtableOf(slice: bool) *const be.Backend.VTable {
        return Vt.of(slice);
    }

    const Vt = @import("xpu_slice_vtable.zig").Vtables(.{ .prefill = prefillFn, .prefill_features = prefillFeaturesFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .tree = treeFn, .release = releaseFn }, prefillStepFn);

    /// What the round loop reads: chains of up to 16 rows, all streams' windows in one shared forward, DFlash2 drafts
    pub fn facts(b: *const Lanes) lanes.Model {
        const head = b.drafter != null;
        return .{ .exact_width = 16, .mtp = head, .speculate = head, .speculate_early = false, .drafts = if (head) b.block - 1 else 0, .mtp_step_ms = if (head) draft_step_ms else 0, .draft_streams = head, .window_costs = b.costs[0..b.timed], .hidden_rows = true, .streams_exact = true, .batch_rows = 16, .max_streams = b.slots, .lone_copies = false };
    }

    /// Each window width's cost on a scratch context in the last slot, for the round loop's depth rule
    pub fn measure(b: *Lanes) !void {
        const r = b.runner;
        const slot = b.slots - 1;
        try r.select(slot);
        try r.reset();
        var ids: [128]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % xq.VOCAB);
        try r.prompt(&ids, false);
        const one = [_]u32{0};
        const time = struct {
            fn f(rr: *xq.Runner, sl: u32, row: []const u32, k: []const u32) !f64 {
                const t0 = xq.now();
                try rr.verifyMulti(&.{.{ .slot = sl, .ids = row }});
                try rr.keepPaths(&.{k});
                try rr.d.sync();
                return xq.now() - t0;
            }
        }.f;
        for (0..3) |_| _ = try time(r, slot, ids[0..16], &one); // the widest first, while the clocks ramp up
        for (&b.costs, 1..) |*c, w| {
            var best: f64 = 1e9;
            for (0..3) |_| best = @min(best, try time(r, slot, ids[0..w], &one));
            c.* = .{ .width = @intCast(w), .ms = best * 1e3 };
        }
        b.timed = b.costs.len;
        b.tapped = 0;
        b.tail = null;
        try r.reset();
    }

    /// Every forward's taps reach the drafter context of the slot it ran (with a drafter attached).
    pub fn initDrafts(b: *Lanes) void {
        b.runner.on_taps = .{ .ctx = b, .f = onTaps };
    }

    fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) anyerror!void {
        const b = self(ctx);
        if (b.sink) |k| { // a prompt chunk, its first row at the runner's position (prompt() moves it after)
            b.tail = .{ .slot = b.runner.slot, .start = b.runner.pos, .rows = n };
            return k.take.take(k.take.ptr, k.s, rowsAt(taps, 0, n), b.runner.pos);
        }
        if (!b.direct) return;
        const d = b.drafter.?;
        try d.use(b.runner.slot);
        const k = @min(n, b.prompt_skip);
        b.prompt_skip -= k;
        d.skip(k);
        if (k < n) try d.addTaps(xq.off(taps, @as(usize, k) * TAP_ROW), n - k);
    }

    /// `n` tapped rows from row `row` of `taps`, as a drafter reads them.
    fn rowsAt(taps: *anyopaque, row: u64, n: u32) be.Features {
        return .{ .buffer = @intFromPtr(taps), .offset = row * TAP_ROW, .rows = n, .row_bytes = TAP_ROW, .space = .device, .dtype = .bf16 };
    }

    /// A stream's rows in acts.taps: the last verify's window (a drafter absorbs its kept rows), else its prompt chunk.
    fn featuresFn(ptr: *anyopaque, s: *lanes.Stream, _: []const u32, start: u64, count: u32) anyerror!be.Features {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (std.mem.indexOfScalar(u32, b.wslots[0..b.tapped], slot)) |i| {
            if (count > b.widths[i]) return error.FeaturesGone;
            return rowsAt(b.runner.acts.taps, b.wfirst[i], count);
        }
        const t = b.tail orelse return error.FeaturesGone;
        if (t.slot != slot or start < t.start or start + count > t.start + t.rows) return error.FeaturesGone;
        return rowsAt(b.runner.acts.taps, start - t.start, count);
    }

    /// The prompt chunk in acts.taps if `slot` ran it last (the drafter takes the prompt's last row with its chunk).
    pub fn promptTail(b: *const Lanes, slot: u32) ?Tail {
        const t = b.tail orelse return null;
        return if (t.slot == slot) t else null;
    }

    /// Keeps the root path `paths[i]` of the last verify's window i.
    fn keepRows(b: *Lanes, paths: []const []const u32) !void {
        try b.runner.keepPaths(paths);
    }

    /// A verify whose rows all stayed: the core calls keep only to drop rows, so the next call keeps them all first.
    fn settle(b: *Lanes) !void {
        if (b.count == 0) return;
        const n = b.count;
        b.count = 0;
        const all = [16]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        var paths: [16][]const u32 = undefined;
        for (paths[0..n], b.widths[0..n]) |*p, w| p.* = all[0..w];
        try b.keepRows(paths[0..n]);
    }

    pub fn slotOf(b: *const Lanes, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn push(b: *Lanes, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn value(b: *const Lanes, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| if (h < b.next and b.next - h <= ring) b.drawn[h % ring] else error.NoSuchToken,
        };
    }

    /// `slot`'s next greedy token (the runner keeps it per slot).
    fn lastOf(b: *const Lanes, slot: u32) u32 {
        const r = b.runner;
        return if (r.slot == slot) r.last else r.seqs[slot].last;
    }

    /// A stream's prompt into its slot (a free one, slot 0 last); its last row's logits are kept when it samples.
    fn prefillSlot(b: *Lanes, s: *lanes.Stream, slot: u32) !void {
        try b.beginPass(s, slot);
        defer b.fill[slot].on = false;
        while (!try b.passChunk(s, slot)) {}
    }

    /// A prompt pass's start in `slot`: the reply fitted, the slot and its drafter context reset, a kept prefix restored.
    fn beginPass(b: *Lanes, s: *lanes.Stream, slot: u32) !void {
        const ids = s.prompt();
        // the reply is fitted to the context less the drafter's 17 slots; only a prompt with no room is refused
        s.max_new = @import("xpu_admit.zig").fitReply(ids.len, s.max_new, b.runner.context, 17) orelse return error.PromptTooLong;
        const r = b.runner;
        try r.select(slot);
        try r.reset();
        b.tapped = 0;
        b.tail = null;
        if (b.drafter) |d| {
            try d.use(slot);
            d.reset();
            r.taps_on = true;
        }
        defer r.taps_on = false;
        s.cached = 0;
        var at: usize = 0;
        if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const k: *const Kept = @ptrCast(@alignCast(saved));
            if (k.snap.at < ids.len) {
                if (b.restoreKept(slot, k)) {
                    at = k.snap.at;
                    s.cached = k.snap.at;
                } else |_| {
                    s.reuse_failed = true;
                    try r.select(slot);
                    try r.reset();
                    if (b.drafter) |d| {
                        try d.use(slot);
                        d.reset();
                    }
                }
            } else s.reuse_failed = true;
        }
        b.fill[slot] = .{ .on = true, .at = at, .skip = if (b.drafter != null) @intCast((ids.len -| xq.drafter.WIN) -| at) else 0 };
        s.fill_at.store(@intCast(at), .monotonic);
    }

    /// The pass's next chunk in `slot` (cut where the request's chunks and the cache's marks are, as a fresh pass cuts
    /// them); true once the prompt is in, its last row's logits kept when the stream samples.
    fn passChunk(b: *Lanes, s: *lanes.Stream, slot: u32) !bool {
        const ids = s.prompt();
        const r = b.runner;
        const f = &b.fill[slot];
        if (f.at >= ids.len) return true;
        if (s.isCancelled()) return error.Cancelled;
        if (r.slot != slot) try r.select(slot); // a decode round between slices made another slot current
        r.taps_on = b.drafter != null;
        b.prompt_skip = f.skip;
        defer {
            r.taps_on = false;
            f.skip = b.prompt_skip;
            b.prompt_skip = 0;
        }
        const at = f.at;
        var end = xq.chunkEnd(at, ids.len);
        for (s.chunks) |cut| if (cut > at and cut < end) {
            end = cut;
        };
        for (s.reuse.marks) |mark| if (mark > at and mark < end) {
            end = mark;
        };
        try r.prompt(ids[at..end], end == ids.len);
        f.at = end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(end));
        if (end < ids.len) return false;
        if (s.sampling != null or s.grammar != null) {
            const row = try r.logits();
            const keep = b.first_logits[slot] orelse try b.gpa.alloc(f32, row.len);
            @memcpy(keep, row);
            b.first_logits[slot] = keep;
        }
        return true;
    }

    /// The bytes keepSave takes at `at` tokens.
    pub fn keepBytes(b: *const Lanes, at: u32) u64 {
        // XPU_KEPT_HOST=1: the drafter context's copy is host memory, not charged to the card's budget
        return b.runner.snapBytes(at) + if (b.drafter != null and !(xq.kv_pool and xq.kept_host)) Drafter.ctxBytes() else 0;
    }

    /// `s`'s slot standing at `at` prompt tokens, copied (its DFlash2 context too).
    pub fn keepSave(b: *Lanes, s: *const lanes.Stream, at: u32) !*Kept {
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const k = try b.gpa.create(Kept);
        errdefer b.gpa.destroy(k);
        k.* = .{ .snap = try b.runner.saveSlot(slot, at) };
        errdefer b.runner.dropSnap(k.snap);
        if (b.drafter) |d| {
            const buf = if (xq.kv_pool and xq.kept_host) try b.runner.d.hostAlloc(Drafter.ctxBytes()) else try b.runner.d.alloc_(Drafter.ctxBytes());
            errdefer xq.freeAfterQueue(&b.runner.d, buf); // saveCtx's copies may be queued
            k.range = try d.saveCtx(slot, buf);
            k.ctx = buf;
        }
        return k;
    }

    pub fn keepDrop(b: *Lanes, k: *Kept) void {
        b.runner.dropSnap(k.snap);
        if (k.ctx) |x| {
            if (b.drafter) |d| d.dropSaved(x); // XPU_DRAFT_TP: the follower drops its copy too
            xq.freeAfterQueue(&b.runner.d, x);
        }
        b.gpa.destroy(k);
    }

    /// Parks the kept state's K and V. The drafter context copy stays.
    pub fn keepPark(b: *Lanes, k: *Kept) !void {
        try b.runner.parkSnap(&k.snap);
    }

    /// Bytes of one pool page on this card: K and V of `PAGE` tokens in every attention layer.
    pub fn keepPageBytes(b: *const Lanes) u64 {
        return @as(u64, @intCast(b.runner.kvBytesPerToken())) * xq.PAGE;
    }

    /// Host bytes a park of `saved` takes, summed over the ranks: each rank's K/V and GDN copy, plus rank 0's drafter context.
    /// Zero when the state is shorter than `XPU_PARK_MIN` or its position is not a whole page.
    pub fn parkTierBytes(b: *const Lanes, saved: *anyopaque) u64 {
        const k: *const Kept = @ptrCast(@alignCast(saved));
        if (k.snap.park != null) return 0;
        if (k.snap.at < xq.park_min or k.snap.at % xq.PAGE != 0) return 0;
        const kv: u64 = @as(u64, k.snap.at) * @as(u64, b.runner.kvBytesPerToken());
        const gdn: u64 = @intCast(k.snap.bytes);
        const ctx: u64 = if (k.ctx != null) Drafter.ctxBytes() else 0;
        return (kv + gdn) * @as(u64, xq.TP) + ctx;
    }

    /// Bytes the kept states hold together: each page with a kept-state reference, once.
    pub fn keepHeld(b: *const Lanes) u64 {
        return @as(u64, b.runner.pages.kept_pages) * b.keepPageBytes();
    }

    /// Pages of `owner`'s slot table from entry 0 to ceil(at / PAGE) that no kept state holds yet, plus `keepBytes(0)`.
    pub fn keepAdds(b: *const Lanes, owner: ?*anyopaque, at: u32) u64 {
        const stream: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return b.keepBytes(at)));
        const slot = b.slotOf(stream) orelse return b.keepBytes(at);
        const r = b.runner;
        if (r.pool == 0) return b.keepBytes(at);
        const entries: u32 = (at + xq.PAGE - 1) / xq.PAGE;
        const have = r.held[slot];
        const up: u32 = @min(entries, have);
        const table = r.pt_host[slot];
        const end: usize = @min(@as(usize, up), table.len);
        var n: u64 = 0;
        for (table[0..end]) |id| n += @intFromBool(r.pages.keptRefs(id) == 0);
        n += @as(u64, up) - @as(u64, end);
        if (entries > up) n += @as(u64, entries - up);
        return n * b.keepPageBytes() + b.keepBytes(0);
    }

    /// Bytes `held` falls by when this state is dropped. A parked state holds no pool page, so it frees nothing.
    pub fn keepFrees(b: *const Lanes, saved: *anyopaque) u64 {
        const k: *const Kept = @ptrCast(@alignCast(saved));
        if (k.snap.park != null) return 0;
        var n: u64 = 0;
        for (k.snap.pages) |id| n += @intFromBool(b.runner.pages.keptRefs(id) == 1);
        return n * b.keepPageBytes();
    }

    /// Bytes a drop gives back to the streams. A parked state holds no pool page, so it returns nothing.
    pub fn keepReturns(b: *const Lanes, saved: *anyopaque) u64 {
        const k: *const Kept = @ptrCast(@alignCast(saved));
        if (k.snap.park != null) return 0;
        var n: u64 = 0;
        for (k.snap.pages) |id| n += @intFromBool(b.runner.pages.sole(id));
        return n * b.keepPageBytes();
    }

    /// Pages for the status line: kept states' pages, the pool, and pages free.
    pub fn keepPageNote(b: *const Lanes) api.prompt_cache.PageNote {
        const p = &b.runner.pages;
        return .{ .held = p.kept_pages, .pool = p.count, .free = p.spare() };
    }

    fn restoreKept(b: *Lanes, slot: u32, k: *const Kept) !void {
        try b.runner.restoreSlot(slot, k.snap);
        if (b.drafter) |d| try d.restoreCtx(slot, k.ctx orelse return error.NoContext, k.range);
    }

    fn freeSlot(b: *const Lanes) ?u32 {
        var k: u32 = b.slots;
        while (k > 0) {
            k -= 1;
            if (b.owners[k] == null) return k;
        }
        return null;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        try b.settle();
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        try b.prefillSlot(s, slot);
    }

    /// A drafting stream's prompt pass (Drafted in front): each chunk's taps to `sink` while acts.taps holds them.
    fn prefillFeaturesFn(ptr: *anyopaque, s: *lanes.Stream, _: []const u32, sink: be.FeatureSink) anyerror!void {
        const b = self(ptr);
        b.sink = .{ .s = s, .take = sink };
        defer b.sink = null;
        return prefillFn(ptr, s);
    }

    /// XPU_PROMPT_SLICE: one chunk of the prompt pass a call, false while chunks remain; the lane host runs decode
    /// rounds of the other streams between calls. The chunks are the whole pass's (chunkEnd from the pass's start,
    /// the request's chunks and the cache's marks), so the prompt's bits are the same.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        _ = max_rows;
        const b = self(ptr);
        try b.settle();
        const held = b.slotOf(s);
        const slot = if (held != null and b.fill[held.?].on) held.? else blk: {
            const sl = held orelse (b.freeSlot() orelse return error.NoFreeSlot);
            b.owners[sl] = s;
            errdefer b.owners[sl] = null;
            try b.beginPass(s, sl);
            break :blk sl;
        };
        errdefer {
            b.fill[slot].on = false;
            b.owners[slot] = null;
        }
        if (!try b.passChunk(s, slot)) return false;
        b.fill[slot].on = false;
        return true;
    }

    fn draw(b: *Lanes, s: *const lanes.Stream, slot: u32, row: ?[]const f32, position: u64) !u32 {
        const sp = s.sampling orelse return b.lastOf(slot);
        return b.pick(b.pick_ctx, row orelse return error.NoLogits, position, sp);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        return b.push(try b.draw(s, slot, b.first_logits[slot], position));
    }

    /// Structured output: the stream's draw among the tokens `mask` allows, greedy (first argmax) or sampled (keyed)
    fn drawMasked(b: *Lanes, s: *const lanes.Stream, row: []const f32, position: u64, mask: []const u32) !u32 {
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const vals = b.masked_vals orelse try b.gpa.alloc(f64, row.len);
            b.masked_vals = vals;
            const ids = b.masked_ids orelse try b.gpa.alloc(u64, row.len);
            b.masked_ids = ids;
            const k: usize = if (sp.top_k == 0) row.len else sp.top_k;
            var n: usize = 0;
            for (row, 0..) |v32, i| {
                if (!lanes.grammar.allows(mask, i)) continue;
                const v: f64 = v32;
                if (!std.math.isFinite(v)) return error.NonfiniteLogits;
                if (n == k and !(v > vals[k - 1])) continue; // equal values keep the lower id
                var j = if (n < k) n else k - 1;
                if (n < k) n += 1;
                if (k < row.len) while (j > 0 and v > vals[j - 1]) : (j -= 1) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                };
                vals[j] = v;
                ids[j] = i;
            }
            if (n == 0) return error.NoAllowedToken;
            return @intCast(try lanes.sampling.choose(b.gpa, vals[0..n], ids[0..n], position, sp));
        };
        var best: ?usize = null;
        for (row, 0..) |v, id| if (lanes.grammar.allows(mask, id) and (best == null or v > row[best.?])) {
            best = id;
        };
        return @intCast(best orelse return error.NoAllowedToken);
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        return b.push(try b.drawMasked(s, b.first_logits[slot] orelse return error.NoLogits, position, mask));
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        try b.settle();
        const r = b.runner;
        try r.select(slot);
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        b.tapped = 0;
        b.tail = null;
        try r.step(try b.value(feed));
        return b.push(try b.draw(s, slot, if (s.sampling != null) try r.logits() else null, position));
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).value(.{ .handle = handle });
    }

    /// Every stream's window (pending token, then its copied drafts as a chain) in one forward, drawn with its sampling
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > 16 or windows.len != out.len) return error.BadWindows;
        if (b.rs.on) try b.rsOpen();
        defer {
            if (b.rs.on) b.rsVerified(windows);
        }
        try b.settle();
        var wins: [16]xq.Runner.Win = undefined;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.held > 0) return error.HeldDrafts;
            const rows = w.rows();
            if (total + rows > 16) return error.WindowTooWide;
            const ids = b.ids[total..][0..rows];
            ids[0] = w.pending;
            @memcpy(ids[1..], w.tokens);
            wins[i] = .{ .slot = b.slotOf(w.stream) orelse return error.UnknownStream, .ids = ids, .parents = w.parents };
            b.wslots[i] = wins[i].slot;
            b.wfirst[i] = @intCast(total);
            b.widths[i] = @intCast(rows);
            total += rows;
        }
        const r = b.runner;
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        b.tapped = 0;
        b.tail = null;
        try r.verifyMulti(wins[0..windows.len]);
        b.tapped = windows.len;
        // structured greedy rows: their tokens among their grammars' on the device, all in one pass
        var nm: usize = 0;
        var mwords: usize = 0;
        for (windows, 0..) |w, i| if (w.masks.len > 0 and w.stream.sampling == null) {
            const words = w.masks.len / b.widths[i];
            if (mwords != 0 and words != mwords) return error.BadMasks;
            mwords = words;
            const bits = b.mask_bits orelse try b.gpa.alloc(u32, 16 * xq.MASK_WORDS);
            b.mask_bits = bits;
            if (words > xq.MASK_WORDS) return error.BadMasks;
            for (0..b.widths[i]) |j| {
                b.mask_rows[nm] = b.wfirst[i] + @as(u32, @intCast(j));
                @memcpy(bits[nm * words ..][0..words], w.masks[j * words ..][0..words]);
                nm += 1;
            }
        };
        var mtok: [16]u32 = undefined;
        if (nm > 0) try r.argmaxMasked(b.mask_rows[0..nm], b.mask_bits.?[0 .. nm * mwords], mwords, &mtok);
        var mi: usize = 0;
        var row: u32 = 0;
        for (windows, out, 0..) |w, o, i| {
            const words = if (w.masks.len > 0) w.masks.len / b.widths[i] else 0;
            for (0..b.widths[i]) |j| {
                if (words > 0 and w.stream.sampling == null) {
                    o.sampled[j] = mtok[mi];
                    mi += 1;
                    row += 1;
                    continue;
                }
                if (words > 0) { // structured output, sampled: the row's grammar bits on the host
                    o.sampled[j] = try b.drawMasked(w.stream, try r.rowLogits(row), w.positions[j], w.masks[j * words ..][0..words]);
                    row += 1;
                    continue;
                }
                const logits = if (w.stream.sampling != null) try r.rowLogits(row) else null;
                o.sampled[j] = if (w.stream.sampling) |sp| try b.pick(b.pick_ctx, logits.?, w.positions[j], sp) else r.block[row];
                row += 1;
            }
            @memcpy(o.drafts, w.tokens);
        }
        b.count = windows.len;
    }

    /// Each window keeps a root path (a prefix of its chain, or a path down its tree).
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != b.count or paths.len != b.count) return error.NothingToKeep;
        for (paths) |p| if (p.len == 0) return error.NothingToKeep;
        if (b.rs.on) b.rsKept(paths);
        b.count = 0;
        try b.keepRows(paths);
    }

    /// Closes the open round into the window and opens a new one at its verify.
    fn rsOpen(b: *Lanes) !void {
        try b.rsClose();
        const rs = &b.rs;
        rs.open = true;
        rs.t_start = xq.now();
        rs.t_last = rs.t_start;
        rs.t_verify = 0;
        rs.t_draft = 0;
    }

    /// The verify's end: its time, its windows and rows, and each window's drafts (kept until a keep says otherwise).
    fn rsVerified(b: *Lanes, windows: []const be.Window) void {
        const rs = &b.rs;
        if (!rs.open) return;
        const t = xq.now();
        rs.t_verify = t - rs.t_start;
        rs.t_last = t;
        rs.streams = @intCast(windows.len);
        rs.rows = 0;
        for (windows, 0..) |w, i| {
            const n: u32 = @intCast(w.rows());
            rs.rows += n;
            rs.win_acc[i] = n - 1;
        }
    }

    /// A keep gives each window's accepted drafts: its kept path less the pending row.
    fn rsKept(b: *Lanes, paths: []const []const u32) void {
        const rs = &b.rs;
        if (!rs.open) return;
        for (paths, 0..) |p, i| rs.win_acc[i] = @intCast(p.len - 1);
        rs.t_last = xq.now();
    }

    fn rsDrafted(b: *Lanes, t0: f64) void {
        const rs = &b.rs;
        const t = xq.now();
        if (rs.open) rs.t_draft += t - t0;
        rs.t_last = t;
    }

    /// Adds the open round to the window. At `every` rounds: one line on stderr, then the XPU_PROFILE report (rank 0 only).
    fn rsClose(b: *Lanes) !void {
        const rs = &b.rs;
        if (!rs.open) return;
        rs.open = false;
        rs.rounds += 1;
        rs.sum_round += rs.t_last - rs.t_start;
        rs.sum_verify += rs.t_verify;
        rs.sum_draft += rs.t_draft;
        rs.sum_streams += rs.streams;
        rs.sum_rows += rs.rows;
        for (rs.win_acc[0..rs.streams]) |x| rs.sum_acc += x;
        if (rs.rounds < rs.every) return;
        const n: f64 = @floatFromInt(rs.rounds);
        const streams = @as(f64, @floatFromInt(rs.sum_streams)) / n;
        const rows = @as(f64, @floatFromInt(rs.sum_rows)) / n;
        const acc = @as(f64, @floatFromInt(rs.sum_acc)) / @as(f64, @floatFromInt(rs.sum_streams));
        const round = rs.sum_round * 1e3 / n;
        const draft = rs.sum_draft * 1e3 / n;
        const verify = rs.sum_verify * 1e3 / n;
        if (xq.prof.on) try b.runner.d.sync(); // drain the event ring into the totals before the report
        std.debug.print("round stats: {d} rounds, streams {d:.2} (mean), rows {d:.1} (mean), accepted {d:.2} a stream a round, round {d:.2} ms, drafter {d:.2} ms, verify {d:.2} ms, other {d:.2} ms, settle {d:.2} ms in {d:.1} waits, peer wait {d:.2} ms, launches {d:.0}, copies {d:.0}, syncs {d:.1}, host copies {d:.1} in {d:.3} ms\n", .{
            rs.rounds,
            streams,
            rows,
            acc,
            round,
            draft,
            verify,
            round - draft - verify,
            xq.stat.settle_s * 1e3 / n, // means a round, XPU_ROUND_STATS counters
            @as(f64, @floatFromInt(xq.stat.settles)) / n,
            xq.stat.peer_s * 1e3 / n,
            @as(f64, @floatFromInt(xq.stat.launches)) / n,
            @as(f64, @floatFromInt(xq.stat.copies)) / n,
            @as(f64, @floatFromInt(xq.stat.syncs)) / n,
            @as(f64, @floatFromInt(xq.stat.hcopies)) / n,
            xq.stat.hcopy_s * 1e3 / n,
        });
        if (xq.stat.paths > 0) { // lane p4-k-sk1 (GS2): the lone tree path's rows and paths of the same rounds
            const paths: f64 = @floatFromInt(xq.stat.paths);
            std.debug.print("tree stats: {d} rounds, {d} paths, tree rows {d:.2} a path, LOAD rows {d:.2} a path, straight paths {d:.1} %\n", .{
                rs.rounds,
                xq.stat.paths,
                @as(f64, @floatFromInt(xq.stat.tree_rows)) / paths,
                @as(f64, @floatFromInt(xq.stat.load_rows)) / paths,
                @as(f64, @floatFromInt(xq.stat.straight)) * 100.0 / paths,
            });
        }
        if (xq.stat.dr_on) { // XPU_DR_STEPS (lane p4-k-drt): the drafter's host steps of the same rounds
            const s = xq.stat.dr_s;
            const ms = 1e3 / n;
            var sum: f64 = 0;
            for (s, 0..) |x, k| {
                if (k != @intFromEnum(xq.DrStep.absorb_sync)) sum += x; // the absorb's sync is inside absorb
            }
            std.debug.print("drafter steps: {d} rounds, ms a round: absorb {d:.3} (its sync {d:.3}), pre {d:.3}, layers {d:.3}, head and gather {d:.3} (settles {d:.3} in {d:.1} waits, inside layers and head), sync {d:.3}, merge {d:.3}, tree {d:.3}, rest {d:.3} of drafter {d:.3}\n", .{
                rs.rounds,
                s[@intFromEnum(xq.DrStep.absorb)] * ms,
                s[@intFromEnum(xq.DrStep.absorb_sync)] * ms,
                s[@intFromEnum(xq.DrStep.pre)] * ms,
                s[@intFromEnum(xq.DrStep.layers)] * ms,
                s[@intFromEnum(xq.DrStep.head)] * ms,
                xq.stat.dr_settle_s * ms,
                @as(f64, @floatFromInt(xq.stat.dr_settles)) / n,
                s[@intFromEnum(xq.DrStep.sync)] * ms,
                s[@intFromEnum(xq.DrStep.merge)] * ms,
                s[@intFromEnum(xq.DrStep.tree)] * ms,
                draft - sum * ms,
                draft,
            });
        }
        xq.stat = .{ .on = xq.stat.on, .dr_on = xq.stat.dr_on }; // the counters reset; on and dr_on stay
        if (xq.prof.on) {
            var buf: [96]u8 = undefined;
            xq.profReport(try std.fmt.bufPrint(&buf, "{d} rounds, {d:.2} streams a round", .{ rs.rounds, streams }));
        }
        rs.* = .{ .on = rs.on, .every = rs.every };
    }

    /// Each asked stream's DFlash2 chain for its next window in one drafter pass, after the kept rows' taps.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        const t0: f64 = if (b.rs.on) xq.now() else 0;
        defer {
            if (b.rs.on) b.rsDrafted(t0);
        }
        const d = b.drafter orelse return;
        var which: [16]usize = undefined;
        var ps: [16][]const u32 = undefined;
        var na: usize = 0;
        var items: [16]xq.drafter.Block = undefined;
        var asked: [16]u32 = undefined;
        var counts: [16]u32 = undefined;
        var reqs: [16]*const be.DraftRequest = undefined;
        var ni: usize = 0;
        var depth: u32 = 0;
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            b.chain_n[slot] = 0;
            if (q.rows) |rows| if (b.count > 0) {
                const i = std.mem.indexOfScalar(u32, b.wslots[0..b.count], slot) orelse return error.UnknownStream;
                if (!b.absorbed[i]) {
                    which[na] = i;
                    ps[na] = rows;
                    na += 1;
                }
            };
            if (q.depth == 0 or ni == items.len) continue;
            items[ni] = .{ .slot = slot, .pending = if (q.rows != null) q.follow[q.follow.len - 1] else try b.value(q.first orelse return error.NoFirstToken) };
            asked[ni] = if (b.nodes == 0) q.depth else @min(q.depth, b.nodes); // XPU_TREE=0: chains of the asked depth
            counts[ni] = @min(asked[ni], b.block - 1);
            reqs[ni] = &q;
            depth = @max(depth, counts[ni]);
            ni += 1;
        }
        const dr_t = xq.stat.dr_on; // XPU_DR_STEPS (lane p4-k-drt): the host time of the drafter's steps
        var tm: f64 = if (dr_t) xq.now() else 0;
        try b.absorb(which[0..na], ps[0..na]);
        if (dr_t) xq.drMark(.absorb, &tm);
        if (ni == 0) return;
        for (requests) |q| if (q.depth > 0) {
            try d.use(b.slotOf(q.stream).?);
            if (d.end() + 1 != q.position) return error.BadDraftPosition;
        };
        // XPU_DRAFT_RAGGED: a batch is split into passes of at most 16 rows, the decode GEMM's limit (`gemm` takes
        // the prompt GEMM above 16 rows, ~6 ms a pass against ~1 ms). Six streams at depth 2 are 18 rows in one pass
        // and pay it; two passes of three streams stay on the decode kernels. A block is at most 8 rows, so a pass
        // always takes at least one block. Off: one pass, every block the longest (latticeMulti).
        const rag_max: u32 = 16;
        var g: usize = 0;
        while (g < ni) {
            var e = ni;
            if (xq.drafter.ragged and ni > 1) {
                e = g;
                var rs: u32 = 0;
                while (e < ni and rs + counts[e] + 1 <= rag_max) {
                    rs += counts[e] + 1;
                    e += 1;
                }
                if (e == g) e = g + 1; // a block alone: 8 rows at most
            }
            if (xq.drafter.ragged and ni > 1) try d.latticeRagged(items[g..e], counts[g..e]) else try d.latticeMulti(items[0..ni], depth);
            var tt: f64 = if (dr_t) xq.now() else 0;
            defer if (dr_t) xq.drMark(.tree, &tt);
            // each stream's best-first tree of its asked drafts over its lattice, with the target's keyed noise if sampled
            for (items[g..e], asked[g..e], reqs[g..e], 0..) |it, n, q, i| {
                d.take(i);
                var noise: [16][xq.drafter.TOPK]f64 = undefined;
                const sp = q.stream.sampling;
                const nd = if (xq.drafter.ragged) counts[g + i] else depth;
                if (sp) |x| for (0..nd) |dd| for (0..xq.drafter.TOPK) |j| {
                    noise[dd][j] = -@log(-@log(lanes.sampling.uniform(x.seed, q.position + dd, d.cand[dd][j])));
                };
                if (n < TREE_MIN or b.nodes == 0) { // few drafts (or XPU_TREE=0): a chain goes deeper than a tree of siblings
                    d.chain(it.pending, n, b.chain[it.slot][0..n]);
                    for (b.chain_par[it.slot][0..n], 0..) |*x, j| x.* = @as(i32, @intCast(j)) - 1;
                    b.chain_n[it.slot] = n;
                    continue;
                }
                b.chain_n[it.slot] = d.tree(it.pending, n, if (sp != null) &noise else null, if (sp) |x| x.temperature else 1.0, b.chain[it.slot][0..n], b.chain_par[it.slot][0..n]);
            }
            g = e;
        }
    }

    /// A slot's held chain as host tokens (taken: the next window spends it).
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return null;
        const n = b.chain_n[slot];
        b.chain_n[slot] = 0;
        const tokens = try gpa.dupe(u32, b.chain[slot][0..n]);
        errdefer gpa.free(tokens);
        const parents = try gpa.dupe(i32, b.chain_par[slot][0..n]); // by draft, −1 the pending token
        return .{ .count = n, .tokens = tokens, .parents = parents };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        b.settle() catch {};
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
        b.chain_n[slot] = 0;
        b.fill[slot].on = false;
    }

    /// A stream the lone driver ended: out of its slot, and out of Drafted's streams if it came from the lanes.
    fn forget(b: *Lanes, s: *lanes.Stream) void {
        if (b.wrap) |w| return w.backend().release(s); // its release reaches releaseFn
        if (b.slotOf(s)) |slot| b.owners[slot] = null;
    }

    /// A lone drafted stream in slot 0 on DFlash2 trees until it ends (false) or another stream arrives (true).
    pub fn lone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        if (hooks.resumed) return resumeLone(ctx, s, hooks);
        const b = self(ctx);
        if (b.drafter == null) return error.DraftNotAttached;
        try b.settle();
        if (b.owners[0] != null) return error.StreamBusy;
        b.owners[0] = s;
        var handed = false;
        defer if (!handed) b.forget(s);
        b.direct = true;
        defer b.direct = false;
        try b.prefillSlot(s, 0);
        const pending = try b.draw(s, 0, b.first_logits[0], s.prompt_len);
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        handed = try b.alone(s, 0, pending, hooks);
        return handed;
    }

    /// The stream the lane core was left with, alone again in its slot, on DFlash2 trees as `lone`.
    fn resumeLone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = self(ctx);
        const d = b.drafter orelse return error.DraftNotAttached;
        try b.settle();
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const r = b.runner;
        try r.select(slot);
        var handed = false;
        defer if (!handed) b.forget(s);
        b.direct = true;
        defer b.direct = false;
        try d.use(slot);
        if (d.end() != r.pos) d.restart(r.pos); // its context should end there: a cold one otherwise
        handed = try b.alone(s, slot, s.pending orelse return error.NoPending, hooks);
        return handed;
    }

    /// DFlash2 trees for `s` in `slot` from `pending` until it ends (false) or another stream arrives (true)
    fn alone(b: *Lanes, s: *lanes.Stream, slot: u32, pending0: u32, hooks: api.LoneHooks) !bool {
        const d = b.drafter.?;
        const r = b.runner;
        try d.use(slot);
        var pending = pending0;
        var out: [16]u32 = undefined;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = r.pos;
                s.pending = pending;
                if (b.wrap) |w| try w.adopt(s); // the lane core drafts for it from its slot's context
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            r.taps_on = true;
            defer r.taps_on = false;
            var sampled: Sampled = .{ .b = b, .s = s.sampling orelse .{ .seed = 0 } };
            const drawer: ?xq.drafter.Draw = if (s.sampling) |sp| .{ .ctx = &sampled, .row = Sampled.row, .gumbel = Sampled.gumbel, .temperature = sp.temperature } else null;
            const t0 = if (b.rs.on) xq.now() else 0; // XPU_ROUND_STATS: the round from its treeRound entry
            const round = try xq.drafter.treeRound(d, pending, b.block - 1, b.nodes, drawer, &out);
            s.rounds += 1;
            s.drafted += round.drafted;
            s.accepted += round.accepted;
            var landed: [17]u32 = undefined;
            const n = round.tokens.len - 1;
            @memcpy(landed[0..n], round.tokens[1..]);
            landed[n] = round.terminal;
            _ = try s.commit(b.gpa, landed[0 .. n + 1], &.{});
            if (b.rs.on) {
                const rs = &b.rs;
                rs.open = true;
                rs.t_start = t0;
                rs.t_last = xq.now();
                rs.t_draft = round.t_draft;
                rs.t_verify = round.t_verify;
                rs.streams = 1;
                rs.rows = round.rows;
                rs.win_acc[0] = @intCast(round.accepted);
                try b.rsClose();
            }
            pending = round.terminal;
            hooks.committed(hooks.ctx);
        }
        return false;
    }

    const Sampled = struct {
        b: *Lanes,
        s: api.Sampling,
        fn row(c_: *anyopaque, row_: u32, position: u64) !u32 {
            const c: *@This() = @ptrCast(@alignCast(c_));
            return c.b.pick(c.b.pick_ctx, try c.b.runner.rowLogits(row_), position, c.s);
        }
        fn gumbel(c_: *anyopaque, position: u64, id: u64) f64 {
            const c: *@This() = @ptrCast(@alignCast(c_));
            return -@log(-@log(lanes.sampling.uniform(c.s.seed, position, id)));
        }
    };
};

//! Sage-A1.5 (qwen3_5_moe) on the lane host: a stream per runner slot, one batched decode step over the running slots.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const lane = @import("xpu_lane.zig");
const sage = xq.sage;
const vl = @import("core").qwen_vl;
const kv = @import("core").kv_pool;
const be = lanes.backend;
const Allocator = std.mem.Allocator;

pub const model_type = "qwen3_5_moe";

/// Drawn tokens a handle can still name (a prompt's first draw, then one a stream a round).
const ring = 64;
/// The prompt cache keeps no state under this many tokens: a cut there splits a chunk, and the state saves little.
const min_kept = 4096;

pub const Slots = struct {
    gpa: Allocator,
    r: *sage.Runner,
    owners: [sage.MAX_SLOTS]?*lanes.Stream = @splat(null),
    half: []f16, // a row of logits as the runner reads it
    row: []f32, // and widened for the host draw
    ids: []i32, // a prompt chunk as the runner takes it
    first_rows: [sage.MAX_SLOTS]?[]f32 = @splat(null), // a structured stream's prompt-row logits, for first_masked
    masked_vals: ?[]f64 = null, // a structured sampled row's allowed candidates
    masked_ids: ?[]u64 = null,
    fed: [sage.MAX_SLOTS]u32 = @splat(1), // rows the last verify fed each slot
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    vision: ?*Vision = null, // --vision
    depth: u32 = 1, // MTP drafts a lone round (XPU_SPEC; 1 keeps 80 % of drafts on chat text)
    held: [sage.MAX_SLOTS]u32 = @splat(0), // drafts a slot holds for its next window (specMulti makes them then)
    sampled_drafts: bool = true, // sampled streams take MTP drafts too (XPU_SAMPLED_DRAFTS=0: one row a round)
    draft_q: bool = true, // a sampled stream's first draft is drawn from the MTP's filtered row (XPU_SAMPLED_Q=0: its argmax)
    spec: ?*Spec = null, // speculative sampling's host buffers, made at the first sampled window
    pool: ?kv.Pool = null, // the K/V pages all slots share (null: no shared pool)
    leases: [sage.MAX_SLOTS]kv.Lease = @splat(.{}), // each slot's pages from the pool
    pages_total: std.atomic.Value(u64) = .init(0), // the pool's counts, for admission and /metrics (other threads)
    pages_free: std.atomic.Value(u64) = .init(0),
    pages_used: std.atomic.Value(u64) = .init(0),
    pages_promised: std.atomic.Value(u64) = .init(0),
    pages_kept: std.atomic.Value(u64) = .init(0), // pages only kept states hold: evicting them frees these
    store: ?*api.prompt_cache.Store = null, // the prompt cache, whose kept states the pool evicts for a lease (reclaimFn)
    kept: u64 = 0, // pages one kept state or more holds
    gdn_pages: u32 = 0, // pool pages a kept state's GDN states take

    fn of(ptr: *anyopaque) *Slots {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn init(gpa: Allocator, r: *sage.Runner) !Slots {
        const v = r.z.s.vocab;
        const half = try gpa.alloc(f16, v);
        errdefer gpa.free(half);
        const row = try gpa.alloc(f32, v);
        errdefer gpa.free(row);
        return .{ .gpa = gpa, .r = r, .half = half, .row = row, .ids = try gpa.alloc(i32, sage.CHUNK) };
    }

    pub fn deinit(b: *Slots) void {
        if (b.spec) |x| x.deinit(b.gpa);
        b.gpa.free(b.half);
        b.gpa.free(b.row);
        b.gpa.free(b.ids);
        for (b.first_rows) |x| if (x) |row| b.gpa.free(row);
        if (b.masked_vals) |x| b.gpa.free(x);
        if (b.masked_ids) |x| b.gpa.free(x);
        if (b.pool) |*p| {
            for (&b.leases) |*l| p.finish(l);
            p.deinit();
        }
    }

    pub fn backend(b: *Slots) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .held = heldFn, .release = releaseFn } };
    }

    /// With an MTP head: greedy windows of the pending row and `depth` held drafts, all in one specMulti forward.
    /// Without: one-row rounds, no drafts; the running streams' rows share one forward (stepMulti). The lane core shares a
    /// forward only for a width of 2 or more, so the width is 2 with copy drafts off: every window is one row.
    /// stepMulti's grouped expert kernels add in another order than step, so a near-tie can pick another token.
    pub fn facts(b: *const Slots) lanes.Model {
        if (b.r.mtp != null) return .{ .exact_width = b.depth + 1, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .sampled_drafts = b.sampled_drafts, .drafts = b.depth, .draft_streams = true, .hidden_rows = true, .streams_exact = true, .batch_rows = sage.WINDOW, .max_streams = b.r.slots };
        return .{ .exact_width = 2, .copy_drafts = false, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = true, .streams_exact = true, .batch_rows = b.r.slots, .max_streams = b.r.slots };
    }

    fn push(b: *Slots, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn slotOf(b: *const Slots, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.r.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn freeSlot(b: *const Slots) ?u32 {
        for (b.owners[0..b.r.slots], 0..) |o, i| if (o == null) return @intCast(i);
        return null;
    }

    /// Pages a stream of `prompt` tokens and `max_new` more takes: its rows and one more verify window, at most ctx.
    fn need(b: *const Slots, prompt: u64, max_new: u64) u64 {
        return (@min(prompt + max_new + sage.WINDOW, b.r.ctx) + sage.PAGE - 1) / sage.PAGE;
    }

    /// The pool's counts where admission and /metrics read them.
    fn publish(b: *Slots) void {
        if (b.pool) |*p| {
            b.pages_total.store(p.total(), .release);
            b.pages_free.store(p.available(), .release);
            b.pages_used.store(p.total() - p.nfree, .release);
            b.pages_promised.store(p.promised, .release);
            var only_kept: u64 = 0;
            for (p.refs, p.leased) |r, l| only_kept += @intFromBool(r > 0 and l == 0);
            b.pages_kept.store(only_kept, .release);
        }
    }

    /// `slot`'s pages for a stream of `prompt` and `max_new` tokens: a kept state's K/V pages `kept` first (the slot
    /// holds them too and writes only past them), then fresh pages, promised and mapped now, so the stream takes no
    /// page while it runs. The pool evicts kept states for the fresh ones if it must (reclaimFn). LaneMemory: they do
    /// not fit beside the running streams now (the request waits).
    fn lease(b: *Slots, slot: u32, prompt: u64, max_new: u64, kept: []const u32) !void {
        const p = if (b.pool) |*x| x else return;
        b.unlease(slot);
        errdefer b.unlease(slot);
        const n = b.need(prompt, max_new);
        if (n > p.total()) return error.KvPoolTooSmall;
        if (kept.len >= n) return error.SnapshotLayout;
        if (kept.len > 0) {
            for (kept) |id| p.retain(id);
            try p.adopt(&b.leases[slot], kept); // its failure releases them
            try b.r.mapPages(slot, 0, kept);
        }
        try p.promise(&b.leases[slot], @intCast(n - kept.len));
        var map: Map = .{ .r = b.r, .slot = slot };
        try p.cover(&b.leases[slot], @intCast(n), .{ .ctx = &map, .write = Map.write });
        b.publish();
    }

    /// Pool.Reclaim: kept states go, least recently used first (a superseded one before them), until `want` pages are
    /// free and not promised. A state a prompt pass is restoring is borrowed and stays. False: none is left to evict.
    fn reclaimFn(ptr: *anyopaque, want: usize) bool {
        const b = of(ptr);
        const p = &b.pool.?;
        const store = b.store orelse return false;
        store.collectExpired();
        while (p.available() < want) {
            const i = store.victimBut(null, &.{}, 0) orelse {
                std.log.warn("prompt cache: no kept state left to drop for pages: {d} free, {d} wanted", .{ p.available(), want });
                return false;
            };
            const at = store.entries.items[i].at;
            store.remove(i);
            store.counts.evicted += 1;
            std.log.info("prompt cache: dropped the kept state of {d} tokens for pages: {d} free, {d} wanted, {d} states left", .{ at, p.available(), want, store.entries.items.len });
        }
        return true;
    }

    /// One more kept state holds page `id`.
    fn keepPage(b: *Slots, p: *kv.Pool, id: u32) void {
        if (p.refs[id] == p.leased[id]) b.kept += 1;
        p.retain(id);
    }

    /// One kept state fewer holds page `id`; the pool frees it with its last holder.
    fn dropPage(b: *Slots, p: *kv.Pool, id: u32) void {
        p.drop(id) catch std.debug.panic("XPU Sage: kept page {d} dropped with no kept holder", .{id});
        if (p.refs[id] == p.leased[id]) b.kept -= 1;
    }

    /// `slot`'s pages back to the pool; its table then maps none.
    fn unlease(b: *Slots, slot: u32) void {
        const p = if (b.pool) |*x| x else return;
        p.finish(&b.leases[slot]);
        b.r.mapPages(slot, 0, &.{}) catch {};
        b.publish();
    }

    /// A lease's new page ids into its slot's table on the device (kv.Table).
    const Map = struct {
        r: *sage.Runner,
        slot: u32,

        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const m: *Map = @ptrCast(@alignCast(ctx));
            try m.r.mapPages(m.slot, @intCast(from), ids[from..]);
        }
    };

    /// api.Admit: whether the pool can promise a request's pages now, kept states evicted (the pages only they hold
    /// count as free). It runs before the lookup, so a request that will resume a state counts all its pages: the count
    /// may also hold a state a pass borrows, so a lease can still wait (LaneMemory requeues it). A request no pool could
    /// hold goes through, so its prompt pass refuses it at once (PromptTooLong, KvPoolTooSmall).
    fn fitsFn(ptr: *anyopaque, prompt: u32, max_tokens: u32) bool {
        const b = of(ptr);
        const n = b.need(prompt, max_tokens);
        if (@as(u64, prompt) + max_tokens + 1 > b.r.ctx or n > b.pages_total.load(.acquire)) return true;
        return n <= b.pages_free.load(.acquire) + b.pages_kept.load(.acquire);
    }

    /// api.KvPoolSource: the pool's counts for /metrics, from the atomics alone.
    fn poolRead(ptr: *anyopaque) api.Status.KvPool {
        const b = of(ptr);
        return .{ .pages = b.pages_total.load(.acquire), .page_tokens = sage.PAGE, .used = b.pages_used.load(.acquire), .promised = b.pages_promised.load(.acquire), .free = b.pages_free.load(.acquire) };
    }

    /// The stream's next token: the runner's greedy one, else drawn on the host from logits row `row` (null: the
    /// current slot's last forward) at `position`.
    fn draw(b: *Slots, s: *const lanes.Stream, greedy: i32, row: ?u32, position: u64) !u32 {
        const sp = s.sampling orelse return @intCast(greedy);
        if (sp.temperature <= 0) return @intCast(greedy);
        return lane.draw(b.gpa, try b.fetchRow(row), sp, position);
    }

    /// Row `row` of the last forward (null: the current slot's last row) as f32 logits in b.row.
    fn fetchRow(b: *Slots, row: ?u32) ![]f32 {
        if (row) |i| try b.r.rowLogits(i, b.half) else try b.r.logits(b.half);
        for (b.row, b.half) |*w, h| w.* = h;
        return b.row;
    }

    /// Structured output: the draw among the tokens `mask` allows, greedy (first argmax) or sampled (keyed).
    fn drawMasked(b: *Slots, s: *const lanes.Stream, row: []const f32, position: u64, mask: []const u32) !u32 {
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

    /// The prompt into a free slot in chunks (a cancel lands between them); its first token drawn at once. A kept
    /// state of the prompt's prefix (s.reuse) is restored first: the slot's table maps its pages, and the pass goes on
    /// from its end in fresh pages.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 1 > r.ctx) return error.PromptTooLong;
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        s.cached = 0;
        var kept: ?*const sage.snapshot.Snap = null;
        if (ids.len < min_kept) {} else if (s.reuse.saved) |saved| { // a short prompt does no cache work
            const k: *const sage.snapshot.Snap = @ptrCast(@alignCast(saved));
            if (k.at < ids.len and r.snapFits(k.*)) kept = k else {
                s.reuse_failed = true;
                if (k.at < ids.len) s.reuse_error = error.SnapshotLayout;
            }
        }
        try b.lease(slot, ids.len, s.max_new, if (kept) |k| k.ids[0 .. k.at / sage.PAGE] else &.{});
        errdefer b.unlease(slot);
        const room = r.room(slot);
        if (ids.len + s.max_new + 1 > room) { // pages short of the request: the reply ends where they end
            std.log.warn("XPU Sage: slot {d} holds {d} positions, short of a {d}-token prompt and {d} new tokens; the reply stops at {d} tokens", .{ slot, room, ids.len, s.max_new, room -| ids.len -| 1 });
            s.max_new = @intCast(room -| ids.len -| 1);
        }
        try r.use(slot);
        try r.reset();
        const v: ?*Vision = if (s.images.len == 0) null else b.vision orelse return error.NoVision;
        const layout = if (v) |x| try x.encode(r, s) else vl.rope.Layout{};
        var at: usize = 0;
        if (kept) |k| {
            if (r.restoreSlot(slot, k.*)) {
                at = k.at;
                s.cached = k.at;
            } else |err| { // the pass from 0 writes from page 0: fresh pages, never the kept state's
                s.reuse_failed = true;
                s.reuse_error = err;
                try b.lease(slot, ids.len, s.max_new, &.{});
                try r.use(slot);
                try r.reset();
            }
        }
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            // chunks end at multiples of CHUNK from position 0, at the request's chunks and at the cache's marks: a
            // pass resumed at a mark cuts its chunks where a pass from 0 does
            var end = @min(ids.len, (at / sage.CHUNK + 1) * sage.CHUNK);
            for (s.chunks) |cut| if (cut > at and cut < end) {
                end = cut;
            };
            for (s.reuse.marks) |mark| if (mark >= min_kept and mark > at and mark < end) {
                end = mark;
            };
            const n = end - at;
            for (b.ids[0..n], ids[at..][0..n]) |*d, t| d.* = @intCast(t);
            const next: ?i32 = if (at + n < ids.len) @intCast(ids[at + n]) else null;
            try r.promptPiece(b.ids[0..n], if (v) |x| x.feed(r, &layout, @intCast(at), @intCast(n)) else .{}, next);
            at += n;
            if (at >= min_kept and std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |hk| hk.at(hk.ptr, s, @intCast(at));
        }
        r.shift[slot] = layout.shift;
        if (s.grammar != null) {
            const keep = b.first_rows[slot] orelse try b.gpa.alloc(f32, b.row.len);
            b.first_rows[slot] = keep;
            @memcpy(keep, try b.fetchRow(null));
        }
        const first = try b.draw(s, r.last, null, ids.len);
        if (b.sampled_drafts and s.drafts and s.grammar == null and first != r.last) try r.refollow(@intCast(first));
        _ = b.push(first);
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return b.push(try b.drawMasked(s, b.first_rows[slot] orelse return error.NoLogits, position, mask));
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return b.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const b = of(ptr);
        if (handle >= b.next or b.next - handle > ring) return error.NoSuchToken;
        return b.drawn[handle % ring];
    }

    /// Every stream's pending token in one forward: the single-row step for one stream, else stepMulti. A window of
    /// two rows (a grammar's forced token after the pending one) runs alone, row by row: its second row is fed only
    /// when the first row's draw equals it, so the slot holds exactly the rows a keep can take.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        if (windows.len == 0 or windows.len > r.slots or windows.len != out.len) return error.BadWindows;
        var slots: [sage.MAX_SLOTS]u32 = undefined;
        var toks: [sage.MAX_SLOTS]i32 = undefined;
        var at: [sage.MAX_SLOTS]usize = undefined; // the batched windows' indices in `windows`
        var n: usize = 0;
        try b.verifyDrafted(windows, out);
        for (windows, 0..) |w, i| {
            if (w.held > 0) continue;
            if (w.rows() > 2 or w.parents != null) return error.WindowTooWide;
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            const pos = if (slot == r.cur) r.pos else r.spos[slot];
            if (w.positions[0] != pos + 1) return error.PositionMismatch;
            b.fed[slot] = 1;
            if (w.rows() == 2) continue;
            slots[n] = slot;
            toks[n] = @intCast(w.pending);
            at[n] = i;
            n += 1;
        }
        if (n == 1) {
            try r.use(slots[0]);
            try r.step(toks[0]);
            toks[0] = r.last;
        } else if (n > 1) try r.stepMulti(slots[0..n], toks[0..n], toks[0..n]);
        for (at[0..n], 0..) |wi, i| {
            const row: ?u32 = if (n == 1) null else @intCast(i);
            out[wi].sampled[0] = try b.drawRow(windows[wi], 0, toks[i], row);
        }
        for (windows, out) |w, o| {
            if (w.rows() != 2 or w.held > 0) continue;
            const slot = b.slotOf(w.stream).?;
            try r.use(slot);
            try r.step(@intCast(w.pending));
            o.sampled[0] = try b.drawRow(w, 0, r.last, null);
            o.sampled[1] = o.sampled[0]; // not read unless the forced token lands
            if (o.sampled[0] == w.tokens[0]) {
                try r.step(@intCast(w.tokens[0]));
                o.sampled[1] = try b.drawRow(w, 1, r.last, null);
                b.fed[slot] = 2;
            }
            @memcpy(o.drafts, w.tokens);
        }
    }

    /// The windows with held drafts, grouped by depth, each group one specMulti: it drafts, verifies and commits.
    /// A window's rows report the kept tokens, and its drafts are the kept ones then one that differs, so the lane
    /// core keeps exactly the rows specMulti committed.
    fn verifyDrafted(b: *Slots, windows: []const be.Window, out: []be.Verified) !void {
        const r = b.r;
        var depth: u32 = 1;
        while (depth < sage.WINDOW) : (depth += 1) {
            var slots: [sage.MAX_SLOTS]u32 = undefined;
            var at: [sage.MAX_SLOTS]usize = undefined;
            var n: usize = 0;
            for (windows, 0..) |w, i| if (w.held == depth) {
                if (w.parents != null or w.masks.len > 0 or w.tokens.len > 0) return error.WindowTooWide;
                const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
                if (b.held[slot] < depth) return error.NoHeldDrafts;
                if (w.positions[0] != r.spos[slot] + 1 and !(slot == r.cur and w.positions[0] == r.pos + 1)) return error.PositionMismatch;
                // specMulti feeds the slot's last token: the runner's greedy one, which a sampled pending token replaces
                if (slot == r.cur) r.last = @intCast(w.pending) else r.slast[slot] = @intCast(w.pending);
                slots[n] = slot;
                at[n] = i;
                n += 1;
            };
            if (n == 0) continue;
            var toks: [sage.MAX_SLOTS][sage.WINDOW]i32 = undefined;
            var kept: [sage.MAX_SLOTS]usize = undefined;
            var judge: ?sage.Runner.Judge = null;
            for (at[0..n]) |wi| if (sampledOf(windows[wi].stream) != null) {
                const x = b.spec orelse try Spec.init(b.gpa, r.z.s.vocab);
                b.spec = x;
                for (at[0..n], slots[0..n], 0..) |wj, slot, i| x.of[i] = .{ .s = windows[wj].stream, .slot = slot, .position = windows[wj].positions[0] };
                judge = .{ .ctx = b, .keep = keepSampled, .draft = draftSampled };
                break;
            };
            try r.specMulti(slots[0..n], depth, toks[0..n], kept[0..n], judge);
            for (slots[0..n], at[0..n], toks[0..n], kept[0..n]) |slot, wi, t, k| {
                const o = out[wi];
                b.held[slot] = 0;
                b.fed[slot] = @intCast(k);
                for (o.sampled, 0..) |*x, j| x.* = @intCast(t[@min(j, k - 1)]);
                for (o.drafts, 0..) |*x, j| x.* = if (j + 1 < k) @intCast(t[j]) else @intCast(t[k - 1] ^ 1);
            }
        }
    }

    /// Judge.keep: speculative sampling of a sampled slot's block. Row j's draft x (from q) is kept with chance
    /// min(1, p(x) / q(x)), p and q both filtered by the stream's sampling. The first rejected row draws from
    /// max(0, p - q) normalized and ends the block; with every draft kept, the last row draws from p. A draft from
    /// the MTP's argmax has q one at x. The kept tokens then have the one-row path's distribution.
    fn keepSampled(ptr: *anyopaque, i: usize, row: u32, ids: []const i32, out: *[sage.WINDOW]i32) anyerror!?u32 {
        const b = of(ptr);
        const x = b.spec.?;
        const o = &x.of[i];
        const sp = sampledOf(o.s) orelse return null;
        const v = b.r.z.s.vocab;
        const rows: u32 = @intCast(ids.len);
        try b.r.logitRows(row, rows, x.rows[0 .. rows * v]);
        var j: u32 = 0;
        while (j + 1 < rows) : (j += 1) {
            const pos = o.position + j;
            const p = try x.dist(x.rows[j * v ..][0..v], sp, &x.p);
            const draft: u32 = @intCast(ids[j + 1]);
            const q: ?*const Dist = if (j == 0 and b.r.sdrawn[o.slot]) &x.q[o.slot] else null;
            const qx = if (q) |d| d.chance(draft) else 1.0;
            if (lanes.sampling.uniform(sp.seed, pos, salt_accept) * qx <= p.chance(draft)) {
                out[j] = @intCast(draft);
                continue;
            }
            out[j] = @intCast(p.residual(q, draft, lanes.sampling.uniform(sp.seed, pos, salt_residual)));
            o.position += j + 1;
            return j + 1;
        }
        const p = try x.dist(x.rows[j * v ..][0..v], sp, &x.p);
        out[j] = @intCast(p.pick(lanes.sampling.uniform(sp.seed, o.position + j, salt_bonus)));
        o.position += rows;
        return rows;
    }

    /// Judge.draft: a sampled slot's next first draft, drawn from the MTP row's filtered distribution (kept as its q).
    fn draftSampled(ptr: *anyopaque, i: usize, row: u32) anyerror!?i32 {
        const b = of(ptr);
        const x = b.spec.?;
        const o = &x.of[i];
        const sp = sampledOf(o.s) orelse return null;
        if (!b.draft_q) return null;
        const v = b.r.z.s.vocab;
        try b.r.logitRows(row, 1, x.rows[0..v]);
        const q = try x.dist(x.rows[0..v], sp, &x.q[o.slot]);
        return @intCast(q.pick(lanes.sampling.uniform(sp.seed, o.position, salt_draft)));
    }

    /// Row `j` of window `w` drawn: masked when the window has grammar masks, else greedy or sampled.
    fn drawRow(b: *Slots, w: be.Window, j: usize, greedy: i32, row: ?u32) !u32 {
        if (w.masks.len == 0) return b.draw(w.stream, greedy, row, w.positions[j]);
        const words = w.masks.len / w.rows();
        return b.drawMasked(w.stream, try b.fetchRow(row), w.positions[j], w.masks[j * words ..][0..words]);
    }

    /// A window keeps at most the rows its verify fed (one, or two when the forced token landed). A shorter path
    /// ends the stream; a next verify of it would fail its position check.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = of(ptr);
        for (windows, paths) |w, p| {
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            if (p.len == 0 or p.len > b.fed[slot]) return error.KeepMismatch;
        }
    }

    /// The MTP absorbed the kept rows in specMulti (or the prompt in prefill): note the drafts each stream's next
    /// window holds. specMulti makes them with its forward.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = of(ptr);
        if (b.r.mtp == null) return error.NoDraftHead;
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            b.held[slot] = @min(q.depth, b.depth);
        }
    }

    /// A lone greedy stream on MTP drafts (Runner.specStep, `depth` a round) until it ends (false) or another stream
    /// arrives (true). The lane core then decodes it beside the others, one row a round, with no drafts.
    fn loneFn(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = of(ptr);
        const r = b.r;
        try prefillFn(ptr, s);
        const slot = b.slotOf(s).?;
        var handed = false;
        defer if (!handed) {
            b.owners[slot] = null;
            b.unlease(slot);
        };
        var pending = b.drawn[(b.next - 1) % ring];
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        var out: [sage.WINDOW]i32 = undefined;
        var landed: [sage.WINDOW]u32 = undefined;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = r.pos;
                s.pending = pending;
                handed = true;
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            const n = try r.specStep(b.depth, &out);
            for (landed[0..n], out[0..n]) |*d, t| d.* = @intCast(t);
            s.rounds += 1;
            s.drafted += b.depth;
            s.accepted += n - 1;
            _ = try s.commit(b.gpa, landed[0..n], &.{});
            pending = landed[n - 1];
            hooks.committed(hooks.ctx);
        }
        return false;
    }

    /// A grammar stream's held draft as a host token, for the grammar to walk: the slot's MTP draft. Its window then
    /// runs on the two-row path of verifyFn, which feeds the draft only when the masked draw equals it.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (out.len != 1) return error.GrammarDraftsDeep; // the two-row path takes one draft
        out[0] = @intCast(@max(b.r.sdraft[slot], 0)); // none yet: any token, the draw decides
        b.held[slot] = 0;
    }

    /// The prompt cache's states, in the pool: a slot's K/V pages through a mark of its prompt pass, held by reference,
    /// and its GDN states copied into pool pages. The pass itself restores one (prefillFn). The budget counts the pages
    /// the states hold together (snapHeld): a turn's state shares the pages of the turn before.
    pub fn snapshots(b: *Slots) api.prompt_cache.Snapshots {
        return .{ .ptr = b, .vtable = &.{ .bytes = snapBytes, .save = snapSave, .restore = snapRestore, .drop = snapDrop, .held = snapHeld, .adds = snapAdds } };
    }

    fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return of(ptr).r.snapBytes(at);
    }

    fn snapHeld(ptr: *anyopaque) u64 {
        const b = of(ptr);
        return b.kept * sage.snapshot.poolPageBytes(b.r);
    }

    /// The pages a save at `at` would make kept: the slot's pages no state holds yet, and its GDN pages.
    fn snapAdds(ptr: *anyopaque, owner: ?*anyopaque, at: u32) u64 {
        const b = of(ptr);
        const p = if (b.pool) |*x| x else return snapBytes(ptr, at);
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return snapBytes(ptr, at)));
        const slot = b.slotOf(s) orelse return snapBytes(ptr, at);
        const ids = b.leases[slot].ids.items;
        var n: u64 = b.gdn_pages;
        for (ids[0..@min(ids.len, at / sage.PAGE)]) |id| n += @intFromBool(p.refs[id] == p.leased[id]);
        return n * sage.snapshot.poolPageBytes(b.r);
    }

    /// Keeps the slot's state at `at`: its K/V pages gain the state as a holder, and its GDN states go into free pages.
    /// Those come from free pages alone, without an eviction: the store is inside a keep, and its budget leaves them.
    fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        const b = of(ptr);
        const p = if (b.pool) |*x| x else return error.NoPool;
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const kv_n = at / sage.PAGE;
        if (at % sage.PAGE != 0 or kv_n > b.leases[slot].ids.items.len) return error.SnapshotLayout;
        const k = try b.gpa.create(sage.snapshot.Snap);
        errdefer b.gpa.destroy(k);
        const ids = try b.gpa.alloc(u32, kv_n + b.gdn_pages);
        errdefer b.gpa.free(ids);
        var gdn: kv.Lease = .{};
        errdefer p.finish(&gdn);
        const reclaim = p.reclaim;
        p.reclaim = null;
        defer p.reclaim = reclaim;
        try p.promise(&gdn, b.gdn_pages);
        try p.cover(&gdn, b.gdn_pages, null);
        @memcpy(ids[kv_n..], gdn.ids.items);
        std.mem.sort(u32, ids[kv_n..], {}, std.sort.asc(u32)); // runs of ids: fewer copies
        try b.r.saveSlot(slot, at, ids[kv_n..]);
        @memcpy(ids[0..kv_n], b.leases[slot].ids.items[0..kv_n]);
        for (ids) |id| b.keepPage(p, id);
        p.finish(&gdn); // the GDN pages' holder is the state now, not the lease
        k.* = .{ .at = at, .ids = ids };
        b.publish();
        return k;
    }

    fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {
        return error.RestoredInPrefill;
    }

    /// The state's pages lose it as a holder. A page no lease maps goes back to the pool at once: a copy still queued
    /// from it runs before any later work that writes it (one in-order list).
    fn snapDrop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const b = of(ptr);
        const k: *sage.snapshot.Snap = @ptrCast(@alignCast(saved));
        if (b.pool) |*p| for (k.ids) |id| b.dropPage(p, id);
        b.gpa.free(k.ids);
        b.gpa.destroy(k);
        b.publish();
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
        b.held[slot] = 0;
        b.unlease(slot);
    }
};

/// A sampled stream's sampling, or null for a greedy one.
fn sampledOf(s: *const lanes.Stream) ?lanes.Sampling {
    const sp = s.sampling orelse return null;
    return if (sp.temperature > 0) sp else null;
}

// Keys of the speculative sampler's uniforms at a position, apart from the one-row draw's token ids.
const salt_accept: u64 = 0xFFFF_0000_0000_0001;
const salt_residual: u64 = 0xFFFF_0000_0000_0002;
const salt_bonus: u64 = 0xFFFF_0000_0000_0003;
const salt_draft: u64 = 0xFFFF_0000_0000_0004;

/// A filtered next-token distribution: the candidates chooseOrdered keeps (top_k, top_p, min_p after temperature),
/// in its order, with their softmax chances.
const Dist = struct {
    ids: []u32 = &.{},
    p: []f64 = &.{},
    n: usize = 0,

    fn chance(d: *const Dist, id: u32) f64 {
        for (d.ids[0..d.n], d.p[0..d.n]) |t, c| if (t == id) return c;
        return 0;
    }

    /// The candidate at uniform `u` of the cumulative chances.
    fn pick(d: *const Dist, u: f64) u32 {
        var cum: f64 = 0;
        for (d.ids[0..d.n], d.p[0..d.n]) |t, c| {
            cum += c;
            if (u <= cum) return t;
        }
        return d.ids[d.n - 1];
    }

    /// A draw from max(0, p - q) normalized, with q the draft's distribution (null: one at `draft`).
    fn residual(p: *const Dist, q: ?*const Dist, draft: u32, u: f64) u32 {
        var total: f64 = 0;
        for (p.ids[0..p.n], p.p[0..p.n]) |t, c| total += @max(0, c - qOf(q, draft, t));
        if (!(total > 0)) return p.pick(u);
        var cum: f64 = 0;
        var last: u32 = p.ids[0];
        for (p.ids[0..p.n], p.p[0..p.n]) |t, c| {
            const w = @max(0, c - qOf(q, draft, t));
            if (w <= 0) continue;
            cum += w / total;
            last = t;
            if (u <= cum) return t;
        }
        return last;
    }

    fn qOf(q: ?*const Dist, draft: u32, t: u32) f64 {
        if (q) |d| return d.chance(t);
        return if (t == draft) 1 else 0;
    }
};

/// The host side of speculative sampling: a block's logit rows, p's scratch, each slot's q.
const Spec = struct {
    rows: []f16,
    vals: []f64, // a row's top candidates, by value then the lower id
    order: []u32,
    p: Dist,
    q: [sage.MAX_SLOTS]Dist,
    of: [sage.MAX_SLOTS]struct { s: *const lanes.Stream, slot: u32, position: u64 } = undefined, // specMulti's slot i

    fn init(gpa: Allocator, vocab: usize) !*Spec {
        const x = try gpa.create(Spec);
        errdefer gpa.destroy(x);
        x.* = .{ .rows = try gpa.alloc(f16, sage.WINDOW * vocab), .vals = try gpa.alloc(f64, vocab), .order = try gpa.alloc(u32, vocab), .p = try newDist(gpa, vocab), .q = undefined };
        for (&x.q) |*d| d.* = try newDist(gpa, vocab);
        return x;
    }

    fn newDist(gpa: Allocator, vocab: usize) !Dist {
        return .{ .ids = try gpa.alloc(u32, vocab), .p = try gpa.alloc(f64, vocab) };
    }

    fn deinit(x: *Spec, gpa: Allocator) void {
        gpa.free(x.rows);
        gpa.free(x.vals);
        gpa.free(x.order);
        gpa.free(x.p.ids);
        gpa.free(x.p.p);
        for (x.q) |d| {
            gpa.free(d.ids);
            gpa.free(d.p);
        }
        gpa.destroy(x);
    }

    /// `row`'s distribution under `sp` into `d`, with chooseOrdered's candidates and cutoffs.
    fn dist(x: *Spec, row: []const f16, sp: lanes.Sampling, d: *Dist) !*const Dist {
        const width = row.len;
        const k: usize = if (sp.top_k == 0) width else @min(width, sp.top_k);
        var filled: usize = 0;
        if (k == width) {
            for (x.order[0..width], 0..) |*o, id| o.* = @intCast(id);
            const Ctx = struct {
                r: []const f16,
                fn lt(c: @This(), a: u32, b: u32) bool {
                    return c.r[a] > c.r[b] or (c.r[a] == c.r[b] and a < b);
                }
            };
            std.mem.sort(u32, x.order[0..width], Ctx{ .r = row }, Ctx.lt);
            for (x.vals[0..width], x.order[0..width]) |*v, id| v.* = row[id];
            filled = width;
        } else for (row, 0..) |h, id| { // insertion into the k best; an equal value stays behind the lower id
            const v: f64 = h;
            if (filled == k and v <= x.vals[k - 1]) continue;
            var at = @min(filled, k - 1);
            while (at > 0 and v > x.vals[at - 1]) : (at -= 1) {
                x.vals[at] = x.vals[at - 1];
                x.order[at] = x.order[at - 1];
            }
            x.vals[at] = v;
            x.order[at] = @intCast(id);
            filled = @min(filled + 1, k);
        }
        for (x.vals[0..filled]) |v| if (!std.math.isFinite(v)) return error.NonfiniteLogits;
        const t = @max(sp.temperature, 1e-6);
        const top = x.vals[0] / t;
        var keep = filled;
        if (sp.top_p > 0.0 and sp.top_p < 1.0) {
            var total: f64 = 0;
            for (x.vals[0..filled], d.p[0..filled]) |v, *c| {
                c.* = @exp(v / t - top);
                total += c.*;
            }
            var cum: f64 = 0;
            var below: usize = 0;
            for (d.p[0..filled]) |c| {
                cum += c / total;
                if (cum < sp.top_p) below += 1;
            }
            keep = @min(keep, below + 1);
        }
        if (sp.min_p > 0.0) {
            const floor = top + @log(sp.min_p);
            var n: usize = 0;
            for (x.vals[0..keep]) |v| {
                if (v / t >= floor) n += 1;
            }
            keep = n;
        }
        var total: f64 = 0;
        for (x.vals[0..keep], d.p[0..keep], d.ids[0..keep], x.order[0..keep]) |v, *c, *id, o| {
            c.* = @exp(v / t - top);
            total += c.*;
            id.* = o;
        }
        for (d.p[0..keep]) |*c| c.* /= total;
        d.n = keep;
        return d;
    }
};

/// --vision: the checkpoint's image tower, its processor, and the rows one prompt's images take.
const Vision = struct {
    io: std.Io,
    c: vl.Config,
    tower: *xq.vision.Tower,
    rows: *anyopaque, // fp32 [image_tokens, hidden]: a prompt's image rows, image after image
    image_tokens: u32,
    per_image: u32, // tokens one image takes at most: the tower's scratch
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text
    spans: [vl.rope.max_images]struct { at: u32, n: u32, row: u32 } = undefined,
    count: u32 = 0,
    rope: [sage.CHUNK][3]i32 = undefined,
    pieces: [vl.rope.max_images]sage.Rows = undefined,

    /// The tower of the checkpoint in `dir` on the runner's card; null with `why` set when it has none it reads.
    fn open(gpa: Allocator, io: std.Io, a: Allocator, r: *sage.Runner, dir: []const u8, o: api.VisionOpen, why: *[]const u8) !?*Vision {
        const read = struct {
            fn f(al: Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
                const path = std.fs.path.join(al, &.{ d, name }) catch return null;
                return std.Io.Dir.cwd().readFileAlloc(i, path, al, .limited(16 << 20)) catch null;
            }
        }.f;
        const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
        const c = (vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json")) catch null) orelse {
            why.* = "--vision: this checkpoint names no image tower the XPU engine reads (config.json vision_config)";
            return null;
        };
        if (c.out_hidden != r.z.s.hidden or c.patch != xq.vision.patch or c.temporal != xq.vision.temporal or c.merge != xq.vision.merge or c.positions != xq.vision.grid_side * xq.vision.grid_side or !textRope(a, config)) {
            why.* = "--vision: the image tower or the text's M-RoPE (interleaved 11, 11, 10) is not the Qwen3.5 one the XPU engine runs";
            return null;
        }
        const per_image: u32 = @intCast(@min(@as(u64, o.image_tokens), c.max_pixels / (@as(u64, c.factor()) * c.factor())));
        const prefix = if (r.ckpt.has("vision_tower.patch_embed.proj.weight")) "vision_tower." else "model.visual.";
        const tower = xq.vision.Tower.load(&r.d, &r.ckpt, prefix, .{ .hidden = c.hidden, .heads = c.heads, .depth = c.depth, .intermediate = c.intermediate, .out_hidden = c.out_hidden }, per_image * c.merge * c.merge) catch |err| {
            why.* = try std.fmt.allocPrint(a, "--vision: the XPU engine reads the tower in bf16 and could not load this one: {s}", .{@errorName(err)});
            return null;
        };
        errdefer tower.deinit();
        const v = try gpa.create(Vision);
        v.* = .{ .io = io, .c = c, .tower = tower, .rows = try r.d.alloc_(@as(usize, o.image_tokens) * c.out_hidden * 4), .image_tokens = o.image_tokens, .per_image = per_image, .markers = .{ c.vision_start, c.image_token, c.vision_end } };
        return v;
    }

    fn deinit(v: *Vision, gpa: Allocator) void {
        v.tower.deinit();
        gpa.destroy(v);
    }

    /// Info.vision's prepare: an image's bytes to the tower's patches (core/qwen_vl geometry), on the HTTP thread.
    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const v: *Vision = @ptrCast(@alignCast(ctx));
        const p = try vl.geometry.prepare(v.io, a, bytes, v.c, @min(max_tokens, v.per_image));
        return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
    }

    fn offer(v: *Vision, o: api.VisionOpen) api.Vision {
        // the template writes Qwen's vision_start and vision_end around the image token itself
        return .{ .ctx = v, .prepare = prepare, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.image_tokens, .max_images = @min(o.max_images, vl.rope.max_images) };
    }

    /// `s`'s images through the tower into `rows`, and its prompt's M-RoPE layout.
    fn encode(v: *Vision, r: *sage.Runner, s: *const lanes.Stream) !vl.rope.Layout {
        const layout = try vl.rope.Layout.init(s.prompt(), s.images, .{ .start = v.c.vision_start, .image = v.c.image_token, .end = v.c.vision_end }, v.c.merge);
        const D: usize = v.c.out_hidden;
        var row: u32 = 0;
        v.count = 0;
        for (s.images) |img| {
            if (row + img.tokens > v.image_tokens or img.tokens > v.per_image) return error.ImageTooLarge;
            try v.tower.encode(&r.d, img.pixels, img.gh, img.gw, off(v.rows, row * D * 4));
            v.spans[v.count] = .{ .at = img.at, .n = img.tokens, .row = row };
            v.count += 1;
            row += img.tokens;
        }
        return layout;
    }

    /// Prompt rows [at, at + n): their positions and the image rows among them.
    fn feed(v: *Vision, r: *const sage.Runner, layout: *const vl.rope.Layout, at: u32, n: u32) sage.Feed {
        const D: usize = r.z.s.hidden;
        for (v.rope[0..n], 0..) |*p, i| p.* = layout.position(at + @as(u32, @intCast(i)));
        var k: usize = 0;
        for (v.spans[0..v.count]) |sp| {
            const lo = @max(sp.at, at);
            const hi = @min(sp.at + sp.n, at + n);
            if (lo >= hi) continue;
            v.pieces[k] = .{ .at = lo - at, .n = hi - lo, .src = off(v.rows, (sp.row + lo - sp.at) * D * 4) };
            k += 1;
        }
        return .{ .rope = v.rope[0..n], .rows = v.pieces[0..k] };
    }
};

/// The text's rotary is Qwen3.5's interleaved M-RoPE over sections 11, 11 and 10 (attn_prep's ROPE).
fn textRope(a: Allocator, config_json: []const u8) bool {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, config_json, .{}) catch return false;
    if (doc != .object) return false;
    const tc = doc.object.get("text_config");
    const text = if (tc != null and tc.? == .object) tc.?.object else doc.object;
    const rope = text.get("rope_parameters") orelse return false;
    if (rope != .object) return false;
    const inter = rope.object.get("mrope_interleaved") orelse return false;
    const sec = rope.object.get("mrope_section") orelse return false;
    if (inter != .bool or !inter.bool or sec != .array or sec.array.items.len != 3) return false;
    for (sec.array.items, [_]i64{ 11, 11, 10 }) |x, want| if (x != .integer or x.integer != want) return false;
    return true;
}

const Served = struct {
    gpa: Allocator,
    back: Slots,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: no budget

    fn close(ptr: *anyopaque) void {
        const h: *Served = @ptrCast(@alignCast(ptr));
        h.host.stop();
        if (h.cache) |*store| store.deinit(); // the states' pages, before the pool's
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        const r = h.back.r;
        if (h.back.vision) |v| v.deinit(h.gpa);
        h.back.deinit();
        r.deinit(); // the card's memory, after the engine has stopped
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
};

/// Sage from `o.dir` (MLX affine 4/64, or AutoRound GPTQ 4/128 shown as MLX at load): --parallel slots (1 to 8), a
/// stream of at most --context tokens (262144 by default), their K/V in one pool of pages (--kv-pool-tokens, else
/// what the card leaves), kept prompt states in pages of the same pool, int8 K/V with XPU_KV_INT8.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const context: u32 = std.math.cast(u32, o.context orelse 262144) orelse 0;
    if (context == 0) {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine needs a context of 1 to 2^32-1 tokens, not {d}", .{o.context.?});
        return null;
    }
    const slots: u32 = @min(@max(o.lanes, 1), sage.MAX_SLOTS);
    // --drafter: the AutoRound checkpoint whose MTP head drafts for a lone greedy stream
    const mtp = if (o.drafts) o.drafter else null;
    const r = sage.Runner.initMtp(o.dir, context, slots, mtp) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine could not load {s} with {d} streams of {d} tokens: {s}", .{ o.dir, slots, context, @errorName(err) });
        return null;
    };
    errdefer r.deinit();
    const h = try gpa.create(Served);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .back = try Slots.init(gpa, r), .cfg = undefined, .wall = .{ .io = io }, .core = undefined, .host = undefined };
    errdefer h.back.deinit();
    if (o.vision) |vo| h.back.vision = try Vision.open(gpa, io, a, r, o.dir, vo, problem) orelse return null;
    errdefer if (h.back.vision) |v| v.deinit(gpa);
    if (std.c.getenv("XPU_SPEC")) |v| h.back.depth = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    if (std.c.getenv("XPU_SAMPLED_DRAFTS")) |v| h.back.sampled_drafts = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("XPU_SAMPLED_Q")) |v| h.back.draft_q = !std.mem.eql(u8, std.mem.span(v), "0");
    if (h.back.depth == 0 or h.back.depth + 1 > sage.WINDOW) {
        problem.* = try std.fmt.allocPrint(a, "XPU_SPEC: 1 to {d} MTP drafts a round, not {d}", .{ sage.WINDOW - 1, h.back.depth });
        return null;
    }
    // the pool takes what the card leaves; kept prompt states hold pages of it, at most the cache's budget
    const cache_on = if (o.prompt_cache_gib) |gib| gib > 0 else true;
    const pages = poolPages(r, slots, context, o.kv_pool_tokens, cache_on);
    const budget = cacheBudget(r, o, pages, context);
    if (!(try fitsCard(a, r, slots, pages, problem) and try poolMade(a, r, pages, problem))) {
        if (h.back.vision) |v| v.deinit(gpa);
        h.back.deinit();
        gpa.destroy(h);
        r.deinit();
        return null;
    }
    h.back.pool = try kv.Pool.init(gpa, pages, r.kvBytesPerToken() * sage.PAGE);
    h.back.pool.?.reclaim = .{ .ctx = &h.back, .call = Slots.reclaimFn }; // a lease evicts kept states it needs pages of
    h.back.gdn_pages = sage.snapshot.gdnPages(r);
    h.back.publish();
    const pool_tokens = @as(u64, pages) * sage.PAGE;
    if (pool_tokens < context) std.log.warn("XPU Sage: the K/V pool holds {d} tokens, less than one stream's --context of {d}: a longer request is refused", .{ pool_tokens, context });
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), if (r.mtp != null) h.back.depth + 1 else 2, if (r.mtp != null) h.back.depth else 0);
    errdefer h.cfg.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    const pool_bytes = pool_tokens * r.kvBytesPerToken();
    const cache_text = if (budget > 0) try std.fmt.allocPrint(a, "kept prompt states in up to {d:.2} GB of it ({d:.0} %)", .{ gb(budget), 100 * @as(f64, @floatFromInt(budget)) / @as(f64, @floatFromInt(pool_bytes)) }) else "no prompt cache";
    h.startup = try std.fmt.allocPrint(gpa, "Intel GPU engine: Sage (qwen3_5_moe), {d} stream(s) of at most {d} tokens, a K/V pool of {d} pages ({d} tokens, {d:.2} GB, {s}), {s}, {d:.2} GB of the card kept free, sampling on the host{s}", .{ slots, context, pages, pool_tokens, gb(pool_bytes), if (r.cap > 0) "int8" else "bf16", cache_text, gb(cardReserve()), if (r.mtp == null) "" else if (h.back.sampled_drafts) ", MTP drafts for greedy and sampled streams" else ", MTP drafts for greedy streams" });
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "sage-xpu", .lanes = slots, .context_window = context, .context_fitted = true, .startup = h.startup, .prefill_step = 0, .structures = true });
    h.host.explain = .{ .text = lane.explain };
    h.host.info_.admit = .{ .ctx = &h.back, .fits = Slots.fitsFn }; // a request waits until the pool can promise its pages
    h.host.kv_pool = .{ .ctx = &h.back, .read = Slots.poolRead };
    if (h.back.vision) |v| h.host.info_.vision = v.offer(o.vision.?);
    if (r.mtp != null) h.host.lone = .{ .ctx = &h.back, .run = Slots.loneFn };
    if (budget > 0) {
        // kept and resumed only at multiples of 64 tokens (planned, grid): every cut a resumed pass adds splits a CHUNK
        // into pieces of more than WINDOW rows, which the chunk kernels compute as the whole chunk does. The MTP's
        // last row reads the token after a state (lookahead).
        h.cache = api.prompt_cache.Store.init(gpa, h.back.snapshots(), .{ .lookahead = if (r.mtp != null) 1 else 0, .planned = true, .grid = 64 }, budget);
        h.host.cache = &h.cache.?;
        h.back.store = &h.cache.?;
        h.host.info_.prompt_cache = true;
        std.log.info("sage-xpu: prompt cache of {d:.2} GiB", .{@as(f64, @floatFromInt(budget)) / (1 << 30)});
    }
    errdefer if (h.cache) |*store| store.deinit();
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Served.close, .ctx = h };
}

/// Card bytes a prompt pass and the driver hold beyond the counted load: the pass's chunk buffers, the driver's page
/// tables and rings, and a margin. Measured with the xe VRAM manager's usage (debugfs tile0/vram_mm) under 4 sessions of
/// 126k to 177k tokens (--parallel 4 --context 262144, the MTP head, --vision, int8 K/V, 2 GiB cache): idle 31.90 GB
/// of the 32.03 GB counted, a pass at most 0.44 GB above idle, so 0.31 GB past the count. Taken with 0.34 GB of margin.
const pass_reserve: u64 = 650_000_000;
/// Bytes the card holds beyond what Dev.allocated counts (the runtime's own buffers). Measured with drm-total-vram0
/// after 4 streams at once, the pool zeroed at load: 32.47 GB counted and 32.71 GB held (--parallel 4 --context
/// 220000, the MTP head, --vision, int8 K/V). Taken above the 0.24 GB seen.
const held_not_counted: u64 = 400_000_000;

/// Card bytes kept free of the load and the pool: pass_reserve. Kept prompt states hold pool pages, so they need none.
fn cardReserve() u64 {
    return pass_reserve;
}

/// The prompt cache's budget, the most pool bytes kept prompt states hold at once: --prompt-cache-gib (0: no cache),
/// at most the pool, else the pool less one stream's whole context, so a lone stream fits beside the states. A lease
/// the free pages do not cover evicts states besides (Slots.reclaimFn).
fn cacheBudget(r: *sage.Runner, o: api.Open, pages: u32, context: u32) u64 {
    const page: u64 = r.kvBytesPerToken() * sage.PAGE;
    const pool = @as(u64, pages) * page;
    if (o.prompt_cache_gib) |gib| {
        if (!(gib > 0)) return 0;
        const asked = std.math.lossyCast(u64, gib * (1 << 30));
        if (asked > pool) std.log.warn("XPU Sage: --prompt-cache-gib {d:.2} is more than the K/V pool ({d:.2} GB): kept prompt states may hold all of it", .{ gib, gb(pool) });
        return @min(asked, pool);
    }
    const lone = (@as(u64, context) + sage.PAGE - 1) / sage.PAGE * page;
    if (pool <= lone) std.log.warn("XPU Sage: the K/V pool ({d:.2} GB) holds no more than one stream's --context, so no prompt cache (--prompt-cache-gib names one)", .{gb(pool)});
    return pool -| lone;
}

/// Whether the card holds the load and a K/V pool of `pages` pages (kept prompt states are pages of it). The pool is
/// zeroed at load, so a pool the card cannot hold fails then, not under a request. Else `problem` names the pool, the
/// most the card holds, and the flags.
fn fitsCard(a: Allocator, r: *sage.Runner, slots: u32, pages: u32, problem: *[]const u8) !bool {
    const mem = r.d.g.memory() catch |err| {
        std.log.warn("XPU: no Sysman memory size ({s}), so the load does not check that the card holds the K/V pool", .{@errorName(err)});
        return true;
    };
    const keep = cardReserve();
    const room = mem.size -| keep;
    const page: u64 = r.kvBytesPerToken() * sage.PAGE;
    const pool: u64 = @as(u64, pages) * page;
    const need: u64 = r.d.allocated + pool + held_not_counted;
    std.log.info("XPU: the load holds {d:.2} GB of the card's {d:.2} GB ({d:.2} GB kept free for a prompt pass and the driver), {d:.2} GB of it the K/V pool", .{ gb(need), gb(mem.size), gb(keep), gb(pool) });
    if (need <= room and pages > 0) return true;
    const most = (room -| (r.d.allocated + held_not_counted)) / page * sage.PAGE;
    problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine needs {d:.2} GB for {d} stream(s){s} with a K/V pool of {d} tokens ({d:.2} GB, --kv-pool-tokens), and the card has {d:.2} GB with {d:.2} GB kept free. The pool fits at most {d} tokens. Set a smaller --kv-pool-tokens, or fewer --parallel slots", .{ gb(need), slots, if (r.mtp != null) " with the MTP head" else "", @as(u64, pages) * sage.PAGE, gb(pool), gb(mem.size), gb(keep), most });
    return false;
}

/// The pool's pages: --kv-pool-tokens rounded up to whole pages, else what the card leaves after the load (the
/// weights, the tower, the activations, every slot's GDN state) and the reserve (cardReserve). Without a prompt cache
/// (`cache_on` false) at most every slot's whole context; with one, the pages past that hold kept states.
fn poolPages(r: *sage.Runner, slots: u32, context: u32, tokens: ?u32, cache_on: bool) u32 {
    if (tokens) |t| return @intCast((@as(u64, t) + sage.PAGE - 1) / sage.PAGE);
    const most: u64 = @as(u64, slots) * ((@as(u64, context) + sage.PAGE - 1) / sage.PAGE);
    const mem = r.d.g.memory() catch return @intCast(most);
    const left = (mem.size -| cardReserve()) -| (r.d.allocated + held_not_counted);
    const pages = left / (r.kvBytesPerToken() * sage.PAGE);
    return @intCast(if (cache_on) pages else @min(most, pages));
}

/// The runner's pool of `pages` pages; false with `problem` set when the card does not give it.
fn poolMade(a: Allocator, r: *sage.Runner, pages: u32, problem: *[]const u8) !bool {
    r.allocPool(pages) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine could not make a K/V pool of {d} pages ({d} tokens): {s}", .{ pages, @as(u64, pages) * sage.PAGE, @errorName(err) });
        return false;
    };
    return true;
}

fn gb(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

//! Exact prompt reuse: states are keyed by tokens and canonical decoded intervals through their position.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");
const modes = @import("cache_modes.zig");
const learned = @import("prompt_learn.zig");
const lifetime = @import("prompt_cache_lifetime.zig");
const policy = @import("prompt_cache_policy.zig");
const cuts = @import("prompt_cuts.zig");
const interior = @import("prompt_interior.zig");
pub const rows = @import("rows_store.zig");

/// A family's copy of one state.
pub const Saved = *anyopaque;

pub const singleReclaim = imprint.singleReclaim; // the learned-state file helpers, beside the Imprint that owns them
pub const singleForgetChecked = imprint.singleForgetChecked;

/// Pages a sharing family prints after the status line: pages kept states hold, the pool, and pages free.
pub const PageNote = struct { held: u32 = 0, pool: u32 = 0, free: u32 = 0 };

/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// The storage a kept state holds: new storage up to `bytes`, or the spare storage it took (null: `bytes`).
        charged: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Free storage the family keeps for later saves, inside the budget (null: none).
        spare: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Free spare storage until at most `room` bytes stay spare.
        trim: ?*const fn (ptr: *anyopaque, room: u64) void = null,
        /// Whether a save at `at` would take spare storage (no new storage).
        reuses: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        /// Bytes the kept states hold together, for a family whose states share storage (one page under several
        /// states, or under a state and a running stream, counts once). Null: the sum of their charges.
        held: ?*const fn (ptr: *anyopaque) u64 = null,
        /// The storage a save at `at` for `owner` adds to what the kept states hold now (with `held`; null: `bytes`).
        adds: ?*const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) u64 = null,
        /// Bytes `held` falls by when this state is dropped. Null: the state's charge. A state that frees nothing stays when bytes are short.
        frees: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Bytes a drop gives back to the streams. Null: the state's charge. `evictOne` skips a state that returns nothing.
        returns: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Host bytes a park of `saved` takes, summed over the ranks. Null: no tier. Zero: this state cannot park.
        park_bytes: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Move `saved` into the host tier. Returns the copy's milliseconds. Null: no tier.
        park: ?*const fn (ptr: *anyopaque, saved: Saved) anyerror!f64 = null,
        /// Milliseconds of the last park copy or parked restore. Null: the resume line prints 0.
        copy_ms: ?*const fn (ptr: *anyopaque) f64 = null,
        /// Pages for the status line (null: the line is unchanged).
        pages: ?*const fn (ptr: *anyopaque) PageNote = null,
        /// Write `saved` as learned state `key`, in files of the family's own under `dir` (null: nothing kept on disk).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void = null,
        /// Learned state `key` of `at` tokens, read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved = null,
        /// Peer disk demand and reservation lifetime; null functions mean a single local rank.
        reclaim: ?*const fn (*anyopaque, [:0]const u8, u64) anyerror!u64 = null,
        peer_other_used: ?*const fn (*anyopaque, u64) anyerror!u64 = null,
        peer_need: ?*const fn (ptr: *anyopaque, saved: Saved, extra: u64) anyerror!u64 = null,
        peer_reclaim: ?*const fn (ptr: *anyopaque, key: u64) anyerror!u64 = null,
        peer_reserve: ?*const fn (ptr: *anyopaque, saved: Saved, key: u64, extra: u64) anyerror!void = null,
        peer_finish: ?*const fn (ptr: *anyopaque, key: u64, success: bool) anyerror!void = null,
        peer_other_next: ?*const fn (*anyopaque, ?u64) anyerror!?u64 = null,
        peer_other_bytes: ?*const fn (*anyopaque, u64) anyerror!u64 = null,
        peer_other_remove: ?*const fn (*anyopaque, u64) anyerror!void = null,
        forget_checked: ?*const fn (*anyopaque, [:0]const u8, u64) anyerror!void = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
        /// --learn-conversations: the state's image as rows and parts, and copies out of and into kept states.
        rows: ?*const rows.Image = null,
    };
};

/// What a paged store asks of the backend's pages: who holds each, and how many are free.
pub const Pages = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// Bytes of one page, over every layer.
    bytes: u64,

    pub const VTable = struct {
        retain: *const fn (ptr: *anyopaque, id: u32) void,
        release: *const fn (ptr: *anyopaque, id: u32) void,
        /// How many hold the page: the store counts as one.
        holders: *const fn (ptr: *anyopaque, id: u32) u32,
        /// Pages free and not promised to a stream.
        available: *const fn (ptr: *anyopaque) usize,
    };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A planned family's own zero-anchored chunk rows, its starts when a request names none (0: none).
    grid: u32 = 0,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep only the history's mark, at a chunk end (`chunk`); a planned family's marks add no call.
    min_prompt: u32 = 4096,
    /// Rows an unplanned pass's chunk runs from its resume point or last cut (1: any row; 0: short prompts keep none).
    chunk: u32 = 0,
    /// Unplanned passes also cut at Request.chunks and decoded-span edges (false: only their own grid and marks).
    chunk_starts: bool = true,
    /// Marks always end chunks; false when a backend may capture them inside a chunk without splitting it.
    chunk_marks: bool = true,
    /// Background reply passes kept at their end: .on drops a history mark near the resume point, .waits keeps it.
    warm: enum { off, on, waits } = .off,
    /// A paged store (prompt_radix.zig): the tokens a page holds, where its marks and resumes sit (0: not paged).
    page: u32 = 0,
    /// A paged store also keeps the prompt's last whole page, for a later turn whose history the request did not name.
    tail: bool = false,
};

pub const Entry = struct {
    tokens: []u32, // the state's tokens and `lookahead` more
    decode_spans: []modes.Span = &.{}, // canonical row arithmetic through at, excluding lookahead
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the prompt length of the request that kept it (a reply's end and its warm pass: that reply's): a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    branched: bool = false, // resumed again after a later prompt kept a longer state extending it: never superseded
    expired: bool = false, // hidden from lookups immediately, reclaimed only at safe host boundaries
    borrowers: u32 = 0, // outstanding lookups: released by resumed or release, never evicted while borrowed
    parked: bool = false, // K and V are in the host tier: no pool page, and it does not count against most
    tier_bytes: u64 = 0, // host bytes this parked state holds, summed over the ranks
};

pub const Counts = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    kept: u64 = 0,
    evicted: u64 = 0,
    refused: u64 = 0,
    failed: u64 = 0,
    /// Evictions by why: the byte budget, the state count, a stream that needed a page, a superseded state.
    evict_budget: u64 = 0,
    evict_count: u64 = 0,
    evict_stream: u64 = 0,
    evict_superseded: u64 = 0,
    parks: u64 = 0,
    park_resumes: u64 = 0,
    park_drops: u64 = 0,
};

const EvictWhy = enum { budget, count, stream, superseded };

fn whyOf(reason: []const u8, bytes_short: bool) EvictWhy {
    if (std.mem.indexOf(u8, reason, "superseded") != null) return .superseded;
    return if (bytes_short) .budget else .count;
}

pub const RestoreDetails = @import("prompt_cache_restore.zig").Details;
pub const RestoreLog = @import("prompt_cache_restore.zig").Log;

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The borrowed entry a backend restores; the caller must report resumed or release it, even on cancellation.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

/// One engine's store. The engine deploys one store to 6 to 8 lanes, one turn in flight per lane at a time. The
/// rewind protection here covers only the last lookup's rewind state, and another lane's rewind state may be
/// evicted. The cost is cold: a rewind edit reprefills.
/// The conversation grows one reply at a time and its turns never share a prompt, so a request's states never
/// supersede each other. The one that supersedes them is a later turn's state of the same conversation, and it
/// takes its storage before the least recently used one leaves. A shorter prompt from a later turn (a rewind)
/// supersedes none. A reply's end and its warm pass, kept at a position past the request's states, supersede
/// none of them and never lose to them.
pub const Store = struct {
    restore_log: ?RestoreLog = null,
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    most: usize = std.math.maxInt(usize), // states kept at most: a family's snapshot slots, one a state. A parked state does not count.
    tier: u64 = 0, // host bytes for parked states, the total over the ranks (0: the tier is off)
    tier_used: u64 = 0,
    park_min: u32 = 0, // a shorter state is dropped, not parked
    rank: u32 = 0, // the rank the park log lines name
    held: u64 = 0,
    forced: u64 = 0, // states a live stream's page need took (XPU_KEEP_SHARED), beside the budget's own evictions
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    disk: ?*rows.Rows = null, // --learn-conversations: every kept state written behind to disk, read back where it pays
    active_rewind_at: u32 = 0, // the last lookup's rewind state (0: none); hosts keep a pass's states before the next lookup

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {
        if (s.disk) |r| rows.write.dropped(r, e);
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.decode_spans);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    }

    /// Drops the least recently used kept state a stream's page need makes room with (never a borrowed one, and the
    /// rewind state last); false when no state is left to drop. With `returns`, a state that gives no page back stays,
    /// and false means none of them does. The family's drop returns the state's pages.
    pub fn evictOne(s: *Store) bool {
        const i = if (s.family.vtable.held != null and s.family.vtable.returns != null)
            policy.victimButIf(s, null, &.{}, 0, .returns)
        else
            s.victimBut(null, &.{}, 0);
        const at_ = i orelse return false;
        const leaving = s.entries.items[at_];
        if (!policy.extended(s, leaving) and s.tryPark(at_, "least recently used")) {
            s.forced += 1;
            return true;
        }
        const j = s.indexOf(leaving) orelse return false;
        if (leaving.parked) {
            s.counts.park_drops += 1;
            s.logParkDrop(leaving.at, leaving.tier_bytes, "superseded by a kept state");
        }
        s.remove(j);
        s.noteEvict(.stream);
        s.forced += 1;
        return true;
    }

    pub fn remove(s: *Store, i: usize) void {
        std.debug.assert(s.entries.items[i].borrowers == 0);
        const e = s.entries.orderedRemove(i);
        if (e.parked) s.tier_used -= e.tier_bytes;
        s.held -= e.bytes;
        s.free(e);
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// A kept state takes `extra` more bytes (a family copied what it shared into it) if the budget has room; never evicts, a pass is under way.
    pub fn grow(s: *Store, saved: Saved, extra: u64) bool {
        for (s.entries.items) |e| if (!e.expired and e.tokens.len > 0 and e.saved == saved) {
            if (extra > s.room()) return false;
            e.bytes += extra;
            s.held += extra;
            return true;
        };
        return false;
    }

    /// Whether an outstanding lookup still borrows `saved`'s state: the family spares it while restoring.
    pub fn borrowed(s: *Store, saved: Saved) bool {
        for (s.entries.items) |e| if (e.saved == saved and !e.expired and e.borrowers > 0) return true;
        return false;
    }

    pub const expired = lifetime.expired;
    pub const collectExpired = lifetime.collect;
    pub const release = lifetime.release;

    /// Bytes the kept states hold: the family's count when they share storage, else the sum of their charges.
    pub fn holding(s: *const Store) u64 {
        return if (s.family.vtable.held) |f| f(s.family.ptr) else s.held;
    }

    fn noteEvict(s: *Store, why: EvictWhy) void {
        s.counts.evicted += 1;
        switch (why) {
            .budget => s.counts.evict_budget += 1,
            .count => s.counts.evict_count += 1,
            .stream => s.counts.evict_stream += 1,
            .superseded => s.counts.evict_superseded += 1,
        }
    }

    /// Bytes the log names for a drop: what `held` falls by when the family counts shared pages, else the state's charge.
    fn freedOf(s: *const Store, e: *const Entry) u64 {
        if (s.family.vtable.held != null) if (s.family.vtable.frees) |f| return f(s.family.ptr, e.saved);
        return e.bytes;
    }

    fn sharingBytes(s: *const Store) bool {
        return s.family.vtable.held != null and s.family.vtable.frees != null;
    }

    /// Today's victim. When shared pages make the bytes short, the first in that order whose `frees` is above 0.
    fn victimFor(s: *const Store, skip: ?*Entry, prompt: []const u32, born: u32, bytes_short: bool) ?usize {
        if (bytes_short and s.sharingBytes()) return policy.victimButIf(s, skip, prompt, born, .frees);
        return policy.victimBut(s, skip, prompt, born);
    }

    fn tierOn(s: *const Store) bool {
        return s.tier > 0 and s.family.vtable.park != null and s.family.vtable.park_bytes != null;
    }

    /// States that count against `most`. A parked state does not. With the tier off this is the list length.
    pub fn occupancy(s: *const Store) usize {
        if (!s.tierOn()) return s.entries.items.len;
        var n: usize = 0;
        for (s.entries.items) |e| n += @intFromBool(!e.parked);
        return n;
    }

    pub fn parkedStates(s: *const Store) usize {
        var n: usize = 0;
        for (s.entries.items) |e| n += @intFromBool(e.parked and !e.expired);
        return n;
    }

    fn lruParked(s: *const Store) ?usize {
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (!e.parked or e.expired or e.borrowers != 0) continue;
            if (best == null or e.used < s.entries.items[best.?].used) best = i;
        }
        return best;
    }

    fn logPark(s: *const Store, at: u32, bytes: u64, ms: f64, reason: []const u8) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: rank {d}: parked a state at {d} tokens ({d} MiB to host, {d:.1} ms): {s}", .{ s.rank, at, bytes >> 20, ms, reason });
    }

    fn logParkDrop(s: *const Store, at: u32, bytes: u64, reason: []const u8) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: rank {d}: dropped a parked state at {d} tokens ({d} MiB): {s}", .{ s.rank, at, bytes >> 20, reason });
    }

    fn logParkResume(s: *const Store, at: u32, bytes: u64, ms: f64) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: rank {d}: resumed a parked state at {d} tokens ({d} MiB from host, {d:.1} ms)", .{ s.rank, at, bytes >> 20, ms });
    }

    /// Drops least recently used parked states until `need` fits. A borrowed parked state stays.
    fn makeTierRoom(s: *Store, need: u64) bool {
        while (s.tier_used + need > s.tier) {
            const i = s.lruParked() orelse return false;
            const at = s.entries.items[i].at;
            const bytes = s.entries.items[i].tier_bytes;
            s.counts.park_drops += 1;
            s.logParkDrop(at, bytes, "host tier full");
            s.remove(i);
        }
        return true;
    }

    /// Where `e` sits after a tier drop has moved the entries before it.
    fn indexOf(s: *const Store, e: *Entry) ?usize {
        for (s.entries.items, 0..) |item, i| if (item == e) return i;
        return null;
    }

    /// Parks the victim when the tier can take it. The entry stays, and it holds no pool page.
    fn tryPark(s: *Store, i: usize, reason: []const u8) bool {
        if (!s.tierOn()) return false;
        const e = s.entries.items[i];
        if (e.parked or e.borrowers != 0 or e.at < s.park_min or policy.extended(s, e)) return false;
        const cost = s.family.vtable.park_bytes.?(s.family.ptr, e.saved);
        if (cost == 0 or cost > s.tier) return false;
        if (!s.makeTierRoom(cost)) return false;
        const ms = s.family.vtable.park.?(s.family.ptr, e.saved) catch return false;
        e.parked = true;
        e.tier_bytes = cost;
        s.tier_used += cost;
        s.counts.parks += 1;
        s.logPark(e.at, cost, ms, reason);
        return true;
    }

    /// The victim leaves the pool. The tier parks it when it can. A superseded state is dropped.
    /// A failed park looks the victim up again: making tier room may have removed an earlier entry.
    fn depart(s: *Store, i: usize, prompt: []const u32, born: u32, bytes_short: bool) void {
        const e = s.entries.items[i];
        const reason = policy.victimReason(s, i, prompt, born);
        if (std.mem.indexOf(u8, reason, "superseded") == null and s.tryPark(i, reason)) return;
        const j = s.indexOf(e) orelse return;
        if (e.parked) {
            s.counts.park_drops += 1;
            s.logParkDrop(e.at, e.tier_bytes, reason);
        } else evicted(e.at, s.freedOf(e), reason);
        s.remove(j);
        s.noteEvict(whyOf(reason, bytes_short));
    }

    /// Parked states this new state supersedes leave now, even when the budget has room. A borrowed one stays.
    fn dropParkedSubsumed(s: *Store, prompt: []const u32, born: u32, at: u32) void {
        var i: usize = 0;
        while (i < s.entries.items.len) {
            const other = s.entries.items[i];
            if (!other.parked or !policy.subsumes(s, prompt, born, at, other)) {
                i += 1;
                continue;
            }
            s.counts.park_drops += 1;
            s.logParkDrop(other.at, other.tier_bytes, "superseded by the state this pass keeps");
            s.remove(i);
            s.noteEvict(.superseded);
        }
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.holding() -| s.spare();
    }

    /// Kept states free their storage, least valuable first and never `restoring`, until `left.fits()` (false: never).
    pub fn giveBack(s: *Store, restoring: ?Saved, prompt: []const u32, left: anytype) bool {
        s.collectExpired();
        while (!left.fits()) {
            const skip = for (s.entries.items) |e| {
                if (restoring != null and e.saved == restoring.?) break e;
            } else null;
            const born: u32 = @intCast(prompt.len);
            const i = s.victimBut(skip, prompt, born) orelse return false;
            s.depart(i, prompt, born, true);
        }
        return true;
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or s.startAt(starts, at) == at);
    }

    /// The chunk start at or before `at`: the request's, else the family's grid (0 when none).
    fn startAt(s: *const Store, starts: []const u32, at: u32) u32 {
        if (starts.len == 0 and s.rules.grid > 0) return at - at % s.rules.grid;
        return cuts.floorStart(starts, at);
    }

    /// The latest kept state this prompt can resume at `at` or under min_gap before it (null: none).
    fn keptNear(s: *const Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, at: u32) ?u32 {
        var best: ?u32 = null;
        for (s.entries.items) |e| {
            if (e.expired or e.at > at or at - e.at >= s.rules.min_gap or e.tokens.len > prompt.len or !s.usable(e.at, starts)) continue;
            if (!std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens) or !modes.equal(e.decode_spans, spans, e.at)) continue;
            if (best == null or e.at > best.?) best = e.at;
        }
        return best;
    }

    /// Find a kept token prefix with matching canonical spans through its position, leaving at least one row.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.expired or e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
            if (best != null and e.at <= best.?.at) continue;
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            if (!modes.equal(e.decode_spans, spans, e.at)) continue;
            best = e;
        }
        return best;
    }

    /// Restore the longest prefix matching tokens and canonical spans, then plan prompt-cache marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        return s.beginRewind(a, prompt, history_len, 0, shared, starts, owner, spans);
    }

    /// `begin`, also keeping a state before the latest user message (`rewind_len`, 0: none) that its edits resume.
    pub fn beginRewind(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        return s.beginUser(a, prompt, history_len, rewind_len, 0, shared, starts, owner, spans);
    }

    /// Begin with a known end of the latest user message, for its optional interior state (0: unknown).
    pub fn beginUser(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, user_end: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        const l = try s.lookupUser(a, prompt, @intCast(prompt.len), history_len, rewind_len, user_end, shared, starts, spans);
        const e = l.entry orelse return .{ .marks = l.marks };
        const failure: ?anyerror = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| null else |err| err;
        const from = if (failure == null) e.at else 0;
        s.resumedWithReason(e, prompt, failure);
        s.reserve(prompt, @intCast(prompt.len), null, l.marks, spans); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// Look up canonical-span state and marks without restoring; the caller later reports resumed.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupRewind(a, prompt, @intCast(prompt.len), history_len, 0, shared, starts, spans);
    }

    /// `lookup` plus a rewind state on the starts or grid (or one under min_gap before it); `born` as in keepBorn.
    pub fn lookupRewind(s: *Store, a: Allocator, prompt: []const u32, born: u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupUser(a, prompt, born, history_len, rewind_len, 0, shared, starts, spans);
    }

    /// `lookupRewind` with an explicit last-user end: history may also include later tool or assistant messages.
    pub fn lookupUser(s: *Store, a: Allocator, prompt: []const u32, born: u32, history_len: u32, rewind_len: u32, user_end: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        s.collectExpired();
        const endpoint = if (s.rules.planned) s.startAt(starts, history_len) else history_len;
        const rewind = if (s.rules.planned) s.startAt(starts, rewind_len -| s.rules.lookahead) else rewind_len -| s.rules.lookahead;
        const endpoint_bytes = if (endpoint > 0 and (endpoint < prompt.len or (s.rules.warm != .off and endpoint == prompt.len)) and endpoint + s.rules.lookahead <= prompt.len) s.family.vtable.bytes(s.family.ptr, endpoint) else 0;
        const rewind_bytes = if (rewind > 0 and rewind < prompt.len and rewind + s.rules.lookahead <= prompt.len and s.usable(rewind, starts)) s.family.vtable.bytes(s.family.ptr, rewind) else 0;
        const priority_bytes = if (endpoint_bytes <= s.budget) endpoint_bytes else 0;
        s.active_rewind_at = if (prompt.len >= s.rules.min_prompt and rewind_bytes > 0 and rewind != endpoint and rewind_bytes <= s.budget - priority_bytes) rewind else 0;
        if (s.active_rewind_at > 0) if (s.keptNear(prompt, starts, spans, s.active_rewind_at)) |at| {
            s.active_rewind_at = at; // a state kept under min_gap before it is the rewind state: no new copy
        };
        const e = rows.read.recall(s, prompt, born, starts, spans, s.recall(prompt, born, starts, spans, s.find(prompt, starts, spans)));
        if (e == null) s.counts.misses += 1;
        const from = if (e) |x| x.at else 0;
        const ordinary = if (s.active_rewind_at == 0)
            try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, 0, shared, starts, spans, if (e) |x| x.last else &.{}))
        else
            try s.fittingRewind(a, try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, s.active_rewind_at, shared, starts, spans, if (e) |x| x.last else &.{})), endpoint, s.active_rewind_at, endpoint_bytes);
        const known_end = if (user_end <= history_len) user_end else 0;
        const marks_ = interior.add(s, a, prompt, from, known_end, rewind_len, starts, spans, ordinary, endpoint, endpoint_bytes, e) catch |err| {
            a.free(ordinary);
            return err;
        };
        for (shared) |w| { // the shared cuts this pass keeps, and a short prompt's mark inside a shared block: other conversations resume them
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            for (marks_) |m| if (m == at or (m < at and !s.rules.planned and prompt.len < s.rules.min_prompt)) s.noteShared(prompt[0 .. m + s.rules.lookahead]);
        }
        s.reserve(prompt, born, e, marks_, spans);
        if (e) |entry| entry.borrowers += 1;
        return .{ .entry = e, .marks = marks_ };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, prompt: []const u32, born: u32, from: ?*Entry, marks_: []const u32, spans: []const modes.Span) void {
        var need: u64 = 0;
        var new: usize = 0; // states it keeps that the store lacks: a slot each
        var prev: u32 = if (from) |e| e.at else 0; // a sharing family's new state holds the storage before it already
        for (marks_) |at| {
            const n = @as(usize, at) + s.rules.lookahead;
            const exists = for (s.entries.items) |e| {
                if (!e.expired and e.at == at and e.tokens.len == n and n <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) break true;
            } else false;
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (!exists) need += if (s.family.vtable.held == null) bytes else bytes -| s.family.vtable.bytes(s.family.ptr, prev) + s.family.vtable.bytes(s.family.ptr, 0);
            new += @intFromBool(!exists);
            prev = @max(prev, at);
        }
        while (s.holding() + need > s.budget or s.occupancy() + @min(new, s.most) > s.most) {
            // the least recently used one leaves. A mark supersedes the conversation's older state only when the
            // state is kept (keepBorn): a supersede is irreversible, and this pass may cancel before it keeps it.
            // When shared pages make the bytes short, a state that frees nothing stays. The count alone keeps today's victim.
            // With the host tier on, a state that can park does, and a parked state does not count against most.
            const bytes_short = s.holding() + need > s.budget;
            const victim = s.victimFor(from, prompt, born, bytes_short) orelse return;
            s.depart(victim, prompt, born, bytes_short);
        }
    }

    /// Every kept state goes after a weight change, and learned states on disk stop being read: old arithmetic.
    pub fn clear(s: *Store) void {
        for (s.entries.items) |e| e.expired = true;
        s.collectExpired();
        s.shared_keys.clearRetainingCapacity();
        s.imprint = null;
        s.disk = null;
        s.active_rewind_at = 0;
    }

    /// Forget a failed peer state matching the token prefix and canonical spans through at.
    pub fn forget(s: *Store, prompt: []const u32, at: u32, spans: []const modes.Span) void {
        for (s.entries.items) |e| if (!e.expired and e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) and modes.equal(e.decode_spans, spans, at)) {
            s.counts.failed += 1;
            e.expired = true;
            return s.collectExpired();
        };
    }

    fn sharedKey(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    fn noteShared(s: *Store, tokens: []const u32) void {
        const k = sharedKey(tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.gpa, k) catch {};
    }

    const fitting = @import("prompt_cache_fit.zig").fitting;
    const fittingRewind = @import("prompt_cache_fit.zig").fittingRewind;

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        return s.resumedWithReason(e, prompt, if (ok) null else error.RestoreRejected);
    }

    pub fn resumedWithReason(s: *Store, e: *Entry, prompt: []const u32, failure: ?anyerror) void {
        if (std.mem.indexOfScalar(*Entry, s.entries.items, e) == null) return;
        defer s.release(e);
        if (failure) |err| {
            @import("prompt_cache_restore.zig").report(s.restore_log, .{ .kept = e.at, .prompt = prompt.len, .expired = e.expired, .err = err });
            s.counts.failed += 1;
            e.expired = true;
            return;
        }
        s.clock += 1;
        e.used = s.clock;
        e.branched = e.branched or s.extended(e); // a turn several continuations share (a re-sent, edited or forked one)
        s.counts.hits += 1;
        if (e.parked) {
            s.counts.park_resumes += 1;
            const ms = if (s.family.vtable.copy_ms) |f| f(s.family.ptr) else 0;
            s.logParkResume(e.at, e.tier_bytes, ms);
        }
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        return s.marksRewind(a, prompt, from, history_len, 0, shared, starts, &.{}, previous);
    }

    /// `marks` with the active rewind second: kept however near the other marks, never at or before the resume point.
    fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, previous: []const u32) ![]const u32 {
        const short = !s.rules.planned and prompt.len < s.rules.min_prompt; // the history's mark only, where the pass cuts a chunk anyway
        if (short and s.rules.chunk == 0) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        // a warm pass (its history the whole prompt) keeps where a state can be keyed, its lookahead before the end
        const warm_pass = s.rules.warm != .off and history_len == prompt.len and s.rules.lookahead > 0;
        const history: u32 = if (warm_pass) @intCast(prompt.len -| s.rules.lookahead) else history_len;
        try want.append(a, if (short) cuts.chunkEnd(s.rules.chunk, from, history, prompt.len, starts, spans) else history);
        if (rewind > 0 and s.active_rewind_at == rewind) try want.append(a, rewind);
        if (previous.len > 0 and !short) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        if (!short) try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            const is_rewind = rewind > 0 and k == 1 and at == rewind;
            if (!is_rewind and !(k == 0 and warm_pass) and at - from < s.rules.min_gap and (k > 0 or (from > 0 and s.rules.warm == .on))) continue; // near the resume point
            if (k > 0 and !is_rewind) { // the history's mark always (warm families: away from the resume point); the others away from it and each other
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// Keep the state at `at` with its span prefix, evicting within the budget; `starts` are a learned cut's chunks.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
        return s.keepBorn(prompt, @intCast(prompt.len), at, owner, starts, spans);
    }

    /// `keep` as a prompt of `born` tokens: a reply's end and its warm pass supersede none of their request's states.
    pub fn keepBorn(s: *Store, prompt: []const u32, born: u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
        s.collectExpired();
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
        for (s.entries.items) |e| if (!e.expired and e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) {
            e.used = s.clock; // the same state again: no copy
            e.shared = e.shared or shared;
            if (shared) learned.learn(s, e, starts);
            return true;
        };
        const decoded = modes.prefix(s.gpa, spans, at) catch |err| return s.fail(at, err);
        var owns_decoded = true;
        defer if (owns_decoded) s.gpa.free(decoded);
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget or s.most == 0) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (s.occupancy() < s.most and f(s.family.ptr, at)) break;
            const free_ = s.spare();
            const adds = if (s.family.vtable.adds) |f| f(s.family.ptr, owner, at) else bytes;
            const bytes_short = s.holding() + free_ + adds > s.budget;
            const count_short = s.occupancy() >= s.most;
            if (!bytes_short and !count_short) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            // the conversation's own older state it extends gives its storage before the least recently used one.
            // When shared pages make the bytes short, a state that frees nothing stays. The count alone keeps today's victim.
            const victim = s.victimFor(null, prompt, born, bytes_short) orelse return s.fail(at, error.CacheBorrowed);
            const superseded = policy.subsumed(s, prompt, born, at);
            if (superseded) |i| {
                if (i != victim) {
                    const other = s.entries.items[i];
                    const stays = !other.parked and bytes_short and s.sharingBytes() and s.family.vtable.frees.?(s.family.ptr, other.saved) == 0;
                    if (!stays) {
                        if (other.parked) {
                            s.counts.park_drops += 1;
                            s.logParkDrop(other.at, other.tier_bytes, "superseded by the state this pass keeps");
                        } else evicted(other.at, s.freedOf(other), "superseded by the state this pass keeps");
                        s.remove(i); // the loop rechecks the budget; the least recently used one leaves only if the room is still short
                        s.noteEvict(.superseded);
                        continue;
                    }
                }
            }
            s.depart(victim, prompt, born, bytes_short); // a superseded victim is dropped; another may park
        }
        s.dropParkedSubsumed(prompt, born, at);
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .tokens = tokens, .decode_spans = decoded, .at = at, .saved = saved, .bytes = charged, .born = born, .used = s.clock, .last = last, .shared = shared };
        owns_decoded = false;
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.held += charged;
        s.counts.kept += 1;
        if (shared) learned.learn(s, e, starts);
        if (s.disk) |r| rows.write.queued(r, e, starts);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only inside what the budget leaves
        return true;
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        if (!modes.equal(&.{}, spans, m.at)) return have; // the request decodes rows the learned prompt pass prefilled
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and (s.holding() + s.spare() + bytes > s.budget or s.occupancy() >= s.most)) {
            const bytes_short = s.holding() + s.spare() + bytes > s.budget;
            const victim = s.victimFor(have, prompt, born, bytes_short) orelse return have;
            s.depart(victim, prompt, born, bytes_short);
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = born, .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            learned.unlearn(s, im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
        s.held += bytes;
        if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
        return e;
    }

    /// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
    fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
        if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
        return have;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        const line = "prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, {d} for streams, refused {d}, failed {d})";
        const tail = ", parked {d} states, {d} MiB of {d} MiB";
        if (s.family.vtable.pages) |f| {
            const n = f(s.family.ptr);
            if (s.tier > 0) {
                std.log.info(line ++ ", pages held {d} of {d}, free {d}" ++ tail, .{ prompt, from, kept, s.entries.items.len, s.holding() >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.forced, s.counts.refused, s.counts.failed, n.held, n.pool, n.free, s.parkedStates(), s.tier_used >> 20, s.tier >> 20 });
            } else std.log.info(line ++ ", pages held {d} of {d}, free {d}", .{ prompt, from, kept, s.entries.items.len, s.holding() >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.forced, s.counts.refused, s.counts.failed, n.held, n.pool, n.free });
        } else if (s.tier > 0) {
            std.log.info(line ++ tail, .{ prompt, from, kept, s.entries.items.len, s.holding() >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.forced, s.counts.refused, s.counts.failed, s.parkedStates(), s.tier_used >> 20, s.tier >> 20 });
        } else std.log.info(line, .{ prompt, from, kept, s.entries.items.len, s.holding() >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.forced, s.counts.refused, s.counts.failed });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip`: a superseded one, no branch point, oldest; else the oldest, rewind last.
    pub const victimBut = policy.victimBut;
    const extended = policy.extended;

    /// The state at `at` of the `born`-token `prompt` supersedes the entry: a shorter state of the same conversation it extends.
    /// A rewind (a later turn's shorter state) and a reply's end (past `born`) supersede none.
    pub fn subsumes(s: *const Store, prompt: []const u32, born: u32, at: u32, e: *const Entry) bool {
        return policy.subsumes(s, prompt, born, at, e);
    }

    /// The first entry the state at `at` supersedes, or null.
    pub fn subsumed(s: *const Store, prompt: []const u32, born: u32, at: u32) ?usize {
        return policy.subsumed(s, prompt, born, at);
    }

    /// The reason the victim `victimBut` chose left: the prompt or a kept state extends it, else least recently used.
    pub fn victimReason(s: *const Store, i: usize, prompt: []const u32, born: u32) []const u8 {
        return policy.victimReason(s, i, prompt, born);
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
pub fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// One log line for a state the store evicted: where it sat, what it held, why it left.
pub fn evicted(at: u32, bytes: u64, reason: []const u8) void {
    if (@import("builtin").is_test) return;
    std.log.info("prompt cache: evicted a state at {d} tokens ({d} MiB): {s}", .{ at, bytes >> 20, reason });
}

test {
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_learn_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("prompt_cache_interior_test.zig");
    _ = @import("prompt_cache_branch_test.zig");
    _ = @import("prompt_cache_keepone_test.zig");
    _ = @import("prompt_cache_most_test.zig");
    _ = @import("prompt_cache_give_test.zig");
    _ = @import("prompt_cache_shared_test.zig");
    _ = @import("prompt_cache_lifetime_test.zig");
    _ = @import("prompt_cache_pages_test.zig");
    _ = @import("prompt_cache_park_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = modes;
    _ = rows;
}

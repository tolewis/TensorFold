//! Exact prompt reuse for any family and backend: states kept at prompt-pass chunk ends, found by their tokens.
//!
//! The durable tier (`Tier`) and continuation grouping (`join`) port wojo's GLM 0088 (PR #78), 0110 (PR #124;
//! BadAd84's block-tier design) and 0111 (PR #128). Every hook is off until a tier is set (`setTier`) or grouping is
//! turned on (`groups`); with both off the store behaves and counts exactly as before.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");

/// A family's copy of one state.
pub const Saved = *anyopaque;

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
        /// Write `saved` as learned state `key`, in files of the family's own under `dir` (null: nothing kept on disk).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void = null,
        /// Learned state `key` of `at` tokens, read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
        /// Whether `saved` still restores for a prompt of `len` tokens (not stale: the family's own rule); null: always.
        /// dsv41 hands Lanes.usable here with its tier (snaps list membership, `stale`, its grid and window).
        /// Lookups pass the rest over before comparing lengths, so a stale long state never shadows a usable one.
        usable: ?*const fn (ptr: *anyopaque, saved: Saved, len: u32) bool = null,
    };
};

// -- the durable tier (spill v2) --------------------------------------------------------------------------------------
// A Tier keeps kept states past the memory budget and past the process (dsv41's spill_tier glues core/spill's per-rank
// block store and the family's packing to it). The store calls it on the engine thread only and never waits on disk:
// `lookup` reads the tier's own index, a restore is a job polled between rounds and prompt chunks, a write is enqueued
// at keep and staged by `pump`. All-rank agreement (presence, validation, commit) is the family's, inside `begin` and
// `poll`: a job is ready only once every rank installed its copy, and failed everywhere when any rank's copy fails.
// families/dsv41/spill_tier.zig implements this vtable over core/spill's store and spill_state's packing; the wire
// commands (has, reserve, copied-block boundary, validate, commit/abort) ride rank_wire's ordered stream inside
// begin/poll/cancel, and native/cuda.zig openWith calls Store.setTier with the backend's tier, grid, window and
// Snapshots.usable.

/// A durable point the tier holds and found for a prompt (its ids are the store's keyed ids, pictures' keys included).
pub const Point = struct {
    /// The tier's name for the point while it lives (a digest of the point on disk).
    id: u64,
    /// Prompt tokens the state stands after, and the ids it covers (`at` plus the store's lookahead).
    at: u32,
    tokens: u32,
    /// Memory a restored copy takes in the store (0: the family's `bytes(at)`).
    bytes: u64 = 0,
    /// The conversation that wrote it and whether it is a shared prefix (persisted with the point).
    group: u64 = 0,
    shared: bool = false,
};

/// A durable lookup: the longest point whose ids are a prefix of `key`, `beyond < at <= max_at`, and on `grid` when
/// set (the native eligibility rule; the store checks the answer again).
pub const Want = struct {
    key: []const u32,
    beyond: u32,
    max_at: u32,
    grid: u32 = 0,
    lookahead: u32 = 0,
};

/// A kept state the tier writes (enqueued at keep). `saved` stays the family's: the tier reads it until `release`.
pub const Kept = struct {
    saved: Saved,
    /// Its ids (`at` plus lookahead), valid during the call only.
    key: []const u32,
    at: u32,
    bytes: u64,
    group: u64 = 0,
    shared: bool = false,
    images: bool = false,
    /// The point its rows' computation forks from (the state its pass resumed from; null: a fresh prefill, or a resumed
    /// state that was never written), and where.
    parent: ?u64 = null,
    fork: u32 = 0,
    /// The computation its rows belong to (0110's lineage; block names are salted with it), valid during the call:
    /// set whether or not the resumed state was ever written, so sibling forks name their shared blocks alike.
    lineage: []const Fork = &.{},
};

/// One step of a state's lineage: computation `name` produced the rows from `at` on (the first step's `at` is 0). A
/// pass that resumes a state at `at` forks that state's lineage there (GLM 0110 `fork`: the same names for the blocks
/// before `at`, its own from the block holding `at` on); a fresh prefill starts a lineage of its own.
pub const Fork = struct { name: u64, at: u32 };

/// A restore job after a poll.
pub const Poll = union(enum) {
    /// Still reading or copying: blocks copied so far of its total.
    pending: struct { done: u64 = 0, total: u64 = 0 },
    /// Installed on every rank: a family state the store adopts (it owns it from here; `bytes` of memory, 0: bytes(at)),
    /// and the point's lineage (valid during the call; the store copies it).
    ready: struct { saved: Saved, bytes: u64 = 0, lineage: []const Fork = &.{} },
    /// Aborted everywhere (absent on a rank, CRC/size, a guard): the tier released what it reserved.
    failed: anyerror,
};

/// The tier's own counters (core/spill fills them; totals unless marked a gauge).
pub const TierCounts = struct {
    points: u64 = 0, // gauge: published points
    blocks: u64 = 0, // gauge: unique blocks on disk
    stored_bytes: u64 = 0, // gauge: unique bytes on disk (blocks, point files, temporaries)
    cap_bytes: u64 = 0, // gauge: --spill-gib
    disk_free_bytes: u64 = 0, // gauge: free bytes on the store's file system
    staging_bytes: u64 = 0, // gauge: host staging held now (writes, reads, small states)
    staging_peak: u64 = 0, // gauge: its peak
    writes_pending: u64 = 0, // gauge: queued points not yet published
    writes_queued: u64 = 0,
    writes_committed: u64 = 0,
    writes_cancelled: u64 = 0, // unpublished points cancelled, for any of the reasons below
    writes_cancelled_guard: u64 = 0, // their source was about to change and staging could not finish in budget
    writes_cancelled_staging: u64 = 0, // the staging cap had no room
    writes_cancelled_shutdown: u64 = 0, // the flush deadline passed
    writes_failed: u64 = 0,
    forced_staging: u64 = 0, // ranges staged at once because their source was about to change
    no_space: u64 = 0, // writes refused by the cap or the free-disk floor
    dedup_blocks: u64 = 0,
    dedup_bytes: u64 = 0,
    written_bytes: u64 = 0,
    presence_refused: u64 = 0, // a rank lacked the point: no rank restored it
    compatible_misses: u64 = 0, // lookups with no compatible point
    crc_errors: u64 = 0,
    size_errors: u64 = 0, // truncated or oversized files, manifest bounds broken
    reads: u64 = 0,
    read_bytes: u64 = 0,
    read_failed: u64 = 0,
    evicted_turns: u64 = 0,
    evicted_forks: u64 = 0,
    evicted_conversations: u64 = 0,
    evicted_shared: u64 = 0,
    quota_trims: u64 = 0,
    recovered_points: u64 = 0, // published points found at start
    dropped_temporaries: u64 = 0, // incomplete records removed at start

    pub const gauges = [_][]const u8{ "points", "blocks", "stored_bytes", "cap_bytes", "disk_free_bytes", "staging_bytes", "staging_peak", "writes_pending" };

    /// Each field's /metrics help (a missing one fails the build).
    pub const help = .{
        .points = "Published points in this rank's disk tier.",
        .blocks = "Unique blocks on this rank's disk.",
        .stored_bytes = "Unique bytes the disk tier holds (blocks, point files, temporaries).",
        .cap_bytes = "The disk tier's cap (--spill-gib).",
        .disk_free_bytes = "Free bytes on the disk tier's file system.",
        .staging_bytes = "Host staging the disk tier holds now.",
        .staging_peak = "The most host staging the disk tier has held.",
        .writes_pending = "Kept states queued for the disk, not yet published.",
        .writes_queued = "Kept states queued for the disk.",
        .writes_committed = "Points published on disk (manifest last).",
        .writes_cancelled = "Unpublished points cancelled (a guard could not stage, staging full, shutdown).",
        .writes_cancelled_guard = "Unpublished points cancelled because their source was about to change.",
        .writes_cancelled_staging = "Unpublished points cancelled because staging had no room.",
        .writes_cancelled_shutdown = "Unpublished points cancelled at the flush deadline.",
        .writes_failed = "Disk writes that failed.",
        .forced_staging = "Ranges staged at once because their source was about to change.",
        .no_space = "Writes refused by the cap or the free-disk floor.",
        .dedup_blocks = "Blocks linked to equal bytes already on disk.",
        .dedup_bytes = "Bytes those links saved.",
        .written_bytes = "Bytes written to disk.",
        .presence_refused = "Restores refused because a rank lacked the point.",
        .compatible_misses = "Durable lookups with no compatible point.",
        .crc_errors = "Blocks or sections that failed their CRC.",
        .size_errors = "Files truncated or past their manifest's bounds.",
        .reads = "Points read for restores.",
        .read_bytes = "Bytes read for restores.",
        .read_failed = "Restore reads that failed.",
        .evicted_turns = "Old turns evicted from disk.",
        .evicted_forks = "Abandoned forks evicted from disk.",
        .evicted_conversations = "Whole conversations evicted from disk.",
        .evicted_shared = "Shared-prefix points evicted from disk.",
        .quota_trims = "Writes that trimmed their conversation to its quota.",
        .recovered_points = "Published points found at start.",
        .dropped_temporaries = "Incomplete records removed at start.",
    };
};

/// Tier.begin's refusals for capacity (robustness): nothing was reserved and the same begin may succeed once a
/// restore in flight ends. dsv41's spill tier: TooManyRestores (its cap of restores in flight); core/spill's store:
/// TooMany (its reads in flight), NoSlot (no staging slot); any tier: Busy.
/// error.TierFull: the store's own refusal when the tier reports itself full (Tier.VTable.full) before any room is made.
pub const capacity_errors = [_]anyerror{ error.TierFull, error.TooManyRestores, error.TooMany, error.NoSlot, error.Busy };

/// The ranks a tier reports counters for (TP ranks; more are not reported).
pub const max_ranks = 8;

/// Each rank's TierCounts (rank 0 first): `n` of `ranks` are filled.
pub const RankCounts = struct {
    n: u32 = 0,
    ranks: [max_ranks]TierCounts = @splat(.{}),

    pub fn slice(r: *const RankCounts) []const TierCounts {
        return r.ranks[0..r.n];
    }
};

/// Lifetime and cost (the store relies on both): the tier outlives the Store (Store.deinit cancels restores and
/// releases written entries through it, so the backend deinitializes the tier after the store). `lookup` and
/// `counts` run on every admission and every round: O(1)-ish, from memory, no I/O and no CUDA. `poll`, `pump` and
/// `release` run between rounds, between a fill's prompt chunks (inside Stream.between, with any Engram read-ahead
/// already joined) and inside the lone driver's commit hook (`pump` only), so each must be bounded per call: by its
/// `budget_ns` and by a block cap of its own (GLM's PUMP_BLOCKS).
pub const Tier = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The longest durable point `want` allows, from the tier's own index (no disk I/O, no CUDA); null: none.
        lookup: *const fn (ptr: *anyopaque, want: *const Want) ?Point,
        /// Start restoring `point` into a new family state (reserve on every rank, then read): a job for `poll`.
        /// A refusal for capacity that a later call may not meet (too many restores in flight, no read slot) is one
        /// of `capacity_errors` with nothing reserved: the host then defers the request briefly and asks again
        /// (Store.deferrable), instead of prefilling a prompt a restore would have served.
        begin: *const fn (ptr: *anyopaque, point: *const Point) anyerror!u64,
        /// Whether `begin` would refuse for capacity now (the restores in flight at the tier's cap, no read slot), asked
        /// before the store makes room for a restore, so a refusal costs no kept state (error.TierFull, nothing evicted).
        /// Null: never full (a capacity refusal from `begin` then comes after the room was made).
        full: ?*const fn (ptr: *anyopaque) bool = null,
        /// Progress the job by at most `budget_ns` of engine-thread copies; after ready or failed the job is gone.
        poll: *const fn (ptr: *anyopaque, job: u64, budget_ns: u64) Poll,
        /// Stop a pending job: reads stop, issued DMA is waited for, reservations go (called once per job at most).
        cancel: *const fn (ptr: *anyopaque, job: u64) void,
        /// Enqueue the write of a state just kept: the point's name (null: not written, e.g. under the minimum).
        keep: *const fn (ptr: *anyopaque, kept: *const Kept) ?u64,
        /// The point's memory twin is about to be dropped: stage what the write still reads from `saved`, or cancel it.
        release: ?*const fn (ptr: *anyopaque, point: u64, saved: Saved) void = null,
        /// The point became a shared prefix: persist the flag (atomically; never resurrecting a deleted point).
        share: ?*const fn (ptr: *anyopaque, point: u64) void = null,
        /// The point's state was resumed: its last use, for eviction order.
        touch: ?*const fn (ptr: *anyopaque, point: u64) void = null,
        /// The point is unusable (a peer failed it, its install failed): drop it.
        forget: *const fn (ptr: *anyopaque, point: u64) void,
        /// Stage queued writes for at most `budget_ns` (engine thread: between rounds and chunks, and at idle).
        pump: ?*const fn (ptr: *anyopaque, budget_ns: u64) void = null,
        /// Whether writes still wait for `pump` (the host keeps servicing while idle).
        busy: ?*const fn (ptr: *anyopaque) bool = null,
        /// Clean shutdown: stage and drain within `budget_ns`; returns the points left unpublished.
        flush: *const fn (ptr: *anyopaque, budget_ns: u64) u64,
        /// Each rank's counters into `out` (rank 0 first; the peers' as their last replies carried them): how many.
        counts: *const fn (ptr: *anyopaque, out: []TierCounts) usize,
    };
};

/// A restore in flight: one job per point, shared by every request waiting for it (0110's duplicate-load avoidance).
pub const Restore = struct {
    point: Point,
    job: u64,
    key: []u32, // the point's ids (store-owned)
    done: u64 = 0, // blocks copied so far of `total` (its last poll)
    total: u64 = 0,
    bytes: u64 = 0, // the memory its destination holds from begin (charged to the budget until it ends)
};

/// How a polled restore stands.
pub const Restored = enum { pending, adopted, dropped, failed };

/// The store's own counters for the tier and continuation (totals).
pub const Stats = struct {
    disk_lookups: u64 = 0,
    disk_hits: u64 = 0, // lookups that found a durable point longer than the best usable memory state
    disk_hits_committed: u64 = 0, // restores adopted as memory states (every rank agreed)
    restores_begun: u64 = 0,
    restores_shared: u64 = 0, // requests that joined a restore already in flight
    restores_failed: u64 = 0,
    restores_cancelled: u64 = 0,
    restores_dropped: u64 = 0, // adopted nothing: the budget cannot hold it, or memory had it meanwhile
    restores_refused: u64 = 0, // not begun: the budget could not hold its destination, or the tier was full (once a request)
    restored_tokens: u64 = 0,
    writes_enqueued: u64 = 0,
    shared_promoted: u64 = 0,
    flushes: u64 = 0,
    flush_timeouts: u64 = 0,
    flush_unpublished: u64 = 0,
    chats_continued: u64 = 0,
    chats_continued_prefill_tokens: u64 = 0,
};

// -- continuation (0111) ----------------------------------------------------------------------------------------------

/// A conversation's first reply: the tokens of it a later prompt must repeat to continue that conversation.
pub const chat_match: u32 = 32;
/// A continued conversation's prefill from this many tokens on is logged.
pub const chat_log_tokens: u32 = 4096;

/// The conversation a request's kept states belong to (Store.join).
pub const Join = struct {
    group: u64 = 0, // 0: grouping off
    /// It continues `group` without resuming that conversation's own turn state (0111's counted case).
    continued: bool = false,
    /// The matched state's ids (`at` plus lookahead), and where the prompt left its evidence (the last prompt that kept
    /// or resumed it): `back < prior_len` is a rewind before the state itself.
    prior_len: u32 = 0,
    back: u32 = 0,
    /// What the lookup resumed from.
    source: Source = .nothing,

    pub const Source = enum { nothing, shared, other };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call costs more than a later turn's reuse saves.
    min_prompt: u32 = 4096,
    /// Rows a resumed pass must still run past its state (DeepSeek's sliding window): a state nearer the end is passed over.
    tail: u32 = 0,
    /// The family's own resume grid (dsv41: its 2048-row chunk; 0: none): durable points and shared marks stand on it.
    grid: u32 = 0,
    /// The rows a resumed pass replays at its end (dsv41: the window): a state is usable only this far from the end.
    window: u32 = 0,

    /// The native rule (Lanes.usable): 0 < at < len, on the grid, at most len - window, its lookahead inside the prompt.
    pub fn eligible(r: Rules, at: u32, len: usize) bool {
        if (at == 0 or at >= len) return false;
        if (r.grid > 0 and at % r.grid != 0) return false;
        return @as(usize, at) + r.window <= len and @as(usize, at) + r.lookahead <= len;
    }

    /// The grid point at or before `at` (`at` itself without a grid).
    pub fn floorGrid(r: Rules, at: u32) u32 {
        return if (r.grid == 0) at else at / r.grid * r.grid;
    }

    /// The largest position `eligible` allows for a prompt of `len` tokens (null: none).
    pub fn maxAt(r: Rules, len: usize) ?u32 {
        if (len < 2) return null;
        var hi: usize = len - 1;
        hi = @min(hi, len -| @max(r.window, r.lookahead));
        const at = r.floorGrid(@intCast(@min(hi, std.math.maxInt(u32))));
        return if (r.eligible(at, len)) at else null;
    }
};

pub const Entry = struct {
    tokens: []u32, // the state's tokens and `lookahead` more
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    group: u64 = 0, // the conversation that kept it (0111; 0: grouping off)
    point: ?u64 = null, // its durable point in the tier (written at keep, or restored from)
    images: bool = false, // its key holds pictures' keys
    restored: bool = false, // read back from the tier: not a turn's own state (it counts against no conversation)
    lineage: []Fork = &.{}, // the computation its rows belong to (store-owned; empty without a tier)
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

/// A side-effect-free look at what a prompt would resume: the best usable memory state, and a longer durable point.
pub const Probe = struct { memory: ?*Entry = null, durable: ?Point = null };

/// What a keep records beyond the state: its conversation, pictures and lineage (the tier writes them with it).
pub const KeepInfo = struct {
    group: u64 = 0,
    images: bool = false,
    parent: ?u64 = null,
    fork: u32 = 0,
    lineage: []const Fork = &.{}, // the pass's (Store.passLineage), copied into the entry
};

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    tier: ?Tier = null, // spill v2: kept states written at keep, read back by restore jobs (setTier)
    groups: bool = false, // 0111: kept states carry their conversation (setTier turns it on)
    next_group: u64 = 0,
    roots: std.AutoHashMapUnmanaged(u64, u32) = .empty, // each conversation's first prompt length (join)
    restoring: std.ArrayList(*Restore) = .empty,
    restoring_bytes: u64 = 0, // destinations the restores in flight reserved (charged like kept states)
    refused_for: Refusal = .none, // why the last restoreBeginOver returned NoRoom (deferrable)
    next_lineage: u64 = 0,
    stats: Stats = .{},

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        while (s.restoring.items.len > 0) s.restoreCancel(s.restoring.items[s.restoring.items.len - 1]);
        s.restoring.deinit(s.gpa);
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.roots.deinit(s.gpa);
        s.held = 0;
    }

    /// The durable tier from here (null: none); a tier turns continuation grouping on (its quotas count conversations).
    /// Conversation ids continue from `group_seed << 32` so they do not repeat a previous process's on disk.
    pub fn setTier(s: *Store, tier: ?Tier, group_seed: u32) void {
        s.tier = tier;
        if (tier != null) {
            s.groups = true;
            if (s.next_group == 0) s.next_group = @as(u64, group_seed) << 32;
            if (s.next_lineage == 0) s.next_lineage = @as(u64, group_seed) << 32;
        }
    }

    fn free(s: *Store, e: *Entry) void {
        if (e.point) |p| if (s.tier) |t| if (t.vtable.release) |f| f(t.ptr, p, e.saved); // a write still reading it stages first
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.free(e.lineage);
        s.gpa.destroy(e);
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.held -= e.bytes;
        s.free(e);
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.restoring_bytes -| s.spare();
    }

    /// Room for `bytes` of new storage beside the kept states, the restores in flight and the spare storage: spare that
    /// is in the way trimmed first, then states evicted (never `skip`; `prompt`'s superseded ones first), trimming
    /// again after each since a dropped state's storage may come back spare. False: the budget cannot hold it.
    fn makeRoom(s: *Store, bytes: u64, skip: ?*Entry, prompt: []const u32) bool {
        // never evict for room that evicting everything evictable could not make
        if (s.restoring_bytes + bytes + (if (skip) |e| e.bytes else 0) > s.budget) return false;
        while (true) {
            const free_ = s.spare();
            if (s.held + s.restoring_bytes + free_ + bytes <= s.budget) return true;
            if (free_ > 0) if (s.family.vtable.trim) |f| {
                f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes -| bytes);
                if (s.spare() < free_) continue;
            };
            s.remove(s.victimBut(skip, prompt) orelse return false);
            s.counts.evicted += 1;
        }
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or std.mem.indexOfScalar(u32, starts, at) != null);
    }

    /// Whether a kept state still restores for a prompt of `len` tokens: the family's own rule, then the native one
    /// (both only where set: a family without them keeps the old behaviour).
    fn live(s: *const Store, e: *const Entry, len: usize) bool {
        if (s.family.vtable.usable) |f| if (!f(s.family.ptr, e.saved, @intCast(@min(len, std.math.maxInt(u32))))) return false;
        if (s.rules.grid == 0 and s.rules.window == 0) return true;
        return s.rules.eligible(e.at, len);
    }

    /// The longest kept state `prompt` resumes exactly: its tokens a prefix of the prompt, at least one row and `tail` left.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.tokens.len > prompt.len or e.at >= prompt.len or @as(usize, e.at) + s.rules.tail > prompt.len or !s.usable(e.at, starts)) continue;
            if (best != null and e.at <= best.?.at) continue;
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            if (!s.live(e, prompt.len)) continue; // stale or ineligible: never shadows a shorter usable state or a point
            best = e;
        }
        return best;
    }

    /// What `key` would resume, changing nothing in memory: the best usable kept state, and the tier's longer point.
    pub fn probe(s: *Store, key: []const u32, starts: []const u32) Probe {
        const m = s.find(key, starts);
        const t = s.tier orelse return .{ .memory = m };
        const have: u32 = if (m) |e| e.at else 0;
        s.stats.disk_lookups += 1;
        const max_at = s.rules.maxAt(key.len) orelse return .{ .memory = m };
        if (max_at <= have) return .{ .memory = m };
        const want: Want = .{ .key = key, .beyond = have, .max_at = max_at, .grid = s.rules.grid, .lookahead = s.rules.lookahead };
        const p = t.vtable.lookup(t.ptr, &want) orelse return .{ .memory = m };
        if (!s.pointFits(p, key, starts, have)) return .{ .memory = m };
        s.stats.disk_hits += 1;
        return .{ .memory = m, .durable = p };
    }

    /// A durable answer checked again: past the memory hit, eligible here, its ids inside the key, inside the budget.
    fn pointFits(s: *const Store, p: Point, key: []const u32, starts: []const u32, have: u32) bool {
        if (p.at <= have or @as(u64, p.at) + s.rules.lookahead != p.tokens or p.tokens > key.len) return false;
        if (!s.rules.eligible(p.at, key.len) or !s.usable(p.at, starts)) return false;
        return s.pointBytes(p) <= s.budget;
    }

    fn pointBytes(s: *const Store, p: Point) u64 {
        return if (p.bytes > 0) p.bytes else s.family.vtable.bytes(s.family.ptr, p.at);
    }

    /// A restore of `p` (found by `probe` for `key`): the job already in flight for it, or a new one. A new job's
    /// destination is charged to the budget from here until it ends (room made first, as for a keep; nobody holds a
    /// looked-up entry at admission, so any state may go): error.NoRoom when the budget cannot hold it.
    pub fn restoreBegin(s: *Store, p: Point, key: []const u32) !*Restore {
        return s.restoreBeginOver(p, key, .{});
    }

    /// How restoreBeginOver treats the prompt's own memory state and counts its refusals.
    pub const Over = struct {
        /// The best usable memory state for the key (Probe.memory): kept while the budget holds it beside the restore.
        have: ?*Entry = null,
        /// The same request asking again (a deferred restore): a refusal is not counted again.
        again: bool = false,
    };

    /// Why a restore's room was refused (NoRoom): restores in flight hold the budget's rest (their charge goes when
    /// they end, so asking again may succeed), or the budget cannot hold the destination however long one waits.
    pub const Refusal = enum { none, restores_in_flight, budget };

    /// restoreBegin for a request whose best memory state is `o.have`. A full tier (Tier.VTable.full) refuses first,
    /// with error.TierFull and nothing evicted, so the shorter state is still there to resume. Otherwise the room
    /// spares `have` when the budget holds both it and the restore (a refusal by the tier then leaves it), and evicts
    /// it as restoreBegin does when the budget cannot hold both: a restore that can start always beats the shorter
    /// state. After a refusal `have` may be gone: look it up again (Store.find), never through the old pointer.
    pub fn restoreBeginOver(s: *Store, p: Point, key: []const u32, o: Over) !*Restore {
        for (s.restoring.items) |r| if (r.point.id == p.id) {
            s.stats.restores_shared += 1;
            return r;
        };
        const t = s.tier orelse return error.NoTier;
        if (p.tokens > key.len) return error.BadPoint;
        if (t.vtable.full) |full| if (full(t.ptr)) {
            if (!o.again) s.stats.restores_refused += 1;
            return error.TierFull;
        };
        const bytes = s.pointBytes(p);
        s.refused_for = .none;
        const spare_have = if (o.have) |e| s.restoring_bytes + bytes + e.bytes <= s.budget else false;
        const made = s.makeRoom(bytes, if (spare_have) o.have else null, key) or (spare_have and s.makeRoom(bytes, null, key));
        if (!made) {
            if (!o.again) s.stats.restores_refused += 1;
            // the room failed with nothing spared: only what the restores in flight hold stands in the way
            s.refused_for = if (s.restoring.items.len > 0 and bytes <= s.budget) .restores_in_flight else .budget;
            return error.NoRoom;
        }
        const r = try s.gpa.create(Restore);
        errdefer s.gpa.destroy(r);
        const k = try s.gpa.dupe(u32, key[0..p.tokens]);
        errdefer s.gpa.free(k);
        try s.restoring.ensureUnusedCapacity(s.gpa, 1);
        const job = try t.vtable.begin(t.ptr, &p);
        r.* = .{ .point = p, .job = job, .key = k, .bytes = bytes };
        s.restoring.appendAssumeCapacity(r);
        s.restoring_bytes += bytes;
        s.stats.restores_begun += 1;
        return r;
    }

    /// Whether a refused restoreBeginOver may succeed if asked again shortly: a refusal for capacity
    /// (capacity_errors), or a NoRoom only the restores in flight caused (their charge goes when they end). Anything
    /// else (a missing point, a damaged one, no tier, a budget that can never hold it) will not change by waiting.
    pub fn deferrable(s: *const Store, err: anyerror) bool {
        if (err == error.NoRoom) return s.refused_for == .restores_in_flight and s.restoring.items.len > 0;
        return std.mem.indexOfScalar(anyerror, &capacity_errors, err) != null;
    }

    /// Progress `r` by at most `budget_ns`; once it is no longer pending `r` is freed. An installed state becomes a
    /// shared or restored memory entry that the next lookup of its prompt finds (adopted).
    pub fn restorePoll(s: *Store, r: *Restore, budget_ns: u64) Restored {
        const t = s.tier.?;
        switch (t.vtable.poll(t.ptr, r.job, budget_ns)) {
            .pending => |p| {
                r.done = p.done;
                r.total = p.total;
                return .pending;
            },
            .failed => |err| {
                s.stats.restores_failed += 1;
                note("restoring {d} tokens from disk failed ({s}); its prompt prefills", .{ r.point.at, @errorName(err) });
                s.endRestore(r);
                return .failed;
            },
            .ready => |rd| {
                const ok = s.adopt(r, rd.saved, rd.bytes, rd.lineage);
                s.endRestore(r);
                return if (ok) .adopted else .dropped;
            },
        }
    }

    /// Stop `r` (its last waiter went, or shutdown): the tier releases its reservations; `r` is freed.
    pub fn restoreCancel(s: *Store, r: *Restore) void {
        if (s.tier) |t| t.vtable.cancel(t.ptr, r.job);
        s.stats.restores_cancelled += 1;
        s.endRestore(r);
    }

    fn endRestore(s: *Store, r: *Restore) void {
        if (std.mem.indexOfScalar(*Restore, s.restoring.items, r)) |i| {
            _ = s.restoring.orderedRemove(i);
            s.restoring_bytes -= r.bytes;
        }
        s.gpa.free(r.key);
        s.gpa.destroy(r);
    }

    /// An installed restore as a memory entry, room made as for any keep; false: dropped (no room, or memory had it).
    /// Its reservation moves from the restores' charge to the kept states'. A memory twin of the same point that is
    /// no longer usable (the family staled it: KV sharding's placements give a kept state's rows away, an overwrite) is
    /// what made the lookup go to the tier: it is replaced, never a reason to drop the restore and recompute.
    fn adopt(s: *Store, r: *Restore, saved: Saved, bytes0: u64, lineage: []const Fork) bool {
        const bytes = if (bytes0 > 0) bytes0 else s.pointBytes(r.point);
        s.restoring_bytes -= r.bytes; // (zeroed: endRestore then releases nothing more)
        r.bytes = 0;
        for (s.entries.items, 0..) |e, i| if (e.at == r.point.at and std.mem.eql(u32, e.tokens, r.key)) {
            if (s.live(e, r.key.len)) return s.unadopted(saved); // a usable copy came meanwhile: it wins
            s.remove(i); // the stale twin goes (its storage back to the family), the restore takes its place
            break;
        };
        if (!s.makeRoom(bytes, null, r.key)) return s.unadopted(saved);
        const e = s.gpa.create(Entry) catch return s.unadopted(saved);
        const tokens = s.gpa.dupe(u32, r.key) catch {
            s.gpa.destroy(e);
            return s.unadopted(saved);
        };
        const last = s.gpa.dupe(u32, r.key) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.unadopted(saved);
        };
        const lin = s.gpa.dupe(Fork, lineage) catch {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.unadopted(saved);
        };
        s.clock += 1;
        e.* = .{ .tokens = tokens, .at = r.point.at, .saved = saved, .bytes = bytes, .born = @intCast(r.key.len), .used = s.clock, .last = last, .shared = r.point.shared, .group = r.point.group, .point = r.point.id, .restored = true, .lineage = lin };
        s.entries.append(s.gpa, e) catch {
            e.point = null; // the point stays the tier's; only the memory copy goes
            s.free(e);
            s.stats.restores_dropped += 1;
            return false;
        };
        s.held += bytes;
        s.stats.disk_hits_committed += 1;
        s.stats.restored_tokens += r.point.at;
        if (!@import("builtin").is_test) std.log.info("prompt cache: restored {d} tokens from disk", .{r.point.at});
        return true;
    }

    fn unadopted(s: *Store, saved: Saved) bool {
        s.family.vtable.drop(s.family.ptr, saved);
        s.stats.restores_dropped += 1;
        return false;
    }

    /// Writes staged for at most `budget_ns` (engine thread).
    pub fn pump(s: *Store, budget_ns: u64) void {
        const t = s.tier orelse return;
        if (t.vtable.pump) |f| f(t.ptr, budget_ns);
    }

    /// Whether the tier has restores in flight or writes to stage (the host keeps servicing it).
    pub fn busy(s: *const Store) bool {
        const t = s.tier orelse return false;
        if (s.restoring.items.len > 0) return true;
        return if (t.vtable.busy) |f| f(t.ptr) else false;
    }

    /// Clean shutdown, engine thread: every restore stopped, writes drained within `budget_ns`; the points left
    /// unpublished at the deadline (counted, never published to make the count).
    pub fn flush(s: *Store, budget_ns: u64) u64 {
        const t = s.tier orelse return 0;
        while (s.restoring.items.len > 0) s.restoreCancel(s.restoring.items[s.restoring.items.len - 1]);
        const left = t.vtable.flush(t.ptr, budget_ns);
        s.stats.flushes += 1;
        if (left > 0) {
            s.stats.flush_timeouts += 1;
            s.stats.flush_unpublished += left;
            note("{d} kept states were still unpublished at the flush deadline; a restart misses them", .{left});
        }
        return left;
    }

    /// Durable point `id` is unusable (the family's all-rank abort, or its install failed): the tier drops it.
    pub fn forgetPoint(s: *Store, id: u64) void {
        const t = s.tier orelse return;
        for (s.entries.items) |e| if (e.point == id) {
            e.point = null;
        };
        t.vtable.forget(t.ptr, id);
    }

    /// Each rank's tier counters, when a tier is set.
    pub fn tierCounts(s: *const Store) ?RankCounts {
        const t = s.tier orelse return null;
        var out: RankCounts = .{};
        out.n = @intCast(@min(t.vtable.counts(t.ptr, &out.ranks), max_ranks));
        return out;
    }

    /// The lineage of a pass that resumes `from` (null: a fresh prefill) and computes its own rows from there: `from`'s
    /// forked at its position, or a new one (gpa-owned; empty without a tier, whose block names are its only use).
    pub fn passLineage(s: *Store, from: ?*const Entry) ![]Fork {
        if (s.tier == null) return &.{};
        s.next_lineage += 1;
        const own: Fork = .{ .name = s.next_lineage, .at = if (from) |e| e.at else 0 };
        const base: []const Fork = if (from) |e| e.lineage else &.{};
        if (base.len == 0) return s.gpa.dupe(Fork, &.{.{ .name = own.name, .at = 0 }});
        const out = try s.gpa.alloc(Fork, base.len + 1);
        @memcpy(out[0..base.len], base);
        out[base.len] = own;
        return out;
    }

    /// Before a prompt pass: restore the longest state `prompt` resumes (from 0: none or a failed copy) and plan its marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque) !Plan {
        const l = try s.lookup(a, prompt, history_len, shared, starts);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        s.reserve(prompt, null, l.marks); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// begin without the restore, for backends that restore inside their own prompt pass and then call `resumed`.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        const e = s.recall(prompt, starts, s.find(prompt, starts));
        if (e == null) s.counts.misses += 1;
        const marks_ = try s.fitting(a, try s.marks(a, prompt, if (e) |x| x.at else 0, history_len, shared, starts, if (e) |x| x.last else &.{}));
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            // the mark as `marks` computed it (a chunk start, or the cut itself), and the state the family keeps for
            // it (a chunk start, or the cut on the family's grid: dsv41 keeps an off-grid mark at its chunk end)
            const m = if (s.rules.planned) floorStart(starts, w) else w;
            if (std.mem.indexOfScalar(u32, marks_, m) == null) continue;
            const at = if (s.rules.planned) m else s.rules.floorGrid(w);
            if (at > 0) s.noteShared(prompt[0 .. at + s.rules.lookahead]);
        }
        s.reserve(prompt, e, marks_);
        return .{ .entry = e, .marks = marks_ };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, prompt: []const u32, from: ?*Entry, marks_: []const u32) void {
        var need: u64 = 0;
        for (marks_) |at| need += s.family.vtable.bytes(s.family.ptr, at);
        while (s.held + s.restoring_bytes + need > s.budget) {
            const i = s.victimBut(from, prompt) orelse return;
            s.remove(i);
            s.counts.evicted += 1;
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        for (s.entries.items, 0..) |e, i| if (e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) {
            s.counts.failed += 1;
            return s.remove(i);
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

    /// The marks whose state the budget can hold (a new slice in `a`); the rest are refused now, so no pass, nor a peer, copies them.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        for (marks_) |at| {
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget) try out.append(a, at) else {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        if (e.point) |p| if (s.tier) |t| if (t.vtable.touch) |f| f(t.ptr, p);
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            if (k > 0) { // the history's mark always; the others only away from it, each other and the resume point
                if (at - from < s.rules.min_gap) continue;
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

    /// The prompt pass stands at `at`: keep its state for `prompt`, evicting to fit; refused (counted) past the budget.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32) bool {
        return s.keepWith(prompt, at, owner, starts, .{});
    }

    /// keep, recording the state's conversation, pictures and lineage; with a tier its write is enqueued here.
    pub fn keepWith(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, info: KeepInfo) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
        for (s.entries.items) |e| if (e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n])) {
            e.used = s.clock; // the same state again: no copy
            if (shared and !e.shared) if (e.point) |p| if (s.tier) |t| if (t.vtable.share) |f| { // now a shared prefix, on disk too
                f(t.ptr, p);
                s.stats.shared_promoted += 1;
            };
            e.shared = e.shared or shared;
            if (e.group == 0) e.group = info.group;
            return true;
        };
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + s.restoring_bytes + free_ + bytes <= s.budget) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            s.remove(s.victimBut(null, prompt).?); // its storage may come back spare and take this state
            s.counts.evicted += 1;
        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.restoring_bytes -| bytes);
        if (s.restoring_bytes > 0 and s.held + s.restoring_bytes + bytes > s.budget) { // restores in flight hold the rest
            const reuses = if (s.family.vtable.reuses) |f| f(s.family.ptr, at) else false;
            if (!reuses) {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: restores in flight hold {d} MiB of the {d} MiB budget", .{ at, s.restoring_bytes >> 20, s.budget >> 20 });
                return false;
            }
        }
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
        const lin = s.gpa.dupe(Fork, info.lineage) catch {
            s.family.vtable.drop(s.family.ptr, saved);
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .tokens = tokens, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared, .group = info.group, .images = info.images, .lineage = lin };
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.held += charged;
        s.counts.kept += 1;
        if (s.tier) |t| { // write at keep: the tier stages the state's bytes between rounds
            e.point = t.vtable.keep(t.ptr, &.{ .saved = saved, .key = tokens, .at = at, .bytes = charged, .group = info.group, .shared = shared, .images = info.images, .parent = info.parent, .fork = info.fork, .lineage = lin });
            if (e.point != null) s.stats.writes_enqueued += 1;
        }
        if (shared) s.learn(e, starts);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| s.restoring_bytes); // spare storage only inside what the budget leaves
        return true;
    }

    /// A shared cut's state written to disk once (--learn): later sessions read it back, after a restart too.
    fn learn(s: *Store, e: *const Entry, starts: []const u32) void {
        const im = s.imprint orelse return;
        const write = s.family.vtable.write orelse return;
        const key = imprint.Imprint.keyOf(e.tokens);
        if (im.has(key) or e.bytes > im.cap) return;
        while (!im.fits(e.bytes)) s.unlearn(im, im.victim() orelse return); // the least recently used go first
        write(s.family.ptr, e.saved, im.dir, key) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        im.add(key, e.at, e.tokens, starts, e.bytes) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        if (!@import("builtin").is_test) std.log.info("prompt cache: learned {d} tokens to disk", .{e.at});
    }

    /// Learned state `key` forgotten: its files (the family's), then its index record.
    fn unlearn(s: *Store, im: *imprint.Imprint, key: u64) void {
        if (s.family.vtable.forget) |f| f(s.family.ptr, im.dir, key);
        im.remove(key) catch |err| note("forgetting a learned state failed ({s})", .{@errorName(err)});
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, starts: []const u32, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        if (!s.makeRoom(bytes, have, prompt)) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            s.unlearn(im, key);
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
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip` (a pass's resume): one a later prompt extends (never a shared cut), oldest first; else the oldest.
    fn victimBut(s: *const Store, skip: ?*Entry, prompt: []const u32) ?usize {
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (e.shared or e == skip) continue; // a shared cut is evicted only as the least recently used
            const now = prompt.len > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // this prompt extends it
            const moved_on = now or for (s.entries.items) |o| {
                if (o != e and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on and (best == null or e.used < s.entries.items[best.?].used)) best = i;
        }
        if (best) |i| return i;
        var oldest: ?usize = null;
        for (s.entries.items, 0..) |e, i| if (e != skip and (oldest == null or e.used < s.entries.items[oldest.?].used)) {
            oldest = i;
        };
        return oldest;
    }

    // -- 0111: continuation ---------------------------------------------------------------------------------------

    /// The conversation a request belongs to, decided at its lookup (`hit`: the state it resumes, null: none). A hit on
    /// one of a conversation's own turn states keeps that conversation, when the prompt shows it is that conversation
    /// (`ownHit`); otherwise (a shared prefix, a state read back from disk, an own state of another conversation, or
    /// nothing) the most recently used own state the prompt repeats past its conversation's first prompt and the first
    /// chat_match reply tokens names it (`matchable`: no pictures and drafted, as the source requires); else a new
    /// conversation. Deterministic in the entries' use order. Off (group 0) unless `groups`.
    pub fn join(s: *Store, prompt: []const u32, hit: ?*const Entry, matchable: bool) Join {
        if (!s.groups) return .{};
        const source: Join.Source = if (hit) |e| (if (e.shared) .shared else .other) else .nothing;
        if (hit) |e| if (s.ownHit(e, prompt)) return .{ .group = e.group, .source = .other };
        if (matchable) if (s.continuedChat(prompt)) |c| return .{
            .group = c.group,
            .continued = true,
            .prior_len = @intCast(c.tokens.len),
            .back = @intCast(commonPrefix(c.last, prompt)),
            .source = source,
        };
        s.next_group += 1;
        const g = s.next_group;
        s.roots.put(s.gpa, g, @intCast(prompt.len)) catch {};
        if (s.roots.count() > 4 * @max(16, s.entries.items.len)) s.pruneRoots(g); // conversations no kept state is in
        return .{ .group = g, .source = source };
    }

    /// Whether resuming `e` keeps `e`'s conversation: one of its own states (not shared, not read back), and the prompt
    /// repeats its evidence (the last prompt that kept or resumed it) up to its conversation's first prompt and first
    /// chat_match reply tokens, or all of it when shorter. GLM's own states always hold conversation-specific tokens;
    /// a native state on the grid or at last_end can lie wholly inside a system prompt other conversations share, so
    /// the hit alone is no evidence. A conversation whose first prompt was forgotten keeps the hit-alone rule.
    fn ownHit(s: *const Store, e: *const Entry, prompt: []const u32) bool {
        if (e.shared or e.restored or e.group == 0) return false;
        const root = s.roots.get(e.group) orelse return true;
        return commonPrefix(e.last, prompt) >= @min(e.last.len, @as(usize, root) + chat_match);
    }

    /// The own kept state whose conversation `prompt` continues (continued_chat): most recently used first; a later
    /// entry wins a tie of use. Shared, restored, pictured and forgotten conversations' states are passed over.
    pub fn continuedChat(s: *const Store, prompt: []const u32) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |c| {
            if (c.shared or c.restored or c.images or c.group == 0) continue;
            const root = s.roots.get(c.group) orelse continue;
            const n = @as(usize, root) + chat_match;
            if (c.last.len < n or prompt.len < n or c.last[n - 1] != prompt[n - 1] or !std.mem.eql(u32, c.last[0..n], prompt[0..n])) continue;
            if (best == null or c.used >= best.?.used) best = c;
        }
        return best;
    }

    fn pruneRoots(s: *Store, keep_: u64) void {
        var gone: std.ArrayList(u64) = .empty;
        defer gone.deinit(s.gpa);
        var it = s.roots.keyIterator();
        while (it.next()) |k| {
            if (k.* == keep_) continue;
            const held = for (s.entries.items) |e| {
                if (e.group == k.*) break true;
            } else false;
            if (!held) gone.append(s.gpa, k.*) catch return;
        }
        for (gone.items) |k| _ = s.roots.remove(k);
    }

    /// After a continued request's pass resumed at `cut` of `len` tokens: counted, and a large prefill logged with where
    /// the prompt left the kept state and what it resumed from (only rank 0 runs the host).
    pub fn continued(s: *Store, j: Join, cut: u32, len: usize) void {
        if (!j.continued) return;
        s.stats.chats_continued += 1;
        s.stats.chats_continued_prefill_tokens += len -| cut;
        if (!shouldLog(j, cut, len) or @import("builtin").is_test) return;
        var buf: [512]u8 = undefined;
        std.log.info("{s}", .{continuedLine(&buf, j, cut, len)});
    }
};

/// Whether `continued` logs a continued request's pass (from `cut` of `len` tokens): its prefill is large.
pub fn shouldLog(j: Join, cut: u32, len: usize) bool {
    return j.continued and len -| cut >= chat_log_tokens;
}

/// continued's log line.
pub fn continuedLine(buf: []u8, j: Join, cut: u32, len: usize) []const u8 {
    const source = switch (j.source) {
        .nothing => "nothing",
        .shared => "a shared prefix",
        .other => "a state that is not a turn's own (read back from disk)",
    };
    var why_buf: [256]u8 = undefined;
    const why = (if (j.back < j.prior_len)
        std.fmt.bufPrint(&why_buf, "it went back to token {d}, before that conversation's newest kept state ({d} tokens)", .{ j.back, j.prior_len })
    else
        std.fmt.bufPrint(&why_buf, "its kept state of {d} tokens could not serve it (stale, off the grid or past the replay cut)", .{j.prior_len})) catch "";
    return std.fmt.bufPrint(buf, "prompt cache: conversation {d}: resumed at {d} of {d} tokens from {s}, prefilling {d}; {s}", .{ j.group, cut, len, source, len -| cut, why }) catch buf[0..0];
}

/// The tokens two id lists share from the start.
pub fn commonPrefix(a: []const u32, b: []const u32) usize {
    return std.mem.indexOfDiff(u32, a, b) orelse @min(a.len, b.len);
}

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

test {
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_tier_test.zig");
}

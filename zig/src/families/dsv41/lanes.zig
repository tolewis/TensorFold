//! DeepSeek-V4.1-Flash behind TensorFold 1.0's lane core (core/lanes): the round loop picks each round's streams and
//! windows and keeps the accepted rows; this backend runs them on the port's model (model.zig) with the served lane's
//! arithmetic. Each stream has a pool slot (its window rings, positional stores and drafter rings) and an extent of
//! the shared compressed plane (first fit, chunk-aligned, as large as its prompt, its budget and a round's rows), and
//! its ids by position on both ranks (Engram's n-grams). verify is one round over every window (round.zig) with each
//! row drawn by the served sampler (sampling.zig) at its keyed position, and the drafting streams' window rows
//! absorbed into their drafter rings at once (the served eager absorb: the rows past the kept ones sit where the
//! drafter never reads, and the next window writes over them); draft is one batched drafter pass (draft.zig) for
//! every stream asked, the drafts held for the next round with their confidences, whose sigmoids are each draft's
//! chance of landing once the earlier ones did (probabilities: their running products, the chance that a draft and
//! every earlier one land; the round loop's allocation trims the windows by them). The caches are
//! positional, so keep only checks that a path is a prefix. Rank 0 runs the core; before each primitive it sends rank
//! 1 the primitive's inputs over the ranks' link, and rank 1 (follow) runs the same primitive: both ranks issue the
//! same kernels and collectives in the same order. The drafted reply is the serial one: every row is drawn with its
//! stream's key at its own position, whatever the round's composition.
const std = @import("std");
const lanes = @import("lanes");
const model = @import("model.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const link = @import("link.zig");
const cuda = @import("cuda");
const spill_tier = @import("spill_tier.zig");
const parts = @import("parts.zig");
const kv_lanes = @import("kv_lanes.zig");
const shard = @import("shard.zig");

const be = lanes.backend;
const Model = model.Model;
const max_streams = model.max_streams;
const max_rows = round.max_rows;

/// The spill tier on rank 0 (the prompt cache's Tier) and on a follower, over this rank's planes (spill_tier.zig).
pub const SpillTier = spill_tier.Leader(Model.SpillDev);
pub const SpillFollower = spill_tier.Follower(Model.SpillDev);

comptime {
    // parts.zig keeps lanes' block and growth slack (it is self-contained so `zig test` runs it alone)
    std.debug.assert(parts.chunk_rows == model.chunk_rows and parts.growth_rows == growth_rows);
}

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The image rows a fill_images frame carries at most.
pub const image_piece = 32 << 10;

/// A gate's diagnostic (the tools' --logits-sha 1): each prompt's logits' sha256 on stderr.
pub var logits_sha = false;

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
/// fill_resume (TF_FILL_CHUNKS' pieces) is 12, after spill's two (both ranks run one build).
/// With the spill tier on: rank_wire's frames (spill_tier: restores' agreement, guards) and the spill notes (rank 0's
/// write-at-keep and metadata). TF_DS_KV_SHARD adds rank_wire too (rank_wire.zig's frames: the parts' agreement, each
/// plan's ops, each transfer's shard_xfer/shard_finish, the digest checks). The two are never on together
/// (SpillWithKvShard), so one rank_wire Leader/Follower a rank serves either.
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9, rank_wire = 10, spill = 11, fill_resume = 12 };

comptime {
    std.debug.assert(@intFromEnum(Kind.rank_wire) == @import("rank_wire.zig").lanes_kind);
    std.debug.assert(@intFromEnum(Kind.spill) == spill_tier.note_kind);
}

/// A frame being written.
pub const Wire = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn begin(w: *Wire, gpa: std.mem.Allocator, kind: Kind, step: u64) !void {
        w.buf.clearRetainingCapacity();
        try w.int(gpa, u8, @backingInt(kind));
        try w.int(gpa, u64, step);
    }

    pub fn int(w: *Wire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }

    pub fn ids(w: *Wire, gpa: std.mem.Allocator, v: []const i32) !void {
        try w.int(gpa, u32, @intCast(v.len));
        try w.buf.ensureUnusedCapacity(gpa, v.len * 4);
        for (v) |x| try w.int(gpa, i32, x);
    }

    pub fn deinit(w: *Wire, gpa: std.mem.Allocator) void {
        w.buf.deinit(gpa);
    }
};

/// A verify frame's window: slot, extent, first position, absorb, ids; TF_DS_KV_SHARD (`sharded`) adds its extent's
/// part (u8) and generation (u64), so a follower refuses a round for a reused extent. Off, the bytes are as ever.
pub fn writeWin(w: *Wire, gpa: std.mem.Allocator, win: Win, sharded: bool) !void {
    try w.int(gpa, u32, @intCast(win.slot));
    try w.int(gpa, u64, win.base);
    try w.int(gpa, u64, win.end);
    try w.int(gpa, u64, win.start);
    try w.int(gpa, u8, @intFromBool(win.absorb));
    try w.ids(gpa, win.ids);
    if (sharded) {
        if (win.part < 0 or win.part >= parts.max_parts) return error.BadPart;
        try w.int(gpa, u8, @intCast(win.part));
        try w.int(gpa, u64, win.gen);
    }
}

/// writeWin's window read back, its ids into `ids` (the window's slice points there).
pub fn readWin(r: *Reader, gpa: std.mem.Allocator, ids: *std.ArrayList(i32), sharded: bool) !Win {
    const slot: usize = try r.int(u32);
    const base: usize = @intCast(try r.int(u64));
    const end: usize = @intCast(try r.int(u64));
    const start: usize = @intCast(try r.int(u64));
    const absorb = try r.int(u8) != 0;
    try r.ids(gpa, ids);
    var win: Win = .{ .slot = slot, .base = base, .end = end, .start = start, .absorb = absorb, .ids = ids.items };
    if (sharded) {
        win.part = try r.int(u8);
        win.gen = try r.int(u64);
    }
    return win;
}

/// A frame being read.
pub const Reader = struct {
    b: []const u8,
    at: usize = 0,

    pub fn int(r: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }

    /// The frame's bytes after those read.
    pub fn rest(r: *Reader) []const u8 {
        const b = r.b[r.at..];
        r.at = r.b.len;
        return b;
    }

    /// A list of ids written by Wire.ids, into `out` (replacing it).
    pub fn ids(r: *Reader, gpa: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const n = try r.int(u32);
        try out.resize(gpa, n);
        for (out.items) |*x| x.* = try r.int(i32);
    }
};

/// A window of a round as both ranks build it: its stream's slot and extent, its first position, whether its rows'
/// taps go into the drafter, its rows' ids; TF_DS_KV_SHARD: its extent's part (the rank holding its compressed rows;
/// -1: its base's, which round.Rows.ownerOf derives and checks) and its extent's generation (parts.zig: a follower
/// refuses a round naming a reused extent).
pub const Win = struct { slot: usize, base: usize, end: usize, start: usize, absorb: bool, ids: []const i32, part: i64 = -1, gen: u64 = 0 };

/// The rows of a round (round.Rows' slices point here).
pub const Built = struct {
    ids: [max_rows]i64 = undefined,
    pos: [max_rows]i64 = undefined,
    slots: [max_rows]i64 = undefined,
    bases: [max_rows]i64 = undefined,
    ends: [max_rows]i64 = undefined,
    parts: [max_rows]i64 = undefined, // TF_DS_KV_SHARD: each row's part, plain data for the attention (Win.part)
    wins: [max_streams]round.Window = undefined,
    absorb: [max_streams]bool = undefined,
    r: usize = 0,
    n: usize = 0,

    /// Each window's ids written at its positions in its slot's sequence (the rows past them dropped: the rejected
    /// rows of the last round), then the round's rows in window order.
    pub fn build(b: *Built, gpa: std.mem.Allocator, seqs: *[max_streams]std.ArrayList(i32), wins: []const Win) !round.Rows {
        if (wins.len == 0 or wins.len > max_streams) return error.BadRound;
        b.r = 0;
        b.n = wins.len;
        for (wins, 0..) |w, i| {
            if (w.slot >= max_streams or w.ids.len == 0) return error.BadRound;
            if (b.r + w.ids.len > max_rows) return error.WindowTooWide;
            const seq = &seqs[w.slot];
            if (seq.items.len < w.start) return error.SequenceGap;
            try seq.resize(gpa, w.start + w.ids.len);
            @memcpy(seq.items[w.start..], w.ids);
            for (w.ids, 0..) |id, j| {
                b.ids[b.r + j] = id;
                b.pos[b.r + j] = @intCast(w.start + j);
                b.slots[b.r + j] = @intCast(w.slot);
                b.bases[b.r + j] = @intCast(w.base);
                b.ends[b.r + j] = @intCast(w.end);
                b.parts[b.r + j] = w.part;
            }
            b.wins[i] = .{ .row = b.r, .n = w.ids.len, .seq = seq.items[0 .. w.start + w.ids.len] };
            b.absorb[i] = w.absorb;
            b.r += w.ids.len;
        }
        const R = b.r;
        return .{ .ids = b.ids[0..R], .pos = b.pos[0..R], .slots = b.slots[0..R], .bases = b.bases[0..R], .ends = b.ends[0..R], .windows = b.wins[0..b.n], .parts = b.parts[0..R] };
    }
};

/// The first chunk-aligned extent of `size` positions in [0, cap) that overlaps none of `used`.
/// Whether [lo, hi) meets any of `used`.
fn clashes(used: []const [2]usize, lo: usize, hi: usize) bool {
    for (used) |u| if (lo < u[1] and u[0] < hi) return true;
    return false;
}

/// Whether one of the prompt cache's marks has its chunk end at `at` (each kept at the chunk end at or before it).
fn markEnds(marks: []const u32, at: usize) bool {
    for (marks) |w| if (w / model.chunk_rows * model.chunk_rows == at) return true;
    return false;
}

/// Room left past a kept state's extent while placements fit so, for its conversation's next turn to resume in place.
pub const growth_rows = 16 * model.chunk_rows;

/// The first fit of `size` clear of `live` and of kept extents plus growth_rows, else of their extents, else of their rows.
pub fn placeSoft(gpa: std.mem.Allocator, live: []const [2]usize, kept: []const [3]usize, size: usize, cap: usize) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.appendSlice(gpa, live);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) + growth_rows });
    if (place(used.items, size, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], @max(k[2], k[0] + k[1]) });
    if (place(used.items, size, cap)) |b| return b;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| try used.append(gpa, .{ k[0], k[0] + k[1] });
    return place(used.items, size, cap);
}

test "a placement leaves a kept state room to grow while one fits so" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 48 * c, 50 * c }}; // rows [0, 48c), its stream's extent to 50c
    try std.testing.expectEqual(@as(?usize, 50 * c + growth_rows), try placeSoft(gpa, &.{}, &kept, 4 * c, 128 * c));
    // no room past the growth: flush against the extent, as before
    try std.testing.expectEqual(@as(?usize, 50 * c), try placeSoft(gpa, &.{}, &kept, 4 * c, 50 * c + growth_rows));
}

test "a kept state's tail stays free while a placement fits elsewhere" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 4 * c, 5 * c }}; // rows [0, 4c), its stream's extent to 5c
    try std.testing.expectEqual(@as(?usize, 5 * c), try placeSoft(gpa, &.{}, &kept, 4 * c, 16 * c));
    // no room clear of the tail: the rows alone are held back
    const live = [_][2]usize{.{ 9 * c, 16 * c }};
    try std.testing.expectEqual(@as(?usize, 4 * c), try placeSoft(gpa, &live, &kept, 5 * c, 16 * c));
    try std.testing.expectEqual(@as(?usize, null), try placeSoft(gpa, &live, &kept, 6 * c, 16 * c));
}

/// Whether, with an extent of `size` at `b` taken besides `used`, another extent of `size` still fits in [0, cap).
pub fn roomAfter(gpa: std.mem.Allocator, used: []const [2]usize, b: usize, size: usize, cap: usize) !bool {
    const all = try gpa.alloc([2]usize, used.len + 1);
    defer gpa.free(all);
    @memcpy(all[0..used.len], used);
    all[used.len] = .{ b, b + size };
    return place(all, size, cap) != null;
}

test "a copy that fills the pool leaves no room for another request of its size" {
    const c = model.chunk_rows;
    // three live extents of 4 chunks in a 16-chunk pool: a fourth fits, and leaves no room for a fifth
    const used = [_][2]usize{ .{ 0, 4 * c }, .{ 4 * c, 8 * c }, .{ 8 * c, 12 * c } };
    const gpa = std.testing.allocator;
    const b = place(&used, 4 * c, 16 * c).?;
    try std.testing.expectEqual(@as(usize, 12 * c), b);
    try std.testing.expect(!try roomAfter(gpa, &used, b, 4 * c, 16 * c));
    // two live extents: the third leaves room for a fourth
    const b2 = place(used[0..2], 4 * c, 16 * c).?;
    try std.testing.expect(try roomAfter(gpa, used[0..2], b2, 4 * c, 16 * c));
}

pub fn place(used: []const [2]usize, size: usize, cap: usize) ?usize {
    var base: usize = 0;
    while (base + size <= cap) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, model.chunk_rows);
    }
    return null;
}

/// The token a row's logits draw at absolute position `position`: greedy without sampling (or at temperature 0),
/// else the served sampler with the stream's key.
pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(sampling.argmax(row));
    if (sm.temperature <= 0) return @intCast(sampling.argmax(row));
    const t = try sampling.sampleRow(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
    return @intCast(t);
}

/// Where rank 0's backend calls spend their time (ns, summed; --profile): a round's frame to rank 1, its forward
/// and absorb (synchronized apart), the logits' copy to the host and the draws; the drafter passes; the prefills.
pub const Profile = struct {
    send: u64 = 0,
    enqueue: u64 = 0,
    forward: u64 = 0,
    absorb: u64 = 0,
    logits: u64 = 0,
    sample: u64 = 0,
    pass: u64 = 0,
    prefill: u64 = 0,
    rounds: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    pass_streams: u64 = 0,
    prefills: u64 = 0,
};

/// A round-cost table "30.0,35.2,..." (TF_DS_ROUND_MS's form): ms by rows from 1.
pub fn parseMs(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    var out: std.ArrayList(f64) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseFloat(f64, std.mem.trim(u8, x, " ")));
    if (out.items.len == 0) return error.BadRoundMs;
    return out.items;
}

/// A stream's place in the pool and the drafts it holds; TF_DS_KV_SHARD: its planner stream, part and generation
/// (parts.zig's Live), and whether its first decode round ran (planDecode).
const Lane = struct {
    slot: usize,
    base: usize,
    end: usize,
    id: u64 = 0,
    part: u8 = 0,
    gen: u64 = 0,
    decoding: bool = false,
    first: u64 = 0, // the prompt's draw (a handle)
    held: [draft.max_block]u32 = undefined,
    confs: [draft.max_block]f32 = undefined,
    nheld: usize = 0,
};

extern "c" fn exp(x: f64) f64;

/// A kept prompt state (the prompt cache's saved state, kept at a chunk end the prompt pass reached before its
/// replay): the slot's own part on the device (Model.snapCopy), and the extent its compressed rows stay in,
/// [base, base + at), held back from placement while it lives. Stale: a placement needed that room (its restore fails
/// and the prompt runs from 0).
pub const Snap = struct {
    id: u64,
    at: usize,
    base: usize,
    buf: cuda.DeviceBuffer,
    stale: bool = false,
    wanted: u64 = 0, // when a request first waited for the live stream that holds these rows (placeKept; 0: none)
    end: usize = 0, // the extent end of the stream that kept it (placement holds [base, end) back while room allows)
    slot: u32 = 0, // the slot it was kept from (the spill tier's keep note: a follower reads the prompt's ids there)
    loading: bool = false, // a restore's destination: its rows held back, not usable until every rank installed it
    restored: bool = false, // read back from the spill tier
    part: u8 = 0, // TF_DS_KV_SHARD: the part holding its rows (parts.zig's Kept; a live move changes it)
};

/// Kept states this long or longer say so in the log when they go stale (their prompts then run again from 0).
const note_rows = 32 * model.chunk_rows;

fn noteStale(sn: *const Snap, why: []const u8) void {
    if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} goes stale: {s}", .{ sn.at, sn.base, why });
}

/// A prompt pass between its pieces: the chunk it runs next and what its chunk loop reads.
const Fill = struct { s: *const lanes.Stream, slot: usize, len: usize, replay: usize, last_end: usize, start: usize, head: bool = false, base: usize = 0, restored: bool = false, part: ?u64 = null };

/// How long a request waits for the live stream that holds the kept state it resumes from to end (then it resumes
/// in place and every kept state stays) before other kept states give their room for a copy of it.
const held_wait_ns: u64 = 10 * std.time.ns_per_s;

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    peer: ?link.Link, // rank 1 (null: one rank); every follower gets each frame (Model.followers)
    vision: ?*@import("vision.zig").Vision = null, // rank 0's tower (a vision checkpoint served with its kit's vision/)
    step: u64 = 0,
    wire: Wire = .{},
    streams: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    wc: [window_rows]lanes.config.Cost = undefined, // the window costs (init)
    sc: [max_rows]lanes.config.Cost = undefined, // the shared forwards' costs (init)
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    built: Built = .{},
    prof: ?Profile = null,
    fix_k: ?usize = null, // a measurement (tf-dsv41-lanes --fix-k): K drafts a stream, chances 1 to K and 0 past it
    solo_overhead: bool = false, // TF_DS_SOLO_OVERHEAD=1: a lone drafted stream's head windows cost the fixed draft allowance
    no_copies: bool = false, // TF_DS_NO_COPIES=1: a drafted stream proposes no suffix-lookup copy windows
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, oldest first
    next_snap: u64 = 1,
    fill: ?Fill = null, // the prompt pass that runs in pieces between rounds, from its next chunk
    paused: ?Fill = null, // the pass a short prompt cut into (lanes' cut_in), resumed by its next piece
    spill: ?*SpillTier = null, // --spill-dir/--spill-gib: the spill tier (native.zig opens it; null: off, nothing changes)
    kv: ?*kv_lanes.Kv = null, // TF_DS_KV_SHARD: the parts' planner (startKv, at open; null: the switch off)

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        return ln;
    }

    /// Tells rank 1 to stop, then frees the host state (the model is the caller's).
    pub fn deinit(self: *Lanes) void {
        self.stopKv();
        for (self.snaps.items) |sn| {
            sn.buf.free();
            self.gpa.destroy(sn);
        }
        self.snaps.deinit(self.gpa);
        if (self.peer != null) self.send(.done) catch {};
        self.streams.deinit(self.gpa);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .prefill_part = prefillPartFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .probabilities = probabilitiesFn,
            .release = releaseFn,
        } };
    }

    /// The facts the round loop reads: windows of the pending row and up to the drafter's block, several streams'
    /// windows in one forward of at most 16 rows (four streams), the drafter's chances, the served lane's round ms.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.m.drafting();
        return .{
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (self.m.dr) |dr| @intCast(dr.n) else 1,
            .window_costs = self.wc[0..],
            .mtp_step_ms = draft.draft_ms,
            .solo_head_overhead_ms = if (drafting and self.solo_overhead) draft.draft_ms else null, // opt-in: the Python engine's fixed draft allowance
            .copy_proposals = !(drafting and self.no_copies), // opt-in: the draft head proposes; the Python engine makes no copies either
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The frame being built goes to rank 1 (after `begin` and the fields), or to ranks 1-3 on the four-node split.
    fn flush(self: *Lanes) !void {
        if (self.peer == null) return;
        for (self.m.followers()) |l| try l.?.send(self.wire.buf.items);
    }

    fn begin(self: *Lanes, kind: Kind) !bool {
        if (self.peer == null) return false;
        try self.wire.begin(self.gpa, kind, self.step);
        self.step += 1;
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    // -- TF_DS_KV_SHARD: kv_lanes.zig drives parts.zig's planner (docs/port/shard-lanes.md) ------------------------

    /// TF_DS_KV_SHARD: the parts' planner made and agreed with the followers (hello, shard_layout, shard_set, the
    /// first digest check), once, at open with the Lanes at their final address (the hooks point here): a follower
    /// started with other switches or another pool fails the start, never a request. Nothing with the switch off.
    pub fn startKv(self: *Lanes) !void {
        const m = self.m;
        if (m.kvs == null or self.kv != null) return;
        const settings = try kv_lanes.Settings.fromEnv(getenv);
        const k = try self.gpa.create(kv_lanes.Kv);
        errdefer self.gpa.destroy(k);
        k.* = try kv_lanes.Kv.init(self.gpa, kvConfig(m), settings, modelDev(m), .{ .ctx = self, .stale = kvStale, .moved = kvMoved });
        errdefer k.deinit();
        // the shard's own leader agrees on the shard's fields (the spill tier is refused with TF_DS_KV_SHARD:
        // SpillWithKvShard). PORT(round3): with both on, pass the spill side's Leader (its full fingerprint,
        // rank_identity.zig) here so the followers hear one hello and one Tracker sees every frame
        if (self.peer != null) try k.lead(.{ .ctx = self, .rw = kvSendRw }, m.followers(), null, m.now() | 1, kv_lanes.shardExchange(0, m.world, k.planner.cfg, settings));
        try k.start();
        self.kv = k;
    }

    /// startKv undone (an open that fails after it).
    pub fn stopKv(self: *Lanes) void {
        const k = self.kv orelse return;
        k.deinit();
        self.gpa.destroy(k);
        self.kv = null;
    }

    fn kvSendRw(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const self = of(ctx);
        if (try self.begin(.rank_wire)) {
            try self.wire.buf.appendSlice(self.gpa, sub);
            try self.flush();
        }
    }

    /// A kept state the planner gave up (a stale op): its Snap too.
    fn kvStale(ctx: *anyopaque, kept: u64) void {
        const self = of(ctx);
        for (self.snaps.items) |sn| if (sn.id == kept) {
            if (!sn.stale) noteStale(sn, "a placement needed its room (its part's oldest first)");
            sn.stale = true;
        };
    }

    /// A live stream moved with its kept states: every Snap's and Lane's rows from the planner again (copyExtent,
    /// usable and the spill tier read them).
    fn kvMoved(ctx: *anyopaque, _: u64) void {
        of(ctx).kvSync();
    }

    fn kvSync(self: *Lanes) void {
        const p = &(self.kv orelse return).planner;
        for (self.snaps.items) |sn| if (p.keptAt(sn.id)) |k| {
            sn.base = k.base;
            sn.end = k.end;
            sn.part = k.part;
        };
        var it = self.streams.valueIterator();
        while (it.next()) |l| if (p.liveIndex(l.id)) |i| {
            const x = p.lives.items[i];
            l.base = x.base;
            l.end = x.base + x.size;
            l.part = x.part;
            l.gen = x.gen;
        };
    }

    const PlacedParts = struct { id: u64, part: u8, gen: u64, base: usize, kept: ?*Snap, moved: bool };

    /// prefillFn's placement with the switch on (kv_lanes.Kv.admit: plan, apply, send, act on the ops in order, the
    /// transfers finished with rank 0's ok): the extent in one part, and the kept state it resumes from when the plan
    /// resumed it (in place, or copied or carried there and finished ok).
    fn placeParts(self: *Lanes, k: *kv_lanes.Kv, s: *lanes.Stream, size: usize, slot: usize, len: usize) !PlacedParts {
        var src: ?*Snap = null;
        if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            if (self.usable(sn, len)) src = sn else s.reuse_failed = true;
        }
        // (lanes owns the clock: patient only when a live stream holds the state, as placeKept asks)
        const waits = if (src) |sn| k.planner.held(sn.id) and self.patient(sn) else false;
        const a = try k.admit(@intFromPtr(s), std.hash.Wyhash.hash(0, s.id), @intCast(slot), size, if (src) |sn| sn.id else null, waits);
        if (src) |sn| {
            sn.wanted = 0;
            if (!a.resumed) {
                if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} was not resumed ({s}); the prompt runs from 0", .{ sn.at, sn.base, if (a.dropped) "no room in its part" else "its transfer failed" });
                s.reuse_failed = true;
            }
        }
        const kept = if (a.resumed) src else null;
        return .{ .id = a.id, .part = a.part, .gen = a.gen, .base = a.base, .kept = kept, .moved = if (kept) |sn| a.base != sn.base else false };
    }

    /// verifyFn's planner side: a stream's first round gives its twins back (planDecode), then the planner's clock and
    /// every window's rows with content once this round ran (its cache length plus the round's rows, which it
    /// writes at [cache_len, cache_len + rows): an upper bound, rejected drafts' rows included, so a live move before
    /// the next round copies every row this one wrote), and the periodic digest check.
    fn kvRound(self: *Lanes, k: *kv_lanes.Kv, windows: []const be.Window) !void {
        var vs: [max_streams]kv_lanes.Kv.Valid = undefined;
        for (windows, 0..) |w, i| {
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            if (!l.decoding) {
                try k.decodeStart(l.id);
                l.decoding = true;
            }
            vs[i] = .{ .id = l.id, .valid = @as(usize, @intCast(w.stream.cache_len)) + w.rows() };
        }
        try k.round(vs[0..windows.len]);
    }

    /// The parts' stats for /health and /metrics (native.Loaded.kv_parts: the host reads them on its engine thread).
    pub fn kvPartsFn(ptr: *anyopaque, out: *lanes.kv_parts.Stats) void {
        const self = of(ptr);
        const k = self.kv orelse return;
        k.stats(out);
        if (self.m.kvs) |x| {
            const st = x.stats;
            out.prompt_fetches = st.prompt_fetches;
            out.prompt_rows = st.prompt_rows;
            out.prompt_picks = st.prompt_picks;
            out.prompt_bytes = st.prompt_bytes;
            out.prompt_syncs = st.prompt_syncs;
            out.fetch_rounds = st.fetch_rounds;
            out.owner_rounds = st.owner_rounds;
            out.copies = st.carries;
            out.copy_bytes = st.carry_bytes;
        }
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The lowest free slot, the first extent that fits, the prompt in chunks of the served size (at multiples of it:
    /// the chunks the gates checked; the host's chunk plan is not followed), the head's draw of the first token.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        if (!try of(ptr).prefillPart(s, std.math.maxInt(usize))) return error.FillUnfinished;
    }

    /// The prompt pass in pieces of `chunks` chunks (opened at its first): true once it reached the head and drew the first token.
    fn prefillPartFn(ptr: *anyopaque, s: *lanes.Stream, chunks: usize) anyerror!bool {
        return of(ptr).prefillPart(s, chunks);
    }

    fn prefillPart(self: *Lanes, s: *lanes.Stream, pieces: usize) !bool {
        const m = self.m;
        // TF_DS_KV_SHARD binds a pass's parts and twins for the whole pass: it runs whole (no pieces, no cut-ins)
        const chunks = if (self.kv != null) std.math.maxInt(usize) else pieces;
        const t0 = m.now();
        defer if (self.prof) |*p| {
            p.prefill += m.now() - t0;
        };
        if (self.fill) |f| if (f.s != s) {
            // a prompt cuts in between this pass's pieces: it waits (one at most, no images), its next piece resumes it
            if (self.paused != null or f.s.images.len > 0 or s.images.len > 0) return error.FillBusy;
            self.paused = f;
            self.fill = null;
        };
        if (self.fill == null) {
            if (self.paused != null and self.paused.?.s == s) try self.fillResume() else try self.fillOpen(s);
        }
        errdefer self.fill = null;
        const f = &self.fill.?;
        const seq = self.seqs[f.slot].items;
        m.fill_seq = seq;
        // a round may run before the next piece: the read-ahead done and the prompt's ids gone, as rank 1 does at its frame
        defer {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        var ran: usize = 0;
        while (f.start < f.len and ran < chunks) : (ran += 1) {
            const start = f.start;
            const len = f.len;
            if (start > 0 and s.isCancelled()) return error.Cancelled;
            const n = @min(model.chunk_rows, len - start);
            if (try self.begin(.fill_chunk)) {
                try self.wire.int(self.gpa, u64, start);
                try self.wire.int(self.gpa, u32, @intCast(n));
                try self.wire.int(self.gpa, u64, f.replay);
                try self.flush();
            }
            if (self.spill) |t| t.bump(f.base + start, f.base + start + n);
            f.head = blk: {
                // TF_DS_KV_SHARD: the stream's twins pinned while its chunk runs (parts.zig setPartial), not between
                if (f.part) |id| self.kv.?.planner.setPartial(id, true);
                defer if (f.part) |id| self.kv.?.planner.setPartial(id, false);
                break :blk try m.fillChunk(seq[0 .. start + n], start, n, f.replay);
            };
            f.start += n;
            if (s.reuse.hook) |k| if (f.start < len and f.start <= f.replay and (f.start == f.last_end or markEnds(s.reuse.marks, f.start))) k.at(k.ptr, s, @intCast(f.start));
            // the host's work between this pass's chunks (Stream.between: the prefill interleave's rounds, parked work's
            // service): the read-ahead joined and the prompt's ids gone first, as at a piece's end and on rank 1, which
            // reads ahead again at the next chunk
            if (f.start < len and s.images.len == 0) if (s.between) |b| if (b.wanted(b.ptr)) {
                m.joinAhead();
                m.fill_seq = &.{};
                try b.run(b.ptr);
                m.fill_seq = seq;
            };
        }
        if (f.start < f.len) return false;
        const done = f.*;
        self.fill = null;
        if (self.prof) |*p| p.prefills += 1;
        if (!done.head) return error.NoPromptLogits;
        // a state read back from disk: the drafter's rings were rebuilt by the replay before the first draft
        if (self.spill != null and m.drafting() and done.restored) try m.spillCheckResumed(done.slot, done.len);
        const logits = try m.promptLogitsHost();
        if (logits_sha) {
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
            std.debug.print("{{\"prompt_logits\": \"{s}\", \"sha256\": \"{s}\"}}\n", .{ s.id, std.fmt.bytesToHex(dg, .lower) });
        }
        const token = try draw(logits, done.len, s.sampling);
        self.streams.getPtr(s).?.first = self.take(token);
        return true;
    }

    /// A paused pass resumes: both ranks' model views back on its slot and extent, whose rows, rings and ids it kept.
    fn fillResume(self: *Lanes) !void {
        const f = self.paused.?;
        const l = self.streams.get(f.s) orelse return error.UnknownStream;
        if (try self.begin(.fill_resume)) {
            try self.wire.int(self.gpa, u32, @intCast(f.slot));
            try self.wire.int(self.gpa, u64, l.base);
            try self.flush();
        }
        try self.m.fillBegin(f.slot, l.base);
        self.fill = f;
        self.paused = null;
    }

    /// A prompt pass's start: its slot and extent, its images, rank 1's fill and the kept state it resumes from.
    fn fillOpen(self: *Lanes, s: *lanes.Stream) !void {
        const m = self.m;
        const gpa = self.gpa;
        const ids = s.prompt();
        const len = ids.len;
        if (len == 0) return error.EmptyPrompt;
        const size = std.mem.alignForward(usize, len + s.max_new + m.eng.round_rows + 2, model.chunk_rows);
        if (size > m.request_cap) return error.PromptTooLong; // (the pool's, unless TF_DS_POOL or TF_DS_KV_SHARD part it)
        // TF_DS_KV_SHARD: parts.zig's planner places the extent inside one part (placeParts: part-local placeSoft and
        // eviction, borrowed twins, carries, live moves) and the lane carries its part into every round (Win.part)
        const kv = self.kv;
        if (kv == null and m.kvs != null) return error.KvShardNotStarted; // (startKv at open)
        if (self.streams.fetchRemove(s)) |old| {
            self.used[old.value.slot] = false;
            if (kv) |k| try k.release(old.value.id);
        }
        const slot = std.mem.indexOfScalar(bool, self.used[0..self.m.streams], false) orelse return error.NoFreeSlot;
        if (self.spill) |t| t.bumpSlot(slot);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        // a kept state of this prompt's prefix: the pass resumes at its chunk end, in its extent, or in one of its own
        // with the kept rows copied there when a live stream holds that extent (a burst's second request on a prompt)
        s.cached = 0;
        const parted: ?PlacedParts = if (kv) |k| try self.placeParts(k, s, size, slot, len) else null;
        errdefer if (parted) |pp| kv.?.release(pp.id) catch |err| std.log.err("dsv41: a failed prompt pass kept its part's extent: {s}", .{@errorName(err)});
        var kept: ?*Snap = if (parted) |pp| pp.kept else null;
        var moved = if (parted) |pp| pp.moved else false;
        if (parted == null) if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            // (the cache matched the images before its rows by their digests: its image rows are this prompt's)
            if (self.usable(sn, len)) {
                kept = sn;
                // in place unless the extent runs past the pool, a live stream holds it, or another kept state has
                // rows past the kept ones (resuming there would stale it: the rows go to a free extent instead, as
                // the Python lane's _place resumes "here" only in a free extent)
                moved = sn.base + size > m.pool_cap or clashes(taken[0..nt], sn.base, sn.base + size) or self.keptOver(sn.base + sn.at, sn.base + size, sn);
            } else s.reuse_failed = true;
        };
        const base = if (parted) |pp| pp.base else if (kept != null and !moved) kept.?.base else try self.placeKept(taken[0..nt], size, kept);
        if (kept) |sn| {
            sn.wanted = 0;
            if (!moved) sn.end = @max(sn.end, base + size); // its extent now: the next resume in place finds it held
        }
        if (kept) |sn| if (moved and sn.stale) { // the placement needed the kept rows' room: the prompt runs from 0
            if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} found no room to move to ({d} positions); the prompt runs from 0", .{ sn.at, sn.base, size });
            kept = null;
            s.reuse_failed = true;
        };
        // the rows this stream writes are no other kept state's
        if (parted == null) self.staleOver(if (kept != null and !moved) base + kept.?.at else base, base + size, kept);
        try self.streams.put(gpa, s, .{ .slot = slot, .base = base, .end = base + size, .id = if (parted) |pp| pp.id else 0, .part = if (parted) |pp| pp.part else 0, .gen = if (parted) |pp| pp.gen else 0 });
        self.used[slot] = true;
        errdefer {
            _ = self.streams.remove(s);
            self.used[slot] = false;
        }
        const seq = &self.seqs[slot];
        try seq.resize(gpa, len);
        for (seq.items, ids) |*q, id| q.* = @intCast(id);
        // images: each span's positions dead in the sequence (negative: no Engram n-gram reaches into or across one),
        // its rows ready (a gate's) or the tower's (rank 0), all of them sent to rank 1 after the fill's start
        var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
        defer spans.deinit(gpa);
        var rows: std.ArrayList(u8) = .empty;
        defer rows.deinit(gpa);
        if (s.images.len > 0) {
            const image_id = m.cfg.image_token orelse return error.NoVision;
            const row_bytes = m.cfg.hidden * 2;
            var at_row: usize = 0;
            for (s.images) |img| {
                if (img.tokens == 0 or @as(usize, img.at) + img.tokens > len) return error.BadImageSpan;
                for (seq.items[img.at..][0..img.tokens]) |*q| {
                    if (q.* != @as(i64, image_id)) return error.BadImageSpan;
                    q.* = -1;
                }
                if (img.rows) {
                    if (img.bytes.len != img.tokens * row_bytes) return error.BadImageRows;
                    try rows.appendSlice(gpa, img.bytes);
                } else {
                    // a prepared picture (vision.zig): the tower's span rows, here on rank 0
                    const v = self.vision orelse return error.NoVisionTower;
                    if (try v.rows(img.bytes, gpa, &rows) != img.tokens) return error.BadImageSpan;
                }
                try spans.append(gpa, .{ .at = img.at, .len = img.tokens, .row = at_row });
                at_row += img.tokens;
            }
        }
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, base);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, base);
        // (its twins' view, bound as the follower binds it at fill_begin; it stays the fill on both ranks past the last
        // chunk, until the next fill or its release: the first decode round's untwin unbinds on both)
        if (parted) |pp| try kv.?.fillBegin(pp.id, base);
        if (spans.items.len > 0) {
            // the spans and their rows, image_piece bytes of rows a frame (a follower's frame buffer holds 64 KiB and
            // the ids of a full pool)
            var off: usize = 0;
            while (off < rows.items.len) {
                const n = @min(image_piece, rows.items.len - off);
                if (try self.begin(.fill_images)) {
                    try self.wire.int(gpa, u32, @intCast(spans.items.len));
                    for (spans.items) |sp| {
                        try self.wire.int(gpa, u64, sp.at);
                        try self.wire.int(gpa, u64, sp.len);
                    }
                    try self.wire.int(gpa, u64, off);
                    try self.wire.buf.appendSlice(gpa, rows.items[off..][0..n]);
                    try self.flush();
                }
                off += n;
            }
            try m.setImages(spans.items, rows.items);
        }
        const replay = len -| m.cfg.window;
        var start: usize = 0;
        if (kept) |sn| {
            if (try self.begin(.snap_restore)) {
                try self.wire.int(gpa, u64, sn.id);
                try self.wire.int(gpa, u32, @intCast(slot));
                try self.wire.int(gpa, u64, if (parted != null) base else sn.base); // (TF_DS_KV_SHARD: the plan's transfer ran)
                try self.wire.int(gpa, u64, base);
                try self.wire.int(gpa, u64, sn.at);
                try self.flush();
            }
            if (base != sn.base and parted == null) {
                if (self.spill) |t| t.bump(base, base + sn.at);
                try m.copyExtent(sn.base, base, sn.at);
            }
            try m.snapCopy(slot, sn.buf.ptr, false);
            start = sn.at;
            s.cached = @intCast(sn.at);
        }
        // where this pass keeps states: the cache's marks and its last chunk end before the replay (the served lane's
        // kept prompt), each at the chunk end at or before it
        const last_end = @min(replay, len - 1) / model.chunk_rows * model.chunk_rows;
        self.fill = .{ .s = s, .slot = slot, .len = len, .replay = replay, .last_end = last_end, .start = start, .base = base, .restored = if (kept) |sn| sn.restored else false, .part = if (parted) |pp| pp.id else null };
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.first;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window as one round at its stream's cache length: the pending token, the held drafts (or the host's), each
    /// row drawn at its keyed position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (windows.len > max_streams) return error.WindowTooWide;
        var ids: [max_streams][max_rows]i32 = undefined;
        var wins: [max_streams]Win = undefined;
        try self.roundWins(windows, &ids, &wins);
        const t0 = m.now();
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            for (wins[0..windows.len]) |w| try writeWin(&self.wire, gpa, w, self.kv != null);
            try self.flush();
        }
        const t1 = m.now();
        if (self.spill) |t| for (wins[0..windows.len]) |w| t.bump(w.base + w.start, w.base + w.start + w.ids.len);
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        try m.verify(rows, self.built.absorb[0..windows.len]);
        const t2 = m.now();
        // every row greedy: the device's argmax (R tokens back); else the logits for the host's draws
        var greedy = true;
        for (windows) |w| {
            if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                greedy = false;
            };
        }
        const tokens: []const u32 = if (greedy) try m.roundArgmax(rows.ids.len) else &.{};
        const logits: []const f32 = if (greedy) &.{} else try m.roundLogits(rows.ids.len);
        const t3 = m.now();
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
        };
        const V = m.vocab;
        for (windows, out, 0..) |w, *o, k| {
            const row0 = self.built.wins[k].row;
            for (0..w.rows()) |r| o.sampled[r] = if (greedy) tokens[row0 + r] else try draw(logits[(row0 + r) * V ..][0..V], w.positions[r], w.stream.sampling);
            const l = self.streams.getPtr(w.stream).?;
            @memcpy(o.drafts[0..w.held], l.held[0..w.held]);
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            l.nheld = 0;
        }
    }

    /// verifyFn's windows as both ranks build them (each stream's Lane: slot, extent, part, generation; its pending
    /// row, held drafts and host drafts), then the planner's round (kvRound: its frames go out before the verify's).
    fn roundWins(self: *Lanes, windows: []const be.Window, ids: *[max_streams][max_rows]i32, wins: *[max_streams]Win) !void {
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            const rows = w.rows();
            if (rows > window_rows) return error.WindowTooWide;
            const start: usize = @intCast(w.stream.cache_len);
            for (w.positions, 0..) |p, r| if (p != start + 1 + r) return error.PositionMismatch;
            if (w.held > l.nheld) return error.HeldMismatch;
            ids[k][0] = @intCast(w.pending);
            for (l.held[0..w.held], 0..) |t, j| ids[k][1 + j] = @intCast(t);
            for (w.tokens, 0..) |t, j| ids[k][1 + w.held + j] = @intCast(t);
            wins[k] = .{ .slot = l.slot, .base = l.base, .end = l.end, .start = start, .absorb = self.m.drafting() and w.stream.drafts, .ids = ids[k][0..rows], .part = if (self.kv != null) l.part else -1, .gen = l.gen };
        }
        if (self.kv) |kvp| try self.kvRound(kvp, windows);
    }

    /// The caches are positional: the next window starts at the stream's new length and writes over the rows past it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One drafter pass for every stream asked: its pending token at its new length, `depth` drafts held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (m.dr == null) return error.NoDraftHead;
        const block = m.dr.?.n;
        var tokens: [max_streams]i64 = undefined;
        var q0: [max_streams]i64 = undefined;
        var slots: [max_streams]i64 = undefined;
        var depths: [max_streams]usize = undefined;
        var asked: [max_streams]*Lane = undefined;
        var n: usize = 0;
        var steps: usize = 0;
        for (requests) |r| {
            if (r.lanes != null or r.early) return error.TreesNotBuilt;
            const l = self.streams.getPtr(r.stream) orelse return error.UnknownStream;
            l.nheld = 0;
            if (r.depth == 0) continue;
            if (n == max_streams) return error.TooManyStreams;
            if (r.position != r.stream.cache_len + 1) return error.PositionMismatch;
            const token: u32 = if (r.rows == null) self.value(r.first orelse return error.NoFirstToken) else (if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.NoFollowToken);
            tokens[n] = token;
            q0[n] = @intCast(r.position - 1);
            slots[n] = @intCast(l.slot);
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        if (try self.begin(.pass)) {
            try self.wire.int(gpa, u32, @intCast(n));
            try self.wire.int(gpa, u32, @intCast(steps));
            for (0..n) |i| {
                try self.wire.int(gpa, i64, tokens[i]);
                try self.wire.int(gpa, i64, q0[i]);
                try self.wire.int(gpa, i64, slots[i]);
            }
            try self.flush();
        }
        const t0 = m.now();
        try m.pass(tokens[0..n], q0[0..n], slots[0..n], steps);
        if (self.prof) |*p| {
            p.pass += m.now() - t0;
            p.passes += 1;
            p.pass_streams += n;
        }
        const d = &m.dr.?;
        for (asked[0..n], depths[0..n], 0..) |l, k, i| {
            for (0..k) |j| {
                l.held[j] = @intCast(d.drafts[i][j]);
                l.confs[j] = d.confs[i][j];
            }
            l.nheld = k;
        }
    }

    /// The chance that each held draft lands with every earlier one, as the round loop's allocation reads it: the
    /// running product of the confidences' sigmoids (multi.py's conf rule: a sigmoid is the draft's survival once
    /// the earlier drafts survived, and the prefix survivals' sum is the expected kept drafts; draft.chooseK).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return false;
        if (out.len > l.nheld) return false;
        if (self.fix_k) |k| {
            for (out, 0..) |*o, j| o.* = if (j < k) 1.0 else 0.0;
            return true;
        }
        survivals(l.confs[0..out.len], out);
        return true;
    }

    /// Whether a kept state still restores for a prompt of `len` tokens: live, not stale, a chunk end before the
    /// prompt's replay (the rows before it ran the encoder layers alone, as a fresh pass of this prompt runs them).
    fn usable(self: *const Lanes, sn: *const Snap, len: usize) bool {
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, @constCast(sn)) == null) return false;
        if (self.kv) |k| if (k.planner.keptIndex(sn.id) == null) return false; // (TF_DS_KV_SHARD: its rows the planner holds)
        return !sn.stale and !sn.loading and sn.at > 0 and sn.at < len and sn.at % model.chunk_rows == 0 and sn.at <= len -| self.m.cfg.window;
    }

    /// The first extent that fits around the live streams and the kept states. Kept states give their room oldest
    /// first and `source` (the state this prompt resumes from, its rows copied here) last, as the Python lane's _place
    /// takes the oldest other kept prompt before the source; a state whose rows lie in a live stream's extent gives
    /// none (that stream holds the room) and stays. When a live stream holds `source`, the request first waits up to
    /// held_wait_ns for that stream to end (error.ContextFull: the host tries it again; it then resumes in place) if
    /// the copy needs another state's room or would leave none for another request of its size: a copy that fills the
    /// pool makes the next prompt evict a kept state (the 5 x 500K check with the duplicate admitted fourth). Kept
    /// states hold their stream's whole extent back while a placement still fits so (placeSoft): its tail stays free
    /// for the resume in place, which then stales nothing.
    fn placeKept(self: *Lanes, live: []const [2]usize, size: usize, source: ?*Snap) !usize {
        const gpa = self.gpa;
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(gpa);
        var rows: std.ArrayList([2]usize) = .empty;
        defer rows.deinit(gpa);
        const held = if (source) |sn| !sn.stale and clashes(live, sn.base, sn.base + sn.at) else false;
        while (true) {
            kept.clearRetainingCapacity();
            rows.clearRetainingCapacity();
            try rows.appendSlice(gpa, live);
            for (self.snaps.items) |sn| if (!sn.stale) {
                try kept.append(gpa, .{ sn.base, sn.at, sn.end });
                try rows.append(gpa, .{ sn.base, sn.base + sn.at });
            };
            const got = try placeSoft(gpa, live, kept.items, size, self.m.pool_cap);
            if (held and self.patient(source.?)) {
                const b = got orelse return error.ContextFull;
                return if (try roomAfter(gpa, rows.items, b, size, self.m.pool_cap)) b else error.ContextFull;
            }
            if (got) |b| return b;
            const victim = for (self.snaps.items) |sn| {
                if (!sn.stale and !sn.loading and sn != source and !clashes(live, sn.base, sn.base + sn.at)) break sn;
            } else if (source) |sn| (if (!sn.stale and !clashes(live, sn.base, sn.base + sn.at)) sn else return error.ContextFull) else return error.ContextFull;
            noteStale(victim, "a placement needed its room (oldest first)");
            if (self.spill) |t| t.guardSnap(victim.id, 0); // its unpublished write stages or cancels before the rows go
            victim.stale = true;
        }
    }

    /// Whether a kept state other than `except` and its shorter siblings has rows in [lo, hi); a longer one may be another conversation's.
    fn keptOver(self: *const Lanes, lo: usize, hi: usize, except: *const Snap) bool {
        for (self.snaps.items) |sn| if (sn != except and !sn.stale and (sn.base != except.base or sn.at > except.at) and sn.base < hi and lo < sn.base + sn.at) return true;
        return false;
    }

    /// Whether a request still waits for the live stream that holds `sn` (held_wait_ns from its first wait).
    fn patient(self: *const Lanes, sn: *Snap) bool {
        const t = self.m.now();
        if (sn.wanted == 0) sn.wanted = t;
        return t - sn.wanted < held_wait_ns;
    }

    /// Kept states with rows in [lo, hi) go stale (a stream writes there), `except` aside.
    fn staleOver(self: *Lanes, lo: usize, hi: usize, except: ?*Snap) void {
        for (self.snaps.items) |sn| if (sn != except and sn.base < hi and lo < sn.base + sn.at) {
            if (!sn.stale) {
                noteStale(sn, "a stream writes over its rows");
                if (self.spill) |t| t.guardSnap(sn.id, @intCast(@max(lo, sn.base) - sn.base));
            }
            sn.stale = true;
        };
    }

    /// The prompt cache's Snapshots.usable (with the spill tier): a stale, loading or ineligible state is passed over
    /// before lengths are compared, so it never shadows a usable state or a durable point.
    pub fn usableFn(ptr: *anyopaque, saved: *anyopaque, len: u32) bool {
        return of(ptr).usable(@ptrCast(@alignCast(saved)), len);
    }

    /// The prompt cache's Snapshots functions (native.zig hands them to the CUDA host).
    pub fn snapBytesFn(ptr: *anyopaque, at: u32) u64 {
        _ = at;
        return of(ptr).m.snapBytes();
    }

    /// The live state at `at` (the pass stands at that chunk end): the slot's own part copied, the extent's rows kept.
    pub fn snapSaveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = of(ptr);
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (at == 0 or at % model.chunk_rows != 0 or at > s.prompt_len -| self.m.cfg.window) return error.NotAChunkEnd;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = l.base, .end = l.end, .part = l.part, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .slot = @intCast(l.slot) };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        if (try self.begin(.snap_save)) {
            try self.wire.int(self.gpa, u64, sn.id);
            try self.wire.int(self.gpa, u32, @intCast(l.slot));
            try self.flush();
        }
        try self.m.snapCopy(l.slot, sn.buf.ptr, true);
        // TF_DS_KV_SHARD: the planner keeps it too (its rows held in its part); a refused keep leaves the state unusable
        // (usable asks the planner), never one whose rows a placement may take
        if (self.kv) |k| k.keep(sn.id, l.id, at) catch |err| std.log.err("dsv41: the parts' planner did not keep a prompt state: {s}", .{@errorName(err)});
        // PORT(round3): the spill tier's writes of this point mark it pending (planner.setPending) and salt its block
        // lineage with its part (planner.saltOf, parts.Root.inPart); spill with TF_DS_KV_SHARD is refused until then
        return sn;
    }

    pub fn snapRestoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDropFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        if (self.spill) |t| t.dropSnap(sn.id); // (rule b: a pending small-state capture of its buffer done first)
        if (self.begin(.snap_drop) catch false) {
            self.wire.int(self.gpa, u64, sn.id) catch {};
            self.flush() catch |err| std.log.err("dsv41: rank 1 kept a dropped prompt state: {s}", .{@errorName(err)});
        }
        // TF_DS_KV_SHARD: the planner's record goes too (deferred while a copy or carry still reads its rows)
        if (self.kv) |k| k.drop(sn.id) catch |err| std.log.err("dsv41: the parts' planner kept a dropped prompt state: {s}", .{@errorName(err)});
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    // -- the spill tier's family and wire (spill_tier.Family, spill_tier.Wire) ---------------------------------------

    pub fn spillFamily(self: *Lanes) spill_tier.Family {
        return .{ .ptr = self, .reserve = spillReserve, .install = spillInstall, .discard = spillDiscard, .info = spillInfo };
    }

    pub fn spillWire(self: *Lanes) spill_tier.Wire {
        return .{ .ctx = self, .frame = spillFrame, .links = self.m.followers() };
    }

    /// A restore's destination: an extent placed like a stream's (kept states give their room oldest first, guarded),
    /// a fresh Snap holding it back (loading: not usable until every rank installed it) and its buffer. `room`: the
    /// waiting prompt's length; when an extent of it and a growth margin fits without taking any kept state's room,
    /// the Snap's extent (`end`) holds that much, so the resume runs in place (no copyExtent of the restored rows).
    fn spillReserve(ptr: *anyopaque, at: usize, room: usize) anyerror!spill_tier.Dest {
        const self = of(ptr);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        const want = @min(std.mem.alignForward(usize, @max(room, at) + growth_rows, model.chunk_rows), self.m.pool_cap);
        const roomy = if (want > at) try self.placeFree(taken[0..nt], want) else null;
        const base = roomy orelse try self.placeKept(taken[0..nt], at, null);
        self.staleOver(base, base + at, null);
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = base, .end = base + (if (roomy != null) want else at), .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .loading = true };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        return .{ .saved = sn, .id = sn.id, .base = base, .buf = sn.buf.ptr };
    }

    /// An extent of `size` around the live streams and the kept states' rows (placeSoft's rules: their held extents
    /// first), staling nobody: null when none.
    fn placeFree(self: *Lanes, live: []const [2]usize, size: usize) !?usize {
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(self.gpa);
        for (self.snaps.items) |sn| if (!sn.stale) try kept.append(self.gpa, .{ sn.base, sn.at, sn.end });
        return placeSoft(self.gpa, live, kept.items, size, self.m.pool_cap);
    }

    fn spillInstall(_: *anyopaque, saved: *anyopaque) void {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.loading = false;
        sn.restored = true;
    }

    /// A restore that did not commit: its Snap goes (the followers drop theirs with the abort, no snap_drop).
    fn spillDiscard(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, sn)) |i| _ = self.snaps.orderedRemove(i);
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    fn spillInfo(_: *anyopaque, saved: *anyopaque) spill_tier.SnapInfo {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf.ptr, .slot = sn.slot };
    }

    /// A spill frame (Kind rank_wire or spill) to every follower, in the ordered stream.
    pub fn spillFrame(ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void {
        const self = of(ctx);
        const k: Kind = switch (kind) {
            10 => .rank_wire,
            11 => .spill,
            else => return error.BadFrame,
        };
        if (try self.begin(k)) {
            try self.wire.buf.appendSlice(self.gpa, sub);
            try self.flush();
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.fill) |f| if (f.s == s) {
            self.m.joinAhead();
            self.m.fill_seq = &.{};
            self.fill = null;
        };
        if (self.paused) |f| if (f.s == s) {
            self.paused = null;
        };
        const kv = self.streams.fetchRemove(s) orelse {
            if (self.kv) |k| k.forgetKey(@intFromPtr(s)); // (a request that never placed: its refusal record goes)
            return;
        };
        self.used[kv.value.slot] = false;
        if (self.kv) |k| k.release(kv.value.id) catch |err| std.log.err("dsv41: the parts' planner kept a released stream: {s}", .{@errorName(err)});
    }
};

/// Prefix survivals: out[j] = the product of sigmoid(conf[i]) for i <= j, in float64 (libm's exp, as chooseK).
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= 1.0 / (1.0 + exp(-@as(f64, cf)));
        o.* = surv;
    }
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The parts' planner for this model: its agreed pool in one part a rank, and the owner-only bytes a carried token
/// costs (carryBytes over the model's kv sources).
pub fn kvConfig(m: *const Model) parts.Config {
    var buf: [16]shard.Source = undefined;
    return .{ .cap = m.pool_cap, .parts = @intCast(m.world), .carry_bytes = carryBytes(m.kvSources(&buf)) };
}

/// The bytes a carried token costs, rounded up: every source's compressed codes and scales at its ratio (the owner's
/// rows, shard.planeBytes over a whole number of every ratio's blocks, so no ratio's share truncates; the index
/// keys copy on every rank and are not sent). (3/2 + 1) x 288 = 720 at this checkpoint.
pub fn carryBytes(sources: []const shard.Source) usize {
    var owner: [16]shard.Source = undefined; // (Model.kvSources' bound)
    std.debug.assert(sources.len <= owner.len);
    var tokens: usize = 1;
    for (sources, owner[0..sources.len]) |s, *o| {
        o.* = .{ .ratio = s.ratio, .comp_row = s.comp_row, .idx_row = 0 };
        tokens = tokens / std.math.gcd(tokens, s.ratio) * s.ratio;
    }
    const bytes = shard.planeBytes(owner[0..sources.len], tokens, 1, false);
    return (bytes + tokens - 1) / tokens;
}

/// The device side of the planner's ops on this rank (Model.copyExtent, Model.bindTwin).
fn modelDev(m: *Model) kv_lanes.Dev {
    return .{ .ctx = m, .check = devCheck, .copy = devCopy, .twin = devTwin, .wait = devWait };
}

fn devCheck(ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    try m.checkExtent(from, to, rows);
}

/// The transfers enqueued so far done within `timeout_ns`: an event after them polled (a stream synchronize would
/// wait forever for an NCCL half whose peer never enqueued its own).
fn devWait(ctx: *anyopaque, timeout_ns: u64) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    var ev = try cuda.Event.init(m.ctx.d, false);
    defer ev.deinit();
    try ev.record(m.stream);
    const deadline = m.now() +| timeout_ns;
    while (!try ev.done()) {
        if (m.now() > deadline) return error.TransferTimeout;
        std.atomic.spinLoopHint();
    }
}

fn devCopy(ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    try m.copyExtent(from, to, rows);
}

fn devTwin(ctx: *anyopaque, base: usize, on: bool, local: ?usize) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    try m.bindTwin(base, on, local);
}

/// A primitive this rank refused: rank 0 runs the same one on the same inputs and refuses it at the same step (its
/// request ends with the error), so the follower reports it and takes the next frame instead of ending.
fn refused(err: anyerror) void {
    std.debug.print("{{\"follow_refused\": \"{s}\"}}\n", .{@errorName(err)});
}

/// A follower's verify times with Model.prof (a profile on ranks 1-3: rounds, issue and wall ns summed).
pub var follow_prof: struct { rounds: u64 = 0, enqueue: u64 = 0, forward: u64 = 0 } = .{};

/// Rank 1 (each of ranks 1-3 on the four-node split): the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model) !void {
    return followWith(gpa, m, null);
}

/// A follower's kept states as its spill side sees them (spill_tier.FollowerFamily).
const FollowKept = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    kept: *std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer),
    loading: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty, // restores' destinations, not yet installed
    seqs: *[max_streams]std.ArrayList(i32),

    fn family(f: *FollowKept) spill_tier.FollowerFamily {
        return .{ .ptr = f, .reserve = reserve, .install = install, .discard = discard, .buf = buf, .ids = ids };
    }

    fn of(p: *anyopaque) *FollowKept {
        return @ptrCast(@alignCast(p));
    }

    fn reserve(p: *anyopaque, id: u64) anyerror!u64 {
        const f = of(p);
        if (f.kept.contains(id) or f.loading.contains(id)) return error.SnapshotExists;
        try f.loading.ensureUnusedCapacity(f.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(f.m.ctx.d, f.m.snapBytes());
        f.loading.putAssumeCapacity(id, b);
        return b.ptr;
    }

    fn install(p: *anyopaque, id: u64) anyerror!void {
        const f = of(p);
        const kv = f.loading.fetchRemove(id) orelse return error.UnknownSnapshot;
        f.kept.put(f.gpa, id, kv.value) catch |e| {
            var b = kv.value;
            b.free();
            return e;
        };
    }

    fn discard(p: *anyopaque, id: u64) void {
        const f = of(p);
        var kv = f.loading.fetchRemove(id) orelse return;
        kv.value.free();
    }

    fn buf(p: *anyopaque, id: u64) ?u64 {
        return if (of(p).kept.get(id)) |b| b.ptr else null;
    }

    fn ids(p: *anyopaque, slot: u32) []const i32 {
        const f = of(p);
        return if (slot < max_streams) f.seqs[slot].items else &.{};
    }

    fn deinit(f: *FollowKept) void {
        var it = f.loading.valueIterator();
        while (it.next()) |b| b.free();
        f.loading.deinit(f.gpa);
    }
};

/// follow, with this rank's spill side when the tier is on (`spill`: made before the model's pool, its store opened at
/// the startup agreement rank 0 starts; the caller destroys it after this returns).
pub fn followWith(gpa: std.mem.Allocator, m: *Model, spill: ?*SpillFollower) !void {
    const peer = m.peer();
    const buf = try gpa.alloc(u8, (m.pool_cap + 1) * 4 + (1 << 16));
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var slot: usize = 0;
    var img_rows: std.ArrayList(u8) = .empty; // a prompt's image rows as their pieces come
    defer img_rows.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty; // rank 0's kept states, this rank's part
    defer {
        var ki = kept.valueIterator();
        while (ki.next()) |b| b.free();
        kept.deinit(gpa);
    }
    var fk: FollowKept = .{ .gpa = gpa, .m = m, .kept = &kept, .seqs = &seqs };
    defer fk.deinit();
    if (spill) |sd| sd.attach(fk.family());
    // on every exit, while fk and kept are alive: open restores ended (their buffers discarded through fk), the
    // family let go, then every copy still on the device waited for (no copy may read a buffer the defers above free)
    defer if (spill) |sd| {
        sd.detach();
        sd.quiesce();
    };
    var fill_base: usize = 0;
    // TF_DS_KV_SHARD: this rank's copy of the parts' planner, fed rank 0's ops (never with the spill tier on:
    // SpillWithKvShard, so the rank_wire frames go to one of the two)
    const kvf: ?*kv_lanes.Follower = if (m.kvs != null) try kv_lanes.Follower.init(gpa, m.rank, kvConfig(m), try kv_lanes.Settings.fromEnv(getenv), modelDev(m)) else null;
    defer if (kvf) |k| k.deinit();
    while (true) {
        // with spill writes or restores pending, they progress while rank 0 sends nothing (2 ms polls)
        if (spill) |sd| while (sd.busy() and !try peer.readable(2 * std.time.ns_per_ms)) sd.service(std.time.ns_per_ms);
        // the next primitive, polled for 20 ms before the read blocks (rank 0 sends it a fraction of a millisecond
        // after this rank's GPU goes idle in a decode; a sleeping thread would add its core's wake-up to every frame)
        var r: Reader = .{ .b = try peer.recvSpin(buf, 20 * std.time.ns_per_ms) };
        const kind: Kind = switch (try r.int(u8)) {
            1 => .fill_begin,
            2 => .fill_chunk,
            3 => .verify,
            4 => .pass,
            5 => .snap_save,
            6 => .snap_restore,
            7 => .snap_drop,
            8 => .fill_images,
            9 => .done,
            10 => .rank_wire,
            11 => .spill,
            12 => .fill_resume,
            else => return error.BadFrame,
        };
        if (try r.int(u64) != step) return error.OutOfStep;
        step += 1;
        if (spill) |sd| sd.service(std.time.ns_per_ms); // (rank 0 pumps between its primitives alike)
        // a chunk's read-ahead of the next is done before anything but that chunk (a prompt's ids may change)
        if (kind != .fill_chunk and kind != .snap_save and kind != .snap_restore and kind != .fill_images and kind != .rank_wire and kind != .spill) {
            m.joinAhead();
            m.fill_seq = &.{};
        }
        switch (kind) {
            .fill_begin => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                try r.ids(gpa, &seqs[slot]);
                fill_base = @intCast(base);
                if (spill) |sd| sd.bumpSlot(slot);
                if (m.fillBegin(slot, @intCast(base))) {
                    if (kvf) |k| try k.fillBegin(slot, @intCast(base)); // (its twins' view, as rank 0 binds it)
                } else |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_resume => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                fill_base = @intCast(base); // the paused pass's extent, for the spill tier's bumps (not a new prompt: no bumpSlot)
                m.fillBegin(slot, @intCast(base)) catch |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                if (m.fill_seq.len == 0) m.fill_seq = seqs[slot].items; // a piece after a round: the read-ahead again
                if (spill) |sd| sd.bump(fill_base + start, fill_base + start + n);
                _ = m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay) catch |err| refused(err);
            },
            .verify => {
                const nw = try r.int(u32);
                if (nw == 0 or nw > max_streams) return error.BadFrame;
                var wins: [max_streams]Win = undefined;
                for (0..nw) |k| {
                    wins[k] = try readWin(&r, gpa, &win_ids[k], kvf != null);
                    // TF_DS_KV_SHARD: a round naming a reused extent (its part or generation not the planner's) fails closed
                    if (kvf) |kf| try kf.checkWin(wins[k].slot, wins[k].base, wins[k].end, @intCast(wins[k].part), wins[k].gen);
                }
                const rows = try built.build(gpa, &seqs, wins[0..nw]);
                if (spill) |sd| for (wins[0..nw]) |w| sd.bump(w.base + w.start, w.base + w.start + w.ids.len);
                m.verify(rows, built.absorb[0..nw]) catch |err| refused(err);
                if (m.prof) {
                    follow_prof.rounds += 1;
                    follow_prof.enqueue += m.t_enqueue;
                    follow_prof.forward += m.t_forward;
                }
            },
            .pass => {
                const n = try r.int(u32);
                const steps = try r.int(u32);
                if (n == 0 or n > max_streams) return error.BadFrame;
                var tokens: [max_streams]i64 = undefined;
                var q0: [max_streams]i64 = undefined;
                var slots: [max_streams]i64 = undefined;
                for (0..n) |i| {
                    tokens[i] = try r.int(i64);
                    q0[i] = try r.int(i64);
                    slots[i] = try r.int(i64);
                }
                m.pass(tokens[0..n], q0[0..n], slots[0..n], steps) catch |err| refused(err);
            },
            .fill_images => {
                const count = try r.int(u32);
                var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
                defer spans.deinit(gpa);
                var row: usize = 0;
                for (0..count) |_| {
                    const at: usize = @intCast(try r.int(u64));
                    const n: usize = @intCast(try r.int(u64));
                    try spans.append(gpa, .{ .at = at, .len = n, .row = row });
                    row += n;
                }
                const off: usize = @intCast(try r.int(u64));
                const piece = r.rest();
                const total = row * m.cfg.hidden * 2;
                if (off != img_rows.items.len or off + piece.len > total or piece.len == 0) return error.BadFrame;
                try img_rows.appendSlice(gpa, piece);
                if (img_rows.items.len == total) {
                    m.setImages(spans.items, img_rows.items) catch |err| refused(err);
                    img_rows.clearRetainingCapacity();
                }
            },
            .snap_save => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                var b = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
                m.snapCopy(s, b.ptr, true) catch |err| {
                    b.free();
                    refused(err);
                    continue;
                };
                try kept.put(gpa, id, b);
            },
            .snap_restore => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                const from: usize = @intCast(try r.int(u64));
                const to: usize = @intCast(try r.int(u64));
                const at: usize = @intCast(try r.int(u64));
                const b = kept.get(id) orelse return error.UnknownSnapshot; // rank 0 restores only states both ranks keep
                if (from != to) {
                    if (spill) |sd| sd.bump(to, to + at);
                    m.copyExtent(from, to, at) catch |err| refused(err);
                }
                m.snapCopy(s, b.ptr, false) catch |err| refused(err);
            },
            .snap_drop => {
                const id = try r.int(u64);
                if (spill) |sd| sd.dropSnap(id); // its write guarded, its small-state capture done, before the free
                if (kept.fetchRemove(id)) |kv| {
                    var b = kv.value;
                    b.free();
                }
            },
            // one rank_wire Follower a rank: the spill tier's or the KV parts' (never both: SpillWithKvShard)
            .rank_wire => if (spill) |sd| try sd.wireFrame(peer, r.rest()) else if (kvf) |k| try k.onFrame(peer, r.rest()) else return error.Unsupported,
            .spill => try (spill orelse return error.SpillNotStarted).note(r.rest()),
            .done => {
                if (spill) |sd| {
                    const left = sd.flush(sd.flush_ns);
                    if (left > 0) std.log.info("dsv41 spill: {d} kept states were still unpublished at the flush deadline", .{left});
                }
                return;
            },
        }
    }
}

test "draft chances are the prefix survivals chooseK sums" {
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

test "extents take the first chunk-aligned gap" {
    const used = [_][2]usize{ .{ 0, 4096 }, .{ 6144, 8192 } };
    try std.testing.expectEqual(@as(?usize, 4096), place(&used, 2048, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 8192), place(&used, 4096, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 0), place(&.{}, 2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), place(&used, 4096, 10240));
}

test "a frame's fields read back as written" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 7);
    try w.int(gpa, u64, 1 << 40);
    try w.ids(gpa, &.{ 1, -2, 129279 });
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.verify)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 1 << 40), try r.int(u64));
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    try r.ids(gpa, &got);
    try std.testing.expectEqualSlices(i32, &.{ 1, -2, 129279 }, got.items);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
}

test "a round's rows: each window at its positions, its ids written over the rejected rows" {
    const gpa = std.testing.allocator;
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    try seqs[0].appendSlice(gpa, &.{ 10, 11, 12, 13, 99, 98 }); // positions 4.. hold a rejected window
    try seqs[2].appendSlice(gpa, &.{ 20, 21 });
    var b: Built = .{};
    const rows = try b.build(gpa, &seqs, &.{
        .{ .slot = 0, .base = 0, .end = 4096, .start = 4, .absorb = true, .ids = &.{ 14, 15 } },
        .{ .slot = 2, .base = 4096, .end = 8192, .start = 2, .absorb = false, .ids = &.{22} },
    });
    try std.testing.expectEqualSlices(i64, &.{ 14, 15, 22 }, rows.ids);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 2 }, rows.pos);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2 }, rows.slots.?);
    try std.testing.expectEqualSlices(i64, &.{ 4096, 4096, 8192 }, rows.ends.?);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 15 }, rows.windows.?[0].seq);
    try std.testing.expectEqualSlices(i32, &.{ 20, 21, 22 }, rows.windows.?[1].seq);
    try std.testing.expectEqual(@as(usize, 2), rows.windows.?[1].row);
    try std.testing.expectError(error.SequenceGap, b.build(gpa, &seqs, &.{.{ .slot = 1, .base = 0, .end = 2048, .start = 3, .absorb = false, .ids = &.{1} }}));
}

test "greedy draws take the first largest logit" {
    const row = [_]f32{ 0.5, 2.0, -1.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, null));
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, .{ .seed = 3, .temperature = 0 }));
}

test "retained histories preserve their replay and output reservations (PR #10: eight 100K histories)" {
    var m: Model = undefined;
    m.pool_cap = 1 << 20;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var snapshots: [8]Snap = undefined;
    for (&snapshots, 0..) |*sn, i| {
        const base = try ln.placeKept(&.{}, 102400, null);
        try std.testing.expectEqual(i * (102400 + growth_rows), base); // each with room to grow while the pool allows

        sn.* = .{ .id = i, .at = 98304, .base = base, .end = base + 102400, .buf = undefined };
        try ln.snaps.append(std.testing.allocator, sn);
    }
    // All eight 100K histories can resume within their reservations without invalidating a neighbor.
    for (&snapshots) |*sn| {
        try std.testing.expect(!ln.keptOver(sn.base + sn.at, sn.end, sn));
        ln.staleOver(sn.base + sn.at, sn.end, sn);
    }
    for (snapshots) |sn| try std.testing.expect(!sn.stale);
    // a larger continuation resumes in place within its room to grow
    try std.testing.expect(!ln.keptOver(snapshots[0].at, 104448, &snapshots[0]));
    // past that room it would overwrite the next cached prefix, so it relocates
    try std.testing.expect(ln.keptOver(snapshots[0].at, 102400 + growth_rows + 2048, &snapshots[0]));
    // At genuine capacity pressure, kept states may still be evicted; a live range never is.
    const live = [_][2]usize{.{ 0, 102400 }};
    try std.testing.expectEqual(@as(usize, 102400), try ln.placeKept(&live, 409600, null));
    try std.testing.expect(ln.snaps.items[1].stale and ln.snaps.items[3].stale);
    try std.testing.expect(!ln.snaps.items[4].stale);
    try std.testing.expect(!ln.snaps.items[0].stale); // its rows lie in the live stream: that stream holds the room
}

test "a prefix two conversations share: the second one resuming it in place would stale the first one's later marks" {
    var m: Model = undefined;
    m.pool_cap = 1 << 20;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    // the first conversation's extent holds the shared prefix's mark (system prompt and tools) and its own later one
    var shared: Snap = .{ .id = 0, .at = 18432, .base = 0, .end = 430080, .buf = undefined };
    var later: Snap = .{ .id = 1, .at = 395264, .base = 0, .end = 430080, .buf = undefined };
    try ln.snaps.append(std.testing.allocator, &shared);
    try ln.snaps.append(std.testing.allocator, &later);
    try std.testing.expect(ln.keptOver(shared.base + shared.at, shared.base + 55296, &shared)); // the second one moves
    try std.testing.expect(!ln.keptOver(later.base + later.at, later.base + 432128, &later)); // the first one's next turn stays
}

test "a prompt with images does not cut in between another pass's pieces" {
    var m: Model = undefined;
    m.io = std.testing.io;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    var long: lanes.Stream = undefined;
    long.images = &.{};
    var short: lanes.Stream = undefined;
    const image = [_]lanes.stream.Image{.{ .at = 0, .tokens = 1, .bytes = "" }};
    short.images = &image;
    ln.fill = .{ .s = &long, .slot = 0, .len = 16, .replay = 0, .last_end = 0, .start = 8 };
    try std.testing.expectError(error.FillBusy, ln.prefillPart(&short, 1));
    try std.testing.expect(ln.paused == null and ln.fill.?.s == &long); // the pass in pieces stays as it was
}

test "a restore's destination is held back from placement, never a victim and never usable until installed" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.pool_cap = 16 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    // a restore reading into [0, 4c), an ordinary kept state at [4c, 8c)
    var loading: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 4 * c, .buf = undefined, .loading = true };
    var kept: Snap = .{ .id = 2, .at = 4 * c, .base = 4 * c, .end = 8 * c, .buf = undefined };
    try ln.snaps.append(gpa, &loading);
    try ln.snaps.append(gpa, &kept);
    try std.testing.expect(!ln.usable(&loading, 8 * c)); // not until every rank installed it
    try std.testing.expect(ln.usable(&kept, 9 * c));
    // a placement that needs room past the free 8 chunks takes the ordinary kept state's, not the destination's
    const base = try ln.placeKept(&.{}, 12 * c, null);
    try std.testing.expect(kept.stale and !loading.stale);
    try std.testing.expectEqual(4 * c, base);
    // with nothing else to give, the destination still holds: the pool is full for this request
    kept.stale = false;
    kept.base = 4 * c;
    try std.testing.expectError(error.ContextFull, ln.placeKept(&.{.{ 4 * c, 16 * c }}, 4 * c, null));
    loading.loading = false;
    loading.restored = true;
    try std.testing.expect(ln.usable(&loading, 8 * c));
    // with the tier off nothing spill-related runs: the frame kinds before it keep their values
    try std.testing.expectEqual(@as(u8, 9), @intFromEnum(Kind.done));
    try std.testing.expectEqual(@as(u8, 10), @intFromEnum(Kind.rank_wire));
    try std.testing.expectEqual(@as(u8, 11), @intFromEnum(Kind.spill));
    try std.testing.expect(ln.spill == null);
}

test "a restore's destination holds the waiting prompt's room past its rows while that fits clear of the others" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.pool_cap = 64 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    var kept: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 5 * c, .buf = undefined };
    try ln.snaps.append(gpa, &kept);
    // clear of the kept state's held extent and its growth: placed past it, nothing staled
    try std.testing.expectEqual(@as(?usize, 5 * c + growth_rows), try ln.placeFree(&.{}, 8 * c));
    try std.testing.expect(!kept.stale);
    // no room for the hint anywhere: null (spillReserve then places the rows alone, as placeKept does)
    try std.testing.expectEqual(@as(?usize, null), try ln.placeFree(&.{.{ 4 * c, 64 * c }}, 8 * c));
    try std.testing.expect(!kept.stale);
}

test "a verify window's frame bytes: as ever with the switch off, its part and generation added with it on" {
    const gpa = std.testing.allocator;
    const win: Win = .{ .slot = 2, .base = 4096, .end = 8192, .start = 7, .absorb = true, .ids = &.{ 5, 6 }, .part = 1, .gen = 9 };
    var w: Wire = .{};
    defer w.deinit(gpa);
    try writeWin(&w, gpa, win, false);
    // the layout before TF_DS_KV_SHARD: verifyFn's fields in their order
    var old: Wire = .{};
    defer old.deinit(gpa);
    try old.int(gpa, u32, 2);
    try old.int(gpa, u64, 4096);
    try old.int(gpa, u64, 8192);
    try old.int(gpa, u64, 7);
    try old.int(gpa, u8, 1);
    try old.ids(gpa, &.{ 5, 6 });
    try std.testing.expectEqualSlices(u8, old.buf.items, w.buf.items);
    var ids: std.ArrayList(i32) = .empty;
    defer ids.deinit(gpa);
    var r: Reader = .{ .b = w.buf.items };
    const off = try readWin(&r, gpa, &ids, false);
    try std.testing.expectEqual(@as(i64, -1), off.part);
    try std.testing.expectEqual(@as(u64, 0), off.gen);
    try std.testing.expectEqual(w.buf.items.len, r.at);
    w.buf.clearRetainingCapacity();
    try writeWin(&w, gpa, win, true);
    try std.testing.expectEqual(old.buf.items.len + 9, w.buf.items.len);
    r = .{ .b = w.buf.items };
    const on = try readWin(&r, gpa, &ids, true);
    try std.testing.expectEqual(@as(i64, 1), on.part);
    try std.testing.expectEqual(@as(u64, 9), on.gen);
    try std.testing.expectEqualSlices(i32, &.{ 5, 6 }, on.ids);
    var bad = win;
    bad.part = -1;
    try std.testing.expectError(error.BadPart, writeWin(&w, gpa, bad, true));
}

/// The device side for the lanes tests: transfers and twin binds counted.
const TestDev = struct {
    copies: usize = 0,
    binds: usize = 0,

    fn dev(d: *TestDev) kv_lanes.Dev {
        return .{ .ctx = d, .check = checkFn, .copy = copyFn, .twin = twinFn, .wait = waitFn };
    }

    fn checkFn(_: *anyopaque, _: usize, _: usize, _: usize) anyerror!void {}

    fn waitFn(_: *anyopaque, _: u64) anyerror!void {}

    fn copyFn(ctx: *anyopaque, _: usize, _: usize, _: usize) anyerror!void {
        const d: *TestDev = @ptrCast(@alignCast(ctx));
        d.copies += 1;
    }

    fn twinFn(ctx: *anyopaque, _: usize, _: bool, _: ?usize) anyerror!void {
        const d: *TestDev = @ptrCast(@alignCast(ctx));
        d.binds += 1;
    }
};

/// A stream as placeParts and releaseFn read it.
fn testStream(id: []const u8, saved: ?*Snap) lanes.Stream {
    var s: lanes.Stream = undefined;
    s.id = id;
    s.reuse = .{ .saved = if (saved) |sn| @ptrCast(sn) else null };
    s.reuse_failed = false;
    return s;
}

fn testLanes(m: *Model) Lanes {
    m.pool_cap = 64 * model.chunk_rows;
    m.cfg.window = 0;
    m.io = std.testing.io;
    return .{ .gpa = std.testing.allocator, .m = m, .peer = null };
}

fn testPut(ln: *Lanes, s: *const lanes.Stream, slot: usize, size: usize, p: anytype) !void {
    try ln.streams.put(ln.gpa, s, .{ .slot = slot, .base = p.base, .end = p.base + size, .id = p.id, .part = p.part, .gen = p.gen });
    ln.used[slot] = true;
}

test "TF_DS_KV_SHARD lanes: part-local placement, a kept state resumed in place, the planner's stales reach the Snaps" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    var dev: TestDev = .{};
    var k = try kv_lanes.Kv.init(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{ .twins = false }, dev.dev(), .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer k.deinit();
    ln.kv = &k;
    // a conversation's first turn, kept at 8 blocks, then released
    var sa = testStream("a", null);
    const pa = try ln.placeParts(&k, &sa, 10 * c, 0, 10 * c);
    try std.testing.expect(pa.kept == null and pa.base + 10 * c <= 32 * c);
    try testPut(&ln, &sa, 0, 10 * c, pa);
    var s1: Snap = .{ .id = 1, .at = 8 * c, .base = pa.base, .end = pa.base + 10 * c, .buf = undefined, .part = pa.part };
    try ln.snaps.append(gpa, &s1);
    try k.keep(1, pa.id, 8 * c);
    Lanes.releaseFn(&ln, &sa);
    try std.testing.expectEqual(@as(usize, 0), k.planner.lives.items.len);
    // its next turn resumes it in place (the extent grows into its slack: the planner's kept end follows)
    var sb = testStream("b", &s1);
    const pb = try ln.placeParts(&k, &sb, 12 * c, 1, 9 * c);
    try std.testing.expect(pb.kept == &s1 and !pb.moved and !sb.reuse_failed);
    try std.testing.expectEqual(s1.base, pb.base);
    try std.testing.expectEqual(pa.base + 12 * c, k.planner.keptAt(1).?.end);
    try testPut(&ln, &sb, 1, 12 * c, pb);
    try std.testing.expectEqual(@as(usize, 0), dev.copies); // (in place: no transfer)
    // a long stream takes the other part; the conversation ends; a request that needs part 0 whole gives up the state
    var sc = testStream("c", null);
    const pc = try ln.placeParts(&k, &sc, 30 * c, 2, 30 * c);
    try std.testing.expect(pc.part != pb.part);
    try testPut(&ln, &sc, 2, 30 * c, pc);
    Lanes.releaseFn(&ln, &sb);
    var sd = testStream("d", null);
    const pd = try ln.placeParts(&k, &sd, 30 * c, 1, 30 * c);
    try std.testing.expectEqual(pb.part, pd.part);
    try std.testing.expect(s1.stale); // (kvStale: the planner's stale op)
    try std.testing.expect(!ln.usable(&s1, 9 * c));
    // a waiting request is counted once and forgotten when the host gives it up
    var se = testStream("e", null);
    try std.testing.expectError(error.ContextFull, ln.placeParts(&k, &se, 20 * c, 0, 20 * c));
    try std.testing.expectEqual(@as(usize, 1), k.waiting.items.len);
    Lanes.releaseFn(&ln, &se);
    try std.testing.expectEqual(@as(usize, 0), k.waiting.items.len);
    try testPut(&ln, &sd, 1, 30 * c, pd);
    Lanes.releaseFn(&ln, &sc);
    Lanes.releaseFn(&ln, &sd);
    try std.testing.expectEqual(@as(usize, 0), k.planner.lives.items.len);
}

test "TF_DS_KV_SHARD lanes: a live move rewrites the moved Lane and the Snaps inside it" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    var dev: TestDev = .{};
    var k = try kv_lanes.Kv.init(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{ .twins = false }, dev.dev(), .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer k.deinit();
    ln.kv = &k;
    var sx = testStream("x", null);
    const px = try ln.placeParts(&k, &sx, 14 * c, 0, 14 * c);
    try testPut(&ln, &sx, 0, 14 * c, px);
    var sz = testStream("z", null);
    const pz = try ln.placeParts(&k, &sz, 10 * c, 1, 10 * c);
    try testPut(&ln, &sz, 1, 10 * c, pz);
    try std.testing.expect(px.part != pz.part);
    var s2: Snap = .{ .id = 2, .at = 2 * c, .base = px.base, .end = px.base + 14 * c, .buf = undefined, .part = px.part };
    try ln.snaps.append(gpa, &s2);
    try k.keep(2, px.id, 2 * c);
    try k.decodeStart(px.id);
    try k.decodeStart(pz.id);
    try k.round(&.{ .{ .id = px.id, .valid = 2 * c }, .{ .id = pz.id, .valid = 3 * c } });
    // a 24-block request fits neither part: x (two blocks used) moves, with its kept state
    var sw = testStream("w", null);
    const pw = try ln.placeParts(&k, &sw, 24 * c, 2, 24 * c);
    const lx = ln.streams.get(&sx).?;
    try std.testing.expectEqual(pz.part, lx.part);
    try std.testing.expect(lx.base != px.base and lx.gen != px.gen);
    try std.testing.expectEqual(lx.base + 14 * c, lx.end);
    try std.testing.expectEqual(lx.base, s2.base);
    try std.testing.expectEqual(lx.part, s2.part);
    try std.testing.expectEqual(px.part, pw.part);
    try std.testing.expectEqual(@as(usize, 1), dev.copies);
    try testPut(&ln, &sw, 2, 24 * c, pw);
    // the next round names the moved extent's generation (a follower checks it)
    try std.testing.expectEqual(k.planner.lives.items[k.planner.liveIndex(lx.id).?].gen, lx.gen);
}

/// A window of the pending row and `n - 1` host drafts at the stream's cache length (verifyFn's input).
fn testWindow(s: *lanes.Stream, positions: []u64, tokens: []const u32) be.Window {
    for (positions, 0..) |*p, r| p.* = s.cache_len + 1 + r;
    return .{ .stream = s, .pending = 1, .held = 0, .tokens = tokens, .parents = null, .positions = positions[0 .. 1 + tokens.len] };
}

test "TF_DS_KV_SHARD lanes: a round's windows reach the follower with part and generation; a move right after a round crossing a block copies its rows; the old generation is refused" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    m.dr = null;
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    // rank 0's Kv with an in-process follower (kv_lanes.Pair), its hooks the lanes'
    const pr = try kv_lanes.Pair.initWith(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{ .twins = false }, .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer pr.deinit();
    ln.kv = &pr.kv;
    var sx = testStream("x", null);
    const px = try ln.placeParts(&pr.kv, &sx, 14 * c, 0, 14 * c);
    try testPut(&ln, &sx, 0, 14 * c, px);
    var sz = testStream("z", null);
    const pz = try ln.placeParts(&pr.kv, &sz, 10 * c, 1, 10 * c);
    try testPut(&ln, &sz, 1, 10 * c, pz);
    // x's round starts 3 rows before its second block ends and writes 6 rows into its third
    sx.cache_len = 2 * c - 3;
    sx.drafts = false;
    sz.cache_len = 3 * c;
    sz.drafts = false;
    var pos: [2][window_rows]u64 = undefined;
    const drafts = [_]u32{ 2, 3, 4, 5, 6 };
    const windows = [_]be.Window{ testWindow(&sx, &pos[0], &drafts), testWindow(&sz, &pos[1], drafts[0..1]) };
    var ids: [max_streams][max_rows]i32 = undefined;
    var wins: [max_streams]Win = undefined;
    try ln.roundWins(&windows, &ids, &wins);
    try std.testing.expectEqual(@as(usize, 2 * c + 3), pr.kv.planner.lives.items[pr.kv.planner.liveIndex(px.id).?].valid);
    try std.testing.expectEqual(@as(usize, 3 * c + 2), pr.kv.planner.lives.items[pr.kv.planner.liveIndex(pz.id).?].valid);
    try pr.same(); // (the first rounds' untwin plans reached the follower)
    // the verify frame's windows as the follower reads and checks them
    var w: Wire = .{};
    defer w.deinit(gpa);
    for (wins[0..2]) |x| try writeWin(&w, gpa, x, true);
    var r: Reader = .{ .b = w.buf.items };
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    var old_x: Win = undefined;
    for (0..2) |k| {
        const x = try readWin(&r, gpa, &got, true);
        try pr.fol.checkWin(x.slot, x.base, x.end, @intCast(x.part), x.gen);
        if (k == 0) old_x = x;
    }
    // a 24-block request fits neither part: x moves with every row the round wrote (whole blocks: 3, not 2)
    var sw = testStream("w", null);
    const pw = try ln.placeParts(&pr.kv, &sw, 24 * c, 2, 24 * c);
    try testPut(&ln, &sw, 2, 24 * c, pw);
    try std.testing.expectEqual(@as(usize, 1), pr.dev0.copies.items.len);
    try std.testing.expectEqual(@as(usize, 3 * c), pr.dev0.copies.items[0][2]);
    try std.testing.expectEqualSlices([3]usize, pr.dev0.copies.items, pr.dev1.copies.items);
    try pr.same();
    // the follower refuses a round naming x's old extent, and takes the next one at the new generation
    try std.testing.expectError(error.StaleExtent, pr.fol.checkWin(old_x.slot, old_x.base, old_x.end, @intCast(old_x.part), old_x.gen));
    sx.cache_len += 6;
    const next = [_]be.Window{testWindow(&sx, &pos[0], drafts[0..1])};
    try ln.roundWins(&next, &ids, &wins);
    try std.testing.expect(wins[0].gen != old_x.gen and wins[0].base != old_x.base);
    try pr.fol.checkWin(wins[0].slot, wins[0].base, wins[0].end, @intCast(wins[0].part), wins[0].gen);
    for ([_]*lanes.Stream{ &sx, &sz, &sw }) |s| Lanes.releaseFn(&ln, s);
    try pr.same();
    try std.testing.expectEqual(@as(usize, 0), pr.fol.planner.?.lives.items.len);
}

test "TF_DS_KV_SHARD lanes: a prompt pass failing after its placement (prefillFn's errdefer) frees the extent on both ranks" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    const pr = try kv_lanes.Pair.initWith(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{}, .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer pr.deinit();
    ln.kv = &pr.kv;
    var sa = testStream("a", null);
    const pa = try ln.placeParts(&pr.kv, &sa, 30 * c, 0, 30 * c);
    try pr.same();
    try std.testing.expect(pr.fol.planner.?.liveIndex(pa.id) != null);
    // (prefillFn's errdefer after placeParts: a fill chunk failed, the request cancelled)
    try pr.kv.release(pa.id);
    try pr.same();
    try std.testing.expectEqual(@as(usize, 0), pr.fol.planner.?.lives.items.len);
    // slot 0 and the whole part take the next request
    var sb = testStream("b", null);
    const pb = try ln.placeParts(&pr.kv, &sb, 30 * c, 0, 30 * c);
    try testPut(&ln, &sb, 0, 30 * c, pb);
    Lanes.releaseFn(&ln, &sb);
    try pr.same();
}

test "TF_DS_KV_SHARD lanes: a carried token's bytes are the owners' rows over whole blocks of every ratio, rounded up" {
    // this checkpoint's four kv sources (ratios 2, 2, 2, 1): (3/2 + 1) x 288 = 720, the index keys not sent
    const row = shard.comp_row_bytes;
    const idx = shard.idx_row_bytes;
    try std.testing.expectEqual(@as(usize, 720), carryBytes(&.{ .{ .ratio = 2, .comp_row = row, .idx_row = idx }, .{ .ratio = 2, .comp_row = row, .idx_row = idx }, .{ .ratio = 2, .comp_row = row, .idx_row = idx }, .{ .ratio = 1, .comp_row = row, .idx_row = idx } }));
    // ratios that do not divide the row: 72 + 2.25 a token rounds up to 75 (per-layer integer division said 74)
    try std.testing.expectEqual(@as(usize, 75), carryBytes(&.{ .{ .ratio = 4, .comp_row = row, .idx_row = idx }, .{ .ratio = 128, .comp_row = row, .idx_row = 0 } }));
}

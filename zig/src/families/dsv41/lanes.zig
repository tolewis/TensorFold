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
const rank_wire = @import("rank_wire.zig");
const fingerprint = @import("fingerprint.zig");

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
/// fill_resume (TF_FILL_CHUNKS' pieces) is 12: 11 stays unused (a frame of it is BadFrame), both ranks run one build.
/// With the spill tier on: rank_wire's frames (spill_tier: restores' agreement, guards, and the notices of rank 0's
/// write at keep and point metadata: keep/share/touch/forget, tracked like every other command). TF_DS_KV_SHARD adds
/// rank_wire too (rank_wire.zig's frames: the parts' agreement, each plan's ops, each transfer's
/// shard_xfer/shard_finish, the digest checks). Both on together share one rank_wire Leader (Lanes.rw, rank 0) and
/// one Follower a follower (PairFollower): one Tracker sees every frame of the pair. (Kind 11, the untracked spill
/// notes of round 2, is gone: a frame of it is BadFrame.)
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9, rank_wire = 10, fill_resume = 12 };

comptime {
    std.debug.assert(@intFromEnum(Kind.rank_wire) == @import("rank_wire.zig").lanes_kind);
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

// -- pictures in a resumed pass (recipe 0006; spill v2 restores take the same path as memory resumes) ------------------

/// A picture's span [at, at + tokens) of the runtime sequence `seq` (the prompt's ids, each the placeholder `image_id`
/// there) made dead: negative, so no Engram n-gram reaches into or across it.
pub fn markPicture(seq: []i32, at: usize, tokens: usize, image_id: u32) !void {
    if (tokens == 0 or at + tokens > seq.len) return error.BadImageSpan;
    for (seq[at..][0..tokens]) |*q| {
        if (q.* != @as(i64, image_id)) return error.BadImageSpan;
        q.* = -1;
    }
}

/// Whether a pass resuming at `resume_at` needs a picture's rows (the tower's run over it): a picture wholly before the
/// cut is in the resumed state's rows and needs none; one straddling or after it gets all of its rows, of which the
/// chunks embed only the positions at or past the cut (prompt.overlap).
pub fn pictureNeedsRows(at: usize, tokens: usize, resume_at: usize) bool {
    return at + tokens > resume_at;
}

/// A resumed sequence checked against its request: `ids` everywhere but the pictures' spans, which are negative
/// (spill_state.checkEngramSeq over the request's own pictures). The tests' oracle for what markPicture builds; a
/// restored resume needs no run of it (the pass's own comment says why).
pub fn checkPictureSeq(gpa: std.mem.Allocator, seq: []const i32, ids: []const u32, images: []const lanes.stream.Image) !void {
    const sp = try gpa.alloc([2]usize, images.len);
    defer gpa.free(sp);
    for (images, sp) |img, *x| x.* = .{ img.at, img.tokens };
    try model.Model.spill.checkEngramSeq(seq, ids, sp);
}

test "pictures in a resumed pass: spans dead in the sequence, rows only for those the cut does not hold, the Engram check" {
    const gpa = std.testing.allocator;
    const ph: u32 = 129264;
    // a prompt of 40 tokens: a picture at [4, 10), one straddling a cut at 16 ([14, 20)), one after it at [30, 34)
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i);
    const imgs = [_]lanes.stream.Image{ .{ .at = 4, .tokens = 6, .bytes = "" }, .{ .at = 14, .tokens = 6, .bytes = "" }, .{ .at = 30, .tokens = 4, .bytes = "" } };
    for (imgs) |img| @memset(ids[img.at..][0..img.tokens], ph);
    var seq: [40]i32 = undefined;
    for (&seq, ids) |*q, id| q.* = @intCast(id);
    for (imgs) |img| try markPicture(&seq, img.at, img.tokens, ph);
    try checkPictureSeq(gpa, &seq, &ids, &imgs);
    // resuming at 16: the first picture is in the state, the straddling one and the later one run the tower
    try std.testing.expect(!pictureNeedsRows(4, 6, 16));
    try std.testing.expect(pictureNeedsRows(14, 6, 16));
    try std.testing.expect(pictureNeedsRows(30, 4, 16));
    try std.testing.expect(!pictureNeedsRows(14, 2, 16)); // (ending exactly at the cut: wholly before it)
    try std.testing.expect(pictureNeedsRows(4, 6, 0)); // a fresh pass: every picture
    // a sequence whose picture positions were not made dead, or a text id changed: refused
    var live = seq;
    live[15] = @intCast(ph);
    try std.testing.expectError(error.EngramSeqMismatch, checkPictureSeq(gpa, &live, &ids, &imgs));
    var other = seq;
    other[12] += 1;
    try std.testing.expectError(error.EngramSeqMismatch, checkPictureSeq(gpa, &other, &ids, &imgs));
    // a picture span that does not hold the placeholder, or runs past the prompt: refused before anything runs
    var bad = seq;
    try std.testing.expectError(error.BadImageSpan, markPicture(&bad, 10, 3, ph));
    try std.testing.expectError(error.BadImageSpan, markPicture(&bad, 38, 4, ph));
    try std.testing.expectError(error.BadImageSpan, markPicture(&bad, 4, 0, ph));
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

/// TF_DS_SERVE_PROFILE: the served rank 0's backend times as one stderr line (ms a round; a pass's ms a pass).
/// `period_ms` is a round's start to the next one's; `other_ms` is what of it the backend calls do not hold (the round
/// loop, the server, the keep). The forward and the absorb are synchronized apart (Model.prof), as tf-dsv41-lanes
/// --profile has them.
fn printServeProfile(p: Profile, period: u64, periods: u64) void {
    const ms = struct {
        fn f(ns: u64, n: u64) f64 {
            return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
        }
    }.f;
    const period_ms = ms(period, periods);
    const pass_a_round = if (p.rounds == 0) 0 else @as(f64, @floatFromInt(p.pass)) / @as(f64, @floatFromInt(p.rounds)) / 1e6;
    const held = ms(p.send, p.rounds) + ms(p.forward, p.rounds) + ms(p.absorb, p.rounds) + ms(p.logits, p.rounds) + ms(p.sample, p.rounds) + pass_a_round;
    std.debug.print("{{\"rank\": 0, \"serve_profile\": {{\"rounds\": {d}, \"rows_a_round\": {d:.2}, \"period_ms\": {d:.3}, \"send_ms\": {d:.3}, \"enqueue_ms\": {d:.3}, \"forward_ms\": {d:.3}, \"absorb_ms\": {d:.3}, \"logits_ms\": {d:.3}, \"sample_ms\": {d:.3}, \"passes\": {d}, \"streams_a_pass\": {d:.2}, \"pass_ms\": {d:.3}, \"pass_ms_a_round\": {d:.3}, \"other_ms\": {d:.3}, \"prefills\": {d}, \"prefill_ms\": {d:.1}}}}}\n", .{
        p.rounds,                               ms(p.rows * 1000000, p.rounds), period_ms,              ms(p.send, p.rounds),                      ms(p.enqueue, p.rounds),
        ms(p.forward, p.rounds),                ms(p.absorb, p.rounds),         ms(p.logits, p.rounds), ms(p.sample, p.rounds),                    p.passes,
        ms(p.pass_streams * 1000000, p.passes), ms(p.pass, p.passes),           pass_a_round,           if (periods == 0) 0 else period_ms - held, p.prefills,
        ms(p.prefill, p.prefills),
    });
}

/// A round-cost table "30.0,35.2,..." (TF_DS_ROUND_MS's form): ms by rows from 1.
pub fn parseMs(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    var out: std.ArrayList(f64) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseFloat(f64, std.mem.trim(u8, x, " ")));
    if (out.items.len == 0) return error.BadRoundMs;
    return out.items;
}

/// A calibration weight ("0.5"): finite, from 0 (off) to 1 (TF_DS_T25_DRAFT_CAL).
pub fn parseCalWeight(text: []const u8) !f64 {
    const v = try std.fmt.parseFloat(f64, std.mem.trim(u8, text, " "));
    if (!std.math.isFinite(v) or v < 0.0 or v > 1.0) return error.BadDraftCal;
    return v;
}

pub const parseCalCeil = lanes.config.parseCalCeil;
pub const parseFullAt = lanes.config.parseFullAt;

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
    // TF_DS_ROUND_MS_<n>: the costs of a round of n streams by its rows (n from 2; setStreamCosts), and each as facts
    // hands it over (empty: none, the round takes TF_DS_ROUND_MS)
    scn: [max_streams - 1][max_rows]lanes.config.Cost = undefined,
    scn_rows: [max_streams - 1]usize = @splat(0),
    scn_view: [max_streams - 1][]const lanes.config.Cost = @splat(&.{}),
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    built: Built = .{},
    prof: ?Profile = null,
    fix_k: ?usize = null, // a measurement (tf-dsv41-lanes --fix-k): K drafts a stream, chances 1 to K and 0 past it
    solo_overhead: bool = false, // TF_DS_SOLO_OVERHEAD=1: a lone drafted stream's head windows cost the fixed draft allowance
    draft_cal: f64 = 0.0, // TF_DS_T25_DRAFT_CAL: the weight of a round in each stream's draft-survival calibration (0: off)
    cal_log: u32 = 0, // TF_DS_T25_CAL_LOG: every n-th calibrated verify of a stream prints to stderr (0: off)
    cal_ceil: f64 = 1.0, // TF_DS_T25_CAL_CEIL: upper clamp of calScale (unset: 1.0)
    full_at: ?f64 = null, // TF_DS_T25_FULL_AT: full draft chances when kept share >= x (unset: off)
    no_copies: bool = false, // TF_DS_NO_COPIES=1: a drafted stream proposes no suffix-lookup copy windows
    prof_every: u64 = 0, // TF_DS_SERVE_PROFILE=N: the served rank 0 prints `prof` every N rounds and starts it again (0: off)
    prof_last: u64 = 0, // the last round's start (Model.now)
    prof_period: u64 = 0, // the rounds' start-to-start times under a second, summed (ns), and their count
    prof_periods: u64 = 0,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, oldest first
    next_snap: u64 = 1,
    fill: ?Fill = null, // the prompt pass that runs in pieces between rounds, from its next chunk
    paused: ?Fill = null, // the pass a short prompt cut into (lanes' cut_in), resumed by its next piece
    spill: ?*SpillTier = null, // --spill-dir/--spill-gib: the spill tier (native.zig opens it; null: off, nothing changes)
    kv: ?*kv_lanes.Kv = null, // TF_DS_KV_SHARD: the parts' planner (startKv, at open; null: the switch off)
    kv_stats: lanes.kv_parts.Stats = .{}, // the last parts' stats kvPartsFn built (the engine loop reads them every
    kv_stats_at: u64 = 0, //   iteration: rebuilt at most each kv_stats_ns, Model.now; 0: never built)
    /// The pair's one rank_wire Leader (native.openPair: agreed with rank_identity's full fingerprint before anything
    /// spill- or shard-related crosses), shared by the spill tier and the KV parts; owned here (null: neither is on).
    rw: ?*rank_wire.Leader = null,

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        return ln;
    }

    /// A round of `streams` streams (2 and up) costs `ms[rows - 1]` at `rows` rows (TF_DS_ROUND_MS_<streams>): a row of
    /// another stream costs more than a draft row of the same stream, which one table by rows cannot hold. Before facts.
    pub fn setStreamCosts(self: *Lanes, streams: usize, ms: []const f64) !void {
        if (streams < 2 or streams > max_streams or ms.len == 0) return error.BadRoundMs;
        const t = &self.scn[streams - 2];
        for (t, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = ms[@min(i, ms.len - 1)] };
        self.scn_rows[streams - 2] = max_rows;
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
        if (self.rw) |l| self.gpa.destroy(l); // (after the tier and the planner that borrowed it)
        self.rw = null;
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
    pub fn facts(self: *Lanes) lanes.Model {
        const drafting = self.m.drafting();
        const rr = self.m.eng.round_rows;
        var given: usize = 0;
        for (&self.scn_view, &self.scn, self.scn_rows, 0..) |*v, *t, n, i| {
            v.* = if (n == 0) &.{} else t[0..@min(n, rr)];
            if (n != 0) given = i + 1;
        }
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
            .draft_cal = self.draft_cal,
            .cal_log = self.cal_log,
            .cal_ceil = self.cal_ceil,
            .full_at = self.full_at,
            .fix_k = if (self.fix_k) |k| @intCast(k) else null,
            .copy_proposals = !(drafting and self.no_copies), // opt-in: the draft head proposes; the Python engine makes no copies either
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .stream_costs = self.scn_view[0..given],
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
        k.* = try kv_lanes.Kv.init(self.gpa, kvConfig(m), settings, modelDev(m), .{ .ctx = self, .stale = kvStale, .moved = kvMoved, .stage = kvStage, .bump = kvBump });
        errdefer k.deinit();
        // the pair's one Leader (native.openPair agreed it with the full fingerprint: the spill tier, when on, posts
        // through the same one), so the followers heard one hello and one Tracker sees every frame
        if (self.peer != null) try k.lead(.{ .ctx = self, .rw = kvSendRw }, m.followers(), self.rw orelse return error.PairNotAgreed, 0, undefined);
        // with the spill tier each follower opens its store (every block file scanned, every manifest read) on its own
        // thread from the agreement, and native.openSpill waited for it (awaitFollowers) before this; start's asks
        // keep the hello's bound as a margin, not shard_ns' 10 s
        const l = self.rw;
        const shard_ns = if (l) |x| x.timeouts.shard_ns else 0;
        if (self.spill != null) if (l) |x| {
            x.timeouts.shard_ns = @max(shard_ns, x.timeouts.hello_ns);
        };
        defer if (l) |x| {
            x.timeouts.shard_ns = shard_ns;
        };
        k.start() catch |err| {
            if (self.spill != null) std.log.err("dsv41: the KV parts' start with the spill tier failed ({s}); a follower that logged \"dsv41 spill: rank N could not open its store\" names the cause", .{@errorName(err)});
            return err;
        };
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

    /// A kept state the planner gave up (a stale op): its Snap too; with the spill tier, its unpublished write stages
    /// or cancels here and on every follower (write_staged) before any row changes, as staleOver guards.
    fn kvStale(ctx: *anyopaque, kept: u64) void {
        const self = of(ctx);
        for (self.snaps.items) |sn| if (sn.id == kept) {
            if (!sn.stale) {
                noteStale(sn, "a placement needed its room (its part's oldest first)");
                if (self.spill) |t| t.guardSnap(sn.id, 0);
            }
            sn.stale = true;
        };
    }

    /// A stage op (TF_DS_KV_SHARD with the spill tier): kept state `kept`'s pending write stages its remaining rows
    /// now (or is cancelled) before the stale or move after it in the plan; followers stage their own at the same op.
    fn kvStage(ctx: *anyopaque, kept: u64) void {
        if (of(ctx).spill) |t| t.guardSnap(kept, 0);
    }

    /// A transfer's destination rows are about to change (copy, carry, move): the spill pins over them go stale.
    fn kvBump(ctx: *anyopaque, lo: usize, hi: usize) void {
        if (of(ctx).spill) |t| t.bump(lo, hi);
    }

    /// The planner's pending flags from the spill tier, before a plan (parts.Kept.pending: rank 0's own writes that
    /// still read a kept state's rows; a plan that stales or moves one stages it first).
    fn kvPending(self: *Lanes) void {
        const t = self.spill orelse return;
        const k = self.kv orelse return;
        // (one store lock a batch, not one a kept state: this runs before every plan, beside the store's writer)
        const batch = 64;
        var ids: [batch]u64 = undefined;
        var out: [batch]bool = undefined;
        const kept = k.planner.kept.items;
        var i: usize = 0;
        while (i < kept.len) : (i += batch) {
            const n = @min(batch, kept.len - i);
            for (kept[i..][0..n], ids[0..n]) |x, *id| id.* = x.id;
            t.writesPending(ids[0..n], out[0..n]);
            for (kept[i..][0..n], out[0..n]) |*x, o| x.pending = o;
        }
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
        // (lanes owns the clock, held_wait_ns from the first wait: the planner waits on it for a live stream that holds
        // the state, as placeKept does, and for the streams over a hit's tail, planRoom)
        const waits = if (src) |sn| self.patient(sn) else false;
        self.kvPending();
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

    /// The parts' stats for /health and /metrics (native.Loaded.kv_parts: the host reads them on its engine thread,
    /// in noteLive, every engine-loop iteration). Rebuilt at most each kv_stats_ns: a rebuild scans the planner and
    /// sorts every part's spans (allocating), which base rounds never pay and a round should not wait for.
    pub fn kvPartsFn(ptr: *anyopaque, out: *lanes.kv_parts.Stats) void {
        const self = of(ptr);
        const k = self.kv orelse return;
        const now = self.m.now();
        if (self.kv_stats_at != 0 and now -| self.kv_stats_at < kv_stats_ns) {
            out.* = self.kv_stats;
            return;
        }
        self.kv_stats_at = now;
        k.stats(out);
        k.wantStats(); // (the followers' rows are asked for again at the next round, at most each stats_ms)
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
            shardRank(self.m, &out.rank[0]); // (rank 0's own row; the followers' ride their stats replies)
        }
        self.kv_stats = out.*;
    }

    /// How stale /health's parts' stats may be (kvPartsFn).
    const kv_stats_ns: u64 = 250 * std.time.ns_per_ms;

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

    fn prefillPart(self: *Lanes, s: *lanes.Stream, chunks: usize) !bool {
        const m = self.m;
        // TF_DS_KV_SHARD: a pass in pieces keeps its extent and part between them (the planner moves decoding streams
        // only); its twins are pinned only while a chunk runs, so a placement between two pieces may give them back
        // (the pass then fetches: Caches.twinned), and a pass a prompt cut into binds them again as it resumes
        const t0 = m.now();
        defer if (self.prof) |*p| {
            p.prefill += m.now() - t0;
        };
        if (self.fill) |f| if (f.s != s) {
            // a prompt cuts in between this pass's pieces: it waits (one at most; neither pass needs picture rows: a cut-in's
            // pictures wholly before the state it resumes run no tower and set no spans), its next piece resumes it
            if (self.paused != null or f.s.images.len > 0 or lanes.Engine.picturesPast(s.images, s.reuse.at)) return error.FillBusy;
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
        // TF_DS_KV_SHARD: the cut-in's fill bound its own twins' view; the paused pass's comes back as the planner holds
        // it now (given back meanwhile: plain, as Model.fillBegin left it), on the follower at the same frame
        if (f.part) |id| try self.kv.?.fillBegin(id, l.base);
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
        // a cut-in (Stream.cut_in) runs whole between another pass's pieces: where it really resumes is known only
        // now (the state may have gone stale, or found no room to move to), and if it would compute more than its
        // bound, or rows for a picture, it is refused before its extent, the kept state's end, other states' rows, its
        // sequence, rank 1's fill or a chunk change (the placement's own moves and evictions stand, as for any pass
        // that fails past it; a part goes back by the errdefer above)
        if (s.cut_in > 0) {
            const resume_at: usize = if (kept) |sn| (if (moved and sn.stale) 0 else sn.at) else 0;
            if (len - resume_at > s.cut_in or lanes.Engine.picturesPast(s.images, resume_at)) {
                std.log.info("dsv41: a cut-in of {d} tokens resumes at {d}, not at {d} as admitted; it waits for the pass in pieces", .{ len, resume_at, s.reuse.at });
                return error.FillBusy;
            }
        }
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
            // a resumed pass starts at the kept state's end: a picture wholly before it is already in the state (its
            // positions stay dead in the sequence for Engram's n-grams, but it needs no rows, no run of the tower and no
            // fill_images frame; pictureNeedsRows)
            const resume_at: usize = if (kept) |sn| sn.at else 0;
            for (s.images) |img| {
                try markPicture(seq.items, img.at, img.tokens, image_id);
                if (!pictureNeedsRows(img.at, img.tokens, resume_at)) continue;
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
            // a state read back from disk resumes with this request's own sequence, built above from its ids and its
            // pictures' spans (the follower gets it at fill_begin). It is the point's by construction: the tier hit only
            // because the point's media name exactly these spans (lane_host.tierMedia binds Request.cache_media to
            // Request.images) and its keyed ids derive from them (format.mediaIds), so no check here could fail;
            // checkPictureSeq is the tests' oracle for that sequence.
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
        if (self.prof_every != 0) {
            // a round's start to the next one's: what the backend's own times leave is the round loop's and the server's
            if (self.prof_last != 0 and t0 - self.prof_last < std.time.ns_per_s) {
                self.prof_period += t0 - self.prof_last;
                self.prof_periods += 1;
            }
            self.prof_last = t0;
        }
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
        if (m.ringFailed()) return error.RingFailed; // (robustness: a graph's collect gave up on a dead peer's flags)
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
            if (self.prof_every != 0 and p.rounds >= self.prof_every) {
                printServeProfile(p.*, self.prof_period, self.prof_periods);
                p.* = .{};
                self.prof_period = 0;
                self.prof_periods = 0;
            }
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
        survivals(l.confs[0..out.len], out);
        if (self.fix_k) |k| {
            if (self.cal_log > 0 or self.draft_cal > 0.0) s.rememberHeld(out);
            for (out, 0..) |*o, j| o.* = if (j < k) 1.0 else 0.0;
            return true;
        }
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
        // (the spill tier's keep follows: spillInfo gives it the planner's part salt, saltOf, for its block lineage;
        // its write's pending flag reaches the planner before each plan, kvPending)
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
    fn spillReserve(ptr: *anyopaque, at: usize, room: usize, part: ?u8) anyerror!spill_tier.Dest {
        const self = of(ptr);
        if (self.kv) |k| return self.spillReserveParts(k, at, room, part orelse return error.SpillNoPart);
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

    /// TF_DS_KV_SHARD: the destination is a planner hold in the point's recorded part (its owner holds the codes
    /// there: Kv.restore over planRestore, on every rank alike), the waiting prompt's room with a growth margin when
    /// that stales nobody, else the point's rows; the hold becomes a kept state at install (restoredOp) or goes at
    /// discard (unholdOp).
    fn spillReserveParts(self: *Lanes, k: *kv_lanes.Kv, at: usize, room: usize, part: u8) !spill_tier.Dest {
        self.kvPending();
        const want = std.mem.alignForward(usize, @max(room, at) + growth_rows, model.chunk_rows);
        const id = self.next_snap;
        const r = try k.restore(id, want, at, part);
        errdefer k.unhold(id) catch |err| std.log.warn("dsv41: a refused restore kept its part's hold: {s}", .{@errorName(err)});
        // (its generation: rank_wire Reserve.generation, each follower checks its own planner's hold against it)
        const gen = for (k.planner.holds.items) |x| {
            if (x.why == .restore and x.id == id) break x.gen;
        } else return error.UnknownHold;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = id, .at = at, .base = r.base, .end = r.base + r.size, .part = part, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .loading = true };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        return .{ .saved = sn, .id = sn.id, .base = sn.base, .buf = sn.buf.ptr, .gen = gen, .size = r.size };
    }

    /// An extent of `size` around the live streams and the kept states' rows (placeSoft's rules: their held extents
    /// first), staling nobody: null when none.
    fn placeFree(self: *Lanes, live: []const [2]usize, size: usize) !?usize {
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(self.gpa);
        for (self.snaps.items) |sn| if (!sn.stale) try kept.append(self.gpa, .{ sn.base, sn.at, sn.end });
        return placeSoft(self.gpa, live, kept.items, size, self.m.pool_cap);
    }

    fn spillInstall(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.loading = false;
        sn.restored = true;
        // TF_DS_KV_SHARD: the hold a kept state of the planner too (every rank's at this op); refused, its hold goes
        // and the Snap is stale (no room of its own in placement, never usable; the prompt cache drops it later, and
        // Kv.drop ignores an id the planner never kept)
        if (self.kv) |k| k.restored(sn.id, sn.id, sn.at) catch |err| {
            std.log.err("dsv41: the parts' planner did not keep a restored prompt state: {s}", .{@errorName(err)});
            k.unhold(sn.id) catch {};
            sn.stale = true;
        };
    }

    /// A restore that did not commit: its Snap goes (the followers drop theirs with the abort, no snap_drop).
    fn spillDiscard(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (self.kv) |k| k.unhold(sn.id) catch |err| std.log.warn("dsv41: a discarded restore kept its part's hold: {s}", .{@errorName(err)});
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, sn)) |i| _ = self.snaps.orderedRemove(i);
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    fn spillInfo(ptr: *anyopaque, saved: *anyopaque) spill_tier.SnapInfo {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        // TF_DS_KV_SHARD: the part its rows lie in now, as the planner holds it (its block lineage's salt)
        const part: ?u8 = if (self.kv) |k| k.planner.saltOf(sn.id) else null;
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf.ptr, .slot = sn.slot, .part = part };
    }

    /// A spill frame (Kind rank_wire) to every follower, in the ordered stream.
    pub fn spillFrame(ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void {
        const self = of(ctx);
        const k: Kind = switch (kind) {
            10 => .rank_wire,
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
/// A rank's attention-side counters (Model.kvs' stats) as its kv_shard row has them.
fn shardRank(m: *Model, out: *lanes.kv_parts.Rank) void {
    const x = m.kvs orelse return;
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

fn readShardRank(ctx: *anyopaque, out: *lanes.kv_parts.Rank) void {
    shardRank(@ptrCast(@alignCast(ctx)), out);
}

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

/// A round or a prompt chunk this rank could not finish: refused as above, unless the error means the ranks cannot go on together
/// (round.pair_ending: the ring's order or rows broken, a sized send or the peer's collect pending, the ring or the
/// communicator dead). Then the follower ends with it (rank 0 ends the pair for the same error, native.pair_fatal, or
/// sees this rank's link go down) instead of taking the next frame.
fn roundRefused(err: anyerror) !void {
    refused(err);
    if (round.endsPair(err)) return err;
}

test "a follower ends on an error that parts the ranks, and refuses the rest" {
    try roundRefused(error.NoFreeSlot);
    try roundRefused(error.ReadFailed); // (a raw read error is made EngramRankFault where it is raised)
    try std.testing.expectError(error.EngramRankFault, roundRefused(error.EngramRankFault));
    try std.testing.expectError(error.CommAborted, roundRefused(error.CommAborted));
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

/// A follower's one rank_wire Follower (spill-shard round 3): rank 0's Leader (Lanes.rw) sends the spill tier's and
/// the KV parts' commands in one ordered stream, so one Tracker admits every frame here too; the spill methods go to
/// the spill side, shard* to the planner's Follower, each failing closed (error.Unsupported) when that side is off on
/// this rank (the fingerprint's features should have refused the start already). Heap only (rank_wire.Follower is
/// about 40 KiB).
pub const PairFollower = PairFollowerOf(SpillFollower);

/// PairFollower over a spill follower type `SF` (spill_tier.Follower of the Model's device; the tests' byte array).
pub fn PairFollowerOf(comptime SF: type) type {
    return struct {
        const Self = @This();
        wf: rank_wire.Follower,
        spill: ?*SF,
        kv: ?*kv_lanes.Follower,

        /// `fx`: this rank's full fingerprint (native.rankExchange; the spill side's own when it has one).
        pub fn create(gpa: std.mem.Allocator, rank: u16, fx: fingerprint.Exchange, spill: ?*SF, kv: ?*kv_lanes.Follower) !*Self {
            const p = try gpa.create(Self);
            p.* = .{ .wf = rank_wire.Follower.init(rank, fx), .spill = spill, .kv = kv };
            return p;
        }

        /// A lanes Kind.rank_wire frame: decoded, admitted in order, dispatched; an ask answered on `l`.
        pub fn onFrame(p: *Self, l: link.Link, payload: []const u8) !void {
            try rank_wire.serve(&p.wf, l, payload, p);
            if (p.kv) |k| if (k.diverged) return error.ShardDiverged;
        }

        fn sp(p: *Self) !*SF {
            return p.spill orelse error.Unsupported;
        }

        fn kf(p: *Self) !*kv_lanes.Follower {
            return p.kv orelse error.Unsupported;
        }

        const H = rank_wire.Head;

        pub fn agreed(p: *Self, a: fingerprint.Agreement) !void {
            // (its store opens under its own compat on its own thread: every block file scanned, every manifest read;
            // rank 0's awaitFollowers waits for it)
            if (p.spill) |s| s.agreed(a) catch |err| {
                std.log.err("dsv41 spill: rank {d} could not open its store at the ranks' agreement: {s}", .{ p.wf.rank, @errorName(err) });
                return err;
            };
        }
        pub fn has(p: *Self, h: H, m: rank_wire.Has) !rank_wire.HasReply {
            return (try p.sp()).has(h, m);
        }
        /// owner_sharded (a nonzero generation): the destination must be this rank's planner hold for the kept id, at
        /// that base, size and generation (planRestore's, applied from rank 0's plan just before): a restore never lands
        /// outside the part and the rows the planner reserved for it (a mismatch is a diverged pair: this rank ends).
        pub fn reserve(p: *Self, h: H, m: rank_wire.Reserve) !void {
            const s = try p.sp();
            if (m.generation != 0) {
                const k = try p.kf();
                const pl = &(k.planner orelse return error.NotAgreed);
                const hold = for (pl.holds.items) |x| {
                    if (x.why == .restore and x.id == m.kept) break x;
                } else return error.UnknownHold;
                if (hold.gen != m.generation or hold.base != m.base or hold.part != m.part or hold.size != m.size or hold.size < m.at) return error.StaleExtent;
            }
            return s.reserve(h, m);
        }
        pub fn status(p: *Self, h: H, m: rank_wire.RestoreStatus) !rank_wire.RestoreStatusReply {
            return (try p.sp()).status(h, m);
        }
        pub fn keep(p: *Self, h: H, m: rank_wire.Keep) !void {
            return (try p.sp()).keep(h, m);
        }
        pub fn share(p: *Self, h: H, m: rank_wire.Share) !void {
            return (try p.sp()).share(h, m);
        }
        pub fn touch(p: *Self, h: H, m: rank_wire.Touch) !void {
            return (try p.sp()).touch(h, m);
        }
        pub fn forget(p: *Self, h: H, m: rank_wire.Forget) !void {
            return (try p.sp()).forget(h, m);
        }
        pub fn start(p: *Self, h: H, m: rank_wire.Start) !void {
            return (try p.sp()).start(h, m);
        }
        pub fn progress(p: *Self, h: H, m: rank_wire.Progress) !void {
            return (try p.sp()).progress(h, m);
        }
        pub fn validate(p: *Self, h: H, m: rank_wire.Validate) !rank_wire.ValidateReply {
            return (try p.sp()).validate(h, m);
        }
        pub fn commit(p: *Self, h: H, m: rank_wire.Commit) !void {
            return (try p.sp()).commit(h, m);
        }
        pub fn abort(p: *Self, h: H, m: rank_wire.Abort) !void {
            return (try p.sp()).abort(h, m);
        }
        pub fn writeStaged(p: *Self, h: H, m: rank_wire.WriteStaged) !void {
            return (try p.sp()).writeStaged(h, m);
        }
        pub fn writeCancelled(p: *Self, h: H, m: rank_wire.WriteCancelled) !void {
            return (try p.sp()).writeCancelled(h, m);
        }
        /// Each set to its side: `tier` the spill side's counters, `shard` the planner side's (a side this rank does not
        /// run answers its set empty, as each side alone does for the other's).
        pub fn stats(p: *Self, h: H, m: rank_wire.Stats) !rank_wire.StatsReply {
            return switch (m.set) {
                .tier => if (p.spill) |x| x.stats(h, m) else .{ .set = .tier, .n = 0, .values = @splat(0) },
                .shard => if (p.kv) |x| x.stats(h, m) else .{ .set = .shard, .n = 0, .values = @splat(0) },
                .open => if (p.spill) |x| x.stats(h, m) else .{ .set = .open, .n = 0, .values = @splat(0) },
            };
        }
        pub fn shardLayout(p: *Self, h: H, m: rank_wire.ShardLayout) !rank_wire.ShardLayoutReply {
            return (try p.kf()).shardLayout(h, m);
        }
        pub fn shardSet(p: *Self, h: H, m: rank_wire.ShardSet) !void {
            return (try p.kf()).shardSet(h, m);
        }
        pub fn shardOps(p: *Self, h: H, m: rank_wire.ShardOps) !void {
            return (try p.kf()).shardOps(h, m);
        }
        pub fn shardCheck(p: *Self, h: H, m: rank_wire.ShardCheck) !rank_wire.ShardCheckReply {
            return (try p.kf()).shardCheck(h, m);
        }
        pub fn shardXfer(p: *Self, h: H, m: rank_wire.ShardXfer) !rank_wire.ShardXferReply {
            return (try p.kf()).shardXfer(h, m);
        }
        pub fn shardFinish(p: *Self, h: H, m: rank_wire.ShardFinish) !void {
            return (try p.kf()).shardFinish(h, m);
        }
    };
}

/// The follower's spill side as its planner's ops reach it (kv_lanes.SpillHooks over a spill follower `sd`).
pub fn followerSpillHooks(sd: anytype) kv_lanes.SpillHooks {
    const SF = @typeInfo(@TypeOf(sd)).pointer.child;
    return .{ .ctx = sd, .guard = struct {
        fn f(ctx: *anyopaque, kept: u64, from: u32) void {
            const x: *SF = @ptrCast(@alignCast(ctx));
            x.guard(kept, from);
        }
    }.f, .bump = struct {
        fn f(ctx: *anyopaque, lo: usize, hi: usize) void {
            const x: *SF = @ptrCast(@alignCast(ctx));
            x.bump(lo, hi);
        }
    }.f };
}

/// follow, with this rank's spill side when the tier is on (`spill`: made before the model's pool, its store opened at
/// the startup agreement rank 0 starts; the caller destroys it after this returns).
pub fn followWith(gpa: std.mem.Allocator, m: *Model, spill: ?*SpillFollower) !void {
    return followPair(gpa, m, spill, null);
}

/// followWith, with `fx` for the pair's one rank_wire Follower's hello instead of its default: the spill side's full
/// fingerprint when the tier is on, else (TF_DS_KV_SHARD alone) round 2's shard-only hello (kv_lanes.Follower.exchange).
pub fn followPair(gpa: std.mem.Allocator, m: *Model, spill: ?*SpillFollower, fx: ?fingerprint.Exchange) !void {
    const peer = m.peer();
    const buf = try gpa.alloc(u8, (m.pool_cap + 1) * 4 + (1 << 16));
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    // on every exit, before `seqs` goes: the Engram readers (the read-ahead, TF_DS_ENGRAM_LEAD's) may still be reading
    // a prompt's ids, and an exit from a bad frame comes before the join the loop does for the next frame
    defer {
        m.joinAhead();
        m.fill_seq = &.{};
    }
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
    // TF_DS_KV_SHARD: this rank's copy of the parts' planner, fed rank 0's ops; with the spill tier its stage, stale
    // and move ops guard this rank's own writes and its transfers bump their destinations (kv_lanes.SpillHooks)
    const kvf: ?*kv_lanes.Follower = if (m.kvs != null) try kv_lanes.Follower.init(gpa, m.rank, kvConfig(m), try kv_lanes.Settings.fromEnv(getenv), modelDev(m)) else null;
    defer if (kvf) |k| k.deinit();
    if (kvf) |k| k.counters = .{ .ctx = m, .read = readShardRank }; // (its own row on rank 0's /health)
    if (kvf) |k| if (spill) |sd| {
        k.spill = followerSpillHooks(sd);
    };
    // one rank_wire Follower for both (rank 0's one Leader): the caller's fingerprint, else the spill side's, else
    // the planner's shard-only one (native.openPair's hello for TF_DS_KV_SHARD alone)
    const pair_fx: ?fingerprint.Exchange = fx orelse if (spill) |sd| sd.wf.fx else if (kvf) |k| k.exchange() else null;
    const pf: ?*PairFollower = if (pair_fx) |x| try PairFollower.create(gpa, @intCast(m.rank), x, spill, kvf) else null;
    defer if (pf) |p| gpa.destroy(p);
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
            12 => .fill_resume,
            else => return error.BadFrame,
        };
        if (try r.int(u64) != step) return error.OutOfStep;
        step += 1;
        if (spill) |sd| sd.service(std.time.ns_per_ms); // (rank 0 pumps between its primitives alike)
        // a chunk's read-ahead of the next is done before anything but that chunk (a prompt's ids may change)
        if (kind != .fill_chunk and kind != .snap_save and kind != .snap_restore and kind != .fill_images and kind != .rank_wire) {
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
                if (m.fillBegin(slot, @intCast(base))) {
                    if (kvf) |k| try k.fillBegin(slot, @intCast(base)); // (its twins' view again, as rank 0 binds it at its resume)
                } else |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                if (m.fill_seq.len == 0) m.fill_seq = seqs[slot].items; // a piece after a round: the read-ahead again
                if (spill) |sd| sd.bump(fill_base + start, fill_base + start + n);
                // an error that parts the ranks (an Engram read this rank could not make) ends the follower; rank 0 ends the pair
                if (m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay)) |_| {} else |err| try roundRefused(err);
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
                m.verify(rows, built.absorb[0..nw]) catch |err| try roundRefused(err);
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
                m.pass(tokens[0..n], q0[0..n], slots[0..n], steps) catch |err| try roundRefused(err);
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
            // one rank_wire Follower a rank, for the spill tier's and the KV parts' frames alike
            .rank_wire => try (pf orelse return error.SpillNotStarted).onFrame(peer, r.rest()), // (rank 0 runs the spill tier)
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

test "a calibration weight parses to a finite value in 0..1" {
    try std.testing.expectEqual(@as(f64, 0.5), try parseCalWeight(" 0.5 "));
    try std.testing.expectEqual(@as(f64, 1.0), try parseCalWeight("1"));
    try std.testing.expectEqual(@as(f64, 0.0), try parseCalWeight("0"));
    for ([_][]const u8{ "-0.1", "1.01", "2", "abc", "", "nan", "inf", "-inf" }) |bad| {
        try std.testing.expect(std.meta.isError(parseCalWeight(bad)));
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

test "a prompt that needs picture rows does not cut in between another pass's pieces, nor any into a pass with pictures" {
    var m: Model = undefined;
    m.io = std.testing.io;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
    var long: lanes.Stream = undefined;
    long.images = &.{};
    var short: lanes.Stream = undefined;
    const image = [_]lanes.stream.Image{.{ .at = 4, .tokens = 6, .bytes = "" }};
    short.images = &image;
    // resuming at 8, the picture [4, 10) straddles the cut: its rows are needed
    short.reuse = .{ .at = 8 };
    ln.fill = .{ .s = &long, .slot = 0, .len = 16, .replay = 0, .last_end = 0, .start = 8 };
    try std.testing.expectError(error.FillBusy, ln.prefillPart(&short, 1));
    try std.testing.expect(ln.paused == null and ln.fill.?.s == &long); // the pass in pieces stays as it was
    // the filling pass has pictures: nothing cuts in, a text prompt neither
    long.images = &image;
    short.images = &.{};
    short.reuse = .{};
    try std.testing.expectError(error.FillBusy, ln.prefillPart(&short, 1));
    try std.testing.expect(ln.paused == null and ln.fill.?.s == &long);
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
    try std.testing.expect(std.enums.fromInt(Kind, 11) == null); // (round 2's spill notes are rank_wire ops now)
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
    // the spill tier's owner_sharded layout of the checkpoint stores the same bytes a token for its part's points
    // (native.openPair refuses a start where the two differ)
    const config = @import("config.zig");
    var why: config.Why = .{};
    const cfg = try config.parse(std.testing.allocator, config.test_config, &why);
    const lay = try spill_tier_state.StateLayout.fromConfig(&cfg, spill_tier_state.Geometry.of(&cfg, model.chunk_rows, 16, 64), .owner_sharded);
    try std.testing.expectEqual(@as(usize, 720), lay.owner_token_bytes);
}

/// A byte-array device for a follower's spill side in the composite's tests (no copy is made there: only the store
/// opens at the agreement and the handlers answer).
const MiniDev = struct {
    pub const Event = u32;
    pub fn blockPlane(_: MiniDev, _: spill_tier_state.Plane) u64 {
        return 4096;
    }
    pub fn slotPlane(_: MiniDev, _: spill_tier_state.Plane, _: usize) u64 {
        return 4096;
    }
    pub fn capacity(_: MiniDev) usize {
        return 64 * model.chunk_rows;
    }
    pub fn slots(_: MiniDev) usize {
        return 4;
    }
    pub fn d2h(_: MiniDev, _: []u8, _: u64) !void {}
    pub fn h2d(_: MiniDev, _: u64, _: []const u8) !void {}
    pub fn fill(_: MiniDev, _: u64, _: u8, _: usize) !void {}
    pub fn record(_: MiniDev, _: u32) !void {}
    pub fn done(_: MiniDev, _: u32) bool {
        return true;
    }
    pub fn wait(_: MiniDev, _: u32) void {}
};
const spill_tier_state = @import("spill_state.zig");
const MiniSpill = spill_tier.Follower(MiniDev);

fn miniFx(rank: u8) fingerprint.Exchange {
    var x: fingerprint.Exchange = .{ .common = @splat(3), .rank = @splat(20 + rank) };
    @memset(&x.fields, .{ .id = 0, .scope = .common, .digest = @splat(0) });
    return x;
}

/// The follower side of the composite test: its spill side (an owner-sharded store on TF_SPILL_TEST_DIR, opened at the
/// agreement), its planner's Follower and the one rank_wire Follower over both, on its own thread.
const CompositePeer = struct {
    port: u16,
    cfg: parts.Config,
    root: [:0]const u8,
    err: ?anyerror = null,
    spill_opened: bool = false,
    has_answered: bool = false,
    digest: [32]u8 = undefined,

    fn run(p: *CompositePeer) void {
        p.loop() catch |e| {
            p.err = e;
        };
    }

    fn loop(p: *CompositePeer) !void {
        const gpa = std.testing.allocator;
        var l = try link.Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p.port, 10);
        defer l.close();
        const shapes = [_]spill_tier_state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
        const raw = try spill_tier_state.StateLayout.init(&shapes, .{ .block_tokens = model.chunk_rows, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .owner_sharded);
        const layout = try raw.withParts(p.cfg.cap / 2);
        const settings: spill_tier.Settings = .{ .dir = p.root, .gib = 1, .min_free_gib = 0, .min_tokens = model.chunk_rows, .staging_bytes = 64 << 20, .readers = 1 };
        const plan = try spill_tier.stagingPlan(settings, &layout);
        const staging = try gpa.alignedAlloc(u8, .fromByteUnits(4096), plan.bytes());
        defer gpa.free(staging);
        const events = try gpa.alloc(u32, spill_tier.eventCount(plan));
        defer gpa.free(events);
        for (events, 0..) |*e, i| e.* = @intCast(i);
        const sf = try MiniSpill.create(gpa, std.testing.io, 1, miniFx(1), null, .{ .dev = .{}, .layout = layout, .gens = try spill_tier_state.Generations.init(gpa, p.cfg.cap, model.chunk_rows), .settings = settings, .max_tokens = @intCast(p.cfg.cap / 2), .staging = staging, .events = events });
        defer sf.destroy();
        var dev: kv_lanes.FakeDev = .{ .gpa = gpa };
        defer dev.deinit();
        const kf = try kv_lanes.Follower.init(gpa, 1, p.cfg, .{ .twins = false }, dev.dev());
        defer kf.deinit();
        kf.spill = followerSpillHooks(sf);
        const pf = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), sf, kf);
        defer gpa.destroy(pf);
        const buf = try gpa.alloc(u8, rank_wire.max_frame);
        defer gpa.free(buf);
        while (true) {
            const bytes = try l.recv(buf);
            if (bytes.len == 0) break;
            try pf.onFrame(l, bytes);
            p.spill_opened = sf.side != null;
        }
        p.digest = kf.planner.?.digest();
    }
};

test "spill + shard: one rank_wire Follower serves both sides over loopback (agreement opens the spill store, shard ops reach the planner, spill asks the store)" {
    const gpa = std.testing.allocator;
    const c = model.chunk_rows;
    const core = @import("core");
    var path: [256]u8 = undefined;
    const root = try core.spill.fsio.testRoot(&path, "lanes-composite");
    const fd = try core.spill.fsio.freshDir(gpa, root);
    _ = std.c.close(fd);
    defer core.spill.fsio.dropDir(gpa, root);
    var b: [2]u8 = undefined;
    std.testing.io.random(&b);
    const port = 30000 + std.mem.readInt(u16, &b, .little) % 20000;
    const cfg: parts.Config = .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 };
    var peer: CompositePeer = .{ .port = port, .cfg = cfg, .root = root };
    const th = try std.Thread.spawn(.{}, CompositePeer.run, .{&peer});
    var joined = false;
    var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
    defer {
        if (!joined) {
            l.send(&.{}) catch {};
            th.join();
        }
        l.close();
    }
    const Out = struct {
        l: link.Link,
        fn rw(ctx: *anyopaque, sub: []const u8) anyerror!void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            try o.l.send(sub);
        }
        fn sink(o: *@This()) rank_wire.Sink {
            return .{ .ctx = o, .send = rw };
        }
    };
    var out: Out = .{ .l = l };
    const links = [_]?link.Link{l};
    // rank 0: the pair's one Leader, agreed once, then the planner over it (Lanes.startKv's order)
    const lead = try gpa.create(rank_wire.Leader);
    defer gpa.destroy(lead);
    lead.* = rank_wire.Leader.init(2, 91, miniFx(0));
    _ = try lead.agree(out.sink(), &links);
    var dev0: kv_lanes.FakeDev = .{ .gpa = gpa };
    defer dev0.deinit();
    const Rec = struct {
        fn stale(_: *anyopaque, _: u64) void {}
        fn moved(_: *anyopaque, _: u64) void {}
    };
    var k = try kv_lanes.Kv.init(gpa, cfg, .{ .twins = false }, dev0.dev(), .{ .ctx = &dev0, .stale = Rec.stale, .moved = Rec.moved });
    defer k.deinit();
    try k.lead(.{ .ctx = &out, .rw = Out.rw }, &links, lead, 0, undefined);
    try k.start(); // shard_layout (answered by the planner side), shard_set, shard_check over the same Tracker
    // a spill ask between plans: the follower's spill side answers from its own store (opened at the agreement)
    var replies: [1]rank_wire.Frame = undefined;
    _ = try lead.ask(out.sink(), &links, 0, .{ .has = .{ .point = @splat(7), .compat = lead.compat(), .at = 2 * c, .blocks = 2, .layout = .owner_sharded, .part = 1 } }, &replies);
    try std.testing.expectEqual(rank_wire.Absence.missing, replies[0].msg.has_reply.answer);
    _ = try lead.post(out.sink(), replies[0].head.txn, .{ .abort = .{ .why = .presence, .rank = 1, .forget = false } });
    // a plan with a stage and a stale op (rank 0's pending write), interleaved with a spill post: one stream
    const a = try k.admit(1, 1, 0, 10 * c, null, false);
    try k.keep(500, a.id, 8 * c);
    try k.release(a.id);
    _ = try lead.post(out.sink(), 0, .{ .write_staged = .{ .point = @splat(9), .kept = 500, .upto = 0 } });
    const f1 = try k.admit(2, 2, 1, 32 * c, null, false);
    k.planner.setPending(500, true);
    const f2 = try k.admit(3, 3, 2, 30 * c, null, false);
    try std.testing.expect(k.planner.keptAt(500).?.stale);
    _ = try lead.ask(out.sink(), &links, 0, .{ .stats = .{ .set = .tier } }, &replies);
    try std.testing.expectEqual(rank_wire.StatsSet.tier, replies[0].msg.stats_reply.set);
    for ([_]u64{ f1.id, f2.id }) |id| try k.release(id);
    try k.check();
    try l.send(&.{});
    th.join();
    joined = true;
    try std.testing.expectEqual(@as(?anyerror, null), peer.err);
    try std.testing.expect(peer.spill_opened);
    try std.testing.expectEqualSlices(u8, &k.planner.digest(), &peer.digest);
}

test "spill + shard: the composite fails closed for a side this rank does not run" {
    const gpa = std.testing.allocator;
    var dev: kv_lanes.FakeDev = .{ .gpa = gpa };
    defer dev.deinit();
    const kf = try kv_lanes.Follower.init(gpa, 1, .{ .cap = 64 * model.chunk_rows, .parts = 2, .carry_bytes = 720 }, .{}, dev.dev());
    defer kf.deinit();
    const only_kv = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), null, kf);
    defer gpa.destroy(only_kv);
    const h: rank_wire.Head = .{ .op = .has, .from = 0, .epoch = 1, .txn = 1, .seq = 2 };
    try std.testing.expectError(error.Unsupported, only_kv.has(h, .{ .point = @splat(1), .compat = @splat(0), .at = 2048, .blocks = 1, .layout = .owner_sharded, .part = 0 }));
    try std.testing.expectError(error.Unsupported, only_kv.writeStaged(h, .{ .point = @splat(1), .kept = 1, .upto = 0 }));
    try only_kv.agreed(.{ .common = @splat(0), .world = 2 }); // (no spill side: nothing to open)
    const none = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), null, null);
    defer gpa.destroy(none);
    try std.testing.expectError(error.Unsupported, none.shardSet(h, .{ .version = parts.layout_version, .parts = 2, .part_rows = 32 * model.chunk_rows, .owner_epoch = 1 }));
}

test "spill + shard: an owner-sharded reserve lands only on the follower planner's own hold (base, part, generation)" {
    const gpa = std.testing.allocator;
    const c = model.chunk_rows;
    const cfg: parts.Config = .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 };
    var dev: kv_lanes.FakeDev = .{ .gpa = gpa };
    defer dev.deinit();
    const kf = try kv_lanes.Follower.init(gpa, 1, cfg, .{ .twins = false }, dev.dev());
    defer kf.deinit();
    const h: rank_wire.Head = .{ .op = .shard_set, .from = 0, .epoch = 1, .txn = 1, .seq = 2 };
    try kf.shardSet(h, .{ .version = parts.layout_version, .parts = 2, .part_rows = kf.cfg.partRows(), .owner_epoch = 1 });
    // rank 0's restore plan as the follower applies it: a hold of 8 blocks in part 1 for kept id 77
    const gen = kf.planner.?.next_gen;
    const hold: kv_lanes.WireOp = .{ .hold = .{ .id = 77, .part = 1, .base = 40 * c, .size = 8 * c, .gen = gen } };
    try kf.shardOps(h, .{ .owner_epoch = 2, .index = 0, .last = true, .ops = &.{hold} });
    // a spill side not yet agreed (no store): what passes the hold check reaches it and is refused there
    const shapes = [_]spill_tier_state.LayerShape{ .{ .ratio = 1, .kv = true, .idx = true } };
    const raw = try spill_tier_state.StateLayout.init(&shapes, .{ .block_tokens = c, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .owner_sharded);
    const layout = try raw.withParts(32 * c);
    const sf = try MiniSpill.create(gpa, std.testing.io, 1, miniFx(1), null, .{ .dev = .{}, .layout = layout, .gens = try spill_tier_state.Generations.init(gpa, cfg.cap, c), .settings = .{ .dir = "/nonexistent", .gib = 1 }, .max_tokens = @intCast(32 * c), .staging = &.{}, .events = &.{} });
    defer sf.destroy();
    const pf = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), sf, kf);
    defer gpa.destroy(pf);
    const want: rank_wire.Reserve = .{ .point = @splat(5), .kept = 77, .slot = 0, .base = 40 * c, .size = 8 * c, .at = 6 * c, .blocks = 6, .generation = gen, .part = 1 };
    try std.testing.expectError(error.SpillNotAgreed, pf.reserve(h, want)); // (the hold matched)
    var other = want;
    other.generation = gen + 1;
    try std.testing.expectError(error.StaleExtent, pf.reserve(h, other));
    other = want;
    other.base = 48 * c;
    try std.testing.expectError(error.StaleExtent, pf.reserve(h, other));
    other = want;
    other.size = 9 * c; // (an extent rank 0 never held: the hold's size is the Reserve's)
    try std.testing.expectError(error.StaleExtent, pf.reserve(h, other));
    other = want;
    other.kept = 78;
    try std.testing.expectError(error.UnknownHold, pf.reserve(h, other));
    // a replicated reserve (generation 0) asks no planner
    other = want;
    other.generation = 0;
    try std.testing.expectError(error.SpillNotAgreed, pf.reserve(h, other));
}

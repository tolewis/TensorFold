//! Port of wojo's GLM 0118 (PR #152) follow-up: TF_DS_KV_SHARD's decode exchanges sized to what the other rank needs
//! and carried by the family's RDMA ring (rdma.zig) instead of NCCL, host half and plans, host-only (std and shard.zig:
//! `zig test` runs it alone).
//!
//! A decode exchange moves whole rows of a round, each row one run of bytes in every one of up to `max_segs` segments
//! (codes and scales of a fetch; a row's q; a row's rotated wo_a inputs in each group's plane). Which rows a rank sends
//! and which it receives are explicit lists (`List`), computed on the host each round from the replicated row owners
//! (`lists`: the same inputs on both ranks), uploaded with the round's glue (`Lists.encode`, one fixed device block,
//! so captured graphs stay valid) and read by the kernels. The sender's stage kernel packs its send
//! list's rows' segments one after another into its ring slot (segment s of `n` rows at n x the row bytes of the
//! segments before it) and publishes the slot's size; the proxy writes exactly those bytes to the peer (split over the
//! rails as every gather is); the receiver's collect kernel waits for the slot and puts the received rows, in its
//! receive list's order, at their rows. Rank r's send list of an exchange IS rank 1-r's receive list (`lists`'s tests),
//! so the two sides agree on order and size without anything but the bytes crossing the wire.
//!
//! The three exchanges (one a compressed layer): the fetch (the sender's rows' picked codes and scales from its slot
//! of the fetch buffer into the peer's slot rows: `fetchXfer`), owner decode's queries (this rank's q of the rows the
//! peer holds, to the peer's gathered-q rows) and owner decode's outputs (the peer's heads of the rows this rank
//! holds, straight into the peer's xhwo rows, so no select pass). Each puts the bytes the fixed-size NCCL path put at
//! the same places for every row the receiver reads (the tests below).
//!
//! The ring order of a layer's transfers is fixed per mode (`steps`, checked as round.zig issues them: `Cursor`), every
//! stage directly followed in ring order by its collect, and rdma.Ring refuses any other ring transfer in between.
//!
//! TF_DS_KV_SHARD_DECODE=auto chooses each round between fetch and owner decode from the round's row counts (a cost
//! model in ns, `Cost`, over the replicated counts: both ranks pick the same and capture the same graphs; the round's
//! graph key carries the choice). Rounds wider than the split path (`decode_rows`) always fetch: owner decode has no
//! one-pass variant. While a round runs the split path's bound is the round bound (round.forward sets tri.decode_rows
//! to TF_DS_ROUND_ROWS), so every decode round takes the split path and auto may pick owner at any of its widths.
const std = @import("std");
const shard = @import("shard.zig");

/// Segments a row of one exchange at most (dsv41_shard.cu TF_DS_RING_SEGS; owner decode's outputs take one a wo_a
/// group).
pub const max_segs = 8;
/// Rows one exchange lists at most (dsv41_shard.cu TF_DS_RING_ROWS: the kernels' row list; round.max_rows fits).
pub const max_rows = 64;

/// One segment of an exchange: a listed row r's `row_bytes` from src + r * src_stride (the sender's) to
/// dst + r * dst_stride (the receiver's). Every value whole 16-byte words.
pub const Seg = struct { src: u64 = 0, dst: u64 = 0, row_bytes: usize = 0, src_stride: usize = 0, dst_stride: usize = 0 };

/// An exchange as a rank names it: its segments (src: where this rank's sent rows are; dst: where the rows it
/// receives go). Which rows: the round's `Lists`.
pub const Xfer = struct {
    segs: [max_segs]Seg = @splat(.{}),
    n: usize = 0,

    pub fn slice(x: *const Xfer) []const Seg {
        return x.segs[0..x.n];
    }

    /// A listed row's bytes over every segment.
    pub fn rowBytes(x: Xfer) usize {
        var t: usize = 0;
        for (x.segs[0..x.n]) |s| t += s.row_bytes;
        return t;
    }

    /// The most bytes a slot of this exchange holds (`rows` rows listed).
    pub fn capacity(x: Xfer, rows: usize) usize {
        return rows * x.rowBytes();
    }

    /// Refused before any launch: no segment or too many, too many rows, a size or address not in whole 16-byte
    /// words, a slot that could exceed the ring's.
    pub fn check(x: Xfer, rows: usize, max_bytes: usize) !void {
        if (x.n == 0 or x.n > max_segs or rows == 0 or rows > max_rows) return error.BadRingRows;
        for (x.segs[0..x.n]) |s| if ((s.src | s.dst) % 16 != 0 or (s.row_bytes | s.src_stride | s.dst_stride) % 16 != 0 or s.row_bytes == 0) return error.BadRingRows;
        if (x.capacity(rows) > max_bytes) return error.TooLarge;
    }
};

/// Rows a rank sends or receives in one exchange, in slot order: indices into the segments' row strides.
pub const List = struct {
    n: usize = 0,
    rows: [max_rows]u8 = @splat(0),

    pub fn slice(l: *const List) []const u8 {
        return l.rows[0..l.n];
    }

    fn push(l: *List, r: usize) void {
        l.rows[l.n] = @intCast(r);
        l.n += 1;
    }

    pub fn eql(a: List, b: List) bool {
        return std.mem.eql(u8, a.slice(), b.slice());
    }
};

pub const Which = enum(u8) { fetch = 0, q = 1, out = 2 };
pub const Dir = enum(u8) { send = 0, recv = 1 };

/// A round's lists on one rank, every exchange's both ways. On the device (Round.rlist, uploaded each ring round):
/// i32 [3 exchanges][2 directions][1 + max_rows]: the count, then the rows.
pub const Lists = struct {
    l: [3][2]List = @splat(@splat(.{})),

    pub const stride = 1 + max_rows;
    pub const words = 3 * 2 * stride;
    pub const bytes = words * 4;

    pub fn get(ls: *const Lists, w: Which, d: Dir) *const List {
        return &ls.l[@intFromEnum(w)][@intFromEnum(d)];
    }

    /// A list's byte offset in the device block.
    pub fn offset(w: Which, d: Dir) usize {
        return (@as(usize, @intFromEnum(w)) * 2 + @intFromEnum(d)) * stride * 4;
    }

    pub fn encode(ls: *const Lists, out: *[words]i32) void {
        out.* = @splat(0);
        for (0..3) |w| for (0..2) |d| {
            const l = &ls.l[w][d];
            const at = (w * 2 + d) * stride;
            out[at] = @intCast(l.n);
            for (l.slice(), 0..) |r, i| out[at + 1 + i] = r;
        };
    }
};

/// A round's rows as both ranks see them (the same on every rank): each row's owner (rows.ownerOf).
pub const View = struct {
    owners: []const u32,

    fn checked(v: View) !void {
        if (v.owners.len > max_rows) return error.BadRingRows;
        for (v.owners) |o| if (o > 1) return error.BadRingRows; // TP2
    }
};

/// A round's row counts: x[r] the rows rank r holds (the rows its exchanges move).
pub const Counts = struct {
    x: [2]usize = .{ 0, 0 },

    pub fn rows(c: Counts) usize {
        return c.x[0] + c.x[1];
    }

    /// The rows rank r attends with picks (its own extent's).
    pub fn held(c: Counts, r: usize) usize {
        return c.x[r];
    }
};

pub fn counts(v: View) !Counts {
    try v.checked();
    var c: Counts = .{};
    for (v.owners) |o| c.x[o] += 1;
    return c;
}

/// Every exchange's send and receive lists on `rank`.
pub fn lists(v: View, rank: u32) !Lists {
    try v.checked();
    if (rank > 1) return error.BadRingRows;
    var mine: List = .{}; // rows this rank holds
    var theirs: List = .{}; // rows the peer holds
    for (v.owners, 0..) |o, i| {
        if (o == rank) mine.push(i) else theirs.push(i);
    }
    var ls: Lists = .{};
    const f = @intFromEnum(Which.fetch);
    const q = @intFromEnum(Which.q);
    const o = @intFromEnum(Which.out);
    ls.l[f] = .{ mine, theirs };
    ls.l[q] = .{ theirs, mine };
    ls.l[o] = .{ mine, theirs };
    return ls;
}

/// Where segment `s` of `n` listed rows starts in the slot (the stage's packing order), and the slot's bytes.
pub fn segOffset(x: Xfer, n: usize, s: usize) usize {
    var off: usize = 0;
    for (x.segs[0..s]) |g| off += n * g.row_bytes;
    return off;
}

pub fn slotBytes(x: Xfer, n: usize) usize {
    return segOffset(x, n, x.n);
}

/// The host model of dsv41_shard.cu's tf_ds_ring_stage_rows: the slot's bytes as the proxy writes them, from the
/// sender's memory `mem` (addresses are offsets into it). Returns the size it publishes.
pub fn stageRef(x: Xfer, mem: []const u8, list: *const List, slot: []u8) usize {
    const n = list.n;
    for (x.segs[0..x.n], 0..) |g, s| {
        const off = segOffset(x, n, s);
        for (list.slice(), 0..) |r, c| {
            const from: usize = @intCast(g.src + r * g.src_stride);
            @memcpy(slot[off + c * g.row_bytes ..][0..g.row_bytes], mem[from..][0..g.row_bytes]);
        }
    }
    return slotBytes(x, n);
}

/// The host model of tf_ds_ring_collect_rows: the received slot's rows into the receiver's memory at their rows.
pub fn collectRef(x: Xfer, slot: []const u8, list: *const List, mem: []u8) void {
    const n = list.n;
    for (x.segs[0..x.n], 0..) |g, s| {
        const off = segOffset(x, n, s);
        for (list.slice(), 0..) |r, c| {
            const to: usize = @intCast(g.dst + r * g.dst_stride);
            @memcpy(mem[to..][0..g.row_bytes], slot[off + c * g.row_bytes ..][0..g.row_bytes]);
        }
    }
}

// -- the three decode exchanges -------------------------------------------------------------------------------------

/// The fetch's exchange on rank `rank`: its listed rows' kk picked rows (codes, then scales) from its slot of the
/// fetch buffers (fc codes, fs scales: [world][R][K] rows of cb, sb bytes, round.fetchDecode) into the peer's slot
/// rows on the peer. Only the first kk of a row's K are sent (kk <= K).
pub fn fetchXfer(fc: u64, fs: u64, rank: u32, R: usize, K: usize, kk: usize, cb: usize, sb: usize) Xfer {
    const me = shard.decodeSlots(rank, R, K, cb, sb);
    const peer = shard.decodeSlots(1 - rank, R, K, cb, sb);
    var x: Xfer = .{ .n = 2 };
    x.segs[0] = .{ .src = fc + me.codes, .dst = fc + peer.codes, .row_bytes = kk * cb, .src_stride = K * cb, .dst_stride = K * cb };
    x.segs[1] = .{ .src = fs + me.scales, .dst = fs + peer.scales, .row_bytes = kk * sb, .src_stride = K * sb, .dst_stride = K * sb };
    return x;
}

/// Owner decode's queries: this rank's q [R][hl][hd] bf16 of the listed rows, into the peer's slot of the gathered
/// queries (qall [world][R][hl][hd]: rank `rank`'s slot on the peer) at the same rows.
pub fn qXfer(q: u64, qall: u64, rank: u32, R: usize, hl: usize, hd: usize) Xfer {
    const row = hl * hd * 2;
    var x: Xfer = .{ .n = 1 };
    // the receiver's destination is the sender's slot: on the receiving rank that is its peer's, so name it from the
    // receiver's side (dst is only read by the receiver: qall + peer * R * row)
    x.segs[0] = .{ .src = q, .dst = qall + @as(u64, 1 - rank) * R * row, .row_bytes = row, .src_stride = row, .dst_stride = row };
    return x;
}

/// Owner decode's outputs: the peer's heads' rotated wo_a inputs of the listed rows (xh_peer, [groups][R][gk] fp16)
/// straight into the peer's own xhwo rows (the same layout), one segment a group.
pub fn outXfer(xh_peer: u64, xhwo: u64, R: usize, groups: usize, gk: usize) !Xfer {
    if (groups == 0 or groups > max_segs) return error.BadRingRows;
    const x = shard.xhPlanes(R, groups, gk);
    var o: Xfer = .{ .n = groups };
    for (0..groups) |g| o.segs[g] = .{ .src = xh_peer + g * x.plane_bytes, .dst = xhwo + g * x.plane_bytes, .row_bytes = x.row_bytes, .src_stride = x.row_bytes, .dst_stride = x.row_bytes };
    return o;
}

/// The most rows whose every listed row fits a slot of `max_bytes` (0: none), capped at max_rows.
pub fn fitRows(row_bytes: usize, max_bytes: usize) usize {
    if (row_bytes == 0) return 0;
    return @min(max_rows, max_bytes / row_bytes);
}

// -- a layer's ring order -------------------------------------------------------------------------------------------

/// One ring transfer of a compressed layer's sharded attention.
pub const Step = enum(u8) {
    stage_fetch,
    collect_fetch,
    stage_q,
    collect_q,
    stage_out,
    collect_out,

    pub fn which(s: Step) Which {
        return switch (s) {
            .stage_fetch, .collect_fetch => .fetch,
            .stage_q, .collect_q => .q,
            .stage_out, .collect_out => .out,
        };
    }

    pub fn dir(s: Step) Dir {
        return switch (s) {
            .stage_fetch, .stage_q, .stage_out => .send,
            .collect_fetch, .collect_q, .collect_out => .recv,
        };
    }
};

/// The ring transfers a compressed layer issues, in ring order, the same on both ranks: each stage directly followed
/// by its collect. Owner decode's early query send (at q, or on the shard side stream) moves where the stage is
/// issued among the layer's other kernels, not its ring order.
pub fn steps(mode: shard.Mode) []const Step {
    return switch (mode) {
        .fetch => &.{ .stage_fetch, .collect_fetch },
        .owner => &.{ .stage_q, .collect_q, .stage_out, .collect_out },
        .off, .auto => &.{},
    };
}

/// round.zig's check of the ring order as it issues a layer's transfers: `begin` at the layer's first, `take` before
/// each, `done` at the attention's end. Any other order is ShardRingOrder (before a launch, so neither rank's ring
/// parts from the other's).
pub const Cursor = struct {
    s: []const Step = &.{},
    i: usize = 0,

    pub fn begin(c: *Cursor, mode: shard.Mode) !void {
        if (!c.done()) return error.ShardRingOrder;
        c.* = .{ .s = steps(mode) };
    }

    pub fn take(c: *Cursor, s: Step) !void {
        if (c.i >= c.s.len or c.s[c.i] != s) return error.ShardRingOrder;
        c.i += 1;
    }

    pub fn done(c: *const Cursor) bool {
        return c.i == c.s.len;
    }

    pub fn reset(c: *Cursor) void {
        c.* = .{};
    }
};

// -- auto: fetch or owner decode a round ----------------------------------------------------------------------------

/// A listed row's bytes in each exchange (the model's: kk picks x 288 B; hl x hd x 2; wo_k x 2).
pub const Sizes = struct { fetch_row: usize, q_row: usize, out_row: usize };

/// What a round's decode exchanges cost beyond the replicated engine's attention, a compressed layer, in ns. The
/// defaults are estimates, not measurements (~11 us a ring round trip, ~22 GB/s over the dual rails, ~128 GB/s for a
/// stage or collect copy, a 32-head pass's launches); TF_DS_KV_SHARD_COST sets them, the same on both ranks, from
/// the TF_DS_CLOCK_MARKS=4 profile at 1, 2, 4 and 8 streams (docs/port/shard-ring.md "Setting the cost model").
/// - fetch: fixed + a row's flight for the busier sender's rows, plus its stage and collect copies.
/// - owner: fixed + a row's output flight (the part the own pass does not hide), plus four copies (q and outputs,
///   staged and collected), plus the second 32-head pass on the busier rank: its launches, a pass term a picked row
///   (two passes over the held rows against one over every row) and a term a masked row (each pass still runs the
///   window attention of the rows it masks).
pub const Cost = struct {
    fetch_fixed: u32 = 13_000,
    fetch_row: u32 = 7_000,
    owner_fixed: u32 = 8_000,
    owner_row: u32 = 1_600,
    pass_fixed: u32 = 6_000,
    pass_row: u32 = 600,
    mask_row: u32 = 200,
    copy_fixed: u32 = 2_000,
    copy_kib: u32 = 8,

    pub const fields = @typeInfo(Cost).@"struct".field_names;

    /// TF_DS_KV_SHARD_COST: nine comma-separated integers in the field order (null or empty: the defaults).
    pub fn parse(text: ?[]const u8) !Cost {
        var c: Cost = .{};
        const t = text orelse return c;
        if (t.len == 0) return c;
        var it = std.mem.splitScalar(u8, t, ',');
        inline for (fields) |f| {
            const v = it.next() orelse return error.BadKvShardCost;
            @field(c, f) = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return error.BadKvShardCost;
        }
        if (it.next() != null) return error.BadKvShardCost;
        return c;
    }

    fn copies(c: Cost, kernels: u64, bytes: u64) u64 {
        return kernels * c.copy_fixed + @as(u64, c.copy_kib) * bytes / 1024;
    }

    /// The fetch's and owner decode's extra ns a compressed layer for a round of these counts.
    pub fn fetchNs(c: Cost, n: Counts, z: Sizes) u64 {
        const m = @max(n.x[0], n.x[1]);
        const moved: u64 = n.x[0] + n.x[1]; // each rank stages its rows and collects the other's
        return @as(u64, c.fetch_fixed) + @as(u64, c.fetch_row) * m + c.copies(2, moved * z.fetch_row);
    }

    pub fn ownerNs(c: Cost, n: Counts, z: Sizes) u64 {
        const m = @max(n.x[0], n.x[1]);
        const moved: u64 = n.x[0] + n.x[1];
        const R: i64 = @intCast(n.rows());
        var pass: i64 = 0;
        for (0..2) |r| {
            const h: i64 = @intCast(n.held(r));
            const p = @as(i64, c.pass_fixed) + @as(i64, c.pass_row) * (2 * h - R) + 2 * @as(i64, c.mask_row) * (R - h);
            pass = @max(pass, p);
        }
        return @as(u64, c.owner_fixed) + @as(u64, c.owner_row) * m + c.copies(4, moved * (z.q_row + z.out_row)) + @as(u64, @intCast(pass));
    }

    /// The round's decode mode: fetch for a round wider than the split path (`decode_rows`: owner decode has no
    /// one-pass variant), else owner decode unless the fetch costs less (a tie goes to owner decode). round.forward
    /// sets the split path's bound (tri.decode_rows) to the round bound (TF_DS_ROUND_ROWS) for the round, so every
    /// decode round is on the split path and may go owner: the wide-round rule binds only a smaller bound.
    pub fn choose(c: Cost, n: Counts, z: Sizes, decode_rows: usize) shard.Mode {
        if (n.rows() > decode_rows) return .fetch;
        return if (c.fetchNs(n, z) < c.ownerNs(n, z)) .fetch else .owner;
    }
};

// -- a round's decision (round.shardRound) ---------------------------------------------------------------------------

/// What round.shardRound decides a round from, every input the same on both ranks: the round's view (owners) and the
/// settings the ranks agreed at useShard (the switch, auto's cost model, the ring, each mode's fit from the model's
/// sizes).
pub const RoundIn = struct {
    view: View,
    /// TF_DS_KV_SHARD_DECODE: fetch, owner or auto.
    mode: shard.Mode,
    cost: Cost = .{},
    sizes: Sizes,
    /// The split path's rows at most (tri.decode_rows while a round runs: the round bound, Engine.round_rows).
    decode_rows: usize,
    /// The agreed ring (KvShard.use_ring) with its row-list block allocated (Round.rlist).
    ring: bool,
    fit_fetch: usize = 0,
    fit_owner: usize = 0,
};

/// A round's decode, as shardRound applies it (Round.smode, sring).
pub const Decision = struct {
    /// fetch or owner.
    mode: shard.Mode,
    /// The round's exchanges ride the RDMA ring (roundLists then makes its row lists).
    ring: bool,
};

/// The round's decode from what every rank holds alike (so both ranks decide the same): the switch's mode or auto's
/// choice from the view's counts, and the ring iff the round's rows fit a slot of the mode's exchanges.
pub fn decide(in: RoundIn) !Decision {
    const R = in.view.owners.len;
    const n = try counts(in.view);
    const m: shard.Mode = switch (in.mode) {
        .fetch, .owner => in.mode,
        .auto => in.cost.choose(n, in.sizes, in.decode_rows),
        .off => return error.NoKvShard,
    };
    const fit = if (m == .fetch) in.fit_fetch else in.fit_owner;
    return .{ .mode = m, .ring = in.ring and R <= fit };
}

/// The ring round's row lists on `rank` (null: the round is not on the ring).
pub fn roundLists(d: Decision, v: View, rank: u32) !?Lists {
    if (!d.ring) return null;
    return try lists(v, rank);
}

/// The settings both ranks must share before the first round (round.Round.useShard gathers them): whether the
/// exchanges ride the RDMA ring, and the cost model. `bad` marks a rank whose own settings did not parse.
pub const Agreed = extern struct {
    ring: u64,
    cost: [Cost.fields.len]u64,

    pub const bad = std.math.maxInt(u64);

    pub fn of(ring: bool, cost: anyerror!Cost) Agreed {
        var a: Agreed = .{ .ring = @intFromBool(ring), .cost = undefined };
        if (cost) |c| {
            inline for (Cost.fields, 0..) |f, i| a.cost[i] = @field(c, f);
        } else |_| {
            a.ring = bad;
            a.cost = @splat(bad);
        }
        return a;
    }

    /// Every rank's (gathered in rank order) equal to this one's; a sentinel refuses (KvShardSettingsMismatch on the
    /// ranks that parsed, their own parse error on the one that did not).
    pub fn same(mine: Agreed, all: []const Agreed) bool {
        for (all) |o| if (o.ring == bad or !std.meta.eql(o, mine)) return false;
        return true;
    }
};

// -- tests ----------------------------------------------------------------------------------------------------------

const testing = std.testing;

fn randomOwners(rng: std.Random, owners: []u32) void {
    const how = rng.uintLessThan(u8, 4);
    for (owners) |*o| o.* = switch (how) {
        0 => 0, // every row on rank 0 (one stream)
        1 => 1,
        else => rng.uintLessThan(u32, 2),
    };
}

/// A rank's `own` glue: 1 where it holds the row (its owner's).
fn ownOf(v: View, rank: u32, out: []i64) void {
    for (v.owners, out) |o, *x| x.* = @intFromBool(o == rank);
}

test "lists: rank r's send list of every exchange is rank 1-r's receive list, in order" {
    var prng = std.Random.DefaultPrng.init(0x11575);
    const rng = prng.random();
    for (0..3000) |_| {
        const R = 1 + rng.uintLessThan(usize, 48);
        var owners: [max_rows]u32 = undefined;
        randomOwners(rng, owners[0..R]);
        const v: View = .{ .owners = owners[0..R] };
        const n = try counts(v);
        try testing.expectEqual(R, n.rows());
        const ls = [2]Lists{ try lists(v, 0), try lists(v, 1) };
        for (0..2) |r| for ([_]Which{ .fetch, .q, .out }) |w| {
            const s = ls[r].get(w, .send).*;
            const g = ls[1 - r].get(w, .recv).*;
            try testing.expect(s.eql(g));
            // sized: the fetch and the outputs carry the sender's own rows, the queries the receiver's
            const want = if (w == .q) n.x[1 - r] else n.x[r];
            try testing.expectEqual(want, s.n);
            for (s.slice()) |row| try testing.expectEqual(@as(u32, @intCast(if (w == .q) 1 - r else r)), owners[row]);
            for (1..@max(s.n, 1)) |i| try testing.expect(s.rows[i - 1] < s.rows[i]); // in row order
        };
    }
    try testing.expectError(error.BadRingRows, lists(.{ .owners = &.{ 0, 2 } }, 0));
    try testing.expectError(error.BadRingRows, lists(.{ .owners = &.{0} }, 2));
}

test "Lists.encode: i32 [3][2][1 + max_rows], the count then the rows, at Lists.offset" {
    const owners = [_]u32{ 0, 1, 1, 0, 1 };
    const ls = try lists(.{ .owners = &owners }, 0);
    var dev: [Lists.words]i32 = undefined;
    ls.encode(&dev);
    try testing.expectEqual(@as(usize, 3 * 2 * 65 * 4), Lists.bytes);
    const at = struct {
        fn f(d: []const i32, w: Which, dr: Dir) []const i32 {
            return d[Lists.offset(w, dr) / 4 ..][0..Lists.stride];
        }
    }.f;
    try testing.expectEqualSlices(i32, &.{ 2, 0, 3 }, at(&dev, .fetch, .send)[0..3]);
    try testing.expectEqualSlices(i32, &.{ 3, 1, 2, 4 }, at(&dev, .fetch, .recv)[0..4]);
    try testing.expectEqualSlices(i32, &.{ 3, 1, 2, 4 }, at(&dev, .q, .send)[0..4]);
    try testing.expectEqualSlices(i32, &.{ 2, 0, 3 }, at(&dev, .q, .recv)[0..3]);
    try testing.expectEqualSlices(i32, &.{ 2, 0, 3 }, at(&dev, .out, .send)[0..3]);
    try testing.expectEqualSlices(i32, &.{ 3, 1, 2, 4 }, at(&dev, .out, .recv)[0..4]);
    try testing.expectEqual(@as(i32, 0), at(&dev, .out, .recv)[4]); // the rest zero
}

test "slot layout: segments one after another, each n rows of its row bytes; sized by the listed rows" {
    var x: Xfer = .{ .n = 3 };
    x.segs[0] = .{ .row_bytes = 256 * 4, .src_stride = 256 * 8, .dst_stride = 256 * 8 };
    x.segs[1] = .{ .row_bytes = 32 * 4, .src_stride = 32 * 8, .dst_stride = 32 * 8 };
    x.segs[2] = .{ .row_bytes = 16, .src_stride = 16, .dst_stride = 16 };
    try testing.expectEqual(@as(usize, 0), segOffset(x, 5, 0));
    try testing.expectEqual(@as(usize, 5 * 1024), segOffset(x, 5, 1));
    try testing.expectEqual(@as(usize, 5 * (1024 + 128)), segOffset(x, 5, 2));
    try testing.expectEqual(@as(usize, 5 * (1024 + 128 + 16)), slotBytes(x, 5));
    try testing.expectEqual(@as(usize, 0), slotBytes(x, 0)); // a rank that holds no row sends an empty slot
    try testing.expectEqual(@as(usize, 7 * 1168), x.capacity(7));
    try x.check(7, 7 * 1168);
    try testing.expectError(error.TooLarge, x.check(7, 7 * 1168 - 16));
    try testing.expectError(error.BadRingRows, x.check(max_rows + 1, 1 << 30));
    var y = x;
    y.segs[1].row_bytes = 24;
    try testing.expectError(error.BadRingRows, y.check(1, 1 << 20));
    y = x;
    y.segs[0].src = 8;
    try testing.expectError(error.BadRingRows, y.check(1, 1 << 20));
    try testing.expectEqual(@as(usize, 7), fitRows(1168, 7 * 1168 + 1167));
    try testing.expectEqual(@as(usize, max_rows), fitRows(16, 1 << 30));
    try testing.expectEqual(@as(usize, 0), fitRows(0, 1 << 30));
}

test "the served sizes fit the ring's 4.25 MiB slot at 16 rows (fetch, queries, outputs); 48 fetch rows do not" {
    const slot: usize = 4456448; // model.ring_bytes
    const K = 512;
    const f = fetchXfer(0, 1 << 30, 0, 16, K, K, 256, 32);
    try f.check(16, slot);
    try testing.expectEqual(@as(usize, 16 * 512 * 288), f.capacity(16));
    try testing.expectError(error.TooLarge, f.check(48, slot));
    try testing.expectEqual(@as(usize, 30), fitRows(f.rowBytes(), slot));
    const q = qXfer(0, 1 << 30, 1, 16, 32, 512);
    try q.check(16, slot);
    try q.check(48, slot);
    const o = try outXfer(0, 1 << 30, 16, 4, 4096);
    try o.check(16, slot);
    try testing.expectEqual(@as(usize, 16 * 16384 * 2), o.capacity(16));
    try testing.expectError(error.BadRingRows, outXfer(0, 0, 16, 9, 4096));
}

test "fetch over the ring: each row reads the replicated plane's bytes in pick order; only held rows are sent" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rng = prng.random();
    const cb = 256;
    const sb = 32;
    for (0..60) |trial| {
        const R = 1 + rng.uintLessThan(usize, 16);
        const K = 16 * (1 + rng.uintLessThan(usize, 4));
        const kk = 1 + rng.uintLessThan(usize, K);
        const part = 4096;
        var owners: [max_rows]u32 = undefined;
        randomOwners(rng, owners[0..R]);
        const v: View = .{ .owners = owners[0..R] };
        // the pool-wide plane (what the replicated engine reads) and each owner's part of it
        const plane_rows = 2 * part;
        const codes = try gpa.alloc(u8, plane_rows * cb);
        defer gpa.free(codes);
        const scales = try gpa.alloc(u8, plane_rows * sb);
        defer gpa.free(scales);
        rng.bytes(codes);
        rng.bytes(scales);
        var bases: [max_rows]i64 = undefined;
        var idx: [max_rows * 64]i64 = undefined;
        for (0..R) |r| {
            bases[r] = @intCast(owners[r] * part + 2048 * rng.uintLessThan(usize, 2));
            for (0..kk) |j| idx[r * kk + j] = if (rng.uintLessThan(u8, 8) == 0) -1 else @intCast(rng.uintLessThan(usize, 2048));
        }
        // each rank's memory: its local plane (its part's rows from 0), then the fetch buffers fc, fs
        const fc_off = part * cb;
        const fs_off = fc_off + 2 * R * K * cb;
        const mem_len = fs_off + 2 * R * K * sb;
        var mem: [2][]u8 = undefined;
        var fixed: [2][]u8 = undefined; // the NCCL path's buffers after the in-place all-gather
        var slots: [2][]u8 = undefined;
        var sent: [2]usize = undefined;
        var outs: [2][]i64 = undefined;
        for (0..2) |k| {
            mem[k] = try gpa.alloc(u8, mem_len);
            fixed[k] = try gpa.alloc(u8, mem_len);
            slots[k] = try gpa.alloc(u8, R * K * (cb + sb));
            outs[k] = try gpa.alloc(i64, R * kk);
            @memset(mem[k], 0xAA);
        }
        defer for (0..2) |k| {
            gpa.free(mem[k]);
            gpa.free(fixed[k]);
            gpa.free(slots[k]);
            gpa.free(outs[k]);
        };
        var own: [2][max_rows]i64 = undefined;
        var ls: [2]Lists = undefined;
        for (0..2) |k| {
            const rank: u32 = @intCast(k);
            ownOf(v, rank, own[k][0..R]);
            ls[k] = try lists(v, rank);
            @memcpy(mem[k][0 .. part * cb], codes[k * part * cb ..][0 .. part * cb]);
            var lbase: [max_rows]i64 = undefined;
            for (0..R) |r| lbase[r] = if (owners[r] == rank) bases[r] - @as(i64, @intCast(k * part)) else 0;
            const sl = shard.decodeSlots(rank, R, K, cb, sb);
            shard.FetchRef.run(.{
                .codes = mem[k][0 .. part * cb],
                .scales = scales[k * part * sb ..][0 .. part * sb],
                .cb = cb,
                .sb = sb,
                .idx = idx[0 .. R * kk],
                .ks = kk,
                .kk = kk,
                .rows = R,
                .sbase = lbase[0..R],
                .own = own[k][0..R],
                .copy = true,
                .dcodes = mem[k][fc_off + sl.codes ..][0..sl.slot_c],
                .dscales = mem[k][fs_off + sl.scales ..][0..sl.slot_s],
                .dk = K,
                .out = outs[k],
                .kos = kk,
                .how = .relative,
            });
        }
        for (0..2) |k| @memcpy(fixed[k], mem[k]);
        for (0..2) |k| for (0..2) |p| {
            const sl = shard.decodeSlots(@intCast(p), R, K, cb, sb);
            @memcpy(fixed[k][fc_off + sl.codes ..][0..sl.slot_c], mem[p][fc_off + sl.codes ..][0..sl.slot_c]);
            @memcpy(fixed[k][fs_off + sl.scales ..][0..sl.slot_s], mem[p][fs_off + sl.scales ..][0..sl.slot_s]);
        };
        const n = try counts(v);
        for (0..2) |k| {
            const x = fetchXfer(fc_off, fs_off, @intCast(k), R, K, kk, cb, sb);
            sent[k] = stageRef(x, mem[k], ls[k].get(.fetch, .send), slots[k]);
            try testing.expectEqual(n.x[k] * kk * (cb + sb), sent[k]);
        }
        for (0..2) |k| {
            const x = fetchXfer(fc_off, fs_off, @intCast(k), R, K, kk, cb, sb);
            collectRef(x, slots[1 - k][0..sent[1 - k]], ls[k].get(.fetch, .recv), mem[k]);
        }
        for (0..2) |k| for (0..R) |r| {
            var want: std.ArrayList(u8) = .empty;
            defer want.deinit(gpa);
            var got: std.ArrayList(u8) = .empty;
            defer got.deinit(gpa);
            var old: std.ArrayList(u8) = .empty;
            defer old.deinit(gpa);
            var abs: [64]i64 = undefined;
            for (0..kk) |j| abs[j] = if (idx[r * kk + j] < 0) -1 else idx[r * kk + j];
            try shard.attnReads(gpa, codes, scales, cb, sb, bases[r], abs[0..kk], &want);
            const fb: i64 = @intCast(shard.fetchBase(owners[r], R, r, K));
            const nfc = 2 * R * K;
            try shard.attnReads(gpa, mem[k][fc_off..][0 .. nfc * cb], mem[k][fs_off..][0 .. nfc * sb], cb, sb, fb, outs[k][r * kk ..][0..kk], &got);
            try shard.attnReads(gpa, fixed[k][fc_off..][0 .. nfc * cb], fixed[k][fs_off..][0 .. nfc * sb], cb, sb, fb, outs[k][r * kk ..][0..kk], &old);
            testing.expectEqualSlices(u8, want.items, got.items) catch |err| {
                std.debug.print("fetch trial {d} rank {d} row {d}\n", .{ trial, k, r });
                return err;
            };
            try testing.expectEqualSlices(u8, old.items, got.items);
        };
    }
}

test "owner decode over the ring: the owner's gathered queries and the receiver's xhwo rows equal the NCCL path's" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0e1d);
    const rng = prng.random();
    for (0..120) |_| {
        const R = 1 + rng.uintLessThan(usize, 16);
        const hl = 2;
        const hd = 8 * (1 + rng.uintLessThan(usize, 4)); // rows of hl * hd * 2 bytes, whole words
        const groups = 1 + rng.uintLessThan(usize, 4);
        const gk = 8 * (1 + rng.uintLessThan(usize, 8));
        var owners: [max_rows]u32 = undefined;
        randomOwners(rng, owners[0..R]);
        const v: View = .{ .owners = owners[0..R] };
        const cnt = try counts(v);
        const qrow = hl * hd * 2;
        const q_off: usize = 0;
        const qall_off = R * qrow;
        const xp_off = qall_off + 2 * R * qrow;
        const xh_off = xp_off + groups * R * gk * 2;
        const len = xh_off + groups * R * gk * 2;
        var mem: [2][]u8 = undefined;
        var slot: [2][]u8 = undefined;
        var n: [2]usize = undefined;
        var own: [2][max_rows]i64 = undefined;
        var ls: [2]Lists = undefined;
        for (0..2) |k| {
            mem[k] = try gpa.alloc(u8, len);
            slot[k] = try gpa.alloc(u8, len);
            rng.bytes(mem[k]);
            ownOf(v, @intCast(k), own[k][0..R]);
            ls[k] = try lists(v, @intCast(k));
        }
        defer for (0..2) |k| {
            gpa.free(mem[k]);
            gpa.free(slot[k]);
        };
        const before = [2][]u8{ try gpa.dupe(u8, mem[0]), try gpa.dupe(u8, mem[1]) };
        defer for (before) |b| gpa.free(b);
        // queries: the NCCL path all-gathers q into qall [world][R] (row r of rank p's q at qall + (p R + r) qrow);
        // the peer-head pass reads the peer's q of the rows this rank holds
        for (0..2) |k| n[k] = stageRef(qXfer(q_off, qall_off, @intCast(k), R, hl, hd), mem[k], ls[k].get(.q, .send), slot[k]);
        for (0..2) |k| collectRef(qXfer(q_off, qall_off, @intCast(k), R, hl, hd), slot[1 - k][0..n[1 - k]], ls[k].get(.q, .recv), mem[k]);
        for (0..2) |k| {
            const p = 1 - k;
            try testing.expectEqual(cnt.x[p] * qrow, n[k]); // q of the rows the peer holds
            for (0..R) |r| {
                const at = qall_off + (p * R + r) * qrow;
                if (owners[r] == k) {
                    try testing.expectEqualSlices(u8, before[p][q_off + r * qrow ..][0..qrow], mem[k][at..][0..qrow]);
                } else try testing.expectEqualSlices(u8, before[k][at..][0..qrow], mem[k][at..][0..qrow]); // untouched
            }
        }
        // outputs: the NCCL path exchanges every row's xh_peer then selects the rows not held (own == 0) into xhwo
        var expect: [2][]u8 = undefined;
        for (0..2) |k| {
            expect[k] = try gpa.dupe(u8, mem[k][xh_off..][0 .. groups * R * gk * 2]);
            shard.selectRef(mem[1 - k][xp_off..][0 .. groups * R * gk * 2], expect[k], own[k][0..R], 0, shard.xhPlanes(R, groups, gk));
        }
        defer for (expect) |x| gpa.free(x);
        for (0..2) |k| n[k] = stageRef(try outXfer(xp_off, xh_off, R, groups, gk), mem[k], ls[k].get(.out, .send), slot[k]);
        for (0..2) |k| collectRef(try outXfer(xp_off, xh_off, R, groups, gk), slot[1 - k][0..n[1 - k]], ls[k].get(.out, .recv), mem[k]);
        for (0..2) |k| {
            try testing.expectEqual(cnt.x[k] * groups * gk * 2, n[k]); // the rows this rank holds
            try testing.expectEqualSlices(u8, expect[k], mem[k][xh_off..][0 .. groups * R * gk * 2]);
        }
    }
}

const served: Sizes = .{ .fetch_row = 512 * 288, .q_row = 32 * 512 * 2, .out_row = 16384 * 2 };

test "auto: both ranks choose the same mode; wide rounds always fetch; balanced rounds go owner; the costs decide" {
    const c: Cost = .{};
    var prng = std.Random.DefaultPrng.init(0xa070);
    const rng = prng.random();
    for (0..4000) |_| {
        const R = 1 + rng.uintLessThan(usize, 48);
        var owners: [max_rows]u32 = undefined;
        randomOwners(rng, owners[0..R]);
        const v: View = .{ .owners = owners[0..R] };
        // each rank holds the same replicated owners, so the same counts and the same choice; the ranks' roles
        // swapped give the same choice too
        const n = try counts(v);
        var swapped = owners;
        for (swapped[0..R]) |*o| o.* = 1 - o.*;
        const ns = try counts(.{ .owners = swapped[0..R] });
        try testing.expectEqual(n.x[0], ns.x[1]);
        const m = c.choose(n, served, 16);
        try testing.expectEqual(m, c.choose(ns, served, 16));
        // rounds wider than the split path's bound (16 here) always fetch, whatever the costs (owner decode refuses
        // a one-pass round). In the engine the bound is the round bound (round.forward), so no served round is wider
        const any: Cost = .{ .fetch_fixed = rng.int(u32), .fetch_row = rng.int(u16), .pass_fixed = rng.int(u16) };
        if (R > 16) {
            try testing.expectEqual(shard.Mode.fetch, m);
            try testing.expectEqual(shard.Mode.fetch, any.choose(n, served, 16));
            try testing.expectEqual(shard.Mode.fetch, any.choose(ns, served, 16));
        }
    }
    try testing.expectEqual(shard.Mode.owner, c.choose(.{ .x = .{ 8, 8 } }, served, 16));
    try testing.expectEqual(shard.Mode.owner, c.choose(.{ .x = .{ 2, 2 } }, served, 16));
    try testing.expectEqual(shard.Mode.owner, c.choose(.{ .x = .{ 4, 0 } }, served, 16)); // one stream with drafts
    try testing.expectEqual(shard.Mode.owner, c.choose(.{ .x = .{ 1, 0 } }, served, 16)); // one stream, one row
    try testing.expectEqual(shard.Mode.fetch, c.choose(.{ .x = .{ 9, 8 } }, served, 16));
    // the copy terms: four 2 us stage/collect kernels plus 8 ns a KiB of the 64 KiB a row moves
    try testing.expectEqual(@as(u64, 8_000 + 1_600 + 8_000 + 8 * 64 + 6_000 + 600 * 1), c.ownerNs(.{ .x = .{ 1, 0 } }, served));
    try testing.expectEqual(@as(u64, 13_000 + 7_000 + 4_000 + 8 * 144), c.fetchNs(.{ .x = .{ 1, 0 } }, served));
    // the masked-row term: a one-sided round's rank 1 masks every row in both passes
    const masked: Cost = .{ .mask_row = 1_000, .pass_row = 0, .pass_fixed = 0 };
    try testing.expectEqual(@as(u64, 8_000 + 4 * 1_600 + 4 * 2_000 + 8 * 4 * 64 + 2 * 1_000 * 4), masked.ownerNs(.{ .x = .{ 4, 0 } }, served));
    // a dear attention pass (as measured, say): one-sided rounds of few rows fetch, balanced ones do not
    const pass: Cost = .{ .pass_fixed = 25_000, .pass_row = 6_000 };
    try testing.expectEqual(shard.Mode.fetch, pass.choose(.{ .x = .{ 1, 0 } }, served, 16));
    try testing.expectEqual(shard.Mode.fetch, pass.choose(.{ .x = .{ 0, 1 } }, served, 16));
    try testing.expectEqual(shard.Mode.owner, pass.choose(.{ .x = .{ 8, 8 } }, served, 16));
    const cheap: Cost = .{ .fetch_fixed = 0, .fetch_row = 0, .copy_fixed = 0, .copy_kib = 0 };
    const dear: Cost = .{ .fetch_fixed = 1 << 30 };
    for (1..17) |m| {
        try testing.expectEqual(shard.Mode.fetch, cheap.choose(.{ .x = .{ m, 16 - @min(m, 16) } }, served, 16));
        try testing.expectEqual(shard.Mode.owner, dear.choose(.{ .x = .{ m, 0 } }, served, 16));
    }
}

test "Cost.parse: nine integers in field order, else refused; Agreed refuses any difference or a sentinel" {
    try testing.expectEqual(Cost{}, try Cost.parse(null));
    try testing.expectEqual(Cost{}, try Cost.parse(""));
    const c = try Cost.parse("1,2,3,4,5,6,7,8,9");
    try testing.expectEqual(Cost{ .fetch_fixed = 1, .fetch_row = 2, .owner_fixed = 3, .owner_row = 4, .pass_fixed = 5, .pass_row = 6, .mask_row = 7, .copy_fixed = 8, .copy_kib = 9 }, c);
    for ([_][]const u8{ "1,2,3", "1,2,3,4,5,6", "1,2,3,4,5,6,7,8,9,10", "a,2,3,4,5,6,7,8,9", "-1,2,3,4,5,6,7,8,9", "1,,3,4,5,6,7,8,9" }) |t|
        try testing.expectError(error.BadKvShardCost, Cost.parse(t));
    const a = Agreed.of(true, c);
    try testing.expect(a.same(&.{ a, a }));
    try testing.expect(!a.same(&.{ a, Agreed.of(false, c) }));
    try testing.expect(!a.same(&.{ a, Agreed.of(true, Cost{}) }));
    const broken = Agreed.of(true, error.BadKvShardCost);
    try testing.expect(!a.same(&.{ a, broken }));
    try testing.expect(!broken.same(&.{ broken, broken })); // a sentinel never agrees, even with itself
    try testing.expectEqual(@as(usize, 8 * 10), @sizeOf(Agreed));
}

test "a layer's ring order: each mode's stages paired with their collects, the same on both ranks, sizes agreeing" {
    var prng = std.Random.DefaultPrng.init(0x0bde);
    const rng = prng.random();
    const c: Cost = .{};
    const K = 512;
    for (0..2000) |_| {
        const R = 1 + rng.uintLessThan(usize, 16);
        var owners: [max_rows]u32 = undefined;
        randomOwners(rng, owners[0..R]);
        const v: View = .{ .owners = owners[0..R] };
        const n = try counts(v);
        // the configured mode: fetch, owner (with the early query send or without: the same ring order), or auto's
        // choice, which each rank makes from its own copy of the replicated counts
        const cfg = rng.uintLessThan(u8, 4);
        var mode: [2]shard.Mode = undefined;
        for (0..2) |r| mode[r] = switch (cfg) {
            0 => .fetch,
            1, 2 => .owner,
            else => c.choose(n, served, 16),
        };
        try testing.expectEqual(mode[0], mode[1]);
        const seq = [2][]const Step{ steps(mode[0]), steps(mode[1]) };
        try testing.expectEqualSlices(Step, seq[0], seq[1]);
        try testing.expect(seq[0].len % 2 == 0 and seq[0].len > 0);
        const ls = [2]Lists{ try lists(v, 0), try lists(v, 1) };
        var i: usize = 0;
        while (i < seq[0].len) : (i += 2) {
            const s = seq[0][i];
            const g = seq[0][i + 1];
            // a stage, then the collect of the same exchange: adjacent in ring order
            try testing.expectEqual(Dir.send, s.dir());
            try testing.expectEqual(Dir.recv, g.dir());
            try testing.expectEqual(s.which(), g.which());
            const x: Xfer = switch (s.which()) {
                .fetch => fetchXfer(0, 1 << 30, 0, R, K, K, 256, 32),
                .q => qXfer(0, 1 << 30, 0, R, 32, 512),
                .out => try outXfer(0, 1 << 30, R, 4, 4096),
            };
            for (0..2) |r| {
                // what rank r stages is the size and the rows rank 1-r collects
                const sent = ls[r].get(s.which(), .send).*;
                const got = ls[1 - r].get(g.which(), .recv).*;
                try testing.expect(sent.eql(got));
                try testing.expectEqual(slotBytes(x, sent.n), slotBytes(x, got.n));
            }
        }
        // the cursor accepts the sequence and refuses anything else
        var cur: Cursor = .{};
        try cur.begin(mode[0]);
        for (seq[0]) |s| try cur.take(s);
        try testing.expect(cur.done());
        try cur.begin(mode[0]);
        const wrong: Step = if (seq[0][0] == .stage_q) .stage_out else .collect_fetch;
        try testing.expectError(error.ShardRingOrder, cur.take(wrong));
        try testing.expectError(error.ShardRingOrder, cur.begin(mode[0])); // the layer before it unfinished
        cur.reset();
    }
    var cur: Cursor = .{};
    try cur.begin(.owner);
    try cur.take(.stage_q);
    try testing.expectError(error.ShardRingOrder, cur.take(.stage_out)); // a stage between a stage and its collect
    try testing.expect(!cur.done());
}

test "shardRound's decision: both ranks alike, the switch's mode or auto's choice, the ring iff the round fits" {
    var prng = std.Random.DefaultPrng.init(0xdec1de);
    const rng = prng.random();
    var seen: struct { fetch: usize = 0, owner: usize = 0, ring: usize = 0, unfit: usize = 0, auto_owner: usize = 0 } = .{};
    for (0..3000) |_| {
        const bound: usize = if (rng.boolean()) 16 else 32; // TF_DS_ROUND_ROWS (tri.decode_rows while the round runs)
        const R = 1 + rng.uintLessThan(usize, bound);
        const mode: shard.Mode = switch (rng.uintLessThan(u8, 3)) {
            0 => .fetch,
            1 => .owner,
            else => .auto,
        };
        var owners: [max_rows]u32 = undefined;
        randomOwners(rng, owners[0..R]);
        const v: View = .{ .owners = owners[0..R] };
        const cost: Cost = if (rng.boolean()) .{} else .{ .fetch_fixed = rng.uintLessThan(u32, 40_000), .fetch_row = rng.uintLessThan(u32, 10_000), .pass_fixed = rng.uintLessThan(u32, 30_000), .pass_row = rng.uintLessThan(u32, 4_000) };
        const ring = rng.uintLessThan(u8, 4) != 0;
        const fit_fetch = rng.uintLessThan(usize, R + 2);
        const fit_owner = rng.uintLessThan(usize, R + 2);
        var d: [2]Decision = undefined;
        for (0..2) |r| d[r] = try decide(.{ .view = v, .mode = mode, .cost = cost, .sizes = served, .decode_rows = bound, .ring = ring, .fit_fetch = fit_fetch, .fit_owner = fit_owner });
        // both ranks decide alike (the collectives and the ring order follow from it)
        try testing.expect(std.meta.eql(d[0], d[1]));
        const n = try counts(v);
        try testing.expectEqual(if (mode == .auto) cost.choose(n, served, bound) else mode, d[0].mode);
        if (d[0].mode == .fetch) seen.fetch += 1 else seen.owner += 1;
        if (mode == .auto and d[0].mode == .owner and n.rows() > 16) seen.auto_owner += 1;
        const fit = if (d[0].mode == .fetch) fit_fetch else fit_owner;
        try testing.expectEqual(ring and R <= fit, d[0].ring);
        if (ring and R > fit) seen.unfit += 1;
        if (!d[0].ring) {
            for (0..2) |r| try testing.expect(try roundLists(d[r], v, @intCast(r)) == null);
            continue;
        }
        seen.ring += 1;
        // every list a ring round sends fits the mode's fit (the stage's cap)
        for (0..2) |r| {
            const ls = (try roundLists(d[r], v, @intCast(r))).?;
            for ([_]Which{ .fetch, .q, .out }) |w| try testing.expect(ls.get(w, .send).n <= fit);
        }
    }
    try testing.expect(seen.fetch > 100 and seen.owner > 100 and seen.ring > 100 and seen.unfit > 100 and seen.auto_owner > 5);
    try testing.expectError(error.NoKvShard, decide(.{ .view = .{ .owners = &.{0} }, .mode = .off, .sizes = served, .decode_rows = 16, .ring = false }));
}

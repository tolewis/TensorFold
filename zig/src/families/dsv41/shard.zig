//! Port of wojo's GLM 0118 (PR #152, TF_GLM_KV_SHARD: owner-only compressed KV, sparse fetch, owner decode) to the
//! native DeepSeek-V4.1 family, its attention and storage half: the geometry, address translation, capacity plan and
//! packing order every rank derives alike, host-only (no CUDA: `zig test` runs it alone). The device side is
//! shard_ops.zig (the fetch and select kernels), prompt.zig (owner-only planes, the prompt fetch), round.zig (decode
//! glue, the decode fetch and owner-computed decode) and model.zig (copyExtent across parts).
//!
//! TF_DS_KV_SHARD=1 (TP2 only, off by default): the pool's positions are split in `world` equal parts, part p on rank
//! p, its owner. Only the owner holds the compressed latents (FP4 codes and E8M0 scales) of the four kv-source layers
//! for the extents in its part; the index keys, the window rings, the compressor's positional stores and the drafter's
//! state stay on every rank, over the whole pool, so both ranks compute the same index picks. An extent never crosses a
//! part (placement inside a part is round 2's, lanes.zig; every address translation here refuses a crossing).
//!
//! A rank's plane of a kv-source layer of ratio r: part / r rows of its own part, `pad_rows` rows as the replicated
//! plane has past its pool, then `trash` rows where a non-owner's decode rows write (one a round row: rounds write at
//! fixed shapes, so a row whose extent this rank does not hold writes its latent somewhere harmless). Its bytes per
//! token and rank: (sum over sources of 1/r) x (codes/world + scales/world + index), 530 at TP2 for this checkpoint's
//! ratios 2, 2, 2, 1 against 890 replicated (calculated, not measured: 360 saved a token a rank).
const std = @import("std");

/// The switch: off (the replicated planes, byte for byte today's engine), or sharded with the decode rounds' attention
/// by sparse fetch (the reference: the owner sends each row's picked latents, both ranks attend their own heads) or
/// owner-computed (queries gathered, the owner attends both ranks' heads, the other rank's rotated wo_a inputs sent
/// back). Prompt chunks always fetch.
pub const Mode = enum(u8) {
    off = 0,
    fetch = 1,
    owner = 2,

    /// TF_DS_KV_SHARD ("1": on) and TF_DS_KV_SHARD_DECODE ("owner", else the fetch reference), as every rank reads them.
    pub fn parse(on: ?[]const u8, decode: ?[]const u8) !Mode {
        const v = on orelse return .off;
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return .off;
        if (!std.mem.eql(u8, v, "1")) return error.BadKvShard;
        const d = decode orelse return .fetch;
        if (d.len == 0 or std.mem.eql(u8, d, "fetch")) return .fetch;
        if (std.mem.eql(u8, d, "owner")) return .owner;
        return error.BadKvShardDecode;
    }
};

/// A pool's chunk (model.chunk_rows): parts are whole blocks, so every ratio divides a part's first position.
pub const block: usize = 2048;
/// Rows past a pool's (or a part's) rows in every compressed plane (prompt.Caches: rows = cap / ratio + 2).
pub const pad_rows: usize = 2;

/// The parts of one rank's sharded pool: every rank holds the same numbers but its own `rank`.
pub const Shard = struct {
    rank: u32,
    world: u32,
    pool: usize, // positions of every part together
    part: usize, // positions a part (pool / world), whole blocks
    trash: usize, // plane rows past a part's (and its pad rows): a non-owner's decode rows, one a round row

    /// TP2 only (the four-node split shards its heads two ways already); a pool of whole blocks a part.
    pub fn init(rank: u32, world: u32, pool: usize, trash: usize) !Shard {
        if (world != 2) return error.KvShardNeedsTwoRanks;
        if (rank >= world) return error.BadRank;
        if (pool == 0 or pool % (world * block) != 0) return error.PoolNotWholeParts;
        return .{ .rank = rank, .world = world, .pool = pool, .part = pool / world, .trash = trash };
    }

    /// The rank that holds the compressed rows of pool position `pos`.
    pub fn owner(s: Shard, pos: usize) u32 {
        return @intCast(pos / s.part);
    }

    pub fn owns(s: Shard, pos: usize) bool {
        return s.owner(pos) == s.rank;
    }

    /// The other rank (TP2).
    pub fn peer(s: Shard) u32 {
        return 1 - s.rank;
    }

    /// The first pool position of part `p`.
    pub fn start(s: Shard, p: u32) usize {
        return @as(usize, p) * s.part;
    }

    /// Whether [base, base + size) lies in one part (and in the pool).
    pub fn within(s: Shard, base: usize, size: usize) bool {
        if (base >= s.pool or size > s.pool - base) return false;
        if (size == 0) return true;
        return s.owner(base) == s.owner(base + size - 1);
    }

    /// Pool position `pos` as a position of its owner's part.
    pub fn local(s: Shard, pos: usize) usize {
        return pos - s.start(s.owner(pos));
    }

    /// Rows of one rank's plane of a ratio-`ratio` source: its part's, the pad rows, the trash rows.
    pub fn planeRows(s: Shard, ratio: usize) usize {
        return s.part / ratio + pad_rows + s.trash;
    }

    /// Plane row of round row `i`'s trash (a non-owner's write lands there).
    pub fn trashRow(s: Shard, ratio: usize, i: usize) usize {
        std.debug.assert(i < s.trash);
        return s.part / ratio + pad_rows + i;
    }

    /// The compressed-row base of the extent at pool position `base` in this rank's plane (base / ratio of its part),
    /// null when this rank does not hold it.
    pub fn planeBase(s: Shard, base: usize, ratio: usize) ?usize {
        if (!s.owns(base)) return null;
        const l = s.local(base);
        std.debug.assert(l % ratio == 0);
        return l / ratio;
    }

    /// A decode row's compressed-store target in this rank's plane: the replicated target (`abs_row`, a row of the
    /// pool-wide plane: cbase + group, or the extent's last row as the ratio-2 scratch) moved into the owner's part,
    /// or this round row's trash row on the other rank.
    pub fn storeRow(s: Shard, abs_row: usize, base: usize, ratio: usize, i: usize) usize {
        if (!s.owns(base)) return s.trashRow(ratio, i);
        const first = s.start(s.owner(base)) / ratio;
        std.debug.assert(abs_row >= first and abs_row < first + s.part / ratio);
        return abs_row - first;
    }

    /// This rank's part of copying an extent's compressed rows (Move).
    pub fn extentMove(s: Shard, from: usize, to: usize, n: usize) !Move {
        return moveOf(s, from, to, n);
    }
};

/// One kv-source layer's compressed row bytes, for the capacity arithmetic: codes + scales (sharded) and the index keys'
/// codes + scales (replicated; 0 when the source has no index keys).
pub const Source = struct { ratio: usize, comp_row: usize, idx_row: usize };

/// This checkpoint's packed rows (config: head_dim 512, index_head_dim 128): 512 / 2 + 512 / 16 and 128 / 2 + 128 / 32.
pub const comp_row_bytes: usize = 288;
pub const idx_row_bytes: usize = 68;

/// Bytes `tokens` positions take on one rank over `sources`' planes (no pad or trash rows): every source's codes and
/// scales on every rank (replicated) or on one of `world` (sharded, each rank's share), its index keys on every rank.
/// `tokens` a multiple of every ratio.
pub fn planeBytes(sources: []const Source, tokens: usize, world: usize, sharded: bool) usize {
    var n: usize = 0;
    for (sources) |src| {
        std.debug.assert(tokens % src.ratio == 0);
        const rows = tokens / src.ratio;
        const comp = if (sharded) src.comp_row * rows / world else src.comp_row * rows;
        n += comp + src.idx_row * rows;
    }
    return n;
}

/// Bytes of one rank's planes for a pool of `pool` positions: replicated (pool / r + pad rows), or sharded (a part's
/// compressed rows with pad and trash rows, the index keys over the whole pool with its pad rows).
pub fn poolBytes(sources: []const Source, pool: usize, world: usize, sharded: bool, trash: usize) usize {
    var n: usize = 0;
    for (sources) |src| {
        const full = pool / src.ratio + pad_rows;
        const comp_rows = if (sharded) pool / world / src.ratio + pad_rows + trash else full;
        n += src.comp_row * comp_rows + src.idx_row * full;
    }
    return n;
}

/// What a start-up serves: the pool, its parts, and the longest single conversation (prompt + reply tokens).
pub const Plan = struct {
    pool: usize,
    part: usize, // the pool when unsharded
    /// The single-conversation limit to report: the requested window, or less when a part (or the pool) holds less.
    window: usize,
    /// The positions one stream's extent may take at most (its prompt, its reply, a round's rows and two, in whole
    /// chunks): the request check's bound (Model.request_cap).
    extent: usize,
    shrunk: bool,
};

/// The start-up's plan for `pool` positions (0008's TF_DS_POOL) and a requested `window` of prompt + reply tokens
/// (0008's TF_DS_WINDOW, --context on rank 0), `round_rows` a round's rows: an extent is
/// alignForward(prompt + reply + round_rows + 2, block) positions, so a part (or the pool) of P positions holds a
/// conversation of P - round_rows - 2 tokens. Sharded, the same pool is cut into one part per rank and every part
/// must hold the whole window (window + round_rows + 2 positions): a pool too small for that is refused
/// (PartTooSmall), never served with a shorter window. Unsharded the pool is today's: when it is the window, a
/// conversation fits a round's rows and two fewer tokens (reported as `shrunk`, as the stock engine serves it).
pub fn plan(pool: usize, window: usize, round_rows: usize, world: usize, sharded: bool) !Plan {
    if (pool == 0 or window == 0) return error.EmptyPool;
    const part = if (sharded) pool / world else pool;
    if (sharded and (pool % (world * block) != 0)) return error.PoolNotWholeParts;
    const room = part -| (round_rows + 2);
    const eff = @min(window, room);
    if (sharded and (room == 0 or eff < window)) return error.PartTooSmall; // every part holds the whole window
    const extent = @min(part, std.mem.alignForward(usize, window + round_rows + 2, block));
    return .{ .pool = pool, .part = part, .window = eff, .extent = extent, .shrunk = eff < window };
}

/// What copyExtent does with one rank's compressed rows of an extent copied from pool position `from` to `to` (the
/// replicated index keys are copied on every rank regardless): positions in each part, for the caller to divide by a
/// ratio. Within one part its owner copies and the other rank does nothing; across parts (a carry) the source's owner
/// sends its rows straight from its plane and the destination's owner receives them into its plane, in the same order
/// on both ranks. The source's rows stay either way (a copy, never a move).
pub const Move = union(enum) {
    none,
    local: struct { from: usize, to: usize },
    send: struct { from: usize, peer: u32 },
    recv: struct { to: usize, peer: u32 },
};

fn moveOf(s: Shard, from: usize, to: usize, n: usize) !Move {
    if (!s.within(from, n) or !s.within(to, n)) return error.CrossesPart;
    if (n == 0) return .none;
    const so = s.owner(from);
    const do = s.owner(to);
    if (so == do) return if (so == s.rank) .{ .local = .{ .from = s.local(from), .to = s.local(to) } } else .none;
    if (so == s.rank) return .{ .send = .{ .from = s.local(from), .peer = do } };
    if (do == s.rank) return .{ .recv = .{ .to = s.local(to), .peer = so } };
    return .none;
}

/// One decode row's TF_DS_KV_SHARD glue on one rank (round.zig's SGlue rows): each ratio's compressed-store target in
/// this rank's plane, each ratio's base of the row's extent in it (0 when this rank does not hold it), whether it
/// holds it, and the row's base in the decode fetch's gathered buffer.
pub const RowGlue = struct { ltarget1: usize, ltarget2: usize, lbase1: usize, lbase2: usize, own: bool, fbase: usize };

/// Round row `i` (of `rows`) at position `pos` of the extent [base, end), its owner `owner`, `k` picks a row: the
/// replicated targets (rounds.py _ix: ratio 1 base + pos; ratio 2 base / 2 + pos / 2 for a complete group, else the
/// extent's last row, the scratch) moved into this rank's plane (storeRow).
pub fn rowGlue(s: Shard, i: usize, rows: usize, pos: usize, base: usize, end: usize, owner_rank: u32, k: usize) !RowGlue {
    if (end <= base or !s.within(base, end - base) or base % block != 0) return error.CrossesPart;
    if (s.owner(base) != owner_rank) return error.BadPart;
    if (pos >= end - base or i >= rows or i >= s.trash) return error.BadRound;
    const t1 = base + pos;
    const t2 = if ((pos + 1) % 2 == 0) base / 2 + pos / 2 else end / 2 - 1;
    return .{
        .ltarget1 = s.storeRow(t1, base, 1, i),
        .ltarget2 = s.storeRow(t2, base, 2, i),
        .lbase1 = s.planeBase(base, 1) orelse 0,
        .lbase2 = s.planeBase(base, 2) orelse 0,
        .own = s.owns(base),
        .fbase = fetchBase(owner_rank, rows, i, k),
    };
}

/// The rank holding the compressed rows of a round row whose extent is [base, end) (signed: the lanes' glue), `part`
/// the lane's part for it (negative: none given, the base's). Malformed extents are refused before any arithmetic
/// (BadRound), a crossing one (CrossesPart), a part that is not the base's (BadPart).
pub fn rowOwner(s: Shard, base: i64, end: i64, part: i64) !u32 {
    if (base < 0 or end <= base) return error.BadRound;
    const b: usize = @intCast(base);
    if (!s.within(b, @as(usize, @intCast(end)) - b)) return error.CrossesPart;
    const own = s.owner(b);
    if (part >= 0 and part != own) return error.BadPart;
    return own;
}

/// TF_DS_KV_SHARD's index tensors by row (int64 [sglue_n][stride] each, a rank's own: round.Round.sglue): each ratio's
/// compressed-store target in this rank's plane (Shard.storeRow: the owner's row, else the row's trash row), each
/// ratio's base of the row's extent in this rank's plane (0 when it does not hold it), whether it holds it (1, 0), and
/// the row's base in the decode fetch's gathered buffer (fetchBase: the same on both ranks).
pub const SGlue = enum(u8) { ltarget1, ltarget2, lbase1, lbase2, own, fbase };
pub const sglue_n = @typeInfo(SGlue).@"enum".field_names.len;

/// A round's SGlue rows on this rank (round.setShardGlue uploads `out` as it is): row i at position pos[i] of the
/// extent [bases[i], ends[i]), its lane part parts[i] (null or negative: the base's), `k` picks a row, written at
/// out[field * stride + i]; every entry past the round's rows stays 0.
pub fn glueRows(s: Shard, pos: []const i64, bases: []const i64, ends: []const i64, parts: ?[]const i64, k: usize, stride: usize, out: []i64) !void {
    const R = pos.len;
    if (R > s.trash or R > stride) return error.WindowTooWide;
    if (bases.len != R or ends.len != R or out.len < sglue_n * stride) return error.BadRound;
    if (parts) |p| if (p.len != R) return error.BadRound;
    @memset(out[0 .. sglue_n * stride], 0);
    for (0..R) |i| {
        if (pos[i] < 0) return error.BadRound;
        const own = try rowOwner(s, bases[i], ends[i], if (parts) |p| p[i] else -1);
        const g = try rowGlue(s, i, R, @intCast(pos[i]), @intCast(bases[i]), @intCast(ends[i]), own, k);
        const at = struct {
            fn f(o: []i64, st: usize, f_: SGlue, r: usize) *i64 {
                return &o[@as(usize, @backingInt(f_)) * st + r];
            }
        }.f;
        at(out, stride, .ltarget1, i).* = @intCast(g.ltarget1);
        at(out, stride, .ltarget2, i).* = @intCast(g.ltarget2);
        at(out, stride, .lbase1, i).* = @intCast(g.lbase1);
        at(out, stride, .lbase2, i).* = @intCast(g.lbase2);
        at(out, stride, .own, i).* = @intFromBool(g.own);
        at(out, stride, .fbase, i).* = @intCast(g.fbase);
    }
}

/// The decode fetch's slot of rank `rank` in the gathered buffers ([world][R][K] rows of `cb` code and `sb` scale
/// bytes: round.fetchDecode packs there, the in-place all-gather joins the slots): byte offsets and sizes.
pub const Slots = struct { codes: usize, scales: usize, slot_c: usize, slot_s: usize };

pub fn decodeSlots(rank: u32, R: usize, K: usize, cb: usize, sb: usize) Slots {
    const first = fetchBase(rank, R, 0, K);
    return .{ .codes = first * cb, .scales = first * sb, .slot_c = R * K * cb, .slot_s = R * K * sb };
}

/// xhwo's layout ([wo_a group][R, gk] fp16: the merge's rotated wo_a input rows) as round.ownerDecode's select moves it:
/// a row's bytes, the planes (groups) and the bytes between them.
pub const XhPlanes = struct { row_bytes: usize, planes: usize, plane_bytes: usize };

pub fn xhPlanes(R: usize, groups: usize, gk: usize) XhPlanes {
    return .{ .row_bytes = gk * 2, .planes = groups, .plane_bytes = R * gk * 2 };
}

/// The host model of shard_ops.select (dsv41_shard.cu's tf_ds_select_rows_kernel): row r of every plane from `src`
/// into `dst` where sel[r] == want.
pub fn selectRef(src: []const u8, dst: []u8, sel: []const i64, want: i64, x: XhPlanes) void {
    for (0..x.planes) |p| for (sel, 0..) |v, r| {
        if (v != want) continue;
        const at = p * x.plane_bytes + r * x.row_bytes;
        @memcpy(dst[at..][0..x.row_bytes], src[at..][0..x.row_bytes]);
    };
}

// -- the sparse fetch's packing order -------------------------------------------------------------------------------

/// Rows of a prompt chunk one fetch block takes (the union fetch past the buffer): the most whole 16-row tiles whose
/// picks fit `fetch_rows` fetched rows (each row `kk` of them: a block's union is at most its picks), at least 16
/// (an even count: a block's positions view stays 16-byte aligned, the alignment the kit's kernels were specialized
/// for).
pub fn blockRows(fetch_rows: usize, kk: usize) usize {
    if (kk == 0) return fetch_rows;
    return @max(16, fetch_rows / kk / 16 * 16);
}

/// The decode fetch's per-row base in its gathered buffer ([world][rows][k] rows: each rank's rows it owns, the
/// all-gather's layout): row i of a round of `rows` whose owner is `own_rank`, `k` picks a row at most.
pub fn fetchBase(own_rank: u32, rows: usize, i: usize, k: usize) usize {
    return (@as(usize, own_rank) * rows + i) * k;
}

/// How a prompt chunk's compressed layer gets its picked rows to the rank that does not hold them; both ranks decide
/// from the same integers (and the union's count, which both compute from the same replicated picks):
/// - contiguous: the visible rows [0, n_comp_end) sent straight from the owner's plane into the fetch buffer at the
///   same rows, when they fit it and are no more rows than the chunk's picks: the other rank attends with the picks
///   unchanged (identical addressing; one launch, the replicated call with its compressed pointers moved);
/// - union (GLM 0118's history): the chunk's distinct picks, a mark a position, their prefix sums, the owner packing the
///   marked rows in position order and the other rank renumbering its picks to them (UnionRef); counted on both ranks
///   (one host read), sent once and attended in one launch when they fit the buffer, else per block of
///   unionBlockRows rows (a block's union is at most its picks, which fit).
pub const PromptFetch = enum { contiguous, @"union" };

pub fn promptFetch(n_comp_end: usize, n: usize, kk: usize, fetch_rows: usize) PromptFetch {
    return if (n_comp_end <= fetch_rows and n_comp_end <= n * kk) .contiguous else .@"union";
}

/// A union fetch's rows a block once the chunk's union is known to hold `count` rows: the whole chunk when they fit
/// the fetch buffer, else blockRows.
pub fn unionBlockRows(count: usize, n: usize, kk: usize, fetch_rows: usize) usize {
    return if (count <= fetch_rows) n else blockRows(fetch_rows, kk);
}

/// Rows [r0, r0 + m) of an n-row prompt chunk cut in blocks of B rows: the byte offsets of its query and output rows
/// (h heads of hd bf16 a row), its positions (int64) and its picks (int64, kk a row: the chunk's cidx).
pub const Block = struct { r0: usize, m: usize, act: usize, pos: usize, idx: usize };

pub fn promptBlock(r0: usize, n: usize, B: usize, h: usize, hd: usize, kk: usize) Block {
    std.debug.assert(r0 < n and B > 0);
    return .{ .r0 = r0, .m = @min(B, n - r0), .act = r0 * h * hd * 2, .pos = r0 * 8, .idx = r0 * kk * 8 };
}

/// The host model of the union kernels (dsv41_shard.cu's mark, scan, compact and renumber; shard_ops.unionCount and
/// friends): the distinct picks below `limit` of rows' idx [rows, ks] (the first kk of each), in ascending position
/// order. `cum` [limit] ends as the inclusive prefix count of the marked positions, so a marked position t is fetched
/// row cum[t] - 1 and the count is cum[limit - 1].
pub const UnionRef = struct {
    pub fn count(idx: []const i64, ks: usize, kk: usize, rows: usize, limit: usize, cum: []i32) usize {
        if (limit == 0) return 0;
        @memset(cum[0..limit], 0);
        for (0..rows) |r| for (0..kk) |j| {
            const v = idx[r * ks + j];
            if (v >= 0 and v < limit) cum[@intCast(v)] = 1;
        };
        var run: i32 = 0;
        for (cum[0..limit]) |*c| {
            run += c.*;
            c.* = run;
        }
        return @intCast(cum[limit - 1]);
    }

    /// The owner: each marked plane row t (codes, scales: the extent's view) into fetched row cum[t] - 1.
    pub fn compact(codes: []const u8, scales: []const u8, cb: usize, sb: usize, cum: []const i32, limit: usize, dcodes: []u8, dscales: []u8) void {
        for (0..limit) |t| {
            const prev: i32 = if (t == 0) 0 else cum[t - 1];
            if (cum[t] == prev) continue;
            const d: usize = @intCast(prev);
            @memcpy(dcodes[d * cb ..][0..cb], codes[t * cb ..][0..cb]);
            @memcpy(dscales[d * sb ..][0..sb], scales[t * sb ..][0..sb]);
        }
    }

    /// The other rank: pick v of row r becomes out[r, j] = cum[v] - 1 (-1 kept, and -1 past kk up to kos entries).
    pub fn renumber(idx: []const i64, ks: usize, kk: usize, rows: usize, cum: []const i32, limit: usize, out: []i64, kos: usize) void {
        for (0..rows) |r| for (0..kos) |j| {
            const v: i64 = if (j < kk) idx[r * ks + j] else -1;
            out[r * kos + j] = if (v >= 0 and v < limit) cum[@intCast(v)] - 1 else -1;
        };
    }
};

/// The remapping the fetch kernel (shard_ops.fetch, dsv41_shard.cu) does, on the host: row r's pick j (idx [rows, ks],
/// -1 none) becomes `j` (relative: the attention adds the row's base) or `r * dk + j` (absolute), -1 kept, and -1 past
/// `kk` up to `kos` entries a row (the prompt chunks' 16-entry padding); with `copy`, the row's picked codes and
/// scales from plane row sbase[r] + idx into fetched row r * dk + j, for rows whose own[r] is set (null: every row).
/// mask: no remap, the pick kept for an own row and -1 for the others (owner decode). Every test of the kernel's
/// order runs this.
pub const FetchRef = struct {
    codes: []const u8,
    scales: []const u8,
    cb: usize,
    sb: usize,
    idx: []const i64,
    ks: usize,
    kk: usize,
    rows: usize,
    sbase: ?[]const i64 = null,
    own: ?[]const i64 = null,
    copy: bool,
    dcodes: []u8,
    dscales: []u8,
    dk: usize,
    out: []i64,
    kos: usize,
    how: How,

    pub const How = enum(i32) { relative = 0, absolute = 1, mask = 2 };

    pub fn run(f: FetchRef) void {
        for (0..f.rows) |r| {
            const b: i64 = if (f.sbase) |sb| sb[r] else 0;
            const mine = if (f.own) |o| o[r] != 0 else true;
            for (0..f.kos) |j| {
                const v: i64 = if (j < f.kk) f.idx[r * f.ks + j] else -1;
                f.out[r * f.kos + j] = switch (f.how) {
                    .mask => if (mine) v else -1,
                    .relative => if (v < 0) -1 else @intCast(j),
                    .absolute => if (v < 0) -1 else @intCast(r * f.dk + j),
                };
                if (!f.copy or f.how == .mask or v < 0 or !mine) continue;
                const src: usize = @intCast(b + v);
                const dst = r * f.dk + j;
                @memcpy(f.dcodes[dst * f.cb ..][0..f.cb], f.codes[src * f.cb ..][0..f.cb]);
                @memcpy(f.dscales[dst * f.sb ..][0..f.sb], f.scales[src * f.sb ..][0..f.sb]);
            }
        }
    }
};

/// What the sparse attention reads for row r (its packed path: codes and scales of comp[cbase[r] + idx[r, j]] in
/// pick order, -1 skipped), appended to `out` as (codes, scales) bytes: the host model of the kernel's compressed
/// reads the byte-exactness tests compare.
pub fn attnReads(gpa: std.mem.Allocator, codes: []const u8, scales: []const u8, cb: usize, sb: usize, cbase: i64, idx: []const i64, out: *std.ArrayList(u8)) !void {
    for (idx) |v| {
        if (v < 0) continue;
        const row: usize = @intCast(cbase + v);
        try out.appendSlice(gpa, codes[row * cb ..][0..cb]);
        try out.appendSlice(gpa, scales[row * sb ..][0..sb]);
    }
}

// -- tests ----------------------------------------------------------------------------------------------------------

const testing = std.testing;

/// This checkpoint's four kv sources (layers 2, 8, 14, 20; ratios 2, 2, 2, 1), every one with index keys.
const ds_sources = [_]Source{
    .{ .ratio = 2, .comp_row = comp_row_bytes, .idx_row = idx_row_bytes },
    .{ .ratio = 2, .comp_row = comp_row_bytes, .idx_row = idx_row_bytes },
    .{ .ratio = 2, .comp_row = comp_row_bytes, .idx_row = idx_row_bytes },
    .{ .ratio = 1, .comp_row = comp_row_bytes, .idx_row = idx_row_bytes },
};

test "packed rows: 288 bytes a latent, 68 an index key (head_dim 512, index_head_dim 128)" {
    try testing.expectEqual(@as(usize, 512 / 2 + 512 / 16), comp_row_bytes);
    try testing.expectEqual(@as(usize, 128 / 2 + 128 / 32), idx_row_bytes);
}

test "capacity: 890 bytes a token a rank replicated, 530 sharded at TP2, 360 saved" {
    // two tokens (a multiple of every ratio): halve for a token
    try testing.expectEqual(@as(usize, 2 * 890), planeBytes(&ds_sources, 2, 2, false));
    try testing.expectEqual(@as(usize, 2 * 530), planeBytes(&ds_sources, 2, 2, true));
    try testing.expectEqual(@as(usize, 2 * 360), planeBytes(&ds_sources, 2, 2, false) - planeBytes(&ds_sources, 2, 2, true));
    // ~180 MiB a rank at 524,288 pool tokens, ~360 MiB at 1,048,576 (planes only)
    const mib = 1 << 20;
    try testing.expectEqual(@as(usize, 180), (planeBytes(&ds_sources, 524288, 2, false) - planeBytes(&ds_sources, 524288, 2, true)) / mib);
    try testing.expectEqual(@as(usize, 360), (planeBytes(&ds_sources, 1 << 20, 2, false) - planeBytes(&ds_sources, 1 << 20, 2, true)) / mib);
    // the 1M-window target: 2,097,152 sharded positions cost ~1,060 MiB a rank, 170 MiB more than 1M replicated
    try testing.expectEqual(@as(usize, 1060), planeBytes(&ds_sources, 2 << 20, 2, true) / mib);
    try testing.expectEqual(@as(usize, 890), planeBytes(&ds_sources, 1 << 20, 2, false) / mib);
    // the same memory holds ~1.68x the tokens
    const ratio = @as(f64, 890) / 530;
    try testing.expect(ratio > 1.67 and ratio < 1.68);
}

test "capacity with pad and trash rows: the planes Caches allocates" {
    const pool: usize = 1 << 20;
    const rep = poolBytes(&ds_sources, pool, 2, false, 0);
    try testing.expectEqual(planeBytes(&ds_sources, pool, 2, false) + 4 * pad_rows * (comp_row_bytes + idx_row_bytes), rep);
    const sh = poolBytes(&ds_sources, pool, 2, true, 48);
    try testing.expectEqual(planeBytes(&ds_sources, pool, 2, true) + 4 * (pad_rows + 48) * comp_row_bytes + 4 * pad_rows * idx_row_bytes, sh);
}

test "plan: unsharded the pool is the window; a separate pool keeps the window" {
    const p = try plan(1 << 20, 1 << 20, 16, 2, false);
    try testing.expectEqual(@as(usize, 1 << 20), p.part);
    try testing.expectEqual(@as(usize, (1 << 20) - 18), p.window);
    try testing.expectEqual(@as(usize, 1 << 20), p.extent);
    const big = try plan(2 << 20, 1 << 20, 16, 2, false);
    try testing.expectEqual(@as(usize, 1 << 20), big.window);
    try testing.expect(!big.shrunk);
    try testing.expectEqual(@as(usize, (1 << 20) + block), big.extent); // window + 18, in whole chunks
}

test "plan: sharded, every part holds the whole window or the pool is refused" {
    // today's 1M pool sharded: each part 512K, short of the 1M window: refused, never silently halved
    try testing.expectError(error.PartTooSmall, plan(1 << 20, 1 << 20, 16, 2, true));
    // 2M positions: a part is 1M, a round's rows and two short of the window: refused
    try testing.expectError(error.PartTooSmall, plan(2 << 20, 1 << 20, 16, 2, true));
    // the recipe's derived pool: two parts of window + one block (2,101,248 at the 1M window)
    const w = try plan((2 << 20) + 2 * block, 1 << 20, 16, 2, true);
    try testing.expectEqual(@as(usize, 1 << 20), w.window);
    try testing.expectEqual(@as(usize, ((1 << 20) + block)), w.part);
    try testing.expect(!w.shrunk);
    try testing.expectEqual(@as(usize, (1 << 20) + block), w.extent);
    // a larger pool (5,218,304: Robert's with the display reserve) keeps the window, an extent stays window-sized
    const big = try plan(5218304, 1 << 20, 16, 2, true);
    try testing.expectEqual(@as(usize, 2609152), big.part);
    try testing.expectEqual(@as(usize, 1 << 20), big.window);
    try testing.expectEqual(@as(usize, (1 << 20) + block), big.extent);
    // a smaller window a part holds is untouched
    const s = try plan(1 << 20, 262144, 16, 2, true);
    try testing.expectEqual(@as(usize, 262144), s.window);
    try testing.expect(!s.shrunk);
}

test "plan: absurdly small parts are refused" {
    try testing.expectError(error.PartTooSmall, plan(200 * 1024, 1 << 20, 16, 2, true)); // a part of ~100K
    try testing.expectError(error.PartTooSmall, plan(400 * 1024, 1 << 20, 16, 2, true)); // a part under a quarter of 1M
    try testing.expectError(error.PoolNotWholeParts, plan((1 << 20) + 2048, 1 << 20, 16, 2, true));
    try testing.expectError(error.EmptyPool, plan(0, 1 << 20, 16, 2, true));
}

test "Mode: TF_DS_KV_SHARD and TF_DS_KV_SHARD_DECODE" {
    try testing.expectEqual(Mode.off, try Mode.parse(null, null));
    try testing.expectEqual(Mode.off, try Mode.parse("0", "owner"));
    try testing.expectEqual(Mode.off, try Mode.parse("", null));
    try testing.expectEqual(Mode.fetch, try Mode.parse("1", null));
    try testing.expectEqual(Mode.fetch, try Mode.parse("1", "fetch"));
    try testing.expectEqual(Mode.owner, try Mode.parse("1", "owner"));
    try testing.expectError(error.BadKvShard, Mode.parse("yes", null));
    try testing.expectError(error.BadKvShardDecode, Mode.parse("1", "both"));
}

test "Shard: TP2 parts of whole blocks" {
    try testing.expectError(error.KvShardNeedsTwoRanks, Shard.init(0, 4, 1 << 20, 16));
    try testing.expectError(error.PoolNotWholeParts, Shard.init(0, 2, 2048, 16));
    try testing.expectError(error.BadRank, Shard.init(2, 2, 1 << 20, 16));
    const s = try Shard.init(1, 2, 1 << 20, 48);
    try testing.expectEqual(@as(usize, 1 << 19), s.part);
    try testing.expectEqual(@as(u32, 0), s.owner(0));
    try testing.expectEqual(@as(u32, 0), s.owner((1 << 19) - 1));
    try testing.expectEqual(@as(u32, 1), s.owner(1 << 19));
    try testing.expect(s.owns(1 << 19) and !s.owns(4096));
    try testing.expectEqual(@as(u32, 0), s.peer());
}

test "address translation: a part's rows from 0 in its owner's plane, pad then trash rows past them" {
    const part: usize = 1 << 19;
    const s0 = try Shard.init(0, 2, 2 * part, 16);
    const s1 = try Shard.init(1, 2, 2 * part, 16);
    for ([_]usize{ 1, 2 }) |r| {
        try testing.expectEqual(part / r + pad_rows + 16, s0.planeRows(r));
        // an extent of part 0 at 8192: rank 0's rows from 8192 / r, rank 1 holds none
        try testing.expectEqual(@as(?usize, 8192 / r), s0.planeBase(8192, r));
        try testing.expectEqual(@as(?usize, null), s1.planeBase(8192, r));
        // an extent of part 1 at part + 8192: rank 1's rows from 8192 / r, as part 0's would be
        try testing.expectEqual(@as(?usize, 8192 / r), s1.planeBase(part + 8192, r));
        try testing.expectEqual(@as(?usize, null), s0.planeBase(part + 8192, r));
        try testing.expectEqual(part / r + pad_rows + 3, s0.trashRow(r, 3));
    }
    // a store target: the pool-wide row moved by the owner's part, the other rank's trash row
    const base = part + 4096;
    const abs = base / 2 + 100; // cbase2 + group 100
    try testing.expectEqual(abs - part / 2, s1.storeRow(abs, base, 2, 5));
    try testing.expectEqual(s0.trashRow(2, 5), s0.storeRow(abs, base, 2, 5));
    // the ratio-2 scratch (an extent's last row) stays inside its part
    const end = base + 16384;
    try testing.expectEqual(end / 2 - 1 - part / 2, s1.storeRow(end / 2 - 1, base, 2, 0));
    // no row of a part reaches past its owner's part rows
    var rng = std.Random.DefaultPrng.init(7);
    for (0..1000) |_| {
        const b = rng.random().uintLessThan(usize, 2 * part / block) * block;
        const r: usize = if (rng.random().boolean()) 1 else 2;
        const g = rng.random().uintLessThan(usize, (part - (b % part)) / r);
        const sh = if (s0.owns(b)) s0 else s1;
        const row = sh.storeRow(b / r + g, b, r, 0);
        try testing.expect(row < part / r);
        try testing.expectEqual(b / r + g, row + sh.start(sh.owner(b)) / r);
    }
}

test "rowGlue: the replicated store targets in the owner's plane, trash rows on the other rank" {
    const part: usize = 1 << 19;
    for ([_]u32{ 0, 1 }) |rank| {
        const s = try Shard.init(rank, 2, 2 * part, 48);
        // an extent of part 1 at part + 8192 .. part + 8192 + 16384, rows at positions 5 (a complete pair) and 6
        const base = part + 8192;
        const end = base + 16384;
        const a = try rowGlue(s, 0, 2, 5, base, end, 1, 512);
        const b = try rowGlue(s, 1, 2, 6, base, end, 1, 512);
        if (rank == 1) {
            try testing.expectEqual(@as(usize, 8192 + 5), a.ltarget1);
            try testing.expectEqual(@as(usize, 8192 / 2 + 2), a.ltarget2); // pos 5: group 2 complete
            try testing.expectEqual(@as(usize, (8192 + 16384) / 2 - 1), b.ltarget2); // pos 6: the scratch row
            try testing.expectEqual(@as(usize, 8192), a.lbase1);
            try testing.expectEqual(@as(usize, 4096), a.lbase2);
            try testing.expect(a.own and b.own);
        } else {
            try testing.expectEqual(s.trashRow(1, 0), a.ltarget1);
            try testing.expectEqual(s.trashRow(2, 1), b.ltarget2);
            try testing.expectEqual(@as(usize, 0), a.lbase1);
            try testing.expect(!a.own and !b.own);
        }
        // the fetch base is the same on both ranks: the owner's slot of the gathered rows
        try testing.expectEqual(fetchBase(1, 2, 1, 512), b.fbase);
        try testing.expectEqual(@as(usize, (2 + 1) * 512), b.fbase);
        // refusals: a crossing extent, a part that is not the base's, a position past the extent, too many rows
        try testing.expectError(error.CrossesPart, rowGlue(s, 0, 1, 5, part - 2048, part + 2048, 0, 512));
        try testing.expectError(error.BadPart, rowGlue(s, 0, 1, 5, base, end, 0, 512));
        try testing.expectError(error.BadRound, rowGlue(s, 0, 1, 16384, base, end, 1, 512));
        try testing.expectError(error.BadRound, rowGlue(s, 48, 49, 5, base, end, 1, 512));
    }
}

test "within: an extent never crosses a part" {
    const s = try Shard.init(0, 2, 1 << 20, 16);
    const part: usize = 1 << 19;
    try testing.expect(s.within(0, part));
    try testing.expect(!s.within(0, part + 1));
    try testing.expect(s.within(part, part));
    try testing.expect(!s.within(part - 2048, 4096));
    try testing.expect(!s.within(1 << 20, 1));
    try testing.expect(s.within(4096, 0));
}

test "extentMove: within a part the owner copies, across parts the owners send and receive" {
    const part: usize = 1 << 19;
    const s0 = try Shard.init(0, 2, 2 * part, 16);
    const s1 = try Shard.init(1, 2, 2 * part, 16);
    // one part (0): rank 0 copies locally, rank 1 does nothing
    try testing.expectEqualDeep(Move{ .local = .{ .from = 0, .to = 4096 } }, try s0.extentMove(0, 4096, 3000));
    try testing.expectEqualDeep(Move.none, try s1.extentMove(0, 4096, 3000));
    // part 1 to part 0: rank 1 sends from its rows, rank 0 receives into its own
    try testing.expectEqualDeep(Move{ .send = .{ .from = 2048, .peer = 0 } }, try s1.extentMove(part + 2048, 4096, 5000));
    try testing.expectEqualDeep(Move{ .recv = .{ .to = 4096, .peer = 1 } }, try s0.extentMove(part + 2048, 4096, 5000));
    // part 0 to part 1
    try testing.expectEqualDeep(Move{ .send = .{ .from = 2048, .peer = 1 } }, try s0.extentMove(2048, part + 4096, 3001));
    try testing.expectEqualDeep(Move{ .recv = .{ .to = 4096, .peer = 0 } }, try s1.extentMove(2048, part + 4096, 3001));
    // nothing to copy, or a crossing refused
    try testing.expectEqualDeep(Move.none, try s0.extentMove(0, part, 0));
    try testing.expectError(error.CrossesPart, s0.extentMove(part - 1000, 4096, 2000));
    try testing.expectError(error.CrossesPart, s0.extentMove(4096, part - 1000, 2000));
    try testing.expectError(error.CrossesPart, s0.extentMove(4096, 2 * part - 1000, 2000));
}

/// Two ranks' planes of one ratio-1 source over a pool of `2 * part` positions, simulated: what a carry leaves.
fn simCarry(gpa: std.mem.Allocator, part: usize, from: usize, to: usize, n: usize) !void {
    const cb = 16;
    var planes: [2][]u8 = undefined;
    var before: [2][]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(from * 31 + to);
    for (0..2) |r| {
        planes[r] = try gpa.alloc(u8, (part + pad_rows + 16) * cb);
        rng.random().bytes(planes[r]);
        before[r] = try gpa.dupe(u8, planes[r]);
    }
    defer for (0..2) |r| {
        gpa.free(planes[r]);
        gpa.free(before[r]);
    };
    // the transport: the sender's bytes in a mailbox, the receiver's call takes them (NCCL matches a pair's calls)
    var box: ?[]const u8 = null;
    var sends: usize = 0;
    var recvs: usize = 0;
    const so = (try Shard.init(0, 2, 2 * part, 16)).owner(from);
    var order = [2]u32{ so, 1 - so };
    for (order[0..]) |r| {
        const s = try Shard.init(r, 2, 2 * part, 16);
        switch (try s.extentMove(from, to, n)) {
            .none => {},
            .local => |l| @memmove(planes[r][l.to * cb ..][0 .. n * cb], planes[r][l.from * cb ..][0 .. n * cb]),
            .send => |x| {
                try testing.expect(box == null);
                box = planes[r][x.from * cb ..][0 .. n * cb];
                sends += 1;
            },
            .recv => |x| {
                const got = box orelse return error.TestUnexpectedResult;
                @memcpy(planes[r][x.to * cb ..][0 .. n * cb], got);
                box = null;
                recvs += 1;
            },
        }
    }
    try testing.expectEqual(sends, recvs);
    // every rank's plane: the source's rows at the destination on its owner, nothing else touched
    const s = try Shard.init(0, 2, 2 * part, 16);
    const do = s.owner(to);
    for (0..2) |r| {
        const exp = try gpa.dupe(u8, before[r]);
        defer gpa.free(exp);
        if (r == do) @memmove(exp[s.local(to) * cb ..][0 .. n * cb], before[so][s.local(from) * cb ..][0 .. n * cb]);
        try testing.expectEqualSlices(u8, exp, planes[r]);
    }
    // the source's rows stay
    try testing.expectEqualSlices(u8, before[so][s.local(from) * cb ..][0 .. n * cb], planes[so][s.local(from) * cb ..][0 .. n * cb]);
}

test "carry: every plane of every rank is the source's rows at the destination, the source kept" {
    const gpa = testing.allocator;
    const part: usize = 16384;
    try simCarry(gpa, part, 2048, part + 4096, 3001); // part 0 to part 1
    try simCarry(gpa, part, part + 2048, 4096, 5000); // part 1 to part 0
    try simCarry(gpa, part, 0, 4096, 3000); // within part 0: a local copy
    try simCarry(gpa, part, part, part + 8192, 6000); // within part 1
    try simCarry(gpa, part, 0, part, 0); // nothing
}

test "blockRows: whole 16-row tiles of picks in the fetch buffer" {
    try testing.expectEqual(@as(usize, 128), blockRows(65536, 512));
    try testing.expectEqual(@as(usize, 16), blockRows(1024, 512)); // at least a tile
    try testing.expectEqual(@as(usize, 64), blockRows(65536, 1000));
    try testing.expectEqual(@as(usize, 65536), blockRows(65536, 0));
}

/// A pool-wide plane (replicated), its two ranks' sharded planes, and random picks: the reads of the fetched copy are
/// the replicated reads, byte for byte, in pick order.
fn fetchCase(gpa: std.mem.Allocator, seed: u64, ratio: usize, rows: usize, kk: usize, padded: bool) !void {
    const cb = 32; // (the codes' and scales' widths do not change the order: small rows keep the test fast)
    const sb = 16;
    const part: usize = 8 * block;
    const pool = 2 * part;
    var rng = std.Random.DefaultPrng.init(seed);
    const R = rng.random();
    const full_rows = pool / ratio + pad_rows;
    const codes = try gpa.alloc(u8, full_rows * cb);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, full_rows * sb);
    defer gpa.free(scales);
    R.bytes(codes);
    R.bytes(scales);
    // each rank's plane: its part's rows of the replicated plane (as an owner's writes leave them), the rest garbage
    var pc: [2][]u8 = undefined;
    var ps: [2][]u8 = undefined;
    for (0..2) |r| {
        const s = try Shard.init(@intCast(r), 2, pool, 16);
        pc[r] = try gpa.alloc(u8, s.planeRows(ratio) * cb);
        ps[r] = try gpa.alloc(u8, s.planeRows(ratio) * sb);
        R.bytes(pc[r]);
        R.bytes(ps[r]);
        const first = s.start(@intCast(r)) / ratio;
        @memcpy(pc[r][0 .. part / ratio * cb], codes[first * cb ..][0 .. part / ratio * cb]);
        @memcpy(ps[r][0 .. part / ratio * sb], scales[first * sb ..][0 .. part / ratio * sb]);
    }
    defer for (0..2) |r| {
        gpa.free(pc[r]);
        gpa.free(ps[r]);
    };
    // the round's rows: each its own extent of 2 blocks in a random part, picks inside it (-1 sprinkled, sorted as
    // the indexer leaves them), the replicated cbase the pool-wide row base
    const kos = if (padded) (kk + 15) / 16 * 16 else kk;
    const idx = try gpa.alloc(i64, rows * kk);
    defer gpa.free(idx);
    const bases = try gpa.alloc(usize, rows);
    defer gpa.free(bases);
    const owners = try gpa.alloc(u32, rows);
    defer gpa.free(owners);
    const s0 = try Shard.init(0, 2, pool, 16);
    for (0..rows) |i| {
        bases[i] = R.uintLessThan(usize, pool / (2 * block)) * 2 * block;
        owners[i] = s0.owner(bases[i]);
        const span = 2 * block / ratio;
        for (0..kk) |j| idx[i * kk + j] = if (R.uintLessThan(u8, 8) == 0) -1 else @intCast(R.uintLessThan(usize, span));
        std.mem.sort(i64, idx[i * kk ..][0..kk], {}, std.sort.asc(i64));
    }
    // the replicated reads
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(gpa);
    var wants = try gpa.alloc(usize, rows + 1);
    defer gpa.free(wants);
    for (0..rows) |i| {
        wants[i] = want.items.len;
        try attnReads(gpa, codes, scales, cb, sb, @intCast(bases[i] / ratio), idx[i * kk ..][0..kk], &want);
    }
    wants[rows] = want.items.len;
    // decode: each rank fills its own rows of [2][rows][kk] (its plane's rows from the local base), the all-gather
    // joins them, both ranks read row i at fetchBase(owner, rows, i, kk) + the remapped pick
    const K = kk;
    const gc = try gpa.alloc(u8, 2 * rows * K * cb);
    defer gpa.free(gc);
    const gs = try gpa.alloc(u8, 2 * rows * K * sb);
    defer gpa.free(gs);
    R.bytes(gc);
    R.bytes(gs);
    var outs: [2][]i64 = undefined;
    for (0..2) |r| {
        const s = try Shard.init(@intCast(r), 2, pool, 16);
        const sbase = try gpa.alloc(i64, rows);
        defer gpa.free(sbase);
        const own = try gpa.alloc(i64, rows);
        defer gpa.free(own);
        for (0..rows) |i| {
            own[i] = @intFromBool(s.owns(bases[i]));
            sbase[i] = if (s.planeBase(bases[i], ratio)) |b| @intCast(b) else 0;
        }
        outs[r] = try gpa.alloc(i64, rows * kos);
        (FetchRef{ .codes = pc[r], .scales = ps[r], .cb = cb, .sb = sb, .idx = idx, .ks = kk, .kk = kk, .rows = rows, .sbase = sbase, .own = own, .copy = true, .dcodes = gc[r * rows * K * cb ..], .dscales = gs[r * rows * K * sb ..], .dk = K, .out = outs[r], .kos = kos, .how = .relative }).run();
    }
    defer for (outs) |o| gpa.free(o);
    try testing.expectEqualSlices(i64, outs[0], outs[1]); // both ranks the same remapped picks
    for (0..rows) |i| {
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(gpa);
        try attnReads(gpa, gc, gs, cb, sb, @intCast(fetchBase(owners[i], rows, i, K)), outs[0][i * kos ..][0..kos], &got);
        try testing.expectEqualSlices(u8, want.items[wants[i]..wants[i + 1]], got.items);
    }
    // a prompt block: the owner of one extent copies its picks densely (absolute), the other rank remaps alone; the
    // remapped picks and the reads of the copy are the replicated ones
    // (every row of a prompt chunk is one stream's: the picks, each inside an extent of 2 blocks, read the first
    // row's extent here)
    const pb = bases[0];
    const po = s0.owner(pb);
    const so = try Shard.init(po, 2, pool, 16);
    const dense_c = try gpa.alloc(u8, rows * kk * cb);
    defer gpa.free(dense_c);
    const dense_s = try gpa.alloc(u8, rows * kk * sb);
    defer gpa.free(dense_s);
    const view = so.planeBase(pb, ratio).?;
    const o_owner = try gpa.alloc(i64, rows * kos);
    defer gpa.free(o_owner);
    const o_other = try gpa.alloc(i64, rows * kos);
    defer gpa.free(o_other);
    (FetchRef{ .codes = pc[po][view * cb ..], .scales = ps[po][view * sb ..], .cb = cb, .sb = sb, .idx = idx, .ks = kk, .kk = kk, .rows = rows, .copy = true, .dcodes = dense_c, .dscales = dense_s, .dk = kk, .out = o_owner, .kos = kos, .how = .absolute }).run();
    (FetchRef{ .codes = &.{}, .scales = &.{}, .cb = cb, .sb = sb, .idx = idx, .ks = kk, .kk = kk, .rows = rows, .copy = false, .dcodes = &.{}, .dscales = &.{}, .dk = kk, .out = o_other, .kos = kos, .how = .absolute }).run();
    try testing.expectEqualSlices(i64, o_owner, o_other);
    for (0..rows) |i| {
        var w: std.ArrayList(u8) = .empty;
        defer w.deinit(gpa);
        try attnReads(gpa, codes, scales, cb, sb, @intCast(pb / ratio), idx[i * kk ..][0..kk], &w);
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(gpa);
        try attnReads(gpa, dense_c, dense_s, cb, sb, 0, o_other[i * kos ..][0..kos], &got);
        try testing.expectEqualSlices(u8, w.items, got.items);
        // the padding past kk is -1
        for (kk..kos) |j| try testing.expectEqual(@as(i64, -1), o_other[i * kos + j]);
    }
    // owner decode's mask: an own row's picks kept, the others' all -1
    for (0..2) |r| {
        const s = try Shard.init(@intCast(r), 2, pool, 16);
        const own = try gpa.alloc(i64, rows);
        defer gpa.free(own);
        for (0..rows) |i| own[i] = @intFromBool(s.owns(bases[i]));
        const m = try gpa.alloc(i64, rows * kos);
        defer gpa.free(m);
        (FetchRef{ .codes = &.{}, .scales = &.{}, .cb = cb, .sb = sb, .idx = idx, .ks = kk, .kk = kk, .rows = rows, .own = own, .copy = false, .dcodes = &.{}, .dscales = &.{}, .dk = kk, .out = m, .kos = kos, .how = .mask }).run();
        for (0..rows) |i| for (0..kos) |j| {
            const want_v: i64 = if (own[i] != 0 and j < kk) idx[i * kk + j] else -1;
            try testing.expectEqual(want_v, m[i * kos + j]);
        };
        // and an own row's reads through the local plane at the local base are the replicated reads
        for (0..rows) |i| if (own[i] != 0) {
            var got: std.ArrayList(u8) = .empty;
            defer got.deinit(gpa);
            try attnReads(gpa, pc[r], ps[r], cb, sb, @intCast(s.planeBase(bases[i], ratio).?), m[i * kos ..][0..kos], &got);
            try testing.expectEqualSlices(u8, want.items[wants[i]..wants[i + 1]], got.items);
        };
    }
}

test "sparse fetch: the fetched copy's reads are the replicated plane's, byte for byte, in pick order" {
    const gpa = testing.allocator;
    var seed: u64 = 1;
    for ([_]usize{ 1, 2 }) |ratio| {
        for ([_]usize{ 1, 3, 16 }) |rows| {
            for ([_]usize{ 1, 70, 512 }) |kk| {
                try fetchCase(gpa, seed, ratio, rows, kk, false);
                try fetchCase(gpa, seed + 1000, ratio, rows, kk, true);
                seed += 1;
            }
        }
    }
}

test "plan: a small part that holds the whole window is served unshrunk" {
    // TF_DS_POOL=131072 with --context 32768: a 65,536-position part serves the full window
    const p = try plan(131072, 32768, 16, 2, true);
    try testing.expectEqual(@as(usize, 65536), p.part);
    try testing.expectEqual(@as(usize, 32768), p.window);
    try testing.expect(!p.shrunk);
    try testing.expectEqual(@as(usize, 32768 + block), p.extent);
    // the same part shrinking a larger window is still refused
    try testing.expectError(error.PartTooSmall, plan(131072, 1 << 20, 16, 2, true));
    // a part with no room at all is refused whatever the window
    try testing.expectError(error.PartTooSmall, plan(4096, 1, 2046, 2, true));
}

test "rowOwner: malformed, crossing and mislabelled extents refused before any arithmetic" {
    const part: usize = 1 << 19;
    const s = try Shard.init(0, 2, 2 * part, 48);
    try testing.expectEqual(@as(u32, 1), try rowOwner(s, part + 4096, part + 8192, -1));
    try testing.expectEqual(@as(u32, 0), try rowOwner(s, 0, 8192, 0));
    try testing.expectError(error.BadRound, rowOwner(s, -2048, 4096, -1));
    try testing.expectError(error.BadRound, rowOwner(s, 8192, 8192, -1));
    try testing.expectError(error.BadRound, rowOwner(s, 8192, 4096, -1));
    try testing.expectError(error.CrossesPart, rowOwner(s, part - 2048, part + 2048, -1));
    try testing.expectError(error.CrossesPart, rowOwner(s, 2 * part, 2 * part + 2048, -1));
    try testing.expectError(error.BadPart, rowOwner(s, 0, 8192, 1));
}

test "glueRows: every field at its SGlue plane, max_rows apart, the replicated targets moved into the owner's part" {
    const part: usize = 1 << 19;
    const stride: usize = 48; // round.max_rows
    const K: usize = 512;
    // three rows: two of one stream in part 1, one in part 0
    const pos = [_]i64{ 7, 8, 100 };
    const bases = [_]i64{ @intCast(part + 8192), @intCast(part + 8192), 16384 };
    const ends = [_]i64{ @intCast(part + 8192 + 16384), @intCast(part + 8192 + 16384), 16384 + 4096 };
    const R = pos.len;
    var out: [sglue_n * stride]i64 = undefined;
    for ([_]u32{ 0, 1 }) |rank| {
        const s = try Shard.init(rank, 2, 2 * part, 48);
        try glueRows(s, &pos, &bases, &ends, null, K, stride, &out);
        for (0..R) |i| {
            const b: usize = @intCast(bases[i]);
            const e: usize = @intCast(ends[i]);
            const p: usize = @intCast(pos[i]);
            const own = s.owns(b);
            const at = struct {
                fn f(o: []const i64, k: SGlue, r: usize) i64 {
                    return o[@as(usize, @backingInt(k)) * 48 + r];
                }
            }.f;
            // round.setGlue's replicated targets: ctarget1 = base + pos; ctarget2 = base/2 + pos/2 (complete) or the
            // extent's last row; the owner's local row adds its part's first row back to them
            const ct1 = b + p;
            const ct2 = if ((p + 1) % 2 == 0) b / 2 + p / 2 else e / 2 - 1;
            const first = s.start(s.owner(b));
            if (own) {
                try testing.expectEqual(@as(i64, @intCast(ct1)), at(&out, .ltarget1, i) + @as(i64, @intCast(first)));
                try testing.expectEqual(@as(i64, @intCast(ct2)), at(&out, .ltarget2, i) + @as(i64, @intCast(first / 2)));
                try testing.expectEqual(@as(i64, @intCast(b - first)), at(&out, .lbase1, i));
                try testing.expectEqual(@as(i64, @intCast((b - first) / 2)), at(&out, .lbase2, i));
                try testing.expectEqual(@as(i64, 1), at(&out, .own, i));
            } else {
                try testing.expectEqual(@as(i64, @intCast(s.trashRow(1, i))), at(&out, .ltarget1, i));
                try testing.expectEqual(@as(i64, @intCast(s.trashRow(2, i))), at(&out, .ltarget2, i));
                try testing.expectEqual(@as(i64, 0), at(&out, .lbase1, i));
                try testing.expectEqual(@as(i64, 0), at(&out, .lbase2, i));
                try testing.expectEqual(@as(i64, 0), at(&out, .own, i));
            }
            try testing.expectEqual(@as(i64, @intCast(fetchBase(s.owner(b), R, i, K))), at(&out, .fbase, i));
        }
        // past the round's rows: zeros (the upload's tail)
        for (0..sglue_n) |f| for (R..stride) |i| try testing.expectEqual(@as(i64, 0), out[f * stride + i]);
        // the lane's parts are checked; refusals: a negative position, too many rows, mismatched lengths
        try glueRows(s, &pos, &bases, &ends, &[_]i64{ 1, 1, 0 }, K, stride, &out);
        try testing.expectError(error.BadPart, glueRows(s, &pos, &bases, &ends, &[_]i64{ 1, 0, 0 }, K, stride, &out));
        try testing.expectError(error.BadRound, glueRows(s, &[_]i64{ 7, -1, 100 }, &bases, &ends, null, K, stride, &out));
        try testing.expectError(error.BadRound, glueRows(s, &pos, bases[0..2], &ends, null, K, stride, &out));
        try testing.expectError(error.WindowTooWide, glueRows(s, &pos, &bases, &ends, null, K, 2, &out));
    }
}

test "decodeSlots: a rank's slot starts at its fetchBase row, the all-gather's rank order" {
    const R: usize = 16;
    const K: usize = 512;
    for ([_]u32{ 0, 1 }) |rank| {
        const sl = decodeSlots(rank, R, K, 256, 32);
        try testing.expectEqual(fetchBase(rank, R, 0, K) * 256, sl.codes);
        try testing.expectEqual(fetchBase(rank, R, 0, K) * 32, sl.scales);
        try testing.expectEqual(R * K * 256, sl.slot_c);
        try testing.expectEqual(R * K * 32, sl.slot_s);
        // the slot's last row is the row before the next rank's first
        try testing.expectEqual(fetchBase(rank, R, R - 1, K) + K, (sl.codes + sl.slot_c) / 256);
    }
    try testing.expectEqual(decodeSlots(0, 16, 512, 256, 32).slot_c, decodeSlots(1, 16, 512, 256, 32).codes);
}

test "xhPlanes and select: xhwo's [group][R][gk] rows of the rows another rank holds, and only those" {
    const gpa = testing.allocator;
    const R: usize = 5;
    const groups: usize = 4;
    const gk: usize = 24;
    const x = xhPlanes(R, groups, gk);
    // element (g, r, k) of xhwo [group][R][gk] fp16 at byte g * plane_bytes + r * row_bytes + 2k
    for (0..groups) |g| for (0..R) |r| for (0..gk) |k| {
        try testing.expectEqual((g * R * gk + r * gk + k) * 2, g * x.plane_bytes + r * x.row_bytes + 2 * k);
    };
    const n = groups * R * gk * 2;
    const src = try gpa.alloc(u8, n);
    defer gpa.free(src);
    const dst = try gpa.alloc(u8, n);
    defer gpa.free(dst);
    var rng = std.Random.DefaultPrng.init(3);
    rng.random().bytes(src);
    rng.random().bytes(dst);
    const before = try gpa.dupe(u8, dst);
    defer gpa.free(before);
    const own = [_]i64{ 1, 0, 0, 1, 0 };
    selectRef(src, dst, &own, 0, x);
    for (0..groups) |g| for (0..R) |r| {
        const at = (g * R + r) * gk * 2;
        const want = if (own[r] == 0) src[at..][0 .. gk * 2] else before[at..][0 .. gk * 2];
        try testing.expectEqualSlices(u8, want, dst[at..][0 .. gk * 2]);
    };
}

test "promptBlock: blocks cover the chunk in order, their offsets the rows'" {
    const n: usize = 2048;
    const h: usize = 32;
    const hd: usize = 512;
    const kk: usize = 512;
    const B = blockRows(65536, kk);
    var r0: usize = 0;
    var blocks: usize = 0;
    while (r0 < n) : (r0 += B) {
        const b = promptBlock(r0, n, B, h, hd, kk);
        try testing.expectEqual(r0, b.r0);
        try testing.expect(b.m > 0 and b.m <= B and b.m * kk <= 65536);
        try testing.expectEqual(r0 * h * hd * 2, b.act);
        try testing.expectEqual(r0 * 8, b.pos);
        try testing.expectEqual(r0 * kk * 8, b.idx);
        try testing.expectEqual(@as(usize, 0), b.act % (16 * h * hd * 2)); // whole 16-row tiles
        blocks += 1;
    }
    try testing.expectEqual(@as(usize, 16), blocks);
    try testing.expectEqual(@as(usize, 1), promptBlock(2047, n, B, h, hd, kk).m);
    // the decision: the visible rows when they fit and are fewer than the picks, else the union
    try testing.expectEqual(PromptFetch.contiguous, promptFetch(60000, 2048, 512, 65536));
    try testing.expectEqual(PromptFetch.@"union", promptFetch(70000, 2048, 512, 65536)); // past the buffer
    try testing.expectEqual(PromptFetch.@"union", promptFetch(30000, 1, 512, 65536)); // more rows than the picks
    try testing.expectEqual(PromptFetch.contiguous, promptFetch(0, 1, 512, 65536));
    try testing.expectEqual(@as(usize, 2048), unionBlockRows(65536, 2048, 512, 65536));
    try testing.expectEqual(B, unionBlockRows(65537, 2048, 512, 65536));
}

/// A prompt chunk of `n` rows from `start` on one ratio-`ratio` extent: the replicated reads of every row against what
/// the other rank reads by each fetch (contiguous; union whole; union in blocks), byte for byte, in pick order.
fn promptCase(gpa: std.mem.Allocator, seed: u64, ratio: usize, start: usize, n: usize, kk: usize, fetch_rows: usize) !void {
    const cb = 32;
    const sb = 16;
    var rng = std.Random.DefaultPrng.init(seed);
    const R = rng.random();
    const nce = (start + n) / ratio;
    const plane_rows = nce + pad_rows + 7; // (the owner's view: rows past the visible ones are never read)
    const codes = try gpa.alloc(u8, plane_rows * cb);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, plane_rows * sb);
    defer gpa.free(scales);
    R.bytes(codes);
    R.bytes(scales);
    // the indexer's picks: row i sees (start + i + 1) / ratio entries, -1 where it has fewer than kk (or sprinkled)
    const idx = try gpa.alloc(i64, n * kk);
    defer gpa.free(idx);
    for (0..n) |i| {
        const vis = (start + i + 1) / ratio;
        for (0..kk) |j| idx[i * kk + j] = if (vis == 0 or R.uintLessThan(u8, 9) == 0) -1 else @intCast(R.uintLessThan(usize, vis));
    }
    const kos = (kk + 15) / 16 * 16;
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(gpa);
    const wants = try gpa.alloc(usize, n + 1);
    defer gpa.free(wants);
    for (0..n) |i| {
        wants[i] = want.items.len;
        try attnReads(gpa, codes, scales, cb, sb, 0, idx[i * kk ..][0..kk], &want);
    }
    wants[n] = want.items.len;
    const fc = try gpa.alloc(u8, fetch_rows * cb);
    defer gpa.free(fc);
    const fs = try gpa.alloc(u8, fetch_rows * sb);
    defer gpa.free(fs);
    // contiguous: rows [0, nce) at the same rows, the picks unchanged
    if (promptFetch(nce, n, kk, fetch_rows) == .contiguous) {
        R.bytes(fc);
        R.bytes(fs);
        @memcpy(fc[0 .. nce * cb], codes[0 .. nce * cb]);
        @memcpy(fs[0 .. nce * sb], scales[0 .. nce * sb]);
        for (0..n) |i| {
            var got: std.ArrayList(u8) = .empty;
            defer got.deinit(gpa);
            try attnReads(gpa, fc, fs, cb, sb, 0, idx[i * kk ..][0..kk], &got);
            try testing.expectEqualSlices(u8, want.items[wants[i]..wants[i + 1]], got.items);
        }
    }
    // union: whole when it fits, else per block; both ranks count the same, the owner compacts, the other renumbers
    const cum = try gpa.alloc(i32, @max(nce, 1));
    defer gpa.free(cum);
    const out = try gpa.alloc(i64, n * kos);
    defer gpa.free(out);
    const total = UnionRef.count(idx, kk, kk, n, nce, cum);
    var distinct = std.AutoHashMap(i64, void).init(gpa);
    defer distinct.deinit();
    for (idx) |v| if (v >= 0) try distinct.put(v, {});
    try testing.expectEqual(distinct.count(), total);
    const B = unionBlockRows(total, n, kk, fetch_rows);
    var r0: usize = 0;
    var sent: usize = 0;
    while (r0 < n) : (r0 += B) {
        const b = promptBlock(r0, n, B, 1, 1, kk);
        const bidx = idx[r0 * kk ..][0 .. b.m * kk];
        const c = UnionRef.count(bidx, kk, kk, b.m, nce, cum);
        try testing.expect(c <= fetch_rows and c <= b.m * kk);
        R.bytes(fc);
        R.bytes(fs);
        UnionRef.compact(codes, scales, cb, sb, cum, nce, fc, fs);
        UnionRef.renumber(bidx, kk, kk, b.m, cum, nce, out, kos);
        sent += c;
        for (0..b.m) |i| {
            var got: std.ArrayList(u8) = .empty;
            defer got.deinit(gpa);
            try attnReads(gpa, fc, fs, cb, sb, 0, out[i * kos ..][0..kos], &got);
            try testing.expectEqualSlices(u8, want.items[wants[r0 + i]..wants[r0 + i + 1]], got.items);
            for (kk..kos) |j| try testing.expectEqual(@as(i64, -1), out[i * kos + j]);
            // renumbered picks keep the replicated order: ascending positions stay ascending
            for (1..kk) |j| {
                const p = idx[(r0 + i) * kk + j - 1];
                const q = idx[(r0 + i) * kk + j];
                if (p >= 0 and q >= 0) try testing.expectEqual(std.math.order(p, q), std.math.order(out[i * kos + j - 1], out[i * kos + j]));
            }
        }
    }
    if (B == n) try testing.expectEqual(total, sent);
    try testing.expect(sent <= n * kk);
}

test "prompt fetch: contiguous and union copies read the replicated bytes, deduplicated, in pick order" {
    const gpa = testing.allocator;
    var seed: u64 = 11;
    for ([_]usize{ 1, 2 }) |ratio| {
        // (start, rows, picks, fetch rows): a short prompt (contiguous), a long one (union whole), a union past the
        // buffer (blocks), a one-row chunk, a ring-mode chunk, odd pick counts
        const cases = [_][4]usize{
            .{ 0, 300, 64, 4096 },
            .{ 6000, 64, 70, 4096 },
            .{ 6000, 64, 70, 2048 },
            .{ 50000, 200, 512, 8192 },
            .{ 20000, 1, 512, 4096 },
            .{ 9001, 13, 33, 4096 },
            .{ 1, 3, 16, 4096 },
        };
        for (cases) |cs| {
            try promptCase(gpa, seed, ratio, cs[0], cs[1], cs[2], cs[3]);
            seed += 1;
        }
    }
}

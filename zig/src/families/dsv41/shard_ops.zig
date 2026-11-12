//! Port of wojo's GLM 0118 (PR #152): TF_DS_KV_SHARD's device side beyond the kit, the byte movers in
//! dsv41_shard.cu (fetch: a round's or prompt block's picked compressed rows packed and renumbered; select: owner
//! decode's returned rows put in place; the union: a prompt chunk's distinct picks marked, counted, packed and
//! renumbered), and what one rank's sharded attention carries (KvShard: the geometry, the mode, the kernels). Every
//! launch moves bytes or integer ids unchanged; the arithmetic stays the kit's.
const std = @import("std");
const cuda = @import("cuda");
const shard = @import("shard.zig");
const shard_ring = @import("shard_ring.zig");
const rdma = @import("rdma.zig");

pub const Kernels = struct {
    module: cuda.Module,
    fetch_fn: cuda.Function,
    select_fn: cuda.Function,
    mark_fn: cuda.Function,
    sums_fn: cuda.Function,
    offsets_fn: cuda.Function,
    scan_fn: cuda.Function,
    compact_fn: cuda.Function,
    renumber_fn: cuda.Function,
    /// The decode exchanges on the RDMA ring (shard_ring.zig): its sized stage and collect. An image built before
    /// them has neither: that rank then asks for NCCL (round.Round.exchangeSettings), which the ranks' settings
    /// gather makes every rank's or refuses (KvShardSettingsMismatch) when another rank has them.
    ring_stage_fn: ?cuda.Function = null,
    ring_collect_fn: ?cuda.Function = null,

    /// dsv41_shard.cu's image (cuda.kernels.dsv41_shard, or a fatbin's bytes).
    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        if (image.len == 0) return error.NoKernelImages;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .module = m,
            .fetch_fn = try m.function("tf_ds_kv_fetch_kernel"),
            .select_fn = try m.function("tf_ds_select_rows_kernel"),
            .mark_fn = try m.function("tf_ds_union_mark_kernel"),
            .sums_fn = try m.function("tf_ds_union_sums_kernel"),
            .offsets_fn = try m.function("tf_ds_union_offsets_kernel"),
            .scan_fn = try m.function("tf_ds_union_scan_kernel"),
            .compact_fn = try m.function("tf_ds_union_compact_kernel"),
            .renumber_fn = try m.function("tf_ds_union_renumber_kernel"),
            .ring_stage_fn = m.function("tf_ds_ring_stage_rows") catch null,
            .ring_collect_fn = m.function("tf_ds_ring_collect_rows") catch null,
        };
    }

    /// The sized ring transfer's kernel pair, as rdma.Ring.sendRows / recvRows take it.
    pub fn ringRows(k: *const Kernels) !rdma.Ring.RowKernels {
        return .{ .stage = k.ring_stage_fn orelse return error.NoKvShard, .collect = k.ring_collect_fn orelse return error.NoKvShard };
    }

    pub fn hasRingRows(k: *const Kernels) bool {
        return k.ring_stage_fn != null and k.ring_collect_fn != null;
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

/// One fetch launch (shard.FetchRef is its host model, and the tests' reference): picks idx [rows, ks] (the first kk
/// of a row), each row's plane base `sbase` (int64 [rows], 0: none, every row 0) and holding flag `own` (int64 [rows],
/// 0: none, every row held), the fetched rows' stride dk picks a row, the remapped picks out [rows, kos].
pub const Fetch = struct {
    codes: u64 = 0,
    scales: u64 = 0,
    cb: usize,
    sb: usize,
    idx: u64,
    ks: usize,
    kk: usize,
    rows: usize,
    sbase: u64 = 0,
    own: u64 = 0,
    copy: bool,
    dcodes: u64 = 0,
    dscales: u64 = 0,
    dk: usize,
    out: u64,
    kos: usize,
    how: shard.FetchRef.How,
};

pub fn fetch(k: *const Kernels, s: cuda.Stream, f: Fetch) !void {
    if (f.rows == 0 or f.kos == 0) return;
    if (f.cb % 16 != 0 or f.sb % 16 != 0 or f.kk > f.kos) return error.BadFetch;
    if (f.copy and (f.codes == 0 or f.dcodes == 0 or (f.codes | f.scales | f.dcodes | f.dscales) % 16 != 0)) return error.BadFetch;
    var a: cuda.Args = .{};
    a.add(f.codes);
    a.add(f.scales);
    a.add(@as(i64, @intCast(f.cb)));
    a.add(@as(i64, @intCast(f.sb)));
    a.add(f.idx);
    a.add(@as(i64, @intCast(f.ks)));
    a.add(@as(c_int, @intCast(f.kk)));
    a.add(f.sbase);
    a.add(f.own);
    a.add(@as(c_int, @intFromBool(f.copy)));
    a.add(f.dcodes);
    a.add(f.dscales);
    a.add(@as(i64, @intCast(f.dk)));
    a.add(f.out);
    a.add(@as(i64, @intCast(f.kos)));
    a.add(@as(c_int, @backingInt(f.how)));
    try cuda.launch.launch(k.fetch_fn, .{ .grid = .{ .x = @intCast(f.rows), .y = @intCast((f.kos + 15) / 16) }, .block = .{ .x = 256 } }, s, &a);
}

/// dst row r of each of `planes` planes = src row r where sel[r] == want (int64 [rows]); rows of `bytes` bytes,
/// planes `plane_bytes` apart; every size and pointer whole 16-byte words.
pub fn select(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, sel: u64, want: i64, rows: usize, bytes: usize, planes: usize, plane_bytes: usize) !void {
    if (rows == 0 or planes == 0) return;
    if ((src | dst | bytes | plane_bytes) % 16 != 0) return error.BadSelect;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(sel);
    a.add(want);
    a.add(@as(i64, @intCast(bytes / 16)));
    a.add(@as(i64, @intCast(plane_bytes / 16)));
    try cuda.launch.launch(k.select_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(planes) }, .block = .{ .x = 256 } }, s, &a);
}

/// The union scan's segments (dsv41_shard.cu's TF_DS_SCAN_SEGS): `sums` holds this many int32 and the count after them.
pub const scan_segs: usize = 1024;

/// The prompt fetch's union of picks idx [rows, ks] (the first kk a row) below `limit`, as shard.UnionRef counts it:
/// `cum` (int32 [limit], zeroed by the caller in the stream's order) marked and scanned in place to the inclusive
/// prefix counts, the count left in `sums` (int32 [scan_segs + 1]) at its last entry for the host to read.
pub fn unionCount(k: *const Kernels, s: cuda.Stream, idx: u64, ks: usize, kk: usize, rows: usize, cum: u64, limit: usize, sums: u64) !void {
    if (limit == 0) return error.BadFetch;
    if (rows > 0 and kk > 0) {
        var a: cuda.Args = .{};
        a.add(idx);
        a.add(@as(i64, @intCast(ks)));
        a.add(@as(c_int, @intCast(kk)));
        a.add(cum);
        a.add(@as(i64, @intCast(limit)));
        try cuda.launch.launch(k.mark_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((kk + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }
    var b: cuda.Args = .{};
    b.add(cum);
    b.add(@as(i64, @intCast(limit)));
    b.add(sums);
    try cuda.launch.launch(k.sums_fn, .{ .grid = .{ .x = scan_segs / 256 }, .block = .{ .x = 256 } }, s, &b);
    var o: cuda.Args = .{};
    o.add(sums);
    try cuda.launch.launch(k.offsets_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, s, &o);
    try cuda.launch.launch(k.scan_fn, .{ .grid = .{ .x = scan_segs / 256 }, .block = .{ .x = 256 } }, s, &b);
}

/// The owner's copy of the union (shard.UnionRef.compact): each marked plane row t < limit of the extent's view
/// (codes, scales) into fetched row cum[t] - 1 of dcodes, dscales.
pub fn unionCompact(k: *const Kernels, s: cuda.Stream, codes: u64, scales: u64, cb: usize, sb: usize, cum: u64, limit: usize, dcodes: u64, dscales: u64) !void {
    if (limit == 0) return;
    if (cb % 16 != 0 or sb % 16 != 0 or (codes | scales | dcodes | dscales) % 16 != 0 or codes == 0 or dcodes == 0) return error.BadFetch;
    var a: cuda.Args = .{};
    a.add(codes);
    a.add(scales);
    a.add(@as(i64, @intCast(cb)));
    a.add(@as(i64, @intCast(sb)));
    a.add(cum);
    a.add(@as(i64, @intCast(limit)));
    a.add(dcodes);
    a.add(dscales);
    try cuda.launch.launch(k.compact_fn, .{ .grid = .{ .x = @intCast((limit + 15) / 16) }, .block = .{ .x = 256 } }, s, &a);
}

/// The other rank's picks renumbered to the union's rows (shard.UnionRef.renumber): out [rows, kos].
pub fn unionRenumber(k: *const Kernels, s: cuda.Stream, idx: u64, ks: usize, kk: usize, rows: usize, cum: u64, limit: usize, out: u64, kos: usize) !void {
    if (rows == 0 or kos == 0) return;
    if (kk > kos) return error.BadFetch;
    var a: cuda.Args = .{};
    a.add(idx);
    a.add(@as(i64, @intCast(ks)));
    a.add(@as(c_int, @intCast(kk)));
    a.add(cum);
    a.add(@as(i64, @intCast(limit)));
    a.add(out);
    a.add(@as(i64, @intCast(kos)));
    try cuda.launch.launch(k.renumber_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((kos + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// An exchange's segments in the ring kernels' form (16-byte words; shard_ring.Xfer.check has refused anything else).
pub fn rowSegs(x: shard_ring.Xfer) rdma.Ring.RowSegs {
    var o: rdma.Ring.RowSegs = .{};
    for (x.slice(), 0..) |g, i| {
        o.src[i] = g.src;
        o.dst[i] = g.dst;
        o.src16[i] = @intCast(g.src_stride / 16);
        o.dst16[i] = @intCast(g.dst_stride / 16);
        o.row16[i] = @intCast(g.row_bytes / 16);
    }
    return o;
}

/// One exchange of a decode round on the RDMA ring: this rank's listed rows sent (`list`: the device address of its
/// send list, shard_ring.Lists.offset(.., .send) into Round.rlist), or the peer's received at the rows of its receive
/// list. `cap` the round's rows (the slot and grid bound); `probing` false keeps an early send out of the ring probes.
pub fn ringSend(k: *const Kernels, ring: *rdma.Ring, s: cuda.Stream, x: shard_ring.Xfer, list: u64, cap: usize, probing: bool) !void {
    try x.check(cap, ring.s.max_bytes);
    try ring.sendRows(try k.ringRows(), s, rowSegs(x), @intCast(x.n), list, @intCast(cap), probing);
}

pub fn ringRecv(k: *const Kernels, ring: *rdma.Ring, s: cuda.Stream, x: shard_ring.Xfer, list: u64, cap: usize, from: u32, probing: bool) !void {
    try x.check(cap, ring.s.max_bytes);
    try ring.recvRows(try k.ringRows(), s, rowSegs(x), @intCast(x.n), list, @intCast(cap), from, probing);
}

/// What one rank's sharded attention carries (prompt.Engine.kv): the parts, the decode mode and the kernels.
pub const KvShard = struct {
    sh: shard.Shard,
    mode: shard.Mode,
    k: Kernels,
    // the decode exchanges' transport and auto's cost model (round.Round.useShard: agreed by every rank before the
    // first round): the RDMA ring when the engine has one and TF_DS_KV_SHARD_RING is not 0, else NCCL
    use_ring: bool = false,
    cost: shard_ring.Cost = .{},
    // owner decode's two 32-head passes on two streams on the ring path (Round.useShard: TF_DS_KV_SHARD_OVERLAP, not
    // "0"; per rank, it changes no ring order and no byte)
    overlap: bool = false,
    // counters (the health/metrics a status line reads; round 2 exports them): prompt fetches (a layer's) and their
    // rows and bytes, rounds by fetch and by owner decode (each of a round's compressed layers fetches or exchanges once;
    // counted when the round is issued, graphs or not), carried extents (copyExtent across parts) and their bytes
    stats: Stats = .{},
    // (exported on /health and /metrics with the parts' rows: lanes.zig kvPartsFn, core lanes.kv_parts)

    pub const Stats = struct {
        prompt_fetches: u64 = 0,
        prompt_rows: u64 = 0, // rows sent (the visible rows, or the union's)
        prompt_picks: u64 = 0, // the picks they served (rows x picks: what every-pick fetching would have sent)
        prompt_bytes: u64 = 0,
        prompt_syncs: u64 = 0, // host reads of a union's count
        fetch_rounds: u64 = 0,
        owner_rounds: u64 = 0,
        carries: u64 = 0,
        carry_bytes: u64 = 0,
        // the decode exchanges (shard_ring.zig): rounds whose exchanges rode the RDMA ring (the rest NCCL), auto's
        // choices, and the rows this rank sent a compressed layer on the ring, summed over the rounds (its send lists:
        // fetch, the rows only it holds; owner decode, the queries of the rows only the peer holds and the outputs of
        // the rows only it holds)
        ring_rounds: u64 = 0,
        auto_fetch: u64 = 0,
        auto_owner: u64 = 0,
        sent_rows: u64 = 0,
    };
};

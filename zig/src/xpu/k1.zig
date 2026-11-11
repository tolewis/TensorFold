//! K1 (XPU_K1): split-K decode attention over int8 pool pages on the XMX for the 27B at TP 2 (lanes/qwen27 brief
//! p22-k1). The kernels are attn_k1_decode in zig/kernels/xpu/qwen27/k1.cl; this file reads the flag, plans the
//! launches (splits and row groups) and loads the kernels. Default off: the live path keeps attn_decode.
const std = @import("std");
const ze = @import("ze");
const dev = @import("dev.zig");

/// Keys a work-group's tile (4 pages), k1.cl K1_TILE.
pub const TILE: u32 = 256;
/// Rows of one stream a work-group holds (its DPAS columns: 6 query heads x rows), k1.cl K1_ROWS.
pub const ROWS: u32 = 8;
/// The 16-row single-group variant (k1.cl attn_k1_decode16): 96 columns and 128-key tiles, so a 16-row verify
/// window reads each K/V block once instead of twice (lanes/qwen27 handoff p22-k1, risk 1).
pub const ROWS16: u32 = 16;
pub const TILE16: u32 = 128;
/// attn_k1_k16's tile (k1.cl K16_T, lane p4-kernel).
pub const TILE_K16: u32 = 128;
/// Xe cores of the B70 (Arc Pro B70, BMG-G31).
pub const XE_CORES: u32 = 32;

/// XPU_K1=1: the 27B decode path uses K1 where a slot has a page table (lane p21-pool). Default off.
pub var on = false;
/// XPU_K1_WAVE: work-groups a core (4).
pub var wave: u32 = 4;
/// XPU_K1_R16: 16-row groups for a 16-row window (1). XPU_K1_R16=0: 8-row groups, as lane p22-k1.
pub var r16 = true;
/// XPU_K1_R16=2 (lane p4-kernel): the lone tree verify's prefix keys as one group of up to 16 rows through
/// attn_k1_k16 (k1.cl), which reads and converts each K/V row once for the 16 rows. Chain and shared windows keep
/// 8-row groups, as XPU_K1_R16=0. The values differ from the 8-row groups by f16 and fp32 rounding (k1.cl).
pub var k16t = false;
/// Set when the k1 image carries attn_k1_decode16 (k1.load); groups() only picks it when it does.
pub var have16 = false;
/// XPU_K1_TREE: the single-stream tree verify's prefix goes through K1, its draft keys through attn_k1_draft
/// (lane p22-k1tree). Default off: the tree path keeps the scalar attn_decode + attn_merge.
pub var tree = false;
/// XPU_MULTI_TREE: a shared round with tree windows runs K1: the prefix keys through the decode kernels with
/// per-row POS, the rows' draft keys (their ancestors and themselves) through attn_k1_draft_mt, one
/// attn_merge_span join (lane p33-multitree, design/multitree-k1.md). Lane p39-k1cover: with XPU_K1_TREE=1 it is
/// on unless XPU_MULTI_TREE=0, so a shared round with tree windows (2 to 5 streams) runs K1 like the lone tree
/// and the chain rounds do. XPU_MULTI_TREE=1 turns it on without XPU_K1_TREE. Off, the tree rows keep the scalar
/// attn_decode + attn_merge.
pub var multi_tree = false;

pub fn readEnv() void {
    if (std.c.getenv("XPU_K1")) |v| on = v[0] != '0';
    if (std.c.getenv("XPU_K1_WAVE")) |v| wave = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 4;
    if (std.c.getenv("XPU_K1_R16")) |v| {
        k16t = v[0] == '2';
        r16 = v[0] != '0' and !k16t;
    }
    if (std.c.getenv("XPU_K1_TREE")) |v| tree = v[0] != '0';
    multi_tree = tree; // the shared tree round follows the lone tree unless XPU_MULTI_TREE says
    if (std.c.getenv("XPU_MULTI_TREE")) |v| multi_tree = v[0] != '0';
}

/// What a forward's attention layer sees: the flags, the round and the kernels in the image.
pub const Cover = struct {
    on: bool,
    tree: bool,
    multi_tree: bool,
    i8: bool, // int8 K/V (the pool)
    pool: bool, // the layer has a page table
    round_tree: bool, // Mode.tree: some window has a non-chain parents array
    shared: bool, // Mode.multi: several streams' windows in one forward
    has_k1: bool,
    has_draft: bool,
    has_draft_mt: bool,
};

pub const Path = enum { scalar, chain, tree, multi_tree };

/// The attention path of a decode-width forward (at most 16 rows). forward.zig takes this branch and no other.
/// chain: attn_k1 (a shared round of chain windows, or one stream); tree: a lone stream's tree verify (XPU_K1_TREE);
/// multi_tree: tree windows of a shared round (XPU_MULTI_TREE); scalar: attn_decode + attn_merge.
pub fn attnPath(c: Cover) Path {
    if (c.tree and c.round_tree and !c.shared and c.i8 and c.pool and c.has_k1 and c.has_draft) return .tree;
    if (c.multi_tree and c.shared and c.round_tree and c.i8 and c.pool and c.has_k1 and c.has_draft_mt) return .multi_tree;
    if (c.on and !c.round_tree and c.pool and c.has_k1) return .chain;
    return .scalar;
}

/// The open line: which rounds K1 serves with these flags (a pool and the K1 image assumed).
pub fn coverageLine(buf: []u8, k1_on: bool, k1_tree: bool, k1_mt: bool) []const u8 {
    const c: Cover = .{ .on = k1_on, .tree = k1_tree, .multi_tree = k1_mt, .i8 = true, .pool = true, .round_tree = false, .shared = false, .has_k1 = true, .has_draft = true, .has_draft_mt = true };
    const chain = attnPath(c);
    const one = attnPath(.{ .on = k1_on, .tree = k1_tree, .multi_tree = k1_mt, .i8 = true, .pool = true, .round_tree = true, .shared = false, .has_k1 = true, .has_draft = true, .has_draft_mt = true });
    const many = attnPath(.{ .on = k1_on, .tree = k1_tree, .multi_tree = k1_mt, .i8 = true, .pool = true, .round_tree = true, .shared = true, .has_k1 = true, .has_draft = true, .has_draft_mt = true });
    return std.fmt.bufPrint(buf, "K1 coverage: chain windows {s}, lone tree {s}, shared tree windows {s} (streams 1 to 6, up to 16 rows a round)", .{
        if (chain == .chain) "K1" else "scalar",
        if (one == .tree) "K1" else "scalar",
        if (many == .multi_tree) "K1" else "scalar",
    }) catch buf[0..0];
}

/// The tree verify's MT table and POS: the slot's own kc, vc and page table, every row's last prefix key p0 - 1.
/// The host buffers must stay alive until the command list drains (the queued copies read them).
/// One table a layer: the layers' kc, vc and page table differ, and a queued copy reads its host buffer when the
/// list runs, which at TP 1 (no per-layer host wait) is after the host has filled the later layers' tables.
pub var mt_host: [64][16 * 8]u64 = undefined;
pub var pos_host: [16]i32 = undefined;
/// The merge's POS: p0 + span - 1, so the draft's slice NS joins (used = NS + 1).
pub var posm_host: [16]i32 = undefined;

pub const Kernels = struct {
    decode: ze.H = null, // attn_k1_decode (k1.cl): 8-row groups
    decode16: ze.H = null, // attn_k1_decode16 (k1.cl): 16-row groups, 128-key tiles
    k16: ze.H = null, // attn_k1_k16 (k1.cl): the lone tree verify's 16-row prefix pass (XPU_K1_R16=2)
    draft: ze.H = null, // attn_k1_draft (k1.cl): the tree verify's draft keys into slice NS
    draft_mt: ze.H = null, // attn_k1_draft_mt (k1.cl): the shared round's tree rows' draft keys (XPU_MULTI_TREE)
    span: ze.H = null, // attn_decode_span (decode.cl): the scalar page-table kernel K1 is checked against
    merge: ze.H = null, // attn_merge_span (decode.cl): the merge of natural-base slice partials (the scalar span kernel)
    merge2: ze.H = null, // attn_merge_span2 (decode.cl): the merge of K1's log2-base partials
    dec: ze.H = null, // attn_decode (decode.cl): the scalar table kernel, the tree bench's reference
    tmerge: ze.H = null, // attn_merge (decode.cl): the scalar tree merge
};

/// Loads K1's kernels: the decode kernels from the k1 image, the scalar reference and the merge from decode.
pub fn load(d: *dev.Dev, md: ze.H, mk1: ze.H) !Kernels {
    const k = Kernels{
        .decode = try d.kernel(mk1, "attn_k1_decode", 256),
        .decode16 = try d.kernel(mk1, "attn_k1_decode16", 256),
        .k16 = try d.kernel(mk1, "attn_k1_k16", 256),
        .draft = try d.kernel(mk1, "attn_k1_draft", 16),
        .draft_mt = try d.kernel(mk1, "attn_k1_draft_mt", 16),
        .span = try d.kernel(md, "attn_decode_span", 16),
        .merge = try d.kernel(md, "attn_merge_span", 256),
        .merge2 = try d.kernel(md, "attn_merge_span2", 256),
        .dec = try d.kernel(md, "attn_decode", 16),
        .tmerge = try d.kernel(md, "attn_merge", 16),
    };
    try d.g.indirect(k.decode);
    try d.g.indirect(k.decode16);
    try d.g.indirect(k.k16);
    try d.g.indirect(k.span);
    have16 = k.decode16 != null;
    return k;
}

/// Splits and span of a (stream, K/V head, row group) over `keys` keys: `groups` groups a launch, so
/// wave x cores / (kv_heads x groups) splits fill one wave, in whole tiles.
pub const Plan = struct { span: i32, ns: u32, wgs: u32 };

pub fn plan(keys: u32, groups: u32, kv_heads: u32, wave_: u32) Plan {
    return planTile(keys, groups, kv_heads, wave_, TILE);
}

pub fn planTile(keys: u32, groups: u32, kv_heads: u32, wave_: u32, tile: u32) Plan {
    const tiles = @max(1, (keys + tile - 1) / tile);
    const want = @max(1, @min(wave_ * XE_CORES / (kv_heads * @max(groups, 1)), tiles));
    const span: i32 = @intCast((tiles + want - 1) / want * tile);
    const sp: u32 = @intCast(span);
    const ns = @max(1, (keys + sp - 1) / sp);
    return .{ .span = span, .ns = ns, .wgs = ns * kv_heads * groups };
}

/// The row groups of windows at rows wf[i] of wn[i] rows, ROWS rows a group, into `host` (2 ints a group).
/// Returns the group count (the launch's z).
pub fn rowGroups(wf: []const u32, wn: []const u32, host: []i32) u32 {
    var n: u32 = 0;
    for (wf, wn) |f, w| {
        var j: u32 = 0;
        while (j < w) : (j += ROWS) {
            host[2 * n] = @intCast(f + j);
            host[2 * n + 1] = @intCast(@min(ROWS, w - j));
            n += 1;
        }
    }
    return n;
}

/// The row groups of windows given as pairs (first row, rows), as verifyMulti's win_host holds them.
pub fn groupsWin(win: []const i32, host: []i32) u32 {
    return roundGroups(win, 0, host).n;
}

/// A round's row groups: `k16` when a window has more than ROWS rows and XPU_K1_R16 is on (one group a
/// window, 16 rows), else ROWS rows a group. `n` is the launch's z (the group count).
pub const Round = struct { k16: bool, rows: u32, n: u32 };

pub fn roundGroups(win: []const i32, t: u32, host: []i32) Round {
    var maxr: u32 = t;
    if (win.len > 0) {
        maxr = 0;
        var i: usize = 0;
        while (i + 1 < win.len) : (i += 2) maxr = @max(maxr, @as(u32, @intCast(win[i + 1])));
    }
    const k16 = r16 and have16 and maxr > ROWS;
    const per: u32 = if (k16) ROWS16 else ROWS;
    var n: u32 = 0;
    if (win.len > 0) {
        var i: usize = 0;
        while (i + 1 < win.len) : (i += 2) {
            const f: u32 = @intCast(win[i]);
            const w: u32 = @intCast(win[i + 1]);
            var j: u32 = 0;
            while (j < w) : (j += per) {
                host[2 * n] = @intCast(f + j);
                host[2 * n + 1] = @intCast(@min(per, w - j));
                n += 1;
            }
        }
    } else {
        var j: u32 = 0;
        while (j < t) : (j += per) {
            host[2 * n] = @intCast(j);
            host[2 * n + 1] = @intCast(@min(per, t - j));
            n += 1;
        }
    }
    return .{ .k16 = k16, .rows = per, .n = n };
}

/// The merge's per-row used for a shared tree round: a tree row (kkind 1) reads all ns + 1 slices, so its kposm
/// is ns * span + span - 1; its empty prefix slices hold PM = -inf (k1_body writes them). A chain row reads its
/// prefix slices only, so its kposm is its kpos, and the draft's slice NS is skipped.
pub fn mergePos(kkind: []const u8, kpos: []const i32, ns: u32, span: i32, out: []i32) void {
    const all: i32 = @as(i32, @intCast(ns)) * span + span - 1;
    for (out, kkind, kpos) |*o, k, kp| o.* = if (k != 0) all else kp;
}

/// Queues the groups of `wf`/`wn` into `grp`: the host buffer must stay alive until the list drains.
pub fn queueGroups(d: *dev.Dev, grp: *anyopaque, host: []i32, n: u32) !void {
    try @import("dev.zig").hostCopy(d, grp, host.ptr, @as(usize, 2 * n) * 4);
}

/// Device bytes a round reads through K1: the K and V pages of `keys` keys for every stream, K/V head and row
/// group (2 x (256 + 4) bytes a key a head). One attention layer.
pub fn bytes(keys: u32, groups: u32, kv_heads: u32) u64 {
    return @as(u64, keys) * groups * kv_heads * 2 * (256 + 4);
}

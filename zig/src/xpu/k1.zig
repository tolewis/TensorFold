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
/// Xe cores of the B70 (Arc Pro B70, BMG-G31).
pub const XE_CORES: u32 = 32;

/// XPU_K1=1: the 27B decode path uses K1 where a slot has a page table (lane p21-pool). Default off.
pub var on = false;
/// XPU_K1_WAVE: work-groups a core (4).
pub var wave: u32 = 4;
/// XPU_K1_R16: 16-row groups for a 16-row window (1). XPU_K1_R16=0: 8-row groups, as lane p22-k1.
pub var r16 = true;
/// Set when the k1 image carries attn_k1_decode16 (k1.load); groups() only picks it when it does.
pub var have16 = false;

pub fn readEnv() void {
    if (std.c.getenv("XPU_K1")) |v| on = v[0] != '0';
    if (std.c.getenv("XPU_K1_WAVE")) |v| wave = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 4;
    if (std.c.getenv("XPU_K1_R16")) |v| r16 = v[0] != '0';
}

pub const Kernels = struct {
    decode: ze.H = null, // attn_k1_decode (k1.cl): 8-row groups
    decode16: ze.H = null, // attn_k1_decode16 (k1.cl): 16-row groups, 128-key tiles
    span: ze.H = null, // attn_decode_span (decode.cl): the scalar page-table kernel K1 is checked against
    merge: ze.H = null, // attn_merge_span (decode.cl): the merge of slice partials
};

/// Loads K1's kernels: the decode kernels from the k1 image, the scalar reference and the merge from decode.
pub fn load(d: *dev.Dev, md: ze.H, mk1: ze.H) !Kernels {
    const k = Kernels{
        .decode = try d.kernel(mk1, "attn_k1_decode", 256),
        .decode16 = try d.kernel(mk1, "attn_k1_decode16", 256),
        .span = try d.kernel(md, "attn_decode_span", 16),
        .merge = try d.kernel(md, "attn_merge_span", 256),
    };
    try d.g.indirect(k.decode);
    try d.g.indirect(k.decode16);
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

/// Queues the groups of `wf`/`wn` into `grp`: the host buffer must stay alive until the list drains.
pub fn queueGroups(d: *dev.Dev, grp: *anyopaque, host: []i32, n: u32) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, grp, host.ptr, @as(usize, 2 * n) * 4, null, 0, null));
}

/// Device bytes a round reads through K1: the K and V pages of `keys` keys for every stream, K/V head and row
/// group (2 x (256 + 4) bytes a key a head). One attention layer.
pub fn bytes(keys: u32, groups: u32, kv_heads: u32) u64 {
    return @as(u64, keys) * groups * kv_heads * 2 * (256 + 4);
}

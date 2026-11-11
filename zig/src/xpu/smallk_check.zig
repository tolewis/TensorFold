//! XPU_SMALLK bit 8 (lane p4-kernel): each new small kernel against the kernel it replaces, on this card, at the load.
//! The logits sit on a coarse grid, so each row holds many ties, and some rows hold the edge cases of a kernel. Every
//! token, list entry and output bit must agree, or the open fails.
const std = @import("std");
const ze = @import("ze");
const dev = @import("dev.zig");
const q = @import("qwen27.zig");
const Dev = dev.Dev;
const alloc = q.alloc;

const ROWS = 16;
const INT_MAX = std.math.maxInt(i32);

var seed: u64 = 0x2545F4914F6CDD1D;
fn next() u32 { // xorshift
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return @truncate(seed >> 32);
}
fn grid(n: u32) f16 { // an integer in [-(n / 2), n / 2]: ties in every row
    return @floatFromInt(@as(i32, @intCast(next() % n)) - @as(i32, @intCast(n / 2)));
}
fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}
fn unit() f32 { // [-1, 1)
    return (@as(f32, @floatFromInt(next() >> 8)) - 8388608.0) / 8388608.0;
}

const Held = struct {
    list: std.ArrayList(*anyopaque) = .empty,
    fn put(h: *Held, d: *Dev, comptime T: type, host: []const T) !*anyopaque {
        const p = try d.g.deviceAlloc(@max(host.len * @sizeOf(T), 64));
        try h.list.append(alloc, p);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
        return p;
    }
    fn room(h: *Held, d: *Dev, bytes: usize) !*anyopaque {
        const p = try d.g.deviceAlloc(@max(bytes, 64));
        try h.list.append(alloc, p);
        return p;
    }
    fn deinit(h: *Held, d: *Dev) void {
        for (h.list.items) |p| d.g.free(p);
        h.list.deinit(alloc);
    }
};
fn back(d: *Dev, comptime T: type, src: *anyopaque, n: usize) ![]T {
    const host = try alloc.alloc(T, n);
    errdefer alloc.free(host);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, host.ptr, src, n * @sizeOf(T), null, 0, null));
    try d.sync();
    return host;
}

/// argmax_part16 against argmax_part, each followed by `fin` (argmax): the token of each row.
pub fn argmax(d: *Dev, old: ze.H, new: ze.H, fin: ze.H) !void {
    const V = q.VOCAB;
    var held: Held = .{};
    defer held.deinit(d);
    const host = try alloc.alloc(f16, ROWS * V);
    defer alloc.free(host);
    for (host) |*x| x.* = grid(41);
    const inf = std.math.inf(f16);
    const row = struct {
        fn f(h: []f16, r: usize) []f16 {
            return h[r * V ..][0..V];
        }
    }.f;
    @memset(row(host, 1), -inf); // no value above -inf: the token is INT_MAX
    for (row(host, 2)) |*x| x.* = @min(x.*, 0);
    row(host, 2)[V - 1] = 30; // the largest value at the last index only
    row(host, 3)[0] = 30;
    row(host, 3)[V - 1] = 30; // at the first and the last index
    for (row(host, 4), 0..) |*x, i| if (i % 3 == 0) {
        x.* = std.math.nan(f16); // NaN is never the largest
    };
    for (row(host, 5), 0..) |*x, i| if (i % 2 == 0) {
        x.* = -inf;
    };
    row(host, 6)[V / 2] = inf; // +inf once, then at the end too
    row(host, 6)[V - 3] = inf;
    @memset(row(host, 7), 0);
    row(host, 7)[123457] = -0.0; // all ties at zero: the first index
    const L = try held.put(d, f16, host);
    const pv = try held.room(d, ROWS * 256 * 4);
    const pi = try held.room(d, ROWS * 256 * 4);
    var tok: [2]*anyopaque = undefined;
    for (&tok, [2]ze.H{ old, new }) |*t, k| {
        t.* = try held.room(d, ROWS * 4);
        try d.launch(k, .{ .x = 256, .y = ROWS }, .{ L, @as(i32, V), pv, pi });
        try d.launch(fin, .{ .x = ROWS }, .{ pv, pi, t.* });
    }
    const a = try back(d, i32, tok[0], ROWS);
    defer alloc.free(a);
    const b = try back(d, i32, tok[1], ROWS);
    defer alloc.free(b);
    var bad: usize = 0;
    for (a, b) |x, y| bad += @intFromBool(x != y);
    std.debug.print("smallk check: argmax_part16 against argmax_part, {d} rows, {d} tokens differ (row 1 {d}, row 3 {d}, row 7 {d})\n", .{ ROWS, bad, b[1], b[3], b[7] });
    if (bad != 0) return error.SmallkCheck;
}

/// The host merge of drafter.zig (Drafter.lattice): the top 16 of a row's part lists, ties to the lower index.
fn merge(pv: []const f32, pi: []const i32, r: usize, bv: *[16]f32, bi: *[16]i32) void {
    bv.* = @splat(-std.math.inf(f32));
    bi.* = @splat(INT_MAX);
    for (0..256 * 16) |j| {
        const v = pv[r * 4096 + j];
        const ix = pi[r * 4096 + j];
        if (ix == INT_MAX) continue;
        if (v < bv[15] or (v == bv[15] and ix > bi[15])) continue;
        var k: usize = 15;
        while (k > 0 and (v > bv[k - 1] or (v == bv[k - 1] and ix < bi[k - 1]))) : (k -= 1) {
            bv[k] = bv[k - 1];
            bi[k] = bi[k - 1];
        }
        bv[k] = v;
        bi[k] = ix;
    }
}

/// ROWS rows of N logits NS apart on a grid of ties, with the edge rows of the top-k kernels. No NaN: the part
/// kernels cut a row into other parts, and a NaN entry depends on its part.
fn logitRows(N: u32, NS: u32) ![]f16 {
    const host = try alloc.alloc(f16, ROWS * NS);
    for (host) |*x| x.* = grid(61);
    const inf = std.math.inf(f16);
    const row = struct {
        fn f(h: []f16, ns: u32, r: usize) []f16 {
            return h[r * ns ..][0..ns];
        }
    }.f;
    @memset(row(host, NS, 1)[0..N], -inf); // no entry
    @memset(row(host, NS, 2)[0..N], 0); // all ties
    for (row(host, NS, 3)[0..N], 0..) |*x, i| x.* = @floatFromInt(i % 2000); // rising runs
    for (row(host, NS, 4)[0..N], 0..) |*x, i| x.* = @floatFromInt(2000 - @as(i32, @intCast(i % 2000))); // falling runs
    for (row(host, NS, 5)[0..N], 0..) |*x, i| if (i % 2 == 0) {
        x.* = -inf;
    };
    for (row(host, NS, 6)[N..NS]) |*x| x.* = 100; // above every value, past N: never an entry
    const nz: f16 = @bitCast(@as(u16, 0x8000));
    for (row(host, NS, 7)[0..N], 0..) |*x, i| x.* = if (i % 3 == 0) nz else if (i % 3 == 1) 0 else -1; // -0 and +0 tie
    for (row(host, NS, 8)[0..N], 0..) |*x, i| x.* = if (i < 5) 7 else -inf; // 5 entries, then only empty entries
    return host;
}

/// topk_part16 against topk_part on N columns of rows NS apart: the merged top 16 of each row. No NaN: the two
/// kernels cut a row into other parts, and a NaN entry of topk_part depends on its part.
pub fn topk(d: *Dev, old: ze.H, new: ze.H, N: u32, NS: u32) !void {
    var held: Held = .{};
    defer held.deinit(d);
    const host = try logitRows(N, NS);
    defer alloc.free(host);
    const L = try held.put(d, f16, host);
    var pv: [2]*anyopaque = undefined;
    var pi: [2]*anyopaque = undefined;
    for (&pv, &pi, [2]ze.H{ old, new }) |*v, *i, k| {
        v.* = try held.room(d, ROWS * 256 * 16 * 4);
        i.* = try held.room(d, ROWS * 256 * 16 * 4);
        try d.fill(v.*, @as(u32, 0), ROWS * 256 * 16 * 4);
        try d.fill(i.*, @as(u32, 0), ROWS * 256 * 16 * 4);
        try d.launch(k, .{ .x = 256, .y = ROWS }, .{ L, @as(i32, @intCast(N)), @as(i32, @intCast(NS)), v.*, i.* });
    }
    var hv: [2][]f32 = undefined;
    var hi: [2][]i32 = undefined;
    for (0..2) |s| {
        hv[s] = try back(d, f32, pv[s], ROWS * 4096);
        hi[s] = try back(d, i32, pi[s], ROWS * 4096);
    }
    defer for (0..2) |s| {
        alloc.free(hv[s]);
        alloc.free(hi[s]);
    };
    var bad: usize = 0;
    var top: [2]i32 = .{ 0, 0 };
    for (0..ROWS) |r| {
        var bv: [2][16]f32 = undefined;
        var bi: [2][16]i32 = undefined;
        for (0..2) |s| merge(hv[s], hi[s], r, &bv[s], &bi[s]);
        for (0..16) |j| bad += @intFromBool(@as(u32, @bitCast(bv[0][j])) != @as(u32, @bitCast(bv[1][j])) or bi[0][j] != bi[1][j]);
        if (r == 3) top = .{ bi[1][0], bi[1][15] };
    }
    std.debug.print("smallk check: topk_part16 against topk_part, {d} rows of {d}, {d} of {d} entries differ (row 3 entries 0 and 15: {d} {d})\n", .{ ROWS, N, bad, ROWS * 16, top[0], top[1] });
    if (bad != 0) return error.SmallkCheck;
}

/// topk_merge (`merge`) against the host merge of drafter.zig, on the lists of `part` (topk_part or topk_part16) for
/// the rows of logitRows. Every entry's value bits and index must agree.
pub fn topkMerge(d: *Dev, part: ze.H, mk: ze.H, N: u32, NS: u32) !void {
    var held: Held = .{};
    defer held.deinit(d);
    const host = try logitRows(N, NS);
    defer alloc.free(host);
    const L = try held.put(d, f16, host);
    const pv = try held.room(d, ROWS * 256 * 16 * 4);
    const pi = try held.room(d, ROWS * 256 * 16 * 4);
    const ov = try held.room(d, ROWS * 16 * 4);
    const oi = try held.room(d, ROWS * 16 * 4);
    try d.fill(ov, @as(u32, 0x7fc00000), ROWS * 16 * 4); // NaN: an entry the kernel does not write shows
    try d.fill(oi, @as(u32, 0x55555555), ROWS * 16 * 4);
    try d.launch(part, .{ .x = 256, .y = ROWS }, .{ L, @as(i32, @intCast(N)), @as(i32, @intCast(NS)), pv, pi });
    try d.launch(mk, .{ .x = ROWS }, .{ pv, pi, ov, oi });
    const hv = try back(d, f32, pv, ROWS * 4096);
    defer alloc.free(hv);
    const hi = try back(d, i32, pi, ROWS * 4096);
    defer alloc.free(hi);
    const dv = try back(d, f32, ov, ROWS * 16);
    defer alloc.free(dv);
    const di = try back(d, i32, oi, ROWS * 16);
    defer alloc.free(di);
    var bad: usize = 0;
    var empty: usize = 0;
    for (0..ROWS) |r| {
        var bv: [16]f32 = undefined;
        var bi: [16]i32 = undefined;
        merge(hv, hi, r, &bv, &bi);
        for (0..16) |j| {
            bad += @intFromBool(@as(u32, @bitCast(bv[j])) != @as(u32, @bitCast(dv[r * 16 + j])) or bi[j] != di[r * 16 + j]);
            empty += @intFromBool(bi[j] == INT_MAX);
        }
    }
    std.debug.print("smallk check: topk_merge against the host merge, {d} rows, {d} of {d} entries differ ({d} empty entries)\n", .{ ROWS, bad, ROWS * 16, empty });
    if (bad != 0) return error.SmallkCheck;
}

/// `name` (dattn_la4 or dattn_la2, `local` lanes a group) against dattn, from module `m` with handles of their own
/// (the per-row table case reads K/V addresses from a buffer: indirect access). Every output bit must agree.
pub fn dattn(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32, CAP: u32, NH: u32, NKV: u32, HD: u32) !void {
    var held: Held = .{};
    defer held.deinit(d);
    const nul: ?*anyopaque = null;
    const old = try d.kernel(m, "dattn", 16);
    const new = try d.kernel(m, name, local);
    try d.g.indirect(old);
    try d.g.indirect(new);
    const qh = try alloc.alloc(u16, ROWS * NH * HD);
    defer alloc.free(qh);
    for (qh) |*x| x.* = bf16(unit());
    const Q = try held.put(d, u16, qh);
    const kvh = try alloc.alloc(u16, @as(usize, NKV) * CAP * HD);
    defer alloc.free(kvh);
    var kv: [4]*anyopaque = undefined; // K and V of two contexts
    for (&kv) |*p| {
        for (kvh) |*x| x.* = bf16(unit() * 2.0);
        p.* = try held.put(d, u16, kvh);
    }
    var O: [2]*anyopaque = undefined;
    for (&O) |*o| o.* = try held.room(d, ROWS * NH * HD * 2);
    const ptrs = try held.put(d, u64, &[4]u64{ @intFromPtr(kv[0]), @intFromPtr(kv[1]), @intFromPtr(kv[2]), @intFromPtr(kv[3]) });
    // the per-row table: rows 0 to 9 a block of 10 at 1000 (context 0), rows 10 to 15 a block of 6 at 50 (context 1)
    var rt: [ROWS][4]i32 = undefined;
    for (&rt, 0..) |*x, r| x.* = if (r < 10) .{ 1000 + @as(i32, @intCast(r)), 0, 1000, (10 << 8) | 0 } else .{ 50 + @as(i32, @intCast(r)) - 10, 0, 50, (6 << 8) | 1 };
    const rows = try held.put(d, [4]i32, &rt);
    const Case = struct { c0: i32, pos0: i32, l: i32, win: i32, r: u32, table: bool, what: []const u8 };
    const cases = [_]Case{
        .{ .c0 = 0, .pos0 = 100, .l = 16, .win = 2047, .r = 16, .table = false, .what = "short context" },
        .{ .c0 = 0, .pos0 = 2200, .l = 16, .win = 256, .r = 16, .table = false, .what = "window 256 across the ring end" },
        .{ .c0 = 500, .pos0 = 2600, .l = 16, .win = 2047, .r = 16, .table = false, .what = "window 2047 across the ring end" },
        .{ .c0 = 0, .pos0 = 0, .l = 16, .win = 2047, .r = 16, .table = false, .what = "the block alone" },
        .{ .c0 = 0, .pos0 = 300, .l = 5, .win = 256, .r = 5, .table = false, .what = "a block of 5" },
        .{ .c0 = 0, .pos0 = 0, .l = 16, .win = 256, .r = 16, .table = true, .what = "two blocks by the row table" },
    };
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HD)));
    var bad: usize = 0;
    var nan: usize = 0;
    for (cases) |c| {
        for ([2]ze.H{ old, new }, O) |k, o| {
            try d.fill(o, @as(u16, 0x7E00), ROWS * NH * HD * 2); // NaN where a kernel writes nothing
            try d.launch(k, .{ .x = c.r, .y = NKV }, .{ Q, kv[0], kv[1], @as(i32, @intCast(CAP)), c.c0, c.pos0, c.l, c.win, scale, o, if (c.table) rows else nul, if (c.table) ptrs else nul });
        }
        const a = try back(d, u16, O[0], c.r * NH * HD);
        defer alloc.free(a);
        const b = try back(d, u16, O[1], c.r * NH * HD);
        defer alloc.free(b);
        var n: usize = 0;
        for (a, b) |x, y| {
            n += @intFromBool(x != y);
            nan += @intFromBool(x & 0x7C00 == 0x7C00);
        }
        if (n != 0) std.debug.print("smallk check: {s} against dattn, {s}: {d} of {d} values differ\n", .{ name, c.what, n, a.len });
        bad += n;
    }
    std.debug.print("smallk check: {s} against dattn, {d} cases, {d} values differ, {d} values of dattn not finite\n", .{ name, cases.len, bad, nan });
    if (bad != 0 or nan != 0) return error.SmallkCheck;
}

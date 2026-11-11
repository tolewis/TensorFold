//! XPU_K1_R16=2 (lane p4-kernel): attn_k1_k16 against the 8-row groups of attn_k1_decode on this card, at the load.
//! Each case is a lone tree verify as forward.zig runs it: the prefix keys through the K1 decode kernel, the draft
//! keys through attn_k1_draft into slice NS, and attn_merge_span2. The two paths differ only in the prefix kernel
//! (two 8-row groups at 256-key tiles, or one 16-row group at 128-key tiles), so their outputs differ by f16 and
//! fp32 rounding only. Every (row, head) output must keep a cosine of MIN_COS and a mean relative error of at most
//! MAX_MEAN_REL against the 8-row path, or the open fails. Random K/V and Q, as k1_bench.zig fills them.
const std = @import("std");
const ze = @import("ze");
const dev = @import("dev.zig");
const q = @import("qwen27.zig");
const k1 = @import("k1.zig");
const Dev = dev.Dev;
const alloc = q.alloc;

const PAGE: usize = 64;
const AQ = q.ATT_Q_HEADS;
const AKV = q.ATT_KV_HEADS;
const PAGE8: usize = PAGE * (AKV * 256 + AKV * 4);
const MIN_COS = 0.999;
const MAX_MEAN_REL = 0.02;

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

const Held = struct {
    list: std.ArrayList(*anyopaque) = .empty,
    fn put(h: *Held, d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.g.deviceAlloc(@max(bytes.len, 64));
        try h.list.append(alloc, p);
        try d.copyIn(p, bytes);
        return p;
    }
    fn room(h: *Held, d: *Dev, n: usize) !*anyopaque {
        const p = try d.g.deviceAlloc(@max(n, 64));
        try h.list.append(alloc, p);
        return p;
    }
    fn deinit(h: *Held, d: *Dev) void {
        for (h.list.items) |p| d.g.free(p);
        h.list.deinit(alloc);
    }
};

const Result = struct { cos: f64, rel: f64 };

/// One case: a tree of `rows` rows (parent row t / 2) over `p0` prefix keys.
fn case(d: *Dev, kk: k1.Kernels, rnd: std.Random, p0: u32, rows: u32) !Result {
    var h: Held = .{};
    defer h.deinit(d);
    const ctx = p0 + rows;
    const pages: usize = (@as(usize, ctx) + PAGE - 1) / PAGE;
    const pool = try alloc.alloc(u8, 2 * pages * PAGE8);
    defer alloc.free(pool);
    for (0..2 * pages) |pg| {
        for (0..PAGE) |r| {
            for (0..AKV) |hh| {
                const row = pg * PAGE8 + r * (AKV * 256) + hh * 256;
                for (0..256) |i| pool[row + i] = @bitCast(@as(i8, @intCast(rnd.intRangeAtMost(i32, -127, 127))));
                const sc = pg * PAGE8 + PAGE * (AKV * 256) + (r * AKV + hh) * 4;
                const sv: f32 = 0.01 + 0.04 * rnd.float(f32);
                @memcpy(pool[sc..][0..4], std.mem.asBytes(&sv));
            }
        }
    }
    const dk = try h.put(d, pool[0 .. pages * PAGE8]);
    const dv = try h.put(d, pool[pages * PAGE8 ..]);
    const pt = try alloc.alloc(u32, pages);
    defer alloc.free(pt);
    for (pt, 0..) |*x, i| x.* = @intCast(i);
    rnd.shuffle(u32, pt);
    const dpt = try h.put(d, std.mem.sliceAsBytes(pt));
    const qh = try alloc.alloc(u16, @as(usize, rows) * AQ * 256);
    defer alloc.free(qh);
    for (qh) |*x| x.* = bf16((rnd.float(f32) - 0.5) * 2.0);
    const dq = try h.put(d, std.mem.sliceAsBytes(qh));
    // the MT table and POS of forward.zig's tree path: every row the slot's kc, vc and page table, POS p0 - 1
    var mt: [16 * 8]u64 = @splat(0);
    var pos: [16]i32 = @splat(0);
    for (0..16) |r| {
        mt[r * 8] = @intFromPtr(dk);
        mt[r * 8 + 1] = @intFromPtr(dv);
        mt[r * 8 + 2] = @intFromPtr(dpt);
        pos[r] = @as(i32, @intCast(p0)) - 1;
    }
    const dmt = try h.put(d, std.mem.sliceAsBytes(&mt));
    const dpos = try h.put(d, std.mem.sliceAsBytes(&pos));
    var tree: [16 * 17]i32 = @splat(0); // a binary tree, parent row t / 2, as verifyTree lays it out
    for (1..rows) |t| {
        const pr = tree[t / 2 * 17 ..][0..17];
        const dp: usize = @intCast(pr[0]);
        tree[t * 17] = pr[0] + 1;
        @memcpy(tree[t * 17 + 1 ..][0 .. dp + 1], pr[1 .. dp + 2]);
        tree[t * 17 + 1 + dp + 1] = @intCast(t);
    }
    const dtree = try h.put(d, std.mem.sliceAsBytes(&tree));
    const n = @as(usize, rows) * AQ * 256;
    var outs: [2][]u16 = .{ &.{}, &.{} };
    defer for (outs) |o| alloc.free(o);
    for (&outs, [_]bool{ false, true }) |*o, k16| {
        var grp: [32]i32 = @splat(0);
        const wf: [1]u32 = .{0};
        const wn: [1]u32 = .{rows};
        const ng: u32 = if (k16) blk: {
            grp[0] = 0;
            grp[1] = @intCast(rows);
            break :blk 1;
        } else k1.rowGroups(&wf, &wn, &grp);
        const pl = k1.planTile(p0, ng, AKV, k1.wave, if (k16) k1.TILE_K16 else k1.TILE);
        const dgrp = try h.put(d, std.mem.sliceAsBytes(grp[0 .. 2 * ng]));
        const posm: [16]i32 = @splat(@as(i32, @intCast(pl.ns)) * pl.span + pl.span - 1);
        const dposm = try h.put(d, std.mem.sliceAsBytes(posm[0..rows]));
        const slices: usize = pl.ns + 1;
        const pm = try h.room(d, @as(usize, rows) * AQ * slices * 4);
        const plb = try h.room(d, @as(usize, rows) * AQ * slices * 4);
        const po = try h.room(d, @as(usize, rows) * AQ * slices * 256 * 4);
        const out = try h.room(d, n * 2);
        const p0i: i32 = @intCast(p0);
        const nsi: i32 = @intCast(pl.ns);
        const scale: f32 = 1.0 / 16.0;
        try d.launch(if (k16) kk.k16 else kk.decode, .{ .x = pl.ns, .y = AKV, .z = ng }, .{ dq, dk, dv, pm, plb, po, p0i, nsi + 1, pl.span, scale, dmt, dpos, @as(i32, 1), dpt, dgrp });
        try d.launch(kk.draft, .{ .x = 1, .y = AKV, .z = rows }, .{ dq, dk, dv, pm, plb, po, p0i, nsi, scale, dtree, dpt });
        try d.launch(kk.merge2, .{ .x = AQ, .y = rows }, .{ pm, plb, po, out, p0i, nsi + 1, pl.span, dposm });
        o.* = try alloc.alloc(u16, n);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, o.*.ptr, out, n * 2, null, 0, null));
        try d.sync();
    }
    var res: Result = .{ .cos = 1.0, .rel = 0.0 };
    for (0..@as(usize, rows) * AQ) |rh| {
        var dot: f64 = 0;
        var n1: f64 = 0;
        var n2: f64 = 0;
        var rel: f64 = 0;
        for (0..256) |i| {
            const x: f64 = @as(f32, @bitCast(@as(u32, outs[1][rh * 256 + i]) << 16));
            const y: f64 = @as(f32, @bitCast(@as(u32, outs[0][rh * 256 + i]) << 16));
            if (std.math.isNan(x) or std.math.isNan(y)) return .{ .cos = 0, .rel = std.math.inf(f64) };
            dot += x * y;
            n1 += x * x;
            n2 += y * y;
            rel += if (y != 0) @abs(x - y) / @abs(y) else 0;
        }
        const cos: f64 = if (n1 > 0 and n2 > 0) dot / @sqrt(n1 * n2) else if (n1 == n2) @as(f64, 1.0) else @as(f64, 0.0);
        res.cos = @min(res.cos, cos);
        res.rel = @max(res.rel, rel / 256.0);
    }
    return res;
}

pub fn run(d: *Dev, kk: k1.Kernels) !void {
    if (kk.k16 == null or kk.decode == null or kk.draft == null or kk.merge2 == null) return error.K16NotLoaded;
    var prng = std.Random.DefaultPrng.init(0x16c4ec);
    const rnd = prng.random();
    // 16 rows (6 passes) and 11 rows (5 passes, a half with 2), short and long prefixes, a prefix that is no whole
    // number of tiles
    const cases = [_][2]u32{ .{ 16, 300 }, .{ 16, 20011 }, .{ 11, 4093 }, .{ 9, 1000 } };
    var worst: Result = .{ .cos = 1.0, .rel = 0.0 };
    for (cases) |c| {
        const r = try case(d, kk, rnd, c[1], c[0]);
        worst.cos = @min(worst.cos, r.cos);
        worst.rel = @max(worst.rel, r.rel);
    }
    const ok = worst.cos >= MIN_COS and worst.rel <= MAX_MEAN_REL;
    std.debug.print("k16 check: attn_k1_k16 against two 8-row groups, {d} cases, min cos {d:.6}, mean rel {d:.5} ({s}, bounds {d}, {d})\n", .{ cases.len, worst.cos, worst.rel, if (ok) "pass" else "FAIL", MIN_COS, MAX_MEAN_REL });
    if (!ok) return error.K16Differs;
}

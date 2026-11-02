//! Device bench: K1 (attn_k1_decode / attn_k1_decode16, k1.cl) against the scalar page-table kernel
//! attn_decode_span (decode.cl). Cases: one stream at 1 and 16 rows, 40k and 128k; a 16-row single group (the
//! variant that reads a 16-row window's K/V once) against the 8-row groups; and 8 streams of 16 rows at 40k, each
//! stream its own pages. Fills the pool with hash data, runs both, compares the outputs (cosine and relative
//! error), and prints GB/s and the work-group counts. Opens a card: run it in a card window.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const kb = @import("xpu_qwen27_kernels");
const k1 = @import("k1.zig");
const ze = q.ze;

const PAGE: usize = 64;
const AQ = q.ATT_Q_HEADS;
const AKV = q.ATT_KV_HEADS;
const PAGE8: usize = PAGE * (AKV * 256 + AKV * 4);
/// Cosine a head's output must keep against the scalar kernel, and the mean relative error allowed.
const MIN_COS = 0.995;
const MAX_MEAN_REL = 0.05;

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)));
}

fn launchTimed(d: *dev.Dev, k: ze.H, g: ze.GroupCount, args: anytype, ev: ze.H) !void {
    inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
    try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &g, ev, 0, null));
}

fn toF32(src: []const u16, dst: []f32) void {
    for (src, dst) |h, *v| v.* = @bitCast(@as(u32, h) << 16);
}

/// One bench case: `streams` streams of `rows` rows each at `ctx` keys; `k16` picks the 16-row variant (one group
/// a stream) against the 8-row groups (two a stream).
const Case = struct { ctx: u32, streams: u32, rows: u32, k16: bool, tree: bool = false };

var worst_cos: f64 = 1.0;
var worst_rel: f64 = 0.0;

fn run(d: *dev.Dev, gpa: std.mem.Allocator, kk: k1.Kernels, evs: []const ze.H, rnd: std.Random, c: Case) !void {
    const t = c.streams * c.rows; // rows of the round
    const tn: usize = @intCast(t);
    const p0: u32 = c.ctx - c.rows; // the tree case's prefix keys: the draft rows sit at p0 .. p0 + rows - 1
    const pages: usize = (@as(usize, c.ctx) + PAGE - 1) / PAGE;
    // every stream its own run of pages: one pool, the ids of stream s in [s * pages, (s + 1) * pages)
    const half: usize = @as(usize, c.streams) * pages * PAGE8;
    const pool = try gpa.alloc(u8, 2 * half);
    defer gpa.free(pool);
    // K pages then V pages: the K row of key k at page + (k % PAGE) * AKV * 256 + h * 256, the scales after
    for (0..@as(usize, c.streams) * pages) |pg| {
        for (0..PAGE) |r| {
            for (0..AKV) |h| {
                const row = pg * PAGE8 + r * (AKV * 256) + h * 256;
                for (0..256) |i| pool[row + i] = @bitCast(@as(i8, @intCast(rnd.intRangeAtMost(i32, -127, 127))));
                const sc = pg * PAGE8 + PAGE * (AKV * 256) + (r * AKV + h) * 4;
                const sv: f32 = 0.01 + 0.04 * rnd.float(f32);
                @memcpy(pool[sc..][0..4], std.mem.asBytes(&sv));
            }
        }
    }
    const dk = try d.upload(pool[0..half]);
    const dv = try d.upload(pool[half..]);
    var pt = try gpa.alloc(u32, @as(usize, c.streams) * pages);
    defer gpa.free(pt);
    for (0..c.streams) |s| {
        for (0..pages) |i| pt[s * pages + i] = @intCast(s * pages + i);
        rnd.shuffle(u32, pt[s * pages ..][0..pages]);
    }
    var dpt = try gpa.alloc(*anyopaque, @as(usize, c.streams));
    defer gpa.free(dpt);
    for (0..@as(usize, c.streams)) |s| dpt[s] = try d.upload(std.mem.sliceAsBytes(pt[s * pages ..][0..pages]));
    const qh = try gpa.alloc(u16, @as(usize, t) * AQ * 256);
    defer gpa.free(qh);
    for (qh) |*x| x.* = bf16((rnd.float(f32) - 0.5) * 2.0);
    const dq = try d.upload(std.mem.sliceAsBytes(qh));
    // the row groups: one a stream with the 16-row variant, two with the 8-row one
    var win = try gpa.alloc(i32, @as(usize, c.streams) * 2);
    defer gpa.free(win);
    for (0..c.streams) |s| {
        win[2 * s] = @intCast(s * c.rows);
        win[2 * s + 1] = @intCast(c.rows);
    }
    k1.r16 = c.k16;
    var grp: [2 * 16]i32 = @splat(0);
    const rd = k1.roundGroups(win, @intCast(t), &grp);
    const tile: u32 = if (rd.k16) k1.TILE16 else k1.TILE;
    const pl = k1.planTile(if (c.tree) p0 else c.ctx, rd.n, AKV, k1.wave, tile);
    const dgrp = try d.upload(std.mem.sliceAsBytes(grp[0 .. @as(usize, 2 * rd.n)]));
    // one row a stream: MT[t * 8] = the stream's K, V and page table, POS[t] = its last key. One stream: PT direct.
    const mt_host = try gpa.alloc(u64, @as(usize, t) * 8);
    defer gpa.free(mt_host);
    @memset(mt_host, 0);
    const pos_host = try gpa.alloc(i32, @as(usize, t));
    defer gpa.free(pos_host);
    for (0..t) |i| {
        mt_host[i * 8] = @intFromPtr(dk);
        mt_host[i * 8 + 1] = @intFromPtr(dv);
        mt_host[i * 8 + 2] = @intFromPtr(dpt[i / c.rows]);
        // chain: every row sees its window's keys to the end; tree: the last prefix key (the draft keys are the
        // draft kernel's), so the K1 groups and the scalar tree kernel read the same prefix
        pos_host[i] = @intCast(if (c.tree) p0 - 1 else c.ctx - 1);
    }
    const dmt = try d.upload(std.mem.sliceAsBytes(mt_host));
    const dpos = try d.upload(std.mem.sliceAsBytes(pos_host));
    // the tree case: a 16-row binary tree (parent row t / 2), the table as verifyTree lays it out
    var tree_host: [16 * 17]i32 = @splat(0);
    for (1..16) |i| {
        const tt: usize = @intCast(i);
        const pr = tree_host[tt / 2 * 17 ..][0..17];
        const dp: usize = @intCast(pr[0]);
        tree_host[tt * 17] = pr[0] + 1;
        @memcpy(tree_host[tt * 17 + 1 ..][0 .. dp + 1], pr[1 .. dp + 2]);
        tree_host[tt * 17 + 1 + dp + 1] = @as(i32, @intCast(tt));
    }
    const dtree: ?*anyopaque = if (c.tree) try d.upload(std.mem.sliceAsBytes(tree_host[0..])) else null;
    // the merge's POS for the tree case: NS * span + span - 1, so used = NS + 1 and the draft's slice NS joins
    var posm_host: [16]i32 = @splat(0);
    const posmi: i32 = @as(i32, @intCast(pl.ns)) * pl.span + pl.span - 1;
    const pmn: usize = @min(tn, 16); // the tree case is one stream of at most 16 rows
    for (0..pmn) |i| posm_host[i] = posmi;
    const dposm: ?*anyopaque = if (c.tree) try d.upload(std.mem.sliceAsBytes(posm_host[0..tn])) else null;
    const mpos: ?*anyopaque = if (c.tree) dposm else dpos;
    // the K1 partials: the prefix slices plus the draft's; the scalar tree kernel: one slice a 32 keys
    const k1slices: u32 = if (c.tree) pl.ns + 1 else pl.ns;
    const scns: u32 = if (c.tree) (p0 + 47) / 32 else pl.ns;
    const scni: i32 = @intCast(scns);
    const pm = try d.alloc_(tn * AQ * k1slices * 4);
    const plb = try d.alloc_(tn * AQ * k1slices * 4);
    const po = try d.alloc_(tn * AQ * k1slices * 256 * 4);
    const out = try d.alloc_(tn * AQ * 256 * 2);
    // the scalar path gets its own partials, so a scalar launch that writes nothing cannot pass by reading K1's
    const pm2 = try d.alloc_(tn * AQ * scns * 4);
    const plb2 = try d.alloc_(tn * AQ * scns * 4);
    const po2 = try d.alloc_(tn * AQ * scns * 256 * 4);
    const out2 = try d.alloc_(tn * AQ * 256 * 2);
    const scale: f32 = 1.0 / 16.0;
    const p0i: i32 = @intCast(c.ctx - c.rows);
    const nsi: i32 = @intCast(pl.ns);
    const dpt1: ?*anyopaque = if (c.streams == 1) dpt[0] else null;

    // K1, timed: the decode launch and the merge, 3 reps, the shortest; the tree case adds the draft launch
    var best: f64 = 1e9;
    for (0..3) |_| {
        if (c.tree) {
            try launchTimed(d, kk.decode, .{ .x = pl.ns, .y = AKV, .z = rd.n }, .{ dq, dk, dv, pm, plb, po, p0i, nsi, pl.span, scale, dmt, dpos, @as(i32, 1), dpt1, dgrp }, evs[0]);
            try launchTimed(d, kk.draft, .{ .x = 1, .y = AKV, .z = @intCast(t) }, .{ dq, dk, dv, pm, plb, po, p0i, nsi, scale, dtree, dpt1 }, evs[1]);
            try launchTimed(d, kk.merge, .{ .x = AQ, .y = @intCast(t) }, .{ pm, plb, po, out, p0i, nsi, pl.span, mpos }, evs[2]);
            try d.sync();
            best = @min(best, try d.g.seconds(evs[0]) + try d.g.seconds(evs[1]) + try d.g.seconds(evs[2]));
            try ze.check(d.g.api.zeEventHostReset(evs[0]));
            try ze.check(d.g.api.zeEventHostReset(evs[1]));
            try ze.check(d.g.api.zeEventHostReset(evs[2]));
        } else {
            const kern = if (rd.k16) kk.decode16 else kk.decode;
            try launchTimed(d, kern, .{ .x = pl.ns, .y = AKV, .z = rd.n }, .{ dq, dk, dv, pm, plb, po, p0i, nsi, pl.span, scale, dmt, dpos, @as(i32, 1), dpt1, dgrp }, evs[0]);
            try launchTimed(d, kk.merge, .{ .x = AQ, .y = @intCast(t) }, .{ pm, plb, po, out, p0i, nsi, pl.span, dpos }, evs[1]);
            try d.sync();
            best = @min(best, try d.g.seconds(evs[0]) + try d.g.seconds(evs[1]));
            try ze.check(d.g.api.zeEventHostReset(evs[0]));
            try ze.check(d.g.api.zeEventHostReset(evs[1]));
        }
    }
    // the scalar page-table kernel, timed the same way; the tree case: attn_decode + attn_merge
    var best2: f64 = 1e9;
    for (0..3) |_| {
        if (c.tree) {
            try launchTimed(d, kk.dec, .{ .x = scns, .y = AKV, .z = @intCast(t) }, .{ dq, dk, dv, pm2, plb2, po2, p0i, scni, scale, dtree, @as(?*anyopaque, null), @as(?*anyopaque, null), @as(i32, 1), dpt1 }, evs[0]);
            try launchTimed(d, kk.tmerge, .{ .x = AQ, .y = @intCast(t) }, .{ pm2, plb2, po2, out2, p0i, scni, dtree, @as(?*anyopaque, null) }, evs[1]);
            try d.sync();
            best2 = @min(best2, try d.g.seconds(evs[0]) + try d.g.seconds(evs[1]));
            try ze.check(d.g.api.zeEventHostReset(evs[0]));
            try ze.check(d.g.api.zeEventHostReset(evs[1]));
        } else {
            try launchTimed(d, kk.span, .{ .x = pl.ns, .y = AKV, .z = @intCast(t) }, .{ dq, dk, dv, pm2, plb2, po2, p0i, nsi, pl.span, scale, dmt, dpos, @as(i32, 1), dpt1 }, evs[0]);
            try launchTimed(d, kk.merge, .{ .x = AQ, .y = @intCast(t) }, .{ pm2, plb2, po2, out2, p0i, nsi, pl.span, dpos }, evs[1]);
            try d.sync();
            best2 = @min(best2, try d.g.seconds(evs[0]) + try d.g.seconds(evs[1]));
            try ze.check(d.g.api.zeEventHostReset(evs[0]));
            try ze.check(d.g.api.zeEventHostReset(evs[1]));
        }
    }
    const n = tn * AQ * 256;
    const b1 = try gpa.alloc(u16, n);
    defer gpa.free(b1);
    const b2 = try gpa.alloc(u16, n);
    defer gpa.free(b2);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, b1.ptr, out, n * 2, null, 0, null));
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, b2.ptr, out2, n * 2, null, 0, null));
    try d.sync();
    const a1 = try gpa.alloc(f32, n);
    defer gpa.free(a1);
    const a2 = try gpa.alloc(f32, n);
    defer gpa.free(a2);
    toF32(b1, a1);
    toF32(b2, a2);
    var wc: f64 = 1.0;
    var mr: f64 = 0.0;
    for (0..t) |tt| {
        for (0..AQ) |h| {
            var dot: f64 = 0;
            var n1: f64 = 0;
            var n2: f64 = 0;
            var rel: f64 = 0;
            for (0..256) |i| {
                const x: f64 = a1[(tt * AQ + h) * 256 + i];
                const y: f64 = a2[(tt * AQ + h) * 256 + i];
                dot += x * y;
                n1 += x * x;
                n2 += y * y;
                rel += if (y != 0) @abs(x - y) / @abs(y) else 0;
            }
            const cos = if (n1 > 0 and n2 > 0) dot / @sqrt(n1 * n2) else 1.0;
            wc = @min(wc, cos);
            mr = @max(mr, rel / 256.0);
        }
    }
    worst_cos = @min(worst_cos, wc);
    worst_rel = @max(worst_rel, mr);
    const bytes = k1.bytes(if (c.tree) p0 else c.ctx, rd.n, AKV);
    const gbs = @as(f64, @floatFromInt(bytes)) / 1e9;
    const label = if (c.tree) "tree, 8-row groups" else if (rd.k16) "16-row group" else "8-row groups";
    std.debug.print("ctx {d}, {d} stream(s) x {d} rows, {s}: {d} work-groups (ns {d}, span {d}, groups {d}), {d:.1} MB, K1 {d:.2} ms {d:.0} GB/s, scalar {d:.2} ms {d:.0} GB/s, min cos {d:.5}, mean rel {d:.4}\n", .{
        c.ctx, c.streams, c.rows, label, pl.wgs, pl.ns, pl.span, rd.n, gbs * 1e3,
        best * 1e3, gbs / best, best2 * 1e3, gbs / best2, wc, mr,
    });
    if (wc < MIN_COS or mr > MAX_MEAN_REL) {
        std.debug.print("MISMATCH at ctx {d}, {d} x {d} rows, {s}\n", .{ c.ctx, c.streams, c.rows, label });
        return error.OutputMismatch;
    }
}

pub fn main() !void {
    const gpa = std.heap.c_allocator;
    var d: dev.Dev = .{ .g = ze.Gpu.open(0xe223) catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    }, .list = undefined, .k = undefined };
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0);
    defer d.deinit();
    const mk1 = try d.g.module(&kb.k1);
    const md = try d.g.module(&kb.decode);
    const kk = try k1.load(&d, md, mk1);
    const evs = try d.g.events(3);

    var prng = std.Random.DefaultPrng.init(0x1ce5);
    const rnd = prng.random();
    std.debug.print("K1 bench: AQ {d}, AKV {d}, rows a group {d} (16-row {d}), page {d} bytes\n", .{ AQ, AKV, k1.ROWS, k1.ROWS16, PAGE8 });

    for ([_]Case{
        // one stream: one row, and a 16-row verify window with the 8-row groups (two reads) and the 16-row one
        .{ .ctx = 40000, .streams = 1, .rows = 1, .k16 = false },
        .{ .ctx = 40000, .streams = 1, .rows = 16, .k16 = false },
        .{ .ctx = 40000, .streams = 1, .rows = 16, .k16 = true },
        .{ .ctx = 128000, .streams = 1, .rows = 1, .k16 = false },
        .{ .ctx = 128000, .streams = 1, .rows = 16, .k16 = false },
        .{ .ctx = 128000, .streams = 1, .rows = 16, .k16 = true },
        // the tree verify: one stream of a 16-row binary tree, the prefix through the 8-row groups, the draft keys
        // through attn_k1_draft, checked against attn_decode + attn_merge
        .{ .ctx = 40000, .streams = 1, .rows = 16, .k16 = false, .tree = true },
        .{ .ctx = 128000, .streams = 1, .rows = 16, .k16 = false, .tree = true },
        // 8 streams at 40k, each its own pages: the round the spec's byte table describes
        .{ .ctx = 40000, .streams = 8, .rows = 16, .k16 = false },
        .{ .ctx = 40000, .streams = 8, .rows = 16, .k16 = true },
    }) |c| try run(&d, gpa, kk, &evs, rnd, c);

    std.debug.print("worst cosine {d:.5}, worst mean relative error {d:.4} (bounds {d}, {d})\n", .{ worst_cos, worst_rel, MIN_COS, MAX_MEAN_REL });
    std.debug.print("ok\n", .{});
}

//! Device bench: K1 (attn_k1_decode, k1.cl) against the scalar page-table kernel attn_decode_span (decode.cl) at
//! 40k and 128k context, 1 and 16 rows. Fills the pool with hash data, runs both, compares the outputs (cosine and
//! relative error), and prints GB/s and the work-group counts. Opens a card: run it in a card window.
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
    const evs = try d.g.events(2);

    var prng = std.Random.DefaultPrng.init(0x1ce5);
    const rnd = prng.random();
    std.debug.print("K1 bench: AQ {d}, AKV {d}, rows a work-group {d}, page {d} bytes\n", .{ AQ, AKV, k1.ROWS, PAGE8 });

    var worst_cos: f64 = 1.0;
    var worst_rel: f64 = 0.0;
    for ([_]u32{ 40000, 128000 }) |ctx| {
        const pages: usize = (@as(usize, ctx) + PAGE - 1) / PAGE;
        const half: usize = pages * PAGE8;
        const pool = try gpa.alloc(u8, 2 * half);
        defer gpa.free(pool);
        // K pages then V pages: the K row of key k at page + (k % PAGE) * AKV * 256 + h * 256, the scales after
        for (0..pages) |pg| {
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
        const pt = try gpa.alloc(u32, pages);
        defer gpa.free(pt);
        for (pt, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(u32, pt);
        const dpt = try d.upload(std.mem.sliceAsBytes(pt));
        for ([_]u32{ 1, 16 }) |rows| {
            const p0: u32 = ctx - rows;
            const qh = try gpa.alloc(u16, rows * AQ * 256);
            defer gpa.free(qh);
            for (qh) |*x| x.* = bf16((rnd.float(f32) - 0.5) * 2.0);
            const dq = try d.upload(std.mem.sliceAsBytes(qh));
            var wf = [_]u32{0};
            var wn = [_]u32{rows};
            var grp: [2 * 16]i32 = @splat(0);
            const ng = k1.rowGroups(&wf, &wn, &grp);
            const pl = k1.plan(ctx, ng, AKV, k1.wave);
            const dgrp = try d.upload(std.mem.sliceAsBytes(grp[0 .. @as(usize, 2 * ng)]));
            const pm = try d.alloc_(16 * AQ * pl.ns * 4);
            const plb = try d.alloc_(16 * AQ * pl.ns * 4);
            const po = try d.alloc_(16 * AQ * pl.ns * 256 * 4);
            const out = try d.alloc_(rows * AQ * 256 * 2);
            const out2 = try d.alloc_(rows * AQ * 256 * 2);
            const scale: f32 = 1.0 / 16.0;
            const p0i: i32 = @intCast(p0);
            const nsi: i32 = @intCast(pl.ns);
            const nul: ?*anyopaque = null;

            // K1, timed: the decode launch and the merge, 3 reps, the shortest
            var best: f64 = 1e9;
            for (0..3) |_| {
                try launchTimed(&d, kk.decode, .{ .x = pl.ns, .y = AKV, .z = ng }, .{ dq, dk, dv, pm, plb, po, p0i, nsi, pl.span, scale, nul, nul, @as(i32, 1), dpt, dgrp }, evs[0]);
                try launchTimed(&d, kk.merge, .{ .x = AQ, .y = rows }, .{ pm, plb, po, out, p0i, nsi, pl.span, nul }, evs[1]);
                try d.sync();
                best = @min(best, try d.g.seconds(evs[0]) + try d.g.seconds(evs[1]));
                try ze.check(d.g.api.zeEventHostReset(evs[0]));
                try ze.check(d.g.api.zeEventHostReset(evs[1]));
            }
            // the scalar page-table kernel, timed the same way
            var best2: f64 = 1e9;
            for (0..3) |_| {
                try launchTimed(&d, kk.span, .{ .x = pl.ns, .y = AKV, .z = rows }, .{ dq, dk, dv, pm, plb, po, p0i, nsi, pl.span, scale, nul, nul, @as(i32, 1), dpt }, evs[0]);
                try launchTimed(&d, kk.merge, .{ .x = AQ, .y = rows }, .{ pm, plb, po, out2, p0i, nsi, pl.span, nul }, evs[1]);
                try d.sync();
                best2 = @min(best2, try d.g.seconds(evs[0]) + try d.g.seconds(evs[1]));
                try ze.check(d.g.api.zeEventHostReset(evs[0]));
                try ze.check(d.g.api.zeEventHostReset(evs[1]));
            }
            const b1 = try gpa.alloc(u16, rows * AQ * 256);
            defer gpa.free(b1);
            const b2 = try gpa.alloc(u16, rows * AQ * 256);
            defer gpa.free(b2);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, b1.ptr, out, rows * AQ * 256 * 2, null, 0, null));
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, b2.ptr, out2, rows * AQ * 256 * 2, null, 0, null));
            try d.sync();
            const a1 = try gpa.alloc(f32, rows * AQ * 256);
            defer gpa.free(a1);
            const a2 = try gpa.alloc(f32, rows * AQ * 256);
            defer gpa.free(a2);
            toF32(b1, a1);
            toF32(b2, a2);
            var wc: f64 = 1.0;
            var mr: f64 = 0.0;
            for (0..rows) |t| {
                for (0..AQ) |h| {
                var dot: f64 = 0;
                var n1: f64 = 0;
                var n2: f64 = 0;
                var rel: f64 = 0;
                for (0..256) |i| {
                    const x: f64 = a1[(t * AQ + h) * 256 + i];
                    const y: f64 = a2[(t * AQ + h) * 256 + i];
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
            const bytes = k1.bytes(ctx, ng, AKV);
            const gbs = @as(f64, @floatFromInt(bytes)) / 1e9;
            std.debug.print("ctx {d}, rows {d}: {d} work-groups (ns {d}, span {d}, groups {d}), {d:.1} MB, K1 {d:.2} ms {d:.0} GB/s, scalar {d:.2} ms {d:.0} GB/s, min cos {d:.5}, mean rel {d:.4}\n", .{
                ctx, rows, pl.wgs, pl.ns, pl.span, ng, gbs * 1e3,
                best * 1e3, gbs / best, best2 * 1e3, gbs / best2, wc, mr,
            });
            if (wc < MIN_COS or mr > MAX_MEAN_REL) {
                std.debug.print("MISMATCH at ctx {d}, rows {d}\n", .{ ctx, rows });
                return error.OutputMismatch;
            }
        }
        // the plan at 8 streams, 16 rows a stream, for the handoff
        var wf8 = [_]u32{0} ** 8;
        var wn8 = [_]u32{16} ** 8;
        for (1..8) |i| wf8[i] = @intCast(i * 16);
        var grp8: [2 * 32]i32 = @splat(0);
        const ng8 = k1.rowGroups(&wf8, &wn8, &grp8);
        const p8 = k1.plan(ctx, ng8, AKV, k1.wave);
        const b8 = @as(f64, @floatFromInt(k1.bytes(ctx, ng8, AKV))) / 1e9;
        std.debug.print("ctx {d}, 8 streams x 16 rows: {d} work-groups (ns {d}, span {d}, groups {d}), {d:.2} MB a round, at 605 GB/s {d:.1} ms\n", .{
            ctx, p8.wgs, p8.ns, p8.span, ng8, b8 * 1e3, b8 / 605.0 * 1e3,
        });
    }
    std.debug.print("worst cosine {d:.5}, worst mean relative error {d:.4} (bounds {d}, {d})\n", .{ worst_cos, worst_rel, MIN_COS, MAX_MEAN_REL });
    std.debug.print("ok\n", .{});
}

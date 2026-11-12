//! XPU_K1B bit 4 (lane p4-kernel, K1B): the picked attn_k1_b kernel against attn_k1_decode on this card, at the load.
//! Both kernels get the same random int8 K/V pages (shuffled page table), the same Q and the same row groups, and
//! write their own partials (PM, PL, PO). K1B only moves the reads and prefetches of a tile, so every written
//! partial must have the same bits. One value that differs fails the open. Cases: the lone tree verify of
//! forward.zig (the MT table, POS p0 - 1, two 8-row groups or a short one) and a chain window (no MT table, the keys
//! up to p0 + row).
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
    fn deinit(h: *Held, d: *Dev) void {
        for (h.list.items) |p| d.g.free(p);
        h.list.deinit(alloc);
    }
};

const Case = struct { rows: u32, p0: u32, mt: bool };

/// One case: returns the count of partial values (PM, PL and PO words) that differ.
fn case(d: *Dev, base: ze.H, kb: ze.H, rnd: std.Random, c: Case) !usize {
    var h: Held = .{};
    defer h.deinit(d);
    const ctx = c.p0 + c.rows;
    const pages: usize = (@as(usize, ctx) + PAGE - 1) / PAGE;
    const pool = try alloc.alloc(u8, 2 * pages * PAGE8);
    defer alloc.free(pool);
    for (0..2 * pages) |pg| {
        for (0..PAGE) |r| {
            for (0..AKV) |hh| {
                const row = pg * PAGE8 + r * (AKV * 256) + hh * 256;
                for (0..256) |i| pool[row + i] = @bitCast(@as(i8, @intCast(rnd.intRangeAtMost(i32, -128, 127))));
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
    const qh = try alloc.alloc(u16, @as(usize, c.rows) * AQ * 256);
    defer alloc.free(qh);
    for (qh) |*x| x.* = bf16((rnd.float(f32) - 0.5) * 2.0);
    const dq = try h.put(d, std.mem.sliceAsBytes(qh));
    var mt: [16 * 8]u64 = @splat(0);
    var pos: [16]i32 = @splat(0);
    for (0..16) |r| {
        mt[r * 8] = @intFromPtr(dk);
        mt[r * 8 + 1] = @intFromPtr(dv);
        mt[r * 8 + 2] = @intFromPtr(dpt);
        pos[r] = @as(i32, @intCast(c.p0)) - 1;
    }
    const dmt: ?*anyopaque = if (c.mt) try h.put(d, std.mem.sliceAsBytes(&mt)) else null;
    const dpos = try h.put(d, std.mem.sliceAsBytes(&pos));
    var grp: [32]i32 = @splat(0);
    const wf: [1]u32 = .{0};
    const wn: [1]u32 = .{c.rows};
    const ng = k1.rowGroups(&wf, &wn, &grp);
    const pl = k1.planTile(if (c.mt) c.p0 else ctx, ng, AKV, k1.wave, k1.TILE);
    const dgrp = try h.put(d, std.mem.sliceAsBytes(grp[0 .. 2 * ng]));
    const slices: usize = pl.ns + 1; // the stride of forward.zig's tree path (slice NS is the draft's)
    const nm = @as(usize, c.rows) * AQ * slices;
    // both kernels write into buffers that start with the same bytes, so an unwritten word compares equal
    const zero = try alloc.alloc(u8, nm * 256 * 4);
    defer alloc.free(zero);
    @memset(zero, 0xA5);
    var host: [2][3][]u32 = .{ .{ &.{}, &.{}, &.{} }, .{ &.{}, &.{}, &.{} } };
    defer for (host) |o| for (o) |b| alloc.free(b);
    for (&host, [_]ze.H{ base, kb }) |*o, kern| {
        const pm = try h.put(d, zero[0 .. nm * 4]);
        const plb = try h.put(d, zero[0 .. nm * 4]);
        const po = try h.put(d, zero);
        const p0i: i32 = @intCast(c.p0);
        try d.launch(kern, .{ .x = pl.ns, .y = AKV, .z = ng }, .{ dq, dk, dv, pm, plb, po, p0i, @as(i32, @intCast(slices)), pl.span, @as(f32, 1.0 / 16.0), dmt, dpos, @as(i32, 1), dpt, dgrp });
        for (o[0..], [_]*anyopaque{ pm, plb, po }, [_]usize{ nm, nm, nm * 256 }) |*buf, src, n| {
            buf.* = try alloc.alloc(u32, n);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, buf.*.ptr, src, n * 4, null, 0, null));
        }
        try d.sync();
    }
    var bad: usize = 0;
    for (host[0], host[1]) |x, y| {
        for (x, y) |a, b| bad += @intFromBool(a != b);
    }
    return bad;
}

/// Runs the cases against attn_k1_decode; error.K1bDiffers if one value differs.
pub fn run(d: *Dev, base: ze.H, kb: ze.H, name: []const u8) !void {
    if (base == null or kb == null) return error.K1bNotLoaded;
    var prng = std.Random.DefaultPrng.init(0x41b1);
    const rnd = prng.random();
    // 16 rows (two 8-row groups) over long and short prefixes, 11 rows (8 and 3), one row, a chain window of 4 rows.
    // The prefixes are no whole number of pages, tiles or 32-key steps, so the slice ends cut a tile and a step.
    const cases = [_]Case{
        .{ .rows = 16, .p0 = 30011, .mt = true },
        .{ .rows = 16, .p0 = 300, .mt = true },
        .{ .rows = 11, .p0 = 4093, .mt = true },
        .{ .rows = 1, .p0 = 1000, .mt = true },
        .{ .rows = 4, .p0 = 7777, .mt = false },
    };
    var bad: usize = 0;
    for (cases) |c| {
        const n = try case(d, base, kb, rnd, c);
        if (n != 0) std.debug.print("  k1b case: {d} rows, {d} prefix keys, MT {}: {d} values differ\n", .{ c.rows, c.p0, c.mt, n });
        bad += n;
    }
    std.debug.print("k1b check: {s} against attn_k1_decode, {d} cases, {d} values differ ({s})\n", .{ name, cases.len, bad, if (bad == 0) "pass" else "FAIL" });
    if (bad != 0) return error.K1bDiffers;
}

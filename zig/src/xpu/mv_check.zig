//! XPU_MV register pipeline picks (lane p4-k-rp): each new kernel against the kernel of its family on this card, at
//! the load. Wide variants run against w4k1_decode (the default wide decode kernel), narrow variants against
//! w4k_dsplit. Random 4-bit weights, scales, bias table, rows and row sums. The shapes cover a ring that ends inside
//! an iteration (K 5120: 80 groups, K 8704: 34 groups a slice) and one that ends on it (K 3072: 12 groups a slice),
//! 16 rows and 11 rows. The new kernel must give the same bits for every output value, or the open fails.
const std = @import("std");
const ze = @import("ze");
const dev = @import("dev.zig");
const q = @import("qwen27.zig");
const Dev = dev.Dev;
const alloc = q.alloc;

const N: usize = 1024;

fn half(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
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

/// The count of output values (of rows x N) whose bits differ between `k` and `ref`.
fn case(d: *Dev, rnd: std.Random, k: ze.H, div: u32, ref: ze.H, ref_div: u32, K: usize, rows: usize) !usize {
    var h: Held = .{};
    defer h.deinit(d);
    const G = K / 64;
    const w = try alloc.alloc(u32, K / 8 * N);
    defer alloc.free(w);
    for (w) |*x| x.* = rnd.int(u32);
    const s = try alloc.alloc(u16, G * N);
    defer alloc.free(s);
    for (s) |*x| x.* = half(0.002 + 0.018 * rnd.float(f32));
    const bg = try alloc.alloc(u16, G * N);
    defer alloc.free(bg);
    for (bg) |*x| x.* = half(rnd.float(f32) - 0.5);
    const a = try alloc.alloc(u16, rows * K);
    defer alloc.free(a);
    for (a) |*x| x.* = half(2.0 * rnd.float(f32) - 1.0);
    const ahi = try alloc.alloc(u16, rows * G);
    defer alloc.free(ahi);
    for (ahi) |*x| x.* = half(2.0 * rnd.float(f32) - 1.0);
    const alo = try alloc.alloc(u16, rows * G);
    defer alloc.free(alo);
    for (alo) |*x| x.* = half(0.002 * (rnd.float(f32) - 0.5));
    const dw = try h.put(d, std.mem.sliceAsBytes(w));
    const ds = try h.put(d, std.mem.sliceAsBytes(s));
    const dbg = try h.put(d, std.mem.sliceAsBytes(bg));
    const da = try h.put(d, std.mem.sliceAsBytes(a));
    const dhi = try h.put(d, std.mem.sliceAsBytes(ahi));
    const dlo = try h.put(d, std.mem.sliceAsBytes(alo));
    const n = rows * N;
    var outs: [2][]u16 = .{ &.{}, &.{} };
    defer for (outs) |o| alloc.free(o);
    const fill = try alloc.alloc(u8, n * 2);
    defer alloc.free(fill);
    const M: i32 = @intCast(rows);
    const mp: i32 = @intCast((rows + 7) / 8 * 8);
    for (&outs, [_]ze.H{ ref, k }, [_]u32{ ref_div, div }, [_]u8{ 0xEE, 0xFF }) |*o, kk, dv, sentinel| {
        @memset(fill, sentinel); // a value the kernel does not write differs between the two
        const dc = try h.put(d, fill);
        try d.launch(kk, .{ .x = 1, .y = @as(u32, @intCast(N)) / dv }, .{ da, dw, ds, dbg, dc, M, @as(i32, @intCast(N)), @as(i32, @intCast(K)), dhi, dlo, mp, @as(i32, @intCast(G)) });
        o.* = try alloc.alloc(u16, n);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, o.*.ptr, dc, n * 2, null, 0, null));
        try d.sync();
    }
    var differ: usize = 0;
    for (outs[0], outs[1]) |x, y| differ += @intFromBool(x != y);
    return differ;
}

/// Kernel `k` (variant `name`, kernel `kname`, `div` columns a work-group) against `ref` on this card.
pub fn run(d: *Dev, k: ze.H, div: u32, ref: ze.H, ref_div: u32, wide: bool, name: []const u8, kname: [*:0]const u8, ref_name: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(0x5250_4b31);
    const rnd = prng.random();
    const ks: []const usize = if (wide) &.{5120} else &.{ 8704, 3072 };
    var differ: usize = 0;
    var total: usize = 0;
    for (ks) |K| {
        for ([_]usize{ 16, 11 }) |rows| {
            differ += try case(d, rnd, k, div, ref, ref_div, K, rows);
            total += rows * N;
        }
    }
    std.debug.print("mv check: {s} ({s}) against {s}: {d} of {d} values differ (K {s}, rows 16 and 11, N {d})\n", .{ name, std.mem.span(kname), ref_name, differ, total, if (wide) "5120" else "8704 and 3072", N });
    if (differ != 0) return error.MvCheckDiffers;
}

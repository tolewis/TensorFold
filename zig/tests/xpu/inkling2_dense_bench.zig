//! tf-xpu-inkling2-dense-bench: Mini's dense projections and norm timed at the served shapes, with GB/s of the weight bytes
//! each call reads. Weights rotate over a pool of copies bigger than the card's caches, as a decode step walks distinct
//! layers; the `warm` column reuses one copy. Inputs are small finite numbers. Run it under the GPU lock.
const std = @import("std");
const xpu = @import("xpu");
const rig_ = @import("contract_rig.zig");
const ze = @import("ze");
const host = @import("inkling2_dense_host");
const images = @import("xpu_contract_images");

const iterations = 200;
const rounds = 7;
/// The card's measured read bandwidth in GB/s (lanes/roof/ROOFLINE.md of the recipe).
const roof = 605.7;
/// Weight copies a rotating run walks: at least this many bytes in all, past the L2.
const pool_bytes: usize = 640 << 20;

const Bench = struct {
    io: std.Io,
    dev: *rig_.Dev,
    k: host.Kernels,
    bufs: std.ArrayList(xpu.DeviceBuffer) = .empty,
    a: std.mem.Allocator,

    fn buf(b: *Bench, bytes: usize, byte: u8) !host.Ref {
        var d = try xpu.DeviceBuffer.alloc(&b.dev.ctx, bytes);
        errdefer d.free();
        try b.bufs.append(b.a, d);
        try d.fill8(b.dev.s, byte);
        return try d.at(0);
    }
    /// A buffer of random bf16 values in [-0.25, 0.25) (copies of `noise`'s bytes, so no pattern for the card to compress).
    fn noisy(b: *Bench, bytes: usize, noise: []const u16) !host.Ref {
        var d = try xpu.DeviceBuffer.alloc(&b.dev.ctx, bytes);
        errdefer d.free();
        try b.bufs.append(b.a, d);
        var at: usize = 0;
        while (at < bytes) {
            const n = @min(noise.len * 2, bytes - at);
            try d.upload(b.dev.s, at, std.mem.sliceAsBytes(noise)[0..n]);
            at += n;
        }
        return try d.at(0);
    }
    fn now(b: *Bench) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(b.io).toNanoseconds()));
    }
    fn release(b: *Bench, keep: usize) void {
        while (b.bufs.items.len > keep) {
            var d = b.bufs.pop().?;
            d.free();
        }
    }

    /// Microseconds of one `dense`, the median of `rounds` rounds of `iterations` launches, the weight copy rotating.
    fn time(b: *Bench, ws: []const host.Ref, x: host.Ref, y: host.Ref, rows: u32, n: u32, k: u32) !f64 {
        for (0..8) |i| try host.dense(&b.k, b.dev.s, ws[i % ws.len], x, y, rows, n, k);
        try b.dev.s.synchronize();
        var t: [rounds]f64 = undefined;
        for (&t) |*r| {
            const t0 = b.now();
            for (0..iterations) |i| try host.dense(&b.k, b.dev.s, ws[i % ws.len], x, y, rows, n, k);
            try b.dev.s.synchronize();
            r.* = (b.now() - t0) / 1e3 / iterations;
        }
        std.mem.sort(f64, &t, {}, std.sort.asc(f64));
        return t[rounds / 2];
    }
};

/// What the card says one launch costs: `count` back-to-back one-row launches of the same call on timestamped events,
/// the kernel's own duration and the time from one launch's start to the next one's.
fn deviceTimes(b: *Bench, ws: []const host.Ref, x: host.Ref, y: host.Ref, rows: u32, n: u32, k: u32) !void {
    const count = 64;
    var gpu = &b.dev.ctx.gpu;
    const events = try gpu.events(count);
    const d = b.dev.ctx.d;
    const kern = b.k.dense[host.variantOf(rows)];
    const groups: ze.GroupCount = .{ .x = (n + b.k.columns - 1) / b.k.columns, .y = 1, .z = 1 };
    // One pass over the pool first, so no weight is read for the first time inside the timing.
    for (0..count + ws.len) |i| {
        const timed = i >= ws.len;
        try kern.setPtr(0, x);
        try kern.setPtr(1, ws[i % ws.len]);
        try kern.setPtr(2, y);
        try kern.setU32(3, rows);
        try kern.setU32(4, n);
        try kern.setU32(5, k);
        try kern.setU32(6, n);
        try d.check(d.api.zeCommandListAppendLaunchKernel(b.dev.s.list, kern.handle, &groups, if (timed) events[i - ws.len] else null, 0, null), "launch");
    }
    try b.dev.s.synchronize();
    var ts: [count]ze.KernelTimestampResult = undefined;
    for (events, &ts) |ev, *t| try d.check(d.api.zeEventQueryKernelTimestamp(ev, t), "timestamp");
    const hz: f64 = @floatFromInt(gpu.props.timerResolution);
    var durs: [count]f64 = undefined;
    var period: f64 = 0;
    var gap: f64 = 0;
    for (ts[8..], 8..) |t, i| {
        durs[i - 8] = @as(f64, @floatFromInt(t.global_end -% t.global_start)) / hz;
        if (i + 1 < count) {
            period += @as(f64, @floatFromInt(ts[i + 1].global_start -% t.global_start)) / hz;
            gap += @as(f64, @floatFromInt(ts[i + 1].global_start -% t.global_end)) / hz;
        }
    }
    const m: f64 = @floatFromInt(count - 8);
    std.mem.sort(f64, durs[0 .. count - 8], {}, std.sort.asc(f64));
    const dur = durs[(count - 8) / 2] * m; // the median launch
    const bytes: f64 = @floatFromInt(@as(usize, n) * k * 2);
    std.debug.print("BENCH device {d} x {d}, {d} rows: median kernel {d:.2} us ({d:.1} GB/s), start to next start {d:.2} us, gap {d:.2} us\n", .{ n, k, rows, dur / m * 1e6, bytes / (dur / m) / 1e9, period / (m - 1) * 1e6, gap / (m - 1) * 1e6 });
}

pub fn main(init: std.process.Init) !u8 {
    var dev: rig_.Dev = undefined;
    dev.init(null) catch |e| {
        std.debug.print("FAIL no card opened ({t})\n", .{e});
        return 1;
    };
    defer dev.deinit();
    // `tf-xpu-inkling2-dense-bench [IMAGE.bin [quick|full [WORK-GROUP]]]`: time another build of dense.cl (with its DENSE_WG), quick: only the two big shapes
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var quick = false;
    if (args.len > 1) {
        const file = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.gpa, .limited(64 << 20));
        const bin = try init.gpa.alignedAlloc(u8, .@"8", file.len);
        @memcpy(bin, file);
        try dev.addImage("inkling2_dense", bin);
        quick = args.len > 2 and std.mem.eql(u8, args[2], "quick");
    } else try dev.addImage("inkling2_dense", &images.inkling2_dense);
    const threads: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else host.matvec_threads;
    var b: Bench = .{ .io = init.io, .dev = &dev, .k = try host.Kernels.loadAt(dev.named("inkling2_dense").?.*, false, threads), .a = init.gpa };
    defer {
        b.k.deinit();
        for (b.bufs.items) |*d| d.free();
        b.bufs.deinit(b.a);
    }
    std.debug.print("BENCH Mini dense XPU kernels, median of {d} rounds of {d} launches, card 0x{x}, roof {d:.1} GB/s\n", .{ rounds, iterations, dev.ctx.gpu.props.deviceId, roof });
    const noise = try b.a.alloc(u16, 8 << 20);
    defer b.a.free(noise);
    var prng = std.Random.DefaultPrng.init(0x7b70);
    for (noise) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 0.5 - 0.25)) >> 16);
    const shapes = [_][3]u32{ .{ 3072, 2304, 0 }, .{ 1024, 2304, 0 }, .{ 384, 2304, 0 }, .{ 24, 2304, 0 }, .{ 2304, 3072, 0 } };
    const names = [_][]const u8{ "wq, gate", "wk, wv", "wr", "sink", "wo" };
    for (shapes, names) |sh, name| {
        if (quick and sh[0] != 3072 and sh[0] != 2304) continue;
        const n = sh[0];
        const k = sh[1];
        const bytes = @as(usize, n) * k * 2;
        const copies = @max(2, (pool_bytes + bytes - 1) / bytes);
        const keep = b.bufs.items.len;
        const x = try b.noisy(32 * @as(usize, k) * 2, noise);
        const y = try b.buf(32 * @as(usize, n) * 2, 0);
        var ws = try b.a.alloc(host.Ref, copies);
        defer b.a.free(ws);
        for (ws) |*w| w.* = try b.noisy(bytes, noise);
        try dev.s.synchronize();
        for ([_]u32{ 1, 2, 4, 8, 16, 32 }) |rows| {
            if (rows == 1 or rows == 4 or rows == 8 or rows == 32) try deviceTimes(&b, ws, x, y, rows, n, k);
            if (quick and rows != 1 and rows != 4 and rows != 8 and rows != 32) continue;
            const cold = try b.time(ws, x, y, rows, n, k);
            const warm = try b.time(ws[0..1], x, y, rows, n, k);
            const gcold = @as(f64, @floatFromInt(bytes)) / (cold * 1e-6) / 1e9;
            const gwarm = @as(f64, @floatFromInt(bytes)) / (warm * 1e-6) / 1e9;
            std.debug.print("BENCH dense {s} {d} x {d}, {d} rows: {d:.1} us, {d:.1} MB, {d:.1} GB/s ({d:.3} of {d:.1}); warm {d:.1} us, {d:.1} GB/s\n", .{ name, n, k, rows, cold, @as(f64, @floatFromInt(bytes)) / 1e6, gcold, gcold / roof, roof, warm, gwarm });
        }
        b.release(keep);
    }
    if (quick) return 0;
    // the norm: 2,304 FP32 values a row in, bf16 out
    const x = try b.buf(256 * 2304 * 4, 0x3c);
    const w = try b.buf(2304 * 2, 0x3c);
    const y = try b.buf(256 * 2304 * 2, 0);
    try dev.s.synchronize();
    for ([_]u32{ 1, 8, 32, 256 }) |rows| {
        for (0..8) |_| try host.norm(&b.k, dev.s, x, w, y, rows);
        try dev.s.synchronize();
        const t0 = b.now();
        for (0..iterations) |_| try host.norm(&b.k, dev.s, x, w, y, rows);
        try dev.s.synchronize();
        const us = (b.now() - t0) / 1e3 / iterations;
        const mb = @as(f64, @floatFromInt(rows)) * 2304 * 6 / 1e6;
        std.debug.print("BENCH norm {d} rows x 2304: {d:.1} us, {d:.2} MB, {d:.1} GB/s\n", .{ rows, us, mb, mb * 1e6 / (us * 1e-6) / 1e9 });
    }
    return 0;
}

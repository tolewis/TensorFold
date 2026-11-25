//! tf-xpu-inkling2-dense-q8-bench: Mini's Q8_0-weight dense projections and head timed at the served shapes, with GB/s of
//! the weight bytes (codes and scales) each call reads. Weights rotate over a pool of copies bigger than the card's
//! caches, as a decode step walks distinct layers. Before the timing each shape is checked against a float64 dot of the
//! decoded weights. Run it under the GPU lock.
//!
//!   tf-xpu-inkling2-dense-q8-bench [IMAGE.bin [quick|full [WORK-GROUP]]]
//!
//! An image path times another build of dense_q8.cl (with its Q8_WG); `quick` times only the one-row calls.
const std = @import("std");
const xpu = @import("xpu");
const rig_ = @import("contract_rig.zig");
const ze = @import("ze");
const host = @import("inkling2_dense_q8_host");
const q8 = @import("inkling2_q8");
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

    fn alloc(b: *Bench, bytes: usize) !xpu.DeviceBuffer {
        var d = try xpu.DeviceBuffer.alloc(&b.dev.ctx, bytes);
        errdefer d.free();
        try b.bufs.append(b.a, d);
        return d;
    }
    fn buf(b: *Bench, bytes: usize, byte: u8) !host.Ref {
        var d = try b.alloc(bytes);
        try d.fill8(b.dev.s, byte);
        return try d.at(0);
    }
    fn noisyBf16(b: *Bench, bytes: usize, noise: []const u16) !host.Ref {
        var d = try b.alloc(bytes);
        var at: usize = 0;
        while (at < bytes) {
            const n = @min(noise.len * 2, bytes - at);
            try d.upload(b.dev.s, at, std.mem.sliceAsBytes(noise)[0..n]);
            at += n;
        }
        return try d.at(0);
    }
    /// A weight copy of `n` rows of `k` values: random codes and scales in a plausible range.
    fn weight(b: *Bench, n: usize, k: usize, codes: []const u8, scales: []const u16, dpas: bool) !host.Ref {
        var d = try b.alloc(if (dpas) q8.dpasBytes(n, k) else q8.bytes(n, k));
        var at: usize = 0;
        const plane = n * k;
        while (at < plane) {
            const m = @min(codes.len, plane - at);
            try d.upload(b.dev.s, at, codes[0..m]);
            at += m;
        }
        at = 0;
        const sbytes = n * (k / 32) * 2;
        while (at < sbytes) {
            const m = @min(scales.len * 2, sbytes - at);
            try d.upload(b.dev.s, plane + at, std.mem.sliceAsBytes(scales)[0..m]);
            at += m;
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

    fn run(b: *Bench, head: bool, w: host.Ref, x: host.Ref, y: host.Ref, rows: u32, n: u32, k: u32) !void {
        if (head) try host.head(&b.k, b.dev.s, w, x, y, rows, n) else try host.dense(&b.k, b.dev.s, w, x, y, rows, n, k);
    }

    /// Microseconds of one call, the median of `rounds` rounds of `iterations` launches, the weight copy rotating.
    fn time(b: *Bench, head: bool, ws: []const host.Ref, x: host.Ref, y: host.Ref, rows: u32, n: u32, k: u32) !f64 {
        for (0..8) |i| try b.run(head, ws[i % ws.len], x, y, rows, n, k);
        try b.dev.s.synchronize();
        var t: [rounds]f64 = undefined;
        for (&t) |*r| {
            const t0 = b.now();
            for (0..iterations) |i| try b.run(head, ws[i % ws.len], x, y, rows, n, k);
            try b.dev.s.synchronize();
            r.* = (b.now() - t0) / 1e3 / iterations;
        }
        std.mem.sort(f64, &t, {}, std.sort.asc(f64));
        return t[rounds / 2];
    }
};

/// What the card says one launch costs: the kernel's own duration (median of timestamped launches).
fn deviceTime(b: *Bench, head: bool, ws: []const host.Ref, x: host.Ref, y: host.Ref, rows: u32, n: u32, k: u32) !f64 {
    const count = 64;
    var gpu = &b.dev.ctx.gpu;
    const events = try gpu.events(count);
    const d = b.dev.ctx.d;
    const kern = if (head) b.k.head_dpas else b.k.dense[host.variantOf(rows)];
    const groups: ze.GroupCount = if (head) .{ .x = (n + 15) / 16, .y = 1, .z = 1 } else .{ .x = (n + b.k.columns - 1) / b.k.columns, .y = 1, .z = 1 };
    var qc: host.Ref = null;
    var qs: host.Ref = null;
    if (head) {
        const sc = try host.actScratch(b.dev.s);
        qc = sc.qc;
        qs = sc.qs;
        try xpu.launch.dispatch(b.k.quant, b.dev.s, .{ rows, 1, 1 }, .{ x, qc, qs, rows, host.hidden });
    }
    for (0..count + ws.len) |i| {
        const timed = i >= ws.len;
        if (head) {
            try kern.setPtr(0, qc);
            try kern.setPtr(1, qs);
            try kern.setPtr(2, ws[i % ws.len]);
            try kern.setPtr(3, y);
            try kern.setPtr(4, x);
            try kern.setU32(5, rows);
            try kern.setU32(6, n);
        } else {
            try kern.setPtr(0, x);
            try kern.setPtr(1, ws[i % ws.len]);
            try kern.setPtr(2, y);
            try kern.setU32(3, rows);
            try kern.setU32(4, n);
            try kern.setU32(5, k);
            try kern.setU32(6, n);
        }
        try d.check(d.api.zeCommandListAppendLaunchKernel(b.dev.s.list, kern.handle, &groups, if (timed) events[i - ws.len] else null, 0, null), "launch");
    }
    try b.dev.s.synchronize();
    var ts: [count]ze.KernelTimestampResult = undefined;
    for (events, &ts) |ev, *t| try d.check(d.api.zeEventQueryKernelTimestamp(ev, t), "timestamp");
    const hz: f64 = @floatFromInt(gpu.props.timerResolution);
    var durs: [count]f64 = undefined;
    for (ts[8..], 8..) |t, i| durs[i - 8] = @as(f64, @floatFromInt(t.global_end -% t.global_start)) / hz;
    std.mem.sort(f64, durs[0 .. count - 8], {}, std.sort.asc(f64));
    return durs[(count - 8) / 2];
}

fn bf(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}
fn unbf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// The kernel against float64 on a small matrix of random weights, for `rows` rows; returns the worst error in bf16 ulps.
fn check(b: *Bench, head: bool, rows: u32, n: u32, k: u32) !f64 {
    const a = b.a;
    var prng = std.Random.DefaultPrng.init(0x51c0 + rows + n);
    const rng = prng.random();
    const wsrc = try a.alloc(u16, @as(usize, n) * k);
    defer a.free(wsrc);
    for (wsrc) |*v| v.* = bf((rng.float(f32) - 0.5) * 0.5);
    const codes = try a.alloc(u8, @as(usize, n) * k);
    defer a.free(codes);
    const scales = try a.alloc(u16, @as(usize, n) * (k / 32));
    defer a.free(scales);
    q8.encodeRows(wsrc, n, k, codes, scales);
    const xs = try a.alloc(u16, @as(usize, rows) * k);
    defer a.free(xs);
    for (xs) |*v| v.* = bf((rng.float(f32) - 0.5) * 4);
    const nbytes = if (head) q8.dpasBytes(n, k) else q8.bytes(n, k);
    var wd = try b.alloc(nbytes);
    if (head) {
        const tile = try a.alloc(u8, nbytes);
        defer a.free(tile);
        q8.repackDpas(codes, scales, n, k, tile);
        try wd.upload(b.dev.s, 0, tile);
    } else {
        try wd.upload(b.dev.s, 0, codes);
        try wd.upload(b.dev.s, @as(usize, n) * k, std.mem.sliceAsBytes(scales));
    }
    var xd = try b.alloc(xs.len * 2);
    try xd.upload(b.dev.s, 0, std.mem.sliceAsBytes(xs));
    const osize: usize = if (head) 4 else 2;
    var yd = try b.alloc(@as(usize, rows) * n * osize);
    try b.dev.s.synchronize();
    try b.run(head, try wd.at(0), try xd.at(0), try yd.at(0), rows, n, k);
    try b.dev.s.synchronize();
    const got = try a.alloc(u8, @as(usize, rows) * n * osize);
    defer a.free(got);
    try yd.download(b.dev.s, 0, got);
    var worst: f64 = 0;
    for (0..rows) |r| for (0..n) |j| {
        var dot: f64 = 0;
        for (0..k / 32) |blk| {
            for (0..32) |e| dot += q8.decode(codes, scales, k, j, blk, e) * unbf(xs[r * k + blk * 32 + e]);
        }
        const o = r * n + j;
        var want: f64 = undefined;
        var have: f64 = undefined;
        if (head) {
            const f: f32 = @bitCast(std.mem.readInt(u32, got[o * 4 ..][0..4], .little));
            want = 30.0 * std.math.tanh(unbf(bf(@floatCast(dot))) / 30.0);
            have = f;
        } else {
            have = unbf(std.mem.readInt(u16, got[o * 2 ..][0..2], .little));
            want = dot;
        }
        const ulp = std.math.pow(f64, 2, @floor(std.math.log2(@max(@abs(want), 1e-30))) - 7);
        worst = @max(worst, @abs(have - want) / ulp);
    };
    return worst;
}

pub fn main(init: std.process.Init) !u8 {
    var dev: rig_.Dev = undefined;
    dev.init(null) catch |e| {
        std.debug.print("FAIL no card opened ({t})\n", .{e});
        return 1;
    };
    defer dev.deinit();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var quick = false;
    if (args.len > 1 and !std.mem.eql(u8, args[1], "-")) {
        const file = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.gpa, .limited(64 << 20));
        const bin = try init.gpa.alignedAlloc(u8, .@"8", file.len);
        @memcpy(bin, file);
        try dev.addImage("inkling2_dense_q8", bin);
    } else try dev.addImage("inkling2_dense_q8", &images.inkling2_dense_q8);
    quick = args.len > 2 and std.mem.eql(u8, args[2], "quick");
    const threads: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else host.group_threads;
    var b: Bench = .{ .io = init.io, .dev = &dev, .k = try host.Kernels.loadAt(dev.named("inkling2_dense_q8").?.*, false, threads), .a = init.gpa };
    defer {
        b.k.deinit();
        for (b.bufs.items) |*d| d.free();
        b.bufs.deinit(b.a);
    }
    std.debug.print("BENCH Mini Q8_0 dense XPU kernels, median of {d} rounds of {d} launches, card 0x{x}, roof {d:.1} GB/s\n", .{ rounds, iterations, dev.ctx.gpu.props.deviceId, roof });
    // correctness first
    for ([_][3]u32{ .{ 1, 24, 2304 }, .{ 1, 100, 3072 }, .{ 3, 50, 2304 }, .{ 8, 70, 4608 }, .{ 32, 40, 2304 }, .{ 63, 17, 2304 } }) |c| {
        const keep = b.bufs.items.len;
        const ulps = try check(&b, false, c[0], c[1], c[2]);
        const hu = try check(&b, true, c[0], c[1], 2304);
        std.debug.print("CHECK {d} rows x {d} x {d}: dense worst {d:.3} bf16 ulps, head {d:.3}\n", .{ c[0], c[1], c[2], ulps, hu });
        b.release(keep);
    }
    const noise = try b.a.alloc(u16, 8 << 20);
    defer b.a.free(noise);
    var prng = std.Random.DefaultPrng.init(0x7b70);
    for (noise) |*v| v.* = bf(prng.random().float(f32) * 0.5 - 0.25);
    const codes = try b.a.alloc(u8, 16 << 20);
    defer b.a.free(codes);
    prng.random().bytes(codes);
    for (codes) |*c| c.* = if (c.* == 0x80) 0 else c.*;
    const scales = try b.a.alloc(u16, 1 << 20);
    defer b.a.free(scales);
    for (scales) |*s| s.* = @bitCast(@as(f16, @floatCast(prng.random().float(f32) * 0.01)));
    const Shape = struct { name: []const u8, n: u32, k: u32, head: bool = false };
    const shapes = [_]Shape{
        .{ .name = "wq, gate", .n = 3072, .k = 2304 },
        .{ .name = "wk, wv", .n = 1024, .k = 2304 },
        .{ .name = "wr", .n = 384, .k = 2304 },
        .{ .name = "wo", .n = 2304, .k = 3072 },
        .{ .name = "head", .n = 200058, .k = 2304, .head = true },
    };
    for (shapes) |sh| {
        if (quick and sh.n == 384) continue;
        const bytes = if (sh.head) q8.dpasBytes(sh.n, sh.k) else q8.bytes(sh.n, sh.k);
        const copies = @max(2, (pool_bytes + bytes - 1) / bytes);
        const keep = b.bufs.items.len;
        const x = try b.noisyBf16(32 * @as(usize, sh.k) * 2, noise);
        const y = try b.buf(32 * @as(usize, sh.n) * 4, 0);
        var ws = try b.a.alloc(host.Ref, copies);
        defer b.a.free(ws);
        for (ws) |*w| w.* = try b.weight(sh.n, sh.k, codes, scales, sh.head);
        try dev.s.synchronize();
        for ([_]u32{ 1, 2, 4, 8, 16, 32 }) |rows| {
            if (quick and rows != 1 and rows != 8) continue;
            if (sh.head and rows > 8) continue;
            const dt = try deviceTime(&b, sh.head, ws, x, y, rows, sh.n, sh.k);
            const cold = try b.time(sh.head, ws, x, y, rows, sh.n, sh.k);
            const warm = try b.time(sh.head, ws[0..1], x, y, rows, sh.n, sh.k);
            const fb: f64 = @floatFromInt(bytes);
            const gcold = fb / (cold * 1e-6) / 1e9;
            std.debug.print("BENCH {s} {d} x {d}, {d} rows: {d:.1} us, {d:.1} MB, {d:.1} GB/s ({d:.3} of {d:.1}); kernel {d:.1} us {d:.1} GB/s; warm {d:.1} us\n", .{ sh.name, sh.n, sh.k, rows, cold, fb / 1e6, gcold, gcold / roof, roof, dt * 1e6, fb / dt / 1e9, warm });
        }
        b.release(keep);
    }
    return 0;
}

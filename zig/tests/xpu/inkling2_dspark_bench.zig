//! tf-xpu-dspark-bench: Mini DSpark's XPU kernels timed at the served shapes, with the bytes each one reads as GB/s.
//! Inputs are small finite BF16 numbers, so the timings do not hide behind NaN fast paths. Run it under the GPU lock.
const std = @import("std");
const xpu = @import("xpu");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_dspark_host");
const images = @import("xpu_contract_images");

const H: usize = host.hidden;
const KV: usize = host.kv_width;
const V: usize = host.vocab;
const iterations = 20;
/// The card's measured read bandwidth in GB/s (lanes/roof/ROOFLINE.md of the recipe).
const roof = 605.7;

const Bench = struct {
    io: std.Io,
    dev: *rig_.Dev,
    k: host.Kernels,
    bufs: std.ArrayList(xpu.DeviceBuffer) = .empty,
    a: std.mem.Allocator,
    round_us: f64 = 0,

    fn buf(b: *Bench, bytes: usize) !host.Ref {
        var d = try xpu.DeviceBuffer.alloc(&b.dev.ctx, bytes);
        errdefer d.free();
        try b.bufs.append(b.a, d);
        // Fill with the BF16 pattern 0x3c3c (about 0.011): finite and small, so sums stay finite.
        try d.fill8(b.dev.s, 0x3c);
        return try d.at(0);
    }
    fn now(b: *Bench) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(b.io).toNanoseconds()));
    }
    /// Mean microseconds of one call: three warm-up calls, then `iterations` queued back to back.
    fn time(b: *Bench, comptime call: anytype, args: anytype) !f64 {
        for (0..3) |_| try @call(.auto, call, .{ &b.k, b.dev.s } ++ args);
        try b.dev.s.synchronize();
        const t0 = b.now();
        for (0..iterations) |_| try @call(.auto, call, .{ &b.k, b.dev.s } ++ args);
        try b.dev.s.synchronize();
        return (b.now() - t0) / 1e3 / iterations;
    }
    fn line(b: *Bench, name: []const u8, shape: []const u8, us: f64, bytes: f64, rounds: f64) void {
        b.round_us += us * rounds;
        if (bytes > 0) {
            const gbs = bytes / (us * 1e-6) / 1e9;
            std.debug.print("BENCH {s} {s}: {d:.1} us, {d:.1} MB, {d:.1} GB/s ({d:.3} of {d:.1})\n", .{ name, shape, us, bytes / 1e6, gbs, gbs / roof, roof });
        } else std.debug.print("BENCH {s} {s}: {d:.1} us\n", .{ name, shape, us });
    }
};

pub fn main(init: std.process.Init) !u8 {
    var dev: rig_.Dev = undefined;
    dev.init(null) catch |e| {
        std.debug.print("FAIL no card opened ({t})\n", .{e});
        return 1;
    };
    defer dev.deinit();
    try dev.addImage("inkling2_dspark", &images.inkling2_dspark);
    var b: Bench = .{ .io = init.io, .dev = &dev, .k = try host.Kernels.load(dev.named("inkling2_dspark").?.*, false), .a = init.gpa };
    defer {
        b.k.deinit();
        for (b.bufs.items) |*d| d.free();
        b.bufs.deinit(b.a);
    }
    const unembed = try b.buf(V * H * 2);
    const phi = try b.buf(V * host.rank * 2);
    const markov = try b.buf(V * host.rank * 2);
    const vecv = try b.buf(V * 2);
    const base_out = try b.buf(V * 8 * 2);
    const rows_a = try b.buf(8 * H * 2);
    const rows_b = try b.buf(8 * H * 2);
    const rows_c = try b.buf(8 * H * 2);
    const rows_d = try b.buf(8 * H * 2);
    const inter_a = try b.buf(8 * host.intermediate * 2);
    const inter_b = try b.buf(8 * host.intermediate * 2);
    const inter_c = try b.buf(8 * host.intermediate * 2);
    const rnn = try b.buf(host.rnn_width * 5120 * 2);
    const small = try b.buf(host.rnn_width * 8 * 4);
    const small2 = try b.buf(host.rnn_width * 8 * 4);
    const kappa_out = try b.buf(8 * host.rnn_width * 4);
    const qbuf = try b.buf(8 * host.query_width * 2);
    const qout = try b.buf(8 * host.query_width * 2);
    const kvbuf = try b.buf(8 * KV * 2);
    const kvbuf2 = try b.buf(8 * KV * 2);
    const kvout = try b.buf(8 * KV * 2);
    const taps = try b.buf(64 * 4 * H * 2);
    const fused = try b.buf(4 * 64 * H * 2);
    const fuse_w = try b.buf(4 * H * H * 2);
    try dev.s.synchronize();

    std.debug.print("BENCH Mini DSpark XPU kernels, mean of {d} launches, card 0x{x}\n", .{ iterations, dev.ctx.gpu.props.deviceId });
    // A drafting round, in launches: base once (7 rows), kappa once, then seven steps of recurrent and logits, three
    // layers of qkv, attention, two norms and swiglu, and the first norm. The dense projections are not kernels of this file.
    b.line("base", "7 rows x 200058", try b.time(host.base, .{ unembed, rows_a, base_out, @as(u32, 7), @as(u32, host.vocab) }), @floatFromInt(V * H * 2), 1);
    b.line("base", "1 row x 200058", try b.time(host.base, .{ unembed, rows_a, base_out, @as(u32, 1), @as(u32, host.vocab) }), @floatFromInt(V * H * 2), 0);
    b.line("logits", "200058", try b.time(host.finalLogits, .{ vecv, phi, vecv, small, vecv }), @floatFromInt(V * host.rank * 2 + V * 6), 7);
    b.line("kappa", "7 rows x 768", try b.time(host.kappa, .{ rows_a, rows_b, rnn, small, kappa_out, @as(u32, 7) }), @floatFromInt(host.rnn_width * 4608 * 2), 1);
    b.line("recurrent", "256", try b.time(host.recurrentStep, .{ small2, kappa_out, rnn, markov, small, small2, small, @as(u32, 1) }), 0, 7);
    b.line("init_rows", "8 x 2304", try b.time(host.initializeRows, .{ rows_a, rows_b, rows_c }), 0, 1);
    b.line("residual_norm", "8 x 2304", try b.time(host.residualNorm, .{ rows_a, rows_b, rows_c, rows_d, rows_a, @as(u32, 8), false }), 0, 7);
    b.line("swiglu", "8 x 9216", try b.time(host.swiglu, .{ inter_a, inter_b, inter_c, @as(u32, 8 * host.intermediate) }), @floatFromInt(2 * 8 * host.intermediate * 2), 3);
    b.line("block_qkv", "8 rows", try b.time(host.blockQkv, .{ qbuf, kvbuf, kvbuf2, small, small, qout, kvout, kvbuf2, @as(u32, 100) }), 0, 3);
    b.line("fuse_taps", "64 rows", try b.time(host.fuseTaps, .{ taps, fuse_w, small, fused, small, @as(u32, 64) }), @floatFromInt(64 * 4 * H * 2), 0);
    b.line("context_kv", "64 rows", try b.time(host.commitContext, .{ kvbuf, kvbuf2, small, fused, fused, @as(u32, 1000), @as(u32, 8), @as(u32, 1152) }), 0, 0);
    var name: [64]u8 = undefined;
    for ([_]usize{ 1024, 5000, 20001, 100000 }) |committed| {
        const slots: usize = if (committed > 4096) 1152 else committed;
        const ck = try b.buf(slots * KV * 2);
        const cv = try b.buf(slots * KV * 2);
        const scores = try b.buf(8 * 24 * (committed + 8) * 4);
        try dev.s.synchronize();
        const label = try std.fmt.bufPrint(&name, "{d} committed", .{committed});
        const ring: u32 = @intCast(slots);
        const sliding = try b.time(host.attend, .{ qbuf, kvbuf, kvbuf2, ck, cv, scores, qout, @as(u32, @intCast(committed)), ring, true, @as(u32, 0) });
        b.line("attention sliding", label, sliding, @as(f64, @floatFromInt(@min(committed, 1024) + 8)) * KV * 2 * 2, 0);
        const global = try b.time(host.attend, .{ qbuf, kvbuf, kvbuf2, ck, cv, scores, qout, @as(u32, @intCast(committed)), ring, false, @as(u32, 0) });
        b.line("attention global", label, global, @as(f64, @floatFromInt(committed + 8)) * KV * 2 * 2, 0);
        if (committed == 1024) b.round_us += 2 * sliding + global;
    }
    std.debug.print("BENCH one drafting round at 1,024 tokens of context, these kernels only: {d:.0} us\n", .{b.round_us});
    return 0;
}

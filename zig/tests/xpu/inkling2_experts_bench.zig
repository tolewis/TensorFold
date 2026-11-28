//! tf-xpu-inkling2-experts-bench: the decode kernels' memory bandwidth at Inkling 2 Mini's served shapes.
//! Every call routes to experts the call before did not read (the weights of all 128 experts are far beyond the cache),
//! so the time is the bytes of the weights and scales the call reads over the card's bandwidth.
const std = @import("std");
const xpu = @import("xpu");
const host = @import("inkling2_xpu_experts");
const images = @import("xpu_contract_images");

const HID: u32 = host.hidden;
const FF: u32 = host.intermediate;
const E: u32 = host.expert_count;
const TOPK: u32 = host.top_k;

/// The card's peak memory bandwidth in GB/s that the lane measured with a streaming read.
const peak_gbs = 605.7;

fn ms(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

const Case = struct { fmt: host.Format, gu: bool, rows: u32 };

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    var drv = try xpu.Driver.open();
    defer drv.close();
    var ctx = try xpu.Context.init(&drv, null);
    defer ctx.deinit();
    var s = try xpu.Stream.init(&ctx);
    defer s.deinit();
    var module = try xpu.Module.loadNative(&ctx, &images.inkling2_experts);
    defer module.unload();
    var ks = try host.Kernels.load(&module, host.default_prefix, null);
    defer ks.deinit();

    var prng = std.Random.DefaultPrng.init(0xb70);
    const rng = prng.random();
    var worst: f64 = 1e9;
    var best: f64 = 0;
    for ([_]host.Format{ .nvfp4, .mxfp8 }) |fmt| for ([_]bool{ true, false }) |gu| {
        const k: u32 = if (gu) HID else FF;
        const n: u32 = if (gu) FF else HID;
        const rpe: usize = if (gu) 2 * @as(usize, n) else n;
        const code_bytes: usize = if (fmt == .nvfp4) k / 2 else k;
        const block: usize = if (fmt == .nvfp4) 16 else 32;
        const per_w = rpe * code_bytes;
        const per_s = rpe * (k / block);
        var wbuf = try xpu.DeviceBuffer.alloc(&ctx, E * per_w);
        defer wbuf.free();
        var sbuf = try xpu.DeviceBuffer.alloc(&ctx, E * per_s);
        defer sbuf.free();
        // random bytes in 64 MiB pieces (one upload buffer) so no matrix is a repeat of its neighbour
        const piece = try a.alloc(u8, 64 << 20);
        defer a.free(piece);
        var at: usize = 0;
        while (at < wbuf.len) : (at += piece.len) {
            rng.bytes(piece);
            const n_bytes = @min(piece.len, wbuf.len - at);
            try wbuf.upload(s, at, piece[0..n_bytes]);
        }
        at = 0;
        while (at < sbuf.len) : (at += piece.len) {
            const n_bytes = @min(piece.len, sbuf.len - at);
            for (piece[0..n_bytes]) |*v| v.* = if (fmt == .nvfp4) 0x30 + rng.uintLessThan(u8, 16) else 120 + rng.uintLessThan(u8, 5);
            try sbuf.upload(s, at, piece[0..n_bytes]);
        }
        var amax = try xpu.DeviceBuffer.alloc(&ctx, E * 4);
        defer amax.free();
        var den = try xpu.DeviceBuffer.alloc(&ctx, 4);
        defer den.free();
        const ones = try a.alloc(f32, E);
        defer a.free(ones);
        @memset(ones, 2688);
        try amax.upload(s, 0, std.mem.sliceAsBytes(ones));
        try den.upload(s, 0, std.mem.sliceAsBytes(ones[0..1]));
        for ([_]u32{ 1, 4, 16 }) |rows| {
            const routes = rows * TOPK;
            const ids = try a.alloc(i32, @as(usize, routes) * 64); // 64 distinct id sets, each calls reads its own
            defer a.free(ids);
            for (ids, 0..) |*id, i| id.* = @intCast((i + (i / routes) * 8) % E);
            var idb = try xpu.DeviceBuffer.alloc(&ctx, ids.len * 4);
            defer idb.free();
            try idb.upload(s, 0, std.mem.sliceAsBytes(ids));
            const x_len: usize = if (gu) rows * k else routes * k;
            var xb = try xpu.DeviceBuffer.alloc(&ctx, x_len * 2);
            defer xb.free();
            const xs = try a.alloc(u16, x_len);
            defer a.free(xs);
            for (xs) |*v| v.* = 0x3c00 | rng.uintLessThan(u16, 0x100); // bf16 near 0.0078 to 0.0156
            try xb.upload(s, 0, std.mem.sliceAsBytes(xs));
            var ob = try xpu.DeviceBuffer.alloc(&ctx, @as(usize, routes) * n * 2);
            defer ob.free();
            const cl: host.Call = .{ .rows = rows, .k = k, .n = n, .format = fmt };
            const reps: usize = 64;
            // the distinct experts each call reads: its routes' ids, once
            var unique: usize = 0;
            {
                var seen: [E]bool = @splat(false);
                for (ids[0..routes]) |id| {
                    if (!seen[@intCast(id)]) unique += 1;
                    seen[@intCast(id)] = true;
                }
            }
            var round: usize = 0;
            var best_ns: i96 = std.math.maxInt(i96);
            while (round < 5) : (round += 1) {
                try s.synchronize();
                const t0 = ms(init.io);
                for (0..reps) |r| {
                    const idp = try idb.at(r * routes * 4);
                    try launchOne(&ks, s, fmt, gu, xb.ptr, wbuf.ptr, sbuf.ptr, amax.ptr, den.ptr, idp, ob.ptr, cl);
                }
                try s.synchronize();
                const dt = ms(init.io) - t0;
                best_ns = @min(best_ns, dt);
            }
            // the output of the last call (ids set 63), as sha256: a variant's bits against the base
            const got = try a.alloc(u8, ob.len);
            defer a.free(got);
            try ob.download(s, 0, got);
            var hash: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(got, &hash, .{});
            std.debug.print("{s} {s} rows {d}: output sha256 {x}\n", .{ @tagName(fmt), if (gu) "gu  " else "down", rows, hash[0..8] });
            const bytes = @as(f64, @floatFromInt(unique)) * @as(f64, @floatFromInt(per_w + per_s)) * @as(f64, @floatFromInt(reps));
            const gbs = bytes / @as(f64, @floatFromInt(best_ns));
            if (rows == 1) {
                worst = @min(worst, gbs);
                best = @max(best, gbs);
            }
            std.debug.print("{s} {s} rows {d}: {d} experts a call, {d:.1} MB a call, {d:.1} us a call, {d:.1} GB/s, {d:.3} of {d:.1}\n", .{
                @tagName(fmt), if (gu) "gu  " else "down", rows, unique, bytes / reps / 1e6, @as(f64, @floatFromInt(best_ns)) / reps / 1e3, gbs, gbs / peak_gbs, peak_gbs,
            });
        }
    };
    std.debug.print("decode, 8 experts a token: {d:.1} to {d:.1} GB/s ({d:.3} to {d:.3} of {d:.1})\n", .{ worst, best, worst / peak_gbs, best / peak_gbs, peak_gbs });
    return 0;
}

fn launchOne(ks: *const host.Kernels, s: xpu.Stream, fmt: host.Format, gu: bool, x: ?*anyopaque, w: ?*anyopaque, sc: ?*anyopaque, amax: ?*anyopaque, den: ?*anyopaque, ids: ?*anyopaque, out: ?*anyopaque, cl: host.Call) !void {
    switch (fmt) {
        .nvfp4 => if (gu) try ks.nvfp4Gu(s, x, w, sc, amax, den, ids, out, cl) else try ks.nvfp4Down(s, x, w, sc, amax, den, ids, out, cl),
        .mxfp8 => if (gu) try ks.mxfp8Gu(s, x, w, sc, ids, out, cl) else try ks.mxfp8Down(s, x, w, sc, ids, out, cl),
    }
}

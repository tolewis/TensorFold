//! Device test of Inkling 2 Mini's sampling candidate selection (tf_ink2_sample_top): for rows of logits with many ties,
//! both zeros, infinities and a flat floor, the device's candidates must equal the first n entries of the host's sort of
//! the whole row (value descending, ties to the lower id), for every candidate count and temperature tried. No model is
//! loaded. Then the draws: `drawTop` from the device's candidates must equal `draw` from the row, for many seeds, positions
//! and sampling rules (top_k set and off, top_p, min_p).
//!   tf-xpu-inkling2-sample [--rows N]
const std = @import("std");
const xpu = @import("xpu");
const ink = @import("inkling2_xpu");
const lanes = @import("lanes");

pub const panic = std.debug.FullPanic(xpu.rt.panicDrain);

const vocab = 200058;
const Cand = lanes.gpu_rule.Cand;

fn fill(prng: *std.Random.DefaultPrng, row: []f32, kind: usize) void {
    const r = prng.random();
    for (row, 0..) |*x, i| {
        x.* = switch (kind) {
            // a peaked row: a few large logits over a normal floor
            0 => r.floatNorm(f32) * 2.0 - 4.0 + (if (i % 9973 == 0) @as(f32, 18.0) else 0.0),
            // heavy ties: values on a coarse grid
            1 => @floatFromInt(@as(i32, @intCast(r.uintLessThan(u32, 41))) - 20),
            // ties among both zeros and infinities
            2 => switch (r.uintLessThan(u32, 6)) {
                0 => 0.0,
                1 => -0.0,
                2 => -std.math.inf(f32),
                3 => 1.5,
                4 => -1.5,
                else => r.floatNorm(f32),
            },
            // one flat floor with a few raised entries
            else => if (i % 4099 == 7) @as(f32, 3.0) else -1.0,
        };
    }
}

fn hostTop(gpa: std.mem.Allocator, row: []const f32, inv_t: f32, n: usize) ![]Cand {
    const all = try gpa.alloc(Cand, row.len);
    defer gpa.free(all);
    for (all, row, 0..) |*c, l, i| c.* = .{ .v = l * inv_t, .i = @intCast(i) };
    std.mem.sort(Cand, all, {}, lanes.gpu_rule.better);
    return gpa.dupe(Cand, all[0..n]);
}

/// The normalizer the kernel sums, in the host's lane order and with the host's exp.
fn hostNorm(row: []const f32, m: f32, inv_t: f32) f64 {
    var lane: [1024]f32 = @splat(0.0);
    for (row, 0..) |l, i| {
        const v = l * inv_t;
        if (v >= -std.math.inf(f32)) lane[i % 1024] += @exp(v - m);
    }
    return lanes.gpu_rule.lanesSum(&lane);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var rows: usize = 4;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--rows")) {
            i += 1;
            rows = try std.fmt.parseInt(usize, argv[i], 10);
        }
    }
    var drv = xpu.Driver.open() catch {
        std.debug.print("skipped: no Level Zero loader\n", .{});
        return;
    };
    var ctx = try xpu.Context.init(&drv, null);
    const stream = try xpu.Stream.init(&ctx);
    var r = try xpu.rt.Runtime.init(&ctx, stream);
    var sampler = try ink.sampler.Sampler.init(gpa, &r, &ctx, 16);
    defer sampler.deinit();

    const host = try gpa.alloc(f32, 16 * vocab);
    defer gpa.free(host);
    var device = try r.alloc(16 * vocab * 4);
    defer device.free();
    var prng = std.Random.DefaultPrng.init(20261123);
    var failures: usize = 0;
    var checked: usize = 0;
    for (0..4) |kind| {
        for (0..rows) |j| fill(&prng, host[j * vocab ..][0..vocab], kind);
        try r.upload(device, std.mem.sliceAsBytes(host[0 .. rows * vocab]));
        for ([_]f32{ 1.0, 1.25, 0.5, 4.0 }) |inv_t| {
            for ([_]u32{ 1, 2, 20, 40, 100, 511, 512, 513, 1000, 1023, 1024 }) |n| {
                var picks: [16]ink.sampler.Row = undefined;
                for (picks[0..rows]) |*p| p.* = .{ .inv_t = inv_t, .n = n };
                const set = try sampler.select(device.ptr, vocab, picks[0..rows]);
                var buf: [1024]Cand = undefined;
                for (0..rows) |j| {
                    const got = set.list(j, n, &buf);
                    const want = try hostTop(gpa, host[j * vocab ..][0..vocab], inv_t, n);
                    defer gpa.free(want);
                    checked += 1;
                    for (got, want, 0..) |g, w, at| {
                        if (@as(u32, @bitCast(g.v)) != @as(u32, @bitCast(w.v)) or g.i != w.i) {
                            std.debug.print("MISMATCH kind {d} row {d} inv_t {d} n {d} at {d}: device ({d}, {d}) host ({d}, {d})\n", .{ kind, j, inv_t, n, at, g.v, g.i, w.v, w.i });
                            failures += 1;
                            break;
                        }
                    }
                }
            }
        }
        std.debug.print("candidates, row kind {d}: checked so far {d}, failures {d}\n", .{ kind, checked, failures });
    }

    // The draws: the device's candidates against the sort of the whole row.
    const rules = [_]lanes.Sampling{
        .{ .seed = 0, .temperature = 0.8, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 0, .temperature = 0.8, .top_k = 40, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 1024, .top_p = 0.9 },
        .{ .seed = 0, .temperature = 0.8, .top_k = 0, .top_p = 0.95 },
        .{ .seed = 0, .temperature = 0.6, .top_k = 0, .top_p = 1.0, .min_p = 0.05 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 0.7, .top_k = 0, .top_p = 0.2, .min_p = 0.02 },
        .{ .seed = 0, .temperature = 0.3, .top_k = 5, .top_p = 0.5, .min_p = 0.1 },
    };
    var draws: usize = 0;
    var differ: usize = 0;
    var normed: usize = 0;
    var needed_row: usize = 0;
    var worst_z: f64 = 0;
    for (0..4) |kind| {
        for (0..rows) |j| fill(&prng, host[j * vocab ..][0..vocab], kind);
        try r.upload(device, std.mem.sliceAsBytes(host[0 .. rows * vocab]));
        for (rules) |base| {
            var picks: [16]ink.sampler.Row = undefined;
            for (picks[0..rows]) |*p| p.* = ink.sampler.Row.of(base);
            const set = try sampler.select(device.ptr, vocab, picks[0..rows]);
            var buf: [1024]Cand = undefined;
            for (0..rows) |j| {
                const row = host[j * vocab ..][0..vocab];
                const top = set.list(j, picks[j].n, &buf);
                if (picks[j].norm) {
                    const want_z = hostNorm(row, top[0].v, picks[j].inv_t);
                    const err = @abs(@as(f64, set.z[j]) - want_z) / want_z;
                    worst_z = @max(worst_z, err);
                }
                for (0..24) |s| {
                    var rule = base;
                    rule.seed = 1000 + s * 7919 + j;
                    const position: u32 = @intCast(17 + s * 3);
                    const want = try lanes.gpu_full.draw(gpa, row, rule, position, null);
                    const got = lanes.gpu_full.drawTop(if (lanes.gpu_full.needsRow(rule)) row else null, top, rule, position);
                    draws += 1;
                    if (rule.top_k == 0) {
                        // the device's normalizer: the draw that does not read the row must equal the draw from the row
                        if (lanes.gpu_full.drawTopNorm(top, set.z[j], rule, position)) |quick| {
                            normed += 1;
                            if (quick != want) {
                                differ += 1;
                                std.debug.print("NORM DRAW DIFFERS kind {d} row {d} top_p {d} min_p {d} seed {d}: device z {d} gives {d}, host {d}\n", .{ kind, j, rule.top_p, rule.min_p, rule.seed, set.z[j], quick, want });
                            }
                        } else needed_row += 1;
                    }
                    if (got != want) {
                        differ += 1;
                        std.debug.print("DRAW DIFFERS kind {d} row {d} top_k {d} top_p {d} min_p {d} seed {d}: device {d} host {d}\n", .{ kind, j, rule.top_k, rule.top_p, rule.min_p, rule.seed, got, want });
                    }
                }
            }
        }
        std.debug.print("draws, row kind {d}: checked so far {d}, differing {d}\n", .{ kind, draws, differ });
    }
    std.debug.print("normalizer: {d} draws answered from the device's sum without the row, {d} needed the row; worst relative error of z against the host's sum {e}\n", .{ normed, needed_row, worst_z });
    std.debug.print("RESULT {s}: {d} candidate lists, {d} draws, {d} candidate mismatches, {d} draw differences\n", .{ if (failures == 0 and differ == 0 and worst_z < 1e-5) "PASS" else "FAIL", checked, draws, failures, differ });
    if (failures != 0 or differ != 0 or worst_z >= 1e-5) std.process.exit(1);
}

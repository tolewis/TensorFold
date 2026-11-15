//! The served target sampler of a decode or verify row (tensorfold.cuda.sampling.sample_rows with
//! exact_sampling.choose_rows): temperature 0 takes the row's argmax (the first of equal values). Otherwise, with top_k
//! > 0, the row's top_k + MARGIN largest logits (torch.topk's set; of equal values the lower ids) ordered by value, then
//! id (np.lexsort), the first top_k of them; each candidate's score is its logit over the temperature plus a Gumbel draw
//! keyed by (seed, position, token id) through a splitmix64 chain, so a verify row draws as the serial run at its
//! position. top_p sets the scores past the smallest prefix whose probability mass reaches top_p to -inf (the scaled
//! logits' exp over their sum: numpy's pairwise sum, then cumulative sums in order); min_p those under ln(min_p) of the
//! top. The token is the first best score. float64 as numpy computes it on the served image (numpy 2.1: its exp and log
//! are libm's, a row's sum of 20 its 8-accumulator pairwise sum; zrec np_libm.py).
const std = @import("std");
const builtin = @import("builtin");

/// exact_sampling.Sampling (the served lane's defaults: temperature 1.0, top_k 20, top_p 0.95, min_p 0).
pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,
};

/// exact_sampling.MARGIN: candidates beyond top_k read from the GPU.
pub const margin = 8;
/// The candidates a row takes at most here (top_k + MARGIN).
pub const max_candidates = 64;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// exact_sampling.uniform: a (0, 1) double from a splitmix64 hash of (seed, position, token id).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's pairwise sum of a contiguous float64 row (umath pairwise_sum, under its 128-element block).
pub fn pairwiseSum(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0.0;
        for (a) |v| res += v;
        return res;
    }
    std.debug.assert(n <= 128);
    var r: [8]f64 = a[0..8].*;
    var i: usize = 8;
    while (i < n - n % 8) : (i += 8) {
        for (0..8) |j| r[j] += a[i + j];
    }
    var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while (i < n) : (i += 1) res += a[i];
    return res;
}

/// a before b in np.lexsort((ids, -values))'s order: the larger value, then the lower id.
fn before(va: f32, ia: i64, vb: f32, ib: i64) bool {
    return va > vb or (va == vb and ia < ib);
}

/// Reference top-k (pre-optimization) for equivalence tests.
fn topCandidatesRef(row: []const f32, count: usize, values: []f32, ids: []i64) usize {
    const k = @min(count, row.len);
    var n: usize = 0;
    for (row, 0..) |v, i| {
        const id: i64 = @intCast(i);
        if (n == k and !before(v, id, values[k - 1], ids[k - 1])) continue;
        var j: usize = if (n < k) n else k - 1;
        while (j > 0 and before(v, id, values[j - 1], ids[j - 1])) : (j -= 1) {
            values[j] = values[j - 1];
            ids[j] = ids[j - 1];
        }
        values[j] = v;
        ids[j] = id;
        if (n < k) n += 1;
    }
    return n;
}

fn worstAmong(values: []const f32, ids: []const i64, n: usize) usize {
    var w: usize = 0;
    for (1..n) |i| {
        if (before(values[w], ids[w], values[i], ids[i])) {
            w = i;
        } else if (!before(values[i], ids[i], values[w], ids[w]) and ids[i] > ids[w]) {
            w = i;
        }
    }
    return w;
}

fn sortCandidatesLex(values: []f32, ids: []i64, n: usize) void {
    var i: usize = 1;
    while (i < n) : (i += 1) {
        const v = values[i];
        const id = ids[i];
        var j = i;
        while (j > 0 and before(v, id, values[j - 1], ids[j - 1])) : (j -= 1) {
            values[j] = values[j - 1];
            ids[j] = ids[j - 1];
        }
        values[j] = v;
        ids[j] = id;
    }
}

fn topCandidatesPass(row: []const f32, count: usize, id_bias: i64, values: []f32, ids: []i64) usize {
    const k = @min(count, row.len);
    var n: usize = 0;
    var worst: usize = 0;
    for (row, 0..) |v, i| {
        const id = id_bias + @as(i64, @intCast(i));
        if (n == k) {
            const wv = values[worst];
            const wi = ids[worst];
            if (v < wv) continue;
            if (v == wv and id >= wi) continue;
            if (!before(v, id, wv, wi)) continue;
            values[worst] = v;
            ids[worst] = id;
            worst = worstAmong(values[0..k], ids[0..k], k);
        } else {
            values[n] = v;
            ids[n] = id;
            n += 1;
            if (n == k) worst = worstAmong(values[0..k], ids[0..k], k);
        }
    }
    sortCandidatesLex(values[0..n], ids[0..n], n);
    return n;
}

/// The row's first `count` (value, id) in that order: torch.topk's set (of equal values the lower ids) as np.lexsort
/// orders it.
pub fn topCandidates(row: []const f32, count: usize, values: []f32, ids: []i64) usize {
    return topCandidatesPass(row, count, 0, values, ids);
}

/// The first index of the largest value (np.argmax / torch.argmax on a row without NaN).
pub fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

/// sample_rows for one row of fp32 logits at absolute position `position`: the token.
pub fn sampleRow(row: []const f32, position: u64, s: Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(argmax(row));
    if (s.top_k == 0) return error.NotPortedYet; // the nucleus path (top_k off: a model setting only top_p)
    const count = @min(row.len, s.top_k + margin);
    if (count > max_candidates) return error.TooManyCandidates;
    var values: [max_candidates]f32 = undefined;
    var ids: [max_candidates]i64 = undefined;
    const width = topCandidates(row, count, &values, &ids);
    // choose_rows on the candidates (already in lexsort's order)
    const k = @max(1, @min(s.top_k, width));
    const t = @max(s.temperature, 1e-6);
    var scaled: [max_candidates]f64 = undefined;
    var score: [max_candidates]f64 = undefined;
    for (0..k) |j| {
        scaled[j] = @as(f64, values[j]) / t;
        const u = uniform(s.seed, position, @bitCast(ids[j]));
        score[j] = scaled[j] - log(-log(u));
    }
    if (0.0 < s.top_p and s.top_p < 1.0) {
        var mx = scaled[0];
        for (scaled[1..k]) |v| mx = @max(mx, v);
        var probs: [max_candidates]f64 = undefined;
        for (0..k) |j| probs[j] = exp(scaled[j] - mx);
        const sum = pairwiseSum(probs[0..k]);
        for (0..k) |j| probs[j] /= sum;
        var acc: f64 = 0.0;
        var below: usize = 0;
        for (0..k) |j| {
            acc += probs[j];
            if (acc < s.top_p) below += 1;
        }
        const keep = below + 1;
        for (keep..k) |j| score[j] = -std.math.inf(f64);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + log(s.min_p);
        for (0..k) |j| {
            if (scaled[j] < floor) score[j] = -std.math.inf(f64);
        }
    }
    var best: usize = 0;
    for (0..k) |j| {
        if (score[j] > score[best]) best = j;
    }
    return ids[best];
}

test "splitmix64 draws as exact_sampling.uniform makes them" {
    // (seed, position, id) -> the bits of exact_sampling.uniform's double
    try std.testing.expectEqual(@as(u64, 0x3fe8cd8867d6a69c), @as(u64, @bitCast(uniform(101, 2229, 2581))));
    try std.testing.expectEqual(@as(u64, 0x3fd08f785f86da1b), @as(u64, @bitCast(uniform(404, 59, 129279))));
    try std.testing.expectEqual(@as(u64, 0x3fc9ff45ea7ce2be), @as(u64, @bitCast(uniform(0, 0, 0))));
}

test "numpy's pairwise sum" {
    const a = [_]f64{ 1e16, 1.0, -1e16, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 0.5, 0.25 };
    // r = a[0..8] (no whole second block of 8); ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the rest in order
    var want = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    for (a[8..]) |v| want += v;
    try std.testing.expectEqual(want, pairwiseSum(&a));
}

test "top candidates by value, then id" {
    const row = [_]f32{ 1, 5, 3, 5, 2, 5, 0 };
    var v: [4]f32 = undefined;
    var ids: [4]i64 = undefined;
    try std.testing.expectEqual(@as(usize, 4), topCandidates(&row, 4, &v, &ids));
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 5, 2 }, &ids);
    try std.testing.expectEqualSlices(f32, &.{ 5, 5, 5, 3 }, &v);
}

const test_vocab: usize = 129_280;

fn sampleRowRef(row: []const f32, position: u64, s: Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(argmax(row));
    if (s.top_k == 0) return error.NotPortedYet;
    const count = @min(row.len, s.top_k + margin);
    if (count > max_candidates) return error.TooManyCandidates;
    var values: [max_candidates]f32 = undefined;
    var ids: [max_candidates]i64 = undefined;
    const width = topCandidatesRef(row, count, &values, &ids);
    const k = @max(1, @min(s.top_k, width));
    const t = @max(s.temperature, 1e-6);
    var scaled: [max_candidates]f64 = undefined;
    var score: [max_candidates]f64 = undefined;
    for (0..k) |j| {
        scaled[j] = @as(f64, values[j]) / t;
        const u = uniform(s.seed, position, @bitCast(ids[j]));
        score[j] = scaled[j] - log(-log(u));
    }
    if (0.0 < s.top_p and s.top_p < 1.0) {
        var mx = scaled[0];
        for (scaled[1..k]) |v| mx = @max(mx, v);
        var probs: [max_candidates]f64 = undefined;
        for (0..k) |j| probs[j] = exp(scaled[j] - mx);
        const sum = pairwiseSum(probs[0..k]);
        for (0..k) |j| probs[j] /= sum;
        var acc: f64 = 0.0;
        var below: usize = 0;
        for (0..k) |j| {
            acc += probs[j];
            if (acc < s.top_p) below += 1;
        }
        const keep = below + 1;
        for (keep..k) |j| score[j] = -std.math.inf(f64);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + log(s.min_p);
        for (0..k) |j| {
            if (scaled[j] < floor) score[j] = -std.math.inf(f64);
        }
    }
    var best: usize = 0;
    for (0..k) |j| {
        if (score[j] > score[best]) best = j;
    }
    return ids[best];
}

fn fillRandomNormal(row: []f32, rand: std.Random) void {
    for (row) |*v| v.* = rand.float(f32) * 4.0 - 2.0;
}

fn fillPeaked(row: []f32, rand: std.Random) void {
    @memset(row, 0.0);
    row[rand.intRangeLessThan(usize, 0, row.len)] = 10.0;
}

fn fillFlat(row: []f32) void {
    @memset(row, 1.0);
}

fn monotonicNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn benchSampleRowUs(
    row: []f32,
    rounds: usize,
    position: u64,
    s: Sampling,
    comptime label: []const u8,
    comptime sample: *const fn ([]const f32, u64, Sampling) anyerror!i64,
) void {
    const t0 = monotonicNs();
    var i: usize = 0;
    while (i < rounds) : (i += 1) {
        _ = sample(row, position +% @as(u64, @intCast(i)), s) catch unreachable;
    }
    const ns = monotonicNs() - t0;
    const us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(rounds)) / 1000.0;
    std.debug.print("sampleRow {s}: {d:.1} us/row\n", .{ label, us });
}

test "topCandidates matches reference on vocab rows" {
    var prng = std.Random.DefaultPrng.init(0xA11CE);
    const rand = prng.random();
    const row = try std.testing.allocator.alloc(f32, test_vocab);
    defer std.testing.allocator.free(row);
    var v0: [max_candidates]f32 = undefined;
    var id0: [max_candidates]i64 = undefined;
    var v1: [max_candidates]f32 = undefined;
    var id1: [max_candidates]i64 = undefined;
    for (0..200) |_| {
        fillRandomNormal(row, rand);
        const n0 = topCandidatesRef(row, 28, &v0, &id0);
        const n1 = topCandidates(row, 28, &v1, &id1);
        try std.testing.expectEqual(n0, n1);
        try std.testing.expectEqualSlices(f32, v0[0..n0], v1[0..n1]);
        try std.testing.expectEqualSlices(i64, id0[0..n0], id1[0..n1]);
        fillFlat(row);
        const nf0 = topCandidatesRef(row, 28, &v0, &id0);
        const nf1 = topCandidates(row, 28, &v1, &id1);
        try std.testing.expectEqual(nf0, nf1);
        try std.testing.expectEqualSlices(f32, v0[0..nf0], v1[0..nf1]);
        try std.testing.expectEqualSlices(i64, id0[0..nf0], id1[0..nf1]);
    }
}

test "sampleRow matches reference" {
    var prng = std.Random.DefaultPrng.init(0x51A1E);
    const rand = prng.random();
    const row = try std.testing.allocator.alloc(f32, test_vocab);
    defer std.testing.allocator.free(row);
    const configs = [_]Sampling{
        .{ .seed = 1, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 },
        .{ .seed = 2, .temperature = 1.0, .top_k = 20, .top_p = 1.0, .min_p = 0.0 },
        .{ .seed = 3, .temperature = 1.0, .top_k = 1, .top_p = 0.95, .min_p = 0.0 },
        .{ .seed = 4, .temperature = 1.0, .top_k = 64, .top_p = 0.95, .min_p = 0.0 },
        .{ .seed = 5, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.05 },
    };
    var trial: usize = 0;
    while (trial < 20_000) : (trial += 1) {
        if (trial % 3 == 0) fillFlat(row) else fillRandomNormal(row, rand);
        const cfg = configs[trial % configs.len];
        const pos: u64 = @intCast(trial * 17 + 3);
        if (cfg.top_k + margin > max_candidates) {
            try std.testing.expectError(error.TooManyCandidates, sampleRow(row, pos, cfg));
            try std.testing.expectError(error.TooManyCandidates, sampleRowRef(row, pos, cfg));
            continue;
        }
        const got = try sampleRow(row, pos, cfg);
        const want = try sampleRowRef(row, pos, cfg);
        try std.testing.expectEqual(want, got);
    }
    // many ties: partial flat
    for (0..500) |_| {
        fillRandomNormal(row, rand);
        for (row) |*v| {
            if (rand.boolean()) v.* = 0.5;
        }
        const cfg = configs[0];
        const pos: u64 = rand.int(u64);
        try std.testing.expectEqual(try sampleRowRef(row, pos, cfg), try sampleRow(row, pos, cfg));
    }
}

test "sampleRow draw timing on server vocab" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;
    const s = Sampling{ .seed = 9001, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const row = try std.testing.allocator.alloc(f32, test_vocab);
    defer std.testing.allocator.free(row);
    fillRandomNormal(row, prng.random());
    const rounds: usize = 400;
    benchSampleRowUs(row, rounds, 100, s, "ref-random", sampleRowRef);
    benchSampleRowUs(row, rounds, 100, s, "random", sampleRow);
    fillPeaked(row, prng.random());
    benchSampleRowUs(row, rounds, 100, s, "ref-peaked", sampleRowRef);
    benchSampleRowUs(row, rounds, 100, s, "peaked", sampleRow);
    fillFlat(row);
    benchSampleRowUs(row, rounds, 100, s, "ref-flat", sampleRowRef);
    benchSampleRowUs(row, rounds, 100, s, "flat", sampleRow);
}

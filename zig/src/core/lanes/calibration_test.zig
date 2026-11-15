//! The draft-survival calibration: a stream's realised over predicted drafts scales the chances its window is allocated by.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const be = @import("backend.zig");
const win = @import("windows.zig");

const gpa = std.testing.allocator;

const chain = [_]f64{ 0.9, 0.81, 0.729, 0.6561 };

fn chances(_: *anyopaque, _: *Stream, out: []f64) anyerror!bool {
    @memcpy(out, chain[0..out.len]);
    return true;
}

fn startNow(_: *anyopaque) void {}

fn elapsed(_: *anyopaque, _: be.Mark) f64 {
    return 0.0;
}

/// Each stream's granted drafts and predicted landings after one allocation of 4 held drafts apiece.
fn grants(weight: f64, cals: []const f64, granted: []u32, predicted: []f64) !void {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 2.0 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = true, .draft_cal = weight }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var vtable = target.backend().vtable.*;
    vtable.probabilities = chances;
    var token: u8 = 0;
    var engine = Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vtable }, .{ .ptr = &token, .vtable = &.{ .start = startNow, .elapsed_ms = elapsed } });
    defer engine.deinit();
    var streams: [2]Stream = undefined;
    var plans: [2]win.Plan = undefined;
    for (cals, 0..) |cal, i| {
        streams[i] = try Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 8 });
        streams[i].cal = cal;
        plans[i] = .{ .stream = &streams[i], .position = 0, .kind = .head, .held = 4 };
    }
    defer for (streams[0..cals.len]) |*s| s.deinit(gpa);
    try win.allocate(&engine, plans[0..cals.len]);
    for (streams[0..cals.len], granted, predicted) |s, *g, *p| {
        g.* = s.granted.?;
        p.* = s.predicted();
    }
}

test "off, the chances and the grant are the uncalibrated ones whatever the stream's cal" {
    var g: [1]u32 = undefined;
    var p: [1]f64 = undefined;
    try grants(0.0, &.{0.3}, &g, &p);
    try std.testing.expectEqual(@as(u32, 4), g[0]);
    try std.testing.expectEqual(@as(f64, 0.0), p[0]);
}

test "a cold stream is uncalibrated and predicts the drafts it was granted" {
    var g: [1]u32 = undefined;
    var p: [1]f64 = undefined;
    try grants(0.5, &.{1.0}, &g, &p);
    try std.testing.expectEqual(@as(u32, 4), g[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9 + 0.81 + 0.729 + 0.6561), p[0], 1e-12);
}

test "a stream that realises half its drafts is granted fewer, and predicts from the uncalibrated chances" {
    var g: [1]u32 = undefined;
    var p: [1]f64 = undefined;
    try grants(0.5, &.{0.5}, &g, &p);
    try std.testing.expectEqual(@as(u32, 2), g[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9 + 0.81), p[0], 1e-12);
}

test "the floor keeps a collapsed stream drafting" {
    var g: [1]u32 = undefined;
    var p: [1]f64 = undefined;
    try grants(0.5, &.{0.05}, &g, &p);
    try std.testing.expectEqual(@as(u32, 1), g[0]);
    var at_floor: [1]u32 = undefined;
    try grants(0.5, &.{0.2}, &at_floor, &p);
    try std.testing.expectEqual(g[0], at_floor[0]);
}

test "a shared round gives each stream its own calibration" {
    var g: [2]u32 = undefined;
    var p: [2]f64 = undefined;
    try grants(0.5, &.{ 1.0, 0.2 }, &g, &p);
    try std.testing.expectEqual(@as(u32, 4), g[0]);
    try std.testing.expectEqual(@as(u32, 0), g[1]);
}

test "the survival solves the chances landed for the kept drafts" {
    const p = [_]f64{ 0.97, 0.94, 0.91, 0.88 };
    const c = Stream.survival(&p, 2.55);
    try std.testing.expect(c > 0.84 and c < 0.87);
    try std.testing.expectEqual(@as(f64, 1.0), Stream.survival(&p, 4.0));
    try std.testing.expectEqual(@as(f64, 1.0), Stream.survival(&p, 3.7));
    try std.testing.expectEqual(@as(f64, 0.2), Stream.survival(&p, 0.0));
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), Stream.survival(&.{0.9}, 0.45), 1e-6);
    try std.testing.expectEqual(@as(f64, 1.0), Stream.survival(&p, 40.0));
}

test "the average moves by the weight toward the round's kept and expected drafts" {
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    s.observeCal(0.5, 3.0);
    try std.testing.expectEqual(@as(f64, 1.0), s.cal);
    s.predict(&.{0.9});
    s.observeCal(0.5, 0.0);
    try std.testing.expectEqual(@as(f64, 0.2), s.cal);
    s.predict(&.{0.9});
    s.observeCal(0.5, 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5 / 0.9), s.cal, 1e-6);
    s.cal = 0.0;
    try std.testing.expectEqual(@as(f64, 0.2), s.calScale(1.0));
}

/// The cal after 20 rounds of 4 drafts whose true per-draft survival is `factor` (its expected kept drafts).
fn converged(factor: f64) f64 {
    var s = Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 }) catch unreachable;
    defer s.deinit(gpa);
    const raw = [_]f64{ 0.97, 0.94, 0.91, 0.88 };
    var kept: f64 = 0.0;
    var f: f64 = 1.0;
    for (raw) |p| {
        f *= factor;
        kept += f * p;
    }
    for (0..20) |_| {
        s.predict(&raw);
        s.observeCal(0.3, kept);
    }
    return s.cal;
}

test "prose-like and code-like streams settle near their per-draft survival" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.85), converged(0.85), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 0.99), converged(0.99), 0.01);
}

/// The mean scale over many rounds of a chain whose drafts survive at `factor` times the predicted rate, drawn at random.
fn meanScale(factor: f64) f64 {
    var s = Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 }) catch unreachable;
    defer s.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const q = 0.85;
    var raw: [4]f64 = undefined;
    var reach: f64 = 1.0;
    for (&raw) |*p| {
        reach *= q;
        p.* = reach;
    }
    var total: f64 = 0.0;
    const rounds = 4000;
    for (0..rounds) |_| {
        var kept: f64 = 0.0;
        for (0..raw.len) |_| {
            if (rng.float(f64) >= factor * q) break;
            kept += 1.0;
        }
        s.predict(&raw);
        s.observeCal(0.2, kept);
        total += s.calScale(1.0);
    }
    return total / rounds;
}

test "noise does not drag a well calibrated stream's scale down, and a short one still shows" {
    try std.testing.expect(meanScale(1.0) > 0.95);
    const low = meanScale(0.8);
    try std.testing.expect(low > 0.72 and low < 0.88);
}

test "the probe logs one line on its period and never on an empty round" {
    var s = try Stream.init(gpa, .{ .id = "probe", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    s.predict(&.{ 0.9, 0.81 });
    s.observeCal(0.2, 1.0);
    s.logCal(1, 1.0, null, 3, 1, 2);
    s.logCal(0, 1.0, null, 3, 1, 2);
    s.cal_n = 0;
    s.logCal(1, 1.0, null, 1, 0, 1);
    try std.testing.expectEqual(@as(u64, 3), s.cal_rounds);
}

test "fix_k raw held chances are not overwritten by forced allocation chances" {
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    const raw = [_]f64{ 0.9, 0.81, 0.729 };
    s.rememberHeld(&raw);
    const forced = [_]f64{ 1.0, 1.0, 0.0 };
    if (s.cal_held == 0) s.rememberHeld(&forced);
    try std.testing.expectEqual(@as(u32, 3), s.cal_held);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9), s.cal_held_raw[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.81), s.cal_held_raw[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.729), s.cal_held_raw[2], 1e-12);
}

test "cal log runs with draft calibration off when cal_log is set" {
    var g: [1]u32 = undefined;
    var p: [1]f64 = undefined;
    try grants(0.0, &.{1.0}, &g, &p);
    try std.testing.expectEqual(@as(u32, 4), g[0]);
}

test "cal_ceil above 1 lets scale rise and clamps calibrated chances" {
    const raw = [_]f64{ 0.9, 0.81, 0.729 };
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    s.cal = 1.35;
    try std.testing.expectEqual(@as(f64, 1.0), s.calScale(1.0));
    try std.testing.expectApproxEqAbs(@as(f64, 1.35), s.calScale(1.5), 1e-12);
    const scaled = try win.calibrated(gpa, &raw, s.calScale(1.5));
    defer gpa.free(scaled);
    for (scaled) |v| try std.testing.expect(v <= 1.0);
    try std.testing.expect(scaled[0] > 0.9);
    const rising = [_]f64{ 0.50, 0.48 };
    const capped = try win.calibrated(gpa, &rising, 1.5);
    defer gpa.free(capped);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), capped[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), capped[1], 1e-12);
    try std.testing.expect(capped[1] < 1.0);
}

test "cal log fields follow each round with draft cal off" {
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    const r1_held = [_]f64{ 0.9, 0.81, 0.729 };
    s.rememberHeld(&r1_held);
    try std.testing.expectEqual(@as(u32, 3), s.cal_held);
    var r1_grant = [_]f64{ r1_held[0], r1_held[1] };
    s.predict(&r1_grant);
    s.granted = 2;
    try std.testing.expectEqual(@as(usize, 2), s.cal_n);
    try std.testing.expectApproxEqAbs(@as(f64, 1.71), s.predicted(), 1e-12);
    s.logCal(1, 1.0, null, 4, 2, 3);
    s.cal_held = 0;
    s.cal_n = 0;
    const r2_held = [_]f64{ 0.5, 0.4 };
    s.rememberHeld(&r2_held);
    try std.testing.expectEqual(@as(u32, 2), s.cal_held);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), s.cal_held_raw[0], 1e-12);
    s.predict(&.{r2_held[0]});
    s.granted = 1;
    try std.testing.expectEqual(@as(usize, 1), s.cal_n);
    s.logCal(1, 1.0, null, 2, 1, 2);
    s.cal_held = 0;
    s.cal_n = 0;
    s.predict(&r1_grant);
    s.cal = 0.5;
    try std.testing.expectEqual(@as(usize, 2), s.cal_n);
    s.observeCal(0.5, 1.0);
    try std.testing.expectEqual(@as(usize, 2), s.cal_n);
    s.logCal(1, 1.0, null, 4, 2, 3);
    try std.testing.expectEqual(@as(usize, 2), s.cal_n);
    s.cal_n = 0;
}

test "the ceiling must lie in 1..2" {
    try std.testing.expectError(error.BadCalCeil, Config.init(gpa, .{ .cal_ceil = 0.5 }, 16, 15));
    var ok = try Config.init(gpa, .{ .cal_ceil = 1.5 }, 16, 15);
    ok.deinit(gpa);
}

test "the weight must lie in 0..1" {
    for ([_]f64{ -0.1, 1.5, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        try std.testing.expectError(error.BadDraftCal, Config.init(gpa, .{ .draft_cal = bad }, 16, 15));
    }
    var ok = try Config.init(gpa, .{ .draft_cal = 1.0 }, 16, 15);
    ok.deinit(gpa);
}

fn warmShare(s: *Stream, granted: u32, kept: usize, rounds: usize) void {
    for (0..rounds) |_| s.observeShare(granted, kept);
}

test "without full_at the present calibration grants are unchanged" {
    var g: [1]u32 = undefined;
    var p: [1]f64 = undefined;
    try grants(0.5, &.{0.5}, &g, &p);
    try std.testing.expectEqual(@as(u32, 2), g[0]);
}

test "a stream that keeps 9 of 10 drafts gets full grants at x = 0.8 after 8 verifies" {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 2.0 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = true, .draft_cal = 0.5, .full_at = 0.8 }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var vtable = target.backend().vtable.*;
    vtable.probabilities = chances;
    var token: u8 = 0;
    var engine = Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vtable }, .{ .ptr = &token, .vtable = &.{ .start = startNow, .elapsed_ms = elapsed } });
    defer engine.deinit();
    var s = try Stream.init(gpa, .{ .id = "hot", .prompt = &.{ 1, 2, 3 }, .max_new = 8 });
    defer s.deinit(gpa);
    warmShare(&s, 10, 9, 8);
    try std.testing.expect(s.fullGrant(0.8));
    const plan: win.Plan = .{ .stream = &s, .position = 0, .kind = .head, .held = 4 };
    try win.allocate(&engine, &.{plan});
    try std.testing.expectEqual(@as(u32, 4), s.granted.?);
}

test "a stream that keeps 6 of 10 drafts does not get full grants at x = 0.8" {
    var s = try Stream.init(gpa, .{ .id = "cold", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    warmShare(&s, 10, 6, 8);
    try std.testing.expect(!s.fullGrant(0.8));
}

test "a full stream whose kept share falls under x reverts to the rule" {
    var s = try Stream.init(gpa, .{ .id = "drop", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    warmShare(&s, 10, 9, 8);
    try std.testing.expect(s.fullGrant(0.8));
    warmShare(&s, 10, 5, 12);
    try std.testing.expect(!s.fullGrant(0.8));
}

test "TF_DS_T25_FULL_AT must lie in 0.5..1" {
    try std.testing.expectError(error.BadFullAt, Config.parseFullAt("0.4"));
    try std.testing.expectEqual(@as(f64, 0.8), try Config.parseFullAt("0.8"));
}

fn lowChances(_: *anyopaque, _: *Stream, out: []f64) anyerror!bool {
    for (out) |*o| o.* = 0.1;
    return true;
}

test "full_at applies only when one plan is in the round" {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 2.0 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = true, .full_at = 0.8 }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var vtable = target.backend().vtable.*;
    vtable.probabilities = lowChances;
    var token: u8 = 0;
    var engine = Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vtable }, .{ .ptr = &token, .vtable = &.{ .start = startNow, .elapsed_ms = elapsed } });
    defer engine.deinit();
    var s0 = try Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 8 });
    defer s0.deinit(gpa);
    var s1 = try Stream.init(gpa, .{ .id = "b", .prompt = &.{2}, .max_new = 8 });
    defer s1.deinit(gpa);
    warmShare(&s0, 10, 9, 8);
    warmShare(&s1, 10, 9, 8);
    const alone: win.Plan = .{ .stream = &s0, .position = 0, .kind = .head, .held = 4 };
    try win.allocate(&engine, &.{alone});
    try std.testing.expectEqual(@as(u32, 4), s0.granted.?);
    s0.granted = null;
    s1.granted = null;
    const pair = [_]win.Plan{
        .{ .stream = &s0, .position = 0, .kind = .head, .held = 4 },
        .{ .stream = &s1, .position = 0, .kind = .head, .held = 4 },
    };
    try win.allocate(&engine, &pair);
    try std.testing.expect((s0.granted orelse 0) < 4);
    try std.testing.expect((s1.granted orelse 0) < 4);
}

test "fix_k wins over full_at" {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 };
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var vtable = target.backend().vtable.*;
    vtable.probabilities = lowChances;
    var token: u8 = 0;
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 });
    defer s.deinit(gpa);
    warmShare(&s, 10, 9, 8);
    const plan: win.Plan = .{ .stream = &s, .position = 0, .kind = .head, .held = 4 };
    var cfg_full = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = true, .full_at = 0.8 }, 16, 15);
    defer cfg_full.deinit(gpa);
    var engine_full = Engine.init(gpa, &cfg_full, .{ .ptr = &target, .vtable = &vtable }, .{ .ptr = &token, .vtable = &.{ .start = startNow, .elapsed_ms = elapsed } });
    defer engine_full.deinit();
    try win.allocate(&engine_full, &.{plan});
    try std.testing.expectEqual(@as(u32, 4), s.granted.?);
    s.granted = null;
    var cfg_k = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = true, .full_at = 0.8, .fix_k = 1 }, 16, 15);
    defer cfg_k.deinit(gpa);
    var engine_k = Engine.init(gpa, &cfg_k, .{ .ptr = &target, .vtable = &vtable }, .{ .ptr = &token, .vtable = &.{ .start = startNow, .elapsed_ms = elapsed } });
    defer engine_k.deinit();
    try win.allocate(&engine_k, &.{plan});
    try std.testing.expect((s.granted orelse 0) < 4);
}

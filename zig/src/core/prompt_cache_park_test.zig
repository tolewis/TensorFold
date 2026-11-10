//! The host tier (XPU_PARK_GIB): a state that leaves for room is parked, and a parked state holds no pool page.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fake = @import("prompt_cache_fake.zig");
const Store = pc.Store;
const Shared = fake.Shared;

fn fill(buf: []u32, base: u32) void {
    for (buf, 0..) |*t, i| t.* = base +% @as(u32, @intCast(i));
}

fn open(f: *Shared, budget: u64, tier: u64) Store {
    var s = Store.init(f.gpa, f.parking(), .{}, budget);
    s.tier = tier;
    s.park_min = 64;
    return s;
}

test "park: the least recently used state is parked, and a later turn hits it" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 64 };
    defer f.deinit();
    var s = open(&f, 64, 512);
    defer s.deinit();
    var a: [65]u32 = undefined;
    var b: [65]u32 = undefined;
    fill(&a, 1);
    fill(&b, 1000);
    try f.grow(64);
    try std.testing.expect(s.keep(&a, 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&b, 64, null, &.{}, &.{}));
    const hit = s.find(&a, &.{}, &.{}).?;
    try std.testing.expect(hit.parked);
    try std.testing.expect(!s.find(&b, &.{}, &.{}).?.parked);
    try std.testing.expectEqual(@as(u64, 1), s.counts.parks);
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    const snaps = f.parking();
    try std.testing.expectEqual(@as(u64, 0), snaps.vtable.frees.?(snaps.ptr, hit.saved));
    try std.testing.expectEqual(@as(u64, 0), snaps.vtable.returns.?(snaps.ptr, hit.saved));
    s.resumed(hit, &a, true);
    try std.testing.expectEqual(@as(u64, 1), s.counts.park_resumes);
    try std.testing.expect(hit.parked);
}

test "park: a parked state is not a victim for room and does not count against most" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 64 };
    defer f.deinit();
    var s = open(&f, 64, 512);
    defer s.deinit();
    s.most = 1;
    var a: [65]u32 = undefined;
    var b: [65]u32 = undefined;
    var c: [65]u32 = undefined;
    fill(&a, 1);
    fill(&b, 1000);
    fill(&c, 2000);
    try f.grow(64);
    try std.testing.expect(s.keep(&a, 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&b, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}).?.parked);
    try std.testing.expect(!s.find(&b, &.{}, &.{}).?.parked);
    try std.testing.expectEqual(@as(usize, 1), s.occupancy());
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&c, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}).?.parked);
    try std.testing.expect(s.find(&c, &.{}, &.{}) != null);
    try std.testing.expect(s.evictOne());
    try std.testing.expect(s.find(&a, &.{}, &.{}).?.parked);
    try std.testing.expect(!s.evictOne());
}

test "park: a full tier drops its least recently used parked state" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 64 };
    defer f.deinit();
    var s = open(&f, 64, 64);
    defer s.deinit();
    var a: [65]u32 = undefined;
    var b: [65]u32 = undefined;
    var c: [65]u32 = undefined;
    fill(&a, 1);
    fill(&b, 1000);
    fill(&c, 2000);
    try f.grow(64);
    try std.testing.expect(s.keep(&a, 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&b, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}).?.parked);
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&c, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}) == null);
    try std.testing.expect(s.find(&b, &.{}, &.{}).?.parked);
    try std.testing.expect(!s.find(&c, &.{}, &.{}).?.parked);
    try std.testing.expect(s.counts.park_drops >= 1);
}

test "park: a borrowed parked state is not dropped to make tier room" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 64 };
    defer f.deinit();
    var s = open(&f, 64, 64);
    defer s.deinit();
    var a: [65]u32 = undefined;
    var b: [65]u32 = undefined;
    var c: [65]u32 = undefined;
    fill(&a, 1);
    fill(&b, 1000);
    fill(&c, 2000);
    try f.grow(64);
    try std.testing.expect(s.keep(&a, 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&b, 64, null, &.{}, &.{}));
    s.find(&a, &.{}, &.{}).?.borrowers = 1;
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&c, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}).?.parked);
    try std.testing.expectEqual(@as(u64, 0), s.counts.park_drops);
}

test "park: a state that a later state extends is dropped, not parked" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 128 };
    defer f.deinit();
    var s = open(&f, 128, 1024);
    defer s.deinit();
    var prompt: [129]u32 = undefined;
    fill(&prompt, 1);
    try f.grow(64);
    try std.testing.expect(s.keep(prompt[0..65], 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(128);
    try std.testing.expect(s.keep(&prompt, 128, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 0), s.counts.parks);
    try std.testing.expect(s.find(prompt[0..65], &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 128), s.find(&prompt, &.{}, &.{}).?.at);
    try std.testing.expect(s.counts.evict_superseded >= 1);
}

test "park: a new state that supersedes a parked one drops it" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 128 };
    defer f.deinit();
    var s = open(&f, 128, 1024);
    defer s.deinit();
    var prompt: [129]u32 = undefined;
    var other: [65]u32 = undefined;
    fill(&prompt, 1);
    fill(&other, 1000);
    try f.grow(64);
    try std.testing.expect(s.keep(prompt[0..65], 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&other, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(prompt[0..65], &.{}, &.{}).?.parked);
    f.resetLive();
    try f.grow(128);
    try std.testing.expect(s.keep(&prompt, 128, null, &.{}, &.{}));
    try std.testing.expect(s.find(prompt[0..65], &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 128), s.find(&prompt, &.{}, &.{}).?.at);
    try std.testing.expect(s.counts.park_drops >= 1);
}

test "park: with the tier off, a short budget still drops the least recently used state" {
    const gpa = std.testing.allocator;
    var f: Shared = .{ .gpa = gpa, .page = 64 };
    defer f.deinit();
    var s = Store.init(gpa, f.snapshots(), .{}, 64);
    defer s.deinit();
    try std.testing.expect(s.family.vtable.park == null);
    var a: [65]u32 = undefined;
    var b: [65]u32 = undefined;
    fill(&a, 1);
    fill(&b, 1000);
    try f.grow(64);
    try std.testing.expect(s.keep(&a, 64, null, &.{}, &.{}));
    f.resetLive();
    try f.grow(64);
    try std.testing.expect(s.keep(&b, 64, null, &.{}, &.{}));
    try std.testing.expect(s.find(&a, &.{}, &.{}) == null);
    try std.testing.expect(s.find(&b, &.{}, &.{}) != null);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 0), s.counts.parks);
}

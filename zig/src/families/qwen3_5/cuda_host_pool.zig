//! One pinned host arena a prompt cache's kept states come from: a first-fit list of free ranges, 2 MiB aligned,
//! merged with their neighbours when a kept state goes. Pinning a 1.4 GB copy on every save costs more than the
//! copy does, so the arena is pinned once and the copies only take and give ranges in it.
const std = @import("std");

/// A taken range starts 2 MiB aligned: the copies' rows land on page-sized runs.
pub const unit = 2 << 20;

/// Milliseconds on the monotonic clock (std.time carries constants only in this Zig).
pub fn monotonicMs() u64 {
    const sys = @import("sys");
    var ts: sys.c.timespec = undefined;
    if (sys.c.clock_gettime(sys.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @intCast(@as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / 1_000_000);
}

pub const Pool = struct {
    bytes: []align(16) u8, // the pinned arena
    host: ?@import("cuda").HostBuffer = null, // the allocation behind `bytes` (null: a test's heap bytes)
    gpa: std.mem.Allocator,
    ranges: std.ArrayListUnmanaged(Range) = .empty, // free ranges, sorted by offset, never touching or overlapping

    pub const Range = struct { off: usize, len: usize };

    /// A pool over bytes someone else owns (a test's).
    pub fn over(gpa: std.mem.Allocator, bytes: []align(16) u8) !*Pool {
        const p = try gpa.create(Pool);
        p.* = .{ .bytes = bytes, .gpa = gpa };
        try p.ranges.append(gpa, .{ .off = 0, .len = bytes.len });
        return p;
    }

    /// The engine's arena: one cuMemHostAlloc, pinned now, its first byte `unit` aligned (the allocation carries a
    /// spare unit for that, so a full-size range still fits).
    pub fn pinned(gpa: std.mem.Allocator, d: *const @import("cuda").Driver, len: usize) !*Pool {
        var host = try @import("cuda").HostBuffer.alloc(d, len + unit);
        errdefer host.free();
        const off = (unit - @intFromPtr(host.bytes.ptr) % unit) % unit;
        const p = try over(gpa, @alignCast(host.bytes[off..][0..len]));
        p.host = host;
        return p;
    }

    pub fn deinit(p: *Pool) void {
        p.ranges.deinit(p.gpa);
        if (p.host) |*h| h.free();
        p.gpa.destroy(p);
    }

    pub fn size(p: *const Pool) usize {
        return p.bytes.len;
    }

    /// The offset a range must start on for the bytes to land `unit` aligned in memory.
    fn phase(p: *const Pool) usize {
        const mis = @intFromPtr(p.bytes.ptr) % unit;
        return if (mis == 0) 0 else unit - mis;
    }

    /// `len` bytes from the first range that holds them, `unit` aligned; null when none does.
    pub fn take(p: *Pool, len: usize) ?[]align(16) u8 {
        if (len == 0 or len > p.bytes.len) return null;
        const want = std.mem.alignForward(usize, len, unit); // whole units: a given range merges back exactly
        const ph = p.phase();
        for (p.ranges.items, 0..) |r, i| {
            const begin = r.off + (ph + unit - r.off % unit) % unit;
            if (begin + want > r.off + r.len) continue;
            const tail_off = begin + want;
            const tail_len = r.off + r.len - tail_off;
            if (tail_len == 0) {
                _ = p.ranges.orderedRemove(i);
            } else {
                p.ranges.items[i] = .{ .off = tail_off, .len = tail_len };
            }
            return @alignCast(p.bytes[begin..][0..len]);
        }
        return null;
    }

    /// A taken range back: it merges with a free range it touches, before or after.
    pub fn give(p: *Pool, taken: []align(16) u8) void {
        const base = @intFromPtr(p.bytes.ptr);
        const at = @intFromPtr(taken.ptr) - base;
        const len = std.mem.alignForward(usize, taken.len, unit);
        var i: usize = 0;
        while (i < p.ranges.items.len and p.ranges.items[i].off < at) i += 1;
        var merged: Range = .{ .off = at, .len = len };
        // the range before, when it ends where this one starts
        if (i > 0 and p.ranges.items[i - 1].off + p.ranges.items[i - 1].len == at) {
            merged = .{ .off = p.ranges.items[i - 1].off, .len = p.ranges.items[i - 1].len + len };
            i -= 1;
            _ = p.ranges.orderedRemove(i);
        }
        // the range after, when it starts where this one ends
        if (i < p.ranges.items.len and p.ranges.items[i].off == merged.off + merged.len) {
            merged.len += p.ranges.items[i].len;
            _ = p.ranges.orderedRemove(i);
        }
        p.ranges.insert(p.gpa, i, merged) catch std.log.warn("qwen3_5: a prompt-cache arena range did not merge", .{});
    }
};

test "an arena's ranges: take, give, merge with both neighbours, no fit" {
    const gpa = std.testing.allocator;
    const bytes = try gpa.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(unit), 8 * unit);
    defer gpa.free(bytes);
    const p = try Pool.over(gpa, bytes);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 8 * unit), p.size());

    const a = p.take(unit).?;
    const b = p.take(unit).?;
    const d = p.take(6 * unit).?;
    try std.testing.expectEqual(@as(usize, 6 * unit), d.len);
    try std.testing.expect(p.take(unit) == null); // none left
    p.give(b);
    p.give(a);
    try std.testing.expectEqual(@as(usize, 1), p.ranges.items.len); // merged across the two
    try std.testing.expectEqual(@as(usize, 2 * unit), p.ranges.items[0].len);
    const c = p.take(2 * unit).?;
    try std.testing.expect(p.take(unit) == null);
    p.give(c);
    p.give(d); // merges behind the two-unit range
    try std.testing.expectEqual(@as(usize, 1), p.ranges.items.len);
    try std.testing.expectEqual(@as(usize, 8 * unit), p.ranges.items[0].len);
    try std.testing.expect(p.take(8 * unit) != null);
}

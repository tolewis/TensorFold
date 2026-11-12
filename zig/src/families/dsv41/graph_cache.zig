//! A map with a cap on its entries: past the cap, a put hands back the least recently used entry for its owner to
//! free (round.Graphs: the captured stretches, each an instantiated CUDA graph on the device and the host).

const std = @import("std");

pub fn Lru(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        const Slot = struct { v: V, used: u64 };

        map: std.AutoHashMapUnmanaged(K, Slot) = .empty,
        cap: usize, // the entries kept at most (0: no cap)
        tick: u64 = 0,
        evicted: u64 = 0, // the entries put has handed back so far

        pub fn init(cap: usize) Self {
            return .{ .cap = cap };
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.map.deinit(gpa);
            self.* = undefined;
        }

        pub fn count(self: *const Self) usize {
            return self.map.count();
        }

        /// The entry of `k`, now the most recently used.
        pub fn get(self: *Self, k: K) ?*V {
            const s = self.map.getPtr(k) orelse return null;
            self.tick += 1;
            s.used = self.tick;
            return &s.v;
        }

        /// Keeps `v` under `k` (not in the map yet); at the cap first takes out the least recently used entry and
        /// returns its value, which the caller frees. A put that fails keeps every entry.
        pub fn put(self: *Self, gpa: std.mem.Allocator, k: K, v: V) !?V {
            try self.map.ensureUnusedCapacity(gpa, 1);
            var out: ?V = null;
            if (self.cap != 0 and self.map.count() >= self.cap) {
                // a scan, on a miss only (each costs a capture anyway): the cap keeps it short
                var it = self.map.iterator();
                var lru = it.next().?;
                while (it.next()) |en| if (en.value_ptr.used < lru.value_ptr.used) {
                    lru = en;
                };
                out = lru.value_ptr.v;
                self.map.removeByPtr(lru.key_ptr);
                self.evicted += 1;
            }
            self.tick += 1;
            self.map.putAssumeCapacityNoClobber(k, .{ .v = v, .used = self.tick });
            return out;
        }

        /// Every value kept, for the owner to free before deinit.
        pub fn valueIterator(self: *const Self) ValueIterator {
            return .{ .it = self.map.valueIterator() };
        }

        pub const ValueIterator = struct {
            it: std.AutoHashMapUnmanaged(K, Slot).ValueIterator,
            pub fn next(i: *ValueIterator) ?*V {
                const s = i.it.next() orelse return null;
                return &s.v;
            }
        };
    };
}

// A captured stretch's key as round.Graphs makes it; the values stand for the graphs' handles.
const Key = struct { rows: u32, bucket: u64, stretch: u32, mode: u8 = 0 };

fn key(rows: u32, bucket: u64, stretch: u32) Key {
    return .{ .rows = rows, .bucket = bucket, .stretch = stretch };
}

test "at the cap a put hands back the least recently used entry, a get counting as a use" {
    const gpa = std.testing.allocator;
    var c = Lru(Key, u32).init(3);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(1, 1024, 0), 10));
    try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(1, 1024, 1), 11));
    try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(1, 1024, 2), 12));
    // replaying the first stretch makes the second the least recently used
    try std.testing.expectEqual(@as(u32, 10), c.get(key(1, 1024, 0)).?.*);
    try std.testing.expectEqual(@as(?u32, 11), try c.put(gpa, key(2, 1024, 0), 20));
    try std.testing.expectEqual(@as(?*u32, null), c.get(key(1, 1024, 1)));
    try std.testing.expectEqual(@as(u32, 10), c.get(key(1, 1024, 0)).?.*);
    try std.testing.expectEqual(@as(u32, 12), c.get(key(1, 1024, 2)).?.*);
    try std.testing.expectEqual(@as(u32, 20), c.get(key(2, 1024, 0)).?.*);
    try std.testing.expectEqual(@as(usize, 3), c.count());
    try std.testing.expectEqual(@as(u64, 1), c.evicted);
}

test "cap 0 keeps every entry: the cache before GRAPH_CAP" {
    const gpa = std.testing.allocator;
    var c = Lru(Key, u32).init(0);
    defer c.deinit(gpa);
    for (0..600) |i| try std.testing.expectEqual(@as(?u32, null), try c.put(gpa, key(@intCast(i % 16 + 1), @as(u64, 1024) << @intCast(i / 48), @intCast(i % 3)), @intCast(i)));
    try std.testing.expectEqual(@as(usize, 600), c.count());
    try std.testing.expectEqual(@as(u64, 0), c.evicted);
}

test "over days of new shapes the cache stays at its cap and hands back each oldest graph once" {
    const gpa = std.testing.allocator;
    var c = Lru(Key, u32).init(128);
    defer c.deinit(gpa);
    // 600 distinct stretches, none replayed: the n-th put past the cap hands back the n-th one kept
    for (0..600) |i| {
        const out = try c.put(gpa, key(@intCast(i % 16 + 1), @as(u64, 1024) << @intCast(i / 48), @intCast(i % 3)), @intCast(i));
        try std.testing.expectEqual(if (i < 128) null else @as(?u32, @intCast(i - 128)), out);
        try std.testing.expect(c.count() <= 128);
    }
    try std.testing.expectEqual(@as(usize, 128), c.count());
    try std.testing.expectEqual(@as(u64, 472), c.evicted);
}

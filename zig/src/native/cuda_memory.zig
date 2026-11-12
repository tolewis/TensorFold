//! CUDA memory admission arithmetic and host-pool validation, without a device.
const std = @import("std");
const gib: f64 = 1 << 30;

/// What /proc/meminfo says: MemTotal and MemAvailable in bytes (the page cache counts as available).
pub const MemInfo = struct { total: u64, available: u64 };

pub fn meminfo(text: []const u8) ?MemInfo {
    var total: ?u64 = null;
    var available: ?u64 = null;
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, ": \t");
        const key = words.next() orelse continue;
        const is_total = std.mem.eql(u8, key, "MemTotal");
        if (!is_total and !std.mem.eql(u8, key, "MemAvailable")) continue;
        const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, words.next() orelse return null, "kB") or words.next() != null) return null;
        const bytes = std.math.mul(u64, kib, 1024) catch return null;
        if (is_total) {
            if (total != null) return null;
            total = bytes;
        } else {
            if (available != null) return null;
            available = bytes;
        }
    }
    const t = total orelse return null;
    const a = available orelse return null;
    return if (t > 0 and a <= t) .{ .total = t, .available = a } else null;
}

/// What the per-CPU page lists hold at rest when /proc/zoneinfo prints no high_min (kernels before 6.7).
pub const parked_rest: u64 = 1 << 29;

/// Bytes free on the kernel's per-CPU page lists beyond what they hold at rest: the `count:` lines under each zone's
/// `pagesets` less their `high_min:` lines (parked_rest when there are none). A bulk free (an image copy, a compiler's
/// exit) leaves GiBs there on kernel 6.7+; they are free and the kernel drains them when an allocation needs them, but
/// neither MemFree nor MemAvailable counts them and they drain back at ~0.05-0.1 GiB/s. A warm start parks ~0.3 GiB,
/// which the budget never counted: that stays ~0. 0 when the text has no page lists or a field does not parse.
/// (Port of wojo's GLM 0119, PR #149.)
pub fn parkedBytes(text: []const u8, page: u64) u64 {
    var count: u64 = 0;
    var rest: u64 = 0;
    var listed = false;
    var rows = std.mem.splitScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, " \t\r");
        const first = words.next();
        if (std.mem.eql(u8, std.mem.trim(u8, row, " \t\r"), "pagesets")) {
            listed = true;
        } else if (listed and !std.mem.startsWith(u8, row, "    ") and !std.mem.startsWith(u8, row, "  vm stats")) {
            listed = false; // the next zone's or the next field of this one
        } else if (listed and first != null and (std.mem.eql(u8, first.?, "count:") or std.mem.eql(u8, first.?, "high_min:"))) {
            const n = std.fmt.parseInt(u64, words.next() orelse return 0, 10) catch return 0;
            const sum = if (first.?[0] == 'c') &count else &rest;
            sum.* = std.math.add(u64, sum.*, n) catch return 0;
        }
    }
    if (count == 0) return 0;
    const held = std.math.mul(u64, count, page) catch return 0;
    const resting = if (rest > 0) std.math.mul(u64, rest, page) catch return 0 else parked_rest;
    return held -| resting;
}

/// A /proc/meminfo reading with `parked` (parkedBytes) added to MemAvailable, at most MemTotal.
pub fn counting(m: MemInfo, parked: u64) MemInfo {
    return .{ .total = m.total, .available = @min(m.total, m.available +| parked) };
}

/// CUDA allocation room after the reserve and optional process-memory cap.
pub const Pool = struct {
    free: u64,
    total: u64,
    reserve: u64,
    limit: ?u64,
    unified: bool,
    parked: u64 = 0, // counted into `free`: pages on the per-CPU lists past their resting level (parkedBytes)

    /// Bytes the engine may still allocate on top of `held`.
    pub fn room(p: Pool, held: u64) u64 {
        const left = p.free -| p.reserve;
        return if (p.limit) |cap| @min(left, cap -| held) else left;
    }
};

/// TENSORFOLD_MEMORY_RESERVE_GIB (2 GiB up to the pool's size), else a tenth of the pool and at least 4 GiB.
pub fn reserveBytes(text: ?[]const u8, total: u64) error{Invalid}!u64 {
    const t = std.mem.trim(u8, text orelse "", " ");
    if (t.len == 0) return @max(4 << 30, total / 10);
    const g = std.fmt.parseFloat(f64, t) catch return error.Invalid;
    if (!(g >= 2)) return error.Invalid;
    const bytes = try scaledBytes(g);
    return if (bytes <= total) bytes else error.Invalid;
}

/// TENSORFOLD_CUDA_MEMORY_LIMIT_GB in bytes; null when unset.
pub fn limitBytes(text: ?[]const u8) error{Invalid}!?u64 {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    const g = std.fmt.parseFloat(f64, t) catch return error.Invalid;
    return try scaledBytes(g);
}

fn scaledBytes(g: f64) error{Invalid}!u64 {
    const bytes = g * gib;
    const upper: f64 = 0x1p64;
    if (!std.math.isFinite(bytes) or bytes < 1 or bytes >= upper) return error.Invalid;
    return @intFromFloat(bytes);
}

/// Streams the room fits, as many as asked when they all fit; an error when none does, or a fixed --parallel doesn't.
pub fn admit(room: u64, stream: u64, asked: u32, fixed: bool) error{ NoStream, TooMany }!u32 {
    const fits = if (stream == 0) asked else std.math.cast(u32, room / stream) orelse std.math.maxInt(u32);
    if (fits == 0) return error.NoStream;
    if (fits >= asked) return asked;
    return if (fixed) error.TooMany else fits;
}

/// Integrated GPUs require validated host counts; discrete devices use CUDA's counts.
pub fn counts(unified: bool, card: MemInfo, text: ?[]const u8) error{HostMemoryUnavailable}!MemInfo {
    if (unified) return meminfo(text orelse return error.HostMemoryUnavailable) orelse error.HostMemoryUnavailable;
    return card;
}

test "the memory plan: reserve, cap, and admission that fails closed" {
    const g: u64 = 1 << 30;
    try std.testing.expectEqual(4 * g, try reserveBytes(null, 24 * g)); // at least 4 GiB
    try std.testing.expectEqual(12 * g, try reserveBytes("", 120 * g)); // a tenth
    try std.testing.expectEqual(3 * g, try reserveBytes("3", 120 * g));
    try std.testing.expectError(error.Invalid, reserveBytes("1", 120 * g));
    try std.testing.expectError(error.Invalid, reserveBytes("200", 120 * g));
    try std.testing.expectEqual(@as(?u64, null), try limitBytes(null));
    try std.testing.expectEqual(@as(?u64, 40 * g), try limitBytes("40"));
    try std.testing.expectError(error.Invalid, limitBytes("0"));
    const p: Pool = .{ .free = 70 * g, .total = 96 * g, .reserve = 10 * g, .limit = null, .unified = false };
    try std.testing.expectEqual(60 * g, p.room(20 * g));
    const capped: Pool = .{ .free = 70 * g, .total = 96 * g, .reserve = 10 * g, .limit = 30 * g, .unified = false };
    try std.testing.expectEqual(10 * g, capped.room(20 * g)); // the cap less what the engine holds
    try std.testing.expectEqual(@as(u32, 8), try admit(60 * g, g, 8, false));
    try std.testing.expectEqual(@as(u32, 5), try admit(5 * g + 1, g, 8, false)); // auto serves what fits
    try std.testing.expectError(error.TooMany, admit(5 * g, g, 8, true)); // a fixed --parallel refuses
    try std.testing.expectError(error.NoStream, admit(g - 1, g, 8, false));
}

test "meminfo: total and available in bytes" {
    const m = meminfo("MemTotal:       131072 kB\nMemFree:          1024 kB\nMemAvailable:    65536 kB\n").?;
    try std.testing.expectEqual(@as(u64, 131072 * 1024), m.total);
    try std.testing.expectEqual(@as(u64, 65536 * 1024), m.available);
    try std.testing.expect(meminfo("MemTotal: 5 kB\n") == null);
}

test "unified admission refuses missing or malformed host memory" {
    const card: MemInfo = .{ .total = 128 << 30, .available = 100 << 30 };
    for ([_]?[]const u8{
        null,
        "",
        "MemTotal: 131072 kB\n",
        "MemTotal: bad kB\nMemAvailable: 1024 kB\n",
        "MemTotal: 131072 MB\nMemAvailable: 1024 MB\n",
        "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
        "MemTotal: 1024 kB\nMemAvailable: 2048 kB\n",
        "MemTotal: 18014398509481984 kB\nMemAvailable: 1024 kB\n",
        "MemTotal: 131072 kB\nMemTotal: 131072 kB\nMemAvailable: 1024 kB\n",
    }) |text| {
        try std.testing.expectError(error.HostMemoryUnavailable, counts(true, card, text));
    }
    try std.testing.expectEqual(card, try counts(false, card, null));
    try std.testing.expectEqual(MemInfo{ .total = 131072 * 1024, .available = 65536 * 1024 }, try counts(true, card, "MemTotal: 131072 kB\nMemAvailable: 65536 kB\n"));
}

test "memory limit refuses an oversized finite value and the rounded u64 boundary" {
    try std.testing.expectError(error.Invalid, limitBytes("1e300"));
    try std.testing.expectError(error.Invalid, limitBytes("17179869184"));
    try std.testing.expectError(error.Invalid, limitBytes("17179869183.9999995")); // parses to 2^34: 2^64 bytes
    try std.testing.expectEqual(@as(?u64, std.math.maxInt(u64) - 2047), try limitBytes("17179869183.9999980926513671875"));
}

/// tools/parked_pages_check.py's /proc/zoneinfo blocks: one zone's stats, its page lists when `counts` is given.
fn zoneText(a: std.mem.Allocator, out: *std.ArrayList(u8), node: u32, name: []const u8, counts_: ?[]const u64, high_min: ?u64) !void {
    try out.print(a, "Node {d}, zone {s:>8}\n  pages free     10\n        managed  1000\n      nr_free_cma  0\n", .{ node, name });
    if (counts_) |cs| {
        try out.appendSlice(a, "  pagesets\n");
        for (cs, 0..) |c, n| {
            try out.print(a, "    cpu: {d}\n              count:    {d}\n              high:     5000\n              batch:    63\n", .{ n, c });
            if (high_min) |h| try out.print(a, "              high_min: {d}\n", .{h});
            try out.appendSlice(a, "              high_max: 198283\n  vm stats threshold: 110\n");
        }
    }
    try out.appendSlice(a, "  start_pfn:           4096\n");
}

test "parked pages: every CPU, zone and node, less each list's resting level" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    const hm: u64 = 2146; // a CPU list's high_min as the kernel prints it on a Spark
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    const ca = [_]u64{ 200000, 3143, 2332, 17 };
    const cb = [_]u64{ 100000, 100000, 100000 };
    const cc = [_]u64{400000};
    try zoneText(a, &t, 0, "DMA", &ca, hm);
    try zoneText(a, &t, 0, "DMA32", null, hm);
    try zoneText(a, &t, 0, "Normal", &cb, hm);
    try zoneText(a, &t, 1, "Normal", &cc, hm);
    try zoneText(a, &t, 1, "Movable", null, hm);
    var sum: u64 = 0;
    for (ca ++ cb ++ cc) |c| sum += c;
    const excess = (sum - 8 * hm) * page;
    try std.testing.expectEqual(excess, parkedBytes(t.items, page));
    const m = counting(.{ .total = 128 << 30, .available = 100 << 30 }, excess);
    try std.testing.expectEqual(@as(u64, (100 << 30) + excess), m.available);
    try std.testing.expectEqual(@as(u64, 128 << 30), m.total);
    try std.testing.expectEqual(@as(u64, 128 << 30), counting(.{ .total = 128 << 30, .available = 120 << 30 }, 200 << 30).available); // capped at MemTotal
}

test "parked pages: an image copy's 6 GiB raise the budget by 6 GiB; a warm start counts ~nothing" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    const hm: u64 = 2146;
    const six = (6 << 30) / page;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    try zoneText(a, &t, 0, "Normal", &.{ six / 4 + hm, six / 4 + hm, six / 4 + hm, six / 4 + hm }, hm);
    try std.testing.expectEqual(@as(u64, 6 << 30), parkedBytes(t.items, page));
    const warm = [_][20]u64{ @splat(hm), @splat(hm + 1440), @splat(0) };
    for (warm) |cs| {
        t.clearRetainingCapacity();
        try zoneText(a, &t, 0, "Normal", &cs, hm);
        try std.testing.expect(parkedBytes(t.items, page) < 1 << 28); // under the log line's 0.25 GiB
    }
    t.clearRetainingCapacity();
    try zoneText(a, &t, 0, "Normal", &@as([20]u64, @splat(100)), hm);
    try std.testing.expectEqual(@as(u64, 0), parkedBytes(t.items, page)); // below the resting level: nothing
}

test "parked pages: no high_min (before 6.7) rests the first 512 MiB" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    const half = ((1 << 30) + (1 << 29)) / page / 2;
    try zoneText(a, &t, 0, "Normal", &.{ half, half }, null);
    try std.testing.expectEqual(@as(u64, 1 << 30), parkedBytes(t.items, page));
    t.clearRetainingCapacity();
    try zoneText(a, &t, 0, "Normal", &.{ 1000, 1000, 1000, 1000 }, null);
    try std.testing.expectEqual(@as(u64, 0), parkedBytes(t.items, page));
}

test "parked pages: only the count: and high_min: lines under pagesets" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    try t.appendSlice(a, "  count: 5\n      count: 6\n");
    try zoneText(a, &t, 0, "Normal", &.{2146 + 10}, 2146);
    try t.appendSlice(a, "      count:    99999\n      high_min: 7\n"); // after start_pfn: not a list
    try t.appendSlice(a, "Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n");
    try std.testing.expectEqual(10 * page, parkedBytes(t.items, page));
    // a list that ends at the next field of its zone
    const other = "Node 0, zone Normal\n  pagesets\n    cpu: 0\n              count: 2200\n              high: 9\n" ++
        "              high_min: 2000\n  vm stats threshold: 8\n    cpu: 1\n              count: 2100\n" ++
        "              high_min: 2000\n  all_unreclaimable: 0\n";
    try std.testing.expectEqual(300 * page, parkedBytes(other, page));
}

test "parked pages: no lists, empty, garbled or binary text counts nothing" {
    const a = std.testing.allocator;
    const page: u64 = 4096;
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(a);
    try zoneText(a, &t, 0, "DMA", null, 2146);
    try zoneText(a, &t, 0, "Normal", null, 2146);
    try std.testing.expectEqual(@as(u64, 0), parkedBytes(t.items, page));
    try std.testing.expectEqual(@as(u64, 0), parkedBytes("", page));
    var bad: std.ArrayList(u8) = .empty;
    defer bad.deinit(a);
    try zoneText(a, &bad, 0, "Normal", &.{1000000}, 2146);
    for ([_][2][]const u8{ .{ "count:    1000000", "count:    x" }, .{ "count:    1000000", "count:" }, .{ "high_min: 2146", "high_min: x" }, .{ "count:    1000000", "count:    \xff\xfe" } }) |swap| {
        const garbled = try std.mem.replaceOwned(u8, a, bad.items, swap[0], swap[1]);
        defer a.free(garbled);
        try std.testing.expectEqual(@as(u64, 0), parkedBytes(garbled, page));
    }
    try std.testing.expectEqual(@as(u64, 0), parkedBytes("\x00\xff\xfe" ++ "\x00\xff\xfe" ++ "\x00\xff\xfe\n  pagesets\x00", page));
    try std.testing.expectEqual(@as(u64, 0), parkedBytes("Node 0, zone Normal\n  pagesets\n    cpu: 0\n      count: 18446744073709551615\n      count: 1\n", page)); // overflow: nothing, not a wrap
}

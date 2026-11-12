//! The memory gate before a prompt is rendered and tokenized (rank 0's host side). A conversation's text is rendered
//! twice and tokenized twice (prompt.prepare: the prompt and its history), so a resumed conversation of a million
//! tokens needs memory in proportion to its text while it prepares, and on a unified-memory machine (a GB10, where the
//! KV pool leaves a few GiB free) eight of them at once can take MemAvailable to zero. Each prepare reserves
//! `per_byte` bytes for every byte of its text; a prepare waits while others are in flight and MemAvailable, less
//! what the ones in flight reserved, would fall under the floor with its own reservation. A prepare with none in
//! flight always goes (it can never wait on itself), so the gate only ever turns concurrent prepares into queued ones;
//! it changes no reply. Off with a floor of 0, and where MemAvailable cannot be read (no /proc/meminfo).
const std = @import("std");
const json = @import("json.zig");
const log = @import("log.zig");
const Value = json.Value;

/// The variable that sets the floor (serve.zig reads it; capabilities lists it beside cli.env).
pub const env = [_][]const u8{"TENSORFOLD_PREPARE_FLOOR_GIB"};
/// TENSORFOLD_PREPARE_FLOOR_GIB unset: the MemAvailable a prepare leaves at least, in GiB.
pub const default_floor_gib: f64 = 3;
/// Bytes reserved a text byte: the two renders' and tokenizations' transient scratch (~11 and ~6 bytes a text byte,
/// one at a time) over what the request holds meanwhile (its body, the parsed messages, two rendered texts and two
/// id lists: ~8), with room to spare.
pub const default_per_byte: u64 = 24;
/// How often a waiting prepare reads MemAvailable again.
const poll_ms = 10;

pub const Gate = struct {
    mutex: std.Io.Mutex = .init,
    floor: u64 = 0, // bytes; 0: off
    per_byte: u64 = default_per_byte,
    reserved: u64 = 0,
    inflight: u32 = 0,
    /// Prepares that had to wait, and the longest wait (ns): for the log and the tests.
    waited: u64 = 0,
    longest_ns: u64 = 0,
    /// MemAvailable in bytes, null when it cannot be read (the gate lets everything through then).
    avail: *const fn (io: std.Io) ?u64 = memAvailable,

    pub const Ticket = struct { need: u64 = 0, held: bool = false };

    /// A gate leaving `floor_gib` GiB (0: off).
    pub fn init(floor_gib: f64) Gate {
        return .{ .floor = if (floor_gib > 0) @intFromFloat(floor_gib * (1 << 30)) else 0 };
    }

    /// The reservation of a prepare of `text_bytes` bytes of text.
    pub fn need(g: *const Gate, text_bytes: usize) u64 {
        return std.math.mul(u64, text_bytes, g.per_byte) catch std.math.maxInt(u64);
    }

    /// Waits until a prepare reserving `bytes` fits (or none is in flight), then holds its reservation until leave.
    /// error.Cancelled when `gone` says the client left while it waited.
    pub fn enter(g: *Gate, io: std.Io, bytes: u64, gone: anytype) error{Cancelled}!Ticket {
        if (g.floor == 0) return .{};
        var started: ?i96 = null;
        while (true) {
            g.mutex.lockUncancelable(io);
            if (g.fits(io, bytes)) {
                g.reserved +|= bytes;
                g.inflight += 1;
                if (started) |t0| {
                    const ns: u64 = @intCast(@max(0, std.Io.Clock.awake.now(io).toNanoseconds() - t0));
                    g.waited += 1;
                    g.longest_ns = @max(g.longest_ns, ns);
                }
                g.mutex.unlock(io);
                return .{ .need = bytes, .held = true };
            }
            const queued = g.inflight;
            const reserved = g.reserved;
            g.mutex.unlock(io);
            if (started == null) {
                started = std.Io.Clock.awake.now(io).toNanoseconds();
                log.line("prepare waits for memory: {d} MiB reserved for it, {d} prepares in flight hold {d} MiB, floor {d} MiB", .{ bytes >> 20, queued, reserved >> 20, g.floor >> 20 });
            }
            if (gone.check()) return error.Cancelled;
            std.Io.sleep(io, .fromMilliseconds(poll_ms), .awake) catch {};
        }
    }

    /// (locked) A reservation of `bytes` leaves MemAvailable, less every reservation in flight, at the floor or over.
    fn fits(g: *Gate, io: std.Io, bytes: u64) bool {
        if (g.inflight == 0) return true;
        const avail = g.avail(io) orelse return true;
        return avail >= g.reserved +| bytes +| g.floor;
    }

    pub fn leave(g: *Gate, io: std.Io, t: Ticket) void {
        if (!t.held) return;
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        g.reserved -|= t.need;
        g.inflight -|= 1;
    }
};

/// TENSORFOLD_PREPARE_FLOOR_GIB: GiB, 0 turns the gate off; unset or blank: the default.
pub fn floorGib(raw: ?[]const u8) error{BadPrepareFloor}!f64 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default_floor_gib;
    const v = std.fmt.parseFloat(f64, t) catch return error.BadPrepareFloor;
    if (!(v >= 0) or v > 1 << 20) return error.BadPrepareFloor;
    return v;
}

/// The bytes of text a request renders: every string in `v` (message contents, tool specs, names), keys included.
pub fn textBytes(v: Value) usize {
    return switch (v) {
        .string => |s| s.len,
        .int => |s| s.len,
        .array => |items| blk: {
            var n: usize = 0;
            for (items) |x| n +|= textBytes(x);
            break :blk n;
        },
        .object => |o| blk: {
            var n: usize = 0;
            for (o.keys(), o.values()) |k, x| n +|= k.len +| textBytes(x);
            break :blk n;
        },
        else => 8,
    };
}

/// /proc/meminfo's MemAvailable in bytes (null where there is none, or it does not parse).
pub fn memAvailable(io: std.Io) ?u64 {
    var buf: [4096]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null;
    return parseAvailable(text);
}

pub fn parseAvailable(text: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var it = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const n = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
        const unit = it.next() orelse return null;
        if (!std.mem.eql(u8, unit, "kB")) return null;
        return std.math.mul(u64, n, 1024) catch null;
    }
    return null;
}

const Never = struct {
    pub fn check(_: Never) bool {
        return false;
    }
};

test "MemAvailable from /proc/meminfo's text" {
    try std.testing.expectEqual(@as(?u64, 6 << 30), parseAvailable("MemTotal:       127535424 kB\nMemFree:         1048576 kB\nMemAvailable:    6291456 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemTotal: 1 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemAvailable: 12 MB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemAvailable: x kB\n"));
}

test "the floor's setting" {
    try std.testing.expectEqual(default_floor_gib, try floorGib(null));
    try std.testing.expectEqual(default_floor_gib, try floorGib(" "));
    try std.testing.expectEqual(@as(f64, 0), try floorGib("0"));
    try std.testing.expectEqual(@as(f64, 2.5), try floorGib("2.5"));
    try std.testing.expectError(error.BadPrepareFloor, floorGib("-1"));
    try std.testing.expectError(error.BadPrepareFloor, floorGib("nan"));
    try std.testing.expectError(error.BadPrepareFloor, floorGib("lots"));
    try std.testing.expectEqual(@as(u64, 0), Gate.init(0).floor);
    try std.testing.expectEqual(@as(u64, 3 << 30), Gate.init(3).floor);
}

test "a request's text bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hello\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hi there\"}]}]");
    try std.testing.expect(parsed == .ok);
    // keys and strings: role+user, content+hello; role+assistant, content, type+text, text+hi there
    try std.testing.expectEqual(@as(usize, 4 + 4 + 7 + 5 + 4 + 9 + 7 + 4 + 4 + 4 + 8), textBytes(parsed.ok));
}

var fake_avail: std.atomic.Value(u64) = .init(0);
fn fakeAvail(_: std.Io) ?u64 {
    return fake_avail.load(.acquire);
}
fn noAvail(_: std.Io) ?u64 {
    return null;
}

test "the first prepare always goes; a second waits while it would take MemAvailable under the floor" {
    const io = std.testing.io;
    var g = Gate.init(3);
    g.avail = fakeAvail;
    fake_avail.store(4 << 30, .release);
    // the first goes whatever MemAvailable says: none is in flight
    const a = try g.enter(io, 2 << 30, Never{});
    try std.testing.expect(a.held);
    // a second of 1 GiB fits only with 2 (reserved) + 1 + 3 (floor) = 6 GiB available
    const Waiter = struct {
        fn run(gate: *Gate, done: *std.atomic.Value(bool), t: *Gate.Ticket) void {
            t.* = gate.enter(std.testing.io, 1 << 30, Never{}) catch unreachable;
            done.store(true, .release);
        }
    };
    var done: std.atomic.Value(bool) = .init(false);
    var second: Gate.Ticket = .{};
    const th = try std.Thread.spawn(.{}, Waiter.run, .{ &g, &done, &second });
    std.Io.sleep(io, .fromMilliseconds(5 * poll_ms), .awake) catch {};
    try std.testing.expect(!done.load(.acquire));
    // the first leaves: the second goes (none in flight)
    g.leave(io, a);
    th.join();
    try std.testing.expect(done.load(.acquire) and second.held);
    try std.testing.expectEqual(@as(u64, 1), g.waited);
    try std.testing.expectEqual(@as(u64, 1 << 30), g.reserved);
    // with room, a third goes beside it at once
    fake_avail.store(16 << 30, .release);
    const third = try g.enter(io, 1 << 30, Never{});
    try std.testing.expectEqual(@as(u32, 2), g.inflight);
    g.leave(io, third);
    g.leave(io, second);
    try std.testing.expectEqual(@as(u64, 0), g.reserved);
    try std.testing.expectEqual(@as(u32, 0), g.inflight);
}

test "a waiting prepare whose client leaves is cancelled; off and unreadable gates never wait" {
    const io = std.testing.io;
    var g = Gate.init(3);
    g.avail = fakeAvail;
    fake_avail.store(1 << 30, .release);
    const a = try g.enter(io, 1 << 30, Never{});
    const Left = struct {
        pub fn check(_: @This()) bool {
            return true;
        }
    };
    try std.testing.expectError(error.Cancelled, g.enter(io, 1 << 30, Left{}));
    try std.testing.expectEqual(@as(u32, 1), g.inflight);
    g.leave(io, a);
    var off = Gate.init(0);
    off.avail = fakeAvail;
    const x = try off.enter(io, 1 << 40, Never{});
    const y = try off.enter(io, 1 << 40, Never{});
    try std.testing.expect(!x.held and !y.held);
    off.leave(io, x);
    var blind = Gate.init(3);
    blind.avail = noAvail;
    const p = try blind.enter(io, 1 << 40, Never{});
    const q = try blind.enter(io, 1 << 40, Never{});
    blind.leave(io, p);
    blind.leave(io, q);
    try std.testing.expectEqual(@as(u32, 0), blind.inflight);
}

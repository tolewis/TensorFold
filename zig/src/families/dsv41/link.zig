//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
//! The other direction carries only the followers' replies to rank 0's asking rank_wire frames (recvWithin,
//! readableAny: bounded waits; see rank_wire.zig and docs/port/rank-wire.md for the rules that keep it deadlock-free).
//! Rank 0 also watches the followers' links for a close (Watch): a follower that died breaks the pair at once.
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (rank 0 to rank 1, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        // (Linux's; a host without it, where only the host tests run, keeps the plain socket)
        if (@hasDecl(c.SO, "BUSY_POLL")) _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        var l = try Listener.bind(ip, port);
        errdefer l.close();
        return l.accept();
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            var a = addr(ip, port);
            if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) return .{ .fd = fd };
            _ = c.close(fd);
            if (tries >= seconds * 4) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    /// A port rank 0 listens on for one peer, before it takes the connection: a peer that connects in the meantime
    /// is completed by the kernel (backlog 1) and waits there until `accept`, so rank 0 can open its ports first
    /// (before it loads its weights) and take the connections afterwards.
    pub const Listener = struct {
        fd: c.fd_t,

        pub fn bind(ip: [4]u8, port: u16) !Listener {
            const fd = try socket();
            errdefer _ = c.close(fd);
            const one: c_int = 1;
            _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
            var a = addr(ip, port);
            if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
            if (c.listen(fd, 1) != 0) return error.ListenFailed;
            return .{ .fd = fd };
        }

        /// The peer's connection (waited for if it has not come yet); the port is closed once it is taken.
        pub fn accept(l: *Listener) !Link {
            const peer = c.accept(l.fd, null, null);
            if (peer < 0) return error.AcceptFailed;
            l.close();
            tune(peer);
            return .{ .fd = peer };
        }

        pub fn close(l: *Listener) void {
            if (l.fd >= 0) _ = c.close(l.fd);
            l.fd = -1;
        }
    };

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// Whether a byte can be read within `timeout_ns` (poll(2); a hang-up or socket error counts as readable, so the
    /// read that follows reports it). Never consumes anything.
    pub fn readable(l: Link, timeout_ns: u64) !bool {
        var one = [_]Link{l};
        var ready = [_]bool{false};
        return try readableAny(&one, &ready, timeout_ns) > 0;
    }

    /// recv with a deadline over the whole frame (rank-to-rank replies, rank_wire.zig): error.Timeout when the frame
    /// does not arrive in full within `timeout_ns`. After a timeout the link may hold part of a frame and is no longer
    /// usable: the caller ends the pair (there is no resynchronizing a byte stream).
    pub fn recvWithin(l: Link, buf: []u8, timeout_ns: u64) ![]u8 {
        const deadline = monotonicNs() +| timeout_ns;
        var len: [4]u8 = undefined;
        try l.readAllBy(&len, deadline);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAllBy(buf[0..n], deadline);
        return buf[0..n];
    }

    fn readAllBy(l: Link, out: []u8, deadline: u64) !void {
        var done: usize = 0;
        while (done < out.len) {
            const now = monotonicNs();
            if (now >= deadline) return error.Timeout;
            if (!try l.readable(deadline - now)) return error.Timeout;
            const n = c.recv(l.fd, out.ptr + done, out.len - done, c.MSG.DONTWAIT);
            if (n > 0) {
                done += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN, .INTR => {},
                else => return error.PeerDown,
            }
        }
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// Which of `links` have a byte to read (or a hang-up/error to report) within `timeout_ns`: their flags in `ready`
/// (one per link), the count. 0 when none did in time. A timeout of 0 only polls.
pub fn readableAny(links: []const Link, ready: []bool, timeout_ns: u64) !usize {
    std.debug.assert(ready.len == links.len and links.len <= 8);
    var fds: [8]c.pollfd = undefined;
    for (links, 0..) |l, i| fds[i] = .{ .fd = l.fd, .events = c.POLL.IN, .revents = 0 };
    const deadline = Link.monotonicNs() +| timeout_ns;
    while (true) {
        const now = Link.monotonicNs();
        const left = if (deadline > now) deadline - now else 0;
        // poll's milliseconds, rounded up so a short wait still waits; capped well inside c_int
        const ms: c_int = @intCast(@min((left + std.time.ns_per_ms - 1) / std.time.ns_per_ms, 1 << 30));
        const n = c.poll(&fds, @intCast(links.len), ms);
        if (n < 0) {
            if (c.errno(n) == .INTR) continue;
            return error.PollFailed;
        }
        var count: usize = 0;
        for (ready, 0..) |*r, i| {
            r.* = fds[i].revents != 0;
            if (r.*) count += 1;
        }
        if (count > 0 or Link.monotonicNs() >= deadline) return count;
    }
}

/// Linux reports a peer's close (FIN or reset) as POLLRDHUP even while unread bytes wait; elsewhere (the host tests'
/// macOS) a close is the peeked end of the stream, seen once nothing is left to read.
const rdhup: i16 = if (builtin.os.tag == .linux) 0x2000 else 0; // POLLRDHUP (asm-generic/poll.h; not in std's aarch64 POLL)

/// Which of `links` is gone (the peer closed it, reset it, or it errs), looking for up to `ms` (0: only now); null if
/// none. Never consumes a byte: replies still waiting stay for their reader.
pub fn gone(links: []const Link, ms: c_int) ?usize {
    std.debug.assert(links.len <= 8);
    var fds: [8]c.pollfd = undefined;
    const ev: i16 = if (rdhup != 0) rdhup else c.POLL.IN;
    for (links, 0..) |l, i| fds[i] = .{ .fd = l.fd, .events = ev, .revents = 0 };
    const n = c.poll(&fds, @intCast(links.len), ms);
    if (n <= 0) return null;
    for (fds[0..links.len], 0..) |f, i| {
        if (f.revents & (c.POLL.HUP | c.POLL.ERR | c.POLL.NVAL | rdhup) != 0) return i;
        if (rdhup == 0 and f.revents & c.POLL.IN != 0) {
            var b: [1]u8 = undefined;
            const r = c.recv(f.fd, &b, 1, c.MSG.PEEK | c.MSG.DONTWAIT);
            if (r == 0) return i; // the end of the stream
            if (r < 0) switch (c.errno(r)) {
                .AGAIN, .INTR => {},
                else => return i,
            };
        }
    }
    // bytes wait to be read and nothing closed: elsewhere than Linux the next poll would return at once, so wait here
    if (rdhup == 0 and ms > 0) _ = c.poll(&fds, 0, ms);
    return null;
}

/// Rank 0's watch over its followers' links (TF_DS_PAIR_BROKEN=exit): a follower whose process ended (killed,
/// crashed) closes its socket at once, so the pair is known broken within `period_ms` instead of the stall monitor's
/// bound, even while rank 0's engine thread is blocked in a round on that peer. `lost` runs once, on this thread, with
/// the follower's index (0: rank 1), `grace_ms` after the close was seen (a stop of both ranks at once, stop.sh's,
/// reaches rank 0 meanwhile and ends as a shutdown); then the watch ends. A stopped follower keeps its socket open:
/// that stays the stall monitor's. End it (`end`) before rank 0 itself closes the links.
pub const Watch = struct {
    links: [7]Link = undefined,
    n: usize = 0,
    ctx: *anyopaque,
    lost: *const fn (ctx: *anyopaque, follower: usize) void,
    period_ms: c_int = 100,
    grace_ms: u32 = 500,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn init(followers: []const ?Link, ctx: *anyopaque, lost: *const fn (*anyopaque, usize) void) Watch {
        var w: Watch = .{ .ctx = ctx, .lost = lost };
        for (followers) |f| if (f) |l| {
            w.links[w.n] = l;
            w.n += 1;
        };
        return w;
    }

    pub fn start(w: *Watch) !void {
        if (w.n == 0) return;
        w.thread = try std.Thread.spawn(.{}, run, .{w});
    }

    /// Stops the watch: it never reports after this returns. Idempotent.
    pub fn end(w: *Watch) void {
        w.stop.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    fn run(w: *Watch) void {
        while (!w.stop.load(.acquire)) {
            const i = gone(w.links[0..w.n], w.period_ms) orelse continue;
            var none: [1]c.pollfd = undefined; // (poll over no descriptors: a 10 ms sleep)
            var waited: u32 = 0;
            while (waited < w.grace_ms and !w.stop.load(.acquire)) : (waited += 10) _ = c.poll(&none, 0, 10);
            if (!w.stop.load(.acquire)) w.lost(w.ctx, i);
            return;
        }
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("192.0.2"));
}

test "frames cross a loopback link in order" {
    const io = std.testing.io;
    const port: u16 = 29000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 500)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

test "a peer that connects before rank 0 takes its port waits in the backlog, and the port closes once taken" {
    const io = std.testing.io;
    const port: u16 = 27700 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 90)));
    var bound = try Link.Listener.bind(.{ 127, 0, 0, 1 }, port);
    defer bound.close();
    // the peer connects now, while nothing accepts: before this change rank 0 listened only after its weights had
    // loaded, and a peer that was ready earlier gave up after its retry window
    var peer = try Link.connect(io, .{ 127, 0, 0, 1 }, port, 1);
    defer peer.close();
    try peer.send("early");
    var l = try bound.accept();
    defer l.close();
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("early", try l.recv(&buf));
    try std.testing.expectEqual(@as(c.fd_t, -1), bound.fd);
    // the port is free again after the accept
    try std.testing.expectError(error.ConnectFailed, Link.connect(io, .{ 127, 0, 0, 1 }, port, 0));
}

test "a reply that does not come in time is a timeout, a readable link says so" {
    const io = std.testing.io;
    const port: u16 = 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch return; // the go-ahead
            // half a frame, then nothing for a while, then the rest
            l.writeAll(&.{ 5, 0, 0, 0, 'h' }) catch return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(300), .awake) catch {};
            l.writeAll("ello") catch return;
            _ = l.recv(&buf) catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [16]u8 = undefined;
    try std.testing.expect(!try l.readable(20 * std.time.ns_per_ms));
    var ready = [_]bool{true};
    try std.testing.expectEqual(@as(usize, 0), try readableAny(&.{l}, &ready, 0));
    try std.testing.expect(!ready[0]);
    try l.send("go");
    try std.testing.expectError(error.Timeout, l.recvWithin(&buf, 100 * std.time.ns_per_ms));
    try std.testing.expect(try l.readable(2 * std.time.ns_per_s)); // the rest arrives
    try l.send("end");
    t.join();
    // the peer is gone: readable (a hang-up), then the read says so
    try std.testing.expect(try l.readable(std.time.ns_per_s));
    var rest: [8]u8 = undefined;
    _ = c.recv(l.fd, &rest, rest.len, 0); // "ello"
    try std.testing.expectError(error.PeerDown, l.recvWithin(&buf, std.time.ns_per_s));
}

/// What a test's Watch saw: how many reports, the follower, and when (Link.monotonicNs).
const Seen = struct {
    n: std.atomic.Value(u32) = .init(0),
    follower: std.atomic.Value(usize) = .init(99),
    at: std.atomic.Value(u64) = .init(0),

    fn lost(ctx: *anyopaque, follower: usize) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.follower.store(follower, .release);
        s.at.store(Link.monotonicNs(), .release);
        _ = s.n.fetchAdd(1, .acq_rel);
    }
    fn waitReport(s: *Seen, ms: u32) bool {
        var waited: u32 = 0;
        while (s.n.load(.acquire) == 0 and waited < ms) : (waited += 5) std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        return s.n.load(.acquire) > 0;
    }
};

/// A follower that sends a reply, waits for "go", then ends (its socket closes as a killed process's would).
const Ending = struct {
    fn run(p: u16) void {
        var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
        l.send("reply") catch {};
        var buf: [8]u8 = undefined;
        _ = l.recv(&buf) catch {};
        l.close();
    }
};

test "the watch sees a follower's link close within its period and grace, once; a reply waiting is no close" {
    const io = std.testing.io;
    const port: u16 = 28100 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const t = try std.Thread.spawn(.{}, Ending.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    try std.testing.expect(try l.readable(2 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(?usize, null), gone(&.{l}, 50)); // the unread reply is not a close
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("reply", try l.recv(&buf));
    var seen: Seen = .{};
    var w = Watch.init(&.{@as(?Link, l)}, &seen, Seen.lost);
    w.grace_ms = 200;
    try w.start();
    defer w.end();
    std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
    try std.testing.expectEqual(@as(u32, 0), seen.n.load(.acquire)); // an open, quiet link: nothing
    const t0 = Link.monotonicNs();
    try l.send("go");
    t.join(); // the follower's socket is closed
    try std.testing.expect(seen.waitReport(2000));
    try std.testing.expect(seen.at.load(.acquire) - t0 < std.time.ns_per_s);
    try std.testing.expectEqual(@as(usize, 0), seen.follower.load(.acquire));
    std.Io.sleep(io, .fromMilliseconds(150), .awake) catch {};
    try std.testing.expectEqual(@as(u32, 1), seen.n.load(.acquire)); // once
}

test "an ended watch never reports, even when the follower then goes" {
    const io = std.testing.io;
    const port: u16 = 28600 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const t = try std.Thread.spawn(.{}, Ending.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("reply", try l.recv(&buf));
    var seen: Seen = .{};
    var w = Watch.init(&.{ null, @as(?Link, l) }, &seen, Seen.lost);
    try std.testing.expectEqual(@as(usize, 1), w.n); // (an absent follower is not watched)
    try w.start();
    w.end();
    w.end(); // (idempotent)
    try l.send("go");
    t.join();
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
    try std.testing.expectEqual(@as(u32, 0), seen.n.load(.acquire));
    try std.testing.expectEqual(@as(?usize, 0), gone(&.{l}, 1000)); // (the close is there to see)
}

test "Linux: a close behind bytes not yet read is seen at once (POLLRDHUP)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const io = std.testing.io;
    const port: u16 = 29100 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.send("unread") catch {};
            l.close();
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    t.join();
    try std.testing.expectEqual(@as(?usize, 0), gone(&.{l}, 1000));
}

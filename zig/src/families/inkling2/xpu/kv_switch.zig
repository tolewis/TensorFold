//! The K/V format of a serving process: `XPU_KV_INT8` read once and logged, and the TP 2 check that both ranks hold the
//! same format. A rank whose cache is int8 while its peer's is bf16 would write and read rows of another size, so the
//! pair refuses to start.
const std = @import("std");
const linux = std.os.linux;
const st = @import("state.zig");

var once: ?st.KvDtype = null;

/// The format of every cache of this process. The first call reads `XPU_KV_INT8` and logs `inkling2: kv int8` or
/// `inkling2: kv bf16`; later calls return that value, so a change of the environment later has no effect.
pub fn get() st.KvDtype {
    if (once) |k| return k;
    const k = st.kvFromEnv();
    once = k;
    std.log.info("inkling2: kv {s}", .{@tagName(k)});
    return k;
}

fn code(k: st.KvDtype) u8 {
    return switch (k) {
        .bf16 => 'b',
        .int8 => 'i',
    };
}

/// Sends this rank's format on the pair socket and reads the peer's. Both ranks call it at the same point of the
/// handshake. A difference is `error.Inkling2KvMismatch` on both ranks, and `peer` holds the peer's format.
pub fn agree(sock: linux.fd_t, mine: st.KvDtype, peer: *[]const u8) !void {
    var out = [1]u8{code(mine)};
    if (linux.errno(linux.write(sock, &out, 1)) != .SUCCESS) return error.ByteSend;
    var in = [1]u8{0};
    const n = linux.read(sock, &in, 1);
    if (linux.errno(n) != .SUCCESS or n != 1) return error.ByteRecv;
    if (in[0] != out[0]) {
        peer.* = if (in[0] == code(.int8)) "int8" else if (in[0] == code(.bf16)) "bf16" else "unknown";
        return error.Inkling2KvMismatch;
    }
}

fn pair() ![2]linux.fd_t {
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM, 0, &fds)) != .SUCCESS) return error.SocketPair;
    return .{ fds[0], fds[1] };
}

test "TP 2: ranks with the same K/V format agree, a difference refuses on both ranks" {
    const cases = [_]struct { st.KvDtype, st.KvDtype, bool }{
        .{ .bf16, .bf16, true },
        .{ .int8, .int8, true },
        .{ .int8, .bf16, false },
        .{ .bf16, .int8, false },
    };
    for (cases) |c| {
        const fds = try pair();
        defer _ = linux.close(fds[0]);
        defer _ = linux.close(fds[1]);
        // a stream socketpair buffers the one byte each side sends, so the two calls run in turn on one thread
        var out = [1]u8{code(c[1])};
        _ = linux.write(fds[1], &out, 1);
        var p0: []const u8 = "";
        const r0 = agree(fds[0], c[0], &p0);
        var got = [1]u8{0};
        _ = linux.read(fds[1], &got, 1);
        try std.testing.expectEqual(code(c[0]), got[0]);
        out[0] = code(c[0]);
        _ = linux.write(fds[0], &out, 1);
        var p1: []const u8 = "";
        const r1 = agree(fds[1], c[1], &p1);
        _ = linux.read(fds[0], &got, 1);
        if (c[2]) {
            try r0;
            try r1;
        } else {
            try std.testing.expectError(error.Inkling2KvMismatch, r0);
            try std.testing.expectError(error.Inkling2KvMismatch, r1);
            try std.testing.expectEqualStrings(@tagName(c[1]), p0);
            try std.testing.expectEqualStrings(@tagName(c[0]), p1);
        }
    }
}

test "the switch is read once" {
    once = null;
    const first = get();
    try std.testing.expectEqual(first, get());
    once = .int8;
    try std.testing.expectEqual(st.KvDtype.int8, get());
    once = null;
}

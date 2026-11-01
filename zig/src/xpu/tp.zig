//! Tensor-parallel ranks on one host: the all-reduce of a partial sum through shared memory.
//!
//! Two processes, one a card (ZE_AFFINITY_MASK 0 and 1), run the same forward over their halves of the weights
//! (qwen27.TP, qwen27.RANK). Twice a layer each holds a partial of the residual delta (D values a row, fp16) and
//! needs the sum: it copies its partial to the host (Dev.copy and settle, 16 us a row of 10 KB on the B70), writes
//! it into its slot of a /dev/shm file, bumps its sequence number, waits for the peer's, adds the peer's partial on
//! the host and copies the sum back. 128 all-reduces a token: 4.5 ms measured at 1 row
//! (lanes/qwen27/TP2-ASSESSMENT.md in Intel-B70-Recipe, 2026-11-01). At TP 1 every call returns at once.
//!
//! The device-direct path (the default when it opens, TF_TP_HOST=1 forces the host path): each rank keeps two
//! slots in device memory and exports them (zeMemGetIpcHandle). The handles cross over a unix socket beside the
//! shm file, the peer opens them (zeMemOpenIpcHandle), and after the counter says a partial is written a device
//! copy brings the peer's slot over PCIe and a kernel adds it into this rank's buffer (ze_ipc.py: 10 KB in 10.9 us, 9.2 MB in 283 us). The
//! host settle before the counter stays.
//!
//! The file: TF_TP_SHM names it (default /dev/shm/tf-tp-qwen27). Rank 0 creates and sizes it, rank 1 opens it.
//! Layout: a header of TP sequence counters (u64, cache-line apart), then two slots a rank of `rows_max` x D fp16
//! (call `seq` uses slot seq % 2: a rank one call ahead writes the other slot, so a slow reader's data stays), then
//! the follower's consumed counter and a ring of `ring_len` step records.
//!
//! The step records: rank 0 writes one before every state-changing Runner call (follow.zig lists them) and the
//! rank above 0 replays the same call with the same arguments, so the two forwards meet at every all-reduce. A
//! prompt longer than a record continues in `more` records.
const std = @import("std");
const linux = std.os.linux;
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const ze = @import("ze");
const Dev = dev.Dev;
const D = q.D;

const line = 64;
const rows_max = 1024 + 128 + 16; // the widest forward (q.maxRows at XPU_CHUNK 1024) and a decode window
const slot_bytes = rows_max * D * 2;
const ring_len = 64;
pub const rec_words = 1024; // 4 KiB a record
pub const rec_data = rec_words - 8;

/// A state-changing Runner call. `more`: the rest of the previous record's data.
pub const Op = enum(u32) { select = 1, store, reset, prompt, more, step, verify, commit, verify_tree, commit_path, verify_multi, keep_multi, keep_paths, step_multi, save_slot, restore_slot, drop_snap, ensure, close };

/// One step record: `seq` is the record's number plus one once written (0: empty), `n` the data words of the
/// whole call, `a` `b` `lo` `hi` the call's small arguments (lo and hi a 64-bit snapshot id).
pub const Rec = extern struct { seq: u64, op: u32, a: u32, b: u32, n: u32, lo: u32, hi: u32, data: [rec_data]u32 };
comptime {
    std.debug.assert(@sizeOf(Rec) == rec_words * 4);
}

/// The group this process is in, opened by Runner.init; null at TP 1.
pub var group: ?Group = null;
var one_slot = false;
var trace = false;

pub const Group = struct {
    map: []align(std.heap.page_size_min) u8,
    rank: u32,
    tp: u32,
    seq: u64 = 0,
    /// Step records written (rank 0) or consumed (a follower).
    step: u64 = 0,
    /// Host-visible staging for this rank's partial and for the sum (Level Zero host memory, so the copies are DMA).
    mine: [*]u16,
    sum: [*]u16,
    /// The device-direct path: this rank's two device slots and the peer's two, opened here. `direct` false: host path.
    dslot: [2]*anyopaque = undefined,
    peer: [2]?*anyopaque = .{ null, null },
    /// A local copy of the peer's slot: a kernel on this card cannot take the other card's memory as an argument
    /// (zeCommandListAppendLaunchKernel: invalid argument), a device copy from it works.
    pbuf: [2]*anyopaque = undefined,
    direct: bool = false,

    fn counter(g: *const Group, r: u32) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + r * line));
    }
    /// Rank r's slot for call `seq` (1-based: the value of g.seq after the bump). TF_TP_ONE_SLOT=1 (a debug switch)
    /// uses one slot a rank as the first form did.
    fn slot(g: *const Group, r: u32, seq: u64) [*]u16 {
        const parity: u32 = if (one_slot) 0 else @intCast(seq % 2);
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + (2 * r + parity) * slot_bytes));
    }
    fn consumed(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + 2 * g.tp * slot_bytes));
    }
    /// Rank 0's K/V pool size in pages, in the unused tail of its counter line (the counters take one word a rank):
    /// a follower sizes the same pool, so both ranks map the same page ids.
    fn poolWord(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + 8));
    }
    pub fn setPool(g: *const Group, pages: u32) void {
        if (g.rank == 0) g.poolWord().store(pages, .seq_cst);
    }
    /// The pages rank 0 published; null after `seconds` (rank 0 zeroes the pool before the first request).
    pub fn awaitPool(g: *const Group, seconds: f64) ?u32 {
        const deadline = q.now() + seconds;
        while (q.now() < deadline) {
            const n = g.poolWord().load(.seq_cst);
            if (n != 0) return @intCast(n);
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 10_000_000 };
            while (std.c.nanosleep(&ts, &ts) != 0) {}
        }
        return null;
    }
    fn rec(g: *const Group, i: u64) *Rec {
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + 2 * g.tp * slot_bytes + line + (i % ring_len) * @sizeOf(Rec)));
    }

    /// Opens the group this process is in. Null at TP 1 (nothing to reduce).
    pub fn open(d: *Dev) !?Group {
        if (q.TP == 1) return null;
        one_slot = std.c.getenv("TF_TP_ONE_SLOT") != null;
        trace = std.c.getenv("TF_TP_TRACE") != null;
        const name: [*:0]const u8 = if (std.c.getenv("TF_TP_SHM")) |v| v else "/dev/shm/tf-tp-qwen27";
        const size = q.TP * line + 2 * q.TP * slot_bytes + line + ring_len * @sizeOf(Rec);
        // the file as the fabric's mailbox opens its own (fabric/tp/mailbox.zig): either rank may create it
        const fd = std.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.ShmOpenFailed;
        defer _ = std.c.close(fd);
        if (std.c.ftruncate(fd, @intCast(size)) != 0) return error.ShmSizeFailed;
        const map = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        var g: Group = .{
            .map = map,
            .rank = q.RANK,
            .tp = q.TP,
            .mine = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
            .sum = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
        };
        g.counter(g.rank).store(0, .seq_cst);
        if (g.rank == 0) { // a fresh ring: the followers start at record 0
            g.consumed().store(0, .seq_cst);
            for (0..ring_len) |i| @atomicStore(u64, &g.rec(i).seq, 0, .seq_cst);
        }
        if (std.c.getenv("TF_TP_HOST") == null) g.openDirect(d) catch |err| {
            std.log.warn("qwen27-xpu: TP all-reduce stays host-staged: {s}", .{@errorName(err)});
            g.direct = false;
        };
        std.log.info("qwen27-xpu: TP {d} rank {d}: all-reduce {s}", .{ g.tp, g.rank, if (g.direct) "device-direct (the peer's slots over PCIe)" else "host-staged" });
        return g;
    }

    /// Rank 0 writes a step record (nothing on another rank): the call `op` with its small arguments and `data`
    /// words. Waits while the ring is full (a follower 64 records behind).
    pub fn record(g: *Group, op: Op, a: u32, b: u32, id: u64, data: []const u32) !void {
        if (g.rank != 0) return;
        if (trace) std.debug.print("record {d}: {s} a {d} b {d} n {d} data[0] {d}\n", .{ g.step, @tagName(op), a, b, data.len, if (data.len > 0) data[0] else 0 });
        var at: usize = 0;
        var first = true;
        while (first or at < data.len) {
            const n = @min(data.len - at, rec_data);
            const deadline = q.now() + 180.0;
            while (g.step >= ring_len and g.consumed().load(.acquire) + ring_len <= g.step) {
                if (q.now() > deadline) return error.PeerLost;
                std.atomic.spinLoopHint();
            }
            const r = g.rec(g.step);
            r.op = @intFromEnum(if (first) op else Op.more);
            r.a = a;
            r.b = b;
            r.n = @intCast(data.len);
            r.lo = @truncate(id);
            r.hi = @truncate(id >> 32);
            @memcpy(r.data[0..n], data[at .. at + n]);
            @atomicStore(u64, &r.seq, g.step + 1, .release);
            g.step += 1;
            at += n;
            first = false;
        }
    }

    /// A follower takes the next step record (waits for it: rank 0 may idle between requests, so the spin sleeps).
    pub fn next(g: *Group, out: *Rec) void {
        const r = g.rec(g.step);
        var spins: usize = 0;
        while (@atomicLoad(u64, &r.seq, .acquire) != g.step + 1) : (spins += 1) {
            if (spins > 20_000) {
                var ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                _ = std.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
        out.* = r.*;
        g.step += 1;
        g.consumed().store(g.step, .release);
    }

    /// Waits until every peer's counter reached this rank's sequence number (180 s: the peer may still be loading).
    fn waitPeers(g: *Group) !void {
        for (0..g.tp) |r| if (r != g.rank) {
            const c = g.counter(@intCast(r));
            const deadline = q.now() + 180.0;
            var spins: usize = 0;
            while (c.load(.seq_cst) < g.seq) : (spins += 1) {
                if (spins % 4096 == 0 and q.now() > deadline) {
                    std.debug.print("rank {d}: all-reduce {d} after record {d}: the peer is at {d}\n", .{ g.rank, g.seq, g.step, c.load(.seq_cst) });
                    return error.PeerLost;
                }
                std.atomic.spinLoopHint();
            }
        };
    }

    /// `partial` on the device holds `rows` x D fp16: on return it holds the ranks' sum. Every rank calls this the
    /// same number of times in the same order.
    pub fn allReduce(g: *Group, d: *Dev, partial: *anyopaque, rows: u32) !void {
        if (rows > rows_max) return error.TooManyRows;
        const n: usize = @as(usize, rows) * D;
        g.seq += 1;
        if (g.direct) { // the partial into this rank's slot, the counter, then the add kernel reads the peer's slot
            const par: usize = @intCast(g.seq % 2);
            try dev.copy(d, g.dslot[par], partial, n * 2);
            d.settle();
            g.counter(g.rank).store(g.seq, .seq_cst);
            try g.waitPeers();
            try dev.copy(d, g.pbuf[par], g.peer[par].?, n * 2); // over PCIe, on the in-order list
            try d.launch(d.k.tp_add, .{ .x = @intCast((n + 255) / 256) }, .{ partial, g.dslot[par], g.pbuf[par], @as(i32, @intCast(n)) });
            return;
        }
        try dev.copy(d, @ptrCast(g.mine), partial, n * 2);
        d.settle();
        @memcpy(g.slot(g.rank, g.seq)[0..n], g.mine[0..n]);
        g.counter(g.rank).store(g.seq, .seq_cst);
        try g.waitPeers();
        // the sum in fp32, rounded once: the ranks add in the same order, so both hold the same bits
        for (0..n) |i| {
            var acc: f32 = 0;
            for (0..g.tp) |r| acc += @floatCast(@as(f16, @bitCast(g.slot(@intCast(r), g.seq)[i])));
            g.sum[i] = @bitCast(@as(f16, @floatCast(acc)));
        }
        try dev.copy(d, partial, @ptrCast(g.sum), n * 2);
        d.settle();
    }

    /// Opens the device-direct path: two device slots exported to the peer, the peer's two opened here. The
    /// handles cross over a unix socket beside the shm file (rank 0 listens, rank 1 connects, SCM_RIGHTS carries
    /// the dma-buf fds). Any failure leaves the host path in place.
    fn openDirect(g: *Group, d: *Dev) !void {
        if (g.tp != 2) return error.DirectNeedsTwoRanks;
        var handles: [2]ze.IpcHandle = undefined;
        var fds: [2]i32 = undefined;
        for (0..2) |i| {
            g.dslot[i] = try d.g.deviceAlloc(slot_bytes);
            g.pbuf[i] = try d.g.deviceAlloc(slot_bytes);
            try ze.check(d.g.api.zeMemGetIpcHandle(d.g.context, g.dslot[i], &handles[i]));
            fds[i] = std.mem.readInt(i32, handles[i].data[0..4], .little);
        }
        const name: [*:0]const u8 = if (std.c.getenv("TF_TP_SHM")) |v| v else "/dev/shm/tf-tp-qwen27";
        const base = std.mem.span(name);
        var addr: linux.sockaddr.un = .{ .path = @splat(0) };
        if (base.len + 6 >= addr.path.len) return error.PathTooLong;
        @memcpy(addr.path[0..base.len], base);
        @memcpy(addr.path[base.len .. base.len + 5], ".sock");
        const alen: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + base.len + 6);
        const sock_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
        if (failed(sock_rc)) return error.SocketFailed;
        const sock: linux.fd_t = @intCast(sock_rc);
        defer _ = linux.close(sock);
        var conn: linux.fd_t = sock;
        if (g.rank == 0) {
            _ = linux.unlink(@ptrCast(&addr.path));
            if (failed(linux.bind(sock, @ptrCast(&addr), alen))) return error.BindFailed;
            if (failed(linux.listen(sock, 1))) return error.ListenFailed;
            var pfd = [1]linux.pollfd{.{ .fd = sock, .events = linux.POLL.IN, .revents = 0 }};
            const ready = linux.poll(&pfd, 1, 180_000);
            if (failed(ready) or ready == 0) return error.PeerLost;
            const acc = linux.accept4(sock, null, null, linux.SOCK.CLOEXEC);
            if (failed(acc)) return error.AcceptFailed;
            conn = @intCast(acc);
        } else {
            const deadline = q.now() + 180.0;
            while (failed(linux.connect(sock, @ptrCast(&addr), alen))) {
                if (q.now() > deadline) return error.PeerLost;
                var ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000_000 };
                _ = std.c.nanosleep(&ts, null);
            }
        }
        defer if (g.rank == 0) {
            _ = linux.close(conn);
            _ = linux.unlink(@ptrCast(&addr.path));
        };
        try sendFds(conn, std.mem.asBytes(&handles), &fds);
        var theirs: [2]ze.IpcHandle = undefined;
        var their_fds: [2]i32 = undefined;
        try recvFds(conn, std.mem.asBytes(&theirs), &their_fds);
        for (0..2) |i| {
            std.mem.writeInt(i32, theirs[i].data[0..4], their_fds[i], .little);
            var p: ?*anyopaque = null;
            try ze.check(d.g.api.zeMemOpenIpcHandle(d.g.context, d.g.device, theirs[i], 0, &p));
            g.peer[i] = p;
        }
        g.direct = true;
    }
};

const cmsg_space = ((@sizeOf(linux.cmsghdr) + 2 * 4 + 7) / 8) * 8; // a control message with two fds

/// A raw syscall's return: negative is -errno.
fn failed(rc: usize) bool {
    return @as(isize, @bitCast(rc)) < 0;
}
const scm_rights = 1;

fn sendFds(sock: linux.fd_t, data: []const u8, fds: []const i32) !void {
    var iov = [1]std.posix.iovec_const{.{ .base = data.ptr, .len = data.len }};
    var cbuf: [cmsg_space]u8 align(8) = @splat(0);
    const h: *linux.cmsghdr = @ptrCast(&cbuf);
    h.* = .{ .len = @sizeOf(linux.cmsghdr) + fds.len * 4, .level = linux.SOL.SOCKET, .type = scm_rights };
    @memcpy(cbuf[@sizeOf(linux.cmsghdr) .. @sizeOf(linux.cmsghdr) + fds.len * 4], std.mem.sliceAsBytes(fds));
    const msg: linux.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &cbuf, .controllen = cbuf.len, .flags = 0 };
    const rc = linux.sendmsg(sock, &msg, 0);
    if (failed(rc) or rc != data.len) return error.FdSend;
}

fn recvFds(sock: linux.fd_t, data: []u8, fds: []i32) !void {
    var iov = [1]std.posix.iovec{.{ .base = data.ptr, .len = data.len }};
    var cbuf: [cmsg_space]u8 align(8) = @splat(0);
    var msg: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &cbuf, .controllen = cbuf.len, .flags = 0 };
    const rc = linux.recvmsg(sock, &msg, linux.MSG.WAITALL);
    if (failed(rc) or rc != data.len) return error.FdRecv;
    const h: *const linux.cmsghdr = @ptrCast(&cbuf);
    if (h.level != linux.SOL.SOCKET or h.type != scm_rights or h.len != @sizeOf(linux.cmsghdr) + fds.len * 4) return error.FdRecv;
    @memcpy(std.mem.sliceAsBytes(fds), cbuf[@sizeOf(linux.cmsghdr) .. @sizeOf(linux.cmsghdr) + fds.len * 4]);
}

test "an fp16 sum rounds once from fp32" {
    const a: f16 = 1.0009765625; // 1 + 2^-10, the fp16 step above 1
    const b: f16 = 0.00048828125; // 2^-11, below the step
    const s: f32 = @as(f32, a) + @as(f32, b);
    try std.testing.expectEqual(@as(f16, 1.0009765625), @as(f16, @floatCast(s)));
    try std.testing.expect(@as(f16, @floatCast(s + b)) > a);
}

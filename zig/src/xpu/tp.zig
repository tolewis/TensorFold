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
//! The device-flag path (TF_TP_DEVFLAG=1, default off; it needs a build with the tp_add_flag, add_rmsnorm_dec_tp and
//! tp_consume kernels): no host wait an exchange. Each rank pushes its partial into the peer's exported slot over
//! PCIe (a copy into the IPC-opened block), then writes the exchange's sequence number into the peer's flag word on
//! the same in-order list. The peer's add kernel polls that flag in device memory, adds, and a small kernel publishes
//! the consumed counter and waits until the peer consumed the slot the next exchange writes again (two slots, so a
//! rank cannot overrun its peer by more than one exchange). A rank that times out sets a timeout word and the host
//! reports PeerLost at the next settle. The mixer's exchange is fused into the residual add and norm that reads the
//! reduced tensor once (add_rmsnorm_dec_tp). At open the path proves that the peer's memory takes this rank's writes
//! and both ranks agree on the mode over the socket; any failure leaves the host path in place.
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
const off = dev.off;
const D = q.D;

const line = 64;
const rows_max = 1024 + 128 + 16; // the widest forward (q.maxRows at XPU_CHUNK 1024) and a decode window
const slot_bytes = rows_max * D * 2;
const ring_len = 64;
/// The segment header: the generation (0), the magic (8), then the rank lines. A segment is only taken by a follower
/// that reads a non-zero generation with the right magic, so a file a dead pair left is never used.
const hdr = 64;
const magic: u64 = 0x5127_5450; // Q27TP
const peer_dead = 10.0; // seconds without an all-reduce: a silent rank is gone, the pair cannot run
/// Seconds rank 0 waits for the follower to consume a step record before it calls the pair lost. TF_TP_RECORD_WAIT
/// raises it (a replayed step that is legitimately slow); 180 s is the load and ring deadline the ring already had.
var record_wait: f64 = 180.0;
// the exported block a rank: the peer's pushed partials (2 slots), the flag words the peer writes (2), the peer's
// consumed counters (2), a timeout word
const blk_slots = 2 * slot_bytes;
const blk_flag = blk_slots;
const blk_pcons = blk_flag + 16;
const blk_timeout = blk_pcons + 16;
const blk_devflag_bytes = blk_timeout + 64;
const blk_legacy_bytes = 4 * slot_bytes; // the device-direct path: this rank's two slots, then its two peer copies
const fh_ring = 512; // host staging slots for the flag value (a round queues at most 128 exchanges before a settle)
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

/// The residual add and norm the device-flag exchange can be fused into (add_rmsnorm_dec_tp).
pub const Norm = struct {
    x: *anyopaque, // fp32 residual [rows, D]
    w: *anyopaque, // the norm weight (bf16)
    h: *anyopaque, // fp16 normed output [rows, D]
    ahi: *anyopaque,
    alo: *anyopaque,
    a8: ?*anyopaque = null, // the W4A8 quantized copy, when the site asks for it
    sa: ?*anyopaque = null,
    ms: i32 = 0,
    eps: f32,
    has_r: i32,
    gs: i32,
    rot: i32,
};

/// The magic a rank writes into the peer's flag word at open (plus its rank), to prove the host-to-peer write
/// lands. `tp_magic2` does the same for the device-to-peer write the partial push uses.
const tp_magic: u64 = 0x5450_464c_4147_3100;
const tp_magic2: u64 = 0x5055_5348_5000_0000;

/// The group this process is in, opened by Runner.init; null at TP 1.
pub var group: ?Group = null;
var one_slot = false;
var trace = false;
var devflag = false; // TF_TP_DEVFLAG: the device-flag all-reduce (default off)

/// A peer that dies in the handshake closes its end: without a timeout the surviving rank blocks in recvFds forever.
/// 60 s is far longer than the fd exchange takes.
fn setTimeouts(sock: linux.fd_t) void {
    var tv: linux.timeval = .{ .sec = 60, .usec = 0 };
    _ = linux.setsockopt(sock, linux.SOL.SOCKET, linux.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(linux.timeval));
    _ = linux.setsockopt(sock, linux.SOL.SOCKET, linux.SO.SNDTIMEO, @ptrCast(&tv), @sizeOf(linux.timeval));
}

/// Rank 1's connect to rank 0's socket, until the deadline (rank 0 may still be loading).
fn connectPeer(base: []const u8, seconds: f64) !linux.fd_t {
    const addr = try Group.sockAddr(base);
    const alen: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + base.len + 6);
    const sock_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    if (failed(sock_rc)) return error.SocketFailed;
    const sock: linux.fd_t = @intCast(sock_rc);
    setTimeouts(sock);
    const deadline = q.now() + seconds;
    while (failed(linux.connect(sock, @ptrCast(&addr), alen))) {
        if (q.now() > deadline) {
            _ = linux.close(sock);
            return error.PeerLost;
        }
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }
    return sock;
}

/// The generation rank 0 published, sent before the fds: a follower takes only the segment that carries it.
fn sendGen(sock: linux.fd_t, gen: u64) !void {
    var v: [8]u8 = undefined;
    std.mem.writeInt(u64, &v, gen, .little);
    var got: usize = 0;
    while (got < 8) {
        const rc = linux.write(sock, v[got..].ptr, 8 - got);
        if (failed(rc) or rc == 0) return error.ByteSend;
        got += rc;
    }
}

fn recvGen(sock: linux.fd_t) !u64 {
    var v: [8]u8 = undefined;
    var got: usize = 0;
    while (got < 8) {
        const rc = linux.read(sock, v[got..].ptr, 8 - got);
        if (failed(rc) or rc == 0) return error.ByteRecv;
        got += rc;
    }
    return std.mem.readInt(u64, &v, .little);
}

/// The generation word of an open segment: 0 when the file is shorter than the header or has none yet.
fn genOf(fd: linux.fd_t) u64 {
    var v: u64 = 0;
    const rc = std.c.pread(fd, @ptrCast(&v), 8, 0);
    return if (rc == 8) v else 0;
}

/// A fresh generation value for rank 0's segment. 0 means "not published", so a generation is never 0.
fn genNow() u64 {
    return @as(u64, @intFromFloat(q.now() * 1e9)) | 1;
}

/// A follower waits for rank 0's fresh segment: the path must still name the inode it opened, the magic must match, and
/// the generation must be published (rank 0 writes it after it resets every shared word). The deadline exits 1, so
/// systemd restarts the pair instead of this rank waiting forever for a segment that no rank will publish.
fn awaitGen(name: [*:0]const u8, size: usize, want: u64, fd_out: *linux.fd_t, gen_out: *u64) ![]align(std.heap.page_size_min) u8 {
    const deadline = q.now() + 180.0;
    while (true) {
        const fd = std.c.open(name, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd >= 0) {
            const gen = genOf(fd);
            // only the generation rank 0 named over the socket: a segment a dead pair left has another one
            if (gen != 0 and gen == want) {
                if (std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0)) |map| {
                    const mg = @as(*align(8) std.atomic.Value(u64), @ptrCast(@alignCast(map.ptr + 8))).load(.acquire);
                    // the generation and the magic together: rank 0 publishes the generation after it resets every
                    // shared word, so a segment read here is clean, and a file another build made fails the magic
                    if (genOf(fd) == gen and mg == magic) {
                        fd_out.* = fd; // kept: the follower re-checks that the path still names this generation
                        gen_out.* = gen;
                        return map;
                    }
                    std.posix.munmap(map);
                } else |_| {}
            }
            _ = std.c.close(fd);
        }
        if (q.now() > deadline) {
            std.log.err("rank {d}: no fresh TP segment at {s} after 180 s: the group is lost\n", .{ q.RANK, name });
            std.process.exit(1);
        }
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000_000 };
        while (std.c.nanosleep(&ts, &ts) != 0) {}
    }
}

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
    /// The device-flag path (TF_TP_DEVFLAG): one exported block a rank (`blk`: the peer's pushed partials, the flag
    /// words the peer writes, the peer's consumed counters, a timeout word) and the peer's block opened here.
    /// `own_cons` holds this rank's consumed counters, which it pushes into the peer's block.
    blk: ?*anyopaque = null,
    peer_blk: ?*anyopaque = null,
    own_cons: ?*anyopaque = null,
    /// Host staging ring for a flag value (seq % fh_ring): the copy that carries it is queued on the in-order list.
    fh: [*]u64 = undefined,
    devflag: bool = false,
    /// The segment's path, the generation this rank mapped, and on a follower the descriptor it opened: a follower
    /// re-checks that the path still names this generation, because a rank 0 that restarted replaced it.
    name: [*:0]const u8 = undefined,
    gen: u64 = 0,
    fd: linux.fd_t = -1,

    /// The generation rank 0 published after it reset every shared word. 0: no rank has published one yet.
    fn genWord(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr));
    }
    fn magicWord(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + 8));
    }
    fn counter(g: *const Group, r: u32) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + hdr + r * line));
    }
    /// Rank r's slot for call `seq` (1-based: the value of g.seq after the bump). TF_TP_ONE_SLOT=1 (a debug switch)
    /// uses one slot a rank as the first form did.
    fn slot(g: *const Group, r: u32, seq: u64) [*]u16 {
        const parity: u32 = if (one_slot) 0 else @intCast(seq % 2);
        return @ptrCast(@alignCast(g.map.ptr + hdr + g.tp * line + (2 * r + parity) * slot_bytes));
    }
    fn consumed(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + hdr + g.tp * line + 2 * g.tp * slot_bytes));
    }
    /// Rank r's liveness word in the shared segment (right after rank r's counter, in its cache line): this rank
    /// writes the current time into it at every all-reduce, so a peer that dies silently is seen by its peer before
    /// the 180 s wait even though no signal reaches it. The pair can no longer run when its last all-reduce is
    /// older than `peer_dead` (one round takes about 40 ms).
    fn aliveWord(g: *const Group, r: u32) *std.atomic.Value(u64) {
        const at = hdr + @as(usize, r) * line + 8;
        return @ptrCast(@alignCast(g.map.ptr + at));
    }

    /// The group's close flag in the shared segment (the header padding after the consumed counter, before the ring):
    /// a rank sets it when its session ends or errors, so a peer that waits for it returns an error at once instead of
    /// the 180 s timeout. A fresh open clears it: each group generation starts live.
    fn closed(g: *const Group) *std.atomic.Value(u64) {
        const at = hdr + g.tp * line + 2 * g.tp * slot_bytes + line - @sizeOf(u64);
        return @ptrCast(@alignCast(g.map.ptr + at));
    }

    /// A rank closes the shared segment (the group's flag and its peers' view of it) when its session ends or errors.
    pub fn markClosed(g: *Group) void {
        closed(g).store(1, .release);
    }

    /// Rank 0's K/V pool size in pages, in the unused tail of its counter line (the counters take one word a rank):
    /// a follower sizes the same pool, so both ranks map the same page ids.
    /// Lane p2-int: the liveness word of lane p28-ops took offset 8 of rank 0's line, so this word moved to offset 16
    /// (rank 0's line: counter 0, liveness 8, pool 16). Rank 0's liveness is written at +8, so the two never collide.
    fn poolWord(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + hdr + 16));
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
        return @ptrCast(@alignCast(g.map.ptr + hdr + g.tp * line + 2 * g.tp * slot_bytes + line + (i % ring_len) * @sizeOf(Rec)));
    }

    /// Opens the group this process is in. Null at TP 1 (nothing to reduce).
    pub fn open(d: *Dev) !?Group {
        if (q.TP == 1) return null;
        one_slot = std.c.getenv("TF_TP_ONE_SLOT") != null;
        trace = std.c.getenv("TF_TP_TRACE") != null;
        devflag = std.c.getenv("TF_TP_DEVFLAG") != null;
        record_wait = 180.0;
        if (std.c.getenv("TF_TP_RECORD_WAIT")) |v| record_wait = std.fmt.parseFloat(f64, std.mem.span(v)) catch 180.0;
        const name: [*:0]const u8 = if (std.c.getenv("TF_TP_SHM")) |v| v else "/dev/shm/tf-tp-qwen27";
        const base = std.mem.span(name);
        const size = hdr + q.TP * line + 2 * q.TP * slot_bytes + line + ring_len * @sizeOf(Rec);
        const sa = try sockAddr(base);
        var map: []align(std.heap.page_size_min) u8 = undefined;
        var fd: linux.fd_t = -1;
        var gen: u64 = 0;
        var peer_conn: ?linux.fd_t = null;
        if (q.RANK == 0) {
            // A fresh generation: the old segment and its socket go first, so no rank reads a dead pair's words and no
            // follower connects to its socket. A plain systemd restart then recovers in any start order.
            _ = linux.unlink(name);
            _ = linux.unlink(@ptrCast(&sa.path));
            const f = std.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            if (f < 0) return error.ShmOpenFailed;
            defer _ = std.c.close(f);
            if (std.c.ftruncate(f, @intCast(size)) != 0) return error.ShmSizeFailed;
            map = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, f, 0);
        } else {
            // connect first: rank 0 names the generation of the segment it made, and this rank takes only that one
            const conn = try connectPeer(base, 180.0);
            gen = try recvGen(conn);
            map = try awaitGen(name, size, gen, &fd, &gen);
            peer_conn = conn;
        }
        var g: Group = .{
            .map = map,
            .rank = q.RANK,
            .tp = q.TP,
            .name = name,
            .fd = fd,
            .gen = gen,
            .mine = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
            .sum = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
        };
        if (g.rank == 0) {
            // Every shared word is reset before the generation is published: a follower that reads the generation reads
            // a segment that is already clean. A follower resets nothing, so it cannot clear a word rank 0 is using.
            g.magicWord().store(magic, .seq_cst);
            g.closed().store(0, .seq_cst);
            g.poolWord().store(0, .seq_cst);
            // no liveness stamp yet: zero means "this rank has not reduced yet", so its peers wait for it up to the
            // 180 s load deadline instead of calling it dead. The weights load for minutes between `open` and the
            // first all-reduce (lane p2-int: a stamp at open would read as a silent rank there).
            for (0..g.tp) |r| {
                g.counter(@intCast(r)).store(0, .seq_cst);
                g.aliveWord(@intCast(r)).store(0, .release);
            }
            g.consumed().store(0, .seq_cst); // the followers start at record 0
            for (0..ring_len) |i| @atomicStore(u64, &g.rec(i).seq, 0, .seq_cst);
            g.gen = genNow();
            g.genWord().store(g.gen, .release); // last: the segment is clean when a follower sees it
        }
        if (std.c.getenv("TF_TP_HOST") == null) g.openDirect(d, peer_conn) catch |err| switch (err) {
            // the peer never came, or it went away in the handshake: exit so systemd restarts the pair
            error.PeerLost, error.AcceptFailed, error.NoPeerSocket, error.FdSend, error.FdRecv, error.ByteSend, error.ByteRecv => lostAt(&g, "rank {d}: the open did not meet its peer ({s}): the group is lost\n", .{ g.rank, @errorName(err) }),
            else => {
                std.log.warn("qwen27-xpu: TP all-reduce stays host-staged: {s}", .{@errorName(err)});
                g.direct = false;
            },
        };
        std.log.info("qwen27-xpu: TP {d} rank {d}: all-reduce {s}", .{ g.tp, g.rank, if (g.devflag) "device-flag (the peer's slot pushed over PCIe, the add polls a flag)" else if (g.direct) "device-direct (the peer's slots over PCIe)" else "host-staged" });
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
            const deadline = q.now() + record_wait;
            while (g.step >= ring_len and g.consumed().load(.acquire) + ring_len <= g.step) {
                // The follower is a whole ring behind: one of its steps has not returned. Rank 0 cannot go on and the
                // pair cannot run, so the group closes and rank 0 exits (systemd restarts both). A plain error here
                // left the pair up but wedged: no rank closed the segment and every later request failed.
                if (q.now() > deadline) lostAt(g, "rank 0: record {d} ({s}) waited {d:.0} s for the follower, which is at {d}: the group is lost\n", .{ g.step, @tagName(op), record_wait, g.consumed().load(.acquire) });
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
        const closed_word = closed(g);
        const r = g.rec(g.step);
        var spins: usize = 0;
        var checked = q.now();
        while (@atomicLoad(u64, &r.seq, .acquire) != g.step + 1) : (spins += 1) {
            if (spins > 20_000) {
                // the peer closed the segment (it exited): no record is coming, so this rank exits and systemd
                // restarts the pair instead of this rank spinning until an operator stops it
                if (spins % 100 == 0 and closed_word.load(.acquire) != 0) lostAt(g, "rank {d}: the group closed while it waited for record {d}\n", .{ g.rank, g.step });
                // a rank 0 that restarted replaced the segment with a new generation: this map is the dead one, and
                // the record this rank waits for will never come here. Exit and let systemd restart the pair.
                if (spins % 5000 == 0 and q.now() - checked > 1.0) {
                    checked = q.now();
                    const fd = std.c.open(g.name, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
                    var same = false;
                    if (fd >= 0) {
                        same = genOf(fd) == g.gen;
                        _ = std.c.close(fd);
                    }
                    if (!same) lostAt(g, "rank {d}: the segment was replaced (a new generation): the group is lost\n", .{g.rank});
                }
                var ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                _ = std.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
        out.* = r.*;
        g.step += 1;
        g.consumed().store(g.step, .release);
    }

    /// Waits until every peer's counter reached this rank's sequence number (180 s: the peer may still be loading).
    /// It returns an error at once when the shared segment says its group closed or the peer is silent: a rank that
    /// dies does not update its liveness word, so its peer sees it before the 180 s wait.
    fn waitPeers(g: *Group) !void {
        const closed_word = closed(g);
        for (0..g.tp) |r| if (r != g.rank) {
            const c = g.counter(@intCast(r));
            const alive = aliveWord(g, @intCast(r));
            const wait_start = q.now();
            const deadline = wait_start + 180.0; // the peer may still be loading
            var spins: usize = 0;
            while (c.load(.seq_cst) < g.seq) : (spins += 1) {
                if (spins % 4096 == 0) {
                    const now = q.now();
                    // a closed segment: the peer is gone, not just late
                    if (closed_word.load(.acquire) != 0 and c.load(.seq_cst) < g.seq) return error.PeerLost;
                    // a silent peer: its last all-reduce is older than peer_dead. Both conditions are needed. A peer
                    // that was idle before this round stamps its liveness within a few ms of the round starting, and
                    // a peer inside a long prompt chunk is alive, so only a wait that has lasted peer_dead with a
                    // stale beat calls it dead. Beat 0: the peer has not reduced yet (its weights are loading).
                    const beat = alive.load(.acquire);
                    if (beat != 0 and now - wait_start > peer_dead and now - @as(f64, @floatFromInt(beat)) > peer_dead) return error.PeerLost;
                    if (now > deadline) {
                        std.debug.print("rank {d}: all-reduce {d} after record {d}: the peer is at {d}\n", .{ g.rank, g.seq, g.step, c.load(.seq_cst) });
                        return error.PeerLost;
                    }
                }
                std.atomic.spinLoopHint();
            }
        };
    }

    /// A rank that lost its group closes the shared segment (its peer sees it at once) and exits non-zero so systemd
    /// restarts the pair instead of serving broken rounds until an operator restarts them.
    fn lostAt(g: *Group, comptime fmt: []const u8, args: anytype) noreturn {
        markClosed(g);
        std.log.err(fmt, args);
        std.process.exit(1);
    }

    fn waitGroupLost(g: *Group, e: anyerror) noreturn {
        lostAt(g, "rank {d}: all-reduce {d}: the group is lost ({s})\n", .{ g.rank, g.seq, @errorName(e) });
    }

    /// `partial` on the device holds `rows` x D fp16: on return it holds the ranks' sum. Every rank calls this the
    /// same number of times in the same order.
    pub fn allReduce(g: *Group, d: *Dev, partial: *anyopaque, rows: u32) !void {
        if (rows > rows_max) return error.TooManyRows;
        const n: usize = @as(usize, rows) * D;
        g.seq += 1;
        if (g.devflag) {
            const par: usize = @intCast(g.seq % 2);
            try g.push(d, partial, n, par);
            try d.launch(d.k.tp_add_flag, .{ .x = @intCast((n + 255) / 256) }, .{
                partial,                                   // out: this rank's partial, the sum lands here
                off(g.blk.?, par * slot_bytes),            // the peer's push (read after the flag)
                @as(i32, @intCast(n)),                     //
                off(g.blk.?, blk_flag + par * 8),          // the flag the peer wrote
                g.seq,                                     //
                off(g.blk.?, blk_timeout),                 //
            });
            try g.consume(d, par);
            return;
        }
        if (g.direct) { // the partial into this rank's slot, the counter, then the add kernel reads the peer's slot
            const par: usize = @intCast(g.seq % 2);
            try dev.copy(d, g.dslot[par], partial, n * 2);
            d.settle();
            g.counter(g.rank).store(g.seq, .seq_cst); // the counter and the liveness word are the rank's heartbeat
            aliveWord(g, g.rank).store(@intFromFloat(q.now()), .release);
            if (g.waitPeers()) |_| {} else |e| {
                // the group is lost: this rank closes it and exits instead of waiting 180 s or serving broken rounds
                waitGroupLost(g, e);
            }
            try dev.copy(d, g.pbuf[par], g.peer[par].?, n * 2); // over PCIe, on the in-order list
            try d.launch(d.k.tp_add, .{ .x = @intCast((n + 255) / 256) }, .{ partial, g.dslot[par], g.pbuf[par], @as(i32, @intCast(n)) });
            return;
        }
        try dev.copy(d, @ptrCast(g.mine), partial, n * 2);
        d.settle();
        @memcpy(g.slot(g.rank, g.seq)[0..n], g.mine[0..n]);
        g.counter(g.rank).store(g.seq, .seq_cst); // the counter and the liveness word are the rank's heartbeat
        aliveWord(g, g.rank).store(@intFromFloat(q.now()), .release);
        if (g.waitPeers()) |_| {} else |e| {
            // the group is lost: this rank closes it and exits instead of waiting 180 s or serving broken rounds
            waitGroupLost(g, e);
        }
        // the sum in fp32, rounded once: the ranks add in the same order, so both hold the same bits
        for (0..n) |i| {
            var acc: f32 = 0;
            for (0..g.tp) |r| acc += @floatCast(@as(f16, @bitCast(g.slot(@intCast(r), g.seq)[i])));
            g.sum[i] = @bitCast(@as(f16, @floatCast(acc)));
        }
        try dev.copy(d, partial, @ptrCast(g.sum), n * 2);
        d.settle();
    }

    /// The mixer's exchange fused into the residual add and norm that reads the reduced tensor once
    /// (add_rmsnorm_dec_tp, TF_TP_DEVFLAG): the same bits as the separate add and norm. `partial` keeps the sum.
    pub fn allReduceNorm(g: *Group, d: *Dev, partial: *anyopaque, rows: u32, nrm: Norm) !void {
        if (!g.devflag) return error.NotDeviceFlag;
        if (rows > rows_max) return error.TooManyRows;
        const n: usize = @as(usize, rows) * D;
        g.seq += 1;
        const par: usize = @intCast(g.seq % 2);
        try g.push(d, partial, n, par);
        try d.launch(d.k.tp_norm_tp, .{ .x = rows }, .{
            nrm.x,                                    // X fp32 residual
            partial,                                  // R fp16, keeps the sum
            off(g.blk.?, par * slot_bytes),           // the peer's push
            off(g.blk.?, blk_flag + par * 8),         //
            g.seq,                                    //
            off(g.blk.?, blk_timeout),                //
            nrm.w,                                    // WN
            nrm.h,                                    //
            nrm.ahi,                                  //
            nrm.alo,                                  //
            @as(i32, D),                              //
            nrm.eps,                                  //
            nrm.has_r,                                //
            nrm.gs,                                   //
            nrm.a8,                                   //
            nrm.sa,                                   //
            nrm.ms,                                   //
            nrm.rot,                                  //
        });
        try g.consume(d, par);
    }

    /// Pushes this rank's partial into the peer's slot and then the flag, both on the in-order list: the peer's
    /// kernel sees the partial complete before the flag.
    fn push(g: *Group, d: *Dev, partial: *anyopaque, n: usize, par: usize) !void {
        const pb = g.peer_blk.?;
        try dev.copy(d, off(pb, par * slot_bytes), partial, n * 2);
        const hi: usize = @intCast(g.seq % fh_ring);
        g.fh[hi] = g.seq;
        try dev.copy(d, off(pb, blk_flag + par * 8), @ptrCast(&g.fh[hi]), 8);
    }

    /// Publishes this rank's consumed counter for the slot and waits until the peer consumed the slot the next
    /// exchange writes again (two slots). Runs after the add kernel, so the slot's data is read out first.
    fn consume(g: *Group, d: *Dev, par: usize) !void {
        const need: u64 = if (g.seq >= 2) g.seq - 1 else 0;
        try d.launch(d.k.tp_consume, .{ .x = 1 }, .{
            off(g.own_cons.?, par * 8),                 // this rank's consumed counter, pushed to the peer next
            g.seq,                                      //
            off(g.blk.?, blk_pcons + (1 - par) * 8),    // the peer's consumption of the slot the next exchange reuses
            need,                                       //
            off(g.blk.?, blk_timeout),                  //
        });
        try dev.copy(d, off(g.peer_blk.?, blk_pcons + par * 8), off(g.own_cons.?, par * 8), 8);
    }

    /// TF_TP_EXBENCH=1: at load, the cost of 128 exchanges (a round) at 1, 8, 16 and 32 rows in the group's
    /// mode. Both ranks run it, so the pair times as it does in a forward. The host waits a mode has are inside
    /// the figure, which is what the gate asks for.
    pub fn exchangeBench(g: *Group, d: *Dev, partial: *anyopaque) !void {
        const counts = [_]u32{ 1, 8, 16, 32 };
        var out: [4]f64 = @splat(0);
        for (counts, 0..) |rows, i| {
            try g.allReduce(d, partial, rows); // warm
            d.settle();
            const t0 = q.now();
            for (0..128) |_| try g.allReduce(d, partial, rows);
            d.settle();
            out[i] = (q.now() - t0) * 1e3;
        }
        std.log.info("qwen27-xpu: exchange ms a round (128 exchanges) by rows: 1 {d:.2}, 8 {d:.2}, 16 {d:.2}, 32 {d:.2}", .{ out[0], out[1], out[2], out[3] });
    }

    /// After a settle: a device-flag exchange that timed out (the peer went away) is PeerLost. Lane p2-int: the
    /// group-close rule of lane p28-ops has no flag, so this mode closes the segment and exits as the other modes do
    /// (the kernel spin cannot see the peer's host liveness word, so its timeout word is the only detector here).
    pub fn check(g: *Group, d: *Dev) !void {
        if (!g.devflag) return;
        var t: u32 = 0;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &t, off(g.blk.?, blk_timeout), 4, null, 0, null));
        try d.sync();
        if (t != 0) {
            std.debug.print("rank {d}: the device-flag all-reduce timed out after record {d}\n", .{ g.rank, g.step });
            waitGroupLost(g, error.PeerLost);
        }
    }

    /// The unix socket beside the segment.
    fn sockAddr(base: []const u8) !linux.sockaddr.un {
        var addr: linux.sockaddr.un = .{ .path = @splat(0) };
        if (base.len + 6 >= addr.path.len) return error.PathTooLong;
        @memcpy(addr.path[0..base.len], base);
        @memcpy(addr.path[base.len .. base.len + 5], ".sock");
        return addr;
    }

    /// Opens the device-direct path: two device slots exported to the peer, the peer's two opened here. The
    /// handles cross over a unix socket beside the shm file (rank 0 listens, rank 1 connects, SCM_RIGHTS carries
    /// the dma-buf fds). A peer that never comes, or that goes away in the handshake, exits this rank: a wedged open
    /// is worse than a restart.
    fn openDirect(g: *Group, d: *Dev, peer_conn: ?linux.fd_t) !void {
        if (g.tp != 2) return error.DirectNeedsTwoRanks;
        const want_flag = devflag;
        const blk = try d.g.deviceAlloc(if (want_flag) blk_devflag_bytes else blk_legacy_bytes);
        var handle: ze.IpcHandle = undefined;
        try ze.check(d.g.api.zeMemGetIpcHandle(d.g.context, blk, &handle));
        var fds = [1]i32{std.mem.readInt(i32, handle.data[0..4], .little)};
        const name: [*:0]const u8 = if (std.c.getenv("TF_TP_SHM")) |v| v else "/dev/shm/tf-tp-qwen27";
        const base = std.mem.span(name);
        const addr = try sockAddr(base);
        const alen: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + base.len + 6);
        var conn: linux.fd_t = undefined;
        if (g.rank == 0) {
            const sock_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
            if (failed(sock_rc)) return error.SocketFailed;
            const sock: linux.fd_t = @intCast(sock_rc);
            defer _ = linux.close(sock);
            setTimeouts(sock);
            _ = linux.unlink(@ptrCast(&addr.path));
            if (failed(linux.bind(sock, @ptrCast(&addr), alen))) return error.BindFailed;
            if (failed(linux.listen(sock, 1))) return error.ListenFailed;
            var pfd = [1]linux.pollfd{.{ .fd = sock, .events = linux.POLL.IN, .revents = 0 }};
            const ready = linux.poll(&pfd, 1, 180_000);
            if (failed(ready) or ready == 0) return error.PeerLost;
            const acc = linux.accept4(sock, null, null, linux.SOCK.CLOEXEC);
            if (failed(acc)) return error.AcceptFailed;
            conn = @intCast(acc);
            setTimeouts(conn);
            defer _ = linux.close(conn);
            defer _ = linux.unlink(@ptrCast(&addr.path));
            // the follower takes only the segment this process made: its generation crosses before the fds
            try sendGen(conn, g.gen);
        } else {
            conn = peer_conn orelse return error.NoPeerSocket;
            defer _ = linux.close(conn);
        }
        try sendFds(conn, std.mem.asBytes(&handle), &fds);
        var theirs: ze.IpcHandle = undefined;
        var their_fds = [1]i32{0};
        try recvFds(conn, std.mem.asBytes(&theirs), &their_fds);
        std.mem.writeInt(i32, theirs.data[0..4], their_fds[0], .little);
        var p: ?*anyopaque = null;
        try ze.check(d.g.api.zeMemOpenIpcHandle(d.g.context, d.g.device, theirs, 0, &p));
        g.blk = blk;
        g.peer_blk = p;
        // The mode byte keeps both ranks in the same path: 0 the device-direct path, 1 the device-flag path,
        // 2 the device-flag path failed its own test (then both take the host path).
        var mode: u8 = 0;
        if (want_flag) {
            g.fh = @ptrCast(@alignCast(try d.g.hostAlloc(fh_ring * 8)));
            g.own_cons = try d.g.deviceAlloc(64);
            const list = try d.g.immediate(0, 0);
            mode = if (try g.selfTest(d, list, conn)) 1 else 2;
        }
        var peer_mode: u8 = 0;
        try sendByte(conn, mode);
        try recvByte(conn, &peer_mode);
        if (mode == 1 and peer_mode == 1) {
            g.devflag = true;
            g.direct = true;
            return;
        }
        if (mode != 0 or peer_mode != 0) {
            // one rank asked for the device flag and the paths did not agree (a test failed, or the env differs): the
            // blocks were allocated at different sizes, so both ranks take the host path
            std.log.warn("qwen27-xpu: TP all-reduce stays host-staged: the device-flag paths did not agree (this rank {d}, the peer {d})", .{ mode, peer_mode });
            g.direct = false;
            return;
        }
        for (0..2) |i| {
            g.dslot[i] = off(blk, i * slot_bytes);
            g.pbuf[i] = off(blk, (2 + i) * slot_bytes);
            g.peer[i] = off(p.?, i * slot_bytes);
        }
        g.direct = true;
    }

    /// Proves both writes the device-flag path uses before it is trusted: a host-to-peer write (the flag) and a
    /// device-to-peer write (the partial push). Each rank writes its magics into the peer's block, waits for the
    /// peer's writes to land (a byte over the socket), reads its own flag and peer-consumed words, checks them and
    /// resets them.
    fn selfTest(g: *Group, d: *Dev, list: ze.H, conn: linux.fd_t) !bool {
        const st = @as([*]u64, @ptrCast(@alignCast(try d.g.hostAlloc(64))));
        @memset(st[0..8], 0);
        st[0] = tp_magic + g.rank; // the flag: a host-to-peer write
        st[1] = tp_magic2 + g.rank; // the push: a device-to-peer write
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, off(g.peer_blk.?, blk_flag), @ptrCast(st), 8, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, g.own_cons.?, @ptrCast(st + 1), 8, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, off(g.peer_blk.?, blk_pcons), g.own_cons.?, 8, null, 0, null));
        try ze.check(d.g.api.zeCommandListHostSynchronize(list, std.math.maxInt(u64)));
        const b: u8 = 1;
        var r: u8 = 0;
        try sendByte(conn, b);
        try recvByte(conn, &r);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, @ptrCast(st + 1), off(g.blk.?, blk_flag), 24, null, 0, null));
        try ze.check(d.g.api.zeCommandListHostSynchronize(list, std.math.maxInt(u64)));
        const got_flag = st[1];
        const got_push = st[3]; // flag[0], flag[1], then pcons[0]
        const ok = got_flag == tp_magic + (1 - g.rank) and got_push == tp_magic2 + (1 - g.rank);
        // reset this rank's flag and peer-consumed words and its own consumed counters (both ranks wrote before the
        // byte exchange, so neither can land here after the reset)
        @memset(st[1..5], 0);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, off(g.blk.?, blk_flag), @ptrCast(st + 1), 32, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, g.own_cons.?, @ptrCast(st + 1), 16, null, 0, null));
        try ze.check(d.g.api.zeCommandListHostSynchronize(list, std.math.maxInt(u64)));
        if (!ok) std.log.warn("qwen27-xpu: TP rank {d}: the peer's block did not take the writes (flag 0x{x}, push 0x{x})", .{ g.rank, got_flag, got_push });
        return ok;
    }
};

const cmsg_space = ((@sizeOf(linux.cmsghdr) + 2 * 4 + 7) / 8) * 8; // a control message with two fds

/// A raw syscall's return: negative is -errno.
fn failed(rc: usize) bool {
    return @as(isize, @bitCast(rc)) < 0;
}
const scm_rights = 1;

fn sendByte(sock: linux.fd_t, b: u8) !void {
    var v = [1]u8{b};
    const rc = linux.write(sock, &v, 1);
    if (failed(rc) or rc != 1) return error.ByteSend;
}

fn recvByte(sock: linux.fd_t, b: *u8) !void {
    var v = [1]u8{0};
    var got: usize = 0;
    while (got < 1) {
        const rc = linux.read(sock, v[got..].ptr, 1);
        if (failed(rc)) return error.ByteRecv;
        if (rc == 0) return error.ByteRecv;
        got += rc;
    }
    b.* = v[0];
}

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

test "the shared segment's header words do not collide" {
    // lane p2-int: lanes p21-pool and p28-ops both took offset 8 of rank 0's cache line for a header word. The pool
    // word moved to 16, so rank 0's line is counter 0, liveness 8, pool 16, and every header word is its own u64.
    // lane p2-int-wedge: the header (generation 0, magic 8) sits before the lines, and every offset moves by `hdr`.
    const size = hdr + 2 * line + 2 * 2 * slot_bytes + line + @sizeOf(Rec); // the ring's first record too
    const map = try std.testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(std.heap.page_size_min), size);
    defer std.testing.allocator.free(map);
    @memset(map, 0);
    const g: Group = .{ .map = map, .rank = 0, .tp = 2, .mine = undefined, .sum = undefined };
    const base = @intFromPtr(map.ptr);
    const words = [_]usize{
        @intFromPtr(g.genWord()) - base,
        @intFromPtr(g.magicWord()) - base,
        @intFromPtr(g.counter(0)) - base,
        @intFromPtr(g.counter(1)) - base,
        @intFromPtr(g.aliveWord(0)) - base,
        @intFromPtr(g.aliveWord(1)) - base,
        @intFromPtr(g.poolWord()) - base,
        @intFromPtr(g.closed()) - base,
        @intFromPtr(g.rec(0)) - base,
    };
    for (words, 0..) |a, i| {
        try std.testing.expect(a % 8 == 0);
        try std.testing.expect(a + 8 <= size);
        for (words[i + 1 ..]) |b| try std.testing.expect(a != b);
    }
    // the header, then rank 0's line: the counter, the liveness word, the pool word, and the ring after both lines
    try std.testing.expectEqual(@as(usize, 0), words[0]);
    try std.testing.expectEqual(@as(usize, 8), words[1]);
    try std.testing.expectEqual(@as(usize, hdr), words[2]);
    try std.testing.expectEqual(@as(usize, hdr + 8), words[4]);
    try std.testing.expectEqual(@as(usize, hdr + 16), words[6]);
    try std.testing.expectEqual(@as(usize, hdr + line + 8), words[5]);
    try std.testing.expectEqual(@as(usize, hdr + 2 * line + 2 * 2 * slot_bytes + line - 8), words[7]);
    try std.testing.expectEqual(@as(usize, hdr + 2 * line + 2 * 2 * slot_bytes + line), words[8]);
}

test "a fresh generation is never zero" {
    // 0 means "not published": a follower waits for a non-zero generation, so a fresh value must not be 0
    try std.testing.expect(genNow() != 0);
    try std.testing.expect(genNow() != 0);
}

test "an fp16 sum rounds once from fp32" {
    const a: f16 = 1.0009765625; // 1 + 2^-10, the fp16 step above 1
    const b: f16 = 0.00048828125; // 2^-11, below the step
    const s: f32 = @as(f32, a) + @as(f32, b);
    try std.testing.expectEqual(@as(f16, 1.0009765625), @as(f16, @floatCast(s)));
    try std.testing.expect(@as(f16, @floatCast(s + b)) > a);
}

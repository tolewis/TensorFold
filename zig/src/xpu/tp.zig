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
/// Seconds a follower's liveness stamp may age before rank 0 calls the pair lost. 30 s, not `peer_dead`: a live
/// follower can spend a few seconds in a call stretch with no all-reduce, and this check runs on every record.
const follower_dead = 30.0;
/// The 5 s warning has fired for the current stale stretch, so a slow follower logs one line, not one a round.
var late_warned = false;
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
pub const Op = enum(u32) { select = 1, store, reset, prompt, more, step, verify, commit, verify_tree, commit_path, verify_multi, keep_multi, keep_paths, step_multi, save_slot, restore_slot, drop_snap, ensure, close, park_snap };

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
/// 180 s matches every other deadline: the device-flag self test builds a kernel on the first open and the other rank
/// waits in recvByte for it.
fn setTimeouts(sock: linux.fd_t) void {
    var tv: linux.timeval = .{ .sec = 180, .usec = 0 };
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
const Segment = struct { map: []align(std.heap.page_size_min) u8, gen: u64 };

fn awaitGen(name: [*:0]const u8, size: usize, want: u64) !Segment {
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
                        _ = std.c.close(fd); // the map keeps the inode alive; the follower re-opens the path to check it
                        return .{ .map = map, .gen = gen };
                    }
                    std.posix.munmap(map);
                } else |_| {}
            }
            _ = std.c.close(fd);
        }
        if (q.now() > deadline) {
            std.log.err("rank {d}: no TP segment with generation {d} at {s} after 180 s: the group is lost\n", .{ q.RANK, want, name });
            std.process.exit(1);
        }
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000_000 };
        while (std.c.nanosleep(&ts, &ts) != 0) {}
    }
}

/// Rank 0's listener: bind, listen, wait for the follower, accept. The socket lives until the caller's open returns.
fn bindAccept(sock_out: *linux.fd_t, base: []const u8, seconds: f64) !linux.fd_t {
    const addr = try Group.sockAddr(base);
    const alen: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + base.len + 6);
    const sock_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    if (failed(sock_rc)) return error.SocketFailed;
    const sock: linux.fd_t = @intCast(sock_rc);
    setTimeouts(sock);
    _ = linux.unlink(@ptrCast(&addr.path));
    if (failed(linux.bind(sock, @ptrCast(&addr), alen))) {
        _ = linux.close(sock);
        return error.BindFailed;
    }
    if (failed(linux.listen(sock, 1))) {
        _ = linux.close(sock);
        return error.ListenFailed;
    }
    var pfd = [1]linux.pollfd{.{ .fd = sock, .events = linux.POLL.IN, .revents = 0 }};
    const ready = linux.poll(&pfd, 1, @intFromFloat(seconds * 1000.0));
    if (failed(ready) or ready == 0) {
        _ = linux.close(sock);
        return error.PeerLost;
    }
    const acc = linux.accept4(sock, null, null, linux.SOCK.CLOEXEC);
    if (failed(acc)) {
        _ = linux.close(sock);
        return error.AcceptFailed;
    }
    sock_out.* = sock;
    setTimeouts(@intCast(acc));
    return @intCast(acc);
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
    /// The segment's path and the generation this rank mapped: a follower re-checks that the path still names this
    /// generation, because a rank 0 that restarted replaced it.
    name: [*:0]const u8 = undefined,
    gen: u64 = 0,
    /// The TF_TP_FLAGLAB socket (the handshake's conn, kept open by open): the lab steps on it at load.
    lab_conn: linux.fd_t = -1,

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

    /// A follower stamps its liveness word while it waits for a record, so rank 0 sees a follower that died while no
    /// request was in flight. The all-reduce path stamps the same word; zero still means "has not reduced yet".
    pub fn stampAlive(g: *Group) void {
        g.aliveWord(g.rank).store(@intFromFloat(q.now()), .release);
    }

    /// Rank 0: a follower whose stamp is older than `follower_dead` is gone, even though no record is due. A follower
    /// stamps at every all-reduce and about once a second while it waits, so a fresh stamp means it is alive. The
    /// 5 s warning fires once for a stale stretch, so a follower that is merely slow shows in the journal without
    /// ending the pair.
    fn checkFollower(g: *Group) void {
        if (g.rank != 0 or g.tp < 2) return;
        const beat = g.aliveWord(1).load(.acquire);
        if (beat == 0) {
            late_warned = false; // it has not reduced yet: it is loading
            return;
        }
        const age = q.now() - @as(f64, @floatFromInt(beat));
        if (age <= 5.0) {
            late_warned = false;
            return;
        }
        if (!late_warned) {
            late_warned = true;
            std.log.warn("qwen27-xpu: the follower's liveness stamp is {d:.1} s old (the pair is called lost at {d:.0} s)", .{ age, follower_dead });
        }
        if (age > follower_dead)
            lostAt(g, "rank 0: the follower's liveness stamp is {d:.1} s old: the group is lost\n", .{age});
    }

    /// Rank 0's idle check, for a caller that runs while no request is in flight (the /health handler): a follower that
    /// died with the pair idle leaves its stamp behind, and rank 0 exits here instead of waiting for the next request.
    pub fn checkIdle(g: *Group) void {
        checkFollower(g);
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

    /// The mode word (offset 24 of rank 0's cache line, next to the pool word at 16): rank 0's run flags
    /// (kv_pool, keep_shared, kv_i8, embed_host, k1, k1_tree, r16, wave). Rank 0 stores it at open before the
    /// generation; a follower reads it right after open and exits on a mismatch before it allocates (item 18 keep-one:
    /// a rank 1 with no pool kept every state whole on its card).
    fn modeWord(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + hdr + 24));
    }
    pub fn readMode(g: *const Group) u64 {
        return g.modeWord().load(.seq_cst);
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
        if (std.c.getenv("TF_TP_RECORD_WAIT")) |v| {
            const want = std.fmt.parseFloat(f64, std.mem.span(v)) catch 180.0;
            // nan and inf parse; inf would turn the ring guard off, so only a finite value of 10 s or more is taken
            if (std.math.isFinite(want) and want >= 10.0) {
                record_wait = want;
            } else {
                std.log.warn("qwen27-xpu: TF_TP_RECORD_WAIT={d} is not finite or is under 10 s, so 180 s is used", .{want});
            }
            std.log.info("qwen27-xpu: TP record wait {d:.0} s (TF_TP_RECORD_WAIT)", .{record_wait});
        }
        const name: [*:0]const u8 = if (std.c.getenv("TF_TP_SHM")) |v| v else "/dev/shm/tf-tp-qwen27";
        const base = std.mem.span(name);
        const size = hdr + q.TP * line + 2 * q.TP * slot_bytes + line + ring_len * @sizeOf(Rec);
        const sa = try sockAddr(base);
        var g: Group = undefined;
        g.lab_conn = -1; // the rank branches re-initialize it from the field default; the early error paths must not read it
        var conn: linux.fd_t = -1;
        var sock: linux.fd_t = -1;
        // The handshake socket lives to the end of this function. A defer inside the rank branch would close it before
        // sendFds, which was the W2 defect. TF_TP_FLAGLAB keeps it for the lab (lab_conn).
        defer {
            if (conn >= 0 and g.lab_conn != conn) _ = linux.close(conn);
        }
        defer {
            if (sock >= 0) {
                _ = linux.close(sock);
                _ = linux.unlink(@ptrCast(&sa.path));
            }
        }
        if (q.RANK == 0) {
            // A fresh generation: the old segment and its socket go first, so no rank reads a dead pair's words and no
            // follower connects to its socket. A plain systemd restart then recovers in any start order.
            _ = linux.unlink(name);
            _ = linux.unlink(@ptrCast(&sa.path));
            const f = std.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            if (f < 0) return error.ShmOpenFailed;
            defer _ = std.c.close(f);
            if (std.c.ftruncate(f, @intCast(size)) != 0) return error.ShmSizeFailed;
            const map = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, f, 0);
            g = .{
                .map = map,
                .rank = 0,
                .tp = q.TP,
                .name = name,
                .gen = 0,
                .mine = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
                .sum = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
            };
            // Every shared word is reset before the generation is published: a follower that reads the generation reads
            // a segment that is already clean. A follower resets nothing, so it cannot clear a word rank 0 is using.
            g.magicWord().store(magic, .seq_cst);
            g.closed().store(0, .seq_cst);
            g.poolWord().store(0, .seq_cst);
            g.modeWord().store(q.modeWord(), .seq_cst); // before the generation: a follower that sees it reads rank 0's flags
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
            // The handshake runs in every mode, host-staged included: the generation crosses it, and a follower takes
            // only the segment that carries it. A rank 0 that cannot listen fails the open, so systemd restarts the pair.
            conn = try bindAccept(&sock, base, 180.0);
            try sendGen(conn, g.gen);
        } else {
            // connect first: rank 0 names the generation of the segment it made, and this rank takes only that one
            conn = try connectPeer(base, 180.0);
            const want = try recvGen(conn);
            const seg = try awaitGen(name, size, want);
            g = .{
                .map = seg.map,
                .rank = q.RANK,
                .tp = q.TP,
                .name = name,
                .gen = seg.gen,
                .mine = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
                .sum = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
            };
        }
        // The listener and the connect are in `open` itself (a peer that never comes fails the open, so systemd
        // restarts the pair). Inside the fd handshake only a broken socket is fatal.
        if (std.c.getenv("TF_TP_HOST") == null) g.openDirect(d, conn) catch |err| switch (err) {
            error.FdSend, error.FdRecv, error.ByteSend, error.ByteRecv => lostAt(&g, "rank {d}: the open did not meet its peer ({s}): the group is lost\n", .{ g.rank, @errorName(err) }),
            else => {
                std.log.warn("qwen27-xpu: TP all-reduce stays host-staged: {s}", .{@errorName(err)});
                g.direct = false;
            },
        };
        if (std.c.getenv("TF_TP_FLAGLAB") != null) g.lab_conn = conn; // the lab steps on the socket at load
        std.log.info("qwen27-xpu: TP {d} rank {d}: all-reduce {s}", .{ g.tp, g.rank, if (g.devflag) "device-flag (the peer's slot pushed over PCIe, the add polls a flag)" else if (g.direct) "device-direct (the peer's slots over PCIe)" else "host-staged" });
        return g;
    }

    /// Rank 0 writes a step record (nothing on another rank): the call `op` with its small arguments and `data`
    /// words. Waits while the ring is full (a follower 64 records behind).
    pub fn record(g: *Group, op: Op, a: u32, b: u32, id: u64, data: []const u32) !void {
        if (g.rank != 0) return;
        // a follower that died while rank 0 was idle leaves its stamp behind: exit here, before the ring wait (180 s)
        // or the all-reduce wait (10 s) would spend the time on it
        checkFollower(g);
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
        stampAlive(g); // rank 0 knows this rank is alive while it waits
        while (@atomicLoad(u64, &r.seq, .acquire) != g.step + 1) : (spins += 1) {
            if (spins > 20_000) {
                // the peer closed the segment (it exited): no record is coming, so this rank exits and systemd
                // restarts the pair instead of this rank spinning until an operator stops it
                if (spins % 100 == 0 and closed_word.load(.acquire) != 0) lostAt(g, "rank {d}: the group closed while it waited for record {d}\n", .{ g.rank, g.step });
                // a rank 0 that restarted replaced the segment with a new generation: this map is the dead one, and
                // the record this rank waits for will never come here. Exit and let systemd restart the pair.
                if (q.now() - checked > 1.0) {
                    checked = q.now();
                    stampAlive(g); // about once a second while it waits
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
            const w0: f64 = if (q.stat.on) q.now() else 0; // XPU_ROUND_STATS: the peer wait time
            if (g.waitPeers()) |_| {} else |e| {
                // the group is lost: this rank closes it and exits instead of waiting 180 s or serving broken rounds
                waitGroupLost(g, e);
            }
            if (q.stat.on) q.stat.peer_s += q.now() - w0;
            try dev.copy(d, g.pbuf[par], g.peer[par].?, n * 2); // over PCIe, on the in-order list
            try d.launch(d.k.tp_add, .{ .x = @intCast((n + 255) / 256) }, .{ partial, g.dslot[par], g.pbuf[par], @as(i32, @intCast(n)) });
            return;
        }
        try dev.copy(d, @ptrCast(g.mine), partial, n * 2);
        d.settle();
        @memcpy(g.slot(g.rank, g.seq)[0..n], g.mine[0..n]);
        g.counter(g.rank).store(g.seq, .seq_cst); // the counter and the liveness word are the rank's heartbeat
        aliveWord(g, g.rank).store(@intFromFloat(q.now()), .release);
        const w0: f64 = if (q.stat.on) q.now() else 0; // XPU_ROUND_STATS: the peer wait time
        if (g.waitPeers()) |_| {} else |e| {
            // the group is lost: this rank closes it and exits instead of waiting 180 s or serving broken rounds
            waitGroupLost(g, e);
        }
        if (q.stat.on) q.stat.peer_s += q.now() - w0;
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
    fn openDirect(g: *Group, d: *Dev, conn: linux.fd_t) !void {
        if (g.tp != 2) return error.DirectNeedsTwoRanks;
        const want_flag = devflag;
        const blk = try d.g.deviceAlloc(if (want_flag) blk_devflag_bytes else blk_legacy_bytes);
        var handle: ze.IpcHandle = undefined;
        try ze.check(d.g.api.zeMemGetIpcHandle(d.g.context, blk, &handle));
        var fds = [1]i32{std.mem.readInt(i32, handle.data[0..4], .little)};
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
            // slot i of this rank's block is where the peer reads it (the peer's block has the same layout): slot 0,
            // slot 1, pbuf 0, pbuf 1. An odd sequence number's exchange writes slot 1, so it must not be 2 * slot_bytes.
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

    /// TF_TP_FLAGLAB: the lab that shows which of the flag's writes a running kernel sees (H6, window p4w0: the
    /// device-flag open and self test pass, the first flag wait never ends). Both ranks run it at load, in step
    /// on the socket open kept for it (lab_conn): one rank is the waiter, the other the writer, and each
    /// experiment runs twice, once a rank the waiter. The word under test is the flag word 0 (blk_flag) of the
    /// waiter's block; E3 writes its own card's buffer instead. It prints one line a experiment and ends the
    /// process. noE4 skips E4 (a give-up there takes about 20 s a kernel).
    pub fn flagLab(g: *Group, d: *Dev, partial: *anyopaque) !void {
        const P = "flaglab rank {d}: ";
        const rank = g.rank;
        const conn = g.lab_conn;
        const forms = [_]i32{ 0, 1, 2 };
        const out = try d.g.deviceAlloc(64); // the probe's result (four words)
        const hout = try d.g.hostAlloc(64); // the host side of it
        const hw = try d.g.hostAlloc(8); // the writer's word
        const w64 = @as(*u64, @ptrCast(@alignCast(hw)));
        const zero = try d.g.hostAlloc(64);
        @memset(@as([*]u8, @ptrCast(zero))[0..64], 0);
        var b: u8 = 0;

        // E1: the write lands before the kernel starts. Expected: spins 0 (a kernel reads the peer's write at all).
        // Zig 0.17 needs a block body for a nested loop (lane p4-cbwait: the base wrote `for (a) |x| for (b) |y| {`).
        for (forms) |form| {
            for (0..2) |i| {
                const w: u32 = @intCast(i); // the waiter
                if (rank == w) {
                    try labCopy(d, off(g.blk.?, blk_flag), zero, 8); // the flag word to 0
                    try sendByte(conn, 1);
                } else {
                    try recvByte(conn, &b);
                    w64.* = 1;
                    try labCopy(d, off(g.peer_blk.?, blk_flag), hw, 8);
                    try sendByte(conn, 2);
                }
                if (rank == w) {
                    try recvByte(conn, &b);
                    const t0 = q.now();
                    try labProbe(d, off(g.blk.?, blk_flag), 1, 1 << 24, form, out);
                    d.settle();
                    const o = try labOut(d, out, hout);
                    std.debug.print(P ++ "E1 form {d} spins {d} last 0x{x} post 0x{x} gave {d} launch+settle {d:.2} ms\n", .{ rank, form, o[0], o[1], o[2], o[3], (q.now() - t0) * 1e3 });
                }
            }
        }

        // E2: the write lands while the kernel spins. The main experiment.
        for (forms) |form| {
            for (0..2) |i| {
                const w: u32 = @intCast(i);
                if (rank == w) {
                    try labCopy(d, off(g.blk.?, blk_flag), zero, 8); // the flag word to 0
                    try sendByte(conn, 1); // the reset is down
                } else {
                    try recvByte(conn, &b);
                }
                var t0: f64 = 0; // the probe's launch time (E2: the waiter's block ends before the print)
                if (rank == w) {
                    try labProbe(d, off(g.blk.?, blk_flag), 2, 1 << 28, form, out);
                    t0 = q.now();
                    try sendByte(conn, 2); // the probe runs
                } else {
                    try recvByte(conn, &b);
                    var ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000_000 };
                    while (std.c.nanosleep(&ts, &ts) != 0) {} // the kernel runs
                    w64.* = 2;
                    try labCopy(d, off(g.peer_blk.?, blk_flag), hw, 8);
                    try sendByte(conn, 3); // the write is down
                }
                if (rank == w) {
                    try recvByte(conn, &b);
                    const t1 = q.now();
                    d.settle();
                    const t2 = q.now();
                    const o = try labOut(d, out, hout);
                    std.debug.print(P ++ "E2 form {d} spins {d} last 0x{x} post 0x{x} gave {d} t1-t0 {d:.2} ms t2-t1 {d:.2} ms\n", .{ rank, form, o[0], o[1], o[2], o[3], (t1 - t0) * 1e3, (t2 - t1) * 1e3 });
                    // E2b: a new kernel right after E2, no new write. Does it see what the spinning kernel did not?
                    try labProbe(d, off(g.blk.?, blk_flag), 2, 1 << 16, form, out);
                    d.settle();
                    const o2 = try labOut(d, out, hout);
                    std.debug.print(P ++ "E2b form {d} spins {d} last 0x{x} post 0x{x} gave {d}\n", .{ rank, form, o2[0], o2[1], o2[2], o2[3] });
                }
            }
        }

        // E3: the same on one card: a kernel spinning on a word its own card's copy engine writes. Both ranks run
        // it at the same time, each alone (no socket).
        const wbuf = try d.g.deviceAlloc(64); // the word (not exported)
        const list2 = try d.g.immediate(1, 0); // the copy queue group, in order
        for (forms) |form| {
            try labCopy(d, wbuf, zero, 8); // the word to 0
            try labProbe(d, wbuf, 3, 1 << 28, form, out);
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000_000 };
            while (std.c.nanosleep(&ts, &ts) != 0) {} // the kernel runs
            w64.* = 3;
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list2, wbuf, hw, 8, null, 0, null));
            const s0 = q.now();
            try ze.check(d.g.api.zeCommandListHostSynchronize(list2, std.math.maxInt(u64)));
            const s1 = q.now();
            d.settle();
            const o = try labOut(d, out, hout);
            std.debug.print(P ++ "E3 form {d} spins {d} last 0x{x} post 0x{x} gave {d} copywait {d:.2} ms\n", .{ rank, form, o[0], o[1], o[2], o[3], (s1 - s0) * 1e3 });
        }

        // E5: the wait as a counter based driver event on the waiter's flag word (spec WP1, "Second design"). The
        // event's counter is a word the application manages: the event is complete when the word is at or above the
        // completion value, and a launch that waits for it starts only then. Variants: v2 the device word and a host
        // word that never changes, v1 the device word only, v0 the device word with no wait scope. Each variant runs
        // with rank 0 and then rank 1 as the waiter: E5a the word is written before the launch, E5b while the launch
        // waits. E5c times one create plus destroy. A create that fails prints its rc and skips that exchange.
        if (d.g.cbapi == null) {
            std.debug.print(P ++ "E5 skipped: no counter based event symbols\n", .{rank});
        } else {
            const cb = d.g.cbapi.?;
            var maxv: u64 = 0;
            const mrc = cb.zeDeviceGetCounterBasedEventMaxValue(d.g.device, &maxv);
            std.debug.print(P ++ "E5 max value rc 0x{x} value {d}\n", .{ rank, mrc, maxv });
            const rrc = cb.zeContextMakeMemoryResident(d.g.context, d.g.device, g.blk.?, blk_devflag_bytes);
            std.debug.print(P ++ "E5 make resident rc 0x{x} bytes {d}\n", .{ rank, rrc, blk_devflag_bytes });
            const hhost = try d.g.hostAlloc(64); // v2's host word: set to 0 and never changed
            @memset(@as([*]u8, @ptrCast(hhost))[0..64], 0);
            const word = off(g.blk.?, blk_flag);
            const peer_word = off(g.peer_blk.?, blk_flag);
            for ([_]u32{ 2, 1, 0 }) |variant| for (0..2) |i| {
                const w: u32 = @intCast(i); // the waiter
                for ([_]bool{ true, false }) |pre| { // E5a (the write first) and E5b (the write while the launch waits)
                    const want: u64 = if (pre) 5 else 6;
                    var ev: ze.H = null;
                    var skip = false;
                    var t0: f64 = 0;
                    var t1: f64 = 0;
                    var s: u8 = 0;
                    if (rank == w) {
                        try labCopy(d, word, zero, 8); // the word to 0
                        if (pre) {
                            try sendByte(conn, 10); // the reset is down
                            try recvByte(conn, &s); // the write is down
                            ev = labE5Create(d, cb, variant, want, word, hhost, rank);
                            skip = ev == null;
                            try sendByte(conn, if (skip) @as(u8, 2) else 1); // go or skip
                            if (!skip) {
                                t0 = q.now();
                                std.debug.print(P ++ "E5a v{d} launch\n", .{ rank, variant });
                                try labE5Launch(d, ev.?, word, out);
                            }
                        } else {
                            ev = labE5Create(d, cb, variant, want, word, hhost, rank);
                            skip = ev == null;
                            try sendByte(conn, if (skip) @as(u8, 2) else 1); // go or skip
                            if (!skip) {
                                std.debug.print(P ++ "E5b v{d} launch\n", .{ rank, variant });
                                try labE5Launch(d, ev.?, word, out);
                                t0 = q.now();
                                try sendByte(conn, 3); // the launch is up
                            }
                        }
                    } else {
                        if (pre) {
                            try recvByte(conn, &s); // the reset is down
                            w64.* = want;
                            try labCopy(d, peer_word, hw, 8);
                            try sendByte(conn, 11); // the write is down
                            try recvByte(conn, &s); // go or skip
                            skip = s == 2;
                        } else {
                            try recvByte(conn, &s); // go or skip
                            skip = s == 2;
                            if (!skip) {
                                try recvByte(conn, &s); // the launch is up
                                std.time.sleep(200 * std.time.MilliSecond); // the launch waits
                                w64.* = want;
                                try labCopy(d, peer_word, hw, 8);
                                try sendByte(conn, 12); // the write is down
                            }
                        }
                    }
                    if (skip) continue;
                    if (rank == w) {
                        if (!pre) {
                            try recvByte(conn, &s); // the write is down
                            t1 = q.now();
                        }
                        const tw = q.now();
                        const wrc = d.g.api.zeCommandListHostSynchronize(d.list, 5_000_000_000);
                        const t2 = q.now();
                        const stuck = wrc != 0;
                        if (stuck) std.debug.print(P ++ "E5 v{d} not complete in 5 s t2-t1 {d:.2} ms\n", .{ rank, variant, (t2 - tw) * 1e3 });
                        try sendByte(conn, if (stuck) @as(u8, 2) else 1); // ok or stuck
                        try recvByte(conn, &s); // the writer's answer: the word again when stuck
                        if (stuck) { // the wait may now end: no settle before the writer's write, or it would block here
                            const wrc2 = d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64));
                            const o2 = try labOut(d, out, hout);
                            std.debug.print(P ++ "E5 v{d} second wait rc 0x{x} out1 {d}\n", .{ rank, variant, wrc2, o2[1] });
                        } else if (pre) {
                            const o = try labOut(d, out, hout);
                            std.debug.print(P ++ "E5a v{d} wait rc 0x{x} out1 {d} launch+wait {d:.2} ms t2-t1 {d:.2} ms\n", .{ rank, variant, wrc, o[1], (tw - t0) * 1e3, (t2 - tw) * 1e3 });
                        } else {
                            const o = try labOut(d, out, hout);
                            std.debug.print(P ++ "E5b v{d} wait rc 0x{x} out1 {d} t1-t0 {d:.2} ms t2-t1 {d:.2} ms\n", .{ rank, variant, wrc, o[1], (t1 - t0) * 1e3, (t2 - t1) * 1e3 });
                            if (o[1] == 0) std.debug.print(P ++ "E5b v{d} out1 0: the driver did not hold the launch\n", .{ rank, variant });
                        }
                        _ = d.g.api.zeEventDestroy(ev.?);
                    } else {
                        try recvByte(conn, &s); // ok or stuck
                        if (s == 2) {
                            w64.* = want + 1;
                            try labCopy(d, peer_word, hw, 8);
                        }
                        try sendByte(conn, 13); // the writer's answer
                    }
                }
            };
            // E5c: the cost of one create and destroy, each rank alone, v2 (1,000 events; the first rc that is not 0).
            var total: f64 = 0;
            var okc: u32 = 0;
            var bad: u32 = 0;
            var first_rc: u32 = 0;
            for (0..1000) |_| {
                const tc = q.now();
                var evc: ze.H = null;
                const waddr: *u64 = @ptrCast(@alignCast(word));
                const haddr: *u64 = @ptrCast(@alignCast(hhost));
                var sync: ze.CbExternalSync = .{ .deviceAddress = waddr, .hostAddress = haddr, .completionValue = 1 };
                var desc: ze.CbEventDesc = .{ .flags = 1, .signal = 2, .wait = 2, .pNext = &sync };
                const rc = cb.zeEventCounterBasedCreate(d.g.context, d.g.device, &desc, &evc);
                if (rc != 0) {
                    if (bad == 0) first_rc = rc;
                    bad += 1;
                    continue;
                }
                _ = d.g.api.zeEventDestroy(evc);
                total += q.now() - tc;
                okc += 1;
            }
            std.debug.print(P ++ "E5c v2 create+destroy {d} of 1000 ok, mean {d:.2} us, first bad rc 0x{x}\n", .{ rank, okc, if (okc > 0) total / @as(f64, @floatFromInt(okc)) * 1e6 else 0.0, first_rc });
        }

        // E4: the real protocol, from the state after selfTest: three all-reduces. noE4 skips it.
        const no_e4: bool = if (std.c.getenv("TF_TP_FLAGLAB")) |v| std.mem.eql(u8, std.mem.span(v), "noE4") else false;
        if (!no_e4) {
            // The reset, as in selfTest: the flag words and the peer-consumed words of this rank's block (32 bytes
            // from blk_flag), this rank's consumed counters, the timeout word, and g.seq.
            try labCopy(d, off(g.blk.?, blk_flag), zero, 32);
            try labCopy(d, g.own_cons.?, zero, 16);
            try labCopy(d, off(g.blk.?, blk_timeout), zero, 4);
            g.seq = 0;
            try sendByte(conn, 4); // both ranks are reset
            try recvByte(conn, &b);
            const rb = try d.g.hostAlloc(48); // the timeout word (4) and the flag and consumed words (32)
            for (0..3) |i| {
                const t0 = q.now();
                try g.allReduce(d, partial, 1);
                d.settle();
                const t1 = q.now();
                try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, rb, off(g.blk.?, blk_timeout), 4, null, 0, null));
                try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(rb, 4), off(g.blk.?, blk_flag), 32, null, 0, null));
                d.settle();
                const r4 = @as(*const [10]u32, @ptrCast(@alignCast(rb)));
                std.debug.print(P ++ "E4 exchange {d} call+settle {d:.2} ms timeout 0x{x} flag0 0x{x} flag1 0x{x} pcons0 0x{x} pcons1 0x{x}\n", .{ rank, i, (t1 - t0) * 1e3, r4[0], r4[1], r4[2], r4[3], r4[4] });
                try labCopy(d, off(g.blk.?, blk_timeout), zero, 4); // the timeout word to 0
            }
        }

        std.debug.print(P ++ "done\n", .{rank});
        std.process.exit(0);
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

/// The flag lab: a copy on d.list, then a settle (as push and selfTest do).
fn labCopy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    try dev.copy(d, dst, src, n);
    d.settle();
}

/// The flag lab: queue the probe's four words to the host and settle; the values are good until the next launch
/// into `out`.
fn labOut(d: *Dev, out: *anyopaque, hout: *anyopaque) !*[4]u64 {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, hout, out, 32, null, 0, null));
    d.settle();
    return @ptrCast(@alignCast(hout));
}

/// The flag lab: the spin probe (one group of 16, only global id 0 works). Five arguments: the pointer, the two
/// u64 words, the i32 form, the pointer.
fn labProbe(d: *Dev, p: *anyopaque, want: u64, max_spins: u64, form: i32, out: *anyopaque) !void {
    try d.launch(d.k.tp_probe, .{ .x = 1 }, .{ p, want, max_spins, form, out });
}

/// The flag lab E5: a counter based event on `word` (device) with `host` as the external host word; null when the
/// driver refuses the create (its rc is printed in hex).
fn labE5Create(d: *Dev, cb: ze.CbApi, variant: u32, want: u64, word: *anyopaque, host: *anyopaque, rank: u32) ze.H {
    const waddr: *u64 = @ptrCast(@alignCast(word));
    const haddr: *u64 = @ptrCast(@alignCast(host));
    var sync: ze.CbExternalSync = .{ .deviceAddress = waddr, .completionValue = want };
    var desc: ze.CbEventDesc = .{ .flags = 1, .signal = 2, .wait = 2 }; // immediate lists; device scope on both sides
    if (variant == 2 or variant == 0) sync.hostAddress = haddr;
    if (variant == 0) desc.wait = 0;
    desc.pNext = &sync;
    var ev: ze.H = null;
    const rc = cb.zeEventCounterBasedCreate(d.g.context, d.g.device, &desc, &ev);
    std.debug.print("flaglab rank {d}: E5 event v{d} create rc 0x{x}\n", .{ rank, variant, rc });
    if (rc != 0) return null;
    return ev;
}

/// The flag lab E5: the probe launch that waits for the counter based event (no signal event, one wait event).
fn labE5Launch(d: *Dev, ev: ze.H, word: *anyopaque, out: *anyopaque) !void {
    try d.g.arg(d.k.tp_probe, 0, word);
    try d.g.arg(d.k.tp_probe, 1, @as(u64, 0)); // want
    try d.g.arg(d.k.tp_probe, 2, @as(u64, 0)); // max_spins
    try d.g.arg(d.k.tp_probe, 3, @as(i32, 0)); // form
    try d.g.arg(d.k.tp_probe, 4, out);
    var groups: ze.GroupCount = .{ .x = 1 };
    var wa = [1]ze.H{ev};
    try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, d.k.tp_probe, &groups, null, 1, &wa));
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
        @intFromPtr(g.modeWord()) - base,
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
    try std.testing.expectEqual(@as(usize, hdr + 24), words[9]);
}

test "the device-direct block: a rank's slot i is where its peer reads it, and the four regions do not overlap" {
    // lane p2-int-wedge: W4 wrote slot i at 2 * slot_bytes, so an odd exchange wrote pbuf 0 and the peer read stale
    // memory at slot 1. This test pins the layout that allReduce uses: this rank's block is slot 0, slot 1, pbuf 0,
    // pbuf 1, and the peer reads slot i at i * slot_bytes of its own block.
    const blk: *anyopaque = @ptrFromInt(0x1000_0000);
    const peer: *anyopaque = @ptrFromInt(0x2000_0000);
    for (0..2) |i| {
        const mine = @intFromPtr(off(blk, i * slot_bytes));
        const theirs = @intFromPtr(off(peer, i * slot_bytes));
        try std.testing.expectEqual(@intFromPtr(blk) + i * slot_bytes, mine);
        try std.testing.expectEqual(@intFromPtr(peer) + i * slot_bytes, theirs);
        // the peer's slot i is the same offset inside its block as this rank's slot i: one all-reduce layout
        try std.testing.expectEqual(mine - @intFromPtr(blk), theirs - @intFromPtr(peer));
    }
    // the four regions of this rank's block: slot 0, slot 1, pbuf 0, pbuf 1
    const starts = [_]usize{
        @intFromPtr(off(blk, 0)),
        @intFromPtr(off(blk, slot_bytes)),
        @intFromPtr(off(blk, 2 * slot_bytes)),
        @intFromPtr(off(blk, 3 * slot_bytes)),
    };
    for (starts, 0..) |a, i| {
        try std.testing.expectEqual(@intFromPtr(blk) + i * slot_bytes, a);
        for (starts, 0..) |b, j| {
            if (i == j) continue;
            // the regions are slot_bytes wide and back to back: no overlap, and no gap
            try std.testing.expect(a + slot_bytes <= b or b + slot_bytes <= a);
        }
    }
    try std.testing.expectEqual(4 * slot_bytes, blk_legacy_bytes);
    // the device-flag block holds the same two slots and then the flags, so its slots have the same offsets
    try std.testing.expectEqual(2 * slot_bytes, blk_slots);
    try std.testing.expect(blk_devflag_bytes > blk_slots);
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

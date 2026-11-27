//! Two-process all-reduce for Inkling. The push, the flag, the socket and the ring are the production Qwen
//! Group (fleet-unified zig/src/xpu/tp.zig at e5506ae, TF_TP_DEVFLAG=3). Width is 2304. The shm name is not Qwen's.
const std = @import("std");
const linux = std.os.linux;
const xpu = @import("xpu");
const ze = xpu.ze;
const images = @import("inkling2_xpu_images");
const place = @import("place.zig");
const rt = xpu.rt;
const qg = @import("qwen_group");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");

const D = place.hidden;
const rows_max: u32 = 64;
const slot_bytes = rows_max * D * 2;
const blk_slots = 2 * slot_bytes;
const blk_flag = blk_slots;
const blk_bytes = blk_flag + 64;
const head_bytes = 100029 * 4;
const shm_default = "/dev/shm/tf-tp-inkling2";
const magic: u64 = 0x494E_4B32_5450_0002; // INK2TP, not the Qwen magic; 0002 carries the segment table
/// The magic's byte in the map: the first 128 bytes hold the done word and nothing else.
const magic_at: usize = 8;

/// Rank 1 checks the map's magic when it attaches: a rank 0 of another binary maps the ring at other offsets, so every
/// record it reads is garbage. A mismatch stops the follower at start with a clear line.
pub fn checkMagic(map: []const u8) !void {
    if (map.len < magic_at + 8) return error.Inkling2TpMagic;
    if (std.mem.readInt(u64, map[magic_at..][0..8], .little) != magic) return error.Inkling2TpMagic;
}

const IpcHandle = extern struct { data: [64]u8 };
const ZexWriteDesc = extern struct { writeScope: u32 = 2 };
const ZexWrite = *const fn (ze.H, *const ZexWriteDesc, ?*anyopaque, u64) callconv(.c) u32;
const CbEventDesc = extern struct { stype: u32 = 0x0002003A, pNext: ?*const anyopaque = null, flags: u32 = 1, signal: u32 = 0, wait: u32 = 0 };
const CbExternalSync = extern struct { stype: u32 = 0x0002003B, pNext: ?*const anyopaque = null, deviceAddress: ?*u64 = null, hostAddress: ?*u64 = null, completionValue: u64 = 0 };
const CbCreate = *const fn (ze.H, ze.H, *const CbEventDesc, *ze.H) callconv(.c) u32;
const GetIpc = *const fn (ze.H, *const anyopaque, *IpcHandle) callconv(.c) u32;
const OpenIpc = *const fn (ze.H, ze.H, IpcHandle, u32, *?*anyopaque) callconv(.c) u32;

fn watchPeer(pid: i32) void {
    var gone: u8 = 0;
    while (true) {
        var ts: linux.timespec = .{ .sec = 1, .nsec = 0 };
        _ = linux.nanosleep(&ts, null);
        if (linux.kill(pid, @enumFromInt(0)) == 0) {
            gone = 0;
            continue;
        }
        gone += 1;
        if (gone >= 2) {
            std.log.err("inkling2: peer {d} is gone; exit", .{pid});
            std.c._exit(1);
        }
    }
}

const TsFn = *const fn (ze.H, *anyopaque, ze.H, u32, ?*ze.H) callconv(.c) u32;

/// The leader's records: a forward or a prompt chunk with its segment table, a keep, a release, and the prompt cache's
/// save, restore and drop. Rank 1 runs each stream's state in its own cache, so the table names the slot of every row.
pub const Op = enum(u32) { forward = 1, close = 2, keep = 3, release = 4, save = 5, restore = 6, drop = 7 };
/// One stream's rows in a record: its slot (0 to `--parallel` - 1, or `--parallel` for the warm-up cache), the rows,
/// the cache position they start at, the cache's capacity, and the rows a keep accepts.
pub const Seg = extern struct { slot: u32 = 0, rows: u32 = 0, base: u32 = 0, capacity: u32 = 0, accepted: u32 = 0 };
pub const max_segs: u32 = 32;
/// `n` is the rows of `ids` for a forward or a chunk. For `save`, `restore` and `drop` it is the snapshot id and `base`
/// is the position a save takes.
pub const Rec = extern struct { seq: u64, op: Op, n: u32, flags: u32, base: u32, nseg: u32, segs: [max_segs]Seg, ids: [2048]u32 };

/// What rank 1 must do to its caches for one record, in the order it must do it. A pure function: the host test checks it
/// without a device. `run_forward` and `run_prefill` run the record's own segment table, so they carry no slot.
pub const Action = union(enum) {
    close,
    /// Make the slot's cache at this capacity, or keep the one it has.
    ensure: Seg,
    /// Set the slot's cache position to this base.
    seek: struct { slot: u32, base: u32 },
    /// Open a cache transaction of these rows on the slot.
    begin: struct { slot: u32, rows: u32 },
    /// A prompt chunk at position 0 reads no history of the request before: zero the slot's convolution states.
    zero_conv: struct { slot: u32 },
    run_forward,
    run_prefill,
    keep: struct { slot: u32, accepted: u32 },
    release: struct { slot: u32 },
    save: struct { slot: u32, capacity: u32, snap: u32, at: u32 },
    restore: struct { slot: u32, capacity: u32, snap: u32 },
    drop: struct { snap: u32 },
};

fn put(out: []Action, at: usize, a: Action) !void {
    if (at >= out.len) return error.Inkling2RecordShape;
    out[at] = a;
}

/// The follower's cache actions for `rec`. Every record that names a slot puts its capacity in `segs[0]`, so a slot's
/// first record (a chunk or a restore) can make the cache.
pub fn plan(rec: Rec, out: []Action) ![]Action {
    if (rec.nseg > max_segs) return error.Inkling2RecordShape;
    const segs = rec.segs[0..rec.nseg];
    var n: usize = 0;
    switch (rec.op) {
        .close => {
            try put(out, n, .close);
            return out[0 .. n + 1];
        },
        .forward => {
            if (rec.nseg == 0 or rec.n > st.max_prefill_rows) return error.Inkling2RecordShape;
            // A prompt chunk carries the whole chunk; a verify window carries one window's rows a segment.
            const rows_cap: u32 = if (rec.flags & 4 != 0) st.max_prefill_rows else rows_max;
            var rows: u64 = 0;
            for (segs) |s| {
                if (s.slot > max_segs or s.rows == 0 or s.rows > rows_cap or s.capacity == 0) return error.Inkling2RecordShape;
                if (@as(u64, s.base) + s.rows > s.capacity) return error.Inkling2RecordShape;
                rows += s.rows;
                try put(out, n, .{ .ensure = .{ .slot = s.slot, .rows = 0, .base = 0, .capacity = s.capacity, .accepted = 0 } });
                n += 1;
                // A prompt chunk at position 0 must not read the previous request's convolution history (as `feed` clears it).
                if (rec.flags & 4 != 0 and s.base == 0) {
                    try put(out, n, .{ .zero_conv = .{ .slot = s.slot } });
                    n += 1;
                }
                try put(out, n, .{ .seek = .{ .slot = s.slot, .base = s.base } });
                n += 1;
                // Bit 3 marks a verify window: the leader opens a transaction on every segment, a one-row window too.
                // Bit 1 alone is a warm-up or plain forward, which opens none.
                if (rec.flags & 8 != 0) {
                    try put(out, n, .{ .begin = .{ .slot = s.slot, .rows = s.rows } });
                    n += 1;
                }
            }
            if (rows != rec.n) return error.Inkling2RecordShape;
            try put(out, n, if (rec.flags & 4 != 0) .run_prefill else .run_forward);
            return out[0 .. n + 1];
        },
        .keep => {
            if (rec.nseg == 0) return error.Inkling2RecordShape;
            for (segs) |s| {
                if (s.slot > max_segs or s.accepted > s.rows) return error.Inkling2RecordShape;
                try put(out, n, .{ .keep = .{ .slot = s.slot, .accepted = s.accepted } });
                n += 1;
            }
            return out[0..n];
        },
        // A release frees the slot's cache. Rank 1 rolls back a transaction that is still open before it frees.
        .release => {
            if (rec.nseg == 0 or segs[0].slot > max_segs) return error.Inkling2RecordShape;
            try put(out, n, .{ .release = .{ .slot = segs[0].slot } });
            return out[0 .. n + 1];
        },
        .save => {
            if (rec.nseg == 0 or segs[0].slot > max_segs or segs[0].capacity == 0 or rec.base > segs[0].capacity) return error.Inkling2RecordShape;
            try put(out, n, .{ .save = .{ .slot = segs[0].slot, .capacity = segs[0].capacity, .snap = rec.n, .at = rec.base } });
            return out[0 .. n + 1];
        },
        .restore => {
            if (rec.nseg == 0 or segs[0].slot > max_segs or segs[0].capacity == 0) return error.Inkling2RecordShape;
            try put(out, n, .{ .ensure = .{ .slot = segs[0].slot, .rows = 0, .base = 0, .capacity = segs[0].capacity, .accepted = 0 } });
            n += 1;
            try put(out, n, .{ .restore = .{ .slot = segs[0].slot, .capacity = segs[0].capacity, .snap = rec.n } });
            return out[0 .. n + 1];
        },
        .drop => {
            try put(out, n, .{ .drop = .{ .snap = rec.n } });
            return out[0 .. n + 1];
        },
    }
}

pub const Group = struct {
    tp: u32,
    rank: u32,
    r: *rt.Runtime,
    ctx: *const xpu.Context,
    add: xpu.Kernel = undefined,
    up: xpu.Kernel = undefined,
    add_f32: xpu.Kernel = undefined,
    down: xpu.Kernel = undefined,
    f32row: ?*anyopaque = null,
    stamp_fn: ?TsFn = null,
    stamp_buf: ?*anyopaque = null,
    stamp_on: bool = false,
    /// Counter events of this token. Qwen destroys them after the stream settles. A live set makes the next wait slower.
    evs: [256]ze.H = undefined,
    ev_n: u32 = 0,
    zero: xpu.Kernel = undefined,
    zero_w: xpu.Kernel = undefined,
    set_launch: xpu.Kernel = undefined,
    /// Three u32s: the indirect group count for one row of experts.
    gc: ?*anyopaque = null,
    filter: xpu.Kernel = undefined,
    mod: xpu.Module = undefined,
    blk: ?*anyopaque = null,
    peer_blk: ?*anyopaque = null,
    head: ?*anyopaque = null,
    peer_head: ?*anyopaque = null,
    zex: ?ZexWrite = null,
    cb: ?CbCreate = null,
    seq: u64 = 0,
    map: []align(4096) u8 = undefined,
    conn: linux.fd_t = -1,
    ex: u8 = 0,
    rec_seq: u64 = 0,
    idle_logged: bool = false,
    qdev: qg.dev.Dev = undefined,
    qgroup: ?qg.tp.Group = null,

    pub fn open(gpa: std.mem.Allocator, r: *rt.Runtime, ctx: *const xpu.Context, p: place.Place) !*Group {
        if (!p.split()) return error.Inkling2Tp;
        const g = try gpa.create(Group);
        g.* = .{ .tp = p.tp, .rank = p.rank, .r = r, .ctx = ctx };
        errdefer gpa.destroy(g);
        g.mod = try xpu.Module.loadNative(ctx, &images.tp);
        g.add = try g.mod.kernel("tp_add", .{ 256, 1, 1 });
        g.up = try g.mod.kernel("tp_up", .{ 256, 1, 1 });
        g.add_f32 = try g.mod.kernel("tp_add_f32", .{ 256, 1, 1 });
        g.down = try g.mod.kernel("tp_down", .{ 256, 1, 1 });
        g.zero = try g.mod.kernel("tp_keep_local", .{ 256, 1, 1 });
        g.zero_w = try g.mod.kernel("tp_zero_foreign_weight", .{ 256, 1, 1 });
        g.set_launch = try g.mod.kernel("tp_set_launch", .{ 1, 1, 1 });
        g.gc = (try g.r.alloc(16)).ptr;
        g.filter = try g.mod.kernel("tp_filter_items", .{ 1, 1, 1 });
        const name = shmName();
        // A stale shm file left by an earlier run is the same path. Rank 0 unlinks it and creates a fresh one before it
        // binds the socket. Rank 1 therefore maps only after the socket handshake connects it to the live rank 0, so it
        // opens the fresh inode. Mapping first let rank 1 keep the stale file and read an old record word (2026-11-26:
        // "wait saw seq 2, wanted 3", a drafted run started after production left /dev/shm/tf-tp-inkling2-prod).
        if (p.rank == 0) {
            g.map = try mapShm(name, p.rank);
            std.mem.writeInt(u64, g.map[magic_at..][0..8], magic, .little);
            try g.handshake(name);
        } else {
            try g.handshake(name);
            g.map = try mapShm(name, p.rank);
            // A rank 0 of another binary maps the ring at other offsets, so every record this rank reads is garbage.
            checkMagic(g.map) catch |err| {
                std.log.err("inkling2: TP rank 1: the shm magic is {x}, this binary wants {x}; rank 0 runs another binary, so stop", .{ std.mem.readInt(u64, g.map[magic_at..][0..8], .little), magic });
                return err;
            };
        }
        qg.q.setRank(p.rank);
        _ = setenv("TF_TP_SHM", "/dev/shm/tf-tp-inkling2-qwen", 1);
        g.qdev = try qg.dev.bind(@constCast(&r.drv.lib), ctx.gpu.driver, r.ctx, r.device, r.list, g.add.handle, ctx.gpu.props.timerResolution);
        g.qgroup = try qg.tp.Group.open(&g.qdev);
        std.log.info("inkling2: TP {d} rank {d}: Qwen Group exchange", .{ g.tp, g.rank });
        if (std.c.getenv("TF_TP_PEER_PID")) |z| {
            const pid = std.fmt.parseInt(i32, std.mem.span(z), 10) catch 0;
            if (pid > 0) {
                const watch = std.Thread.spawn(.{}, watchPeer, .{pid}) catch null;
                if (watch) |t| t.detach();
            }
        }
        live = g;
        if (std.c.getenv("TF_TP_EXBENCH")) |v| if (v[0] != '0') try g.exchangeBench();
        return g;
    }

    /// 128 exchanges at 1 row of 2304 bf16. Both ranks must call it. Prints the Qwen bench line shape.
    fn exchangeBench(g: *Group) !void {
        var buf = try g.r.alloc(D * 2);
        defer buf.free();
        try g.r.stream(g.ctx).fill8(buf.ptr, 0, D * 2);
        g.stamp_on = std.c.getenv("TF_TP_DEVSTAMP") != null;
        try g.allReduce(buf.ptr.?, 1);
        try g.r.sync();
        xpu.profile.reportAppend(g.rank);
        const t0 = now();
        for (0..128) |_| try g.allReduce(buf.ptr.?, 1);
        try g.r.sync();
        const ms = (now() - t0) * 1e3;
        xpu.profile.reportAppend(g.rank);
        std.log.info("inkling2: exchange ms a round (128 exchanges) by rows: 1 {d:.2}", .{ ms });
        if (std.c.getenv("TF_TP_PING") != null) if (g.qgroup) |*qg_g| try qg_g.pingBench(&g.qdev);
    }

    /// `partial` holds `rows` x 2304 fp16. On return it holds the sum. Both ranks call this in the same order.
    /// The sequence is exPush then tp_add, as Group.allReduce does for TF_TP_DEVFLAG=3 at e5506ae.
    pub fn allReduce(g: *Group, partial: *anyopaque, rows: u32) !void {
        if (rows == 0 or rows > 2048) return error.Inkling2Tp;
        if (g.qgroup == null) return error.Inkling2Tp;
        qg.dev.noteKernel(xpu.profile.last_kernel);
        if (std.c.getenv("TF_TP_LAUNCHLOG") != null) std.log.info("inkling2: reduce rank {d} rows {d} seq {d}", .{ g.rank, rows, g.qgroup.?.seq + 1 });
        // TF_TP_AR1: one row an exchange, the shape a plain decode takes. A test switch, not a speed path.
        if (rows > 1 and std.c.getenv("TF_TP_AR1") != null) {
            var r: u32 = 0;
            while (r < rows) : (r += 1) {
                try g.qgroup.?.allReduce(&g.qdev, @ptrFromInt(@intFromPtr(partial) + @as(usize, r) * D * 2), 1);
            }
            return;
        }
        return g.qgroup.?.allReduce(&g.qdev, partial, rows);
    }

    /// TF_TP_SEQPAR: rank r keeps rows [r*half, (r+1)*half) summed and leaves the other half summed as well, which the
    /// gather overwrites. Each rank pushes the half the other rank owns, so the bytes are half an all-reduce's.
    pub fn reduceScatterRows(g: *Group, partial: *anyopaque, rows: u32, row_bytes: usize) !void {
        const half = rows / 2;
        if (rows < 4 or rows % 2 != 0 or row_bytes * rows > rows_max * D * 2) return error.Inkling2Tp;
        if (g.qgroup == null) return error.Inkling2Tp;
        qg.dev.noteKernel(xpu.profile.last_kernel);
        if (g.rank == 0) {
            return g.qgroup.?.allReduceBytes(&g.qdev, @ptrFromInt(@intFromPtr(partial) + half * row_bytes), rows - half, row_bytes);
        }
        return g.qgroup.?.allReduceBytes(&g.qdev, partial, half, row_bytes);
    }

    /// TF_TP_SEQPAR: each rank processed its own half. Zero the other half, then sum, so each rank ends with both halves
    /// processed. A copy exchange would save the zero fill's bytes; this keeps one exchange shape.
    pub fn allGatherRows(g: *Group, partial: *anyopaque, rows: u32, row_bytes: usize) !void {
        const half = rows / 2;
        if (rows < 4 or rows % 2 != 0 or row_bytes * rows > rows_max * D * 2) return error.Inkling2Tp;
        if (g.qgroup == null) return error.Inkling2Tp;
        qg.dev.noteKernel(xpu.profile.last_kernel);
        const own: usize = if (g.rank == 0) 0 else half * row_bytes;
        const other: usize = if (g.rank == 0) half * row_bytes else 0;
        const src: *anyopaque = @ptrFromInt(@intFromPtr(partial) + own);
        const dst: *anyopaque = @ptrFromInt(@intFromPtr(partial) + other);
        return g.qgroup.?.allReduceCopy(&g.qdev, src, dst, half, row_bytes);
    }

    /// Rank 1 pushes its f32 logit slice. Rank 0 copies it to `dst` (the high half of its logit row).
    pub fn headPull(g: *Group, src: *anyopaque, dst: *anyopaque, n_f32: u32) !void {
        const bytes: usize = @as(usize, n_f32) * 4;
        if (bytes > head_bytes) return error.Inkling2Tp;
        g.seq += 1;
        // The handshake holds one row. Rank 1 must not write the next row until rank 0 has finished reading this one.
        const ready = blk_flag + 16;
        const consumed = blk_flag + 24;
        if (g.rank != 0) {
            try g.r.stream(g.ctx).copy(g.peer_head.?, src, bytes);
            try g.writeFlag(off(g.peer_blk.?, ready));
            try g.waitFlag(off(g.blk.?, consumed));
        } else {
            try g.waitFlag(off(g.blk.?, ready));
            try g.r.stream(g.ctx).copy(dst, g.head, bytes);
            try g.r.sync();
            try g.writeFlag(off(g.peer_blk.?, consumed));
        }
    }

    /// Rank 0 tells rank 1 the next record. `segs` is the segment table of a forward or a prompt chunk: each slot with
    /// its rows, base and capacity. `keep` names a segment with the rows it accepts, `release` names the slot, and
    /// `save`, `restore` and `drop` put the snapshot id in `n` (`snap`) with a save's position in `base`. The ring holds
    /// eight records, so this waits while eight are unread; a release or a drop has no all-reduce to keep the ranks
    /// together, so the wait is the only brake there.
    pub fn note(g: *Group, op: Op, ids: []const u32, flags: u32, segs: []const Seg, snap: u32) !void {
        if (g.rank != 0) return;
        if (ids.len > 2048 or segs.len > max_segs) return error.Inkling2Tp;
        var rec: Rec = .{ .seq = 0, .op = op, .n = @intCast(ids.len), .flags = flags, .base = 0, .nseg = @intCast(segs.len), .segs = @splat(.{ .slot = 0, .rows = 0, .base = 0, .capacity = 0, .accepted = 0 }), .ids = @splat(0) };
        if (op == .save or op == .restore or op == .drop) rec.n = snap;
        if (segs.len > 0) rec.base = segs[0].base;
        @memcpy(rec.segs[0..segs.len], segs);
        @memcpy(rec.ids[0..ids.len], ids);
        const next = g.rec_seq + 1;
        var deadline = now() + 180.0;
        while (next >= g.doneWord().* + 8) {
            if (now() > deadline) {
                if (peerAlive(g)) {
                    deadline = now() + 180.0;
                    continue;
                }
                std.log.err("inkling2: note saw done {d}, seq {d}", .{ g.doneWord().*, next });
                return error.PeerLost;
            }
            std.atomic.spinLoopHint();
        }
        g.rec_seq = next;
        rec.seq = next;
        const slot = g.recSlot(next);
        @memcpy(slot[0..@sizeOf(Rec)], std.mem.asBytes(&rec));
        g.recWord().* = next;
        std.log.info("inkling2: note seq {d} op {d} base {d} n {d} segs {d}", .{ next, @intFromEnum(op), rec.base, rec.n, rec.nseg });
    }

    /// Rank 1 records the last record it finished, so that rank 0 keeps eight records in the ring and no more.
    pub fn markDone(g: *Group, seq: u64) void {
        if (g.rank == 0) return;
        g.doneWord().* = seq;
    }

    /// Poll the pair socket of the handshake. Any hang-up or error bit means the peer is gone.
    fn peerAlive(g: *Group) bool {
        if (g.conn < 0) return false;
        const rdhup: i16 = 0x2000;
        var fd = [1]linux.pollfd{.{ .fd = g.conn, .events = linux.POLL.IN, .revents = 0 }};
        const rc = linux.poll(&fd, 1, 0);
        if (linux.errno(rc) != .SUCCESS) return false;
        if (rc == 0) return true;
        return (fd[0].revents & (linux.POLL.HUP | rdhup | linux.POLL.ERR)) == 0;
    }

    pub fn wait(g: *Group) !Rec {
        const want = g.rec_seq + 1;
        var deadline = now() + 180.0;
        while (g.recWord().* < want) {
            if (now() > deadline) {
                // An idle pair sends no next record. A live peer gets a new period. A lost peer does not.
                if (peerAlive(g)) {
                    if (!g.idle_logged) {
                        g.idle_logged = true;
                        std.log.info("inkling2: follower idle, peer alive", .{});
                    }
                    deadline = now() + 180.0;
                    continue;
                }
                std.log.err("inkling2: wait saw seq {d}, wanted {d}", .{ g.recWord().*, want });
                return error.PeerLost;
            }
            std.atomic.spinLoopHint();
        }
        g.rec_seq = want;
        var rec: Rec = undefined;
        @memcpy(std.mem.asBytes(&rec), g.recSlot(want)[0..@sizeOf(Rec)]);
        return rec;
    }

    /// Exchange f32 rows, add in f32, store bf16 once. One row fits the slot.
    fn writeFlag(g: *Group, flag: *anyopaque) !void {
        if (g.ex != 3) return;
        const desc: ZexWriteDesc = .{ .writeScope = 4 };
        const t0 = xpu.profile.nowNs();
        try g.r.drv.check(g.zex.?(g.r.list, &desc, flag, g.seq), "zexWrite");
        xpu.profile.noteAppend(1, xpu.profile.nowNs() - t0);
    }

    /// Qwen's check() after a settle: the waits have completed, so the counter events go back to the driver.
    pub fn releaseEvents(g: *Group) void {
        if (g.qgroup != null) qg.dev.reportWaits(&g.qdev, g.rank) catch {};
        for (g.evs[0..g.ev_n]) |e| _ = g.r.drv.api.zeEventDestroy(e);
        g.ev_n = 0;
    }

    fn waitFlag(g: *Group, word: *anyopaque) !void {
        if (g.ex != 3) return g.waitHost();
        const cb = g.cb.?;
        var sync: CbExternalSync = .{ .deviceAddress = @ptrCast(@alignCast(word)), .completionValue = g.seq };
        var desc: CbEventDesc = .{ .flags = 1, .signal = 2, .wait = 2 };
        desc.pNext = &sync;
        var ev: ze.H = null;
        const t0 = xpu.profile.nowNs();
        try g.r.drv.check(cb(g.r.ctx, g.r.device, &desc, &ev), "zeEventCounterBasedCreate");
        if (g.ev_n < g.evs.len) {
            g.evs[g.ev_n] = ev;
            g.ev_n += 1;
        }
        xpu.profile.noteAppend(2, xpu.profile.nowNs() - t0);
        var wa = [1]ze.H{ev};
        const t1 = xpu.profile.nowNs();
        try g.r.drv.check(g.r.drv.api.zeCommandListAppendWaitOnEvents(g.r.list, 1, &wa), "wait");
        xpu.profile.noteAppend(3, xpu.profile.nowNs() - t1);
    }

    fn handshake(g: *Group, name: []const u8) !void {
        const blk = try g.r.allocAlign(blk_bytes, 4096);
        const head = try g.r.alloc(head_bytes);
        g.blk = blk.ptr;
        g.head = head.ptr;
        g.conn = try openSock(name, g.rank);
        const lib: *std.DynLib = @constCast(&g.r.drv.lib);
        const getipc = lib.lookup(GetIpc, "zeMemGetIpcHandle") orelse return error.MissingSymbol;
        const openipc = lib.lookup(OpenIpc, "zeMemOpenIpcHandle") orelse return error.MissingSymbol;
        var handle: IpcHandle = undefined;
        try g.r.drv.check(getipc(g.r.ctx, g.blk.?, &handle), "zeMemGetIpcHandle");
        var fds = [1]i32{std.mem.readInt(i32, handle.data[0..4], .little)};
        try sendFds(g.conn, std.mem.asBytes(&handle), &fds);
        var theirs: IpcHandle = undefined;
        var their_fds = [1]i32{0};
        try recvFds(g.conn, std.mem.asBytes(&theirs), &their_fds);
        std.mem.writeInt(i32, theirs.data[0..4], their_fds[0], .little);
        var peer: ?*anyopaque = null;
        try g.r.drv.check(openipc(g.r.ctx, g.r.device, theirs, 0, &peer), "zeMemOpenIpcHandle");
        g.peer_blk = peer;
        var h2: IpcHandle = undefined;
        try g.r.drv.check(getipc(g.r.ctx, g.head.?, &h2), "zeMemGetIpcHandle");
        var fds2 = [1]i32{std.mem.readInt(i32, h2.data[0..4], .little)};
        try sendFds(g.conn, std.mem.asBytes(&h2), &fds2);
        var t2: IpcHandle = undefined;
        var tf2 = [1]i32{0};
        try recvFds(g.conn, std.mem.asBytes(&t2), &tf2);
        std.mem.writeInt(i32, t2.data[0..4], tf2[0], .little);
        var ph: ?*anyopaque = null;
        try g.r.drv.check(openipc(g.r.ctx, g.r.device, t2, 0, &ph), "zeMemOpenIpcHandle");
        _ = g.rangeOff(g.blk.?);
        const head_off = g.rangeOff(g.head.?);
        var off_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &off_bytes, head_off, .little);
        if (linux.write(g.conn, &off_bytes, 8) != 8) return error.ByteSend;
        off_bytes = @splat(0);
        if (linux.read(g.conn, &off_bytes, 8) != 8) return error.ByteRecv;
        const peer_off = std.mem.readInt(u64, &off_bytes, .little);
        g.peer_head = if (peer_off == 0) ph else off(ph.?, @intCast(peer_off));
        std.log.info("inkling2: head IPC base offset {d}, peer offset {d}", .{ head_off, peer_off });
        g.zex = g.lookupZex();
        g.cb = @constCast(&g.r.drv.lib).lookup(CbCreate, "zeEventCounterBasedCreate");
        const mode: u8 = if (g.zex != null and g.cb != null) 5 else 2;
        var peer_mode: u8 = 0;
        try sendByte(g.conn, mode);
        try recvByte(g.conn, &peer_mode);
        if (mode == 5 and peer_mode == 5) g.ex = 3;
    }

    fn rangeOff(g: *Group, ptr: *anyopaque) u64 {
        const GetRange = *const fn (ze.H, *const anyopaque, *?*anyopaque, *usize) callconv(.c) u32;
        const get = @constCast(&g.r.drv.lib).lookup(GetRange, "zeMemGetAddressRange") orelse {
            std.log.err("inkling2: zeMemGetAddressRange missing", .{});
            return 0;
        };
        var base: ?*anyopaque = null;
        var size: usize = 0;
        if (get(g.r.ctx, ptr, &base, &size) != 0 or base == null) {
            std.log.err("inkling2: address range failed for {*} size {d}", .{ ptr, size });
            return 0;
        }
        const off_b = @intFromPtr(ptr) - @intFromPtr(base.?);
        std.log.info("inkling2: range ptr {*} base {*} size {d} offset {d}", .{ ptr, base.?, size, off_b });
        return @intCast(off_b);
    }

    fn logRange(g: *Group, tag: []const u8, ptr: *anyopaque) void {
        std.log.info("inkling2: {s} is the head-kernel buffer, not the handshake export", .{tag});
        _ = g.rangeOff(ptr);
    }

    fn lookupZex(g: *Group) ?ZexWrite {
        const Get = *const fn (ze.H, [*:0]const u8, *?*anyopaque) callconv(.c) u32;
        const get = @constCast(&g.r.drv.lib).lookup(Get, "zeDriverGetExtensionFunctionAddress") orelse return null;
        var fnp: ?*anyopaque = null;
        if (get(g.ctx.gpu.driver, "zexCommandListAppendWriteToMemory", &fnp) != 0) return null;
        return @ptrCast(fnp);
    }

    fn waitHost(g: *Group) !void {
        g.counter().store(g.seq, .release);
        const deadline = now() + 30.0;
        while (g.peerCounter().load(.acquire) < g.seq) {
            if (now() > deadline) return error.PeerLost;
            std.atomic.spinLoopHint();
        }
    }

    fn recWord(g: *Group) *volatile u64 {
        return @ptrCast(@alignCast(g.map.ptr + 128));
    }
    /// The first 128 bytes of the map held nothing: rank 1 writes the last record it finished here.
    fn doneWord(g: *Group) *volatile u64 {
        return @ptrCast(@alignCast(g.map.ptr));
    }
    fn counter(g: *Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + 128 + @as(usize, g.rank) * 64));
    }
    fn peerCounter(g: *Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + 128 + @as(usize, 1 - g.rank) * 64));
    }
    fn hostSlot(g: *Group, rank: u32, seq: u64) []u8 {
        const at = 256 + (@as(usize, rank) * 2 + @as(usize, @intCast(seq % 2))) * slot_bytes;
        return g.map[at .. at + slot_bytes];
    }
    fn recSlot(g: *Group, seq: u64) []u8 {
        const at = 256 + 4 * slot_bytes + @as(usize, @intCast((seq - 1) % 8)) * @sizeOf(Rec);
        return g.map[at .. at + @sizeOf(Rec)];
    }
};

pub var live: ?*Group = null;

fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}
fn now() f64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(linux.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}
fn shmName() [:0]const u8 {
    return if (std.c.getenv("TF_TP_SHM")) |v| std.mem.span(v) else shm_default;
}
fn gpaBytes(n: usize) ![]u8 {
    return std.heap.c_allocator.alloc(u8, n * 2);
}

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn mapShm(name: [:0]const u8, rank: u32) ![]align(4096) u8 {
    const bytes = 256 + 4 * slot_bytes + 8 * @sizeOf(Rec) + 4096;
    const rdwr: c_int = 2;
    const creat: c_int = 64;
    const excl: c_int = 128;
    var fd: c_int = -1;
    if (rank == 0) {
        _ = linux.unlink(name);
        fd = open(name, rdwr | creat | excl, 0o600);
    } else {
        var n: u32 = 0;
        while (fd < 0) : (n += 1) {
            fd = open(name, rdwr, 0);
            if (fd >= 0) break;
            if (n > 300) return error.ShmOpen;
            var ts: linux.timespec = .{ .sec = 0, .nsec = 100_000_000 };
            _ = linux.nanosleep(&ts, null);
        }
    }
    if (fd < 0) return error.ShmOpen;
    if (rank == 0 and linux.ftruncate(fd, @intCast(bytes)) != 0) return error.ShmOpen;
    const p = c_mmap(null, bytes, 3, 1, fd, 0);
    if (p == @as(*anyopaque, @ptrFromInt(std.math.maxInt(usize)))) return error.ShmOpen;
    _ = linux.close(fd);
    return @as([*]align(4096) u8, @ptrCast(@alignCast(p)))[0..bytes];
}

extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
fn c_mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, file_off: i64) *anyopaque {
    return mmap(addr, len, prot, flags, fd, file_off) orelse @ptrFromInt(std.math.maxInt(usize));
}

fn failed(rc: usize) bool {
    return @as(isize, @bitCast(rc)) < 0;
}
fn sendByte(sock: linux.fd_t, b: u8) !void {
    var v = [1]u8{b};
    if (failed(linux.write(sock, &v, 1))) return error.ByteSend;
}
fn recvByte(sock: linux.fd_t, b: *u8) !void {
    var v = [1]u8{0};
    if (failed(linux.read(sock, &v, 1))) return error.ByteRecv;
    b.* = v[0];
}
const cmsg_space = ((@sizeOf(linux.cmsghdr) + 8 + 7) / 8) * 8;
fn sendFds(sock: linux.fd_t, data: []const u8, fds: []const i32) !void {
    var iov = [1]std.posix.iovec_const{.{ .base = data.ptr, .len = data.len }};
    var cbuf: [cmsg_space]u8 align(8) = @splat(0);
    const h: *linux.cmsghdr = @ptrCast(&cbuf);
    h.* = .{ .len = @sizeOf(linux.cmsghdr) + fds.len * 4, .level = linux.SOL.SOCKET, .type = 1 };
    @memcpy(cbuf[@sizeOf(linux.cmsghdr) .. @sizeOf(linux.cmsghdr) + fds.len * 4], std.mem.sliceAsBytes(fds));
    const msg: linux.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &cbuf, .controllen = cbuf.len, .flags = 0 };
    if (failed(linux.sendmsg(sock, &msg, 0))) return error.FdSend;
}
fn recvFds(sock: linux.fd_t, data: []u8, fds: []i32) !void {
    var iov = [1]std.posix.iovec{.{ .base = data.ptr, .len = data.len }};
    var cbuf: [cmsg_space]u8 align(8) = @splat(0);
    var msg: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &cbuf, .controllen = cbuf.len, .flags = 0 };
    if (failed(linux.recvmsg(sock, &msg, linux.MSG.WAITALL))) return error.FdRecv;
    @memcpy(std.mem.sliceAsBytes(fds), cbuf[@sizeOf(linux.cmsghdr) .. @sizeOf(linux.cmsghdr) + fds.len * 4]);
}
fn openSock(name: []const u8, rank: u32) !linux.fd_t {
    var addr: linux.sockaddr.un = .{ .path = @splat(0) };
    const path = name[0..@min(name.len, addr.path.len - 6)];
    @memcpy(addr.path[0..path.len], path);
    @memcpy(addr.path[path.len .. path.len + 5], ".sock");
    const fd = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM, 0);
    if (failed(fd)) return error.Socket;
    const s: linux.fd_t = @intCast(fd);
    if (rank == 0) {
        _ = linux.unlink(@ptrCast(&addr.path));
        if (failed(linux.bind(s, @ptrCast(&addr), @sizeOf(linux.sockaddr.un)))) return error.Socket;
        if (failed(linux.listen(s, 1))) return error.Socket;
        const c = linux.accept(s, null, null);
        if (failed(c)) return error.Socket;
        return @intCast(c);
    }
    var n: u32 = 0;
    while (failed(linux.connect(s, @ptrCast(&addr), @sizeOf(linux.sockaddr.un)))) {
        n += 1;
        if (n > 3000) return error.Socket;
        var ts: linux.timespec = .{ .sec = 0, .nsec = 10_000_000 };
        _ = linux.nanosleep(&ts, null);
    }
    return s;
}

fn recOf(op: Op, flags: u32, n: u32, segs: []const Seg) Rec {
    var r: Rec = .{ .seq = 1, .op = op, .n = n, .flags = flags, .base = 0, .nseg = @intCast(segs.len), .segs = @splat(.{ .slot = 0, .rows = 0, .base = 0, .capacity = 0, .accepted = 0 }), .ids = @splat(0) };
    @memcpy(r.segs[0..segs.len], segs);
    if (segs.len > 0) r.base = segs[0].base;
    return r;
}

test "the follower's plan: a cache a segment, a keep, a release, a save and a restore" {
    const t = std.testing;
    var out: [40]Action = undefined;
    // Four streams of two rows each: a cache and a transaction a segment, then one shared forward.
    const four = [_]Seg{
        .{ .slot = 0, .rows = 2, .base = 7, .capacity = 64 },
        .{ .slot = 1, .rows = 2, .base = 9, .capacity = 64 },
        .{ .slot = 2, .rows = 2, .base = 4, .capacity = 64 },
        .{ .slot = 3, .rows = 2, .base = 11, .capacity = 64 },
    };
    const shared = try plan(recOf(.forward, 1 | 8, 8, four[0..]), &out);
    try t.expectEqual(@as(usize, 13), shared.len);
    for (four, 0..) |seg, i| {
        try t.expectEqual(seg.slot, shared[i * 3].ensure.slot);
        try t.expectEqual(seg.capacity, shared[i * 3].ensure.capacity);
        try t.expectEqual(seg.base, shared[i * 3 + 1].seek.base);
        try t.expectEqual(seg.rows, shared[i * 3 + 2].begin.rows);
    }
    // Bit 1 without bit 3 is the warm-up forward: it opens no transaction, so its plan is a cache, a seek and the run.
    const warm = try plan(recOf(.forward, 1, 8, four[0..]), &out);
    try t.expectEqual(@as(usize, 9), warm.len);
    for (warm, 0..) |a, i| try t.expect(i % 3 != 2 or a != .begin);
    try t.expect(shared[12] == .run_forward);
    // A prompt chunk at position 0 zeroes that stream's convolution states before it runs.
    const chunk = try plan(recOf(.forward, 4, 8, &.{.{ .slot = 5, .rows = 8, .base = 0, .capacity = 64 }}), &out);
    try t.expectEqual(@as(usize, 4), chunk.len);
    try t.expectEqual(@as(u32, 5), chunk[0].ensure.slot);
    try t.expectEqual(@as(u32, 5), chunk[1].zero_conv.slot);
    try t.expectEqual(@as(u32, 0), chunk[2].seek.base);
    try t.expect(chunk[3] == .run_prefill);
    // A keep names the rows it accepts; the follower commits them, not the next base.
    const kept = try plan(recOf(.keep, 0, 0, &.{.{ .slot = 1, .rows = 3, .base = 20, .capacity = 64, .accepted = 1 }}), &out);
    try t.expectEqual(@as(usize, 1), kept.len);
    try t.expectEqual(@as(u32, 1), kept[0].keep.slot);
    try t.expectEqual(@as(u32, 1), kept[0].keep.accepted);
    // A release while a transaction is open: the follower rolls that cache back, then frees it.
    const freed = try plan(recOf(.release, 0, 0, &.{.{ .slot = 2, .capacity = 64 }}), &out);
    try t.expectEqual(@as(usize, 1), freed.len);
    try t.expectEqual(@as(u32, 2), freed[0].release.slot);
    // A save carries its slot, capacity, snapshot id and position; a restore makes the cache and then restores it.
    const saved = try plan(recOf(.save, 0, 7, &.{.{ .slot = 2, .capacity = 500, .base = 100 }}), &out);
    try t.expectEqual(@as(usize, 1), saved.len);
    try t.expectEqual(@as(u32, 7), saved[0].save.snap);
    try t.expectEqual(@as(u32, 100), saved[0].save.at);
    try t.expectEqual(@as(u32, 500), saved[0].save.capacity);
    const restored = try plan(recOf(.restore, 0, 7, &.{.{ .slot = 2, .capacity = 500 }}), &out);
    try t.expectEqual(@as(usize, 2), restored.len);
    try t.expectEqual(@as(u32, 500), restored[0].ensure.capacity);
    try t.expectEqual(@as(u32, 7), restored[1].restore.snap);
    const dropped = try plan(recOf(.drop, 0, 7, &.{}), &out);
    try t.expectEqual(@as(usize, 1), dropped.len);
    try t.expectEqual(@as(u32, 7), dropped[0].drop.snap);
    const closed = try plan(recOf(.close, 0, 0, &.{}), &out);
    try t.expect(closed.len == 1 and closed[0] == .close);
    // A forward whose table does not name every row is refused.
    try t.expectError(error.Inkling2RecordShape, plan(recOf(.forward, 1, 9, four[0..]), &out));
    try t.expectError(error.Inkling2RecordShape, plan(recOf(.forward, 1, 0, &.{}), &out));
}

/// A host follower: the plan's cache actions applied to real `state.Cache` values. Only the fields the length and
/// transaction rules use are set, so the test needs no device. Device work (a copy, a kernel) is the mark it leaves.
const Host = struct {
    cache: [4]st.Cache = undefined,
    made: [4]bool = .{ false, false, false, false },
    snap: [4]snapshot.Snap = undefined,
    held: [4]bool = .{ false, false, false, false },

    fn at(self: *Host, slot: u32) !*st.Cache {
        if (slot > 3 or !self.made[slot]) return error.NoCache;
        return &self.cache[slot];
    }

    /// One record through the follower: plan it, then each action on the host caches.
    fn feed(self: *Host, rec: Rec) !void {
        var acts: [40]Action = undefined;
        for (try plan(rec, &acts)) |a| switch (a) {
            .close => {},
            .ensure => |e| {
                if (e.slot > 3) return error.NoCache;
                if (!self.made[e.slot]) {
                    self.cache[e.slot] = .{ .slots = undefined, .conv = undefined, .taps = undefined, .backup = undefined, .logits = undefined, .kv = .bf16, .capacity = e.capacity, .slot = e.slot, .r = undefined };
                    self.made[e.slot] = true;
                }
            },
            .seek => |e| (try self.at(e.slot)).len = e.base,
            .zero_conv => {},
            // `begin` copies the convolution states on the device; on the host the transaction is its mark.
            .begin => |e| (try self.at(e.slot)).pending = e.rows,
            .run_forward, .run_prefill => {},
            .keep => |e| try (try self.at(e.slot)).commit(e.accepted),
            .release => |e| if (self.made[e.slot]) {
                self.cache[e.slot].pending = null;
                self.made[e.slot] = false;
            },
            .save => |e| {
                const c = try self.at(e.slot);
                try c.saveAt(e.at);
                self.snap[e.snap % 4] = .{ .buf = undefined, .at = c.len, .bytes = snapshot.bytes(c.len, c.kv), .kv = c.kv, .id = e.snap };
                self.held[e.snap % 4] = true;
            },
            .restore => |e| {
                const c = try self.at(e.slot);
                if (!self.held[e.snap % 4]) return error.NoSnap;
                try snapshot.accept(c, &self.snap[e.snap % 4]);
                c.len = self.snap[e.snap % 4].at;
            },
            .drop => |e| self.held[e.snap % 4] = false,
        };
    }
};

test "finding 1: the follower opens a transaction on a verify window of one row" {
    const t = std.testing;
    var h: Host = .{};
    // Rank 0's verify opens a transaction on every segment, a one-row window too, then keeps the accepted row.
    try h.feed(recOf(.forward, 1 | 8, 1, &.{.{ .slot = 0, .rows = 1, .base = 0, .capacity = 64 }}));
    try t.expectEqual(@as(usize, 1), h.cache[0].pending orelse 0);
    try h.feed(recOf(.keep, 0, 0, &.{.{ .slot = 0, .rows = 1, .base = 0, .capacity = 64, .accepted = 1 }}));
    try t.expectEqual(@as(usize, 1), h.cache[0].len);
}

test "finding 1: a round of two segments begins each one, the one-row segment included" {
    const t = std.testing;
    var h: Host = .{};
    const segs = [_]Seg{
        .{ .slot = 0, .rows = 1, .base = 0, .capacity = 64 },
        .{ .slot = 1, .rows = 2, .base = 0, .capacity = 64 },
    };
    try h.feed(recOf(.forward, 1 | 8, 3, segs[0..]));
    try t.expectEqual(@as(usize, 1), h.cache[0].pending orelse 0);
    try t.expectEqual(@as(usize, 2), h.cache[1].pending orelse 0);
    try h.feed(recOf(.keep, 0, 0, &.{
        .{ .slot = 0, .rows = 1, .base = 0, .capacity = 64, .accepted = 1 },
        .{ .slot = 1, .rows = 2, .base = 0, .capacity = 64, .accepted = 2 },
    }));
    try t.expectEqual(@as(usize, 1), h.cache[0].len);
    try t.expectEqual(@as(usize, 2), h.cache[1].len);
}

test "finding 1: the production shape, a lone round with no drafts at TP 2" {
    const t = std.testing;
    var h: Host = .{};
    // A prompt fills the slot, then a lone round verifies its one pending token and keeps it.
    try h.feed(recOf(.forward, 4, 8, &.{.{ .slot = 0, .rows = 8, .base = 0, .capacity = 4096 }}));
    try h.feed(recOf(.forward, 4, 2048, &.{.{ .slot = 0, .rows = 2048, .base = 8, .capacity = 4096 }}));
    try h.feed(recOf(.forward, 1 | 8, 1, &.{.{ .slot = 0, .rows = 1, .base = 2056, .capacity = 4096 }}));
    try h.feed(recOf(.keep, 0, 0, &.{.{ .slot = 0, .rows = 1, .base = 2056, .capacity = 4096, .accepted = 1 }}));
    try t.expectEqual(@as(usize, 2057), h.cache[0].len);
}

test "finding 2: a follower's save covers the rows past its last chunk base" {
    const t = std.testing;
    var h: Host = .{};
    // A prompt in two chunks of an 8k stream: 2048 rows at 0, then 1898 at 2048.
    try h.feed(recOf(.forward, 4, 2048, &.{.{ .slot = 0, .rows = 2048, .base = 0, .capacity = 8192 }}));
    try h.feed(recOf(.forward, 4, 1898, &.{.{ .slot = 0, .rows = 1898, .base = 2048, .capacity = 8192 }}));
    try t.expectEqual(@as(usize, 2048), h.cache[0].len);
    // The leader saves at 3946: the follower's snapshot must carry all 3946 rows, not the last chunk base.
    try h.feed(recOf(.save, 0, 7, &.{.{ .slot = 0, .rows = 0, .base = 3946, .capacity = 8192 }}));
    try t.expectEqual(@as(usize, 3946), h.cache[0].len);
    try t.expectEqual(@as(usize, 3946), h.snap[3].at);
    try t.expectEqual(snapshot.bytes(3946, .bf16), h.snap[3].bytes);
    // The rows between the last chunk base and the save position are the rows that were fed.
    try t.expectEqual(@as(usize, 1898), h.snap[3].at - 2048);
    // A restore into a new slot takes that length, so the cached prompt starts at 3946.
    try h.feed(recOf(.restore, 0, 7, &.{.{ .slot = 2, .rows = 0, .base = 0, .capacity = 8192 }}));
    try t.expectEqual(@as(usize, 3946), h.cache[2].len);
}

test "finding 3: a slot a failed prompt left rows in restores the next stream" {
    const t = std.testing;
    var h: Host = .{};
    // A prompt on slot 0 fills 2048 rows and saves them, then its stream ends and the follower frees the slot.
    try h.feed(recOf(.forward, 4, 2048, &.{.{ .slot = 0, .rows = 2048, .base = 0, .capacity = 8192 }}));
    try h.feed(recOf(.save, 0, 1, &.{.{ .slot = 0, .rows = 0, .base = 2048, .capacity = 8192 }}));
    try h.feed(recOf(.release, 0, 0, &.{.{ .slot = 0, .capacity = 8192 }}));
    // Stream A hits the cache on slot 0, then its prompt fails: rank 0 sends no release and the rows stay.
    try h.feed(recOf(.restore, 0, 1, &.{.{ .slot = 0, .rows = 0, .base = 0, .capacity = 8192 }}));
    try t.expectEqual(@as(usize, 2048), h.cache[0].len);
    // Stream B hits the same cache on the same slot: the follower must drop its own rows, not stop.
    try h.feed(recOf(.restore, 0, 1, &.{.{ .slot = 0, .rows = 0, .base = 0, .capacity = 8192 }}));
    try t.expectEqual(@as(usize, 2048), h.cache[0].len);
}

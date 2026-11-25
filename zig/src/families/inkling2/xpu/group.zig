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

const D = place.hidden;
const rows_max: u32 = 64;
const slot_bytes = rows_max * D * 2;
const blk_slots = 2 * slot_bytes;
const blk_flag = blk_slots;
const blk_bytes = blk_flag + 64;
const head_bytes = 100029 * 4;
const shm_default = "/dev/shm/tf-tp-inkling2";
const magic: u64 = 0x494E_4B32_5450_0001; // INK2TP, not the Qwen magic

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

pub const Op = enum(u32) { forward = 1, close = 2 };
pub const Rec = extern struct { seq: u64, op: Op, n: u32, flags: u32, base: u32, ids: [2048]u32 };

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
        g.map = try mapShm(name, p.rank);
        try g.handshake(name);
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

    pub fn note(g: *Group, op: Op, ids: []const u32, flags: u32, base: u32) !void {
        if (g.rank != 0) return;
        if (ids.len > 2048) return error.Inkling2Tp;
        g.rec_seq += 1;
        var rec: Rec = .{ .seq = g.rec_seq, .op = op, .n = @intCast(ids.len), .flags = flags, .base = base, .ids = @splat(0) };
        @memcpy(rec.ids[0..ids.len], ids);
        const slot = g.recSlot(g.rec_seq);
        @memcpy(slot[0..@sizeOf(Rec)], std.mem.asBytes(&rec));
        g.recWord().* = g.rec_seq;
        std.log.info("inkling2: note seq {d} base {d} n {d}", .{ g.rec_seq, base, ids.len });
    }

    pub fn wait(g: *Group) !Rec {
        const want = g.rec_seq + 1;
        const deadline = now() + 180.0;
        while (g.recWord().* < want) {
            if (now() > deadline) {
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

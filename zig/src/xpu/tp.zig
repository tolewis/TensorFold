//! Tensor-parallel ranks on one host: the all-reduce of a partial sum through shared memory.
//!
//! Two processes, one a card (ZE_AFFINITY_MASK 0 and 1), run the same forward over their halves of the weights
//! (qwen27.TP, qwen27.RANK). Twice a layer each holds a partial of the residual delta (D values a row, fp16) and
//! needs the sum: it copies its partial to the host (Dev.copy and settle, 16 us a row of 10 KB on the B70), writes
//! it into its slot of a /dev/shm file, bumps its sequence number, waits for the peer's, adds the peer's partial on
//! the host and copies the sum back. 128 all-reduces a token: 4.5 ms measured at 1 row
//! (lanes/qwen27/TP2-ASSESSMENT.md in Intel-B70-Recipe, 2026-11-01). At TP 1 every call returns at once.
//!
//! The file: TF_TP_SHM names it (default /dev/shm/tf-tp-qwen27). Rank 0 creates and sizes it, rank 1 opens it.
//! Layout: a header of TP sequence counters (u64, cache-line apart), then TP slots of `rows_max` x D fp16, then the
//! follower's consumed counter and a ring of `ring_len` step records.
//!
//! The step records: rank 0 writes one before every state-changing Runner call (follow.zig lists them) and the
//! rank above 0 replays the same call with the same arguments, so the two forwards meet at every all-reduce. A
//! prompt longer than a record continues in `more` records.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
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

    fn counter(g: *const Group, r: u32) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + r * line));
    }
    fn slot(g: *const Group, r: u32) [*]u16 {
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + r * slot_bytes));
    }
    fn consumed(g: *const Group) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + g.tp * slot_bytes));
    }
    fn rec(g: *const Group, i: u64) *Rec {
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + g.tp * slot_bytes + line + (i % ring_len) * @sizeOf(Rec)));
    }

    /// Opens the group this process is in. Null at TP 1 (nothing to reduce).
    pub fn open(d: *Dev) !?Group {
        if (q.TP == 1) return null;
        const name: [*:0]const u8 = if (std.c.getenv("TF_TP_SHM")) |v| v else "/dev/shm/tf-tp-qwen27";
        const size = q.TP * line + q.TP * slot_bytes + line + ring_len * @sizeOf(Rec);
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
        return g;
    }

    /// Rank 0 writes a step record (nothing on another rank): the call `op` with its small arguments and `data`
    /// words. Waits while the ring is full (a follower 64 records behind).
    pub fn record(g: *Group, op: Op, a: u32, b: u32, id: u64, data: []const u32) !void {
        if (g.rank != 0) return;
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

    /// `partial` on the device holds `rows` x D fp16: on return it holds the ranks' sum. Every rank calls this the
    /// same number of times in the same order.
    pub fn allReduce(g: *Group, d: *Dev, partial: *anyopaque, rows: u32) !void {
        if (rows > rows_max) return error.TooManyRows;
        const n: usize = @as(usize, rows) * D;
        try dev.copy(d, @ptrCast(g.mine), partial, n * 2);
        d.settle();
        @memcpy(g.slot(g.rank)[0..n], g.mine[0..n]);
        g.seq += 1;
        g.counter(g.rank).store(g.seq, .seq_cst);
        for (0..g.tp) |r| if (r != g.rank) {
            const c = g.counter(@intCast(r));
            // the peer may still be loading its half at the first call: 180 s, then the peer is lost
            const deadline = q.now() + 180.0;
            var spins: usize = 0;
            while (c.load(.seq_cst) < g.seq) : (spins += 1) {
                if (spins % 4096 == 0 and q.now() > deadline) return error.PeerLost;
                std.atomic.spinLoopHint();
            }
        };
        // the sum in fp32, rounded once: the ranks add in the same order, so both hold the same bits
        for (0..n) |i| {
            var acc: f32 = 0;
            for (0..g.tp) |r| acc += @floatCast(@as(f16, @bitCast(g.slot(@intCast(r))[i])));
            g.sum[i] = @bitCast(@as(f16, @floatCast(acc)));
        }
        try dev.copy(d, partial, @ptrCast(g.sum), n * 2);
        d.settle();
    }
};

test "an fp16 sum rounds once from fp32" {
    const a: f16 = 1.0009765625; // 1 + 2^-10, the fp16 step above 1
    const b: f16 = 0.00048828125; // 2^-11, below the step
    const s: f32 = @as(f32, a) + @as(f32, b);
    try std.testing.expectEqual(@as(f16, 1.0009765625), @as(f16, @floatCast(s)));
    try std.testing.expect(@as(f16, @floatCast(s + b)) > a);
}

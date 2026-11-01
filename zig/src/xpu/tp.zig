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
//! Layout: a header of TP sequence counters (u64, cache-line apart), then TP slots of `rows_max` x D fp16.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const Dev = dev.Dev;
const D = q.D;

const line = 64;
const rows_max = 1024 + 128 + 16; // the widest forward (q.maxRows at XPU_CHUNK 1024) and a decode window
const slot_bytes = rows_max * D * 2;

/// The group this process is in, opened by Runner.init; null at TP 1.
pub var group: ?Group = null;

pub const Group = struct {
    map: []align(std.heap.page_size_min) u8,
    rank: u32,
    tp: u32,
    seq: u64 = 0,
    /// Host-visible staging for this rank's partial and for the sum (Level Zero host memory, so the copies are DMA).
    mine: [*]u16,
    sum: [*]u16,

    fn counter(g: *const Group, r: u32) *std.atomic.Value(u64) {
        return @ptrCast(@alignCast(g.map.ptr + r * line));
    }
    fn slot(g: *const Group, r: u32) [*]u16 {
        return @ptrCast(@alignCast(g.map.ptr + g.tp * line + r * slot_bytes));
    }

    /// Opens the group this process is in. Null at TP 1 (nothing to reduce).
    pub fn open(d: *Dev) !?Group {
        if (q.TP == 1) return null;
        const name = if (std.c.getenv("TF_TP_SHM")) |v| std.mem.span(v) else "/dev/shm/tf-tp-qwen27";
        const size = q.TP * line + q.TP * slot_bytes;
        const f = try std.fs.cwd().createFile(name, .{ .read = true, .truncate = false, .mode = 0o600 });
        defer f.close();
        try f.setEndPos(size);
        const map = try std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, f.handle, 0);
        var g: Group = .{
            .map = map,
            .rank = q.RANK,
            .tp = q.TP,
            .mine = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
            .sum = @ptrCast(@alignCast(try d.g.hostAlloc(slot_bytes))),
        };
        g.counter(g.rank).store(0, .seq_cst);
        return g;
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
            var spins: usize = 0;
            while (c.load(.seq_cst) < g.seq) : (spins += 1) {
                if (spins > 200_000_000) return error.PeerLost; // about 20 s of spinning
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

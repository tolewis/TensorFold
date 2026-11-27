//! Compact retained snapshots of every live KV ring prefix and all convolution states, in device memory.
const std = @import("std");
const xpu = @import("xpu");
const state = @import("state.zig");
const ap = @import("attention_plan.zig");

const rt = xpu.rt;

/// Bytes one slot's K and V take for `live` positions. An int8 K code piece sits in whole 16-position tiles, so the last
/// tile is counted whole: the snapshot carries its padding, and restore writes stale bytes only at or past `at`.
fn slotBytes(live: usize, kv: state.KvDtype) u64 {
    const heads: u64 = 8;
    const d: u64 = ap.head_dim;
    const n: u64 = @intCast(live);
    return switch (kv) {
        .bf16 => n * heads * d * 2 * 2,
        .int8 => heads * (@as(u64, @intCast(ap.kCodeBytes(live))) + n * d + n * 4 * 2),
    };
}

/// Eight global layers retain [0,at); forty sliding layers retain ring prefixes, with independent K/V pieces per visit.
/// An int8 snapshot carries each retained position's fp32 scales next to its values.
pub fn bytes(at: usize, kv: state.KvDtype) u64 {
    const sliding = @min(at, state.sliding_ring);
    return 8 * slotBytes(at, kv) + 40 * slotBytes(sliding, kv) + state.conv_bytes;
}

pub const Snap = struct {
    buf: rt.Buffer,
    at: usize,
    bytes: u64,
    kv: state.KvDtype,
    /// The leader's snapshot id, so the follower saves, restores and drops the same snapshot on its own half.
    id: u32 = 0,
    freed: bool = false,

    pub fn free(s: *Snap) void {
        if (!s.freed) {
            s.buf.free();
            s.freed = true;
        }
    }

    pub fn deinit(s: *Snap) void {
        s.free();
    }
};

const Direction = enum { save, restore };

/// Key-value heads of a visit; the XPU attention kernels keep each head's ring contiguous, [head][ring position][value].
const kv_heads = 8;

/// One queued pass over every slot, K then V, values then scales: the layout `bytes` counts. Inside a piece the live
/// positions of head 0 come first, then head 1, and so on, so the device's head-major rings are copied head by head.
fn encode(cache: *const state.Cache, snap_buf: rt.Buffer, rows: usize, comptime dir: Direction) !void {
    const r = cache.r;
    var at: usize = 0;
    for (0..state.visit_slots) |i| {
        const s = &cache.slots[i];
        const live = if (s.global) rows else @min(rows, s.ring);
        const pieces = [_]struct { off: usize, per: usize }{
            .{ .off = s.key_at, .per = cache.kv.rowBytes() },
            .{ .off = s.key_scale_at, .per = cache.kv.scaleBytes() },
            .{ .off = s.value_at, .per = cache.kv.rowBytes() },
            .{ .off = s.value_scale_at, .per = cache.kv.scaleBytes() },
        };
        for (pieces, 0..) |p, pi| {
            const per_head = p.per / kv_heads;
            // the K codes of an int8 cache are copied in whole tiles: `live` rounded up to 16 positions a head
            const n = if (cache.kv == .int8 and pi == 0) ap.kCodeBytes(live) else live * per_head;
            if (n == 0) continue;
            for (0..kv_heads) |h| {
                const device = state.view(s.buf, p.off + h * @as(usize, s.ring) * per_head, n);
                const packed_ = state.view(snap_buf, at, n);
                if (dir == .save) try r.copy(packed_, device, n) else try r.copy(device, packed_, n);
                at += n;
            }
        }
    }
    const conv = state.view(snap_buf, at, state.conv_bytes);
    if (dir == .save) try r.copy(conv, cache.conv, state.conv_bytes) else try r.copy(cache.conv, conv, state.conv_bytes);
    try r.sync();
}

/// Device snapshots retain only live global and sliding prefixes, never full-capacity allocations.
pub fn save(r: *rt.Runtime, cache: *const state.Cache) !Snap {
    if (cache.pending != null) return error.Inkling2CacheTransaction;
    const size = bytes(cache.len, cache.kv);
    var buf = try r.alloc(@intCast(size));
    errdefer buf.free();
    try encode(cache, buf, cache.len, .save);
    return .{ .buf = buf, .at = cache.len, .bytes = size, .kv = cache.kv };
}

/// The geometry a restore needs, and the follower's own rows it drops first.
pub fn accept(cache: *state.Cache, snap: *const Snap) !void {
    if (snap.freed or cache.pending != null or snap.at > cache.capacity or snap.kv != cache.kv or snap.bytes < bytes(snap.at, snap.kv)) return error.Inkling2SnapshotGeometry;
    // A slot reused after a failed prompt keeps rows the snapshot does not cover: the snapshot rewrites [0, at).
    cache.len = 0;
}

/// Restore into an empty, large-enough cache; global capacity may grow while sliding positions stay unchanged.
pub fn restore(cache: *state.Cache, snap: *const Snap) !void {
    try accept(cache, snap);
    try encode(cache, snap.buf, snap.at, .restore);
    cache.len = snap.at;
}

test "snapshot bytes: bf16 matches the Metal layout, int8 adds its scales and the K tile padding, sliding prefixes stop at the ring" {
    const t = std.testing;
    try t.expectEqual(@as(u64, state.conv_bytes), bytes(0, .bf16));
    try t.expectEqual(@as(u64, (8 * 100 + 40 * 100) * 1024 * 4 + state.conv_bytes), bytes(100, .bf16));
    // 100 positions a slot take 112 in whole K tiles a head, so 48 slots * 8 heads * 12 positions * 128 bytes more
    const pad: u64 = 48 * 8 * (112 - 100) * ap.head_dim;
    try t.expectEqual(@as(u64, (8 * 100 + 40 * 100) * 2 * (1024 + 32) + state.conv_bytes + pad), bytes(100, .int8));
    const far = state.sliding_ring * 10;
    try t.expectEqual(@as(u64, (8 * far + 40 * state.sliding_ring) * 1024 * 4 + state.conv_bytes), bytes(far, .bf16));
    // a ring that is already a whole number of tiles takes no padding
    try t.expectEqual(@as(u64, (8 * 128 + 40 * 128) * 2 * (1024 + 32) + state.conv_bytes), bytes(128, .int8));
}

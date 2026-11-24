//! Forty-eight independent visit caches and the device buffers used by one whole-loop forward, on the XPU backend.
//! Same geometry as `metal/state.zig`; K/V rows are bf16 or int8 with one fp32 scale per position and K/V head.
const std = @import("std");
const xpu = @import("xpu");
const plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");

const rt = xpu.rt;

pub const window_rows = 16;
pub const batch_rows = 32;
pub const prompt_rows = 64;
pub const max_rows = 64;
pub const max_segments = batch_rows;
pub const hidden = 2304;
pub const feature_taps = 4;
pub const feature_row_bytes = feature_taps * hidden * 2;

pub fn captureBytes(rows: usize) usize {
    return rows * feature_row_bytes;
}
pub const query_width = 3072;
pub const kv_width = 1024;
pub const kv_heads = 8;
pub const rel_width = 384;
pub const experts_per_token = 8;
pub const intermediate = 1152;
pub const vocabulary = 200058;
pub const message_entries = 48;
pub const block_slots = 4;
pub const conv_states = 4;
pub const visit_slots = 48;
pub const max_capacity = 1048576;
pub const sliding_ring = ap.ringFor(1024, max_rows);
pub const conv_widths = [conv_states]usize{ 1024, 1024, 2304, 2304 };
pub const conv_row_values = 1024 + 1024 + 2304 + 2304;
pub const conv_bytes = visit_slots * 3 * conv_row_values * 2;
pub const taps_bytes = visit_slots * (3 + max_rows) * conv_row_values * 2;

/// How K and V rows are stored. `int8` keeps one signed byte per value and one fp32 scale per position and K/V head
/// (as the Qwen XPU path's XPU_KV_INT8 does); `bf16` keeps the value as computed.
pub const KvDtype = enum {
    bf16,
    int8,

    /// Bytes of one position's row of values (all 8 K/V heads).
    pub fn rowBytes(kv: KvDtype) usize {
        return switch (kv) {
            .bf16 => kv_width * 2,
            .int8 => kv_width,
        };
    }

    /// Bytes of one position's scales (one fp32 a K/V head); none for bf16.
    pub fn scaleBytes(kv: KvDtype) usize {
        return switch (kv) {
            .bf16 => 0,
            .int8 => kv_heads * 4,
        };
    }

    /// Bytes a position holds in one of K or V.
    pub fn positionBytes(kv: KvDtype) usize {
        return kv.rowBytes() + kv.scaleBytes();
    }
};

pub fn globalSlots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, ap.tile_keys);
}

/// Partial slots a row and head hold in `part_o` and `part_ml`: the most any visit of a stream with a global capacity of
/// `capacity` positions needs, at the chunk each layer kind uses (the global chunk is 256, the sliding chunk 64).
pub fn partialStride(capacity: usize) usize {
    const sliding = ap.slotsFor(ap.Layer.sliding(1024, max_rows), 1024 + max_rows);
    const global = ap.slotsFor(ap.Layer.global(@intCast(capacity)), capacity);
    return @max(sliding, global);
}

/// Bytes of every visit's K and V (values and scales) for a global capacity of `capacity` positions.
pub fn kvBytes(capacity: usize, kv: KvDtype) usize {
    return (40 * sliding_ring + 8 * globalSlots(capacity)) * 2 * kv.positionBytes();
}

pub fn cacheBytes(capacity: usize, kv: KvDtype) usize {
    return kvBytes(capacity, kv) + conv_bytes + taps_bytes + conv_bytes + vocabulary * 4;
}

/// The largest global capacity (at most `max_capacity`) whose cache fits `budget` bytes; zero when none does.
pub fn largestCapacity(budget: u64, kv: KvDtype) usize {
    if (cacheBytes(1, kv) > budget) return 0;
    var lo: usize = 1;
    var hi: usize = max_capacity;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (cacheBytes(mid, kv) <= budget) lo = mid else hi = mid - 1;
    }
    return lo;
}

pub fn convWidth(state: usize) usize {
    std.debug.assert(state < conv_states);
    return conv_widths[state];
}

/// Values before a convolution state's rows in one slot's tightly sized layout.
pub fn convBase(state: usize, tap_rows: usize) usize {
    var out: usize = 0;
    for (0..state) |s| out += tap_rows * conv_widths[s];
    return out;
}

pub fn convByteOffset(slot: usize, state: usize, rows_before: usize, tap_rows: usize) usize {
    std.debug.assert(slot < visit_slots and rows_before <= tap_rows);
    return (slot * tap_rows * conv_row_values + convBase(state, tap_rows) + rows_before * convWidth(state)) * 2;
}

/// A byte offset into a device buffer; the buffer value is borrowed, never freed through a Ref.
pub const Ref = struct {
    buf: rt.Buffer,
    off: usize = 0,

    pub fn ptr(r: Ref) ?*anyopaque {
        return if (r.buf.ptr) |p| @ptrFromInt(@intFromPtr(p) + r.off) else null;
    }
};

/// A borrowed window of `len` bytes into `b` at `off`, for the runtime's offset-free copies.
pub fn view(b: rt.Buffer, off: usize, len: usize) rt.Buffer {
    std.debug.assert(off <= b.len and len <= b.len - off);
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = len, .host = b.host };
}

/// Tap rows cover [start,start+rows); scratch rows carry a generation, unlike the prompt's stable last row.
pub const Tap = struct { buf: ?rt.Buffer = null, off: usize = 0, start: u64 = 0, rows: usize = 0, gen: ?u64 = null };

/// Byte offset of a requested held feature range, rejecting stale and rejected rows.
pub fn featureOffset(t: Tap, start: u64, count: u32, generation: u64) !usize {
    if (t.gen) |g| if (g != generation) return error.StaleInkling2Taps;
    if (count == 0 or start < t.start) return error.Inkling2TapsNotHeld;
    const delta = start - t.start;
    const held: u64 = @intCast(t.rows);
    if (delta >= held or count > held - delta) return error.Inkling2TapsNotHeld;
    return t.off + @as(usize, @intCast(delta)) * feature_row_bytes;
}

/// Queues a fill of the first `bytes` of `b` with zero bytes.
pub fn zero(r: *rt.Runtime, b: rt.Buffer, bytes: usize) !void {
    if (bytes == 0) return;
    const v: u8 = 0;
    try r.drv.check(r.drv.x.zeCommandListAppendMemoryFill(r.list, b.ptr, &v, 1, bytes, null, 0, null), "inkling2 fill");
}

/// One visit slot's K and V in one allocation: [K values][K scales][V values][V scales] (scales only for int8).
pub const Slot = struct {
    buf: rt.Buffer,
    ring: u32,
    global: bool,
    key_at: usize,
    key_scale_at: usize,
    value_at: usize,
    value_scale_at: usize,
};

pub const Cache = struct {
    slots: [visit_slots]Slot,
    conv: rt.Buffer,
    taps: rt.Buffer,
    backup: rt.Buffer,
    logits: rt.Buffer,
    kv: KvDtype,
    tap: Tap = .{},
    last_tap: ?rt.Buffer = null,
    capacity: usize,
    len: usize = 0,
    pending: ?usize = null,
    r: *rt.Runtime,

    pub fn init(r: *rt.Runtime, capacity: usize, schedule: *const plan.Plan, kv: KvDtype) !Cache {
        if (capacity == 0 or capacity > max_capacity or schedule.n_visits != visit_slots) return error.Inkling2CacheShape;
        var out: Cache = .{ .slots = undefined, .conv = undefined, .taps = undefined, .backup = undefined, .logits = undefined, .kv = kv, .capacity = capacity, .r = r };
        var made: usize = 0;
        errdefer for (out.slots[0..made]) |*s| s.buf.free();
        var total: usize = 0;
        for (0..visit_slots) |i| {
            const global = schedule.at(schedule.visit(@intCast(i)).layer).kind == .global;
            const positions: usize = if (global) globalSlots(capacity) else sliding_ring;
            const val_bytes = positions * kv.rowBytes();
            const scale_bytes = positions * kv.scaleBytes();
            const s = &out.slots[i];
            s.* = .{ .buf = try r.alloc(@max(2 * (val_bytes + scale_bytes), 16)), .ring = @intCast(positions), .global = global, .key_at = 0, .key_scale_at = val_bytes, .value_at = val_bytes + scale_bytes, .value_scale_at = 2 * val_bytes + scale_bytes };
            made += 1;
            total += 2 * (val_bytes + scale_bytes);
        }
        std.debug.assert(total == kvBytes(capacity, kv));
        out.conv = try r.alloc(conv_bytes);
        errdefer out.conv.free();
        out.taps = try r.alloc(taps_bytes);
        errdefer out.taps.free();
        out.backup = try r.alloc(conv_bytes);
        errdefer out.backup.free();
        out.logits = try r.alloc(vocabulary * 4);
        errdefer out.logits.free();
        try zero(r, out.conv, conv_bytes);
        try r.sync();
        return out;
    }

    pub fn ring(c: *const Cache, slot: usize) usize {
        return c.slots[slot].ring;
    }

    pub fn keys(c: *const Cache, slot: usize) Ref {
        return .{ .buf = c.slots[slot].buf, .off = c.slots[slot].key_at };
    }

    pub fn values(c: *const Cache, slot: usize) Ref {
        return .{ .buf = c.slots[slot].buf, .off = c.slots[slot].value_at };
    }

    /// Per-position, per-K/V-head fp32 scales of the slot's keys; meaningful for int8 only.
    pub fn keyScales(c: *const Cache, slot: usize) Ref {
        return .{ .buf = c.slots[slot].buf, .off = c.slots[slot].key_scale_at };
    }

    pub fn valueScales(c: *const Cache, slot: usize) Ref {
        return .{ .buf = c.slots[slot].buf, .off = c.slots[slot].value_scale_at };
    }

    pub fn history(c: *const Cache, slot: usize, state: usize) Ref {
        return .{ .buf = c.conv, .off = convByteOffset(slot, state, 0, 3) };
    }

    pub fn staged(c: *const Cache, slot: usize, state: usize, rows_before: usize) Ref {
        return .{ .buf = c.taps, .off = convByteOffset(slot, state, rows_before, 3 + max_rows) };
    }

    pub fn attentionLayer(c: *const Cache, schedule: *const plan.Plan, slot: usize, rows: usize) ap.Layer {
        std.debug.assert(rows <= max_rows);
        var layer = if (schedule.at(schedule.visit(@intCast(slot)).layer).kind == .global) ap.Layer.global(@intCast(c.capacity)) else ap.Layer.sliding(1024, @intCast(rows));
        layer.ring = @intCast(c.ring(slot));
        return layer;
    }

    /// Save all four pre-window three-tap histories before KV/conv speculation.
    pub fn begin(c: *Cache, rows: usize) !void {
        if (c.pending != null or rows == 0 or rows > window_rows or c.len + rows > c.capacity) return error.Inkling2CacheTransaction;
        try c.r.copy(c.backup, c.conv, conv_bytes);
        try c.r.sync();
        c.pending = rows;
    }

    /// Commit accepted rows: staged taps select the new state; rejected KV positions are overwritten on replay.
    pub fn commit(c: *Cache, accepted: usize) !void {
        const rows = c.pending orelse return error.Inkling2NoTransaction;
        if (accepted > rows) return error.Inkling2CacheTransaction;
        if (accepted < rows) {
            try c.r.copy(c.conv, c.backup, conv_bytes);
            for (0..visit_slots) |slot| for (0..conv_states) |state| {
                const bytes = 3 * convWidth(state) * 2;
                const from = convByteOffset(slot, state, accepted, 3 + max_rows);
                const to = convByteOffset(slot, state, 0, 3);
                try c.r.copy(view(c.conv, to, bytes), view(c.taps, from, bytes), bytes);
            };
            try c.r.sync();
        }
        const start = c.len;
        c.len += accepted;
        if (c.tap.buf != null and c.tap.start == start) {
            c.tap.rows = accepted;
        } else c.tap = .{};
        c.pending = null;
    }

    pub fn rollback(c: *Cache) !void {
        _ = c.pending orelse return error.Inkling2NoTransaction;
        try c.r.copy(c.conv, c.backup, conv_bytes);
        try c.r.sync();
        c.pending = null;
        c.tap = .{};
    }

    /// The last `logits` the forward wrote, read back to the host.
    pub fn readLogits(c: *Cache, out: []f32) !void {
        if (out.len > vocabulary) return error.Inkling2CacheShape;
        try c.r.download(std.mem.sliceAsBytes(out), c.logits);
        try c.r.sync();
    }

    pub fn deinit(c: *Cache) void {
        c.logits.free();
        if (c.last_tap) |*b| b.free();
        c.backup.free();
        c.taps.free();
        c.conv.free();
        for (&c.slots) |*s| s.buf.free();
        c.* = undefined;
    }
};

pub const fields = [_][]const u8{
    "ids",      "pos",        "seg_of",     "segs",      "p",          "prelude",   "blocks",        "bank0",    "bank1",   "messages",
    "partial",  "ar",         "gated",      "mlp_ar",    "mlp_gated",  "q_norm",    "k_norm",        "v_norm",   "q_raw",   "k_raw",
    "v_raw",    "rel_raw",    "sink",       "gate",      "k_conv",     "v_conv",    "q_prep",        "rel_prep", "rel_tab", "part_o",
    "part_ml",  "att",        "attn_dense", "attn_conv", "mlp_normed", "route_ids", "route_weights", "gu",       "down",    "mlp_out",
    "mlp_conv", "logits_all", "members",    "items",     "count",     "zeta",
};

fn sizes(rows: usize, capacity: usize) [fields.len]usize {
    const r = rows;
    const h = hidden;
    const q = query_width;
    const kv = kv_width;
    const bf = 2;
    const fp = 4;
    const stride = partialStride(capacity);
    const items = @min(r * experts_per_token, 128 + r * experts_per_token / 32);
    return .{
        r * 4,                     r * 4,                              r * 4,                                     max_segments * @sizeOf(ap.Seg), r * h * bf,     r * h * bf,     r * block_slots * h * bf,
        r * h * bf,                r * h * bf,                         r * message_entries * h * fp,              r * h * fp,                     3 * r * h * fp, 3 * r * h * fp, r * h * fp,
        r * h * fp,                r * h * bf,                         r * h * bf,                                r * h * bf,                     r * q * bf,     r * kv * bf,    r * kv * bf,
        r * rel_width * bf,        r * 24 * bf,                        r * q * bf,                                r * kv * bf,                    r * kv * bf,    r * q * bf,     r * rel_width * bf,
        r * 24 * ap.extent * bf,   r * 24 * stride * ap.head_dim * fp, r * 24 * stride * 2 * fp,                  r * q * bf,                     r * h * bf,     r * h * bf,     r * h * bf,
        r * experts_per_token * 4, r * experts_per_token * fp,         r * experts_per_token * intermediate * bf, r * experts_per_token * h * bf, r * h * bf,     r * h * bf,     r * vocabulary * fp,
        r * experts_per_token * 4, items * 3 * 4,                      4,                                         r * 64 * fp,
    };
}

pub fn scratchBytes(rows: usize, capacity: usize) usize {
    var out: usize = 0;
    for (sizes(rows, capacity)) |n| out += @max(n, 16);
    return out;
}

pub const Scratch = struct {
    ids: rt.Buffer,
    pos: rt.Buffer,
    seg_of: rt.Buffer,
    segs: rt.Buffer,
    p: rt.Buffer,
    prelude: rt.Buffer,
    blocks: rt.Buffer,
    bank0: rt.Buffer,
    bank1: rt.Buffer,
    messages: rt.Buffer,
    partial: rt.Buffer,
    ar: rt.Buffer,
    gated: rt.Buffer,
    mlp_ar: rt.Buffer,
    mlp_gated: rt.Buffer,
    q_norm: rt.Buffer,
    k_norm: rt.Buffer,
    v_norm: rt.Buffer,
    q_raw: rt.Buffer,
    k_raw: rt.Buffer,
    v_raw: rt.Buffer,
    rel_raw: rt.Buffer,
    sink: rt.Buffer,
    gate: rt.Buffer,
    k_conv: rt.Buffer,
    v_conv: rt.Buffer,
    q_prep: rt.Buffer,
    rel_prep: rt.Buffer,
    rel_tab: rt.Buffer,
    part_o: rt.Buffer,
    part_ml: rt.Buffer,
    att: rt.Buffer,
    attn_dense: rt.Buffer,
    attn_conv: rt.Buffer,
    mlp_normed: rt.Buffer,
    route_ids: rt.Buffer,
    route_weights: rt.Buffer,
    gu: rt.Buffer,
    down: rt.Buffer,
    mlp_out: rt.Buffer,
    mlp_conv: rt.Buffer,
    logits_all: rt.Buffer,
    /// The prompt experts' grouping of the routes: members by expert, items (expert, first member, count), item count.
    members: rt.Buffer,
    items: rt.Buffer,
    count: rt.Buffer,
    /// The message gates' D projection [row][64] binary32, written by one launch and read by the next.
    zeta: rt.Buffer,
    /// Opt-in DSpark features [row][tap][H], allocated by prepare_features.
    taps: ?rt.Buffer = null,
    rows: usize,
    /// The largest global capacity the scratch was sized for, and the partial slots a row and head hold from it.
    capacity: usize,
    stride: u32,

    /// `capacity` is the largest global capacity of any cache the scratch serves; it sizes the attention partials.
    pub fn init(r: *rt.Runtime, rows: usize, capacity: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        if (rows == 0 or rows > max_rows or capacity == 0 or capacity > max_capacity) return error.Inkling2ScratchRows;
        const ns = sizes(rows, capacity);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).free();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try r.alloc(@max(ns[i], 16));
            made += 1;
        }
        s.taps = null;
        s.rows = rows;
        s.capacity = capacity;
        s.stride = @intCast(partialStride(capacity));
        // a fresh allocation holds whatever the driver last left there; the forward reads only what it wrote, but zero is the safe start
        inline for (fields) |f| try zero(r, @field(s, f), @field(s, f).len);
        try r.sync();
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).free();
        if (s.taps) |*b| b.free();
        s.* = undefined;
    }
};

comptime {
    if (sliding_ring != 1088 or conv_bytes != 1916928) @compileError("Inkling cache geometry drift");
}

test "capacity math: bytes a context costs and the largest one a budget holds" {
    const t = std.testing;
    // 8 global visits cost 32 KiB a position in bf16 and 16.25 KiB in int8 (scales included)
    try t.expectEqual(@as(usize, 64 * 8 * 2 * 2048), kvBytes(64, .bf16) - kvBytes(0, .bf16));
    try t.expectEqual(@as(usize, 64 * 8 * 2 * (1024 + 32)), kvBytes(64, .int8) - kvBytes(0, .int8));
    // 40 sliding rings do not grow with the context
    try t.expectEqual(kvBytes(64, .int8), kvBytes(1, .int8));
    try t.expect(cacheBytes(1000, .int8) < cacheBytes(1000, .bf16));
    for ([_]KvDtype{ .bf16, .int8 }) |kv| {
        const budget: u64 = 6 << 30;
        const best = largestCapacity(budget, kv);
        try t.expect(best > 0 and best < max_capacity);
        try t.expect(cacheBytes(best, kv) <= budget);
        try t.expect(cacheBytes(best + 64, kv) > budget);
        try t.expectEqual(@as(usize, 0), largestCapacity(1 << 20, kv));
        try t.expectEqual(@as(usize, max_capacity), largestCapacity(1 << 40, kv));
    }
}

test "conv offsets tile one slot exactly" {
    try std.testing.expectEqual(conv_bytes, convByteOffset(visit_slots - 1, conv_states - 1, 3, 3) + 0 + 0);
}

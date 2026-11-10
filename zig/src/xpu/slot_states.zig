//! A slot's states: allocation on first use, and the prompt cache's snapshots (save, restore, drop).
//! With the shared K/V pool (XPU_KV_POOL) a snapshot keeps the slot's pages (their ids, one reference each) and copies
//! only the GDN and convolution states; a restore maps the pages back, so a resume costs a page-table copy.
//! A parked snapshot (XPU_PARK_GIB) holds no pool page. It holds one host buffer of the K and V pages of positions
//! 0 to `at`, and the same GDN copy.
const std = @import("std");
const q = @import("qwen27.zig");
const tp = @import("tp.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const park_runs = @import("park_runs.zig");
const Runner = q.Runner;
const copy = dev.copy;
const kvBytes = fwd.kvBytes;
const off = dev.off;
const alloc = q.alloc;

extern "c" fn aligned_alloc(alignment: usize, size: usize) ?*anyopaque;
extern "c" fn free(p: ?*anyopaque) void;

/// Bytes of the GDN and convolution states a slot takes whatever its context: each GDN layer's states and convolution.
pub fn gdnBytes(r: *const Runner) usize {
    var layers: usize = 0;
    for (r.model.layers) |l| layers += @intFromBool(!l.attn);
    return layers * (2 * q.GDN_V_HEADS * 128 * 128 * 4 + 2 * 3 * q.GDN_QKV * 2);
}

/// Bytes a kept state's GDN and convolution copy takes: each GDN layer's recurrent state and current convolution (what
/// copyGdnOut writes; gdnBytes counts a slot's double buffers). The host copy's size with XPU_KEPT_HOST=1.
pub fn keptGdnBytes(r: *const Runner) usize {
    var layers: usize = 0;
    for (r.model.layers) |l| layers += @intFromBool(!l.attn);
    return layers * (REC_BYTES + CONV_BYTES);
}

/// Bytes a slot's states take: GDN and convolution states, and the K/V cache of `context` positions (none with the pool).
pub fn slotBytes(r: *const Runner) usize {
    return gdnBytes(r) + if (q.kv_pool) 0 else (q.LAYERS / 4) * 2 * kvBytes();
}

/// A slot's state after `at` prompt tokens (prompt cache): GDN and convolution states, K/V rows 0 .. at, one buffer.
/// With the pool: the GDN and convolution states only, and `pages`, the slot's page ids from entry 0 to ceil(at / PAGE).
/// Parked: `pages` is empty. `park` holds the K and V of positions 0 .. at, in position order, one host buffer.
/// The GDN copy stays in `buf`.
pub const Snap = struct {
    at: u32,
    buf: *anyopaque,
    bytes: usize,
    pages: []u32 = &.{},
    park: ?*anyopaque = null,
    park_bytes: usize = 0,
    park_pinned: bool = false,
};

pub const REC_BYTES = q.GDN_V_HEADS * 128 * 128 * 4;
pub const CONV_BYTES = 3 * q.GDN_QKV * 2;
/// The K/V cache layout of this rank: its own K/V heads (half at TP 2), as the kernels index it (Q27_KVROW).
pub fn kv() q.limits.KvLayout {
    return .{ .heads = q.ATT_KV_HEADS, .cap = q.MAX_CTX, .int8 = q.kv_i8 };
}

/// The bytes saveSlot takes at `at` tokens; with the pool, the pages the state holds are charged too.
pub fn snapBytes(r: *const Runner, at: u32) usize {
    // XPU_KEPT_HOST=1: the GDN copy is host memory, so a state charges the card only its pool pages
    if (q.kv_pool and q.kept_host) return @as(usize, at) * r.kvBytesPerToken();
    if (q.kv_pool) return gdnBytes(r) + @as(usize, at) * r.kvBytesPerToken();
    var n: usize = 0;
    for (&r.model.layers) |*l| n += if (l.attn) 2 * kv().snapBytes(at) else REC_BYTES + CONV_BYTES;
    return n;
}

/// Copies `slot`'s state standing at `at` tokens into new device memory (with the pool: the GDN states, and the pages
/// the state holds stay where they are).
pub fn saveSlot(r: *Runner, slot: u32, at: u32) !Snap {
    const snap = try saveSlotHere(r, slot, at);
    // recorded after the save: the follower saves its own half under the buffer's address as the id
    if (tp.group) |*g| try g.record(.save_slot, slot, at, @intFromPtr(snap.buf), &.{});
    return snap;
}

fn saveSlotHere(r: *Runner, slot: u32, at: u32) !Snap {
    try r.select(slot);
    if (r.pos != at or r.multi_count != 0) return error.SlotNotThere;
    const d = &r.d;
    const bytes = r.snapBytes(at);
    if (q.kv_pool) {
        if (at % q.PAGE != 0) return error.SlotNotThere; // the last page is partial: a restore maps it, and two streams write into it
        // XPU_KEPT_HOST=1: pinned host memory, the copy goes over PCIe and takes no card memory
        const buf = if (q.kept_host) try d.hostAlloc(keptGdnBytes(r)) else try d.alloc_(gdnBytes(r));
        errdefer freeAfterQueue(d, buf); // copies into it may be queued
        try copyGdnOut(d, r, buf);
        const entries = (at + q.PAGE - 1) / q.PAGE;
        if (entries > r.held[slot]) return error.SlotNotThere;
        const pages = try alloc.alloc(u32, @as(usize, entries));
        errdefer alloc.free(pages);
        @memcpy(pages, r.pt_host[slot][0..entries]);
        for (pages) |id| r.keepPage(id); // the state holds them until dropSnap; the kept count runs with the flag on or off
        return .{ .at = at, .buf = buf, .bytes = bytes, .pages = pages };
    }
    const buf = try d.alloc_(bytes);
    errdefer freeAfterQueue(d, buf); // copies into it may be queued
    const lay = kv();
    var o: usize = 0;
    for (&r.model.layers) |*l| {
        if (l.attn) {
            for ([_]*anyopaque{ l.kc.?, l.vc.? }) |c| {
                try copy(d, off(buf, o), c, @as(usize, at) * lay.row());
                o += @as(usize, at) * lay.row();
                if (q.kv_i8) { // the scales after the cap's rows (cap * Q27_KVROW: this rank's heads, not the model's)
                    try copy(d, off(buf, o), off(c, lay.scales()), @as(usize, at) * lay.scaleRow());
                    o += @as(usize, at) * lay.scaleRow();
                }
            }
            continue;
        }
        try copy(d, off(buf, o), l.rec.?, REC_BYTES);
        try copy(d, off(buf, o + REC_BYTES), l.conv[l.conv_cur].?, CONV_BYTES);
        o += REC_BYTES + CONV_BYTES;
    }
    return .{ .at = at, .buf = buf, .bytes = bytes };
}

/// A slot's GDN and convolution states into `snap` (a snapshot's copy): the current slot's, after `select`.
fn copyGdnOut(d: *dev.Dev, r: *Runner, snap: *anyopaque) !void {
    var o: usize = 0;
    for (&r.model.layers) |*l| {
        if (l.attn) continue;
        try copy(d, off(snap, o), l.rec.?, REC_BYTES);
        try copy(d, off(snap, o + REC_BYTES), l.conv[l.conv_cur].?, CONV_BYTES);
        o += REC_BYTES + CONV_BYTES;
    }
}

/// The GDN and convolution states of a snapshot's copy back into the current slot (after `select`).
fn copyGdnIn(d: *dev.Dev, r: *Runner, snap: *anyopaque) !void {
    var o: usize = 0;
    for (&r.model.layers) |*l| {
        if (l.attn) continue;
        try copy(d, l.rec.?, off(snap, o), REC_BYTES);
        try copy(d, l.conv[l.conv_cur].?, off(snap, o + REC_BYTES), CONV_BYTES);
        o += REC_BYTES + CONV_BYTES;
    }
}

/// Copies `snap`'s K and V pages to one host buffer and drops its pool pages. The GDN copy stays in `snap.buf`.
/// On any error the snap is unchanged. Rank 0 records `park_snap` only after the copy has landed. Every follower
/// then parks its own share. The id is `snap.buf`, as `drop_snap`. A follower does not refuse for the tier budget.
pub fn parkSnap(r: *Runner, snap: *Snap) !void {
    if (snap.park != null) return;
    if (!q.kv_pool or r.page_bytes == 0 or snap.at % q.PAGE != 0) return error.NotParkable;
    const entries: usize = snap.pages.len;
    const expect: usize = (snap.at + q.PAGE - 1) / q.PAGE;
    if (entries == 0 or entries != expect) return error.NotParkable;
    const bytes = @as(usize, snap.at) * r.kvBytesPerToken();
    const d = &r.d;
    const host = try parkAlloc(d, bytes);
    var own = true;
    errdefer if (own) {
        d.sync() catch {};
        parkFree(d, host.ptr, host.pinned);
    };
    const t0 = q.now();
    try copyKv(r, host.ptr, snap.pages, host.pinned, .to_host);
    try d.sync(); // a page may be reused only after this
    const ms = (q.now() - t0) * 1e3;
    if (tp.group) |*g| try g.record(.park_snap, 0, 0, @intFromPtr(snap.buf), &.{});
    snap.park = host.ptr;
    snap.park_bytes = bytes;
    snap.park_pinned = host.pinned;
    own = false;
    for (snap.pages) |id| r.unkeepPage(id);
    alloc.free(snap.pages);
    snap.pages = &.{};
    r.park_moved = bytes;
    r.park_ms = ms;
    if (q.park_log) std.log.info("qwen27-xpu: rank {d}: park copy {d} tokens, {d:.1} MiB, {d:.1} ms", .{ q.RANK, snap.at, @as(f64, @floatFromInt(bytes)) / (1 << 20), ms });
}

/// Makes `slot` stand where `snap` was taken (its prompt pass goes on from snap.at).
/// A parked snap stays parked. Rank 0 makes room for the new pages before it records `restore_slot`.
pub fn restoreSlot(r: *Runner, slot: u32, snap: Snap) !void {
    if (snap.park != null) r.makeRoomParked(slot, snap.at);
    if (tp.group) |*g| try g.record(.restore_slot, slot, 0, @intFromPtr(snap.buf), &.{});
    try r.select(slot);
    const d = &r.d;
    const at = snap.at;
    if (q.kv_pool) {
        r.releaseSlot(slot); // the slot's own pages go back; the kept state's stay
        if (snap.park) |host| {
            const entries: u32 = (at + q.PAGE - 1) / q.PAGE;
            try r.mapNewPages(slot, entries);
            const t0 = q.now();
            try copyKv(r, host, r.pt_host[slot][0..entries], snap.park_pinned, .to_card);
            try copyGdnIn(d, r, snap.buf);
            r.pos = at;
            r.block_rows = 0;
            r.tree_on = false;
            try d.sync();
            const ms = (q.now() - t0) * 1e3;
            r.park_moved = snap.park_bytes;
            r.park_ms = ms;
            if (q.park_log) std.log.info("qwen27-xpu: rank {d}: restore copy {d} tokens, {d:.1} MiB, {d:.1} ms", .{ q.RANK, at, @as(f64, @floatFromInt(snap.park_bytes)) / (1 << 20), ms });
            return;
        }
        try r.mapPages(slot, 0, snap.pages);
        try copyGdnIn(d, r, snap.buf);
        r.pos = at;
        r.block_rows = 0;
        r.tree_on = false;
        try d.sync();
        return;
    }
    const lay = kv();
    var o: usize = 0;
    for (&r.model.layers) |*l| {
        if (l.attn) {
            for ([_]*anyopaque{ l.kc.?, l.vc.? }) |c| {
                try copy(d, c, off(snap.buf, o), @as(usize, at) * lay.row());
                o += @as(usize, at) * lay.row();
                if (q.kv_i8) {
                    try copy(d, off(c, lay.scales()), off(snap.buf, o), @as(usize, at) * lay.scaleRow());
                    o += @as(usize, at) * lay.scaleRow();
                }
            }
            continue;
        }
        try copy(d, l.rec.?, off(snap.buf, o), REC_BYTES);
        try copy(d, l.conv[l.conv_cur].?, off(snap.buf, o + REC_BYTES), CONV_BYTES);
        o += REC_BYTES + CONV_BYTES;
    }
    r.pos = at;
    r.block_rows = 0;
    r.tree_on = false;
    try d.sync();
}

pub fn dropSnap(r: *Runner, snap: Snap) void {
    if (tp.group) |*g| g.record(.drop_snap, 0, 0, @intFromPtr(snap.buf), &.{}) catch {};
    if (q.kv_pool) {
        for (snap.pages) |id| r.unkeepPage(id); // the pages go back to the pool at zero references
        if (snap.pages.len > 0) alloc.free(snap.pages);
    }
    freeAfterQueue(&r.d, snap.buf);
    if (snap.park) |p| {
        r.d.sync() catch {};
        parkFree(&r.d, p, snap.park_pinned);
    }
}

const Way = enum { to_host, to_card };
const park_align: usize = 4096;

fn parkAlloc(d: *dev.Dev, bytes: usize) !struct { ptr: *anyopaque, pinned: bool } {
    if (q.park_pinned) return .{ .ptr = try d.hostAlloc(bytes), .pinned = true };
    const n = (bytes + park_align - 1) & ~(park_align - 1);
    const p = aligned_alloc(park_align, n) orelse return error.OutOfMemory;
    return .{ .ptr = p, .pinned = false };
}

fn parkFree(d: *dev.Dev, ptr: *anyopaque, pinned: bool) void {
    if (pinned) d.g.free(ptr) else free(ptr);
}

fn copyKv(r: *Runner, host: *anyopaque, ids: []const u32, pinned: bool, way: Way) !void {
    const d = &r.d;
    const pb = r.page_bytes;
    const entries = ids.len;
    if (entries == 0 or pb == 0) return;
    const runs = try alloc.alloc(park_runs.Run, entries);
    defer alloc.free(runs);
    const nruns = park_runs.pageRuns(ids, runs);
    const span = entries * pb; // one of K or V for this layer
    var base: usize = 0;
    for (&r.model.layers) |*l| {
        if (!l.attn) continue;
        const kc = l.kc orelse return error.NotParkable;
        const vc = l.vc orelse return error.NotParkable;
        const k_base = base;
        base += span;
        const v_base = base;
        base += span;
        for (runs[0..nruns]) |run| {
            const nbytes = @as(usize, run.len) * pb;
            const atb = @as(usize, run.at) * pb;
            const id0 = ids[@as(usize, run.at)];
            const dev_k = off(kc, @as(usize, id0) * pb);
            const dev_v = off(vc, @as(usize, id0) * pb);
            const host_k = off(host, k_base + atb);
            const host_v = off(host, v_base + atb);
            if (way == .to_host) {
                try copySpan(d, host_k, dev_k, nbytes, pinned, true);
                try copySpan(d, host_v, dev_v, nbytes, pinned, true);
            } else {
                try copySpan(d, dev_k, host_k, nbytes, pinned, false);
                try copySpan(d, dev_v, host_v, nbytes, pinned, false);
            }
        }
    }
}

fn copySpan(d: *dev.Dev, dst: *anyopaque, src: *anyopaque, n: usize, pinned: bool, src_is_card: bool) !void {
    if (n == 0) return;
    if (pinned) {
        try copy(d, dst, src, n);
        return;
    }
    if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
    // A queued copy may still read staging. Drain it once before the first host write.
    if (!src_is_card) try d.sync();
    var done: usize = 0;
    while (done < n) {
        const c = @min(d.staging.len, n - done);
        const stage: *anyopaque = @ptrCast(d.staging.ptr);
        if (src_is_card) {
            try copy(d, stage, off(src, done), c);
            try d.sync();
            const dst_bytes: [*]u8 = @ptrCast(off(dst, done));
            @memcpy(dst_bytes[0..c], d.staging[0..c]);
        } else {
            const src_bytes: [*]const u8 = @ptrCast(off(src, done));
            @memcpy(d.staging[0..c], src_bytes[0..c]);
            try copy(d, off(dst, done), stage, c);
            try d.sync();
        }
        done += c;
    }
}

/// Frees `p` once the queue has drained; kept (until the device closes) when the queue cannot be waited for.
pub fn freeAfterQueue(d: *dev.Dev, p: *anyopaque) void {
    d.sync() catch return;
    d.g.free(p);
}

/// Slots whose states are not allocated yet (they take slotBytes on first use).
pub fn unallocatedSlots(r: *const Runner, slots: u32) u32 {
    var n: u32 = 0;
    for (0..slots) |i| n += @intFromBool(for (&r.model.layers) |*l| {
        if (l.attn) break q.kv_pool and l.slot_rec[i] == null or !q.kv_pool and l.slot_kc[i] == null;
    } else true);
    return n;
}

/// `slot`'s states, allocated (zero) on its first use: all of them, or none when an allocation fails. With the pool
/// the K/V comes from it, so only the GDN and convolution states are allocated here.
pub fn ensure(r: *Runner, slot: u32) !void {
    if (tp.group) |*g| try g.record(.ensure, slot, 0, 0, &.{});
    const d = &r.d;
    var had: [q.LAYERS]bool = undefined;
    for (&r.model.layers, &had) |*l, *h| h.* = if (l.attn) (q.kv_pool or l.slot_kc[slot] != null) else l.slot_rec[slot] != null;
    errdefer {
        for (&r.model.layers, had) |*l, h| {
            if (h) continue;
            const c = &l.slot_conv[slot];
            for ([_]*?*anyopaque{ &l.slot_kc[slot], &l.slot_vc[slot], &l.slot_rec[slot], &l.slot_save[slot], &c[0], &c[1] }) |pp| if (pp.*) |p| {
                freeAfterQueue(d, p);
                pp.* = null;
            };
        }
    }
    for (&r.model.layers) |*l| {
        if (l.attn) {
            if (q.kv_pool) { // every slot's view is the pool; its page table says which pages
                l.slot_kc[slot] = l.kc;
                l.slot_vc[slot] = l.vc;
                continue;
            }
            if (l.slot_kc[slot] != null) continue;
            l.slot_kc[slot] = try d.alloc_(kvBytes());
            l.slot_vc[slot] = try d.alloc_(kvBytes());
        } else {
            if (l.slot_rec[slot] != null) continue;
            l.slot_rec[slot] = try d.alloc_(q.GDN_V_HEADS * 128 * 128 * 4);
            l.slot_save[slot] = try d.alloc_(q.GDN_V_HEADS * 128 * 128 * 4);
            try d.fill(l.slot_rec[slot].?, @as(u32, 0), q.GDN_V_HEADS * 128 * 128 * 4);
            for (&l.slot_conv[slot]) |*cv| {
                cv.* = try d.alloc_(3 * q.GDN_QKV * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * q.GDN_QKV * 2);
            }
            l.slot_cur[slot] = 0;
        }
    }
    try d.sync();
}

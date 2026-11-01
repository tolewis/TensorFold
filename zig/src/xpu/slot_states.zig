//! A slot's states: allocation on first use, and the prompt cache's snapshots (save, restore, drop).
//! With the shared K/V pool (XPU_KV_POOL) a snapshot keeps the slot's pages (their ids, one reference each) and copies
//! only the GDN and convolution states; a restore maps the pages back, so a resume costs a page-table copy.
const q = @import("qwen27.zig");
const tp = @import("tp.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const Runner = q.Runner;
const copy = dev.copy;
const kvBytes = fwd.kvBytes;
const off = dev.off;
const alloc = q.alloc;

/// Bytes of the GDN and convolution states a slot takes whatever its context: each GDN layer's states and convolution.
pub fn gdnBytes(r: *const Runner) usize {
    var layers: usize = 0;
    for (r.model.layers) |l| layers += @intFromBool(!l.attn);
    return layers * (2 * q.GDN_V_HEADS * 128 * 128 * 4 + 2 * 3 * q.GDN_QKV * 2);
}

/// Bytes a slot's states take: GDN and convolution states, and the K/V cache of `context` positions (none with the pool).
pub fn slotBytes(r: *const Runner) usize {
    return gdnBytes(r) + if (q.kv_pool) 0 else (q.LAYERS / 4) * 2 * kvBytes();
}

/// A slot's state after `at` prompt tokens (prompt cache): GDN and convolution states, K/V rows 0 .. at, one buffer.
/// With the pool: the GDN and convolution states only, and `pages`, the slot's page ids from entry 0 to ceil(at / PAGE).
pub const Snap = struct { at: u32, buf: *anyopaque, bytes: usize, pages: []u32 = &.{} };

pub const REC_BYTES = q.GDN_V_HEADS * 128 * 128 * 4;
pub const CONV_BYTES = 3 * q.GDN_QKV * 2;
/// The K/V cache layout of this rank: its own K/V heads (half at TP 2), as the kernels index it (Q27_KVROW).
pub fn kv() q.limits.KvLayout {
    return .{ .heads = q.ATT_KV_HEADS, .cap = q.MAX_CTX, .int8 = q.kv_i8 };
}

/// The bytes saveSlot takes at `at` tokens; with the pool, the pages the state holds are charged too.
pub fn snapBytes(r: *const Runner, at: u32) usize {
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
        const gdn = gdnBytes(r);
        const buf = try d.alloc_(gdn);
        errdefer freeAfterQueue(d, buf); // copies into it may be queued
        try copyGdnOut(d, r, buf);
        const entries = (at + q.PAGE - 1) / q.PAGE;
        if (entries > r.held[slot]) return error.SlotNotThere;
        const pages = try alloc.alloc(u32, @as(usize, entries));
        errdefer alloc.free(pages);
        @memcpy(pages, r.pt_host[slot][0..entries]);
        for (pages) |id| r.retainPage(id); // the state holds them until dropSnap
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

/// Makes `slot` stand where `snap` was taken (its prompt pass goes on from snap.at).
pub fn restoreSlot(r: *Runner, slot: u32, snap: Snap) !void {
    if (tp.group) |*g| try g.record(.restore_slot, slot, 0, @intFromPtr(snap.buf), &.{});
    try r.select(slot);
    const d = &r.d;
    const at = snap.at;
    if (q.kv_pool) {
        r.releaseSlot(slot); // the slot's own pages go back; the kept state's stay
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
        for (snap.pages) |id| r.releasePage(id); // the pages go back to the pool at zero references
        if (snap.pages.len > 0) alloc.free(snap.pages);
    }
    freeAfterQueue(&r.d, snap.buf);
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

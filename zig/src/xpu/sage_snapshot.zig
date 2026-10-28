//! A Sage slot's prompt state for the prompt cache, kept in the K/V pool. Its K/V rows (attention and MTP layers, with
//! their int8 scales) are the slot's own pages 0 .. at / PAGE, held by reference: the slot only writes past `at`, so
//! nothing copies them. Each GDN layer's recurrent and convolution states are copied into pool pages that no table maps.
//! The verify stash and rec_save are scratch of a decode round: a prompt pass that stands at a chunk end does not read
//! them. The caller (the lane host's Slots) owns the page ids and their holders; this file moves the bytes.
const sage = @import("sage.zig");
const dev = @import("dev.zig");
const Runner = sage.Runner;
const off = dev.off;

/// A slot's state after `at` prompt tokens: ids[0 .. at / PAGE] its K/V pages, then its GDN pages (gdnPages).
pub const Snap = struct { at: u32, ids: []u32 };

fn recBytes(r: *const Runner) usize {
    return @as(usize, r.z.s.gdn_v_heads) * 128 * 128 * 4;
}

fn convBytes(r: *const Runner) usize {
    return 3 * @as(usize, r.z.nconv) * 2;
}

/// Bytes of every GDN layer's recurrent and convolution state.
pub fn gdnBytes(r: *const Runner) usize {
    var n: usize = 0;
    for (r.model.layers) |l| {
        if (!l.attn) n += recBytes(r) + convBytes(r);
    }
    return n;
}

/// Pool bytes of one page over every attention layer and the MTP layer, K and V.
pub fn poolPageBytes(r: *const Runner) usize {
    return r.kvBytesPerToken() * sage.PAGE;
}

/// Pool pages the GDN states of one kept state take.
pub fn gdnPages(r: *const Runner) u32 {
    return @intCast((gdnBytes(r) + poolPageBytes(r) - 1) / poolPageBytes(r));
}

/// Pool bytes a state at `at` tokens holds: its K/V pages and its GDN pages.
pub fn bytes(r: *const Runner, at: u32) usize {
    return (@as(usize, (at + sage.PAGE - 1) / sage.PAGE) + gdnPages(r)) * poolPageBytes(r);
}

/// The GDN states laid over pool pages, in order: each run of consecutive page ids gives one piece of every attention
/// layer's K, then V (and the MTP layer's), that run's pages long. A piece of a state may cross pieces of pages.
const Gdn = struct {
    r: *Runner,
    ids: []const u32,
    to_pages: bool,
    halves: [2 * (MAX_LAYERS + 1)]*anyopaque = undefined, // the pool's K and V halves of each layer
    nh: usize = 0,
    i: usize = 0, // the first id of the current run
    n: usize = 0, // ids in the current run (0: none yet)
    h: usize = 0, // the current half
    o: usize = 0, // bytes used of the current piece

    const MAX_LAYERS = 64;

    fn init(r: *Runner, ids: []const u32, to_pages: bool) !Gdn {
        var g: Gdn = .{ .r = r, .ids = ids, .to_pages = to_pages };
        for (r.model.layers) |*l| if (l.attn) try g.half(l.kc.?, l.vc.?);
        if (r.mtp) |mt| try g.half(mt.layer.kc.?, mt.layer.vc.?);
        return g;
    }

    fn half(g: *Gdn, k: *anyopaque, v: *anyopaque) !void {
        if (g.nh + 2 > g.halves.len) return error.SnapshotLayout;
        g.halves[g.nh] = k;
        g.halves[g.nh + 1] = v;
        g.nh += 2;
    }

    fn piece(g: *const Gdn) usize {
        return g.n * g.r.page_bytes;
    }

    /// The next piece of pages: the next half of this run, else the next run's first.
    fn next(g: *Gdn) !void {
        g.o = 0;
        g.h += 1;
        if (g.n > 0 and g.h < g.nh) return;
        g.i += g.n;
        g.h = 0;
        if (g.i >= g.ids.len) return error.SnapshotSize;
        g.n = 1;
        while (g.i + g.n < g.ids.len and g.ids[g.i + g.n] == g.ids[g.i] + g.n) g.n += 1;
    }

    /// `n` bytes of a live state to the pages, or back.
    fn copy(g: *Gdn, live: *anyopaque, n: usize) !void {
        var done: usize = 0;
        while (done < n) {
            if (g.n == 0 or g.o == g.piece()) try g.next();
            const k = @min(n - done, g.piece() - g.o);
            const page = off(g.halves[g.h], @as(usize, g.ids[g.i]) * g.r.page_bytes + g.o);
            if (g.to_pages) try dev.copy(&g.r.d, page, off(live, done), k) else try dev.copy(&g.r.d, off(live, done), page, k);
            g.o += k;
            done += k;
        }
    }

    fn all(g: *Gdn) !void {
        for (g.r.model.layers) |*l| if (!l.attn) {
            try g.copy(l.rec.?, recBytes(g.r));
            try g.copy(l.conv[l.conv_cur].?, convBytes(g.r)); // the state the next chunk reads
        };
    }
};

/// Copies `slot`'s GDN states, standing at `at` prompt tokens, into pool pages `gdn` (gdnPages of them). The copies are
/// queued on the in-order list: the slot's next chunk runs after them.
pub fn save(r: *Runner, slot: u32, at: u32, gdn: []const u32) !void {
    try r.use(slot);
    if (at == 0 or r.pos != at or r.block_rows != 0 or (r.mtp != null and r.mpos[slot] != at)) return error.SlotNotThere;
    if (r.pt[slot] == null or at % sage.PAGE != 0 or at / sage.PAGE > r.held[slot] or gdn.len != gdnPages(r)) return error.SnapshotLayout;
    var g = try Gdn.init(r, gdn, true);
    try g.all();
}

/// Whether `snap` can be restored into a slot: a whole number of pages, inside the context, its ids in the pool.
pub fn fits(r: *const Runner, snap: Snap) bool {
    if (snap.at == 0 or snap.at >= r.ctx or snap.at % sage.PAGE != 0) return false;
    if (snap.ids.len != snap.at / sage.PAGE + gdnPages(r)) return false;
    for (snap.ids) |id| if (id >= r.pool) return false;
    return true;
}

/// Makes `slot` stand where `snap` was taken: its table must map the state's K/V pages from entry 0 (the lease adopts
/// them), and its GDN states are copied back. Its prompt pass goes on from snap.at, in pages past the state's.
pub fn restore(r: *Runner, slot: u32, snap: Snap) !void {
    if (!fits(r, snap) or r.pt[slot] == null) return error.SnapshotLayout;
    const kv = snap.at / sage.PAGE;
    if (r.held[slot] <= kv) return error.SlotPagesShort; // the prompt pass leases past the kept tokens first
    for (r.pt_host[slot][0..kv], snap.ids[0..kv]) |a, b| if (a != b) return error.SnapshotLayout;
    try r.use(slot);
    var g = try Gdn.init(r, snap.ids[kv..], false);
    try g.all();
    r.pos = snap.at;
    r.last = -1; // the next chunk's
    r.shift[slot] = 0;
    r.mpos[slot] = snap.at;
    r.sdraft[slot] = -1;
    if (r.df) |df| df.restart(slot, snap.at); // the drafter's context starts at the state: its rows before are not kept
    r.block_rows = 0;
}

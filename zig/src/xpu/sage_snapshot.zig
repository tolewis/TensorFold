//! A Sage slot's prompt state for the prompt cache: K/V rows of the attention layers and of the MTP layer, and each
//! GDN layer's recurrent and convolution states. The verify stash and rec_save are scratch of a decode round: a prompt
//! pass that stands at a chunk end does not read them.
const sage = @import("sage.zig");
const dev = @import("dev.zig");
const Runner = sage.Runner;
const off = dev.off;

/// A slot's state after `at` prompt tokens, in one device buffer.
pub const Snap = struct { at: u32, buf: *anyopaque, bytes: usize };

/// Bytes of one cache position of one K or V cache: AKV heads of 256 int8 (then their fp32 scales), or of 256 bf16.
fn rowBytes(r: *const Runner) usize {
    return @as(usize, r.z.s.kv_heads) * if (r.cap > 0) @as(usize, 256) else 512;
}

fn recBytes(r: *const Runner) usize {
    return @as(usize, r.z.s.gdn_v_heads) * 128 * 128 * 4;
}

fn convBytes(r: *const Runner) usize {
    return 3 * @as(usize, r.z.nconv) * 2;
}

/// The bytes save takes at `at` tokens.
pub fn bytes(r: *const Runner, at: u32) usize {
    var n: usize = @as(usize, at) * r.kvBytesPerToken();
    for (r.model.layers) |l| {
        if (!l.attn) n += recBytes(r) + convBytes(r);
    }
    return n;
}

/// The current slot's state at `at` tokens, piece after piece in a fixed order: into the snapshot `buf`, or back.
const Walk = struct {
    r: *Runner,
    buf: *anyopaque,
    to_snap: bool,
    o: usize = 0,

    fn piece(w: *Walk, live: *anyopaque, n: usize) !void {
        const d = &w.r.d;
        if (w.to_snap) try dev.copy(d, off(w.buf, w.o), live, n) else try dev.copy(d, live, off(w.buf, w.o), n);
        w.o += n;
    }

    /// One K or V cache's rows 0 .. at, and with int8 their scales (fp32 [cap, AKV] after the cap's rows).
    fn cache(w: *Walk, c: *anyopaque, at: u32) !void {
        const r = w.r;
        try w.piece(c, @as(usize, at) * rowBytes(r));
        if (r.cap > 0) try w.piece(off(c, @as(usize, @intCast(r.cap)) * rowBytes(r)), @as(usize, at) * r.z.s.kv_heads * 4);
    }

    fn all(w: *Walk, at: u32) !void {
        const r = w.r;
        for (r.model.layers) |*l| if (l.attn) {
            try w.cache(l.kc.?, at);
            try w.cache(l.vc.?, at);
        };
        if (r.mtp) |mt| {
            try w.cache(mt.layer.kc.?, at);
            try w.cache(mt.layer.vc.?, at);
        }
        for (r.model.layers) |*l| if (!l.attn) {
            try w.piece(l.rec.?, recBytes(r));
            try w.piece(l.conv[l.conv_cur].?, convBytes(r)); // the state the next chunk reads
        };
        if (w.o != bytes(r, at)) return error.SnapshotSize;
    }
};

/// Copies `slot`'s state, standing at `at` prompt tokens, into new device memory.
pub fn save(r: *Runner, slot: u32, at: u32) !Snap {
    try r.use(slot);
    if (at == 0 or r.pos != at or r.block_rows != 0 or (r.mtp != null and r.mpos[slot] != at)) return error.SlotNotThere;
    const n = bytes(r, at);
    const buf = try r.d.alloc_(n);
    errdefer free(r, buf, n);
    var w: Walk = .{ .r = r, .buf = buf, .to_snap = true };
    try w.all(at);
    return .{ .at = at, .buf = buf, .bytes = n };
}

/// Makes `slot` stand where `snap` was taken: its prompt pass goes on from snap.at.
pub fn restore(r: *Runner, slot: u32, snap: Snap) !void {
    if (snap.at == 0 or snap.at >= r.ctx or snap.bytes != bytes(r, snap.at)) return error.SnapshotLayout;
    try r.use(slot);
    var w: Walk = .{ .r = r, .buf = snap.buf, .to_snap = false };
    try w.all(snap.at);
    r.pos = snap.at;
    r.last = -1; // the next chunk's
    r.shift[slot] = 0;
    r.mpos[slot] = snap.at;
    r.sdraft[slot] = -1;
    r.block_rows = 0;
    try r.d.sync();
}

pub fn drop(r: *Runner, snap: Snap) void {
    free(r, snap.buf, snap.bytes);
}

/// Frees `p` once the queue has drained (copies into or out of it may be queued); kept when it cannot be waited for.
fn free(r: *Runner, p: *anyopaque, n: usize) void {
    r.d.sync() catch return;
    r.d.g.free(p);
    r.d.allocated -|= @max(n, 64);
}

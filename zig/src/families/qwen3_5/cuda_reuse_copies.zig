//! A kept state's copies between a cache and its host or device copy: device runs, bf16 page copies, and runs
//! split where a pooled copy's arena ranges end.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const caches = @import("cuda_cache.zig");
const reuse = @import("cuda_reuse.zig");

const Engine = engine.Engine;
const Cache = engine.Cache;
const Layout = reuse.Layout;
const Piece = reuse.Piece;
const Copy = reuse.Copy;
const head_dim = @import("cuda_geometry.zig").head_dim;

/// A piece's device spans: one, or with a page table a run a page (the host copy keeps the contiguous order).
pub const Runs = struct {
    base: u64,
    pages: ?caches.Paged,
    page_bytes: usize,
    in_page: usize,
    span: usize = 0, // a full page's run of the piece
    len: usize,
    page: usize = 0,
    done: usize = 0,

    const Run = struct { dev: u64, at: usize, len: usize };

    pub fn of(cache: *const Cache, l: Layout, i: usize, p: Piece) Runs {
        if (!p.kv or cache.paged == null) return .{ .base = (if (p.kv) cache.kv.ptr else cache.state.ptr) + p.off, .pages = null, .page_bytes = 0, .in_page = 0, .len = p.len };
        const pg = cache.paged.?;
        // the piece's offset inside a page: its layout at cap = tokens (Layout.piece), a run of tokens rows a page
        const in_page: Layout = .{ .g = l.g, .format = l.format, .cap = pg.tokens, .at = pg.tokens };
        const q = in_page.piece(i);
        return .{ .base = pg.base, .pages = pg, .page_bytes = in_page.kvBytes(), .in_page = q.off, .span = q.len, .len = p.len };
    }

    pub fn next(r: *Runs) ?Run {
        if (r.done >= r.len) return null;
        const pg = r.pages orelse {
            r.done = r.len;
            return .{ .dev = r.base, .at = 0, .len = r.len };
        };
        const n = @min(r.span, r.len - r.done);
        const run: Run = .{ .dev = r.base + pg.ids[r.page] * r.page_bytes + r.in_page, .at = r.done, .len = n };
        r.page += 1;
        r.done += n;
        return run;
    }
};

/// Whether a cache's keys and values go a page a copy: bf16 pages, whose pieces lie back to back a page's rows apart.
pub fn wholePages(cache: *const Cache, l: Layout) bool {
    return cache.paged != null and l.format == .bf16 and l.at > 0;
}

/// Every bf16 piece's rows from `at`'s pages on `s`, one 2D copy a page (`tokens` rows apart there, `at` in `copy`).
pub fn pageCopies(e: *Engine, cache: *const Cache, l: Layout, copy: *const Copy, off: usize, way: Way, s: cuda.abi.Stream) !void {
    const pg = cache.paged.?;
    const row = head_dim * 2;
    const page_bytes = (Layout{ .g = l.g, .format = l.format, .cap = pg.tokens, .at = pg.tokens }).kvBytes();
    const pieces = l.count() - 1;
    var p: usize = 0;
    while (p * pg.tokens < l.at) : (p += 1) {
        const rows = @min(pg.tokens, l.at - p * pg.tokens);
        const dev = pg.base + pg.ids[p] * page_bytes;
        const o = off + p * pg.tokens * row;
        // the kept side: device memory on an integrated GPU (copy.dev), else the pinned host copy
        const kept: cuda.abi.DevicePtr = if (copy.dev) |b| b.ptr + o else 0;
        const host: ?*u8 = if (copy.dev == null) &copy.host.bytes.ptr[o] else null;
        const kind: c_int = @intCast(if (copy.dev == null) cuda.abi.memory_type_host else cuda.abi.memory_type_device);
        const m: cuda.abi.Memcpy2D = switch (way) {
            .save => .{ .src = dev, .src_pitch = pg.tokens * row, .dst_type = kind, .dst_host = host, .dst = kept, .dst_pitch = l.at * row, .width = rows * row, .height = pieces },
            .restore => .{ .src_type = kind, .src_host = host, .src = kept, .src_pitch = l.at * row, .dst = dev, .dst_pitch = pg.tokens * row, .width = rows * row, .height = pieces },
        };
        try e.d.check(e.d.api.cuMemcpy2DAsync_v2(&m, s), "cuMemcpy2DAsync");
    }
}

pub const Way = enum { save, restore };

/// One device run to or from the host copy at offset `o` on `s`, split where a pooled copy's arena ranges end.
pub fn hostRun(e: *Engine, copy: *const Copy, o: usize, dev: u64, len: usize, way: Way, s: cuda.abi.Stream) !void {
    var done: usize = 0;
    while (done < len) {
        const h = if (copy.pooled) copy.parts.at(o + done) else copy.host.bytes[o + done ..];
        const n = @min(h.len, len - done);
        if (n == 0) return error.CopyPastHost;
        switch (way) {
            .save => try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(h.ptr, dev + done, n, s), "cuMemcpyDtoHAsync"),
            .restore => try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dev + done, h.ptr, n, s), "cuMemcpyHtoDAsync"),
        }
        done += n;
    }
}

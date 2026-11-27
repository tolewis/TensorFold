//! Pure host geometry and argument declarations for Inkling 2 Mini attention on the XPU backend.
//!
//! The model constants and the ring and window rules are the Metal plan's. The launch geometry is not: a sub-group is
//! 16 lanes, a decode sub-group serves the three query heads of one K/V head, and the key chunks are smaller so that a
//! single row still fills the card.
const std = @import("std");

pub const head_dim = 128;
pub const d_rel = 16;
pub const extent = 1024;
/// Query heads one K/V head serves; the decode kernel is built for this ratio.
pub const group = 3;
pub const tile_keys = 64;
pub const sliding_chunk = 64;
pub const global_chunk = 256;
/// The prompt kernel's chunk. It is a multiple of the key tile and wider than the decode chunk to keep the partials small.
pub const prompt_chunk = 1024;

/// Positions one K code tile holds and its bytes, for an int8 cache: [dim / 4][16 positions][4 int8] inside a head's ring.
/// `ring` is a multiple of 16 for every slot, so a tile never straddles the ring end.
pub const kv_tile = 16;
pub const kv_tile_bytes = head_dim * kv_tile;

/// Byte of dim `d` of the position `p` in the K codes of head `h` of a ring of `ring` positions.
pub fn kCodeOffset(h: usize, p: usize, d: usize, ring: usize) usize {
    const at = p % ring;
    return h * ring * head_dim + (at / kv_tile) * kv_tile_bytes + (d / 4) * 64 + (at % kv_tile) * 4 + d % 4;
}

/// Bytes a head's K codes take for `live` positions: whole tiles, so the last tile is padded to 16 positions.
pub fn kCodeBytes(live: usize) usize {
    return std.mem.alignForward(usize, live, kv_tile) * head_dim;
}

/// Byte of dim `d` of the position `p` in the V codes of head `h`: row-major, one byte a value.
pub fn vCodeOffset(h: usize, p: usize, d: usize, ring: usize) usize {
    return h * ring * head_dim + (p % ring) * head_dim + d;
}

/// Byte of the fp32 scale of the position `p` of K or V head `h`: head-major, as the snapshot expects.
pub fn kvScaleOffset(h: usize, p: usize, ring: usize) usize {
    return (h * ring + p % ring) * 4;
}
pub const split_rows = 64;
pub const prompt_pairs = 16;
pub const prompt_threads = 16;
pub const decode_threads = 16;
pub const merge_threads = head_dim;

/// When set, decode uses a chunk that yields about this many K/V splits a head, if the partial buffer can hold them.
/// Zero keeps the layer chunk. The logits tool sets it for an A/B.
pub var flash_splits: u32 = 0;
pub var flash_user_set: bool = false;

/// A verify window runs its attention in one launch: the one-row decode kernel's grid is the rows' K/V heads, and each
/// work-group reads its own row's position, first key and partial slots. `INK2_ROWS_OLD=1` runs one launch a row for the
/// A/B.
pub var rows_old: ?bool = null;
pub fn rowBatched() bool {
    if (rows_old) |v| return !v;
    const v = if (std.c.getenv("INK2_ROWS_OLD")) |p| std.mem.eql(u8, std.mem.span(p), "1") else false;
    rows_old = v;
    return !v;
}

/// A stream's slice of the K/V cache: rows `row0 ..` hold positions `p0 ..`; `off` is in elements.
pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
comptime {
    if (@sizeOf(Seg) != 24) @compileError("Inkling attention segment ABI drift");
}

/// The int8 kernels add `off` to the code pointers only (bytes in prep8 and decode8, bf16 values in prep8q) and
/// not to the scale pointers, and dequant8 reads no segment. So an int8 cache takes no segment offset: refuse one.
pub fn checkSegs(segs: []const Seg, int8: bool) error{Inkling2Int8SegmentOffset}!void {
    if (!int8) return;
    for (segs) |s| if (s.off != 0) return error.Inkling2Int8SegmentOffset;
}

/// What `tf_ink2_prep` takes after its buffers, in order; the kernel takes each as its own scalar argument.
pub const PrepArgs = struct { heads: u32, kv_heads: u32, ring: u32, global: u32, qk_clip: f32, rel_clip: f32, floor_pos: f32, alpha: f32, eps: f32 };
pub const AttnArgs = struct { heads: u32, kv_heads: u32, ring: u32, window: u32, chunk: u32, c0: u32, chunks: u32, scale: f32 };
pub const MergeArgs = struct { heads: u32, window: u32, chunk: u32, c0: u32, chunks: u32, subs: u32 = 1 };
pub const PromptArgs = struct { heads: u32, kv_heads: u32, ring: u32, chunk: u32, c0: u32, chunks: u32, scale: f32 };

pub const Layer = struct {
    window: u32,
    ring: u32,
    chunk: u32,
    tau: bool,
    pub fn sliding(window: u32, rows: u32) Layer {
        return .{ .window = window, .ring = ringFor(window, rows), .chunk = sliding_chunk, .tau = false };
    }
    pub fn global(capacity: u32) Layer {
        return .{ .window = 0, .ring = @intCast(std.mem.alignForward(usize, capacity, tile_keys)), .chunk = global_chunk, .tau = true };
    }
    pub fn cacheBytes(l: Layer, kv: usize) usize {
        return @as(usize, l.ring) * kv * head_dim * 2;
    }
};
pub const Prep = struct { qk_clip: f32 = 22.627416997969522, rel_clip: f32 = 8, floor: f32 = 8192, alpha: f32 = 0.1, eps: f32 = 1e-6 };
pub const Rows = struct { rows: u32, heads: u32 = 24, kv_heads: u32 = 8, last_pos: u32, streams: u32 = 1 };

pub fn ringFor(window: u32, rows: u32) u32 {
    return @intCast(std.mem.alignForward(usize, @as(usize, window) + rows - 1, tile_keys));
}
pub fn chunks(last_pos: usize, chunk: u32) u32 {
    return @intCast(last_pos / chunk + 1);
}
/// Partial slots a row and head need over `context` keys, at the layer's decode chunk.
pub fn slotsFor(l: Layer, context: usize) u32 {
    if (l.window > 0) return (l.window - 1) / l.chunk + 2;
    return chunks(context - 1, l.chunk);
}
pub fn usePrompt(r: Rows, l: Layer) bool {
    return l.window == 0 and r.streams == 1 and r.rows > split_rows;
}
pub fn partialSlot(l: Layer, c: usize, pos: usize) ?usize {
    const lo = if (l.window > 0) (pos + 1) -| l.window else 0;
    if (c < lo / l.chunk or c > pos / l.chunk) return null;
    return c - lo / l.chunk;
}
pub fn tableBytes(rows: usize, heads: u32) usize {
    return rows * heads * extent * 2;
}
pub fn partialBytes(rows: usize, heads: u32, n_chunks: u32) usize {
    return rows * heads * n_chunks * (head_dim * 4 + 8);
}

pub const Path = enum { decode, prompt };
/// `chunk` is the key chunk the launched kernel and the merge both use; `partials_rh` the partial slots a row and head hold.
pub const Launch = struct { path: Path, grid: [3]u32, threads: u32, merge_grid: [3]u32, partials_rh: u32, chunk: u32, subs: u32 = 1 };
pub const Error = error{ Inkling2Partials, Inkling2Heads, Inkling2Path };

/// `stride` is the partial slots a row and head the caller holds: at least `slotsFor` at the layer's chunk.
pub fn attend(r: Rows, l: Layer, stride: u32) Error!Launch {
    return attendAs(r, l, stride, if (usePrompt(r, l)) .prompt else .decode);
}

/// The launch for a path the caller names; the prompt kernel serves global layers of one stream only.
pub fn attendAs(r: Rows, l: Layer, stride: u32, path: Path) Error!Launch {
    if (r.heads != r.kv_heads * group) return error.Inkling2Heads;
    if (slotsFor(l, @as(usize, r.last_pos) + 1) > stride) return error.Inkling2Partials;
    const merge_grid = [3]u32{ r.rows * r.heads, 1, 1 };
    if (path == .prompt) {
        if (l.window != 0 or r.streams != 1) return error.Inkling2Path;
        const blocks = std.math.divCeil(u32, r.rows * group, prompt_pairs) catch unreachable;
        const slots = chunks(@as(usize, r.last_pos), prompt_chunk);
        return .{ .path = .prompt, .grid = .{ blocks, r.kv_heads, slots }, .threads = prompt_threads, .merge_grid = merge_grid, .partials_rh = stride, .chunk = prompt_chunk };
    }
    return .{ .path = .decode, .grid = .{ r.rows * r.kv_heads, stride, 1 }, .threads = decode_threads, .merge_grid = merge_grid, .partials_rh = stride, .chunk = decodeChunk(l, r.last_pos, stride) };
}

/// A chunk no smaller than 16 that yields about `flash_splits` splits, or the layer chunk when that does not fit.
fn decodeChunk(l: Layer, last_pos: u32, stride: u32) u32 {
    if (flash_splits == 0) return l.chunk;
    const keys: u32 = if (l.window > 0) @min(l.window, last_pos + 1) else last_pos + 1;
    if (keys <= 16) return l.chunk;
    var chunk: u32 = std.math.divCeil(u32, keys, flash_splits) catch l.chunk;
    chunk = std.mem.alignForward(u32, @max(chunk, 16), 16);
    var trial = l;
    trial.chunk = chunk;
    while (slotsFor(trial, @as(usize, last_pos) + 1) > stride and chunk < l.chunk) {
        chunk += 16;
        trial.chunk = chunk;
    }
    if (slotsFor(trial, @as(usize, last_pos) + 1) > stride) return l.chunk;
    return chunk;
}

test {
    _ = @import("attention_test.zig");
}

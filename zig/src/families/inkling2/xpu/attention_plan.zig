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
pub const split_rows = 64;
pub const prompt_pairs = 16;
pub const prompt_threads = 16;
pub const decode_threads = 16;
pub const merge_threads = head_dim;

/// A stream's slice of the K/V cache: rows `row0 ..` hold positions `p0 ..`; `off` is in elements.
pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
comptime {
    if (@sizeOf(Seg) != 24) @compileError("Inkling attention segment ABI drift");
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
    return .{ .path = .decode, .grid = .{ r.rows * r.kv_heads, stride, 1 }, .threads = decode_threads, .merge_grid = merge_grid, .partials_rh = stride, .chunk = l.chunk };
}

test {
    _ = @import("attention_test.zig");
}

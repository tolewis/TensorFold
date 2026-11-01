//! The XPU runner's size settings, checked before any buffer is sized from them (host-tested, no device).
const std = @import("std");

/// The model's own context limit: a stream holds 1 to this many positions.
pub const MAX_CONTEXT = 262144;

/// XPU_CHUNK: prompt rows per forward, a multiple of the GDN chunk (64) from 64 to 8192.
pub fn chunk(text: []const u8) error{BadChunk}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.BadChunk;
    if (n == 0 or n % 64 != 0 or n > 8192) return error.BadChunk;
    return n;
}

/// A stream's positions, refused outside 1 .. MAX_CONTEXT.
pub fn context(n: i64) error{BadContext}!u32 {
    if (n <= 0 or n > MAX_CONTEXT) return error.BadContext;
    return @intCast(n);
}

/// GDN chunks of 64 rows over `rows`, rounded up: the last one is padded.
pub fn chunks(rows: u32) u32 {
    return (rows + 63) / 64;
}

/// Attention splits of `split` keys over `keys`, rounded up: the last one may be partial.
pub fn splits(keys: u32, split: u32) u32 {
    return (keys + split - 1) / split;
}

/// One layer's K or V cache buffer as the attention kernels lay it out (glue_attn.cl store_kv8, decode.cl): `heads`
/// K/V heads a rank (4 at TP 1, 2 at TP 2) of 256 values a position. Int8: 256 bytes a head and position, then
/// a float scale a head and position after `cap` positions (Q27_KVROW = heads * 256). Bf16: 512 bytes a head.
pub const KvLayout = struct {
    heads: usize,
    cap: usize,
    int8: bool,

    /// The values' bytes a position (Q27_KVROW at int8).
    pub fn row(l: KvLayout) usize {
        return l.heads * 256 * @as(usize, if (l.int8) 1 else 2);
    }
    /// Where the scales start (int8): after `cap` positions of values.
    pub fn scales(l: KvLayout) usize {
        return l.cap * l.heads * 256;
    }
    /// The scales' bytes a position (int8; none at bf16).
    pub fn scaleRow(l: KvLayout) usize {
        return if (l.int8) l.heads * 4 else 0;
    }
    /// The bytes positions 0 .. at take in a snapshot: their values, then their scales.
    pub fn snapBytes(l: KvLayout, at: usize) usize {
        return at * (l.row() + l.scaleRow());
    }
};

test "XPU_CHUNK must be a positive multiple of 64 within 8192" {
    try std.testing.expectEqual(@as(u32, 1024), try chunk("1024"));
    try std.testing.expectEqual(@as(u32, 64), try chunk("64"));
    for ([_][]const u8{ "66", "0", "8256", "-64", "x", "" }) |t| try std.testing.expectError(error.BadChunk, chunk(t));
}

test "a context outside 1 .. MAX_CONTEXT is refused" {
    try std.testing.expectEqual(@as(u32, 32768), try context(32768));
    try std.testing.expectError(error.BadContext, context(0));
    try std.testing.expectError(error.BadContext, context(-1));
    try std.testing.expectError(error.BadContext, context(MAX_CONTEXT + 1));
    try std.testing.expectError(error.BadContext, context(1 << 33));
}

test "chunk and split counts round up" {
    try std.testing.expectEqual(@as(u32, 4), chunks(194)); // 193 rows + a rest pad to 256
    try std.testing.expectEqual(@as(u32, 18), chunks(1152));
    try std.testing.expectEqual(@as(u32, 129), splits(4097, 32));
    try std.testing.expectEqual(@as(u32, 128), splits(4096, 32));
}

/// A test buffer filled as store_kv8 writes it: values and scales of positions 0 .. at, the rest 0xEE.
fn kernelFill(buf: []u8, l: KvLayout, at: usize) void {
    @memset(buf, 0xEE);
    for (0..at) |pos| for (0..l.heads) |h| {
        for (0..256) |i| buf[(pos * l.heads + h) * 256 + i] = @truncate(pos * 31 + h * 7 + i);
        const s: f32 = @floatFromInt(1 + pos * l.heads + h);
        @memcpy(buf[l.cap * l.heads * 256 + (pos * l.heads + h) * 4 ..][0..4], std.mem.asBytes(&s));
    };
}

/// A save then a restore through `l` into a fresh buffer, as slot_states copies them.
fn saveRestore(src: []const u8, dst: []u8, snap: []u8, l: KvLayout, at: usize) void {
    @memcpy(snap[0 .. at * l.row()], src[0 .. at * l.row()]);
    @memcpy(snap[at * l.row() ..][0 .. at * l.scaleRow()], src[l.scales()..][0 .. at * l.scaleRow()]);
    @memset(dst, 0);
    @memcpy(dst[0 .. at * l.row()], snap[0 .. at * l.row()]);
    @memcpy(dst[l.scales()..][0 .. at * l.scaleRow()], snap[at * l.row() ..][0 .. at * l.scaleRow()]);
}

test "an int8 K/V snapshot carries every value and scale the kernels read, at TP 1 and TP 2" {
    const gpa = std.testing.allocator;
    const cap = 64;
    const at = 37;
    for ([_]usize{ 4, 2 }) |heads| {
        const l: KvLayout = .{ .heads = heads, .cap = cap, .int8 = true };
        try std.testing.expectEqual(heads * 256, l.row()); // Q27_KVROW: 1024 at TP 1, 512 at TP 2 (build/xpu.zig)
        const src = try gpa.alloc(u8, cap * heads * (256 + 4));
        defer gpa.free(src);
        const dst = try gpa.alloc(u8, src.len);
        defer gpa.free(dst);
        const snap = try gpa.alloc(u8, l.snapBytes(at));
        defer gpa.free(snap);
        kernelFill(src, l, at);
        saveRestore(src, dst, snap, l, at);
        for (0..at) |pos| for (0..heads) |h| {
            const v = (pos * heads + h) * 256;
            try std.testing.expectEqualSlices(u8, src[v..][0..256], dst[v..][0..256]);
            const s = cap * heads * 256 + (pos * heads + h) * 4; // the kernel's scale: cap * Q27_KVROW, then [pos * AKV + h]
            try std.testing.expectEqualSlices(u8, src[s..][0..4], dst[s..][0..4]);
        };
    }
}

test "the TP 2 int8 scales sit at cap * 512, not at TP 1's cap * 1024" {
    const tp1: KvLayout = .{ .heads = 4, .cap = 131072, .int8 = true };
    const tp2: KvLayout = .{ .heads = 2, .cap = 131072, .int8 = true };
    try std.testing.expectEqual(@as(usize, 131072 * 1024), tp1.scales());
    try std.testing.expectEqual(@as(usize, 16), tp1.scaleRow());
    try std.testing.expectEqual(@as(usize, 131072 * 512), tp2.scales());
    try std.testing.expectEqual(@as(usize, 8), tp2.scaleRow());
    try std.testing.expectEqual(@as(usize, 14336 * 520), tp2.snapBytes(14336));
    const bf: KvLayout = .{ .heads = 2, .cap = 4096, .int8 = false };
    try std.testing.expectEqual(@as(usize, 1024), bf.row());
    try std.testing.expectEqual(@as(usize, 10 * 1024), bf.snapBytes(10));
}

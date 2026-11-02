//! XPU_PREFILL_OVERLAP: a prompt chunk's all-reduce on the copy engine, hidden behind the other sub-chunk's compute.
//!
//! A prompt chunk at TP 2 has one chain: mixer in, mixer, mixer out, the ranks' sum, the residual add and norm,
//! gate|up, SwiGLU, down, the ranks' sum, the next layer's norm. Nothing inside one row's chain is independent of
//! its sum, so the only work that can run while the sum's traffic crosses PCIe is another row's chain. The plan
//! splits the chunk into two sub-chunks and runs the two chains interleaved: the push of one sub-chunk's partial
//! goes on the copy queue group while the other sub-chunk's kernels run on the compute queue.
//!
//! Both ranks read the flag and must agree on the plan: they run the same exchanges in the same order, so a rank
//! with a different split would pair its pushes with the wrong adds. `tp.Group.open` refuses to start when they
//! differ. Default off: with no env set the live path is unchanged.
const std = @import("std");

/// XPU_PREFILL_OVERLAP=1: the split prompt path (it also needs TF_TP_DEVFLAG=1: the add polls the peer's flag, so
/// no host round trip sits between the push and the add).
pub var on = false;

/// XPU_OVERLAP_ROWS: the first sub-chunk's rows. 0: half the chunk. The other sub-chunk takes the rest.
pub var rows: u32 = 0;

/// A GEMM row block (8 x GEMM_M 24) and a GDN chunk (64): a split boundary is a whole number of both, so no prompt
/// GEMM launch is a partial row block and the GDN state passes a chunk boundary as it does in one call.
pub const block: u32 = 192;

pub const Plan = struct { a: u32, b: u32 };

/// The plan for a chunk of `chunk` rows, or null when the chunk is too short to split (fewer than two blocks).
/// The first sub-chunk is at least half the chunk and a whole number of row blocks, so the second one is at most half:
/// that bounds the second activation set the caller allocates.
pub fn planFor(chunk: u32) ?Plan {
    if (chunk < 2 * block) return null;
    const half = std.math.divCeil(u32, chunk, 2) catch unreachable;
    var a = if (rows != 0) @min(rows, chunk - block) else chunk / 2;
    a = std.math.divCeil(u32, a, block) catch unreachable;
    a *= block;
    const h = (std.math.divCeil(u32, half, block) catch unreachable) * block;
    a = @max(a, h);
    if (a < block or a >= chunk) return null;
    const b = chunk - a;
    if (b < block) return null;
    return .{ .a = a, .b = b };
}

/// Reads the env once, at Runner.init (the flags do not change while a runner lives).
pub fn read() void {
    on = std.c.getenv("XPU_PREFILL_OVERLAP") != null;
    rows = 0;
    if (std.c.getenv("XPU_OVERLAP_ROWS")) |v| rows = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
}

test "a chunk splits into two whole row blocks" {
    // 1,152 rows: 6 blocks, 3 and 3
    try std.testing.expectEqual(Plan{ .a = 576, .b = 576 }, planFor(1152).?);
    // 896 rows: 2.33 blocks rounded up to 3, then the rest
    try std.testing.expectEqual(Plan{ .a = 576, .b = 320 }, planFor(896).?);
    // 1,024 rows
    try std.testing.expectEqual(Plan{ .a = 576, .b = 448 }, planFor(1024).?);
    // every sub-chunk is a multiple of the GDN chunk
    for ([_]u32{ 320, 448, 576 }) |n| try std.testing.expectEqual(@as(u32, 0), n % 64);
    // too short to split
    try std.testing.expect(planFor(320) == null);
    try std.testing.expect(planFor(16) == null);
}

test "the second sub-chunk is at most half the chunk, so one activation set of half plus 64 rows holds it" {
    for (384..1281) |n| {
        const p = planFor(@intCast(n)) orelse continue;
        try std.testing.expectEqual(@as(u32, @intCast(n)), p.a + p.b);
        try std.testing.expect(p.a >= std.math.divCeil(u32, @intCast(n), 2) catch unreachable);
        try std.testing.expect(p.b <= @as(u32, @intCast(n)) / 2 + 64);
        // the boundary is a whole row block (and so a whole GDN chunk); the rest of the chunk is whatever the
        // prompt leaves, so it need not be a multiple of 64: the GDN pads its last chunk with neutral rows
        try std.testing.expect(p.a % block == 0);
        try std.testing.expect(p.b >= block);
    }
}

test "an explicit first sub-chunk is rounded up and never below half" {
    rows = 500;
    defer rows = 0;
    try std.testing.expectEqual(Plan{ .a = 576, .b = 576 }, planFor(1152).?);
    rows = 1000;
    try std.testing.expectEqual(Plan{ .a = 960, .b = 192 }, planFor(1152).?);
    rows = 100; // below half: half the chunk, rounded up
    try std.testing.expectEqual(Plan{ .a = 576, .b = 576 }, planFor(1152).?);
}

test "the largest second sub-chunk a 1,280-row forward can ask for" {
    var worst: u32 = 0;
    for (384..1281) |n| {
        const p = planFor(@intCast(n)) orelse continue;
        worst = @max(worst, p.b);
    }
    try std.testing.expect(worst <= 576);
    try std.testing.expect(worst <= 1280 / 2 + 64);
}

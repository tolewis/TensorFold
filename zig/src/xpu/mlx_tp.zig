//! Rank slices of an MLX affine 4/64 matrix, on the host (pure: no device, no card).
//! A stored matrix is N rows of K inputs: words [N, K/8] (u32), scales and biases [N, K/64] (bf16).
//! A column split of the matmul takes whole rows. A row split takes a K range on 64-input group boundaries.
const std = @import("std");

pub const GS = 64;

/// Rows [n0, n0 + n) and inputs [k0, k0 + k) of a stored matrix. `n == 0`: all rows. `k == 0`: all inputs.
pub const Cut = struct {
    n0: u32 = 0,
    n: u32 = 0,
    k0: u32 = 0,
    k: u32 = 0,

    /// Output rows [n0, n0 + n): a column split of the matmul.
    pub fn cols(n0: u32, n: u32) Cut {
        return .{ .n0 = n0, .n = n };
    }
    /// Inputs [k0, k0 + k): a row split of the matmul (the rank's output is a partial sum).
    pub fn rows(k0: u32, k: u32) Cut {
        return .{ .k0 = k0, .k = k };
    }
};

/// The resolved cut of an `n_all` x `k_all` matrix.
pub const Span = struct { n0: u32, n: u32, k0: u32, k: u32 };

pub fn span(c: Cut, n_all: u32, k_all: u32) !Span {
    const s: Span = .{ .n0 = if (c.n == 0) 0 else c.n0, .n = if (c.n == 0) n_all else c.n, .k0 = if (c.k == 0) 0 else c.k0, .k = if (c.k == 0) k_all else c.k };
    if (@as(u64, s.n0) + s.n > n_all or @as(u64, s.k0) + s.k > k_all) return error.CutOutOfMatrix;
    // a K range starts and ends on a group boundary, so a whole group of scales and biases goes with it
    if (s.k0 % GS != 0 or s.k % GS != 0 or k_all % GS != 0) return error.CutOffGroupBoundary;
    return s;
}

pub fn wordBytes(n: u32, k: u32) usize {
    return @as(usize, n) * (k / 8) * 4;
}
pub fn groupBytes(n: u32, k: u32) usize {
    return @as(usize, n) * (k / GS) * 2;
}

/// The slice of (`w`, `s`, `b`), a stored `n_all` x `k_all` matrix, into `ow`, `os`, `ob` (sized by wordBytes and groupBytes).
pub fn slice(ow: []u8, os: []u8, ob: []u8, w: []const u8, s: []const u8, b: []const u8, n_all: u32, k_all: u32, c: Cut) !void {
    const sp = try span(c, n_all, k_all);
    if (w.len != wordBytes(n_all, k_all) or s.len != groupBytes(n_all, k_all) or b.len != s.len) return error.UnexpectedShape;
    if (ow.len != wordBytes(sp.n, sp.k) or os.len != groupBytes(sp.n, sp.k) or ob.len != os.len) return error.UnexpectedShape;
    const wr_all = @as(usize, k_all / 8) * 4;
    const gr_all = @as(usize, k_all / GS) * 2;
    const wr = @as(usize, sp.k / 8) * 4;
    const gr = @as(usize, sp.k / GS) * 2;
    for (0..sp.n) |r| {
        const src = @as(usize, sp.n0) + r;
        @memcpy(ow[r * wr ..][0..wr], w[src * wr_all + @as(usize, sp.k0 / 8) * 4 ..][0..wr]);
        @memcpy(os[r * gr ..][0..gr], s[src * gr_all + @as(usize, sp.k0 / GS) * 2 ..][0..gr]);
        @memcpy(ob[r * gr ..][0..gr], b[src * gr_all + @as(usize, sp.k0 / GS) * 2 ..][0..gr]);
    }
}

fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}
fn fromBf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Reference: y[r] = sum over the matrix's inputs of x[k0 + j] * (q * scale + bias), rows 0 .. n of the (sliced) matrix.
/// `x` starts at the matrix's first input.
pub fn refMatvec(y: []f64, x: []const f32, w: []const u8, s: []const u8, b: []const u8, n: u32, k: u32) void {
    for (0..n) |r| {
        var acc: f64 = 0;
        for (0..k) |j| {
            const word = std.mem.readInt(u32, w[(r * (k / 8) + j / 8) * 4 ..][0..4], .little);
            const qv: f32 = @floatFromInt((word >> @intCast(4 * (j % 8))) & 0xF);
            const g = r * (k / GS) + j / GS;
            const sc = fromBf16(std.mem.readInt(u16, s[g * 2 ..][0..2], .little));
            const bi = fromBf16(std.mem.readInt(u16, b[g * 2 ..][0..2], .little));
            acc += @as(f64, x[j]) * @as(f64, qv * sc + bi);
        }
        y[r] = acc;
    }
}

const test_n = 48;
const test_k = 512;

fn synthetic(a: std.mem.Allocator, w: *[]u8, s: *[]u8, b: *[]u8) !void {
    w.* = try a.alloc(u8, wordBytes(test_n, test_k));
    s.* = try a.alloc(u8, groupBytes(test_n, test_k));
    b.* = try a.alloc(u8, groupBytes(test_n, test_k));
    var prng = std.Random.DefaultPrng.init(0x4d4c58);
    const rnd = prng.random();
    rnd.bytes(w.*);
    for (0..s.len / 2) |i| {
        std.mem.writeInt(u16, s.*[i * 2 ..][0..2], bf16(@as(f32, @floatFromInt(rnd.intRangeAtMost(i32, 1, 8))) / 16.0), .little);
        std.mem.writeInt(u16, b.*[i * 2 ..][0..2], bf16(@as(f32, @floatFromInt(rnd.intRangeAtMost(i32, -8, 8))) / 8.0), .little);
    }
}

test "mlx tp slice: a column split of two ranks concatenates to the full matvec" {
    const a = std.testing.allocator;
    var w: []u8 = undefined;
    var s: []u8 = undefined;
    var b: []u8 = undefined;
    try synthetic(a, &w, &s, &b);
    defer a.free(w);
    defer a.free(s);
    defer a.free(b);
    var x: [test_k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 4.0;
    var full: [test_n]f64 = undefined;
    refMatvec(&full, &x, w, s, b, test_n, test_k);
    const half = test_n / 2;
    for (0..2) |rank| {
        const r: u32 = @intCast(rank);
        const ow = try a.alloc(u8, wordBytes(half, test_k));
        const os = try a.alloc(u8, groupBytes(half, test_k));
        const ob = try a.alloc(u8, groupBytes(half, test_k));
        defer a.free(ow);
        defer a.free(os);
        defer a.free(ob);
        try slice(ow, os, ob, w, s, b, test_n, test_k, Cut.cols(r * half, half));
        var y: [half]f64 = undefined;
        refMatvec(&y, &x, ow, os, ob, half, test_k);
        for (y, full[rank * half ..][0..half]) |got, want| try std.testing.expectEqual(want, got);
    }
}

test "mlx tp slice: a row split of two ranks sums to the full matvec" {
    const a = std.testing.allocator;
    var w: []u8 = undefined;
    var s: []u8 = undefined;
    var b: []u8 = undefined;
    try synthetic(a, &w, &s, &b);
    defer a.free(w);
    defer a.free(s);
    defer a.free(b);
    var x: [test_k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 4.0;
    var full: [test_n]f64 = undefined;
    refMatvec(&full, &x, w, s, b, test_n, test_k);
    var sum: [test_n]f64 = @splat(0);
    const half = test_k / 2; // 256 = four groups
    for (0..2) |rank| {
        const r: u32 = @intCast(rank);
        const ow = try a.alloc(u8, wordBytes(test_n, half));
        const os = try a.alloc(u8, groupBytes(test_n, half));
        const ob = try a.alloc(u8, groupBytes(test_n, half));
        defer a.free(ow);
        defer a.free(os);
        defer a.free(ob);
        try slice(ow, os, ob, w, s, b, test_n, test_k, Cut.rows(r * half, half));
        var y: [test_n]f64 = undefined;
        refMatvec(&y, x[r * half ..][0..half], ow, os, ob, test_n, half);
        for (&sum, y) |*t, v| t.* += v;
    }
    // values are multiples of 2^-6: the f64 sums are exact in any order
    for (sum, full) |got, want| try std.testing.expectEqual(want, got);
}

test "mlx tp slice: a split off a 64-input group boundary is refused" {
    const a = std.testing.allocator;
    var w: []u8 = undefined;
    var s: []u8 = undefined;
    var b: []u8 = undefined;
    try synthetic(a, &w, &s, &b);
    defer a.free(w);
    defer a.free(s);
    defer a.free(b);
    var ow: [0]u8 = .{};
    try std.testing.expectError(error.CutOffGroupBoundary, slice(&ow, &ow, &ow, w, s, b, test_n, test_k, Cut.rows(32, 64)));
    try std.testing.expectError(error.CutOffGroupBoundary, slice(&ow, &ow, &ow, w, s, b, test_n, test_k, Cut.rows(0, 96)));
    try std.testing.expectError(error.CutOutOfMatrix, slice(&ow, &ow, &ow, w, s, b, test_n, test_k, Cut.cols(40, 16)));
}

test "mlx tp slice: every TP 2 split of Qwen3.8-27B lies on a group boundary" {
    // [stored rows, stored inputs, cut] for rank 0 and rank 1
    const D = 5120;
    const cases = [_]struct { n: u32, k: u32, c: [2]Cut }{
        .{ .n = 12288, .k = D, .c = .{ Cut.cols(0, 6144), Cut.cols(6144, 6144) } }, // q_proj, q | gate per head
        .{ .n = 1024, .k = D, .c = .{ Cut.cols(0, 512), Cut.cols(512, 512) } }, // k_proj, v_proj
        .{ .n = 5120, .k = 6144, .c = .{ Cut.rows(0, 3072), Cut.rows(3072, 3072) } }, // o_proj, GDN out_proj
        .{ .n = 17408, .k = D, .c = .{ Cut.cols(0, 8704), Cut.cols(8704, 8704) } }, // gate, up
        .{ .n = 5120, .k = 17408, .c = .{ Cut.rows(0, 8704), Cut.rows(8704, 8704) } }, // down
        .{ .n = 10240, .k = D, .c = .{ Cut.cols(0, 1024), Cut.cols(1024, 1024) } }, // in_proj_qkv, q
        .{ .n = 10240, .k = D, .c = .{ Cut.cols(2048, 1024), Cut.cols(3072, 1024) } }, // k
        .{ .n = 10240, .k = D, .c = .{ Cut.cols(4096, 3072), Cut.cols(7168, 3072) } }, // v
        .{ .n = 6144, .k = D, .c = .{ Cut.cols(0, 3072), Cut.cols(3072, 3072) } }, // in_proj_z
        .{ .n = 48, .k = D, .c = .{ Cut.cols(0, 24), Cut.cols(24, 24) } }, // in_proj_a, in_proj_b
    };
    for (cases) |cs| for (cs.c) |c| {
        const sp = try span(c, cs.n, cs.k);
        try std.testing.expect(sp.k % GS == 0 and sp.k0 % GS == 0);
    }
}

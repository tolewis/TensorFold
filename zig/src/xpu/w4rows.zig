//! The int4 many-row decode (XPU_ROWS_MAX above 16): which kernel serves a row count, and a CPU reference of the
//! kernel's tiling. Lane p38-w4rows. No card: every test here runs on the host.

const std = @import("std");

/// Rows of one decode window: today's kernel, and the launch it takes.
pub const window = 16;
/// The most rows one work-group of the many-row kernel holds (RB 4 x 16).
pub const rows_cap = 64;
/// Columns a work-group of each kernel covers: SG_N 32 x 4 sub-groups, SG_N 16 x 4.
pub const columns16 = 128;
pub const columns32 = 128;
pub const columns64 = 64;

pub const Kernel = enum {
    /// w4k1_decode: 16 rows a sub-group, 32 columns a sub-group, 4 sub-groups a work-group.
    decode16,
    /// w4k1_mr32: two row blocks of 16 rows from one weight read.
    mr32,
    /// w4k1_mr64: four row blocks of 16 rows from one weight read.
    mr64,
    /// The nGEN prompt GEMM (above rows_max, or whenever the many-row path is off).
    prompt,
};

pub const Plan = struct {
    kernel: Kernel,
    /// Row blocks of 16 rows a work-group (1 for the 16-row kernel, 0 for the prompt GEMM).
    rb: u32,
    /// Columns a sub-group holds.
    sg_n: u32,
    /// The decode launch's grid: x, y, z. The prompt GEMM keeps its own grid (gemmRows).
    grid: [3]u32,
};

/// The int4 kernel for `m` rows of an `n`-column GEMM at `rows_max`. Up to 16 rows is today's kernel and launch.
/// From 17 to rows_max it is the many-row decode, two row blocks up to 32 rows and four above. Above rows_max, or
/// with the cap at 16, it is the prompt GEMM, as today.
pub fn plan(m: u32, n: u32, rows_max: u32) Plan {
    if (m <= window) return .{ .kernel = .decode16, .rb = 1, .sg_n = 32, .grid = .{ 1, n / columns16, 1 } };
    if (rows_max > window and m <= rows_max) {
        if (m <= 32) return .{ .kernel = .mr32, .rb = 2, .sg_n = 32, .grid = .{ 1, n / columns32, 1 } };
        return .{ .kernel = .mr64, .rb = 4, .sg_n = 16, .grid = .{ 1, n / columns64, 1 } };
    }
    return .{ .kernel = .prompt, .rb = 0, .sg_n = 0, .grid = .{ 0, 0, 0 } };
}

/// Rows a launch of the decode path covers: the row blocks x 16.
pub fn coveredRows(p: Plan) u32 {
    return if (p.rb == 0) 0 else p.rb * window;
}

/// y[m][n] = sum_k a[m][k] * (q[k][n] * s[g][n] + b[g][n]), the plain dequantized matvec of MLX 4/64 or GPTQ
/// 4/128, in fp32. q holds one 4-bit value a byte, g = k / gsz.
pub fn refMatvec(a: []const f32, q: []const u8, s: []const f32, b: []const f32, y: []f32, m: u32, n: u32, k: u32, gsz: u32) void {
    std.debug.assert(a.len == m * k and q.len == k * n and y.len == m * n);
    const groups = k / gsz;
    std.debug.assert(s.len == groups * n and b.len == groups * n);
    for (0..m) |r| for (0..n) |c| {
        var acc: f32 = 0;
        for (0..k) |kk| {
            const g = kk / gsz;
            const w = @as(f32, @floatFromInt(q[kk * n + c])) * s[g * n + c] + b[g * n + c];
            acc += a[r * k + kk] * w;
        }
        y[r * n + c] = acc;
    }
}

/// The same product as the kernel's tiling walks it: one column tile a work-group (`columns` wide), one row block
/// of 16 rows a sub-group, `rb` blocks, the groups in order. Every (row, column) must be written exactly once.
pub fn tiledMatvec(a: []const f32, q: []const u8, s: []const f32, b: []const f32, y: []f32, writes: []u32, m: u32, n: u32, k: u32, gsz: u32, columns: u32, rb: u32) void {
    std.debug.assert(a.len == m * k and q.len == k * n and y.len == m * n and writes.len == m * n);
    @memset(y, 0);
    @memset(writes, 0);
    const groups = k / gsz;
    var n0: u32 = 0;
    while (n0 < n) : (n0 += columns) {
        const wcols = @min(columns, n - n0);
        var rb0: u32 = 0;
        while (rb0 < rb) : (rb0 += 1) {
            const m0 = rb0 * window;
            for (0..window) |ri| {
                const r = m0 + ri;
                if (r >= m) continue;
                for (0..wcols) |ci| {
                    const c = n0 + ci;
                    var acc: f32 = 0;
                    for (0..groups) |g| for (0..gsz) |u| {
                        const kk = g * gsz + u;
                        const w = @as(f32, @floatFromInt(q[kk * n + c])) * s[g * n + c] + b[g * n + c];
                        acc += a[r * k + kk] * w;
                    }
                    y[r * n + c] = acc;
                    writes[r * n + c] += 1;
                }
            }
        }
    }
}

fn fill(a: []f32, q: []u8, s: []f32, b: []f32, seed: u64) void {
    var rnd = std.Random.DefaultPrng.init(seed);
    const r = rnd.random();
    for (a) |*v| v.* = r.float(f32) * 2 - 1;
    for (q) |*v| v.* = r.int(u4);
    for (s) |*v| v.* = (r.float(f32) * 2 - 1) * 0.25;
    for (b) |*v| v.* = (r.float(f32) * 2 - 1) * 0.5;
}

test "w4rows plan: today's kernel up to 16 rows, the many-row decode from 17 to the cap, the prompt GEMM above" {
    // a TP 2 rank's shapes: K 3072 and 8704, N 7168, 8448, 17408
    const shapes = [_][2]u32{ .{ 3072, 7168 }, .{ 8704, 8448 }, .{ 8704, 17408 } };
    for (shapes) |sh| {
        const p1 = plan(1, sh[1], 16);
        try std.testing.expectEqual(Kernel.decode16, p1.kernel);
        try std.testing.expectEqual(@as(u32, 1), p1.grid[0]);
        try std.testing.expectEqual(sh[1] / 128, p1.grid[1]);
        try std.testing.expectEqual(Kernel.decode16, plan(16, sh[1], 64).kernel);
        // the cap at 16: above it the prompt GEMM, as today
        try std.testing.expectEqual(Kernel.prompt, plan(17, sh[1], 16).kernel);
        try std.testing.expectEqual(Kernel.prompt, plan(64, sh[1], 16).kernel);
        // the cap at 64: the many-row decode
        const p17 = plan(17, sh[1], 64);
        try std.testing.expectEqual(Kernel.mr32, p17.kernel);
        try std.testing.expectEqual(@as(u32, 2), p17.rb);
        try std.testing.expectEqual(@as(u32, 32), coveredRows(p17));
        try std.testing.expectEqual(Kernel.mr32, plan(32, sh[1], 64).kernel);
        const p33 = plan(33, sh[1], 64);
        try std.testing.expectEqual(Kernel.mr64, p33.kernel);
        try std.testing.expectEqual(@as(u32, 4), p33.rb);
        try std.testing.expectEqual(@as(u32, 64), coveredRows(p33));
        try std.testing.expectEqual(sh[1] / 64, p33.grid[1]);
        try std.testing.expectEqual(Kernel.mr64, plan(48, sh[1], 64).kernel);
        try std.testing.expectEqual(Kernel.mr64, plan(64, sh[1], 64).kernel);
        // the cap at 32: 33 rows and up take the prompt GEMM
        try std.testing.expectEqual(Kernel.mr32, plan(32, sh[1], 32).kernel);
        try std.testing.expectEqual(Kernel.prompt, plan(33, sh[1], 32).kernel);
        // above the cap the prompt GEMM whatever the cap
        try std.testing.expectEqual(Kernel.prompt, plan(65, sh[1], 64).kernel);
    }
}

test "w4rows tiling: the many-row walk covers every element once and equals the plain matvec" {
    // small shapes that keep the real divisibility: K a multiple of 64, N a multiple of 128
    const cases = [_]struct { m: u32, n: u32, k: u32, gsz: u32, columns: u32, rb: u32 }{
        .{ .m = 1, .n = 256, .k = 256, .gsz = 64, .columns = 128, .rb = 1 },
        .{ .m = 16, .n = 256, .k = 256, .gsz = 64, .columns = 128, .rb = 1 },
        .{ .m = 17, .n = 256, .k = 256, .gsz = 64, .columns = 128, .rb = 2 },
        .{ .m = 32, .n = 256, .k = 256, .gsz = 64, .columns = 128, .rb = 2 },
        .{ .m = 48, .n = 256, .k = 256, .gsz = 64, .columns = 64, .rb = 4 },
        .{ .m = 64, .n = 256, .k = 256, .gsz = 64, .columns = 64, .rb = 4 },
        // GPTQ 4/128 at a TP 2 K
        .{ .m = 64, .n = 256, .k = 384, .gsz = 128, .columns = 64, .rb = 4 },
    };
    for (cases, 0..) |cs, ci| {
        const alloc = std.testing.allocator;
        const a = try alloc.alloc(f32, cs.m * cs.k);
        defer alloc.free(a);
        const q = try alloc.alloc(u8, cs.k * cs.n);
        defer alloc.free(q);
        const s = try alloc.alloc(f32, cs.k / cs.gsz * cs.n);
        defer alloc.free(s);
        const b = try alloc.alloc(f32, cs.k / cs.gsz * cs.n);
        defer alloc.free(b);
        const y1 = try alloc.alloc(f32, cs.m * cs.n);
        defer alloc.free(y1);
        const y2 = try alloc.alloc(f32, cs.m * cs.n);
        defer alloc.free(y2);
        const writes = try alloc.alloc(u32, cs.m * cs.n);
        defer alloc.free(writes);
        fill(a, q, s, b, 0x5151 + ci);
        refMatvec(a, q, s, b, y1, cs.m, cs.n, cs.k, cs.gsz);
        tiledMatvec(a, q, s, b, y2, writes, cs.m, cs.n, cs.k, cs.gsz, cs.columns, cs.rb);
        for (writes) |w| try std.testing.expectEqual(@as(u32, 1), w);
        for (y1, y2) |x, z| try std.testing.expectApproxEqRel(x, z, 1e-6);
    }
}

test "w4rows tiling: a missing row block or column tile is caught" {
    const alloc = std.testing.allocator;
    const m: u32 = 32;
    const n: u32 = 256;
    const k: u32 = 256;
    const a = try alloc.alloc(f32, m * k);
    defer alloc.free(a);
    const q = try alloc.alloc(u8, k * n);
    defer alloc.free(q);
    const s = try alloc.alloc(f32, k / 64 * n);
    defer alloc.free(s);
    const b = try alloc.alloc(f32, k / 64 * n);
    defer alloc.free(b);
    const y = try alloc.alloc(f32, m * n);
    defer alloc.free(y);
    const writes = try alloc.alloc(u32, m * n);
    defer alloc.free(writes);
    fill(a, q, s, b, 7);
    // one row block instead of two: rows 16 to 31 are never written
    tiledMatvec(a, q, s, b, y, writes, m, n, k, 64, 128, 1);
    for (writes, 0..) |w, i| try std.testing.expectEqual(@as(u32, if (i / n < 16) 1 else 0), w);
    // the full walk writes every row
    tiledMatvec(a, q, s, b, y, writes, m, n, k, 64, 128, 2);
    for (writes) |w| try std.testing.expectEqual(@as(u32, 1), w);
}

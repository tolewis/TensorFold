//! The exl3_matvec builds in the binary: one SPIR-V image per (bits, rows, NS, HS), and the rule that picks one.
//! The prompt int8 images (exl3_int8.cl) are one a width, and the fused routed MLP images (exl3_moe) one a width.
//! zig/build/xpu.zig and zig/src/xpu/exl3.zig both import this file, so the images and the picks agree.

const std = @import("std");

pub const Variant = struct { bits: u32, rows: u32, ns: u32, hs: bool };

/// Sage EXL3: 4 bits (language layers and MTP), 6 bits (vision and lm_head).
pub const bits = [_]u32{ 4, 6 };
/// Row buckets: a launch of r rows runs the smallest bucket >= r. 32 is lane p25-mtp32's verify window
/// (XPU_VERIFY_ROWS=32): the 24-row window runs the 32-row image (the next bucket up).
pub const rows = [_]u32{ 1, 2, 4, 8, 16, 32 };
pub const rows_max = rows[rows.len - 1];
const strips_per_group = [_]u32{ 1, 2, 4 };

/// HS (split half) at 2 to 4 rows; it needs two strips a sub-group.
fn hsFor(r: u32, ns: u32) bool {
    return r >= 2 and r <= 4 and ns >= 2;
}

/// Every image: bits x row buckets x NS 1, 2, 4.
pub const all = blk: {
    var out: [bits.len * rows.len * strips_per_group.len]Variant = undefined;
    var i: usize = 0;
    for (bits) |b| for (rows) |r| for (strips_per_group) |ns| {
        out[i] = .{ .bits = b, .rows = r, .ns = ns, .hs = hsFor(r, ns) };
        i += 1;
    };
    break :blk out;
};

/// exl3_mv2: the 1 row matvec with the trellis tile staged in local memory (XPU_EXL3_MV2, lane p29-exl3bw): one
/// image a (bits, NS, KU), the NS strips and KU k tiles of a sub-group against the staging. The host picks one
/// when rows are 1 and the group is wide.
pub const mv2 = struct {
    pub const mbits = [_]u32{ 4, 6 };
    pub const mns = [_]u32{ 2, 4, 8 };
    pub const mku = [_]u32{ 1, 2, 4 };

    pub const count = mbits.len * mns.len * mku.len;

    pub fn index(b: u32, n: u32, k: u32) u32 {
        const bi = std.mem.indexOfScalar(u32, &mbits, b) orelse unreachable;
        const ni = std.mem.indexOfScalar(u32, &mns, n) orelse unreachable;
        const ki = std.mem.indexOfScalar(u32, &mku, k) orelse unreachable;
        return (bi * mns.len + ni) * mku.len + ki;
    }

    pub fn nsOf(v: u32) u32 {
        return mns[(v / mku.len) % mns.len];
    }

    pub fn name(comptime v: u32) []const u8 {
        return std.fmt.comptimePrint("exl3_mv2_b{d}_ns{d}_ku{d}", .{ mbits[v / (mns.len * mku.len)], mns[(v / mku.len) % mns.len], v % mku.len });
    }

    pub fn options(comptime v: u32) []const u8 {
        return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DNS={d} -DKU={d}", .{ mbits[v / (mns.len * mku.len)], mns[(v / mku.len) % mns.len], v % mku.len });
    }
};

/// K splits for a launch of `sub_groups` sub-groups over groups whose K / 16 all divide by `k_tiles_gcd`.
/// Wide launches above wide_wave_rows stop at one wave (wide_wave); every other launch goes to sub_group_target.
pub fn split(sub_groups: u32, k_tiles_gcd: u32, is_wide: bool) u32 {
    var ks: u32 = 1;
    if (is_wide) {
        while (ks < ks_max and sub_groups * ks * 2 <= wide_wave and k_tiles_gcd % (2 * ks) == 0 and k_tiles_gcd / (2 * ks) >= ks_min_tiles) ks *= 2;
        return ks;
    }
    while (ks < ks_max and sub_groups * ks < sub_group_target and k_tiles_gcd % (2 * ks) == 0 and k_tiles_gcd / (2 * ks) >= ks_min_tiles) ks *= 2;
    return ks;
}

pub fn name(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("exl3_b{d}_r{d}_n{d}{s}", .{ v.bits, v.rows, v.ns, if (v.hs) "_hs" else "" });
}

/// K tiles a matvec loop step on the DPAS path (exl3.cl KU): that many tiles' loads are in flight before a decode.
pub const ku: u32 = 1;

pub fn options(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DROWS={d} -DNS={d} -DHS={d} -DKU={d}", .{ v.bits, v.rows, v.ns, @intFromBool(v.hs), ku });
}

/// The fused routed MLP (exl3_moe_*): one image a width. K splits and strips measured on the B70 with one Sage
/// layer (8 picks and the shared expert, 1 token) by tools/xpu/exl3_moe_check.py (2026-10-25).
pub const moe = struct {
    pub const bits = [_]u32{ 4, 6 };
    /// gate|up K splits (a work-group each), down K splits and down strips a sub-group.
    pub const p = 4;
    pub const pd = 2;
    pub const ds = 2;
    /// Largest K of gate|up (hidden) and of down (intermediate): local memory a work-group.
    pub const k_max = 4096;

    /// Prompt rows (P3.4, the same image): exl3.cl GEMM_M rows a sub-group and GEMM_S strips a work-group of
    /// exl3_gemm, GEMM_KC k tiles a local-memory step, TILE_NS strips a sub-group of exl3_moe_tile, and the most rows
    /// of a prompt launch. The kernel defaults are these values.
    /// 24 rows a sub-group and 4 k tiles a local-memory step since 2026-11-01 (was 16 and 8): on the 27B's 896-row
    /// chunks 2.6x on gate|up (14.3 to 5.6 ms a layer), 3.4x on down. 32 rows spill (13.6 ms), 8 rows and KC 16 lose:
    /// the decoded tiles in local memory are the bound, so more rows a sub-group and a smaller step both pay.
    /// Measured by tools/xpu/exl3_27b_bench.py at ROWS=896 (lanes/qwen27/OPT-matvec.md in Intel-B70-Recipe).
    pub const gemm_m = 24;
    pub const gemm_s = 4;
    pub const gemm_kc = 4;
    pub const tile_ns = 2;
    pub const tile_rows = 16;
    /// 1280 rows: q.maxRows() is XPU_CHUNK + 128, and 1152 (6 row blocks of 8 x GEMM_M 24, no partial block) is the
    /// largest chunk the 27B prompt path takes (lane p26-prefill, 2026-11-01). Was 1024, which the default chunk of
    /// 1024 rows already exceeded.
    pub const prompt_rows_max = 1280;

    pub fn name(comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("exl3_moe_b{d}", .{b});
    }

    /// exl3_gemm x planes: 1 = hi + lo fp16 (two DPAS a tile), 0 = fp16 x alone. Prompt rows only. 0 since
    /// 2026-11-01: 1.5x on the 27B's 896-row chunks (gu 14.3 to 9.6 ms, down 10.1 to 6.1), x in fp16 as the
    /// MLX prompt path has it; the 41-item eval gates the change.
    pub const gemm_lo = 0;

    pub fn options(comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DMOE_P={d} -DMOE_PD={d} -DMOE_DS={d} -DMOE_KMAX={d} -DGEMM_M={d} -DGEMM_S={d} -DGEMM_KC={d} -DGEMM_LO={d}", .{ b, p, pd, ds, k_max, gemm_m, gemm_s, gemm_kc, gemm_lo });
    }
};

/// The prompt int8 (W4A8 on decoded tiles) images of exl3_int8.cl: one a width, behind XPU_PROMPT_INT8
/// (exl3.encodeRows). Prompt rows only; the decode path never reads them.
pub const w4a8 = struct {
    pub const bits = [_]u32{ 4, 6 };
    /// Rows a sub-group, 16-column strips a work-group, 16-row k tiles a local-memory step. The row count matches
    /// exl3_gemm's measured tile (24 rows, 4 strips); the int8 DPAS takes two k tiles a step.
    pub const gemm_m = 24;
    pub const gemm_s = 4;
    pub const gemm_kc = 8;

    pub fn name(comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("exl3_int8_b{d}", .{b});
    }

    pub fn options(comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DGEMM_M={d} -DGEMM_S={d} -DGEMM_KC={d}", .{ b, gemm_m, gemm_s, gemm_kc });
    }
};

/// The prompt GEMM with the A tile in local memory (exl3_gemm2.cl, lane p26-gemm). Prompt rows only, behind
/// XPU_PROMPT_GEMM2=1, default off. One image a width and tile.
/// m: rows a sub-group, s: 16-column strips a sub-group, kc: k tiles a local-memory step. Three tiles to measure:
/// 0 is the safe one, 1 spends registers on rows (a lower decode cost a row block), 2 spends registers on column
/// reuse instead (fewer A reads a work-group). The host picks a tile by the flag's value: 1, 2 or 3.
pub const gemm2 = struct {
    pub const widths = [_]u32{ 4, 6 };
    pub const Tile = struct { m: u32, s: u32, kc: u32 };
    pub const tiles = [_]Tile{
        .{ .m = 24, .s = 8, .kc = 2 },
        .{ .m = 48, .s = 8, .kc = 2 },
        .{ .m = 24, .s = 16, .kc = 2 },
    };
    pub const count = tiles.len * widths.len;

    pub fn name(comptime v: usize, comptime b: u32) []const u8 {
        return std.fmt.comptimePrint("exl3_gemm2_t{d}_b{d}", .{ v, b });
    }

    pub fn options(comptime v: usize, comptime b: u32) []const u8 {
        const t = tiles[v];
        return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DGEMM2_M={d} -DGEMM2_S={d} -DGEMM2_KC={d}", .{ b, t.m, t.s, t.kc });
    }

    /// Local memory an image asks for: the A tile of a step (8 * m rows of 16 * kc halves), then the decoded B
    /// tiles (kc * s * 16 lanes of 32 bytes).
    pub fn localBytes(v: usize) usize {
        const t = tiles[v];
        return 8 * t.m * 16 * t.kc * 2 + t.kc * t.s * 16 * 32;
    }

    /// The image index of a tile and a width, as kernels.exl3_gemm2 lists them.
    pub fn index(v: usize, b: usize) usize {
        return v * widths.len + b;
    }
};

/// Sub-groups a launch needs to fill the GPU; fewer strips a sub-group and K splits get it there (narrow groups:
/// Sage experts, N 512 and 2048, measured 2026-10-24 and -25).
pub const sub_group_target = 4096;
/// Wide groups (every strip count at or above wide_strips: the 27B's projections, N 5120 to 17408, and lm_head).
/// Measured in the engine's window table on the B70 (the 27B, 2026-11-01, ms a window at 1, 8 and 16 rows):
///   NS 4, one wave: 73.0, 70.0, 89.6    NS 1, one wave: 59.5, 89.1, 121.2    NS 2, one wave: 56.5, 74.2, 89.5
///   NS 4, target:   71.2, 81.3, 100.4   NS 1, target:   51.8, 85.5, 112.4    NS 2, target:   53.6, 82.4, 99.7
/// So NS 1 and the sub-group target up to wide_wave_rows (the 1-row path: 51.8 against the 56.6 of the 2026-10-25
/// rule), NS 4 and one wave of sub-groups (wide_wave) at 8 rows, NS 2 and one wave at 16. The back-to-back loop of
/// tools/xpu/exl3_27b_bench.py had picked NS 4 and one wave at every row count: it overlaps one launch's tail with
/// the next, which the engine's dependent launches cannot (EXL3_SYNC=1 in the test program times them apart).
pub const wide_strips = 128;
pub const wide_wave = 1536; // 24 Xe cores x 8 XVE x 8 threads on the B70 (BMG G31 cut)
pub const wide_wave_rows = 4; // launches of more rows than this on wide groups stop at one wave
/// Most K splits (the sums scratch holds this many) and fewest 16-row K tiles a split.
pub const ks_max = 16;
const ks_min_tiles = 8;

/// The strip counts (N / 16) of a launch's groups: smallest, common divisor and total.
pub const Strips = struct { min: u32, gcd: u32, total: u32 };

/// The image for `r` rows of `b` bits. Measured on the B70 by tools/xpu/exl3_ab.py (2026-10-24 and -25), the rule
/// of Exl3.shape there: the per-group NS, then fewer strips a sub-group until the launch reaches
/// sub_group_target. Null: no image for these bits or more than rows_max rows.
pub fn pick(b: u32, r: u32, s: Strips) ?usize {
    const bucket = for (rows) |x| {
        if (x >= r) break x;
    } else return null;
    var ns: u32 = undefined;
    if (wide(s)) {
        // see wide_strips. At 16 rows the DPAS path holds two row blocks of accumulators a strip, and NS 4 spills:
        // gu 16 rows 442 us at NS 2 against 490 at NS 4, down 181 against 217 (sweep 3, 2026-11-01).
        ns = if (bucket <= wide_wave_rows) 1 else if (bucket <= 8) 4 else 2;
    } else {
        ns = if (s.min < 128) // few strips a group (expert gate/up, N 512): keep the sub-group count up
            (if (bucket <= 4) 2 else 1)
        else if (bucket <= 4) 4 else if (bucket <= 8) 2 else 1;
        while (ns > 1 and s.total / ns < sub_group_target) ns /= 2;
    }
    while (s.gcd % ns != 0) ns /= 2;
    for (all, 0..) |v, i| if (v.bits == b and v.rows == bucket and v.ns == ns) return i;
    return null;
}

/// Whether a launch's groups are all wide (see wide_strips).
pub fn wide(s: Strips) bool {
    return s.min >= wide_strips;
}

test "picks follow the measured rule" {
    const at = struct {
        fn v(b: u32, r: u32, smin: u32, sgcd: u32) Variant {
            return all[pick(b, r, .{ .min = smin, .gcd = sgcd, .total = smin * 320 }).?];
        }
    }.v;
    // lm_head: 15,520 strips, a wide group, so NS 1 at 1 to 4 rows as every wide group (the 2026-11-01 rule)
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 1, .ns = 1, .hs = false }, at(6, 1, 15520, 15520));
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 4, .ns = 1, .hs = false }, at(6, 3, 15520, 15520));
    // wide groups: NS 1 to 4 rows, NS 4 at 8, NS 2 at 16 (the 27B, 2026-11-01)
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 1, .ns = 1, .hs = false }, at(4, 1, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 4, .ns = 1, .hs = false }, at(4, 3, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 8, .ns = 4, .hs = false }, at(4, 5, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 16, .ns = 2, .hs = false }, at(4, 16, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 2, .ns = 2, .hs = true }, at(4, 2, 32, 32));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 1, .ns = 1, .hs = false }, at(4, 1, 33, 33));
    try std.testing.expectEqual(@as(?usize, null), pick(5, 1, .{ .min = 128, .gcd = 128, .total = 1024 }));
    try std.testing.expectEqual(@as(?usize, null), pick(4, 33, .{ .min = 128, .gcd = 128, .total = 1024 }));
    // the 32-row bucket: a 17-row or 24-row launch runs the 32-row image (lane p25-mtp32)
    try std.testing.expectEqual(@as(u32, 32), all[pick(4, 17, .{ .min = 128, .gcd = 128, .total = 1024 }).?].rows);
    try std.testing.expectEqual(@as(u32, 32), all[pick(4, 24, .{ .min = 128, .gcd = 128, .total = 1024 }).?].rows);
    try std.testing.expectEqual(@as(u32, 32), all[pick(6, 32, .{ .min = 15520, .gcd = 15520, .total = 15520 * 320 }).?].rows);
    // one layer's top-8: NS 1, then K splits up to the target
    try std.testing.expectEqual(@as(u32, 1), all[pick(4, 1, .{ .min = 32, .gcd = 32, .total = 256 }).?].ns);
    try std.testing.expectEqual(@as(u32, 16), split(256, 128, false));
    try std.testing.expectEqual(@as(u32, 4), split(1024, 32, false));
    try std.testing.expectEqual(@as(u32, 2), split(3880, 128, false));
    // the 27B at 8 rows and NS 4: gate|up 544 sub-groups, down 80 over 1088 k tiles, in_proj_qkv 160, lm_head 3880
    try std.testing.expectEqual(@as(u32, 2), split(544, 320, true));
    // the 27B at 1 row and NS 1: gate|up 2176 sub-groups to the target, down 320 to 16 splits
    try std.testing.expectEqual(@as(u32, 2), split(2176, 320, false));
    try std.testing.expectEqual(@as(u32, 16), split(320, 1088, false));
    try std.testing.expectEqual(@as(u32, 16), split(80, 1088, true));
    try std.testing.expectEqual(@as(u32, 8), split(160, 320, true));
    try std.testing.expectEqual(@as(u32, 1), split(3880, 320, true));
    // the last image is the widest bucket (32 rows, lane p25-mtp32)
    try std.testing.expectEqualStrings("exl3_b6_r32_n4", name(all[all.len - 1]));
}

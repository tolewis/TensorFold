//! The exl3_matvec builds in the binary: one SPIR-V image per (bits, rows, NS, HS), and the rule that picks one.
//! zig/build/xpu.zig and zig/src/xpu/exl3.zig both import this file, so the images and the picks agree.

const std = @import("std");

pub const Variant = struct { bits: u32, rows: u32, ns: u32, hs: bool };

/// Sage EXL3: 4 bits (language layers and MTP), 6 bits (vision and lm_head).
pub const bits = [_]u32{ 4, 6 };
/// Row buckets: a launch of r rows runs the smallest bucket >= r.
pub const rows = [_]u32{ 1, 2, 4, 8, 16 };
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

pub fn name(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("exl3_b{d}_r{d}_n{d}{s}", .{ v.bits, v.rows, v.ns, if (v.hs) "_hs" else "" });
}

pub fn options(comptime v: Variant) []const u8 {
    return std.fmt.comptimePrint("-cl-std=CL3.0 -DEXL3_ABS -DBITS={d} -DROWS={d} -DNS={d} -DHS={d}", .{ v.bits, v.rows, v.ns, @intFromBool(v.hs) });
}

/// The image for `r` rows of `b` bits over groups whose strip counts (N / 16) all divide by `strips_gcd`, the
/// smallest of them `strips_min`. Measured on the B70 by tools/xpu/exl3_ab.py (2026-10-24), the rule of
/// Exl3.ns and Exl3.hs there. Null: no image for these bits or more than rows_max rows.
pub fn pick(b: u32, r: u32, strips_min: u32, strips_gcd: u32) ?usize {
    const bucket = for (rows) |x| {
        if (x >= r) break x;
    } else return null;
    var ns: u32 = if (strips_min < 128) // few strips a group (expert gate/up, N 512): keep the sub-group count up
        (if (bucket <= 4) 2 else 1)
    else if (bucket <= 4) 4 else if (bucket <= 8) 2 else 1;
    while (strips_gcd % ns != 0) ns /= 2;
    for (all, 0..) |v, i| if (v.bits == b and v.rows == bucket and v.ns == ns) return i;
    return null;
}

test "picks follow the measured rule" {
    const at = struct {
        fn v(b: u32, r: u32, smin: u32, sgcd: u32) Variant {
            return all[pick(b, r, smin, sgcd).?];
        }
    }.v;
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 1, .ns = 4, .hs = false }, at(6, 1, 15520, 15520));
    try std.testing.expectEqual(Variant{ .bits = 6, .rows = 4, .ns = 4, .hs = true }, at(6, 3, 15520, 15520));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 8, .ns = 2, .hs = false }, at(4, 5, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 16, .ns = 1, .hs = false }, at(4, 16, 128, 128));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 2, .ns = 2, .hs = true }, at(4, 2, 32, 32));
    try std.testing.expectEqual(Variant{ .bits = 4, .rows = 1, .ns = 1, .hs = false }, at(4, 1, 33, 33));
    try std.testing.expectEqual(@as(?usize, null), pick(5, 1, 128, 128));
    try std.testing.expectEqual(@as(?usize, null), pick(4, 17, 128, 128));
    try std.testing.expectEqualStrings("exl3_b6_r4_n4_hs", name(all[all.len - 7]));
}

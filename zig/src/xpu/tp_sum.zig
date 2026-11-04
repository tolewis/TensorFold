//! The TP exchange's arithmetic: the ranks' partials summed in rank order in fp32, rounded once to the activation type.

const std = @import("std");

/// A partial's element type. `size` is the bytes an element takes in the exchange.
pub const DType = enum(u8) {
    f16,
    f32,

    pub fn size(t: DType) usize {
        return switch (t) {
            .f16 => 2,
            .f32 => 4,
        };
    }
};

pub const Op = enum(u8) { sum };

/// One rank's partial as f32 elements, added into `acc` (`count` elements of `in`).
fn addRank(src: []const u8, in: DType, count: usize, acc: []f32) void {
    switch (in) {
        .f16 => {
            for (0..count) |i| acc[i] += @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[i * 2 ..][0..2], .little))));
        },
        .f32 => {
            for (0..count) |i| acc[i] += @bitCast(std.mem.readInt(u32, src[i * 4 ..][0..4], .little));
        },
    }
}

/// `acc[0..count]` gets the sum of `ranks` (rank order, first to last) in fp32; `dst` gets it rounded once as `out`.
pub fn reduce(ranks: []const []const u8, in: DType, out: DType, count: usize, acc: []f32, dst: []u8) !void {
    if (ranks.len == 0) return error.NoRanks;
    if (acc.len < count or dst.len < count * out.size()) return error.ShortBuffer;
    @memset(acc[0..count], 0);
    for (ranks) |r| {
        if (r.len != count * in.size()) return error.PartialSize;
        addRank(r, in, count, acc);
    }
    switch (out) {
        .f16 => {
            for (0..count) |i| std.mem.writeInt(u16, dst[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(acc[i]))), .little);
        },
        .f32 => {
            for (0..count) |i| std.mem.writeInt(u32, dst[i * 4 ..][0..4], @bitCast(acc[i]), .little);
        },
    }
}

test "tp sum: both ranks hold the same bits after the sum" {
    // each rank holds its own partial; both call the sum with the partials in rank order and write the same bytes
    var p0: [8]u8 = undefined;
    var p1: [8]u8 = undefined;
    for ([_]f16{ 1.0, 2.0, -3.5, 7.25 }, 0..) |v, i| {
        std.mem.writeInt(u16, p0[i * 2 ..][0..2], @bitCast(v), .little);
        std.mem.writeInt(u16, p1[i * 2 ..][0..2], @bitCast(v * 0.5), .little);
    }
    var acc: [4]f32 = undefined;
    var a: [8]u8 = undefined;
    var b: [8]u8 = undefined;
    try reduce(&.{ &p0, &p1 }, .f16, .f16, 4, &acc, &a);
    try reduce(&.{ &p0, &p1 }, .f16, .f16, 4, &acc, &b);
    try std.testing.expectEqualSlices(u8, &a, &b);
    try std.testing.expectEqual(@as(f16, 1.5), @as(f16, @bitCast(std.mem.readInt(u16, a[0..2], .little))));
}

test "tp sum: an fp32 partial rounds once, not per rank" {
    var p0: [8]u8 = undefined;
    var p1: [8]u8 = undefined;
    std.mem.writeInt(u32, p0[0..4], @bitCast(@as(f32, 0.1)), .little);
    std.mem.writeInt(u32, p0[4..8], @bitCast(@as(f32, 3.0)), .little);
    std.mem.writeInt(u32, p1[0..4], @bitCast(@as(f32, 0.2)), .little);
    std.mem.writeInt(u32, p1[4..8], @bitCast(@as(f32, -1.0)), .little);
    var acc: [2]f32 = undefined;
    var out: [4]u8 = undefined;
    try reduce(&.{ &p0, &p1 }, .f32, .f16, 2, &acc, &out);
    // 0.1f + 0.2f in fp32, rounded once to fp16: the fp16 sum of two fp16 partials would be 0.30004883
    const s = @as(f32, 0.1) + @as(f32, 0.2);
    try std.testing.expectEqual(@as(f16, @floatCast(s)), @as(f16, @bitCast(std.mem.readInt(u16, out[0..2], .little))));
    try std.testing.expectEqual(@as(f32, 2.0), acc[1]);
}

test "tp sum: a NaN partial stays a NaN on both ranks" {
    var p0: [8]u8 = undefined;
    var p1: [8]u8 = undefined;
    std.mem.writeInt(u32, p0[0..4], @bitCast(std.math.nan(f32)), .little);
    std.mem.writeInt(u32, p0[4..8], @bitCast(@as(f32, 2.0)), .little);
    std.mem.writeInt(u32, p1[0..4], @bitCast(@as(f32, 1.0)), .little);
    std.mem.writeInt(u32, p1[4..8], @bitCast(@as(f32, 3.0)), .little);
    var acc: [2]f32 = undefined;
    var out: [4]u8 = undefined;
    try reduce(&.{ &p0, &p1 }, .f32, .f16, 2, &acc, &out);
    try std.testing.expect(std.math.isNan(@as(f16, @bitCast(std.mem.readInt(u16, out[0..2], .little)))));
    try std.testing.expectEqual(@as(f16, 5.0), @as(f16, @bitCast(std.mem.readInt(u16, out[2..4], .little))));
    // the fp32 store keeps the NaN too
    var out32: [8]u8 = undefined;
    try reduce(&.{ &p0, &p1 }, .f32, .f32, 2, &acc, &out32);
    try std.testing.expect(std.math.isNan(@as(f32, @bitCast(std.mem.readInt(u32, out32[0..4], .little)))));
}

test "tp sum: an fp32 partial keeps a term an fp16 partial loses" {
    // as fp32 partials the sum rounds once to 1.0009765625; rounded to fp16 first, the tie rounds to 1.001953125
    const a: f32 = 1.0005859375;
    const b: f32 = 0.0005859375;
    var pa: [4]u8 = undefined;
    var pb: [4]u8 = undefined;
    std.mem.writeInt(u32, pa[0..4], @bitCast(a), .little);
    std.mem.writeInt(u32, pb[0..4], @bitCast(b), .little);
    var acc: [1]f32 = undefined;
    var out: [2]u8 = undefined;
    try reduce(&.{ &pa, &pb }, .f32, .f16, 1, &acc, &out);
    try std.testing.expectEqual(@as(f16, 1.0009765625), @as(f16, @bitCast(std.mem.readInt(u16, out[0..2], .little))));
    var ha: [2]u8 = undefined;
    var hb: [2]u8 = undefined;
    std.mem.writeInt(u16, ha[0..2], @bitCast(@as(f16, @floatCast(a))), .little);
    std.mem.writeInt(u16, hb[0..2], @bitCast(@as(f16, @floatCast(b))), .little);
    try reduce(&.{ &ha, &hb }, .f16, .f16, 1, &acc, &out);
    try std.testing.expectEqual(@as(f16, 1.001953125), @as(f16, @bitCast(std.mem.readInt(u16, out[0..2], .little))));
}

test "tp sum: the reduce refuses a partial of the wrong size" {
    var p0: [4]u8 = undefined;
    var acc: [2]f32 = undefined;
    var out: [4]u8 = undefined;
    try std.testing.expectError(error.PartialSize, reduce(&.{&p0}, .f32, .f16, 2, &acc, &out));
    try std.testing.expectError(error.NoRanks, reduce(&[_][]const u8{}, .f32, .f16, 2, &acc, &out));
}

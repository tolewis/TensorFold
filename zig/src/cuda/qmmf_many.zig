//! Decode projections of one input in one launch (qmmf.cu parts_kernel). Each part keeps the bits of its own call.
const std = @import("std");
const q = @import("qmmf.zig");
const launch = @import("launch.zig");
const module = @import("module.zig");
const Stream = @import("stream.zig").Stream;

pub const most = 4;
const threads = 128;

/// parts_kernel<FP8G, TAIL>: FP8G parts alone, then FP8G parts before BF16 half tiles.
pub const symbols = [2][:0]const u8{
    "_ZN7tf_qmmf12parts_kernelILi3ELb0EEEvPK13__nv_bfloat16NS_5PartsEiiii",
    "_ZN7tf_qmmf12parts_kernelILi3ELb1EEEvPK13__nv_bfloat16NS_5PartsEiiii",
};

/// qmmf.cu's Parts, passed by value. An unused part starts past every block.
pub const Parts = extern struct {
    w: [most]u64 = @splat(0),
    bs: [most]u64 = @splat(0),
    out: [most]u64 = @splat(0),
    scale: [most]f32 = @splat(0),
    n: [most]i32 = @splat(0),
    sk: [most]i32 = @splat(1),
    first: [most]i32 = @splat(std.math.maxInt(i32)),
};

/// The dynamic shared bytes of a launch: the larger of the FP8G 16-row tile's and the BF16 half tile's.
pub fn sharedBytes() u32 {
    return @max(q.sharedBytes(.fp8g, 16), q.halfShared(.bf16));
}

/// TF_QMMF_PARTS=0 keeps each projection in its own launch. The default is one launch where `tail` takes the parts.
pub fn enabled() !bool {
    comptime @import("sys").environment.registered("TF_QMMF_PARTS");
    return try @import("sys").environment.read(usize, "TF_QMMF_PARTS", 1) != 0;
}

pub fn load(mod: module.Module) ![2]module.Function {
    var fns: [2]module.Function = undefined;
    for (symbols, &fns) |name, *f| {
        f.* = try mod.function(name);
        try f.allowDynamicShared(sharedBytes());
    }
    return fns;
}

/// The count of leading FP8G parts if one launch takes `ws` at `m` rows. Null: the caller runs each part alone.
pub fn tail(l: q.Lane, m: usize, ws: []const q.Weight) ?usize {
    if (l.parts == null or m == 0 or m > 16 or ws.len < 2 or ws.len > most) return null;
    var t: usize = ws.len;
    for (ws, 0..) |w, i| {
        if (w.n == 0 or w.k != ws[0].k or w.k % q.group != 0) return null;
        switch (w.mode) {
            .fp8g => if (t != ws.len) return null,
            // a BF16 part must be one that takes the half tile when it runs alone
            .bf16 => {
                if (((w.n + 63) / 64) * q.slices(w) >= l.half_below) return null;
                if (t == ws.len) t = i;
            },
            else => return null,
        }
    }
    return if (t == 0) null else t;
}

/// ys[i] [m, n] bf16 = x [m, k] bf16 (rows `ldx` apart) times ws[i]. False: no launch, the caller runs each alone.
pub fn matmulMany(l: q.Lane, s: Stream, x: u64, ldx: usize, m: usize, ws: []const q.Weight, ys: []const u64) !bool {
    if (ws.len != ys.len) return false;
    const t = tail(l, m, ws) orelse return false;
    var parts: Parts = .{};
    var blocks: usize = 0;
    for (ws, ys, 0..) |w, y, i| {
        const cols: usize = if (i < t) 64 else 32;
        parts.w[i] = w.codes;
        parts.bs[i] = w.scales;
        parts.out[i] = y;
        parts.scale[i] = w.scale;
        parts.n[i] = @intCast(w.n);
        parts.sk[i] = @intCast(q.slices(w));
        parts.first[i] = @intCast(blocks);
        blocks += (w.n + cols - 1) / cols;
    }
    var a: launch.Args = .{};
    a.add(x);
    a.add(parts);
    for ([_]usize{ m, ws[0].k, if (m == 1) ws[0].k else ldx, t }) |v| a.add(@as(i32, @intCast(v)));
    try launch.launch(l.parts.?[@intFromBool(t < ws.len)], .{
        .grid = .{ .x = @intCast(blocks) },
        .block = .{ .x = threads },
        .shared = sharedBytes(),
    }, s, &a);
    return true;
}

test "lane parts: the Qwen3.8 delta and attention inputs take one launch, other groups none" {
    const abi = @import("abi.zig");
    const Mock = struct {
        var id: usize = 0;
        var grid: c_uint = 0;
        var smem: c_uint = 0;
        var parts: Parts = .{};
        var ints: [4]i32 = @splat(0);
        fn run(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            id = @intFromPtr(f.?);
            grid = gx;
            smem = sm;
            parts = @as(*const Parts, @ptrCast(@alignCast(args.?[1].?))).*;
            for (&ints, 2..) |*v, i| v.* = @as(*const i32, @ptrCast(@alignCast(args.?[i].?))).*;
            return 0;
        }
    };
    var d: @import("driver.zig").Driver = undefined;
    d.api.cuLaunchKernel = Mock.run;
    var l: q.Lane = undefined;
    l.clusters = true;
    l.half_below = q.half_below_default;
    l.parts = .{ .{ .d = &d, .handle = @ptrFromInt(201) }, .{ .d = &d, .handle = @ptrFromInt(202) } };
    const s: Stream = .{ .d = &d, .handle = null };
    const w = struct {
        fn of(mode: q.Mode, n: u32, at: u64) q.Weight {
            return .{ .mode = mode, .codes = at, .scales = at + 1, .scale = 0.5, .n = n, .k = 5120, .npad = (n + 127) / 128 * 128 };
        }
    }.of;
    try std.testing.expectEqual(@as(usize, 160), @sizeOf(Parts));
    // delta inputs: qkv and z in FP8G, a and b in BF16 half tiles
    const delta = [_]q.Weight{ w(.fp8g, 10240, 10), w(.fp8g, 6144, 20), w(.bf16, 48, 30), w(.bf16, 48, 40) };
    try std.testing.expect(try matmulMany(l, s, 7, 5120, 16, &delta, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 202), Mock.id);
    try std.testing.expectEqual(@as(c_uint, 160 + 96 + 2 + 2), Mock.grid);
    try std.testing.expectEqual(@as(c_uint, 25600), Mock.smem);
    try std.testing.expectEqualSlices(i32, &.{ 0, 160, 256, 258 }, &Mock.parts.first);
    try std.testing.expectEqualSlices(i32, &.{ 2, 2, 8, 8 }, &Mock.parts.sk);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4 }, &Mock.parts.out);
    try std.testing.expectEqualSlices(i32, &.{ 16, 5120, 5120, 2 }, &Mock.ints);
    // attention inputs: q, k and v in FP8G. One row reads x at its own stride
    const attention = [_]q.Weight{ w(.fp8g, 12288, 10), w(.fp8g, 1024, 20), w(.fp8g, 1024, 30) };
    try std.testing.expect(try matmulMany(l, s, 7, 9999, 1, &attention, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(usize, 201), Mock.id);
    try std.testing.expectEqual(@as(c_uint, 192 + 16 + 16), Mock.grid);
    try std.testing.expectEqualSlices(i32, &.{ 0, 192, 208, std.math.maxInt(i32) }, &Mock.parts.first);
    try std.testing.expectEqualSlices(i32, &.{ 1, 8, 8, 1 }, &Mock.parts.sk);
    try std.testing.expectEqualSlices(i32, &.{ 1, 5120, 5120, 3 }, &Mock.ints);
    // refused before any launch: 17 rows, an FP4 part, BF16 before FP8G, a wide BF16 part, two input widths, one part
    Mock.id = 0;
    const mlp = [_]q.Weight{ w(.fp4, 17408, 10), w(.fp4, 17408, 20) };
    const swapped = [_]q.Weight{ w(.bf16, 48, 10), w(.fp8g, 6144, 20) };
    const wide = [_]q.Weight{ w(.fp8g, 6144, 10), w(.bf16, 6144, 20) };
    var short = w(.fp8g, 1024, 20);
    short.k = 2560;
    const widths = [_]q.Weight{ w(.fp8g, 12288, 10), short };
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 17, &delta, &.{ 1, 2, 3, 4 }));
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 16, &mlp, &.{ 1, 2 }));
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 16, &swapped, &.{ 1, 2 }));
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 16, &wide, &.{ 1, 2 }));
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 16, &widths, &.{ 1, 2 }));
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 16, delta[0..1], &.{1}));
    l.parts = null; // TF_QMMF_PARTS=0, or a device the launch is not measured on
    try std.testing.expect(!try matmulMany(l, s, 7, 5120, 16, &delta, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 0), Mock.id);
}

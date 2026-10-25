//! EXL3 projections on the Intel GPU: zig/kernels/xpu/exl3.cl over Level Zero.
//! The same steps as core/exl3_projection.zig (Metal) and cuda/exl3.zig: rotate in, matvec, rotate out.
//! Model code gives device addresses and shapes only; it never sees a Level Zero type.
//! A launch covers many groups (projections) of one width through a descriptor table on the device, built once
//! at load time and run every step. Routed experts build their slot table on the device (exl3_route).

const std = @import("std");
const Context = @import("context.zig").Context;
const Error = @import("context.zig").Error;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const mod = @import("module.zig");
const launch = @import("launch.zig").launch;
const kernels = @import("kernels.zig");
pub const variants = @import("exl3_variants.zig");

pub const Output = enum(u32) { bf16 = 0, f32 = 1 };

/// One stored EXL3 weight W[K,N] (codebook mul1, fp16 suh and svh): device addresses of the checkpoint tensors.
pub const Matrix = struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    k: u32,
    n: u32,
    /// Twice the bits: 8 for 4 bits, 12 for 6 bits.
    halves: u32,

    pub fn validate(m: Matrix) error{ UnsupportedExl3Shape, UnsupportedExl3Width }!void {
        if (m.k == 0 or m.n == 0 or m.k % 128 != 0 or m.n % 128 != 0) return error.UnsupportedExl3Shape;
        if (m.halves % 2 != 0 or m.halves < 4 or m.halves > 16) return error.UnsupportedExl3Width;
    }
};

/// One projection of a launch. x is bf16 [rows][x_stride], y is [rows][y_stride] in `output`, and scratch holds
/// scratchBytes(k, n, rows) bytes that no other group of the launch uses.
pub const Group = struct {
    m: Matrix,
    rows: u32,
    x: u64,
    x_stride: u32,
    y: u64,
    y_stride: u32,
    output: Output = .bf16,
    scratch: u64,
};

/// The table row of exl3.cl (exl3_desc).
pub const Desc = extern struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    x: u64,
    rot: u64,
    sums: u64,
    out: u64,
    k: u32,
    n: u32,
    halves: u32,
    rows: u32,
    x_stride: u32,
    out_stride: u32,
    out_f32: u32,
    pad: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Desc) == 88);
}

fn rotBytes(k: usize, rows: usize) usize {
    return std.mem.alignForward(usize, k * rows * 4, 256);
}

/// fp32 rotated input [rows][k], then fp32 sums [rows][n].
pub fn scratchBytes(k: usize, n: usize, rows: usize) usize {
    return rotBytes(k, rows) + n * rows * 4;
}

pub fn descriptor(g: Group) !Desc {
    try g.m.validate();
    if (g.rows == 0 or g.rows > variants.rows_max) return error.InvalidExl3Rows;
    if (g.x_stride < g.m.k or g.y_stride < g.m.n) return error.InvalidExl3Stride;
    return .{
        .trellis = g.m.trellis,
        .suh = g.m.suh,
        .svh = g.m.svh,
        .x = g.x,
        .rot = g.scratch,
        .sums = g.scratch + rotBytes(g.m.k, g.rows),
        .out = g.y,
        .k = g.m.k,
        .n = g.m.n,
        .halves = g.m.halves,
        .rows = g.rows,
        .x_stride = g.x_stride,
        .out_stride = g.y_stride,
        .out_f32 = @intFromEnum(g.output),
    };
}

const local_rotate = [3]u32{ 16, 1, 1 };
const local_matvec = [3]u32{ 128, 1, 1 };

pub const Pipelines = struct {
    base: mod.Module,
    rotate_in: mod.Kernel,
    rotate_out: mod.Kernel,
    route: mod.Kernel,
    matvec_modules: [variants.all.len]mod.Module,
    matvec: [variants.all.len]mod.Kernel,

    pub fn load(ctx: *const Context) Error!Pipelines {
        var p: Pipelines = undefined;
        p.base = try mod.Module.load(ctx, kernels.exl3);
        errdefer p.base.unload();
        p.rotate_in = try p.base.kernel("exl3_rotate_in", local_rotate);
        errdefer p.rotate_in.deinit();
        p.rotate_out = try p.base.kernel("exl3_rotate_out", local_rotate);
        errdefer p.rotate_out.deinit();
        p.route = try p.base.kernel("exl3_route", .{ 1, 1, 1 }); // one work-item a slot, no bound check in the kernel
        errdefer p.route.deinit();
        var done: usize = 0;
        errdefer for (0..done) |i| {
            p.matvec[i].deinit();
            p.matvec_modules[i].unload();
        };
        for (kernels.exl3_matvec, 0..) |image, i| {
            p.matvec_modules[i] = try mod.Module.load(ctx, image);
            p.matvec[i] = p.matvec_modules[i].kernel("exl3_matvec", local_matvec) catch |e| {
                p.matvec_modules[i].unload();
                return e;
            };
            done += 1;
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (&p.matvec, &p.matvec_modules) |*k, *m| {
            k.deinit();
            m.unload();
        }
        p.route.deinit();
        p.rotate_out.deinit();
        p.rotate_in.deinit();
        p.base.unload();
        p.* = undefined;
    }
};

/// A resident descriptor table: groups of one width and at most variants.rows_max rows, run as one launch.
pub const Plan = struct {
    table: DeviceBuffer,
    count: u32,
    bits: u32,
    rows: u32,
    k_max: u32,
    n_max: u32,
    strips_min: u32,
    strips_gcd: u32,

    /// Checks every group and uploads the table (waits for the copy).
    pub fn init(ctx: *const Context, s: Stream, gpa: std.mem.Allocator, groups: []const Group) !Plan {
        if (groups.len == 0 or groups.len > std.math.maxInt(u16)) return error.InvalidExl3Groups;
        const descs = try gpa.alloc(Desc, groups.len);
        defer gpa.free(descs);
        var p: Plan = .{ .table = undefined, .count = @intCast(groups.len), .bits = groups[0].m.halves / 2, .rows = 0, .k_max = 0, .n_max = 0, .strips_min = std.math.maxInt(u32), .strips_gcd = 0 };
        for (groups, descs) |g, *d| {
            d.* = try descriptor(g);
            if (g.m.halves / 2 != p.bits) return error.MixedExl3Widths; // the host sorts groups by width
            p.rows = @max(p.rows, g.rows);
            p.k_max = @max(p.k_max, g.m.k);
            p.n_max = @max(p.n_max, g.m.n);
            p.strips_min = @min(p.strips_min, g.m.n / 16);
            p.strips_gcd = std.math.gcd(p.strips_gcd, g.m.n / 16);
        }
        if (variants.pick(p.bits, p.rows, p.strips_min, p.strips_gcd) == null) return error.UnsupportedExl3Width;
        p.table = try DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(descs));
        return p;
    }

    /// Expert tables for `encodeRouted`: only trellis, suh, svh, k, n and halves are read; all experts share K and N.
    pub fn experts(ctx: *const Context, s: Stream, gpa: std.mem.Allocator, ms: []const Matrix) !Plan {
        if (ms.len == 0) return error.InvalidExl3Groups;
        const descs = try gpa.alloc(Desc, ms.len);
        defer gpa.free(descs);
        for (ms, descs) |m, *d| {
            try m.validate();
            if (m.k != ms[0].k or m.n != ms[0].n or m.halves != ms[0].halves) return error.MixedExl3Experts;
            d.* = .{ .trellis = m.trellis, .suh = m.suh, .svh = m.svh, .x = 0, .rot = 0, .sums = 0, .out = 0, .k = m.k, .n = m.n, .halves = m.halves, .rows = 1, .x_stride = m.k, .out_stride = m.n, .out_f32 = 0 };
        }
        const strips = ms[0].n / 16;
        return .{ .table = try DeviceBuffer.fromHost(ctx, s, std.mem.sliceAsBytes(descs)), .count = @intCast(ms.len), .bits = ms[0].halves / 2, .rows = 1, .k_max = ms[0].k, .n_max = ms[0].n, .strips_min = strips, .strips_gcd = strips };
    }

    pub fn deinit(p: *Plan) void {
        p.table.free();
        p.* = undefined;
    }
};

fn setU64(k: mod.Kernel, i: u32, v: u64) Error!void {
    try k.setBytes(i, std.mem.asBytes(&v));
}

/// Queues the three stages over a table of `count` groups (no host wait).
fn stages(p: *const Pipelines, s: Stream, table: ?*anyopaque, count: u32, bits: u32, rows: u32, k_max: u32, n_max: u32, strips_min: u32, strips_gcd: u32) Error!void {
    const vi = variants.pick(bits, rows, strips_min, strips_gcd) orelse return error.Invalid;
    const mv = p.matvec[vi];
    const ns = variants.all[vi].ns;
    for ([_]mod.Kernel{ p.rotate_in, mv, p.rotate_out }) |k| {
        try k.setPtr(0, null); // base: unused with -DEXL3_ABS
        try k.setPtr(1, table);
    }
    try launch(p.rotate_in, s, .{ k_max / 128, rows, count });
    try launch(mv, s, .{ std.math.divCeil(u32, n_max / 16 / ns, local_matvec[0] / 16) catch unreachable, count, 1 });
    try launch(p.rotate_out, s, .{ n_max / 128, rows, count });
}

/// y = x @ W for every group of the plan.
pub fn encode(p: *const Pipelines, s: Stream, plan: *const Plan) Error!void {
    try stages(p, s, plan.table.ptr, plan.count, plan.bits, plan.rows, plan.k_max, plan.n_max, plan.strips_min, plan.strips_gcd);
}

/// Routed experts, one row a (token, pick) slot. ids are u32 [tokens][topk] on the device. Slot t * topk + j reads
/// x row t (gate and up) or row t * topk + j (down, x_per_slot) and writes y row t * topk + j.
pub const Route = struct {
    ids: u64,
    tokens: u32,
    topk: u32,
    x: u64,
    x_stride: u32,
    x_per_slot: bool,
    y: u64,
    y_stride: u32,
    output: Output = .bf16,
    /// scratchBytes(k, n, tokens * topk) bytes.
    scratch: u64,
    /// slotTableBytes(tokens * topk) bytes for the descriptors exl3_route writes.
    slots: u64,
};

pub fn slotTableBytes(slots: usize) usize {
    return slots * @sizeOf(Desc);
}

pub fn encodeRouted(p: *const Pipelines, s: Stream, experts: *const Plan, r: Route) Error!void {
    const slots = r.tokens * r.topk;
    if (slots == 0) return error.Invalid;
    const k = p.route;
    try k.setPtr(0, experts.table.ptr);
    try k.setPtr(1, @ptrFromInt(r.ids));
    try k.setPtr(2, @ptrFromInt(r.slots));
    try k.setU32(3, r.topk);
    try setU64(k, 4, r.x);
    try k.setU32(5, r.x_stride);
    try k.setU32(6, @intFromBool(r.x_per_slot));
    try setU64(k, 7, r.scratch);
    try setU64(k, 8, r.scratch + rotBytes(experts.k_max, slots));
    try setU64(k, 9, r.y);
    try k.setU32(10, r.y_stride);
    try k.setU32(11, @intFromEnum(r.output));
    try launch(k, s, .{ slots, 1, 1 });
    try stages(p, s, @ptrFromInt(r.slots), slots, experts.bits, 1, experts.k_max, experts.n_max, experts.strips_min, experts.strips_gcd);
}

test "descriptor layout and scratch" {
    const m = Matrix{ .trellis = 0x1000, .suh = 0x2000, .svh = 0x3000, .k = 2048, .n = 512, .halves = 8 };
    const d = try descriptor(.{ .m = m, .rows = 3, .x = 0x4000, .x_stride = 2048, .y = 0x5000, .y_stride = 512, .output = .f32, .scratch = 0x10000 });
    try std.testing.expectEqual(@as(u64, 0x10000 + 2048 * 3 * 4), d.sums);
    try std.testing.expectEqual(@as(u32, 1), d.out_f32);
    try std.testing.expectEqual(@as(usize, 2048 * 3 * 4 + 512 * 3 * 4), scratchBytes(2048, 512, 3));
    try std.testing.expectError(error.InvalidExl3Rows, descriptor(.{ .m = m, .rows = 17, .x = 0, .x_stride = 2048, .y = 0, .y_stride = 512, .scratch = 0 }));
    try std.testing.expectError(error.UnsupportedExl3Shape, (Matrix{ .trellis = 0, .suh = 0, .svh = 0, .k = 100, .n = 512, .halves = 8 }).validate());
}

test {
    _ = variants;
}

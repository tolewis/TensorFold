//! Fixed eight-row DSpark scratch and committed-only per-stream context rings, in XPU device memory.
//! Sizes and byte accounting match the Metal state (metal/dspark_state.zig), so the engine reserves the same bytes.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const intermediate = 9216;
pub const rank = 256;
pub const vocabulary = 200058;
pub const block = 8;
pub const absorb_rows = 64;
pub const local_ring = std.mem.alignForward(usize, 1024 + block, 128);

const fields = [_][]const u8{
    "ids",    "resid", "next_resid", "x",    "y",     "q_raw", "k_raw", "v_raw", "q",     "kb",         "vb", "attn",
    "gate",   "up",    "act",        "base", "fused", "ck",    "cv",    "kappa", "state", "next_state", "z",  "lambda",
    "logits",
};

fn sizes() [fields.len]usize {
    const r = block;
    const a = absorb_rows;
    const b = 2;
    return .{
        r * 4,                r * hidden * b,     r * hidden * b,       r * hidden * b,       r * hidden * b,
        r * query * b,        r * kv * b,         r * kv * b,           r * query * b,        r * kv * b,
        r * kv * b,           r * query * b,      r * intermediate * b, r * intermediate * b, r * intermediate * b,
        (r - 1) * 201024 * b, 4 * a * hidden * b, a * kv * b,           a * kv * b,           (r - 1) * 3 * rank * 4,
        rank * b,             rank * b,           rank * b,             rank * b,             vocabulary * b,
    };
}

pub fn scratchBytes() usize {
    var bytes: usize = 0;
    for (sizes()) |n| bytes += n;
    return bytes;
}

pub const Scratch = struct {
    ids: xpu.DeviceBuffer,
    resid: xpu.DeviceBuffer,
    next_resid: xpu.DeviceBuffer,
    x: xpu.DeviceBuffer,
    y: xpu.DeviceBuffer,
    q_raw: xpu.DeviceBuffer,
    k_raw: xpu.DeviceBuffer,
    v_raw: xpu.DeviceBuffer,
    q: xpu.DeviceBuffer,
    kb: xpu.DeviceBuffer,
    vb: xpu.DeviceBuffer,
    attn: xpu.DeviceBuffer,
    gate: xpu.DeviceBuffer,
    up: xpu.DeviceBuffer,
    act: xpu.DeviceBuffer,
    base: xpu.DeviceBuffer,
    fused: xpu.DeviceBuffer,
    ck: xpu.DeviceBuffer,
    cv: xpu.DeviceBuffer,
    kappa: xpu.DeviceBuffer,
    state: xpu.DeviceBuffer,
    next_state: xpu.DeviceBuffer,
    z: xpu.DeviceBuffer,
    lambda: xpu.DeviceBuffer,
    logits: xpu.DeviceBuffer,

    pub fn init(ctx: *const xpu.Context) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = sizes();
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).free();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try xpu.DeviceBuffer.alloc(ctx, @max(ns[i], 16));
            made += 1;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).free();
        s.* = undefined;
    }
};
pub fn globalRing(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, 128);
}

pub fn scoreBytes(capacity: usize) u64 {
    return @as(u64, 8 * 24 * 4) * (capacity + block);
}

/// Two sliding layers hold two fixed K/V rings; layer 1 holds two global K/V rings.
pub fn laneFixedBytes() u64 {
    return 2 * 2 * local_ring * kv * 2 + hidden * 2;
}

pub fn laneTokenBytes() u64 {
    return 2 * kv * 2;
}

pub fn laneBytes(capacity: usize) u64 {
    return laneFixedBytes() + globalRing(capacity) * laneTokenBytes();
}

pub const Lane = struct {
    k: [3]xpu.DeviceBuffer,
    v: [3]xpu.DeviceBuffer,
    rings: [3]usize,
    newest: xpu.DeviceBuffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [7]u32 = @splat(0),
    held_len: u32 = 0,

    pub fn init(ctx: *const xpu.Context, capacity: usize) !Lane {
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        var l: Lane = .{ .k = undefined, .v = undefined, .rings = .{ local_ring, globalRing(capacity), local_ring }, .newest = undefined };
        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                l.k[i].free();
                l.v[i].free();
            }
        }
        for (l.rings, 0..) |ring, i| {
            const bytes = ring * kv * 2;
            l.k[i] = try xpu.DeviceBuffer.alloc(ctx, bytes);
            l.v[i] = xpu.DeviceBuffer.alloc(ctx, bytes) catch |err| {
                l.k[i].free();
                return err;
            };
            made += 1;
        }
        l.newest = try xpu.DeviceBuffer.alloc(ctx, hidden * 2);
        return l;
    }

    pub fn deinit(l: *Lane) void {
        l.newest.free();
        for (0..3) |i| {
            l.k[i].free();
            l.v[i].free();
        }
        l.* = undefined;
    }
};

comptime {
    if (local_ring != 1152 or laneTokenBytes() != 4096) @compileError("DSpark ring geometry drift");
}

test "DSpark stream accounting includes sliding, global and shared attention scratch" {
    try std.testing.expectEqual(@as(u64, 9_437_184 + 4608), laneFixedBytes());
    try std.testing.expectEqual(laneFixedBytes() + 128 * 4096, laneBytes(1));
    try std.testing.expectEqual(laneFixedBytes() + 1152 * 4096, laneBytes(1025));
    try std.testing.expectEqual(@as(u64, 768 * 1032), scoreBytes(1024));
    // Ring and scores bytes are reserved with the target cache before the lane opens.
}

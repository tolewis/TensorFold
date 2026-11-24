//! DFlash 2's eight-row block scratch and each stream's context rings, in XPU device memory. The rings hold the last
//! 2,048 committed rows' keys and values for each of the five layers; nothing else of a stream lives on the card.
const std = @import("std");
const xpu = @import("xpu");
const k = @import("dflash2_kernels.zig");
const weights = @import("dflash2_weights.zig");

pub const hidden = 2304;
pub const query = 4096;
pub const kv = 1024;
pub const intermediate = 6912;
pub const rank = 256;
pub const block = 8;
pub const layers = 5;
/// The window: a block row sees the 2,047 rows before it, and the ring holds one more.
pub const ring = k.window;
/// Rows one absorb launch takes.
pub const absorb_rows = 64;
pub const vocabulary = weights.valid_vocabulary;
/// Candidate lists the top-k pass leaves per row.
pub const lists = k.topk_groups * k.topk;

const fields = [_][]const u8{
    "ids",  "x",   "resid", "next_resid", "y",    "conv", "dyn",  "q_raw", "k_raw", "v_raw", "q", "kb",
    "attn", "out", "gate",  "up",         "act",  "final", "sel", "base",  "part",  "top_v", "top_i",
    "fc",   "hn",  "ck",    "cv",         "junk",
};

fn sizes() [fields.len]usize {
    const r = block;
    const a = absorb_rows;
    const b = 2;
    return .{
        r * 4,                           r * hidden * b, r * hidden * b,        r * hidden * b,       r * hidden * b,    r * hidden * b, r * weights.dyn * b, r * query * b, r * kv * b, r * kv * b, r * query * b, r * kv * b,
        r * query * b,                   r * hidden * b, r * intermediate * b,  r * intermediate * b, r * intermediate * b, r * hidden * b, r * rank * b, (r - 1) * 201024 * b, r * k.kv_heads * k.max_chunks * k.part_floats * 4, r * lists * 4, r * lists * 4,
        a * hidden * b,                  a * hidden * b, a * kv * b,            a * kv * b,           a * hidden * b,
    };
}

pub fn scratchBytes() usize {
    var bytes: usize = 0;
    for (sizes()) |n| bytes += n;
    return bytes;
}

pub const Scratch = struct {
    ids: xpu.DeviceBuffer,
    x: xpu.DeviceBuffer,
    resid: xpu.DeviceBuffer,
    next_resid: xpu.DeviceBuffer,
    y: xpu.DeviceBuffer,
    conv: xpu.DeviceBuffer,
    dyn: xpu.DeviceBuffer,
    q_raw: xpu.DeviceBuffer,
    k_raw: xpu.DeviceBuffer,
    v_raw: xpu.DeviceBuffer,
    q: xpu.DeviceBuffer,
    kb: xpu.DeviceBuffer,
    attn: xpu.DeviceBuffer,
    out: xpu.DeviceBuffer,
    gate: xpu.DeviceBuffer,
    up: xpu.DeviceBuffer,
    act: xpu.DeviceBuffer,
    final: xpu.DeviceBuffer,
    sel: xpu.DeviceBuffer,
    base: xpu.DeviceBuffer,
    part: xpu.DeviceBuffer,
    top_v: xpu.DeviceBuffer,
    top_i: xpu.DeviceBuffer,
    fc: xpu.DeviceBuffer,
    hn: xpu.DeviceBuffer,
    ck: xpu.DeviceBuffer,
    cv: xpu.DeviceBuffer,
    junk: xpu.DeviceBuffer,

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

/// Device bytes one stream holds: five layers' key and value rings.
pub fn laneBytes() u64 {
    return @as(u64, layers) * 2 * ring * kv * 2;
}

pub const Lane = struct {
    k: [layers]xpu.DeviceBuffer,
    v: [layers]xpu.DeviceBuffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [7]u32 = @splat(0),
    held_len: u32 = 0,

    pub fn init(ctx: *const xpu.Context) !Lane {
        var l: Lane = .{ .k = undefined, .v = undefined };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].free();
            l.v[i].free();
        };
        for (0..layers) |i| {
            const bytes = ring * kv * 2;
            l.k[i] = try xpu.DeviceBuffer.alloc(ctx, bytes);
            l.v[i] = xpu.DeviceBuffer.alloc(ctx, bytes) catch |err| {
                l.k[i].free();
                return err;
            };
            made += 1;
        }
        return l;
    }

    pub fn deinit(l: *Lane) void {
        for (0..layers) |i| {
            l.k[i].free();
            l.v[i].free();
        }
        l.* = undefined;
    }
};

test "a stream's rings hold five layers of 2,048 rows, keys and values" {
    try std.testing.expectEqual(@as(u64, 5 * 2 * 2048 * 1024 * 2), laneBytes());
    try std.testing.expect(scratchBytes() < (64 << 20));
}

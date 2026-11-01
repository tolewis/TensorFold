//! XPU_WINDOW_PROFILE=1: device-event time per kernel class in the load-time window measure, one table a row count.
//! The launch sites hand each kernel an event (zeEventQueryKernelTimestamp after the queue drains), so the
//! per-class split costs nothing in the queue order. Off: no event, no clock, no table.
//!
//! Classes: EXL3 by projection, the GDN and attention decode kernels, norms and glue, the all-reduce add kernel
//! and its host wait, the drafter. Host gaps are the wall time of a pass that the kernels do not account for.

const std = @import("std");
const ze = @import("ze");

pub const Class = enum(u8) {
    exl3_in = 0,
    exl3_out,
    exl3_gate_up,
    exl3_down,
    exl3_head,
    gemm,
    gdn,
    attn,
    glue,
    ar_add,
    ar_wait,
    drafter,
    other,
};

const nclasses = @typeInfo(Class).@"enum".fields.len;

pub const class_names = [_][]const u8{
    "exl3 mixer in (qkv, gdn in)",
    "exl3 mixer out (o, gdn out)",
    "exl3 gate up",
    "exl3 down",
    "exl3 lm_head",
    "gemm other (int4, w4a8)",
    "gdn step",
    "attention decode",
    "norms and glue",
    "all-reduce add",
    "all-reduce wait (host)",
    "drafter",
    "other",
};

const max_events = 8192;

/// On: the measure loop reads events. The launch sites check this first.
pub var on: bool = false;

var events: [max_events]ze.H = @splat(null);
var classes: [max_events]Class = @splat(.other);
var used: usize = 0;
var dropped: usize = 0;
var total: [nclasses]f64 = @splat(0);
var calls: [nclasses]u32 = @splat(0);
var wall: f64 = 0;
var wait_s: f64 = 0;
var passes: u32 = 0;
var rows: u32 = 0;
var started: f64 = 0;
/// The projection (or pass) whose launches are queued now; `.other` falls back to the kernel handle's class.
var tag: Class = .other;

const Reg = struct { h: ze.H = null, c: Class = .other };
var reg: [128]Reg = @splat(.{});
var reg_n: usize = 0;

extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;

fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn starts(a: []const u8, b: []const u8) bool {
    return std.mem.startsWith(u8, a, b);
}

/// The class of a kernel by its field name in the runner's kernel set.
pub fn classOfName(name: []const u8) Class {
    if (eql(name, "attn_decode") or eql(name, "attn_merge") or eql(name, "attn_prep") or eql(name, "attn_gate") or eql(name, "rot_heads") or eql(name, "attn_prompt") or eql(name, "attn_prompt8")) return .attn;
    if (starts(name, "gdn_") or eql(name, "conv_commit")) return .gdn;
    if (eql(name, "tp_add")) return .ar_add;
    if (starts(name, "dec_") or starts(name, "ngen_") or eql(name, "a8") or eql(name, "a8_swiglu") or eql(name, "a4")) return .gemm;
    return .glue;
}

/// The runner's kernel set by handle: a launch of one of these needs no tag at its call site.
pub fn register(ks: anytype) void {
    inline for (@typeInfo(@TypeOf(ks)).@"struct".field_names) |name| {
        const h = @field(ks, name);
        if (h == null or reg_n == reg.len) continue;
        reg[reg_n] = .{ .h = h, .c = classOfName(name) };
        reg_n += 1;
    }
}

fn classOf(h: ze.H) Class {
    if (tag != .other) return tag;
    for (reg[0..reg_n]) |r| if (r.h == h) return r.c;
    return .other;
}

/// The event a launch on this class takes, or null (profile off, or the pool full: the launch is then untimed).
pub fn attach(g: *ze.Gpu, h: ze.H) ze.H {
    if (!on) return null;
    if (used == events.len) {
        dropped += 1;
        return null;
    }
    classes[used] = classOf(h);
    const e = events[used];
    used += 1;
    return e;
}

/// The class the launches queued now belong to (a projection or the drafter pass).
pub fn setTag(c: Class) void {
    tag = c;
}

/// The event pool for the measure; call once, before the model load (the load's launches stay out of the tables).
pub fn enable(g: *ze.Gpu) !void {
    events = try g.events(max_events);
    on = true;
}

pub fn beginRow(r: u32) void {
    if (!on) return;
    rows = r;
    passes = 0;
    wall = 0;
    wait_s = 0;
    total = @splat(0);
    calls = @splat(0);
    dropped = 0;
    used = 0;
}

pub fn beginPass() void {
    if (!on) return;
    started = now();
}

pub fn endPass() void {
    if (!on) return;
    wall += now() - started;
    passes += 1;
}

/// Host time in the all-reduce's peer wait and settle (tp.allReduce), added to the wait class.
pub fn hostWait(secs: f64) void {
    if (!on) return;
    wait_s += secs;
}

/// Reads every pending event's GPU timestamps into its class; the queue must be idle (the caller syncs).
pub fn flush(g: *ze.Gpu) void {
    if (!on) return;
    for (events[0..used], classes[0..used]) |e, c| {
        const t = g.seconds(e) catch continue;
        total[@intFromEnum(c)] += t;
        calls[@intFromEnum(c)] += 1;
        _ = g.api.zeEventHostReset(e);
    }
    used = 0;
}

/// One table for the row count just measured: class, ms, percent of the GPU time, launches, and the host gaps.
pub fn report() void {
    if (!on) return;
    total[@intFromEnum(.ar_wait)] += wait_s;
    var gpu: f64 = 0;
    for (total, 0..) |t, i| if (i != @intFromEnum(.ar_wait)) {
        gpu += t;
    };
    const gap = wall - gpu - wait_s;
    std.debug.print("-- window profile, rows {d}, {d} passes: wall {d:.1} ms, gpu {d:.1} ms ({d:.1} %), host gaps {d:.1} ms\n", .{ rows, passes, wall * 1e3, gpu * 1e3, if (wall > 0) gpu / wall * 100 else 0, gap * 1e3 });
    std.debug.print("   {s:<28} {s:>9} {s:>6} {s:>9}\n", .{ "class", "ms", "%", "launches" });
    for (class_names, total, calls) |name, t, n| {
        if (t <= 0 and n == 0) continue;
        std.debug.print("   {s:<28} {d:>9.2} {d:>5.1} {d:>9}\n", .{ name, t * 1e3, if (gpu > 0) t / gpu * 100 else 0, n });
    }
    if (dropped > 0) std.debug.print("   ({d} launches untimed: the event pool was full)\n", .{dropped});
}

test "the kernel classes by name" {
    try std.testing.expectEqual(Class.attn, classOfName("attn_decode"));
    try std.testing.expectEqual(Class.gdn, classOfName("gdn_step"));
    try std.testing.expectEqual(Class.gdn, classOfName("conv_commit"));
    try std.testing.expectEqual(Class.ar_add, classOfName("tp_add"));
    try std.testing.expectEqual(Class.gemm, classOfName("dec_wide"));
    try std.testing.expectEqual(Class.glue, classOfName("add_rmsnorm"));
    try std.testing.expectEqual(Class.glue, classOfName("embed"));
}

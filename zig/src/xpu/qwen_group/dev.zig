//! The device the e5506ae Group calls. The context, list and device are the Inkling process's.
const std = @import("std");
const ze = @import("ze.zig");
const q = @import("qwen27.zig");

pub var settle_ns: u64 = std.math.maxInt(u64);
pub var lost: *const fn (u32) void = struct {
    fn f(_: u32) void {}
}.f;

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

pub const Kernels = struct {
    tp_add: ze.H = null,
    tp_add_flag: ze.H = null,
    tp_consume: ze.H = null,
    tp_norm_tp: ze.H = null,
    tp_probe: ze.H = null,
    argmax_tp: ze.H = null,
};

pub const Gpu = struct {
    api: ze.Api,
    cbapi: ?ze.CbApi = null,
    zex_write: ?ze.ZexWrite = null,
    context: ze.H,
    device: ze.H,

    pub fn deviceAlloc(g: *Gpu, size: usize) ! *anyopaque {
        var p: ?*anyopaque = null;
        try ze.check(g.api.zeMemAllocDevice(g.context, &.{}, size, 4096, g.device, &p));
        return p.?;
    }

    pub fn hostAlloc(g: *Gpu, size: usize) ! *anyopaque {
        var p: ?*anyopaque = null;
        try ze.check(g.api.zeMemAllocHost(g.context, &.{}, size, 4096, &p));
        return p.?;
    }

    pub fn immediate(g: *Gpu, ordinal: u32, priority: u32) !ze.H {
        var l: ze.H = null;
        try ze.check(g.api.zeCommandListCreateImmediate(g.context, g.device, &.{ .ordinal = ordinal, .flags = ze.queue_in_order, .mode = 2, .priority = priority }, &l));
        return l.?;
    }

    pub fn arg(g: *Gpu, k: ze.H, i: u32, v: anytype) !void {
        try ze.check(g.api.zeKernelSetArgumentValue(k, i, @sizeOf(@TypeOf(v)), @ptrCast(&v)));
    }
};

pub const Dev = struct {
    g: Gpu,
    list: ze.H,
    k: Kernels = .{},

    pub fn settle(d: *Dev) void {
        const rc = d.g.api.zeCommandListHostSynchronize(d.list, settle_ns);
        if (rc != 0) lost(rc);
    }

    pub fn sync(d: *Dev) !void {
        d.settle();
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, null, 0, null));
    }

    pub fn launchEv(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype, ev: ze.H) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        var wa = [1]ze.H{ev};
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, null, 1, &wa));
    }
};

pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    if (q.stat.on) q.stat.copies += 1;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, null, 0, null));
}

pub fn bind(lib: *std.DynLib, driver: ze.H, context: ze.H, device: ze.H, list: ze.H, add: ze.H, hz: u64) !Dev {
    const api = try ze.resolve(ze.Api, lib);
    var g: Gpu = .{ .api = api, .context = context, .device = device };
    g.cbapi = ze.resolveOptional(ze.CbApi, lib);
    if (ze.resolveOptional(ze.ExtApi, lib)) |x| {
        var fp: ?*anyopaque = null;
        if (x.zeDriverGetExtensionFunctionAddress(driver, "zexCommandListAppendWriteToMemory", &fp) == 0) {
            if (fp) |f| g.zex_write = @ptrCast(f);
        }
    }
    timer_hz = if (hz == 0) 1 else hz;
    ze_lib = lib;
    return .{ .g = g, .list = list, .k = .{ .tp_add = add } };
}

const TsFn = *const fn (ze.H, *anyopaque, ze.H, u32, ?*ze.H) callconv(.c) u32;
var timer_hz: u64 = 1;
var wait_checked: bool = false;
var wait_on: bool = false;
var ze_lib: ?*std.DynLib = null;
var stamp_fn: ?TsFn = null;
var stamp_buf: ?*anyopaque = null;
var last_name: [48]u8 = undefined;
var last_len: u8 = 0;
const WaitRec = struct { name: [48]u8, nlen: u8 };
var waits: [128]WaitRec = undefined;
var wait_n: u16 = 0;

fn waitOn() bool {
    if (!wait_checked) {
        wait_on = std.c.getenv("TF_TP_WAITLOG") != null;
        wait_checked = true;
    }
    return wait_on;
}

pub fn noteKernel(name: []const u8) void {
    const n = @min(name.len, 47);
    if (n > 0) @memcpy(last_name[0..n], name[0..n]);
    last_len = @intCast(n);
}

fn writeStamp(d: *Dev, slot: u16, which: u16) !void {
    if (stamp_fn == null) {
        const lib = ze_lib orelse return;
        stamp_fn = lib.lookup(TsFn, "zeCommandListAppendWriteGlobalTimestamp");
        if (stamp_fn == null) return;
        stamp_buf = try d.g.deviceAlloc(128 * 2 * 8);
    }
    const f = stamp_fn orelse return;
    const base: [*]u8 = @ptrCast(stamp_buf.?);
    const p: *anyopaque = @ptrCast(base + (@as(usize, slot) * 2 + which) * 8);
    try ze.check(f(d.list, p, null, 0, null));
}

pub fn waitBefore(d: *Dev) !void {
    if (!waitOn() or wait_n >= 128) return;
    try writeStamp(d, wait_n, 0);
}

pub fn waitAfter(d: *Dev) !void {
    if (!waitOn() or wait_n >= 128) return;
    try writeStamp(d, wait_n, 1);
    waits[wait_n].nlen = last_len;
    if (last_len > 0) @memcpy(waits[wait_n].name[0..last_len], last_name[0..last_len]);
    wait_n += 1;
}

pub fn reportWaits(d: *Dev, rank: u32) !void {
    if (!waitOn() or wait_n == 0 or stamp_buf == null) return;
    const n = wait_n;
    const host = try d.g.hostAlloc(@as(usize, n) * 2 * 8);
    try copy(d, host, stamp_buf.?, @as(usize, n) * 2 * 8);
    d.settle();
    const ticks = std.mem.bytesAsSlice(u64, @as([*]u8, @ptrCast(host))[0 .. @as(usize, n) * 16]);
    const hz: f64 = @floatFromInt(timer_hz);
    var sum: f64 = 0;
    var best: [5]f64 = .{ 0, 0, 0, 0, 0 };
    var best_i: [5]u16 = .{ 0, 0, 0, 0, 0 };
    var i: u16 = 0;
    while (i < n) : (i += 1) {
        const t0 = ticks[@as(usize, i) * 2];
        const t1 = ticks[@as(usize, i) * 2 + 1];
        if (t1 < t0) continue;
        const us = @as(f64, @floatFromInt(t1 - t0)) / hz * 1e6;
        sum += us;
        var k: usize = 0;
        while (k < 5 and us > best[k]) : (k += 1) {}
        if (k > 0) {
            var j: usize = 0;
            while (j + 1 < k) : (j += 1) {
                best[j] = best[j + 1];
                best_i[j] = best_i[j + 1];
            }
            best[k - 1] = us;
            best_i[k - 1] = i;
        }
    }
    std.log.info("inkling2: wait rank {d} n {d} sum {d:.1} us", .{ rank, n, sum });
    var k: usize = 5;
    while (k > 0) {
        k -= 1;
        if (best[k] == 0) continue;
        const rec = waits[best_i[k]];
        const name = if (rec.nlen == 0) "?" else rec.name[0..rec.nlen];
        std.log.info("inkling2: wait rank {d} {d:.1} us after {s}", .{ rank, best[k], name });
    }
    wait_n = 0;
}

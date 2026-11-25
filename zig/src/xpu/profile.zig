//! An optional launch profile for one stream: every kernel launch counted by kernel name, the host time of its
//! argument setting and queueing, and (with `device_time`) the device time of each kernel from a timestamp event.
//! Off unless a program installs one in `active`; a launch then costs two clock reads more.

const std = @import("std");
const platform = @import("sys");
const ze = @import("ze");
const Context = @import("context.zig").Context;
const Error = @import("context.zig").Error;

/// The profile launches report to, or none.
pub var active: ?*Profile = null;

/// A name the caller sets before launches that share one kernel (the dense projections); the profile keeps a row for each
/// (kernel, label). Static strings only.
pub var label: []const u8 = "";

/// The kernel queued most recently. The exchange wait log names this as the work before the wait.
pub var last_kernel: []const u8 = "";

/// Host time of the exchange appends, when TF_TP_APPEND=1. Sites: copy, flag write, event create, event wait, kernel launch.
pub var host_append_ns: [5]u64 = @splat(0);
pub var append_n: [5]u64 = @splat(0);
pub var append_max: [5]u64 = @splat(0);
pub var append_slow: [5]u64 = @splat(0);

var append_checked: bool = false;
var append_on: bool = false;

fn appendEnabled() bool {
    if (!append_checked) {
        append_on = std.c.getenv("TF_TP_APPEND") != null;
        append_checked = true;
    }
    return append_on;
}

pub fn noteAppend(site: u8, dt: u64) void {
    if (!appendEnabled()) return;
    const i: usize = site;
    append_n[i] += 1;
    host_append_ns[i] += dt;
    if (dt > append_max[i]) append_max[i] = dt;
    if (dt > 50_000) append_slow[i] += 1;
}

pub fn reportAppend(rank: u32) void {
    if (!appendEnabled()) return;
    const name = [_][]const u8{ "copy", "write", "create", "wait", "launch" };
    for (name, 0..) |s, i| {
        const n: f64 = @floatFromInt(@max(append_n[i], 1));
        std.log.info("inkling2: append {s} rank {d} n {d} total {d:.1} us mean {d:.1} us max {d:.1} us over50us {d}", .{ s, rank, append_n[i], @as(f64, @floatFromInt(host_append_ns[i])) / 1e3, @as(f64, @floatFromInt(host_append_ns[i])) / 1e3 / n, @as(f64, @floatFromInt(append_max[i])) / 1e3, append_slow[i] });
    }
    host_append_ns = @splat(0);
    append_n = @splat(0);
    append_max = @splat(0);
    append_slow = @splat(0);
}

const Stamp = *const fn (ze.H, *u64, *u64) callconv(.c) u32;

const HostClock = struct {
    host: u64,
    device: u64,
    res: f64,
    mask: u64,

    fn hostNs(c: HostClock, tick: u64) i64 {
        const delta = @as(i64, @intCast((tick -% c.device) & c.mask));
        const wrapped = if (delta > @as(i64, @intCast(c.mask / 2))) delta - @as(i64, @intCast(c.mask)) - 1 else delta;
        const ns: i64 = @intFromFloat(@as(f64, @floatFromInt(wrapped)) * 1e9 / c.res);
        return @as(i64, @intCast(c.host)) + ns;
    }
};

fn hostClock(ctx: *const Context) ?HostClock {
    const get = @constCast(ctx.d).lib.lookup(Stamp, "zeDeviceGetGlobalTimestamps") orelse return null;
    var host: u64 = 0;
    var device: u64 = 0;
    if (get(ctx.gpu.device, &host, &device) != 0) return null;
    const bits = ctx.gpu.props.timestampValidBits;
    const mask: u64 = if (bits >= 64 or bits == 0) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(bits)) - 1;
    const res: f64 = @floatFromInt(@max(ctx.gpu.props.timerResolution, 1));
    return .{ .host = host, .device = device, .res = res, .mask = mask };
}

pub fn nowNs() u64 {
    var ts: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Rec = struct { handle: ze.H, set_ns: u32, append_ns: u32, args: u16, label: []const u8 };

pub const Row = struct {
    name: [64]u8 = @splat(0),
    label: []const u8 = "",
    calls: u64 = 0,
    device_ns: u64 = 0,
    /// The device time between this launch's end and the next launch's start, in the same pass.
    gap_ns: u64 = 0,
    set_ns: u64 = 0,
    append_ns: u64 = 0,
    args: u64 = 0,
};

pub const Profile = struct {
    ctx: *const Context,
    pool: ze.H = null,
    events: []ze.H = &.{},
    recs: []Rec,
    used: usize = 0,
    /// Launches past the capacity (counted, not timed on the device).
    dropped: usize = 0,
    device_time: bool,
    rows: std.ArrayList(Row) = .empty,
    names: std.ArrayList(struct { handle: ze.H, label: []const u8, row: usize }) = .empty,
    /// Totals over every `collect`.
    launches: u64 = 0,
    device_busy_ns: u64 = 0,
    device_span_ns: u64 = 0,
    tokens: u64 = 0,
    enqueued_at: u64 = 0,

    const gpa = std.heap.c_allocator;

    /// Room for `capacity` launches a pass; with `device_time` each gets a kernel-timestamp event.
    pub fn init(ctx: *const Context, capacity: usize, device_time: bool) Error!Profile {
        var p: Profile = .{ .ctx = ctx, .recs = try gpa.alloc(Rec, capacity), .device_time = device_time };
        if (device_time) {
            const d = ctx.d;
            var device = ctx.gpu.device;
            p.events = try gpa.alloc(ze.H, capacity);
            try d.check(d.api.zeEventPoolCreate(ctx.gpu.context, &.{ .flags = ze.pool_host_visible | ze.pool_kernel_timestamp, .count = @intCast(capacity) }, 1, @ptrCast(&device), &p.pool), "zeEventPoolCreate");
            for (p.events, 0..) |*e, i| try d.check(d.api.zeEventCreate(p.pool, &.{ .index = @intCast(i) }, e), "zeEventCreate");
        }
        return p;
    }

    pub fn deinit(p: *Profile) void {
        const d = p.ctx.d;
        for (p.events) |e| _ = d.api.zeEventDestroy(e);
        if (p.pool != null) _ = d.api.zeEventPoolDestroy(p.pool);
        gpa.free(p.events);
        gpa.free(p.recs);
        p.rows.deinit(gpa);
        p.names.deinit(gpa);
    }

    /// The event the next launch signals, or none.
    pub fn nextEvent(p: *Profile) ze.H {
        return if (p.device_time and p.used < p.events.len) p.events[p.used] else null;
    }

    pub fn record(p: *Profile, handle: ze.H, set_ns: u64, append_ns: u64, args: usize) void {
        if (p.used >= p.recs.len) {
            p.dropped += 1;
            return;
        }
        p.recs[p.used] = .{ .label = label, .handle = handle, .set_ns = @intCast(@min(set_ns, std.math.maxInt(u32))), .append_ns = @intCast(@min(append_ns, std.math.maxInt(u32))), .args = @intCast(args) };
        p.used += 1;
    }

    /// Called when the host has queued the last launch of a pass and is about to wait.
    pub fn enqueued(p: *Profile) void {
        p.enqueued_at = nowNs();
    }

    fn row(p: *Profile, handle: ze.H, lab: []const u8) !*Row {
        for (p.names.items) |n| if (n.handle == handle and n.label.ptr == lab.ptr and n.label.len == lab.len) return &p.rows.items[n.row];
        var buf: [64]u8 = @splat(0);
        var size: usize = 0;
        const d = p.ctx.d;
        try d.check(d.x.zeKernelGetName(handle, &size, null), "zeKernelGetName");
        var full: [256]u8 = undefined;
        if (size > full.len) return error.Invalid;
        try d.check(d.x.zeKernelGetName(handle, &size, &full), "zeKernelGetName");
        const name = std.mem.sliceTo(full[0..size], 0);
        for (p.rows.items, 0..) |*r, i| if (std.mem.eql(u8, std.mem.sliceTo(&r.name, 0), name) and std.mem.eql(u8, r.label, lab)) {
            try p.names.append(gpa, .{ .handle = handle, .label = lab, .row = i });
            return r;
        };
        @memcpy(buf[0..@min(name.len, 63)], name[0..@min(name.len, 63)]);
        try p.rows.append(gpa, .{ .name = buf, .label = lab });
        try p.names.append(gpa, .{ .handle = handle, .label = lab, .row = p.rows.items.len - 1 });
        return &p.rows.items[p.rows.items.len - 1];
    }

    /// Adds the launches since the last collect to the totals, once the stream has finished them; one pass.
    pub fn collect(p: *Profile) Error!void {
        const d = p.ctx.d;
        const res: f64 = 1e9 / @as(f64, @floatFromInt(p.ctx.gpu.props.timerResolution));
        const bits = p.ctx.gpu.props.kernelTimestampValidBits;
        const mask: u64 = if (bits >= 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(bits)) - 1;
        const overlap = std.c.getenv("TF_TP_OVERLAP");
        const clock = if (overlap != null and p.device_time) hostClock(p.ctx) else null;
        var overlap_fd: i32 = -1;
        if (overlap) |path| {
            if (clock != null) overlap_fd = std.c.open(path, @bitCast(@as(u32, 1 | 64 | 1024)), @as(c_uint, 0o644));
        }
        var first: u64 = 0;
        var last_end: u64 = 0;
        var prev_end: u64 = 0;
        var busy: f64 = 0;
        var gap_ns: u64 = 0;
        var reduce_gap_ns: u64 = 0;
        var reduce_gaps: u64 = 0;
        var big: [8]struct { ns: u64, name: [24]u8 } = @splat(.{ .ns = 0, .name = @splat(0) });
        var prev_row: ?*Row = null;
        for (p.recs[0..p.used], 0..) |rec, i| {
            const r = p.row(rec.handle, rec.label) catch return error.Invalid;
            r.calls += 1;
            r.set_ns += rec.set_ns;
            r.append_ns += rec.append_ns;
            r.args += rec.args;
            if (p.device_time) {
                var ts: ze.KernelTimestampResult = .{};
                try d.check(d.api.zeEventQueryKernelTimestamp(p.events[i], &ts), "zeEventQueryKernelTimestamp");
                const dur = (ts.global_end -% ts.global_start) & mask;
                const ns: u64 = @intFromFloat(@as(f64, @floatFromInt(dur)) * res);
                r.device_ns += ns;
                busy += @floatFromInt(ns);
                if (overlap_fd >= 0) if (clock) |c| {
                    var line: [80]u8 = undefined;
                    if (std.fmt.bufPrint(&line, "{d} {d}\n", .{ c.hostNs(ts.global_start), c.hostNs(ts.global_end) })) |s| {
                        _ = std.c.write(overlap_fd, s.ptr, s.len);
                    } else |_| {}
                };
                if (i == 0) first = ts.global_start else {
                    const gap: u64 = @intFromFloat(@as(f64, @floatFromInt((ts.global_start -% prev_end) & mask)) * res);
                    gap_ns += gap;
                    const name = std.mem.sliceTo(&r.name, 0);
                    if (std.mem.startsWith(u8, name, "tp_")) {
                        reduce_gap_ns += gap;
                        reduce_gaps += 1;
                    }
                    var slot: usize = 8;
                    for (big, 0..) |b, k| if (gap > b.ns) {
                        slot = k;
                        break;
                    };
                    if (slot < 8) {
                        var j: usize = 7;
                        while (j > slot) : (j -= 1) big[j] = big[j - 1];
                        big[slot] = .{ .ns = gap, .name = @splat(0) };
                        @memcpy(big[slot].name[0..@min(name.len, 23)], name[0..@min(name.len, 23)]);
                    }
                }
                prev_end = ts.global_end;
                last_end = ts.global_end;
                // the idle time before this launch belongs to the launch before it, in this pass
                if (prev_row) |pr| pr.gap_ns += @intFromFloat(@as(f64, @floatFromInt((ts.global_start -% prev_end) & mask)) * res);
                prev_row = r;
                prev_end = ts.global_end;
                try d.check(d.api.zeEventHostReset(p.events[i]), "zeEventHostReset");
            }
        }
        if (p.device_time and std.c.getenv("TF_TP_GAPS") != null) {
            std.log.info("inkling2: idle between kernels {d:.2} ms; before tp_* {d} gaps {d:.2} ms", .{ @as(f64, @floatFromInt(gap_ns)) / 1e6, reduce_gaps, @as(f64, @floatFromInt(reduce_gap_ns)) / 1e6 });
            for (big) |b| if (b.ns > 0) std.log.info("inkling2: gap {d:.1} us before {s}", .{ @as(f64, @floatFromInt(b.ns)) / 1e3, std.mem.sliceTo(&b.name, 0) });
        }
        if (p.device_time and p.used > 0) {
            p.device_busy_ns += @intFromFloat(busy);
            p.device_span_ns += @intFromFloat(@as(f64, @floatFromInt((last_end -% first) & mask)) * res);
        }
        p.launches += p.used;
        p.tokens += 1;
        p.used = 0;
        if (overlap_fd >= 0) _ = std.c.close(overlap_fd);
    }

    /// Forgets every total (the capacity stays).
    pub fn reset(p: *Profile) void {
        p.rows.clearRetainingCapacity();
        p.names.clearRetainingCapacity();
        p.launches = 0;
        p.device_busy_ns = 0;
        p.device_span_ns = 0;
        p.tokens = 0;
        p.used = 0;
    }

    fn lessDevice(_: void, a: Row, b: Row) bool {
        return if (a.device_ns != b.device_ns) a.device_ns > b.device_ns else a.set_ns + a.append_ns > b.set_ns + b.append_ns;
    }

    /// The table: per kernel name, calls, device time, host time, per pass.
    pub fn print(p: *Profile) void {
        const n: f64 = @floatFromInt(@max(p.tokens, 1));
        std.mem.sort(Row, p.rows.items, {}, lessDevice);
        std.debug.print("{s:<40} {s:>7} {s:>10} {s:>9} {s:>9} {s:>10} {s:>10} {s:>6}\n", .{ "kernel", "calls", "dev ms", "dev us/c", "gap us/c", "set ms", "append ms", "args" });
        var calls: u64 = 0;
        var dev: u64 = 0;
        var set: u64 = 0;
        var app: u64 = 0;
        var args: u64 = 0;
        for (p.rows.items) |r| {
            const c: f64 = @floatFromInt(@max(r.calls, 1));
            var full_name: [96]u8 = undefined;
            const shown = if (r.label.len > 0) (std.fmt.bufPrint(&full_name, "{s}:{s}", .{ r.label, std.mem.sliceTo(&r.name, 0) }) catch "?") else std.mem.sliceTo(&r.name, 0);
            std.debug.print("{s:<40} {d:>7.1} {d:>10.3} {d:>9.1} {d:>9.1} {d:>10.3} {d:>10.3} {d:>6.1}\n", .{ shown, @as(f64, @floatFromInt(r.calls)) / n, @as(f64, @floatFromInt(r.device_ns)) / 1e6 / n, @as(f64, @floatFromInt(r.device_ns)) / 1e3 / c, @as(f64, @floatFromInt(r.gap_ns)) / 1e3 / c, @as(f64, @floatFromInt(r.set_ns)) / 1e6 / n, @as(f64, @floatFromInt(r.append_ns)) / 1e6 / n, @as(f64, @floatFromInt(r.args)) / n });
            calls += r.calls;
            dev += r.device_ns;
            set += r.set_ns;
            app += r.append_ns;
            args += r.args;
        }
        std.debug.print("{s:<40} {d:>7.1} {d:>10.3} {s:>9} {d:>10.3} {d:>10.3} {d:>6.1}\n", .{ "all (a pass)", @as(f64, @floatFromInt(calls)) / n, @as(f64, @floatFromInt(dev)) / 1e6 / n, "", @as(f64, @floatFromInt(set)) / 1e6 / n, @as(f64, @floatFromInt(app)) / 1e6 / n, @as(f64, @floatFromInt(args)) / n });
        if (p.device_time) std.debug.print("device busy {d:.3} ms, first start to last end {d:.3} ms, idle between kernels {d:.3} ms (a pass); {d} launches not timed\n", .{ @as(f64, @floatFromInt(p.device_busy_ns)) / 1e6 / n, @as(f64, @floatFromInt(p.device_span_ns)) / 1e6 / n, @as(f64, @floatFromInt(p.device_span_ns -| p.device_busy_ns)) / 1e6 / n, p.dropped });
    }
};

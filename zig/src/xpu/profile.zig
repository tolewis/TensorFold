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

/// Every kernel queued through `launch.launchSignal` since the process started (indirect launches are not counted).
pub var launches: u64 = 0;

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
    /// The sum of the "gap us/c" column: start of a launch minus end of the launch before it, 0 when they overlap.
    device_gap_ns: u64 = 0,
    tokens: u64 = 0,
    /// Host clock (`nowNs`) of the parts of one forward: its start, the end of its uploads, its first and last launch
    /// append, and the end of its wait. The forward sets them; a tool reads them before the next forward.
    started_at: u64 = 0,
    uploaded_at: u64 = 0,
    first_append_at: u64 = 0,
    enqueued_at: u64 = 0,
    waited_at: u64 = 0,

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

    /// Called at the start of a forward, before its uploads.
    pub fn started(p: *Profile) void {
        p.started_at = nowNs();
        p.uploaded_at = p.started_at;
        p.first_append_at = 0;
        p.enqueued_at = 0;
        p.waited_at = 0;
    }

    /// Called when the forward's uploads are queued.
    pub fn uploaded(p: *Profile) void {
        p.uploaded_at = nowNs();
    }

    /// Called when the wait of the forward on its stream has returned.
    pub fn waited(p: *Profile) void {
        p.waited_at = nowNs();
    }

    /// The host clock of the first launch append of a forward (the launch path calls it before each append).
    pub fn appending(p: *Profile, t: u64) void {
        if (p.first_append_at == 0) p.first_append_at = t;
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
                    // A launch can start before the launch before it ends (the two overlap on the clock). Its gap is 0, not
                    // the wrapped difference.
                    const diff = (ts.global_start -% prev_end) & mask;
                    const gap: u64 = if (diff > mask / 2) 0 else @intFromFloat(@as(f64, @floatFromInt(diff)) * res);
                    gap_ns += gap;
                    // the idle time before this launch belongs to the launch before it, in this pass
                    if (prev_row) |pr| pr.gap_ns += gap;
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
                last_end = ts.global_end;
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
            p.device_gap_ns += gap_ns;
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
        p.device_gap_ns = 0;
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
        if (p.device_time) std.debug.print("device busy {d:.3} ms, first start to last end {d:.3} ms, idle between kernels {d:.3} ms (span minus busy, a pass); sum of the gap column {d:.3} ms (a pass); {d} launches not timed\n", .{ @as(f64, @floatFromInt(p.device_busy_ns)) / 1e6 / n, @as(f64, @floatFromInt(p.device_span_ns)) / 1e6 / n, @as(f64, @floatFromInt(p.device_span_ns -| p.device_busy_ns)) / 1e6 / n, @as(f64, @floatFromInt(p.device_gap_ns)) / 1e6 / n, p.dropped });
    }
};

// V11: the time of a decode token that is not device time. `TF_TOKLOG=<file>` (off by default) writes one line a
// one-row decode token: host clock marks of the feed, the forward and the draw, and three device stamps (the forward's
// first kernel, its end, the sampler kernel) from zeCommandListAppendWriteGlobalTimestamp. A line is written in the
// device-busy part of a later token (at that forward's last append), so the write is not in the gap it measures.

/// The host marks of one token, in the order the code reaches them.
pub const Tok = enum(u8) { entry, fstart, uploaded, first, enqueued, waited, fret, sel_app, sel_sync, dl, draw, cp, ret };
const tok_marks = @as(usize, @intFromEnum(Tok.ret)) + 1;
/// Device stamp slots of a token: the forward's first kernel, the forward's end, the sampler kernel start and end.
pub const TokDev = enum(u8) { fstart, fend, sstart, send, estart, eend };
const tok_slots = 8;
const tok_ring = 4;

const TokRec = struct {
    t: [tok_marks]u64 = @splat(0),
    path: u8 = 0,
    /// The host clock (midpoint of the call) and the device clock of one zeDeviceGetGlobalTimestamps.
    cal_host: u64 = 0,
    cal_dev: u64 = 0,
    cal_ns: u64 = 0,
    stamped: [6]bool = @splat(false),
    gen: u64 = 0,
    /// Kernel launches of the token (forward and draw).
    launches: u32 = 0,
    /// TF_TOKLOG_EV=1: the forward's launches with a timestamp event each, device busy, first start to last end, idle.
    ev_launches: u64 = 0,
    ev_busy: u64 = 0,
    ev_span: u64 = 0,
    ev_gap: u64 = 0,
    /// TF_TOKLOG_EMPTY=<n>: n empty launches after the forward's end stamp.
    empties: u32 = 0,
    /// Host waits on the stream and bytes downloaded to the host, while the token is open.
    syncs: u32 = 0,
    dl_bytes: u64 = 0,
};

pub fn tokSync() void {
    tok_recs[tok_k % tok_ring].syncs += 1;
}

pub fn tokDownload(n: usize) void {
    tok_recs[tok_k % tok_ring].dl_bytes += n;
}

pub var tok_on: bool = false;
var tok_read: bool = false;
var tok_fd: c_int = -1;
var tok_recs: [tok_ring]TokRec = @splat(.{});
var tok_k: u64 = 0;
var tok_gen: u64 = 0;

/// A prefill starts: the next token does not follow the one before (no line spans two requests).
pub fn tokBreak() void {
    tok_gen += 1;
}
/// Pinned host memory of `tok_ring * 4` u64 device stamps (set by the forward on the first token).
pub var tok_dev: ?[*]u64 = null;
/// True while a one-row decode forward or its draw runs: launch marks and device stamps go to the current token.
pub var tok_live: bool = false;

/// Reads TF_TOKLOG one time a process.
pub fn tokInit() void {
    if (tok_read) return;
    tok_read = true;
    const path = std.c.getenv("TF_TOKLOG") orelse return;
    tok_fd = platform.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(c_uint, 0o644));
    tok_on = tok_fd >= 0;
    if (tok_on) tokWrite("TOKLOG v11: us from the device end of the token's forward (fend). path 1 queue, 2 verify, 3 tool\n");
}

fn tokWrite(s: []const u8) void {
    if (tok_fd >= 0) _ = std.c.write(tok_fd, s.ptr, s.len);
}

/// A feed starts: a new token record. `path` 1 queueFn, 2 verifyFn (a one-row window, the served path), 3 the logits tool.
pub fn tokEntry(path: u8) void {
    tokInit();
    if (!tok_on) return;
    tok_k += 1;
    const r = &tok_recs[tok_k % tok_ring];
    r.* = .{ .path = path, .gen = tok_gen };
    r.t[@intFromEnum(Tok.entry)] = nowNs();
    tok_live = true;
}

/// Sets mark `m` of the current token, the first time only.
pub fn tokMark(m: Tok) void {
    if (!tok_on or !tok_live) return;
    const r = &tok_recs[tok_k % tok_ring];
    if (r.t[@intFromEnum(m)] == 0) r.t[@intFromEnum(m)] = nowNs();
}

/// The current token's slot for device stamp `d`, or null when the log is off.
pub fn tokSlot(d: TokDev) ?*u64 {
    if (!tok_on or !tok_live) return null;
    const p = tok_dev orelse return null;
    tok_recs[tok_k % tok_ring].stamped[@intFromEnum(d)] = true;
    return &p[(tok_k % tok_ring) * tok_slots + @intFromEnum(d)];
}

/// A kernel launch of the current token: the first sets the `first` mark.
pub fn tokLaunch() void {
    const r = &tok_recs[tok_k % tok_ring];
    if (r.t[@intFromEnum(Tok.first)] == 0) r.t[@intFromEnum(Tok.first)] = nowNs();
    r.launches += 1;
}

var tok_ev_read = false;
var tok_ev_on = false;
var tok_empty_n: u32 = 0;
var tok_prof: ?Profile = null;

fn tokEvRead() void {
    if (tok_ev_read) return;
    tok_ev_read = true;
    if (std.c.getenv("TF_TOKLOG_EV")) |p| tok_ev_on = p[0] == '1';
    if (std.c.getenv("TF_TOKLOG_EMPTY")) |p| tok_empty_n = std.fmt.parseInt(u32, std.mem.span(p), 10) catch 0;
}

/// TF_TOKLOG_EMPTY: the count of empty launches a token, 0 when off.
pub fn tokEmpties() u32 {
    if (!tok_on or !tok_live) return 0;
    tokEvRead();
    return tok_empty_n;
}

/// TF_TOKLOG_EV=1: a profile with a timestamp event on every launch of this token's forward (installed if none is).
pub fn tokEvBegin(ctx: *const Context) Error!bool {
    if (!tok_on or !tok_live) return false;
    tokEvRead();
    if (!tok_ev_on or active != null) return false;
    if (tok_prof == null) tok_prof = try Profile.init(ctx, 4096, true);
    active = &tok_prof.?;
    return true;
}

/// After the forward's wait: the event totals of the token into its record, then the profile is removed.
pub fn tokEvEnd() Error!void {
    const p = &(tok_prof orelse return);
    active = null;
    try p.collect();
    const r = &tok_recs[tok_k % tok_ring];
    r.ev_launches = p.launches;
    r.ev_busy = p.device_busy_ns;
    r.ev_span = p.device_span_ns;
    r.ev_gap = p.device_gap_ns;
    p.reset();
}

pub fn tokSetEmpties(n: u32) void {
    tok_recs[tok_k % tok_ring].empties = n;
}

/// Empty launches go uncounted and without events.
pub fn tokQuiet(on: bool) void {
    tok_live = !on;
}

/// The feed has returned its token.
pub fn tokRet() void {
    tokMark(.ret);
    tok_live = false;
}

const GlobalTs = *const fn (ze.H, *u64, *u64) callconv(.c) u32;
var tok_gts: ?GlobalTs = null;

/// At the forward's last append of token k: the clock pair of token k, then the line of token k - 2 (all its stamps
/// and those of token k - 1 have landed: token k - 1's wait has returned).
pub fn tokEmit(ctx: *const Context) void {
    if (!tok_on or !tok_live) return;
    const r = &tok_recs[tok_k % tok_ring];
    if (tok_gts == null) tok_gts = @constCast(ctx.d).lib.lookup(GlobalTs, "zeDeviceGetGlobalTimestamps");
    if (tok_gts) |f| {
        var h: u64 = 0;
        var d: u64 = 0;
        const a = nowNs();
        if (f(ctx.gpu.device, &h, &d) == 0) {
            const b = nowNs();
            r.cal_host = a + (b - a) / 2;
            r.cal_dev = d;
            r.cal_ns = b - a;
        }
    }
    if (tok_k < 3) return;
    const p = tok_dev orelse return;
    const o = &tok_recs[(tok_k - 2) % tok_ring];
    const n = &tok_recs[(tok_k - 1) % tok_ring];
    if (o.gen != n.gen or o.cal_ns == 0 or !o.stamped[1] or !n.stamped[0]) return;
    const bits = ctx.gpu.props.timestampValidBits;
    const mask: u64 = if (bits >= 64 or bits == 0) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(bits)) - 1;
    const tick_ns = 1e9 / @as(f64, @floatFromInt(ctx.gpu.props.timerResolution));
    const half = (mask >> 1) + 1;
    const devHost = struct {
        fn f(cal: *const TokRec, v: u64, m: u64, hf: u64, ns: f64) f64 {
            const diff = (v -% cal.cal_dev) & m;
            const signed: f64 = if (diff >= hf) -@as(f64, @floatFromInt((m - diff) + 1)) else @as(f64, @floatFromInt(diff));
            return @as(f64, @floatFromInt(cal.cal_host)) + signed * ns;
        }
    }.f;
    const ob = ((tok_k - 2) % tok_ring) * tok_slots;
    const nb = ((tok_k - 1) % tok_ring) * tok_slots;
    const e = devHost(o, p[ob + 1], mask, half, tick_ns);
    const us = struct {
        fn f(t: u64, base: f64) f64 {
            return if (t == 0) -1 else (@as(f64, @floatFromInt(t)) - base) / 1e3;
        }
    }.f;
    var buf: [900]u8 = undefined;
    const empty_us: f64 = if (o.stamped[4] and o.stamped[5] and o.empties > 0) @as(f64, @floatFromInt((p[ob + 5] -% p[ob + 4]) & mask)) * tick_ns / 1e3 else -1;
    const fwd_us = @as(f64, @floatFromInt((p[ob + 1] -% p[ob]) & mask)) * tick_ns / 1e3;
    const sel_s: f64 = if (o.stamped[2]) (devHost(o, p[ob + 2], mask, half, tick_ns) - e) / 1e3 else -1;
    const sel_e: f64 = if (o.stamped[3]) (devHost(o, p[ob + 3], mask, half, tick_ns) - e) / 1e3 else -1;
    const next_dev = (devHost(o, p[nb], mask, half, tick_ns) - e) / 1e3;
    const T = struct {
        fn at(rec: *const TokRec, m: Tok) u64 {
            return rec.t[@intFromEnum(m)];
        }
    };
    const line = std.fmt.bufPrint(&buf, "TOK {d} gen {d} path {d} fwd_dev {d:.1} launches {d} ev_launches {d} ev_busy {d:.1} ev_span {d:.1} ev_gap {d:.1} empties {d} empty_us {d:.2} syncs {d} dl_bytes {d} | waited {d:.1} fret {d:.1} sel_app {d:.1} sel_dev_start {d:.1} sel_dev_end {d:.1} sel_sync {d:.1} dl {d:.1} draw {d:.1} cp {d:.1} ret {d:.1} | next entry {d:.1} fstart {d:.1} uploaded {d:.1} first {d:.1} dev_start {d:.1} | own fstart {d:.1} uploaded {d:.1} first {d:.1} enqueued {d:.1} | cal {d} ns\n", .{
        tok_k - 2,                     o.gen,                         o.path,                        fwd_us,
        o.launches,                    o.ev_launches,                 @as(f64, @floatFromInt(o.ev_busy)) / 1e3,
        @as(f64, @floatFromInt(o.ev_span)) / 1e3, @as(f64, @floatFromInt(o.ev_gap)) / 1e3, o.empties,
        empty_us,                      o.syncs,                       o.dl_bytes,
        us(T.at(o, .waited), e),       us(T.at(o, .fret), e),         us(T.at(o, .sel_app), e),
        sel_s,                         sel_e,                         us(T.at(o, .sel_sync), e),
        us(T.at(o, .dl), e),           us(T.at(o, .draw), e),         us(T.at(o, .cp), e),
        us(T.at(o, .ret), e),
        us(T.at(n, .entry), e),        us(T.at(n, .fstart), e),       us(T.at(n, .uploaded), e),
        us(T.at(n, .first), e),        next_dev,                      us(T.at(o, .fstart), e),
        us(T.at(o, .uploaded), e),     us(T.at(o, .first), e),        us(T.at(o, .enqueued), e),
        o.cal_ns,
    }) catch return;
    tokWrite(line);
}

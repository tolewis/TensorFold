//! Kernel launches: every argument set in order, then the grid queued on a stream.

const std = @import("std");
const ze = @import("ze");
const Kernel = @import("module.zig").Kernel;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const Error = @import("context.zig").Error;
const profile = @import("profile.zig");

/// How a launch argument is passed to the kernel parameter at its position.
pub const ArgKind = enum { buffer, pointer, float, integer };

pub fn argKind(comptime T: type) ArgKind {
    if (T == DeviceBuffer) return .buffer;
    if (T == ?*anyopaque) return .pointer;
    if (T == f32 or T == comptime_float) return .float;
    return switch (@typeInfo(T)) {
        .int, .comptime_int => .integer,
        else => @compileError("kernel argument of type " ++ @typeName(T)),
    };
}

/// Queues `k` over `groups` work-groups; the group size was fixed when the kernel was made.
pub fn launch(k: Kernel, s: Stream, groups: [3]u32) Error!void {
    return launchSignal(k, s, groups, null);
}

fn launchSignal(k: Kernel, s: Stream, groups: [3]u32, signal: ze.H) Error!void {
    const g: ze.GroupCount = .{ .x = groups[0], .y = groups[1], .z = groups[2] };
    if (g.x == 0 or g.y == 0 or g.z == 0) return error.Invalid;
    const t0 = profile.nowNs();
    try k.ctx.d.check(k.ctx.d.api.zeCommandListAppendLaunchKernel(s.list, k.handle, &g, signal, 0, null), "zeCommandListAppendLaunchKernel");
    profile.launches += 1;
    profile.noteAppend(4, profile.nowNs() - t0);
}

const IndirectFn = *const fn (ze.H, ze.H, *const anyopaque, ze.H, u32, ?[*]ze.H) callconv(.c) u32;

/// A launch must set every argument of the kernel and no more (p3 Y-args: a tuple one short put a row count in a
/// pointer argument and the kernel wrote through it). Checked before the first set, so a wrong tuple appends nothing.
pub fn checkArgCount(k: Kernel, have: usize) Error!void {
    const want = k.n_args orelse return;
    if (have != want) {
        std.log.err("launch of {s} with {d} arguments, the kernel takes {d}", .{ k.name, have, want });
        return error.ArgCount;
    }
}

/// Sets the arguments, then launches with the group count in device memory. Y is this rank's route count.
pub fn dispatchIndirect(k: Kernel, s: Stream, groups: *anyopaque, args: anytype) Error!void {
    try checkArgCount(k, args.len);
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
        switch (comptime argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
    const fnp = @constCast(k.ctx.d).lib.lookup(IndirectFn, "zeCommandListAppendLaunchKernelIndirect") orelse return error.MissingSymbol;
    try k.ctx.d.check(fnp(s.list, k.handle, groups, null, 0, null), "zeCommandListAppendLaunchKernelIndirect");
}

/// Sets every argument of the tuple in order (buffer, device pointer, f32 or u32) and queues the kernel.
pub fn dispatch(k: Kernel, s: Stream, groups: [3]u32, args: anytype) Error!void {
    try checkArgCount(k, args.len);
    if (profile.active) |p| return dispatchProfiled(p, k, s, groups, args);
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
        switch (comptime argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
    profile.last_kernel = k.name;
    try launch(k, s, groups);
    try traceLaunch(k, s, groups, args);
}

fn dispatchProfiled(p: *profile.Profile, k: Kernel, s: Stream, groups: [3]u32, args: anytype) Error!void {
    const t0 = profile.nowNs();
    p.appending(t0);
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
        switch (comptime argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
    const t1 = profile.nowNs();
    profile.last_kernel = k.name;
    try launchSignal(k, s, groups, p.nextEvent());
    p.record(k.handle, t1 - t0, profile.nowNs() - t1, args.len);
    try traceLaunch(k, s, groups, args);
}

fn traceLaunch(k: Kernel, s: Stream, groups: [3]u32, args: anytype) Error!void {
    const path = std.c.getenv("TF_TP_LAUNCHLOG") orelse return;
    var buf: [768]u8 = undefined;
    var n: usize = 0;
    const head = std.fmt.bufPrint(buf[n..], "{s} {d},{d},{d}", .{ k.name, groups[0], groups[1], groups[2] }) catch return;
    n += head.len;
    inline for (args) |v| {
        const bit = switch (comptime argKind(@TypeOf(v))) {
            .integer => std.fmt.bufPrint(buf[n..], " {d}", .{v}) catch break,
            .float => std.fmt.bufPrint(buf[n..], " {d:.4}", .{v}) catch break,
            else => std.fmt.bufPrint(buf[n..], " p", .{}) catch break,
        };
        n += bit.len;
    }
    if (n + 1 < buf.len) {
        buf[n] = '\n';
        n += 1;
    }
    const fd = std.c.open(path, @bitCast(@as(u32, 1 | 64 | 1024)), @as(c_uint, 0o644));
    if (fd >= 0) {
        _ = std.c.write(fd, &buf, n);
        _ = std.c.fsync(fd);
        _ = std.c.close(fd);
    }
    try s.synchronize();
}

test "argument kinds follow the Zig type" {
    try std.testing.expectEqual(ArgKind.buffer, argKind(DeviceBuffer));
    try std.testing.expectEqual(ArgKind.pointer, argKind(?*anyopaque));
    try std.testing.expectEqual(ArgKind.float, argKind(f32));
    try std.testing.expectEqual(ArgKind.integer, argKind(u32));
    try std.testing.expectEqual(ArgKind.integer, argKind(comptime_int));
    try std.testing.expectEqual(ArgKind.integer, argKind(usize));
}

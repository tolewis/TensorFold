//! The XPU collective over host memory: `host.zig`'s ranks answering as `Kind.xpu`, the entry the IPC group replaces.
const std = @import("std");
const collective = @import("collective.zig");
const host = @import("host.zig");

pub const max_ranks = host.max_ranks;
pub const Group = host.Group;
pub const Rank = host.Rank;
pub const Options = host.Options;
pub const Control = host.Control;

/// A group of XPU ranks over host memory, answering as `Kind.xpu`.
pub fn init(world: u32) Group {
    var g = Group.init(world);
    g.kind = .xpu;
    return g;
}

/// Marks rank `r` dead, as the transport does when its process exits or its IPC handle fails.
pub fn kill(g: *Group, r: u32) void {
    g.kill(r);
}

/// Runs `body` on `world` XPU ranks at once and returns the first error any rank hit.
pub fn run(world: u32, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    return runWith(world, .{}, ctx, body);
}

/// `run` with a timeout, a peer deadline, or a planted fault; the kind is always `.xpu`.
pub fn runWith(world: u32, opts: Options, ctx: anytype, comptime body: fn (collective.Collective, u32, @TypeOf(ctx)) anyerror!void) !void {
    var o = opts;
    o.kind = .xpu;
    return host.runWith(world, o, ctx, body);
}

test "the group answers as the xpu kind" {
    var g = init(2);
    var r = g.rank(1);
    try std.testing.expectEqual(collective.Kind.xpu, r.iface().kind());
    try std.testing.expectEqual(@as(u32, 1), r.iface().rank());
    try std.testing.expectEqual(@as(u32, 2), r.iface().world());
}

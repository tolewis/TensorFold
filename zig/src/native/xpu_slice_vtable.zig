//! XPU_PROMPT_SLICE's two backend vtables (xpu_lanes.zig): whole prompt passes, or a chunk a call. Both are
//! comptime constants, so the pointer `of` returns is static. Window 1 (2026-11-03) crashed rank 0: the sliced
//! vtable was built in a block at run time and backend() returned a pointer into its own stack frame.
const lanes = @import("lanes");
const be = lanes.backend;
const VTable = be.Backend.VTable;

pub fn Vtables(comptime whole: VTable, comptime step: *const fn (ptr: *anyopaque, s: *lanes.Stream) anyerror!bool) type {
    return struct {
        pub const plain: VTable = whole;
        pub const sliced: VTable = blk: {
            var v = whole;
            v.prefill_step = step;
            v.reports_fill = true;
            break :blk v;
        };
        pub fn of(slice: bool) *const VTable {
            return if (slice) &sliced else &plain;
        }
    };
}

const std = @import("std");

const Probe = struct {
    steps: u32 = 0,
    fn self(p: *anyopaque) *Probe {
        return @ptrCast(@alignCast(p));
    }
    fn prefill(_: *anyopaque, _: *lanes.Stream) anyerror!void {}
    fn step(p: *anyopaque, _: *lanes.Stream) anyerror!bool {
        self(p).steps += 1;
        return true;
    }
    fn first(_: *anyopaque, _: *lanes.Stream, _: u64) anyerror!u64 {
        return 0;
    }
    fn queue(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return 0;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return 0;
    }
    fn verify(_: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {}
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
    fn release(_: *anyopaque, _: *lanes.Stream) void {}
    const V = Vtables(.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release }, step);

    /// As Lanes.backend(): the Backend built in its own frame and returned.
    noinline fn backend(p: *Probe, slice: bool) be.Backend {
        return .{ .ptr = p, .vtable = V.of(slice) };
    }
};

/// Writes over the stack the returned frame used, as the next calls of the engine thread do.
noinline fn clobber() u64 {
    var junk: [4096]u8 = undefined;
    @memset(@as([]volatile u8, &junk), 0xAA);
    var sum: u64 = 0;
    for (@as([]volatile u8, &junk)) |x| sum += x;
    return sum;
}

test "slice vtable: the pointer backend() returns stays valid after it returns, sliced and whole" {
    var p: Probe = .{};
    const sliced = p.backend(true);
    const whole = p.backend(false);
    std.mem.doNotOptimizeAway(clobber());
    try std.testing.expect(sliced.vtable.prefill_step != null);
    try std.testing.expect(sliced.vtable.reports_fill);
    try std.testing.expect(whole.vtable.prefill_step == null);
    try std.testing.expect(!whole.vtable.reports_fill);
    var s: lanes.Stream = undefined;
    try std.testing.expect(try sliced.vtable.prefill_step.?(sliced.ptr, &s));
    try std.testing.expectEqual(@as(u32, 1), p.steps);
    try whole.vtable.prefill(whole.ptr, &s);
    // static: both live in the binary's data, not a stack frame
    try std.testing.expect(sliced.vtable == Probe.V.of(true) and whole.vtable == Probe.V.of(false));
}

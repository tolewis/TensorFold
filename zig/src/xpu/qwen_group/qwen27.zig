//! Symbols the e5506ae Group reads. D is the Inkling row width, so allReduce's rows * D is an Inkling row.
const std = @import("std");

pub const alloc = std.heap.c_allocator;
pub const D: u32 = 2304;
pub const TP: u32 = 2;
pub var RANK: u32 = 0;
pub const VOCAB: u32 = 248320;

pub fn setRank(r: u32) void {
    RANK = r;
}

pub fn maxRows() u32 {
    return 64;
}

pub fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, @ptrCast(&ts));
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

pub fn modeWord() u64 {
    return 1;
}

pub const Stat = struct {
    on: bool = false,
    peer_s: f64 = 0,
    copies: u64 = 0,
};
pub var stat: Stat = .{};

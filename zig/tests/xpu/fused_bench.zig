//! tf-xpu-fused-bench: the Part 1 benches of lane fused-visit (Recipe lanes/inkspeed/fused-visit/DESIGN.md section 8).
//! `tf-xpu-fused-bench <fused_bench.bin> <fused_bench128.bin> --device N --bench props,phase,work,wgbar,launch,q8,stream,l2keep`
//! Each number is a host wall time from the first append to the end of a stream synchronize, the median of 5 runs, and a
//! slope between two work counts, so the launch cost drops out. Every line starts with BENCH.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");

const reps = 5;
// The spin cap in tm0 ticks (brief section 6 rule 1, a faulted launch ends in 100 ms or less). tm0 counts at the GT
// clock, and the clock moves (review C1: fb_tsc read 1,946.7 and 2,794.9 ticks a us in two runs of block 4). So the
// cap is 90,000 us at the lowest clock of the card (sysfs rpn_freq), and less at every faster clock. Block 4 had
// 50,000 us at the fb_tsc rate, 349 ms at 400 MHz. Block 2 had a cap of 20,000 polls (M-f 121 ms at G 32).
// 50,000 us at the fb_tsc rate, 349 ms at 400 MHz. Block 2 had a cap of 20,000 polls (M-f 121 ms at G 32).
// The cap of one wait at the lowest clock of the card (rpn). Block fix1 (2026-11-27 20:22Z) showed a forced launch
// that ends up to 5.1 ms after its first cap at 2,800 MHz (M-f G 1,024: 17.98 ms against a 12.86 ms cap). A launch
// at rpn takes up to 7 times as long, so a cap of 50 ms at rpn keeps the whole forced launch under 100 ms at rpn.
const cap_low_us: f64 = 50_000;
/// The GT clock range of the opened card in MHz (sysfs tile0/gt0/freq0 of its PCI address).
const Freq = struct { rpn: u32 = 0, rp0: u32 = 0, min: u32 = 0, max: u32 = 0 };
var card_freq: Freq = .{};
var freq_dir_buf: [128]u8 = undefined;
var freq_dir: []const u8 = "";
// The cap of one wait at the lowest clock of the card (rpn). Block fix1 (2026-11-27 20:22Z) showed a forced launch
// that ends up to 5.1 ms after its first cap at 2,800 MHz (M-f G 1,024: 17.98 ms against a 12.86 ms cap). A launch
// at rpn takes up to 7 times as long, so a cap of 50 ms at rpn keeps the whole forced launch under 100 ms at rpn.
const cap_low_us: f64 = 50_000;
/// The GT clock range of the opened card in MHz (sysfs tile0/gt0/freq0 of its PCI address).
const Freq = struct { rpn: u32 = 0, rp0: u32 = 0, min: u32 = 0, max: u32 = 0 };
var card_freq: Freq = .{};
var freq_dir_buf: [128]u8 = undefined;
var freq_dir: []const u8 = "";
// The cap of one wait at the lowest clock of the card (rpn). Block fix1 (2026-11-27 20:22Z) showed a forced launch
// that ends up to 5.1 ms after its first cap at 2,800 MHz (M-f G 1,024: 17.98 ms against a 12.86 ms cap). A launch
// at rpn takes up to 7 times as long, so a cap of 50 ms at rpn keeps the whole forced launch under 100 ms at rpn.
const cap_low_us: f64 = 50_000;
/// The GT clock range of the opened card in MHz (sysfs tile0/gt0/freq0 of its PCI address).
const Freq = struct { rpn: u32 = 0, rp0: u32 = 0, min: u32 = 0, max: u32 = 0 };
var card_freq: Freq = .{};
var freq_dir_buf: [128]u8 = undefined;
var freq_dir: []const u8 = "";
fn readMhz(io: std.Io, name: []const u8) u32 {
    var pbuf: [192]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ freq_dir, name }) catch return 0;
    const txt = std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.page_allocator, .limited(4096)) catch return 0;
    defer std.heap.page_allocator.free(txt);
    return std.fmt.parseInt(u32, std.mem.trim(u8, txt, " \n"), 10) catch 0;
}

const PciGet = *const fn (ze.H, *PciProps) callconv(.c) u32;
const PciProps = extern struct {
    stype: u32 = 0x10008,
    pNext: ?*anyopaque = null,
    domain: u32 = 0,
    bus: u32 = 0,
    device: u32 = 0,
    function: u32 = 0,
    gen: i32 = -1,
    width: i32 = -1,
    bandwidth: i64 = -1,
};

/// Finds the sysfs clock folder of the opened card by its PCI address and reads the clock range (review C1).
fn readFreq(io: std.Io, ctx: *xpu.Context) !void {
    const get = @constCast(ctx.d).lib.lookup(PciGet, "zeDevicePciGetPropertiesExt") orelse return error.NoPciAddress;
    var pci: PciProps = .{};
    if (get(ctx.gpu.device, &pci) != 0) return error.NoPciAddress;
    freq_dir = try std.fmt.bufPrint(&freq_dir_buf, "/sys/bus/pci/devices/{x:0>4}:{x:0>2}:{x:0>2}.{x}/tile0/gt0/freq0", .{ pci.domain, pci.bus, pci.device, pci.function });
    card_freq = .{ .rpn = readMhz(io, "rpn_freq"), .rp0 = readMhz(io, "rp0_freq"), .min = readMhz(io, "min_freq"), .max = readMhz(io, "max_freq") };
}

/// Sets every argument of the tuple in order, as xpu.launch.dispatch does.
fn setAll(k: xpu.Kernel, args: anytype) !void {
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
        switch (comptime xpu.launch.argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
}

// The cap of one wait at the lowest clock of the card (rpn). Block fix1 (2026-11-27 20:22Z) showed a forced launch
// that ends up to 5.1 ms after its first cap at 2,800 MHz (M-f G 1,024: 17.98 ms against a 12.86 ms cap). A launch
// at rpn takes up to 7 times as long, so a cap of 50 ms at rpn keeps the whole forced launch under 100 ms at rpn.
const cap_low_us: f64 = 50_000;
/// The GT clock range of the opened card in MHz (sysfs tile0/gt0/freq0 of its PCI address).
const Freq = struct { rpn: u32 = 0, rp0: u32 = 0, min: u32 = 0, max: u32 = 0 };
var card_freq: Freq = .{};
var freq_dir_buf: [128]u8 = undefined;
var freq_dir: []const u8 = "";
fn readMhz(io: std.Io, name: []const u8) u32 {
    var pbuf: [192]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ freq_dir, name }) catch return 0;
    const txt = std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.page_allocator, .limited(4096)) catch return 0;
    defer std.heap.page_allocator.free(txt);
    return std.fmt.parseInt(u32, std.mem.trim(u8, txt, " \n"), 10) catch 0;
}

const PciGet = *const fn (ze.H, *PciProps) callconv(.c) u32;
const PciProps = extern struct {
    stype: u32 = 0x10008,
    pNext: ?*anyopaque = null,
    domain: u32 = 0,
    bus: u32 = 0,
    device: u32 = 0,
    lag: bool = false, // fb_phase runs: at a fault, print the groups whose stamp is behind the fault round
    function: u32 = 0,
    gen: i32 = -1,
    width: i32 = -1,
    bandwidth: i64 = -1,
};

/// Finds the sysfs clock folder of the opened card by its PCI address and reads the clock range (review C1).
fn readFreq(io: std.Io, ctx: *xpu.Context) !void {
    const get = @constCast(ctx.d).lib.lookup(PciGet, "zeDevicePciGetPropertiesExt") orelse return error.NoPciAddress;
    var pci: PciProps = .{};
    if (get(ctx.gpu.device, &pci) != 0) return error.NoPciAddress;
    freq_dir = try std.fmt.bufPrint(&freq_dir_buf, "/sys/bus/pci/devices/{x:0>4}:{x:0>2}:{x:0>2}.{x}/tile0/gt0/freq0", .{ pci.domain, pci.bus, pci.device, pci.function });
    card_freq = .{ .rpn = readMhz(io, "rpn_freq"), .rp0 = readMhz(io, "rp0_freq"), .min = readMhz(io, "min_freq"), .max = readMhz(io, "max_freq") };
}

/// Sets every argument of the tuple in order, as xpu.launch.dispatch does.
fn setAll(k: xpu.Kernel, args: anytype) !void {
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
    /// Review C2: the fault record of fused_sync.h and the sync words after the launch, one line.
    fn sayFault(b: *B, tag: []const u8, g: u32) !void {
        var w: [48]u8 = undefined;
        try b.fault.download(b.s, 0, &w);
        var f: [12]u32 = undefined;
        for (&f, 0..) |*v, i| v.* = std.mem.readInt(u32, w[i * 4 ..][0..4], .little);
        var c: [12]u8 = undefined;
        try b.sync.download(b.s, 0, c[0..4]);
        try b.sync.download(b.s, 64, c[4..8]);
        try b.sync.download(b.s, 32000 * 4, c[8..12]);
        const ticks = (@as(u64, f[8]) << 32) | f[7];
        const tf: f64 = @floatFromInt(ticks);
        std.debug.print("BENCH fault record {s} G {d}: round {d}, first capped group {d}, its last poll {d}, entered {d} at its cap, ticks {d} ({d:.0} us at rpn {d} MHz, {d:.0} us at rp0 {d} MHz), polls {d}, groups capped {d}, groups left on the fault word {d}; after the launch counter {d}, generation {d}, entered {d}\n", .{
            tag,                                            g,                                        f[2], f[4], f[5], f[6], ticks,
            tf / @as(f64, @floatFromInt(@max(card_freq.rpn, 1))), card_freq.rpn, tf / @as(f64, @floatFromInt(@max(card_freq.rp0, 1))), card_freq.rp0,
            f[9],                                           f[10],                                    f[11],
            std.mem.readInt(u32, c[0..4], .little),         std.mem.readInt(u32, c[4..8], .little),   std.mem.readInt(u32, c[8..12], .little),
        });
        if (b.lag) {
            const sw = try std.heap.page_allocator.alloc(u8, @as(usize, g) * 64);
            defer std.heap.page_allocator.free(sw);
            try b.stamp.download(b.s, 0, sw);
            var behind: u32 = 0;
            var lo: u32 = std.math.maxInt(u32);
            var hi: u32 = 0;
            std.debug.print("BENCH fault stamps {s} G {d}: groups behind round {d}:", .{ tag, g, f[2] });
            for (0..g) |gi| {
                const st = std.mem.readInt(u32, sw[gi * 64 ..][0..4], .little);
                lo = @min(lo, st);
                hi = @max(hi, st);
                if (st < f[2]) {
                    if (behind < 8) std.debug.print(" {d} (stamp {d})", .{ gi, st });
                    behind += 1;
                }
            }
            std.debug.print("; {d} behind, stamps {d} to {d}\n", .{ behind, lo, hi });
        }
    }

        switch (comptime xpu.launch.argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
}

const gib: usize = 1 << 30;

const Shape = struct { g: u32, s: u32, grf: u32 };
/// DESIGN section 8: the shapes at 256 GRF, then at 128 GRF.
const shapes = [_]Shape{
    .{ .g = 16, .s = 32, .grf = 256 },  .{ .g = 32, .s = 16, .grf = 256 }, .{ .g = 32, .s = 32, .grf = 256 },
    .{ .g = 64, .s = 8, .grf = 256 },   .{ .g = 64, .s = 16, .grf = 256 }, .{ .g = 128, .s = 8, .grf = 256 },
    .{ .g = 1024, .s = 1, .grf = 256 }, .{ .g = 16, .s = 64, .grf = 128 }, .{ .g = 32, .s = 32, .grf = 128 },
    .{ .g = 32, .s = 64, .grf = 128 },
fn readMhz(io: std.Io, name: []const u8) u32 {
    var pbuf: [192]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ freq_dir, name }) catch return 0;
    const txt = std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.page_allocator, .limited(4096)) catch return 0;
    defer std.heap.page_allocator.free(txt);
    return std.fmt.parseInt(u32, std.mem.trim(u8, txt, " \n"), 10) catch 0;
}

const PciGet = *const fn (ze.H, *PciProps) callconv(.c) u32;
const PciProps = extern struct {
    stype: u32 = 0x10008,
    pNext: ?*anyopaque = null,
    domain: u32 = 0,
    bus: u32 = 0,
    device: u32 = 0,
    lag: bool = false, // fb_phase runs: at a fault, print the groups whose stamp is behind the fault round
    function: u32 = 0,
    gen: i32 = -1,
            try b.sayFault("timeCoop", g);
    width: i32 = -1,
    bandwidth: i64 = -1,
};

/// Finds the sysfs clock folder of the opened card by its PCI address and reads the clock range (review C1).
fn readFreq(io: std.Io, ctx: *xpu.Context) !void {
    const get = @constCast(ctx.d).lib.lookup(PciGet, "zeDevicePciGetPropertiesExt") orelse return error.NoPciAddress;
    var pci: PciProps = .{};
    if (get(ctx.gpu.device, &pci) != 0) return error.NoPciAddress;
    freq_dir = try std.fmt.bufPrint(&freq_dir_buf, "/sys/bus/pci/devices/{x:0>4}:{x:0>2}:{x:0>2}.{x}/tile0/gt0/freq0", .{ pci.domain, pci.bus, pci.device, pci.function });
    card_freq = .{ .rpn = readMhz(io, "rpn_freq"), .rp0 = readMhz(io, "rp0_freq"), .min = readMhz(io, "min_freq"), .max = readMhz(io, "max_freq") };
}

/// Sets every argument of the tuple in order, as xpu.launch.dispatch does.
fn setAll(k: xpu.Kernel, args: anytype) !void {
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
    /// Review C2: the fault record of fused_sync.h and the sync words after the launch, one line.
    fn sayFault(b: *B, tag: []const u8, g: u32) !void {
        var w: [48]u8 = undefined;
        try b.fault.download(b.s, 0, &w);
        var f: [12]u32 = undefined;
        for (&f, 0..) |*v, i| v.* = std.mem.readInt(u32, w[i * 4 ..][0..4], .little);
        var c: [12]u8 = undefined;
        try b.sync.download(b.s, 0, c[0..4]);
        try b.sync.download(b.s, 64, c[4..8]);
        try b.sync.download(b.s, 32000 * 4, c[8..12]);
        const ticks = (@as(u64, f[8]) << 32) | f[7];
        const tf: f64 = @floatFromInt(ticks);
        std.debug.print("BENCH fault record {s} G {d}: round {d}, first capped group {d}, its last poll {d}, entered {d} at its cap, ticks {d} ({d:.0} us at rpn {d} MHz, {d:.0} us at rp0 {d} MHz), polls {d}, groups capped {d}, groups left on the fault word {d}; after the launch counter {d}, generation {d}, entered {d}\n", .{
            tag,                                            g,                                        f[2], f[4], f[5], f[6], ticks,
            tf / @as(f64, @floatFromInt(@max(card_freq.rpn, 1))), card_freq.rpn, tf / @as(f64, @floatFromInt(@max(card_freq.rp0, 1))), card_freq.rp0,
            f[9],                                           f[10],                                    f[11],
            std.mem.readInt(u32, c[0..4], .little),         std.mem.readInt(u32, c[4..8], .little),   std.mem.readInt(u32, c[8..12], .little),
        });
        if (b.lag) {
            const sw = try std.heap.page_allocator.alloc(u8, @as(usize, g) * 64);
            defer std.heap.page_allocator.free(sw);
            try b.stamp.download(b.s, 0, sw);
            var behind: u32 = 0;
            var lo: u32 = std.math.maxInt(u32);
            var hi: u32 = 0;
            std.debug.print("BENCH fault stamps {s} G {d}: groups behind round {d}:", .{ tag, g, f[2] });
            for (0..g) |gi| {
                const st = std.mem.readInt(u32, sw[gi * 64 ..][0..4], .little);
                lo = @min(lo, st);
                hi = @max(hi, st);
    b.lag = true;
    defer b.lag = false;
                if (st < f[2]) {
                    if (behind < 8) std.debug.print(" {d} (stamp {d})", .{ gi, st });
                    behind += 1;
                }
            }
            std.debug.print("; {d} behind, stamps {d} to {d}\n", .{ behind, lo, hi });
        }
    }

        switch (comptime xpu.launch.argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
}

};
/// One sub-group a group, the B0 grid counts, for M-e and the M-a control.
const s1_counts = [_]u32{ 32, 64, 128, 256, 512, 1024 };
const mech_names = [_][]const u8{ "none", "a", "b", "c", "e", "f" };

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn readMhz(io: std.Io, name: []const u8) u32 {
    var pbuf: [192]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ freq_dir, name }) catch return 0;
    const txt = std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.page_allocator, .limited(4096)) catch return 0;
    defer std.heap.page_allocator.free(txt);
    return std.fmt.parseInt(u32, std.mem.trim(u8, txt, " \n"), 10) catch 0;
}

const PciGet = *const fn (ze.H, *PciProps) callconv(.c) u32;
const PciProps = extern struct {
    stype: u32 = 0x10008,
    pNext: ?*anyopaque = null,
    domain: u32 = 0,
    bus: u32 = 0,
    device: u32 = 0,
    lag: bool = false, // fb_phase runs: at a fault, print the groups whose stamp is behind the fault round
    function: u32 = 0,
    gen: i32 = -1,
            try b.sayFault("timeCoop", g);
    width: i32 = -1,
    bandwidth: i64 = -1,
};

/// Finds the sysfs clock folder of the opened card by its PCI address and reads the clock range (review C1).
fn readFreq(io: std.Io, ctx: *xpu.Context) !void {
    const get = @constCast(ctx.d).lib.lookup(PciGet, "zeDevicePciGetPropertiesExt") orelse return error.NoPciAddress;
    var pci: PciProps = .{};
    if (get(ctx.gpu.device, &pci) != 0) return error.NoPciAddress;
    freq_dir = try std.fmt.bufPrint(&freq_dir_buf, "/sys/bus/pci/devices/{x:0>4}:{x:0>2}:{x:0>2}.{x}/tile0/gt0/freq0", .{ pci.domain, pci.bus, pci.device, pci.function });
    card_freq = .{ .rpn = readMhz(io, "rpn_freq"), .rp0 = readMhz(io, "rp0_freq"), .min = readMhz(io, "min_freq"), .max = readMhz(io, "max_freq") };
}

/// Sets every argument of the tuple in order, as xpu.launch.dispatch does.
fn setAll(k: xpu.Kernel, args: anytype) !void {
    inline for (args, 0..) |v, i| {
        const idx: u32 = i;
    /// Review C2: the fault record of fused_sync.h and the sync words after the launch, one line.
    fn sayFault(b: *B, tag: []const u8, g: u32) !void {
        var w: [48]u8 = undefined;
        try b.fault.download(b.s, 0, &w);
        var f: [12]u32 = undefined;
        for (&f, 0..) |*v, i| v.* = std.mem.readInt(u32, w[i * 4 ..][0..4], .little);
        var c: [12]u8 = undefined;
        try b.sync.download(b.s, 0, c[0..4]);
        try b.sync.download(b.s, 64, c[4..8]);
        try b.sync.download(b.s, 32000 * 4, c[8..12]);
        const ticks = (@as(u64, f[8]) << 32) | f[7];
        const tf: f64 = @floatFromInt(ticks);
        std.debug.print("BENCH fault record {s} G {d}: round {d}, first capped group {d}, its last poll {d}, entered {d} at its cap, ticks {d} ({d:.0} us at rpn {d} MHz, {d:.0} us at rp0 {d} MHz), polls {d}, groups capped {d}, groups left on the fault word {d}; after the launch counter {d}, generation {d}, entered {d}\n", .{
            tag,                                            g,                                        f[2], f[4], f[5], f[6], ticks,
            tf / @as(f64, @floatFromInt(@max(card_freq.rpn, 1))), card_freq.rpn, tf / @as(f64, @floatFromInt(@max(card_freq.rp0, 1))), card_freq.rp0,
            f[9],                                           f[10],                                    f[11],
            std.mem.readInt(u32, c[0..4], .little),         std.mem.readInt(u32, c[4..8], .little),   std.mem.readInt(u32, c[8..12], .little),
        });
        if (b.lag) {
            const sw = try std.heap.page_allocator.alloc(u8, @as(usize, g) * 64);
            defer std.heap.page_allocator.free(sw);
            try b.stamp.download(b.s, 0, sw);
            var behind: u32 = 0;
            var lo: u32 = std.math.maxInt(u32);
            var hi: u32 = 0;
            std.debug.print("BENCH fault stamps {s} G {d}: groups behind round {d}:", .{ tag, g, f[2] });
            for (0..g) |gi| {
                const st = std.mem.readInt(u32, sw[gi * 64 ..][0..4], .little);
                lo = @min(lo, st);
                hi = @max(hi, st);
    b.lag = true;
    defer b.lag = false;
                if (st < f[2]) {
                    if (behind < 8) std.debug.print(" {d} (stamp {d})", .{ gi, st });
                    behind += 1;
                }
            }
            std.debug.print("; {d} behind, stamps {d} to {d}\n", .{ behind, lo, hi });
        }
    }

        switch (comptime xpu.launch.argKind(@TypeOf(v))) {
            .buffer => try k.setPtr(idx, v.ptr),
            .pointer => try k.setPtr(idx, v),
            .float => try k.setF32(idx, v),
            .integer => try k.setU32(idx, @intCast(v)),
        }
    }
}

fn median(v: *[reps]i96) i96 {
    std.mem.sort(i96, v, {}, std.sort.asc(i96));
    return v[reps / 2];
}

fn usOf(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

const B = struct {
    io: std.Io,
    ctx: *xpu.Context,
    s: xpu.Stream,
    m256: xpu.Module,
    m128: xpu.Module,
    big: xpu.DeviceBuffer, // 1 GiB, every weight window and data window
    sync: xpu.DeviceBuffer,
    stamp: xpu.DeviceBuffer,
    fault: xpu.DeviceBuffer,
    sink: xpu.DeviceBuffer,
    x: xpu.DeviceBuffer, // two bf16 rows of 2,304
    y: xpu.DeviceBuffer,
    deadline: i96, // the bench time limit, checked before each launch (brief section 6 rule 3)
    cfg_fault: bool = false, // set at the first fault of a configuration: its other launches do not run (rule 2)

    fn checkTime(b: *B) !void {
        if (nowNs(b.io) > b.deadline) {
            try b.sayFault("timeCoop", g);
            std.debug.print("BENCH time limit reached, the bench ends between launches\n", .{});
            return error.TimeLimit;
        }
    }

    fn module(b: *B, grf: u32) xpu.Module {
        return if (grf == 256) b.m256 else b.m128;
    }

    fn reset(b: *B) !void {
        try b.sync.fill8(b.s, 0);
        try b.stamp.fill8(b.s, 0);
        try b.fault.fill8(b.s, 0);
        try b.s.synchronize();
    }

    fn faultWords(b: *B) ![3]u32 {
        var w: [12]u8 = undefined;
        try b.fault.download(b.s, 0, &w);
        return .{ std.mem.readInt(u32, w[0..4], .little), std.mem.readInt(u32, w[4..8], .little), std.mem.readInt(u32, w[8..12], .little) };
    }
};

/// Median ns of up to `reps` cooperative launches of `k` over `g` groups with `args`. It stops at the first fault and
/// sets b.cfg_fault, and runs nothing while b.cfg_fault is set (brief section 6 rule 2). It refuses a launch above the
/// cooperative count of this kernel at its group size (rule 4) and checks the time limit before each launch (rule 3).
fn timeCoop(b: *B, k: xpu.Kernel, g: u32, args: anytype) !struct { ns: i96, faults: u32, checks: u32, round: u32 } {
    if (b.cfg_fault) return .{ .ns = 0, .faults = 1, .checks = 0, .round = 0 };
    const max = xpu.launch.suggestCooperative(k) catch 0;
    if (max < g) {
        std.debug.print("BENCH refused: {d} groups above the cooperative count {d}\n", .{ g, max });
        return error.AboveCooperativeCount;
    }
    var t: [reps]i96 = undefined;
    var faults: u32 = 0;
    var checks: u32 = 0;
    var round: u32 = 0;
    var n: usize = 0;
    for (0..reps) |r| {
        try b.checkTime();
        try b.reset();
        const t0 = nowNs(b.io);
        try xpu.launch.dispatchCooperative(k, b.s, .{ g, 1, 1 }, args);
        try b.s.synchronize();
        t[r] = nowNs(b.io) - t0;
        n = r + 1;
        const f = try b.faultWords();
    b.lag = true;
    defer b.lag = false;
        checks += f[1];
        if (f[0] != 0) {
            faults += 1;
            round = f[2];
            b.cfg_fault = true;
            break;
        }
    }
    std.mem.sort(i96, t[0..n], {}, std.sort.asc(i96));
    return .{ .ns = t[n / 2], .faults = faults, .checks = checks, .round = round };
}

/// The cooperative count at `local` work-items, or null when the driver refuses the group size.
fn coopCount(k: xpu.Kernel, local: u32) ?u32 {
    if (k.ctx.d.api.zeKernelSetGroupSize(k.handle, local, 1, 1) != 0) return null;
    return xpu.launch.suggestCooperative(k) catch null;
}

fn sayProps(b: *B, k: xpu.Kernel, name: []const u8, grf: u32) !void {
    var p: ze.KernelProperties = .{};
    try b.ctx.d.check(b.ctx.d.api.zeKernelGetProperties(k.handle, &p), "zeKernelGetProperties");
    std.debug.print("BENCH props {s} grf {d}: args {d}, required group {d}, sub-group {d}, max sub-groups {d}, local {d} B, private {d} B, spill {d} B\n", .{
        name,                          grf,                   p.numKernelArgs, @as(u64, p.requiredGroupSize[0]) * p.requiredGroupSize[1] * p.requiredGroupSize[2],
        p.requiredSubgroupSize,        p.maxNumSubgroups,     p.localMemSize,  p.privateMemSize,
        p.spillMemSize,
    });
}

const kernel_names = [_][:0]const u8{ "fb_phase", "fb_wgbar", "fb_empty", "fb_q8", "fb_stream", "fb_l2keep", "tf_ink2_dense_q8_1" };

fn benchProps(b: *B) !void {
    const dp = b.ctx.gpu.props;
    std.debug.print("BENCH device {s}: slices {d}, sub-slices a slice {d}, EUs a sub-slice {d}, threads an EU {d}, SIMD {d}, clock {d} MHz\n", .{
        std.mem.sliceTo(&dp.name, 0), dp.numSlices, dp.numSubslicesPerSlice, dp.numEUsPerSubslice, dp.numThreadsPerEU, dp.physicalEUSimdWidth, dp.coreClockRate,
    });
    for ([_]u32{ 256, 128 }) |grf| {
        const m = b.module(grf);
        for (kernel_names) |name| {
            var k = try m.kernelAt(name);
            defer k.deinit();
            try sayProps(b, k, name, grf);
            for ([_]u32{ 16, 128, 256, 512, 1024 }) |local| {
                if (coopCount(k, local)) |n| {
                    std.debug.print("BENCH coop {s} grf {d} local {d}: {d} groups ({d} sub-groups)\n", .{ name, grf, local, n, n * (local / 16) });
                } else std.debug.print("BENCH coop {s} grf {d} local {d}: refused\n", .{ name, grf, local });
            }
        }
    }
}

/// B2 (work_u4 0) and B3 (work_u4 4,096: 64 KB a group a round).
fn benchPhase(b: *B, work_u4: u32) !void {
    const rounds: u32 = 2000;
    const tag = if (work_u4 == 0) "phase" else "work";
    const Run = struct { g: u32, s: u32, grf: u32, mech: u32 };
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(std.heap.page_allocator);
    for (shapes) |sh| {
        for ([_]u32{ 1, 5, 2, 3 }) |mech| {
            if (mech == 3 and sh.g > 64) continue;
            try runs.append(std.heap.page_allocator, .{ .g = sh.g, .s = sh.s, .grf = sh.grf, .mech = mech });
        }
    }
    for (s1_counts) |g| {
        for ([_]u32{ 4, 1, 5 }) |mech| try runs.append(std.heap.page_allocator, .{ .g = g, .s = 1, .grf = 256, .mech = mech });
    }
    for (runs.items) |run| {
        b.cfg_fault = false;
const Body = struct { name: []const u8, ref: [:0]const u8, img: usize, local: u32, grid: [3]u32, args: []const Arg, runner: []const u8 = "" };
        defer k.deinit();
        const max = xpu.launch.suggestCooperative(k) catch 0;
        if (max < run.g) {
            std.debug.print("BENCH {s} G {d} S {d} grf {d} mech {s}: refused (cooperative count {d})\n", .{ tag, run.g, run.s, run.grf, mech_names[run.mech], max });
            continue;
        }
        const data = try b.big.at(0);
        const zero = try timeCoop(b, k, run.g, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 0), spin_cap, run.mech, work_u4 });
        const full = try timeCoop(b, k, run.g, .{ b.sync, b.stamp, b.fault, data, b.sink, rounds, spin_cap, run.mech, work_u4 });
        var line_extra: [96]u8 = undefined;
        var extra: []const u8 = "";
        if (work_u4 != 0) {
            const only = try timeCoop(b, k, run.g, .{ b.sync, b.stamp, b.fault, data, b.sink, rounds, spin_cap, @as(u32, 0), work_u4 });
            extra = try std.fmt.bufPrint(&line_extra, ", read alone {d:.3} us a round, change over the read {d:.3} us", .{
                usOf(only.ns - zero.ns) / rounds, usOf(full.ns - only.ns) / rounds,
            });
    .{ .name = "fuse_pre_q", .runner = "fuse_pre_q", .ref = "tf_ink2_fuse_pre_q", .img = 0, .local = 256, .grid = .{ 3, 1, 1 }, .args = &.{
        std.debug.print("BENCH {s} G {d} S {d} grf {d} mech {s}: {d:.3} us a round, faults {d} of {d}, check errors {d}, last fault round {d}{s}\n", .{
            tag, run.g, run.s, run.grf, mech_names[run.mech], usOf(full.ns - zero.ns) / rounds, full.faults, reps, full.checks, full.round, extra,
        });
    .{ .name = "pre_zeta", .runner = "pre_zeta", .ref = "tf_ink2_pre_zeta", .img = 0, .local = 256, .grid = .{ 16, 3, 1 }, .args = &.{
}

/// B1: us a native barrier by group size and mode, on a full grid (the cooperative count).
    .{ .name = "fuse_gates", .runner = "fuse_gates", .ref = "tf_ink2_fuse_gates", .img = 0, .local = 64, .grid = .{ 36, 3, 1 }, .args = &.{
    const rounds: u32 = 20000;
    const Size = struct { local: u32, grf: u32 };
    for ([_]Size{ .{ .local = 128, .grf = 256 }, .{ .local = 256, .grf = 256 }, .{ .local = 512, .grf = 256 }, .{ .local = 512, .grf = 128 }, .{ .local = 1024, .grf = 128 } }) |sz| {
    .{ .name = "fuse_norms", .runner = "fuse_norms", .ref = "tf_ink2_fuse_norms", .img = 0, .local = 256, .grid = .{ 3, 1, 1 }, .args = &.{
            std.debug.print("BENCH wgbar local {d} grf {d}: refused\n", .{ sz.local, sz.grf });
            continue;
    .{ .name = "conv_dynamic", .runner = "conv_dynamic", .ref = "tf_inkling2_conv_dynamic", .img = 1, .local = 256, .grid = .{ 1, 1, 1 }, .args = &.{
        defer k.deinit();
        const g = xpu.launch.suggestCooperative(k) catch 0;
    .{ .name = "merge", .runner = "merge", .ref = "tf_ink2_merge", .img = 2, .local = 128, .grid = .{ 24, 1, 1 }, .args = &.{
            std.debug.print("BENCH wgbar local {d} grf {d}: cooperative count 0\n", .{ sz.local, sz.grf });
            continue;
        }
    .{ .name = "router_norm_gemv", .runner = "router_norm_gemv", .ref = "tf_ink2_router_norm_gemv", .img = 0, .local = 256, .grid = .{ 8, 1, 1 }, .args = &.{
            b.cfg_fault = false;
            const z = try timeCoop(b, k, g, .{ b.sink, @as(u32, 0), mode });
    .{ .name = "router_pick", .runner = "router_pick", .ref = "tf_ink2_router_pick", .img = 0, .local = 128, .grid = .{ 1, 1, 1 }, .args = &.{
            std.debug.print("BENCH wgbar local {d} grf {d} groups {d} mode {d}: {d:.4} us a round\n", .{ sz.local, sz.grf, g, mode, usOf(f.ns - z.ns) / rounds });
        }
    .{ .name = "down_epilogue", .runner = "down_epilogue", .ref = "tf_ink2_down_epilogue", .img = 0, .local = 256, .grid = .{ 9, 1, 1 }, .args = &.{
}

/// B6: the fixed cost of a launch. An empty resident kernel (32 x 512, 64 KB local, 5 arguments), plain and
/// cooperative, back to back and one at a time. Then plain launches of today's dense_q8_1 at 1 to 40 MB.
fn benchLaunch(b: *B) !void {
    var k = try b.m256.kernel("fb_empty", .{ 512, 1, 1 });
    defer k.deinit();
    const sink = try b.sink.at(0);
    for ([_]bool{ false, true }) |coop| {
        const name = if (coop) "cooperative" else "plain";
        var t: [reps]i96 = undefined;
        const n_b2b: u32 = 1000;
        for (0..reps) |r| {
            const t0 = nowNs(b.io);
            for (0..n_b2b) |_| {
                if (coop) try xpu.launch.dispatchCooperative(k, b.s, .{ 32, 1, 1 }, .{ sink, sink, sink, @as(u32, 1), @as(u32, 2) }) else try xpu.launch.dispatch(k, b.s, .{ 32, 1, 1 }, .{ sink, sink, sink, @as(u32, 1), @as(u32, 2) });
            }
            try b.s.synchronize();
            t[r] = nowNs(b.io) - t0;
        }
        const b2b = usOf(median(&t)) / n_b2b;
        var one: [101]i96 = undefined;
        for (&one) |*o| {
            const t0 = nowNs(b.io);
            if (coop) try xpu.launch.dispatchCooperative(k, b.s, .{ 32, 1, 1 }, .{ sink, sink, sink, @as(u32, 1), @as(u32, 2) }) else try xpu.launch.dispatch(k, b.s, .{ 32, 1, 1 }, .{ sink, sink, sink, @as(u32, 1), @as(u32, 2) });
            try b.s.synchronize();
            o.* = nowNs(b.io) - t0;
        }
        std.mem.sort(i96, &one, {}, std.sort.asc(i96));
        std.debug.print("BENCH launch empty {s} 32 x 512: {d:.3} us a launch back to back (1,000), {d:.3} us launch to sync (median of 101)\n", .{ name, b2b, usOf(one[50]) });
    }
    ev2: ze.H,
    var d = try b.m256.kernel("tf_ink2_dense_q8_1", .{ 128, 1, 1 });
    defer d.deinit();
    const mbs = [_]f64{ 1, 4, 10, 20, 40 };
    var xs: [mbs.len]f64 = undefined;
    var ys: [mbs.len]f64 = undefined;
    const x = try b.x.at(0);
        try ctx.d.check(api.zeEventPoolCreate(ctx.gpu.context, &.{ .flags = ze.pool_host_visible | ze.pool_kernel_timestamp, .count = 2 }, 1, @ptrCast(&ctx.gpu.device), &pool), "zeEventPoolCreate");
    const nil: ?*anyopaque = null;
    for (mbs, 0..) |mb, i| {
        var ev2: ze.H = null;
        try ctx.d.check(api.zeEventCreate(pool, &.{ .index = 1 }, &ev2), "zeEventCreate");
        const n: u32 = @intFromFloat(mb * 1e6 / 2448.0);
        return .{ .ev = ev, .ev2 = ev2, .ns_per_tick = 1e9 / @as(f64, @floatFromInt(@max(ctx.gpu.props.timerResolution, 1))), .mask = if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1 };
    }

    /// Review C9: device us from the start of the first of `n` plain launches of `k` to the end of the last, in one
    /// event span (the gaps between launches are in it, as in today's step).
    fn span(t: *Timer, b: *B, k: xpu.Kernel, groups: [3]u32, n: u32) !f64 {
        if (n == 1) return t.run(b, k, groups, false);
        try b.checkTime();
        const api = b.ctx.d.api;
        try b.ctx.d.check(api.zeEventHostReset(t.ev), "zeEventHostReset");
        try b.ctx.d.check(api.zeEventHostReset(t.ev2), "zeEventHostReset");
        const g: ze.GroupCount = .{ .x = groups[0], .y = groups[1], .z = groups[2] };
        for (0..n) |i| {
            const e: ze.H = if (i == 0) t.ev else if (i == n - 1) t.ev2 else null;
            try b.ctx.d.check(api.zeCommandListAppendLaunchKernel(b.s.list, k.handle, &g, e, 0, null), "zeCommandListAppendLaunchKernel");
        }
        try b.s.synchronize();
        var a: ze.KernelTimestampResult = .{};
        var z: ze.KernelTimestampResult = .{};
        try b.ctx.d.check(api.zeEventQueryKernelTimestamp(t.ev, &a), "zeEventQueryKernelTimestamp");
        try b.ctx.d.check(api.zeEventQueryKernelTimestamp(t.ev2, &z), "zeEventQueryKernelTimestamp");
        return @as(f64, @floatFromInt((z.global_end -% a.global_start) & t.mask)) * t.ns_per_tick / 1000.0;
        const nwin: usize = gib / stride;
        const count: u32 = 200;
        var t: [reps]i96 = undefined;
        for (0..reps) |r| {
            const t0 = nowNs(b.io);
            for (0..count) |c| {
                const w = try b.big.at((c % nwin) * stride);
const Body = struct { name: []const u8, ref: [:0]const u8, img: usize, local: u32, grid: [3]u32, args: []const Arg, runner: []const u8 = "" };
            }
            try b.s.synchronize();
            t[r] = nowNs(b.io) - t0;
        }
        const us = usOf(median(&t)) / count;
        const bytes = @as(f64, @floatFromInt(n)) * 2448.0;
        xs[i] = bytes / 1e6;
        ys[i] = us;
        std.debug.print("BENCH launch dense_q8_1 plain {d:.0} MB ({d} columns): {d:.3} us a launch, {d:.1} GB/s\n", .{ mb, n, us, bytes / us / 1e3 });
    }
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (xs, ys) |a, c| {
        sx += a;
    .{ .name = "fuse_pre_q", .runner = "fuse_pre_q", .ref = "tf_ink2_fuse_pre_q", .img = 0, .local = 256, .grid = .{ 3, 1, 1 }, .args = &.{
        sxx += a * a;
        sxy += a * c;
    }
    .{ .name = "pre_zeta", .runner = "pre_zeta", .ref = "tf_ink2_pre_zeta", .img = 0, .local = 256, .grid = .{ 16, 3, 1 }, .args = &.{
    const slope = (nf * sxy - sx * sy) / (nf * sxx - sx * sx);
    const icpt = (sy - slope * sx) / nf;
    std.debug.print("BENCH launch dense_q8_1 fit: fixed {d:.3} us a launch, marginal {d:.1} GB/s\n", .{ icpt, 1e3 / slope });
    .{ .name = "fuse_gates", .runner = "fuse_gates", .ref = "tf_ink2_fuse_gates", .img = 0, .local = 64, .grid = .{ 36, 3, 1 }, .args = &.{

/// B4: GB/s of one Q8 phase inside a resident kernel, by shape and phase size, with a phase change (M-a) after each
/// phase and with none. The slope between 1 and R phases a launch.
    .{ .name = "fuse_norms", .runner = "fuse_norms", .ref = "tf_ink2_fuse_norms", .img = 0, .local = 256, .grid = .{ 3, 1, 1 }, .args = &.{
    const mbs = [_]f64{ 4, 10, 20, 40, 245 };
    const x = try b.x.at(0);
    .{ .name = "conv_dynamic", .runner = "conv_dynamic", .ref = "tf_inkling2_conv_dynamic", .img = 1, .local = 256, .grid = .{ 1, 1, 1 }, .args = &.{
    for (shapes) |sh| {
        var k = try b.module(sh.grf).kernel("fb_q8", .{ sh.s * 16, 1, 1 });
    .{ .name = "merge", .runner = "merge", .ref = "tf_ink2_merge", .img = 2, .local = 128, .grid = .{ 24, 1, 1 }, .args = &.{
        const max = xpu.launch.suggestCooperative(k) catch 0;
        if (max < sh.g) {
            std.debug.print("BENCH q8 G {d} S {d} grf {d}: refused (cooperative count {d})\n", .{ sh.g, sh.s, sh.grf, max });
    .{ .name = "router_norm_gemv", .runner = "router_norm_gemv", .ref = "tf_ink2_router_norm_gemv", .img = 0, .local = 256, .grid = .{ 8, 1, 1 }, .args = &.{
        }
        for (mbs) |mb| {
    .{ .name = "router_pick", .runner = "router_pick", .ref = "tf_ink2_router_pick", .img = 0, .local = 128, .grid = .{ 1, 1, 1 }, .args = &.{
            const stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, n) * 2448, 4096));
            const nwin: u32 = @intCast(gib / stride);
    .{ .name = "down_epilogue", .runner = "down_epilogue", .ref = "tf_ink2_down_epilogue", .img = 0, .local = 256, .grid = .{ 9, 1, 1 }, .args = &.{
            const big_r: u32 = @intFromFloat(@max(4.0, @min(100.0, 2e9 / bytes)));
            for ([_]u32{ 1, 5, 0 }) |mech| {
                b.cfg_fault = false;
                const t1 = try timeCoop(b, k, sh.g, .{ b.big, x, y, b.sync, b.fault, n, @as(u32, 1), nwin, stride, spin_cap, mech });
                const tr = try timeCoop(b, k, sh.g, .{ b.big, x, y, b.sync, b.fault, n, big_r, nwin, stride, spin_cap, mech });
                const us = usOf(tr.ns - t1.ns) / @as(f64, @floatFromInt(big_r - 1));
                std.debug.print("BENCH q8 G {d} S {d} grf {d} {d:.0} MB wait {s}: {d:.3} us a phase, {d:.1} GB/s, faults {d}\n", .{
                    sh.g, sh.s, sh.grf, mb, mech_names[mech], us, bytes / us / 1e3, tr.faults,
                });
            }
        }
    }
}

/// B5: 40 MB as 4 to 100 dependent phases, mech none, a and c, prefetch 0 to 2.4 MB a phase change. The time of one
/// pass is the slope between 1 and 11 passes; the cost of a phase change is the slope of a pass over `phases`.
fn benchStream(b: *B) !void {
    const phase_counts = [_]u32{ 4, 10, 40, 100 };
    const pf_mbs = [_]f64{ 0, 0.6, 1.2, 2.4 };
    const y = try b.y.at(0);
    for ([_]Shape{ .{ .g = 32, .s = 32, .grf = 256 }, .{ .g = 64, .s = 16, .grf = 256 } }) |sh| {
        var k = try b.module(sh.grf).kernel("fb_stream", .{ sh.s * 16, 1, 1 });
        defer k.deinit();
        const cmax = xpu.launch.suggestCooperative(k) catch 0;
        if (cmax < sh.g) {
            std.debug.print("BENCH stream G {d} S {d}: refused (cooperative count {d})\n", .{ sh.g, sh.s, cmax });
            continue;
        }
        const t_sg = sh.g * sh.s;
        for ([_]u32{ 0, 1, 5, 3 }) |mech| {
          for (pf_mbs) |pf| {
        const rlen = (try std.fmt.bufPrint(name_buf[0..63], "fbr_{s}", .{if (body.runner.len > 0) body.runner else body.name})).len;
            if (mech == 3 and sh.g > 64) continue;
            const pf_lines: u32 = @intFromFloat(@round(pf * 1e6 / @as(f64, @floatFromInt(t_sg)) / 64.0));
            var px: [phase_counts.len]f64 = undefined;
            var py: [phase_counts.len]f64 = undefined;
            for (phase_counts, 0..) |ph, i| {
                b.cfg_fault = false;
                const n: u32 = @intFromFloat(40e6 / 2448.0 / @as(f64, @floatFromInt(ph)));
                const stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, n) * 2448, 4096));
                const t1 = try timeCoop(b, k, sh.g, .{ b.big, b.x, y, b.sync, b.fault, n, ph, @as(u32, 1), stride, spin_cap, mech, pf_lines });
                const t11 = try timeCoop(b, k, sh.g, .{ b.big, b.x, y, b.sync, b.fault, n, ph, @as(u32, 11), stride, spin_cap, mech, pf_lines });
                const us = usOf(t11.ns - t1.ns) / 10.0;
                const bytes = @as(f64, @floatFromInt(n)) * 2448.0 * @as(f64, @floatFromInt(ph));
                px[i] = @floatFromInt(ph);
                py[i] = us;
                std.debug.print("BENCH stream G {d} S {d} mech {s} prefetch {d:.1} MB ({d} lines a sub-group) phases {d}: {d:.3} us a pass of {d:.2} MB, {d:.1} GB/s, faults {d}\n", .{
                    sh.g, sh.s, mech_names[mech], pf, pf_lines, ph, us, bytes / 1e6, bytes / us / 1e3, t11.faults,
                });
            }
            var sx: f64 = 0;
            var sy: f64 = 0;
            var sxx: f64 = 0;
            var sxy: f64 = 0;
            for (px, py) |a, c| {
                sx += a;
                sy += c;
                sxx += a * a;
                sxy += a * c;
            }
            const nf: f64 = @floatFromInt(phase_counts.len);
            const slope = (nf * sxy - sx * sy) / (nf * sxx - sx * sx);
            std.debug.print("BENCH stream G {d} S {d} mech {s} prefetch {d:.1} MB fit: {d:.3} us a phase change, {d:.3} us for 40 MB with no change\n", .{
                sh.g, sh.s, mech_names[mech], pf, slope, (sy - slope * sx) / nf,
            });
          }
        }
    }
}

/// B8: us to read a 2.4 MB block after a stream of 4 to 36 MB, prefetched into L2 before the stream, and cold.
fn benchL2keep(b: *B) !void {
    var k = try b.m256.kernel("fb_l2keep", .{ 512, 1, 1 });
    defer k.deinit();
    const cmax = xpu.launch.suggestCooperative(k) catch 0;
    if (cmax < 32) {
        std.debug.print("BENCH l2keep G 32 S 32: refused (cooperative count {d})\n", .{cmax});
        return;
    }
    const rounds: u32 = 100;
    for ([_]u32{ 4, 8, 16, 24, 36 }) |mb| {
        const sb: u32 = mb * 1_000_000;
        var per: [2][2]f64 = undefined;
        for ([_]u32{ 0, 1 }) |pf| {
                    if (f[0] != 0) {
                        if (faults == 0) try b.sayFault(body.name, 32);
                        faults += 1;
                    }
            b.cfg_fault = false;
            const t1 = try timeCoop(b, k, 32, .{ b.big, b.sink, b.sync, b.fault, @as(u32, 1), pf, rd, sb, spin_cap });
            const tr = try timeCoop(b, k, 32, .{ b.big, b.sink, b.sync, b.fault, rounds, pf, rd, sb, spin_cap });
            per[pf][rd] = usOf(tr.ns - t1.ns) / @as(f64, @floatFromInt(rounds - 1));
          }
        }
        std.debug.print("BENCH l2keep stream {d} MB: read of 2.4 MB after the stream {d:.3} us prefetched, {d:.3} us cold; round {d:.3} us (prefetch, read), {d:.3} (prefetch, no read), {d:.3} (cold, read), {d:.3} (cold, no read)\n", .{
            mb, per[1][1] - per[1][0], per[0][1] - per[0][0], per[1][1], per[1][0], per[0][1], per[0][0],
        });
    }
}

// ---------------------------------------------------------------- B10: small phases in a resident kernel
const H: usize = 2304;
const Gen = enum { bf16, f32, f32pos, ml, ids, pos40k, zero };
const Arg = union(enum) { buf: struct { bytes: usize, gen: Gen }, u: u32 };
const Body = struct { name: []const u8, ref: [:0]const u8, img: usize, local: u32, grid: [3]u32, args: []const Arg };

fn bf(n: usize) Arg {
    return .{ .buf = .{ .bytes = n * 2, .gen = .bf16 } };
}
fn f32b(n: usize) Arg {
    return .{ .buf = .{ .bytes = n * 4, .gen = .f32 } };
}

fn uploadAll(b: *B, d: []xpu.DeviceBuffer, host: []const []u8) !void {
    for (d, host) |*x, h| try x.upload(b.s, 0, h);
    try b.s.synchronize();
}

fn diffBytes(b: *B, d: []xpu.DeviceBuffer, ref: []const []u8, tmp: [][]u8) !usize {
    var n: usize = 0;
    for (d, ref, tmp) |*x, r, t| {
        try x.download(b.s, 0, t);
        for (r, t) |p, q| n += @intFromBool(p != q);
    }
    return n;
}

fn slope3(ys: [3]f64) f64 {
    const xs = [_]f64{ 1, 10, 100 };
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (xs, ys) |x, y| {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (3.0 * sxy - sx * sy) / (3.0 * sxx - sx * sx);
}

/// B10 by the corrected method (review C9, C11, C10, C4). `form` is the resident image: "fix" (fused_small, each task
/// one time, spare slots idle, equal barrier counts) or "old" (fused_small_old, the Part 1 layout). Every timed launch
/// and every timed span starts from the uploaded inputs. Today's kernel: one launch (median of 5) and the slope over
/// spans of 1, 10 and 100 launches. Resident: one launch of 1 repeat and the slope over 1, 10 and 100 repeats, with no
/// wait and with M-a. The old method of Part 1 (no upload between launches) runs too, for its numbers beside the new
/// ones. Bytes: 1 repeat with no wait and with M-a against one launch of today's kernel, and 10 repeats with M-a against
/// 10 launches. Then the tm0 stamps of 10 repeats (C10).
fn benchSmall2(b: *B, imgs: [5]xpu.Module, form: []const u8) !void {
    const alloc = std.heap.page_allocator;
    const old = std.mem.eql(u8, form, "old");
    const rimg = if (old) imgs[4] else imgs[3];
    var timer = try Timer.init(b.ctx);
    var prng = std.Random.DefaultPrng.init(0x5eed_f00d);
    const rng = prng.random();
    const all = bodies ++ [_]Body{ merge_sliding_body, q1_body };
    for (all, 0..) |body, bi| {
        const is_q1 = bi == bodies.len + 1;
        var nbuf: usize = 0;
        for (body.args) |a| {
            if (a == .buf) nbuf += 1;
        }
        const host = try alloc.alloc([]u8, nbuf);
        const outa = try alloc.alloc([]u8, nbuf);
        const outa10 = try alloc.alloc([]u8, nbuf);
        const tmp = try alloc.alloc([]u8, nbuf);
        const da = try alloc.alloc(xpu.DeviceBuffer, nbuf);
        const db = try alloc.alloc(xpu.DeviceBuffer, nbuf);
        {
            var i: usize = 0;
            for (body.args) |a| {
                if (a != .buf) continue;
                host[i] = try alloc.alloc(u8, a.buf.bytes);
                outa[i] = try alloc.alloc(u8, a.buf.bytes);
                outa10[i] = try alloc.alloc(u8, a.buf.bytes);
                tmp[i] = try alloc.alloc(u8, a.buf.bytes);
                fillGen(rng, a.buf.gen, host[i]);
                da[i] = try xpu.DeviceBuffer.alloc(b.ctx, a.buf.bytes);
                db[i] = try xpu.DeviceBuffer.alloc(b.ctx, a.buf.bytes);
                i += 1;
            }
        }
        defer for (0..nbuf) |i| {
            da[i].free();
            db[i].free();
        };
        if (is_q1 and old) {
            std.debug.print("BENCH small2 {s} form old: not run (a slot that leaves at its range check passes fewer barriers than its neighbour; a barrier wait has no time cap)\n", .{body.name});
            continue;
        }
        // today's kernel
        var rk = try imgs[body.img].kernel(body.ref, .{ body.local, 1, 1 });
        defer rk.deinit();
        try setArgs(rk, body.args, da);
        try uploadAll(b, da, host);
        var t_old: [5]f64 = undefined;
        t_old[0] = try timer.run(b, rk, body.grid, false);
        for (0..nbuf) |i| try da[i].download(b.s, 0, outa[i]);
        for (1..5) |r| t_old[r] = try timer.run(b, rk, body.grid, false);
        var t_one: [5]f64 = undefined;
        for (&t_one) |*t| {
            try uploadAll(b, da, host);
            t.* = try timer.run(b, rk, body.grid, false);
        }
        var ys_today: [3]f64 = undefined;
        for ([_]u32{ 1, 10, 100 }, 0..) |c, ci| {
            var tt: [5]f64 = undefined;
            for (&tt) |*t| {
                try uploadAll(b, da, host);
                t.* = try timer.span(b, rk, body.grid, c);
            }
            ys_today[ci] = medianF(&tt);
        }
        try uploadAll(b, da, host);
        _ = try timer.span(b, rk, body.grid, 10);
        for (0..nbuf) |i| try da[i].download(b.s, 0, outa10[i]);
        // the resident kernel
        var name_buf: [64]u8 = undefined;
        const rlen = (try std.fmt.bufPrint(name_buf[0..63], "fbr_{s}", .{body.runner})).len;
        name_buf[rlen] = 0;
        var k = try rimg.kernel(name_buf[0..rlen :0], .{ 512, 1, 1 });
        defer k.deinit();
        const max = xpu.launch.suggestCooperative(k) catch 0;
        if (max < 32) {
            std.debug.print("BENCH small2 {s}: refused (cooperative count {d} at 512)\n", .{ body.name, max });
            continue;
        }
        const na: u32 = @intCast(body.args.len);
        try setArgs(k, body.args, db);
        try k.setPtr(na, b.sync.ptr);
        try k.setPtr(na + 1, b.fault.ptr);
        try k.setU32(na + 4, spin_cap);
        var faults: u32 = 0;
        const Run = struct {
            fn one(bb: *B, tm: *Timer, kk: xpu.Kernel, n: u32, reps_: u32, mech: u32, f: *u32, name: []const u8) !f64 {
                try bb.reset();
                try kk.setU32(n + 2, reps_);
                try kk.setU32(n + 3, mech);
                const us = try tm.run(bb, kk, .{ 32, 1, 1 }, true);
                const w = try bb.faultWords();
                if (w[0] != 0) {
                    if (f.* == 0) try bb.sayFault(name, 32);
                    f.* += 1;
                }
                return us;
            }
        };
        // bytes
        var eq: [3]usize = undefined;
        try uploadAll(b, db, host);
        _ = try Run.one(b, &timer, k, na, 1, 0, &faults, body.name);
        eq[0] = try diffBytes(b, db, outa, tmp);
        try uploadAll(b, db, host);
        _ = try Run.one(b, &timer, k, na, 1, 1, &faults, body.name);
        eq[1] = try diffBytes(b, db, outa, tmp);
        try uploadAll(b, db, host);
        _ = try Run.one(b, &timer, k, na, 10, 1, &faults, body.name);
        eq[2] = try diffBytes(b, db, outa10, tmp);
        // times, corrected method: upload before each timed launch
        var one_r: [2]f64 = undefined;
        var slope_r: [2]f64 = undefined;
        var slope_o: [2]f64 = undefined;
        for ([_]u32{ 0, 1 }) |mech| {
            var ys: [3]f64 = undefined;
            for ([_]u32{ 1, 10, 100 }, 0..) |c, ci| {
                var tt: [5]f64 = .{ 0, 0, 0, 0, 0 };
                for (&tt) |*t| {
                    if (faults != 0) break;
                    try uploadAll(b, db, host);
                    t.* = try Run.one(b, &timer, k, na, c, mech, &faults, body.name);
                }
                ys[ci] = medianF(&tt);
            }
            one_r[mech] = ys[0];
            slope_r[mech] = slope3(ys);
            // the old method: the inputs as the launch before left them
            try uploadAll(b, db, host);
            for ([_]u32{ 1, 10, 100 }, 0..) |c, ci| {
                var tt: [5]f64 = .{ 0, 0, 0, 0, 0 };
                for (&tt) |*t| {
                    if (faults != 0) break;
                    t.* = try Run.one(b, &timer, k, na, c, mech, &faults, body.name);
                }
                ys[ci] = medianF(&tt);
            }
            slope_o[mech] = slope3(ys);
        }
        const thr_today = body.grid[0] * body.grid[1] * body.grid[2] * body.local;
        std.debug.print("BENCH small2 {s} form {s}: today {s} ({d}x{d}x{d} of {d}, {d} work-items): one launch {d:.3} us (old method {d:.3}), slope over 1, 10, 100 launches {d:.3} us ({d:.3}, {d:.3}, {d:.3}); resident 32 x 512 (16384 work-items): one launch of 1 repeat {d:.3} us no wait, {d:.3} us M-a; slope a repeat {d:.3} us no wait, {d:.3} us M-a (old method {d:.3}, {d:.3}); bytes differ: 1 repeat no wait {d}, 1 repeat M-a {d}, 10 repeats M-a against 10 launches {d}; faults {d}\n", .{
            body.name, form,         body.ref,   body.grid[0], body.grid[1], body.grid[2], body.local, thr_today,
            medianF(&t_one), medianF(&t_old), slope3(ys_today), ys_today[0], ys_today[1], ys_today[2],
            one_r[0],  one_r[1],     slope_r[0], slope_r[1],   slope_o[0],   slope_o[1],   eq[0],      eq[1],
            eq[2],     faults,
        });
        // C10: tm0 stamps of 10 repeats, every group
        // C10: the stamps with no wait, with M-a, and with M-a less its acquire fence (FB_NO_ACQ)
        if (faults == 0) for ([_]u32{ 0, 1, 1 | 0x20000 }) |mech| {
            try uploadAll(b, db, host);
            _ = try Run.one(b, &timer, k, na, 10, mech | 0x400, &faults, body.name);
            var sw: [32 * 10 * 3 * 8]u8 = undefined;
            try b.sync.download(b.s, 20000 * 4, &sw);
            var body0: [9]f64 = undefined;
            var wait0: [9]f64 = undefined;
            var bmax: [9]f64 = undefined;
            var wmin: [9]f64 = undefined;
            var bmax_g: u32 = 0;
            for (1..10) |r| {
                bmax[r - 1] = 0;
                wmin[r - 1] = 1e30;
                for (0..32) |g| {
                    var st: [3]u64 = undefined;
                    for (0..3) |q| st[q] = std.mem.readInt(u64, sw[((g * 10 + r) * 3 + q) * 8 ..][0..8], .little);
                    const bt: f64 = @floatFromInt(st[1] -% st[0]);
                    const wt: f64 = @floatFromInt(st[2] -% st[1]);
                    if (g == 0) {
                        body0[r - 1] = bt;
                        wait0[r - 1] = wt;
                    }
                    if (bt > bmax[r - 1]) {
                        bmax[r - 1] = bt;
                        bmax_g = @intCast(g);
                    }
                    wmin[r - 1] = @min(wmin[r - 1], wt);
                }
            }
            std.debug.print("BENCH small2 stamps {s} form {s} wait {s}: tm0 ticks a repeat (median of repeats 2 to 10): group 0 tasks {d:.0}, group 0 after the tasks to the wait end {d:.0}; largest tasks time of any group {d:.0} (group {d} in the last repeat), shortest time after the tasks {d:.0}\n", .{
                body.name, form, if (mech == 0) "none" else if (mech == 1) "a" else "a, no acquire fence", medianF(&body0), medianF(&wait0), medianF(&bmax), bmax_g, medianF(&wmin),
            });
        };
    }
}

fn zb(bytes: usize) Arg {
    return .{ .buf = .{ .bytes = bytes, .gen = .zero } };
}
fn uv(v: u32) Arg {
    return .{ .u = v };
}

/// Today's grid and arguments at one decode row (rows 1), the shapes of DESIGN section 1.
const bodies = [_]Body{
    .{ .name = "fuse_pre_q", .ref = "tf_ink2_fuse_pre_q", .img = 0, .local = 256, .grid = .{ 3, 1, 1 }, .args = &.{
        bf(H), zb(H * 4), bf(4 * H), f32b(3 * H), zb(3 * H * 4), bf(H), bf(H), bf(H), zb(H * 2), zb(H * 2), zb(H * 2),
        uv(1), uv(4 * H), uv(5), uv(3), uv(3),
    } },
    .{ .name = "pre_zeta", .ref = "tf_ink2_pre_zeta", .img = 0, .local = 256, .grid = .{ 16, 3, 1 }, .args = &.{
        bf(H), zb(H * 4), bf(4 * H), f32b(3 * H), zb(3 * H * 4), f32b(12 * H), bf(64 * 2 * H), bf(64 * 2 * H), bf(64 * 2 * H),
        zb(4 * 64 * 4), uv(1), uv(4 * H), uv(5), uv(3), uv(0), uv(1), uv(12),
    } },
        const rlen = (try std.fmt.bufPrint(name_buf[0..63], "fbr_{s}", .{if (body.runner.len > 0) body.runner else body.name})).len;
        f32b(3 * H), f32b(12 * H), bf(H * 64), bf(H * 64), bf(H * 64), bf(H * 64), bf(H * 64), bf(H * 64), f32b(4 * 64),
        zb(3 * H * 4), uv(1), uv(3), uv(0), uv(1), uv(12),
    } },
    .{ .name = "fuse_norms", .ref = "tf_ink2_fuse_norms", .img = 0, .local = 256, .grid = .{ 3, 1, 1 }, .args = &.{
        f32b(3 * H), bf(H), bf(H), bf(H), zb(H * 2), zb(H * 2), zb(H * 2), uv(1), uv(3),
    } },
    .{ .name = "conv_dynamic", .ref = "tf_inkling2_conv_dynamic", .img = 1, .local = 256, .grid = .{ 1, 1, 1 }, .args = &.{
        bf(3 * 1024), bf(1024), zb(4 * 1024 * 2), bf(H), bf(4 * 1024), bf(32 * H), zb(1024 * 2), uv(1), uv(1024),
    } },
    .{ .name = "merge", .ref = "tf_ink2_merge", .img = 2, .local = 128, .grid = .{ 24, 1, 1 }, .args = &.{
        f32b(24 * 128 * 128), .{ .buf = .{ .bytes = 24 * 128 * 8, .gen = .ml } }, bf(24), bf(24 * 128),
        .{ .buf = .{ .bytes = 4, .gen = .pos40k } }, zb(24 * 128 * 2), uv(24), uv(0), uv(313), uv(0), uv(128), uv(1),
    } },
    .{ .name = "router_norm_gemv", .ref = "tf_ink2_router_norm_gemv", .img = 0, .local = 256, .grid = .{ 8, 1, 1 }, .args = &.{
        f32b(H), bf(H), bf(128 * H), zb(H * 2), zb(128 * 4), uv(1),
    } },
    .{ .name = "router_pick", .ref = "tf_ink2_router_pick", .img = 0, .local = 128, .grid = .{ 1, 1, 1 }, .args = &.{
        .{ .buf = .{ .bytes = 128 * 4, .gen = .f32pos } }, f32b(128), zb(8 * 4), zb(8 * 4), uv(1),
    } },
    .{ .name = "down_epilogue", .ref = "tf_ink2_down_epilogue", .img = 0, .local = 256, .grid = .{ 9, 1, 1 }, .args = &.{
        bf(8 * H), .{ .buf = .{ .bytes = 8 * 4, .gen = .ids } }, .{ .buf = .{ .bytes = 8 * 4, .gen = .f32pos } }, bf(3 * H),
        zb(4 * H * 2), bf(4 * H), zb(H * 2), zb(H * 2), f32b(H), zb(H * 2), uv(@intCast(H)), uv(8), uv(@intCast(H)), uv(1), uv(1),
    } },
};

fn fillGen(rng: std.Random, gen: Gen, out: []u8) void {
    switch (gen) {
        .zero => @memset(out, 0),
        .bf16 => for (0..out.len / 2) |i| {
            const v: f32 = rng.float(f32) * 2.0 - 1.0;
            std.mem.writeInt(u16, out[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
        },
        .f32, .f32pos => for (0..out.len / 4) |i| {
            const v: f32 = if (gen == .f32) rng.float(f32) * 2.0 - 1.0 else 0.01 + rng.float(f32) * 0.99;
            std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
        },
        .ml => for (0..out.len / 8) |i| {
            const m: f32 = rng.float(f32) * 8.0 - 4.0;
            const l: f32 = 1.0 + rng.float(f32) * 99.0;
            std.mem.writeInt(u32, out[i * 8 ..][0..4], @bitCast(m), .little);
            std.mem.writeInt(u32, out[i * 8 + 4 ..][0..4], @bitCast(l), .little);
        },
        .ids => for (0..out.len / 4) |i| std.mem.writeInt(u32, out[i * 4 ..][0..4], @intCast(i * 13 + 5), .little),
        .pos40k => std.mem.writeInt(u32, out[0..4], 40000, .little),
    }
}

const CoopFn = *const fn (ze.H, ze.H, *const ze.GroupCount, ze.H, u32, ?[*]ze.H) callconv(.c) u32;

const Timer = struct {
    ev: ze.H,
    ns_per_tick: f64,
                    if (f[0] != 0) {
                        if (faults == 0) try b.sayFault(body.name, 32);
                        faults += 1;
                    }

    fn init(ctx: *xpu.Context) !Timer {
        var pool: ze.H = null;
        const api = ctx.d.api;
        try ctx.d.check(api.zeEventPoolCreate(ctx.gpu.context, &.{ .flags = ze.pool_host_visible | ze.pool_kernel_timestamp, .count = 1 }, 1, @ptrCast(&ctx.gpu.device), &pool), "zeEventPoolCreate");
        var ev: ze.H = null;
        try ctx.d.check(api.zeEventCreate(pool, &.{ .index = 0 }, &ev), "zeEventCreate");
        const bits = ctx.gpu.props.kernelTimestampValidBits;
        return .{ .ev = ev, .ns_per_tick = 1e9 / @as(f64, @floatFromInt(@max(ctx.gpu.props.timerResolution, 1))), .mask = if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1 };
    }
/// before any other spin run. Then the other shapes, each mechanism once. Review C1: each launch prints its host time,
/// its device time (kernel timestamp event), the fault record with the tm0 ticks of the first capped group, and the
/// tick rate in the spin (those ticks over the device time: rounds 1 and 2 take microseconds).
    /// Device us of one launch of `k` (arguments already set), plain or cooperative.
    fn run(t: *Timer, b: *B, k: xpu.Kernel, groups: [3]u32, coop: bool) !f64 {
        try b.checkTime();
        const api = b.ctx.d.api;
        try b.ctx.d.check(api.zeEventHostReset(t.ev), "zeEventHostReset");
        const g: ze.GroupCount = .{ .x = groups[0], .y = groups[1], .z = groups[2] };
        if (coop) {
            const f = @constCast(b.ctx.d).lib.lookup(CoopFn, "zeCommandListAppendLaunchCooperativeKernel") orelse return error.MissingSymbol;
            try b.ctx.d.check(f(b.s.list, k.handle, &g, t.ev, 0, null), "zeCommandListAppendLaunchCooperativeKernel");
        } else try b.ctx.d.check(api.zeCommandListAppendLaunchKernel(b.s.list, k.handle, &g, t.ev, 0, null), "zeCommandListAppendLaunchKernel");
        try b.s.synchronize();
    var timer = try Timer.init(b.ctx);
        var ts: ze.KernelTimestampResult = .{};
    var worst: f64 = 0;
        try b.ctx.d.check(api.zeEventQueryKernelTimestamp(t.ev, &ts), "zeEventQueryKernelTimestamp");

fn uploadAll(b: *B, d: []xpu.DeviceBuffer, host: []const []u8) !void {
    for (d, host) |*x, h| try x.upload(b.s, 0, h);
    try b.s.synchronize();
}

fn diffBytes(b: *B, d: []xpu.DeviceBuffer, ref: []const []u8, tmp: [][]u8) !usize {
    var n: usize = 0;
    for (d, ref, tmp) |*x, r, t| {
        try x.download(b.s, 0, t);
            try setAll(k, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech | 0x100, @as(u32, 0) });
            const mhz0 = readMhz(b.io, "act_freq");
        for (r, t) |p, q| n += @intFromBool(p != q);
            const dev_us = try timer.run(b, k, .{ run.g, 1, 1 }, true);
}
            const mhz1 = readMhz(b.io, "act_freq");
            // The launch time at rpn: the device time at the clock of the launch, scaled to the lowest clock.
            const mhz = if (@max(mhz0, mhz1) == 0) card_freq.rp0 else @max(mhz0, mhz1);
            const at_rpn = dev_us / 1000.0 * @as(f64, @floatFromInt(mhz)) / @as(f64, @floatFromInt(card_freq.rpn));
            worst = @max(worst, at_rpn);

            var fw: [36]u8 = undefined;
            try b.fault.download(b.s, 0, &fw);
            const ticks = (@as(u64, std.mem.readInt(u32, fw[32..36], .little)) << 32) | std.mem.readInt(u32, fw[28..32], .little);
fn slope3(ys: [3]f64) f64 {
    const xs = [_]f64{ 1, 10, 100 };
    var sx: f64 = 0;
            std.debug.print("BENCH force G {d} S {d} mech {s}: launch {d:.3} ms host, {d:.3} ms device at act_freq {d} to {d} MHz, {d:.3} ms at rpn, fault word {d}, fault round {d}, counter {d}, generation {d}, tick rate in the spin {d:.1} ticks a us\n", .{
                run.g,                                  run.s,                                  mech_names[run.mech], ms, dev_us / 1000.0, mhz0, mhz1, at_rpn, f[0], f[2],
                std.mem.readInt(u32, cw[0..4], .little), std.mem.readInt(u32, cw[4..8], .little), @as(f64, @floatFromInt(ticks)) / dev_us,
    var sxy: f64 = 0;
            try b.sayFault("force", run.g);
            if (f[0] == 0 or ms > 100.0 or at_rpn > 100.0) {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (3.0 * sxy - sx * sy) / (3.0 * sxx - sx * sx);
    std.debug.print("BENCH force passed: longest forced launch {d:.3} ms at rpn\n", .{worst});
}

/// Review C1: the tm0 rate in three states, each 3 times, with the card clock (sysfs act_freq) read before and after.
/// (1) fb_tsc alone, (2) fb_tsc queued right after a 40 MB read over the full grid, (3) fb_tsc_grid, the rate inside a
/// full grid that reads 1 GiB 4 times.
fn benchTsc(b: *B) !void {
    var timer = try Timer.init(b.ctx);
    var kt = try b.m256.kernel("fb_tsc", .{ 16, 1, 1 });
    defer kt.deinit();
    var kg = try b.m256.kernel("fb_tsc_grid", .{ 512, 1, 1 });
    defer kg.deinit();
    const groups = xpu.launch.suggestCooperative(kg) catch 0;
    if (groups == 0) return error.AboveCooperativeCount;
    const data = try b.big.at(0);
    const sink = try b.sink.at(0);
    const ticks: u32 = 1 << 20;
    for ([_][]const u8{ "alone", "after a 40 MB read", "beside a full grid" }, 0..) |name, mode| {
        for (0..3) |_| {
            try b.checkTime();
            const f0 = readMhz(b.io, "act_freq");
            var us: f64 = 0;
            if (mode == 2) {
                const per: u32 = @intCast(gib / 16 / groups); // uint4 a group
                try setAll(kg, .{ data, sink, @as(u32, 4), per });
                us = try timer.run(b, kg, .{ groups, 1, 1 }, false);
            } else {
                if (mode == 1) {
                    const per: u32 = @intCast(40_000_000 / 16 / groups);
                    try xpu.launch.dispatch(kg, b.s, .{ groups, 1, 1 }, .{ data, sink, @as(u32, 1), per });
                }
                try setAll(kt, .{ sink, ticks });
                us = try timer.run(b, kt, .{ 1, 1, 1 }, false);
            }
            const f1 = readMhz(b.io, "act_freq");
            var w: [8]u8 = undefined;
            try b.sink.download(b.s, 0, &w);
            const delta = std.mem.readInt(u64, w[0..8], .little);
            std.debug.print("BENCH tsc {s}: {d} ticks in {d:.3} us device time, {d:.1} ticks a us, act_freq {d} MHz before, {d} MHz after\n", .{ name, delta, us, @as(f64, @floatFromInt(delta)) / us, f0, f1 });
        }
    }
}

/// Review C2: each shape and mechanism, `n` launches of B2 (2,000 changes, no work), B3 (2,000 changes over a 64 KB
/// read) and B4 (100 Q8 phases of 4 MB). A configuration stops at its first fault (rule 2) with the fault record.
fn benchSoak(b: *B) !void {
    const n_launch: u32 = 50;
    const Run = struct { g: u32, s: u32, grf: u32, mech: u32 };
    const runs = [_]Run{
        .{ .g = 32, .s = 16, .grf = 256, .mech = 1 },  .{ .g = 32, .s = 32, .grf = 256, .mech = 1 }, .{ .g = 16, .s = 32, .grf = 256, .mech = 1 },
        .{ .g = 64, .s = 16, .grf = 256, .mech = 1 },  .{ .g = 128, .s = 8, .grf = 256, .mech = 1 }, .{ .g = 32, .s = 32, .grf = 128, .mech = 1 },
        .{ .g = 32, .s = 16, .grf = 256, .mech = 2 },  .{ .g = 32, .s = 32, .grf = 256, .mech = 2 }, .{ .g = 32, .s = 16, .grf = 256, .mech = 3 },
        .{ .g = 32, .s = 1, .grf = 256, .mech = 4 },   .{ .g = 256, .s = 1, .grf = 256, .mech = 4 }, .{ .g = 512, .s = 1, .grf = 256, .mech = 4 },
        .{ .g = 512, .s = 1, .grf = 256, .mech = 1 },  .{ .g = 1024, .s = 1, .grf = 256, .mech = 1 }, .{ .g = 1024, .s = 1, .grf = 256, .mech = 2 },
        .{ .g = 1024, .s = 1, .grf = 256, .mech = 4 },
    };
    const data = try b.big.at(0);
    const x = try b.x.at(0);
    const y = try b.y.at(0);
    const q8_n: u32 = @intFromFloat(4e6 / 2448.0);
    const q8_stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, q8_n) * 2448, 4096));
    const q8_nwin: u32 = @intCast(gib / q8_stride);
    for ([_][]const u8{ "B2", "B3", "B4" }, 0..) |bench, bi| {
        for (runs) |run| {
            var k = try b.module(run.grf).kernel(if (bi == 2) "fb_q8" else "fb_phase", .{ run.s * 16, 1, 1 });
            defer k.deinit();
            const max = xpu.launch.suggestCooperative(k) catch 0;
            if (max < run.g) {
                std.debug.print("BENCH soak {s} G {d} S {d} grf {d} mech {s}: refused (cooperative count {d})\n", .{ bench, run.g, run.s, run.grf, mech_names[run.mech], max });
                continue;
            }
            b.lag = bi != 2;
            var faults: u32 = 0;
            var done: u32 = 0;
            var worst: i96 = 0;
            for (0..n_launch) |_| {
                try b.checkTime();
                try b.reset();
                const t0 = nowNs(b.io);
                if (bi == 2) {
                    try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.big, x, y, b.sync, b.fault, q8_n, @as(u32, 100), q8_nwin, q8_stride, spin_cap, run.mech });
                } else {
                    try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech, @as(u32, if (bi == 1) 4096 else 0) });
                }
                try b.s.synchronize();
                worst = @max(worst, nowNs(b.io) - t0);
                done += 1;
                const f = try b.faultWords();
                if (f[0] != 0) {
                    faults += 1;
                    var tagb: [48]u8 = undefined;
                    try b.sayFault(try std.fmt.bufPrint(&tagb, "soak {s} launch {d}", .{ bench, done }), run.g);
                    break;
                }
            }
            b.lag = false;
            std.debug.print("BENCH soak {s} G {d} S {d} grf {d} mech {s}: faults {d} in {d} launches, longest launch {d:.3} ms\n", .{ bench, run.g, run.s, run.grf, mech_names[run.mech], faults, done, usOf(worst) / 1000.0 });
        }
    }
}

/// B10 by the corrected method (review C9, C11, C10, C4). `form` is the resident image: "fix" (fused_small, each task
/// one time, spare slots idle, equal barrier counts) or "old" (fused_small_old, the Part 1 layout). Every timed launch
/// and every timed span starts from the uploaded inputs. Today's kernel: one launch (median of 5) and the slope over
/// spans of 1, 10 and 100 launches. Resident: one launch of 1 repeat and the slope over 1, 10 and 100 repeats, with no
/// wait and with M-a. The old method of Part 1 (no upload between launches) runs too, for its numbers beside the new
/// ones. Bytes: 1 repeat with no wait and with M-a against one launch of today's kernel, and 10 repeats with M-a against
/// 10 launches. Then the tm0 stamps of 10 repeats (C10).
fn benchSmall2(b: *B, imgs: [5]xpu.Module, form: []const u8) !void {
    const alloc = std.heap.page_allocator;
    const old = std.mem.eql(u8, form, "old");
    const rimg = if (old) imgs[4] else imgs[3];
    var timer = try Timer.init(b.ctx);
    var prng = std.Random.DefaultPrng.init(0x5eed_f00d);
    const rng = prng.random();
    const all = bodies ++ [_]Body{ merge_sliding_body, q1_body };
    for (all, 0..) |body, bi| {
        const is_q1 = bi == bodies.len + 1;
        var nbuf: usize = 0;
        for (body.args) |a| {
            if (a == .buf) nbuf += 1;
        }
        const host = try alloc.alloc([]u8, nbuf);
        const outa = try alloc.alloc([]u8, nbuf);
        const outa10 = try alloc.alloc([]u8, nbuf);
        const tmp = try alloc.alloc([]u8, nbuf);
        const da = try alloc.alloc(xpu.DeviceBuffer, nbuf);
        const db = try alloc.alloc(xpu.DeviceBuffer, nbuf);
        {
            var i: usize = 0;
            for (body.args) |a| {
                if (a != .buf) continue;
                host[i] = try alloc.alloc(u8, a.buf.bytes);
                outa[i] = try alloc.alloc(u8, a.buf.bytes);
                outa10[i] = try alloc.alloc(u8, a.buf.bytes);
                tmp[i] = try alloc.alloc(u8, a.buf.bytes);
                fillGen(rng, a.buf.gen, host[i]);
                da[i] = try xpu.DeviceBuffer.alloc(b.ctx, a.buf.bytes);
                db[i] = try xpu.DeviceBuffer.alloc(b.ctx, a.buf.bytes);
                i += 1;
            }
        }
        defer for (0..nbuf) |i| {
            da[i].free();
            db[i].free();
        };
        if (is_q1 and old) {
            std.debug.print("BENCH small2 {s} form old: not run (a slot that leaves at its range check passes fewer barriers than its neighbour; a barrier wait has no time cap)\n", .{body.name});
            continue;
        }
        // today's kernel
        var rk = try imgs[body.img].kernel(body.ref, .{ body.local, 1, 1 });
        defer rk.deinit();
        try setArgs(rk, body.args, da);
        try uploadAll(b, da, host);
        var t_old: [5]f64 = undefined;
        t_old[0] = try timer.run(b, rk, body.grid, false);
        for (0..nbuf) |i| try da[i].download(b.s, 0, outa[i]);
        for (1..5) |r| t_old[r] = try timer.run(b, rk, body.grid, false);
        var t_one: [5]f64 = undefined;
        for (&t_one) |*t| {
            try uploadAll(b, da, host);
            t.* = try timer.run(b, rk, body.grid, false);
        }
        var ys_today: [3]f64 = undefined;
        for ([_]u32{ 1, 10, 100 }, 0..) |c, ci| {
            var tt: [5]f64 = undefined;
            for (&tt) |*t| {
                try uploadAll(b, da, host);
                t.* = try timer.span(b, rk, body.grid, c);
            }
            ys_today[ci] = medianF(&tt);
        }
        try uploadAll(b, da, host);
        _ = try timer.span(b, rk, body.grid, 10);
        for (0..nbuf) |i| try da[i].download(b.s, 0, outa10[i]);
        // the resident kernel
        var name_buf: [64]u8 = undefined;
        const rlen = (try std.fmt.bufPrint(name_buf[0..63], "fbr_{s}", .{body.runner})).len;
        name_buf[rlen] = 0;
        var k = try rimg.kernel(name_buf[0..rlen :0], .{ 512, 1, 1 });
        defer k.deinit();
        const max = xpu.launch.suggestCooperative(k) catch 0;
        if (max < 32) {
            std.debug.print("BENCH small2 {s}: refused (cooperative count {d} at 512)\n", .{ body.name, max });
            continue;
        }
        const na: u32 = @intCast(body.args.len);
        try setArgs(k, body.args, db);
        try k.setPtr(na, b.sync.ptr);
        try k.setPtr(na + 1, b.fault.ptr);
        try k.setU32(na + 4, spin_cap);
        var faults: u32 = 0;
        const Run = struct {
            fn one(bb: *B, tm: *Timer, kk: xpu.Kernel, n: u32, reps_: u32, mech: u32, f: *u32, name: []const u8) !f64 {
                try bb.reset();
                try kk.setU32(n + 2, reps_);
                try kk.setU32(n + 3, mech);
                const us = try tm.run(bb, kk, .{ 32, 1, 1 }, true);
                const w = try bb.faultWords();
                if (w[0] != 0) {
                    if (f.* == 0) try bb.sayFault(name, 32);
                    f.* += 1;
                }
                return us;
            }
        readFreq(init.io, &ctx) catch |e| {
            std.debug.print("BENCH clock: {s}, no spin run\n", .{@errorName(e)});
            return 3;
        };
        if (card_freq.rpn == 0 or card_freq.rp0 < card_freq.rpn) {
            std.debug.print("BENCH clock: no rpn_freq or rp0_freq in {s}, no spin run\n", .{freq_dir});
            return 3;
        }
        const cap = @as(f64, @floatFromInt(card_freq.rpn)) * cap_low_us;
        // bytes
        std.debug.print("BENCH tsc: {d:.3} ticks a us (best of 3 at 2^20 ticks, not used for the cap)\n", .{best});
        std.debug.print("BENCH cap: {s}: rpn {d} MHz, rp0 {d} MHz, min_freq {d}, max_freq {d}; spin cap {d} ticks = {d:.0} us at rpn, {d:.0} us at rp0\n", .{
            freq_dir, card_freq.rpn, card_freq.rp0, card_freq.min, card_freq.max, spin_cap, cap_low_us, cap / @as(f64, @floatFromInt(card_freq.rp0)),
        });
        try uploadAll(b, db, host);
        _ = try Run.one(b, &timer, k, na, 1, 0, &faults, body.name);
        eq[0] = try diffBytes(b, db, outa, tmp);
        try uploadAll(b, db, host);
        _ = try Run.one(b, &timer, k, na, 1, 1, &faults, body.name);
        eq[1] = try diffBytes(b, db, outa, tmp);
        try uploadAll(b, db, host);
        _ = try Run.one(b, &timer, k, na, 10, 1, &faults, body.name);
        eq[2] = try diffBytes(b, db, outa10, tmp);
        // times, corrected method: upload before each timed launch
        var one_r: [2]f64 = undefined;
        var slope_r: [2]f64 = undefined;
        var slope_o: [2]f64 = undefined;
        for ([_]u32{ 0, 1 }) |mech| {
            var ys: [3]f64 = undefined;
    else if (std.mem.eql(u8, name, "tsc")) try benchTsc(b)
    else if (std.mem.eql(u8, name, "soak")) try benchSoak(b)
            for ([_]u32{ 1, 10, 100 }, 0..) |c, ci| {
                var tt: [5]f64 = .{ 0, 0, 0, 0, 0 };
                for (&tt) |*t| {
                    if (faults != 0) break;
                    try uploadAll(b, db, host);
                    t.* = try Run.one(b, &timer, k, na, c, mech, &faults, body.name);
                }
                ys[ci] = medianF(&tt);
    else if (std.mem.eql(u8, name, "small2") or std.mem.eql(u8, name, "small2old")) {
        const d = dir orelse return error.BadArgs;
        var mods: [5]xpu.Module = undefined;
        for ([_][]const u8{ "ref_fuse", "ref_glue", "ref_attention", "fused_small", "fused_small_old" }, 0..) |n, j| {
            const path = try std.fmt.allocPrint(a, "{s}/{s}.bin", .{ d, n });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(64 << 20));
            defer a.free(file);
            const bin = try a.alignedAlloc(u8, .@"8", file.len);
            @memcpy(bin, file);
            mods[j] = try xpu.Module.loadNative(ctx, bin);
        }
        try benchSmall2(b, mods, if (std.mem.eql(u8, name, "small2")) "fix" else "old");
    } else if (std.mem.eql(u8, name, "small")) {
            one_r[mech] = ys[0];
            slope_r[mech] = slope3(ys);
            // the old method: the inputs as the launch before left them
            try uploadAll(b, db, host);
            for ([_]u32{ 1, 10, 100 }, 0..) |c, ci| {
                var tt: [5]f64 = .{ 0, 0, 0, 0, 0 };
                for (&tt) |*t| {
                    if (faults != 0) break;
                    t.* = try Run.one(b, &timer, k, na, c, mech, &faults, body.name);
                }
                ys[ci] = medianF(&tt);
            }
            slope_o[mech] = slope3(ys);
        }
        const thr_today = body.grid[0] * body.grid[1] * body.grid[2] * body.local;
        std.debug.print("BENCH small2 {s} form {s}: today {s} ({d}x{d}x{d} of {d}, {d} work-items): one launch {d:.3} us (old method {d:.3}), slope over 1, 10, 100 launches {d:.3} us ({d:.3}, {d:.3}, {d:.3}); resident 32 x 512 (16384 work-items): one launch of 1 repeat {d:.3} us no wait, {d:.3} us M-a; slope a repeat {d:.3} us no wait, {d:.3} us M-a (old method {d:.3}, {d:.3}); bytes differ: 1 repeat no wait {d}, 1 repeat M-a {d}, 10 repeats M-a against 10 launches {d}; faults {d}\n", .{
            body.name, form,         body.ref,   body.grid[0], body.grid[1], body.grid[2], body.local, thr_today,
            medianF(&t_one), medianF(&t_old), slope3(ys_today), ys_today[0], ys_today[1], ys_today[2],
            one_r[0],  one_r[1],     slope_r[0], slope_r[1],   slope_o[0],   slope_o[1],   eq[0],      eq[1],
            eq[2],     faults,
        });
        // C10: tm0 stamps of 10 repeats, every group
        // C10: the stamps with no wait, with M-a, and with M-a less its acquire fence (FB_NO_ACQ)
        if (faults == 0) for ([_]u32{ 0, 1, 1 | 0x20000 }) |mech| {
            try uploadAll(b, db, host);
            _ = try Run.one(b, &timer, k, na, 10, mech | 0x400, &faults, body.name);
            var sw: [32 * 10 * 3 * 8]u8 = undefined;
            try b.sync.download(b.s, 20000 * 4, &sw);
            var body0: [9]f64 = undefined;
            var wait0: [9]f64 = undefined;
            var bmax: [9]f64 = undefined;
            var wmin: [9]f64 = undefined;
            var bmax_g: u32 = 0;
            for (1..10) |r| {
                bmax[r - 1] = 0;
                wmin[r - 1] = 1e30;
                for (0..32) |g| {
                    var st: [3]u64 = undefined;
                    for (0..3) |q| st[q] = std.mem.readInt(u64, sw[((g * 10 + r) * 3 + q) * 8 ..][0..8], .little);
                    const bt: f64 = @floatFromInt(st[1] -% st[0]);
                    const wt: f64 = @floatFromInt(st[2] -% st[1]);
                    if (g == 0) {
                        body0[r - 1] = bt;
                        wait0[r - 1] = wt;
                    }
                    if (bt > bmax[r - 1]) {
                        bmax[r - 1] = bt;
                        bmax_g = @intCast(g);
                    }
                    wmin[r - 1] = @min(wmin[r - 1], wt);
                }
            }
            std.debug.print("BENCH small2 stamps {s} form {s} wait {s}: tm0 ticks a repeat (median of repeats 2 to 10): group 0 tasks {d:.0}, group 0 after the tasks to the wait end {d:.0}; largest tasks time of any group {d:.0} (group {d} in the last repeat), shortest time after the tasks {d:.0}\n", .{
                body.name, form, if (mech == 0) "none" else if (mech == 1) "a" else "a, no acquire fence", medianF(&body0), medianF(&wait0), medianF(&bmax), bmax_g, medianF(&wmin),
            });
        };
    }
}

        return @as(f64, @floatFromInt((ts.global_end -% ts.global_start) & t.mask)) * t.ns_per_tick / 1000.0;
    }
};

fn medianF(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

fn setArgs(k: xpu.Kernel, args: []const Arg, bufs: []xpu.DeviceBuffer) !void {
    var bi: usize = 0;
    for (args, 0..) |a, i| {
        switch (a) {
            .buf => {
                try k.setPtr(@intCast(i), bufs[bi].ptr);
                bi += 1;
            },
            .u => |v| try k.setU32(@intCast(i), v),
        }
    }
}

fn benchSmall(b: *B, imgs: [4]xpu.Module) !void {
    const alloc = std.heap.page_allocator;
    var timer = try Timer.init(b.ctx);
    var prng = std.Random.DefaultPrng.init(0x5eed_f00d);
    const rng = prng.random();
    for (bodies) |body| {
        var nbuf: usize = 0;
        for (body.args) |a| {
            if (a == .buf) nbuf += 1;
        }
        const host = try alloc.alloc([]u8, nbuf);
        const outa = try alloc.alloc([]u8, nbuf);
        const outb = try alloc.alloc([]u8, nbuf);
        const da = try alloc.alloc(xpu.DeviceBuffer, nbuf);
        const db = try alloc.alloc(xpu.DeviceBuffer, nbuf);
        {
            var bi: usize = 0;
            for (body.args) |a| {
                if (a != .buf) continue;
                host[bi] = try alloc.alloc(u8, a.buf.bytes);
                outa[bi] = try alloc.alloc(u8, a.buf.bytes);
                outb[bi] = try alloc.alloc(u8, a.buf.bytes);
                fillGen(rng, a.buf.gen, host[bi]);
                da[bi] = try xpu.DeviceBuffer.alloc(b.ctx, a.buf.bytes);
                db[bi] = try xpu.DeviceBuffer.alloc(b.ctx, a.buf.bytes);
                bi += 1;
            }
        }
        defer {
            for (0..nbuf) |i| {
                da[i].free();
                db[i].free();
            }
        }
        // today's kernel, one launch on the fresh inputs, then 4 more for the time
        var rk = try imgs[body.img].kernel(body.ref, .{ body.local, 1, 1 });
        defer rk.deinit();
        for (0..nbuf) |i| try da[i].upload(b.s, 0, host[i]);
        try b.s.synchronize();
        try setArgs(rk, body.args, da);
        var tr: [5]f64 = undefined;
        tr[0] = try timer.run(b, rk, body.grid, false);
        for (0..nbuf) |i| try da[i].download(b.s, 0, outa[i]);
        for (1..5) |r| tr[r] = try timer.run(b, rk, body.grid, false);
        const ref_us = medianF(&tr);
        // the resident kernel: 32 groups of 512, one repeat, no wait, on fresh inputs
        var name_buf: [64]u8 = undefined;
        const rlen = (try std.fmt.bufPrint(name_buf[0..63], "fbr_{s}", .{body.name})).len;
                    if (f[0] != 0) {
                        if (faults == 0) try b.sayFault(body.name, 32);
                        faults += 1;
                    }
        const rname: [:0]const u8 = name_buf[0..rlen :0];
        var k = try imgs[3].kernel(rname, .{ 512, 1, 1 });
        defer k.deinit();
        const max = xpu.launch.suggestCooperative(k) catch 0;
        if (max < 32) {
            std.debug.print("BENCH small {s}: refused (cooperative count {d} at 512)\n", .{ body.name, max });
            continue;
        }
        var p: ze.KernelProperties = .{};
        try b.ctx.d.check(b.ctx.d.api.zeKernelGetProperties(k.handle, &p), "zeKernelGetProperties");
/// before any other spin run. Then the other shapes, each mechanism once. Review C1: each launch prints its host time,
/// its device time (kernel timestamp event), the fault record with the tm0 ticks of the first capped group, and the
/// tick rate in the spin (those ticks over the device time: rounds 1 and 2 take microseconds).
        try b.reset();
        const na: u32 = @intCast(body.args.len);
        try setArgs(k, body.args, db);
        try k.setPtr(na, b.sync.ptr);
        try k.setPtr(na + 1, b.fault.ptr);
        try k.setU32(na + 2, 1);
        try k.setU32(na + 3, 0);
        try k.setU32(na + 4, spin_cap);
        _ = try timer.run(b, k, .{ 32, 1, 1 }, true);
        var diff_bytes: usize = 0;
        var diff_bufs: [16]u8 = undefined;
    var timer = try Timer.init(b.ctx);
        var nd: usize = 0;
    var worst: f64 = 0;
        for (0..nbuf) |i| {
            try db[i].download(b.s, 0, outb[i]);
            var d: usize = 0;
            for (outa[i], outb[i]) |x, y| d += @intFromBool(x != y);
            if (d != 0 and nd < diff_bufs.len) {
                diff_bufs[nd] = @intCast(i);
                nd += 1;
            }
            diff_bytes += d;
        }
        // the slope over 1, 10 and 100 repeats, with no wait and with M-a after each repeat
            try setAll(k, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech | 0x100, @as(u32, 0) });
            const mhz0 = readMhz(b.io, "act_freq");
        var slope: [2]f64 = undefined;
            const dev_us = try timer.run(b, k, .{ run.g, 1, 1 }, true);
            if (faults != 0) {
            const mhz1 = readMhz(b.io, "act_freq");
            // The launch time at rpn: the device time at the clock of the launch, scaled to the lowest clock.
            const mhz = if (@max(mhz0, mhz1) == 0) card_freq.rp0 else @max(mhz0, mhz1);
            const at_rpn = dev_us / 1000.0 * @as(f64, @floatFromInt(mhz)) / @as(f64, @floatFromInt(card_freq.rpn));
            worst = @max(worst, at_rpn);
                slope[mech] = 0;
            var fw: [36]u8 = undefined;
            try b.fault.download(b.s, 0, &fw);
            const ticks = (@as(u64, std.mem.readInt(u32, fw[32..36], .little)) << 32) | std.mem.readInt(u32, fw[28..32], .little);
                continue;
            }
            const counts = [_]f64{ 1, 10, 100 };
            std.debug.print("BENCH force G {d} S {d} mech {s}: launch {d:.3} ms host, {d:.3} ms device at act_freq {d} to {d} MHz, {d:.3} ms at rpn, fault word {d}, fault round {d}, counter {d}, generation {d}, tick rate in the spin {d:.1} ticks a us\n", .{
                run.g,                                  run.s,                                  mech_names[run.mech], ms, dev_us / 1000.0, mhz0, mhz1, at_rpn, f[0], f[2],
                std.mem.readInt(u32, cw[0..4], .little), std.mem.readInt(u32, cw[4..8], .little), @as(f64, @floatFromInt(ticks)) / dev_us,
                var tt: [5]f64 = undefined;
            try b.sayFault("force", run.g);
            if (f[0] == 0 or ms > 100.0 or at_rpn > 100.0) {
                    try b.reset();
                    try k.setU32(na + 2, @intFromFloat(c));
                    try k.setU32(na + 3, mech);
                    tt[r] = if (faults == 0) try timer.run(b, k, .{ 32, 1, 1 }, true) else 0;
                    const f = try b.faultWords();
                    if (f[0] != 0) faults += 1;
    std.debug.print("BENCH force passed: longest forced launch {d:.3} ms at rpn\n", .{worst});
}

/// Review C1: the tm0 rate in three states, each 3 times, with the card clock (sysfs act_freq) read before and after.
/// (1) fb_tsc alone, (2) fb_tsc queued right after a 40 MB read over the full grid, (3) fb_tsc_grid, the rate inside a
/// full grid that reads 1 GiB 4 times.
fn benchTsc(b: *B) !void {
    var timer = try Timer.init(b.ctx);
    var kt = try b.m256.kernel("fb_tsc", .{ 16, 1, 1 });
    defer kt.deinit();
    var kg = try b.m256.kernel("fb_tsc_grid", .{ 512, 1, 1 });
    defer kg.deinit();
    const groups = xpu.launch.suggestCooperative(kg) catch 0;
    if (groups == 0) return error.AboveCooperativeCount;
    const data = try b.big.at(0);
    const sink = try b.sink.at(0);
    const ticks: u32 = 1 << 20;
    for ([_][]const u8{ "alone", "after a 40 MB read", "beside a full grid" }, 0..) |name, mode| {
        for (0..3) |_| {
            try b.checkTime();
            const f0 = readMhz(b.io, "act_freq");
            var us: f64 = 0;
            if (mode == 2) {
                const per: u32 = @intCast(gib / 16 / groups); // uint4 a group
                try setAll(kg, .{ data, sink, @as(u32, 4), per });
                us = try timer.run(b, kg, .{ groups, 1, 1 }, false);
            } else {
                if (mode == 1) {
                    const per: u32 = @intCast(40_000_000 / 16 / groups);
                    try xpu.launch.dispatch(kg, b.s, .{ groups, 1, 1 }, .{ data, sink, @as(u32, 1), per });
                }
                try setAll(kt, .{ sink, ticks });
                us = try timer.run(b, kt, .{ 1, 1, 1 }, false);
            }
            const f1 = readMhz(b.io, "act_freq");
            var w: [8]u8 = undefined;
            try b.sink.download(b.s, 0, &w);
            const delta = std.mem.readInt(u64, w[0..8], .little);
            std.debug.print("BENCH tsc {s}: {d} ticks in {d:.3} us device time, {d:.1} ticks a us, act_freq {d} MHz before, {d} MHz after\n", .{ name, delta, us, @as(f64, @floatFromInt(delta)) / us, f0, f1 });
        }
    }
}

/// Review C2: each shape and mechanism, `n` launches of B2 (2,000 changes, no work), B3 (2,000 changes over a 64 KB
/// read) and B4 (100 Q8 phases of 4 MB). A configuration stops at its first fault (rule 2) with the fault record.
fn benchSoak(b: *B) !void {
    const n_launch: u32 = 50;
    const Run = struct { g: u32, s: u32, grf: u32, mech: u32 };
    const runs = [_]Run{
        .{ .g = 32, .s = 16, .grf = 256, .mech = 1 },  .{ .g = 32, .s = 32, .grf = 256, .mech = 1 }, .{ .g = 16, .s = 32, .grf = 256, .mech = 1 },
        .{ .g = 64, .s = 16, .grf = 256, .mech = 1 },  .{ .g = 128, .s = 8, .grf = 256, .mech = 1 }, .{ .g = 32, .s = 32, .grf = 128, .mech = 1 },
        .{ .g = 32, .s = 16, .grf = 256, .mech = 2 },  .{ .g = 32, .s = 32, .grf = 256, .mech = 2 }, .{ .g = 32, .s = 16, .grf = 256, .mech = 3 },
        .{ .g = 32, .s = 1, .grf = 256, .mech = 4 },   .{ .g = 256, .s = 1, .grf = 256, .mech = 4 }, .{ .g = 512, .s = 1, .grf = 256, .mech = 4 },
        .{ .g = 512, .s = 1, .grf = 256, .mech = 1 },  .{ .g = 1024, .s = 1, .grf = 256, .mech = 1 }, .{ .g = 1024, .s = 1, .grf = 256, .mech = 2 },
        .{ .g = 1024, .s = 1, .grf = 256, .mech = 4 },
    };
    const data = try b.big.at(0);
    const x = try b.x.at(0);
    const y = try b.y.at(0);
    const q8_n: u32 = @intFromFloat(4e6 / 2448.0);
    const q8_stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, q8_n) * 2448, 4096));
    const q8_nwin: u32 = @intCast(gib / q8_stride);
    for ([_][]const u8{ "B2", "B3", "B4" }, 0..) |bench, bi| {
        for (runs) |run| {
            var k = try b.module(run.grf).kernel(if (bi == 2) "fb_q8" else "fb_phase", .{ run.s * 16, 1, 1 });
            defer k.deinit();
            const max = xpu.launch.suggestCooperative(k) catch 0;
            if (max < run.g) {
                std.debug.print("BENCH soak {s} G {d} S {d} grf {d} mech {s}: refused (cooperative count {d})\n", .{ bench, run.g, run.s, run.grf, mech_names[run.mech], max });
                continue;
            }
            b.lag = bi != 2;
            var faults: u32 = 0;
            var done: u32 = 0;
            var worst: i96 = 0;
            for (0..n_launch) |_| {
                try b.checkTime();
                try b.reset();
                const t0 = nowNs(b.io);
                if (bi == 2) {
                    try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.big, x, y, b.sync, b.fault, q8_n, @as(u32, 100), q8_nwin, q8_stride, spin_cap, run.mech });
                } else {
                    try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech, @as(u32, if (bi == 1) 4096 else 0) });
                }
                try b.s.synchronize();
                worst = @max(worst, nowNs(b.io) - t0);
                done += 1;
                const f = try b.faultWords();
                if (f[0] != 0) {
                    faults += 1;
                    var tagb: [48]u8 = undefined;
                    try b.sayFault(try std.fmt.bufPrint(&tagb, "soak {s} launch {d}", .{ bench, done }), run.g);
                    break;
                }
            }
            b.lag = false;
            std.debug.print("BENCH soak {s} G {d} S {d} grf {d} mech {s}: faults {d} in {d} launches, longest launch {d:.3} ms\n", .{ bench, run.g, run.s, run.grf, mech_names[run.mech], faults, done, usOf(worst) / 1000.0 });
        }
    }
                }
                ys[ci] = medianF(&tt);
            }
            var sx: f64 = 0;
            var sy: f64 = 0;
            var sxx: f64 = 0;
            var sxy: f64 = 0;
            for (counts, ys) |x, y| {
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            slope[mech] = (3.0 * sxy - sx * sy) / (3.0 * sxx - sx * sx);
            std.debug.print("BENCH small {s} wait {s}: device us at 1, 10, 100 repeats {d:.3}, {d:.3}, {d:.3}\n", .{ body.name, mech_names[mech], ys[0], ys[1], ys[2] });
        }
        std.debug.print("BENCH small {s}: today {s} one launch {d:.3} us device (grid {d}x{d}x{d} of {d}); resident 32 x 512 {d:.3} us a repeat with no wait, {d:.3} us with M-a; outputs equal {s} ({d} bytes differ", .{
            body.name, body.ref, ref_us, body.grid[0], body.grid[1], body.grid[2], body.local, slope[0], slope[1], if (diff_bytes == 0) "yes" else "NO", diff_bytes,
        });
        for (diff_bufs[0..nd]) |i| std.debug.print(", buffer {d}", .{i});
        std.debug.print("); faults {d}; local {d} B, private {d} B, spill {d} B, cooperative count {d}\n", .{ faults, p.localMemSize, p.privateMemSize, p.spillMemSize, max });
    }
}

// ---------------------------------------------------------------- B9: the NVFP4 phase, today's decode and the new one
fn benchNv(b: *B) !void {
    const alloc = std.heap.page_allocator;
    const m = b.module(256);
    // hashed codes and scales over the whole big buffer
    {
        var kf = try m.kernel("fb_fill_hash", .{ 256, 1, 1 });
        defer kf.deinit();
        try xpu.launch.dispatch(kf, b.s, .{ 1024, 1, 1 }, .{ b.big, @as(u32, @intCast(gib / 4)), @as(u32, 0x2545f491) });
        try b.s.synchronize();
    }
    // the decode check: 65,536 blocks, 16 scale codes each (all 256 over the blocks), 1,024 factors
    const nphi = 1024;
    var phis: [nphi]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x0b70_f4f4);
    const rng = prng.random();
    for (&phis, 0..) |*f, i| {
        // log-uniform from 2^-40 to 2^40, random mantissa, each sign; times 2^26 as the kernels pass it
        const e: f32 = rng.float(f32) * 80.0 - 40.0;
        const v = std.math.exp2(e) * (1.0 + rng.float(f32));
        f.* = (if (i % 2 == 0) v else -v) * 0x1p26;
    }
    {
        var pbuf = try xpu.DeviceBuffer.alloc(b.ctx, nphi * 4);
        defer pbuf.free();
        try pbuf.upload(b.s, 0, std.mem.sliceAsBytes(&phis));
        try b.fault.fill8(b.s, 0);
        try b.s.synchronize();
        var kc = try m.kernel("fb_nv_check", .{ 256, 1, 1 });
        defer kc.deinit();
        const blocks: u32 = 65536;
        try xpu.launch.dispatch(kc, b.s, .{ 64, 1, 1 }, .{ b.big, pbuf, @as(u32, nphi), blocks, b.fault });
/// before any other spin run. Then the other shapes, each mechanism once. Review C1: each launch prints its host time,
/// its device time (kernel timestamp event), the fault record with the tm0 ticks of the first capped group, and the
/// tick rate in the spin (those ticks over the device time: rounds 1 and 2 take microseconds).
        var w: [8]u8 = undefined;
        try b.fault.download(b.s, 0, &w);
        const bad = std.mem.readInt(u32, w[0..4], .little);
        const guard = std.mem.readInt(u32, w[4..8], .little);
        const total: u64 = @as(u64, blocks) * 16 * nphi;
        std.debug.print("BENCH nv check: {d} blocks x 16 scale codes x {d} factors = {d} block decodes, {d} dwords differ, {d} took the guard\n", .{ blocks, nphi, total, bad, guard });
    }
    // the phase at 32 x 512, 10 triples a group, factor 1/2688 (amax 1, denominator 6 * 448) times 2^26
    const phi26: f32 = (1.0 / 2688.0) * 0x1p26;
    var k = try m.kernel("fb_nv_phase", .{ 512, 1, 1 });
    defer k.deinit();
    var timer = try Timer.init(b.ctx);
    const cmax = xpu.launch.suggestCooperative(k) catch 0;
    var worst: f64 = 0;
    if (cmax < 32) {
        std.debug.print("BENCH nv G 32 S 32: refused (cooperative count {d})\n", .{cmax});
        return;
    }
    const x = try b.x.at(0);
    const y = try b.y.at(0);
    const ya = try alloc.alloc(u8, 1 << 20);
    const yb = try alloc.alloc(u8, 1 << 20);
    for ([_]f64{ 4, 10, 20, 40, 245 }) |mb| {
        const rows: u32 = @intFromFloat(@round(mb * 1e6 / 1296.0 / 16.0) * 16.0);
        const stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, rows) * 1296, 4096));
            try setAll(k, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech | 0x100, @as(u32, 0) });
            const mhz0 = readMhz(b.io, "act_freq");
        const nwin: u32 = @intCast(gib / stride);
            const dev_us = try timer.run(b, k, .{ run.g, 1, 1 }, true);
        var us: [2]f64 = undefined;
            const mhz1 = readMhz(b.io, "act_freq");
            // The launch time at rpn: the device time at the clock of the launch, scaled to the lowest clock.
            const mhz = if (@max(mhz0, mhz1) == 0) card_freq.rp0 else @max(mhz0, mhz1);
            const at_rpn = dev_us / 1000.0 * @as(f64, @floatFromInt(mhz)) / @as(f64, @floatFromInt(card_freq.rpn));
            worst = @max(worst, at_rpn);
        var guard: u32 = 0;
            var fw: [36]u8 = undefined;
            try b.fault.download(b.s, 0, &fw);
            const ticks = (@as(u64, std.mem.readInt(u32, fw[32..36], .little)) << 32) | std.mem.readInt(u32, fw[28..32], .little);
        for ([_]u32{ 0, 1 }, 0..) |dec, di| {
            b.cfg_fault = false;
            try b.y.fill8(b.s, 0);
            std.debug.print("BENCH force G {d} S {d} mech {s}: launch {d:.3} ms host, {d:.3} ms device at act_freq {d} to {d} MHz, {d:.3} ms at rpn, fault word {d}, fault round {d}, counter {d}, generation {d}, tick rate in the spin {d:.1} ticks a us\n", .{
                run.g,                                  run.s,                                  mech_names[run.mech], ms, dev_us / 1000.0, mhz0, mhz1, at_rpn, f[0], f[2],
                std.mem.readInt(u32, cw[0..4], .little), std.mem.readInt(u32, cw[4..8], .little), @as(f64, @floatFromInt(ticks)) / dev_us,
            try b.y.download(b.s, 0, out[0 .. rows * 2]);
            try b.sayFault("force", run.g);
            if (f[0] == 0 or ms > 100.0 or at_rpn > 100.0) {
        readFreq(init.io, &ctx) catch |e| {
            std.debug.print("BENCH clock: {s}, no spin run\n", .{@errorName(e)});
            return 3;
        };
        if (card_freq.rpn == 0 or card_freq.rp0 < card_freq.rpn) {
            std.debug.print("BENCH clock: no rpn_freq or rp0_freq in {s}, no spin run\n", .{freq_dir});
            return 3;
        }
        const cap = @as(f64, @floatFromInt(card_freq.rpn)) * cap_low_us;
            try b.fault.download(b.s, 0, &w);
        std.debug.print("BENCH tsc: {d:.3} ticks a us (best of 3 at 2^20 ticks, not used for the cap)\n", .{best});
        std.debug.print("BENCH cap: {s}: rpn {d} MHz, rp0 {d} MHz, min_freq {d}, max_freq {d}; spin cap {d} ticks = {d:.0} us at rpn, {d:.0} us at rp0\n", .{
            freq_dir, card_freq.rpn, card_freq.rp0, card_freq.min, card_freq.max, spin_cap, cap_low_us, cap / @as(f64, @floatFromInt(card_freq.rp0)),
        });
            us[di] = usOf(tr.ns - t1.ns) / @as(f64, @floatFromInt(big_r - 1));
            std.debug.print("BENCH nv G 32 S 32 {d:.0} MB ({d} rows) decode {s}: {d:.3} us a phase, {d:.1} GB/s, faults {d}\n", .{
                mb, rows, if (dec == 0) "old" else "new", us[di], bytes / us[di] / 1e3, t1.faults + tr.faults,
    std.debug.print("BENCH force passed: longest forced launch {d:.3} ms at rpn\n", .{worst});
}

/// Review C1: the tm0 rate in three states, each 3 times, with the card clock (sysfs act_freq) read before and after.
/// (1) fb_tsc alone, (2) fb_tsc queued right after a 40 MB read over the full grid, (3) fb_tsc_grid, the rate inside a
/// full grid that reads 1 GiB 4 times.
fn benchTsc(b: *B) !void {
    var timer = try Timer.init(b.ctx);
    var kt = try b.m256.kernel("fb_tsc", .{ 16, 1, 1 });
    defer kt.deinit();
    var kg = try b.m256.kernel("fb_tsc_grid", .{ 512, 1, 1 });
    defer kg.deinit();
    else if (std.mem.eql(u8, name, "tsc")) try benchTsc(b)
    else if (std.mem.eql(u8, name, "soak")) try benchSoak(b)
    const groups = xpu.launch.suggestCooperative(kg) catch 0;
    if (groups == 0) return error.AboveCooperativeCount;
    const data = try b.big.at(0);
    const sink = try b.sink.at(0);
    const ticks: u32 = 1 << 20;
    for ([_][]const u8{ "alone", "after a 40 MB read", "beside a full grid" }, 0..) |name, mode| {
        for (0..3) |_| {
            try b.checkTime();
    else if (std.mem.eql(u8, name, "small2") or std.mem.eql(u8, name, "small2old")) {
        const d = dir orelse return error.BadArgs;
        var mods: [5]xpu.Module = undefined;
        for ([_][]const u8{ "ref_fuse", "ref_glue", "ref_attention", "fused_small", "fused_small_old" }, 0..) |n, j| {
            const path = try std.fmt.allocPrint(a, "{s}/{s}.bin", .{ d, n });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(64 << 20));
            defer a.free(file);
            const bin = try a.alignedAlloc(u8, .@"8", file.len);
            @memcpy(bin, file);
            mods[j] = try xpu.Module.loadNative(ctx, bin);
        }
        try benchSmall2(b, mods, if (std.mem.eql(u8, name, "small2")) "fix" else "old");
    } else if (std.mem.eql(u8, name, "small")) {
            var us: f64 = 0;
            if (mode == 2) {
                const per: u32 = @intCast(gib / 16 / groups); // uint4 a group
                try setAll(kg, .{ data, sink, @as(u32, 4), per });
                us = try timer.run(b, kg, .{ groups, 1, 1 }, false);
            } else {
                if (mode == 1) {
                    const per: u32 = @intCast(40_000_000 / 16 / groups);
                    try xpu.launch.dispatch(kg, b.s, .{ groups, 1, 1 }, .{ data, sink, @as(u32, 1), per });
                }
                try setAll(kt, .{ sink, ticks });
                us = try timer.run(b, kt, .{ 1, 1, 1 }, false);
            }
            const f1 = readMhz(b.io, "act_freq");
            var w: [8]u8 = undefined;
            try b.sink.download(b.s, 0, &w);
            const delta = std.mem.readInt(u64, w[0..8], .little);
            std.debug.print("BENCH tsc {s}: {d} ticks in {d:.3} us device time, {d:.1} ticks a us, act_freq {d} MHz before, {d} MHz after\n", .{ name, delta, us, @as(f64, @floatFromInt(delta)) / us, f0, f1 });
        }
    }
}

/// Review C2: each shape and mechanism, `n` launches of B2 (2,000 changes, no work), B3 (2,000 changes over a 64 KB
/// read) and B4 (100 Q8 phases of 4 MB). A configuration stops at its first fault (rule 2) with the fault record.
fn benchSoak(b: *B) !void {
    const n_launch: u32 = 50;
    const Run = struct { g: u32, s: u32, grf: u32, mech: u32 };
    const runs = [_]Run{
        .{ .g = 32, .s = 16, .grf = 256, .mech = 1 },  .{ .g = 32, .s = 32, .grf = 256, .mech = 1 }, .{ .g = 16, .s = 32, .grf = 256, .mech = 1 },
        .{ .g = 64, .s = 16, .grf = 256, .mech = 1 },  .{ .g = 128, .s = 8, .grf = 256, .mech = 1 }, .{ .g = 32, .s = 32, .grf = 128, .mech = 1 },
        .{ .g = 32, .s = 16, .grf = 256, .mech = 2 },  .{ .g = 32, .s = 32, .grf = 256, .mech = 2 }, .{ .g = 32, .s = 16, .grf = 256, .mech = 3 },
        .{ .g = 32, .s = 1, .grf = 256, .mech = 4 },   .{ .g = 256, .s = 1, .grf = 256, .mech = 4 }, .{ .g = 512, .s = 1, .grf = 256, .mech = 4 },
        .{ .g = 512, .s = 1, .grf = 256, .mech = 1 },  .{ .g = 1024, .s = 1, .grf = 256, .mech = 1 }, .{ .g = 1024, .s = 1, .grf = 256, .mech = 2 },
        .{ .g = 1024, .s = 1, .grf = 256, .mech = 4 },
    };
    const data = try b.big.at(0);
    const x = try b.x.at(0);
    const y = try b.y.at(0);
    const q8_n: u32 = @intFromFloat(4e6 / 2448.0);
    const q8_stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, q8_n) * 2448, 4096));
    const q8_nwin: u32 = @intCast(gib / q8_stride);
    for ([_][]const u8{ "B2", "B3", "B4" }, 0..) |bench, bi| {
        for (runs) |run| {
            var k = try b.module(run.grf).kernel(if (bi == 2) "fb_q8" else "fb_phase", .{ run.s * 16, 1, 1 });
            defer k.deinit();
            const max = xpu.launch.suggestCooperative(k) catch 0;
            if (max < run.g) {
                std.debug.print("BENCH soak {s} G {d} S {d} grf {d} mech {s}: refused (cooperative count {d})\n", .{ bench, run.g, run.s, run.grf, mech_names[run.mech], max });
                continue;
            }
            b.lag = bi != 2;
            var faults: u32 = 0;
            var done: u32 = 0;
            var worst: i96 = 0;
            for (0..n_launch) |_| {
                try b.checkTime();
                try b.reset();
                const t0 = nowNs(b.io);
                if (bi == 2) {
                    try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.big, x, y, b.sync, b.fault, q8_n, @as(u32, 100), q8_nwin, q8_stride, spin_cap, run.mech });
                } else {
                    try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech, @as(u32, if (bi == 1) 4096 else 0) });
                }
                try b.s.synchronize();
                worst = @max(worst, nowNs(b.io) - t0);
                done += 1;
                const f = try b.faultWords();
                if (f[0] != 0) {
                    faults += 1;
                    var tagb: [48]u8 = undefined;
                    try b.sayFault(try std.fmt.bufPrint(&tagb, "soak {s} launch {d}", .{ bench, done }), run.g);
                    break;
                }
            }
            b.lag = false;
            std.debug.print("BENCH soak {s} G {d} S {d} grf {d} mech {s}: faults {d} in {d} launches, longest launch {d:.3} ms\n", .{ bench, run.g, run.s, run.grf, mech_names[run.mech], faults, done, usOf(worst) / 1000.0 });
        }
    }
            });
        }
        var diff: usize = 0;
        for (ya[0 .. rows * 2], yb[0 .. rows * 2]) |p, q| diff += @intFromBool(p != q);
        std.debug.print("BENCH nv {d:.0} MB: outputs equal {s} ({d} bytes differ), guard blocks in the last run {d}\n", .{ mb, if (diff == 0) "yes" else "NO", diff, guard });
    }
}

/// Rule 1 of brief section 6: a forced fault (the last group stops its arrivals at round 3) on fb_phase, 2,000 rounds.
/// The small shape G 32 S 16 runs first, each mechanism once; any launch over 100 ms ends the bench with an error
/// before any other spin run. Then the other shapes, each mechanism once. Review C1: each launch prints its host time,
/// its device time (kernel timestamp event), the fault record with the tm0 ticks of the first capped group, and the
/// tick rate in the spin (those ticks over the device time: rounds 1 and 2 take microseconds).
fn benchForce(b: *B) !void {
    const Run = struct { g: u32, s: u32, mech: u32 };
    const first = [_]Run{ .{ .g = 32, .s = 16, .mech = 1 }, .{ .g = 32, .s = 16, .mech = 2 }, .{ .g = 32, .s = 16, .mech = 3 }, .{ .g = 32, .s = 16, .mech = 5 }, .{ .g = 32, .s = 1, .mech = 4 } };
    const rest = [_]Run{
        .{ .g = 16, .s = 32, .mech = 1 }, .{ .g = 16, .s = 32, .mech = 2 }, .{ .g = 16, .s = 32, .mech = 3 },
        .{ .g = 32, .s = 32, .mech = 1 }, .{ .g = 32, .s = 32, .mech = 2 }, .{ .g = 32, .s = 32, .mech = 3 },
        .{ .g = 64, .s = 16, .mech = 1 }, .{ .g = 64, .s = 16, .mech = 2 }, .{ .g = 64, .s = 16, .mech = 3 },
        .{ .g = 128, .s = 8, .mech = 1 }, .{ .g = 128, .s = 8, .mech = 2 },
        .{ .g = 64, .s = 16, .mech = 5 }, .{ .g = 128, .s = 8, .mech = 5 },
        .{ .g = 1024, .s = 1, .mech = 1 }, .{ .g = 1024, .s = 1, .mech = 2 }, .{ .g = 1024, .s = 1, .mech = 4 }, .{ .g = 1024, .s = 1, .mech = 5 },
    };
    var timer = try Timer.init(b.ctx);
    const data = try b.big.at(0);
    var worst: f64 = 0;
    for ([_][]const Run{ &first, &rest }, 0..) |list, li| {
        for (list) |run| {
            var k = try b.m256.kernel("fb_phase", .{ run.s * 16, 1, 1 });
            defer k.deinit();
            const max = xpu.launch.suggestCooperative(k) catch 0;
            if (max < run.g) {
                std.debug.print("BENCH force G {d} S {d} mech {s}: refused (cooperative count {d})\n", .{ run.g, run.s, mech_names[run.mech], max });
                continue;
            }
            try b.checkTime();
            try b.reset();
            try setAll(k, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech | 0x100, @as(u32, 0) });
            const mhz0 = readMhz(b.io, "act_freq");
            const t0 = nowNs(b.io);
            const dev_us = try timer.run(b, k, .{ run.g, 1, 1 }, true);
            const ms = usOf(nowNs(b.io) - t0) / 1000.0;
            const mhz1 = readMhz(b.io, "act_freq");
            // The launch time at rpn: the device time at the clock of the launch, scaled to the lowest clock.
            const mhz = if (@max(mhz0, mhz1) == 0) card_freq.rp0 else @max(mhz0, mhz1);
            const at_rpn = dev_us / 1000.0 * @as(f64, @floatFromInt(mhz)) / @as(f64, @floatFromInt(card_freq.rpn));
            worst = @max(worst, at_rpn);
            const f = try b.faultWords();
            var fw: [36]u8 = undefined;
            try b.fault.download(b.s, 0, &fw);
            const ticks = (@as(u64, std.mem.readInt(u32, fw[32..36], .little)) << 32) | std.mem.readInt(u32, fw[28..32], .little);
            var cw: [8]u8 = undefined;
            try b.sync.download(b.s, 0, cw[0..4]);
            try b.sync.download(b.s, 64, cw[4..8]);
            std.debug.print("BENCH force G {d} S {d} mech {s}: launch {d:.3} ms host, {d:.3} ms device at act_freq {d} to {d} MHz, {d:.3} ms at rpn, fault word {d}, fault round {d}, counter {d}, generation {d}, tick rate in the spin {d:.1} ticks a us\n", .{
                run.g,                                  run.s,                                  mech_names[run.mech], ms, dev_us / 1000.0, mhz0, mhz1, at_rpn, f[0], f[2],
                std.mem.readInt(u32, cw[0..4], .little), std.mem.readInt(u32, cw[4..8], .little), @as(f64, @floatFromInt(ticks)) / dev_us,
            });
            if (f[0] == 0 or ms > 100.0) {
                std.debug.print("BENCH force FAILED at G {d} S {d} mech {s}: {s}\n", .{ run.g, run.s, mech_names[run.mech], if (f[0] == 0) "no fault" else "over 100 ms" });
                return error.ForceFailed;
            }
        }
        if (li == 0) std.debug.print("BENCH force small shape passed\n", .{});
    }
    std.debug.print("BENCH force passed: longest forced launch {d:.3} ms at rpn\n", .{worst});
}

/// Review C1: the tm0 rate in three states, each 3 times, with the card clock (sysfs act_freq) read before and after.
/// (1) fb_tsc alone, (2) fb_tsc queued right after a 40 MB read over the full grid, (3) fb_tsc_grid, the rate inside a
/// full grid that reads 1 GiB 4 times.
fn benchTsc(b: *B) !void {
    var timer = try Timer.init(b.ctx);
    var kt = try b.m256.kernel("fb_tsc", .{ 16, 1, 1 });
    defer kt.deinit();
    var kg = try b.m256.kernel("fb_tsc_grid", .{ 512, 1, 1 });
    defer kg.deinit();
    const groups = xpu.launch.suggestCooperative(kg) catch 0;
    if (groups == 0) return error.AboveCooperativeCount;
    const data = try b.big.at(0);
    const sink = try b.sink.at(0);
    const ticks: u32 = 1 << 20;
    for ([_][]const u8{ "alone", "after a 40 MB read", "beside a full grid" }, 0..) |name, mode| {
        for (0..3) |_| {
            try b.checkTime();
            const f0 = readMhz(b.io, "act_freq");
            var us: f64 = 0;
            if (mode == 2) {
                const per: u32 = @intCast(gib / 16 / groups); // uint4 a group
                try setAll(kg, .{ data, sink, @as(u32, 4), per });
                us = try timer.run(b, kg, .{ groups, 1, 1 }, false);
            } else {
                if (mode == 1) {
                    const per: u32 = @intCast(40_000_000 / 16 / groups);
                    try xpu.launch.dispatch(kg, b.s, .{ groups, 1, 1 }, .{ data, sink, @as(u32, 1), per });
                }
                try setAll(kt, .{ sink, ticks });
                us = try timer.run(b, kt, .{ 1, 1, 1 }, false);
            }
            const f1 = readMhz(b.io, "act_freq");
            var w: [8]u8 = undefined;
            try b.sink.download(b.s, 0, &w);
            const delta = std.mem.readInt(u64, w[0..8], .little);
            std.debug.print("BENCH tsc {s}: {d} ticks in {d:.3} us device time, {d:.1} ticks a us, act_freq {d} MHz before, {d} MHz after\n", .{ name, delta, us, @as(f64, @floatFromInt(delta)) / us, f0, f1 });
        }
    }
        readFreq(init.io, &ctx) catch |e| {
            std.debug.print("BENCH clock: {s}, no spin run\n", .{@errorName(e)});
            return 3;
        };
        if (card_freq.rpn == 0 or card_freq.rp0 < card_freq.rpn) {
            std.debug.print("BENCH clock: no rpn_freq or rp0_freq in {s}, no spin run\n", .{freq_dir});
            return 3;
        }
        const cap = @as(f64, @floatFromInt(card_freq.rpn)) * cap_low_us;

        std.debug.print("BENCH tsc: {d:.3} ticks a us (best of 3 at 2^20 ticks, not used for the cap)\n", .{best});
        std.debug.print("BENCH cap: {s}: rpn {d} MHz, rp0 {d} MHz, min_freq {d}, max_freq {d}; spin cap {d} ticks = {d:.0} us at rpn, {d:.0} us at rp0\n", .{
            freq_dir, card_freq.rpn, card_freq.rp0, card_freq.min, card_freq.max, spin_cap, cap_low_us, cap / @as(f64, @floatFromInt(card_freq.rp0)),
        });

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var ordinal: ?u32 = null;
    var paths: [2]?[]const u8 = .{ null, null };
    var benches: []const u8 = "props";
    var dir: ?[]const u8 = null;
    var limit_s: u32 = 480;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--device") and i + 1 < args.len) {
            ordinal = std.fmt.parseInt(u32, args[i + 1], 10) catch return 2;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--limit-s") and i + 1 < args.len) {
    else if (std.mem.eql(u8, name, "tsc")) try benchTsc(b)
    else if (std.mem.eql(u8, name, "soak")) try benchSoak(b)
            limit_s = std.fmt.parseInt(u32, args[i + 1], 10) catch return 2;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--dir") and i + 1 < args.len) {
            dir = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--bench") and i + 1 < args.len) {
            benches = args[i + 1];
            i += 1;
        } else if (args[i].len > 0 and args[i][0] != '-' and (paths[0] == null or paths[1] == null)) {
            if (paths[0] == null) paths[0] = args[i] else paths[1] = args[i];
        } else {
            std.debug.print("usage: tf-xpu-fused-bench <fused_bench.bin> <fused_bench128.bin> [--device N] [--bench a,b,...]\n", .{});
            return 2;
        }
    }
    if (paths[0] == null or paths[1] == null) {
        std.debug.print("usage: tf-xpu-fused-bench <fused_bench.bin> <fused_bench128.bin> [--device N] [--bench a,b,...]\n", .{});
        return 2;
    }
    var bins: [2][]align(8) u8 = undefined;
    for (paths, 0..) |p, j| {
        const file = try std.Io.Dir.cwd().readFileAlloc(init.io, p.?, a, .limited(64 << 20));
        defer a.free(file);
        bins[j] = try a.alignedAlloc(u8, .@"8", file.len);
        @memcpy(bins[j], file);
    }
    defer for (bins) |bin| a.free(bin);

    var drv = try xpu.Driver.open();
    defer drv.close();
    var ctx = try xpu.Context.init(&drv, ordinal);
    defer ctx.deinit();
    var s = try xpu.Stream.init(&ctx);
    defer s.deinit();
    var m256 = try xpu.Module.loadNative(&ctx, bins[0]);
    defer m256.unload();
    var m128 = try xpu.Module.loadNative(&ctx, bins[1]);
    defer m128.unload();

    var b: B = .{
        .io = init.io,
        .ctx = &ctx,
        .s = s,
        .m256 = m256,
        .m128 = m128,
        .big = try xpu.DeviceBuffer.alloc(&ctx, gib),
        .sync = try xpu.DeviceBuffer.alloc(&ctx, 128 << 10),
        .stamp = try xpu.DeviceBuffer.alloc(&ctx, 128 << 10),
        .fault = try xpu.DeviceBuffer.alloc(&ctx, 64),
        .sink = try xpu.DeviceBuffer.alloc(&ctx, 64 << 10),
        .x = try xpu.DeviceBuffer.alloc(&ctx, 2 * 2304 * 2),
        .y = try xpu.DeviceBuffer.alloc(&ctx, 1 << 20),
        .deadline = nowNs(init.io) + @as(i96, limit_s) * std.time.ns_per_s,
    };
    defer {
        b.big.free();
        b.sync.free();
        b.stamp.free();
        b.fault.free();
        b.sink.free();
        b.x.free();
        b.y.free();
    }
    // Every byte 0x11: int8 code 17, fp16 scale 0x1111 (a small normal). x is bf16 1.0.
    try b.big.fill8(s, 0x11);
    var xh: [2 * 2304]u16 = @splat(0x3f80);
    try b.x.upload(s, 0, std.mem.sliceAsBytes(&xh));
    try s.synchronize();

    {
        var timer = try Timer.init(&ctx);
        var kt = try m256.kernel("fb_tsc", .{ 16, 1, 1 });
        defer kt.deinit();
        const ticks: u32 = 1 << 20;
        try kt.setPtr(0, b.sink.ptr);
        try kt.setU32(1, ticks);
        var best: f64 = 0;
        for (0..3) |_| {
            const us = try timer.run(&b, kt, .{ 1, 1, 1 }, false);
            var w: [16]u8 = undefined;
            try b.sink.download(b.s, 0, &w);
            const delta = std.mem.readInt(u64, w[0..8], .little);
            const reads = std.mem.readInt(u64, w[8..16], .little);
            const rate = @as(f64, @floatFromInt(delta)) / us;
            std.debug.print("BENCH tsc run: {d} ticks in {d:.3} us device time, {d} reads\n", .{ delta, us, reads });
            if (rate > best) best = rate;
        }
        if (best < 1.0) {
            std.debug.print("BENCH tsc: rate {d:.3} ticks a us is too low for a time cap, no spin run\n", .{best});
            return 3;
        }
        readFreq(init.io, &ctx) catch |e| {
            std.debug.print("BENCH clock: {s}, no spin run\n", .{@errorName(e)});
            return 3;
        };
        if (card_freq.rpn == 0 or card_freq.rp0 < card_freq.rpn) {
            std.debug.print("BENCH clock: no rpn_freq or rp0_freq in {s}, no spin run\n", .{freq_dir});
            return 3;
        }
        const cap = @as(f64, @floatFromInt(card_freq.rpn)) * cap_low_us;
        spin_cap = @intFromFloat(cap);
        std.debug.print("BENCH tsc: {d:.3} ticks a us (best of 3 at 2^20 ticks, not used for the cap)\n", .{best});
        std.debug.print("BENCH cap: {s}: rpn {d} MHz, rp0 {d} MHz, min_freq {d}, max_freq {d}; spin cap {d} ticks = {d:.0} us at rpn, {d:.0} us at rp0\n", .{
            freq_dir, card_freq.rpn, card_freq.rp0, card_freq.min, card_freq.max, spin_cap, cap_low_us, cap / @as(f64, @floatFromInt(card_freq.rp0)),
        });
    }
    var it = std.mem.splitScalar(u8, benches, ',');
    while (it.next()) |name| runOne(&b, &ctx, init, a, name, dir) catch |e| {
        std.debug.print("BENCH stopped: {s}\n", .{@errorName(e)});
        return 3;
    };
    std.debug.print("BENCH end\n", .{});
    return 0;
}

fn runOne(b: *B, ctx: *xpu.Context, init: std.process.Init, a: std.mem.Allocator, name: []const u8, dir: ?[]const u8) !void {
    b.cfg_fault = false;
    std.debug.print("BENCH start {s}\n", .{name});
    if (std.mem.eql(u8, name, "props")) try benchProps(b)
    else if (std.mem.eql(u8, name, "force")) try benchForce(b)
    else if (std.mem.eql(u8, name, "tsc")) try benchTsc(b)
    else if (std.mem.eql(u8, name, "phase")) try benchPhase(b, 0)
    else if (std.mem.eql(u8, name, "work")) try benchPhase(b, 4096)
    else if (std.mem.eql(u8, name, "wgbar")) try benchWgbar(b)
    else if (std.mem.eql(u8, name, "launch")) try benchLaunch(b)
    else if (std.mem.eql(u8, name, "q8")) try benchQ8(b)
    else if (std.mem.eql(u8, name, "stream")) try benchStream(b)
    else if (std.mem.eql(u8, name, "l2keep")) try benchL2keep(b)
    else if (std.mem.eql(u8, name, "nv")) try benchNv(b)
    else if (std.mem.eql(u8, name, "small")) {
        const d = dir orelse {
            std.debug.print("--bench small needs --dir <folder of fused_small.bin, ref_fuse.bin, ref_glue.bin, ref_attention.bin>\n", .{});
            return error.BadArgs;
        };
        var mods: [4]xpu.Module = undefined;
        for ([_][]const u8{ "ref_fuse", "ref_glue", "ref_attention", "fused_small" }, 0..) |n, j| {
            const path = try std.fmt.allocPrint(a, "{s}/{s}.bin", .{ d, n });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(64 << 20));
            defer a.free(file);
            const bin = try a.alignedAlloc(u8, .@"8", file.len);
            @memcpy(bin, file);
            mods[j] = try xpu.Module.loadNative(ctx, bin);
        }
        try benchSmall(b, mods);
    } else {
        std.debug.print("unknown bench {s}\n", .{name});
        return error.BadArgs;
    }
}

//! tf-xpu-fused-bench: the Part 1 benches of lane fused-visit (Recipe lanes/inkspeed/fused-visit/DESIGN.md section 8).
//! `tf-xpu-fused-bench <fused_bench.bin> <fused_bench128.bin> --device N --bench props,phase,work,wgbar,launch,q8,stream,l2keep`
//! Each number is a host wall time from the first append to the end of a stream synchronize, the median of 5 runs, and a
//! slope between two work counts, so the launch cost drops out. Every line starts with BENCH.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");

const reps = 5;
const spin_cap: u32 = 1_000_000;
const gib: usize = 1 << 30;

const Shape = struct { g: u32, s: u32, grf: u32 };
/// DESIGN section 8: the shapes at 256 GRF, then at 128 GRF.
const shapes = [_]Shape{
    .{ .g = 16, .s = 32, .grf = 256 },  .{ .g = 32, .s = 16, .grf = 256 }, .{ .g = 32, .s = 32, .grf = 256 },
    .{ .g = 64, .s = 8, .grf = 256 },   .{ .g = 64, .s = 16, .grf = 256 }, .{ .g = 128, .s = 8, .grf = 256 },
    .{ .g = 1024, .s = 1, .grf = 256 }, .{ .g = 16, .s = 64, .grf = 128 }, .{ .g = 32, .s = 32, .grf = 128 },
    .{ .g = 32, .s = 64, .grf = 128 },
};
/// One sub-group a group, the B0 grid counts, for M-e and the M-a control.
const s1_counts = [_]u32{ 32, 64, 128, 256, 512, 1024 };
const mech_names = [_][]const u8{ "none", "a", "b", "c", "e" };

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
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

/// Median ns of `reps` cooperative launches of `k` over `g` groups with `args`, the fault words of all runs summed.
fn timeCoop(b: *B, k: xpu.Kernel, g: u32, args: anytype) !struct { ns: i96, faults: u32, checks: u32 } {
    var t: [reps]i96 = undefined;
    var faults: u32 = 0;
    var checks: u32 = 0;
    for (0..reps) |r| {
        try b.reset();
        const t0 = nowNs(b.io);
        try xpu.launch.dispatchCooperative(k, b.s, .{ g, 1, 1 }, args);
        try b.s.synchronize();
        t[r] = nowNs(b.io) - t0;
        const f = try b.faultWords();
        if (f[0] != 0) faults += 1;
        checks += f[1];
    }
    return .{ .ns = median(&t), .faults = faults, .checks = checks };
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
        for ([_]u32{ 1, 2, 3 }) |mech| {
            if (mech == 3 and sh.g > 64) continue;
            try runs.append(std.heap.page_allocator, .{ .g = sh.g, .s = sh.s, .grf = sh.grf, .mech = mech });
        }
    }
    for (s1_counts) |g| {
        for ([_]u32{ 4, 1 }) |mech| try runs.append(std.heap.page_allocator, .{ .g = g, .s = 1, .grf = 256, .mech = mech });
    }
    for (runs.items) |run| {
        var k = try b.module(run.grf).kernel("fb_phase", .{ run.s * 16, 1, 1 });
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
        }
        std.debug.print("BENCH {s} G {d} S {d} grf {d} mech {s}: {d:.3} us a round, faults {d} of {d}, check errors {d}{s}\n", .{
            tag, run.g, run.s, run.grf, mech_names[run.mech], usOf(full.ns - zero.ns) / rounds, full.faults, reps, full.checks, extra,
        });
    }
}

/// B1: us a native barrier by group size and mode, on a full grid (the cooperative count).
fn benchWgbar(b: *B) !void {
    const rounds: u32 = 20000;
    const Size = struct { local: u32, grf: u32 };
    for ([_]Size{ .{ .local = 128, .grf = 256 }, .{ .local = 256, .grf = 256 }, .{ .local = 512, .grf = 256 }, .{ .local = 512, .grf = 128 }, .{ .local = 1024, .grf = 128 } }) |sz| {
        var k = b.module(sz.grf).kernel("fb_wgbar", .{ sz.local, 1, 1 }) catch {
            std.debug.print("BENCH wgbar local {d} grf {d}: refused\n", .{ sz.local, sz.grf });
            continue;
        };
        defer k.deinit();
        const g = xpu.launch.suggestCooperative(k) catch 0;
        if (g == 0) {
            std.debug.print("BENCH wgbar local {d} grf {d}: cooperative count 0\n", .{ sz.local, sz.grf });
            continue;
        }
        for ([_]u32{ 0, 1, 2, 3 }) |mode| {
            const z = try timeCoop(b, k, g, .{ b.sink, @as(u32, 0), mode });
            const f = try timeCoop(b, k, g, .{ b.sink, rounds, mode });
            std.debug.print("BENCH wgbar local {d} grf {d} groups {d} mode {d}: {d:.4} us a round\n", .{ sz.local, sz.grf, g, mode, usOf(f.ns - z.ns) / rounds });
        }
    }
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
    var d = try b.m256.kernel("tf_ink2_dense_q8_1", .{ 128, 1, 1 });
    defer d.deinit();
    const mbs = [_]f64{ 1, 4, 10, 20, 40 };
    var xs: [mbs.len]f64 = undefined;
    var ys: [mbs.len]f64 = undefined;
    const x = try b.x.at(0);
    const y = try b.y.at(0);
    const nil: ?*anyopaque = null;
    for (mbs, 0..) |mb, i| {
        const n: u32 = @intFromFloat(mb * 1e6 / 2448.0);
        const stride: usize = std.mem.alignForward(usize, @as(usize, n) * 2448, 4096);
        const nwin: usize = gib / stride;
        const count: u32 = 200;
        var t: [reps]i96 = undefined;
        for (0..reps) |r| {
            const t0 = nowNs(b.io);
            for (0..count) |c| {
                const w = try b.big.at((c % nwin) * stride);
                try xpu.launch.dispatch(d, b.s, .{ (n + 7) / 8, 1, 1 }, .{ x, w, y, nil, @as(u32, 1), n, @as(u32, 2304), n });
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
        sy += c;
        sxx += a * a;
        sxy += a * c;
    }
    const nf: f64 = @floatFromInt(mbs.len);
    const slope = (nf * sxy - sx * sy) / (nf * sxx - sx * sx);
    const icpt = (sy - slope * sx) / nf;
    std.debug.print("BENCH launch dense_q8_1 fit: fixed {d:.3} us a launch, marginal {d:.1} GB/s\n", .{ icpt, 1e3 / slope });
}

/// B4: GB/s of one Q8 phase inside a resident kernel, by shape and phase size, with a phase change (M-a) after each
/// phase and with none. The slope between 1 and R phases a launch.
fn benchQ8(b: *B) !void {
    const mbs = [_]f64{ 4, 10, 20, 40, 245 };
    const x = try b.x.at(0);
    const y = try b.y.at(0);
    for (shapes) |sh| {
        var k = try b.module(sh.grf).kernel("fb_q8", .{ sh.s * 16, 1, 1 });
        defer k.deinit();
        const max = xpu.launch.suggestCooperative(k) catch 0;
        if (max < sh.g) {
            std.debug.print("BENCH q8 G {d} S {d} grf {d}: refused (cooperative count {d})\n", .{ sh.g, sh.s, sh.grf, max });
            continue;
        }
        for (mbs) |mb| {
            const n: u32 = @intFromFloat(mb * 1e6 / 2448.0);
            const stride: u32 = @intCast(std.mem.alignForward(usize, @as(usize, n) * 2448, 4096));
            const nwin: u32 = @intCast(gib / stride);
            const bytes = @as(f64, @floatFromInt(n)) * 2448.0;
            const big_r: u32 = @intFromFloat(@max(4.0, @min(100.0, 2e9 / bytes)));
            for ([_]u32{ 1, 0 }) |mech| {
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
        const t_sg = sh.g * sh.s;
        for ([_]u32{ 0, 1, 3 }) |mech| {
          for (pf_mbs) |pf| {
            if (mech == 0 and pf != 0) continue;
            if (mech == 3 and sh.g > 64) continue;
            const pf_lines: u32 = @intFromFloat(@round(pf * 1e6 / @as(f64, @floatFromInt(t_sg)) / 64.0));
            var px: [phase_counts.len]f64 = undefined;
            var py: [phase_counts.len]f64 = undefined;
            for (phase_counts, 0..) |ph, i| {
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
    const rounds: u32 = 100;
    for ([_]u32{ 4, 8, 16, 24, 36 }) |mb| {
        const sb: u32 = mb * 1_000_000;
        var per: [2][2]f64 = undefined;
        for ([_]u32{ 0, 1 }) |pf| {
          for ([_]u32{ 0, 1 }) |rd| {
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

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var ordinal: ?u32 = null;
    var paths: [2]?[]const u8 = .{ null, null };
    var benches: []const u8 = "props";
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--device") and i + 1 < args.len) {
            ordinal = std.fmt.parseInt(u32, args[i + 1], 10) catch return 2;
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

    var it = std.mem.splitScalar(u8, benches, ',');
    while (it.next()) |name| {
        std.debug.print("BENCH start {s}\n", .{name});
        if (std.mem.eql(u8, name, "props")) try benchProps(&b)
        else if (std.mem.eql(u8, name, "phase")) try benchPhase(&b, 0)
        else if (std.mem.eql(u8, name, "work")) try benchPhase(&b, 4096)
        else if (std.mem.eql(u8, name, "wgbar")) try benchWgbar(&b)
        else if (std.mem.eql(u8, name, "launch")) try benchLaunch(&b)
        else if (std.mem.eql(u8, name, "q8")) try benchQ8(&b)
        else if (std.mem.eql(u8, name, "stream")) try benchStream(&b)
        else if (std.mem.eql(u8, name, "l2keep")) try benchL2keep(&b)
        else {
            std.debug.print("unknown bench {s}\n", .{name});
            return 2;
        }
    }
    std.debug.print("BENCH end\n", .{});
    return 0;
}

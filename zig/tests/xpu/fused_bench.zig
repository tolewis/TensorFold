//! tf-xpu-fused-bench: the Part 1 benches of lane fused-visit (Recipe lanes/inkspeed/fused-visit/DESIGN.md section 8).
//! `tf-xpu-fused-bench <fused_bench.bin> <fused_bench128.bin> --device N --bench props,phase,work,wgbar,launch,q8,stream,l2keep`
//! Each number is a host wall time from the first append to the end of a stream synchronize, the median of 5 runs, and a
//! slope between two work counts, so the launch cost drops out. Every line starts with BENCH.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");

const reps = 5;
// Polls before a fault. Run 1 used 1,000,000 and a faulted launch took seconds. Brief section 6 rule 1: a faulted
// launch ends in 100 ms or less (the force bench shows it), every group leaves when it sees the fault word.
const spin_cap: u32 = 20_000;
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
const mech_names = [_][]const u8{ "none", "a", "b", "c", "e", "f" };

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
    deadline: i96, // the bench time limit, checked before each launch (brief section 6 rule 3)
    cfg_fault: bool = false, // set at the first fault of a configuration: its other launches do not run (rule 2)

    fn checkTime(b: *B) !void {
        if (nowNs(b.io) > b.deadline) {
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
        std.debug.print("BENCH {s} G {d} S {d} grf {d} mech {s}: {d:.3} us a round, faults {d} of {d}, check errors {d}, last fault round {d}{s}\n", .{
            tag, run.g, run.s, run.grf, mech_names[run.mech], usOf(full.ns - zero.ns) / rounds, full.faults, reps, full.checks, full.round, extra,
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
            b.cfg_fault = false;
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
            if (mech == 0 and pf != 0) continue;
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
          for ([_]u32{ 0, 1 }) |rd| {
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
    .{ .name = "fuse_gates", .ref = "tf_ink2_fuse_gates", .img = 0, .local = 64, .grid = .{ 36, 3, 1 }, .args = &.{
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
    mask: u64,

    fn init(ctx: *xpu.Context) !Timer {
        var pool: ze.H = null;
        const api = ctx.d.api;
        try ctx.d.check(api.zeEventPoolCreate(ctx.gpu.context, &.{ .flags = ze.pool_host_visible | ze.pool_kernel_timestamp, .count = 1 }, 1, @ptrCast(&ctx.gpu.device), &pool), "zeEventPoolCreate");
        var ev: ze.H = null;
        try ctx.d.check(api.zeEventCreate(pool, &.{ .index = 0 }, &ev), "zeEventCreate");
        const bits = ctx.gpu.props.kernelTimestampValidBits;
        return .{ .ev = ev, .ns_per_tick = 1e9 / @as(f64, @floatFromInt(@max(ctx.gpu.props.timerResolution, 1))), .mask = if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1 };
    }

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
        var ts: ze.KernelTimestampResult = .{};
        try b.ctx.d.check(api.zeEventQueryKernelTimestamp(t.ev, &ts), "zeEventQueryKernelTimestamp");
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
        name_buf[rlen] = 0;
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
        for (0..nbuf) |i| try db[i].upload(b.s, 0, host[i]);
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
        var nd: usize = 0;
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
        var slope: [2]f64 = undefined;
        var faults: u32 = 0;
        for ([_]u32{ 0, 1 }) |mech| {
            if (faults != 0) {
                slope[mech] = 0;
                continue;
            }
            const counts = [_]f64{ 1, 10, 100 };
            var ys: [3]f64 = undefined;
            for (counts, 0..) |c, ci| {
                var tt: [5]f64 = undefined;
                for (0..5) |r| {
                    try b.reset();
                    try k.setU32(na + 2, @intFromFloat(c));
                    try k.setU32(na + 3, mech);
                    tt[r] = if (faults == 0) try timer.run(b, k, .{ 32, 1, 1 }, true) else 0;
                    const f = try b.faultWords();
                    if (f[0] != 0) faults += 1;
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
        try b.s.synchronize();
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
    const cmax = xpu.launch.suggestCooperative(k) catch 0;
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
        const nwin: u32 = @intCast(gib / stride);
        const bytes = @as(f64, @floatFromInt(rows)) * 1296.0;
        const big_r: u32 = @intFromFloat(@max(4.0, @min(100.0, 2e9 / bytes)));
        var us: [2]f64 = undefined;
        var guard: u32 = 0;
        for ([_]u32{ 0, 1 }, 0..) |dec, di| {
            b.cfg_fault = false;
            try b.y.fill8(b.s, 0);
            const t1 = try timeCoop(b, k, 32, .{ b.big, x, y, b.sync, b.fault, rows, @as(u32, 1), nwin, stride, spin_cap, dec, phi26 });
            const out = if (dec == 0) ya else yb;
            try b.y.download(b.s, 0, out[0 .. rows * 2]);
            const tr = try timeCoop(b, k, 32, .{ b.big, x, y, b.sync, b.fault, rows, big_r, nwin, stride, spin_cap, dec, phi26 });
            var w: [16]u8 = undefined;
            try b.fault.download(b.s, 0, &w);
            if (dec == 1) guard = std.mem.readInt(u32, w[12..16], .little);
            us[di] = usOf(tr.ns - t1.ns) / @as(f64, @floatFromInt(big_r - 1));
            std.debug.print("BENCH nv G 32 S 32 {d:.0} MB ({d} rows) decode {s}: {d:.3} us a phase, {d:.1} GB/s, faults {d}\n", .{
                mb, rows, if (dec == 0) "old" else "new", us[di], bytes / us[di] / 1e3, t1.faults + tr.faults,
            });
        }
        var diff: usize = 0;
        for (ya[0 .. rows * 2], yb[0 .. rows * 2]) |p, q| diff += @intFromBool(p != q);
        std.debug.print("BENCH nv {d:.0} MB: outputs equal {s} ({d} bytes differ), guard blocks in the last run {d}\n", .{ mb, if (diff == 0) "yes" else "NO", diff, guard });
    }
}

/// Rule 1 of brief section 6: a forced fault (the last group stops its arrivals at round 3) on fb_phase, 2,000 rounds.
/// The small shape G 32 S 16 runs first, each mechanism once; any launch over 100 ms ends the bench with an error
/// before any other spin run. Then the other shapes, each mechanism once.
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
    const data = try b.big.at(0);
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
            const t0 = nowNs(b.io);
            try xpu.launch.dispatchCooperative(k, b.s, .{ run.g, 1, 1 }, .{ b.sync, b.stamp, b.fault, data, b.sink, @as(u32, 2000), spin_cap, run.mech | 0x100, @as(u32, 0) });
            try b.s.synchronize();
            const ms = usOf(nowNs(b.io) - t0) / 1000.0;
            const f = try b.faultWords();
            var cw: [8]u8 = undefined;
            try b.sync.download(b.s, 0, cw[0..4]);
            try b.sync.download(b.s, 64, cw[4..8]);
            std.debug.print("BENCH force G {d} S {d} mech {s}: launch {d:.3} ms, fault word {d}, fault round {d}, counter {d}, generation {d}\n", .{
                run.g, run.s, mech_names[run.mech], ms, f[0], f[2], std.mem.readInt(u32, cw[0..4], .little), std.mem.readInt(u32, cw[4..8], .little),
            });
            if (f[0] == 0 or ms > 100.0) {
                std.debug.print("BENCH force FAILED at G {d} S {d} mech {s}: {s}\n", .{ run.g, run.s, mech_names[run.mech], if (f[0] == 0) "no fault" else "over 100 ms" });
                return error.ForceFailed;
            }
        }
        if (li == 0) std.debug.print("BENCH force small shape passed\n", .{});
    }
}

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

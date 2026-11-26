//! tf-xpu-barrier-bench: the cost of one grid-wide barrier on a card, from 32 to 1,024 work-groups.
//! Each launch is timed from the dispatch to the stream draining; the fault word says whether the spin cap ended it.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");

/// Work-groups of 16 lanes: at 256 GRF the card holds 1,024 of them, so never above that.
const group_counts = [_]u32{ 32, 64, 128, 256, 512, 1024 };
const reps = 5;
const rounds: u32 = 2000;
const spin_cap: u32 = 1_000_000;

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn median(v: *[reps]i96) i96 {
    std.mem.sort(i96, v, {}, std.sort.asc(i96));
    return v[reps / 2];
}

const Bufs = struct {
    arrive: xpu.DeviceBuffer,
    gen: xpu.DeviceBuffer,
    fault: xpu.DeviceBuffer,
    sink: xpu.DeviceBuffer,

    fn zero(b: *Bufs, s: xpu.Stream) !void {
        try b.arrive.fill8(s, 0);
        try b.gen.fill8(s, 0);
        try b.fault.fill8(s, 0);
        try b.sink.fill8(s, 0);
        try s.synchronize();
    }

    fn faulted(b: *Bufs, s: xpu.Stream) !bool {
        var w: [4]u8 = undefined;
        try b.fault.download(s, 0, &w);
        return std.mem.readInt(u32, &w, .little) != 0;
    }

    fn deinit(b: *Bufs) void {
        b.arrive.free();
        b.gen.free();
        b.fault.free();
        b.sink.free();
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-xpu-barrier-bench <barrier_bench.bin> [--device N]\n", .{});
        return 2;
    }
    var ordinal: ?u32 = null;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--device") and i + 1 < args.len) {
            ordinal = std.fmt.parseInt(u32, args[i + 1], 10) catch return 2;
            i += 1;
        } else {
            std.debug.print("usage: tf-xpu-barrier-bench <barrier_bench.bin> [--device N]\n", .{});
            return 2;
        }
    }
    const file = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(64 << 20));
    defer a.free(file);
    const bin = try a.alignedAlloc(u8, .@"8", file.len);
    defer a.free(bin);
    @memcpy(bin, file);

    var drv = try xpu.Driver.open();
    defer drv.close();
    var ctx = try xpu.Context.init(&drv, ordinal);
    defer ctx.deinit();
    var s = try xpu.Stream.init(&ctx);
    defer s.deinit();
    var m = try xpu.Module.loadNative(&ctx, bin);
    defer m.unload();
    var k = try m.kernel("tf_barrier_bench", .{ 16, 1, 1 });
    defer k.deinit();

    var p: ze.KernelProperties = .{};
    try ctx.d.check(ctx.d.api.zeKernelGetProperties(k.handle, &p), "zeKernelGetProperties");
    std.debug.print("BENCH kernel tf_barrier_bench: {d} work-items required, sub-group {d}, {d} SLM bytes, {d} spill bytes, {d} private bytes, {d} arguments\n", .{
        @as(u64, p.requiredGroupSize[0]) * p.requiredGroupSize[1] * p.requiredGroupSize[2],
        p.requiredSubgroupSize,
        p.localMemSize,
        p.spillMemSize,
        p.privateMemSize,
        p.numKernelArgs,
    });
    if (p.spillMemSize != 0) std.debug.print("BENCH note: the kernel spills {d} bytes\n", .{p.spillMemSize});

    var b: Bufs = .{
        .arrive = try xpu.DeviceBuffer.alloc(&ctx, 4),
        .gen = try xpu.DeviceBuffer.alloc(&ctx, 4),
        .fault = try xpu.DeviceBuffer.alloc(&ctx, 4),
        .sink = try xpu.DeviceBuffer.alloc(&ctx, 4),
    };
    defer b.deinit();
    const arrive = try b.arrive.at(0);
    const gen = try b.gen.at(0);
    const fault = try b.fault.at(0);
    const sink = try b.sink.at(0);

    for (group_counts) |g| {
        try b.zero(s);
        var t: [2][reps]i96 = undefined;
        var any_fault = false;
        for (0..2) |which| {
            const n: u32 = if (which == 0) 0 else rounds;
            for (0..reps) |r| {
                const t0 = nowNs(init.io);
                try xpu.launch.dispatch(k, s, .{ g, 1, 1 }, .{ arrive, gen, fault, sink, n, spin_cap });
                try s.synchronize();
                t[which][r] = nowNs(init.io) - t0;
                if (try b.faulted(s)) any_fault = true;
            }
        }
        const us = @as(f64, @floatFromInt(median(&t[1]) - median(&t[0]))) / rounds / 1000.0;
        std.debug.print("BENCH G {d} barrier {d:.3} us fault {d}\n", .{ g, us, @intFromBool(any_fault) });
        if (any_fault) {
            std.debug.print("BENCH G {d} fault\n", .{g});
            break;
        }
    }
    return 0;
}

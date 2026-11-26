//! tf-xpu-barrier-bench: the cost of one grid-wide barrier on a card, flat (one counter) or two-level (leaf and root).
//! Each launch is timed from the dispatch to the stream draining; the fault word says whether the spin cap ended it.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");

/// Work-groups of 16 lanes: at 256 GRF the card holds 1,024 of them, so never above that.
const max_groups: u32 = 1024;
/// Leaf counters sit 16 uints apart, one cache line a leaf.
const stride: u32 = 16;
const flat_counts = [_]u32{ 32, 64, 128, 256, 512, 1024 };
const tree_counts = [_]u32{ 128, 256, 512, 1024 };
const tree_leafs = [_]u32{ 16, 32 };
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
    root_arrive: xpu.DeviceBuffer,
    root_gen: xpu.DeviceBuffer,
    fault: xpu.DeviceBuffer,
    sink: xpu.DeviceBuffer,

    fn zero(b: *Bufs, s: xpu.Stream) !void {
        try b.arrive.fill8(s, 0);
        try b.gen.fill8(s, 0);
        try b.root_arrive.fill8(s, 0);
        try b.root_gen.fill8(s, 0);
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
        b.root_arrive.free();
        b.root_gen.free();
        b.fault.free();
        b.sink.free();
    }
};

/// Median microseconds of one barrier: the 2000-round launch minus the empty launch, both at `reps` runs.
fn price(io: std.Io, b: *Bufs, s: xpu.Stream, k: xpu.Kernel, g: u32, leaf: u32, comptime tree: bool) !struct { us: f64, fault: bool } {
    try b.zero(s);
    const arrive = try b.arrive.at(0);
    const gen = try b.gen.at(0);
    const root_arrive = try b.root_arrive.at(0);
    const root_gen = try b.root_gen.at(0);
    const fault = try b.fault.at(0);
    const sink = try b.sink.at(0);
    var t: [2][reps]i96 = undefined;
    var any_fault = false;
    for (0..2) |which| {
        const n: u32 = if (which == 0) 0 else rounds;
        for (0..reps) |r| {
            const t0 = nowNs(io);
            if (tree) {
                try xpu.launch.dispatch(k, s, .{ g, 1, 1 }, .{ arrive, gen, root_arrive, root_gen, fault, sink, n, spin_cap, leaf });
            } else {
                try xpu.launch.dispatch(k, s, .{ g, 1, 1 }, .{ arrive, gen, fault, sink, n, spin_cap });
            }
            try s.synchronize();
            t[which][r] = nowNs(io) - t0;
            if (try b.faulted(s)) any_fault = true;
        }
    }
    return .{ .us = @as(f64, @floatFromInt(median(&t[1]) - median(&t[0]))) / rounds / 1000.0, .fault = any_fault };
}

fn sayProps(ctx: *const xpu.Context, k: xpu.Kernel, name: []const u8) !void {
    var p: ze.KernelProperties = .{};
    try ctx.d.check(ctx.d.api.zeKernelGetProperties(k.handle, &p), "zeKernelGetProperties");
    std.debug.print("BENCH kernel {s}: {d} work-items required, sub-group {d}, {d} SLM bytes, {d} spill bytes, {d} private bytes, {d} arguments\n", .{
        name,
        @as(u64, p.requiredGroupSize[0]) * p.requiredGroupSize[1] * p.requiredGroupSize[2],
        p.requiredSubgroupSize,
        p.localMemSize,
        p.spillMemSize,
        p.privateMemSize,
        p.numKernelArgs,
    });
    if (p.spillMemSize != 0) std.debug.print("BENCH note: {s} spills {d} bytes\n", .{ name, p.spillMemSize });
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var ordinal: ?u32 = null;
    var tree = false;
    var path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--device") and i + 1 < args.len) {
            ordinal = std.fmt.parseInt(u32, args[i + 1], 10) catch return 2;
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--tree")) {
            tree = true;
        } else if (path == null and args[i].len > 0 and args[i][0] != '-') {
            path = args[i];
        } else {
            std.debug.print("usage: tf-xpu-barrier-bench <barrier_bench.bin> [--device N] [--tree]\n", .{});
            return 2;
        }
    }
    const bin_path = path orelse {
        std.debug.print("usage: tf-xpu-barrier-bench <barrier_bench.bin> [--device N] [--tree]\n", .{});
        return 2;
    };
    const file = try std.Io.Dir.cwd().readFileAlloc(init.io, bin_path, a, .limited(64 << 20));
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

    var b: Bufs = .{
        .arrive = try xpu.DeviceBuffer.alloc(&ctx, (max_groups / stride) * stride * 4),
        .gen = try xpu.DeviceBuffer.alloc(&ctx, (max_groups / stride) * stride * 4),
        .root_arrive = try xpu.DeviceBuffer.alloc(&ctx, 4),
        .root_gen = try xpu.DeviceBuffer.alloc(&ctx, 4),
        .fault = try xpu.DeviceBuffer.alloc(&ctx, 4),
        .sink = try xpu.DeviceBuffer.alloc(&ctx, 4),
    };
    defer b.deinit();

    if (tree) {
        var k = try m.kernel("tf_barrier_bench_tree", .{ 16, 1, 1 });
        defer k.deinit();
        try sayProps(&ctx, k, "tf_barrier_bench_tree");
        for (tree_counts) |g| {
            for (tree_leafs) |leaf| {
                const r = try price(init.io, &b, s, k, g, leaf, true);
                std.debug.print("BENCH G {d} leaf {d} barrier {d:.3} us fault {d}\n", .{ g, leaf, r.us, @intFromBool(r.fault) });
                if (r.fault) std.debug.print("BENCH G {d} leaf {d} fault\n", .{ g, leaf });
            }
        }
        return 0;
    }

    var k = try m.kernel("tf_barrier_bench", .{ 16, 1, 1 });
    defer k.deinit();
    try sayProps(&ctx, k, "tf_barrier_bench");
    for (flat_counts) |g| {
        const r = try price(init.io, &b, s, k, g, 0, false);
        std.debug.print("BENCH G {d} barrier {d:.3} us fault {d}\n", .{ g, r.us, @intFromBool(r.fault) });
        if (r.fault) {
            std.debug.print("BENCH G {d} fault\n", .{g});
            break;
        }
    }
    return 0;
}

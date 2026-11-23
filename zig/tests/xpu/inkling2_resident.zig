//! Device test for Inkling 2 Mini's resident model, stream state and snapshots on the XPU backend: loads the checkpoint,
//! prints resident and free bytes and the largest single-stream context that fits with int8 and bf16 K/V (each one
//! allocated for real), checks snapshot save and restore and the begin/commit/rollback transaction on the device, then
//! frees everything and compares the card's free memory with its start value.
//! Usage: tf-xpu-inkling2-resident-test <model dir> [--embed-host]   (or TF_XPU_INKLING2 for the directory)
const std = @import("std");
const xpu = @import("xpu");
const ink = @import("inkling2_xpu");

pub const panic = std.debug.FullPanic(xpu.rt.panicDrain);

const model = ink.model;
const state = ink.state;
const snapshot = ink.snapshot;
/// The desktop shares the card: its own use moves the free figure by this much between two readings.
const tolerance: u64 = 256 << 20;
const gib = 1024.0 * 1024.0 * 1024.0;
/// The global capacity the scratch's attention partials are sized for.
const scratch_capacity = 8192;

fn gb(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / gib;
}

fn freeBytes(ctx: *xpu.Context) !u64 {
    return (try ctx.gpu.memory()).free;
}

fn downloadAll(gpa: std.mem.Allocator, r: *xpu.rt.Runtime, b: xpu.rt.Buffer) ![]u8 {
    const out = try gpa.alloc(u8, b.len);
    errdefer gpa.free(out);
    try r.download(out, b);
    try r.sync();
    return out;
}

fn fillRandom(gpa: std.mem.Allocator, r: *xpu.rt.Runtime, b: xpu.rt.Buffer, seed: u64) !void {
    const host = try gpa.alloc(u8, b.len);
    defer gpa.free(host);
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().bytes(host);
    try r.upload(b, host);
    try r.sync();
}

/// The cache's K/V, conv and taps are random in cache `a`; a snapshot of `len` positions restores into `b` exactly.
fn snapshotCheck(gpa: std.mem.Allocator, r: *xpu.rt.Runtime, schedule: *const ink.plan.Plan, kv: state.KvDtype) !void {
    var a = try state.Cache.init(r, 2048, schedule, kv);
    defer a.deinit();
    var b = try state.Cache.init(r, 4096, schedule, kv);
    defer b.deinit();
    for (&a.slots, 0..) |*s, i| try fillRandom(gpa, r, s.buf, 100 + i);
    try fillRandom(gpa, r, a.conv, 7);
    try fillRandom(gpa, r, a.taps, 8);
    const len = 1500;
    a.len = len;
    var snap = try snapshot.save(r, &a);
    defer snap.deinit();
    if (snap.bytes != snapshot.bytes(len, kv)) return error.SnapshotBytes;
    try snapshot.restore(&b, &snap);
    if (b.len != len) return error.SnapshotLen;
    for (0..state.visit_slots) |i| {
        const live = if (a.slots[i].global) len else @min(len, a.slots[i].ring);
        const x = try downloadAll(gpa, r, a.slots[i].buf);
        defer gpa.free(x);
        const y = try downloadAll(gpa, r, b.slots[i].buf);
        defer gpa.free(y);
        const sa = a.slots[i];
        const sb = b.slots[i];
        const pieces = [_]struct { a: usize, b: usize, n: usize }{
            .{ .a = sa.key_at, .b = sb.key_at, .n = live * kv.rowBytes() },
            .{ .a = sa.key_scale_at, .b = sb.key_scale_at, .n = live * kv.scaleBytes() },
            .{ .a = sa.value_at, .b = sb.value_at, .n = live * kv.rowBytes() },
            .{ .a = sa.value_scale_at, .b = sb.value_scale_at, .n = live * kv.scaleBytes() },
        };
        for (pieces) |p| if (!std.mem.eql(u8, x[p.a..][0..p.n], y[p.b..][0..p.n])) return error.SnapshotMismatch;
    }
    const ca = try downloadAll(gpa, r, a.conv);
    defer gpa.free(ca);
    const cb = try downloadAll(gpa, r, b.conv);
    defer gpa.free(cb);
    if (!std.mem.eql(u8, ca, cb)) return error.SnapshotConvMismatch;
    // a snapshot restores only into an empty cache of the same K/V dtype
    if (snapshot.restore(&b, &snap)) |_| return error.RestoreIntoUsedCache else |_| {}
    std.debug.print("snapshot {s}: {d} positions, {d:.1} MiB, restored exactly\n", .{ @tagName(kv), len, @as(f64, @floatFromInt(snap.bytes)) / (1 << 20) });
}

/// begin copies the conv states aside; commit(k) keeps the first k rows' staged taps; rollback puts the copy back.
fn transactionCheck(gpa: std.mem.Allocator, r: *xpu.rt.Runtime, schedule: *const ink.plan.Plan) !void {
    var c = try state.Cache.init(r, 128, schedule, .int8);
    defer c.deinit();
    try fillRandom(gpa, r, c.conv, 21);
    try fillRandom(gpa, r, c.taps, 22);
    const taps = try downloadAll(gpa, r, c.taps);
    defer gpa.free(taps);
    try c.begin(3);
    try fillRandom(gpa, r, c.conv, 23); // the forward overwrites the states
    try c.commit(1);
    const after = try downloadAll(gpa, r, c.conv);
    defer gpa.free(after);
    for (0..state.visit_slots) |slot| for (0..state.conv_states) |s| {
        const n = 3 * state.convWidth(s) * 2;
        const want = taps[state.convByteOffset(slot, s, 1, 3 + state.max_rows)..][0..n];
        if (!std.mem.eql(u8, after[state.convByteOffset(slot, s, 0, 3)..][0..n], want)) return error.CommitMismatch;
    };
    if (c.len != 1 or c.pending != null) return error.CommitState;
    try c.begin(2);
    try fillRandom(gpa, r, c.conv, 24);
    try c.rollback();
    const rolled = try downloadAll(gpa, r, c.conv);
    defer gpa.free(rolled);
    if (!std.mem.eql(u8, rolled, after)) return error.RollbackMismatch;
    std.debug.print("transaction: commit(1 of 3) took the staged taps, rollback restored the saved states\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var dir: ?[]const u8 = if (std.c.getenv("TF_XPU_INKLING2")) |v| std.mem.span(v) else null;
    var options: model.Options = .{};
    for (argv[1..]) |a| {
        if (std.mem.eql(u8, a, "--embed-host")) options.embed_host = true else dir = a;
    }
    const path = dir orelse {
        std.debug.print("skipped: no model directory (argument or TF_XPU_INKLING2)\n", .{});
        return;
    };
    xpu.stop.install();
    var drv = xpu.Driver.open() catch {
        std.debug.print("skipped: no Level Zero loader\n", .{});
        return;
    };
    var ctx = try xpu.Context.init(&drv, null);
    var stream = try xpu.Stream.init(&ctx);
    var r = try xpu.rt.Runtime.init(&ctx, stream);

    const start_free = try freeBytes(&ctx);
    const mem = try ctx.gpu.memory();
    std.debug.print("{s}: size {d:.2} GiB, free at start {d:.2} GiB, largest single allocation {d:.2} GiB\n", .{ ctx.name(), gb(mem.size), gb(start_free), gb(ctx.maxAlloc()) });

    const t0 = std.Io.Clock.awake.now(init.io);
    var m = try model.Model.load(gpa, &r, path, options);
    const load_ms = @divTrunc(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds(), 1_000_000);
    const after_load = try freeBytes(&ctx);
    std.debug.print("loaded in {d} ms: resident device {d} bytes ({d:.3} GiB), pinned host {d} bytes, embedding on {s}\n", .{ load_ms, m.device_bytes, gb(m.device_bytes), m.host_bytes, if (options.embed_host) "host" else "device" });
    std.debug.print("free after load {d} bytes ({d:.3} GiB); card free dropped {d:.3} GiB\n", .{ after_load, gb(after_load), gb(start_free -| after_load) });

    // spot checks that the bytes on the device are the checkpoint's
    const w = &m.weights;
    if (w.layers[0].gu.amax == null or w.layers[20].gu.amax != null) return error.ExpertFormats;
    if (w.layers[0].attn.k_norm.buf.ptr != w.layers[0].attn.q_norm.buf.ptr) return error.SharedNorm;
    if (w.layers[5].attn.k_norm.buf.ptr == w.layers[5].attn.q_norm.buf.ptr) return error.GlobalNorms;
    if (w.layers[6].controllers == null or w.layers[5].controllers != null) return error.Controllers;
    std.debug.print("weights: layer 0 experts NVFP4 ({d} bytes w13), layer 20 MXFP8 ({d} bytes w13), controllers on layers 6 to 17\n", .{ w.layers[0].gu.weight.bytes, w.layers[20].gu.weight.bytes });

    const scratch_bytes = state.scratchBytes(state.max_rows, scratch_capacity);
    var scratch = try state.Scratch.init(&r, state.max_rows, scratch_capacity);
    const with_scratch = try freeBytes(&ctx);
    std.debug.print("scratch for {d} rows: {d:.1} MiB (free now {d:.3} GiB)\n", .{ state.max_rows, @as(f64, @floatFromInt(scratch_bytes)) / (1 << 20), gb(with_scratch) });

    var prec = model.precisions();
    const schedule = try ink.plan.build(&m.config.text, &prec);

    for ([_]state.KvDtype{ .int8, .bf16 }) |kv| {
        const budget = with_scratch;
        var cap = state.largestCapacity(budget, kv);
        var cache: state.Cache = undefined;
        while (true) {
            if (cap == 0) return error.NothingFits;
            if (state.Cache.init(&r, cap, &schedule, kv)) |c| {
                cache = c;
                break;
            } else |err| {
                if (err != error.OutOfDeviceMemory) return err;
                cap -= @max(cap / 100, 64);
            }
        }
        // allocation alone commits nothing on this driver: touch every byte so the free figure is what the cache holds
        for (&cache.slots) |*sl| try state.zero(&r, sl.buf, sl.buf.len);
        try stream.synchronize();
        const left = try freeBytes(&ctx);
        std.debug.print("K/V {s}: largest single-stream context allocated {d} positions (math ceiling {d}), cache {d:.3} GiB, free left {d:.3} GiB, global K+V {d} bytes a position\n", .{ @tagName(kv), cap, state.largestCapacity(budget, kv), gb(state.cacheBytes(cap, kv)), gb(left), 8 * 2 * kv.positionBytes() });
        cache.deinit();
    }

    for ([_]state.KvDtype{ .int8, .bf16 }) |kv| try snapshotCheck(gpa, &r, &schedule, kv);
    try transactionCheck(gpa, &r, &schedule);

    scratch.deinit();
    m.deinit();
    try stream.synchronize();
    // the driver keeps freed device allocations for reuse and trims them on its own clock: wait for it, up to 30 s
    var end_free = try freeBytes(&ctx);
    var waited: u32 = 0;
    while (end_free + tolerance < start_free and waited < 60) : (waited += 1) {
        _ = std.c.nanosleep(&.{ .sec = 0, .nsec = 500_000_000 }, null);
        end_free = try freeBytes(&ctx);
        if (waited % 4 == 3) std.debug.print("  free {d:.3} GiB after {d} ms\n", .{ gb(end_free), (waited + 1) * 500 });
    }
    std.debug.print("free memory settled after {d} ms\n", .{waited * 500});
    const drift = if (end_free > start_free) end_free - start_free else start_free - end_free;
    std.debug.print("free at end {d} bytes, at start {d} bytes, difference {d} bytes; runtime still counts {d} bytes allocated\n", .{ end_free, start_free, drift, xpu.rt.alloc_now });
    if (xpu.rt.alloc_now != 0 or drift > tolerance) return error.MemoryNotReturned;
    std.debug.print("inkling2 resident: ok\n", .{});
}

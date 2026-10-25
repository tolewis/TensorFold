//! EXL3 through the Level Zero host (xpu/exl3.zig): runs one case file from tools/xpu/exl3_zig_check.py and
//! writes the fp32 outputs. The Python side checks them against the numpy reference (gate 3, route check).
//! Usage: tf-xpu-exl3-test <case.bin> <out.bin> [reps]
//! With reps: the same call queued reps times back to back, then the wall time a call and the weight GB/s
//! (launch gaps included). Routed: every slot counts its expert's weight once.
//! Case: u32 magic "EXL3", mode (0 project, 1 routed), count (groups or experts), k, n, halves, rows (or tokens),
//! topk, x_per_slot, out_f32; then per group or expert trellis, suh, svh; then u32 ids [tokens][topk] (routed);
//! then bf16 x. Project groups each read their own x [rows][k].

const std = @import("std");
const xpu = @import("xpu");
const exl3 = xpu.exl3;

const Header = extern struct { magic: u32, mode: u32, count: u32, k: u32, n: u32, halves: u32, rows: u32, topk: u32, x_per_slot: u32, out_f32: u32 };

const Arena = struct {
    size: usize = 0,
    fn add(a: *Arena, n: usize) usize {
        const at = std.mem.alignForward(usize, a.size, 256);
        a.size = at + n;
        return at;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.Usage;
    const case = try std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .unlimited);
    defer gpa.free(case);
    const h = std.mem.bytesToValue(Header, case[0..@sizeOf(Header)]);
    if (h.magic != 0x334C5845) return error.BadCase;
    const routed = h.mode == 1;
    const tile_bytes: usize = 16 * h.halves;
    const tr_bytes = (h.k / 16) * (h.n / 16) * tile_bytes;
    const slots: usize = if (routed) h.rows * h.topk else h.rows;
    const x_rows: usize = if (routed) (if (h.x_per_slot != 0) slots else h.rows) else h.count * h.rows;
    const y_rows: usize = if (routed) slots else h.count * h.rows;
    const y_elem: usize = if (h.out_f32 != 0) 4 else 2;

    // one device arena: weights, ids, x, y, scratch, slot table
    var ar: Arena = .{};
    var at: usize = @sizeOf(Header);
    const weights = try gpa.alloc([3]usize, h.count);
    defer gpa.free(weights);
    const src = try gpa.alloc([3]usize, h.count);
    defer gpa.free(src);
    for (weights, src) |*w, *s| {
        for ([3]usize{ tr_bytes, h.k * 2, h.n * 2 }, 0..) |n, j| {
            w[j] = ar.add(n);
            s[j] = at;
            at += n;
        }
    }
    const ids_n: usize = if (routed) slots * 4 else 0;
    const ids_off = ar.add(ids_n);
    const ids_src = at;
    at += ids_n;
    const x_off = ar.add(x_rows * h.k * 2);
    const x_src = at;
    at += x_rows * h.k * 2;
    if (at != case.len) return error.BadCase;
    const y_off = ar.add(y_rows * h.n * y_elem);
    const scratch_off = ar.add(if (routed) exl3.scratchBytes(h.k, h.n, slots) else h.count * std.mem.alignForward(usize, exl3.scratchBytes(h.k, h.n, h.rows), 256));
    const slots_off = ar.add(exl3.slotTableBytes(slots));

    var drv = try xpu.Driver.open();
    defer drv.close();
    var ctx = try xpu.Context.init(&drv, null);
    defer ctx.deinit();
    var s = try xpu.Stream.init(&ctx);
    defer s.deinit();
    std.debug.print("{s}\n", .{ctx.name()});
    var p = try exl3.Pipelines.load(&ctx);
    defer p.deinit();
    var buf = try xpu.DeviceBuffer.alloc(&ctx, ar.size);
    defer buf.free();
    for (weights, src) |w, sr| for (w, sr, [3]usize{ tr_bytes, h.k * 2, h.n * 2 }) |d, o, n| try buf.uploadAsync(s, d, case[o..][0..n]);
    try buf.uploadAsync(s, ids_off, case[ids_src..][0..ids_n]);
    try buf.uploadAsync(s, x_off, case[x_src..][0 .. x_rows * h.k * 2]);
    try s.synchronize();
    const base: u64 = @intFromPtr(buf.ptr.?);
    const output: exl3.Output = if (h.out_f32 != 0) .f32 else .bf16;

    const ms = try gpa.alloc(exl3.Matrix, h.count);
    defer gpa.free(ms);
    for (ms, weights) |*m, w| m.* = .{ .trellis = base + w[0], .suh = base + w[1], .svh = base + w[2], .k = h.k, .n = h.n, .halves = h.halves };

    const reps: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 0;
    const weight_bytes: f64 = @floatFromInt((tr_bytes + h.k * 2 + h.n * 2) * (if (routed) slots else h.count));
    if (routed) {
        var experts = try exl3.Plan.experts(&ctx, s, gpa, ms);
        defer experts.deinit();
        const r: exl3.Route = .{ .ids = base + ids_off, .tokens = h.rows, .topk = h.topk, .x = base + x_off, .x_stride = h.k, .x_per_slot = h.x_per_slot != 0, .y = base + y_off, .y_stride = h.n, .output = output, .scratch = base + scratch_off, .slots = base + slots_off };
        try exl3.encodeRouted(&p, s, &experts, r);
        if (reps > 0) {
            try s.synchronize();
            const t0 = std.Io.Clock.awake.now(io);
            for (0..reps) |_| try exl3.encodeRouted(&p, s, &experts, r);
            try s.synchronize();
            report(io, t0, reps, weight_bytes);
        }
    } else {
        const groups = try gpa.alloc(exl3.Group, h.count);
        defer gpa.free(groups);
        const per = std.mem.alignForward(usize, exl3.scratchBytes(h.k, h.n, h.rows), 256);
        for (groups, ms, 0..) |*g, m, i| g.* = .{ .m = m, .rows = h.rows, .x = base + x_off + i * h.rows * h.k * 2, .x_stride = h.k, .y = base + y_off + i * h.rows * h.n * y_elem, .y_stride = h.n, .output = output, .scratch = base + scratch_off + i * per };
        var plan = try exl3.Plan.init(&ctx, s, gpa, groups);
        defer plan.deinit();
        try exl3.encode(&p, s, &plan);
        if (reps > 0) {
            try s.synchronize();
            const t0 = std.Io.Clock.awake.now(io);
            for (0..reps) |_| try exl3.encode(&p, s, &plan);
            try s.synchronize();
            report(io, t0, reps, weight_bytes);
        }
    }
    const y = try gpa.alloc(u8, y_rows * h.n * y_elem);
    defer gpa.free(y);
    try buf.download(s, y_off, y);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = y });
    return 0;
}

fn report(io: std.Io, t0: std.Io.Timestamp, reps: usize, weight_bytes: f64) void {
    const dt = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9 / @as(f64, @floatFromInt(reps));
    std.debug.print("call {d:.1} us, weights {d:.2} MB, {d:.1} GB/s\n", .{ dt * 1e6, weight_bytes / 1e6, weight_bytes / dt / 1e9 });
}

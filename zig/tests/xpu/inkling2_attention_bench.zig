//! tf-xpu-inkling2-attention-bench: decode and prompt attention at the served shapes, in time and bytes read a second.
//! Each case rotates through enough layers' caches that the K/V it reads is not in the card's cache from the last call.
const std = @import("std");
const xpu = @import("xpu");
const rig_ = @import("contract_rig.zig");
const attn = @import("inkling2_xpu_attention");

const plan = attn.planning;
const D = plan.head_dim;
const H = 24;
const KV = 8;
const EXTENT = plan.extent;
const reps = 40;
const roof_gbs = 605.7;

fn ms(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

const Case = struct { name: []const u8, rows: u32, k: u32, window: u32, prompt: bool = false };

fn fillBf16(rng: std.Random, v: []u16, spread: f32) void {
    for (v) |*x| x.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * spread)) >> 16);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    var dev: rig_.Dev = undefined;
    try dev.init(null);
    defer dev.deinit();
    var k = try attn.Kernels.init(dev.ink2_attention, dev.ink2_prompt);
    defer k.deinit();
    const cases = [_]Case{
        .{ .name = "sliding decode 1 row, 1,024 keys", .rows = 1, .k = 2049, .window = 1024 },
        .{ .name = "sliding verify 8 rows, 1,024 keys", .rows = 8, .k = 2056, .window = 1024 },
        .{ .name = "global decode 1 row, 8,192 keys", .rows = 1, .k = 8192, .window = 0 },
        .{ .name = "global decode 1 row, 32,768 keys", .rows = 1, .k = 32768, .window = 0 },
        .{ .name = "global decode 1 row, 100,000 keys", .rows = 1, .k = 100000, .window = 0 },
        .{ .name = "global prompt 256 rows over 8,192 keys", .rows = 256, .k = 8192, .window = 0, .prompt = true },
    };
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    std.debug.print("BENCH inkling2 attention on {s}, roof {d:.1} GB/s read\n", .{ "bmg-g31", roof_gbs });
    for (cases) |cs| {
        var rig: rig_.Rig = .{ .a = a, .dev = &dev };
        defer rig.deinit();
        const layer: plan.Layer = if (cs.window > 0) plan.Layer.sliding(cs.window, cs.rows) else plan.Layer.global(cs.k);
        const rows = cs.rows;
        const stride = @max(plan.slotsFor(layer, cs.k), plan.chunks(cs.k - 1, plan.prompt_chunk));
        const cache_n = @as(usize, KV) * layer.ring * D;
        // layers to rotate through: about 96 MiB of K and V at least, at most 8 sets
        const sets: usize = @min(8, @max(2, (96 << 20) / (cache_n * 4)));
        const host_k = try a.alloc(u16, cache_n);
        defer a.free(host_k);
        fillBf16(rng, host_k, 0.6);
        const host_v = try a.alloc(u16, cache_n);
        defer a.free(host_v);
        fillBf16(rng, host_v, 2);
        var kv: [8]attn.Cache = undefined;
        for (kv[0..sets]) |*c| c.* = .{ .k = .{ .ptr = try rig.input(std.mem.sliceAsBytes(host_k)) }, .v = .{ .ptr = try rig.input(std.mem.sliceAsBytes(host_v)) } };
        const q = try a.alloc(u16, rows * H * D);
        defer a.free(q);
        fillBf16(rng, q, 0.6);
        const tab = try a.alloc(u16, rows * H * EXTENT);
        defer a.free(tab);
        fillBf16(rng, tab, 0.5);
        const sink = try a.alloc(u16, rows * H);
        defer a.free(sink);
        @memset(sink, 0x4040);
        const gate = try a.alloc(u16, rows * H * D);
        defer a.free(gate);
        fillBf16(rng, gate, 2);
        const pos = try a.alloc(i32, rows);
        defer a.free(pos);
        for (pos, 0..) |*v, i| v.* = @intCast(cs.k - rows + i);
        const seg_of = try a.alloc(i32, rows);
        defer a.free(seg_of);
        @memset(seg_of, 0);
        const seg = [1]plan.Seg{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(cs.k - rows), .off = 0 }};
        const meta: attn.RowMeta = .{
            .pos = .{ .ptr = try rig.input(std.mem.sliceAsBytes(pos)) },
            .seg_of = .{ .ptr = try rig.input(std.mem.sliceAsBytes(seg_of)) },
            .segs = .{ .ptr = try rig.input(std.mem.sliceAsBytes(&seg)) },
        };
        const qd: attn.Ref = .{ .ptr = try rig.input(std.mem.sliceAsBytes(q)) };
        const tabd: attn.Ref = .{ .ptr = try rig.input(std.mem.sliceAsBytes(tab)) };
        const sinkd: attn.Ref = .{ .ptr = try rig.input(std.mem.sliceAsBytes(sink)) };
        const gated: attn.Ref = .{ .ptr = try rig.input(std.mem.sliceAsBytes(gate)) };
        const po: attn.Ref = .{ .ptr = try rig.scratch(@as(usize, rows) * H * stride * (D * 4)) };
        const pml: attn.Ref = .{ .ptr = try rig.scratch(@as(usize, rows) * H * stride * 8) };
        const out: attn.Ref = .{ .ptr = try rig.output(@as(usize, rows) * H * D * 2) };
        const r: plan.Rows = .{ .rows = rows, .last_pos = cs.k - 1 };
        const path: plan.Path = if (cs.prompt) .prompt else .decode;
        var t_decode: i96 = 0;
        var t_enqueue: i96 = 0;
        for (0..2) |pass| {
            const t0 = ms(init.io);
            for (0..reps) |i| try attn.attendAs(&k, dev.s, path, r, layer, stride, meta, qd, kv[i % sets], tabd, po, pml, sinkd, gated, out);
            const t1 = ms(init.io);
            try dev.s.synchronize();
            if (pass == 1) {
                t_decode = ms(init.io) - t0;
                t_enqueue = t1 - t0;
            }
        }
        // the same two launches timed apart: the attention kernel alone, then the merge alone
        const launch = try plan.attendAs(r, layer, stride, path);
        const sc: f32 = 1.0 / @sqrt(@as(f32, D));
        var t_part: [2]i96 = .{ 0, 0 };
        for (0..2) |which| {
            for (0..2) |pass| {
                const t0 = ms(init.io);
                for (0..reps) |i| {
                    const c = kv[i % sets];
                    if (which == 0 and !cs.prompt) {
                        try xpu.launch.dispatch(k.decode, dev.s, launch.grid, .{ qd.at(), c.k.at(), c.v.at(), tabd.at(), meta.segs.at(), po.at(), pml.at(), meta.pos.at(), meta.seg_of.at(), @as(u32, H), @as(u32, KV), layer.ring, layer.window, launch.chunk, @as(u32, 0), launch.partials_rh, sc });
                    } else if (which == 0) {
                        try xpu.launch.dispatch(k.prompt, dev.s, launch.grid, .{ qd.at(), c.k.at(), c.v.at(), tabd.at(), meta.segs.at(), po.at(), pml.at(), @as(u32, H), @as(u32, KV), layer.ring, launch.chunk, @as(u32, 0), launch.partials_rh, sc });
                    } else {
                        try xpu.launch.dispatch(k.merge, dev.s, launch.merge_grid, .{ po.at(), pml.at(), sinkd.at(), gated.at(), meta.pos.at(), out.at(), @as(u32, H), layer.window, launch.chunk, @as(u32, 0), launch.partials_rh, @as(u32, 1) });
                    }
                }
                try dev.s.synchronize();
                if (pass == 1) t_part[which] = ms(init.io) - t0;
            }
        }
        std.debug.print("BENCH   parts: attention kernel {d:.1} us, merge {d:.1} us\n", .{ @as(f64, @floatFromInt(t_part[0])) / reps / 1000, @as(f64, @floatFromInt(t_part[1])) / reps / 1000 });
        const per_ns = @as(f64, @floatFromInt(t_decode)) / reps;
        const lo: usize = if (cs.window > 0) cs.k - rows + 1 -| cs.window else 0;
        const keys: usize = cs.k - lo;
        const bytes = @as(f64, @floatFromInt(keys)) * KV * D * 2 * 2;
        const gbs = bytes / per_ns;
        if (cs.prompt) {
            // rows share every K/V byte, so the rate that matters is the matrix work: 4 * D flops a (row, head, key)
            const pairs = @as(f64, @floatFromInt(rows)) * H;
            const flops = pairs * (@as(f64, @floatFromInt(keys)) - @as(f64, @floatFromInt(rows)) / 2) * 4 * D;
            std.debug.print("BENCH {s}: {d:.1} us a call (attention kernel and merge), {d:.2} TFLOP/s\n", .{ cs.name, per_ns / 1000, flops / per_ns / 1000 });
            continue;
        }
        std.debug.print("BENCH {s}: {d:.1} us a call (attention kernel and merge; host enqueue {d:.1} us), {d:.1} MB of K and V, {d:.1} GB/s = {d:.2} of {d:.1}\n", .{ cs.name, per_ns / 1000, @as(f64, @floatFromInt(t_enqueue)) / reps / 1000, bytes / 1e6, gbs, gbs / roof_gbs, roof_gbs });
    }
    return 0;
}

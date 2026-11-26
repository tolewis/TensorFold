//! tf-xpu-inkling2-attention-bench: decode and prompt attention at the served shapes, in time and bytes read a second.
//! Each case rotates through enough layers' caches that the K/V it reads is not in the card's cache from the last call.
//! `--kv int8` fills an int8 cache from the same random bf16 data with the prep kernel's scale rule; `--kv bf16` is the
//! default. `--splits N` sets the decode chunk the server uses (`flash_splits`, 64 live), zero keeps the layer chunk.
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

fn fillBf16(rng: std.Random, v: []u16, spread: f32) void {
    for (v) |*x| x.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * spread)) >> 16);
}

fn bf2f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// One (position, head) row quantized as `tf_ink2_prep8` does: the largest magnitude of the bf16 value, one fp32 scale
/// (`amax / 127`), then one code a value. Returns the scale.
fn quantize(row: []const u16, codes: []u8) f32 {
    var amax: f32 = 0;
    for (row) |h| amax = @max(amax, @abs(bf2f(h)));
    const scale: f32 = if (amax > 0) amax / 127.0 else 0;
    for (row, codes) |h, *c| {
        const t = if (scale > 0) @round(bf2f(h) / scale) else 0;
        c.* = @intCast(@as(i8, @intFromFloat(std.math.clamp(t, -127.0, 127.0))));
    }
    return scale;
}

const Case = struct { name: []const u8, rows: u32, k: u32, window: u32, prompt: bool = false };

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var kv_int8 = false;
    var splits: u32 = 0;
    var ai: usize = 1;
    while (ai < args.len) : (ai += 1) {
        const arg = args[ai];
        if (std.mem.eql(u8, arg, "--kv")) {
            ai += 1;
            if (std.mem.eql(u8, args[ai], "int8")) {
                kv_int8 = true;
            } else if (std.mem.eql(u8, args[ai], "bf16")) {
                kv_int8 = false;
            } else return error.BadKv;
        } else if (std.mem.eql(u8, arg, "--splits")) {
            ai += 1;
            splits = try std.fmt.parseInt(u32, args[ai], 10);
        } else return error.BadArgument;
    }
    plan.flash_splits = splits;
    plan.flash_user_set = splits != 0;
    var dev: rig_.Dev = undefined;
    try dev.init(null);
    defer dev.deinit();
    var k = try attn.Kernels.init(dev.ink2_attention, dev.ink2_prompt);
    defer k.deinit();
    if (kv_int8) {
        const p8 = try dev.properties(k.prep8);
        const d8 = try dev.properties(k.decode8);
        std.debug.print("BENCH kernel properties: tf_ink2_prep8 {d} spill bytes, {d} private bytes; tf_ink2_attn_decode8 {d} spill bytes, {d} private bytes\n", .{ p8.spillMemSize, p8.privateMemSize, d8.spillMemSize, d8.privateMemSize });
    }
    const cases = [_]Case{
        .{ .name = "sliding decode 1 row, 1,024 keys", .rows = 1, .k = 2049, .window = 1024 },
        .{ .name = "sliding verify 8 rows, 1,024 keys", .rows = 8, .k = 2056, .window = 1024 },
        .{ .name = "global decode 1 row, 8,192 keys", .rows = 1, .k = 8192, .window = 0 },
        .{ .name = "global decode 1 row, 32,768 keys", .rows = 1, .k = 32768, .window = 0 },
        .{ .name = "global decode 1 row, 40,000 keys", .rows = 1, .k = 40000, .window = 0 },
        .{ .name = "global decode 1 row, 100,000 keys", .rows = 1, .k = 100000, .window = 0 },
        .{ .name = "global decode 1 row, 125,000 keys", .rows = 1, .k = 125000, .window = 0 },
        .{ .name = "global prompt 256 rows over 8,192 keys", .rows = 256, .k = 8192, .window = 0, .prompt = true },
    };
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    std.debug.print("BENCH inkling2 attention on {s}, kv {s}, splits {d}, roof {d:.1} GB/s read\n", .{ "bmg-g31", if (kv_int8) "int8" else "bf16", splits, roof_gbs });
    for (cases) |cs| {
        // the prompt kernel reads bf16 rows only, so an int8 run keeps the decode cases
        if (kv_int8 and cs.prompt) continue;
        var rig: rig_.Rig = .{ .a = a, .dev = &dev };
        defer rig.deinit();
        const layer: plan.Layer = if (cs.window > 0) plan.Layer.sliding(cs.window, cs.rows) else plan.Layer.global(cs.k);
        const rows = cs.rows;
        const ring: usize = layer.ring;
        const stride = @max(plan.slotsFor(layer, cs.k), plan.chunks(cs.k - 1, plan.prompt_chunk));
        const cache_n = @as(usize, KV) * ring * D;
        const host_k = try a.alloc(u16, cache_n);
        defer a.free(host_k);
        fillBf16(rng, host_k, 0.6);
        const host_v = try a.alloc(u16, cache_n);
        defer a.free(host_v);
        fillBf16(rng, host_v, 2);
        // layers to rotate through: about 96 MiB of K and V at least, at most 8 sets
        const set_bytes: usize = if (kv_int8) 2 * cache_n + 2 * KV * ring * 4 else 2 * cache_n * 2;
        const sets: usize = @min(8, @max(2, (96 << 20) / set_bytes));
        var kv: [8]attn.Cache = undefined;
        if (kv_int8) {
            const kc = try a.alloc(u8, cache_n);
            defer a.free(kc);
            const vc = try a.alloc(u8, cache_n);
            defer a.free(vc);
            const ks = try a.alloc(f32, KV * ring);
            defer a.free(ks);
            const vs = try a.alloc(f32, KV * ring);
            defer a.free(vs);
            var codes: [D]u8 = undefined;
            for (0..KV) |h| {
                for (0..ring) |p| {
                    const at = (h * ring + p) * D;
                    ks[h * ring + p] = quantize(host_k[at..][0..D], &codes);
                    for (0..D) |d| kc[plan.kCodeOffset(h, p, d, ring)] = codes[d];
                    vs[h * ring + p] = quantize(host_v[at..][0..D], &codes);
                    for (0..D) |d| vc[plan.vCodeOffset(h, p, d, ring)] = codes[d];
                }
            }
            const ksd: attn.Ref = .{ .ptr = try rig.input(std.mem.sliceAsBytes(ks)) };
            const vsd: attn.Ref = .{ .ptr = try rig.input(std.mem.sliceAsBytes(vs)) };
            for (kv[0..sets]) |*c| c.* = .{
                .k = .{ .ptr = try rig.input(std.mem.sliceAsBytes(kc)) },
                .v = .{ .ptr = try rig.input(std.mem.sliceAsBytes(vc)) },
                .int8 = true,
                .kscale = ksd,
                .vscale = vsd,
            };
        } else {
            for (kv[0..sets]) |*c| c.* = .{ .k = .{ .ptr = try rig.input(std.mem.sliceAsBytes(host_k)) }, .v = .{ .ptr = try rig.input(std.mem.sliceAsBytes(host_v)) } };
        }
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
                        if (kv_int8) {
                            try xpu.launch.dispatch(k.decode8, dev.s, launch.grid, .{
                                qd.at(),          c.k.at(),      c.kscale.at(), c.v.at(),         c.vscale.at(),
                                tabd.at(),        meta.segs.at(), po.at(),       pml.at(),        meta.pos.at(),
                                meta.seg_of.at(), @as(u32, H),  @as(u32, KV),  layer.ring,      layer.window,
                                launch.chunk,     @as(u32, 0),   launch.partials_rh, sc,
                            });
                        } else {
                            try xpu.launch.dispatch(k.decode, dev.s, launch.grid, .{ qd.at(), c.k.at(), c.v.at(), tabd.at(), meta.segs.at(), po.at(), pml.at(), meta.pos.at(), meta.seg_of.at(), @as(u32, H), @as(u32, KV), layer.ring, layer.window, launch.chunk, @as(u32, 0), launch.partials_rh, sc });
                        }
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
        const per_ns = @as(f64, @floatFromInt(t_decode)) / reps;
        const att_ns = @as(f64, @floatFromInt(t_part[0])) / reps;
        std.debug.print("BENCH   parts: attention kernel {d:.1} us, merge {d:.1} us\n", .{ att_ns / 1000, @as(f64, @floatFromInt(t_part[1])) / reps / 1000 });
        const lo: usize = if (cs.window > 0) cs.k - rows + 1 -| cs.window else 0;
        const keys: usize = cs.k - lo;
        // the bytes the attention kernel reads: bf16 values of K and V, or int8 codes and their scales
        const bytes = if (kv_int8)
            @as(f64, @floatFromInt(keys)) * KV * (D + 4) * 2
        else
            @as(f64, @floatFromInt(keys)) * KV * D * 2 * 2;
        const gbs = bytes / att_ns;
        if (cs.prompt) {
            // rows share every K/V byte, so the rate that matters is the matrix work: 4 * D flops a (row, head, key)
            const pairs = @as(f64, @floatFromInt(rows)) * H;
            const flops = pairs * (@as(f64, @floatFromInt(keys)) - @as(f64, @floatFromInt(rows)) / 2) * 4 * D;
            std.debug.print("BENCH {s}: {d:.1} us a call (attention kernel and merge), {d:.2} TFLOP/s\n", .{ cs.name, per_ns / 1000, flops / per_ns / 1000 });
            continue;
        }
        std.debug.print("BENCH {s}: attention kernel {d:.1} us, merge {d:.1} us, both {d:.1} us a call (host enqueue {d:.1} us), {d:.1} MB of K and V, {d:.1} GB/s = {d:.2} of {d:.1}\n", .{ cs.name, att_ns / 1000, @as(f64, @floatFromInt(t_part[1])) / reps / 1000, per_ns / 1000, @as(f64, @floatFromInt(t_enqueue)) / reps / 1000, bytes / 1e6, gbs, gbs / roof_gbs, roof_gbs });
    }
    return 0;
}

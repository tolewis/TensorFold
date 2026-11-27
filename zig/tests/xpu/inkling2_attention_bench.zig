//! tf-xpu-inkling2-attention-bench: decode and prompt attention at the served shapes, in time and bytes read a second.
//! Each case rotates through enough layers' caches that the K/V it reads is not in the card's cache from the last call.
//! `--kv int8` fills an int8 cache from the same random bf16 data with the prep kernel's scale rule; `--kv bf16` is the
//! default. `--splits N` sets the decode chunk the server uses (`flash_splits`, 64 live), zero keeps the layer chunk.
//! `--check` times nothing: it fills int8 codes with power-of-two scales and a bf16 cache that holds each code times its
//! scale exactly, then requires `tf_ink2_attn_decode8` to give the partials and the output of `tf_ink2_attn_decode` bit
//! for bit, at the layer chunk and at 128 splits. An int8 prompt case reads the bf16 copy (`tf_ink2_kv_dequant8`, then
//! `tf_ink2_attn_prompt`, the prefill's global visit) and must give the bf16 cache's bits. Exit 1 on a difference.
//! `--check-real` uses the prep rule's own scales (`amax / 127`, rows of very different size) and a host reference for
//! each int8 kernel: `tf_ink2_prep8` (codes and scales), `tf_ink2_prep8q` (the bf16 stage) and
//! `tf_ink2_kv_dequant8` must equal the bf16 prep followed by the host quantize rule bit for bit, and the prep8q stage
//! must equal dequant8 of the prep8 ring. `tf_ink2_attn_decode8` and the int8 prompt read go against an fp64 host
//! partial (top, weight sum, weighted mean of V) on the exact values `code * scale`, beside the bf16 kernels on
//! `bf16(code * scale)`. The int8 prompt read must give the bits of the bf16 kernel on `bf16(code * scale)` (its design).
//! Exit 1 on a difference or a decode8 error above the bounds of `real_bounds`.
const std = @import("std");
const xpu = @import("xpu");
const rig_ = @import("contract_rig.zig");
const ze = @import("ze");
const attn = @import("inkling2_xpu_attention");

const plan = attn.planning;
const D = plan.head_dim;
const H = 24;
const KV = 8;
const EXTENT = plan.extent;
const reps = 40;
/// `--sync-each`: one launch and one wait a repeat. Without it the repeats queue on one stream with one wait at the end.
var sync_each = false;
fn method() []const u8 {
    return if (sync_each) "method: one launch and one wait, mean of 40" else "method: 40 queued launches, one wait, total over 40";
}
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

/// fp32 to bf16, round to nearest even (the device `intel_convert_bfloat16_as_ushort`, finite values).
fn f2bf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Round to nearest, ties to even (the device `rint`).
fn rintEven(t: f32) f32 {
    if (@abs(t - @trunc(t)) == 0.5) return 2.0 * @round(t / 2.0);
    return @round(t);
}

/// One (position, head) row quantized as `tf_ink2_prep8` does: the largest magnitude of the bf16 value, one fp32 scale
/// (`amax / 127`), then one code a value. Returns the scale.
fn quantize(row: []const u16, codes: []u8) f32 {
    var amax: f32 = 0;
    for (row) |h| amax = @max(amax, @abs(bf2f(h)));
    const scale: f32 = if (amax > 0) amax / 127.0 else 0;
    for (row, codes) |h, *c| {
        const t = if (scale > 0) rintEven(bf2f(h) / scale) else 0;
        c.* = @intCast(@as(i8, @intFromFloat(std.math.clamp(t, -127.0, 127.0))));
    }
    return scale;
}

const Case = struct { name: []const u8, rows: u32, k: u32, window: u32, prompt: bool = false };

/// The int8 read against the bf16 read on the same values. A code times a power-of-two scale is exact in bf16, and a
/// power-of-two factor commutes with every fp32 rounding, so the two kernels must agree bit for bit wherever their key
/// blocks start at the same key: global layers, and a sliding row whose window start is a multiple of 16.
fn check(a: std.mem.Allocator, dev: *rig_.Dev, k: *const attn.Kernels, rng: std.Random) !usize {
    const cases = [_]Case{
        .{ .name = "global 1 row, 3,000 keys", .rows = 1, .k = 3000, .window = 0 },
        .{ .name = "global 8 rows, 3,000 keys", .rows = 8, .k = 3000, .window = 0 },
        .{ .name = "global 1 row, 40,000 keys", .rows = 1, .k = 40000, .window = 0 },
        .{ .name = "sliding 1 row, 2,048 keys, window start 1,024", .rows = 1, .k = 2048, .window = 1024 },
        .{ .name = "global prompt 100 rows, 3,000 keys", .rows = 100, .k = 3000, .window = 0, .prompt = true },
        .{ .name = "global prompt 256 rows, 2,056 keys", .rows = 256, .k = 2056, .window = 0, .prompt = true },
    };
    var failed: usize = 0;
    for ([_]u32{ 0, 128 }) |splits| {
        plan.flash_splits = splits;
        plan.flash_user_set = splits != 0;
        for (cases) |cs| {
            var rig: rig_.Rig = .{ .a = a, .dev = dev };
            defer rig.deinit();
            const layer: plan.Layer = if (cs.window > 0) plan.Layer.sliding(cs.window, cs.rows) else plan.Layer.global(cs.k);
            const rows = cs.rows;
            const ring: usize = layer.ring;
            const stride = @max(plan.slotsFor(layer, cs.k), plan.chunks(cs.k - 1, plan.prompt_chunk));
            const cache_n = @as(usize, KV) * ring * D;
            const kb = try a.alloc(u16, cache_n);
            defer a.free(kb);
            const vb = try a.alloc(u16, cache_n);
            defer a.free(vb);
            const kc = try a.alloc(u8, cache_n);
            defer a.free(kc);
            const vc = try a.alloc(u8, cache_n);
            defer a.free(vc);
            const ks = try a.alloc(f32, KV * ring);
            defer a.free(ks);
            const vs = try a.alloc(f32, KV * ring);
            defer a.free(vs);
            for (0..KV) |h| for (0..ring) |p| {
                const ksc = std.math.ldexp(@as(f32, 1), -6 - @as(i32, rng.intRangeAtMost(i32, 0, 2)));
                const vsc = std.math.ldexp(@as(f32, 1), -5 - @as(i32, rng.intRangeAtMost(i32, 0, 2)));
                ks[h * ring + p] = ksc;
                vs[h * ring + p] = vsc;
                for (0..D) |d| {
                    const ck = rng.intRangeAtMost(i8, -127, 127);
                    const cv = rng.intRangeAtMost(i8, -127, 127);
                    kc[plan.kCodeOffset(h, p, d, ring)] = @bitCast(ck);
                    vc[plan.vCodeOffset(h, p, d, ring)] = @bitCast(cv);
                    const at = (h * ring + p) * D + d;
                    kb[at] = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(ck)) * ksc)) >> 16);
                    vb[at] = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(cv)) * vsc)) >> 16);
                }
            };
            const bf: attn.Cache = .{ .k = .{ .ptr = try rig.input(std.mem.sliceAsBytes(kb)) }, .v = .{ .ptr = try rig.input(std.mem.sliceAsBytes(vb)) } };
            const i8c: attn.Cache = .{
                .k = .{ .ptr = try rig.input(kc) },
                .v = .{ .ptr = try rig.input(vc) },
                .int8 = true,
                .kscale = .{ .ptr = try rig.input(std.mem.sliceAsBytes(ks)) },
                .vscale = .{ .ptr = try rig.input(std.mem.sliceAsBytes(vs)) },
            };
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
            const po_n = @as(usize, rows) * H * stride * (D * 4);
            const pml_n = @as(usize, rows) * H * stride * 8;
            const out_n = @as(usize, rows) * H * D * 2;
            const po: attn.Ref = .{ .ptr = try rig.scratch(po_n) };
            const pml: attn.Ref = .{ .ptr = try rig.scratch(pml_n) };
            const out: attn.Ref = .{ .ptr = try rig.output(out_n) };
            const r: plan.Rows = .{ .rows = rows, .last_pos = cs.k - 1 };
            const path: plan.Path = if (cs.prompt) .prompt else .decode;
            const zeros = try a.alloc(u8, po_n);
            defer a.free(zeros);
            @memset(zeros, 0);
            var got: [2][3][]u8 = undefined;
            const copy = try promptCopy(&rig, ring);
            for ([_]attn.Cache{ bf, i8c }, 0..) |c, w| {
                // a slot no chunk writes keeps its bytes from the first kernel: clear both before each
                try rig.put(po.ptr, zeros);
                try rig.put(pml.ptr, zeros[0..pml_n]);
                try attn.attendAs(k, dev.s, path, r, layer, stride, meta, qd, try readAs(k, dev, c, copy, cs, layer), tabd, po, pml, sinkd, gated, out);
                got[w][0] = try a.dupe(u8, try rig.bytes(po.ptr, po_n));
                got[w][1] = try a.dupe(u8, try rig.bytes(pml.ptr, pml_n));
                got[w][2] = try a.dupe(u8, try rig.bytes(out.ptr, out_n));
            }
            defer for (got) |g| for (g) |b| a.free(b);
            var diff: [3]usize = .{ 0, 0, 0 };
            for (0..3) |j| {
                const x = std.mem.bytesAsSlice(u32, got[0][j][0 .. got[0][j].len / 4 * 4]);
                const y = std.mem.bytesAsSlice(u32, got[1][j][0 .. got[1][j].len / 4 * 4]);
                for (x, y) |u, v| diff[j] += @intFromBool(u != v);
            }
            const launch = try plan.attendAs(r, layer, stride, path);
            std.debug.print("CHECK splits {d} chunk {d}, {s}: int8 against bf16 on equal values, words that differ: partials {d} of {d}, max and sum {d} of {d}, output {d} of {d}\n", .{ splits, launch.chunk, cs.name, diff[0], po_n / 4, diff[1], pml_n / 4, diff[2], out_n / 4 });
            if (diff[0] + diff[1] + diff[2] != 0) failed += 1;
        }
    }
    std.debug.print("CHECK {s}: {d} of {d} cases differ\n", .{ if (failed == 0) "PASS" else "FAIL", failed, 2 * cases.len });
    return failed;
}

/// The bounds of `--check-real` for the int8 reads against the fp64 partial: the top score (log2 units), the relative
/// weight sum and the weighted mean of V over the RMS of V. A wrong scale moves a key by a factor of up to 64 here.
const real_bounds = struct {
    const top: f64 = 1e-3;
    const sum: f64 = 1e-2;
    const mean: f64 = 2e-2;
};

const RealErr = struct {
    top: f64 = 0,
    sum: f64 = 0,
    mean: f64 = 0,
    fn ok(e: RealErr) bool {
        return e.top <= real_bounds.top and e.sum <= real_bounds.sum and e.mean <= real_bounds.mean;
    }
};

/// The partials of one launch against the fp64 reference on `kref`/`vref` (`[head][ring][D]`, exact `code * scale`).
fn realErrors(cs: Case, layer: plan.Layer, chunk: u32, stride: usize, q: []const u16, tab: []const u16, pos: []const i32, kref: []const f64, vref: []const f64, vrms: f64, po: []const f32, pml: []const [2]f32) RealErr {
    var e: RealErr = .{};
    const ring: usize = layer.ring;
    const scale: f64 = 1.0 / @sqrt(@as(f64, D));
    const log2e: f64 = 1.4426950408889634;
    var x = std.heap.page_allocator.alloc(f64, chunk) catch unreachable;
    defer std.heap.page_allocator.free(x);
    for (0..cs.rows) |row| {
        const p: usize = @intCast(pos[row]);
        const lo: usize = if (layer.window > 0 and p + 1 > layer.window) p + 1 - layer.window else 0;
        const first: usize = if (layer.window > 0) lo / chunk else 0;
        for (0..H) |h| {
            const kvh = h / 3;
            const rh = row * H + h;
            for (0..stride) |j| {
                const c = first + j;
                const from = @max(lo, c * chunk);
                const to = @min(p + 1, (c + 1) * chunk);
                if (from >= to) continue;
                var top: f64 = -std.math.inf(f64);
                for (from..to) |kp| {
                    var dot: f64 = 0;
                    const kr = kref[(kvh * ring + kp % ring) * D ..][0..D];
                    for (0..D) |d| dot += @as(f64, bf2f(q[rh * D + d])) * kr[d];
                    const dist = p - kp;
                    const bias: f64 = if (dist < EXTENT) bf2f(tab[rh * EXTENT + dist]) else 0;
                    x[kp - from] = (dot * scale + bias) * log2e;
                    top = @max(top, x[kp - from]);
                }
                var sum: f64 = 0;
                var mean: [D]f64 = @splat(0);
                for (from..to) |kp| {
                    const w = std.math.pow(f64, 2, x[kp - from] - top);
                    sum += w;
                    const vr = vref[(kvh * ring + kp % ring) * D ..][0..D];
                    for (0..D) |d| mean[d] += w * vr[d];
                }
                const slot = rh * stride + j;
                const ml = pml[slot];
                e.top = @max(e.top, @abs(@as(f64, ml[0]) - top));
                e.sum = @max(e.sum, @abs(@as(f64, ml[1]) / sum - 1));
                for (0..D) |d| e.mean = @max(e.mean, @abs(@as(f64, po[slot * D + d]) / @as(f64, ml[1]) - mean[d] / sum) / vrms);
            }
        }
    }
    return e;
}

/// B6: the groups of one kernel that the device holds at one time (`zeKernelSuggestMaxCooperativeGroupCount` at the
/// kernel's group size of 16 x 1 x 1), and that count over the Xe cores of the device properties.
fn occupancy(dev: *rig_.Dev, kern: xpu.Kernel) struct { groups: u32, cores: u32, rc: u32 } {
    var lib = std.DynLib.open("libze_loader.so.1") catch return .{ .groups = 0, .cores = 0, .rc = 1 };
    defer lib.close();
    const f = lib.lookup(*const fn (ze.H, *u32) callconv(.c) u32, "zeKernelSuggestMaxCooperativeGroupCount") orelse return .{ .groups = 0, .cores = 0, .rc = 2 };
    _ = dev.ctx.d.api.zeKernelSetGroupSize(kern.handle, 16, 1, 1);
    var n: u32 = 0;
    const rc = f(kern.handle, &n);
    const pr = dev.ctx.gpu.props;
    return .{ .groups = n, .cores = pr.numSlices * pr.numSubslicesPerSlice, .rc = rc };
}

fn checkReal(a: std.mem.Allocator, dev: *rig_.Dev, k: *const attn.Kernels, rng: std.Random) !usize {
    var failed: usize = 0;
    var cases_run: usize = 0;
    // 1. the three prep kernels against the bf16 prep and the host quantize rule
    for ([_]plan.Layer{ plan.Layer.sliding(1024, 64), plan.Layer.global(4096) }) |layer| {
        var rig: rig_.Rig = .{ .a = a, .dev = dev };
        defer rig.deinit();
        const rows: u32 = 64;
        const ring: usize = layer.ring;
        const stage_ring: u32 = 1104;
        const p0: u32 = if (layer.window > 0) 3000 else 4000;
        const raw_q = try a.alloc(u16, rows * H * D);
        defer a.free(raw_q);
        const raw_k = try a.alloc(u16, rows * KV * D);
        defer a.free(raw_k);
        const raw_v = try a.alloc(u16, rows * KV * D);
        defer a.free(raw_v);
        const raw_rel = try a.alloc(u16, rows * H * 16);
        defer a.free(raw_rel);
        const q_scale = try a.alloc(u16, H * D);
        defer a.free(q_scale);
        const rel_scale = try a.alloc(u16, H * 16);
        defer a.free(rel_scale);
        fillBf16(rng, raw_q, 4);
        fillBf16(rng, raw_rel, 4);
        for (q_scale) |*v| v.* = f2bf(0.5 + rng.float(f32));
        for (rel_scale) |*v| v.* = f2bf(0.5 + rng.float(f32));
        // rows of very different size, so that the scales of two positions differ
        for (0..rows * KV) |r| {
            const f = std.math.pow(f32, 2, rng.float(f32) * 12 - 6);
            for (0..D) |d| {
                raw_k[r * D + d] = f2bf((rng.float(f32) - 0.5) * f);
                raw_v[r * D + d] = f2bf((rng.float(f32) - 0.5) * f);
            }
        }
        const pos = try a.alloc(i32, rows);
        defer a.free(pos);
        for (pos, 0..) |*v, i| v.* = @intCast(p0 + i);
        const seg_of = try a.alloc(i32, rows);
        defer a.free(seg_of);
        @memset(seg_of, 0);
        const seg = [1]plan.Seg{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p0), .off = 0 }};
        const meta: attn.RowMeta = .{
            .pos = .{ .ptr = try rig.input(std.mem.sliceAsBytes(pos)) },
            .seg_of = .{ .ptr = try rig.input(std.mem.sliceAsBytes(seg_of)) },
            .segs = .{ .ptr = try rig.input(std.mem.sliceAsBytes(&seg)) },
        };
        const raw = [4]attn.Ref{
            .{ .ptr = try rig.input(std.mem.sliceAsBytes(raw_q)) },  .{ .ptr = try rig.input(std.mem.sliceAsBytes(raw_k)) },
            .{ .ptr = try rig.input(std.mem.sliceAsBytes(raw_v)) },  .{ .ptr = try rig.input(std.mem.sliceAsBytes(raw_rel)) },
        };
        const scales = [2]attn.Ref{ .{ .ptr = try rig.input(std.mem.sliceAsBytes(q_scale)) }, .{ .ptr = try rig.input(std.mem.sliceAsBytes(rel_scale)) } };
        const qn = rows * H * D * 2;
        const reln = rows * H * 16 * 2;
        const bfn = KV * ring * D * 2;
        const stn = KV * @as(usize, stage_ring) * D * 2;
        const q_out = [3]attn.Ref{ .{ .ptr = try rig.output(qn) }, .{ .ptr = try rig.output(qn) }, .{ .ptr = try rig.output(qn) } };
        const rel_out = [3]attn.Ref{ .{ .ptr = try rig.output(reln) }, .{ .ptr = try rig.output(reln) }, .{ .ptr = try rig.output(reln) } };
        const bfc: attn.Cache = .{ .k = .{ .ptr = try rig.output(bfn) }, .v = .{ .ptr = try rig.output(bfn) } };
        const i8c: attn.Cache = .{ .k = .{ .ptr = try rig.output(KV * ring * D) }, .v = .{ .ptr = try rig.output(KV * ring * D) }, .int8 = true, .kscale = .{ .ptr = try rig.output(KV * ring * 4) }, .vscale = .{ .ptr = try rig.output(KV * ring * 4) } };
        const stq: attn.Cache = .{ .k = .{ .ptr = try rig.output(stn) }, .v = .{ .ptr = try rig.output(stn) } };
        const std8: attn.Cache = .{ .k = .{ .ptr = try rig.output(stn) }, .v = .{ .ptr = try rig.output(stn) } };
        // prep8q's own ring write: the rows from `ring_from` on, which must equal the prep8 ring
        const i8r: attn.Cache = .{ .k = .{ .ptr = try rig.output(KV * ring * D) }, .v = .{ .ptr = try rig.output(KV * ring * D) }, .int8 = true, .kscale = .{ .ptr = try rig.output(KV * ring * 4) }, .vscale = .{ .ptr = try rig.output(KV * ring * 4) } };
        const ring_from: u32 = p0 + rows / 2;
        const r: plan.Rows = .{ .rows = rows, .last_pos = p0 + rows - 1 };
        const c: plan.Prep = .{};
        var stage_layer = layer;
        stage_layer.ring = stage_ring;
        try attn.prep(k, dev.s, r, layer, c, meta, raw, scales, q_out[0], rel_out[0], bfc);
        try attn.prep(k, dev.s, r, layer, c, meta, raw, scales, q_out[1], rel_out[1], i8c);
        try attn.prepStage8(k, dev.s, r, stage_layer, c, meta, raw, scales, q_out[2], rel_out[2], stq, i8r, @intCast(ring), ring_from);
        try attn.dequant8(k, dev.s, i8c, std8, p0, @intCast(ring), stage_ring, rows, KV);
        var qx: [3][]u8 = undefined;
        for (0..3) |w| qx[w] = try a.dupe(u8, try rig.bytes(q_out[w].ptr, qn));
        defer for (qx) |b| a.free(b);
        var rx: [3][]u8 = undefined;
        for (0..3) |w| rx[w] = try a.dupe(u8, try rig.bytes(rel_out[w].ptr, reln));
        defer for (rx) |b| a.free(b);
        var bfk: [2][]u16 = undefined; // the bf16 prep's K and V
        for (0..2) |w| bfk[w] = try a.dupe(u16, @alignCast(std.mem.bytesAsSlice(u16, try rig.bytes(if (w == 0) bfc.k.ptr else bfc.v.ptr, bfn))));
        defer for (bfk) |b| a.free(b);
        var codes: [2][]u8 = undefined;
        for (0..2) |w| codes[w] = try a.dupe(u8, try rig.bytes(if (w == 0) i8c.k.ptr else i8c.v.ptr, KV * ring * D));
        defer for (codes) |b| a.free(b);
        var scl: [2][]f32 = undefined;
        for (0..2) |w| scl[w] = try a.dupe(f32, @alignCast(std.mem.bytesAsSlice(f32, try rig.bytes(if (w == 0) i8c.kscale.ptr else i8c.vscale.ptr, KV * ring * 4))));
        defer for (scl) |b| a.free(b);
        var rcodes: [2][]u8 = undefined;
        for (0..2) |w| rcodes[w] = try a.dupe(u8, try rig.bytes(if (w == 0) i8r.k.ptr else i8r.v.ptr, KV * ring * D));
        defer for (rcodes) |b| a.free(b);
        var rscl: [2][]f32 = undefined;
        for (0..2) |w| rscl[w] = try a.dupe(f32, @alignCast(std.mem.bytesAsSlice(f32, try rig.bytes(if (w == 0) i8r.kscale.ptr else i8r.vscale.ptr, KV * ring * 4))));
        defer for (rscl) |b| a.free(b);
        var ring_bad: usize = 0; // prep8q ring codes and scales against the prep8 ring, rows from ring_from
        var ring_n: usize = 0;
        for (0..2) |w| for (0..KV) |h| for (ring_from - p0..rows) |i| {
            const at = (p0 + i) % ring;
            ring_bad += @intFromBool(@as(u32, @bitCast(rscl[w][h * ring + at])) != @as(u32, @bitCast(scl[w][h * ring + at])));
            for (0..D) |d| {
                const o = if (w == 0) plan.kCodeOffset(h, at, d, ring) else plan.vCodeOffset(h, at, d, ring);
                ring_bad += @intFromBool(rcodes[w][o] != codes[w][o]);
            }
            ring_n += D + 1;
        };
        var stg: [4][]u16 = undefined; // prep8q K, V, then dequant8 K, V
        for (0..4) |w| stg[w] = try a.dupe(u16, @alignCast(std.mem.bytesAsSlice(u16, try rig.bytes(switch (w) {
            0 => stq.k.ptr,
            1 => stq.v.ptr,
            2 => std8.k.ptr,
            else => std8.v.ptr,
        }, stn))));
        defer for (stg) |b| a.free(b);
        // The attention image builds without -cl-fp32-correctly-rounded-divide-sqrt, so the device `amax / 127` and
        // `v / qs` may sit up to 2.5 ulp from the IEEE quotient. The check accepts exactly that and nothing more: a scale
        // 1 ulp from the IEEE `amax / 127`, and a code one step from `rint(v / qs)` (the device qs, IEEE divide) only
        // where that quotient is within 3 ulp of a tie. The stage and dequant8 must equal bf16(device code * device qs).
        var bad: [6]usize = @splat(0); // codes, scales, prep8q stage, dequant8, stage against dequant8, q and rel
        var near_scale: usize = 0; // scales 1 ulp from the IEEE quotient
        var near_tie: usize = 0; // codes one step off at a near tie
        var want_codes: [D]u8 = undefined;
        for (0..2) |w| for (0..KV) |h| for (0..rows) |i| {
            const p = p0 + i;
            const at = p % ring;
            const st = (h * stage_ring + p % stage_ring) * D;
            const bfrow = bfk[w][(h * ring + at) * D ..][0..D];
            const sh = quantize(bfrow, &want_codes);
            const sd = scl[w][h * ring + at];
            const ulps = @abs(@as(i64, @as(u32, @bitCast(sh))) - @as(i64, @as(u32, @bitCast(sd))));
            if (ulps == 1) near_scale += 1 else if (ulps > 1) bad[1] += 1;
            for (0..D) |d| {
                const got: i8 = @bitCast(codes[w][if (w == 0) plan.kCodeOffset(h, at, d, ring) else plan.vCodeOffset(h, at, d, ring)]);
                const t = if (sd > 0) bf2f(bfrow[d]) / sd else 0;
                const want: i8 = @intFromFloat(@max(-127.0, @min(127.0, rintEven(t))));
                if (got != want) {
                    const tie = @floor(t) + 0.5;
                    const ulp_t = @abs(std.math.nextAfter(f32, t, std.math.inf(f32)) - t);
                    const one_step = @abs(@as(i16, got) - @as(i16, want)) == 1;
                    if (one_step and @abs(t - tie) <= 3 * ulp_t) near_tie += 1 else bad[0] += 1;
                }
                const want_bf = f2bf(@as(f32, @floatFromInt(got)) * sd);
                bad[2] += @intFromBool(stg[w][st + d] != want_bf);
                bad[3] += @intFromBool(stg[2 + w][st + d] != want_bf);
                bad[4] += @intFromBool(stg[w][st + d] != stg[2 + w][st + d]);
            }
        };
        for (1..3) |w| {
            bad[5] += @intFromBool(!std.mem.eql(u8, qx[0], qx[w]));
            bad[5] += @intFromBool(!std.mem.eql(u8, rx[0], rx[w]));
        }
        const n = 2 * KV * rows * D;
        std.debug.print("CHECK-REAL prep {s} ring {d}, {d} rows from {d}: prep8 scales against the IEEE amax / 127: {d} of {d} 1 ulp off, {d} further; prep8 codes against rint(v / qs) on the device qs: {d} of {d} one step off at a tie within 3 ulp, {d} other; prep8q stage against bf16(code * qs) {d} of {d} differ; dequant8 {d} of {d}; prep8q stage against dequant8 of the prep8 ring {d} of {d}; prep8q ring write (rows from {d}) against the prep8 ring {d} of {d}; q and rel outputs of prep8 and prep8q against prep: {d} of 4 differ\n", .{ if (layer.window > 0) "sliding" else "global", ring, rows, p0, near_scale, 2 * KV * rows, bad[1], near_tie, n, bad[0], bad[2], n, bad[3], n, bad[4], n, ring_from, ring_bad, ring_n, bad[5] });
        var sum: usize = ring_bad;
        for (bad) |b| sum += b;
        if (sum != 0) failed += 1;
        cases_run += 1;
    }
    // 2. the two attention reads against the fp64 partial
    const cases = [_]Case{
        .{ .name = "global 1 row, 3,000 keys", .rows = 1, .k = 3000, .window = 0 },
        .{ .name = "global 8 rows, 3,000 keys", .rows = 8, .k = 3000, .window = 0 },
        .{ .name = "global 1 row, 40,000 keys", .rows = 1, .k = 40000, .window = 0 },
        .{ .name = "sliding 1 row, 2,050 keys, window 1,024", .rows = 1, .k = 2050, .window = 1024 },
        .{ .name = "global prompt 100 rows, 3,000 keys", .rows = 100, .k = 3000, .window = 0, .prompt = true },
        .{ .name = "global prompt 256 rows, 2,056 keys", .rows = 256, .k = 2056, .window = 0, .prompt = true },
    };
    for ([_]u32{ 0, 128 }) |splits| {
        plan.flash_splits = splits;
        plan.flash_user_set = splits != 0;
        for (cases) |cs| {
            var rig: rig_.Rig = .{ .a = a, .dev = dev };
            defer rig.deinit();
            const layer: plan.Layer = if (cs.window > 0) plan.Layer.sliding(cs.window, cs.rows) else plan.Layer.global(cs.k);
            const rows = cs.rows;
            const ring: usize = layer.ring;
            const stride = @max(plan.slotsFor(layer, cs.k), plan.chunks(cs.k - 1, plan.prompt_chunk));
            const cache_n = @as(usize, KV) * ring * D;
            const kb = try a.alloc(u16, cache_n);
            defer a.free(kb);
            const vb = try a.alloc(u16, cache_n);
            defer a.free(vb);
            const kc = try a.alloc(u8, cache_n);
            defer a.free(kc);
            const vc = try a.alloc(u8, cache_n);
            defer a.free(vc);
            const ks = try a.alloc(f32, KV * ring);
            defer a.free(ks);
            const vs = try a.alloc(f32, KV * ring);
            defer a.free(vs);
            const kref = try a.alloc(f64, cache_n);
            defer a.free(kref);
            const vref = try a.alloc(f64, cache_n);
            defer a.free(vref);
            var row: [D]u16 = undefined;
            var cd: [D]u8 = undefined;
            var vsq: f64 = 0;
            for (0..KV) |h| for (0..ring) |p| for (0..2) |w| {
                // a row's size varies by up to 64 times between positions, so each position has its own real scale
                const f = std.math.pow(f32, 2, rng.float(f32) * 6 - 3);
                for (&row) |*x| x.* = f2bf((rng.float(f32) - 0.5) * f);
                const sc = quantize(&row, &cd);
                (if (w == 0) ks else vs)[h * ring + p] = sc;
                for (0..D) |d| {
                    const code: i8 = @bitCast(cd[d]);
                    const exact = @as(f64, @floatFromInt(code)) * @as(f64, sc);
                    const at = (h * ring + p) * D + d;
                    if (w == 0) {
                        kc[plan.kCodeOffset(h, p, d, ring)] = cd[d];
                        kb[at] = f2bf(@as(f32, @floatFromInt(code)) * sc);
                        kref[at] = exact;
                    } else {
                        vc[plan.vCodeOffset(h, p, d, ring)] = cd[d];
                        vb[at] = f2bf(@as(f32, @floatFromInt(code)) * sc);
                        vref[at] = exact;
                        vsq += exact * exact;
                    }
                }
            };
            const vrms = @sqrt(vsq / @as(f64, @floatFromInt(cache_n)));
            const bf: attn.Cache = .{ .k = .{ .ptr = try rig.input(std.mem.sliceAsBytes(kb)) }, .v = .{ .ptr = try rig.input(std.mem.sliceAsBytes(vb)) } };
            const i8c: attn.Cache = .{
                .k = .{ .ptr = try rig.input(kc) },
                .v = .{ .ptr = try rig.input(vc) },
                .int8 = true,
                .kscale = .{ .ptr = try rig.input(std.mem.sliceAsBytes(ks)) },
                .vscale = .{ .ptr = try rig.input(std.mem.sliceAsBytes(vs)) },
            };
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
            const po_n = @as(usize, rows) * H * stride * (D * 4);
            const pml_n = @as(usize, rows) * H * stride * 8;
            const po: attn.Ref = .{ .ptr = try rig.scratch(po_n) };
            const pml: attn.Ref = .{ .ptr = try rig.scratch(pml_n) };
            const out: attn.Ref = .{ .ptr = try rig.output(@as(usize, rows) * H * D * 2) };
            const r: plan.Rows = .{ .rows = rows, .last_pos = cs.k - 1 };
            const path: plan.Path = if (cs.prompt) .prompt else .decode;
            const launch = try plan.attendAs(r, layer, stride, path);
            var errs: [2]RealErr = undefined;
            var words: [2][]u8 = undefined;
            const copy = try promptCopy(&rig, ring);
            for ([_]attn.Cache{ bf, i8c }, 0..) |c, w| {
                try attn.attendAs(k, dev.s, path, r, layer, stride, meta, qd, try readAs(k, dev, c, copy, cs, layer), tabd, po, pml, sinkd, gated, out);
                const pob = try a.dupe(u8, try rig.bytes(po.ptr, po_n));
                defer a.free(pob);
                const pmlb = try a.dupe(u8, try rig.bytes(pml.ptr, pml_n));
                defer a.free(pmlb);
                errs[w] = realErrors(cs, layer, launch.chunk, stride, q, tab, pos, kref, vref, vrms, @alignCast(std.mem.bytesAsSlice(f32, pob)), @alignCast(std.mem.bytesAsSlice([2]f32, pmlb)));
                words[w] = try std.mem.concat(a, u8, &.{ pob, pmlb });
            }
            defer for (words) |x| a.free(x);
            // The int8 prompt read is bf16(code * scale) by design (the bf16 copy): it must give the bf16 kernel's bits on
            // the bf16 cache of those values. The fp64 bounds hold for the reads of code * scale (decode8) only.
            var differ: usize = 0;
            for (std.mem.bytesAsSlice(u32, words[0]), std.mem.bytesAsSlice(u32, words[1])) |x, y| differ += @intFromBool(x != y);
            const ok = if (cs.prompt) differ == 0 else errs[1].ok();
            if (cs.prompt) std.debug.print("CHECK-REAL splits {d} chunk {d}, {s}: the int8 prompt read against the bf16 kernel on bf16(code * scale), partial words that differ {d} of {d}\n", .{ splits, launch.chunk, cs.name, differ, words[0].len / 4 });
            std.debug.print("CHECK-REAL splits {d} chunk {d}, {s}, {s}: against fp64, top / sum / mean error: int8 {e:.2} / {e:.2} / {e:.2}, bf16 cache of bf16(code * scale) {e:.2} / {e:.2} / {e:.2}{s}\n", .{ splits, launch.chunk, cs.name, if (cs.prompt) "tf_ink2_kv_dequant8 then tf_ink2_attn_prompt" else "tf_ink2_attn_decode8", errs[1].top, errs[1].sum, errs[1].mean, errs[0].top, errs[0].sum, errs[0].mean, if (ok) "" else if (cs.prompt) ", DIFFERS" else ", ABOVE BOUND" });
            if (!ok) failed += 1;
            cases_run += 1;
        }
    }
    std.debug.print("CHECK-REAL {s}: {d} of {d} cases fail (bounds: top {e:.1}, sum {e:.1}, mean {e:.1})\n", .{ if (failed == 0) "PASS" else "FAIL", failed, cases_run, real_bounds.top, real_bounds.sum, real_bounds.mean });
    return failed;
}

/// The bf16 copy an int8 cache's global prompt visit reads: K and V of KV heads over `ring` positions.
fn promptCopy(rig: anytype, ring: usize) !attn.Cache {
    return .{ .k = .{ .ptr = try rig.scratch(KV * ring * D * 2) }, .v = .{ .ptr = try rig.scratch(KV * ring * D * 2) } };
}

/// The cache a launch reads: an int8 cache on the prompt path enters as its bf16 copy, `tf_ink2_kv_dequant8` of every
/// key up to the case's last one, as the prefill does a global visit. Every other launch reads the cache itself.
fn readAs(k: *const attn.Kernels, dev: anytype, c: attn.Cache, copy: attn.Cache, cs: Case, layer: plan.Layer) !attn.Cache {
    if (!c.int8 or !cs.prompt) return c;
    try attn.dequant8(k, dev.s, c, copy, 0, layer.ring, layer.ring, cs.k, KV);
    return copy;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var kv_int8 = false;
    var splits: u32 = 0;
    var do_check = false;
    var do_check_real = false;
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
        } else if (std.mem.eql(u8, arg, "--check")) {
            do_check = true;
        } else if (std.mem.eql(u8, arg, "--sync-each")) {
            sync_each = true;
        } else if (std.mem.eql(u8, arg, "--check-real")) {
            do_check_real = true;
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
    for ([_]struct { []const u8, xpu.Kernel }{.{ "tf_ink2_attn_prompt", k.prompt }}) |e| {
        const p = try dev.properties(e[1]);
        std.debug.print("BENCH kernel properties: {s} {d} spill bytes, {d} private bytes, {d} local bytes, {d} arguments\n", .{ e[0], p.spillMemSize, p.privateMemSize, p.localMemSize, p.numKernelArgs });
        const occ = occupancy(&dev, e[1]);
        std.debug.print("BENCH occupancy: {s} {d} groups of 16 lanes resident on the device ({d} Xe cores: {d} a core), zeKernelSuggestMaxCooperativeGroupCount result {d}\n", .{ e[0], occ.groups, occ.cores, if (occ.cores > 0) occ.groups / occ.cores else 0, occ.rc });
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
        .{ .name = "global prompt 256 rows over 40,000 keys", .rows = 256, .k = 40000, .window = 0, .prompt = true },
        .{ .name = "global prompt 2,048 rows over 40,000 keys", .rows = 2048, .k = 40000, .window = 0, .prompt = true },
    };
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    if (do_check) return if (try check(a, &dev, &k, rng) == 0) 0 else 1;
    if (do_check_real) return if (try checkReal(a, &dev, &k, rng) == 0) 0 else 1;
    std.debug.print("BENCH inkling2 attention on {s}, kv {s}, splits {d}, roof {d:.1} GB/s read\n", .{ "bmg-g31", if (kv_int8) "int8" else "bf16", splits, roof_gbs });
    for (cases) |cs| {
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
        const copy = try promptCopy(&rig, ring);
        var t_decode: i96 = 0;
        var t_enqueue: i96 = 0;
        for (0..2) |pass| {
            const t0 = ms(init.io);
            for (0..reps) |i| {
                try attn.attendAs(&k, dev.s, path, r, layer, stride, meta, qd, try readAs(&k, &dev, kv[i % sets], copy, cs, layer), tabd, po, pml, sinkd, gated, out);
                if (sync_each) try dev.s.synchronize();
            }
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
                        // an int8 cache: the dequant of every key to the bf16 copy, then the prompt kernel (the prefill's order)
                        const b = try readAs(&k, &dev, c, copy, cs, layer);
                        try xpu.launch.dispatch(k.prompt, dev.s, launch.grid, .{ qd.at(), b.k.at(), b.v.at(), tabd.at(), meta.segs.at(), po.at(), pml.at(), @as(u32, H), @as(u32, KV), layer.ring, launch.chunk, @as(u32, 0), launch.partials_rh, sc });
                    } else {
                        try xpu.launch.dispatch(k.merge, dev.s, launch.merge_grid, .{ po.at(), pml.at(), sinkd.at(), gated.at(), meta.pos.at(), out.at(), @as(u32, H), layer.window, launch.chunk, @as(u32, 0), launch.partials_rh, @as(u32, 1) });
                    }
                    if (sync_each) try dev.s.synchronize();
                }
                try dev.s.synchronize();
                if (pass == 1) t_part[which] = ms(init.io) - t0;
            }
        }
        const per_ns = @as(f64, @floatFromInt(t_decode)) / reps;
        const att_ns = @as(f64, @floatFromInt(t_part[0])) / reps;
        std.debug.print("BENCH   parts: attention kernel {d:.1} us, merge {d:.1} us ({s})\n", .{ att_ns / 1000, @as(f64, @floatFromInt(t_part[1])) / reps / 1000, method() });
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
            std.debug.print("BENCH {s}: {d:.1} us a call (attention kernel and merge), {d:.2} TFLOP/s ({s})\n", .{ cs.name, per_ns / 1000, flops / per_ns / 1000, method() });
            continue;
        }
        std.debug.print("BENCH {s}: attention kernel {d:.1} us, merge {d:.1} us, both {d:.1} us a call (host enqueue {d:.1} us), {d:.1} MB of K and V, {d:.1} GB/s = {d:.2} of {d:.1} ({s})\n", .{ cs.name, att_ns / 1000, @as(f64, @floatFromInt(t_part[1])) / reps / 1000, per_ns / 1000, @as(f64, @floatFromInt(t_enqueue)) / reps / 1000, bytes / 1e6, gbs, gbs / roof_gbs, roof_gbs, method() });
    }
    return 0;
}

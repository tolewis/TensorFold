//! Inkling 2 Mini's routed experts on XPU under the contract: the stored NVFP4 and MXFP8 weights against an independent
//! float64 decode and dot at the served shapes (hidden 2,304, expert width 1,152, top 8 of 128 experts), on both the
//! decode kernels (a gather a route) and the prompt kernels (routes grouped by expert), the weight decoders, the
//! ascending-expert combine and the grouping plan. Each kernel has a planted twin that must go red at gate 3.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const host = @import("inkling2_xpu_experts");
const rig_ = @import("contract_rig.zig");

const c = core.spec;
const g = core.gates;
const rows_ = core.rows;
const bf = core.bf16_round.bf16;

const HID: usize = host.hidden;
const FF: usize = host.intermediate;
const E: usize = host.expert_count;
const TOPK: usize = host.top_k;
/// The experts a test fills with weights; every id a call routes to is one of them, spread over the 128.
const POOL: usize = 16;

fn poolId(i: usize) usize {
    return 3 + 8 * i;
}
fn slotOf(id: usize) usize {
    return (id - 3) / 8;
}

pub const Kind = enum {
    nv_gu,
    nv_down,
    mx_gu,
    mx_down,
    nv_gu_prompt,
    nv_down_prompt,
    mx_gu_prompt,
    mx_down_prompt,
    nv_quant,
    mx_quant,
    combine,
    plan,

    pub fn op(k: Kind) host.Op {
        return switch (k) {
            .nv_gu => .nvfp4_gu,
            .nv_down => .nvfp4_down,
            .mx_gu => .mxfp8_gu,
            .mx_down => .mxfp8_down,
            .nv_gu_prompt => .nvfp4_gu_prompt,
            .nv_down_prompt => .nvfp4_down_prompt,
            .mx_gu_prompt => .mxfp8_gu_prompt,
            .mx_down_prompt => .mxfp8_down_prompt,
            .nv_quant => .nvfp4_quantize,
            .mx_quant => .mxfp8_quantize,
            .combine => .combine,
            .plan => .plan,
        };
    }
    fn isNv(k: Kind) bool {
        return switch (k) {
            .nv_gu, .nv_down, .nv_gu_prompt, .nv_down_prompt, .nv_quant => true,
            else => false,
        };
    }
    fn isGu(k: Kind) bool {
        return k == .nv_gu or k == .mx_gu or k == .nv_gu_prompt or k == .mx_gu_prompt;
    }
    fn isDown(k: Kind) bool {
        return k == .nv_down or k == .mx_down or k == .nv_down_prompt or k == .mx_down_prompt;
    }
    fn isPrompt(k: Kind) bool {
        return k == .nv_gu_prompt or k == .nv_down_prompt or k == .mx_gu_prompt or k == .mx_down_prompt;
    }
    fn isQuant(k: Kind) bool {
        return k == .nv_quant or k == .mx_quant;
    }
    fn isMatmul(k: Kind) bool {
        return k.isGu() or k.isDown();
    }
};

/// The image's symbol prefix a planted twin of `k` carries (one kernel an include of experts.cl).
pub fn controlPrefix(a: std.mem.Allocator, k: Kind) ![]u8 {
    return std.fmt.allocPrint(a, "tf_ctl_i2_{s}_", .{k.op().name()});
}

const gu_decode = [_]c.Shape{ .{ .rows = 1, .n = FF, .k = HID }, .{ .rows = 4, .n = FF, .k = HID }, .{ .rows = 16, .n = FF, .k = HID } };
const down_decode = [_]c.Shape{ .{ .rows = 1, .n = HID, .k = FF }, .{ .rows = 4, .n = HID, .k = FF }, .{ .rows = 16, .n = HID, .k = FF } };
const gu_prompt = [_]c.Shape{ .{ .rows = 16, .n = FF, .k = HID }, .{ .rows = 40, .n = FF, .k = HID } };
const down_prompt = [_]c.Shape{ .{ .rows = 16, .n = HID, .k = FF }, .{ .rows = 40, .n = HID, .k = FF } };
const quant_served = [_]c.Shape{ .{ .rows = 64, .n = HID, .k = HID }, .{ .rows = 2304, .n = FF, .k = FF } };
const combine_served = [_]c.Shape{ .{ .rows = 1, .n = HID, .k = TOPK }, .{ .rows = 4, .n = HID, .k = TOPK }, .{ .rows = 16, .n = HID, .k = TOPK } };
const plan_served = [_]c.Shape{ .{ .rows = 16, .n = E, .k = TOPK }, .{ .rows = 40, .n = E, .k = TOPK }, .{ .rows = 1, .n = E, .k = TOPK } };
// A planted twin is red at its first shape, so one small shape is the whole proof and the run stays short.
const gu_control = [_]c.Shape{.{ .rows = 4, .n = FF, .k = HID }};
const down_control = [_]c.Shape{.{ .rows = 4, .n = HID, .k = FF }};
const gu_prompt_control = [_]c.Shape{.{ .rows = 16, .n = FF, .k = HID }};
const down_prompt_control = [_]c.Shape{.{ .rows = 16, .n = HID, .k = FF }};
const quant_control = [_]c.Shape{.{ .rows = 8, .n = HID, .k = HID }};
const combine_control = [_]c.Shape{.{ .rows = 4, .n = HID, .k = TOPK }};
const plan_control = [_]c.Shape{.{ .rows = 16, .n = E, .k = TOPK }};

fn depth(s: c.Shape) usize {
    return s.k + 3;
}
fn zero(_: c.Shape) usize {
    return 0;
}

fn title(k: Kind) []const u8 {
    return switch (k) {
        .nv_gu => "xpu inkling2 experts tf_ink2_nvfp4_gu",
        .nv_down => "xpu inkling2 experts tf_ink2_nvfp4_down",
        .mx_gu => "xpu inkling2 experts tf_ink2_mxfp8_gu",
        .mx_down => "xpu inkling2 experts tf_ink2_mxfp8_down",
        .nv_gu_prompt => "xpu inkling2 experts tf_ink2_nvfp4_gu_prompt",
        .nv_down_prompt => "xpu inkling2 experts tf_ink2_nvfp4_down_prompt",
        .mx_gu_prompt => "xpu inkling2 experts tf_ink2_mxfp8_gu_prompt",
        .mx_down_prompt => "xpu inkling2 experts tf_ink2_mxfp8_down_prompt",
        .nv_quant => "xpu inkling2 experts tf_ink2_nvfp4_quantize",
        .mx_quant => "xpu inkling2 experts tf_ink2_mxfp8_quantize",
        .combine => "xpu inkling2 experts tf_ink2_combine",
        .plan => "xpu inkling2 experts tf_ink2_plan",
    };
}

pub fn spec(k: Kind, control: bool) c.Spec {
    var s = c.Spec{
        .name = if (control) controlName(k) else title(k),
        .accum = .f32,
        .depth = depth,
        .ops = if (k.isGu()) 12 else if (k.isQuant()) 1 else 3,
        .mids = if (k.isGu()) 3 else 0,
        .out = .bf16,
        .served = &gu_decode,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and explicit BF16 nodes" },
    };
    s.served = switch (k) {
        .nv_gu, .mx_gu => if (control) &gu_control else &gu_decode,
        .nv_down, .mx_down => if (control) &down_control else &down_decode,
        .nv_gu_prompt, .mx_gu_prompt => if (control) &gu_prompt_control else &gu_prompt,
        .nv_down_prompt, .mx_down_prompt => if (control) &down_prompt_control else &down_prompt,
        .nv_quant, .mx_quant => if (control) &quant_control else &quant_served,
        .combine => if (control) &combine_control else &combine_served,
        .plan => if (control) &plan_control else &plan_served,
    };
    if (k.isQuant()) s.accum = .none;
    if (k == .plan) {
        s.accum = .none;
        s.out = .raw;
        s.ops = 0;
        s.depth = zero;
    }
    if (k == .combine) s.sums = .{ .f16 = 0, .bf16 = 100, .why = "bounded cancellation is exact in fp32 but loses its residual in bf16" };
    return s;
}

fn controlName(k: Kind) []const u8 {
    return switch (k) {
        .nv_gu => "control: xpu inkling2 experts tf_ink2_nvfp4_gu up row",
        .nv_down => "control: xpu inkling2 experts tf_ink2_nvfp4_down factor",
        .mx_gu => "control: xpu inkling2 experts tf_ink2_mxfp8_gu up row",
        .mx_down => "control: xpu inkling2 experts tf_ink2_mxfp8_down scale",
        .nv_gu_prompt => "control: xpu inkling2 experts tf_ink2_nvfp4_gu_prompt up row",
        .nv_down_prompt => "control: xpu inkling2 experts tf_ink2_nvfp4_down_prompt factor",
        .mx_gu_prompt => "control: xpu inkling2 experts tf_ink2_mxfp8_gu_prompt up row",
        .mx_down_prompt => "control: xpu inkling2 experts tf_ink2_mxfp8_down_prompt scale",
        .nv_quant => "control: xpu inkling2 experts tf_ink2_nvfp4_quantize nibble",
        .mx_quant => "control: xpu inkling2 experts tf_ink2_mxfp8_quantize scale",
        .combine => "control: xpu inkling2 experts tf_ink2_combine route-slot order",
        .plan => "control: xpu inkling2 experts tf_ink2_plan dropped route",
    };
}

// ---- the formats, decoded in float64 as the checkpoint defines them -------------------------------------------------

fn e2(code: u8) f64 {
    const v = ([_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[code & 7];
    return if (code & 8 != 0) -v else v;
}

fn e4(code: u8) f64 {
    const exponent = (code >> 3) & 15;
    const mantissa = code & 7;
    if (exponent == 15 and mantissa == 7) return std.math.nan(f64);
    const v = if (exponent == 0) @as(f64, @floatFromInt(mantissa)) / 512 else std.math.ldexp(@as(f64, 8 + @as(u16, mantissa)), @as(i32, exponent) - 10);
    return if (code & 0x80 != 0) -v else v;
}

fn round32(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

/// The per-expert NVFP4 factor: both globals rounded to bf16, one correctly rounded fp32 divide.
fn phiOf(amax: f32, den: f32) f32 {
    return round32(amax) / round32(den);
}

/// One NVFP4 weight as bf16 bits: (e2m1 * e4m3) exact, times the factor, one fp32 rounding, one bf16 rounding.
fn nvWeight(code: u8, scale: u8, phi: f32) u16 {
    return c.toBf16(@floatCast(e2(code) * e4(scale) * @as(f64, phi)));
}

/// One MXFP8 weight as bf16 bits: e4m3 * 2^(scale - 127) is exact in float64, then one fp32 and one bf16 rounding.
fn mxWeight(code: u8, scale: u8) u16 {
    return c.toBf16(@floatCast(std.math.ldexp(e4(code), @as(i32, scale) - 127)));
}

const Sim = struct { acc: c.Accum, order: g.Order };
const Dot = struct { value: f64, magnitude: f64 };
const Sample = struct { value: f64, magnitude: f64 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    ks: host.Kernels,
    kind: Kind,
    ctl: bool,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    // The pool's weights, kept between prepares of one fill: raw checkpoint bytes and their float64-checked bf16 decode.
    cached: ?g.Fill = null,
    cached_dims: [2]usize = .{ 0, 0 }, // the quantizers' rows and k: their weights are the shape
    plan_ks: ?host.Kernels = null, // the prompt kernels' grouping, from the plain image
    pw: []u8 = &.{}, // POOL matrices of one expert's codes
    ps: []u8 = &.{}, // and their scales
    dec: []u16 = &.{}, // the decoded bf16 weights of the pool's matrices
    amax: [E]f32 = undefined,
    den: [1]f32 = undefined,
    // One call's host inputs.
    x: []u16 = &.{},
    ids: []i32 = &.{},
    weights: []f32 = &.{},
    terms: []f64 = &.{},
    // Device addresses.
    d_x: ?*anyopaque = null,
    d_w: ?*anyopaque = null,
    d_s: ?*anyopaque = null,
    d_amax: ?*anyopaque = null,
    d_den: ?*anyopaque = null,
    d_ids: ?*anyopaque = null,
    d_weights: ?*anyopaque = null,
    d_out: ?*anyopaque = null,
    d_members: ?*anyopaque = null,
    d_items: ?*anyopaque = null,
    d_count: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kind: Kind, control: bool) !Probe {
        var ks: host.Kernels = undefined;
        if (control) {
            const prefix = try controlPrefix(a, kind);
            defer a.free(prefix);
            ks = try host.Kernels.load(try dev.module(.inkling2_ctl), prefix, kind.op());
        } else {
            ks = try host.Kernels.load(try dev.module(.inkling2), host.default_prefix, kind.op());
        }
        var out: Probe = .{ .a = a, .rig = .{ .a = a, .dev = dev }, .ks = ks, .kind = kind, .ctl = control, .arena = .init(a) };
        if (kind.isPrompt()) out.plan_ks = try host.Kernels.load(try dev.module(.inkling2), host.default_prefix, .plan);
        return out;
    }

    fn dropCache(p: *Probe) void {
        p.a.free(p.pw);
        p.a.free(p.ps);
        p.a.free(p.dec);
        p.pw = &.{};
        p.ps = &.{};
        p.dec = &.{};
        p.cached = null;
    }

    pub fn deinit(p: *Probe) void {
        p.dropCache();
        p.ks.deinit();
        if (p.plan_ks) |*k| k.deinit();
        p.arena.deinit();
        p.rig.deinit();
    }

    fn releaseAll(ctx: *anyopaque) void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.rig.reset();
        p.dropCache();
        _ = p.arena.reset(.free_all);
        p.x = &.{};
        p.ids = &.{};
        p.weights = &.{};
        p.terms = &.{};
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.ctl), .ctx = p, .vt = if (p.kind == .plan) &vt_plan else &vt, .control = if (p.ctl) .shapes else null };
    }

    /// Every kernel but the plan compares to the float64 reference; the plan's bytes are checked exactly instead.
    const vt = blk: {
        var t = g.hooks(Probe);
        t.release = releaseAll;
        t.check = null;
        break :blk t;
    };
    const vt_plan = blk: {
        var t = g.hooks(Probe);
        t.release = releaseAll;
        break :blk t;
    };

    fn call(p: *const Probe) host.Call {
        return .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n), .format = if (p.kind.isNv()) .nvfp4 else .mxfp8 };
    }

    /// Matrix rows an expert holds: gate and up interleaved for a GU, the output columns for a down.
    fn rowsPerExpert(p: *const Probe) usize {
        return if (p.kind.isGu()) 2 * p.shape.n else p.shape.n;
    }
    fn codeBytes(p: *const Probe) usize {
        return if (p.kind.isNv()) p.shape.k / 2 else p.shape.k;
    }
    fn blockSize(p: *const Probe) usize {
        return if (p.kind.isNv()) 16 else 32;
    }

    fn accepts(p: *const Probe, s: c.Shape) bool {
        const k = p.kind;
        if (s.rows == 0) return false;
        if (k.isGu()) return s.n == FF and s.k == HID and s.rows <= 64;
        if (k.isDown()) return s.n == HID and s.k == FF and s.rows <= 64;
        if (k.isQuant()) return s.n == s.k and s.k % 32 == 0 and s.rows * s.k <= 1 << 26;
        if (k == .combine) return s.n % 64 == 0 and s.n <= HID and s.k == TOPK and s.rows <= 64;
        return s.n == E and s.k == TOPK and s.rows <= 64; // plan
    }

    /// The pool's weights and scales for `fill`, decoded; regenerated only when the fill changes.
    fn ensureWeights(p: *Probe, fill: g.Fill) !void {
        const dims = [2]usize{ if (p.kind.isQuant()) p.shape.rows else 0, p.shape.k };
        if (p.cached) |f| if (f == fill and std.mem.eql(usize, &dims, &p.cached_dims)) return;
        p.dropCache();
        const nv = p.kind.isNv();
        const mats: usize = if (p.kind.isQuant()) 1 else POOL;
        const rpe: usize = if (p.kind.isQuant()) p.shape.rows else p.rowsPerExpert();
        const cb = p.codeBytes();
        const blk = p.blockSize();
        p.pw = try p.a.alloc(u8, mats * rpe * cb);
        p.ps = try p.a.alloc(u8, mats * rpe * (p.shape.k / blk));
        var prng = std.Random.DefaultPrng.init(0x1c2b5 + @as(u64, @intFromEnum(p.kind)));
        const rng = prng.random();
        rng.bytes(p.pw);
        // scales that keep a dot over 2,304 terms near 1 to 10, so the activation stays off its saturated ends
        for (p.ps, 0..) |*v, i| v.* = if (nv) @intCast(0x30 + (i % 17)) else @intCast(119 + (i % 5));
        if (!nv) for (p.pw) |*v| if (v.* & 0x7f == 0x7f) {
            v.* = (v.* & 0x80) | 0x7e; // a stored NaN would make every sum NaN; the quantizers plant it on purpose
        };
        // amax / denominator: both exact in bf16, a ratio near 1 that is no power of two
        p.den[0] = 2688;
        for (&p.amax, 0..) |*v, i| v.* = 2688 + 16 * @as(f32, @floatFromInt(i % 9));
        if (fill == .stress) {
            // every weight a small multiple of a power of two and the factor 1: each fp32 sum of the products is exact
            @memset(p.ps, if (nv) 0x38 else 127);
            for (&p.amax) |*v| v.* = 2688;
            if (!nv) for (p.pw) |*v| {
                v.* = (v.* & 0x87) | 0x38; // e4m3 in [1, 2)
            };
            // A GU's gate and up stay far from zero: the activation then saturates to 100 or to 0 on both sides, and
            // no stress sum lands where fp32's exp overflows (gate near -90) but float64's does not.
            if (p.kind.isGu()) for (p.pw) |*v| {
                v.* &= if (nv) 0x77 else 0x7f;
            };
        }
        if (p.kind == .mx_quant) {
            p.pw[0] = 0x30;
            p.ps[0] = 255; // 0.5 * 2^128: decodes to 2^127 without ever forming a 2^128 that overflows
            if (fill == .nonfinite) {
                p.pw[0] = 0x7f; // NaN
                p.pw[1] = 0x7e; // 448 * 2^128: infinity in bf16
                p.pw[2] = 0xfe;
            }
        }
        if (p.kind == .nv_quant and fill == .nonfinite) p.ps[0] = 0x7f; // a NaN scale: sixteen NaN weights
        if (!p.kind.isQuant()) try p.decodePool();
        p.cached = fill;
        p.cached_dims = dims;
    }

    fn decodePool(p: *Probe) !void {
        const nv = p.kind.isNv();
        const rpe = p.rowsPerExpert();
        const k = p.shape.k;
        const cb = p.codeBytes();
        const blk = p.blockSize();
        p.dec = try p.a.alloc(u16, POOL * rpe * k);
        for (0..POOL) |slot| {
            const phi = phiOf(p.amax[poolId(slot)], p.den[0]);
            for (0..rpe) |row| {
                const m = slot * rpe + row;
                for (0..k) |col| {
                    p.dec[m * k + col] = if (nv) blk_nv: {
                        const byte = p.pw[m * cb + col / 2];
                        const code = if (col % 2 == 0) byte & 15 else byte >> 4;
                        break :blk_nv nvWeight(code, p.ps[m * (k / blk) + col / blk], phi);
                    } else mxWeight(p.pw[m * cb + col], p.ps[m * (k / blk) + col / blk]);
                }
            }
        }
    }

    /// The full 128-expert buffers from the pool: the rest of the experts hold zeros no call routes to.
    fn expand(p: *Probe, src: []const u8, per_expert: usize) ![]u8 {
        const out = try p.a.alloc(u8, E * per_expert);
        @memset(out, 0);
        for (0..POOL) |slot| @memcpy(out[poolId(slot) * per_expert ..][0..per_expert], src[slot * per_expert ..][0..per_expert]);
        return out;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!p.accepts(shape)) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        const a = p.arena.allocator();
        const k = p.kind;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const routes = shape.rows * TOPK;
        p.x = &.{};
        p.weights = &.{};
        if (k != .nv_quant and k != .mx_quant) {
            // top 8 distinct of the pool a token; the first pool expert is in every token's picks, so it gathers a long run
            p.ids = try a.alloc(i32, routes);
            for (0..shape.rows) |t| {
                var perm: [POOL]usize = undefined;
                for (&perm, 0..) |*v, i| v.* = i;
                rng.shuffle(usize, &perm);
                if (std.mem.indexOfScalar(usize, perm[0..TOPK], 0) == null) {
                    const at = std.mem.indexOfScalar(usize, &perm, 0).?;
                    std.mem.swap(usize, &perm[0], &perm[at]);
                }
                for (0..TOPK) |s| p.ids[t * TOPK + s] = @intCast(poolId(perm[s]));
            }
            if (fill == .stress) for (p.ids, 0..) |*id, i| {
                id.* = @intCast(poolId((i % TOPK) % 4));
            };
        }
        if (k.isMatmul() or k == .nv_quant or k == .mx_quant) try p.ensureWeights(fill);
        if (k.isMatmul()) {
            const nx = if (k.isGu()) shape.rows * shape.k else routes * shape.k;
            p.x = try a.alloc(u16, nx);
            for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1) / 48);
            if (fill == .stress) for (p.x, 0..) |*v, i| {
                // a down's inputs change sign half way, so its sums cancel; a GU's tokens keep one sign each (see below)
                const neg = if (k.isGu()) (i / shape.k) % 2 == 1 else i >= p.x.len / 2;
                v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (neg) @as(f32, -1) else 1));
            };
            if (fill == .nonfinite) {
                p.x[0] = c.toBf16(std.math.nan(f32));
                p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
                p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
            }
            p.terms = try a.alloc(f64, shape.k);
            const per_w = p.rowsPerExpert() * p.codeBytes();
            const per_s = p.rowsPerExpert() * (shape.k / p.blockSize());
            const wfull = try p.expand(p.pw, per_w);
            defer p.a.free(wfull);
            const sfull = try p.expand(p.ps, per_s);
            defer p.a.free(sfull);
            p.d_x = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.d_w = try p.rig.input(wfull);
            p.d_s = try p.rig.input(sfull);
            if (k.isNv()) {
                p.d_amax = try p.rig.input(std.mem.sliceAsBytes(&p.amax));
                p.d_den = try p.rig.input(std.mem.sliceAsBytes(&p.den));
            }
            p.d_ids = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.d_out = try p.rig.output(routes * shape.n * 2);
            if (k.isPrompt()) {
                p.d_members = try p.rig.scratch(routes * 4);
                p.d_items = try p.rig.scratch(host.maxItems(p.call()) * 12);
                p.d_count = try p.rig.scratch(4);
            }
        } else if (k.isQuant()) {
            p.d_w = try p.rig.input(p.pw);
            p.d_s = try p.rig.input(p.ps);
            if (k == .nv_quant) {
                p.d_amax = try p.rig.input(std.mem.sliceAsBytes(p.amax[0..1]));
                p.d_den = try p.rig.input(std.mem.sliceAsBytes(&p.den));
            }
            p.d_out = try p.rig.output(shape.rows * shape.k * 2);
        } else if (k == .combine) {
            // down outputs of every route, the router's weights, and the ids that order them
            p.x = try a.alloc(u16, routes * shape.n);
            p.weights = try a.alloc(f32, routes);
            for (p.x, 0..) |*v, i| v.* = c.toBf16(@floatFromInt(1 + i % 8));
            if (p.ctl) {
                // Slots 0 and 1 add to 2^128, which overflows fp32, when summed in slot order. Ascending expert order
                // puts slot 2 (-2^127) between them, so the true sum is finite.
                const pattern = [_]usize{ 2, 0, 1, 7, 3, 6, 4, 5 };
                for (p.ids, 0..) |*id, i| id.* = @intCast(poolId(pattern[i % TOPK]));
                const cw = [_]f32{ 0x1p127, 0x1p127, -0x1p127, 1, 1, 1, 1, 1 };
                for (p.weights, 0..) |*v, i| v.* = cw[i % TOPK];
                for (p.x) |*v| v.* = c.toBf16(1);
            } else if (fill == .stress) {
                const terms = [_]f32{ 1024, 1, 1, 1, 1, 1, 1, -1029 };
                for (p.x) |*v| v.* = c.toBf16(1);
                for (0..shape.rows) |t| {
                    // the ascending sum is 1024 + six 1s - 1029: exact in fp32, not in bf16
                    var order: [TOPK]usize = undefined;
                    for (&order, 0..) |*o, i| o.* = i;
                    const ids = p.ids[t * TOPK ..][0..TOPK];
                    std.mem.sort(usize, &order, @as([]const i32, ids), struct {
                        fn less(m: []const i32, l: usize, r: usize) bool {
                            return if (m[l] != m[r]) m[l] < m[r] else l < r;
                        }
                    }.less);
                    for (order, 0..) |slot, rank| p.weights[t * TOPK + slot] = terms[rank];
                }
            } else {
                const ordinary = [_]f32{ 0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1 };
                for (p.weights, 0..) |*v, i| v.* = ordinary[i % TOPK];
            }
            if (fill == .nonfinite) {
                p.x[0] = c.toBf16(std.math.nan(f32));
                p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
                p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
            }
            p.d_x = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.d_ids = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.d_weights = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            p.d_out = try p.rig.output(shape.rows * shape.n * 2);
        } else {
            p.d_ids = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.d_members = try p.rig.output(routes * 4);
            p.d_items = try p.rig.output(host.maxItems(p.call()) * 12);
            p.d_count = try p.rig.output(4);
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.rig.dev.s;
        const cl = p.call();
        const ks = &p.ks;
        switch (p.kind) {
            .nv_gu => try ks.nvfp4Gu(s, p.d_x, p.d_w, p.d_s, p.d_amax, p.d_den, p.d_ids, p.d_out, cl),
            .nv_down => try ks.nvfp4Down(s, p.d_x, p.d_w, p.d_s, p.d_amax, p.d_den, p.d_ids, p.d_out, cl),
            .mx_gu => try ks.mxfp8Gu(s, p.d_x, p.d_w, p.d_s, p.d_ids, p.d_out, cl),
            .mx_down => try ks.mxfp8Down(s, p.d_x, p.d_w, p.d_s, p.d_ids, p.d_out, cl),
            .nv_gu_prompt, .nv_down_prompt, .mx_gu_prompt, .mx_down_prompt => {
                try planCall(p);
                switch (p.kind) {
                    .nv_gu_prompt => try ks.nvfp4GuPrompt(s, p.d_x, p.d_w, p.d_s, p.d_amax, p.d_den, p.d_items, p.d_count, p.d_members, p.d_out, cl),
                    .nv_down_prompt => try ks.nvfp4DownPrompt(s, p.d_x, p.d_w, p.d_s, p.d_amax, p.d_den, p.d_items, p.d_count, p.d_members, p.d_out, cl),
                    .mx_gu_prompt => try ks.mxfp8GuPrompt(s, p.d_x, p.d_w, p.d_s, p.d_items, p.d_count, p.d_members, p.d_out, cl),
                    else => try ks.mxfp8DownPrompt(s, p.d_x, p.d_w, p.d_s, p.d_items, p.d_count, p.d_members, p.d_out, cl),
                }
            },
            .nv_quant => try ks.nvfp4Quantize(s, p.d_w, p.d_s, p.d_amax, p.d_den, p.d_out, @intCast(p.shape.rows), @intCast(p.shape.k)),
            .mx_quant => try ks.mxfp8Quantize(s, p.d_w, p.d_s, p.d_out, @intCast(p.shape.rows), @intCast(p.shape.k)),
            .combine => try ks.combine(s, p.d_x, p.d_ids, p.d_weights, p.d_out, cl),
            .plan => try ks.plan(s, p.d_ids, p.d_members, p.d_items, p.d_count, cl),
        }
        try p.rig.dev.s.synchronize();
    }

    /// The prompt kernels' grouping comes from the plan kernel of the same image (or the planted twin's own plan).
    fn planCall(p: *Probe) !void {
        try p.plan_ks.?.plan(p.rig.dev.s, p.d_ids, p.d_members, p.d_items, p.d_count, p.call());
    }

    pub fn count(p: *const Probe) usize {
        const sh = p.shape;
        return switch (p.kind) {
            .nv_quant, .mx_quant => sh.rows * sh.k,
            .combine => sh.rows * sh.n,
            .plan => sh.rows * TOPK * 4 + host.maxItems(p.call()) * 12 + 4,
            else => sh.rows * TOPK * sh.n,
        };
    }

    // ---- the float64 reference ----------------------------------------------------------------------------------

    fn dot(p: *const Probe, slot: usize, row: usize, x: []const u16, sim: ?Sim) Dot {
        const k = p.shape.k;
        const w = p.dec[(slot * p.rowsPerExpert() + row) * k ..][0..k];
        var magnitude: f64 = 0;
        var sum: f64 = 0;
        if (sim) |s| {
            for (0..k) |i| {
                const t = @as(f64, c.fromBf16(w[i])) * @as(f64, c.fromBf16(x[i]));
                p.terms[i] = t;
                magnitude += @abs(t);
            }
            sum = rows_.sumIn(s.acc, s.order, p.terms[0..k]);
        } else {
            for (0..k) |i| {
                const t = @as(f64, c.fromBf16(w[i])) * @as(f64, c.fromBf16(x[i]));
                magnitude += @abs(t);
                sum += t;
            }
        }
        return .{ .value = sum, .magnitude = magnitude };
    }

    fn evaluate(p: *const Probe, at: usize, sim: ?Sim) Sample {
        const sh = p.shape;
        switch (p.kind) {
            .nv_quant => {
                const row = at / sh.k;
                const col = at % sh.k;
                const byte = p.pw[row * (sh.k / 2) + col / 2];
                const code = if (col % 2 == 0) byte & 15 else byte >> 4;
                return .{ .value = c.fromBf16(nvWeight(code, p.ps[row * (sh.k / 16) + col / 16], phiOf(p.amax[0], p.den[0]))), .magnitude = 0 };
            },
            .mx_quant => {
                const row = at / sh.k;
                const col = at % sh.k;
                return .{ .value = c.fromBf16(mxWeight(p.pw[row * sh.k + col], p.ps[row * (sh.k / 32) + col / 32])), .magnitude = 0 };
            },
            .combine => {
                const token = at / sh.n;
                const col = at % sh.n;
                var order: [TOPK]usize = undefined;
                for (&order, 0..) |*o, i| o.* = i;
                const ids = p.ids[token * TOPK ..][0..TOPK];
                std.mem.sort(usize, &order, @as([]const i32, ids), struct {
                    fn less(m: []const i32, l: usize, r: usize) bool {
                        return if (m[l] != m[r]) m[l] < m[r] else l < r;
                    }
                }.less);
                var terms: [TOPK]f64 = undefined;
                var magnitude: f64 = 0;
                for (order, 0..) |slot, i| {
                    const v = @as(f64, p.weights[token * TOPK + slot]) * @as(f64, c.fromBf16(p.x[(token * TOPK + slot) * sh.n + col]));
                    terms[i] = v;
                    magnitude += @abs(v);
                }
                const sum = if (sim) |s| rows_.sumIn(s.acc, s.order, &terms) else blk: {
                    var total: f64 = 0;
                    for (terms) |t| total += t;
                    break :blk total;
                };
                return .{ .value = bf(sum), .magnitude = magnitude };
            },
            else => {},
        }
        const pair = at / sh.n;
        const n = at % sh.n;
        const token = pair / TOPK;
        const slot = slotOf(@intCast(p.ids[pair]));
        const x = if (p.kind.isGu()) p.x[token * sh.k ..][0..sh.k] else p.x[pair * sh.k ..][0..sh.k];
        if (p.kind.isDown()) {
            const d = p.dot(slot, n, x, sim);
            return .{ .value = bf(d.value), .magnitude = d.magnitude };
        }
        const gd = p.dot(slot, 2 * n, x, sim);
        const ud = p.dot(slot, 2 * n + 1, x, sim);
        const gate = bf(gd.value);
        const up = bf(ud.value);
        const left = 4 * std.math.tanh(gate / 4) * (1 / (1 + @exp(-gate)));
        return .{ .value = bf(left * (25 * std.math.tanh(up / 25))), .magnitude = gd.magnitude + ud.magnitude };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.kind == .plan) {
            @memset(want, 0);
            @memset(mag, 0);
            return;
        }
        for (want, mag, 0..) |*v, *m, i| {
            const sample = p.evaluate(i, null);
            v.* = sample.value;
            m.* = sample.magnitude;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| v.* = p.evaluate(i, .{ .acc = acc, .order = order }).value;
    }

    /// A down's output is bf16 of an fp32 sum. The sum may differ from the exact one by the bound's `work`, and rounding
    /// two values that far apart can land on neighbours of the bf16 grid, one ulp apart: so the allowance is the generic
    /// one plus one more half ulp. A planted fault (a factor of 2, a flipped scale bit) misses by far more.
    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        // Where float64 gives inf or nan the kernel must match it exactly; the generic allowance there is not a number.
        for (allow, want) |*l, w| {
            if (!std.math.isFinite(w) or !std.math.isFinite(l.*)) {
                l.* = 0;
            } else if (p.kind.isDown()) {
                l.* += 0x1p-8 * (@abs(w) + l.*);
            }
        }
    }

    /// The plan's bytes are a pure function of the ids: members, items (poison past the count) and the count.
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (p.kind != .plan or gate != .shapes) return null;
        const routes = p.shape.rows * TOPK;
        const max_items = host.maxItems(p.call());
        const words = try p.a.alloc(i32, p.count() / 4);
        defer p.a.free(words);
        @memset(words, -1); // the poison the outputs start with
        const made = host.planHost(p.ids, @intCast(p.shape.n), words[0..routes], words[routes..][0 .. 3 * max_items]);
        words[routes + 3 * max_items] = @intCast(made);
        const got = try p.rig.outputBytes();
        if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(words))) return "the plan differs from the host grouping";
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    kernels: *std.ArrayList(g.Kernel),
    probes: [24]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// Every kernel and its planted twin, in the order of `Kind`; the plan twin first so the prompt twins can use it.
    pub fn collect(s: *Suite) !void {
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            s.probes[s.made] = try Probe.init(s.a, s.dev, kind, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
    }
};

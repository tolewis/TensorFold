//! Inkling 2 Mini's routed experts for long prompt chunks on XPU under the contract (kernels/xpu/inkling2/prefill_experts.cl):
//! the stable plan that sorts the routes by expert, the gather into sorted order, the four grouped GEMMs on DPAS (NVFP4 and
//! MXFP8 weights decoded in registers) against an independent float64 decode and dot, and the combine over the inverse map.
//! Every kernel has a planted twin that must go red at gate 3. The weights, factors and fills are those of inkling2_experts.zig.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const host = @import("inkling2_prefill_experts_host");
const rig_ = @import("contract_rig.zig");
const images = @import("xpu_contract_images");

const c = core.spec;
const g = core.gates;
const rows_ = core.rows;
const bf = core.bf16_round.bf16;

const HID: usize = 2304;
const FF: usize = 1152;
const E: usize = host.experts;
const TOPK: usize = host.top_k;
const MT: usize = host.item_rows;
/// The experts a test fills with weights; every id a call routes to is one of them, spread over the 128.
const POOL: usize = 16;

fn poolId(i: usize) usize {
    return 3 + 8 * i;
}
fn slotOf(id: usize) usize {
    return (id - 3) / 8;
}

pub const Kind = enum {
    plan,
    gather,
    nv_gu,
    nv_down,
    mx_gu,
    mx_down,
    combine,

    fn which(k: Kind) host.Kernels.Which {
        return switch (k) {
            .plan => .plan,
            .gather => .gather,
            .nv_gu => .nvfp4_gu,
            .nv_down => .nvfp4_down,
            .mx_gu => .mxfp8_gu,
            .mx_down => .mxfp8_down,
            .combine => .combine,
        };
    }
    fn isNv(k: Kind) bool {
        return k == .nv_gu or k == .nv_down;
    }
    fn isGu(k: Kind) bool {
        return k == .nv_gu or k == .mx_gu;
    }
    fn isDown(k: Kind) bool {
        return k == .nv_down or k == .mx_down;
    }
    fn isMatmul(k: Kind) bool {
        return k.isGu() or k.isDown();
    }
    /// The kernels whose output is checked byte for byte against a host computation.
    fn isRaw(k: Kind) bool {
        return k == .plan or k == .gather;
    }
};

/// The control twin's symbol prefix: one kernel of prefill_experts.cl under its own name.
pub fn controlPrefix(k: Kind) []const u8 {
    return switch (k) {
        .plan => "tf_ctl_pf_plan_",
        .gather => "tf_ctl_pf_gather_",
        .nv_gu => "tf_ctl_pf_nvfp4_gu_",
        .nv_down => "tf_ctl_pf_nvfp4_down_",
        .mx_gu => "tf_ctl_pf_mxfp8_gu_",
        .mx_down => "tf_ctl_pf_mxfp8_down_",
        .combine => "tf_ctl_pf_combine_",
    };
}

/// The argument counts gate 5 checks a loaded kernel against.
pub fn argsOf(symbol: []const u8) ?u32 {
    const table = [_]struct { []const u8, u32 }{
        .{ "plan", 7 },
        .{ "gather", 5 },
        .{ "nvfp4_gu", 10 },
        .{ "nvfp4_down", 10 },
        .{ "mxfp8_gu", 8 },
        .{ "mxfp8_down", 8 },
        .{ "combine", 7 },
    };
    if (!std.mem.startsWith(u8, symbol, "tf_ink2_pf_") and !std.mem.startsWith(u8, symbol, "tf_ctl_pf_")) return null;
    for (table) |e| if (std.mem.endsWith(u8, symbol, e[0])) return e[1];
    return null;
}

// Call shapes: `rows` tokens, so rows * 8 routes. 72 rows route 72 times to the pool's first expert: two items (64 and 8 routes).
const gu_served = [_]c.Shape{ .{ .rows = 16, .n = FF, .k = HID }, .{ .rows = 40, .n = FF, .k = HID }, .{ .rows = 72, .n = FF, .k = HID } };
const down_served = [_]c.Shape{ .{ .rows = 16, .n = HID, .k = FF }, .{ .rows = 40, .n = HID, .k = FF }, .{ .rows = 72, .n = HID, .k = FF } };
const combine_served = [_]c.Shape{ .{ .rows = 1, .n = HID, .k = TOPK }, .{ .rows = 16, .n = HID, .k = TOPK }, .{ .rows = 72, .n = HID, .k = TOPK } };
const plan_served = [_]c.Shape{ .{ .rows = 16, .n = E, .k = TOPK }, .{ .rows = 72, .n = E, .k = TOPK }, .{ .rows = 1, .n = E, .k = TOPK } };
const gather_served = [_]c.Shape{ .{ .rows = 16, .n = HID, .k = TOPK }, .{ .rows = 72, .n = HID, .k = TOPK } };
// A planted twin is red at its first shape, so one small shape is the whole proof and the run stays short.
const gu_control = [_]c.Shape{.{ .rows = 16, .n = FF, .k = HID }};
const down_control = [_]c.Shape{.{ .rows = 16, .n = HID, .k = FF }};
const combine_control = [_]c.Shape{.{ .rows = 16, .n = HID, .k = TOPK }};
const plan_control = [_]c.Shape{.{ .rows = 16, .n = E, .k = TOPK }};
const gather_control = [_]c.Shape{.{ .rows = 16, .n = HID, .k = TOPK }};

fn depth(s: c.Shape) usize {
    return s.k + 3;
}
fn zero(_: c.Shape) usize {
    return 0;
}

fn title(k: Kind) []const u8 {
    return switch (k) {
        .plan => "xpu inkling2 prefill experts tf_ink2_pf_plan",
        .gather => "xpu inkling2 prefill experts tf_ink2_pf_gather",
        .nv_gu => "xpu inkling2 prefill experts tf_ink2_pf_nvfp4_gu",
        .nv_down => "xpu inkling2 prefill experts tf_ink2_pf_nvfp4_down",
        .mx_gu => "xpu inkling2 prefill experts tf_ink2_pf_mxfp8_gu",
        .mx_down => "xpu inkling2 prefill experts tf_ink2_pf_mxfp8_down",
        .combine => "xpu inkling2 prefill experts tf_ink2_pf_combine",
    };
}

fn controlName(k: Kind) []const u8 {
    return switch (k) {
        .plan => "control: xpu inkling2 prefill experts tf_ink2_pf_plan route index",
        .gather => "control: xpu inkling2 prefill experts tf_ink2_pf_gather source row",
        .nv_gu => "control: xpu inkling2 prefill experts tf_ink2_pf_nvfp4_gu up row",
        .nv_down => "control: xpu inkling2 prefill experts tf_ink2_pf_nvfp4_down factor",
        .mx_gu => "control: xpu inkling2 prefill experts tf_ink2_pf_mxfp8_gu up row",
        .mx_down => "control: xpu inkling2 prefill experts tf_ink2_pf_mxfp8_down scale",
        .combine => "control: xpu inkling2 prefill experts tf_ink2_pf_combine route-slot order",
    };
}

pub fn spec(k: Kind, control: bool) c.Spec {
    var s = c.Spec{
        .name = if (control) controlName(k) else title(k),
        .accum = .f32,
        .depth = depth,
        .ops = if (k.isGu()) 12 else 3,
        .mids = if (k.isGu()) 3 else 0,
        .out = .bf16,
        .served = &gu_served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and explicit BF16 nodes" },
    };
    s.served = switch (k) {
        .nv_gu, .mx_gu => if (control) &gu_control else &gu_served,
        .nv_down, .mx_down => if (control) &down_control else &down_served,
        .combine => if (control) &combine_control else &combine_served,
        .plan => if (control) &plan_control else &plan_served,
        .gather => if (control) &gather_control else &gather_served,
    };
    if (k.isRaw()) {
        s.accum = .none;
        s.out = .raw;
        s.ops = 0;
        s.depth = zero;
    }
    if (k == .combine) s.sums = .{ .f16 = 0, .bf16 = 100, .why = "bounded cancellation is exact in fp32 but loses its residual in bf16" };
    return s;
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

/// The stable plan on the host: routes sorted by expert (ties in route order), items of at most `MT` rows of one expert.
fn planHost(ids: []const i32, order: []i32, inv: []i32, items: []i32) u32 {
    var counts: [E]u32 = @splat(0);
    for (ids) |id| counts[@intCast(id)] += 1;
    var at: u32 = 0;
    var made: u32 = 0;
    for (0..E) |e| {
        var first: u32 = 0;
        while (first < counts[e]) : (first += MT) {
            items[3 * made] = @intCast(e);
            items[3 * made + 1] = @intCast(at + first);
            items[3 * made + 2] = @intCast(@min(MT, counts[e] - first));
            made += 1;
        }
        var put = at;
        for (ids, 0..) |id, p| if (@as(usize, @intCast(id)) == e) {
            order[put] = @intCast(p);
            inv[p] = @intCast(put);
            put += 1;
        };
        at += counts[e];
    }
    return made;
}

const Sim = struct { acc: c.Accum, order: g.Order };
const Dot = struct { value: f64, magnitude: f64 };
const Sample = struct { value: f64, magnitude: f64 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    ks: *const host.Kernels, // the kernel under test (a control: its twin, the rest production)
    kind: Kind,
    ctl: bool,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    // The pool's weights, kept between prepares of one fill: raw checkpoint bytes and their float64-checked bf16 decode.
    cached: ?g.Fill = null,
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
    order: []i32 = &.{}, // the host plan's
    inv: []i32 = &.{},
    items: []i32 = &.{},
    made: u32 = 0,
    // Device addresses.
    d_x: ?*anyopaque = null,
    d_xs: ?*anyopaque = null,
    d_w: ?*anyopaque = null,
    d_s: ?*anyopaque = null,
    d_amax: ?*anyopaque = null,
    d_den: ?*anyopaque = null,
    d_ids: ?*anyopaque = null,
    d_weights: ?*anyopaque = null,
    d_out: ?*anyopaque = null,
    d_order: ?*anyopaque = null,
    d_inv: ?*anyopaque = null,
    d_items: ?*anyopaque = null,
    d_count: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, ks: *const host.Kernels, kind: Kind, control: bool) Probe {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .ks = ks, .kind = kind, .ctl = control, .arena = .init(a) };
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
        p.order = &.{};
        p.inv = &.{};
        p.items = &.{};
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.ctl), .ctx = p, .vt = if (p.kind.isRaw()) &vt_raw else &vt, .control = if (p.ctl) .shapes else null };
    }

    /// Every numeric kernel compares to the float64 reference; the plan and the gather are checked byte for byte instead.
    const vt = blk: {
        var t = g.hooks(Probe);
        t.release = releaseAll;
        t.check = null;
        break :blk t;
    };
    const vt_raw = blk: {
        var t = g.hooks(Probe);
        t.release = releaseAll;
        break :blk t;
    };

    fn format(p: *const Probe) host.Format {
        return if (p.kind.isNv()) .nvfp4 else .mxfp8;
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
        if (s.rows == 0 or s.rows > 128) return false;
        if (k.isGu()) return s.n == FF and s.k == HID;
        if (k.isDown()) return s.n == HID and s.k == FF;
        if (k == .combine) return s.n % 64 == 0 and s.n <= HID and s.k == TOPK;
        if (k == .gather) return s.n == HID and s.k == TOPK;
        return s.n == E and s.k == TOPK; // plan
    }

    /// The pool's weights and scales for `fill`, decoded; regenerated only when the fill changes.
    fn ensureWeights(p: *Probe, fill: g.Fill) !void {
        if (p.cached) |f| if (f == fill) return;
        p.dropCache();
        const nv = p.kind.isNv();
        const rpe = p.rowsPerExpert();
        const cb = p.codeBytes();
        const blk = p.blockSize();
        p.pw = try p.a.alloc(u8, POOL * rpe * cb);
        p.ps = try p.a.alloc(u8, POOL * rpe * (p.shape.k / blk));
        var prng = std.Random.DefaultPrng.init(0x1c2b5 + @as(u64, @intFromEnum(p.kind)));
        const rng = prng.random();
        rng.bytes(p.pw);
        // scales that keep a dot over 2,304 terms near 1 to 10, so the activation stays off its saturated ends
        for (p.ps, 0..) |*v, i| v.* = if (nv) @intCast(0x30 + (i % 17)) else @intCast(119 + (i % 5));
        if (!nv) for (p.pw) |*v| if (v.* & 0x7f == 0x7f) {
            v.* = (v.* & 0x80) | 0x7e; // a stored NaN would make every sum NaN
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
        try p.decodePool();
        p.cached = fill;
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
        // top 8 distinct of the pool a token; the first pool expert is in every token's picks, so it gets one long run
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
        const max_items = host.maxItems(@intCast(shape.rows));
        p.order = try a.alloc(i32, routes);
        p.inv = try a.alloc(i32, routes);
        p.items = try a.alloc(i32, 3 * max_items);
        @memset(p.items, -1);
        p.made = planHost(p.ids, p.order, p.inv, p.items);
        if (k.isMatmul()) {
            try p.ensureWeights(fill);
            // a GU reads one row a token (the gather sorts them); a down reads its routes already in sorted order
            const nx = if (k.isGu()) shape.rows * shape.k else routes * shape.k;
            p.x = try a.alloc(u16, nx);
            for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1) / 48);
            if (fill == .stress) for (p.x, 0..) |*v, i| {
                // a down's inputs change sign half way, so its sums cancel; a GU's tokens keep one sign each
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
            p.d_xs = try p.rig.scratch(routes * shape.k * 2);
            p.d_order = try p.rig.scratch(routes * 4);
            p.d_inv = try p.rig.scratch(routes * 4);
            p.d_items = try p.rig.scratch(max_items * 12);
            p.d_count = try p.rig.scratch(4);
        } else if (k == .combine) {
            // down outputs of every route (the host puts them in sorted order), the router's weights, and the ids that order them
            const by_route = try a.alloc(u16, routes * shape.n);
            p.weights = try a.alloc(f32, routes);
            for (by_route, 0..) |*v, i| v.* = c.toBf16(@floatFromInt(1 + i % 8));
            if (p.ctl) {
                // Slots 0 and 1 add to 2^128, which overflows fp32, when summed in slot order. Ascending expert order
                // puts slot 2 (-2^127) between them, so the true sum is finite.
                const pattern = [_]usize{ 2, 0, 1, 7, 3, 6, 4, 5 };
                for (p.ids, 0..) |*id, i| id.* = @intCast(poolId(pattern[i % TOPK]));
                const cw = [_]f32{ 0x1p127, 0x1p127, -0x1p127, 1, 1, 1, 1, 1 };
                for (p.weights, 0..) |*v, i| v.* = cw[i % TOPK];
                for (by_route) |*v| v.* = c.toBf16(1);
            } else if (fill == .stress) {
                const terms = [_]f32{ 1024, 1, 1, 1, 1, 1, 1, -1029 };
                for (by_route) |*v| v.* = c.toBf16(1);
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
                by_route[0] = c.toBf16(std.math.nan(f32));
                by_route[by_route.len / 2] = c.toBf16(std.math.inf(f32));
                by_route[by_route.len - 1] = c.toBf16(-std.math.inf(f32));
            }
            // the host plan again, for the ids this call routes to
            p.made = planHost(p.ids, p.order, p.inv, p.items);
            p.x = by_route; // route order: the reference's
            const sorted = try a.alloc(u16, routes * shape.n);
            for (0..routes) |r| @memcpy(sorted[@as(usize, @intCast(p.inv[r])) * shape.n ..][0..shape.n], by_route[r * shape.n ..][0..shape.n]);
            p.d_x = try p.rig.input(std.mem.sliceAsBytes(sorted));
            p.d_inv = try p.rig.input(std.mem.sliceAsBytes(p.inv));
            p.d_ids = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.d_weights = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            p.d_out = try p.rig.output(shape.rows * shape.n * 2);
        } else if (k == .gather) {
            p.x = try a.alloc(u16, shape.rows * shape.n);
            for (p.x, 0..) |*v, i| v.* = @truncate((i *% 2654435761) >> 7);
            p.d_x = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.d_order = try p.rig.input(std.mem.sliceAsBytes(p.order));
            p.d_out = try p.rig.output(routes * shape.n * 2);
        } else {
            p.d_ids = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.d_order = try p.rig.output(routes * 4);
            p.d_inv = try p.rig.output(routes * 4);
            p.d_items = try p.rig.output(max_items * 12);
            p.d_count = try p.rig.output(4);
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.rig.dev.s;
        const ks = p.ks;
        const rows: u32 = @intCast(p.shape.rows);
        const kdim: u32 = @intCast(p.shape.k);
        const n: u32 = @intCast(p.shape.n);
        switch (p.kind) {
            .plan => try host.plan(ks, s, p.d_ids, p.d_order, p.d_inv, p.d_items, p.d_count, rows),
            .gather => try host.gather(ks, s, p.d_x, p.d_order, p.d_out, rows, n),
            .nv_gu, .mx_gu, .nv_down, .mx_down => {
                // the grouping and the gather are the production kernels' (a twin swaps only its own), the GEMM is the one under test
                try host.plan(ks, s, p.d_ids, p.d_order, p.d_inv, p.d_items, p.d_count, rows);
                const w: host.Weights = .{ .w = p.d_w, .scales = p.d_s, .amax = p.d_amax, .den = p.d_den };
                if (p.kind.isGu()) {
                    try host.gather(ks, s, p.d_x, p.d_order, p.d_xs, rows, kdim);
                    try host.gu(ks, s, p.format(), p.d_xs, w, p.d_items, p.d_count, p.d_out, rows, kdim, n);
                } else {
                    try host.down(ks, s, p.format(), p.d_x, w, p.d_items, p.d_count, p.d_out, rows, kdim, n);
                }
            },
            .combine => try host.combine(ks, s, p.d_x, p.d_inv, p.d_ids, p.d_weights, p.d_out, rows, n),
        }
        try p.rig.dev.s.synchronize();
    }

    pub fn count(p: *const Probe) usize {
        const sh = p.shape;
        return switch (p.kind) {
            .combine => sh.rows * sh.n,
            .plan => sh.rows * TOPK * 8 + host.maxItems(@intCast(sh.rows)) * 12 + 4,
            .gather => sh.rows * TOPK * sh.n * 2,
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
        if (p.kind == .combine) {
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
        }
        // the output is in sorted order: sorted row `at / n` holds the route the host plan puts there
        const sorted = at / sh.n;
        const n = at % sh.n;
        const route: usize = @intCast(p.order[sorted]);
        const token = route / TOPK;
        const slot = slotOf(@intCast(p.ids[route]));
        const x = if (p.kind.isGu()) p.x[token * sh.k ..][0..sh.k] else p.x[sorted * sh.k ..][0..sh.k];
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
        if (p.kind.isRaw()) {
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

    /// The plan's bytes (the order, the inverse, the items with poison past the count, the count) and the gather's rows
    /// are pure functions of the ids: compared exactly.
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes) return null;
        const got = try p.rig.outputBytes();
        const routes = p.shape.rows * TOPK;
        if (p.kind == .gather) {
            const want = try p.a.alloc(u16, routes * p.shape.n);
            defer p.a.free(want);
            for (0..routes) |s| @memcpy(want[s * p.shape.n ..][0..p.shape.n], p.x[@as(usize, @intCast(p.order[s])) / TOPK * p.shape.n ..][0..p.shape.n]);
            if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(want))) return "a sorted row is not its token's activation row";
            return null;
        }
        const max_items = host.maxItems(@intCast(p.shape.rows));
        const words = try p.a.alloc(i32, 2 * routes + 3 * max_items + 1);
        defer p.a.free(words);
        @memset(words, -1); // the poison the outputs start with
        @memcpy(words[0..routes], p.order);
        @memcpy(words[routes..][0..routes], p.inv);
        @memcpy(words[2 * routes ..][0 .. 3 * p.made], p.items[0 .. 3 * p.made]);
        words[2 * routes + 3 * max_items] = @intCast(p.made);
        if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(words))) return "the plan differs from the host grouping";
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    production: host.Kernels,
    twins: [7]host.Kernels = undefined,
    twins_made: usize = 0,
    probes: std.ArrayList(*Probe) = .empty,

    /// Loads both images into `dev`.
    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Suite {
        try dev.addImage("inkling2_prefill_experts", &images.inkling2_prefill_experts);
        try dev.addImage("contract_inkling2_prefill_experts_controls", &images.contract_inkling2_prefill_experts_controls);
        const production = try host.Kernels.load(dev.named("inkling2_prefill_experts").?.*);
        return .{ .a = a, .dev = dev, .production = production };
    }

    /// Every kernel and, after it, its planted twin: the production kernels with the one kernel swapped for the twin.
    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel)) !void {
        const planted = dev_module(s);
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            var ks: *const host.Kernels = &s.production;
            if (control) {
                s.twins[s.twins_made] = try host.Kernels.load(s.dev.named("inkling2_prefill_experts").?.*);
                try s.twins[s.twins_made].swap(planted.*, controlPrefix(kind), kind.which());
                ks = &s.twins[s.twins_made];
                s.twins_made += 1;
            }
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = Probe.init(s.a, s.dev, ks, kind, control);
            errdefer probe.deinit();
            try s.probes.append(s.a, probe);
            try kernels.append(s.a, probe.kernel());
        };
    }

    fn dev_module(s: *Suite) *xpu.Module {
        return s.dev.named("contract_inkling2_prefill_experts_controls").?;
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
        for (s.twins[0..s.twins_made]) |*t| t.deinit();
        s.production.deinit();
    }
};

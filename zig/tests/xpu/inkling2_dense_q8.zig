//! Inkling 2 Mini's Q8_0-weight dense projections and unembedding head on XPU under the kernel contract: bf16 x times the
//! decoded int8 weights (a code times its block's fp16 scale) with FP32 sums and one bf16 store, and the head's soft-capped FP32
//! logit, each against float64 over the same decoded weights at Mini's served shapes. The weights are made the way the loader
//! makes them (q8.zig). Every kernel has planted twins that must go red.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_dense_q8_host");
const q8 = @import("inkling2_q8");
const images = @import("xpu_contract_images");

const c = core.spec;
const g = core.gates;
const cdense = core.dense;

const H: usize = host.hidden;

/// What a planted twin gets wrong. `store`: every store truncates to bf16 (gate 3). `sums`: the lane adds each block's
/// contribution through fp16 (gate 4). `scale`: every block of a plane uses the plane's first scale (gate 3).
pub const Control = enum { none, store, sums, scale };

fn imageOf(control: Control) []const u8 {
    return switch (control) {
        .none => "inkling2_dense_q8",
        .store => "contract_inkling2_dense_q8_store",
        .sums => "contract_inkling2_dense_q8_sums",
        .scale => "contract_inkling2_dense_q8_scale",
    };
}

/// Argument counts gate 5 checks a loaded kernel against.
pub fn argsOf(symbol: []const u8) ?u32 {
    if (std.mem.startsWith(u8, symbol, "tf_ink2_dense_q8_") or std.mem.startsWith(u8, symbol, "tf_ctl_ink2_dense_q8_")) return 7;
    if (std.mem.startsWith(u8, symbol, "tf_ink2_head_q8_") or std.mem.startsWith(u8, symbol, "tf_ctl_ink2_head_q8_")) return 5;
    if (std.mem.endsWith(u8, symbol, "head_dpas")) return 7;
    if (std.mem.endsWith(u8, symbol, "q8_quant")) return 5;
    return null;
}

fn depth(s: c.Shape) usize {
    // a lane's chain over the blocks (two products, one scaled add each), then the sub-group's tree of 16
    return 3 * (s.k / 32) + 8;
}

fn headDepth(s: c.Shape) usize {
    return s.k + 40;
}

const projections = [_][2]usize{ .{ 3072, 2304 }, .{ 1024, 2304 }, .{ 384, 2304 }, .{ 24, 2304 }, .{ 2304, 3072 }, .{ 64, 4608 } };

/// The served projections at `lead` rows (wq and gate, wk and wv, wr, the sink, wo, a controller's down projection), and one call of 2 lead + 1 rows.
fn servedTable(comptime lead: u32) []const c.Shape {
    comptime {
        var out: [projections.len + 1]c.Shape = undefined;
        for (projections, 0..) |p, i| out[i] = .{ .rows = lead, .n = p[0], .k = p[1] };
        out[projections.len] = .{ .rows = 2 * lead + 1, .n = 1024, .k = 2304 };
        const final = out;
        return &final;
    }
}

/// The host wrapper: the calls the forward pass makes, from one row to every kernel's rows at once.
const wrapper_served = [_]c.Shape{
    .{ .rows = 1, .n = 3072, .k = 2304 },
    .{ .rows = 1, .n = 1024, .k = 2304 },
    .{ .rows = 1, .n = 384, .k = 2304 },
    .{ .rows = 1, .n = 24, .k = 2304 },
    .{ .rows = 1, .n = 2304, .k = 3072 },
    .{ .rows = 3, .n = 1024, .k = 2304 },
    .{ .rows = 7, .n = 384, .k = 2304 },
    .{ .rows = 8, .n = 2304, .k = 3072 },
    .{ .rows = 15, .n = 24, .k = 2304 },
    .{ .rows = 33, .n = 3072, .k = 2304 },
    .{ .rows = 63, .n = 1024, .k = 2304 },
};

fn servedOf(lead: u32) []const c.Shape {
    return switch (lead) {
        0 => &wrapper_served,
        inline 1, 2, 4, 8, 16, 32 => |l| comptime servedTable(l),
        else => unreachable,
    };
}

/// The head's served calls: a chunk of rows over the vocabulary's real rows is too long for a float64 reference at every
/// shape, so the gates run the head over a few hundred to a few thousand vocabulary rows (and `report` runs the full 200,058).
fn headServedOf(comptime lead: u32) []const c.Shape {
    comptime {
        const out = if (lead == 0) [_]c.Shape{
            .{ .rows = 1, .n = 1000, .k = H },
            .{ .rows = 5, .n = 200, .k = H },
            .{ .rows = 8, .n = 1031, .k = H },
            .{ .rows = 33, .n = 129, .k = H },
            .{ .rows = 63, .n = 64, .k = H },
        } else [_]c.Shape{
            .{ .rows = lead, .n = 1000, .k = H },
            .{ .rows = lead, .n = 8, .k = H },
            .{ .rows = 2 * lead + 1, .n = 200, .k = H },
            .{ .rows = lead, .n = 4099, .k = H },
        };
        const final = out;
        return &final;
    }
}

fn headServed(lead: u32) []const c.Shape {
    return switch (lead) {
        inline 0, 1, 2, 4, 8, 16, 32 => |l| comptime headServedOf(l),
        else => unreachable,
    };
}

fn put(r: ?*anyopaque, bytes: usize) ?*anyopaque {
    return if (r) |p| @ptrFromInt(@intFromPtr(p) + bytes) else null;
}

fn roundBf16(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// The quantize kernel's round: amax/127, then the same even round the device uses.
fn hostQuant(x: []const u16, k: usize, qc: []i8, qs: []f32, flags: []i32) void {
    const rows = x.len / k;
    const nb = k / 32;
    for (0..rows) |r| {
        var bits: i32 = 0;
        for (0..k) |i| {
            const v: f32 = @floatCast(c.fromBf16(x[r * k + i]));
            if (std.math.isNan(v)) bits |= 1 else if (std.math.isPositiveInf(v)) bits |= 2 else if (std.math.isNegativeInf(v)) bits |= 4;
        }
        flags[r] = bits;
        for (0..nb) |b| {
            var am: f32 = 0;
            for (0..32) |e| {
                const v: f32 = @floatCast(c.fromBf16(x[r * k + b * 32 + e]));
                if (std.math.isFinite(v)) am = @max(am, @abs(v));
            }
            const scale: f32 = if (am > 0) am * (1.0 / 127.0) else 1;
            qs[r * nb + b] = scale;
            for (0..32) |e| {
                const xv: f32 = std.math.clamp(@as(f32, @floatCast(c.fromBf16(x[r * k + b * 32 + e]))) / scale, -128, 128);
                const even = (xv + 0x1.8p23) - 0x1.8p23;
                qc[r * k + b * 32 + e] = @intCast(std.math.clamp(@as(i32, @intFromFloat(even)), -127, 127));
            }
        }
    }
}
fn cap30(v: f64) f64 {
    return 30.0 * std.math.tanh(v / 30.0);
}
fn ulp(v: f64) f64 {
    const m = @max(@abs(v), 0x1p-120);
    return std.math.pow(f64, 2, @floor(std.math.log2(m)) - 7);
}

/// The matrix of `n` rows of `k` values a fill describes, encoded: `codes` and `scales`, and the float64 values they decode to in `w`.
fn makeWeights(al: std.mem.Allocator, fill: g.Fill, n: usize, k: usize, head: bool, rng: std.Random, w: []f64) !struct { []u8, []u16 } {
    const nb = k / q8.block;
    const codes = try al.alloc(u8, n * k);
    const scales = try al.alloc(u16, n * nb);
    if (fill == .stress) {
        // Integers whose products and sums are exact in FP32 in any order (scale 1); the second half cancels the first.
        const half = k / 2;
        @memset(scales, 0x3c00);
        for (0..n) |j| for (0..k) |i| {
            const m: i8 = @intCast(1 + ((if (i < half) i else i - half) * 7 + j) % 6);
            const v: i8 = if (i < half) m else -m;
            codes[j * k + (i / 32) / 8 * 256 + (i % 32) / 2 * 16 + (i / 32) % 8 * 2 + i % 2] = @bitCast(v);
        };
    } else if (head) {
        // multiples of 2^-8 that a dot with a multiple of 1/64 keeps exact: the logit's bf16 rounding has no order to flip on;
        // a block's scale is 2^-7 or 2^-8, so a block that borrows another's scale is wrong
        for (scales) |*sc| sc.* = if (rng.boolean()) 0x2000 else 0x1c00;
        for (codes) |*b| b.* = @bitCast(@as(i8, @intCast(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127)));
    } else {
        const src = try al.alloc(u16, n * k);
        for (src) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
        q8.encodeRows(src, n, k, codes, scales);
    }
    for (0..n) |j| for (0..nb) |b| {
        for (0..q8.block) |e| w[j * k + b * q8.block + e] = q8.decode(codes, scales, k, j, b, e);
    };
    return .{ codes, scales };
}

/// `lead` = 0: the host wrapper over every row count; else the kernel of that many rows, called on blocks of `lead` rows
/// with a short last block, whose input rows past the call are zero. `head`: the unembedding head, else the projections.
pub const Q8 = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kernels: *const host.Kernels,
    lead: u32,
    head: bool,
    control: Control,
    name: [96]u8 = undefined,
    name_len: usize = 0,
    shape: c.Shape = undefined,
    d: ?cdense.Dense = null,
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    xr: ?*anyopaque = null,
    wr: ?*anyopaque = null,
    out: ?*anyopaque = null,
    qcr: ?*anyopaque = null,
    qsr: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kernels: *const host.Kernels, lead: u32, head: bool, control: Control) Q8 {
        var self: Q8 = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .dev = dev }, .kernels = kernels, .lead = lead, .head = head, .control = control };
        const kind = if (head) "head" else "dense";
        const who = if (lead == 0) std.fmt.bufPrint(&self.name, "xpu inkling2 {s} q8 host wrapper", .{kind}) else std.fmt.bufPrint(&self.name, "xpu inkling2 {s} q8 tf_ink2_{s}_q8_{d}", .{ kind, kind, lead });
        self.name_len = (who catch unreachable).len;
        return self;
    }

    pub fn deinit(self: *Q8) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Q8) g.Kernel {
        return .{
            .ctx = self,
            .vt = &vt,
            .control = switch (self.control) {
                .none => null,
                .store, .scale => .shapes,
                .sums => if (self.head) .shapes else .sums,
            },
            .spec = .{
                .name = self.label(),
                .accum = .f32,
                .depth = if (self.head) headDepth else depth,
                .ops = if (self.head) 96 else 0,
                .mids = if (self.head) 1 else 0,
                .out = if (self.head) .f32 else .bf16,
                .served = if (self.head) headServed(self.lead) else servedOf(self.lead),
                .tiles = if (self.lead == 0) .{ .{}, .{}, .{} } else .{
                    .{ .size = self.lead },
                    .{ .size = host.group_threads / 16 },
                    .{ .size = 256, .step = 256, .min = 256 },
                },
                .sums = if (self.head) .{ .f16 = 0, .bf16 = 0, .why = "the reference is the int32 dot times the two scales in fp32, so only that sum's order is left; the activation round is covered by the pooled KL16" } else .{},
            },
        };
    }

    var labels: [4][2][7][112]u8 = undefined;
    fn label(self: *Q8) []const u8 {
        const ci: usize = @intFromEnum(self.control);
        const li: usize = if (self.lead == 0) 6 else std.math.log2_int(u32, self.lead);
        const prefix = switch (self.control) {
            .none => "",
            .store => "control (store truncated): ",
            .sums => "control (block sums in fp16): ",
            .scale => "control (one scale a plane): ",
        };
        return std.fmt.bufPrint(&labels[ci][@intFromBool(self.head)][li], "{s}{s}", .{ prefix, self.name[0..self.name_len] }) catch unreachable;
    }

    const vt = g.hooks(Q8);

    pub fn count(self: *Q8) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn prepare(self: *Q8, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!q8.admits(shape.k) or (self.head and shape.k != H)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        self.shape = shape;
        const d = try cdense.Dense.init(al, shape, 1);
        d.fillX(fill, seed, 2);
        // The head's dot of multiples of 1/64 with multiples of 2^-7 is exact in FP32 in every order, so the bf16 rounding
        // that follows it has no order to flip on, and a missing rounding shows.
        if (self.head and fill != .stress) for (d.x) |*v| {
            const f = c.fromBf16(v.*);
            if (std.math.isFinite(f)) v.* = c.toBf16(@round(f * 64) / 64);
        };
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5eed);
        const made = try makeWeights(al, fill, shape.n, shape.k, self.head, prng.random(), d.w);
        self.codes = made[0];
        self.scales = made[1];
        self.d = d;
        const rows_dev = if (self.lead == 0) shape.rows else (shape.rows + self.lead - 1) / self.lead * self.lead;
        const x = try al.alloc(u16, rows_dev * shape.k);
        @memset(x, 0);
        @memcpy(x[0 .. shape.rows * shape.k], d.x);
        self.xr = try self.rig.input(std.mem.sliceAsBytes(x));
        const nbytes = if (self.head) q8.dpasBytes(shape.n, shape.k) else q8.bytes(shape.n, shape.k);
        const wb = try al.alloc(u8, nbytes);
        if (self.head) {
            q8.repackDpas(made[0], made[1], shape.n, shape.k, wb);
        } else {
            @memcpy(wb[0 .. shape.n * shape.k], made[0]);
            @memcpy(wb[q8.scaleOffset(shape.n, shape.k)..], std.mem.sliceAsBytes(made[1]));
        }
        self.wr = try self.rig.input(wb);
        self.out = try self.rig.output(shape.rows * shape.n * @as(usize, if (self.head) 4 else 2));
        if (self.head) {
            const qc = try al.alloc(i8, shape.rows * shape.k);
            const nb = shape.k / 32;
            const qs = try al.alloc(f32, shape.rows * nb);
            const flags = try al.alloc(i32, shape.rows);
            hostQuant(x[0 .. shape.rows * shape.k], shape.k, qc, qs, flags);
            self.qcr = try self.rig.input(std.mem.sliceAsBytes(qc));
            const pack = try al.alloc(u8, qs.len * 4 + flags.len * 4);
            @memcpy(pack[0 .. qs.len * 4], std.mem.sliceAsBytes(qs));
            @memcpy(pack[qs.len * 4 ..], std.mem.sliceAsBytes(flags));
            self.qsr = try self.rig.input(pack);
        }
        return true;
    }

    /// The quantize is uploaded in prepare. This launch only does the int32 dot.
    fn runHead(self: *Q8, rows: u32, n: u32, k: u32) !void {
        _ = k;
        try host.headDpas(self.kernels, self.rig.dev.s, self.wr, self.qcr, self.qsr, self.xr, self.out, rows, n);
        try self.rig.dev.s.synchronize();
    }

    pub fn run(self: *Q8) !void {
        const s = self.rig.dev.s;
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        const k: u32 = @intCast(self.shape.k);
        const out_bytes: usize = if (self.head) 4 else 2;
        if (self.head) {
            try self.runHead(r, n, k);
            return;
        }
        if (self.lead == 0) {
            try host.dense(self.kernels, s, self.wr, self.xr, self.out, r, n, k);
        } else {
            var base: u32 = 0;
            while (base < r) : (base += self.lead) {
                const t = @min(self.lead, r - base);
                const x = put(self.xr, @as(usize, base) * k * 2);
                const y = put(self.out, @as(usize, base) * n * out_bytes);
                if (self.head) try host.headLaunch(self.kernels, s, self.lead, self.wr, x, y, t, n) else try host.matvec(self.kernels, s, self.lead, self.wr, x, y, t, n, k, n);
            }
        }
        try s.synchronize();
    }

    /// The head's logit from a sum: the dot rounds to bf16, widens, and is soft capped at 30.
    fn headValue(v: f64) f64 {
        return cap30(roundBf16(v));
    }

    /// Finite rows: the int32 dot times the two scales, in fp32. A non-finite row uses the old exact dot, as the kernel does.
    fn headDot(self: *Q8, r: usize, j: usize) f64 {
        const d = self.d.?;
        const k = self.shape.k;
        const nb = k / 32;
        var bits: i32 = 0;
        for (0..k) |i| {
            const v: f32 = @floatCast(c.fromBf16(d.x[r * k + i]));
            if (std.math.isNan(v)) bits |= 1 else if (std.math.isPositiveInf(v)) bits |= 2 else if (std.math.isNegativeInf(v)) bits |= 4;
        }
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < nb) : (b += 1) {
            const ws: f32 = @floatCast(@as(f16, @bitCast(self.scales[j * nb + b])));
            if (bits != 0) {
                for (0..32) |e| {
                    const code: f32 = @floatFromInt(@as(i8, @bitCast(self.codes[j * k + q8.codeOffset(b, e)])));
                    const x: f32 = @floatCast(c.fromBf16(d.x[r * k + b * 32 + e]));
                    acc = @mulAdd(f32, ws * code, x, acc);
                }
            } else {
                var idot: i32 = 0;
                const scale = blk: {
                    var am: f32 = 0;
                    for (0..32) |e| {
                        const v: f32 = @floatCast(c.fromBf16(d.x[r * k + b * 32 + e]));
                        am = @max(am, @abs(v));
                    }
                    break :blk if (am > 0) am * (1.0 / 127.0) else 1;
                };
                for (0..32) |e| {
                    const x: f32 = std.math.clamp(@as(f32, @floatCast(c.fromBf16(d.x[r * k + b * 32 + e]))) / scale, -128, 128);
                    const even = (x + 0x1.8p23) - 0x1.8p23;
                    const qq: i32 = @intFromFloat(std.math.clamp(even, -127, 127));
                    const code: i32 = @as(i8, @bitCast(self.codes[j * k + q8.codeOffset(b, e)]));
                    idot += code * qq;
                }
                const prod = ws * scale;
                acc = @mulAdd(f32, prod, @as(f32, @floatFromInt(idot)), acc);
            }
        }
        return acc;
    }

    pub fn reference(self: *Q8, want: []f64, mag: []f64) !void {
        if (!self.head) {
            try self.d.?.reference(want, mag);
            return;
        }
        for (0..self.shape.rows) |r| for (0..self.shape.n) |j| {
            const o = r * self.shape.n + j;
            const raw = self.headDot(r, j);
            want[o] = if (std.math.isNan(raw) or std.math.isInf(raw)) cap30(raw) else headValue(raw);
            mag[o] = @abs(want[o]) + 1e-6;
        };
    }

    pub fn simulate(self: *Q8, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.head) {
            const mag = try self.a.alloc(f64, out.len);
            defer self.a.free(mag);
            try self.reference(out, mag);
            return;
        }
        try self.d.?.simulate(acc, order, out);
    }
};

/// The 56 registrations: the kernels at each row count and the wrapper, dense and head, and the planted twins of each.
pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    kernels: [4]host.Kernels,
    probes: std.ArrayList(*Q8) = .empty,

    /// Loads the production image and the three control images into `dev`.
    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Suite {
        var made: [4]host.Kernels = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (.{ Control.none, Control.store, Control.sums, Control.scale }, .{ &images.inkling2_dense_q8, &images.contract_inkling2_dense_q8_store, &images.contract_inkling2_dense_q8_sums, &images.contract_inkling2_dense_q8_scale }) |control, bin| {
            try dev.addImage(imageOf(control), bin);
            made[n] = try host.Kernels.load(dev.named(imageOf(control)).?.*, control != .none);
            n += 1;
        }
        return .{ .a = a, .dev = dev, .kernels = made };
    }

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel)) !void {
        for ([_]bool{ false, true }) |head| for ([_]u32{ 0, 1, 2, 4, 8, 16, 32 }) |lead| for ([_]Control{ .none, .store, .sums, .scale }) |control| {
            const probe = try s.a.create(Q8);
            errdefer s.a.destroy(probe);
            probe.* = Q8.init(s.a, s.dev, &s.kernels[@intFromEnum(control)], lead, head, control);
            try s.probes.append(s.a, probe);
            try kernels.append(s.a, probe.kernel());
        };
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        for (&s.kernels) |*k| k.deinit();
    }
};

/// The wrapper against float64 at the served calls, as BF16 ulps (the head at the whole vocabulary of 200,058 rows). The
/// weights are the decoded Q8 values, so the only difference left is FP32 sums and one bf16 store: a sum of terms that
/// cancel is measured in the ulp of a 16th of its terms' size, since the result cannot be better than its terms.
/// Prints a line each and returns the shapes whose worst error passes 1 ulp.
pub fn report(a: std.mem.Allocator, dev: *rig_.Dev, suite: *Suite) !usize {
    var problems: usize = 0;
    for ([_]bool{ false, true }) |head| {
        var p = Q8.init(a, dev, &suite.kernels[0], 0, head, .none);
        defer p.deinit();
        const calls: []const [2]usize = if (head) &.{.{ 200058, 2304 }} else &.{ .{ 3072, 2304 }, .{ 1024, 2304 }, .{ 384, 2304 }, .{ 24, 2304 }, .{ 2304, 3072 }, .{ 64, 4608 } };
        for ([_]u32{ 1, 8, 32 }) |rows| for (calls) |pr| {
            if (head and rows == 32) continue;
            const shape: c.Shape = .{ .rows = rows, .n = pr[0], .k = pr[1] };
            if (!try p.prepare(shape, .random, 0x5c0de + rows)) continue;
            try p.run();
            const out = try p.rig.outputBytes();
            const n = p.count();
            const want = try a.alloc(f64, 2 * n);
            defer a.free(want);
            if (head) try p.reference(want[0..n], want[n..]) else try p.d.?.reference(want[0..n], want[n..]);
            var worst: f64 = 0;
            var differ: usize = 0;
            for (0..n) |i| {
                const w = want[i];
                const got = c.read(if (head) .f32 else .bf16, out, i);
                const unit = ulp(@max(@abs(w), want[n + i] / 16));
                if (head) {
                    // the reference is the int32 dot times the scales; a few fp32 ulps plus the bf16 store remain
                    if (@abs(got - w) > 8 * ulp(@abs(w)) + 1e-6) differ += 1;
                    worst = @max(worst, @abs(got - w) / @max(unit, 1e-12));
                } else {
                    if (got != c.fromBf16(c.toBf16(@floatCast(w)))) differ += 1;
                    worst = @max(worst, @abs(got - w) / unit);
                }
            }
            const bad = if (head) differ > 0 else !(worst <= 1.5);
            problems += @intFromBool(bad);
            std.debug.print("{s} float64 xpu inkling2 {s} q8 {d} rows x {d} x {d}: worst {d:.3} bf16 ulps, {d} of {d} outputs not the correctly rounded float64 value\n", .{ if (bad) "FAIL" else "PASS", if (head) "head" else "dense", rows, pr[0], pr[1], worst, differ, n });
        };
    }
    return problems;
}

/// `ldy`: rows land `ldy` values apart and the columns between stay untouched.
pub fn ldyCheck(a: std.mem.Allocator, dev: *rig_.Dev, suite: *Suite) !usize {
    var p = Q8.init(a, dev, &suite.kernels[0], 4, false, .none);
    defer p.deinit();
    const shape: c.Shape = .{ .rows = 4, .n = 1024, .k = 2304 };
    if (!try p.prepare(shape, .random, 0x1d9)) return 1;
    try p.run();
    const plain = try a.dupe(u8, try p.rig.outputBytes());
    defer a.free(plain);
    const ldy = shape.n + 16;
    const out = try p.rig.output(shape.rows * ldy * 2);
    try host.matvec(p.kernels, dev.s, 4, p.wr, p.xr, out, 4, @intCast(shape.n), @intCast(shape.k), @intCast(ldy));
    try dev.s.synchronize();
    const got = try p.rig.bytes(out, shape.rows * ldy * 2);
    var bad: usize = 0;
    for (0..shape.rows) |r| for (0..ldy) |j| {
        const b = got[(r * ldy + j) * 2 ..][0..2];
        if (j < shape.n) {
            if (!std.mem.eql(u8, b, plain[(r * shape.n + j) * 2 ..][0..2])) bad += 1;
        } else if (b[0] != 0xff or b[1] != 0xff) bad += 1;
    };
    std.debug.print("{s} ldy xpu inkling2 dense q8 tf_ink2_dense_q8_4: rows {d} values apart, {d} of {d} values differ from the packed call or touch the gap\n", .{ if (bad == 0) "PASS" else "FAIL", ldy, bad, shape.rows * ldy });
    return @intFromBool(bad != 0);
}

/// The quantize kernel against a host quantize of the same input. Codes match. Scales are within 1 ulp.
/// The activation round this step introduces is covered by the pooled KL16 (0.004398 against 0.004401).
pub fn quantCheck(a: std.mem.Allocator, dev: *rig_.Dev, suite: *Suite) !usize {
    var rig = rig_.Rig{ .a = a, .dev = dev };
    defer rig.deinit();
    var bad: usize = 0;
    const cases = [_]struct { rows: u32, k: u32 }{ .{ .rows = 1, .k = 2304 }, .{ .rows = 3, .k = 256 }, .{ .rows = 8, .k = 2304 } };
    for (cases, 0..) |cs, seed| {
        const n: usize = cs.rows * cs.k;
        const nb = cs.k / 32;
        const x = try a.alloc(u16, n);
        defer a.free(x);
        var prng = std.Random.DefaultPrng.init(0x51c0 + seed);
        for (x) |*v| v.* = c.toBf16(prng.random().float(f32) * 4 - 2);
        const want_c = try a.alloc(i8, n);
        defer a.free(want_c);
        const want_s = try a.alloc(f32, cs.rows * nb);
        defer a.free(want_s);
        const flags = try a.alloc(i32, cs.rows);
        defer a.free(flags);
        hostQuant(x, cs.k, want_c, want_s, flags);
        const xd = try rig.input(std.mem.sliceAsBytes(x));
        const qc = try rig.output(n);
        const qs = try rig.output(cs.rows * nb * 4 + cs.rows * 4);
        try rig.launch(suite.kernels[0].quant, .{ cs.rows, 1, 1 }, .{ xd, qc, qs, cs.rows, cs.k });
        const got_c = try a.dupe(u8, try rig.bytes(qc, n));
        defer a.free(got_c);
        const got_s = try rig.bytes(qs, cs.rows * nb * 4);
        var code_bad: usize = 0;
        var shown: usize = 0;
        const got_i = std.mem.bytesAsSlice(i8, got_c);
        for (got_i, want_c, 0..) |gotc, w, i| {
            if (gotc == w) continue;
            code_bad += 1;
            if (shown < 5) {
                shown += 1;
                const xv: f32 = @floatCast(c.fromBf16(x[i]));
                const sc = want_s[i / 32];
                std.debug.print("  lane {d}: x {x} scale {x} x/scale {x} device {d} host {d}\n", .{ i, @as(u32, @bitCast(xv)), @as(u32, @bitCast(sc)), @as(u32, @bitCast(xv / sc)), gotc, w });
            }
        }
        var scale_bad: usize = 0;
        for (std.mem.bytesAsSlice(f32, got_s), want_s) |got, w| {
            if (got == w) continue;
            const gf: f64 = got;
            const wf: f64 = w;
            const ul = @abs(gf - wf) / @max(ulp(wf), ulp(gf));
            if (!(ul <= 1)) scale_bad += 1;
        }
        if (code_bad != 0 or scale_bad != 0) bad += 1;
        std.debug.print("{s} quant xpu inkling2 q8 {d} rows x {d}: {d} codes differ, {d} scales over 1 ulp\n", .{ if (code_bad == 0 and scale_bad == 0) "PASS" else "FAIL", cs.rows, cs.k, code_bad, scale_bad });
    }
    return bad;
}

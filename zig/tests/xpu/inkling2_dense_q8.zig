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
    xr: ?*anyopaque = null,
    wr: ?*anyopaque = null,
    out: ?*anyopaque = null,

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
                .sums = if (self.head) .{ .f16 = 0, .bf16 = 0, .why = "the logit is capped and rounded to bf16 after its sum, so a lower-precision sum has no uniform separability factor" } else .{},
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
        self.d = d;
        const rows_dev = if (self.lead == 0) shape.rows else (shape.rows + self.lead - 1) / self.lead * self.lead;
        const x = try al.alloc(u16, rows_dev * shape.k);
        @memset(x, 0);
        @memcpy(x[0 .. shape.rows * shape.k], d.x);
        self.xr = try self.rig.input(std.mem.sliceAsBytes(x));
        const wb = try al.alloc(u8, q8.bytes(shape.n, shape.k));
        @memcpy(wb[0 .. shape.n * shape.k], made[0]);
        @memcpy(wb[q8.scaleOffset(shape.n, shape.k)..], std.mem.sliceAsBytes(made[1]));
        self.wr = try self.rig.input(wb);
        self.out = try self.rig.output(shape.rows * shape.n * @as(usize, if (self.head) 4 else 2));
        return true;
    }

    pub fn run(self: *Q8) !void {
        const s = self.rig.dev.s;
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        const k: u32 = @intCast(self.shape.k);
        const out_bytes: usize = if (self.head) 4 else 2;
        if (self.lead == 0) {
            if (self.head) try host.head(self.kernels, s, self.wr, self.xr, self.out, r, n) else try host.dense(self.kernels, s, self.wr, self.xr, self.out, r, n, k);
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

    pub fn reference(self: *Q8, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
        if (self.head) for (want, mag) |*w, *m| {
            w.* = if (std.math.isNan(w.*) or std.math.isInf(w.*)) cap30(w.*) else headValue(w.*);
            m.* = 10 * @abs(w.*) + 1e-5;
        };
    }

    pub fn simulate(self: *Q8, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
        if (self.head) for (out) |*v| {
            v.* = headValue(v.*);
        };
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
            try p.d.?.reference(want[0..n], want[n..]);
            var worst: f64 = 0;
            var differ: usize = 0;
            for (0..n) |i| {
                const sum = want[i];
                const w = if (head) Q8.headValue(sum) else sum;
                const got = c.read(if (head) .f32 else .bf16, out, i);
                const unit = ulp(@max(@abs(w), want[n + i] / 16));
                if (head) {
                    // the logit is the cap of the bf16-rounded dot, in FP32
                    if (@abs(got - w) > 2e-6 * @abs(w) + 1e-9) differ += 1;
                    worst = @max(worst, @abs(got - w) / ulp(@max(@abs(sum), want[n + i] / 16)));
                } else {
                    if (got != c.fromBf16(c.toBf16(@floatCast(w)))) differ += 1;
                    worst = @max(worst, @abs(got - w) / unit);
                }
            }
            const bad = !(worst <= 1.5) or (head and differ > 0);
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

//! Inkling 2 Mini's dense projections and RMS norm on XPU under the kernel contract: bf16 x times bf16 W with FP32 sums
//! and one bf16 store (the six row counts of the matvec, and the host wrapper that splits rows 32, 16, 8, 4, 2, 1), and
//! the FP32-row norm to bf16, each against float64 at Mini's served shapes. Every kernel has planted twins that must go red.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_dense_host");
const images = @import("xpu_contract_images");

const c = core.spec;
const g = core.gates;
const cdense = core.dense;
const rows_ = core.rows;

const H: usize = host.hidden;

/// What a planted twin gets wrong. `store`: every store truncates to bf16 (gate 3). `sums`: the matvec adds each
/// chunk's sum through fp16 (gate 4), and the norm drops its eps (gate 3).
pub const Control = enum { none, store, sums };

fn imageOf(control: Control) []const u8 {
    return switch (control) {
        .none => "inkling2_dense",
        .store => "contract_inkling2_dense_store",
        .sums => "contract_inkling2_dense_sums",
    };
}

/// Argument counts gate 5 checks a loaded kernel against.
pub fn argsOf(symbol: []const u8) ?u32 {
    const names = [_][]const u8{ "tf_ink2_dense_1", "tf_ink2_dense_2", "tf_ink2_dense_4", "tf_ink2_dense_8", "tf_ink2_dense_16", "tf_ink2_dense_32" };
    for (names) |n| if (std.mem.eql(u8, n, symbol)) return 7;
    if (std.mem.eql(u8, symbol, "tf_ink2_norm")) return 4;
    return null;
}

fn depth(s: c.Shape) usize {
    // a lane's chunks (eight products a chain, then one add each), then the sub-group's tree of 16
    return 8 + s.k / 128 + 5;
}

fn normDepth(s: c.Shape) usize {
    // a work-item's fused chain over its share, the sub-group's tree, then the sixteen sub-group sums in order
    return s.k / 256 + 4 + 16;
}

const projections = [_][2]usize{ .{ 3072, 2304 }, .{ 1024, 2304 }, .{ 384, 2304 }, .{ 24, 2304 }, .{ 2304, 3072 } };

/// The served projections at `lead` rows (wq and gate, wk and wv, wr, the sink, wo), and one call of 2 lead + 1 rows.
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

fn bits(values: []const f32, out: []u16) void {
    for (out, values) |*b, v| b.* = c.toBf16(v);
}

fn put(r: ?*anyopaque, bytes: usize) ?*anyopaque {
    return if (r) |p| @ptrFromInt(@intFromPtr(p) + bytes) else null;
}

/// `lead` = 0: the host wrapper over every row count; else the matvec kernel of that many rows, called on blocks of
/// `lead` rows with a short last block, whose input rows past the call are zero.
pub const Dense = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kernels: *const host.Kernels,
    lead: u32,
    control: Control,
    name: [96]u8 = undefined,
    name_len: usize = 0,
    shape: c.Shape = undefined,
    d: ?cdense.Dense = null,
    xr: ?*anyopaque = null,
    wr: ?*anyopaque = null,
    out: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kernels: *const host.Kernels, lead: u32, control: Control) Dense {
        var self: Dense = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .dev = dev }, .kernels = kernels, .lead = lead, .control = control };
        const who = if (lead == 0) std.fmt.bufPrint(&self.name, "xpu inkling2 dense host wrapper", .{}) else std.fmt.bufPrint(&self.name, "xpu inkling2 dense tf_ink2_dense_{d}", .{lead});
        self.name_len = (who catch unreachable).len;
        return self;
    }

    pub fn deinit(self: *Dense) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Dense) g.Kernel {
        return .{
            .ctx = self,
            .vt = &vt,
            .control = switch (self.control) {
                .none => null,
                .store => .shapes,
                .sums => .sums,
            },
            .spec = .{
                .name = self.label(),
                .accum = .f32,
                .depth = depth,
                .out = .bf16,
                .served = servedOf(self.lead),
                .tiles = if (self.lead == 0) .{ .{}, .{}, .{} } else .{
                    .{ .size = self.lead },
                    .{ .size = host.matvec_columns },
                    .{ .size = 128, .step = 8, .min = 8 },
                },
            },
        };
    }

    var labels: [3][7][112]u8 = undefined;
    fn label(self: *Dense) []const u8 {
        const ci: usize = @intFromEnum(self.control);
        const li: usize = if (self.lead == 0) 6 else std.math.log2_int(u32, self.lead);
        const prefix = switch (self.control) {
            .none => "",
            .store => "control (store truncated): ",
            .sums => "control (chunk sums in fp16): ",
        };
        return std.fmt.bufPrint(&labels[ci][li], "{s}{s}", .{ prefix, self.name[0..self.name_len] }) catch unreachable;
    }

    const vt = g.hooks(Dense);

    pub fn count(self: *Dense) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn prepare(self: *Dense, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 8 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        self.shape = shape;
        const d = try cdense.Dense.init(al, shape, 1);
        d.fillX(fill, seed, 2);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5eed);
        const rng = prng.random();
        const wb = try al.alloc(u16, shape.n * shape.k);
        const half = shape.k / 2;
        for (0..shape.n) |j| for (0..shape.k) |i| {
            const v: f32 = switch (fill) {
                // integers whose products and sums are exact in FP32 in any order; the second half cancels the first
                .stress => blk: {
                    const m: f32 = @floatFromInt(1 + ((if (i < half) i else i - half) * 7 + j) % 6);
                    break :blk if (i < half) m else -m;
                },
                else => rng.float(f32) * 0.5 - 0.25,
            };
            wb[j * shape.k + i] = c.toBf16(v);
            d.w[j * shape.k + i] = c.fromBf16(wb[j * shape.k + i]);
        };
        self.d = d;
        const rows_dev = if (self.lead == 0) shape.rows else (shape.rows + self.lead - 1) / self.lead * self.lead;
        const x = try al.alloc(u16, rows_dev * shape.k);
        @memset(x, 0);
        @memcpy(x[0 .. shape.rows * shape.k], d.x);
        self.xr = try self.rig.input(std.mem.sliceAsBytes(x));
        self.wr = try self.rig.input(std.mem.sliceAsBytes(wb));
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Dense) !void {
        const s = self.rig.dev.s;
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        const k: u32 = @intCast(self.shape.k);
        if (self.lead == 0) {
            try host.dense(self.kernels, s, self.wr, self.xr, self.out, r, n, k);
        } else {
            var base: u32 = 0;
            while (base < r) : (base += self.lead) {
                const t = @min(self.lead, r - base);
                try host.matvec(self.kernels, s, self.lead, self.wr, put(self.xr, @as(usize, base) * k * 2), put(self.out, @as(usize, base) * n * 2), t, n, k, n);
            }
        }
        try s.synchronize();
    }

    pub fn reference(self: *Dense, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Dense, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

/// The norm of FP32 rows at hidden 2,304: bf16(w * (x * inv)).
pub const Norm = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kernels: *const host.Kernels,
    control: Control,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f64 = &.{},
    xr: ?*anyopaque = null,
    wr: ?*anyopaque = null,
    out: ?*anyopaque = null,

    const eps: f64 = @as(f32, 1e-6);
    /// A row's amplitude, cycling so that some rows have a mean square near the eps.
    const scales = [_]f32{ 1, 0.125, 1e-3, 3e-4 };
    const served = [_]c.Shape{
        .{ .rows = 1, .n = H, .k = H },
        .{ .rows = 2, .n = H, .k = H },
        .{ .rows = 5, .n = H, .k = H },
        .{ .rows = 7, .n = H, .k = H },
        .{ .rows = 32, .n = H, .k = H },
        .{ .rows = 256, .n = H, .k = H },
    };

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kernels: *const host.Kernels, control: Control) Norm {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .dev = dev }, .kernels = kernels, .control = control };
    }

    pub fn deinit(self: *Norm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{
            .ctx = self,
            .vt = &vt,
            .control = switch (self.control) {
                .none => null,
                .store, .sums => .shapes,
            },
            .spec = .{
                .name = switch (self.control) {
                    .none => "xpu inkling2 dense tf_ink2_norm",
                    .store => "control (store truncated): xpu inkling2 dense tf_ink2_norm",
                    .sums => "control (eps dropped): xpu inkling2 dense tf_ink2_norm",
                },
                .accum = .f32,
                .depth = normDepth,
                // the mean, the eps, the root, the reciprocal, the two products
                .ops = 6,
                .out = .bf16,
                .served = &served,
                .sums = .{ .f16 = 0, .bf16 = 0, .why = "the output follows the sum of squares as its inverse root, so a bf16 sum's error stays near the output's own rounding" },
            },
        };
    }

    const vt = g.hooks(Norm);

    pub fn count(self: *Norm) usize {
        return self.shape.rows * H;
    }

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.k != H) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed ^ 0xa11ce);
        const rng = prng.random();
        const r = shape.rows;
        self.x = try al.alloc(f32, r * H);
        for (0..r) |row| for (0..H) |d| {
            self.x[row * H + d] = switch (fill) {
                // integers up to 20: the squares sum to about 2^20, exactly in FP32 in any order
                .stress => @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 20))) * (if (rng.boolean()) @as(f32, 1) else -1),
                else => (rng.float(f32) * 4 - 2) * scales[row % scales.len],
            };
        };
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, H / 3 }, .{ 1, H / 2 }, .{ 2, H - 1 }, .{ 3, 0 }, .{ 3, H - 1 } };
            const v = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, v) |p, val| if (p[0] < r) {
                self.x[p[0] * H + p[1]] = val;
            };
        }
        const wb = try al.alloc(u16, H);
        self.w = try al.alloc(f64, H);
        for (wb, self.w) |*b, *v| {
            b.* = c.toBf16(rng.float(f32) * 2 - 1);
            v.* = c.fromBf16(b.*);
        }
        self.xr = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.wr = try self.rig.input(std.mem.sliceAsBytes(wb));
        self.out = try self.rig.output(r * H * 2);
        return true;
    }

    pub fn run(self: *Norm) !void {
        const s = self.rig.dev.s;
        try host.norm(self.kernels, s, self.xr, self.wr, self.out, @intCast(self.shape.rows));
        try s.synchronize();
    }

    fn scaled(self: *Norm, r: usize, sum: f64, want: []f64) void {
        const inv = 1 / @sqrt(sum / @as(f64, H) + eps);
        for (0..H) |d| want[r * H + d] = self.w[d] * (self.x[r * H + d] * inv);
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| {
            var sum: f64 = 0;
            for (0..H) |d| sum += @as(f64, self.x[r * H + d]) * self.x[r * H + d];
            self.scaled(r, sum, want);
            // the output follows the sum as its inverse root: an error in the sum moves it by half
            for (0..H) |d| mag[r * H + d] = @abs(want[r * H + d]) / 2;
        }
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.a.alloc(f64, H);
        defer self.a.free(terms);
        for (0..self.shape.rows) |r| {
            for (terms, 0..) |*t, d| t.* = @as(f64, self.x[r * H + d]) * self.x[r * H + d];
            self.scaled(r, rows_.sumIn(acc, order, terms), out);
        }
    }
};

/// The 25 registrations: the matvec at each row count and the wrapper, the norm, and the planted twins of each.
pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    kernels: [3]host.Kernels,
    dense: std.ArrayList(*Dense) = .empty,
    norms: std.ArrayList(*Norm) = .empty,

    /// Loads the production image and the two control images into `dev`.
    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Suite {
        var made: [3]host.Kernels = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |*k| k.deinit();
        inline for (.{ Control.none, Control.store, Control.sums }, .{ &images.inkling2_dense, &images.contract_inkling2_dense_store, &images.contract_inkling2_dense_sums }) |control, bin| {
            try dev.addImage(imageOf(control), bin);
            made[n] = try host.Kernels.load(dev.named(imageOf(control)).?.*, control != .none);
            n += 1;
        }
        return .{ .a = a, .dev = dev, .kernels = made };
    }

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel)) !void {
        for ([_]u32{ 0, 1, 2, 4, 8, 16, 32 }) |lead| for ([_]Control{ .none, .store, .sums }) |control| {
            const probe = try s.a.create(Dense);
            errdefer s.a.destroy(probe);
            probe.* = Dense.init(s.a, s.dev, &s.kernels[@intFromEnum(control)], lead, control);
            try s.dense.append(s.a, probe);
            try kernels.append(s.a, probe.kernel());
        };
        for ([_]Control{ .none, .store, .sums }) |control| {
            const probe = try s.a.create(Norm);
            errdefer s.a.destroy(probe);
            probe.* = Norm.init(s.a, s.dev, &s.kernels[@intFromEnum(control)], control);
            try s.norms.append(s.a, probe);
            try kernels.append(s.a, probe.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.dense.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.dense.deinit(s.a);
        for (s.norms.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.norms.deinit(s.a);
        for (&s.kernels) |*k| k.deinit();
    }
};

/// The BF16 ulp of `v`: 2^(exponent - 7).
fn ulp(v: f64) f64 {
    const m = @max(@abs(v), 0x1p-120);
    return std.math.pow(f64, 2, @floor(std.math.log2(m)) - 7);
}

/// The wrapper against float64 at the served calls, as BF16 ulps. A sum of terms that cancel is measured in the ulp of
/// a 16th of its terms' size, since the result cannot be better than its terms: that is the unit when it is larger.
/// Prints a line each and returns the shapes whose worst error passes 1 ulp.
pub fn report(a: std.mem.Allocator, dev: *rig_.Dev, suite: *Suite) !usize {
    var p = Dense.init(a, dev, &suite.kernels[0], 0, .none);
    defer p.deinit();
    var problems: usize = 0;
    for ([_]u32{ 1, 8, 32 }) |rows| for (projections) |pr| {
        const shape: c.Shape = .{ .rows = rows, .n = pr[0], .k = pr[1] };
        if (!try p.prepare(shape, .random, 0x5c0de + rows)) continue;
        try p.run();
        const out = try p.rig.outputBytes();
        const n = p.count();
        const want = try a.alloc(f64, 2 * n);
        defer a.free(want);
        try p.reference(want[0..n], want[n..]);
        var worst: f64 = 0;
        var differ: usize = 0;
        for (0..n) |i| {
            const got = c.read(.bf16, out, i);
            const w = want[i];
            if (got != c.fromBf16(c.toBf16(@floatCast(w)))) differ += 1;
            worst = @max(worst, @abs(got - w) / ulp(@max(@abs(w), want[n + i] / 16)));
        }
        const bad = !(worst <= 1);
        problems += @intFromBool(bad);
        std.debug.print("{s} float64 xpu inkling2 dense {d} rows x {d} x {d}: worst {d:.3} bf16 ulps, {d} of {d} outputs not the correctly rounded float64 value\n", .{ if (bad) "FAIL" else "PASS", rows, pr[0], pr[1], worst, differ, n });
    };
    return problems;
}

/// `ldy`: rows land `ldy` values apart and the columns between stay untouched.
pub fn ldyCheck(a: std.mem.Allocator, dev: *rig_.Dev, suite: *Suite) !usize {
    var p = Dense.init(a, dev, &suite.kernels[0], 4, .none);
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
    std.debug.print("{s} ldy xpu inkling2 dense tf_ink2_dense_4: rows {d} values apart, {d} of {d} values differ from the packed call or touch the gap\n", .{ if (bad == 0) "PASS" else "FAIL", ldy, bad, shape.rows * ldy });
    return @intFromBool(bad != 0);
}

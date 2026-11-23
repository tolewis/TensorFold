//! Inkling 2 Mini glue on XPU under FP64 references, guarded buffers and the production launches.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_xpu_glue");

const c = core.spec;
const g = core.gates;
const p = host.plan;

fn depth(s: c.Shape) usize {
    return s.k + 40;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}

fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

/// Multiples of 1/64: a dot of 2,304 products of two such values is an integer over 4,096 below 2^24, so it is exact in
/// FP32 in every order. The logit's BF16 rounding of the dot then has no order to flip on, and a missing rounding shows.
fn coarse(values: []f32) void {
    for (values) |*v| v.* = @round(v.* * 64) / 64;
}

/// The name a registration carries: `xpu inkling2 glue <part> <symbol>`, with `control: ` before a planted one.
pub fn label(kind: p.Kind, control: bool) []const u8 {
    return switch (kind) {
        inline else => |k| if (control) "control: " ++ comptime base(k) else comptime base(k),
    };
}

fn base(comptime k: p.Kind) []const u8 {
    return "xpu inkling2 glue " ++ @tagName(k) ++ " " ++ comptime k.symbol();
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    control: bool,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: []const c.Shape = &.{},
    f: [4][]f32 = @splat(&.{}),
    ids: []u32 = &.{},
    route_ids: ?*anyopaque = null,
    refs: [5]?*anyopaque = @splat(null),
    out: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kind: p.Kind, control: bool) !Probe {
        const module = (try dev.module(if (control) .ink2_glue_ctl else .ink2_glue)).*;
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .dev = dev },
            .kind = kind,
            .control = control,
            .pipes = try host.Pipelines.load(module, control),
        };
        self.served = switch (kind) {
            .embed_norm => &.{ .{ .rows = 1, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden }, .{ .rows = 7, .n = p.hidden, .k = p.hidden } },
            .ar => &.{ .{ .rows = 2, .n = p.hidden, .k = p.hidden }, .{ .rows = 4, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden } },
            .conv_static => &.{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 1024, .k = 4 }, .{ .rows = 7, .n = p.hidden, .k = 4 } },
            .conv_dynamic => &.{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 512, .k = 4 }, .{ .rows = 7, .n = 1024, .k = 4 } },
            .router => &.{ .{ .rows = 1, .n = 8, .k = p.hidden }, .{ .rows = 5, .n = 8, .k = p.hidden }, .{ .rows = 7, .n = 8, .k = p.hidden } },
            .swiglu => &.{ .{ .rows = 1, .n = p.intermediate, .k = 2 }, .{ .rows = 5, .n = p.intermediate, .k = 2 }, .{ .rows = 7, .n = p.intermediate, .k = 2 } },
            .logits => &.{ .{ .rows = 1, .n = 128, .k = p.hidden }, .{ .rows = 5, .n = 256, .k = p.hidden }, .{ .rows = 7, .n = 128, .k = p.hidden }, .{ .rows = 2, .n = 4096, .k = p.hidden }, .{ .rows = 3, .n = 200, .k = p.hidden } },
            .expert_pick => &.{ .{ .rows = 1, .n = p.intermediate, .k = 8 }, .{ .rows = 5, .n = p.intermediate, .k = 8 }, .{ .rows = 7, .n = p.intermediate, .k = 8 } },
            .expert_combine => &.{ .{ .rows = 1, .n = p.hidden, .k = 8 }, .{ .rows = 5, .n = p.hidden, .k = 8 }, .{ .rows = 7, .n = p.hidden, .k = 8 } },
        };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }
    const vt_checked = g.hooks(Probe);
    const vt_plain = blk: {
        var vt = g.hooks(Probe);
        vt.check = null;
        break :blk vt;
    };
    pub fn kernel(self: *Probe) g.Kernel {
        const out: c.Elem = switch (self.kind) {
            .ar, .router, .logits => .f32,
            else => .bf16,
        };
        return .{
            .ctx = self,
            .vt = if (self.kind == .router) &vt_checked else &vt_plain,
            .control = if (self.control) .shapes else null,
            .spec = .{
                .name = label(self.kind, self.control),
                .accum = .f32,
                .depth = depth,
                .ops = 96,
                .mids = if (self.kind == .logits) 1 else 0,
                .out = out,
                .served = self.served,
                .nan = if (self.kind == .router) .{ .stated = "a NaN selected score makes the shared normalization and every selected weight NaN" } else .propagate,
                // AR's integer stress terms are exact in FP32 and must discriminate FP16/BF16 by 100x.
                .sums = if (self.kind == .ar) .{} else .{ .f16 = 0, .bf16 = 0, .why = "nonlinear norms, exponentials and BF16 endpoints have no uniform lower-precision separability factor" },
            },
        };
    }
    pub fn count(self: *Probe) usize {
        return if (self.kind == .ar) 2 * p.hidden else p.count(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
    }
    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const out = try self.arena.allocator().alloc(f32, n);
        for (out) |*x| x.* = value(rng);
        return out;
    }
    fn input(self: *Probe, values: []const f32, bf: bool) !?*anyopaque {
        if (!bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, x| b.* = c.toBf16(x);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }
    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!p.valid(self.kind, .{ .rows = shape.rows, .n = shape.n, .k = shape.k })) return false;
        if (self.kind == .embed_norm and shape.n != p.hidden) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const n = shape.n;
        const nf = fill == .nonfinite;
        self.f = @splat(&.{});
        switch (self.kind) {
            .embed_norm => {
                self.f[0] = try self.floats(64 * p.hidden, rng);
                self.f[1] = try self.floats(p.hidden, rng);
                self.ids = try self.arena.allocator().alloc(u32, r);
                for (self.ids, 0..) |*id, i| id.* = @intCast(i % 64);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
                self.refs[1] = try self.input(self.f[0], true);
                self.refs[2] = try self.input(self.f[1], true);
            },
            .ar => {
                self.f[0] = try self.floats((r - 1) * p.hidden, rng);
                self.f[1] = try self.floats(p.hidden, rng);
                self.f[2] = try self.floats(2 * p.hidden, rng);
                if (fill == .stress) {
                    // Each 2304-term square sum is exactly 9 * 2^24 in FP32 under every order.
                    for (0..r) |s| {
                        const values = if (s + 1 == r) self.f[1] else self.f[0][s * p.hidden ..][0..p.hidden];
                        @memset(values, if (s % 2 == 0) 256 else -256);
                    }
                    // Normalized mirrored queries leave +/-8; partial sums stay exact integers in FP32.
                    for (0..2) |query_row| {
                        const row = self.f[2][query_row * p.hidden ..][0..p.hidden];
                        for (0..p.hidden / 2) |d| {
                            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                            const nudge: f32 = if (d < 8) 1 else 0;
                            row[d] = v;
                            row[p.hidden / 2 + d] = nudge - v;
                        }
                    }
                }
                if (nf) self.f[2][0] = std.math.inf(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], false);
                self.refs[2] = try self.input(self.f[2], false);
            },
            .conv_static => {
                self.f[0] = try self.floats((r + 3) * n, rng);
                self.f[1] = try self.floats(4 * n, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .conv_dynamic => {
                self.f[0] = try self.floats((r + 3) * n, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(4 * n, rng);
                self.f[3] = try self.floats(32 * p.hidden, rng);
                if (nf) self.f[1][0] = std.math.nan(f32);
                for (0..4) |j| self.refs[j] = try self.input(self.f[j], true);
            },
            .router => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(p.expert_count * p.hidden, rng);
                self.f[2] = try self.floats(p.expert_count, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], false);
                self.route_ids = try self.rig.scratch(r * 8 * 4);
            },
            .swiglu => {
                self.f[0] = try self.floats(r * n * 2, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
            },
            .logits => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(n * p.hidden, rng);
                coarse(self.f[0]);
                coarse(self.f[1]);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .expert_pick => {
                self.f[0] = try self.floats(r * p.expert_count * n, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.ids = try self.arena.allocator().alloc(u32, r * 8);
                for (self.ids, 0..) |*id, j| id.* = @intCast(j % 8 * 13 + j / 8);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
            },
            .expert_combine => {
                self.f[0] = try self.floats(r * 8 * n, rng);
                self.f[1] = try self.floats(r * 8, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], false);
            },
        }
        // Every BF16 input and the FP64 oracle see exactly the same stored values.
        for (self.f, 0..) |values, j| if (values.len > 0 and switch (self.kind) {
            .ar => j == 0,
            .router => j < 2,
            .expert_combine => j == 0,
            else => true,
        }) round(values);
        const bytes = self.count() * @as(usize, if (self.kind == .ar or self.kind == .router or self.kind == .logits) 4 else 2);
        self.out = try self.rig.output(bytes);
        return true;
    }
    pub fn run(self: *Probe) !void {
        const s = self.rig.dev.s;
        const pl = &self.pipes;
        const R = self.refs;
        const y = self.out;
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        switch (self.kind) {
            .embed_norm => try host.embedNorm(pl, s, R[0], R[1], R[2], y, r),
            .ar => try host.ar(pl, s, R[0], R[1], R[2], y, r, 2),
            .conv_static => try host.convStatic(pl, s, R[0], R[1], y, r, n),
            .conv_dynamic => try host.convDynamic(pl, s, R[0], R[1], R[2], R[3], y, r, n),
            .router => try host.router(pl, s, R[0], R[1], R[2], self.route_ids, y, r),
            .swiglu => try host.swiglu(pl, s, R[0], y, r, n),
            .logits => try host.logits(pl, s, R[0], R[1], y, r, n),
            .expert_pick => try host.expertPick(pl, s, R[0], R[1], y, r, n),
            .expert_combine => try host.expertCombine(pl, s, R[0], R[1], y, r, n),
        }
        try s.synchronize();
    }
    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const r = self.shape.rows;
        const n = self.shape.n;
        for (0..r) |row| switch (self.kind) {
            .embed_norm => {
                const x = self.f[0][self.ids[row] * p.hidden ..][0..p.hidden];
                for (0..p.hidden) |d| want[row * p.hidden + d] = p.embedValue(x, self.f[1], d);
            },
            .ar => {
                var qout: [2 * p.hidden]f64 = undefined;
                for (0..2) |q| p.blockResidual(self.f[0], self.f[1], self.f[2], r, q, qout[q * p.hidden ..][0..p.hidden]);
                @memcpy(want, &qout);
                break;
            },
            .conv_static => {
                for (0..n) |d| want[row * n + d] = p.staticConv(self.f[0], self.f[1], n, row, d);
            },
            .conv_dynamic => {
                for (0..n) |d| want[row * n + d] = p.dynamicConv(self.f[0], self.f[1], self.f[2], self.f[3], n, row, d);
            },
            .router => {
                const route = p.route(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], self.f[2]);
                @memcpy(want[row * 8 ..][0..8], &route.weights);
            },
            .swiglu => {
                for (0..n) |d| want[row * n + d] = p.swiglu(self.f[0][(row * n + d) * 2], self.f[0][(row * n + d) * 2 + 1]);
            },
            .logits => {
                for (0..n) |v| want[row * n + v] = p.logit(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], v);
            },
            .expert_pick => {
                for (0..8) |j| {
                    for (0..n) |d| want[(row * 8 + j) * n + d] = p.pick(self.f[0], self.ids, n, row, j, d);
                }
            },
            .expert_combine => {
                for (0..n) |d| want[row * n + d] = p.combine(self.f[0][row * 8 * n ..], self.f[1][row * 8 ..][0..8], n, d);
            },
        };
        for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
        // Where terms cancel, the sums' error follows the terms, not the result: the contract's mag is their magnitude.
        switch (self.kind) {
            .ar => {
                var inverse: [5]f64 = undefined;
                for (0..r) |src| inverse[src] = 1 / @sqrt(p.arSourceSquares(self.f[0], self.f[1], r, src, null) / p.hidden + 1e-6);
                // the softmax weights sum to one, so the weighted terms are at most the largest normalized source
                for (0..p.hidden) |d| {
                    var top: f64 = 0;
                    for (0..r) |src| {
                        const x = if (src + 1 == r) self.f[1][d] else self.f[0][src * p.hidden + d];
                        top = @max(top, @abs(@as(f64, x)) * inverse[src]);
                    }
                    for (0..2) |q| mag[q * p.hidden + d] = top;
                }
            },
            .expert_combine => {
                for (0..r) |row| for (0..n) |d| {
                    var terms: f64 = 0;
                    for (0..8) |j| terms += @abs(@as(f64, self.f[1][row * 8 + j]) * self.f[0][(row * 8 + j) * n + d]);
                    mag[row * n + d] = terms;
                };
            },
            else => {},
        }
    }
    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .ar) {
            const sim: p.ArSum = .{
                .accum = switch (acc) {
                    .none => .none,
                    .f32 => .f32,
                    .f16 => .f16,
                    .bf16 => .bf16,
                },
                .order = switch (order) {
                    .sequential => .sequential,
                    .pairwise => .pairwise,
                    .strided => .strided,
                },
            };
            for (0..2) |q| p.blockResidualIn(self.f[0], self.f[1], self.f[2], self.shape.rows, q, sim, out[q * p.hidden ..][0..p.hidden]);
            return;
        }
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.reference(out, mag);
    }
    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (self.kind != .router) return "numeric hooks must be used";
        const bytes = try self.rig.outputBytes();
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        if (gate == .nonfinite) {
            for (want, 0..) |w, i| if (std.math.isNan(c.read(.f32, bytes, i)) != std.math.isNan(w)) return "NaN normalization rule differs";
            return null;
        }
        const got_ids = std.mem.bytesAsSlice(u32, try self.rig.bytes(self.route_ids, self.shape.rows * 8 * 4));
        for (0..self.shape.rows) |row| {
            const route = p.route(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], self.f[2]);
            for (0..8) |j| {
                if (got_ids[row * 8 + j] != route.ids[j]) return "selected IDs differ";
                const got = c.read(.f32, bytes, row * 8 + j);
                if (@abs(got - want[row * 8 + j]) > 1e-5 * @max(1, @abs(want[row * 8 + j]))) return "router weight differs";
            }
        }
        return null;
    }
};

/// Every glue kernel and, after each, its planted control, filtered by a substring of the registration's name.
pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, kind: p.Kind, control: bool, only: ?[]const u8) !void {
        if (only) |text| if (std.mem.indexOf(u8, label(kind, control), text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        probe.* = try Probe.init(s.a, s.dev, kind, control);
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        try s.kernels.append(s.a, probe.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(p.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};

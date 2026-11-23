//! Inkling 2 Mini's cross-pass messages and rank-64 gates on XPU: gates 1 to 4 against float64 oracles, the planted
//! controls, and the host float model of `loop_model.zig`. The specs and the fills are the Metal registration's.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_xpu_loop");
const model = @import("inkling2_loop_model");

const c = core.spec;
const g = core.gates;
const p = host.plan;

/// Which kernel, and whether it is the production image or the controls image (one arithmetic anchor dropped).
pub const Kind = p.Kind;

pub fn label(kind: Kind) []const u8 {
    return switch (kind) {
        .messages => "xpu inkling2 loop messages tf_inkling2_loop_messages",
        .gates => "xpu inkling2 loop gates tf_inkling2_loop_gates",
    };
}

/// The control drops the post-dot 1/48 of the messages, or the bf16 rounding of the gates' zeta.
pub fn controlLabel(kind: Kind) []const u8 {
    return switch (kind) {
        .messages => "control: xpu inkling2 loop messages tf_inkling2_loop_messages without the 1/48 scale",
        .gates => "control: xpu inkling2 loop gates tf_inkling2_loop_gates without the bf16 zeta",
    };
}


fn depth(s: c.Shape) usize {
    return if (s.n == p.message_entries * p.hidden) 2 * s.k + 16 else 2 * p.hidden + s.k + 16;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}


pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    control: bool,
    module: xpu.Module,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: [3]c.Shape = undefined,
    f: [6][]f32 = @splat(&.{}),
    refs: [6]?*anyopaque = @splat(null),
    out: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kind: p.Kind, control: bool) !Probe {
        const module = (try dev.module(if (control) .inkling2_loop_controls else .inkling2_loop)).*;
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .dev = dev },
            .kind = kind,
            .control = control,
            .module = module,
            .pipes = try host.Pipelines.load(module),
        };
        self.served = switch (kind) {
            .messages => .{
                .{ .rows = 1, .n = p.message_entries * p.hidden, .k = p.hidden },
                .{ .rows = 5, .n = p.message_entries * p.hidden, .k = p.hidden },
                .{ .rows = 7, .n = p.message_entries * p.hidden, .k = p.hidden },
            },
            .gates => .{
                .{ .rows = 1, .n = p.hidden, .k = p.rank },
                .{ .rows = 5, .n = p.hidden, .k = p.rank },
                .{ .rows = 7, .n = p.hidden, .k = p.rank },
            },
        };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.pipes.deinit();
        self.rig.deinit();
        self.arena.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{
            .ctx = self,
            .vt = &vt,
            .spec = .{
                .name = if (self.control) controlLabel(self.kind) else label(self.kind),
                .accum = .f32,
                .depth = depth,
                .ops = if (self.kind == .messages) 32 else 64,
                .mids = if (self.kind == .gates) 2 else 0,
                .out = .f32,
                .served = &self.served,
                .sums = .{},
            },
            .control = if (self.control) .shapes else null,
        };
    }
    pub fn count(self: *Probe) usize {
        return p.count(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = value(rng);
        return values;
    }
    fn input(self: *Probe, values: []const f32, bf: bool) !?*anyopaque {
        if (!bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const call = p.Call{ .rows = shape.rows, .n = shape.n, .k = shape.k };
        if (!p.valid(self.kind, call)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.f = @splat(&.{});
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        switch (self.kind) {
            .messages => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(2 * p.core_layers * p.hidden, rng);
                self.f[3] = try self.floats(2 * p.core_layers * p.hidden, rng);
                if (fill == .stress) {
                    // rho(+256) = +1 and rho(-256) = -1 in FP32: their square sum is 9 * 2^24.
                    @memset(self.f[0], 256);
                    @memset(self.f[1], -256);
                    // Mirrored integer queries leave 8, while every partial sum stays below 2^24.
                    for ([_][]f32{ self.f[2], self.f[3] }) |queries| for (0..2 * p.core_layers) |query_row| {
                        const row = queries[query_row * p.hidden ..][0..p.hidden];
                        for (0..p.hidden / 2) |d| {
                            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                            const nudge: f32 = if (d < 8) 1 else 0;
                            row[d] = v;
                            row[p.hidden / 2 + d] = nudge - v;
                        }
                    };
                }
                if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
                round(self.f[0]);
                round(self.f[1]);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], false);
                self.refs[3] = try self.input(self.f[3], false);
            },
            .gates => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(p.rank * 2 * p.hidden, rng);
                self.f[3] = try self.floats(p.hidden * p.rank, rng);
                self.f[4] = try self.floats(p.hidden * p.rank, rng);
                if (fill == .stress) {
                    @memset(self.f[0], 256);
                    @memset(self.f[1], -256);
                    // D sees [+1; -1]: mirrored integer rows leave zeta = 1, with exact FP32 partials.
                    for (0..p.rank) |rank| for (0..p.hidden) |d| {
                        const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                        self.f[2][rank * 2 * p.hidden + d] = w;
                        self.f[2][rank * 2 * p.hidden + p.hidden + d] = if (d == 0) w - 1 else w;
                    };
                    // With zeta = 1, mirrored F/Gamma rows leave one, also exactly in every FP32 order.
                    for ([_][]f32{ self.f[3], self.f[4] }) |weights| for (0..p.hidden) |d| for (0..p.rank / 2) |rank| {
                        const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                        weights[d * p.rank + rank] = w;
                        weights[d * p.rank + rank + p.rank / 2] = if (rank == 0) 1 - w else -w;
                    };
                }
                if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
                round(self.f[2]);
                round(self.f[3]);
                round(self.f[4]);
                self.refs[0] = try self.input(self.f[0], false);
                self.refs[1] = try self.input(self.f[1], false);
                self.refs[2] = try self.input(self.f[2], true);
                self.refs[3] = try self.input(self.f[3], true);
                self.refs[4] = try self.input(self.f[4], true);
            },
        }
        self.out = try self.rig.output(self.count() * @sizeOf(f32));
        return true;
    }

    pub fn run(self: *Probe) !void {
        const rows: u32 = @intCast(self.shape.rows);
        const s = self.rig.dev.s;
        switch (self.kind) {
            .messages => try host.crossPassMessageAttention(&self.pipes, s, self.refs[0], self.refs[1], self.refs[2], self.refs[3], self.out, rows),
            .gates => try host.gates(&self.pipes, s, self.refs[0], self.refs[1], self.refs[2], self.refs[3], self.refs[4], self.out, rows),
        }
        try s.synchronize();
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .messages => for (0..self.shape.rows) |row| {
                const b0 = self.f[0][row * p.hidden ..][0..p.hidden];
                const b1 = self.f[1][row * p.hidden ..][0..p.hidden];
                for (0..p.message_entries) |entry| {
                    const role: p.Role = @fromBackingInt(@intCast(entry % p.roles));
                    const dst = want[(row * p.message_entries + entry) * p.hidden ..][0..p.hidden];
                    p.message(b0, b1, self.f[2], self.f[3], entry / p.roles, role, dst);
                }
            },
            .gates => for (0..self.shape.rows) |row| {
                const h = self.f[0][row * p.hidden ..][0..p.hidden];
                const m = self.f[1][row * p.hidden ..][0..p.hidden];
                p.gate(h, m, self.f[2], self.f[3], self.f[4], want[row * p.hidden ..][0..p.hidden]);
            },
        }
        for (mag, want) |*m, w| m.* = if (self.kind == .messages)
            10 * @abs(w) + 1e-3
        else
            32 * @abs(w) + 8;
    }
    fn simRms(x: []const f32, acc: c.Accum, order: g.Order, terms: []f64) f64 {
        for (x, 0..) |v, d| terms[d] = @as(f64, v) * v;
        const ss = core.rows.sumIn(acc, order, terms[0..p.hidden]);
        return 1 / @sqrt(ss / @as(f64, @floatFromInt(p.hidden)) + p.eps);
    }

    fn simMessage(bank0: []const f32, bank1: []const f32, q_mlp: []const f32, kv: []const f32, layer: usize, role: p.Role, inverse: [2]f64, acc: c.Accum, order: g.Order, terms: []f64, out: []f64) void {
        const queries = if (p.usesQueryMessageTensor(role)) q_mlp else kv;
        const query = queries[p.queryRow(role, layer) * p.hidden ..][0..p.hidden];
        var logits: [2]f64 = undefined;
        var peak = -std.math.inf(f64);
        inline for (0..2) |s| {
            const bank = if (s == 0) bank0 else bank1;
            for (0..p.hidden) |d| terms[d] = @as(f64, query[d]) * (@as(f64, bank[d]) * inverse[s]);
            logits[s] = core.rows.sumIn(acc, order, terms[0..p.hidden]) * p.message_scale;
            peak = @max(peak, logits[s]);
        }
        var weights: [2]f64 = undefined;
        var den: f64 = 0;
        for (0..2) |s| {
            weights[s] = @exp(logits[s] - peak);
            den += weights[s];
        }
        for (0..p.hidden) |d| out[d] = weights[0] / den * (@as(f64, bank0[d]) * inverse[0]) + weights[1] / den * (@as(f64, bank1[d]) * inverse[1]);
    }

    fn simGate(h: []const f32, m: []const f32, down: []const f32, forget: []const f32, gain: []const f32, inverse: [2]f64, acc: c.Accum, order: g.Order, terms: []f64, normalized: *[2 * p.hidden]f64, zeta: *[p.rank]f64, out: []f64) void {
        for (0..p.hidden) |d| {
            normalized[d] = p.roundBf16(@as(f64, h[d]) * inverse[0]);
            normalized[p.hidden + d] = p.roundBf16(@as(f64, m[d]) * inverse[1]);
        }
        for (0..p.rank) |r| {
            for (0..2 * p.hidden) |d| terms[d] = @as(f64, down[r * 2 * p.hidden + d]) * normalized[d];
            zeta[r] = p.roundBf16(core.rows.sumIn(acc, order, terms));
        }
        for (0..p.hidden) |d| {
            for (0..p.rank) |r| terms[r] = @as(f64, forget[d * p.rank + r]) * zeta[r];
            const fdot = core.rows.sumIn(acc, order, terms[0..p.rank]);
            for (0..p.rank) |r| terms[r] = @as(f64, gain[d * p.rank + r]) * zeta[r];
            const gdot = core.rows.sumIn(acc, order, terms[0..p.rank]);
            out[d] = p.blend(std.math.tanh(fdot), std.math.tanh(gdot), h[d], m[d]);
        }
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.a.alloc(f64, 2 * p.hidden);
        defer self.a.free(terms);
        switch (self.kind) {
            .messages => for (0..self.shape.rows) |row| {
                const banks = [2][]const f32{
                    self.f[0][row * p.hidden ..][0..p.hidden],
                    self.f[1][row * p.hidden ..][0..p.hidden],
                };
                var inverse: [2]f64 = undefined;
                inline for (0..2) |s| inverse[s] = simRms(banks[s], acc, order, terms);
                for (0..p.message_entries) |entry| {
                    const role: p.Role = @fromBackingInt(@intCast(entry % p.roles));
                    const dst = out[(row * p.message_entries + entry) * p.hidden ..][0..p.hidden];
                    simMessage(banks[0], banks[1], self.f[2], self.f[3], entry / p.roles, role, inverse, acc, order, terms, dst);
                }
            },
            .gates => {
                var normalized: [2 * p.hidden]f64 = undefined;
                var zeta: [p.rank]f64 = undefined;
                for (0..self.shape.rows) |row| {
                    const h = self.f[0][row * p.hidden ..][0..p.hidden];
                    const m = self.f[1][row * p.hidden ..][0..p.hidden];
                    var inverse: [2]f64 = undefined;
                    inverse[0] = simRms(h, acc, order, terms);
                    inverse[1] = simRms(m, acc, order, terms);
                    simGate(h, m, self.f[2], self.f[3], self.f[4], inverse, acc, order, terms, &normalized, &zeta, out[row * p.hidden ..][0..p.hidden]);
                }
            },
        }
    }

    /// What the host model's own summation order can move each output beyond the spec's allowance. A message's sums
    /// (the rms, the two dots) carry gamma(depth) of their terms, which the output's two betas bound. A gate's zeta
    /// is rounded to bf16, so an order that lands a sum across a rounding step moves one zeta by one bf16 step and
    /// F and Gamma by that times a weight; two such steps are allowed.
    pub fn hostSlack(self: *Probe, out: []f64) !void {
        const hs = p.hidden;
        const gamma = c.gamma(@floatFromInt(depth(self.shape)), 0x1p-24);
        switch (self.kind) {
            .messages => for (0..self.shape.rows) |row| {
                const b0 = self.f[0][row * hs ..][0..hs];
                const b1 = self.f[1][row * hs ..][0..hs];
                const inv0 = p.rmsScale(b0);
                const inv1 = p.rmsScale(b1);
                var scale: f64 = 0;
                for (b0, b1) |x, y| scale = @max(scale, @abs(@as(f64, x)) * inv0 + @abs(@as(f64, y)) * inv1);
                @memset(out[row * p.message_entries * hs ..][0 .. p.message_entries * hs], 2 * gamma * scale);
            },
            .gates => for (0..self.shape.rows) |row| {
                const hh = self.f[0][row * hs ..][0..hs];
                const mm = self.f[1][row * hs ..][0..hs];
                var zeta: [p.rank]f64 = undefined;
                p.downProjection(hh, mm, self.f[2], &zeta);
                for (0..hs) |d| {
                    var step: f64 = 0;
                    for (0..p.rank) |r| step = @max(step, @max(@abs(@as(f64, self.f[3][d * p.rank + r])), @abs(@as(f64, self.f[4][d * p.rank + r]))) * @abs(zeta[r]));
                    out[row * hs + d] = (@abs(@as(f64, hh[d])) + @abs(@as(f64, mm[d]))) * 2 * 0x1p-7 * step;
                }
            },
        }
    }

    /// Every output of the prepared call through `loop_model.zig`: f64 values holding the model's binary32 results.
    pub fn hostModel(self: *Probe, a: std.mem.Allocator, out: []f64) !void {
        const hs = p.hidden;
        const widen = struct {
            fn of(al: std.mem.Allocator, x: []const f32) ![]f64 {
                const w = try al.alloc(f64, x.len);
                for (w, x) |*d, v| d.* = v;
                return w;
            }
        }.of;
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const al = arena.allocator();
        switch (self.kind) {
            .messages => {
                const queries = model.MessageQueries{ .q_mlp = try widen(al, self.f[2]), .kv = try widen(al, self.f[3]) };
                for (0..self.shape.rows) |row| {
                    const bank = model.MessageBank{
                        .block = try widen(al, self.f[0][row * hs ..][0..hs]),
                        .partial = try widen(al, self.f[1][row * hs ..][0..hs]),
                    };
                    try model.crossPassMessages(al, bank, queries, model.rms_eps, out[row * p.message_entries * hs ..][0 .. p.message_entries * hs]);
                }
            },
            .gates => {
                const weights = model.GateWeights{ .down = try widen(al, self.f[2]), .forget = try widen(al, self.f[3]), .gain = try widen(al, self.f[4]), .rank = p.rank };
                for (0..self.shape.rows) |row| {
                    const hh = try widen(al, self.f[0][row * hs ..][0..hs]);
                    const mm = try widen(al, self.f[1][row * hs ..][0..hs]);
                    try model.messageGate(al, hh, mm, weights, model.rms_eps, out[row * hs ..][0..hs]);
                }
            },
        }
    }
};

/// The device's bytes against `loop_model.zig`'s host float model (ordered binary32 sums) at the served shapes. The two
/// differ by summation order alone, so each output may differ by twice the spec's allowance at float64's value, plus
/// what the model's own order can move: `hostSlack`.
pub fn modelCheck(a: std.mem.Allocator, dev: *rig_.Dev) !usize {
    var problems: usize = 0;
    for (std.enums.values(p.Kind)) |kind| {
        var probe = try Probe.init(a, dev, kind, false);
        defer probe.deinit();
        const spec = probe.kernel().spec;
        for (probe.served) |shape| {
            if (!try probe.prepare(shape, .random, 0x1007)) continue;
            try probe.run();
            const n = probe.count();
            const got = try a.alloc(f32, n);
            defer a.free(got);
            @memcpy(std.mem.sliceAsBytes(got), try probe.rig.bytes(probe.out, n * 4));
            const want = try a.alloc(f64, n);
            defer a.free(want);
            const mag = try a.alloc(f64, n);
            defer a.free(mag);
            try probe.reference(want, mag);
            const mod = try a.alloc(f64, n);
            defer a.free(mod);
            try probe.hostModel(a, mod);
            const slack = try a.alloc(f64, n);
            defer a.free(slack);
            try probe.hostSlack(slack);
            var worst: f64 = 0;
            var bad: usize = 0;
            for (got, mod, want, mag, slack) |x, y, w, m, extra| {
                const d = @abs(@as(f64, x) - y);
                worst = @max(worst, d);
                if (!(d <= 2 * c.bound(spec, shape, w, m) + extra)) bad += 1;
            }
            problems += @intFromBool(bad != 0);
            std.debug.print("{s} model {s} rows {d}: {d} of {d} outputs past the allowance, largest |device - loop_model| {e}\n", .{
                if (bad == 0) "PASS" else "FAIL", label(kind), shape.rows, bad, n, worst,
            });
        }
    }
    return problems;
}

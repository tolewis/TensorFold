//! Mini DSpark's eleven XPU kernels through the kernel contract: guarded buffers, explicit FP32 references with the BF16
//! rounding points the model requires, a float64 comparison at the served shapes, and one planted arithmetic mistake
//! (the contract_dspark_controls.cl copies) that gate 3 must catch in every kernel.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_dspark_host");
const sources = @import("inkling2_dspark_sources");
const images = @import("xpu_contract_images");

const c = core.spec;
const g = core.gates;

const H = host.hidden;
const Q = host.query_width;
const KV = host.kv_width;
const RANK = host.rank;
const RNN = host.rnn_width;
const INTER = host.intermediate;

/// The arguments a DSpark kernel takes, for gate 5's live check.
pub const argsOf = host.argsOf;

fn depth(s: c.Shape) usize {
    return s.k + 16;
}
fn bf(v: anytype) f64 {
    return host.roundBf16(v);
}
fn rounded(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}
fn rms(comptime F: type, values: []const f32, start: usize, width: usize) F {
    var sum: F = 0;
    for (0..width) |d| {
        const x: F = values[start + d];
        sum += x * x;
    }
    return 1.0 / @sqrt(sum / @as(F, @floatFromInt(width)) + 1e-6);
}
/// The size of the terms a rotated element is the difference of: its rounding error is a ulp of this, not of the result.
fn ropeTerms(comptime F: type, d: usize, position: usize, x: F, other: F) f64 {
    const frequency = std.math.pow(F, 1e6, -@as(F, @floatFromInt(d % 64)) / 64.0);
    const angle = @as(F, @floatFromInt(position)) * frequency;
    return @abs(x * @cos(angle)) + @abs(other * @sin(angle));
}
fn rope(comptime F: type, d: usize, position: usize, x: F, other: F) f64 {
    const frequency = std.math.pow(F, 1e6, -@as(F, @floatFromInt(d % 64)) / 64.0);
    const angle = @as(F, @floatFromInt(position)) * frequency;
    const rotated: F = if (d < 64) x * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x * @cos(angle);
    return bf(rotated);
}

/// The macro each control redefines, one for every kernel (the arithmetic anchor of its planted mistake).
fn anchor(kind: host.Kind) []const u8 {
    return switch (kind) {
        .init_rows => "DSP_INIT_SRC",
        .fuse_taps => "DSP_ALPHA",
        .block_qkv => "DSP_QKV_ROPE_HI",
        .context_kv => "DSP_CTX_ROPE_HI",
        .attention => "DSP_FIRST",
        .residual_norm => "DSP_NORM_SRC",
        .swiglu => "DSP_SILU",
        .kappa => "DSP_KAPPA_Y_COL",
        .recurrent => "DSP_LEFT",
        .base => "DSP_BASE_OUT",
        .logits => "DSP_BIAS",
    };
}

fn controlName(kind: host.Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: " ++ comptime std.fmt.comptimePrint("xpu inkling2 dspark {s} {s}", .{ @tagName(k), k.controlSymbol() }),
    };
}

fn isToken(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

/// Times `name` appears as a whole word in the lines of `text` that are neither comments nor preprocessor lines.
fn uses(text: []const u8, name: []const u8) usize {
    var n: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t");
        if (std.mem.startsWith(u8, line, "#") or std.mem.startsWith(u8, line, "//")) continue;
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, line, at, name)) |i| {
            const end = i + name.len;
            if ((i == 0 or !isToken(line[i - 1])) and (end == line.len or !isToken(line[end]))) n += 1;
            at = end;
        }
    }
    return n;
}

/// Every kernel uses its anchor macro once, the production file defaults it once, and the control file redefines it
/// once: a planted mistake edits exactly one place in exactly one kernel.
pub fn controlsApplyOnce() !void {
    for (std.enums.values(host.Kind)) |kind| {
        const name = anchor(kind);
        if (uses(sources.dspark, name) != 1) return error.ControlAnchorNotUsedOnce;
        var defaults: usize = 0;
        var redefined: usize = 0;
        var lines = std.mem.splitScalar(u8, sources.dspark, '\n');
        while (lines.next()) |line| if (std.mem.startsWith(u8, line, "#ifndef ") and std.mem.eql(u8, std.mem.trim(u8, line["#ifndef ".len..], " \t\r"), name)) {
            defaults += 1;
        };
        lines = std.mem.splitScalar(u8, sources.controls, '\n');
        while (lines.next()) |line| if (std.mem.startsWith(u8, line, "#define ")) {
            const rest = line["#define ".len..];
            const end = std.mem.indexOfAny(u8, rest, " (\t\r") orelse rest.len;
            if (std.mem.eql(u8, rest[0..end], name)) redefined += 1;
        };
        if (defaults != 1 or redefined != 1) return error.ControlAnchorNotDefinedOnce;
    }
}

test "every DSpark control mutates one anchor, once" {
    try controlsApplyOnce();
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: host.Kind,
    kernels: *const host.Kernels,
    control: bool,
    shape: c.Shape = undefined,
    served: [8]c.Shape = undefined,
    served_len: usize = 0,
    f: [8][]f32 = @splat(&.{}),
    phi: []u16 = &.{},
    refs: [8]host.Ref = @splat(null),
    outs: [3]host.Ref = @splat(null),
    score_scratch: host.Ref = null,
    /// The worst a kernel's bytes have been from the float64 reference at its served shapes, in BF16 ulps of the value.
    worst_ulps: f64 = 0,
    /// Where the float64 reference records the size of the terms of an output that is a difference of terms.
    terms: ?[]f64 = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kernels: *const host.Kernels, kind: host.Kind, control: bool) Probe {
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .dev = dev }, .kind = kind, .kernels = kernels, .control = control };
        switch (kind) {
            .init_rows => self.setServed(&.{.{ .rows = 8, .n = H, .k = 0 }}),
            .fuse_taps => self.setServed(&.{ .{ .rows = 5, .n = H, .k = 0 }, .{ .rows = 1, .n = H, .k = 0 }, .{ .rows = 3, .n = H, .k = 0 }, .{ .rows = 7, .n = H, .k = 0 } }),
            .block_qkv => self.setServed(&.{ .{ .rows = 8, .n = Q, .k = 17 }, .{ .rows = 8, .n = Q, .k = 67 } }),
            .context_kv => self.setServed(&.{ .{ .rows = 5, .n = KV, .k = 17 }, .{ .rows = 1, .n = KV, .k = 31 }, .{ .rows = 3, .n = KV, .k = 47 }, .{ .rows = 8, .n = KV, .k = 67 } }),
            .attention => self.setServed(&.{
                .{ .rows = 8, .n = 0, .k = 0 },
                .{ .rows = 8, .n = 1, .k = 0 },
                .{ .rows = 8, .n = 3, .k = 0 },
                .{ .rows = 8, .n = 3, .k = 5 },
                .{ .rows = 8, .n = 1025, .k = 1 },
                .{ .rows = 8, .n = 1025, .k = 7 },
                // a sliding layer past its ring (5,000 tokens in 1,152 slots), and a global layer of 20,001 tokens
                .{ .rows = 8, .n = 5000, .k = 1 },
                .{ .rows = 8, .n = 20001, .k = 0 },
            }),
            .residual_norm => self.setServed(&.{ .{ .rows = 8, .n = H, .k = 1 }, .{ .rows = 8, .n = H, .k = 0 } }),
            .swiglu => self.setServed(&.{.{ .rows = 8, .n = INTER, .k = 0 }}),
            .kappa => self.setServed(&.{ .{ .rows = 5, .n = RNN, .k = 0 }, .{ .rows = 1, .n = RNN, .k = 0 }, .{ .rows = 3, .n = RNN, .k = 0 }, .{ .rows = 7, .n = RNN, .k = 0 } }),
            .recurrent => self.setServed(&.{.{ .rows = 8, .n = RNN, .k = 0 }}),
            // the last shape is the served one: seven draft rows against the whole vocabulary, checked at sampled ids
            .base => self.setServed(&.{ .{ .rows = 5, .n = 5, .k = 0 }, .{ .rows = 1, .n = 1, .k = 0 }, .{ .rows = 3, .n = 17, .k = 0 }, .{ .rows = 7, .n = 129, .k = 0 }, .{ .rows = 7, .n = host.vocab, .k = 0 } }),
            .logits => self.setServed(&.{.{ .rows = 8, .n = host.vocab, .k = 0 }}),
        }
        return self;
    }
    fn setServed(self: *Probe, shapes: []const c.Shape) void {
        @memcpy(self.served[0..shapes.len], shapes);
        self.served_len = shapes.len;
    }
    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        var k: g.Kernel = .{ .ctx = self, .vt = &vt, .spec = .{
            .name = self.kind.label(),
            .accum = .none,
            .depth = depth,
            .ops = 0,
            .mids = 0,
            .out = if (self.kind == .kappa) .f32 else .bf16,
            .served = self.served[0..self.served_len],
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "explicit serial FP32 reference and required BF16 nodes are checked directly" },
        } };
        if (self.control) {
            k.spec.name = controlName(self.kind);
            k.control = .shapes;
        }
        return k;
    }
    pub fn count(self: *Probe) usize {
        return switch (self.kind) {
            .init_rows => 8 * H,
            .fuse_taps => (4 * self.shape.rows + 1) * H,
            .block_qkv => 8 * (Q + 2 * KV),
            .context_kv => 2 * self.shape.rows * KV,
            .attention => 8 * Q,
            .residual_norm => 2 * 8 * H,
            .swiglu => 8 * INTER,
            .kappa => self.shape.rows * RNN,
            .recurrent => RNN + 2 * RANK,
            .base => self.shape.rows * self.shape.n,
            .logits => host.vocab,
        };
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
        return values;
    }
    fn input(self: *Probe, values: []const f32, is_bf: bool) !host.Ref {
        if (!is_bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }
    fn output(self: *Probe, slot: usize, elements: usize, elem: c.Elem) !void {
        self.outs[slot] = try self.rig.output(elements * c.size(elem));
    }
    /// The slots of the context ring: the committed count, or Mini's 1,152 for a sliding layer that has outgrown it.
    fn ring(self: *const Probe) usize {
        return if (self.shape.k % 2 != 0 and self.shape.n > 4096) 1152 else @max(self.shape.n, 1);
    }
    fn first(self: *const Probe) usize {
        const from = self.shape.k / 2;
        const local = self.shape.k % 2 != 0;
        return @max(from, if (local and self.shape.n > 1024) self.shape.n - 1024 else 0);
    }
    /// The base kernel's whole-vocabulary shape is checked at every 509th id and the last few, not at all 1.4 million.
    fn sampled(self: *const Probe, i: usize) bool {
        if (self.kind != .base or self.shape.n <= 4096) return true;
        const id = i % self.shape.n;
        return id % 509 == 0 or id + 8 >= self.shape.n;
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        var admitted = false;
        for (self.served[0..self.served_len]) |served| if (std.meta.eql(shape, served)) {
            admitted = true;
            break;
        };
        if (!admitted) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.f = @splat(&.{});
        self.phi = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const rows = shape.rows;
        switch (self.kind) {
            .init_rows => {
                self.f[0] = try self.floats(H, rng);
                self.f[1] = try self.floats(H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[0]);
                rounded(self.f[1]);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                try self.output(0, 8 * H, .bf16);
            },
            .fuse_taps => {
                self.f[0] = try self.floats(rows * 4 * H, rng);
                self.f[1] = try self.floats(16, rng);
                self.f[2] = try self.floats(4 * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                @memcpy(self.f[1][0..4], &[_]f32{ 0, 0.5, -0.25, 2 });
                for (self.f[0][0 .. 4 * H]) |*v| v.* *= 400;
                rounded(self.f[0]);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, 4 * rows * H, .bf16);
                try self.output(1, H, .bf16);
            },
            .block_qkv => {
                const sizes = [_]usize{ 8 * Q, 8 * KV, 8 * KV, 128, 128 };
                for (sizes, 0..) |n, j| {
                    self.f[j] = try self.floats(n, rng);
                    rounded(self.f[j]);
                    self.refs[j] = try self.input(self.f[j], true);
                }
                if (nf) {
                    self.f[0][0] = std.math.nan(f32);
                    self.refs[0] = try self.input(self.f[0], true);
                }
                try self.output(0, 8 * Q, .bf16);
                try self.output(1, 8 * KV, .bf16);
                try self.output(2, 8 * KV, .bf16);
            },
            .context_kv => {
                self.f[0] = try self.floats(rows * KV, rng);
                self.f[1] = try self.floats(rows * KV, rng);
                self.f[2] = try self.floats(128, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, rows * KV, .bf16);
                try self.output(1, rows * KV, .bf16);
            },
            .attention => {
                const slots = self.ring();
                const sizes = [_]usize{ 8 * Q, 8 * KV, 8 * KV, slots * KV, slots * KV };
                for (sizes, 0..) |n, j| {
                    self.f[j] = try self.floats(n, rng);
                    rounded(self.f[j]);
                    self.refs[j] = try self.input(self.f[j], true);
                }
                if (nf) {
                    self.f[0][0] = std.math.nan(f32);
                    self.refs[0] = try self.input(self.f[0], true);
                }
                self.score_scratch = try self.rig.scratch(8 * 24 * (shape.n + 8) * 4);
                self.refs[5] = self.score_scratch;
                try self.output(0, 8 * Q, .bf16);
            },
            .residual_norm => {
                self.f[0] = try self.floats(8 * H, rng);
                self.f[1] = try self.floats(8 * H, rng);
                self.f[2] = try self.floats(H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, 8 * H, .bf16);
                try self.output(1, 8 * H, .bf16);
            },
            .swiglu => {
                self.f[0] = try self.floats(8 * INTER, rng);
                self.f[1] = try self.floats(8 * INTER, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..2]) |values| rounded(values);
                for (0..2) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, 8 * INTER, .bf16);
            },
            .kappa => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(H, rng);
                self.f[2] = try self.floats(RNN * 5120, rng);
                self.f[3] = try self.floats(RNN, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..4]) |values| rounded(values);
                for (0..4) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, rows * RNN, .f32);
            },
            .recurrent => {
                self.f[0] = try self.floats(RANK, rng);
                self.f[1] = try self.floats(RNN, rng);
                self.f[2] = try self.floats(RNN * 5120, rng);
                self.f[3] = try self.floats(2 * RANK, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..4]) |values| rounded(values);
                @memset(self.f[0], 99.5);
                self.f[1][0] = c.fromBf16(c.toBf16(0.6));
                self.f[1][RANK] = c.fromBf16(c.toBf16(0.8));
                self.f[1][2 * RANK] = c.fromBf16(c.toBf16(0.3));
                for ([_]usize{ 0, RANK, 2 * RANK }) |channel| @memset(self.f[2][channel * 5120 ..][0..512], 0);
                if (nf) self.f[0][0] = std.math.nan(f32);
                // kappa is the FP32 output of the kappa kernel, as the engine passes it
                for (0..4) |j| self.refs[j] = try self.input(self.f[j], j != 1);
                try self.output(0, RNN, .bf16);
                try self.output(1, RANK, .bf16);
                try self.output(2, RANK, .bf16);
            },
            .logits => {
                self.f[0] = try self.floats(host.vocab, rng);
                self.f[1] = try self.floats(host.vocab, rng);
                self.f[2] = try self.floats(RANK, rng);
                for (self.f[0]) |*v| v.* = rng.float(f32) * 120 - 60;
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                self.f[0][0] = 40;
                self.f[0][1] = 1000;
                self.f[1][0] = 0.125;
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.f[2][0] = c.fromBf16(c.toBf16(0.0001));
                self.phi = try self.arena.allocator().alloc(u16, host.vocab * RANK);
                for (self.phi) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
                @memset(self.phi[0..RANK], 0);
                self.phi[0] = c.toBf16(1);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.phi));
                self.refs[2] = try self.input(self.f[1], true);
                self.refs[3] = try self.input(self.f[2], true);
                try self.output(0, host.vocab, .bf16);
            },
            .base => {
                self.f[0] = try self.floats(shape.n * H, rng);
                self.f[1] = try self.floats(rows * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[0]);
                rounded(self.f[1]);
                if (!nf) {
                    @memset(self.f[0][0..H], 1);
                    @memset(self.f[1][0..H], 1);
                }
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                try self.output(0, rows * shape.n, .bf16);
            },
        }
        return true;
    }

    pub fn run(self: *Probe) !void {
        const R = self.refs;
        const O = self.outs;
        const k = self.kernels;
        const s = self.rig.dev.s;
        const n: u32 = @intCast(self.shape.rows);
        switch (self.kind) {
            .init_rows => try host.initializeRows(k, s, R[0], R[1], O[0]),
            .fuse_taps => try host.fuseTaps(k, s, R[0], R[1], R[2], O[0], O[1], n),
            .block_qkv => try host.blockQkv(k, s, R[0], R[1], R[2], R[3], R[4], O[0], O[1], O[2], @intCast(self.shape.k)),
            .context_kv => try host.commitContext(k, s, R[0], R[1], R[2], O[0], O[1], @intCast(self.shape.k), n, n),
            .attention => try host.attend(k, s, R[0], R[1], R[2], R[3], R[4], self.score_scratch, O[0], @intCast(self.shape.n), @intCast(self.ring()), self.shape.k % 2 != 0, @intCast(self.shape.k / 2)),
            .residual_norm => try host.residualNorm(k, s, R[0], R[1], R[2], O[0], O[1], 8, self.shape.k != 0),
            .swiglu => try host.swiglu(k, s, R[0], R[1], O[0], 8 * INTER),
            .kappa => try host.kappa(k, s, R[0], R[1], R[2], R[3], O[0], n),
            .recurrent => try host.recurrentStep(k, s, R[0], R[1], R[2], R[3], O[0], O[1], O[2], 1),
            .base => try host.base(k, s, R[0], R[1], O[0], n, @intCast(self.shape.n)),
            .logits => try host.finalLogits(k, s, R[0], R[1], R[2], R[3], O[0]),
        }
        try s.synchronize();
    }

    fn fusionReference(self: *Probe, comptime F: type, want: []f64) void {
        const rows = self.shape.rows;
        for (0..4) |fusion| {
            var scores: [4]F = undefined;
            var peak: F = -std.math.inf(F);
            for (0..4) |j| {
                scores[j] = self.f[1][fusion * 4 + j];
                peak = @max(peak, scores[j]);
            }
            var den: F = 0;
            for (0..4) |j| {
                scores[j] = @exp(scores[j] - peak);
                den += scores[j];
            }
            var alpha: [4]F = undefined;
            for (0..4) |j| alpha[j] = @floatCast(bf(scores[j] / den));
            for (0..rows) |row| {
                var square: F = 0;
                for (0..H) |d| {
                    var mixed: F = 0;
                    for (0..4) |j| mixed += alpha[j] * @as(F, self.f[0][(row * 4 + j) * H + d]);
                    square += mixed * mixed;
                }
                const scale: F = 1.0 / @sqrt(square / H + 1e-6);
                for (0..H) |d| {
                    var mixed: F = 0;
                    for (0..4) |j| mixed += alpha[j] * @as(F, self.f[0][(row * 4 + j) * H + d]);
                    const out = bf(mixed * scale * @as(F, self.f[2][fusion * H + d]));
                    want[(fusion * rows + row) * H + d] = out;
                    if (fusion == 3 and row + 1 == rows) want[4 * rows * H + d] = out;
                }
            }
        }
    }
    fn qkvReference(self: *Probe, comptime F: type, want: []f64) void {
        for (0..8) |row| for (0..24) |h| for (0..128) |d| {
            const at = row * Q + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const other_at = row * Q + h * 128 + partner;
            const scale = rms(F, self.f[0], row * Q + h * 128, 128);
            const x: F = @floatCast(bf(@as(F, self.f[0][at]) * scale * @as(F, self.f[3][d])));
            const other: F = @floatCast(bf(@as(F, self.f[0][other_at]) * scale * @as(F, self.f[3][partner])));
            want[at] = rope(F, d, self.shape.k + row, x, other);
            if (self.terms) |t| t[at] = ropeTerms(F, d, self.shape.k + row, x, other);
        };
        const offset = 8 * Q;
        for (0..8) |row| for (0..8) |h| for (0..128) |d| {
            const at = row * KV + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const other_at = row * KV + h * 128 + partner;
            const scale = rms(F, self.f[1], row * KV + h * 128, 128);
            const x: F = @floatCast(bf(@as(F, self.f[1][at]) * scale * @as(F, self.f[4][d])));
            const other: F = @floatCast(bf(@as(F, self.f[1][other_at]) * scale * @as(F, self.f[4][partner])));
            want[offset + at] = rope(F, d, self.shape.k + row, x, other);
            if (self.terms) |t| t[offset + at] = ropeTerms(F, d, self.shape.k + row, x, other);
            want[offset + 8 * KV + at] = self.f[2][at];
        };
    }
    fn contextReference(self: *Probe, comptime F: type, want: []f64) void {
        const rows = self.shape.rows;
        for (0..rows) |row| for (0..8) |h| for (0..128) |d| {
            const src = row * KV + h * 128 + d;
            const dst = ((self.shape.k + row) % rows) * KV + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const other_src = row * KV + h * 128 + partner;
            const scale = rms(F, self.f[0], row * KV + h * 128, 128);
            const x: F = @floatCast(bf(@as(F, self.f[0][src]) * scale * @as(F, self.f[2][d])));
            const other: F = @floatCast(bf(@as(F, self.f[0][other_src]) * scale * @as(F, self.f[2][partner])));
            want[dst] = rope(F, d, self.shape.k + row, x, other);
            if (self.terms) |t| t[dst] = ropeTerms(F, d, self.shape.k + row, x, other);
            want[rows * KV + dst] = self.f[1][src];
        };
    }
    fn attentionReference(self: *Probe, comptime F: type, want: []f64) !void {
        const committed = self.shape.n;
        const first_kept = self.first();
        const retained = committed - first_kept;
        const score_count = retained + 8;
        const slots = self.ring();
        const scores = try self.a.alloc(F, score_count);
        defer self.a.free(scores);
        for (0..8) |row| for (0..24) |h| {
            const kv = h / 3;
            const q_at = row * Q + h * 128;
            var peak: F = -std.math.inf(F);
            for (0..score_count) |j| {
                var dot: F = 0;
                const k_at = if (j < retained) ((first_kept + j) % slots) * KV + kv * 128 else (j - retained) * KV + kv * 128;
                const keys = if (j < retained) self.f[3] else self.f[1];
                for (0..128) |d| dot += @as(F, self.f[0][q_at + d]) * @as(F, keys[k_at + d]);
                scores[j] = dot * @as(F, 0x1.6a09e6p-7);
                peak = @max(peak, scores[j]);
            }
            var den: F = 0;
            for (scores) |s| den += @exp(s - peak);
            for (0..128) |d| {
                var sum: F = 0;
                var size: f64 = 0;
                for (0..score_count) |j| {
                    const v_at = if (j < retained) ((first_kept + j) % slots) * KV + kv * 128 + d else (j - retained) * KV + kv * 128 + d;
                    const values = if (j < retained) self.f[4] else self.f[2];
                    const term = (@exp(scores[j] - peak) / den) * @as(F, values[v_at]);
                    sum += term;
                    size += @abs(@as(f64, @floatCast(term)));
                }
                want[q_at + d] = bf(sum);
                // The weighted values cancel; the FP32 chain rounds at the size of its terms, not of their sum.
                if (self.terms) |t| t[q_at + d] = size;
            }
        };
    }
    fn recurrentReference(self: *Probe, comptime F: type, want: []f64) void {
        for (0..RANK) |i| {
            var z: [3]F = undefined;
            for (0..3) |part| {
                const channel = part * RANK + i;
                var sum: F = 0;
                for (0..RANK) |d| sum += @as(F, self.f[0][d]) * @as(F, self.f[2][channel * 5120 + d]);
                for (0..RANK) |d| sum += @as(F, self.f[3][RANK + d]) * @as(F, self.f[2][channel * 5120 + 256 + d]);
                sum += self.f[1][channel];
                z[part] = @floatCast(bf(sum));
                want[channel] = z[part];
            }
            const gamma: F = @floatCast(bf(1.0 / (1.0 + @exp(-z[0]))));
            const candidate: F = @floatCast(bf(std.math.tanh(z[1])));
            const keep: F = @floatCast(bf(1.0 - gamma));
            const left: F = @floatCast(bf(gamma * @as(F, self.f[0][i])));
            const right: F = @floatCast(bf(keep * candidate));
            want[RNN + i] = bf(left + right);
            want[RNN + RANK + i] = bf(std.math.tanh(z[2]));
        }
    }

    /// The outputs in float64 (`F` = f64) or in the explicit FP32 chain the kernels follow (`F` = f32); every BF16
    /// rounding point of the model is applied in both.
    fn referenceAs(self: *Probe, comptime F: type, want: []f64) !void {
        switch (self.kind) {
            .init_rows => for (0..8) |row| {
                for (0..H) |d| want[row * H + d] = self.f[if (row == 0) 0 else 1][d];
            },
            .fuse_taps => self.fusionReference(F, want),
            .block_qkv => self.qkvReference(F, want),
            .context_kv => self.contextReference(F, want),
            .attention => try self.attentionReference(F, want),
            .residual_norm => for (0..8) |row| {
                var square: F = 0;
                for (0..H) |d| {
                    const at = row * H + d;
                    const sigma: F = if (self.shape.k != 0) self.f[0][at] else @as(F, self.f[0][at]) + @as(F, self.f[1][at]);
                    square += sigma * sigma;
                }
                const scale: F = 1.0 / @sqrt(square / H + 1e-6);
                for (0..H) |d| {
                    const at = row * H + d;
                    const sigma: F = if (self.shape.k != 0) self.f[0][at] else @as(F, self.f[0][at]) + @as(F, self.f[1][at]);
                    want[at] = bf(sigma);
                    want[8 * H + at] = bf(sigma * scale * @as(F, self.f[2][d]));
                }
            },
            .swiglu => for (want, 0..) |*w, i| {
                const gate: F = self.f[0][i];
                w.* = bf((gate / (1.0 + @exp(-gate))) * @as(F, self.f[1][i]));
            },
            .kappa => for (0..self.shape.rows) |row| for (0..RNN) |channel| {
                var sum: F = 0;
                for (0..H) |d| sum += @as(F, self.f[0][row * H + d]) * @as(F, self.f[2][channel * 5120 + 512 + d]);
                for (0..H) |d| sum += @as(F, self.f[1][d]) * @as(F, self.f[2][channel * 5120 + 2816 + d]);
                sum += self.f[3][channel];
                want[row * RNN + channel] = sum;
            },
            .recurrent => self.recurrentReference(F, want),
            .base => for (0..self.shape.rows) |row| for (0..self.shape.n) |id| {
                if (!self.sampled(id)) continue;
                var dot: F = 0;
                for (0..H) |d| dot += @as(F, self.f[0][id * H + d]) * @as(F, self.f[1][row * H + d]);
                want[row * self.shape.n + id] = bf(dot);
            },
            .logits => for (0..host.vocab) |i| {
                var dot: F = 0;
                for (0..RANK) |d| dot += @as(F, c.fromBf16(self.phi[i * RANK + d])) * @as(F, self.f[2][d]);
                const bias: F = @floatCast(bf(@as(F, self.f[1][i]) + dot));
                const logit: F = @floatCast(bf(@as(F, self.f[0][i]) + bias));
                want[i] = bf(30.0 * std.math.tanh(logit / 30.0));
            },
        }
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        try self.referenceAs(f32, want);
        for (mag, want) |*m, w| m.* = 1 + @abs(w);
    }

    /// The distance of `got` from `want` in units of the last place of a BF16 value as large as `want` or the terms
    /// it is the difference of, whichever is larger; zero is exact.
    fn ulps(got: f64, want: f64, terms: f64) f64 {
        if (got == want) return 0;
        if (!std.math.isFinite(got) or !std.math.isFinite(want)) return std.math.inf(f64);
        const unit = std.math.pow(f64, 2, @floor(std.math.log2(@max(@abs(want), terms, 0x1p-120))) - 7);
        return @abs(got - want) / unit;
    }

    /// The kernel's bytes against float64's formula at this shape: the worst distance in BF16 ulps (relative error
    /// for the FP32 kappa) and how many sampled outputs are not the float64 reference's BF16 value.
    fn againstFloat64(self: *Probe, bytes: []const u8) !void {
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        @memset(want, 0);
        const terms = try self.a.alloc(f64, self.count());
        defer self.a.free(terms);
        @memset(terms, 0);
        self.terms = terms;
        defer self.terms = null;
        try self.referenceAs(f64, want);
        var worst: f64 = 0;
        var differ: usize = 0;
        var seen: usize = 0;
        for (want, 0..) |w, i| {
            if (!self.sampled(i)) continue;
            seen += 1;
            if (self.kind == .kappa) {
                const got = c.read(.f32, bytes, i);
                worst = @max(worst, @abs(got - w) / @max(1, @abs(w)));
                if (got != w) differ += 1;
            } else {
                const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
                const got = c.fromBf16(bits);
                const u = ulps(got, w, terms[i]);
                worst = @max(worst, u);
                if (u != 0) differ += 1;
            }
        }
        self.worst_ulps = @max(self.worst_ulps, worst);
        const unit = if (self.kind == .kappa) "relative error (FP32 output)" else "BF16 ulps";
        std.debug.print("F64  {s} at {d}x{d}x{d}: worst {d:.4} {s} from float64; {d} of {d} checked outputs differ from float64's BF16 value\n", .{ self.kind.label(), self.shape.rows, self.shape.n, self.shape.k, worst, unit, differ, seen });
        if (!(worst <= if (self.kind == .kappa) @as(f64, 1e-5) else 4.0)) return error.DsparkFloat64Disagrees;
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return "numeric hooks must be used";
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        @memset(want, 0);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        const bytes = try self.rig.outputBytes();
        for (want, 0..) |w, i| {
            if (!self.sampled(i)) continue;
            if (self.kind == .kappa) {
                const got = c.read(.f32, bytes, i);
                if (gate == .nonfinite and c.classOf(got) != c.classOf(w)) return "kappa floating-point class differs";
                if (gate == .shapes and std.math.isFinite(w) and @abs(got - w) > 2e-5 * @max(1, @abs(w))) return "FP32 kappa dot or column split differs";
            } else {
                const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
                const expected = c.toBf16(@floatCast(w));
                if (std.math.isNan(w)) {
                    if (!std.math.isNan(c.fromBf16(bits))) return "BF16 NaN was not propagated";
                } else if (bits != expected) {
                    const relative: f64 = switch (self.kind) {
                        .fuse_taps, .block_qkv, .context_kv, .swiglu, .logits => 0.02,
                        .attention => 0.03,
                        else => 0,
                    };
                    const got = c.fromBf16(bits);
                    if (@abs(@as(f64, got) - w) > @max(1e-5, relative * @abs(w))) {
                        if (!self.control) std.debug.print("DSPARK {s} output {d}: got {d} (0x{x}) want {d}\n", .{ self.kind.label(), i, got, bits, w });
                        return "BF16 endpoint differs from the explicit FP32/FP64 reference";
                    }
                }
            }
        }
        if (self.kind == .logits and gate == .shapes) {
            if (!(want[1] <= 30 and want[1] > 29.9)) return "the planted large logit does not prove softcap saturation";
        }
        if (self.kind == .base and gate == .shapes and self.f[0][0] == 1) {
            if (want[0] != 2304) return "raw base projection was softened instead of rounded once";
        }
        if (gate == .shapes and !self.control) self.againstFloat64(bytes) catch |e| return @errorName(e);
        return null;
    }
};

/// The 22 registrations: each kernel as served, and its planted copy.
pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    production: host.Kernels,
    planted: host.Kernels,
    probes: std.ArrayList(*Probe) = .empty,

    /// Loads both images into `dev` and checks every planted mistake edits one place.
    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Suite {
        try controlsApplyOnce();
        try dev.addImage("inkling2_dspark", &images.inkling2_dspark);
        try dev.addImage("contract_inkling2_dspark_controls", &images.contract_inkling2_dspark_controls);
        var production = try host.Kernels.load(dev.named("inkling2_dspark").?.*, false);
        errdefer production.deinit();
        const planted = try host.Kernels.load(dev.named("contract_inkling2_dspark_controls").?.*, true);
        return .{ .a = a, .dev = dev, .production = production, .planted = planted };
    }

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel)) !void {
        for (std.enums.values(host.Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = Probe.init(s.a, s.dev, if (control) &s.planted else &s.production, kind, control);
            errdefer probe.deinit();
            try s.probes.append(s.a, probe);
            try kernels.append(s.a, probe.kernel());
        };
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
        s.planted.deinit();
        s.production.deinit();
    }
};

//! Mini DFlash 2's six XPU kernels through the kernel contract: guarded buffers, explicit FP32 references with the BF16
//! rounding points the model requires, a float64 comparison at the served shapes, and one planted arithmetic mistake
//! (the contract_dflash2_controls.cl copies) that gate 3 must catch in every kernel.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_dflash2_host");
const sources = @import("inkling2_dflash2_sources");
const images = @import("xpu_contract_images");

const c = core.spec;
const g = core.gates;

const H = host.hidden;
const Q = host.query_width;
const KV = host.kv_width;
const DYN = host.dyn_width;
const LISTS = host.topk_groups * host.topk;
/// Ids the head keeps: the target's real vocabulary.
const VALID: usize = 200058;

/// The arguments a DFlash 2 kernel takes, for gate 5's live check.
pub const argsOf = host.argsOf;

fn depth(s: c.Shape) usize {
    return @max(s.n, 128) + 16;
}
fn bf(v: anytype) f64 {
    return @floatCast(c.fromBf16(c.toBf16(@floatCast(v))));
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
fn ropeTerms(comptime F: type, d: usize, position: usize, x: F, other: F) f64 {
    const frequency = std.math.pow(F, 10000, -@as(F, @floatFromInt(d % 64)) / 64.0);
    const angle = @as(F, @floatFromInt(position)) * frequency;
    return @abs(x * @cos(angle)) + @abs(other * @sin(angle));
}
fn rope(comptime F: type, d: usize, position: usize, x: F, other: F) f64 {
    const frequency = std.math.pow(F, 10000, -@as(F, @floatFromInt(d % 64)) / 64.0);
    const angle = @as(F, @floatFromInt(position)) * frequency;
    const rotated: F = if (d < 64) x * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x * @cos(angle);
    return bf(rotated);
}

/// The macro each control redefines, one for every kernel (the arithmetic anchor of its planted mistake).
fn anchor(kind: host.Kind) []const u8 {
    return switch (kind) {
        .conv => "DF_CONV_KERNEL",
        .block_qk => "DF_QK_ROPE_HI",
        .context_kv => "DF_CTX_ROPE_HI",
        .attention => "DF_FIRST",
        .merge => "DF_MERGE_WEIGHT",
        .topk => "DF_TOPK_VALUE",
    };
}

fn controlName(kind: host.Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: " ++ comptime std.fmt.comptimePrint("xpu inkling2 dflash2 {s} {s}", .{ @tagName(k), k.controlSymbol() }),
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
        if (uses(sources.dflash2, name) != 1) return error.ControlAnchorNotUsedOnce;
        var defaults: usize = 0;
        var redefined: usize = 0;
        var lines = std.mem.splitScalar(u8, sources.dflash2, '\n');
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

test "every DFlash 2 control mutates one anchor, once" {
    try controlsApplyOnce();
}

fn plus(p: host.Ref, bytes: usize) host.Ref {
    return @ptrFromInt(@intFromPtr(p.?) + bytes);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: host.Kind,
    kernels: *const host.Kernels,
    /// The production kernels: the other pass of the attention pair runs on these.
    production: *const host.Kernels,
    control: bool,
    shape: c.Shape = undefined,
    served: [10]c.Shape = undefined,
    served_len: usize = 0,
    f: [6][]f32 = @splat(&.{}),
    refs: [6]host.Ref = @splat(null),
    outs: [2]host.Ref = @splat(null),
    part: host.Ref = null,
    /// The worst a kernel's bytes have been from the float64 reference at its served shapes, in BF16 ulps of the value.
    worst_ulps: f64 = 0,
    terms: ?[]f64 = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kernels: *const host.Kernels, production: *const host.Kernels, kind: host.Kind, control: bool) Probe {
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .dev = dev }, .kind = kind, .kernels = kernels, .production = production, .control = control };
        switch (kind) {
            // k is the branch: 0 reads the first taps and columns, 1 the second
            .conv => self.setServed(&.{ .{ .rows = 8, .n = H, .k = 0 }, .{ .rows = 8, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 0 }, .{ .rows = 3, .n = H, .k = 1 } }),
            .block_qk => self.setServed(&.{ .{ .rows = 8, .n = Q, .k = 17 }, .{ .rows = 8, .n = Q, .k = 509 } }),
            .context_kv => self.setServed(&.{ .{ .rows = 5, .n = KV, .k = 17 }, .{ .rows = 1, .n = KV, .k = 31 }, .{ .rows = 3, .n = KV, .k = 47 }, .{ .rows = 8, .n = KV, .k = 67 }, .{ .rows = 8, .n = KV, .k = 509 } }),
            .attention, .merge => self.setServed(&.{
                .{ .rows = 8, .n = 0, .k = 0 },
                .{ .rows = 8, .n = 1, .k = 0 },
                .{ .rows = 8, .n = 3, .k = 0 },
                .{ .rows = 8, .n = 3, .k = 2 },
                // two chunks, then the window's edge, then a context far past its ring
                .{ .rows = 8, .n = 129, .k = 0 },
                .{ .rows = 8, .n = 1030, .k = 5 },
                .{ .rows = 8, .n = 2048, .k = 0 },
                .{ .rows = 8, .n = 5000, .k = 0 },
                .{ .rows = 8, .n = 5000, .k = 4990 },
                .{ .rows = 8, .n = 20001, .k = 0 },
            }),
            // the last shape is the served one: seven draft rows against the whole real vocabulary
            .topk => self.setServed(&.{ .{ .rows = 5, .n = 17, .k = 0 }, .{ .rows = 1, .n = 4097, .k = 0 }, .{ .rows = 3, .n = 40000, .k = 0 }, .{ .rows = 7, .n = VALID, .k = 0 } }),
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
            .out = if (self.kind == .topk) .f32 else .bf16,
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
            .conv => self.shape.rows * H,
            .block_qk => 8 * (Q + KV),
            .context_kv => 2 * self.shape.rows * KV,
            .attention, .merge => 8 * Q,
            .topk => 2 * self.shape.rows * LISTS,
        };
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
        return values;
    }
    fn input(self: *Probe, values: []const f32) !host.Ref {
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }
    fn output(self: *Probe, slot: usize, elements: usize, elem: c.Elem) !void {
        self.outs[slot] = try self.rig.output(elements * c.size(elem));
    }
    /// The slots of the context ring: the committed count, or the window's 2,048 once the context has outgrown it.
    fn ring(self: *const Probe) usize {
        return @min(@max(self.shape.n, 1), 2048);
    }
    fn first(self: *const Probe, row: usize) usize {
        const low = if (self.shape.n + row > 2047) self.shape.n + row - 2047 else 0;
        return @max(self.shape.k, low);
    }
    /// The top-k's whole-vocabulary shape is checked in every list of the first and last rows' groups, not 1.4 million values.
    fn sampled(self: *const Probe, i: usize) bool {
        _ = self;
        _ = i;
        return true;
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
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const rows = shape.rows;
        switch (self.kind) {
            .conv => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(rows * DYN, rng);
                self.f[2] = try self.floats(4 * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j]);
                try self.output(0, rows * H, .bf16);
            },
            .block_qk => {
                const sizes = [_]usize{ 8 * Q, 8 * KV, 128, 128 };
                for (sizes, 0..) |n, j| {
                    self.f[j] = try self.floats(n, rng);
                    rounded(self.f[j]);
                    self.refs[j] = try self.input(self.f[j]);
                }
                if (nf) {
                    self.f[0][0] = std.math.nan(f32);
                    self.refs[0] = try self.input(self.f[0]);
                }
                try self.output(0, 8 * Q, .bf16);
                try self.output(1, 8 * KV, .bf16);
            },
            .context_kv => {
                self.f[0] = try self.floats(rows * KV, rng);
                self.f[1] = try self.floats(rows * KV, rng);
                self.f[2] = try self.floats(128, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j]);
                try self.output(0, rows * KV, .bf16);
                try self.output(1, rows * KV, .bf16);
            },
            .attention, .merge => {
                const slots = self.ring();
                const sizes = [_]usize{ 8 * Q, 8 * KV, 8 * KV, slots * KV, slots * KV };
                for (sizes, 0..) |n, j| {
                    self.f[j] = try self.floats(n, rng);
                    rounded(self.f[j]);
                    self.refs[j] = try self.input(self.f[j]);
                }
                if (nf) {
                    self.f[0][0] = std.math.nan(f32);
                    self.refs[0] = try self.input(self.f[0]);
                }
                self.part = try self.rig.scratch(8 * host.kv_heads * host.max_chunks * host.part_floats * 4);
                try self.output(0, 8 * Q, .bf16);
            },
            .topk => {
                self.f[0] = try self.floats(rows * shape.n, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[0]);
                // a value no other row holds, at an id that is not a group's first, and ties for the 16th place
                if (!nf) {
                    self.f[0][shape.n - 1] = 3.0;
                    if (shape.n > 40) @memset(self.f[0][20..40], 2.0);
                }
                self.refs[0] = try self.input(self.f[0]);
                try self.output(0, rows * LISTS, .f32);
                try self.output(1, rows * LISTS, .f32);
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
            .conv => {
                const branch: usize = self.shape.k;
                try host.conv(k, s, R[0], plus(R[1], branch * 288 * 2), plus(R[2], branch * 2 * H * 2), O[0], n);
            },
            .block_qk => try host.blockQk(k, s, R[0], R[1], R[2], R[3], O[0], O[1], @intCast(self.shape.k)),
            .context_kv => try host.commitContext(k, s, R[0], R[1], R[2], O[0], O[1], @intCast(self.shape.k), n, n),
            .attention => {
                try host.attendPartial(k, s, R[0], R[1], R[2], R[3], R[4], self.part, @intCast(self.shape.n), @intCast(self.ring()), @intCast(self.shape.k));
                try host.attendMerge(self.production, s, self.part, O[0], @intCast(self.shape.n), @intCast(self.shape.k));
            },
            .merge => {
                try host.attendPartial(self.production, s, R[0], R[1], R[2], R[3], R[4], self.part, @intCast(self.shape.n), @intCast(self.ring()), @intCast(self.shape.k));
                try host.attendMerge(k, s, self.part, O[0], @intCast(self.shape.n), @intCast(self.shape.k));
            },
            .topk => try host.topKeep(k, s, R[0], @intCast(self.shape.n), @intCast(self.shape.n), n, O[0], O[1]),
        }
        try s.synchronize();
    }

    fn convReference(self: *Probe, comptime F: type, want: []f64) void {
        const branch: usize = self.shape.k;
        for (0..self.shape.rows) |row| for (0..H) |ch| {
            var sum: F = 0;
            for (0..2) |off| {
                if (row < off) continue;
                const base: F = self.f[2][branch * 2 * H + off * H + ch];
                const dyn: F = self.f[1][row * DYN + branch * 288 + off * 144 + ch / 16];
                sum += (base + dyn) * @as(F, self.f[0][(row - off) * H + ch]);
            }
            want[row * H + ch] = bf(sum);
        };
    }
    fn qkReference(self: *Probe, comptime F: type, want: []f64) void {
        for (0..8) |row| for (0..32) |h| for (0..128) |d| {
            const at = row * Q + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const scale = rms(F, self.f[0], row * Q + h * 128, 128);
            const x: F = @floatCast(bf(@as(F, self.f[0][at]) * scale * @as(F, self.f[2][d])));
            const other: F = @floatCast(bf(@as(F, self.f[0][row * Q + h * 128 + partner]) * scale * @as(F, self.f[2][partner])));
            want[at] = rope(F, d, self.shape.k + row, x, other);
            if (self.terms) |t| t[at] = ropeTerms(F, d, self.shape.k + row, x, other);
        };
        const offset = 8 * Q;
        for (0..8) |row| for (0..8) |h| for (0..128) |d| {
            const at = row * KV + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const scale = rms(F, self.f[1], row * KV + h * 128, 128);
            const x: F = @floatCast(bf(@as(F, self.f[1][at]) * scale * @as(F, self.f[3][d])));
            const other: F = @floatCast(bf(@as(F, self.f[1][row * KV + h * 128 + partner]) * scale * @as(F, self.f[3][partner])));
            want[offset + at] = rope(F, d, self.shape.k + row, x, other);
            if (self.terms) |t| t[offset + at] = ropeTerms(F, d, self.shape.k + row, x, other);
        };
    }
    fn contextReference(self: *Probe, comptime F: type, want: []f64) void {
        const rows = self.shape.rows;
        for (0..rows) |row| for (0..8) |h| for (0..128) |d| {
            const src = row * KV + h * 128 + d;
            const dst = ((self.shape.k + row) % rows) * KV + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const scale = rms(F, self.f[0], row * KV + h * 128, 128);
            const x: F = @floatCast(bf(@as(F, self.f[0][src]) * scale * @as(F, self.f[2][d])));
            const other: F = @floatCast(bf(@as(F, self.f[0][row * KV + h * 128 + partner]) * scale * @as(F, self.f[2][partner])));
            want[dst] = rope(F, d, self.shape.k + row, x, other);
            if (self.terms) |t| t[dst] = ropeTerms(F, d, self.shape.k + row, x, other);
            want[rows * KV + dst] = self.f[1][src];
        };
    }
    fn attentionReference(self: *Probe, comptime F: type, want: []f64) !void {
        const committed = self.shape.n;
        const slots = self.ring();
        for (0..8) |row| {
            const first_kept = self.first(row);
            const retained = if (committed > first_kept) committed - first_kept else 0;
            const score_count = retained + 8;
            const scores = try self.a.alloc(F, score_count);
            defer self.a.free(scores);
            for (0..32) |h| {
                const kv = h / 4;
                const q_at = row * Q + h * 128;
                var peak: F = -std.math.inf(F);
                for (0..score_count) |j| {
                    var dot: F = 0;
                    const k_at = if (j < retained) ((first_kept + j) % slots) * KV + kv * 128 else (j - retained) * KV + kv * 128;
                    const keys = if (j < retained) self.f[3] else self.f[1];
                    for (0..128) |d| dot += @as(F, self.f[0][q_at + d]) * @as(F, keys[k_at + d]);
                    scores[j] = dot * @as(F, 0x1.6a09e6p-4);
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
                    if (self.terms) |t| t[q_at + d] = size;
                }
            }
        }
    }
    /// Per row and group the 16 largest of the group's ids (value descending, lower id first), padded with -inf and the
    /// largest id; NaN and -inf never enter.
    fn topkReference(self: *Probe, want: []f64) !void {
        const rows = self.shape.rows;
        const n = self.shape.n;
        const per = (n + host.topk_groups - 1) / host.topk_groups;
        const List = struct {
            v: f32,
            id: i32,
            fn before(_: void, x: @This(), y: @This()) bool {
                return x.v > y.v or (x.v == y.v and x.id < y.id);
            }
        };
        var items: std.ArrayList(List) = .empty;
        defer items.deinit(self.a);
        for (0..rows) |row| for (0..host.topk_groups) |group| {
            items.clearRetainingCapacity();
            const a = group * per;
            const b = @min(n, a + per);
            var j = a;
            while (j < b) : (j += 1) {
                const v = self.f[0][row * n + j];
                if (v > -std.math.inf(f32)) try items.append(self.a, .{ .v = v, .id = @intCast(j) });
            }
            std.mem.sort(List, items.items, {}, List.before);
            for (0..host.topk) |i| {
                const at = (row * host.topk_groups + group) * host.topk + i;
                want[at] = if (i < items.items.len) items.items[i].v else -std.math.inf(f32);
                want[rows * LISTS + at] = if (i < items.items.len) @floatFromInt(items.items[i].id) else 0x7fffffff;
            }
        };
    }

    fn referenceAs(self: *Probe, comptime F: type, want: []f64) !void {
        switch (self.kind) {
            .conv => self.convReference(F, want),
            .block_qk => self.qkReference(F, want),
            .context_kv => self.contextReference(F, want),
            .attention, .merge => try self.attentionReference(F, want),
            .topk => try self.topkReference(want),
        }
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        try self.referenceAs(f32, want);
        for (mag, want) |*m, w| m.* = 1 + @abs(w);
    }

    fn ulps(got: f64, want: f64, terms: f64) f64 {
        if (got == want) return 0;
        if (!std.math.isFinite(got) or !std.math.isFinite(want)) return std.math.inf(f64);
        const unit = std.math.pow(f64, 2, @floor(std.math.log2(@max(@abs(want), terms, 0x1p-120))) - 7);
        return @abs(got - want) / unit;
    }

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
        for (want, 0..) |w, i| {
            const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
            const got = c.fromBf16(bits);
            const u = ulps(got, w, terms[i]);
            worst = @max(worst, u);
            if (u != 0) differ += 1;
        }
        self.worst_ulps = @max(self.worst_ulps, worst);
        std.debug.print("F64  {s} at {d}x{d}x{d}: worst {d:.4} BF16 ulps from float64; {d} of {d} outputs differ from float64's BF16 value\n", .{ self.kind.label(), self.shape.rows, self.shape.n, self.shape.k, worst, differ, want.len });
        if (!(worst <= 4.0)) return error.Dflash2Float64Disagrees;
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
        if (self.kind == .topk) {
            const half = self.count() / 2;
            for (want, 0..) |w, i| {
                if (i < half) {
                    const got = c.read(.f32, bytes, i);
                    if (got != w) return "a group's list of largest logits differs from the reference";
                } else {
                    const got = std.mem.readInt(i32, bytes[4 * i ..][0..4], .little);
                    if (@as(f64, @floatFromInt(got)) != w) return "a group's list of ids differs from the reference";
                }
            }
            return null;
        }
        for (want, 0..) |w, i| {
            const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
            const expected = c.toBf16(@floatCast(w));
            if (std.math.isNan(w)) {
                if (!std.math.isNan(c.fromBf16(bits))) return "BF16 NaN was not propagated";
            } else if (bits != expected) {
                const relative: f64 = switch (self.kind) {
                    .block_qk, .context_kv => 0.02,
                    .attention, .merge => 0.03,
                    else => 0,
                };
                const got = c.fromBf16(bits);
                if (@abs(@as(f64, got) - w) > @max(1e-5, relative * @abs(w))) {
                    if (!self.control) std.debug.print("DFLASH2 {s} output {d}: got {d} (0x{x}) want {d}\n", .{ self.kind.label(), i, got, bits, w });
                    return "BF16 endpoint differs from the explicit FP32/FP64 reference";
                }
            }
        }
        if (gate == .shapes and !self.control) self.againstFloat64(bytes) catch |e| return @errorName(e);
        return null;
    }
};

/// The 12 registrations: each kernel as served, and its planted copy.
pub const Suite = struct {
    a: std.mem.Allocator,
    dev: *rig_.Dev,
    production: host.Kernels,
    planted: host.Kernels,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Suite {
        try controlsApplyOnce();
        try dev.addImage("inkling2_dflash2", &images.inkling2_dflash2);
        try dev.addImage("contract_inkling2_dflash2_controls", &images.contract_inkling2_dflash2_controls);
        var production = try host.Kernels.load(dev.named("inkling2_dflash2").?.*, false);
        errdefer production.deinit();
        const planted = try host.Kernels.load(dev.named("contract_inkling2_dflash2_controls").?.*, true);
        return .{ .a = a, .dev = dev, .production = production, .planted = planted };
    }

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel)) !void {
        for (std.enums.values(host.Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = Probe.init(s.a, s.dev, if (control) &s.planted else &s.production, &s.production, kind, control);
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

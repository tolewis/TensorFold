//! Inkling 2 Mini runtime helpers on XPU under FP64 references, guarded buffers and the production launches.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const host = @import("inkling2_xpu_runtime");

const c = core.spec;
const g = core.gates;
const p = host.plan;

fn depth(_: c.Shape) usize {
    return 1;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

/// The name a registration carries: `xpu inkling2 runtime <part> <symbol>`, with `control: ` before a planted one.
pub fn label(kind: p.Kind, control: bool) []const u8 {
    return switch (kind) {
        inline else => |k| if (control) "control: " ++ comptime base(k) else comptime base(k),
    };
}

fn base(comptime k: p.Kind) []const u8 {
    return "xpu inkling2 runtime " ++ @tagName(k) ++ " " ++ comptime k.symbol();
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
    f: [2][]f32 = @splat(&.{}),
    refs: [2]?*anyopaque = @splat(null),
    blocks: ?*anyopaque = null,
    blocks_before: []u16 = &.{},
    out: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kind: p.Kind, control: bool) !Probe {
        const module = (try dev.module(if (control) .ink2_runtime_ctl else .ink2_runtime)).*;
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .dev = dev },
            .kind = kind,
            .control = control,
            .pipes = try host.Pipelines.load(module, control),
        };
        self.served = switch (kind) {
            .attention_partial => &.{
                .{ .rows = 1, .n = p.hidden, .k = 0 },
                .{ .rows = 5, .n = p.hidden, .k = 1 },
                .{ .rows = 3, .n = p.hidden, .k = 2 },
                .{ .rows = 2, .n = p.hidden, .k = 3 },
                .{ .rows = 7, .n = p.hidden, .k = 4 },
            },
            .copy_bf16, .copy_f32 => &.{
                .{ .rows = 1, .n = p.hidden, .k = 1 },
                .{ .rows = 5, .n = 1024, .k = 1 },
                .{ .rows = 7, .n = p.hidden, .k = 1 },
                .{ .rows = 16, .n = 1024, .k = 1 },
                .{ .rows = 32, .n = p.hidden, .k = 1 },
            },
            .capture_tap => &.{
                .{ .rows = 1, .n = p.hidden, .k = 0 },
                .{ .rows = 5, .n = p.hidden, .k = 1 },
                .{ .rows = 7, .n = p.hidden, .k = 2 },
                .{ .rows = 16, .n = p.hidden, .k = 3 },
                .{ .rows = 32, .n = p.hidden, .k = 3 },
            },
            else => &.{
                .{ .rows = 1, .n = p.hidden, .k = 1 },
                .{ .rows = 5, .n = p.hidden, .k = 1 },
                .{ .rows = 7, .n = p.hidden, .k = 1 },
                .{ .rows = 16, .n = p.hidden, .k = 1 },
                .{ .rows = 32, .n = p.hidden, .k = 1 },
            },
        };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.pipes.deinit();
        self.rig.deinit();
        self.arena.deinit();
    }

    const vt_plain = blk: {
        var vt = g.hooks(Probe);
        vt.check = null;
        break :blk vt;
    };
    const vt_checked = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = if (self.kind == .attention_partial) &vt_checked else &vt_plain, .control = if (self.control) .shapes else null, .spec = .{
            .name = label(self.kind, self.control),
            .accum = .none,
            .depth = depth,
            .ops = if (self.kind == .attention_partial or self.kind == .mlp_partial) 1 else 0,
            .out = switch (self.kind) {
                .copy_bf16, .mlp_partial, .capture_tap => .bf16,
                .copy_f32, .widen, .attention_partial => .f32,
            },
            .served = self.served,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "pointwise moves and single additions have no reduction accumulator to discriminate" },
        } };
    }
    pub fn count(self: *Probe) usize {
        return p.outputCount(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
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
        self.blocks_before = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        self.f[0] = try self.floats(n, rng);
        switch (self.kind) {
            .copy_bf16, .widen, .capture_tap => round(self.f[0]),
            .attention_partial => {
                self.f[1] = try self.floats(n, rng);
                round(self.f[0]);
                round(self.f[1]);
                const blocks = try self.arena.allocator().alloc(u16, shape.rows * p.block_slots * p.hidden);
                for (blocks) |*b| b.* = c.toBf16(value(rng));
                self.blocks_before = blocks;
                self.blocks = try self.rig.scratch(std.mem.sliceAsBytes(blocks).len);
                try self.rig.put(self.blocks, std.mem.sliceAsBytes(blocks));
                self.refs[1] = try self.input(self.f[1], true);
            },
            .mlp_partial => {
                self.f[1] = try self.floats(n, rng);
                round(self.f[1]);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .copy_f32 => {},
        }
        if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
        self.refs[0] = try self.input(self.f[0], self.kind == .copy_bf16 or self.kind == .widen or self.kind == .attention_partial or self.kind == .capture_tap);
        const elem: c.Elem = switch (self.kind) {
            .copy_bf16, .mlp_partial, .capture_tap => .bf16,
            .copy_f32, .widen, .attention_partial => .f32,
        };
        self.out = try self.rig.output(self.count() * c.size(elem));
        if (self.kind == .capture_tap) {
            // The channels a launch does not write must read back as they were: bf16(1) everywhere.
            const untouched = try self.arena.allocator().alloc(u16, self.count());
            @memset(untouched, c.toBf16(1));
            try self.rig.put(self.out, std.mem.sliceAsBytes(untouched));
        }
        return true;
    }

    pub fn run(self: *Probe) !void {
        const s = self.rig.dev.s;
        const pl = &self.pipes;
        const rows = self.shape.rows;
        switch (self.kind) {
            .copy_bf16 => try host.copyBf16(pl, s, self.refs[0], self.out, rows, self.shape.n),
            .copy_f32 => try host.copyF32(pl, s, self.refs[0], self.out, rows, self.shape.n),
            .widen => try host.widen(pl, s, self.refs[0], self.out, rows, self.shape.n),
            .attention_partial => try host.attentionPartial(pl, s, self.refs[0], self.refs[1], self.blocks, self.out, rows, self.shape.k),
            .mlp_partial => try host.mlpPartial(pl, s, self.refs[0], self.refs[1], self.out, rows),
            .capture_tap => try host.captureTap(pl, s, self.refs[0], self.out, rows, self.shape.k),
        }
        try s.synchronize();
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        if (self.kind == .capture_tap) {
            for (want, mag, 0..) |*w, *m, i| {
                const row = i / (p.tap_channels * p.hidden);
                const channel = (i / p.hidden) % p.tap_channels;
                const x = self.f[0][row * p.hidden + i % p.hidden];
                w.* = if (channel == self.shape.k) x else 1;
                m.* = @abs(@as(f64, x));
            }
            return;
        }
        for (want, mag, 0..) |*w, *m, i| {
            const x = self.f[0][i];
            const y = if (self.f[1].len == 0) 0 else self.f[1][i];
            w.* = switch (self.kind) {
                .copy_bf16, .copy_f32, .widen, .capture_tap => x,
                .attention_partial => p.attentionPartial(x, y, self.shape.k),
                .mlp_partial => p.mlpPartial(x, y),
            };
            m.* = @abs(@as(f64, x)) + @abs(@as(f64, y));
        }
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (self.kind != .attention_partial or (gate != .shapes and gate != .nonfinite)) return "numeric hooks must be used";
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        const bytes = try self.rig.outputBytes();
        var planted: usize = 0;
        for (want, 0..) |w, i| {
            const got = c.read(.f32, bytes, i);
            if (c.classOf(got) != c.classOf(w)) return "partial's floating-point class differs";
            if (c.classOf(w) != .finite) planted += 1;
            if (gate == .shapes and std.math.isFinite(w) and got != w) return "FP32 partial differs from the FP64 reference's explicit rounding";
        }
        if (gate == .nonfinite and (planted == 0 or planted == want.len)) return "the planted NaN does not discriminate finite and non-finite partials";
        const block_bytes = try self.rig.bytes(self.blocks, std.mem.sliceAsBytes(self.blocks_before).len);
        const got_blocks = std.mem.bytesAsSlice(u16, block_bytes);
        for (0..self.shape.rows) |row| for (0..p.block_slots) |slot| for (0..p.hidden) |d| {
            const at = (row * p.block_slots + slot) * p.hidden + d;
            const expected = if (self.shape.k != 0 and slot == self.shape.k - 1)
                c.toBf16(self.f[0][row * p.hidden + d])
            else
                self.blocks_before[at];
            if (got_blocks[at] != expected) return "completed-block row/slot layout differs";
        };
        return null;
    }
};

/// Every runtime kernel and, after each, its planted control, filtered by a substring of the registration's name.
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

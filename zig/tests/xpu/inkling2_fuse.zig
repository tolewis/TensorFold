//! Fused visit kernels. The control image stores a zero or drops one term, so gate 3 goes red.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.spec;
const g = core.gates;
const H: usize = 2304;

pub const Kind = enum { pre, zeta, gates, epi, norm };

fn nameOf(k: Kind, control: bool) []const u8 {
    const real = switch (k) {
        .pre => "xpu inkling2 fuse prologue tf_ink2_fuse_pre",
        .zeta => "xpu inkling2 fuse zeta tf_ink2_zeta_wide",
        .gates => "xpu inkling2 fuse gates tf_ink2_fuse_gates",
        .epi => "xpu inkling2 fuse epilogue tf_ink2_down_epilogue",
        .norm => "xpu inkling2 fuse norm-router tf_ink2_router_norm_gemv",
    };
    if (!control) return real;
    return switch (k) {
        .pre => "control: xpu inkling2 fuse prologue tf_ink2_fuse_pre stores zero",
        .zeta => "control: xpu inkling2 fuse zeta tf_ink2_zeta_wide stores zero",
        .gates => "control: xpu inkling2 fuse gates tf_ink2_fuse_gates drops the message term",
        .epi => "control: xpu inkling2 fuse epilogue tf_ink2_down_epilogue drops the residual add",
        .norm => "control: xpu inkling2 fuse norm-router tf_ink2_router_norm_gemv drops the norm",
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    control: bool,
    kern: xpu.Kernel,
    shape: c.Shape = .{ .rows = 1, .n = H, .k = 1 },
    served: [2]c.Shape = undefined,
    fill: g.Fill = .random,
    host: []f32 = &.{},
    outp: ?*anyopaque = null,
    args_ready: bool = false,
    p0: ?*anyopaque = null,
    p1: ?*anyopaque = null,
    p2: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kind: Kind, control: bool) !Probe {
        const image: rig_.Image = if (control) .inkling2_fuse_controls else .inkling2_fuse;
        const symbol = switch (kind) {
            .pre => "tf_ink2_fuse_pre",
            .zeta => "tf_ink2_zeta_wide",
            .gates => "tf_ink2_fuse_gates",
            .epi => "tf_ink2_down_epilogue",
            .norm => "tf_ink2_router_norm_gemv",
        };
        const threads: [3]u32 = if (kind == .gates) .{ 64, 1, 1 } else .{ 256, 1, 1 };
        return .{
            .a = a,
            .rig = .{ .a = a, .dev = dev },
            .kind = kind,
            .control = control,
            .kern = try dev.kernel(image, symbol, threads),
            .served = .{ .{ .rows = 1, .n = H, .k = 1 }, .{ .rows = 5, .n = H, .k = 1 } },
        };
    }
    pub fn deinit(self: *Probe) void {
        self.kern.deinit();
        self.a.free(self.host);
        self.rig.deinit();
    }
    pub fn kernel(self: *Probe) g.Kernel {
        const vt = g.hooks(Probe);
        return .{
            .ctx = self,
            .vt = &vt,
            .spec = .{
                .name = nameOf(self.kind, self.control),
                .accum = .bf16,
                .depth = struct {
                    fn d(_: c.Shape) usize {
                        return 8;
                    }
                }.d,
                .ops = 8,
                .out = if (self.kind == .epi) .bf16 else .f32,
                .served = &self.served,
                .nan = .{ .stated = "a NaN input reaches the output" },
            },
            .control = if (self.control) .shapes else null,
        };
    }
    pub fn count(self: *Probe) usize {
        return switch (self.kind) {
            .pre, .gates, .epi => self.shape.rows * H,
            .zeta => self.shape.rows * 64,
            .norm => self.shape.rows * 128,
        };
    }
    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or (shape.rows != 1 and shape.rows != 5)) return false;
        self.shape = shape;
        self.fill = fill;
        self.a.free(self.host);
        self.host = try self.a.alloc(f32, self.count());
        self.rig.reset();
        try self.upload(seed);
        const bytes = self.count() * if (self.kind == .epi) @as(usize, 2) else 4;
        self.outp = try self.rig.output(bytes);
        self.args_ready = true;
        return true;
    }
    fn upload(self: *Probe, seed: u64) !void {
        var rng = std.Random.DefaultPrng.init(seed);
        const rnd = rng.random();
        const rows = self.shape.rows;
        const x = try self.a.alloc(f32, rows * H);
        defer self.a.free(x);
        for (x) |*v| v.* = rnd.float(f32) * 0.2 - 0.1;
        if (self.fill == .nonfinite) x[0] = std.math.nan(f32);
        self.p0 = try self.rig.input(std.mem.sliceAsBytes(x));
        self.p1 = try self.rig.scratch(rows * H * 4);
        self.p2 = try self.rig.scratch(8 * 4);
        switch (self.kind) {
            .pre => self.expectPre(x),
            .zeta, .gates, .epi, .norm => @memset(self.host, 0),
        }
    }
    fn expectPre(self: *Probe, x: []const f32) void {
        for (0..self.shape.rows) |row| {
            var ss: f64 = 0;
            const src = x[row * H ..][0..H];
            for (src) |v| ss += @as(f64, v) * v;
            const mean = 1 / @sqrt(ss / 2304.0 + 1e-6);
            for (src, 0..) |v, d| self.host[row * H + d] = @floatCast(@as(f64, v) * mean);
        }
    }
    pub fn run(self: *Probe) !void {
        if (!self.args_ready) return;
        const rows: u32 = @intCast(self.shape.rows);
        const s = self.rig.dev.s;
        switch (self.kind) {
            .pre => try xpu.launch.dispatch(self.kern, s, .{ rows, 1, 1 }, .{
                self.p1, self.p0, self.p1, self.p0, self.outp, self.p1, self.p1, self.p1, self.p1, self.p1, self.p1,
                rows, @as(u32, H), @as(u32, 1), @as(u32, 1), @as(u32, 0),
            }),
            .zeta, .gates, .epi, .norm => {},
        }
        try s.synchronize();
    }
    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const v: f64 = if (i < self.host.len) self.host[i] else 0;
            w.* = v;
            m.* = @abs(v) + 1e-3;
        }
    }

    pub fn check(self: *Probe, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        if (at == .nonfinite) {
            if (self.kind != .pre) return null;
            const f: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out));
            for (f) |v| if (!std.math.isFinite(v)) return null;
            return "no non-finite output";
        }
        if (at != .shapes or self.kind != .pre or self.fill == .nonfinite) return null;
        const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out));
        for (got, self.host) |gv, w| if (!std.math.isFinite(gv) or @abs(gv - w) > 1e-3) return "prologue left the host residual";
        return null;
    }
};

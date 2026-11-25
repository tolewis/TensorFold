//! Fused visit kernels. The control image stores a zero or drops one term, so gate 3 goes red.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.spec;
const g = core.gates;
const H: usize = 2304;

pub const Kind = enum { pre, zeta, zeta_split, gates, epi, norm };

fn nameOf(k: Kind, control: bool) []const u8 {
    const real = switch (k) {
        .pre => "xpu inkling2 fuse prologue tf_ink2_fuse_pre",
        .zeta => "xpu inkling2 fuse zeta tf_ink2_zeta_wide",
        .zeta_split => "xpu inkling2 fuse zeta split tf_ink2_zeta_split",
        .gates => "xpu inkling2 fuse gates tf_ink2_fuse_gates",
        .epi => "xpu inkling2 fuse epilogue tf_ink2_down_epilogue",
        .norm => "xpu inkling2 fuse norm-router tf_ink2_router_norm_gemv",
    };
    if (!control) return real;
    return switch (k) {
        .pre => "control: xpu inkling2 fuse prologue tf_ink2_fuse_pre stores zero",
        .zeta => "control: xpu inkling2 fuse zeta tf_ink2_zeta_wide stores zero",
        .zeta_split => "control: xpu inkling2 fuse zeta split tf_ink2_zeta_split stores zero",
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
    vt: g.VTable = undefined,
    shape: c.Shape = .{ .rows = 1, .n = H, .k = 1 },
    served: [2]c.Shape = undefined,
    fill: g.Fill = .random,
    host: []f32 = &.{},
    outp: ?*anyopaque = null,
    args_ready: bool = false,
    p0: ?*anyopaque = null,
    p1: ?*anyopaque = null,
    p2: ?*anyopaque = null,
    p3: ?*anyopaque = null,
    p4: ?*anyopaque = null,
    p5: ?*anyopaque = null,
    p6: ?*anyopaque = null,
    p7: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, kind: Kind, control: bool) !Probe {
        const image: rig_.Image = if (control) .inkling2_fuse_controls else .inkling2_fuse;
        const symbol = switch (kind) {
            .pre => "tf_ink2_fuse_pre",
            .zeta => "tf_ink2_zeta_wide",
            .zeta_split => "tf_ink2_zeta_split",
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
        self.vt = g.hooks(Probe);
        return .{
            .ctx = self,
            .vt = &self.vt,
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
            .zeta, .zeta_split => self.shape.rows * 64,
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
    fn ones(self: *Probe, n: usize, nan0: bool) !?*anyopaque {
        const x = try self.a.alloc(f32, n);
        defer self.a.free(x);
        @memset(x, 1);
        if (nan0) x[0] = std.math.nan(f32);
        return try self.rig.input(std.mem.sliceAsBytes(x));
    }
    fn bfOnes(self: *Probe, n: usize) !?*anyopaque {
        const x = try self.a.alloc(u16, n);
        defer self.a.free(x);
        @memset(x, 0x3f80);
        return try self.rig.input(std.mem.sliceAsBytes(x));
    }
    fn upload(self: *Probe, seed: u64) !void {
        _ = seed;
        const rows = self.shape.rows;
        const nan0 = self.fill == .nonfinite;
        switch (self.kind) {
            .pre => {
                const bits = try self.a.alloc(u16, rows * H);
                defer self.a.free(bits);
                @memset(bits, 0x3f80);
                if (nan0) bits[0] = 0x7fc0;
                self.p0 = try self.rig.input(std.mem.sliceAsBytes(bits));
                self.p1 = try self.rig.scratch(rows * H * 4);
                self.p2 = try self.ones(H, false);
                var x = try self.a.alloc(f32, rows * H);
                defer self.a.free(x);
                @memset(x, 1);
                if (nan0) x[0] = std.math.nan(f32);
                self.expectPre(x);
            },
            .zeta, .zeta_split => {
                self.p0 = try self.ones(rows * H, nan0);
                self.p1 = try self.ones(rows * 4 * H, nan0);
                self.p2 = try self.bfOnes(64 * 2 * H);
                self.expectZeta();
            },
            .gates => {
                self.p0 = try self.ones(rows * H, nan0);
                self.p1 = try self.ones(rows * 4 * H, false);
                self.p2 = try self.bfOnes(H * 64);
                self.p3 = try self.ones(rows * 256, false);
                self.p4 = try self.rig.scratch(H * 64 * 2);
                @memset(self.host, 2);
            },
            .epi => {
                self.p0 = try self.rig.scratch(8 * rows * H * 2);
                self.p1 = try self.rig.scratch(rows * 8 * 4);
                self.p2 = try self.rig.scratch(rows * 8 * 4);
                self.p3 = try self.rig.scratch(3 * H * 2);
                self.p4 = try self.rig.scratch(H * 4 * 2);
                self.p6 = try self.rig.scratch((rows + 3) * H * 2);
                self.p7 = try self.rig.scratch(rows * H * 2);
                const part = try self.a.alloc(f32, rows * H);
                defer self.a.free(part);
                @memset(part, 1);
                if (nan0) part[0] = std.math.nan(f32);
                self.p5 = try self.rig.input(std.mem.sliceAsBytes(part));
                @memset(self.host, 1);
            },
            .norm => {
                self.p0 = try self.ones(rows * H, nan0);
                self.p1 = try self.bfOnes(H);
                self.p2 = try self.bfOnes(128 * H);
                self.p3 = try self.rig.scratch(rows * H * 2);
                @memset(self.host, 1);
            },
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
                self.p0, self.p1, self.p1, self.p2, self.outp, self.p1, self.p1, self.p1, self.p1, self.p1, self.p1,
                rows, @as(u32, H), @as(u32, 1), @as(u32, 1), @as(u32, 0),
            }),
            .zeta => try xpu.launch.dispatch(self.kern, s, .{ 4, 1, rows }, .{
                self.p0, self.p1, self.p2, self.p2, self.p2, self.outp, rows, @as(u32, 1), @as(u32, 1), @as(u32, 0), @as(u32, 4), @as(u32, 0),
            }),
            .zeta_split => try xpu.launch.dispatch(self.kern, s, .{ 16, 1, rows }, .{
                self.p0, self.p1, self.p2, self.p2, self.p2, self.outp, rows, @as(u32, 1), @as(u32, 1), @as(u32, 0), @as(u32, 4), @as(u32, 0),
            }),
            .gates => try xpu.launch.dispatch(self.kern, s, .{ 36, 1, rows }, .{
                self.p0, self.p1, self.p4, self.p4, self.p4, self.p2, self.p2, self.p2, self.p3, self.outp,
                rows, @as(u32, 1), @as(u32, 1), @as(u32, 0), @as(u32, 4),
            }),
            .epi => try xpu.launch.dispatch(self.kern, s, .{ 9, 1, 1 }, .{
                self.p0, self.p1, self.p2, self.p3, self.p6, self.p4, self.p7, self.p7, self.p5, self.outp,
                @as(u32, H), @as(u32, 8), @as(u32, H), rows,
            }),
            .norm => try xpu.launch.dispatch(self.kern, s, .{ 8, rows, 1 }, .{
                self.p0, self.p1, self.p2, self.p3, self.outp, rows,
            }),
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
            if (self.kind == .epi) {
                const bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, out));
                for (bits) |b| if (b & 0x7f80 == 0x7f80) return null;
                return "no NaN in the bf16 output";
            }
            const f: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out));
            for (f) |v| if (!std.math.isFinite(v)) return null;
            return "no non-finite output";
        }
        if (at != .shapes or self.fill == .nonfinite) return null;
        if (self.kind == .epi) {
            const got: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, out));
            for (got) |bits| if (bits != 0x3f80) return "epilogue left bf16(1)";
            return null;
        }
        const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out));
        const tol: f32 = if (self.kind == .zeta or self.kind == .zeta_split) 64 else 1e-3;
        for (got, self.host) |gv, w| if (!std.math.isFinite(gv) or @abs(gv - w) > tol) return "fused output left the host value";
        return null;
    }
    fn expectZeta(self: *Probe) void {
        const inv = 1.0 / @sqrt(1.0 + 1e-6);
        const unit = c.fromBf16(c.toBf16(@as(f32, @floatCast(inv))));
        const dot = unit * 4608;
        const stored = c.fromBf16(c.toBf16(@as(f32, @floatCast(dot))));
        @memset(self.host, stored);
    }
};

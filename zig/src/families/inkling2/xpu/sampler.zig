//! Sampling's candidate selection on the XPU: `tf_ink2_sample_top` takes rows of logits that stay on the device and
//! returns each row's best tokens in the host sampler's order (see kernels/xpu/inkling2/sample.cl). The host copies back
//! the candidates, not the row, and finishes the draw with `lanes.gpu_full.drawTop`.
const std = @import("std");
const xpu = @import("xpu");
const lanes = @import("lanes");
const images = @import("inkling2_xpu_images");
const rt = xpu.rt;
const rule = lanes.gpu_rule;

pub const top_max: u32 = lanes.gpu_full.top_max;

/// One row's request: the sampling rule and how many candidates it needs.
pub const Row = struct {
    inv_t: f32,
    n: u32,

    /// The row of `s`: top_k candidates, or all 1,024 with top_k off.
    pub fn of(s: lanes.Sampling) Row {
        const r = rule.rule(s);
        return .{ .inv_t = r.inv_t, .n = if (s.top_k == 0) top_max else s.top_k };
    }

    /// Whether the device can select this row's candidates: top_k beyond 1,024 needs a deeper sort than the kernel makes.
    pub fn supported(s: lanes.Sampling) bool {
        return s.top_k <= top_max;
    }
};

pub const Sampler = struct {
    gpa: std.mem.Allocator,
    r: *rt.Runtime,
    ctx: *const xpu.Context,
    module: xpu.Module,
    kernel: xpu.Kernel,
    cfg: rt.Buffer,
    out: rt.Buffer,
    host_cfg: []u32,
    host_out: []u32,
    rows_max: usize,

    pub fn init(gpa: std.mem.Allocator, r: *rt.Runtime, ctx: *const xpu.Context, rows_max: usize) !Sampler {
        var module = try xpu.Module.loadNative(ctx, &images.sample);
        errdefer module.unload();
        var kernel = try module.kernel("tf_ink2_sample_top", .{ 256, 1, 1 });
        errdefer kernel.deinit();
        var cfg = try r.alloc(rows_max * 8);
        errdefer cfg.free();
        var out = try r.alloc(rows_max * top_max * 8);
        errdefer out.free();
        const host_cfg = try gpa.alloc(u32, rows_max * 2);
        errdefer gpa.free(host_cfg);
        const host_out = try gpa.alloc(u32, rows_max * top_max * 2);
        return .{ .gpa = gpa, .r = r, .ctx = ctx, .module = module, .kernel = kernel, .cfg = cfg, .out = out, .host_cfg = host_cfg, .host_out = host_out, .rows_max = rows_max };
    }

    pub fn deinit(s: *Sampler) void {
        s.gpa.free(s.host_out);
        s.gpa.free(s.host_cfg);
        s.out.free();
        s.cfg.free();
        s.kernel.deinit();
        s.module.unload();
        s.* = undefined;
    }

    /// The candidates of `rows.len` consecutive rows of `vocab` logits at device address `logits`, waiting for the stream
    /// first. Row r's list is `list(r)` of the returned set, valid until the next call.
    pub fn select(s: *Sampler, logits: ?*anyopaque, vocab: u32, rows: []const Row) !Set {
        if (rows.len == 0 or rows.len > s.rows_max) return error.Inkling2SampleRows;
        var stride: u32 = 1;
        for (rows, 0..) |row, i| {
            if (row.n == 0 or row.n > top_max) return error.Inkling2SampleTop;
            s.host_cfg[2 * i] = @bitCast(row.inv_t);
            s.host_cfg[2 * i + 1] = row.n;
            stride = @max(stride, row.n);
        }
        try s.r.upload(s.cfg, std.mem.sliceAsBytes(s.host_cfg[0 .. rows.len * 2]));
        try xpu.launch.dispatch(s.kernel, s.r.stream(s.ctx), .{ @intCast(rows.len), 1, 1 }, .{ logits, s.cfg.ptr, s.out.ptr, vocab, stride });
        const words = rows.len * stride * 2;
        try s.r.download(std.mem.sliceAsBytes(s.host_out[0..words]), s.out);
        try s.r.sync();
        return .{ .words = s.host_out[0..words], .stride = stride };
    }
};

/// The candidate lists of one `select`.
pub const Set = struct {
    words: []const u32,
    stride: u32,

    /// Row `r`'s first `n` candidates into `into` (at least `n` long), in the host sampler's order.
    pub fn list(set: Set, r: usize, n: u32, into: []rule.Cand) []const rule.Cand {
        const base = r * set.stride * 2;
        for (into[0..n], 0..) |*c, j| c.* = .{ .v = @bitCast(set.words[base + 2 * j]), .i = set.words[base + 2 * j + 1] };
        return into[0..n];
    }
};

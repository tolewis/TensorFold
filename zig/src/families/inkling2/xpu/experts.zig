//! Inkling 2 Mini's routed experts on Intel Xe2 (Level Zero): the host side of kernels/xpu/inkling2/experts.cl.
//! NVFP4 (layers 0 to 19) and MXFP8 (layers 20 to 23) expert weights decode in registers, activations are BF16,
//! every sum is FP32. A decode call gathers the routes of 1 to 16 rows; a prompt call groups the routes by expert.
//! Mirrors metal/experts.zig: the same call shapes and the same device buffers, with raw device addresses.
const std = @import("std");
const xpu = @import("xpu");

pub const hidden: u32 = 2304;
pub const intermediate: u32 = 1152;
pub const expert_count: u32 = 128;
pub const top_k: u32 = 8;

/// The kernels' reduction split: one decode work-group holds this many sub-groups, each over a slice of k
/// (INK2_KS_NV_GU, INK2_KS_NV_DOWN, INK2_KS_MX_GU and INK2_KS_MX_DOWN in the source).
pub const decode_subgroups = struct {
    pub const nvfp4_gu: u32 = 4;
    pub const nvfp4_down: u32 = 3;
    pub const mxfp8_gu: u32 = 4;
    pub const mxfp8_down: u32 = 4;
};
/// The most routes of one expert in one prompt item (INK2_RUN in the source).
pub const run_routes: u32 = 32;
/// The most experts the plan kernel counts (INK2_MAX_EXPERTS in the source).
pub const max_experts: u32 = 512;

pub const Format = enum {
    nvfp4,
    mxfp8,

    pub fn block(f: Format) u32 {
        return if (f == .nvfp4) 16 else 32;
    }
    pub fn codeBytes(f: Format, k: u32) u32 {
        return if (f == .nvfp4) k / 2 else k;
    }
};

pub const Call = struct {
    rows: u32,
    k: u32,
    n: u32,
    experts: u32 = expert_count,
    topk: u32 = top_k,
    format: Format,
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.n == 0 or c.experts == 0 or c.topk == 0 or c.topk > c.experts)
        return error.BadExpertShape;
    if (c.experts > max_experts) return error.BadExpertShape;
    if (c.k % c.format.block() != 0) return error.BadExpertBlock;
}

/// The routes alone (the combine and the plan have no weight block).
pub fn checkRoutes(c: Call) !void {
    if (c.rows == 0 or c.n == 0 or c.experts == 0 or c.topk == 0 or c.topk > c.experts or c.experts > max_experts)
        return error.BadExpertShape;
}

pub fn formatForLayer(layer: u32) !Format {
    if (layer < 20) return .nvfp4;
    if (layer < 24) return .mxfp8;
    return error.BadExpertLayer;
}

/// The routes of a call: one per (row, slot).
pub fn routes(c: Call) u32 {
    return c.rows * c.topk;
}

/// The most prompt items the plan can list: every expert's last run is short, the others hold `run_routes` each.
pub fn maxItems(c: Call) u32 {
    const r = routes(c);
    return @min(r, c.experts + r / run_routes);
}

pub const Op = enum {
    nvfp4_gu,
    nvfp4_down,
    mxfp8_gu,
    mxfp8_down,
    nvfp4_quantize,
    mxfp8_quantize,
    combine,
    plan,
    nvfp4_gu_prompt,
    nvfp4_down_prompt,
    mxfp8_gu_prompt,
    mxfp8_down_prompt,

    /// The kernel's name without the image's prefix.
    pub fn name(o: Op) []const u8 {
        return @tagName(o);
    }

    /// The work-group the kernel is built for: its reqd_work_group_size.
    pub fn threads(o: Op) [3]u32 {
        return switch (o) {
            .nvfp4_gu => .{ 16 * decode_subgroups.nvfp4_gu, 1, 1 },
            .nvfp4_down => .{ 16 * decode_subgroups.nvfp4_down, 1, 1 },
            .mxfp8_gu => .{ 16 * decode_subgroups.mxfp8_gu, 1, 1 },
            .mxfp8_down => .{ 16 * decode_subgroups.mxfp8_down, 1, 1 },
            .nvfp4_quantize, .mxfp8_quantize, .plan => .{ 256, 1, 1 },
            .combine => .{ 64, 1, 1 },
            .nvfp4_gu_prompt, .nvfp4_down_prompt, .mxfp8_gu_prompt, .mxfp8_down_prompt => .{ 16, 1, 1 },
        };
    }
};

pub const default_prefix = "tf_ink2_";

/// A device address, as the launcher takes it.
pub const Ptr = ?*anyopaque;

pub const Kernels = struct {
    k: [std.enums.values(Op).len]?xpu.Kernel = @splat(null),

    /// The kernels of `module` named `<prefix><op>`; `only` loads one (a control image holds one a prefix).
    pub fn load(module: *xpu.Module, prefix: []const u8, only: ?Op) !Kernels {
        var out: Kernels = .{};
        errdefer out.deinit();
        for (std.enums.values(Op)) |op| {
            if (only) |o| if (o != op) continue;
            var buf: [96]u8 = undefined;
            const sym = try std.fmt.bufPrintSentinel(&buf, "{s}{s}", .{ prefix, op.name() }, 0);
            out.k[@intFromEnum(op)] = try module.kernel(sym, op.threads());
        }
        return out;
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.k) |*slot| if (slot.*) |*k| {
            k.deinit();
            slot.* = null;
        };
    }

    fn get(self: *const Kernels, op: Op) !xpu.Kernel {
        return self.k[@intFromEnum(op)] orelse error.KernelNotLoaded;
    }

    fn tiles(c: Call, is_gu: bool) u32 {
        const total = if (is_gu) 2 * c.n else c.n;
        return (total + 15) / 16;
    }

    /// Packed NVFP4 GU: X [rows,k] BF16, ids [rows,topk] i32, Y [rows*topk,n] BF16 (the activation).
    pub fn nvfp4Gu(self: *const Kernels, s: xpu.Stream, x: Ptr, w: Ptr, scales: Ptr, amax: Ptr, den: Ptr, ids: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .nvfp4) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.nvfp4_gu), s, .{ tiles(c, true), routes(c), 1 }, .{ x, w, scales, amax, den, ids, out, c.k, c.n, c.topk, c.experts });
    }

    /// Packed MXFP8 GU with UE8M0 scales per row and block of 32.
    pub fn mxfp8Gu(self: *const Kernels, s: xpu.Stream, x: Ptr, w: Ptr, scales: Ptr, ids: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .mxfp8) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.mxfp8_gu), s, .{ tiles(c, true), routes(c), 1 }, .{ x, w, scales, ids, out, c.k, c.n, c.topk, c.experts });
    }

    /// NVFP4 down: act [rows*topk,k] BF16, Y [rows*topk,n] BF16.
    pub fn nvfp4Down(self: *const Kernels, s: xpu.Stream, act: Ptr, w: Ptr, scales: Ptr, amax: Ptr, den: Ptr, ids: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .nvfp4) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.nvfp4_down), s, .{ tiles(c, false), routes(c), 1 }, .{ act, w, scales, amax, den, ids, out, c.k, c.n, c.topk, c.experts });
    }

    pub fn mxfp8Down(self: *const Kernels, s: xpu.Stream, act: Ptr, w: Ptr, scales: Ptr, ids: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .mxfp8) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.mxfp8_down), s, .{ tiles(c, false), routes(c), 1 }, .{ act, w, scales, ids, out, c.k, c.n, c.topk, c.experts });
    }

    /// Decodes `rows` rows of one NVFP4 matrix (the factor of expert 0 of `amax`) to BF16; never an activation quantizer.
    pub fn nvfp4Quantize(self: *const Kernels, s: xpu.Stream, w: Ptr, scales: Ptr, amax: Ptr, den: Ptr, out: Ptr, rows: u32, k: u32) !void {
        if (k == 0 or k % 16 != 0 or rows == 0) return error.BadExpertBlock;
        const items = @as(u64, rows) * k;
        try xpu.launch.dispatch(try self.get(.nvfp4_quantize), s, .{ @intCast((items + 255) / 256), 1, 1 }, .{ w, scales, amax, den, out, rows, k });
    }

    pub fn mxfp8Quantize(self: *const Kernels, s: xpu.Stream, w: Ptr, scales: Ptr, out: Ptr, rows: u32, k: u32) !void {
        if (k == 0 or k % 32 != 0 or rows == 0) return error.BadExpertBlock;
        const items = @as(u64, rows) * k;
        try xpu.launch.dispatch(try self.get(.mxfp8_quantize), s, .{ @intCast((items + 255) / 256), 1, 1 }, .{ w, scales, out, rows, k });
    }

    /// Sums each token's route outputs in ascending expert order (slot order on ties) in FP32, rounds once to BF16.
    pub fn combine(self: *const Kernels, s: xpu.Stream, down: Ptr, ids: Ptr, weights: Ptr, out: Ptr, c: Call) !void {
        try checkRoutes(c);
        try xpu.launch.dispatch(try self.get(.combine), s, .{ (c.n + 63) / 64, c.rows, 1 }, .{ down, ids, weights, out, c.n, c.topk, c.experts });
    }

    /// Groups the routes by expert: `members` [rows*topk] i32, `items` [maxItems,3] i32 (expert, first member, count),
    /// `count` one i32. Entries of `items` past `count` stay as they were.
    pub fn plan(self: *const Kernels, s: xpu.Stream, ids: Ptr, members: Ptr, items: Ptr, count: Ptr, c: Call) !void {
        try checkRoutes(c);
        try xpu.launch.dispatch(try self.get(.plan), s, .{ 1, 1, 1 }, .{ ids, members, items, count, routes(c), c.experts, run_routes });
    }

    pub fn nvfp4GuPrompt(self: *const Kernels, s: xpu.Stream, x: Ptr, w: Ptr, scales: Ptr, amax: Ptr, den: Ptr, items: Ptr, count: Ptr, members: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .nvfp4) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.nvfp4_gu_prompt), s, .{ tiles(c, true), maxItems(c), 1 }, .{ x, w, scales, amax, den, items, count, members, out, c.k, c.n, c.topk });
    }

    pub fn nvfp4DownPrompt(self: *const Kernels, s: xpu.Stream, act: Ptr, w: Ptr, scales: Ptr, amax: Ptr, den: Ptr, items: Ptr, count: Ptr, members: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .nvfp4) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.nvfp4_down_prompt), s, .{ tiles(c, false), maxItems(c), 1 }, .{ act, w, scales, amax, den, items, count, members, out, c.k, c.n, c.topk });
    }

    pub fn mxfp8GuPrompt(self: *const Kernels, s: xpu.Stream, x: Ptr, w: Ptr, scales: Ptr, items: Ptr, count: Ptr, members: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .mxfp8) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.mxfp8_gu_prompt), s, .{ tiles(c, true), maxItems(c), 1 }, .{ x, w, scales, items, count, members, out, c.k, c.n, c.topk });
    }

    pub fn mxfp8DownPrompt(self: *const Kernels, s: xpu.Stream, act: Ptr, w: Ptr, scales: Ptr, items: Ptr, count: Ptr, members: Ptr, out: Ptr, c: Call) !void {
        if (c.format != .mxfp8) return error.WrongExpertFormat;
        try check(c);
        try xpu.launch.dispatch(try self.get(.mxfp8_down_prompt), s, .{ tiles(c, false), maxItems(c), 1 }, .{ act, w, scales, items, count, members, out, c.k, c.n, c.topk });
    }

    /// GU for either format; the NVFP4 globals are required for NVFP4.
    pub fn gu(self: *const Kernels, s: xpu.Stream, x: Ptr, w: Ptr, scales: Ptr, nv_globals: ?struct { amax: Ptr, den: Ptr }, ids: Ptr, out: Ptr, c: Call) !void {
        switch (c.format) {
            .nvfp4 => {
                const g = nv_globals orelse return error.MissingNvfp4Globals;
                return self.nvfp4Gu(s, x, w, scales, g.amax, g.den, ids, out, c);
            },
            .mxfp8 => return self.mxfp8Gu(s, x, w, scales, ids, out, c),
        }
    }
};

/// The grouping the plan kernel makes, on the host: members by expert in route order, items of at most
/// `run_routes` members in expert order. Returns the item count; `items` holds 3 values an item.
pub fn planHost(ids: []const i32, experts: u32, members: []i32, items: []i32) u32 {
    var counts: [max_experts]u32 = @splat(0);
    for (ids) |id| {
        if (id >= 0 and @as(u32, @intCast(id)) < experts) counts[@intCast(id)] += 1;
    }
    var at: u32 = 0;
    var made: u32 = 0;
    for (0..experts) |e| {
        var first: u32 = 0;
        while (first < counts[e]) : (first += run_routes) {
            items[3 * made] = @intCast(e);
            items[3 * made + 1] = @intCast(at + first);
            items[3 * made + 2] = @intCast(@min(run_routes, counts[e] - first));
            made += 1;
        }
        var put = at;
        for (ids, 0..) |id, p| {
            if (id >= 0 and @as(usize, @intCast(id)) == e) {
                members[put] = @intCast(p);
                put += 1;
            }
        }
        at += counts[e];
    }
    return made;
}

test "the host plan groups routes by expert in route order" {
    const ids = [_]i32{ 5, 2, 5, 7, 2, 5 };
    var members: [6]i32 = undefined;
    var items: [3 * 8]i32 = undefined;
    const n = planHost(&ids, 8, &members, &items);
    try std.testing.expectEqual(@as(u32, 3), n);
    try std.testing.expectEqualSlices(i32, &.{ 1, 4, 0, 2, 5, 3 }, &members);
    try std.testing.expectEqualSlices(i32, &.{ 2, 0, 2, 5, 2, 3, 7, 5, 1 }, items[0..9]);
}

test "calls are checked" {
    try check(.{ .rows = 1, .k = hidden, .n = intermediate, .format = .nvfp4 });
    try std.testing.expectError(error.BadExpertBlock, check(.{ .rows = 1, .k = 40, .n = 8, .format = .mxfp8 }));
    try std.testing.expectError(error.BadExpertShape, check(.{ .rows = 0, .k = 32, .n = 8, .format = .nvfp4 }));
    try std.testing.expectEqual(@as(u32, 144), maxItems(.{ .rows = 64, .k = 32, .n = 8, .format = .nvfp4 }));
}

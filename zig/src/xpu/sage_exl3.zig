//! Sage-A1.5 on EXL3 weights (trellis, mul1 codebook): the mixer projections, the fused routed MLP (every pick and the
//! shared expert) and lm_head run through xpu/exl3.zig on the runner's list. The norms, router, GDN, attention and
//! combine stay the runner's (sage.zig). Activations are fp16 in and out, as the runner keeps them.
const std = @import("std");
const ze = @import("ze");
const xpu = @import("xpu");
const exl3 = xpu.exl3;
const dev = @import("dev.zig");
const Dev = dev.Dev;
const Tensor = dev.Tensor;
const alloc = std.heap.c_allocator;

/// Rows of one forward (sage.WINDOW): a plan a row count, built once.
pub const WINDOW = 16;
/// The shared expert's index in the expert tables: after the routed experts.
fn sharedIndex(experts: u32) u32 {
    return experts;
}

/// Checkpoint names without the language model prefix ("layers.0.mlp.gate.weight", "lm_head.trellis"), hashed:
/// Checkpoint.get scans every tensor, and an EXL3 checkpoint has about 128k of them.
pub const Index = struct {
    map: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn init(c: *const dev.Checkpoint) !Index {
        var x: Index = .{};
        try x.map.ensureTotalCapacity(alloc, @intCast(c.tensors.items.len));
        for (c.tensors.items) |t| x.map.putAssumeCapacity(strip(t.name), t);
        return x;
    }

    pub fn deinit(x: *Index) void {
        x.map.deinit(alloc);
    }

    fn strip(n: []const u8) []const u8 {
        for ([_][]const u8{ "model.language_model.", "language_model.model.", "language_model.", "model." }) |p|
            if (std.mem.startsWith(u8, n, p)) return n[p.len..];
        return n;
    }

    pub fn get(x: *const Index, comptime f: []const u8, args: anytype) !Tensor {
        var buf: [256]u8 = undefined;
        const n = try std.fmt.bufPrint(&buf, f, args);
        return x.map.get(n) orelse {
            std.debug.print("missing tensor: {s}\n", .{n});
            return error.MissingTensor;
        };
    }
};

/// One layer's weights as a host image, uploaded in one copy; each piece starts on a 256-byte boundary.
const Pack = struct {
    host: std.ArrayList(u8) = .empty,

    fn add(p: *Pack, bytes: []const u8) !usize {
        const at = std.mem.alignForward(usize, p.host.items.len, 256);
        try p.host.appendNTimes(alloc, 0, at - p.host.items.len);
        try p.host.appendSlice(alloc, bytes);
        return at;
    }
};

/// A matrix's place in its layer's pack.
const Mat = struct {
    k: u32,
    n: u32,
    halves: u32,
    at: [3]usize,

    fn on(m: Mat, base: u64) exl3.Matrix {
        return .{ .trellis = base + m.at[0], .suh = base + m.at[1], .svh = base + m.at[2], .k = m.k, .n = m.n, .halves = m.halves };
    }
};

fn addMatrix(p: *Pack, x: *const Index, comptime f: []const u8, args: anytype) !Mat {
    const tr = try x.get(f ++ ".trellis", args);
    const suh = try x.get(f ++ ".suh", args);
    const svh = try x.get(f ++ ".svh", args);
    const mul1 = try x.get(f ++ ".mul1", args);
    if (!std.mem.eql(u8, tr.dtype, "I16") or tr.rank != 3) return error.UnexpectedShape;
    // the mul1 flag of the exllamav3 export: other codebooks have no kernel here
    if (mul1.bytes.len != 4 or std.mem.readInt(i32, mul1.bytes[0..4], .little) == 0) return error.UnsupportedExl3Codebook;
    const m: Mat = .{ .k = @intCast(tr.shape[0] * 16), .n = @intCast(tr.shape[1] * 16), .halves = @intCast(tr.shape[2] / 8), .at = undefined };
    if (suh.bytes.len != @as(usize, m.k) * 2 or svh.bytes.len != @as(usize, m.n) * 2) return error.UnexpectedShape;
    var out = m;
    out.at = .{ try p.add(tr.bytes), try p.add(suh.bytes), try p.add(svh.bytes) };
    return out;
}

const Layer = struct {
    base: u64,
    attn: bool,
    mixer_in: [3]Mat, // GDN: qkv, z ; attention: q (q | gate per head), k, v
    mixer_out: Mat,
    ab: u64 = 0, // GDN: fp16 [2 value heads, D], in_proj_a then in_proj_b
    gate: exl3.Plan,
    up: exl3.Plan,
    down: exl3.Plan,
    in_plans: [WINDOW]exl3.Plan = undefined,
    out_plans: [WINDOW]exl3.Plan = undefined,
};

/// The runner's buffers the plans read and write (sage.Acts), and the sizes their strides need.
pub const Bind = struct {
    h: *anyopaque,
    p: *anyopaque,
    r: *anyopaque,
    logits: *anyopaque,
    mids: *anyopaque,
    my: *anyopaque,
    hidden: u32,
    vocab: u32,
    gdn_in: u32,
    att_in: u32,
    top_k: u32,
    experts: u32,
    /// GDN: the first column of a | b in p (Q27_GA) and their rows (2 value heads).
    ab_col: u32,
    ab_rows: u32,
};

pub const Weights = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    p: exl3.Pipelines,
    layers: []Layer,
    head_base: u64,
    lm_head: Mat,
    head_plans: [WINDOW]exl3.Plan = undefined,
    scratch: [3]u64 = .{ 0, 0, 0 },
    moe_scratch: u64 = 0,
    b: Bind = undefined,
    list: ze.H = null,

    fn stream(w: *const Weights) xpu.Stream {
        return .{ .ctx = &w.ctx, .list = w.list };
    }

    /// Uploads every EXL3 tensor of the language model (one copy a layer) and the expert tables.
    pub fn load(d: *Dev, c: *const dev.Checkpoint, layers: u32, experts: u32) !*Weights {
        const w = try alloc.create(Weights);
        errdefer alloc.destroy(w);
        w.* = .{ .drv = try xpu.Driver.open(), .ctx = undefined, .p = undefined, .layers = undefined, .head_base = 0, .lm_head = undefined, .list = d.list };
        w.ctx = .{ .d = &w.drv, .gpu = d.g };
        w.p = try exl3.Pipelines.load(&w.ctx);
        var x = try Index.init(c);
        defer x.deinit();
        var pk: Pack = .{};
        defer pk.host.deinit(alloc);
        const s = w.stream();

        w.lm_head = try addMatrix(&pk, &x, "lm_head", .{});
        w.head_base = @intFromPtr(try d.upload(pk.host.items));

        w.layers = try alloc.alloc(Layer, layers);
        const ms = try alloc.alloc([3]exl3.Matrix, experts + 1);
        defer alloc.free(ms);
        const mats = try alloc.alloc([3]Mat, experts + 1);
        defer alloc.free(mats);
        for (w.layers, 0..) |*l, i| {
            pk.host.clearRetainingCapacity();
            const attn = (i + 1) % 4 == 0;
            var mi: [3]Mat = undefined;
            var mo: Mat = undefined;
            var ab_at: usize = 0;
            if (attn) {
                for ([_][]const u8{ "q", "k", "v" }, &mi) |n, *m| m.* = try addMatrix(&pk, &x, "layers.{d}.self_attn.{s}_proj", .{ i, n });
                mo = try addMatrix(&pk, &x, "layers.{d}.self_attn.o_proj", .{i});
            } else {
                mi[0] = try addMatrix(&pk, &x, "layers.{d}.linear_attn.in_proj_qkv", .{i});
                mi[1] = try addMatrix(&pk, &x, "layers.{d}.linear_attn.in_proj_z", .{i});
                mo = try addMatrix(&pk, &x, "layers.{d}.linear_attn.out_proj", .{i});
                const a = try x.get("layers.{d}.linear_attn.in_proj_a.weight", .{i});
                const b = try x.get("layers.{d}.linear_attn.in_proj_b.weight", .{i});
                if (!std.mem.eql(u8, a.dtype, "F16") or !std.mem.eql(u8, b.dtype, "F16") or a.bytes.len != b.bytes.len) return error.UnexpectedDtype;
                ab_at = try pk.add(a.bytes);
                _ = try pk.add(b.bytes); // a's rows are a multiple of 256 bytes: b follows a directly
                if (a.bytes.len % 256 != 0) return error.UnexpectedShape;
            }
            for (mats, 0..) |*e, j| {
                const shared = j == experts;
                for ([_][]const u8{ "gate", "up", "down" }, e) |n, *m| m.* = if (shared)
                    try addMatrix(&pk, &x, "layers.{d}.mlp.shared_expert.{s}_proj", .{ i, n })
                else
                    try addMatrix(&pk, &x, "layers.{d}.mlp.experts.{d}.{s}_proj", .{ i, j, n });
            }
            const base: u64 = @intFromPtr(try d.upload(pk.host.items));
            for (mats, ms) |e, *m| for (e, m) |a, *b| {
                b.* = a.on(base);
            };
            l.* = .{ .base = base, .attn = attn, .mixer_in = mi, .mixer_out = mo, .ab = if (attn) 0 else base + ab_at, .gate = undefined, .up = undefined, .down = undefined };
            l.gate = try tablePlan(w, s, ms, 0);
            l.up = try tablePlan(w, s, ms, 1);
            l.down = try tablePlan(w, s, ms, 2);
            if (i % 8 == 7) std.debug.print("  EXL3 layers loaded: {d}/{d}\n", .{ i + 1, layers });
        }
        try d.sync();
        return w;
    }

    fn tablePlan(w: *const Weights, s: xpu.Stream, ms: []const [3]exl3.Matrix, j: usize) !exl3.Plan {
        const one = try alloc.alloc(exl3.Matrix, ms.len);
        defer alloc.free(one);
        for (one, ms) |*o, m| o.* = m[j];
        return exl3.Plan.experts(&w.ctx, s, alloc, one);
    }

    /// The plans for 1 to WINDOW rows over the runner's buffers, and the scratch they share (launches run in order).
    pub fn bind(w: *Weights, d: *Dev, b: Bind) !void {
        w.b = b;
        const s = w.stream();
        var need: [3]usize = .{ 0, 0, 0 };
        for (w.layers) |l| {
            for (l.mixer_in, 0..) |m, j| if (l.attn or j < 2) {
                need[j] = @max(need[j], exl3.scratchBytes(m.k, m.n, WINDOW));
            };
            need[0] = @max(need[0], exl3.scratchBytes(l.mixer_out.k, l.mixer_out.n, WINDOW));
        }
        need[0] = @max(need[0], exl3.scratchBytes(w.lm_head.k, w.lm_head.n, 1));
        for (&w.scratch, need) |*sc, n| sc.* = @intFromPtr(try d.alloc_(n));
        w.moe_scratch = @intFromPtr(try d.alloc_(exl3.moeScratchBytes(b.hidden, w.layers[0].gate.n_max, WINDOW * (b.top_k + 1))));
        const h: u64 = @intFromPtr(b.h);
        const p: u64 = @intFromPtr(b.p);
        for (w.layers) |*l| {
            const stride = if (l.attn) b.att_in else b.gdn_in;
            for (0..WINDOW) |t| {
                const rows: u32 = @intCast(t + 1);
                var gs: [3]exl3.Group = undefined;
                var y = p;
                const count: usize = if (l.attn) 3 else 2;
                for (l.mixer_in[0..count], gs[0..count], 0..) |m, *g, j| {
                    g.* = .{ .m = m.on(l.base), .rows = rows, .x = h, .x_stride = b.hidden, .y = y, .y_stride = stride, .output = .f16, .input = .f16, .scratch = w.scratch[j] };
                    y += @as(u64, m.n) * 2;
                }
                l.in_plans[t] = try exl3.Plan.init(&w.ctx, s, alloc, gs[0..count]);
                const mo = l.mixer_out;
                l.out_plans[t] = try exl3.Plan.init(&w.ctx, s, alloc, &.{.{ .m = mo.on(l.base), .rows = rows, .x = h, .x_stride = mo.k, .y = @intFromPtr(b.r), .y_stride = b.hidden, .output = .f16, .input = .f16, .scratch = w.scratch[0] }});
            }
        }
        // lm_head reads the window's last row only
        for (&w.head_plans, 0..) |*hp, t| hp.* = try exl3.Plan.init(&w.ctx, s, alloc, &.{.{ .m = w.lm_head.on(w.head_base), .rows = 1, .x = h + t * b.hidden * 2, .x_stride = b.hidden, .y = @intFromPtr(b.logits), .y_stride = b.vocab, .output = .f16, .input = .f16, .scratch = w.scratch[0] }});
    }

    /// h (the input norm) to the mixer's input projections in p: the EXL3 groups, then GDN's dense a and b gates.
    pub fn mixerIn(w: *const Weights, d: *Dev, i: usize, T: u32) !void {
        const l = &w.layers[i];
        try exl3.encode(&w.p, w.stream(), &l.in_plans[T - 1]);
        if (!l.attn) {
            const ab: *anyopaque = @ptrFromInt(l.ab);
            try d.launch(d.k.dense_rows, .{ .x = (T + 15) / 16, .y = w.b.ab_rows / 16 }, .{ w.b.h, ab, w.b.p, @as(i32, @intCast(T)), @as(i32, @intCast(w.b.gdn_in)), @as(i32, @intCast(w.b.ab_col)) });
        }
    }

    /// h (the mixer's gated output) to r.
    pub fn mixerOut(w: *const Weights, i: usize, T: u32) !void {
        try exl3.encode(&w.p, w.stream(), &w.layers[i].out_plans[T - 1]);
    }

    /// Every pick and the shared expert of T rows: h to my (fp32, the runner's combine order).
    pub fn moe(w: *const Weights, i: usize, T: u32) !void {
        const l = &w.layers[i];
        try exl3.encodeMoe(&w.p, w.stream(), &l.gate, &l.up, &l.down, .{
            .ids = @intFromPtr(w.b.mids),
            .tokens = T,
            .topk = w.b.top_k,
            .shared = sharedIndex(w.b.experts),
            .x = @intFromPtr(w.b.h),
            .x_stride = w.b.hidden,
            .y = @intFromPtr(w.b.my),
            .y_stride = w.b.hidden,
            .output = .f32,
            .input = .f16,
            .scratch = w.moe_scratch,
        });
    }

    /// The final norm's last row (of T) to fp16 logits.
    pub fn head(w: *const Weights, T: u32) !void {
        try exl3.encode(&w.p, w.stream(), &w.head_plans[T - 1]);
    }
};

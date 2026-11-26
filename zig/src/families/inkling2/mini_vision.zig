//! Inkling 2 Mini's vision tower (inkling2_vision) on CUDA: 23-layer PE-Lang ViT with 2D pairwise axial RoPE,
//! LayerScale, bilinear pos_embed interpolation, 2x2 spatial merge + 2-layer GELU projector, and per-slot splicing.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const ops = @import("ops.zig");
const mini_image = @import("mini_image.zig");

const Allocator = std.mem.Allocator;
const DeviceBuffer = cuda.DeviceBuffer;
const vit = cuda.vit;

pub const Limits = struct { image_tokens: u32, max_images: u32 };

/// Default per-request image token budget when --vision-image-tokens is not given.
pub const default_image_tokens: u32 = 4096;

/// Inkling 2's added-token markers from tokenizer.json: <|content_image|>, <|unused_200054|>, <|end_message|>.
pub const Marks = struct { open: u32, image: u32, close: u32 };
pub const default_marks: Marks = .{ .open = 200005, .image = 200054, .close = 200010 };

const LayerWeights = struct {
    ln1_w: u64,
    ln1_b: u64,
    ln2_w: u64,
    ln2_b: u64,
    ls1: u64,
    ls2: u64,
    q_w: u64,
    q_b: u64,
    k_w: u64,
    k_b: u64,
    v_w: u64,
    v_b: u64,
    o_w: u64,
    o_b: u64,
    fc_w: u64,
    fc_b: u64,
    proj_w: u64,
    proj_b: u64,
};

const TowerWeights = struct {
    buf: DeviceBuffer,
    cls_emb: u64,
    cls_pos: u64,
    patch_w: u64,
    pos_emb: u64,
    ln_pre_w: u64,
    ln_pre_b: u64,
    layers: []LayerWeights,
    proj_in: u64,
    proj_out: u64,
    bytes: usize,

    fn deinit(w: *TowerWeights, gpa: Allocator) void {
        w.buf.free();
        gpa.free(w.layers);
        w.* = undefined;
    }
};

fn loadWeights(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: config.Vision) !TowerWeights {
    const pre = "model.visual.";
    var ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &.{pre});
    defer core.checkpoint.cache.close(&ck);

    var total: usize = 0;
    for (ck.files.items) |f| {
        for (f.names.keys()) |name| {
            if (!std.mem.startsWith(u8, name, pre)) continue;
            const ten = try ck.get(name);
            if (ten.dtype != .bf16) return error.UnsupportedVisionDType;
            total += std.mem.alignForward(usize, ten.bytes.len, 256);
        }
    }
    if (total == 0) return error.NoVisionWeights;

    var buf = try DeviceBuffer.alloc(d, total);
    errdefer buf.free();

    var map = std.StringHashMap(u64).init(gpa);
    defer map.deinit();

    var at: usize = 0;
    for (ck.files.items) |f| {
        for (f.names.keys()) |name| {
            if (!std.mem.startsWith(u8, name, pre)) continue;
            const ten = try ck.get(name);
            const ptr = buf.ptr + at;
            try cuda.copy_source.synchronousCopy(d, ptr, ten.bytes, gpa);
            try map.put(name[pre.len..], ptr);
            at += std.mem.alignForward(usize, ten.bytes.len, 256);
        }
    }

    const req = struct {
        fn get(m: *const std.StringHashMap(u64), key: []const u8) !u64 {
            return m.get(key) orelse {
                std.log.err("inkling2 vision: missing tensor model.visual.{s}", .{key});
                return error.MissingVisionTensor;
            };
        }
    }.get;

    const layers = try gpa.alloc(LayerWeights, c.layers);
    errdefer gpa.free(layers);

    var kb: [128]u8 = undefined;
    for (0..c.layers) |i| {
        const getL = struct {
            fn f(m: *const std.StringHashMap(u64), b: *[128]u8, idx: usize, suffix: []const u8) !u64 {
                const k = std.fmt.bufPrint(b, "backbone.layers.{d}.{s}", .{ idx, suffix }) catch unreachable;
                return req(m, k);
            }
        }.f;
        layers[i] = .{
            .ln1_w = try getL(&map, &kb, i, "ln_1.weight"),
            .ln1_b = try getL(&map, &kb, i, "ln_1.bias"),
            .ln2_w = try getL(&map, &kb, i, "ln_2.weight"),
            .ln2_b = try getL(&map, &kb, i, "ln_2.bias"),
            .ls1 = try getL(&map, &kb, i, "ls_1.gamma"),
            .ls2 = try getL(&map, &kb, i, "ls_2.gamma"),
            .q_w = try getL(&map, &kb, i, "self_attn.q_proj.weight"),
            .q_b = try getL(&map, &kb, i, "self_attn.q_proj.bias"),
            .k_w = try getL(&map, &kb, i, "self_attn.k_proj.weight"),
            .k_b = try getL(&map, &kb, i, "self_attn.k_proj.bias"),
            .v_w = try getL(&map, &kb, i, "self_attn.v_proj.weight"),
            .v_b = try getL(&map, &kb, i, "self_attn.v_proj.bias"),
            .o_w = try getL(&map, &kb, i, "self_attn.out_proj.weight"),
            .o_b = try getL(&map, &kb, i, "self_attn.out_proj.bias"),
            .fc_w = try getL(&map, &kb, i, "mlp.c_fc.weight"),
            .fc_b = try getL(&map, &kb, i, "mlp.c_fc.bias"),
            .proj_w = try getL(&map, &kb, i, "mlp.c_proj.weight"),
            .proj_b = try getL(&map, &kb, i, "mlp.c_proj.bias"),
        };
    }

    return .{
        .buf = buf,
        .cls_emb = try req(&map, "backbone.embeddings.class_embedding"),
        .cls_pos = try req(&map, "backbone.embeddings.cls_position_embedding"),
        .patch_w = try req(&map, "backbone.embeddings.patch_embedding.weight"),
        .pos_emb = try req(&map, "backbone.embeddings.position_embedding"),
        .ln_pre_w = try req(&map, "backbone.ln_pre.weight"),
        .ln_pre_b = try req(&map, "backbone.ln_pre.bias"),
        .layers = layers,
        .proj_in = try req(&map, "proj_in.weight"),
        .proj_out = try req(&map, "proj_out.weight"),
        .bytes = total,
    };
}

/// Reusable scratch for encoding one image of up to `cap_patches` native patches.
const Scratch = struct {
    buf: ?DeviceBuffer = null,
    cap_patches: usize = 0,
    px_f32: u64 = 0, // f32 [P, 1200]
    px_bf16: u64 = 0, // bf16 [P, 1200]
    x: u64 = 0, // bf16 [S, H]
    xn: u64 = 0, // bf16 [S, H]
    y: u64 = 0, // bf16 [S, H]
    q: u64 = 0, // bf16 [S, H]
    k: u64 = 0, // bf16 [S, H]
    v: u64 = 0, // bf16 [S, H]
    qkv: u64 = 0, // bf16 [S, 3*H]
    att: u64 = 0, // bf16 [S, H]
    mlp: u64 = 0, // bf16 [S, max(FF, proj_hidden)]
    acc: u64 = 0, // f32 [S, max(FF, proj_hidden)] (fallback if cuBLASLt cannot fuse bias)

    fn deinit(sc: *Scratch) void {
        if (sc.buf) |*b| b.free();
        sc.* = .{};
    }

    fn ensure(sc: *Scratch, d: *const cuda.Driver, c: config.Vision, patches: usize) !void {
        if (patches <= sc.cap_patches and sc.buf != null) return;
        const cap = @max(patches, 256);
        const s_max = cap + 1;
        const h: usize = c.hidden;
        const ff: usize = @max(@as(usize, c.hidden) * c.mlp_ratio, @as(usize, c.proj_hidden));
        const pw: usize = 3 * @as(usize, c.patch) * c.patch;

        var total: usize = 0;
        const add = struct {
            fn f(tot: *usize, bytes: usize) u64 {
                const off = tot.*;
                tot.* += std.mem.alignForward(usize, @max(bytes, 256), 256);
                return off;
            }
        }.f;

        const o_px_f32 = add(&total, cap * pw * 4);
        const o_px_bf16 = add(&total, cap * pw * 2);
        const o_x = add(&total, s_max * h * 2);
        const o_xn = add(&total, s_max * h * 2);
        const o_y = add(&total, s_max * h * 2);
        const o_q = add(&total, s_max * h * 2);
        const o_k = add(&total, s_max * h * 2);
        const o_v = add(&total, s_max * h * 2);
        const o_qkv = add(&total, s_max * 3 * h * 2);
        const o_att = add(&total, s_max * h * 2);
        const o_mlp = add(&total, s_max * ff * 2);
        const o_acc = add(&total, s_max * ff * 4);

        const next = try DeviceBuffer.alloc(d, total);
        if (sc.buf) |*old| old.free();
        sc.buf = next;
        sc.cap_patches = cap;
        const base = next.ptr;
        sc.px_f32 = base + o_px_f32;
        sc.px_bf16 = base + o_px_bf16;
        sc.x = base + o_x;
        sc.xn = base + o_xn;
        sc.y = base + o_y;
        sc.q = base + o_q;
        sc.k = base + o_k;
        sc.v = base + o_v;
        sc.qkv = base + o_qkv;
        sc.att = base + o_att;
        sc.mlp = base + o_mlp;
        sc.acc = base + o_acc;
    }
};

/// Prompt rows [at, at + n) take vision rows [row, row + n).
pub const Span = struct { at: u32, n: u32, row: u32 };

const SlotState = struct {
    rows: ?DeviceBuffer = null, // bf16 [limits.image_tokens, text_hidden], allocated on first image use
    spans: std.ArrayList(Span) = .empty,

    fn deinit(st: *SlotState, gpa: Allocator) void {
        if (st.rows) |*r| r.free();
        st.spans.deinit(gpa);
        st.* = .{};
    }
};

pub const Images = struct {
    /// The chat template writes <|content_image|><|unused_200054|><|end_message|> itself.
    pub const template_wraps = true;

    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Vision,
    limits: Limits,
    marks: Marks,
    mark_ids: [3]u32,
    w: TowerWeights,
    vit_k: vit.Kernels,
    att: vit.Attention,
    gemms: vit.Gemms,
    scratch: Scratch = .{},
    slots: []SlotState,
    map_dev: DeviceBuffer, // i32 [max_prompt_rows]
    map_host: []i32,

    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, cfg: config.Config, limits: Limits, max_prompt_rows: usize, n_slots: usize) !*Images {
        if (limits.image_tokens == 0 or limits.max_images == 0) return error.Inkling2VisionLimits;
        const c = cfg.vision;
        if (c.patch != mini_image.patch_size or c.merge != mini_image.merge_size or c.hidden / c.heads != 64) {
            return error.UnsupportedInkling2VisionConfig;
        }
        const self = try gpa.create(Images);
        errdefer gpa.destroy(self);

        var w = try loadWeights(gpa, io, d, dir, c);
        errdefer w.deinit(gpa);
        var vit_k = try vit.Kernels.load(d);
        errdefer vit_k.unload();
        var att = try vit.Attention.load(d);
        errdefer att.unload();
        var gemms = try vit.Gemms.init(gpa, d);
        errdefer gemms.deinit();
        var map_dev = try DeviceBuffer.alloc(d, max_prompt_rows * 4);
        errdefer map_dev.free();
        const map_host = try gpa.alloc(i32, max_prompt_rows);
        errdefer gpa.free(map_host);
        const slots = try gpa.alloc(SlotState, n_slots);
        errdefer gpa.free(slots);
        @memset(slots, .{});

        const marks: Marks = .{ .open = default_marks.open, .image = cfg.image_token, .close = default_marks.close };
        self.* = .{
            .gpa = gpa,
            .io = io,
            .d = d,
            .c = c,
            .limits = limits,
            .marks = marks,
            .mark_ids = .{ marks.open, marks.image, marks.close },
            .w = w,
            .vit_k = vit_k,
            .att = att,
            .gemms = gemms,
            .slots = slots,
            .map_dev = map_dev,
            .map_host = map_host,
        };
        return self;
    }

    pub fn deinit(v: *Images) void {
        for (v.slots) |*sl| sl.deinit(v.gpa);
        v.gpa.free(v.slots);
        v.gpa.free(v.map_host);
        v.map_dev.free();
        v.scratch.deinit();
        v.gemms.deinit();
        v.att.unload();
        v.vit_k.unload();
        v.w.deinit(v.gpa);
        v.gpa.destroy(v);
    }

    /// Server-side image preparation (runs on HTTP worker thread).
    pub fn prepare(v: *Images, a: Allocator, bytes: []const u8, max_tokens: u32) mini_image.Error!mini_image.Prepared {
        return mini_image.prepare(v.io, a, bytes, max_tokens);
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        if (core.picture.decoderProblem(err)) |msg| return msg;
        return switch (err) {
            error.ImageTooSmall => "image token budget is too small for this image",
            error.TooManyImageTokens => "request's images exceed --vision-image-tokens",
            error.BadImageGrid => "image grid dimensions do not match its patch count",
            else => null,
        };
    }

    pub fn clear(v: *Images, slot: usize) void {
        if (slot < v.slots.len) v.slots[slot].spans.clearRetainingCapacity();
    }

    /// Encode all images whose placeholder tokens end past `from` into `slot`'s feature buffer.
    pub fn begin(v: *Images, ops_k: *const ops.Kernels, s: cuda.Stream, slot: usize, images: []const lanes.stream.Image, from: usize) !void {
        const sl = &v.slots[slot];
        sl.spans.clearRetainingCapacity();
        if (images.len == 0) return;

        var total_tokens: u32 = 0;
        for (images) |im| {
            if (@as(usize, im.at) + im.tokens <= from) continue;
            total_tokens = std.math.add(u32, total_tokens, im.tokens) catch return error.TooManyImageTokens;
        }
        if (total_tokens == 0) return;
        if (total_tokens > v.limits.image_tokens) return error.TooManyImageTokens;

        if (sl.rows == null) {
            sl.rows = try DeviceBuffer.alloc(v.d, @as(usize, v.limits.image_tokens) * v.c.text_hidden * 2);
        }
        const rows_base = sl.rows.?.ptr;
        var row_off: u32 = 0;
        for (images) |im| {
            if (@as(usize, im.at) + im.tokens <= from) continue;
            const dst = rows_base + @as(usize, row_off) * v.c.text_hidden * 2;
            try v.encodeOne(ops_k, s, im, dst);
            try sl.spans.append(v.gpa, .{ .at = im.at, .n = im.tokens, .row = row_off });
            row_off += im.tokens;
        }
    }

    /// Run the 23-layer ViT + 2x2 merger on one image and write [im.tokens, text_hidden] bf16 to `dst`.
    fn encodeOne(v: *Images, ops_k: *const ops.Kernels, s: cuda.Stream, im: lanes.stream.Image, dst: u64) !void {
        const c = v.c;
        const gh: usize = im.gh;
        const gw: usize = im.gw;
        const merge: usize = c.merge;
        if (gh == 0 or gw == 0 or gh % merge != 0 or gw % merge != 0) return error.BadImageGrid;
        const patches = gh * gw;
        const tokens = patches / (merge * merge);
        if (tokens != im.tokens) return error.BadImageGrid;
        const pw: usize = 3 * @as(usize, c.patch) * c.patch;
        if (im.pixels.len != patches * pw) return error.BadImageGrid;

        try v.scratch.ensure(v.d, c, patches);
        const sc = &v.scratch;
        const seq = patches + 1;
        const h: usize = c.hidden;
        const ff: usize = @as(usize, c.hidden) * c.mlp_ratio;
        const heads: usize = c.heads;
        const hd: usize = h / heads;
        const eps: f32 = 1e-5;
        const rope_theta: f32 = 10000.0;
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        // 1. Upload f32 patches -> cast to bf16
        try cuda.copy_source.enqueue(v.d, sc.px_f32, std.mem.sliceAsBytes(im.pixels), s);
        try v.vit_k.cast(s, sc.px_f32, sc.px_bf16, patches * pw);

        // 2. Patch linear projection [P, 1200] @ [1024, 1200]^T -> y [P, 1024]
        try v.gemms.run(s, sc.px_bf16, v.w.patch_w, sc.y, patches, h, pw, .bf16);

        // 3. Bilinear pos_embed + prepend CLS -> x [P+1, 1024]
        try ops.visEmbed(ops_k, s, sc.y, v.w.pos_emb, v.w.cls_emb, v.w.cls_pos, sc.x, gh, gw, merge, c.grid, h);

        // 4. Pre-LayerNorm: x -> xn, then copy xn -> x
        try v.vit_k.layerNorm(s, sc.x, v.w.ln_pre_w, v.w.ln_pre_b, sc.xn, seq, h, eps);
        try v.d.check(v.d.api.cuMemcpyDtoDAsync_v2(sc.x, sc.xn, seq * h * 2, s.handle), "cuMemcpyDtoDAsync");

        // 5. 23 ViT encoder layers
        for (v.w.layers) |l| {
            try v.vit_k.layerNorm(s, sc.x, l.ln1_w, l.ln1_b, sc.xn, seq, h, eps);
            try vit.linear(&v.gemms, &v.vit_k, s, sc.acc, sc.xn, l.q_w, l.q_b, sc.q, seq, h, h);
            try vit.linear(&v.gemms, &v.vit_k, s, sc.acc, sc.xn, l.k_w, l.k_b, sc.k, seq, h, h);
            try vit.linear(&v.gemms, &v.vit_k, s, sc.acc, sc.xn, l.v_w, l.v_b, sc.v, seq, h, h);
            try ops.visPackQkvRope(ops_k, s, sc.q, sc.k, sc.v, sc.qkv, gh, gw, merge, heads, hd, rope_theta);
            try v.att.full(s, sc.qkv, sc.att, seq, heads, hd, attn_scale);
            try vit.linear(&v.gemms, &v.vit_k, s, sc.acc, sc.att, l.o_w, l.o_b, sc.y, seq, h, h);
            try ops.visLsAdd(ops_k, s, sc.x, sc.y, l.ls1, seq, h);

            try v.vit_k.layerNorm(s, sc.x, l.ln2_w, l.ln2_b, sc.xn, seq, h, eps);
            try vit.linearAct(&v.gemms, &v.vit_k, s, sc.acc, sc.xn, l.fc_w, l.fc_b, sc.mlp, seq, ff, h, null, sc.mlp, .gelu);
            try vit.linear(&v.gemms, &v.vit_k, s, sc.acc, sc.mlp, l.proj_w, l.proj_b, sc.y, seq, h, ff);
            try ops.visLsAdd(ops_k, s, sc.x, sc.y, l.ls2, seq, h);
        }

        // 6. Drop CLS token (row 0); remaining P rows are already in 2x2 merge-block order [T, 4*H]
        const x_patches = sc.x + h * 2;
        const merge_w = merge * merge * h;
        const proj_h: usize = c.proj_hidden;
        try v.gemms.run(s, x_patches, v.w.proj_in, sc.mlp, tokens, proj_h, merge_w, .bf16);
        try v.vit_k.pointwise(s, sc.mlp, null, sc.mlp, tokens * proj_h, .gelu);
        try v.gemms.run(s, sc.mlp, v.w.proj_out, dst, tokens, c.text_hidden, proj_h, .bf16);
    }

    /// Splice any vision feature rows overlapping prompt chunk [p0, p0 + rows) into `h_dst` [rows, text_hidden].
    pub fn splice(v: *Images, ops_k: *const ops.Kernels, s: cuda.Stream, slot: usize, p0: usize, rows: usize, h_dst: u64) !void {
        const sl = &v.slots[slot];
        if (sl.spans.items.len == 0 or rows == 0) return;
        const p1 = p0 + rows;
        var any = false;
        for (sl.spans.items) |sp| {
            const s0: usize = sp.at;
            const s1: usize = s0 + sp.n;
            if (s0 < p1 and s1 > p0) {
                any = true;
                break;
            }
        }
        if (!any) return;

        @memset(v.map_host[0..rows], -1);
        for (sl.spans.items) |sp| {
            const s0: usize = sp.at;
            const s1: usize = s0 + sp.n;
            const lo = @max(p0, s0);
            const hi = @min(p1, s1);
            if (lo < hi) {
                for (lo..hi) |pos| {
                    v.map_host[pos - p0] = @intCast(sp.row + (pos - s0));
                }
            }
        }
        try cuda.copy_source.enqueue(v.d, v.map_dev.ptr, std.mem.sliceAsBytes(v.map_host[0..rows]), s);
        try ops.visSplice(ops_k, s, h_dst, sl.rows.?.ptr, v.map_dev.ptr, rows, v.c.text_hidden);
    }
};

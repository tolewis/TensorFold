//! Inkling 2 Mini's DSpark drafter (mtp.safetensors) on CUDA: committed target features into three layers' context
//! caches, an eight-row proposal block, and the folded Markov head, greedy on the device. It runs on the target's
//! stream, so the target's tapped rows are read in order with no fence. Math: the Metal Mini drafter (#4337, #4447)
//! and Inkling 2 Small's CUDA DSpark kernels (#4128), at Mini's widths (mtp.cu).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const config = @import("config.zig");
const wts = @import("mini_weights.zig");
const mm = @import("mini_model.zig");
const ops = @import("ops.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Lane = cuda.qmmf.Weight;
const DeviceBuffer = cuda.DeviceBuffer;

/// The target's tapped visits, in the order a features row concatenates them (mtp_config target_layer_ids).
pub const taps = [_]u32{ 5, 23, 41, 48 };
pub const hidden = 2304;
pub const heads = 24;
pub const kv_heads = 8;
pub const head_dim = 128;
const q_width = heads * head_dim;
const kv_width = kv_heads * head_dim;
pub const inter = 9216;
pub const rank = 256;
/// A proposal block's rows (the pending token and seven masks) and the drafts it offers.
pub const block = 8;
pub const depth = block - 1;
pub const window = 1024;
/// A sliding layer's ring: the window, the block, rounded to 128.
pub const local_ring = std.mem.alignForward(usize, window + block, 128);
/// Committed rows one absorb pass takes (a prompt chunk's).
pub const absorb_rows = mm.prompt_rows;
const chunk = 1024;
const row_bytes = taps.len * hidden * 2;
const cap = 30.0;
/// Streams one batched hold encodes and walks together (their weight reads shared); more are taken in groups.
pub const max_batch = 8;
const HoldCost = lanes.hold_price.HoldCost;
/// A hold pass's ms on one GB10 (TF_INKLING2_DRAFT_TRACE=1, 2026-11-23): by streams held together, 1 and 7 drafts deep.
const batched_costs = [_]HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 4.4 }, .{ .streams = 1, .depth = 7, .ms = 6.85 },
    .{ .streams = 2, .depth = 1, .ms = 7.0 }, .{ .streams = 2, .depth = 7, .ms = 10.1 },
    .{ .streams = 4, .depth = 1, .ms = 8.0 }, .{ .streams = 4, .depth = 7, .ms = 11.6 },
};
const serial_costs = [_]HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 4.4 }, .{ .streams = 1, .depth = 7, .ms = 6.85 },
    .{ .streams = 4, .depth = 1, .ms = 17.6 }, .{ .streams = 4, .depth = 7, .ms = 27.4 },
};
/// The same with the FP8 draft head: its read is 1.8 ms shorter a hold pass on the GB10, at every width and depth.
const batched_costs8 = shifted(&batched_costs, false, 1.8);
const serial_costs8 = shifted(&serial_costs, true, 1.8); // serial holds read the head once a stream
fn shifted(comptime t: []const HoldCost, comptime per_stream: bool, comptime saved: f64) [t.len]HoldCost {
    var out: [t.len]HoldCost = undefined;
    for (t, &out) |c, *o| {
        o.* = c;
        o.ms = c.ms - saved * @as(f64, @floatFromInt(if (per_stream) c.streams else 1));
    }
    return out;
}
/// TF_INKLING2_HOLD_COSTS=0: holds priced free, as before.
fn costsOn() bool {
    const v = std.c.getenv("TF_INKLING2_HOLD_COSTS") orelse return true;
    return v[0] != '0';
}
/// TF_INKLING2_DRAFT_FP8=0: draft logits from the BF16 head, as before.
fn fp8HeadOn() bool {
    const v = std.c.getenv("TF_INKLING2_DRAFT_FP8") orelse return true;
    return v[0] != '0';
}
/// TF_INKLING2_DRAFT_LAYERS8=0: the three layers in BF16, as before. On: their qkv, o, gate-up and down as per-tensor
/// FP8 lanes (draft hidden states alone: verification is the target's, so outputs are unchanged; a hold reads half the
/// layer bytes).
fn layers8On() bool {
    const v = std.c.getenv("TF_INKLING2_DRAFT_LAYERS8") orelse return true;
    return v[0] != '0';
}
/// TF_INKLING2_DRAFT_PHI8=0: the walk reads the BF16 phi, as before.
fn phi8On() bool {
    const v = std.c.getenv("TF_INKLING2_DRAFT_PHI8") orelse return true;
    return v[0] != '0';
}
/// TF_INKLING2_DRAFT_KEYED=0: sampled streams get greedy drafts, as before (keyed drafts need the FP8 phi walk).
fn keyedOn() bool {
    const v = std.c.getenv("TF_INKLING2_DRAFT_KEYED") orelse return true;
    return v[0] != '0';
}
fn batchOn() bool {
    const v = std.c.getenv("TF_INKLING2_DRAFT_BATCH") orelse return true;
    return v[0] != '0';
}

/// The global layer's ring: the served window, its block, rounded to 128 (a block past the window never wraps).
pub fn globalRing(context: usize) usize {
    return std.mem.alignForward(usize, context + block, 128);
}

/// A stream's caches: two sliding rings and the global ring (K and V bf16), the newest f row and the walk.
pub fn laneBytes(context: usize) usize {
    return 2 * 2 * local_ring * kv_width * 2 + 2 * globalRing(context) * kv_width * 2 + hidden * 2 + 256;
}

const Layer = struct {
    qkv: Lane, // [q | k | v][H]
    ctx: Lane, // [k_ctx | v_ctx][H]
    o: Lane,
    gu: Lane, // [gate | up][H]
    down: Lane,
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    k_norm_ctx: u64,
    dflare: u64,
};

const Kernels = struct {
    mod: cuda.Module,
    fuse: cuda.Function,
    ctx_kv: cuda.Function,
    init_rows: cuda.Function,
    resnorm: cuda.Function,
    block_qkv: cuda.Function,
    attn: cuda.Function,
    merge: cuda.Function,
    swiglu: cuda.Function,
    kappa: cuda.Function,
    rnn_raw: cuda.Function,
    rnn_gate: cuda.Function,
    vocab: cuda.Function,
    pick_part: cuda.Function,
    pick: cuda.Function,
    fold: cuda.Function,
    vocab_n: cuda.Function,
    amax: cuda.Function,
    fp8_lane: cuda.Function,
    phi8: cuda.Function,
    vocab8_n: cuda.Function,

    fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_mtp);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (.{ "fuse", "ctx_kv", "init_rows", "resnorm", "block_qkv", "attn", "merge", "swiglu", "kappa", "rnn_raw", "rnn_gate", "vocab", "pick_part", "pick", "fold" }) |name| {
            @field(k, name) = try mod.function("tf_ink2m_" ++ name);
        }
        k.vocab_n = try mod.function("tf_ink2m_vocab_n");
        k.amax = try mod.function("tf_ink2m_amax");
        k.fp8_lane = try mod.function("tf_ink2m_fp8_lane");
        k.phi8 = try mod.function("tf_ink2m_phi8");
        k.vocab8_n = try mod.function("tf_ink2m_vocab8_n");
        return k;
    }
};

const AttnArgs = extern struct {
    q: u64,
    kc: u64,
    vc: u64,
    po: u64,
    pml: u64,
    T: i32,
    hq: i32,
    hk: i32,
    ring: i32,
    lo: i32,
    hi: i32,
    c0: i32,
    scale: f32,
};

/// One stream's draft state: its ring slot, the committed range, the newest f row and the last walk.
const Slot = struct {
    k: [3]u64,
    v: [3]u64,
    rings: [3]usize,
    newest: u64,
    walk: u64, // int32 [block]: the pending token, then the drafts
    used: bool = false,
};

const LaneState = struct {
    slot: usize,
    from: u64 = 0, // the first committed row the context holds (a resumed prompt starts past 0)
    end: u64 = 0, // rows [from, end) absorbed
    held: u32 = 0,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    target: *const wts.Weights,
    tk: *const mm.Kernels,
    k: Kernels,
    eps: f32,
    theta: f32,
    context: usize,
    vocab: usize, // the unpadded vocabulary the target's head writes
    layers: [3]Layer = undefined,
    fuse: u64 = 0,
    mask: u64 = 0,
    norm: u64 = 0,
    rnn: u64 = 0,
    rnn_bias: u64 = 0,
    w1: u64 = 0,
    phi: u64 = 0,
    beta: u64 = 0,
    dflare: u64 = 0, // [4][H]: the three layers' then the Markov head's
    head: ?Lane = null, // TF_INKLING2_DRAFT_FP8: an FP8 copy of the target's unembedding for draft logits
    phi8: u64 = 0, // TF_INKLING2_DRAFT_PHI8: e4m3 phi [V][R] for the walk, row scales in phi_scales (0: the bf16 phi)
    phi_scales: u64 = 0,
    keys: u64 = 0, // DraftKey [max_batch]: each held stream's keyed-draw seed, 1 / T and first position (fp8 walk only)
    keyed: bool = false, // this hold has a sampled stream: its drafts race with the target's noise
    // scratch, one hold or one absorb pass at a time on the target's stream
    c: u64 = 0, // bf16 [4][absorb_rows][H]
    kv: u64 = 0, // bf16 [absorb_rows][2 * kv]
    x: u64 = 0,
    resid: u64 = 0,
    next_resid: u64 = 0,
    y: u64 = 0,
    anchor: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    att: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    base: u64 = 0, // bf16 [7][V]
    kappa: u64 = 0, // fp32 [7][3R]
    state: u64 = 0,
    latent: u64 = 0,
    logits: u64 = 0, // fp32 [V]
    raw: u64 = 0, // fp32 [3R]: a recurrence step's sums
    picks: u64 = 0, // float2 [V / 4096]: the pick's first stage
    po: u64 = 0,
    pml: u64 = 0,
    part: u64 = 0,
    chunks: usize = 0,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    weight_bytes: usize = 0,
    slots: std.ArrayList(Slot) = .empty,
    slot_bufs: std.ArrayList(DeviceBuffer) = .empty,
    lanes: std.AutoHashMapUnmanaged(*Stream, LaneState) = .empty,

    fn alloc(x: *Drafter, bytes: usize) !u64 {
        var b = try DeviceBuffer.alloc(x.d, @max(bytes, 256));
        errdefer b.free();
        try x.buffers.append(x.gpa, b);
        return b.ptr;
    }

    fn upload(x: *Drafter, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(x.d, bytes);
        errdefer b.free();
        try x.buffers.append(x.gpa, b);
        x.weight_bytes += bytes.len;
        return b.ptr;
    }

    /// Rows of `parts` stacked as one [sum n][k] weight, packed for the qmmf bf16 lane.
    fn pack(x: *Drafter, parts: []const core.checkpoint.Tensor) !Lane {
        const k = parts[0].dim(1);
        var n: usize = 0;
        for (parts) |p| {
            if (p.dim(1) != k) return error.DSparkTensorShape;
            n += p.dim(0);
        }
        const src = try x.gpa.alloc(u16, n * k);
        defer x.gpa.free(src);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(std.mem.sliceAsBytes(src[at..][0 .. p.dim(0) * k]), p.bytes[0 .. p.dim(0) * k * 2]);
            at += p.dim(0) * k;
        }
        const out = try x.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer x.gpa.free(out);
        @memset(out, 0);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try x.upload(std.mem.sliceAsBytes(out)), n, k);
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, target: *const wts.Weights, tk: *const mm.Kernels, context: usize, dir: []const u8) !*Drafter {
        const c = &target.config;
        try admit(c);
        const x = try gpa.create(Drafter);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .d = d, .s = s, .target = target, .tk = tk, .k = try Kernels.load(d), .eps = c.text.eps, .theta = c.mtp.?.rope_theta, .context = context, .vocab = c.text.unpadded_vocab };
        errdefer x.deinit();
        var ck: core.checkpoint.Checkpoint = .{ .gpa = gpa, .io = io };
        defer ck.close();
        if (std.mem.endsWith(u8, dir, ".safetensors")) {
            try ck.add(std.fs.path.dirname(dir) orelse ".", std.fs.path.basename(dir));
        } else ck.add(dir, "mtp.safetensors") catch |e| {
            std.log.err("Inkling 2 Mini DSpark: cannot open {s}/mtp.safetensors ({s})", .{ dir, @errorName(e) });
            return error.DSparkMissingFile;
        };
        try x.loadWeights(&ck);
        if (layers8On()) for (&x.layers) |*l| {
            l.qkv = try x.fp8Of(l.qkv);
            l.o = try x.fp8Of(l.o);
            l.gu = try x.fp8Of(l.gu);
            l.down = try x.fp8Of(l.down);
        };
        try x.scratch();
        try x.foldHead(&ck);
        if (fp8HeadOn()) try x.fp8Head();
        if (phi8On()) try x.fp8Phi();
        try x.s.synchronize();
        return x;
    }

    /// The target's unembedding as a per-tensor FP8 lane weight (e4m3 at amax / 448), repacked on the GPU. Only draft
    /// logits read it: verification keeps the BF16 head, so outputs are unchanged and the hold reads half the bytes.
    fn fp8Head(x: *Drafter) !void {
        x.head = try x.fp8Of(x.target.unembed);
    }

    /// A bf16 lane weight as a per-tensor FP8 lane weight (e4m3 at amax / 448), repacked on the GPU.
    fn fp8Of(x: *Drafter, u: Lane) !Lane {
        const total: usize = @as(usize, u.npad) * u.k;
        const amax = try x.alloc(4);
        try x.d.check(x.d.api.cuMemsetD8Async(amax, 0, 4, x.s.handle), "cuMemsetD8Async");
        var a: cuda.Args = .{};
        a.add(u.codes);
        a.add(@as(i64, @intCast(total)));
        a.add(amax);
        try go(x.k.amax, x.s, .{ .x = 1024 }, 256, &a);
        try x.s.synchronize();
        var bits: u32 = 0;
        try cuda.memory.download(x.d, amax, std.mem.asBytes(&bits));
        const m: f32 = @bitCast(bits);
        if (!(m > 0) or !std.math.isFinite(m)) return error.DSparkHeadRange;
        const scale = m / 448.0;
        const codes = try x.alloc(total);
        x.weight_bytes += total;
        var b: cuda.Args = .{};
        b.add(u.codes);
        b.add(codes);
        b.add(@as(i32, @intCast(u.k)));
        b.add(@as(f32, 1.0 / scale));
        b.add(@as(i64, @intCast(total)));
        try go(x.k.fp8_lane, x.s, .{ .x = @intCast((total + 255) / 256) }, 256, &b);
        return cuda.modelopt.fp8Weight(codes, scale, u.n, u.k);
    }

    /// phi as e4m3 rows at amax / 448 each: the walk reads half the bytes a step (draft logits alone).
    fn fp8Phi(x: *Drafter) !void {
        const V = x.vocab;
        x.keys = try x.alloc(max_batch * @sizeOf(DraftKey));
        x.phi8 = try x.alloc(V * rank);
        x.phi_scales = try x.alloc(V * 4);
        x.weight_bytes += V * (rank + 4);
        var a: cuda.Args = .{};
        for ([_]u64{ x.phi, x.phi8, x.phi_scales }) |v| a.add(v);
        a.add(@as(i32, @intCast(V)));
        try go(x.k.phi8, x.s, .{ .x = @intCast((V + 7) / 8) }, 256, &a);
    }

    /// One walk step's folded logits for `n` streams: base rows `base_stride` apart, latents [n][R], out rows V apart.
    /// The hold's keys into x.keys: a sampled stream races at its rule's seed and 1 / T from its hold's position.
    fn upKeys(x: *Drafter, items: []const dr.Hold) !void {
        x.keyed = false;
        if (x.phi8 == 0 or !keyedOn()) return;
        var k: [max_batch]DraftKey = @splat(.{ .seed = 0, .inv_t = 0, .pos = 0 });
        for (items, 0..) |it, i| if (cuda.sample.ruleFor(it.sampling)) |r| {
            k[i] = .{ .seed = r.seed, .inv_t = r.inv_t, .pos = @intCast(it.position) };
            x.keyed = true;
        };
        if (x.keyed) try cuda.copy_source.enqueue(x.d, x.keys, std.mem.sliceAsBytes(k[0..items.len]), x.s);
    }

    fn vocabStep(x: *Drafter, base: u64, base_stride: usize, n: usize, step: usize) !void {
        const V = x.vocab;
        var b: cuda.Args = .{};
        b.add(base);
        b.add(@as(i64, @intCast(base_stride)));
        b.add(x.latent);
        if (x.phi8 != 0) {
            b.add(x.phi8);
            b.add(x.phi_scales);
        } else b.add(x.phi);
        b.add(x.beta);
        b.add(@as(f32, cap));
        b.add(@as(i32, @intCast(V)));
        b.add(@as(i32, @intCast(V)));
        b.add(x.logits);
        b.add(@as(i64, @intCast(V)));
        b.add(@as(i32, @intCast(n)));
        if (x.phi8 != 0) {
            b.add(if (x.keyed) x.keys else @as(u64, 0));
            b.add(@as(i32, @intCast(step)));
        }
        try go(if (x.phi8 != 0) x.k.vocab8_n else x.k.vocab_n, x.s, .{ .x = @intCast((V + 7) / 8) }, 256, &b);
    }

    /// The weight draft logits read: the FP8 copy if built, else the target's own head.
    fn draftHead(x: *const Drafter) Lane {
        return x.head orelse x.target.unembed;
    }

    fn get(ck: *core.checkpoint.Checkpoint, name: []const u8, shape: []const usize) !core.checkpoint.Tensor {
        const t = ck.get(name) catch {
            std.log.err("Inkling 2 Mini DSpark: mtp.safetensors has no {s}", .{name});
            return error.DSparkMissingTensor;
        };
        if (!t.is(.bf16, shape)) {
            std.log.err("Inkling 2 Mini DSpark: {s} is not bf16 of the expected shape", .{name});
            return error.DSparkTensorShape;
        }
        return t;
    }

    fn layerTensor(ck: *core.checkpoint.Checkpoint, i: usize, suffix: []const u8, shape: []const usize) !core.checkpoint.Tensor {
        var b: [96]u8 = undefined;
        return get(ck, try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, suffix }), shape);
    }

    fn loadWeights(x: *Drafter, ck: *core.checkpoint.Checkpoint) !void {
        const H = hidden;
        var dflare: [4 * H]u16 = undefined;
        for (&x.layers, 0..) |*l, i| {
            const q = try layerTensor(ck, i, "self_attn.q_proj.weight", &.{ q_width, H });
            const k = try layerTensor(ck, i, "self_attn.k_proj.weight", &.{ kv_width, H });
            const v = try layerTensor(ck, i, "self_attn.v_proj.weight", &.{ kv_width, H });
            const kc = try layerTensor(ck, i, "self_attn.k_proj_ctx.weight", &.{ kv_width, H });
            const vc = try layerTensor(ck, i, "self_attn.v_proj_ctx.weight", &.{ kv_width, H });
            const g = try layerTensor(ck, i, "mlp.gate_proj.weight", &.{ inter, H });
            const u = try layerTensor(ck, i, "mlp.up_proj.weight", &.{ inter, H });
            l.qkv = try x.pack(&.{ q, k, v });
            l.ctx = try x.pack(&.{ kc, vc });
            l.o = try x.pack(&.{try layerTensor(ck, i, "self_attn.o_proj.weight", &.{ H, q_width })});
            l.gu = try x.pack(&.{ g, u });
            l.down = try x.pack(&.{try layerTensor(ck, i, "mlp.down_proj.weight", &.{ H, inter })});
            l.input_norm = try x.upload((try layerTensor(ck, i, "input_layernorm.weight", &.{H})).bytes);
            l.post_norm = try x.upload((try layerTensor(ck, i, "post_attention_layernorm.weight", &.{H})).bytes);
            l.q_norm = try x.upload((try layerTensor(ck, i, "self_attn.q_norm.weight", &.{head_dim})).bytes);
            l.k_norm = try x.upload((try layerTensor(ck, i, "self_attn.k_norm.weight", &.{head_dim})).bytes);
            l.k_norm_ctx = try x.upload((try layerTensor(ck, i, "self_attn.k_norm_ctx.weight", &.{head_dim})).bytes);
            @memcpy(std.mem.sliceAsBytes(dflare[i * H ..][0..H]), (try layerTensor(ck, i, "dflare_norm.weight", &.{H})).bytes);
            ck.release();
        }
        @memcpy(std.mem.sliceAsBytes(dflare[3 * H ..][0..H]), (try get(ck, "markov_head.dflare_norm.weight", &.{H})).bytes);
        x.dflare = try x.upload(std.mem.sliceAsBytes(&dflare));
        x.fuse = try x.upload((try get(ck, "fuse_weight", &.{ 4, 4 })).bytes);
        x.mask = try x.upload((try get(ck, "mask_embed", &.{H})).bytes);
        x.norm = try x.upload((try get(ck, "norm.weight", &.{H})).bytes);
        x.rnn = try x.upload((try get(ck, "markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * H + 2 * rank })).bytes);
        x.rnn_bias = try x.upload((try get(ck, "markov_head.rnn_proj.bias", &.{3 * rank})).bytes);
        x.w1 = try x.upload((try get(ck, "markov_head.markov_w1.weight", &.{ x.target.config.text.vocab, rank })).bytes);
    }

    fn scratch(x: *Drafter) !void {
        const H = hidden;
        const R = absorb_rows;
        const V = x.vocab;
        x.chunks = globalRing(x.context) / chunk + 1;
        x.c = try x.alloc(4 * R * H * 2);
        x.kv = try x.alloc(R * 2 * kv_width * 2);
        x.x = try x.alloc(max_batch * block * H * 2);
        x.resid = try x.alloc(max_batch * block * H * 2);
        x.next_resid = try x.alloc(max_batch * block * H * 2);
        x.y = try x.alloc(max_batch * block * H * 2);
        x.anchor = try x.alloc(max_batch * H * 2);
        x.qkv = try x.alloc(max_batch * block * (q_width + 2 * kv_width) * 2);
        x.q = try x.alloc(max_batch * block * q_width * 2);
        x.att = try x.alloc(max_batch * block * q_width * 2);
        x.gu = try x.alloc(max_batch * block * 2 * inter * 2);
        x.act = try x.alloc(max_batch * block * inter * 2);
        x.base = try x.alloc(max_batch * block * V * 2);
        x.kappa = try x.alloc(max_batch * depth * 3 * rank * 4);
        x.state = try x.alloc(max_batch * rank * 2);
        x.latent = try x.alloc(max_batch * rank * 2);
        x.logits = try x.alloc(max_batch * V * 4);
        x.raw = try x.alloc(max_batch * 3 * rank * 4);
        x.picks = try x.alloc(max_batch * pickParts(V) * 8);
        x.po = try x.alloc(x.chunks * block * heads * head_dim * 4);
        x.pml = try x.alloc(x.chunks * block * heads * 2 * 4);
        var part: usize = 0;
        for (x.layers) |l| {
            for ([_]Lane{ l.qkv, l.o, l.gu, l.down }) |w| for (1..max_batch + 1) |nb| {
                part = @max(part, x.tk.lane.partBytes(nb * block, w));
            };
            for (1..R + 1) |m| part = @max(part, x.tk.lane.partBytes(m, l.ctx));
        }
        part = @max(part, x.tk.lane.partBytes(depth, x.target.unembed));
        for (1..max_batch + 1) |nb| part = @max(part, x.tk.lane.partBytes(nb * block, x.target.unembed));
        x.part = try x.alloc(part);
    }

    /// phi = U . hidden_out and beta = U . bias over the target's head rows: one lane call of the head with
    /// hidden_out's columns (and the bias) as its rows, then split and rounded to bf16 (fp32 sums, not binary64).
    fn foldHead(x: *Drafter, ck: *core.checkpoint.Checkpoint) !void {
        const H = hidden;
        const V = x.vocab;
        const ho = try get(ck, "markov_head.hidden_out.weight", &.{ H, rank });
        const hb = try get(ck, "markov_head.hidden_out.bias", &.{H});
        const rows = try x.gpa.alloc(u16, (rank + 1) * H);
        defer x.gpa.free(rows);
        const src: []align(1) const u16 = std.mem.bytesAsSlice(u16, ho.bytes);
        for (0..H) |h| for (0..rank) |r| {
            rows[r * H + h] = src[h * rank + r];
        };
        @memcpy(std.mem.sliceAsBytes(rows[rank * H ..][0..H]), hb.bytes);
        var xin = try DeviceBuffer.fromHost(x.d, std.mem.sliceAsBytes(rows));
        defer xin.free();
        var y = try DeviceBuffer.alloc(x.d, (rank + 1) * V * 4);
        defer y.free();
        var part = try DeviceBuffer.alloc(x.d, @max(256, x.tk.lane.partBytes(rank + 1, x.target.unembed)));
        defer part.free();
        try x.tk.lane.matmulOut(x.s, xin.ptr, H, rank + 1, x.target.unembed, y.ptr, .f32, part.ptr);
        x.phi = try x.alloc(V * rank * 2);
        x.beta = try x.alloc(V * 2);
        x.weight_bytes += V * (rank + 1) * 2;
        var a: cuda.Args = .{};
        a.add(y.ptr);
        a.add(@as(i32, @intCast(V)));
        a.add(@as(i32, rank));
        a.add(x.phi);
        a.add(x.beta);
        const n = V * (rank + 1);
        try go(x.k.fold, x.s, .{ .x = @intCast((n + 255) / 256) }, 256, &a);
        try x.s.synchronize();
    }

    pub fn deinit(x: *Drafter) void {
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs.items) |*b| b.free();
        x.slot_bufs.deinit(x.gpa);
        x.slots.deinit(x.gpa);
        for (x.buffers.items) |*b| b.free();
        x.buffers.deinit(x.gpa);
        x.k.mod.unload();
        x.gpa.destroy(x);
    }

    /// Device bytes besides the streams': weights, the folded head and the scratch.
    pub fn fixedBytes(x: *const Drafter) usize {
        var n: usize = 0;
        for (x.buffers.items) |b| n += b.len;
        return n;
    }

    /// `streams` slots' caches, allocated now (the host's count); a smaller count than held keeps them.
    pub fn reserve(x: *Drafter, streams: u32) !void {
        while (x.slots.items.len < streams) {
            var sl: Slot = .{ .k = undefined, .v = undefined, .rings = .{ local_ring, globalRing(x.context), local_ring }, .newest = 0, .walk = 0 };
            for (0..3) |i| {
                const bytes = sl.rings[i] * kv_width * 2;
                sl.k[i] = try x.slotAlloc(bytes);
                sl.v[i] = try x.slotAlloc(bytes);
            }
            sl.newest = try x.slotAlloc(hidden * 2);
            sl.walk = try x.slotAlloc(block * 4);
            try x.slots.append(x.gpa, sl);
        }
    }

    fn slotAlloc(x: *Drafter, bytes: usize) !u64 {
        var b = try DeviceBuffer.alloc(x.d, bytes);
        errdefer b.free();
        try x.slot_bufs.append(x.gpa, b);
        return b.ptr;
    }

    pub fn drafter(x: *Drafter) dr.Drafter {
        return .{ .kind = "DSpark", .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn self(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &taps;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const x = self(ptr);
        const fp8 = fp8HeadOn();
        const costs: []const HoldCost = if (!costsOn()) &.{} else if (batchOn()) (if (fp8) &batched_costs8 else &batched_costs) else (if (fp8) &serial_costs8 else &serial_costs);
        return .{ .depth = depth, .step_ms = 0, .fixed_block = true, .plain_guard = true, .batched = batchOn(), .stream_bytes = laneBytes(x.context), .hold_costs = costs };
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.lanes.contains(s)) return error.DraftStreamOpen;
        const slot = for (x.slots.items, 0..) |sl, i| {
            if (!sl.used) break i;
        } else blk: {
            try x.reserve(@intCast(x.slots.items.len + 1)); // past the host's count: one more, if it fits
            break :blk x.slots.items.len - 1;
        };
        try x.lanes.put(x.gpa, s, .{ .slot = slot });
        x.slots.items[slot].used = true;
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.lanes.fetchRemove(s)) |e| x.slots.items[e.value.slot].used = false;
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = self(ptr);
        try x.s.synchronize();
        for (streams, out) |s, toks| {
            const l = x.lanes.getPtr(s) orelse return error.DraftStreamUnknown;
            if (toks.len > l.held) return error.DraftHeldFewer;
            var steps: [block]i32 = undefined;
            try cuda.memory.download(x.d, x.slots.items[l.slot].walk, std.mem.sliceAsBytes(&steps));
            for (toks, steps[1..][0..toks.len]) |*t, w| {
                if (w < 0 or w >= x.vocab) return error.DraftNonfinite;
                t.* = @intCast(w);
            }
        }
    }

    /// Rows [start, start + rows) of `features` (device bf16 [rows][4][H]) into the stream's three context caches.
    fn absorbRows(x: *Drafter, sl: *const Slot, features: u64, start: u64, rows: usize) !void {
        const H = hidden;
        var a: cuda.Args = .{};
        for ([_]u64{ features, x.fuse, x.dflare, x.c, sl.newest }) |v| a.add(v);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, H));
        a.add(x.eps);
        try go(x.k.fuse, x.s, .{ .x = @intCast(rows), .y = 4 }, 256, &a);
        for (x.layers, 0..) |l, i| {
            try x.tk.lane.matmulOut(x.s, x.c + i * rows * H * 2, H, rows, l.ctx, x.kv, .bf16, x.part);
            var b: cuda.Args = .{};
            for ([_]u64{ x.kv, x.kv + kv_width * 2 }) |v| b.add(v);
            b.add(@as(i32, 2 * kv_width));
            for ([_]u64{ l.k_norm_ctx, sl.k[i], sl.v[i] }) |v| b.add(v);
            b.add(@as(i32, kv_width));
            b.add(@as(i32, @intCast(start)));
            b.add(@as(i32, @intCast(sl.rings[i])));
            b.add(x.eps);
            b.add(x.theta);
            try go(x.k.ctx_kv, x.s, .{ .x = @intCast(rows), .y = kv_heads }, head_dim, &b);
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = self(ptr);
        for (items) |item| {
            const l = x.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const f = item.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
            if (item.rows != null) return error.TreesNotBuilt;
            if (f.rows == 0) continue;
            if (item.start + f.rows > x.context) return error.DraftContextGap;
            if (item.start > l.end or l.end == 0) l.from = item.start; // a gap (a resumed prompt): attend from here
            const sl = &x.slots.items[l.slot];
            var done: usize = 0;
            while (done < f.rows) {
                const n = @min(absorb_rows, f.rows - done);
                try x.absorbRows(sl, f.buffer + f.offset + done * row_bytes, item.start + done, n);
                done += n;
            }
            l.end = item.start + f.rows; // rows past a rewound end are rewritten in place
            l.held = 0;
        }
    }

    fn resnorm(x: *Drafter, in: u64, resid: u64, w: u64, new_resid: u64, out: u64, init_: bool) !void {
        return x.resnormRows(in, resid, w, new_resid, out, init_, block);
    }

    fn resnormRows(x: *Drafter, in: u64, resid: u64, w: u64, new_resid: u64, out: u64, init_: bool, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ in, resid, w, new_resid, out }) |v| a.add(v);
        a.add(@as(i32, hidden));
        a.add(@as(i32, @intFromBool(init_)));
        a.add(x.eps);
        try go(x.k.resnorm, x.s, .{ .x = @intCast(rows) }, 256, &a);
    }

    fn mm_(x: *Drafter, in: u64, ld: usize, m: usize, w: Lane, out: u64) !void {
        try x.tk.lane.matmulOut(x.s, in, ld, m, w, out, .bf16, x.part);
    }

    /// The proposal block at `anchor` through the three layers, the head's base rows and the Markov contexts.
    fn encodeBlock(x: *Drafter, sl: *const Slot, l: *const LaneState, anchor: u64) !void {
        const H = hidden;
        try ops.embed(&x.tk.ops, x.s, sl.walk, 1, x.target.embed, x.target.embed_norm, x.anchor, H, x.eps);
        var a: cuda.Args = .{};
        for ([_]u64{ x.anchor, x.mask, x.x }) |v| a.add(v);
        a.add(@as(i32, H));
        try go(x.k.init_rows, x.s, .{ .x = block }, 256, &a);
        try x.resnorm(x.x, x.resid, x.layers[0].input_norm, x.resid, x.x, true);
        const hi: usize = @intCast(anchor + block - 1);
        for (x.layers, 0..) |ly, i| {
            const ld = q_width + 2 * kv_width;
            try x.mm_(x.x, H, block, ly.qkv, x.qkv);
            var b: cuda.Args = .{};
            b.add(x.qkv);
            b.add(@as(i32, ld));
            for ([_]u64{ ly.q_norm, ly.k_norm, x.q, sl.k[i], sl.v[i] }) |v| b.add(v);
            b.add(@as(i32, heads));
            b.add(@as(i32, kv_heads));
            b.add(@as(i32, @intCast(anchor)));
            b.add(@as(i32, @intCast(sl.rings[i])));
            b.add(x.eps);
            b.add(x.theta);
            try go(x.k.block_qkv, x.s, .{ .x = block, .y = heads + kv_heads }, head_dim, &b);
            // keys: the committed context (a sliding layer's last window of it) and the block itself
            const local = i != 1;
            var lo: usize = @intCast(l.from);
            if (local and anchor > window) lo = @max(lo, @as(usize, @intCast(anchor)) - window);
            const c0 = lo / chunk;
            const nc = hi / chunk - c0 + 1;
            const aa: AttnArgs = .{ .q = x.q, .kc = sl.k[i], .vc = sl.v[i], .po = x.po, .pml = x.pml, .T = block, .hq = heads, .hk = kv_heads, .ring = @intCast(sl.rings[i]), .lo = @intCast(lo), .hi = @intCast(hi), .c0 = @intCast(c0), .scale = 1.0 / @sqrt(@as(f32, head_dim)) };
            var c: cuda.Args = .{};
            c.add(aa);
            try go(x.k.attn, x.s, .{ .x = @intCast(nc), .y = kv_heads }, 256, &c);
            var m: cuda.Args = .{};
            for ([_]u64{ x.po, x.pml }) |v| m.add(v);
            m.add(@as(i32, @intCast(nc)));
            m.add(@as(i32, block * heads));
            m.add(x.att);
            try go(x.k.merge, x.s, .{ .x = block * heads }, head_dim, &m);
            try x.mm_(x.att, q_width, block, ly.o, x.y);
            try x.resnorm(x.y, x.resid, ly.post_norm, x.next_resid, x.x, false);
            try x.mm_(x.x, H, block, ly.gu, x.gu);
            var g: cuda.Args = .{};
            for ([_]u64{ x.gu, x.gu + inter * 2 }) |v| g.add(v);
            g.add(@as(i32, 2 * inter));
            g.add(x.act);
            g.add(@as(i32, inter));
            try go(x.k.swiglu, x.s, .{ .x = (inter + 255) / 256, .y = block }, 256, &g);
            try x.mm_(x.act, inter, block, ly.down, x.y);
            const next = if (i + 1 < x.layers.len) x.layers[i + 1].input_norm else x.norm;
            try x.resnorm(x.y, x.next_resid, next, x.resid, x.x, false);
        }
        // rows 1..7: the head's base logits, and each row's Markov context with the stream's newest f
        try x.mm_(x.x + H * 2, H, depth, x.draftHead(), x.base);
        var k: cuda.Args = .{};
        for ([_]u64{ x.x + H * 2, sl.newest, x.rnn, x.rnn_bias, x.kappa }) |v| k.add(v);
        k.add(@as(i32, depth));
        k.add(@as(i32, H));
        k.add(@as(i32, rank));
        try go(x.k.kappa, x.s, .{ .x = 3 * rank / 8 }, 256, &k);
    }

    /// `n` greedy drafts: each step's state and latent, its folded logits, its pick into walk[j + 1].
    fn draftWalk(x: *Drafter, sl: *const Slot, n: usize) !void {
        const V = x.vocab;
        for (0..n) |j| {
            var a: cuda.Args = .{};
            for ([_]u64{ x.kappa, x.rnn, x.w1, sl.walk }) |v| a.add(v);
            a.add(@as(i32, @intCast(j)));
            for ([_]u64{ x.state, x.raw }) |v| a.add(v);
            a.add(@as(i32, rank));
            a.add(@as(i32, hidden));
            try go(x.k.rnn_raw, x.s, .{ .x = 3 * rank / 8 }, 256, &a);
            var g: cuda.Args = .{};
            g.add(x.raw);
            g.add(@as(i32, @intCast(j)));
            for ([_]u64{ x.state, x.latent }) |v| g.add(v);
            g.add(@as(i32, rank));
            try go(x.k.rnn_gate, x.s, .{ .x = 1 }, rank, &g);
            if (x.phi8 != 0) try x.vocabStep(x.base + j * V * 2, 0, 1, j) else {
                var b: cuda.Args = .{};
                for ([_]u64{ x.base + j * V * 2, x.latent, x.phi, x.beta }) |v| b.add(v);
                b.add(@as(f32, cap));
                b.add(@as(i32, @intCast(V)));
                b.add(@as(i32, @intCast(V)));
                b.add(x.logits);
                try go(x.k.vocab, x.s, .{ .x = @intCast((V + 7) / 8) }, 256, &b);
            }
            var c: cuda.Args = .{};
            c.add(x.logits);
            c.add(@as(i32, @intCast(V)));
            c.add(x.picks);
            try go(x.k.pick_part, x.s, .{ .x = @intCast(pickParts(V)) }, 256, &c);
            var e: cuda.Args = .{};
            e.add(x.picks);
            e.add(@as(i32, @intCast(pickParts(V))));
            e.add(sl.walk);
            e.add(@as(i32, @intCast(j)));
            try go(x.k.pick, x.s, .{ .x = 1 }, 256, &e);
        }
    }

    fn holdBatch(x: *Drafter, items: []const dr.Hold) !void {
        return holdBatchImpl(x, items);
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = self(ptr);
        if (std.c.getenv("TF_INKLING2_DRAFT_TRACE") != null) {
            const T = struct {
                var hist: [9]u64 = @splat(0);
                var calls: u64 = 0;
            };
            T.hist[@min(items.len, 8)] += 1;
            T.calls += 1;
            if (T.calls % 200 == 0) std.debug.print("[holds] {d} calls, by size: {any}\n", .{ T.calls, T.hist });
            const M = struct {
                var ms: [9]f64 = @splat(0);
                var n: [9]u64 = @splat(0);
            };
            try x.s.synchronize();
            var t0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t0);
            try x.holdInner(items);
            try x.s.synchronize();
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            const dt = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
            const b = @min(items.len, 8);
            M.ms[b] += dt;
            M.n[b] += 1;
            if (T.calls % 200 == 0) for (1..9) |i| if (M.n[i] > 0) std.debug.print("[holds] size {d}: {d:.2} ms avg over {d}\n", .{ i, M.ms[i] / @as(f64, @floatFromInt(M.n[i])), M.n[i] });
            return;
        }
        return x.holdInner(items);
    }

    fn holdInner(x: *Drafter, items: []const dr.Hold) !void {
        if (items.len > 1 and batchOn()) {
            var i: usize = 0;
            while (i < items.len) : (i += max_batch) try x.holdBatch(items[i..@min(items.len, i + max_batch)]);
            return;
        }
        for (items) |item| {
            const l = x.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.lanes != null) return error.TreesNotBuilt;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (l.end != anchor) return error.DraftContextGap;
            if (anchor + block > x.context + block) return error.DraftContextGap;
            const n: usize = @min(item.depth, depth);
            const sl = &x.slots.items[l.slot];
            const pending = [1]i32{@intCast(item.pending)};
            try cuda.copy_source.enqueue(x.d, sl.walk, std.mem.sliceAsBytes(&pending), x.s);
            try x.encodeBlock(sl, l, anchor);
            try x.upKeys((&item)[0..1]);
            try x.draftWalk(sl, n);
            l.held = @intCast(n);
        }
    }
};

/// mtp.cu's ink2m::DraftKey.
const DraftKey = extern struct { seed: u64, inv_t: f32, pos: u32 };

const BatchItem = struct { l: *LaneState, sl: *const Slot, anchor: u64, n: usize };

fn holdBatchImpl(x: *Drafter, items: []const dr.Hold) !void {
    var bi: [max_batch]BatchItem = undefined;
    const N = items.len;
    for (items, 0..) |item, b| {
        const l = x.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
        if (item.lanes != null) return error.TreesNotBuilt;
        if (item.depth == 0 or item.position == 0) return error.DraftDepth;
        const anchor = item.position - 1;
        if (l.end != anchor) return error.DraftContextGap;
        if (anchor + block > x.context + block) return error.DraftContextGap;
        const sl = &x.slots.items[l.slot];
        const pending = [1]i32{@intCast(item.pending)};
        try cuda.copy_source.enqueue(x.d, sl.walk, std.mem.sliceAsBytes(&pending), x.s);
        bi[b] = .{ .l = l, .sl = sl, .anchor = anchor, .n = @min(item.depth, depth) };
    }
    try encodeBatch(x, bi[0..N]);
    try x.upKeys(items);
    try walkBatch(x, bi[0..N]);
    for (bi[0..N]) |b| b.l.held = @intCast(b.n);
}

/// encodeBlock for N streams: the matmuls over all N blocks' rows at once, attention and the caches per stream.
fn encodeBatch(x: *Drafter, bi: []const BatchItem) !void {
    const H = hidden;
    const N = bi.len;
    const rows = N * block;
    const ld = q_width + 2 * kv_width;
    for (bi, 0..) |b, s| {
        try ops.embed(&x.tk.ops, x.s, b.sl.walk, 1, x.target.embed, x.target.embed_norm, x.anchor + s * H * 2, H, x.eps);
        var a: cuda.Args = .{};
        for ([_]u64{ x.anchor + s * H * 2, x.mask, x.x + s * block * H * 2 }) |v| a.add(v);
        a.add(@as(i32, H));
        try go(x.k.init_rows, x.s, .{ .x = block }, 256, &a);
    }
    try x.resnormRows(x.x, x.resid, x.layers[0].input_norm, x.resid, x.x, true, rows);
    for (x.layers, 0..) |ly, i| {
        try x.tk.lane.matmulOut(x.s, x.x, H, rows, ly.qkv, x.qkv, .bf16, x.part);
        for (bi, 0..) |b, s| {
            const sl = b.sl;
            const anchor = b.anchor;
            const hi: usize = @intCast(anchor + block - 1);
            var c: cuda.Args = .{};
            c.add(x.qkv + s * block * ld * 2);
            c.add(@as(i32, ld));
            for ([_]u64{ ly.q_norm, ly.k_norm, x.q + s * block * q_width * 2, sl.k[i], sl.v[i] }) |v| c.add(v);
            c.add(@as(i32, heads));
            c.add(@as(i32, kv_heads));
            c.add(@as(i32, @intCast(anchor)));
            c.add(@as(i32, @intCast(sl.rings[i])));
            c.add(x.eps);
            c.add(x.theta);
            try go(x.k.block_qkv, x.s, .{ .x = block, .y = heads + kv_heads }, head_dim, &c);
            const local = i != 1;
            var lo: usize = @intCast(b.l.from);
            if (local and anchor > window) lo = @max(lo, @as(usize, @intCast(anchor)) - window);
            const c0 = lo / chunk;
            const nc = hi / chunk - c0 + 1;
            const aa: AttnArgs = .{ .q = x.q + s * block * q_width * 2, .kc = sl.k[i], .vc = sl.v[i], .po = x.po, .pml = x.pml, .T = block, .hq = heads, .hk = kv_heads, .ring = @intCast(sl.rings[i]), .lo = @intCast(lo), .hi = @intCast(hi), .c0 = @intCast(c0), .scale = 1.0 / @sqrt(@as(f32, head_dim)) };
            var e: cuda.Args = .{};
            e.add(aa);
            try go(x.k.attn, x.s, .{ .x = @intCast(nc), .y = kv_heads }, 256, &e);
            var m: cuda.Args = .{};
            for ([_]u64{ x.po, x.pml }) |v| m.add(v);
            m.add(@as(i32, @intCast(nc)));
            m.add(@as(i32, block * heads));
            m.add(x.att + s * block * q_width * 2);
            try go(x.k.merge, x.s, .{ .x = block * heads }, head_dim, &m);
        }
        try x.tk.lane.matmulOut(x.s, x.att, q_width, rows, ly.o, x.y, .bf16, x.part);
        try x.resnormRows(x.y, x.resid, ly.post_norm, x.next_resid, x.x, false, rows);
        try x.tk.lane.matmulOut(x.s, x.x, H, rows, ly.gu, x.gu, .bf16, x.part);
        var g: cuda.Args = .{};
        for ([_]u64{ x.gu, x.gu + inter * 2 }) |v| g.add(v);
        g.add(@as(i32, 2 * inter));
        g.add(x.act);
        g.add(@as(i32, inter));
        try go(x.k.swiglu, x.s, .{ .x = (inter + 255) / 256, .y = @intCast(rows) }, 256, &g);
        try x.tk.lane.matmulOut(x.s, x.act, inter, rows, ly.down, x.y, .bf16, x.part);
        const next = if (i + 1 < x.layers.len) x.layers[i + 1].input_norm else x.norm;
        try x.resnormRows(x.y, x.next_resid, next, x.resid, x.x, false, rows);
    }
    // every row's head logits (row 0 of each block unused), then each stream's Markov contexts
    try x.tk.lane.matmulOut(x.s, x.x, H, rows, x.draftHead(), x.base, .bf16, x.part);
    for (bi, 0..) |b, s| {
        var k: cuda.Args = .{};
        for ([_]u64{ x.x + (s * block + 1) * H * 2, b.sl.newest, x.rnn, x.rnn_bias, x.kappa + s * depth * 3 * rank * 4 }) |v| k.add(v);
        k.add(@as(i32, depth));
        k.add(@as(i32, H));
        k.add(@as(i32, rank));
        try go(x.k.kappa, x.s, .{ .x = 3 * rank / 8 }, 256, &k);
    }
}

/// draftWalk for N streams: the recurrence and picks per stream, each step's folded logits for all in one phi read.
fn walkBatch(x: *Drafter, bi: []const BatchItem) !void {
    const V = x.vocab;
    const N = bi.len;
    var steps: usize = 0;
    for (bi) |b| steps = @max(steps, b.n);
    const parts = pickParts(V);
    for (0..steps) |j| {
        for (bi, 0..) |b, s| {
            var a: cuda.Args = .{};
            for ([_]u64{ x.kappa + s * depth * 3 * rank * 4, x.rnn, x.w1, b.sl.walk }) |v| a.add(v);
            a.add(@as(i32, @intCast(j)));
            for ([_]u64{ x.state + s * rank * 2, x.raw + s * 3 * rank * 4 }) |v| a.add(v);
            a.add(@as(i32, rank));
            a.add(@as(i32, hidden));
            try go(x.k.rnn_raw, x.s, .{ .x = 3 * rank / 8 }, 256, &a);
            var g: cuda.Args = .{};
            g.add(x.raw + s * 3 * rank * 4);
            g.add(@as(i32, @intCast(j)));
            for ([_]u64{ x.state + s * rank * 2, x.latent + s * rank * 2 }) |v| g.add(v);
            g.add(@as(i32, rank));
            try go(x.k.rnn_gate, x.s, .{ .x = 1 }, rank, &g);
        }
        try x.vocabStep(x.base + (1 + j) * V * 2, block * V, N, j);
        for (bi, 0..) |b, s| {
            var c: cuda.Args = .{};
            c.add(x.logits + s * V * 4);
            c.add(@as(i32, @intCast(V)));
            c.add(x.picks + s * parts * 8);
            try go(x.k.pick_part, x.s, .{ .x = @intCast(parts) }, 256, &c);
            var e: cuda.Args = .{};
            e.add(x.picks + s * parts * 8);
            e.add(@as(i32, @intCast(parts)));
            e.add(b.sl.walk);
            e.add(@as(i32, @intCast(j)));
            try go(x.k.pick, x.s, .{ .x = 1 }, 256, &e);
        }
    }
}

fn pickParts(v: usize) usize {
    return (v + 4095) / 4096;
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// The checkpoint's mtp_config must be the head this engine was written for.
fn admit(c: *const config.Config) !void {
    const m = c.mtp orelse return error.DSparkNotConfigured;
    if (c.text.hidden != hidden or m.layers != 3 or m.q_heads != heads or m.kv_heads != kv_heads or m.head_dim != head_dim or
        m.inter != inter or m.window != window or m.block != block or m.rank != rank or m.taps_n != taps.len or
        !m.local[0] or m.local[1] or !m.local[2] or !std.mem.eql(u32, m.taps[0..m.taps_n], &taps))
        return error.UnsupportedDSparkConfig;
}

test "rings hold a block past the window, and a stream's bytes count all three layers" {
    try std.testing.expectEqual(@as(usize, 1152), local_ring);
    try std.testing.expectEqual(@as(usize, 32896), globalRing(32768));
    try std.testing.expectEqual(2 * 2 * 1152 * 1024 * 2 + 2 * 32896 * 1024 * 2 + 2304 * 2 + 256, laneBytes(32768));
    try std.testing.expectEqual(@as(usize, 4 * 2304 * 2), row_bytes);
}

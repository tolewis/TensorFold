//! Inkling 2 Mini's DFlash2 drafter (--drafter <folder of incoai/Inkling-2-Mini-DFlash2>, Apache-2.0, read unmodified)
//! on CUDA: the target's committed features (visits 5, 23, 41 and the final norm, as the DSpark head taps) through fc and
//! the hidden norm into five layers' context rings, then one eight-row block (the pending token and seven masks) through
//! the layers' two-tap convs, non-causal attention and SwiGLU, the target's head (an FP8 copy) for each row's 16 best
//! tokens, and the candidate selector's greedy path. It runs on the target's stream, so the tapped rows are read in
//! order with no fence. Math: z-lab/dflash (MIT) as SGLang serves it, and GLM-5.3-Flash's CUDA DFlash2 (dflash.cu).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const wts = @import("mini_weights.zig");
const mm = @import("mini_model.zig");
const ops = @import("ops.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Lane = cuda.qmmf.Weight;
const DeviceBuffer = cuda.DeviceBuffer;

/// The target's tapped visits, in the order fc's input concatenates them (dflash_config.target_layer_ids).
pub const taps = [_]u32{ 5, 23, 41, 48 };
pub const hidden = 2304;
const H = hidden;
pub const layers_n = 5;
const q_heads = 32;
const kv_heads = 8;
const head_dim = 128;
const q_width = q_heads * head_dim;
const kv_width = kv_heads * head_dim;
const qkv_width = q_width + 2 * kv_width;
pub const inter = 6912;
const group = 16;
const dyn_width = 2 * 2 * (H / group); // the conv projections' [input/output][tap][group]
pub const rank = 256;
const top = 16;
/// A proposal block's rows (the pending token and seven masks) and the drafts it offers.
pub const block = 8;
pub const depth = block - 1;
/// Every layer's sliding window, and a ring's slots.
pub const window = 2048;
pub const mask_token: u32 = 200056;
const theta: f32 = 10000;
const eps: f32 = 1e-6;
const cap: f32 = 30;
/// Streams one block pass drafts for; more are taken in groups.
pub const max_batch = 16;
const max_rows = max_batch * block;
const row_bytes = taps.len * H * 2;
const chunk_cols = 4096;
const attn_chunks = 16;

/// A stream's context ring: K and V of every layer, a window of positions each.
pub const ring_bytes: usize = layers_n * 2 * window * kv_width * 2;

/// Device bytes a stream holds: its ring and its walk.
pub fn laneBytes() usize {
    return ring_bytes + 256;
}

fn envOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return true;
    return v[0] != '0';
}
/// TF_INKLING2_HOLD_COSTS=0: holds priced free.
fn costsOn() bool {
    return envOn("TF_INKLING2_HOLD_COSTS");
}
/// TF_INKLING2_DRAFT_FP8=0: candidate logits from the target's BF16 head instead of an FP8 copy (draft only).
fn fp8HeadOn() bool {
    return envOn("TF_INKLING2_DRAFT_FP8");
}
/// TF_INKLING2_DFLASH2_HEAD4=0: the FP8 head copy (above). On (the default): an NVFP4 copy of the target's head, a
/// quarter of its bytes, quantized on the host at load (draft logits only: verification keeps the BF16 head).
/// After tonyd2wild's patch 0018 (github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark): hold 4.4 ms FP8, 3.6 NVFP4.
fn head4On() bool {
    return envOn("TF_INKLING2_DFLASH2_HEAD4");
}
/// TF_INKLING2_DRAFT_KEYED=0: sampled streams get greedy paths (on: the target's keyed noise picks among the candidates).
fn keyedOn() bool {
    return envOn("TF_INKLING2_DRAFT_KEYED");
}
/// dflash.cu's DraftKey (mtp.cu's ink2m::DraftKey).
const DraftKey = extern struct { seed: u64, inv_t: f32, pos: u32 };
/// TF_INKLING2_DRAFT_LAYERS8=0: the block pass's linears in BF16; on: per-tensor FP8 (draft only, verify unchanged).
fn layers8On() bool {
    return envOn("TF_INKLING2_DRAFT_LAYERS8");
}

const HoldCost = lanes.hold_price.HoldCost;
/// A hold pass's ms on one GB10 (TF_INKLING2_DRAFT_TRACE=1): by streams held together; one block a stream at any depth.
/// (2026-11-24, FP8 layers and head, 262,144-token streams.)
const hold_costs = [_]HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 4.3 }, .{ .streams = 1, .depth = 7, .ms = 4.3 },
    .{ .streams = 2, .depth = 1, .ms = 4.6 }, .{ .streams = 2, .depth = 7, .ms = 4.6 },
    .{ .streams = 4, .depth = 1, .ms = 5.2 }, .{ .streams = 4, .depth = 7, .ms = 5.2 },
    .{ .streams = 8, .depth = 1, .ms = 6.4 }, .{ .streams = 8, .depth = 7, .ms = 6.4 },
};
/// TF_INKLING2_DFLASH2_CHANCES=0: the depth rule's per-depth rates price the drafts. On (the default): a stream alone
/// gives each draft's chance of landing, the selector's own probability calibrated by what landed (lanes' Landing
/// table, greedy and sampled apart), so a round verifies only the rows worth their cost. The raw probabilities alone
/// were too low (1.08 tokens a round); calibrated and lone-stream only, after tonyd2wild's patch 0018.
fn chancesOn() bool {
    return envOn("TF_INKLING2_DFLASH2_CHANCES");
}
/// TF_INKLING2_DFLASH2_CHANCES_SHARED=1: chances also while other streams are open (off: shared rounds keep the
/// depth rule's rates, measured faster there).
fn chancesSharedOn() bool {
    const v = std.c.getenv("TF_INKLING2_DFLASH2_CHANCES_SHARED") orelse return false;
    return v[0] == '1';
}
/// TF_INKLING2_DRAFT_CONFIDENCE (with chances): every round holds the full block and verifies the drafts at least
/// this likely; unset: each draft priced by its chance.
fn confidence() ?f64 {
    const v = std.c.getenv("TF_INKLING2_DRAFT_CONFIDENCE") orelse return null;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch null;
}

const Layer = struct {
    qkv: Lane, // [q | k | v][H]
    kv: Lane, // [k | v][H]: the context rows' (bf16, prompt-sized passes)
    o: Lane,
    gu: Lane, // [gate | up][H]
    down: Lane,
    a_proj: Lane, // attention_conv.kernel_projection
    m_proj: Lane, // mlp_conv.kernel_projection
    in_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    a_base: u64, // bf16 [2][2][H]
    m_base: u64,
};

const Kernels = struct {
    mod: cuda.Module,
    mtp: cuda.Module,
    norm: cuda.Function,
    conv: cuda.Function,
    swiglu: cuda.Function,
    block_qkv: cuda.Function,
    ctx_kv: cuda.Function,
    attn_part: cuda.Function,
    attn_merge: cuda.Function,
    top16_part: cuda.Function,
    top16_merge: cuda.Function,
    select: cuda.Function,
    amax: cuda.Function,
    fp8_lane: cuda.Function,

    fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_dflash);
        errdefer mod.unload();
        var mtp = try cuda.Module.load(d, cuda.kernels.inkling2_mtp);
        errdefer mtp.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        k.mtp = mtp;
        inline for (.{ "norm", "conv", "swiglu", "block_qkv", "ctx_kv", "attn_part", "attn_merge", "top16_part", "top16_merge", "select" }) |name| {
            @field(k, name) = try mod.function("tf_ink2d_" ++ name);
        }
        k.amax = try mtp.function("tf_ink2m_amax");
        k.fp8_lane = try mtp.function("tf_ink2m_fp8_lane");
        return k;
    }
};

/// One stream's draft state: its ring and its walk (int32 [block]: the pending token, then the drafts).
const Slot = struct {
    ring: u64,
    walk: u64,
    used: bool = false,
};

const LaneState = struct {
    slot: usize,
    from: u64 = 0, // the first committed position the ring holds for this sequence (a resumed prompt starts past 0)
    end: u64 = 0, // positions [from, end) absorbed
    held: u32 = 0,
    anchor: u64 = 0, // the last hold's pending position
    fetched: bool = false, // the last hold's walk is on the host
    walk: [2 * block]u32 = @splat(0), // the walk as read back: the pending token and drafts, then each step's chance

    fn draft(l: *const LaneState, i: usize) u32 {
        return l.walk[1 + i];
    }

    /// Draft i's log-chance under the selector (the softmax of its step's 16 soft-capped scores).
    fn own(l: *const LaneState, i: usize) f64 {
        const c: f32 = @bitCast(l.walk[block + 1 + i]);
        if (!(c > 0) or !std.math.isFinite(c)) return -20;
        return @log(@min(@as(f64, c), 1));
    }
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    target: *const wts.Weights,
    tk: *const mm.Kernels,
    k: Kernels,
    context: usize,
    vocab: usize, // the unpadded vocabulary the target's head writes
    target_eps: f32,
    layers: [layers_n]Layer = undefined,
    fc: Lane = undefined,
    hproj: Lane = undefined,
    hnorm: u64 = 0,
    norm: u64 = 0,
    pred: u64 = 0, // the selector's codebooks, bf16 [vocab][rank]
    succ: u64 = 0,
    head: ?Lane = null, // TF_INKLING2_DRAFT_FP8: an FP8 copy of the target's unembedding for candidate logits
    // scratch: one block pass or one absorb at a time on the target's stream
    x: u64 = 0,
    y: u64 = 0,
    y2: u64 = 0,
    dyn: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kb: u64 = 0,
    vb: u64 = 0,
    att: u64 = 0,
    a: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    logits: u64 = 0, // bf16 [max_rows][vocab]
    pv: u64 = 0,
    pi: u64 = 0,
    cand: u64 = 0, // int32 [max_rows][2 * top]
    g: u64 = 0, // fp32 [max_rows][rank]
    pm: u64 = 0,
    pl: u64 = 0,
    pacc: u64 = 0,
    ids: u64 = 0, // int32 [max_rows]
    bpos: u64 = 0, // int32 [max_rows]
    rings: u64 = 0, // u64 [max_batch]
    starts: u64 = 0, // i64 [max_batch]
    froms: u64 = 0, // i64 [max_batch]
    walks: u64 = 0, // u64 [max_batch]
    keys: u64 = 0, // DraftKey [max_batch]
    c1: u64 = 0, // bf16 [window][H]
    ctx: u64 = 0, // bf16 [window][H]
    ckv: u64 = 0, // bf16 [window][2 * kv]
    part: u64 = 0,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    weight_bytes: usize = 0,
    slots: std.ArrayList(Slot) = .empty,
    slot_bufs: std.ArrayList(DeviceBuffer) = .empty,
    lanes: std.AutoHashMapUnmanaged(*Stream, LaneState) = .empty,
    passes: u64 = 0,
    landing: lanes.draft_tree.Landing = .{},
    chances: bool = false,
    chances_shared: bool = false,

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
            if (p.dim(1) != k) return error.DFlash2TensorShape;
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
        try admit(gpa, io, dir);
        const c = &target.config;
        if (c.text.hidden != hidden or c.text.unpadded_vocab <= mask_token) return error.UnsupportedDFlash2Config;
        const x = try gpa.create(Drafter);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .d = d, .s = s, .target = target, .tk = tk, .k = try Kernels.load(d), .context = context, .vocab = c.text.unpadded_vocab, .target_eps = c.text.eps };
        errdefer x.deinit();
        var ck: core.checkpoint.Checkpoint = .{ .gpa = gpa, .io = io };
        defer ck.close();
        ck.add(dir, "model.safetensors") catch |e| {
            std.log.err("Inkling 2 Mini DFlash2: cannot open {s}/model.safetensors ({s})", .{ dir, @errorName(e) });
            return error.DFlash2MissingFile;
        };
        try x.loadWeights(&ck);
        if (layers8On()) for (&x.layers) |*l| {
            inline for (.{ "qkv", "o", "gu", "down", "a_proj", "m_proj" }) |f| @field(l, f) = try x.fp8Of(@field(l, f));
        };
        if (head4On()) x.head = try x.fp4Of(x.target.unembed) else if (fp8HeadOn()) x.head = try x.fp8Of(x.target.unembed);
        x.chances = chancesOn();
        x.chances_shared = chancesSharedOn();
        try x.scratch();
        try x.s.synchronize();
        return x;
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
        if (!(m > 0) or !std.math.isFinite(m)) return error.DFlash2WeightRange;
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

    /// A bf16 lane weight as an NVFP4 lane weight: read back 64-row tiles at a time, unpacked (qmmf.packBf16's
    /// order), quantized on the host as ModelOpt does (global scale amax / (6 * 448), an e4m3 scale a 16 inputs at
    /// the block's amax / 6, e2m1 codes to nearest), packed (nvfp4.packWords / packScales) and uploaded. From
    /// tonyd2wild's patch 0018 (Apache-2.0).
    fn fp4Of(x: *Drafter, u: Lane) !Lane {
        const n: usize = u.n;
        const k: usize = u.k;
        const npad: usize = u.npad;
        if (k % 64 != 0) return error.DFlash2WeightRange;
        var amax = try DeviceBuffer.alloc(x.d, 256);
        defer amax.free();
        try x.d.check(x.d.api.cuMemsetD8Async(amax.ptr, 0, 4, x.s.handle), "cuMemsetD8Async");
        var a: cuda.Args = .{};
        a.add(u.codes);
        a.add(@as(i64, @intCast(npad * k)));
        a.add(amax.ptr);
        try go(x.k.amax, x.s, .{ .x = 1024 }, 256, &a);
        try x.s.synchronize();
        var bits: u32 = 0;
        try cuda.memory.download(x.d, amax.ptr, std.mem.asBytes(&bits));
        const m: f32 = @bitCast(bits);
        if (!(m > 0) or !std.math.isFinite(m)) return error.DFlash2WeightRange;
        const gs: f32 = m / (6.0 * 448.0);
        var e4: [127]f32 = undefined; // e4m3's non-negative values by code
        for (&e4, 0..) |*v, c| {
            const ex: i32 = @intCast(c >> 3);
            const mant: f32 = @floatFromInt(c & 7);
            v.* = if (ex == 0) mant / 8.0 * std.math.pow(f32, 2, -6) else (1.0 + mant / 8.0) * std.math.pow(f32, 2, @floatFromInt(ex - 7));
        }
        const codes = try x.gpa.alloc(u8, n * k / 2);
        defer x.gpa.free(codes);
        const scales = try x.gpa.alloc(u8, n * k / 16);
        defer x.gpa.free(scales);
        const tile = 64 * k; // one 64-row tile, packed
        const per = 64;
        const buf = try x.gpa.alloc(u16, per * tile);
        defer x.gpa.free(buf);
        const rows = try x.gpa.alloc(u16, per * tile);
        defer x.gpa.free(rows);
        const kg = k / 64;
        var t: usize = 0;
        while (t < npad / 64) : (t += per) {
            const nt = @min(per, npad / 64 - t);
            try cuda.memory.download(x.d, u.codes + t * tile * 2, std.mem.sliceAsBytes(buf[0 .. nt * tile]));
            for (buf[0 .. nt * tile], 0..) |v, li| {
                const i = t * tile + li;
                const e = i % 4;
                const lane = (i / 4) % 32;
                const kt = (i / 128) % 4;
                const jj = (i / 512) % 8;
                const tg = i / 4096;
                const row = (tg / kg) * 64 + jj * 8 + lane / 4;
                const col = (tg % kg) * 64 + kt * 16 + 2 * (lane % 4) + (e & 1) + 8 * (e >> 1);
                rows[(row - t * 64) * k + col] = v;
            }
            const r_end = @min(n, (t + nt) * 64);
            var r = t * 64;
            while (r < r_end) : (r += 1) {
                const src = rows[(r - t * 64) * k ..][0..k];
                for (0..k / 16) |b| {
                    var bmax: f32 = 0;
                    for (src[b * 16 ..][0..16]) |h| bmax = @max(bmax, @abs(bf32(h)));
                    const want = bmax / 6.0 / gs;
                    // the nearest e4m3 code: the first value at or past `want`, or the one below it
                    var lo: usize = 0;
                    var hi: usize = e4.len - 1;
                    while (lo < hi) {
                        const mid = (lo + hi) / 2;
                        if (e4[mid] < want) lo = mid + 1 else hi = mid;
                    }
                    const sc: usize = if (lo > 0 and want - e4[lo - 1] <= e4[lo] - want) lo - 1 else lo;
                    scales[r * (k / 16) + b] = @intCast(sc);
                    const step = e4[sc] * gs;
                    for (0..8) |q| {
                        var byte: u8 = 0;
                        for (0..2) |h| {
                            const val = bf32(src[b * 16 + 2 * q + h]);
                            const mag = if (step > 0) @abs(val) / step else 0;
                            const idx: u8 = if (mag < 0.25) 0 else if (mag < 0.75) 1 else if (mag < 1.25) 2 else if (mag < 1.75) 3 else if (mag < 2.5) 4 else if (mag < 3.5) 5 else if (mag < 5.0) 6 else 7;
                            const nib: u8 = idx | (if (val < 0 and idx != 0) @as(u8, 8) else 0);
                            byte |= nib << @intCast(4 * h);
                        }
                        codes[(r * k + b * 16) / 2 + q] = byte;
                    }
                }
            }
        }
        const words = try x.gpa.alloc(u32, npad / 64 * kg * 512);
        defer x.gpa.free(words);
        cuda.nvfp4.packWords(words, codes, n, k);
        const tiles = try x.gpa.alloc(u8, npad * (k / 16));
        defer x.gpa.free(tiles);
        cuda.nvfp4.packScales(tiles, scales, n, k);
        return cuda.nvfp4.weight(try x.upload(std.mem.sliceAsBytes(words)), try x.upload(tiles), gs, n, k);
    }

    fn get(ck: *core.checkpoint.Checkpoint, name: []const u8, shape: []const usize) !core.checkpoint.Tensor {
        const t = ck.get(name) catch {
            std.log.err("Inkling 2 Mini DFlash2: model.safetensors has no {s}", .{name});
            return error.DFlash2MissingTensor;
        };
        if (!t.is(.bf16, shape)) {
            std.log.err("Inkling 2 Mini DFlash2: {s} is not bf16 of the expected shape", .{name});
            return error.DFlash2TensorShape;
        }
        return t;
    }

    fn layerTensor(ck: *core.checkpoint.Checkpoint, i: usize, suffix: []const u8, shape: []const usize) !core.checkpoint.Tensor {
        var b: [96]u8 = undefined;
        return get(ck, try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, suffix }), shape);
    }

    fn loadWeights(x: *Drafter, ck: *core.checkpoint.Checkpoint) !void {
        for (&x.layers, 0..) |*l, i| {
            const q = try layerTensor(ck, i, "self_attn.q_proj.weight", &.{ q_width, H });
            const k = try layerTensor(ck, i, "self_attn.k_proj.weight", &.{ kv_width, H });
            const v = try layerTensor(ck, i, "self_attn.v_proj.weight", &.{ kv_width, H });
            const g = try layerTensor(ck, i, "mlp.gate_proj.weight", &.{ inter, H });
            const u = try layerTensor(ck, i, "mlp.up_proj.weight", &.{ inter, H });
            l.qkv = try x.pack(&.{ q, k, v });
            l.kv = try x.pack(&.{ k, v });
            l.o = try x.pack(&.{try layerTensor(ck, i, "self_attn.o_proj.weight", &.{ H, q_width })});
            l.gu = try x.pack(&.{ g, u });
            l.down = try x.pack(&.{try layerTensor(ck, i, "mlp.down_proj.weight", &.{ H, inter })});
            l.a_proj = try x.pack(&.{try layerTensor(ck, i, "attention_conv.kernel_projection.weight", &.{ dyn_width, H })});
            l.m_proj = try x.pack(&.{try layerTensor(ck, i, "mlp_conv.kernel_projection.weight", &.{ dyn_width, H })});
            l.in_norm = try x.upload((try layerTensor(ck, i, "input_layernorm.weight", &.{H})).bytes);
            l.post_norm = try x.upload((try layerTensor(ck, i, "post_attention_layernorm.weight", &.{H})).bytes);
            l.q_norm = try x.upload((try layerTensor(ck, i, "self_attn.q_norm.weight", &.{head_dim})).bytes);
            l.k_norm = try x.upload((try layerTensor(ck, i, "self_attn.k_norm.weight", &.{head_dim})).bytes);
            l.a_base = try x.upload((try layerTensor(ck, i, "attention_conv.base_kernel", &.{ 2, 2, H })).bytes);
            l.m_base = try x.upload((try layerTensor(ck, i, "mlp_conv.base_kernel", &.{ 2, 2, H })).bytes);
            ck.release();
        }
        x.fc = try x.pack(&.{try get(ck, "fc.weight", &.{ H, taps.len * H })});
        x.hproj = try x.pack(&.{try get(ck, "candidate_selector.hidden_projection.weight", &.{ rank, H })});
        x.hnorm = try x.upload((try get(ck, "hidden_norm.weight", &.{H})).bytes);
        x.norm = try x.upload((try get(ck, "norm.weight", &.{H})).bytes);
        const book_rows = x.target.config.text.vocab;
        const pred = try get(ck, "candidate_selector.predecessor_codebook", &.{ book_rows, rank });
        x.pred = try x.upload(pred.bytes[0 .. x.vocab * rank * 2]);
        const succ = try get(ck, "candidate_selector.successor_codebook", &.{ book_rows, rank });
        x.succ = try x.upload(succ.bytes[0 .. x.vocab * rank * 2]);
        ck.release();
    }

    fn scratch(x: *Drafter) !void {
        const M = max_rows;
        const R = window;
        const V = x.vocab;
        const chunks = (V + chunk_cols - 1) / chunk_cols;
        if (chunks > 64) return error.UnsupportedDFlash2Config;
        x.x = try x.alloc(M * H * 2);
        x.y = try x.alloc(M * H * 2);
        x.y2 = try x.alloc(M * H * 2);
        x.dyn = try x.alloc(M * dyn_width * 2);
        x.qkv = try x.alloc(M * qkv_width * 2);
        x.q = try x.alloc(M * q_width * 2);
        x.kb = try x.alloc(M * kv_width * 2);
        x.vb = try x.alloc(M * kv_width * 2);
        x.att = try x.alloc(M * q_width * 2);
        x.a = try x.alloc(M * H * 2);
        x.gu = try x.alloc(M * 2 * inter * 2);
        x.act = try x.alloc(M * inter * 2);
        x.logits = try x.alloc(M * V * 2);
        x.pv = try x.alloc(M * chunks * top * 4);
        x.pi = try x.alloc(M * chunks * top * 4);
        x.cand = try x.alloc(M * 2 * top * 4);
        x.g = try x.alloc(M * rank * 4);
        x.pm = try x.alloc(M * q_heads * attn_chunks * 4);
        x.pl = try x.alloc(M * q_heads * attn_chunks * 4);
        x.pacc = try x.alloc(M * q_heads * attn_chunks * head_dim * 4);
        x.ids = try x.alloc(M * 4);
        x.bpos = try x.alloc(M * 4);
        x.rings = try x.alloc(max_batch * 8);
        x.starts = try x.alloc(max_batch * 8);
        x.froms = try x.alloc(max_batch * 8);
        x.walks = try x.alloc(max_batch * 8);
        x.keys = try x.alloc(max_batch * @sizeOf(DraftKey));
        x.c1 = try x.alloc(R * H * 2);
        x.ctx = try x.alloc(R * H * 2);
        x.ckv = try x.alloc(R * 2 * kv_width * 2);
        var part: usize = 0;
        for (x.layers) |l| {
            for ([_]Lane{ l.qkv, l.o, l.gu, l.down, l.a_proj, l.m_proj }) |w| for (1..max_batch + 1) |nb| {
                part = @max(part, x.tk.lane.partBytes(nb * block, w));
            };
            for (1..R + 1) |m| part = @max(part, x.tk.lane.partBytes(m, l.kv));
        }
        for (1..R + 1) |m| part = @max(part, x.tk.lane.partBytes(m, x.fc));
        for (1..max_batch + 1) |nb| {
            part = @max(part, x.tk.lane.partBytes(nb * block, x.draftHead()));
            part = @max(part, x.tk.lane.partBytes(nb * block, x.hproj));
        }
        x.part = try x.alloc(part);
    }

    /// The weight candidate logits read: the FP8 copy if built, else the target's own head.
    fn draftHead(x: *const Drafter) Lane {
        return x.head orelse x.target.unembed;
    }

    pub fn deinit(x: *Drafter) void {
        x.lanes.deinit(x.gpa);
        for (x.slot_bufs.items) |*b| b.free();
        x.slot_bufs.deinit(x.gpa);
        x.slots.deinit(x.gpa);
        for (x.buffers.items) |*b| b.free();
        x.buffers.deinit(x.gpa);
        x.k.mtp.unload();
        x.k.mod.unload();
        x.gpa.destroy(x);
    }

    /// Device bytes besides the streams': weights, the head copy and the scratch.
    pub fn fixedBytes(x: *const Drafter) usize {
        var n: usize = 0;
        for (x.buffers.items) |b| n += b.len;
        return n;
    }

    /// `streams` slots' rings, allocated now (the host's count); a smaller count than held keeps them.
    pub fn reserve(x: *Drafter, streams: u32) !void {
        while (x.slots.items.len < streams) {
            const ring = try x.slotAlloc(ring_bytes);
            const walk = try x.slotAlloc(2 * block * 4); // int32 [block] path, fp32 [block] chances
            try x.slots.append(x.gpa, .{ .ring = ring, .walk = walk });
        }
    }

    fn slotAlloc(x: *Drafter, bytes: usize) !u64 {
        var b = try DeviceBuffer.alloc(x.d, bytes);
        errdefer b.free();
        try x.slot_bufs.append(x.gpa, b);
        try x.d.check(x.d.api.cuMemsetD8_v2(b.ptr, 0, bytes), "cuMemsetD8");
        return b.ptr;
    }

    pub fn drafter(x: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .chances = chancesFn } };
    }

    fn self(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &taps;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const x = self(ptr);
        const costs: []const HoldCost = if (costsOn()) &hold_costs else &.{};
        return .{ .depth = depth, .step_ms = 0, .fixed_block = true, .plain_guard = true, .batched = true, .stream_bytes = laneBytes(), .hold_costs = costs, .chances = x.chances, .slack = if (x.chances) depth else 0, .confidence_floor = confidence() };
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

    /// The stream's last walk on the host (one sync, then a small read).
    fn fetch(x: *Drafter, l: *LaneState) !void {
        if (l.fetched) return;
        try x.s.synchronize();
        try cuda.memory.download(x.d, x.slots.items[l.slot].walk, std.mem.sliceAsBytes(&l.walk));
        for (0..l.held) |i| if (l.draft(i) >= x.vocab) return error.DraftNonfinite;
        l.fetched = true;
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = self(ptr);
        for (streams, out) |s, toks| {
            const l = x.lanes.getPtr(s) orelse return error.DraftStreamUnknown;
            if (toks.len > l.held) return error.DraftHeldFewer;
            try x.fetch(l);
            for (toks, 0..) |*t, i| t.* = l.draft(i);
        }
    }

    /// Each held draft's chance of landing, its predecessors' included: the selector's own probability, calibrated by
    /// the drafts that landed (lanes' Landing table, by log-chance). A stream alone only (chances_shared: always).
    fn chancesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const x = self(ptr);
        if (!x.chances) return false;
        const l = x.lanes.getPtr(s) orelse return false;
        if (out.len > l.held) return false;
        if (x.lanes.count() > 1 and !x.chances_shared) return false;
        if (out.len == 0) return true;
        try x.fetch(l);
        var run: f64 = 1;
        for (out, 0..) |*o, i| {
            run *= x.landing.chance(s.sampling != null, l.own(i));
            o.* = run;
        }
        return true;
    }

    /// The rows a verify kept (the pending row and the drafts that landed) and the target's token after the last:
    /// each kept draft landed, the next one landed iff the target's token is it.
    fn observe(x: *Drafter, l: *const LaneState, item: dr.Absorb) void {
        if (!l.fetched or l.held == 0 or item.start != l.anchor or item.prompt) return;
        const rows: usize = @intCast(item.features.rows);
        if (rows == 0) return;
        const kept = @min(rows - 1, l.held);
        const sampled = item.stream.sampling != null;
        for (0..kept) |i| x.landing.observe(sampled, l.own(i), true);
        if (kept < l.held and item.follow.len > kept) x.landing.observe(sampled, l.own(kept), item.follow[kept] == l.draft(kept));
    }

    fn mm_(x: *Drafter, in: u64, ld: usize, m: usize, w: Lane, out: u64) !void {
        try x.tk.lane.matmulOut(x.s, in, ld, m, w, out, .bf16, x.part);
    }

    fn normRows(x: *Drafter, in: u64, w: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ in, w, out }) |v| a.add(v);
        a.add(@as(i32, H));
        a.add(eps);
        try go(x.k.norm, x.s, .{ .x = @intCast(rows) }, 256, &a);
    }

    /// `n` committed rows of features (bf16 [n][4][H]) at positions pos0.. into `ring`: fc, the hidden norm, each
    /// layer's k (normed, rotated) and v.
    fn ingest(x: *Drafter, ring: u64, features: u64, pos0: u64, n: usize) !void {
        try x.mm_(features, taps.len * H, n, x.fc, x.c1);
        try x.normRows(x.c1, x.hnorm, x.ctx, n);
        for (x.layers, 0..) |l, li| {
            try x.mm_(x.ctx, H, n, l.kv, x.ckv);
            var b: cuda.Args = .{};
            b.add(x.ckv);
            b.add(l.k_norm);
            b.add(@as(i64, @intCast(pos0)));
            b.add(ring);
            b.add(@as(i32, @intCast(li)));
            b.add(eps);
            b.add(theta);
            try go(x.k.ctx_kv, x.s, .{ .x = @intCast(n), .y = kv_heads }, 64, &b);
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
            if (x.chances) x.observe(l, item);
            if (item.start > l.end or l.end == 0) l.from = item.start; // a gap (a resumed prompt): attend from here
            const sl = &x.slots.items[l.slot];
            // only the last window of rows can be attended to: earlier ones would be overwritten in the ring
            const skip: usize = @as(usize, f.rows) -| window;
            const n: usize = f.rows - skip;
            try x.ingest(sl.ring, f.buffer + f.offset + skip * row_bytes, item.start + skip, n);
            l.end = item.start + f.rows; // rows past a rewound end are rewritten in place
            l.held = 0;
            l.fetched = false;
        }
    }

    fn conv(x: *Drafter, z: u64, which: i32, base: u64, out: u64, add: bool, M: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ z, x.dyn }) |v| a.add(v);
        a.add(which);
        for ([_]u64{ base, out }) |v| a.add(v);
        a.add(@as(i32, @intFromBool(add)));
        a.add(@as(i32, H));
        try go(x.k.conv, x.s, .{ .x = @intCast(M) }, 256, &a);
    }

    /// The block pass for `S` streams (tables uploaded): embed, five layers, norm, each row's 16 best, the path.
    fn pass(x: *Drafter, S: usize, keyed: bool) !void {
        const M = S * block;
        try ops.embed(&x.tk.ops, x.s, x.ids, M, x.target.embed, x.target.embed_norm, x.x, H, x.target_eps);
        for (x.layers, 0..) |l, li| {
            // attention, wrapped in the attention conv
            try x.normRows(x.x, l.in_norm, x.y, M);
            try x.mm_(x.y, H, M, l.a_proj, x.dyn);
            try x.conv(x.y, 0, l.a_base, x.y2, false, M);
            try x.mm_(x.y2, H, M, l.qkv, x.qkv);
            {
                var a: cuda.Args = .{};
                a.add(x.qkv);
                a.add(@as(i32, qkv_width));
                for ([_]u64{ l.q_norm, l.k_norm, x.bpos }) |v| a.add(v);
                a.add(eps);
                a.add(theta);
                for ([_]u64{ x.q, x.kb, x.vb }) |v| a.add(v);
                try go(x.k.block_qkv, x.s, .{ .x = @intCast(M), .y = q_heads + kv_heads }, 64, &a);
            }
            {
                var a: cuda.Args = .{};
                for ([_]u64{ x.q, x.kb, x.vb, x.rings, x.starts, x.froms }) |v| a.add(v);
                a.add(@as(i32, @intCast(li)));
                a.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
                for ([_]u64{ x.pm, x.pl, x.pacc }) |v| a.add(v);
                try go(x.k.attn_part, x.s, .{ .x = @intCast(S), .y = kv_heads, .z = attn_chunks }, 256, &a);
                var b: cuda.Args = .{};
                for ([_]u64{ x.pm, x.pl, x.pacc, x.att }) |v| b.add(v);
                try go(x.k.attn_merge, x.s, .{ .x = @intCast(M), .y = q_heads }, head_dim, &b);
            }
            try x.mm_(x.att, q_width, M, l.o, x.a);
            try x.conv(x.a, 1, l.a_base + 2 * H * 2, x.x, true, M);
            // the MLP, wrapped in the MLP conv
            try x.normRows(x.x, l.post_norm, x.y, M);
            try x.mm_(x.y, H, M, l.m_proj, x.dyn);
            try x.conv(x.y, 0, l.m_base, x.y2, false, M);
            try x.mm_(x.y2, H, M, l.gu, x.gu);
            {
                var a: cuda.Args = .{};
                a.add(x.gu);
                a.add(x.act);
                a.add(@as(i32, inter));
                try go(x.k.swiglu, x.s, .{ .x = @intCast(M), .y = (inter + 1023) / 1024 }, 256, &a);
            }
            try x.mm_(x.act, inter, M, l.down, x.a);
            try x.conv(x.a, 1, l.m_base + 2 * H * 2, x.x, true, M);
        }
        // every row through the head, its 16 best; the selector's projections; the greedy path a stream
        try x.normRows(x.x, x.norm, x.y, M);
        const head = x.draftHead();
        try x.mm_(x.y, H, M, head, x.logits);
        const V = x.vocab;
        const chunks = (V + chunk_cols - 1) / chunk_cols;
        {
            var a: cuda.Args = .{};
            a.add(x.logits);
            a.add(@as(i32, @intCast(V)));
            a.add(x.pv);
            a.add(x.pi);
            try go(x.k.top16_part, x.s, .{ .x = @intCast(M), .y = @intCast(chunks) }, 256, &a);
            var b: cuda.Args = .{};
            b.add(x.pv);
            b.add(x.pi);
            b.add(@as(i32, @intCast(chunks)));
            b.add(x.cand);
            try go(x.k.top16_merge, x.s, .{ .x = @intCast(M) }, 256, &b);
        }
        try x.tk.lane.matmulOut(x.s, x.y, H, M, x.hproj, x.g, .f32, x.part);
        {
            var a: cuda.Args = .{};
            for ([_]u64{ x.ids, x.cand, x.g, x.pred, x.succ, x.walks }) |v| a.add(v);
            a.add(cap);
            a.add(if (keyed) x.keys else @as(u64, 0));
            try go(x.k.select, x.s, .{ .x = @intCast(S) }, 256, &a);
        }
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = self(ptr);
        if (std.c.getenv("TF_INKLING2_DRAFT_TRACE") != null) {
            const T = struct {
                var ms: [max_batch + 1]f64 = @splat(0);
                var n: [max_batch + 1]u64 = @splat(0);
                var calls: u64 = 0;
            };
            try x.s.synchronize();
            var t0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t0);
            try x.holdInner(items);
            try x.s.synchronize();
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            const dt = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
            const b = @min(items.len, max_batch);
            T.ms[b] += dt;
            T.n[b] += 1;
            T.calls += 1;
            if (T.calls % 200 == 0) for (1..max_batch + 1) |i| if (T.n[i] > 0) std.debug.print("[dflash2 holds] size {d}: {d:.2} ms avg over {d}\n", .{ i, T.ms[i] / @as(f64, @floatFromInt(T.n[i])), T.n[i] });
            return;
        }
        return x.holdInner(items);
    }

    fn holdInner(x: *Drafter, items: []const dr.Hold) !void {
        var i: usize = 0;
        while (i < items.len) : (i += max_batch) try x.holdBatch(items[i..@min(items.len, i + max_batch)]);
    }

    fn holdBatch(x: *Drafter, items: []const dr.Hold) !void {
        const S = items.len;
        var ids: [max_rows]i32 = undefined;
        var bpos: [max_rows]i32 = undefined;
        var rings: [max_batch]u64 = undefined;
        var starts: [max_batch]i64 = undefined;
        var froms: [max_batch]i64 = undefined;
        var walks: [max_batch]u64 = undefined;
        var states: [max_batch]*LaneState = undefined;
        var depths: [max_batch]u32 = undefined;
        var keys: [max_batch]DraftKey = @splat(.{ .seed = 0, .inv_t = 0, .pos = 0 });
        var keyed = false;
        for (items, 0..) |item, s| {
            const l = x.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            if (item.lanes != null) return error.TreesNotBuilt;
            if (item.depth == 0 or item.position == 0) return error.DraftDepth;
            const anchor = item.position - 1;
            if (l.end != anchor) return error.DraftContextGap;
            if (anchor + block > std.math.maxInt(i32)) return error.DraftContextGap;
            const sl = &x.slots.items[l.slot];
            for (0..block) |t| {
                ids[s * block + t] = @intCast(if (t == 0) item.pending else mask_token);
                bpos[s * block + t] = @intCast(anchor + t);
            }
            rings[s] = sl.ring;
            starts[s] = @intCast(anchor);
            froms[s] = @intCast(l.from);
            walks[s] = sl.walk;
            states[s] = l;
            depths[s] = @min(item.depth, depth);
            l.anchor = anchor;
            l.fetched = false;
            if (keyedOn()) if (cuda.sample.ruleFor(item.sampling)) |r| {
                keys[s] = .{ .seed = r.seed, .inv_t = r.inv_t, .pos = @intCast(item.position) };
                keyed = true;
            };
        }
        const M = S * block;
        try cuda.copy_source.enqueue(x.d, x.ids, std.mem.sliceAsBytes(ids[0..M]), x.s);
        try cuda.copy_source.enqueue(x.d, x.bpos, std.mem.sliceAsBytes(bpos[0..M]), x.s);
        try cuda.copy_source.enqueue(x.d, x.rings, std.mem.sliceAsBytes(rings[0..S]), x.s);
        try cuda.copy_source.enqueue(x.d, x.starts, std.mem.sliceAsBytes(starts[0..S]), x.s);
        try cuda.copy_source.enqueue(x.d, x.froms, std.mem.sliceAsBytes(froms[0..S]), x.s);
        try cuda.copy_source.enqueue(x.d, x.walks, std.mem.sliceAsBytes(walks[0..S]), x.s);
        if (keyed) try cuda.copy_source.enqueue(x.d, x.keys, std.mem.sliceAsBytes(keys[0..S]), x.s);
        try x.pass(S, keyed);
        for (states[0..S], depths[0..S]) |l, n| l.held = n;
        x.passes += 1;
        if (std.c.getenv("TF_INKLING2_DFLASH2_LOG") != null) {
            try x.s.synchronize();
            for (0..S) |s| {
                var w: [block]i32 = undefined;
                try cuda.memory.download(x.d, walks[s], std.mem.sliceAsBytes(&w));
                std.log.info("dflash2 pass {d}: stream {d} anchor {d} at {d} drafts {any}", .{ x.passes, s, w[0], starts[s], w[1..] });
            }
        }
    }
};

fn bf32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// Whether `dir` holds a DFlash2 draft model (a config.json with a dflash_config), not a target checkpoint.
pub fn isDFlash2(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    if (std.mem.endsWith(u8, dir, ".safetensors")) return false;
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return false;
    defer gpa.free(text);
    const p = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return false;
    defer p.deinit();
    if (p.value != .object) return false;
    return p.value.object.get("dflash_config") != null;
}

fn int(o: std.json.ObjectMap, key: []const u8) ?i64 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .integer => |i| i,
        else => null,
    };
}

fn num(o: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn is(o: std.json.ObjectMap, key: []const u8, want: i64) bool {
    const v = int(o, key) orelse return false;
    return v == want;
}

fn near(o: std.json.ObjectMap, key: []const u8, want: f32) bool {
    const v = num(o, key) orelse return false;
    return @abs(v - want) <= 1e-6 * @abs(@as(f64, want));
}

/// The draft's config.json must be the DFlash2 head this engine was written for.
fn admit(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return error.DFlash2MissingFile;
    defer gpa.free(text);
    const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer p.deinit();
    if (p.value != .object) return error.UnsupportedDFlash2Config;
    const o = p.value.object;
    const dcv = o.get("dflash_config") orelse return error.UnsupportedDFlash2Config;
    if (dcv != .object) return error.UnsupportedDFlash2Config;
    const dc = dcv.object;
    var ok = is(o, "hidden_size", hidden) and is(o, "num_hidden_layers", layers_n) and is(o, "num_attention_heads", q_heads) and
        is(o, "num_key_value_heads", kv_heads) and is(o, "head_dim", head_dim) and is(o, "intermediate_size", inter) and
        is(o, "sliding_window", window) and near(o, "rms_norm_eps", eps) and
        is(dc, "block_size", block) and is(dc, "mask_token_id", mask_token) and is(dc, "selector_top_k", top) and
        is(dc, "selector_rank", rank) and is(dc, "conv_kernel_size", 2) and is(dc, "conv_group_size", group) and
        near(dc, "final_logit_softcapping", cap);
    if (o.get("rope_parameters")) |rp| {
        if (rp != .object or !near(rp.object, "rope_theta", theta)) ok = false;
    } else if (o.get("rope_theta") != null and !near(o, "rope_theta", theta)) ok = false;
    if (dc.get("selector_softcap_mode")) |m| {
        if (m != .string or !std.mem.eql(u8, m.string, "after_correction")) ok = false;
    }
    if (dc.get("target_layer_ids")) |tv| {
        if (tv != .array or tv.array.items.len != taps.len) {
            ok = false;
        } else for (tv.array.items, taps) |t, w| {
            if (t != .integer or t.integer != w) ok = false;
        }
    } else ok = false;
    if (!ok) return error.UnsupportedDFlash2Config;
}

test "a ring holds a window of every layer's K and V" {
    try std.testing.expectEqual(@as(usize, 5 * 2 * 2048 * 1024 * 2), ring_bytes);
    try std.testing.expectEqual(@as(usize, 576), dyn_width);
    try std.testing.expectEqual(@as(usize, 4 * 2304 * 2), row_bytes);
}

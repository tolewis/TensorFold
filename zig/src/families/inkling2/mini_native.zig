//! Inkling 2 Mini (inkling2_loop) behind the native CUDA registry (native/cuda.zig): one rank, text only, with its
//! DSpark drafter (--drafter: the checkpoint's mtp.safetensors). Images (C6) and --loops are refused by name before
//! any weight is read.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wts = @import("mini_weights.zig");
const mm = @import("mini_model.zig");
const be = @import("mini_backend.zig");
const snap = @import("mini_snap.zig");
const dspark = @import("mini_dspark.zig");
const dflash = @import("mini_dflash.zig");
const mini_vision = @import("mini_vision.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 32768;
pub const default_image_tokens: u32 = mini_vision.default_image_tokens;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
/// --drafter: the checkpoint folder (its mtp.safetensors) or the .safetensors file itself.
pub const takes_drafter = true;
pub const explains_open = true;
/// A kept state is the caches and conv states through its position, nothing past it (no draft head reads on).
pub const snap_lookahead: u32 = 0;

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// `loops` is here to refuse it by name: the pass count is the checkpoint's loop_config.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 4, lanes_fixed: bool = false, room: ?Room = null, loops: ?u32 = null, kv8: bool = false, vision: ?mini_vision.Limits = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Mini,
    vision: ?*mini_vision.Images = null,
};

const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: mm.Kernels, x: be.Draws, s: cuda.Stream, m: mm.Model, b: be.Mini, v: ?*mini_vision.Images = null };

/// Streams to allocate in `room`: as asked, else (auto) the most that fit; a fixed count past them refuses.
fn fit(room: u64, asked: u32, fixed: bool, plan: anytype) !usize {
    const most = be.round_rows;
    if (fixed and asked > most) return error.ParallelPastRound;
    var n = std.math.clamp(@as(usize, asked), 1, most);
    while (try plan.bytes(n) > room) : (n -= 1) {
        if (fixed) return error.ParallelPastMemory;
        if (n == 1) return error.NoStreamFits;
    }
    return n;
}

const Plan = struct {
    own: *const Owned,
    d: *const cuda.Driver,
    context: usize,
    kv8: bool,

    fn bytes(p: Plan, n: usize) !u64 {
        const o = p.own;
        return try mm.Model.openBytes(o.gpa, p.d, o.s, &o.w, &o.k, p.context, n, p.kv8) + be.Mini.openBytes(o.w.config.text.unpadded_vocab, o.w.config.text.vocab, n);
    }
};

/// What open refuses before any weight is read: --loops and captured kernels (--drafter: the host, drafter_refusal).
/// `drafts` is the server's default on: with no head it drafts nothing (facts: mtp and copies off).
fn refuse(kernels: ?[]const u8, o: Options) !void {
    if (o.loops != null) return error.Inkling2MiniLoopsFixed;
    if (kernels != null) return error.CapturedKernels;
    if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound;
}

/// The checkpoint, the kernels and every stream's caches.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    try refuse(kernels, o);
    const context = try be.capacity(o.context);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.w = try wts.load(gpa, io, ctx.d, dir);
    errdefer own.w.deinit();
    if (context > own.w.config.text.max_context) return error.PastContext;
    own.k = try mm.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.x = try be.Draws.load(ctx.d);
    errdefer own.x.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    const plan: Plan = .{ .own = own, .d = ctx.d, .context = context, .kv8 = o.kv8 };
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const n = fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
        if (e == error.ParallelPastMemory) {
            const most = fit(room, o.lanes, false, plan) catch 0;
            std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window, and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, gib(room), most, if (most == 1) "" else "s" });
        }
        return e;
    };
    own.m = try mm.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, n, o.kv8);
    errdefer own.m.deinit();
    own.b = try be.Mini.init(gpa, ctx.d, &own.m, &own.k, &own.x, own.s, own.w.config.image_token);
    errdefer own.b.deinit();
    own.v = if (o.vision) |lim| try mini_vision.Images.init(gpa, io, ctx.d, dir, own.w.config, lim, mm.prompt_rows, n) else null;
    own.m.vision = own.v;
    own.b.vision = own.v;
    if (own.v) |vis| std.log.info("Inkling 2 Mini vision: {d} layers, {d:.2} GiB weights, up to {d} image tokens / {d} images", .{ vis.c.layers, gib(vis.w.bytes), vis.limits.image_tokens, vis.limits.max_images });
    std.log.info("Inkling 2 Mini: {d} visits a token ({d} sliding, {d} full, {d} conv slots a stream), {d} streams of {d} positions ({s} caches), {d:.2} GiB of weights", .{ own.w.plan.visits.len, own.w.plan.visits.count(.sliding), own.w.plan.visits.count(.full), own.w.plan.visits.count(.conv), n, context, if (o.kv8) "fp8" else "bf16", gib(own.w.bytes) });
    return .{
        .backend = own.b.backend(),
        .facts = own.b.facts(),
        .rows = be.verify_rows,
        .stream_bytes = own.m.streamBytes(),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
        .target = &own.b,
        .vision = own.v,
    };
}

/// The drafter of `dir` on the target's stream, reading its tapped rows (lanes' Drafted wrapper): a DFlash2 draft
/// model's folder (config.json with a dflash_config, incoai/Inkling-2-Mini-DFlash2), else the checkpoint's DSpark head.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void } {
    if (bits != 0 and bits != 4) return error.DSparkBits;
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.b.lanes.count() != 0) return error.TapsAfterStreams;
    if (dflash.isDFlash2(gpa, io, dir)) {
        const d = try dflash.Drafter.init(gpa, io, own.m.d, own.s, &own.w, &own.k, own.m.context, dir);
        std.log.info("Inkling 2 Mini DFlash2: {d} drafts a block from {s} (Apache-2.0), {s} head, chances {s}, {d:.2} GiB fixed, {d:.2} GiB a stream", .{ dflash.depth, dir, if (d.head) |h| (if (h.mode == .fp4) "nvfp4" else "fp8") else "bf16", if (d.chances) "lone streams" else "off", gib(d.fixedBytes()), gib(dflash.laneBytes()) });
        return .{ .drafter = d.drafter(), .ctx = d, .close = closeDFlash, .reserve = reserveDFlash };
    }
    const d = try dspark.Drafter.init(gpa, io, own.m.d, own.s, &own.w, &own.k, own.m.context, dir);
    std.log.info("Inkling 2 Mini DSpark: {d} drafts a block from {s}, bf16 weights (--drafter-bits {d} serves bf16), {d:.2} GiB fixed, {d:.2} GiB a stream", .{ dspark.depth, dir, bits, gib(d.fixedBytes()), gib(dspark.laneBytes(own.m.context)) });
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveDrafter };
}

fn closeDrafter(p: *anyopaque) void {
    const d: *dspark.Drafter = @ptrCast(@alignCast(p));
    d.deinit();
}

fn reserveDrafter(p: *anyopaque, streams: u32) anyerror!void {
    const d: *dspark.Drafter = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeDFlash(p: *anyopaque) void {
    const d: *dflash.Drafter = @ptrCast(@alignCast(p));
    d.deinit();
}

fn reserveDFlash(p: *anyopaque, streams: u32) anyerror!void {
    const d: *dflash.Drafter = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn mini(ptr: *anyopaque) *be.Mini {
    return @ptrCast(@alignCast(ptr));
}

/// Bytes a state at `at` prompt tokens takes.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return snap.bytes(mini(ptr).m, at);
}

/// The prompt pass stands at `at` (a chunk start or a mark): copy the stream's state there.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try mini(ptr).saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return mini(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    mini(ptr).dropState(@ptrCast(@alignCast(saved)));
}

/// Whether a request's caches could start now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(p));
    return own.b.admits(prompt, max_tokens);
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Inkling2MiniLoopsFixed => "--loops: Inkling 2 Mini runs the passes its loop_config names (3 over layers 6 to 17); serve without --loops",
        error.Inkling2MiniImages => "images: serve with --vision to enable Inkling 2 Mini's vision tower, or send text parts only",
        error.Inkling2VisionLimits => "--vision: --vision-image-tokens and --vision-max-images must both be positive",
        error.UnsupportedInkling2VisionConfig => "--vision: unsupported vision_config layout (expected patch 20, merge 2, head_dim 64)",
        error.NoVisionWeights, error.MissingVisionTensor, error.UnsupportedVisionDType => "--vision: checkpoint vision weights (model.visual.*) are missing or not bf16",
        error.TooManyImageTokens => "images: request exceeds --vision-image-tokens",
        error.BadImageGrid => "images: image grid does not match patch count",
        error.Inkling2MiniNoRollback => "Inkling 2 Mini on CUDA rolls a window back only right after its verify (another forward ran in between)",
        error.Inkling2KeepPath => "Inkling 2 Mini on CUDA keeps a prefix of a verified window: a kept path is empty or longer than the window",
        error.Inkling2TapsNotHeld => "the DSpark drafter asked for tapped rows the target no longer holds (they are kept only until the stream's next forward)",
        error.UnsupportedInkling2Taps => "a drafter asked Inkling 2 Mini for taps other than its DSpark head's (visits 5, 23, 41 and the final norm, 48)",
        error.UnpreparedInkling2Taps => "Inkling 2 Mini's tapped rows were read before the drafter attached",
        error.Inkling2NotLooped => "Inkling 2 Small (model_type inkling2) has no CUDA engine yet: this one serves inkling2_loop",
        error.Inkling2WeightMap => "the checkpoint's tensors differ from what its config.json implies (the log names each one)",
        error.Inkling2Nvfp4Denominator => "an NVFP4 expert's outer_scale_denominator is not 2688, the one the checkpoint's factors are checked against",
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Inkling 2 Mini drafts with its DSpark head: serve with --drafter <checkpoint folder>",
        error.DSparkBits => "--drafter-bits: Inkling 2 Mini's DSpark head serves its bf16 weights (0 or 4 both load them as stored)",
        error.DSparkMissingFile => "--drafter: no mtp.safetensors in that folder; name the checkpoint folder or the .safetensors file",
        error.DSparkMissingTensor, error.DSparkTensorShape => "--drafter: that mtp.safetensors is not Inkling 2 Mini's DSpark head (the log names the tensor)",
        error.DSparkNotConfigured, error.UnsupportedDSparkConfig => "--drafter: the checkpoint's mtp_config is not the DSpark head this engine serves (3 layers, block 8, rank 256, taps 5/23/41/48)",
        error.TapsAfterStreams => "the drafter attaches before any stream opens",
        error.DFlash2MissingFile => "--drafter: that folder has no model.safetensors or config.json of a DFlash2 draft model",
        error.DFlash2MissingTensor, error.DFlash2TensorShape => "--drafter: that model.safetensors is not Inkling 2 Mini's DFlash2 draft (the log names the tensor)",
        error.UnsupportedDFlash2Config => "--drafter: that config.json is not the DFlash2 draft this engine serves (incoai/Inkling-2-Mini-DFlash2: 5 layers, block 8, taps 5/23/41/48)",
        error.DFlash2WeightRange => "--drafter: a DFlash2 weight has no finite nonzero range for its FP8 copy (TF_INKLING2_DRAFT_LAYERS8=0 serves it in BF16)",
        error.StructuresUnsupported => "structured output is not built for Inkling 2 Mini on CUDA yet",
        error.NoAllowedToken => "structured output: the grammar allowed no token with a usable score at this step",
        error.CapturedKernels => "Inkling 2 Mini runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BuiltWithoutKernels => "this build carries no CUDA kernel images: build with -Dnvcc",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: Inkling 2 Mini on CUDA holds at most {d} streams (a round's rows)", .{be.round_rows}),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer Inkling 2 Mini streams than asked (each stream's 48 visit slots are allocated at load): lower --parallel or --context, or serve --parallel auto",
        error.NoStreamFits => "the CUDA memory budget fits no Inkling 2 Mini stream after the weights: lower --context or free device memory",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.v) |v| v.deinit();
    own.b.deinit();
    own.m.deinit();
    own.s.deinit();
    own.x.deinit();
    own.k.deinit();
    own.w.deinit();
    own.gpa.destroy(own);
}

test "Inkling 2 Mini open: --loops and captured kernels are refused by name before any load" {
    const base: Options = .{ .context = 4096, .drafts = false };
    try refuse(null, base);
    var o = base;
    o.loops = 3;
    try std.testing.expectError(error.Inkling2MiniLoopsFixed, refuse(null, o));
    o = base;
    o.drafts = true; // the server's default: nothing to draft with, not a refusal
    try refuse(null, o);
    try std.testing.expectError(error.CapturedKernels, refuse("dir", base));
    // open refuses before it allocates: a failing allocator sees no request
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var d: cuda.Driver = undefined;
    const ctx: cuda.Context = .{ .d = &d, .handle = undefined, .device = 0 };
    o = base;
    o.loops = 2;
    try std.testing.expectError(error.Inkling2MiniLoopsFixed, open(failing.allocator(), std.testing.io, &ctx, "/nonexistent", null, o));
    try std.testing.expectEqual(@as(usize, 0), failing.allocations);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.Inkling2MiniLoopsFixed, error.Inkling2MiniImages, error.Inkling2MiniNoRollback, error.Inkling2NotLooped, error.Inkling2WeightMap, error.Inkling2Nvfp4Denominator, error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.Inkling2KeepPath, error.Inkling2TapsNotHeld, error.UnsupportedInkling2Taps, error.UnpreparedInkling2Taps, error.TapsAfterStreams, error.DFlash2MissingFile, error.UnsupportedDFlash2Config, error.DFlash2TensorShape, error.DSparkBits, error.DSparkMissingFile, error.DSparkTensorShape, error.UnsupportedDSparkConfig, error.StructuresUnsupported, error.NoAllowedToken, error.CapturedKernels, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

/// A synthetic plan: `stream` bytes a stream plus `fixed` bytes of scratch.
const Fake = struct {
    stream: u64,
    fixed: u64,
    fn bytes(f: Fake, n: usize) !u64 {
        return f.fixed + n * f.stream;
    }
};

test "the streams allocated at open fit the room: auto takes what fits, a fixed count refuses" {
    const g: u64 = 1 << 30;
    const one: Fake = .{ .stream = g, .fixed = 0 };
    try std.testing.expectEqual(@as(usize, 2), try fit(2 * g, 8, false, one));
    try std.testing.expectError(error.ParallelPastMemory, fit(2 * g, 8, true, one));
    try std.testing.expectError(error.NoStreamFits, fit(g - 1, 8, false, one));
    try std.testing.expectError(error.ParallelPastRound, fit(1024 * g, 65, true, one));
}

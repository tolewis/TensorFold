//! ModelOpt checkpoints of the Qwen3.5 hybrids (nvidia/Qwen3.8-27B-NVFP4): NVFP4, FP8 and bf16 projections on qlinear.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Geometry = geometry.Geometry;
const Kind = geometry.Kind;
const Kinds = geometry.Kinds;
const Slot = geometry.Slot;
const Engine = engine.Engine;
const Proj = engine.Proj;
const Weight = cuda.qlinear.Weight;
const DeviceBuffer = cuda.DeviceBuffer;
const aligned = engine.aligned;
const padded = cuda.modelopt.padded;

/// The language model's tensors under this prefix, the head at `lm_head` (HF's spelling, not MLX's).
pub const prefix = "model.language_model.";

/// A module's recipe as this engine serves it: NVFP4 (W4A16 too), per-tensor FP8, or bf16 where none is declared.
fn kindOf(r: ?core.modelopt.Recipe) !Kind {
    const recipe = r orelse return .bf16;
    return switch (recipe) {
        .nvfp4 => .nvfp4,
        .w4a16_nvfp4 => .nvfp4_w16,
        .fp8 => .fp8,
        else => error.UnsupportedQwenQuantization,
    };
}

/// A single-format NVFP4 config's `ignore` (or `exclude_modules`) names: modules, or prefixes ending in `*`.
const Ignored = struct {
    doc: std.json.Parsed(std.json.Value),
    names: []const std.json.Value,

    fn read(gpa: std.mem.Allocator, config: []const u8) !Ignored {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, config, .{});
        errdefer doc.deinit();
        const q = if (doc.value.object.get("quantization_config")) |v| v.object else return error.UnsupportedQwenQuantization;
        const list = q.get("ignore") orelse q.get("exclude_modules") orelse return .{ .doc = doc, .names = &.{} };
        if (list != .array) return error.UnsupportedQwenQuantization;
        return .{ .doc = doc, .names = list.array.items };
    }

    fn has(i: Ignored, module: []const u8) bool {
        for (i.names) |v| if (v == .string) {
            const n = v.string;
            if (std.mem.endsWith(u8, n, "*") and std.mem.startsWith(u8, module, n[0 .. n.len - 1])) return true;
            if (std.mem.eql(u8, n, module)) return true;
        };
        return false;
    }
};

/// Whether layer `i` of `g` has `slot`: a dense MLP, linear or full attention.
fn slotUsed(g: Geometry, i: usize, slot: Slot) bool {
    return switch (slot) {
        .gate, .up, .down => !g.moe(),
        .qkv, .z, .a, .b, .out => g.linear[i],
        .q, .k, .v, .o => !g.linear[i],
    };
}

/// A plain BF16 checkpoint's kinds: every module and the head bf16 (a tied head packs embed_tokens), on HF names.
pub fn plain(g: Geometry) !Kinds {
    if (g.moe()) return error.UnsupportedQwenQuantization;
    var out: Kinds = .{ .modelopt = true, .head = .bf16 };
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        if (slotUsed(g, i, slot)) out.proj[i][@backingInt(slot)] = .bf16;
    };
    return out;
}

/// Every module's kind: a MIXED_PRECISION config's quantized_layers, or a single-format NVFP4 one's `ignore` list.
pub fn kinds(gpa: std.mem.Allocator, config: []const u8, g: Geometry) !Kinds {
    if (try compressedTensors(gpa, config)) return .{ .modelopt = true, .ct = true }; // filled from the index
    var why: core.modelopt.Why = .{};
    var r = core.modelopt.admitRecipes(gpa, config, &why) catch |err| {
        std.log.err("{s}", .{why.text()});
        return if (err == error.OutOfMemory) err else error.UnsupportedQwenQuantization;
    };
    defer r.deinit();
    if (g.tied) return error.UnsupportedQwenQuantization;
    if (r.modules.count() == 0) return single(gpa, config, g);
    if (r.of(prefix ++ "embed_tokens") != null) return error.UnsupportedQwenQuantization; // the rows stay bf16
    var out: Kinds = .{ .modelopt = true, .head = try kindOf(r.of("lm_head")), .experts = if (g.moe()) try moeKinds(&r, g) else .affine };
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        if (slotUsed(g, i, slot)) out.proj[i][@backingInt(slot)] = try kindOf(r.of(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) })));
    };
    return out;
}

/// A MoE layer's module names under `prefix`: experts 0..experts-1 then the shared one, each projection; the router's.
const Names = struct {
    buf: [160]u8 = undefined,

    fn expert(n: *Names, layer: usize, x: usize, g: Geometry, proj: []const u8) ![]const u8 {
        if (x < g.experts) return std.fmt.bufPrint(&n.buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ layer, x, proj });
        return std.fmt.bufPrint(&n.buf, prefix ++ "layers.{d}.mlp.shared_expert.{s}", .{ layer, proj });
    }

    fn router(n: *Names, layer: usize, which: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&n.buf, prefix ++ "layers.{d}.mlp.{s}", .{ layer, which });
    }
};

const expert_projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
const router_rows = [_][]const u8{ "gate", "shared_expert_gate" };

/// The routed and shared experts' one NVFP4 recipe, read for every projection (a child entry overrides its parent).
fn moeKinds(r: *const core.modelopt.Recipes, g: Geometry) !Kind {
    var first: ?core.modelopt.Recipe = null;
    var n: Names = .{};
    for (0..g.layers) |i| {
        for (0..g.experts + 1) |x| for (expert_projs) |proj| {
            const got = r.of(try n.expert(i, x, g, proj)) orelse return error.UnsupportedQwenQuantization;
            if (got != .nvfp4 and got != .w4a16_nvfp4) return error.UnsupportedQwenQuantization;
            if (first != null and first.? != got) return error.UnsupportedQwenQuantization;
            first = got;
        };
        for (router_rows) |w| if (r.of(try n.router(i, w)) != null) return error.UnsupportedQwenQuantization; // bf16 rows
    }
    return kindOf(first orelse return error.UnsupportedQwenQuantization);
}

/// A single-format NVFP4 MoE: no expert projection ignored (they are one table), the router's rows ignored (bf16).
fn moeIgnored(ig: Ignored, g: Geometry) !void {
    var n: Names = .{};
    for (0..g.layers) |i| {
        for (0..g.experts + 1) |x| for (expert_projs) |proj| if (ig.has(try n.expert(i, x, g, proj))) return error.UnsupportedQwenQuantization;
        for (router_rows) |w| if (!ig.has(try n.router(i, w))) return error.UnsupportedQwenQuantization;
    }
}

/// Whether quantization_config is compressed-tensors' (unsloth's export): its kinds are read from the tensors.
fn compressedTensors(gpa: std.mem.Allocator, config: []const u8) !bool {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, config, .{});
    defer doc.deinit();
    const q = (doc.value.object.get("quantization_config") orelse return false);
    if (q != .object) return false;
    const m = q.object.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "compressed-tensors");
}

/// A compressed-tensors export's kinds by its index: `weight_packed` NVFP4, a `weight_scale` FP8 (a row's), else bf16.
pub fn ctKinds(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, g: Geometry) !Kinds {
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);
    return ctKindsBytes(gpa, bytes, g);
}

/// The same index admission without filesystem allocation, for deterministic OOM coverage.
pub fn ctKindsBytes(gpa: std.mem.Allocator, bytes: []const u8, g: Geometry) !Kinds {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.UnsupportedQwenQuantization;
    const value = doc.value.object.get("weight_map") orelse return error.UnsupportedQwenQuantization;
    if (value != .object) return error.UnsupportedQwenQuantization;
    const map = value.object;
    const K = struct {
        fn of(m: std.json.ObjectMap, module: []const u8) !Kind {
            var b: [192]u8 = undefined;
            if (m.contains(try std.fmt.bufPrint(&b, "{s}.weight_packed", .{module}))) return .nvfp4;
            if (m.contains(try std.fmt.bufPrint(&b, "{s}.weight_scale", .{module}))) return .fp8c;
            return .bf16;
        }
    };
    if (try K.of(map, prefix ++ "embed_tokens") != .bf16) return error.UnsupportedQwenQuantization;
    var out: Kinds = .{ .modelopt = true, .ct = true, .head = try K.of(map, "lm_head") };
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        const used = switch (slot) {
            .gate, .up, .down => true,
            .qkv, .z, .a, .b, .out => g.linear[i],
            .q, .k, .v, .o => !g.linear[i],
        };
        if (used) out.proj[i][@backingInt(slot)] = try K.of(map, try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) }));
    };
    return out;
}

/// quant_algo NVFP4 throughout (Inferact's export): every linear NVFP4 but those `ignore` names, which stay bf16.
fn single(gpa: std.mem.Allocator, config: []const u8, g: Geometry) !Kinds {
    const ig = try Ignored.read(gpa, config);
    defer ig.doc.deinit();
    if (!ig.has(prefix ++ "embed_tokens")) return error.UnsupportedQwenQuantization;
    if (g.moe()) try moeIgnored(ig, g);
    var out: Kinds = .{ .modelopt = true, .head = if (ig.has("lm_head")) .bf16 else .nvfp4, .experts = if (g.moe()) .nvfp4 else .affine };
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| for (std.enums.values(Slot)) |slot| {
        const used = switch (slot) {
            .gate, .up, .down => true,
            .qkv, .z, .a, .b, .out => g.linear[i],
            .q, .k, .v, .o => !g.linear[i],
        };
        const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, geometry.slot_modules.get(slot) });
        if (used) out.proj[i][@backingInt(slot)] = if (ig.has(name)) .bf16 else .nvfp4;
    };
    return out;
}

/// The device bytes of an n-by-k projection of `kind` in the lane's layout (rows padded to 128).
pub fn packedBytes(n: usize, k: usize, kind: Kind) usize {
    const np = padded(n);
    return switch (kind) {
        .nvfp4, .nvfp4_w16 => aligned(np * k / 2) + aligned(np * k / 16),
        .fp8 => aligned(np * k),
        .fp8c => aligned(np * k) + aligned(np * (k / 64) * 4),
        .bf16 => aligned(np * k * 2),
        .affine => unreachable,
    };
}

/// The checkpoint's bytes of an n-by-k projection of `kind`: codes, then block scales.
pub fn rawBytes(n: usize, k: usize, kind: Kind) usize {
    return switch (kind) {
        .nvfp4, .nvfp4_w16 => n * k / 2 + n * k / 16,
        .fp8, .fp8c => n * k,
        .bf16 => n * k * 2,
        .affine => unreachable,
    };
}

/// FP8 and NVFP4 weights decode to bf16 for mma.sync (sm_80 up), as the Nemotron lane does.
pub const modelopt_min: u32 = 80;

/// Why a ModelOpt checkpoint cannot load at `capability` (10 * major + minor), or null when it may.
pub fn modeloptRefusal(capability: u32) ?[]const u8 {
    if (capability >= modelopt_min) return null;
    return "FP8 and NVFP4 weights decode to bf16 for mma.sync: " ++
        "a ModelOpt checkpoint needs compute capability 8.0 or newer";
}

/// The kernels a ModelOpt checkpoint adds: the lane matmul, the prompt GEMM, the packers and the bf16 token rows.
pub const Kernels = struct {
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    pack: cuda.modelopt.Packer,
    embed: cuda.embed_rows.EmbedRows,
    /// Slice partials where no cluster meets them (before sm_90): the most any projection takes below fused_rows.
    part: ?DeviceBuffer = null,
    /// The checkpoint's own math for compressed-tensors FP8 weights: e4m3 rows under a per-row activation
    /// scale, times the weight's row scale. Prompt rows only.
    ck: ?cuda.ck.Ck = null,
    ck_rows: ?DeviceBuffer = null, // e4m3 rows: max_rows * the widest k a row-scale weight takes
    ck_sa: ?DeviceBuffer = null, // per-row activation scales [max_rows] f32
    experts: ?cuda.experts.Nvfp4Experts = null, // a MoE's routed experts (cuda_modelopt_experts.zig)

    pub fn load(d: *const cuda.Driver, capability: u32, sms: usize, moe: bool) !Kernels {
        if (modeloptRefusal(capability) != null) return error.ModeloptNeedsSm80;
        var lane = try cuda.qmmf.Lane.load(d, capability / 10);
        errdefer lane.unload();
        var prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer prompt.unload();
        var pack = try cuda.modelopt.Packer.load(d);
        errdefer pack.unload();
        var experts: ?cuda.experts.Nvfp4Experts = if (moe) try cuda.experts.Nvfp4Experts.load(d, sms) else null;
        errdefer if (experts) |*x| x.unload();
        var ckm: ?cuda.ck.Ck = null;
        errdefer if (ckm) |*x| x.unload();
        if (capability >= 89) ckm = try cuda.ck.Ck.load(d, capability / 10, sms);
        return .{ .lane = lane, .prompt = prompt, .pack = pack, .embed = try cuda.embed_rows.EmbedRows.load(d), .experts = experts, .ck = ckm };
    }

    pub fn unload(k: *Kernels) void {
        if (k.ck_rows) |*b| b.free();
        if (k.ck_sa) |*b| b.free();
        if (k.ck) |*x| x.unload();
        if (k.part) |*p| p.free();
        if (k.experts) |*x| x.unload();
        k.embed.unload();
        k.pack.unload();
        k.prompt.unload();
        k.lane.unload();
    }

    pub fn linear(k: *const Kernels) cuda.qlinear.Linear {
        return .{
            .ck8 = envOn("TF_CK8"),
            .ck4 = envOn("TF_CK4"),
            .lane = &k.lane,
            .nvfp4_prompt = &k.prompt,
            .ck = if (k.ck) |*c| c else null,
            .ck_rows = if (k.ck_rows) |b| b.ptr else 0,
            .ck_sa = if (k.ck_sa) |b| b.ptr else 0,
            .ck_part = if (k.part) |p| p.ptr else 0,
        };
    }

    /// The slice partials every projection of `e` takes below fused_rows (none with clusters).
    pub fn reserveParts(k: *Kernels, e: *const Engine) !void {
        var most: usize = 0;
        const Each = struct {
            fn bytes(lane: cuda.qmmf.Lane, p: Proj) usize {
                return switch (p) {
                    .ql => |w| switch (w) {
                        .nvfp4, .fp8, .bf16, .fp8g => |q| lane.partBytes(@min(engine.max_rows, cuda.qmmf.fused_rows - 1), q),
                        else => 0,
                    },
                    else => 0,
                };
            }
        };
        most = Each.bytes(k.lane, e.head);
        for (e.layers) |l| {
            switch (l.mlp) {
                .dense => |m| for ([_]Proj{ m.gate, m.up, m.down }) |p| {
                    most = @max(most, Each.bytes(k.lane, p));
                },
                .moe => {},
            }
            switch (l.mixer) {
                .delta => |m| for ([_]Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| {
                    most = @max(most, Each.bytes(k.lane, p));
                },
                .attention => |m| for ([_]Proj{ m.q, m.k, m.v, m.o }) |p| {
                    most = @max(most, Each.bytes(k.lane, p));
                },
            }
        }
        if (most > 0) k.part = try DeviceBuffer.alloc(e.d, most);
        if (k.ck != null) {
            var widest: usize = 0;
            const Widest = struct {
                fn of(p: Proj) usize {
                    return switch (p) {
                        .ql => |w| switch (w) {
                            .fp8g => |q| q.k,
                            else => 0,
                        },
                        else => 0,
                    };
                }
            };
            widest = Widest.of(e.head);
            for (e.layers) |l| {
                switch (l.mlp) {
                    .dense => |m| for ([_]Proj{ m.gate, m.up, m.down }) |p| {
                        widest = @max(widest, Widest.of(p));
                    },
                    .moe => {},
                }
                switch (l.mixer) {
                    .delta => |m| for ([_]Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| {
                        widest = @max(widest, Widest.of(p));
                    },
                    .attention => |m| for ([_]Proj{ m.q, m.k, m.v, m.o }) |p| {
                        widest = @max(widest, Widest.of(p));
                    },
                }
            }
            if (widest > 0) {
                k.ck_rows = try DeviceBuffer.alloc(e.d, engine.max_rows * widest);
                k.ck_sa = try DeviceBuffer.alloc(e.d, engine.max_rows * 4);
            }
        }
    }
};

/// `name` unset or anything but "0": on. A prompt path behind one of these can be A/B'd with a restart.
fn envOn(comptime name: [:0]const u8) bool {
    const v = std.c.getenv(name) orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

var ck_checked = false;
var ck8_checked = false;

/// A checkpoint-math prompt GEMM's first call, both ways on the same rows: logs the largest difference and keeps
/// the checkpoint-math result. An order slip reads near the full range, a scale slip a few percent.
fn checkCk(e: *Engine, l: cuda.qlinear.Linear, w: Weight, in: cuda.qlinear.Rows, out: u64, rows: usize) !void {
    const n = w.outputs();
    const bytes = rows * n * 2;
    var a = try DeviceBuffer.alloc(e.d, bytes);
    defer a.free();
    var b = try DeviceBuffer.alloc(e.d, bytes);
    defer b.free();
    var tile = l;
    tile.ck = null; // the prompt tile on bf16 rows
    try tile.prompt(e.stream, w, in, a.ptr, rows);
    try l.prompt(e.stream, w, in, b.ptr, rows);
    try e.stream.synchronize();
    const host_a = try e.gpa.alloc(u16, rows * n);
    defer e.gpa.free(host_a);
    const host_b = try e.gpa.alloc(u16, rows * n);
    defer e.gpa.free(host_b);
    try a.download(0, std.mem.sliceAsBytes(host_a));
    try b.download(0, std.mem.sliceAsBytes(host_b));
    var diff: f32 = 0;
    var most: f32 = 0;
    var aa: f64 = 0; // sum a^2, sum (a - b)^2 and sum a * b: the relative L2 error and the cosine
    var dd: f64 = 0;
    var ab: f64 = 0;
    var bb: f64 = 0;
    for (host_a, host_b) |ua, ub| {
        const va: f32 = @bitCast(@as(u32, ua) << 16);
        const vb: f32 = @bitCast(@as(u32, ub) << 16);
        diff = @max(diff, @abs(va - vb));
        most = @max(most, @abs(va));
        aa += @as(f64, va) * va;
        dd += @as(f64, va - vb) * (va - vb);
        ab += @as(f64, va) * vb;
        bb += @as(f64, vb) * vb;
    }
    const rel_l2 = if (aa > 0) @sqrt(dd / aa) else 0;
    const cos = if (aa > 0 and bb > 0) ab / @sqrt(aa * bb) else 0;
    std.log.info("qwen3_5: {s} prompt GEMM n={d} k={d} rows={d}: largest |diff| {e:.3} against a max |value| {e:.3}; rel_l2 {e:.3} cosine {d:.4}", .{ if (w == .nvfp4) "NVFP4" else "FP8", n, w.inputs(), rows, diff, most, rel_l2, cos });
    try e.d.check(e.d.api.cuMemcpyDtoD_v2(out, b.ptr, bytes), "cuMemcpyDtoD");
}

/// out (rows, n) = x (rows, k): prompt chunks past 16 rows on prompt16, else the lane (a row's bits at any count).
pub fn project(e: *Engine, w: Weight, x: u64, out: u64, rows: usize, prompt: bool) !void {
    const k = if (e.mo) |*m| m else return error.FormatNotLoaded;
    const l = k.linear();
    const in: cuda.qlinear.Rows = .{ .x = x, .part = if (k.part) |p| p.ptr else 0 };
    if (prompt and rows > 16) {
        // once a format, on the first prompt rows: the checkpoint-math GEMM against the tile that takes bf16 rows, so
        // a layout or scale slip shows as a relative error, not as junk tokens (TF_CK4_CHECK=0, TF_CK8_CHECK=0: off)
        const wide = rows >= cuda.ck.prompt_tokens and l.ck != null;
        if (!ck_checked and envOn("TF_CK4_CHECK") and w == .nvfp4 and l.ck4 and wide) {
            ck_checked = true;
            try checkCk(e, l, w, in, out, @min(rows, 256)); // the chunk's first rows: the kernels mask the rest
        }
        if (!ck8_checked and envOn("TF_CK8_CHECK") and w == .fp8g and l.ck8 and wide) {
            ck8_checked = true;
            try checkCk(e, l, w, in, out, @min(rows, 256));
        }
        return l.prompt(e.stream, w, in, out, rows);
    }
    if (rows >= cuda.qmmf.fused_rows) return error.WindowTooWide; // the fused tile sums its slices in another order
    return l.decode(e.stream, w, in, out, rows);
}

/// An fp32 tensor of one element.
pub fn scalar(ck: *core.Checkpoint, name: []const u8) !f32 {
    const t = try ck.get(name);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// `{pre}{module}.{tail}`, a module's tensor.
fn tname(b: []u8, pre: []const u8, module: []const u8, tail: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "{s}{s}.{s}", .{ pre, module, tail });
}

/// `{pre}{name}` as stored, staged and packed into the lane's layout out of `e`'s arena.
pub fn linear(e: *Engine, ck: *core.Checkpoint, stage: *DeviceBuffer, pre: []const u8, name: []const u8, n: usize, k: usize, kind: Kind) !Proj {
    if (k % 64 != 0) return error.UnsupportedQwenGeometry;
    const p = &(e.mo orelse return error.FormatNotLoaded).pack;
    var buf: [192]u8 = undefined;
    const out: Weight = switch (kind) {
        .bf16 => blk: {
            const t = try ck.expect(try tname(&buf, pre, name, "weight"), .bf16, &.{ n, k });
            try stage.upload(0, t.bytes);
            const w = try e.alloc(padded(n) * k * 2);
            try p.bf16Lane(e.stream, stage.ptr, w, n, k);
            break :blk .{ .bf16 = cuda.modelopt.bf16Weight(w, n, k) };
        },
        .fp8 => blk: {
            const t = try ck.expect(try tname(&buf, pre, name, "weight"), .f8_e4m3, &.{ n, k });
            const scale = try scalar(ck, try tname(&buf, pre, name, "weight_scale"));
            try stage.upload(0, t.bytes);
            const w = try e.alloc(padded(n) * k);
            try p.fp8Lane(e.stream, stage.ptr, w, n, k);
            break :blk .{ .fp8 = cuda.modelopt.fp8Weight(w, scale, n, k) };
        },
        .fp8c => blk: { // compressed-tensors FP8, a bf16 scale a row: the FP8G lane, every 64 inputs a row's scale
            const t = try ck.expect(try tname(&buf, pre, name, "weight"), .f8_e4m3, &.{ n, k });
            const rs = try ck.expect(try tname(&buf, pre, name, "weight_scale"), .bf16, &.{ n, 1 });
            try stage.upload(0, t.bytes);
            const w = try e.alloc(padded(n) * k);
            try p.fp8Lane(e.stream, stage.ptr, w, n, k);
            const tiles = try e.gpa.alloc(f32, padded(n) * (k / 64));
            defer e.gpa.free(tiles);
            rowScales(tiles, std.mem.bytesAsSlice(u16, rs.bytes), n, k);
            const sd = try e.alloc(tiles.len * 4);
            try e.arena.upload(sd - e.arena.ptr, std.mem.sliceAsBytes(tiles));
            break :blk .{ .fp8g = cuda.fp8.weight(w, sd, n, k) };
        },
        .nvfp4, .nvfp4_w16 => blk: {
            // ModelOpt: weight, weight_scale_2 (multiplies); compressed-tensors: weight_packed, weight_global_scale
            const ct = ck.has(try tname(&buf, pre, name, "weight_packed"));
            const codes = try ck.expect(try tname(&buf, pre, name, if (ct) "weight_packed" else "weight"), .u8, &.{ n, k / 2 });
            const blocks = try ck.expect(try tname(&buf, pre, name, "weight_scale"), .f8_e4m3, &.{ n, k / 16 });
            const global = if (ct) 1.0 / try scalar(ck, try tname(&buf, pre, name, "weight_global_scale")) else try scalar(ck, try tname(&buf, pre, name, "weight_scale_2"));
            try stage.upload(0, codes.bytes);
            try stage.upload(codes.bytes.len, blocks.bytes);
            const words = try e.alloc(padded(n) * k / 2);
            const tiles = try e.alloc(padded(n) * k / 16);
            try p.fp4Lane(e.stream, stage.ptr, stage.ptr + codes.bytes.len, words, tiles, n, k);
            // the input scale the checkpoint quantizes activations with, for the checkpoint-math prompt GEMM
            const ig = if (ck.has(try tname(&buf, pre, name, "input_global_scale")))
                try scalar(ck, try tname(&buf, pre, name, "input_global_scale"))
            else
                0;
            // `act` is 1 / the input scale: the quantizer multiplies by its inverse (the full e4m3 range for a
            // block's scale), and the epilogue by `act` itself, so the two global scales cancel exactly.
            var q = cuda.nvfp4.weight(words, tiles, global, n, k);
            q.act = if (ig != 0) 1.0 / ig else 0;
            break :blk .{ .nvfp4 = q };
        },
        .affine => unreachable,
    };
    try e.stream.synchronize(); // the next projection's staging overwrites this one's
    return .{ .ql = out };
}

/// A bf16 scale a row as FP8G's tiles [npad/64][k/64][64] of fp32 (fp8.packScales' order), rows past n one.
pub fn rowScales(out: []f32, rows: []align(1) const u16, n: usize, k: usize) void {
    const kg = k / 64;
    for (0..padded(n) / 64) |t| for (0..kg) |g| for (0..64) |c| {
        const row = t * 64 + c;
        out[(t * kg + g) * 64 + c] = if (row < n) @bitCast(@as(u32, rows[row]) << 16) else 1.0;
    };
}

/// Token rows of the bf16 embedding into `out`: an exact copy of each id's row.
pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize) !void {
    const k = if (e.mo) |*m| m else return error.FormatNotLoaded;
    return k.embed.bf16Copies(e.stream, ids, e.embed_w, out, rows, e.g.hidden, 1); // the registered copy kernel
}

test "kinds from nvidia/Qwen3.8-27B-NVFP4's recipes: NVFP4 MLPs and head, FP8 mixers, bf16 a and b" {
    const text =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
        \\   "lm_head":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.0.linear_attn.in_proj_qkv":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.0.linear_attn.in_proj_z":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.0.mlp.gate_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.0.mlp.up_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.0.mlp.down_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.1.self_attn.q_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.self_attn.k_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.self_attn.v_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.self_attn.o_proj":{"quant_algo":"FP8"},
        \\   "model.language_model.layers.1.mlp.gate_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.1.mlp.up_proj":{"quant_algo":"NVFP4","group_size":16},
        \\   "model.language_model.layers.1.mlp.down_proj":{"quant_algo":"NVFP4","group_size":16}}},
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":2,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
        \\ "layer_types":["linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    const g = try geometry.parse(std.testing.allocator, text);
    try std.testing.expect(g.kinds.modelopt);
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.head);
    try std.testing.expectEqual(Kind.fp8, g.kinds.of(0, .qkv));
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .a));
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .b));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(0, .down));
    try std.testing.expectEqual(Kind.fp8, g.kinds.of(1, .o));
    try std.testing.expectEqual(Kind.affine, g.kinds.of(1, .qkv)); // not this layer's mixer
    try std.testing.expectEqual(@as(u8, 16), g.bits.embed); // bf16 rows: learning refuses them
    // a recipe the lane does not serve here (block FP8) is refused by name, as is a quantized embedding
    const pb = try std.mem.replaceOwned(u8, std.testing.allocator, text, "layers.0.linear_attn.out_proj\":{\"quant_algo\":\"FP8\"}", "layers.0.linear_attn.out_proj\":{\"quant_algo\":\"FP8_PB_WO\",\"weight_block_size\":[128,128]}");
    defer std.testing.allocator.free(pb);
    try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(std.testing.allocator, pb));
    try std.testing.expectEqual(@as(usize, 512 * 5120 / 2 + 512 * 5120 / 16), packedBytes(500, 5120, .nvfp4));
    try std.testing.expectEqual(@as(usize, 48 * 5120 * 2), rawBytes(48, 5120, .bf16));
}

test "kinds from a single-format NVFP4 config (Inferact's): NVFP4 but its ignore list, W4A16 recipes weight-only" {
    const head =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","config_groups":{"group_0":
        \\   {"weights":{"num_bits":4,"type":"float","group_size":16},"targets":["Linear"]}},
        \\   "ignore":["lm_head","model.language_model.embed_tokens","model.language_model.layers.0.linear_attn.in_proj_qkv",
        \\     "model.language_model.layers.0.linear_attn.in_proj_a","model.language_model.layers.0.linear_attn.in_proj_b","mtp*"]},
    ;
    const text_config =
        \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":2,"vocab_size":248320,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
        \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
        \\ "layer_types":["linear_attention","full_attention"],
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
    ;
    const g = try geometry.parse(std.testing.allocator, head ++ text_config);
    try std.testing.expectEqual(Kind.bf16, g.kinds.head);
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .qkv));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(0, .z)); // not in the list
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(0, .out));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(1, .q));
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.of(1, .down));
    // an embedding the list leaves out would be NVFP4 rows: refused
    const quantized_rows = try std.mem.replaceOwned(u8, std.testing.allocator, head ++ text_config, "\"model.language_model.embed_tokens\",", "");
    defer std.testing.allocator.free(quantized_rows);
    try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(std.testing.allocator, quantized_rows));
    const w16 =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,
        \\ "quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
        \\   "model.language_model.layers.0.mlp.down_proj":{"quant_algo":"W4A16_NVFP4","group_size":16},
        \\   "model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"FP8"}}},
    ;
    const h = try geometry.parse(std.testing.allocator, w16 ++ text_config);
    try std.testing.expectEqual(Kind.nvfp4_w16, h.kinds.of(0, .down));
    try std.testing.expectEqual(Kind.bf16, h.kinds.head);
}

test "a row's FP8 scale fills its column of every 64-input tile, padding rows one" {
    const n = 130;
    const k = 128;
    var rows: [n]u16 = undefined;
    for (&rows, 0..) |*r, i| r.* = @truncate(0x3f80 + i); // 1.0 plus a few ulps
    var out: [256 * 2]f32 = undefined;
    rowScales(&out, &rows, n, k);
    const bf = struct {
        fn f(h: u16) f32 {
            return @bitCast(@as(u32, h) << 16);
        }
    }.f;
    try std.testing.expectEqual(bf(rows[5]), out[(0 * 2 + 1) * 64 + 5]); // tile 0, group 1, row 5
    try std.testing.expectEqual(bf(rows[129]), out[(2 * 2 + 0) * 64 + 1]); // tile 2: rows 128..
    try std.testing.expectEqual(@as(f32, 1.0), out[(2 * 2 + 1) * 64 + 2]); // row 130: padding
}

test "ModelOpt's floor refuses pre-Ampere cards and admits non-cluster reductions" {
    for ([_]u32{ 70, 75 }) |c| try std.testing.expect(modeloptRefusal(c) != null);
    for ([_]u32{ 80, 86, 87, 89, 90, 100, 120, 121 }) |c| try std.testing.expect(modeloptRefusal(c) == null);
    try std.testing.expectEqualStrings("FP8 and NVFP4 weights decode to bf16 for mma.sync: a ModelOpt checkpoint needs compute capability 8.0 or newer", modeloptRefusal(75).?);
}

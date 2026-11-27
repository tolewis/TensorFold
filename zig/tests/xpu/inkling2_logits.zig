//! Device check of Inkling 2 Mini's whole XPU forward: teacher-forced logits against a reference, and greedy decode.
//!
//!   tf-xpu-inkling2-logits <model dir> --ref <ref.bin> [--ref <ref.bin> ...] [--chunk N] [--decode N]
//!   tf-xpu-inkling2-logits <model dir> --ids <ids file> --generate N [--stop ID]
//!   tf-xpu-inkling2-logits <model dir> --ids <ids file> --resume AT   (a snapshot at AT against one prefill pass)
//!   tf-xpu-inkling2-logits <model dir> --bench-context C [--decode N]
//!
//! `--dense bf16|q8` picks the dense weights' device format (default: the model's, Q8_0; `XPU_DENSE` in the environment
//! does the same) and `--drop-bf16-head` leaves the BF16 head off the device in Q8 mode. `--dump-logits PREFIX` writes every
//! row's full logits (raw f32) of each reference prompt to PREFIX.<reference file name>.f32; `--compare-logits PREFIX`
//! reads those files back and prints, per prompt, KL(dumped || ours) per row (mean, max) and top-1 agreement with the dump.
//! `--dump-from N` writes and reads only the rows at or past N of each prompt (a 125k dump of every row is 100 GB).
//! `XPU_KV_INT8=1` makes every cache of the run int8: one signed byte a value and one fp32 scale a position and K/V head.
//! Unset or any other value keeps bf16.
//!
//! `--digest` prints SHA-256 digests of the bits of every logit row computed (prompt rows, then decode steps), to compare
//! two builds for bit identity. `--profile` adds eight profiled decode steps: launches by kernel, host and device time.
//! `--bench-context C` times N decode steps (default 32) at context C with no prompt: K/V is zeroed, only the cost is real.
//!
//! A reference file holds the token ids of a prompt and, for every position, the reference's top 16 token ids and
//! logits, its log-sum-exp and its top-1 margin (flat little-endian layout of ref_to_bin.py: "IKRF", u32 version 1,
//! u32 rows, u32 top; i32 ids[rows]; i32 top_ids[rows][top]; f32 top_logits[rows][top]; f32 lse[rows]; f32 margin[rows]).
//! The prompt runs in chunks of `--chunk` rows (default and most: 64) with every row's logits; each row is compared:
//! top-1 agreement, every top-1 difference with the reference margin, the largest absolute difference over the
//! reference's top 16 logits. Then `--decode` greedy steps of one row each (default 32) give the decode rate.
//! An ids file is whitespace-separated decimal token ids, or raw little-endian u32 with the extension `.u32`.
const std = @import("std");
const xpu = @import("xpu");
const ink = @import("inkling2_xpu");
const lanes = @import("lanes");

pub const panic = std.debug.FullPanic(xpu.rt.panicDrain);

const state = ink.state;
const vocab = state.vocabulary;
const top_n = 16;

const Ref = struct {
    rows: usize,
    ids: []const u32,
    top_ids: []const i32,
    top_logits: []const f32,
    lse: []const f32,
    margin: []const f32,
};

fn readAll(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const dir = std.fs.path.dirname(path) orelse ".";
    return xpu.loader.readFile(gpa, dir, std.fs.path.basename(path));
}

fn loadRef(gpa: std.mem.Allocator, path: []const u8) !Ref {
    const bytes = try readAll(gpa, path);
    if (bytes.len < 16 or !std.mem.eql(u8, bytes[0..4], "IKRF")) return error.BadReference;
    const version = std.mem.readInt(u32, bytes[4..8], .little);
    const rows: usize = std.mem.readInt(u32, bytes[8..12], .little);
    const top: usize = std.mem.readInt(u32, bytes[12..16], .little);
    if (version != 1 or top != top_n) return error.BadReference;
    const want = 16 + 4 * (rows + rows * top + rows * top + rows + rows);
    if (bytes.len != want) return error.BadReference;
    // the allocation is 1-byte aligned; copy each array out so it is typed and aligned
    var at: usize = 16;
    const ids = try gpa.alloc(u32, rows);
    const top_ids = try gpa.alloc(i32, rows * top);
    const top_logits = try gpa.alloc(f32, rows * top);
    const lse = try gpa.alloc(f32, rows);
    const margin = try gpa.alloc(f32, rows);
    inline for (.{ ids, top_ids, top_logits, lse, margin }) |dst| {
        const n = dst.len * 4;
        @memcpy(std.mem.sliceAsBytes(dst), bytes[at..][0..n]);
        at += n;
    }
    gpa.free(bytes);
    return .{ .rows = rows, .ids = ids, .top_ids = top_ids, .top_logits = top_logits, .lse = lse, .margin = margin };
}

fn loadIds(gpa: std.mem.Allocator, path: []const u8) ![]u32 {
    const bytes = try readAll(gpa, path);
    defer gpa.free(bytes);
    if (std.mem.endsWith(u8, path, ".u32")) {
        const out = try gpa.alloc(u32, bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * 4]);
        return out;
    }
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, bytes, " \t\r\n,");
    while (it.next()) |tok| try out.append(gpa, try std.fmt.parseInt(u32, tok, 10));
    return out.toOwnedSlice(gpa);
}

fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

/// The two largest values of a row: the index of the first and the values.
const Top2 = struct { id: u32, first: f32, second: f32 };
fn top2(row: []const f32) Top2 {
    var a: usize = 0;
    var b: usize = if (row.len > 1) 1 else 0;
    if (row[b] > row[a]) std.mem.swap(usize, &a, &b);
    for (row[2..], 2..) |v, i| {
        if (v > row[a]) {
            b = a;
            a = i;
        } else if (v > row[b]) b = i;
    }
    return .{ .id = @intCast(a), .first = row[a], .second = row[b] };
}

const Stats = struct {
    rows: usize = 0,
    agree: usize = 0,
    max_abs: f64 = 0,
    max_abs_row: usize = 0,
    sum_abs: f64 = 0,
    count_abs: usize = 0,
    /// the largest |ours - reference| of the reference's top-1 logit only
    max_top1_abs: f64 = 0,
    max_diff_margin: f32 = 0,
    nan: usize = 0,
    /// |our log-sum-exp over the vocabulary - the reference's|: sum and max
    lse_sum: f64 = 0,
    lse_max: f64 = 0,
    kl16_sum: f64 = 0,
    /// next-token loss summed over the rows whose next token is in the reference's top 16, ours and the reference's
    nll_ours: f64 = 0,
    nll_ref: f64 = 0,
    nll_n: usize = 0,
};

const Diff = struct { pos: usize, ref_id: i32, our_id: u32, margin: f32, our_gap: f32 };

/// Quantiles of the per-row largest difference and the worst rows.
fn rowSpread(gpa: std.mem.Allocator, row_max: []const f32) !void {
    const sorted = try gpa.dupe(f32, row_max);
    defer gpa.free(sorted);
    std.mem.sort(f32, sorted, {}, std.sort.asc(f32));
    const n = sorted.len;
    std.debug.print("  per-row max difference: median {d:.3}, p90 {d:.3}, p99 {d:.3}, max {d:.3}\n", .{ sorted[n / 2], sorted[n * 9 / 10], sorted[n * 99 / 100], sorted[n - 1] });
    var worst: [3]usize = .{ 0, 0, 0 };
    var count_over: usize = 0;
    for (row_max, 0..) |v, p| {
        if (v > 2) count_over += 1;
        if (v > row_max[worst[0]]) {
            worst[2] = worst[1];
            worst[1] = worst[0];
            worst[0] = p;
        } else if (v > row_max[worst[1]] and p != worst[0]) {
            worst[2] = worst[1];
            worst[1] = p;
        } else if (v > row_max[worst[2]] and p != worst[0] and p != worst[1]) worst[2] = p;
    }
    var bucket_sum = [_]f64{ 0, 0, 0, 0 };
    var bucket_n = [_]usize{ 0, 0, 0, 0 };
    for (row_max, 0..) |v, p| {
        const b: usize = if (p < 64) 0 else if (p < 512) 1 else if (p < 2048) 2 else 3;
        bucket_sum[b] += v;
        bucket_n[b] += 1;
    }
    std.debug.print("  mean per-row max difference by position: <64 {d:.3} (n {d}), <512 {d:.3} (n {d}), <2048 {d:.3} (n {d}), later {d:.3} (n {d})\n", .{
        bucket_sum[0] / @as(f64, @floatFromInt(@max(bucket_n[0], 1))), bucket_n[0], bucket_sum[1] / @as(f64, @floatFromInt(@max(bucket_n[1], 1))), bucket_n[1],
        bucket_sum[2] / @as(f64, @floatFromInt(@max(bucket_n[2], 1))), bucket_n[2], bucket_sum[3] / @as(f64, @floatFromInt(@max(bucket_n[3], 1))), bucket_n[3],
    });
    std.debug.print("  rows over 2.0: {d}; worst rows {d} ({d:.2}), {d} ({d:.2}), {d} ({d:.2})\n", .{ count_over, worst[0], row_max[worst[0]], worst[1], row_max[worst[1]], worst[2], row_max[worst[2]] });
}

fn openDump(gpa: std.mem.Allocator, prefix: []const u8, ref_path: []const u8, mode: [*:0]const u8) !*std.c.FILE {
    const name = try std.fmt.allocPrintSentinel(gpa, "{s}.{s}.f32", .{ prefix, std.fs.path.basename(ref_path) }, 0);
    defer gpa.free(name);
    return std.c.fopen(name, mode) orelse error.DumpOpen;
}

/// KL(p || q) of the softmax of a dumped row `p` against ours `q`, and whether the arg-max agrees, over a prompt's rows.
const KlStats = struct {
    rows: usize = 0,
    agree: usize = 0,
    sum: f64 = 0,
    max: f64 = 0,
    max_row: usize = 0,
    /// The token position of the row with the largest KL (`max_row` counts from the first compared row).
    max_pos: usize = 0,

    fn add(k: *KlStats, p: []const f32, q: []const f32, pos: usize) void {
        const tp = top2(p);
        const tq = top2(q);
        var zp: f64 = 0;
        var zq: f64 = 0;
        for (p, q) |a, b| {
            zp += @exp(@as(f64, a - tp.first));
            zq += @exp(@as(f64, b - tq.first));
        }
        const lp = @log(zp) + @as(f64, tp.first);
        const lq = @log(zq) + @as(f64, tq.first);
        var d: f64 = 0;
        for (p, q) |a, b| {
            const lpi = @as(f64, a) - lp;
            d += @exp(lpi) * (lpi - (@as(f64, b) - lq));
        }
        k.sum += d;
        if (d > k.max) {
            k.max = d;
            k.max_row = k.rows;
            k.max_pos = pos;
        }
        if (std.c.getenv("TF_TP_ROWKL") != null) {
            std.debug.print("row {d} kl {e:.4} top1 {d} dump {d} ours {d} pdump {d:.4} pours {d:.4} pos {d}\n", .{ k.rows, d, @intFromBool(tp.id == tq.id), tp.id, tq.id, @exp(@as(f64, tp.first) - lp), @exp(@as(f64, tq.first) - lq), pos });
        }
        k.rows += 1;
        if (tp.id == tq.id) k.agree += 1;
    }

    fn print(k: KlStats, name: []const u8) void {
        std.debug.print("  vs dump of {s}: KL(dump || ours) mean {e:.4}, max {e:.4} (row {d}, position {d}); top-1 agreement with the dump {d}/{d} ({d:.3} %)\n", .{ name, k.sum / @as(f64, @floatFromInt(@max(k.rows, 1))), k.max, k.max_row, k.max_pos, k.agree, k.rows, 100.0 * @as(f64, @floatFromInt(k.agree)) / @as(f64, @floatFromInt(@max(k.rows, 1))) });
    }
};

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var dir: ?[]const u8 = if (std.c.getenv("TF_XPU_INKLING2")) |v| std.mem.span(v) else null;
    var refs: std.ArrayList([]const u8) = .empty;
    var ids_path: ?[]const u8 = null;
    var generate: usize = 0;
    var decode_steps: usize = 32;
    var chunk: usize = state.max_rows;
    var stop: ?u32 = null;
    var resume_at: usize = 0;
    var embed_host = false;
    var digest = false;
    var profile_steps: usize = 0;
    var profile_all = false;
    var fuse_check = false;
    var router_check = false;
    var verify_bench = false;
    var route_union = false;
    var bench_context: ?usize = null;
    var bench_prefill: usize = 0;
    var options: ink.model.Options = .{ .embed_host = false };
    var tp: u32 = 1;
    var rank: u32 = 0;
    var device: ?u32 = null;
    var dump_prefix: ?[]const u8 = null;
    var compare_prefix: ?[]const u8 = null;
    // Only the rows at or past this one are dumped and compared: a 125k dump of every row is 100 GB.
    var dump_from: usize = 0;
    var probe_pos: std.ArrayList(usize) = .empty;
    var probe_out: ?[]const u8 = null;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--ref")) {
            i += 1;
            try refs.append(gpa, argv[i]);
        } else if (std.mem.eql(u8, a, "--ids")) {
            i += 1;
            ids_path = argv[i];
        } else if (std.mem.eql(u8, a, "--generate")) {
            i += 1;
            generate = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--decode")) {
            i += 1;
            decode_steps = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--chunk")) {
            i += 1;
            chunk = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--resume")) {
            i += 1;
            resume_at = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--stop")) {
            i += 1;
            stop = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--embed-host")) {
            embed_host = true;
        } else if (std.mem.eql(u8, a, "--dense")) {
            i += 1;
            options.dense = std.meta.stringToEnum(ink.model.DenseFormat, argv[i]) orelse return error.BadDense;
        } else if (std.mem.eql(u8, a, "--drop-bf16-head")) {
            options.keep_bf16_head = false;
        } else if (std.mem.eql(u8, a, "--dump-logits")) {
            i += 1;
            dump_prefix = argv[i];
        } else if (std.mem.eql(u8, a, "--compare-logits")) {
            i += 1;
            compare_prefix = argv[i];
        } else if (std.mem.eql(u8, a, "--old-forward")) {
            // the forward of 64-row chunks, every launch a row at a time (the control of an A/B)
            old_forward = true;
        } else if (std.mem.eql(u8, a, "--old-experts")) {
            // the prompt chunk with the decode path's expert kernels (the control of an A/B)
            old_experts = true;
        } else if (std.mem.eql(u8, a, "--old-global")) {
            old_global = true;
        } else if (std.mem.eql(u8, a, "--old-sliding")) {
            old_sliding = true;
        } else if (std.mem.eql(u8, a, "--old-loop")) {
            old_loop = true;
        } else if (std.mem.eql(u8, a, "--old-dense")) {
            // the prompt chunk with the decode path's dense projections (the control of an A/B)
            old_dense = true;
        } else if (std.mem.eql(u8, a, "--digest")) {
            digest = true;
        } else if (std.mem.eql(u8, a, "--profile")) {
            if (profile_steps == 0) profile_steps = 8;
        } else if (std.mem.eql(u8, a, "--profile-steps")) {
            i += 1;
            profile_steps = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--verify-bench")) {
            // the first --ref prompt prefilled, then windows of 1, 2, 3, 4, 6, 8, 12 and 16 real rows run as the server's verify
            // does (every row's logits, the DSpark taps, the cache transaction), timed; then stop
            verify_bench = true;
        } else if (std.mem.eql(u8, a, "--route-union")) {
            // with --verify-bench: also count the distinct experts a window of real rows reads in each visit
            route_union = true;
        } else if (std.mem.eql(u8, a, "--fuse")) {
            ink.fuse.force = true;
        } else if (std.mem.eql(u8, a, "--seg")) {
            ink.fuse.seg = true;
        } else if (std.mem.eql(u8, a, "--no-seg")) {
            ink.fuse.seg = false;
        } else if (std.mem.eql(u8, a, "--attn-splits")) {
            i += 1;
            ink.attention_plan.flash_splits = try std.fmt.parseInt(u32, argv[i], 10);
            ink.attention_plan.flash_user_set = true;
        } else if (std.mem.eql(u8, a, "--fuse-check")) {
            fuse_check = true;
        } else if (std.mem.eql(u8, a, "--router-check")) {
            router_check = true;
        } else if (std.mem.eql(u8, a, "--router-wide")) {
            ink.fuse.router_wide = true;
        } else if (std.mem.eql(u8, a, "--conv-fold")) {
            ink.fuse.conv_fold = true;
        } else if (std.mem.eql(u8, a, "--zeta-wide")) {
            ink.fuse.zeta_wide = true;
        } else if (std.mem.eql(u8, a, "--merge-split")) {
            ink.attention.merge_split = true;
        } else if (std.mem.eql(u8, a, "--pre-zeta")) {
            ink.fuse.pre_zeta = true;
        } else if (std.mem.eql(u8, a, "--pre-q")) {
            ink.fuse.pre_q = true;
        } else if (std.mem.eql(u8, a, "--gates-norms")) {
            ink.fuse.gates_norms = true;
        } else if (std.mem.eql(u8, a, "--zeta-split")) {
            ink.fuse.zeta_split = true;
        } else if (std.mem.eql(u8, a, "--kv-conv")) {
            ink.fuse.kv_conv = true;
        } else if (std.mem.eql(u8, a, "--down-epi")) {
            ink.fuse.down_epi = true;
        } else if (std.mem.eql(u8, a, "--norm-router")) {
            ink.fuse.norm_router = true;
        } else if (std.mem.eql(u8, a, "--role-conv")) {
            ink.fuse.role_conv = true;
        } else if (std.mem.eql(u8, a, "--sink-seg")) {
            ink.fuse.sink_seg = true;
        } else if (std.mem.eql(u8, a, "--profile-all")) {
            // the first --ref prompt prefilled in --chunk row chunks (last row's logits only, as the server does), profiled per
            // chunk; then plain decode profiled greedy and sampled; then stop
            profile_all = true;
            if (profile_steps == 0) profile_steps = 8;
        } else if (std.mem.eql(u8, a, "--bench-prefill")) {
            // the first --ref prompt, repeated to N rows, prefilled in --chunk row chunks with the last row's logits only; cold first token
            i += 1;
            bench_prefill = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--bench-context")) {
            i += 1;
            bench_context = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--tp")) {
            i += 1;
            tp = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--rank")) {
            i += 1;
            rank = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--dump-from")) {
            i += 1;
            dump_from = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--probe-att")) {
            // positions, comma-separated: each visit's attention output row of each goes to --probe-out
            i += 1;
            var it = std.mem.splitScalar(u8, argv[i], ',');
            while (it.next()) |t| try probe_pos.append(gpa, try std.fmt.parseInt(usize, t, 10));
        } else if (std.mem.eql(u8, a, "--skip-below-dump")) {
            // rows below --dump-from make no logits on the host: the KL runs at long context skip their download
            skip_below_dump = true;
        } else if (std.mem.eql(u8, a, "--probe-out")) {
            i += 1;
            probe_out = argv[i];
        } else if (std.mem.eql(u8, a, "--device")) {
            i += 1;
            device = try std.fmt.parseInt(u32, argv[i], 10);
        } else dir = a;
    }
    kv_dtype = state.kvFromEnv();
    std.debug.print("kv {s}\n", .{@tagName(kv_dtype)});
    const path = dir orelse {
        std.debug.print("skipped: no model directory (argument or TF_XPU_INKLING2)\n", .{});
        return;
    };
    if (chunk == 0 or chunk > state.max_prefill_rows) return error.BadChunk;
    if (refs.items.len == 0 and ids_path == null and bench_context == null) return error.NothingToDo;

    // inputs first, so a bad file fails before the model loads
    var loaded_refs: std.ArrayList(Ref) = .empty;
    var longest: usize = 0;
    for (refs.items) |p| {
        const r = try loadRef(gpa, p);
        longest = @max(longest, r.rows + decode_steps);
        try loaded_refs.append(gpa, r);
    }
    var prompt: []u32 = &.{};
    if (ids_path) |p| {
        prompt = try loadIds(gpa, p);
        longest = @max(longest, prompt.len + generate);
    }
    if (bench_context) |c| longest = @max(longest, c + decode_steps + 4 * profile_steps + 8);
    if (bench_prefill > 0) {
        if (loaded_refs.items.len == 0) return error.NothingToDo;
        const src = loaded_refs.items[0].ids;
        const cycled = try gpa.alloc(u32, bench_prefill);
        for (cycled, 0..) |*x, k| x.* = src[k % src.len];
        loaded_refs.items[0].ids = cycled;
        loaded_refs.items[0].rows = bench_prefill;
        longest = @max(longest, bench_prefill + decode_steps + 4 * profile_steps + 8);
    }
    const capacity = std.mem.alignForward(usize, longest, 64);

    xpu.stop.install();
    var drv = xpu.Driver.open() catch {
        std.debug.print("skipped: no Level Zero loader\n", .{});
        return;
    };
    if (device == null) {
        if (std.c.getenv("XPU_DEVICE")) |v| device = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    }
    options.place = try ink.place.Place.parse(tp, rank);
    var ctx = try xpu.Context.init(&drv, device);
    const stream = try xpu.Stream.init(&ctx);
    var r = try xpu.rt.Runtime.init(&ctx, stream);
    std.debug.print("{s}\n", .{ctx.name()});

    const t_load = nowNs(init.io);
    var m = try ink.model.Model.load(gpa, &r, path, blk: {
        options.embed_host = embed_host;
        break :blk options;
    });
    defer m.deinit();
    try m.loadKernels(&ctx);
    if (m.options.place.split()) _ = try ink.group.Group.open(gpa, &r, &ctx, m.options.place);
    std.debug.print("loaded {d:.2} GiB and built the kernels in {d:.1} s\n", .{ @as(f64, @floatFromInt(m.device_bytes)) / (1 << 30), secs(nowNs(init.io) - t_load) });
    if (fuse_check) {
        var scratch = try state.Scratch.init(&r, state.max_rows, 64);
        defer scratch.deinit();
        const bad = try ink.forward.checkPhaseA(m, &scratch, &r);
        std.debug.print("FUSE-A-CHECK mismatches {d}\n", .{bad});
        if (bad != 0) return error.FuseMismatch;
        if (!router_check) return;
    }
    if (router_check) {
        var scratch_r = try state.Scratch.init(&r, state.max_rows, 64);
        defer scratch_r.deinit();
        const bad_r = try ink.forward.checkRouter(m, &scratch_r, &r);
        std.debug.print("ROUTER-CHECK mismatches {d}\n", .{bad_r});
        if (bad_r != 0) return error.FuseMismatch;
        return;
    }
    var scratch = try state.Scratch.init(&r, @max(chunk, state.max_rows), capacity);
    defer scratch.deinit();
    var prec = ink.model.precisions();
    const schedule = try ink.plan.build(&m.config.text, &prec);

    const host_logits = try gpa.alloc(f32, state.max_rows * vocab);
    defer gpa.free(host_logits);

    if (verify_bench) {
        try verifyBench(gpa, init.io, &ctx, m, &scratch, &r, &schedule, capacity, loaded_refs.items[0].ids, host_logits, profile_steps > 0, route_union);
        return;
    }
    if (bench_prefill > 0 and !profile_all) {
        var cache = try state.Cache.init(&r, capacity, &schedule, kv_dtype);
        defer cache.deinit();
        const ids = loaded_refs.items[0].ids;
        var pos: usize = 0;
        const t0 = nowNs(init.io);
        while (pos < ids.len) {
            const n = @min(chunk, ids.len - pos);
            try promptRun(m, &scratch, &cache, ids[pos..][0..n], .{});
            cache.len += n;
            pos += n;
        }
        _ = try lastLogits(&r, &cache, host_logits[0..vocab]);
        const dt = secs(nowNs(init.io) - t0);
        std.debug.print("\nBENCH prefill {d} rows in chunks of {d}: {d:.3} s, {d:.1} tok/s (cold first token)\n", .{ ids.len, chunk, dt, @as(f64, @floatFromInt(ids.len)) / dt });
        return;
    }
    if (profile_all) {
        try profileAll(gpa, init.io, &ctx, m, &scratch, &r, &schedule, capacity, loaded_refs.items[0].ids, chunk, host_logits[0..vocab], profile_steps);
        return;
    }
    var overall_diffs: usize = 0;
    var prefill_digest = std.crypto.hash.sha2.Sha256.init(.{});
    var decode_digest = std.crypto.hash.sha2.Sha256.init(.{});
    // the probe: [position][visit][qWidth] bf16 of the attention output, written after the first prompt
    const probe_width: usize = m.options.place.qWidth();
    const probe_buf = try gpa.alloc(u16, probe_pos.items.len * schedule.n_visits * probe_width);
    defer gpa.free(probe_buf);
    @memset(probe_buf, 0);
    var probe: ink.prefill.AttProbe = .{ .pos = probe_pos.items, .out = probe_buf, .width = probe_width, .n_visits = schedule.n_visits };
    if (probe_pos.items.len > 0) {
        if (probe_out == null or old_forward) return error.ProbeNeedsOutAndPromptPath;
        ink.prefill.att_probe = &probe;
    }
    for (loaded_refs.items, refs.items) |ref, ref_path| {
        var cache = try state.Cache.init(&r, capacity, &schedule, kv_dtype);
        defer cache.deinit();
        var st: Stats = .{};
        // the largest |ours - reference| over each row's reference top 16, to see where the error sits
        const row_max = try gpa.alloc(f32, ref.rows);
        defer gpa.free(row_max);
        @memset(row_max, 0); // rows that --skip-below-dump leaves out read as 0
        var diffs: std.ArrayList(Diff) = .empty;
        defer diffs.deinit(gpa);
        var dump_file: ?*std.c.FILE = null;
        var compare_file: ?*std.c.FILE = null;
        defer {
            if (dump_file) |f| _ = std.c.fclose(f);
            if (compare_file) |f| _ = std.c.fclose(f);
        }
        if (dump_prefix) |pre| dump_file = try openDump(gpa, pre, ref_path, "wb");
        if (compare_prefix) |pre| compare_file = try openDump(gpa, pre, ref_path, "rb");
        var kl: KlStats = .{};
        const other = try gpa.alloc(f32, if (compare_file != null) vocab else 0);
        defer gpa.free(other);
        const t0 = nowNs(init.io);
        var pos: usize = 0;
        while (pos < ref.rows) {
            const n = @min(chunk, ref.rows - pos);
            if (old_forward or n == 1) {
                if (n > state.max_rows) return error.BadChunk;
                try ink.forward.run(m, &scratch, &cache, ref.ids[pos..][0..n], .{ .all = true });
            } else try ink.prefill.run(m, &scratch, &cache, ref.ids[pos..][0..n], .{ .defer_head = true, .old_experts = old_experts, .old_dense = old_dense, .old_loop = old_loop, .old_global = old_global, .old_sliding = old_sliding });
            cache.len += n;
            if (std.c.getenv("TF_TP_ONE_CHUNK") != null) break;
            var sub: usize = 0;
            while (sub < n) : (sub += state.max_rows) {
            const sn: usize = @min(state.max_rows, n - sub);
            if (skip_below_dump and pos + sub + sn <= dump_from) continue;
            if (!old_forward and n != 1) try ink.prefill.headRows(m, &scratch, sub, sn);
            try r.download(std.mem.sliceAsBytes(host_logits[0 .. sn * vocab]), scratch.logits_all);
            try r.sync();
            if (digest) prefill_digest.update(std.mem.sliceAsBytes(host_logits[0 .. sn * vocab]));
            for (0..sn) |j| {
                const p = pos + sub + j;
                const row = host_logits[j * vocab ..][0..vocab];
                const t = top2(row);
                st.rows += 1;
                if (dump_file) |f| if (p >= dump_from) {
                    if (std.c.fwrite(@ptrCast(row.ptr), 4, vocab, f) != vocab) return error.DumpWrite;
                };
                if (compare_file) |f| if (p >= dump_from) {
                    if (std.c.fread(@ptrCast(other.ptr), 4, vocab, f) != vocab) return error.DumpRead;
                    kl.add(other, row, p);
                };
                const ref_top = ref.top_ids[p * top_n ..][0..top_n];
                const ref_logit = ref.top_logits[p * top_n ..][0..top_n];
                if (!std.math.isFinite(t.first)) st.nan += 1;
                if (t.id == @as(u32, @intCast(ref_top[0]))) {
                    st.agree += 1;
                } else {
                    const gap = row[@intCast(ref_top[0])] - t.first;
                    st.max_diff_margin = @max(st.max_diff_margin, ref.margin[p]);
                    try diffs.append(gpa, .{ .pos = p, .ref_id = ref_top[0], .our_id = t.id, .margin = ref.margin[p], .our_gap = gap });
                }
                {
                    var mx: f32 = t.first;
                    var sum: f64 = 0;
                    for (row) |v| sum += @exp(@as(f64, v - mx));
                    const lse: f64 = @as(f64, mx) + @log(sum);
                    const dl = @abs(lse - @as(f64, ref.lse[p]));
                    st.lse_sum += dl;
                    // the next token's loss under each side, where the reference's top 16 hold it
                    if (p + 1 < ref.rows) for (ref_top, ref_logit) |id, want| if (@as(u32, @intCast(id)) == ref.ids[p + 1]) {
                        st.nll_ours += lse - @as(f64, row[ref.ids[p + 1]]);
                        st.nll_ref += @as(f64, ref.lse[p]) - @as(f64, want);
                        st.nll_n += 1;
                    };
                    st.lse_max = @max(st.lse_max, dl);
                    // KL of the reference to ours over the reference's top 16 (the tail is left out)
                    for (ref_top, ref_logit) |id, want| {
                        const lp: f64 = @as(f64, want) - @as(f64, ref.lse[p]);
                        const lq: f64 = @as(f64, row[@intCast(id)]) - lse;
                        st.kl16_sum += @exp(lp) * (lp - lq);
                    }
                    mx = 0;
                }
                row_max[p] = 0;
                for (ref_top, ref_logit, 0..) |id, want, k| {
                    const d: f64 = @abs(@as(f64, row[@intCast(id)]) - @as(f64, want));
                    row_max[p] = @max(row_max[p], @as(f32, @floatCast(d)));
                    if (!(d <= st.max_abs)) {
                        st.max_abs = if (std.math.isNan(d)) std.math.inf(f64) else d;
                        st.max_abs_row = p;
                    }
                    if (k == 0) {
                        st.max_top1_abs = @max(st.max_top1_abs, d);
                        if (d > 1.0 and std.c.getenv("TF_LOGITS_VERBOSE") != null) std.debug.print("  top-1 logit off by {d:.3} at {d}: reference {d:.3} (token {d}), ours {d:.3}, our top-1 {d} at {d:.3}\n", .{ d, p, want, id, row[@intCast(id)], t.id, t.first });
                    }
                    st.sum_abs += d;
                    st.count_abs += 1;
                }
            }
            }
            pos += n;
        }
        const prefill_s = secs(nowNs(init.io) - t0);
        std.debug.print("\n{s}: {d} rows in chunks of {d}, {d:.1} s ({d:.1} tok/s with every row's logits)\n", .{ std.fs.path.basename(ref_path), st.rows, chunk, prefill_s, @as(f64, @floatFromInt(st.rows)) / prefill_s });
        std.debug.print("  top-1 agreement {d}/{d} ({d:.3} %), top-1 differences {d}, non-finite rows {d}\n", .{ st.agree, st.rows, 100.0 * @as(f64, @floatFromInt(st.agree)) / @as(f64, @floatFromInt(st.rows)), diffs.items.len, st.nan });
        std.debug.print("  reference top-16 logits: max abs difference {d:.4} (at position {d}), mean {d:.5}, max on the top-1 logit {d:.4}\n", .{ st.max_abs, st.max_abs_row, st.sum_abs / @as(f64, @floatFromInt(st.count_abs)), st.max_top1_abs });
        std.debug.print("  KL16 reference to ours over the reference top 16: mean {d:.6}\n", .{st.kl16_sum / @as(f64, @floatFromInt(st.rows))});
        std.debug.print("  log-sum-exp difference: mean {d:.4}, max {d:.4}\n", .{ st.lse_sum / @as(f64, @floatFromInt(st.rows)), st.lse_max });
        std.debug.print("  next-token loss over {d} rows with the token in the reference top 16: ours {d:.4}, reference {d:.4}\n", .{ st.nll_n, st.nll_ours / @as(f64, @floatFromInt(@max(st.nll_n, 1))), st.nll_ref / @as(f64, @floatFromInt(@max(st.nll_n, 1))) });
        if (compare_file != null) kl.print(std.fs.path.basename(ref_path));
        if (ink.prefill.att_probe != null) {
            ink.prefill.att_probe = null;
            const name = try std.fmt.allocPrintSentinel(gpa, "{s}", .{probe_out.?}, 0);
            defer gpa.free(name);
            const f = std.c.fopen(name, "wb") orelse return error.DumpOpen;
            defer _ = std.c.fclose(f);
            if (std.c.fwrite(@ptrCast(probe_buf.ptr), 2, probe_buf.len, f) != probe_buf.len) return error.DumpWrite;
            std.debug.print("PROBE {d} positions x {d} visits x {d} bf16 to {s}\n", .{ probe_pos.items.len, schedule.n_visits, probe_width, probe_out.? });
            for (schedule.visits[0..schedule.n_visits], 0..) |v, vi| std.debug.print("PROBE visit {d} layer {d} pass {d} kind {s}\n", .{ vi, v.layer, v.pass, @tagName(schedule.at(v.layer).kind) });
        }
        try rowSpread(gpa, row_max);
        for (diffs.items) |d| std.debug.print("  diff at {d}: reference token {d}, ours {d}, reference margin {d:.4}, our logit gap {d:.4}\n", .{ d.pos, d.ref_id, d.our_id, d.margin, d.our_gap });
        if (diffs.items.len > 0) std.debug.print("  largest reference margin among the differences: {d:.4}\n", .{st.max_diff_margin});
        overall_diffs += diffs.items.len;

        // greedy decode of one row a step after the prompt: the cost of a token for a user
        if (decode_steps > 0) {
            var token = argmax((try lastLogits(&r, &cache, host_logits[0..vocab])));
            var fwd_ns: i96 = 0;
            const t1 = nowNs(init.io);
            for (0..decode_steps) |_| {
                const f0 = nowNs(init.io);
                try ink.forward.run(m, &scratch, &cache, &.{token}, .{});
                fwd_ns += nowNs(init.io) - f0;
                cache.len += 1;
                token = argmax(try lastLogits(&r, &cache, host_logits[0..vocab]));
                if (digest) decode_digest.update(std.mem.sliceAsBytes(host_logits[0..vocab]));
            }
            const total = secs(nowNs(init.io) - t1);
            std.debug.print("  decode: {d} steps at context {d}: {d:.2} tok/s ({d:.2} ms a token, {d:.2} ms of it the forward)\n", .{ decode_steps, ref.rows, @as(f64, @floatFromInt(decode_steps)) / total, 1000 * total / @as(f64, @floatFromInt(decode_steps)), 1000 * secs(fwd_ns) / @as(f64, @floatFromInt(decode_steps)) });
            if (profile_steps > 0 and ref_path.ptr == refs.items[0].ptr) try profileDecode(&ctx, init.io, m, &scratch, &cache, &r, host_logits[0..vocab], token, profile_steps, null);
        }
    }

    if (ids_path != null and resume_at > 0) {
        // A prompt prefilled in one pass, against the same prompt resumed from a snapshot taken at `resume_at`.
        if (resume_at >= prompt.len) return error.BadResume;
        var fresh = try state.Cache.init(&r, capacity, &schedule, kv_dtype);
        defer fresh.deinit();
        var snap: ?ink.snapshot.Snap = null;
        defer if (snap) |*sn| sn.free();
        var pos: usize = 0;
        while (pos < prompt.len) {
            var n = @min(chunk, prompt.len - pos);
            if (pos < resume_at) n = @min(n, resume_at - pos);
            try promptRun(m, &scratch, &fresh, prompt[pos..][0..n], .{});
            fresh.len += n;
            pos += n;
            if (pos == resume_at) snap = try ink.snapshot.save(&r, &fresh);
        }
        const logits_fresh = try gpa.alloc(f32, vocab);
        defer gpa.free(logits_fresh);
        try fresh.readLogits(logits_fresh);
        var resumed = try state.Cache.init(&r, capacity, &schedule, kv_dtype);
        defer resumed.deinit();
        try ink.snapshot.restore(&resumed, &snap.?);
        pos = resume_at;
        while (pos < prompt.len) {
            const n = @min(chunk, prompt.len - pos);
            try promptRun(m, &scratch, &resumed, prompt[pos..][0..n], .{});
            resumed.len += n;
            pos += n;
        }
        const logits_resumed = try gpa.alloc(f32, vocab);
        defer gpa.free(logits_resumed);
        try resumed.readLogits(logits_resumed);
        var differing: usize = 0;
        var worst: f32 = 0;
        for (logits_fresh, logits_resumed) |a, b| {
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) differing += 1;
            worst = @max(worst, @abs(a - b));
        }
        std.debug.print("resume at {d} of {d}: {d} of {d} logits differ, largest {d:.6}, top-1 {d} against {d}\n", .{ resume_at, prompt.len, differing, vocab, worst, argmax(logits_fresh), argmax(logits_resumed) });
        return;
    }

    if (ids_path != null) {
        var cache = try state.Cache.init(&r, capacity, &schedule, kv_dtype);
        defer cache.deinit();
        var pos: usize = 0;
        const t0 = nowNs(init.io);
        while (pos < prompt.len) {
            const n = @min(chunk, prompt.len - pos);
            try promptRun(m, &scratch, &cache, prompt[pos..][0..n], .{});
            cache.len += n;
            pos += n;
        }
        const prefill_s = secs(nowNs(init.io) - t0);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        const t1 = nowNs(init.io);
        for (0..generate) |_| {
            if (digest) decode_digest.update(std.mem.sliceAsBytes(host_logits[0..vocab]));
            const token = argmax(try lastLogits(&r, &cache, host_logits[0..vocab]));
            try out.append(gpa, token);
            if (stop) |s| if (token == s) break;
            try ink.forward.run(m, &scratch, &cache, &.{token}, .{});
            cache.len += 1;
        }
        const gen_s = secs(nowNs(init.io) - t1);
        std.debug.print("\nprompt {d} tokens in {d:.2} s; generated {d} tokens in {d:.2} s ({d:.2} tok/s)\ntokens:", .{ prompt.len, prefill_s, out.items.len, gen_s, @as(f64, @floatFromInt(out.items.len)) / gen_s });
        for (out.items) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n", .{});
    }
    if (bench_context) |context| {
        var cache = try state.Cache.init(&r, capacity, &schedule, kv_dtype);
        defer cache.deinit();
        // zeroed K/V: every global position reads as a valid, empty row
        for (&cache.slots) |*sl| try state.zero(&r, sl.buf, sl.buf.len);
        try r.sync();
        cache.len = context;
        var token: u32 = 1;
        for (0..4) |_| {
            try ink.forward.run(m, &scratch, &cache, &.{token}, .{});
            cache.len += 1;
            token = argmax(try lastLogits(&r, &cache, host_logits[0..vocab]));
        }
        const t1 = nowNs(init.io);
        for (0..decode_steps) |_| {
            try ink.forward.run(m, &scratch, &cache, &.{token}, .{});
            cache.len += 1;
            token = argmax(try lastLogits(&r, &cache, host_logits[0..vocab]));
        }
        const total = secs(nowNs(init.io) - t1);
        std.debug.print("\nbench: {d} decode steps at context {d} (no prompt): {d:.2} tok/s, {d:.2} ms a token\n", .{ decode_steps, context, @as(f64, @floatFromInt(decode_steps)) / total, 1000 * total / @as(f64, @floatFromInt(decode_steps)) });
        if (profile_steps > 0) try profileDecode(&ctx, init.io, m, &scratch, &cache, &r, host_logits[0..vocab], token, profile_steps, null);
        if (profile_steps > 0) {
            var smp = try ink.sampler.Sampler.init(gpa, &r, &ctx, 1);
            defer smp.deinit();
            try profileDecode(&ctx, init.io, m, &scratch, &cache, &r, host_logits[0..vocab], token, profile_steps, &smp);
        }
    }
    if (digest) {
        var a: [32]u8 = undefined;
        var b: [32]u8 = undefined;
        prefill_digest.final(&a);
        decode_digest.final(&b);
        std.debug.print("\nDIGEST prompt logits {x}\nDIGEST decode logits {x}\n", .{ &a, &b });
    }
    std.debug.print("\nDONE top-1 differences {d}\n", .{overall_diffs});
}

/// Plain decode steps again, once with launch counts and host times only and once with a device timestamp on every kernel.
/// `smp` null: the token is the host argmax of the downloaded row (the check tool's loop). Else the server's sampled path: the
/// device selects the candidates of the last row (tf_ink2_sample_top) and the host draws (T 0.8, top_k 20, top_p 0.95).
fn profileDecode(ctx: *xpu.Context, io: std.Io, m: *const ink.model.Model, scratch: *state.Scratch, cache: *state.Cache, r: *xpu.rt.Runtime, logits: []f32, first: u32, steps: usize, smp: ?*ink.sampler.Sampler) !void {
    var token = first;
    const rule: lanes.Sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 20, .top_p = 0.95 };
    for ([_]bool{ false, true }) |device| {
        var prof = try xpu.profile.Profile.init(ctx, 8192, device);
        defer prof.deinit();
        xpu.profile.active = &prof;
        defer xpu.profile.active = null;
        var wall: i96 = 0;
        var enqueue: i96 = 0;
        var draw: i96 = 0;
        for (0..steps) |_| {
            const f0 = nowNs(io);
            const e0 = xpu.profile.nowNs();
            try ink.forward.run(m, scratch, cache, &.{token}, .{});
            wall += nowNs(io) - f0;
            enqueue += @as(i96, prof.enqueued_at) - e0;
            cache.len += 1;
            if (smp == null) {
                try prof.collect();
                const d0 = nowNs(io);
                token = argmax(try lastLogits(r, cache, logits));
                draw += nowNs(io) - d0;
            } else {
                const d0 = nowNs(io);
                const row = ink.sampler.Row.of(rule);
                const set = try smp.?.select(cache.logits.ptr, @intCast(vocab), &.{row});
                var buf: [ink.sampler.top_max]lanes.gpu_rule.Cand = undefined;
                const top = set.list(0, row.n, &buf);
                const whole: ?[]const f32 = if (lanes.gpu_full.needsRow(rule)) try lastLogits(r, cache, logits) else null;
                token = lanes.gpu_full.drawTop(whole, top, rule, @intCast(cache.len));
                draw += nowNs(io) - d0;
                try prof.collect();
            }
        }
        const n: f64 = @floatFromInt(steps);
        std.debug.print("\nprofile ({s}), {s}: {d} decode steps, {d:.0} launches a token, forward {d:.3} ms a token, of it queueing on the host {d:.3} ms, waiting {d:.3} ms; draw {d:.3} ms a token (needsRow {})\n", .{ if (smp == null) "greedy, host argmax" else "sampled T 0.8 top_k 20 top_p 0.95, device candidates", if (device) "with a timestamp event on every launch" else "host timing only", steps, @as(f64, @floatFromInt(prof.launches)) / n, 1000 * secs(wall) / n, 1000 * secs(enqueue) / n, 1000 * secs(wall - enqueue) / n, 1000 * secs(draw) / n, lanes.gpu_full.needsRow(rule) });
        prof.print();
    }
}

/// The old forward (64-row chunks, a launch a row) with `--old-forward`, else the prompt-chunk forward.
var old_forward = false;
/// The K/V format every cache of this run takes: `XPU_KV_INT8=1` selects int8, anything else bf16.
var kv_dtype: state.KvDtype = .bf16;
var old_experts = false;
var old_dense = false;
var old_loop = false;
var old_global = false;
var old_sliding = false;
/// With `--skip-below-dump`, a block of rows wholly below `--dump-from` makes no host logits (no head on the prompt path,
/// no download), so the per-prompt lines count only the rows from `--dump-from` on.
var skip_below_dump = false;
fn promptRun(m: *const ink.model.Model, scratch: *state.Scratch, cache: *state.Cache, ids: []const u32, _: anytype) !void {
    if (old_forward) {
        if (ids.len > state.max_rows) return error.BadChunk;
        return ink.forward.run(m, scratch, cache, ids, .{});
    }
    return ink.prefill.run(m, scratch, cache, ids, .{ .old_experts = old_experts, .old_dense = old_dense, .old_loop = old_loop, .old_global = old_global, .old_sliding = old_sliding });
}

/// The first reference prompt in chunks of `chunk` rows with the last row's logits only (the server's prompt path), a
/// profile of every chunk (the table after the first 512 rows, then after the whole prompt), then decode at that context.
fn profileAll(gpa: std.mem.Allocator, io: std.Io, ctx: *xpu.Context, m: *const ink.model.Model, scratch: *state.Scratch, r: *xpu.rt.Runtime, schedule: anytype, capacity: usize, ids: []const u32, chunk: usize, logits: []f32, steps: usize) !void {
    var cache = try state.Cache.init(r, capacity, schedule, kv_dtype);
    defer cache.deinit();
    var prof = try xpu.profile.Profile.init(ctx, 131072, true);
    defer prof.deinit();
    xpu.profile.active = &prof;
    var pos: usize = 0;
    var chunks: usize = 0;
    var wall: i96 = 0;
    const first_rows: usize = 512;
    var first_done = false;
    while (pos < ids.len) {
        const n = @min(chunk, ids.len - pos);
        const f0 = nowNs(io);
        try promptRun(m, scratch, &cache, ids[pos..][0..n], .{});
        const dt = nowNs(io) - f0;
        wall += dt;
        try prof.collect();
        cache.len += n;
        pos += n;
        chunks += 1;
        if (pos <= first_rows) {
        }
        if (!first_done and pos >= first_rows) {
            first_done = true;
            std.debug.print("\nPREFILL first {d} rows (context 0 to {d}), {d} chunks of {d}: forward wall {d:.2} ms, {d:.1} tok/s; the table is per chunk (a pass)\n", .{ pos, pos, chunks, chunk, 1000 * secs(wall), @as(f64, @floatFromInt(pos)) / secs(wall) });
            prof.print();
            if (std.c.getenv("TF_TP_ONE_CHUNK") != null) return;
            prof.reset();
        }
    }
    std.debug.print("\nPREFILL rows {d} to {d} ({d} chunks of {d}, wall of the whole prompt {d:.2} s = {d:.1} tok/s): table per chunk (a pass) over the chunks after the first 512 rows\n", .{ first_rows, pos, chunks, chunk, secs(wall), @as(f64, @floatFromInt(pos)) / secs(wall) });
    prof.print();
    xpu.profile.active = null;
    var token = argmax(try lastLogits(r, &cache, logits));
    std.debug.print("\nDECODE at context {d}\n", .{cache.len});
    try profileDecode(ctx, io, m, scratch, &cache, r, logits, token, steps, null);
    var smp = try ink.sampler.Sampler.init(gpa, r, ctx, 1);
    defer smp.deinit();
    token = argmax(try lastLogits(r, &cache, logits));
    try profileDecode(ctx, io, m, scratch, &cache, r, logits, token, steps, &smp);
}

/// Windows of n real rows (the prompt's own next tokens, so the routing is real) through `runShared` the way the backend's
/// verify runs them: all rows' logits, the taps, begin and rollback around each. Median of the reps, in ms: the forward,
/// then the forward and the host argmax of every row's downloaded logits (the greedy verify today).
fn verifyBench(gpa: std.mem.Allocator, io: std.Io, ctx: *xpu.Context, m: *const ink.model.Model, scratch: *state.Scratch, r: *xpu.rt.Runtime, schedule: anytype, capacity: usize, ids: []const u32, host: []f32, profile: bool, route_union: bool) !void {
    var cache = try state.Cache.init(r, capacity, schedule, kv_dtype);
    defer cache.deinit();
    scratch.taps = try r.alloc(state.captureBytes(state.max_rows));
    const prefix: usize = @min(ids.len / 2, 512);
    var pos: usize = 0;
    while (pos < prefix) {
        const n = @min(state.max_rows, prefix - pos);
        try ink.forward.run(m, scratch, &cache, ids[pos..][0..n], .{});
        cache.len += n;
        pos += n;
    }
    const sizes = [_]usize{ 1, 2, 3, 4, 6, 8, 12, 16 };
    const reps = 15;
    var base_ms: f64 = 0;
    std.debug.print("\nVERIFY BENCH at context {d}: windows of real rows, all rows' logits and taps; median of {d}\n", .{ cache.len, reps });
    std.debug.print("rows   forward ms   x one row   forward+host argmax ms   x one row\n", .{});
    for (sizes) |n| {
        var fw: [reps]f64 = undefined;
        var tot: [reps]f64 = undefined;
        for (0..reps) |rep| {
            const from = prefix + (rep * 7) % 64;
            try cache.begin(n);
            const t0 = nowNs(io);
            try ink.forward.runShared(m, scratch, &.{.{ .cache = &cache, .rows = n }}, ids[from..][0..n], .{ .all = true, .taps = scratch.taps.?.ptr });
            const t1 = nowNs(io);
            try r.download(std.mem.sliceAsBytes(host[0 .. n * vocab]), scratch.logits_all);
            try r.sync();
            var sink: u32 = 0;
            for (0..n) |j| sink +%= argmax(host[j * vocab ..][0..vocab]);
            std.mem.doNotOptimizeAway(sink);
            const t2 = nowNs(io);
            fw[rep] = 1000 * secs(t1 - t0);
            tot[rep] = 1000 * secs(t2 - t0);
            try cache.rollback();
        }
        std.mem.sort(f64, &fw, {}, std.sort.asc(f64));
        std.mem.sort(f64, &tot, {}, std.sort.asc(f64));
        if (n == 1) base_ms = fw[reps / 2];
        std.debug.print("{d:4}   {d:10.2}   {d:9.2}   {d:22.2}   {d:9.2}\n", .{ n, fw[reps / 2], fw[reps / 2] / base_ms, tot[reps / 2], tot[reps / 2] / base_ms });
    }
    if (route_union) try routeUnion(gpa, m, scratch, r, &cache, ids, prefix);
    try verifyBits(gpa, io, m, scratch, r, &cache, ids, host, prefix);
    if (profile) {
        var prof = try xpu.profile.Profile.init(ctx, 8192, true);
        defer prof.deinit();
        for ([_]usize{ 1, 8 }) |n| {
            xpu.profile.active = &prof;
            prof.reset();
            for (0..4) |rep| {
                try cache.begin(n);
                try ink.forward.runShared(m, scratch, &.{.{ .cache = &cache, .rows = n }}, ids[prefix + rep * 5 ..][0..n], .{ .all = true, .taps = scratch.taps.?.ptr });
                try prof.collect();
                try cache.rollback();
            }
            xpu.profile.active = null;
            std.debug.print("\nPROFILE of a verify window of {d} rows (mean of 4 windows, per window)\n", .{n});
            prof.print();
        }
    }
}

/// Every verify row's logits against plain decode of that row at that position, bit for bit, and every slot's K and V
/// bytes whole (codes and scales for int8; the rings are head-major, so a row's bytes are not contiguous).
/// The plain side runs eight real decode steps from the window's start: each step's conv history is the one the step
/// before it left. A begin and rollback around each step would put the history back to the window's start, so plain row
/// i would run with the history of row 0 and only row 0 could match. The conv state at the window's start is saved
/// before the plain steps and put back after them, and the cache length goes back to the window's start.
fn verifyBits(gpa: std.mem.Allocator, io: std.Io, m: *const ink.model.Model, scratch: *state.Scratch, r: *xpu.rt.Runtime, cache: *state.Cache, ids: []const u32, host: []f32, prefix: usize) !void {
    _ = io;
    const n = 8;
    const from = prefix;
    const plain = try gpa.alloc(f32, n * vocab);
    defer gpa.free(plain);
    var slot_max: usize = 0;
    for (0..state.visit_slots) |slot| slot_max = @max(slot_max, slotBytes(cache, slot));
    const pk = try gpa.alloc(u8, slot_max);
    defer gpa.free(pk);
    const pv = try gpa.alloc(u8, slot_max);
    defer gpa.free(pv);
    const wk = try gpa.alloc(u8, slot_max);
    defer gpa.free(wk);
    const wv = try gpa.alloc(u8, slot_max);
    defer gpa.free(wv);
    const att_row = state.hidden * 4;
    const patt = try gpa.alloc(u8, n * att_row);
    defer gpa.free(patt);
    const watt = try gpa.alloc(u8, n * att_row);
    defer gpa.free(watt);
    var saved = try r.alloc(state.conv_bytes);
    defer saved.free();
    // the window: one launch of n rows, then a rollback, which puts the conv history back to the window's start
    cache.len = from;
    try cache.begin(n);
    try ink.forward.runShared(m, scratch, &.{.{ .cache = cache, .rows = n }}, ids[from..][0..n], .{ .all = true, .taps = scratch.taps.?.ptr });
    try r.download(std.mem.sliceAsBytes(host[0 .. n * vocab]), scratch.logits_all);
    try r.sync();
    try grabAtt(r, scratch, watt, n);
    try cache.rollback();
    // each slot of the window's K/V, then the plain steps overwrite the same positions
    const wslots = try gpa.alloc([2][]u8, state.visit_slots);
    defer {
        for (wslots) |w| {
            gpa.free(w[0]);
            gpa.free(w[1]);
        }
        gpa.free(wslots);
    }
    for (0..state.visit_slots) |slot| {
        const sb = slotBytes(cache, slot);
        try grabAny(r, cache, slot, wk[0..sb], wv[0..sb]);
        wslots[slot] = .{ try gpa.dupe(u8, wk[0..sb]), try gpa.dupe(u8, wv[0..sb]) };
    }
    try r.copy(saved, cache.conv, state.conv_bytes);
    try r.sync();
    var i: usize = 0;
    while (i < n) : (i += 1) {
        try ink.forward.runShared(m, scratch, &.{.{ .cache = cache, .rows = 1 }}, ids[from + i ..][0..1], .{});
        try r.download(std.mem.sliceAsBytes(plain[i * vocab ..][0..vocab]), cache.logits);
        try r.sync();
        try grabAtt(r, scratch, patt[i * att_row ..][0..att_row], 1);
        cache.len += 1;
    }
    var diff: usize = 0;
    var worst: f64 = 0;
    var first_at: usize = 0;
    var per_row: [8]usize = @splat(0);
    for (0..n) |row| for (0..vocab) |j| {
        const a = plain[row * vocab + j];
        const b = host[row * vocab + j];
        if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) {
            if (diff == 0) first_at = row * vocab + j;
            diff += 1;
            per_row[row] += 1;
            worst = @max(worst, @abs(@as(f64, a) - @as(f64, b)));
        }
    };
    std.debug.print("\nVERIFY BITS: {d} of {d} logits differ from plain decode at the same position (first at {d}, worst {e:.6})\n", .{ diff, n * vocab, first_at, worst });
    std.debug.print("VERIFY BITS per row: {d} {d} {d} {d} {d} {d} {d} {d}\n", .{ per_row[0], per_row[1], per_row[2], per_row[3], per_row[4], per_row[5], per_row[6], per_row[7] });
    var kd: usize = 0;
    var vd: usize = 0;
    var kv_total: usize = 0;
    var first_slot: ?usize = null;
    var krow: usize = 0;
    var vrow: usize = 0;
    var slot0_k: usize = 0;
    var slot0_v: usize = 0;
    for (0..state.visit_slots) |slot| {
        const sb = slotBytes(cache, slot);
        try grabAny(r, cache, slot, pk[0..sb], pv[0..sb]);
        kv_total += sb;
        for (0..sb) |b| {
            if (pk[b] != wslots[slot][0][b]) {
                if (first_slot == null) {
                    first_slot = slot;
                    krow = b;
                }
                kd += 1;
                if (slot == 0) slot0_k += 1;
            }
            if (pv[b] != wslots[slot][1][b]) {
                if (vd == 0) vrow = b;
                vd += 1;
                if (slot == 0) slot0_v += 1;
            }
        }
    }
    std.debug.print("VERIFY KV: {d} of {d} K bytes and {d} of {d} V bytes differ (every slot whole, codes and scales; slot 0 {d} K and {d} V; first slot {?d}, first K byte {d}, first V byte {d})\n", .{ kd, kv_total, vd, kv_total, slot0_k, slot0_v, first_slot, krow, vrow });
    var ad: usize = 0;
    var arow: usize = 0;
    for (0..n * att_row) |b| {
        if (patt[b] != watt[b]) {
            if (ad == 0) arow = b / att_row;
            ad += 1;
        }
    }
    std.debug.print("VERIFY ATT: {d} of {d} attention bytes differ (the last visit's output, first row {d})\n", .{ ad, n * att_row, arow });
    // back to the window's start for whatever runs next
    try r.copy(cache.conv, saved, state.conv_bytes);
    try r.sync();
    cache.len = from;
}

/// One slot's K piece (codes, then scales) or V piece: the whole ring.
fn slotBytes(cache: *const state.Cache, slot: usize) usize {
    const ring = cache.ring(slot);
    return ring * cache.kv.rowBytes() + ring * cache.kv.scaleBytes();
}

fn grabAtt(r: *xpu.rt.Runtime, scratch: *state.Scratch, out: []u8, n: usize) !void {
    const row = state.hidden * 4;
    const b = xpu.rt.Buffer{ .rt = r, .ptr = scratch.att.ptr, .len = n * row };
    try r.download(out, b);
}

/// One slot's K piece and V piece, whole: for bf16 the head-major rings, for int8 the codes and then the scales.
fn grabAny(r: *xpu.rt.Runtime, cache: *state.Cache, slot: usize, k: []u8, v: []u8) !void {
    const s = &cache.slots[slot];
    const n = slotBytes(cache, slot);
    const kb = xpu.rt.Buffer{ .rt = r, .ptr = @ptrFromInt(@intFromPtr(s.buf.ptr.?) + s.key_at), .len = n };
    const vb = xpu.rt.Buffer{ .rt = r, .ptr = @ptrFromInt(@intFromPtr(s.buf.ptr.?) + s.value_at), .len = n };
    try r.download(k[0..n], kb);
    try r.download(v[0..n], vb);
    // a download is only queued: the host bytes are there after the sync
    try r.sync();
}

/// The distinct experts a window of n real rows reads in a visit, averaged over the visits of the same precision and over
/// windows that slide through the prompt, against the random-routing model 128 (1 - (120/128)^n) of PLAN section 3.
fn routeUnion(gpa: std.mem.Allocator, m: *const ink.model.Model, scratch: *state.Scratch, r: *xpu.rt.Runtime, cache: *state.Cache, ids: []const u32, prefix: usize) !void {
    const visits = 48;
    var trace = try r.alloc(visits * 16 * 32);
    defer trace.free();
    ink.forward.route_trace = trace;
    defer ink.forward.route_trace = null;
    const host = try gpa.alloc(i32, visits * 16 * 8);
    defer gpa.free(host);
    std.debug.print("\nROUTE UNION: distinct experts of a window of n real rows, mean over visits and windows (NVFP4 visits 0 to 43, MXFP8 visits 44 to 47)\n", .{});
    std.debug.print("rows   nvfp4 measured   random model   ratio   mxfp8 measured   random model\n", .{});
    const windows = 40;
    for ([_]usize{ 1, 2, 3, 4, 6, 8, 12, 16 }) |n| {
        var sum_nv: f64 = 0;
        var sum_mx: f64 = 0;
        for (0..windows) |w| {
            const from = prefix + (w * 13) % 400;
            try cache.begin(n);
            try ink.forward.runShared(m, scratch, &.{.{ .cache = cache, .rows = n }}, ids[from..][0..n], .{ .all = true });
            try r.download(std.mem.sliceAsBytes(host[0 .. visits * n * 8]), trace);
            try r.sync();
            try cache.rollback();
            for (0..visits) |v| {
                var seen: [128]bool = @splat(false);
                var count: f64 = 0;
                for (host[v * n * 8 ..][0 .. n * 8]) |e| {
                    const ei: usize = @intCast(e);
                    if (!seen[ei]) count += 1;
                    seen[ei] = true;
                }
                if (v < 44) sum_nv += count else sum_mx += count;
            }
        }
        const w: f64 = @floatFromInt(windows);
        const model_u = 128.0 * (1.0 - std.math.pow(f64, 120.0 / 128.0, @floatFromInt(n)));
        std.debug.print("{d:4}   {d:14.2}   {d:12.2}   {d:5.2}   {d:14.2}   {d:12.2}\n", .{ n, sum_nv / (44 * w), model_u, sum_nv / (44 * w) / model_u, sum_mx / (4 * w), model_u });
    }
}

fn lastLogits(r: *xpu.rt.Runtime, cache: *state.Cache, out: []f32) ![]f32 {
    _ = r;
    try cache.readLogits(out);
    return out;
}

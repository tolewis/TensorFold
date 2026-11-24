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

    fn add(k: *KlStats, p: []const f32, q: []const f32) void {
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
        }
        k.rows += 1;
        if (tp.id == tq.id) k.agree += 1;
    }

    fn print(k: KlStats, name: []const u8) void {
        std.debug.print("  vs dump of {s}: KL(dump || ours) mean {e:.4}, max {e:.4} (row {d}); top-1 agreement with the dump {d}/{d} ({d:.3} %)\n", .{ name, k.sum / @as(f64, @floatFromInt(@max(k.rows, 1))), k.max, k.max_row, k.agree, k.rows, 100.0 * @as(f64, @floatFromInt(k.agree)) / @as(f64, @floatFromInt(@max(k.rows, 1))) });
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
    var bench_context: ?usize = null;
    var options: ink.model.Options = .{ .embed_host = false };
    var tp: u32 = 1;
    var rank: u32 = 0;
    var device: ?u32 = null;
    var dump_prefix: ?[]const u8 = null;
    var compare_prefix: ?[]const u8 = null;
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
        } else if (std.mem.eql(u8, a, "--digest")) {
            digest = true;
        } else if (std.mem.eql(u8, a, "--profile")) {
            profile_steps = 8;
        } else if (std.mem.eql(u8, a, "--bench-context")) {
            i += 1;
            bench_context = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--tp")) {
            i += 1;
            tp = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--rank")) {
            i += 1;
            rank = try std.fmt.parseInt(u32, argv[i], 10);
        } else if (std.mem.eql(u8, a, "--device")) {
            i += 1;
            device = try std.fmt.parseInt(u32, argv[i], 10);
        } else dir = a;
    }
    const path = dir orelse {
        std.debug.print("skipped: no model directory (argument or TF_XPU_INKLING2)\n", .{});
        return;
    };
    if (chunk == 0 or chunk > state.max_rows) return error.BadChunk;
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
    if (bench_context) |c| longest = @max(longest, c + decode_steps + profile_steps + 2);
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
    var scratch = try state.Scratch.init(&r, state.max_rows, capacity);
    defer scratch.deinit();
    var prec = ink.model.precisions();
    const schedule = try ink.plan.build(&m.config.text, &prec);

    const host_logits = try gpa.alloc(f32, state.max_rows * vocab);
    defer gpa.free(host_logits);

    var overall_diffs: usize = 0;
    var prefill_digest = std.crypto.hash.sha2.Sha256.init(.{});
    var decode_digest = std.crypto.hash.sha2.Sha256.init(.{});
    for (loaded_refs.items, refs.items) |ref, ref_path| {
        var cache = try state.Cache.init(&r, capacity, &schedule, .bf16);
        defer cache.deinit();
        var st: Stats = .{};
        // the largest |ours - reference| over each row's reference top 16, to see where the error sits
        const row_max = try gpa.alloc(f32, ref.rows);
        defer gpa.free(row_max);
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
            try ink.forward.run(m, &scratch, &cache, ref.ids[pos..][0..n], .{ .all = true });
            cache.len += n;
            try r.download(std.mem.sliceAsBytes(host_logits[0 .. n * vocab]), scratch.logits_all);
            try r.sync();
            if (digest) prefill_digest.update(std.mem.sliceAsBytes(host_logits[0 .. n * vocab]));
            for (0..n) |j| {
                const p = pos + j;
                const row = host_logits[j * vocab ..][0..vocab];
                const t = top2(row);
                st.rows += 1;
                if (dump_file) |f| if (std.c.fwrite(@ptrCast(row.ptr), 4, vocab, f) != vocab) return error.DumpWrite;
                if (compare_file) |f| {
                    if (std.c.fread(@ptrCast(other.ptr), 4, vocab, f) != vocab) return error.DumpRead;
                    kl.add(other, row);
                }
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
            pos += n;
        }
        const prefill_s = secs(nowNs(init.io) - t0);
        std.debug.print("\n{s}: {d} rows in chunks of {d}, {d:.1} s ({d:.1} tok/s with every row's logits)\n", .{ std.fs.path.basename(ref_path), st.rows, chunk, prefill_s, @as(f64, @floatFromInt(st.rows)) / prefill_s });
        std.debug.print("  top-1 agreement {d}/{d} ({d:.3} %), top-1 differences {d}, non-finite rows {d}\n", .{ st.agree, st.rows, 100.0 * @as(f64, @floatFromInt(st.agree)) / @as(f64, @floatFromInt(st.rows)), diffs.items.len, st.nan });
        std.debug.print("  reference top-16 logits: max abs difference {d:.4} (at position {d}), mean {d:.5}, max on the top-1 logit {d:.4}\n", .{ st.max_abs, st.max_abs_row, st.sum_abs / @as(f64, @floatFromInt(st.count_abs)), st.max_top1_abs });
        std.debug.print("  log-sum-exp difference: mean {d:.4}, max {d:.4}\n", .{ st.lse_sum / @as(f64, @floatFromInt(st.rows)), st.lse_max });
        std.debug.print("  next-token loss over {d} rows with the token in the reference top 16: ours {d:.4}, reference {d:.4}\n", .{ st.nll_n, st.nll_ours / @as(f64, @floatFromInt(@max(st.nll_n, 1))), st.nll_ref / @as(f64, @floatFromInt(@max(st.nll_n, 1))) });
        if (compare_file != null) kl.print(std.fs.path.basename(ref_path));
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
            if (profile_steps > 0 and ref_path.ptr == refs.items[0].ptr) try profileDecode(&ctx, init.io, m, &scratch, &cache, &r, host_logits[0..vocab], token, profile_steps);
        }
    }

    if (ids_path != null and resume_at > 0) {
        // A prompt prefilled in one pass, against the same prompt resumed from a snapshot taken at `resume_at`.
        if (resume_at >= prompt.len) return error.BadResume;
        var fresh = try state.Cache.init(&r, capacity, &schedule, .bf16);
        defer fresh.deinit();
        var snap: ?ink.snapshot.Snap = null;
        defer if (snap) |*sn| sn.free();
        var pos: usize = 0;
        while (pos < prompt.len) {
            var n = @min(chunk, prompt.len - pos);
            if (pos < resume_at) n = @min(n, resume_at - pos);
            try ink.forward.run(m, &scratch, &fresh, prompt[pos..][0..n], .{});
            fresh.len += n;
            pos += n;
            if (pos == resume_at) snap = try ink.snapshot.save(&r, &fresh);
        }
        const logits_fresh = try gpa.alloc(f32, vocab);
        defer gpa.free(logits_fresh);
        try fresh.readLogits(logits_fresh);
        var resumed = try state.Cache.init(&r, capacity, &schedule, .bf16);
        defer resumed.deinit();
        try ink.snapshot.restore(&resumed, &snap.?);
        pos = resume_at;
        while (pos < prompt.len) {
            const n = @min(chunk, prompt.len - pos);
            try ink.forward.run(m, &scratch, &resumed, prompt[pos..][0..n], .{});
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
        var cache = try state.Cache.init(&r, capacity, &schedule, .bf16);
        defer cache.deinit();
        var pos: usize = 0;
        const t0 = nowNs(init.io);
        while (pos < prompt.len) {
            const n = @min(chunk, prompt.len - pos);
            try ink.forward.run(m, &scratch, &cache, prompt[pos..][0..n], .{});
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
        var cache = try state.Cache.init(&r, capacity, &schedule, .bf16);
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
        if (profile_steps > 0) try profileDecode(&ctx, init.io, m, &scratch, &cache, &r, host_logits[0..vocab], token, profile_steps);
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
fn profileDecode(ctx: *xpu.Context, io: std.Io, m: *const ink.model.Model, scratch: *state.Scratch, cache: *state.Cache, r: *xpu.rt.Runtime, logits: []f32, first: u32, steps: usize) !void {
    var token = first;
    for ([_]bool{ false, true }) |device| {
        var prof = try xpu.profile.Profile.init(ctx, 8192, device);
        defer prof.deinit();
        xpu.profile.active = &prof;
        defer xpu.profile.active = null;
        var wall: i96 = 0;
        var enqueue: i96 = 0;
        for (0..steps) |_| {
            const f0 = nowNs(io);
            const e0 = xpu.profile.nowNs();
            try ink.forward.run(m, scratch, cache, &.{token}, .{});
            wall += nowNs(io) - f0;
            enqueue += @as(i96, prof.enqueued_at) - e0;
            try prof.collect();
            cache.len += 1;
            token = argmax(try lastLogits(r, cache, logits));
        }
        const n: f64 = @floatFromInt(steps);
        std.debug.print("\nprofile, {s}: {d} decode steps, {d:.0} launches a token, forward {d:.2} ms a token, of it queueing on the host {d:.2} ms, waiting {d:.2} ms\n", .{ if (device) "with a timestamp event on every launch" else "host timing only", steps, @as(f64, @floatFromInt(prof.launches)) / n, 1000 * secs(wall) / n, 1000 * secs(enqueue) / n, 1000 * secs(wall - enqueue) / n });
        prof.print();
    }
}

fn lastLogits(r: *xpu.rt.Runtime, cache: *state.Cache, out: []f32) ![]f32 {
    _ = r;
    try cache.readLogits(out);
    return out;
}

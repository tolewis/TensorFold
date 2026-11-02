//! Device test: the single-stream tree verify with the K1 tree path on (XPU_K1_TREE: the prefix keys
//! through attn_k1, the draft keys through attn_k1_draft, one attn_merge_span join) agrees with the
//! scalar tree path (attn_decode + attn_merge) on the greedy block and every row's logits. The prompt
//! prefix runs the K1 chain in both rounds, so the comparison isolates the tree verify.
const std = @import("std");
const q = @import("qwen27.zig");
const k1mod = @import("k1.zig");

const PROMPT: u32 = 256; // the prefix keys
const MIN_COS: f64 = 0.9999;

fn cos(a: []const f32, b: []const f32) f64 {
    var dab: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        const xa: f64 = x;
        const yb: f64 = y;
        dab += xa * yb;
        na += xa * xa;
        nb += yb * yb;
    }
    if (na == 0 or nb == 0) return 1.0;
    return dab / std.math.sqrt(na * nb);
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const dir = if (argv.len > 1) argv[1] else if (std.c.getenv("TF_XPU_QWEN27")) |v| std.mem.span(v) else {
        std.debug.print("skipped: TF_XPU_QWEN27 not set\n", .{});
        return;
    };
    var probe = q.ze.Gpu.open(0xe223) catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    probe.close();

    var prng = std.Random.DefaultPrng.init(0x5ea4);
    var prompt: [PROMPT]u32 = undefined;
    var draft: [16]u32 = undefined;
    for (&prompt) |*t| t.* = 1000 + prng.random().uintLessThan(u32, 100000);
    for (&draft) |*t| t.* = 1000 + prng.random().uintLessThan(u32, 100000);

    k1mod.on = true; // the K1 kernels load at init; the prefix chain is K1 in both rounds
    var r = try q.Runner.init(dir, 4096);
    defer r.deinit();
    if (r.d.k.attn_k1 == null or r.d.k.attn_k1_draft == null) {
        std.debug.print("skipped: the K1 kernels did not load\n", .{});
        return;
    }
    try r.select(0);
    try r.reset();
    try r.prompt(prompt[0..], true);
    if (r.pt[r.slot] == null) {
        std.debug.print("skipped: no page table (kv pool off)\n", .{});
        return;
    }

    var worst: f64 = 1.0;
    for ([_]u32{ 16, 8 }) |rows| { // the 16-row verify and an 8-row one
        var parents: [16]i32 = @splat(0);
        for (1..16) |t| parents[t] = @as(i32, @intCast(t)) - 1; // a straight comb
        var a_tok: [16]i32 = undefined;
        var b_tok: [16]i32 = undefined;
        var a_log: [16][]const f32 = undefined;
        var b_log: [16][]const f32 = undefined;
        for ([_]bool{ true, false }) |k1t| { // round A the K1 tree, round B the scalar tree
            k1mod.tree = k1t;
            std.debug.print("dbg: rows {d} k1t {any}: reset\n", .{ rows, k1t });
            try r.reset();
            std.debug.print("dbg: prompt\n", .{});
            try r.prompt(prompt[0..], true);
            std.debug.print("dbg: verify\n", .{});
            const blk = try r.verifyTree(draft[0..rows], parents[0..rows]);
            std.debug.print(" done\n", .{});
            const tok: *[16]i32 = if (k1t) &a_tok else &b_tok;
            const log: *[16][]const f32 = if (k1t) &a_log else &b_log;
            for (0..rows) |t| {
                tok[t] = @intCast(blk[t]);
                log[t] = try gpa.dupe(f32, try r.rowLogits(@intCast(t)));
            }
        }
        for (0..rows) |t| {
            if (a_tok[t] != b_tok[t]) {
                std.debug.print("mismatch at rows {d}, row {d}: K1 token {d}, scalar token {d}\n", .{ rows, t, a_tok[t], b_tok[t] });
                return error.GreedyMismatch;
            }
            const c: f64 = cos(a_log[t], b_log[t]);
            if (c < MIN_COS) {
                std.debug.print("mismatch at rows {d}, row {d}: cosine {d}\n", .{ rows, t, c });
                return error.LogitsMismatch;
            }
            worst = @min(worst, c);
        }
        std.debug.print("rows {d}: K1 tree and scalar tree agree (min cosine {d})\n", .{ rows, worst });
    }
    std.debug.print("ok\n", .{});
}

//! kernels.py's q and window-KV norms with their rotations and RoPE, RoPE on heads, the router's and the indexer's row
//! matmuls (rowmm2, rowmm_wts, rowmm_gate) and the MoE routing, each launched as its Python wrapper launches it.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.DECODE_ROWS (TF_DS_DECODE_ROWS unset in the served lane): windows up to this many rows take the row kernels.
pub const decode_rows = 16;
/// kernels.ROWMM2_ROWS: rowmm2's own kernel up to this many rows.
pub const rowmm2_rows = 2;
/// kernels.ROWMM_RB (TF_DS_ROWMM_RB unset in the served lane): _rowmm2r's rows a program at most.
pub const rowmm_rb = 16;
/// kernels.ROWMM_PARTS_MIN (TF_DS_ROWMM_PARTS_MIN unset in the served lane): rowmm_gate's chunk sums from this many rows.
pub const rowmm_parts_min = 1;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

/// A rotation of rmsnorm_rot / q_kv_norm (an EXL3 linear reading the normed rows): its suh [D] and input rows h [R, D], fp16.
pub const Rot = struct { suh: u64, h: u64 };

/// S0, H0, S1, H1 as the wrappers pass them: each rotation's (suh, h), then out (bf16) for each missing one.
fn rotArgs(rot: []const Rot, out: u64) ![4]aot.Arg {
    if (rot.len > 2) return error.TooManyRotations;
    const names = [4][]const u8{ "S0", "H0", "S1", "H1" };
    var a: [4]aot.Arg = undefined;
    for (0..2) |i| {
        a[2 * i] = if (i < rot.len) p(names[2 * i], "*fp16", rot[i].suh) else p(names[2 * i], "*bf16", out);
        a[2 * i + 1] = if (i < rot.len) p(names[2 * i + 1], "*fp16", rot[i].h) else p(names[2 * i + 1], "*bf16", out);
    }
    return a;
}

/// q_kv_norm: rmsnorm_rot(x, w, eps, rot) into out [R, D] (row stride D) and kv_norm_rope(y, wk, ...) into kout [R, DK], one launch.
pub fn qKvNorm(t: Tri, x: u64, xs: usize, w: u64, eps: f32, rot: []const Rot, y: u64, wk: u64, cos: u64, sin: u64, pos: u64, ring: u64, ring_rows: usize, slots: u64, quant: bool, rd: usize, out: u64, kout: u64, rows: usize, d: usize, dk: usize) !void {
    if (d > 2048) return error.RowTooWide;
    const st = try rotArgs(rot, out);
    // switch "qkv_split" (on in the served build): each rotation a program of its own beside the norm's
    const split = rot.len > 0;
    const progs = 2 + (if (split) rot.len else 0);
    try t.run("_q_kv_norm", .{ u(rows), u(progs), 1 }, &.{
        p("X", "*bf16", x),       aot.int("xs", @intCast(xs)), p("W", "*bf16", w),          p("OUT", "*bf16", out),                    aot.int("os_", @intCast(d)),
        aot.float("eps", eps),    st[0],                       st[1],                       st[2],                                     st[3],
        p("Y", "*bf16", y),       p("WK", "*bf16", wk),        p("COS", "*fp32", cos),      p("SIN", "*fp32", sin),                    p("POS", "*i64", pos),
        p("KOUT", "*bf16", kout), p("RING", "*bf16", ring),    p("SLOT_OF", "*i64", slots), aot.int("ring_size", @intCast(ring_rows)),
    }, &.{ cb("QUANT", quant), ci("D", d), ci("BLOCK", tri.pow2(d)), ci("NROT", rot.len), ci("DK", dk), ci("RD", rd), t.pdlConst(), cb("SPLIT", split) });
}

/// rmsnorm_rot: out [R, D] (row stride os) = RMSNorm(x) bf16 and, from its rows, each rotation's h (rot_many's bits).
pub fn rmsnormRot(t: Tri, x: u64, xs: usize, w: u64, eps: f32, rot: []const Rot, out: u64, os: usize, rows: usize, d: usize) !void {
    const st = try rotArgs(rot, out);
    try t.run("_rmsnorm_rot", .{ u(rows), 1, 1 }, &.{
        p("X", "*bf16", x),    aot.int("xs", @intCast(xs)), p("W", "*bf16", w), p("OUT", "*bf16", out), aot.int("os_", @intCast(os)),
        aot.float("eps", eps), st[0],                       st[1],              st[2],                  st[3],
    }, &.{ ci("D", d), ci("BLOCK", tri.pow2(d)), ci("NROT", rot.len), t.pdlConst() });
}

/// kv_norm_rope: the window KV out [R, D] = RMSNorm(y), RoPE on its last rd dims, FP8 quant-dequant, also into ring row slots[r].
pub fn kvNormRope(t: Tri, y: u64, w: u64, cos: u64, sin: u64, pos: u64, ring: u64, ring_rows: usize, slots: u64, eps: f32, quant: bool, rd: usize, out: u64, rows: usize, d: usize) !void {
    try t.run("_kv_norm_rope", .{ u(rows), 1, 1 }, &.{
        p("Y", "*bf16", y),     p("W", "*bf16", w),       p("COS", "*fp32", cos),      p("SIN", "*fp32", sin),                    p("POS", "*i64", pos),
        p("OUT", "*bf16", out), p("RING", "*bf16", ring), p("SLOT_OF", "*i64", slots), aot.int("ring_size", @intCast(ring_rows)), aot.float("eps", eps),
    }, &.{ cb("QUANT", quant), ci("D", d), ci("RD", rd), t.pdlConst() });
}

/// rope_heads: RoPE (inverse: the inverse rotation) on the last rd dims of every head of x [R, H, HD] bf16, in place.
pub fn ropeHeads(t: Tri, x: u64, cos: u64, sin: u64, pos: u64, rd: usize, inverse: bool, rows: usize, h: usize, hd: usize) !void {
    const hb = @min(h, 16);
    try t.run("_rope_heads", .{ u(rows), u(h / hb), 1 }, &.{ p("X", "*bf16", x), p("COS", "*fp32", cos), p("SIN", "*fp32", sin), p("POS", "*i64", pos) }, &.{ ci("H", h), ci("HD", hd), ci("RD", rd), cb("INV", inverse), ci("HB", hb), t.pdlConst() });
}

/// rowmm2: x [R, K] bf16 (row stride xs) @ w [N, K]^T fp16 -> out [R, N] fp32, rowmm's bits: _rowmm2, _rowmm2r past 2 rows, or rowmm.
pub fn rowmm2(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !void {
    // a narrow layer: one output (one warp) a program
    const bn: usize = if (n > 64) 4 else 1;
    if (rows > rowmm2_rows and rowmm_rb > 0 and k % 256 == 0 and rows <= tri.decode_rows) {
        const rb = @min(rowmm_rb, rows);
        return t.run("_rowmm2r", .{ tri.cdiv(rows, rb), tri.cdiv(n, bn), 1 }, &.{
            p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*fp32", out), aot.float("scale", 1.0), aot.int("rows", @intCast(rows)),
        }, &.{ ci("K", k), ci("N", n), ci("BN", bn), ci("BK", 256), ci("RB", rb), t.pdlConst(), cb("WTS", false) });
    }
    if (rows > rowmm2_rows or k % 256 != 0) return rowmm(t, x, xs, w, out, rows, k, n);
    try t.run("_rowmm2", .{ u(rows), tri.cdiv(n, bn), 1 }, &.{
        p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*fp32", out), aot.float("scale", 1.0),
    }, &.{ ci("K", k), ci("N", n), ci("BN", bn), ci("BK", 256), t.pdlConst(), cb("WTS", false) });
}

/// rowmm: x [R, K] bf16 @ w [N, K]^T fp16 -> out [R, N] fp32, a row alone (rowmm2's fallback; no _rowmm launch is recorded).
pub fn rowmm(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !void {
    try t.run("_rowmm", .{ u(rows), tri.cdiv(n, 8), 1 }, &.{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*fp32", out) }, &.{ ci("K", k), ci("N", n), ci("BN", 8), ci("BK", 256), t.pdlConst() });
}

/// rowmm_wts: out [R, N] bf16 = (rowmm(x, w).to(bf16) * scale).to(bf16) in one launch (the indexer's weights).
pub fn rowmmWts(t: Tri, x: u64, xs: usize, w: u64, scale: f32, out: u64, rows: usize, k: usize, n: usize) !void {
    const bn: usize = if (n > 64) 4 else 1;
    try t.run("_rowmm2", .{ u(rows), tri.cdiv(n, bn), 1 }, &.{
        p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*bf16", out), aot.float("scale", scale),
    }, &.{ ci("K", k), ci("N", n), ci("BN", bn), ci("BK", 256), t.pdlConst(), cb("WTS", true) });
}

/// rowmm_parts' tile: outputs a program, K chunks a program and warps (the variant's num_warps).
pub const RpTile = struct { bn: usize, cpg: usize, nw: usize };

/// _rp_tile: rowmm_parts' tile by rows (TF_DS_ROWMM_PARTS unset in the served lane).
pub fn rpTile(rows: usize) RpTile {
    return if (rows <= 6) .{ .bn = 16, .cpg = 1, .nw = 2 } else .{ .bn = 8, .cpg = 1, .nw = 2 };
}

/// rowmm_gate: the gate's logits as chunk sums out [R, K / 256, N] fp32 (decode windows) or rowmm2's [R, N]; returns route()'s kc (0: plain).
pub fn rowmmGate(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !usize {
    const tile = rpTile(rows);
    const kc = k / 256;
    // switch "rowmm_parts" (on in the served build); the wrapper's dtype check (x bf16, w fp16) holds by these pointer types
    if (!(rowmm_parts_min <= rows and rows <= tri.decode_rows and k % 256 == 0 and kc % tile.cpg == 0 and n >= 32)) {
        try rowmm2(t, x, xs, w, out, rows, k, n);
        return 0;
    }
    try rowmmGatePieces(t, x, xs, w, out, rows, k, n, rowmmPartsPiece(t, x, xs, w, out, rows, k, n));
    return kc;
}

/// rowmm_gate's chunk sums as _rowmm_parts launches of at most `piece` rows over consecutive rows (rowmmGate's
/// launches past its checks; tf-dsv41-port --gate compares the piece sizes on a GPU).
pub fn rowmmGatePieces(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize, piece: usize) !void {
    if (piece == 0) return error.BadPiece;
    const kc = k / 256;
    var r0: usize = 0;
    while (r0 < rows) : (r0 += piece) {
        const nr = @min(piece, rows - r0);
        // row r0 of X [R, xs] bf16 and of S [R, kc, N] fp32 (16-byte aligned: xs and kc * N are whole 16-byte rows)
        try rowmmParts(t, x + r0 * xs * 2, xs, w, out + r0 * kc * n * 4, nr, k, n);
    }
}

/// The most rows one _rowmm_parts launch takes when the kit cannot run the whole gate in one: RB 16, which every round
/// of the served 16-row config launches (as RB 1..16 with the leftover piece's rows).
pub const rowmm_parts_piece = decode_rows;

/// Said once on stderr when a gate first runs as 16-row pieces.
var told_split: std.atomic.Value(bool) = .init(false);

/// Rows a _rowmm_parts launch of a `rows`-row gate takes: all of them, unless rows passes the qualified bound and that
/// one launch would not resolve in the kit (Set.has: the exact launch rowmmParts makes, its RB, BN, PDL and the rows
/// argument's specialization, exact or by fallback): a round of TF_DS_ROUND_ROWS > 16 rows on the published kit,
/// which stops at RB 16 (the R32 MissingTritonVariant). Then the gate runs as launches of at most 16 rows over
/// consecutive rows of X and S. A kit with the RB 17..32 variants (tools/dsv41-zig/build_kit.sh EXTRA_ROWS=32) keeps
/// the single launch. Rounds of up to 16 rows are untouched (one launch, no kit check).
///
/// Expected, not yet shown on a GPU: the kernel is row-invariant (a row's chunk sums do not depend on the rows beside
/// it or on RB), so each row's bytes equal one launch's. The pieces' tiles differ from a single 17..32-row launch's
/// (a leftover of 1..6 rows takes BN 16, a 17..32-row launch BN 8), so docs/port/kit.md section 8 compares S per row,
/// bitwise, before token hashes.
///
/// The check costs one scan of the set's variants per gate past 16 rows; Set.find already scans them on every launch.
pub fn rowmmPartsPiece(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) usize {
    if (rows <= rowmm_parts_piece) return rows;
    const l = partsLaunch(t, x, xs, w, out, rows, k, n);
    const one = if (t.set) |set| set.has("_rowmm_parts", &l.args, &l.consts) else if (t.kit) |specs| aot.specsHave(specs, "_rowmm_parts", &l.args, &l.consts) else true;
    if (one) return rows;
    if (!told_split.swap(true, .monotonic)) std.log.info("_rowmm_parts: the kit has no variant for a {d}-row gate (N {d}); rounds past {d} rows run it as launches of at most {d} rows", .{ rows, n, rowmm_parts_piece, rowmm_parts_piece });
    return rowmm_parts_piece;
}

/// One _rowmm_parts launch as rowmm_gate's wrapper makes it: grid, runtime arguments, constexprs.
const PartsLaunch = struct { grid: [3]u32, args: [5]aot.Arg, consts: [7]aot.Const };

fn partsLaunch(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) PartsLaunch {
    const tile = rpTile(rows);
    const kc = k / 256;
    return .{
        .grid = .{ tri.cdiv(n, tile.bn), u(kc / tile.cpg), 1 },
        .args = .{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("S", "*fp32", out), aot.int("rows", @intCast(rows)) },
        .consts = .{ ci("K", k), ci("N", n), ci("BN", tile.bn), ci("BK", 256), ci("CPG", tile.cpg), ci("RB", rows), t.pdlConst() },
    };
}

fn rowmmParts(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !void {
    const l = partsLaunch(t, x, xs, w, out, rows, k, n);
    try t.run("_rowmm_parts", l.grid, &l.args, &l.consts);
}

/// route: picks [R, SLOTS] i32 and weights [R, SLOTS] fp32 (shared expert last) from logits [R, NE] (kc 0) or chunk sums [R, kc, NE].
pub fn route(t: Tri, logits: u64, kc: usize, bias: u64, topk: usize, scale: f32, shared_id: usize, pick: u64, wts: u64, rows: usize, ne: usize, slots: usize) !void {
    if (kc > 0) return t.run("_route_parts", .{ u(rows), 1, 1 }, &.{
        p("S", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale), aot.int("shared_id", @intCast(shared_id)),
    }, &.{ ci("NE", ne), ci("NB", tri.pow2(ne)), ci("TOPK", topk), ci("SLOTS", slots), ci("SP", tri.pow2(slots)), ci("KC", kc), t.pdlConst() });
    // one warp a row (switch "rowmm", on in the served build): the variant's num_warps
    try t.run("_route", .{ u(rows), 1, 1 }, &.{
        p("L", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale), aot.int("shared_id", @intCast(shared_id)),
    }, &.{ ci("NE", ne), ci("NB", tri.pow2(ne)), ci("TOPK", topk), ci("SLOTS", slots), ci("SP", tri.pow2(slots)), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

/// A recorded tensor's dtype as torch names it ("bfloat16", "float32", ...).
fn dtype(c: Case, tensor: []const u8) []const u8 {
    return c.v.get("tensors").?.object.get(tensor).?.array.items[0].string;
}

/// The wrapper's rot list from the recorded S0 H0 S1 H1: a rotation's suh is a vector [D], a missing one's slot holds out [R, D].
fn rotOf(c: Case, buf: *[2]Rot) []const Rot {
    const pairs = [2][2][]const u8{ .{ "S0", "H0" }, .{ "S1", "H1" } };
    var n: usize = 0;
    for (pairs) |pr| {
        if (c.rank(pr[0]) != 1) break;
        buf[n] = .{ .suh = c.ptr(pr[0]), .h = c.ptr(pr[1]) };
        n += 1;
    }
    return buf[0..n];
}

test "q_kv_norm launches as recorded" {
    try tri.conform("_q_kv_norm", struct {
        fn call(t: Tri, c: Case) !void {
            var buf: [2]Rot = undefined;
            const quant = c.constInt("QUANT") != 0;
            const rd: usize = @intCast(c.constInt("RD"));
            try qKvNorm(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.float("eps"), rotOf(c, &buf), c.ptr("Y"), c.ptr("WK"), c.ptr("COS"), c.ptr("SIN"), c.ptr("POS"), c.ptr("RING"), c.dim("RING", 0), c.ptr("SLOT_OF"), quant, rd, c.ptr("OUT"), c.ptr("KOUT"), c.dim("X", 0), c.dim("X", 1), c.dim("Y", 1));
        }
    }.call);
}

test "rmsnorm_rot launches as recorded" {
    try tri.conform("_rmsnorm_rot", struct {
        fn call(t: Tri, c: Case) !void {
            var buf: [2]Rot = undefined;
            try rmsnormRot(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.float("eps"), rotOf(c, &buf), c.ptr("OUT"), c.stride("OUT", 0), c.dim("X", 0), c.dim("X", 1));
        }
    }.call);
}

test "kv_norm_rope launches as recorded" {
    try tri.conform("_kv_norm_rope", struct {
        fn call(t: Tri, c: Case) !void {
            const quant = c.constInt("QUANT") != 0;
            const rd: usize = @intCast(c.constInt("RD"));
            try kvNormRope(t, c.ptr("Y"), c.ptr("W"), c.ptr("COS"), c.ptr("SIN"), c.ptr("POS"), c.ptr("RING"), c.dim("RING", 0), c.ptr("SLOT_OF"), c.float("eps"), quant, rd, c.ptr("OUT"), c.dim("Y", 0), c.dim("Y", 1));
        }
    }.call);
}

test "rope_heads launches as recorded" {
    try tri.conform("_rope_heads", struct {
        fn call(t: Tri, c: Case) !void {
            try ropeHeads(t, c.ptr("X"), c.ptr("COS"), c.ptr("SIN"), c.ptr("POS"), @intCast(c.constInt("RD")), c.constInt("INV") != 0, c.dim("X", 0), c.dim("X", 1), c.dim("X", 2));
        }
    }.call);
}

test "rowmm2 (1-2 rows) and rowmm_wts launch _rowmm2 as recorded" {
    try tri.conform("_rowmm2", struct {
        fn call(t: Tri, c: Case) !void {
            // rowmm_wts writes the indexer's bf16 weights, rowmm2 fp32 rows
            if (std.mem.eql(u8, dtype(c, "OUT"), "bfloat16")) return rowmmWts(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.float("scale"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
            try rowmm2(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
        }
    }.call);
}

test "rowmm2 (windows past 2 rows) launches _rowmm2r as recorded" {
    try tri.conform("_rowmm2r", struct {
        fn call(t: Tri, c: Case) !void {
            try rowmm2(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
        }
    }.call);
}

test "rowmm_gate launches _rowmm_parts as recorded" {
    try tri.conform("_rowmm_parts", struct {
        fn call(t: Tri, c: Case) !void {
            const kc = try rowmmGate(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.ptr("S"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
            try std.testing.expectEqual(c.dim("S", 1), kc);
        }
    }.call);
}

/// An in-memory kit of _rowmm_parts variants at the round's gate (K 5120, N 384), one per (RB, rows specialization,
/// PDL): what aot.zig's Set holds as specs. rows: "one" (folded to a constexpr), "d16" or "any".
const PartsVariant = struct { rb: usize, rows: []const u8, pdl: u1 = 1 };

fn partsKit(gpa: std.mem.Allocator, variants: []const PartsVariant) !std.json.Parsed(aot.SetJson) {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try text.appendSlice(gpa, "{\"kernels\": [");
    for (variants, 0..) |v, i| {
        const one = std.mem.eql(u8, v.rows, "one");
        const rows_param = if (one) "" else try std.fmt.allocPrint(gpa, ",{{\"name\":\"rows\",\"type\":\"i32\",\"div16\":{},\"nospec\":false}}", .{std.mem.eql(u8, v.rows, "d16")});
        defer if (!one) gpa.free(rows_param);
        const ptr = "\"div16\":true,\"nospec\":false";
        const entry = try std.fmt.allocPrint(gpa,
            \\{s}{{"fn":"_rowmm_parts","hash":"h{d}","name":"k{d}","num_warps":2,"params":[
            \\{{"name":"X","type":"*bf16",{s}}},{{"name":"xs","type":"i32",{s}}},{{"name":"W","type":"*fp16",{s}}},
            \\{{"name":"S","type":"*fp32",{s}}}{s}],
            \\"consts":{{"K":{{"int":5120}},"N":{{"int":384}},"BN":{{"int":{d}}},"BK":{{"int":256}},"CPG":{{"int":1}},"RB":{{"int":{d}}},"PDL":{{"int":{d}}}{s}}}}}
        , .{ if (i == 0) "" else ",", i, i, ptr, ptr, ptr, ptr, rows_param, rpTile(v.rb).bn, v.rb, v.pdl, if (one) ",\"rows\":{\"int\":1}" else "" });
        defer gpa.free(entry);
        try text.appendSlice(gpa, entry);
    }
    try text.appendSlice(gpa, "]}");
    return std.json.parseFromSlice(aot.SetJson, gpa, text.items, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

test "rowmm_gate past 16 rows: one launch when the kit resolves it, else 16-row pieces over consecutive rows" {
    const gpa = std.testing.allocator;
    const saved = tri.decode_rows;
    defer tri.decode_rows = saved;
    tri.decode_rows = 32; // a TF_DS_ROUND_ROWS=32 round
    const x: u64 = 0x1000_0000;
    const s: u64 = 0x2000_0000;
    // the published kit's shape: RB 1..16 (rows folded at 1, plain, div16 at 16), no RB > 16
    const base = [_]PartsVariant{ .{ .rb = 1, .rows = "one" }, .{ .rb = 16, .rows = "d16" } };
    const published = try partsKit(gpa, &base);
    defer published.deinit();
    // RB = rows present but compiled with PDL 0 while the launch passes PDL 1: a different constexpr never resolves
    const wrong_pdl = try partsKit(gpa, &(base ++ [_]PartsVariant{ .{ .rb = 17, .rows = "any", .pdl = 0 }, .{ .rb = 32, .rows = "d16", .pdl = 0 } }));
    defer wrong_pdl.deinit();
    // RB = 32 compiled for rows div16 only: a 32-row launch fits it, and no 17-row one
    const r32 = try partsKit(gpa, &(base ++ [_]PartsVariant{ .{ .rb = 17, .rows = "any" }, .{ .rb = 32, .rows = "d16" } }));
    defer r32.deinit();
    for ([_]usize{ 17, 32 }) |rows| {
        // RB absent, and RB present with the wrong PDL: the split, each piece resolving in that kit
        for ([_][]const aot.KernelJson{ published.value.kernels, wrong_pdl.value.kernels }) |kit| {
            var log: tri.Log = .{ .gpa = gpa };
            defer log.deinit();
            const kc = try rowmmGate(.{ .log = &log, .kit = kit }, x, 5120, 0x3000_0000, s, rows, 5120, 384);
            try std.testing.expectEqual(@as(usize, 20), kc);
            try std.testing.expectEqual(@as(usize, 2), log.items.items.len);
            const want_rows = [_]usize{ 16, rows - 16 };
            for (log.items.items, 0..) |l, i| {
                try std.testing.expectEqualStrings("_rowmm_parts", l.name);
                try std.testing.expectEqual(x + i * 16 * 5120 * 2, l.args[0].value.ptr.addr);
                try std.testing.expectEqual(s + i * 16 * 20 * 384 * 4, l.args[3].value.ptr.addr);
                try std.testing.expectEqual(@as(i32, @intCast(want_rows[i])), l.args[4].value.i32);
                for (l.consts) |c| if (std.mem.eql(u8, c.name, "RB")) try std.testing.expectEqual(@as(?i64, @intCast(want_rows[i])), c.int);
                try std.testing.expectEqual(tri.cdiv(384, rpTile(want_rows[i]).bn), l.grid[0]);
                try std.testing.expect(aot.specsHave(kit, l.name, l.args, l.consts));
            }
        }
        // RB = rows present and fitting: one launch, which that kit resolves
        var one: tri.Log = .{ .gpa = gpa };
        defer one.deinit();
        _ = try rowmmGate(.{ .log = &one, .kit = r32.value.kernels }, x, 5120, 0x3000_0000, s, rows, 5120, 384);
        try std.testing.expectEqual(@as(usize, 1), one.items.items.len);
        try std.testing.expect(aot.specsHave(r32.value.kernels, "_rowmm_parts", one.items.items[0].args, one.items.items[0].consts));
        // PDL off in the engine (TF_DS_TRITON_PDL=0) against that PDL-1 kit: the single launch does not resolve, so the split
        var off: tri.Log = .{ .gpa = gpa };
        defer off.deinit();
        _ = try rowmmGate(.{ .log = &off, .kit = r32.value.kernels, .pdl = false }, x, 5120, 0x3000_0000, s, rows, 5120, 384);
        try std.testing.expectEqual(@as(usize, 2), off.items.items.len);
    }
    // an argument specialization the variant does not hold (S off 16 bytes; RB 32 assumes it aligned) is not "has":
    // the split (whose pieces miss in this kit as well, so Set.run reports MissingTritonVariant as before)
    var mis: tri.Log = .{ .gpa = gpa };
    defer mis.deinit();
    _ = try rowmmGate(.{ .log = &mis, .kit = r32.value.kernels }, x, 5120, 0x3000_0000, s + 4, 32, 5120, 384);
    try std.testing.expectEqual(@as(usize, 2), mis.items.items.len);
    // 16 rows and fewer: one launch whatever the kit
    var log: tri.Log = .{ .gpa = gpa };
    defer log.deinit();
    _ = try rowmmGate(.{ .log = &log, .kit = published.value.kernels }, x, 5120, 0x3000_0000, s, 16, 5120, 384);
    try std.testing.expectEqual(@as(usize, 1), log.items.items.len);
}

/// route() on a recorded case: rowmm_gate's chunk sums S [R, KC, NE] or plain logits L [R, NE].
fn callRoute(t: Tri, c: Case) !void {
    const parts = c.has("S");
    const lg = if (parts) "S" else "L";
    try route(t, c.ptr(lg), if (parts) c.dim("S", 1) else 0, c.ptr("BIAS"), @intCast(c.constInt("TOPK")), c.float("scale"), @intCast(c.int("shared_id")), c.ptr("PICK"), c.ptr("WTS"), c.dim(lg, 0), c.dim(lg, c.rank(lg) - 1), c.dim("PICK", 1));
}

test "route (logits) launches _route as recorded" {
    try tri.conform("_route", callRoute);
}

test "route (rowmm_gate's chunk sums) launches _route_parts as recorded" {
    try tri.conform("_route_parts", callRoute);
}

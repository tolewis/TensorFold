//! XPU_GDN bit 4 (lane p4-k-gdn): gdn_step2 (or gdn_step2_uc, gdn_step3) against gdn_step on this card, at the load.
//! Four cases cover the uses of the kernel: a tree block (LOAD and SAVE), the commit replay (ROWS, the state in place),
//! a plain chain of 40 rows, and two stream windows (MT). Every output value must agree, or the open fails. The chain
//! is longer than 16 rows because gdn_step3 reads the first 16 rows' tree words in another way than the rows after.
const std = @import("std");
const ze = @import("ze");
const dev = @import("dev.zig");
const q = @import("qwen27.zig");
const Dev = dev.Dev;
const alloc = q.alloc;

const HK = q.GDN_K_HEADS;
const HV = q.GDN_V_HEADS;
const ROWS = 16;
const LONG = 40; // the rows of the chain case, and of the inputs
const STATE = HV * 128 * 128; // floats of one layer's state

var seed: u64 = 0x9E3779B97F4A7C15;
fn next() f32 { // xorshift: a value in [-1, 1)
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    const bits: u32 = @truncate(seed >> 40); // 24 bits
    return (@as(f32, @floatFromInt(bits)) - 8388608.0) / 8388608.0;
}
fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

fn put(d: *Dev, comptime T: type, host: []const T) !*anyopaque {
    const p = try d.g.deviceAlloc(@max(host.len * @sizeOf(T), 64));
    try d.copyIn(p, std.mem.sliceAsBytes(host));
    return p;
}
fn room(d: *Dev, bytes: usize) !*anyopaque {
    return d.g.deviceAlloc(bytes);
}
fn back(d: *Dev, comptime T: type, src: *anyopaque, n: usize) ![]T {
    const host = try alloc.alloc(T, n);
    errdefer alloc.free(host);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, host.ptr, src, n * @sizeOf(T), null, 0, null));
    try d.sync();
    return host;
}
/// Values of `n` that differ between the two device buffers.
fn differ(d: *Dev, comptime T: type, a: *anyopaque, b: *anyopaque, n: usize) !usize {
    const x = try back(d, T, a, n);
    defer alloc.free(x);
    const y = try back(d, T, b, n);
    defer alloc.free(y);
    var bad: usize = 0;
    for (x, y) |u, w| bad += @intFromBool(u != w);
    return bad;
}

pub fn run(d: *Dev, old: ze.H, new: ze.H, name: []const u8) !void {
    try d.g.indirect(old);
    try d.g.indirect(new);
    const nul: ?*anyopaque = null;
    var held: std.ArrayList(*anyopaque) = .empty;
    defer {
        for (held.items) |p| d.g.free(p);
        held.deinit(alloc);
    }
    // inputs: 40 rows of q, k, v (bf16), g and beta, and two states
    const qk = try alloc.alloc(u16, LONG * HK * 128);
    defer alloc.free(qk);
    const v = try alloc.alloc(u16, LONG * HV * 128);
    defer alloc.free(v);
    const gb = try alloc.alloc(f32, LONG * HV);
    defer alloc.free(gb);
    const st = try alloc.alloc(f32, STATE);
    defer alloc.free(st);
    for (qk) |*x| x.* = bf16(next() * 0.09);
    const Q = try put(d, u16, qk);
    try held.append(alloc, Q);
    for (qk) |*x| x.* = bf16(next() * 0.09);
    const K = try put(d, u16, qk);
    try held.append(alloc, K);
    for (v) |*x| x.* = bf16(next());
    const V = try put(d, u16, v);
    try held.append(alloc, V);
    for (gb) |*x| x.* = 0.95 + 0.05 * next();
    const G = try put(d, f32, gb);
    try held.append(alloc, G);
    for (gb) |*x| x.* = 0.5 + 0.5 * next();
    const B = try put(d, f32, gb);
    try held.append(alloc, B);
    var S: [6]*anyopaque = undefined; // 0, 1: state x (read only, and a spare); 2, 3: copies of x for the in-place case; 4: state y; 5: spare
    for (st) |*x| x.* = next() * 0.5;
    for ([_]usize{ 0, 1, 2, 3 }) |i| {
        S[i] = try put(d, f32, st);
        try held.append(alloc, S[i]);
    }
    for (st) |*x| x.* = next() * 0.5;
    for ([_]usize{ 4, 5 }) |i| {
        S[i] = try put(d, f32, st);
        try held.append(alloc, S[i]);
    }
    var out: [2]struct { y: *anyopaque, so: *anyopaque, so2: *anyopaque, sl: *anyopaque } = undefined;
    for (&out) |*o| {
        o.* = .{ .y = try room(d, LONG * HV * 128 * 2), .so = try room(d, STATE * 4), .so2 = try room(d, STATE * 4), .sl = try room(d, ROWS * STATE * 4) };
        for ([_]*anyopaque{ o.y, o.so, o.so2, o.sl }) |p| try held.append(alloc, p);
    }
    const ks = [2]ze.H{ old, new };
    const grid: ze.GroupCount = .{ .x = 16, .y = HV, .z = 1 };

    // 1. a tree block, as Runner.verifyTree builds its table: LOAD where the parent is not the row before, SAVE on that parent
    const parents = [ROWS]i32{ -1, 0, 1, 2, 0, 4, 5, 1, 7, 8, 0, 10, 11, 4, 13, 14 };
    var tree: [ROWS * 17]i32 = @splat(0);
    for (parents, 0..) |pj, j| {
        const row = tree[j * 17 ..][0..17];
        if (j == 0) {
            row[1] = 0;
            continue;
        }
        const pr = tree[@as(usize, @intCast(pj)) * 17 ..][0..17];
        const dp: usize = @intCast(pr[0] & 255);
        row[0] = @intCast(dp + 1);
        @memcpy(row[1 .. dp + 2], pr[1 .. dp + 2]);
        row[dp + 2] = @intCast(j);
        if (pj + 1 != @as(i32, @intCast(j))) {
            row[0] |= 512;
            pr[0] |= 256;
        }
    }
    const TREE = try put(d, i32, &tree);
    try held.append(alloc, TREE);
    for (ks, out) |k, o| try d.launch(k, grid, .{ Q, K, V, G, B, S[0], o.so, o.y, @as(i32, ROWS), nul, TREE, o.sl, nul });
    try d.sync();
    const tree_y = try differ(d, u16, out[0].y, out[1].y, ROWS * HV * 128);
    var tree_sl: usize = 0;
    var saves: usize = 0;
    for (0..ROWS) |t| if (tree[t * 17] & 256 != 0) {
        saves += 1;
        tree_sl += try differ(d, u32, dev.off(out[0].sl, t * STATE * 4), dev.off(out[1].sl, t * STATE * 4), STATE);
    };

    // 2. the commit replay: the rows of a path, the state read and written in place
    const path = [_]i32{ 0, 4, 13, 14 };
    const PATH = try put(d, i32, &path);
    try held.append(alloc, PATH);
    for (ks, out, [_]usize{ 2, 3 }) |k, o, si| try d.launch(k, grid, .{ Q, K, V, G, B, S[si], S[si], o.y, @as(i32, path.len), PATH, nul, nul, nul });
    try d.sync();
    const rows_y = try differ(d, u16, out[0].y, out[1].y, path.len * HV * 128);
    const rows_s = try differ(d, u32, S[2], S[3], STATE);

    // 3. a plain chain of 40 rows, the state to a second buffer
    for (ks, out) |k, o| try d.launch(k, grid, .{ Q, K, V, G, B, S[0], o.so, o.y, @as(i32, LONG), nul, nul, nul, nul });
    try d.sync();
    const chain_y = try differ(d, u16, out[0].y, out[1].y, LONG * HV * 128);
    const chain_so = try differ(d, u32, out[0].so, out[1].so, STATE);

    // 4. two stream windows (rows 0 to 9 on state x, rows 10 to 15 on state y) through the per-row table
    var mts: [2]*anyopaque = undefined;
    for (out, &mts) |o, *m| {
        var mt: [ROWS * 8]u64 = @splat(0);
        for (0..ROWS) |t| {
            const second = t >= 10;
            mt[t * 8 + 4] = @intFromPtr(if (second) S[4] else S[0]);
            mt[t * 8 + 5] = @intFromPtr(if (second) o.so2 else o.so);
            mt[t * 8 + 6] = if (second) 10 else 0;
            mt[t * 8 + 7] = if (second) 6 else 10;
        }
        m.* = try put(d, u64, &mt);
        try held.append(alloc, m.*);
    }
    for (ks, out, mts) |k, o, m| try d.launch(k, .{ .x = 16, .y = HV, .z = ROWS }, .{ Q, K, V, G, B, S[1], S[5], o.y, @as(i32, ROWS), nul, nul, nul, m });
    try d.sync();
    const mt_y = try differ(d, u16, out[0].y, out[1].y, ROWS * HV * 128);
    const mt_so = try differ(d, u32, out[0].so, out[1].so, STATE) + try differ(d, u32, out[0].so2, out[1].so2, STATE);

    const bad = tree_y + tree_sl + rows_y + rows_s + chain_y + chain_so + mt_y + mt_so;
    std.debug.print("gdn check: {s} against gdn_step, {d} values differ (tree y {d}, {d} saved states {d}; rows y {d}, state {d}; chain y {d}, state {d}; streams y {d}, states {d})\n", .{ name, bad, tree_y, saves, tree_sl, rows_y, rows_s, chain_y, chain_so, mt_y, mt_so });
    if (bad != 0) return error.GdnStep2Differs;
}

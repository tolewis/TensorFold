//! TF_DS_ENGRAM_LEAD=1: each Engram layer reads its rows of the next prompt chunk on a reader thread of its own.
const std = @import("std");
const Config = @import("config.zig").Config;
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const prompt = @import("prompt.zig");
const EngramHost = prompt.EngramHost;

/// TF_DS_ENGRAM_LEAD's value: "1" turns it on; unset or anything else keeps the reference read-ahead.
pub fn enabled(v: ?[]const u8) bool {
    return std.mem.eql(u8, v orelse return false, "1");
}

/// The layers' readers: fillChunk calls begin at a chunk's start, then before and after around each Engram layer.
pub const Lead = struct {
    threads: [2]?std.Thread = .{ null, null },

    /// Layer j's read of the chunk at [start, start + n) on its thread; when the spawn fails the layer reads its own.
    fn spawn(ld: *Lead, eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) void {
        ld.join(j);
        eh.lead_start[j] = start;
        eh.lead_n[j] = n;
        eh.lead_ready[j] = false;
        eh.lead_err[j] = null;
        ld.threads[j] = std.Thread.spawn(.{}, read, .{ eh, c, seq, start, n, j }) catch null;
    }

    pub fn join(ld: *Lead, j: usize) void {
        if (ld.threads[j]) |th| th.join();
        ld.threads[j] = null;
    }

    /// Every read done, before the prompt changes or a round runs; rows already read stay this prompt's.
    pub fn joinAll(ld: *Lead) void {
        for (0..ld.threads.len) |j| ld.join(j);
    }

    /// A new prompt (fillBegin): every read done, and no row read for another prompt is taken, even at its positions.
    pub fn reset(ld: *Lead, eh: *EngramHost) void {
        ld.joinAll();
        eh.lead_start = @splat(std.math.maxInt(usize));
        eh.lead_n = @splat(0);
        eh.lead_ready = @splat(false);
        eh.lead_err = @splat(null);
    }

    fn mine(eh: *const EngramHost, j: usize, start: usize, n: usize) bool {
        return eh.lead_start[j] == start and eh.lead_n[j] == n;
    }

    /// Layer j's rows of this chunk are read or on their way.
    fn covered(ld: *const Lead, eh: *const EngramHost, j: usize, start: usize, n: usize) bool {
        return mine(eh, j, start, n) and (ld.threads[j] != null or eh.lead_ready[j]);
    }

    /// A chunk's start: the first Engram layer's read starts here when none for this chunk is ready or on its way.
    pub fn begin(ld: *Lead, eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize) void {
        if (c.engram_layers.slice().len > 0 and !ld.covered(eh, 0, start, n)) {
            eh.lead_late += 1;
            ld.spawn(eh, c, seq, start, n, 0);
        }
    }

    /// Before Engram layer j of the chunk: its read done; its error when it failed for this chunk.
    pub fn before(ld: *Lead, eh: *EngramHost, j: usize, start: usize, n: usize) !void {
        ld.join(j);
        if (mine(eh, j, start, n)) if (eh.lead_err[j]) |err| return err;
    }

    /// After layer j: the next layer's read of this chunk when none, then layer j's of the next chunk (`next`, `nn`).
    pub fn after(ld: *Lead, eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize, next: ?[]const i32, nn: usize) void {
        const ne = @min(2, c.engram_layers.slice().len);
        if (j + 1 < ne and !ld.covered(eh, j + 1, start, n)) {
            eh.lead_late += 1;
            ld.spawn(eh, c, seq, start, n, j + 1);
        }
        // layer j's buffers are free: its rows went up from pageable memory, which the copy stages before returning
        if (next) |nx| ld.spawn(eh, c, nx, start + n, nn, j);
    }
};

/// Lead's reader of layer j: the chunk's rows into ahead_rows[j] as engramRows reads them; errors land in lead_err[j].
fn read(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) void {
    read1(eh, c, seq, start, n, j) catch |err| {
        eh.lead_err[j] = prompt.rankFault(err);
    };
}

fn read1(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) !void {
    const t = eh.tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
    const ring = eh.lead_ring[j] orelse return error.NoEngramRing;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    const m = n * k;
    const hs = eh.lead_hashes[j][0 .. n * eh.hasher.layers * cols];
    eh.hasher.hashes(seq, start, n, hs);
    for (0..n) |r| {
        for (0..k) |q| eh.ahead_flat[j][r * k + q] = hs[(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    try prompt.aioGather(ring, t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]);
    engram.decodeRows(eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s], eh.ahead_rows[j][0 .. m * t.row_w], m, t.row_w, t.row_s, 8);
    eh.lead_ready[j] = true;
}

test "TF_DS_ENGRAM_LEAD: only 1 turns it on" {
    try std.testing.expect(enabled("1"));
    for ([_]?[]const u8{ null, "", "0", "2", "on", "true", " 1", "1 " }) |v| try std.testing.expect(!enabled(v));
}

/// The lead test's tables: layers 1 and 14, rows[j] rows of 32 weight bytes and 1 scale, each byte set by its offset.
fn leadToyFile(a: std.mem.Allocator, rows: [2]usize) ![]u8 {
    const w0 = rows[0] * 32;
    const s0 = rows[0];
    const w1 = rows[1] * 32;
    const s1 = rows[1];
    var json_buf: [1024]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},32],\"data_offsets\":[0,{d}]}}," ++
        "\"layers.1.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},1],\"data_offsets\":[{d},{d}]}}," ++
        "\"layers.14.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},32],\"data_offsets\":[{d},{d}]}}," ++
        "\"layers.14.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},1],\"data_offsets\":[{d},{d}]}}}}", .{ rows[0], w0, rows[0], w0, w0 + s0, rows[1], w0 + s0, w0 + s0 + w1, rows[1], w0 + s0 + w1, w0 + s0 + w1 + s1 });
    var file: std.ArrayList(u8) = .empty;
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..w0 + s0 + w1 + s1) |i| try file.append(a, @intCast((i * 131 + 17) % 251));
    return file.items;
}

/// Layer slot j's rows of the chunk at [start, start + n), read independently: hashed, the pool's preads, decoded.
fn leadWant(a: std.mem.Allocator, eh: *const EngramHost, t: engram_io.Table, j: usize, seq: []const i32, start: usize, n: usize) ![]u16 {
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    const m = n * k;
    const hs = try a.alloc(i64, n * eh.hasher.layers * cols);
    eh.hasher.hashes(seq, start, n, hs);
    const ids = try a.alloc(i64, m);
    for (0..n) |r| {
        for (0..k) |q| ids[r * k + q] = hs[(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    const w = try a.alloc(u8, m * t.row_w);
    const s = try a.alloc(u8, m * t.row_s);
    try eh.pool.gather(t, ids, w, s);
    const out = try a.alloc(u16, m * t.row_w);
    engram.decodeRows(w, s, out, m, t.row_w, t.row_s, 1);
    return out;
}

test "TF_DS_ENGRAM_LEAD: each layer's rows of the next chunk are read while this one runs, the bytes read at the layer" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    // layers 1 and 14, up to 4-grams, one head: primes 11, 13, 17 (layer 1: 41 rows) and 19, 23, 29 (layer 14: 71)
    var map: [40]i32 = undefined;
    for (&map, 0..) |*x, i| x.* = @intCast((i * 7 + 3) % 31);
    const hasher = engram.Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = try leadToyFile(a, .{ 41, 71 }) });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    var tables = try engram_io.Tables.open(gpa, io, dir);
    defer tables.close();
    const pool = try engram_io.Pool.init(gpa, io, 2);
    defer pool.deinit(gpa);
    const cap = 8;
    // rank 1 of 2: hash columns [1, 3) of 3, the high-order ones (the worker's)
    var eh = try EngramHost.init(a, &c, hasher, &tables, pool, 1, 2, cap);
    var rings: [2]*engram_aio.Aio = undefined;
    for (&rings, 0..) |*r, i| r.* = engram_aio.Aio.init(gpa, io, 64) catch |err| {
        for (rings[0..i]) |x| x.deinit();
        return if (err == error.AioUnavailable) error.SkipZigTest else err;
    };
    defer for (rings) |r| r.deinit();
    eh.paio = rings[0];
    const tabs = [2]engram_io.Table{ tables.layers.get(1).?, tables.layers.get(14).? };
    const k = eh.hi - eh.lo;
    // a 37-token prompt (an image position at 20: no n-gram reaches across it) and another of the same length
    var p1: [37]i32 = undefined;
    var p2: [37]i32 = undefined;
    for (&p1, &p2, 0..) |*x, *y, i| {
        x.* = @intCast((i * 13 + 5) % 40);
        y.* = @intCast((i * 17 + 11) % 40);
    }
    p1[20] = -1;

    // the reference path (switch off): read at the layer, the same bytes
    const want0 = try leadWant(a, &eh, tabs[0], 0, p1[0..cap], 0, cap);
    const ref = try prompt.engramRows(&eh, 0, tabs[0], p1[0..cap], 0, cap);
    try std.testing.expect(ref.ptr == eh.rows.ptr);
    try std.testing.expectEqualSlices(u16, want0, ref[0 .. cap * k * 32]);

    try eh.useLead(a, cap, rings);
    const Pass = struct {
        /// fillChunk's Engram steps over chunks [from, to), a round before chunk `round`; returns the reader hits.
        fn run(al: std.mem.Allocator, ld: *Lead, h: *EngramHost, cf: *const Config, tb: [2]engram_io.Table, seq: []const i32, from: usize, to: usize, round: ?usize) !usize {
            var hits: usize = 0;
            var chunk = from;
            while (chunk < to) : (chunk += 1) {
                const start = chunk * cap;
                const n = @min(cap, seq.len - start);
                if (round != null and round.? == chunk) ld.joinAll();
                ld.begin(h, cf, seq[0 .. start + n], start, n);
                for (0..2) |j| {
                    try ld.before(h, j, start, n);
                    const got = try prompt.engramRows(h, j, tb[j], seq[0 .. start + n], start, n);
                    if (got.ptr == h.ahead_rows[j].ptr) hits += 1;
                    try std.testing.expectEqualSlices(u16, try leadWant(al, h, tb[j], j, seq[0 .. start + n], start, n), got[0 .. n * (h.hi - h.lo) * 32]);
                    const more = seq.len > start + n;
                    const nn = if (more) @min(cap, seq.len - start - n) else 0;
                    ld.after(h, cf, seq[0 .. start + n], start, n, j, if (more) seq[0 .. start + n + nn] else null, nn);
                }
            }
            return hits;
        }
    };
    var ld: Lead = .{};
    defer ld.joinAll();
    // 5 chunks (the last of 5 rows), a round before chunk 3: every row from its reader, only chunk 0's two reads late
    ld.reset(&eh);
    try std.testing.expectEqual(@as(usize, 10), try Pass.run(a, &ld, &eh, &c, tabs, &p1, 0, 5, 3));
    try std.testing.expectEqual(@as(u64, 2), eh.lead_late);
    // a pass stopped after two chunks: the third's rows already read
    ld.reset(&eh);
    try std.testing.expectEqual(@as(usize, 4), try Pass.run(a, &ld, &eh, &c, tabs, &p1, 0, 2, null));
    ld.joinAll();
    try std.testing.expect(eh.lead_ready[0] and eh.lead_start[0] == 2 * cap);
    // rows ready for another chunk are never taken: the layer reads its own (and the ready ones stay)
    const other = try prompt.engramRows(&eh, 0, tabs[0], p1[0 .. 2 * cap], cap, cap);
    try std.testing.expect(other.ptr == eh.rows.ptr and eh.lead_ready[0]);
    try std.testing.expectEqualSlices(u16, try leadWant(a, &eh, tabs[0], 0, p1[0 .. 2 * cap], cap, cap), other[0 .. cap * k * 32]);
    // a chunk other than the one read ahead starts its own readers
    try std.testing.expectEqual(@as(usize, 2), try Pass.run(a, &ld, &eh, &c, tabs, &p1, 1, 2, null));
    try std.testing.expectEqual(@as(u64, 6), eh.lead_late);
    // another prompt resumed at the positions the first one's rows were read for: those rows are never taken
    ld.reset(&eh);
    try std.testing.expectEqual(@as(usize, 6), try Pass.run(a, &ld, &eh, &c, tabs, &p2, 2, 5, null));
    try std.testing.expectEqual(@as(u64, 8), eh.lead_late);
}

test "an Engram read this rank could not make ends the pair at the layer, on every path" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    var map: [40]i32 = undefined;
    for (&map, 0..) |*x, i| x.* = @intCast((i * 7 + 3) % 31);
    const hasher = engram.Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // the header of the toy tables, and a single byte of their data: every row read runs past the end of the file
    const whole = try leadToyFile(a, .{ 41, 71 });
    const header = 8 + std.mem.readInt(u64, whole[0..8], .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = whole[0 .. header + 1] });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    var tables = try engram_io.Tables.open(gpa, io, dir);
    defer tables.close();
    const pool = try engram_io.Pool.init(gpa, io, 2);
    defer pool.deinit(gpa);
    const cap = 8;
    var eh = try EngramHost.init(a, &c, hasher, &tables, pool, 1, 2, cap);
    var rings: [2]*engram_aio.Aio = undefined;
    for (&rings, 0..) |*r, i| r.* = engram_aio.Aio.init(gpa, io, 64) catch |err| {
        for (rings[0..i]) |x| x.deinit();
        return if (err == error.AioUnavailable) error.SkipZigTest else err;
    };
    defer for (rings) |r| r.deinit();
    const tabs = [2]engram_io.Table{ tables.layers.get(1).?, tables.layers.get(14).? };
    var seq: [cap]i32 = undefined;
    for (&seq, 0..) |*x, i| x.* = @intCast((i * 13 + 5) % 40);
    const round_mod = @import("round.zig");
    const fault: anyerror = error.EngramRankFault;

    // read at the layer, by the pool's preads and by the prompt's ring
    try std.testing.expectError(error.EngramRankFault, prompt.engramRows(&eh, 0, tabs[0], &seq, 0, cap));
    eh.paio = rings[0];
    try std.testing.expectError(error.EngramRankFault, prompt.engramRows(&eh, 0, tabs[0], &seq, 0, cap));
    // the read-ahead
    eh.ahead_err = null;
    prompt.engramReadAhead(&eh, &c, &seq, 0, cap, 0);
    try std.testing.expectEqual(fault, eh.ahead_err.?);
    eh.ahead_err = null;
    prompt.engramReadAheadAll(&eh, &c, &seq, 0, cap);
    try std.testing.expectEqual(fault, eh.ahead_err.?);
    // TF_DS_ENGRAM_LEAD's readers: the layer's wait returns the fault, for each layer
    try eh.useLead(a, cap, rings);
    var ld: Lead = .{};
    defer ld.joinAll();
    ld.reset(&eh);
    ld.begin(&eh, &c, &seq, 0, cap);
    for (0..2) |j| {
        const err = ld.before(&eh, j, 0, cap);
        try std.testing.expectError(error.EngramRankFault, err);
        if (j == 0) ld.after(&eh, &c, &seq, 0, cap, 0, null, 0); // (layer 1's reader starts after layer 0)
    }
    // and all of them end the pair
    try std.testing.expect(round_mod.endsPair(fault));
}

test "a table short of any column's ids is refused at start on both ranks, before a read can part them" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    var map: [40]i32 = undefined;
    for (&map, 0..) |*x, i| x.* = @intCast((i * 7 + 3) % 31);
    const hasher = engram.Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    try std.testing.expectEqual(@as(u64, 41), hasher.rowsNeeded(0));
    try std.testing.expectEqual(@as(u64, 71), hasher.rowsNeeded(1));
    const pool = try engram_io.Pool.init(gpa, io, 2);
    defer pool.deinit(gpa);
    // rank 0 reads column 0 (rows < 11), rank 1 columns 1 and 2 (up to 41): 40 rows serve rank 0 and fail rank 1
    for ([_][2]usize{ .{ 40, 71 }, .{ 41, 70 } }) |rows| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = try leadToyFile(a, rows) });
        const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
        var tables = try engram_io.Tables.open(gpa, io, dir);
        defer tables.close();
        for (0..2) |rank| try std.testing.expectError(error.BadEngramTable, EngramHost.init(a, &c, hasher, &tables, pool, rank, 2, 8));
    }
    // whole tables open on both
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = try leadToyFile(a, .{ 41, 71 }) });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    var tables = try engram_io.Tables.open(gpa, io, dir);
    defer tables.close();
    for (0..2) |rank| _ = try EngramHost.init(a, &c, hasher, &tables, pool, rank, 2, 8);
}

fn openFds() !usize {
    var d = try std.Io.Dir.cwd().openDir(std.testing.io, "/proc/self/fd", .{ .iterate = true });
    defer d.close(std.testing.io);
    var n: usize = 0;
    var it = d.iterate();
    while (try it.next(std.testing.io)) |_| n += 1;
    return n;
}

fn openTables(gpa: std.mem.Allocator, dir: []const u8) !void {
    var t = try engram_io.Tables.open(gpa, std.testing.io, dir);
    t.close();
}

fn openPool(gpa: std.mem.Allocator) !void {
    const p = try engram_io.Pool.init(gpa, std.testing.io, 2);
    p.deinit(gpa);
}

test "the Engram tables and the read pool, failing at any allocation, leave no memory and no descriptor behind" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "engram.safetensors", .data = try leadToyFile(a, .{ 41, 71 }) });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    const before = try openFds();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, openTables, .{dir});
    try std.testing.expectEqual(before, try openFds());
    try std.testing.checkAllAllocationFailures(std.testing.allocator, openPool, .{});
}

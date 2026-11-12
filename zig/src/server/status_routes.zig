//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const live = @import("live.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// With keys on, health names nothing about the model. A rank pair whose round is past its stall bound, or that
/// broke, answers 503 with status "stalled" or "broken" (only a backend with a broken-pair policy reports one).
pub fn health(srv: *Server, conn: *Conn, a: Allocator) !void {
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const word = if (status.pair) |p| p.health() else "ok";
    const code: u16 = if (std.mem.eql(u8, word, "ok")) 200 else 503;
    if (srv.keys) |k| if (k.enabled()) {
        if (code == 200) return conn.sendJson(200, "{\"status\": \"ok\"}");
        return conn.sendJson(code, if (std.mem.eql(u8, word, "broken")) "{\"status\": \"broken\"}" else "{\"status\": \"stalled\"}");
    };
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = word });
    try o.put(a, "model", .{ .string = srv.config.served_name });
    const ids = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, ids) |id, *slot| slot.* = .{ .string = id };
    try o.put(a, "model_ids", .{ .array = ids });
    try o.put(a, "max_batch_size", try json.intValue(a, srv.info.lanes));
    try o.put(a, "warming", .{ .bool = status.warming });
    try putTotals(a, o, srv.metrics.totals(srv.io), srv.info.prompt_cache);
    const memory = try json.newObject(a);
    if (srv.engine.memory(std.mem.indexOf(u8, conn.path, "reset_peak=1") != null)) |mem| {
        try memory.put(a, "active", try json.intValue(a, mem.active));
        try memory.put(a, "cache", try json.intValue(a, mem.cache));
        try memory.put(a, "peak", try json.intValue(a, mem.peak));
    }
    try o.put(a, "memory", .{ .object = memory });
    try o.put(a, "live", try live.snapshot(a, srv.engine));
    if (status.cache) |c| try putCache(a, o, c);
    if (status.kv_parts) |k| try putKvParts(a, o, k);
    if (status.pair) |p| try putPair(a, o, p);
    routes.sendValue(conn, a, code, .{ .object = o });
}

/// The token counts of finished requests, as /metrics counts them (the same totals, read together): `prompt_tokens_total`
/// (cached ones included), `cached_tokens_total` (the part restored from kept prompt states, only from an engine that
/// keeps some; the name a dashboard such as sparkDash reads) and `generation_tokens_total`. Computed prompt tokens are
/// the difference.
pub fn putTotals(a: Allocator, o: *json.Object, t: @import("metrics.zig").Metrics.Totals, keeps: bool) !void {
    try o.put(a, "prompt_tokens_total", try json.intValue(a, t.prompt));
    if (keeps) try o.put(a, "cached_tokens_total", try json.intValue(a, t.cached));
    try o.put(a, "generation_tokens_total", try json.intValue(a, t.generation));
}

/// The rank pair's robustness counters and state, under "pair" (a backend with a broken-pair policy or a disk tier).
pub fn putPair(a: Allocator, o: *json.Object, p: api.PairStatus) !void {
    const po = try json.newObject(a);
    inline for (@typeInfo(api.PairStatus).@"struct".field_names, @typeInfo(api.PairStatus).@"struct".field_types) |name, T| {
        if (T == bool) try po.put(a, name, .{ .bool = @field(p, name) }) else try po.put(a, name, try json.intValue(a, @field(p, name)));
    }
    try o.put(a, "pair", .{ .object = po });
}

/// The prompt cache's counters, when its tier or grouping is on: the GLM engine's top-level names (chats_continued,
/// chats_continued_prefill_tokens), then every
/// counter under prompt_cache, the tier's own under prompt_cache.spill (an array, one object per rank, rank 0 first).
pub fn putCache(a: Allocator, o: *json.Object, c: api.CacheStats) Allocator.Error!void {
    try o.put(a, "chats_continued", try json.intValue(a, c.chats_continued));
    try o.put(a, "chats_continued_prefill_tokens", try json.intValue(a, c.chats_continued_prefill_tokens));
    const pc = try json.newObject(a);
    const info = @typeInfo(api.CacheStats).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64) try pc.put(a, name, try json.intValue(a, @field(c, name)));
    if (c.spill) |rc| {
        const ranks = rc.slice();
        const arr = try a.alloc(json.Value, ranks.len);
        for (ranks, arr, 0..) |t, *v, r| {
            const sp = try json.newObject(a);
            try sp.put(a, "rank", try json.intValue(a, r));
            inline for (@typeInfo(@TypeOf(t)).@"struct".field_names) |name| try sp.put(a, name, try json.intValue(a, @field(t, name)));
            v.* = .{ .object = sp };
        }
        try pc.put(a, "spill", .{ .array = arr });
    }
    try o.put(a, "prompt_cache", .{ .object = pc });
}

/// KV ownership's counters under `kv_shard` (a backend that shards its pool over the ranks: wojo's GLM 0118), every
/// part's rows under kv_shard.per_part (an array, one object per part, part 0 first), and every rank's own counters
/// under kv_shard.per_rank (one object per rank that reported, rank 0 first; empty when none did).
pub fn putKvParts(a: Allocator, o: *json.Object, k: api.KvParts) Allocator.Error!void {
    const ks = try json.newObject(a);
    const info = @typeInfo(api.KvParts).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64) try ks.put(a, name, try json.intValue(a, @field(k, name)));
    const n: usize = @intCast(@min(k.parts, k.part.len));
    const arr = try a.alloc(json.Value, n);
    for (k.part[0..n], arr, 0..) |p, *v, i| {
        const po = try json.newObject(a);
        try po.put(a, "part", try json.intValue(a, i));
        inline for (@typeInfo(@TypeOf(p)).@"struct".field_names) |name| try po.put(a, name, try json.intValue(a, @field(p, name)));
        v.* = .{ .object = po };
    }
    try ks.put(a, "per_part", .{ .array = arr });
    const nr: usize = @intCast(@min(k.ranks, k.rank.len));
    const rows = try a.alloc(json.Value, nr);
    for (k.rank[0..nr], rows, 0..) |r, *v, i| {
        const ro = try json.newObject(a);
        try ro.put(a, "rank", try json.intValue(a, i));
        inline for (@typeInfo(@TypeOf(r)).@"struct".field_names) |name| try ro.put(a, name, try json.intValue(a, @field(r, name)));
        v.* = .{ .object = ro };
    }
    try ks.put(a, "per_rank", .{ .array = rows });
    try o.put(a, "kv_shard", .{ .object = ks });
}

pub fn models(srv: *Server, conn: *Conn, a: Allocator) !void {
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    const data = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, data) |id, *slot| {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = id });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "created", try json.intValue(a, created));
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        if (srv.info.context_window > 0) try m.put(a, "max_model_len", try json.intValue(a, srv.info.context_window)); // the window one request may use
        slot.* = .{ .object = m };
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Prometheus text, version 0.0.4.
pub fn metrics(srv: *Server, conn: *Conn, a: Allocator) void {
    var out: std.Io.Writer.Allocating = .init(a);
    srv.metrics.render(srv.io, &out.writer, srv.engine, srv.info.context_window) catch return;
    const body = out.written();
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

pub fn dashboard(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{dashboard_page.len}) catch return) catch return;
    conn.finish(dashboard_page) catch {};
}

pub fn stats(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const memory = srv.engine.memory(false);
    srv.metrics.mutex.lockUncancelable(srv.io);
    const prompt = srv.metrics.prompt;
    const generation = srv.metrics.generation;
    const drafted = srv.metrics.drafted;
    const accepted = srv.metrics.accepted;
    const rounds = srv.metrics.rounds;
    srv.metrics.mutex.unlock(srv.io);
    const body = statsSnapshot(a, status, memory, prompt, generation, drafted, accepted, rounds) catch return;
    conn.quiet_log = true;
    routes.sendValue(conn, a, 200, body);
}

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

const dashboard_page = @embedFile("dashboard.html");

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
}

test "health's prompt cache counters: the GLM names on top, every counter below, the tier's under spill" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    var ranks: api.prompt_cache.RankCounts = .{ .n = 2 };
    ranks.ranks[0].points = 5;
    ranks.ranks[1].points = 4;
    try putCache(a, o, .{ .chats_continued = 1, .disk_hits_committed = 3, .spill = ranks });
    try std.testing.expectEqualStrings("1", o.get("chats_continued").?.int);
    const pc = o.get("prompt_cache").?.object;
    try std.testing.expectEqualStrings("3", pc.get("disk_hits_committed").?.int);
    const spill = pc.get("spill").?.array;
    try std.testing.expectEqual(@as(usize, 2), spill.len);
    try std.testing.expectEqualStrings("5", spill[0].object.get("points").?.int);
    try std.testing.expectEqualStrings("1", spill[1].object.get("rank").?.int);
    try std.testing.expectEqualStrings("4", spill[1].object.get("points").?.int);
    const none = try json.newObject(a);
    try putCache(a, none, .{});
    try std.testing.expect(none.get("prompt_cache").?.object.get("spill") == null);
}

test "health's kv shard counters: every counter under kv_shard, each part's rows under per_part" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    var k: api.KvParts = .{ .parts = 2, .carries = 5, .twins_made = 1 };
    k.part[1] = .{ .free = 7, .live = 2 };
    try putKvParts(a, o, k);
    const ks = o.get("kv_shard").?.object;
    try std.testing.expectEqualStrings("5", ks.get("carries").?.int);
    try std.testing.expectEqualStrings("1", ks.get("twins_made").?.int);
    const pp = ks.get("per_part").?.array;
    try std.testing.expectEqual(@as(usize, 2), pp.len);
    try std.testing.expectEqualStrings("1", pp[1].object.get("part").?.int);
    try std.testing.expectEqualStrings("7", pp[1].object.get("free").?.int);
    try std.testing.expectEqualStrings("2", pp[1].object.get("live").?.int);
    try std.testing.expectEqual(@as(usize, 0), ks.get("per_rank").?.array.len); // (no rank reported)
    // every rank's own counters under per_rank, rank 0 first
    k.ranks = 2;
    k.rank[0] = .{ .owner_rounds = 11, .plans = 3 };
    k.rank[1] = .{ .owner_rounds = 12, .plans = 3, .transfers_failed = 1 };
    const o2 = try json.newObject(a);
    try putKvParts(a, o2, k);
    const pr = o2.get("kv_shard").?.object.get("per_rank").?.array;
    try std.testing.expectEqual(@as(usize, 2), pr.len);
    try std.testing.expectEqualStrings("1", pr[1].object.get("rank").?.int);
    try std.testing.expectEqualStrings("12", pr[1].object.get("owner_rounds").?.int);
    try std.testing.expectEqualStrings("1", pr[1].object.get("transfers_failed").?.int);
    try std.testing.expectEqualStrings("11", pr[0].object.get("owner_rounds").?.int);
}

test "health's pair status: every field under pair, the word is ok, stalled or broken" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    try putPair(a, o, .{ .service_refused = 2, .pair_failed = 1, .busy_ms = 7, .stall_ms = 120000, .stalled = true });
    const po = o.get("pair").?.object;
    try std.testing.expectEqualStrings("2", po.get("service_refused").?.int);
    try std.testing.expectEqualStrings("1", po.get("pair_failed").?.int);
    try std.testing.expectEqualStrings("120000", po.get("stall_ms").?.int);
    try std.testing.expect(po.get("stalled").?.bool);
    try std.testing.expect(!po.get("broken").?.bool);
    try std.testing.expectEqualStrings("ok", (api.PairStatus{}).health());
    try std.testing.expectEqualStrings("stalled", (api.PairStatus{ .stalled = true }).health());
    try std.testing.expectEqualStrings("broken", (api.PairStatus{ .stalled = true, .broken = true }).health());
}

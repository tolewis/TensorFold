//! Cached prompt tokens in /metrics and /health, and max_model_len in /v1/models (the harness is late_system_test.zig's).
const std = @import("std");
const server = @import("root.zig");
const harness = @import("late_system_test.zig");

const chat_body =
    \\{"model":"m","max_tokens":4,"messages":[{"role":"user","content":"hi there"}]}
;

fn start(text: *harness.DropLate, backend: *harness.Capture) !*server.Server {
    return server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
}

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

test "restored prompt tokens are counted in /metrics (with vLLM's names) and /health, from the same totals" {
    var text: harness.DropLate = .{};
    var backend: harness.Capture = .{ .cached = 4, .window = 4096, .keeps = true };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try start(&text, &backend);
    defer srv.deinit();

    // before any request: all zero, and present
    const idle = try harness.get(srv, "/metrics");
    defer std.testing.allocator.free(idle.body);
    try std.testing.expect(contains(idle.body, "tensorfold:prompt_tokens_cached_total 0\n"));

    for (0..2) |_| {
        const r = try harness.exchange(srv, "/v1/chat/completions", chat_body);
        defer std.testing.allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expect(contains(r.body, "\"cached_tokens\": 4"));
    }
    const prompt = backend.prompt.len; // one request's prompt; the backend restored 4 of them each time

    const scrape = try harness.get(srv, "/metrics");
    defer std.testing.allocator.free(scrape.body);
    var want: [3][96]u8 = undefined;
    const lines = [_][]const u8{
        try std.fmt.bufPrint(&want[0], "tensorfold:prompt_tokens_total {d}\n", .{2 * prompt}),
        "tensorfold:prompt_tokens_cached_total 8\n",
        "tensorfold:prefix_cache_queries_total ",
    };
    for (lines) |l| try std.testing.expect(contains(scrape.body, l));
    try std.testing.expect(contains(scrape.body, "tensorfold:prefix_cache_hits_total 8\n"));

    const health = try harness.get(srv, "/health");
    defer std.testing.allocator.free(health.body);
    try std.testing.expectEqual(@as(u16, 200), health.status);
    try std.testing.expect(contains(health.body, "\"cached_tokens_total\": 8"));
    var buf: [64]u8 = undefined;
    try std.testing.expect(contains(health.body, try std.fmt.bufPrint(&buf, "\"prompt_tokens_total\": {d}", .{2 * prompt})));
}

test "/v1/models names the context window as max_model_len, and leaves it out when the engine reports none" {
    var text: harness.DropLate = .{};
    var backend: harness.Capture = .{ .window = 4096 };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try start(&text, &backend);
    defer srv.deinit();
    const models = try harness.get(srv, "/v1/models");
    defer std.testing.allocator.free(models.body);
    try std.testing.expectEqual(@as(u16, 200), models.status);
    try std.testing.expect(contains(models.body, "\"max_model_len\": 4096"));

    var text0: harness.DropLate = .{};
    var backend0: harness.Capture = .{};
    defer std.testing.allocator.free(backend0.prompt);
    const srv0 = try start(&text0, &backend0);
    defer srv0.deinit();
    const none = try harness.get(srv0, "/v1/models");
    defer std.testing.allocator.free(none.body);
    try std.testing.expect(!contains(none.body, "max_model_len"));
}

test "an engine that keeps no prompt states publishes no cached counts" {
    var text: harness.DropLate = .{};
    var backend: harness.Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    const srv = try start(&text, &backend);
    defer srv.deinit();
    const r = try harness.exchange(srv, "/v1/chat/completions", chat_body);
    defer std.testing.allocator.free(r.body);
    const scrape = try harness.get(srv, "/metrics");
    defer std.testing.allocator.free(scrape.body);
    try std.testing.expect(contains(scrape.body, "tensorfold:prompt_tokens_total "));
    try std.testing.expect(!contains(scrape.body, "cached_total") and !contains(scrape.body, "prefix_cache"));
    const health = try harness.get(srv, "/health");
    defer std.testing.allocator.free(health.body);
    try std.testing.expect(contains(health.body, "\"prompt_tokens_total\""));
    try std.testing.expect(!contains(health.body, "cached_tokens_total"));
}

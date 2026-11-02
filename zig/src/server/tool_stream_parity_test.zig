//! Single-call streamed and whole chat replies agree on prose (#1396).
const std = @import("std");
const json = @import("root.zig").json;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const pre = @import("stream_preflight_test.zig");
const TestText = pre.TestText;
const CallEngine = pre.CallEngine;
const sendWith = pre.sendWith;
const Policy = @import("tool_stream.zig").Policy;
const parse = @import("tool_parse.zig").parse;

const head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
const write_head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}]";
const opened = "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\nThe history of";
const glm_opened = "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value><arg_key>content</arg_key><arg_value>The history of";

fn expectSingleParity(a: Allocator, tools: []const Value, full: []const u8) !void {
    var policy: Policy = .{ .single = true, .tools = tools };
    var sent: std.ArrayList(u8) = .empty;
    for (full) |b| try sent.appendSlice(a, try policy.text(a, &.{b}));
    try sent.appendSlice(a, policy.flush());
    const parsed = try parse(a, full, tools, 1);
    const whole = try policy.parsedContent(a, parsed.content);
    try std.testing.expectEqualStrings(whole, sent.items);
}

test "single-call streamed prose keeps rejected envelopes the whole reply keeps (#1396 case 1)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const forms = [_][]const u8{
        "Before.<foo:tool_call>{\"name\":\"other\",\"arguments\":{}}</foo:tool_call>Tail.",
        "Before.<TOOL_CALL>{\"name\":\"other\",\"arguments\":{}}</TOOL_CALL>Tail.",
        "Before.<Tool_Call>{\"name\":\"other\",\"arguments\":{}}</Tool_Call>Tail.",
        "Before.<tool_call>{invalid}</tool_call>Tail.",
    };
    for (forms) |text| try expectSingleParity(a, tools, text);
}

fn expectChatParity(cut: []const u8, calls: usize, arguments: []const u8, content: []const u8, extra: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try TestText.encode(undefined, a, cut, false);
    const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}{s}", .{ write_head, cut.len, extra });
    for ([_]bool{ false, true }) |batched| {
        var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
        const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
        const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(streamed.opened and streamed.status == null);
        try std.testing.expectEqualStrings(whole.content(), streamed.content());
        for ([_]@TypeOf(whole){ whole, streamed }) |seen| {
            try std.testing.expectEqual(calls, seen.calls);
            try std.testing.expectEqualStrings(arguments, seen.arguments[0..seen.arguments_len]);
            if (content.len > 0) try std.testing.expectEqualStrings(content, seen.content());
            try std.testing.expectEqual(.length, seen.finish.?);
        }
    }
}

test "single-call max_tokens keeps rejected envelopes and trailing whitespace like the whole reply (#1396 case 2)" {
    const kept = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>Between.";
    try expectChatParity(kept, 0, "", kept, ",\"parallel_tool_calls\":false");
    try expectChatParity("Writing it.\n" ++ opened, 0, "", "", ",\"parallel_tool_calls\":false");
    try expectChatParity("Writing it.\n" ++ glm_opened, 0, "", "", ",\"parallel_tool_calls\":false");
}

const dsml_body = "{\"model\":\"test-model\",\"max_tokens\":2000,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
const dsml_spellings = [_]struct { prefix: []const u8 }{
    .{ .prefix = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"q\" string=\"true\">" },
    .{ .prefix = "<｜DSML｜ calls><｜DSML｜ invoke name=\"lookup\"><｜DSML｜ parameter name=\"q\" string=\"true\">" },
};

test "single-call unclosed DSML trailing whitespace matches the whole reply (#1396 case 4)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = dsml_body ++ ",\"parallel_tool_calls\":false";
    inline for (dsml_spellings) |sp| {
        const raw = try std.mem.concat(a, u8, &.{ sp.prefix, "x   \n" });
        const tokens = try TestText.encode(undefined, a, raw, false);
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
            const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
            try std.testing.expectEqual(@as(?u16, 200), whole.status);
            try std.testing.expect(streamed.opened and streamed.status == null);
            try std.testing.expectEqual(@as(usize, 0), whole.calls);
            try std.testing.expectEqualStrings(whole.content(), streamed.content());
        }
    }
}

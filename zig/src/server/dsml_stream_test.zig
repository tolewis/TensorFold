//! DeepSeek-V4.1's DSML tool calls stream as they decode. Real token sequences (the DeepSeek-V4.1
//! tokenizer, testdata/dsml_stream.json) run through the chat route and the Messages route with a CPU fake engine. Each
//! stream's argument pieces join to the whole reply's arguments byte for byte, in order, sent once; a stream without
//! tools, or with the one-call policy, is the same as before the patch event for event (testdata/dsml_stream_golden.json,
//! recorded without the patch).
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const tokenizer = @import("tokenizer");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic = @import("anthropic.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const fixture_text = @embedFile("testdata/dsml_stream.json");
const golden_text = @embedFile("testdata/dsml_stream_golden.json");

/// The fixture's tokenizer (only the cases' tokens, under their real ids) behind the server's text seam; the
/// rendered prompt is a placeholder the fake engine does not read.
const FixtureText = struct {
    tok: tokenizer.Tokenizer,
    eos: [1]u32,

    fn text(t: *FixtureText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn self(ctx: *anyopaque) *FixtureText {
        return @ptrCast(@alignCast(ctx));
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return self(ctx).tok.decode(a, ids, false) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else a.dupe(u8, "\u{fffd}");
    }

    fn tokenId(ctx: *anyopaque, piece: []const u8) ?u32 {
        return self(ctx).tok.specialTokenId(piece);
    }

    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return a.dupe(u8, self(ctx).tok.id_to_token.get(id) orelse "");
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 129280;
    }

    fn eosIds(ctx: *anyopaque) []const u32 {
        return &self(ctx).eos;
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that answers with a case's ids in rounds of `pattern` tokens (cycled), as a drafting engine commits them,
/// and ends as the ids end: with their end token, or at the length (a reply cut by max_tokens).
const ScriptEngine = struct {
    ids: []const u32,
    pattern: []const usize,
    cut: bool,

    fn engine(e: *ScriptEngine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 1 << 20 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *ScriptEngine = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var pos: usize = 0;
        var round: usize = 0;
        while (pos < e.ids.len) : (round += 1) {
            const end = @min(e.ids.len, pos + e.pattern[round % e.pattern.len]);
            sink.event(sink.ctx, id, &.{ .tokens = e.ids[pos..end] });
            pos = end;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (e.cut) .length else .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the route sent: each stream event's JSON, or the whole reply.
const Seen = struct {
    a: Allocator,
    events: std.ArrayList([]const u8) = .empty,
    done: bool = false,
    whole: ?json.Value = null,

    fn out(s: *Seen) openai.Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        const p = payload orelse {
            s.done = true;
            return;
        };
        s.events.append(s.a, json.stringify(s.a, p, .{}) catch return error.Closed) catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, _: u16, payload: json.Value) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.whole = payload;
    }
};

const Fixture = struct {
    tok: tokenizer.Tokenizer,
    eos: u32,
    cases: json.Value,

    fn load(a: Allocator) !Fixture {
        const root = (try json.parse(a, fixture_text)).ok;
        const tok = try tokenizer.parse(std.testing.allocator, try json.stringify(a, root.get("tokenizer").?, .{ .ascii = false }));
        return .{ .tok = tok, .eos = @intCast(root.get("eos").?.int64().?), .cases = root.get("cases").? };
    }

    fn ids(f: *const Fixture, a: Allocator, name: []const u8) ![]u32 {
        for (f.cases.array) |c| if (std.mem.eql(u8, c.get("name").?.string, name)) {
            const list = c.get("ids").?.array;
            const out = try a.alloc(u32, list.len);
            for (list, out) |v, *o| o.* = @intCast(v.int64().?);
            return out;
        };
        return error.NoSuchCase;
    }
};

/// A coding agent's tools as an OpenAI client offers them (WebFetch is not offered).
const tools_json =
    \\[{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}}}}},
    \\{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"timeout":{"type":"number"}}}}},
    \\{"type":"function","function":{"name":"TodoWrite","parameters":{"type":"object","properties":{"todos":{"type":"array"}}}}},
    \\{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}}}}}]
;

const Ask = struct {
    case: []const u8,
    stream: bool = true,
    tools: bool = true,
    single: bool = false, // parallel_tool_calls: false
    pattern: []const usize = &.{ 3, 1, 4, 1, 5, 2 },
};

/// One request of `ask` through the chat route (or `messages`'s raw Messages body through the Messages route).
fn chatRun(a: Allocator, f: *Fixture, ask: Ask) !Seen {
    const ids = try f.ids(a, ask.case);
    const cut = std.mem.eql(u8, ask.case, "cut");
    var text: FixtureText = .{ .tok = f.tok, .eos = .{f.eos} };
    var backend: ScriptEngine = .{ .ids = ids, .pattern = ask.pattern, .cut = cut };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const max = if (cut) ids.len else 4096;
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":{s},\"max_tokens\":{d}{s}{s}}}", .{
        if (ask.stream) "true" else "false",
        max,
        if (ask.tools) ",\"tools\":" ++ tools_json else "",
        if (ask.single) ",\"parallel_tool_calls\":false" else "",
    });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{ .a = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const Call = struct { id: []const u8, name: []const u8, arguments: std.ArrayList(u8) = .empty, pieces: usize = 0 };

/// A chat stream read as a client reads it, its order checked on the way: the role first, each call's header (index,
/// id, type, name, empty arguments) once and in index order, its pieces after it and before the next header, no text
/// between a call's header and its last piece (LiteLLM's Messages stream ends a tool_use block at a text delta), valid
/// UTF-8 each, and nothing after the chunk that carries the finish reason.
const Read = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,
    finish: ?[]const u8 = null,

    fn of(a: Allocator, seen: *const Seen) !Read {
        var r: Read = .{};
        try std.testing.expect(seen.done);
        var role = false;
        var text_after_header = false; // text came after the newest call's header
        for (seen.events.items) |text| {
            const ev = (try json.parse(a, text)).ok;
            const choices = ev.get("choices") orelse continue;
            if (choices.array.len == 0) continue; // a usage chunk
            try std.testing.expect(r.finish == null); // nothing after the finish
            const choice = choices.array[0];
            if (choice.get("finish_reason")) |fr| if (fr == .string) {
                r.finish = fr.string;
            };
            const delta = choice.get("delta") orelse continue;
            if (delta.get("role")) |_| role = true;
            const said = delta.strField("content") orelse "";
            const thought = delta.strField("reasoning_content") orelse "";
            if (said.len > 0 or thought.len > 0 or delta.get("tool_calls") != null) try std.testing.expect(role);
            try r.content.appendSlice(a, said);
            try r.reasoning.appendSlice(a, thought);
            if (r.calls.items.len > 0 and (said.len > 0 or thought.len > 0)) text_after_header = true;
            const tc = delta.get("tool_calls") orelse continue;
            if (r.finish != null) return error.CallsWithTheFinish;
            for (tc.array) |one| {
                const index: usize = @intCast(one.get("index").?.int64().?);
                const function = one.get("function").?;
                const piece = function.strField("arguments").?;
                try std.testing.expect(std.unicode.utf8ValidateSlice(piece));
                if (one.get("id")) |id| {
                    try std.testing.expectEqual(r.calls.items.len, index); // a new call, in order, once
                    try std.testing.expectEqualStrings("function", one.strField("type").?);
                    try std.testing.expectEqualStrings("", piece);
                    try r.calls.append(a, .{ .id = id.string, .name = function.strField("name").? });
                    text_after_header = false;
                    continue;
                }
                try std.testing.expectEqual(r.calls.items.len -| 1, index); // pieces only for the call being written
                if (text_after_header) return error.PieceAfterText; // a client ended this call at that text
                const c = &r.calls.items[index];
                try c.arguments.appendSlice(a, piece);
                c.pieces += 1;
            }
        }
        return r;
    }
};

/// The whole reply's message: content (null as ""), reasoning, calls and finish reason.
fn wholeMessage(seen: *const Seen) struct { content: []const u8, reasoning: []const u8, calls: []const json.Value, finish: []const u8 } {
    const choice = seen.whole.?.get("choices").?.array[0];
    const message = choice.get("message").?;
    return .{
        .content = message.strField("content") orelse "",
        .reasoning = message.strField("reasoning_content") orelse "",
        .calls = if (message.get("tool_calls")) |c| c.array else &.{},
        .finish = choice.strField("finish_reason").?,
    };
}

/// The stream of `case` at each round pattern against the whole reply: the same calls, names and arguments byte for byte,
/// call i's arguments in at least `min_pieces[i]` pieces; the same content, reasoning and finish reason.
fn expectStreamed(case: []const u8, min_pieces: []const usize) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const whole_seen = try chatRun(a, &f, .{ .case = case, .stream = false });
    const whole = wholeMessage(&whole_seen);
    try std.testing.expect(whole.calls.len > 0);
    try std.testing.expectEqualStrings("tool_calls", whole.finish);
    for ([_][]const usize{ &.{1}, &.{ 3, 1, 4, 1, 5, 2 }, &.{64} }) |pattern| {
        const seen = try chatRun(a, &f, .{ .case = case, .pattern = pattern });
        const r = try Read.of(a, &seen);
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        try std.testing.expectEqualStrings(whole.content, r.content.items);
        try std.testing.expectEqualStrings(whole.reasoning, r.reasoning.items);
        try std.testing.expectEqual(whole.calls.len, r.calls.items.len);
        for (whole.calls, r.calls.items, min_pieces) |w, s, least| {
            const function = w.get("function").?;
            try std.testing.expectEqualStrings(function.strField("name").?, s.name);
            try std.testing.expectEqualStrings(function.strField("arguments").?, s.arguments.items);
            if (pattern.len > 1 or pattern[0] == 1) try std.testing.expect(s.pieces >= least);
        }
    }
}

test "a long Write streams its arguments as they decode, the parse's bytes joined (quotes, backslashes, CRLF, split characters)" {
    try expectStreamed("write", &.{40});
}

test "an Edit streams its strings and sends its boolean when the parameter closes, as the parse writes it" {
    try expectStreamed("edit", &.{6});
}

test "two calls stream one after the other: a Bash command with a number, then a TodoWrite whose array the parse re-writes" {
    // the TodoWrite array goes out whole as its parameter closes, then the call's brace: two pieces
    try expectStreamed("two_calls", &.{ 6, 2 });
}

test "a call to a tool the request did not offer streams as the parse returns it, under its own name" {
    try expectStreamed("unoffered", &.{ 4, 4 });
}

test "DSML written in the reasoning, or quoted in the answer, is no call: no tool delta, the same content" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const whole = wholeMessage(&try chatRun(a, &f, .{ .case = "quoted", .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    try std.testing.expectEqualStrings("stop", whole.finish);
    const r = try Read.of(a, &try chatRun(a, &f, .{ .case = "quoted", .pattern = &.{1} }));
    try std.testing.expectEqual(@as(usize, 0), r.calls.items.len);
    try std.testing.expectEqualStrings(whole.finish, r.finish.?);
    try std.testing.expectEqualStrings(whole.content, r.content.items);
    try std.testing.expectEqualStrings(whole.reasoning, r.reasoning.items);
}

test "a call cut by max_tokens ends as the whole reply reports it: length, its content, no brace for the open call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const whole = wholeMessage(&try chatRun(a, &f, .{ .case = "cut", .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len); // the parse rejects a block without its closer
    try std.testing.expectEqualStrings("length", whole.finish);
    const full = wholeMessage(&try chatRun(a, &f, .{ .case = "write", .stream = false })).calls[0].get("function").?.strField("arguments").?;
    for ([_][]const usize{ &.{1}, &.{ 3, 1, 4, 1, 5, 2 } }) |pattern| {
        const r = try Read.of(a, &try chatRun(a, &f, .{ .case = "cut", .pattern = pattern }));
        try std.testing.expectEqualStrings("length", r.finish.?);
        try std.testing.expectEqualStrings(whole.content, r.content.items);
        // the call the reply began went out as it decoded and stays open: a prefix of the whole call's arguments
        try std.testing.expectEqual(@as(usize, 1), r.calls.items.len);
        const sent = r.calls.items[0].arguments.items;
        try std.testing.expect(sent.len > 100 and sent.len < full.len and std.mem.startsWith(u8, full, sent));
        try std.testing.expect((try json.parse(a, sent)) == .err);
    }
}

/// A stream event without what changes between runs: the reply's id and time, its runtime block, the call ids.
fn normal(a: Allocator, text: []const u8) ![]const u8 {
    const v = (try json.parse(a, text)).ok;
    for ([_][]const u8{ "id", "created", "tensorfold", "speculative" }) |k| _ = v.object.orderedRemove(k);
    if (v.get("choices")) |cs| for (cs.array) |c| if (c.get("delta")) |d| if (d.get("tool_calls")) |tc| for (tc.array) |one| {
        if (one.get("id") != null) try one.object.put(a, "id", .{ .string = "call_X" });
    };
    return json.stringify(a, v, .{});
}

/// The streams the patch must leave as they were: no tools (a reply with DSML in it, and one with DSML only quoted),
/// tools without a call, and the one-call policy (its calls go out when the reply ends).
pub const golden_asks = [_]struct { key: []const u8, ask: Ask }{
    .{ .key = "no_tools_write", .ask = .{ .case = "write", .tools = false, .pattern = &.{ 29, 61 } } },
    .{ .key = "no_tools_two_calls", .ask = .{ .case = "two_calls", .tools = false, .pattern = &.{ 7, 13 } } },
    .{ .key = "no_tools_quoted", .ask = .{ .case = "quoted", .tools = false } },
    .{ .key = "tools_quoted", .ask = .{ .case = "quoted" } },
    .{ .key = "single_unoffered", .ask = .{ .case = "unoffered", .single = true } },
    .{ .key = "single_two_calls_by_token", .ask = .{ .case = "two_calls", .single = true, .pattern = &.{1} } },
};

/// The golden file's text: each golden ask's normalized events (how testdata/dsml_stream_golden.json was recorded).
pub fn goldenJson(a: Allocator) ![]const u8 {
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const o = try json.newObject(a);
    for (golden_asks) |g| {
        const seen = try chatRun(a, &f, g.ask);
        const list = try a.alloc(json.Value, seen.events.items.len);
        for (seen.events.items, list) |e, *slot| slot.* = .{ .string = try normal(a, e) };
        try o.put(a, g.key, .{ .array = list });
    }
    return json.stringify(a, .{ .object = o }, .{ .ascii = true });
}

test "without tools, without a call, or with the one-call policy, a stream is event for event what it was before the patch" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const golden = (try json.parse(a, golden_text)).ok;
    for (golden_asks) |g| {
        const want = golden.get(g.key).?.array;
        const seen = try chatRun(a, &f, g.ask);
        try std.testing.expect(seen.done);
        try std.testing.expectEqual(want.len, seen.events.items.len);
        for (want, seen.events.items) |w, e| try std.testing.expectEqualStrings(w.string, try normal(a, e));
    }
}

/// Drains a socket into `out` until its other end closes.
fn drain(fd: std.posix.fd_t, out: *std.ArrayList(u8)) void {
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.posix.read(fd, &buf) catch return;
        if (n == 0) return;
        out.appendSlice(std.heap.page_allocator, buf[0..n]) catch return;
    }
}

/// `body` (a Messages request) through the engine's /v1/messages route, as a client reads its bytes.
fn messagesRun(a: Allocator, f: *Fixture, case: []const u8, body: []const u8) ![]const u8 {
    const ids = try f.ids(a, case);
    var text: FixtureText = .{ .tok = f.tok, .eos = .{f.eos} };
    var backend: ScriptEngine = .{ .ids = ids, .pattern = &.{ 3, 1, 4, 1, 5, 2 }, .cut = false };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM, 0, &fds)) != .SUCCESS) return error.SocketPair;
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(std.heap.page_allocator);
    const reader = try std.Thread.spawn(.{}, drain, .{ fds[1], &got });
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{ .fd = fds[0], .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .headers = &headers, .path = "/v1/messages", .method = "POST", .version = "HTTP/1.1" };
    anthropic.post(srv, &conn, a);
    _ = std.posix.system.close(fds[0]);
    reader.join();
    _ = std.posix.system.close(fds[1]);
    return a.dupe(u8, got.items);
}

test "the engine's Messages route streams a DSML call as input_json_delta pieces that join to the chat reply's arguments" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.load(a);
    defer f.tok.deinit();
    const want = wholeMessage(&try chatRun(a, &f, .{ .case = "two_calls", .stream = false })).calls;
    const tools =
        \\"tools":[{"name":"Bash","input_schema":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"timeout":{"type":"number"}}}},
        \\{"name":"TodoWrite","input_schema":{"type":"object","properties":{"todos":{"type":"array"}}}}]
    ;
    const body = "{\"model\":\"test-model\",\"max_tokens\":4096,\"stream\":true,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}," ++ tools ++ ",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    const bytes = try messagesRun(a, &f, "two_calls", body);
    const head = std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.NoHead;
    try std.testing.expect(std.mem.startsWith(u8, bytes, "HTTP/1.1 200"));
    // the tool_use blocks in order, each one's partial_json pieces joined
    var names: std.ArrayList([]const u8) = .empty;
    var joined: std.ArrayList(std.ArrayList(u8)) = .empty;
    var pieces: std.ArrayList(usize) = .empty;
    var blocks: std.AutoHashMapUnmanaged(i64, usize) = .empty;
    var stop: []const u8 = "";
    var frames = std.mem.splitSequence(u8, bytes[head + 4 ..], "\n\n");
    while (frames.next()) |frame| {
        const at = std.mem.indexOf(u8, frame, "data: ") orelse continue;
        const ev = (try json.parse(a, frame[at + 6 ..])).ok;
        const kind = ev.strField("type") orelse continue;
        if (std.mem.eql(u8, kind, "content_block_start")) {
            const cb = ev.get("content_block").?;
            if (!std.mem.eql(u8, cb.strField("type").?, "tool_use")) continue;
            try blocks.put(a, ev.get("index").?.int64().?, names.items.len);
            try names.append(a, cb.strField("name").?);
            try joined.append(a, .empty);
            try pieces.append(a, 0);
        } else if (std.mem.eql(u8, kind, "content_block_delta")) {
            const d = ev.get("delta").?;
            if (!std.mem.eql(u8, d.strField("type").?, "input_json_delta")) continue;
            const i = blocks.get(ev.get("index").?.int64().?).?;
            try joined.items[i].appendSlice(a, d.strField("partial_json").?);
            pieces.items[i] += 1;
        } else if (std.mem.eql(u8, kind, "message_delta")) stop = ev.get("delta").?.strField("stop_reason").?;
    }
    try std.testing.expectEqualStrings("tool_use", stop);
    try std.testing.expectEqual(want.len, names.items.len);
    for (want, names.items, joined.items, pieces.items, [_]usize{ 6, 2 }) |w, name, j, n, least| {
        try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, name);
        try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, j.items);
        try std.testing.expect(n >= least);
    }
}

/// Bytes as tokens, for replies the fixture does not hold; with `eos`, id 256 is the end token.
const ByteText = struct {
    eos: bool = false,

    fn text(t: *ByteText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = FixtureText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = FixtureText.render, .template_source = FixtureText.templateSource } };
    }

    /// The bytes, a character cut short at the end as U+FFFD (the text seam's contract, as the real decoder keeps it).
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @intCast(id);
        var back: usize = 0;
        while (back < @min(4, out.len)) {
            back += 1;
            const lead = out[out.len - back];
            if (lead & 0xc0 == 0x80) continue;
            const n = std.unicode.utf8ByteSequenceLength(lead) catch break;
            if (n > back) return std.mem.concat(a, u8, &.{ out[0 .. out.len - back], "\u{fffd}" });
            break;
        }
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 257;
    }

    fn eosIds(ctx: *anyopaque) []const u32 {
        const t: *ByteText = @ptrCast(@alignCast(ctx));
        return if (t.eos) &.{256} else &.{};
    }
};

/// `answer` (no thinking) through the chat route with the tools above, whole or streamed in rounds of 5 bytes.
fn bytesRun(a: Allocator, answer: []const u8, stream: bool) !Seen {
    const ids = try a.alloc(u32, answer.len);
    for (answer, ids) |b, *id| id.* = b;
    var text: ByteText = .{};
    var backend: ScriptEngine = .{ .ids = ids, .pattern = &.{5}, .cut = false };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":", if (stream) "true" else "false", ",\"max_tokens\":4096,\"tools\":" ++ tools_json ++ "}" });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{ .a = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

test "a call in another form after the DSML block goes out when the reply ends, under the next index" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = "\u{ff5c}DSML\u{ff5c}";
    const answer = "Listing it.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"Bash\">\n<" ++ D ++ " parameter name=\"command\" string=\"true\">ls -la \"/tmp\"</" ++ D ++
        " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>\n<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/etc/hosts\"}}</tool_call>";
    const whole = wholeMessage(&try bytesRun(a, answer, false));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    const r = try Read.of(a, &try bytesRun(a, answer, true));
    try std.testing.expectEqualStrings(whole.finish, r.finish.?);
    // the stream shows the newline between the two forms as prose, the whole reply strips it (as before the patch)
    try std.testing.expectEqualStrings(whole.content, @import("reply_text.zig").pyStrip(r.content.items));
    try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
    for (whole.calls, r.calls.items) |w, s| {
        try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, s.name);
        try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
    }
    try std.testing.expect(r.calls.items[0].pieces >= 3); // the DSML call as it decoded
    try std.testing.expectEqual(@as(usize, 1), r.calls.items[1].pieces); // the other one whole, at the end
}

/// How `bytesAsk` runs a reply: its rounds, thinking off, cut by max_tokens at its end, its end token, a stop string.
const Bytes = struct { pattern: []const usize = &.{5}, stream: bool = true, cut: bool = false, eos: bool = false, stop: ?[]const u8 = null };

/// `answer` (no thinking) through the chat route with the tools above, as `o` says.
fn bytesAsk(a: Allocator, answer: []const u8, o: Bytes) !Seen {
    const ids = try a.alloc(u32, answer.len + @intFromBool(o.eos));
    for (answer, ids[0..answer.len]) |b, *id| id.* = b;
    if (o.eos) ids[answer.len] = 256;
    var text: ByteText = .{ .eos = o.eos };
    var backend: ScriptEngine = .{ .ids = ids, .pattern = o.pattern, .cut = o.cut };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const stop = if (o.stop) |st| try std.fmt.allocPrint(a, ",\"stop\":[{s}]", .{try json.quote(a, st, .{})}) else "";
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":{s},\"max_tokens\":{d}{s},\"tools\":{s}}}", .{
        if (o.stream) "true" else "false", if (o.cut) answer.len else 4096, stop, tools_json,
    });
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{ .a = a };
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const BD = "\u{ff5c}DSML\u{ff5c}";
const b_open = "\n\n<" ++ BD ++ " calls>\n";
const b_close = "</" ++ BD ++ " calls>";

fn bInvoke(comptime name: []const u8, comptime params: []const u8) []const u8 {
    return "<" ++ BD ++ " invoke name=\"" ++ name ++ "\">\n" ++ params ++ "</" ++ BD ++ " invoke>\n";
}

fn bParam(comptime name: []const u8, comptime value: []const u8) []const u8 {
    return "<" ++ BD ++ " parameter name=\"" ++ name ++ "\" string=\"true\">" ++ value ++ "</" ++ BD ++ " parameter>\n";
}

/// A stream that ends inside a block (the cases below) against the whole reply: the same finish reason and content,
/// no call in the whole reply, and in the stream `open` headers whose arguments are not JSON (a client cannot read
/// them as a whole call). Returns the stream as read.
fn expectCutBlock(a: Allocator, answer: []const u8, o: Bytes, open: usize) !Read {
    var whole_o = o;
    whole_o.stream = false;
    const whole = wholeMessage(&try bytesAsk(a, answer, whole_o));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    const r = try Read.of(a, &try bytesAsk(a, answer, o));
    try std.testing.expectEqualStrings(whole.finish, r.finish.?);
    try std.testing.expectEqualStrings(whole.content, r.content.items);
    try std.testing.expectEqual(open, r.calls.items.len);
    for (r.calls.items) |c| try std.testing.expect((try json.parse(a, c.arguments.items)) == .err);
    return r;
}

test "thinking off, a Harmony marker in a string value and the reply cut inside it: no read past the text, the call stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the finish reads Harmony channels out of this text (content ""), so its parse no longer reads the streamed answer
    const answer = "Searching." ++ b_open ++ "<" ++ BD ++ " invoke name=\"Bash\">\n<" ++ BD ++ " parameter name=\"command\" string=\"true\">grep -n '<|channel|>' zig/src/server/reply_text.zig and some more words that the cut ends";
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .cut = true, .stream = false }));
    try std.testing.expectEqualStrings("length", whole.finish);
    try std.testing.expectEqualStrings("", whole.content);
    for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 } }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .cut = true, .pattern = p }));
        try std.testing.expectEqualStrings("length", r.finish.?);
        // the prose before the block streamed before the marker came, as without the patch
        try std.testing.expectEqualStrings("Searching.", r.content.items);
        try std.testing.expectEqual(@as(usize, 1), r.calls.items.len);
        try std.testing.expect(std.mem.startsWith(u8, "{\"command\":\"grep -n '<|channel|>'", r.calls.items[0].arguments.items));
        try std.testing.expect((try json.parse(a, r.calls.items[0].arguments.items)) == .err);
    }
    // a stop string and an end token inside the value end the same way
    const stopped = try Read.of(a, &try bytesAsk(a, answer ++ " STOPHERE more", .{ .stop = "STOPHERE" }));
    try std.testing.expectEqualStrings("stop", stopped.finish.?);
    try std.testing.expect(stopped.calls.items.len == 1 and (try json.parse(a, stopped.calls.items[0].arguments.items)) == .err);
    const ended = try Read.of(a, &try bytesAsk(a, answer, .{ .eos = true }));
    try std.testing.expectEqualStrings("stop", ended.finish.?);
    try std.testing.expect(ended.calls.items.len == 1 and (try json.parse(a, ended.calls.items[0].arguments.items)) == .err);
}

test "thinking off, a Harmony marker in a whole block: the parse takes no call, and the stream stops at the marker" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = comptime ("Searching." ++ b_open ++ bInvoke("Bash", bParam("command", "grep -n '<|channel|>' zig/src/server/reply_text.zig")) ++ b_close);
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .stream = false }));
    try std.testing.expectEqual(@as(usize, 0), whole.calls.len);
    try std.testing.expectEqualStrings("stop", whole.finish);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .pattern = p }));
        try std.testing.expectEqualStrings("stop", r.finish.?);
        // a round that holds the marker turns the streamer off before its call closes: no brace, no whole call
        try std.testing.expect(r.calls.items.len <= 1);
        for (r.calls.items) |c| try std.testing.expect((try json.parse(a, c.arguments.items)) == .err);
    }
}

test "prose after a block, in the round that closes it, goes out after the call's last piece" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = comptime ("Listing." ++ b_open ++ bInvoke("Bash", bParam("command", "ls -la")) ++ b_close ++ " Then I read it.");
    // two blocks with prose between them and after them, in rounds that hold a closer, prose and the next opener
    const two = comptime (one ++ b_open ++ bInvoke("Read", bParam("file_path", "/etc/hosts")) ++ b_close ++ " Done.");
    for ([_][]const u8{ one, two }, [_]usize{ 1, 2 }) |answer, n| {
        const whole = wholeMessage(&try bytesAsk(a, answer, .{ .stream = false }));
        try std.testing.expectEqual(n, whole.calls.len);
        try std.testing.expectEqualStrings("tool_calls", whole.finish);
        for ([_][]const usize{ &.{1}, &.{5}, &.{16}, &.{ 3, 4 }, &.{ 40, 7 }, &.{1 << 20} }) |p| {
            const r = try Read.of(a, &try bytesAsk(a, answer, .{ .pattern = p }));
            try std.testing.expectEqualStrings(whole.finish, r.finish.?);
            // the stream hides a block's leading blank line with the block (reply_text.calls), as without the patch
            try std.testing.expectEqualStrings(try std.mem.replaceOwned(u8, a, whole.content, "\n\n", ""), r.content.items);
            try std.testing.expectEqual(n, r.calls.items.len);
            for (whole.calls, r.calls.items) |w, s| {
                try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, s.name);
                try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
            }
        }
    }
}

test "a block cut in its second invoke sends no whole call: the first call's brace and the second call wait for the block's closer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const head = comptime ("Two reads." ++ b_open ++ bInvoke("Read", bParam("file_path", "/etc/hosts")) ++ "<" ++ BD ++ " invoke name=\"Write\">\n" ++
        bParam("file_path", "/tmp/a") ++ "<" ++ BD ++ " parameter name=\"content\" string=\"true\">line one\nline two");
    // cut by max_tokens, by a stop string, and by the end token: the parse takes no call from the block
    for ([_]Bytes{ .{ .cut = true }, .{ .cut = true, .pattern = &.{1} }, .{ .cut = true, .pattern = &.{1 << 20} }, .{ .stop = "STOPHERE" }, .{ .eos = true } }) |o| {
        const answer = if (o.stop != null) head ++ " STOPHERE and more" else head;
        const r = try expectCutBlock(a, answer, o, 1);
        // the first call went out as it decoded and stays open; the second never went out
        try std.testing.expectEqualStrings("Read", r.calls.items[0].name);
        try std.testing.expectEqualStrings("{\"file_path\":\"/etc/hosts\"", r.calls.items[0].arguments.items);
    }
    // the block whole: both calls, the second as the block closes
    const whole_text = comptime (head ++ "</" ++ BD ++ " parameter>\n</" ++ BD ++ " invoke>\n" ++ b_close);
    const whole = wholeMessage(&try bytesAsk(a, whole_text, .{ .stream = false }));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    for ([_][]const usize{ &.{1}, &.{5}, &.{ 3, 4 } }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, whole_text, .{ .pattern = p }));
        try std.testing.expectEqualStrings("tool_calls", r.finish.?);
        try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
        for (whole.calls, r.calls.items) |w, s| try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
        if (p[0] == 1) try std.testing.expect(r.calls.items[0].pieces >= 5); // the first call streamed
    }
    // a whole block, then a second block cut in its second invoke: the first block's call is whole and is the parse's
    const after = comptime ("First." ++ b_open ++ bInvoke("Bash", bParam("command", "ls")) ++ b_close ++ head["Two reads.".len..]);
    const w2 = wholeMessage(&try bytesAsk(a, after, .{ .cut = true, .stream = false }));
    try std.testing.expectEqual(@as(usize, 1), w2.calls.len);
    const r2 = try Read.of(a, &try bytesAsk(a, after, .{ .cut = true }));
    try std.testing.expectEqualStrings(w2.finish, r2.finish.?);
    try std.testing.expectEqualStrings(w2.content, @import("reply_text.zig").pyStrip(r2.content.items));
    try std.testing.expectEqual(@as(usize, 2), r2.calls.items.len);
    try std.testing.expectEqualStrings(w2.calls[0].get("function").?.strField("arguments").?, r2.calls.items[0].arguments.items);
    try std.testing.expect((try json.parse(a, r2.calls.items[1].arguments.items)) == .err);
}

test "an XML call after a DSML block that the end token leaves open goes out once, under the next index" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const answer = comptime ("Listing it." ++ b_open ++ bInvoke("Bash", bParam("command", "ls")) ++ b_close ++ "\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/etc/hosts\n</parameter>\n");
    const whole = wholeMessage(&try bytesAsk(a, answer, .{ .eos = true, .stream = false }));
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    for ([_][]const usize{ &.{1}, &.{5}, &.{1 << 20} }) |p| {
        const r = try Read.of(a, &try bytesAsk(a, answer, .{ .eos = true, .pattern = p }));
        try std.testing.expectEqualStrings(whole.finish, r.finish.?);
        try std.testing.expectEqual(@as(usize, 2), r.calls.items.len);
        for (whole.calls, r.calls.items) |w, s| {
            try std.testing.expectEqualStrings(w.get("function").?.strField("name").?, s.name);
            try std.testing.expectEqualStrings(w.get("function").?.strField("arguments").?, s.arguments.items);
        }
    }
}

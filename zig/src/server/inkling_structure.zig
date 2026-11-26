//! Inkling 2 chat: structured output framed in its reply blocks, and tool_choice as vLLM's Inkling renderer words it.
//! A reply is any number of thinking blocks, then a text block whose payload alone holds the constrained output, or
//! the tool blocks a required or named choice must call (vLLM parser/inkling.py adjust_request; renderers/
//! inkling_encoding.py _inkling2_tool_choice_instruction).
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const tool_specs = @import("tool_specs.zig");
const tool_grammar = @import("tool_grammar.zig");
const markers = @import("inkling_reply.zig").markers;
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The sampling field the chat route records the instruction in (not a request field: plan() writes it).
pub const instruction_field = "inkling_tool_instruction";

/// The most blank characters between a constrained JSON value's tokens (grammar.zig's BLANKS for plain JSON).
const value_blanks = 32;

/// Every framing token (vLLM INKLING_CONTROL_TOKENS): free text in a grammar never spells one.
const control = [_][]const u8{
    "<|endoftext|>",              "<|message_user|>",           "<|message_system|>",   "<|message_tool|>",
    "<|content_image|>",          "<|content_xml|>",            "<|content_audio_input|>", "<|audio_end|>",
    markers.model,                markers.text,                 markers.thinking,       markers.tool_json,
    markers.tool_text,            markers.tool_error,           markers.end_sampling,   markers.end,
};

/// vLLM's sentence for a tool_choice (null: none to add). `choice` is the request's tool_choice; `parallel` its
/// parallel_tool_calls (default true). The name is spelled as Python's json.dumps spells it (ASCII escapes).
pub fn instruction(a: Allocator, choice: ?Value, parallel: bool) Allocator.Error!?[]const u8 {
    const c = choice orelse return if (parallel) null else try a.dupe(u8, auto_single);
    switch (c) {
        .null => return if (parallel) null else try a.dupe(u8, auto_single),
        .string => |s| {
            if (std.mem.eql(u8, s, "required")) return if (parallel)
                try a.dupe(u8, "Call at least one of the provided functions in your next response.")
            else
                try a.dupe(u8, "Call exactly one of the provided functions in your next response.");
            if (std.mem.eql(u8, s, "auto") and !parallel) return try a.dupe(u8, auto_single);
            return null;
        },
        .object => {
            const kind = c.get("type") orelse return null;
            if (kind != .string or !std.mem.eql(u8, kind.string, "function")) return null;
            const function = c.get("function");
            const name_v = if (function) |f| (if (f == .object) f.get("name") else null) else c.get("name");
            const name = if (name_v) |n| (if (n == .string) n.string else return null) else return null;
            const quoted = try json.stringify(a, .{ .string = name }, .{ .ascii = true });
            return try std.fmt.allocPrint(a, "Call the function named {s}{s} in your next response.", .{ quoted, if (parallel) "" else " exactly once" });
        },
        else => return null,
    }
}

const auto_single = "If you call a function, call at most one function in your next response.";

/// The instruction a render takes: the chat route's recorded sentence, or (a raw body, as /tokenize renders it)
/// the one its tool_choice and parallel_tool_calls ask for.
pub fn instructionOf(a: Allocator, fields: Value) Allocator.Error!?[]const u8 {
    if (fields != .object) return null;
    if (fields.get(instruction_field)) |v| return if (v == .string) v.string else null;
    if (fields.get("messages") == null) return null; // a sampling object without the field: no instruction
    const p = fields.get("parallel_tool_calls");
    return instruction(a, fields.get("tool_choice"), !(p != null and p.? == .bool and !p.?.bool));
}

/// vLLM `_with_appended_text` on message 0 when it is a system or developer message: a copy of `messages` with the
/// sentence added, or null when message 0 is any other role (the sentence then joins the effort block).
pub fn withInstruction(a: Allocator, messages: Value, text: []const u8) Allocator.Error!?Value {
    if (messages != .array or messages.array.len == 0) return null;
    const first = messages.array[0];
    if (first != .object) return null;
    const role = first.get("role") orelse return null;
    if (role != .string or !(std.mem.eql(u8, role.string, "system") or std.mem.eql(u8, role.string, "developer"))) return null;
    const copy = try json.newObject(a);
    var it = first.object.iterator();
    while (it.next()) |e| try copy.put(a, e.key_ptr.*, e.value_ptr.*);
    const content = first.get("content");
    const joined: Value = if (content) |cv| switch (cv) {
        .string => |s| .{ .string = if (s.len > 0) try std.mem.concat(a, u8, &.{ s, "\n\n", text }) else text },
        .array => |parts| blk: {
            const part = try json.newObject(a);
            try part.put(a, "type", .{ .string = "text" });
            try part.put(a, "text", .{ .string = text });
            break :blk .{ .array = try std.mem.concat(a, Value, &.{ parts, &.{.{ .object = part }} }) };
        },
        .null => .{ .string = text },
        else => cv,
    } else .{ .string = text };
    try copy.put(a, "content", joined);
    const out = try a.dupe(Value, messages.array);
    out[0] = .{ .object = copy };
    return .{ .array = out };
}

fn node(a: Allocator, kind: []const u8, entries: []const struct { []const u8, Value }) Allocator.Error!Value {
    return tool_grammar.node(a, kind, entries);
}

/// A framing token by id (the tokenizer's own; a 400 when the checkpoint lacks it).
fn token(srv: *Server, cx: *Cx, spelled: []const u8) errors.Refused!Value {
    const id = srv.text.tokenId(spelled) orelse return cx.fail(.request, "structured output: the tokenizer has no {s} token", .{spelled});
    return node(cx.a, "token", &.{.{ "token", try json.intValue(cx.a, id) }});
}

fn constant(a: Allocator, text: []const u8) Allocator.Error!Value {
    return node(a, "const_string", &.{.{ "value", .{ .string = text } }});
}

fn schemaNode(a: Allocator, schema: Value) Allocator.Error!Value {
    return node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "style", .{ .string = "json" } }, .{ "max_whitespace_cnt", try json.intValue(a, value_blanks) } });
}

/// Zero or more thinking blocks: `<|message_model|><|content_thinking|>` free text `<|end_message|>`.
fn thinking(srv: *Server, cx: *Cx) errors.Refused!Value {
    const a = cx.a;
    var excludes: std.ArrayList(Value) = .empty;
    for (control) |c| if (srv.text.tokenId(c) != null) try excludes.append(a, .{ .string = c });
    const text = try node(a, "any_text", &.{.{ "excludes", .{ .array = excludes.items } }});
    const block = try node(a, "sequence", &.{.{ "elements", .{ .array = try a.dupe(Value, &.{ try token(srv, cx, markers.model), try token(srv, cx, markers.thinking), text, try token(srv, cx, markers.end) }) } }});
    return node(a, "star", &.{.{ "content", block }});
}

fn tag(a: Allocator, format: Value) Allocator.Error![]const u8 {
    return json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// The inner format a grammar request asks for, as a structural tag node.
fn inner(cx: *Cx, s: grammar.Spec) errors.Refused!Value {
    const a = cx.a;
    return switch (s.kind) {
        .json => schemaNode(a, try node(a, "object", &.{})),
        .json_schema => schemaNode(a, switch (try json.parse(a, s.text)) {
            .ok => |v| v,
            .err => return cx.fail(.request, "{s}: the schema is not JSON", .{s.field}),
        }),
        .regex => node(a, "regex", &.{.{ "pattern", .{ .string = s.text } }}),
        .grammar => node(a, "grammar", &.{.{ "grammar", .{ .string = s.text } }}),
        .choice => blk: {
            const list = switch (try json.parse(a, s.text)) {
                .ok => |v| v,
                .err => return cx.fail(.request, "{s}: the choices are not JSON", .{s.field}),
            };
            const items = list.items() orelse return cx.fail(.request, "{s} must be a list", .{s.field});
            const options = try a.alloc(Value, items.len);
            for (items, options) |it, *o| o.* = try constant(a, it.string);
            if (options.len == 1) break :blk options[0];
            break :blk node(a, "or", &.{.{ "elements", .{ .array = options } }});
        },
        .structural_tag => unreachable,
    };
}

/// The structural tag (xgrammar JSON) for a grammar request on an Inkling chat reply: thinking blocks, then one
/// text block holding the asked-for output, then the end of sampling (xgrammar's stop tokens).
pub fn framed(srv: *Server, cx: *Cx, s: grammar.Spec) errors.Refused![]const u8 {
    const a = cx.a;
    const elements = try a.dupe(Value, &.{ try thinking(srv, cx), try token(srv, cx, markers.model), try token(srv, cx, markers.text), try inner(cx, s), try token(srv, cx, markers.end) });
    return tag(a, try node(a, "sequence", &.{.{ "elements", .{ .array = elements } }}));
}

/// One tool block for `tool`: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":ARGS}<|end_message|>`,
/// ARGS in the tool's schema when strict (or `all`), else any object; the schema keeps its own root ($ref, $defs).
fn callBlock(srv: *Server, cx: *Cx, tool: Value, all: bool) errors.Refused!Value {
    const a = cx.a;
    const name = try tool_specs.toolName(a, tool);
    const quoted = try json.stringify(a, .{ .string = name }, .{ .ascii = false });
    const args = if (all or tool_grammar.isStrict(tool)) try tool_grammar.parametersOf(a, tool) else try node(a, "object", &.{.{ "additionalProperties", .{ .bool = true } }});
    const elements = try a.dupe(Value, &.{
        try token(srv, cx, markers.model),
        try constant(a, name),
        try token(srv, cx, markers.tool_json),
        try constant(a, try std.mem.concat(a, u8, &.{ "{\"name\":", quoted, ",\"args\":" })),
        try schemaNode(a, args),
        try constant(a, "}"),
        try token(srv, cx, markers.end),
    });
    return node(a, "sequence", &.{.{ "elements", .{ .array = elements } }});
}

/// The structural tag for a reply that must call one of `tools` (a named choice offers one): thinking blocks, then
/// one call, or (`parallel`) one or more, then the end of sampling.
pub fn calls(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, all: bool) errors.Refused![]const u8 {
    const a = cx.a;
    const blocks = try a.alloc(Value, tools.len);
    for (tools, blocks) |tool, *b| b.* = try callBlock(srv, cx, tool, all);
    const one = if (blocks.len == 1) blocks[0] else try node(a, "or", &.{.{ "elements", .{ .array = blocks } }});
    const body = if (parallel) try node(a, "plus", &.{.{ "content", one }}) else one;
    return tag(a, try node(a, "sequence", &.{.{ "elements", .{ .array = try a.dupe(Value, &.{ try thinking(srv, cx), body }) } }}));
}

const testing = std.testing;

test "tool_choice sentences are vLLM's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("Call at least one of the provided functions in your next response.", (try instruction(a, .{ .string = "required" }, true)).?);
    try testing.expectEqualStrings("Call exactly one of the provided functions in your next response.", (try instruction(a, .{ .string = "required" }, false)).?);
    try testing.expectEqualStrings(auto_single, (try instruction(a, .{ .string = "auto" }, false)).?);
    try testing.expectEqual(@as(?[]const u8, null), try instruction(a, .{ .string = "auto" }, true));
    try testing.expectEqual(@as(?[]const u8, null), try instruction(a, null, true));
    try testing.expectEqual(@as(?[]const u8, null), try instruction(a, .{ .string = "none" }, false));
    const named = (try json.parse(a, "{\"type\":\"function\",\"function\":{\"name\":\"get_w\\u00e9ather\"}}")).ok;
    try testing.expectEqualStrings("Call the function named \"get_w\\u00e9ather\" in your next response.", (try instruction(a, named, true)).?);
    try testing.expectEqualStrings("Call the function named \"get_w\\u00e9ather\" exactly once in your next response.", (try instruction(a, named, false)).?);
}

test "message 0 takes the sentence as vLLM's _with_appended_text adds it" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"hi\"}]", "Be brief.\n\nX" },
        .{ "[{\"role\":\"developer\",\"content\":\"\"},{\"role\":\"user\",\"content\":\"hi\"}]", "X" },
        .{ "[{\"role\":\"system\",\"content\":null}]", "X" },
        .{ "[{\"role\":\"user\",\"content\":\"hi\"}]", null },
    };
    for (cases) |c| {
        const got = try withInstruction(a, (try json.parse(a, c[0])).ok, "X");
        if (c[1]) |want| try testing.expectEqualStrings(want, got.?.array[0].get("content").?.string) else try testing.expectEqual(@as(?Value, null), got);
    }
    const parts = (try withInstruction(a, (try json.parse(a, "[{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\"A\"}]}]")).ok, "X")).?;
    const list = parts.array[0].get("content").?.array;
    try testing.expectEqual(@as(usize, 2), list.len);
    try testing.expectEqualStrings("X", list[1].get("text").?.string);
}

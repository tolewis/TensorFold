//! TF_TOOL_CALL_GUARD=1: a chat request that offers no tools (none sent, or tool_choice "none") although its history
//! called them gets a closing system note that tools are unavailable, and any tool-call markup the model writes anyway
//! is hidden from the reply (tool_stream.Policy.guard) instead of reaching the client as text. A request that does offer
//! tools has its well-formed calls to tools it did not offer hidden (Policy.drop_rejected, tool_parse.Options); a
//! malformed call stays as text, and a reply that hiding would leave empty keeps its calls, since an empty reply reads
//! as a failed generation that agents retry blind. Off by default. Every rejected envelope is logged with its reason
//! (``tool call rejected from ID: REASON ['NAME'] ...``); TF_TOOL_CALL_DEBUG=1 adds its text.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The note closing the prompt of a guarded request.
pub const note = "Tools are not available for this reply. Answer in plain text from what you already have; do not write tool calls.";

var cached: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on
var cached_debug: std.atomic.Value(u8) = .init(0);

/// Whether TF_TOOL_CALL_GUARD is 1 (read once).
pub fn on() bool {
    return flag(&cached, "TF_TOOL_CALL_GUARD");
}

/// Tests set the guard on or off; ``null`` makes the next ``on`` read the environment again.
pub fn setForTest(v: ?bool) void {
    cached.store(if (v) |set| (if (set) 2 else 1) else 0, .monotonic);
}

/// Whether TF_TOOL_CALL_DEBUG is 1 (read once): each rejected tool envelope's text goes to the log beside its reason.
pub fn debug() bool {
    return flag(&cached_debug, "TF_TOOL_CALL_DEBUG");
}

fn flag(slot: *std.atomic.Value(u8), comptime name: [:0]const u8) bool {
    const v = slot.load(.monotonic);
    if (v != 0) return v == 2;
    const set = if (std.c.getenv(name)) |s| std.mem.eql(u8, std.mem.span(s), "1") else false;
    slot.store(if (set) 2 else 1, .monotonic);
    return set;
}

/// `s` on one log line in `buf`: newline, tab, CR, backslash and other control bytes escaped, and when it does not fit,
/// its head, "...", and the count of bytes left out.
pub fn escaped(buf: []u8, s: []const u8) []const u8 {
    const room = buf.len -| 32; // the tail note's room
    var n: usize = 0;
    for (s, 0..) |ch, i| {
        var tmp: [4]u8 = undefined;
        const piece: []const u8 = switch (ch) {
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            '\\' => "\\\\",
            0...8, 11, 12, 14...31, 127 => std.fmt.bufPrint(&tmp, "\\x{x:0>2}", .{ch}) catch unreachable,
            else => blk: {
                tmp[0] = ch;
                break :blk tmp[0..1];
            },
        };
        if (n + piece.len > room) {
            const tail = std.fmt.bufPrint(buf[n..], "...(+{d} bytes)", .{s.len - i}) catch "";
            return buf[0 .. n + tail.len];
        }
        @memcpy(buf[n..][0..piece.len], piece);
        n += piece.len;
    }
    return buf[0..n];
}

/// Whether `messages` (the normalized list) holds a tool result or an assistant turn with tool calls.
pub fn usedTools(messages: Value) bool {
    if (messages != .array) return false;
    for (messages.array) |m| {
        if (m != .object) continue;
        if (m.object.get("role")) |r| if (r == .string and std.mem.eql(u8, r.string, "tool")) return true;
        if (m.object.get("tool_calls")) |c| if (c.truthy()) return true;
    }
    return false;
}

/// `messages` with the guard's system note appended.
pub fn withNote(a: Allocator, messages: Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "system" });
    try o.put(a, "content", .{ .string = note });
    const list = try a.alloc(Value, messages.array.len + 1);
    @memcpy(list[0..messages.array.len], messages.array);
    list[messages.array.len] = .{ .object = o };
    return .{ .array = list };
}

test "usedTools: a tool result or assistant tool_calls, not plain chat" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello\"}]")).ok;
    try std.testing.expect(!usedTools(plain));
    const called = (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\"}]}]")).ok;
    try std.testing.expect(usedTools(called));
    const result = (try json.parseText(a, "[{\"role\":\"tool\",\"content\":\"r\"}]")).ok;
    try std.testing.expect(usedTools(result));
    const noted = try withNote(a, called);
    try std.testing.expectEqual(@as(usize, 3), noted.array.len);
    try std.testing.expectEqualStrings("system", noted.array[2].object.get("role").?.string);
}

test "Policy.guard: whole and unclosed call envelopes leave the reply, prose stays, streamed or whole" {
    const ts = @import("tool_stream.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off: ts.Policy = .{};
    const call = "<tool_call>session_search<arg_key>after</arg_key><arg_value>7d</arg_value></tool_call>";
    try std.testing.expectEqualStrings("Done." ++ call, try off.content(a, "Done." ++ call));
    const guarded: ts.Policy = .{ .guard = true };
    try std.testing.expectEqualStrings("Done. More.", try guarded.content(a, "Done." ++ call ++ call ++ " More."));
    try std.testing.expectEqualStrings("Summary", try guarded.content(a, "Summary<tool_call>session_search<arg_key>q"));
    try std.testing.expectEqualStrings("a < b and <b>bold</b>", try guarded.content(a, "a < b and <b>bold</b>"));
    // streamed: split across deltas, the markup never reaches the text
    var p: ts.Policy = .{ .guard = true };
    var shown: std.ArrayList(u8) = .empty;
    for ([_][]const u8{ "Here", " it is <tool", "_call>session_search<arg_key>a</arg_key>", "<arg_value>1</arg_value></tool_", "call><tool_call>x</tool_call>", " end" }) |d| try shown.appendSlice(a, try p.text(a, d));
    try shown.appendSlice(a, p.flush());
    try std.testing.expectEqualStrings("Here it is  end", shown.items);
    try std.testing.expect(p.filter.hid);
}

test "hideCalls: drop_rejected's filter keeps prose around an unoffered call" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hide = @import("tool_policy.zig").hideCalls;
    try std.testing.expectEqualStrings("Let me check. ok", try hide(a, "Let me check.<tool_call>session_search<arg_key>q</arg_key><arg_value>x</arg_value></tool_call> ok"));
    try std.testing.expectEqualStrings("plain", try hide(a, "plain"));
}

test "parse: each rejected envelope carries its reason and name; hide_unoffered drops only well-formed unoffered calls" {
    const tp = @import("tool_parse.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"shell_command\",\"parameters\":{\"type\":\"object\"}}}]")).ok.array;
    const unoffered = "<tool_call>drive_preview<arg_key>id</arg_key><arg_value>7</arg_value></tool_call>";
    // a dropped key name is not a rejection outside single-call mode: the call parses with no arguments
    const missing_key = "<tool_call>shell_command<arg_key><arg_value>echo hi</arg_value></tool_call>";
    const multiline_name = "<tool_call>shell\ncommand<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>";
    const nameless = "<tool_call>{\"arguments\":{\"command\":\"ls\"}}</tool_call>";
    const empty_name = "<tool_call><arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>";
    const good = "<tool_call>shell_command<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>";
    const Case = struct { text: []const u8, reason: tp.Reason, name: []const u8 };
    for ([_]Case{
        .{ .text = unoffered, .reason = .unknown_tool_name, .name = "drive_preview" },
        .{ .text = multiline_name, .reason = .malformed, .name = "shell" },
        .{ .text = nameless, .reason = .malformed, .name = "" },
        .{ .text = empty_name, .reason = .malformed, .name = "" },
    }) |c| {
        const r = try tp.parse(a, try std.mem.concat(a, u8, &.{ "Prose. ", c.text }), tools, null);
        try std.testing.expect(r.calls == null); // a nameless or unoffered call is never a tool_calls entry
        try std.testing.expectEqual(@as(usize, 1), r.rejections.len);
        try std.testing.expectEqual(c.reason, r.rejections[0].reason);
        try std.testing.expectEqualStrings(c.name, r.rejections[0].name);
        try std.testing.expectEqualStrings(c.text, r.rejections[0].envelope);
        try std.testing.expect(!r.rejections[0].hidden);
    }
    const lenient = try tp.parse(a, missing_key, tools, null);
    try std.testing.expectEqualStrings("{}", lenient.calls.?[0].get("function").?.get("arguments").?.string);
    try std.testing.expectEqual(@as(usize, 1), (try tp.parse(a, missing_key, tools, 1)).rejections.len);
    // a mixed batch under the guard: the good call parses, the unoffered one leaves, the malformed one stays, prose stays
    const hid = try tp.parseWith(a, "Let me look." ++ good ++ unoffered ++ nameless, tools, null, .{ .hide_unoffered = true });
    try std.testing.expectEqual(@as(usize, 1), hid.calls.?.len);
    try std.testing.expectEqualStrings("Let me look." ++ nameless, hid.content);
    try std.testing.expectEqual(@as(usize, 2), hid.rejections.len);
    try std.testing.expect(hid.rejections[0].hidden and !hid.rejections[1].hidden);
}

test "parse: a call to Hermes' offered tool_call bridge is that tool's call, whatever its arguments hold" {
    const tp = @import("tool_parse.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"tool_call\",\"parameters\":{\"type\":\"object\",\"properties\":{\"calls\":{\"type\":\"array\"}}}}}]")).ok.array;
    // calls[0] has no name: the client's tool refuses it ("tool_call calls[0] requires a 'name'"); the server forwards it
    const r = try tp.parse(a, "<tool_call>tool_call<arg_key>calls</arg_key><arg_value>[{\"arguments\":{}}]</arg_value></tool_call>", tools, null);
    try std.testing.expectEqualStrings("tool_call", r.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqual(@as(usize, 0), r.rejections.len);
}

test "escaped: one log line, control bytes escaped, a long envelope cut with its count" {
    var wide: [64]u8 = undefined;
    try std.testing.expectEqualStrings("a\\nb\\tc\\\\\\x01", escaped(&wide, "a\nb\tc\\\x01"));
    var buf: [40]u8 = undefined; // 8 bytes of text beside the 32 kept for the note
    const long: [100]u8 = @splat('x');
    try std.testing.expectEqualStrings("xxxxxxxx...(+92 bytes)", escaped(&buf, &long));
}

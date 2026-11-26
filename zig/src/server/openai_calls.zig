//! Finished calls from native parsers or the shared markup parser, under the request's tool policy, with their ids.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const tool_parse = @import("tool_parse.zig");
const tool_policy = @import("tool_policy.zig");
const tool_stream = @import("tool_stream.zig");
const tool_guard = @import("tool_guard.zig");
const cortheon = @import("cortheon/cortheon.zig");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// The one choice a chat reply has.
const choice: u32 = 0;

pub fn attach(r: anytype, reply: *chat.Reply) Allocator.Error!?[]Value {
    const tools = r.plan.input.tools;
    if (tools.len == 0) {
        if (r.plan.policy.guard) {
            const shown = try r.plan.policy.content(r.a, reply.content);
            if (shown.len != reply.content.len) log.line("tool-call guard: hid call markup from {s}, a reply to a request without tools", .{r.id});
            reply.content = shown;
        }
        return null;
    }
    if (reply.native_tool_format) {
        const found = reply.native_calls;
        const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse return null;
        reply.content = try r.plan.policy.parsedContent(r.a, reply.content);
        reply.finish_reason = "tool_calls";
        reply.tool_calls_streamed = false;
        // calls the model wrote no ids for (Inkling 2) take the request's, as parsed calls do; Kimi's keep their own
        for (calls, 0..) |c, i| if (c == .object and !c.object.contains("id")) try c.object.put(r.a, "id", .{ .string = try r.call_ids.id(r.a, choice, i) });
        return if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls;
    }
    // TF_TOOL_CALL_GUARD: a well-formed call to a tool the request did not offer leaves the reply; a malformed one stays
    var parsed = try tool_parse.parseWith(r.a, reply.content, tools, r.plan.policy.maxCalls(), .{ .hide_unoffered = r.plan.policy.drop_rejected });
    var hidden: usize = 0;
    for (parsed.rejections) |j| hidden += @intFromBool(j.hidden);
    // ...unless hiding would leave the reply with nothing in it: an empty reply reads as a failed generation, and an
    // agent retries it blind, to the same call hidden again
    const kept_whole = hidden > 0 and parsed.calls == null and parsed.content.len == 0;
    if (kept_whole) {
        parsed = try tool_parse.parse(r.a, reply.content, tools, r.plan.policy.maxCalls());
        hidden = 0;
    }
    logRejections(r, reply, parsed.rejections, tools.len, kept_whole);
    if (parsed.rejected_envelopes > 0) {
        r.srv.metrics.keptToolCalls(r.srv.io, parsed.kept_unoffered, parsed.kept_malformed);
        if (reply.runtime == .object) try reply.runtime.object.put(r.a, "kept_tool_calls", try keptBlock(r.a, parsed));
    }
    // max_tokens cut a call the stream had begun: the whole reply returns it cut too, as the stream did (#1238)
    const cut = if (!r.plan.policy.single and std.mem.eql(u8, reply.finish_reason, "length")) try tool_stream.cutCalls(r.a, reply.content, tools) else null;
    if (cut != null) reply.content = try @import("reply_text.zig").cutProse(r.a, reply.content);
    const found = cut orelse parsed.calls;
    // a call the end token cut before its closer (not max_tokens): the reply keeps it as text, the log says so
    if (cut == null and !std.mem.eql(u8, reply.finish_reason, "length")) if (tool_parse.openCall(parsed.content)) |at|
        logRejection(r, reply, "unclosed", tool_parse.leadName(parsed.content[at + "<tool_call>".len ..]), parsed.content[at..], tools.len, "kept as text");
    const dropped = hidden > 0;
    const base = parsed.content;
    if (dropped) log.line("tool-call guard: hid {d} call(s) to unoffered tools from {s}", .{ hidden, r.id });
    const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse {
        if (r.plan.policy.single or parsed.calls != null or dropped) reply.content = try r.plan.policy.parsedContent(r.a, base);
        return null;
    };
    if (cut != null) return try numbered(r, calls);
    reply.content = try r.plan.policy.parsedContent(r.a, base);
    reply.finish_reason = "tool_calls";
    if (r.plan.policy.single) reply.tool_calls_streamed = false;
    return try numbered(r, if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls);
}

/// Each call the client gets takes the request's id for its place in the reply (``ids.Calls``).
fn numbered(r: anytype, calls: []Value) Allocator.Error![]Value {
    for (calls, 0..) |c, i| if (c == .object) try c.object.put(r.a, "id", .{ .string = try r.call_ids.id(r.a, choice, i) });
    return calls;
}

/// A streamed call's opening delta takes the id ``attach`` gives the call at its index, so both replies agree.
pub fn streamed(r: anytype, delta: Value) Allocator.Error!void {
    const list = delta.get("tool_calls") orelse return;
    if (list != .array) return;
    for (list.array) |one| {
        if (one != .object or !one.object.contains("id")) continue;
        const at = std.math.cast(u64, (one.get("index") orelse continue).int64() orelse continue) orelse continue;
        try one.object.put(r.a, "id", .{ .string = try r.call_ids.id(r.a, choice, at) });
    }
}

/// The reply's `tensorfold.kept_tool_calls`: both counts, and the unoffered names the model wrote.
fn keptBlock(a: Allocator, parsed: tool_parse.Parsed) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "unoffered", try json.intValue(a, parsed.kept_unoffered));
    try o.put(a, "malformed", try json.intValue(a, parsed.kept_malformed));
    if (parsed.unoffered_names.len > 0) {
        const list = try a.alloc(Value, parsed.unoffered_names.len);
        for (parsed.unoffered_names, 0..) |name, i| list[i] = .{ .string = name };
        try o.put(a, "unoffered_names", .{ .array = list });
    }
    return .{ .object = o };
}

/// One line per envelope the parser rejected: the request, why, the name, the mode, what became of it.
fn logRejections(r: anytype, reply: *const chat.Reply, list: []const tool_parse.Rejection, offered: usize, kept_whole: bool) void {
    for (list) |j| {
        const fate = if (j.hidden) "hidden (TF_TOOL_CALL_GUARD)" else if (kept_whole) "kept as text (hiding it would leave the reply empty)" else "kept as text";
        logRejection(r, reply, j.reason.label(), j.name, j.envelope, offered, fate);
    }
}

/// ``tool call rejected from ID: REASON ['NAME'] (N tools offered) thinking= effort=; FATE``, and with
/// TF_TOOL_CALL_DEBUG=1 the envelope itself, escaped and cut to a line.
fn logRejection(r: anytype, reply: *const chat.Reply, reason: []const u8, name: []const u8, envelope: []const u8, offered: usize, fate: []const u8) void {
    var nb: [160]u8 = undefined;
    const effort = if (reply.thinking) reply.effort orelse "none" else "none";
    log.line("tool call rejected from {s}: {s} ['{s}'] ({d} tools offered) thinking={s} effort={s}; {s}", .{ r.id, reason, tool_guard.escaped(&nb, name), offered, if (reply.thinking) "True" else "False", effort, fate });
    if (!tool_guard.debug()) return;
    var eb: [1200]u8 = undefined;
    log.line("tool call rejected from {s}: envelope ({d} bytes) {s}", .{ r.id, envelope.len, tool_guard.escaped(&eb, envelope) });
}

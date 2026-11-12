//! Tool calls while they are written: XML and DSML calls as argument deltas, and the one-call policy's text filter.
const std = @import("std");
const json = @import("json.zig");
const ids = @import("ids.zig");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const space = reply_text.isSpaceByte;

/// ``\s*<TAG=([^>\n]+)>\n?`` at the start of ``rest``: the name and the bytes matched.
fn opening(rest: []const u8, tag: []const u8) ?struct { name: []const u8, len: usize, newline: bool } {
    var i: usize = 0;
    while (i < rest.len and space(rest[i])) i += 1;
    if (!std.mem.startsWith(u8, rest[i..], tag)) return null;
    i += tag.len;
    const start = i;
    while (i < rest.len and rest[i] != '>' and rest[i] != '\n') i += 1;
    if (i == start or i >= rest.len or rest[i] != '>') return null;
    const name = rest[start..i];
    i += 1;
    const newline = i < rest.len and rest[i] == '\n';
    return .{ .name = name, .len = i + @intFromBool(newline), .newline = newline };
}

/// ``\s*TEXT`` at the start of ``rest``: bytes matched.
fn closing(rest: []const u8, text: []const u8) ?usize {
    var i: usize = 0;
    while (i < rest.len and space(rest[i])) i += 1;
    return if (std.mem.startsWith(u8, rest[i..], text)) i + text.len else null;
}

/// Streams Qwen's XML calls as JSON argument deltas matching the final parser.
pub const Streamer = struct {
    a: Allocator,
    schemas: tool_params.Schemas,
    known: std.StringHashMapUnmanaged([]const u8) = .empty,
    current: []const u8 = "",
    value_schema: Value = .null,
    typed: bool = false,
    buffer: std.ArrayList(u8) = .empty,
    pos: usize = 0,
    state: enum { outside, call, function, value, closing, abort } = .outside,
    index: i64 = -1,
    args_open: bool = false,
    lead: bool = false,
    held: []const u8 = "",
    streamed: bool = false,
    searched: usize = 0, // the answer is only ever extended, so prose already searched holds no opener
    dsml: Dsml = .{}, // DeepSeek-V4.1's calls (feedDsml)

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!Streamer {
        var s: Streamer = .{ .a = a, .schemas = try tool_params.schemas(a, tools) };
        // the offered names as tool_parse.parse reads them for a DSML block
        for (tools) |tool| {
            const name = try tool_specs.toolName(a, tool);
            try s.dsml.known.put(a, try std.ascii.allocLowerString(a, name), name);
        }
        for (tools) |tool| {
            if (tool != .object) continue;
            const spec = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
            const name = if (spec.get("name")) |n| (if (n.truthy()) try tool_specs.pyStr(a, n) else "") else "";
            if (name.len == 0) continue;
            try s.known.put(a, try std.ascii.allocLowerString(a, name), name);
        }
        return s;
    }

    fn args(s: *Streamer, fragment: []const u8) Allocator.Error!Value {
        const function = try json.newObject(s.a);
        try function.put(s.a, "arguments", .{ .string = fragment });
        const one = try json.newObject(s.a);
        try one.put(s.a, "index", try json.intValue(s.a, s.index));
        try one.put(s.a, "function", .{ .object = function });
        return wrap(s.a, .{ .object = one });
    }

    fn esc(s: *Streamer, text: []const u8) Allocator.Error![]const u8 {
        const q = try json.quote(s.a, text, .{ .ascii = false });
        return q[1 .. q.len - 1];
    }

    /// The deltas the reply's ``text`` (its whole answer so far) adds.
    pub fn feed(s: *Streamer, text: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        while (s.state != .abort) {
            const rest = text[@min(s.pos, text.len)..];
            switch (s.state) {
                .outside => {
                    if (text.len < s.searched) s.searched = s.pos;
                    const from = @min(text.len, @max(s.pos, s.searched -| ("<tool_call>".len - 1)));
                    const j = reply_text.findTag(text, from, "<tool_call>") orelse {
                        s.searched = text.len;
                        return;
                    };
                    s.pos = j + "<tool_call>".len;
                    s.state = .call;
                },
                .call => {
                    const m = opening(rest, "<function=") orelse {
                        const trimmed = reply_text.pyStrip(rest);
                        if (reply_text.charCount(rest) > 256 or (trimmed.len > 0 and trimmed[0] != '<')) s.state = .abort;
                        return;
                    };
                    const lower = try std.ascii.allocLowerString(s.a, reply_text.pyStrip(m.name));
                    const name = s.known.get(lower) orelse {
                        s.state = .abort;
                        return;
                    };
                    s.current = name;
                    s.index += 1;
                    s.streamed = true;
                    const function = try json.newObject(s.a);
                    try function.put(s.a, "name", .{ .string = name });
                    try function.put(s.a, "arguments", .{ .string = "" });
                    const one = try json.newObject(s.a);
                    try one.put(s.a, "index", try json.intValue(s.a, s.index));
                    try one.put(s.a, "id", .{ .string = try ids.make(s.a, "call_", 24) });
                    try one.put(s.a, "type", .{ .string = "function" });
                    try one.put(s.a, "function", .{ .object = function });
                    try out.append(s.a, try wrap(s.a, .{ .object = one }));
                    s.args_open = false;
                    s.pos += m.len;
                    s.state = .function;
                },
                .function => {
                    if (opening(rest, "<parameter=")) |m| {
                        const key = reply_text.pyStrip(m.name);
                        s.value_schema = try tool_params.schemaOf(&s.schemas, s.current, key, s.a);
                        s.typed = typedSchema(s.value_schema);
                        s.buffer.clearRetainingCapacity();
                        const quoted = try json.quote(s.a, key, .{ .ascii = false });
                        try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (s.typed) ":" else ":\"" })));
                        s.args_open = true;
                        s.pos += m.len;
                        s.state = .value;
                        s.lead = !m.newline;
                        s.held = "";
                        continue;
                    }
                    if (closing(rest, "</function>")) |n| {
                        try out.append(s.a, try s.args(if (s.args_open) "}" else "{}"));
                        s.pos += n;
                        s.state = .closing;
                        continue;
                    }
                    return;
                },
                .value => {
                    const tail = "</parameter>";
                    const end = reply_text.findTag(rest, 0, tail);
                    const chunk = if (end) |e| rest[0..e] else rest[0..trimTail(rest, tail.len)];
                    s.pos += chunk.len;
                    var piece = try std.mem.concat(s.a, u8, &.{ s.held, chunk });
                    if (s.lead and piece.len > 0) {
                        if (piece[0] == '\n') piece = piece[1..];
                        s.lead = false;
                    }
                    var keep: []const u8 = undefined;
                    if (end != null) {
                        keep = if (std.mem.endsWith(u8, piece, "\n")) piece[0 .. piece.len - 1] else piece; // the framing newline
                        s.held = "";
                    } else {
                        keep = reply_text.pyRstrip(piece); // its last newline may be the framing one
                        s.held = piece[keep.len..];
                    }
                    if (s.typed) {
                        try s.buffer.appendSlice(s.a, keep);
                    } else if (keep.len > 0) try out.append(s.a, try s.args(try s.esc(keep)));
                    if (end == null) return;
                    if (s.typed) {
                        const v = try tool_params.decode(s.a, s.buffer.items, s.value_schema, true);
                        try out.append(s.a, try s.args(try json.stringify(s.a, v, .{ .ascii = false })));
                    } else try out.append(s.a, try s.args("\""));
                    s.pos += tail.len;
                    s.state = .function;
                },
                .closing => {
                    const n = closing(rest, "</tool_call>") orelse return;
                    s.pos += n;
                    s.state = .outside;
                },
                .abort => return,
            }
        }
    }

    /// The deltas the reply's ``text`` (its whole answer so far, never the reasoning) adds to its DSML calls.
    pub fn feedDsml(s: *Streamer, t: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        return s.feedDsmlTo(t, out, false);
    }

    /// ``feedDsml`` that stops at the end of the block it is in: a round's prose after that block goes out after the
    /// block's last deltas and before the next block's first (``Dsml.midCall``).
    pub fn feedDsmlBlock(s: *Streamer, t: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        return s.feedDsmlTo(t, out, true);
    }

    fn feedDsmlTo(s: *Streamer, t: []const u8, out: *std.ArrayList(Value), one_block: bool) Allocator.Error!void {
        const d = &s.dsml;
        const a = s.a;
        // the saved positions hold only in a text that extends the one read before; a shorter one turns the streamer
        // off, and the finish then follows the parse (Dsml.verify checks the whole text once)
        if (t.len < d.read) d.off = true;
        if (!d.off and t.len > d.read) {
            try d.seen.appendSlice(a, t[d.read..]);
            d.read = t.len;
        }
        while (!d.off) {
            d.bounds(t);
            switch (d.state) {
                .outside => {
                    if (one_block) return;
                    // envelopes(): the next block from the end of the last, and only one that closes is the parse's
                    const i = findIn(t, tool_parse.dsml41_open, @max(d.pos, d.from), null);
                    if (i != .at) {
                        d.from = @max(d.pos, t.len -| (tool_parse.dsml41_open.len - 1));
                        return;
                    }
                    if ((d.sent.items.len == 0 and s.streamed) or foreignCall(t[d.gap..i.at])) {
                        d.off = true;
                        return;
                    }
                    d.pos = i.at + tool_parse.dsml41_open.len;
                    d.block_end = null;
                    d.block_from = d.pos;
                    d.block_calls = 0;
                    d.state = .block;
                },
                .block => switch (findIn(t, tool_parse.invoke41_open, d.pos, d.block_end)) {
                    .wait => return,
                    .none => {
                        // the block's closer: every invoke in it is decided, and the calls it held go out
                        try d.closePending(a, out);
                        try d.release(a, out);
                        d.pos = d.block_end.? + tool_parse.dsml41_close.len;
                        d.gap = d.pos;
                        d.from = d.pos;
                        d.state = .outside;
                    },
                    .at => |i| {
                        d.pos = i + 1; // where the parse looks again when this invoke is not one
                        d.name_start = i + tool_parse.invoke41_open.len;
                        d.state = .name;
                    },
                },
                .name => switch (findIn(t, "\"", d.name_start, d.block_end)) {
                    .wait => return,
                    .none => d.state = .block,
                    .at => |q| {
                        if (q == d.name_start) {
                            d.state = .block;
                            continue;
                        }
                        if (q + 1 >= (d.block_end orelse t.len)) {
                            if (d.block_end == null) return; // its '>' has not come yet
                            d.state = .block;
                            continue;
                        }
                        if (t[q + 1] != '>') {
                            d.state = .block;
                            continue;
                        }
                        d.q = q;
                        d.end = null;
                        d.end_pending = true;
                        d.end_from = q + 2;
                        d.state = .skip;
                        d.bounds(t);
                        if (!d.end_pending and d.end == null) {
                            d.state = .block; // no closer in its block: the parse takes no call here
                            continue;
                        }
                        const name = reply_text.pyStrip(t[d.name_start..q]);
                        if (name.len == 0) continue; // the parse reads it, then takes no call: skip to its closer
                        // the block's first call streams; a later one waits with the first call's brace for the
                        // block's closer, since the parse takes no call from a block that never closes
                        d.block_calls += 1;
                        if (d.block_calls > 1) d.holding = true;
                        try d.closePending(a, out);
                        // the parse's name: the offered spelling, else the model's (a copy: the answer's buffer moves)
                        const called = d.known.get(try std.ascii.allocLowerString(a, name)) orelse try a.dupe(u8, name);
                        const function = try json.newObject(a);
                        try function.put(a, "name", .{ .string = called });
                        try function.put(a, "arguments", .{ .string = "" });
                        const one = try json.newObject(a);
                        try one.put(a, "index", try json.intValue(a, d.sent.items.len));
                        try one.put(a, "id", .{ .string = try ids.make(a, "call_", 24) });
                        try one.put(a, "type", .{ .string = "function" });
                        try one.put(a, "function", .{ .object = function });
                        try d.dest(out).append(a, try wrap(a, .{ .object = one }));
                        try d.sent.append(a, .{ .name = called });
                        if (!d.holding) d.released = d.sent.items.len;
                        d.current = d.sent.items.len - 1;
                        s.streamed = true;
                        d.p = q + 2;
                        d.state = .body;
                    },
                },
                .skip => {
                    if (d.end_pending) return;
                    if (d.end) |e| d.pos = e + tool_parse.invoke41_close.len;
                    d.state = .block;
                },
                .body, .key, .string_value, .json_value => if (!d.end_pending and d.end == null) {
                    // the block closed with this invoke open: the parse drops it and looks again after its opener
                    d.hold();
                    d.current = null;
                    d.state = .block;
                    continue;
                } else switch (d.state) {
                    .body => switch (findIn(t, tool_parse.param41_open, d.p, d.limit())) {
                        .wait => return,
                        .none => {
                            // the invoke's closer: the call is whole, and its brace waits
                            d.closing = d.current;
                            d.current = null;
                            d.pos = d.end.? + tool_parse.invoke41_close.len;
                            d.state = .block;
                        },
                        .at => |ps| {
                            d.p = ps + 1; // where the parse looks again when this parameter is not one
                            d.ns = ps + tool_parse.param41_open.len;
                            d.state = .key;
                        },
                    },
                    .key => switch (findIn(t, "\"", d.ns, d.limit())) {
                        .wait => return,
                        .none => d.state = .body,
                        .at => |nq| {
                            if (nq == d.ns) {
                                d.state = .body;
                                continue;
                            }
                            const lim = d.limit() orelse t.len;
                            var v = nq + 1;
                            const flag = t[@min(v, lim)..lim];
                            var json_value = false;
                            if (std.mem.startsWith(u8, flag, " string=\"true\"")) {
                                v += " string=\"true\"".len;
                            } else if (std.mem.startsWith(u8, flag, " string=\"false\"")) {
                                v += " string=\"false\"".len;
                                json_value = true;
                            } else if (d.limit() == null and (std.mem.startsWith(u8, " string=\"true\"", flag) or std.mem.startsWith(u8, " string=\"false\"", flag))) return;
                            if (v >= lim) {
                                if (d.limit() == null) return;
                                d.state = .body;
                                continue;
                            }
                            if (t[v] != '>') {
                                d.state = .body;
                                continue;
                            }
                            d.nq = nq;
                            d.v = v + 1;
                            d.at = v + 1;
                            if (json_value) {
                                d.state = .json_value;
                            } else {
                                try d.key(a, t[d.ns..nq], ":\"", out);
                                d.state = .string_value;
                            }
                        },
                    },
                    .string_value => switch (findIn(t, tool_parse.param41_close, d.at, d.limit())) {
                        .at => |pe| {
                            try d.value(a, t[d.at..pe], out);
                            try d.piece(a, "\"", out);
                            d.p = pe + tool_parse.param41_close.len;
                            d.state = .body;
                        },
                        .none => {
                            // the invoke ends inside the value: the parse drops the parameter and reads on from its opener
                            d.hold();
                            d.state = .body;
                        },
                        .wait => {
                            const rest = t[d.at..];
                            const keep = @max(reply_text.partialTag(rest, tool_parse.param41_close), openSequence(rest));
                            try d.value(a, rest[0 .. rest.len - keep], out);
                            d.at = t.len - keep;
                            return;
                        },
                    },
                    .json_value => switch (findIn(t, tool_parse.param41_close, d.at, d.limit())) {
                        .at => |pe| {
                            const raw = t[d.v..pe];
                            const parsed: Value = switch (try json.parseText(a, raw)) {
                                .ok => |x| x,
                                .err => .{ .string = raw },
                            };
                            const text = try json.stringify(a, parsed, .{ .ascii = false, .compact = true });
                            try d.key(a, t[d.ns..d.nq], try std.mem.concat(a, u8, &.{ ":", text }), out);
                            d.p = pe + tool_parse.param41_close.len;
                            d.state = .body;
                        },
                        .none => d.state = .body, // nothing of it went out
                        .wait => {
                            d.at = @max(d.v, t.len -| (tool_parse.param41_close.len - 1));
                            return;
                        },
                    },
                    else => unreachable,
                },
            }
        }
    }
};

/// A DSML call a stream sent: its name, the arguments text that went out, and whether it closed.
pub const SentCall = struct {
    name: []const u8,
    arguments: std.ArrayList(u8) = .empty,
    opened: bool = false, // its "{" went out with the first parameter
    closed: bool = false, // its closing brace went out: its invoke and its block closed
    held: bool = false, // its pieces stopped: the parse drops a value or the whole invoke that already went out

    /// The parse's call ``call`` is this one: same name, the same arguments byte for byte, and closed.
    pub fn same(c: *const SentCall, call: Value) bool {
        const function = call.get("function") orelse return false;
        const name = function.strField("name") orelse return false;
        const args = function.strField("arguments") orelse return false;
        return c.closed and !c.held and std.mem.eql(u8, name, c.name) and std.mem.eql(u8, args, c.arguments.items);
    }
};

/// Where ``needle`` starts in ``t`` at or after ``from`` and before ``limit``: the parse's ``find`` on a slice that ends at
/// ``limit``. With no limit yet (its closer has not arrived) a miss waits for more text: a complete needle in the text so
/// far is inside the slice, since no DSML closer can overlap another piece of DSML markup.
const Found = union(enum) { at: usize, none, wait };

fn findIn(t: []const u8, needle: []const u8, from: usize, limit: ?usize) Found {
    const hay = t[0..@min(t.len, limit orelse t.len)];
    if (from <= hay.len) if (std.mem.indexOfPos(u8, hay, from, needle)) |i| return .{ .at = i };
    return if (limit != null) .none else .wait;
}

/// Bytes at the end of ``s`` that begin a character the text does not hold whole yet.
fn openSequence(s: []const u8) usize {
    var back: usize = 0;
    while (back < @min(4, s.len)) {
        back += 1;
        const b = s[s.len - back];
        if (b & 0xc0 == 0x80) continue;
        const n = std.unicode.utf8ByteSequenceLength(b) catch return 0;
        return if (n > back) back else 0;
    }
    return 0;
}

/// Another call form in the text before a DSML block: tool_parse.parse reads its calls there too, in text order.
fn foreignCall(text: []const u8) bool {
    return std.ascii.findIgnoreCase(text, "<tool_call>") != null or std.ascii.findIgnoreCase(text, ":tool_call>") != null or
        std.mem.indexOf(u8, text, "<|tool_call>") != null or std.mem.indexOf(u8, text, tool_parse.dsml_open) != null;
}

/// DeepSeek-V4.1's DSML calls as they decode, read the way ``tool_parse.dsml41Calls`` reads the finished reply: each
/// search is the parse's search, made on the text so far. A header goes out when its invoke name is whole and the parse
/// would take it; a string value goes out as JSON-escaped pieces while it decodes (a tail that may begin its closer or a
/// character is held); any other value goes out when its parameter closes, as the parse writes it; a call's closing
/// brace waits for the next header or for its block's closer.
pub const Dsml = struct {
    known: std.StringArrayHashMapUnmanaged([]const u8) = .empty, // tool_parse.parse's offered names, lower-cased
    sent: std.ArrayList(SentCall) = .empty,
    off: bool = false, // another call form came first: the reply's calls go out when it ends, as without this patch
    state: enum { outside, block, name, skip, body, key, string_value, json_value } = .outside,
    pos: usize = 0, // the parse's next invoke search (outside: its next block search)
    from: usize = 0, // where a search that waits starts again (the text before it holds no complete needle)
    gap: usize = 0, // the end of the last block: the text from here to the next one holds no other call form
    block_end: ?usize = null, // the block's closer, once it is in the text
    block_from: usize = 0,
    name_start: usize = 0,
    q: usize = 0, // the invoke name's closing quote
    end: ?usize = null, // the invoke's closer (null and not end_pending: the block holds none)
    end_pending: bool = false,
    end_from: usize = 0,
    p: usize = 0, // the parse's next parameter search
    ns: usize = 0,
    nq: usize = 0,
    v: usize = 0, // the value's start
    at: usize = 0, // the value's bytes before this went out (a string) or hold no closer (any value)
    current: ?usize = null, // the call whose arguments are going out
    closing: ?usize = null, // a whole call whose brace waits
    read: usize = 0, // the length of the text read last
    seen: std.ArrayList(u8) = .empty, // the text read so far (verify checks the finish's text against it)
    block_calls: usize = 0, // the calls this block made so far
    holding: bool = false, // the block made a second call: its deltas wait for the block's closer
    queue: std.ArrayList(Value) = .empty, // the deltas that wait
    released: usize = 0, // the calls whose header went out: sent[0..released]
    late: ?usize = null, // the block's first call, whose brace waits in the queue

    /// A call is under way: its block opened, or its brace waits.
    pub fn midCall(d: *const Dsml) bool {
        return !d.off and (d.state != .outside or d.closing != null);
    }

    /// The calls the client got (their headers went out), for the finish to check against the parse.
    pub fn streamed(d: *const Dsml) []const SentCall {
        return d.sent.items[0..d.released];
    }

    /// The finish's ``text`` must extend the text the stream read, else its positions do not hold there: the streamer
    /// turns off, and the finish follows the parse.
    pub fn verify(d: *Dsml, text: []const u8) void {
        if (!std.mem.startsWith(u8, text, d.seen.items)) d.off = true;
    }

    fn dest(d: *Dsml, out: *std.ArrayList(Value)) *std.ArrayList(Value) {
        return if (d.holding) &d.queue else out;
    }

    /// The block closed: what it held goes out, the first call's brace first.
    fn release(d: *Dsml, a: Allocator, out: *std.ArrayList(Value)) Allocator.Error!void {
        if (d.late) |i| {
            const c = &d.sent.items[i];
            try c.arguments.appendSlice(a, if (c.opened) "}" else "{}");
            c.closed = true;
            d.late = null;
        }
        try out.appendSlice(a, d.queue.items);
        d.queue.clearRetainingCapacity();
        d.holding = false;
        d.released = d.sent.items.len;
    }

    fn limit(d: *const Dsml) ?usize {
        return if (d.end_pending) null else d.end;
    }

    /// The block's closer and the invoke's closer, found as soon as they are in the text.
    fn bounds(d: *Dsml, t: []const u8) void {
        if (d.state == .outside) return;
        if (d.block_end == null) {
            if (d.block_from <= t.len) if (std.mem.indexOfPos(u8, t, d.block_from, tool_parse.dsml41_close)) |i| {
                d.block_end = i;
            };
            if (d.block_end == null) d.block_from = @max(d.block_from, t.len -| (tool_parse.dsml41_close.len - 1));
        }
        if (!d.end_pending) return;
        switch (findIn(t, tool_parse.invoke41_close, d.end_from, d.block_end)) {
            .at => |i| {
                d.end = i;
                d.end_pending = false;
            },
            .none => {
                d.end = null;
                d.end_pending = false;
            },
            .wait => d.end_from = @max(d.end_from, t.len -| (tool_parse.invoke41_close.len - 1)),
        }
    }

    fn delta(a: Allocator, index: usize, fragment: []const u8) Allocator.Error!Value {
        const function = try json.newObject(a);
        try function.put(a, "arguments", .{ .string = fragment });
        const one = try json.newObject(a);
        try one.put(a, "index", try json.intValue(a, index));
        try one.put(a, "function", .{ .object = function });
        return wrap(a, .{ .object = one });
    }

    /// ``fragment`` of the current call's arguments, unless its pieces stopped.
    fn piece(d: *Dsml, a: Allocator, fragment: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        const i = d.current orelse return;
        const c = &d.sent.items[i];
        if (c.held or fragment.len == 0) return;
        try c.arguments.appendSlice(a, fragment);
        try d.dest(out).append(a, try delta(a, i, fragment));
    }

    /// A parameter's key, after "{" or ",", then ``after``.
    fn key(d: *Dsml, a: Allocator, name: []const u8, after: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        const i = d.current orelse return;
        const c = &d.sent.items[i];
        const quoted = try json.quote(a, name, .{ .ascii = false });
        try d.piece(a, try std.mem.concat(a, u8, &.{ if (c.opened) "," else "{", quoted, after }), out);
        c.opened = true;
    }

    /// The JSON-escaped text of a string value's bytes (escaping is per byte, so pieces join to the whole value's).
    fn value(d: *Dsml, a: Allocator, bytes: []const u8, out: *std.ArrayList(Value)) Allocator.Error!void {
        if (bytes.len == 0) return;
        const q = try json.quote(a, bytes, .{ .ascii = false });
        try d.piece(a, q[1 .. q.len - 1], out);
    }

    /// The brace of the whole call before this one: its invoke closed, and a new header or the block's closer came.
    fn closePending(d: *Dsml, a: Allocator, out: *std.ArrayList(Value)) Allocator.Error!void {
        const i = d.closing orelse return;
        d.closing = null;
        const c = &d.sent.items[i];
        if (c.held) return;
        if (d.holding and i < d.released) {
            // the block's first call went out: its brace waits with the block's later calls, so a block that never
            // closes leaves it open, as the parse takes none of the block's calls
            try d.queue.append(a, try delta(a, i, if (c.opened) "}" else "{}"));
            d.late = i;
            return;
        }
        const was = d.current;
        d.current = i;
        try d.piece(a, if (c.opened) "}" else "{}", out);
        d.current = was;
        c.closed = true;
    }

    /// The current invoke's call stops: the parse drops something that already went out.
    fn hold(d: *Dsml) void {
        if (d.current) |i| d.sent.items[i].held = true;
    }
};

/// Bytes of ``rest`` that cannot be the start of a ``tail``-byte closer (Python slices characters).
fn trimTail(rest: []const u8, tail: usize) usize {
    const chars = reply_text.charCount(rest);
    if (chars <= tail) return 0;
    return rest.len - reply_text.afterChars(rest, chars - tail).len;
}

fn typedSchema(schema: Value) bool {
    const t = schema.get("type") orelse return false;
    if (t != .string) return false;
    for ([_][]const u8{ "array", "object", "boolean", "integer", "number", "null" }) |k| if (std.mem.eql(u8, k, t.string)) return true;
    return false;
}

const wrap = @import("tool_parse.zig").wrapCalls;

fn nameLike(s: []const u8) bool {
    if (s.len == 0 or !(std.ascii.isAlphabetic(s[0]) or s[0] == '_')) return false;
    for (s[1..]) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '-' or ch >= 0x80)) return false;
    return true;
}

/// Hides tool envelopes from streamed prose, holding only a partial tag between deltas.
pub const TextFilter = struct {
    pending: std.ArrayList(u8) = .empty,
    hidden: ?[]const u8 = null,

    pub fn feed(f: *TextFilter, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        try f.pending.appendSlice(a, text);
        var out: std.ArrayList(u8) = .empty;
        while (f.pending.items.len > 0) {
            const p = f.pending.items;
            if (f.hidden) |name| {
                const close = try std.mem.concat(a, u8, &.{ "</", name, ">" });
                const end = std.ascii.findIgnoreCase(p, close) orelse {
                    const keep = reply_text.charCount(close) - 1;
                    const chars = reply_text.charCount(p);
                    const tail = if (chars > keep) reply_text.afterChars(p, chars - keep) else p;
                    try f.reset(a, tail);
                    break;
                };
                try f.reset(a, p[end + close.len ..]);
                f.hidden = null;
                continue;
            }
            const start = std.mem.indexOfScalar(u8, p, '<') orelse {
                try out.appendSlice(a, p);
                f.pending.clearRetainingCapacity();
                break;
            };
            try out.appendSlice(a, p[0..start]);
            const rest = p[start..];
            const end = std.mem.indexOfScalar(u8, rest, '>') orelse {
                const name = rest[1..];
                const colon = std.mem.indexOfScalar(u8, name, ':');
                const possible = name.len == 0 or nameLike(name) or (colon != null and nameLike(name[0..colon.?]) and
                    std.ascii.startsWithIgnoreCase("tool_call", name[colon.? + 1 ..]));
                if (possible) {
                    try f.reset(a, rest);
                    break;
                }
                try out.append(a, '<');
                try f.reset(a, rest[1..]);
                continue;
            };
            const name = rest[1..end];
            if (callTag(name)) {
                f.hidden = try std.ascii.allocLowerString(a, name);
            } else try out.appendSlice(a, rest[0 .. end + 1]);
            try f.reset(a, rest[end + 1 ..]);
        }
        return out.items;
    }

    fn reset(f: *TextFilter, a: Allocator, rest: []const u8) Allocator.Error!void {
        const copy = try a.dupe(u8, rest);
        f.pending.clearRetainingCapacity();
        try f.pending.appendSlice(a, copy);
    }

    pub fn finish(f: *TextFilter) []const u8 {
        const tail = if (f.hidden != null) "" else f.pending.items;
        f.pending = .empty;
        return tail;
    }
};

/// ``(?:NAME:)?tool_call``, any case.
fn callTag(name: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(name, "tool_call")) return true;
    const colon = std.mem.indexOfScalar(u8, name, ':') orelse return false;
    return nameLike(name[0..colon]) and std.ascii.eqlIgnoreCase(name[colon + 1 ..], "tool_call");
}

/// ``parallel_tool_calls: false`` keeps one call and hides call markup from streamed prose.
pub const Policy = struct {
    single: bool = false,
    filter: TextFilter = .{},

    pub fn maxCalls(p: *const Policy) ?usize {
        return if (p.single) 1 else null;
    }

    /// Filters a str delta's text (dict deltas lose tool_calls and filter content), or passes it through.
    pub fn text(p: *Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        return if (p.single) p.filter.feed(a, s) else s;
    }

    pub fn flush(p: *Policy) []const u8 {
        return if (p.single) p.filter.finish() else "";
    }

    pub fn content(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var f: TextFilter = .{};
        const head = try f.feed(a, s);
        return std.mem.concat(a, u8, &.{ head, f.finish() });
    }

    /// ``parsed_content``: a parsed reply's content, whole envelopes the parser left as text kept and the rest filtered, so an unclosed envelope stays hidden.
    pub fn parsedContent(p: *const Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        if (!p.single) return s;
        var out: std.ArrayList(u8) = .empty;
        var cursor: usize = 0;
        while (try wholeEnvelope(a, s, cursor)) |e| : (cursor = e.end) {
            try out.appendSlice(a, try p.content(a, s[cursor..e.start]));
            try out.appendSlice(a, s[e.start..e.end]);
        }
        try out.appendSlice(a, try p.content(a, s[cursor..]));
        return out.items;
    }

    /// What the filter held back that the reply keeps as text (a malformed call): ``final`` past the prose ``sent``.
    pub fn kept(p: *const Policy, final: []const u8, sent: []const u8) ?[]const u8 {
        if (!p.single) return null;
        const shown = reply_text.pyLstrip(sent); // the parsed content starts stripped
        if (!std.mem.startsWith(u8, final, shown)) return null;
        const rest = final[shown.len..];
        return if (reply_text.pyStrip(rest).len > 0) rest else null;
    }
};

/// The first whole envelope from ``from`` (``<((?:NAME:)?tool_call)>.*?</\1>``, any case).
fn wholeEnvelope(a: Allocator, s: []const u8, from: usize) Allocator.Error!?struct { start: usize, end: usize } {
    var at = from;
    while (std.mem.indexOfScalarPos(u8, s, at, '<')) |start| : (at = start + 1) {
        const close = std.mem.indexOfScalarPos(u8, s, start, '>') orelse return null;
        if (!callTag(s[start + 1 .. close])) continue;
        const closer = try std.mem.concat(a, u8, &.{ "</", s[start + 1 .. close], ">" });
        const end = std.ascii.findIgnoreCasePos(s, close + 1, closer) orelse continue;
        return .{ .start = start, .end = end + closer.len };
    }
    return null;
}

test "a streamed call its end token left open completes as its closing markup completes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars\n</parameter>\n<parameter=n>\n5\n</parameter>\n";
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    const ended = try streamed(a, tools, &.{ open, try std.mem.concat(a, u8, &.{ open, try tool_parse.closeCall(a, open, tools) }) });
    const marked = try streamed(a, tools, &.{ open, open ++ "</function>\n</tool_call>" });
    // the arguments sent before the end, closed by closedJson, are the ones the client ends with
    try std.testing.expectEqualStrings((try tool_params.closedJson(a, sent)).?, try sentArguments(a, ended));
    try std.testing.expectEqual(marked.len, ended.len);
    for (marked[1..], ended[1..]) |x, y| try std.testing.expectEqualStrings(try json.stringify(a, x, .{}), try json.stringify(a, y, .{}));
}

test "a streamed call cut inside a value stays open" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const open = "<tool_call>\n<function=lookup>\n<parameter=q>\nfast cars and more";
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, open, tools));
    // its opening delta went out and cannot be taken back; what it sent does not close under closedJson
    const sent = try sentArguments(a, try streamed(a, tools, &.{open}));
    try std.testing.expectEqualStrings("{\"q\":\"fast c", sent);
    try std.testing.expect((try tool_params.closedJson(a, sent)) == null);
}

test "a one-call reply drops an unoffered call and still hides an unclosed one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const single: Policy = .{ .single = true };
    const text = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    // what attachCalls makes of the reply: the parse, then the policy's content
    const r = try tool_parse.parse(a, text, tools, single.maxCalls());
    try std.testing.expect(r.calls == null);
    try std.testing.expectEqualStrings("Launching.", r.content);
    try std.testing.expectEqualStrings("Launching.", try single.parsedContent(a, r.content));
    try std.testing.expectEqualStrings("Launching. ", try single.parsedContent(a, "Launching. <tool_call>{\"name\""));
    // a stream that sent the prose before the block ends with the block
    try std.testing.expectEqualStrings(text["Launching. ".len..], single.kept(text, "\nLaunching. ").?);
    try std.testing.expect(single.kept("Launching.", "Launching. ") == null);
    const parallel: Policy = .{};
    try std.testing.expect(parallel.kept(text, "Launching. ") == null);
}

const tool_parse = @import("tool_parse.zig");

/// The deltas one streamer sends while the reply grows through ``texts``.
fn streamed(a: Allocator, tools: []const Value, texts: []const []const u8) Allocator.Error![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (texts) |t| try s.feed(t, &out);
    return out.items;
}

/// The arguments a client joins from streamed deltas.
fn sentArguments(a: Allocator, deltas: []const Value) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    };
    return out.items;
}

// DSML calls as they decode, fed as the answer grows at many split points.
const D = "\u{ff5c}DSML\u{ff5c}";
const calls_open = "\n\n<" ++ D ++ " calls>\n";
const calls_close = "\n</" ++ D ++ " calls>";

fn dsmlInvoke(comptime name: []const u8, comptime body: []const u8) []const u8 {
    return "<" ++ D ++ " invoke name=\"" ++ name ++ "\">\n" ++ body ++ "</" ++ D ++ " invoke>\n";
}

fn dsmlParam(comptime name: []const u8, comptime flag: []const u8, comptime value: []const u8) []const u8 {
    return "<" ++ D ++ " parameter name=\"" ++ name ++ "\"" ++ flag ++ ">" ++ value ++ "</" ++ D ++ " parameter>\n";
}

const dsml_tools =
    \\[{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"}}}}},
    \\{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}}}]
;

/// One streamer fed `text` as it grows by `step` bytes (`chars`: kept to whole characters), then once whole, as the
/// finish feeds it; its deltas checked: each call's header once and in order, then only its own pieces, each valid UTF-8,
/// joined to what the streamer records as sent.
fn dsmlFeed(a: Allocator, text: []const u8, step: usize, chars: bool) !Streamer {
    var s = try Streamer.init(a, (try json.parse(a, dsml_tools)).ok.array);
    var out: std.ArrayList(Value) = .empty;
    var end: usize = 0;
    while (end < text.len) {
        end = @min(text.len, end + step);
        if (chars) {
            while (end < text.len and text[end] & 0xc0 == 0x80) end += 1;
        }
        try s.feedDsml(text[0..end], &out);
    }
    try s.feedDsml(text, &out);
    var joined: std.ArrayList(std.ArrayList(u8)) = .empty;
    for (out.items) |d| for (d.get("tool_calls").?.array) |one| {
        const index: usize = @intCast(one.get("index").?.int64().?);
        const piece = one.get("function").?.strField("arguments").?;
        try std.testing.expect(std.unicode.utf8ValidateSlice(piece));
        if (one.get("id") != null) {
            try std.testing.expectEqual(joined.items.len, index);
            try std.testing.expectEqualStrings("", piece);
            try joined.append(a, .empty);
            continue;
        }
        try std.testing.expectEqual(joined.items.len - 1, index);
        try joined.items[index].appendSlice(a, piece);
    };
    // what went out: the calls whose header went out (a block that never closes holds its later calls)
    try std.testing.expectEqual(s.dsml.streamed().len, joined.items.len);
    for (s.dsml.streamed(), joined.items) |c, j| try std.testing.expectEqualStrings(c.arguments.items, j.items);
    return s;
}

const splits = [_]struct { step: usize, chars: bool }{ .{ .step = 1, .chars = false }, .{ .step = 1, .chars = true }, .{ .step = 2, .chars = false }, .{ .step = 3, .chars = true }, .{ .step = 7, .chars = false }, .{ .step = 1 << 20, .chars = true } };

/// At every split the stream sends the parse's calls: the same names, the arguments byte for byte, each closed.
fn expectDsmlParsed(text: []const u8, calls: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = (try tool_parse.parse(a, text, (try json.parse(a, dsml_tools)).ok.array, null)).calls orelse &.{};
    try std.testing.expectEqual(calls, parsed.len);
    for (splits) |sp| {
        const s = try dsmlFeed(a, text, sp.step, sp.chars);
        try std.testing.expectEqual(parsed.len, s.dsml.streamed().len);
        for (parsed, s.dsml.streamed()) |p, *c| try std.testing.expect(c.same(p));
    }
}

test "DSML calls stream as the parse reads them, at every split: escapes, a split character, a name to strip, odd parameters" {
    try expectDsmlParsed(comptime ("Running it." ++ calls_open ++ dsmlInvoke(" bash ", dsmlParam("command", " string=\"true\"", "printf '%s\\n' \"a\\\\b\" \u{1f30a}\t| tr a-z A-Z") ++
        dsmlParam("timeout", " string=\"false\"", " 120000 ") ++ dsmlParam("note", "", "no flag: a string") ++ dsmlParam("bad", " string=\"maybe\"", "skipped") ++
        dsmlParam("n", " string=\"false\"", "not json") ++ dsmlParam("x", " string=\"false\"", "NaN") ++ dsmlParam("empty", " string=\"true\"", "")) ++
        dsmlInvoke("Write", dsmlParam("file_path", " string=\"true\"", "/tmp/\u{6d77}.md") ++ dsmlParam("content", " string=\"true\"", "a < b </div> </\u{ff5c}DSML\u{ff5c} par\n\u{1f469}\u{200d}\u{1f4bb}\n")) ++ calls_close), 2);
}

test "DSML calls stream as the parse reads them: an empty name, a bad invoke line, a far quote, a quoted opener, two blocks" {
    // an invoke whose name strips to nothing is read and skipped; one without its '">' is not an invoke
    try expectDsmlParsed(comptime (calls_open ++ dsmlInvoke("  ", dsmlParam("command", " string=\"true\"", "ls")) ++ "<" ++ D ++ " invoke name=\"Bash\" x=\"1\">\n" ++
        dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "pwd")) ++ calls_close), 1);
    // a parameter name without its quotes: the parse's first quote lies in the next parameter, which it then reads
    try expectDsmlParsed(comptime (calls_open ++ dsmlInvoke("Bash", "<" ++ D ++ " parameter name=command>ls</" ++ D ++ " parameter>\n" ++
        dsmlParam("command", " string=\"true\"", "pwd")) ++ calls_close), 1);
    // an opener quoted in prose pairs with the real block's closer: the parse reads the invoke after it
    try expectDsmlParsed(comptime ("Write `<" ++ D ++ " calls>` blocks, like this one:" ++ calls_open ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "ls")) ++ calls_close), 1);
    // two blocks with text between them, and a call to a tool the request did not offer
    try expectDsmlParsed(comptime ("First." ++ calls_open ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "ls")) ++ calls_close ++ "\nThen." ++
        calls_open ++ dsmlInvoke("WebFetch", dsmlParam("url", " string=\"true\"", "https://example.com/?q=\"x\"")) ++ calls_close), 2);
    // DSML markup without its calls block is text
    try expectDsmlParsed(comptime ("Write " ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "ls")) ++ " inside a block."), 0);
}

test "a DSML call's brace waits for its block's closer; a block left open sends none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = comptime (calls_open ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "ls -la")) ++ calls_close);
    const open = whole[0 .. whole.len - calls_close.len];
    const s = try dsmlFeed(a, open, 1, true);
    try std.testing.expectEqualStrings("{\"command\":\"ls -la\"", s.dsml.sent.items[0].arguments.items);
    try std.testing.expect(!s.dsml.sent.items[0].closed);
    try std.testing.expect((try tool_parse.parse(a, open, (try json.parse(a, dsml_tools)).ok.array, null)).calls == null);
    const closed = try dsmlFeed(a, whole, 1, true);
    try std.testing.expectEqualStrings("{\"command\":\"ls -la\"}", closed.dsml.sent.items[0].arguments.items);
    try std.testing.expect(closed.dsml.sent.items[0].closed);
}

test "where the parse drops or changes what already went out, the stream says so: the finish then follows the parse" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, dsml_tools)).ok.array;
    // a repeated parameter: the parse keeps the first place and the last value; the stream sent both, so its text
    // differs, and reads back (last value wins) as the parse's object
    const twice = comptime (calls_open ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "ls") ++ dsmlParam("timeout", " string=\"false\"", "5") ++
        dsmlParam("command", " string=\"true\"", "pwd")) ++ calls_close);
    const parsed = (try tool_parse.parse(a, twice, tools, null)).calls.?[0];
    const s = try dsmlFeed(a, twice, 1, true);
    const c = &s.dsml.sent.items[0];
    try std.testing.expect(c.closed and !c.same(parsed));
    try std.testing.expectEqualStrings("{\"command\":\"ls\",\"timeout\":5,\"command\":\"pwd\"}", c.arguments.items);
    try std.testing.expectEqualStrings(parsed.get("function").?.strField("arguments").?, try json.stringify(a, (try json.parse(a, c.arguments.items)).ok, .{ .ascii = false, .compact = true }));
    // an invoke the block closes without its own closer: the parse drops it; the stream sent its start and holds it
    const unclosed = comptime (calls_open ++ "<" ++ D ++ " invoke name=\"Bash\">\n" ++ dsmlParam("command", " string=\"true\"", "ls") ++ calls_close);
    try std.testing.expect((try tool_parse.parse(a, unclosed, tools, null)).calls == null);
    const u = try dsmlFeed(a, unclosed, 1, true);
    try std.testing.expect(u.dsml.sent.items.len == 1 and u.dsml.sent.items[0].held and !u.dsml.sent.items[0].closed);
    // a value that holds an invoke's closer: the parse ends the invoke there and drops the value
    const cut = comptime (calls_open ++ dsmlInvoke("Write", dsmlParam("file_path", " string=\"true\"", "/a") ++ "<" ++ D ++ " parameter name=\"content\" string=\"true\">x</" ++ D ++ " invoke>y</" ++ D ++ " parameter>\n") ++ calls_close);
    const w = (try tool_parse.parse(a, cut, tools, null)).calls.?;
    try std.testing.expectEqualStrings("{\"file_path\":\"/a\"}", w[0].get("function").?.strField("arguments").?);
    const v = try dsmlFeed(a, cut, 1, true);
    try std.testing.expect(v.dsml.sent.items[0].held and !v.dsml.sent.items[0].same(w[0]));
    // another call form first: the stream leaves every call to the finish, as before
    const xml = comptime ("<tool_call>{\"name\":\"Bash\",\"arguments\":{\"command\":\"ls\"}}</tool_call>" ++ calls_open ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "pwd")) ++ calls_close);
    const x = try dsmlFeed(a, xml, 3, true);
    try std.testing.expect(x.dsml.off and x.dsml.sent.items.len == 0);
}

test "a block's second call and the first call's brace wait for the block's closer; a block cut before it sends no whole call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, dsml_tools)).ok.array;
    const first = comptime (calls_open ++ dsmlInvoke("Bash", dsmlParam("command", " string=\"true\"", "ls")));
    const second = comptime (first ++ dsmlInvoke("Write", dsmlParam("file_path", " string=\"true\"", "/tmp/a") ++ dsmlParam("content", " string=\"true\"", "x")));
    // cut in the second invoke, at every split: one call went out, open; the parse takes none
    for ([_]usize{ second.len - 3, second.len - 1, second.len }) |cut| for (splits) |sp| {
        const s = try dsmlFeed(a, second[0..cut], sp.step, sp.chars);
        try std.testing.expectEqual(@as(usize, 1), s.dsml.streamed().len);
        try std.testing.expectEqualStrings("{\"command\":\"ls\"", s.dsml.streamed()[0].arguments.items);
        try std.testing.expect(!s.dsml.streamed()[0].closed);
        try std.testing.expect((try tool_parse.parse(a, second[0..cut], tools, null)).calls == null);
    };
    // the block's closer releases both, the first call's brace first
    const s = try dsmlFeed(a, second ++ calls_close, 1, true);
    try std.testing.expectEqual(@as(usize, 2), s.dsml.streamed().len);
    for (s.dsml.streamed()) |c| try std.testing.expect(c.closed);
    try std.testing.expectEqualStrings("{\"command\":\"ls\"}", s.dsml.streamed()[0].arguments.items);
    try std.testing.expectEqualStrings("{\"file_path\":\"/tmp/a\",\"content\":\"x\"}", s.dsml.streamed()[1].arguments.items);
}

test "a text that does not extend the one read turns the streamer off instead of reading past its end" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try Streamer.init(a, (try json.parse(a, dsml_tools)).ok.array);
    var out: std.ArrayList(Value) = .empty;
    const streamed_text = comptime (calls_open ++ "<" ++ D ++ " invoke name=\"Bash\">\n<" ++ D ++ " parameter name=\"command\" string=\"true\">grep -n '<|channel|>' x.zig and more");
    var end: usize = 0;
    while (end < streamed_text.len) : (end = @min(streamed_text.len, end + 4)) try s.feedDsml(streamed_text[0..end], &out);
    try s.feedDsml(streamed_text, &out);
    // chat.finish, thinking off: content = parseHarmony(text).content, "" here
    try s.feedDsml("", &out);
    try std.testing.expect(s.dsml.off);
    // the finish's check: a longer text with another start
    var t = try Streamer.init(a, (try json.parse(a, dsml_tools)).ok.array);
    try t.feedDsml(streamed_text[0..40], &out);
    t.dsml.verify("other text, longer than forty bytes, that does not start with the first");
    try std.testing.expect(t.dsml.off);
}

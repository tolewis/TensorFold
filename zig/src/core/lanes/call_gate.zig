//! tool_choice "required": outside a think block the answer opens a call (Python engine/call_gate.py).
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Phase = enum { answer, lead, name, done };

/// Tokenizer the gate asks while a round is decided. `text` and `encode` slices live until the next call.
pub const Lex = struct {
    ctx: *anyopaque,
    blank: *const fn (ctx: *anyopaque, token: u32) bool,
    text: *const fn (ctx: *anyopaque, a: Allocator, token: u32) anyerror![]const u8,
    encode: *const fn (ctx: *anyopaque, a: Allocator, text: []const u8) anyerror![]const u32,
};

pub const Gate = struct {
    opener: u32,
    think_open: i64 = -1,
    think_end: i64 = -1,
    lead: []const u8 = "",
    names: []const []const u8 = &.{},
    tail: []const u8 = "",
    lex: Lex,
    armed: bool = true,
    phase: Phase = .answer,
    seen: std.ArrayList(u8) = .empty,

    pub fn afterPrompt(prompt: []const u32, opener: u32, think_open: i64, think_end: i64, lex: Lex, lead: []const u8, names: []const []const u8, tail: []const u8) Gate {
        var last_open: i64 = -1;
        var last_end: i64 = -1;
        for (prompt, 0..) |t, i| {
            if (think_open >= 0 and t == @as(u32, @intCast(think_open))) last_open = @intCast(i);
            if (think_end >= 0 and t == @as(u32, @intCast(think_end))) last_end = @intCast(i);
        }
        const held = think_open >= 0 and think_end >= 0 and last_open > last_end;
        return .{
            .opener = opener,
            .think_open = think_open,
            .think_end = think_end,
            .lead = lead,
            .names = names,
            .tail = tail,
            .lex = lex,
            .armed = !held,
        };
    }

    pub fn deinit(g: *Gate, a: Allocator) void {
        g.seen.deinit(a);
    }

    pub fn done(g: *const Gate) bool {
        return g.phase == .done;
    }

    const Snap = struct { armed: bool, phase: Phase, seen: []const u8 };

    fn ends(g: *const Gate, char: u8) bool {
        if (std.ascii.isWhitespace(char)) return true;
        if (g.tail.len > 0) return char == g.tail[0];
        return switch (char) {
            '<', '>', '{', '}', '"', '\'', '(', ')', ',' => true,
            else => false,
        };
    }

    fn step(g: *const Gate, a: Allocator, snap: Snap, token: u32) !struct { snap: Snap, fix: ?[]const u32 } {
        if (snap.phase == .answer) {
            if (!snap.armed) {
                const end = g.think_end >= 0 and token == @as(u32, @intCast(g.think_end));
                return .{ .snap = .{ .armed = end, .phase = .answer, .seen = "" }, .fix = null };
            }
            if (g.think_open >= 0 and token == @as(u32, @intCast(g.think_open))) {
                return .{ .snap = .{ .armed = false, .phase = .answer, .seen = "" }, .fix = null };
            }
            if (token == g.opener) return .{ .snap = .{ .armed = true, .phase = .lead, .seen = "" }, .fix = null };
            if (g.lex.blank(g.lex.ctx, token)) return .{ .snap = snap, .fix = null };
            var fix: std.ArrayList(u32) = .empty;
            try fix.append(a, g.opener);
            if (g.names.len > 0 and g.lead.len > 0) {
                const ids = try g.lex.encode(g.lex.ctx, a, g.lead);
                try fix.appendSlice(a, ids);
            }
            return .{ .snap = snap, .fix = fix.items };
        }
        if (snap.phase == .done or g.names.len == 0) {
            return .{ .snap = .{ .armed = snap.armed, .phase = .done, .seen = "" }, .fix = null };
        }
        const piece = try g.lex.text(g.lex.ctx, a, token);
        const written = try std.mem.concat(a, u8, &.{ snap.seen, piece });
        var owed: []const u8 = "";
        var seen = snap.seen;
        var body = written;
        if (snap.phase == .lead) {
            if (std.mem.startsWith(u8, g.lead, written)) {
                const full = written.len == g.lead.len;
                return .{ .snap = .{ .armed = snap.armed, .phase = if (full) .name else .lead, .seen = if (full) "" else written }, .fix = null };
            }
            if (!std.mem.startsWith(u8, written, g.lead)) {
                const rest = g.lead[seen.len..];
                const ids = try g.lex.encode(g.lex.ctx, a, rest);
                return .{ .snap = snap, .fix = ids };
            }
            owed = g.lead[seen.len..];
            seen = "";
            body = written[g.lead.len..];
        }
        var end: usize = body.len;
        for (body, 0..) |c, i| if (g.ends(c)) {
            end = i;
            break;
        };
        if (end == body.len) {
            for (g.names) |n| if (std.mem.startsWith(u8, n, body)) {
                return .{ .snap = .{ .armed = snap.armed, .phase = .name, .seen = body }, .fix = null };
            };
        }
        if (end < body.len) {
            for (g.names) |n| if (std.mem.eql(u8, n, body[0..end])) {
                return .{ .snap = .{ .armed = snap.armed, .phase = .done, .seen = "" }, .fix = null };
            };
        }
        const name = for (g.names) |n| {
            if (std.mem.startsWith(u8, n, seen)) break n;
        } else null;
        if (name == null) return .{ .snap = .{ .armed = snap.armed, .phase = .done, .seen = "" }, .fix = null };
        const rest = try std.mem.concat(a, u8, &.{ owed, name.?[seen.len..], g.tail });
        const ids = try g.lex.encode(g.lex.ctx, a, rest);
        if (ids.len == 0) return .{ .snap = .{ .armed = snap.armed, .phase = .done, .seen = "" }, .fix = null };
        return .{ .snap = snap, .fix = ids };
    }

    pub const Hit = struct { at: usize, fix: []const u32 };

    pub fn cut(g: *const Gate, a: Allocator, tokens: []const u32) !?Hit {
        if (g.phase == .done) return null;
        var snap: Snap = .{ .armed = g.armed, .phase = g.phase, .seen = g.seen.items };
        for (tokens, 0..) |token, i| {
            if (snap.phase == .done) return null;
            const next = try g.step(a, snap, token);
            if (next.fix) |fix| return .{ .at = i, .fix = fix };
            snap = next.snap;
        }
        return null;
    }

    pub fn observe(g: *Gate, a: Allocator, token: u32) !void {
        if (g.done()) return;
        var scratch: std.heap.ArenaAllocator = .init(a);
        defer scratch.deinit();
        const snap: Snap = .{ .armed = g.armed, .phase = g.phase, .seen = g.seen.items };
        const next = try g.step(scratch.allocator(), snap, token);
        if (next.fix != null) {
            g.phase = .done;
            g.seen.clearRetainingCapacity();
            return;
        }
        g.armed = next.snap.armed;
        g.phase = next.snap.phase;
        const seen = next.snap.seen;
        g.seen.clearRetainingCapacity();
        try g.seen.appendSlice(a, seen);
    }
};

const TestLex = struct {
    fn blank(_: *anyopaque, token: u32) bool {
        return token == 1;
    }
    fn text(_: *anyopaque, _: Allocator, token: u32) anyerror![]const u8 {
        return switch (token) {
            2 => "=",
            3 => "ab",
            4 => "ac>",
            5 => "zz",
            else => "?",
        };
    }
    fn encode(_: *anyopaque, a: Allocator, s: []const u8) anyerror![]const u32 {
        if (std.mem.eql(u8, s, "=")) return a.dupe(u32, &.{2});
        if (std.mem.eql(u8, s, "ab>")) return a.dupe(u32, &.{ 3, 4 });
        return a.dupe(u32, &.{});
    }
    fn lex() Lex {
        return .{ .ctx = undefined, .blank = blank, .text = text, .encode = encode };
    }
};

test "a required call replaces a plain answer with the opener and finishes the offered name" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var g = Gate.afterPrompt(&.{}, 9, -1, -1, TestLex.lex(), "=", &.{ "ab", "ac" }, ">");
    defer g.deinit(gpa);
    const hit = (try g.cut(a, &.{ 7, 8 })).?;
    try std.testing.expectEqual(@as(usize, 0), hit.at);
    try std.testing.expectEqualSlices(u32, &.{ 9, 2 }, hit.fix);
    try g.observe(gpa, 9);
    try g.observe(gpa, 2);
    try std.testing.expect(g.phase == .name);
    try g.observe(gpa, 3);
    try std.testing.expect(g.phase == .name);
    try g.observe(gpa, 4);
    try std.testing.expect(g.done());
}

test "whitespace before the call is kept and an open think block waits" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var g = Gate.afterPrompt(&.{ 4, 8 }, 9, 8, 4, TestLex.lex(), "", &.{}, "");
    defer g.deinit(gpa);
    try std.testing.expect(!g.armed);
    try std.testing.expect((try g.cut(a, &.{7})) == null);
    try g.observe(gpa, 4);
    try std.testing.expect(g.armed);
    const blank = (try g.cut(a, &.{1}));
    try std.testing.expect(blank == null);
}

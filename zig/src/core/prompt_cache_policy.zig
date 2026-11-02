//! Snapshot eviction order, excluding entries borrowed by unfinished prompt passes.
const std = @import("std");
const pc = @import("prompt_cache.zig");

pub fn victimBut(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or isActiveRewind(s, e, prompt)) continue;
        const now = born > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // a longer request's tokens extend it
        if ((now or extended(s, e)) and (best == null or e.used < s.entries.items[best.?].used)) best = i;
    }
    if (best) |i| return i;
    var oldest: ?usize = null;
    for (s.entries.items, 0..) |e, i| if (e.borrowers == 0 and e != skip and (oldest == null or before(s, e, s.entries.items[oldest.?], prompt))) {
        oldest = i;
    };
    return oldest;
}

/// Only live continuations make a prefix a branch point or a superseded state.
pub fn extended(s: *const pc.Store, e: *const pc.Entry) bool {
    for (s.entries.items) |o| if (!o.expired and o != e and o.born > e.born and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

fn before(s: *const pc.Store, e: *const pc.Entry, o: *const pc.Entry, prompt: []const u32) bool {
    const e_rewind = isActiveRewind(s, e, prompt);
    const o_rewind = isActiveRewind(s, o, prompt);
    return if (e_rewind != o_rewind) o_rewind else e.used < o.used;
}

fn isActiveRewind(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32) bool {
    return !e.expired and e.at == s.active_rewind_at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
}

/// The state at `at` of the `born`-token `prompt` supersedes the entry: a shorter state of the same conversation it
/// extends. A turn's next token keys the entry, so it may differ here; the states the two share are the first
/// `e.at` tokens. A later turn's shorter state (a rewind) and a reply's end (past `born`) supersede none.
pub fn subsumes(s: *const pc.Store, prompt: []const u32, born: u32, at: u32, e: *const pc.Entry) bool {
    if (at > born) return false;
    if (e.expired or e.borrowers > 0 or e.shared or e.branched or e.at >= at or e.born > born) return false;
    const n = e.at;
    return n <= prompt.len and std.mem.eql(u32, e.tokens[0..n], prompt[0..n]);
}

/// The first entry the state at `at` supersedes, or null.
pub fn subsumed(s: *const pc.Store, prompt: []const u32, born: u32, at: u32) ?usize {
    for (s.entries.items, 0..) |e, i| if (subsumes(s, prompt, born, at, e)) return i;
    return null;
}

/// The reason the victim victimBut chose left: the prompt or a kept state extends it, else least recently used.
pub fn victimReason(s: *const pc.Store, i: usize, prompt: []const u32, born: u32) []const u8 {
    const e = &s.entries.items[i];
    var reason = "least recently used";
    if (born > e.born and prompt.len > e.at and std.mem.eql(u32, prompt[0..e.at], e.tokens[0..e.at])) reason = "superseded by the prompt";
    for (s.entries.items) |o| if (o.born > e.born and o.at > e.at and std.mem.eql(u32, o.tokens[0..e.at], e.tokens[0..e.at])) {
        reason = "superseded by a kept state";
        break;
    };
    return reason;
}

//! Snapshot eviction order, excluding entries borrowed by unfinished prompt passes.
const std = @import("std");
const pc = @import("prompt_cache.zig");

/// Which states `victimButIf` may choose. `.any` is today's order. `.frees` and `.returns` skip a state the hook says frees nothing.
pub const Gate = enum { any, frees, returns };

pub fn victimBut(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    return victimIn(s, skip, prompt, born, false);
}

/// `victimBut` in the same order, keeping only a state `gate` accepts. The first superseded one that accepts wins, else the oldest that accepts.
pub fn victimButIf(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32, gate: Gate) ?usize {
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or isActiveRewind(s, e, prompt)) continue;
        if (!admits(s, e, gate)) continue;
        const now = born > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // a longer request's tokens extend it
        if (e.parked and !(now or extended(s, e))) continue; // a parked state is not a victim for room unless a later state extends it
        if ((now or extended(s, e)) and (best == null or e.used < s.entries.items[best.?].used)) best = i;
    }
    if (best) |i| return i;
    var oldest: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e.parked or e == skip or !admits(s, e, gate)) continue;
        if (oldest == null or before(s, e, s.entries.items[oldest.?], prompt)) oldest = i;
    }
    return oldest;
}

/// `victimBut`'s order over every state, or with `own_only` over the prompt's own conversation's states alone.
pub fn victimIn(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32, own_only: bool) ?usize {
    var best: ?usize = null;
    var oldest: ?usize = null;
    var oldest_rewind = false;
    for (s.entries.items, 0..) |e, i| { // one prefix compare an entry: the superseded scan and the oldest scan share it
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |g| if (g.protects(s, e)) continue;
        if (own_only and !own(e, skip, prompt)) continue;
        const on = e.tokens.len <= prompt.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
        const rewind = !e.expired and e.at == s.active_rewind_at and on; // the active rewind state goes last
        if (e.borrowers == 0 and e != skip and (oldest == null or (if (rewind != oldest_rewind) oldest_rewind else e.used < s.entries.items[oldest.?].used))) {
            oldest = i;
            oldest_rewind = rewind;
        }
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or rewind) continue;
        const now = (born > e.born or (born == e.born and !e.rewind)) and prompt.len > e.tokens.len and on;
        if ((now or extended(s, e)) and (best == null or e.used < s.entries.items[best.?].used)) best = i;
    }
    return best orelse oldest;
}

/// A reservation may free redundant copies, but not a newest state just to plan more copies.
pub fn redundant(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32, born: u32) bool {
    if (e.expired) return true;
    if (e.shared or e.branched) return false;
    const on = e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
    if (e.at == s.active_rewind_at and on) return false;
    const now = (born > e.born or (born == e.born and !e.rewind)) and prompt.len > e.tokens.len and on;
    return now or extended(s, e);
}

/// A state of the prompt's own conversation that may go: unborrowed, on its token path, neither shared nor a branch.
pub fn own(e: *const pc.Entry, skip: ?*pc.Entry, prompt: []const u32) bool {
    if (e.borrowers > 0 or e == skip or e.shared or e.branched or e.tokens.len > prompt.len) return false;
    return std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
}

fn admits(s: *const pc.Store, e: *const pc.Entry, gate: Gate) bool {
    return switch (gate) {
        .any => true,
        .frees => (s.family.vtable.frees orelse return true)(s.family.ptr, e.saved) > 0,
        .returns => (s.family.vtable.returns orelse return true)(s.family.ptr, e.saved) > 0,
    };
}

/// Only live continuations make a prefix a branch point or a superseded state.
pub fn extended(s: *const pc.Store, e: *const pc.Entry) bool {
    if (s.links_lost) return extendedScan(s, e);
    for (e.ext.items) |o| if (!o.expired and (!e.rewind or o.born > e.born)) return true;
    return false;
}

/// A real later turn, excluding an endpoint's equal-born reply end, can establish a branch.
pub fn continuedLater(s: *const pc.Store, e: *const pc.Entry) bool {
    if (!s.links_lost) {
        for (e.ext.items) |o| if (!o.expired and o.born > e.born) return true;
        return false;
    }
    for (s.entries.items) |o| if (!o.expired and o.born > e.born and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

/// `extended` over every entry, for a store whose links an allocation failure left incomplete.
pub fn extendedScan(s: *const pc.Store, e: *const pc.Entry) bool {
    for (s.entries.items) |o| if (!o.expired and o != e and (o.born > e.born or (o.born == e.born and !e.rewind)) and o.tokens.len > e.tokens.len and
        std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
    return false;
}

/// A new entry linked to those it extends and those extending it, no earlier born, including one request's reply end.
pub fn link(s: *pc.Store, x: *pc.Entry) void {
    for (s.entries.items) |o| {
        if (o == x or o.tokens.len == x.tokens.len) continue;
        const short, const long = if (o.tokens.len < x.tokens.len) .{ o, x } else .{ x, o };
        if (long.born < short.born or !std.mem.eql(u32, long.tokens[0..short.tokens.len], short.tokens)) continue;
        short.ext.append(s.gpa, long) catch {
            s.links_lost = true;
        };
    }
}

/// An entry leaving the store leaves every link to it.
pub fn unlink(s: *pc.Store, x: *pc.Entry) void {
    for (s.entries.items) |o| if (std.mem.indexOfScalar(*pc.Entry, o.ext.items, x)) |i| {
        _ = o.ext.swapRemove(i);
    };
}

/// The state at `at` of the `born`-token `prompt` supersedes the entry: a shorter state of the same conversation it
/// extends. A turn's next token keys the entry, so it may differ here; the states the two share are the first
/// `e.at` tokens. A later turn's shorter state (a rewind) and a reply's end (past `born`) supersede none.
pub fn subsumes(s: *const pc.Store, prompt: []const u32, born: u32, at: u32, e: *const pc.Entry) bool {
    if (at > born) return false;
    if (e.expired or e.borrowers > 0 or e.shared or e.branched or e.at >= at or e.born > born) return false;
    if (isActiveRewind(s, e, prompt)) return false; // the rewind state serves a later edit; its conversation's older state may not take it
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
    const e = s.entries.items[i];
    var reason: []const u8 = "least recently used";
    if (born > e.born and prompt.len > e.at and std.mem.eql(u32, prompt[0..e.at], e.tokens[0..e.at])) reason = "superseded by the prompt";
    for (s.entries.items) |o| if (o.born > e.born and o.at > e.at and std.mem.eql(u32, o.tokens[0..e.at], e.tokens[0..e.at])) {
        reason = "superseded by a kept state";
        break;
    };
    return reason;
}

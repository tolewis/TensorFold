//! One stream (Python LaneStream) and the round loop's state for it (Python's dicts keyed by stream id).
const std = @import("std");
const shape = @import("shape.zig");
const call_gate = @import("call_gate.zig");
const plan_lanes = @import("plan_lanes.zig");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;
const State = @import("depth.zig").State;

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};

pub const Mode = enum { pipe, drain, verify, exit };

pub const loop_max_period: usize = 8;
pub const loop_min_run: usize = 256;
pub const loop_warm_in: usize = 64;
pub const cal_floor: f64 = 0.2; // the lowest survival scale a stream's calibration applies
pub const cal_max: usize = 32; // the granted drafts a round keeps for its calibration
const cal_steps = 24; // bisections of a round's per-draft survival

/// Return the shortest exact cycle at the end of a combined token prefix.
fn cycleAt(head: []const u32, tail: []const u32, n: usize) ?u32 {
    if (n < loop_min_run + 1) return null;
    for (1..loop_max_period + 1) |period| {
        if (n < loop_min_run + period) break;
        if (n - loop_min_run - period < loop_warm_in) break;
        var matches = true;
        for (1..loop_min_run + 1) |k| {
            const right = n - k;
            const left = right - period;
            const right_token = if (right < head.len) head[right] else tail[right - head.len];
            const left_token = if (left < head.len) head[left] else tail[left - head.len];
            if (right_token != left_token) {
                matches = false;
                break;
            }
        }
        if (matches) return @intCast(period);
    }
    return null;
}

/// Return the first token index in a candidate batch that would fire the guard.
pub fn loopCut(s: *const Stream, tokens: []const u32) ?usize {
    if (!s.loop_guard or !s.think_open or s.think_end < 0 or s.loop_period != null) return null;
    for (tokens, 0..) |token, i| {
        if (@as(i64, token) == s.think_end) return null;
        if (cycleAt(s.emitted(), tokens, s.emitted().len + i + 1) != null) return i;
    }
    return null;
}

/// Return the detected period at the end of committed tokens, or null.
pub fn detectLoop(emitted: []const u32) ?u32 {
    return cycleAt(emitted, &.{}, emitted.len);
}

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes their tail).
pub const StopCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque, emitted: []const u32) bool,
};

/// The host's answer to whether the stream's request was cancelled, asked between prompt chunks.
pub const CancelCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
};

/// A kept prompt state the backend restores before its prompt pass (core/prompt_cache.zig), and where it keeps new ones.
pub const Reuse = struct {
    saved: ?*anyopaque = null, // the backend's copy of the state after `at` prompt tokens (null: start at 0)
    at: u32 = 0,
    marks: []const u32 = &.{}, // ascending: the pass cuts a chunk at each and calls `hook` there
    hook: ?MarkHook = null,
};

/// Called between prompt chunks, the pass standing at `at` (the GPU done with the chunk that ends there).
pub const MarkHook = struct {
    ptr: *anyopaque,
    at: *const fn (ptr: *anyopaque, s: *Stream, at: u32) void,
};

/// Drafts the backend holds for the stream's next round; a tree keeps its tokens and parents on the host.
pub const Held = struct {
    count: u32,
    tokens: ?[]u32 = null,
    parents: ?[]i32 = null,
};

/// An image in the prompt (a vision family's): the span of `tokens` positions from `at` it fills (each position carries
/// the image token id), and the payload its family's Vision.prepare made, or with `rows` its span's embedding rows
/// ready (a gate's input).
pub const Image = struct {
    at: u32,
    tokens: u32,
    bytes: []const u8,
    rows: bool = false,
};

/// A vision family's image input as a server renders it (engine_api.Info.vision): each image part becomes
/// `placeholder` (one token: `token`) in the template's text, and `prepare` turns the image's bytes into its span's
/// length and the payload the backend reads (Image.bytes), on the request's thread; on BadImage `problem` says why.
pub const Vision = struct {
    ctx: *anyopaque,
    placeholder: []const u8,
    token: u32,
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, problem: *[]const u8) PrepareError!Prepared,

    pub const Prepared = struct { tokens: u32, payload: []const u8 };
    pub const PrepareError = error{ BadImage, OutOfMemory };
};

pub const Spec = struct {
    id: []const u8,
    prompt: []const u32,
    max_new: u32,
    eos: []const u32 = &.{},
    sampling: ?Sampling = null,
    drafts: bool = true,
    proposer: ?Proposer = null,
    stop_check: ?StopCheck = null,
    cancel_check: ?CancelCheck = null,
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: i64 = -1,
    think_open: ?bool = null, // null: open when a budget is set (the server's rule)
    loop_guard: bool = false,
    chunks: []const u32 = &.{}, // where prefill chunks start after 0 (Python's PrefillPlan); empty: the backend's step
    reuse: Reuse = .{},
    images: []const Image = &.{}, // in prompt order; the caller's until the stream is released
    /// tool_choice required: the answer must open a call (null: the model may answer in text).
    call: ?struct {
        opener: u32,
        think_open: i64 = -1,
        think_end: i64 = -1,
        lead: []const u8 = "",
        names: []const []const u8 = &.{},
        tail: []const u8 = "",
        lex: call_gate.Lex,
    } = null,
};

/// Decode rounds for the other live streams between two prompt chunks (Stream.between).
pub const Between = struct {
    ptr: *anyopaque,
    /// Whether any other stream would take a round now (false: the pass keeps going, nothing joined or waited on).
    wanted: *const fn (ptr: *anyopaque) bool,
    run: *const fn (ptr: *anyopaque) anyerror!void,
};

pub const Stream = struct {
    id: []const u8,
    prompt_len: usize,
    max_new: u32,
    eos: []const u32,
    sampling: ?Sampling,
    drafts: bool,
    proposer: ?Proposer,
    stop_check: ?StopCheck,
    cancel_check: ?CancelCheck,
    think_budget: u32,
    think_close: []const u32,
    think_end: i64,
    think_open: bool,
    loop_guard: bool,
    loop_period: ?u32 = null,
    chunks: []const u32,
    reuse: Reuse = .{},
    images: []const Image = &.{},
    /// Prefill interleave (TF_PREFILL_INTERLEAVE_MS): the host's decode rounds for the OTHER live streams, which a
    /// backend may call between two of this stream's prompt chunks; null: the pass runs whole, as before.
    between: ?Between = null,
    /// A cut-in's bound (Engine.cut_in; 0: none): its pass, which runs whole between another's pieces, computes at
    /// most this many tokens past the state it resumes; a backend that finds the state gone as the pass opens refuses
    /// it (error.FillBusy) before it runs, and it waits for that pass as a long prompt does.
    cut_in: u32 = 0,
    cached: u32 = 0, // prompt tokens the backend restored from `reuse` (its prompt pass started there)
    reuse_failed: bool = false, // the backend's restore of `reuse` failed: it prefilled from 0
    context: std.ArrayList(u32) = .empty,
    pending: ?u32 = null,
    force: std.ArrayList(u32) = .empty,
    cache_len: u64 = 0,
    finished: bool = false,
    reason: Reason = .none,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    branch_rows: u64 = 0, // suffix-match branch rows verified (lanes/fill.zig)
    branch_accepted: u64 = 0, // of those, rows on the accepted path
    graft_hit: f64 = 1, // recent share of grafted rows kept; a read of the chain that grafted none counts as a miss
    odds: ?shape.Odds = null, // how often the target took the head's rank-r lane at each depth (head trees)
    lanes: ?shape.Shape = null, // the shape of the held tree drafts
    landed: [shape.max_depth][shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes verified and taken
    picks: [plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks: each tree size, then chain + grafts
    graft_room: ?u32 = null, // grafted rows the planner left this round's window (null: no planner)
    held_levels: u32 = 0, // head levels drafted for the held window
    round_over: ?f64 = null, // the stream's round time beyond its window and head levels (ms, recent): chains
    tree_over: ?f64 = null, // and trees
    measured: [plan_lanes.sizes.len + 1]plan_lanes.Measured = @splat(.{}), // each pick's round times past its table price
    priced: ?struct { choice: usize, ms: f64 } = null, // the pick that shaped the held window, at its table price
    min_rows: u32 = 0,
    paused: bool = false,
    depth: ?State = null,
    mode: ?Mode = null,
    next: ?Held = null,
    inflight: ?u64 = null,
    served: ?u64 = null,
    granted: ?u32 = null,
    cal: f64 = 1.0, // realised over predicted drafts, a moving average (TF_DS_T25_DRAFT_CAL)
    cal_held: u32 = 0, // head drafts offered before the grant (TF_DS_T25_CAL_LOG held)
    cal_held_raw: [cal_max]f64 = @splat(0.0), // their raw cumulative chances (held_chances)
    cal_raw: [cal_max]f64 = @splat(0.0), // the granted drafts' cumulative chances this round, before calibration
    cal_n: usize = 0, // and how many (0: none)
    cal_kept: f64 = 0.0, // moving averages of the drafts kept and of the drafts the chances expected, over verified rounds
    cal_sum: f64 = 0.0, // (0: no round yet)
    cal_c: f64 = 1.0, // the per-draft survival the last verified round solved for
    cal_rounds: u64 = 0, // calibrated verifies, for the log
    copy_width: ?u32 = null,
    gate: ?call_gate.Gate = null,

    pub fn init(gpa: Allocator, spec: Spec) !Stream {
        var s: Stream = .{
            .id = spec.id,
            .prompt_len = spec.prompt.len,
            .max_new = spec.max_new,
            .eos = spec.eos,
            .sampling = spec.sampling,
            .drafts = spec.drafts,
            .proposer = if (spec.drafts) spec.proposer else null,
            .stop_check = spec.stop_check,
            .cancel_check = spec.cancel_check,
            .think_budget = spec.think_budget,
            .think_close = spec.think_close,
            .think_end = spec.think_end,
            .think_open = spec.think_open orelse (spec.think_budget > 0 or (spec.loop_guard and spec.think_end >= 0)),
            .loop_guard = spec.loop_guard,
            .chunks = spec.chunks,
            .reuse = spec.reuse,
            .images = spec.images,
        };
        if (spec.call) |c| {
            s.gate = call_gate.Gate.afterPrompt(spec.prompt, c.opener, c.think_open, c.think_end, c.lex, c.lead, c.names, c.tail);
        }
        try s.context.appendSlice(gpa, spec.prompt);
        return s;
    }

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        if (s.gate) |*g| g.deinit(gpa);
        s.context.deinit(gpa);
        s.force.deinit(gpa);
        s.dropRoundState(gpa);
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    pub fn dropRoundState(s: *Stream, gpa: Allocator) void {
        if (s.depth) |*d| d.deinit(gpa);
        s.dropHeld(gpa);
        s.depth = null;
        s.mode = null;
        s.inflight = null;
        s.served = null;
        s.granted = null;
        s.cal_n = 0;
        s.cal_held = 0;
        s.copy_width = null;
    }

    pub fn dropHeld(s: *Stream, gpa: Allocator) void {
        if (s.next) |h| {
            if (h.tokens) |t| gpa.free(t);
            if (h.parents) |p| gpa.free(p);
        }
        s.next = null;
        if (s.lanes) |l| l.deinit(gpa);
        s.lanes = null;
    }

    pub fn prompt(s: *const Stream) []const u32 {
        return s.context.items[0..s.prompt_len];
    }

    pub fn emitted(s: *const Stream) []const u32 {
        return s.context.items[s.prompt_len..];
    }

    pub fn budgetLeft(s: *const Stream) i64 {
        return @as(i64, s.max_new) - @as(i64, @intCast(s.emitted().len));
    }

    /// Whether the request was cancelled (false with no hook); a prompt pass asks before each chunk.
    pub fn isCancelled(s: *const Stream) bool {
        const c = s.cancel_check orelse return false;
        return c.check(c.ptr);
    }

    fn budgetActive(s: *const Stream) bool {
        return s.think_open and s.think_budget > 0 and s.think_close.len > 0;
    }

    /// The index the thinking budget replaces with `think_close[0]`, or null if the block closed first.
    pub fn thinkCut(s: *const Stream, tokens: anytype) ?usize {
        if (!s.budgetActive()) return null;
        const done = s.emitted().len;
        for (tokens, 0..) |t, i| {
            if (done + i + 1 >= s.think_budget) return i;
            if (@as(i64, t) == s.think_end) return null;
        }
        return null;
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    pub fn cutsNext(s: *const Stream) bool {
        return s.budgetActive() and s.emitted().len + 1 >= s.think_budget;
    }

    /// Begin the thinking budget's close: its first token now, the rest forced after it.
    pub fn startClose(s: *Stream, gpa: Allocator) !u32 {
        s.think_open = false;
        s.force.clearRetainingCapacity();
        try s.force.appendSlice(gpa, s.think_close[1..]);
        return s.think_close[0];
    }

    /// Tokens a round may commit before the length limit or the thinking budget's cut.
    pub fn draftRoom(s: *const Stream) i64 {
        var room = s.budgetLeft();
        if (s.budgetActive()) room = @min(room, @as(i64, s.think_budget) - @as(i64, @intCast(s.emitted().len)));
        return room;
    }

    pub fn isEos(s: *const Stream, token: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, token) != null;
    }

    /// The survival scale the allocation uses: a cold or noisy stream cannot collapse its depth to nothing.
    pub fn calScale(s: *const Stream, ceil: f64) f64 {
        return std.math.clamp(s.cal, cal_floor, ceil);
    }

    /// Raw cumulative chances for each held draft before a grant trims them (TF_DS_T25_CAL_LOG).
    pub fn rememberHeld(s: *Stream, chances: []const f64) void {
        const n = @min(chances.len, cal_max);
        s.cal_held = @intCast(chances.len);
        @memcpy(s.cal_held_raw[0..n], chances[0..n]);
    }

    /// Remember the granted drafts' raw cumulative chances for the round's verify.
    pub fn predict(s: *Stream, chances: []const f64) void {
        s.cal_n = @min(chances.len, cal_max);
        @memcpy(s.cal_raw[0..s.cal_n], chances[0..s.cal_n]);
    }

    /// The drafts the allocation expected to land this round.
    pub fn predicted(s: *const Stream) f64 {
        var sum: f64 = 0.0;
        for (s.cal_raw[0..s.cal_n]) |p| sum += p;
        return sum;
    }

    /// Fold the round into the averages of kept and expected drafts, weight `w`, and set `cal` to the per-draft
    /// survival whose expected drafts are the average's share of this round's chances. A per-round ratio would bias
    /// `cal` low: its noise is clipped at 1 on one side only.
    pub fn observeCal(s: *Stream, w: f64, kept: f64) void {
        const n = s.cal_n;
        if (n == 0) return;
        const expected = s.predicted();
        if (!(expected > 0.0)) return;
        const realised = std.math.clamp(kept, 0.0, @as(f64, @floatFromInt(n)));
        if (s.cal_sum > 0.0) {
            s.cal_kept = (1.0 - w) * s.cal_kept + w * realised;
            s.cal_sum = (1.0 - w) * s.cal_sum + w * expected;
        } else {
            s.cal_kept = realised;
            s.cal_sum = expected;
        }
        s.cal_c = survival(s.cal_raw[0..n], s.cal_kept / s.cal_sum * expected);
        s.cal = s.cal_c;
    }

    /// One line on stderr for the probe: the round, what it predicted, and what the next allocation will see.
    pub fn logCal(s: *Stream, every: u32, ceil: f64, rows: usize, kept: usize, tokens: usize) void {
        s.cal_rounds += 1;
        if (every == 0 or s.cal_rounds % every != 0) return;
        var buf: [1536]u8 = undefined;
        var at: usize = 0;
        const n = s.cal_n;
        const held_n = @min(s.cal_held, cal_max);
        at += (std.fmt.bufPrint(buf[at..], "tf-cal {{\"stream\": \"{s}\", \"rows\": {d}, \"held\": {d}, \"held_chances\": [", .{ s.id, rows, s.cal_held }) catch return).len;
        for (s.cal_held_raw[0..held_n], 0..) |p, i| {
            at += (std.fmt.bufPrint(buf[at..], "{s}{d:.3}", .{ if (i > 0) ", " else "", p }) catch return).len;
        }
        at += (std.fmt.bufPrint(buf[at..], "], \"granted\": {d}, \"kept\": {d}, \"pred_sum\": {d:.3}, \"chances\": [", .{ n, kept, s.predicted() }) catch return).len;
        for (s.cal_raw[0..n], 0..) |p, i| at += (std.fmt.bufPrint(buf[at..], "{s}{d:.3}", .{ if (i > 0) ", " else "", p }) catch return).len;
        at += (std.fmt.bufPrint(buf[at..], "], \"c\": {d:.3}, \"cal\": {d:.3}, \"next\": [", .{ s.cal_c, s.cal }) catch return).len;
        const scale = s.calScale(ceil);
        var factor: f64 = 1.0;
        for (s.cal_raw[0..n], 0..) |p, i| {
            factor *= scale;
            at += (std.fmt.bufPrint(buf[at..], "{s}{d:.3}", .{ if (i > 0) ", " else "", p * factor }) catch return).len;
        }
        at += (std.fmt.bufPrint(buf[at..], "], \"tokens\": {d}}}\n", .{tokens}) catch return).len;
        std.debug.print("{s}", .{buf[0..at]});
    }

    /// The per-draft survival c in [cal_floor, 1] with sum_i c^i * chances[i-1] = kept (kept clamped to the drafts).
    pub fn survival(chances: []const f64, kept: f64) f64 {
        var total: f64 = 0.0;
        for (chances) |p| total += p;
        const want = std.math.clamp(kept, 0.0, @as(f64, @floatFromInt(chances.len)));
        if (want >= total) return 1.0;
        if (expectedKept(chances, cal_floor) >= want) return cal_floor;
        var lo: f64 = cal_floor;
        var hi: f64 = 1.0;
        for (0..cal_steps) |_| {
            const mid = 0.5 * (lo + hi);
            if (expectedKept(chances, mid) < want) lo = mid else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    fn expectedKept(chances: []const f64, c: f64) f64 {
        var sum: f64 = 0.0;
        var factor: f64 = 1.0;
        for (chances) |p| {
            factor *= c;
            sum += factor * p;
        }
        return sum;
    }

    /// Append tokens until the stream finishes; how many landed (a prefix of `tokens`).
    pub fn commit(s: *Stream, gpa: Allocator, tokens: []const u32) !usize {
        var landed: usize = 0;
        for (tokens) |t| {
            if (s.finished) break;
            try s.context.append(gpa, t);
            landed += 1;
            if (s.gate) |*g| g.observe(gpa, t) catch return error.OutOfMemory;
            if (@as(i64, t) == s.think_end) s.think_open = false;
            const fire = if (s.loop_guard and s.think_open and s.think_end >= 0 and s.loop_period == null) detectLoop(s.emitted()) else null;
            if (s.isEos(t) or s.stopped()) {
                s.finished = true;
                s.reason = .stop;
            } else if (fire) |period| {
                s.loop_period = period;
                if (s.emitted().len >= s.max_new) {
                    s.finished = true;
                    s.reason = .length;
                } else if (s.think_close.len == 0) {
                    s.finished = true;
                    s.reason = .stop;
                } else {
                    s.think_open = false;
                    s.force.clearRetainingCapacity();
                    try s.force.appendSlice(gpa, s.think_close);
                }
                break;
            } else if (s.emitted().len >= s.max_new) {
                s.finished = true;
                s.reason = .length;
            }
        }
        return landed;
    }

    fn stopped(s: *const Stream) bool {
        const c = s.stop_check orelse return false;
        return c.check(c.ptr, s.emitted());
    }

    pub fn popForce(s: *Stream) ?u32 {
        if (s.force.items.len == 0) return null;
        return s.force.orderedRemove(0);
    }
};

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}

test "loop guard stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop guard fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items));
    try std.testing.expectEqual(@as(u32, 1), s.loop_period.?);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8});
    try std.testing.expect(!s.finished and s.loop_period.? == 1);
}

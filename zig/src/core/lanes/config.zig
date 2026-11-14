//! The round loop's constants from the model's attributes and the engine's settings (Python _family_setup).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
const Constants = @import("depth.zig").Constants;

pub const Cost = alloc.Cost;

/// TF_DS_T25_CAL_CEIL: finite, from 1.0 (default) to 2.0.
pub fn parseCalCeil(text: []const u8) !f64 {
    const v = try std.fmt.parseFloat(f64, std.mem.trim(u8, text, " "));
    if (!std.math.isFinite(v) or v < 1.0 or v > 2.0) return error.BadCalCeil;
    return v;
}

/// TF_DS_T25_FULL_AT: finite, from 0.5 (a low kept share) to 1.0 (always full grants).
pub fn parseFullAt(text: []const u8) !f64 {
    const v = try std.fmt.parseFloat(f64, std.mem.trim(u8, text, " "));
    if (!std.math.isFinite(v) or v < 0.5 or v > 1.0) return error.BadFullAt;
    return v;
}

/// The class default acceptance by depth (Python DraftDepth.depth_prior).
pub const default_prior = [_]f64{ 0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

/// What a backend reports about its loaded model: the attributes the Python engine reads at setup.
pub const Model = struct {
    solo_head_overhead_ms: ?f64 = null, // fixed overhead for a lone head window; null keeps the live estimate
    draft_cal: f64 = 0.0, // the weight of a round in each stream's realised-over-predicted draft survival (0: off)
    cal_log: u32 = 0, // print every n-th calibrated verify of a stream to stderr (0: off)
    cal_ceil: f64 = 1.0, // TF_DS_T25_CAL_CEIL: upper clamp of calScale (unset: 1.0)
    full_at: ?f64 = null, // TF_DS_T25_FULL_AT: head grants use chance 1.0 when kept share >= this (unset: off)
    fix_k: ?u32 = null, // TF_DS_FIX_K: every stream verifies K drafts (wins over full_at)
    copy_proposals: bool = true, // suffix-lookup copy windows; false opts a backend out
    exact_width: u32 = 1, // the widest window that reproduces one-token steps bit for bit
    first_copy_rows: u32 = 0, // a lone stream's first copy window (0: the exact width)
    gpu_tokens: bool = false, // one-token rounds may run a step ahead (pipelined)
    mtp: bool = false, // a draft head is loaded
    speculate: bool = false, // the head can draft from a verify's rows
    speculate_early: bool = true, // draft every row's first draft behind the verify
    draft_prior: []const f64 = &.{}, // the head's acceptance by depth until a stream has its own
    plain_guard: bool = false, // plain rounds compete with drafted depths
    drafts: u32 = 1, // head drafts a round at most
    window_costs: []const Cost = &.{}, // forward ms by exact window width (timed at load)
    mtp_step_ms: f64 = 0.0, // one chained head step (timed at load)
    streams_exact: bool = true, // shared forwards keep each stream's bits
    hidden_rows: bool = false, // the backend runs several streams' windows in one forward
    batch_rows: u32 = 0, // a shared forward's rows at most (0: 32)
    max_streams: u32 = 0, // a shared forward's streams at most (0: 32)
    shared_costs: []const Cost = &.{}, // shared forwards' ms by total rows (timed at load)
    // shared forwards' ms by total rows for a round of 2, 3, ... streams (the first table: two streams). A row of
    // another stream and a draft row of the same stream cost differently, so one table by rows cannot price both.
    // An empty table, or a count past the last, takes shared_costs
    stream_costs: []const []const Cost = &.{},
    draft_probabilities: bool = false, // the head gives each drafted node's chance of landing
    draft_streams: bool = false, // the head drafts every stream of a shared round in one batch
    head_trees: bool = false, // the head drafts trees of lanes (DraftRequest.lanes) and holds them
    lane_costs: []const Cost = &.{}, // a lone stream's window ms by rows, up to its widest tree (timed at load)
};

pub const Config = struct {
    solo_head_overhead_ms: ?f64 = null, // fixed overhead for a lone head window; null keeps the live estimate
    draft_cal: f64 = 0.0, // Model.draft_cal
    cal_log: u32 = 0, // Model.cal_log
    cal_ceil: f64 = 1.0, // Model.cal_ceil
    full_at: ?f64 = null, // Model.full_at
    fix_k: ?u32 = null, // Model.fix_k
    copy_proposals: bool = true, // suffix-lookup copy windows; false opts a backend out
    family_width: u32,
    max_copy: u32,
    first_copy: u32,
    base_width: u32,
    family_mtp: bool,
    speculate_early: bool,
    pipelined: bool,
    plain_guard: bool,
    most_drafts: u32,
    family_streams: bool,
    batch_rows: u32,
    batch_streams: u32,
    node_probabilities: bool,
    fill_lanes: u32 = 0, // rows a window may grow to with suffix-match branches (0: off; lanes/fill.zig)
    fill_match: u32 = 8, // context tokens a suffix-match branch must match (shorter ones are coincidental phrases)
    fill_floor: f64 = 0.35, // the share of grafted rows a stream must keep to go on reading its chain back every round
    draft_streams: bool,
    head_trees: bool,
    head_lanes: u32 = 0, // a lone stream's lanes at most, the planner picking each round's (0: the depth rule's chain)
    head_depth: u32 = 16, // and their depth at most
    own_prices: bool = true, // the planner prices each pick from the stream's own measured rounds (seeded from the table)
    depth_prior: []f64,
    family_costs: Table,
    shared_costs: Table,
    stream_costs: []Table = &.{}, // Model.stream_costs, each extended to batch_rows (an empty table: none)
    lane_costs: Table, // window ms by rows to 64 (lanes/plan_lanes.zig)
    mtp_step_ms: f64,
    enter_match: i64 = 8, // matching tokens a copy needs (rejects coincidental indentation)
    copy_rate: f64 = 0.94, // a copied token's chance of landing in allocation
    k: Constants = .{},

    pub fn init(gpa: Allocator, m: Model, max_rows: u32, max_draft: u32) !Config {
        if (!std.math.isFinite(m.draft_cal) or m.draft_cal < 0.0 or m.draft_cal > 1.0) return error.BadDraftCal;
        if (!std.math.isFinite(m.cal_ceil) or m.cal_ceil < 1.0 or m.cal_ceil > 2.0) return error.BadCalCeil;
        if (max_rows < 1) return error.BadMaxRows;
        if (m.solo_head_overhead_ms) |cost| {
            if (!std.math.isFinite(cost) or cost < 0.0) return error.BadSoloHeadOverhead;
        }
        const width: u32 = @max(1, @min(@max(m.exact_width, 1), max_rows));
        const max_copy = width - 1;
        const first_rows: i64 = if (m.first_copy_rows != 0) m.first_copy_rows else width;
        const first_copy: u32 = @intCast(@max(1, @min(@as(i64, max_copy), first_rows - 1)));
        const prior = if (m.draft_prior.len > 0) m.draft_prior else &default_prior;
        const streams = width >= 2 and m.streams_exact and m.hidden_rows;
        const batch_rows: u32 = if (m.batch_rows != 0) m.batch_rows else 32;
        var family_costs: Table = .{};
        errdefer family_costs.deinit(gpa);
        for (m.window_costs) |c| {
            if (c.width >= 1 and c.width <= width) try family_costs.put(gpa, c.width, c.ms);
        }
        // shared forwards: window costs then timed shared rows (a later table overrides), extended to batch_rows
        var timed: std.ArrayList(Cost) = .empty;
        defer timed.deinit(gpa);
        for ([_][]const Cost{ m.window_costs, m.shared_costs }) |table| {
            for (table) |c| {
                if (c.width < 1 or c.width > batch_rows) continue;
                for (timed.items) |*t| {
                    if (t.width == c.width) {
                        t.ms = c.ms;
                        break;
                    }
                } else try timed.append(gpa, c);
            }
        }
        const depth_prior = try gpa.dupe(f64, prior);
        errdefer gpa.free(depth_prior);
        // a round's costs by its streams: each table as given, extended like the shared one
        const stream_costs = try gpa.alloc(Table, m.stream_costs.len);
        for (stream_costs) |*t| t.* = .{};
        errdefer {
            for (stream_costs) |*t| t.deinit(gpa);
            gpa.free(stream_costs);
        }
        for (stream_costs, m.stream_costs) |*t, given| {
            var kept: std.ArrayList(Cost) = .empty;
            defer kept.deinit(gpa);
            for (given) |c| {
                if (c.width >= 1 and c.width <= batch_rows) try kept.append(gpa, c);
            }
            t.* = try alloc.extendCosts(gpa, kept.items, batch_rows);
        }
        return .{
            .family_width = width,
            .max_copy = max_copy,
            .first_copy = first_copy,
            .base_width = @min(width, first_copy + 1),
            .family_mtp = m.mtp and width >= 2 and m.speculate,
            .speculate_early = m.speculate_early,
            .pipelined = m.gpu_tokens,
            .plain_guard = m.plain_guard,
            .most_drafts = @min(m.drafts, @min(width - 1, max_draft)),
            .family_streams = streams,
            .batch_rows = batch_rows,
            .batch_streams = if (streams) (if (m.max_streams != 0) m.max_streams else 32) else 1,
            .node_probabilities = m.draft_probabilities,
            .draft_streams = m.draft_streams,
            .head_trees = m.head_trees,
            .depth_prior = depth_prior,
            .family_costs = family_costs,
            .shared_costs = try alloc.extendCosts(gpa, timed.items, batch_rows),
            .stream_costs = stream_costs,
            .lane_costs = try alloc.extendCosts(gpa, m.lane_costs, 64),
            .mtp_step_ms = m.mtp_step_ms,
            .solo_head_overhead_ms = m.solo_head_overhead_ms,
            .draft_cal = m.draft_cal,
            .cal_log = m.cal_log,
            .cal_ceil = m.cal_ceil,
            .full_at = m.full_at,
            .fix_k = m.fix_k,
            .copy_proposals = m.copy_proposals,
        };
    }

    pub fn deinit(c: *Config, gpa: Allocator) void {
        gpa.free(c.depth_prior);
        c.family_costs.deinit(gpa);
        c.shared_costs.deinit(gpa);
        for (c.stream_costs) |*t| t.deinit(gpa);
        gpa.free(c.stream_costs);
        c.lane_costs.deinit(gpa);
    }

    /// The cost table of a shared forward of this many streams: its own (Model.stream_costs) when one is given.
    pub fn sharedCosts(c: *const Config, streams: usize) *const Table {
        return streamTable(c.stream_costs, &c.shared_costs, streams);
    }
};

/// `tables[streams - 2]` when it is there and not empty, else `shared`.
pub fn streamTable(tables: []const Table, shared: *const Table, streams: usize) *const Table {
    if (streams >= 2 and streams - 2 < tables.len and !tables[streams - 2].empty()) return &tables[streams - 2];
    return shared;
}

test "a round of several streams takes its own cost table when one is given" {
    const gpa = std.testing.allocator;
    const one = [_]Cost{ .{ .width = 1, .ms = 20.0 }, .{ .width = 2, .ms = 24.0 }, .{ .width = 3, .ms = 28.0 } };
    const two = [_]Cost{ .{ .width = 2, .ms = 27.0 }, .{ .width = 3, .ms = 31.0 }, .{ .width = 4, .ms = 35.0 } };
    // no table for three streams, one for four
    const four = [_]Cost{ .{ .width = 4, .ms = 40.0 }, .{ .width = 5, .ms = 44.5 } };
    const by = [_][]const Cost{ &two, &.{}, &four };
    var c = try Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true, .batch_rows = 8, .shared_costs = &one, .stream_costs = &by }, 8, 7);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 24.0), c.sharedCosts(1).get(2));
    try std.testing.expectEqual(@as(?f64, 27.0), c.sharedCosts(2).get(2));
    try std.testing.expectEqual(@as(?f64, 35.0), c.sharedCosts(2).get(4));
    try std.testing.expectEqual(@as(?f64, 24.0), c.sharedCosts(3).get(2)); // (an empty table: the shared one)
    try std.testing.expectEqual(@as(?f64, 44.5), c.sharedCosts(4).get(5));
    try std.testing.expectEqual(@as(?f64, 53.4), c.sharedCosts(4).get(6)); // (past the widest: its ms a row)
    try std.testing.expectEqual(@as(?f64, 28.0), c.sharedCosts(5).get(3)); // (a count past the last table)
    // unset: every count takes the shared table
    var plain = try Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true, .batch_rows = 8, .shared_costs = &one }, 8, 7);
    defer plain.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 28.0), plain.sharedCosts(4).get(3));
}

test "setup derives widths like Python" {
    const gpa = std.testing.allocator;
    var c = try Config.init(gpa, .{ .exact_width = 64, .first_copy_rows = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .drafts = 4, .hidden_rows = true, .batch_rows = 128, .max_streams = 64 }, 64, 63);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 15), c.first_copy);
    try std.testing.expectEqual(@as(u32, 16), c.base_width);
    try std.testing.expectEqual(@as(u32, 4), c.most_drafts);
    try std.testing.expect(c.family_mtp and c.family_streams);
}

test "solo allowance is opt-in and preserved by setup" {
    const a = std.testing.allocator;
    var original = try Config.init(a, .{}, 8, 7);
    defer original.deinit(a);
    try std.testing.expect(original.solo_head_overhead_ms == null);
    var selected = try Config.init(a, .{ .solo_head_overhead_ms = 3.5 }, 8, 7);
    defer selected.deinit(a);
    try std.testing.expectEqual(@as(?f64, 3.5), selected.solo_head_overhead_ms);
}
test "solo allowance rejects invalid costs before allocating" {
    for ([_]f64{ -1.0, std.math.inf(f64), std.math.nan(f64) }) |cost| {
        try std.testing.expectError(error.BadSoloHeadOverhead, Config.init(std.testing.allocator, .{ .solo_head_overhead_ms = cost }, 8, 7));
    }
}
test "copy windows stay on unless a backend opts out" {
    const a = std.testing.allocator;
    var original = try Config.init(a, .{}, 8, 7);
    defer original.deinit(a);
    try std.testing.expect(original.copy_proposals);
    var off = try Config.init(a, .{ .copy_proposals = false }, 8, 7);
    defer off.deinit(a);
    try std.testing.expect(!off.copy_proposals);
}

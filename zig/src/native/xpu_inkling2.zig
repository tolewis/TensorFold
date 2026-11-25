//! Text-only Inkling 2 Mini on the XPU server (Intel Xe2 over Level Zero), with exact prompt reuse, budgeted stream
//! state and the checkpoint's DSpark head. The Metal engine's opener (native/inkling2.zig) on the device runtime.
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const lanes = @import("lanes");
const ink = @import("inkling2_xpu");
const Model = ink.model.Model;
const st = ink.state;
const backend = ink.backend;
const dspark = ink.dspark;
const dflash2 = ink.dflash2;

const Allocator = std.mem.Allocator;

/// Prompt plus reply tokens when --context names none (and the card holds more).
const default_context: u64 = 32768;
/// Device memory kept free beside the engine: the driver's own use, activations of later chunks, the desktop's growth.
const margin: u64 = 1536 << 20;
/// The prompt cache's budget when --prompt-cache-gib names none.
const default_cache_gib: f64 = 2.0;

const Host = struct {
    gpa: Allocator,
    driver: xpu.Driver,
    ctx: xpu.Context,
    stream: xpu.Stream,
    rt: xpu.rt.Runtime,
    model: *Model,
    lane: *backend.Xpu,
    drafter: ?*dspark.Drafter = null,
    flash: ?*dflash2.Drafter = null,
    drafted: ?lanes.drafted.Drafted = null,
    clock: lanes.backend.WallClock,
    config: lanes.Config = undefined,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null,
    cache_cap: u64 = 0,

    fn close(ptr: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        if (h.drafted) |*d| d.deinit();
        if (h.drafter) |d| d.deinit();
        if (h.flash) |d| d.deinit();
        h.lane.deinit();
        h.model.deinit();
        h.rt.deinit();
        h.stream.deinit();
        h.ctx.deinit();
        h.driver.close();
        h.gpa.destroy(h);
    }
};

pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.vision != null or o.cfg_enabled or o.cfg_batched) return "inkling2_loop: this XPU engine serves text only; images and audio are unsupported";
    if (o.loops != null) return "inkling2_loop: the model has exactly three core passes; --loops cannot override them";
    if (o.mtp_drafts != null or o.mtp_confidence != null) return "inkling2_loop: use --drafter for the checkpoint's DSpark head; external draft controls remain unsupported";
    if (o.precision != null) return "inkling2_loop: the checkpoint's NVFP4, MXFP8 and BF16 arithmetic is fixed; serve without --precision";
    if (o.speed_up != null) return "inkling2_loop: this XPU engine serves one card; --speed-up is unsupported";
    if (o.tp != 1 and o.tp != 2) return "inkling2_loop: --tp is 1 or 2";
    if (o.learn != null or o.learn_conversations) return "inkling2_loop: learned disk states are not implemented; serve without --learn and --learn-conversations";
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0 or g > 1024 * 1024) return "inkling2_loop: --prompt-cache-gib must be a finite, nonnegative budget";
    return null;
}

fn drafterDirectory(o: api.Open) ?[]const u8 {
    return if (o.drafts) o.drafter else null;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// Device memory left: Sysman's free bytes, at most the card's size less what this engine allocated.
fn freeMemory(ctx: *xpu.Context) u64 {
    const mem = ctx.gpu.memory() catch return 0;
    return @min(mem.free, mem.size -| xpu.rt.alloc_now);
}

fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const m: api.Memory = .{ .active = xpu.rt.alloc_now, .cache = 0, .peak = xpu.rt.alloc_peak };
    if (reset_peak) xpu.rt.alloc_peak = xpu.rt.alloc_now;
    return m;
}

/// Which drafter `--drafter` names: Mini's DSpark head, or the DFlash 2 model (its config names the architecture).
const Draft = enum {
    none,
    dspark,
    dflash2,

    fn kind(gpa: Allocator, io: std.Io, dir: ?[]const u8) Draft {
        const d = dir orelse return .none;
        return if (dflash2.loader.isDraft(gpa, io, d)) .dflash2 else .dspark;
    }
    fn streamBytes(k: Draft, positions: usize) u64 {
        return switch (k) {
            .none => 0,
            .dspark => dspark.Drafter.streamBytes(positions),
            .dflash2 => dflash2.Drafter.streamBytes(positions),
        };
    }
    fn scoreBytes(k: Draft, positions: usize) u64 {
        return switch (k) {
            .none => 0,
            .dspark => dspark.Drafter.scoreBytes(positions),
            .dflash2 => dflash2.Drafter.scoreBytes(positions),
        };
    }
    fn label(k: Draft) []const u8 {
        return switch (k) {
            .none => "",
            .dspark => "DSpark",
            .dflash2 => "DFlash2",
        };
    }
};

/// What `positions` of context cost beside the weights: the scratch, `streams` states and, with a drafter, its rings and scores.
fn contextBytes(positions: usize, streams: u64, drafter: Draft) u64 {
    const lane: u64 = if (drafter != .none) streams * drafter.streamBytes(positions) + drafter.scoreBytes(positions) else 0;
    return backend.scratchBytes(positions) + backend.streamBytes(positions) + lane;
}

/// The longest context (at most `most`) a budget of `bytes` holds with one stream; zero when none does.
fn longest(bytes: u64, most: usize, streams: u64, drafter: Draft) usize {
    if (contextBytes(1, streams, drafter) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (contextBytes(mid, streams, drafter) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

fn contextProblem(window: i64, fits: usize) ?[]const u8 {
    if (window < 1) return "inkling2_loop: --context must be at least one token, and the selected context must fit the card beside the checkpoint";
    if (window > fits) return "inkling2_loop: --context exceeds the largest window this card can hold beside the checkpoint and DSpark state";
    return null;
}

/// Prompt-cache states leave room for the streams: its budget is what the live streams do not hold.
fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = if (h.cache) |*owned| owned else return;
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (refusal(o)) |why| {
        problem.* = why;
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.driver = xpu.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "cannot open the Level Zero driver ({s})", .{@errorName(e)});
        gpa.destroy(h);
        return null;
    };
    errdefer h.driver.close();
    h.ctx = xpu.Context.init(&h.driver, deviceOrdinal(o.device)) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no Intel GPU device ({s})", .{@errorName(e)});
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    errdefer h.ctx.deinit();
    h.stream = try xpu.Stream.init(&h.ctx);
    errdefer h.stream.deinit();
    h.rt = xpu.rt.Runtime.init(&h.ctx, h.stream) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine cannot start: {s}", .{@errorName(e)});
        h.stream.deinit();
        h.ctx.deinit();
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    errdefer h.rt.deinit();
    const embed_host = if (getenv("XPU_EMBED_HOST")) |v| std.mem.eql(u8, v, "1") else false;
    const place = ink.place.Place.parse(o.tp, o.rank) catch {
        problem.* = "inkling2_loop: --tp is 1 or 2, and --rank is below --tp";
        h.rt.deinit();
        h.stream.deinit();
        h.ctx.deinit();
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    if (std.c.getenv("TF_TP_NOMODEL") != null and place.split()) {
        _ = ink.group.Group.open(gpa, &h.rt, &h.ctx, place) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "inkling2_loop: the TP group did not open ({s})", .{@errorName(err)});
            return null;
        };
        std.process.exit(0);
    }
    h.model = Model.load(gpa, &h.rt, o.dir, .{ .embed_host = embed_host, .place = place }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(err)});
        h.rt.deinit();
        h.stream.deinit();
        h.ctx.deinit();
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    errdefer h.model.deinit();
    h.model.loadKernels(&h.ctx) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot build its kernels ({s})", .{@errorName(err)});
        h.model.deinit();
        h.rt.deinit();
        h.stream.deinit();
        h.ctx.deinit();
        h.driver.close();
        gpa.destroy(h);
        return null;
    };
    if (place.split() and place.rank == 0) {
        _ = ink.group.Group.open(gpa, &h.rt, &h.ctx, place) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "inkling2_loop: the TP group did not open ({s})", .{@errorName(err)});
            h.model.deinit();
            h.rt.deinit();
            h.stream.deinit();
            h.ctx.deinit();
            h.driver.close();
            gpa.destroy(h);
            return null;
        };
    }
    const drafter_dir = drafterDirectory(o);
    h.drafter = null;
    h.flash = null;
    const draft_kind = Draft.kind(gpa, io, drafter_dir);
    if (drafter_dir) |dir| {
        const loaded = if (draft_kind == .dflash2) blk: {
            h.flash = dflash2.Drafter.init(gpa, io, h.model, dir) catch |err| break :blk err;
            break :blk {};
        } else blk: {
            h.drafter = dspark.Drafter.init(gpa, io, h.model, dir) catch |err| break :blk err;
            break :blk {};
        };
        loaded catch |err| {
            problem.* = try std.fmt.allocPrint(a, "--drafter: Inkling 2 Mini {s} cannot load {s} ({s})", .{ draft_kind.label(), dir, @errorName(err) });
            h.model.deinit();
            h.rt.deinit();
            h.stream.deinit();
            h.ctx.deinit();
            h.driver.close();
            gpa.destroy(h);
            return null;
        };
    }
    errdefer if (h.drafter) |d| d.deinit();
    errdefer if (h.flash) |d| d.deinit();
    const free = freeMemory(&h.ctx);
    const streams: u64 = @max(o.lanes, 1);
    const room: u64 = if (free > margin) free - margin else 0;
    const fits = longest(room, h.model.config.text.max_context, streams, draft_kind);
    const window: i64 = o.context orelse @intCast(@min(fits, default_context));
    if (contextProblem(window, fits)) |why| {
        problem.* = why;
        if (h.drafter) |d| d.deinit();
        if (h.flash) |d| d.deinit();
        h.model.deinit();
        h.rt.deinit();
        h.stream.deinit();
        h.ctx.deinit();
        h.driver.close();
        gpa.destroy(h);
        return null;
    }
    const context: usize = @intCast(window);
    // what the streams and the prompt cache share: the room, less the scratch and the drafter's rings
    const fixed: u64 = backend.scratchBytes(context) + if (draft_kind != .none) streams * draft_kind.streamBytes(context) + draft_kind.scoreBytes(context) else 0;
    const budget: u64 = room - @min(room, fixed);
    std.log.info("inkling2_loop: {d:.1} GiB free for stream and prompt state, context up to {d} tokens, {d} streams", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), window, streams });
    h.lane = try backend.Xpu.init(gpa, h.model, .{ .capacity = context, .streams = @intCast(streams), .budget = budget });
    errdefer h.lane.deinit();
    try h.lane.warm();
    h.drafted = null;
    if (h.drafter) |d| {
        try d.setContext(@intCast(window));
        h.drafted = try lanes.drafted.Drafted.init(gpa, h.lane.backend(), d.drafter());
        h.drafted.?.first_rows = true;
    } else if (h.flash) |d| {
        try d.setContext(@intCast(window));
        h.drafted = try lanes.drafted.Drafted.init(gpa, h.lane.backend(), d.drafter());
        h.drafted.?.first_rows = true;
    }
    errdefer if (h.drafted) |*d| d.deinit();
    const facts = if (h.drafted) |*d| d.facts(h.lane.facts()) else h.lane.facts();
    h.config = try lanes.Config.init(gpa, facts, st.window_rows, st.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.drafted) |*d| d.backend() else h.lane.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "inkling2-xpu", .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = st.prompt_rows, .decode_share = o.decode_share, .structures = true, .logprobs = true, .draft_sources = .{ .drafter = if (drafter_dir) |dir| .{ .kind = if (draft_kind == .dflash2) "DFlash2" else "DSpark", .name = o.drafter_name orelse dir } else null } });
    h.cache = null;
    const want: u64 = std.math.lossyCast(u64, (o.prompt_cache_gib orelse default_cache_gib) * (1 << 30));
    h.cache_cap = @min(want, budget);
    if (h.cache_cap > 0) {
        const X = backend.Xpu;
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = st.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.lane, .vtable = &.{ .bytes = X.snapBytes, .save = X.snapSave, .restore = X.snapRestore, .drop = X.snapDrop } }, rules, h.cache_cap);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.lane.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    h.host.memory = .{ .read = readMemory };
    try h.host.start();
    std.log.info("inkling2-xpu: {s}, {d:.2} GB on the device, {d} streams of {d} tokens{s}", .{ h.ctx.name(), @as(f64, @floatFromInt(xpu.rt.alloc_now)) / 1e9, streams, window, if (draft_kind == .none) "" else if (draft_kind == .dflash2) ", DFlash2" else ", DSpark" });
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

test "Inkling XPU admission keeps DSpark opt-in and refuses learned modes" {
    var o: api.Open = .{ .dir = "checkpoint", .model_type = "inkling2_loop" };
    try std.testing.expect(refusal(o) == null);
    try std.testing.expect(drafterDirectory(o) == null);
    o.drafter = "checkpoint";
    try std.testing.expectEqualStrings("checkpoint", drafterDirectory(o).?);
    o.drafts = false;
    try std.testing.expect(drafterDirectory(o) == null);
    o.speed_up = "distributed";
    try std.testing.expect(refusal(o) != null);
    o.speed_up = null;
    o.learn = "learned";
    try std.testing.expect(refusal(o) != null);
    o.learn = null;
    o.tp = 2;
    try std.testing.expect(refusal(o) != null);
    try std.testing.expect(contextProblem(0, 10) != null);
    try std.testing.expect(contextProblem(-1, 10) != null);
    try std.testing.expect(contextProblem(11, 10) != null);
    try std.testing.expect(contextProblem(10, 10) == null);
}

test "Inkling XPU longest context accounts the scratch, the streams and the drafter's rings" {
    const gib: u64 = 1 << 30;
    const fits = longest(8 * gib, 1048576, 4, .dspark);
    try std.testing.expect(fits > 1024);
    try std.testing.expect(contextBytes(fits, 4, .dspark) <= 8 * gib);
    try std.testing.expect(contextBytes(fits + 1, 4, .dspark) > 8 * gib or fits == 1048576);
    try std.testing.expect(longest(8 * gib, 1048576, 4, .none) > fits);
    try std.testing.expectEqual(@as(usize, 0), longest(1 << 20, 1048576, 1, .none));
}

/// Rank 1 loads its half and replays rank 0's forwards. It serves no HTTP.
/// `--device` wins. The serve path otherwise ignored XPU_DEVICE, so rank 1 opened the same card as rank 0.
fn deviceOrdinal(named: ?u32) ?u32 {
    if (named) |n| return n;
    const z = std.c.getenv("XPU_DEVICE") orelse return null;
    return std.fmt.parseInt(u32, std.mem.span(z), 10) catch null;
}

pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !bool {
    _ = io;
    const place = ink.place.Place.parse(o.tp, o.rank) catch {
        problem.* = "inkling2_loop: --tp is 1 or 2, and --rank is below --tp";
        return false;
    };
    if (!place.split() or place.rank == 0) {
        problem.* = "inkling2_loop: rank 1 is the follower";
        return false;
    }
    var driver = xpu.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "cannot open the Level Zero driver ({s})", .{@errorName(e)});
        return false;
    };
    defer driver.close();
    var ctx = xpu.Context.init(&driver, deviceOrdinal(o.device)) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no Intel GPU device ({s})", .{@errorName(e)});
        return false;
    };
    defer ctx.deinit();
    var stream = xpu.Stream.init(&ctx) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine cannot start: {s}", .{@errorName(e)});
        return false;
    };
    defer stream.deinit();
    var rt = xpu.rt.Runtime.init(&ctx, stream) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine cannot start: {s}", .{@errorName(e)});
        return false;
    };
    defer rt.deinit();
    if (std.c.getenv("TF_TP_NOMODEL") != null) {
        const gp = ink.group.Group.open(gpa, &rt, &ctx, place) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "inkling2_loop: the TP group did not open ({s})", .{@errorName(e)});
            return false;
        };
        gpa.destroy(gp);
        std.process.exit(0);
    }
    var model = Model.load(gpa, &rt, o.dir, .{ .place = place }) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(e)});
        return false;
    };
    defer model.deinit();
    model.loadKernels(&ctx) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot build its kernels ({s})", .{@errorName(e)});
        return false;
    };
    var gp = ink.group.Group.open(gpa, &rt, &ctx, place) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "inkling2_loop: the TP group did not open ({s})", .{@errorName(e)});
        return false;
    };
    defer gpa.destroy(gp);
    const cap: usize = @intCast(o.context orelse 8192);
    var scratch = st.Scratch.init(&rt, st.prompt_rows, cap) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "inkling2_loop: scratch ({s})", .{@errorName(e)});
        return false;
    };
    defer scratch.deinit();
    var cache = st.Cache.init(&rt, cap, &model.weight_plan, .bf16) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "inkling2_loop: cache ({s})", .{@errorName(e)});
        return false;
    };
    defer cache.deinit();
    std.log.info("inkling2_loop: TP rank 1 following", .{});
    while (true) {
        const rec = gp.wait() catch |e| {
            std.log.err("inkling2_loop: rank 1 stopped ({s})", .{@errorName(e)});
            return false;
        };
        if (rec.op == .close) return true;
        std.log.info("inkling2: follow base {d} n {d}", .{ rec.base, rec.n });
        cache.len = rec.base;
        const profile_one = std.c.getenv("TF_TP_PROFILE") != null and xpu.profile.active == null;
        var prof: xpu.profile.Profile = undefined;
        if (profile_one) {
            prof = xpu.profile.Profile.init(&ctx, 131072, true) catch |e| {
                std.log.err("inkling2_loop: profile ({s})", .{@errorName(e)});
                return false;
            };
            xpu.profile.active = &prof;
            xpu.rt.Runtime.resetSyncLog();
        }
        // Bit 2 marks a prompt chunk. A verify window is a forward, even when it has more than one row.
        if (rec.flags & 4 != 0) {
            ink.prefill.run(model, &scratch, &cache, rec.ids[0..rec.n], .{ .defer_head = rec.flags & 2 != 0, .all = rec.flags & 1 != 0 }) catch |e| {
                std.log.err("inkling2_loop: rank 1 prefill ({s})", .{@errorName(e)});
                return false;
            };
            if (rec.flags & 2 != 0) ink.prefill.headRows(model, &scratch, 0, rec.n) catch |e| {
                std.log.err("inkling2_loop: rank 1 head ({s})", .{@errorName(e)});
                return false;
            };
        } else ink.forward.run(model, &scratch, &cache, rec.ids[0..rec.n], .{ .all = rec.flags != 0 }) catch |e| {
            std.log.err("inkling2_loop: rank 1 forward ({s})", .{@errorName(e)});
            return false;
        };
        if (profile_one) {
            prof.collect() catch {};
            std.log.info("inkling2: rank 1 one token, {d} launches", .{prof.launches});
            prof.print();
            xpu.rt.Runtime.logSync("rank 1");
            xpu.profile.active = null;
            prof.deinit();
        }
    }
}

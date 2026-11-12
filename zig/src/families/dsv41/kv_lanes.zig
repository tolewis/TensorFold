//! TF_DS_KV_SHARD round 2: parts.zig's ownership planner driven from the lanes (port of wojo's GLM 0118 lane wiring,
//! PR #152 at fork main 8641c90: multi.py's placement calls, the ordered op frames, carry/move finish and the borrowed
//! twins' lifetimes). Rank 0 (`Kv`) plans every placement, applies the plan, sends its ops to the followers as one
//! rank_wire shard_ops transaction (lanes Kind.rank_wire), then runs the transfers the ops name in their order (both
//! ranks call the part-aware Model.copyExtent, shard-attn's transport), asks every follower whether its side completed
//! (shard_xfer) and posts the verdict (shard_finish: its own side and every follower's); every rank finishes with rank
//! 0's ok. A follower (`Follower`) applies the same ops to its own planner (applyOps), runs the same transfers in the
//! same order, answers shard_xfer with its own side's result and finishes with rank 0's ok, never its own (without the
//! ask, a follower whose own transfer failed under rank 0's ok ends the pair). The planners' digests are compared after
//! the parts are set and every `check_rounds` decode rounds (rank_wire shard_check).
//!
//! The lanes keep their bookkeeping (Lane, Snap); this file owns the planner and calls back into the lanes through
//! `Hooks` (a kept state staled, a stream moved) and into the device through `Dev` (the transfer, the twin view), so
//! the whole flow runs on the host in tests with a recorder for the device and an in-process follower.
//!
//! Twins (stage 6): a fresh prompt's planner twins are one extent in every other part (all or none). While its prompt
//! fills, both ranks hold its compressed rows (the owner in its extent, the other rank in its twin), so both attend
//! locally with no fetch (Dev.twin binds the fill's view: prompt.Caches.twinned). The binding follows the planner on
//! both ranks at the same frames (a fill's begin and every applied op list), so it is all or none and the ranks agree;
//! the planner pins the twins while a chunk runs (setPartial) and the first decode round gives them back (planDecode).
const std = @import("std");
const lanes = @import("lanes");
const parts = @import("parts.zig");
const rank_wire = @import("rank_wire.zig");
const fingerprint = @import("fingerprint.zig");
const link = @import("link.zig");
const Allocator = std.mem.Allocator;

pub const Op = parts.Op;
pub const WireOp = rank_wire.ShardOp;

comptime {
    // the wire's shard-op records are parts.Op field for field (rank_wire.ShardOp is its copy until rank-wire imports
    // parts.Op itself): a drift fails the build instead of a frame
    if (rank_wire.schemaOf(parts.Op) != rank_wire.shard_schema) @compileError("kv_lanes: rank_wire.ShardOp differs from parts.Op");
}

/// A planner op as the wire carries it (rank_wire.ShardOp is parts.Op now: the identity, kept so a copy of the type
/// would still convert field for field).
pub fn toWire(op: Op) WireOp {
    return conv(WireOp, op);
}

/// A wire op as the planner applies it.
pub fn fromWire(op: WireOp) Op {
    return conv(Op, op);
}

/// The same value in a type with the same fields and tags (enums by value).
fn conv(comptime To: type, v: anytype) To {
    const From = @TypeOf(v);
    if (From == To) return v;
    switch (@typeInfo(To)) {
        .@"union" => switch (v) {
            inline else => |p, tag| return @unionInit(To, @tagName(tag), conv(@FieldType(To, @tagName(tag)), p)),
        },
        .@"struct" => |s| {
            var out: To = undefined;
            inline for (s.field_names) |n| @field(out, n) = conv(@FieldType(To, n), @field(v, n));
            return out;
        },
        .@"enum" => return @enumFromInt(@intFromEnum(v)),
        else => return v,
    }
}

/// Rows a transfer of `rows` valid rows copies: whole blocks (Model.copyExtent needs a multiple of every ratio; a live
/// stream's partly filled last block travels whole), never past the reservation (`size`, chunk-aligned).
pub fn transferRows(rows: usize, size: usize) usize {
    return @min(std.mem.alignForward(usize, rows, parts.chunk_rows), size);
}

/// What the device side does for the planner's ops (the Model on a GPU; a recorder in tests).
pub const Dev = struct {
    ctx: *anyopaque,
    /// copy's refusals with nothing enqueued (Model.checkExtent): the same on every rank for the same arguments under
    /// the agreed layout, so a refused transfer is refused alike everywhere and no rank's NCCL half waits for another.
    check: *const fn (ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void,
    /// The compressed rows [0, rows) of the extent at `from` into the extent at `to` (Model.copyExtent: within a part
    /// the owner copies, across parts the source's owner sends and the destination's receives), in the frame order.
    copy: *const fn (ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void,
    /// The filling prompt's view at `base` with its twins (`on`: both ranks hold its rows and attend locally; `local`:
    /// this rank's twin base when it is not the owner), or plain again (Model.bindTwin).
    twin: *const fn (ctx: *anyopaque, base: usize, on: bool, local: ?usize) anyerror!void,
    /// Every transfer enqueued so far done (an event after them polled: its NCCL halves completed) within
    /// `timeout_ns`, else error.TransferTimeout (never an unbounded stream synchronize: a peer half that was never
    /// enqueued would hold it forever).
    wait: *const fn (ctx: *anyopaque, timeout_ns: u64) anyerror!void,
};

/// A follower's bound on its own side of a transfer when shard_xfer asks (below rank 0's deadline for the reply,
/// rank_wire.Timeouts.xfer_ns, by a margin): past it the follower ends instead of answering late.
pub const follower_wait_ns: u64 = 20 * std.time.ns_per_s;

/// Whether an op's transfer crosses parts: an NCCL send on one rank and its receive on another, so one rank's side
/// that never runs leaves the other's waiting (a copy within one part has no peer half).
fn crosses(op: Op) bool {
    return switch (op) {
        .carry => |x| x.src_part != x.dst_part,
        .move => |x| x.src_part != x.dst_part,
        else => false,
    };
}

/// Rank 0's frames to the followers (lanes.zig's ordered command stream).
pub const Out = struct {
    ctx: *anyopaque,
    /// A rank_wire frame's bytes (lanes Kind.rank_wire).
    rw: *const fn (ctx: *anyopaque, sub: []const u8) anyerror!void,
};

/// What rank 0's lanes do for the ops (their Snaps and Lanes follow the planner).
pub const Hooks = struct {
    ctx: *anyopaque,
    /// A kept state given up (Snap.stale).
    stale: *const fn (ctx: *anyopaque, kept: u64) void,
    /// A live stream moved with its kept states: every Snap's and Lane's rows again from the planner.
    moved: *const fn (ctx: *anyopaque, stream: u64) void,
};

/// The switches beyond TF_DS_KV_SHARD (rank 0 plans with them; every rank reads them for the agreement).
pub const Settings = struct {
    carry: bool = true, // TF_DS_KV_SHARD_CARRY: carries and live moves across parts (GLM TF_GLM_KV_SHARD_CARRY)
    twins: bool = true, // TF_DS_KV_SHARD_TWINS: borrowed twins for filling prompts
    check_rounds: u64 = 1024, // TF_DS_KV_SHARD_CHECK: decode rounds between digest checks (0: at start only)

    pub fn fromEnv(get: *const fn ([:0]const u8) ?[]const u8) !Settings {
        var s: Settings = .{};
        if (get("TF_DS_KV_SHARD_CARRY")) |v| s.carry = try flag(v);
        if (get("TF_DS_KV_SHARD_TWINS")) |v| s.twins = try flag(v);
        if (get("TF_DS_KV_SHARD_CHECK")) |v| if (v.len > 0) {
            s.check_rounds = std.fmt.parseInt(u64, v, 10) catch return error.BadKvShardSetting;
        };
        return s;
    }

    fn flag(v: []const u8) !bool {
        if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
        if (std.mem.eql(u8, v, "0")) return false;
        return error.BadKvShardSetting;
    }
};

/// The startup agreement when nothing else ran one (the spill tier's full fingerprint, rank_identity.zig, replaces it
/// when both are on: Lanes.rw is one Leader): every field the shard path depends on (the wire, the block, the owner
/// layout and parts, the split, the switches, the rank), the others marked as not covered.
pub fn shardExchange(rank: u32, world: u32, cfg: parts.Config, s: Settings) fingerprint.Exchange {
    var b: fingerprint.Builder = .{};
    for (fingerprint.all) |f| switch (f) {
        .wire_version => b.part(f, &rank_wire.stampBytes()),
        .chunk_rows => b.int(f, u32, parts.chunk_rows),
        .owner_layout => {
            b.part(f, "owner_sharded");
            b.int(f, u16, parts.layout_version);
            b.int(f, u8, cfg.parts);
            b.int(f, u64, cfg.cap);
        },
        .tp_split => b.int(f, u32, world),
        .features => {
            var buf: [64]u8 = undefined;
            b.part(f, std.fmt.bufPrint(&buf, "kv_shard=1;kv_carry={d};twins={d};", .{ @intFromBool(s.carry), @intFromBool(s.twins) }) catch unreachable);
        },
        .rank_split => b.int(f, u32, rank),
        else => b.part(f, "shard-only"),
    };
    const fp = b.finish(null) catch unreachable; // (every field given)
    return fp.exchange();
}

const Fill = struct { stream: u64, base: usize };
const Bound = struct { on: bool, local: ?usize };

/// The twin binding a planner asks for the filling stream on rank `rank`: none, or on with this rank's twin (null on
/// the owner, whose part holds the extent).
fn wantTwin(p: *const parts.Planner, f: Fill, rank: u32) Bound {
    if (!p.hasTwins(f.stream)) return .{ .on = false, .local = null };
    var local: ?usize = null;
    for (p.twins.items) |t| if (t.stream == f.stream and t.part == rank) {
        local = t.base;
    };
    return .{ .on = true, .local = local };
}

fn sameBound(a: ?Bound, b: Bound) bool {
    const x = a orelse return !b.on; // (a fill begins plain)
    return x.on == b.on and x.local == b.local;
}

/// Lanes-side counts (the planner's own are parts.Counters).
pub const Counts = struct {
    plans: u64 = 0, // op lists sent
    ops: u64 = 0,
    transfers: u64 = 0,
    relocations: u64 = 0,
    checks: u64 = 0,
    deferred_drops: u64 = 0,
};

// -- rank 0 ---------------------------------------------------------------------------------------------------------

pub const Admitted = struct {
    id: u64, // the planner's stream (Lane.id)
    part: u8,
    base: usize,
    gen: u64,
    resumed: bool, // the source's rows are the stream's (in place, or copied/carried and finished ok)
    dropped: bool, // the hit was given up (the prompt runs from 0)
    mode: parts.Mode,
};

pub const Kv = struct {
    gpa: Allocator,
    planner: parts.Planner,
    settings: Settings,
    dev: Dev,
    hooks: Hooks,
    out: ?Out = null, // null: no followers
    leader: ?*rank_wire.Leader = null, // the followers' protocol (Lanes.rw when the spill side made one)
    own_leader: bool = false,
    links: []const ?link.Link = &.{}, // the followers' links for asks (Model.followers())
    started: bool = false,
    next_id: u64 = 1,
    waiting: std.ArrayList(Wait) = .empty, // requests not yet placed, by their lanes.Stream (refusals once a request)
    drops: std.ArrayList(u64) = .empty, // kept states whose drop waits for a transfer reading them (error.InFlight)
    rounds: u64 = 0,
    filling: ?Fill = null,
    bound: ?Bound = null,
    counts: Counts = .{},
    wire_ops: [rank_wire.max_shard_ops]WireOp = undefined,

    const Wait = struct { key: usize, tag: u64, id: u64 };
    /// Waiting requests remembered at most (a request the host gave up while it waited is never told to the backend:
    /// the oldest record is forgotten past this).
    pub const max_waiting = 64;
    /// Live moves one admission makes at most before it waits (error.ContextFull).
    pub const max_relocations = 2;

    pub fn init(gpa: Allocator, cfg: parts.Config, settings: Settings, dev: Dev, hooks: Hooks) !Kv {
        var c = cfg;
        c.carry = settings.carry;
        c.twins = settings.twins;
        return .{ .gpa = gpa, .planner = try parts.Planner.init(c), .settings = settings, .dev = dev, .hooks = hooks };
    }

    pub fn deinit(kv: *Kv) void {
        kv.planner.deinit(kv.gpa);
        kv.waiting.deinit(kv.gpa);
        kv.drops.deinit(kv.gpa);
        if (kv.own_leader) if (kv.leader) |l| kv.gpa.destroy(l);
    }

    /// The followers: frames through `out`, asks over `links`, under `leader` (null: a leader of its own at `epoch`
    /// whose hello carries `fx`).
    pub fn lead(kv: *Kv, out: Out, links: []const ?link.Link, leader: ?*rank_wire.Leader, epoch: u64, fx: fingerprint.Exchange) !void {
        kv.out = out;
        kv.links = links;
        if (leader) |l| {
            kv.leader = l;
            return;
        }
        const l = try kv.gpa.create(rank_wire.Leader);
        l.* = rank_wire.Leader.init(@intCast(links.len + 1), epoch, fx);
        kv.leader = l;
        kv.own_leader = true;
    }

    fn sink(kv: *Kv) rank_wire.Sink {
        return .{ .ctx = kv, .send = sendRw };
    }

    fn sendRw(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const kv: *Kv = @ptrCast(@alignCast(ctx));
        const o = kv.out orelse return error.NoFollowers;
        try o.rw(o.ctx, sub);
    }

    fn asks(kv: *const Kv) bool {
        return kv.links.len > 0;
    }

    /// Rank 0, once before the first plan: the startup agreement (unless one ran: the spill side's full fingerprint),
    /// the followers' layouts (shard_layout), the parts (shard_set) and the first digest check. Without links (tests)
    /// only shard_set goes out.
    pub fn start(kv: *Kv) !void {
        if (kv.started) return;
        const l = kv.leader orelse {
            kv.started = true;
            return;
        };
        const cfg = kv.planner.cfg;
        if (kv.asks()) {
            if (l.agreement == null) _ = try l.agree(kv.sink(), kv.links);
            var replies: [rank_wire.max_world - 1]rank_wire.Frame = undefined;
            _ = try l.ask(kv.sink(), kv.links, 0, .{ .shard_layout = .{ .version = parts.layout_version, .parts = cfg.parts, .cap = cfg.cap, .carry = cfg.carry, .schema = rank_wire.shard_schema } }, replies[0..kv.links.len]);
            for (replies[0..kv.links.len]) |f| {
                const r = f.msg.shard_layout_reply;
                if (r.version != parts.layout_version or r.parts != cfg.parts or r.cap != cfg.cap or r.schema != rank_wire.shard_schema) return error.ShardLayoutMismatch;
            }
        }
        _ = try l.post(kv.sink(), 0, .{ .shard_set = .{ .version = parts.layout_version, .parts = cfg.parts, .part_rows = kv.planner.part_rows, .owner_epoch = l.t.owner_epoch + 1 } });
        kv.started = true;
        try kv.check();
    }

    /// The planner's digest against every follower's (rank_wire shard_check); a difference ends the pair.
    pub fn check(kv: *Kv) !void {
        const l = kv.leader orelse return;
        if (!kv.asks()) return;
        const d = kv.planner.digest();
        var replies: [rank_wire.max_world - 1]rank_wire.Frame = undefined;
        _ = try l.ask(kv.sink(), kv.links, 0, .{ .shard_check = .{ .owner_epoch = l.t.owner_epoch, .digest = d } }, replies[0..kv.links.len]);
        for (replies[0..kv.links.len]) |f| if (!std.mem.eql(u8, &f.msg.shard_check_reply.digest, &d)) return error.ShardDiverged;
        kv.counts.checks += 1;
    }

    /// A plan as the followers heard it: its transaction and owner epoch (its transfers' shard_xfer and shard_finish
    /// name both).
    const Sent = struct { txn: u64, epoch: u64 };

    /// The ops to every follower as one plan (one owner epoch; frames of max_shard_ops each).
    fn sendOps(kv: *Kv, ops: []const Op) !?Sent {
        if (ops.len == 0) return null;
        const l = kv.leader orelse return null;
        const epoch = l.t.owner_epoch + 1;
        var i: usize = 0;
        var index: u16 = 0;
        var txn: u64 = 0;
        while (i < ops.len) {
            const n = @min(rank_wire.max_shard_ops, ops.len - i);
            for (ops[i..][0..n], kv.wire_ops[0..n]) |o, *w| w.* = toWire(o);
            const head = try l.post(kv.sink(), txn, .{ .shard_ops = .{ .owner_epoch = epoch, .index = index, .last = i + n == ops.len, .ops = kv.wire_ops[0..n] } });
            if (index == 0) txn = head.txn;
            i += n;
            index += 1;
        }
        kv.counts.plans += 1;
        kv.counts.ops += ops.len;
        return .{ .txn = txn, .epoch = epoch };
    }

    /// A plan applied (checked, all or nothing) and then sent: a refused plan sends nothing.
    fn applyPlan(kv: *Kv, plan: *const parts.Plan) !?Sent {
        try kv.planner.apply(kv.gpa, plan);
        return kv.sendOps(plan.ops.items);
    }

    /// A `full` plan applied: no ops, only its refusals counted (once a request). Ops here would change rank 0's
    /// planner and no follower's (a full plan is never sent): refused before anything changes.
    fn applyFull(kv: *Kv, plan: *const parts.Plan) !void {
        if (plan.ops.items.len != 0) return error.BadPlan;
        try kv.planner.apply(kv.gpa, plan);
    }

    /// One op outside a plan (keep, drop), applied and sent.
    fn applyOne(kv: *Kv, op: Op) !void {
        try kv.planner.applyOps(kv.gpa, &.{op});
        _ = try kv.sendOps(&.{op});
    }

    /// The ops' effects past the planner, in plan order: stales into the Snaps, the transfers (each finished with
    /// rank 0's ok over every rank), a moved stream's rows back into the lanes, the fill's twin view.
    fn runOps(kv: *Kv, ops: []const Op, sent: ?Sent) !bool {
        var resumed = false;
        for (ops) |op| switch (op) {
            .stage => {}, // PORT(round3): the spill tier stages the point's pending rows (spill with TF_DS_KV_SHARD is refused until then)
            .stale => |x| kv.hooks.stale(kv.hooks.ctx, x.kept),
            .copy => |x| resumed = try kv.transfer(sent, x.stream, x.src, x.dst, x.rows, crosses(op)),
            .carry => |x| resumed = try kv.transfer(sent, x.stream, x.src, x.dst, x.rows, crosses(op)),
            .move => |x| if (try kv.transfer(sent, x.stream, x.src, x.dst, transferRows(x.rows, x.size), crosses(op))) kv.hooks.moved(kv.hooks.ctx, x.stream),
            else => {},
        };
        try kv.syncTwin();
        kv.retryDrops();
        return resumed;
    }

    /// Both ranks' transfer (each enqueues its own side in this order, the followers at the plan's frame): rank 0's
    /// side completed, every follower's asked (shard_xfer), the verdict posted (shard_finish) and every rank's finish
    /// with it. Without links (tests) rank 0's own side is the verdict.
    ///
    /// Failures: a refusal (Dev.check: deterministic, nothing enqueued) is every rank's alike, and an enqueue failure
    /// of a copy within one part leaves no peer waiting: both finish failed (the prompt runs from 0) and the pair goes
    /// on. An enqueue failure across parts (`cross`: the peer's NCCL half would wait for this one forever), or a side
    /// that does not complete within its bound (a peer that never enqueued its half or ended, a device fault), ends
    /// the pair: no wait is unbounded, but the device stream holds an unmatched NCCL call, so the engine restarts.
    fn transfer(kv: *Kv, sent: ?Sent, stream: u64, from: usize, to: usize, rows: usize, cross: bool) !bool {
        var ok = true;
        if (kv.dev.check(kv.dev.ctx, from, to, rows)) |_| {
            if (kv.dev.copy(kv.dev.ctx, from, to, rows)) |_| {
                kv.dev.wait(kv.dev.ctx, kv.waitNs()) catch |err| {
                    std.log.warn("dsv41: a KV transfer of {d} rows from {d} to {d} did not complete: {s}; the ranks end", .{ rows, from, to, @errorName(err) });
                    return kv.endPair(err);
                };
            } else |err| {
                if (cross) {
                    std.log.warn("dsv41: a KV transfer of {d} rows from {d} to {d} across parts failed: {s}; the ranks end", .{ rows, from, to, @errorName(err) });
                    return kv.endPair(err);
                }
                std.log.warn("dsv41: a KV transfer of {d} rows from {d} to {d} failed: {s}", .{ rows, from, to, @errorName(err) });
                ok = false;
            }
        } else |err| {
            std.log.warn("dsv41: a KV transfer of {d} rows from {d} to {d} was refused: {s}", .{ rows, from, to, @errorName(err) });
            ok = false;
        }
        if (kv.leader) |l| if (sent) |p| {
            if (kv.asks()) {
                var replies: [rank_wire.max_world - 1]rank_wire.Frame = undefined;
                _ = try l.ask(kv.sink(), kv.links, p.txn, .{ .shard_xfer = .{ .owner_epoch = p.epoch, .stream = stream } }, replies[0..kv.links.len]);
                for (replies[0..kv.links.len], 1..) |f, r| if (!f.msg.shard_xfer_reply.ok) {
                    std.log.warn("dsv41: rank {d}'s side of a KV transfer of {d} rows failed", .{ r, rows });
                    ok = false;
                };
            }
            _ = try l.post(kv.sink(), p.txn, .{ .shard_finish = .{ .owner_epoch = p.epoch, .stream = stream, .ok = ok } });
        };
        const r = try kv.planner.finish(stream, ok);
        kv.counts.transfers += 1;
        return r.resumed;
    }

    /// Whether the followers' pair ended (nothing more reaches them).
    fn ended(kv: *const Kv) bool {
        const l = kv.leader orelse return false;
        return l.broken != null;
    }

    /// Rank 0's bound on its own side of a transfer: the followers' reply deadline (rank_wire.Timeouts.xfer_ns).
    fn waitNs(kv: *const Kv) u64 {
        return if (kv.leader) |l| l.timeouts.xfer_ns else (rank_wire.Timeouts{}).xfer_ns;
    }

    /// Rank 0 ends the pair for a cause of its own (as rank_wire's refused posts do: `broken` names rank 0 and the
    /// cause; PORT(rank-wire): Leader.fail is private, so its first-cause rule is repeated here). No followers: the
    /// cause itself.
    fn endPair(kv: *Kv, err: anyerror) anyerror {
        const l = kv.leader orelse return err;
        if (l.broken == null) l.broken = .{ .rank = 0, .err = err };
        return error.PairBroken;
    }

    /// The planner's stream for a request (`key`: its lanes.Stream, `tag`: its id's hash), the same while it waits.
    fn idFor(kv: *Kv, key: usize, tag: u64) !u64 {
        for (kv.waiting.items, 0..) |w, i| if (w.key == key) {
            if (w.tag == tag) return w.id;
            kv.planner.forget(w.id); // another request at the same address: the old one was given up
            _ = kv.waiting.orderedRemove(i);
            break;
        };
        if (kv.waiting.items.len >= max_waiting) {
            kv.planner.forget(kv.waiting.items[0].id);
            _ = kv.waiting.orderedRemove(0);
        }
        const id = kv.next_id;
        kv.next_id += 1;
        try kv.waiting.append(kv.gpa, .{ .key = key, .tag = tag, .id = id });
        return id;
    }

    /// A request the host gave up (cancelled while waiting, or placed elsewhere): its refusal record goes.
    pub fn forgetKey(kv: *Kv, key: usize) void {
        for (kv.waiting.items, 0..) |w, i| if (w.key == key) {
            kv.planner.forget(w.id);
            _ = kv.waiting.orderedRemove(i);
            return;
        };
    }

    fn placed(kv: *Kv, key: usize) void {
        for (kv.waiting.items, 0..) |w, i| if (w.key == key) {
            _ = kv.waiting.orderedRemove(i);
            return;
        };
    }

    /// lanes.prefillFn's placement (shard-parts.md's six steps): plan; `full` applied (its refusals counted once a
    /// request) and error.ContextFull; `relocate` applied, sent, its move run and finished, then planned again;
    /// `placed` applied, sent, its ops acted on in order. `source`: the kept state it resumes from (lanes checked it
    /// usable), `patient`: held(source) and lanes' wait clock still runs.
    pub fn admit(kv: *Kv, key: usize, tag: u64, slot: u32, size: usize, source: ?u64, patient: bool) !Admitted {
        try kv.start();
        const id = try kv.idFor(key, tag);
        return kv.place(key, id, slot, size, source, patient) catch |err| {
            if (err != error.ContextFull) kv.forgetKey(key); // (a waiting request keeps its record)
            return err;
        };
    }

    fn place(kv: *Kv, key: usize, id: u64, slot: u32, size: usize, source: ?u64, patient: bool) !Admitted {
        var relocations: usize = 0;
        while (true) {
            var plan = try kv.planner.planAdmit(kv.gpa, .{ .stream = id, .slot = slot, .size = size, .source = source, .patient = patient });
            defer plan.deinit(kv.gpa);
            switch (plan.status) {
                .full => {
                    try kv.applyFull(&plan);
                    return error.ContextFull;
                },
                .relocate => {
                    if (relocations == max_relocations) return error.ContextFull;
                    relocations += 1;
                    const sent = try kv.applyPlan(&plan);
                    _ = try kv.runOps(plan.ops.items, sent);
                    kv.counts.relocations += 1;
                },
                .placed => {
                    const sent = try kv.applyPlan(&plan);
                    // a later step failing (a twin bind, a finish) releases the extent on every rank (planRelease's
                    // ops sent): else its Live and its lanes slot stay taken on both ranks. A broken pair sends
                    // nothing more (and may leave the transfer in flight): nothing to release then.
                    errdefer if (!kv.ended()) kv.release(id) catch |err| std.log.warn("dsv41: a failed placement kept its part's extent: {s}", .{@errorName(err)});
                    kv.placed(key);
                    const ran = try kv.runOps(plan.ops.items, sent);
                    const l = kv.planner.lives.items[kv.planner.liveIndex(id) orelse return error.UnknownStream];
                    const resumed = plan.source != null and !plan.dropped and (plan.mode == .in_place or ran);
                    return .{ .id = id, .part = l.part, .base = l.base, .gen = l.gen, .resumed = resumed, .dropped = plan.dropped, .mode = plan.mode };
                },
            }
        }
    }

    /// A stream ends (released, or its prompt pass failed after placement): its twins, then its extent.
    pub fn release(kv: *Kv, id: u64) !void {
        if (kv.filling) |f| if (f.stream == id) {
            kv.filling = null;
            kv.bound = null;
        };
        var plan = kv.planner.planRelease(kv.gpa, id) catch |err| switch (err) {
            error.UnknownStream => return,
            else => return err,
        };
        defer plan.deinit(kv.gpa);
        _ = try kv.applyPlan(&plan);
        kv.retryDrops();
    }

    /// lanes.snapSaveFn: the live stream's state at `at` kept.
    pub fn keep(kv: *Kv, kept: u64, stream: u64, at: usize) !void {
        try kv.applyOne(try kv.planner.keepOp(kept, stream, at));
    }

    /// lanes.snapDropFn: a kept state dropped; deferred while a copy or carry reads it (retried after each finish).
    pub fn drop(kv: *Kv, kept: u64) !void {
        const op = kv.planner.dropOp(kept) catch |err| switch (err) {
            error.UnknownKept => return, // (its keep never reached the planner: nothing to drop)
            error.InFlight => {
                try kv.drops.append(kv.gpa, kept);
                kv.counts.deferred_drops += 1;
                return;
            },
        };
        try kv.applyOne(op);
    }

    fn retryDrops(kv: *Kv) void {
        var i: usize = 0;
        while (i < kv.drops.items.len) {
            const id = kv.drops.items[i];
            const op = kv.planner.dropOp(id) catch |err| switch (err) {
                error.InFlight => {
                    i += 1;
                    continue;
                },
                else => {
                    _ = kv.drops.orderedRemove(i);
                    continue;
                },
            };
            kv.applyOne(op) catch |err| std.log.err("dsv41: a deferred kept-state drop failed: {s}", .{@errorName(err)});
            _ = kv.drops.orderedRemove(i);
        }
    }

    /// A stream's first decode round: its twins given back first (decode reads the owners' rows only).
    pub fn decodeStart(kv: *Kv, id: u64) !void {
        var plan = try kv.planner.planDecode(kv.gpa, id);
        defer plan.deinit(kv.gpa);
        _ = try kv.applyPlan(&plan);
        kv.planner.setDecoding(id, true);
        try kv.syncTwin();
    }

    pub const Valid = struct { id: u64, valid: usize };

    /// Each verify round: the planner's clock, every window's cache length, and the periodic digest check.
    pub fn round(kv: *Kv, wins: []const Valid) !void {
        kv.planner.tick();
        for (wins) |w| kv.planner.setValid(w.id, w.valid);
        kv.rounds += 1;
        if (kv.settings.check_rounds > 0 and kv.rounds % kv.settings.check_rounds == 0) try kv.check();
    }

    /// A prompt pass of stream `id` begins at `base` (after Model.fillBegin): its twin view as the planner has it.
    /// The fill stays this rank's `filling` past its last chunk, until the next fill begins or the stream is
    /// released, exactly as on a follower (which hears no fill end): both ranks then bind and unbind at the same
    /// frames (the first decode round's untwin, a revoke between).
    pub fn fillBegin(kv: *Kv, id: u64, base: usize) !void {
        kv.filling = .{ .stream = id, .base = base };
        kv.bound = null;
        try kv.syncTwin();
    }

    /// The fill's view bound to its twins exactly while the planner holds them (both ranks at the same frames).
    fn syncTwin(kv: *Kv) !void {
        const f = kv.filling orelse return;
        const want = wantTwin(&kv.planner, f, 0);
        if (sameBound(kv.bound, want)) return;
        try kv.dev.twin(kv.dev.ctx, f.base, want.on, want.local);
        kv.bound = want;
    }

    /// The counters and every part's rows (health), the attention side's transfer stats added by the caller.
    pub fn stats(kv: *const Kv, out: *lanes.kv_parts.Stats) void {
        const c = kv.planner.counters;
        out.parts = kv.planner.cfg.parts;
        out.part_rows = kv.planner.part_rows;
        out.carries = c.hits;
        out.carry_rescued = c.rescued;
        out.carry_pressure = c.pressure;
        out.carry_others = c.others;
        out.carry_paused = c.paused;
        out.moves = c.live;
        out.moved_tokens = c.tokens;
        out.moved_bytes = c.bytes;
        out.transfers_failed = c.failed;
        out.staged = c.staged;
        out.staled = c.staled;
        out.twins_made = c.twins_made;
        out.twins_revoked = c.twins_revoked;
        out.refused_pinned = c.refusals(.pinned);
        out.refused_twin = c.refusals(.twin);
        out.refused_loading = c.refusals(.loading);
        out.refused_partial = c.refusals(.partial);
        out.refused_no_target = c.refusals(.no_target);
        out.refused_small = c.refusals(.small);
        out.no_part = c.no_part;
        out.plans = kv.counts.plans;
        out.checks = kv.counts.checks;
        for (0..@min(kv.planner.cfg.parts, lanes.kv_parts.max_parts)) |p| {
            // (a failed allocation reads as an empty part; health is best effort)
            const s = kv.planner.partStats(kv.gpa, @intCast(p)) catch continue;
            out.part[p] = .{ .free = s.free, .live_rows = s.live_rows, .kept_rows = s.kept_rows, .twin_rows = s.twin_rows, .hold_rows = s.hold_rows, .live = s.live };
        }
    }
};

// -- followers -------------------------------------------------------------------------------------------------------

pub const Follower = struct {
    gpa: Allocator,
    rank: u32,
    cfg: parts.Config,
    dev: Dev,
    rw: rank_wire.Follower, // (the spill side's handlers join this one when both are on: one Follower a rank)
    planner: ?parts.Planner = null, // from shard_set
    filling: ?Fill = null,
    bound: ?Bound = null,
    failed: std.ArrayList(u64) = .empty, // streams whose own transfer failed, until rank 0's finish
    diverged: bool = false,
    ops: [rank_wire.max_shard_ops]Op = undefined,

    pub fn init(gpa: Allocator, rank: u32, cfg: parts.Config, settings: Settings, dev: Dev) !*Follower {
        // (the switches as rank 0's Kv.init sets them: both planners' configs alike, the features fingerprint agreed)
        var c = cfg;
        c.carry = settings.carry;
        c.twins = settings.twins;
        try c.check();
        const f = try gpa.create(Follower);
        f.* = .{ .gpa = gpa, .rank = rank, .cfg = c, .dev = dev, .rw = rank_wire.Follower.init(@intCast(rank), shardExchange(rank, c.parts, c, settings)) };
        return f;
    }

    pub fn deinit(f: *Follower) void {
        if (f.planner) |*p| p.deinit(f.gpa);
        f.failed.deinit(f.gpa);
        f.gpa.destroy(f);
    }

    /// A lanes Kind.rank_wire frame's payload: decoded, admitted in order, handled; an ask answered on `l`.
    pub fn onFrame(f: *Follower, l: link.Link, payload: []const u8) !void {
        try rank_wire.serve(&f.rw, l, payload, f);
        if (f.diverged) return error.ShardDiverged;
    }

    // rank_wire.serve's handler methods

    pub fn shardLayout(f: *Follower, _: rank_wire.Head, _: rank_wire.ShardLayout) !rank_wire.ShardLayoutReply {
        return .{ .version = parts.layout_version, .parts = f.cfg.parts, .cap = f.cfg.cap, .schema = rank_wire.shard_schema };
    }

    pub fn shardSet(f: *Follower, _: rank_wire.Head, m: rank_wire.ShardSet) !void {
        if (m.version != parts.layout_version or m.parts != f.cfg.parts or m.part_rows != f.cfg.partRows()) return error.ShardLayoutMismatch;
        if (f.planner == null) f.planner = try parts.Planner.init(f.cfg);
    }

    pub fn shardOps(f: *Follower, _: rank_wire.Head, m: rank_wire.ShardOps) !void {
        const p = &(f.planner orelse return error.NotAgreed);
        if (m.ops.len > f.ops.len) return error.BadCount;
        for (m.ops, f.ops[0..m.ops.len]) |w, *o| o.* = fromWire(w);
        const ops = f.ops[0..m.ops.len];
        try p.applyOps(f.gpa, ops);
        // the same transfers as rank 0, in its order (a pair of NCCL calls across parts): enqueued now, before any ask
        // (rank_wire's deadlock rule 3); shard_xfer asks how this rank's side ended, shard_finish brings rank 0's ok
        for (ops) |op| switch (op) {
            .copy => |x| try f.transfer(x.stream, x.src, x.dst, x.rows, crosses(op)),
            .carry => |x| try f.transfer(x.stream, x.src, x.dst, x.rows, crosses(op)),
            .move => |x| try f.transfer(x.stream, x.src, x.dst, transferRows(x.rows, x.size), crosses(op)),
            else => {},
        };
        try f.syncTwin();
    }

    pub fn shardCheck(f: *Follower, _: rank_wire.Head, m: rank_wire.ShardCheck) !rank_wire.ShardCheckReply {
        const p = &(f.planner orelse return error.NotAgreed);
        const d = p.digest();
        if (!std.mem.eql(u8, &d, &m.digest)) {
            std.log.err("dsv41: rank {d}'s KV-part planner differs from rank 0's at owner epoch {d}", .{ f.rank, m.owner_epoch });
            f.diverged = true; // (answered first: rank 0 sees the difference and ends too)
        }
        return .{ .owner_epoch = m.owner_epoch, .digest = d };
    }

    /// This rank's side, as Kv.transfer's: a refusal (rank 0 refuses alike) or a failed copy within one part is
    /// recorded for shard_xfer; a failed enqueue across parts ends this rank (rank 0's half would wait for it; its own
    /// wait is bounded, so rank 0 ends too).
    fn transfer(f: *Follower, stream: u64, from: usize, to: usize, rows: usize, cross: bool) !void {
        f.dev.check(f.dev.ctx, from, to, rows) catch |err| {
            std.log.warn("dsv41: rank {d}'s KV transfer of {d} rows from {d} to {d} was refused: {s}", .{ f.rank, rows, from, to, @errorName(err) });
            try f.failed.append(f.gpa, stream);
            return;
        };
        f.dev.copy(f.dev.ctx, from, to, rows) catch |err| {
            if (cross) {
                std.log.warn("dsv41: rank {d}'s KV transfer of {d} rows from {d} to {d} across parts failed: {s}; this rank ends", .{ f.rank, rows, from, to, @errorName(err) });
                return error.TransferFailed;
            }
            std.log.warn("dsv41: rank {d}'s KV transfer of {d} rows from {d} to {d} failed: {s}", .{ f.rank, rows, from, to, @errorName(err) });
            try f.failed.append(f.gpa, stream);
        };
    }

    fn mine(f: *const Follower, stream: u64) bool {
        return std.mem.indexOfScalar(u64, f.failed.items, stream) == null;
    }

    /// rank_wire shard_xfer: whether this rank's side of the stream's transfer completed (enqueued without error, and
    /// done within follower_wait_ns: its NCCL half needs only rank 0's half, enqueued before the ask). A side that does
    /// not complete in time ends this rank instead of answering (the stream holds an unmatched call; rank 0 sees the
    /// link go or its deadline pass and ends too).
    pub fn shardXfer(f: *Follower, _: rank_wire.Head, m: rank_wire.ShardXfer) !rank_wire.ShardXferReply {
        const ok = f.mine(m.stream);
        if (ok) f.dev.wait(f.dev.ctx, follower_wait_ns) catch |err| {
            std.log.warn("dsv41: rank {d}'s side of a KV transfer did not complete: {s}; this rank ends", .{ f.rank, @errorName(err) });
            return err;
        };
        return .{ .owner_epoch = m.owner_epoch, .stream = m.stream, .ok = ok };
    }

    /// rank_wire shard_finish: rank 0's ok, never this rank's own result.
    pub fn shardFinish(f: *Follower, _: rank_wire.Head, m: rank_wire.ShardFinish) !void {
        try f.finish(m.stream, m.ok);
    }

    fn finish(f: *Follower, stream: u64, ok: bool) !void {
        const p = &(f.planner orelse return error.NotAgreed);
        const own = f.mine(stream);
        if (std.mem.indexOfScalar(u64, f.failed.items, stream)) |i| _ = f.failed.orderedRemove(i);
        // rank 0 says the rows are good where this rank's are not (no shard_xfer told it): the pair ends
        if (ok and !own) return error.TransferFailed;
        _ = try p.finish(stream, ok);
        try f.syncTwin();
    }

    /// The live stream in pool slot `slot`.
    fn liveAt(f: *const Follower, slot: usize) ?parts.Live {
        const p = &(f.planner orelse return null);
        for (p.lives.items) |l| if (l.slot == slot) return l;
        return null;
    }

    /// A fill_begin (after Model.fillBegin): the twin view as this rank's planner has it.
    pub fn fillBegin(f: *Follower, slot: usize, base: usize) !void {
        const l = f.liveAt(slot) orelse return error.UnknownStream; // rank 0 placed it before the fill began
        if (l.base != base) return error.StaleExtent;
        f.filling = .{ .stream = l.stream, .base = base };
        f.bound = null;
        try f.syncTwin();
    }

    fn syncTwin(f: *Follower) !void {
        const fl = f.filling orelse return;
        const p = &(f.planner orelse return);
        if (p.liveIndex(fl.stream) == null) {
            f.filling = null;
            f.bound = null;
            return;
        }
        const want = wantTwin(p, fl, f.rank);
        if (sameBound(f.bound, want)) return;
        try f.dev.twin(f.dev.ctx, fl.base, want.on, want.local);
        f.bound = want;
    }

    /// A verify window as rank 0 built it: the live stream in its slot at its extent, part and generation (a frame
    /// for a reused extent fails closed).
    pub fn checkWin(f: *const Follower, slot: usize, base: usize, end: usize, part: u8, gen: u64) !void {
        const l = f.liveAt(slot) orelse return error.StaleExtent;
        if (l.base != base or l.base + l.size != end or l.part != part or l.gen != gen) return error.StaleExtent;
    }
};

// -- tests -------------------------------------------------------------------------------------------------------------

const testing = std.testing;
const C = parts.chunk_rows;

/// A device that records what it was asked (and fails transfers on request).
pub const FakeDev = struct {
    copies: std.ArrayList([3]usize) = .empty,
    twins: std.ArrayList(Bound) = .empty,
    fail_next: bool = false, // the next copy's enqueue fails
    fail_check: bool = false, // the next check refuses (every rank's alike in the tests that set it)
    fail_wait: bool = false, // the next wait does not complete in time
    fail_twin: bool = false, // the next twin bind fails
    gpa: Allocator,

    pub fn dev(d: *FakeDev) Dev {
        return .{ .ctx = d, .check = checkFn, .copy = copyFn, .twin = twinFn, .wait = waitFn };
    }

    fn checkFn(ctx: *anyopaque, _: usize, _: usize, _: usize) anyerror!void {
        const d: *FakeDev = @ptrCast(@alignCast(ctx));
        if (d.fail_check) {
            d.fail_check = false;
            return error.CrossesPart;
        }
    }

    fn waitFn(ctx: *anyopaque, _: u64) anyerror!void {
        const d: *FakeDev = @ptrCast(@alignCast(ctx));
        if (d.fail_wait) {
            d.fail_wait = false;
            return error.TransferTimeout;
        }
    }

    fn copyFn(ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void {
        const d: *FakeDev = @ptrCast(@alignCast(ctx));
        if (d.fail_next) {
            d.fail_next = false;
            return error.InjectedFailure;
        }
        try d.copies.append(d.gpa, .{ from, to, rows });
    }

    fn twinFn(ctx: *anyopaque, _: usize, on: bool, local: ?usize) anyerror!void {
        const d: *FakeDev = @ptrCast(@alignCast(ctx));
        if (d.fail_twin) {
            d.fail_twin = false;
            return error.BadView;
        }
        try d.twins.append(d.gpa, .{ .on = on, .local = local });
    }

    pub fn deinit(d: *FakeDev) void {
        d.copies.deinit(d.gpa);
        d.twins.deinit(d.gpa);
    }
};

/// Rank 0's lanes as far as the hooks see them.
const FakeLanes = struct {
    staled: std.ArrayList(u64) = .empty,
    moved: std.ArrayList(u64) = .empty,
    gpa: Allocator,

    fn hooks(x: *FakeLanes) Hooks {
        return .{ .ctx = x, .stale = staleFn, .moved = movedFn };
    }

    fn staleFn(ctx: *anyopaque, kept: u64) void {
        const x: *FakeLanes = @ptrCast(@alignCast(ctx));
        x.staled.append(x.gpa, kept) catch unreachable;
    }

    fn movedFn(ctx: *anyopaque, stream: u64) void {
        const x: *FakeLanes = @ptrCast(@alignCast(ctx));
        x.moved.append(x.gpa, stream) catch unreachable;
    }

    fn deinit(x: *FakeLanes) void {
        x.staled.deinit(x.gpa);
        x.moved.deinit(x.gpa);
    }
};

/// Rank 0 and an in-process follower: rank 0's frames go straight to the follower's handler, in order (lanes.zig's
/// tests drive it with the lanes' own hooks: initWith).
pub const Pair = struct {
    gpa: Allocator,
    kv: Kv,
    dev0: FakeDev,
    dev1: FakeDev,
    lanes0: FakeLanes,
    fol: *Follower,
    leader: rank_wire.Leader,
    frames: u64 = 0,

    fn init(gpa: Allocator, cfg: parts.Config, settings: Settings) !*Pair {
        return initWith(gpa, cfg, settings, null);
    }

    /// `hooks`: rank 0's lanes (null: a recorder).
    pub fn initWith(gpa: Allocator, cfg: parts.Config, settings: Settings, hooks: ?Hooks) !*Pair {
        const p = try gpa.create(Pair);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .kv = undefined, .dev0 = .{ .gpa = gpa }, .dev1 = .{ .gpa = gpa }, .lanes0 = .{ .gpa = gpa }, .fol = undefined, .leader = undefined };
        p.kv = try Kv.init(gpa, cfg, settings, p.dev0.dev(), hooks orelse p.lanes0.hooks());
        p.fol = try Follower.init(gpa, 1, cfg, settings, p.dev1.dev());
        p.leader = rank_wire.Leader.init(2, 9, shardExchange(0, 2, cfg, settings));
        try agreeInProcess(&p.leader, &p.fol.rw);
        try p.kv.lead(.{ .ctx = p, .rw = rwFn }, &.{}, &p.leader, 0, undefined);
        try p.kv.start();
        return p;
    }

    fn rwFn(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const p: *Pair = @ptrCast(@alignCast(ctx));
        p.frames += 1;
        try p.fol.onFrame(undefined, sub); // (no asks without links: the link is never used)
    }

    pub fn same(p: *Pair) !void {
        try testing.expectEqualSlices(u8, &p.kv.planner.digest(), &p.fol.planner.?.digest());
    }

    pub fn deinit(p: *Pair) void {
        p.kv.deinit();
        p.fol.deinit();
        p.dev0.deinit();
        p.dev1.deinit();
        p.lanes0.deinit();
        p.gpa.destroy(p);
    }
};

/// The startup agreement between a leader and an in-process follower, without the reply link (tests only).
fn agreeInProcess(l: *rank_wire.Leader, f: *rank_wire.Follower) !void {
    var buf: [rank_wire.max_frame]u8 = undefined;
    const hello: rank_wire.Msg = .{ .hello = .{ .world = l.world, .fx = l.fx } };
    var head: rank_wire.Head = .{ .op = .hello, .from = 0, .epoch = l.epoch, .txn = 0, .seq = l.t.next_seq };
    try l.t.admit(&.{ .head = head, .msg = hello });
    _ = try f.accept(try rank_wire.encode(&buf, head, hello));
    f.owed = null;
    var a: fingerprint.Agreement = .{ .common = l.fx.common, .world = l.world };
    a.ranks[0] = l.fx.rank;
    a.ranks[1] = f.fx.rank;
    const agreed: rank_wire.Msg = .{ .agreed = .{ .world = l.world, .common = a.common, .ranks = a.ranks } };
    head = .{ .op = .agreed, .from = 0, .epoch = l.epoch, .txn = 0, .seq = l.t.next_seq };
    try l.t.admit(&.{ .head = head, .msg = agreed });
    _ = try f.accept(try rank_wire.encode(&buf, head, agreed));
    l.agreement = a;
}

const sharded: parts.Config = .{ .cap = 64 * C, .parts = 2, .carry_bytes = 720 };

test "ops cross the wire unchanged" {
    const samples = [_]Op{
        .{ .stage = .{ .kept = 3, .gen = 4 } },
        .{ .carry = .{ .kept = 1, .kept_gen = 2, .stream = 3, .src = 4 * C, .src_part = 0, .dst = 40 * C, .dst_part = 1, .rows = 2 * C, .why = .pressure } },
        .{ .move = .{ .stream = 9, .gen = 2, .src = 0, .src_part = 0, .dst = 33 * C, .dst_part = 1, .size = 4 * C, .rows = 3000, .hold_gen = 7, .tick = -5 } },
        .{ .untwin = .{ .stream = 1, .gen = 2, .revoke = true } },
    };
    for (samples) |o| try testing.expectEqualDeep(o, fromWire(toWire(o)));
    try testing.expectEqual(@as(usize, 4 * C), transferRows(3 * C + 1, 4 * C));
    try testing.expectEqual(@as(usize, 2 * C), transferRows(2 * C, 4 * C));
}

test "settings: carry and twins on by default, 0 turns each off, anything else is refused" {
    const env = struct {
        var vals: [3]?[]const u8 = .{ null, null, null };
        fn get(n: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, n, "TF_DS_KV_SHARD_CARRY")) return vals[0];
            if (std.mem.eql(u8, n, "TF_DS_KV_SHARD_TWINS")) return vals[1];
            return vals[2];
        }
    };
    try testing.expectEqualDeep(Settings{}, try Settings.fromEnv(env.get));
    env.vals = .{ "0", "0", "16" };
    try testing.expectEqualDeep(Settings{ .carry = false, .twins = false, .check_rounds = 16 }, try Settings.fromEnv(env.get));
    env.vals = .{ "yes", null, null };
    try testing.expectError(error.BadKvShardSetting, Settings.fromEnv(env.get));
}

test "admissions are part-local, the follower's planner stays rank 0's, and a release frees the part" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{ .twins = false });
    defer p.deinit();
    const half = sharded.cap / 2;
    var ids: [4]u64 = undefined;
    for (&ids, 0..) |*id, i| {
        const a = try p.kv.admit(100 + i, 7, @intCast(i), 12 * C, null, false);
        id.* = a.id;
        try testing.expect(a.base + 12 * C <= (@as(usize, a.part) + 1) * half and a.base >= @as(usize, a.part) * half);
        try p.same();
    }
    // the parts fill evenly (fewer live streams first)
    var per: [2]usize = .{ 0, 0 };
    for (p.kv.planner.lives.items) |l| per[l.part] += 1;
    try testing.expectEqual([2]usize{ 2, 2 }, per);
    // a request no part holds waits (no ops, nothing sent), counted once however often it is planned again
    const frames = p.frames;
    try testing.expectError(error.ContextFull, p.kv.admit(200, 1, 5, 20 * C, null, false));
    try testing.expectError(error.ContextFull, p.kv.admit(200, 1, 5, 20 * C, null, false));
    try testing.expectEqual(frames, p.frames);
    try testing.expectEqual(@as(u64, 1), p.kv.planner.counters.no_part);
    p.kv.forgetKey(200);
    for (ids) |id| try p.kv.release(id);
    try p.same();
    try testing.expectEqual(@as(usize, 0), p.kv.planner.lives.items.len);
    try testing.expectEqual(@as(usize, 0), p.kv.waiting.items.len);
}

test "a hit its part cannot hold is carried: both ranks transfer in order and finish with rank 0's ok" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{ .twins = false });
    defer p.deinit();
    const half = sharded.cap / 2;
    // a conversation in part 0 keeps a state of 8 blocks, then a long stream fills the rest of part 0 around it
    const a = try p.kv.admit(1, 1, 0, 10 * C, null, false);
    try testing.expectEqual(@as(u8, 0), a.part);
    try p.kv.keep(500, a.id, 8 * C);
    try p.same();
    const big = try p.kv.admit(2, 2, 1, 16 * C, null, false); // part 1 (fewer streams)
    try testing.expectEqual(@as(u8, 1), big.part);
    const fill0 = try p.kv.admit(3, 3, 2, 20 * C, null, false); // part 0 past the conversation
    try testing.expectEqual(@as(u8, 0), fill0.part);
    try p.kv.release(a.id);
    // the conversation's next turn: its own part has no room for 12 blocks without dropping it; part 1 has
    const next = try p.kv.admit(4, 4, 0, 12 * C, 500, false);
    try p.same();
    try testing.expect(next.resumed);
    try testing.expectEqual(parts.Mode.carry, next.mode);
    try testing.expectEqual(@as(u8, 1), next.part);
    try testing.expect(next.base >= half);
    // both ranks ran the one transfer: the kept rows from their base to the new extent
    try testing.expectEqual(@as(usize, 1), p.dev0.copies.items.len);
    try testing.expectEqualSlices([3]usize, p.dev0.copies.items, p.dev1.copies.items);
    try testing.expectEqual([3]usize{ 0, next.base, 8 * C }, p.dev0.copies.items[0]);
    try testing.expectEqual(@as(u64, 1), p.kv.planner.counters.hits);
    try testing.expectEqual(@as(u64, 8 * C * 720), p.kv.planner.counters.bytes);
    // the source stays, readable again once finished
    try testing.expect(!p.kv.planner.readBy(500));
    try testing.expect(p.kv.planner.keptAt(500) != null);
    try p.kv.release(next.id);
    try p.kv.release(big.id);
    try p.kv.release(fill0.id);
    try p.same();
}

test "a failed transfer on rank 0 runs the prompt from 0 on both ranks; a follower's own failure under rank 0's ok ends the pair" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{ .twins = false });
    defer p.deinit();
    const a = try p.kv.admit(1, 1, 0, 10 * C, null, false);
    try p.kv.keep(500, a.id, 8 * C);
    // a second request on the same prefix while the first holds it: copied within the part (it would wait, but not
    // patient: a copy)
    p.dev0.fail_next = true;
    const b = try p.kv.admit(2, 2, 1, 10 * C, 500, false);
    try testing.expect(!b.resumed);
    try testing.expectEqual(@as(u64, 1), p.kv.planner.counters.failed);
    try p.same();
    try testing.expectEqual(@as(usize, 0), p.kv.planner.lives.items[p.kv.planner.liveIndex(b.id).?].valid);
    try p.kv.release(b.id);
    // the follower's own copy fails while rank 0's works: rank 0's ok reaches it and the pair ends
    p.dev1.fail_next = true;
    try testing.expectError(error.PairBroken, p.kv.admit(3, 3, 1, 10 * C, 500, false));
    try testing.expectEqual(@as(anyerror, error.TransferFailed), p.leader.broken.?.err); // (rank_wire: a refused post ends the pair)
}

test "twins: a fresh prompt borrows the other part all or none, both ranks bind it, the first decode round gives it back" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{});
    defer p.deinit();
    const half = sharded.cap / 2;
    const a = try p.kv.admit(1, 1, 0, 8 * C, null, false);
    try testing.expect(p.kv.planner.hasTwins(a.id));
    try p.same();
    try p.kv.fillBegin(a.id, a.base);
    try p.fol.fillBegin(0, a.base);
    // rank 0 owns part 0 (the extent): on, no local twin; rank 1 binds its twin in part 1
    try testing.expectEqual(@as(usize, 1), p.dev0.twins.items.len);
    try testing.expect(p.dev0.twins.items[0].on and p.dev0.twins.items[0].local == null);
    try testing.expect(p.dev1.twins.items[0].on and p.dev1.twins.items[0].local.? >= half);
    // pinned while a chunk runs: no room for an admission that would need it back
    p.kv.planner.setPartial(a.id, true);
    try testing.expect(!try p.kv.planner.fits(gpa, 28 * C));
    p.kv.planner.setPartial(a.id, false);
    try testing.expect(try p.kv.planner.fits(gpa, 28 * C));
    // (the fill's last chunk ran: no fill end on either rank, both still hold it as their fill)
    try p.kv.decodeStart(a.id);
    try testing.expect(!p.kv.planner.hasTwins(a.id));
    try p.same();
    // both ranks unbind at the same frame (the untwin): the same binds, the views plain again
    try testing.expectEqual(p.dev0.twins.items.len, p.dev1.twins.items.len);
    try testing.expectEqual(@as(usize, 2), p.dev0.twins.items.len);
    try testing.expect(!p.dev0.twins.items[1].on and !p.dev1.twins.items[1].on);
    try p.kv.release(a.id);
    try p.same();
}

test "twins taken back between chunks unbind the fill's view on both ranks" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{});
    defer p.deinit();
    const half = sharded.cap / 2;
    const a = try p.kv.admit(1, 1, 0, 20 * C, null, false);
    try p.kv.fillBegin(a.id, a.base);
    try p.fol.fillBegin(0, a.base);
    try testing.expect(p.kv.planner.hasTwins(a.id));
    // between two chunks (not pinned) a request needs part 1's rows: the twin is revoked, both views go plain
    const b = try p.kv.admit(2, 2, 1, half, null, false);
    try testing.expect(!p.kv.planner.hasTwins(a.id));
    try testing.expectEqual(@as(u8, 1), b.part);
    try p.same();
    try testing.expect(!p.dev0.twins.items[p.dev0.twins.items.len - 1].on);
    try testing.expect(!p.dev1.twins.items[p.dev1.twins.items.len - 1].on);
    try testing.expectEqual(@as(u64, 1), p.kv.planner.counters.twins_revoked);
}

test "a request that would wait moves the cheapest decoding stream to the other part; the lanes hear it" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{ .twins = false });
    defer p.deinit();
    const half = sharded.cap / 2;
    // part 0: x (14 blocks, 18 free); part 1: z (10 blocks, 22 free); a 24-block request fits neither
    const x = try p.kv.admit(1, 1, 0, 14 * C, null, false);
    const z = try p.kv.admit(2, 2, 1, 10 * C, null, false);
    try testing.expectEqual(@as(u8, 0), x.part);
    try testing.expectEqual(@as(u8, 1), z.part);
    try p.kv.decodeStart(x.id);
    try p.kv.decodeStart(z.id);
    // x has used one block and a few rows, z three blocks: x is the cheaper move
    try p.kv.round(&.{ .{ .id = x.id, .valid = C + 5 }, .{ .id = z.id, .valid = 3 * C } });
    const w = try p.kv.admit(3, 3, 2, 24 * C, null, false);
    try p.same();
    try testing.expectEqual(@as(u64, 1), p.kv.counts.relocations);
    try testing.expectEqualSlices(u64, &.{x.id}, p.lanes0.moved.items);
    try testing.expectEqual(@as(usize, 1), p.dev0.copies.items.len);
    try testing.expectEqualSlices([3]usize, p.dev0.copies.items, p.dev1.copies.items);
    const xl = p.kv.planner.lives.items[p.kv.planner.liveIndex(x.id).?];
    try testing.expectEqual(@as(u8, 1), xl.part);
    try testing.expectEqual([3]usize{ 0, xl.base, 2 * C }, p.dev0.copies.items[0]); // its used rows, in whole blocks
    try testing.expectEqual(@as(u8, 0), w.part);
    try testing.expect(w.base + 24 * C <= half);
    try testing.expectEqual(@as(u64, 1), p.kv.planner.counters.live);
}

test "random traffic: rank 0's and the follower's planners never differ, every op list replays" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0118);
    const rnd = prng.random();
    for (0..6) |trial| {
        const p = try Pair.init(gpa, .{ .cap = 96 * C, .parts = 2, .carry_bytes = 720 }, .{ .twins = trial % 2 == 0, .carry = trial % 3 != 2 });
        defer p.deinit();
        var live: std.ArrayList(struct { id: u64, slot: u32, decoding: bool }) = .empty;
        defer live.deinit(gpa);
        var kept: std.ArrayList(u64) = .empty;
        defer kept.deinit(gpa);
        var next_kept: u64 = 1;
        var key: usize = 1;
        for (0..400) |_| {
            const roll = rnd.uintLessThan(u32, 10);
            if (roll < 4 and live.items.len < 8) {
                var used: [8]bool = @splat(false);
                for (live.items) |l| used[l.slot] = true;
                const slot: u32 = @intCast(std.mem.indexOfScalar(bool, &used, false).?);
                const size = (1 + rnd.uintLessThan(usize, 20)) * C;
                var src: ?u64 = null;
                if (kept.items.len > 0 and rnd.boolean()) {
                    const k = kept.items[rnd.uintLessThan(usize, kept.items.len)];
                    if (p.kv.planner.keptAt(k)) |kk| if (!kk.stale and kk.at < size) {
                        src = k;
                    };
                }
                key += 1;
                const a = p.kv.admit(key, key, slot, size, src, false) catch |err| switch (err) {
                    error.ContextFull => {
                        p.kv.forgetKey(key);
                        continue;
                    },
                    else => return err,
                };
                try live.append(gpa, .{ .id = a.id, .slot = slot, .decoding = false });
            } else if (roll < 6 and live.items.len > 0) {
                const i = rnd.uintLessThan(usize, live.items.len);
                const l = p.kv.planner.lives.items[p.kv.planner.liveIndex(live.items[i].id).?];
                const at = (1 + rnd.uintLessThan(usize, l.size / C)) * C;
                if (at < l.size) {
                    try p.kv.keep(next_kept, l.stream, at);
                    try kept.append(gpa, next_kept);
                    next_kept += 1;
                }
            } else if (roll < 7 and kept.items.len > 0) {
                const i = rnd.uintLessThan(usize, kept.items.len);
                try p.kv.drop(kept.items[i]);
                _ = kept.swapRemove(i);
            } else if (roll < 8 and live.items.len > 0) {
                const i = rnd.uintLessThan(usize, live.items.len);
                try p.kv.release(live.items[i].id);
                _ = live.swapRemove(i);
            } else if (live.items.len > 0) {
                var vs: [8]Kv.Valid = undefined;
                for (live.items, 0..) |*l, i| {
                    if (!l.decoding) {
                        try p.kv.decodeStart(l.id);
                        l.decoding = true;
                    }
                    vs[i] = .{ .id = l.id, .valid = C };
                }
                try p.kv.round(vs[0..live.items.len]);
            }
            try p.same();
        }
        try testing.expectEqualSlices([3]usize, p.dev0.copies.items, p.dev1.copies.items);
    }
}

/// Part 0 holds a kept state whose next turn its part cannot hold (the carry test's setup): the next admission carries.
fn carrySetup(p: *Pair) !struct { a: u64, big: u64, fill0: u64 } {
    const a = try p.kv.admit(1, 1, 0, 10 * C, null, false);
    try p.kv.keep(500, a.id, 8 * C);
    const big = try p.kv.admit(2, 2, 1, 16 * C, null, false);
    const fill0 = try p.kv.admit(3, 3, 2, 20 * C, null, false);
    try p.kv.release(a.id);
    return .{ .a = a.id, .big = big.id, .fill0 = fill0.id };
}

test "a carry across parts: a follower half that fails to enqueue, or rank 0's side not completing in time, ends the pair (never a hang)" {
    const gpa = testing.allocator;
    {
        const p = try Pair.init(gpa, sharded, .{ .twins = false });
        defer p.deinit();
        _ = try carrySetup(p);
        // rank 1's NCCL half fails to enqueue: it ends (rank 0's post of the ops fails here; over a link rank 0's
        // bounded wait or the reply deadline ends it)
        p.dev1.fail_next = true;
        try testing.expectError(error.PairBroken, p.kv.admit(4, 4, 0, 12 * C, 500, false));
        try testing.expectEqual(@as(anyerror, error.TransferFailed), p.leader.broken.?.err);
    }
    {
        const p = try Pair.init(gpa, sharded, .{ .twins = false });
        defer p.deinit();
        _ = try carrySetup(p);
        p.dev0.fail_wait = true;
        try testing.expectError(error.PairBroken, p.kv.admit(4, 4, 0, 12 * C, 500, false));
        try testing.expectEqual(@as(anyerror, error.TransferTimeout), p.leader.broken.?.err);
        try testing.expectEqual(@as(u16, 0), p.leader.broken.?.rank);
    }
    {
        // rank 0's own enqueue across parts fails: its peer's half would wait, so the pair ends too
        const p = try Pair.init(gpa, sharded, .{ .twins = false });
        defer p.deinit();
        _ = try carrySetup(p);
        p.dev0.fail_next = true;
        try testing.expectError(error.PairBroken, p.kv.admit(4, 4, 0, 12 * C, 500, false));
        try testing.expectEqual(@as(anyerror, error.InjectedFailure), p.leader.broken.?.err);
    }
}

test "a carry refused alike on every rank (nothing enqueued) finishes failed on both and the pair goes on" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{ .twins = false });
    defer p.deinit();
    const s = try carrySetup(p);
    p.dev0.fail_check = true;
    p.dev1.fail_check = true;
    const next = try p.kv.admit(4, 4, 0, 12 * C, 500, false);
    try testing.expect(!next.resumed);
    try testing.expectEqual(@as(usize, 0), p.dev0.copies.items.len);
    try testing.expectEqual(@as(usize, 0), p.dev1.copies.items.len);
    try testing.expectEqual(@as(u64, 1), p.kv.planner.counters.failed);
    try p.same();
    try testing.expectEqual(@as(usize, 0), p.fol.failed.items.len);
    for ([_]u64{ next.id, s.big, s.fill0 }) |id| try p.kv.release(id);
    try p.same();
}

test "a step failing after a placed plan releases its extent on both ranks; the slot is admitted again" {
    const gpa = testing.allocator;
    const p = try Pair.init(gpa, sharded, .{});
    defer p.deinit();
    const half = sharded.cap / 2;
    // a fills with its twin in part 1; b needs part 1 whole: its plan revokes a's twin, and rank 0's unbind fails
    const a = try p.kv.admit(1, 1, 0, 20 * C, null, false);
    try p.kv.fillBegin(a.id, a.base);
    try p.fol.fillBegin(0, a.base);
    p.dev0.fail_twin = true;
    try testing.expectError(error.BadView, p.kv.admit(2, 2, 1, half, null, false));
    try p.same();
    try testing.expectEqual(@as(usize, 1), p.kv.planner.lives.items.len);
    try testing.expectEqual(@as(usize, 1), p.fol.planner.?.lives.items.len);
    try testing.expectEqual(@as(usize, 0), p.kv.waiting.items.len);
    // the same lanes slot takes the next request (no SlotTaken on either rank)
    const b = try p.kv.admit(3, 3, 1, half, null, false);
    try testing.expectEqual(@as(u8, 1), b.part);
    try p.same();
    try p.kv.release(b.id);
    try p.kv.release(a.id);
    try p.same();
    try testing.expectEqual(@as(usize, 0), p.kv.planner.lives.items.len);
}

test "a full plan carrying ops is refused before rank 0's planner changes" {
    const gpa = testing.allocator;
    var dev: FakeDev = .{ .gpa = gpa };
    defer dev.deinit();
    var fl: FakeLanes = .{ .gpa = gpa };
    defer fl.deinit();
    var kv = try Kv.init(gpa, sharded, .{ .twins = false }, dev.dev(), fl.hooks());
    defer kv.deinit();
    try kv.start();
    _ = try kv.admit(1, 1, 0, 30 * C, null, false);
    _ = try kv.admit(2, 2, 1, 30 * C, null, false);
    const before = kv.planner.digest();
    // (a real planner never emits one: the guard is for a mutant or a future change)
    var plan = try kv.planner.planAdmit(gpa, .{ .stream = 9, .slot = 2, .size = 20 * C, .source = null, .patient = false });
    defer plan.deinit(gpa);
    try testing.expectEqual(parts.Status.full, plan.status);
    try plan.ops.append(gpa, .{ .stale = .{ .kept = 1, .gen = 1 } });
    try testing.expectError(error.BadPlan, kv.applyFull(&plan));
    try testing.expectEqualSlices(u8, &before, &kv.planner.digest());
    try testing.expectError(error.ContextFull, kv.admit(3, 3, 2, 20 * C, null, false)); // (the honest plan: no ops)
}

test "the follower's planner config takes the switches as rank 0's does" {
    const gpa = testing.allocator;
    var dev: FakeDev = .{ .gpa = gpa };
    defer dev.deinit();
    const f = try Follower.init(gpa, 1, sharded, .{ .carry = false, .twins = false }, dev.dev());
    defer f.deinit();
    try testing.expect(!f.cfg.carry and !f.cfg.twins);
}

/// A follower on its own thread over a loopback link: every frame through Follower.onFrame until an empty one.
const LoopPeer = struct {
    port: u16,
    cfg: parts.Config,
    settings: Settings, // (rank 0's: the hello's features must agree)
    dev: FakeDev,
    fol: ?*Follower = null,
    err: ?anyerror = null,

    fn run(p: *LoopPeer) void {
        p.loop() catch |e| {
            p.err = e;
        };
    }

    fn loop(p: *LoopPeer) !void {
        var l = try link.Link.connect(testing.io, .{ 127, 0, 0, 1 }, p.port, 10);
        defer l.close();
        p.fol = try Follower.init(testing.allocator, 1, p.cfg, p.settings, p.dev.dev());
        const buf = try testing.allocator.alloc(u8, rank_wire.max_frame);
        defer testing.allocator.free(buf);
        while (true) {
            const bytes = try l.recv(buf);
            if (bytes.len == 0) return;
            try p.fol.?.onFrame(l, bytes);
        }
    }
};

const LoopOut = struct {
    l: link.Link,
    fn rw(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const o: *LoopOut = @ptrCast(@alignCast(ctx));
        try o.l.send(sub);
    }
};

test "over a loopback link: hello, layouts, parts and checks; each transfer asks the follower's side (shard_xfer) before finishing; a failed local copy finishes failed" {
    const gpa = testing.allocator;
    var b: [2]u8 = undefined;
    testing.io.random(&b);
    const port = 30000 + std.mem.readInt(u16, &b, .little) % 20000;
    const peer = try gpa.create(LoopPeer);
    defer gpa.destroy(peer);
    const settings: Settings = .{ .twins = false, .check_rounds = 2 };
    peer.* = .{ .port = port, .cfg = sharded, .settings = settings, .dev = .{ .gpa = gpa } };
    defer peer.dev.deinit();
    defer if (peer.fol) |f| f.deinit();
    const th = try std.Thread.spawn(.{}, LoopPeer.run, .{peer});
    var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var out: LoopOut = .{ .l = l };
    var dev0: FakeDev = .{ .gpa = gpa };
    defer dev0.deinit();
    var lanes0: FakeLanes = .{ .gpa = gpa };
    defer lanes0.deinit();
    var kv = try Kv.init(gpa, sharded, settings, dev0.dev(), lanes0.hooks());
    defer kv.deinit();
    const links = [_]?link.Link{l};
    try kv.lead(.{ .ctx = &out, .rw = LoopOut.rw }, &links, null, 77, shardExchange(0, 2, sharded, settings));
    try kv.start(); // hello/agreed, shard_layout, shard_set, shard_check
    try testing.expectEqual(@as(u64, 1), kv.counts.checks);
    // a conversation kept in part 0, a resume while its stream still runs: copied in its part, both sides asked
    const a = try kv.admit(1, 1, 0, 10 * C, null, false);
    try kv.keep(500, a.id, 8 * C);
    const c1 = try kv.admit(2, 2, 1, 10 * C, 500, false);
    try testing.expect(c1.resumed);
    // the follower's side of this copy within part 0 fails (no peer half waits on it): its reply says so, both
    // ranks finish failed (the prompt from 0), the pair goes on (across parts the pair would end: next tests)
    peer.dev.fail_next = true;
    const c2 = try kv.admit(3, 3, 2, 10 * C, 500, false);
    try testing.expect(!c2.resumed);
    try testing.expectEqual(@as(u64, 1), kv.planner.counters.failed);
    try kv.decodeStart(c2.id);
    try kv.round(&.{.{ .id = c2.id, .valid = 1 }});
    try kv.round(&.{.{ .id = c2.id, .valid = 2 }}); // (the periodic check: equal)
    try testing.expectEqual(@as(u64, 2), kv.counts.checks);
    for ([_]u64{ a.id, c1.id, c2.id }) |id| try kv.release(id);
    try kv.check();
    try l.send(&.{}); // done
    th.join();
    try testing.expectEqual(@as(?anyerror, null), peer.err);
    const fol = peer.fol.?;
    try testing.expectEqualSlices(u8, &kv.planner.digest(), &fol.planner.?.digest());
    try testing.expectEqual(@as(usize, 2), dev0.copies.items.len);
    try testing.expectEqual(@as(usize, 1), peer.dev.copies.items.len); // (its second side failed)
    try testing.expectEqual(@as(usize, 0), fol.failed.items.len);
}

test "over a loopback link: a follower started with other shard switches is refused at hello, naming the field" {
    const gpa = testing.allocator;
    var b: [2]u8 = undefined;
    testing.io.random(&b);
    const port = 30000 + std.mem.readInt(u16, &b, .little) % 20000;
    const peer = try gpa.create(LoopPeer);
    defer gpa.destroy(peer);
    peer.* = .{ .port = port, .cfg = sharded, .settings = .{ .twins = false }, .dev = .{ .gpa = gpa } };
    defer peer.dev.deinit();
    defer if (peer.fol) |f| f.deinit();
    const th = try std.Thread.spawn(.{}, LoopPeer.run, .{peer});
    var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var out: LoopOut = .{ .l = l };
    var dev0: FakeDev = .{ .gpa = gpa };
    defer dev0.deinit();
    var lanes0: FakeLanes = .{ .gpa = gpa };
    defer lanes0.deinit();
    var kv = try Kv.init(gpa, sharded, .{}, dev0.dev(), lanes0.hooks());
    defer kv.deinit();
    const links = [_]?link.Link{l};
    try kv.lead(.{ .ctx = &out, .rw = LoopOut.rw }, &links, null, 78, shardExchange(0, 2, sharded, .{}));
    try testing.expectError(error.FingerprintMismatch, kv.start());
    try testing.expectEqual(@as(u16, @intFromEnum(fingerprint.Field.features)), kv.leader.?.broken.?.field);
    th.join();
    try testing.expectEqual(@as(?anyerror, error.FingerprintMismatch), peer.err);
}

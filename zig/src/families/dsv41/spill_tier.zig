//! The DeepSeek-V4.1 spill tier: core/prompt_cache's `Tier` over core/spill's per-rank block store and
//! spill_state's packing, with the ranks' agreement over rank_wire. Port of wojo's GLM 0088 (PR #78: the spill
//! foundation, per-rank identity and agreement) and 0110 (PR #124: BadAd84's block tier, write at keep, the staging
//! ring, HAS/LOAD/LOADPART/LOADED, bounded read and copy scheduling, slot recycling while a follower waits) as
//! PORT-SPILL-V2.md "Write path and publication" and "Async restore and agreement between ranks" ask of the native
//! engine. docs/port/spill-lanes.md describes the flow.
//!
//! Every rank keeps its own store (its own disk, its own planes) and its own extent generations. Rank 0 runs the
//! `Leader`: the prompt cache's tier, every decision (what is written, what is restored, whether a restore commits) and
//! the commands; each follower runs a `Follower` from its ordered command stream (lanes.follow), answering rank 0's
//! asks on the same link. Nothing here touches CUDA except through the `Dev` the caller gives (Model.SpillDev on the
//! GPU, a byte-array device in the tests), and nothing waits on a disk while the engine runs: the store's workers do
//! the file I/O, the engine thread only enqueues copies and polls their events.
//!
//! A restore is one rank_wire transaction: has (every rank holds the committed point) -> reserve (the destination on
//! every rank) -> start (reads) -> progress* (rank 0's copied-block boundary) -> validate (every rank copied and
//! installed it) -> commit, or abort (with the forget flag when a rank's copy was damaged) at any step. A rank that
//! cannot answer ends the pair: there is no local fallback while the peer follows another command sequence.
const std = @import("std");
const core = @import("core");
pub const prompt_cache = @import("engine_api").prompt_cache;
const state = @import("spill_state.zig");
const rank_wire = @import("rank_wire.zig");
const fingerprint = @import("fingerprint.zig");
const link = @import("link.zig");

const spill = core.spill;
const format = spill.format;
const pc = prompt_cache;
const Allocator = std.mem.Allocator;

pub const Digest = format.Digest;

/// The lanes.zig Kind that carries the spill notes (rank 0's write-at-keep and metadata, not answered).
pub const note_kind: u8 = 11;
/// The small state's one section in a point (snapCopy's bytes, the layout's small planes in order).
pub const section_name = "dsv41.small";
/// Blocks a pump stages at most (GLM 0110's PUMP_BLOCKS).
pub const pump_blocks = 4;

/// The smallest staging cap a rank takes (TF_DS_SPILL_STAGING_MIB / --spill-staging-mib), on every rank alike.
pub const min_staging_mib: u32 = 16;

/// The tier's switches on one rank (rank 0: the server's --spill-* and TF_SPILL_*; a follower: its own, see
/// Settings.follower). Off unless `dir` and `gib > 0`.
pub const Settings = struct {
    dir: []const u8,
    gib: f64,
    min_free_gib: f64 = 50,
    quota: f64 = 0,
    min_tokens: u32 = 8192,
    turns: u32 = 4,
    flush_ms: u32 = 60_000,
    /// Pinned host staging for every write, read and small state together (the plan's 128 MiB a rank).
    staging_mib: u32 = 128,
    /// Exact staging bytes instead of staging_mib (0: staging_mib's; tests and measurements).
    staging_bytes: u64 = 0,
    /// Small-state slots for queued writes, and restores at once (each holds a small-state slot).
    smalls: u32 = 4,
    restores: u32 = 2,
    readers: u32 = 4,
    direct: bool = true,

    pub fn enabled(s: Settings) bool {
        return s.dir.len > 0 and s.gib > 0;
    }

    pub fn capBytes(s: Settings) u64 {
        return @intFromFloat(s.gib * (1 << 30));
    }

    pub fn floorBytes(s: Settings) u64 {
        return @intFromFloat(@max(s.min_free_gib, 0) * (1 << 30));
    }

    pub fn stagingBytes(s: Settings) u64 {
        if (s.staging_bytes > 0) return s.staging_bytes;
        return @as(u64, s.staging_mib) << 20;
    }

    /// A follower's settings: its flags first (`flag` returns a flag's value or null), else the environment
    /// (TF_DS_SPILL_DIR, TF_DS_SPILL_GIB, TF_DS_SPILL_MIN_FREE_GIB, TF_DS_SPILL_MIN_TOKENS, TF_DS_SPILL_QUOTA,
    /// TF_DS_SPILL_STAGING_MIB, and the server's TF_SPILL_TURNS / TF_SPILL_FLUSH_MS); null when spill is off there.
    pub fn follower(flag: *const fn ([]const u8) ?[]const u8, env: *const fn ([:0]const u8) ?[]const u8) !?Settings {
        const dir = flag("--spill-dir") orelse env("TF_DS_SPILL_DIR") orelse return null;
        if (dir.len == 0) return null;
        var s: Settings = .{ .dir = dir, .gib = 0 };
        s.gib = try num(f64, flag("--spill-gib") orelse env("TF_DS_SPILL_GIB"), 0);
        if (!s.enabled()) return null;
        s.min_free_gib = try num(f64, flag("--spill-min-free-gib") orelse env("TF_DS_SPILL_MIN_FREE_GIB"), s.min_free_gib);
        s.min_tokens = try num(u32, flag("--spill-min-tokens") orelse env("TF_DS_SPILL_MIN_TOKENS"), s.min_tokens);
        s.quota = try num(f64, flag("--spill-quota") orelse env("TF_DS_SPILL_QUOTA"), s.quota);
        s.staging_mib = try num(u32, flag("--spill-staging-mib") orelse env("TF_DS_SPILL_STAGING_MIB"), s.staging_mib);
        s.turns = try num(u32, env("TF_SPILL_TURNS"), s.turns);
        s.flush_ms = try num(u32, env("TF_SPILL_FLUSH_MS"), s.flush_ms);
        if (!std.math.isFinite(s.gib) or !std.math.isFinite(s.quota) or s.quota < 0 or s.quota > 1) return error.BadSpillSetting;
        if (s.staging_mib < min_staging_mib) return error.BadSpillSetting;
        return s;
    }

    fn num(comptime T: type, text: ?[]const u8, default: T) !T {
        const t = std.mem.trim(u8, text orelse return default, " ");
        if (t.len == 0) return default;
        return switch (@typeInfo(T)) {
            .float => std.fmt.parseFloat(T, t) catch error.BadSpillSetting,
            else => std.fmt.parseInt(T, t, 10) catch error.BadSpillSetting,
        };
    }
};

/// The staging cap split for a layout (slots.plan: small slots first, the rest half writes, half reads).
pub fn stagingPlan(s: Settings, l: *const state.StateLayout) !spill.slots.Plan {
    return spill.slots.plan(s.stagingBytes(), l.block_bytes, l.small_bytes, s.smalls, s.restores);
}

/// Events a rank needs (one a staging slot, stable for the store's life, and one for clears).
pub fn eventCount(p: spill.slots.Plan) usize {
    return p.write + p.read + p.small + p.restore + 1;
}

/// The id the prompt cache knows a point by: its key's first 8 bytes (the key is a SHA-256).
pub fn pointId(key: Digest) u64 {
    return std.mem.readInt(u64, key[0..8], .little);
}

fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A failed restore's reason as an error the prompt cache logs.
fn failureError(why: rank_wire.Failure) anyerror {
    return switch (why) {
        .presence => error.SpillNotOnEveryRank,
        .reserve => error.SpillReserve,
        .read => error.SpillRead,
        .crc => error.SpillDamaged,
        .truncated => error.SpillTruncated,
        .missing => error.SpillMissing,
        .cancelled => error.Cancelled,
        .timeout => error.SpillTimeout,
        .unpack => error.SpillUnpack,
        .shutdown => error.SpillShutdown,
        else => error.SpillAborted,
    };
}

/// Whether a failure is damage to the stored point (every rank then forgets it).
fn damage(why: rank_wire.Failure) bool {
    return switch (why) {
        .crc, .truncated, .missing, .unpack => true,
        else => false,
    };
}

fn readWhy(e: ?spill.store.ReadError) rank_wire.Failure {
    return switch (e orelse return .read) {
        .missing => .missing,
        .damaged, .small => .crc,
        .io => .read,
        .cancelled => .cancelled,
    };
}

/// TierCounts as rank_wire's stats_reply carries them (its fields in declaration order).
pub fn countsValues(c: pc.TierCounts) struct { n: u8, values: [rank_wire.max_stats]u64 } {
    const fields = @typeInfo(pc.TierCounts).@"struct".field_names;
    comptime std.debug.assert(fields.len <= rank_wire.max_stats);
    var v: [rank_wire.max_stats]u64 = @splat(0);
    inline for (fields, 0..) |name, i| v[i] = @field(c, name);
    return .{ .n = fields.len, .values = v };
}

/// A stats_reply's values as TierCounts (fields past `n` stay zero: an older peer knows fewer).
pub fn valuesCounts(n: u8, values: *const [rank_wire.max_stats]u64) pc.TierCounts {
    var c: pc.TierCounts = .{};
    inline for (@typeInfo(pc.TierCounts).@"struct".field_names, 0..) |name, i| {
        if (i < n) @field(c, name) = values[i];
    }
    return c;
}

// -- one rank's side: its store, packing, staging events, writes and restores ----------------------------------------

/// What a keep stores on a rank.
pub const KeepArgs = struct {
    keyed: []const u32, // [0, at)
    key: Digest, // format.pointKey(keyed[0..at], block tokens, no media): the caller's, hashed once
    at: u32,
    base: usize, // the kept state's extent
    snap: u64, // its Snap.id
    buf: u64, // its device buffer (the small state)
    lineage: format.Lineage,
    origin: ?format.Origin = null,
    shared: bool = false,
    group: u64 = 0,
};

pub const Kept = enum { queued, stored, refused };

/// A restore's destination and progress on one rank.
pub const Rx = struct {
    id: u64,
    key: Digest,
    at: u32,
    blocks: u32,
    rid: ?spill.ReadId = null,
    dst: ?state.ExtentRef = null,
    snap: state.SnapRef = .{ .id = 0, .buf = 0 },
    copied: u32 = 0, // blocks whose copy into the destination is enqueued (in order)
    inflight: u32 = 0, // of those, copies whose event has not completed
    small: enum { none, pending, done } = .none,
    why: rank_wire.Failure = .none,
    bad_block: u32 = rank_wire.no_block,

    fn fail(rx: *Rx, why: rank_wire.Failure) void {
        if (rx.why == .none) {
            rx.why = why;
            rx.bad_block = rx.copied;
        }
    }
};

pub fn Side(comptime Dev: type) type {
    return struct {
        const Self = @This();
        pub const Packer = state.Packer(Dev);

        const Write = struct { job: spill.JobId, key: Digest, pin: state.ExtentRef, capturing: bool = true };
        const Copy = struct {
            ev: u32,
            what: union(enum) {
                stage: struct { t: spill.Ticket, pin: state.ExtentRef, pt: state.Ticket },
                capture: struct { job: spill.JobId, snap: u64 },
                read: struct { rx: u64, slot: spill.Slot },
                small: struct { rx: u64 },
            },
        };
        /// A write that ended without publishing (rank 0 tells the followers to drop theirs).
        pub const Lost = struct { snap: u64, key: Digest, why: rank_wire.CancelWhy };

        gpa: Allocator,
        io: std.Io,
        store: *spill.Store,
        layout: state.StateLayout,
        gens: state.Generations,
        pk: Packer,
        dev: Dev,
        plan: spill.slots.Plan,
        regions: [4][]align(spill.slots.alignment) u8,
        lists: [4][1]spill.slots.Buffer,
        events: []Dev.Event,
        bt: u32,
        copies: std.ArrayList(Copy) = .empty,
        writes: std.AutoHashMapUnmanaged(u64, Write) = .empty, // by Snap.id
        rxs: std.AutoHashMapUnmanaged(u64, *Rx) = .empty,
        reqs: []spill.StageReq,
        lost: std.ArrayList(Lost) = .empty,
        cached: pc.TierCounts = .{},
        cancelled_shutdown: u64 = 0,

        /// Opens this rank's store under `s.dir` (core/spill names the namespace from `compat`) over `staging`
        /// (pinned, at least stagingPlan's bytes, 4 KiB aligned) and `events` (eventCount of them); both stay the
        /// caller's. `gens` is moved in. `max_tokens`: the longest point (the window).
        pub fn create(gpa: Allocator, io: std.Io, dev: Dev, layout: state.StateLayout, gens: state.Generations, s: Settings, rank: u32, compat: Digest, max_tokens: u32, staging: []align(spill.slots.alignment) u8, events: []Dev.Event, hooks: spill.store.Hooks) !*Self {
            const plan = try stagingPlan(s, &layout);
            if (staging.len < plan.bytes()) return error.StagingTooSmall;
            if (events.len < eventCount(plan)) return error.TooFewEvents;
            if (layout.owner_layout != .replicated) return error.OwnerShardedNotPorted; // PORT(round3): owner_sharded
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .store = undefined, .layout = layout, .gens = gens, .pk = undefined, .dev = dev, .plan = plan, .regions = undefined, .lists = undefined, .events = events, .bt = @intCast(layout.geo.block_tokens), .reqs = &.{} };
            self.pk = .{ .dev = dev, .layout = &self.layout, .gens = &self.gens };
            var off: usize = 0;
            for ([_]usize{ plan.write * plan.block_slot, plan.read * plan.block_slot, plan.small * plan.small_slot, plan.restore * plan.small_slot }, 0..) |n, i| {
                self.regions[i] = @alignCast(staging[off..][0..n]);
                self.lists[i] = .{self.regions[i]};
                off += n;
            }
            self.reqs = try gpa.alloc(spill.StageReq, @max(plan.write, 1));
            errdefer gpa.free(self.reqs);
            const dir = try gpa.dupeSentinel(u8, s.dir, 0);
            defer gpa.free(dir);
            self.store = try spill.Store.open(gpa, io, .{
                .root = dir,
                .rank = rank,
                .compat = compat,
                .owner_layout = .replicated,
                .limits = .{ .max_tokens = max_tokens, .block_tokens = self.bt, .block_bytes = layout.block_bytes, .max_small_bytes = std.mem.alignForward(u64, layout.small_bytes, 256) },
                .cap_bytes = s.capBytes(),
                .floor_bytes = s.floorBytes(),
                .quota = s.quota,
                .turns = s.turns,
                .min_tokens = s.min_tokens,
                .staging = .{ .write = &self.lists[0], .read = &self.lists[1], .small = &self.lists[2], .restore = &self.lists[3] },
                .readers = s.readers,
                .max_reads = s.restores,
                .direct = s.direct,
                .hooks = hooks,
            });
            self.refresh();
            return self;
        }

        /// Waits every copy still on the device, closes the store (unpublished points stay unpublished), frees.
        pub fn destroy(self: *Self) void {
            for (self.copies.items) |cp| self.dev.wait(self.events[cp.ev]);
            _ = self.retire();
            var it = self.rxs.valueIterator();
            while (it.next()) |rx| self.gpa.destroy(rx.*);
            self.rxs.deinit(self.gpa);
            self.store.close();
            self.gens.deinit(self.gpa);
            self.copies.deinit(self.gpa);
            self.writes.deinit(self.gpa);
            self.lost.deinit(self.gpa);
            self.gpa.free(self.reqs);
            self.gpa.destroy(self);
        }

        fn slotEvent(self: *const Self, slot: spill.Slot) u32 {
            const p = self.plan;
            const base: usize = switch (slot.pool) {
                .write => 0,
                .read => p.write,
                .small => p.write + p.read,
                .restore => p.write + p.read + p.small,
            };
            return @intCast(base + slot.index);
        }

        /// The event of the slot a store-given slice starts in (small and restore slots: Queued.small, SmallView).
        fn addrEvent(self: *const Self, pool: spill.slots.Pool, bytes: []const u8) u32 {
            const region = self.regions[@intFromEnum(pool)];
            const per = if (pool == .write or pool == .read) self.plan.block_slot else self.plan.small_slot;
            const i = (@intFromPtr(bytes.ptr) - @intFromPtr(region.ptr)) / per;
            return self.slotEvent(.{ .pool = pool, .index = @intCast(i) });
        }

        fn clearEvent(self: *const Self) u32 {
            return @intCast(eventCount(self.plan) - 1);
        }

        // -- generations ------------------------------------------------------------------------------------------

        /// Pool positions [lo, hi) are about to change (a prompt chunk, a round's rows, a copy's destination).
        pub fn bump(self: *Self, lo: usize, hi: usize) void {
            self.gens.bump(lo, @min(hi, self.gens.cap));
        }

        pub fn bumpSlot(self: *Self, slot: usize) void {
            self.gens.bumpSlot(slot);
        }

        // -- writes -----------------------------------------------------------------------------------------------

        /// The write of a kept state: its rows pinned as they stand, its small state copied from its Snap buffer now
        /// (packSnap, rule a: the Snap is live and owns `buf` at this call), its blocks staged later by `service`.
        pub fn keepPoint(self: *Self, k: KeepArgs) Kept {
            const at = k.at;
            if (at == 0 or at % self.bt != 0 or k.keyed.len < at) return .refused;
            if (self.writes.contains(k.snap)) return .stored;
            const pin = self.gens.pin(k.base, k.base + at) catch return .refused;
            self.writes.ensureUnusedCapacity(self.gpa, 1) catch return .refused;
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return .refused;
            const sections = [_]spill.store.SectionSpec{.{ .name = section_name, .len = self.layout.small_bytes }};
            const r = self.store.keep(.{ .keyed = k.keyed[0..at], .lineage = k.lineage, .origin = k.origin, .shared = k.shared, .group = k.group, .sections = &sections, .source = .{ .id = k.snap, .generation = pin.stamp } });
            switch (r) {
                .stored => return .stored,
                .queued => |q| {
                    const ev = self.addrEvent(.small, q.small);
                    if (q.small.len < self.layout.small_bytes or q.sections.len != 1 or q.sections[0].offset != 0) {
                        self.store.captured(q.job, false);
                        return .refused;
                    }
                    _ = self.pk.packSnap(.{ .id = k.snap, .buf = k.buf }, q.small[0..self.layout.small_bytes], self.events[ev]) catch {
                        self.dev.wait(self.events[ev]); // (a copy that failed partway still recorded its event)
                        self.store.captured(q.job, false);
                        return .refused;
                    };
                    self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .capture = .{ .job = q.job, .snap = k.snap } } });
                    self.writes.putAssumeCapacity(k.snap, .{ .job = q.job, .key = k.key, .pin = pin });
                    return .queued;
                },
                else => return .refused,
            }
        }

        /// One staged block's pack copy (its event the slot's), or the store told it could not be made.
        fn packReq(self: *Self, req: spill.StageReq) void {
            const w = self.writes.get(req.source.id) orelse return self.store.staged(req.ticket, false);
            if (w.pin.stamp != req.source.generation or req.first % self.bt != 0) return self.store.staged(req.ticket, false);
            const ev = self.slotEvent(req.ticket.slot);
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return self.store.staged(req.ticket, false);
            const pt = self.pk.packBlocks(w.pin, req.first / self.bt, 1, req.dst, self.events[ev]) catch {
                self.dev.wait(self.events[ev]);
                return self.store.staged(req.ticket, false); // StaleExtent: the rows changed under the pin
            };
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .stage = .{ .t = req.ticket, .pin = w.pin, .pt = pt } } });
        }

        /// Kept state `snap`'s rows from token `from` on are about to change: its unpublished write stages what it
        /// still reads there now (the copies queued before the change, in stream order) or is cancelled.
        pub fn guard(self: *Self, snap: u64, from: u32) void {
            const w = self.writes.get(snap) orelse return;
            const r = self.store.guard(.{ .id = snap, .generation = w.pin.stamp }, from, self.reqs);
            for (self.reqs[0..r.staged]) |req| self.packReq(req);
        }

        /// Kept state `snap` is about to be freed: its write guarded from 0, its small-state capture waited for
        /// (rule b: the buffer's address cannot come back under a copy still reading it).
        pub fn dropSnap(self: *Self, snap: u64) void {
            self.guard(snap, 0);
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (cp.what == .capture and cp.what.capture.snap == snap) {
                    self.dev.wait(self.events[cp.ev]);
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                } else i += 1;
            }
        }

        /// Whether a write of point `key` is still queued here (not yet published, cancelled or failed).
        pub fn writing(self: *const Self, key: Digest) bool {
            var it = self.writes.valueIterator();
            while (it.next()) |w| if (std.mem.eql(u8, &w.key, &key)) return true;
            return false;
        }

        /// Cancel kept state `snap`'s unpublished write (rank 0 dropped its own).
        pub fn cancelWrite(self: *Self, snap: u64) void {
            const w = self.writes.get(snap) orelse return;
            _ = self.store.cancel(w.job);
        }

        /// A copy's event completed: hand its result on.
        fn done(self: *Self, cp: Copy) void {
            switch (cp.what) {
                .stage => |x| self.store.staged(x.t, x.pt.belongsTo(x.pin)),
                .capture => |x| {
                    self.store.captured(x.job, true);
                    if (self.writes.getPtr(x.snap)) |w| w.capturing = false;
                },
                .read => |x| {
                    self.store.readRelease(x.slot);
                    if (self.rxs.get(x.rx)) |rx| rx.inflight -= 1;
                },
                .small => |x| if (self.rxs.get(x.rx)) |rx| {
                    rx.small = .done;
                },
            }
        }

        /// Every copy whose event completed, handed on (never waits); how many.
        pub fn retire(self: *Self) usize {
            var n: usize = 0;
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (self.dev.done(self.events[cp.ev])) {
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                    n += 1;
                } else i += 1;
            }
            return n;
        }

        /// Between rounds and prompt chunks, and at idle: completed copies handed on, at most `max_blocks` more
        /// blocks staged within `budget_ns`, finished writes collected. Bounded; never waits. The copies retired and
        /// blocks staged (0: nothing moved).
        pub fn service(self: *Self, budget_ns: u64, max_blocks: usize) usize {
            const deadline = monotonicNs() +| budget_ns;
            var moved = self.retire();
            var left = max_blocks;
            while (left > 0 and monotonicNs() < deadline) {
                const n = self.store.stage(self.reqs[0..@min(self.reqs.len, left, 4)]);
                if (n == 0) break;
                for (self.reqs[0..n]) |req| self.packReq(req);
                left -|= n;
                moved += n;
            }
            self.collect();
            self.refresh();
            return moved;
        }

        /// The store's completions: finished writes leave `writes` (a lost one is listed for rank 0's notice).
        fn collect(self: *Self) void {
            var out: [16]spill.Completion = undefined;
            while (true) {
                const n = self.store.poll(&out);
                for (out[0..n]) |c| {
                    const job: spill.JobId = switch (c) {
                        inline else => |x| x.job,
                    };
                    const why: ?rank_wire.CancelWhy = switch (c) {
                        .published => null,
                        .failed => .io,
                        .cancelled => |x| switch (x.why) {
                            .guard, .source_gone => .overwrite,
                            .explicit => .drop,
                            .shutdown => .shutdown,
                        },
                    };
                    var it = self.writes.iterator();
                    while (it.next()) |e| if (e.value_ptr.job == job) {
                        const snap = e.key_ptr.*;
                        const key = e.value_ptr.key;
                        _ = self.writes.remove(snap);
                        if (why) |y| self.lost.append(self.gpa, .{ .snap = snap, .key = key, .why = y }) catch {};
                        break;
                    };
                }
                if (n < out.len) break;
            }
        }

        /// Copies on the device or writes waiting for the engine.
        pub fn busy(self: *Self) bool {
            return self.copies.items.len > 0 or self.store.pendingStage() > 0;
        }

        /// A clean stop: every queued write staged and drained by `deadline_ns` (monotonic); the points left
        /// unpublished (never published to make the count).
        pub fn flush(self: *Self, deadline_ns: u64) u64 {
            while (monotonicNs() < deadline_ns and (self.store.pendingStage() > 0 or self.copies.items.len > 0)) {
                // nothing retired or staged (the device busy, or every write slot still held by a writer): sleep, never spin
                if (self.service(deadline_ns -| monotonicNs(), std.math.maxInt(usize)) == 0) std.Io.sleep(self.io, .fromMilliseconds(1), .awake) catch {};
            }
            for (self.copies.items) |cp| self.dev.wait(self.events[cp.ev]);
            _ = self.retire();
            self.collect();
            const d = self.store.drain(deadline_ns -| monotonicNs());
            self.cancelled_shutdown += d.unpublished;
            self.refresh();
            return d.unpublished;
        }

        // -- restores ---------------------------------------------------------------------------------------------

        pub fn newRx(self: *Self, id: u64, key: Digest, at: u32, blocks: u32) !*Rx {
            try self.rxs.ensureUnusedCapacity(self.gpa, 1);
            const rx = try self.gpa.create(Rx);
            rx.* = .{ .id = id, .key = key, .at = at, .blocks = blocks };
            self.rxs.putAssumeCapacity(id, rx);
            return rx;
        }

        /// Ready blocks before `upto` (null: all) copied into the destination, within `budget_ns`; never waits.
        pub fn rxStep(self: *Self, rx: *Rx, upto: ?u32, budget_ns: u64) void {
            const rid = rx.rid orelse return;
            const dst = rx.dst orelse return;
            const stop = @min(upto orelse rx.blocks, rx.blocks);
            const deadline = monotonicNs() +| budget_ns;
            while (rx.why == .none and rx.copied < stop) {
                var one: [1]spill.ReadyBlock = undefined;
                const got = self.store.readReady(rid, stop, &one);
                if (got.n == 0) {
                    if (got.status == .failed) rx.fail(readWhy(self.store.readFailure(rid).err));
                    return;
                }
                const b = one[0];
                const ev = self.slotEvent(b.slot);
                self.copies.ensureUnusedCapacity(self.gpa, 1) catch {
                    self.store.readRelease(b.slot);
                    return rx.fail(.unpack);
                };
                _ = self.pk.unpackBlocks(dst, b.index, 1, b.data, self.events[ev]) catch |e| {
                    self.dev.wait(self.events[ev]);
                    self.store.readRelease(b.slot);
                    return rx.fail(if (e == error.StaleExtent) .reserve else .unpack);
                };
                self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .read = .{ .rx = rx.id, .slot = b.slot } } });
                rx.inflight += 1;
                rx.copied = b.index + 1;
                if (monotonicNs() >= deadline) return;
            }
        }

        /// With every block copied: the small state, once read and checked, into the destination's Snap buffer.
        pub fn rxSmall(self: *Self, rx: *Rx) void {
            if (rx.why != .none or rx.small != .none or rx.copied < rx.blocks) return;
            const rid = rx.rid orelse return;
            const view = self.store.readSmallView(rid) orelse {
                const f = self.store.readFailure(rid);
                if (f.err != null) rx.fail(readWhy(f.err));
                return;
            };
            const l = self.layout.small_bytes;
            if (view.sections.len != 1 or !std.mem.eql(u8, view.sections[0].name.slice(), section_name) or view.sections[0].offset != 0 or view.sections[0].len != l or view.bytes.len < l)
                return rx.fail(.unpack);
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return rx.fail(.unpack);
            const ev = self.addrEvent(.restore, view.bytes);
            _ = self.pk.unpackSnap(view.bytes[0..l], rx.snap, self.events[ev]) catch {
                self.dev.wait(self.events[ev]);
                return rx.fail(.unpack);
            };
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .small = .{ .rx = rx.id } } });
            rx.small = .pending;
        }

        /// Every block and the small state installed (their events completed).
        pub fn rxReady(rx: *const Rx) bool {
            return rx.why == .none and rx.copied == rx.blocks and rx.inflight == 0 and rx.small == .done;
        }

        /// Nothing moved the destination since it was reserved (Generations.valid of every unpack).
        pub fn rxValid(self: *const Self, rx: *const Rx) bool {
            const dst = rx.dst orelse return false;
            return self.gens.current(dst);
        }

        /// Every copy of `rx` on the device waited for and handed on.
        pub fn rxQuiesce(self: *Self, rx: *const Rx) void {
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                const mine = switch (cp.what) {
                    .read => |x| x.rx == rx.id,
                    .small => |x| x.rx == rx.id,
                    else => false,
                };
                if (mine) {
                    self.dev.wait(self.events[cp.ev]);
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                } else i += 1;
            }
        }

        /// Every copy of `rx` on the device completed and handed on by `deadline_ns` (monotonic), polling its events
        /// (never an unbounded device wait); false when one is still running at the deadline.
        pub fn rxSettle(self: *Self, rx: *const Rx, deadline_ns: u64) bool {
            while (true) {
                _ = self.retire();
                const left = for (self.copies.items) |cp| {
                    const mine = switch (cp.what) {
                        .read => |x| x.rx == rx.id,
                        .small => |x| x.rx == rx.id,
                        else => false,
                    };
                    if (mine) break true;
                } else false;
                if (!left) return true;
                if (monotonicNs() >= deadline_ns) return false;
                std.Io.sleep(self.io, .fromMicroseconds(200), .awake) catch {};
            }
        }

        /// A restore ends without publishing: its copies waited for, the point forgotten on this rank when
        /// `forget` (with this rank's own damaged/missing block lists), its reads ended, its destination cleared to
        /// fresh-pool bytes (scale sentinel rows) unless the rows already went to someone else.
        pub fn rxEnd(self: *Self, rx: *Rx, outcome: spill.Store.Outcome, forget: bool) void {
            self.rxQuiesce(rx);
            if (rx.rid) |rid| {
                if (forget) {
                    const f = self.store.readFailure(rid);
                    self.store.forget(rx.key, f.bad, f.lost);
                }
                _ = self.store.endRead(rid, outcome) catch {};
            } else if (forget) self.store.forget(rx.key, &.{}, &.{});
            if (rx.dst) |dst| {
                const ev = self.events[self.clearEvent()];
                if (self.pk.clearExtent(dst, ev)) |_| self.dev.wait(ev) else |_| {}
            }
            self.rxDrop(rx);
        }

        /// A committed restore's lineage (gpa-owned): its reads ended as committed. Its destination stays.
        pub fn rxCommit(self: *Self, rx: *Rx) !spill.ReadDone {
            const rid = rx.rid orelse return error.Incomplete;
            const done_ = try self.store.endRead(rid, .commit) orelse return error.Incomplete;
            return done_;
        }

        pub fn rxDrop(self: *Self, rx: *Rx) void {
            _ = self.rxs.remove(rx.id);
            self.gpa.destroy(rx);
        }

        // -- counters ---------------------------------------------------------------------------------------------

        /// The cached counters brought up to date (from service and flush: counts() never locks the store).
        pub fn refresh(self: *Self) void {
            const c = self.store.counts();
            self.cached = .{
                .points = c.points,
                .blocks = c.blocks,
                .stored_bytes = c.stored_bytes,
                .cap_bytes = self.store.cfg.cap_bytes,
                .disk_free_bytes = c.disk_free_bytes,
                .staging_bytes = c.staging_bytes,
                .staging_peak = c.staging_peak_bytes,
                .writes_pending = c.pending_jobs,
                .writes_queued = c.queued,
                .writes_committed = c.committed,
                .writes_cancelled = c.cancelled,
                .writes_cancelled_guard = c.cancelled_guard + c.cancelled_source,
                .writes_cancelled_staging = c.no_staging + c.backlog_refused,
                .writes_cancelled_shutdown = self.cancelled_shutdown,
                .writes_failed = c.write_failed,
                .forced_staging = c.guard_staged,
                .no_space = c.no_space,
                .dedup_blocks = c.dedup_blocks,
                .dedup_bytes = c.dedup_bytes,
                .written_bytes = c.written_bytes + c.written_small_bytes,
                .crc_errors = c.crc_errors,
                .size_errors = c.size_errors + c.missing_errors,
                .reads = c.loaded,
                .read_bytes = c.loaded_bytes,
                .read_failed = c.load_failed,
                .evicted_turns = c.pruned_turns + c.trimmed_turns,
                .evicted_forks = c.pruned_forks,
                .evicted_conversations = c.pruned_conversations,
                .evicted_shared = c.pruned_shared,
                .quota_trims = c.quota_trims,
                .recovered_points = c.recovered_points,
                .dropped_temporaries = c.temps_removed + c.recovered_dropped,
            };
        }
    };
}

// -- notes: rank 0's write-at-keep and metadata, in the ordered stream (lanes Kind 11, not answered) ----------------

pub const NoteOp = enum(u8) { keep = 1, share = 2, touch = 3, forget = 4 };

/// A keep note: what a follower needs to write its own copy of the point rank 0 just kept.
pub const KeepNote = struct {
    slot: u32,
    snap: u64,
    base: u64,
    at: u32,
    key: Digest,
    group: u64,
    shared: bool,
    lineage: format.Lineage, // (decoded: forks in the caller's buffer)
    origin: ?format.Origin,
};

const NW = struct {
    buf: *std.ArrayList(u8),
    gpa: Allocator,

    fn bytes(w: NW, b: []const u8) !void {
        try w.buf.appendSlice(w.gpa, b);
    }

    fn int(w: NW, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.bytes(&b);
    }
};

const NR = struct {
    b: []const u8,
    at: usize = 0,

    fn take(r: *NR, n: usize) ![]const u8 {
        if (n > r.b.len - r.at) return error.ShortNote;
        defer r.at += n;
        return r.b[r.at..][0..n];
    }

    fn int(r: *NR, comptime T: type) !T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn flag(r: *NR) !bool {
        return switch (try r.int(u8)) {
            0 => false,
            1 => true,
            else => error.BadNote,
        };
    }
};

pub fn encodeKeep(gpa: Allocator, out: *std.ArrayList(u8), n: KeepNote) !void {
    out.clearRetainingCapacity();
    const w: NW = .{ .buf = out, .gpa = gpa };
    try w.int(u8, @intFromEnum(NoteOp.keep));
    try w.int(u32, n.slot);
    try w.int(u64, n.snap);
    try w.int(u64, n.base);
    try w.int(u32, n.at);
    try w.bytes(&n.key);
    try w.int(u64, n.group);
    try w.int(u8, @intFromBool(n.shared));
    try w.bytes(&n.lineage.root);
    try w.int(u8, @intFromBool(n.lineage.part != null));
    try w.int(u32, n.lineage.part orelse 0);
    try w.int(u32, @intCast(n.lineage.forks.len));
    for (n.lineage.forks) |f| {
        try w.bytes(&f.name);
        try w.int(u32, f.at);
    }
    try w.int(u8, @intFromBool(n.origin != null));
    const o = n.origin orelse format.Origin{ .key = @splat(0), .at = 0 };
    try w.bytes(&o.key);
    try w.int(u32, o.at);
}

pub fn encodeKey(gpa: Allocator, out: *std.ArrayList(u8), op: NoteOp, key: Digest) !void {
    out.clearRetainingCapacity();
    const w: NW = .{ .buf = out, .gpa = gpa };
    try w.int(u8, @intFromEnum(op));
    try w.bytes(&key);
}

pub const Note = union(NoteOp) { keep: KeepNote, share: Digest, touch: Digest, forget: Digest };

/// A note's fields, bounded (`forks` holds the lineage's forks: more than it holds is refused), nothing after them.
pub fn decodeNote(bytes: []const u8, forks: []format.Fork) !Note {
    var r: NR = .{ .b = bytes };
    const op = std.enums.fromInt(NoteOp, try r.int(u8)) orelse return error.BadNote;
    const note: Note = switch (op) {
        .keep => blk: {
            var n: KeepNote = undefined;
            n.slot = try r.int(u32);
            n.snap = try r.int(u64);
            n.base = try r.int(u64);
            n.at = try r.int(u32);
            @memcpy(&n.key, try r.take(32));
            n.group = try r.int(u64);
            n.shared = try r.flag();
            var root: [16]u8 = undefined;
            @memcpy(&root, try r.take(16));
            const has_part = try r.flag();
            const part = try r.int(u32);
            const nf = try r.int(u32);
            if (nf > forks.len) return error.BadNote;
            for (forks[0..nf]) |*f| {
                @memcpy(&f.name, try r.take(16));
                f.at = try r.int(u32);
            }
            n.lineage = .{ .root = root, .part = if (has_part) part else null, .forks = forks[0..nf] };
            const has_origin = try r.flag();
            var ok: Digest = undefined;
            @memcpy(&ok, try r.take(32));
            const oat = try r.int(u32);
            n.origin = if (has_origin) .{ .key = ok, .at = oat } else null;
            if (n.at == 0 or n.snap == 0) return error.BadNote;
            break :blk .{ .keep = n };
        },
        inline .share, .touch, .forget => |t| blk: {
            var k: Digest = undefined;
            @memcpy(&k, try r.take(32));
            break :blk @unionInit(Note, @tagName(t), k);
        },
    };
    if (r.at != bytes.len) return error.BadNote;
    return note;
}

// -- rank 0 ---------------------------------------------------------------------------------------------------------

/// A restore's destination as the family made it: a fresh kept state (lanes.zig: a Snap with its extent reserved, in
/// `snaps` but not usable until installed) and its device buffer.
pub const Dest = struct { saved: *anyopaque, id: u64, base: usize, buf: u64 };
/// A kept state as the tier needs it.
pub const SnapInfo = struct { id: u64, base: usize, at: usize, buf: u64, slot: u32 };

/// Rank 0's family side (lanes.zig's Lanes).
pub const Family = struct {
    ptr: *anyopaque,
    /// A destination for a restore of `at` positions (placement, its buffer); an error: no room now. `room`: the
    /// positions the request waiting on it needs from the same base (its prompt; >= at), a hint: the family holds
    /// that much free past the rows while room allows, so the resume runs in place instead of copying the extent.
    reserve: *const fn (ptr: *anyopaque, at: usize, room: usize) anyerror!Dest,
    /// The restored state is installed on every rank: usable from now (Lanes.usable).
    install: *const fn (ptr: *anyopaque, saved: *anyopaque) void,
    /// A restore that did not commit: its state removed and freed.
    discard: *const fn (ptr: *anyopaque, saved: *anyopaque) void,
    info: *const fn (ptr: *anyopaque, saved: *anyopaque) SnapInfo,
};

/// How rank 0's frames reach the followers (lanes.zig: a frame of `kind` with `sub` as its payload, to every
/// follower in the ordered stream) and their links for replies (Model.followers()).
pub const Wire = struct {
    ctx: *anyopaque,
    frame: *const fn (ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void,
    links: []const ?link.Link,
};

pub fn Leader(comptime Dev: type) type {
    return struct {
        const Self = @This();
        pub const S = Side(Dev);

        const Job = struct { rx: *Rx, txn: u64, point: pc.Point, saved: *anyopaque, sent: u32 = 0 };
        const PointRef = struct { key: Digest, at: u32, room: u32 = 0 }; // room: the last lookup's prompt length
        const Noted = struct { key: Digest };

        gpa: Allocator,
        side: *S,
        fam: Family,
        wire: ?Wire,
        lead: ?*rank_wire.Leader,
        min_tokens: u32,
        max_restores: u32, // restores in flight at most (Settings.restores, the store's reads; <= rank_wire.max_restores)
        flush_ms: u32,
        jobs: std.AutoHashMapUnmanaged(u64, *Job) = .empty,
        next_job: u64 = 1,
        points: std.AutoHashMapUnmanaged(u64, PointRef) = .empty,
        noted: std.AutoHashMapUnmanaged(u64, Noted) = .empty, // kept states the followers write too (by Snap.id)
        fwd: std.AutoHashMapUnmanaged(u64, [16]u8) = .empty, // lineage names: the cache's u64s to the store's
        rev: std.AutoHashMapUnmanaged([16]u8, u64) = .empty,
        next_name: u64 = 0xC000_0000_0000_0000,
        boot: [16]u8,
        lin: std.ArrayList(pc.Fork) = .empty, // a ready restore's lineage (valid during its poll)
        note_buf: std.ArrayList(u8) = .empty,
        presence_refused: u64 = 0,
        misses: u64 = 0,
        peers: [rank_wire.max_world - 1]pc.TierCounts = @splat(.{}), // the followers' counters (stats, at most 1/s)
        peers_n: usize = 0,
        stats_at: u64 = 0,
        broken: ?anyerror = null,
        on_broken: ?*const fn (ctx: ?*anyopaque, err: anyerror) void = null,
        broken_ctx: ?*anyopaque = null,

        const vt: pc.Tier.VTable = .{
            .lookup = lookupFn,
            .begin = beginFn,
            .poll = pollFn,
            .cancel = cancelFn,
            .keep = keepFn,
            .release = releaseFn,
            .share = shareFn,
            .touch = touchFn,
            .forget = forgetFn,
            .pump = pumpFn,
            .busy = busyFn,
            .flush = flushFn,
            .counts = countsFn,
        };

        /// `side` is moved in. `wire` and `lead` (heap, its agreement made: rank_wire.Leader.agree; every has carries
        /// its `compat()`, the agreed pair) are null with one rank. `boot` salts this process's lineage names.
        pub fn create(gpa: Allocator, side: *S, fam: Family, wire: ?Wire, lead: ?*rank_wire.Leader, s: Settings, boot: [16]u8) !*Self {
            const self = try gpa.create(Self);
            self.* = .{ .gpa = gpa, .side = side, .fam = fam, .wire = wire, .lead = lead, .min_tokens = s.min_tokens, .max_restores = @intCast(@max(1, @min(s.restores, rank_wire.max_restores))), .flush_ms = s.flush_ms, .boot = boot };
            return self;
        }

        /// Cancels what is still restoring, closes the store; `lead` is freed too.
        pub fn destroy(self: *Self) void {
            var it = self.jobs.valueIterator();
            while (it.next()) |j| self.endJob(j.*, .cancelled, false, true);
            self.jobs.deinit(self.gpa);
            self.side.destroy();
            self.points.deinit(self.gpa);
            self.noted.deinit(self.gpa);
            self.fwd.deinit(self.gpa);
            self.rev.deinit(self.gpa);
            self.lin.deinit(self.gpa);
            self.note_buf.deinit(self.gpa);
            if (self.lead) |l| self.gpa.destroy(l);
            self.gpa.destroy(self);
        }

        pub fn tier(self: *Self) pc.Tier {
            return .{ .ptr = self, .vtable = &vt };
        }

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        // -- the wire ---------------------------------------------------------------------------------------------

        fn sinkSend(ctx: *anyopaque, sub: []const u8) anyerror!void {
            const self = of(ctx);
            const w = self.wire.?;
            return w.frame(w.ctx, rank_wire.lanes_kind, sub);
        }

        fn sink(self: *Self) rank_wire.Sink {
            return .{ .ctx = self, .send = sinkSend };
        }

        /// The pair cannot go on (a rank could not answer, the link went down): nothing more is sent or restored.
        fn breakPair(self: *Self, err: anyerror) anyerror {
            self.markBroken(err);
            return err;
        }

        fn markBroken(self: *Self, err: anyerror) void {
            if (self.broken == null) {
                self.broken = err;
                var buf: [160]u8 = undefined;
                const why = if (self.lead) |l| (if (l.broken) |b| rank_wire.describeBroken(b, &buf) else @errorName(err)) else @errorName(err);
                std.log.err("dsv41 spill: the ranks' agreement broke ({s}); the pair must restart", .{why});
                if (self.on_broken) |f| f(self.broken_ctx, err);
            }
        }

        /// A command nobody answers. Any refusal ends the pair (rank_wire: rank 0 may already have acted on it).
        fn post(self: *Self, txn: u64, msg: rank_wire.Msg) !rank_wire.Head {
            if (self.broken) |e| return e;
            return self.lead.?.post(self.sink(), txn, msg) catch |e| self.breakPair(e);
        }

        /// An ask. One refused before it was sent (WrongCompat, ordering, BadWorld: nothing crossed, rank 0 did not
        /// act on it) returns its error and the pair goes on; every other failure ends the pair.
        fn ask(self: *Self, txn: u64, msg: rank_wire.Msg, replies: []rank_wire.Frame) !rank_wire.Head {
            if (self.broken) |e| return e;
            const w = self.wire.?;
            const l = self.lead.?;
            return l.ask(self.sink(), w.links, txn, msg, replies[0..w.links.len]) catch |e| {
                if (l.broken != null) self.markBroken(e);
                return e;
            };
        }

        fn note(self: *Self) void {
            if (self.broken != null) return;
            const w = self.wire orelse return;
            w.frame(w.ctx, note_kind, self.note_buf.items) catch |e| self.markBroken(e);
        }

        fn noteKey(self: *Self, op: NoteOp, key: Digest) void {
            if (self.wire == null) return;
            encodeKey(self.gpa, &self.note_buf, op, key) catch return;
            self.note();
        }

        /// Write cancellations the followers should hear (their own copies of a point rank 0 lost).
        fn tellLost(self: *Self) void {
            defer self.side.lost.clearRetainingCapacity();
            if (self.wire == null) return;
            for (self.side.lost.items) |l| {
                if (!self.noted.contains(l.snap)) continue;
                _ = self.post(0, .{ .write_cancelled = .{ .point = l.key, .kept = l.snap, .generation = 0, .why = l.why } }) catch return;
            }
        }

        // -- lineage names ----------------------------------------------------------------------------------------

        pub fn name16(self: *const Self, n: u64) [16]u8 {
            if (self.fwd.get(n)) |x| return x;
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update("tf-dsv41-lineage");
            h.update(&self.boot);
            h.update(std.mem.asBytes(&n));
            return h.finalResult()[0..16].*;
        }

        fn nameOf(self: *Self, n: [16]u8) !u64 {
            if (self.rev.get(n)) |x| return x;
            try self.fwd.ensureUnusedCapacity(self.gpa, 1);
            try self.rev.ensureUnusedCapacity(self.gpa, 1);
            const id = self.next_name;
            self.next_name += 1;
            self.fwd.putAssumeCapacity(id, n);
            self.rev.putAssumeCapacity(n, id);
            return id;
        }

        /// The store's lineage of a cache lineage (forks in `out`).
        fn toLineage(self: *Self, forks: []const pc.Fork, out: *std.ArrayList(format.Fork)) !format.Lineage {
            if (forks.len == 0) {
                self.next_name += 1;
                return .{ .root = self.name16(self.next_name) };
            }
            for (forks[1..]) |f| try out.append(self.gpa, .{ .name = self.name16(f.name), .at = f.at });
            return .{ .root = self.name16(forks[0].name), .part = null, .forks = out.items };
        }

        fn fromLineage(self: *Self, l: format.Lineage) ![]const pc.Fork {
            self.lin.clearRetainingCapacity();
            try self.lin.append(self.gpa, .{ .name = try self.nameOf(l.root), .at = 0 });
            for (l.forks) |f| try self.lin.append(self.gpa, .{ .name = try self.nameOf(f.name), .at = f.at });
            return self.lin.items;
        }

        // -- lanes.zig's guards -----------------------------------------------------------------------------------

        /// Kept state `snap`'s rows from token `from` on are about to change (placeKept victim, staleOver, carry):
        /// stage or cancel its write here and on every follower (write_staged), before the change.
        pub fn guardSnap(self: *Self, snap: u64, from: u32) void {
            self.side.guard(snap, from);
            if (self.wire == null) return;
            const n = self.noted.get(snap) orelse return;
            _ = self.post(0, .{ .write_staged = .{ .point = n.key, .kept = snap, .generation = 0, .upto = from / self.side.bt } }) catch {};
        }

        /// Kept state `snap` is about to be freed (snapDropFn, the store's release): guarded from 0, its small-state
        /// capture waited for; the followers do the same at their snap_drop.
        pub fn dropSnap(self: *Self, snap: u64) void {
            self.guardSnap(snap, 0);
            self.side.dropSnap(snap);
            _ = self.noted.remove(snap);
        }

        pub fn bump(self: *Self, lo: usize, hi: usize) void {
            self.side.bump(lo, hi);
        }

        pub fn bumpSlot(self: *Self, slot: usize) void {
            self.side.bumpSlot(slot);
        }

        // -- the Tier ---------------------------------------------------------------------------------------------

        fn lookupFn(ptr: *anyopaque, want: *const pc.Want) ?pc.Point {
            const self = of(ptr);
            if (self.broken != null) return null;
            const hit = self.side.store.lookup(.{ .keyed = want.key, .longer_than = want.beyond, .max_at = want.max_at, .grid = want.grid }) orelse {
                self.misses += 1;
                return null;
            };
            const id = pointId(hit.key);
            self.points.put(self.gpa, id, .{ .key = hit.key, .at = hit.at, .room = @intCast(@min(want.key.len, std.math.maxInt(u32))) }) catch return null;
            return .{ .id = id, .at = hit.at, .tokens = hit.at + want.lookahead, .bytes = 0, .group = hit.group, .shared = hit.shared };
        }

        fn keepFn(ptr: *anyopaque, kept: *const pc.Kept) ?u64 {
            const self = of(ptr);
            if (self.broken != null or kept.images) return null; // (pictures need full payload digests: not written)
            const bt = self.side.bt;
            const at = kept.at;
            if (at == 0 or at % bt != 0 or at < self.min_tokens or kept.key.len < at) return null;
            const info = self.fam.info(self.fam.ptr, kept.saved);
            if (info.at != at) return null;
            var forks: std.ArrayList(format.Fork) = .empty;
            defer forks.deinit(self.gpa);
            const lin = self.toLineage(kept.lineage, &forks) catch return null;
            const origin: ?format.Origin = if (kept.parent) |p| (if (self.points.get(p)) |r| .{ .key = r.key, .at = kept.fork } else null) else null;
            const keyed = kept.key[0..at];
            const key = format.pointKey(keyed, bt, &.{});
            const r = self.side.keepPoint(.{ .keyed = keyed, .key = key, .at = at, .base = info.base, .snap = info.id, .buf = info.buf, .lineage = lin, .origin = origin, .shared = kept.shared, .group = kept.group });
            if (r == .refused) return null;
            const id = pointId(key);
            const gop = self.points.getOrPut(self.gpa, id) catch return null;
            if (!gop.found_existing) gop.value_ptr.room = 0;
            gop.value_ptr.key = key;
            gop.value_ptr.at = at;
            // the note for a dedup (.stored) too: a follower whose own store pruned or never wrote the point writes it
            // now (its store.keep dedups when it holds it), so a point rank 0 holds is repaired on the pair
            if (self.wire != null) {
                self.noted.put(self.gpa, info.id, .{ .key = key }) catch {};
                encodeKeep(self.gpa, &self.note_buf, .{ .slot = info.slot, .snap = info.id, .base = info.base, .at = at, .key = key, .group = kept.group, .shared = kept.shared, .lineage = lin, .origin = origin }) catch return id;
                self.note();
            }
            return id;
        }

        fn releaseFn(ptr: *anyopaque, point: u64, saved: pc.Saved) void {
            _ = point;
            const self = of(ptr);
            self.dropSnap(self.fam.info(self.fam.ptr, saved).id);
        }

        fn shareFn(ptr: *anyopaque, point: u64) void {
            const self = of(ptr);
            const r = self.points.get(point) orelse return;
            self.side.store.markShared(r.key);
            self.noteKey(.share, r.key);
        }

        fn touchFn(ptr: *anyopaque, point: u64) void {
            const self = of(ptr);
            const r = self.points.get(point) orelse return;
            self.side.store.touch(r.key);
            self.noteKey(.touch, r.key);
        }

        fn forgetFn(ptr: *anyopaque, point: u64) void {
            const self = of(ptr);
            const r = self.points.fetchRemove(point) orelse return;
            self.side.store.forget(r.value.key, &.{}, &.{});
            self.noteKey(.forget, r.value.key);
        }

        fn pumpFn(ptr: *anyopaque, budget_ns: u64) void {
            const self = of(ptr);
            _ = self.side.service(budget_ns, pump_blocks);
            self.tellLost();
            self.askStats();
        }

        /// The followers' counters for counts(), asked between rounds at most once a second (rank_wire stats).
        pub fn askStats(self: *Self) void {
            const w = self.wire orelse return;
            if (self.broken != null) return;
            const now = monotonicNs();
            if (self.stats_at != 0 and now -| self.stats_at < std.time.ns_per_s) return;
            self.stats_at = now;
            var replies: [rank_wire.max_world - 1]rank_wire.Frame = undefined;
            _ = self.ask(0, .{ .stats = .{ .set = .tier } }, &replies) catch return;
            for (replies[0..w.links.len], 0..) |f, i| {
                const r = f.msg.stats_reply;
                self.peers[i] = if (r.set == .tier) valuesCounts(r.n, &r.values) else .{};
            }
            self.peers_n = w.links.len;
        }

        fn busyFn(ptr: *anyopaque) bool {
            return of(ptr).side.busy();
        }

        fn flushFn(ptr: *anyopaque, budget_ns: u64) u64 {
            const self = of(ptr);
            var it = self.jobs.valueIterator();
            while (it.next()) |j| self.endJob(j.*, .shutdown, false, true);
            self.jobs.clearRetainingCapacity();
            const left = self.side.flush(monotonicNs() +| budget_ns);
            self.tellLost();
            return left;
        }

        fn countsFn(ptr: *anyopaque, out: []pc.TierCounts) usize {
            const self = of(ptr);
            if (out.len == 0) return 0;
            var c = self.side.cached;
            c.presence_refused = self.presence_refused;
            c.compatible_misses = self.misses;
            out[0] = c;
            const n = @min(self.peers_n, out.len - 1);
            @memcpy(out[1..][0..n], self.peers[0..n]); // (as their last stats replies carried them)
            return 1 + n;
        }

        // -- restores ---------------------------------------------------------------------------------------------

        fn beginFn(ptr: *anyopaque, point: *const pc.Point) anyerror!u64 {
            const self = of(ptr);
            if (self.broken) |e| return e;
            const side = self.side;
            // refused before anything changes here or on a follower: the store's reads (Settings.restores) and
            // rank_wire's open restores bound how many run at once, and a placement must not stale kept states for
            // a restore that cannot start
            if (self.jobs.count() >= self.max_restores) return error.TooManyRestores;
            const ref = self.points.get(point.id) orelse return error.UnknownPoint;
            if (point.at != ref.at or point.at % side.bt != 0) return error.BadPoint;
            const blocks: u32 = point.at / side.bt;
            if (!side.store.presence(ref.key)) {
                self.presence_refused += 1;
                return error.SpillNotOnEveryRank;
            }
            try self.jobs.ensureUnusedCapacity(self.gpa, 1);
            const id = self.next_job;
            const rx = try side.newRx(id, ref.key, point.at, blocks);
            // this rank's reads first (TooMany / NoSlot / Missing: nothing placed, nothing sent); they only need
            // staging slots, the destination is first written by rxStep
            rx.rid = side.store.beginRead(ref.key) catch |e| {
                side.rxDrop(rx);
                return e;
            };
            var txn: u64 = 0;
            if (self.wire) |w| {
                var replies: [rank_wire.max_world - 1]rank_wire.Frame = undefined;
                const head = self.ask(0, .{ .has = .{ .point = ref.key, .compat = self.lead.?.compat(), .at = point.at, .blocks = blocks, .layout = .replicated, .part = 0 } }, &replies) catch |e| {
                    side.rxEnd(rx, .cancel, false);
                    return e;
                };
                txn = head.txn;
                var refused: ?u16 = null;
                var forget = false;
                for (replies[0..w.links.len], 0..) |f, i| {
                    const ans = f.msg.has_reply.answer;
                    if (ans == .present) continue;
                    if (refused == null) refused = @intCast(i + 1);
                    // a follower that lacks the point (pruned by its own cap/floor/turns, a write it cancelled) or
                    // holds it damaged: rank 0's copy can never restore, so every rank drops it and the next keep
                    // writes it again on both; one still writing it (incomplete) or with spill off keeps it
                    if (ans == .missing or ans == .damaged or ans == .incompatible) forget = true;
                }
                if (refused) |rank| {
                    self.presence_refused += 1;
                    side.rxEnd(rx, .cancel, forget);
                    if (forget) _ = self.points.remove(point.id);
                    _ = try self.post(txn, .{ .abort = .{ .why = .presence, .rank = rank, .forget = forget } });
                    return error.SpillNotOnEveryRank;
                }
            }
            const room = @max(@as(usize, point.at), ref.room);
            const dest = self.fam.reserve(self.fam.ptr, point.at, room) catch |e| {
                side.rxEnd(rx, .cancel, false);
                return self.refuse(txn, .reserve, e);
            };
            rx.snap = .{ .id = dest.id, .buf = dest.buf };
            rx.dst = side.gens.reserve(dest.base, dest.base + point.at) catch |e| {
                side.rxEnd(rx, .cancel, false);
                self.fam.discard(self.fam.ptr, dest.saved);
                return self.refuse(txn, .reserve, e);
            };
            const job = self.gpa.create(Job) catch |e| {
                side.rxEnd(rx, .cancel, false);
                self.fam.discard(self.fam.ptr, dest.saved);
                return self.refuse(txn, .reserve, e);
            };
            job.* = .{ .rx = rx, .txn = txn, .point = point.*, .saved = dest.saved };
            if (self.wire != null) {
                const sent: ?anyerror = blk: {
                    _ = self.post(txn, .{ .reserve = .{ .point = ref.key, .slot = 0, .base = dest.base, .size = @max(room, point.at), .at = point.at, .blocks = blocks, .generation = dest.id, .part = 0 } }) catch |e| break :blk e;
                    _ = self.post(txn, .{ .start = .{ .point = ref.key, .blocks = blocks } }) catch |e| break :blk e;
                    break :blk null;
                };
                if (sent) |e| {
                    self.endJob(job, .read, false, false);
                    return e;
                }
            }
            self.next_job += 1;
            self.jobs.putAssumeCapacity(id, job);
            return id;
        }

        /// A restore refused before it began on this rank: the followers' transaction aborted.
        fn refuse(self: *Self, txn: u64, why: rank_wire.Failure, err: anyerror) anyerror {
            if (txn != 0) _ = self.post(txn, .{ .abort = .{ .why = why, .rank = 0, .forget = false } }) catch {};
            return err;
        }

        /// A job ends without publishing: here (its copies waited, its destination cleared and discarded) and, when
        /// `send`, on every follower (abort with `forget` when the point is damaged on some rank).
        fn endJob(self: *Self, job: *Job, why: rank_wire.Failure, forget: bool, send: bool) void {
            self.side.rxEnd(job.rx, if (why == .cancelled or why == .shutdown) .cancel else .abort, forget);
            if (send and job.txn != 0) _ = self.post(job.txn, .{ .abort = .{ .why = why, .rank = rank_wire.no_rank, .forget = forget } }) catch {};
            self.fam.discard(self.fam.ptr, job.saved);
            if (forget) _ = self.points.remove(job.point.id);
            self.gpa.destroy(job);
        }

        fn failJob(self: *Self, id: u64, job: *Job, why: rank_wire.Failure, rank: u16) pc.Poll {
            _ = self.jobs.remove(id);
            const forget = damage(why);
            self.side.rxEnd(job.rx, .abort, forget);
            if (job.txn != 0) _ = self.post(job.txn, .{ .abort = .{ .why = why, .rank = rank, .forget = forget } }) catch {};
            self.fam.discard(self.fam.ptr, job.saved);
            if (forget) _ = self.points.remove(job.point.id);
            self.gpa.destroy(job);
            return .{ .failed = failureError(why) };
        }

        fn pollFn(ptr: *anyopaque, id: u64, budget_ns: u64) pc.Poll {
            const self = of(ptr);
            const side = self.side;
            const job = self.jobs.get(id) orelse return .{ .failed = error.UnknownJob };
            if (self.broken) |e| {
                _ = self.jobs.remove(id);
                self.endJob(job, .cancelled, false, false);
                return .{ .failed = e };
            }
            const rx = job.rx;
            _ = side.retire();
            side.rxStep(rx, null, budget_ns);
            side.rxSmall(rx);
            if (rx.why != .none) return self.failJob(id, job, rx.why, 0);
            if (self.wire != null and rx.copied > job.sent) {
                _ = self.post(job.txn, .{ .progress = .{ .upto = rx.copied } }) catch return self.failJob(id, job, .read, rank_wire.no_rank);
                job.sent = rx.copied;
            }
            if (!S.rxReady(rx)) return .{ .pending = .{ .done = rx.copied, .total = rx.blocks } };
            if (!side.rxValid(rx)) return self.failJob(id, job, .reserve, 0);
            if (self.wire != null) {
                var replies: [rank_wire.max_world - 1]rank_wire.Frame = undefined;
                _ = self.ask(job.txn, .{ .validate = .{ .upto = rx.blocks, .shared = job.point.shared } }, &replies) catch return self.failJob(id, job, .timeout, rank_wire.no_rank);
                if (rank_wire.firstFailure(replies[0..self.wire.?.links.len])) |f| return self.failJob(id, job, f.why, f.rank);
            }
            const commit: rank_wire.Msg = .{ .commit = .{ .point = rx.key, .at = rx.at, .generation = rx.snap.id, .shared = job.point.shared } };
            if (self.lead) |l| l.admissible(job.txn, commit) catch return self.failJob(id, job, .read, 0); // (before rank 0 commits)
            // rank 0's own end of the read: a failure here (out of memory for the lineage, an incomplete or failed
            // read the checks above missed) is not damage to the stored point, so nobody forgets it
            var done_ = side.rxCommit(rx) catch return self.failJob(id, job, .read, 0);
            defer done_.lineage.deinit(self.gpa);
            if (self.wire != null) {
                _ = self.post(job.txn, .{ .commit = .{ .point = rx.key, .at = rx.at, .generation = rx.snap.id, .shared = done_.shared } }) catch |e| {
                    _ = self.jobs.remove(id);
                    side.rxDrop(rx);
                    self.fam.discard(self.fam.ptr, job.saved);
                    self.gpa.destroy(job);
                    return .{ .failed = e };
                };
            }
            const lin = self.fromLineage(done_.lineage) catch &.{};
            _ = self.jobs.remove(id);
            side.rxDrop(rx);
            self.fam.install(self.fam.ptr, job.saved);
            const saved = job.saved;
            self.gpa.destroy(job);
            return .{ .ready = .{ .saved = saved, .bytes = 0, .lineage = lin } };
        }

        fn cancelFn(ptr: *anyopaque, id: u64) void {
            const self = of(ptr);
            const kv = self.jobs.fetchRemove(id) orelse return;
            self.endJob(kv.value, .cancelled, false, true);
        }
    };
}

// -- followers ------------------------------------------------------------------------------------------------------

/// A follower's family side (lanes.follow over its kept states).
pub const FollowerFamily = struct {
    ptr: *anyopaque,
    /// A device buffer for kept state `id` (a restore's destination; not visible to snap_restore until installed).
    reserve: *const fn (ptr: *anyopaque, id: u64) anyerror!u64,
    install: *const fn (ptr: *anyopaque, id: u64) anyerror!void,
    discard: *const fn (ptr: *anyopaque, id: u64) void,
    /// Kept state `id`'s buffer (null: none).
    buf: *const fn (ptr: *anyopaque, id: u64) ?u64,
    /// The ids of a slot's prompt as this rank holds them (Lanes.seqs; pictures' positions negative).
    ids: *const fn (ptr: *anyopaque, slot: u32) []const i32,
};

/// What a follower opens its side with once the ranks agreed (rank_wire's agreed): everything but the compat digest.
pub fn Opening(comptime Dev: type) type {
    return struct {
        dev: Dev,
        layout: state.StateLayout,
        gens: state.Generations,
        settings: Settings,
        max_tokens: u32,
        staging: []align(spill.slots.alignment) u8,
        events: []Dev.Event,
        hooks: spill.store.Hooks = .{},
    };
}

pub fn Follower(comptime Dev: type) type {
    return struct {
        const Self = @This();
        pub const S = Side(Dev);
        const Head = rank_wire.Head;
        const Fx = struct { rx: *Rx, snap: u64, allowed: u32 = 0, buf: bool = false };

        gpa: Allocator,
        io: std.Io,
        rank: u16,
        wf: rank_wire.Follower,
        fam: FollowerFamily,
        fam_live: bool = false, // fam set by attach (lanes.followWith) and not yet detached
        opening: ?Opening(Dev),
        side: ?*S = null,
        agreement: ?fingerprint.Agreement = null,
        restores: std.AutoHashMapUnmanaged(u64, Fx) = .empty, // by transaction
        asked: std.AutoHashMapUnmanaged(u64, Digest) = .empty, // has answered, not yet reserved (an abort's forget)
        forks: []format.Fork,
        keyed: std.ArrayList(u32) = .empty,
        skipped: u64 = 0, // keep notes this rank could not write (ids or state not as rank 0's)
        flush_ns: u64, // rank 0's done: the drain's bound (TF_SPILL_FLUSH_MS)

        /// `fam`: the kept states restores install into, or null until `attach` (lanes.followWith).
        pub fn create(gpa: Allocator, io: std.Io, rank: u16, fx: fingerprint.Exchange, fam: ?FollowerFamily, opening: Opening(Dev)) !*Self {
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .rank = rank, .wf = rank_wire.Follower.init(rank, fx), .fam = fam orelse undefined, .fam_live = fam != null, .opening = opening, .forks = try gpa.alloc(format.Fork, 4096), .flush_ns = @as(u64, opening.settings.flush_ms) * std.time.ns_per_ms };
            return self;
        }

        /// The family the restores install into (lanes.followWith's kept states), valid until `detach`.
        pub fn attach(self: *Self, fam: FollowerFamily) void {
            self.fam = fam;
            self.fam_live = true;
        }

        /// Every open restore ended here (its copies waited for, its read cancelled, its destination cleared and its
        /// buffer discarded) and the family let go: lanes.followWith calls it on every exit, while its kept states
        /// are still alive, so nothing later reaches into them.
        pub fn detach(self: *Self) void {
            self.endRestores();
            self.fam_live = false;
        }

        fn endRestores(self: *Self) void {
            var it = self.restores.valueIterator();
            while (it.next()) |fx| {
                if (self.side) |s| s.rxEnd(fx.rx, .cancel, false);
                if (fx.buf and self.fam_live) self.fam.discard(self.fam.ptr, fx.snap);
            }
            self.restores.clearRetainingCapacity();
            self.asked.clearRetainingCapacity();
        }

        /// At `done`: writes drained within the settings' flush bound, then everything closed.
        pub fn destroy(self: *Self) void {
            self.endRestores();
            self.restores.deinit(self.gpa);
            self.asked.deinit(self.gpa);
            if (self.side) |s| s.destroy() else if (self.opening) |*o| o.gens.deinit(self.gpa);
            self.keyed.deinit(self.gpa);
            self.gpa.free(self.forks);
            self.gpa.destroy(self);
        }

        /// Every copy still on the device waited for (before the buffers they read go).
        pub fn quiesce(self: *Self) void {
            const s = self.side orelse return;
            for (s.copies.items) |cp| s.dev.wait(s.events[cp.ev]);
            _ = s.retire();
        }

        /// A clean stop (rank 0's done): unpublished writes staged and drained within the flush bound.
        pub fn flush(self: *Self, budget_ns: u64) u64 {
            const s = self.side orelse return 0;
            return s.flush(monotonicNs() +| budget_ns);
        }

        /// A rank_wire frame from rank 0's stream (lanes Kind 10), answered on `l` when it asks.
        pub fn wireFrame(self: *Self, l: link.Link, payload: []const u8) !void {
            try rank_wire.serve(&self.wf, l, payload, self);
        }

        /// A spill note (lanes Kind 11).
        pub fn note(self: *Self, payload: []const u8) !void {
            const n = try decodeNote(payload, self.forks);
            const s = self.side orelse return error.SpillNotAgreed;
            switch (n) {
                .keep => |k| self.keepNote(s, k),
                .share => |k| s.store.markShared(k),
                .touch => |k| s.store.touch(k),
                .forget => |k| s.store.forget(k, &.{}, &.{}),
            }
        }

        fn keepNote(self: *Self, s: *S, k: KeepNote) void {
            if (!self.fam_live) return self.skip();
            const ids = self.fam.ids(self.fam.ptr, k.slot);
            const buf = self.fam.buf(self.fam.ptr, k.snap) orelse return self.skip();
            if (ids.len < k.at) return self.skip();
            self.keyed.resize(self.gpa, k.at) catch return self.skip();
            for (ids[0..k.at], self.keyed.items) |q, *o| {
                if (q < 0) return self.skip();
                o.* = @intCast(q);
            }
            // the point is rank 0's: this rank's ids must name the same one, or a restore could never agree on it
            if (!std.mem.eql(u8, &format.pointKey(self.keyed.items, s.bt, &.{}), &k.key)) return self.skip();
            _ = s.keepPoint(.{ .keyed = self.keyed.items, .key = k.key, .at = k.at, .base = @intCast(k.base), .snap = k.snap, .buf = buf, .lineage = k.lineage, .origin = k.origin, .shared = k.shared, .group = k.group });
        }

        fn skip(self: *Self) void {
            self.skipped += 1;
        }

        /// At every frame and while idle: copies handed on, restores copied as far as rank 0 allowed, writes staged.
        pub fn service(self: *Self, budget_ns: u64) void {
            const s = self.side orelse return;
            _ = s.retire();
            var it = self.restores.valueIterator();
            while (it.next()) |fx| s.rxStep(fx.rx, fx.allowed, budget_ns);
            _ = s.service(budget_ns, pump_blocks);
            s.lost.clearRetainingCapacity();
        }

        pub fn busy(self: *Self) bool {
            const s = self.side orelse return false;
            return s.busy() or self.restores.count() > 0;
        }

        pub fn bump(self: *Self, lo: usize, hi: usize) void {
            if (self.side) |s| s.bump(lo, hi);
        }

        pub fn bumpSlot(self: *Self, slot: usize) void {
            if (self.side) |s| s.bumpSlot(slot);
        }

        pub fn dropSnap(self: *Self, snap: u64) void {
            if (self.side) |s| s.dropSnap(snap);
        }

        // -- rank_wire.serve's handler ----------------------------------------------------------------------------

        pub fn agreed(self: *Self, a: fingerprint.Agreement) !void {
            const o = self.opening orelse return error.SpillAgreedTwice;
            const compat = format.compatDigest(a.storeCommon(self.rank), .replicated, @intCast(o.layout.geo.block_tokens), o.layout.block_bytes);
            self.side = try S.create(self.gpa, self.io, o.dev, o.layout, o.gens, o.settings, self.rank, compat, o.max_tokens, o.staging, o.events, o.hooks);
            self.opening = null;
            self.agreement = a;
        }

        /// serve already checked `m.compat` against the agreed pair. This rank's store holds only points under its
        /// own compat (storeCommon(rank): the manifests are parsed against it, another namespace is never indexed),
        /// so presence is the compat check; a layout or part this rank does not keep is `incompatible`. A point whose
        /// write this rank still has queued is `incomplete` (rank 0 keeps it; the write publishes here soon).
        pub fn has(self: *Self, h: Head, m: rank_wire.Has) !rank_wire.HasReply {
            const s = self.side orelse return .{ .answer = .disabled, .point = m.point, .compat = m.compat };
            const answer: rank_wire.Absence = if (m.layout != .replicated or m.part != 0)
                .incompatible
            else if (s.store.presence(m.point))
                .present
            else if (s.writing(m.point))
                .incomplete
            else
                .missing;
            try self.asked.put(self.gpa, h.txn, m.point);
            return .{ .answer = answer, .point = m.point, .compat = m.compat };
        }

        pub fn reserve(self: *Self, h: Head, m: rank_wire.Reserve) !void {
            const s = self.side orelse return error.SpillNotAgreed;
            _ = self.asked.remove(h.txn);
            if (!self.fam_live) return error.SpillNotAttached;
            try self.restores.ensureUnusedCapacity(self.gpa, 1);
            const rx = try s.newRx(h.txn, m.point, @intCast(m.at), m.blocks);
            var fx: Fx = .{ .rx = rx, .snap = m.generation };
            if (self.fam.reserve(self.fam.ptr, m.generation)) |buf| {
                fx.buf = true;
                rx.snap = .{ .id = m.generation, .buf = buf };
            } else |_| rx.fail(.reserve);
            if (rx.why == .none) rx.dst = s.gens.reserve(@intCast(m.base), @intCast(m.base + m.at)) catch blk: {
                rx.fail(.reserve);
                break :blk null;
            };
            self.restores.putAssumeCapacity(h.txn, fx);
        }

        pub fn start(self: *Self, h: Head, _: rank_wire.Start) !void {
            const s = self.side orelse return error.SpillNotAgreed;
            const fx = self.restores.get(h.txn) orelse return error.UnknownTxn;
            if (fx.rx.why != .none) return;
            fx.rx.rid = s.store.beginRead(fx.rx.key) catch |e| blk: {
                fx.rx.fail(if (e == error.Missing) .missing else .read);
                break :blk null;
            };
        }

        pub fn progress(self: *Self, h: Head, m: rank_wire.Progress) !void {
            const s = self.side orelse return error.SpillNotAgreed;
            const fx = self.restores.getPtr(h.txn) orelse return error.UnknownTxn;
            fx.allowed = m.upto;
            _ = s.retire();
            s.rxStep(fx.rx, fx.allowed, 2 * std.time.ns_per_ms); // (well inside follower_progress_ns: no disk wait)
        }

        fn recycle(ctx: ?*anyopaque) void {
            const s: *S = @ptrCast(@alignCast(ctx.?));
            _ = s.retire();
        }

        /// Copy the rest (bounded waits that recycle slots: PR 124's follower hang), install the small state,
        /// report; a deadline passed is `timeout`, never silence.
        pub fn validate(self: *Self, h: Head, _: rank_wire.Validate) !rank_wire.ValidateReply {
            const s = self.side orelse return error.SpillNotAgreed;
            const fx = self.restores.getPtr(h.txn) orelse return error.UnknownTxn;
            const rx = fx.rx;
            fx.allowed = rx.blocks;
            const deadline = monotonicNs() +| rank_wire.follower_copy_ns;
            while (rx.why == .none and rx.copied < rx.blocks) {
                _ = s.retire();
                s.rxStep(rx, rx.blocks, 5 * std.time.ns_per_ms);
                if (rx.why != .none or rx.copied >= rx.blocks) break;
                const now = monotonicNs();
                if (now >= deadline) {
                    rx.fail(.timeout);
                    break;
                }
                _ = s.store.readAwait(rx.rid orelse break, rx.copied, @min(deadline - now, 50 * std.time.ns_per_ms), .{ .ctx = s, .recycle = recycle });
            }
            if (rx.rid == null and rx.why == .none) rx.fail(.read);
            while (rx.why == .none and rx.small != .done) {
                _ = s.retire();
                s.rxSmall(rx);
                if (rx.why != .none or rx.small == .done) break;
                if (monotonicNs() >= deadline) {
                    rx.fail(.timeout);
                    break;
                }
                std.Io.sleep(self.io, .fromMicroseconds(200), .awake) catch {};
            }
            // every copy completed within the same bound (events polled, never an unbounded device wait); one still
            // running at the deadline is a timeout, and rank 0's abort then waits it before the buffer goes
            if (rx.why == .none and !s.rxSettle(rx, deadline)) rx.fail(.timeout);
            if (rx.why == .none and !s.rxValid(rx)) rx.fail(.reserve);
            const ok = rx.why == .none;
            return .{ .ok = ok, .why = rx.why, .copied = rx.copied, .bad_block = if (ok) rank_wire.no_block else @min(rx.bad_block, rank_wire.max_blocks - 1), .point = rx.key };
        }

        pub fn commit(self: *Self, h: Head, _: rank_wire.Commit) !void {
            const s = self.side orelse return error.SpillNotAgreed;
            const kv = self.restores.fetchRemove(h.txn) orelse return error.UnknownTxn;
            var d = try s.rxCommit(kv.value.rx);
            d.lineage.deinit(self.gpa);
            s.rxDrop(kv.value.rx);
            if (!self.fam_live) return error.SpillNotAttached;
            try self.fam.install(self.fam.ptr, kv.value.snap);
        }

        pub fn abort(self: *Self, h: Head, m: rank_wire.Abort) !void {
            const s = self.side orelse return; // (a has refused with spill off here)
            const kv = self.restores.fetchRemove(h.txn) orelse {
                // aborted at has: nothing reserved here; a forget (a rank lacked the point) drops this rank's copy
                if (self.asked.fetchRemove(h.txn)) |a| if (m.forget) s.store.forget(a.value, &.{}, &.{});
                return;
            };
            _ = self.asked.remove(h.txn);
            s.rxEnd(kv.value.rx, if (m.why == .cancelled or m.why == .shutdown) .cancel else .abort, m.forget);
            if (kv.value.buf and self.fam_live) self.fam.discard(self.fam.ptr, kv.value.snap);
        }

        pub fn writeStaged(self: *Self, _: Head, m: rank_wire.WriteStaged) !void {
            const s = self.side orelse return;
            s.guard(m.kept, m.upto * s.bt);
        }

        pub fn writeCancelled(self: *Self, _: Head, m: rank_wire.WriteCancelled) !void {
            const s = self.side orelse return;
            s.cancelWrite(m.kept);
        }

        pub fn stats(self: *Self, _: Head, m: rank_wire.Stats) !rank_wire.StatsReply {
            var c: pc.TierCounts = .{};
            if (self.side) |s| {
                s.refresh();
                c = s.cached;
            }
            const v = countsValues(c);
            return switch (m.set) {
                .tier => .{ .set = .tier, .n = v.n, .values = v.values },
            };
        }
    };
}

// -- tests --------------------------------------------------------------------------------------------------------------

const testing = std.testing;

test {
    _ = @import("spill_tier_test.zig");
}

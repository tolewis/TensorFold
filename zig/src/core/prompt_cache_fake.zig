//! A host-memory snapshot family shared by the prompt cache's tests and the learned-disk fault tests.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const Snapshots = pc.Snapshots;
const Saved = pc.Saved;
const Store = pc.Store;
const Plan = pc.Plan;
const Allocator = std.mem.Allocator;
/// Rank 1 applies queued drops before each request's keeps; drops made during a pass wait for the next request.
pub const Peer = struct {
    held: u64 = 0,
    pending: u64 = 0,
    max: u64 = 0,
    in_pass: bool = false,

    pub fn request(p: *Peer) void {
        p.held -= p.pending;
        p.pending = 0;
    }
};

/// A family over host memory for tests: its live state is a position and a running sum of the prompt's tokens.
pub const Fake = struct {
    gpa: Allocator,
    at: u32 = 0,
    sum: u64 = 0,
    disk_need_bytes: u64 = 0,
    disk_queries: usize = 0,
    disk_reserved: bool = false,
    disk_peer_writes: usize = 0,
    disk_fail_peer: bool = false,
    disk_fail_reserve_ack: bool = false,
    disk_fail_delete: bool = false,
    disk_finishes: usize = 0,
    disk_forgets: usize = 0,
    disk_release: ?*const fn () void = null,
    fail_save: bool = false,
    fail_write: bool = false,
    writes: usize = 0,
    fail_restore: bool = false,
    live: usize = 0,
    spare_bytes: u64 = 0,
    peer: ?*Peer = null, // speed-up mode's rank 1: it keeps the same states and drops them only when a request names them

    pub const State = struct { at: u32, sum: u64 };

    pub fn snapshots(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn } };
    }
    /// With learned states on disk: a state's position and sum in one file.
    pub fn learned(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .write = writeFn, .read = readFn, .forget = forgetFn, .forget_checked = checkedForgetFn, .reclaim = pc.singleReclaim } };
    }
    pub fn paired(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .write = pairWrite, .read = readFn, .forget = forgetFn, .forget_checked = checkedForgetFn, .reclaim = pc.singleReclaim, .peer_need = pairNeed, .peer_reclaim = pairReclaim, .peer_reserve = pairReserve, .peer_finish = pairFinish } };
    }
    fn pairNeed(ptr: *anyopaque, _: Saved, _: u64) anyerror!u64 {
        const f = of(ptr);
        f.disk_queries += 1;
        return f.disk_need_bytes;
    }
    fn pairReclaim(_: *anyopaque, _: u64) anyerror!u64 {
        return 16;
    }
    fn pairReserve(ptr: *anyopaque, _: Saved, _: u64, _: u64) anyerror!void {
        const f = of(ptr);
        if (f.disk_reserved or f.disk_need_bytes > 0) return error.PeerDiskFull;
        f.disk_reserved = true;
        if (f.disk_fail_reserve_ack) return error.LostReserveAck;
    }
    fn pairFinish(ptr: *anyopaque, _: u64, _: bool) anyerror!void {
        const f = of(ptr);
        if (!f.disk_reserved) return error.PeerDiskOutOfStep;
        f.disk_reserved = false;
        f.disk_finishes += 1;
    }
    fn pairWrite(ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void {
        const f = of(ptr);
        try writeFn(ptr, saved, dir, key);
        if (!f.disk_reserved) return error.PeerUnreserved;
        f.disk_peer_writes += 1;
        if (f.disk_fail_peer) return error.PeerWrite;
    }
    fn checkedForgetFn(ptr: *anyopaque, dir: [:0]const u8, key: u64) !void {
        if (of(ptr).disk_fail_delete) return error.DeleteFailed;
        try pc.singleForgetChecked(ptr, dir, key);
        forgetFn(ptr, dir, key);
    }
    fn forgetFn(ptr: *anyopaque, dir: [:0]const u8, key: u64) void {
        const f = of(ptr);
        f.disk_forgets += 1;
        if (f.disk_release) |release| {
            release();
            f.disk_need_bytes = f.disk_need_bytes -| 16;
        }

        var path: [512]u8 = undefined;
        _ = platform.c.unlink(file(&path, dir, key) catch return);
    }
    pub fn file(buf: []u8, dir: []const u8, key: u64) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0);
    }
    fn writeFn(ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void {
        const f = of(ptr);
        f.writes += 1;
        if (f.fail_write) return error.WriteFailed;
        var path: [512]u8 = undefined;
        const fd = platform.c.open(try file(&path, dir, key), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
        if (fd < 0) return error.WriteFailed;
        defer _ = platform.c.close(fd);
        try imprint.writeAll(fd, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn readFn(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved {
        const f = of(ptr);
        var path: [512]u8 = undefined;
        const fd = platform.c.open(try file(&path, dir, key), .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
        if (fd < 0) return error.ReadFailed;
        defer _ = platform.c.close(fd);
        const st = try f.gpa.create(State);
        errdefer f.gpa.destroy(st);
        if (!imprint.readAt(fd, std.mem.asBytes(st), 0) or st.at != at) return error.ReadFailed;
        f.live += 1;
        return st;
    }
    /// A pool-like family: a kept state holds 10 bytes under `bytes`, and a dropped one's storage stays spare.
    pub fn pooled(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = savePooledFn, .restore = restoreFn, .drop = dropSpareFn, .charged = chargedFn, .spare = spareFn, .trim = trimFn, .reuses = reusesFn } };
    }
    fn reusesFn(ptr: *anyopaque, at: u32) bool {
        return of(ptr).spare_bytes >= 90 + at;
    }
    fn savePooledFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved {
        const st = try saveFn(ptr, owner, at);
        const f = of(ptr);
        if (f.spare_bytes >= 90 + at) f.spare_bytes -= 90 + at; // the spare storage took it
        return st;
    }
    fn chargedFn(_: *anyopaque, saved: Saved) u64 {
        const st: *State = @ptrCast(@alignCast(saved));
        return 90 + st.at;
    }
    fn dropSpareFn(ptr: *anyopaque, saved: Saved) void {
        const st: *State = @ptrCast(@alignCast(saved));
        of(ptr).spare_bytes += 90 + st.at;
        dropFn(ptr, saved);
    }
    fn spareFn(ptr: *anyopaque) u64 {
        return of(ptr).spare_bytes;
    }
    fn trimFn(ptr: *anyopaque, room_: u64) void {
        const f = of(ptr);
        f.spare_bytes = @min(f.spare_bytes, room_);
    }
    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (f.fail_save) return error.CopyFailed;
        if (at != f.at) return error.NotAtMark;
        const st = try f.gpa.create(State);
        st.* = .{ .at = f.at, .sum = f.sum };
        f.live += 1;
        if (f.peer) |p| {
            p.held += 100 + at;
            p.max = @max(p.max, p.held);
        }
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: Saved) anyerror!void {
        const f = of(ptr);
        if (f.fail_restore) return error.CopyFailed;
        const st: *State = @ptrCast(@alignCast(saved));
        f.at, f.sum = .{ st.at, st.sum };
    }
    fn dropFn(ptr: *anyopaque, saved: Saved) void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        if (f.peer) |p| { // dropped before the pass: named in this request; during it: in the next one
            if (p.in_pass) p.pending += 100 + st.at else p.held -= 100 + st.at;
        }
        f.gpa.destroy(st);
        f.live -= 1;
    }

    /// A prompt pass from `plan.from`, keeping at each mark; returns the sum a fresh pass would give.
    pub fn pass(f: *Fake, s: *Store, prompt: []const u32, plan: Plan) u64 {
        if (plan.from == 0) f.* = .{ .gpa = f.gpa, .fail_save = f.fail_save, .fail_restore = f.fail_restore, .fail_write = f.fail_write, .writes = f.writes, .live = f.live, .spare_bytes = f.spare_bytes, .peer = f.peer };
        if (f.peer) |p| p.in_pass = true;
        defer if (f.peer) |p| {
            p.in_pass = false;
        };
        var mi: usize = 0;
        for (prompt[plan.from..]) |t| {
            f.sum = f.sum *% 31 +% t;
            f.at += 1;
            if (mi < plan.marks.len and plan.marks[mi] == f.at) {
                _ = s.keep(prompt, f.at, null, &.{}, &.{});
                mi += 1;
            }
        }
        return f.sum;
    }
};

/// States that share pages, one page every `page` tokens. `bytes(0)` is 0. `held` is the distinct pages.
/// `frees` is pages with one kept state. `returns` is pages with one holder in all.
pub const Shared = struct {
    gpa: Allocator,
    page: u32 = 64,
    token: u64 = 1,
    refs: std.ArrayList(u16) = .empty,
    kept: std.ArrayList(u16) = .empty,
    kept_pages: u32 = 0,
    live: std.ArrayList(u32) = .empty, // the stream's page table
    at: u32 = 0,
    fail_park: bool = false, // the next park hook returns an error and changes nothing

    const State = struct {
        ids: []u32,
        held_ids: []u32 = &.{},
        parked: bool = false,
    };
    const counted: Snapshots.VTable = .{
        .bytes = bytesFn,
        .save = saveFn,
        .restore = restoreFn,
        .drop = dropFn,
        .held = heldFn,
        .adds = addsFn,
        .frees = freesFn,
        .returns = returnsFn,
    };
    const with_park: Snapshots.VTable = .{
        .bytes = bytesFn,
        .save = saveFn,
        .restore = restoreFn,
        .drop = dropFn,
        .held = heldFn,
        .adds = addsFn,
        .frees = freesFn,
        .returns = returnsFn,
        .park_bytes = parkBytesFn,
        .park = parkFn,
    };

    pub fn deinit(f: *Shared) void {
        f.refs.deinit(f.gpa);
        f.kept.deinit(f.gpa);
        f.live.deinit(f.gpa);
    }

    pub fn snapshots(f: *Shared) Snapshots {
        return .{ .ptr = f, .vtable = &counted };
    }

    /// The counted family plus the host tier. A parked state keeps no page, so `frees` and `returns` are 0.
    pub fn parking(f: *Shared) Snapshots {
        return .{ .ptr = f, .vtable = &with_park };
    }

    /// The stream's table covers `at` tokens, taking a fresh page where it has none yet.
    pub fn grow(f: *Shared, at: u32) !void {
        const need: usize = (@as(usize, at) + f.page - 1) / f.page;
        while (f.live.items.len < need) {
            const id: u32 = @intCast(f.refs.items.len);
            try f.refs.append(f.gpa, 0);
            try f.kept.append(f.gpa, 0);
            try f.live.append(f.gpa, id);
        }
        f.at = at;
    }

    /// Drop the stream's table. Page ids already handed out stay, so the next conversation takes new pages.
    pub fn resetLive(f: *Shared) void {
        f.live.clearRetainingCapacity();
        f.at = 0;
    }

    /// The running stream holds `id` as well as any kept state, so a drop of that state may free no page.
    pub fn hold(f: *Shared, id: u32) void {
        f.refs.items[id] += 1;
    }

    pub fn pageBytes(f: *const Shared) u64 {
        return @as(u64, f.page) * f.token;
    }

    fn of(ptr: *anyopaque) *Shared {
        return @ptrCast(@alignCast(ptr));
    }

    fn bytesFn(ptr: *anyopaque, at: u32) u64 {
        return @as(u64, at) * of(ptr).token;
    }

    fn heldFn(ptr: *anyopaque) u64 {
        const f = of(ptr);
        return @as(u64, f.kept_pages) * f.pageBytes();
    }

    fn addsFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) u64 {
        const f = of(ptr);
        const n: usize = (@as(usize, at) + f.page - 1) / f.page;
        const up = @min(n, f.live.items.len);
        var fresh: u64 = 0;
        for (f.live.items[0..up]) |id| fresh += @intFromBool(f.kept.items[id] == 0);
        fresh += @as(u64, @intCast(n - up));
        return fresh * f.pageBytes() + bytesFn(ptr, 0);
    }

    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (at != f.at) return error.NotAtMark;
        const n: usize = (@as(usize, at) + f.page - 1) / f.page;
        if (n > f.live.items.len) return error.NotAtMark;
        const st = try f.gpa.create(State);
        errdefer f.gpa.destroy(st);
        st.ids = try f.gpa.alloc(u32, n);
        st.held_ids = &.{};
        st.parked = false;
        @memcpy(st.ids, f.live.items[0..n]);
        for (st.ids) |id| {
            f.refs.items[id] += 1;
            if (f.kept.items[id] == 0) f.kept_pages += 1;
            f.kept.items[id] += 1;
        }
        return st;
    }

    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: Saved) anyerror!void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        f.live.clearRetainingCapacity();
        try f.live.appendSlice(f.gpa, st.ids);
        f.at = @intCast(st.ids.len * @as(usize, f.page));
    }

    fn dropFn(ptr: *anyopaque, saved: Saved) void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        for (st.ids) |id| {
            if (f.kept.items[id] > 0) {
                f.kept.items[id] -= 1;
                if (f.kept.items[id] == 0) f.kept_pages -= 1;
            }
            if (f.refs.items[id] > 0) f.refs.items[id] -= 1;
        }
        if (st.held_ids.len > 0) f.gpa.free(st.held_ids);
        f.gpa.free(st.ids);
        f.gpa.destroy(st);
    }

    fn parkBytesFn(ptr: *anyopaque, saved: Saved) u64 {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        if (st.parked) return 0;
        return @as(u64, st.ids.len) * f.pageBytes();
    }

    fn parkFn(ptr: *anyopaque, saved: Saved) anyerror!f64 {
        const f = of(ptr);
        if (f.fail_park) return error.ParkFailed;
        const st: *State = @ptrCast(@alignCast(saved));
        if (st.parked) return 0;
        st.held_ids = try f.gpa.dupe(u32, st.ids);
        for (st.ids) |id| {
            if (f.kept.items[id] > 0) {
                f.kept.items[id] -= 1;
                if (f.kept.items[id] == 0) f.kept_pages -= 1;
            }
            if (f.refs.items[id] > 0) f.refs.items[id] -= 1;
        }
        f.gpa.free(st.ids);
        st.ids = &.{};
        st.parked = true;
        return 1;
    }

    fn freesFn(ptr: *anyopaque, saved: Saved) u64 {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        if (st.parked) return 0;
        var n: u64 = 0;
        for (st.ids) |id| n += @intFromBool(f.kept.items[id] == 1);
        return n * f.pageBytes();
    }

    fn returnsFn(ptr: *anyopaque, saved: Saved) u64 {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        if (st.parked) return 0;
        var n: u64 = 0;
        for (st.ids) |id| n += @intFromBool(f.refs.items[id] == 1);
        return n * f.pageBytes();
    }
};

//! Pictures and the durable tier (spill v2 over recipe 0006's content keys): `mediaKeys`, and what the store and the
//! lane host hand a tier for a prompt with pictures (Want.images/media at lookup, Kept.media/raw at keep), over the
//! fake backend and the in-memory tier.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const ft = @import("prompt_cache_fake_tier.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

fn digestOf(payload: []const u8) [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &d, .{});
    return d;
}

test "mediaKeys: every picture id named by a span and made from its digest, nothing else" {
    const a = digestOf("a");
    const b = digestOf("b");
    var key = [_]u32{ 1, 2, 0, 0, 0, 3, 0, 0, 0, 0, 4 };
    for (0..3) |j| key[2 + j] = pc.pictureId(&a, @intCast(j));
    for (0..4) |j| key[6 + j] = pc.pictureId(&b, @intCast(j));
    const m = [_]pc.Media{ .{ .start = 2, .len = 3, .digest = a }, .{ .start = 6, .len = 4, .digest = b } };
    try std.testing.expect(pc.mediaKeys(&key, key.len, &m));
    try std.testing.expect(pc.mediaKeys(&key, 8, &m)); // a cut inside the second picture
    try std.testing.expect(pc.mediaKeys(&key, 2, &.{})); // a cut before every picture: text
    try std.testing.expect(!pc.mediaKeys(&key, key.len, m[0..1])); // the second picture unnamed
    try std.testing.expect(!pc.mediaKeys(&key, key.len, &.{})); // none named
    try std.testing.expect(!pc.mediaKeys(&key, key.len, &.{ m[1], m[0] })); // out of order
    try std.testing.expect(!pc.mediaKeys(&key, key.len, &.{ m[0], .{ .start = 4, .len = 2, .digest = b } })); // overlapping
    try std.testing.expect(!pc.mediaKeys(&key, key.len, &.{ .{ .start = 2, .len = 3, .digest = b }, m[1] })); // another payload
    try std.testing.expect(!pc.mediaKeys(&key, key.len, &.{ .{ .start = 2, .len = 2, .digest = a }, m[1] })); // a short span
    try std.testing.expect(!pc.mediaKeys(&key, key.len + 1, &m)); // past the key
    // a picture id's high bit keeps it off every vocabulary id
    for (0..1000) |j| try std.testing.expect(pc.pictureId(&a, @intCast(j)) >= pc.picture_ids);
}

const Box = struct {
    mutex: std.Io.Mutex = .init,
    cached: ?u32 = null,
    done: ?Reason = null,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => {},
            .finished => |f| b.done = f.reason,
        }
    }
    fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }
    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// The in-memory tier, recording what the store hands it for pictures (lookups and keeps), everything else its own.
const Rec = struct {
    inner: *ft.FakeTier,
    mutex: std.Io.Mutex = .init,
    lookups: u32 = 0,
    picture_lookups: u32 = 0, // with Want.images
    want_media: [4]pc.Media = undefined,
    want_n: usize = 0,
    keeps: u32 = 0,
    kept_media: [4]pc.Media = undefined,
    kept_n: usize = 0,
    kept_raw: [64]u32 = undefined,
    kept_raw_n: usize = 0,
    kept_images: bool = false,
    /// The real tier's gate (spill_tier): with pictures, a lookup or keep whose media do not name its picture ids misses
    /// or keeps nothing.
    strict: bool = false,
    refused: u32 = 0,

    fn tier(r: *Rec) pc.Tier {
        return .{ .ptr = r, .vtable = &.{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .release = release, .share = share, .touch = touch, .forget = forget, .pump = pump, .busy = busy, .flush = flush, .counts = counts } };
    }

    fn of(p: *anyopaque) *Rec {
        return @ptrCast(@alignCast(p));
    }
    fn in(p: *anyopaque) struct { *anyopaque, *const pc.Tier.VTable } {
        const t = of(p).inner.tier();
        return .{ t.ptr, t.vtable };
    }
    fn begin(p: *anyopaque, point: *const pc.Point) anyerror!u64 {
        const x = in(p);
        return x[1].begin(x[0], point);
    }
    fn poll(p: *anyopaque, job: u64, budget: u64) pc.Poll {
        const x = in(p);
        return x[1].poll(x[0], job, budget);
    }
    fn cancel(p: *anyopaque, job: u64) void {
        const x = in(p);
        x[1].cancel(x[0], job);
    }
    fn release(p: *anyopaque, point: u64, saved: pc.Saved) void {
        const x = in(p);
        x[1].release.?(x[0], point, saved);
    }
    fn share(p: *anyopaque, point: u64) void {
        const x = in(p);
        x[1].share.?(x[0], point);
    }
    fn touch(p: *anyopaque, point: u64) void {
        const x = in(p);
        x[1].touch.?(x[0], point);
    }
    fn forget(p: *anyopaque, point: u64) void {
        const x = in(p);
        x[1].forget(x[0], point);
    }
    fn pump(p: *anyopaque, budget: u64) void {
        const x = in(p);
        x[1].pump.?(x[0], budget);
    }
    fn busy(p: *anyopaque) bool {
        const x = in(p);
        return x[1].busy.?(x[0]);
    }
    fn flush(p: *anyopaque, budget: u64) u64 {
        const x = in(p);
        return x[1].flush(x[0], budget);
    }
    fn counts(p: *anyopaque, out: []pc.TierCounts) usize {
        const x = in(p);
        return x[1].counts(x[0], out);
    }
    fn lookup(p: *anyopaque, want: *const pc.Want) ?pc.Point {
        const r = of(p);
        r.mutex.lockUncancelable(io);
        r.lookups += 1;
        if (want.images) r.picture_lookups += 1;
        r.want_n = @min(want.media.len, r.want_media.len);
        @memcpy(r.want_media[0..r.want_n], want.media[0..r.want_n]);
        const refuse = r.strict and want.images and !pc.mediaKeys(want.key, want.key.len, want.media);
        if (refuse) r.refused += 1;
        r.mutex.unlock(io);
        if (refuse) return null;
        const x = in(p);
        return x[1].lookup(x[0], want);
    }
    fn keep(p: *anyopaque, kept: *const pc.Kept) ?u64 {
        const r = of(p);
        r.mutex.lockUncancelable(io);
        r.keeps += 1;
        r.kept_images = kept.images;
        r.kept_n = @min(kept.media.len, r.kept_media.len);
        @memcpy(r.kept_media[0..r.kept_n], kept.media[0..r.kept_n]);
        r.kept_raw_n = @min(kept.raw.len, r.kept_raw.len);
        @memcpy(r.kept_raw[0..r.kept_raw_n], kept.raw[0..r.kept_raw_n]);
        const refuse = r.strict and kept.images and !pc.mediaKeys(kept.key, kept.at, kept.media);
        if (refuse) r.refused += 1;
        r.mutex.unlock(io);
        if (refuse) return null;
        const x = in(p);
        return x[1].keep(x[0], kept);
    }
};

const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    tier: ft.FakeTier,
    rec: Rec,
    host: LaneHost,

    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
    fn pack(_: *anyopaque, a: std.mem.Allocator, saved: pc.Saved) anyerror![]u8 {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        return a.dupe(u8, std.mem.sliceAsBytes(kept.items));
    }
    fn unpack(_: *anyopaque, b: []const u8) anyerror!pc.Saved {
        const kept = try gpa.create(std.ArrayList(u32));
        kept.* = .empty;
        const n = b.len / 4;
        try kept.resize(gpa, n);
        @memcpy(std.mem.sliceAsBytes(kept.items), b[0 .. n * 4]);
        return kept;
    }

    fn create() !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.* = .{ .cfg = undefined, .target = .{ .gpa = gpa }, .core = undefined, .store = undefined, .tier = .{ .gpa = gpa, .codec = .{ .ctx = r, .pack = pack, .unpack = unpack } }, .rec = undefined, .host = undefined };
        r.rec = .{ .inner = &r.tier };
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.store = pc.Store.init(gpa, .{ .ptr = &r.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.store.setTier(r.rec.tier(), 0);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = 2 });
        r.host.interleave_ns = 0;
        r.host.cache = &r.store;
        r.host.configure(.{ .load_ms = 5, .flush_ms = 1000 });
        try r.host.start();
        return r;
    }

    fn destroy(r: *Rig) void {
        r.host.stop();
        r.store.deinit();
        r.tier.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
        gpa.destroy(r);
    }
};

test "a prompt with pictures: its lookups carry the pictures' identities, its keeps their identities and model ids; a disk hit resumes it" {
    const r = try Rig.create();
    defer r.destroy();
    const d = digestOf("a picture's payload");
    // the picture's three positions hold the placeholder (99) in the prompt; the cache keys them by the picture
    const prompt = [_]u32{ 3, 1, 99, 99, 99, 5, 9, 2, 6, 5, 7, 7, 4, 4 };
    var key = prompt;
    for (0..3) |j| key[2 + j] = pc.pictureId(&d, @intCast(j));
    const media = [_]pc.Media{.{ .start = 2, .len = 3, .digest = d }};
    const img = [_]api.Image{.{ .at = 2, .tokens = 3, .bytes = "" }};
    // a point of 8 tokens on "disk" (its state the prompt's own tokens: the fake backend's state is its history)
    _ = try r.tier.put(key[0..8], 8, std.mem.sliceAsBytes(prompt[0..8]), 0, 0, false);
    var b: Box = .{};
    const req: Request = .{ .prompt = &prompt, .images = &img, .cache_key = &key, .cache_media = &media, .max_tokens = 4, .history_len = 12 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // restored, then resumed
    r.rec.mutex.lockUncancelable(io);
    defer r.rec.mutex.unlock(io);
    try std.testing.expect(r.rec.picture_lookups >= 1);
    try std.testing.expectEqual(@as(usize, 1), r.rec.want_n);
    try std.testing.expect(r.rec.want_media[0].eql(media[0]));
    try std.testing.expect(r.rec.keeps >= 1);
    try std.testing.expect(r.rec.kept_images);
    try std.testing.expectEqual(@as(usize, 1), r.rec.kept_n);
    try std.testing.expect(r.rec.kept_media[0].eql(media[0]));
    try std.testing.expectEqualSlices(u32, &prompt, r.rec.kept_raw[0..r.rec.kept_raw_n]); // the model's ids, not the key
}

test "tierMedia: the tier gets a request's media only when they name exactly the spans the engine runs" {
    const d = digestOf("p");
    const prompt = [_]u32{ 3, 1, 99, 99, 99, 5, 99, 99, 2 };
    const img = [_]api.Image{ .{ .at = 2, .tokens = 3, .bytes = "" }, .{ .at = 6, .tokens = 2, .bytes = "" } };
    const m = [_]pc.Media{ .{ .start = 2, .len = 3, .digest = d }, .{ .start = 6, .len = 2, .digest = d } };
    const ok: Request = .{ .prompt = &prompt, .images = &img, .cache_media = &m, .max_tokens = 1 };
    try std.testing.expectEqual(@as(usize, 2), LaneHost.tierMedia(&ok).len);
    const short: Request = .{ .prompt = &prompt, .images = &img, .cache_media = m[0..1], .max_tokens = 1 };
    try std.testing.expectEqual(@as(usize, 0), LaneHost.tierMedia(&short).len); // one picture unnamed
    const moved = [_]pc.Media{ m[0], .{ .start = 5, .len = 2, .digest = d } };
    const r2: Request = .{ .prompt = &prompt, .images = &img, .cache_media = &moved, .max_tokens = 1 };
    try std.testing.expectEqual(@as(usize, 0), LaneHost.tierMedia(&r2).len); // a span elsewhere
    const longer = [_]pc.Media{ m[0], .{ .start = 6, .len = 3, .digest = d } };
    const r3: Request = .{ .prompt = &prompt, .images = &img, .cache_media = &longer, .max_tokens = 1 };
    try std.testing.expectEqual(@as(usize, 0), LaneHost.tierMedia(&r3).len); // a span of another length
    const text: Request = .{ .prompt = &prompt, .max_tokens = 1 };
    try std.testing.expectEqual(@as(usize, 0), LaneHost.tierMedia(&text).len);
}

test "a prompt whose cache_media name other spans than its images: no disk hit, nothing kept with pictures" {
    const r = try Rig.create();
    defer r.destroy();
    r.rec.strict = true;
    const d = digestOf("a picture's payload");
    const prompt = [_]u32{ 3, 1, 99, 99, 99, 5, 9, 2, 6, 5, 7, 7, 4, 4 };
    var key = prompt;
    for (0..3) |j| key[2 + j] = pc.pictureId(&d, @intCast(j));
    // the key and media agree with each other (a picture at [2, 5)), but the engine is told the picture sits at [1, 4)
    const media = [_]pc.Media{.{ .start = 2, .len = 3, .digest = d }};
    const img = [_]api.Image{.{ .at = 1, .tokens = 3, .bytes = "" }};
    _ = try r.tier.put(key[0..8], 8, std.mem.sliceAsBytes(prompt[0..8]), 0, 0, false);
    var b: Box = .{};
    const req: Request = .{ .prompt = &prompt, .images = &img, .cache_key = &key, .cache_media = &media, .max_tokens = 4, .history_len = 12 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached); // the point on disk is not resumed
    r.rec.mutex.lockUncancelable(io);
    defer r.rec.mutex.unlock(io);
    try std.testing.expect(r.rec.picture_lookups >= 1);
    try std.testing.expectEqual(@as(usize, 0), r.rec.want_n); // the tier was handed no media
    try std.testing.expect(r.rec.keeps >= 1);
    try std.testing.expectEqual(@as(usize, 0), r.rec.kept_n);
    try std.testing.expect(r.rec.refused >= 2); // a lookup and a keep, both turned away
}

test "a prompt with pictures and no content key (TF_DS_IMAGE_CACHE=0): the tier is never asked, nothing is kept" {
    const r = try Rig.create();
    defer r.destroy();
    const prompt = [_]u32{ 3, 1, 99, 99, 99, 5, 9, 2, 6, 5, 7, 7, 4, 4 };
    const img = [_]api.Image{.{ .at = 2, .tokens = 3, .bytes = "" }};
    var b: Box = .{};
    const req: Request = .{ .prompt = &prompt, .images = &img, .max_tokens = 4, .history_len = 12 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    r.rec.mutex.lockUncancelable(io);
    defer r.rec.mutex.unlock(io);
    try std.testing.expectEqual(@as(u32, 0), r.rec.lookups);
    try std.testing.expectEqual(@as(u32, 0), r.rec.keeps);
}

test "a text prompt's lookups and keeps carry no pictures" {
    const r = try Rig.create();
    defer r.destroy();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7, 7, 5, 3 };
    var b: Box = .{};
    const req: Request = .{ .prompt = &prompt, .max_tokens = 4, .history_len = 12 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    r.rec.mutex.lockUncancelable(io);
    defer r.rec.mutex.unlock(io);
    try std.testing.expect(r.rec.lookups >= 1 and r.rec.keeps >= 1);
    try std.testing.expectEqual(@as(u32, 0), r.rec.picture_lookups);
    try std.testing.expect(!r.rec.kept_images and r.rec.kept_n == 0 and r.rec.kept_raw_n == 0);
}

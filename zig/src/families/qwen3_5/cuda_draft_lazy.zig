//! A drafter's prompt absorbs run late: when its window is next read, and only those whose rows the ring keeps.
//! A prompt step of more rows than the ring keeps sets the window by itself, so the held steps before it drop.
const std = @import("std");

/// The rows the ring keeps of one absorb, and the rows behind the newest that the target's tap ring holds.
pub const Limits = struct { keep: usize, held: usize };

/// A prompt absorb held back: the target's tap ring, its first row's position, its rows.
pub const Step = struct { ring: u64, start: u64, n: usize };

/// A stream's prompt absorbs not run yet, oldest first.
pub const Pending = struct {
    steps: [most]Step = undefined,
    n: usize = 0,

    pub const most = 8;
};

/// The rows an absorb computes: the last `keep` of the rows it is given.
pub const Rows = struct { start: u64, n: usize };

/// An absorb of rows [start, start + n) opens the window: begin and end as the absorb sets them before its rows run.
pub fn open(ctx: anytype, start: u64, n: usize, keep: usize) Rows {
    if (start < ctx.end) ctx.end = start;
    if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
    var r: Rows = .{ .start = start, .n = n };
    if (n > keep) { // the target's ring holds no more behind its newest
        r = .{ .start = start + (n - keep), .n = keep };
        ctx.begin = r.start;
    }
    ctx.end = r.start;
    return r;
}

/// The window after an absorb: no wider than the drafter's sliding window.
pub fn close(ctx: anytype, window: usize) void {
    if (ctx.end > window and ctx.begin < ctx.end - window) ctx.begin = ctx.end - window;
}

/// Holds back a prompt absorb. Runs the oldest held steps whose rows a next step could push out of the tap ring.
pub fn put(d: anytype, ctx: anytype, s: Step, l: Limits) !void {
    const p = &ctx.late;
    if (s.n > l.keep) p.n = 0; // its begin, end and ring rows owe nothing to the steps before it
    if (p.n == Pending.most) try runOldest(d, ctx);
    p.steps[p.n] = s;
    p.n += 1;
    // a next step of `keep` rows or fewer, and a verify pass, must find every row a held step reads
    while (p.n > 1 and reach(p.*, l.keep) + l.keep > l.held) try runOldest(d, ctx);
}

/// Runs the held steps in order: before the window is read or another absorb moves it.
pub fn flush(d: anytype, ctx: anytype) !void {
    const p = &ctx.late;
    const n = p.n;
    p.n = 0;
    for (p.steps[0..n]) |s| try d.absorb(ctx, s.ring, s.start, s.n, null, true);
}

fn runOldest(d: anytype, ctx: anytype) !void {
    const p = &ctx.late;
    const s = p.steps[0];
    std.mem.copyForwards(Step, p.steps[0 .. p.n - 1], p.steps[1..p.n]);
    p.n -= 1;
    try d.absorb(ctx, s.ring, s.start, s.n, null, true);
}

/// Rows from the oldest row a held step reads to the newest row of any held step.
fn reach(p: Pending, keep: usize) u64 {
    var lo: u64 = std.math.maxInt(u64);
    var hi: u64 = 0;
    for (p.steps[0..p.n]) |s| {
        lo = @min(lo, s.start + s.n - @min(s.n, keep));
        hi = @max(hi, s.start + s.n);
    }
    return hi - lo;
}

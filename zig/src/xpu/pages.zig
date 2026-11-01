//! The K/V pool's page bookkeeping: which page a slot's table or a kept prompt state holds, and which pages are free.
//! Host only, no device (tested here), so the reference counts are right before a card sees them.
const std = @import("std");

pub const Pages = struct {
    gpa: std.mem.Allocator,
    count: u32,
    refs: []u32, // references to each page: one a slot's table, one a kept prompt state
    free: std.ArrayList(u32) = .empty, // pages whose references fell to zero
    next: u32 = 0, // pages never handed out yet

    pub fn init(gpa: std.mem.Allocator, count: u32) !Pages {
        const refs = try gpa.alloc(u32, @as(usize, count));
        @memset(refs, 0);
        return .{ .gpa = gpa, .count = count, .refs = refs };
    }

    pub fn deinit(p: *Pages) void {
        p.gpa.free(p.refs);
        p.free.deinit(p.gpa);
    }

    /// A page no reference holds, with the caller's reference on it: from the freed pages, else never handed out
    /// yet; null when the pool is spent.
    pub fn take(p: *Pages) ?u32 {
        while (p.free.pop()) |id| if (p.refs[id] == 0) {
            p.refs[id] = 1;
            return id;
        };
        if (p.next >= p.count) return null;
        p.next += 1;
        p.refs[p.next - 1] = 1;
        return p.next - 1;
    }

    /// One reference to page `id` more.
    pub fn retain(p: *Pages, id: u32) void {
        if (id < p.count) p.refs[id] += 1;
    }

    /// One reference to page `id` less; the page comes back at zero.
    pub fn release(p: *Pages, id: u32) void {
        if (id >= p.count or p.refs[id] == 0) return;
        p.refs[id] -= 1;
        if (p.refs[id] == 0) p.free.append(p.gpa, id) catch {};
    }

    /// References to page `id` (0: free).
    pub fn held(p: *const Pages, id: u32) u32 {
        return if (id < p.count) p.refs[id] else 0;
    }

    /// Pages that can still be handed out: never handed out, plus freed ones.
    pub fn spare(p: *const Pages) u32 {
        return (p.count - p.next) + @as(u32, @intCast(p.free.items.len));
    }
};

test "a page is handed out once, and comes back only when no reference holds it" {
    var p = try Pages.init(std.testing.allocator, 4);
    defer p.deinit();
    const a = p.take().?;
    const b = p.take().?;
    try std.testing.expect(a != b);
    try std.testing.expectEqual(@as(u32, 1), p.held(a)); // the caller's reference
    p.retain(a); // a kept state holds it too
    p.release(a);
    try std.testing.expectEqual(@as(u32, 1), p.held(a)); // the kept state still does
    p.release(a);
    try std.testing.expectEqual(@as(u32, 0), p.held(a));
    try std.testing.expectEqual(@as(u32, 3), p.spare()); // b is held, a and one unused page are free
    const c = p.take().?;
    try std.testing.expectEqual(a, c); // the freed page before a new one
    try std.testing.expectEqual(@as(u32, 1), p.held(c));
}

test "the pool refuses a page it does not have, and a release of nothing does not wrap" {
    var p = try Pages.init(std.testing.allocator, 2);
    defer p.deinit();
    _ = p.take().?;
    _ = p.take().?;
    try std.testing.expectEqual(@as(?u32, null), p.take());
    p.release(7); // outside the pool: no change
    try std.testing.expectEqual(@as(u32, 0), p.spare());
    p.release(1);
    try std.testing.expectEqual(@as(u32, 1), p.spare());
    try std.testing.expectEqual(@as(u32, 1), p.take().?);
    try std.testing.expectEqual(@as(u32, 0), p.held(1));
}

test "a page released twice is not handed out twice at once" {
    var p = try Pages.init(std.testing.allocator, 1);
    defer p.deinit();
    const a = p.take().?;
    p.release(a);
    p.release(a); // a second release of the same reference does nothing
    try std.testing.expectEqual(@as(u32, 1), p.spare());
    try std.testing.expectEqual(@as(u32, 0), p.take().?);
    try std.testing.expectEqual(@as(?u32, null), p.take());
}

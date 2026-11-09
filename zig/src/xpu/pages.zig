//! The K/V pool's page bookkeeping: which page a slot's table or a kept prompt state holds, and which pages are free.
//! Host only, no device (tested here), so the reference counts are right before a card sees them.
const std = @import("std");

pub const Pages = struct {
    gpa: std.mem.Allocator,
    count: u32,
    refs: []u32, // references to each page: one a slot's table, one a kept prompt state
    /// References to a page that kept prompt states hold. A part of `refs`: a slot's table is the rest.
    kept: []u16 = &.{},
    /// Pages with `kept` above 0. One page under two states counts once.
    kept_pages: u32 = 0,
    free: std.ArrayList(u32) = .empty, // pages whose references fell to zero
    next: u32 = 0, // pages never handed out yet

    pub fn init(gpa: std.mem.Allocator, count: u32) !Pages {
        const refs = try gpa.alloc(u32, @as(usize, count));
        errdefer gpa.free(refs);
        const kept = try gpa.alloc(u16, @as(usize, count));
        @memset(refs, 0);
        @memset(kept, 0);
        return .{ .gpa = gpa, .count = count, .refs = refs, .kept = kept };
    }

    pub fn deinit(p: *Pages) void {
        p.gpa.free(p.refs);
        p.gpa.free(p.kept);
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

    /// `retain` plus one kept-state reference. The page counts in `kept_pages` while any kept state holds it.
    pub fn keep(p: *Pages, id: u32) void {
        p.retain(id);
        if (id >= p.count) return;
        if (p.kept[id] == 0) p.kept_pages += 1;
        p.kept[id] += 1;
    }

    /// `release` plus one kept-state reference less. A page no kept state holds leaves `kept_pages`.
    pub fn unkeep(p: *Pages, id: u32) void {
        if (id < p.count and p.kept[id] > 0) {
            p.kept[id] -= 1;
            if (p.kept[id] == 0) p.kept_pages -= 1;
        }
        p.release(id);
    }

    /// Kept-state references to page `id` (0: no kept state holds it).
    pub fn keptRefs(p: *const Pages, id: u32) u16 {
        return if (id < p.count) p.kept[id] else 0;
    }

    /// True when one holder in all has page `id` (`refs` is 1).
    pub fn sole(p: *const Pages, id: u32) bool {
        return id < p.count and p.refs[id] == 1;
    }

    /// References to page `id` (0: free).
    pub fn held(p: *const Pages, id: u32) u32 {
        return if (id < p.count) p.refs[id] else 0;
    }

    /// Pages that can still be handed out: never handed out, plus freed ones.
    pub fn spare(p: *const Pages) u32 {
        return (p.count - p.next) + @as(u32, @intCast(p.free.items.len));
    }

    /// Whether the pool can give `want` pages and still keep `reserve` pages free for the live streams.
    pub fn roomFor(p: *const Pages, want: u32, reserve: u32) bool {
        return @as(u64, p.spare()) >= @as(u64, want) + @as(u64, reserve);
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
    try std.testing.expectEqual(@as(u32, 1), p.held(1)); // the reference the take gave
}

test "kept states give way to a stream, and a stream's own pages are never freed under it" {
    var p = try Pages.init(std.testing.allocator, 8);
    defer p.deinit();
    // stream A takes two pages; a kept state of A's slot shares them (saveSlot retains the slot's pages)
    const a0 = p.take().?;
    const a1 = p.take().?;
    p.retain(a0);
    p.retain(a1);
    // stream B takes two pages, and a kept state of a slot whose sequence ended holds two more
    const b0 = p.take().?;
    const b1 = p.take().?;
    try std.testing.expect(b0 != b1);
    const k0 = p.take().?;
    const k1 = p.take().?;
    p.retain(k0); // the kept state's reference
    p.retain(k1);
    p.release(k0); // its slot's reference went when that sequence ended
    p.release(k1);
    try std.testing.expectEqual(@as(u32, 1), p.held(k0));
    try std.testing.expectEqual(@as(u32, 2), p.spare());
    // B grows past its pages and the pool has two spare: no kept state goes
    try std.testing.expect(p.roomFor(2, 0));
    _ = p.take().?;
    _ = p.take().?;
    try std.testing.expectEqual(@as(u32, 0), p.spare());
    try std.testing.expect(!p.roomFor(1, 0));
    // the least recently used kept state gives way: its pages come back, A's shared pages do not
    p.release(k0);
    p.release(k1);
    try std.testing.expectEqual(@as(u32, 2), p.spare());
    try std.testing.expect(p.roomFor(2, 0));
    const c0 = p.take().?;
    try std.testing.expect(c0 == k0 or c0 == k1); // a freed page before a new one
    // dropping A's own kept state frees nothing: A still holds those pages
    p.release(a0);
    p.release(a1);
    try std.testing.expectEqual(@as(u32, 1), p.held(a0));
    try std.testing.expectEqual(@as(u32, 1), p.held(a1));
    try std.testing.expectEqual(@as(u32, 1), p.spare()); // only the second freed page: one was taken
    try std.testing.expectEqual(@as(u32, 1), p.held(b0));
}

test "kept states count a shared page once, and sole is the one holder in all" {
    var p = try Pages.init(std.testing.allocator, 4);
    defer p.deinit();
    const a = p.take().?; // the slot's reference
    p.keep(a); // a kept state holds it too
    try std.testing.expectEqual(@as(u16, 1), p.keptRefs(a));
    try std.testing.expectEqual(@as(u32, 1), p.kept_pages);
    try std.testing.expectEqual(@as(u32, 2), p.held(a));
    try std.testing.expect(!p.sole(a)); // the slot and the state
    p.keep(a); // a second kept state shares it
    try std.testing.expectEqual(@as(u16, 2), p.keptRefs(a));
    try std.testing.expectEqual(@as(u32, 1), p.kept_pages); // still one page
    p.unkeep(a);
    try std.testing.expectEqual(@as(u16, 1), p.keptRefs(a));
    try std.testing.expectEqual(@as(u32, 1), p.kept_pages);
    try std.testing.expectEqual(@as(u32, 2), p.held(a)); // the slot and the state that remains
    p.unkeep(a);
    try std.testing.expectEqual(@as(u16, 0), p.keptRefs(a));
    try std.testing.expectEqual(@as(u32, 0), p.kept_pages);
    try std.testing.expect(p.sole(a)); // only the slot
    // a page only a kept state holds: one reference, so a drop gives it back
    const b = p.take().?;
    p.release(b);
    p.keep(b);
    try std.testing.expect(p.sole(b));
    try std.testing.expectEqual(@as(u16, 1), p.keptRefs(b));
    try std.testing.expectEqual(@as(u32, 1), p.kept_pages);
    p.unkeep(b);
    try std.testing.expectEqual(@as(u32, 0), p.held(b));
    try std.testing.expectEqual(@as(u32, 0), p.kept_pages);
    p.keep(9); // outside the pool: no count
    p.unkeep(9);
    try std.testing.expectEqual(@as(u16, 0), p.keptRefs(9));
    try std.testing.expectEqual(@as(u32, 0), p.kept_pages);
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

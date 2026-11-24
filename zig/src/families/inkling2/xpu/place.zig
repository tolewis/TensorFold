//! Which heads and experts one TP rank holds. TP 1 is the full model. The kernels are not told the rank:
//! they see the slice this rank loaded, and the forward joins the partials.
const std = @import("std");

pub const hidden: u32 = 2304;
pub const experts: u32 = 128;
pub const q_heads: u32 = 24;
pub const kv_heads: u32 = 8;
pub const head_dim: u32 = 128;
pub const d_rel: u32 = 16;
pub const unpadded: u32 = 200058;

pub const Place = struct {
    tp: u32 = 1,
    rank: u32 = 0,

    pub fn parse(tp: u32, rank: u32) !Place {
        if (tp != 1 and tp != 2) return error.Inkling2Tp;
        if (rank >= tp) return error.Inkling2Tp;
        return .{ .tp = tp, .rank = rank };
    }

    pub fn split(p: Place) bool {
        return p.tp > 1;
    }

    pub fn qHeads(p: Place) u32 {
        return if (p.split()) q_heads / p.tp else q_heads;
    }

    pub fn kvHeads(p: Place) u32 {
        return if (p.split()) kv_heads / p.tp else kv_heads;
    }

    pub fn qWidth(p: Place) u32 {
        return p.qHeads() * head_dim;
    }

    pub fn kvWidth(p: Place) u32 {
        return p.kvHeads() * head_dim;
    }

    pub fn relWidth(p: Place) u32 {
        return p.qHeads() * d_rel;
    }

    /// Expert ids this rank holds: a contiguous half, 0..63 or 64..127.
    pub fn expertBase(p: Place) u32 {
        return if (p.split()) p.rank * (experts / p.tp) else 0;
    }

    pub fn expertCount(p: Place) u32 {
        return if (p.split()) experts / p.tp else experts;
    }

    pub fn holdsExpert(p: Place, id: u32) bool {
        if (!p.split()) return id < experts;
        const b = p.expertBase();
        return id >= b and id < b + p.expertCount();
    }

    pub fn localExpert(p: Place, id: u32) u32 {
        return id - p.expertBase();
    }

    /// Unpadded vocab rows this rank owns. 200058 does not divide by 2: rank 0 takes the low 100029.
    pub fn vocabBase(p: Place) u32 {
        if (!p.split()) return 0;
        return if (p.rank == 0) 0 else (unpadded + 1) / 2;
    }

    pub fn vocabCount(p: Place) u32 {
        if (!p.split()) return unpadded;
        const mid = (unpadded + 1) / 2;
        return if (p.rank == 0) mid else unpadded - mid;
    }
};

test "tp 1 is the full model and tp 2 splits even heads" {
    const one = try Place.parse(1, 0);
    try std.testing.expectEqual(@as(u32, 24), one.qHeads());
    try std.testing.expectEqual(@as(u32, 128), one.expertCount());
    const a = try Place.parse(2, 0);
    const b = try Place.parse(2, 1);
    try std.testing.expectEqual(@as(u32, 12), a.qHeads());
    try std.testing.expectEqual(@as(u32, 4), b.kvHeads());
    try std.testing.expect(a.holdsExpert(0) and !a.holdsExpert(64));
    try std.testing.expect(b.holdsExpert(64) and !b.holdsExpert(63));
    try std.testing.expectEqual(@as(u32, 100029), a.vocabCount());
    try std.testing.expectEqual(@as(u32, 100029), b.vocabCount());
    try std.testing.expectEqual(unpadded, a.vocabCount() + b.vocabCount());
}

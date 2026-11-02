//! Host-side bookkeeping of a kept prompt state's K/V pages that move to pinned host RAM when the pool runs short:
//! which state is spilled and how much host memory that holds. The runner copies the pages' contents over the card.

const std = @import("std");
const q = @import("qwen27.zig");
const assert = std.testing.expect;

/// A kept prompt state at a slot's page table, in the store's keep/restore order: `used` ages a restore to the end.
pub const Kept = struct { id: u64, slot: u32, at: u32, used: u64 };

test "the least recently used kept state is the one the pool runs short on" {
    var kept = std.ArrayList(q.Kept).init(std.testing.allocator);
    defer kept.deinit();
    try kept.append(.{ .id = 1, .slot = 0, .at = 40960, .used = 1 });
    try kept.append(.{ .id = 2, .slot = 0, .at = 81920, .used = 2 });
    try kept.append(.{ .id = 3, .slot = 0, .at = 16_384, .used = 3 });

    var spill_bytes: u64 = 0;
    const bytes = @as(u64, 40960) * 16_640; // a rank's K/V for a 40k state at TP 2
    try assert(q.spillLru(&kept, &spill_bytes, 64 << 30, 16_640) != null);
    try assert(q.spillLru(&kept, &spill_bytes, 64 << 30, 16_640).?.id == 1); // the lowest `used`: the oldest kept state
}

test "the spill budget holds one state's K/V before the next one" {
    var kept = std.ArrayList(q.Kept).init(std.testing.allocator);
    defer kept.deinit();
    try kept.append(.{ .id = 1, .slot = 0, .at = 40960, .used = 1 });
    try kept.append(.{ .id = 2, .slot = 0, .at = 40960, .used = 2 });
    var spill_bytes: u64 = 0;
    const bytes = @as(u64, 40960) * 16_640; // a rank's K/V for a 40k state at TP 2
    try assert(q.spillLru(&kept, &spill_bytes, bytes, 16_640) != null);
    // the budget is exceeded: the next kept state stays on device
    spill_bytes = bytes;
    try assert(q.spillLru(&kept, &spill_bytes, bytes, 16_640) == null);
}

test "no kept state is spilled and none is dropped" {
    var kept = std.ArrayList(q.Kept).init(std.testing.allocator);
    defer kept.deinit();
    var spill_bytes: u64 = 0;
    try assert(q.spillLru(&kept, &spill_bytes, 64 << 30, 16_640) == null);
}

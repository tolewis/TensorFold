//! Host bytes one parked prompt state takes, summed over the ranks. No device.
const std = @import("std");

/// K and V bytes of one token on one rank. `kv_heads` is 4/TP. Sixteen attention layers, K and V.
pub fn kvPerToken(kv_heads: u64) u64 {
    return 16 * 2 * (kv_heads * 256 + kv_heads * 4);
}

/// One rank's kept GDN host copy. `v_heads` is 48/TP and `k_heads` is 16/TP. Forty-eight GDN layers.
pub fn keptGdnCopy(v_heads: u64, k_heads: u64) u64 {
    const qkv = (2 * k_heads + v_heads) * 128;
    const rec = v_heads * 128 * 128 * 4;
    const conv = 3 * qkv * 2;
    return 48 * (rec + conv);
}

/// Sum over `tp` ranks of one rank's K/V and its kept GDN host copy, plus rank 0's drafter context.
/// `gdn` is that host copy. With the pool and XPU_KEPT_HOST, `snap.bytes` is the K/V page charge, so it is not `gdn`.
pub fn parkTierCharge(at: u64, kv_per_token: u64, gdn: u64, ctx: u64, tp: u64) u64 {
    return (at * kv_per_token + gdn) * tp + ctx;
}

test "a 100k park at TP 2 charges K and V once" {
    const ctx: u64 = 5 * 2 * 8 * 2080 * 128 * 2;
    const kv = kvPerToken(2);
    const gdn = keptGdnCopy(24, 8);
    const charge = parkTierCharge(100_000, kv, gdn, ctx, 2);
    try std.testing.expectEqual(@as(u64, 3_524_542_464), charge);
    // The old charge passed snap.bytes, the K/V size, as the GDN copy.
    const doubled = parkTierCharge(100_000, kv, 100_000 * kv, ctx, 2);
    try std.testing.expectEqual(@as(u64, 6_698_598_400), doubled);
    try std.testing.expect(charge != doubled);
    // TP 4 uses its own head counts, then multiplies by 4. The heads divide, so the sum matches TP 2.
    const at_tp4 = parkTierCharge(100_000, kvPerToken(1), keptGdnCopy(12, 4), ctx, 4);
    try std.testing.expectEqual(charge, at_tp4);
}

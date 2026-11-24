//! Mini's Q8_0-weight dense projections and unembedding head on XPU (zig/tests/xpu/inkling2_dense_q8.zig): the twelve kernels
//! of zig/kernels/xpu/inkling2/dense_q8.cl and their three planted twins each, as gate 5 records them. Checked by
//! `xpu-kernel-contract-limits`, which prints each RECORD line.
const limits = @import("xpu_limits.zig");
const xpu_records = @import("xpu_records.zig");

const native = limits.Format.native;
const R = xpu_records.Record;

fn rec(comptime name: []const u8, comptime symbol: []const u8, comptime image: []const u8) R {
    return .{
        .launch = .{ .name = name, .symbol = symbol, .image = image, .format = native, .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    };
}

const kinds = [_][]const u8{ "dense_q8_1", "dense_q8_2", "dense_q8_4", "dense_q8_8", "dense_q8_16", "dense_q8_32", "head_q8_1", "head_q8_2", "head_q8_4", "head_q8_8", "head_q8_16", "head_q8_32" };
const twins = [_][3][]const u8{
    .{ "contract_inkling2_dense_q8_store", " (store truncated)", "" },
    .{ "contract_inkling2_dense_q8_sums", " (block sums in fp16)", "" },
    .{ "contract_inkling2_dense_q8_scale", " (one scale a plane)", "" },
};

pub const records = blk: {
    @setEvalBranchQuota(20000);
    var out: [kinds.len * 4]R = undefined;
    for (kinds, 0..) |kind, i| {
        out[i] = rec("xpu inkling2 q8 tf_ink2_" ++ kind, "tf_ink2_" ++ kind, "inkling2_dense_q8");
        for (twins, 0..) |t, j| out[(j + 1) * kinds.len + i] = rec("planted xpu inkling2 q8 tf_ink2_" ++ kind ++ t[1], "tf_ctl_ink2_" ++ kind, t[0]);
    }
    break :blk out;
};

//! Mini's dense projections and norm on XPU (zig/tests/xpu/inkling2_dense.zig): the seven kernels of
//! zig/kernels/xpu/inkling2/dense.cl and their two planted twins each, as gate 5 records them. Checked by
//! `xpu-kernel-contract-limits`, which prints each RECORD line.
const limits = @import("xpu_limits.zig");
const xpu_records = @import("xpu_records.zig");

const native = limits.Format.native;
const R = xpu_records.Record;

fn rec(comptime name: []const u8, comptime symbol: []const u8, comptime image: []const u8, comptime threads: u32, comptime slm: u32) R {
    return .{
        .launch = .{ .name = name, .symbol = symbol, .image = image, .format = native, .threads = .{ threads, 1, 1 }, .slm = slm, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ threads, 1, 1 }, .slm = slm, .sub_group = 16, .grf = 256 }},
    };
}

const kinds = [_][]const u8{ "dense_1", "dense_2", "dense_4", "dense_8", "dense_16", "dense_32", "norm" };

pub const records = blk: {
    @setEvalBranchQuota(10000);
    var out: [kinds.len * 3]R = undefined;
    for (kinds, 0..) |kind, i| {
        const norm = i == kinds.len - 1;
        const threads: u32 = if (norm) 256 else 128;
        const slm: u32 = if (norm) 64 else 0;
        out[i] = rec("xpu inkling2 dense tf_ink2_" ++ kind, "tf_ink2_" ++ kind, "inkling2_dense", threads, slm);
        out[kinds.len + i] = rec("planted xpu inkling2 dense tf_ink2_" ++ kind ++ " (store truncated)", "tf_ctl_ink2_" ++ kind, "contract_inkling2_dense_store", threads, slm);
        out[2 * kinds.len + i] = rec("planted xpu inkling2 dense tf_ink2_" ++ kind ++ (if (norm) " (eps dropped)" else " (chunk sums in fp16)"), "tf_ctl_ink2_" ++ kind, "contract_inkling2_dense_sums", threads, slm);
    }
    break :blk out;
};

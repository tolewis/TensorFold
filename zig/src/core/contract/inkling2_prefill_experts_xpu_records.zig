//! Inkling 2 Mini's routed experts for prompt chunks on XPU (zig/tests/xpu/inkling2_prefill_experts.zig): the seven kernels of
//! zig/kernels/xpu/inkling2/prefill_experts.cl and their planted twins, as gate 5 records them. Checked by
//! `xpu-kernel-contract-limits`, which prints each RECORD line.
const limits = @import("xpu_limits.zig");
const xpu_records = @import("xpu_records.zig");

const native = limits.Format.native;
const R = xpu_records.Record;
const U = limits.XpuUsage;

fn rec(comptime name: []const u8, comptime symbol: []const u8, comptime image: []const u8, comptime usage: U) R {
    return .{
        .launch = .{ .name = name, .symbol = symbol, .image = image, .format = native, .threads = usage.threads, .slm = usage.slm, .sub_group = 16, .grf = 256 },
        .usage = .{usage},
    };
}

const kinds = [_][]const u8{ "plan", "gather", "nvfp4_gu", "nvfp4_down", "mxfp8_gu", "mxfp8_down", "combine" };
const usages = [_]U{
    .{ .threads = .{ 256, 1, 1 }, .slm = 66592, .sub_group = 16, .grf = 256 },
    .{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
    .{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
    .{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
    .{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
    .{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
    .{ .threads = .{ 256, 1, 1 }, .slm = 64, .sub_group = 16, .grf = 256 },
};

pub const records = blk: {
    var out: [kinds.len * 2]R = undefined;
    for (kinds, usages, 0..) |kind, usage, i| {
        out[i] = rec("xpu inkling2 prefill experts tf_ink2_pf_" ++ kind, "tf_ink2_pf_" ++ kind, "inkling2_prefill_experts", usage);
        out[kinds.len + i] = rec("planted xpu inkling2 prefill experts tf_ink2_pf_" ++ kind, "tf_ctl_pf_" ++ kind ++ "_" ++ kind, "contract_inkling2_prefill_experts_controls", usage);
    }
    break :blk out;
};

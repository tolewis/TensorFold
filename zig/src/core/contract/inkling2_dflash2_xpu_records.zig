//! Mini DFlash 2's XPU launches (zig/tests/xpu/inkling2_dflash2.zig): the production kernels and the contract's planted copies.
const limits = @import("xpu_limits.zig");
const xpu_records = @import("xpu_records.zig");

const glue = limits.Format.native;
const R = xpu_records.Record;

/// The six kernels of zig/kernels/xpu/inkling2/dflash2.cl, then the six planted copies that each carry one arithmetic
/// mistake (zig/kernels/xpu/inkling2/contract_dflash2_controls.cl). Filled from `xpu-kernel-contract-limits`.
pub const records = [_]R{
    .{
        .launch = .{ .name = "inkling2 dflash2 conv", .symbol = "tf_ink2_dflash2_conv", .image = "inkling2_dflash2", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dflash2 block_qk", .symbol = "tf_ink2_dflash2_block_qk", .image = "inkling2_dflash2", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dflash2 context_kv", .symbol = "tf_ink2_dflash2_context_kv", .image = "inkling2_dflash2", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dflash2 attention", .symbol = "tf_ink2_dflash2_attention", .image = "inkling2_dflash2", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 4224, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 4224, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dflash2 merge", .symbol = "tf_ink2_dflash2_merge", .image = "inkling2_dflash2", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dflash2 topk", .symbol = "tf_ink2_dflash2_topk", .image = "inkling2_dflash2", .format = glue, .threads = .{ 16, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dflash2 conv", .symbol = "tf_ctl_ink2_dflash2_conv", .image = "contract_inkling2_dflash2_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dflash2 block_qk", .symbol = "tf_ctl_ink2_dflash2_block_qk", .image = "contract_inkling2_dflash2_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dflash2 context_kv", .symbol = "tf_ctl_ink2_dflash2_context_kv", .image = "contract_inkling2_dflash2_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dflash2 attention", .symbol = "tf_ctl_ink2_dflash2_attention", .image = "contract_inkling2_dflash2_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 4224, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 4224, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dflash2 merge", .symbol = "tf_ctl_ink2_dflash2_merge", .image = "contract_inkling2_dflash2_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dflash2 topk", .symbol = "tf_ctl_ink2_dflash2_topk", .image = "contract_inkling2_dflash2_controls", .format = glue, .threads = .{ 16, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 16, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
};

//! Mini DSpark's XPU launches (zig/tests/xpu/inkling2_dspark.zig): the production kernels and the contract's planted copies.
const limits = @import("xpu_limits.zig");
const xpu_records = @import("xpu_records.zig");

const glue = limits.Format.native;
const R = xpu_records.Record;

/// The eleven kernels of zig/kernels/xpu/inkling2/dspark.cl, then the eleven planted copies that each carry one
/// arithmetic mistake (zig/kernels/xpu/inkling2/contract_dspark_controls.cl). Filled from `xpu-kernel-contract-limits`.
pub const records = [_]R{
    .{
        .launch = .{ .name = "inkling2 dspark init_rows", .symbol = "tf_ink2_dspark_init_rows", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark fuse_taps", .symbol = "tf_ink2_dspark_fuse_taps", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark block_qkv", .symbol = "tf_ink2_dspark_block_qkv", .image = "inkling2_dspark", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark context_kv", .symbol = "tf_ink2_dspark_context_kv", .image = "inkling2_dspark", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark attention", .symbol = "tf_ink2_dspark_attention", .image = "inkling2_dspark", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 1056, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 1056, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark residual_norm", .symbol = "tf_ink2_dspark_residual_norm", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark swiglu", .symbol = "tf_ink2_dspark_swiglu", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark kappa", .symbol = "tf_ink2_dspark_kappa", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark recurrent", .symbol = "tf_ink2_dspark_recurrent", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark base", .symbol = "tf_ink2_dspark_base", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "inkling2 dspark logits", .symbol = "tf_ink2_dspark_logits", .image = "inkling2_dspark", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark init_rows", .symbol = "tf_ctl_ink2_dspark_init_rows", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark fuse_taps", .symbol = "tf_ctl_ink2_dspark_fuse_taps", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark block_qkv", .symbol = "tf_ctl_ink2_dspark_block_qkv", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark context_kv", .symbol = "tf_ctl_ink2_dspark_context_kv", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark attention", .symbol = "tf_ctl_ink2_dspark_attention", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 128, 1, 1 }, .slm = 1056, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 128, 1, 1 }, .slm = 1056, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark residual_norm", .symbol = "tf_ctl_ink2_dspark_residual_norm", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 32, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark swiglu", .symbol = "tf_ctl_ink2_dspark_swiglu", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark kappa", .symbol = "tf_ctl_ink2_dspark_kappa", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark recurrent", .symbol = "tf_ctl_ink2_dspark_recurrent", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark base", .symbol = "tf_ctl_ink2_dspark_base", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 dspark logits", .symbol = "tf_ctl_ink2_dspark_logits", .image = "contract_inkling2_dspark_controls", .format = glue, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 }},
    },
};

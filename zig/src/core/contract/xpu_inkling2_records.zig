//! Inkling 2 Mini's XPU launches for gate 5: the glue and runtime kernels, and the planted control of each.
const R = @import("xpu_records.zig");
const limits = @import("xpu_limits.zig");

const native = limits.Format.native;

/// Each kernel's declaration and what ocloc made of it (the `RECORD` lines of `xpu-kernel-contract-limits`).
pub const records = [_]R.Record{
    .{
        .launch = .{ .name = "inkling2 glue embed_norm", .symbol = "tf_inkling2_embed_norm", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 64, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 64, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue ar", .symbol = "tf_inkling2_ar", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 640, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 640, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue conv_static", .symbol = "tf_inkling2_conv_static", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue conv_dynamic", .symbol = "tf_inkling2_conv_dynamic", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 128, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 128, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue router", .symbol = "tf_inkling2_router", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 10304, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 10304, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue swiglu", .symbol = "tf_inkling2_swiglu", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue logits", .symbol = "tf_inkling2_logits", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue expert_pick", .symbol = "tf_inkling2_expert_pick", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 glue expert_combine", .symbol = "tf_inkling2_expert_combine", .image = "inkling2_glue", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 runtime copy_bf16", .symbol = "tf_ink2_copy_bf16", .image = "inkling2_runtime", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 runtime copy_f32", .symbol = "tf_ink2_copy_f32", .image = "inkling2_runtime", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 runtime widen", .symbol = "tf_ink2_widen", .image = "inkling2_runtime", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 runtime attention_partial", .symbol = "tf_ink2_attention_partial", .image = "inkling2_runtime", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 runtime mlp_partial", .symbol = "tf_ink2_mlp_partial", .image = "inkling2_runtime", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "inkling2 runtime capture_tap", .symbol = "tf_ink2_capture_tap", .image = "inkling2_runtime", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue embed_norm (one edit)", .symbol = "tf_ctl_inkling2_embed_norm", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 64, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 64, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue ar (one edit)", .symbol = "tf_ctl_inkling2_ar", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 640, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 640, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue conv_static (one edit)", .symbol = "tf_ctl_inkling2_conv_static", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue conv_dynamic (one edit)", .symbol = "tf_ctl_inkling2_conv_dynamic", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 128, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 128, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue router (one edit)", .symbol = "tf_ctl_inkling2_router", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 10304, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 10304, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue swiglu (one edit)", .symbol = "tf_ctl_inkling2_swiglu", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue logits (one edit)", .symbol = "tf_ctl_inkling2_logits", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue expert_pick (one edit)", .symbol = "tf_ctl_inkling2_expert_pick", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 glue expert_combine (one edit)", .symbol = "tf_ctl_inkling2_expert_combine", .image = "inkling2_glue_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 runtime copy_bf16 (one edit)", .symbol = "tf_ctl_ink2_copy_bf16", .image = "inkling2_runtime_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 runtime copy_f32 (one edit)", .symbol = "tf_ctl_ink2_copy_f32", .image = "inkling2_runtime_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 runtime widen (one edit)", .symbol = "tf_ctl_ink2_widen", .image = "inkling2_runtime_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 runtime attention_partial (one edit)", .symbol = "tf_ctl_ink2_attention_partial", .image = "inkling2_runtime_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 runtime mlp_partial (one edit)", .symbol = "tf_ctl_ink2_mlp_partial", .image = "inkling2_runtime_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
    .{
        .launch = .{ .name = "planted inkling2 runtime capture_tap (one edit)", .symbol = "tf_ctl_ink2_capture_tap", .image = "inkling2_runtime_ctl", .format = native, .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256 },
        .usage = .{.{ .threads = .{ 256, 1, 1 }, .slm = 0, .sub_group = 16, .grf = 256, .spill = 0, .private = 0 }},
    },
};

//! Inkling 2 Mini on the TensorFold XPU backend (Intel Arc Pro B70, Level Zero): the resident model, stream state, snapshots and the whole forward, the lane adapter and DSpark.
pub const plan = @import("plan.zig");
pub const model = @import("xpu/model.zig");
pub const state = @import("xpu/state.zig");
pub const snapshot = @import("xpu/snapshot.zig");
pub const forward = @import("xpu/forward.zig");
pub const fuse = @import("xpu/fuse.zig");
pub const attention = @import("xpu/attention.zig");
pub const attention_plan = @import("xpu/attention_plan.zig");
pub const prefill = @import("xpu/prefill.zig");
pub const experts = @import("xpu/experts.zig");
pub const backend = @import("xpu/backend.zig");
pub const dspark = @import("xpu/dspark.zig");
pub const dflash2 = @import("xpu/dflash2.zig");
pub const sampler = @import("xpu/sampler.zig");

test {
    _ = state;
    _ = snapshot;
    _ = forward;
    _ = prefill;
    _ = backend;
    _ = dspark;
    _ = dflash2;
    _ = @import("xpu/dflash2_kernels.zig");
    _ = @import("xpu/dflash2_state.zig");
    _ = @import("xpu/dflash2_weights.zig");
    _ = sampler;
    _ = @import("xpu/dspark_fold.zig");
    _ = @import("xpu/dspark_kernels.zig");
    _ = @import("xpu/dspark_state.zig");
    _ = @import("xpu/dspark_weights.zig");
    _ = model;
}

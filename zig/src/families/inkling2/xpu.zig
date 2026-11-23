//! Inkling 2 Mini on the TensorFold XPU backend (Intel Arc Pro B70, Level Zero): the resident model, stream state, snapshots and the whole forward.
pub const plan = @import("plan.zig");
pub const model = @import("xpu/model.zig");
pub const state = @import("xpu/state.zig");
pub const snapshot = @import("xpu/snapshot.zig");
pub const forward = @import("xpu/forward.zig");

test {
    _ = state;
    _ = snapshot;
    _ = forward;
    _ = model;
}

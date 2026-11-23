//! Inkling 2 Mini on the TensorFold XPU backend (Intel Arc Pro B70, Level Zero): the resident model, stream state and snapshots.
pub const plan = @import("plan.zig");
pub const model = @import("xpu/model.zig");
pub const state = @import("xpu/state.zig");
pub const snapshot = @import("xpu/snapshot.zig");

test {
    _ = state;
    _ = snapshot;
    _ = model;
}

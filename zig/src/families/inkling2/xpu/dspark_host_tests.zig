//! The drafter's XPU host files, tested together without a card (build step test-inkling2-dspark-xpu-host).
test {
    _ = @import("dspark_kernels.zig");
    _ = @import("dspark_state.zig");
    _ = @import("dspark_weights.zig");
    _ = @import("dspark_fold.zig");
}

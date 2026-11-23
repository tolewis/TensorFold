//! Mini DSpark's OpenCL C text, and the planted-control file that includes it with one arithmetic edit per kernel.

pub const dspark = @embedFile("dspark.cl");
pub const controls = @embedFile("contract_dspark_controls.cl");

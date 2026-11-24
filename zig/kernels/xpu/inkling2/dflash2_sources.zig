//! Mini DFlash 2's OpenCL C text, and the planted-control file that includes it with one arithmetic edit per kernel.

pub const dflash2 = @embedFile("dflash2.cl");
pub const controls = @embedFile("contract_dflash2_controls.cl");

//! Inkling 2 Mini cross-pass messages and message gates on Intel Xe2: Level Zero launches of `loop.cl`.
//! The same functions and argument meaning as `metal/loop.zig`, with device addresses in place of buffer offsets.
const std = @import("std");
const xpu = @import("xpu");
pub const plan = @import("inkling2_loop_plan");

pub const hidden: u32 = plan.hidden;
pub const rank: u32 = plan.rank;
pub const message_entries: u32 = plan.message_entries;
pub const RowsArgs = plan.RowsArgs;

/// A device address (`DeviceBuffer.at`), the offset already added.
pub const Ref = ?*anyopaque;

pub const messages_symbol: [:0]const u8 = "tf_inkling2_loop_messages";
pub const gates_symbol: [:0]const u8 = "tf_inkling2_loop_gates";
/// The first of the two launches of a gate tuple: the D projection to zeta.
pub const zeta_symbol: [:0]const u8 = "tf_inkling2_loop_zeta";
/// Work-items a work-group: 16 sub-groups of 16.
pub const threads: u32 = 256;
/// Work-items a work-group of the gates launch, and the ranks a work-group of the zeta launch covers.
pub const gate_threads: u32 = 64;
pub const zeta_ranks: u32 = 2;

comptime {
    std.debug.assert(std.mem.eql(u8, messages_symbol, plan.Kind.messages.symbol()));
    std.debug.assert(std.mem.eql(u8, gates_symbol, plan.Kind.gates.symbol()));
}

pub const Pipelines = struct {
    messages: xpu.Kernel,
    gates: xpu.Kernel,
    zeta: xpu.Kernel,

    /// The three kernels of a loaded `loop.cl` image, each at the work-group size its record states.
    pub fn load(module: xpu.Module) !Pipelines {
        var messages = try module.kernel(messages_symbol, .{ threads, 1, 1 });
        errdefer messages.deinit();
        var zeta = try module.kernel(zeta_symbol, .{ threads, 1, 1 });
        errdefer zeta.deinit();
        const gate_kernel = try module.kernel(gates_symbol, .{ gate_threads, 1, 1 });
        return .{ .messages = messages, .gates = gate_kernel, .zeta = zeta };
    }

    pub fn deinit(p: *Pipelines) void {
        p.gates.deinit();
        p.zeta.deinit();
        p.messages.deinit();
        p.* = undefined;
    }
};

/// Output layout is [token][12 core layers][q,k,v,mlp][H], all binary32. The banks are bf16 [token][H].
pub fn crossPassMessageAttention(p: *const Pipelines, s: xpu.Stream, bank0: Ref, bank1: Ref, q_mlp: Ref, kv: Ref, out: Ref, rows: u32) !void {
    std.debug.assert(rows > 0 and rows <= 64);
    try xpu.launch.dispatch(p.messages, s, .{ rows * message_entries, 1, 1 }, .{ bank0, bank1, q_mlp, kv, out, rows });
}

/// One controller tuple; h and output are [token][H] binary32 tensors, m's token rows are `m_stride` binary32 values apart
/// (H for a [token][H] tensor), down is bf16 [64][2H], forget and gain are bf16 [H][64]. `zeta` is scratch for [token][64]
/// binary32: the first launch writes it, the second reads it.
pub fn gatesOnly(p: *const Pipelines, s: xpu.Stream, h: Ref, m: Ref, forget: Ref, gain: Ref, zeta: Ref, out: Ref, rows: u32, m_stride: u32) !void {
    std.debug.assert(rows > 0 and rows <= 64);
    try xpu.launch.dispatch(p.gates, s, .{ hidden / gate_threads, rows, 1 }, .{ h, m, forget, gain, zeta, out, rows, m_stride });
}

pub fn gates(p: *const Pipelines, s: xpu.Stream, h: Ref, m: Ref, down: Ref, forget: Ref, gain: Ref, zeta: Ref, out: Ref, rows: u32, m_stride: u32) !void {
    std.debug.assert(rows > 0 and rows <= 64);
    try xpu.launch.dispatch(p.zeta, s, .{ rank / zeta_ranks, rows, 1 }, .{ h, m, down, zeta, rows, m_stride });
    try gatesOnly(p, s, h, m, forget, gain, zeta, out, rows, m_stride);
}

//! ModelOpt decode projections of one input in one lane launch (cuda/qmmf_many.zig). Each keeps its own call's bits.
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");

/// True: one launch wrote every out. False: no launch, the caller runs each projection alone.
pub fn decode(e: *engine.Engine, ls: []const engine.Proj, x: u64, outs: []const u64, rows: usize) !bool {
    const k = if (e.mo) |*m| m else return false;
    var ws: [cuda.qmmf_many.most]cuda.qmmf.Weight = undefined;
    if (ls.len > ws.len) return false;
    for (ls, ws[0..ls.len]) |l, *w| switch (l) {
        .ql => |q| switch (q) {
            .fp8g, .bf16 => |v| w.* = v,
            else => return false,
        },
        else => return false,
    };
    return k.lane.matmulMany(e.stream, x, ws[0].k, rows, ws[0..ls.len], outs);
}

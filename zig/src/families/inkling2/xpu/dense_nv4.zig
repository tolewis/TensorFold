//! Dense projections over NVFP4 weights, by the expert kernels (experts.cl's nvfp4_down and nvfp4_down_prompt).
//! One matrix is one expert: codes and e4m3 scales in nv4.zig's layout, factor in amax, denominator 1. One row of x
//! takes the decode kernel (the bandwidth path). Two or more take the prompt kernel, which reads each weight row
//! once for up to 32 rows of x. The head stays on the Q8 kernels.
const std = @import("std");
const xpu = @import("xpu");
const experts = @import("experts.zig");
const nv4 = @import("nv4.zig");

pub const Ref = ?*anyopaque;

fn plus(p: Ref, off: usize) Ref {
    return @ptrFromInt(@intFromPtr(p.?) + off);
}

fn call(rows: u32, n: u32, k: u32) experts.Call {
    return .{ .rows = rows, .k = k, .n = n, .experts = 1, .topk = 1, .format = .nvfp4 };
}

/// `weight` is nv4.bytes(n, k) with the factor and a denominator of 1 at the end. `aux` is nv4.fillAux.
/// Every row count takes the decode kernel: a row's dot is an M=1 DPAS in the same order as a one-row call, so an
/// 8-row verify matches eight one-row calls bit for bit. `XPU_NV4_MULTI=prompt` uses the prompt kernel instead
/// (one weight read for up to 32 rows; the sum order can differ).
pub fn dense(ks: *const experts.Kernels, s: xpu.Stream, aux: Ref, weight: Ref, input: Ref, out: Ref, rows: u32, n: u32, k: u32) !void {
    std.debug.assert(rows > 0 and n > 0 and nv4.admits(k));
    const codes = weight;
    const scales = plus(weight, nv4.scaleOffset(n, k));
    const amax = plus(weight, nv4.factorOffset(n, k));
    const den = plus(weight, nv4.factorOffset(n, k) + 4);
    const prompt_multi = if (std.c.getenv("XPU_NV4_MULTI")) |v| std.mem.eql(u8, std.mem.span(v), "prompt") else false;
    var base: u32 = 0;
    while (base < rows) {
        const left = rows - base;
        const x = plus(input, @as(usize, base) * k * 2);
        const y = plus(out, @as(usize, base) * n * 2);
        if (left == 1 or !prompt_multi) {
            const t = @min(left, 128);
            try ks.nvfp4Down(s, x, codes, scales, amax, den, aux, y, call(t, n, k));
            base += t;
        } else {
            const t = @min(left, 32);
            const plan = plus(aux, (nv4.aux_ids + nv4.aux_members + @as(usize, t - 1) * 4) * 4);
            const members = plus(aux, nv4.aux_ids * 4);
            try ks.nvfp4DownPrompt(s, x, codes, scales, amax, den, plan, plus(plan, 12), members, y, call(t, n, k));
            base += t;
        }
    }
}

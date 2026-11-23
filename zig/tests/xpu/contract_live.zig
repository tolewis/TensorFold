//! Gate 5 on XPU: the card's own properties against the BMG-G31 row, and every loaded kernel's against its record.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const limits = core.xpu_limits;
const records = core.xpu_records;

fn isControl(r: records.Record) bool {
    return std.mem.startsWith(u8, r.launch.name, "control");
}

fn imageOf(name: []const u8) ?rig_.Image {
    if (std.mem.eql(u8, name, "glue")) return .glue;
    if (std.mem.eql(u8, name, "tp")) return .tp;
    if (std.mem.eql(u8, name, "decode")) return .decode;
    if (std.mem.eql(u8, name, "contract_attn_upstream")) return .attn_upstream;
    if (std.mem.eql(u8, name, "contract_merge_upstream")) return .merge_upstream;
    if (std.mem.eql(u8, name, "contract_gate_controls")) return .gate;
    if (std.mem.eql(u8, name, "contract_controls")) return .controls;
    if (std.mem.eql(u8, name, "contract_inkling2_experts")) return .inkling2;
    if (std.mem.eql(u8, name, "contract_inkling2_controls")) return .inkling2_ctl;
    return null;
}

/// The arguments the launcher sets for a symbol, as the source's parameter list holds them.
fn argsOf(symbol: []const u8) u32 {
    const table = [_]struct { []const u8, u32 }{
        .{ "add_rmsnorm", 14 },
        .{ "add_rmsnorm_dec", 14 },
        .{ "gated_norm", 12 },
        .{ "swiglu", 7 },
        .{ "argmax", 3 },
        .{ "topk_part", 6 },
        .{ "topk_merge", 6 },
        .{ "embed", 6 },
        .{ "tp2_allreduce_f16", 4 },
        .{ "tp4_allreduce_f16", 6 },
        .{ "tp2_rs_f16", 5 },
        .{ "tp4_rs_f16", 7 },
        .{ "tp2_allreduce_bf16", 4 },
        .{ "tp4_allreduce_bf16", 6 },
        .{ "tp2_allreduce_f32", 4 },
        .{ "tp4_allreduce_f32", 6 },
        .{ "tp4_rs_f32", 7 },
        .{ "attn_decode", 13 },
        .{ "attn_merge", 8 },
        .{ "tf_ctl_attn_drop_slice", 8 },
        .{ "tf_ctl_attn_merge_unguarded", 8 },
    };
    for (table) |e| if (std.mem.eql(u8, e[0], symbol)) return e[1];
    if (inkling2Args(symbol)) |n| return n;
    return 1; // gate 5's controls take one output pointer
}

/// Inkling 2's experts: a symbol is an image prefix then the operation's name, so it is told by its ending.
fn inkling2Args(symbol: []const u8) ?u32 {
    const table = [_]struct { []const u8, u32 }{
        .{ "nvfp4_gu", 11 },
        .{ "nvfp4_down", 11 },
        .{ "mxfp8_gu", 9 },
        .{ "mxfp8_down", 9 },
        .{ "nvfp4_quantize", 7 },
        .{ "mxfp8_quantize", 5 },
        .{ "combine", 7 },
        .{ "plan", 7 },
        .{ "nvfp4_gu_prompt", 12 },
        .{ "nvfp4_down_prompt", 12 },
        .{ "mxfp8_gu_prompt", 10 },
        .{ "mxfp8_down_prompt", 10 },
    };
    if (!std.mem.startsWith(u8, symbol, "tf_ink2_") and !std.mem.startsWith(u8, symbol, "tf_ctl_i2_")) return null;
    for (table) |e| if (std.mem.endsWith(u8, symbol, e[0])) return e[1];
    return null;
}

/// Gate 5 on this card: the device's limits against its row, each loaded kernel against its record, each control red.
pub fn live(a: std.mem.Allocator, dev: *rig_.Dev) !usize {
    var n: usize = 0;
    const t = limits.xe[0];
    const p = dev.ctx.gpu.props;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const device = limits.Device{
        .deviceId = p.deviceId,
        .numEUsPerSubslice = p.numEUsPerSubslice,
        .numThreadsPerEU = p.numThreadsPerEU,
        .physicalEUSimdWidth = p.physicalEUSimdWidth,
        .numSlices = p.numSlices,
        .numSubslicesPerSlice = p.numSubslicesPerSlice,
    };
    const row = try limits.liveRow(device, t, &out, a);
    n += row;
    const cores = @as(u64, device.numSlices) * device.numSubslicesPerSlice;
    std.debug.print("{s} limits live {s}: PCI id 0x{x}, {d} EUs a Xe-core, {d} threads an EU, SIMD {d}, {d} Xe-cores{s}", .{
        if (row == 0) "PASS" else "FAIL", t.name,                     device.deviceId, device.numEUsPerSubslice,
        device.numThreadsPerEU,           device.physicalEUSimdWidth, cores,           if (out.items.len > 0) ": " else "\n",
    });
    if (out.items.len > 0) std.debug.print("{s}", .{out.items});
    const built = dev.controls != null;
    if (!built) std.debug.print("N/A  limits live contract_controls: the driver does not build the image ({t}); gate 5's table holds its limits\n", .{dev.controls_err.?});
    for (records.records ++ records.controls) |rec| {
        out.clearRetainingCapacity();
        const control = isControl(rec);
        if (!rec.compiled) {
            const fits = try limits.fits(rec.launch, t, &out, a);
            if (control and fits == 0) n += 1;
            if (!control) n += fits;
            const word = if (control) (if (fits != 0) "RED " else "MISSED") else if (fits != 0) "FAIL" else "PASS";
            std.debug.print("{s} limits live {s}: declared only, ocloc does not build it for this device{s}", .{
                word, rec.launch.name, if (out.items.len > 0) ": " else "\n",
            });
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            continue;
        }
        const img = imageOf(rec.launch.image) orelse return error.UnknownImage;
        if (!built and img == .controls) continue;
        const sym = try a.allocSentinel(u8, rec.launch.symbol.len, 0);
        defer a.free(sym);
        @memcpy(sym, rec.launch.symbol);
        var k = try dev.probe(img, sym);
        defer k.deinit();
        const kp = try dev.properties(k);
        const u = rec.usage[0];
        const loaded = limits.Loaded{
            .threads = if (kp.requiredGroupSize[0] == 0) u.threads else kp.requiredGroupSize,
            .sub_group = if (kp.requiredSubgroupSize == 0) u.sub_group else kp.requiredSubgroupSize,
            .max_sub_group = if (kp.maxSubgroupSize == 0) u.sub_group else kp.maxSubgroupSize,
            .slm = kp.localMemSize,
            .spill = kp.spillMemSize,
            .args = kp.numKernelArgs,
        };
        const problems = try limits.liveKernel(rec.launch, u, loaded, argsOf(rec.launch.symbol), t, &out, a);
        const fits = try limits.fits(rec.launch, t, &out, a);
        if (control and (problems != 0 or fits == 0)) n += 1;
        if (!control) n += problems + fits;
        const word = if (control) (if (problems == 0 and fits != 0) "RED " else "MISSED") else if (problems != 0 or fits != 0) "FAIL" else "PASS";
        const items = @as(u64, loaded.threads[0]) * loaded.threads[1] * loaded.threads[2];
        std.debug.print("{s} limits live {s}: {d} work-items required, sub-group {d} (the kernel's most {d}), " ++
            "{d} SLM bytes, {d} spill bytes, {d} private bytes, {d} arguments{s}", .{
            word,       rec.launch.name, items,             loaded.sub_group, loaded.max_sub_group,
            loaded.slm, loaded.spill,    kp.privateMemSize, loaded.args,      if (out.items.len > 0) ": " else "\n",
        });
        if (out.items.len > 0) std.debug.print("{s}", .{out.items});
    }
    return n;
}

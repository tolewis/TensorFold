//! XPU (Intel Xe2 over Level Zero): opt-in steps for the Qwen3.8-27B runner (`xpu-qwen27`) and Nemotron (`-Dxpu`).

const std = @import("std");
const exlr8_build = @import("exlr8.zig");
const sage_config = @import("../src/families/qwen3_5/sage_config.zig");
const exl3_variants = @import("../src/xpu/exl3_variants.zig");
const xpu_contract = @import("xpu_contract.zig");

/// ze.zig as its own module: the Qwen27 runner and the Nemotron runtime share one Level Zero table.
fn zeModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

var tp_ranks: u32 = 1;

/// The qwen27 kernels whose compiled widths follow the rank (shape.h): the layer kernels, not the drafter's.
fn tpKernel(name: []const u8) bool {
    for ([_][]const u8{ "glue", "decode", "gemm", "gemm128", "gdn", "k1" }) |n| if (std.mem.eql(u8, name, n)) return true;
    return false;
}

/// shape.h at TP 2: every head count and width halved, the hidden size and the vocabulary whole. The GDN input
/// row is qkv 5120 | z 3072 | a 24 | b 24 = 8240, padded to 8448; the attention row (q | gate) x 12 | k 2 | v 2.
const tp2_defines = " -DQ27_FF=8704 -DQ27_GK=8 -DQ27_GV=24 -DQ27_GVK=3 -DQ27_NQK=1024 -DQ27_NCONV=5120 -DQ27_VD=3072" ++
    " -DQ27_GDN_IN=8448 -DQ27_GA=8192 -DQ27_GB=8216 -DQ27_AQ=12 -DQ27_AKV=2 -DQ27_AH=16 -DQ27_AV0=14 -DQ27_GQA=6 -DQ27_QPG=6" ++
    " -DQ27_QPARTS=1 -DQ27_ATT_IN=7168 -DQ27_AK0=6144 -DQ27_AV0C=6656 -DQ27_OD=3072 -DQ27_OB=24 -DQ27_KVROW=512";

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    const ocloc = b.option([]const u8, "ocloc", "Intel offline compiler") orelse "ocloc";
    const device = b.option([]const u8, "xpu-device", "ocloc device for the XPU kernels") orelse "bmg";
    const dpas = b.option(bool, "xpu-dpas", "The device has DPAS (XMX); off: plain fallbacks (default off for Gen12LP: tgllp, rkl, adl-*, dg1)") orelse hasDpas(device);
    const ngen_include = b.option([]const u8, "ngen-include", "nGEN header directory (oneDNN)") orelse "ngen";
    // -Dtp=2: the two-card tensor-parallel rank of the 27B (lanes/qwen27/TP2-ASSESSMENT.md in Intel-B70-Recipe). The
    // rank holds half the heads and half the MLP columns: the qwen27 kernels (not the drafter's, not Sage's) get
    // the halved widths as defines, and xpu/qwen27.zig reads `tp` from kernel_options.
    tp_ranks = b.option(u32, "tp", "tensor-parallel ranks of the XPU 27B build: 1 (default) or 2") orelse 1;
    if (tp_ranks != 1 and tp_ranks != 2) @panic("-Dtp takes 1 or 2");
    const files = b.addWriteFiles();
    const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread";
    // the Gemma 4 prompt-rows attention: 128 registers a thread, twice the resident threads to hide latency
    const opts_rows = "-cl-std=CL3.0 -cl-intel-128-GRF-per-thread";
    for ([_]struct { []const u8, []const u8, []const u8 }{
        .{ "glue", "zig/kernels/xpu/qwen27/glue.cl", "" },
        .{ "decode", "zig/kernels/xpu/qwen27/decode.cl", "" },
        // K1: the split-K DPAS decode attention over int8 pool pages for the 27B (lane p22-k1)
        .{ "k1", "zig/kernels/xpu/qwen27/k1.cl", "" },
        .{ "gemm", "zig/kernels/xpu/qwen27/gemm.cl", " -DGSZ=64" },
        .{ "gemm128", "zig/kernels/xpu/qwen27/gemm.cl", " -DGSZ=128" },
        .{ "gdn", "zig/kernels/xpu/qwen27/gdn_chunk.cl", "" },
        .{ "drafter", "zig/kernels/xpu/qwen27/drafter.cl", "" },
        // XPU_DRAFT_TP (lane p4-kernel): the drafter of one TP 2 rank, half its heads and half the head's top-k parts.
        // Built at -Dtp=2 only.
        .{ "drafter_tp", "zig/kernels/xpu/qwen27/drafter.cl", " -DNH=16 -DNKV=4 -DTOPK_PARTS=128" },
        // Sage-A1.5 (qwen3_5_moe): the same sources at its shape (shape.h); the GEMM and GDN chunk images are shared
        .{ "sage_glue", "zig/kernels/xpu/qwen27/glue.cl", " " ++ sage_config.sage_defines },
        .{ "sage_decode", "zig/kernels/xpu/qwen27/decode.cl", " " ++ sage_config.sage_defines },
        .{ "sage_moe", "zig/kernels/xpu/sage_moe.cl", "" },
        .{ "qwen_vision", "zig/kernels/xpu/qwen_vision.cl", "" },
        .{ "glue_g4s", "zig/kernels/xpu/gemma4/glue_g4.cl", " -I zig/kernels/xpu/gemma4" },
        .{ "glue_g4g", "zig/kernels/xpu/gemma4/glue_g4.cl", " -DHD=512 -I zig/kernels/xpu/gemma4" },
        .{ "rows_g4s", "zig/kernels/xpu/gemma4/glue_g4.cl", " -I zig/kernels/xpu/gemma4" },
        .{ "rows_g4g", "zig/kernels/xpu/gemma4/glue_g4.cl", " -DHD=512 -I zig/kernels/xpu/gemma4" },
    }) |k| {
        if (std.mem.eql(u8, k[0], "drafter_tp") and tp_ranks != 2) continue;
        const gemma = std.mem.startsWith(u8, k[0], "glue_g4") or std.mem.startsWith(u8, k[0], "rows_g4");
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
        c.addFileArg(b.path(k[1]));
        if (gemma) {
            c.addArgs(&.{ "-device", device, "-options", b.fmt("{s}{s} -I zig/kernels/xpu/qwen27", .{ if (std.mem.startsWith(u8, k[0], "rows_")) opts_rows else opts, k[2] }), "-output_no_suffix", "-output", k[0], "-out_dir" });
            c.setCwd(b.path("."));
            c.addFileInput(b.path("zig/kernels/xpu/gemma4/attn_rows_g4.cl"));
            for ([_][]const u8{ "glue.cl", "glue_attn.cl", "glue_quant.cl", "shape.h" }) |part| c.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
        } else {
            const tp_defs = if (tp_ranks == 2 and tpKernel(k[0])) tp2_defines else "";
            const o = if (std.mem.eql(u8, k[0], "sage_moe")) b.fmt("{s}{s}", .{ opts, k[2] }) else b.fmt("{s}{s}{s} -I zig/kernels/xpu/qwen27", .{ opts, k[2], tp_defs });
            c.addArgs(&.{ "-device", device, "-options", o, "-output_no_suffix", "-output", k[0], "-out_dir" });
            c.setCwd(b.path("."));
            if (!std.mem.eql(u8, k[0], "sage_moe")) for ([_][]const u8{ "glue_attn.cl", "glue_quant.cl", "shape.h", "k1_body.h" }) |part| c.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
        }
        const dir = c.addOutputDirectoryArg(b.fmt("ocloc-{s}", .{k[0]}));
        _ = files.addCopyFile(dir.path(b, b.fmt("{s}.bin", .{k[0]})), b.fmt("{s}.bin", .{k[0]}));
    }
    // nGEN generator, built for the host and run at build time for the two prompt GEMM families (groups of 64 and 128).
    const gen = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O1", "-fno-operator-names", b.fmt("-I{s}", .{ngen_include}), "-o" });
    const gen_bin = gen.addOutputFileArg("w4_ngen");
    gen.addFileArg(b.path("zig/tools/xpu/w4_ngen.cpp"));
    for ([_][]const u8{ "w4_prompt.hpp", "w4a8_prompt.hpp" }) |h| gen.addFileInput(b.path(b.fmt("zig/tools/xpu/{s}", .{h})));
    for ([_]struct { []const u8, []const u8, []const []const u8 }{
        .{ "ngen_wide", "64", &.{"b3"} },
        .{ "ngen_narrow", "64", &.{ "s4", "asplit" } },
        .{ "ngen_wide128", "128", &.{"b3"} },
        .{ "ngen_narrow128", "128", &.{ "s4", "asplit" } },
        .{ "ngen_swiglu", "64", &.{ "b3", "swiglu" } },
        .{ "ngen_swiglu128", "128", &.{ "b3", "swiglu" } },
        .{ "ngen_geglu", "64", &.{ "b3", "geglu" } },
        .{ "ngen_a8", "128", &.{"a8"} },
        .{ "ngen_a8_swiglu", "128", &.{ "a8", "swiglu" } },
        .{ "ngen_a4", "128", &.{"a4"} },
    }) |z| {
        const run = std.Build.Step.Run.create(b, b.fmt("nGEN {s}", .{z[0]}));
        run.addFileArg(gen_bin);
        run.addArgs(&.{ "xe2", z[1] });
        const out = run.addOutputFileArg(b.fmt("{s}.zebin", .{z[0]}));
        run.addArgs(z[2]);
        _ = files.addCopyFile(out, b.fmt("{s}.zebin", .{z[0]}));
    }
    const embedded = b.createModule(.{ .root_source_file = files.add("xpu_qwen27_kernels.zig", b.fmt("{s}{s}", .{
        if (tp_ranks == 2) "pub const drafter_tp align(8) = @embedFile(\"drafter_tp.bin\").*;\n" else "",
        \\pub const glue align(8) = @embedFile("glue.bin").*;
        \\pub const decode align(8) = @embedFile("decode.bin").*;
        \\pub const k1 align(8) = @embedFile("k1.bin").*;
        \\pub const gemm align(8) = @embedFile("gemm.bin").*;
        \\pub const gemm128 align(8) = @embedFile("gemm128.bin").*;
        \\pub const gdn align(8) = @embedFile("gdn.bin").*;
        \\pub const drafter align(8) = @embedFile("drafter.bin").*;
        \\pub const ngen_wide align(8) = @embedFile("ngen_wide.zebin").*;
        \\pub const ngen_narrow align(8) = @embedFile("ngen_narrow.zebin").*;
        \\pub const ngen_wide128 align(8) = @embedFile("ngen_wide128.zebin").*;
        \\pub const ngen_narrow128 align(8) = @embedFile("ngen_narrow128.zebin").*;
        \\pub const ngen_swiglu align(8) = @embedFile("ngen_swiglu.zebin").*;
        \\pub const ngen_swiglu128 align(8) = @embedFile("ngen_swiglu128.zebin").*;
        \\pub const ngen_a8 align(8) = @embedFile("ngen_a8.zebin").*;
        \\pub const ngen_a8_swiglu align(8) = @embedFile("ngen_a8_swiglu.zebin").*;
        \\pub const ngen_a4 align(8) = @embedFile("ngen_a4.zebin").*;
        \\pub const sage_glue align(8) = @embedFile("sage_glue.bin").*;
        \\pub const sage_decode align(8) = @embedFile("sage_decode.bin").*;
        \\pub const sage_moe align(8) = @embedFile("sage_moe.bin").*;
        \\pub const qwen_vision align(8) = @embedFile("qwen_vision.bin").*;
        \\
    })) });
    // Gemma 4 E4B: its glue, and the Qwen GEMMs (same files, same bits)
    const embedded_g4 = b.createModule(.{ .root_source_file = files.add("xpu_gemma4_kernels.zig",
        \\pub const glue align(8) = @embedFile("glue_g4s.bin").*; // head 256: the sliding layers
        \\pub const glue_g align(8) = @embedFile("glue_g4g.bin").*; // head 512: the global layers
        \\pub const rows align(8) = @embedFile("rows_g4s.bin").*; // the same at 128 registers: the prompt-rows attention only
        \\pub const rows_g align(8) = @embedFile("rows_g4g.bin").*;
        \\pub const gemm align(8) = @embedFile("gemm.bin").*;
        \\pub const ngen_wide align(8) = @embedFile("ngen_wide.zebin").*;
        \\pub const ngen_narrow align(8) = @embedFile("ngen_narrow.zebin").*;
        \\pub const ngen_geglu align(8) = @embedFile("ngen_geglu.zebin").*; // gate_up with the GeGLU epilogue
        \\
    ) });
    const ze_mod = zeModule(b, target, optimize);
    // one `core` for these trees: the Gemma 4 config, the safetensors readers, Nemotron, the CLI and server
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    {
        const g4 = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/xpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
        g4.addImport("xpu_gemma4_kernels", embedded_g4);
        g4.addImport("core", core);
        g4.addImport("ze", ze_mod);
        const g4_exe = b.addExecutable(.{ .name = "xpu-gemma4", .root_module = g4 });
        b.step("xpu-gemma4", "Build the Gemma 4 E4B XPU runner (needs ocloc, a C++17 compiler and nGEN headers)").dependOn(&b.addInstallArtifact(g4_exe, .{}).step);
    }
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    module.addImport("xpu_qwen27_kernels", embedded);
    // the device runners' own `core` at `optimize` (Qwen27's FP64 references and headers, Nemotron's binaries)
    const runner_core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner_core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    module.addImport("core", runner_core);
    module.addImport("ze", ze_mod);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nem = nemotronModules(b, target, optimize, ocloc, device, dpas, ze_mod, runner_core, lanes);
    module.addImport("xpu", nem.xpu);
    const exe = b.addExecutable(.{ .name = "xpu-qwen27", .root_module = module });
    const install = b.addInstallArtifact(exe, .{});
    b.step("xpu-qwen27", "Build the Qwen3.8-27B XPU forward runner (needs ocloc, a C++17 compiler and nGEN headers)").dependOn(&install.step);
    const sage_mod = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/sage_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    var sage_it = module.import_table.iterator();
    while (sage_it.next()) |e| sage_mod.addImport(e.key_ptr.*, e.value_ptr.*);
    sage_mod.addImport("sage_config", b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/sage_config.zig"), .target = target, .optimize = optimize }));
    sage_mod.addImport("sage_load", b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/sage_load.zig"), .target = target, .optimize = optimize }));
    const sage_install = b.addInstallArtifact(b.addExecutable(.{ .name = "xpu-sage", .root_module = sage_mod }), .{});
    b.step("xpu-sage", "Build the Sage-A1.5 (qwen3_5_moe) XPU runner, first form (MLX 4/64, decode kernels)").dependOn(&sage_install.step);
    // device tests: a skip message and exit 0 without a Level Zero device or a model (`xpu-tests` is Nemotron's, -Dxpu)
    const tests_step = b.step("xpu-qwen27-tests", "Build the 27B XPU device tests (tf-xpu-qwen27-means-test, -seam-test, -close-test, -k1-bench, -tree-test)");
    for ([_][2][]const u8{
        .{ "tf-xpu-qwen27-means-test", "zig/src/xpu/means_test.zig" },
        .{ "tf-xpu-qwen27-seam-test", "zig/src/xpu/seam_test.zig" },
        .{ "tf-xpu-qwen27-close-test", "zig/src/xpu/close_test.zig" },
        .{ "tf-xpu-qwen27-k1-bench", "zig/src/xpu/k1_bench.zig" },
        .{ "tf-xpu-qwen27-tree-test", "zig/src/xpu/tree_test.zig" },
    }) |t| {
        const m = b.createModule(.{ .root_source_file = b.path(t[1]), .target = target, .optimize = optimize, .link_libc = true });
        m.addImport("xpu_qwen27_kernels", embedded);
        m.addImport("ze", ze_mod);
        m.addImport("xpu", nem.xpu);
        m.addImport("core", runner_core); // the drafter's EXL3 head (core/exl3_format.zig)
        tests_step.dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = t[0], .root_module = m }), .{}).step);
    }

    // tensorfold-native with the XPU engines (native/xpu.zig), as the CUDA build's `native` step makes it.
    const qwen27 = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen27.addImport("xpu_qwen27_kernels", embedded);
    qwen27.addImport("ze", ze_mod);
    qwen27.addImport("xpu", nem.xpu);
    qwen27.addImport("sage_config", sage_mod.import_table.get("sage_config").?);
    qwen27.addImport("sage_load", sage_mod.import_table.get("sage_load").?);
    qwen27.addImport("core", runner_core);
    // the runner's open rules (one runner at a time, every option back to the process's own), run on this host
    const host = b.graph.host.result;
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) tests_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen27 })).step);
    // the same tests compiled and not run: no card is opened, so a lane can check a test without a window
    const tests_build = b.step("xpu-qwen27-tests-build", "Compile the 27B XPU tests, do not run them (no card)");
    // the rank mode word's bit layout: a pure host test, run by name only, so no runner opens and no card is touched
    const mode_step = b.step("test-mode-word", "The TP 2 rank mode word's bit layout (no card)");
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) mode_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen27, .filters = &.{"mode word:"} })).step);
    // XPU_DRAFT_WIN (lane p40-roundcost): the drafter's window start and the flag's text, no card
    const win_step = b.step("test-draft-win", "The drafter window flag XPU_DRAFT_WIN (no card)");
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) win_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen27, .filters = &.{"draft win:"} })).step);
    // MLX 4/64 rank slices (lane p36-mlx-tp2): pure host slicing and a CPU reference matvec, no card
    const mlx_step = b.step("test-mlx-tp-slice", "MLX 4/64 TP rank slices: concat for column splits, sum for row splits (no card)");
    const mlx_mod = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/mlx_tp.zig"), .target = target, .optimize = optimize });
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) mlx_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mlx_mod })).step);
    // XPU_MULTI_TREE: the shared tree round's host fill and the merge's per-row used (lane p33-multitree), no card
    const mt_step = b.step("test-k1-mt", "The shared tree round's host fill and merge per-row used (no card)");
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) mt_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen27, .filters = &.{"k1 mt:"} })).step);
    // K1 coverage (lane p39-k1cover): the attention path of every stream count and window split, no card
    const cover_step = b.step("test-k1-cover", "K1 serves streams 1 to 6 and every window split of a 16-row round (no card)");
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) cover_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen27, .filters = &.{"k1 cover:"} })).step);
    // K1's slice merge (lane p39-k1cover): the exp2 merge against the one-pass softmax, no card
    const merge_step = b.step("test-k1-merge", "K1 slice merge: exp2 joins log2-base partials exactly, exp does not (no card)");
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) merge_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = qwen27, .filters = &.{"k1 merge:"} })).step);
    if (target.result.cpu.arch == host.cpu.arch and target.result.os.tag == host.os.tag) tests_build.dependOn(&b.addTest(.{ .root_module = qwen27 }).step);
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "xpu", .module = nem.xpu }, .{ .name = "nemotron_xpu", .module = nem.nemotron }, .{ .name = "lfm3_xpu", .module = lfm3Module(b, target, optimize, nem.xpu) }, .{ .name = "core", .module = core } } });
    // the Qwen3.8-27B runner and Gemma 4 E4B (DPAS, nGEN) only with DPAS: a Gen12LP server serves Nemotron and LFM3
    const engine_options = b.addOptions();
    engine_options.addOption(bool, "qwen27", dpas);
    engine_options.addOption(bool, "gemma4", dpas);
    engines.addOptions("xpu_engine_options", engine_options);
    if (dpas) {
        engines.addImport("xpu_qwen27", qwen27);
        const gemma4 = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/xpu_engine.zig"), .target = target, .optimize = optimize, .link_libc = true });
        gemma4.addImport("xpu_gemma4_kernels", embedded_g4);
        gemma4.addImport("core", core);
        gemma4.addImport("ze", ze_mod);
        engines.addImport("gemma4_xpu", gemma4);
    }
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const checkpoint_cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, .ReleaseSafe, core), .{ .name = "core", .module = core } } });
    const server = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = checkpoint_cli } },
    }) });
    server.root_module.addOptions("build_options", build_options);
    _ = @import("server.zig").addTo(b, server.root_module, target, .ReleaseSafe);
    const install_server = b.addInstallArtifact(server, .{ .dest_dir = .{ .override = .{ .custom = "xpu/bin" } } });
    b.step("xpu-native", "tensorfold-native with the XPU engines into zig-out/xpu/bin").dependOn(&install_server.step);
    // Nemotron's device binaries: the same modules over the runners' `core`, the SPIR-V images shared
    var runners = family(b, target, optimize, runtime(b, target, optimize, &nem.images, ze_mod, runner_core, dpas), runner_core, lanes);
    runners.images = nem.images;
    nemotronSteps(b, target, optimize, runners);
    const with_contract = b.option(bool, "xpu-contract", "Build the XPU kernel contract's limits audit (needs ocloc for the objects)") orelse false;
    const images = b.option([]const u8, "xpu-images", "A directory of zebins made on a host with ocloc: glue.bin, contract_controls.bin, contract_gate_controls.bin");
    if (with_contract) xpu_contract.steps(b, target, optimize, ocloc, device, images);
}

/// One SPIR-V image: the .cl in zig/kernels/xpu (`src`, else `name`) and its ocloc options (dot/matrix kernels: CL3.0).
const Kernel = struct { name: []const u8, src: ?[]const u8 = null, options: []const u8 = "", include: bool = false };

const cl3 = "-cl-std=CL3.0";

/// An explicit list: other files in the folder never join the build. The SPIR-V is device independent.
const kernels = [_]Kernel{
    .{ .name = "attn", .options = cl3 },
    .{ .name = "basic", .options = cl3 },
    .{ .name = "moe", .options = cl3 },
    .{ .name = "mamba" },
    .{ .name = "glue" },
    .{ .name = "qmv4" },
    .{ .name = "vadd" },
    .{ .name = "nem_rows", .options = cl3 },
    .{ .name = "nem_rows_r1", .src = "nem_rows", .options = cl3 ++ " -DMAXR=1" },
    .{ .name = "nem_rows_r2", .src = "nem_rows", .options = cl3 ++ " -DMAXR=2" },
    .{ .name = "nem_rows_r4", .src = "nem_rows", .options = cl3 ++ " -DMAXR=4" },
    .{ .name = "nem_rows_r8", .src = "nem_rows", .options = cl3 ++ " -DMAXR=8" },
    .{ .name = "nem_prow", .options = cl3 },
    .{ .name = "nem_prow_gen12", .src = "nem_prow", .options = cl3 ++ " -DPO=4" },
    .{ .name = "nem_attn_dec", .options = cl3 },
    .{ .name = "lfm3", .options = cl3 },
    .{ .name = "lfm3_mv", .options = cl3 },
    .{ .name = "sage_exl3", .options = cl3 },
    .{ .name = "exl3", .options = cl3 ++ " -DEXL3_ABS" },
} ++ exl3Matvec() ++ exl3Moe();

/// Fused routed MLP images: one a width of exl3_variants.moe.
fn exl3Moe() [exl3_variants.moe.bits.len]Kernel {
    var out: [exl3_variants.moe.bits.len]Kernel = undefined;
    inline for (exl3_variants.moe.bits, &out) |b, *k| k.* = .{ .name = exl3_variants.moe.name(b), .src = "exl3", .options = exl3_variants.moe.options(b) };
    return out;
}

/// EXL3 matvec images: one a (bits, rows, NS, HS) of src/xpu/exl3_variants.zig.
fn exl3Matvec() [exl3_variants.all.len]Kernel {
    var out: [exl3_variants.all.len]Kernel = undefined;
    inline for (exl3_variants.all, &out) |v, *k| k.* = .{ .name = exl3_variants.name(v), .src = "exl3", .options = exl3_variants.options(v) };
    return out;
}

/// The runtime module; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const std.Build.LazyPath, ze: *std.Build.Module, core: *std.Build.Module, dpas: bool) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", images.len == kernels.len);
    options.addOption(u32, "tp", tp_ranks);
    options.addOption(bool, "dpas", dpas);
    const xpu = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    xpu.addOptions("kernel_options", options);
    xpu.addImport("ze", ze);
    xpu.addImport("core", core);
    if (images.len == kernels.len) for (kernels, images) |k, image| xpu.addAnonymousImport(b.fmt("spv_{s}", .{k.name}), .{ .root_source_file = image });
    return xpu;
}

const Modules = struct { xpu: *std.Build.Module, core: *std.Build.Module, nemotron: *std.Build.Module, images: [kernels.len]std.Build.LazyPath = undefined };

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) Modules {
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("xpu", xpu);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    return .{ .xpu = xpu, .core = core, .nemotron = nemotron };
}

/// Nemotron's SPIR-V images (built only when a step needs them), its runtime and its family module.
fn nemotronModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8, dpas: bool, ze: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) Modules {
    var images: [kernels.len]std.Build.LazyPath = undefined;
    for (kernels, &images) |k, *image| image.* = spirv(b, ocloc, device, dpas, k);
    var m = family(b, target, optimize, runtime(b, target, optimize, &images, ze, core, dpas), core, lanes);
    m.images = images;
    return m;
}

fn cliModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module, mods: Modules) *std.Build.Module {
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/xpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("xpu", xpu);
    cli.addImport("core", mods.core);
    cli.addImport("nemotron_xpu", mods.nemotron);
    return cli;
}

/// ocloc -spv_only on one kernel source: `name`.spv in its own output directory.
fn spirv(b: *std.Build, ocloc: []const u8, device: []const u8, dpas: bool, k: Kernel) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ ocloc, "compile", "-device", device, "-spv_only", "-output", k.name, "-output_no_suffix", "-q" });
    // ocloc builds from a copy of the source, so the header directory is named (the Run step's cwd is the build root)
    const options = if (dpas) k.options else b.fmt("{s} -DTF_XPU_NO_DPAS", .{k.options}); // Gen12LP: the kernels' plain fallbacks
    if (options.len > 0) run.addArgs(&.{ "-options", if (k.include) b.fmt("{s} -I zig/kernels/xpu", .{options}) else options });
    _ = k.include;
    run.addArg("-file");
    run.addFileArg(b.path(b.fmt("zig/kernels/xpu/{s}.cl", .{k.src orelse k.name})));
    run.addArg("-out_dir");
    const out = run.addOutputDirectoryArg(b.fmt("{s}_spirv", .{k.name}));
    return out.path(b, b.fmt("{s}.spv", .{k.name}));
}

/// Whether ocloc's `device` has DPAS (XMX): not Gen12LP (Tiger Lake, Rocket Lake, Alder Lake, DG1).
fn hasDpas(device: []const u8) bool {
    for ([_][]const u8{ "tgllp", "rkl", "adl", "dg1", "gen12lp" }) |p| if (std.mem.startsWith(u8, device, p)) return false;
    return true;
}

/// LFM3's forward and engine on the runtime module `xpu`.
fn lfm3Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm3/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("xpu", xpu);
    const core = xpu.import_table.get("core") orelse @panic("lfm3Module: the xpu module imports core");
    const wf = b.addWriteFiles();
    const rope = wf.add("rope_freq.zig", "const r = @import(\"core\").rope_freq;\npub const inverseFrequencies = r.inverseFrequencies;\npub const proportional = r.proportional;\n");
    m.addImport("rope_freq", b.createModule(.{ .root_source_file = rope, .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} }));
    return m;
}

/// The standalone op and model tests, each its own program `tf-xpu-<name>-test` (step `xpu-tests`).
const test_programs = [_][]const u8{
    "panic", "exl3", "nem_rows", "nem_prefill", "nem_gen", "nem_mtp", "nem_kl", "nem_norm_bench", "nem_mv_bench", "nem_slot", "nem_format", "nem_seam", "nem_attn_dec", "nem_fp64", "memory", "lfm3_conv", "lfm3_rope", "lfm3_attn", "lfm3_norm", "lfm3_fwd", "lfm3_bench", "lfm3_rows", "lfm3_streams", "lfm3_chunks", "lfm3_pgemm",
};

/// `-Dxpu` (Linux): Nemotron's SPIR-V images, `tensorfold-xpu` and `tf-xpu-test`; needs ocloc, not nvcc or CUDA.
fn nemotronSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Modules) void {
    const enabled = b.option(bool, "xpu", "Build Nemotron on the Intel GPU into the install step (needs ocloc)") orelse false;
    if (!enabled or target.result.os.tag != .linux) return;
    const spirv_step = b.step("xpu-spirv", "Compile the Intel GPU kernels to SPIR-V and install them");
    for (kernels, mods.images) |k, image| spirv_step.dependOn(&b.addInstallFile(image, b.fmt("spirv/{s}.spv", .{k.name})).step);
    const xpu = mods.xpu;
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold-xpu", .root_module = cliModule(b, target, optimize, xpu, mods) }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/xpu/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("xpu", xpu);
    runner.addImport("core", mods.core);
    runner.addImport("nemotron_xpu", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tf-xpu-test", .root_module = runner }));
    const tests_step = b.step("xpu-tests", "Build the standalone Intel GPU op and model tests (tf-xpu-<name>-test)");
    const lfm3 = lfm3Module(b, target, optimize, xpu);
    for (test_programs) |name| {
        const mod = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/tests/xpu/{s}.zig", .{name})), .target = target, .optimize = optimize, .link_libc = true });
        mod.addImport("xpu", xpu);
        mod.addImport("core", mods.core);
        mod.addImport("nemotron_xpu", mods.nemotron);
        mod.addImport("lfm3_xpu", lfm3);
        tests_step.dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = b.fmt("tf-xpu-{s}-test", .{name}), .root_module = mod }), .{}).step);
    }
}

/// Host unit tests of the Intel GPU runtime, the Nemotron family on it and its CLI (no GPU, no ocloc), on any host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .optimize = .debug, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, host));
    const xpu = runtime(b, host, .debug, &.{}, zeModule(b, host, .debug), core, true);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const mods = family(b, host, .debug, xpu, core, lanes);
    const ze = xpu.import_table.get("ze").?;
    const empty = b.createModule(.{ .root_source_file = b.addWriteFiles().add("draft_xpu_images.zig", "pub const drafter = [_]u8{};\npub const glue_g = [_]u8{};\npub const glue = [_]u8{};\npub const rows = [_]u8{};\npub const rows_g = [_]u8{};\npub const decode = [_]u8{};\npub const gemm = [_]u8{};\npub const gdn = [_]u8{};\npub const ngen_wide = [_]u8{};\npub const ngen_narrow = [_]u8{};\npub const gemm128 = [_]u8{};\npub const ngen_wide128 = [_]u8{};\npub const ngen_narrow128 = [_]u8{};\npub const ngen_swiglu = [_]u8{};\npub const ngen_swiglu128 = [_]u8{};\npub const ngen_a8 = [_]u8{};\npub const ngen_a8_swiglu = [_]u8{};\npub const ngen_a4 = [_]u8{};\npub const ngen_geglu = [_]u8{};\n") });
    const qwen = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "ze", .module = ze }, .{ .name = "core", .module = core }, .{ .name = "xpu_qwen27_kernels", .module = empty } } });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/src/native/draft_sources_xpu_test.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api }, .{ .name = "xpu", .module = xpu }, .{ .name = "xpu_qwen27", .module = qwen }, .{ .name = "nemotron_xpu", .module = mods.nemotron } } });
    const gemma_native = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/xpu_engine.zig"), .target = host, .link_libc = true });
    gemma_native.addImport("core", mods.core);
    gemma_native.addImport("ze", ze);
    gemma_native.addImport("xpu_gemma4_kernels", empty);
    sources.addImport("gemma4_xpu", gemma_native);
    sources.addImport("lfm3_xpu", lfm3Module(b, b.graph.host, .debug, xpu));
    // the host test checks every family's admission, so it builds with both DPAS engines
    const source_options = b.addOptions();
    source_options.addOption(bool, "qwen27", true);
    source_options.addOption(bool, "gemma4", true);
    sources.addOptions("xpu_engine_options", source_options);
    const inventory = b.step("test-xpu-refusal-host", "Unknown-model inventory with and without DPAS, without devices");
    inventory.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = sources, .filters = &.{"capability audit: XPU unknown-model"} })).step);
    const no_dpas = b.createModule(.{ .root_source_file = b.path("zig/src/native/draft_sources_xpu_test.zig"), .target = host, .link_libc = true });
    var imports = sources.import_table.iterator();
    while (imports.next()) |entry| no_dpas.addImport(entry.key_ptr.*, entry.value_ptr.*);
    const no_dpas_options = b.addOptions();
    no_dpas_options.addOption(bool, "qwen27", false);
    no_dpas_options.addOption(bool, "gemma4", false);
    no_dpas.addOptions("xpu_engine_options", no_dpas_options);
    inventory.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = no_dpas, .filters = &.{"capability audit: XPU unknown-model"} })).step);
    const source_tests = b.addRunArtifact(b.addTest(.{ .root_module = sources, .filters = &.{"draft sources:"} }));
    b.step("test-slide-xpu-host", "Unsupported XPU learning refuses before device or checkpoint access").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = sources, .filters = &.{"slide admission:"} })).step);
    // the Gemma 4 runner's shard header reader on core (gemma4/xpu_dev.zig), so its malformed inputs run
    const dev = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/xpu_dev.zig"), .target = host, .optimize = .debug, .link_libc = true });
    dev.addImport("core", mods.core);
    dev.addImport("ze", ze);
    const dev_tests = b.addRunArtifact(b.addTest(.{ .root_module = dev }));
    const draft_step = b.step("test-draft-xpu-host", "XPU draft source metadata and the Gemma 4 shard reader without devices or kernel compilation");
    draft_step.dependOn(&source_tests.step);
    draft_step.dependOn(&dev_tests.step);
    step.dependOn(&source_tests.step);
    step.dependOn(&dev_tests.step);
    // xpu/safetensors.zig's own tests, on the same `core`
    const st_tests = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/safetensors.zig"), .target = host, .imports = &.{.{ .name = "core", .module = core }} });
    for ([_]*std.Build.Module{ xpu, mods.nemotron, cliModule(b, host, .debug, xpu, mods), zeModule(b, host, .debug), st_tests }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    // the Gemma 4 E4B runner's admission, over the family config (which takes `core` by name)
    const gemma4 = b.createModule(.{ .root_source_file = b.path("zig/src/families/gemma4/xpu_config.zig"), .target = host, .optimize = .debug });
    gemma4.addImport("core", core);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = gemma4 })).step);
    // the Qwen27 runner's GDN FP64 reference (xpu/fp64_gdn.zig), which only the device runner otherwise compiles
    const gdn = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/fp64_gdn.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "ze", .module = ze }, .{ .name = "core", .module = core }, .{ .name = "xpu_qwen27_kernels", .module = empty } } });
    const gdn_tests = b.addRunArtifact(b.addTest(.{ .root_module = gdn }));
    b.step("test-fp64-gdn-xpu-host", "The XPU GDN FP64 reference's bf16 rounding without devices or ocloc").dependOn(&gdn_tests.step);
    step.dependOn(&gdn_tests.step);
    // the Qwen27 runner's config.json check (xpu/qwen27_config.zig), which Runner.init makes before any load
    const shape = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27_config.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const shape_tests = b.addRunArtifact(b.addTest(.{ .root_module = shape }));
    b.step("test-qwen27-config-xpu-host", "The XPU 27B runner's config.json shape and layer pattern without devices or ocloc").dependOn(&shape_tests.step);
    step.dependOn(&shape_tests.step);
}

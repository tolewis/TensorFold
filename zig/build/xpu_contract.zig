//! The XPU kernel contract's limits audit: the compiled images' .ze_info and gate 5, with no GPU.
const std = @import("std");
const kernel_contract = @import("kernel_contract.zig");
const inkling2 = @import("xpu_inkling2.zig");

/// `-Dxpu-contract=true`: tf-xpu-kernel-contract, the objects it reads, and the no-GPU audit.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8, prebuilt: ?[]const u8) void {
    const core = kernel_contract.module(b, target, optimize);
    const exe = b.addExecutable(.{ .name = "tf-xpu-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = core }},
    }) });
    b.step("tf-xpu-kernel-contract", "Build the XPU limits audit (no GPU, no ocloc)").dependOn(&b.addInstallArtifact(exe, .{}).step);

    // ocloc writes a native zebin beside the SPIR-V; the audit reads the zebin's .ze_info
    const files = b.addWriteFiles();
    const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread";
    const glue = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    glue.addFileArg(b.path("zig/kernels/xpu/qwen27/glue.cl"));
    glue.addArgs(&.{ "-device", device, "-options", b.fmt("{s} -I zig/kernels/xpu/qwen27", .{opts}), "-output_no_suffix", "-output", "glue", "-out_dir" });
    glue.setCwd(b.path("."));
    for ([_][]const u8{ "glue_attn.cl", "glue_quant.cl" }) |part| glue.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
    const glue_dir = glue.addOutputDirectoryArg("ocloc-glue");
    _ = files.addCopyFile(glue_dir.path(b, "glue.bin"), b.fmt("{s}/glue.bin", .{device}));
    const tp_obj = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    tp_obj.addFileArg(b.path("zig/kernels/xpu/tp.cl"));
    tp_obj.addArgs(&.{ "-device", device, "-options", opts, "-output_no_suffix", "-output", "tp", "-out_dir" });
    tp_obj.setCwd(b.path("."));
    const tp_dir = tp_obj.addOutputDirectoryArg("ocloc-tp");
    _ = files.addCopyFile(tp_dir.path(b, "tp.bin"), b.fmt("{s}/tp.bin", .{device}));
    const ctl = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    ctl.addFileArg(b.path("zig/kernels/xpu/contract_controls.cl"));
    ctl.addArgs(&.{ "-device", device, "-options", opts, "-output_no_suffix", "-output", "contract_controls", "-out_dir" });
    ctl.setCwd(b.path("."));
    const ctl_dir = ctl.addOutputDirectoryArg("ocloc-controls");
    _ = files.addCopyFile(ctl_dir.path(b, "contract_controls.bin"), b.fmt("{s}/contract_controls.bin", .{device}));
    for (contract_images ++ inkling2_images ++ ink2_dense_images ++ ink2_attn_images ++ ink2_attn_controls) |src| {
        _ = files.addCopyFile(object(b, ocloc, device, src), b.fmt("{s}/{s}.bin", .{ device, src.name }));
    }
    for (inkling2.names, inkling2.images(b, ocloc, device)) |name, image| _ = files.addCopyFile(image, b.fmt("{s}/{s}.bin", .{ device, name }));
    inkling2.controlsTest(b);
    const install = b.addInstallDirectory(.{
        .source_dir = files.getDirectory(),
        .install_dir = .prefix,
        .install_subdir = "xpu-contract",
    });
    b.step("xpu-kernel-contract-objects", "Compile every registered XPU image and the controls to zebins (needs ocloc, no GPU)").dependOn(&install.step);

    const audit = b.addRunArtifact(exe);
    audit.addArg("limits");
    audit.addDirectoryArg(files.getDirectory());
    b.step("xpu-kernel-contract-limits", "Every registered XPU launch against the BMG card and its compiled images (no GPU)").dependOn(&audit.step);
    runner(b, target, optimize, ocloc, device, prebuilt);
}

/// The device runner: gates 1 to 4 and gate 5 live on a card, with the images linked in; `prebuilt` is -Dxpu-images.
pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: []const u8, device: []const u8, prebuilt: ?[]const u8) void {
    const core = kernel_contract.module(b, target, optimize);
    const ze = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false); // the runner links its own images
    options.addOption(bool, "dpas", true);
    const xpu = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    xpu.addOptions("kernel_options", options);
    xpu.addImport("ze", ze);
    xpu.addImport("core", core);

    // Inkling 2 Mini's DSpark: the host wrappers the contract file launches through, and the kernels' source text
    const dspark_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/dspark_kernels.zig"), .target = target, .optimize = optimize });
    dspark_host.addImport("xpu", xpu);
    // Mini's DFlash 2: the same two modules
    const dflash2_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/dflash2_kernels.zig"), .target = target, .optimize = optimize });
    dflash2_host.addImport("xpu", xpu);
    const dflash2_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/xpu/inkling2/dflash2_sources.zig"), .target = target, .optimize = optimize });
    // Inkling 2 Mini's dense projections and norm: the host wrapper the contract file launches through
    const dense_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/dense.zig"), .target = target, .optimize = optimize });
    dense_host.addImport("xpu", xpu);
    // The same over Q8_0-style weights, and the host encoder both the loader and the contract's reference use
    const dense_q8_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/dense_q8.zig"), .target = target, .optimize = optimize });
    dense_q8_host.addImport("xpu", xpu);
    const q8_format = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/q8.zig"), .target = target, .optimize = optimize });
    const dspark_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/xpu/inkling2/dspark_sources.zig"), .target = target, .optimize = optimize });

    const files = b.addWriteFiles();
    if (prebuilt) |dir| {
        for ([_][]const u8{ "glue", "tp", "contract_controls", "contract_gate_controls", "decode", "contract_attn_upstream", "contract_merge_upstream", "contract_inkling2_loop", "contract_inkling2_loop_controls", "inkling2_dspark", "contract_inkling2_dspark_controls", "inkling2_dflash2", "contract_inkling2_dflash2_controls", "contract_inkling2_experts", "contract_inkling2_controls", "inkling2_dense", "contract_inkling2_dense_store", "contract_inkling2_dense_sums", "inkling2_dense_q8", "contract_inkling2_dense_q8_store", "contract_inkling2_dense_q8_sums", "contract_inkling2_dense_q8_scale", "inkling2_prefill_experts", "contract_inkling2_prefill_experts_controls" } ++ inkling2.names) |name| {
            _ = files.addCopyFile(.{ .cwd_relative = b.fmt("{s}/{s}.bin", .{ dir, name }) }, b.fmt("{s}.bin", .{name}));
        }
        for (ink2_attn_images ++ ink2_attn_controls) |src| {
            _ = files.addCopyFile(.{ .cwd_relative = b.fmt("{s}/{s}.bin", .{ dir, src.name }) }, b.fmt("{s}.bin", .{src.name}));
        }
    } else {
        const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread";
        const glue = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
        glue.addFileArg(b.path("zig/kernels/xpu/qwen27/glue.cl"));
        glue.addArgs(&.{ "-device", device, "-options", b.fmt("{s} -I zig/kernels/xpu/qwen27", .{opts}), "-output_no_suffix", "-output", "glue", "-out_dir" });
        glue.setCwd(b.path("."));
        for ([_][]const u8{ "glue_attn.cl", "glue_quant.cl" }) |part| glue.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
        const glue_dir = glue.addOutputDirectoryArg("ocloc-glue-run");
        _ = files.addCopyFile(glue_dir.path(b, "glue.bin"), "glue.bin");
        for ([_][3][]const u8{
            .{ "tp", "zig/kernels/xpu/tp.cl", "" },
            .{ "contract_controls", "zig/kernels/xpu/contract_controls.cl", "" },
            .{ "contract_gate_controls", "zig/kernels/xpu/contract_gate_controls.cl", "" },
            .{ "contract_inkling2_experts", "zig/kernels/xpu/inkling2/experts.cl", inkling2_options },
            .{ "contract_inkling2_controls", "zig/kernels/xpu/inkling2/contract_experts_controls.cl", inkling2_options },
        }) |pair| {
            const run = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
            run.addFileArg(b.path(pair[1]));
            if (pair[2].len > 0) run.addFileInput(b.path("zig/kernels/xpu/inkling2/experts.cl"));
            run.addArgs(&.{ "-device", device, "-options", b.fmt("{s}{s}", .{ opts, pair[2] }), "-output_no_suffix", "-output", pair[0], "-out_dir" });
            run.setCwd(b.path("."));
            const dir = run.addOutputDirectoryArg(b.fmt("ocloc-{s}-run", .{pair[0]}));
            _ = files.addCopyFile(dir.path(b, b.fmt("{s}.bin", .{pair[0]})), b.fmt("{s}.bin", .{pair[0]}));
        }
        for (contract_images[0..3] ++ contract_images[4..] ++ inkling2_images ++ ink2_dense_images ++ ink2_attn_images ++ ink2_attn_controls) |src| {
            _ = files.addCopyFile(object(b, ocloc, device, src), b.fmt("{s}.bin", .{src.name}));
        }
        for (inkling2.names, inkling2.images(b, ocloc, device)) |name, image| _ = files.addCopyFile(image, b.fmt("{s}.bin", .{name}));
    }
    const images = b.createModule(.{ .root_source_file = files.add("xpu_contract_images.zig",
        \\pub const glue align(8) = @embedFile("glue.bin").*;
        \\pub const tp align(8) = @embedFile("tp.bin").*;
        \\pub const decode align(8) = @embedFile("decode.bin").*;
        \\pub const attn_upstream align(8) = @embedFile("contract_attn_upstream.bin").*;
        \\pub const merge_upstream align(8) = @embedFile("contract_merge_upstream.bin").*;
        \\pub const controls align(8) = @embedFile("contract_controls.bin").*;
        \\pub const gate_controls align(8) = @embedFile("contract_gate_controls.bin").*;
        \\pub const inkling2_loop align(8) = @embedFile("contract_inkling2_loop.bin").*;
        \\pub const inkling2_loop_controls align(8) = @embedFile("contract_inkling2_loop_controls.bin").*;
        \\pub const inkling2_glue align(8) = @embedFile("inkling2_glue.bin").*;
        \\pub const inkling2_runtime align(8) = @embedFile("inkling2_runtime.bin").*;
        \\pub const inkling2_glue_ctl align(8) = @embedFile("inkling2_glue_ctl.bin").*;
        \\pub const inkling2_runtime_ctl align(8) = @embedFile("inkling2_runtime_ctl.bin").*;
        \\pub const inkling2_dspark align(8) = @embedFile("inkling2_dspark.bin").*;
        \\pub const contract_inkling2_dspark_controls align(8) = @embedFile("contract_inkling2_dspark_controls.bin").*;
        \\pub const inkling2_dflash2 align(8) = @embedFile("inkling2_dflash2.bin").*;
        \\pub const contract_inkling2_dflash2_controls align(8) = @embedFile("contract_inkling2_dflash2_controls.bin").*;
        \\pub const ink2_attention align(8) = @embedFile("inkling2_attention.bin").*;
        \\pub const ink2_prompt align(8) = @embedFile("inkling2_prompt.bin").*;
        \\pub const ink2_attention_ctl1 align(8) = @embedFile("inkling2_attention_ctl1.bin").*;
        \\pub const ink2_attention_ctl2 align(8) = @embedFile("inkling2_attention_ctl2.bin").*;
        \\pub const ink2_attention_ctl3 align(8) = @embedFile("inkling2_attention_ctl3.bin").*;
        \\pub const ink2_attention_ctl4 align(8) = @embedFile("inkling2_attention_ctl4.bin").*;
        \\pub const ink2_attention_ctl5 align(8) = @embedFile("inkling2_attention_ctl5.bin").*;
        \\pub const ink2_prompt_ctl6 align(8) = @embedFile("inkling2_prompt_ctl6.bin").*;
        \\pub const inkling2_experts align(8) = @embedFile("contract_inkling2_experts.bin").*;
        \\pub const inkling2_controls align(8) = @embedFile("contract_inkling2_controls.bin").*;
        \\pub const inkling2_dense align(8) = @embedFile("inkling2_dense.bin").*;
        \\pub const contract_inkling2_dense_store align(8) = @embedFile("contract_inkling2_dense_store.bin").*;
        \\pub const contract_inkling2_dense_sums align(8) = @embedFile("contract_inkling2_dense_sums.bin").*;
        \\pub const inkling2_dense_q8 align(8) = @embedFile("inkling2_dense_q8.bin").*;
        \\pub const contract_inkling2_dense_q8_store align(8) = @embedFile("contract_inkling2_dense_q8_store.bin").*;
        \\pub const contract_inkling2_dense_q8_sums align(8) = @embedFile("contract_inkling2_dense_q8_sums.bin").*;
        \\pub const contract_inkling2_dense_q8_scale align(8) = @embedFile("contract_inkling2_dense_q8_scale.bin").*;
        \\pub const inkling2_prefill_experts align(8) = @embedFile("inkling2_prefill_experts.bin").*;
        \\pub const contract_inkling2_prefill_experts_controls align(8) = @embedFile("contract_inkling2_prefill_experts_controls.bin").*;
    ), .target = target, .optimize = optimize });
    // Inkling 2 Mini's loop: the host wrapper, its pure launch plan, and the host float model the device is held to
    const loop_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/loop_plan.zig"), .target = target, .optimize = optimize });
    const loop_model = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/loop_model.zig"), .target = target, .optimize = optimize });
    const loop_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/loop.zig"), .target = target, .optimize = optimize, .link_libc = true });
    loop_host.addImport("xpu", xpu);
    loop_host.addImport("inkling2_loop_plan", loop_plan);
    const ink2_hosts = inkling2.hosts(b, target, optimize, xpu);

    const ink2_attention = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2/xpu/attention.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "xpu", .module = xpu }},
    });
    const experts_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/experts.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "xpu", .module = xpu }} });
    const prefill_experts_host = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/prefill_experts.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "xpu", .module = xpu }} });
    b.step("test-inkling2-xpu-experts-host", "Inkling 2's XPU experts host wrapper: plan and call checks (no GPU)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = experts_host })).step);
    const exe = b.addExecutable(.{ .name = "tf-xpu-contract-run", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/contract_run.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "inkling2_xpu_experts", .module = experts_host },
            .{ .name = "inkling2_prefill_experts_host", .module = prefill_experts_host },
            .{ .name = "ze", .module = ze },
            .{ .name = "xpu_contract_images", .module = images },
            .{ .name = "inkling2_xpu_loop", .module = loop_host },
            .{ .name = "inkling2_loop_model", .module = loop_model },
            .{ .name = "inkling2_xpu_glue", .module = ink2_hosts[0] },
            .{ .name = "inkling2_xpu_runtime", .module = ink2_hosts[1] },
            .{ .name = "inkling2_dspark_host", .module = dspark_host },
            .{ .name = "inkling2_dspark_sources", .module = dspark_sources },
            .{ .name = "inkling2_dflash2_host", .module = dflash2_host },
            .{ .name = "inkling2_dflash2_sources", .module = dflash2_sources },
            .{ .name = "inkling2_xpu_attention", .module = ink2_attention },
            .{ .name = "inkling2_dense_host", .module = dense_host },
            .{ .name = "inkling2_dense_q8_host", .module = dense_q8_host },
            .{ .name = "inkling2_q8", .module = q8_format },
        },
    }) });
    const attn_bench = b.addExecutable(.{ .name = "tf-xpu-inkling2-attention-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_attention_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "ze", .module = ze },
            .{ .name = "xpu_contract_images", .module = images },
            .{ .name = "inkling2_xpu_attention", .module = ink2_attention },
        },
    }) });
    b.step("tf-xpu-inkling2-attention-bench", "Build the Inkling 2 Mini XPU attention bandwidth bench").dependOn(&b.addInstallArtifact(attn_bench, .{}).step);
    b.step("tf-xpu-contract-run", "Build the XPU kernel-contract runner: gates 1 to 4 and gate 5 live on a card").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const dspark_bench = b.addExecutable(.{ .name = "tf-xpu-dspark-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_dspark_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "ze", .module = ze },
            .{ .name = "xpu_contract_images", .module = images },
            .{ .name = "inkling2_dspark_host", .module = dspark_host },
        },
    }) });
    b.step("tf-xpu-dspark-bench", "Build the Mini DSpark XPU kernel timing (GB/s at the served shapes)").dependOn(&b.addInstallArtifact(dspark_bench, .{}).step);
    // The drafter's host files tested without a card; `core` here is only the safetensors reader they use.
    const safetensors = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    const core_reader = b.addWriteFiles();
    const dspark_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2/xpu/dspark_host_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "xpu", .module = xpu },
            .{ .name = "core", .module = b.createModule(.{
                .root_source_file = core_reader.add("core_reader.zig", "pub const safetensors = @import(\"safetensors\");"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "safetensors", .module = safetensors }},
            }) },
        },
    }) });
    b.step("test-inkling2-dspark-xpu-host", "Mini DSpark's XPU host files (kernels, state, weights, fold) without a card").dependOn(&b.addRunArtifact(dspark_tests).step);
    const dense_bench = b.addExecutable(.{ .name = "tf-xpu-inkling2-dense-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_dense_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "ze", .module = ze },
            .{ .name = "xpu_contract_images", .module = images },
            .{ .name = "inkling2_dense_host", .module = dense_host },
        },
    }) });
    b.step("tf-xpu-inkling2-dense-bench", "Build the Mini dense and norm XPU timing (GB/s at the served shapes)").dependOn(&b.addInstallArtifact(dense_bench, .{}).step);
    const dense_q8_bench = b.addExecutable(.{ .name = "tf-xpu-inkling2-dense-q8-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_dense_q8_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "xpu", .module = xpu },
            .{ .name = "ze", .module = ze },
            .{ .name = "xpu_contract_images", .module = images },
            .{ .name = "inkling2_dense_q8_host", .module = dense_q8_host },
            .{ .name = "inkling2_q8", .module = q8_format },
        },
    }) });
    b.step("tf-xpu-inkling2-dense-q8-bench", "Build the Mini Q8_0 dense and head XPU timing (GB/s at the served shapes)").dependOn(&b.addInstallArtifact(dense_q8_bench, .{}).step);
    const experts_bench = b.addExecutable(.{ .name = "tf-xpu-inkling2-experts-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_experts_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "xpu", .module = xpu },
            .{ .name = "inkling2_xpu_experts", .module = experts_host },
            .{ .name = "xpu_contract_images", .module = images },
        },
    }) });
    b.step("tf-xpu-inkling2-experts-bench", "Build the Inkling 2 experts decode bandwidth bench (run it on a card under the GPU lock)").dependOn(&b.addInstallArtifact(experts_bench, .{}).step);
    const run = b.addRunArtifact(exe);
    b.step("xpu-contract-run", "Run the XPU kernel contract on this card (tf-xpu-contract-run [--device N])").dependOn(&run.step);
}

/// Inkling 2's experts: IEEE divides (the per-expert factor is a correctly rounded quotient) and its own include path.
const inkling2_options = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2";

/// `extra` is appended to ocloc's options; `inputs` are files the source includes, so a change rebuilds the object.
const Source = struct { name: []const u8, path: []const u8, extra: []const u8 = "", inputs: []const []const u8 = &.{} };
const contract_images = [_]Source{
    .{ .name = "decode", .path = "zig/kernels/xpu/qwen27/decode.cl" },
    .{ .name = "contract_attn_upstream", .path = "zig/kernels/xpu/contract_attn_upstream.cl" },
    .{ .name = "contract_merge_upstream", .path = "zig/kernels/xpu/contract_merge_upstream.cl" },
    .{ .name = "contract_gate_controls", .path = "zig/kernels/xpu/contract_gate_controls.cl" },
    // Inkling 2 Mini's loop: divide and sqrt correctly rounded, as loop.metal's precise:: calls are; the controls image
    // is the same source with the 1/48 scale and the bf16 zeta dropped.
    .{ .name = "contract_inkling2_loop", .path = "zig/kernels/xpu/inkling2/loop.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt" },
    .{ .name = "contract_inkling2_loop_controls", .path = "zig/kernels/xpu/inkling2/loop.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -DTF_CTL_LOOP_NO_SCALE -DTF_CTL_LOOP_NO_ZETA_ROUND" },
    .{ .name = "contract_inkling2_experts", .path = "zig/kernels/xpu/inkling2/experts.cl", .extra = inkling2_options, .inputs = &.{"zig/kernels/xpu/inkling2/experts.cl"} },
    .{ .name = "contract_inkling2_controls", .path = "zig/kernels/xpu/inkling2/contract_experts_controls.cl", .extra = inkling2_options, .inputs = &.{"zig/kernels/xpu/inkling2/experts.cl"} },
};

/// Inkling 2 Mini's DSpark kernels and their planted controls (the controls file includes the kernels).
const inkling2_images = [_]Source{
    // The prompt chunk's grouped expert GEMMs (plan, gather, four GEMMs, combine) and their planted twins (the controls file includes the source).
    .{ .name = "inkling2_prefill_experts", .path = "zig/kernels/xpu/inkling2/prefill_experts.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    .{
        .name = "contract_inkling2_prefill_experts_controls",
        .path = "zig/kernels/xpu/inkling2/contract_prefill_experts_controls.cl",
        .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2",
        .inputs = &.{"zig/kernels/xpu/inkling2/prefill_experts.cl"},
    },
    .{ .name = "inkling2_dspark", .path = "zig/kernels/xpu/inkling2/dspark.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    .{
        .name = "contract_inkling2_dspark_controls",
        .path = "zig/kernels/xpu/inkling2/contract_dspark_controls.cl",
        .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2",
        .inputs = &.{"zig/kernels/xpu/inkling2/dspark.cl"},
    },
    .{ .name = "inkling2_dflash2", .path = "zig/kernels/xpu/inkling2/dflash2.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    .{
        .name = "contract_inkling2_dflash2_controls",
        .path = "zig/kernels/xpu/inkling2/contract_dflash2_controls.cl",
        .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2",
        .inputs = &.{"zig/kernels/xpu/inkling2/dflash2.cl"},
    },
};

/// Inkling 2 Mini's dense projections and norm, and the two images of their planted controls.
const ink2_dense_options = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2";
const ink2_dense_images = [_]Source{
    .{ .name = "inkling2_dense", .path = "zig/kernels/xpu/inkling2/dense.cl", .extra = ink2_dense_options },
    .{ .name = "contract_inkling2_dense_store", .path = "zig/kernels/xpu/inkling2/contract_dense_controls.cl", .extra = ink2_dense_options ++ " -DDENSE_CTL=1", .inputs = &.{"zig/kernels/xpu/inkling2/dense.cl"} },
    .{ .name = "contract_inkling2_dense_sums", .path = "zig/kernels/xpu/inkling2/contract_dense_controls.cl", .extra = ink2_dense_options ++ " -DDENSE_CTL=2", .inputs = &.{"zig/kernels/xpu/inkling2/dense.cl"} },
    // The Q8_0-weight kernels (dense_q8.cl) and their three planted twins.
    .{ .name = "inkling2_dense_q8", .path = "zig/kernels/xpu/inkling2/dense_q8.cl", .extra = ink2_dense_options },
    .{ .name = "contract_inkling2_dense_q8_store", .path = "zig/kernels/xpu/inkling2/contract_dense_q8_controls.cl", .extra = ink2_dense_options ++ " -DQ8_CTL=1", .inputs = &.{"zig/kernels/xpu/inkling2/dense_q8.cl"} },
    .{ .name = "contract_inkling2_dense_q8_sums", .path = "zig/kernels/xpu/inkling2/contract_dense_q8_controls.cl", .extra = ink2_dense_options ++ " -DQ8_CTL=2", .inputs = &.{"zig/kernels/xpu/inkling2/dense_q8.cl"} },
    .{ .name = "contract_inkling2_dense_q8_scale", .path = "zig/kernels/xpu/inkling2/contract_dense_q8_controls.cl", .extra = ink2_dense_options ++ " -DQ8_CTL=3", .inputs = &.{"zig/kernels/xpu/inkling2/dense_q8.cl"} },
};

/// Inkling 2 Mini attention (OpenCL C on bmg): the attention image and the prompt image.
const ink2_attn_images = [_]Source{
    .{ .name = "inkling2_attention", .path = "zig/kernels/xpu/inkling2/attention.cl" },
    .{ .name = "inkling2_prompt", .path = "zig/kernels/xpu/inkling2/prompt_attention.cl" },
};
/// The same sources with one planted fault each (-DINK2_CTL=n), for the contract's controls.
const ink2_attn_controls = [_]Source{
    .{ .name = "inkling2_attention_ctl1", .path = "zig/kernels/xpu/inkling2/attention.cl", .extra = " -DINK2_CTL=1" },
    .{ .name = "inkling2_attention_ctl2", .path = "zig/kernels/xpu/inkling2/attention.cl", .extra = " -DINK2_CTL=2" },
    .{ .name = "inkling2_attention_ctl3", .path = "zig/kernels/xpu/inkling2/attention.cl", .extra = " -DINK2_CTL=3" },
    .{ .name = "inkling2_attention_ctl4", .path = "zig/kernels/xpu/inkling2/attention.cl", .extra = " -DINK2_CTL=4" },
    .{ .name = "inkling2_attention_ctl5", .path = "zig/kernels/xpu/inkling2/attention.cl", .extra = " -DINK2_CTL=5" },
    .{ .name = "inkling2_prompt_ctl6", .path = "zig/kernels/xpu/inkling2/prompt_attention.cl", .extra = " -DINK2_CTL=6" },
};

fn object(b: *std.Build, ocloc: []const u8, device: []const u8, src: Source) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    run.addFileArg(b.path(src.path));
    const opts = b.fmt("-cl-std=CL3.0 -cl-intel-256-GRF-per-thread -I zig/kernels/xpu/qwen27{s}", .{src.extra});
    run.addArgs(&.{ "-device", device, "-options", opts, "-output_no_suffix", "-output", src.name, "-out_dir" });
    run.setCwd(b.path("."));
    run.addFileInput(b.path("zig/kernels/xpu/qwen27/shape.h"));
    for (src.inputs) |input| run.addFileInput(b.path(input));
    const dir = run.addOutputDirectoryArg(b.fmt("ocloc-contract-{s}", .{src.name}));
    return dir.path(b, b.fmt("{s}.bin", .{src.name}));
}

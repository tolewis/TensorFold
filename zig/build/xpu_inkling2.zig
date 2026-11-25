//! Inkling 2 Mini's XPU kernels in the contract build: the glue and runtime images, their planted-control images
//! (the production source with one edit a kernel, made by a build tool) and the host wrappers the runner links.
const std = @import("std");

/// ocloc's options for these images: IEEE `/` and sqrt, as the Metal source's precise:: calls are.
const options = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread -cl-fp32-correctly-rounded-divide-sqrt";

/// The image names the records and the runner use.
pub const names = [_][]const u8{ "inkling2_glue", "inkling2_runtime", "inkling2_glue_ctl", "inkling2_runtime_ctl", "inkling2_prefill", "inkling2_prefill_ctl" };

fn compile(b: *std.Build, ocloc: []const u8, device: []const u8, name: []const u8, source: std.Build.LazyPath) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
    run.addFileArg(source);
    run.addArgs(&.{ "-device", device, "-options", options, "-output_no_suffix", "-output", name, "-out_dir" });
    run.setCwd(b.path("."));
    const dir = run.addOutputDirectoryArg(b.fmt("ocloc-{s}", .{name}));
    return dir.path(b, b.fmt("{s}.bin", .{name}));
}

fn control(b: *std.Build, gen: *std.Build.Step.Compile, part: []const u8, production: []const u8, out: []const u8) std.Build.LazyPath {
    const run = b.addRunArtifact(gen);
    run.addArgs(&.{part});
    run.addFileArg(b.path(production));
    return run.addOutputFileArg(out);
}

/// The four zebins for `device`, in `names` order.
pub fn images(b: *std.Build, ocloc: []const u8, device: []const u8) [6]std.Build.LazyPath {
    const gen = b.addExecutable(.{ .name = "inkling2-control-gen", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_control_gen.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    }) });
    const glue = "zig/kernels/xpu/inkling2/glue.cl";
    const runtime = "zig/kernels/xpu/inkling2/runtime.cl";
    const prefill = "zig/kernels/xpu/inkling2/prefill.cl";
    return .{
        compile(b, ocloc, device, names[0], b.path(glue)),
        compile(b, ocloc, device, names[1], b.path(runtime)),
        compile(b, ocloc, device, names[2], control(b, gen, "glue", glue, "glue_ctl.cl")),
        compile(b, ocloc, device, names[3], control(b, gen, "runtime", runtime, "runtime_ctl.cl")),
        compile(b, ocloc, device, names[4], b.path(prefill)),
        compile(b, ocloc, device, names[5], control(b, gen, "prefill", prefill, "prefill_ctl.cl")),
    };
}

/// The two host wrappers over the xpu runtime, with the pure launch plans they share with the Metal wrappers.
pub fn hosts(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module) [2]*std.Build.Module {
    const glue_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/glue_plan.zig"), .target = target, .optimize = optimize });
    const runtime_plan = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/runtime_plan.zig"), .target = target, .optimize = optimize });
    return .{
        b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/glue.zig"), .target = target, .optimize = optimize, .imports = &.{
            .{ .name = "xpu", .module = xpu },
            .{ .name = "inkling2_glue_plan", .module = glue_plan },
        } }),
        b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu/runtime.zig"), .target = target, .optimize = optimize, .imports = &.{
            .{ .name = "xpu", .module = xpu },
            .{ .name = "inkling2_runtime_plan", .module = runtime_plan },
        } }),
    };
}

/// `zig build test-inkling2-xpu-controls`: every control applies once and renames only its own kernel (no device).
pub fn controlsTest(b: *std.Build) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/tests/xpu/inkling2_controls.zig"), .target = b.graph.host, .optimize = .Debug });
    m.addAnonymousImport("inkling2_glue_cl", .{ .root_source_file = b.path("zig/kernels/xpu/inkling2/glue.cl") });
    m.addAnonymousImport("inkling2_runtime_cl", .{ .root_source_file = b.path("zig/kernels/xpu/inkling2/runtime.cl") });
    m.addAnonymousImport("inkling2_prefill_cl", .{ .root_source_file = b.path("zig/kernels/xpu/inkling2/prefill.cl") });
    b.step("test-inkling2-xpu-controls", "Inkling 2 Mini's XPU source controls plant one fault each (no device, no ocloc)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}

/// One production image: the source, ocloc's options after the common ones, and the files the source includes.
const Production = struct { name: []const u8, path: []const u8, extra: []const u8 = "", inputs: []const []const u8 = &.{}, grf: u32 = 256 };

/// The kernels the whole forward runs, built as the contract builds them (the same sources and options, no controls).
const production_images = [_]Production{
    .{ .name = "attention", .path = "zig/kernels/xpu/inkling2/attention.cl" },
    .{ .name = "prompt", .path = "zig/kernels/xpu/inkling2/prompt_attention.cl" },
    .{ .name = "glue", .path = "zig/kernels/xpu/inkling2/glue.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt" },
    .{ .name = "experts", .path = "zig/kernels/xpu/inkling2/experts.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    // Multi-row verify only. Same source, GRF 128. The one-row launch keeps `experts`.
    .{ .name = "experts_mrow", .path = "zig/kernels/xpu/inkling2/experts.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2", .grf = 128 },
    .{ .name = "loop", .path = "zig/kernels/xpu/inkling2/loop.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt" },
    .{ .name = "fuse", .path = "zig/kernels/xpu/inkling2/fuse.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt" },
    .{ .name = "runtime", .path = "zig/kernels/xpu/inkling2/runtime.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt" },
    .{ .name = "dense", .path = "zig/kernels/xpu/inkling2/dense.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    // the sampler's candidate selection
    .{ .name = "sample", .path = "zig/kernels/xpu/inkling2/sample.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt" },
    // the drafter's kernels: the engine loads this one only with a drafter
    .{ .name = "dspark", .path = "zig/kernels/xpu/inkling2/dspark.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    .{ .name = "dflash2", .path = "zig/kernels/xpu/inkling2/dflash2.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    .{ .name = "dense_q8", .path = "zig/kernels/xpu/inkling2/dense_q8.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    // the prompt chunk's row-batched glue, grouped expert GEMMs, DPAS projections and attention
    .{ .name = "prefill", .path = "zig/kernels/xpu/inkling2/prefill.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
    .{ .name = "prefill_experts", .path = "zig/kernels/xpu/inkling2/prefill_experts.cl", .extra = " -cl-fp32-correctly-rounded-divide-sqrt -I zig/kernels/xpu/inkling2" },
};

/// The module `inkling2_xpu_images`: every production kernel as a native zebin for `device`, one `align(8)` array each,
/// named as in `production_images`. Pass `ocloc` null for a module of empty arrays (host tests build without a compiler).
pub fn productionModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, ocloc: ?[]const u8, device: []const u8) *std.Build.Module {
    const files = b.addWriteFiles();
    var source: std.ArrayList(u8) = .empty;
    for (production_images) |img| {
        if (ocloc) |tool| {
            const run = b.addSystemCommand(&.{ tool, "compile", "-q", "-file" });
            run.addFileArg(b.path(img.path));
            const opts = b.fmt("-cl-std=CL3.0 -cl-intel-{d}-GRF-per-thread -I zig/kernels/xpu/qwen27{s}", .{ img.grf, img.extra });
            run.addArgs(&.{ "-device", device, "-options", opts, "-output_no_suffix", "-output", img.name, "-out_dir" });
            run.setCwd(b.path("."));
            run.addFileInput(b.path("zig/kernels/xpu/qwen27/shape.h"));
            run.addFileInput(b.path("zig/kernels/xpu/inkling2/experts.cl"));
            const dir = run.addOutputDirectoryArg(b.fmt("ocloc-production-{s}", .{img.name}));
            _ = files.addCopyFile(dir.path(b, b.fmt("{s}.bin", .{img.name})), b.fmt("{s}.bin", .{img.name}));
            source.print(b.allocator, "pub const {s} align(8) = @embedFile(\"{s}.bin\").*;\n", .{ img.name, img.name }) catch @panic("OOM");
        } else {
            source.print(b.allocator, "pub const {s} align(8) = [_]u8{{}};\n", .{img.name}) catch @panic("OOM");
        }
    }
    return b.createModule(.{ .root_source_file = files.add("inkling2_xpu_images.zig", source.items), .target = target, .optimize = optimize });
}

/// The module `inkling2_xpu` and the pure launch plans it takes by name, over the xpu runtime `xpu`, `core` and the lane core.
pub fn familyModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, xpu: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, images_module: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/xpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("xpu", xpu);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("inkling2_xpu_images", images_module);
    for ([_][2][]const u8{
        .{ "inkling2_glue_plan", "zig/src/families/inkling2/metal/glue_plan.zig" },
        .{ "inkling2_runtime_plan", "zig/src/families/inkling2/metal/runtime_plan.zig" },
        .{ "inkling2_loop_plan", "zig/src/families/inkling2/metal/loop_plan.zig" },
    }) |plan| m.addImport(plan[0], b.createModule(.{ .root_source_file = b.path(plan[1]), .target = target, .optimize = optimize }));
    return m;
}

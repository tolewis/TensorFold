//! Inkling 2 Mini's XPU kernels in the contract build: the glue and runtime images, their planted-control images
//! (the production source with one edit a kernel, made by a build tool) and the host wrappers the runner links.
const std = @import("std");

/// ocloc's options for these images: IEEE `/` and sqrt, as the Metal source's precise:: calls are.
const options = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread -cl-fp32-correctly-rounded-divide-sqrt";

/// The image names the records and the runner use.
pub const names = [_][]const u8{ "inkling2_glue", "inkling2_runtime", "inkling2_glue_ctl", "inkling2_runtime_ctl" };

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
pub fn images(b: *std.Build, ocloc: []const u8, device: []const u8) [4]std.Build.LazyPath {
    const gen = b.addExecutable(.{ .name = "inkling2-control-gen", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/xpu/inkling2_control_gen.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    }) });
    const glue = "zig/kernels/xpu/inkling2/glue.cl";
    const runtime = "zig/kernels/xpu/inkling2/runtime.cl";
    return .{
        compile(b, ocloc, device, names[0], b.path(glue)),
        compile(b, ocloc, device, names[1], b.path(runtime)),
        compile(b, ocloc, device, names[2], control(b, gen, "glue", glue, "glue_ctl.cl")),
        compile(b, ocloc, device, names[3], control(b, gen, "runtime", runtime, "runtime_ctl.cl")),
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
    b.step("test-inkling2-xpu-controls", "Inkling 2 Mini's XPU source controls plant one fault each (no device, no ocloc)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}

//! Build tool: writes an Inkling 2 Mini XPU control image's OpenCL source from the production source.
//! Usage: inkling2-control-gen glue|runtime PRODUCTION.cl OUT.cl
const std = @import("std");
const controls = @import("inkling2_controls.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: inkling2-control-gen glue|runtime PRODUCTION.cl OUT.cl\n", .{});
        return 2;
    }
    const entries: []const controls.Entry = if (std.mem.eql(u8, args[1], "glue")) &controls.glue else if (std.mem.eql(u8, args[1], "runtime")) &controls.runtime else {
        std.debug.print("inkling2-control-gen: unknown part {s}\n", .{args[1]});
        return 2;
    };
    const production = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 20));
    defer a.free(production);
    const text = try controls.source(a, production, entries);
    defer a.free(text);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = text });
    return 0;
}

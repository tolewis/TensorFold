//! A Responses request's `include`: `reasoning.encrypted_content`, which Pi and other clients built on the Responses
//! API send with every reasoning request, asks for nothing this server leaves out (reasoning comes as text in its
//! reasoning item, and nothing is encrypted), so it is accepted and ignored; any other value is still refused.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const errors = @import("errors.zig");
const responses_translate = @import("responses_translate.zig");

const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

/// `body` through the Responses translation: refused with a message containing `refused`, or translated when null.
fn expectInclude(body: []const u8, refused: ?[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    if (responses_translate.translate(&cx, raw, NoHistory{})) |_| {
        if (refused) |want| {
            std.debug.print("translated, wanted a refusal naming {s}\n", .{want});
            return error.TestUnexpectedResult;
        }
    } else |err| {
        try std.testing.expectEqual(error.Refused, err);
        const want = refused orelse {
            std.debug.print("refused: {s}\n", .{cx.message});
            return error.TestUnexpectedResult;
        };
        try std.testing.expect(std.mem.indexOf(u8, cx.message, want) != null);
    }
}

test "include reasoning.encrypted_content is accepted and ignored" {
    try expectInclude("{\"input\":\"x\",\"include\":[\"reasoning.encrypted_content\"]}", null);
    try expectInclude("{\"input\":\"x\",\"reasoning\":{\"effort\":\"high\"},\"include\":[\"reasoning.encrypted_content\"]}", null);
    try expectInclude("{\"input\":\"x\",\"include\":[]}", null);
}

test "any other include value is still refused, and named" {
    try expectInclude("{\"input\":\"x\",\"include\":[\"message.output_text.logprobs\"]}", "message.output_text.logprobs");
    try expectInclude("{\"input\":\"x\",\"include\":[\"reasoning.encrypted_content\",\"file_search_call.results\"]}", "file_search_call.results");
    try expectInclude("{\"input\":\"x\",\"include\":\"reasoning.encrypted_content\"}", "include is not supported");
}

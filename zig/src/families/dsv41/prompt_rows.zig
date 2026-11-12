//! The two row counts that decide which kernels a prompt chunk takes (TF_DS_PROMPT_WINDOW_ROWS, TF_DS_PROMPT_GEMM_ROWS;
//! both off by default): the parsing and the path choices. Pure: no cuda import.
//! The two paths round differently, so a token's bits can differ between them; unset, nothing changes.
const std = @import("std");

pub const window_env: [:0]const u8 = "TF_DS_PROMPT_WINDOW_ROWS";
pub const gemm_env: [:0]const u8 = "TF_DS_PROMPT_GEMM_ROWS";

/// Chunks under `window` rows take the MoE decode window; unset: the served 64 (exl3_experts.exact_rows).
pub const default_window = 64;
/// grouped_cp_launch refuses a member list past 32 * 4 warps rows.
pub const max_window = 128;
/// Chunks over `gemm` rows take the prefill GEMM for the dense linears; unset: the served 128.
pub const default_gemm = 128;
/// The glinear group takes 1..128 rows; a GEMM under 16 rows would be a 1-row tile of padding.
pub const min_gemm = 16;
pub const max_gemm = 128;

pub const window_hint = std.fmt.comptimePrint("{s}: the rows under which a prompt chunk's MoE takes the decode window, {d} to {d}", .{ window_env, default_window, max_window });
pub const gemm_hint = std.fmt.comptimePrint("{s}: the rows over which a prompt chunk's dense linears take the prefill GEMM, {d} to {d}", .{ gemm_env, min_gemm, max_gemm });

pub fn validWindow(rows: usize) bool {
    return rows >= default_window and rows <= max_window;
}

pub fn validGemm(rows: usize) bool {
    return rows >= min_gemm and rows <= max_gemm;
}

/// TF_DS_PROMPT_WINDOW_ROWS's text as the window's rows (error.BadPromptWindowRows: not a number or outside 64..128).
pub fn parseWindow(text: []const u8) error{BadPromptWindowRows}!usize {
    const rows = std.fmt.parseInt(usize, std.mem.trim(u8, text, " "), 10) catch return error.BadPromptWindowRows;
    if (!validWindow(rows)) return error.BadPromptWindowRows;
    return rows;
}

/// TF_DS_PROMPT_GEMM_ROWS's text as the GEMM's switch (error.BadPromptGemmRows: not a number or outside 16..128).
pub fn parseGemm(text: []const u8) error{BadPromptGemmRows}!usize {
    const rows = std.fmt.parseInt(usize, std.mem.trim(u8, text, " "), 10) catch return error.BadPromptGemmRows;
    if (!validGemm(rows)) return error.BadPromptGemmRows;
    return rows;
}

/// The window's rows from the environment, the same read on every rank (unset: the default); a bad value is named on stderr.
pub fn windowFromEnv() error{BadPromptWindowRows}!usize {
    const v = std.c.getenv(window_env) orelse return default_window;
    return parseWindow(std.mem.span(v)) catch |err| {
        std.debug.print("{s}\n", .{window_hint});
        return err;
    };
}

/// The GEMM's switch from the environment, the same read on every rank (unset: the default); a bad value is named on stderr.
pub fn gemmFromEnv() error{BadPromptGemmRows}!usize {
    const v = std.c.getenv(gemm_env) orelse return default_gemm;
    return parseGemm(std.mem.span(v)) catch |err| {
        std.debug.print("{s}\n", .{gemm_hint});
        return err;
    };
}

/// Whether a chunk of `n` rows takes the MoE decode window (else exl3_experts.prompt).
pub fn moeInWindow(n: usize, window: usize) bool {
    return n < window;
}

/// Whether `m` rows of a dense linear take the prefill GEMM (else the glinear group).
pub fn denseOnGemm(m: usize, gemm: usize) bool {
    return m > gemm;
}

// -- tests -----------------------------------------------------------------------------------------------------------

const expectEqual = std.testing.expectEqual;

test "the window switch takes 64 to 128 rows and nothing else" {
    try expectEqual(@as(usize, 64), try parseWindow("64"));
    try expectEqual(@as(usize, 100), try parseWindow(" 100 "));
    try expectEqual(@as(usize, 128), try parseWindow("128"));
    for ([_][]const u8{ "", "0", "63", "129", "-64", "64.5", "x", "0x80", "64,", "1e2", "99999999999999999999" }) |bad| {
        try std.testing.expectError(error.BadPromptWindowRows, parseWindow(bad));
    }
}

test "the GEMM switch takes 16 to 128 rows and nothing else" {
    try expectEqual(@as(usize, 16), try parseGemm("16"));
    try expectEqual(@as(usize, 90), try parseGemm("90"));
    try expectEqual(@as(usize, 128), try parseGemm(" 128"));
    for ([_][]const u8{ "", "0", "1", "15", "129", "2048", "-16", "16.5", "x", "0x20", "32,", "99999999999999999999" }) |bad| {
        try std.testing.expectError(error.BadPromptGemmRows, parseGemm(bad));
    }
}

test "the defaults keep the served paths" {
    // the MoE window is the 63 rows and fewer, the dense GEMM the 129 rows and more
    try std.testing.expect(moeInWindow(63, default_window) and !moeInWindow(64, default_window));
    try std.testing.expect(!denseOnGemm(128, default_gemm) and denseOnGemm(129, default_gemm));
}

test "the MoE path at 63, 64, 100, 128 and 129 rows" {
    const rows = [_]usize{ 63, 64, 100, 128, 129 };
    const unset = [_]bool{ true, false, false, false, false };
    const w101 = [_]bool{ true, true, true, false, false };
    const w128 = [_]bool{ true, true, true, false, false };
    for (rows, unset, w101, w128) |n, a, b, c| {
        try expectEqual(a, moeInWindow(n, default_window));
        try expectEqual(b, moeInWindow(n, 101));
        try expectEqual(c, moeInWindow(n, max_window));
    }
    try expectEqual(true, moeInWindow(127, max_window));
}

test "the dense path at 63, 64, 100, 128 and 129 rows" {
    const rows = [_]usize{ 63, 64, 100, 128, 129 };
    const unset = [_]bool{ false, false, false, false, true };
    const g64 = [_]bool{ false, false, true, true, true };
    const g16 = [_]bool{ true, true, true, true, true };
    for (rows, unset, g64, g16) |n, a, b, c| {
        try expectEqual(a, denseOnGemm(n, default_gemm));
        try expectEqual(b, denseOnGemm(n, 64));
        try expectEqual(c, denseOnGemm(n, min_gemm));
    }
}

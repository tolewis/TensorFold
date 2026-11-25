//! Level Zero kernels and launches for Inkling 2 Mini attention: the same entry points and argument meaning as the
//! Metal host (`../metal/attention.zig`), with device addresses in place of Metal buffers.
const std = @import("std");
const xpu = @import("xpu");
pub const planning = @import("attention_plan.zig");

/// A device address and a byte offset into the allocation it lies in.
pub const Ref = struct {
    ptr: ?*anyopaque,
    off: usize = 0,
    pub fn at(r: Ref) ?*anyopaque {
        return if (r.ptr) |p| @ptrFromInt(@intFromPtr(p) + r.off) else null;
    }
};
pub const Cache = struct { k: Ref, v: Ref };
pub const RowMeta = struct { pos: Ref, seg_of: Ref, segs: Ref };

pub const Error = planning.Error || xpu.Error || error{OutOfMemory};

/// The merge over eight sub-groups of chunk ranges. It regroups the fold, so it is bit-identical only where the row
/// count is one; every other row count keeps the serial fold. Off by default: at long context a one-row decode regroups
/// while a verify window keeps the serial fold, so drafted output parts from plain. `INK2_MERGE_SPLIT=1` selects it.
pub var merge_split: bool = false;

pub fn mergeSplit() bool {
    if (merge_split) return true;
    if (std.c.getenv("INK2_MERGE_SPLIT")) |p| {
        const s = std.mem.span(p);
        return s.len > 0 and s[0] == '1';
    }
    return false;
}

/// Kernel names, with the work-group size each is made at.
const Entry = struct { name: [:0]const u8, threads: u32 };
const names = struct {
    const prep: Entry = .{ .name = "tf_ink2_prep", .threads = 16 };
    const rel_table: Entry = .{ .name = "tf_ink2_rel_table", .threads = 256 };
    const decode: Entry = .{ .name = "tf_ink2_attn_decode", .threads = planning.decode_threads };
    const merge: Entry = .{ .name = "tf_ink2_merge", .threads = planning.merge_threads };
    const merge_split: Entry = .{ .name = "tf_ink2_merge_split", .threads = planning.merge_threads };
    const prompt: Entry = .{ .name = "tf_ink2_attn_prompt", .threads = planning.prompt_threads };
};

/// The five kernels of two loaded images: `attention` holds prep, relative table, decode and merge; `prompt` the prompt kernel.
pub const Kernels = struct {
    prep: xpu.Kernel,
    rel_table: xpu.Kernel,
    decode: xpu.Kernel,
    merge: xpu.Kernel,
    merge_split: xpu.Kernel,
    prompt: xpu.Kernel,

    pub fn init(attention: xpu.Module, prompt: xpu.Module) xpu.Error!Kernels {
        var k: Kernels = undefined;
        k.prep = try attention.kernel(names.prep.name, .{ names.prep.threads, 1, 1 });
        errdefer k.prep.deinit();
        k.rel_table = try attention.kernel(names.rel_table.name, .{ names.rel_table.threads, 1, 1 });
        errdefer k.rel_table.deinit();
        k.decode = try attention.kernel(names.decode.name, .{ names.decode.threads, 1, 1 });
        errdefer k.decode.deinit();
        k.merge = try attention.kernel(names.merge.name, .{ names.merge.threads, 1, 1 });
        errdefer k.merge.deinit();
        k.merge_split = try attention.kernel(names.merge_split.name, .{ names.merge_split.threads, 1, 1 });
        errdefer k.merge_split.deinit();
        k.prompt = try prompt.kernel(names.prompt.name, .{ names.prompt.threads, 1, 1 });
        return k;
    }
    pub fn deinit(k: *Kernels) void {
        k.prompt.deinit();
        k.merge_split.deinit();
        k.merge.deinit();
        k.decode.deinit();
        k.rel_table.deinit();
        k.prep.deinit();
        k.* = undefined;
    }
};

/// raw holds independent q/k/v rows followed by q-side relative rows; gated sliding visits never infer K or V from Q.
pub fn prep(k: *const Kernels, s: xpu.Stream, r: planning.Rows, l: planning.Layer, c: planning.Prep, meta: RowMeta, raw: [4]Ref, scales: [2]Ref, q: Ref, rel: Ref, kv: Cache) xpu.Error!void {
    try xpu.launch.dispatch(k.prep, s, .{ r.rows, r.heads + 2 * r.kv_heads, 1 }, .{
        raw[0].at(),     raw[1].at(),  raw[2].at(), raw[3].at(),
        scales[0].at(),  scales[1].at(), meta.pos.at(), meta.seg_of.at(),
        meta.segs.at(),  q.at(),       rel.at(),    kv.k.at(),
        kv.v.at(),       r.heads,      r.kv_heads,  l.ring,
        @intFromBool(l.tau), c.qk_clip, c.rel_clip, c.floor,
        c.alpha,         c.eps,
    });
}

pub fn relTable(k: *const Kernels, s: xpu.Stream, r: planning.Rows, rel: Ref, proj: Ref, tab: Ref) xpu.Error!void {
    try xpu.launch.dispatch(k.rel_table, s, .{ r.rows * r.heads, 1, 1 }, .{ rel.at(), proj.at(), tab.at() });
}

/// Attention over the cache for `r.rows` rows: the prompt kernel or the chunked decode kernel by the plan, then the merge.
/// `stride` is the partial slots a row and head hold in `part_o` and `part_ml`.
pub fn attend(k: *const Kernels, s: xpu.Stream, r: planning.Rows, l: planning.Layer, stride: u32, meta: RowMeta, q: Ref, kv: Cache, tab: Ref, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) Error!void {
    return attendAs(k, s, if (planning.usePrompt(r, l)) .prompt else .decode, r, l, stride, meta, q, kv, tab, part_o, part_ml, sink, gate, out);
}

/// `attend` on a path the caller names (the contract runs the prompt kernel at a few rows too).
pub fn attendAs(k: *const Kernels, s: xpu.Stream, path: planning.Path, r: planning.Rows, l: planning.Layer, stride: u32, meta: RowMeta, q: Ref, kv: Cache, tab: Ref, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) Error!void {
    const launch = try planning.attendAs(r, l, stride, path);
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    if (launch.path == .decode) {
        try xpu.launch.dispatch(k.decode, s, launch.grid, .{
            q.at(),          kv.k.at(),       kv.v.at(),         tab.at(),      meta.segs.at(), part_o.at(), part_ml.at(), meta.pos.at(), meta.seg_of.at(),
            r.heads,         r.kv_heads,      l.ring,            l.window,      launch.chunk,   @as(u32, 0), launch.partials_rh, scale,
        });
    } else {
        try xpu.launch.dispatch(k.prompt, s, launch.grid, .{
            q.at(),     kv.k.at(),   kv.v.at(),       tab.at(),     meta.segs.at(),
            part_o.at(), part_ml.at(), r.heads,        r.kv_heads,   l.ring,
            launch.chunk, @as(u32, 0), launch.partials_rh, scale,
        });
    }
    const m = if (mergeSplit() and r.rows == 1) k.merge_split else k.merge;
    try xpu.launch.dispatch(m, s, launch.merge_grid, .{
        part_o.at(), part_ml.at(), sink.at(), gate.at(), meta.pos.at(), out.at(),
        r.heads,     l.window,     launch.chunk, @as(u32, 0), launch.partials_rh, launch.subs,
    });
}

test {
    _ = @import("attention_plan.zig");
}

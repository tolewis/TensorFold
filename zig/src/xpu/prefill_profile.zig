//! The prefill profile (`XPU_PREFILL_PROFILE=1`): the prompt path's GPU time per kernel class, summed per chunk
//! and per prompt, plus the host gap (the chunk wall time neither the device work nor an all-reduce wait covers).
//! A pass is the windows of one prompt: `start` opens it at the prompt's first window, `endChunk` closes it
//! (and prints the table) when the last window ends. Decode, verify and the load-time measure are outside a
//! pass, so they cost nothing. With the flag unset, every hook checks one global and does no more.
//!
//! How the windows work: `d.launch` attaches a kernel-timestamp event to every launch of an open pass and
//! remembers the launch's class. `flush` (at every sync, and on demand) reads the pending events in enqueue
//! order; each event marks the end of one device window, and the window is charged to that launch's class. The
//! device work between two classified launches (the all-reduce's PCIe copy, a padding copy) is charged to the
//! later launch. The first event of a pass is a baseline and charges nothing.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");

/// The kernel classes of the prompt path. The EXL3 GEMM classes name the projection site (4 per layer, then
/// lm_head), not the quant path: the live build runs these sites as EXL3, W4A8 or W4A4.
pub const Class = enum(u8) {
    exl3_mixer_in, // site 0: the layer's first wide projection (the attention qkv, or the GDN qkv | z | a | b)
    exl3_mixer_out, // site 1: the mixer's out projection (an all-reduce follows it)
    exl3_gu, // site 2: gate | up and its swiglu
    exl3_down, // site 3: down (an all-reduce follows it)
    exl3_head, // lm_head, on the pass's last window only
    gdn_chunk, // the GDN chunk (gdn_pre_rows, gdn_prep, gdn_state, gdn_out) and the dense a | b projection
    attn_prefill, // the attention prefill (attn_prep, attn_prompt, attn_prompt8, rot_heads, attn_gate)
    norms_glue, // add_rmsnorm, gated_norm, the fake-quantization kernels, rowsums, argmax, the taps
    all_reduce_wait, // the host settle and the peer wait inside an all-reduce
    all_reduce_add, // tp_add and the peer slot's PCIe copy (or the host sum on the fallback path)
    embed, // the token embedding gather before the layer loop
    n_classes, // keep last
};

const n_classes = @intFromEnum(Class.n_classes);

const names = [_][]const u8{
    "exl3_mixer_in",
    "exl3_mixer_out",
    "exl3_gu",
    "exl3_down",
    "exl3_head",
    "gdn_chunk",
    "attn_prefill",
    "norms_glue",
    "all_reduce_wait",
    "all_reduce_add",
    "embed",
    "n_classes",
};

const EVENTS = 2048; // one event pool a pass (a 131k pass is a few thousand launches at most)
const ENTRIES = 2048; // pending launches between flushes
const CHUNK_CAP = 4096; // per-chunk lines of the table (131k / 32 rows)

pub const Profile = struct {
    on: bool = false, // `XPU_PREFILL_PROFILE=1` (set at load)
    active: bool = false, // a pass is open
    resume: u32 = 0, // the open pass's start position (0: a cold prompt)
    cur: Class = .norms_glue, // the class the next `d.launch` is charged to
    cur_gflop: f64 = 0, // the next launch's GEMM work (GFLOP)
    cur_bytes: f64 = 0, // the next launch's KV reads (GB)
    gpu: [n_classes]f64 = @splat(0), // seconds, by class
    work: [n_classes]f64 = @splat(0), // the GEMM GFLOP charged, by class
    kv: [n_classes]f64 = @splat(0), // the KV reads charged (GB), by class
    host_gap: f64 = 0, // the chunk wall time neither the device work nor a wait covers
    n_chunk: u32 = 0, // the pass's chunk count
    chunk_start: f64 = 0, // the host time of the current chunk's start
    last_ts: f64 = 0, // the GPU time of the last event read (the previous window's end)
    first: bool = true, // the pass's first flush takes its first event as the baseline
    used: usize = 0,
    events: [EVENTS]ze.H = @splat(null),
    entries: [ENTRIES]Class = @splat(.norms_glue),
    chunk_rows: [CHUNK_CAP]u32 = @splat(0), // the pass's per-chunk rows
    chunk_ms: [CHUNK_CAP]f64 = @splat(0), // the pass's per-chunk device seconds
    chunk_rec: u32 = 0,

    /// The class the next `d.launch` is charged to, and its GEMM work (GFLOP) and KV reads (GB).
    pub fn mark(p: *Profile, cls: Class, gflop: f64, bytes: f64) void {
        p.cur = cls;
        p.cur_gflop = gflop;
        p.cur_bytes = bytes;
    }

    /// Opens a pass at the prompt's first window. `at` is the prompt position the pass resumes from.
    pub fn start(p: *Profile, at: u32) void {
        if (!p.on) return;
        p.active = true;
        p.resume = at;
        p.gpu = @splat(0);
        p.work = @splat(0);
        p.kv = @splat(0);
        p.host_gap = 0;
        p.n_chunk = 0;
        p.chunk_rec = 0;
        p.used = 0;
        p.first = true;
        p.chunk_start = q.now();
    }

    /// One prompt window ended (one forward chunk of `rows` rows): flush, add the chunk's host gap and its
    /// per-chunk line. `last` prints the pass's table and closes it.
    pub fn endChunk(p: *Profile, d: *dev.Dev, last: bool, rows: u32) !void {
        if (!p.on or !p.active) return;
        const before = p.gpu;
        const arw_before = before[@intFromEnum(Class.all_reduce_wait)];
        try p.flush(d);
        var dev_delta: f64 = 0;
        var i: usize = 0;
        while (i < before.len) : (i += 1) dev_delta += p.gpu[i] - before[i];
        const arw = p.gpu[@intFromEnum(Class.all_reduce_wait)] - arw_before;
        const wall = q.now() - p.chunk_start;
        if (wall > dev_delta + arw) p.host_gap += wall - dev_delta - arw;
        if (p.chunk_rec < CHUNK_CAP) {
            p.chunk_rows[p.chunk_rec] = rows;
            p.chunk_ms[p.chunk_rec] = dev_delta * 1e3;
            p.chunk_rec += 1;
        }
        p.n_chunk += 1;
        if (last) {
            p.print();
            p.active = false;
        } else p.chunk_start = q.now();
    }

    /// Records the next launch of the open pass: the current class and its work, and the event `d.launch`
    /// will signal (read by the next `flush`). `null` when the pool is not armed (the launch goes on unprofiled).
    pub fn nextEvent(p: *Profile, d: *dev.Dev) !ze.H {
        if (p.events[0] == null) try p.arm(d);
        if (p.events[0] == null) return null;
        if (p.used + 1 >= ENTRIES) try p.flush(d);
        if (p.used + 1 >= ENTRIES) return error.ProfileFull;
        const ev = p.events[p.used];
        p.entries[p.used] = p.cur;
        const c: usize = @intFromEnum(p.cur);
        p.work[c] += p.cur_gflop;
        p.kv[c] += p.cur_bytes;
        p.cur_gflop = 0;
        p.cur_bytes = 0;
        p.used += 1;
        return ev;
    }

    /// Reads the pending events in enqueue order and charges each device window to its launch's class.
    pub fn flush(p: *Profile, d: *dev.Dev) !void {
        if (p.used == 0) return;
        var ts: [ENTRIES]f64 = @splat(0);
        var i: usize = 0;
        while (i < p.used) : (i += 1) {
            ts[i] = try d.g.seconds(p.events[i]);
            try ze.check(d.g.api.zeEventHostReset(p.events[i]));
        }
        var prev = if (p.first) ts[0] else p.last_ts;
        p.first = false;
        var j: usize = 0;
        while (j < p.used) : (j += 1) {
            const dt = ts[j] - prev;
            prev = ts[j];
            if (dt > 0) {
                const c: usize = @intFromEnum(p.entries[j]);
                if (c < n_classes) p.gpu[c] += dt;
            }
        }
        p.last_ts = ts[p.used - 1];
        p.used = 0;
    }

    /// The event pool, once at load (the launches then attach one event apiece until the pool is full).
    pub fn arm(p: *Profile, d: *dev.Dev) !void {
        if (!p.on) return;
        const pool = try d.g.events(EVENTS);
        @memcpy(&p.events, pool);
    }

    /// The pass's table: the per-chunk lines (where the pass's device time grows), then the class table.
    pub fn print(p: *Profile) void {
        var total: f64 = 0;
        var rows_tot: u64 = 0;
        var i: usize = 0;
        while (i < n_classes) : (i += 1) total += p.gpu[i];
        i = 0;
        while (i < p.chunk_rec) : (i += 1) rows_tot += p.chunk_rows[i];
        std.debug.print("prefill profile rank {d}: {d} chunks, {d} rows (resume at {d}), {d:.1f} ms device\n", .{
            q.RANK, p.n_chunk, rows_tot, p.resume, total * 1e3,
        });
        i = 0;
        while (i < p.chunk_rec) : (i += 1) {
            std.debug.print("  chunk {d:>4} rows {d:>5} {d:>8.1f} ms\n", .{ i, p.chunk_rows[i], p.chunk_ms[i] });
        }
        i = 0;
        while (i < n_classes) : (i += 1) {
            const t = p.gpu[i];
            if (t <= 0) continue;
            std.debug.print("  {s:<18} {d:>8.2} ms {d:>5.1f}%", .{ names[i], t * 1e3, if (total > 0) t / total * 100 else 0 });
            if (p.work[i] > 0) std.debug.print("  {d:>8.2} GFLOP {d:>8.0f} GFLOPS\n", .{ p.work[i], p.work[i] / t / 1e9 });
            else if (p.kv[i] > 0) std.debug.print("  {d:>8.2} GB kv   {d:>8.1f} GB/s\n", .{ p.kv[i], p.kv[i] / t });
            else std.debug.print("\n");
        }
        std.debug.print("  {s:<18} {d:>8.2} ms  chunk wall minus device and all-reduce wait\n", .{ "host gap", p.host_gap * 1e3 });
    }
};

pub var prefill_profile: Profile = .{};

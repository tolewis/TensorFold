//! Qwen3.8-27B forward on Intel Xe2 over Level Zero (MLX affine 4-bit g64), chunked prompt, serial decode.
const std = @import("std");
pub const ze = @import("ze");
const kb = @import("xpu_qwen27_kernels");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const tp = @import("tp.zig");
const loader = @import("load.zig");
const multi = @import("runner_multi.zig");
const slot_states = @import("slot_states.zig");
const pages_mod = @import("pages.zig");
const k1mod = @import("k1.zig"); // XPU_K1: the split-K DPAS decode over int8 pool pages (lane p22-k1)
pub const follow = @import("follow.zig");
pub const limits = @import("limits.zig");
const Prof = dev.Prof;
pub const Site = dev.Site;
const a8 = fwd.a8;
const forward = fwd.forward;
const kvCap = fwd.kvCap;
const load = loader.load;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
pub const MAX_SLOTS = loader.MAX_SLOTS;
pub const freeAfterQueue = slot_states.freeAfterQueue;
pub const gdnBytes = slot_states.gdnBytes;
pub const keptGdnBytes = slot_states.keptGdnBytes;
pub const Tensor = dev.Tensor;
pub const Checkpoint = dev.Checkpoint;
pub const Dev = dev.Dev;
pub const copy = dev.copy;
pub const off = dev.off;
pub const profReport = dev.profReport;
pub const Proj = loader.Proj;
pub const Model = loader.Model;
pub const Acts = fwd.Acts;
pub const embedRows = fwd.embedRows;
pub const gemm = fwd.gemm;
pub const Want = fwd.Want;
pub const no_q8 = fwd.no_q8;
pub const Mode = fwd.Mode;

pub const alloc = std.heap.c_allocator;

// Qwen3.8-27B shape (checked against config.json at load).
pub const D = 5120;
pub const FF = 17408;
/// Tensor-parallel ranks of this build. TP 1 is the one-card engine. Every per-rank width below is the whole
/// model's divided by TP: a rank holds its share of the heads and of the MLP columns, and the hidden state (D)
/// whole (lanes/qwen27/TP2-ASSESSMENT.md in Intel-B70-Recipe, 2026-11-01).
pub const TP: u32 = @import("xpu").kernel_options.tp; // -Dtp (zig/build/xpu.zig), 1 by default
/// This process's rank, 0 to TP - 1 (TF_TP_RANK at start). Rank 0 serves and drafts. 0 at TP 1.
pub var RANK: u32 = 0;
pub const GDN_K_HEADS = 16 / TP; // GDN query and key heads of 128
pub const GDN_V_HEADS = 48 / TP; // GDN value heads of 128 (and the z gate, a and b)
pub const ATT_Q_HEADS = 24 / TP; // attention query heads of 256, each with a gate of 256
pub const ATT_KV_HEADS = 4 / TP; // attention key and value heads of 256
pub const FF_R = FF / TP; // gate and up columns a rank, down rows a rank
pub const GDN_QKV = (2 * GDN_K_HEADS + GDN_V_HEADS) * 128; // 10240 at TP 1
pub const GDN_Z = GDN_V_HEADS * 128; // 6144 at TP 1
pub const GDN_AB = GDN_QKV + GDN_Z; // the dense a and b gates' first column (16384 at TP 1)
pub const LAYERS = 64;
pub const VOCAB = 248320;
pub const MASK_WORDS = (VOCAB + 31) / 32; // a grammar row's bits (structured output), one a token
/// The layers' quantization group (64: MLX affine, 128: GPTQ), set at load; the head and the drafter keep 64.
pub var GS: u32 = 64;
pub const GDN_IN = (GDN_AB + 2 * GDN_V_HEADS + 255) / 256 * 256; // qkv 10240 | z 6144 | a 48 | b 48 = 16480, padded to the GEMM tile (256): 16640 at TP 1
pub const ATT_IN = ATT_Q_HEADS * 512 + 2 * ATT_KV_HEADS * 256; // (q 256 | gate 256) × 24 | k 1024 | v 1024: 14336 at TP 1
pub var CHUNK: u32 = 1024; // prompt rows per forward (XPU_CHUNK, a multiple of 64)
/// A forward's rows at most: a chunk, plus a rest under 128 rows joined to a prompt's last chunk.
pub fn maxRows() u32 {
    return CHUNK + 128;
}
/// Where a prompt chunk from `at` ends: CHUNK rows on, or the prompt's end when the rest is under 128 rows more.
pub fn chunkEnd(at: usize, len: usize) usize {
    return if (len - at < CHUNK + 128) len else at + CHUNK;
}
pub var MAX_CTX: u32 = 4096; // KV cache rows (XPU_CTX)
pub const SPLIT = 32; // keys per attention split (decode.cl)
pub const EPS: f32 = 1e-6;

/// K/V positions a page holds (shape.h Q27_PAGE): a stream's positions reach its pages through its page table.
pub const PAGE: u32 = 64;
/// Shared K/V pool (XPU_KV_POOL=1, or --kv-pool-tokens): int8 pages every slot reads through its own page table.
/// Off by default: each slot keeps its own contiguous cache, as before the pool.
pub var kv_pool = false;
/// --kv-pool-tokens: the pool's size in tokens; null: what the card leaves after the load and the reserve.
pub var pool_tokens: ?u32 = null;
/// XPU_KEEP_SHARED=1 (lane p29-poolkeep): kept prompt states may hold any free pool page. When a live stream needs a
/// page and the pool has none free, the least recently used kept state goes first, and a reserve of pages stays free
/// for the streams. Off by default: kept states get only the pages past the streams' contexts, as before.
pub var keep_shared = false;
/// XPU_KEEP_PAGES=1: the prompt-cache budget counts each pool page once, however many kept states hold it.
/// Needs XPU_KV_POOL=1 and XPU_KEEP_SHARED=1. Without them it logs one warning and stays off.
pub var keep_pages = false;
/// XPU_KEPT_HOST=1 (lane p43-keptcap): a kept prompt state's GDN, convolution and drafter-context copies live in pinned
/// host memory, not on the card. The card's 1536 MiB reserve for them goes back to the pool, and the number of kept
/// states comes from a host budget (XPU_KEPT_HOST_GIB, default 16 GiB a rank). Off by default.
pub var kept_host = false;
/// XPU_KEPT_HOST_GIB: the host bytes a rank gives kept states' copies with XPU_KEPT_HOST=1.
pub var kept_host_gib: f64 = 16;
/// XPU_PARK_GIB: host bytes for idle kept prompt states, the total over all ranks. Unset or 0: off.
/// Rank 0 checks parked bytes summed over the ranks. A follower never refuses a park for this budget.
/// It needs XPU_KEEP_PAGES=1 and XPU_KEPT_HOST=1.
pub var park_gib: f64 = 0;
/// The host tier is on (mode word bit 23). False when the flag is off or a required flag is missing.
pub var park_on = false;
/// XPU_PARK_PINNED=1: pinned host memory for the tier. Off: plain page-aligned host memory.
pub var park_pinned = false;
/// XPU_PARK_LOG=1: print the bytes and the time of each park copy and each restore copy.
pub var park_log = false;
/// XPU_PARK_MIN: a shorter state is dropped, not parked. Tokens. Default 16384.
pub var park_min: u32 = 16384;

/// Drops the least recently used kept prompt state the prompt cache may evict; false when none is left.
pub const Evict = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque) bool };

// Layout of a GDN layer's stash (bytes): q, k, v, g, β, then the qkv rows of P.
pub const ST_Q = 0;
pub const ST_K = ST_Q + 16 * GDN_K_HEADS * 128 * 2;
pub const ST_V = ST_K + 16 * GDN_K_HEADS * 128 * 2;
pub const ST_G = ST_V + 16 * GDN_V_HEADS * 128 * 2;
pub const ST_B = ST_G + 16 * GDN_V_HEADS * 4;
pub const ST_P = ST_B + 16 * GDN_V_HEADS * 4;
pub const STASH = ST_P + 16 * GDN_IN * 2;
/// Target layers whose hidden states feed the DFlash2 drafter (its config's target_layer_ids).
pub const TAP_LAYERS = [5]usize{ 5, 19, 33, 47, 61 };

pub fn isAttention(i: usize) bool {
    return (i + 1) % 4 == 0;
}

/// XPU_FUSE=1 bit 0: the device-direct exchange's tp_add joins the norm that follows it (mixer and MLP exchanges).
/// A bit mask; with it unset or 0 the engine takes today's path. Bit 0 is the exchange-into-norm step (WP2 step 3).
pub var fuse: u32 = 0;
/// XPU_DUMP=DIR: the first forward writes x (fp32 [T, D]) after each of the first four layers, for comparison.
pub var dump_dir: ?[]const u8 = null;
pub var verbose = false;
pub var dump_h: ?c_int = null;
/// W4A8 study (XPU_A8): prompt GEMM inputs quantized to int8 and back in place; 128 per group, "token" per row.
pub var a8_group: u32 = 0;
/// K/V cache study (XPU_KV8): 1 rotates q, k, v by Hadamard, 2 quantizes k, v to int8 per head, 3 both.
pub var kv8: i32 = 0;
/// XPU_GDN (lane p4-k-gdn), bits. 1: a stashed GDN layer's GEMM and gdn_pre write the stash, so no copy follows.
/// 2: gdn_step2, gdn_step with block loads (the same bits). 4: at the load, gdn_step2 against gdn_step on the card
/// (gdn_check.zig), and the open fails when a value differs. Default 0.
pub var gdn_bits: u32 = 0;
/// XPU_MV (lane p4-k-mv): the decode matvec kernel of a call site, `site:variant` pairs separated by commas
/// (XPU_MV=mixer_in_gdn:d16,mixer_in_attn:d16). A site is a `Site` name. A variant is a kernel of gemm.cl:
/// wide sites (the family with no K slices): d16, d64, w8, p0, p4, the same bits as the default. `slice` puts a wide
/// site on the K-slice kernel: its bits change, so it is a timing probe only. Narrow sites (K slices): s16, seq, the
/// same bits as the default. Default: no pick.
/// A variant after an `@` (gate_up:@d64) is a shadow: the kernel runs before the site's own kernel and writes a scratch
/// buffer. The model keeps its bits, and the row `mv_<variant>_<site>` holds the time of the variant on this shape.
/// Up to 8 shadows a site, any variant on any site. `ref` and `nref` are the default wide and narrow kernels.
/// w0, a0, wa0, h8 and wseq are probes with wrong results (gemm.cl), so they run as shadows only.
pub const MvPick = struct { k: ze.H, div: u32, wide: bool };
pub var mv_pick: [16]?MvPick = @splat(null);
pub var mv_shadow: [16][8]?MvPick = @splat(@splat(null));
pub var mv_scratch: ?*anyopaque = null;
pub const mv_scratch_bytes: usize = 32 << 20;
const MvVariant = struct { name: []const u8, row: []const u8, kernel: [*:0]const u8, local: u32, div: u32, wide: bool, bits: bool = false, probe: bool = false };
const mv_variants = [_]MvVariant{
    .{ .name = "d16", .row = "mv_d16", .kernel = "w4k1_d16", .local = 64, .div = 64, .wide = true }, // 16 columns a subgroup: two times the threads
    .{ .name = "d64", .row = "mv_d64", .kernel = "w4k1_d64", .local = 64, .div = 256, .wide = true }, // 64 columns: half the threads
    .{ .name = "w8", .row = "mv_w8", .kernel = "w4k1_w8", .local = 128, .div = 256, .wide = true }, // 32 columns, 8 subgroups a work-group
    .{ .name = "p0", .row = "mv_p0", .kernel = "w4k1_p0", .local = 64, .div = 128, .wide = true }, // no prefetch
    .{ .name = "p4", .row = "mv_p4", .kernel = "w4k1_p4", .local = 64, .div = 128, .wide = true }, // prefetch 4 blocks ahead
    .{ .name = "slice", .row = "mv_slice", .kernel = "w4k_dsplit", .local = 64, .div = 32, .wide = true, .bits = true }, // 4 K slices a tile: four times the threads, other bits
    .{ .name = "s16", .row = "mv_s16", .kernel = "w4k_s16x16", .local = 64, .div = 16, .wide = false }, // 16 columns a tile, 4 K slices
    .{ .name = "seq", .row = "mv_seq", .kernel = "w4k_decode", .local = 64, .div = 128, .wide = false }, // the K slices one after the other in one subgroup
    .{ .name = "ref", .row = "mv_ref", .kernel = "w4k1_decode", .local = 64, .div = 128, .wide = true }, // the default wide kernel (a shadow of it is the control)
    .{ .name = "nref", .row = "mv_nref", .kernel = "w4k_dsplit", .local = 64, .div = 32, .wide = false }, // the default narrow kernel
    .{ .name = "w0", .row = "mv_w0", .kernel = "w4k1_w0", .local = 64, .div = 128, .wide = true, .probe = true }, // no W stream
    .{ .name = "a0", .row = "mv_a0", .kernel = "w4k1_a0", .local = 64, .div = 128, .wide = true, .probe = true }, // no A stream
    .{ .name = "wa0", .row = "mv_wa0", .kernel = "w4k1_wa0", .local = 64, .div = 128, .wide = true, .probe = true }, // no W and no A stream
    .{ .name = "h8", .row = "mv_h8", .kernel = "w4k1_h8", .local = 64, .div = 128, .wide = true, .probe = true }, // the DPAS of 8 rows
    .{ .name = "wseq", .row = "mv_wseq", .kernel = "w4k1_wseq", .local = 64, .div = 128, .wide = true, .probe = true }, // one W stream a 16-column block
};
/// Int8 K/V cache (XPU_KV_INT8=1): 256 int8 per position and head, then float scales after MAX_CTX positions.
pub var kv_i8 = false;
/// Rotated weights (`hadamard128` in the checkpoint): every GEMM input is rotated by blocks of 128 where it is made.
pub var rot_on: i32 = 0;
/// W4A4 on down (and mixer out projections with XPU_PREFILL_INT4_SITES=10): int4 inputs per row group, DPAS u4 × s4.
pub var prefill_int4 = false;
pub var int4_sites: u32 = 8; // XPU_PREFILL_INT4_SITES: 8 is down only, 2 adds the mixers' out projections
/// W4A8 prompt GEMMs (XPU_PREFILL_INT8, local option, GPTQ g128 only): int8 activations by (row, group of 128).
pub var prefill_int8 = false;
pub var a8_sites: u32 = 15; // XPU_A8_SITES: 1 mixer in, 2 mixer out, 4 gate|up, 8 down
pub var a8_rot: i32 = 0; // XPU_A8_ROT=1: Hadamard by blocks of 128 before the quantization (weights rotated too)
pub var a8_levels: i32 = 127; // XPU_A8_BITS=4: int4 activations (7 levels a side)
pub var a8_sites4: u32 = 15; // XPU_A8_SITES4: the sites in int4 when XPU_A8_BITS=4 (the others in int8)
/// Host-backed embedding table (XPU_EMBED_HOST=1): `embedTable` keeps the fp16 table in pinned host memory for
/// the kernels to read over PCIe, so a card frees ~2.5 GB of VRAM (a research arm, off by default).
pub var embed_host: bool = false;
/// The checkpoint is MLX 4/64 (not EXL3 or GPTQ): a bit of the mode word, so a pair of mixed models fails at start.
pub var fmt_mlx: bool = false;
pub var prof: Prof = .{};

/// XPU_ROUND_STATS: host waits and command counts, summed until the round stats line prints them.
pub const Stat = struct {
    on: bool = false,
    settles: u64 = 0,
    settle_s: f64 = 0,
    peer_s: f64 = 0,
    launches: u64 = 0,
    copies: u64 = 0,
    syncs: u64 = 0,
    hcopies: u64 = 0, // host-pointer copies of the tree verify and its commit (dev.hostCopy)
    hcopy_s: f64 = 0, // host time inside those calls
};
pub var stat: Stat = .{};

/// The XPU_* options as the process had them before its first open, put back at every open (Runner.init).
const Options = struct {
    dump_dir: ?[]const u8,
    dump_h: ?c_int,
    a8_group: u32,
    kv8: i32,
    kv_i8: bool,
    prefill_int4: bool,
    int4_sites: u32,
    prefill_int8: bool,
    a8_sites: u32,
    a8_rot: i32,
    a8_levels: i32,
    a8_sites4: u32,
    embed_host: bool,
    chunk: u32,
    kv_pool: bool,
    pool_tokens: ?u32,
    keep_shared: bool,
    keep_pages: bool,
    kept_host: bool,
    kept_host_gib: f64,
    park_gib: f64,
    park_on: bool,
    park_pinned: bool,
    park_log: bool,
    park_min: u32,
};
var defaults: ?Options = null;
/// One runner at a time: the options, MAX_CTX and GS above belong to the process, not to a runner.
var live = false;

fn options() Options {
    return .{ .dump_dir = dump_dir, .dump_h = dump_h, .a8_group = a8_group, .kv8 = kv8, .kv_i8 = kv_i8, .prefill_int4 = prefill_int4, .int4_sites = int4_sites, .prefill_int8 = prefill_int8, .a8_sites = a8_sites, .a8_rot = a8_rot, .a8_levels = a8_levels, .a8_sites4 = a8_sites4, .embed_host = embed_host, .chunk = CHUNK, .kv_pool = kv_pool, .pool_tokens = pool_tokens, .keep_shared = keep_shared, .keep_pages = keep_pages, .kept_host = kept_host, .kept_host_gib = kept_host_gib, .park_gib = park_gib, .park_on = park_on, .park_pinned = park_pinned, .park_log = park_log, .park_min = park_min };
}

fn restore(o: Options) void {
    dump_dir = o.dump_dir;
    dump_h = o.dump_h;
    a8_group = o.a8_group;
    kv8 = o.kv8;
    kv_i8 = o.kv_i8;
    prefill_int4 = o.prefill_int4;
    int4_sites = o.int4_sites;
    prefill_int8 = o.prefill_int8;
    a8_sites = o.a8_sites;
    a8_rot = o.a8_rot;
    a8_levels = o.a8_levels;
    a8_sites4 = o.a8_sites4;
    embed_host = o.embed_host;
    CHUNK = o.chunk;
    kv_pool = o.kv_pool;
    pool_tokens = o.pool_tokens;
    keep_shared = o.keep_shared;
    keep_pages = o.keep_pages;
    kept_host = o.kept_host;
    kept_host_gib = o.kept_host_gib;
    park_gib = o.park_gib;
    park_on = o.park_on;
    park_pinned = o.park_pinned;
    park_log = o.park_log;
    park_min = o.park_min;
}

extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;
pub fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

/// The pair's idle check: rank 0 exits when the follower's liveness stamp is stale. The XPU health source calls it,
/// so a follower that dies with the pair idle is found at /health instead of at the next request.
pub fn checkIdle() void {
    if (tp.group) |*g| g.checkIdle();
}

/// The whole model on the device and one sequence's state: what a serial host drives.
pub const Runner = struct {
    d: Dev,
    model: Model,
    acts: Acts,
    pos: u32 = 0,
    last: u32 = 0, // greedy next token of the last forward
    context: u32,
    ckpt: Checkpoint = .{},
    host_logits: []f32 = &.{},
    block: [16]u32 = @splat(0), // the last verified block's greedy tokens
    block_rows: u32 = 0,
    tree_on: bool = false, // the last verified block was a tree (commitPath)
    tree_host: [16 * 17]i32 = @splat(0),
    path_host: [16]i32 = @splat(0),
    slot: u32 = 0, // the selected sequence (select)
    mt_host: [LAYERS * 16 * 8]u64 = @splat(0), // verifyMulti's per-row state addresses, by layer
    pos_host: [16]i32 = @splat(0),
    kpos_host: [16]i32 = @splat(0), // XPU_MULTI_TREE: each row's last prefix key (chain: mpos, tree: the window base - 1)
    kkind_host: [16]u8 = @splat(0), // and 1 for a tree row, 0 for a chain row
    win_host: [32]i32 = @splat(0), // and each window's first row and width (keepMulti)
    ids_host: [16]u32 = @splat(0),
    multi_slots: [16]u32 = @splat(0), // the last verifyMulti's slots, by window
    multi_count: u32 = 0,
    multi_tree: bool = false, // the last verifyMulti had tree windows: every window's states are replayed at the keep
    mpath_host: [16 * 2 * 16]i32 = @splat(0),
    seqs: [MAX_SLOTS]struct { pos: u32 = 0, last: u32 = 0 } = @splat(.{}),
    // the shared K/V pool (allocPool): every attention layer's pages, each slot's page table and the pages it holds
    pool: u32 = 0, // pages in each attention layer's pool (0: per-slot caches)
    page_bytes: usize = 0, // bytes of one K (or V) page of one layer: PAGE positions of the card's K/V heads
    pt: [MAX_SLOTS]?*anyopaque = @splat(null), // each slot's page table on the device: u32 [ceil(context / PAGE)]
    pt_host: [MAX_SLOTS][]u32 = @splat(&.{}), // and on the host, where the runner maps and keeps it
    held: [MAX_SLOTS]u32 = @splat(0), // entries of each slot's table that map a page
    pages: pages_mod.Pages = .{ .gpa = alloc, .count = 0, .refs = &.{} }, // the pool's references and free pages
    /// XPU_KEEP_SHARED: kept prompt states hold pool pages beside the live streams.
    keep_shared: bool = false,
    /// Pages the pool keeps free for the streams' next calls: a page a slot for its next row, a chunk's pages a slot.
    reserve: u32 = 0,
    /// The kept states' eviction when a stream needs pages the pool does not have free (null: a follower, or no cache).
    evict: ?Evict = null,
    /// Last park or restore K/V copy: bytes and milliseconds.
    park_moved: u64 = 0,
    park_ms: f64 = 0,
    /// Target hidden states for a drafter: when on, every forward fills acts.taps and calls `on_taps` with its rows.
    taps_on: bool = false,
    on_taps: ?struct { ctx: *anyopaque, f: *const fn (*anyopaque, *anyopaque, u32) anyerror!void } = null,

    /// Loads `dir` (MLX affine 4/64 or GPTQ 4/128) onto the first Arc Pro B70 (PCI id 0xe223), room for `context` rows.
    pub fn init(dir: []const u8, context: u32) !*Runner {
        if (live) return error.RunnerOpen;
        // the process's own options first, then this open's environment: an earlier open's XPU_* settings stay out
        if (defaults) |o| restore(o) else defaults = options();
        live = true;
        errdefer live = false;
        if (std.c.getenv("XPU_DUMP")) |v| dump_dir = std.mem.span(v);
        if (std.c.getenv("XPU_PREFILL_INT8") != null) prefill_int8 = true;
        if (std.c.getenv("XPU_EMBED_HOST") != null) embed_host = true; // host-backed embedding table (research arm)
        if (std.c.getenv("XPU_KV_INT8") != null) kv_i8 = true;
        if (std.c.getenv("XPU_PREFILL_INT4") != null) prefill_int4 = true;
        if (std.c.getenv("XPU_PREFILL_INT4_SITES")) |v| int4_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_KV8")) |v| kv8 = try std.fmt.parseInt(i32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_GDN")) |v| gdn_bits = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8_ROT") != null) a8_rot = 1;
        if (std.c.getenv("XPU_A8_SITES4")) |v| a8_sites4 = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8_BITS")) |v| a8_levels = if (std.mem.eql(u8, std.mem.span(v), "4")) 7 else 127;
        if (std.c.getenv("XPU_A8_SITES")) |v| a8_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8")) |v| a8_group = if (std.mem.eql(u8, std.mem.span(v), "token")) std.math.maxInt(u32) else try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_FUSE")) |v| fuse = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_DUMP_H")) |v| dump_h = open(v, 0o2101, @as(c_uint, 0o644)); // append
        if (std.c.getenv("XPU_CHUNK")) |v| CHUNK = try limits.chunk(std.mem.span(v));
        if (std.c.getenv("TF_TP_RANK")) |v| RANK = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        k1mod.readEnv(); // XPU_K1, XPU_K1_WAVE
        if (RANK >= TP) return error.RankOutOfRange;
        if (fuse != 0) std.debug.print("fuse: rank {d}: XPU_FUSE={d} bit 1: the exchange joined into the norm that follows (mixer and MLP)\n", .{ RANK, fuse });
        // the shared K/V pool: a page holds int8 rows and their scales, so pool mode is int8 K/V
        if (std.c.getenv("XPU_KV_POOL")) |v| kv_pool = v[0] != '0';
        if (std.c.getenv("XPU_KV_POOL_TOKENS")) |v| pool_tokens = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (pool_tokens != null) kv_pool = true;
        if (kv_pool) kv_i8 = true;
        if (std.c.getenv("XPU_KEEP_SHARED")) |v| keep_shared = v[0] != '0';
        if (std.c.getenv("XPU_KEEP_PAGES")) |v| keep_pages = v[0] != '0';
        if (keep_pages and !(kv_pool and keep_shared)) {
            if (RANK == 0) std.log.warn("qwen27-xpu: rank {d}: XPU_KEEP_PAGES needs XPU_KV_POOL=1 and XPU_KEEP_SHARED=1; left off", .{RANK});
            keep_pages = false;
        }
        if (std.c.getenv("XPU_KEPT_HOST")) |v| kept_host = v[0] != '0';
        if (std.c.getenv("XPU_KEPT_HOST_GIB")) |v| kept_host_gib = try std.fmt.parseFloat(f64, std.mem.span(v));
        if (std.c.getenv("XPU_PARK_GIB")) |v| park_gib = try std.fmt.parseFloat(f64, std.mem.span(v));
        if (std.c.getenv("XPU_PARK_PINNED")) |v| park_pinned = v[0] != '0';
        if (std.c.getenv("XPU_PARK_LOG")) |v| park_log = v[0] != '0';
        if (std.c.getenv("XPU_PARK_MIN")) |v| park_min = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (park_gib > 0) {
            if (!keep_pages or !kept_host) {
                std.log.warn("qwen27-xpu: XPU_PARK_GIB needs XPU_KEEP_PAGES=1 and XPU_KEPT_HOST=1; the host tier stays off", .{});
                park_gib = 0;
                park_on = false;
            } else park_on = true;
        }
        // K1 tree (XPU_K1=1 with XPU_K1_TREE=1) needs the int8 pool; print once so a run shows which path it takes
        if (k1mod.on and k1mod.tree and kv_i8) std.debug.print("K1 tree verify is on\n", .{});
        if (k1mod.on or k1mod.tree or k1mod.multi_tree) {
            var cov: [200]u8 = undefined;
            std.debug.print("{s}\n", .{k1mod.coverageLine(&cov, k1mod.on and kv_i8, k1mod.tree and kv_i8, k1mod.multi_tree and kv_i8)});
        }
        MAX_CTX = try limits.context(context);
        {
            const marker = try std.fmt.allocPrintSentinel(alloc, "{s}/hadamard128", .{dir}, 0);
            defer alloc.free(marker);
            const fd = open(marker.ptr, 0);
            rot_on = if (fd >= 0) 1 else 0;
            if (fd >= 0) _ = close(fd);
        }
        if (prefill_int4 and (rot_on == 0 or !prefill_int8)) return error.Int4NeedsRotatedWeightsAndInt8;
        var ckpt: Checkpoint = .{};
        errdefer ckpt.deinit();
        {
            const index_path = try std.fmt.allocPrintSentinel(alloc, "{s}/model.safetensors.index.json", .{dir}, 0);
            defer alloc.free(index_path);
            const fd = open(index_path.ptr, 0);
            if (fd < 0) return error.FileNotFound;
            defer _ = close(fd);
            const end = lseek(fd, 0, 2);
            if (end <= 0) return error.BadIndex;
            const size: usize = @intCast(end);
            const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed;
            if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed; // MAP_FAILED
            defer _ = munmap(base, size);
            const parsed = std.json.parseFromSlice(std.json.Value, alloc, @as([*]const u8, @ptrCast(base))[0..size], .{}) catch return error.BadIndex;
            defer parsed.deinit();
            const map = if (parsed.value == .object) parsed.value.object.get("weight_map") else null;
            if (map == null or map.? != .object) return error.BadIndex;
            var seen: std.ArrayList([]const u8) = .empty;
            defer seen.deinit(alloc);
            var it = map.?.object.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* != .string) return error.BadIndex;
                const f = e.value_ptr.string;
                for (seen.items) |s| {
                    if (std.mem.eql(u8, s, f)) break;
                } else {
                    try seen.append(alloc, f);
                    const shard = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f }, 0);
                    defer alloc.free(shard);
                    try ckpt.addShard(shard);
                }
            }
        }
        const r = try alloc.create(Runner);
        errdefer alloc.destroy(r);
        r.* = .{ .d = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined }, .model = undefined, .acts = undefined, .context = context, .ckpt = ckpt, .keep_shared = keep_shared };
        fmt_mlx = !ckpt.has("layers.0.linear_attn.in_proj_qkv.trellis") and !ckpt.has("in_proj_qkv.qweight") and ckpt.has("lm_head.scales");
        tp.group = try tp.Group.open(&r.d); // null at TP 1
        if (tp.group) |*g| {
            if (g.rank != 0 and g.readMode() != modeWord()) {
                std.log.err("rank {d}: the mode word does not match rank 0's: this rank's XPU_KV_POOL, XPU_KEEP_SHARED, XPU_KEPT_HOST, XPU_PARK_GIB, XPU_KV_INT8, XPU_EMBED_HOST, XPU_K1, XPU_K1_TREE, XPU_MULTI_TREE, XPU_K1_R16 or XPU_K1_WAVE diverges, or this rank loads a different weight format (EXL3 or MLX), and the pair sizes its cards differently\n", .{RANK});
                g.markClosed(); // rank 0 sees the closed segment at its first all-reduce and exits at once
                return error.ModeMismatch;
            }
        }
        if (park_on and RANK == 0) std.log.info("qwen27-xpu: rank {d}: idle kept prompt states park in host memory (XPU_PARK_GIB): {d} GiB a rank", .{ RANK, park_gib / @as(f64, @floatFromInt(TP)) });
        const d = &r.d;
        errdefer d.g.close();
        d.list = try d.g.immediate(0, 0);
        const mg = try d.g.module(&kb.glue);
        const md = try d.g.module(&kb.decode);
        const mm = try d.g.module(&kb.gemm);
        const mgd = try d.g.module(&kb.gdn);
        const mw = try d.g.module(&kb.ngen_wide);
        const mn = try d.g.module(&kb.ngen_narrow);
        const mm128 = try d.g.module(&kb.gemm128);
        const mw128 = try d.g.module(&kb.ngen_wide128);
        const mn128 = try d.g.module(&kb.ngen_narrow128);
        d.k = .{
            .embed = try d.kernel(mg, "embed", 256),
            .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 128),
            .add_rmsnorm_dec = try d.kernel(mg, "add_rmsnorm_dec", 512),
            .gdn_pre = try d.kernel(mg, "gdn_pre", 256),
            .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256),
            .gated_norm = try d.kernel(mg, "gated_norm", 256),
            .attn_prep = try d.kernel(mg, "attn_prep", 256),
            .attn_gate = try d.kernel(mg, "attn_gate", 256),
            .swiglu = try d.kernel(mg, "swiglu", 256),
            .argmax = try d.kernel(mg, "argmax", 256),
            .argmax_part = try d.kernel(mg, "argmax_part", 256),
            .argmax_part_masked = try d.kernel(mg, "argmax_part_masked", 256),
            .tap = try d.kernel(mg, "tap", 256),
            .conv_commit = try d.kernel(mg, "conv_commit", 256),
            .repack = try d.kernel(mg, "repack", 16),
            .gdn_step = try d.kernel(md, "gdn_step", 16),
            .attn_decode = try d.kernel(md, "attn_decode", 16),
            .attn_merge = try d.kernel(md, "attn_merge", 16),
            .kv_move = try d.kernel(md, "kv_move", 16),
            .exl3_gemv = try d.kernel(md, "exl3_gemv", 16),
            .rot_heads = try d.kernel(mg, "rot_heads", 16),
            .tp_add = try d.kernel(mg, "tp_add", 256),
            .tp_add_flag = try d.kernel(mg, "tp_add_flag", 256),
            .tp_norm_tp = try d.kernel(mg, "add_rmsnorm_dec_tp", 512),
            .tp_norm_peer = try d.kernel(mg, "add_rmsnorm_dec_peer", 512),
            .tp_consume = try d.kernel(mg, "tp_consume", 16),
            .tp_probe = try d.kernel(mg, "tp_probe", 16),
            .attn_prompt = try d.kernel(md, "attn_prompt", 128),
            .attn_prompt8 = try d.kernel(md, "attn_prompt8", 128),
            .embed_dense = try d.kernel(mg, "embed_dense", 256),
            .dense_rows = try d.kernel(mg, "dense_rows", 256),
            .repack_gptq = try d.kernel(mg, "repack_gptq", 16),
            .quantize_mlx = try d.kernel(mg, "quantize_mlx", 16),
            .dec_wide_g128 = try d.kernel(mm128, "w4k1_decode", 64),
            .dec_wide64_g128 = try d.kernel(mm128, "w4k1_d64", 64),
            .dec_narrow_g128 = try d.kernel(mm128, "w4k_dsplit", 64),
            .ngen_wide_g128 = try d.kernel(mw128, "w4g_dpb", 256),
            .ngen_narrow_g128 = try d.kernel(mn128, "w4g_s4dpb", 256),
            .ngen_swiglu = try d.kernel(try d.g.module(&kb.ngen_swiglu), "w4g_dpb_swiglu", 256),
            .ngen_swiglu_g128 = try d.kernel(try d.g.module(&kb.ngen_swiglu128), "w4g_dpb_swiglu", 256),
            .rowsums_ff = try d.kernel(mg, "rowsums_ff", 128),
            .fakequant_rows = try d.kernel(mg, "fakequant_rows", 256),
            .a8 = try d.kernel(try d.g.module(&kb.ngen_a8), "w4a8", 256),
            .a8_swiglu = try d.kernel(try d.g.module(&kb.ngen_a8_swiglu), "w4a8_swiglu", 256),
            .quant_a8 = try d.kernel(mg, "quant_a8", 128),
            .quant_a4 = try d.kernel(mg, "quant_a4", 128),
            .a4 = try d.kernel(try d.g.module(&kb.ngen_a4), "w4a4", 256),
            .dec_wide = try d.kernel(mm, "w4k1_decode", 64),
            .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64),
            .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
            .gdn_prep = try d.kernel(mgd, "gdn_prep", 64),
            .gdn_state = try d.kernel(mgd, "gdn_state", 256),
            .gdn_out = try d.kernel(mgd, "gdn_out", 128),
            .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
            .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256),
        };
        // the shared rounds' kernels reach each slot's states through the per-row table (acts.mt)
        if (gdn_bits != 0) { // lane p4-k-gdn: one line a rank, the check of the flag
            const step1 = d.k.gdn_step;
            const step2 = if (gdn_bits & 6 != 0) try d.kernel(md, "gdn_step2", 16) else step1;
            std.debug.print("gdn: XPU_GDN={d} (stash in place {s}, gdn_step2 {s})\n", .{ gdn_bits, if (gdn_bits & 1 != 0) "on" else "off", if (gdn_bits & 2 != 0) "on" else "off" });
            if (gdn_bits & 4 != 0) try @import("gdn_check.zig").run(d, step1, step2); // the two kernels on this card: the same bits
            if (gdn_bits & 2 != 0) d.k.gdn_step = step2;
        }
        for ([_]ze.H{ d.k.attn_prep, d.k.attn_decode, d.k.gdn_pre, d.k.gdn_step }) |k| try d.g.indirect(k);
        if (k1mod.on or k1mod.multi_tree) { // XPU_K1: the K1 decode kernel (k1.cl), the scalar page-table kernel and the merge
            const mk1 = try d.g.module(&kb.k1);
            const kk = try k1mod.load(d, md, mk1);
            d.k.attn_k1 = kk.decode;
            d.k.attn_k1_16 = kk.decode16;
            d.k.attn_k1_draft = kk.draft;
            d.k.attn_k1_draft_mt = kk.draft_mt;
            k1mod.have16 = kk.decode16 != null;
            d.k.attn_span = kk.span;
            d.k.merge_span = kk.merge2; // K1's partials are log2-base: the exp2 merge
        }
        if (std.c.getenv("XPU_PROFILE")) |pv| {
            prof.on = true;
            prof.sites_on = pv[0] == '2'; // a row a call site and an event a device copy (lane p4-k-gdn)
            prof.events = try d.g.events(4096);
            inline for (@typeInfo(@TypeOf(d.k)).@"struct".field_names, 0..) |name, i| prof.names[i] = .{ .k = @field(d.k, name), .name = name };
        }
        if (std.c.getenv("XPU_MV")) |mv| { // lane p4-k-mv: one line a rank, the check of the flag
            var it = std.mem.tokenizeScalar(u8, std.mem.span(mv), ',');
            while (it.next()) |pair| {
                const colon = std.mem.indexOfScalar(u8, pair, ':') orelse return error.BadMvPick;
                const site = std.meta.stringToEnum(Site, pair[0..colon]) orelse return error.BadMvSite;
                const shadow = pair.len > colon + 1 and pair[colon + 1] == '@';
                const vname = pair[colon + 1 + @intFromBool(shadow) ..];
                const v = for (mv_variants) |v| {
                    if (std.mem.eql(u8, v.name, vname)) break v;
                } else return error.BadMvVariant;
                if (v.probe and !shadow) return error.MvProbeIsShadowOnly;
                const k = try d.kernel(mm, v.kernel, v.local);
                const row: []const u8 = if (shadow) try std.fmt.allocPrint(alloc, "mv_{s}_{s}", .{ v.name, @tagName(site) }) else v.row;
                if (shadow) {
                    if (mv_scratch == null) mv_scratch = try d.g.deviceAlloc(mv_scratch_bytes);
                    const slot = for (&mv_shadow[@intFromEnum(site)]) |*sl| {
                        if (sl.* == null) break sl;
                    } else return error.TooManyMvShadows;
                    slot.* = .{ .k = k, .div = v.div, .wide = v.wide };
                } else mv_pick[@intFromEnum(site)] = .{ .k = k, .div = v.div, .wide = v.wide };
                if (prof.on) {
                    const free = for (&prof.names) |*slot| {
                        if (slot.name.len == 0) break slot;
                    } else return error.NoProfileRowLeft; // a row of its own in the kernel table
                    free.* = .{ .k = k, .name = row };
                }
                if (shadow)
                    std.debug.print("mv: XPU_MV {s} shadow {s} ({s}), row {s}\n", .{ @tagName(site), v.name, std.mem.span(v.kernel), row })
                else
                    std.debug.print("mv: XPU_MV {s} on {s} ({s}){s}\n", .{ @tagName(site), v.name, std.mem.span(v.kernel), if (v.bits) ", the bits change: a timing probe" else "" });
            }
        }
        const prof_on = prof.on;
        prof.on = false; // not during the load
        r.model = try load(d, &ckpt);
        prof.on = prof_on;
        r.acts = try Acts.init(d);
        if (r.model.exl3) |x| try x.bind(d, &r.model, &r.acts); // the EXL3 plans over the activations
        r.store(); // slot 0: the states the load allocated
        try d.fill(r.acts.zeros, @as(u16, 0), maxRows() * GDN_V_HEADS * 128 * 2);
        try d.fill(r.acts.ones, @as(f32, 1.0), maxRows() * GDN_V_HEADS * 4);
        r.host_logits = try alloc.alloc(f32, VOCAB);
        // TF_TP_FLAGLAB=1: the flag lab (the socket open kept for it, lab_conn), just before the bench;
        // it ends the process. noE4 skips the real-protocol exchanges.
        if (tp.group) |*g| {
            if (std.c.getenv("TF_TP_FLAGLAB") != null and g.devflag) try g.flagLab(d, r.acts.r);
            // TF_TP_EXBENCH=1: the exchange cost a round at 1, 8, 16 and 32 rows, in the mode the group opened in.
            // Both ranks run it at load (the same count of exchanges), so the pair times as it does in a forward.
            if (std.c.getenv("TF_TP_EXBENCH") != null) try g.exchangeBench(d, r.acts.r);
        }
        return r;
    }

    /// Gives back the card's memory, the shards' mappings and the host buffers; `r` is gone after it.
    pub fn deinit(r: *Runner) void {
        if (tp.group) |*g| {
            g.record(.close, 0, 0, 0, &.{}) catch {};
            g.markClosed(); // a rank that stops closes the group: its peer sees it at once, not after the wait
        }
        r.d.deinit();
        prof.on = false;
        live = false;
        r.ckpt.deinit();
        if (r.host_logits.len > 0) alloc.free(r.host_logits);
        for (r.pt_host) |t| if (t.len > 0) alloc.free(t);
        if (r.pages.count > 0) r.pages.deinit();
        alloc.destroy(r);
    }

    /// Makes `slot` the one every call works on (states allocated on first use); other slots keep their state.
    pub fn select(r: *Runner, slot: u32) !void {
        if (tp.group) |*g| try g.record(.select, slot, 0, 0, &.{});
        if (slot >= MAX_SLOTS) return error.NoSuchSlot;
        if (slot == r.slot) return;
        try r.ensure(slot);
        r.store();
        r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
        for (&r.model.layers) |*l| {
            if (l.attn) {
                l.kc = l.slot_kc[slot];
                l.vc = l.slot_vc[slot];
                l.pt = l.slot_pt[slot];
            } else {
                l.rec = l.slot_rec[slot];
                l.rec_save = l.slot_save[slot];
                l.conv = l.slot_conv[slot];
                l.conv_cur = l.slot_cur[slot];
            }
        }
        r.pos = r.seqs[slot].pos;
        r.last = r.seqs[slot].last;
        r.slot = slot;
        r.block_rows = 0;
        r.tree_on = false;
    }

    pub const Win = multi.Win;
    pub const verifyMulti = multi.verifyMulti;
    pub const keepMulti = multi.keepMulti;
    pub const keepPaths = multi.keepPaths;
    pub const stepMulti = multi.stepMulti;

    /// The selected slot's state pointers back into its slot arrays (a verify's commit may have swapped them).
    pub fn store(r: *Runner) void {
        if (tp.group) |*g| g.record(.store, 0, 0, 0, &.{}) catch {};
        for (&r.model.layers) |*l| {
            if (l.attn) {
                l.slot_kc[r.slot] = l.kc;
                l.slot_vc[r.slot] = l.vc;
                l.slot_pt[r.slot] = l.pt;
            } else {
                l.slot_rec[r.slot] = l.rec;
                l.slot_save[r.slot] = l.rec_save;
                l.slot_conv[r.slot] = l.conv;
                l.slot_cur[r.slot] = l.conv_cur;
            }
        }
        r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
    }

    pub const Snap = slot_states.Snap;
    pub const slotBytes = slot_states.slotBytes;
    pub const snapBytes = slot_states.snapBytes;
    pub const saveSlot = slot_states.saveSlot;
    pub const restoreSlot = slot_states.restoreSlot;
    pub const dropSnap = slot_states.dropSnap;
    pub const parkSnap = slot_states.parkSnap;
    pub const unallocatedSlots = slot_states.unallocatedSlots;
    pub const ensure = slot_states.ensure;

    /// A fresh sequence: GDN and convolution states to zero (the KV cache is overwritten position by position).
    pub fn reset(r: *Runner) !void {
        if (tp.group) |*g| try g.record(.reset, 0, 0, 0, &.{});
        for (&r.model.layers) |*l| if (!l.attn) {
            try r.d.fill(l.rec.?, @as(u32, 0), GDN_V_HEADS * 128 * 128 * 4);
            for (l.conv) |cv| try r.d.fill(cv.?, @as(u16, 0), 3 * GDN_QKV * 2);
            l.conv_cur = 0;
        };
        r.pos = 0;
        if (r.pool != 0) r.releaseSlot(r.slot); // a fresh sequence takes fresh pages
    }

    // ------------------------------------------------------------ the shared K/V pool (XPU_KV_POOL)

    /// Bytes of one K (or V) page of one attention layer: PAGE positions of the card's K/V heads, their values then
    /// their scales (limits.KvLayout: Q27_KVROW is this rank's heads, half the model's at TP 2).
    pub fn pageBytes(r: *const Runner) usize {
        _ = r;
        const lay = slot_states.kv();
        return PAGE * (lay.row() + lay.scaleRow());
    }

    /// Bytes one position takes in K/V on this card: every attention layer, K and V.
    pub fn kvBytesPerToken(r: *const Runner) usize {
        _ = r;
        const lay = slot_states.kv();
        return (LAYERS / 4) * 2 * (lay.row() + lay.scaleRow());
    }

    /// The pool's device bytes: K pages then V pages in every attention layer.
    pub fn poolBytes(r: *const Runner) usize {
        var layers: usize = 0;
        for (r.model.layers) |l| layers += @intFromBool(l.attn);
        return @as(usize, r.pool) * r.page_bytes * 2 * layers;
    }

    /// The pool's tokens: what its pages hold for one stream.
    pub fn poolTokens(r: *const Runner) u64 {
        return @as(u64, r.pool) * PAGE;
    }

    /// The shared K/V pool: `pages` pages in each attention layer (K pages, then V pages, one allocation a layer,
    /// zeroed so the card holds them now), and each slot's page table, mapping nothing.
    pub fn allocPool(r: *Runner, pages: u32) !void {
        if (r.pool != 0) return error.PoolTwice;
        if (!kv_i8) return error.PoolNeedsInt8;
        if (pages == 0) return error.EmptyPool;
        r.pool = pages;
        r.page_bytes = r.pageBytes();
        const half = @as(usize, pages) * r.page_bytes;
        for (&r.model.layers) |*l| {
            if (!l.attn) continue;
            const base = try r.d.alloc_(2 * half);
            try zeroed(&r.d, base, 2 * half);
            l.kc = base;
            l.vc = off(base, half);
            for (&l.slot_kc) |*x| x.* = base; // every slot's view is the pool: its table says which pages
            for (&l.slot_vc) |*x| x.* = off(base, half);
        }
        const entries = (r.context + PAGE - 1) / PAGE;
        for (0..MAX_SLOTS) |i| {
            r.pt[i] = try r.d.alloc_(@as(usize, entries) * 4);
            r.pt_host[i] = try alloc.alloc(u32, @as(usize, entries));
            r.held[i] = 0;
        }
        r.pages = try pages_mod.Pages.init(alloc, pages);
        // K1 (XPU_K1) and the paged launches read a slot's table through the layer: `slot_pt` is the runner's
        // per-slot table (page ids into every attention layer's pool), `pt` the selected slot's.
        for (&r.model.layers) |*l| {
            if (!l.attn) continue;
            for (0..MAX_SLOTS) |i| l.slot_pt[i] = r.pt[i];
            l.pt = r.pt[r.slot];
        }
        try r.d.sync();
        std.debug.print("K/V pool: {d} pages, {d} tokens, {d:.2} GB a card, int8\n", .{ pages, @as(u64, pages) * PAGE, @as(f64, @floatFromInt(r.poolBytes())) / 1e9 });
    }

    /// One reference to page `id` less; the page returns to the pool at zero.
    pub fn releasePage(r: *Runner, id: u32) void {
        r.pages.release(id);
    }

    /// One reference to page `id` more (a kept prompt state holds the pages it maps).
    pub fn retainPage(r: *Runner, id: u32) void {
        r.pages.retain(id);
    }

    /// `retainPage` and one kept-state reference. The kept count runs with the page-count flag on or off.
    /// Named keepPage: other Runner methods already bind a parameter named keep.
    pub fn keepPage(r: *Runner, id: u32) void {
        r.pages.keep(id);
    }

    /// `releasePage` and one kept-state reference less.
    pub fn unkeepPage(r: *Runner, id: u32) void {
        r.pages.unkeep(id);
    }

    /// Page ids `ids` into `slot`'s table from entry `from`; the slot maps from + ids.len on. The caller holds them.
    pub fn mapPages(r: *Runner, slot: u32, from: u32, ids: []const u32) !void {
        if (r.pool == 0) return;
        if (slot >= MAX_SLOTS or r.pt[slot] == null or from > r.held[slot]) return error.BadSlot;
        if (from + @as(u32, @intCast(ids.len)) > r.pt_host[slot].len) return error.ContextFull;
        @memcpy(r.pt_host[slot][from..][0..ids.len], ids);
        for (ids) |id| r.pages.retain(id);
        r.held[slot] = from + @as(u32, @intCast(ids.len));
        // the host table stays put until the next mapping: the queued copy reads it before the kernels that follow
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, off(r.pt[slot].?, @as(usize, from) * 4), r.pt_host[slot].ptr + from, ids.len * 4, null, 0, null));
    }

    /// Pages for `slot`'s positions up to `upTo`, mapped in its table; error.ContextFull when the pool has none left.
    pub fn ensurePages(r: *Runner, slot: u32, upTo: u32) !void {
        if (r.pool == 0) return;
        if (upTo > r.context) return error.ContextFull;
        if (slot >= MAX_SLOTS or r.pt[slot] == null) return error.BadSlot;
        const need = (upTo + PAGE - 1) / PAGE;
        const from = r.held[slot];
        if (need <= from) return;
        var i = from;
        var full = false;
        while (i < need) : (i += 1) {
            const id = r.pages.take() orelse {
                full = true;
                break;
            };
            r.pt_host[slot][i] = id; // take gave the slot's reference
        }
        if (i > from) {
            r.held[slot] = i;
            try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, off(r.pt[slot].?, @as(usize, from) * 4), r.pt_host[slot].ptr + from, (i - from) * 4, null, 0, null));
        }
        if (full) return error.ContextFull;
    }

    /// `n` new pages for `slot`, from a held of 0. `take` already holds each page, so this does not retain again.
    /// Writes `pt_host`, `held`, and the device table, as `ensurePages` does.
    pub fn mapNewPages(r: *Runner, slot: u32, n: u32) !void {
        if (r.pool == 0 or slot >= MAX_SLOTS or r.pt[slot] == null or r.held[slot] != 0) return error.BadSlot;
        if (@as(usize, n) > r.pt_host[slot].len) return error.ContextFull;
        var i: u32 = 0;
        errdefer {
            for (r.pt_host[slot][0..i]) |id| r.releasePage(id);
            r.held[slot] = 0;
        }
        while (i < n) : (i += 1) {
            r.pt_host[slot][i] = r.pages.take() orelse return error.ContextFull;
        }
        r.held[slot] = n;
        if (n == 0) return;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.pt[slot].?, r.pt_host[slot].ptr, @as(usize, n) * 4, null, 0, null));
    }

    /// Pages a call on `slot` takes up to `upTo` positions, made free before the call. With XPU_KEEP_SHARED the least
    /// recently used kept states go until the pool has them and keeps `reserve` free for the streams. Rank 0 only: the
    /// drops it makes are step records the follower replays before this call, so both ranks free the same pages.
    pub fn makeRoom(r: *Runner, slot: u32, upTo: u32) !void {
        if (r.pool == 0 or !r.keep_shared) return;
        const e = r.evict orelse return; // a follower: its drops arrive as records, in order before the call
        if (slot >= MAX_SLOTS or r.pt[slot] == null or upTo > r.context) return;
        const need = (upTo + PAGE - 1) / PAGE;
        const from = r.held[slot];
        if (need <= from) return;
        const want = need - from;
        while (!r.pages.roomFor(want, r.reserve)) {
            if (!e.f(e.ctx)) break; // no kept state left: the pass takes what the pool has
        }
    }

    /// Pages a parked restore takes, made free before rank 0 records `restore_slot`. The slot's pages with one
    /// reference come back when the restore releases the slot, so they count. A follower has no evict hook: it
    /// never decides. Every follower replays the records this call writes before it sees `restore_slot`.
    pub fn makeRoomParked(r: *Runner, slot: u32, at: u32) void {
        if (r.pool == 0 or !r.keep_shared) return;
        const e = r.evict orelse return;
        if (slot >= MAX_SLOTS or r.pt[slot] == null or at > r.context) return;
        const need = (at + PAGE - 1) / PAGE;
        while (!r.parkedRoom(slot, need)) {
            if (!e.f(e.ctx)) break;
        }
    }

    fn parkedRoom(r: *const Runner, slot: u32, need: u32) bool {
        var back: u32 = 0;
        for (r.pt_host[slot][0..r.held[slot]]) |id| {
            if (r.pages.sole(id)) back += 1;
        }
        return @as(u64, r.pages.spare()) + back >= @as(u64, need) + r.reserve;
    }

    /// Drops `slot`'s hold on its pages: a fresh sequence, or a restore that maps a kept state's pages.
    pub fn releaseSlot(r: *Runner, slot: u32) void {
        if (r.pool == 0 or slot >= MAX_SLOTS or r.pt[slot] == null) return;
        for (r.pt_host[slot][0..r.held[slot]]) |id| r.releasePage(id);
        r.held[slot] = 0;
    }

    /// Positions `slot` may hold: what its pages cover, at most its context.
    pub fn room(r: *const Runner, slot: u32) u32 {
        if (r.pool == 0) return r.context;
        if (slot >= MAX_SLOTS) return 0;
        return @min(r.context, r.held[slot] * PAGE);
    }

    /// A pool that gives each slot its own run of ceil(context / PAGE) pages: a slot holds its context, as before the pool.
    pub fn poolFixed(r: *Runner) !void {
        const e = (r.context + PAGE - 1) / PAGE;
        try r.allocPool(MAX_SLOTS * e);
        const ids = try alloc.alloc(u32, @as(usize, e));
        defer alloc.free(ids);
        for (0..MAX_SLOTS) |i| {
            for (ids, 0..) |*p, j| p.* = @intCast(i * e + j);
            try r.mapPages(@intCast(i), 0, ids);
        }
    }

    /// Feeds prompt tokens in chunks; `last`: the final chunk, whose last row's logits are kept.
    pub fn prompt(r: *Runner, ids: []const u32, last: bool) !void {
        // a resumed pass needs the pages of the whole request before it starts: rank 0 makes room first, so the
        // drops it makes are step records the follower replays before this call
        const upto: u64 = @as(u64, r.pos) + ids.len;
        if (upto <= r.context) try r.makeRoom(r.slot, @intCast(upto));
        if (tp.group) |*g| try g.record(.prompt, @intFromBool(last), 0, 0, ids);
        var at: usize = 0;
        while (at < ids.len) {
            const end = chunkEnd(at, ids.len);
            const want = last and end == ids.len;
            if (r.pos + (end - at) > r.context) return error.ContextFull;
            try r.ensurePages(r.slot, r.pos + @as(u32, @intCast(end - at)));
            if (try forward(&r.d, &r.model, &r.acts, @ptrCast(ids[at..end]), r.pos, .{ .prompt = true, .want = if (want) .last else .none, .taps = r.taps_on, .pt = r.pt[r.slot] })) |t| r.last = @intCast(t);
            if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, @intCast(end - at));
            r.pos += @intCast(end - at);
            at = end;
        }
    }

    /// One decoded token in; the next one's logits kept.
    pub fn step(r: *Runner, token: u32) !void {
        try r.makeRoom(r.slot, r.pos + 1);
        if (tp.group) |*g| try g.record(.step, token, 0, 0, &.{});
        if (r.pos + 1 > r.context) return error.ContextFull;
        try r.ensurePages(r.slot, r.pos + 1);
        const id: i32 = @intCast(token);
        r.last = @intCast((try forward(&r.d, &r.model, &r.acts, &.{id}, r.pos, .{ .taps = r.taps_on, .pt = r.pt[r.slot] })).?);
        r.pos += 1;
        if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, 1);
    }

    /// Verifies a block: `ids` (≤ 16 rows) run from `pos`, each row's greedy token returned; state moves till `commit`.
    pub fn verify(r: *Runner, ids: []const u32) ![]const u32 {
        if (ids.len > 0 and @as(u64, r.pos) + ids.len <= r.context) try r.makeRoom(r.slot, r.pos + @as(u32, @intCast(ids.len)));
        if (tp.group) |*g| try g.record(.verify, 0, 0, 0, ids);
        if (ids.len == 0 or ids.len > 16) return error.BadBlock;
        if (r.pos + ids.len > r.context) return error.ContextFull;
        try r.ensurePages(r.slot, r.pos + @as(u32, @intCast(ids.len)));
        _ = try forward(&r.d, &r.model, &r.acts, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on, .pt = r.pt[r.slot] });
        var toks: [16]i32 = undefined;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, ids.len * 4, null, 0, null));
        try r.d.sync();
        for (r.block[0..ids.len], toks[0..ids.len]) |*o, t| o.* = @intCast(t);
        r.block_rows = @intCast(ids.len);
        return r.block[0..ids.len];
    }

    /// Keeps the first `keep` rows of the last verified block: GDN and convolution states replay those rows from entry.
    pub fn commit(r: *Runner, rows: u32) !void {
        if (tp.group) |*g| try g.record(.commit, rows, 0, 0, &.{});
        if (rows == 0 or rows > r.block_rows or r.tree_on) return error.BadCommit;
        const nul: ?*anyopaque = null;
        for (&r.model.layers) |*l| if (!l.attn) {
            if (rows == r.block_rows) { // every row kept: the block's final state is the state
                std.mem.swap(?*anyopaque, &l.rec, &l.rec_save);
                continue;
            }
            const st = l.stash.?;
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = GDN_V_HEADS }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, l.rec.?, r.acts.y, @as(i32, @intCast(rows)), nul, nul, nul, nul });
            // the block's forward flipped conv_cur: conv[cur] is the state after all rows, conv[1 - cur] the entry
            try r.d.launch(r.d.k.conv_commit, .{ .x = GDN_QKV / 256 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, @as(i32, @intCast(rows)), nul });
        };
        r.pos += rows;
        r.last = r.block[rows - 1];
        r.block_rows = 0; // no wait: the next commands queue behind these on the in-order list
    }

    /// Verifies a tree: row 0 is the pending token, `parents[t]` the row below row t; returns each row's greedy token.
    pub fn verifyTree(r: *Runner, ids: []const u32, parents: []const i32) ![]const u32 {
        // a tree's deepest row is at most ids.len - 1 below pos: ids.len pages cover it
        if (ids.len > 0 and @as(u64, r.pos) + ids.len <= r.context) try r.makeRoom(r.slot, r.pos + @as(u32, @intCast(ids.len)));
        if (tp.group) |*g| { // ids then parents, each ids.len words
            var both: [32]u32 = undefined;
            const n: usize = @min(ids.len, 16); // usize: @min with 16 narrows to u5, and 2 * n would wrap to 0
            @memcpy(both[0..n], ids[0..n]);
            for (parents[0..@min(parents.len, n)], 0..) |p, i| both[n + i] = @bitCast(p);
            try g.record(.verify_tree, @intCast(n), 0, 0, both[0 .. 2 * n]);
        }
        if (ids.len == 0 or ids.len > 16 or parents.len != ids.len) return error.BadBlock;
        var depth_max: u32 = 0;
        for (0..ids.len) |t| {
            const row = r.tree_host[t * 17 ..][0..17];
            if (t == 0) {
                row[0] = 0;
                row[1] = 0;
                continue;
            }
            const p: usize = @intCast(parents[t]);
            if (p >= t) return error.BadBlock;
            const pr = r.tree_host[p * 17 ..][0..17];
            row[0] = pr[0] + 1;
            @memcpy(row[1 .. 1 + @as(usize, @intCast(pr[0])) + 1], pr[1 .. 1 + @as(usize, @intCast(pr[0])) + 1]);
            row[1 + @as(usize, @intCast(row[0]))] = @intCast(t);
            depth_max = @max(depth_max, @as(u32, @intCast(row[0])));
        }
        for (1..ids.len) |t| { // LOAD: the parent is not the row before ; SAVE: a child is not the row after
            const p: usize = @intCast(parents[t]);
            if (p + 1 != t) {
                r.tree_host[t * 17] |= 512;
                r.tree_host[p * 17] |= 256;
            }
        }
        if (r.pos + depth_max + 1 > r.context or r.pos + ids.len > r.context) return error.ContextFull;
        // The draft K/V rows sit at pos + the row index, which reaches ids.len - 1: the tree table's rows past
        // depth_max - 1 (a comb's row 15) need the same pages as the draft (lane p22-k1tree, issue 3).
        try r.ensurePages(r.slot, r.pos + @as(u32, @intCast(ids.len)));
        try dev.hostCopy(&r.d, r.acts.tree, &r.tree_host, ids.len * 17 * 4);
        _ = try forward(&r.d, &r.model, &r.acts, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on, .tree = true, .pt = r.pt[r.slot] });
        var toks: [16]i32 = undefined;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, ids.len * 4, null, 0, null));
        try r.d.sync();
        for (r.block[0..ids.len], toks[0..ids.len]) |*o, t| o.* = @intCast(t);
        r.block_rows = @intCast(ids.len);
        r.tree_on = true;
        return r.block[0..ids.len];
    }

    /// Keeps the root path `path` of the last verified tree: states replayed along it, K/V moved, taps gathered.
    pub fn commitPath(r: *Runner, path: []const u32) !void {
        if (tp.group) |*g| try g.record(.commit_path, 0, 0, 0, path);
        if (!r.tree_on or path.len == 0 or path.len > r.block_rows or path[0] != 0) return error.BadCommit;
        var straight = true;
        for (path, 0..) |row, dd| {
            r.path_host[dd] = @intCast(row);
            straight = straight and row == dd;
        }
        const rows: i32 = @intCast(path.len);
        try dev.hostCopy(&r.d, r.acts.path, &r.path_host, path.len * 4);
        const nul: ?*anyopaque = null;
        for (&r.model.layers) |*l| {
            if (l.attn) {
                if (!straight) try r.d.launch(r.d.k.kv_move, .{ .x = ATT_KV_HEADS }, .{ l.kc.?, l.vc.?, @as(i32, @intCast(r.pos)), r.acts.path, rows, kvCap(), r.pt[r.slot] });
                continue;
            }
            const st = l.stash.?;
            prof.site = .gdn_commit;
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = GDN_V_HEADS }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, l.rec.?, r.acts.y, rows, r.acts.path, nul, nul, nul });
            prof.site = .none;
            try r.d.launch(r.d.k.conv_commit, .{ .x = GDN_QKV / 256 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, rows, r.acts.path });
        }
        if (r.taps_on and !straight) for (path, 0..) |row, dd| if (row != dd)
            try copy(&r.d, off(r.acts.taps, dd * 5 * D * 2), off(r.acts.taps, row * 5 * D * 2), 5 * D * 2);
        r.pos += @intCast(path.len);
        r.last = r.block[path[path.len - 1]];
        r.block_rows = 0;
        r.tree_on = false;
        try r.d.sync(); // path_host is read by the queued copy
    }

    /// Structured output: each row's greedy token among those its grammar `bits` allow, computed on the device.
    pub fn argmaxMasked(r: *Runner, rows: []const u32, bits: []const u32, words: usize, out: []u32) !void {
        const n = rows.len;
        if (n == 0 or n > 16 or words > MASK_WORDS or bits.len != n * words or out.len < n) return error.BadMasks;
        const d = &r.d;
        var rh: [16]i32 = undefined;
        for (rows, 0..) |x, j| rh[j] = @intCast(x);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mrows, &rh, n * 4, null, 0, null));
        errdefer d.settle(); // rh and toks live in this frame and bits is the caller's: errors wait for the queue
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mbits, bits.ptr, bits.len * 4, null, 0, null));
        try d.launch(d.k.argmax_part_masked, .{ .x = 256, .y = @intCast(n) }, .{ r.acts.logits, @as(i32, VOCAB), r.acts.mbits, @as(i32, @intCast(words)), r.acts.mrows, r.acts.pv, r.acts.pi });
        try d.launch(d.k.argmax, .{ .x = @intCast(n) }, .{ r.acts.pv, r.acts.pi, r.acts.mtok });
        var toks: [16]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, r.acts.mtok, n * 4, null, 0, null));
        try d.sync();
        for (out[0..n], toks[0..n]) |*o, t| o.* = @intCast(t);
    }

    /// The logits of row `row` of the last verified block (or of the last forward's row, row 0), fp32 on the host.
    pub fn rowLogits(r: *Runner, row: u32) ![]const f32 {
        const raw = try alloc.alloc(f16, VOCAB);
        defer alloc.free(raw);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, off(r.acts.logits, row * VOCAB * 2), VOCAB * 2, null, 0, null));
        try r.d.sync();
        for (r.host_logits, raw) |*o, v| o.* = v;
        return r.host_logits;
    }

    /// The kept row's logits, fp32 on the host (for sampling).
    pub fn logits(r: *Runner) ![]const f32 {
        const raw = try alloc.alloc(f16, VOCAB);
        defer alloc.free(raw);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, r.acts.logits, VOCAB * 2, null, 0, null));
        try r.d.sync();
        for (r.host_logits, raw) |*o, v| o.* = v;
        return r.host_logits;
    }

    /// The device's name, as the server reports the chip.
    pub fn deviceName(r: *Runner) []const u8 {
        return std.mem.sliceTo(&r.d.g.props.name, 0);
    }
};

/// Zeroes `bytes` of device memory `p` in chunks: the pool's pages are held by the card now, not at a request.
fn zeroed(d: *Dev, p: *anyopaque, bytes: usize) !void {
    const first = @min(bytes, 64 << 20);
    try d.fill(p, @as(u32, 0), first);
    var at: usize = first;
    while (at < bytes) {
        const n = @min(first, bytes - at);
        try dev.copy(d, off(p, at), p, n);
        at += n;
    }
    try d.sync();
}

/// Card bytes a prompt pass and the driver hold beyond the counted load (Sage's measured 0.31 GB past the count).
const pass_reserve: u64 = 650_000_000;
/// Bytes the card holds beyond what Dev.allocated counts (the runtime's own buffers).
const held_not_counted: u64 = 400_000_000;
/// Card bytes kept for the GDN, convolution and drafter-context copies of kept prompt states (their K/V pages come
/// from the pool). Not taken with XPU_KEPT_HOST=1: the copies live in host memory.
pub const cache_gdn: u64 = 1536 << 20;
/// Pages a pool is rounded down to: both ranks of a TP 2 pair size the same pool, so they map the same page ids.
const sync_pages: u32 = 256;

/// The pool's pages: `tokens` rounded up to whole pages, else what the card leaves after the load, the drafter, the
/// GDN states of `slots` streams, their drafter contexts (`extra` bytes a stream), the activations and the reserve
/// (and, with a prompt cache, its GDN copies).
pub fn poolPages(r: *Runner, slots: u32, tokens: ?u32, cache: bool, extra: u64) !u32 {
    if (tokens) |t| return @max(1, (t + PAGE - 1) / PAGE);
    const mem = try r.d.g.memory();
    const keep = pass_reserve + held_not_counted + (if (cache and !kept_host) cache_gdn else 0);
    const gdn: u64 = @intCast(slot_states.slotBytes(r));
    const counted = @as(u64, r.d.allocated) + (gdn + extra) * @as(u64, if (slots > 1) slots - 1 else 0);
    const left = (mem.size -| keep) -| counted;
    var layers: usize = 0;
    for (r.model.layers) |l| layers += @intFromBool(l.attn);
    const page = @as(u64, r.pageBytes()) * 2 * layers; // one page id: K and V pages in every attention layer
    const pages: u64 = left / page;
    return @intCast(pages - pages % sync_pages);
}

/// The run flags a pair must agree on: one bit each for kv_pool, keep_shared, kv_i8, embed_host, k1, k1_tree and
/// k1 r16, four bits for the k1 wave (XPU_K1_WAVE, a small count), then multi_tree (XPU_MULTI_TREE), then the weight
/// format (bit 12: MLX), then kept_host (bit 13: XPU_KEPT_HOST), then the host tier (bit 23: XPU_PARK_GIB).
pub fn modeWord() u64 {
    var w: u64 = 0;
    w |= @as(u64, @intFromBool(kv_pool));
    w |= @as(u64, @intFromBool(keep_shared)) << 1;
    w |= @as(u64, @intFromBool(kv_i8)) << 2;
    w |= @as(u64, @intFromBool(embed_host)) << 3;
    w |= @as(u64, @intFromBool(k1mod.on)) << 4;
    w |= @as(u64, @intFromBool(k1mod.tree)) << 5;
    w |= @as(u64, @intFromBool(k1mod.r16)) << 6;
    w |= (@as(u64, k1mod.wave) & 15) << 7;
    w |= @as(u64, @intFromBool(k1mod.multi_tree)) << 11;
    w |= @as(u64, @intFromBool(fmt_mlx)) << 12; // 0: EXL3, 1: MLX 4/64
    w |= @as(u64, @intFromBool(kept_host)) << 13; // kept states' copies in host memory: both ranks hold theirs there
    w |= @as(u64, @intFromBool(park_on)) << 23; // idle kept states park in host memory: all ranks, or neither
    w |= @as(u64, @intFromBool(fuse & 1 != 0)) << 14; // XPU_FUSE bit 1: both ranks fuse the exchange into the norm
    return w;
}

test "mode word: each flag sits at the bit both ranks agree on" {
    const saved_pool = kv_pool;
    const saved_keep = keep_shared;
    const saved_i8 = kv_i8;
    const saved_embed = embed_host;
    const saved_on = k1mod.on;
    const saved_tree = k1mod.tree;
    const saved_mt = k1mod.multi_tree;
    const saved_r16 = k1mod.r16;
    const saved_wave = k1mod.wave;
    const saved_fmt = fmt_mlx;
    const saved_kh = kept_host;
    const saved_park = park_on;
    defer {
        park_on = saved_park;
        kept_host = saved_kh;
        fmt_mlx = saved_fmt;
        kv_pool = saved_pool;
        keep_shared = saved_keep;
        kv_i8 = saved_i8;
        embed_host = saved_embed;
        k1mod.on = saved_on;
        k1mod.tree = saved_tree;
        k1mod.multi_tree = saved_mt;
        k1mod.r16 = saved_r16;
        k1mod.wave = saved_wave;
    }
    kv_pool = false;
    keep_shared = false;
    kv_i8 = false;
    embed_host = false;
    k1mod.on = false;
    k1mod.tree = false;
    k1mod.multi_tree = false;
    k1mod.r16 = true;
    k1mod.wave = 4;
    fmt_mlx = false;
    kept_host = false;
    park_on = false;
    try std.testing.expectEqual(@as(u64, 1) << 6 | @as(u64, 4) << 7, modeWord());
    kept_host = true; // XPU_KEPT_HOST on one rank only: the words differ
    try std.testing.expectEqual(@as(u64, 1) << 6 | @as(u64, 4) << 7 | @as(u64, 1) << 13, modeWord());
    kept_host = false;
    park_on = true; // XPU_PARK_GIB on one rank only: the words differ
    try std.testing.expectEqual(@as(u64, 1) << 6 | @as(u64, 4) << 7 | @as(u64, 1) << 23, modeWord());
    park_on = false;
    kept_host = false;
    fmt_mlx = true; // the format bit: a pair of an EXL3 rank and an MLX rank differs
    try std.testing.expectEqual(@as(u64, 1) << 6 | @as(u64, 4) << 7 | @as(u64, 1) << 12, modeWord());
    fmt_mlx = false;
    k1mod.multi_tree = true;
    try std.testing.expectEqual(@as(u64, 1) << 6 | @as(u64, 4) << 7 | @as(u64, 1) << 11, modeWord());
    kv_pool = true;
    keep_shared = true;
    kv_i8 = true;
    embed_host = true;
    k1mod.on = true;
    k1mod.tree = true;
    try std.testing.expectEqual(@as(u64, 0x7F) | (@as(u64, 4) << 7) | (@as(u64, 1) << 11), modeWord());
    k1mod.wave = 5;
    try std.testing.expectEqual(@as(u64, 0x7F) | (@as(u64, 5) << 7) | (@as(u64, 1) << 11), modeWord());
    k1mod.multi_tree = false;
    try std.testing.expectEqual(@as(u64, 0x7F) | (@as(u64, 5) << 7), modeWord());
}

/// The pool's pages for a TP 2 pair: rank 0 sizes it and publishes the count, a follower waits for it and takes the
/// same, so both ranks map the same page ids.
pub fn poolPagesShared(r: *Runner, slots: u32, cache: bool, extra: u64) !u32 {
    const g = if (tp.group) |*x| x else return poolPages(r, slots, pool_tokens, cache, extra);
    if (g.rank == 0) {
        const pages = try poolPages(r, slots, pool_tokens, cache, extra);
        g.setPool(pages);
        return pages;
    }
    return g.awaitPool(600) orelse error.PoolNotPublished;
}

/// The pool for a runner opened on its own (qwen27_main): sized from the card, `slots` streams' GDN states counted.
pub fn openPool(r: *Runner, slots: u32, drafter_ctx: u64) !void {
    const pages = try poolPagesShared(r, slots, true, drafter_ctx);
    if (pages == 0) return error.PoolDoesNotFit;
    try r.allocPool(pages);
}

pub const drafter = @import("drafter.zig");
/// The Sage-A1.5 (qwen3_5_moe) runner on the same device layer and kernels.
pub const sage = @import("sage.zig");
pub const vision = @import("vision.zig");

test "every open starts from the process's options, and a second runner is refused while one is open" {
    const before = options();
    defer {
        restore(before);
        defaults = null;
        live = false;
    }
    defaults = before;
    prefill_int8 = true; // what an earlier open's XPU_PREFILL_INT8 left
    CHUNK = 64;
    kv_i8 = true;
    restore(defaults.?);
    try std.testing.expect(!prefill_int8 and !kv_i8);
    try std.testing.expectEqual(before.chunk, CHUNK);
    live = true;
    try std.testing.expectError(error.RunnerOpen, Runner.init("/nonexistent", 4096));
}

test {
    _ = @import("runner_drain_test.zig");
    _ = @import("drafter.zig"); // test-draft-win
}

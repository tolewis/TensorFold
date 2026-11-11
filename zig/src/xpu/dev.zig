//! Low-level XPU device plumbing for the Qwen3.8-27B runner: safetensors shards, the Level Zero device, profiling.
const std = @import("std");
const ze = @import("ze");
const safetensors = @import("safetensors.zig");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const a8 = fwd.a8;
const alloc = q.alloc;

// ---------------------------------------------------------------- safetensors (mmap through libc)

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;

pub const Tensor = safetensors.Tensor;

pub const Checkpoint = struct {
    tensors: std.ArrayList(Tensor) = .empty,
    maps: std.ArrayList([]const u8) = .empty, // the shards, mapped as long as the tensors' bytes are read
    headers: std.ArrayList(std.json.Parsed(std.json.Value)) = .empty, // the tensors' names live in them
    /// Optional (buildIndex): tensor index by name without its model prefix. With it, get and has look names up
    /// instead of scanning, and `put` tensors (converted at load) take the place of the stored ones.
    index: std.StringHashMapUnmanaged(usize) = .empty,
    owned: std.ArrayList([]u8) = .empty, // the put tensors' bytes and names
    stored: usize = 0, // tensors from the shards; the put ones follow

    /// A name without the prefixes of a multimodal or MLX export: `model.language_model.layers.0.x` is `layers.0.x`.
    pub fn keyOf(n: []const u8) []const u8 {
        var s = n;
        for ([_][]const u8{ "model.language_model.", "language_model.model.", "language_model.", "model." }) |p| {
            if (std.mem.startsWith(u8, s, p)) s = s[p.len..];
        }
        return s;
    }

    pub fn buildIndex(c: *Checkpoint) !void {
        c.stored = c.tensors.items.len;
        for (c.tensors.items, 0..) |t, i| try c.index.put(alloc, keyOf(t.name), i);
    }

    /// A tensor converted at load (`bytes` from `alloc`, now owned here): later gets of `name` return it.
    pub fn put(c: *Checkpoint, name: []const u8, dtype: []const u8, shape: []const u64, bytes: []u8) !void {
        try c.owned.ensureUnusedCapacity(alloc, 2);
        const n = try alloc.dupe(u8, name);
        c.owned.appendAssumeCapacity(n);
        c.owned.appendAssumeCapacity(bytes);
        var t: Tensor = .{ .name = n, .dtype = dtype, .shape = @splat(1), .rank = shape.len, .bytes = bytes };
        @memcpy(t.shape[0..shape.len], shape);
        try c.tensors.append(alloc, t);
        try c.index.put(alloc, keyOf(n), c.tensors.items.len - 1);
    }

    /// Frees the put tensors (after the load copied them to the device); stored ones keep their index entries.
    pub fn dropPut(c: *Checkpoint) void {
        for (c.owned.items) |b| alloc.free(b);
        c.owned.clearAndFree(alloc);
        c.tensors.shrinkRetainingCapacity(c.stored);
        c.index.clearAndFree(alloc);
        for (c.tensors.items, 0..) |t, i| c.index.put(alloc, keyOf(t.name), i) catch {
            c.index.clearAndFree(alloc);
            return;
        };
    }

    pub fn addShard(c: *Checkpoint, path: [:0]const u8) !void {
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        defer _ = close(fd);
        const end = lseek(fd, 0, 2);
        if (end < 8) return error.BadSafetensors;
        const size: usize = @intCast(end);
        const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed; // PROT_READ, MAP_PRIVATE
        if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed; // MAP_FAILED
        const bytes = @as([*]const u8, @ptrCast(base))[0..size];
        errdefer _ = munmap(base, size);
        try c.maps.ensureUnusedCapacity(alloc, 1);
        try c.headers.ensureUnusedCapacity(alloc, 1);
        c.headers.appendAssumeCapacity(try safetensors.parse(alloc, bytes, &c.tensors));
        c.maps.appendAssumeCapacity(bytes);
    }

    pub fn deinit(c: *Checkpoint) void {
        for (c.owned.items) |b| alloc.free(b);
        c.owned.deinit(alloc);
        c.index.deinit(alloc);
        for (c.headers.items) |h| h.deinit();
        for (c.maps.items) |m| _ = munmap(@ptrCast(@constCast(m.ptr)), m.len);
        c.tensors.deinit(alloc);
        c.maps.deinit(alloc);
        c.headers.deinit(alloc);
        c.* = .{};
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        if (c.index.count() > 0) if (c.index.get(keyOf(name))) |i| return c.tensors.items[i];
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name) and
            (t.name.len == name.len or std.mem.startsWith(u8, t.name, "language_model.") or std.mem.startsWith(u8, t.name, "model.language_model."))) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
    pub fn has(c: *const Checkpoint, name: []const u8) bool {
        if (c.index.count() > 0) return c.index.contains(keyOf(name));
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name)) return true;
        return false;
    }
};

// ---------------------------------------------------------------- device

/// A list that will not drain may still read or write host memory its callers free: the process ends (tests swap it).
pub var lost: *const fn (rc: u32) void = exitLost;

/// The limit of `settle` in ns (maxInt: none). The device-flag event mode sets it (TF_TP_DEVFLAG=2), so a launch
/// that waits for a peer's word that never comes ends the process instead of holding the card.
pub var settle_ns: u64 = std.math.maxInt(u64);

fn exitLost(rc: u32) void {
    std.log.err("XPU: a command list did not drain (Level Zero 0x{x}): its copies may still use host memory, so the process ends", .{rc});
    std.process.exit(1);
}

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{}, // USM host memory: the driver cannot copy from file-backed mmap pages directly
    allocated: usize = 0, // device bytes allocated (frees not counted: buffers live as long as the runner)
    k: struct {
        embed: ze.H,
        add_rmsnorm_dec: ze.H,
        add_rmsnorm: ze.H,
        gdn_pre: ze.H,
        gdn_pre_rows: ze.H,
        gated_norm: ze.H,
        attn_prep: ze.H,
        attn_gate: ze.H,
        swiglu: ze.H,
        argmax: ze.H,
        argmax_part: ze.H,
        argmax_part_masked: ze.H,
        tap: ze.H,
        conv_commit: ze.H,
        repack: ze.H,
        gdn_step: ze.H,
        attn_decode: ze.H,
        attn_merge: ze.H,
        attn_k1: ze.H = null, // XPU_K1: the split-K DPAS decode over int8 pool pages (k1.cl)
        attn_k1_16: ze.H = null, // XPU_K1_R16: the 16-row single-group variant (k1.cl)
        attn_k1_draft: ze.H = null, // XPU_K1_TREE: the tree verify's draft keys into slice NS (k1.cl)
        attn_k1_draft_mt: ze.H = null, // XPU_MULTI_TREE: the shared round's tree rows' draft keys (k1.cl)
        attn_span: ze.H = null, // the scalar page-table slice kernel (decode.cl), K1's reference
        merge_span: ze.H = null, // the merge of slice partials (decode.cl)
        kv_move: ze.H,
        exl3_gemv: ze.H,
        rot_heads: ze.H,
        tp_add: ze.H = null, // the TP 2 all-reduce add (the 27B loads it, Sage has none)
        tp_add_flag: ze.H = null, // the TP 2 device-flag add (TF_TP_DEVFLAG, the 27B only)
        tp_norm_tp: ze.H = null, // the exchange fused into the residual add and norm (TF_TP_DEVFLAG)
        tp_norm_peer: ze.H = null, // the device-direct exchange joined into the norm that follows (XPU_FUSE bit 1)
        tp_consume: ze.H = null, // the consumed counter and the ring's backpressure (TF_TP_DEVFLAG)
        tp_probe: ze.H = null, // the flag lab spin probe (TF_TP_FLAGLAB)
        attn_prompt: ze.H,
        attn_prompt8: ze.H,
        embed_dense: ze.H,
        dense_rows: ze.H,
        repack_gptq: ze.H,
        quantize_mlx: ze.H,
        dec_wide_g128: ze.H,
        dec_wide64_g128: ze.H,
        dec_narrow_g128: ze.H,
        ngen_wide_g128: ze.H,
        ngen_narrow_g128: ze.H,
        ngen_swiglu: ze.H,
        ngen_swiglu_g128: ze.H,
        rowsums_ff: ze.H,
        fakequant_rows: ze.H,
        a8: ze.H,
        a8_swiglu: ze.H,
        quant_a8: ze.H,
        a4: ze.H,
        quant_a4: ze.H,
        dec_wide: ze.H,
        dec_wide64: ze.H,
        dec_narrow: ze.H,
        gdn_prep: ze.H,
        gdn_state: ze.H,
        gdn_out: ze.H,
        ngen_wide: ze.H,
        ngen_narrow: ze.H,
    },

    /// Waits for the queue, then gives the card back everything made on it (Gpu.close).
    pub fn deinit(d: *Dev) void {
        d.settle(); // the runner's host tables and staging go next: they must be idle
        d.g.close();
    }

    /// Waits for every command on the list; callers settle before host memory a queued copy reads or writes goes.
    pub fn settle(d: *Dev) void {
        const t0: f64 = if (q.stat.on) q.now() else 0; // XPU_ROUND_STATS: the host wait time of this call
        const rc = d.g.api.zeCommandListHostSynchronize(d.list, settle_ns);
        if (rc != 0) lost(rc);
        if (q.stat.on) {
            q.stat.settles += 1;
            q.stat.settle_s += q.now() - t0;
        }
    }

    pub fn kernel(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        d.allocated += @max(bytes, 64);
        return d.g.deviceAlloc(@max(bytes, 64));
    }

    /// Pinned host USM (`zeMemAllocHost`): the kernels read it over PCIe (the embedding table at XPU_EMBED_HOST=1),
    /// so it takes no device memory. The driver maps it into the GPU's address space and reads resolve over PCIe.
    pub fn hostAlloc(d: *Dev, bytes: usize) !*anyopaque {
        return try d.g.hostAlloc(bytes);
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    /// Host bytes (any memory, mmap included) to the device, through the USM staging buffer.
    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(d.staging.len, bytes.len - done);
            @memcpy(d.staging[0..n], bytes[done .. done + n]);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(dst, done), d.staging.ptr, n, null, 0, null));
            try d.sync();
            done += n;
        }
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        if (q.stat.on) q.stat.launches += 1;
        var ev: ze.H = null;
        if (q.prof.on) {
            if (q.prof.used == q.prof.events.len) try d.sync();
            ev = q.prof.events[q.prof.used];
            q.prof.kernel[q.prof.used] = k;
            q.prof.at[q.prof.used] = q.prof.site;
            q.prof.used += 1;
        }
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, ev, 0, null));
    }

    /// `launch` with one wait event: the launch starts only when the event is complete (TF_TP_DEVFLAG=2, the
    /// counter based event on the peer's flag word). The signal event of a profile run is kept as `launch` keeps it.
    pub fn launchEv(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype, ev: ze.H) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        if (q.stat.on) q.stat.launches += 1;
        var sig: ze.H = null;
        if (q.prof.on) {
            if (q.prof.used == q.prof.events.len) try d.sync();
            sig = q.prof.events[q.prof.used];
            q.prof.kernel[q.prof.used] = k;
            q.prof.used += 1;
        }
        var wa = [1]ze.H{ev};
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, sig, 1, &wa));
    }

    pub fn sync(d: *Dev) !void {
        d.settle();
        if (q.stat.on) q.stat.syncs += 1;
        if (q.prof.on) {
            for (q.prof.events[0..q.prof.used], q.prof.kernel[0..q.prof.used], q.prof.at[0..q.prof.used]) |e, k, site| {
                // a copy (no kernel): its event may hold no kernel timestamp, which counts as 0
                const t = if (k == null) d.g.seconds(e) catch 0 else try d.g.seconds(e);
                try ze.check(d.g.api.zeEventHostReset(e));
                if (site != .none) {
                    q.prof.site_total[@intFromEnum(site)] += t;
                    q.prof.site_count[@intFromEnum(site)] += 1;
                }
                if (k != null) for (&q.prof.names, &q.prof.total, &q.prof.count) |n, *tot, *cnt| {
                    if (n.k == k) {
                        tot.* += t;
                        cnt.* += 1; // one launch of this kernel in the synced window
                        break;
                    }
                };
            }
            q.prof.used = 0;
        }
    }

    pub fn fill(d: *Dev, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        const host = try alloc.alloc(T, bytes / @sizeOf(T));
        defer alloc.free(host);
        @memset(host, pattern);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
    }
};

pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    if (q.stat.on) q.stat.copies += 1;
    var ev: ze.H = null;
    if (q.prof.on and q.prof.sites_on) { // XPU_PROFILE=2: the copy gets an event and a site row (no kernel row)
        if (q.prof.used == q.prof.events.len) try d.sync();
        ev = q.prof.events[q.prof.used];
        q.prof.kernel[q.prof.used] = null;
        q.prof.at[q.prof.used] = if (q.prof.site == .copy_stash) .copy_stash else .copy_other;
        q.prof.used += 1;
    }
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, ev, 0, null));
}

/// A queued copy with plain host memory on one side (a table to the card, a result to the host). XPU_ROUND_STATS
/// counts the calls and the host time inside them (lane p4-k-gdn): a call that waits for the card shows there.
pub fn hostCopy(d: *Dev, dst: ?*anyopaque, src: ?*const anyopaque, n: usize) !void {
    const t0: f64 = if (q.stat.on) q.now() else 0;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, null, 0, null));
    if (q.stat.on) {
        q.stat.hcopies += 1;
        q.stat.hcopy_s += q.now() - t0;
    }
}

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// A call site of the decode pass (XPU_PROFILE=2, lane p4-k-gdn): the matvec shapes, the two uses of gdn_step, the
/// norms and the device copies each get a row of their own under the kernel rows.
pub const Site = enum(u8) { none, norm_in, mixer_in_gdn, mixer_in_attn, mixer_out_gdn, mixer_out_attn, norm_post, gate_up, down, head, gdn_verify, gdn_commit, copy_stash, copy_other, gate_up_attn };

/// XPU_PROFILE=1: GPU time per kernel, summed over the run. XPU_PROFILE=2: also one row a call site (`Site`) and an
/// event on every device copy that goes through `copy`.
pub const Prof = struct {
    on: bool = false,
    sites_on: bool = false, // XPU_PROFILE=2
    site: Site = .none, // the call site of the launches that follow (forward sets and clears it)
    events: [4096]ze.H = @splat(null),
    kernel: [4096]ze.H = @splat(null),
    at: [4096]Site = @splat(.none),
    site_total: [16]f64 = @splat(0),
    site_count: [16]u64 = @splat(0),
    used: usize = 0,
    names: [192]struct { k: ze.H = null, name: []const u8 = "" } = @splat(.{}),
    total: [192]f64 = @splat(0),
    count: [192]u64 = @splat(0), // launches a kernel, reset at each report
};

pub fn profReport(label: []const u8) void {
    if (!q.prof.on) return;
    var sum: f64 = 0;
    for (q.prof.total) |t| sum += t;
    std.debug.print("GPU time, {s}: {d:.1} ms\n", .{ label, sum * 1e3 });
    for (q.prof.names, &q.prof.total, &q.prof.count) |n, *t, *c| {
        if (t.* > 0) std.debug.print("  {s:<14} {d:8.2} ms {d:5.1} % {d:>8} launches\n", .{ n.name, t.* * 1e3, t.* / sum * 100, c.* });
        t.* = 0;
        c.* = 0;
    }
    if (q.prof.sites_on) for (std.enums.values(Site)) |site| {
        const i = @intFromEnum(site);
        if (q.prof.site_count[i] > 0) std.debug.print("  @{s:<13} {d:8.2} ms {d:5.1} % {d:>8} launches\n", .{ @tagName(site), q.prof.site_total[i] * 1e3, q.prof.site_total[i] / sum * 100, q.prof.site_count[i] });
        q.prof.site_total[i] = 0;
        q.prof.site_count[i] = 0;
    };
}

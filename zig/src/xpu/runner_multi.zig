//! Shared verify rounds across slots: one forward over several windows, then each window keeps its accepted rows.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const tp = @import("tp.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const k1mod = @import("k1.zig"); // XPU_MULTI_TREE: the shared tree round's host fill and the merge's per-row used
const GDN_IN = q.GDN_IN;
const HK = q.GDN_K_HEADS;
const HV = q.GDN_V_HEADS;
const LAYERS = q.LAYERS;
const Runner = q.Runner;
const ST_B = q.ST_B;
const ST_G = q.ST_G;
const ST_K = q.ST_K;
const ST_P = q.ST_P;
const ST_Q = q.ST_Q;
const ST_V = q.ST_V;
const forward = fwd.forward;
const kvCap = fwd.kvCap;
const off = dev.off;

/// A window of a shared verify: `ids` (pending token, then drafts) from its position; `parents` for a tree, else chain.
pub const Win = struct { slot: u32, ids: []const u32, parents: ?[]const i32 = null };

/// XPU_MULTI_TREE host inputs (lane p33-multitree): a chain row (a window without parents) takes its mpos as its
/// last prefix key, kkind 0; a tree row takes its window base - 1, kkind 1: the window region is owned by the
/// draft kernel. The window base is the window first row's mpos. When `sent`, a chain row's tree-table row
/// carries the NO_DRAFT sentinel (bit 1024) so the draft kernel skips it; the scalar path never reads bit 1024.
pub fn k1mtFill(win_host: []const i32, pos_host: []const i32, tree_host: []i32, wins: []const Win, kpos: []i32, kkind: []u8, sent: bool) void {
    for (wins, 0..) |w, i| {
        const first: i32 = @intCast(win_host[2 * i]);
        const width: i32 = @intCast(win_host[2 * i + 1]);
        const base: i32 = pos_host[@as(usize, @intCast(first))]; // the window first row's mpos is the window base
        for (0..@as(usize, @intCast(width))) |j| {
            const t: usize = @as(usize, @intCast(first)) + @as(usize, @intCast(j));
            if (w.parents != null) {
                kkind[t] = 1;
                kpos[t] = base - 1;
            } else {
                kkind[t] = 0;
                kpos[t] = pos_host[t];
                if (sent) tree_host[t * 17] |= 1024;
            }
        }
    }
}

/// Every window in one forward (16 rows at most): each row's greedy token in `block`; nothing moves until `keepMulti`.
pub fn verifyMulti(r: *Runner, wins: []const Win) !void {
    // every window's pages first (rank 0: the kept states it drops are step records before this call)
    for (wins) |w| if (w.ids.len > 0 and w.slot < q.MAX_SLOTS) {
        const p = r.seqs[w.slot].pos;
        if (@as(u64, p) + w.ids.len <= r.context) try r.makeRoom(w.slot, p + @as(u32, @intCast(w.ids.len)));
    };
    if (tp.group) |*g| { // a window: slot, rows, has parents, the ids, the parents
        var words: [16 * 35]u32 = undefined;
        var at: usize = 0;
        for (wins[0..@min(wins.len, 16)]) |w| {
            const n = @min(w.ids.len, 16);
            words[at] = w.slot;
            words[at + 1] = @intCast(n);
            words[at + 2] = @intFromBool(w.parents != null);
            @memcpy(words[at + 3 .. at + 3 + n], w.ids[0..n]);
            at += 3 + n;
            if (w.parents) |ps| {
                for (ps[0..@min(ps.len, n)], 0..) |p, i| words[at + i] = @bitCast(p);
                at += n;
            }
        }
        try g.record(.verify_multi, @intCast(@min(wins.len, 16)), 0, 0, words[0..at]);
    }
    if (wins.len == 0 or wins.len > 16) return error.BadBlock;
    r.store();
    var T: usize = 0;
    var max_pos: u32 = 0;
    for (wins, 0..) |w, i| {
        for (wins[0..i]) |o| if (o.slot == w.slot) return error.SlotTwice;
        if (w.ids.len == 0 or T + w.ids.len > 16) return error.BadBlock;
        const p = r.seqs[w.slot].pos;
        if (p + w.ids.len > r.context) return error.ContextFull;
        r.win_host[2 * i] = @intCast(T);
        r.win_host[2 * i + 1] = @intCast(w.ids.len);
        for (w.ids, 0..) |id, j| {
            r.pos_host[T + j] = @intCast(p + j);
            r.ids_host[T + j] = id;
        }
        max_pos = @max(max_pos, p + @as(u32, @intCast(w.ids.len)) - 1);
        try r.ensurePages(w.slot, p + @as(u32, @intCast(w.ids.len)));
        T += w.ids.len;
    }
    // tree windows: one tree table for the block, each row at its window's position plus its depth
    var tree = false;
    for (wins) |w| if (w.parents) |pa| {
        if (pa.len != w.ids.len) return error.BadBlock;
        for (pa, 0..) |x, j| if (x != @as(i32, @intCast(j)) - 1) {
            tree = true;
        };
    };
    if (tree) {
        max_pos = 0;
        for (wins, 0..) |w, i| {
            const first: usize = @intCast(r.win_host[2 * i]);
            const p = r.seqs[w.slot].pos;
            for (0..w.ids.len) |j| {
                const t = first + j;
                const row = r.tree_host[t * 17 ..][0..17];
                const pj: i32 = if (w.parents) |pa| pa[j] else @as(i32, @intCast(j)) - 1;
                if (j == 0) {
                    if (pj != -1) return error.BadBlock;
                    row[0] = 0;
                    row[1] = @intCast(t);
                } else {
                    if (pj < 0 or pj >= j) return error.BadBlock;
                    const pr = r.tree_host[(first + @as(usize, @intCast(pj))) * 17 ..][0..17];
                    const dp: usize = @intCast(pr[0] & 255);
                    row[0] = @intCast(dp + 1);
                    @memcpy(row[1 .. dp + 2], pr[1 .. dp + 2]);
                    row[dp + 2] = @intCast(t);
                    if (pj + 1 != j) { // LOAD: the parent is not the row before; SAVE: a child is not the row after
                        row[0] |= 512;
                        pr[0] |= 256;
                    }
                }
                r.pos_host[t] = @intCast(p + @as(u32, @intCast(row[0] & 255)));
                max_pos = @max(max_pos, p + @as(u32, @intCast(row[0] & 255)));
            }
        }
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.acts.tree, &r.tree_host, T * 17 * 4, null, 0, null));
    }
    // XPU_MULTI_TREE host inputs, after the tree table (k1mtFill): chain rows their mpos, tree rows the window
    // base - 1; the NO_DRAFT sentinel on the chain rows' table rows when the draft kernel will read the table
    k1mtFill(r.win_host[0 .. 2 * wins.len], &r.pos_host, &r.tree_host, wins, &r.kpos_host, &r.kkind_host, k1mod.multi_tree and tree);
    for (&r.model.layers, 0..) |*l, i| for (wins, 0..) |w, wi| {
        const first: usize = @intCast(r.win_host[2 * wi]);
        for (0..w.ids.len) |j| {
            const e = r.mt_host[(i * 16 + first + j) * 8 ..][0..8];
            e.* = @splat(0);
            e[6] = first;
            e[7] = w.ids.len;
            if (l.attn) {
                e[0] = @intFromPtr(l.slot_kc[w.slot].?);
                e[1] = @intFromPtr(l.slot_vc[w.slot].?);
                e[2] = if (r.pt[w.slot]) |p| @intFromPtr(p) else 0; // the slot's page table (0: no pool); K1 reads it
            } else {
                const cur = l.slot_cur[w.slot];
                e[2] = @intFromPtr(l.slot_conv[w.slot][cur].?);
                e[3] = @intFromPtr(l.slot_conv[w.slot][1 - cur].?);
                e[4] = @intFromPtr(l.slot_rec[w.slot].?);
                e[5] = @intFromPtr(l.slot_save[w.slot].?);
            }
        }
    };
    const d = &r.d;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mt, &r.mt_host, LAYERS * 16 * 8 * 8, null, 0, null));
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mpos, &r.pos_host, T * 4, null, 0, null));
    _ = try forward(d, &r.model, &r.acts, @ptrCast(r.ids_host[0..T]), 0, .{ .want = .rows, .stash = true, .taps = r.taps_on, .tree = tree, .multi = .{ .mt = r.acts.mt, .pos = r.acts.mpos, .max_pos = max_pos, .win = r.win_host[0 .. 2 * wins.len], .kpos = r.kpos_host[0..16], .kkind = r.kkind_host[0..16] } });
    var toks: [16]i32 = undefined;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, r.acts.tok, T * 4, null, 0, null));
    try d.sync();
    for (r.block[0..T], toks[0..T]) |*o, t| o.* = @intCast(t);
    for (wins, 0..) |w, i| r.multi_slots[i] = w.slot;
    r.multi_count = @intCast(wins.len);
    r.multi_tree = tree;
}

/// Keeps the first `keep[i]` rows of window i of the last verifyMulti: whole windows keep their state, others replay.
pub fn keepMulti(r: *Runner, keep: []const u32) !void {
    if (tp.group) |*g| try g.record(.keep_multi, 0, 0, 0, keep);
    var rows: [16]u32 = undefined;
    for (&rows, 0..) |*x, i| x.* = @intCast(i);
    var paths: [16][]const u32 = undefined;
    if (keep.len > 16) return error.BadCommit;
    for (keep, 0..) |k, i| paths[i] = rows[0..@min(k, 16)];
    return r.keepPaths(paths[0..keep.len]);
}

/// Keeps window i's root path `paths[i]`: a whole chain window keeps its state, otherwise GDN states replay, K/V move.
pub fn keepPaths(r: *Runner, paths: []const []const u32) !void {
    if (tp.group) |*g| { // a path: its length, then its rows
        var words: [16 * 17]u32 = undefined;
        var at: usize = 0;
        for (paths[0..@min(paths.len, 16)]) |p| {
            const n = @min(p.len, 16);
            words[at] = @intCast(n);
            @memcpy(words[at + 1 .. at + 1 + n], p[0..n]);
            at += 1 + n;
        }
        try g.record(.keep_paths, @intCast(@min(paths.len, 16)), 0, 0, words[0..at]);
    }
    if (paths.len != r.multi_count) return error.BadCommit;
    const d = &r.d;
    const nul: ?*anyopaque = null;
    for (paths, 0..) |path, i| {
        const first: u32 = @intCast(r.win_host[2 * i]);
        const width: u32 = @intCast(r.win_host[2 * i + 1]);
        if (path.len == 0 or path.len > width or path[0] != 0) return error.BadCommit;
        for (path, 0..) |row, dd| {
            if (row >= width or (dd > 0 and row <= path[dd - 1])) return error.BadCommit;
            r.mpath_host[(i * 2) * 16 + dd] = @intCast(first + row);
            r.mpath_host[(i * 2 + 1) * 16 + dd] = @intCast(row);
        }
    }
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mpath, &r.mpath_host, paths.len * 2 * 16 * 4, null, 0, null));
    for (paths, 0..) |path, i| {
        const sl = r.multi_slots[i];
        const first: u32 = @intCast(r.win_host[2 * i]);
        const width: u32 = @intCast(r.win_host[2 * i + 1]);
        const k: u32 = @intCast(path.len);
        var straight = true;
        for (path, 0..) |row, dd| straight = straight and row == dd;
        const brows = off(r.acts.mpath, (i * 2) * 16 * 4); // the path in rows of the block
        const wrows = off(r.acts.mpath, (i * 2 + 1) * 16 * 4); // and of the window
        for (&r.model.layers) |*l| {
            if (l.attn) {
                if (!straight) try d.launch(d.k.kv_move, .{ .x = q.ATT_KV_HEADS }, .{ l.slot_kc[sl].?, l.slot_vc[sl].?, @as(i32, @intCast(r.seqs[sl].pos)), wrows, @as(i32, @intCast(k)), kvCap(), r.pt[sl] });
                continue;
            }
            const cur = l.slot_cur[sl];
            if (k == width and straight and !r.multi_tree) {
                std.mem.swap(?*anyopaque, &l.slot_rec[sl], &l.slot_save[sl]);
            } else if (straight and !r.multi_tree) {
                const st = l.stash.?;
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = HV }, .{ off(st, ST_Q + first * HK * 128 * 2), off(st, ST_K + first * HK * 128 * 2), off(st, ST_V + first * HV * 128 * 2), off(st, ST_G + first * HV * 4), off(st, ST_B + first * HV * 4), l.slot_rec[sl].?, l.slot_rec[sl].?, r.acts.y, @as(i32, @intCast(k)), nul, nul, nul, nul });
                try d.launch(d.k.conv_commit, .{ .x = q.GDN_QKV / 256 }, .{ l.slot_conv[sl][cur].?, off(st, ST_P + first * GDN_IN * 2), l.slot_conv[sl][1 - cur].?, @as(i32, @intCast(k)), nul });
            } else {
                const st = l.stash.?;
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = HV }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.slot_rec[sl].?, l.slot_rec[sl].?, r.acts.y, @as(i32, @intCast(k)), brows, nul, nul, nul });
                try d.launch(d.k.conv_commit, .{ .x = q.GDN_QKV / 256 }, .{ l.slot_conv[sl][cur].?, off(st, ST_P), l.slot_conv[sl][1 - cur].?, @as(i32, @intCast(k)), brows });
            }
            l.slot_cur[sl] = 1 - cur;
        }
        r.seqs[sl].pos += k;
        r.seqs[sl].last = r.block[first + path[k - 1]];
    }
    r.multi_count = 0;
    r.multi_tree = false;
    // the selected slot's views follow its arrays
    for (&r.model.layers) |*l| if (!l.attn) {
        l.rec = l.slot_rec[r.slot];
        l.rec_save = l.slot_save[r.slot];
        l.conv_cur = l.slot_cur[r.slot];
    };
    r.pos = r.seqs[r.slot].pos;
    r.last = r.seqs[r.slot].last; // mpath_host stays until the next verifyMulti, which waits for the list
}

/// One token for each of several slots: a verifyMulti of one-row windows, all kept.
pub fn stepMulti(r: *Runner, slots: []const u32, tokens: []const u32) !void {
    var wins: [16]Win = undefined;
    if (tp.group) |*g| { // the slots, then the tokens
        var words: [32]u32 = undefined;
        const n: usize = @min(slots.len, 16); // usize: see verifyTree
        @memcpy(words[0..n], slots[0..n]);
        @memcpy(words[n .. 2 * n], tokens[0..@min(tokens.len, n)]);
        try g.record(.step_multi, @intCast(n), 0, 0, words[0 .. 2 * n]);
    }
    if (slots.len > 16 or tokens.len != slots.len) return error.BadBlock;
    for (slots, tokens, 0..) |sl, *t, i| wins[i] = .{ .slot = sl, .ids = t[0..1] };
    try r.verifyMulti(wins[0..slots.len]);
    const ones: [16]u32 = @splat(1);
    try r.keepMulti(ones[0..slots.len]);
}

test "k1 mt: the host fill keeps chain rows on their mpos and cuts tree rows at the window base" {
    // two windows: a 4-row tree window at base 40000 (rows 0..3), a 2-row chain window at 41000 (rows 4..5)
    const parents: [4]i32 = .{ -1, 0, 1, 1 };
    const ids: [4]u32 = .{ 1, 2, 3, 4 };
    const cids: [2]u32 = .{ 5, 6 };
    const wins = [_]Win{.{ .slot = 1, .ids = &ids, .parents = &parents }, .{ .slot = 2, .ids = &cids }};
    const win_host = [32]i32{ 0, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var pos_host: [16]i32 = @splat(0);
    for (0..4) |j| pos_host[j] = 40000 + @as(i32, @intCast(j));
    for (4..6) |j| pos_host[j] = 41000 + @as(i32, j - 4);
    var tree_host: [16 * 17]i32 = @splat(0);
    var kpos: [16]i32 = @splat(99);
    var kkind: [16]u8 = @splat(99);
    k1mtFill(&win_host, &pos_host, &tree_host, &wins, &kpos, &kkind, true);
    for (0..4) |j| {
        try std.testing.expectEqual(@as(u8, 1), kkind[j]);
        try std.testing.expectEqual(@as(i32, 39999), kpos[j]);
        try std.testing.expectEqual(@as(i32, 0), tree_host[j * 17]);
    }
    for (4..6) |j| {
        try std.testing.expectEqual(@as(u8, 0), kkind[j]);
        try std.testing.expectEqual(pos_host[j], kpos[j]);
        try std.testing.expectEqual(@as(i32, 1024), tree_host[j * 17]); // the NO_DRAFT sentinel
    }
    // without sent, the table rows stay as the tree table left them
    var kpos2: [16]i32 = @splat(99);
    var kkind2: [16]u8 = @splat(99);
    k1mtFill(&win_host, &pos_host, &tree_host, &wins, &kpos2, &kkind2, false);
    try std.testing.expectEqual(@as(i32, 1024), tree_host[4 * 17]); // already set; the fill never clears it
}

test "k1 mt: the merge per-row used reads all slices for a tree row and the chain rows up to their mpos" {
    const kkind: [16]u8 = .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const kpos: [16]i32 = .{ 39999, 41000, 41001, 40960, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var out: [16]i32 = undefined;
    k1mod.mergePos(&kkind, &kpos, 3, 128, &out);
    try std.testing.expectEqual(@as(i32, 3 * 128 + 127), out[0]); // tree row: every slice incl. the draft
    try std.testing.expectEqual(@as(i32, 41000), out[1]); // chain row: up to its mpos, the draft slice skipped
    try std.testing.expectEqual(@as(i32, 41001), out[2]);
    try std.testing.expectEqual(@as(i32, 40960), out[3]); // exactly at a span edge: 40960 / 128 + 1 slices
    try std.testing.expectEqual(@as(i32, 0), out[4]);
}

/// One window split of a round for the cover test: the widths of up to 6 windows (16 rows at most) and which are trees.
fn coverCheck(widths: []const u32, tree_mask: u32, flags_on: bool) !void {
    var win: [32]i32 = undefined;
    var host: [32]i32 = undefined;
    var first: u32 = 0;
    var any_tree = false;
    for (widths, 0..) |w, i| {
        win[2 * i] = @intCast(first);
        win[2 * i + 1] = @intCast(w);
        first += w;
        any_tree = any_tree or (tree_mask >> @intCast(i)) & 1 != 0;
    }
    const t = first;
    // the row groups: every row once, in order, none longer than the kernel's group, at most the 16 the launch holds
    const rd = k1mod.roundGroups(win[0 .. 2 * widths.len], t, &host);
    try std.testing.expect(rd.n >= widths.len and rd.n <= 16);
    var rows: u32 = 0;
    for (0..rd.n) |g| {
        try std.testing.expectEqual(rows, @as(u32, @intCast(host[2 * g])));
        try std.testing.expect(host[2 * g + 1] >= 1 and @as(u32, @intCast(host[2 * g + 1])) <= rd.rows);
        rows += @intCast(host[2 * g + 1]);
    }
    try std.testing.expectEqual(t, rows);
    // the path: a shared round (2 or more windows, or one window in the shared forward) and a lone stream
    for ([_]bool{ true, false }) |shared| {
        if (!shared and widths.len != 1) continue;
        const c: k1mod.Cover = .{ .on = flags_on, .tree = flags_on, .multi_tree = flags_on, .i8 = true, .pool = true, .round_tree = any_tree, .shared = shared, .has_k1 = true, .has_draft = true, .has_draft_mt = true };
        const want: k1mod.Path = if (!flags_on) .scalar else if (!any_tree) .chain else if (shared) .multi_tree else .tree;
        try std.testing.expectEqual(want, k1mod.attnPath(c));
    }
}

test "k1 cover: streams 1 to 6 and every window split of 16 rows take a K1 path" {
    const saved16 = k1mod.have16;
    const savedr = k1mod.r16;
    defer {
        k1mod.have16 = saved16;
        k1mod.r16 = savedr;
    }
    var checked: u32 = 0;
    for ([_]bool{ true, false }) |r16on| {
        k1mod.have16 = true;
        k1mod.r16 = r16on;
        // the scheduler's even splits of 16 rows (remainder to the first streams), each window a tree or a chain
        for (1..7) |n| {
            var widths: [6]u32 = undefined;
            for (widths[0..n], 0..) |*w, i| w.* = @intCast(16 / n + @intFromBool(i < 16 % n));
            var mask: u32 = 0;
            while (mask < (@as(u32, 1) << @intCast(n))) : (mask += 1) {
                try coverCheck(widths[0..n], mask, true);
                try coverCheck(widths[0..n], mask, false);
                checked += 2;
            }
        }
        // every split: 1 to 6 windows of 1 to 16 rows, 16 rows at most, any window a tree (3 rows or more) or a chain
        var widths: [6]u32 = undefined;
        for (1..7) |n| {
            @memset(widths[0..n], 1);
            while (true) {
                var sum: u32 = 0;
                for (widths[0..n]) |w| sum += w;
                if (sum <= 16) {
                    var mask: u32 = 0;
                    for (widths[0..n], 0..) |w, i| if (w >= 3) {
                        mask |= @as(u32, 1) << @intCast(i);
                    };
                    // all trees, all chains, and the first window alone
                    try coverCheck(widths[0..n], mask, true);
                    try coverCheck(widths[0..n], 0, true);
                    try coverCheck(widths[0..n], mask & 1, true);
                    checked += 3;
                }
                var k: usize = 0; // odometer over widths 1 to 16
                while (k < n) : (k += 1) {
                    widths[k] += 1;
                    if (widths[k] <= 16) break;
                    widths[k] = 1;
                }
                if (k == n) break;
            }
        }
    }
    try std.testing.expect(checked > 1000);
    // XPU_MULTI_TREE=0 under XPU_K1_TREE=1: the shared tree windows go back to the scalar path, nothing else moves
    const off_mt: k1mod.Cover = .{ .on = true, .tree = true, .multi_tree = false, .i8 = true, .pool = true, .round_tree = true, .shared = true, .has_k1 = true, .has_draft = true, .has_draft_mt = true };
    try std.testing.expectEqual(k1mod.Path.scalar, k1mod.attnPath(off_mt));
    // a layer without a page table, or a bf16 cache, keeps the scalar path in every shape
    var nopool = off_mt;
    nopool.multi_tree = true;
    nopool.pool = false;
    try std.testing.expectEqual(k1mod.Path.scalar, k1mod.attnPath(nopool));
    nopool.pool = true;
    nopool.i8 = false;
    try std.testing.expectEqual(k1mod.Path.scalar, k1mod.attnPath(nopool));
    // the open line names the coverage
    var buf: [200]u8 = undefined;
    try std.testing.expectEqualStrings("K1 coverage: chain windows K1, lone tree K1, shared tree windows K1 (streams 1 to 6, up to 16 rows a round)", k1mod.coverageLine(&buf, true, true, true));
    try std.testing.expectEqualStrings("K1 coverage: chain windows K1, lone tree K1, shared tree windows scalar (streams 1 to 6, up to 16 rows a round)", k1mod.coverageLine(&buf, true, true, false));
}

/// Split-K partials as K1 writes them (m the slice max, l the sum of 2^(s - m), o the unnormalized sum of 2^(s - m) v;
/// scores s in log2 units) joined with the weight base `base` (attn_merge_span: e, attn_merge_span2: 2), against the
/// one-pass softmax 2^s over every key. Returns the largest absolute error over the head dimension.
fn mergeError(delta: f64, slices: usize, per: usize, base: f64) f64 {
    const dim = 4;
    var m: [8]f64 = undefined;
    var l: [8]f64 = undefined;
    var o: [8][dim]f64 = undefined;
    var ref_num: [dim]f64 = @splat(0);
    var ref_den: f64 = 0;
    var mx: f64 = -1e300;
    for (0..slices) |k| {
        for (0..per) |j| mx = @max(mx, -@as(f64, @floatFromInt(k)) * delta - 0.5 * @as(f64, @floatFromInt(j)));
    }
    for (0..slices) |k| {
        m[k] = -1e300;
        for (0..per) |j| m[k] = @max(m[k], -@as(f64, @floatFromInt(k)) * delta - 0.5 * @as(f64, @floatFromInt(j)));
        l[k] = 0;
        o[k] = @splat(0);
        for (0..per) |j| {
            const s = -@as(f64, @floatFromInt(k)) * delta - 0.5 * @as(f64, @floatFromInt(j));
            const w = std.math.pow(f64, 2.0, s - m[k]);
            l[k] += w;
            const wg = std.math.pow(f64, 2.0, s - mx);
            ref_den += wg;
            for (0..dim) |d| {
                const v = @as(f64, @floatFromInt(k + 1)) * (@as(f64, @floatFromInt(d)) - 1.5) + 0.1 * @as(f64, @floatFromInt(j)); // differs by slice
                o[k][d] += w * v;
                ref_num[d] += wg * v;
            }
        }
    }
    var gm: f64 = -1e300;
    for (0..slices) |k| gm = @max(gm, m[k]);
    var num: [dim]f64 = @splat(0);
    var den: f64 = 0;
    for (0..slices) |k| {
        const w = std.math.pow(f64, base, m[k] - gm);
        den += l[k] * w;
        for (0..dim) |d| num[d] += o[k][d] * w;
    }
    var err: f64 = 0;
    for (0..dim) |d| err = @max(err, @abs(num[d] / den - ref_num[d] / ref_den));
    return err;
}

test "k1 merge: the exp2 merge joins K1 slices exactly, the exp merge does not" {
    // slice maxima 1 to 20 log2 units apart, 3 to 8 slices (the plan makes 16 to 64; the sum is the same shape)
    var delta: f64 = 1;
    while (delta <= 20) : (delta += 1) {
        for ([_]usize{ 3, 5, 8 }) |slices| {
            try std.testing.expect(mergeError(delta, slices, 16, 2.0) < 1e-12); // attn_merge_span2
            if (delta <= 8) try std.testing.expect(mergeError(delta, slices, 16, std.math.e) > 1e-3); // attn_merge_span on K1 partials
        }
    }
}

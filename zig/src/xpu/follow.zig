//! A tensor-parallel rank above 0: replays rank 0's state-changing Runner calls from the step ring (tp.zig), so
//! its forward meets rank 0's at every all-reduce. The results of a call (greedy tokens, logits) are rank 0's to
//! read: the lm_head is whole on every rank and a follower's copy is the same bits, unread.
//!
//! An error rank 0's call returned before its forward (a bad block, a full context) returns here the same way,
//! and the follower goes on to the next record: the two stay in step. An error inside a forward is fatal on both.
const std = @import("std");
const q = @import("qwen27.zig");
const tp = @import("tp.zig");
const multi = @import("runner_multi.zig");
const slot_states = @import("slot_states.zig");
const alloc = q.alloc;

pub fn run(r: *q.Runner) !void {
    const g = if (tp.group) |*x| x else return error.NotARank;
    const trace = std.c.getenv("TF_TP_TRACE") != null; // every record to stderr
    if (g.rank == 0) return error.NotAFollower;
    var snaps: std.AutoHashMapUnmanaged(u64, slot_states.Snap) = .empty;
    defer snaps.deinit(alloc);
    var buf: std.ArrayList(u32) = .empty;
    defer buf.deinit(alloc);
    var rec: tp.Rec = undefined;
    while (true) {
        g.next(&rec);
        const op: tp.Op = @enumFromInt(rec.op);
        const total: usize = rec.n;
        buf.clearRetainingCapacity();
        try buf.appendSlice(alloc, rec.data[0..@min(total, tp.rec_data)]);
        while (buf.items.len < total) { // the rest of a long prompt
            var m: tp.Rec = undefined;
            g.next(&m);
            if (m.op != @intFromEnum(tp.Op.more)) return error.BadRecord;
            try buf.appendSlice(alloc, m.data[0..@min(total - buf.items.len, tp.rec_data)]);
        }
        const data = buf.items;
        const id = @as(u64, rec.lo) | (@as(u64, rec.hi) << 32);
        if (trace) std.debug.print("follow: {s} a {d} b {d} n {d} data[0] {d}\n", .{ @tagName(op), rec.a, rec.b, rec.n, if (data.len > 0) data[0] else 0 });
        const n: usize = rec.a;
        const done = switch (op) {
            .select => r.select(rec.a),
            .store => r.store(),
            .reset => r.reset(),
            .prompt => r.prompt(data, rec.a != 0),
            .step => r.step(rec.a),
            .verify => if (r.verify(data)) |_| {} else |e| e,
            .commit => r.commit(rec.a),
            .verify_tree => if (r.verifyTree(data[0..n], @ptrCast(data[n .. 2 * n]))) |_| {} else |e| e,
            .commit_path => r.commitPath(data),
            .verify_multi => blk: {
                var wins: [multi.MAX_WINDOWS]multi.Win = undefined;
                var at: usize = 0;
                for (wins[0..n]) |*w| {
                    const rows: usize = data[at + 1];
                    const has_parents = data[at + 2] != 0;
                    w.* = .{ .slot = data[at], .ids = data[at + 3 .. at + 3 + rows], .parents = null };
                    at += 3 + rows;
                    if (has_parents) {
                        w.parents = @ptrCast(data[at .. at + rows]);
                        at += rows;
                    }
                }
                break :blk multi.verifyMulti(r, wins[0..n]);
            },
            .keep_multi => multi.keepMulti(r, data),
            .keep_paths => blk: {
                var paths: [multi.MAX_WINDOWS][]const u32 = undefined;
                var at: usize = 0;
                for (paths[0..n]) |*p| {
                    const rows: usize = data[at];
                    p.* = data[at + 1 .. at + 1 + rows];
                    at += 1 + rows;
                }
                break :blk multi.keepPaths(r, paths[0..n]);
            },
            .step_multi => multi.stepMulti(r, data[0..n], data[n .. 2 * n]),
            .save_slot => blk: {
                const s = slot_states.saveSlot(r, rec.a, rec.b) catch |e| break :blk e;
                break :blk snaps.put(alloc, id, s);
            },
            .restore_slot => if (snaps.get(id)) |s| slot_states.restoreSlot(r, rec.a, s) else error.NoSuchSnap,
            .drop_snap => if (snaps.fetchRemove(id)) |kv| slot_states.dropSnap(r, kv.value) else {},
            .ensure => slot_states.ensure(r, rec.a),
            .mtp_absorb => if (r.mtp) |m| blk: {
                const b = q.mtp.KeptBuf.from(data);
                break :blk m.absorbKept(b.list());
            } else {},
            .mtp_draft => if (r.mtp) |m| blk: {
                const b = q.mtp.AskBuf.from(data);
                break :blk m.draft(b.list());
            } else {},
            .close => return,
            .more => error.BadRecord,
        };
        done catch |err| switch (err) {
            error.BadBlock, error.BadCommit, error.ContextFull, error.NoSuchSlot, error.SlotNotThere => {
                std.debug.print("follow: rank 0's {s} returned {s} here too\n", .{ @tagName(op), @errorName(err) });
            },
            else => return err,
        };
    }
}

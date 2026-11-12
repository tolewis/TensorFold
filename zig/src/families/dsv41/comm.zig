//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }

    /// TF_DS_KV_SHARD (GLM 0118's send_rows / recv_rows): device byte spans one way, `spans` to rank `peer` in one NCCL
    /// group (the peer calls recvSpans with spans of the same sizes in the same order; each send meets its receive).
    /// A plane's rows go straight from the sender's plane into the receiver's: no staging.
    pub fn sendSpans(c: Comm, spans: []const Span, peer: u32, stream: cuda.Stream) !void {
        try c.p2p(spans, peer, stream, true);
    }

    pub fn recvSpans(c: Comm, spans: []const Span, peer: u32, stream: cuda.Stream) !void {
        try c.p2p(spans, peer, stream, false);
    }

    fn p2p(c: Comm, spans: []const Span, peer: u32, stream: cuda.Stream, out: bool) !void {
        if (peer == c.rank or peer >= c.world) return error.BadPeer;
        var any = false;
        for (spans) |sp| any = any or sp.bytes > 0;
        if (!any) return;
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        var first: nccl.Result = 0;
        for (spans) |sp| {
            if (sp.bytes == 0) continue;
            const r = if (out)
                c.lib.api.ncclSend(sp.ptr, sp.bytes, .u8, @intCast(peer), c.comm, stream.handle)
            else
                c.lib.api.ncclRecv(sp.ptr, sp.bytes, .u8, @intCast(peer), c.comm, stream.handle);
            if (first == 0) first = r;
        }
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(first, if (out) "ncclSend" else "ncclRecv");
    }

    /// `count` bytes from this rank's slot of `buf` (buf + rank * count) to every rank's slot, in place (NCCL's
    /// in-place all-gather: the send buffer is the receive buffer's own slot).
    pub fn allGatherInPlace(c: Comm, buf: u64, count: usize, stream: cuda.Stream) !void {
        try c.allGather(buf + c.rank * count, buf, count, .u8, stream);
    }
};

/// A device byte span (Comm.sendSpans / recvSpans).
pub const Span = struct { ptr: u64, bytes: usize };

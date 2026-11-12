// DeepSeek-V4.1 TF_DS_KV_SHARD (zig/src/families/dsv41/shard.zig, shard_ops.zig): port of wojo's GLM 0118 (PR #152)
// sparse fetch and owner decode: the byte movers the sharded attention needs (fetch, select) and the prompt fetch's
// union (mark, scan, compact, renumber). Nothing here computes on the data: every byte lands unchanged and every id
// is a count, so the attention reads the replicated engine's bytes in the replicated engine's order.
// Built like dsv41_ops (no fast math); raw device pointers, the caller's stream.
// PORT(kit): a new fatbin (zig/build/cuda.zig's list): an nvcc build of the image (or the prebuilt fatbins' directory)
// must carry it; written on a host without nvcc, so never compiled yet: compile, then the GPU checks in
// docs/port/shard-attn.md.
#include <cuda_runtime.h>
#include <stdint.h>

// A round's (or a prompt block's) picked compressed rows, packed and renumbered: row r's pick j (idx [rows, ks], -1
// none, the first kk of a row; -1 past them up to kos entries) becomes out[r, j] =
//   how 0: j            (relative: the attention adds the row's base, its CBASE)
//   how 1: r * dk + j   (absolute: a prompt block's copy, no base)
//   how 2: idx[r, j] when own[r], else -1 (owner decode: a row this rank does not hold attends no compressed key)
// and with `copy` (how 0, 1), for a row this rank holds (own null: every row), its codes and scales (cb, sb bytes a
// row, multiples of 16) from plane row sbase[r] + idx[r, j] (sbase null: 0) into fetched row r * dk + j of dcodes,
// dscales. Grid (rows, ceil(kos / 16)), 256 threads: 16 picks a block, a pick's 16 lanes moving its 16-byte words.
extern "C" __global__ void __launch_bounds__(256) tf_ds_kv_fetch_kernel(
    const uint8_t* codes, const uint8_t* scales, long long cb, long long sb, const long long* idx, long long ks,
    int kk, const long long* sbase, const long long* own, int copy, uint8_t* dcodes, uint8_t* dscales, long long dk,
    long long* out, long long kos, int how) {
    const long long r = blockIdx.x;
    const long long j = (long long)blockIdx.y * 16 + threadIdx.x / 16;
    const int lane = threadIdx.x % 16;
    if (j >= kos) return;
    const long long v = j < kk ? idx[r * ks + j] : -1;
    const bool mine = own == nullptr || own[r] != 0;
    if (lane == 0) {
        long long o;
        if (how == 2) o = mine ? v : -1;
        else if (v < 0) o = -1;
        else o = how == 1 ? r * dk + j : j;
        out[r * kos + j] = o;
    }
    if (!copy || how == 2 || v < 0 || !mine) return;
    const long long src = (sbase != nullptr ? sbase[r] : 0) + v;
    const long long dst = r * dk + j;
    const uint4* sc = (const uint4*)(codes + src * cb);
    uint4* dc = (uint4*)(dcodes + dst * cb);
    for (long long w = lane; w < cb / 16; w += 16) dc[w] = sc[w];
    const uint4* ss = (const uint4*)(scales + src * sb);
    uint4* ds = (uint4*)(dscales + dst * sb);
    for (long long w = lane; w < sb / 16; w += 16) ds[w] = ss[w];
}

// Owner decode's return: dst row r of each of `planes` planes = src row r for the rows whose sel[r] equals `want` (the
// rows this rank does not hold take the owner's rotated wo_a inputs), the others untouched. Rows of `words` 16-byte
// words, planes `plane_words` words apart (xhwo's [group][rows, K] layout). Grid (rows, planes), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ds_select_rows_kernel(const uint4* src, uint4* dst,
                                                                            const long long* sel, long long want,
                                                                            long long words, long long plane_words) {
    const long long r = blockIdx.x;
    if (sel[r] != want) return;
    const long long at = (long long)blockIdx.y * plane_words + r * words;
    for (long long w = threadIdx.x; w < words; w += blockDim.x) dst[at + w] = src[at + w];
}

// The prompt fetch's union (GLM 0118's shard.history: a mark a position, then their prefix sums, in time linear in the
// picks; shard.UnionRef is the host model). mark[v] = 1 for every pick v below `limit` of idx [rows, ks] (the first kk
// of a row), the marks zeroed first (cuMemsetD32). Grid (rows, ceil(kk / 256)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ds_union_mark_kernel(const long long* idx, long long ks, int kk,
                                                                          int* mark, long long limit) {
    const long long r = blockIdx.x;
    const long long j = (long long)blockIdx.y * 256 + threadIdx.x;
    if (j >= kk) return;
    const long long v = idx[r * ks + j];
    if (v >= 0 && v < limit) mark[v] = 1;
}

// The marks' inclusive prefix sums in place (a[n]), in three passes over TF_DS_SCAN_SEGS segments of ceil(n / segs)
// entries, no barrier inside a launch: sums[s] = segment s's total; one thread turns sums into exclusive offsets;
// each segment rescans itself from its offset, the last writing the total to sums[segs] (the count the host reads).
// sums: int32 [segs + 1]. Grids (segs / 256) x 256 threads, (1) x 1 thread, (segs / 256) x 256 threads.
// PORT(perf): a decoupled look-back scan would make this one pass; a chunk's marks are at most its extent's rows.
#define TF_DS_SCAN_SEGS 1024
extern "C" __global__ void __launch_bounds__(256) tf_ds_union_sums_kernel(const int* a, long long n, int* sums) {
    const long long s = (long long)blockIdx.x * 256 + threadIdx.x;
    if (s >= TF_DS_SCAN_SEGS) return;
    const long long per = (n + TF_DS_SCAN_SEGS - 1) / TF_DS_SCAN_SEGS;
    const long long lo = s * per;
    const long long hi = lo + per < n ? lo + per : n;
    int t = 0;
    for (long long i = lo; i < hi; i++) t += a[i];
    sums[s] = t;
}

extern "C" __global__ void tf_ds_union_offsets_kernel(int* sums) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int run = 0;
    for (int s = 0; s < TF_DS_SCAN_SEGS; s++) {
        const int v = sums[s];
        sums[s] = run;
        run += v;
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_ds_union_scan_kernel(int* a, long long n, int* sums) {
    const long long s = (long long)blockIdx.x * 256 + threadIdx.x;
    if (s >= TF_DS_SCAN_SEGS) return;
    const long long per = (n + TF_DS_SCAN_SEGS - 1) / TF_DS_SCAN_SEGS;
    const long long lo = s * per;
    const long long hi = lo + per < n ? lo + per : n;
    int run = sums[s];
    for (long long i = lo; i < hi; i++) {
        run += a[i];
        a[i] = run;
    }
    if (s == TF_DS_SCAN_SEGS - 1) sums[TF_DS_SCAN_SEGS] = run;
}

// The owner's copy of the union: plane row t (codes, scales: the extent's view, cb / sb bytes a row, multiples of 16)
// into fetched row cum[t] - 1 for each marked t < limit (cum the scanned marks: t is marked where cum steps). Grid
// (ceil(limit / 16)), 256 threads: 16 rows a block, a row's 16 lanes moving its 16-byte words.
extern "C" __global__ void __launch_bounds__(256) tf_ds_union_compact_kernel(const uint8_t* codes, const uint8_t* scales,
                                                                             long long cb, long long sb, const int* cum,
                                                                             long long limit, uint8_t* dcodes,
                                                                             uint8_t* dscales) {
    const long long t = (long long)blockIdx.x * 16 + threadIdx.x / 16;
    const int lane = threadIdx.x % 16;
    if (t >= limit) return;
    const int prev = t > 0 ? cum[t - 1] : 0;
    if (cum[t] == prev) return;
    const long long d = prev;
    const uint4* sc = (const uint4*)(codes + t * cb);
    uint4* dc = (uint4*)(dcodes + d * cb);
    for (long long w = lane; w < cb / 16; w += 16) dc[w] = sc[w];
    const uint4* ss = (const uint4*)(scales + t * sb);
    uint4* ds = (uint4*)(dscales + d * sb);
    for (long long w = lane; w < sb / 16; w += 16) ds[w] = ss[w];
}

// The other rank's picks renumbered to the union's rows: out[r, j] = cum[v] - 1 for pick v = idx[r, j] below `limit`
// (j < kk), else -1 (none, and the padding up to kos entries a row). Grid (rows, ceil(kos / 256)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ds_union_renumber_kernel(const long long* idx, long long ks, int kk,
                                                                              const int* cum, long long limit,
                                                                              long long* out, long long kos) {
    const long long r = blockIdx.x;
    const long long j = (long long)blockIdx.y * 256 + threadIdx.x;
    if (j >= kos) return;
    const long long v = j < kk ? idx[r * ks + j] : -1;
    out[r * kos + j] = v >= 0 && v < limit ? (long long)cum[v] - 1 : -1;
}

// ---- TF_DS_KV_SHARD's decode exchanges on the family's RDMA ring (shard_ring.zig; rdma.zig Ring.sendRows/recvRows) --
// A sized one-hop transfer of a round's listed rows, each row one run of row16 16-byte words in every one of nseg
// segments. The host lists the rows (shard_ring.lists, from the replicated owners: rank r's send list
// is rank 1-r's receive list) and uploads them each round to a fixed block (int [1 + TF_DS_RING_ROWS]: the count, then
// the rows in slot order), so a captured launch serves every round. The stage kernel packs segment s of the n listed
// rows after the segments before it into this rank's ring slot and publishes the slot's size (n x the row words) with
// the doorbell, as dsv41/rdma_gather.cu's stage does (the same slot, sequence, sent, staged, abort and probe words);
// the proxy writes exactly those bytes to the peer; the collect kernel waits for the peer's flag at the sequence and
// puts each received row at its listed row in the destinations. shard_ring.zig's stageRef / collectRef are the host
// models; tools/dsv41_shard_emulate runs the packing below against them.
// PORT(kit): written without nvcc (never compiled or run); compile with the image, then docs/port/shard-ring.md's
// GPU checks.
#define TF_DS_RING_SEGS 8
#define TF_DS_RING_ROWS 64
#define TF_DS_STAGE_THREADS 512
#define TF_DS_COLLECT_THREADS 256

// rdma.zig RowSegs (extern struct, the same layout): a segment's source and destination, their row strides and a
// row's words (16-byte units). The stage reads src, the collect writes dst.
struct TfDsRingRows {
    const uint4* src[TF_DS_RING_SEGS];
    uint4* dst[TF_DS_RING_SEGS];
    long long src16[TF_DS_RING_SEGS];
    long long dst16[TF_DS_RING_SEGS];
    int row16[TF_DS_RING_SEGS];
};

#ifdef TF_DS_EMULATE
#define TF_DS_HD static inline
#define TF_DS_LDCV(p) (*(p))
#else
#define TF_DS_HD __device__ __forceinline__
#define TF_DS_LDCV(p) __ldcv(p)
#endif

// A list's count (the host writes 0 .. TF_DS_RING_ROWS; clamped so a bad block can never index past the list).
TF_DS_HD int tf_ds_ring_count(const int* list) {
    const int n = list[0];
    return n < 0 ? 0 : (n > TF_DS_RING_ROWS ? TF_DS_RING_ROWS : n);
}

// The slot's words for n listed rows (every segment's).
TF_DS_HD long long tf_ds_ring_words(const TfDsRingRows& s, int nseg, int n) {
    long long t = 0;
    for (int g = 0; g < nseg; ++g) t += (long long)n * s.row16[g];
    return t;
}

// This thread's share (word tid, tid + step, ...) of packing the listed rows (rows[0 .. n)) into the slot. 32-bit word
// indices: a slot is at most the ring's max_bytes (4.25 MiB, 2^18 words).
TF_DS_HD void tf_ds_ring_pack(const TfDsRingRows& s, int nseg, const int* rows, int n, uint4* slot, unsigned tid, unsigned step) {
    size_t off = 0;
    for (int g = 0; g < nseg; ++g) {
        const unsigned r16 = (unsigned)s.row16[g], words = (unsigned)n * r16;
        const uint4* src = s.src[g];
        const long long st = s.src16[g];
        for (unsigned i = tid; i < words; i += step) {
            const unsigned c = i / r16;
            slot[off + i] = src[(long long)rows[c] * st + (i - c * r16)];
        }
        off += words;
    }
}

// This thread's share of putting a received slot's rows at their listed rows: up to four slot loads in flight before
// their stores (as rdma_gather.cu's collect: the first reads after the flag wait out the pinned memory's latency).
TF_DS_HD void tf_ds_ring_unpack(const TfDsRingRows& s, int nseg, const int* rows, int n, const uint4* slot, unsigned tid, unsigned step) {
    size_t off = 0;
    for (int g = 0; g < nseg; ++g) {
        const unsigned r16 = (unsigned)s.row16[g], words = (unsigned)n * r16;
        uint4* dst = s.dst[g];
        const long long st = s.dst16[g];
        for (unsigned b = tid; b < words; b += 4 * step) {
            uint4 v[4];
            for (unsigned u = 0; u < 4; ++u)
                if (b + u * step < words) v[u] = TF_DS_LDCV(slot + off + b + u * step);
            for (unsigned u = 0; u < 4; ++u) {
                const unsigned i = b + u * step;
                if (i < words) {
                    const unsigned c = i / r16;
                    dst[(long long)rows[c] * st + (i - c * r16)] = v[u];
                }
            }
        }
        off += words;
    }
}

#ifndef TF_DS_EMULATE
namespace {
__device__ __forceinline__ uint64_t tf_ds_load_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void tf_ds_store_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ void tf_ds_store_relaxed_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t tf_ds_now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// rdma_gather.cu's v7 PDL entry: wait for the kernel before (its memory visible), then let the next one launch.
__device__ __forceinline__ void tf_ds_pdl_enter(int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

// The block's copy of a row list (the count then the rows), every thread loading a part of it.
__device__ __forceinline__ void tf_ds_ring_load_list(const int* list, int* s_list) {
    for (int i = threadIdx.x; i <= TF_DS_RING_ROWS; i += blockDim.x) s_list[i] = list[i];
}
}  // namespace

// dsv41_rdma_stage's protocol over the listed rows: every block waits until the slot's previous writes are sent,
// copies its share in, counts itself in; the last block advances the sequence and rings the doorbell. Block 0 writes
// the slot's size (fence_mode 1: before the blocks' releases, as dsv41_rdma_stage). The list loads overlap the slot
// wait (thread 0 waits, the others load).
extern "C" __global__ void __launch_bounds__(TF_DS_STAGE_THREADS) tf_ds_ring_stage_rows(TfDsRingRows s, int nseg,
        const int* list, char* send_ring, uint64_t* sizes, uint64_t* doorbell, const uint64_t* sent, uint64_t* seq,
        int slots, uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, unsigned* staged, int fence_mode,
        int pdl) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
    __shared__ int s_list[TF_DS_RING_ROWS + 1];
    tf_ds_pdl_enter(pdl);                        // before anything reads the rows, their list or the sequence
    const uint64_t t0 = tf_ds_now_ns();
    tf_ds_ring_load_list(list, s_list);
    if (threadIdx.x == 0) {
        const uint64_t q = *seq + 1;
        const int slot = (int)(q % slots);
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; tf_ds_load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && tf_ds_load_sys(abort_word)) break;
        s_seq = q;
    }
    __syncthreads();
    const int n = tf_ds_ring_count(s_list);
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    if (fence_mode && blockIdx.x == 0 && threadIdx.x == 0) tf_ds_store_relaxed_sys(sizes + slot, (uint64_t)tf_ds_ring_words(s, nseg, n) * 16);
    uint4* dst = reinterpret_cast<uint4*>(send_ring + (size_t)slot * max_bytes);
    tf_ds_ring_pack(s, nseg, s_list + 1, n, dst, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
    if (fence_mode) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
        }
    } else {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *staged = 0;
        *seq = q;
        if (fence_mode) {
            tf_ds_store_relaxed_sys(doorbell, q);
        } else {
            tf_ds_store_sys(sizes + slot, (uint64_t)tf_ds_ring_words(s, nseg, n) * 16);
            tf_ds_store_sys(doorbell, q);
        }
        if (probe) {
            probe[0] += tf_ds_now_ns() - t0;
            probe[4] = tf_ds_now_ns();
        }
    }
}

// dsv41_rdma_collect_into's wait (rank `from`'s flag at this rank's latest sequence), then the slot's rows put at
// their listed rows (the rows `from` sent, in its send list's order). The list loads overlap the flag wait.
extern "C" __global__ void __launch_bounds__(TF_DS_COLLECT_THREADS) tf_ds_ring_collect_rows(TfDsRingRows s, int nseg,
        const int* list, const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int from, int world,
        int slots, uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, int pdl) {
    __shared__ int s_list[TF_DS_RING_ROWS + 1];
    tf_ds_pdl_enter(pdl);                        // the stage kernel's sequence is written
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    tf_ds_ring_load_list(list, s_list);
    if (threadIdx.x == 0)
        for (uint32_t i = 1; tf_ds_load_sys(flags + (size_t)slot * world + from) < q; ++i)
            if ((i & 1023) == 0 && tf_ds_load_sys(abort_word)) break;
    __syncthreads();
    const uint64_t t2 = tf_ds_now_ns();
    const uint4* in = reinterpret_cast<const uint4*>(recv_ring + ((size_t)slot * world + from) * max_bytes);
    tf_ds_ring_unpack(s, nseg, s_list + 1, tf_ds_ring_count(s_list), in, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += tf_ds_now_ns() - t2;
            probe[3] += 1;
        }
    }
}
#endif  // TF_DS_EMULATE

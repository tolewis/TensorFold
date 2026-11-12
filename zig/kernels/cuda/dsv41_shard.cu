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

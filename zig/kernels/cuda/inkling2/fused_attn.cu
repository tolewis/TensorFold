// The fused visit's unit of inkling2/attention.cu (the -O3 flags of inkling2_attn): prep8, the rel table, the fp8
// attention tiles on the 68,608 B ring of tf_ink2_attn8_db, and the merge, each with today's code and blockIdx named by
// a block-shared value (fused-cuda DESIGN section 11). blockDim stays 128.
#include <cstdint>
#include <cuda_runtime.h>

__shared__ uint3 fz_bi_attn;
#define blockIdx fz_bi_attn
#undef __launch_bounds__
#define __launch_bounds__(...)
#pragma push_macro("__global__")
#undef __global__
#define __global__ __device__
#include "inkling2/attention.cu"
#pragma pop_macro("__global__")
#undef blockIdx

#include "inkling2/fused_visit.cuh"

namespace {
__device__ __forceinline__ void enter(uint3 b) {
    __syncthreads();
    if (threadIdx.x == 0) fz_bi_attn = b;
    __syncthreads();
}
}  // namespace

// tf_ink2_prep8 block (row 0, y) on warp 0 (today's block is 32 threads).
extern "C" __device__ void fz_prep8(int y, TF_INK2_PREP_ARGS, float* __restrict__ ksc, float* __restrict__ vsc) {
    enter(make_uint3(0, y, 0));
    if (threadIdx.x < 32) ink2::prepRow<true>(TF_INK2_PREP_PASS, ksc, vsc);
}

// tf_ink2_rel_table block rh.
extern "C" __device__ void fz_rel_table(int rh, const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj,
                                        uint16_t* __restrict__ tab) {
    enter(make_uint3(rh, 0, 0));
    tf_ink2_rel_table(rel, proj, tab);
}

// tf_ink2_attn8_db (db 1) or tf_ink2_attn8 (db 0) block (tile x, kv head y, z block zb) of a call with gridDim.z =
// zn: today's zb loop of that block. `dyn` holds the 68,608 B of db8_shared, or the two tiles of tf_ink2_attn8.
extern "C" __device__ void fz_attn8(uint3 b, int zn_grid, const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                    const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab,
                                    const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                                    float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads,
                                    int ring, int window, int chunk, int c0, int chunks, float scale, int split,
                                    const float* __restrict__ ksc, const float* __restrict__ vsc, int db, unsigned char* dyn) {
    using namespace ink2;
    enter(b);
    uint16_t* ks = reinterpret_cast<uint16_t*>(dyn);
    uint16_t* vs = ks + kKeys * kPad;
    const int zn = split ? chunks : zn_grid;
    if (db)
        for (int zb = static_cast<int>(b.z); zb < zn; zb += zn_grid)
            attnTile<false, true, true>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window,
                                        chunk, c0, chunks, scale, split, ks, vs, ksc, vsc, zb);
    else
        for (int zb = static_cast<int>(b.z); zb < zn; zb += zn_grid)
            attnTile<false, true>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window, chunk,
                                  c0, chunks, scale, split, ks, vs, ksc, vsc, zb);
}

// tf_ink2_merge block rh.
extern "C" __device__ void fz_merge(int rh, const float* __restrict__ part_o, const float2* __restrict__ part_ml,
                                    const uint16_t* __restrict__ sink, const uint16_t* __restrict__ gate,
                                    const int* __restrict__ rpos, uint16_t* __restrict__ out, int heads, int window,
                                    int chunk, int c0, int chunks, int subs) {
    enter(make_uint3(rh, 0, 0));
    tf_ink2_merge(part_o, part_ml, sink, gate, rpos, out, heads, window, chunk, c0, chunks, subs);
}

// The attention phase: today's grid (1 tile, kv heads, zgrid) as units over the blocks. With fz_fix::kBusy a split
// (global) call deals only its busy units, z blocks whose first key is at or below the tile's last position (the
// early return of attnTile), round robin: no block holds more than ceil(busy / G) of them (review A9).
extern "C" __device__ void fz_attn8_phase(const FzAttn& a, int fix, unsigned char* dyn) {
    using namespace ink2;
    int nz = a.zgrid;
    if ((fix & fz_fix::kBusy) && a.split) {
        const Seg s = static_cast<const Seg*>(a.segs)[a.tiles[0].x];
        const int group = a.heads / a.kv_heads, total = s.n * group;
        const int tile_hi = s.p0 + min(a.tiles[0].y + 63, total - 1) / group;
        nz = min(a.zgrid, tile_hi / kSub + 1);
    }
    for (int u = blockIdx.x; u < a.kv_heads * nz; u += gridDim.x)
        fz_attn8(make_uint3(0, u / nz, u % nz), a.zgrid, a.q, a.kc, a.vc, a.tab, static_cast<const Seg*>(a.segs), a.tiles,
                 a.part_o, a.part_ml, a.heads, a.kv_heads, a.ring, a.window, a.chunk, 0, a.chunks, a.scale, a.split, a.ksc,
                 a.vsc, a.db, dyn);
}

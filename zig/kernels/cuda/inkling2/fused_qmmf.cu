// The fused visit's unit of qmmf.cu (the -O3 -DTF_NATIVE_FP_DECODE flags of the qmmf module): one (64-column tile,
// K-slice) unit of qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, false, CLUSTER, false> with today's body, and the tile sum that
// replaces the cluster sum (fused-cuda DESIGN section 4.2, bench B3). A unit of an SK > 1 call stores its fp32
// accumulators to `part` (the body's non-cluster path). The last slice of a tile then adds the slices in the order of
// the cluster sum, acc = ((p0 + p1) + p2) + p3, scales and rounds to bf16 as the body's epilogue.
#include "qmmf.cu"

namespace tf_qmmf {
struct FusedBI {
    int x, y, z;
};

template <int MODE>
__device__ __forceinline__ void fused_unit(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
                                        const uint8_t* __restrict__ bs, float scale, void* __restrict__ out,
                                        float* __restrict__ part, int N, int K, int SK, int npad, int tile, int slc) {
    constexpr int BM = 16, BN = 64, WM = 1, WN = 4, STAGES = 4;
    constexpr bool F32 = false, CLUSTER = false, FUSE = false;
    const int M = 1, ldx = K, group = 1;
    const FusedBI fbi = {tile, 0, slc};
#define blockIdx fbi
#include "qmmf_body.cuh"
#undef blockIdx
}
}  // namespace tf_qmmf

// Unit (tile, slc) of a one-row MXFP8 call. `part` is [SK][N] fp32 when SK > 1. Dynamic shared memory: the body's.
extern "C" __device__ void fz_qmmf_unit(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
                                        const uint8_t* __restrict__ bs, float scale, void* __restrict__ out,
                                        float* __restrict__ part, int N, int K, int SK, int npad, int tile, int slc,
                                        int mode) {
    __syncthreads();
    // the attention lanes are per-tensor FP8 (mini_model fp8Lane), the head MXFP8
    if (mode == tf_qmmf::FP8)
        tf_qmmf::fused_unit<tf_qmmf::FP8>(x, w, bs, scale, out, part, N, K, SK, npad, tile, slc);
    else
        tf_qmmf::fused_unit<tf_qmmf::MXFP8>(x, w, bs, scale, out, part, N, K, SK, npad, tile, slc);
}

// The tile sum of tile `tile`, threads 0 to 63 a column each, after every slice of the tile stored its part.
extern "C" __device__ void fz_qmmf_tile_sum(const float* __restrict__ part, __nv_bfloat16* __restrict__ out, int N,
                                            int SK, float scale, int tile) {
    if (threadIdx.x >= 64) return;
    const int col = tile * 64 + threadIdx.x;
    if (col >= N) return;
    float acc = __ldcg(part + col);
    for (int s = 1; s < SK; ++s) acc = acc + __ldcg(part + static_cast<size_t>(s) * N + col);
    out[col] = __float2bfloat16_rn(acc * scale);
}

// The fused visit's unit of inkling2/ops.cu (the glue flags of inkling2_ops). Every kernel of ops.cu is compiled
// here as a __device__ function, with blockIdx named by a block-shared value, so a block of tf_ink2_fused_visit runs
// one block of today's kernel with today's code (fused-cuda DESIGN section 11). blockDim stays 128.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__shared__ uint3 fz_bi_ops;
#define blockIdx fz_bi_ops
#undef __launch_bounds__
#define __launch_bounds__(...)
#pragma push_macro("__global__")
#undef __global__
#define __global__ __device__
#include "inkling2/ops.cu"
#pragma pop_macro("__global__")
#undef blockIdx

namespace {
__device__ __forceinline__ void enter(uint3 b) {
    __syncthreads();
    if (threadIdx.x == 0) fz_bi_ops = b;
    __syncthreads();
}
}  // namespace

// tf_ink2_conv block (x, y, z) of today's grid [ceil(C / 128), streams, rows / 32].
extern "C" __device__ void fz_conv(uint3 b, const ConvArgs& a) {
    enter(b);
    tf_ink2_conv(a);
}

// tf_ink2_rowdot output j of row 0 on the calling warp: today's warp code (block y = j / 8, warp j % 8).
extern "C" __device__ void fz_rowdot_warp(int j, const uint16_t* __restrict__ x, const uint16_t* __restrict__ W,
                                          void* __restrict__ out, int J, int C, int out_bf16, int act) {
    const int lane = threadIdx.x & 31;
    if (j >= J) return;
    const size_t row = 0;
    const uint16_t* xr = x + row * C;
    const uint16_t* wr = W + static_cast<size_t>(j) * C;
    float s = 0.0f;
    for (int c = lane; c < C; c += 32) s += bf(xr[c]) * bf(wr[c]);
    s = warpSum(s);
    if (lane != 0) return;
    if (act == 1) s = 1.0f / (1.0f + expf(-s));
    if (out_bf16)
        static_cast<uint16_t*>(out)[row * J + j] = tobf(s);
    else
        static_cast<float*>(out)[row * J + j] = s;
}

// tf_ink2_topk_warp for row 0 on warp 0 of the block (block 0 of today's grid; warps 1 to 3 have no row).
extern "C" __device__ void fz_topk_warp(const float* __restrict__ scores, const float* __restrict__ bias,
                                        int* __restrict__ pick, float* __restrict__ wts, int E, int K, float scale) {
    enter(make_uint3(0, 0, 0));
    tf_ink2_topk_warp(scores, bias, pick, wts, E, K, scale, 1);
}

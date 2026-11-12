// The FP8 prompt GEMM with a checkpoint's own scales on a 128 x 256 tile: 8 warps of 64 x 64, one barrier a step.
#pragma once

#include "ck4_prompt2.cuh"

namespace tf_ck8w {

constexpr int WM = 128, WN = 256, WSTAGES = 3;  // rows, columns, staged 64-input steps

// One staged 64-input step: rows with an m8 block's 16-byte chunks in a line for ldmatrix, four weight tiles as packed.
struct Step {
    uint4 a[WM / 8][4][8];
    uint4 b[4][256];
};
static_assert(sizeof(Step) == 24576, "ck.zig's lane_w_shared");

// tf_ck::lane's sums, row by row, for a column count that 256 divides and no K slices.
__device__ __forceinline__ void lane(const uint8_t* __restrict__ x, const uint8_t* __restrict__ w, float alpha,
                                     __nv_bfloat16* __restrict__ out, int M, int N, int K, int group,
                                     const float* __restrict__ sa, const float* __restrict__ ws) {
    extern __shared__ __align__(128) unsigned char tf_ck8w_raw[];
    Step* st = reinterpret_cast<Step*>(tf_ck8w_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3, q = lane >> 3, rr = lane & 7;
    const int KG = K / 64;
    const int2 at = tile_of(blockIdx.x, M, N, WM, WN, group);
    const int m0 = at.x, n0 = at.y;
    // loader roles: chunk tid % 4 of rows tid / 4 and 64 + tid / 4; a warp, four 512-byte runs of weight tile warp / 2
    const int ch = tid & 3, ar = tid >> 2, bo = (warp & 1) * 128 + lane;
    const int rlo = min(m0 + ar, M - 1), rhi = min(m0 + 64 + ar, M - 1);  // past M: the last row, masked at the end
    const uint8_t* xlo = x + static_cast<size_t>(rlo) * K + ch * 16;
    const uint8_t* xhi = x + static_cast<size_t>(rhi) * K + ch * 16;
    const uint4* wt = reinterpret_cast<const uint4*>(w) + static_cast<size_t>(n0 / 64 + (warp >> 1)) * KG * 256 + bo;
    auto load = [&](int kg) {
        Step& p = st[kg % WSTAGES];
        tf_ck4p::cp16_l1(&p.a[ar >> 3][ch][ar & 7], xlo + kg * 64);
        tf_ck4p::cp16_l1(&p.a[8 + (ar >> 3)][ch][ar & 7], xhi + kg * 64);
#pragma unroll
        for (int i = 0; i < 4; ++i) cp16(&p.b[warp >> 1][bo + 32 * i], wt + static_cast<size_t>(kg) * 256 + 32 * i);
    };
#pragma unroll
    for (int s = 0; s < WSTAGES - 1; ++s) {
        if (s < KG) load(s);
        commit();
    }
    wait<WSTAGES - 2>();
    __syncthreads();
    float acc[4][8][4] = {};
    for (int kg = 0; kg < KG; ++kg) {
        const Step& p = st[kg % WSTAGES];
        uint4 b[8];
        uint32_t a[4][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = p.b[wn][j * 32 + lane];
        // ldmatrix: lanes 8q .. 8q + 7 give rows 8 (q % 2) .. + 7 of chunk q / 2 of the half
#pragma unroll
        for (int i = 0; i < 4; ++i) ldmatrix4(a[i], &p.a[wm * 8 + i * 2 + (q & 1)][q >> 1][rr]);
        // the first half's operands are in registers: the copies for step kg + WSTAGES - 1 fill under the mma block
        if (kg + WSTAGES - 1 < KG) load(kg + WSTAGES - 1);
        commit();
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) mma_fp8(acc[i][j], a[i], b[j].x, b[j].y);
#pragma unroll
        for (int i = 0; i < 4; ++i) ldmatrix4(a[i], &p.a[wm * 8 + i * 2 + (q & 1)][2 + (q >> 1)][rr]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) mma_fp8(acc[i][j], a[i], b[j].z, b[j].w);
        wait<WSTAGES - 2>();  // step kg + 1 is in
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = n0 + wn * 64 + j * 8 + t * 2;
            const float* rs = ws + static_cast<size_t>(col >> 6) * KG * 64 + (col & 63);
            const float r0 = ws ? rs[0] : 1.0f, r1 = ws ? rs[1] : 1.0f;  // an even column's scale and its right one's
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 64 + i * 16 + g + h * 8;
                if (row >= M) continue;
                const float f = alpha * (sa ? sa[row] : 1.0f);
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                const float v0 = acc[i][j][2 * h] * (f * r0), v1 = acc[i][j][2 * h + 1] * (f * r1);
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
            }
        }
}

}  // namespace tf_ck8w

// As tf_ck8_128r at SK 1: ceil(M / 128) * (N / 256) blocks of 256 threads, WSTAGES * sizeof(Step) dynamic shared.
extern "C" __global__ void __launch_bounds__(256, 1)
    tf_ck8_256r(const uint8_t* __restrict__ x, const uint8_t* __restrict__ w, float alpha,
                __nv_bfloat16* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int group,
                const float* __restrict__ sa, const float* __restrict__ ws) {
    tf_ck8w::lane(x, w, alpha, out, M, N, K, group, sa, ws);
}

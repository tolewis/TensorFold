// qwen35.cu's row-wise kernels: the tap ring's residuals and the MLP activation, eight entries a thread.
#pragma once

#include "nvfp4q.cuh"

// A tapped layer's residual bf16(h + r) into the segment's tap ring at the row's position; grid (rows, D / 2048).
extern "C" __global__ void __launch_bounds__(256) tf_q35_tap(const __nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                              const Seg* __restrict__ segs, const int* __restrict__ rowseg, int tap, int D,
                                                              const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const Seg sg = segs[rowseg[m]];
    if (sg.tap == 0) return;
    const long long pos = sg.len + (long long)m - sg.row0;
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(sg.tap) + ((size_t)(pos % sg.tap_rows) * sg.taps + tap) * D;
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;  // eight entries a thread, grid (rows, D / 2048)
    if (i >= D) return;
    const uint4 hw = *reinterpret_cast<const uint4*>(H + m * D + i), rw = *reinterpret_cast<const uint4*>(R + m * D + i);
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hw);
    const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rw);
    uint4 ow;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ow);
#pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = __float2bfloat16_rn(bf(hb[j]) + bf(rb[j]));
    *reinterpret_cast<uint4*>(out + i) = ow;
}

// out = bf16(silu(gate) * up) over `total` values, rows D wide, four a thread; XS (may be null) their group sums.
extern "C" __global__ void __launch_bounds__(256) tf_q35_mlp_act(const __nv_bfloat16* __restrict__ GATE, const __nv_bfloat16* __restrict__ UP,
                                            __nv_bfloat16* __restrict__ OUT, long long total, const int* __restrict__ P, int D, float* __restrict__ XS) {
    __shared__ __align__(16) __nv_bfloat16 done[1024];
    if (P != nullptr) total = (long long)P[2] * D;
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < total) {
        const uint2 gw = *reinterpret_cast<const uint2*>(GATE + i), uw = *reinterpret_cast<const uint2*>(UP + i);
        const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&gw);
        const __nv_bfloat16* ub = reinterpret_cast<const __nv_bfloat16*>(&uw);
        uint2 ow;
        __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ow);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float gf = bf(gb[j]);
            ob[j] = __float2bfloat16_rn(gf / (1.0f + expf(-gf)) * bf(ub[j]));
        }
        *reinterpret_cast<uint2*>(OUT + i) = ow;
        *reinterpret_cast<uint2*>(done + threadIdx.x * 4) = ow;
    }
    if (XS == nullptr) return;
    __syncthreads();
    const long long first = (long long)blockIdx.x * 1024 + threadIdx.x * 64;  // the block's 16 groups, each in order
    if (threadIdx.x < 16 && first < total) XS[first / 64] = group_sum(done + threadIdx.x * 64);
}

// tf_q35_mlp_act's bf16 values quantized as ck_lane.cu's tf_ck_quant4_rows, a 16-value block a thread, for the
// down projection's NVFP4 prompt GEMM: e2m1 codes [rows][D / 2] at CODES, e4m3 block scales [rows][D / 16] at SCALES.
extern "C" __global__ void __launch_bounds__(128) tf_q35_mlp_act_q4(const __nv_bfloat16* __restrict__ GATE, const __nv_bfloat16* __restrict__ UP,
                                                                     int rows, int D, float G, uint8_t* __restrict__ CODES, uint8_t* __restrict__ SCALES) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * (D / 16)) return;
    const int row = (int)(i / (D / 16)), blk = (int)(i % (D / 16));
    const size_t at = (size_t)row * D + blk * 16;
    float f[16];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const uint4 gw = *reinterpret_cast<const uint4*>(GATE + at + 8 * h), uw = *reinterpret_cast<const uint4*>(UP + at + 8 * h);
        const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&gw);
        const __nv_bfloat16* ub = reinterpret_cast<const __nv_bfloat16*>(&uw);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float gf = bf(gb[j]);
            f[8 * h + j] = bf(__float2bfloat16_rn(gf / (1.0f + expf(-gf)) * bf(ub[j])));  // tf_q35_mlp_act's value
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float a = fabsf(f[j]);
        amax = (amax > a || amax != amax) ? amax : a;  // tf_ck4p::max_nan
    }
    const nvfp4q::Scale sc = nvfp4q::block_scale(amax, G);
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 16; ++j) w[j / 8] |= nvfp4q::e2m1(f[j] * sc.mul) << (4 * (j % 8));
    *reinterpret_cast<uint2*>(CODES + (size_t)row * (D / 2) + blk * 8) = make_uint2(w[0], w[1]);
    SCALES[(size_t)row * (D / 16) + blk] = static_cast<uint8_t>(sc.sf8);
}

// The fused visit's unit of moe_glue.cu, dense_rows.cu and experts.cu's plan (the -O3 flags of those modules): the
// weighted combine, the k and v taps of a global visit, and the expert plan of one decode row.
#include "moe_glue.cu"

#include <cuda_bf16.h>

// tf_moe_combine column d of row 0, y fp32, out bf16 (combine_row<0, 2>, today's slot order).
extern "C" __device__ void fz_combine(const float* y, const float* w, uint16_t* out, int slots, int dims, int d) {
    if (d >= dims) return;
    combine_row<0, 2>(y, w, out, slots, dims, 0, d);
}

namespace fz_dense {
__device__ __forceinline__ void unpack(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
}  // namespace fz_dense

// tf_dense_rows_scaled_1 output `col` of row 0 on the calling warp (dense_rows.cu tile<1, true>, today's code).
extern "C" __device__ void fz_dense_scaled_warp(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* scale,
                                                float* out, int n, int k, int col) {
    constexpr int VEC = 8;
    const int lane = threadIdx.x & 31;
    if (col >= n) return;
    float acc = 0.f;
    const __nv_bfloat16* wr = w + static_cast<int64_t>(col) * k;
#pragma unroll 4
    for (int k0 = lane * VEC; k0 < k; k0 += 32 * VEC) {
        float wf[VEC], xf[VEC];
        fz_dense::unpack(*reinterpret_cast<const uint4*>(wr + k0), wf);
        fz_dense::unpack(*reinterpret_cast<const uint4*>(x + k0), xf);
#pragma unroll
        for (int i = 0; i < VEC; ++i) acc = __fmaf_rn(xf[i], wf[i], acc);
    }
    float v = acc;
    for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    v = __fmul_rn(__bfloat162float(__float2bfloat16_rn(v)), scale[col]);
    if (lane == 0) out[col] = v;
}

// experts.cu plan_kernel for P <= 64 pairs and E <= 128 experts on one 128-thread block: integers only. The scan of
// place_items over 4 warps, the warps 4 to 31 of today's 1,024 threads counted as 0 (their experts have no pairs).
extern "C" __device__ void fz_plan(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                                   int* __restrict__ items, int* __restrict__ counts) {
    __shared__ int pk[64];
    __shared__ int cnt[128];
    __shared__ int off[128];
    __shared__ int wsum[3][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    for (int e = tid; e < 128; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = tid; p < P; p += blockDim.x) {
        const int e = __ldcg(picks + p);
        pk[p] = e;
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    const int c = tid < E ? cnt[tid] : 0;
    const int tiles = (c + T - 1) / T;
    int a = c, b = tiles, u = c > 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ua = __shfl_up_sync(0xffffffffu, a, o), ub = __shfl_up_sync(0xffffffffu, b, o);
        const int uu = __shfl_up_sync(0xffffffffu, u, o);
        if (lane >= o) {
            a += ua;
            b += ub;
            u += uu;
        }
    }
    if (lane == 31) {
        wsum[0][warp] = a;
        wsum[1][warp] = b;
        wsum[2][warp] = u;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            int v = lane < nw ? wsum[s][lane] : 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int w = __shfl_up_sync(0xffffffffu, v, o);
                if (lane >= o) v += w;
            }
            wsum[s][lane] = v;
        }
    }
    __syncthreads();
    const int o = (warp ? wsum[0][warp - 1] : 0) + a - c;
    const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
    if (tid == 0) {
        counts[0] = wsum[1][31];
        counts[1] = wsum[2][31];
    }
    if (tid < E)
        for (int j = 0; j < tiles; ++j) {
            int* it = items + 3 * (ioff + j);
            it[0] = tid;
            it[1] = o + T * j;
            it[2] = min(T, c - T * j);
        }
    if (tid < E) off[tid] = o;
    __syncthreads();
    for (int p = tid; p < P; p += blockDim.x) {
        const int e = pk[p];
        int rank = 0;
        for (int q = 0; q < p; ++q) rank += pk[q] == e;
        members[off[e] + rank] = p;
    }
    __syncthreads();
}

// The fused visit's unit of inkling2/loop.cu (the glue flags of inkling2_loop): the gates of a decode row on 128
// threads. tf_ink2_loop_gate_down and tf_ink2_loop_gate_mix are 256-thread kernels: real thread t runs virtual
// threads t and t + 128 with their own sums, and blockSum adds the 8 virtual warp partials in warp order, as today
// (fused-cuda DESIGN section 4.1 rule 3). tf_ink2_loop_gate_lin_rows is a 128-thread kernel: its body with the down row
// in the caller's shared memory instead of a static array. Every product and sum is today's, in today's order.
#include <cstdint>
#include <cuda_runtime.h>

#pragma push_macro("__global__")
#undef __global__
#define __global__ __device__
#undef __launch_bounds__
#define __launch_bounds__(...)
#include "inkling2/loop.cu"
#pragma pop_macro("__global__")

namespace {
constexpr int kR = 128;  // real threads a block

// Today's blockSum of a 256-thread block, the virtual warp partials of v[0] (virtual threads t) and v[1] (t + 128).
__device__ __forceinline__ float blockSum2(const float (&v)[2], float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float a = warpSum(v[0]), b = warpSum(v[1]);
    __syncthreads();
    if (lane == 0) {
        sh[w] = a;
        sh[w + 4] = b;
    }
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < kWarps; ++i) t += sh[i];
    return t;
}
}  // namespace

// tf_ink2_loop_gate_down block (row 0, gate g, z): rank rows 8 z to 8 z + 7, two a real warp (virtual warps w, w + 4).
// `xs` is 2 H floats of the caller's shared memory, `sh` 8 floats.
extern "C" __device__ void fz_gate_down(const LoopGateArgs& a, int g, int z, float* xs, float* sh) {
    const size_t row = 0;
    const int H = a.H;
    const float* h = a.h[g] + row * H;
    const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
    float sh_acc[2] = {0.0f, 0.0f}, sm_acc[2] = {0.0f, 0.0f};
    for (int k = 0; k < 2; ++k)
        for (int i = threadIdx.x + k * kR; i < H; i += kThreads) {
            sh_acc[k] = fmaf(h[i], h[i], sh_acc[k]);
            sm_acc[k] = fmaf(m[i], m[i], sm_acc[k]);
        }
    const float ih = rsqrtf(blockSum2(sh_acc, sh) / static_cast<float>(H) + a.eps_unit);
    const float im = rsqrtf(blockSum2(sm_acc, sh) / static_cast<float>(H) + a.eps_unit);
    for (int i = threadIdx.x; i < H; i += kR) {
        xs[i] = bf(tobf(h[i] * ih));
        xs[H + i] = bf(tobf(m[i] * im));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < 2; ++k) {
        const int w = (threadIdx.x >> 5) + 4 * k;
        const int r = z * kWarps + w;
        if (r >= a.rank) continue;
        const uint16_t* wr = a.down_w[g] + static_cast<size_t>(r) * 2 * H;
        float acc = 0.0f;
        for (int i = lane; i < 2 * H; i += 32) acc = fmaf(bf(wr[i]), xs[i], acc);
        acc = warpSum(acc);
        if (lane == 0) a.down[(static_cast<size_t>(g) * a.rows + row) * a.rank + r] = tobf(acc);
    }
    __syncthreads();
}

// tf_ink2_loop_gate_lin_rows block (row tile 0, gate g, z) for one row, rank 64. `dn` is 64 floats of shared memory.
extern "C" __device__ void fz_gate_lin(const LoopGateArgs& a, int g, int z, float* dn) {
    const int H = a.H;
    const int i = z * kLin + threadIdx.x;
    const size_t row0 = 0;
    const int nr = min(kLR, a.rows - static_cast<int>(row0));
    const uint16_t* dsrc = a.down + (static_cast<size_t>(g) * a.rows + row0) * kLRank;
    __syncthreads();
    for (int e = threadIdx.x; e < nr * kLRank; e += kLin) dn[e] = bf(dsrc[e]);
    uint32_t wf[kLRank / 2], wg[kLRank / 2];
    if (i < H) {
        const uint4* pf = reinterpret_cast<const uint4*>(a.forget[g] + static_cast<size_t>(i) * kLRank);
        const uint4* pg = reinterpret_cast<const uint4*>(a.gain[g] + static_cast<size_t>(i) * kLRank);
#pragma unroll
        for (int v = 0; v < kLRank / 8; ++v) {
            const uint4 f = pf[v], q = pg[v];
            wf[4 * v] = f.x; wf[4 * v + 1] = f.y; wf[4 * v + 2] = f.z; wf[4 * v + 3] = f.w;
            wg[4 * v] = q.x; wg[4 * v + 1] = q.y; wg[4 * v + 2] = q.z; wg[4 * v + 3] = q.w;
        }
    }
    __syncthreads();
    if (i < H)
        for (int j = 0; j < nr; ++j) {
            float f = 0.0f, gn = 0.0f;
#pragma unroll
            for (int r = 0; r < kLRank; ++r) {
                const uint32_t pf = wf[r >> 1], pg = wg[r >> 1];
                const float fw = (r & 1) ? __uint_as_float(pf & 0xFFFF0000u) : __uint_as_float(pf << 16);
                const float gw = (r & 1) ? __uint_as_float(pg & 0xFFFF0000u) : __uint_as_float(pg << 16);
                f = fmaf(fw, dn[j * kLRank + r], f);
                gn = fmaf(gw, dn[j * kLRank + r], gn);
            }
            const size_t row = row0 + j;
            const float* h = a.h[g] + row * H;
            const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
            a.ys[(static_cast<size_t>(g) * a.rows + row) * H + i] = (1.0f - tanhf(f)) * h[i] + tanhf(gn) * m[i];
        }
    __syncthreads();
}

// tf_ink2_loop_gate_mix block (row 0, gate g) on 128 threads. `sh` is 8 floats.
extern "C" __device__ void fz_gate_mix(const LoopGateArgs& a, int g, float* sh) {
    const size_t row = 0;
    const int H = a.H;
    const float* ys = a.ys + (static_cast<size_t>(g) * a.rows + row) * H;
    float ss[2] = {0.0f, 0.0f};
    for (int k = 0; k < 2; ++k)
        for (int i = threadIdx.x + k * kR; i < H; i += kThreads) ss[k] = fmaf(ys[i], ys[i], ss[k]);
    const float inv = rsqrtf(blockSum2(ss, sh) / static_cast<float>(H) + a.eps);
    uint16_t* dst = a.out + g * a.out_stride + row * H;
    for (int i = threadIdx.x; i < H; i += kR) dst[i] = tobf(bf(a.norm[g][i]) * (ys[i] * inv));
    __syncthreads();
}

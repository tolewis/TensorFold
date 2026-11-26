// Inkling 2 glue: embedding, block attention-residual mix, row dots, router top-k, softcapped SwiGLU, short convs.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "exl3_decode.cuh"
#include <cuda_fp8.h>
#include "common/qmm_frag.cuh"

namespace ink2o {

constexpr int kMaxBlocks = 8;
constexpr int kMaxQueries = 3;
constexpr int kConvTaps = 4;
constexpr int kMixCache = 10;  // tf_ink2_mix holds a row of up to kMixCache x blockDim columns in registers

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfr(float f) { return bf(tobf(f)); }
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sums of N values over the block in a fixed order: warp shuffles, then warp 0 over the warps' partials.
template <int N>
__device__ __forceinline__ void blockSums(float (&v)[N], float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = 0; i < N; ++i) v[i] = warpSum(v[i]);
    __syncthreads();
    if (lane == 0)
        for (int i = 0; i < N; ++i) sh[i * 32 + w] = v[i];
    __syncthreads();
    for (int i = 0; i < N; ++i) {
        float t = lane < nw ? sh[i * 32 + lane] : 0.0f;
        v[i] = __shfl_sync(0xffffffffu, warpSum(t), 0);
    }
}

}  // namespace ink2o

using namespace ink2o;

// Per row: the embedding row times rsqrt(mean square + eps) times the norm weight, one bf16 round (embed_norm).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_embed(const int* __restrict__ ids, const uint16_t* __restrict__ table,
                                                               const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
                                                               int C, float eps) {
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const uint16_t* e = table + static_cast<size_t>(ids[row]) * C;
    float s[1] = {0.0f};
    for (int c = threadIdx.x; c < C; c += blockDim.x) s[0] += bf(e[c]) * bf(e[c]);
    blockSums<1>(s, sh);
    const float r = rsqrtf(s[0] / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[row * C + c] = tobf(bf(w[c]) * (bf(e[c]) * r));
}

// The block attention-residual mix (Inkling2BlockAttentionResidual) and the norm after it, per row and query.
struct MixArgs {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, C] each: the completed blocks' inputs
    const void* partial;                 // [rows, C]: bf16 when partial_bf16, else fp32
    const float* query;                  // fp32 [nq, C]; null: no mix, the partial alone (layer 0's attention)
    const uint16_t* norm[kMaxQueries];   // bf16 [C] per query: the RMSNorm after the mix
    uint16_t* out;                       // bf16 [nq, rows, C]
    int n_blocks, partial_bf16, nq, rows, C;
    float eps;
};

constexpr int kMixStage = 4, kMixCols = kMixStage * 512;  // tf_ink2_mix: slices of 512 columns staged at a time

__device__ __forceinline__ void mixStage(const MixArgs& a, size_t row, int c0, uint16_t (*sb)[kMixCols], unsigned char* sp) {
    const int cols = min(kMixCols, a.C - c0);  // a multiple of 8: C % 128 == 0 (ops.zig)
    const int chunks = cols / 8;               // 16 bytes of bf16 a chunk
    for (int j = 0; j < a.n_blocks; ++j)
        for (int i = threadIdx.x; i < chunks; i += blockDim.x)
            __pipeline_memcpy_async(&sb[j][i * 8], a.blocks[j] + row * a.C + c0 + i * 8, 16);
    const int pe = a.partial_bf16 ? 2 : 4;
    const unsigned char* p = static_cast<const unsigned char*>(a.partial) + (row * a.C + c0) * pe;
    for (int i = threadIdx.x; i < cols * pe / 16; i += blockDim.x) __pipeline_memcpy_async(sp + i * 16, p + i * 16, 16);
    __pipeline_commit();  // one memory round trip a slice instead of one a block
    __pipeline_wait_prior(0);
    __syncthreads();
}

// Narrow calls (decode rows): staged slices; the same sums in the same order as tf_ink2_mix_rows.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_mix(const MixArgs a) {
    constexpr int kV = (kMaxBlocks + 1) * (1 + kMaxQueries);
    __shared__ float sh[32 * kV];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    __shared__ __align__(16) uint16_t sb[kMaxBlocks][kMixCols];
    __shared__ __align__(16) unsigned char sp[kMixCols * 4];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1;
    // block j's column c, staged: thread t reads only the columns t + 512 k it adds, in the order the sums always took
    auto entry = [&](int j, int i) -> float {
        if (j < a.n_blocks) return bf(sb[j][i]);
        return a.partial_bf16 ? bf(reinterpret_cast<const uint16_t*>(sp)[i]) : reinterpret_cast<const float*>(sp)[i];
    };
    if (a.query) {
        // every block's sums in one pass and one reduction: each sum adds in the order it did block by block
        float v[kV];
#pragma unroll
        for (int i = 0; i < kV; ++i) v[i] = 0.0f;
        for (int c0 = 0; c0 < C; c0 += kMixCols) {
            mixStage(a, row, c0, sb, sp);
            for (int i = threadIdx.x; i < kMixCols && c0 + i < C; i += blockDim.x) {
                const int c = c0 + i;
                float qc[kMaxQueries];
#pragma unroll
                for (int q = 0; q < kMaxQueries; ++q) qc[q] = q < a.nq ? a.query[static_cast<size_t>(q) * C + c] : 0.0f;
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j) {
                    if (j < n) {
                        const float x = entry(j, i);
                        v[j * (1 + kMaxQueries)] += x * x;
#pragma unroll
                        for (int q = 0; q < kMaxQueries; ++q)
                            if (q < a.nq) v[j * (1 + kMaxQueries) + 1 + q] += x * qc[q];
                    }
                }
            }
            __syncthreads();  // the next slice overwrites the staged one
        }
        blockSums<kV>(v, sh);
        if (threadIdx.x == 0)
            for (int j = 0; j < n; ++j) {
                const float r = rsqrtf(v[j * (1 + kMaxQueries)] / C + 1e-6f);
                rr[j] = r;
                for (int q = 0; q < a.nq; ++q)
                    wts[q][j] = v[j * (1 + kMaxQueries) + 1 + q] * r * rsqrtf(static_cast<float>(C));
            }
        __syncthreads();
        if (threadIdx.x < a.nq) {
            float* l = wts[threadIdx.x];
            float m = -INFINITY, t = 0.0f;
            for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
            for (int j = 0; j < n; ++j) t += expf(l[j] - m);
            for (int j = 0; j < n; ++j) l[j] = expf(l[j] - m) / t;
        }
        __syncthreads();
    }
    // every query's mixed row kept in registers (C <= kMixCache x blockDim, ops.zig): read once, normed, written
    float s[kMaxQueries] = {0.0f, 0.0f, 0.0f};
    float hc[kMaxQueries][kMixCache];
#pragma unroll
    for (int k = 0; k < kMixCache; ++k) {
        const int c = threadIdx.x + k * 512;
        if (k % kMixStage == 0) {
            if (k > 0) __syncthreads();
            if (k * 512 < C) mixStage(a, row, k * 512, sb, sp);
        }
        const int i = threadIdx.x + (k % kMixStage) * 512;
        float h[kMaxQueries] = {0.0f, 0.0f, 0.0f};
        if (c < C) {
            if (a.query) {
#pragma unroll
                for (int j = 0; j < kMaxBlocks + 1; ++j)
                    if (j < n) {
                        const float x = entry(j, i);
#pragma unroll
                        for (int q = 0; q < kMaxQueries; ++q)
                            if (q < a.nq) h[q] += wts[q][j] * (x * rr[j]);
                    }
            } else {
                h[0] = entry(a.n_blocks, i);
            }
        }
#pragma unroll
        for (int q = 0; q < kMaxQueries; ++q) {
            s[q] += h[q] * h[q];  // rows past C add 0.0f * 0.0f: the sums keep their bits
            hc[q][k] = h[q];
        }
    }
    blockSums<kMaxQueries>(s, sh);
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q >= a.nq) break;
        const float r = rsqrtf(s[q] / C + a.eps);
        uint16_t* o = a.out + (static_cast<size_t>(q) * a.rows + row) * C;
#pragma unroll
        for (int k = 0; k < kMixCache; ++k) {
            const int c = threadIdx.x + k * 512;
            if (c < C) o[c] = tobf(bf(a.norm[q][c]) * (hc[q][k] * r));
        }
    }
}

// Wide calls (prompt chunks, wide verifies): the block-by-block mix, which hides its loads behind other rows.
extern "C" __global__ void __launch_bounds__(512) tf_ink2_mix_rows(const MixArgs a) {
    __shared__ float sh[32 * (1 + kMaxQueries)];
    __shared__ float wts[kMaxQueries][kMaxBlocks + 1];
    __shared__ float rr[kMaxBlocks + 1];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1;
    auto entry = [&](int j, int c) -> float {
        if (j < a.n_blocks) return bf(a.blocks[j][row * C + c]);
        return a.partial_bf16 ? bf(static_cast<const uint16_t*>(a.partial)[row * C + c])
                              : static_cast<const float*>(a.partial)[row * C + c];
    };
    if (a.query) {
        for (int j = 0; j < n; ++j) {
            float v[1 + kMaxQueries] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int c = threadIdx.x; c < C; c += blockDim.x) {
                const float x = entry(j, c);
                v[0] += x * x;
                for (int q = 0; q < a.nq; ++q) v[1 + q] += x * a.query[static_cast<size_t>(q) * C + c];
            }
            blockSums<1 + kMaxQueries>(v, sh);
            const float r = rsqrtf(v[0] / C + 1e-6f);
            if (threadIdx.x == 0) {
                rr[j] = r;
                for (int q = 0; q < a.nq; ++q) wts[q][j] = v[1 + q] * r * rsqrtf(static_cast<float>(C));
            }
        }
        __syncthreads();
        if (threadIdx.x < a.nq) {
            float* l = wts[threadIdx.x];
            float m = -INFINITY, t = 0.0f;
            for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
            for (int j = 0; j < n; ++j) t += expf(l[j] - m);
            for (int j = 0; j < n; ++j) l[j] = expf(l[j] - m) / t;
        }
        __syncthreads();
    }
    for (int q = 0; q < a.nq; ++q) {
        float s[1] = {0.0f};
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float h = 0.0f;
            if (a.query)
                for (int j = 0; j < n; ++j) h += wts[q][j] * (entry(j, c) * rr[j]);
            else
                h = entry(a.n_blocks, c);
            s[0] += h * h;
        }
        blockSums<1>(s, sh);
        const float r = rsqrtf(s[0] / C + a.eps);
        uint16_t* o = a.out + (static_cast<size_t>(q) * a.rows + row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float h = 0.0f;
            if (a.query)
                for (int j = 0; j < n; ++j) h += wts[q][j] * (entry(j, c) * rr[j]);
            else
                h = entry(a.n_blocks, c);
            o[c] = tobf(bf(a.norm[q][c]) * (h * r));
        }
    }
}

// out[row, j] = x[row, :] . W[j, :] in fp32 (bf16 inputs), a warp an output; act 1 takes the sigmoid; bf16 or fp32 out.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_rowdot(const uint16_t* __restrict__ x, const uint16_t* __restrict__ W,
                                                                void* __restrict__ out, int J, int C, int out_bf16, int act) {
    const int lane = threadIdx.x & 31, j = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (j >= J) return;
    const size_t row = blockIdx.x;
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

// Router picks per row: top-K of score + bias (ties to the lower id), weights score / sum(scores) * scale (fp32).
extern "C" __global__ void __launch_bounds__(512) tf_ink2_topk(const float* __restrict__ scores, const float* __restrict__ bias,
                                                              int* __restrict__ pick, float* __restrict__ wts, int E, int K,
                                                              float scale) {
    __shared__ float bv[16];
    __shared__ int bi[16];
    __shared__ float chosen[32];
    __shared__ int chosen_id[32];
    const size_t row = blockIdx.x;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float sel[8];
    const int per = (E + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < per && i < 8; ++i) {
        const int e = threadIdx.x + i * blockDim.x;
        sel[i] = e < E ? scores[row * E + e] + bias[e] : -INFINITY;
    }
    for (int k = 0; k < K; ++k) {
        float v = -INFINITY;
        int id = 0x7fffffff;
        for (int i = 0; i < per && i < 8; ++i) {
            const int e = threadIdx.x + i * blockDim.x;
            if (e < E && (sel[i] > v || (sel[i] == v && e < id))) { v = sel[i]; id = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if (lane == 0) { bv[w] = v; bi[w] = id; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = bv[0];
            int mi = bi[0];
            for (int i = 1; i < nw; ++i)
                if (bv[i] > m || (bv[i] == m && bi[i] < mi)) { m = bv[i]; mi = bi[i]; }
            chosen_id[k] = mi;
            chosen[k] = scores[row * E + mi];
        }
        __syncthreads();
        const int got = chosen_id[k];
        if (got % blockDim.x == static_cast<int>(threadIdx.x)) sel[got / blockDim.x] = -INFINITY;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int k = 0; k < K; ++k) t += chosen[k];
        for (int k = 0; k < K; ++k) {
            pick[row * K + k] = chosen_id[k];
            wts[row * K + k] = chosen[k] / (t + 1e-20f) * scale;
        }
    }
}

// tf_ink2_topk with a warp a row (E <= 256, K <= 32): the same picks (largest, lowest id on ties), the same chosen
// scores in pick order, and the same sum and quotients, so the same bits; no block barriers.
extern "C" __global__ void __launch_bounds__(128) tf_ink2_topk_warp(const float* __restrict__ scores, const float* __restrict__ bias,
                                                                   int* __restrict__ pick, float* __restrict__ wts, int E, int K,
                                                                   float scale, int rows) {
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= static_cast<size_t>(rows)) return;
    float sel[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = lane + i * 32;
        sel[i] = e < E ? scores[row * E + e] + bias[e] : -INFINITY;
    }
    float t = 0.0f;
    float chosen = 0.0f;  // lane k keeps pick k's score
    int chosen_id = 0;
    for (int k = 0; k < K; ++k) {
        float v = -INFINITY;
        int id = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = lane + i * 32;
            if (e < E && (sel[i] > v || (sel[i] == v && e < id))) { v = sel[i]; id = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        const float sc = scores[row * E + id];
        if (lane == k) { chosen = sc; chosen_id = id; }
        t += sc;  // every lane adds the picks in pick order: lane 0's t is tf_ink2_topk's
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (lane + i * 32 == id) sel[i] = -INFINITY;
    }
    if (lane < K) {
        pick[row * K + lane] = chosen_id;
        wts[row * K + lane] = chosen / (t + 1e-20f) * scale;
    }
}

// The shared gate/up epilogue with Inkling's softcapped SwiGLU: gcap tanh(g/gcap) sigmoid(g) ucap tanh(u/ucap).
struct GateUpArgs {
    const float* z;      // [(mat * sk + split) * P + p] * N + col
    const int* pick;     // int32 [P]
    const half* svh_g;   // fp16 [E, N]
    const half* svh_u;   // fp16 [E, N]
    const half* suh_d;   // fp16 [E, N]
    half* xd;            // fp16 [P, N]
    int p, n, sk, experts;
    float gate_cap, up_cap;
};

extern "C" __global__ void tf_ink2_gateup(const GateUpArgs a) {
    using namespace tf_exl3;
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = a.pick[p];
    if (e < 0 || e >= a.experts) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < a.sk; ++s) {
            sg += a.z[((size_t)(0 * a.sk + s) * a.p + p) * a.n + n + j];
            su += a.z[((size_t)(1 * a.sk + s) * a.p + p) * a.n + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float g = gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * a.n + n + j]);
        const float u = uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * a.n + n + j]);
        const float act = a.gate_cap * tanhf(g / a.gate_cap) * (1.f / (1.f + expf(-g))) * a.up_cap * tanhf(u / a.up_cap);
        v[j] = act * __half2float(a.suh_d[(size_t)e * a.n + n + j]);
    }
    fwht128(v, lane);
    half* o = a.xd + (size_t)p * a.n + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Short causal conv over a stream's rows (4 taps, last 3 inputs in `state`): static [C][4] or dynamic [4][C] + dyn.
struct ConvArgs {
    const void* x;          // [rows, C]: fp32 (`parts` rank partials, part_stride apart, added in rank order) or bf16
    float* state;           // fp32 [3, C], oldest first
    const uint16_t* w;      // bf16 taps
    const float* dyn;       // fp32 [rows, heads * 4] or null
    const uint16_t* base;   // bf16 [rows, C] (mode 1, null: zero)
    float* resid;           // fp32 [rows, C] (modes 1 and 2)
    uint16_t* out;          // bf16 [rows, C] (modes 0 and 2)
    const int4* segs;       // per stream (blockIdx.y): row0, rows, state slot; null: one stream, all rows
    float* hist;            // fp32 [rows, C]: each row's input as the state keeps it (null: none)
    long long part_stride;
    long long state_stride;  // elements between the streams' states
    int parts, x_bf16, round_in, heads, C, rows, mode;  // mode 0: out = y; 1: resid = base + bf16(y); 2: resid + bf16(y)
};

// Rows in blocks of 32 (z), each from its three prior inputs recomputed (same bits); block 0 owns the state.
extern "C" __global__ void __launch_bounds__(128) tf_ink2_conv(const ConvArgs a) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = a.C, r0 = blockIdx.z * 32, hd = a.dyn ? c / (C / a.heads) : 0;
    const int4 sg = a.segs ? a.segs[blockIdx.y] : make_int4(0, a.rows, 0, 0);
    if (c >= C || r0 >= sg.y) return;
    float* state = a.state + sg.z * a.state_stride;
    const auto input = [&](int t) {
        const size_t at = static_cast<size_t>(t) * C + c;
        float x = a.x_bf16 ? bf(static_cast<const uint16_t*>(a.x)[at]) : static_cast<const float*>(a.x)[at];
        for (int r = 1; r < a.parts; ++r) x += static_cast<const float*>(a.x)[at + r * a.part_stride];
        return a.round_in ? bfr(x) : x;
    };
    const int t0 = sg.x + r0, t1 = sg.x + min(sg.y, r0 + 32), end = sg.x + sg.y;
    float h0 = r0 == 0 ? state[c] : input(t0 - 3), h1 = r0 == 0 ? state[C + c] : input(t0 - 2);
    float h2 = r0 == 0 ? state[2 * C + c] : input(t0 - 1);
    for (int t = t0; t < t1; ++t) {
        const size_t at = static_cast<size_t>(t) * C + c;
        const float x = input(t);
        float y;
        if (a.hist) a.hist[at] = x;
        if (a.dyn) {
            const float* d = a.dyn + (static_cast<size_t>(t) * a.heads + hd) * kConvTaps;
            y = (bf(a.w[c]) + d[0]) * x + (bf(a.w[C + c]) + d[1]) * h2;
            y += (bf(a.w[2 * C + c]) + d[2]) * h1;
            y += (bf(a.w[3 * C + c]) + d[3]) * h0;
            y = (y + x) / 3.0f;
        } else {
            const uint16_t* w = a.w + static_cast<size_t>(c) * kConvTaps;
            y = bf(w[0]) * h0 + bf(w[1]) * h1;
            y += bf(w[2]) * h2;
            y += bf(w[3]) * x;
            y = (y + x) / 2.2360679774997896f;
        }
        if (a.mode == 0) a.out[at] = tobf(y);
        else if (a.mode == 1) a.resid[at] = (a.base ? bf(a.base[at]) : 0.0f) + bfr(y);
        else a.out[at] = tobf(a.resid[at] + bfr(y));
        h0 = h1, h1 = h2, h2 = x;
    }
    if (r0 != 0) return;
    const bool last = t1 == end;  // else the stream's last three inputs, recomputed
    state[c] = last ? h0 : input(end - 3);
    state[C + c] = last ? h1 : input(end - 2);
    state[2 * C + c] = last ? h2 : input(end - 1);
}

// A conv site of every slot: slot 0's state, its C, and byte offsets of its rows in a hist and in a slot's backup.
struct RewindSite {
    float* state;
    long long hist, backup;
    int C, pad;
};

// A window keeps its first n rows: each site's state becomes the last 3 of (backup's 3 inputs, rows' inputs from hist).
extern "C" __global__ void __launch_bounds__(128) tf_ink2_conv_rewind(const RewindSite* __restrict__ sites, const char* backup,
                                                                      const char* hist, int slot, int row0, int n) {
    const RewindSite s = sites[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= s.C) return;
    const int C = s.C;
    const float* b = reinterpret_cast<const float*>(backup + s.backup);
    const float* x = reinterpret_cast<const float*>(hist + s.hist) + static_cast<size_t>(row0) * C;
    float* st = s.state + static_cast<size_t>(slot) * 3 * C;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = n - 3 + k;
        st[k * C + c] = i < 0 ? b[(3 + i) * C + c] : x[static_cast<size_t>(i) * C + c];
    }
}

// The final logit softcap in place: x = tanh(x / cap) * cap.
extern "C" __global__ void tf_ink2_softcap(float* __restrict__ x, long long n, float cap) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

namespace ink2o {

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace ink2o

// Each row's greedy pick over its first n fp32 columns: the largest value, the lowest id among equals, plus base.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_top1(const float* __restrict__ x, long long stride, int n,
                                                                int base, float* __restrict__ val, int* __restrict__ id) {
    using namespace ink2o;
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* r = x + static_cast<size_t>(blockIdx.x) * stride;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (better(r[i], i, bv, bi)) {
            bv = r[i];
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = blockDim.x >> 5;
    bv = lane < nw ? sv[lane] : -INFINITY;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        val[blockIdx.x] = bv;
        id[blockIdx.x] = (bi == 0x7fffffff ? 0 : bi) + base;
    }
}

namespace ink2r {

// sample.cu's keys, unchanged: a draft raced with them lands on the token a keyed draw at that position would take
__device__ __forceinline__ unsigned long long mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float race_uniform(unsigned long long seed, unsigned pos, unsigned id) {
    unsigned long long x = mix(seed + 0x9E3779B97F4A7C15ull);
    x = mix(x ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    x = mix(x ^ static_cast<unsigned long long>(id));
    const float u = (static_cast<float>(static_cast<unsigned>(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
    return fminf(u, __uint_as_float(0x3F7FFFFFu));
}

}  // namespace ink2r

// One fp32 row's Gumbel race at `pos` under sample.cu's keys (ids base + i): the best score and its id.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_race1(const float* __restrict__ x, int n, int base,
                                                                 unsigned long long seed, float inv_t, unsigned pos,
                                                                 float* __restrict__ val, int* __restrict__ id) {
    using namespace ink2o;
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = bfr(x[i]) * inv_t - logf(-logf(ink2r::race_uniform(seed, pos, static_cast<unsigned>(base + i))));
        if (better(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    bv = lane < static_cast<int>(blockDim.x >> 5) ? sv[lane] : -INFINITY;
    bi = lane < static_cast<int>(blockDim.x >> 5) ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        val[0] = bv;
        id[0] = (bi == 0x7fffffff ? 0 : bi) + base;
    }
}

// fp32 logits rows into bf16 whole rows at column col0: the first n columns rounded, the rest of `width` -inf.
extern "C" __global__ void tf_ink2_tobf(const float* __restrict__ x, long long stride, int n, uint16_t* __restrict__ out,
                                        long long out_stride, long long col0, int width) {
    using namespace ink2o;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t row = blockIdx.y;
    out[row * out_stride + col0 + i] = i < n ? tobf(x[row * stride + i]) : static_cast<uint16_t>(0xff80u);
}

// A link step's words (header and ids) written to the device buffer every rank broadcasts from.
struct Ink2Words {
    unsigned w[96];
};

extern "C" __global__ void tf_ink2_put_words(unsigned* __restrict__ dst, int n, const Ink2Words words) {
    const int i = threadIdx.x;
    if (i < n) dst[i] = words.w[i];
}


// ---- Certified lazy head (TF_INKLING2_LAZY_HEAD): a cheap proxy of every logit with a per-row error bound, then the
// exact logits (qmmf's bf16 lane bits, then the softcap) of the tokens the bound cannot rule out; every other logit
// is -inf, so the argmax (and the keyed race) over the row picks what the full head's row would.
namespace ink2z {

constexpr int kExactWarps = 4;

__device__ __forceinline__ float bfz(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

// The packed bf16 lane element of row n, input k (qmmf.zig packBf16: [npad/64][k/64][8][4][32][4]).
__device__ __forceinline__ size_t packedAt(int n, int k, int K) {
    const size_t tg = static_cast<size_t>(n / 64) * (K / 64) + k / 64;
    const int jj = (n % 64) / 8, kt = (k % 64) / 16, q = k % 16, lane = (n % 8) * 4 + (q % 8) / 2, e = (q % 2) + 2 * (q / 8);
    return (((tg * 8 + jj) * 4 + kt) * 32 + lane) * 4 + e;
}

// Floats as unsigned that order as the floats do (for atomicMax).
__device__ __forceinline__ unsigned ord(float f) {
    const unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float unord(unsigned u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

// Row m's 8 columns cols[0..7] (lane g holds column g's) through qmmf's bf16 mma chain from zero, k in order: the
// fp32 sums the lane writes for those columns (sk = 1), returned in lanes 0..3 (columns 2t, 2t + 1).
__device__ __forceinline__ void exact8(const uint16_t* __restrict__ xr, const uint16_t* __restrict__ w, int K, int col,
                                       float (&acc)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
    const uint2* wp = reinterpret_cast<const uint2*>(w);
    const size_t row_tg = static_cast<size_t>(col / 64) * (K / 64);
    const int jj = (col % 64) / 8, plane = (col % 8) * 4 + t;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const uint32_t a0 = g == 0 ? *reinterpret_cast<const uint32_t*>(xr + k0 + 2 * t) : 0u;
        const uint32_t a2 = g == 0 ? *reinterpret_cast<const uint32_t*>(xr + k0 + 2 * t + 8) : 0u;
        const size_t tg = row_tg + k0 / 64;
        const uint2 b = wp[((tg * 8 + jj) * 4 + (k0 % 64) / 16) * 32 + plane];
        qmm_frag::mma(acc, a0, 0u, a2, 0u, b.x, b.y);
    }
}

}  // namespace ink2z

// delta[v] >= || W_v - scale e4m3(W_v inv) ||_2: the FP8 proxy row's error (fp64 sums, rounded up), a warp a row.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_delta(const uint16_t* __restrict__ w, int K, int V, float inv,
                                                                   float scale, float* __restrict__ delta) {
    const int v = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= V) return;
    double s = 0.0;
    for (int k = lane; k < K; k += 32) {
        const float x = ink2z::bfz(w[ink2z::packedAt(v, k, K)]);
        const __nv_fp8_storage_t c = __nv_cvt_float_to_fp8(x * inv, __NV_SATFINITE, __NV_E4M3);
        const float q = __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
        const double d = static_cast<double>(x) - static_cast<double>(q) * static_cast<double>(scale);
        s += d * d;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) delta[v] = static_cast<float>(sqrt(s) * 1.0001 + 1e-7);
}

namespace ink2z {
// A head row's draw: mode 0 greedy (argmax of the fp32 logits), 1 the keyed race of sample.cu at top_p 1, top_k off.
struct Info {
    unsigned long long seed;
    float inv_t, min_log;
    unsigned pos, mode;
};
// sample.cu's keyed noise, bit for bit (uniform, race_uniform), and the race's score noise -log(-log(u)).
__device__ __forceinline__ unsigned long long mixz(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float gumbel(unsigned long long seed, unsigned pos, unsigned id) {
    unsigned long long x = mixz(seed + 0x9E3779B97F4A7C15ull);
    x = mixz(x ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    x = mixz(x ^ static_cast<unsigned long long>(id));
    const float u = fminf((static_cast<float>(static_cast<unsigned>(x >> 40)) + 0.5f) * (1.0f / 16777216.0f),
                          __uint_as_float(0x3F7FFFFFu));
    return -logf(-logf(u));
}
constexpr float kNever = 24.0f;  // sample.cu NEVER
// Bounds of the race value bf16(tanh(z / cap) cap) inv_t for z in [lo, hi] (tanh, bf16 and product roundings widened).
__device__ __forceinline__ void raceBounds(float lo, float hi, float cap, float inv_t, float& vl, float& vh) {
    float cl = tanhf(lo / cap) * cap, ch = tanhf(hi / cap) * cap;
    cl -= 1e-5f * (fabsf(cl) + 1.0f);
    ch += 1e-5f * (fabsf(ch) + 1.0f);
    cl -= fabsf(cl) * (1.0f / 256.0f);
    ch += fabsf(ch) * (1.0f / 256.0f);
    vl = cl * inv_t;
    vh = ch * inv_t;
    vl -= 1e-6f * fabsf(vl);
    vh += 1e-6f * fabsf(vh);
}
__device__ __forceinline__ float gslack(float g) { return 1e-4f * (fabsf(g) + 1.0f); }
}  // namespace ink2z

// Per row: ||x|| (rounded up) and its bounds' running maxima and candidate count reset.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_prep(const uint16_t* __restrict__ x, int ldx, int K,
                                                                  float* __restrict__ xn, unsigned* __restrict__ lg,
                                                                  int* __restrict__ count) {
    __shared__ double sh[8];
    const int m = blockIdx.x;
    double s = 0.0;
    for (int k = threadIdx.x; k < K; k += 256) {
        const double v = ink2z::bfz(x[static_cast<size_t>(m) * ldx + k]);
        s += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int i = 0; i < 8; ++i) t += sh[i];
        xn[m] = static_cast<float>(sqrt(t) * 1.0001);
        for (int j = 0; j < 3; ++j) lg[3 * m + j] = ink2z::ord(-INFINITY);
        count[m] = 0;
    }
}

// Pass 1. Greedy rows: lg[0] = max over v of z~_v - B_v (B_v = (delta_v + eps) ||x|| + abs), the largest logit's
// certified floor. Race rows: lg[0] and lg[1] = the largest race value's floor and ceiling.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_low(const float* __restrict__ zt, const float* __restrict__ delta,
                                                                 const float* __restrict__ xn, int V, float eps, float abs_,
                                                                 unsigned* __restrict__ lg, const ink2z::Info* __restrict__ info,
                                                                 float cap) {
    __shared__ float sh[2][8];
    const int m = blockIdx.y;
    const ink2z::Info in = info[m];
    const float n = xn[m];
    float a = -INFINITY, b = -INFINITY;
    for (int v = blockIdx.x * 256 + threadIdx.x; v < V; v += gridDim.x * 256) {
        const float z = zt[static_cast<size_t>(m) * V + v], B = (delta[v] + eps) * n + abs_;
        if (in.mode == 0) {
            a = fmaxf(a, z - B);
        } else {
            float vl, vh;
            ink2z::raceBounds(z - B, z + B, cap, in.inv_t, vl, vh);
            a = fmaxf(a, vl);
            b = fmaxf(b, vh);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, o));
    }
    if ((threadIdx.x & 31) == 0) {
        sh[0][threadIdx.x >> 5] = a;
        sh[1][threadIdx.x >> 5] = b;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < 8; ++i) {
            a = fmaxf(a, sh[0][i]);
            b = fmaxf(b, sh[1][i]);
        }
        atomicMax(lg + 3 * m, ink2z::ord(a));
        if (in.mode != 0) atomicMax(lg + 3 * m + 1, ink2z::ord(b));
    }
}

// Pass 2, race rows: lg[2] = the race winner's floor, over tokens surely past the draw's cut max(m + min_log, m - 24).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_low2(const float* __restrict__ zt, const float* __restrict__ delta,
                                                                  const float* __restrict__ xn, int V, float eps, float abs_,
                                                                  unsigned* __restrict__ lg, const ink2z::Info* __restrict__ info,
                                                                  float cap) {
    __shared__ float sh[8];
    const int m = blockIdx.y;
    const ink2z::Info in = info[m];
    if (in.mode == 0) return;
    const float n = xn[m], mhi = ink2z::unord(lg[3 * m + 1]);
    const float sure = fmaxf(mhi + in.min_log, mhi - ink2z::kNever);
    float a = -INFINITY;
    for (int v = blockIdx.x * 256 + threadIdx.x; v < V; v += gridDim.x * 256) {
        const float z = zt[static_cast<size_t>(m) * V + v], B = (delta[v] + eps) * n + abs_;
        float vl, vh;
        ink2z::raceBounds(z - B, z + B, cap, in.inv_t, vl, vh);
        if (vl >= sure) {
            const float g = ink2z::gumbel(in.seed, in.pos, v);
            a = fmaxf(a, vl + g - ink2z::gslack(g));
        }
    }
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < 8; ++i) a = fmaxf(a, sh[i]);
        atomicMax(lg + 3 * m + 2, ink2z::ord(a));
    }
}

// Pass 3, the candidates (unordered; a count past C: overflow). Greedy: z~ + B reaches the floor. Race: the token may
// win (its score's ceiling reaches the winner's floor) or may be the row's largest value (the draw's cut needs it).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_pick(const float* __restrict__ zt, const float* __restrict__ delta,
                                                                  const float* __restrict__ xn, int V, float eps, float abs_,
                                                                  const unsigned* __restrict__ lg, int* __restrict__ cand,
                                                                  int C, int* __restrict__ count,
                                                                  const ink2z::Info* __restrict__ info, float cap) {
    const int m = blockIdx.y;
    const ink2z::Info in = info[m];
    const float n = xn[m], l0 = ink2z::unord(lg[3 * m]), l2 = ink2z::unord(lg[3 * m + 2]);
    for (int v = blockIdx.x * 256 + threadIdx.x; v < V; v += gridDim.x * 256) {
        const float z = zt[static_cast<size_t>(m) * V + v], B = (delta[v] + eps) * n + abs_;
        bool keep;
        if (in.mode == 0) {
            keep = z + B >= l0;
        } else {
            float vl, vh;
            ink2z::raceBounds(z - B, z + B, cap, in.inv_t, vl, vh);
            keep = vh >= l0;
            if (!keep) {
                const float g = ink2z::gumbel(in.seed, in.pos, v);
                keep = vh + g + ink2z::gslack(g) >= l2;
            }
        }
        if (keep) {
            const int i = atomicAdd(count + m, 1);
            if (i < C) cand[static_cast<size_t>(m) * C + i] = v;
        }
    }
}

// Every logit of the rows -inf (after the picks read the proxy).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_fill(float* __restrict__ logits, long long n) {
    for (long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * 256)
        logits[i] = -INFINITY;
}

// The candidates' exact logits: the lane's fp32 sum times its scale, then tanh(x / cap) * cap (tf_ink2_softcap).
extern "C" __global__ void __launch_bounds__(128) tf_ink2_lazy_exact(const uint16_t* __restrict__ x, int ldx,
                                                                   const uint16_t* __restrict__ w, int K, int V, float scale,
                                                                   const int* __restrict__ cand, int C,
                                                                   const int* __restrict__ count, float* __restrict__ logits,
                                                                   float cap) {
    const int m = blockIdx.y, n = count[m];
    if (n > C) return;  // overflow: tf_ink2_lazy_full computes the row
    const int base = (blockIdx.x * ink2z::kExactWarps + (threadIdx.x >> 5)) * 8;
    if (base >= n) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int* cr = cand + static_cast<size_t>(m) * C;
    const int col = cr[min(base + g, n - 1)];
    float acc[4];
    ink2z::exact8(x + static_cast<size_t>(m) * ldx, w, K, col, acc);
    if (g == 0) {
        for (int h = 0; h < 2; ++h) {
            const int j = base + 2 * t + h;
            if (j < n) {
                const float y = acc[h] * scale;
                logits[static_cast<size_t>(m) * V + cr[j]] = tanhf(y / cap) * cap;
            }
        }
    }
}

// Rows whose candidates overflowed: every logit exact, 8 columns a warp (blocks exit at once when none overflowed).
extern "C" __global__ void __launch_bounds__(128) tf_ink2_lazy_full(const uint16_t* __restrict__ x, int ldx,
                                                                  const uint16_t* __restrict__ w, int K, int V, float scale,
                                                                  int C, const int* __restrict__ count, int M,
                                                                  float* __restrict__ logits, float cap) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    for (int m = 0; m < M; ++m) {
        if (count[m] <= C) continue;
        for (int base = (blockIdx.x * ink2z::kExactWarps + (threadIdx.x >> 5)) * 8; base < V;
             base += gridDim.x * ink2z::kExactWarps * 8) {
            float acc[4];
            ink2z::exact8(x + static_cast<size_t>(m) * ldx, w, K, min(base + g, V - 1), acc);
            if (g == 0)
                for (int h = 0; h < 2; ++h) {
                    const int v = base + 2 * t + h;
                    if (v < V) logits[static_cast<size_t>(m) * V + v] = tanhf((acc[h] * scale) / cap) * cap;
                }
        }
    }
}

// The fused projection's rows [rows][ld] split into each part's own [rows][w] plane (16-byte copies; offsets and
// widths are multiples of 8): the n > 1 forwards' one GEMM in place of six (TF_INKLING2_FUSED_ROWS).
struct SplitCols {
    uint16_t* dst[6];
    int off[6];
    int w[6];
    int n;
};
extern "C" __global__ void __launch_bounds__(256) tf_ink2_split_cols(const uint16_t* __restrict__ src, int ld,
                                                                    const SplitCols a) {
    const size_t row = blockIdx.x;
    for (int p = 0; p < a.n; ++p) {
        const uint4* in = reinterpret_cast<const uint4*>(src + row * ld + a.off[p]);
        uint4* out = reinterpret_cast<uint4*>(a.dst[p] + row * a.w[p]);
        for (int i = threadIdx.x; i < a.w[p] / 8; i += 256) out[i] = in[i];
    }
}

// Structured output, greedy: row r's argmax over the tokens its grammar bits allow (bit t % 32 of word t / 32,
// W words a row, tokens past V ignored). The first maximum wins, as the unmasked argmax orders ties; a row with no
// allowed token holding a finite or +inf score writes 0xFFFFFFFF. The logits are read only.
template <class T> __device__ __forceinline__ float tf_ink2_score(const T* x, unsigned t);
template <> __device__ __forceinline__ float tf_ink2_score<float>(const float* x, unsigned t) { return x[t]; }
template <> __device__ __forceinline__ float tf_ink2_score<__nv_bfloat16>(const __nv_bfloat16* x, unsigned t) { return __bfloat162float(x[t]); }

template <class T> __device__ __forceinline__ void tf_ink2_masked_argmax_rows(const T* L, unsigned V, const unsigned* BITS, unsigned W, unsigned* OUT) {
    const unsigned row = blockIdx.x;
    const T* x = L + static_cast<size_t>(row) * V;
    const unsigned* bits = BITS + static_cast<size_t>(row) * W;
    float best = 0.0f;
    unsigned at = 0xFFFFFFFFu;
    for (unsigned t = threadIdx.x; t < V; t += blockDim.x) {
        if (((bits[t >> 5] >> (t & 31u)) & 1u) == 0u) continue;
        const float v = tf_ink2_score<T>(x, t);
        if (isnan(v) || v == -INFINITY) continue;
        if (at == 0xFFFFFFFFu || v > best) { best = v; at = t; } // t only grows within a thread: ties keep the first
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float bv = __shfl_down_sync(0xFFFFFFFFu, best, o);
        const unsigned ba = __shfl_down_sync(0xFFFFFFFFu, at, o);
        if (ba != 0xFFFFFFFFu && (at == 0xFFFFFFFFu || bv > best || (bv == best && ba < at))) { best = bv; at = ba; }
    }
    __shared__ float sv[32];
    __shared__ unsigned sa[32];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = best; sa[warp] = at; }
    __syncthreads();
    if (warp == 0) {
        const unsigned nw = (blockDim.x + 31u) >> 5;
        best = lane < nw ? sv[lane] : 0.0f;
        at = lane < nw ? sa[lane] : 0xFFFFFFFFu;
        for (int o = 16; o > 0; o >>= 1) {
            const float bv = __shfl_down_sync(0xFFFFFFFFu, best, o);
            const unsigned ba = __shfl_down_sync(0xFFFFFFFFu, at, o);
            if (ba != 0xFFFFFFFFu && (at == 0xFFFFFFFFu || bv > best || (bv == best && ba < at))) { best = bv; at = ba; }
        }
        if (lane == 0) OUT[row] = at;
    }
}

extern "C" __global__ void __launch_bounds__(1024) tf_ink2_masked_argmax(const float* L, unsigned V, const unsigned* BITS, unsigned W, unsigned* OUT) {
    tf_ink2_masked_argmax_rows<float>(L, V, BITS, W, OUT);
}

// The same over the bf16 rows a sampled draw reads: the guard judges exactly the scores the sampler sees.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_masked_argmax16(const __nv_bfloat16* L, unsigned V, const unsigned* BITS, unsigned W, unsigned* OUT) {
    tf_ink2_masked_argmax_rows<__nv_bfloat16>(L, V, BITS, W, OUT);
}

// Structured output, sampled: `guard` holds each row's tf_ink2_masked_argmax16 (0xFFFFFFFF: no allowed token has a
// usable score). Before the draw (after = 0) such a row's bf16 logits get one finite entry, so the sampler never sees
// an all -inf row. After it (after = 1) a row's drawn token becomes 0xFFFFFFFF when the row had nothing usable or the
// token is not one its bits allow (W words a row), which the host refuses.
extern "C" __global__ void tf_ink2_grammar_guard(__nv_bfloat16* L, unsigned V, const unsigned* guard, unsigned* tok, unsigned rows, unsigned after, const unsigned* BITS, unsigned W) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    if (!after) {
        if (guard[r] == 0xFFFFFFFFu) L[static_cast<size_t>(r) * V] = __float2bfloat16(0.0f);
        return;
    }
    const unsigned t = tok[r];
    const bool allowed = t < V && ((BITS[static_cast<size_t>(r) * W + (t >> 5)] >> (t & 31u)) & 1u);
    if (guard[r] == 0xFFFFFFFFu || !allowed) tok[r] = 0xFFFFFFFFu;
}

// Structured output, sampled: a masked row's NaN logits become -inf and its finite ones stay at or above -1e30, so
// temperature scaling cannot turn an allowed score into -inf; the guard then judges exactly these scores. Only
// grammar rows run this; unconstrained draws are unchanged.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_grammar_scrub(__nv_bfloat16* L, unsigned V) {
    __nv_bfloat16* x = L + static_cast<size_t>(blockIdx.x) * V;
    const __nv_bfloat16 floor = __float2bfloat16(-1e30f);
    for (unsigned t = threadIdx.x; t < V; t += blockDim.x) {
        const __nv_bfloat16 v = x[t];
        if (__hisnan(v)) x[t] = __ushort_as_bfloat16(static_cast<unsigned short>(0xFF80u));
        else if (!__hisinf(v) && __hlt(v, floor)) x[t] = floor;
    }
}

// Vision embeddings: row 0 = bf16(cls_emb + cls_pos); row 1+p = bf16(patch_emb[p] + bf16(bilinear(pos_tab, (r,c)))).
// Patch index p is in spatial-merge-block order (block_row, block_col, in_row, in_col) over (gh, gw).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_vis_embed(
    const __nv_bfloat16* __restrict__ patch_emb,
    const __nv_bfloat16* __restrict__ pos_tab,
    const __nv_bfloat16* __restrict__ cls_emb,
    const __nv_bfloat16* __restrict__ cls_pos,
    __nv_bfloat16* __restrict__ out,
    int gh, int gw, int merge, int grid, int H)
{
    const int row_idx = blockIdx.x;
    __nv_bfloat16* dst = out + static_cast<size_t>(row_idx) * H;
    if (row_idx == 0) {
        for (int c = threadIdx.x; c < H; c += blockDim.x) {
            dst[c] = __float2bfloat16(__bfloat162float(cls_emb[c]) + __bfloat162float(cls_pos[c]));
        }
        return;
    }
    const int p = row_idx - 1;
    const int blocks_w = gw / merge;
    const int m2 = merge * merge;
    const int in_col = p % merge;
    const int in_row = (p / merge) % merge;
    const int block_col = (p / m2) % blocks_w;
    const int block_row = p / (m2 * blocks_w);
    const int r = block_row * merge + in_row;
    const int col = block_col * merge + in_col;

    const float src_h = (static_cast<float>(r) + 0.5f) * static_cast<float>(grid) / static_cast<float>(gh) - 0.5f;
    const float src_w = (static_cast<float>(col) + 0.5f) * static_cast<float>(grid) / static_cast<float>(gw) - 0.5f;
    const float fh = floorf(src_h);
    const float fw = floorf(src_w);
    const float dh = src_h - fh;
    const float dw = src_w - fw;
    const int ih = static_cast<int>(fh);
    const int iw = static_cast<int>(fw);
    const int h0 = max(0, min(grid - 1, ih));
    const int h1 = max(0, min(grid - 1, ih + 1));
    const int w0 = max(0, min(grid - 1, iw));
    const int w1 = max(0, min(grid - 1, iw + 1));
    const float wh0 = 1.0f - dh, wh1 = dh;
    const float ww0 = 1.0f - dw, ww1 = dw;
    const float wt00 = wh0 * ww0, wt01 = wh0 * ww1, wt10 = wh1 * ww0, wt11 = wh1 * ww1;

    const __nv_bfloat16* p00 = pos_tab + static_cast<size_t>(h0 * grid + w0) * H;
    const __nv_bfloat16* p01 = pos_tab + static_cast<size_t>(h0 * grid + w1) * H;
    const __nv_bfloat16* p10 = pos_tab + static_cast<size_t>(h1 * grid + w0) * H;
    const __nv_bfloat16* p11 = pos_tab + static_cast<size_t>(h1 * grid + w1) * H;
    const __nv_bfloat16* pe = patch_emb + static_cast<size_t>(p) * H;

    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        const float pos = wt00 * __bfloat162float(p00[c]) + wt01 * __bfloat162float(p01[c])
                        + wt10 * __bfloat162float(p10[c]) + wt11 * __bfloat162float(p11[c]);
        const float pos16 = __bfloat162float(__float2bfloat16(pos));
        dst[c] = __float2bfloat16(__bfloat162float(pe[c]) + pos16);
    }
}

// Pack separate Q, K, V [S, H] into QKV [S, 3*H] with 2D pairwise axial RoPE on Q and K.
// Row 0 (CLS) has (pos_h=0, pos_w=0); row 1+p has (row+1, col+1). Channels [0..hd/2) rotate by pos_w, [hd/2..hd) by pos_h.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_vis_pack_qkv_rope(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ qkv,
    int gh, int gw, int merge, int heads, int hd, float theta)
{
    const int s = blockIdx.x;
    const int H = heads * hd;
    float pos_h = 0.0f, pos_w = 0.0f;
    if (s > 0) {
        const int p = s - 1;
        const int blocks_w = gw / merge;
        const int m2 = merge * merge;
        const int in_col = p % merge;
        const int in_row = (p / merge) % merge;
        const int block_col = (p / m2) % blocks_w;
        const int block_row = p / (m2 * blocks_w);
        pos_h = static_cast<float>(block_row * merge + in_row + 1);
        pos_w = static_cast<float>(block_col * merge + in_col + 1);
    }
    const int pairs_per_head = hd / 2;
    const int half_pairs = hd / 4;
    const float spatial_dim = static_cast<float>(hd / 2);
    const int total_pairs = heads * pairs_per_head;

    const __nv_bfloat16* qs = q + static_cast<size_t>(s) * H;
    const __nv_bfloat16* ks = k + static_cast<size_t>(s) * H;
    const __nv_bfloat16* vs = v + static_cast<size_t>(s) * H;
    __nv_bfloat16* qd = qkv + static_cast<size_t>(s) * (3 * H);
    __nv_bfloat16* kd = qd + H;
    __nv_bfloat16* vd = kd + H;

    for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
        const int pair = idx % pairs_per_head;
        const int c0 = idx * 2;
        const int c1 = c0 + 1;
        const bool is_w = pair < half_pairs;
        const int freq_idx = is_w ? pair : (pair - half_pairs);
        const float inv_f = powf(theta, -static_cast<float>(2 * freq_idx) / spatial_dim);
        const float angle = (is_w ? pos_w : pos_h) * inv_f;
        float sn, cs;
        __sincosf(angle, &sn, &cs);

        const float q0 = __bfloat162float(qs[c0]), q1 = __bfloat162float(qs[c1]);
        qd[c0] = __float2bfloat16(q0 * cs - q1 * sn);
        qd[c1] = __float2bfloat16(q1 * cs + q0 * sn);

        const float k0 = __bfloat162float(ks[c0]), k1 = __bfloat162float(ks[c1]);
        kd[c0] = __float2bfloat16(k0 * cs - k1 * sn);
        kd[c1] = __float2bfloat16(k1 * cs + k0 * sn);

        vd[c0] = vs[c0];
        vd[c1] = vs[c1];
    }
}

// LayerScale residual add: x[r, c] = bf16(x[r, c] + gamma[c] * y[r, c]).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_vis_ls_add(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ gamma,
    int H)
{
    const size_t base = static_cast<size_t>(blockIdx.x) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        const float xv = __bfloat162float(x[base + c]);
        const float yv = __bfloat162float(y[base + c]);
        const float gv = __bfloat162float(gamma[c]);
        x[base + c] = __float2bfloat16(xv + gv * yv);
    }
}

// Splice vision features into the language model's embedded rows: for each row r with map[r] >= 0, copy src[map[r], :] to dst[r, :].
extern "C" __global__ void __launch_bounds__(256) tf_ink2_vis_splice(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ map,
    int H)
{
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    __nv_bfloat16* d = dst + static_cast<size_t>(r) * H;
    const __nv_bfloat16* s = src + static_cast<size_t>(m) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) d[c] = s[c];
}

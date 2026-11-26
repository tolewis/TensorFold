// Inkling 2 Mini's loop (inkling2_loop): the cross-pass message mixes and the rank-r gates on the core's inputs.
// C2 stand-in: the arithmetic #4337's Metal loop.metal runs (H3/C2 are other slices); fp32 sums, fixed orders.
#include <cstdint>
#include <cuda_bf16.h>

namespace ink2l {

constexpr int kMaxBlocks = 8;    // finished blocks a mix reads beside the partial state
constexpr int kMaxQueries = 4;   // queries of one mix: a message's q, k, v and mlp, or an input's 1 or 3
constexpr int kMaxGates = 3;     // gates one launch runs: the attention's q, k and v inputs
constexpr int kMaxRank = 128;    // gate rank (config.max_gate_rank)
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The block's sum of v in one order: warp shuffles, then the warps' partials in warp order; every thread gets it.
__device__ __forceinline__ float blockSum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warpSum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < kWarps; ++i) t += sh[i];
    return t;
}

}  // namespace ink2l

using namespace ink2l;

// An attention-residual mix without the norm: sources are the blocks then the partial; queries attend over them.
struct LoopMixArgs {
    const uint16_t* blocks[kMaxBlocks];  // bf16 [rows, H] each
    const void* partial;                 // [rows, H]: fp32 when partial_f32, else bf16
    const float* query[kMaxQueries];     // fp32 [H] each
    float* out;                          // fp32 [nq][rows, H], planes out_stride elements apart
    long long out_stride;
    int n_blocks, partial_f32, nq, rows, H;
    float eps, scale;  // eps: the sources' rms; scale: H^-0.5 on the scores
};

// Per row: each source's 1/rms, each query's softmax over (q . source) / rms * scale, then the fp32 mixes.
__device__ __forceinline__ void loopMixRow(const LoopMixArgs& a, size_t row);

extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_mix(const LoopMixArgs a) { loopMixRow(a, blockIdx.x); }

// tf_ink2_loop_mix for several layers at once (blockIdx.y): layer y's queries sit query_step bytes past layer 0's,
// its output planes out_step floats past. Each (row, layer) block does exactly what tf_ink2_loop_mix's would.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_mix_layers(const LoopMixArgs a0, long long query_step, long long out_step) {
    LoopMixArgs a = a0;
    for (int q = 0; q < kMaxQueries; ++q)
        if (a.query[q]) a.query[q] = reinterpret_cast<const float*>(reinterpret_cast<const char*>(a.query[q]) + blockIdx.y * query_step);
    a.out = a.out + blockIdx.y * out_step;
    loopMixRow(a, blockIdx.x);
}

__device__ __forceinline__ void loopMixRow(const LoopMixArgs& a, size_t row) {
    __shared__ float sh[kWarps];
    __shared__ float inv[kMaxBlocks + 1];
    __shared__ float coef[kMaxQueries][kMaxBlocks + 1];
    const int H = a.H, S = a.n_blocks + 1;
    auto src = [&](int s, int i) -> float {
        if (s < a.n_blocks) return bf(a.blocks[s][row * H + i]);
        if (a.partial_f32) return static_cast<const float*>(a.partial)[row * H + i];
        return bf(static_cast<const uint16_t*>(a.partial)[row * H + i]);
    };
    for (int s = 0; s < S; ++s) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < H; i += kThreads) {
            const float v = src(s, i);
            acc = fmaf(v, v, acc);
        }
        const float t = blockSum(acc, sh);
        if (threadIdx.x == 0) inv[s] = rsqrtf(t / static_cast<float>(H) + a.eps);
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q) {
        float score[kMaxBlocks + 1];
        for (int s = 0; s < S; ++s) {
            float acc = 0.0f;
            for (int i = threadIdx.x; i < H; i += kThreads) acc = fmaf(a.query[q][i], src(s, i), acc);
            score[s] = blockSum(acc, sh) * inv[s] * a.scale;
        }
        if (threadIdx.x == 0) {
            float m = -INFINITY, l = 0.0f;
            for (int s = 0; s < S; ++s) m = fmaxf(m, score[s]);
            for (int s = 0; s < S; ++s) {
                score[s] = expf(score[s] - m);
                l += score[s];
            }
            for (int s = 0; s < S; ++s) coef[q][s] = (score[s] / l) * inv[s];
        }
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q) {
        float* dst = a.out + q * a.out_stride + row * H;
        for (int i = threadIdx.x; i < H; i += kThreads) {
            float u = 0.0f;
            for (int s = 0; s < S; ++s) u = fmaf(coef[q][s], src(s, i), u);
            dst[i] = u;
        }
    }
}

// The gates of one launch: gate g reads state h[g] (fp32 [rows, H]) and message plane slot[g] of msg.
struct LoopGateArgs {
    const float* h[kMaxGates];
    const float* msg;            // fp32 [planes][rows, H]
    long long msg_stride;        // elements between the message planes
    int slot[kMaxGates];
    const uint16_t* down_w[kMaxGates];   // bf16 [rank][2 H]: [state, message]
    const uint16_t* forget[kMaxGates];   // bf16 [H][rank]
    const uint16_t* gain[kMaxGates];     // bf16 [H][rank]
    const uint16_t* norm[kMaxGates];     // bf16 [H]
    uint16_t* down;              // bf16 [gates][rows][rank]
    uint16_t* out;               // bf16 [gates][rows, H], planes out_stride elements apart
    long long out_stride;
    int rows, H, rank;
    float eps_unit, eps;  // eps_unit: the unweighted norms of state and message; eps: the gated output's norm
    float* ys;            // fp32 [gates][rows, H]: the gated rows before their norm
};

// down[g][row] = bf16(W_down [unit(h) bf16, unit(m) bf16]) for gate blockIdx.y of row blockIdx.x; block z takes
// rank rows [8 z, 8 z + 8), one a warp. Every block forms the same norms in the same order.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_gate_down(const LoopGateArgs a) {
    extern __shared__ float xs[];  // [2 H]: the normed state, then the normed message, each rounded to bf16
    __shared__ float sh[kWarps];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H;
    const float* h = a.h[g] + row * H;
    const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
    float sh_acc = 0.0f, sm_acc = 0.0f;
    for (int i = threadIdx.x; i < H; i += kThreads) {
        sh_acc = fmaf(h[i], h[i], sh_acc);
        sm_acc = fmaf(m[i], m[i], sm_acc);
    }
    const float ih = rsqrtf(blockSum(sh_acc, sh) / static_cast<float>(H) + a.eps_unit);
    const float im = rsqrtf(blockSum(sm_acc, sh) / static_cast<float>(H) + a.eps_unit);
    for (int i = threadIdx.x; i < H; i += kThreads) {
        xs[i] = bf(tobf(h[i] * ih));
        xs[H + i] = bf(tobf(m[i] * im));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r = blockIdx.z * kWarps + w;
    if (r >= a.rank) return;
    const uint16_t* wr = a.down_w[g] + static_cast<size_t>(r) * 2 * H;
    float acc = 0.0f;
    for (int i = lane; i < 2 * H; i += 32) acc = fmaf(bf(wr[i]), xs[i], acc);
    acc = warpSum(acc);
    if (lane == 0) a.down[(static_cast<size_t>(g) * a.rows + row) * a.rank + r] = tobf(acc);
}

// ys[g][row][i] = (1 - tanh(Wf d)) h + tanh(Wg d) m, d the gate's bf16 down row: kLin outputs a block (z).
constexpr int kLin = 128;
extern "C" __global__ void __launch_bounds__(kLin) tf_ink2_loop_gate_lin(const LoopGateArgs a) {
    __shared__ float dn[kMaxRank];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H, R = a.rank;
    if (threadIdx.x < R) dn[threadIdx.x] = bf(a.down[(static_cast<size_t>(g) * a.rows + row) * R + threadIdx.x]);
    if (threadIdx.x + kLin < R) dn[threadIdx.x + kLin] = bf(a.down[(static_cast<size_t>(g) * a.rows + row) * R + threadIdx.x + kLin]);
    __syncthreads();
    const int i = blockIdx.z * kLin + threadIdx.x;
    if (i >= H) return;
    const uint16_t* wf = a.forget[g] + static_cast<size_t>(i) * R;
    const uint16_t* wg = a.gain[g] + static_cast<size_t>(i) * R;
    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < R; ++r) {
        f = fmaf(bf(wf[r]), dn[r], f);
        gn = fmaf(bf(wg[r]), dn[r], gn);
    }
    const float* h = a.h[g] + row * H;
    const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
    a.ys[(static_cast<size_t>(g) * a.rows + row) * H + i] = (1.0f - tanhf(f)) * h[i] + tanhf(gn) * m[i];
}

// out[g][row] = norm * rms(ys[g][row]): the squares summed in the order a 256-thread block strides them.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_gate_mix(const LoopGateArgs a) {
    __shared__ float sh[kWarps];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H;
    const float* ys = a.ys + (static_cast<size_t>(g) * a.rows + row) * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += kThreads) ss = fmaf(ys[i], ys[i], ss);
    const float inv = rsqrtf(blockSum(ss, sh) / static_cast<float>(H) + a.eps);
    uint16_t* dst = a.out + g * a.out_stride + row * H;
    for (int i = threadIdx.x; i < H; i += kThreads) dst[i] = tobf(bf(a.norm[g][i]) * (ys[i] * inv));
}

// ---- Row-tiled, coalesced gates (bitwise the same as gate_down / gate_lin above: every output keeps its order). ----

// The normed, bf16-rounded [state, message] inputs of gate_down, once per (row, gate): xs[g][row][2H] bf16 into `xs`
// (the ys scratch, free until gate_lin writes it). Same 256-thread strides and block sums as gate_down.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_gate_prep(const LoopGateArgs a, uint16_t* xs) {
    __shared__ float sh[kWarps];
    const size_t row = blockIdx.x;
    const int g = blockIdx.y, H = a.H;
    const float* h = a.h[g] + row * H;
    const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
    float sh_acc = 0.0f, sm_acc = 0.0f;
    for (int i = threadIdx.x; i < H; i += kThreads) {
        sh_acc = fmaf(h[i], h[i], sh_acc);
        sm_acc = fmaf(m[i], m[i], sm_acc);
    }
    const float ih = rsqrtf(blockSum(sh_acc, sh) / static_cast<float>(H) + a.eps_unit);
    const float im = rsqrtf(blockSum(sm_acc, sh) / static_cast<float>(H) + a.eps_unit);
    uint16_t* dst = xs + (static_cast<size_t>(g) * a.rows + row) * 2 * H;
    for (int i = threadIdx.x; i < H; i += kThreads) {
        dst[i] = tobf(h[i] * ih);
        dst[H + i] = tobf(m[i] * im);
    }
}

// down for a tile of kDR rows: warp w owns rank rows [8 w, 8 w + 8); lane l accumulates i = l + 32 k over 2H in ascending
// order as gate_down's warps do (chunks of kDC staged in shared memory with 16-byte loads), then the same XOR tree.
constexpr int kDR = 8;
constexpr int kDK = 8;
constexpr int kDC = 256;
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2_loop_gate_down_rows(const LoopGateArgs a, const uint16_t* xs) {
    __shared__ __align__(16) uint16_t sx[kDR][kDC];
    __shared__ __align__(16) uint16_t sw[kWarps * kDK][kDC];
    const int g = blockIdx.y, H = a.H, R = a.rank, H2 = 2 * H;
    const size_t row0 = static_cast<size_t>(blockIdx.x) * kDR;
    const int nr = min(kDR, a.rows - static_cast<int>(row0));
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int rb = blockIdx.z * kWarps * kDK;  // the block's first rank
    const int nrk = min(kWarps * kDK, R - rb);
    float acc[kDK][kDR];
#pragma unroll
    for (int k = 0; k < kDK; ++k)
#pragma unroll
        for (int j = 0; j < kDR; ++j) acc[k][j] = 0.0f;
    const uint16_t* x0 = xs + (static_cast<size_t>(g) * a.rows + row0) * H2;
    const uint16_t* w0 = a.down_w[g] + static_cast<size_t>(rb) * H2;
    constexpr int V = kDC / 8;  // uint4 per staged row
    for (int c0 = 0; c0 < H2; c0 += kDC) {
        __syncthreads();
        for (int e = threadIdx.x; e < kDR * V; e += kThreads) {
            const int j = e / V, v = e - j * V;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (j < nr) val = *reinterpret_cast<const uint4*>(x0 + static_cast<size_t>(j) * H2 + c0 + v * 8);
            *reinterpret_cast<uint4*>(&sx[j][v * 8]) = val;
        }
        for (int e = threadIdx.x; e < kWarps * kDK * V; e += kThreads) {
            const int k = e / V, v = e - k * V;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (k < nrk) val = *reinterpret_cast<const uint4*>(w0 + static_cast<size_t>(k) * H2 + c0 + v * 8);
            *reinterpret_cast<uint4*>(&sw[k][v * 8]) = val;
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < kDC / 32; ++c) {
            const int i = lane + 32 * c;
            float xv[kDR];
#pragma unroll
            for (int j = 0; j < kDR; ++j) xv[j] = bf(sx[j][i]);
#pragma unroll
            for (int k = 0; k < kDK; ++k) {
                const float wv = bf(sw[w * kDK + k][i]);
#pragma unroll
                for (int j = 0; j < kDR; ++j) acc[k][j] = fmaf(wv, xv[j], acc[k][j]);
            }
        }
    }
    const int r0 = rb + w * kDK;
#pragma unroll
    for (int k = 0; k < kDK; ++k)
#pragma unroll
        for (int j = 0; j < kDR; ++j) {
            const float v = warpSum(acc[k][j]);
            if (lane == 0 && r0 + k < R && j < nr) a.down[(static_cast<size_t>(g) * a.rows + row0 + j) * R + r0 + k] = tobf(v);
        }
}

// gate_lin for a tile of kLR rows: thread t owns output i0 + t with its forget and gain rows in registers (R = 64 only),
// the tile's down rows in shared memory; each output's ascending-rank chain per row, as gate_lin.
constexpr int kLR = 64;
constexpr int kLRank = 64;
extern "C" __global__ void __launch_bounds__(kLin) tf_ink2_loop_gate_lin_rows(const LoopGateArgs a) {
    __shared__ float dn[kLR][kLRank];
    const int g = blockIdx.y, H = a.H;
    const int i = blockIdx.z * kLin + threadIdx.x;
    const size_t row0 = static_cast<size_t>(blockIdx.x) * kLR;
    const int nr = min(kLR, a.rows - static_cast<int>(row0));
    const uint16_t* dsrc = a.down + (static_cast<size_t>(g) * a.rows + row0) * kLRank;
    for (int e = threadIdx.x; e < nr * kLRank; e += kLin) dn[e / kLRank][e % kLRank] = bf(dsrc[e]);
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
    if (i >= H) return;
    for (int j = 0; j < nr; ++j) {
        float f = 0.0f, gn = 0.0f;
#pragma unroll
        for (int r = 0; r < kLRank; ++r) {
            const uint32_t pf = wf[r >> 1], pg = wg[r >> 1];
            const float fw = (r & 1) ? __uint_as_float(pf & 0xFFFF0000u) : __uint_as_float(pf << 16);
            const float gw = (r & 1) ? __uint_as_float(pg & 0xFFFF0000u) : __uint_as_float(pg << 16);
            f = fmaf(fw, dn[j][r], f);
            gn = fmaf(gw, dn[j][r], gn);
        }
        const size_t row = row0 + j;
        const float* h = a.h[g] + row * H;
        const float* m = a.msg + a.slot[g] * a.msg_stride + row * H;
        a.ys[(static_cast<size_t>(g) * a.rows + row) * H + i] = (1.0f - tanhf(f)) * h[i] + tanhf(gn) * m[i];
    }
}

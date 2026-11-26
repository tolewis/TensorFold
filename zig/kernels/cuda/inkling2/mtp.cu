// Inkling 2 Mini's DSpark drafter (mtp.safetensors) on CUDA, and the target-side taps and conv backups it needs.
// The math follows the Metal Mini drafter (#4337 with #4447's attention scale and FP32 kappa) and Inkling 2
// Small's CUDA DSpark kernels (#4128); widths are Mini's: H 2304, 24 query heads, 8 key heads of 128, rank 256.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "common/reduce.cuh"

namespace ink2m {

constexpr int kD = 128;
constexpr int kRows = 8;   // a proposal block: the pending token's row and seven mask rows
constexpr int kMaxQ = 24;  // a key head's query rows in a block: 8 rows x 3 heads
constexpr int kKT = 32;
constexpr int kChunk = 1024;

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfr(float f) { return bf(tobf(f)); }

// One sum over the block in a fixed order: warp shuffles, then warp 0 over the warps' partials.
__device__ __forceinline__ float blockSum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = tf_common::warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const float t = lane < nw ? sh[lane] : 0.0f;
    const float s = __shfl_sync(0xffffffffu, tf_common::warp_sum(t), 0);
    __syncthreads();
    return s;
}

// RoPE as the Metal drafter: halves (d, d + 64), frequency theta^(-(d % 64) / 64).
__device__ __forceinline__ float angle(int d, int pos, float theta) {
    return static_cast<float>(pos) * powf(theta, -static_cast<float>(d % 64) / 64.0f);
}

}  // namespace ink2m

// ---- target side ----------------------------------------------------------------------------------------------

// dst[row][tap][C] = src[row][C]: one tapped visit's rows into the features' interleaved rows.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_tap(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int C,
                                                               int taps, int tap) {
    const size_t row = blockIdx.x;
    const uint4* s = reinterpret_cast<const uint4*>(src + row * C);
    uint4* d = reinterpret_cast<uint4*>(dst + (row * taps + tap) * C);
    for (int i = threadIdx.x; i < C / 8; i += blockDim.x) d[i] = s[i];
}

// ops.cu's RewindSite: a conv site's slot-0 state, byte offsets of its rows in a hist and in a slot's backup, its C.
struct RewindSite {
    float* state;
    long long hist, backup;
    int C, pad;
};

// Every site's state of `slot` into the slot's backup (what tf_ink2_conv_rewind restores the first rows from).
extern "C" __global__ void __launch_bounds__(128) tf_ink2m_conv_backup(const RewindSite* __restrict__ sites, char* backup, int slot) {
    const RewindSite s = sites[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= s.C) return;
    const float* st = s.state + static_cast<size_t>(slot) * 3 * s.C;
    float* b = reinterpret_cast<float*>(backup + s.backup);
#pragma unroll
    for (int k = 0; k < 3; ++k) b[k * s.C + c] = st[k * s.C + c];
}

// ---- drafter: committed context --------------------------------------------------------------------------------

// c[f][row] = RMSNorm(sum_j bf16(softmax(fuse[f])_j) * taps[row][j]) * dflare[f]; the Markov head's f (fusion 3)
// of the last row also to `newest`. Block (row, fusion).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_fuse(const uint16_t* __restrict__ taps, const uint16_t* __restrict__ fuse,
                                                                const uint16_t* __restrict__ dflare, uint16_t* __restrict__ c,
                                                                uint16_t* __restrict__ newest, int rows, int C, float eps) {
    using namespace ink2m;
    __shared__ float sh[32];
    const int row = blockIdx.x, f = blockIdx.y;
    float alpha[4];
    float peak = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        alpha[j] = bf(fuse[f * 4 + j]);
        peak = fmaxf(peak, alpha[j]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        alpha[j] = expf(alpha[j] - peak);
        sum += alpha[j];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) alpha[j] = bfr(alpha[j] / sum);
    const uint16_t* x = taps + static_cast<size_t>(row) * 4 * C;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < C; d += blockDim.x) {
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) m += alpha[j] * bf(x[j * C + d]);
        ss += m * m;
    }
    const float r = rsqrtf(blockSum(ss, sh) / C + eps);
    uint16_t* out = c + (static_cast<size_t>(f) * rows + row) * C;
    for (int d = threadIdx.x; d < C; d += blockDim.x) {
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) m += alpha[j] * bf(x[j * C + d]);
        const uint16_t o = tobf(m * r * bf(dflare[static_cast<size_t>(f) * C + d]));
        out[d] = o;
        if (f == 3 && row + 1 == rows) newest[d] = o;
    }
}

// Committed rows' context keys and values: k normed per head (k_norm_ctx) and RoPE'd at base + row, into the ring.
// kv [rows][2][kvw] (k then v, a matmul's output), caches [ring][kvw]. Block (row, key head), 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_ink2m_ctx_kv(const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
                                                                  int ld, const uint16_t* __restrict__ w, uint16_t* __restrict__ kc,
                                                                  uint16_t* __restrict__ vc, int kvw, int base, int ring, float eps,
                                                                  float theta) {
    using namespace ink2m;
    __shared__ float sh[32];
    __shared__ float y[kD];
    const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t src = static_cast<size_t>(row) * ld + h * kD + d;
    const float x = bf(k[src]);
    const float r = rsqrtf(blockSum(x * x, sh) / kD + eps);
    y[d] = bfr(x * r * bf(w[d]));
    __syncthreads();
    const int pos = base + row;
    const float a = angle(d, pos, theta), cs = cosf(a), sn = sinf(a);
    const float o = d < 64 ? y[d] * cs - y[d + 64] * sn : y[d - 64] * sn + y[d] * cs;
    const size_t at = static_cast<size_t>(pos % ring) * kvw + h * kD + d;
    kc[at] = tobf(o);
    vc[at] = v[src];
}

// ---- drafter: the proposal block --------------------------------------------------------------------------------

// Row 0: the pending token's embedding (already normed); rows 1..7: mask_embed.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_init_rows(const uint16_t* __restrict__ anchor, const uint16_t* __restrict__ mask,
                                                                     uint16_t* __restrict__ rows, int C) {
    const int row = blockIdx.x;
    const uint16_t* src = row == 0 ? anchor : mask;
    for (int d = threadIdx.x; d < C; d += blockDim.x) rows[static_cast<size_t>(row) * C + d] = src[d];
}

// sigma = x (init) or x + resid; new_resid = bf16(sigma), out = bf16(RMSNorm(sigma) * w). Block a row.
// x may be `out` and resid `new_resid` (in place): no __restrict__, each element read before its own write.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_resnorm(const uint16_t* x, const uint16_t* resid, const uint16_t* __restrict__ w,
                                                                   uint16_t* new_resid, uint16_t* out, int C, int init, float eps) {
    using namespace ink2m;
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < C; d += blockDim.x) {
        const size_t at = row * C + d;
        const float s = init ? bf(x[at]) : bf(x[at]) + bf(resid[at]);
        ss += s * s;
    }
    const float r = rsqrtf(blockSum(ss, sh) / C + eps);
    for (int d = threadIdx.x; d < C; d += blockDim.x) {
        const size_t at = row * C + d;
        const float s = init ? bf(x[at]) : bf(x[at]) + bf(resid[at]);
        new_resid[at] = tobf(s);
        out[at] = tobf(s * r * bf(w[d]));
    }
}

// The block's q (q_norm) to `qo`, its keys (k_norm) and values into the ring at anchor + row, both RoPE'd.
// qkv [rows][ld]: q heads, then key heads, then value heads (one fused matmul's output). Block (row, head):
// heads [0, hq) query, [hq, hq + hk) key; 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_ink2m_block_qkv(const uint16_t* __restrict__ qkv, int ld, const uint16_t* __restrict__ qw,
                                                                     const uint16_t* __restrict__ kw, uint16_t* __restrict__ qo,
                                                                     uint16_t* __restrict__ kc, uint16_t* __restrict__ vc, int hq, int hk,
                                                                     int anchor, int ring, float eps, float theta) {
    using namespace ink2m;
    __shared__ float sh[32];
    __shared__ float y[kD];
    const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const bool isq = h < hq;
    const int j = isq ? h : h - hq;
    const uint16_t* x = qkv + static_cast<size_t>(row) * ld;
    const float v0 = bf(x[h * kD + d]);
    const float r = rsqrtf(blockSum(v0 * v0, sh) / kD + eps);
    y[d] = bfr(v0 * r * bf((isq ? qw : kw)[d]));
    __syncthreads();
    const int pos = anchor + row;
    const float a = angle(d, pos, theta), cs = cosf(a), sn = sinf(a);
    const float o = d < 64 ? y[d] * cs - y[d + 64] * sn : y[d - 64] * sn + y[d] * cs;
    if (isq) {
        qo[(static_cast<size_t>(row) * hq + j) * kD + d] = tobf(o);
        return;
    }
    const size_t at = (static_cast<size_t>(pos % ring) * hk + j) * kD + d;
    kc[at] = tobf(o);
    vc[at] = x[(hq + hk + j) * kD + d];
}

struct AttnArgs {
    const uint16_t* q;  // bf16 [T][hq][128]
    const uint16_t* kc;  // bf16 [ring][hk][128]
    const uint16_t* vc;
    float* po;   // fp32 [chunks][T * hq][128]: unnormalised sums
    float* pml;  // fp32 [chunks][T * hq][2]: max, total
    int T, hq, hk, ring, lo, hi, c0;
    float scale;
};

// A block's rows against keys [lo, hi], unmasked (the block sees itself whole); a 1,024-key chunk and a key head a
// block (Inkling 2 Small's tf_ink2_dsp_attn, #4128, at Mini's 24 query rows a key head).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_attn(AttnArgs a) {
    using namespace ink2m;
    __shared__ float sq[kMaxQ][kD];
    __shared__ uint16_t sk[kKT][kD + 8];
    __shared__ uint16_t sv[kKT][kD];
    __shared__ float sp[kMaxQ][kKT];
    __shared__ float sm[kMaxQ], sl[kMaxQ], sa[kMaxQ];
    const int j = blockIdx.y, c = a.c0 + blockIdx.x, tid = threadIdx.x;
    const int g = a.hq / a.hk, nq = a.T * g, nqt = a.T * a.hq;
    const int k0 = max(a.lo, c * kChunk), k1 = min(a.hi + 1, (c + 1) * kChunk);
    for (int idx = tid; idx < nq * kD; idx += blockDim.x) {
        const int qi = idx / kD, d = idx % kD;
        sq[qi][d] = bf(a.q[(static_cast<size_t>(qi / g) * a.hq + j * g + qi % g) * kD + d]);
    }
    for (int qi = tid; qi < nq; qi += blockDim.x) {
        sm[qi] = -INFINITY;
        sl[qi] = 0.0f;
    }
    const int d = tid & (kD - 1), half = tid >> 7;
    float o[kMaxQ / 2];
#pragma unroll
    for (int r = 0; r < kMaxQ / 2; ++r) o[r] = 0.0f;
    __syncthreads();
    for (int kt = k0; kt < k1; kt += kKT) {
        const int nk = min(kKT, k1 - kt);
        for (int idx = tid; idx < nk * kD; idx += blockDim.x) {
            const int kk = idx / kD, dd = idx % kD;
            const size_t at = (static_cast<size_t>((kt + kk) % a.ring) * a.hk + j) * kD + dd;
            sk[kk][dd] = a.kc[at];
            sv[kk][dd] = a.vc[at];
        }
        __syncthreads();
        for (int idx = tid; idx < nq * kKT; idx += blockDim.x) {
            const int qi = idx / kKT, kk = idx % kKT;
            float s = -INFINITY;
            if (kk < nk) {
                s = 0.0f;
                for (int dd = 0; dd < kD; ++dd) s += sq[qi][dd] * bf(sk[kk][dd]);
                s *= a.scale;
            }
            sp[qi][kk] = s;
        }
        __syncthreads();
        for (int qi = tid; qi < nq; qi += blockDim.x) {
            float mx = sm[qi];
            for (int kk = 0; kk < nk; ++kk) mx = fmaxf(mx, sp[qi][kk]);
            const float al = expf(sm[qi] - mx);
            float sum = 0.0f;
            for (int kk = 0; kk < nk; ++kk) {
                const float p = expf(sp[qi][kk] - mx);
                sp[qi][kk] = p;
                sum += p;
            }
            sl[qi] = sl[qi] * al + sum;
            sm[qi] = mx;
            sa[qi] = al;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kMaxQ / 2; ++r) {
            const int qi = half * (kMaxQ / 2) + r;
            if (qi < nq) {
                float acc = o[r] * sa[qi];
                for (int kk = 0; kk < nk; ++kk) acc += sp[qi][kk] * bf(sv[kk][d]);
                o[r] = acc;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kMaxQ / 2; ++r) {
        const int qi = half * (kMaxQ / 2) + r;
        if (qi >= nq) continue;
        const size_t rh = static_cast<size_t>(blockIdx.x) * nqt + (qi / g) * a.hq + j * g + qi % g;
        a.po[rh * kD + d] = o[r];
        if (d == 0) {
            a.pml[rh * 2] = sm[qi];
            a.pml[rh * 2 + 1] = sl[qi];
        }
    }
}

// Each (row, head)'s chunks folded in order: out bf16 [T][hq][128].
extern "C" __global__ void __launch_bounds__(128) tf_ink2m_merge(const float* __restrict__ po, const float* __restrict__ pml, int chunks,
                                                                 int nqt, uint16_t* __restrict__ out) {
    using namespace ink2m;
    const size_t rh = blockIdx.x;
    const int d = threadIdx.x;
    float mx = -INFINITY;
    for (int c = 0; c < chunks; ++c) mx = fmaxf(mx, pml[(c * nqt + rh) * 2]);
    float l = 0.0f, acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float m = pml[(c * nqt + rh) * 2];
        const float wt = m == -INFINITY ? 0.0f : expf(m - mx);
        l += pml[(c * nqt + rh) * 2 + 1] * wt;
        acc += po[(c * nqt + rh) * kD + d] * wt;
    }
    out[rh * kD + d] = tobf(acc / l);
}

// out = bf16(silu(g) * u) over gate and up halves of one matmul's rows: gu [rows][2][n].
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_swiglu(const uint16_t* __restrict__ g, const uint16_t* __restrict__ u, int ld,
                                                                  uint16_t* __restrict__ out, int n) {
    using namespace ink2m;
    const size_t row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gate = bf(g[row * ld + i]);
    out[row * n + i] = tobf((gate / (1.0f + expf(-gate))) * bf(u[row * ld + i]));
}

// ---- drafter: the Markov head ------------------------------------------------------------------------------------

// The Markov head's context: out fp32 [rows][3R] = bias + rnn_proj[:, 2R:] . [hidden_row, f], a warp an output (#4128).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_kappa(const uint16_t* __restrict__ hidden, const uint16_t* __restrict__ f,
                                                                 const uint16_t* __restrict__ W, const uint16_t* __restrict__ bias,
                                                                 float* __restrict__ out, int rows, int C, int R) {
    using namespace ink2m;
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= 3 * R) return;
    const uint16_t* w = W + static_cast<size_t>(o) * (2 * R + 2 * C) + 2 * R;
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;
    float fsum = 0.0f;
    for (int k = lane; k < 2 * C; k += 32) {
        const float wk = bf(w[k]);
        if (k < C) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                if (r < rows) acc[r] += wk * bf(hidden[static_cast<size_t>(r) * C + k]);
        } else {
            fsum += wk * bf(f[k - C]);
        }
    }
    fsum = tf_common::warp_sum(fsum);
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const float s = tf_common::warp_sum(acc[r]);
        if (lane == 0 && r < rows) out[static_cast<size_t>(r) * 3 * R + o] = s + fsum + bf(bias[o]);
    }
}

// A recurrence step's raw sums: raw[o] = ctx[j][o] + rnn_proj[o, :2R] . [state (0 at j = 0), w1[walk[j]]].
// A warp an output, 8 outputs a block (96 blocks for 3R = 768): wide enough to stream the 786 KB at bandwidth.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_rnn_raw(const float* __restrict__ ctx, const uint16_t* __restrict__ W,
                                                                   const uint16_t* __restrict__ w1, const int* __restrict__ walk, int j,
                                                                   const uint16_t* __restrict__ state, float* __restrict__ raw, int R,
                                                                   int C) {
    using namespace ink2m;
    __shared__ float in[512];
    const int t = threadIdx.x, lane = t & 31;
    const size_t prev = static_cast<size_t>(walk[j]);
    for (int c = t; c < R; c += blockDim.x) {
        in[c] = j == 0 ? 0.0f : bf(state[c]);
        in[R + c] = bf(w1[prev * R + c]);
    }
    __syncthreads();
    const int o = blockIdx.x * (blockDim.x >> 5) + (t >> 5);
    if (o >= 3 * R) return;
    const uint16_t* w = W + static_cast<size_t>(o) * (2 * R + 2 * C);
    float s = 0.0f;
    for (int k = lane * 8; k < 2 * R; k += 32 * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(w + k);
        const uint32_t pw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s += bf(static_cast<uint16_t>(pw[i] & 0xffff)) * in[k + 2 * i];
            s += bf(static_cast<uint16_t>(pw[i] >> 16)) * in[k + 2 * i + 1];
        }
    }
    s = tf_common::warp_sum(s);
    if (lane == 0) raw[o] = s + ctx[static_cast<size_t>(j) * 3 * R + o];
}

// The step's gates from its raw sums (one block, R threads), rounded as the bf16 ops (#4128).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_rnn_gate(const float* __restrict__ raw, int j, uint16_t* __restrict__ state,
                                                                    uint16_t* __restrict__ latent, int R) {
    using namespace ink2m;
    const int c = threadIdx.x;
    if (c >= R) return;
    const float prior = j == 0 ? 0.0f : bf(state[c]);
    const float gate = bfr(1.0f / (1.0f + expf(-bfr(raw[c]))));
    const float cand = bfr(tanhf(bfr(raw[R + c])));
    const float keep = bfr(gate * prior);
    const float upd = bfr(bfr(1.0f - gate) * cand);
    state[c] = tobf(keep + upd);
    latent[c] = tobf(tanhf(bfr(raw[2 * R + c])));
}

// A step's logits: softcap(bf16(base + bf16(beta + phi . latent))), -inf past `valid`; a warp a token (#4128).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_vocab(const uint16_t* __restrict__ base, const uint16_t* __restrict__ latent,
                                                                 const uint16_t* __restrict__ phi, const uint16_t* __restrict__ beta, float cap,
                                                                 int valid, int nv, float* __restrict__ out) {
    using namespace ink2m;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint4 pv = reinterpret_cast<const uint4*>(phi + static_cast<size_t>(v) * 256)[lane];
    const uint4 lv = reinterpret_cast<const uint4*>(latent)[lane];
    const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w}, lw[4] = {lv.x, lv.y, lv.z, lv.w};
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        s += bf(static_cast<uint16_t>(pw[i] & 0xffff)) * bf(static_cast<uint16_t>(lw[i] & 0xffff));
        s += bf(static_cast<uint16_t>(pw[i] >> 16)) * bf(static_cast<uint16_t>(lw[i] >> 16));
    }
    s = tf_common::warp_sum(s);
    if (lane != 0) return;
    const float b = bfr(s + bf(beta[v]));
    const float x = bfr(bf(base[v]) + b);
    out[v] = v < valid ? bfr(tanhf(x / cap) * cap) : -INFINITY;
}

// tf_ink2m_vocab for n streams at once: each warp reads its token's phi row once, then each stream's sum in
// tf_ink2m_vocab's order. Stream s: base + s * base_stride, latent + s * 256, out + s * out_stride.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_vocab_n(const uint16_t* __restrict__ base, long long base_stride,
                                                                   const uint16_t* __restrict__ latent, const uint16_t* __restrict__ phi,
                                                                   const uint16_t* __restrict__ beta, float cap, int valid, int nv,
                                                                   float* __restrict__ out, long long out_stride, int n) {
    using namespace ink2m;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint4 pv = reinterpret_cast<const uint4*>(phi + static_cast<size_t>(v) * 256)[lane];
    const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
    const float bt = bf(beta[v]);
    for (int st = 0; st < n; ++st) {
        const uint4 lv = reinterpret_cast<const uint4*>(latent + static_cast<size_t>(st) * 256)[lane];
        const uint32_t lw[4] = {lv.x, lv.y, lv.z, lv.w};
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s += bf(static_cast<uint16_t>(pw[i] & 0xffff)) * bf(static_cast<uint16_t>(lw[i] & 0xffff));
            s += bf(static_cast<uint16_t>(pw[i] >> 16)) * bf(static_cast<uint16_t>(lw[i] >> 16));
        }
        s = tf_common::warp_sum(s);
        if (lane == 0) {
            const float b = bfr(s + bt);
            const float x = bfr(bf(base[st * base_stride + v]) + b);
            out[st * out_stride + v] = v < valid ? bfr(tanhf(x / cap) * cap) : -INFINITY;
        }
    }
}

namespace ink2m {

__device__ __forceinline__ void better(float& bv, int& bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

// The block's first maximum (value, id), lowest id among equals; valid in thread 0.
__device__ __forceinline__ void blockBest(float& bv, int& bi) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) better(bv, bi, __shfl_down_sync(0xffffffffu, bv, o), __shfl_down_sync(0xffffffffu, bi, o));
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) {
        sv[w] = bv;
        si[w] = bi;
    }
    __syncthreads();
    if (w != 0) return;
    bv = lane < nw ? sv[lane] : -INFINITY;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) better(bv, bi, __shfl_down_sync(0xffffffffu, bv, o), __shfl_down_sync(0xffffffffu, bi, o));
}

}  // namespace ink2m

// Stage 1: each block's best of its 4,096 logits into part[block] (value, id).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_pick_part(const float* __restrict__ x, int n, float2* __restrict__ part) {
    using namespace ink2m;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    const int base = blockIdx.x * 4096;
    for (int i = base + threadIdx.x; i < min(n, base + 4096); i += blockDim.x) better(bv, bi, x[i], i);
    blockBest(bv, bi);
    if (threadIdx.x == 0) part[blockIdx.x] = make_float2(bv, __int_as_float(bi));
}

// Stage 2: walk[j + 1] = the best of the parts (the first maximum overall); 0 when no logit beats -inf.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_pick(const float2* __restrict__ part, int parts, int* __restrict__ walk, int j) {
    using namespace ink2m;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < parts; i += blockDim.x) better(bv, bi, part[i].x, __float_as_int(part[i].y));
    blockBest(bv, bi);
    if (threadIdx.x == 0) walk[j + 1] = bi == 0x7fffffff ? 0 : bi;
}

// ---- load: the folded Markov output head ------------------------------------------------------------------------

// y fp32 [R + 1][V] (U . hidden_out's columns, then U . bias) to phi bf16 [V][R] and beta bf16 [V].
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_fold(const float* __restrict__ y, int V, int R, uint16_t* __restrict__ phi,
                                                                uint16_t* __restrict__ beta) {
    using namespace ink2m;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(V) * (R + 1)) return;
    const int v = static_cast<int>(i / (R + 1)), r = static_cast<int>(i % (R + 1));
    const uint16_t o = tobf(y[static_cast<size_t>(r) * V + v]);
    if (r < R) phi[static_cast<size_t>(v) * R + r] = o;
    else beta[v] = o;
}

// An FP8 copy of the target's packed BF16 unembedding, for draft logits alone (verification keeps the BF16 head).
// tf_ink2m_amax: the largest |w| of `total` packed bf16 words into *amax (float bits; zero it first).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_amax(const uint16_t* __restrict__ w, long long total,
                                                                unsigned* __restrict__ amax) {
    float m = 0.0f;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(ink2m::bf(w[i])));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}

namespace ink2m {
// The k of byte `b` of `lane` in a 64-input FP8 group's 16 bytes a lane (modelopt_pack.cu kin, fp8.zig kin).
__device__ __forceinline__ int kin8(int lane, int b) {
    const int p = 16 * (b / 4) + 4 * (lane % 4) + b % 4;
    return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
}
}  // namespace ink2m

// Packed BF16 lane [npad/64][k/64][8][4][32][4] into the FP8 lane [npad/64][k/64][8][32][2][8] at e4m3(w * inv).
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_fp8_lane(const uint16_t* __restrict__ w, uint8_t* __restrict__ out,
                                                                    int k, float inv, long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int b = i % 8, h = (i / 8) % 2, lane = (i / 16) % 32, a = (i / 512) % 8;
    const long long tg = i / 4096;
    const int c = h * 32 + ink2m::kin8(lane, b); // the column within the 64-input group
    const int r = a * 8 + lane / 4;               // the row within the 64-row tile
    const int kt = c / 16, q = c % 16, q2 = q % 8;
    const int blane = (r % 8) * 4 + q2 / 2, e = (q / 8) * 2 + q2 % 2;
    const long long src = (((tg * 8 + r / 8) * 4 + kt) * 32 + blane) * 4 + e;
    out[i] = __nv_cvt_float_to_fp8(ink2m::bf(w[src]) * inv, __NV_SATFINITE, __NV_E4M3);
}

// An FP8 phi for the draft walk (draft logits alone): row v's e4m3 codes at amax_v / 448, its scale in scales[v].
namespace ink2m {
__device__ __forceinline__ float f8(uint32_t byte) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte), __NV_E4M3);
    return __half2float(__half(h));
}
// lane's 8 phi entries (row bytes [8 lane, 8 lane + 8)) dotted with its 8 latent entries, unscaled
__device__ __forceinline__ float dot8(uint2 pv, uint4 lv) {
    const uint32_t lw[4] = {lv.x, lv.y, lv.z, lv.w};
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t p = i < 2 ? pv.x : pv.y, sh = 16 * (i % 2);
        s += f8((p >> sh) & 0xff) * bf(static_cast<uint16_t>(lw[i] & 0xffff));
        s += f8((p >> (sh + 8)) & 0xff) * bf(static_cast<uint16_t>(lw[i] >> 16));
    }
    return s;
}
}  // namespace ink2m

extern "C" __global__ void __launch_bounds__(256) tf_ink2m_phi8(const uint16_t* __restrict__ phi, uint8_t* __restrict__ out,
                                                                float* __restrict__ scales, int nv) {
    using namespace ink2m;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint4 pv = reinterpret_cast<const uint4*>(phi + static_cast<size_t>(v) * 256)[lane];
    const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) m = fmaxf(m, fmaxf(fabsf(bf(pw[i] & 0xffff)), fabsf(bf(pw[i] >> 16))));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float scale = m > 0.0f ? m / 448.0f : 1.0f, inv = 1.0f / scale;
    uint32_t q[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = __nv_cvt_float_to_fp8(bf(pw[i] & 0xffff) * inv, __NV_SATFINITE, __NV_E4M3);
        const uint32_t hi = __nv_cvt_float_to_fp8(bf(pw[i] >> 16) * inv, __NV_SATFINITE, __NV_E4M3);
        q[i / 2] |= (lo | (hi << 8)) << (16 * (i % 2));
    }
    reinterpret_cast<uint2*>(out + static_cast<size_t>(v) * 256)[lane] = make_uint2(q[0], q[1]);
    if (lane == 0) scales[v] = scale;
}

namespace ink2m {
// A sampled stream's draft key: sample.cu's seed and 1 / T, and the position its first walk step's pick is drawn at.
struct DraftKey {
    unsigned long long seed;
    float inv_t; // 0: greedy drafts
    unsigned pos;
};
// sample.cu's keyed uniform (mix, uniform, race_uniform), bit for bit
__device__ __forceinline__ unsigned long long kmix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float race_u(unsigned long long seed, unsigned pos, unsigned id) {
    unsigned long long x = kmix(seed + 0x9E3779B97F4A7C15ull);
    x = kmix(x ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    x = kmix(x ^ static_cast<unsigned long long>(id));
    const float u = (static_cast<float>(static_cast<unsigned>(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
    return fminf(u, __uint_as_float(0x3F7FFFFFu));
}
}  // namespace ink2m

// tf_ink2m_vocab_n over the FP8 phi (n = 1 is tf_ink2m_vocab's shape): base + beta + scale_v * phi8_v . latent.
// keys (optional): a sampled stream's logits become its race scores at pos + step, the very noise the target's
// keyed draw adds there, so its argmax pick is the token the target samples when the two heads agree.
extern "C" __global__ void __launch_bounds__(256) tf_ink2m_vocab8_n(const uint16_t* __restrict__ base, long long base_stride,
                                                                    const uint16_t* __restrict__ latent, const uint8_t* __restrict__ phi,
                                                                    const float* __restrict__ scales, const uint16_t* __restrict__ beta,
                                                                    float cap, int valid, int nv, float* __restrict__ out,
                                                                    long long out_stride, int n, const ink2m::DraftKey* __restrict__ keys,
                                                                    int step) {
    using namespace ink2m;
    const int v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint2 pv = reinterpret_cast<const uint2*>(phi + static_cast<size_t>(v) * 256)[lane];
    const float bt = bf(beta[v]), sc = scales[v];
    for (int st = 0; st < n; ++st) {
        const uint4 lv = reinterpret_cast<const uint4*>(latent + static_cast<size_t>(st) * 256)[lane];
        const float s = tf_common::warp_sum(dot8(pv, lv)) * sc;
        if (lane == 0) {
            const float b = bfr(s + bt);
            const float x = bfr(bf(base[st * base_stride + v]) + b);
            float o = v < valid ? bfr(tanhf(x / cap) * cap) : -INFINITY;
            if (keys != nullptr && v < valid) {
                const DraftKey k = keys[st];
                if (k.inv_t > 0.0f) o = o * k.inv_t - logf(-logf(race_u(k.seed, k.pos + static_cast<unsigned>(step), v)));
            }
            out[st * out_stride + v] = o;
        }
    }
}

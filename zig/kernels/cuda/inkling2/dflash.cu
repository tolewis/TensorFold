// Inkling 2 Mini's DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, Apache-2.0) on CUDA: context keys and values into
// per-stream rings, the block's two-tap convs, attention and SwiGLU, the head's top 16 a row and the candidate selector's
// greedy path. After z-lab/dflash (MIT) and GLM-5.3-Flash's CUDA DFlash2 kernels (glm_dflash.cu), at Mini's widths:
// hidden 2304, 32 query heads and 8 key heads of 128, a 2,048-position window, blocks of 8.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int HD = 128;
constexpr int KVH = 8;
constexpr int QH = 32;
constexpr int KVW = KVH * HD;  // 1024
constexpr int WIN = 2048;      // the sliding window (and a ring's slots)
constexpr int BLOCK = 8;
constexpr int GROUP = 16;  // channels sharing one dynamic tap weight
constexpr int CH = 16;     // key chunks of the chunked attention
constexpr int RANK = 256;
constexpr int TOP = 16;

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// A stream's ring: per layer K then V, [WIN][KVW] each, layers one after another.
__device__ __forceinline__ __nv_bfloat16* ringK(uint64_t base, int layer) {
    return reinterpret_cast<__nv_bfloat16*>(base) + (long long)layer * 2 * WIN * KVW;
}
__device__ __forceinline__ __nv_bfloat16* ringV(uint64_t base, int layer) { return ringK(base, layer) + (long long)WIN * KVW; }

// Qwen3's per-head RMSNorm then rotate-half RoPE of one head (64 threads: dims d and d + 64), to out.
__device__ __forceinline__ void normRope(const __nv_bfloat16* x, const __nv_bfloat16* w, float eps, long long pos, float theta,
                                         __nv_bfloat16* out, float* red) {
    const int d = threadIdx.x;  // 0..63
    const float a = bf(x[d]), b = bf(x[d + 64]);
    float s = a * a + b * b;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float rinv = rsqrtf((red[0] + red[1]) / HD + eps);
    __syncthreads();
    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 64]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / HD);
    const float ang = (float)pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 64] = tobf(nb * c + na * sn);
}

__device__ __forceinline__ float blockSum256(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = lane < (int)(blockDim.x >> 5) ? sh[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    return t;
}
}  // namespace

// out = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w) a row of C. grid (R), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_norm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                                __nv_bfloat16* __restrict__ out, int C, float eps) {
    __shared__ float sh[32];
    const long long r = blockIdx.x;
    const __nv_bfloat16* xr = x + r * C;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += bf(xr[c]) * bf(xr[c]);
    const float rinv = rsqrtf(blockSum256(s, sh) / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[r * C + c] = tobf(bf(tobf(bf(xr[c]) * rinv)) * bf(w[c]));
}

// Two-tap dynamic causal conv over a block: y[t] = sum_o (base[o] + dyn[t][which][o][c / 16]) * z[t - o]; add or store.
// dyn rows hold [2][2][C / 16] (input side, then output side; tap 0, then tap 1). grid (R), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_conv(const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ dyn,
                                                                int which, const __nv_bfloat16* __restrict__ base,
                                                                __nv_bfloat16* __restrict__ out, int add, int C) {
    const int r = blockIdx.x, t = r % BLOCK, groups = C / GROUP;
    const __nv_bfloat16* dr = dyn + (long long)r * 4 * groups + which * 2 * groups;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const int g = c / GROUP;
        float y = (bf(base[c]) + bf(dr[g])) * bf(z[(long long)r * C + c]);
        if (t > 0) y += (bf(base[C + c]) + bf(dr[groups + g])) * bf(z[(long long)(r - 1) * C + c]);
        __nv_bfloat16* o = out + (long long)r * C + c;
        *o = add ? tobf(bf(*o) + bf(tobf(y))) : tobf(y);
    }
}

// SwiGLU over [gate | up] rows (width F each): bf16(bf16(silu(g)) * u). grid (R, ceil(F / 1024)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, int F) {
    const int r = blockIdx.x;
    const __nv_bfloat16* g = gu + (long long)r * 2 * F;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024 && c < F; c += blockDim.x) {
        const float x = bf(g[c]);
        const float s = bf(tobf(x / (1.0f + expf(-x))));
        out[(long long)r * F + c] = tobf(s * bf(g[F + c]));
    }
}

// The block's queries (fused qkv, stride ldq) normed and rotated at pos[r] into q, keys into kb, values into vb.
// grid (R, QH + KVH), 64 threads.
extern "C" __global__ void __launch_bounds__(64) tf_ink2d_block_qkv(const __nv_bfloat16* __restrict__ qkv, int ldq,
                                                                    const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
                                                                    const int* __restrict__ pos, float eps, float theta,
                                                                    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb,
                                                                    __nv_bfloat16* __restrict__ vb) {
    __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < QH) {
        normRope(row + h * HD, qn, eps, pos[r], theta, q + (long long)r * QH * HD + h * HD, red);
    } else {
        const int kh = h - QH;
        normRope(row + QH * HD + kh * HD, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * HD, red);
        const __nv_bfloat16* v = row + QH * HD + KVW + kh * HD;
        __nv_bfloat16* vo = vb + (long long)r * KVW + kh * HD;
        vo[threadIdx.x] = v[threadIdx.x];
        vo[threadIdx.x + 64] = v[threadIdx.x + 64];
    }
}

// Context rows' keys and values (k | v) normed, rotated at pos0 + r, into `ring` at slot (pos0 + r) % WIN of `layer`.
// grid (N, KVH), 64 threads.
extern "C" __global__ void __launch_bounds__(64) tf_ink2d_ctx_kv(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn,
                                                                 long long pos0, uint64_t ring, int layer, float eps, float theta) {
    __shared__ float red[2];
    const int r = blockIdx.x, kh = blockIdx.y;
    const long long p = pos0 + r;
    const int slot = (int)(p % WIN);
    const __nv_bfloat16* row = kv + (long long)r * 2 * KVW;
    normRope(row + kh * HD, kn, eps, p, theta, ringK(ring, layer) + (long long)slot * KVW + kh * HD, red);
    __nv_bfloat16* vo = ringV(ring, layer) + (long long)slot * KVW + kh * HD;
    vo[threadIdx.x] = row[KVW + kh * HD + threadIdx.x];
    vo[threadIdx.x + 64] = row[KVW + kh * HD + threadIdx.x + 64];
}

// Each block's 8 rows attend (non-causally) to their ring's window (max(from, p - 2047) .. start - 1) and all 8 block
// keys, split over CH key chunks (running max, sum, fp32 output), merged by tf_ink2d_attn_merge.
// grid (S, KVH, CH), 256 threads: a warp a block row, its 4 query heads of this key head.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_attn_part(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts,
        const long long* __restrict__ froms, int layer, float scale, float* __restrict__ pm, float* __restrict__ pl,
        float* __restrict__ pacc) {
    const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    for (long long n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const long long at = ((long long)r * QH + kh * 4 + h) * CH + c;
        float* pa = pacc + at * HD + lane * 4;
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];  // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

// The chunks' partials of row r, head h into o[r][h] (bf16). grid (R, QH), 128 threads (a thread a dim).
extern "C" __global__ void __launch_bounds__(128) tf_ink2d_attn_merge(const float* __restrict__ pm, const float* __restrict__ pl,
                                                                      const float* __restrict__ pacc, __nv_bfloat16* __restrict__ o) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * QH * HD + h * HD + d] = tobf(a / l);
}

// A head row's 16 largest logits as ids and fp32 values: per-chunk part, then merge; ties keep the lower id.
namespace {
constexpr int CHUNK_COLS = 4096;

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// The block's best (value, index) of sv/si [n] (256 threads), the winner's slot cleared to -inf.
__device__ void takeBest(float* sv, int* si, int n, float* rv, int* ri, float* outv, int* outi) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bs = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (better(sv[j], si[j], bv, bi)) {
            bv = sv[j];
            bi = si[j];
            bs = j;
        }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o), os = __shfl_xor_sync(0xffffffffu, bs, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
            bs = os;
        }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        rv[w] = bv;
        ri[w] = bi;
        ri[8 + w] = bs;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
            if (better(rv[k], ri[k], rv[best], ri[best])) best = k;
        *outv = rv[best];
        *outi = ri[best];
        if (ri[8 + best] >= 0) sv[ri[8 + best]] = -INFINITY;
    }
    __syncthreads();
}
}  // namespace

// grid (R, ceil(cols / 4096)), 256 threads; logits bf16 [R][cols].
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_top16_part(const __nv_bfloat16* __restrict__ logits, int cols,
                                                                      float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = bf(logits[(long long)r * cols + c0 + j]);
        si[j] = c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

// out [R][2 * TOP]: 16 ids, then their values' bits, best first. grid (R), 256 threads; chunks <= 64.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_top16_merge(const float* __restrict__ pv, const int* __restrict__ pi, int chunks,
                                                                       int* __restrict__ out) {
    __shared__ float sv[64 * TOP];
    __shared__ int si[64 * TOP];
    __shared__ float rv[8];
    __shared__ int ri[16];
    __shared__ float ov[TOP];
    __shared__ int oi[TOP];
    const int r = blockIdx.x, n = chunks * TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(long long)r * n + j];
        si[j] = pi[(long long)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, ov + t, oi + t);
    if (threadIdx.x < TOP) {
        out[r * 2 * TOP + threadIdx.x] = oi[threadIdx.x];
        out[r * 2 * TOP + TOP + threadIdx.x] = __float_as_int(ov[threadIdx.x]);
    }
}

// A sampled stream's draft key (mtp.cu's ink2m::DraftKey): sample.cu's seed and 1 / T, and the position of the first draft.
struct DraftKey {
    unsigned long long seed;
    float inv_t;  // 0: greedy drafts
    unsigned pos;
};
namespace {
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
}  // namespace

// The candidate selector's greedy path for stream s: row t's candidate maximizing logit + <pred[prev] * g[t], succ[c]>,
// prev the anchor (ids[s * 8]) then the path so far; walk[0..7] written (the anchor, then the drafts), and after them in
// the walk fp32 [8] each draft's chance (softmax over the 16 at temperature 1 of the soft-capped scores: cap * tanh(score /
// cap), the trained selector's "after_correction" cap).
// keys (optional): a sampled stream's soft-capped scores become race scores (times 1 / T, plus the keyed Gumbel noise the
// target's draw adds at that position), so the path follows the tokens the target samples when the two agree.
// cand [R][2 * TOP] as tf_ink2d_top16_merge writes, g fp32 [R][RANK]; pred, succ bf16 [V][RANK]. grid (S), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_select(const int* __restrict__ ids, const int* __restrict__ cand, const float* __restrict__ g,
                                                                  const __nv_bfloat16* __restrict__ pred,
                                                                  const __nv_bfloat16* __restrict__ succ,
                                                                  const uint64_t* __restrict__ walks, float cap,
                                                                  const DraftKey* __restrict__ keys) {
    __shared__ float pg[RANK];
    __shared__ float sc[TOP];
    __shared__ int prev_s;
    const int s = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int* walk = reinterpret_cast<int*>(walks[s]);
    float* chances = reinterpret_cast<float*>(walk + BLOCK);
    if (tid == 0) {
        prev_s = ids[s * BLOCK];
        walk[0] = prev_s;
    }
    __syncthreads();
    for (int t = 1; t < BLOCK; ++t) {
        const int r = s * BLOCK + t;
        const int prev = prev_s;
        pg[tid] = bf(pred[(long long)prev * RANK + tid]) * g[(long long)r * RANK + tid];
        __syncthreads();
        for (int k = 0; k < 2; ++k) {
            const int c = w * 2 + k;
            const int id = cand[r * 2 * TOP + c];
            const __nv_bfloat16* S = succ + (long long)id * RANK;
            float d = 0.f;
            for (int i = lane; i < RANK; i += 32) d += pg[i] * bf(S[i]);
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            if (lane == 0) sc[c] = __int_as_float(cand[r * 2 * TOP + TOP + c]) + d;
        }
        __syncthreads();
        if (tid == 0) {
            const DraftKey k = keys != nullptr ? keys[s] : DraftKey{0, 0.0f, 0};
            int best = 0;
            if (k.inv_t > 0.0f) {
                float bestv = -INFINITY;
                for (int c = 0; c < TOP; ++c) {
                    const unsigned id = static_cast<unsigned>(cand[r * 2 * TOP + c]);
                    const float v = cap * tanhf(sc[c] / cap) * k.inv_t - logf(-logf(race_u(k.seed, k.pos + static_cast<unsigned>(t - 1), id)));
                    if (v > bestv) {
                        bestv = v;
                        best = c;
                    }
                }
            } else {
                for (int c = 1; c < TOP; ++c)
                    if (sc[c] > sc[best]) best = c;
            }
            const float top = cap * tanhf(sc[best] / cap);
            float z = 0.f;
            for (int c = 0; c < TOP; ++c) z += expf(cap * tanhf(sc[c] / cap) - top);
            const int id = cand[r * 2 * TOP + best];
            walk[t] = id;
            chances[t] = 1.0f / z;
            prev_s = id;
        }
        __syncthreads();
    }
}

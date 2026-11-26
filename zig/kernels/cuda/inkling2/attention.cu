// Inkling 2 attention: q/k/v/rel preparation, the relative-bias table, chunked tensor-core attention and the merge.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/qmm_frag.cuh"
#include "common/cp_async.cuh"

namespace ink2 {

constexpr int kDim = 128;
constexpr int kRel = 16;
constexpr int kExtent = 1024;
constexpr int kKeys = 64;
constexpr int kPad = kDim + 8;
constexpr float kLog2e = 1.4426950408889634f;

// One stream's rows in a call: rows row0..row0+n-1 sit at positions p0..p0+n-1; `off` is its cache's element offset.
struct Seg {
    int row0;
    int n;
    int p0;
    int pad;
    long long off;
};

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

__device__ __forceinline__ uint32_t pack(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// fp8 caches (--kv-cache fp8): e4m3 codes at the bf16 cache's element index, an fp32 scale per (key head, slot).
__device__ __forceinline__ float fp8f(uint32_t byte) {
    __nv_fp8_e4m3 c;
    c.__x = static_cast<__nv_fp8_storage_t>(byte);
    return static_cast<float>(c);
}

// Eight codes times their row's scale, each rounded to bf16 as the bf16 cache would hold it.
__device__ __forceinline__ uint4 deq8(uint2 c, float sc) {
    uint32_t w[4];
    const uint32_t in[2] = {c.x, c.y};
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = in[i >> 1] >> ((i & 1) * 16);
        w[i] = pack(tobf(fp8f(word & 0xff) * sc), tobf(fp8f((word >> 8) & 0xff) * sc));
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ float warpMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// L2 clip of one head vector held 4 values a lane: scale down to `thr` when longer, as Inkling2L2Clip.
__device__ __forceinline__ void clip(float* x, float thr) {
    const float norm = sqrtf(warpSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]));
    const float f = fminf(thr / (norm + 1e-6f), 1.0f);
    for (int i = 0; i < 4; ++i) x[i] *= f;
}

// Per row and head: q = clip(q * q_scale) (* tau), rel = clip(rel * rel_scale) (* tau), k = clip(k), v = rms(v).
// KV8: k and v rows as e4m3 codes of their bf16 values over amax / 448, the scale at (off / 128) + h * ring + slot.
template <bool KV8>
__device__ __forceinline__ void prepRow(const uint16_t* __restrict__ q_raw, const uint16_t* __restrict__ k_raw,
                                        const uint16_t* __restrict__ v_raw, const uint16_t* __restrict__ rel_raw,
                                        const uint16_t* __restrict__ q_scale, const uint16_t* __restrict__ rel_scale,
                                        const int* __restrict__ rpos, const int* __restrict__ rseg,
                                        const Seg* __restrict__ segs, uint16_t* __restrict__ q_out,
                                        uint16_t* __restrict__ rel_out, uint16_t* __restrict__ kc,
                                        uint16_t* __restrict__ vc, int heads, int kv_heads, int ring, int global,
                                        float qk_clip, float rel_clip, float floor_pos, float alpha, float eps,
                                        float* __restrict__ ksc, float* __restrict__ vsc) {

    const int row = blockIdx.x, y = blockIdx.y, lane = threadIdx.x;
    const int pos = rpos[row];
    const float tau = global ? 1.0f + alpha * logf(fmaxf(static_cast<float>(pos + 1) / floor_pos, 1.0f)) : 1.0f;
    float x[4];
    if (y < heads) {
        const uint16_t* src = q_raw + (static_cast<size_t>(row) * heads + y) * kDim;
        for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]) * bf(q_scale[y * kDim + lane * 4 + i]);
        clip(x, qk_clip);
        uint16_t* dst = q_out + (static_cast<size_t>(row) * heads + y) * kDim;
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = global ? tobf(bf(tobf(x[i])) * tau) : tobf(x[i]);
        float r = lane < kRel ? bf(rel_raw[(static_cast<size_t>(row) * heads + y) * kRel + lane]) *
                                    bf(rel_scale[y * kRel + lane])
                              : 0.0f;
        const float norm = sqrtf(warpSum(r * r));
        r *= fminf(rel_clip / (norm + 1e-6f), 1.0f);
        if (lane < kRel) {
            const uint16_t v = global ? tobf(bf(tobf(r)) * tau) : tobf(r);
            rel_out[(static_cast<size_t>(row) * heads + y) * kRel + lane] = v;
        }
        return;
    }
    const bool key = y < heads + kv_heads;
    const int h = key ? y - heads : y - heads - kv_heads;
    const uint16_t* src = (key ? k_raw : v_raw) + (static_cast<size_t>(row) * kv_heads + h) * kDim;
    for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]);
    if (key) {
        clip(x, qk_clip);
    } else {
        const float r = rsqrtf(warpSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / kDim + eps);
        for (int i = 0; i < 4; ++i) x[i] *= r;
    }
    const Seg s = segs[rseg[row]];
    const size_t slot = static_cast<size_t>(h) * ring + pos % ring;
    if constexpr (KV8) {
        float amax = 0.0f;
        for (int i = 0; i < 4; ++i) {
            x[i] = bf(tobf(x[i]));
            amax = fmaxf(amax, fabsf(x[i]));
        }
        amax = warpMax(amax);
        const float sc = amax > 0.0f ? amax / 448.0f : 1.0f;
        uint32_t word = 0;
        for (int i = 0; i < 4; ++i) {
            const __nv_fp8_e4m3 c(x[i] / sc);
            word |= static_cast<uint32_t>(c.__x) << (8 * i);
        }
        uint8_t* dst = reinterpret_cast<uint8_t*>(key ? kc : vc) + s.off + slot * kDim;
        reinterpret_cast<uint32_t*>(dst)[lane] = word;
        if (lane == 0) (key ? ksc : vsc)[s.off / kDim + slot] = sc;
    } else {
        uint16_t* dst = (key ? kc : vc) + s.off + slot * kDim;
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = tobf(x[i]);
    }
}

}  // namespace ink2

using ink2::Seg;

#define TF_INK2_PREP_ARGS                                                                                          \
    const uint16_t *__restrict__ q_raw, const uint16_t *__restrict__ k_raw, const uint16_t *__restrict__ v_raw,    \
        const uint16_t *__restrict__ rel_raw, const uint16_t *__restrict__ q_scale,                                \
        const uint16_t *__restrict__ rel_scale, const int *__restrict__ rpos, const int *__restrict__ rseg,        \
        const Seg *__restrict__ segs, uint16_t *__restrict__ q_out, uint16_t *__restrict__ rel_out,                \
        uint16_t *__restrict__ kc, uint16_t *__restrict__ vc, int heads, int kv_heads, int ring, int global,       \
        float qk_clip, float rel_clip, float floor_pos, float alpha, float eps
#define TF_INK2_PREP_PASS                                                                                          \
    q_raw, k_raw, v_raw, rel_raw, q_scale, rel_scale, rpos, rseg, segs, q_out, rel_out, kc, vc, heads, kv_heads,   \
        ring, global, qk_clip, rel_clip, floor_pos, alpha, eps

extern "C" __global__ void tf_ink2_prep(TF_INK2_PREP_ARGS) {
    ink2::prepRow<false>(TF_INK2_PREP_PASS, nullptr, nullptr);
}

// tf_ink2_prep into fp8 caches with their scales.
extern "C" __global__ void tf_ink2_prep8(TF_INK2_PREP_ARGS, float* __restrict__ ksc, float* __restrict__ vsc) {
    ink2::prepRow<true>(TF_INK2_PREP_PASS, ksc, vsc);
}

// tab[row, head, d] = bf16(rel[row, head, :] . proj[:, d]) for d < 1024, fp32 sums in tap order (rel_logits_proj).
extern "C" __global__ void tf_ink2_rel_table(const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj,
                                             uint16_t* __restrict__ tab) {
    using namespace ink2;
    const size_t rh = blockIdx.x;
    float r[kRel];
    for (int j = 0; j < kRel; ++j) r[j] = bf(rel[rh * kRel + j]);
    // 8 consecutive columns a thread, 16-byte loads and stores; each column's sum as before
    for (int d0 = threadIdx.x * 8; d0 < kExtent; d0 += blockDim.x * 8) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < kRel; ++j) {
            const uint4 pv = *reinterpret_cast<const uint4*>(proj + j * kExtent + d0);
            const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[2 * e] += r[j] * bf(static_cast<uint16_t>(pw[e] & 0xffff));
                acc[2 * e + 1] += r[j] * bf(static_cast<uint16_t>(pw[e] >> 16));
            }
        }
        uint4 o;
        o.x = static_cast<uint32_t>(tobf(acc[0])) | (static_cast<uint32_t>(tobf(acc[1])) << 16);
        o.y = static_cast<uint32_t>(tobf(acc[2])) | (static_cast<uint32_t>(tobf(acc[3])) << 16);
        o.z = static_cast<uint32_t>(tobf(acc[4])) | (static_cast<uint32_t>(tobf(acc[5])) << 16);
        o.w = static_cast<uint32_t>(tobf(acc[6])) | (static_cast<uint32_t>(tobf(acc[7])) << 16);
        *reinterpret_cast<uint4*>(tab + rh * kExtent + d0) = o;
    }
}

namespace ink2 {

// A global chunk's keys fold in sub-chunks of kSub, in a block each (split) or in order in the chunk's: same bits.
constexpr int kSub = 1024;

// A part (m, l) into a running (mm, ll); a, b scale old and new sums (explicit roundings: both kernels fold alike).
__device__ __forceinline__ void fold(float& mm, float& ll, float m, float l, float& a, float& b) {
    const float mn = fmaxf(mm, m);
    a = exp2f(mm - mn);
    b = exp2f(m - mn);
    ll = __fadd_rn(__fmul_rn(ll, a), __fmul_rn(l, b));
    mm = mn;
}

__device__ __forceinline__ float scaled(float o, float a, float p, float b) {
    return __fadd_rn(__fmul_rn(o, a), __fmul_rn(p, b));
}

// 64 (row, head) pairs of a key head over a block's keys (FOLD: a global chunk, split: a sub-chunk, else a chunk).
template <bool FOLD, bool KV8, bool DB = false>
__device__ __forceinline__ void attnTile(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                         const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab,
                                         const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                                         float* __restrict__ part_o, float2* __restrict__ part_ml, int heads,
                                         int kv_heads, int ring, int window, int chunk, int c0, int chunks,
                                         float scale, int split, uint16_t* ks, uint16_t* vs,
                                         const float* __restrict__ ksc, const float* __restrict__ vsc, int zb) {
    const int2 tile = tiles[blockIdx.x];
    const Seg s = segs[tile.x];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = tile.y;
    const int m_last = min(m0 + 63, total - 1);
    const int tile_hi = s.p0 + m_last / group;
    const int tile_lo = window > 0 ? max(0, s.p0 + m0 / group - window + 1) : 0;
    const int subs = chunk / kSub;
    const int kvh = blockIdx.y;
    // sliding layers count chunks from the tile's first key and store them from each row's first chunk
    int c, sub = 0;
    if (split) {
        c = c0 + zb / subs;
        sub = zb % subs;
    } else {
        c = (window > 0 ? tile_lo / chunk : c0) + zb;
    }
    const int b_lo = c * chunk + sub * kSub, b_hi = b_lo + (split ? kSub : chunk) - 1;
    if (b_lo > tile_hi || b_hi < tile_lo) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bool busy = m0 + warp * 16 < total;  // a warp past the tile's pairs only loads keys
    int mr[2], pos[2], lo[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        mr[i] = m0 + warp * 16 + g + i * 8;
        ok[i] = mr[i] < total;
        const int lr = ok[i] ? mr[i] / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        lo[i] = window > 0 ? max(0, pos[i] - window + 1) : 0;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr[i] % group : 0);
    }
    for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < 2; ++i) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(q + prow[i] * kDim + 16 * k + 2 * t);
            qa[k][i] = ok[i] ? src[0] : 0u;
            qa[k][i + 2] = ok[i] ? src[4] : 0u;
        }
    }
    float acc[16][4];
    float m[2], l[2];
    float O[FOLD ? 16 : 1][4];
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    if (FOLD)
        for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * ring * kDim;
    const int step = FOLD ? kSub : b_hi - b_lo + 1;
    for (int s_lo = b_lo; s_lo <= b_hi && s_lo <= tile_hi; s_lo += step) {
        const int s_hi = s_lo + step - 1;
        for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.0f;
        const int k_lo = max(s_lo, tile_lo), k_hi = min(s_hi, tile_hi);
        auto compute = [&](const int kt, const uint16_t* ks, const uint16_t* vs) {
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                const uint16_t* krow = ks + (8 * j + g) * kPad + 2 * t;
                for (int k = 0; k < 8; ++k) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + 16 * k);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16 * k + 8);
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
                }
            }
            float mx[2] = {-INFINITY, -INFINITY};
            for (int j = 0; j < 8; ++j) {
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    const int kp = kt + 8 * j + 2 * t + (e & 1);
                    const bool live = ok[i] && kp >= lo[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                    const int d = pos[i] - kp;
                    const float bias = live && d < kExtent ? bf(tab[prow[i] * kExtent + d]) : 0.0f;
                    sc[j][e] = live ? (sc[j][e] * scale + bias) * kLog2e : -INFINITY;
                    mx[i] = fmaxf(mx[i], sc[j][e]);
                }
            }
            float alpha[2], mu[2];
            for (int i = 0; i < 2; ++i) {
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
                mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
                const float mn = fmaxf(m[i], mx[i]);
                mu[i] = mn == -INFINITY ? 0.0f : mn;
                alpha[i] = exp2f(m[i] - mu[i]);
                m[i] = mn;
                l[i] *= alpha[i];
            }
            uint16_t p[8][4];
            for (int j = 0; j < 8; ++j) {
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[j][e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[j][e]);
                }
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pack(p[2 * u][0], p[2 * u][1]), a1 = pack(p[2 * u][2], p[2 * u][3]);
                const uint32_t a2 = pack(p[2 * u + 1][0], p[2 * u + 1][1]);
                const uint32_t a3 = pack(p[2 * u + 1][2], p[2 * u + 1][3]);
                const uint16_t* v0 = vs + (16 * u + 2 * t) * kPad + g;
                for (int n = 0; n < 16; ++n) {
                    const uint16_t* vp = v0 + 8 * n;
                    const uint32_t b0 = pack(vp[0], vp[kPad]);
                    const uint32_t b1 = pack(vp[8 * kPad], vp[9 * kPad]);
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
                };
        if constexpr (DB) {
            // TF_INKLING2_ATTN_DB: tile it + 1's keys stream in (cp.async, zero outside [k_lo, k_hi]) while tile it is summed;
            // the same tiles, data and sums as the loop below, so the same bits
            const int kt0 = k_lo & ~(kKeys - 1), nt = (k_hi - kt0) / kKeys + 1;
            // bf16: two stages of K/V tiles; KV8: two stages of codes and scales, dequantized into one K/V tile
            uint8_t* stage8 = reinterpret_cast<uint8_t*>(vs + kKeys * kPad);  // KV8 only: after the K/V tile
            constexpr int kCodes = kKeys * kDim;                                 // bytes of a tile's K (or V) codes
            auto issue = [&](int it) {
                const int kt = kt0 + it * kKeys;
                const size_t slot0 = static_cast<size_t>(kt % ring);
                if constexpr (KV8) {
                    uint8_t* st = stage8 + (it & 1) * (2 * kCodes + 2 * kKeys * 4);
                    for (int e = threadIdx.x; e < kKeys * (kDim / 16); e += 128) {
                        const int key = e >> 3, col = (e & 7) * 16;
                        const int kp = kt + key;
                        const bool in = kp >= k_lo && kp <= k_hi;
                        const size_t at = in ? head_base + (slot0 + key) * kDim + col : head_base;
                        tf_common::cp16z(st + key * kDim + col, reinterpret_cast<const uint8_t*>(kc) + at, in);
                        tf_common::cp16z(st + kCodes + key * kDim + col, reinterpret_cast<const uint8_t*>(vc) + at, in);
                    }
                    if (threadIdx.x < 2 * (kKeys / 4)) {  // a row's scale each: 16 rows a 16-byte copy, K then V
                        const int side = threadIdx.x / (kKeys / 4), q4 = threadIdx.x % (kKeys / 4);
                        float* dst = reinterpret_cast<float*>(st + 2 * kCodes) + side * kKeys + q4 * 4;
                        const float* src = (side ? vsc : ksc) + head_base / kDim + slot0 + q4 * 4;
                        bool in = true;
                        for (int u = 0; u < 4; ++u) in = in && kt + q4 * 4 + u >= k_lo && kt + q4 * 4 + u <= k_hi;
                        if (in) {
                            tf_common::cp16z(dst, src, true);
                        } else {
                            for (int u = 0; u < 4; ++u) {
                                const int kp = kt + q4 * 4 + u;
                                dst[u] = kp >= k_lo && kp <= k_hi ? src[u] : 0.0f;
                            }
                        }
                    }
                } else {
                    uint16_t* kd = ks + (it & 1) * kKeys * kPad;
                    uint16_t* vd = vs + (it & 1) * kKeys * kPad;
                    for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += 128) {
                        const int key = e >> 4, col = (e & 15) * 8;
                        const int kp = kt + key;
                        const bool in = kp >= k_lo && kp <= k_hi;
                        const size_t at = in ? head_base + (slot0 + key) * kDim + col : head_base;
                        tf_common::cp16z(kd + key * kPad + col, kc + at, in);
                        tf_common::cp16z(vd + key * kPad + col, vc + at, in);
                    }
                }
                tf_common::cp_commit();
            };
            issue(0);
            for (int it = 0; it < nt; ++it) {
                if (it + 1 < nt) issue(it + 1);
                else tf_common::cp_commit();
                tf_common::cp_wait<1>();
                __syncthreads();
                if constexpr (KV8) {
                    // the staged codes and scales through deq8 into the tile, as the synchronous load writes it
                    const uint8_t* st = stage8 + (it & 1) * (2 * kCodes + 2 * kKeys * 4);
                    const float* scl = reinterpret_cast<const float*>(st + 2 * kCodes);
                    for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += 128) {
                        const int key = e >> 4, col = (e & 15) * 8;
                        *reinterpret_cast<uint4*>(ks + key * kPad + col) =
                            deq8(*reinterpret_cast<const uint2*>(st + key * kDim + col), scl[key]);
                        *reinterpret_cast<uint4*>(vs + key * kPad + col) =
                            deq8(*reinterpret_cast<const uint2*>(st + kCodes + key * kDim + col), scl[kKeys + key]);
                    }
                    __syncthreads();
                    if (busy) compute(kt0 + it * kKeys, ks, vs);
                } else {
                    if (busy) compute(kt0 + it * kKeys, ks + (it & 1) * kKeys * kPad, vs + (it & 1) * kKeys * kPad);
                }
                __syncthreads();  // every warp is past this stage (and the tile) before tile it + 2 refills it
            }
        } else {
            for (int kt = k_lo & ~(kKeys - 1); kt <= k_hi; kt += kKeys) {
                __syncthreads();
                const size_t slot0 = static_cast<size_t>(kt % ring);
                for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += 128) {
                    const int key = e >> 4, col = (e & 15) * 8;
                    const int kp = kt + key;
                    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
                    if (kp >= k_lo && kp <= k_hi) {
                        const size_t at = head_base + (slot0 + key) * kDim + col;
                        if constexpr (KV8) {
                            kv = deq8(*reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(kc) + at), ksc[at / kDim]);
                            vv = deq8(*reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(vc) + at), vsc[at / kDim]);
                        } else {
                            kv = *reinterpret_cast<const uint4*>(kc + at);
                            vv = *reinterpret_cast<const uint4*>(vc + at);
                        }
                    }
                    *reinterpret_cast<uint4*>(ks + key * kPad + col) = kv;
                    *reinterpret_cast<uint4*>(vs + key * kPad + col) = vv;
                }
                __syncthreads();
                if (!busy) continue;
                compute(kt, ks, vs);
            }
        }
        for (int i = 0; i < 2; ++i) {
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        }
        if (FOLD) {
            for (int i = 0; i < 2; ++i) {
                if (!ok[i] || s_lo > pos[i]) continue;  // the merge folds a row's sub-chunks up to its own
                float a, b;
                fold(M[i], L[i], m[i], l[i], a, b);
                for (int n = 0; n < 16; ++n) {
                    O[n][2 * i] = scaled(O[n][2 * i], a, acc[n][2 * i], b);
                    O[n][2 * i + 1] = scaled(O[n][2 * i + 1], a, acc[n][2 * i + 1], b);
                }
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        // a chunk outside the row's keys stores nothing: its slot is a neighbour (row, head)'s (kernels.partialSlot)
        if (!ok[i] || c < lo[i] / chunk || c > pos[i] / chunk || b_lo > pos[i]) continue;
        const size_t at = split ? prow[i] * chunks + (c - c0) * subs + sub
                                : prow[i] * chunks + (c - (window > 0 ? lo[i] / chunk : c0));
        if (t == 0) part_ml[at] = FOLD ? make_float2(M[i], L[i]) : make_float2(m[i], l[i]);
        float* o = part_o + at * kDim + 2 * t;
        for (int n = 0; n < 16; ++n) {
            const float x = FOLD ? O[FOLD ? n : 0][2 * i] : acc[n][2 * i];
            const float y = FOLD ? O[FOLD ? n : 0][2 * i + 1] : acc[n][2 * i + 1];
            *reinterpret_cast<float2*>(o + 8 * n) = make_float2(x, y);
        }
    }
}

}  // namespace ink2

// Sliding layers (a chunk a block) and split global calls (a sub-chunk a block, `chunks` partials a row and head).
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                 const uint16_t* __restrict__ tab, const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                 float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                 int window, int chunk, int c0, int chunks, float scale, int split) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    // split global calls may run fewer z blocks than slots (TF_INKLING2_GLOBAL_ZCAP): each takes every gridDim.z-th
    // sub-chunk in turn, the same per-slot math, so the partials' bits do not depend on the grid
    const int zn = split ? chunks : static_cast<int>(gridDim.z);
    for (int zb = blockIdx.z; zb < zn; zb += gridDim.z)
        attnTile<false, false>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window, chunk, c0,
                               chunks, scale, split, ks, vs, nullptr, nullptr, zb);
}

// tf_ink2_attn over fp8 caches.
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn8(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                  const uint16_t* __restrict__ tab, const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                  float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                  int window, int chunk, int c0, int chunks, float scale, int split, const float* __restrict__ ksc,
                  const float* __restrict__ vsc) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    // split global calls may run fewer z blocks than slots (TF_INKLING2_GLOBAL_ZCAP): each takes every gridDim.z-th
    // sub-chunk in turn, the same per-slot math, so the partials' bits do not depend on the grid
    const int zn = split ? chunks : static_cast<int>(gridDim.z);
    for (int zb = blockIdx.z; zb < zn; zb += gridDim.z)
        attnTile<false, true>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window, chunk, c0,
                               chunks, scale, split, ks, vs, ksc, vsc, zb);
}

// Global layers a chunk a block, its sub-chunks folded in order (prompt calls: no partial a sub-chunk).
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn_fold(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                      const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab, const Seg* __restrict__ segs,
                      const int2* __restrict__ tiles, float* __restrict__ part_o, float2* __restrict__ part_ml,
                      int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    attnTile<true, false>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, 0, chunk, c0, chunks,
                          scale, 0, ks, vs, nullptr, nullptr, static_cast<int>(blockIdx.z));
}

// tf_ink2_attn_fold over fp8 caches.
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn_fold8(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                       const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab, const Seg* __restrict__ segs,
                       const int2* __restrict__ tiles, float* __restrict__ part_o, float2* __restrict__ part_ml,
                       int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale,
                       const float* __restrict__ ksc, const float* __restrict__ vsc) {
    using namespace ink2;
    __shared__ __align__(16) uint16_t ks[kKeys * kPad];
    __shared__ __align__(16) uint16_t vs[kKeys * kPad];
    attnTile<true, true>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, 0, chunk, c0, chunks,
                         scale, 0, ks, vs, ksc, vsc, static_cast<int>(blockIdx.z));
}

// Per (row, head): its chunks folded in order (subs > 1: each from its sub-chunks first), the sink, then the gate.
extern "C" __global__ void tf_ink2_merge(const float* __restrict__ part_o, const float2* __restrict__ part_ml,
                                         const uint16_t* __restrict__ sink, const uint16_t* __restrict__ gate,
                                         const int* __restrict__ rpos, uint16_t* __restrict__ out, int heads,
                                         int window, int chunk, int c0, int chunks, int subs) {
    using namespace ink2;
    const size_t rh = blockIdx.x;
    const int row = static_cast<int>(rh / heads), d = threadIdx.x;
    const int pos = rpos[row];
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    float mm = -INFINITY, ll = 0.0f, o = 0.0f;
    for (int c = lo / chunk; c <= pos / chunk; ++c) {
        float2 ml;
        float po;
        if (subs > 1) {
            float M = -INFINITY, L = 0.0f, O = 0.0f;
            const int last = min(subs - 1, (pos - c * chunk) / kSub);
            for (int u = 0; u <= last; ++u) {
                const size_t at = rh * chunks + (c - c0) * subs + u;
                const float2 sm = part_ml[at];
                float a, b;
                fold(M, L, sm.x, sm.y, a, b);
                O = scaled(O, a, part_o[at * kDim + d], b);
            }
            ml = make_float2(M, L);
            po = O;
        } else {
            const size_t at = rh * chunks + (c - (window > 0 ? lo / chunk : c0));
            ml = part_ml[at];
            po = part_o[at * kDim + d];
        }
        const float mn = fmaxf(mm, ml.x);
        const float a = exp2f(mm - mn), b = exp2f(ml.x - mn);
        ll = ll * a + ml.y * b;
        o = o * a + po * b;
        mm = mn;
    }
    const float sk = bf(sink[rh]) * kLog2e;
    const float mf = fmaxf(mm, sk);
    const float a = exp2f(mm - mf);
    const float y = o * a / (ll * a + exp2f(sk - mf));
    const float gv = bf(gate[rh * kDim + d]);
    out[rh * kDim + d] = tobf(y * (1.0f / (1.0f + expf(-gv))));
}

// tf_ink2_attn with each block's next K/V tile prefetched by cp.async while the current one is summed (bf16 caches,
// TF_INKLING2_ATTN_DB): the same tiles and sums, so the same bits. Two stages of K and V in dynamic shared memory.
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn_db(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                    const uint16_t* __restrict__ tab, const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                    float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                    int window, int chunk, int c0, int chunks, float scale, int split) {
    using namespace ink2;
    extern __shared__ __align__(16) uint16_t dsm[];
    uint16_t* ks = dsm;
    uint16_t* vs = dsm + 2 * kKeys * kPad;
    const int zn = split ? chunks : static_cast<int>(gridDim.z);
    for (int zb = blockIdx.z; zb < zn; zb += gridDim.z)
        attnTile<false, false, true>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window, chunk,
                                     c0, chunks, scale, split, ks, vs, nullptr, nullptr, zb);
}

// tf_ink2_attn8 with each block's next tile's codes and scales prefetched by cp.async while the current tile is
// dequantized and summed (fp8 caches, TF_INKLING2_ATTN8_DB): the same tiles, codes and sums, so the same bits. One K
// and V tile, then two stages of codes and scales, in dynamic shared memory.
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_attn8_db(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                     const uint16_t* __restrict__ tab, const Seg* __restrict__ segs, const int2* __restrict__ tiles,
                     float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                     int window, int chunk, int c0, int chunks, float scale, int split, const float* __restrict__ ksc,
                     const float* __restrict__ vsc) {
    using namespace ink2;
    extern __shared__ __align__(16) uint16_t dsm[];
    uint16_t* ks = dsm;
    uint16_t* vs = dsm + kKeys * kPad;
    // split global calls may run fewer z blocks than slots (TF_INKLING2_GLOBAL_ZCAP): each takes every gridDim.z-th
    // sub-chunk in turn, the same per-slot math, so the partials' bits do not depend on the grid
    const int zn = split ? chunks : static_cast<int>(gridDim.z);
    for (int zb = blockIdx.z; zb < zn; zb += gridDim.z)
        attnTile<false, true, true>(q, kc, vc, tab, segs, tiles, part_o, part_ml, heads, kv_heads, ring, window, chunk,
                                    c0, chunks, scale, split, ks, vs, ksc, vsc, zb);
}

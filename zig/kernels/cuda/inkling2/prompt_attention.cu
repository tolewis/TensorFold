// Inkling 2 global-layer prompt attention: 128 (row, head) pairs share each K/V tile, with the split path's bits.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/qmm_frag.cuh"

namespace ink2p {

constexpr int kDim = 128;
constexpr int kExtent = 1024;
constexpr int kKeys = 64;
constexpr int kSub = 1024;
constexpr int kWarps = 8;
constexpr int kPairs = kWarps * 16;
constexpr int kThreads = kWarps * 32;
constexpr int kStages = 3;
constexpr int kTile = kKeys * kDim;  // halves in a K or V tile: [key][128], 16-byte pieces swizzled by key
constexpr int kStage = 2 * kTile;    // a stage: its K tile, then its V tile
constexpr float kLog2e = 1.4426950408889634f;

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

__device__ __forceinline__ float fp8f(uint32_t byte) {
    __nv_fp8_e4m3 c;
    c.__x = static_cast<__nv_fp8_storage_t>(byte);
    return static_cast<float>(c);
}

// Eight e4m3 codes times their row's scale, each rounded to bf16 (attention.cu's deq8).
__device__ __forceinline__ uint4 deq8(uint2 c, float sc) {
    uint32_t w[4];
    const uint32_t in[2] = {c.x, c.y};
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = in[i >> 1] >> ((i & 1) * 16);
        w[i] = pack(tobf(fp8f(word & 0xff) * sc), tobf(fp8f((word >> 8) & 0xff) * sc));
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// The half offset of a tile's 16-byte piece `c` (dims 8c..8c+7) of key `key`: pieces XOR-swizzled by key & 7.
__device__ __forceinline__ int piece(int key, int c) { return key * kDim + ((c ^ (key & 7)) << 3); }

// The K/V tile load: keys kt..kt+63 of a key head into the stage's bf16 tiles, zeros outside [k_lo, k_hi] (async;
// KV8: fp8 codes dequantized and stored in place, the group committed empty so the stage count stays the same).
template <bool KV8>
__device__ __forceinline__ void loadTile(uint16_t* ks, uint16_t* vs, const uint16_t* __restrict__ kc,
                                         const uint16_t* __restrict__ vc, size_t head_base, int ring, int kt,
                                         int k_lo, int k_hi, const float* __restrict__ ksc,
                                         const float* __restrict__ vsc) {
    const size_t slot0 = static_cast<size_t>(kt % ring);
    for (int e = threadIdx.x; e < kKeys * (kDim / 8); e += kThreads) {
        const int key = e >> 4, col = (e & 15) * 8;
        const int kp = kt + key;
        const bool in = kp >= k_lo && kp <= k_hi;
        const size_t at = in ? head_base + (slot0 + key) * kDim + col : head_base;
        if constexpr (KV8) {
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (in) {
                kv = deq8(*reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(kc) + at), ksc[at / kDim]);
                vv = deq8(*reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(vc) + at), vsc[at / kDim]);
            }
            *reinterpret_cast<uint4*>(ks + piece(key, e & 15)) = kv;
            *reinterpret_cast<uint4*>(vs + piece(key, e & 15)) = vv;
        } else {
            tf_common::cp16z(ks + piece(key, e & 15), kc + at, in);
            tf_common::cp16z(vs + piece(key, e & 15), vc + at, in);
        }
    }
    tf_common::cp_commit();
}

// A block's prompt pairs (tile of kPairs pairs, key head, chunk).
template <bool KV8>
__device__ __forceinline__ void promptTile(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                           const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab,
                                           const Seg* __restrict__ segs, float* __restrict__ part_o,
                                           float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                                           int chunk, int c0, int chunks, float scale, uint16_t* smem,
                                           const float* __restrict__ ksc, const float* __restrict__ vsc) {
    const Seg s = segs[0];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = static_cast<int>(blockIdx.x) * kPairs;
    if (m0 >= total) return;
    const int tile_hi = s.p0 + min(m0 + kPairs - 1, total - 1) / group;
    const int kvh = blockIdx.y;
    const int c = c0 + static_cast<int>(blockIdx.z);
    const int b_lo = c * chunk, b_hi = b_lo + chunk - 1;
    if (b_lo > tile_hi) return;
    const int k_hi = min(b_hi, tile_hi);
    const int n_tiles = (k_hi - b_lo) / kKeys + 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int w0 = m0 + warp * 16;
    const bool busy = w0 < total;
    const bool full = w0 + 15 < total;
    const int w_first = s.p0 + w0 / group, w_last = s.p0 + min(w0 + 15, total - 1) / group;
    int pos[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        const int mr = w0 + g + i * 8;
        ok[i] = mr < total;
        const int lr = ok[i] ? mr / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr % group : 0);
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
    float O[16][4];
    float M[2] = {-INFINITY, -INFINITY}, L[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * ring * kDim;
    // ldmatrix lanes: K rows 8j + (lane & 7), pieces 2k + lane / 8; V rows 16u + lane % 16, pieces n + lane / 16
    const int k_row = lane & 7, k_pc = lane >> 3;
    const int v_row = lane & 15, v_pc = lane >> 4;
    const uint16_t* k16 = kc;
    const uint16_t* v16 = vc;
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < n_tiles) loadTile<KV8>(smem + st * kStage, smem + st * kStage + kTile, k16, v16, head_base,
                                        ring, b_lo + st * kKeys, b_lo, k_hi, ksc, vsc);
        else tf_common::cp_commit();
    }
    for (int it = 0; it < n_tiles; ++it) {
        const int kt = b_lo + it * kKeys;
        const int s_lo = b_lo + (it / (kSub / kKeys)) * kSub, s_hi = s_lo + kSub - 1;
        if (it % (kSub / kKeys) == 0) {
            for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
            m[0] = m[1] = -INFINITY;
            l[0] = l[1] = 0.0f;
        }
        tf_common::cp_wait<kStages - 2>();
        __syncthreads();  // tile `it` is in; every warp is past tile it-1, whose stage the next load takes
        if (it + kStages - 1 < n_tiles) {
            uint16_t* nx = smem + ((it + kStages - 1) % kStages) * kStage;
            loadTile<KV8>(nx, nx + kTile, k16, v16, head_base, ring, kt + (kStages - 1) * kKeys, b_lo, k_hi, ksc, vsc);
        } else {
            tf_common::cp_commit();
        }
        const uint16_t* ks = smem + (it % kStages) * kStage;
        const uint16_t* vs = ks + kTile;
        if (busy && kt <= w_last) {  // a tile past all of a warp's rows is an exact no-op
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                for (int k = 0; k < 8; k += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4(b, tf_common::smem(ks + piece(8 * j + k_row, 2 * k + k_pc)));
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b[0], b[1]);
                    qmm_frag::mma(sc[j], qa[k + 1][0], qa[k + 1][1], qa[k + 1][2], qa[k + 1][3], b[2], b[3]);
                }
            }
            float mx[2] = {-INFINITY, -INFINITY};
            if (full && kt + kKeys - 1 + kExtent <= w_first) {  // every key live, every bias 0
                for (int j = 0; j < 8; ++j) {
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        const float bias = 0.0f;
                        sc[j][e] = (sc[j][e] * scale + bias) * kLog2e;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    for (int e = 0; e < 4; ++e) {
                        const int i = e >> 1;
                        const int kp = kt + 8 * j + 2 * t + (e & 1);
                        const bool live = ok[i] && kp <= pos[i] && kp >= s_lo && kp <= s_hi;
                        const int d = pos[i] - kp;
                        const float bias = live && d < kExtent ? bf(tab[prow[i] * kExtent + d]) : 0.0f;
                        sc[j][e] = live ? (sc[j][e] * scale + bias) * kLog2e : -INFINITY;
                        mx[i] = fmaxf(mx[i], sc[j][e]);
                    }
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
            uint32_t pk[8][2];  // P as the A fragments: pk[j][i] packs keys 8j+2t, +1 of row g + 8i
            for (int j = 0; j < 8; ++j) {
                uint16_t p[4];
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[e]);
                }
                pk[j][0] = pack(p[0], p[1]);
                pk[j][1] = pack(p[2], p[3]);
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pk[2 * u][0], a1 = pk[2 * u][1], a2 = pk[2 * u + 1][0], a3 = pk[2 * u + 1][1];
                for (int n = 0; n < 16; n += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4t(b, tf_common::smem(vs + piece(16 * u + v_row, n + v_pc)));
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b[0], b[1]);
                    qmm_frag::mma(acc[n + 1], a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        if (busy && (it % (kSub / kKeys) == kSub / kKeys - 1 || it == n_tiles - 1)) {
            for (int i = 0; i < 2; ++i) {
                l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
                l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
            }
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
        if (!ok[i] || b_lo > pos[i]) continue;  // the merge reads chunks up to the row's own
        const size_t at = prow[i] * chunks + (c - c0);
        if (t == 0) part_ml[at] = make_float2(M[i], L[i]);
        float* o = part_o + at * kDim + 2 * t;
        for (int n = 0; n < 16; ++n) *reinterpret_cast<float2*>(o + 8 * n) = make_float2(O[n][2 * i], O[n][2 * i + 1]);
    }
}


// A sliding layer's prompt rows: 128 (row, head) pairs a block share each K/V tile; every 256-key chunk of a row's
// window gives tf_ink2_attn's partial bits (the same 64-key tiles, masks and sums: keys a 64-pair tile would load as
// zeros are masked for these rows), folded in tf_ink2_merge's order and roundings, then its sink and gate. No partials.
template <bool KV8, bool PART = false>
__device__ __forceinline__ void slideTile(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                          const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab,
                                          const Seg* __restrict__ segs, const uint16_t* __restrict__ sink,
                                          const uint16_t* __restrict__ gate, uint16_t* __restrict__ out, int heads,
                                          int kv_heads, int ring, int window, int chunk, float scale, uint16_t* smem,
                                          const float* __restrict__ ksc, const float* __restrict__ vsc,
                                          float* __restrict__ part_o = nullptr, float2* __restrict__ part_ml = nullptr,
                                          int chunks = 0) {
    const Seg s = segs[0];
    const int group = heads / kv_heads;
    const int total = s.n * group;
    const int m0 = static_cast<int>(blockIdx.x) * kPairs;
    if (m0 >= total) return;
    const int tile_hi = s.p0 + min(m0 + kPairs - 1, total - 1) / group;
    const int tile_lo = max(0, s.p0 + m0 / group - window + 1);
    const int kvh = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int w0 = m0 + warp * 16;
    const bool busy = w0 < total;
    const int w_last = s.p0 + min(w0 + 15, total - 1) / group;
    int pos[2], lo[2];
    bool ok[2];
    size_t prow[2];
    uint32_t qa[8][4];
    for (int i = 0; i < 2; ++i) {
        const int mr = w0 + g + i * 8;
        ok[i] = mr < total;
        const int lr = ok[i] ? mr / group : 0;
        pos[i] = ok[i] ? s.p0 + lr : -1;
        lo[i] = max(0, pos[i] - window + 1);
        prow[i] = static_cast<size_t>(s.row0 + lr) * heads + kvh * group + (ok[i] ? mr % group : 0);
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
    float O[16][4];  // the merge's running o, ll and mm a row
    float MM[2] = {-INFINITY, -INFINITY}, LL[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.0f;
    const size_t head_base = s.off + static_cast<size_t>(kvh) * ring * kDim;
    const int k_row = lane & 7, k_pc = lane >> 3;
    const int v_row = lane & 15, v_pc = lane >> 4;
    // the block's 64-key tiles, chunk by chunk: tile it is chunk c's keys [kt, kt + 63]
    const int c_lo = tile_lo / chunk, c_hi = tile_hi / chunk;
    auto first = [&](int c) { return max(c * chunk, tile_lo) & ~(kKeys - 1); };
    auto tilesOf = [&](int c) { return (min(c * chunk + chunk - 1, tile_hi) - first(c)) / kKeys + 1; };
    int n_tiles = 0;
    for (int c = c_lo; c <= c_hi; ++c) n_tiles += tilesOf(c);
    // tile `it`'s chunk and key start
    auto at = [&](int it, int& c, int& kt) {
        c = c_lo;
        while (it >= tilesOf(c)) { it -= tilesOf(c); ++c; }
        kt = first(c) + it * kKeys;
    };
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < n_tiles) {
            int c, kt;
            at(st, c, kt);
            loadTile<KV8>(smem + st * kStage, smem + st * kStage + kTile, kc, vc, head_base, ring, kt,
                          max(c * chunk, tile_lo), min(c * chunk + chunk - 1, tile_hi), ksc, vsc);
        } else {
            tf_common::cp_commit();
        }
    }
    int c = c_lo, kt = first(c_lo), left = tilesOf(c_lo);
    for (int it = 0; it < n_tiles; ++it) {
        const int s_lo = c * chunk, s_hi = s_lo + chunk - 1;
        if (kt == first(c)) {
            for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
            m[0] = m[1] = -INFINITY;
            l[0] = l[1] = 0.0f;
        }
        tf_common::cp_wait<kStages - 2>();
        __syncthreads();
        if (it + kStages - 1 < n_tiles) {
            int c2, kt2;
            at(it + kStages - 1, c2, kt2);
            uint16_t* nx = smem + ((it + kStages - 1) % kStages) * kStage;
            loadTile<KV8>(nx, nx + kTile, kc, vc, head_base, ring, kt2, max(c2 * chunk, tile_lo),
                          min(c2 * chunk + chunk - 1, tile_hi), ksc, vsc);
        } else {
            tf_common::cp_commit();
        }
        const uint16_t* ks = smem + (it % kStages) * kStage;
        const uint16_t* vs = ks + kTile;
        if (busy && kt <= w_last) {
            float sc[8][4];
            for (int j = 0; j < 8; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
                for (int k = 0; k < 8; k += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4(b, tf_common::smem(ks + piece(8 * j + k_row, 2 * k + k_pc)));
                    qmm_frag::mma(sc[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b[0], b[1]);
                    qmm_frag::mma(sc[j], qa[k + 1][0], qa[k + 1][1], qa[k + 1][2], qa[k + 1][3], b[2], b[3]);
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
            uint32_t pk[8][2];
            for (int j = 0; j < 8; ++j) {
                uint16_t p[4];
                for (int e = 0; e < 4; ++e) {
                    const int i = e >> 1;
                    p[e] = tobf(exp2f(sc[j][e] - mu[i]));
                    l[i] += bf(p[e]);
                }
                pk[j][0] = pack(p[0], p[1]);
                pk[j][1] = pack(p[2], p[3]);
            }
            for (int n = 0; n < 16; ++n) {
                acc[n][0] *= alpha[0];
                acc[n][1] *= alpha[0];
                acc[n][2] *= alpha[1];
                acc[n][3] *= alpha[1];
            }
            for (int u = 0; u < 4; ++u) {
                const uint32_t a0 = pk[2 * u][0], a1 = pk[2 * u][1], a2 = pk[2 * u + 1][0], a3 = pk[2 * u + 1][1];
                for (int n = 0; n < 16; n += 2) {
                    uint32_t b[4];
                    qmm_frag::ldmatrix4t(b, tf_common::smem(vs + piece(16 * u + v_row, n + v_pc)));
                    qmm_frag::mma(acc[n], a0, a1, a2, a3, b[0], b[1]);
                    qmm_frag::mma(acc[n + 1], a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        if (--left == 0) {  // the chunk's last tile: its partial, folded as the merge folds it
            if (busy) {
                for (int i = 0; i < 2; ++i) {
                    l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
                    l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
                }
                for (int i = 0; i < 2; ++i) {
                    // the merge reads chunks lo / chunk .. pos / chunk of a row, and attnTile stored nothing else
                    if (!ok[i] || c < lo[i] / chunk || c > pos[i] / chunk) continue;
                    if constexpr (PART) {
                        const size_t at = prow[i] * chunks + (c - lo[i] / chunk);
                        if (t == 0) part_ml[at] = make_float2(m[i], l[i]);
                        float* o = part_o + at * kDim + 2 * t;
                        for (int n = 0; n < 16; ++n) *reinterpret_cast<float2*>(o + 8 * n) = make_float2(acc[n][2 * i], acc[n][2 * i + 1]);
                        continue;
                    }
                    const float mn = fmaxf(MM[i], m[i]);
                    const float a = exp2f(MM[i] - mn), b = exp2f(m[i] - mn);
                    // tf_ink2_merge's (one chunk a slot) roundings: ll = fma(a, ll, l b), o = fma(a, o, po b)
                    LL[i] = __fmaf_rn(a, LL[i], __fmul_rn(l[i], b));
                    for (int n = 0; n < 16; ++n) {
                        O[n][2 * i] = __fmaf_rn(a, O[n][2 * i], __fmul_rn(acc[n][2 * i], b));
                        O[n][2 * i + 1] = __fmaf_rn(a, O[n][2 * i + 1], __fmul_rn(acc[n][2 * i + 1], b));
                    }
                    MM[i] = mn;
                }
            }
            if (c < c_hi) {
                ++c;
                kt = first(c);
                left = tilesOf(c);
            }
        } else {
            kt += kKeys;
        }
    }
    if (!busy || PART) return;
    for (int i = 0; i < 2; ++i) {
        if (!ok[i]) continue;
        const size_t rh = prow[i];
        const float sk = bf(sink[rh]) * kLog2e;
        const float mf = fmaxf(MM[i], sk);
        const float a = exp2f(MM[i] - mf);
        // tf_ink2_merge's roundings: ll a + exp2(sk - mf) as one fma, then (o a) / den
        const float den = __fmaf_rn(a, LL[i], exp2f(sk - mf));
        for (int n = 0; n < 16; ++n)
            for (int h = 0; h < 2; ++h) {
                const int d = 8 * n + 2 * t + h;
                const float y = __fdiv_rn(__fmul_rn(a, O[n][2 * i + h]), den);
                const float gv = bf(gate[rh * kDim + d]);
                out[rh * kDim + d] = tobf(y * (1.0f / (1.0f + expf(-gv))));
            }
    }
}
}  // namespace ink2p

#define TF_INK2_PROMPT_ARGS                                                                                       \
    const uint16_t *__restrict__ q, const uint16_t *__restrict__ kc, const uint16_t *__restrict__ vc,             \
        const uint16_t *__restrict__ tab, const ink2p::Seg *__restrict__ segs, float *__restrict__ part_o,      \
        float2 *__restrict__ part_ml, int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale

// One stream's prompt rows on a global layer with bf16 caches.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1) tf_ink2_attn_prompt(TF_INK2_PROMPT_ARGS) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::promptTile<false>(q, kc, vc, tab, segs, part_o, part_ml, heads, kv_heads, ring, chunk, c0, chunks, scale,
                             smem, nullptr, nullptr);
}

// tf_ink2_attn_prompt over fp8 caches.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1)
    tf_ink2_attn_prompt8(TF_INK2_PROMPT_ARGS, const float* __restrict__ ksc, const float* __restrict__ vsc) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::promptTile<true>(q, kc, vc, tab, segs, part_o, part_ml, heads, kv_heads, ring, chunk, c0, chunks, scale,
                            smem, ksc, vsc);
}

#define TF_INK2_SLIDE_ARGS                                                                                        \
    const uint16_t *__restrict__ q, const uint16_t *__restrict__ kc, const uint16_t *__restrict__ vc,             \
        const uint16_t *__restrict__ tab, const ink2p::Seg *__restrict__ segs, const uint16_t *__restrict__ sink, \
        const uint16_t *__restrict__ gate, uint16_t *__restrict__ out, int heads, int kv_heads, int ring, int window, \
        int chunk, float scale

// One stream's prompt rows on a sliding layer, attention through the sink and gate in one launch (bf16 caches).
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1) tf_ink2_attn_slide(TF_INK2_SLIDE_ARGS) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::slideTile<false>(q, kc, vc, tab, segs, sink, gate, out, heads, kv_heads, ring, window, chunk, scale, smem,
                            nullptr, nullptr);
}

// tf_ink2_attn_slide over fp8 caches.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1)
    tf_ink2_attn_slide8(TF_INK2_SLIDE_ARGS, const float* __restrict__ ksc, const float* __restrict__ vsc) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::slideTile<true>(q, kc, vc, tab, segs, sink, gate, out, heads, kv_heads, ring, window, chunk, scale, smem,
                           ksc, vsc);
}

// Debug (TF_INKLING2_SLIDE_CHECK): count differing bf16 words of a and b; record the first difference.
extern "C" __global__ void tf_ink2_slide_cmp(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, long long n,
                                             unsigned* __restrict__ res) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<long long>(gridDim.x) * blockDim.x)
        if (a[i] != b[i]) {
            if (atomicAdd(res, 1u) == 0) { res[1] = static_cast<unsigned>(i); res[2] = a[i]; res[3] = b[i]; }
        }
}

// Debug: tf_ink2_attn_slide's chunk partials into tf_ink2_attn's slots (no fold), for tf_ink2_merge.
extern "C" __global__ void __launch_bounds__(ink2p::kThreads, 1)
    tf_ink2_attn_slidep(TF_INK2_SLIDE_ARGS, float* __restrict__ part_o, float2* __restrict__ part_ml, int chunks) {
    extern __shared__ __align__(16) uint16_t smem[];
    ink2p::slideTile<false, true>(q, kc, vc, tab, segs, sink, gate, out, heads, kv_heads, ring, window, chunk, scale,
                                  smem, nullptr, nullptr, part_o, part_ml, chunks);
}

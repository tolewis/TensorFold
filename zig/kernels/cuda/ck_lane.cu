// FP8 projections in the checkpoint's own math (nvfp4/lane4.cu's A8): rows alone, K slices by shape, so rows invariant.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "mma4.cuh"
#include "nvfp4q.cuh"
#include "common/qmm_frag.cuh"

using namespace qmm_frag;
using namespace mma4;

// The checkpoint's NVFP4 rows for the prompt GEMM: e2m1 codes [m][k/2] then e4m3 block scales [m][k/16].
namespace tf_ck4p {

// The larger, or a NaN when either is (fmaxf drops it): a block holding a NaN gets a NaN scale.
__device__ __forceinline__ float max_nan(float a, float b) { return (a > b || a != a) ? a : b; }

#include "common/unpack8_value.cuh"

}  // namespace tf_ck4p

namespace tf_ck {

constexpr int STAGES = 4;

template <int BM_, int BN_, int WM_, int WN_, bool SKIP_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, THREADS = WM_ * WN_ * 32;
    static constexpr bool SKIP = SKIP_;
};

template <typename C, bool CLUSTER>
struct Tile {
    static constexpr int MT = C::BM / C::WM / 16, NT = C::BN / C::WN / 8;
    static constexpr int ROW = 64, X = C::BM * ROW, W64 = 64 * 64, W = C::BN / 64 * W64;
    static constexpr int STAGE = (X + W + 127) / 128 * 128;
    static constexpr int PARTIALS = CLUSTER ? C::BM * C::BN * 4 : 0;
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

#include "common/unpack8_value.cuh"

// x e4m3 [M, K] and w e4m3 [npad/64, K/64, 8, 32, 2, 8], both in the fragment order quant8 and the FP8 pack write.
template <typename C, bool CLUSTER>
__device__ __forceinline__ void lane(const uint8_t* __restrict__ x, const uint8_t* __restrict__ w, float alpha,
                                     __nv_bfloat16* __restrict__ out, float* __restrict__ part, int M, int N, int K,
                                     int SK, int group, const float* __restrict__ sa = nullptr,
                                     const float* __restrict__ ws = nullptr) {
    using T = Tile<C, CLUSTER>;
    constexpr int THREADS = C::THREADS, CH = T::ROW / 16;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int r0 = warp / C::WN * (C::BM / C::WM), c0 = warp % C::WN * (C::BN / C::WN);
    const int KG = K / 64, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, C::BM, C::BN, group);
    const int m0 = at.x, n0 = at.y, slice = blockIdx.z, g0 = slice * per;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int kg) {
        unsigned char* p = stage(s);
        for (int c = tid; c < C::BM * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            const bool in = m0 + r < M;
            cp16z(p + r * T::ROW + chunk<A8>(r, ch) * 16, x + static_cast<size_t>(in ? m0 + r : 0) * K + kg * T::ROW +
                  ch * 16, in);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < T::W / 16; c += THREADS) {
            const int tj = c / (T::W64 / 16), off = c % (T::W64 / 16);
            const bool in = C::BN == 64 || n0 + 64 * tj < N;
            const size_t from = (static_cast<size_t>(n0 / 64 + (in ? tj : 0)) * KG + kg) * T::W64 + off * 16;
            cp16z(pw + c * 16, w + from, in);
        }
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    const int q = lane >> 3, rr = lane & 7;
    const int live = C::SKIP ? min(T::MT, max(0, (M - m0 - r0 + 15) / 16)) : T::MT;
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pw = p + T::X;
        uint4 b[T::NT];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) b[j] = reinterpret_cast<const uint4*>(pw)[(c0 / 8 + j) * 32 + lane];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                if (i >= live) break;  // m16 tiles wholly past M: no mma (their rows are never stored)
                uint32_t a[4];
                const int r = r0 + i * 16 + rr + (q & 1) * 8;
                ldmatrix4(a, p + r * T::ROW + chunk<A8>(r, 2 * h + (q >> 1)) * 16);
#pragma unroll
                for (int j = 0; j < T::NT; ++j) mma_fp8(acc[i][j], a, h ? b[j].z : b[j].x, h ? b[j].w : b[j].y);
            }
        }
    }
    wait<0>();
    __syncthreads();
    // A row's activation scale and an output row's weight scale when the checkpoint carries them. The weight
    // scale a row is its FP8 tile table's, replicated over the K groups: [(row / 64) * KG * 64 + row % 64].
    auto mul = [&](int row, int col) {
        const float r = ws ? ws[static_cast<size_t>(col >> 6) * KG * 64 + (col & 63)] : 1.0f;
        return alpha * (sa ? sa[row] : 1.0f) * r;
    };
    auto put = [&](int row, int col, float v0, float v1) {
        __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
        if (col + 1 < N && (N & 1) == 0) {
            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
        } else {
            if (col < N) dst[0] = __float2bfloat16_rn(v0);
            if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
        }
    };
    if constexpr (CLUSTER) {
#if __CUDA_ARCH__ < 900
        __trap();
#else
        // every slice parks its partials; slice s sums pairs s, s + SK, .. over slices 0, 1, .. in that order
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * THREADS + tid] = acc[i][j][e];
        cluster.sync();
        for (int pr = slice; pr < T::MT * T::NT * 2; pr += SK) {
            const int i = pr / (T::NT * 2), j = pr / 2 % T::NT, h = pr % 2;
            const int row = m0 + r0 + i * 16 + g + 8 * h, col = n0 + c0 + j * 8 + t * 2;
            const float* p0 = cluster.map_shared_rank(mine, 0);
            float v0 = p0[2 * pr * THREADS + tid], v1 = p0[(2 * pr + 1) * THREADS + tid];
            for (int peer = 1; peer < SK; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
                v0 = v0 + theirs[2 * pr * THREADS + tid];
                v1 = v1 + theirs[(2 * pr + 1) * THREADS + tid];
            }
            if (row < M) {
                const float f = mul(row, col);
                put(row, col, v0 * f, v1 * f);
            }
        }
        cluster.sync();
#endif
    } else {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + c0 + j * 8 + t * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + r0 + i * 16 + g + h * 8;
                    if (row >= M) continue;
                    if (SK > 1) {  // unscaled slice partials: the reduce scales their sum
                        float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                        if (col < N) dst[0] = acc[i][j][2 * h];
                        if (col + 1 < N) dst[1] = acc[i][j][2 * h + 1];
                        continue;
                    }
                    const float f = mul(row, col);
                    put(row, col, acc[i][j][2 * h] * f, acc[i][j][2 * h + 1] * f);
                }
            }
    }
}

}  // namespace tf_ck

// S skips m16 tiles past M; C meets K slices in a cluster without changing their summation order.
#define TF_CK8(NAME, BM, BN, WM, SKIP, CLUSTER)                                                                      \
    extern "C" __global__ void __launch_bounds__(WM * 128)                                                           \
        NAME(const uint8_t* __restrict__ x, const uint8_t* __restrict__ w, float alpha, __nv_bfloat16* __restrict__ out, \
             float* __restrict__ part, int M, int N, int K, int SK, int group) {                                     \
        tf_ck::lane<tf_ck::Cfg<BM, BN, WM, 4, SKIP>, CLUSTER>(x, w, alpha, out, part, M, N, K, SK, group);           \
    }
TF_CK8(tf_ck8_16, 16, 64, 1, false, false)
TF_CK8(tf_ck8_16c, 16, 64, 1, false, true)
TF_CK8(tf_ck8_32, 32, 64, 1, false, false)
TF_CK8(tf_ck8_32c, 32, 64, 1, false, true)
TF_CK8(tf_ck8_64, 64, 64, 1, false, false)
TF_CK8(tf_ck8_64c, 64, 64, 1, false, true)
TF_CK8(tf_ck8_64s, 64, 64, 1, true, false)
TF_CK8(tf_ck8_64sc, 64, 64, 1, true, true)
TF_CK8(tf_ck8_128, 128, 64, 2, true, false)
TF_CK8(tf_ck8_128c, 128, 128, 2, true, true)

// The same tiles with a checkpoint's own scales: sa[row] the activation's, rs[col] the weight's, alpha 1.
#define TF_CK8R(NAME, BM, BN, WM, SKIP, CLUSTER)                                                                     \
    extern "C" __global__ void __launch_bounds__(WM * 128)                                                           \
        NAME(const uint8_t* __restrict__ x, const uint8_t* __restrict__ w, float alpha, __nv_bfloat16* __restrict__ out, \
             float* __restrict__ part, int M, int N, int K, int SK, int group, const float* __restrict__ sa,          \
             const float* __restrict__ ws) {                                                                          \
        tf_ck::lane<tf_ck::Cfg<BM, BN, WM, 4, SKIP>, CLUSTER>(x, w, alpha, out, part, M, N, K, SK, group, sa, ws);    \
    }
TF_CK8R(tf_ck8_128r, 128, 64, 2, true, false)
TF_CK8R(tf_ck8_128rc, 128, 128, 2, true, true)

// Per-row activation scale: e4m3(x / sa[row]) in the FP8 weights' fragment order, sa = amax / 448 (1e-6 floor).
extern "C" __global__ void __launch_bounds__(128)
    tf_ck_quant8_rows(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K, uint8_t* __restrict__ out,
                      float* __restrict__ sa) {
    __shared__ float red[128];
    const int row = blockIdx.x, tid = threadIdx.x;
    float amax = 0.0f;
    for (int i = tid; i < K; i += 128) {
        const float v = __bfloat162float(x[static_cast<size_t>(row) * ldx + i]);
        amax = fmaxf(amax, fabsf(v));
    }
    red[tid] = amax;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float scale = fmaxf(red[0] / 448.0f, 1e-6f);
    if (tid == 0) sa[row] = scale;
    __syncthreads();
    const float inv = 1.0f / scale;
    for (int grp = tid; grp < K / 16; grp += 128) {
        const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + grp * 16);
        float f[16];
        tf_ck::unpack8(src[0], *reinterpret_cast<float(*)[8]>(f));
        tf_ck::unpack8(src[1], *reinterpret_cast<float(*)[8]>(f + 8));
        uint32_t w[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 2 * q + (j % 2) + 8 * (j / 2);
                packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(f[k] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
            }
            w[q] = packed;
        }
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * K + grp * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// The slices summed in slice order, one fp32 add a slice, then the factor and one bf16 rounding.
extern "C" __global__ void __launch_bounds__(256)
    tf_ck8_reduce(const float* __restrict__ part, __nv_bfloat16* __restrict__ out, long long total, int SK, float alpha) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    out[i] = __float2bfloat16_rn(acc * alpha);
}

// One thread 16 inputs: e4m3(x * inv), saturating, byte 4q + j holding input 2q + j % 2 + 8 (j / 2) (fragment order).
extern "C" __global__ void __launch_bounds__(128)
    tf_ck_quant8(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K, float inv, uint8_t* __restrict__ out) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;  // one thread 16 inputs
    if (i >= static_cast<long long>(M) * (K / 16)) return;
    const int row = static_cast<int>(i / (K / 16)), grp = static_cast<int>(i % (K / 16));
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + grp * 16);
    float f[16];
    tf_ck::unpack8(src[0], *reinterpret_cast<float(*)[8]>(f));
    tf_ck::unpack8(src[1], *reinterpret_cast<float(*)[8]>(f + 8));
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = 2 * q + (j % 2) + 8 * (j / 2);
            packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(f[k] * inv, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
        }
        w[q] = packed;
    }
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * K + grp * 16) = make_uint4(w[0], w[1], w[2], w[3]);
}

// The checkpoint's NVFP4 prompt GEMM: 128 rows by 128 columns a CTA, K steps in flight, block-scaled FP4 mma.
namespace tf_ck4p {

constexpr int PM = 128, PN = 128, PSTAGES = 4;  // rows, columns, staged 64-input steps

// One staged step: the rows' codes (32 bytes, swizzled for ldmatrix) and block scales, two 64-column weight
// tiles as packed (fragment order) and their scales.
struct Stage {
    uint4 a[PM][2];
    uint32_t sa[PM];
    uint2 b[2][256];  // as packed: the dense packer's bf16 m16n8k16 order
    uint32_t sb[2][64];
};

}  // namespace tf_ck4p

// A 16-input block a thread: e4m3 scale e4m3(g * amax / 6) at scales[r][b], codes e2m1(x * g / scale) at codes[r].
extern "C" __global__ void __launch_bounds__(128)
    tf_ck_quant4_rows(const __nv_bfloat16* __restrict__ x, int ldx, int M, int K, float g, uint8_t* __restrict__ codes,
                      uint8_t* __restrict__ scales) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;  // one thread a block of 16
    if (i >= static_cast<long long>(M) * (K / 16)) return;
    const int row = static_cast<int>(i / (K / 16)), blk = static_cast<int>(i % (K / 16));
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + blk * 16);
    float f[16];
    tf_ck4p::unpack8(src[0], *reinterpret_cast<float(*)[8]>(f));
    tf_ck4p::unpack8(src[1], *reinterpret_cast<float(*)[8]>(f + 8));
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) amax = tf_ck4p::max_nan(amax, fabsf(f[j]));
    const nvfp4q::Scale sc = nvfp4q::block_scale(amax, g);
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 16; ++j) w[j / 8] |= nvfp4q::e2m1(f[j] * sc.mul) << (4 * (j % 8));
    *reinterpret_cast<uint2*>(codes + static_cast<size_t>(row) * (K / 2) + blk * 8) = make_uint2(w[0], w[1]);
    scales[static_cast<size_t>(row) * (K / 16) + blk] = static_cast<uint8_t>(sc.sf8);
}

// out (M, N) bf16 = codes (M, K/2) times the packed words (npad/64, K/64, 8, 32, 2) under their scales, times `scale`.
extern "C" __global__ void __launch_bounds__(256)
    tf_ck4_prompt(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, const uint32_t* __restrict__ w,
                  const uint8_t* __restrict__ ws, float scale, __nv_bfloat16* __restrict__ out, int M, int N, int K,
                  int npad, int group) {
    using namespace tf_ck4p;
    __shared__ __align__(128) Stage st[PSTAGES];
    __shared__ __align__(128) uint2 btx[2][256];  // one stage's words in the mxf4nvf4 order (the barrier guards reuse)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int NB = npad / 64, KG = K / 64;
    const int2 at = tile_of(blockIdx.x, M, N, PM, PN, group);
    const int m0 = at.x, n0 = at.y;
    // loader roles: a row tid / 2 (16 bytes tid % 2) and its scales; weight tile tid / 128, 16 bytes tid % 128
    const int arow = tid >> 1, half = tid & 1, bt = tid >> 7, bq = tid & 127;
    // ldmatrix rows: lane l gives row (l % 8) + 8 ((l / 8) % 2) of chunk l / 16
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lchunk = lane >> 4;
    const int nbt = n0 / 64 + bt < NB ? n0 / 64 + bt : NB - 1;  // past npad: the last tile, unread
    const size_t tile = static_cast<size_t>(nbt) * KG;
    auto load = [&](int stage, int kg) {
        Stage& p = st[stage];
        const int r = m0 + arow < M ? m0 + arow : M - 1;  // rows past M read the last, masked at the epilogue
        cp16(&p.a[arow][chunk<A4>(arow, half)], codes + static_cast<size_t>(r) * (K / 2) + kg * 32 + half * 16);
        if (!half) cp4(&p.sa[arow], xs + static_cast<size_t>(r) * (K / 16) + kg * 4);
        cp16(reinterpret_cast<uint4*>(p.b[bt]) + bq, reinterpret_cast<const uint4*>(w) + (tile + kg) * 128 + bq);
        if (bq < 16) cp16(reinterpret_cast<uint4*>(p.sb[bt]) + bq, reinterpret_cast<const uint4*>(ws) + (tile + kg) * 16 + bq);
    };
#pragma unroll
    for (int s = 0; s < PSTAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    float acc[4][4][4] = {};
    for (int kg = 0; kg < KG; ++kg) {
        wait<PSTAGES - 2>();
        __syncthreads();
        if (kg + PSTAGES - 1 < KG) load((kg + PSTAGES - 1) % PSTAGES, kg + PSTAGES - 1);
        commit();
        Stage& p = st[kg % PSTAGES];
        // The dense packer orders words for the bf16 m16n8k16 mma. The block-scaled mma reads lane t, reg v as
        // inputs 32v + 8t .. 8t + 7, so its nibble i comes from nibble t + 4(i & 1) of word v on quad-mate lane
        // (g << 2) | (i >> 1): a 4x4 nibble-pair transpose inside each quad, once a stage.
        for (int c = tid; c < 512; c += 256) {
            const int tile = c >> 8, wd = c & 255, nt = wd >> 5, dl = wd & 31;
            const int dg = dl >> 2, dt = dl & 3;
            uint32_t v0 = 0, v1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint2 src = p.b[tile][nt * 32 + (dg << 2) + j];
                v0 |= ((src.x >> (4 * dt)) & 0xFu) << (8 * j);
                v0 |= ((src.x >> (4 * dt + 16)) & 0xFu) << (8 * j + 4);
                v1 |= ((src.y >> (4 * dt)) & 0xFu) << (8 * j);
                v1 |= ((src.y >> (4 * dt + 16)) & 0xFu) << (8 * j + 4);
            }
            btx[tile][wd] = make_uint2(v0, v1);
        }
        __syncthreads();
        uint2 b[4];
        uint32_t sb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int nt = (wn & 1) * 4 + j;
            b[j] = btx[wn >> 1][nt * 32 + lane];
            sb[j] = p.sb[wn >> 1][nt * 8 + g];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r0 = wm * 64 + i * 16;
            uint32_t a[4];
            const int r = r0 + lrow;
            ldmatrix4(a, &p.a[r][chunk<A4>(r, lchunk)]);
            const uint32_t sa = p.sa[r0 + g + (t & 1) * 8];
#pragma unroll
            for (int j = 0; j < 4; ++j) mma_fp4(acc[i][j], a, b[j].x, b[j].y, sa, sb[j]);
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = m0 + wm * 64 + i * 16 + (lane >> 2) + h * 8;
            if (row >= M) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = n0 + wn * 32 + j * 8 + (lane & 3) * 2;
                const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else if (col < N) {
                    dst[0] = __float2bfloat16_rn(v0);
                }
            }
        }
}

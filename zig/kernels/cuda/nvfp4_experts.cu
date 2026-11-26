// Device code of nvfp4/experts.cu (lines 14-149, host launchers, comments and ATen dropped) and its instances.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "experts.cuh"
#include "common/fp_pairs.cuh"
#include "mma4.cuh"
#include "nvfp4q.cuh"

namespace tf_nvfp4_experts {

constexpr int BLOCK4 = 36;

using tf_common::fp4pair;

using tf_common::e4m3f;

// A k32 group's stage: the lane's words (Word = uint2 for half a unit's two n8 tiles), its scales and inputs.
template <int M, typename Word = uint4>
struct Stage {
  Word w[M];
  uint4 s[M];
  uint2 xa[2], xb[2];
};

template <int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st, bool hi) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
        float(&a)[4] = acc[m][0][j];
        a[0] = fmaf(p[0], s0, a[0]);
        a[1] = fmaf(p[1], s1, a[1]);
        if (hi) {
          a[2] = fmaf(p[2], s0, a[2]);
          a[3] = fmaf(p[3], s1, a[3]);
        }
      }
    }
}

#if __CUDA_ARCH__ == 1210
// GB10's register form of Ring: two k32 stages held in registers, each refilled as soon as it has been summed.
#define TF_RING_STORAGE
template <int M, int WARPS, typename Word>
struct Ring {
  __device__ __forceinline__ void load(Stage<M, Word>& st, int g, int lane, int t, int jh, const uint4* blk,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
    const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if constexpr (sizeof(Word) == 16)
        st.w[m] = tf_common::ld_nc(b + m * BLOCK4 + lane);
      else
        st.w[m] = tf_common::ld_nc2(reinterpret_cast<const uint2*>(b + m * BLOCK4 + lane) + jh);
      st.s[m] = tf_common::ld_nc(b + m * BLOCK4 + 32 + t);
    }
    const uint2 zero = make_uint2(0u, 0u);
    const int k0 = g * 32 + 4 * t;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
      st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
    }
  }

  // All KG groups in k order: `sum` takes each group's stage.
  template <typename Sum>
  __device__ __forceinline__ void run(int KG, int, int lane, int t, int jh, const uint4* blk, const __nv_bfloat16* x0,
                                      const __nv_bfloat16* x1, bool v0, bool v1, Sum sum) {
#ifndef TF_NVFP4_RING_D
#define TF_NVFP4_RING_D 2
#endif
    constexpr int D = TF_NVFP4_RING_D;  // stages in flight: the sums and their k order do not depend on it
    Stage<M, Word> st[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
      if (d < KG) load(st[d], d, lane, t, jh, blk, x0, x1, v0, v1);
    for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const int g = g0 + d;
        if (g < KG) {
          sum(st[d]);
          if (g + D < KG) load(st[d], g + D, lane, t, jh, blk, x0, x1, v0, v1);
        }
      }
    }
  }
};
#else
#define TF_RING_STORAGE __shared__
// A warp's ring of D shared k32 stages, filled by cp.async: the next D - 1 groups load while one group sums.
template <int M, int WARPS, typename Word>
struct Ring {
  static constexpr int D = 4;
  Word w[WARPS][D][M][32];   // a lane's words
  uint4 s[WARPS][D][M][4];   // the group's scales, read by t
  uint2 x[WARPS][D][4][32];  // a lane's x0 then x1 inputs, h = 0 and 1

  // Group g into stage g % D: words (half jh of each lane's uint4 when Word is uint2), scales, the rows' inputs.
  __device__ __forceinline__ void issue(int g, int wi, int lane, int t, int jh, const uint4* blk,
                                        const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
    const uint4* b = blk + (size_t)g * (M * BLOCK4);
    const int d = g % D, k0 = g * 32 + 4 * t;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if constexpr (sizeof(Word) == 16)
        tf_common::cp16(&w[wi][d][m][lane], b + m * BLOCK4 + lane);
      else
        tf_common::cp8(&w[wi][d][m][lane], reinterpret_cast<const uint2*>(b + m * BLOCK4 + lane) + jh);
      if (lane < 4) tf_common::cp16(&s[wi][d][m][lane], b + m * BLOCK4 + 32 + lane);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (v0) tf_common::cp8(&x[wi][d][h][lane], x0 + k0 + 16 * h);
      if (v1) tf_common::cp8(&x[wi][d][2 + h][lane], x1 + k0 + 16 * h);
    }
  }

  // All KG groups in k order: `sum` takes each group's stage once its loads land.
  template <typename Sum>
  __device__ __forceinline__ void run(int KG, int wi, int lane, int t, int jh, const uint4* blk,
                                      const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1, Sum sum) {
#pragma unroll
    for (int d = 0; d < D - 1; ++d) {
      if (d < KG) issue(d, wi, lane, t, jh, blk, x0, x1, v0, v1);
      tf_common::cp_commit();
    }
    const uint2 zero = make_uint2(0u, 0u);
    for (int g = 0; g < KG; ++g) {
      if (g + D - 1 < KG) issue(g + D - 1, wi, lane, t, jh, blk, x0, x1, v0, v1);
      tf_common::cp_commit();
      tf_common::cp_wait<D - 1>();
      __syncwarp();
      const int d = g % D;
      Stage<M, Word> st;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        st.w[m] = w[wi][d][m][lane];
        st.s[m] = s[wi][d][m][t];
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        st.xa[h] = v0 ? x[wi][d][h][lane] : zero;
        st.xb[h] = v1 ? x[wi][d][2 + h][lane] : zero;
      }
      sum(st);
      __syncwarp();  // every lane has read stage d before the next issue refills it
    }
  }
};
#endif

// One unit of the decode form: a warp's 32 columns of item `unit / NB` over all K, then the epilogue.
template <int M, int EPI, int WARPS>
__device__ __forceinline__ void unit_full(int unit, int lane, int gq, int t, int wi,
                                          const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                          const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                          const int* __restrict__ items, const int* __restrict__ members,
                                          void* __restrict__ out, int N, float limit, int skip,
                                          float limit2 = 0.f) {
  TF_RING_STORAGE Ring<M, WARPS, uint4> ring;
  const int it = unit / NB, cb = unit - it * NB;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
    const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    ring.run(KG, wi, lane, t, 0, blk, x0, x1, v0, v1, [&](const Stage<M>& st) { compute_stage<M>(acc, st, v1); });
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit, limit2);
  }
}

template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS)
    unit_full<M, EPI, WARPS>(unit, lane, gq, t, threadIdx.x >> 5, X, x_stride, slots, W, scale, KG, NB, items,
                             members, out, N, limit, skip);
}

// The decode form's units dealt in equal waves: ceil(units / waves) warps run, the others leave at once.
template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_waves(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                       const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                       const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                       float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  if (units == 0) return;  // no items: no wave to divide by
  const int warps = gridDim.x * WARPS, waves = (units + warps - 1) / warps;
  const int run = (units + waves - 1) / waves, w = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (w >= run) return;
  for (int unit = w; unit < units; unit += run)
    unit_full<M, EPI, WARPS>(unit, lane, gq, t, threadIdx.x >> 5, X, x_stride, slots, W, scale, KG, NB, items,
                             members, out, N, limit, skip);
}

// Half a unit's stage: the lane's words of n8 tiles 2jh and 2jh + 1, all its scales, its inputs.
template <int M>
using Half = Stage<M, uint2>;

// compute_stage's sums for tiles j = 2jh + jj: scale word 2h + jh, byte pair jj * 16.
template <int M>
__device__ __forceinline__ void compute_half(float (&acc)[M][2][4], const Half<M>& st, bool hi, int jh) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
      const uint32_t sw = h == 0 ? (jh ? st.s[m].y : st.s[m].x) : (jh ? st.s[m].w : st.s[m].z);
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const uint32_t word = jj ? st.w[m].y : st.w[m].x;
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const int sh = jj * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
        float(&a)[4] = acc[m][jj];
        a[0] = fmaf(p[0], s0, a[0]);
        a[1] = fmaf(p[1], s1, a[1]);
        if (hi) {
          a[2] = fmaf(p[2], s0, a[2]);
          a[3] = fmaf(p[3], s1, a[3]);
        }
      }
    }
}

// experts.cuh's epilogue on a half unit's two n8 tiles.
template <int EPI, int M>
__device__ __forceinline__ void epilogue_half(const float (&acc)[M][2][4], void* out, int N, int col0, int pr0, int pr1,
                                              bool v0, bool v1, float limit, float limit2 = 0.f) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][j][2 * h], a1 = acc[0][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0, o1;
        if constexpr (EPI == 1) {
          o0 = relu2(a0);
          o1 = relu2(a1);
        } else if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][j][2 * h], limit);
          o1 = swiglu(a1, acc[M - 1][j][2 * h + 1], limit);
        } else if constexpr (EPI == 4) {  // experts.cuh's epilogue 4, as written there
          if (limit2 < 0.f) {
            o0 = softcap_swiglu(bf(a0), bf(acc[M - 1][j][2 * h]), limit, -limit2);
            o1 = softcap_swiglu(bf(a1), bf(acc[M - 1][j][2 * h + 1]), limit, -limit2);
          } else {
            o0 = softcap_swiglu(a0, acc[M - 1][j][2 * h], limit, limit2);
            o1 = softcap_swiglu(a1, acc[M - 1][j][2 * h + 1], limit, limit2);
          }
        } else {
          o0 = a0;
          o1 = a1;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

// The decode form with two warps a unit (16 columns each): the same sums, twice the warps for few items.
template <int M, int EPI, int WARPS>
__device__ __forceinline__ void half_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                                          const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                                          const int* __restrict__ counts, const int* __restrict__ members,
                                          void* __restrict__ out, int N, float limit, int skip, float limit2);

template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_half(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                      const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                      float limit, int skip) {
  half_body<M, EPI, WARPS>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N, limit, skip, 0.f);
}

template <int M, int EPI, int WARPS>
__device__ __forceinline__ void half_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                                          const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                                          const int* __restrict__ counts, const int* __restrict__ members,
                                          void* __restrict__ out, int N, float limit, int skip, float limit2) {
  TF_RING_STORAGE Ring<M, WARPS, uint2> ring;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3, wi = threadIdx.x >> 5;
  const int units = __ldg(counts) * NB * 2;
  for (int unit = blockIdx.x * WARPS + wi; unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / (NB * 2), rest = unit - it * NB * 2, cb = rest >> 1, jh = rest & 1;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
      const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
      const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
      float acc[M][2][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
      ring.run(KG, wi, lane, t, jh, blk, x0, x1, v0, v1, [&](const Half<M>& st) { compute_half<M>(acc, st, v1, jh); });
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][j][q] *= g;
      }
      epilogue_half<EPI, M>(acc, out, N, cb * COLS + 16 * jh + 2 * t, pr0, pr1, v0, v1, limit, limit2);
    }
  }
}

// The prompt form's tile: 16 RT rows a warp, WM x WN warps (BM pairs of one item by WN 32-column blocks a CTA).
template <int M, int RT, int WM, int WN>
struct Prompt4 {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, XC = 8, SG = 2, WB = M * BLOCK4;
  static constexpr int XU = BM * XC, WU = WN * SG * WB, SU = XU + WU;  // uint4 a stage: 64 inputs of X, then weights
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGES = SU * 16 * 4 <= 49152 ? 4 : SU * 16 * 3 <= 49152 ? 3 : 2;
  // row r's 16-byte chunk c: rows r .. r + 3 put a 16-input block's lanes on distinct banks
  static __device__ __forceinline__ int chunk(int r, int c) { return r * XC + (c ^ ((r & 3) << 1)); }
};

// An e4m3 scale in both bf16 halves (bytes 0 and 2 of x, the others zero), exact.
__device__ __forceinline__ uint32_t e4m3x2(uint32_t x) {
  const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

// a * b in bf16 pairs; exact for e2m1 codes times e4m3 scales (six significant bits at most).
using tf_common::mul2;

// mma from zero (C = 0): the same sums as the decode form's zeroed p[4], no register moves.
__device__ __forceinline__ void mma_z(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) { qmm_frag::mma0(d, a0, a1, a2, a3, b0, b1); }

// Prompt rows, 64 inputs a stage, NT row tiles a warp. FACT: the decode form's sums (per-16 mma from zero, then fma).
template <int M, int EPI, int RT, int WM, int WN, int NT, bool FACT = false>
__device__ __forceinline__ void prompt_tiles(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                             int NB, const int* __restrict__ members, void* __restrict__ out, int N,
                                             float limit, int cbt, int e, int first, int cnt, float limit2 = 0.f) {
  using P = Prompt4<M, RT, WM, WN>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::chunk(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g0) {  // k32 groups g0 .. g0 + ng - 1 (ng < SG only at an odd end)
    const int ng = min(P::SG, KG - g0);
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::SG * P::WB; q += P::THREADS) {
      const int j = q / (P::SG * P::WB), o = q - j * P::SG * P::WB;
      if (o < ng * P::WB) cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;  // array extent
  const int base = 16 * RT * wm;
  const bool active = NT > 0 && cb < NB;
  const uint32_t b = gq & 1;  // byte 2 (j & 1) + (gq & 1) of a scale word holds column j * 8 + gq's scale
  const uint32_t pick[2] = {b | 0x4040u | (b << 8), (b + 2) | 0x4040u | ((b + 2) << 8)};
  float acc[M][TA][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < NT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + P::SG - 1) / P::SG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) stage(s, s * P::SG);
    cp_commit();
  }
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, (k + STAGES - 1) * P::SG);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
#pragma unroll
    for (int gi = 0; gi < P::SG; ++gi) {
      if (k * P::SG + gi >= KG) break;
      const uint4* ws = xs + P::XU + (wn * P::SG + gi) * P::WB;
      uint4 wv[M];
      uint2 sc[M][2];
      uint4 sd[M];  // FACT: the decode form's scale word of quad t (columns 2t, 2t + 1 of each n8 tile)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        wv[m] = ws[m * BLOCK4 + lane];
        if constexpr (FACT) {
          sd[m] = ws[m * BLOCK4 + 32 + t];
        } else {
        // column j * 8 + gq's scales of 16-block h: bytes 2j + (gq & 1) of an 8-byte run
#pragma unroll
        for (int h = 0; h < 2; ++h)
          sc[m][h] = reinterpret_cast<const uint2*>(ws + m * BLOCK4 + 32)[(gq >> 1) * 2 + h];
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[TA], xb[TA];
        const int u = gi * 8 + h * 4 + t;  // the 8-byte unit holding lane t's inputs 16h + 4t .. + 3 of group gi
#pragma unroll
        for (int r = 0; r < NT; ++r) {
          const int r0 = base + 16 * r + gq;
          xa[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0, u >> 1))[u & 1];
          xb[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0 + 8, u >> 1))[u & 1];
        }
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t word = comp(wv[m], j);
            if constexpr (FACT) {
              const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
              const uint32_t sw = comp(sd[m], 2 * h + (j >> 1));
              const int sh = (j & 1) * 16;
              const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
              for (int r = 0; r < NT; ++r) {
                float p[4];
                mma_z(p, xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
                float(&a)[4] = acc[m][r][j];
                a[0] = fmaf(p[0], s0, a[0]);
                a[1] = fmaf(p[1], s1, a[1]);
                a[2] = fmaf(p[2], s0, a[2]);
                a[3] = fmaf(p[3], s1, a[3]);
              }
            } else {
            const uint32_t sv = e4m3x2(__byte_perm(j < 2 ? sc[m][h].x : sc[m][h].y, 0u, pick[j & 1]));
            const uint32_t b0 = mul2(fp4pair(word, 8 * h), sv), b1 = mul2(fp4pair(word, 8 * h + 4), sv);
#pragma unroll
            for (int r = 0; r < NT; ++r) mma(acc[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
            }
          }
      }
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int r = 0; r < NT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
    }
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int m0 = base + 16 * r + gq, m1 = m0 + 8;
      const bool v0 = m0 < cnt, v1 = m1 < cnt;
      const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
      epilogue<EPI, M, TA>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit, limit2);
    }
  }
}

template <int M, int EPI, int RT, int WM, int WN, bool FACT = false>
__device__ __forceinline__ void prompt_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                            int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N,
                                            float limit, int skip, float limit2 = 0.f) {
  static_assert(RT == 4, "the dispatch below spells out 0 .. 4 row tiles");
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;  // an expert the family computes itself, left alone as the decode form does
  const int nt = max(0, min(RT, (cnt - 16 * RT * ((threadIdx.x >> 5) / WN) + 15) >> 4));
#define TF_TILES(NT) \
  prompt_tiles<M, EPI, RT, WM, WN, NT, FACT>(X, x_stride, slots, W, scale, KG, NB, members, out, N, limit, cbt, e, \
                                             first, cnt, limit2)
  switch (nt) {
    case 4: TF_TILES(4); break;
    case 3: TF_TILES(3); break;
    case 2: TF_TILES(2); break;
    case 1: TF_TILES(1); break;
    default: TF_TILES(0);
  }
#undef TF_TILES
}

// The instances: SwiGLU gate-up, fp32 and bf16 down, and relu^2 up for experts without a gate (Nemotron).
template __global__ void nvfp4_expert_kernel<2, 2, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 0, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 3, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 1, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);

// Equal waves (SwiGLU gate-up, fp32 down), two warps a unit (SwiGLU gate-up): Flash Next's decode rows, 4 or 1 warps.
template __global__ void nvfp4_expert_waves<2, 2, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_waves<1, 0, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_half<2, 2, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_half<2, 2, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_waves<2, 2, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_waves<1, 0, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);

// One warp per CTA exposes more independent expert tiles for narrow rank-local projections.
template __global__ void nvfp4_expert_kernel<2, 2, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 0, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 3, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 1, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
// Extract 4 consecutive FP4 nibbles [4t..4t+3] (half=0) or [16+4t..16+4t+3] (half=1) from fp4pair's word order.
__device__ __forceinline__ uint32_t unpack_k4(uint32_t w, int half) {
  const uint32_t v = half ? (w >> 8) : w;
  return (v & 0x000Fu) | ((v >> 12) & 0x00F0u) | ((v << 4) & 0x0F00u) | ((v >> 8) & 0xF000u);
}

// Pre-quantized FP4 prompt tile: 64 rows x 64 inputs (2 uint4 of codes + 1 uint32 of scales per row), STAGES=4 in 46 KiB.
template <int M, int RT, int WM, int WN>
struct Prompt4Q {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, XC = 2, SG = 2, WB = M * BLOCK4;
  static constexpr int XU = BM * XC, XSU = BM / 4, WU = WN * SG * WB, SU = XU + XSU + WU;
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGES = SU * 16 * 4 <= 49152 ? 4 : SU * 16 * 3 <= 49152 ? 3 : 2;
};

template <int M, int EPI, int RT, int WM, int WN, int NT>
__device__ __forceinline__ void prompt_tiles_tc_q(const uint8_t* __restrict__ xc, const uint8_t* __restrict__ xsc,
                                                  int x_stride, int slots, const uint4* __restrict__ W,
                                                  const float* __restrict__ scale, int KG, int NB,
                                                  const int* __restrict__ members, void* __restrict__ out, int N,
                                                  float limit, int cbt, int e, int first, int cnt, float limit2 = 0.f) {
  using P = Prompt4Q<M, RT, WM, WN>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 31 & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const int c_stride = x_stride >> 1;   // bytes per row of FP4 codes (K / 2)
  const int s_stride = x_stride >> 4;   // bytes per row of UE4M3 scales (K / 16)
  const uint8_t* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = r * P::XC + c;
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = xc + (size_t)(slots ? p / slots : p) * c_stride + 16 * c;
    }
  }
  const uint8_t* ssrc = nullptr;
  if (tid < P::BM && tid < cnt) {
    const int p = __ldg(members + first + tid);
    ssrc = xsc + (size_t)(slots ? p / slots : p) * s_stride;
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g0) {
    uint4* xcs = sm + s * P::SU;
    uint32_t* xss = reinterpret_cast<uint32_t*>(xcs + P::XU);
    uint4* ws = xcs + P::XU + P::XSU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i]) cp16(xcs + xdst[i], xsrc[i] + (size_t)g0 * 16);
    if (tid < P::BM)
      xss[tid] = ssrc ? *reinterpret_cast<const uint32_t*>(ssrc + (size_t)g0 * 2) : 0u;
    for (int q = tid; q < cbs * P::SG * P::WB; q += P::THREADS) {
      const int j = q / (P::SG * P::WB), o = q - j * P::SG * P::WB;
      cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;
  const int base = 16 * RT * wm;
  const bool active = NT > 0 && cb < NB;
  float acc[M][TA][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < NT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = KG >> 1;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) stage(s, s * P::SG);
    cp_commit();
  }
  const int t_lo = (t & 1) * 2, t_hi = t_lo + 1, thalf = t >> 1;
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, (k + STAGES - 1) * P::SG);
    cp_commit();
    if (!active) continue;
    const uint4* xcs = sm + (k % STAGES) * P::SU;
    const uint32_t* xss = reinterpret_cast<const uint32_t*>(xcs + P::XU);
    const uint4* ws0 = xcs + P::XU + P::XSU + (wn * P::SG) * P::WB;
    const uint4* ws1 = ws0 + P::WB;
    uint32_t a[TA][4], sa[TA];
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int r0 = base + 16 * r;
      const uint32_t* p0 = reinterpret_cast<const uint32_t*>(xcs + (r0 + gq) * P::XC);
      const uint32_t* p1 = reinterpret_cast<const uint32_t*>(xcs + (r0 + gq + 8) * P::XC);
      a[r][0] = p0[t];
      a[r][1] = p1[t];
      a[r][2] = p0[4 + t];
      a[r][3] = p1[4 + t];
      sa[r] = (t & 1) ? xss[r0 + gq + 8] : xss[r0 + gq];
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4 w0_lo = ws0[m * BLOCK4 + gq * 4 + t_lo];
      const uint4 w0_hi = ws0[m * BLOCK4 + gq * 4 + t_hi];
      const uint4 w1_lo = ws1[m * BLOCK4 + gq * 4 + t_lo];
      const uint4 w1_hi = ws1[m * BLOCK4 + gq * 4 + t_hi];
      const uint2 sc0_0 = reinterpret_cast<const uint2*>(ws0 + m * BLOCK4 + 32)[(gq >> 1) * 2 + 0];
      const uint2 sc0_1 = reinterpret_cast<const uint2*>(ws0 + m * BLOCK4 + 32)[(gq >> 1) * 2 + 1];
      const uint2 sc1_0 = reinterpret_cast<const uint2*>(ws1 + m * BLOCK4 + 32)[(gq >> 1) * 2 + 0];
      const uint2 sc1_1 = reinterpret_cast<const uint2*>(ws1 + m * BLOCK4 + 32)[(gq >> 1) * 2 + 1];
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t b0 = unpack_k4(comp(w0_lo, j), thalf) | (unpack_k4(comp(w0_hi, j), thalf) << 16);
        const uint32_t b1 = unpack_k4(comp(w1_lo, j), thalf) | (unpack_k4(comp(w1_hi, j), thalf) << 16);
        const int sh = (2 * (j & 1) + (gq & 1)) * 8;
        const uint32_t u00 = ((j < 2 ? sc0_0.x : sc0_0.y) >> sh) & 0xFFu;
        const uint32_t u01 = ((j < 2 ? sc0_1.x : sc0_1.y) >> sh) & 0xFFu;
        const uint32_t u10 = ((j < 2 ? sc1_0.x : sc1_0.y) >> sh) & 0xFFu;
        const uint32_t u11 = ((j < 2 ? sc1_1.x : sc1_1.y) >> sh) & 0xFFu;
        const uint32_t sb = u00 | (u01 << 8) | (u10 << 16) | (u11 << 24);
#pragma unroll
        for (int r = 0; r < NT; ++r) mma4::mma_fp4(acc[m][r][j], a[r], b0, b1, sa[r], sb);
      }
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int r = 0; r < NT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
    }
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int m0 = base + 16 * r + gq, m1 = m0 + 8;
      const bool v0 = m0 < cnt, v1 = m1 < cnt;
      const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
      epilogue<EPI, M, TA>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit, limit2);
    }
  }
}

template <int M, int EPI, int RT, int WM, int WN>
__device__ __forceinline__ void prompt_body_tc_q(const uint8_t* __restrict__ xc, const uint8_t* __restrict__ xsc,
                                                 int x_stride, int slots, const uint4* __restrict__ W,
                                                 const float* __restrict__ scale, int KG, int NB,
                                                 const int* __restrict__ items, const int* __restrict__ counts,
                                                 const int* __restrict__ members, void* __restrict__ out, int N,
                                                 float limit, int skip, float limit2 = 0.f) {
  static_assert(RT == 4, "the dispatch below spells out 0 .. 4 row tiles");
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const int nt = max(0, min(RT, (cnt - 16 * RT * ((threadIdx.x >> 5) / WN) + 15) >> 4));
#define TF_TILES_TCQ(NT) \
  prompt_tiles_tc_q<M, EPI, RT, WM, WN, NT>(xc, xsc, x_stride, slots, W, scale, KG, NB, members, out, N, limit, cbt, \
                                            e, first, cnt, limit2)
  switch (nt) {
    case 4: TF_TILES_TCQ(4); break;
    case 3: TF_TILES_TCQ(3); break;
    case 2: TF_TILES_TCQ(2); break;
    case 1: TF_TILES_TCQ(1); break;
    default: TF_TILES_TCQ(0);
  }
#undef TF_TILES_TCQ
}

}  // namespace tf_nvfp4_experts

// Pre-quantize BF16 rows [M, K] into NVFP4 codes [M, K/2] and UE4M3 scales [M, K/16] (one thread per 16-value block).
extern "C" __global__ void __launch_bounds__(256)
    tf_nvfp4_quant_x(const __nv_bfloat16* __restrict__ X, int x_stride, int M, int K, uint8_t* __restrict__ codes,
                     uint8_t* __restrict__ scales) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int kb = K >> 4;
  if (idx >= static_cast<long long>(M) * kb) return;
  const int row = static_cast<int>(idx / kb), blk = static_cast<int>(idx % kb);
  const __nv_bfloat16* src = X + static_cast<size_t>(row) * x_stride + blk * 16;
  const uint4 v0 = *reinterpret_cast<const uint4*>(src);
  const uint4 v1 = *reinterpret_cast<const uint4*>(src + 8);
  const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
  float f[16];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = __bfloat1622float2(p0[i]);
    const float2 b = __bfloat1622float2(p1[i]);
    f[2 * i] = a.x;
    f[2 * i + 1] = a.y;
    f[8 + 2 * i] = b.x;
    f[8 + 2 * i + 1] = b.y;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
  }
  const nvfp4q::Scale sc = nvfp4q::block_scale(amax, 1.0f);
  uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    w0 |= nvfp4q::e2m1(f[i] * sc.mul) << (4 * i);
    w1 |= nvfp4q::e2m1(f[8 + i] * sc.mul) << (4 * i);
  }
  *reinterpret_cast<uint2*>(codes + static_cast<size_t>(row) * (K >> 1) + blk * 8) = make_uint2(w0, w1);
  scales[static_cast<size_t>(row) * kb + blk] = static_cast<uint8_t>(sc.sf8);
}

// The prompt form's entries, 4 row tiles a warp in one row of WN warps: relu^2 up, SwiGLU gate-up, bf16 sums down.
#define TF_NVFP4_PROMPT(NAME, M, EPI, WN)                                                                              \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN>(X, x_stride, slots, W, scale, KG, NB, items, counts, members,    \
                                                    out, N, limit, skip);                                            \
  }
TF_NVFP4_PROMPT(tf_nvfp4_prompt_relu2, 1, 1, 8)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_swiglu, 2, 2, 4)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_down, 1, 3, 8)

// The decode form's factored sums on the prompt tiles, bit for bit: SwiGLU gate-up to bf16, down to fp32 slot rows.
#define TF_NVFP4_FACT(NAME, M, EPI, WN)                                                                                \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN, true>(X, x_stride, slots, W, scale, KG, NB, items, counts,       \
                                                          members, out, N, limit, skip);                             \
  }
TF_NVFP4_FACT(tf_nvfp4_fact_swiglu, 2, 2, 4)
TF_NVFP4_FACT(tf_nvfp4_fact_down_f32, 1, 0, 8)

// Soft-capped SwiGLU gate-up (Inkling 2 Mini): the decode units of nvfp4_expert_kernel<2, 2, 4> and the prompt tiles of
// tf_nvfp4_prompt_swiglu, with experts.cuh's epilogue 4 over the gate's and the up's caps.
extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_softcap(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                     const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                     const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                     float gate_cap, float up_cap, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * 4 + (threadIdx.x >> 5); unit < units; unit += gridDim.x * 4)
    tf_nvfp4_experts::unit_full<2, 4, 4>(unit, lane, gq, t, threadIdx.x >> 5, X, x_stride, slots, W, scale, KG, NB,
                                         items, members, out, N, gate_cap, skip, up_cap);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_prompt_softcap(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                            const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                            const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                            int N, float gate_cap, float up_cap, int skip) {
  tf_nvfp4_experts::prompt_body<2, 4, 4, 1, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N,
                                               gate_cap, skip, up_cap);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_prompt_softcap_tc_q(const uint8_t* __restrict__ xc, const uint8_t* __restrict__ xsc, int x_stride,
                                 int slots, const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                 int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                 const int* __restrict__ members, void* __restrict__ out, int N, float gate_cap,
                                 float up_cap, int skip) {
  tf_nvfp4_experts::prompt_body_tc_q<2, 4, 4, 1, 4>(xc, xsc, x_stride, slots, W, scale, KG, NB, items, counts, members,
                                                    out, N, gate_cap, skip, up_cap);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_nvfp4_prompt_down_tc_q(const uint8_t* __restrict__ xc, const uint8_t* __restrict__ xsc, int x_stride, int slots,
                              const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                              const int* __restrict__ items, const int* __restrict__ counts,
                              const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  tf_nvfp4_experts::prompt_body_tc_q<1, 3, 4, 1, 8>(xc, xsc, x_stride, slots, W, scale, KG, NB, items, counts, members,
                                                    out, N, limit, skip);
}

// Inkling 2 Mini's decode experts two warps a unit (16 columns each): tf_nvfp4_softcap's and
// nvfp4_expert_kernel<1, 0, 4>'s outputs bit for bit (each output's MMA tile, scale word and k-ordered fmaf chain).
extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_softcap_half(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                          const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                          const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                          float gate_cap, float up_cap, int skip) {
  tf_nvfp4_experts::half_body<2, 4, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N, gate_cap,
                                       skip, up_cap);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_down_half(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                       const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                       const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                       float limit, int skip) {
  tf_nvfp4_experts::half_body<1, 0, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N, limit,
                                       skip, 0.f);
}

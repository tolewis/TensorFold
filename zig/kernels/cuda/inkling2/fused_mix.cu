// The fused visit's mixes in mode S (fused-cuda DESIGN section 10 item 3, bench B9 `bench/b9.cu`, the glue flags of
// inkling2_ops and inkling2_loop): tf_ink2_mix_rows and loopMixRow of one decode row, the work of one phase spread
// over the warps of the grid, today's sums and orders, M4 waits with the R8 limit. Code from b9.cu (mixS, loopS):
// the staging is 5,760 B a warp (loop nq 4), and the phase takes its counters, partials and fault word as arguments.
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace fz_mix {
constexpr int NB = 8, MAXQ_MIX = 3, MAXQ_LOOP = 4;
__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float warpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct MixArgs {
  const uint16_t* blocks[NB];
  const void* partial;
  const float* query;
  const uint16_t* norm[MAXQ_MIX];
  uint16_t* out;
  int n_blocks, partial_bf16, nq, rows, C;
  float eps;
};

struct LoopMixArgs {
  const uint16_t* blocks[NB];
  const void* partial;
  const float* query[MAXQ_LOOP];
  float* out;
  long long out_stride;
  int n_blocks, partial_f32, nq, rows, H;
  float eps, scale;
};

#ifndef SPIN_CAP
#define SPIN_CAP 100000000u  // DESIGN section 3, from briefs/b0c-cuda-barrier.md
#endif

__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_rel_add(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Spin until *p >= target (or != old when NE). Returns false at the iteration limit.
template <bool SLEEP>
__device__ __forceinline__ bool spin_ge(const unsigned* p, unsigned target) {
  for (unsigned it = 0; it < SPIN_CAP; it++) {
    if (ld_acq(p) >= target) return true;
    if (SLEEP) __nanosleep(32);
  }
  return false;
}
template <bool SLEEP>
__device__ __forceinline__ bool spin_ne(const unsigned* p, unsigned old) {
  for (unsigned it = 0; it < SPIN_CAP; it++) {
    if (ld_acq(p) != old) return true;
    if (SLEEP) __nanosleep(32);
  }
  return false;
}

// Set the fault word once: (phase << 16) | (block + 1).
__device__ __forceinline__ void set_fault(unsigned* fault, unsigned phase) {
  atomicCAS(fault, 0u, (phase << 16) | (blockIdx.x + 1u));
}
// ---------------------------------------------------------------- mode S
constexpr int MIXW = 16, LOOPW = 8;   // virtual warps of today's block
constexpr int MIXRUNS = 5;            // column runs of 32 a virtual warp: ceil(2304 / 512)
constexpr int LOOPRUNS = 9;           // 2304 / 256
constexpr int STG = 5760;  // staging bytes a warp (loop pass 1: 5 planes x 9 runs x 128 B)

struct SArgs {
  MixArgs mix;
  LoopMixArgs loop;
  float* part1;     // [9][16][4]
  float* part2;     // [4][16]
  unsigned* cnt;    // [reps][8]
  unsigned* fault;
  unsigned* sink;
  int reps;
  int drop;         // a test's skipped arrival: counter 0 (unit 0), 1 + q (unit w 0 of q), 7 (block 0); -1 none
};

__device__ __forceinline__ void cpa16(void* s, const void* g) { __pipeline_memcpy_async(s, g, 16); }
// Copy `runs` runs of `rb` bytes (a multiple of 16) from g + run_stride_bytes r to s + rb r, with one warp.
__device__ __forceinline__ void stage_runs(unsigned char* s, const void* g, int runs, int rb, int stride) {
  const int lane = threadIdx.x & 31, per = rb / 16;
  for (int k = lane; k < runs * per; k += 32) {
    const int r = k / per, o = (k % per) * 16;
    cpa16(s + r * rb + o, (const char*)g + (size_t)r * stride + o);
  }
}

// Thread 0 of the warp spins; the warp then syncs.
__device__ __forceinline__ void warp_wait(unsigned* c, unsigned target, unsigned* fault, unsigned ph) {
  if ((threadIdx.x & 31) == 0)
    if (!spin_ge<false>(c, target)) set_fault(fault, ph);
  __syncwarp();
}
__device__ __forceinline__ void warp_arrive(unsigned* c, bool skip = false) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0 && !skip) { __threadfence(); red_rel_add(c, 1u); }
}
__device__ __forceinline__ void block_all(unsigned* c, unsigned* fault, unsigned ph, bool skip = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!skip) red_rel_add(c, 1u);
    if (!spin_ge<false>(c, gridDim.x)) set_fault(fault, ph);
  }
  __syncthreads();
}

// Today's blockSums second level for one value: lanes below nw hold the warp partials, the rest 0, warpSum, lane 0.
__device__ __forceinline__ float sum_level2(const float* p, int nw, int stride) {
  const int lane = threadIdx.x & 31;
  const float t = lane < nw ? __ldcg(p + lane * stride) : 0.0f;
  return __shfl_sync(0xffffffffu, warpSum(t), 0);
}

__device__ void mixS(const SArgs& A, unsigned char* stg, unsigned* cnt, unsigned& acc) {
  const MixArgs& a = A.mix;
  const int G = gridDim.x, slot = (threadIdx.x >> 5) * G + blockIdx.x, lane = threadIdx.x & 31;
  const int C = a.C, n = a.n_blocks + 1, nq = a.nq, P1 = n * MIXW, P2 = nq * MIXW;
  const int psz = a.partial_bf16 ? 2 : 4;
  // ---- stage (R12): every unit copies the inputs of its columns first
  if (slot < P1) {
    const int j = slot / MIXW, w = slot % MIXW, runs = w < 8 ? 5 : 4;
    const int xb = j < a.n_blocks ? 2 : psz;
    const void* src = j < a.n_blocks ? (const void*)a.blocks[j] : a.partial;
    stage_runs(stg, (const char*)src + 32 * w * xb, runs, 32 * xb, 512 * xb);
    for (int q = 0; q < nq; ++q)
      stage_runs(stg + 640 + q * 640, a.query + (size_t)q * C + 32 * w, runs, 128, 2048);
  } else if (slot < P1 + P2) {
    const int u = slot - P1, w = u % MIXW, q = u / MIXW, runs = w < 8 ? 5 : 4;
    for (int j = 0; j < a.n_blocks; ++j) stage_runs(stg + j * 320, a.blocks[j] + 32 * w, runs, 64, 1024);
    stage_runs(stg + a.n_blocks * 320, (const char*)a.partial + 32 * w * psz, runs, 32 * psz, 512 * psz);
    stage_runs(stg + a.n_blocks * 320 + 640, a.norm[q] + 32 * w, runs, 64, 1024);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
  if (slot < P1) {
    // ---- pass 1 unit (j, w)
    const int j = slot / MIXW, w = slot % MIXW, t = w * 32 + lane;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = t, r = 0; c < C; c += 512, ++r) {
      float x;
      if (j < a.n_blocks) x = bf(((const uint16_t*)stg)[r * 32 + lane]);
      else x = a.partial_bf16 ? bf(((const uint16_t*)stg)[r * 32 + lane]) : ((const float*)stg)[r * 32 + lane];
      v[0] += x * x;
      for (int q = 0; q < nq; ++q) v[1 + q] += x * ((const float*)(stg + 640 + q * 640))[r * 32 + lane];
    }
    for (int i = 0; i < 4; ++i) v[i] = warpSum(v[i]);
    if (lane == 0)
      for (int i = 0; i < 1 + nq; ++i) A.part1[(j * MIXW + w) * 4 + i] = v[i];
    warp_arrive(cnt + 0, A.drop == 0 && slot == 0);
  } else if (slot < P1 + P2) {
    // ---- pass 2 unit (q, w)
    const int u = slot - P1, w = u % MIXW, q = u / MIXW, t = w * 32 + lane;
    warp_wait(cnt + 0, P1, A.fault, 1);
    float rr[NB + 1], wq[NB + 1];
#pragma unroll
    for (int j = 0; j < NB + 1; ++j) {
      if (j >= n) break;
      float v0 = sum_level2(A.part1 + (j * MIXW) * 4 + 0, MIXW, 4);
      float vq = sum_level2(A.part1 + (j * MIXW) * 4 + 1 + q, MIXW, 4);
      const float r = rsqrtf(v0 / C + 1e-6f);
      rr[j] = r;
      wq[j] = vq * r * rsqrtf(static_cast<float>(C));
    }
    {  // the softmax of thread q
      float m = -INFINITY, tt = 0.0f;
#pragma unroll
      for (int j = 0; j < NB + 1; ++j) if (j < n) m = fmaxf(m, wq[j]);
#pragma unroll
      for (int j = 0; j < NB + 1; ++j) if (j < n) tt += expf(wq[j] - m);
#pragma unroll
      for (int j = 0; j < NB + 1; ++j) if (j < n) wq[j] = expf(wq[j] - m) / tt;
    }
    float hv[MIXRUNS];
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < MIXRUNS; ++r) {
      const int c = t + 512 * r;
      if (c >= C) break;
      float h = 0.0f;
#pragma unroll
      for (int j = 0; j < NB + 1; ++j) {
        if (j >= n) break;
        float x;
        if (j < a.n_blocks) x = bf(((const uint16_t*)(stg + j * 320))[r * 32 + lane]);
        else x = a.partial_bf16 ? bf(((const uint16_t*)(stg + a.n_blocks * 320))[r * 32 + lane])
                                : ((const float*)(stg + a.n_blocks * 320))[r * 32 + lane];
        h += wq[j] * (x * rr[j]);
      }
      hv[r] = h;
      s += h * h;
    }
    s = warpSum(s);
    if (lane == 0) A.part2[q * MIXW + w] = s;
    warp_arrive(cnt + 1 + q, A.drop == 1 + q && w == 0);
    warp_wait(cnt + 1 + q, MIXW, A.fault, 2);
    const float tot = sum_level2(A.part2 + q * MIXW, MIXW, 1);
    const float rq = rsqrtf(tot / C + a.eps);
    uint16_t* o = a.out + (size_t)q * a.rows * C;
#pragma unroll
    for (int r = 0; r < MIXRUNS; ++r) {
      const int c = t + 512 * r;
      if (c >= C) break;
      const float nv = bf(((const uint16_t*)(stg + a.n_blocks * 320 + 640))[r * 32 + lane]);
      o[c] = tobf(nv * (hv[r] * rq));
    }
    acc ^= __float_as_uint(hv[0]);
  }
  block_all(cnt + 7, A.fault, 3, A.drop == 7 && blockIdx.x == 0);
}

__device__ void loopS(const SArgs& A, unsigned char* stg, unsigned* cnt, unsigned& acc) {
  const LoopMixArgs& a = A.loop;
  const int G = gridDim.x, slot = (threadIdx.x >> 5) * G + blockIdx.x, lane = threadIdx.x & 31;
  const int Hh = a.H, S = a.n_blocks + 1, nq = a.nq, P1 = S * LOOPW, P2 = nq * LOOPW;
  const int psz = a.partial_f32 ? 4 : 2;
  if (slot < P1) {
    const int s = slot / LOOPW, w = slot % LOOPW;
    const int xb = s < a.n_blocks ? 2 : psz;
    const void* src = s < a.n_blocks ? (const void*)a.blocks[s] : a.partial;
    stage_runs(stg, (const char*)src + 32 * w * xb, LOOPRUNS, 32 * xb, 256 * xb);
    for (int q = 0; q < nq; ++q) stage_runs(stg + 1152 + q * 1152, a.query[q] + 32 * w, LOOPRUNS, 128, 1024);
  } else if (slot < P1 + P2) {
    const int w = (slot - P1) % LOOPW;
    for (int s = 0; s < a.n_blocks; ++s) stage_runs(stg + s * 576, a.blocks[s] + 32 * w, LOOPRUNS, 64, 512);
    stage_runs(stg + a.n_blocks * 576, (const char*)a.partial + 32 * w * psz, LOOPRUNS, 32 * psz, 256 * psz);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
  if (slot < P1) {
    const int s = slot / LOOPW, w = slot % LOOPW;
    float an = 0.0f, aq[MAXQ_LOOP] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < LOOPRUNS; ++k) {
      float v;
      if (s < a.n_blocks) v = bf(((const uint16_t*)stg)[k * 32 + lane]);
      else v = a.partial_f32 ? ((const float*)stg)[k * 32 + lane] : bf(((const uint16_t*)stg)[k * 32 + lane]);
      an = fmaf(v, v, an);
      for (int q = 0; q < nq; ++q) aq[q] = fmaf(((const float*)(stg + 1152 + q * 1152))[k * 32 + lane], v, aq[q]);
    }
    an = warpSum(an);
    for (int q = 0; q < MAXQ_LOOP; ++q) aq[q] = warpSum(aq[q]);
    if (lane == 0) {
      A.part1[(s * LOOPW + w) * 4 + 0] = an;
      for (int q = 0; q < nq; ++q) A.part1[(s * LOOPW + w) * 4 + 1 + q] = aq[q];
    }
    warp_arrive(cnt + 0, A.drop == 0 && slot == 0);
  } else if (slot < P1 + P2) {
    const int u = slot - P1, w = u % LOOPW, q = u / LOOPW;
    warp_wait(cnt + 0, P1, A.fault, 1);
    // today's blockSum: warp partials added in warp order by every thread
    float inv[NB + 1], score[NB + 1];
#pragma unroll
    for (int s = 0; s < NB + 1; ++s) {
      if (s >= S) break;
      float t = 0.0f;
      for (int i = 0; i < LOOPW; ++i) t += __ldcg(A.part1 + (s * LOOPW + i) * 4 + 0);
      inv[s] = rsqrtf(t / static_cast<float>(Hh) + a.eps);
    }
#pragma unroll
    for (int s = 0; s < NB + 1; ++s) {
      if (s >= S) break;
      float t = 0.0f;
      for (int i = 0; i < LOOPW; ++i) t += __ldcg(A.part1 + (s * LOOPW + i) * 4 + 1 + q);
      score[s] = t * inv[s] * a.scale;
    }
    float m = -INFINITY, l = 0.0f, coef[NB + 1];
#pragma unroll
    for (int s = 0; s < NB + 1; ++s) if (s < S) m = fmaxf(m, score[s]);
#pragma unroll
    for (int s = 0; s < NB + 1; ++s) if (s < S) {
      score[s] = expf(score[s] - m);
      l += score[s];
    }
#pragma unroll
    for (int s = 0; s < NB + 1; ++s) if (s < S) coef[s] = (score[s] / l) * inv[s];
    float* dst = a.out + q * a.out_stride;
    for (int k = 0; k < LOOPRUNS; ++k) {
      float uu = 0.0f;
#pragma unroll
      for (int s = 0; s < NB + 1; ++s) {
        if (s >= S) break;
        float v;
        if (s < a.n_blocks) v = bf(((const uint16_t*)(stg + s * 576))[k * 32 + lane]);
        else v = a.partial_f32 ? ((const float*)(stg + a.n_blocks * 576))[k * 32 + lane]
                               : bf(((const uint16_t*)(stg + a.n_blocks * 576))[k * 32 + lane]);
        uu = fmaf(coef[s], v, uu);
      }
      dst[w * 32 + lane + 256 * k] = uu;
      acc ^= __float_as_uint(uu);
    }
  }
  block_all(cnt + 7, A.fault, 3, A.drop == 7 && blockIdx.x == 0);
}
}  // namespace fz_mix

// One mix phase (mixS of B9). cnt: 8 zeroed counters. part1 [9][16][4], part2 [4][16] fp32. stg: 4 x 5,760 B shared.
extern "C" __device__ void fz_mix_s(const fz_mix::MixArgs& a, float* part1, float* part2, unsigned* cnt,
                                    unsigned* fault, unsigned char* stg, int drop) {
    fz_mix::SArgs A;
    A.mix = a;
    A.drop = drop;
    A.part1 = part1;
    A.part2 = part2;
    A.fault = fault;
    unsigned acc = 0;
    fz_mix::mixS(A, stg + (threadIdx.x >> 5) * fz_mix::STG, cnt, acc);
}

// One loop mix phase (loopS of B9).
extern "C" __device__ void fz_loop_s(const fz_mix::LoopMixArgs& a, float* part1, float* part2, unsigned* cnt,
                                     unsigned* fault, unsigned char* stg, int drop) {
    fz_mix::SArgs A;
    A.loop = a;
    A.drop = drop;
    A.part1 = part1;
    A.part2 = part2;
    A.fault = fault;
    unsigned acc = 0;
    fz_mix::loopS(A, stg + (threadIdx.x >> 5) * fz_mix::STG, cnt, acc);
}

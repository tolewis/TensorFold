// Inkling 2 Mini's dense projections and RMS norm on Intel Xe2 (OpenCL C), the port of the Metal BF16 matvec
// (kernels/metal/core/bf16_matvec.metal, TF_BF16_MATVECS) and of sp_norm (kernels/metal/spark25/glue.metal).
//
// Matvec: y[rows][n] = bf16(x[rows][k] . W[n][k]), x and W bf16, W row major, every sum FP32, one bf16 store (round
// to nearest even) a value, rows `ldy` apart. A sub-group owns one output column j and reads W[j] once for every
// row of x (1, 2, 4, 8, 16 or 32). Lane l takes the 8-wide chunks l, l + SG, l + 2 SG, ... in turn: each chunk's
// eight products are one serial chain (multiply, then add, never fused), added to the lane's running sum, and the
// sub-group adds the lanes' sums. Metal's lanes take the same chunks at stride 32; the sum order inside a chunk and
// across lanes is the compiler's there, and the contract only needs FP32 sums, so this order is pinned here, not there.
//
// Norm: out[row][d] = bf16(w[d] * (x[row][d] * inv)) with inv = 1 / sqrt(sum(x^2) / 2304 + 1e-6), x FP32.
//
// Build with -cl-fp32-correctly-rounded-divide-sqrt so `/` and sqrt are IEEE.

#pragma OPENCL FP_CONTRACT OFF

#ifndef DENSE_SG
#define DENSE_SG 16 // sub-group size
#endif
#ifndef DENSE_WG
#define DENSE_WG 128 // work-items a work-group: DENSE_WG / DENSE_SG output columns
#endif
#ifndef DENSE_UNROLL
#define DENSE_UNROLL 6 // chunks a lane has in flight
#endif
#define DENSE_HIDDEN 2304

#define DENSE_KERNEL(N, SG) __attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even
static inline float lo_bf(uint u) { return as_float(u << 16); }
static inline float hi_bf(uint u) { return as_float(u & 0xffff0000u); }

// The contract plants one mistake in each kernel through the macros below; production keeps these defaults.
// contract_dense_controls.cl redefines them (and the names) before it includes this file.
#ifndef DENSE_NAME
#define DENSE_NAME(x) tf_ink2_##x
#endif
// A chunk's eight-product sum added to the lane's running sum.
#ifndef DENSE_ADD
#define DENSE_ADD(acc, part) ((acc) + (part))
#endif
// The projection's one store.
#ifndef DENSE_STORE
#define DENSE_STORE(v) f2bf(v)
#endif
#ifndef NORM_EPS
#define NORM_EPS 1e-6f
#endif
#ifndef NORM_STORE
#define NORM_STORE(v) f2bf(v)
#endif

// One chunk of eight bf16 values of W against the same chunk of every row of x.
#define DENSE_STEP(ROWS, wv, xrow, c)                                                                              \
  {                                                                                                                \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                            \
      const uint4 xv = ((global const uint4 *)(xrow + (ulong)r * k))[c];                                          \
      float part = lo_bf(wv.x) * lo_bf(xv.x);                                                                      \
      part += hi_bf(wv.x) * hi_bf(xv.x);                                                                           \
      part += lo_bf(wv.y) * lo_bf(xv.y);                                                                           \
      part += hi_bf(wv.y) * hi_bf(xv.y);                                                                           \
      part += lo_bf(wv.z) * lo_bf(xv.z);                                                                           \
      part += hi_bf(wv.z) * hi_bf(xv.z);                                                                           \
      part += lo_bf(wv.w) * lo_bf(xv.w);                                                                           \
      part += hi_bf(wv.w) * hi_bf(xv.w);                                                                           \
      acc[r] = DENSE_ADD(acc[r], part);                                                                                  \
    }                                                                                                              \
  }

// Grid (ceil(n / (DENSE_WG / DENSE_SG))). x holds ROWS rows of k values, k a multiple of 8; rows <= ROWS are stored.
#define DENSE_MATVEC(ROWS)                                                                                         \
  DENSE_KERNEL(DENSE_WG, DENSE_SG) void DENSE_NAME(dense_##ROWS)(global const ushort *x, global const ushort *w,   \
                                                                  global ushort *y, uint rows, uint n, uint k,     \
                                                                  uint ldy) {                                      \
    const uint j = get_group_id(0) * (DENSE_WG / DENSE_SG) + get_sub_group_id();                                   \
    if (j >= n) return;                                                                                            \
    const uint lane = get_sub_group_local_id(), chunks = k / 8;                                                    \
    global const uint4 *wr = (global const uint4 *)(w + (ulong)j * k);                                             \
    float acc[ROWS];                                                                                               \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;                                               \
    uint c = lane;                                                                                                 \
    for (; c + (DENSE_UNROLL - 1) * DENSE_SG < chunks; c += DENSE_UNROLL * DENSE_SG) {                             \
      uint4 wv[DENSE_UNROLL];                                                                                      \
      _Pragma("unroll") for (uint u = 0; u < DENSE_UNROLL; ++u) wv[u] = wr[c + u * DENSE_SG];                      \
      _Pragma("unroll") for (uint u = 0; u < DENSE_UNROLL; ++u) DENSE_STEP(ROWS, wv[u], x, c + u * DENSE_SG)       \
    }                                                                                                              \
    for (; c < chunks; c += DENSE_SG) {                                                                            \
      const uint4 wv = wr[c];                                                                                      \
      DENSE_STEP(ROWS, wv, x, c)                                                                                   \
    }                                                                                                              \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                            \
      const float s = sub_group_reduce_add(acc[r]);                                                                \
      if (lane == 0 && r < rows) y[(ulong)r * ldy + j] = DENSE_STORE(s);                                           \
    }                                                                                                              \
  }

DENSE_MATVEC(1)
DENSE_MATVEC(2)
DENSE_MATVEC(4)
DENSE_MATVEC(8)
DENSE_MATVEC(16)
DENSE_MATVEC(32)

// Grid (rows), 256 work-items a row. A work-item sums x[t], x[t + 256], ... as fused multiply-adds, in order; the
// sub-groups' sums are then added in order, so every work-item holds the same total.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1))) kernel void
DENSE_NAME(norm)(global const float *x, global const ushort *w, global ushort *out, uint rows) {
  const uint row = get_group_id(0), t = get_local_id(0);
  if (row >= rows) return;
  local float red[16];
  global const float *xr = x + (ulong)row * DENSE_HIDDEN;
  float sum = 0.0f;
  for (uint d = t; d < DENSE_HIDDEN; d += 256) sum = fma(xr[d], xr[d], sum);
  const float s = sub_group_reduce_add(sum);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float total = 0.0f;
  for (uint i = 0; i < 16; ++i) total += red[i];
  const float inverse = 1.0f / sqrt(total / (float)DENSE_HIDDEN + NORM_EPS);
  for (uint d = t; d < DENSE_HIDDEN; d += 256) out[(ulong)row * DENSE_HIDDEN + d] = NORM_STORE(bf2f(w[d]) * (xr[d] * inverse));
}

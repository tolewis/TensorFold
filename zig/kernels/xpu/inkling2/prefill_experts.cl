// Inkling 2 Mini routed experts for long prompt chunks on Intel Xe2 (OpenCL C).
//
// A chunk of R rows makes R * 8 routes. `pf_plan` sorts the routes by expert (stable, so a launch repeats its bytes) and
// lists items of at most PF_MT sorted rows of one expert; `pf_gather` copies each sorted route's activation row into
// sorted order. The GEMM kernels then run on sorted rows: a sub-group owns 32 weight rows (two DPAS columns) of one item
// and all of the item's rows (up to eight DPAS row tiles), so each weight value is decoded to BF16 in registers once for
// up to 64 routes, and every activation tile is read once for both columns. Arithmetic matches experts.cl: a decoded
// weight is the BF16 `(e2m1 * e4m3(scale)) * factor` (NVFP4) or `e4m3(code) * 2^(scale - 127)` (MXFP8), activations are
// BF16, every sum is FP32 (DPAS), gate and up round to BF16 before the soft-capped SwiGLU. The order of the FP32 sums is
// the hardware's, and no output depends on which other routes share its tile.
//
// Outputs stay in sorted order: `pf_gu` writes the activation [sorted row][n], `pf_down` the expert output [sorted row][hidden];
// `pf_combine` reads a token's eight routes through the inverse map and sums them in ascending expert order.
// Block reads and writes carry the item's end as the surface height, so rows of the next item neither enter a tile as
// data (they read as zero) nor are overwritten (a store past the height is dropped).

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// (no FP_CONTRACT pragma: experts.cl has none, and the activation must round as it does)

#ifndef PF_PREFIX
#define PF_PREFIX tf_ink2_pf_
#endif
#define PF_CAT_(a, b) a##b
#define PF_CAT(a, b) PF_CAT_(a, b)
#define P(name) PF_CAT(PF_PREFIX, name)

#define PF_MT 64        // sorted rows of one item: eight DPAS row tiles
#define PF_NSG 8        // sub-groups of a work-group, each over 32 weight rows
#define PF_EXPERTS 128

#define PF_DOT(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c)

static inline float P(bf_to_f)(ushort v) { return as_float((uint)v << 16); }
static inline ushort P(f_to_bf)(float v) { return intel_convert_bfloat16_as_ushort(v); }

static inline float P(e4m3)(uint c) {
  float v = as_float((c & 0x7fu) << 20) * 0x1p120f;
  v = ((c & 0x7fu) == 0x7fu) ? NAN : v;
  return as_float(as_uint(v) | ((c & 0x80u) << 24));
}

static inline float P(phi)(__global const float *amax, __global const float *den, uint expert) {
  const float a = P(bf_to_f)(P(f_to_bf)(amax[expert]));
  const float d = P(bf_to_f)(P(f_to_bf)(den[0]));
  return a / d;
}

#define PF_NIBBLE_T(w, i) \
  as_float((((i) < 6 ? ((w) << (22 - 4 * (i))) : ((w) >> (4 * (i) - 22))) & 0x01c00000u) | (((w) << (28 - 4 * (i))) & 0x80000000u))

// The B operand of 16 columns of one NVFP4 block: lane = weight row, dword p holds k = 2p (low) and 2p + 1.
static inline int8 P(nv_frag)(uint w0, uint w1, uint scale, float phi26) {
  const float k1 = P(e4m3)(scale) * 0x1p100f;
  int8 b;
#define PF_NV_PAIR(j, w, i) \
  b[j] = (int)as_uint(intel_convert_bfloat162_as_ushort2((float2)((PF_NIBBLE_T(w, i) * k1) * phi26, (PF_NIBBLE_T(w, (i) + 1) * k1) * phi26)))
  PF_NV_PAIR(0, w0, 0);
  PF_NV_PAIR(1, w0, 2);
  PF_NV_PAIR(2, w0, 4);
  PF_NV_PAIR(3, w0, 6);
  PF_NV_PAIR(4, w1, 0);
  PF_NV_PAIR(5, w1, 2);
  PF_NV_PAIR(6, w1, 4);
  PF_NV_PAIR(7, w1, 6);
#undef PF_NV_PAIR
  return b;
}

static inline float P(mx_value)(uint code, float f1, float f2) {
  const uint c7 = code & 0x7fu;
  const float t = as_float((c7 << 20) | ((code & 0x80u) << 24));
  const float v = (t * f1) * f2;
  return c7 == 0x7fu ? NAN : v;
}

static inline void P(mx_factors)(uint scale, float *f1, float *f2) {
  const int e = (int)scale - 7;
  const int a = e >> 1;
  *f1 = as_float((uint)(a + 127) << 23);
  *f2 = as_float((uint)(e - a + 127) << 23);
}

static inline int8 P(mx_frag)(uint4 w, float f1, float f2) {
  int8 b;
#pragma unroll
  for (int p = 0; p < 8; p++) {
    const uint dw = w[p >> 1] >> (16 * (p & 1));
    b[p] = (int)as_uint(intel_convert_bfloat162_as_ushort2((float2)(P(mx_value)(dw & 255u, f1, f2), P(mx_value)((dw >> 8) & 255u, f1, f2))));
  }
  return b;
}

static inline float P(cap)(float x, float cap) { return cap * tanh(x / cap); }

static inline ushort P(act)(float gate_dot, float up_dot) {
  const float gate = P(bf_to_f)(P(f_to_bf)(gate_dot));
  const float up = P(bf_to_f)(P(f_to_bf)(up_dot));
  const float left = P(cap)(gate, 4.0f) * (1.0f / (1.0f + exp(-gate)));
  return P(f_to_bf)(left * P(cap)(up, 25.0f));
}

// ---- plan: stable counting sort of the routes by expert, items of at most `mt` sorted rows --------------------------
// One work-group of 256. Thread t counts its own run of routes, the runs' counts become offsets in thread order (so the
// sort is stable), thread 0 lists the items, every thread places its run. order[sorted] = route, inv[route] = sorted.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void P(plan)(__global const int *ids, __global int *order, __global int *inv, __global int *items, __global int *item_count,
                      uint routes, uint mt) {
  local ushort cnt[256 * PF_EXPERTS];
  local int total[PF_EXPERTS];
  local int eoff[PF_EXPERTS + 1];
  const uint t = get_local_id(0);
  const uint per = (routes + 255u) / 256u;
  const uint lo = min(t * per, routes), hi = min(lo + per, routes);
  for (uint i = t; i < 256 * PF_EXPERTS; i += 256) cnt[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lo; p < hi; p++) {
    const uint e = (uint)ids[p];
    if (e < PF_EXPERTS) cnt[t * PF_EXPERTS + e]++;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < PF_EXPERTS) {
    uint run = 0;
    for (uint u = 0; u < 256; u++) {
      const uint c = cnt[u * PF_EXPERTS + t];
      cnt[u * PF_EXPERTS + t] = (ushort)run;
      run += c;
    }
    total[t] = (int)run;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    int at = 0, made = 0;
    for (uint e = 0; e < PF_EXPERTS; e++) {
      eoff[e] = at;
      for (int first = 0; first < total[e]; first += (int)mt) {
        items[3 * made] = (int)e;
        items[3 * made + 1] = at + first;
        items[3 * made + 2] = min((int)mt, total[e] - first);
        made++;
      }
      at += total[e];
    }
    eoff[PF_EXPERTS] = at;
    *item_count = made;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lo; p < hi; p++) {
    const uint e = (uint)ids[p];
    if (e < PF_EXPERTS) {
      const int pos = eoff[e] + (int)cnt[t * PF_EXPERTS + e]++;
      order[pos] = (int)p;
      inv[p] = pos;
    }
  }
}

// Grid (sorted rows), work-group 256: xs[sorted] = x[order[sorted] / topk], `k` BF16 values a row, a multiple of 8.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void P(gather)(__global const ushort *x, __global const int *order, __global ushort *xs, uint k, uint topk) {
  const uint sp = get_group_id(0);
  const uint src = (uint)order[sp] / topk;
  global const uint4 *from = (global const uint4 *)(x + (ulong)src * k);
  global uint4 *to = (global uint4 *)(xs + (ulong)sp * k);
  for (uint i = get_local_id(0); i < k / 8; i += 256) to[i] = from[i];
}

// ---- GEMM kernels --------------------------------------------------------------------------------------------------

#define PF_ITEM_SETUP                                                                                              \
  const uint item = get_group_id(1);                                                                               \
  if (item >= (uint)item_count[0]) return;                                                                         \
  const uint e = (uint)items[3 * item], first = (uint)items[3 * item + 1], cnt = (uint)items[3 * item + 2];       \
  const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();                                             \
  const uint col0 = (get_group_id(0) * PF_NSG + sg) * 32u;                                                         \
  const uint tiles = (cnt + 7u) / 8u;                                                                              \
  const uint height = first + cnt;

#define PF_READ_A(xs, k, kk, i, dst) \
  intel_sub_group_2d_block_read_16b_8r16x1c((global void *)(xs), (k) * 2, height, (k) * 2, (int2)((kk), first + 8u * (i)), (private ushort *)&(dst))

// One k16 step: tile i reads its rows once and feeds both columns.
#define PF_STEP(xs, k, kk, b0, b1)                                                                                  \
  _Pragma("unroll") for (uint i = 0; i < 8; i++) {                                                                  \
    if (i < tiles) {                                                                                                \
      short8 a;                                                                                                     \
      PF_READ_A(xs, k, kk, i, a);                                                                                   \
      acc[i][0] = PF_DOT(a, b0, acc[i][0]);                                                                         \
      acc[i][1] = PF_DOT(a, b1, acc[i][1]);                                                                         \
    }                                                                                                               \
  }

#define PF_ZERO_ACC                                                                                                 \
  float8 acc[8][2];                                                                                                 \
  _Pragma("unroll") for (uint i = 0; i < 8; i++) {                                                                  \
    acc[i][0] = 0.0f;                                                                                               \
    acc[i][1] = 0.0f;                                                                                               \
  }

// gate/up rows are interleaved (2c gate, 2c + 1 up); 32 weight rows make 16 activation columns, one block write a tile.
#define PF_STORE_GU(ys, n)                                                                                          \
  _Pragma("unroll") for (uint i = 0; i < 8; i++) {                                                                  \
    if (i < tiles) {                                                                                                \
      ushort8 o;                                                                                                    \
      _Pragma("unroll") for (uint r = 0; r < 8; r++) {                                                              \
        const float g0 = acc[i][0][r], g1 = acc[i][1][r];                                                           \
        const float p0 = sub_group_shuffle_xor(g0, 1), p1 = sub_group_shuffle_xor(g1, 1);                           \
        const bool even = (lane & 1u) == 0;                                                                         \
        const ushort a = P(act)(even ? g0 : p1, even ? p0 : g1);                                                    \
        o[r] = sub_group_shuffle(a, lane < 8 ? 2u * lane : 2u * (lane - 8u) + 1u);                                  \
      }                                                                                                             \
      intel_sub_group_2d_block_write_16b_8r16x1c((global void *)(ys), (n) * 2, height, (n) * 2, (int2)(col0 / 2u, first + 8u * i), (private ushort *)&o); \
    }                                                                                                               \
  }

#define PF_STORE_DOWN(ds, n)                                                                                        \
  _Pragma("unroll") for (uint i = 0; i < 8; i++) {                                                                  \
    if (i < tiles) {                                                                                                \
      _Pragma("unroll") for (uint j = 0; j < 2; j++) {                                                              \
        ushort8 o;                                                                                                  \
        _Pragma("unroll") for (uint r = 0; r < 8; r++) o[r] = P(f_to_bf)(acc[i][j][r]);                             \
        intel_sub_group_2d_block_write_16b_8r16x1c((global void *)(ds), (n) * 2, height, (n) * 2, (int2)(col0 + 16u * j, first + 8u * i), (private ushort *)&o); \
      }                                                                                                             \
    }                                                                                                               \
  }

// NVFP4. W: k / 2 bytes a row, S: k / 16 e4m3 bytes a row; a lane owns one row of each of its two columns and reads 64
// bytes (eight blocks) at a time.
static inline void P(nv_gemm)(__global const ushort *xs, __global const uchar *w, __global const uchar *s, __global const float *amax,
                              __global const float *den, __global const int *items, __global const int *item_count, __global ushort *out,
                              uint k, uint n, bool gu) {
  PF_ITEM_SETUP
  const uint rows_total = gu ? 2u * n : n;
  if (col0 >= rows_total) return;
  const uint nb = k / 16u;
  global const uchar *wr0 = w + ((ulong)e * rows_total + col0 + lane) * (k / 2u);
  global const uchar *wr1 = wr0 + 16ul * (k / 2u);
  global const uchar *sr0 = s + ((ulong)e * rows_total + col0 + lane) * nb;
  global const uchar *sr1 = sr0 + 16ul * nb;
  const float phi26 = P(phi)(amax, den, e) * 0x1p26f;
  PF_ZERO_ACC
  for (uint c = 0; c < nb / 8u; c++) {
    const uint16 v0 = vload16(0, (__global const uint *)(wr0 + 64u * c));
    const uint16 v1 = vload16(0, (__global const uint *)(wr1 + 64u * c));
    const uint2 s0 = vload2(0, (__global const uint *)(sr0 + 8u * c));
    const uint2 s1 = vload2(0, (__global const uint *)(sr1 + 8u * c));
#pragma unroll
    for (uint u = 0; u < 8; u++) {
      const int8 b0 = P(nv_frag)(v0[2 * u], v0[2 * u + 1], (s0[u >> 2] >> (8 * (u & 3))) & 255u, phi26);
      const int8 b1 = P(nv_frag)(v1[2 * u], v1[2 * u + 1], (s1[u >> 2] >> (8 * (u & 3))) & 255u, phi26);
      const uint kk = 128u * c + 16u * u;
      PF_STEP(xs, k, kk, b0, b1)
    }
  }
  if (gu) {
    PF_STORE_GU(out, n)
  } else {
    PF_STORE_DOWN(out, n)
  }
}

// MXFP8. W: k e4m3 bytes a row, S: k / 32 UE8M0 bytes a row; 64 bytes (two blocks) at a time.
static inline void P(mx_gemm)(__global const ushort *xs, __global const uchar *w, __global const uchar *s, __global const int *items,
                              __global const int *item_count, __global ushort *out, uint k, uint n, bool gu) {
  PF_ITEM_SETUP
  const uint rows_total = gu ? 2u * n : n;
  if (col0 >= rows_total) return;
  const uint nb = k / 32u;
  global const uchar *wr0 = w + ((ulong)e * rows_total + col0 + lane) * k;
  global const uchar *wr1 = wr0 + 16ul * k;
  global const uchar *sr0 = s + ((ulong)e * rows_total + col0 + lane) * nb;
  global const uchar *sr1 = sr0 + 16ul * nb;
  PF_ZERO_ACC
  for (uint c = 0; c < nb / 2u; c++) {
    const uint16 v0 = vload16(0, (__global const uint *)(wr0 + 64u * c));
    const uint16 v1 = vload16(0, (__global const uint *)(wr1 + 64u * c));
    const uint sc0 = (uint)sr0[2u * c] | ((uint)sr0[2u * c + 1u] << 8);
    const uint sc1 = (uint)sr1[2u * c] | ((uint)sr1[2u * c + 1u] << 8);
#pragma unroll
    for (uint blk = 0; blk < 2; blk++) {
      float f1a, f2a, f1b, f2b;
      P(mx_factors)((sc0 >> (8 * blk)) & 255u, &f1a, &f2a);
      P(mx_factors)((sc1 >> (8 * blk)) & 255u, &f1b, &f2b);
#pragma unroll
      for (uint h = 0; h < 2; h++) {
        uint4 q0, q1;
        if (blk == 0 && h == 0) { q0 = v0.s0123; q1 = v1.s0123; }
        else if (blk == 0) { q0 = v0.s4567; q1 = v1.s4567; }
        else if (h == 0) { q0 = v0.s89ab; q1 = v1.s89ab; }
        else { q0 = v0.scdef; q1 = v1.scdef; }
        const int8 b0 = P(mx_frag)(q0, f1a, f2a);
        const int8 b1 = P(mx_frag)(q1, f1b, f2b);
        const uint kk = 64u * c + 32u * blk + 16u * h;
        PF_STEP(xs, k, kk, b0, b1)
      }
    }
  }
  if (gu) {
    PF_STORE_GU(out, n)
  } else {
    PF_STORE_DOWN(out, n)
  }
}

#define PF_GEMM_ATTR __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * PF_NSG, 1, 1)))

PF_GEMM_ATTR __kernel void P(nvfp4_gu)(__global const ushort *xs, __global const uchar *w, __global const uchar *s, __global const float *amax,
                                       __global const float *den, __global const int *items, __global const int *item_count,
                                       __global ushort *ys, uint k, uint n) {
  P(nv_gemm)(xs, w, s, amax, den, items, item_count, ys, k, n, true);
}
PF_GEMM_ATTR __kernel void P(nvfp4_down)(__global const ushort *xs, __global const uchar *w, __global const uchar *s, __global const float *amax,
                                         __global const float *den, __global const int *items, __global const int *item_count,
                                         __global ushort *ds, uint k, uint n) {
  P(nv_gemm)(xs, w, s, amax, den, items, item_count, ds, k, n, false);
}
PF_GEMM_ATTR __kernel void P(mxfp8_gu)(__global const ushort *xs, __global const uchar *w, __global const uchar *s, __global const int *items,
                                       __global const int *item_count, __global ushort *ys, uint k, uint n) {
  P(mx_gemm)(xs, w, s, items, item_count, ys, k, n, true);
}
PF_GEMM_ATTR __kernel void P(mxfp8_down)(__global const ushort *xs, __global const uchar *w, __global const uchar *s, __global const int *items,
                                         __global const int *item_count, __global ushort *ds, uint k, uint n) {
  P(mx_gemm)(xs, w, s, items, item_count, ds, k, n, false);
}

// ---- combine: the route outputs of a token weigh in ascending expert order, then round once to BF16 -----------------
// Grid (ceil(n / 64), rows). ds is in sorted order; inv maps a route to its sorted row.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void P(combine)(__global const ushort *ds, __global const int *inv, __global const int *ids, __global const float *weights,
                         __global ushort *y, uint n, uint topk) {
  local int order[8];
  const uint col = get_global_id(0), token = get_group_id(1);
  const uint lid = get_local_id(0);
  // slot `lid` takes the place its (expert, slot) pair holds in ascending order
  if (lid < topk) {
    const int mine = ids[token * topk + lid];
    uint rank = 0;
    for (uint slot = 0; slot < topk; slot++) {
      const int id = ids[token * topk + slot];
      rank += (id < mine || (id == mine && slot < lid)) ? 1u : 0u;
    }
    order[rank] = (int)lid;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (col >= n) return;
  float sum = 0.0f;
  for (uint t = 0; t < topk; t++) {
    const int slot = order[t];
    sum += weights[token * topk + slot] * P(bf_to_f)(ds[(ulong)inv[token * topk + slot] * n + col]);
  }
  y[(ulong)token * n + col] = P(f_to_bf)(sum);
}

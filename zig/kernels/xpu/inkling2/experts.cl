// Inkling 2 Mini routed experts on Intel Xe2: the checkpoint's NVFP4 (layers 0 to 19) and MXFP8 (layers 20 to 23)
// codes decode to BF16 in registers, activations stay BF16, every sum is FP32 (XMX DPAS products and sums).
// The per-expert NVFP4 factor is bf16(amax[e]) / bf16(denominator[0]); a decoded weight is
// bf16((e2m1 * e4m3(scale)) * factor), an MXFP8 weight bf16(e4m3(code) * 2^(scale - 127)).
// Same arithmetic as the Metal and CUDA ports; the order of the FP32 sums is the hardware's.
//
// Layouts (rows are contiguous, the matrix of an expert follows the matrix of the expert before it):
//   gate/up  rows 2c (gate) and 2c + 1 (up) make output column c; down rows are the output columns.
//   NVFP4    W: k/2 bytes a row (low nibble the even column), S: k/16 e4m3 bytes a row, amax[expert], denominator[0].
//   MXFP8    W: k e4m3 bytes a row, S: k/32 UE8M0 bytes a row.
//
// Decode path: one sub-group per 16 rows of one route (lane = row, one DPAS row of the activation), bandwidth bound.
// Prompt path: tf_ink2_plan groups the routes by expert; one sub-group per (16 rows, run of at most 32 routes).
//
// `P(name)` prefixes every symbol so contract controls can include this file again with a fault planted.

#ifndef INK2_PREFIX
#define INK2_PREFIX tf_ink2_
#endif
#ifndef INK2_ONLY
#define INK2_ONLY 0xfff
#endif
#ifndef INK2_KS
#define INK2_KS 4 // sub-groups of a decode work-group, each over a slice of k
#endif
// Each decode kernel may take its own split: a slice of k should hold the same number of 64-byte lines in every sub-group.
#ifndef INK2_KS_NV_GU
#define INK2_KS_NV_GU 4
#endif
#ifndef INK2_KS_NV_DOWN
#define INK2_KS_NV_DOWN 3
#endif
#ifndef INK2_KS_MX_GU
#define INK2_KS_MX_GU INK2_KS
#endif
#ifndef INK2_KS_MX_DOWN
#define INK2_KS_MX_DOWN INK2_KS
#endif
#define INK2_CAT_(a, b) a##b
#define INK2_CAT(a, b) INK2_CAT_(a, b)
#define P(name) INK2_CAT(INK2_PREFIX, name)

#define INK2_NV_GU 1
#define INK2_NV_DOWN 2
#define INK2_MX_GU 4
#define INK2_MX_DOWN 8
#define INK2_NV_QUANT 16
#define INK2_MX_QUANT 32
#define INK2_COMBINE 64
#define INK2_PLAN 128
#define INK2_NV_GU_PROMPT 256
#define INK2_NV_DOWN_PROMPT 512
#define INK2_MX_GU_PROMPT 1024
#define INK2_MX_DOWN_PROMPT 2048

#define INK2_SG 16
#define INK2_RUN 32 // routes of one prompt item at most
#define INK2_MAX_EXPERTS 512

static inline float P(bf_to_f)(ushort v) { return as_float((uint)v << 16); }
static inline ushort P(f_to_bf)(float v) { return intel_convert_bfloat16_as_ushort(v); }

// E4M3 (the finite-only variant: 0x7f and 0xff are NaN) as FP32: the code's 7 bits placed under an FP32 exponent
// (bias 127) read as e - 127 + 120, the FP32 denormal range gives the E4M3 denormals.
static inline float P(e4m3)(uint c) {
  float v = as_float((c & 0x7fu) << 20) * 0x1p120f;
  v = ((c & 0x7fu) == 0x7fu) ? NAN : v;
  return as_float(as_uint(v) | ((c & 0x80u) << 24));
}

// The two BF16 globals divide in IEEE FP32 (the image builds with -cl-fp32-correctly-rounded-divide-sqrt).
static inline float P(phi)(__global const float *amax, __global const float *den, uint expert) {
  float a = P(bf_to_f)(P(f_to_bf)(amax[expert]));
#ifdef INK2_PLANT_PHI
  a *= 2.0f;
#endif
  const float d = P(bf_to_f)(P(f_to_bf)(den[0]));
  return a / d;
}

// One NVFP4 weight as the FP32 value before the BF16 rounding. `k1` is e4m3(scale) * 2^100 and `phi26` the factor *
// 2^26: the nibble's bits under an FP32 exponent are e2m1 * 2^-126 (the FP32 denormal gives 0.5), so (t * k1) * phi26
// is ((e2m1 * scale) * factor) with the first product exact and the one rounding in the second, as the reference.
// `t` is the nibble at bits 4 * i of a dword, moved in one shift each for the magnitude and for the sign.
#define INK2_NIBBLE_T(w, i) \
  as_float((((i) < 6 ? ((w) << (22 - 4 * (i))) : ((w) >> (4 * (i) - 22))) & 0x01c00000u) | (((w) << (28 - 4 * (i))) & 0x80000000u))

static inline float P(nv_value)(uint nibble, float k1, float phi26) {
  const float t = as_float(((nibble & 7u) << 22) | ((nibble & 8u) << 28));
  return (t * k1) * phi26;
}

// The weights of one 16-column block as the B operand of a DPAS: dword p holds columns 2p (low) and 2p + 1.
static inline int8 P(nv_frag)(uint w0, uint w1, uint scale, float phi26) {
  const float k1 = P(e4m3)(scale) * 0x1p100f;
  int8 b;
#define INK2_NV_PAIR(j, w, i) \
  b[j] = (int)as_uint(intel_convert_bfloat162_as_ushort2((float2)((INK2_NIBBLE_T(w, i) * k1) * phi26, (INK2_NIBBLE_T(w, (i) + 1) * k1) * phi26)))
  INK2_NV_PAIR(0, w0, 0);
  INK2_NV_PAIR(1, w0, 2);
  INK2_NV_PAIR(2, w0, 4);
  INK2_NV_PAIR(3, w0, 6);
  INK2_NV_PAIR(4, w1, 0);
  INK2_NV_PAIR(5, w1, 2);
  INK2_NV_PAIR(6, w1, 4);
  INK2_NV_PAIR(7, w1, 6);
#undef INK2_NV_PAIR
  return b;
}

// MXFP8: e4m3(code) * 2^(scale - 127) is exact in FP32 and BF16. The code's bits under an FP32 exponent are
// the E4M3 value * 2^-120, so the scale factor is 2^(scale - 7), split in two when it would overflow.
static inline float P(mx_value)(uint code, float f1, float f2) {
  const uint c7 = code & 0x7fu;
  const float t = as_float((c7 << 20) | ((code & 0x80u) << 24));
  const float v = (t * f1) * f2;
  return c7 == 0x7fu ? NAN : v;
}

static inline void P(mx_factors)(uint scale, float *f1, float *f2) {
#ifdef INK2_PLANT_SCALE
  scale ^= 1u;
#endif
  const int e = (int)scale - 7; // 2^e = 2^(e >> 1) * 2^(e - (e >> 1))
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

// Both GEMM results round to BF16 before the soft-capped SwiGLU; the sigmoid reads the rounded gate.
static inline ushort P(act)(float gate_dot, float up_dot) {
  const float gate = P(bf_to_f)(P(f_to_bf)(gate_dot));
  const float up = P(bf_to_f)(P(f_to_bf)(up_dot));
  const float left = P(cap)(gate, 4.0f) * (1.0f / (1.0f + exp(-gate)));
  return P(f_to_bf)(left * P(cap)(up, 25.0f));
}

// ---- decode path: the routes of up to 16 rows, one DPAS row ------------------------------------------------------

#define INK2_DOT(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c)

// A decode work-group owns one expert of the call: the group of the expert's first route lists the routes that
// pick it (up to INK2_MAX_MEMBERS, in route order), and the groups of the later routes leave at once. The weights of
// each k slice are decoded once and every member route runs its own M=1 DPAS over them, in the same k order and with
// the same sum over sub-groups as a group of its own would, so each route's bits do not depend on the other rows.
#define INK2_MAX_MEMBERS 256
#define INK2_RB 4 // routes a pass over the weights serves

// Returns the member count (0: an earlier route has this expert). mem[0..count) are the member routes.
static inline uint P(members)(__global const int *ids, uint route, uint routes, local uint *mem, local uint *count) {
  const uint lane = get_sub_group_local_id();
  const int e = ids[route];
  uint dup = 0;
  for (uint p = lane; p < route; p += INK2_SG) dup |= ids[p] == e ? 1u : 0u;
  if (sub_group_reduce_max(dup) != 0) return 0;
  if (get_sub_group_id() == 0) {
    uint cnt = 1;
    if (lane == 0) mem[0] = route;
    for (uint base = route + 1; base < routes; base += INK2_SG) {
      const uint p = base + lane;
      const bool hit = p < routes && ids[p] == e;
      const uint bits = sub_group_reduce_add(hit ? (1u << lane) : 0u);
      const uint pos = cnt + popcount(bits & ((1u << lane) - 1u));
      if (hit && pos < INK2_MAX_MEMBERS) mem[pos] = p;
      cnt += popcount(bits);
    }
    if (lane == 0) *count = min(cnt, (uint)INK2_MAX_MEMBERS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return *count;
}

// Sums the slices of k the work-group's sub-groups hold, in sub-group order, and writes the results of 16 rows of
// one route.
static inline void P(finish)(const uint ks, local float *part, __global ushort *Y, uint route, uint n, uint tile, uint rows_total, bool gu) {
  const uint lane = get_sub_group_local_id();
  float sum = part[lane];
#pragma unroll
  for (uint s = 1; s < ks; s++) sum += part[s * INK2_SG + lane];
  const uint row = tile * INK2_SG + lane;
  if (gu) {
    const float other = sub_group_shuffle_xor(sum, 1);
#ifdef INK2_PLANT_UP
    const float up = sum;
#else
    const float up = other;
#endif
    if ((lane & 1u) == 0 && row < rows_total) Y[(ulong)route * n + (row >> 1)] = P(act)(sum, up);
  } else if (row < rows_total) {
    Y[(ulong)route * n + row] = P(f_to_bf)(sum);
  }
}

// part holds INK2_RB routes of ks sub-group slices; sub-group 0 then sums and writes each route in turn.
static inline void P(finish_all)(const uint ks, float4 acc, local float *part, __global ushort *Y, local const uint *mem,
                                 uint first, uint count, uint n, uint tile, uint rows_total, bool gu) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  part[(0 * ks + sg) * INK2_SG + lane] = acc.s0;
  part[(1 * ks + sg) * INK2_SG + lane] = acc.s1;
  part[(2 * ks + sg) * INK2_SG + lane] = acc.s2;
  part[(3 * ks + sg) * INK2_SG + lane] = acc.s3;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == 0) {
#pragma unroll
    for (uint r = 0; r < INK2_RB; r++)
      if (first + r < count) P(finish)(ks, part + r * ks * INK2_SG, Y, mem[first + r], n, tile, rows_total, gu);
  }
  if (first + INK2_RB < count) barrier(CLK_LOCAL_MEM_FENCE); // before the next pass writes part again
}

// One pass over the expert's k slice of this sub-group for `live` (a literal at every call) member routes.
static inline __attribute__((always_inline)) float4 P(nv_pass)(const uint live, const uint ks, const uint sg,
                                                               __global const uchar *wr, __global const uchar *sr, const uint nb,
                                                               const float phi26, __global const ushort *x0, __global const ushort *x1,
                                                               __global const ushort *x2, __global const ushort *x3) {
  float4 acc = 0.0f;
    if ((nb & 7u) == 0) { // whole 64-byte lines a row: eight blocks, one line a lane
      const uint chunks = nb / 8u;
      for (uint c = sg; c < chunks; c += ks) {
        const uint16 v = vload16(0, (__global const uint *)(wr + 64u * c));
        const uint2 s2 = vload2(0, (__global const uint *)(sr + 8u * c));
#pragma unroll
        for (int u = 0; u < 8; u++) {
          const uint sc = (s2[u >> 2] >> (8 * (u & 3))) & 255u;
          const int8 b = P(nv_frag)(v[2 * u], v[2 * u + 1], sc, phi26);
          const uint at = 128u * c + 16u * u;
          acc.s0 = INK2_DOT((short)intel_sub_group_block_read_us(x0 + at), b, acc.s0);
          if (live > 1) acc.s1 = INK2_DOT((short)intel_sub_group_block_read_us(x1 + at), b, acc.s1);
          if (live > 2) acc.s2 = INK2_DOT((short)intel_sub_group_block_read_us(x2 + at), b, acc.s2);
          if (live > 3) acc.s3 = INK2_DOT((short)intel_sub_group_block_read_us(x3 + at), b, acc.s3);
        }
      }
    } else {
      for (uint blk = sg; blk < nb; blk += ks) {
        const uint2 v = vload2(0, (__global const uint *)(wr + 8u * blk));
        const int8 b = P(nv_frag)(v.x, v.y, (uint)sr[blk], phi26);
        const uint at = 16u * blk;
        acc.s0 = INK2_DOT((short)intel_sub_group_block_read_us(x0 + at), b, acc.s0);
        if (live > 1) acc.s1 = INK2_DOT((short)intel_sub_group_block_read_us(x1 + at), b, acc.s1);
        if (live > 2) acc.s2 = INK2_DOT((short)intel_sub_group_block_read_us(x2 + at), b, acc.s2);
        if (live > 3) acc.s3 = INK2_DOT((short)intel_sub_group_block_read_us(x3 + at), b, acc.s3);
      }
    }
  return acc;
}

#define INK2_XROW(r) (X + (ulong)(gu ? mem[first + (r)] / topk : mem[first + (r)]) * k)

static inline void P(nv_decode_rows)(const uint ks, __global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const float *amax, __global const float *den, __global const int *ids,
                                __global ushort *Y, uint k, uint n, uint topk, bool gu, local float *part,
                                local uint *mem) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile = get_group_id(0), route = get_group_id(1);
  const uint count = P(members)(ids, route, get_num_groups(1), mem, mem + INK2_MAX_MEMBERS);
  if (count == 0) return;
  const uint e = (uint)ids[route];
  const uint rows_total = gu ? 2u * n : n;
  const uint raw = tile * INK2_SG + lane;
  const uint row = raw < rows_total ? raw : 0u;
  const uint nb = k / 16u;
  __global const uchar *wr = W + ((ulong)e * rows_total + row) * (k / 2u);
  __global const uchar *sr = S + ((ulong)e * rows_total + row) * nb;
  const float phi26 = P(phi)(amax, den, e) * 0x1p26f;
  for (uint first = 0; first < count; first += INK2_RB) {
    __global const ushort *x0 = INK2_XROW(0);
    __global const ushort *x1 = first + 1 < count ? INK2_XROW(1) : x0;
    __global const ushort *x2 = first + 2 < count ? INK2_XROW(2) : x0;
    __global const ushort *x3 = first + 3 < count ? INK2_XROW(3) : x0;
    const uint live = min(count - first, (uint)INK2_RB);
    float4 acc;
    switch (live) {
      case 1: acc = P(nv_pass)(1, ks, sg, wr, sr, nb, phi26, x0, x1, x2, x3); break;
      case 2: acc = P(nv_pass)(2, ks, sg, wr, sr, nb, phi26, x0, x1, x2, x3); break;
      case 3: acc = P(nv_pass)(3, ks, sg, wr, sr, nb, phi26, x0, x1, x2, x3); break;
      default: acc = P(nv_pass)(4, ks, sg, wr, sr, nb, phi26, x0, x1, x2, x3); break;
    }
    P(finish_all)(ks, acc, part, Y, mem, first, count, n, tile, rows_total, gu);
  }
}

static inline void P(mx_decode_rows)(const uint ks, __global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const int *ids, __global ushort *Y, uint k, uint n, uint topk, bool gu,
                                local float *part, local uint *mem) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile = get_group_id(0), route = get_group_id(1);
  const uint count = P(members)(ids, route, get_num_groups(1), mem, mem + INK2_MAX_MEMBERS);
  if (count == 0) return;
  const uint e = (uint)ids[route];
  const uint rows_total = gu ? 2u * n : n;
  const uint raw = tile * INK2_SG + lane;
  const uint row = raw < rows_total ? raw : 0u;
  const uint nb = k / 32u;
  __global const uchar *wr = W + ((ulong)e * rows_total + row) * k;
  __global const uchar *sr = S + ((ulong)e * rows_total + row) * nb;
  for (uint first = 0; first < count; first += INK2_RB) {
    __global const ushort *x0 = INK2_XROW(0);
    __global const ushort *x1 = first + 1 < count ? INK2_XROW(1) : x0;
    __global const ushort *x2 = first + 2 < count ? INK2_XROW(2) : x0;
    __global const ushort *x3 = first + 3 < count ? INK2_XROW(3) : x0;
    const uint live = min(count - first, (uint)INK2_RB);
    float4 acc = 0.0f;
    for (uint blk = sg; blk < nb; blk += ks) {
      const uint8 v = vload8(0, (__global const uint *)(wr + 32u * blk));
      float f1, f2;
      P(mx_factors)((uint)sr[blk], &f1, &f2);
      const int8 b0 = P(mx_frag)(v.s0123, f1, f2);
      const int8 b1 = P(mx_frag)(v.s4567, f1, f2);
      const uint at = 32u * blk;
      acc.s0 = INK2_DOT((short)intel_sub_group_block_read_us(x0 + at), b0, acc.s0);
      acc.s0 = INK2_DOT((short)intel_sub_group_block_read_us(x0 + at + 16u), b1, acc.s0);
      if (live > 1) {
        acc.s1 = INK2_DOT((short)intel_sub_group_block_read_us(x1 + at), b0, acc.s1);
        acc.s1 = INK2_DOT((short)intel_sub_group_block_read_us(x1 + at + 16u), b1, acc.s1);
      }
      if (live > 2) {
        acc.s2 = INK2_DOT((short)intel_sub_group_block_read_us(x2 + at), b0, acc.s2);
        acc.s2 = INK2_DOT((short)intel_sub_group_block_read_us(x2 + at + 16u), b1, acc.s2);
      }
      if (live > 3) {
        acc.s3 = INK2_DOT((short)intel_sub_group_block_read_us(x3 + at), b0, acc.s3);
        acc.s3 = INK2_DOT((short)intel_sub_group_block_read_us(x3 + at + 16u), b1, acc.s3);
      }
    }
    P(finish_all)(ks, acc, part, Y, mem, first, count, n, tile, rows_total, gu);
  }
}

// One row of a call (every route has its own expert): a work-group of its own for each route, as before the grouping above.
// Sums the slices of k the work-group's sub-groups hold, in sub-group order, and writes the results of 16 rows.
static inline void P(finish_one)(const uint ks, float acc, local float *part, __global ushort *Y, uint route, uint n, uint tile, uint rows_total, bool gu) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  part[sg * INK2_SG + lane] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg != 0) return;
  float sum = part[lane];
#pragma unroll
  for (uint s = 1; s < ks; s++) sum += part[s * INK2_SG + lane];
  const uint row = tile * INK2_SG + lane;
  if (gu) {
    const float other = sub_group_shuffle_xor(sum, 1);
#ifdef INK2_PLANT_UP
    const float up = sum;
#else
    const float up = other;
#endif
    if ((lane & 1u) == 0 && row < rows_total) Y[(ulong)route * n + (row >> 1)] = P(act)(sum, up);
  } else if (row < rows_total) {
    Y[(ulong)route * n + row] = P(f_to_bf)(sum);
  }
}

static inline void P(nv_decode_one)(const uint ks, __global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const float *amax, __global const float *den, __global const int *ids,
                                __global ushort *Y, uint k, uint n, uint topk, bool gu, local float *part) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile = get_group_id(0), route = get_group_id(1);
  const uint e = (uint)ids[route];
  const uint rows_total = gu ? 2u * n : n;
  const uint raw = tile * INK2_SG + lane;
  if (raw >= rows_total) return;
  const uint row = raw;
  const uint nb = k / 16u;
  __global const uchar *wr = W + ((ulong)e * rows_total + row) * (k / 2u);
  __global const uchar *sr = S + ((ulong)e * rows_total + row) * nb;
  __global const ushort *xr = X + (ulong)(gu ? route / topk : route) * k;
  const float phi26 = P(phi)(amax, den, e) * 0x1p26f;
  float acc = 0.0f;
  if ((nb & 7u) == 0) { // whole 64-byte lines a row: eight blocks, one line a lane
    const uint chunks = nb / 8u;
    for (uint c = sg; c < chunks; c += ks) {
      const uint16 v = vload16(0, (__global const uint *)(wr + 64u * c));
      const uint2 s2 = vload2(0, (__global const uint *)(sr + 8u * c));
#pragma unroll
      for (int u = 0; u < 8; u++) {
        const uint sc = (s2[u >> 2] >> (8 * (u & 3))) & 255u;
        const int8 b = P(nv_frag)(v[2 * u], v[2 * u + 1], sc, phi26);
        const short a = (short)intel_sub_group_block_read_us(xr + 128u * c + 16u * u);
        acc = INK2_DOT(a, b, acc);
      }
    }
  } else {
    for (uint blk = sg; blk < nb; blk += ks) {
      const uint2 v = vload2(0, (__global const uint *)(wr + 8u * blk));
      const int8 b = P(nv_frag)(v.x, v.y, (uint)sr[blk], phi26);
      const short a = (short)intel_sub_group_block_read_us(xr + 16u * blk);
      acc = INK2_DOT(a, b, acc);
    }
  }
  P(finish_one)(ks, acc, part, Y, route, n, tile, rows_total, gu);
}

static inline void P(mx_decode_one)(const uint ks, __global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const int *ids, __global ushort *Y, uint k, uint n, uint topk, bool gu,
                                local float *part) {
  const uint lane = get_sub_group_local_id();
  const uint sg = get_sub_group_id();
  const uint tile = get_group_id(0), route = get_group_id(1);
  const uint e = (uint)ids[route];
  const uint rows_total = gu ? 2u * n : n;
  const uint raw = tile * INK2_SG + lane;
  const uint row = raw < rows_total ? raw : 0u;
  const uint nb = k / 32u;
  __global const uchar *wr = W + ((ulong)e * rows_total + row) * k;
  __global const uchar *sr = S + ((ulong)e * rows_total + row) * nb;
  __global const ushort *xr = X + (ulong)(gu ? route / topk : route) * k;
  float acc = 0.0f;
  for (uint blk = sg; blk < nb; blk += ks) {
    const uint8 v = vload8(0, (__global const uint *)(wr + 32u * blk));
    float f1, f2;
    P(mx_factors)((uint)sr[blk], &f1, &f2);
    const int8 b0 = P(mx_frag)(v.s0123, f1, f2);
    const int8 b1 = P(mx_frag)(v.s4567, f1, f2);
    const short a0 = (short)intel_sub_group_block_read_us(xr + 32u * blk);
    const short a1 = (short)intel_sub_group_block_read_us(xr + 32u * blk + 16u);
    acc = INK2_DOT(a0, b0, acc);
    acc = INK2_DOT(a1, b1, acc);
  }
  P(finish_one)(ks, acc, part, Y, route, n, tile, rows_total, gu);
}

static inline void P(nv_decode)(const uint ks, __global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const float *amax, __global const float *den, __global const int *ids,
                                __global ushort *Y, uint k, uint n, uint topk, bool gu, local float *part, local uint *mem) {
  if (get_num_groups(1) <= topk) P(nv_decode_one)(ks, X, W, S, amax, den, ids, Y, k, n, topk, gu, part);
  else P(nv_decode_rows)(ks, X, W, S, amax, den, ids, Y, k, n, topk, gu, part, mem);
}

static inline void P(mx_decode)(const uint ks, __global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const int *ids, __global ushort *Y, uint k, uint n, uint topk, bool gu,
                                local float *part, local uint *mem) {
  if (get_num_groups(1) <= topk) P(mx_decode_one)(ks, X, W, S, ids, Y, k, n, topk, gu, part);
  else P(mx_decode_rows)(ks, X, W, S, ids, Y, k, n, topk, gu, part, mem);
}

#if INK2_ONLY & INK2_NV_GU
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * INK2_KS_NV_GU, 1, 1)))
__kernel void P(nvfp4_gu)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                          __global const float *amax, __global const float *den, __global const int *ids,
                          __global ushort *Y, uint k, uint n, uint topk, uint experts) {
  local float part[INK2_RB * INK2_KS_NV_GU * INK2_SG];
  local uint mem[INK2_MAX_MEMBERS + 1];
  P(nv_decode)(INK2_KS_NV_GU, X, W, S, amax, den, ids, Y, k, n, topk, true, part, mem);
}
#endif

#if INK2_ONLY & INK2_NV_DOWN
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * INK2_KS_NV_DOWN, 1, 1)))
__kernel void P(nvfp4_down)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                            __global const float *amax, __global const float *den, __global const int *ids,
                            __global ushort *Y, uint k, uint n, uint topk, uint experts) {
  local float part[INK2_RB * INK2_KS_NV_DOWN * INK2_SG];
  local uint mem[INK2_MAX_MEMBERS + 1];
  P(nv_decode)(INK2_KS_NV_DOWN, X, W, S, amax, den, ids, Y, k, n, topk, false, part, mem);
}
#endif

#if INK2_ONLY & INK2_MX_GU
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * INK2_KS_MX_GU, 1, 1)))
__kernel void P(mxfp8_gu)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                          __global const int *ids, __global ushort *Y, uint k, uint n, uint topk, uint experts) {
  local float part[INK2_RB * INK2_KS_MX_GU * INK2_SG];
  local uint mem[INK2_MAX_MEMBERS + 1];
  P(mx_decode)(INK2_KS_MX_GU, X, W, S, ids, Y, k, n, topk, true, part, mem);
}
#endif

#if INK2_ONLY & INK2_MX_DOWN
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * INK2_KS_MX_DOWN, 1, 1)))
__kernel void P(mxfp8_down)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                            __global const int *ids, __global ushort *Y, uint k, uint n, uint topk, uint experts) {
  local float part[INK2_RB * INK2_KS_MX_DOWN * INK2_SG];
  local uint mem[INK2_MAX_MEMBERS + 1];
  P(mx_decode)(INK2_KS_MX_DOWN, X, W, S, ids, Y, k, n, topk, false, part, mem);
}
#endif

// ---- weight decode helpers: checkpoint bytes to BF16, never an activation quantizer ---------------------------------

#if INK2_ONLY & INK2_NV_QUANT
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void P(nvfp4_quantize)(__global const uchar *W, __global const uchar *S, __global const float *amax,
                                __global const float *den, __global ushort *Y, uint rows, uint k) {
  const ulong at = get_global_id(0);
  if (at >= (ulong)rows * k) return;
  const ulong row = at / k;
  uint col = (uint)(at - row * k);
  const float phi26 = P(phi)(amax, den, 0) * 0x1p26f;
  const uint byte = W[row * (k / 2u) + col / 2u];
#ifdef INK2_PLANT_NIBBLE
  col += 1u;
#endif
  const uint nibble = (col & 1u) ? byte >> 4 : byte & 15u;
  const float k1 = P(e4m3)((uint)S[row * (k / 16u) + col / 16u]) * 0x1p100f;
  Y[at] = P(f_to_bf)(P(nv_value)(nibble, k1, phi26));
}

#endif

#if INK2_ONLY & INK2_MX_QUANT
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void P(mxfp8_quantize)(__global const uchar *W, __global const uchar *S, __global ushort *Y, uint rows,
                                uint k) {
  const ulong at = get_global_id(0);
  if (at >= (ulong)rows * k) return;
  const ulong row = at / k;
  const uint col = (uint)(at - row * k);
#ifdef INK2_PLANT_MX_QUANT
  const uint scale = S[row * (k / 32u) + col / 16u];
#else
  const uint scale = S[row * (k / 32u) + col / 32u];
#endif
  float f1, f2;
  P(mx_factors)(scale, &f1, &f2);
  Y[at] = P(f_to_bf)(P(mx_value)((uint)W[at], f1, f2));
}
#endif

// ---- combine: the route outputs of a token weigh in ascending expert order, then round once to BF16 -----------------

#if INK2_ONLY & INK2_COMBINE
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void P(combine)(__global const ushort *down, __global const int *ids, __global const float *weights,
                         __global ushort *Y, uint n, uint topk, uint experts) {
  const uint col = get_global_id(0), token = get_group_id(1);
  if (col >= n) return;
  float sum = 0.0f;
#ifdef INK2_PLANT_ORDER
  for (uint slot = 0; slot < topk; slot++)
    sum += weights[token * topk + slot] * P(bf_to_f)(down[(ulong)(token * topk + slot) * n + col]);
#else
  // Outer expert order, then slot order, fixes the accumulation order: the next route is the least (expert, slot)
  // past the last one taken.
  int last_id = INT_MIN;
  int last_slot = -1;
  for (uint t = 0; t < topk; t++) {
    int best_id = INT_MAX;
    int best_slot = -1;
    for (uint slot = 0; slot < topk; slot++) {
      const int id = ids[token * topk + slot];
      const bool after = id > last_id || (id == last_id && (int)slot > last_slot);
      if (after && (best_slot < 0 || id < best_id)) {
        best_id = id;
        best_slot = (int)slot;
      }
    }
    if (best_slot < 0) break;
    sum += weights[token * topk + best_slot] * P(bf_to_f)(down[(ulong)(token * topk + best_slot) * n + col]);
    last_id = best_id;
    last_slot = best_slot;
  }
#endif
  Y[(ulong)token * n + col] = P(f_to_bf)(sum);
}
#endif

// ---- prompt path ---------------------------------------------------------------------------------------------------

// Groups routes by expert, deterministically: members lists each expert's routes in route order, items lists
// (expert, first member, count) runs of at most `run` members, in expert order. One work-group.
#if INK2_ONLY & INK2_PLAN
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void P(plan)(__global const int *ids, __global int *members, __global int *items, __global int *item_count,
                      uint routes, uint experts, uint run) {
  local int counts[INK2_MAX_EXPERTS];
  local int offsets[INK2_MAX_EXPERTS + 1];
  const uint lid = get_local_id(0), lsz = get_local_size(0);
  for (uint e = lid; e < experts; e += lsz) counts[e] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint p = lid; p < routes; p += lsz) {
    const uint e = (uint)ids[p];
    if (e < experts) atomic_inc(&counts[e]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    int at = 0, made = 0;
    for (uint e = 0; e < experts; e++) {
      offsets[e] = at;
      for (int first = 0; first < counts[e]; first += (int)run) {
        items[3 * made] = (int)e;
        items[3 * made + 1] = at + first;
        items[3 * made + 2] = min((int)run, counts[e] - first);
        made++;
      }
      at += counts[e];
    }
    offsets[experts] = at;
    *item_count = made;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint e = lid; e < experts; e += lsz) {
    int at = offsets[e];
#ifdef INK2_PLANT_PLAN
    const int stop = offsets[e] + max(counts[e] - 1, 0); // the last member of each expert is never listed
#else
    const int stop = offsets[e] + counts[e];
#endif
    for (uint p = 0; p < routes && at < stop; p++)
      if ((uint)ids[p] == e) members[at++] = (int)p;
  }
}
#endif

// A run of members is 4 tiles of 8 DPAS rows; lane l holds the route of member l and of member l + 16.
#define INK2_PROMPT_SETUP                                                                                         \
  const uint lane = get_sub_group_local_id();                                                                     \
  const uint tile = get_group_id(0), item = get_group_id(1);                                                      \
  if (item >= (uint)item_count[0]) return;                                                                        \
  const uint e = (uint)items[3 * item], first = (uint)items[3 * item + 1], cnt = (uint)items[3 * item + 2];      \
  const uint rows_total = gu ? 2u * n : n;                                                                        \
  const uint raw = tile * INK2_SG + lane;                                                                         \
  const uint row = raw < rows_total ? raw : 0u;                                                                   \
  int route0 = lane < cnt ? members[first + lane] : -1;                                                           \
  int route1 = lane + 16u < cnt ? members[first + lane + 16u] : -1;                                               \
  const uint tiles = (cnt + 7u) / 8u;                                                                             \
  float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;                                                      \
  uint xo0 = route0 < 0 ? 0xffffffffu : (uint)(gu ? route0 / (int)topk : route0);                                 \
  uint xo1 = route1 < 0 ? 0xffffffffu : (uint)(gu ? route1 / (int)topk : route1);

// The A operand of 8 members at a k offset: the activation row of each member, lane l the column k0 + l.
#define INK2_A8(T, K0)                                                                                            \
  ({                                                                                                              \
    short8 a_ = (short8)(0);                                                                                      \
    _Pragma("unroll") for (int r_ = 0; r_ < 8; r_++) {                                                            \
      const uint m_ = (T) * 8 + r_;                                                                               \
      const uint xo_ = sub_group_broadcast((m_ >> 4) ? xo1 : xo0, m_ & 15u);                                      \
      a_[r_] = xo_ == 0xffffffffu ? (short)0 : (short)X[(ulong)xo_ * k + (K0) + lane];                            \
    }                                                                                                             \
    a_;                                                                                                           \
  })

#define INK2_STEP(B, K0)                                                                                          \
  acc0 = intel_sub_group_bf16_bf16_matrix_mad_k16(INK2_A8(0, K0), B, acc0);                                      \
  if (tiles > 1) acc1 = intel_sub_group_bf16_bf16_matrix_mad_k16(INK2_A8(1, K0), B, acc1);                       \
  if (tiles > 2) acc2 = intel_sub_group_bf16_bf16_matrix_mad_k16(INK2_A8(2, K0), B, acc2);                       \
  if (tiles > 3) acc3 = intel_sub_group_bf16_bf16_matrix_mad_k16(INK2_A8(3, K0), B, acc3);

// Writes the member results of each 8-row tile: a GU output column is the activation of rows (2c, 2c + 1).
static inline void P(prompt_store)(float8 acc, uint t, uint cnt, int route0, int route1, __global ushort *Y, uint n,
                                   uint raw, uint rows_total, bool gu) {
  const uint lane = get_sub_group_local_id();
#pragma unroll
  for (int r = 0; r < 8; r++) {
    const uint m = t * 8 + r;
    const int route = sub_group_broadcast((m >> 4) ? route1 : route0, m & 15u);
    const float v = acc[r];
    if (gu) {
      const float other = sub_group_shuffle_xor(v, 1);
#ifdef INK2_PLANT_UP
      const float up = v;
#else
      const float up = other;
#endif
      if (m < cnt && (lane & 1u) == 0 && raw < rows_total) Y[(ulong)route * n + (raw >> 1)] = P(act)(v, up);
    } else if (m < cnt && raw < rows_total) {
      Y[(ulong)route * n + raw] = P(f_to_bf)(v);
    }
  }
}

static inline void P(nv_prompt)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const float *amax, __global const float *den, __global const int *items,
                                __global const int *item_count, __global const int *members, __global ushort *Y,
                                uint k, uint n, uint topk, bool gu) {
  INK2_PROMPT_SETUP
  const uint nb = k / 16u;
  __global const uchar *wr = W + ((ulong)e * rows_total + row) * (k / 2u);
  __global const uchar *sr = S + ((ulong)e * rows_total + row) * nb;
  const float phi26 = P(phi)(amax, den, e) * 0x1p26f;
  for (uint blk = 0; blk < nb; blk++) {
    const uint2 v = vload2(0, (__global const uint *)(wr + 8u * blk));
    const int8 b = P(nv_frag)(v.x, v.y, (uint)sr[blk], phi26);
    INK2_STEP(b, 16u * blk)
  }
  P(prompt_store)(acc0, 0, cnt, route0, route1, Y, n, raw, rows_total, gu);
  if (tiles > 1) P(prompt_store)(acc1, 1, cnt, route0, route1, Y, n, raw, rows_total, gu);
  if (tiles > 2) P(prompt_store)(acc2, 2, cnt, route0, route1, Y, n, raw, rows_total, gu);
  if (tiles > 3) P(prompt_store)(acc3, 3, cnt, route0, route1, Y, n, raw, rows_total, gu);
}

static inline void P(mx_prompt)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                                __global const int *items, __global const int *item_count,
                                __global const int *members, __global ushort *Y, uint k, uint n, uint topk, bool gu) {
  INK2_PROMPT_SETUP
  const uint nb = k / 32u;
  __global const uchar *wr = W + ((ulong)e * rows_total + row) * k;
  __global const uchar *sr = S + ((ulong)e * rows_total + row) * nb;
  for (uint blk = 0; blk < nb; blk++) {
    const uint8 v = vload8(0, (__global const uint *)(wr + 32u * blk));
    float f1, f2;
    P(mx_factors)((uint)sr[blk], &f1, &f2);
    const int8 b0 = P(mx_frag)(v.s0123, f1, f2);
    const int8 b1 = P(mx_frag)(v.s4567, f1, f2);
    INK2_STEP(b0, 32u * blk)
    INK2_STEP(b1, 32u * blk + 16u)
  }
  P(prompt_store)(acc0, 0, cnt, route0, route1, Y, n, raw, rows_total, gu);
  if (tiles > 1) P(prompt_store)(acc1, 1, cnt, route0, route1, Y, n, raw, rows_total, gu);
  if (tiles > 2) P(prompt_store)(acc2, 2, cnt, route0, route1, Y, n, raw, rows_total, gu);
  if (tiles > 3) P(prompt_store)(acc3, 3, cnt, route0, route1, Y, n, raw, rows_total, gu);
}

#if INK2_ONLY & INK2_NV_GU_PROMPT
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void P(nvfp4_gu_prompt)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                                 __global const float *amax, __global const float *den, __global const int *items,
                                 __global const int *item_count, __global const int *members, __global ushort *Y,
                                 uint k, uint n, uint topk) {
  P(nv_prompt)(X, W, S, amax, den, items, item_count, members, Y, k, n, topk, true);
}
#endif

#if INK2_ONLY & INK2_NV_DOWN_PROMPT
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void P(nvfp4_down_prompt)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                                   __global const float *amax, __global const float *den, __global const int *items,
                                   __global const int *item_count, __global const int *members, __global ushort *Y,
                                   uint k, uint n, uint topk) {
  P(nv_prompt)(X, W, S, amax, den, items, item_count, members, Y, k, n, topk, false);
}
#endif

#if INK2_ONLY & INK2_MX_GU_PROMPT
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void P(mxfp8_gu_prompt)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                                 __global const int *items, __global const int *item_count,
                                 __global const int *members, __global ushort *Y, uint k, uint n, uint topk) {
  P(mx_prompt)(X, W, S, items, item_count, members, Y, k, n, topk, true);
}
#endif

#if INK2_ONLY & INK2_MX_DOWN_PROMPT
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void P(mxfp8_down_prompt)(__global const ushort *X, __global const uchar *W, __global const uchar *S,
                                   __global const int *items, __global const int *item_count,
                                   __global const int *members, __global ushort *Y, uint k, uint n, uint topk) {
  P(mx_prompt)(X, W, S, items, item_count, members, Y, k, n, topk, false);
}
#endif

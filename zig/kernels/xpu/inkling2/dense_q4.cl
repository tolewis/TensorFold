// Inkling 2 Mini's dense projections on Intel Xe2 with Q4 blocks (zig/src/families/inkling2/xpu/q4.zig): signed 4-bit
// codes and one fp16 scale for every 16 values. Same split as dense_q8.cl, half the code bytes. A sub-group of 16
// owns one output column and reads its row once for every row of x. Nibbles are widened once per plane, then each
// row of x reuses them, so an 8-row verify does not pay the unpack eight times.
//
// y[r][j] = bf16(x[r] . dequant(W)[j]), FP32 sums, one bf16 store. Build with -cl-fp32-correctly-rounded-divide-sqrt.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#ifndef Q4_WG
#define Q4_WG 128
#endif
#ifndef Q4_UNROLL
#define Q4_UNROLL 9
#endif
#ifndef Q4_TAIL
#define Q4_TAIL 3
#endif

#define Q4_KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float q4_bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort q4_f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); }

#ifndef Q4_NAME
#define Q4_NAME(x) tf_ink2_##x
#endif
#ifndef Q4_ACC
#define Q4_ACC(acc, s, part) fma((s), (part), (acc))
#endif
#ifndef Q4_SCALE
#define Q4_SCALE(s, i) ((s)[i])
#endif
#ifndef Q4_STORE
#define Q4_STORE(v) q4_f2bf(v)
#endif

// One byte holds the lane's two signed nibbles of a block. Sign-extend, then the same pair product as dense_q8.cl.
#define Q4_NIB(byte, lo, hi)                                                                                       \
  {                                                                                                                \
    const int b_ = (int)(byte);                                                                                    \
    lo = (float)(b_ << 28 >> 28);                                                                                  \
    hi = (float)(b_ << 24 >> 28);                                                                                  \
  }

#define Q4_BLOCK(acc, w0, w1, u, s)                                                                                \
  {                                                                                                                \
    const float part = fma((w1), as_float((u) & 0xffff0000u), (w0) * as_float((u) << 16));                          \
    acc = Q4_ACC(acc, (s), part);                                                                                  \
  }

// Two fp16 scales a block of 32 (groups of 16). Lane l owns element 2l, so lanes 0 to 7 take the first.
#define Q4_PICK(pair) ((get_sub_group_local_id() < 8u) ? (pair).x : (pair).y)
#define Q4_SCALES(s, sc0, sc1)                                                                                     \
  float s[8];                                                                                                      \
  {                                                                                                                \
    const float2 a0 = convert_float2(as_half2(sc0.x)), b0 = convert_float2(as_half2(sc0.y));                       \
    const float2 c0 = convert_float2(as_half2(sc0.z)), d0 = convert_float2(as_half2(sc0.w));                       \
    const float2 a1 = convert_float2(as_half2(sc1.x)), b1 = convert_float2(as_half2(sc1.y));                       \
    const float2 c1 = convert_float2(as_half2(sc1.z)), d1 = convert_float2(as_half2(sc1.w));                       \
    s[0] = Q4_PICK(a0); s[1] = Q4_PICK(b0); s[2] = Q4_PICK(c0); s[3] = Q4_PICK(d0);                               \
    s[4] = Q4_PICK(a1); s[5] = Q4_PICK(b1); s[6] = Q4_PICK(c1); s[7] = Q4_PICK(d1);                               \
  }

// Widen the lane's eight bytes once, then every row of x uses those weights.
#define Q4_PLANE(ROWS, acc, packed, sc0, sc1, xw, xs, plane)                                                        \
  {                                                                                                                \
    float w0[8], w1[8];                                                                                            \
    const uchar8 nb = as_uchar8(packed);                                                                           \
    Q4_NIB(nb.s0, w0[0], w1[0]) Q4_NIB(nb.s1, w0[1], w1[1])                                                       \
    Q4_NIB(nb.s2, w0[2], w1[2]) Q4_NIB(nb.s3, w0[3], w1[3])                                                       \
    Q4_NIB(nb.s4, w0[4], w1[4]) Q4_NIB(nb.s5, w0[5], w1[5])                                                       \
    Q4_NIB(nb.s6, w0[6], w1[6]) Q4_NIB(nb.s7, w0[7], w1[7])                                                       \
    Q4_SCALES(s, sc0, sc1)                                                                                         \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                            \
      const uint4 xa = intel_sub_group_block_read4(xw + r * xs + (plane) * 128);                                   \
      const uint4 xb = intel_sub_group_block_read4(xw + r * xs + (plane) * 128 + 64);                              \
      Q4_BLOCK(acc[r], w0[0], w1[0], xa.x, Q4_SCALE(s, 0))                                                         \
      Q4_BLOCK(acc[r], w0[1], w1[1], xa.y, Q4_SCALE(s, 1))                                                         \
      Q4_BLOCK(acc[r], w0[2], w1[2], xa.z, Q4_SCALE(s, 2))                                                         \
      Q4_BLOCK(acc[r], w0[3], w1[3], xa.w, Q4_SCALE(s, 3))                                                         \
      Q4_BLOCK(acc[r], w0[4], w1[4], xb.x, Q4_SCALE(s, 4))                                                         \
      Q4_BLOCK(acc[r], w0[5], w1[5], xb.y, Q4_SCALE(s, 5))                                                         \
      Q4_BLOCK(acc[r], w0[6], w1[6], xb.z, Q4_SCALE(s, 6))                                                         \
      Q4_BLOCK(acc[r], w0[7], w1[7], xb.w, Q4_SCALE(s, 7))                                                         \
    }                                                                                                              \
  }

#define Q4_GROUP(ROWS, U, p)                                                                                       \
  {                                                                                                                \
    uint2 wv[U];                                                                                                   \
    uint4 sc0[U], sc1[U];                                                                                          \
    _Pragma("unroll") for (uint u = 0; u < (U); ++u) {                                                             \
      wv[u] = wr[((p) + u) * 16 + lane];                                                                           \
      sc0[u] = sr[2 * ((p) + u)];                                                                                  \
      sc1[u] = sr[2 * ((p) + u) + 1];                                                                              \
    }                                                                                                              \
    _Pragma("unroll") for (uint u = 0; u < (U); ++u) Q4_PLANE(ROWS, acc, wv[u], sc0[u], sc1[u], xw, xs, (p) + u)  \
    (p) += (U);                                                                                                    \
  }

#define Q4_DOT(ROWS, x, codes, scales, j, k)                                                                       \
  {                                                                                                                \
    const uint planes = (k) / 256, lane = get_sub_group_local_id();                                                \
    global const uint2 *wr = (global const uint2 *)((codes) + (ulong)(j) * ((k) / 2));                             \
    global const uint4 *sr = (global const uint4 *)((scales) + (ulong)(j) * ((k) / 16));                           \
    global const uint *xw = (global const uint *)(x);                                                              \
    const ulong xs = (k) / 2;                                                                                      \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;                                               \
    uint p = 0;                                                                                                    \
    for (; p + Q4_UNROLL <= planes;) Q4_GROUP(ROWS, Q4_UNROLL, p)                                                  \
    for (; p + Q4_TAIL <= planes;) Q4_GROUP(ROWS, Q4_TAIL, p)                                                      \
    for (; p < planes;) Q4_GROUP(ROWS, 1, p)                                                                       \
  }

#define Q4_MATVEC(ROWS)                                                                                            \
  Q4_KERNEL(Q4_WG) void Q4_NAME(dense_q4_##ROWS)(global const ushort *x, global const uchar *w, global ushort *y,   \
                                                  uint rows, uint n, uint k, uint ldy) {                           \
    const uint j = get_group_id(0) * (Q4_WG / 16) + get_sub_group_id();                                            \
    if (j >= n) return;                                                                                            \
    global const half *scales = (global const half *)(w + (ulong)n * (k / 2));                                     \
    float acc[ROWS];                                                                                               \
    Q4_DOT(ROWS, x, w, scales, j, k)                                                                               \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                            \
      const float s = sub_group_reduce_add(acc[r]);                                                                \
      if (get_sub_group_local_id() == 0 && r < rows) y[(ulong)r * ldy + j] = Q4_STORE(s);                          \
    }                                                                                                              \
  }

Q4_MATVEC(1)
Q4_MATVEC(2)
Q4_MATVEC(4)
Q4_MATVEC(8)
Q4_MATVEC(16)
Q4_MATVEC(32)

// Inkling 2 Mini's dense projections and unembedding head on Intel Xe2 (OpenCL C) with Q8_0-style weights: int8 codes and
// one fp16 scale for every 32 values of a row (zig/src/families/inkling2/xpu/q8.zig makes them from the checkpoint's bf16
// at load). Decode is bound by the bytes of the weights, and this format reads 17 bytes for every 16 bf16 would (1.0625
// bytes a weight against 2).
//
// y[rows][n] = bf16(x[rows][k] . dequant(W)[n][k]): x is bf16, a weight is code * scale, every sum is FP32, one bf16 store
// (round to nearest even), rows `ldy` apart. The weights are dequantized in registers; activations stay bf16, so the
// only change against the bf16 kernel (dense.cl) is the weights' own rounding.
//
// Work split. A sub-group of 16 owns one output column j and reads its row once for every row of x (1, 2, 4, 8, 16 or
// 32). A row of codes is k / 256 planes of 256 bytes (see q8.zig): in plane p lane l loads 16 bytes, its two values of each
// of the eight blocks 8p to 8p + 7. The activation's matching words come from a sub-group block read, so lane l holds
// values 2l and 2l + 1 of every block, and a block's scale is the same in all lanes:
//
//     acc += scale[b] * (code[2l] * x[2l] + code[2l + 1] * x[2l + 1])        for each block b of the row, per lane
//
// then the sub-group adds the 16 lanes' sums. Sums are FP32; the order is fixed (a lane's chain over blocks in order,
// then the sub-group tree), so a run repeats its bytes. Grid (ceil(n / (Q8_WG / 16))).
//
// The head is the same loop over the 200,058 real vocabulary rows with the logits epilogue of glue.cl's
// tf_inkling2_logits: round the dot to bf16, widen, soft cap 30.
//
// Build with -cl-fp32-correctly-rounded-divide-sqrt (the cap's divide and tanh).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#ifndef Q8_WG
#define Q8_WG 128 // work-items a work-group: Q8_WG / 16 output columns
#endif
#ifndef Q8_UNROLL
#define Q8_UNROLL 9 // planes a lane has in flight (a k = 2,304 row is nine)
#endif
#ifndef Q8_TAIL
#define Q8_TAIL 3 // planes in flight for what is left of a longer row
#endif
#define Q8_HIDDEN 2304

#define Q8_KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float q8_bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort q8_f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even
static inline float q8_cap(float x, float c) { return c * tanh(x / c); }

// The contract plants one mistake in each kernel through the macros below; production keeps these defaults.
// contract_dense_q8_controls.cl redefines them (and the names) before it includes this file.
#ifndef Q8_NAME
#define Q8_NAME(x) tf_ink2_##x
#endif
// A block's contribution added to the lane's running sum.
#ifndef Q8_ACC
#define Q8_ACC(acc, s, part) fma((s), (part), (acc))
#endif
// The block scale a block uses: s is the plane's eight scales, i the block in the plane.
#ifndef Q8_SCALE
#define Q8_SCALE(s, i) ((s)[i])
#endif
// The projection's one store.
#ifndef Q8_STORE
#define Q8_STORE(v) q8_f2bf(v)
#endif
#ifndef Q8_CAP_ROUND
#define Q8_CAP_ROUND(v) q8_bf2f(q8_f2bf(v))
#endif

// One block of the lane's two values: u holds the activation's pair (low half is the even value).
#define Q8_BLOCK(acc, w0, w1, u, s)                                                                                \
  {                                                                                                                \
    const float part = fma((w1), as_float((u) & 0xffff0000u), (w0) * as_float((u) << 16));                          \
    acc = Q8_ACC(acc, (s), part);                                                                                  \
  }

// Eight blocks of one plane against one row of x: wv the lane's 16 code bytes, s the eight scales, xa and xb the
// activation words of blocks 0 to 3 and 4 to 7.
#define Q8_PLANE_ROW(acc, wv, s, xa, xb)                                                                           \
  {                                                                                                                \
    const float4 f0 = convert_float4(as_char4(wv.x)), f1 = convert_float4(as_char4(wv.y));                         \
    const float4 f2 = convert_float4(as_char4(wv.z)), f3 = convert_float4(as_char4(wv.w));                         \
    Q8_BLOCK(acc, f0.x, f0.y, xa.x, Q8_SCALE(s, 0))                                                                \
    Q8_BLOCK(acc, f0.z, f0.w, xa.y, Q8_SCALE(s, 1))                                                                \
    Q8_BLOCK(acc, f1.x, f1.y, xa.z, Q8_SCALE(s, 2))                                                                \
    Q8_BLOCK(acc, f1.z, f1.w, xa.w, Q8_SCALE(s, 3))                                                                \
    Q8_BLOCK(acc, f2.x, f2.y, xb.x, Q8_SCALE(s, 4))                                                                \
    Q8_BLOCK(acc, f2.z, f2.w, xb.y, Q8_SCALE(s, 5))                                                                \
    Q8_BLOCK(acc, f3.x, f3.y, xb.z, Q8_SCALE(s, 6))                                                                \
    Q8_BLOCK(acc, f3.z, f3.w, xb.w, Q8_SCALE(s, 7))                                                                \
  }

#define Q8_SCALES(s, sc)                                                                                           \
  float s[8];                                                                                                      \
  {                                                                                                                \
    const float2 a = convert_float2(as_half2(sc.x)), b = convert_float2(as_half2(sc.y));                           \
    const float2 c = convert_float2(as_half2(sc.z)), d = convert_float2(as_half2(sc.w));                           \
    s[0] = a.x; s[1] = a.y; s[2] = b.x; s[3] = b.y; s[4] = c.x; s[5] = c.y; s[6] = d.x; s[7] = d.y;                \
  }

// `U` planes at once: every load of the group first, then the sums.
#define Q8_GROUP(ROWS, U, p)                                                                                       \
  {                                                                                                                \
    uint4 wv[U], sc[U];                                                                                            \
    _Pragma("unroll") for (uint u = 0; u < (U); ++u) {                                                             \
      wv[u] = wr[((p) + u) * 16 + lane];                                                                           \
      sc[u] = sr[(p) + u];                                                                                         \
    }                                                                                                              \
    _Pragma("unroll") for (uint u = 0; u < (U); ++u) {                                                             \
      Q8_SCALES(s, sc[u])                                                                                          \
      _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                          \
        const uint4 xa = intel_sub_group_block_read4(xw + r * xs + ((p) + u) * 128);                               \
        const uint4 xb = intel_sub_group_block_read4(xw + r * xs + ((p) + u) * 128 + 64);                          \
        Q8_PLANE_ROW(acc[r], wv[u], s, xa, xb)                                                                     \
      }                                                                                                            \
    }                                                                                                              \
    (p) += (U);                                                                                                    \
  }

// The lane's sums for output column j over ROWS rows of x (`k` values a row, a multiple of 256), into acc[ROWS]: groups
// of Q8_UNROLL planes, then of Q8_TAIL, then single planes.
#define Q8_DOT(ROWS, x, codes, scales, j, n, k)                                                                    \
  {                                                                                                                \
    const uint planes = (k) / 256, lane = get_sub_group_local_id();                                                \
    global const uint4 *wr = (global const uint4 *)((codes) + (ulong)(j) * (k));                                   \
    global const uint4 *sr = (global const uint4 *)((scales) + (ulong)(j) * ((k) / 32));                           \
    global const uint *xw = (global const uint *)(x);                                                              \
    const ulong xs = (k) / 2; /* a row of x in words */                                                            \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;                                               \
    uint p = 0;                                                                                                    \
    for (; p + Q8_UNROLL <= planes;) Q8_GROUP(ROWS, Q8_UNROLL, p)                                                  \
    for (; p + Q8_TAIL <= planes;) Q8_GROUP(ROWS, Q8_TAIL, p)                                                      \
    for (; p < planes;) Q8_GROUP(ROWS, 1, p)                                                                       \
  }

// y[r][j] for r < rows, rows <= ROWS; x holds ROWS rows of k values (k a multiple of 256); w is the codes then the scales.
#define Q8_MATVEC(ROWS)                                                                                            \
  Q8_KERNEL(Q8_WG) void Q8_NAME(dense_q8_##ROWS)(global const ushort *x, global const uchar *w, global ushort *y,   \
                                                  uint rows, uint n, uint k, uint ldy) {                           \
    const uint j = get_group_id(0) * (Q8_WG / 16) + get_sub_group_id();                                            \
    if (j >= n) return;                                                                                            \
    global const uchar *codes = w;                                                                                 \
    global const half *scales = (global const half *)(w + (ulong)n * k);                                           \
    float acc[ROWS];                                                                                               \
    Q8_DOT(ROWS, x, codes, scales, j, n, k)                                                                        \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                            \
      const float s = sub_group_reduce_add(acc[r]);                                                                \
      if (get_sub_group_local_id() == 0 && r < rows) y[(ulong)r * ldy + j] = Q8_STORE(s);                          \
    }                                                                                                              \
  }

// out[r][j] = cap30(bf16(h[r] . dequant(U)[j])) for r < rows, rows <= ROWS, j < vocab; k is 2,304.
#define Q8_HEAD(ROWS)                                                                                              \
  Q8_KERNEL(Q8_WG) void Q8_NAME(head_q8_##ROWS)(global const ushort *h, global const uchar *w, global float *out,   \
                                                 uint rows, uint vocab) {                                          \
    const uint j = get_group_id(0) * (Q8_WG / 16) + get_sub_group_id();                                            \
    if (j >= vocab) return;                                                                                        \
    global const uchar *codes = w;                                                                                 \
    global const half *scales = (global const half *)(w + (ulong)vocab * Q8_HIDDEN);                               \
    float acc[ROWS];                                                                                               \
    Q8_DOT(ROWS, h, codes, scales, j, vocab, Q8_HIDDEN)                                                            \
    _Pragma("unroll") for (uint r = 0; r < ROWS; ++r) {                                                            \
      const float s = sub_group_reduce_add(acc[r]);                                                                \
      if (get_sub_group_local_id() == 0 && r < rows) out[(ulong)r * vocab + j] = q8_cap(Q8_CAP_ROUND(s), 30.0f);   \
    }                                                                                                              \
  }

Q8_MATVEC(1)
Q8_MATVEC(2)
Q8_MATVEC(4)
Q8_MATVEC(8)
Q8_MATVEC(16)
Q8_MATVEC(32)

Q8_HEAD(1)
Q8_HEAD(2)
Q8_HEAD(4)
Q8_HEAD(8)
Q8_HEAD(16)
Q8_HEAD(32)

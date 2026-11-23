// Inkling 2 Mini loop on Intel Xe2 (OpenCL C): the cross-pass messages and the rank-64 message gates.
//
// A port of kernels/metal/inkling2/loop.metal. The formulas, the rounding points and the binary32 blend are the
// Metal kernels'. Sums run in fixed order: each work-item adds its strided share in order, a sub-group of 16
// reduces, and the 16 sub-group sums add in order. No bf16 value accumulates.

#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

#define LOOP_EPS 1e-6f
#define LOOP_SCALE 0x1.555556p-6f // binary32(1 / 48)
#define LOOP_H 2304
#define LOOP_RANK 64
#define LOOP_MESSAGES 48
#define LOOP_SGS 16 // sub-groups in a work-group of 256

// Planted controls (the contract image `contract_inkling2_loop_controls` is this file built with both defined).
#ifdef TF_CTL_LOOP_NO_SCALE
#define LOOP_LOGIT(dot) (dot) // drops the post-dot binary32(1 / 48)
#else
#define LOOP_LOGIT(dot) ((dot) * LOOP_SCALE)
#endif
#ifdef TF_CTL_LOOP_NO_ZETA_ROUND
#define LOOP_ZETA(dot) (dot) // drops the bf16 rounding between D and F, Gamma
#else
#define LOOP_ZETA(dot) rbf(dot)
#endif

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // nearest even
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// The sum of a work-group's partials in fixed order: sub-group sums, then the 16 of them in order.
static float group_sum(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int j = 0; j < LOOP_SGS; ++j) t += red[j];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static inline float rms_inverse(float ss) { return 1.0f / sqrt(ss / 2304.0f + LOOP_EPS); }

// One of the 48 messages of a token: work-group `row * 48 + entry`. out is [row][12 layers][q, k, v, mlp][H].
KERNEL(256) void tf_inkling2_loop_messages(global const ushort *bank0, global const ushort *bank1,
                                           global const float *q_mlp, global const float *kv, global float *out,
                                           uint rows) {
  local float red[LOOP_SGS];
  const uint t = get_local_id(0), work = get_group_id(0);
  const uint row = work / LOOP_MESSAGES;
  const uint entry = work - row * LOOP_MESSAGES;
  if (row >= rows) return;
  const uint layer = entry / 4;
  const uint role = entry - 4 * layer;
  // q/mlp and k/v occupy even/odd rows respectively.
  const global float *query = ((role == 0 || role == 3) ? q_mlp : kv) + (size_t)(2 * layer + (role >= 2 ? 1 : 0)) * LOOP_H;
  global const ushort *b0 = bank0 + (size_t)row * LOOP_H;
  global const ushort *b1 = bank1 + (size_t)row * LOOP_H;
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint d = t; d < LOOP_H; d += 256) {
    const float x0 = bf2f(b0[d]), x1 = bf2f(b1[d]);
    ss0 += x0 * x0;
    ss1 += x1 * x1;
  }
  const float inv0 = rms_inverse(group_sum(ss0, red));
  const float inv1 = rms_inverse(group_sum(ss1, red));
  float dot0 = 0.0f, dot1 = 0.0f;
  for (uint d = t; d < LOOP_H; d += 256) {
    const float beta0 = bf2f(b0[d]) * inv0;
    const float beta1 = bf2f(b1[d]) * inv1;
    dot0 += query[d] * beta0;
    dot1 += query[d] * beta1;
  }
  const float l0 = LOOP_LOGIT(group_sum(dot0, red));
  const float l1 = LOOP_LOGIT(group_sum(dot1, red));
  const float peak = fmax(fmax(-INFINITY, l0), l1);
  float w0 = exp(l0 - peak);
  float w1 = exp(l1 - peak);
  float den = 0.0f;
  den += w0;
  den += w1;
  w0 = w0 / den;
  w1 = w1 / den;
  global float *dst = out + ((size_t)row * LOOP_MESSAGES + entry) * LOOP_H;
  for (uint d = t; d < LOOP_H; d += 256) {
    const float beta0 = bf2f(b0[d]) * inv0;
    const float beta1 = bf2f(b1[d]) * inv1;
    dst[d] = w0 * beta0 + w1 * beta1;
  }
}

// One controller tuple of a token: h and m are [row][H] binary32, down [64][2H], forget and gain [H][64] bf16.
KERNEL(256) void tf_inkling2_loop_gates(global const float *h, global const float *m, global const ushort *down,
                                        global const ushort *forget, global const ushort *gain, global float *out,
                                        uint rows) {
  local float red[LOOP_SGS];
  local float zeta[LOOP_RANK];
  local ushort normalized[2 * LOOP_H];
  const uint t = get_local_id(0), row = get_group_id(0);
  if (row >= rows) return;
  const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
  global const float *hr = h + (size_t)row * LOOP_H;
  global const float *mr = m + (size_t)row * LOOP_H;
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint d = t; d < LOOP_H; d += 256) {
    const float x0 = hr[d], x1 = mr[d];
    ss0 += x0 * x0;
    ss1 += x1 * x1;
  }
  const float inv0 = rms_inverse(group_sum(ss0, red));
  const float inv1 = rms_inverse(group_sum(ss1, red));
  for (uint d = t; d < LOOP_H; d += 256) {
    normalized[d] = f2bf(hr[d] * inv0);
    normalized[LOOP_H + d] = f2bf(mr[d] * inv1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // D: sub-group s owns ranks 4s to 4s + 3; a lane adds its 8-wide chunks in order, then the 16 lanes reduce.
  for (uint r = 4 * sg; r < 4 * sg + 4; ++r) {
    global const ushort *dr = down + (size_t)r * (2 * LOOP_H);
    float dot = 0.0f;
    for (uint d = 8 * lane; d < 2 * LOOP_H; d += 128) {
      const ushort8 w = vload8(0, dr + d);
      const ushort8 x = vload8(0, normalized + d);
      for (int j = 0; j < 8; ++j) dot += bf2f(w[j]) * bf2f(x[j]);
    }
    dot = sub_group_reduce_add(dot);
    if (lane == 0) zeta[r] = LOOP_ZETA(dot);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint d = t; d < LOOP_H; d += 256) {
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < LOOP_RANK; r += 8) {
      const ushort8 fw = vload8(0, forget + (size_t)d * LOOP_RANK + r);
      const ushort8 gw = vload8(0, gain + (size_t)d * LOOP_RANK + r);
      for (int j = 0; j < 8; ++j) {
        fdot += bf2f(fw[j]) * zeta[r + j];
        gdot += bf2f(gw[j]) * zeta[r + j];
      }
    }
    const float f = tanh(fdot);
    const float g = tanh(gdot);
    // Four separately rounded binary32 steps; do not fold these into one expression.
    const float keep = 1.0f - f;
    const float left = keep * hr[d];
    const float right = g * mr[d];
    out[(size_t)row * LOOP_H + d] = left + right;
  }
}

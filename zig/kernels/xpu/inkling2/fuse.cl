// Phase A as parallel launches: one group a row for the residual, and the controller ranks spread as in loop.cl.
// Arithmetic and reduction order match glue.cl, loop.cl and dense.cl. Roles of one visit share a launch.

#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel
#define H 2304
#define EPS 1e-6f
#define AR_SCALE 0x1.555556p-6f
#ifdef FUSE_CTL
#define FUSE_AR_SCALE 0.0f
#define FUSE_PRE(acc) (0.0f)
#define FUSE_ZETA(dot) (0.0f)
#define FUSE_GATE 0.0f
#define FUSE_EPI 0
#define FUSE_NORM 0.0f
#else
#define FUSE_AR_SCALE AR_SCALE
#define FUSE_PRE(acc) (acc)
#define FUSE_ZETA(dot) rbf(dot)
#define FUSE_GATE 1.0f
#define FUSE_EPI 1
#define FUSE_NORM 1.0f
#endif
#define RANK 64

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); }
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// S9: lane s < 5 of a sub-group adds the 16 sub-group sums of source s in order. Each sub-group gets the 5 sums
// from 5 lanes, not from 80 uniform loads, and the means and weights run one time a lane.
static inline float lane_sum16(local const float *r) {
  const uint l = get_sub_group_local_id();
  float ss = 0.0f;
  if (l < 5)
    for (uint j = 0; j < 16; ++j) ss += r[l * 16 + j];
  return ss;
}

static float wg_sum(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; ++i) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static void do_norm(global const float *x, global const ushort *w, global ushort *out, uint t, local float *red) {
  float sum = 0.0f;
  for (uint d = t; d < H; d += 256) sum = fma(x[d], x[d], sum);
  const float s = sub_group_reduce_add(sum);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float total = 0.0f;
  for (uint i = 0; i < 16; ++i) total += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inverse = 1.0f / sqrt(total / (float)H + EPS);
  for (uint d = t; d < H; d += 256) out[d] = f2bf(bf2f(w[d]) * (x[d] * inverse));
}

// Grid (rows). norm_n 0 skips the norms (they follow the gates). n_queries 0 widens p into ar.
KERNEL(256) void tf_ink2_fuse_pre(
    global const ushort *p, global float *partial, global const ushort *blocks, global const float *queries, global float *ar,
    global const ushort *wq, global const ushort *wk, global const ushort *wv, global ushort *q_norm, global ushort *k_norm,
    global ushort *v_norm, uint rows, uint blocks_stride, uint sources, uint n_queries, uint norm_n) {
  const uint row = get_group_id(0), t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (row >= rows || sources == 0 || sources > 5 || n_queries > 3) return;
  local float red[160];
  global const ushort *pb = p + (size_t)row * H;
  global float *part = partial + (size_t)row * H;
  for (uint d = t; d < H; d += 256) part[d] = bf2f(pb[d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (n_queries == 0) {
    for (uint d = t; d < H; d += 256) ar[(size_t)row * H + d] = bf2f(pb[d]);
  } else {
    for (uint i = t; i < 160; i += 256) red[i] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    global const ushort *blk = blocks + (size_t)row * blocks_stride;
    float v[5][H / 256];
    for (uint s = 0; s < 5; ++s)
      for (uint i = 0; i < H / 256; ++i)
        v[s][i] = s < sources ? (s + 1 == sources ? part[t + 256 * i] : bf2f(blk[(size_t)s * H + t + 256 * i])) : 0.0f;
    float means[5];
    for (uint s = 0; s < 5; ++s) {
      float ss = 0.0f;
      for (uint i = 0; i < H / 256; ++i) ss += v[s][i] * v[s][i];
      const float reduced = sub_group_reduce_add(ss);
      if (s < sources && lane == 0) red[s * 16 + sg] = reduced;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      const float mean_l = rsqrt(lane_sum16(red) / 2304.0f + EPS);
      for (uint s = 0; s < 5; ++s) means[s] = sub_group_broadcast(mean_l, s);
    }
    for (uint q = 0; q < n_queries; ++q) {
      float query[H / 256];
      for (uint i = 0; i < H / 256; ++i) query[i] = queries[(size_t)q * H + t + 256 * i];
      local float *dots = red + 80;
      for (uint s = 0; s < 5; ++s) {
        float dot = 0.0f;
        for (uint i = 0; i < H / 256; ++i) dot += query[i] * (v[s][i] * means[s]);
        const float reduced = sub_group_reduce_add(dot);
        if (s < sources && lane == 0) dots[s * 16 + sg] = reduced;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      float weights[5];
      float peak = -INFINITY;
      const float logit_l = lane_sum16(dots) * FUSE_AR_SCALE;
      for (uint s = 0; s < 5; ++s) {
        weights[s] = sub_group_broadcast(logit_l, s);
        if (s < sources) peak = peak < weights[s] ? weights[s] : peak;
      }
      const float e_l = exp(logit_l - peak);
      float den = 0.0f;
      for (uint s = 0; s < 5; ++s) {
        weights[s] = sub_group_broadcast(e_l, s);
        if (s < sources) den += weights[s];
      }
      const float w_l = e_l / den;
      for (uint s = 0; s < 5; ++s) weights[s] = sub_group_broadcast(w_l, s);
      global float *dst = ar + ((size_t)q * rows + row) * H;
      for (uint i = 0; i < H / 256; ++i) {
        float acc = 0.0f;
        for (uint s = 0; s < 5; ++s)
          if (s < sources) acc += weights[s] * (v[s][i] * means[s]);
        dst[t + 256 * i] = FUSE_PRE(acc);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  if (norm_n == 0) return;
  barrier(CLK_LOCAL_MEM_FENCE);
  do_norm(ar + (size_t)row * H, wq, q_norm + (size_t)row * H, t, red);
  if (norm_n > 1) {
    do_norm(ar + ((size_t)rows + row) * H, wk, k_norm + (size_t)row * H, t, red);
    do_norm(ar + ((size_t)2 * rows + row) * H, wv, v_norm + (size_t)row * H, t, red);
  }
}

// Grid (32, roles, rows). Part and sub-group own the same two ranks as tf_inkling2_loop_zeta.
KERNEL(256) void tf_ink2_fuse_zeta(
    global const float *ar, global const float *messages, global const ushort *down0, global const ushort *down1,
    global const ushort *down2, global float *zeta, uint rows, uint roles, uint local_src, uint core, uint msg_entries) {
  const uint part = get_group_id(0), role = get_group_id(1), row = get_group_id(2);
  if (row >= rows || role >= roles || part >= 32) return;
  const uint t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  local float red[16];
  local ushort normalized[2 * H];
  const uint src = local_src ? 0 : role;
  global const float *h = ar + ((size_t)src * rows + row) * H;
  global const float *m = messages + ((size_t)row * msg_entries + core * 4 + role) * H;
  global const ushort *down = role == 0 ? down0 : role == 1 ? down1 : down2;
  float hv[H / 256], mv[H / 256];
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint i = 0; i < H / 256; ++i) {
    hv[i] = h[t + 256 * i];
    mv[i] = m[t + 256 * i];
    ss0 += hv[i] * hv[i];
    ss1 += mv[i] * mv[i];
  }
  const float inv0 = 1.0f / sqrt(wg_sum(ss0, red) / 2304.0f + EPS);
  const float inv1 = 1.0f / sqrt(wg_sum(ss1, red) / 2304.0f + EPS);
  for (uint i = 0; i < H / 256; ++i) {
    normalized[t + 256 * i] = f2bf(hv[i] * inv0);
    normalized[H + t + 256 * i] = f2bf(mv[i] * inv1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg < 2) {
    const uint r = 2 * part + sg;
    global const ushort *dr = down + (size_t)r * (2 * H);
    float dot = 0.0f;
    for (uint d = 8 * lane; d < 2 * H; d += 512) {
      const ushort8 w0 = vload8(0, dr + d), w1 = vload8(0, dr + d + 128);
      const ushort8 w2 = vload8(0, dr + d + 256), w3 = vload8(0, dr + d + 384);
      const ushort8 x0 = vload8(0, normalized + d), x1 = vload8(0, normalized + d + 128);
      const ushort8 x2 = vload8(0, normalized + d + 256), x3 = vload8(0, normalized + d + 384);
      for (int j = 0; j < 8; ++j) dot += bf2f(w0[j]) * bf2f(x0[j]);
      for (int j = 0; j < 8; ++j) dot += bf2f(w1[j]) * bf2f(x1[j]);
      for (int j = 0; j < 8; ++j) dot += bf2f(w2[j]) * bf2f(x2[j]);
      for (int j = 0; j < 8; ++j) dot += bf2f(w3[j]) * bf2f(x3[j]);
    }
    dot = sub_group_reduce_add(dot);
    if (lane == 0) zeta[((size_t)row * 4 + role) * RANK + r] = FUSE_ZETA(dot);
  }
}

// Grid (4, roles, rows). Every sub-group owns one rank. The norm and the lane dot match tf_ink2_fuse_zeta.
// pack4 stores (row * 4 + role); otherwise row * 64, for the MLP controller.
KERNEL(256) void tf_ink2_zeta_wide(
    global const float *ar, global const float *messages, global const ushort *down0, global const ushort *down1,
    global const ushort *down2, global float *zeta, uint rows, uint roles, uint local_src, uint core, uint msg_entries,
    uint pack4) {
  const uint part = get_group_id(0), role = get_group_id(1), row = get_group_id(2);
  if (row >= rows || role >= roles || part >= 4) return;
  const uint t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const uint r = part * 16 + sg;
  local float red[16];
  local ushort normalized[2 * H];
  const uint src = local_src ? 0 : role;
  global const float *h = ar + ((size_t)src * rows + row) * H;
  global const float *m = messages + ((size_t)row * msg_entries + core * 4 + role) * H;
  global const ushort *down = role == 0 ? down0 : role == 1 ? down1 : down2;
  float hv[H / 256], mv[H / 256];
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint i = 0; i < H / 256; ++i) {
    hv[i] = h[t + 256 * i];
    mv[i] = m[t + 256 * i];
    ss0 += hv[i] * hv[i];
    ss1 += mv[i] * mv[i];
  }
  const float inv0 = 1.0f / sqrt(wg_sum(ss0, red) / 2304.0f + EPS);
  const float inv1 = 1.0f / sqrt(wg_sum(ss1, red) / 2304.0f + EPS);
  for (uint i = 0; i < H / 256; ++i) {
    normalized[t + 256 * i] = f2bf(hv[i] * inv0);
    normalized[H + t + 256 * i] = f2bf(mv[i] * inv1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  global const ushort *dr = down + (size_t)r * (2 * H);
  float dot = 0.0f;
  for (uint d = 8 * lane; d < 2 * H; d += 512) {
    const ushort8 w0 = vload8(0, dr + d), w1 = vload8(0, dr + d + 128);
    const ushort8 w2 = vload8(0, dr + d + 256), w3 = vload8(0, dr + d + 384);
    const ushort8 x0 = vload8(0, normalized + d), x1 = vload8(0, normalized + d + 128);
    const ushort8 x2 = vload8(0, normalized + d + 256), x3 = vload8(0, normalized + d + 384);
    for (int j = 0; j < 8; ++j) dot += bf2f(w0[j]) * bf2f(x0[j]);
    for (int j = 0; j < 8; ++j) dot += bf2f(w1[j]) * bf2f(x1[j]);
    for (int j = 0; j < 8; ++j) dot += bf2f(w2[j]) * bf2f(x2[j]);
    for (int j = 0; j < 8; ++j) dot += bf2f(w3[j]) * bf2f(x3[j]);
  }
  dot = sub_group_reduce_add(dot);
  if (lane == 0) zeta[pack4 ? ((size_t)row * 4 + role) * RANK + r : (size_t)row * RANK + r] = FUSE_ZETA(dot);
}

// Grid (16, roles, rows). The prologue's AR for the group's source row, then the message gate projection over four
// sub-groups a rank. The AR is the code of tf_ink2_fuse_pre_q, run by every group, so the bits do not move. Only group
// 0 stores it.
KERNEL(256) void tf_ink2_pre_zeta(
    global const ushort *p, global float *partial, global const ushort *blocks, global const float *queries, global float *ar,
    global const float *messages, global const ushort *down0, global const ushort *down1, global const ushort *down2,
    global float *zeta, uint rows, uint blocks_stride, uint sources, uint roles, uint local_src, uint core, uint msg_entries) {
  const uint g = get_group_id(0), role = get_group_id(1), row = get_group_id(2);
  const uint t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (row >= rows || role >= roles || g >= 16 || sources == 0 || sources > 5) return;
  const uint src = local_src ? 0 : role;
  const uint r = g * 4 + (sg >> 2), part = sg & 3u;
  local float red[160];
  local float parts[16];
  local ushort normalized[2 * H];
  // The down row of this rank part, loaded before the residual: the residual and the norms run under the reads.
  global const ushort *down = role == 0 ? down0 : role == 1 ? down1 : down2;
  global const ushort *dr = down + (size_t)r * (2 * H);
  ushort8 dw[2 * H / 512];
  for (uint k = 0; k < 2 * H / 512; ++k) dw[k] = vload8(0, dr + 8 * lane + 128 * part + 512 * k);
  global const ushort *pb = p + (size_t)row * H;
  global float *part_row = partial + (size_t)row * H;
  // Every group computes the same residual. Group 0 of role 0 stores partial, group 0 of each source role stores ar.
  if (g == 0 && role == 0)
    for (uint d = t; d < H; d += 256) part_row[d] = bf2f(pb[d]);
  for (uint i = t; i < 160; i += 256) red[i] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  global const ushort *blk = blocks + (size_t)row * blocks_stride;
  float v[5][H / 256];
  for (uint s = 0; s < 5; ++s)
    for (uint i = 0; i < H / 256; ++i)
      v[s][i] = s < sources ? (s + 1 == sources ? bf2f(pb[t + 256 * i]) : bf2f(blk[(size_t)s * H + t + 256 * i])) : 0.0f;
  float means[5];
  for (uint s = 0; s < 5; ++s) {
    float ss = 0.0f;
    for (uint i = 0; i < H / 256; ++i) ss += v[s][i] * v[s][i];
    const float reduced = sub_group_reduce_add(ss);
    if (s < sources && lane == 0) red[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    const float mean_l = rsqrt(lane_sum16(red) / 2304.0f + EPS);
    for (uint s = 0; s < 5; ++s) means[s] = sub_group_broadcast(mean_l, s);
  }
  float query[H / 256];
  for (uint i = 0; i < H / 256; ++i) query[i] = queries[(size_t)src * H + t + 256 * i];
  local float *dots = red + 80;
  for (uint s = 0; s < 5; ++s) {
    float dot = 0.0f;
    for (uint i = 0; i < H / 256; ++i) dot += query[i] * (v[s][i] * means[s]);
    const float reduced = sub_group_reduce_add(dot);
    if (s < sources && lane == 0) dots[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float weights[5];
  float peak = -INFINITY;
  const float logit_l = lane_sum16(dots) * FUSE_AR_SCALE;
  for (uint s = 0; s < 5; ++s) {
    weights[s] = sub_group_broadcast(logit_l, s);
    if (s < sources) peak = peak < weights[s] ? weights[s] : peak;
  }
  const float e_l = exp(logit_l - peak);
  float den = 0.0f;
  for (uint s = 0; s < 5; ++s) {
    weights[s] = sub_group_broadcast(e_l, s);
    if (s < sources) den += weights[s];
  }
  const float w_l = e_l / den;
  for (uint s = 0; s < 5; ++s) weights[s] = sub_group_broadcast(w_l, s);
  float hv[H / 256];
  const bool store_ar = g == 0 && role == src;
  for (uint i = 0; i < H / 256; ++i) {
    float acc = 0.0f;
    for (uint s = 0; s < 5; ++s)
      if (s < sources) acc += weights[s] * (v[s][i] * means[s]);
    hv[i] = FUSE_PRE(acc);
    if (store_ar) ar[((size_t)src * rows + row) * H + t + 256 * i] = hv[i];
  }
  // No barrier here: hv is in registers, and the red entries that wg_sum writes next were last read before the
  // barrier of the dots.
  global const float *m = messages + ((size_t)row * msg_entries + core * 4 + role) * H;
  float mv[H / 256];
  float ss1 = 0.0f, ss0 = 0.0f;
  for (uint i = 0; i < H / 256; ++i) {
    mv[i] = m[t + 256 * i];
    ss1 += mv[i] * mv[i];
    ss0 += hv[i] * hv[i];
  }
  const float inv0 = 1.0f / sqrt(wg_sum(ss0, red) / 2304.0f + EPS);
  const float inv1 = 1.0f / sqrt(wg_sum(ss1, red) / 2304.0f + EPS);
  for (uint i = 0; i < H / 256; ++i) {
    normalized[t + 256 * i] = f2bf(hv[i] * inv0);
    normalized[H + t + 256 * i] = f2bf(mv[i] * inv1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = 0.0f;
  for (uint k = 0; k < 2 * H / 512; ++k) {
    const ushort8 w0 = dw[k];
    const ushort8 x0 = vload8(0, normalized + 8 * lane + 128 * part + 512 * k);
    for (int j = 0; j < 8; ++j) dot += bf2f(w0[j]) * bf2f(x0[j]);
  }
  dot = sub_group_reduce_add(dot);
  if (lane == 0) parts[sg] = dot;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == (sg & ~3u) && lane == 0) {
    float total = parts[sg];
    total += parts[sg + 1];
    total += parts[sg + 2];
    total += parts[sg + 3];
    zeta[((size_t)row * 4 + role) * RANK + r] = FUSE_ZETA(total);
  }
}

// Grid (16, roles, rows). Four sub-groups a rank, each over one of the four 128-term blocks of a step, so 256 sub-groups
// a call instead of 64. The four partial dots are summed in SLM, in sub-group order, and stored once.
KERNEL(256) void tf_ink2_zeta_split(
    global const float *ar, global const float *messages, global const ushort *down0, global const ushort *down1,
    global const ushort *down2, global float *zeta, uint rows, uint roles, uint local_src, uint core, uint msg_entries,
    uint pack4) {
  const uint g = get_group_id(0), role = get_group_id(1), row = get_group_id(2);
  if (row >= rows || role >= roles || g >= 16) return;
  const uint t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const uint r = g * 4 + (sg >> 2), part = sg & 3u;
  local float red[16];
  local float parts[16];
  local ushort normalized[2 * H];
  const uint src = local_src ? 0 : role;
  global const float *h = ar + ((size_t)src * rows + row) * H;
  global const float *m = messages + ((size_t)row * msg_entries + core * 4 + role) * H;
  global const ushort *down = role == 0 ? down0 : role == 1 ? down1 : down2;
  // The down row of this rank part, loaded before the norms: the norms run under the reads.
  global const ushort *dr = down + (size_t)r * (2 * H);
  ushort8 dw[2 * H / 512];
  for (uint k = 0; k < 2 * H / 512; ++k) dw[k] = vload8(0, dr + 8 * lane + 128 * part + 512 * k);
  float hv[H / 256], mv[H / 256];
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint i = 0; i < H / 256; ++i) {
    hv[i] = h[t + 256 * i];
    mv[i] = m[t + 256 * i];
    ss0 += hv[i] * hv[i];
    ss1 += mv[i] * mv[i];
  }
  const float inv0 = 1.0f / sqrt(wg_sum(ss0, red) / 2304.0f + EPS);
  const float inv1 = 1.0f / sqrt(wg_sum(ss1, red) / 2304.0f + EPS);
  for (uint i = 0; i < H / 256; ++i) {
    normalized[t + 256 * i] = f2bf(hv[i] * inv0);
    normalized[H + t + 256 * i] = f2bf(mv[i] * inv1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float dot = 0.0f;
  for (uint k = 0; k < 2 * H / 512; ++k) {
    const ushort8 w0 = dw[k];
    const ushort8 x0 = vload8(0, normalized + 8 * lane + 128 * part + 512 * k);
    for (int j = 0; j < 8; ++j) dot += bf2f(w0[j]) * bf2f(x0[j]);
  }
  dot = sub_group_reduce_add(dot);
  if (lane == 0) parts[sg] = dot;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg == (sg & ~3u) && lane == 0) {
    float total = parts[sg];
    total += parts[sg + 1];
    total += parts[sg + 2];
    total += parts[sg + 3];
    zeta[pack4 ? ((size_t)row * 4 + role) * RANK + r : (size_t)row * RANK + r] = FUSE_ZETA(total);
  }
}

// Grid (36, roles, rows). One work-item one channel, the same 64-term order as tf_inkling2_loop_gates.
KERNEL(64) void tf_ink2_fuse_gates(
    global const float *ar, global const float *messages, global const ushort *forget0, global const ushort *forget1,
    global const ushort *forget2, global const ushort *gain0, global const ushort *gain1, global const ushort *gain2,
    global const float *zeta, global float *gated, uint rows, uint roles, uint local_src, uint core, uint msg_entries) {
  const uint role = get_group_id(1), row = get_group_id(2), t = get_local_id(0);
  const uint d = get_group_id(0) * 64 + t;
  if (row >= rows || role >= roles || d >= H) return;
  local float z[RANK];
  z[t] = zeta[((size_t)row * 4 + role) * RANK + t];
  const uint src = local_src ? 0 : role;
  global const float *h = ar + ((size_t)src * rows + row) * H;
  global const float *m = messages + ((size_t)row * msg_entries + core * 4 + role) * H;
  global const ushort *forget = role == 0 ? forget0 : role == 1 ? forget1 : forget2;
  global const ushort *gain = role == 0 ? gain0 : role == 1 ? gain1 : gain2;
  ushort8 fw[RANK / 8], gw[RANK / 8];
  for (uint r = 0; r < RANK / 8; ++r) {
    fw[r] = vload8(0, forget + (size_t)d * RANK + 8 * r);
    gw[r] = vload8(0, gain + (size_t)d * RANK + 8 * r);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float fdot = 0.0f, gdot = 0.0f;
  for (uint r = 0; r < RANK / 8; ++r)
    for (int j = 0; j < 8; ++j) {
      fdot += bf2f(fw[r][j]) * z[8 * r + j];
      gdot += bf2f(gw[r][j]) * z[8 * r + j];
    }
  const float f = tanh(fdot);
  const float g = tanh(gdot);
  const float keep = 1.0f - f;
  const float left = keep * h[d];
  const float right = FUSE_GATE * g * m[d];
  gated[((size_t)role * rows + row) * H + d] = left + right;
}

// Grid (roles, rows). One group a gated row: every channel of the row in the group, so the norm of the row follows in
// the same group. The per-channel mix is the gates kernel's, and do_norm keeps its 256-thread order, so the bits do not move.
KERNEL(256) void tf_ink2_gates_norms(
    global const float *ar, global const float *messages, global const ushort *forget0, global const ushort *forget1,
    global const ushort *forget2, global const ushort *gain0, global const ushort *gain1, global const ushort *gain2,
    global const float *zeta, global float *gated, global const ushort *wq, global const ushort *wk, global const ushort *wv,
    global ushort *q_norm, global ushort *k_norm, global ushort *v_norm,
    uint rows, uint roles, uint local_src, uint core, uint msg_entries, uint norm_n) {
  const uint role = get_group_id(0), row = get_group_id(1), t = get_local_id(0);
  if (row >= rows || role >= roles) return;
  local float z[RANK];
  local float red[16];
  if (t < RANK) z[t] = zeta[((size_t)row * 4 + role) * RANK + t];
  const uint src = local_src ? 0 : role;
  global const float *h = ar + ((size_t)src * rows + row) * H;
  global const float *m = messages + ((size_t)row * msg_entries + core * 4 + role) * H;
  global const ushort *forget = role == 0 ? forget0 : role == 1 ? forget1 : forget2;
  global const ushort *gain = role == 0 ? gain0 : role == 1 ? gain1 : gain2;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint d = t; d < H; d += 256) {
    ushort8 fw[RANK / 8], gw[RANK / 8];
    for (uint r = 0; r < RANK / 8; ++r) {
      fw[r] = vload8(0, forget + (size_t)d * RANK + 8 * r);
      gw[r] = vload8(0, gain + (size_t)d * RANK + 8 * r);
    }
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < RANK / 8; ++r)
      for (int j = 0; j < 8; ++j) {
        fdot += bf2f(fw[r][j]) * z[8 * r + j];
        gdot += bf2f(gw[r][j]) * z[8 * r + j];
      }
    const float f = tanh(fdot);
    const float g = tanh(gdot);
    const float keep = 1.0f - f;
    const float left = keep * h[d];
    const float right = FUSE_GATE * g * m[d];
    gated[((size_t)role * rows + row) * H + d] = left + right;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (role >= norm_n) return;
  global const ushort *w = role == 0 ? wq : role == 1 ? wk : wv;
  global ushort *out = (role == 0 ? q_norm : role == 1 ? k_norm : v_norm) + (size_t)row * H;
  do_norm(gated + ((size_t)role * rows + row) * H, w, out, t, red);
}

// Grid (norm_n, rows). The three gated rows, one group each.
KERNEL(256) void tf_ink2_fuse_norms(
    global const float *gated, global const ushort *wq, global const ushort *wk, global const ushort *wv,
    global ushort *q_norm, global ushort *k_norm, global ushort *v_norm, uint rows, uint norm_n) {
  const uint which = get_group_id(0), row = get_group_id(1), t = get_local_id(0);
  if (row >= rows || which >= norm_n) return;
  local float red[16];
  global const float *x = gated + ((size_t)which * rows + row) * H;
  global const ushort *w = which == 0 ? wq : which == 1 ? wk : wv;
  global ushort *out = (which == 0 ? q_norm : which == 1 ? k_norm : v_norm) + (size_t)row * H;
  do_norm(x, w, out, t, red);
}

// Grid (n_queries, rows). One group a query row: the same AR as tf_ink2_fuse_pre, whose query rows ran in a loop in one
// group. The source means are recomputed by each group, the same code and the same order, so the bits do not move.
KERNEL(256) void tf_ink2_fuse_pre_q(
    global const ushort *p, global float *partial, global const ushort *blocks, global const float *queries, global float *ar,
    global const ushort *wq, global const ushort *wk, global const ushort *wv,
    global ushort *q_norm, global ushort *k_norm, global ushort *v_norm,
    uint rows, uint blocks_stride, uint sources, uint n_queries, uint norm_n) {
  const uint q = get_group_id(0), row = get_group_id(1), t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (row >= rows || q >= n_queries || sources == 0 || sources > 5) return;
  local float red[160];
  global const ushort *pb = p + (size_t)row * H;
  global float *part = partial + (size_t)row * H;
  for (uint d = t; d < H; d += 256) part[d] = bf2f(pb[d]);
  for (uint i = t; i < 160; i += 256) red[i] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  global const ushort *blk = blocks + (size_t)row * blocks_stride;
  float v[5][H / 256];
  for (uint s = 0; s < 5; ++s)
    for (uint i = 0; i < H / 256; ++i)
      v[s][i] = s < sources ? (s + 1 == sources ? part[t + 256 * i] : bf2f(blk[(size_t)s * H + t + 256 * i])) : 0.0f;
  float means[5];
  for (uint s = 0; s < 5; ++s) {
    float ss = 0.0f;
    for (uint i = 0; i < H / 256; ++i) ss += v[s][i] * v[s][i];
    const float reduced = sub_group_reduce_add(ss);
    if (s < sources && lane == 0) red[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  {
    const float mean_l = rsqrt(lane_sum16(red) / 2304.0f + EPS);
    for (uint s = 0; s < 5; ++s) means[s] = sub_group_broadcast(mean_l, s);
  }
  float query[H / 256];
  for (uint i = 0; i < H / 256; ++i) query[i] = queries[(size_t)q * H + t + 256 * i];
  local float *dots = red + 80;
  for (uint s = 0; s < 5; ++s) {
    float dot = 0.0f;
    for (uint i = 0; i < H / 256; ++i) dot += query[i] * (v[s][i] * means[s]);
    const float reduced = sub_group_reduce_add(dot);
    if (s < sources && lane == 0) dots[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float weights[5];
  float peak = -INFINITY;
  const float logit_l = lane_sum16(dots) * FUSE_AR_SCALE;
  for (uint s = 0; s < 5; ++s) {
    weights[s] = sub_group_broadcast(logit_l, s);
    if (s < sources) peak = peak < weights[s] ? weights[s] : peak;
  }
  const float e_l = exp(logit_l - peak);
  float den = 0.0f;
  for (uint s = 0; s < 5; ++s) {
    weights[s] = sub_group_broadcast(e_l, s);
    if (s < sources) den += weights[s];
  }
  const float w_l = e_l / den;
  for (uint s = 0; s < 5; ++s) weights[s] = sub_group_broadcast(w_l, s);
  global float *dst = ar + ((size_t)q * rows + row) * H;
  for (uint i = 0; i < H / 256; ++i) {
    float acc = 0.0f;
    for (uint s = 0; s < 5; ++s)
      if (s < sources) acc += weights[s] * (v[s][i] * means[s]);
    dst[t + 256 * i] = FUSE_PRE(acc);
  }
  if (norm_n == 0) return;
  barrier(CLK_GLOBAL_MEM_FENCE);
  global const ushort *w = q == 0 ? wq : q == 1 ? wk : wv;
  global ushort *out = (q == 0 ? q_norm : q == 1 ? k_norm : v_norm) + (size_t)row * H;
  do_norm(dst, w, out, t, red);
}

static inline float4 rx4(global const uint2 *xg, uint i) {
  const uint2 v = xg[i];
  return (float4)(as_float(v.x << 16), as_float(v.x & 0xffff0000u), as_float(v.y << 16), as_float(v.y & 0xffff0000u));
}

static inline float rsig(float x) {
  const float e = exp(-fabs(x));
  return x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
}

// Grid (8, rows). Sixteen sub-groups, one expert each. The lane map matches tf_inkling2_router.
KERNEL(256) void tf_ink2_router_gemv(global const ushort *x, global const ushort *weight, global float *scores, uint rows) {
  const uint row = get_group_id(1), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const uint expert = get_group_id(0) * 16 + sg;
  if (row >= rows || expert >= 128) return;
  global const uint2 *xg = (global const uint2 *)(x + (size_t)row * H);
  global const uint4 *w = (global const uint4 *)(weight + (size_t)expert * H);
  float dot = 0.0f;
  for (uint part = 0; part < 2; ++part) {
    for (uint u = 0; u < 9; ++u) {
      const uint at = (part * 9 + u) * 16 + lane;
      const uint4 a = w[(part * 9 + u) * 16 + lane];
      const float4 b0 = rx4(xg, 2 * at), b1 = rx4(xg, 2 * at + 1);
      dot += as_float(a.x << 16) * b0.x;
      dot += as_float(a.x & 0xffff0000u) * b0.y;
      dot += as_float(a.y << 16) * b0.z;
      dot += as_float(a.y & 0xffff0000u) * b0.w;
      dot += as_float(a.z << 16) * b1.x;
      dot += as_float(a.z & 0xffff0000u) * b1.y;
      dot += as_float(a.w << 16) * b1.z;
      dot += as_float(a.w & 0xffff0000u) * b1.w;
    }
  }
  dot = sub_group_reduce_add(dot);
  if (lane == 0) scores[(size_t)row * 128 + expert] = rsig(dot);
}

// The dense norm, then the wide router GEMV. Group 0 stores the bf16 row. The GEMV reads that row from local.
// Each lane loads its 18 weight pieces before the norm, so the weight reads run under the norm (the 8 groups of the
// launch each compute the norm). The dot order is the one of tf_ink2_router_gemv.
KERNEL(256) void tf_ink2_router_norm_gemv(
    global const float *x, global const ushort *nw, global const ushort *weight, global ushort *normed_out,
    global float *scores, uint rows) {
  const uint row = get_group_id(1), lane = get_sub_group_local_id(), sg = get_sub_group_id(), t = get_local_id(0);
  if (row >= rows) return;
  local float red[16];
  local uint4 normed4[H / 8];
  local ushort *normed = (local ushort *)normed4;
  const uint expert = get_group_id(0) * 16 + sg;
  global const uint4 *w = (global const uint4 *)(weight + (size_t)(expert < 128 ? expert : 127) * H);
  uint4 a[18];
  for (uint k = 0; k < 18; ++k) a[k] = w[k * 16 + lane];
  global const float *xr = x + (size_t)row * H;
  float sum = 0.0f;
  for (uint d = t; d < H; d += 256) sum = fma(xr[d], xr[d], sum);
  const float s = sub_group_reduce_add(sum);
  if (lane == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float total = 0.0f;
  for (uint i = 0; i < 16; ++i) total += red[i];
  const float inverse = FUSE_NORM / sqrt(total / (float)H + EPS);
  for (uint d = t; d < H; d += 256) normed[d] = f2bf(bf2f(nw[d]) * (xr[d] * inverse));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_group_id(0) == 0) {
    for (uint d = t; d < H; d += 256) normed_out[(size_t)row * H + d] = normed[d];
  }
  if (expert >= 128) return;
  float dot = 0.0f;
  for (uint part = 0; part < 2; ++part) {
    for (uint u = 0; u < 9; ++u) {
      const uint at = (part * 9 + u) * 16 + lane;
      const uint4 wa = a[part * 9 + u];
      // the 8 row values of this piece in one 16 B local load: value 8 * at + i is half i of the four words
      const uint4 xv = normed4[at];
      dot += as_float(wa.x << 16) * as_float(xv.x << 16);
      dot += as_float(wa.x & 0xffff0000u) * as_float(xv.x & 0xffff0000u);
      dot += as_float(wa.y << 16) * as_float(xv.y << 16);
      dot += as_float(wa.y & 0xffff0000u) * as_float(xv.y & 0xffff0000u);
      dot += as_float(wa.z << 16) * as_float(xv.z << 16);
      dot += as_float(wa.z & 0xffff0000u) * as_float(xv.z & 0xffff0000u);
      dot += as_float(wa.w << 16) * as_float(xv.w << 16);
      dot += as_float(wa.w & 0xffff0000u) * as_float(xv.w & 0xffff0000u);
    }
  }
  dot = sub_group_reduce_add(dot);
  if (lane == 0) scores[(size_t)row * 128 + expert] = rsig(dot);
}

// Grid (rows). Same tie rule as tf_inkling2_router: a larger key wins, an equal key keeps the lower index. A NaN key
// ranks as -inf, below every number, so the ranks are a permutation and exactly 8 work-items store.
// Two barriers: the keys, then one 16-bit flag mask a sub-group. A picked work-item finds its place and the sum of the 8
// picked scores (ascending index, the order of the old work-item 0 loop) from the masks, and stores its own id and weight.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1))) kernel void tf_ink2_router_pick(
    global const float *scores_in, global const float *bias, global uint *ids, global float *weights, uint rows) {
  const uint row = get_group_id(0), t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (row >= rows) return;
  local float4 keys4[32];
  local float scores[128];
  local uint masks[8];
  local float *keys = (local float *)keys4;
  const float sc = scores_in[(size_t)row * 128 + t];
  scores[t] = sc;
  const float key = sc + bias[t];
  const float ke = isnan(key) ? -INFINITY : key;
  keys[t] = ke;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint rank = 0;
  for (uint q = 0; q < 32; ++q) {
    const float4 k4 = keys4[q];
    const uint j = 4 * q;
    rank += (k4.x > ke || (k4.x == ke && j < t)) ? 1u : 0u;
    rank += (k4.y > ke || (k4.y == ke && j + 1 < t)) ? 1u : 0u;
    rank += (k4.z > ke || (k4.z == ke && j + 2 < t)) ? 1u : 0u;
    rank += (k4.w > ke || (k4.w == ke && j + 3 < t)) ? 1u : 0u;
  }
  const uint flag = rank < 8 ? 1u : 0u;
  const uint mask = sub_group_reduce_add(flag << lane); // the bits are disjoint, so the sum is the mask
  if (lane == 0) masks[sg] = mask;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!flag) return;
  uint at = popcount(mask & ((1u << lane) - 1u));
  for (uint g = 0; g < sg; ++g) at += popcount(masks[g]);
  float den = 1e-20f;
  for (uint g = 0; g < 8; ++g) {
    uint m = masks[g];
    while (m != 0u) {
      den += scores[16 * g + ctz(m)];
      m &= m - 1u;
    }
  }
  if (at < 8) {
    ids[(size_t)row * 8 + at] = t;
    weights[(size_t)row * 8 + at] = sc / den * 8.0f;
  }
}

#define SQRT5 0x1.1e377ap+1f

// commit1 bit 0: the kernel commits the history of a one-row launch. Bit 1: no reader needs the staged window (no
// transaction is open), so the tap stores are skipped.
static inline ushort conv_y(global const ushort *history, global const ushort *current, global ushort *taps,
                            global const ushort *weight, uint r, uint c, uint channels, uint commit1) {
  float window[4];
  for (uint tap = 0; tap < 4; ++tap) {
    const uint k = r + tap;
    window[tap] = bf2f(k < 3 ? history[(size_t)k * channels + c] : current[(size_t)(k - 3) * channels + c]);
  }
  if ((commit1 & 2) == 0) {
    if (r == 0)
      for (uint k = 0; k < 3; ++k) taps[(size_t)k * channels + c] = history[(size_t)k * channels + c];
    taps[(size_t)(r + 3) * channels + c] = current[(size_t)r * channels + c];
  }
  float sum = bf2f(current[(size_t)r * channels + c]);
  for (uint tap = 0; tap < 4; ++tap) sum += bf2f(weight[c * 4 + tap]) * window[tap];
  return f2bf(sum / SQRT5);
}

// The host commit for one row: the next history is history[1], history[2], current[0]. Read all three first.
static void commit3(global ushort *history, global const ushort *current, uint c, uint channels) {
  const ushort h1 = history[channels + c];
  const ushort h2 = history[2 * channels + c];
  const ushort cur = current[c];
  history[c] = h1;
  history[channels + c] = h2;
  history[2 * channels + c] = cur;
}

// Grid (ceil(channels/256), rows). The attention convolution, then the partial, with the bf16 round between them.
KERNEL(256) void tf_ink2_conv_attn_partial(
    global const ushort *history, global const ushort *current, global ushort *taps, global const ushort *weight,
    global ushort *conv_out, global const ushort *p, global ushort *blocks, global float *partial, uint rows, uint channels,
    uint boundary, uint commit1) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  const ushort y = conv_y(history, current, taps, weight, r, c, channels, commit1);
  conv_out[(size_t)r * channels + c] = y;
  const uint at = r * channels + c;
  partial[at] = boundary ? bf2f(y) : bf2f(p[at]) + bf2f(y);
  if (boundary) blocks[(size_t)r * 4 * channels + (boundary - 1) * channels + c] = p[at];
  if ((commit1 & 1) && rows == 1 && r == 0) commit3(history, current, c, channels);
}

// Grid (ceil(channels/256), rows). The MLP convolution, then the partial add into p.
KERNEL(256) void tf_ink2_conv_mlp_partial(
    global const ushort *history, global const ushort *current, global ushort *taps, global const ushort *weight,
    global ushort *conv_out, global const float *t, global ushort *p, uint rows, uint channels, uint commit1) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  const ushort y = conv_y(history, current, taps, weight, r, c, channels, commit1);
  conv_out[(size_t)r * channels + c] = y;
  const uint at = r * channels + c;
  p[at] = f2bf(t[at] + bf2f(y));
  if ((commit1 & 1) && rows == 1 && r == 0) commit3(history, current, c, channels);
}

// Grid (ceil(channels/256), rows). K folded into the V convolution. Each channel of each is the static formula.
KERNEL(256) void tf_ink2_conv_kv(
    global const ushort *hk, global const ushort *ck, global ushort *tk, global const ushort *wk, global ushort *ok,
    global const ushort *hv, global const ushort *cv, global ushort *tv, global const ushort *wv, global ushort *ov,
    uint rows, uint channels, uint commit1) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  ok[(size_t)r * channels + c] = conv_y(hk, ck, tk, wk, r, c, channels, commit1);
  ov[(size_t)r * channels + c] = conv_y(hv, cv, tv, wv, r, c, channels, commit1);
  if ((commit1 & 1) && rows == 1 && r == 0) {
    commit3(hk, ck, c, channels);
    commit3(hv, cv, c, channels);
  }
}

// Every row of the chunk. One thread owns a column and walks rows in order, so the convolution window
// sees this column's earlier combine results. The combine order matches the expert kernel. Row 0 matches the
// old one-row formula because conv_y at row 0 is that formula.
KERNEL(256) void tf_ink2_down_epilogue(
    global const ushort *down, global const int *ids, global const float *weights,
    global const ushort *history, global ushort *taps, global const ushort *conv_w, global ushort *current,
    global ushort *conv_out, global const float *partial, global ushort *p, uint n, uint topk, uint channels, uint rows, uint commit1) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), t = get_local_id(0);
  local uint order[8];
  for (uint r = 0; r < rows; ++r) {
    global const int *id_row = ids + (size_t)r * topk;
    global const float *w_row = weights + (size_t)r * topk;
    // One thread a slot works out where its slot sits in the expert order the serial selection gives, so every
    // channel reads the order from SLM instead of replaying the scan over global memory.
    if (t < topk && t < 8) {
      const int me = id_row[t];
      uint rank = 0;
      for (uint s = 0; s < 8; ++s) {
        if (s >= topk) break;
        const int id = id_row[s];
        rank += (id < me || (id == me && s < t)) ? 1u : 0u;
      }
      order[rank] = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (c < n && c < channels) {
      float sum = 0.0f;
      for (uint k = 0; k < 8; ++k) {
        if (k >= topk) break;
        const uint slot = order[k];
        sum += w_row[slot] * bf2f(down[((size_t)r * topk + slot) * n + c]);
      }
      const ushort y = f2bf(sum);
      current[(size_t)r * channels + c] = y;
      const ushort cy = conv_y(history, current, taps, conv_w, r, c, channels, commit1);
      conv_out[(size_t)r * channels + c] = cy;
      const uint at = r * channels + c;
      p[at] = FUSE_EPI ? f2bf(partial[at] + bf2f(cy)) : (ushort)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((commit1 & 1) && rows == 1 && c < n && c < channels) commit3(history, current, c, channels);
}

// Empty bodies at the three work-group sizes of the decode kernels: the fixed cost of a launch at a grid
// (INK2_EMPTY_PROBE=1, forward.zig). Nothing reads or writes memory.
KERNEL(64) void tf_ink2_empty64(uint unused) {}
KERNEL(128) void tf_ink2_empty128(uint unused) {}
KERNEL(256) void tf_ink2_empty256(uint unused) {}

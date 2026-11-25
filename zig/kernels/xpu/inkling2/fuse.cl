// Phase A as parallel launches: one group a row for the residual, and the controller ranks spread as in loop.cl.
// Arithmetic and reduction order match glue.cl, loop.cl and dense.cl. Roles of one visit share a launch.

#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel
#define H 2304
#define EPS 1e-6f
#define AR_SCALE 0x1.555556p-6f
#define RANK 64

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); }
static inline float rbf(float f) { return bf2f(f2bf(f)); }

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
    for (uint s = 0; s < 5; ++s) {
      float ss = 0.0f;
      for (uint j = 0; j < 16; ++j) ss += red[s * 16 + j];
      means[s] = rsqrt(ss / 2304.0f + EPS);
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
      for (uint s = 0; s < 5; ++s) {
        float dot = 0.0f;
        for (uint j = 0; j < 16; ++j) dot += dots[s * 16 + j];
        weights[s] = dot * AR_SCALE;
        if (s < sources) peak = peak < weights[s] ? weights[s] : peak;
      }
      float den = 0.0f;
      for (uint s = 0; s < 5; ++s) {
        weights[s] = exp(weights[s] - peak);
        if (s < sources) den += weights[s];
      }
      for (uint s = 0; s < 5; ++s) weights[s] = weights[s] / den;
      global float *dst = ar + ((size_t)q * rows + row) * H;
      for (uint i = 0; i < H / 256; ++i) {
        float acc = 0.0f;
        for (uint s = 0; s < 5; ++s)
          if (s < sources) acc += weights[s] * (v[s][i] * means[s]);
        dst[t + 256 * i] = acc;
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
    if (lane == 0) zeta[((size_t)row * 4 + role) * RANK + r] = rbf(dot);
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
  if (lane == 0) zeta[pack4 ? ((size_t)row * 4 + role) * RANK + r : (size_t)row * RANK + r] = rbf(dot);
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
  const float right = g * m[d];
  gated[((size_t)role * rows + row) * H + d] = left + right;
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

static inline float4 rx4u(local const ushort *n, uint i) {
  const uint b = i * 4;
  return (float4)(bf2f(n[b]), bf2f(n[b + 1]), bf2f(n[b + 2]), bf2f(n[b + 3]));
}

// The dense norm, then the wide router GEMV. Group 0 stores the bf16 row. The GEMV reads that row from local.
KERNEL(256) void tf_ink2_router_norm_gemv(
    global const float *x, global const ushort *nw, global const ushort *weight, global ushort *normed_out,
    global float *scores, uint rows) {
  const uint row = get_group_id(1), lane = get_sub_group_local_id(), sg = get_sub_group_id(), t = get_local_id(0);
  if (row >= rows) return;
  local float red[16];
  local ushort normed[H];
  global const float *xr = x + (size_t)row * H;
  float sum = 0.0f;
  for (uint d = t; d < H; d += 256) sum = fma(xr[d], xr[d], sum);
  const float s = sub_group_reduce_add(sum);
  if (lane == 0) red[sg] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float total = 0.0f;
  for (uint i = 0; i < 16; ++i) total += red[i];
  const float inverse = 1.0f / sqrt(total / (float)H + EPS);
  for (uint d = t; d < H; d += 256) normed[d] = f2bf(bf2f(nw[d]) * (xr[d] * inverse));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_group_id(0) == 0) {
    for (uint d = t; d < H; d += 256) normed_out[(size_t)row * H + d] = normed[d];
  }
  const uint expert = get_group_id(0) * 16 + sg;
  if (expert >= 128) return;
  global const uint4 *w = (global const uint4 *)(weight + (size_t)expert * H);
  float dot = 0.0f;
  for (uint part = 0; part < 2; ++part) {
    for (uint u = 0; u < 9; ++u) {
      const uint at = (part * 9 + u) * 16 + lane;
      const uint4 a = w[(part * 9 + u) * 16 + lane];
      const float4 b0 = rx4u(normed, 2 * at), b1 = rx4u(normed, 2 * at + 1);
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

// Grid (rows). Same tie rule as tf_inkling2_router: a larger key wins, an equal key keeps the lower index.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1))) kernel void tf_ink2_router_pick(
    global const float *scores_in, global const float *bias, global uint *ids, global float *weights, uint rows) {
  const uint row = get_group_id(0), t = get_local_id(0);
  if (row >= rows) return;
  local float keys[128], scores[128], chosen[8];
  local uint flag[128], picked[8];
  const float sc = scores_in[(size_t)row * 128 + t];
  scores[t] = sc;
  keys[t] = sc + bias[t];
  barrier(CLK_LOCAL_MEM_FENCE);
  const float ke = keys[t];
  uint rank = 0;
  for (uint j = 0; j < 128; ++j) {
    const float kj = keys[j];
    rank += (kj > ke || (kj == ke && j < t)) ? 1u : 0u;
  }
  flag[t] = rank < 8 ? 1u : 0u;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (flag[t]) {
    uint at = 0;
    for (uint j = 0; j < t; ++j) at += flag[j];
    picked[at] = t;
    chosen[at] = scores[t];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    float den = 1e-20f;
    for (uint j = 0; j < 8; ++j) den += chosen[j];
    for (uint j = 0; j < 8; ++j) {
      ids[(size_t)row * 8 + j] = picked[j];
      weights[(size_t)row * 8 + j] = chosen[j] / den * 8.0f;
    }
  }
}

#define SQRT5 0x1.1e377ap+1f

static inline ushort conv_y(global const ushort *history, global const ushort *current, global ushort *taps,
                            global const ushort *weight, uint r, uint c, uint channels) {
  float window[4];
  for (uint tap = 0; tap < 4; ++tap) {
    const uint k = r + tap;
    window[tap] = bf2f(k < 3 ? history[(size_t)k * channels + c] : current[(size_t)(k - 3) * channels + c]);
  }
  if (r == 0)
    for (uint k = 0; k < 3; ++k) taps[(size_t)k * channels + c] = history[(size_t)k * channels + c];
  taps[(size_t)(r + 3) * channels + c] = current[(size_t)r * channels + c];
  float sum = bf2f(current[(size_t)r * channels + c]);
  for (uint tap = 0; tap < 4; ++tap) sum += bf2f(weight[c * 4 + tap]) * window[tap];
  return f2bf(sum / SQRT5);
}

// Grid (ceil(channels/256), rows). The attention convolution, then the partial, with the bf16 round between them.
KERNEL(256) void tf_ink2_conv_attn_partial(
    global const ushort *history, global const ushort *current, global ushort *taps, global const ushort *weight,
    global ushort *conv_out, global const ushort *p, global ushort *blocks, global float *partial, uint rows, uint channels,
    uint boundary) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  const ushort y = conv_y(history, current, taps, weight, r, c, channels);
  conv_out[(size_t)r * channels + c] = y;
  const uint at = r * channels + c;
  partial[at] = boundary ? bf2f(y) : bf2f(p[at]) + bf2f(y);
  if (boundary) blocks[(size_t)r * 4 * channels + (boundary - 1) * channels + c] = p[at];
}

// Grid (ceil(channels/256), rows). The MLP convolution, then the partial add into p.
KERNEL(256) void tf_ink2_conv_mlp_partial(
    global const ushort *history, global const ushort *current, global ushort *taps, global const ushort *weight,
    global ushort *conv_out, global const float *t, global ushort *p, uint rows, uint channels) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  const ushort y = conv_y(history, current, taps, weight, r, c, channels);
  conv_out[(size_t)r * channels + c] = y;
  const uint at = r * channels + c;
  p[at] = f2bf(t[at] + bf2f(y));
}

// Grid (ceil(channels/256), rows). K folded into the V convolution. Each channel of each is the static formula.
KERNEL(256) void tf_ink2_conv_kv(
    global const ushort *hk, global const ushort *ck, global ushort *tk, global const ushort *wk, global ushort *ok,
    global const ushort *hv, global const ushort *cv, global ushort *tv, global const ushort *wv, global ushort *ov,
    uint rows, uint channels) {
  const uint c = get_group_id(0) * 256 + get_local_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  ok[(size_t)r * channels + c] = conv_y(hk, ck, tk, wk, r, c, channels);
  ov[(size_t)r * channels + c] = conv_y(hv, cv, tv, wv, r, c, channels);
}

// One row. Combine in expert order, round to bf16, then the MLP convolution and the residual add into p.
KERNEL(256) void tf_ink2_down_epilogue(
    global const ushort *down, global const int *ids, global const float *weights,
    global const ushort *history, global ushort *taps, global const ushort *conv_w, global ushort *conv_out,
    global const float *partial, global ushort *p, uint n, uint topk, uint channels) {
  const uint c = get_group_id(0) * 256 + get_local_id(0);
  if (c >= n || c >= channels) return;
  float sum = 0.0f;
  int last_id = INT_MIN;
  int last_slot = -1;
  for (uint t = 0; t < topk; ++t) {
    int best_id = INT_MAX;
    int best_slot = -1;
    for (uint slot = 0; slot < topk; ++slot) {
      const int id = ids[slot];
      const bool after = id > last_id || (id == last_id && (int)slot > last_slot);
      if (after && (best_slot < 0 || id < best_id)) {
        best_id = id;
        best_slot = (int)slot;
      }
    }
    if (best_slot < 0) break;
    sum += weights[best_slot] * bf2f(down[(size_t)best_slot * n + c]);
    last_id = best_id;
    last_slot = best_slot;
  }
  const ushort y = f2bf(sum);
  conv_out[c] = y;
  float window[4];
  for (uint tap = 0; tap < 4; ++tap) {
    window[tap] = bf2f(tap < 3 ? history[(size_t)tap * channels + c] : y);
  }
  for (uint k = 0; k < 3; ++k) taps[(size_t)k * channels + c] = history[(size_t)k * channels + c];
  taps[3 * channels + c] = y;
  float acc = bf2f(y);
  for (uint tap = 0; tap < 4; ++tap) acc += bf2f(conv_w[c * 4 + tap]) * window[tap];
  const ushort cy = f2bf(acc / SQRT5);
  p[c] = f2bf(partial[c] + bf2f(cy));
}

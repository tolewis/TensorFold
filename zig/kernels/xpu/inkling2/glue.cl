// Inkling 2 Mini glue on Intel Xe2 (OpenCL C), the port of kernels/metal/inkling2/glue.metal.
// Every reduction is FP32; bf16 values are ushort bits and every bf16 store rounds to nearest even.
// Build with -cl-fp32-correctly-rounded-divide-sqrt so that `/` and sqrt are IEEE, as Metal's precise:: calls are.
//
// Layout rule for the contract's control generator: helpers come first, then one kernel at a time, each starting on
// the line that holds its `void tf_inkling2_<name>(` and ending at the first `}` in column 0.

#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

#define INK_HIDDEN 2304
#define INK_EPS 1e-6f
#define INK_AR_SCALE 0x1.555556p-6f // binary32(1 / 48)
#define INK_SQRT5 0x1.1e377ap+1f    // binary32(sqrt(5))

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even

static inline float ink_sigmoid(float x) {
  const float e = exp(-fabs(x));
  return x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
}
static inline float ink_cap(float x, float c) { return c * tanh(x / c); }
static inline float ink_value(global const ushort *blocks, global const float *partial, uint source, uint sources, uint d) {
  return source + 1 == sources ? partial[d] : bf2f(blocks[(ulong)source * INK_HIDDEN + d]);
}

// The sum of one value a work-item over a work-group of 256: the 16 sub-group sums, added in order, in every item.
static float ink_wg_sum(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// Grid (rows). out[row] = bf16(weight * (table[ids[row]] * rsqrt(mean(table[ids[row]]^2) + eps))).
KERNEL(256) void tf_inkling2_embed_norm(global const uint *ids, global const ushort *table, global const ushort *weight,
                                        global ushort *out, uint rows) {
  const uint row = get_group_id(0), t = get_local_id(0);
  if (row >= rows) return;
  local float red[16];
  global const ushort *x = table + (ulong)ids[row] * INK_HIDDEN;
  float part = 0.0f;
  for (uint d = t; d < INK_HIDDEN; d += 256) {
    const float v = bf2f(x[d]);
    part += v * v;
  }
  const float total = ink_wg_sum(part, red);
  const float inverse = rsqrt(total / 2304.0f + INK_EPS);
  for (uint d = t; d < INK_HIDDEN; d += 256) out[(ulong)row * INK_HIDDEN + d] = f2bf(bf2f(weight[d]) * (bf2f(x[d]) * inverse));
}

// Grid (queries). sources are completed bf16 blocks plus one unrounded fp32 partial (the last source); one fp32 row a query.
// Every source value of a work-item (its 9 channels of each source) is loaded once, up front, and kept in registers;
// the source loops have constant bounds so the arrays stay in registers, and a source past `sources` is skipped.
KERNEL(256) void tf_inkling2_ar(global const ushort *blocks, global const float *partial, global const float *queries,
                                global float *out, uint sources, uint count) {
  const uint q = get_group_id(0), t = get_local_id(0), sg = get_sub_group_id();
  if (q >= count || sources == 0 || sources > 5) return;
  local float red[160]; // 5 sources of 16 sub-group sums for the squares, then for the dots
  float v[5][INK_HIDDEN / 256];
  float query[INK_HIDDEN / 256];
#pragma unroll
  for (uint s = 0; s < 5; ++s)
#pragma unroll
    for (uint i = 0; i < INK_HIDDEN / 256; ++i) v[s][i] = s < sources ? ink_value(blocks, partial, s, sources, t + 256 * i) : 0.0f;
#pragma unroll
  for (uint i = 0; i < INK_HIDDEN / 256; ++i) query[i] = queries[(ulong)q * INK_HIDDEN + t + 256 * i];
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float ss = 0.0f;
#pragma unroll
    for (uint i = 0; i < INK_HIDDEN / 256; ++i) ss += v[s][i] * v[s][i];
    const float reduced = sub_group_reduce_add(ss);
    if (s < sources && get_sub_group_local_id() == 0) red[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float means[5];
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 16; ++j) ss += red[s * 16 + j];
    means[s] = rsqrt(ss / 2304.0f + INK_EPS);
  }
  local float *dots = red + 80;
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float dot = 0.0f;
#pragma unroll
    for (uint i = 0; i < INK_HIDDEN / 256; ++i) dot += query[i] * (v[s][i] * means[s]);
    const float reduced = sub_group_reduce_add(dot);
    if (s < sources && get_sub_group_local_id() == 0) dots[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float weights[5];
  float peak = -INFINITY;
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float dot = 0.0f;
    for (uint j = 0; j < 16; ++j) dot += dots[s * 16 + j];
    weights[s] = dot * INK_AR_SCALE;
    if (s < sources) peak = peak < weights[s] ? weights[s] : peak;
  }
  float den = 0.0f;
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    weights[s] = exp(weights[s] - peak);
    if (s < sources) den += weights[s];
  }
#pragma unroll
  for (uint s = 0; s < 5; ++s) weights[s] = weights[s] / den;
#pragma unroll
  for (uint i = 0; i < INK_HIDDEN / 256; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (uint s = 0; s < 5; ++s)
      if (s < sources) acc += weights[s] * (v[s][i] * means[s]);
    out[(ulong)q * INK_HIDDEN + t + 256 * i] = acc;
  }
}

// Grid (ceil(channels / 256), rows). The window of a row is three old rows (history) and then the current rows, as the
// cache stages them: the kernel also writes that window to `taps` (rows 0 to 2 from the work-items of row 0), so no
// separate copies are needed. weight is [channel][4].
KERNEL(256) void tf_inkling2_conv_static(global const ushort *history, global const ushort *current, global ushort *taps,
                                         global const ushort *weight, global ushort *out, uint rows, uint channels) {
  const uint c = get_global_id(0), r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  float window[4];
  for (uint tap = 0; tap < 4; ++tap) {
    const uint k = r + tap;
    window[tap] = bf2f(k < 3 ? history[(ulong)k * channels + c] : current[(ulong)(k - 3) * channels + c]);
  }
  if (r == 0)
    for (uint k = 0; k < 3; ++k) taps[(ulong)k * channels + c] = history[(ulong)k * channels + c];
  taps[(ulong)(r + 3) * channels + c] = current[(ulong)r * channels + c];
  float sum = bf2f(current[(ulong)r * channels + c]);
  for (uint tap = 0; tap < 4; ++tap) sum += bf2f(weight[c * 4 + tap]) * window[tap];
  out[(ulong)r * channels + c] = f2bf(sum / INK_SQRT5);
}

// Grid (rows). static_w is [lag][channel]; projection is [32][2304]; four lag weights a group of 128 channels. history,
// current and taps are as in the static convolution: the window is written to `taps`.
KERNEL(256) void tf_inkling2_conv_dynamic(global const ushort *history, global const ushort *current, global ushort *taps,
                                          global const ushort *input, global const ushort *static_w, global const ushort *projection,
                                          global ushort *out, uint rows, uint channels) {
  const uint r = get_group_id(0), t = get_local_id(0);
  if (r >= rows) return;
  local float projected[32];
  local float4 xs[INK_HIDDEN / 4];
  global const uint2 *xg = (global const uint2 *)(input + (ulong)r * INK_HIDDEN);
  for (uint i = t; i < INK_HIDDEN / 4; i += 256) {
    const uint2 v = xg[i];
    xs[i] = (float4)(as_float(v.x << 16), as_float(v.x & 0xffff0000u), as_float(v.y << 16), as_float(v.y & 0xffff0000u));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 32) {
    // One work-item adds one projected value's products in order d = 0 to 2303, each product rounded, then added.
    global const uint4 *w = (global const uint4 *)(projection + (ulong)t * INK_HIDDEN);
    float dot = 0.0f;
    for (uint i = 0; i < INK_HIDDEN / 8; i += 4) {
      uint4 a[4];
      for (uint u = 0; u < 4; ++u) a[u] = w[i + u];
      for (uint u = 0; u < 4; ++u) {
        const float4 b0 = xs[2 * (i + u)], b1 = xs[2 * (i + u) + 1];
        dot += as_float(a[u].x << 16) * b0.x;
        dot += as_float(a[u].x & 0xffff0000u) * b0.y;
        dot += as_float(a[u].y << 16) * b0.z;
        dot += as_float(a[u].y & 0xffff0000u) * b0.w;
        dot += as_float(a[u].z << 16) * b1.x;
        dot += as_float(a[u].z & 0xffff0000u) * b1.y;
        dot += as_float(a[u].w << 16) * b1.z;
        dot += as_float(a[u].w & 0xffff0000u) * b1.w;
      }
    }
    projected[t] = bf2f(f2bf(dot));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint c = t; c < channels; c += 256) {
    float window[4]; // window[j] is the row r + 3 - j of the history-then-current rows
    for (uint lag = 0; lag < 4; ++lag) {
      const uint k = r + 3 - lag;
      window[lag] = bf2f(k < 3 ? history[(ulong)k * channels + c] : current[(ulong)(k - 3) * channels + c]);
    }
    if (r == 0)
      for (uint k = 0; k < 3; ++k) taps[(ulong)k * channels + c] = history[(ulong)k * channels + c];
    taps[(ulong)(r + 3) * channels + c] = current[(ulong)r * channels + c];
    float sum = bf2f(current[(ulong)r * channels + c]);
    for (uint lag = 0; lag < 4; ++lag) {
      const float w = bf2f(static_w[lag * channels + c]) + projected[4 * (c / 128) + lag];
      sum += w * window[lag];
    }
    out[(ulong)r * channels + c] = f2bf(sum / 3.0f);
  }
}

// Grid (rows). Sigmoid scores of 128 experts; the top 8 by score + bias, sorted by ascending expert index.
// Work-item e < 128 adds the products of expert e's row in order d = 0 to 2303 (each product rounded, then added), so
// the score of an expert does not depend on how many work-items run. The row of x is staged once in local memory.
KERNEL(256) void tf_inkling2_router(global const ushort *x, global const ushort *weight, global const float *bias,
                                    global uint *ids, global float *weights, uint rows) {
  const uint row = get_group_id(0), t = get_local_id(0);
  if (row >= rows) return;
  local float4 xs[INK_HIDDEN / 4];
  local float scores[128], keys[128], chosen[8];
  local uint picked[8];
  global const uint2 *xg = (global const uint2 *)(x + (ulong)row * INK_HIDDEN);
  for (uint i = t; i < INK_HIDDEN / 4; i += 256) {
    const uint2 v = xg[i];
    xs[i] = (float4)(as_float(v.x << 16), as_float(v.x & 0xffff0000u), as_float(v.y << 16), as_float(v.y & 0xffff0000u));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 128) {
    global const uint4 *w = (global const uint4 *)(weight + (ulong)t * INK_HIDDEN);
    float dot = 0.0f;
    for (uint i = 0; i < INK_HIDDEN / 8; i += 4) {
      uint4 a[4];
      for (uint u = 0; u < 4; ++u) a[u] = w[i + u];
      for (uint u = 0; u < 4; ++u) {
        const float4 b0 = xs[2 * (i + u)], b1 = xs[2 * (i + u) + 1];
        dot += as_float(a[u].x << 16) * b0.x;
        dot += as_float(a[u].x & 0xffff0000u) * b0.y;
        dot += as_float(a[u].y << 16) * b0.z;
        dot += as_float(a[u].y & 0xffff0000u) * b0.w;
        dot += as_float(a[u].z << 16) * b1.x;
        dot += as_float(a[u].z & 0xffff0000u) * b1.y;
        dot += as_float(a[u].w << 16) * b1.z;
        dot += as_float(a[u].w & 0xffff0000u) * b1.w;
      }
    }
    scores[t] = ink_sigmoid(dot);
    keys[t] = scores[t] + bias[t];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // The first sub-group picks: lane l owns experts 8l to 8l + 7. A pick is the largest key, the lowest index on a tie;
  // a NaN key never beats another, so it is picked only when it is the first expert left (as a scan in index order does).
  if (get_sub_group_id() == 0) {
    const uint lane = get_sub_group_local_id();
    float mine[8];
    for (uint j = 0; j < 8; ++j) mine[j] = keys[lane * 8 + j];
    uint taken = 0;
    for (uint j = 0; j < 8; ++j) {
      uint first_here = 128u;
      for (int e = 7; e >= 0; --e)
        if (!(taken & (1u << e))) first_here = lane * 8 + e;
      const uint first = sub_group_reduce_min(first_here);
      uint local_best = 8;
      for (uint e = 0; e < 8; ++e)
        if (!(taken & (1u << e)) && !isnan(mine[e]) && (local_best == 8 || mine[e] > mine[local_best])) local_best = e;
      const float local_key = local_best == 8 ? -INFINITY : mine[local_best];
      const float top = sub_group_reduce_max(local_key);
      const uint candidate = (local_best != 8 && local_key == top) ? lane * 8 + local_best : 128u;
      const uint largest = sub_group_reduce_min(candidate);
      const uint best = isnan(keys[first]) || largest == 128u ? first : largest;
      if (best / 8 == lane) taken |= 1u << (best % 8);
      if (lane == 0) {
        picked[j] = best;
        chosen[j] = scores[best];
      }
    }
    if (lane == 0) {
      // Combining, including the denominator, is in ascending expert index.
      for (uint j = 1; j < 8; ++j) {
        const uint id = picked[j];
        const float score = chosen[j];
        uint k = j;
        while (k > 0 && picked[k - 1] > id) {
          picked[k] = picked[k - 1];
          chosen[k] = chosen[k - 1];
          --k;
        }
        picked[k] = id;
        chosen[k] = score;
      }
      float den = 1e-20f;
      for (uint j = 0; j < 8; ++j) den += chosen[j];
      for (uint j = 0; j < 8; ++j) {
        ids[row * 8 + j] = picked[j];
        weights[row * 8 + j] = chosen[j] / den * 8.0f;
      }
    }
  }
}

// Grid (ceil(width / 256), rows). gu is [row][width][gate, up] bf16.
KERNEL(256) void tf_inkling2_swiglu(global const ushort *gu, global ushort *out, uint rows, uint width) {
  const uint j = get_global_id(0), r = get_group_id(1);
  if (j >= width || r >= rows) return;
  const float g = bf2f(gu[((ulong)r * width + j) * 2]);
  const float u = bf2f(gu[((ulong)r * width + j) * 2 + 1]);
  out[(ulong)r * width + j] = f2bf(ink_cap(g, 4.0f) * ink_sigmoid(g) * ink_cap(u, 25.0f));
}

// Grid (ceil(vocab / 16), rows). One sub-group a vocabulary row: the lanes take 8 values each in turn, the sum is FP32.
KERNEL(256) void tf_inkling2_logits(global const ushort *h, global const ushort *unembed, global float *out, uint rows,
                                    uint vocab) {
  const uint v = get_group_id(0) * 16 + get_sub_group_id(), r = get_group_id(1), lane = get_sub_group_local_id();
  if (v >= vocab || r >= rows) return;
  global const ushort *w = unembed + (ulong)v * INK_HIDDEN;
  global const ushort *x = h + (ulong)r * INK_HIDDEN;
  float dot = 0.0f;
  for (uint i = lane; i < INK_HIDDEN / 8; i += 16) {
    const ushort8 a = vload8(i, w), b = vload8(i, x);
    float part = bf2f(a.s0) * bf2f(b.s0);
    part += bf2f(a.s1) * bf2f(b.s1);
    part += bf2f(a.s2) * bf2f(b.s2);
    part += bf2f(a.s3) * bf2f(b.s3);
    part += bf2f(a.s4) * bf2f(b.s4);
    part += bf2f(a.s5) * bf2f(b.s5);
    part += bf2f(a.s6) * bf2f(b.s6);
    part += bf2f(a.s7) * bf2f(b.s7);
    dot += part;
  }
  dot = sub_group_reduce_add(dot);
  // The unembedding rounds to bf16 before it widens for softcap30.
  if (lane == 0) out[(ulong)r * vocab + v] = ink_cap(bf2f(f2bf(dot)), 30.0f);
}

// Grid (ceil(channels / 256), 8, rows). experts is [row][128][channels]; out is [row][8][channels].
KERNEL(256) void tf_inkling2_expert_pick(global const ushort *experts, global const uint *ids, global ushort *out, uint rows,
                                         uint channels) {
  const uint d = get_global_id(0), j = get_group_id(1), r = get_group_id(2);
  if (d >= channels || r >= rows) return;
  out[((ulong)r * 8 + j) * channels + d] = experts[((ulong)r * 128 + ids[r * 8 + j]) * channels + d];
}

// Grid (ceil(channels / 256), rows). The eight picked rows are added in order j = 0 to 7 in FP32.
KERNEL(256) void tf_inkling2_expert_combine(global const ushort *picked, global const float *weights, global ushort *out,
                                            uint rows, uint channels) {
  const uint d = get_global_id(0), r = get_group_id(1);
  if (d >= channels || r >= rows) return;
  float acc = 0.0f;
  for (uint j = 0; j < 8; ++j) acc += weights[r * 8 + j] * bf2f(picked[((ulong)r * 8 + j) * channels + d]);
  out[(ulong)r * channels + d] = f2bf(acc);
}

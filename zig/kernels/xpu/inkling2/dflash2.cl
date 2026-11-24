// Inkling 2 Mini's DFlash 2 block drafter on Intel Xe2 (OpenCL C): the arithmetic that has no kernel in dense.cl or
// dspark.cl. Five layers of a width-2304 Qwen3-style block (32 query heads and 8 key/value heads of 128, window 2,048,
// rotate-half RoPE with theta 10,000), a grouped dynamic two-tap convolution before the attention and the MLP and after
// each, and a candidate head that keeps the 16 best logits of every block row.
//
// The model follows DFlash 2 (Inco AI, Apache 2.0, model card incoai/Inkling-2-Mini-DFlash2) and the reference in
// z-lab/dflash (MIT). The projections run on the dense matvec; the residual norm, the swiglu and the base projection
// are the DSpark kernels of the same width.
//
// Every dot is a serial FP32 chain (multiply, then add, never fused); BF16 stores round to nearest even. Build with
// -cl-fp32-correctly-rounded-divide-sqrt.

#pragma OPENCL FP_CONTRACT OFF

#define DF_H 2304
#define DF_Q 4096
#define DF_KV 1024
#define DF_HEAD 128
#define DF_QHEADS 32
#define DF_KVHEADS 8
#define DF_GROUPS 144
#define DF_GROUP 16
#define DF_DYN 576
#define DF_BLOCK 8
#define DF_EPS 1e-6f
#define DF_SCALE 0x1.6a09e6p-4f
#define DF_THETA 10000.0f
#define DF_CHUNK 128
#define DF_TOPK 16
#define DF_TOPK_GROUPS 256

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// The contract plants one arithmetic mistake in each kernel through the macros below; production keeps the defaults.
// contract_dflash2_controls.cl redefines them (and DF_NAME) before it includes this file.
#ifndef DF_NAME
#define DF_NAME(x) tf_ink2_dflash2_##x
#endif
#ifndef DF_CONV_KERNEL
#define DF_CONV_KERNEL(base, dyn) ((base) + (dyn))
#endif
#ifndef DF_QK_ROPE_HI
#define DF_QK_ROPE_HI(x, other, cs, sn) ((other) * (sn) + (x) * (cs))
#endif
#ifndef DF_CTX_ROPE_HI
#define DF_CTX_ROPE_HI(x, other, cs, sn) ((other) * (sn) + (x) * (cs))
#endif
#ifndef DF_FIRST
#define DF_FIRST(from, position, row) df_first((from), (position), (row))
#endif
#ifndef DF_MERGE_WEIGHT
#define DF_MERGE_WEIGHT(m, mx) exp((m) - (mx))
#endif
#ifndef DF_TOPK_VALUE
#define DF_TOPK_VALUE(v) (v)
#endif

// The first context row a block row at position committed + row sees: the window is 2,048 positions.
static inline uint df_first(uint from, uint committed, uint row) {
  const uint low = committed + row > 2047u ? committed + row - 2047u : 0u;
  return max(from, low);
}

static inline float df_angle(uint d, uint position) {
  const float frequency = pow(DF_THETA, -(float)(d % 64) / 64.0f);
  return (float)position * frequency;
}

// y[row][c] = bf16(sum over off in {0, 1} of (base[off][c] + dyn[row][off * 144 + c / 16]) * x[row - off][c]).
// The block's first row has no earlier row (the convolution is causal inside the block). `dyn` points at the branch's
// first column of the kernel projection ([row][576]); `base` at the branch's [2][2304] taps.
KERNEL(256) void DF_NAME(conv)(global const ushort *x, global const ushort *dyn, global const ushort *base, global ushort *y) {
  const uint row = get_group_id(0), c0 = get_group_id(1) * 256 + get_local_id(0);
  if (c0 >= DF_H) return;
  const uint g = c0 / DF_GROUP;
  float sum = 0.0f;
  for (uint off = 0; off < 2; ++off) {
    if (row < off) continue;
    const float k = DF_CONV_KERNEL(bf2f(base[off * DF_H + c0]), bf2f(dyn[row * DF_DYN + off * DF_GROUPS + g]));
    sum += k * bf2f(x[(row - off) * DF_H + c0]);
  }
  y[row * DF_H + c0] = f2bf(sum);
}

// Per-head RMS norm with a learned weight, then rotate-half RoPE, for the block's queries (32 heads) and keys (8).
// Rows are positions base .. base + 7. V needs nothing: the attention reads the projection's output.
KERNEL(128) void DF_NAME(block_qk)(global const ushort *q_raw, global const ushort *k_raw, global const ushort *q_weight,
                                   global const ushort *k_weight, global ushort *q, global ushort *k, uint base) {
  const uint group = get_group_id(0), d = get_local_id(0);
  const uint row = group / 40, work = group % 40;
  const bool query = work < 32;
  const uint h = query ? work : work - 32;
  global const ushort *raw = query ? q_raw + row * DF_Q + h * DF_HEAD : k_raw + row * DF_KV + h * DF_HEAD;
  global const ushort *weight = query ? q_weight : k_weight;
  local float scale;
  if (d == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < DF_HEAD; ++i) {
      const float x = bf2f(raw[i]);
      sum += x * x;
    }
    scale = 1.0f / sqrt(sum / (float)DF_HEAD + DF_EPS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint partner = d < 64 ? d + 64 : d - 64;
  const float x = rbf(bf2f(raw[d]) * scale * bf2f(weight[d]));
  const float other = rbf(bf2f(raw[partner]) * scale * bf2f(weight[partner]));
  const float angle = df_angle(d, base + row);
  const float cs = cos(angle), sn = sin(angle);
  const float rotated = d < 64 ? x * cs - other * sn : DF_QK_ROPE_HI(x, other, cs, sn);
  if (query) q[row * DF_Q + h * DF_HEAD + d] = f2bf(rotated);
  else k[row * DF_KV + h * DF_HEAD + d] = f2bf(rotated);
}

// Committed rows' context keys and values land at absolute-position modulo ring: keys normalized and rotated, values copied.
KERNEL(128) void DF_NAME(context_kv)(global const ushort *k_raw, global const ushort *v_raw, global const ushort *weight,
                                     global ushort *keys, global ushort *values, uint base, uint ring) {
  const uint group = get_group_id(0), d = get_local_id(0);
  const uint row = group / DF_KVHEADS, h = group % DF_KVHEADS;
  local float scale;
  if (d == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < DF_HEAD; ++i) {
      const float x = bf2f(k_raw[row * DF_KV + h * DF_HEAD + i]);
      sum += x * x;
    }
    scale = 1.0f / sqrt(sum / (float)DF_HEAD + DF_EPS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint position = base + row;
  const uint at = (position % ring) * DF_KV + h * DF_HEAD + d;
  const uint src = row * DF_KV + h * DF_HEAD;
  const uint partner = d < 64 ? d + 64 : d - 64;
  const float x = rbf(bf2f(k_raw[src + d]) * scale * bf2f(weight[d]));
  const float other = rbf(bf2f(k_raw[src + partner]) * scale * bf2f(weight[partner]));
  const float angle = df_angle(d, position);
  const float cs = cos(angle), sn = sin(angle);
  keys[at] = f2bf(d < 64 ? x * cs - other * sn : DF_CTX_ROPE_HI(x, other, cs, sn));
  values[at] = v_raw[src + d];
}

// Block attention, first pass. A row at position P + r sees the context rows max(from, P + r - 2047) .. P - 1 (kept in
// a ring) and all eight block rows (the block is not causal). One work-group takes one (row, key/value head, chunk of 128
// keys): four query heads share the chunk's keys. Output: the chunk's running max, sum of exponentials and weighted
// values, per query head: part[(row * 8 + kv) * chunks + chunk] holds m[4], l[4], o[4][128] as 4 + 4 + 512 floats.
KERNEL(128) void DF_NAME(attention)(global const ushort *q, global const ushort *block_k, global const ushort *block_v,
                                    global const ushort *context_k, global const ushort *context_v, global float *part,
                                    uint committed, uint ring, uint from, uint chunks) {
  const uint t = get_local_id(0);
  const uint chunk = get_group_id(0), kv = get_group_id(1), row = get_group_id(2);
  const uint first = DF_FIRST(from, committed, row);
  const uint retained = committed > first ? committed - first : 0u;
  const uint count = retained + DF_BLOCK;
  global float *out = part + ((ulong)(row * DF_KVHEADS + kv) * chunks + chunk) * 520;
  local float ql[4][DF_HEAD];
  local float p[4][DF_CHUNK];
  local float red[4][8];
  for (uint a = 0; a < 4; ++a) ql[a][t] = bf2f(q[row * DF_Q + (kv * 4 + a) * DF_HEAD + t]);
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint j = chunk * DF_CHUNK + t;
  const bool live = j < count;
  float score[4];
  for (uint a = 0; a < 4; ++a) score[a] = -INFINITY;
  if (live) {
    global const ushort *key = j < retained ? context_k + (ulong)((first + j) % ring) * DF_KV + kv * DF_HEAD
                                            : block_k + (j - retained) * DF_KV + kv * DF_HEAD;
    float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < DF_HEAD; i += 8) {
      const ushort8 x = vload8(0, key + i);
      const float kf[8] = {bf2f(x.s0), bf2f(x.s1), bf2f(x.s2), bf2f(x.s3), bf2f(x.s4), bf2f(x.s5), bf2f(x.s6), bf2f(x.s7)};
      for (uint a = 0; a < 4; ++a) {
        for (uint e = 0; e < 8; ++e) dot[a] += ql[a][i + e] * kf[e];
      }
    }
    for (uint a = 0; a < 4; ++a) score[a] = dot[a] * DF_SCALE;
  }
  // chunk maxima: sub-group reduce, then eight sub-groups
  const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
  float mx[4];
  for (uint a = 0; a < 4; ++a) {
    const float s = sub_group_reduce_max(score[a]);
    if (lane == 0) red[a][sg] = s;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint a = 0; a < 4; ++a) {
    float m = red[a][0];
    for (uint i = 1; i < 8; ++i) m = fmax(m, red[a][i]);
    mx[a] = m;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint a = 0; a < 4; ++a) {
    const float e = live ? exp(score[a] - mx[a]) : 0.0f;
    p[a][t] = e;
    const float s = sub_group_reduce_add(e);
    if (lane == 0) red[a][sg] = s;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 4) {
    float sum = 0.0f;
    for (uint i = 0; i < 8; ++i) sum += red[t][i];
    out[t] = t == 0 ? mx[0] : t == 1 ? mx[1] : t == 2 ? mx[2] : mx[3];
    out[4 + t] = sum;
  }
  // values: work-item t owns dimension t; keys of the chunk in order
  float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint n = min((uint)DF_CHUNK, count - min(count, chunk * DF_CHUNK));
  for (uint u = 0; u < n; ++u) {
    const uint jj = chunk * DF_CHUNK + u;
    const float v = jj < retained ? bf2f(context_v[(ulong)((first + jj) % ring) * DF_KV + kv * DF_HEAD + t])
                                  : bf2f(block_v[(jj - retained) * DF_KV + kv * DF_HEAD + t]);
    for (uint a = 0; a < 4; ++a) o[a] += p[a][u] * v;
  }
  for (uint a = 0; a < 4; ++a) out[8 + a * DF_HEAD + t] = o[a];
}

// Second pass: one (row, query head) a work-group; the chunks' maxima, sums and values merge, BF16 out [row][4096].
// A chunk with no live key has max -inf and sum 0 and counts for nothing.
KERNEL(128) void DF_NAME(merge)(global const float *part, global ushort *out, uint committed, uint from, uint chunks) {
  const uint d = get_local_id(0);
  const uint row = get_group_id(0), h = get_group_id(1), kv = h / 4, a = h % 4;
  const uint first = df_first(from, committed, row);
  const uint retained = committed > first ? committed - first : 0u;
  const uint used = (retained + DF_BLOCK + DF_CHUNK - 1) / DF_CHUNK;
  global const float *base = part + (ulong)(row * DF_KVHEADS + kv) * chunks * 520;
  float mx = -INFINITY;
  for (uint c = 0; c < used; ++c) mx = fmax(mx, base[(ulong)c * 520 + a]);
  float den = 0.0f, value = 0.0f;
  for (uint c = 0; c < used; ++c) {
    const float w = DF_MERGE_WEIGHT(base[(ulong)c * 520 + a], mx);
    den += w * base[(ulong)c * 520 + 4 + a];
    value += w * base[(ulong)c * 520 + 8 + a * DF_HEAD + d];
  }
  out[row * DF_Q + h * DF_HEAD + d] = f2bf(value / den);
}

// The 16 largest logits of each row, in two passes. Pass one: each of 256 groups keeps the top 16 of its slice of the
// first `valid` ids (ties go to the lower id); the host merges the 256 lists of a row.
KERNEL(16) void DF_NAME(topk)(global const ushort *logits, uint stride, uint valid, global float *values, global int *ids) {
  const uint group = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
  const uint per = (valid + DF_TOPK_GROUPS - 1) / DF_TOPK_GROUPS;
  const uint a = group * per, b = min(valid, a + per);
  float bv[DF_TOPK];
  int bi[DF_TOPK];
  for (uint i = 0; i < DF_TOPK; ++i) {
    bv[i] = -INFINITY;
    bi[i] = 0x7fffffff;
  }
  if (lane == 0) {
    for (uint j = a; j < b; ++j) {
      const float v = DF_TOPK_VALUE(bf2f(logits[(ulong)row * stride + j]));
      if (!(v > bv[DF_TOPK - 1])) continue;
      uint k = DF_TOPK - 1;
      while (k > 0 && v > bv[k - 1]) {
        bv[k] = bv[k - 1];
        bi[k] = bi[k - 1];
        --k;
      }
      bv[k] = v;
      bi[k] = (int)j;
    }
    for (uint i = 0; i < DF_TOPK; ++i) {
      values[((ulong)row * DF_TOPK_GROUPS + group) * DF_TOPK + i] = bv[i];
      ids[((ulong)row * DF_TOPK_GROUPS + group) * DF_TOPK + i] = bi[i];
    }
  }
}

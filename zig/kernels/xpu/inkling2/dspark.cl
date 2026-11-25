// Inkling 2 Mini DSpark drafter arithmetic on Intel Xe2 (OpenCL C), the port of kernels/metal/inkling2/dspark.metal.
//
// The same formulas and rounding points as the Metal kernels: BF16 storage rounds to nearest even, every dot is one
// serial FP32 chain over its inputs (multiply, then add, never fused), the RMS sums are serial, and a row's scale
// is 1 / sqrt, so the kernels agree with the contract's explicit FP32 references. Build with
// -cl-fp32-correctly-rounded-divide-sqrt: the divide and the square root are then correctly rounded.
//
// Every kernel has a fixed work-group (reqd_work_group_size) at sub-group 16. Where a Metal kernel had one thread
// run a long serial chain while the rest of the group waited (the attention scores, the base projection), the work
// is spread over other work-items, each chain keeping its own order.

#pragma OPENCL FP_CONTRACT OFF

#define DSP_H 2304
#define DSP_Q 3072
#define DSP_KV 1024
#define DSP_HEAD 128
#define DSP_RANK 256
#define DSP_RNN 768
#define DSP_EPS 1e-6f
#define DSP_SCALE 0x1.6a09e6p-4f
#define DSP_ROWS_MAX 8

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// The contract plants one arithmetic mistake in each kernel through the macros below; production keeps the defaults.
// contract_inkling2_dspark_controls.cl redefines them (and DSP_NAME) before it includes this file.
#ifndef DSP_NAME
#define DSP_NAME(x) tf_ink2_dspark_##x
#endif
#ifndef DSP_INIT_SRC
#define DSP_INIT_SRC(row, anchor, mask) ((row) == 0 ? (anchor) : (mask))
#endif
#ifndef DSP_ALPHA
#define DSP_ALPHA(x) rbf(x)
#endif
#ifndef DSP_QKV_ROPE_HI
#define DSP_QKV_ROPE_HI(x, other, cs, sn) ((other) * (sn) + (x) * (cs))
#endif
#ifndef DSP_CTX_ROPE_HI
#define DSP_CTX_ROPE_HI(x, other, cs, sn) ((other) * (sn) + (x) * (cs))
#endif
#ifndef DSP_FIRST
#define DSP_FIRST(from, local, committed) max((from), (local) && (committed) > 1024u ? (committed) - 1024u : 0u)
#endif
#ifndef DSP_NORM_SRC
#define DSP_NORM_SRC(sigma) (sigma)
#endif
#ifndef DSP_SILU
#define DSP_SILU(gate) ((gate) / (1.0f + exp(-(gate))))
#endif
#ifndef DSP_KAPPA_Y_COL
#define DSP_KAPPA_Y_COL 512
#endif
#ifndef DSP_LEFT
#define DSP_LEFT(gamma, previous) rbf((gamma) * (previous))
#endif
#ifndef DSP_BASE_OUT
#define DSP_BASE_OUT(dot) (dot)
#endif
#ifndef DSP_BIAS
#define DSP_BIAS(beta, dot) rbf((beta) + (dot))
#endif
#ifndef DSP_DRAW_SCORE
#define DSP_DRAW_SCORE(v, u) ((v) - log(-log(u)))
#endif

// sum += a[d] * b[d] in order, for d in [0, n), n a multiple of 8; both BF16.
static inline float dot8(float sum, global const ushort *a, global const ushort *b, uint n) {
  for (uint d = 0; d < n; d += 8) {
    const ushort8 x = vload8(0, a + d), y = vload8(0, b + d);
    sum += bf2f(x.s0) * bf2f(y.s0);
    sum += bf2f(x.s1) * bf2f(y.s1);
    sum += bf2f(x.s2) * bf2f(y.s2);
    sum += bf2f(x.s3) * bf2f(y.s3);
    sum += bf2f(x.s4) * bf2f(y.s4);
    sum += bf2f(x.s5) * bf2f(y.s5);
    sum += bf2f(x.s6) * bf2f(y.s6);
    sum += bf2f(x.s7) * bf2f(y.s7);
  }
  return sum;
}

// Row zero copies the target's normalized anchor; the other seven rows copy the raw mask embedding.
KERNEL(256) void DSP_NAME(init_rows)(global const ushort *anchor, global const ushort *mask, global ushort *rows) {
  const uint row = get_group_id(0), t = get_local_id(0);
  for (uint d = t; d < DSP_H; d += 256) rows[row * DSP_H + d] = DSP_INIT_SRC(row, anchor, mask)[d];
}

// taps [row][tap][H] become c [fusion][row][H]; the newest row of the last fusion is also written to newest_f.
KERNEL(256) void DSP_NAME(fuse_taps)(global const ushort *taps, global const ushort *fuse_weight, global const ushort *norm_weight,
                                     global ushort *c, global ushort *newest_f, uint count) {
  const uint group = get_group_id(0), t = get_local_id(0);
  const uint fusion = group / count, row = group % count;
  local float alpha[4];
  local float scale;
  if (t == 0) {
    float scores[4];
    float peak = -INFINITY;
    for (uint j = 0; j < 4; ++j) {
      scores[j] = bf2f(fuse_weight[fusion * 4 + j]);
      peak = fmax(peak, scores[j]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
      scores[j] = exp(scores[j] - peak);
      sum += scores[j];
    }
    for (uint j = 0; j < 4; ++j) alpha[j] = DSP_ALPHA(scores[j] / sum);
    float square = 0.0f;
    for (uint d = 0; d < DSP_H; ++d) {
      float mixed = 0.0f;
      for (uint j = 0; j < 4; ++j) mixed += alpha[j] * bf2f(taps[(row * 4 + j) * DSP_H + d]);
      square += mixed * mixed;
    }
    scale = 1.0f / sqrt(square / (float)DSP_H + DSP_EPS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint d = t; d < DSP_H; d += 256) {
    float mixed = 0.0f;
    for (uint j = 0; j < 4; ++j) mixed += alpha[j] * bf2f(taps[(row * 4 + j) * DSP_H + d]);
    const ushort out = f2bf(mixed * scale * bf2f(norm_weight[fusion * DSP_H + d]));
    c[(fusion * count + row) * DSP_H + d] = out;
    if (fusion == 3 && row + 1 == count) newest_f[d] = out;
  }
}

static inline float dspark_rope_angle(uint d, uint position) {
  const uint pair = d % 64;
  const float frequency = pow(1e6f, -(float)pair / 64.0f);
  return (float)position * frequency;
}

// Q and K of the block: per-head RMS norm with a learned weight, then rotate-half RoPE; V is copied.
KERNEL(128) void DSP_NAME(block_qkv)(global const ushort *q_raw, global const ushort *k_raw, global const ushort *v_raw,
                                     global const ushort *q_weight, global const ushort *k_weight, global ushort *q,
                                     global ushort *k, global ushort *v, uint base) {
  const uint group = get_group_id(0), d = get_local_id(0);
  const uint row = group / 40, work = group % 40;
  local float scale;
  if (work < 32 && d == 0) {
    float sum = 0.0f;
    if (work < 24) {
      const uint h = work;
      for (uint i = 0; i < DSP_HEAD; ++i) {
        const float x = bf2f(q_raw[row * DSP_Q + h * DSP_HEAD + i]);
        sum += x * x;
      }
    } else {
      const uint h = work - 24;
      for (uint i = 0; i < DSP_HEAD; ++i) {
        const float x = bf2f(k_raw[row * DSP_KV + h * DSP_HEAD + i]);
        sum += x * x;
      }
    }
    scale = 1.0f / sqrt(sum / (float)DSP_HEAD + DSP_EPS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (work >= 32) {
    const uint h = work - 32;
    v[row * DSP_KV + h * DSP_HEAD + d] = v_raw[row * DSP_KV + h * DSP_HEAD + d];
    return;
  }
  const bool query = work < 24;
  const uint h = query ? work : work - 24;
  const uint at = row * (query ? DSP_Q : DSP_KV) + h * DSP_HEAD + d;
  global const ushort *raw = query ? q_raw : k_raw;
  global const ushort *weight = query ? q_weight : k_weight;
  const float x = rbf(bf2f(raw[at]) * scale * bf2f(weight[d]));
  const uint partner = d < 64 ? d + 64 : d - 64;
  const uint other_at = row * (query ? DSP_Q : DSP_KV) + h * DSP_HEAD + partner;
  const float other = rbf(bf2f(raw[other_at]) * scale * bf2f(weight[partner]));
  const float angle = dspark_rope_angle(d, base + row);
  const float cs = cos(angle), sn = sin(angle);
  const float rotated = d < 64 ? x * cs - other * sn : DSP_QKV_ROPE_HI(x, other, cs, sn);
  if (query) q[at] = f2bf(rotated); else k[at] = f2bf(rotated);
}

// Committed rows' context K and V land at absolute-position modulo ring.
KERNEL(128) void DSP_NAME(context_kv)(global const ushort *k_raw, global const ushort *v_raw, global const ushort *weight,
                                      global ushort *keys, global ushort *values, uint base, uint rows, uint ring) {
  (void)rows;
  const uint group = get_group_id(0), d = get_local_id(0);
  const uint row = group / 8, h = group % 8;
  local float scale;
  if (d == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < DSP_HEAD; ++i) {
      const float x = bf2f(k_raw[row * DSP_KV + h * DSP_HEAD + i]);
      sum += x * x;
    }
    scale = 1.0f / sqrt(sum / (float)DSP_HEAD + DSP_EPS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint position = base + row;
  const uint at = (position % ring) * DSP_KV + h * DSP_HEAD + d;
  const uint src = row * DSP_KV + h * DSP_HEAD + d;
  const uint partner = d < 64 ? d + 64 : d - 64;
  const float x = rbf(bf2f(k_raw[src]) * scale * bf2f(weight[d]));
  const float other = rbf(bf2f(k_raw[row * DSP_KV + h * DSP_HEAD + partner]) * scale * bf2f(weight[partner]));
  const float angle = dspark_rope_angle(d, position);
  const float cs = cos(angle), sn = sin(angle);
  keys[at] = f2bf(d < 64 ? x * cs - other * sn : DSP_CTX_ROPE_HI(x, other, cs, sn));
  values[at] = v_raw[src];
}

// One (row, head) a work-group. The scores are one serial FP32 dot each, spread over the work-items; the softmax sum
// keeps the serial order of the Metal kernel (one work-item adds the exponentials in key order), and each output
// element adds its weighted values in key order, the retained context first, then the block's eight rows.
// `scores` holds the scores, then the exponentials, then the weights, in place: 8 * 24 * (retained + 8) floats.
KERNEL(128) void DSP_NAME(attention)(global const ushort *q, global const ushort *block_k, global const ushort *block_v,
                                     global const ushort *context_k, global const ushort *context_v, global float *scores,
                                     global ushort *out, uint committed, uint ring, uint local_layer, uint from) {
  const uint group = get_group_id(0), d = get_local_id(0);
  const uint row = group / 24, h = group % 24, kv = h / 3;
  const uint q_at = row * DSP_Q + h * DSP_HEAD;
  const uint first = DSP_FIRST(from, local_layer, committed);
  const uint retained = committed - first;
  const uint count = retained + 8;
  global float *s = scores + (ulong)group * count;
  local float ql[DSP_HEAD];
  local float red[128];
  local float peak, denominator;
  ql[d] = bf2f(q[q_at + d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float m = -INFINITY;
  for (uint j = d; j < count; j += 128) {
    global const ushort *key;
    if (j < retained) key = context_k + (ulong)((first + j) % ring) * DSP_KV + kv * DSP_HEAD;
    else key = block_k + (j - retained) * DSP_KV + kv * DSP_HEAD;
    float dot = 0.0f;
    for (uint i = 0; i < DSP_HEAD; i += 8) {
      const ushort8 x = vload8(0, key + i);
      dot += ql[i + 0] * bf2f(x.s0);
      dot += ql[i + 1] * bf2f(x.s1);
      dot += ql[i + 2] * bf2f(x.s2);
      dot += ql[i + 3] * bf2f(x.s3);
      dot += ql[i + 4] * bf2f(x.s4);
      dot += ql[i + 5] * bf2f(x.s5);
      dot += ql[i + 6] * bf2f(x.s6);
      dot += ql[i + 7] * bf2f(x.s7);
    }
    const float score = dot * DSP_SCALE;
    s[j] = score;
    m = fmax(m, score);
  }
  red[d] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint o = 64; o > 0; o >>= 1) {
    if (d < o) red[d] = fmax(red[d], red[d + o]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mx = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint j = d; j < count; j += 128) s[j] = exp(s[j] - mx);
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  if (d == 0) {
    // Eight loads in flight, added in key order.
    float sum = 0.0f;
    uint j = 0;
    for (; j + 8 <= count; j += 8) {
      const float8 t = vload8(0, s + j);
      sum += t.s0;
      sum += t.s1;
      sum += t.s2;
      sum += t.s3;
      sum += t.s4;
      sum += t.s5;
      sum += t.s6;
      sum += t.s7;
    }
    for (; j < count; ++j) sum += s[j];
    peak = mx;
    denominator = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float den = denominator;
  for (uint j = d; j < count; j += 128) s[j] = s[j] / den;
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  float value = 0.0f;
  uint slot = first % ring;
  uint j = 0;
  // Eight weights and values loaded together, then added in key order.
  for (; j + 8 <= retained; j += 8) {
    const float8 w = vload8(0, s + j);
    float v[8];
    #pragma unroll
    for (uint n = 0; n < 8; ++n) {
      v[n] = bf2f(context_v[(ulong)slot * DSP_KV + kv * DSP_HEAD + d]);
      slot = slot + 1 == ring ? 0 : slot + 1;
    }
    value += w.s0 * v[0];
    value += w.s1 * v[1];
    value += w.s2 * v[2];
    value += w.s3 * v[3];
    value += w.s4 * v[4];
    value += w.s5 * v[5];
    value += w.s6 * v[6];
    value += w.s7 * v[7];
  }
  for (; j < retained; ++j) {
    value += s[j] * bf2f(context_v[(ulong)slot * DSP_KV + kv * DSP_HEAD + d]);
    slot = slot + 1 == ring ? 0 : slot + 1;
  }
  for (uint j = 0; j < 8; ++j) value += s[retained + j] * bf2f(block_v[j * DSP_KV + kv * DSP_HEAD + d]);
  out[q_at + d] = f2bf(value);
  (void)peak;
}

// new_residual = x + residual in FP32 (x alone on the first call), stored as BF16; normalized uses the FP32 sum.
KERNEL(256) void DSP_NAME(residual_norm)(global const ushort *x, global const ushort *residual, global const ushort *weight,
                                         global ushort *new_residual, global ushort *normalized, uint init) {
  const uint row = get_group_id(0), t = get_local_id(0);
  local float scale;
  if (t == 0) {
    float sum = 0.0f;
    for (uint d = 0; d < DSP_H; ++d) {
      const uint at = row * DSP_H + d;
      const float v = init ? bf2f(x[at]) : bf2f(x[at]) + bf2f(residual[at]);
      sum += v * v;
    }
    scale = 1.0f / sqrt(sum / (float)DSP_H + DSP_EPS);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint n = 0; n < 9; ++n) {
    const uint d = n * 256 + t;
    const uint at = row * DSP_H + d;
    const float sigma = init ? bf2f(x[at]) : bf2f(x[at]) + bf2f(residual[at]);
    new_residual[at] = f2bf(sigma);
    normalized[at] = f2bf(DSP_NORM_SRC(sigma) * scale * bf2f(weight[d]));
  }
}

KERNEL(256) void DSP_NAME(swiglu)(global const ushort *g, global const ushort *u, global ushort *out, uint count) {
  const uint i = get_global_id(0);
  if (i >= count) return;
  const float gate = bf2f(g[i]);
  out[i] = f2bf(DSP_SILU(gate) * bf2f(u[i]));
}

// kappa [row][768] (FP32) over the RNN projection's columns 512..2815 (y) and 2816..5119 (newest_f), plus the bias.
KERNEL(256) void DSP_NAME(kappa)(global const ushort *y, global const ushort *f, global const ushort *weight,
                                 global const ushort *bias, global float *out, uint count) {
  const uint i = get_global_id(0);
  const uint row = i / DSP_RNN, channel = i % DSP_RNN;
  if (row >= count) return;
  float sum = 0.0f;
  sum = dot8(sum, y + (ulong)row * DSP_H, weight + (ulong)channel * 5120 + DSP_KAPPA_Y_COL, DSP_H);
  sum = dot8(sum, f, weight + (ulong)channel * 5120 + 2816, DSP_H);
  sum += bf2f(bias[channel]);
  out[i] = sum;
}

// One Markov recurrent step: z (3 x 256, rounded), the next state and lambda, all BF16.
KERNEL(256) void DSP_NAME(recurrent)(global const ushort *state, global const float *kappa, global const ushort *rnn_weight,
                                     global const ushort *markov, global ushort *z, global ushort *next_state,
                                     global ushort *lambda, global const uint *token_in) {
  const uint i = get_global_id(0);
  if (i >= DSP_RANK) return;
  const uint token = token_in[0];
  float values[3];
  for (uint part = 0; part < 3; ++part) {
    const uint channel = part * DSP_RANK + i;
    float sum = 0.0f;
    sum = dot8(sum, state, rnn_weight + (ulong)channel * 5120, DSP_RANK);
    sum = dot8(sum, markov + (ulong)token * DSP_RANK, rnn_weight + (ulong)channel * 5120 + 256, DSP_RANK);
    sum += kappa[channel];
    const ushort rounded = f2bf(sum);
    z[channel] = rounded;
    values[part] = bf2f(rounded);
  }
  const float gamma = rbf(1.0f / (1.0f + exp(-values[0])));
  const float candidate = rbf(tanh(values[1]));
  const float keep = rbf(1.0f - gamma);
  const float left = DSP_LEFT(gamma, bf2f(state[i]));
  const float right = rbf(keep * candidate);
  next_state[i] = f2bf(left + right);
  lambda[i] = f2bf(tanh(values[2]));
}

// out [row][vocab] = BF16(U[id] . y[row]) over the 2304 inputs, for up to eight rows of y. A sub-group takes 16 vocabulary
// ids as the columns of a DPAS (bf16 products, FP32 sums: lane n holds id n's 16 weights of a k step as its B operand)
// with the rows of y as its eight A rows, so each U row is read from memory once for all rows at the memory's pace. A
// row's sum is the DPAS's: 16 products added in a fixed order a step, the steps added in order (not the serial chain
// of the Metal kernel; the result is within an FP32 rounding of it, so BF16 can differ by a unit in the last place).
KERNEL(256) void DSP_NAME(base)(global const ushort *unembed, global const ushort *y, global ushort *out, uint rows, uint vocab) {
  const uint lane = get_sub_group_local_id();
  const uint raw = get_group_id(0) * 256 + get_sub_group_id() * 16 + lane;
  const uint id = raw < vocab ? raw : vocab - 1; // a lane past the last id reads the last row and writes nothing
  global const uint *u = (global const uint *)(unembed + (ulong)id * DSP_H);
  float8 acc = 0.0f;
  for (uint d = 0; d < DSP_H; d += 32) {
    const uint16 w = vload16(0, u + d / 2); // 32 weights: two k steps of 16
#pragma unroll
    for (uint half_step = 0; half_step < 2; ++half_step) {
      short8 a = (short8)(0);
#pragma unroll
      for (uint r = 0; r < DSP_ROWS_MAX; ++r)
        if (r < rows) a[r] = (short)intel_sub_group_block_read_us(y + (ulong)r * DSP_H + d + 16 * half_step);
      const int8 b = half_step == 0 ? as_int8(w.s01234567) : as_int8(w.s89abcdef);
      acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
    }
  }
  if (raw < vocab) {
#pragma unroll
    for (uint r = 0; r < DSP_ROWS_MAX; ++r)
      if (r < rows) out[(ulong)r * vocab + raw] = f2bf(DSP_BASE_OUT(acc[r]));
  }
}

// Folded head: bias = BF16(beta + Phi . lambda), logit = BF16(base + bias), out = BF16(30 tanh(logit / 30)).
KERNEL(256) void DSP_NAME(logits)(global const ushort *base, global const ushort *phi, global const ushort *beta,
                                  global const ushort *lambda, global ushort *out, uint count) {
  const uint i = get_global_id(0);
  if (i >= count) return;
  const float dot = dot8(0.0f, phi + (ulong)i * DSP_RANK, lambda, DSP_RANK);
  const float bias = DSP_BIAS(bf2f(beta[i]), dot);
  const float logit = rbf(bf2f(base[i]) + bias);
  out[i] = f2bf(30.0f * tanh(logit / 30.0f));
}

// ---- the draft draw, on the device ---------------------------------------------------------------------------------
// One work-group takes the row of BF16 logits a draft step made and writes the token the host sampler (gpu_rule.sample)
// would draw: the best n candidates by a radix select over the order key of logit * (1 / T), a bitonic sort of them,
// then the nucleus, min_p and the keyed Gumbel race of gpu_rule.pick. The selection is integer comparison, so the
// candidate list is the host sort's; the exp and log are the device's, so a draw can differ from the host's where two
// scores tie within a few ulps. A draft only has to be a good guess: the target's own draw decides every token.
// cfg: 0 1/T bits, 1 top_p bits, 2 ln(min_p) bits, 3 top_k (0: off), 4 seed low, 5 seed high, 6 unused, 7 greedy (1) or not.
// `position` is the draw's position in the keyed race. token_out[0] is written; the next recurrent step reads it there.
#define DSP_TOP 1024

static inline uint dsp_key(float v) {
  const uint b = as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static inline ulong dsp_mix(ulong x) {
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9UL;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBUL;
  return x ^ (x >> 31);
}

KERNEL(256) void DSP_NAME(draw)(global const ushort *logits, global const uint *cfg, global uint *token_out, uint vocab, uint position32) {
  local uint hist[256];
  local ulong order[DSP_TOP];
  local float cv[DSP_TOP];
  local uint cid[DSP_TOP];
  local float partial[256];
  local uint partial_at[256];
  local uint cut_s, need_s, base_g, base_e, tot_g, tot_e, keep_s;
  local float norm_s;
  const uint t = get_local_id(0);
  const float inv_t = as_float(cfg[0]);
  const float top_p = as_float(cfg[1]);
  const float min_log = as_float(cfg[2]);
  const uint top_k = cfg[3];
  const bool greedy = cfg[7] != 0;
  const uint n = greedy ? 1u : (top_k == 0 || top_k > DSP_TOP ? (uint)DSP_TOP : top_k);
  uint np2 = 2;
  while (np2 < n) np2 <<= 1;

  // radix select: the n-th largest key
  uint cut = 0;
  uint need = n;
  for (int p = 0; p < 4; p++) {
    const int shift = 24 - 8 * p;
    hist[t] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = t; i < vocab; i += 256) {
      const uint k = dsp_key(bf2f(logits[i]) * inv_t);
      if (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8))) atomic_inc(&hist[(k >> shift) & 0xFFu]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      uint s = 0;
      uint d = 255;
      while (d > 0 && s + hist[d] < need) {
        s += hist[d];
        d--;
      }
      cut_s = cut | (d << shift);
      need_s = need - s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    cut = cut_s;
    need = need_s;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // the entries above the cut, and the first `need` equal to it in id order
  const uint above = n - need;
  for (uint j = t; j < np2; j += 256) order[j] = 0;
  if (t == 0) {
    base_g = 0;
    base_e = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint c = 0; c < vocab; c += 256 * 4) {
    uint keys[4];
    uint tg = 0;
    uint te = 0;
    for (uint j = 0; j < 4; j++) {
      const uint i = c + t * 4 + j;
      keys[j] = i < vocab ? dsp_key(bf2f(logits[i]) * inv_t) : 0u;
      tg += (i < vocab && keys[j] > cut) ? 1u : 0u;
      te += (i < vocab && keys[j] == cut) ? 1u : 0u;
    }
    const uint sg = work_group_scan_exclusive_add(tg);
    const uint se = work_group_scan_exclusive_add(te);
    if (t == 255) {
      tot_g = sg + tg;
      tot_e = se + te;
    }
    uint pg = base_g + sg;
    uint pe = base_e + se;
    for (uint j = 0; j < 4; j++) {
      const uint i = c + t * 4 + j;
      if (i >= vocab) break;
      const ulong w = ((ulong)keys[j] << 32) | (ulong)(0xFFFFFFFFu - i);
      if (keys[j] > cut) {
        order[pg++] = w;
      } else if (keys[j] == cut) {
        if (pe < need) order[above + pe] = w;
        pe++;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      base_g += tot_g;
      base_e += tot_e;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (uint size = 2; size <= np2; size <<= 1) {
    for (uint gap = size >> 1; gap > 0; gap >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint q = t; q < np2 / 2; q += 256) {
        const uint a = 2 * q - (q & (gap - 1));
        const uint b = a + gap;
        const bool descending = (q & (size >> 1)) == 0;
        const ulong x = order[a];
        const ulong y = order[b];
        if (descending ? (x < y) : (x > y)) {
          order[a] = y;
          order[b] = x;
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint j = t; j < n; j += 256) {
    const ulong w = order[j];
    const uint k = (uint)(w >> 32);
    cv[j] = as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
    cid[j] = 0xFFFFFFFFu - (uint)w;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (greedy) {
    if (t == 0) token_out[0] = cid[0];
    return;
  }

  // normalizer: the candidates' sum with top_k on, the whole row's with it off
  const float m = cv[0];
  float mine = 0.0f;
  if (top_k == 0 || top_k > DSP_TOP) {
    for (uint i = t; i < vocab; i += 256) mine += exp(bf2f(logits[i]) * inv_t - m);
    partial[t] = mine;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint hv = 128; hv > 0; hv >>= 1) {
      if (t < hv) partial[t] += partial[t + hv];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) norm_s = partial[0];
  } else {
    if (t == 0) {
      float sum = 0.0f;
      for (uint j = 0; j < n; j++) sum += exp(cv[j] - m);
      norm_s = sum;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    const float norm = norm_s;
    uint keep = n;
    if (top_p > 0.0f && top_p < 1.0f) {
      float cum = 0.0f;
      for (uint j = 0; j < n; j++) {
        cum += exp(cv[j] - m) / norm;
        if (cum >= top_p) {
          keep = j + 1;
          break;
        }
      }
    }
    if (min_log > -INFINITY) {
      const float floor_v = m + min_log;
      uint j = 0;
      while (j < keep && cv[j] >= floor_v) j++;
      keep = j;
    }
    keep_s = keep;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint keep = keep_s;
  const ulong seed = (ulong)cfg[4] | ((ulong)cfg[5] << 32);
  const ulong position = (ulong)position32;
  const ulong base = dsp_mix(dsp_mix(seed + 0x9E3779B97F4A7C15UL) ^ (position * 0xD1B54A32D192ED03UL));
  float best = -INFINITY;
  uint best_j = 0xFFFFFFFFu;
  for (uint j = t; j < keep; j += 256) {
    const ulong x = dsp_mix(base ^ (ulong)cid[j]);
    const float u = ((float)(uint)(x >> 40) + 0.5f) * (1.0f / 16777216.0f);
    const float score = DSP_DRAW_SCORE(cv[j], u);
    if (best_j == 0xFFFFFFFFu || score > best) {
      best = score;
      best_j = j;
    }
  }
  partial[t] = best;
  partial_at[t] = best_j;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t == 0) {
    float top = -INFINITY;
    uint at = 0;
    bool any = false;
    for (uint q = 0; q < 256; q++) {
      const uint j = partial_at[q];
      if (j == 0xFFFFFFFFu) continue;
      if (!any || partial[q] > top || (partial[q] == top && j < at)) {
        top = partial[q];
        at = j;
        any = true;
      }
    }
    token_out[0] = cid[at];
  }
}

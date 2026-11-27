// Inkling 2 Mini prompt-chunk glue on Intel Xe2 (OpenCL C): the row-batched forms of the per-row launches of glue.cl and
// loop.cl. Each kernel does exactly the arithmetic of the kernel it batches, in the same order, so a row's result does
// not depend on how many other rows share the launch.
//
// Layout rule for the contract's control generator: helpers come first, then one kernel at a time, each starting on
// the line that holds its `void tf_ink2_pf_<name>(` and ending at the first `}` in column 0.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

#define PF_HIDDEN 2304
#define PF_EPS 1e-6f
#define PF_AR_SCALE 0x1.555556p-6f // binary32(1 / 48)
#define PF_LOOP_SGS 16
#define PF_MESSAGES 48

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even

static inline float pf_value(global const ushort *blocks, global const float *partial, uint source, uint sources, uint d) {
  return source + 1 == sources ? partial[d] : bf2f(blocks[(ulong)source * PF_HIDDEN + d]);
}

static float pf_group_sum(float v, local float *red) {
  const float s = sub_group_reduce_add(v);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int j = 0; j < PF_LOOP_SGS; ++j) t += red[j];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static inline float pf_rms_inverse(float ss) { return 1.0f / sqrt(ss / 2304.0f + PF_EPS); }





// ---- dense projections of a chunk on DPAS --------------------------------------------------------------------------
//
// y[row][n] = bf16(x[row][k] . dequant(W)[n][k]) for the Q8 weights of q8.zig, any number of rows, FP32 sums. The weights are
// the A operand (8 weight rows a tile, lane = k) and the rows of x the B operand (lane = row), so one fragment of codes feeds
// every row tile of the sub-group. `pf_perm` first reorders each 256-value plane of x so that a k16 step (block b,
// parity p) is 16 contiguous values: step s = 2 b + p holds the block's values 2 l + p for l = 0 to 15, which are the values
// lane l of the weights' plane holds (q8.zig). A block's 32 products go through DPAS into a fresh sum, which the block's fp16
// scale then weighs into the running sum: acc += scale * (sum of 32 code * x). A row's result does not depend on the other
// rows of the launch.

#define PF_WA 2 // weight tiles of a sub-group (8 rows each)
#define PF_TB 2 // row tiles of a sub-group (16 rows each)


static inline ushort pf_code_bf16(uint word, uint shift) {
  const int c = (int)(word << (24 - shift)) >> 24; // the signed byte at `shift`
  return (ushort)(as_uint((float)c) >> 16);        // an integer of 7 bits is exact in bf16
}


// ---- the message controllers of a chunk on DPAS --------------------------------------------------------------------
//
// The controller tuple of tf_inkling2_loop_zeta and _gates for every row at once. `zeta_norm` writes each row's h and m
// normed to bf16 (h first, then m: 2 * 2304 values); `zeta` is the D projection, zeta[row][r] = bf16(normed . down[r]), on
// DPAS (the rows of x the A operand); `gates` makes F and Gamma from zeta and the weights, also on DPAS, and ends in the
// blend out = (1 - tanh F) h + tanh Gamma m with every step rounded in binary32, as the per-row kernel does.




// ---- sliding-window prompt attention on DPAS -------------------------------------------------------------------------
//
// The kernel is prompt_attention.cl's, cut for a window: a sub-group owns 16 (row, query head) pairs of one K/V head and
// walks the keys of its rows' windows in tiles of 16, S = Q.K and O = P.V on DPAS, every P rounded to bf16 against the
// running maximum before it enters the denominator and the product. Keys come from a staging ring (position % ring) that
// holds the window before the chunk and the chunk itself, so a chunk longer than the cache ring never overwrites a key a
// row still needs. One partial (max, sum, P.V) a (row, head) is left for the merge of attention.cl, which is launched with a
// key chunk too large to split a window.

#define PF_D 128
#define PF_GROUP 3
#define PF_KEYS 16
#define PF_PAIRS 16
#define PF_WINDOW 1024
#define PF_LOG2E 1.4426950408889634f

typedef struct {
  int row0, n, p0, pad;
  long off;
} PfSeg;




static inline float pf_sigmoid(float x) {
  const float e = exp(-fabs(x));
  return x < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
}


#define PF_SQRT5 0x1.1e377ap+1f // binary32(sqrt(5))


#define PF_REL 16
#define PF_REL_BATCH 8 // (row, head) pairs a work-group
#define PF_EXTENT 1024



// Grid (queries, rows): the attention residual of tf_inkling2_ar for every row at once. Row r reads its sources at
// blocks[r][0..sources - 1] (a row holds 4 block slots) and partial[r], and writes out[q][r], one fp32 row a (query, row).
KERNEL(256) void tf_ink2_pf_ar(global const ushort *blocks, global const float *partial, global const float *queries,
                               global float *out, uint sources, uint count, uint rows) {
  const uint q = get_group_id(0), row = get_group_id(1), t = get_local_id(0), sg = get_sub_group_id();
  if (q >= count || row >= rows || sources == 0 || sources > 5) return;
  global const ushort *my_blocks = blocks + (ulong)row * 4 * PF_HIDDEN;
  global const float *my_partial = partial + (ulong)row * PF_HIDDEN;
  local float red[160];
  float v[5][PF_HIDDEN / 256];
  float query[PF_HIDDEN / 256];
#pragma unroll
  for (uint s = 0; s < 5; ++s)
#pragma unroll
    for (uint i = 0; i < PF_HIDDEN / 256; ++i) v[s][i] = s < sources ? pf_value(my_blocks, my_partial, s, sources, t + 256 * i) : 0.0f;
#pragma unroll
  for (uint i = 0; i < PF_HIDDEN / 256; ++i) query[i] = queries[(ulong)q * PF_HIDDEN + t + 256 * i];
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float ss = 0.0f;
#pragma unroll
    for (uint i = 0; i < PF_HIDDEN / 256; ++i) ss += v[s][i] * v[s][i];
    const float reduced = sub_group_reduce_add(ss);
    if (s < sources && get_sub_group_local_id() == 0) red[s * 16 + sg] = reduced;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float means[5];
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 16; ++j) ss += red[s * 16 + j];
    means[s] = rsqrt(ss / 2304.0f + PF_EPS);
  }
  local float *dots = red + 80;
#pragma unroll
  for (uint s = 0; s < 5; ++s) {
    float dot = 0.0f;
#pragma unroll
    for (uint i = 0; i < PF_HIDDEN / 256; ++i) dot += query[i] * (v[s][i] * means[s]);
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
    weights[s] = dot * PF_AR_SCALE;
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
  global float *dst = out + ((ulong)q * rows + row) * PF_HIDDEN;
#pragma unroll
  for (uint i = 0; i < PF_HIDDEN / 256; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (uint s = 0; s < 5; ++s)
      if (s < sources) acc += weights[s] * (v[s][i] * means[s]);
    dst[t + 256 * i] = acc;
  }
}

// Grid (rows * 48): tf_inkling2_loop_messages with the output laid out [entry][row][H], so that the gates of one entry
// read a contiguous [row][H] message block.
KERNEL(256) void tf_ink2_pf_messages(global const ushort *bank0, global const ushort *bank1, global const float *q_mlp,
                                     global const float *kv, global float *out, uint rows) {
  local float red[PF_LOOP_SGS];
  const uint t = get_local_id(0), work = get_group_id(0);
  const uint row = work / PF_MESSAGES;
  const uint entry = work - row * PF_MESSAGES;
  if (row >= rows) return;
  const uint layer = entry / 4;
  const uint role = entry - 4 * layer;
  const global float *query = ((role == 0 || role == 3) ? q_mlp : kv) + (size_t)(2 * layer + (role >= 2 ? 1 : 0)) * PF_HIDDEN;
  global const ushort *b0 = bank0 + (size_t)row * PF_HIDDEN;
  global const ushort *b1 = bank1 + (size_t)row * PF_HIDDEN;
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint d = t; d < PF_HIDDEN; d += 256) {
    const float x0 = bf2f(b0[d]), x1 = bf2f(b1[d]);
    ss0 += x0 * x0;
    ss1 += x1 * x1;
  }
  const float inv0 = pf_rms_inverse(pf_group_sum(ss0, red));
  const float inv1 = pf_rms_inverse(pf_group_sum(ss1, red));
  float dot0 = 0.0f, dot1 = 0.0f;
  for (uint d = t; d < PF_HIDDEN; d += 256) {
    const float beta0 = bf2f(b0[d]) * inv0;
    const float beta1 = bf2f(b1[d]) * inv1;
    dot0 += query[d] * beta0;
    dot1 += query[d] * beta1;
  }
  const float l0 = pf_group_sum(dot0, red) * PF_AR_SCALE;
  const float l1 = pf_group_sum(dot1, red) * PF_AR_SCALE;
  const float peak = fmax(fmax(-INFINITY, l0), l1);
  float w0 = exp(l0 - peak);
  float w1 = exp(l1 - peak);
  float den = 0.0f;
  den += w0;
  den += w1;
  w0 = w0 / den;
  w1 = w1 / den;
  global float *dst = out + ((size_t)entry * rows + row) * PF_HIDDEN;
  for (uint d = t; d < PF_HIDDEN; d += 256) {
    const float beta0 = bf2f(b0[d]) * inv0;
    const float beta1 = bf2f(b1[d]) * inv1;
    dst[d] = w0 * beta0 + w1 * beta1;
  }
}

// Grid (ceil(n / 256), rows): y[r][0..n-1] = x[r * ldx + ...], the block of one row (bf16 values).
KERNEL(256) void tf_ink2_pf_copy_rows(global const ushort *x, global ushort *y, uint rows, uint n, uint ldx, uint ldy) {
  const uint i = get_global_id(0), r = get_group_id(1);
  if (i < n && r < rows) y[(ulong)r * ldy + i] = x[(ulong)r * ldx + i];
}

// Grid (k / 256, rows): xp[row][plane][s][l] = x[row][plane * 256 + (s / 2) * 32 + 2 l + (s & 1)].
KERNEL(256) void tf_ink2_pf_perm(global const ushort *x, global ushort *xp, uint rows, uint k) {
  const uint plane = get_group_id(0), row = get_group_id(1), o = get_local_id(0);
  if (row >= rows) return;
  const uint s = o >> 4, l = o & 15u;
  xp[(ulong)row * k + plane * 256 + o] = x[(ulong)row * k + plane * 256 + (s >> 1) * 32 + 2 * l + (s & 1)];
}

// Grid (ceil(n / (8 * PF_WA * 8)), ceil(rows / (16 * PF_TB))), work-group of 8 sub-groups. ldy: values between output rows.
KERNEL(128) void tf_ink2_pf_q8(global const ushort *xp, global const uchar *w, global ushort *y, uint rows, uint n, uint k, uint ldy) {
  const uint lane = get_sub_group_local_id();
  const uint wrow0 = (get_group_id(0) * 8 + get_sub_group_id()) * (8 * PF_WA);
  if (wrow0 >= n) return;
  const uint row0 = get_group_id(1) * (16 * PF_TB);
  const uint planes = k / 256;
  global const half *scales = (global const half *)(w + (ulong)n * k);
  const uint nb = k / 32;
  float8 acc[PF_WA][PF_TB];
  _Pragma("unroll") for (uint a = 0; a < PF_WA; a++)
    _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) acc[a][t] = (float8)(0.0f);
  global const uint *xr[PF_TB];
  _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) xr[t] = (global const uint *)(xp + (ulong)min(row0 + 16 * t + lane, rows - 1) * k);
  for (uint plane = 0; plane < planes; plane++) {
    uint4 cv[PF_WA][8];
    _Pragma("unroll") for (uint a = 0; a < PF_WA; a++)
      _Pragma("unroll") for (uint r = 0; r < 8; r++) {
        const uint wr = min(wrow0 + 8 * a + r, n - 1);
        cv[a][r] = intel_sub_group_block_read4((global const uint *)(w + (ulong)wr * k + plane * 256)); // q8.zig word layout
      }
    _Pragma("unroll") for (uint b = 0; b < 8; b++) {
      float8 sc[PF_WA];
      _Pragma("unroll") for (uint a = 0; a < PF_WA; a++)
        _Pragma("unroll") for (uint r = 0; r < 8; r++) {
          const uint wr = min(wrow0 + 8 * a + r, n - 1);
          sc[a][r] = (float)scales[(ulong)wr * nb + plane * 8 + b];
        }
      float8 blk[PF_WA][PF_TB];
      _Pragma("unroll") for (uint a = 0; a < PF_WA; a++)
        _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) blk[a][t] = (float8)(0.0f);
      _Pragma("unroll") for (uint p = 0; p < 2; p++) {
        const uint s = 2 * b + p;
        int8 bf[PF_TB];
        _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) {
          const uint8 v = vload8(0, xr[t] + plane * 128 + s * 8);
          bf[t] = as_int8(v);
        }
        const uint shift = 8 * (2 * (b & 1) + p);
        _Pragma("unroll") for (uint a = 0; a < PF_WA; a++) {
          short8 af;
          _Pragma("unroll") for (uint r = 0; r < 8; r++) af[r] = (short)pf_code_bf16(cv[a][r][b >> 1], shift);
          _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) blk[a][t] = intel_sub_group_bf16_bf16_matrix_mad_k16(af, bf[t], blk[a][t]);
        }
      }
      _Pragma("unroll") for (uint a = 0; a < PF_WA; a++)
        _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) acc[a][t] = fma(blk[a][t], sc[a], acc[a][t]);
    }
  }
  _Pragma("unroll") for (uint t = 0; t < PF_TB; t++) {
    const uint row = row0 + 16 * t + lane;
    if (row >= rows) continue;
    _Pragma("unroll") for (uint a = 0; a < PF_WA; a++) {
      const uint col = wrow0 + 8 * a;
      if (col >= n) continue;
      ushort8 o;
      _Pragma("unroll") for (uint r = 0; r < 8; r++) o[r] = f2bf(acc[a][t][r]);
      vstore8(o, 0, y + (ulong)row * ldy + col);
    }
  }
}

// Grid (rows): zn[row][0..2303] = bf16(h[row] * inverse rms of h[row]), zn[row][2304..4607] the same for m.
KERNEL(256) void tf_ink2_pf_zeta_norm(global const float *h, global const float *m, global ushort *zn, uint rows) {
  local float red[PF_LOOP_SGS];
  const uint t = get_local_id(0), row = get_group_id(0);
  if (row >= rows) return;
  global const float *hr = h + (ulong)row * PF_HIDDEN;
  global const float *mr = m + (ulong)row * PF_HIDDEN;
  float hv[PF_HIDDEN / 256], mv[PF_HIDDEN / 256];
  float ss0 = 0.0f, ss1 = 0.0f;
#pragma unroll
  for (uint i = 0; i < PF_HIDDEN / 256; ++i) {
    hv[i] = hr[t + 256 * i];
    mv[i] = mr[t + 256 * i];
  }
#pragma unroll
  for (uint i = 0; i < PF_HIDDEN / 256; ++i) {
    ss0 += hv[i] * hv[i];
    ss1 += mv[i] * mv[i];
  }
  const float inv0 = pf_rms_inverse(pf_group_sum(ss0, red));
  const float inv1 = pf_rms_inverse(pf_group_sum(ss1, red));
  global ushort *dst = zn + (ulong)row * 2 * PF_HIDDEN;
#pragma unroll
  for (uint i = 0; i < PF_HIDDEN / 256; ++i) {
    dst[t + 256 * i] = f2bf(hv[i] * inv0);
    dst[PF_HIDDEN + t + 256 * i] = f2bf(mv[i] * inv1);
  }
}

// Grid (ceil(rows / 64)), work-group of 4 sub-groups: a sub-group makes 8 rows of the 64 ranks.
KERNEL(64) void tf_ink2_pf_zeta(global const ushort *zn, global const ushort *down, global ushort *zeta, uint rows) {
  const uint row0 = (get_group_id(0) * 4 + get_sub_group_id()) * 8;
  if (row0 >= rows) return;
  const uint k = 2 * PF_HIDDEN;
  float8 acc[4];
#pragma unroll
  for (uint j = 0; j < 4; j++) acc[j] = (float8)(0.0f);
  for (uint kk = 0; kk < k; kk += 16) {
    short8 a;
    intel_sub_group_2d_block_read_16b_8r16x1c((global void *)zn, k * 2, rows, k * 2, (int2)(kk, row0), (private ushort *)&a);
#pragma unroll
    for (uint j = 0; j < 4; j++) {
      int8 b;
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)down, k * 2, 64, k * 2, (int2)(kk / 2, 16 * j), (private uint *)&b);
      acc[j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc[j]);
    }
  }
#pragma unroll
  for (uint j = 0; j < 4; j++) {
    ushort8 o;
#pragma unroll
    for (uint r = 0; r < 8; r++) o[r] = f2bf(acc[j][r]);
    intel_sub_group_2d_block_write_16b_8r16x1c((global void *)zeta, 128, rows, 128, (int2)(16 * j, row0), (private ushort *)&o);
  }
}

// Grid (2304 / 16, ceil(rows / 16)), work-group of one sub-group: 16 rows by 16 outputs.
KERNEL(16) void tf_ink2_pf_gates(global const float *h, global const float *m, global const ushort *zeta, global const ushort *forget,
                                 global const ushort *gain, global float *out, uint rows) {
  const uint lane = get_sub_group_local_id();
  const uint d0 = get_group_id(0) * 16, row0 = get_group_id(1) * 16;
  float8 f[2], g[2];
#pragma unroll
  for (uint a = 0; a < 2; a++) {
    f[a] = (float8)(0.0f);
    g[a] = (float8)(0.0f);
  }
#pragma unroll
  for (uint kk = 0; kk < 64; kk += 16) {
    int8 bf, bg;
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)forget, 128, PF_HIDDEN, 128, (int2)(kk / 2, d0), (private uint *)&bf);
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)gain, 128, PF_HIDDEN, 128, (int2)(kk / 2, d0), (private uint *)&bg);
#pragma unroll
    for (uint a = 0; a < 2; a++) {
      short8 z;
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)zeta, 128, rows, 128, (int2)(kk, row0 + 8 * a), (private ushort *)&z);
      f[a] = intel_sub_group_bf16_bf16_matrix_mad_k16(z, bf, f[a]);
      g[a] = intel_sub_group_bf16_bf16_matrix_mad_k16(z, bg, g[a]);
    }
  }
#pragma unroll
  for (uint a = 0; a < 2; a++)
#pragma unroll
    for (uint r = 0; r < 8; r++) {
      const uint row = row0 + 8 * a + r;
      if (row >= rows) continue;
      const ulong at = (ulong)row * PF_HIDDEN + d0 + lane;
      const float fv = tanh(f[a][r]);
      const float gv = tanh(g[a][r]);
      // Four separately rounded binary32 steps; do not fold these into one expression.
      const float keep = 1.0f - fv;
      const float left = keep * h[at];
      const float right = gv * m[at];
      out[at] = left + right;
    }
}

// Copies `n` rows of every K/V head from position `from` of one ring to the same positions of another.
KERNEL(16) void tf_ink2_pf_kv_copy(global const ushort *src, global ushort *dst, uint from, uint src_ring, uint dst_ring, uint n, uint kv_heads) {
  const uint i = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id();
  if (i >= n || h >= kv_heads) return;
  const uint p = from + i;
  const ushort8 v = vload8(0, src + ((ulong)h * src_ring + p % src_ring) * PF_D + lane * 8);
  vstore8(v, 0, dst + ((ulong)h * dst_ring + p % dst_ring) * PF_D + lane * 8);
}

KERNEL(16) void tf_ink2_pf_attn_window(global const ushort *q, global const ushort *kc, global const ushort *vc, global const ushort *tab,
                                       global const PfSeg *segs, global float *part_o, global float2 *part_ml, uint heads, uint kv_heads,
                                       uint ring, float scale) {
  local ushort pl[PF_PAIRS * PF_KEYS];
  const PfSeg s = segs[0];
  const uint total = (uint)s.n * PF_GROUP, m0 = get_group_id(0) * PF_PAIRS, kvh = get_group_id(1), lane = get_sub_group_local_id();
  if (m0 >= total) return;
  const uint last_m = min(m0 + PF_PAIRS - 1, total - 1);
  const uint pos_lo = (uint)s.p0 + m0 / PF_GROUP, tile_hi = (uint)s.p0 + last_m / PF_GROUP;
  const uint first = pos_lo + 1 > PF_WINDOW ? pos_lo + 1 - PF_WINDOW : 0;
  const uint kt0 = first & ~(uint)(PF_KEYS - 1);
  const uint pair = m0 + lane, live_pair = pair < total, pair_c = min(pair, total - 1);
  const uint lpos = (uint)s.p0 + pair_c / PF_GROUP;
  const size_t prow = ((size_t)s.row0 + pair_c / PF_GROUP) * heads + kvh * PF_GROUP + pair_c % PF_GROUP;
  global const uint *qrow = (global const uint *)(q + prow * PF_D);
  global const ushort *khead = kc + s.off + (size_t)kvh * ring * PF_D;
  global const ushort *vhead = vc + s.off + (size_t)kvh * ring * PF_D;
  float8 o[2][PF_D / 16];
#pragma unroll
  for (uint h = 0; h < 2; ++h)
#pragma unroll
    for (uint j = 0; j < PF_D / 16; ++j) o[h][j] = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint kt = kt0; kt <= tile_hi; kt += PF_KEYS) {
    const int nv = (int)min((uint)PF_KEYS, tile_hi + 1 - kt);
    global const ushort *kt_k = khead + (size_t)(kt % ring) * PF_D;
    global const ushort *kt_v = vhead + (size_t)(kt % ring) * PF_D;
    float8 sc[PF_KEYS / 8];
#pragma unroll
    for (uint b = 0; b < PF_KEYS / 8; ++b) sc[b] = 0.0f;
    for (uint ds = 0; ds < PF_D / 16; ++ds) {
      ushort ka[PF_KEYS];
      intel_sub_group_2d_block_read_16b_16r16x1c((global void *)kt_k, PF_D * 2, nv, PF_D * 2, (int2)(ds * 16, 0), ka);
      const int8 qb = live_pair ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
#pragma unroll
      for (uint b = 0; b < PF_KEYS / 8; ++b) {
        short8 a;
#pragma unroll
        for (uint r = 0; r < 8; ++r) a[r] = (short)ka[b * 8 + r];
        sc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, qb, sc[b]);
      }
    }
    float peak = top;
#pragma unroll
    for (uint b = 0; b < PF_KEYS / 8; ++b)
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        const uint kp = kt + b * 8 + r;
        const bool live = live_pair && kp <= lpos && lpos - kp < PF_WINDOW;
        const float bias = live ? bf2f(tab[prow * 1024 + (lpos - kp)]) : 0.0f;
        const float x = live ? (sc[b][r] * scale + bias) * PF_LOG2E : -INFINITY;
        sc[b][r] = x;
        peak = fmax(peak, x);
      }
    const float mu = peak == -INFINITY ? 0.0f : peak;
    const float keep = exp2(top - mu);
    top = peak;
    mass *= keep;
#pragma unroll
    for (uint b = 0; b < PF_KEYS / 8; ++b)
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        const ushort pb = f2bf(exp2(sc[b][r] - mu));
        pl[lane * PF_KEYS + b * 8 + r] = pb;
        mass += bf2f(pb);
      }
#pragma unroll
    for (uint h = 0; h < 2; ++h)
#pragma unroll
      for (uint r = 0; r < 8; ++r) {
        const float kr = sub_group_broadcast(keep, h * 8 + r);
#pragma unroll
        for (uint j = 0; j < PF_D / 16; ++j) o[h][j][r] *= kr;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    short8 a0, a1;
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      a0[r] = (short)pl[r * PF_KEYS + lane];
      a1[r] = (short)pl[(8 + r) * PF_KEYS + lane];
    }
#pragma unroll
    for (uint j = 0; j < PF_D / 16; ++j) {
      uint8 vw;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)kt_v, PF_D * 2, nv, PF_D * 2, (int2)(j * 16, 0), (private uint *)&vw);
      o[0][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, as_int8(vw), o[0][j]);
      o[1][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, as_int8(vw), o[1][j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (uint h = 0; h < 2; ++h)
#pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint m = m0 + h * 8 + r;
      if (m >= total) continue;
      const size_t at = ((size_t)s.row0 + m / PF_GROUP) * heads + kvh * PF_GROUP + m % PF_GROUP;
#pragma unroll
      for (uint j = 0; j < PF_D / 16; ++j) part_o[at * PF_D + j * 16 + lane] = o[h][j][r];
    }
  if (live_pair) part_ml[prow] = (float2)(top, mass);
}

// y[row][n] = bf16(x[row][k] . W[n][k]) for BF16 W, any number of rows: dense.cl's 32-row matvec with the row groups on the
// grid's second axis, so a row's result is bit for bit the one a 32-row launch gives. Grid (ceil(n / 8), ceil(rows / 32)).
// A sub-group owns one column; lane l sums the 8-wide chunks l, l + 16, ... each as one serial chain; k a multiple of 8.
KERNEL(128) void tf_ink2_pf_dense(global const ushort *x, global const ushort *w, global ushort *y, uint rows, uint n, uint k, uint ldy) {
  const uint j = get_group_id(0) * 8 + get_sub_group_id();
  if (j >= n) return;
  const uint base = get_group_id(1) * 32, rem = min(32u, rows - base);
  const uint lane = get_sub_group_local_id(), chunks = k / 8;
  global const uint4 *wr = (global const uint4 *)(w + (ulong)j * k);
  global const ushort *xb = x + (ulong)base * k;
  float acc[32];
  #pragma unroll
  for (uint r = 0; r < 32; ++r) acc[r] = 0.0f;
  for (uint c = lane; c < chunks; c += 16) {
    const uint4 wv = wr[c];
    #pragma unroll
    for (uint r = 0; r < 32; ++r) {
      const uint4 xv = ((global const uint4 *)(xb + (ulong)min(r, rem - 1) * k))[c];
      float part = as_float(wv.x << 16) * as_float(xv.x << 16);
      part += as_float(wv.x & 0xffff0000u) * as_float(xv.x & 0xffff0000u);
      part += as_float(wv.y << 16) * as_float(xv.y << 16);
      part += as_float(wv.y & 0xffff0000u) * as_float(xv.y & 0xffff0000u);
      part += as_float(wv.z << 16) * as_float(xv.z << 16);
      part += as_float(wv.z & 0xffff0000u) * as_float(xv.z & 0xffff0000u);
      part += as_float(wv.w << 16) * as_float(xv.w << 16);
      part += as_float(wv.w & 0xffff0000u) * as_float(xv.w & 0xffff0000u);
      acc[r] += part;
    }
  }
  #pragma unroll
  for (uint r = 0; r < 32; ++r) {
    const float s = sub_group_reduce_add(acc[r]);
    if (lane == 0 && r < rem) y[(ulong)(base + r) * ldy + j] = f2bf(s);
  }
}

// glue.cl's router for 4 rows a work-group: the same sums in the same order (so the scores, the picks and the weights are bit
// for bit a one-row launch's), but each expert's weight row is read once for four independent chains. Grid (ceil(rows / 4)).
// Work-item e < 128 sums expert e's products; sub-group r of the group then picks the top 8 of row r.
KERNEL(256) void tf_ink2_pf_router(global const ushort *x, global const ushort *weight, global const float *bias,
                                   global uint *ids, global float *weights, uint rows) {
  const uint row0 = get_group_id(0) * 4, t = get_local_id(0);
  local float4 xs[4][PF_HIDDEN / 4];
  local float scores[4][128], keys[4][128], chosen[4][8];
  local uint picked[4][8];
  for (uint r = 0; r < 4; ++r) {
    global const uint2 *xg = (global const uint2 *)(x + (ulong)min(row0 + r, rows - 1) * PF_HIDDEN);
    for (uint i = t; i < PF_HIDDEN / 4; i += 256) {
      const uint2 v = xg[i];
      xs[r][i] = (float4)(as_float(v.x << 16), as_float(v.x & 0xffff0000u), as_float(v.y << 16), as_float(v.y & 0xffff0000u));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < 128) {
    global const uint4 *w = (global const uint4 *)(weight + (ulong)t * PF_HIDDEN);
    float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < PF_HIDDEN / 8; i += 4) {
      uint4 a[4];
      for (uint u = 0; u < 4; ++u) a[u] = w[i + u];
      for (uint u = 0; u < 4; ++u) {
        for (uint r = 0; r < 4; ++r) {
          const float4 b0 = xs[r][2 * (i + u)], b1 = xs[r][2 * (i + u) + 1];
          dot[r] += as_float(a[u].x << 16) * b0.x;
          dot[r] += as_float(a[u].x & 0xffff0000u) * b0.y;
          dot[r] += as_float(a[u].y << 16) * b0.z;
          dot[r] += as_float(a[u].y & 0xffff0000u) * b0.w;
          dot[r] += as_float(a[u].z << 16) * b1.x;
          dot[r] += as_float(a[u].z & 0xffff0000u) * b1.y;
          dot[r] += as_float(a[u].w << 16) * b1.z;
          dot[r] += as_float(a[u].w & 0xffff0000u) * b1.w;
        }
      }
    }
    for (uint r = 0; r < 4; ++r) {
      scores[r][t] = pf_sigmoid(dot[r]);
      keys[r][t] = scores[r][t] + bias[t];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint r = get_sub_group_id(), row = row0 + r;
  if (r < 4 && row < rows) {
    const uint lane = get_sub_group_local_id();
    float mine[8];
    for (uint j = 0; j < 8; ++j) mine[j] = keys[r][lane * 8 + j];
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
      const uint best = isnan(keys[r][first]) || largest == 128u ? first : largest;
      if (best / 8 == lane) taken |= 1u << (best % 8);
      if (lane == 0) {
        picked[r][j] = best;
        chosen[r][j] = scores[r][best];
      }
    }
    if (lane == 0) {
      for (uint j = 1; j < 8; ++j) {
        const uint id = picked[r][j];
        const float score = chosen[r][j];
        uint k = j;
        while (k > 0 && picked[r][k - 1] > id) {
          picked[r][k] = picked[r][k - 1];
          chosen[r][k] = chosen[r][k - 1];
          --k;
        }
        picked[r][k] = id;
        chosen[r][k] = score;
      }
      float den = 1e-20f;
      for (uint j = 0; j < 8; ++j) den += chosen[r][j];
      for (uint j = 0; j < 8; ++j) {
        ids[row * 8 + j] = picked[r][j];
        weights[row * 8 + j] = chosen[r][j] / den * 8.0f;
      }
    }
  }
}

// wt[d * 128 + e] = weight[e * 2304 + d]. One work-group an expert. The read is contiguous; the write is the transposed form the 16-row router reads.
KERNEL(256) void tf_ink2_pf_router_xt(global const ushort *weight, global ushort *wt) {
  const uint e = get_group_id(0), t = get_local_id(0);
  if (e >= 128) return;
  global const ushort *src = weight + (ulong)e * PF_HIDDEN;
  for (uint d = t; d < PF_HIDDEN; d += 256) wt[(ulong)d * 128u + e] = src[d];
}

// 16 rows a work-group, 128 work-items. Work-item e reads expert e from the transposed weight (one value, shared by the 16 rows) and adds the products in index order, the same order as the 4-row router. Sub-groups then pick the top 8 with that router's tie break. Grid (ceil(rows / 16)).
KERNEL(128) void tf_ink2_pf_router16(global const ushort *x, global const ushort *wt, global const float *bias,
                                     global uint *ids, global float *weights, uint rows) {
  const uint row0 = get_group_id(0) * 16, e = get_local_id(0);
  local ushort xs[16][32];
  local float scores[16][128], keys[16][128], chosen[16][8];
  local uint picked[16][8];
  float dot[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (uint base = 0; base < PF_HIDDEN; base += 32) {
    for (uint k = 0; k < 4; ++k) {
      const uint i = e + k * 128;
      const uint r = i >> 5, c = i & 31u;
      const uint row = min(row0 + r, rows - 1);
      xs[r][c] = x[(ulong)row * PF_HIDDEN + base + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint c = 0; c < 32; ++c) {
      const float w = bf2f(wt[((ulong)base + c) * 128u + e]);
      #pragma unroll
      for (uint r = 0; r < 16; ++r) dot[r] += w * bf2f(xs[r][c]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    scores[r][e] = pf_sigmoid(dot[r]);
    keys[r][e] = scores[r][e] + bias[e];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint pass = 0; pass < 2; ++pass) {
    const uint r = pass * 8 + get_sub_group_id();
    const uint row = row0 + r;
    if (row < rows) {
      const uint lane = get_sub_group_local_id();
      float mine[8];
      for (uint j = 0; j < 8; ++j) mine[j] = keys[r][lane * 8 + j];
      uint taken = 0;
      for (uint j = 0; j < 8; ++j) {
        uint first_here = 128u;
        for (int ee = 7; ee >= 0; --ee)
          if (!(taken & (1u << ee))) first_here = lane * 8 + (uint)ee;
        const uint first = sub_group_reduce_min(first_here);
        uint local_best = 8;
        for (uint ee = 0; ee < 8; ++ee)
          if (!(taken & (1u << ee)) && !isnan(mine[ee]) && (local_best == 8 || mine[ee] > mine[local_best])) local_best = ee;
        const float local_key = local_best == 8 ? -INFINITY : mine[local_best];
        const float top = sub_group_reduce_max(local_key);
        const uint candidate = (local_best != 8 && local_key == top) ? lane * 8 + local_best : 128u;
        const uint largest = sub_group_reduce_min(candidate);
        const uint best = isnan(keys[r][first]) || largest == 128u ? first : largest;
        if (best / 8 == lane) taken |= 1u << (best % 8);
        if (lane == 0) {
          picked[r][j] = best;
          chosen[r][j] = scores[r][best];
        }
      }
      if (lane == 0) {
        for (uint j = 1; j < 8; ++j) {
          const uint id = picked[r][j];
          const float score = chosen[r][j];
          uint k = j;
          while (k > 0 && picked[r][k - 1] > id) {
            picked[r][k] = picked[r][k - 1];
            chosen[r][k] = chosen[r][k - 1];
            --k;
          }
          picked[r][k] = id;
          chosen[r][k] = score;
        }
        float den = 1e-20f;
        for (uint j = 0; j < 8; ++j) den += chosen[r][j];
        for (uint j = 0; j < 8; ++j) {
          ids[row * 8 + j] = picked[r][j];
          weights[row * 8 + j] = chosen[r][j] / den * 8.0f;
        }
      }
    }
  }
}

// glue.cl's static convolution, four channels a work-item: the same arithmetic a channel, so the result is bit for bit.
// Grid (ceil(channels / 4 / 256), rows); channels a multiple of 4.
KERNEL(256) void tf_ink2_pf_conv_static(global const ushort *history, global const ushort *current, global ushort *taps,
                                        global const ushort *weight, global ushort *out, uint rows, uint channels) {
  const uint c = get_global_id(0) * 4, r = get_group_id(1);
  if (c >= channels || r >= rows) return;
  float4 window[4];
  for (uint tap = 0; tap < 4; ++tap) {
    const uint k = r + tap;
    const ushort4 v = k < 3 ? vload4(0, history + (ulong)k * channels + c) : vload4(0, current + (ulong)(k - 3) * channels + c);
    window[tap] = (float4)(bf2f(v.x), bf2f(v.y), bf2f(v.z), bf2f(v.w));
  }
  if (r == 0)
    for (uint k = 0; k < 3; ++k) vstore4(vload4(0, history + (ulong)k * channels + c), 0, taps + (ulong)k * channels + c);
  const ushort4 cur = vload4(0, current + (ulong)r * channels + c);
  vstore4(cur, 0, taps + (ulong)(r + 3) * channels + c);
  float4 sum = (float4)(bf2f(cur.x), bf2f(cur.y), bf2f(cur.z), bf2f(cur.w));
  const ushort16 wt = vload16(0, weight + (ulong)c * 4);
  for (uint tap = 0; tap < 4; ++tap) {
    const float4 wv = (float4)(bf2f(wt[tap]), bf2f(wt[4 + tap]), bf2f(wt[8 + tap]), bf2f(wt[12 + tap]));
    sum += wv * window[tap];
  }
  const float4 q = sum / PF_SQRT5;
  vstore4((ushort4)(f2bf(q.x), f2bf(q.y), f2bf(q.z), f2bf(q.w)), 0, out + (ulong)r * channels + c);
}

// attention.cl's relative-bias table, four distances a work-item and PF_REL_BATCH (row, head) pairs a work-group (the 16 projection values of a distance load once for all of them): each distance's 16 taps add in
// the same order, so the table is bit for bit the same. Grid ((rows * heads + PF_REL_BATCH - 1) / PF_REL_BATCH). A split rank's head count need not divide by the batch, so pairs past `n` are skipped.
KERNEL(256) void tf_ink2_pf_rel_table(global const ushort *rel, global const ushort *proj, global ushort *tab, uint n) {
  const uint d = get_local_id(0) * 4;
  float4 pj[PF_REL];
  for (uint j = 0; j < PF_REL; ++j) {
    const ushort4 p = vload4(0, proj + j * PF_EXTENT + d);
    pj[j] = (float4)(bf2f(p.x), bf2f(p.y), bf2f(p.z), bf2f(p.w));
  }
  for (uint b = 0; b < PF_REL_BATCH; ++b) {
    const ulong rh = (ulong)get_group_id(0) * PF_REL_BATCH + b;
    if (rh >= n) return;
    const ushort16 rr = vload16(0, rel + rh * PF_REL);
    float4 acc = (float4)(0.0f);
    for (uint j = 0; j < PF_REL; ++j) acc += bf2f(rr[j]) * pj[j];
    vstore4((ushort4)(f2bf(acc.x), f2bf(acc.y), f2bf(acc.z), f2bf(acc.w)), 0, tab + rh * PF_EXTENT + d);
  }
}

// One work-group a row. The two RMS norms are computed once and kept in local memory; each of the 48 entries then dots in the same order as the one-entry kernel.
KERNEL(256) void tf_ink2_pf_messages_row(global const ushort *bank0, global const ushort *bank1, global const float *q_mlp,
                                         global const float *kv, global float *out, uint rows) {
  local float red[PF_LOOP_SGS];
  local float beta0[PF_HIDDEN];
  local float beta1[PF_HIDDEN];
  const uint t = get_local_id(0), row = get_group_id(0);
  if (row >= rows) return;
  global const ushort *b0 = bank0 + (size_t)row * PF_HIDDEN;
  global const ushort *b1 = bank1 + (size_t)row * PF_HIDDEN;
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint d = t; d < PF_HIDDEN; d += 256) {
    const float x0 = bf2f(b0[d]), x1 = bf2f(b1[d]);
    ss0 += x0 * x0;
    ss1 += x1 * x1;
  }
  const float inv0 = pf_rms_inverse(pf_group_sum(ss0, red));
  const float inv1 = pf_rms_inverse(pf_group_sum(ss1, red));
  for (uint d = t; d < PF_HIDDEN; d += 256) {
    beta0[d] = bf2f(b0[d]) * inv0;
    beta1[d] = bf2f(b1[d]) * inv1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint entry = 0; entry < PF_MESSAGES; ++entry) {
    const uint layer = entry / 4;
    const uint role = entry - 4 * layer;
    const global float *query = ((role == 0 || role == 3) ? q_mlp : kv) + (size_t)(2 * layer + (role >= 2 ? 1 : 0)) * PF_HIDDEN;
    float dot0 = 0.0f, dot1 = 0.0f;
    for (uint d = t; d < PF_HIDDEN; d += 256) {
      dot0 += query[d] * beta0[d];
      dot1 += query[d] * beta1[d];
    }
    const float l0 = pf_group_sum(dot0, red) * PF_AR_SCALE;
    const float l1 = pf_group_sum(dot1, red) * PF_AR_SCALE;
    const float peak = fmax(fmax(-INFINITY, l0), l1);
    float w0 = exp(l0 - peak);
    float w1 = exp(l1 - peak);
    float den = w0 + w1;
    w0 = w0 / den;
    w1 = w1 / den;
    global float *dst = out + ((size_t)entry * rows + row) * PF_HIDDEN;
    for (uint d = t; d < PF_HIDDEN; d += 256) dst[d] = w0 * beta0[d] + w1 * beta1[d];
  }
}

// The same permutation as tf_ink2_pf_perm, with the 256 source values staged in local memory so the read and the write are contiguous.
KERNEL(256) void tf_ink2_pf_perm_c(global const ushort *x, global ushort *xp, uint rows, uint k) {
  local ushort tile[256];
  const uint plane = get_group_id(0), row = get_group_id(1), o = get_local_id(0);
  if (row >= rows) return;
  tile[o] = x[(ulong)row * k + plane * 256 + o];
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint s = o >> 4, l = o & 15u;
  xp[(ulong)row * k + plane * 256 + o] = tile[(s >> 1) * 32 + 2 * l + (s & 1)];
}
// Four columns a work-item and eight (row, head) pairs a work-group (two sub-groups a pair). Each column follows
// tf_ink2_merge, so the stored bf16 matches. Grid (ceil(npairs / 8)).
KERNEL(256) void tf_ink2_pf_merge(global const float *part_o, global const float2 *part_ml, global const ushort *sink, global const ushort *gate,
                                  global const int *rpos, global ushort *out, uint npairs, uint heads, uint window, uint chunk, uint c0, uint chunks, uint subs) {
  const uint sg = get_sub_group_id();
  const uint pair = get_group_id(0) * 8u + sg / 2u;
  if (pair >= npairs) return;
  const uint lane = get_sub_group_local_id() + (sg & 1u) * 16u;
  const uint d0 = lane * 4u;
  const uint row = pair / heads;
  const uint pos = (uint)rpos[row];
  const uint lo = window && pos + 1 > window ? pos + 1 - window : 0;
  float4 mm = (float4)(-INFINITY), ll = (float4)(0.0f), o = (float4)(0.0f);
  for (uint c = lo / chunk; c <= pos / chunk; ++c) {
    float4 cm, cl, co;
    if (subs > 1) {
      float4 sm = (float4)(-INFINITY), sl = (float4)(0.0f), so = (float4)(0.0f);
      const uint last = min(subs - 1u, (pos - c * chunk) / 1024u);
      for (uint u = 0; u <= last; ++u) {
        const size_t at = (size_t)pair * chunks + (c - c0) * subs + u;
        const float2 ml = part_ml[at];
        const float4 mn = fmax(sm, (float4)(ml.x));
        const float4 x = exp2(sm - mn), y = exp2((float4)(ml.x) - mn);
        sl = sl * x + (float4)(ml.y) * y;
        so = so * x + vload4(0, part_o + at * 128 + d0) * y;
        sm = mn;
      }
      cm = sm;
      cl = sl;
      co = so;
    } else {
      const size_t at = (size_t)pair * chunks + c - (window ? lo / chunk : c0);
      const float2 ml = part_ml[at];
      cm = (float4)(ml.x);
      cl = (float4)(ml.y);
      co = vload4(0, part_o + at * 128 + d0);
    }
    const float4 mn = fmax(mm, cm), x = exp2(mm - mn), y = exp2(cm - mn);
    ll = ll * x + cl * y;
    o = o * x + co * y;
    mm = mn;
  }
  const float sk = bf2f(sink[pair]) * PF_LOG2E;
  const float4 mf = fmax(mm, (float4)(sk)), a0 = exp2(mm - mf);
  const float4 yv = o * a0 / (ll * a0 + exp2((float4)(sk) - mf));
  const ushort4 gv = vload4(0, gate + (size_t)pair * 128 + d0);
  const float4 g = (float4)(bf2f(gv.x), bf2f(gv.y), bf2f(gv.z), bf2f(gv.w));
  const float4 h = yv * ((float4)(1.0f) / ((float4)(1.0f) + exp(-g)));
  vstore4((ushort4)(f2bf(h.x), f2bf(h.y), f2bf(h.z), f2bf(h.w)), 0, out + (size_t)pair * 128 + d0);
}

// Eight rows a work-group, 32 threads a row. The RMS sum is a contiguous chain a thread, then the two sub-groups of the row, so it is not the 256-thread order.
KERNEL(256) void tf_ink2_pf_norm(global const float *x, global const ushort *w, global ushort *out, uint rows) {
  local float red[16];
  const uint r = get_local_id(0) >> 5;
  const uint t = get_local_id(0) & 31u;
  const uint row0 = get_group_id(0) * 8u;
  const uint row = min(row0 + r, rows - 1u);
  global const float *xr = x + (ulong)row * PF_HIDDEN;
  float sum = 0.0f;
  for (uint d = t; d < PF_HIDDEN; d += 32) sum = fma(xr[d], xr[d], sum);
  const float sg = sub_group_reduce_add(sum);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = sg;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint sg0 = (get_local_id(0) >> 5) * 2u;
  const float total = red[sg0] + red[sg0 + 1u];
  const float inverse = 1.0f / sqrt(total / 2304.0f + PF_EPS);
  if (row0 + r < rows) {
    for (uint d = t; d < PF_HIDDEN; d += 32) {
      out[(ulong)row * PF_HIDDEN + d] = f2bf(bf2f(w[d]) * (xr[d] * inverse));
    }
  }
}

// Inkling 2 Mini prompt attention on Intel Xe2: global layers over many rows, bf16 DPAS for S = Q.K and O = P.V.
// The arithmetic follows kernels/metal/inkling2/prompt_attention.metal: score = (S * scale + bias) * log2(e), every P is
// rounded to bf16 against the running maximum after its key tile and enters the denominator and the P.V product as that
// bf16 value, and a chunk's (max, sum, P.V) are the partials the merge in attention.cl folds.
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define D 128
#define EXTENT 1024
#define GROUP 3  // query heads one K/V head serves
#define KEYS 16  // keys a tile; the ring is a multiple of it. 32 spills 3,200 bytes a thread at 256 GRF
#define PAIRS 16 // (row, query head) pairs a sub-group: one a lane, the DPAS columns

#define LOG2E 1.4426950408889634f

#define UNROLL __attribute__((opencl_unroll_hint))

#ifndef INK2_CTL
#define INK2_CTL 0
#endif

typedef struct {
  int row0, n, p0, pad;
  long off;
} InkSeg;

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); }

// One sub-group a (16 pairs, K/V head, key chunk): lane m holds pair m0 + m for the scores and the softmax state, and
// the output rows of a DPAS tile are the pairs m0 + 0..7 and m0 + 8..15. Dead keys (past a pair's position or the
// chunk) enter S as -inf and the matrix units see zero V rows, so a stale cache slot cannot reach an output.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf_ink2_attn_prompt(global const ushort *q, global const ushort *kc, global const ushort *vc, global const ushort *tab,
                                global const InkSeg *segs, global float *part_o, global float2 *part_ml, uint heads, uint kv_heads,
                                uint ring, uint chunk, uint c0, uint chunks, float scale) {
  local ushort pl[PAIRS * KEYS]; // P as A tiles: pair a lane, then (pair, key) for the matrix units
  const InkSeg s = segs[0];
  const uint total = (uint)s.n * GROUP, m0 = get_group_id(0) * PAIRS, kvh = get_group_id(1), lane = get_sub_group_local_id();
  const uint c = c0 + get_group_id(2), b_lo = c * chunk;
  if (m0 >= total) return;
  const uint tile_hi = (uint)s.p0 + min(m0 + PAIRS - 1, total - 1) / GROUP;
  if (b_lo > tile_hi) return;
  const uint k_hi = min(b_lo + chunk - 1, tile_hi);
  const uint pair = m0 + lane, live_pair = pair < total, pair_c = min(pair, total - 1);
  const uint lpos = (uint)s.p0 + pair_c / GROUP;
  const size_t prow = ((size_t)s.row0 + pair_c / GROUP) * heads + kvh * GROUP + pair_c % GROUP;
  global const uint *qrow = (global const uint *)(q + prow * D);
  global const ushort *khead = kc + s.off + (size_t)kvh * ring * D;
  global const ushort *vhead = vc + s.off + (size_t)kvh * ring * D;
  float8 o[2][D / 16];
  UNROLL
  for (uint h = 0; h < 2; ++h)
    UNROLL
    for (uint j = 0; j < D / 16; ++j) o[h][j] = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint kt = b_lo; kt <= k_hi; kt += KEYS) {
    const int nv = (int)min((uint)KEYS, k_hi + 1 - kt); // keys of the tile that exist; the rest read as zero
    global const ushort *kt_k = khead + (size_t)(kt % ring) * D;
    global const ushort *kt_v = vhead + (size_t)(kt % ring) * D;
    float8 sc[KEYS / 8];
    UNROLL
    for (uint b = 0; b < KEYS / 8; ++b) sc[b] = 0.0f;
    for (uint ds = 0; ds < D / 16; ++ds) {
      ushort ka[KEYS];
      intel_sub_group_2d_block_read_16b_16r16x1c((global void *)kt_k, D * 2, nv, D * 2, (int2)(ds * 16, 0), ka);
      const int8 qb = live_pair ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
      UNROLL
      for (uint b = 0; b < KEYS / 8; ++b) {
        short8 a;
        UNROLL
        for (uint r = 0; r < 8; ++r) a[r] = (short)ka[b * 8 + r];
        sc[b] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, qb, sc[b]);
      }
    }
    float peak = top;
    UNROLL
    for (uint b = 0; b < KEYS / 8; ++b)
      UNROLL
      for (uint r = 0; r < 8; ++r) {
        const uint kp = kt + b * 8 + r;
        const bool live = live_pair && kp <= lpos && kp <= k_hi;
        const uint distance = live ? lpos - kp : 0;
        const float bias = live && distance < EXTENT ? bf2f(tab[prow * EXTENT + distance]) : 0.0f;
#if INK2_CTL == 6
        const float x = live ? (sc[b][r] * scale + bias + (kp == b_lo ? 128.0f : 0.0f)) * LOG2E : -INFINITY;
#else
        const float x = live ? (sc[b][r] * scale + bias) * LOG2E : -INFINITY;
#endif
        sc[b][r] = x;
        peak = fmax(peak, x);
      }
    const float mu = peak == -INFINITY ? 0.0f : peak;
    const float keep = exp2(top - mu);
    top = peak;
    mass *= keep;
    UNROLL
    for (uint b = 0; b < KEYS / 8; ++b)
      UNROLL
      for (uint r = 0; r < 8; ++r) {
        const ushort pb = f2bf(exp2(sc[b][r] - mu));
        pl[lane * KEYS + b * 8 + r] = pb;
        mass += bf2f(pb); // the denominator sums the rounded P
      }
    UNROLL
    for (uint h = 0; h < 2; ++h)
      UNROLL
      for (uint r = 0; r < 8; ++r) {
        const float kr = sub_group_broadcast(keep, h * 8 + r);
        UNROLL
        for (uint j = 0; j < D / 16; ++j) o[h][j][r] *= kr;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint g = 0; g < KEYS / 16; ++g) {
      if (kt + g * 16 > k_hi) break;
      short8 a0, a1;
      UNROLL
      for (uint r = 0; r < 8; ++r) {
        a0[r] = (short)pl[r * KEYS + g * 16 + lane];
        a1[r] = (short)pl[(8 + r) * KEYS + g * 16 + lane];
      }
      UNROLL
      for (uint j = 0; j < D / 16; ++j) {
        uint8 vw;
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)kt_v, D * 2, nv, D * 2, (int2)(j * 16, g * 16), (private uint *)&vw);
        o[0][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, as_int8(vw), o[0][j]);
        o[1][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, as_int8(vw), o[1][j]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // the next tile rewrites pl
  }
  const uint slot = get_group_id(2);
  UNROLL
  for (uint h = 0; h < 2; ++h)
    UNROLL
    for (uint r = 0; r < 8; ++r) {
      const uint m = m0 + h * 8 + r;
      if (m >= total || b_lo > (uint)s.p0 + m / GROUP) continue;
      const size_t at = (((size_t)s.row0 + m / GROUP) * heads + kvh * GROUP + m % GROUP) * chunks + slot;
      UNROLL
      for (uint j = 0; j < D / 16; ++j) part_o[at * D + j * 16 + lane] = o[h][j][r];
    }
  if (live_pair && b_lo <= lpos) part_ml[prow * chunks + slot] = (float2)(top, mass);
}

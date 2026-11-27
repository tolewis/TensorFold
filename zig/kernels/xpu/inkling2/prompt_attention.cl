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

// The int8 prompt read: the grid, the softmax, the roundings and the partials of tf_ink2_attn_prompt, over the int8 cache
// of tf_ink2_prep8 (K codes in 16-position tiles [dim / 4][16 positions][4 codes], V codes row-major, one fp32 scale a
// position and K/V head, head-major). K: the sub-group reads a tile's codes a key a lane, writes them to local memory as a
// row-major bf16 tile (a code from -127 to 127 is exact in bf16) and gives the DPAS the codes, then each key's score takes
// its K scale: S = (q . code) * kscale * scale. V: a 2D read gives each lane one dim of the tile's 16 keys, and each value
// enters the P.V DPAS as bf16(code * vscale), one bf16 rounding more than the decode read. A key past the chunk enters S as
// -inf and V as an exact zero by a select: a position never written can hold any bits, also an inf or a NaN scale.
#define KROW 132 // ushorts a key row of the local K tile: 128 dims and a pad that spreads the writes over the banks
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf_ink2_attn_prompt8(global const ushort *q, global const char *kc, global const float *ksc, global const char *vc,
                                 global const float *vsc, global const ushort *tab, global const InkSeg *segs, global float *part_o,
                                 global float2 *part_ml, uint heads, uint kv_heads, uint ring, uint chunk, uint c0, uint chunks,
                                 float scale) {
  local ushort pl[PAIRS * KEYS]; // P as A tiles: pair a lane, then (pair, key) for the matrix units
  local ushort kl[KEYS * KROW];  // the tile's K codes as bf16, a key a row
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
  global const char *khead = kc + s.off + (size_t)kvh * ring * D;
  global const char *vhead = vc + s.off + (size_t)kvh * ring * D;
  global const float *ks = ksc + (size_t)kvh * ring;
  global const float *vs = vsc + (size_t)kvh * ring;
  float8 o[2][D / 16];
  UNROLL
  for (uint h = 0; h < 2; ++h)
    UNROLL
    for (uint j = 0; j < D / 16; ++j) o[h][j] = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint kt = b_lo; kt <= k_hi; kt += KEYS) {
    const int nv = (int)min((uint)KEYS, k_hi + 1 - kt); // keys of the tile that exist
    const uint at0 = kt % ring;                         // a tile start: the chunk and the ring are multiples of 16
    global const char *kt_k = khead + (size_t)at0 * D;  // tile at0 / 16 of 2,048 bytes
    global const char *kt_v = vhead + (size_t)at0 * D;
    // lane l: key l's codes. Word i of slab j sits at byte 64 (8j + i) + 4l: dims 4 (8j + i) .. 4 (8j + i) + 3.
    UNROLL
    for (uint j = 0; j < 4; ++j) {
      const uint8 x = intel_sub_group_block_read_ui8((global const uint *)(kt_k + (size_t)j * 512));
      UNROLL
      for (uint i = 0; i < 8; ++i) {
        const float4 f = convert_float4(as_char4(x[i]));
        const ushort4 b = (ushort4)((ushort)(as_uint(f.s0) >> 16), (ushort)(as_uint(f.s1) >> 16), (ushort)(as_uint(f.s2) >> 16), (ushort)(as_uint(f.s3) >> 16));
        vstore4(b, 0, kl + lane * KROW + 4 * (8 * j + i));
      }
    }
    const float kown = ks[at0 + lane]; // key l's scales; the tile lies inside the ring, so every lane reads in bounds
    const float vown = vs[at0 + lane];
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 sc[KEYS / 8];
    UNROLL
    for (uint b = 0; b < KEYS / 8; ++b) sc[b] = 0.0f;
    for (uint ds = 0; ds < D / 16; ++ds) {
      const int8 qb = live_pair ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
      UNROLL
      for (uint b = 0; b < KEYS / 8; ++b) {
        short8 a;
        UNROLL
        for (uint r = 0; r < 8; ++r) a[r] = (short)kl[(b * 8 + r) * KROW + ds * 16 + lane];
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
        const float kscale = sub_group_broadcast(kown, b * 8 + r);
        const float x = live ? (sc[b][r] * kscale * scale + bias) * LOG2E : -INFINITY;
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
    short8 a0, a1;
    UNROLL
    for (uint r = 0; r < 8; ++r) {
      a0[r] = (short)pl[r * KEYS + lane];
      a1[r] = (short)pl[(8 + r) * KEYS + lane];
    }
    UNROLL
    for (uint j = 0; j < D / 16; ++j) {
      // lane l: dim 16j + l of the tile's keys, word w holds keys 4w .. 4w + 3; rows past nv read as zero codes
      uint w8[8];
      intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)kt_v, D, nv, D, (int2)(j * 16, 0), w8);
      uint8 vw;
      UNROLL
      for (uint w = 0; w < 4; ++w) {
        const float4 f = convert_float4(as_char4(w8[w]));
        ushort e[4];
        UNROLL
        for (uint t = 0; t < 4; ++t) {
          const uint key = 4 * w + t;
          const float vscale = sub_group_broadcast(vown, key);
          e[t] = (int)key < nv ? f2bf(f[t] * vscale) : (ushort)0;
        }
        vw[2 * w] = (uint)e[0] | ((uint)e[1] << 16);
        vw[2 * w + 1] = (uint)e[2] | ((uint)e[3] << 16);
      }
      o[0][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a0, as_int8(vw), o[0][j]);
      o[1][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a1, as_int8(vw), o[1][j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE); // the next tile rewrites pl and kl
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

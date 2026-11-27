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
// position and K/V head, head-major). Both products run with the operands in the other roles, so that a key sits in a
// lane and takes its own lane's scales with no shuffle and no local tile:
// - S^T = Q . K^T. A: Q, 16 pairs as rows, staged once in 4 KB of local memory as bf16. B: the tile's K codes as bf16
//   (a code from -127 to 127 is exact in bf16), a key a lane, from one block read a slab of 16 dims. Then each score
//   takes its key's K scale: S = (q . code) * kscale * scale. A 16 x 16 transpose (32 shuffles) gives the softmax its
//   lane = pair layout, and the softmax is the one of tf_ink2_attn_prompt, in the same order.
// - O^T = V^T . P^T. A: V^T from a transposed 2D read, a key a lane, each value bf16(code * vscale), one bf16 rounding
//   more than the decode read. B: P^T, a pair a lane, packed in registers. The output rows are dims, a pair a lane.
// A key past the chunk enters S as -inf by a select and V as an exact zero: a dead key's scale is replaced by 0 before
// it multiplies a code (a code is always finite, a position never written can hold any scale, also an inf or a NaN).
// With power-of-two scales every product is the one of tf_ink2_attn_prompt, and the bench --check finds no word that
// differs. Local tiles of 8 KB a sub-group (the first forms of this kernel) cost 2.5 times the time in occupancy.

// One stage of TRANS: element (lane l, index i) and element (l ^ m, i ^ m) change places where bit m of l and of i differ.
#define TSTAGE(t, m)                                                                                                   \
  UNROLL                                                                                                               \
  for (uint i = 0; i < 16; ++i)                                                                                        \
    if ((i & (m)) == 0) {                                                                                              \
      const bool hi = (lane & (m)) != 0;                                                                               \
      const float r_ = sub_group_shuffle_xor(hi ? t[i] : t[i | (m)], (m));                                             \
      t[i] = hi ? r_ : t[i];                                                                                           \
      t[i | (m)] = hi ? t[i | (m)] : r_;                                                                               \
    }
// v0, v1 hold M[l][i] at lane l, index i (i < 8 in v0, i >= 8 in v1). After: M[i][l]. 32 shuffles.
#define TRANS(v0, v1)                                                                                                  \
  {                                                                                                                    \
    float t_[16];                                                                                                      \
    UNROLL                                                                                                             \
    for (uint i = 0; i < 8; ++i) {                                                                                     \
      t_[i] = v0[i];                                                                                                   \
      t_[i + 8] = v1[i];                                                                                               \
    }                                                                                                                  \
    TSTAGE(t_, 8)                                                                                                      \
    TSTAGE(t_, 4)                                                                                                      \
    TSTAGE(t_, 2)                                                                                                      \
    TSTAGE(t_, 1)                                                                                                      \
    UNROLL                                                                                                             \
    for (uint i = 0; i < 8; ++i) {                                                                                     \
      v0[i] = t_[i];                                                                                                   \
      v1[i] = t_[i + 8];                                                                                               \
    }                                                                                                                  \
  }

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf_ink2_attn_prompt8(global const ushort *q, global const char *kc, global const float *ksc, global const char *vc,
                                 global const float *vsc, global const ushort *tab, global const InkSeg *segs, global float *part_o,
                                 global float2 *part_ml, uint heads, uint kv_heads, uint ring, uint chunk, uint c0, uint chunks,
                                 float scale) {
  local ushort ql[PAIRS * D]; // Q as A tiles: (ds, half h, row r, dim 16 ds + e) at 256 ds + 128 h + 16 r + e
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
  {
    global const ushort *qrow = q + prow * D;
    __attribute__((opencl_unroll_hint(1)))
    for (uint ds = 0; ds < D / 16; ++ds) {
      const ushort16 v = live_pair ? vload16(0, qrow + ds * 16) : (ushort16)(0);
      vstore16(v, 0, ql + ds * 256 + (lane / 8) * 128 + (lane % 8) * 16);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  global const char *khead = kc + s.off + (size_t)kvh * ring * D;
  global const char *vhead = vc + s.off + (size_t)kvh * ring * D;
  global const float *ks = ksc + (size_t)kvh * ring;
  global const float *vs = vsc + (size_t)kvh * ring;
  float8 o[D / 8]; // o[db][r] at lane l: pair m0 + l, dim 8 db + r
  UNROLL
  for (uint db = 0; db < D / 8; ++db) o[db] = 0.0f;
  float top = -INFINITY, mass = 0.0f;
  for (uint kt = b_lo; kt <= k_hi; kt += KEYS) {
    const int nv = (int)min((uint)KEYS, k_hi + 1 - kt); // keys of the tile that exist
    const uint at0 = kt % ring;                         // a tile start: the chunk and the ring are multiples of 16
    global const char *kt_k = khead + (size_t)at0 * D;  // tile at0 / 16 of 2,048 bytes
    global const char *kt_v = vhead + (size_t)at0 * D;
    const float kown = ks[at0 + lane]; // key l's scales; the tile lies inside the ring, so every lane reads in bounds
    const float vown = vs[at0 + lane];
    const float vsel = (int)lane < nv ? vown : 0.0f; // a dead key: a finite code times 0, an exact zero
    float8 sc[2]; // S^T: sc[h][r] at lane l: pair m0 + 8h + r, key kt + l
    sc[0] = 0.0f;
    sc[1] = 0.0f;
    for (uint ds = 0; ds < D / 16; ++ds) {
      // lane l: key l, word w = dims 16 ds + 4w .. + 3
      const uint4 x = intel_sub_group_block_read_ui4((global const uint *)(kt_k + (size_t)ds * 256));
      int8 kb;
      UNROLL
      for (uint w = 0; w < 4; ++w) {
        const float4 f = convert_float4(as_char4(x[w]));
        kb[2 * w] = (int)((as_uint(f.s0) >> 16) | (as_uint(f.s1) & 0xffff0000u));
        kb[2 * w + 1] = (int)((as_uint(f.s2) >> 16) | (as_uint(f.s3) & 0xffff0000u));
      }
      UNROLL
      for (uint h = 0; h < 2; ++h) {
        const short8 a = as_short8(intel_sub_group_block_read_us8(ql + ds * 256 + h * 128)); // a[r]: pair 8h + r, dim 16 ds + lane
        sc[h] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, kb, sc[h]);
      }
    }
    sc[0] *= kown; // each key's own K scale
    sc[1] *= kown;
    TRANS(sc[0], sc[1]); // now sc[b][r] at lane l: pair m0 + l, key kt + 8b + r
    float peak = top;
    UNROLL
    for (uint b = 0; b < KEYS / 8; ++b)
      UNROLL
      for (uint r = 0; r < 8; ++r) {
        const uint kp = kt + b * 8 + r;
        const bool live = live_pair && kp <= lpos && kp <= k_hi;
        const uint distance = live ? lpos - kp : 0;
        const float bias = live && distance < EXTENT ? bf2f(tab[prow * EXTENT + distance]) : 0.0f;
        const float x = live ? (sc[b][r] * scale + bias) * LOG2E : -INFINITY;
        sc[b][r] = x;
        peak = fmax(peak, x);
      }
    const float mu = peak == -INFINITY ? 0.0f : peak;
    const float keep = exp2(top - mu);
    top = peak;
    mass *= keep;
    int8 pw; // P^T as the B operand: lane = pair, word i = keys 2i, 2i + 1
    UNROLL
    for (uint b = 0; b < KEYS / 8; ++b)
      UNROLL
      for (uint r = 0; r < 8; r += 2) {
        const ushort p0 = f2bf(exp2(sc[b][r] - mu)), p1 = f2bf(exp2(sc[b][r + 1] - mu));
        mass += bf2f(p0); // the denominator sums the rounded P
        mass += bf2f(p1);
        pw[(b * 8 + r) / 2] = (int)((uint)p0 | ((uint)p1 << 16));
      }
    UNROLL
    for (uint db = 0; db < D / 8; ++db) o[db] *= keep;
    UNROLL
    for (uint cb = 0; cb < D / 32; ++cb) {
      // V^T: lane l = key l, word w = dims 32 cb + 4w .. + 3; rows past nv read as zero
      uint v8[8];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)kt_v, D, nv, D, (int2)(cb * 8, 0), v8);
      UNROLL
      for (uint sb = 0; sb < 4; ++sb) {
        short8 a; // a[r]: dim 32 cb + 8 sb + r, key lane: bf16(code * vscale), dead keys an exact zero
        const float4 f0 = convert_float4(as_char4(v8[2 * sb])), f1 = convert_float4(as_char4(v8[2 * sb + 1]));
        UNROLL
        for (uint r = 0; r < 4; ++r) {
          a[r] = (short)f2bf(f0[r] * vsel);
          a[r + 4] = (short)f2bf(f1[r] * vsel);
        }
        o[4 * cb + sb] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, pw, o[4 * cb + sb]);
      }
    }
  }
  if (live_pair && b_lo <= lpos) {
    const uint slot = get_group_id(2);
    global float *po = part_o + (prow * chunks + slot) * D;
    UNROLL
    for (uint db = 0; db < D / 8; ++db) vstore8(o[db], 0, po + 8 * db);
    part_ml[prow * chunks + slot] = (float2)(top, mass);
  }
}

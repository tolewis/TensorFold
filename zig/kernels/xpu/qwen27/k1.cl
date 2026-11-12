// Qwen3.8-27B decode attention over int8 pool pages on the XMX (K1), ported from Sage's attn_decode_dpas
// (decode.cl) to the 27B shape: Q27_GQA (6) query heads of a K/V head, a row group of up to K1_ROWS rows of one
// stream (a verify window's rows, or one row), pages through the stream's page table. Two variants share k1_body.h:
// attn_k1_decode (8 rows a group, 256-key tiles) and attn_k1_decode16 (16 rows a group, 96 columns, 128-key
// tiles: a 16-row window reads each K/V block once). See k1_body.h for the layout and the argument list.
#pragma OPENCL FP_CONTRACT OFF
#include "shape.h"

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }

#if Q27_GQA == 6
// P is f16 with the V scale folded in: at 128k keys a p of 1e-5 times a V scale of 0.02 is 2e-7, below the f16
// normal range (6e-5), so the tail of the softmax loses its bits while l is summed in fp32. P is scaled by
// K1_PSCALE before the half conversion (p x vs at most about 0.5, so 2048 at most) and o divided once at the end.
#define K1_PSCALE 4096.0f

#define K1_FN attn_k1_decode
#define K1_R 8
#define K1_C (8 * Q27_GQA)
#define K1_P ((K1_C + 15) / 16)
#define K1_T 256
#define K1_G 16
#define K1_PROBE 0
#include "k1_body.h"
#undef K1_PROBE
#undef K1_FN
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G

#define K1_FN attn_k1_decode16
#define K1_R 16
#define K1_C (16 * Q27_GQA)
#define K1_P ((K1_C + 15) / 16)
#define K1_T 128
#define K1_G 8
#define K1_PROBE 0
#include "k1_body.h"
#undef K1_PROBE
#undef K1_FN
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G

// K1B (lane p4-kernel, XPU_K1B): attn_k1_decode with the reads of a tile ahead (k1_body.h, K1_KAHEAD and
// K1_VAHEAD). b1: K ahead. b2: V ahead at the start of PV. b3: V ahead right after the QK DPAS. b5 to b7 (K1_PFSPREAD): the next
// tile's prefetches one a DPAS step, b5 V, b6 K, b7 both. b8: b7 with two blocks a prefetch, every second step. Arguments, grid,
// values and bits are those of attn_k1_decode. K and V ahead together spill 3520 bytes (IGC asm), so no variant has both.
#define K1_R 8
#define K1_C (8 * Q27_GQA)
#define K1_P ((K1_C + 15) / 16)
#define K1_T 256
#define K1_G 16
#define K1_PROBE 0
#define K1_FN attn_k1_b1
#define K1_KAHEAD 1
#include "k1_body.h"
#undef K1_FN
#define K1_FN attn_k1_b2
#define K1_VAHEAD 1
#include "k1_body.h"
#undef K1_FN
#define K1_FN attn_k1_b3
#define K1_VAHEAD 2
#include "k1_body.h"
#undef K1_FN
#define K1_FN attn_k1_b5
#define K1_PFSPREAD 1
#include "k1_body.h"
#undef K1_FN
#define K1_FN attn_k1_b6
#define K1_PFSPREAD 2
#include "k1_body.h"
#undef K1_FN
#define K1_FN attn_k1_b7
#define K1_PFSPREAD 3
#include "k1_body.h"
#undef K1_FN
#define K1_FN attn_k1_b8
#define K1_PFSPREAD 7
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G

// Timing probes of attn_k1_decode (lane p4-k-k1, XPU_K1_SHADOW): shadow launches into scratch partials. See
// K1_PROBE in k1_body.h. They never feed the model.
#define K1_R 8
#define K1_C (8 * Q27_GQA)
#define K1_P ((K1_C + 15) / 16)
#define K1_T 256
#define K1_G 16
#define K1_FN attn_k1_nb3
#define K1_PROBE 1
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#define K1_FN attn_k1_nb
#define K1_PROBE 2
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#define K1_FN attn_k1_nv
#define K1_PROBE 4
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#define K1_FN attn_k1_nq
#define K1_PROBE 8
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#define K1_FN attn_k1_ns
#define K1_PROBE 16
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#define K1_FN attn_k1_nqi
#define K1_PROBE 32
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#define K1_FN attn_k1_nx
#define K1_PROBE 64
#include "k1_body.h"
#undef K1_FN
#undef K1_PROBE
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G

// attn_k1_k16 (lane p4-kernel, XPU_K1_R16=2): one pass of 16 rows over the K/V, at the 8-row kernel's cost a tile and
// thread. A 16-row tree verify runs attn_k1_decode as two 8-row groups, so each K and V row is read and each V row is
// converted to f16 twice. attn_k1_decode16 reads them once, but QK runs on 8 of its 16 sub-groups with all 6 passes,
// and a key and row costs it about 1.75 times the instructions of attn_k1_decode.
// Here QK splits the 6 passes over two halves of the work-group: sub-group sg takes the 16 keys of block sg % 8 and
// the 3 passes of half sg / 8. Each sub-group then holds the scores of 3 passes, as attn_k1_decode does. PV stays as
// in attn_k1_decode: 16 sub-groups of 16 dimensions, with 12 column blocks over 128 keys, so 96 DPAS a tile and
// one V conversion for 16 rows. Tiles of 128 keys keep the local memory at 54.75 KB a work-group (two a core).
// IGC asm: no spill, about 2704 instructions a tile and thread for 128 keys and 16 rows (attn_k1_decode: 3438 for
// 256 keys and 8 rows), so 21 % fewer a key and row, with the same DPAS count a key and row.
// Arguments, grid and partials are those of attn_k1_decode with 16-row groups (GRP). The values differ from the two
// 8-row groups by the f16 rounding of P at other tile maxima and by the fp32 order of the split merge.
#define K16_C (16 * Q27_GQA)                       // 96 columns (row, head)
#define K16_P (K16_C / 16)                         // 6 DPAS passes of 16 columns
#define K16_H (K16_P / 2)                          // 3 passes a sub-group in QK
#define K16_T 128                                  // keys a tile
#define K16_B (K16_T / 16)                         // 8 key blocks a tile
#ifndef K16_UNROLL
#define K16_UNROLL 2                               // PV: two 32-key steps a loop pass (a full unroll spills 3008 bytes)
#endif
KERNEL(256) void attn_k1_k16(global const ushort *Q, global const char *KC, global const char *VC,
                             global float *PM, global float *PL, global float *PO, int p0, int NS, int span,
                             float scale, global const ulong *MT, global const int *POS, int cap,
                             global const uint *PT, global const int *GRP) {
  local uint QI[K16_P * 64 * 16];                 // (pass, 32-dimension step, word, lane): Q in int8, column (row, head)
  local ushort P[K16_B * K16_C * 16];             // (16-key block, column, key): P in f16 with the V scale folded in
  local float QS[K16_C], MX[K16_P][K16_B][16], SM[K16_P][K16_B][16], AL[K16_C];
  const int sp = get_group_id(0), hk = get_group_id(1), g = get_group_id(2);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int kbk = sg % K16_B, ph = sg / K16_B;     // QK: the sub-group's key block and its half of the passes
  const int t0 = GRP[2 * g], nr = GRP[2 * g + 1], npass = (nr * Q27_GQA + 15) / 16;
  const int myp = npass - ph * K16_H;              // passes of this half that hold columns (at most K16_H)
  if (MT) {                                        // the rows of a group share a stream
    KC = (global const char *)MT[t0 * 8];
    VC = (global const char *)MT[t0 * 8 + 1];
    PT = (global const uint *)MT[t0 * 8 + 2];
  }
  int lastr[16], max_last = -1;
  for (int r = 0; r < 16; r++) {
    lastr[r] = r < nr ? (MT ? POS[t0 + r] : p0 + t0 + r) : -1;
    max_last = max(max_last, lastr[r]);
  }
  const int nkeys = max_last + 1, j0 = sp * span, j1 = min(j0 + span, nkeys);
  const int qh0 = hk * Q27_GQA;
  for (int c = sg; c < K16_C; c += 16) {           // column c: row c / Q27_GQA, head c % Q27_GQA; the lane's 16 dims
    const int pass = c / 16, col = c % 16, row = c / Q27_GQA, head = c % Q27_GQA;
    if (c < Q27_GQA * nr) {
      const ushort16 x = vload16(0, Q + ((long)(t0 + row) * Q27_AQ + qh0 + head) * 256 + lane * 16);
      float mq = 0.0f;
      for (int b = 0; b < 16; b++) mq = fmax(mq, fabs(bf2f(x[b])));
      mq = sub_group_reduce_max(mq);
      const float qs = mq > 0.0f ? mq / 127.0f : 1.0f, iq = 1.0f / qs;
      for (int j = 0; j < 4; j++) {
        uint u = 0;
        for (int b = 0; b < 4; b++) u |= ((uint)(uchar)(char)rint(bf2f(x[j * 4 + b]) * iq)) << (8 * b);
        QI[(pass * 64 + (lane / 2) * 8 + (lane % 2) * 4 + j) * 16 + col] = u;
      }
      if (lane == 0) QS[c] = qs;
    } else {
      for (int j = 0; j < 4; j++) QI[(pass * 64 + (lane / 2) * 8 + (lane % 2) * 4 + j) * 16 + col] = 0;
      if (lane == 0) QS[c] = 0.0f;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float qsl[K16_H];                                // the lane's column of each own pass: (ph * K16_H + pp) * 16 + lane
  int lastc[K16_H];
  for (int pp = 0; pp < K16_H; pp++) {
    const int p = ph * K16_H + pp;
    qsl[pp] = QS[p * 16 + lane] * scale * M_LOG2E_F;
    lastc[pp] = lastr[(p * 16 + lane) / Q27_GQA];
  }
  float8 o[K16_C / 8];                             // lane: dimension 16 sg + lane; o[c / 8]: column c % 8
  for (int q = 0; q < K16_C / 8; q++) o[q] = 0.0f;
  float m[K16_H], l[K16_H];                        // the own passes' running max and sum
  for (int pp = 0; pp < K16_H; pp++) {
    m[pp] = -INFINITY;
    l[pp] = 0.0f;
  }
  if (j0 < j1 && ph == 0) {                        // the first tile's K rows (half 0 prefetches K, half 1 V)
    const int kb = j0 + kbk * 16;
    if (kb < j1) {
      global const char *kp = KC + (long)PT[kb / Q27_PAGE] * Q27_PAGE8 + hk * 256;
      const int rows = min(Q27_PAGE, nkeys - (kb / Q27_PAGE) * Q27_PAGE);
      for (int ds = 0; ds < 8; ds++)
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)kp, 256, rows, Q27_KVROW, (int2)(ds * 16, kb % Q27_PAGE));
    }
  }
  for (int k0 = j0; k0 < j1; k0 += K16_T) {
    const int kb = k0 + kbk * 16, kpg = kb / Q27_PAGE;
    const bool kin = kb < j1;
    const long kpb = kin ? (long)PT[kpg] * Q27_PAGE8 : 0;
    const int krows = kin ? min(Q27_PAGE, nkeys - kpg * Q27_PAGE) : 0;
    global const char *kpage = KC + kpb + hk * 256;
    { // the next tile's V rows: half 1, the 16 keys of block kbk
      const int vb = k0 + K16_T + kbk * 16;
      if (ph == 1 && vb < j1) {
        const int pg = vb / Q27_PAGE;
        global const char *vp = VC + (long)PT[pg] * Q27_PAGE8 + hk * 256;
        const int rows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
        for (int ds = 0; ds < 8; ds++)
          intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)vp, 256, rows, Q27_KVROW, (int2)(ds * 16, vb % Q27_PAGE));
      }
    }
    int8 si[K16_H][2];
    for (int pp = 0; pp < K16_H; pp++) si[pp][0] = si[pp][1] = 0;
    if (kin) {
      __attribute__((opencl_unroll_hint))
      for (int ds = 0; ds < 8; ds++) {             // steps of 32 dimensions: 16 keys x 32 bytes, 2 A tiles of 8 keys
        ushort ka[16];
        intel_sub_group_2d_block_read_16b_16r16x1c((global void *)kpage, 256, krows, Q27_KVROW, (int2)(ds * 16, kb % Q27_PAGE), ka);
        short8 a[2];
        for (int b = 0; b < 2; b++)
          for (int r = 0; r < 8; r++) a[b][r] = (short)ka[b * 8 + r];
        for (int pp = 0; pp < K16_H; pp++)
          if (pp < myp) {
            const int8 qb = as_int8(intel_sub_group_block_read8(QI + ((ph * K16_H + pp) * 64 + ds * 8) * 16));
            si[pp][0] = intel_sub_group_i8_i8_matrix_mad_k32(a[0], qb, si[pp][0]);
            si[pp][1] = intel_sub_group_i8_i8_matrix_mad_k32(a[1], qb, si[pp][1]);
          }
      }
    }
    const int key = kb + lane;
    const bool kok = kin && key < j1;
    const float ksc = kok ? ((global const float *)(KC + kpb + (long)Q27_PAGE * Q27_KVROW))[(key % Q27_PAGE) * Q27_AKV + hk] : 0.0f;
    const float vsc = kok ? ((global const float *)(VC + kpb + (long)Q27_PAGE * Q27_KVROW))[(key % Q27_PAGE) * Q27_AKV + hk] : 0.0f;
    float s[K16_H][16], mx[K16_H], alpha[K16_H], mn[K16_H];
    for (int pp = 0; pp < K16_H; pp++) {
      mx[pp] = -INFINITY;
      if (pp < myp) {
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < 16; kk++) {
          const float ks = sub_group_broadcast(ksc, kk);
          const float x = (kin && kb + kk <= lastc[pp]) ? (float)si[pp][kk / 8][kk % 8] * ks * qsl[pp] : -INFINITY;
          s[pp][kk] = x;
          mx[pp] = fmax(mx[pp], x);
        }
      }
      MX[ph * K16_H + pp][kbk][lane] = mx[pp];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int pp = 0; pp < K16_H; pp++) {
      const int p = ph * K16_H + pp;
      float tm = -INFINITY;
      for (int i = 0; i < K16_B; i++) tm = fmax(tm, MX[p][i][lane]);
      mn[pp] = fmax(m[pp], tm);
      alpha[pp] = mn[pp] == -INFINITY ? 1.0f : native_exp2(m[pp] - mn[pp]);
      float sum = 0.0f;
      if (pp < myp) {
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < 16; kk++) {
          const float pr = mn[pp] == -INFINITY ? 0.0f : native_exp2(s[pp][kk] - mn[pp]);
          sum += pr;
          P[(kbk * K16_C + p * 16 + lane) * 16 + kk] = as_ushort(convert_half(pr * sub_group_broadcast(vsc, kk) * K1_PSCALE));
        }
      }
      SM[p][kbk][lane] = sum;
      if (kbk == 0) AL[p * 16 + lane] = alpha[pp];
      m[pp] = mn[pp];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int pp = 0; pp < K16_H; pp++) {
      const int p = ph * K16_H + pp;
      float ls = 0.0f;
      for (int i = 0; i < K16_B; i++) ls += SM[p][i][lane];
      l[pp] = l[pp] * alpha[pp] + ls;
    }
    for (int q = 0; q < K16_C / 8; q++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) o[q][r] *= AL[q * 8 + r];
    { // the next tile's K rows: half 0, the 16 keys of block kbk
      const int nb = k0 + K16_T + kbk * 16;
      if (ph == 0 && nb < j1) {
        const int pg = nb / Q27_PAGE;
        global const char *np = KC + (long)PT[pg] * Q27_PAGE8 + hk * 256;
        const int rows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
        for (int ds = 0; ds < 8; ds++)
          intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)np, 256, rows, Q27_KVROW, (int2)(ds * 16, nb % Q27_PAGE));
      }
    }
    __attribute__((opencl_unroll_hint(K16_UNROLL)))
    for (int c2 = 0; c2 < K16_T / 32; c2++) {     // PV, 32 keys a read: lane = dimension 16 sg + lane, word j = keys 4j..
      const int y0 = k0 + c2 * 32;
      if (y0 >= j1) break;
      const int pg = y0 / Q27_PAGE;
      global const char *vpage = VC + (long)PT[pg] * Q27_PAGE8 + hk * 256;
      const int vrows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
      uint w8[8];
      intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)vpage, 256, vrows, Q27_KVROW, (int2)(sg * 16, y0 % Q27_PAGE), w8);
      __attribute__((opencl_unroll_hint))
      for (int hc = 0; hc < 2; hc++) {             // 16 keys: 8 fp16 pairs, int8 to fp16 exact
        uint8 vw;
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 4; j++) {
          const half4 h = convert_half4(as_char4(w8[hc * 4 + j]));
          vw[2 * j] = as_uint(h.s01);
          vw[2 * j + 1] = as_uint(h.s23);
        }
        const int kg = c2 * 2 + hc;
        for (int q = 0; q < K16_C / 8; q++) {
          const short8 pa = as_short8(intel_sub_group_block_read_us8(P + (kg * K16_C + q * 8) * 16));
          o[q] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(vw), o[q]);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);                  // P, MX, SM and AL are reused by the next tile
  }
  if (kbk == 0)                                    // sub-groups 0 and 8: the m and l of their passes
    for (int pp = 0; pp < K16_H; pp++) {
      const int c = (ph * K16_H + pp) * 16 + lane;
      if (pp < myp && c < Q27_GQA * nr) {
        const int row = c / Q27_GQA, head = c % Q27_GQA;
        const long base = ((long)(t0 + row) * Q27_AQ + qh0 + head) * NS + sp;
        PM[base] = m[pp];
        PL[base] = l[pp];
      }
    }
  for (int q = 0; q < K16_C / 8; q++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      const int c = q * 8 + r;
      if (c < Q27_GQA * nr) {
        const int row = c / Q27_GQA, head = c % Q27_GQA;
        PO[(((long)(t0 + row) * Q27_AQ + qh0 + head) * NS + sp) * 256 + sg * 16 + lane] = o[q][r] * (1.0f / K1_PSCALE);
      }
    }
}
#undef K16_C
#undef K16_P
#undef K16_H
#undef K16_T
#undef K16_B
#undef K16_UNROLL

// Tree verify draft (lane p22-k1tree): a row's draft keys are the tree table's mask of its cache rows - its own
// row and its ancestors', rows p0 + the ancestor rows, at most 16. The prefix keys are attn_k1_decode's slices
// 0 .. NS - 1; this kernel writes slice NS, the partials in the same domain (m in log2, l in fp32, o unnormalized),
// so attn_merge_span joins the two. The partials of a tree verify have stride NS + 1 per (row, head): the prefix
// launch and attn_merge_span get NS + 1 as their NS, and this kernel takes NS as the slice count (its slice index)
// and writes at stride NS + 1, so slice NS sits after the prefix slices 0 .. NS - 1 and never aliases the next head.
// One sub-group of 16 lanes per (K/V head, row): the 6 query heads x 16 dims a lane. Grid (1, AKV, rows). The draft
// rows' K/V pages must cover p0 + rows - 1 (verifyTree's ensurePages).
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]
KERNEL(16) void attn_k1_draft(global const ushort *Q, global const char *KC, global const char *VC,
                              global float *PM, global float *PL, global float *PO, int p0, int NS,
                              float scale, global const int *TREE, global const uint *PT) {
  const int hk = get_group_id(1), t = get_group_id(2), lane = get_sub_group_local_id();
  const int qh0 = hk * Q27_GQA;
  float q[Q27_GQA][16], o[Q27_GQA][16], m[Q27_GQA], l[Q27_GQA];
  for (int a = 0; a < Q27_GQA; a++) {
    for (int i = 0; i < 16; i++) {
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + lane * 16 + i]) * scale * M_LOG2E_F;
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int d = 0; d <= DEPTH(t); d++) {            // its own row and its ancestors' cache rows
    const int cr = p0 + ANC(t, d);
    const long pb = (long)PT[cr / Q27_PAGE] * Q27_PAGE8;
    const int pr = cr % Q27_PAGE;
    const long kr = (long)pr * Q27_AKV + hk;
    const char16 k8 = vload16(lane, KC + pb + kr * 256);
    const float ks = ((global const float *)(KC + pb + (long)Q27_PAGE * Q27_KVROW))[pr * Q27_AKV + hk];
    float sc[Q27_GQA];
    for (int a = 0; a < Q27_GQA; a++) {
      float dd = 0.0f;
      for (int i = 0; i < 16; i++) dd += q[a][i] * (float)k8[i] * ks;
      sc[a] = sub_group_reduce_add(dd);
    }
    const char16 v8 = vload16(lane, VC + pb + kr * 256);
    const float vs = ((global const float *)(VC + pb + (long)Q27_PAGE * Q27_KVROW))[pr * Q27_AKV + hk];
    for (int a = 0; a < Q27_GQA; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = native_exp2(m[a] - mn), p = native_exp2(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * (float)v8[i] * vs;
      m[a] = mn;
    }
  }
  for (int a = 0; a < Q27_GQA; a++) {              // slice NS: the partials' layout as attn_k1_decode's
    const long base = ((long)t * Q27_AQ + qh0 + a) * (NS + 1) + NS;
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + lane * 16 + i] = o[a][i];
  }
}
#undef DEPTH
#undef ANC

// Shared-round tree draft (lane p33-multitree, XPU_MULTI_TREE): attn_k1_draft across the block's tree table, the
// rows of several streams. A tree row reads its D + 1 draft keys - its own cache row and its ancestors' - through
// its slot's page table (MT[t*8+2]), at key POS[wk] + the ancestor row - wk: wk, the row's window first row
// (MT[t*8+6]), and POS the round's per-row positions, so POS[wk] is the window base. A chain row carries the
// NO_DRAFT sentinel (bit 1024) and skips, its merge slice skipped by its kposm. The partials write slice NS at
// the row's NS + 1 stride, as attn_k1_draft. Grid (1, AKV, rows).
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]
#define NODR(t) (TREE[(t) * 17] & 1024)
KERNEL(16) void attn_k1_draft_mt(global const ushort *Q, global const char *KC, global const char *VC,
                                 global float *PM, global float *PL, global float *PO, int p0, int NS,
                                 float scale, global const int *TREE, global const ulong *MT, global const int *POS) {
  const int hk = get_group_id(1), t = get_group_id(2), lane = get_sub_group_local_id();
  if (NODR(t)) return;
  const int D = DEPTH(t), wk = (int)MT[(t) * 8 + 6], b = POS[wk];
  const global const char *kc8 = (global const char *)MT[(t) * 8];
  const global const char *vc8 = (global const char *)MT[(t) * 8 + 1];
  const global const uint *PT = (global const uint *)MT[(t) * 8 + 2];
  const int qh0 = hk * Q27_GQA;
  float q[Q27_GQA][16], o[Q27_GQA][16], m[Q27_GQA], l[Q27_GQA];
  for (int a = 0; a < Q27_GQA; a++) {
    for (int i = 0; i < 16; i++) {
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + lane * 16 + i]) * scale * M_LOG2E_F;
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int d = 0; d <= D; d++) {            // its own row and its ancestors' cache rows
    const int cr = b + ANC(t, d) - wk;
    const long pb = (long)PT[cr / Q27_PAGE] * Q27_PAGE8;
    const int pr = cr % Q27_PAGE;
    const long kr = (long)pr * Q27_AKV + hk;
    const char16 k8 = vload16(lane, kc8 + pb + kr * 256);
    const float ks = ((global const float *)(kc8 + pb + (long)Q27_PAGE * Q27_KVROW))[pr * Q27_AKV + hk];
    float sc[Q27_GQA];
    for (int a = 0; a < Q27_GQA; a++) {
      float dd = 0.0f;
      for (int i = 0; i < 16; i++) dd += q[a][i] * (float)k8[i] * ks;
      sc[a] = sub_group_reduce_add(dd);
    }
    const char16 v8 = vload16(lane, vc8 + pb + kr * 256);
    const float vs = ((global const float *)(vc8 + pb + (long)Q27_PAGE * Q27_KVROW))[pr * Q27_AKV + hk];
    for (int a = 0; a < Q27_GQA; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = native_exp2(m[a] - mn), p = native_exp2(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * (float)v8[i] * vs;
      m[a] = mn;
    }
  }
  for (int a = 0; a < Q27_GQA; a++) {              // slice NS: the partials' layout as attn_k1_decode's
    const long base = ((long)t * Q27_AQ + qh0 + a) * (NS + 1) + NS;
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + lane * 16 + i] = o[a][i];
  }
}
#undef DEPTH
#undef ANC
#undef NODR
#endif // Q27_GQA == 6

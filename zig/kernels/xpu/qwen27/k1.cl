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
#include "k1_body.h"
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
#include "k1_body.h"
#undef K1_FN
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G

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
#endif // Q27_GQA == 6

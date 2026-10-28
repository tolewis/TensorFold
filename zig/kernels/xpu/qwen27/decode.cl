// Qwen3.8-27B decode kernels on Intel Xe2: serial GDN (a few tokens) and attention over the cache.
#pragma OPENCL FP_CONTRACT OFF
#include "shape.h"

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}

// Tree table of a verified block (<= 16 rows): TREE[t*17] = depth | SAVE | LOAD flags, then the ancestor row per depth.
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define SAVE(t) (TREE[(t) * 17] & 256)
#define LOAD(t) (TREE[(t) * 17] & 512)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]

// Serial GDN over T tokens: one subgroup per (8 v rows, V head), the lane holds 8 k columns; ROWS, TREE, MT: modes.
KERNEL(16) void gdn_step(global const ushort *Q, global const ushort *K, global const ushort *V, global const float *G,
                         global const float *BETA, global const float *S, global float *SO, global ushort *Y, int T,
                         global const int *ROWS, global const int *TREE, global float *SL, global const ulong *MT) {
  // MT (if non-null): several streams, grid (16, Q27_GV, rows); a window's first row starts from its stream's state.
  const int vb = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / Q27_GVK;
  int t0 = 0;
  if (MT) {
    t0 = get_group_id(2);
    if ((int)MT[t0 * 8 + 6] != t0) return;            // not a window's first row
    S = (global const float *)MT[t0 * 8 + 4];
    SO = (global float *)MT[t0 * 8 + 5];
    T = t0 + (int)MT[t0 * 8 + 7];
  }
  float s[8][8];
  for (int r = 0; r < 8; r++)
    for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
  for (int t = t0; t < T; t++) {
    const int row = ROWS ? ROWS[t] : t;
    if (TREE && DEPTH(t) > 0 && LOAD(t)) {              // the parent's state, from its slot
      const int par = ANC(t, DEPTH(t) - 1);
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = SL[(((long)par * Q27_GV + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    } else if (TREE && DEPTH(t) == 0)
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    float q[8], k[8];
    for (int i = 0; i < 8; i++) {
      q[i] = bf2f(Q[((long)row * Q27_GK + kh) * 128 + i * 16 + lane]);
      k[i] = bf2f(K[((long)row * Q27_GK + kh) * 128 + i * 16 + lane]);
    }
    const float g = G[(long)row * Q27_GV + h], beta = BETA[(long)row * Q27_GV + h];
    for (int r = 0; r < 8; r++) {
      float mem = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] *= g;
        mem += s[r][i] * k[i];
      }
      mem = sub_group_reduce_add(mem);
      const float u = (bf2f(V[((long)row * Q27_GV + h) * 128 + vb * 8 + r]) - mem) * beta;
      float y = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] += k[i] * u;
        y += s[r][i] * q[i];
      }
      y = sub_group_reduce_add(y);
      if (lane == 0) Y[((long)t * Q27_GV + h) * 128 + vb * 8 + r] = f2bf(y);
    }
    if (TREE && SAVE(t))
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) SL[(((long)t * Q27_GV + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
  }
  if (!TREE)
    for (int r = 0; r < 8; r++)
      for (int i = 0; i < 8; i++) SO[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
}

// Decode attention by key slices: one subgroup per (slice, K/V head and query part, row) handles Q27_QPG query heads
// (27B: 6, all of the K/V head's); partials PM, PL, PO. Grid (slices, Q27_AKV * Q27_QPARTS, rows).
#define SPLIT 32
KERNEL(16) void attn_decode(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,
                            global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,
                            global const ulong *MT, global const int *POS, int cap) {
  // MT (if non-null): several streams, one row each; with TREE, tree windows; cap > 0: int8 cache (256 bytes per row).
  const int sp = get_group_id(0), hk = get_group_id(1) / Q27_QPARTS, t = get_group_id(2), lane = get_sub_group_local_id();
  const int qh0 = hk * Q27_GQA + (get_group_id(1) % Q27_QPARTS) * Q27_QPG; // the first query head held
  int base = p0, wf = 0;
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
    wf = (int)MT[t * 8 + 6];
    base = TREE ? POS[wf] : 0x7fffffff;
  }
  const int last = MT ? POS[t] : p0 + DEPTH(t), j0 = sp * SPLIT, j1 = min(j0 + SPLIT, last + 1);
  float q[Q27_QPG][16], o[Q27_QPG][16], m[Q27_QPG], l[Q27_QPG];
  for (int a = 0; a < Q27_QPG; a++) {
    for (int i = 0; i < 16; i++) {
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int j = j0; j < j1; j++) {
    const long kr = j < base ? j : base + ANC(t, j - base) - wf; // cache row of key j
    float kv[16];
    if (cap) {
      const global char *k8 = (const global char *)KC + (kr * Q27_AKV + hk) * 256;
      const float ks = ((const global float *)((const global char *)KC + (long)cap * Q27_KVROW))[kr * Q27_AKV + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)k8[i * 16 + lane] * ks;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(KC[(kr * Q27_AKV + hk) * 256 + i * 16 + lane]);
    float sc[Q27_QPG];
    for (int a = 0; a < Q27_QPG; a++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    if (cap) {
      const global char *v8 = (const global char *)VC + (kr * Q27_AKV + hk) * 256;
      const float vs = ((const global float *)((const global char *)VC + (long)cap * Q27_KVROW))[kr * Q27_AKV + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)v8[i * 16 + lane] * vs;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(VC[(kr * Q27_AKV + hk) * 256 + i * 16 + lane]);
    for (int a = 0; a < Q27_QPG; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < Q27_QPG; a++) {
    const long base = ((long)t * Q27_AQ + qh0 + a) * NS + sp;
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + i * 16 + lane] = o[a][i];
  }
}

// Merge of the slices, in order: O bf16 [T, Q27_AQ, 256]. One subgroup per (head, row).
KERNEL(16) void attn_merge(global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                           int p0, int NS, global const int *TREE, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int used = (POS ? POS[t] : p0 + DEPTH(t)) / SPLIT + 1; // non-empty slices of row t
  const long base = ((long)t * Q27_AQ + hh) * NS;
  float m = -INFINITY;
  for (int s = 0; s < used; s++) m = fmax(m, PM[base + s]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = 0; s < used; s++) {
    const float w = exp(PM[base + s] - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  for (int i = 0; i < 16; i++) O[((long)t * Q27_AQ + hh) * 256 + i * 16 + lane] = f2bf(o[i] / l);
}

// Commit of a kept path: its K, V rows move to positions p0 + depth in depth order, never overwriting an unread row.
KERNEL(16) void kv_move(global ushort *KC, global ushort *VC, int p0, global const int *PATH, int n, int cap) {
  const int hk = get_group_id(0), lane = get_sub_group_local_id();
  for (int d = 0; d < n; d++) {
    const long src = ((long)(p0 + PATH[d]) * Q27_AKV + hk) * 256, dst = ((long)(p0 + d) * Q27_AKV + hk) * 256;
    if (src == dst) continue;
    if (cap) {                                         // int8: 256 bytes (16 per lane) and the scale
      global uint *k = (global uint *)KC, *v = (global uint *)VC;
      for (int i = 0; i < 4; i++) {
        k[dst / 4 + i * 16 + lane] = k[src / 4 + i * 16 + lane];
        v[dst / 4 + i * 16 + lane] = v[src / 4 + i * 16 + lane];
      }
      if (lane == 0) {
        global float *ks = (global float *)((global char *)KC + (long)cap * Q27_KVROW),
                     *vs = (global float *)((global char *)VC + (long)cap * Q27_KVROW);
        ks[dst / 256] = ks[src / 256];
        vs[dst / 256] = vs[src / 256];
      }
      continue;
    }
    for (int i = 0; i < 16; i++) {
      KC[dst + i * 16 + lane] = KC[src + i * 16 + lane];
      VC[dst + i * 16 + lane] = VC[src + i * 16 + lane];
    }
  }
}

// EXL3 probe (mul1 codebook, docs/recipes/exl3.md): X * W_q over K-tile slices, W_q decoded on the fly from TR.
KERNEL(16) void exl3_gemv(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                          int kslices) {
  // the tile is read once per subgroup and distributed by shuffle, like the activations of its 16 K rows
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const float c1 = (float)as_half((ushort)0x1EEE), c0 = (float)as_half((ushort)0xC931);
  float acc0[8], acc1[8];
  for (int t = 0; t < 8; t++) acc0[t] = acc1[t] = 0.0f;
  for (int kt = kt0; kt < kt0 + per; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    const uint r0 = tw[lane], r1 = lane + 16 < nw ? tw[lane + 16] : 0u;
    float xr[8];
    for (int t = 0; t < 8; t++) xr[t] = t < T ? (float)X[(long)t * K + kt * 16 + lane] : 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int p = lane * 16 + v;
      int a = (p + 1) * bits - 16;
      a += a < 0 ? R : 0;
      const int wi = a >> 5, wj = wi + 1 == nw ? 0 : wi + 1, sh = a & 31;
      const uint w0a = sub_group_shuffle(r0, wi & 15), w0b = sub_group_shuffle(r1, wi & 15);
      const uint w1a = sub_group_shuffle(r0, wj & 15), w1b = sub_group_shuffle(r1, wj & 15);
      const uint w0 = wi < 16 ? w0a : w0b, w1 = wj < 16 ? w1a : w1b;
      const uint hi = sh ? (w0 << sh) | (w1 >> (32 - sh)) : w0; // the 32 bits starting at bit a
      const uint s = hi >> 16;
      const uint xx = s * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      const float w = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * c1 + c0;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        const float xv = sub_group_shuffle(xr[t], row);
        if (j >> 2) acc1[t] += xv * w; else acc0[t] += xv * w;
      }
    }
  }
  for (int t = 0; t < T && t < 8; t++) {
    const float a0 = acc0[t] + sub_group_shuffle_xor(acc0[t], 1), a1 = acc1[t] + sub_group_shuffle_xor(acc1[t], 1);
    if ((lane & 1) == 0) {
      const long base = ((long)ks * T + t) * N + nt * 16 + lane / 2;
      Y[base] = a0;
      Y[base + 8] = a1;
    }
  }
}

// The prompt attention kernels hold the Q27_GQA query heads of a K/V head (groups of 2), one QK subgroup for each
// half of a group's 128-key block: GQA 6 (the 27B) or 8 (Sage-A1.5) for attn_prompt and attn_prompt8.
#if Q27_GQA == 6 || Q27_GQA == 8
// Prompt attention (FlashAttention-2 on bf16 DPAS), causal, 256-wide heads; fixed key order, so a row's bits ignore T.
// PT (if non-null): the stream's page table; a page is a bf16 cache of Q27_PAGE positions (Q27_PAGE16 bytes).
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt(global const ushort *Q, global const ushort *KC, global const ushort *VC, global ushort *O,
                        int p0, int T, float scale, global const uint *PT) {
  local ushort P[Q27_GQA * 8 * 128];                         // A tiles: (tile of 8 queries, 16 keys), r × 16 + lane
  local float MX[Q27_GQA / 2][2][16], SM[Q27_GQA / 2][2][16], A[Q27_GQA * 8];
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const bool qk = sg < Q27_GQA;
  const int g = sg / 2, kh = sg % 2;
  const int tq = lane % 8, hq = 2 * g + lane / 8;      // the lane's query: token t0 + tq, head Q27_GQA hk + hq
  const int qpos = p0 + t0 + tq;
  const bool qok = t0 + tq < T;
  global const uint *qrow = (global const uint *)(Q + ((long)(t0 + tq) * Q27_AQ + hk * Q27_GQA + hq) * 256);
  const float sl = scale * M_LOG2E_F;
  float8 o[Q27_GQA][2];
  for (int i = 0; i < Q27_GQA; i++) o[i][0] = o[i][1] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += 128) {
    if (qk) {
      const int kb = k0 + kh * 64;                   // the 64 keys of this subgroup
      const bool kin = !PT || kb < nkeys;            // a page past the keys: its keys are all masked
      global const ushort *kpage = KC;               // the K reads' surface: the cache, or the page of keys kb..
      int krows = nkeys, ky = kb;
      if (PT && kin) {
        kpage = (global const ushort *)((global const char *)KC + (long)PT[kb / Q27_PAGE] * Q27_PAGE16);
        krows = min(Q27_PAGE, nkeys - kb);
        ky = 0;
      }
      float8 s[8];
      for (int b = 0; b < 8; b++) s[b] = 0.0f;
      for (int ds = 0; kin && ds < 16; ds++) {
        const int8 qb = qok ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
        __attribute__((opencl_unroll_hint))
        for (int h2 = 0; h2 < 2; h2++) {
          ushort ka[32];                               // 32 keys × 16 dimensions: 4 A tiles of 8 keys
          intel_sub_group_2d_block_read_16b_32r16x1c((global void *)(kpage + hk * 256), 256 * 2, krows, Q27_AKV * 256 * 2,
                                                     (int2)(ds * 16, ky + h2 * 32), ka);
          __attribute__((opencl_unroll_hint))
          for (int b = 0; b < 4; b++) {
            short8 a;
            for (int r = 0; r < 8; r++) a[r] = (short)ka[b * 8 + r];
            s[h2 * 4 + b] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, qb, s[h2 * 4 + b]);
          }
        }
      }
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const float x = kb + b * 8 + r > qpos ? -INFINITY : s[b][r] * sl;
          s[b][r] = x;
          mx = fmax(mx, x);
        }
      MX[g][kh][lane] = mx;
      barrier(CLK_LOCAL_MEM_FENCE);
      const float mn = fmax(m, fmax(mx, MX[g][kh ^ 1][lane]));
      const float alpha = mn == -INFINITY ? 1.0f : native_exp2(m - mn);
      float sum = 0.0f;
      local ushort *pt = P + (hq * 8) * 128 + tq * 16;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const float p = mn == -INFINITY ? 0.0f : native_exp2(s[b][r] - mn);
          sum += p;
          const int key = kh * 64 + b * 8 + r;      // within the block
          pt[(key / 16) * 128 + key % 16] = f2bf(p);
        }
      SM[g][kh][lane] = sum;
      if (kh == 0) A[hq * 8 + tq] = alpha;
      m = mn;
      barrier(CLK_LOCAL_MEM_FENCE);
      l = l * alpha + SM[g][0][lane] + SM[g][1][lane];
    } else {
      barrier(CLK_LOCAL_MEM_FENCE);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < Q27_GQA; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c = 0; c < 8; c++) {
      if (k0 + c * 16 >= nkeys) break;
      const int y0 = k0 + c * 16;
      global const ushort *vpage = VC;               // the V reads' surface: the cache, or the page of keys y0..
      int vrows = nkeys, vy = y0;
      if (PT) {
        const int pg = y0 / Q27_PAGE;
        vpage = (global const ushort *)((global const char *)VC + (long)PT[pg] * Q27_PAGE16);
        vrows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
        vy = y0 % Q27_PAGE;
      }
      short8 pa[Q27_GQA];
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < Q27_GQA; i++) pa[i] = as_short8(intel_sub_group_block_read_us8(P + (i * 8 + c) * 128));
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint8 vw;
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(vpage + hk * 256), 256 * 2, vrows,
                                                             Q27_AKV * 256 * 2, (int2)((sg * 2 + dt) * 16, vy),
                                                             (private uint *)&vw);
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < Q27_GQA; i++) o[i][dt] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[i], as_int8(vw), o[i][dt]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (qk && kh == 0) A[hq * 8 + tq] = 1.0f / l;
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < Q27_GQA; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * Q27_AQ + hk * Q27_GQA + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}

#endif // Q27_GQA == 6 || Q27_GQA == 8

#if Q27_GQA == 6 || Q27_GQA == 8
// Prompt attention over the int8 cache: same partitioning as attn_prompt, int8 DPAS for S and fp16 for O += P*V.
// PT (if non-null): the stream's page table; a page is an int8 cache of Q27_PAGE positions (Q27_PAGE8 bytes).
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt8(global const ushort *Q, global const char *KC, global const char *VC, global ushort *O,
                         int p0, int T, float scale, int cap, global const uint *PT) {
  local ushort P[Q27_GQA * 8 * 128];
  local uint QI[Q27_GQA / 2 * 8 * 8 * 16];                       // (group, 32 dimensions, word) × lane: Q in int8, 4 per word
  local float MX[Q27_GQA / 2][2][16], SM[Q27_GQA / 2][2][16], A[Q27_GQA * 8], QS[Q27_GQA * 8];
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const bool qk = sg < Q27_GQA;
  const int g = sg / 2, kh = sg % 2;
  const int tq = lane % 8, hq = 2 * g + lane / 8;
  const int qpos = p0 + t0 + tq;
  const bool qok = t0 + tq < T;
  global const float *KS = (global const float *)(KC + (long)cap * Q27_KVROW), *VS = (global const float *)(VC + (long)cap * Q27_KVROW);
  if (qk && kh == 0) {                                 // the lane's query, Q in int8
    global const ushort *qrow = Q + ((long)(t0 + tq) * Q27_AQ + hk * Q27_GQA + hq) * 256;
    float mq = 0.0f;
    if (qok)
      for (int e = 0; e < 16; e++) {
        const ushort16 x = vload16(e, qrow);
        for (int b = 0; b < 16; b++) mq = fmax(mq, fabs(bf2f(x[b])));
      }
    const float qs = mq > 0.0f ? mq / 127.0f : 1.0f, iq = 1.0f / qs;
    for (int e = 0; e < 16; e++) {                     // 16 dimensions: 4 words
      const ushort16 x = qok ? vload16(e, qrow) : (ushort16)(0);
      for (int j = 0; j < 4; j++) {
        uint u = 0;
        for (int b = 0; b < 4; b++) u |= ((uint)(uchar)(char)rint(bf2f(x[j * 4 + b]) * iq)) << (8 * b);
        const int w = e * 4 + j;
        QI[((g * 8 + w / 8) * 8 + w % 8) * 16 + lane] = u;
      }
    }
    QS[hq * 8 + tq] = qs;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float qsl = qk ? QS[hq * 8 + tq] * scale * M_LOG2E_F : 0.0f;
  float8 o[Q27_GQA][2];
  for (int i = 0; i < Q27_GQA; i++) o[i][0] = o[i][1] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += 128) {
    if (qk) {
      const int kb = k0 + kh * 64;
      const bool kin = !PT || kb < nkeys;              // a page past the keys: its keys are all masked
      global const char *kpage = KC;                   // the K reads' surface: the cache, or the page of keys kb..
      int krows = nkeys, ky = kb, kbase = 0;
      global const float *ksp = KS, *vsp = VS;
      if (PT && kin) {
        const long pb = (long)PT[kb / Q27_PAGE] * Q27_PAGE8;
        kpage = KC + pb;
        krows = min(Q27_PAGE, nkeys - kb);
        ky = 0;
        kbase = kb;
        ksp = (global const float *)(kpage + (long)Q27_PAGE * Q27_KVROW);
        vsp = (global const float *)(VC + pb + (long)Q27_PAGE * Q27_KVROW);
      }
      int8 si[8];
      for (int b = 0; b < 8; b++) si[b] = 0;
      for (int ds = 0; kin && ds < 8; ds++) {          // steps of 32 dimensions
        const int8 qb = as_int8(intel_sub_group_block_read8(QI + (g * 8 + ds) * 128));
        __attribute__((opencl_unroll_hint))
        for (int h2 = 0; h2 < 2; h2++) {
          ushort ka[32];                               // 32 keys × 32 bytes: 4 A tiles of 8 keys
          intel_sub_group_2d_block_read_16b_32r16x1c((global void *)(kpage + hk * 256), 256, krows, Q27_AKV * 256,
                                                     (int2)(ds * 16, ky + h2 * 32), ka);
          __attribute__((opencl_unroll_hint))
          for (int b = 0; b < 4; b++) {
            short8 a;
            for (int r = 0; r < 8; r++) a[r] = (short)ka[b * 8 + r];
            si[h2 * 4 + b] = intel_sub_group_i8_i8_matrix_mad_k32(a, qb, si[h2 * 4 + b]);
          }
        }
      }
      float ksc[4];                                    // scales of the 64 keys, 4 per lane
      for (int c = 0; c < 4; c++) {
        const int key = kb + c * 16 + lane;
        ksc[c] = key < nkeys ? ksp[(long)(key - kbase) * Q27_AKV + hk] : 0.0f;
      }
      float8 s[8];
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const int kk = b * 8 + r;
          const float ks = sub_group_broadcast(ksc[kk / 16], kk % 16);
          const float x = kb + kk > qpos ? -INFINITY : (float)si[b][r] * ks * qsl;
          s[b][r] = x;
          mx = fmax(mx, x);
        }
      MX[g][kh][lane] = mx;
      barrier(CLK_LOCAL_MEM_FENCE);
      const float mn = fmax(m, fmax(mx, MX[g][kh ^ 1][lane]));
      const float alpha = mn == -INFINITY ? 1.0f : native_exp2(m - mn);
      float vsc[4];
      for (int c = 0; c < 4; c++) {
        const int key = kb + c * 16 + lane;
        vsc[c] = key < nkeys ? vsp[(long)(key - kbase) * Q27_AKV + hk] : 0.0f;
      }
      float sum = 0.0f;
      local ushort *pt = P + (hq * 8) * 128 + tq * 16;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const int kk = b * 8 + r;
          const float p = mn == -INFINITY ? 0.0f : native_exp2(s[b][r] - mn);
          sum += p;
          const int key = kh * 64 + kk;
          pt[(key / 16) * 128 + key % 16] = as_ushort(convert_half(p * sub_group_broadcast(vsc[kk / 16], kk % 16)));
        }
      SM[g][kh][lane] = sum;
      if (kh == 0) A[hq * 8 + tq] = alpha;
      m = mn;
      barrier(CLK_LOCAL_MEM_FENCE);
      l = l * alpha + SM[g][0][lane] + SM[g][1][lane];
    } else {
      barrier(CLK_LOCAL_MEM_FENCE);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < Q27_GQA; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c2 = 0; c2 < 4; c2++) {                   // 32 keys at a time
      if (k0 + c2 * 32 >= nkeys) break;
      const int y0 = k0 + c2 * 32;
      global const char *vpage = VC;                   // the V reads' surface: the cache, or the page of keys y0..
      int vrows = nkeys, vy = y0;
      if (PT) {
        const int pg = y0 / Q27_PAGE;
        vpage = VC + (long)PT[pg] * Q27_PAGE8;
        vrows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
        vy = y0 % Q27_PAGE;
      }
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint w8[8];                                    // lane: dimension 16 (2 sg + dt) + lane; word j: keys 4j … 4j + 3
        intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)(vpage + hk * 256), 256, vrows, Q27_AKV * 256,
                                                            (int2)((sg * 2 + dt) * 16, vy), w8);
        __attribute__((opencl_unroll_hint))
        for (int hc = 0; hc < 2; hc++) {               // 16 keys: 8 fp16 pairs (keys 2j, 2j + 1), int8 → fp16 exact
          uint8 vw;
          __attribute__((opencl_unroll_hint))
          for (int j = 0; j < 4; j++) {
            const half4 h = convert_half4(as_char4(w8[hc * 4 + j]));
            vw[2 * j] = as_uint(h.s01);
            vw[2 * j + 1] = as_uint(h.s23);
          }
          __attribute__((opencl_unroll_hint))
          for (int i = 0; i < Q27_GQA; i++) {
            const short8 pa = as_short8(intel_sub_group_block_read_us8(P + (i * 8 + c2 * 2 + hc) * 128));
            o[i][dt] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(vw), o[i][dt]);
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (qk && kh == 0) A[hq * 8 + tq] = 1.0f / l;
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < Q27_GQA; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * Q27_AQ + hk * Q27_GQA + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}
#endif // Q27_GQA == 6 || Q27_GQA == 8

#if Q27_GQA == 8
// Prompt attention over the int8 cache in the FlashAttention-2 orientation: a sub-group holds 8 query rows of one
// head (the DPAS A operand, Q in int8), the 16 sub-groups of a work-group 16 rows x the 8 heads of a K/V head. K is
// the B operand straight from a transposed 2d block read, the scores come back as C tiles (lane = key, element =
// row), which is the fp16 A layout, so P goes to the PV DPAS without SLM. V is converted to fp16 B tiles once a
// work-group (64 keys in SLM, 16 dims a sub-group). P carries 4096 before the half conversion (an unscaled fp16 P
// underflows at long context), divided out with 1/l at the end. Grid ((T + 15) / 16, Q27_AKV); the 9 arguments of
// attn_prompt8.
#define FA_SCALE 4096.0f
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1)))
kernel void attn_prompt8_fa(global const ushort *Q, global const char *KC, global const char *VC, global ushort *O,
                            int p0, int T, float scale, int cap, global const uint *PT) {
  local uint VH[16 * 4 * 8 * 16];                          // (16 column tiles of 16 dims, 4 key tiles of 16, word) x lane
  const int t0 = get_group_id(0) * 16, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int hq = sg % 8, r0 = t0 + (sg / 8) * 8;           // the sub-group's rows r0 .. r0 + 7 of head Q27_GQA hk + hq
  const int nkeys = p0 + min(t0 + 15, T - 1) + 1;
  const int qlo = p0 + r0;                                 // the first row's position: keys past it need the mask
  global const float *KS = (global const float *)(KC + (long)cap * Q27_KVROW), *VS = (global const float *)(VC + (long)cap * Q27_KVROW);
  short8 qa[8];                                            // Q in int8 as A tiles: qa[ds][r] = dims 32 ds + 2 lane, + 1 of row r
  float qsl[8];
  for (int r = 0; r < 8; r++) {
    const bool ok = r0 + r < T;
    global const ushort *qrow = Q + ((long)(r0 + r) * Q27_AQ + hk * Q27_GQA + hq) * 256;
    float mq = 0.0f;
    if (ok) {
      const ushort16 x = vload16(lane, qrow);
      for (int b = 0; b < 16; b++) mq = fmax(mq, fabs(bf2f(x[b])));
    }
    mq = sub_group_reduce_max(mq);
    const float qs = mq > 0.0f ? mq / 127.0f : 1.0f, iq = 1.0f / qs;
    qsl[r] = qs * scale * M_LOG2E_F;
    __attribute__((opencl_unroll_hint))
    for (int ds = 0; ds < 8; ds++) {
      const ushort2 x = ok ? vload2(ds * 16 + lane, qrow) : (ushort2)(0);
      const uint lo = (uchar)(char)rint(bf2f(x.s0) * iq), hi = (uchar)(char)rint(bf2f(x.s1) * iq);
      qa[ds][r] = (short)(lo | (hi << 8));
    }
  }
  float8 o[16];
  for (int c = 0; c < 16; c++) o[c] = 0.0f;
  float m[8], l[8];                                        // the row max (uniform) and the lane's part of the row sum
  for (int r = 0; r < 8; r++) {
    m[r] = -INFINITY;
    l[r] = 0.0f;
  }
  for (int kb = 0; kb < nkeys; kb += 64) {                 // 64 keys: a page
    global const char *kpage = KC, *vpage = VC;
    global const float *ksp = KS, *vsp = VS;
    int krows = nkeys, ky = kb, kbase = 0;
    if (PT) {
      const long pb = (long)PT[kb / Q27_PAGE] * Q27_PAGE8;
      kpage = KC + pb;
      vpage = VC + pb;
      krows = min(Q27_PAGE, nkeys - kb);
      ky = 0;
      kbase = kb;
      ksp = (global const float *)(kpage + (long)Q27_PAGE * Q27_KVROW);
      vsp = (global const float *)(vpage + (long)Q27_PAGE * Q27_KVROW);
    }
    __attribute__((opencl_unroll_hint))
    for (int h2 = 0; h2 < 2; h2++) {                       // V -> fp16 B tiles: this sub-group's 16 dims, 32 keys a read
      uint w8[8];                                          // lane: dim 16 sg + lane; word j: keys 4j .. 4j + 3
      intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)(vpage + hk * 256), 256, krows, Q27_AKV * 256,
                                                          (int2)(sg * 16, ky + h2 * 32), w8);
      __attribute__((opencl_unroll_hint))
      for (int hc = 0; hc < 2; hc++)
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 4; j++) {
          const half4 h = convert_half4(as_char4(w8[hc * 4 + j]));
          local uint *vt = VH + ((sg * 4 + h2 * 2 + hc) * 8) * 16 + lane;
          vt[(2 * j) * 16] = as_uint(h.s01);
          vt[(2 * j + 1) * 16] = as_uint(h.s23);
        }
    }
    int8 si[4];
    for (int t = 0; t < 4; t++) si[t] = 0;
    __attribute__((opencl_unroll_hint))
    for (int ds = 0; ds < 8; ds++)                         // S = Q K^T in 32-dim steps, K the B operand (lane = key)
      __attribute__((opencl_unroll_hint))
      for (int t = 0; t < 4; t++) {
        uint8 kw;
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c((global void *)(kpage + hk * 256), 256, krows, Q27_AKV * 256,
                                                            (int2)(ds * 8, ky + t * 16), (private uint *)&kw);
        si[t] = intel_sub_group_i8_i8_matrix_mad_k32(qa[ds], as_int8(kw), si[t]);
      }
    float ksc[4], vsc[4];                                  // the lane's key of each tile
    for (int t = 0; t < 4; t++) {
      const int key = kb + t * 16 + lane;
      const bool in = key < nkeys;
      ksc[t] = in ? ksp[(long)(key - kbase) * Q27_AKV + hk] : 0.0f;
      vsc[t] = in ? vsp[(long)(key - kbase) * Q27_AKV + hk] * FA_SCALE : 0.0f;
    }
    float8 s[4];
    const bool masked = kb + 63 > qlo;
    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < 4; t++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++)
        s[t][r] = masked && kb + t * 16 + lane > qlo + r ? -INFINITY : (float)si[t][r] * ksc[t] * qsl[r];
    float mn[8];
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      const float mx = sub_group_reduce_max(fmax(fmax(s[0][r], s[1][r]), fmax(s[2][r], s[3][r])));
      mn[r] = fmax(m[r], mx);
      const float alpha = mn[r] == -INFINITY ? 1.0f : native_exp2(m[r] - mn[r]);
      m[r] = mn[r];
      if (alpha != 1.0f) {                                 // uniform: the row max moved
        l[r] *= alpha;
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < 16; c++) o[c][r] *= alpha;
      }
    }
    short8 pa[4];                                          // P as fp16 A tiles, straight from the C layout
    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < 4; t++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float p = mn[r] == -INFINITY ? 0.0f : native_exp2(s[t][r] - mn[r]);
        l[r] += p;
        pa[t][r] = as_short(convert_half(p * vsc[t]));
      }
    barrier(CLK_LOCAL_MEM_FENCE);                          // VH filled
    __attribute__((opencl_unroll_hint))
    for (int c = 0; c < 16; c++)
      __attribute__((opencl_unroll_hint))
      for (int t = 0; t < 4; t++) {
        const int8 vb = as_int8(intel_sub_group_block_read8(VH + (c * 4 + t) * 128));
        o[c] = intel_sub_group_f16_f16_matrix_mad_k16(pa[t], vb, o[c]);
      }
    barrier(CLK_LOCAL_MEM_FENCE);                          // VH free
  }
  for (int r = 0; r < 8; r++) {
    const float inv = 1.0f / (sub_group_reduce_add(l[r]) * FA_SCALE);
    if (r0 + r >= T) continue;
    global ushort *dst = O + ((long)(r0 + r) * Q27_AQ + hk * Q27_GQA + hq) * 256 + lane;
    __attribute__((opencl_unroll_hint))
    for (int c = 0; c < 16; c++) dst[c * 16] = f2bf(o[c][r] * inv);
  }
}
#endif // Q27_GQA == 8

// Decode attention for long caches: as attn_decode, but a slice holds `span` keys (a launch argument), so the partials
// stay few at any context. Rows: one stream each (MT, POS), or a window at p0 (row t at position p0 + t). Grid (NS,
// Q27_AKV * Q27_QPARTS, rows). cap > 0: int8 cache. PT (if non-null): the stream's page table (with MT, MT[8t + 2]).
KERNEL(16) void attn_decode_span(global const ushort *Q, global const ushort *KC, global const ushort *VC,
                                 global float *PM, global float *PL, global float *PO, int p0, int NS, int span,
                                 float scale, global const ulong *MT, global const int *POS, int cap,
                                 global const uint *PT) {
  const int sp = get_group_id(0), hk = get_group_id(1) / Q27_QPARTS, t = get_group_id(2), lane = get_sub_group_local_id();
  const int qh0 = hk * Q27_GQA + (get_group_id(1) % Q27_QPARTS) * Q27_QPG;
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
    PT = (global const uint *)MT[t * 8 + 2];
  }
  const int last = MT ? POS[t] : p0 + t, j0 = sp * span, j1 = min(j0 + span, last + 1);
  float q[Q27_QPG][16], o[Q27_QPG][16], m[Q27_QPG], l[Q27_QPG];
  for (int a = 0; a < Q27_QPG; a++) {
    for (int i = 0; i < 16; i++) {
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + lane * 16 + i]) * scale; // the lane's 16 dimensions
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  const global ushort *kp = KC, *vp = VC;
  const global float *ks = (const global float *)((const global char *)KC + (long)cap * Q27_KVROW);
  const global float *vs = (const global float *)((const global char *)VC + (long)cap * Q27_KVROW);
  for (int j = j0; j < j1; j++) {
    if (PT && (j == j0 || j % Q27_PAGE == 0)) { // move to the page holding key j
      const long pb = (long)PT[j / Q27_PAGE] * (cap ? Q27_PAGE8 : Q27_PAGE16);
      kp = (const global ushort *)((const global char *)KC + pb);
      vp = (const global ushort *)((const global char *)VC + pb);
      ks = (const global float *)((const global char *)kp + (long)Q27_PAGE * Q27_KVROW);
      vs = (const global float *)((const global char *)vp + (long)Q27_PAGE * Q27_KVROW);
    }
    const long kr = (long)(PT ? j % Q27_PAGE : j) * Q27_AKV + hk; // the cache row of key j
    float kv[16];
    if (cap) {
      const char16 k8 = vload16(lane, (const global char *)kp + kr * 256);
      const float s = ks[kr];
      for (int i = 0; i < 16; i++) kv[i] = (float)k8[i] * s;
    } else {
      const ushort16 kb = vload16(lane, kp + kr * 256);
      for (int i = 0; i < 16; i++) kv[i] = bf2f(kb[i]);
    }
    float sc[Q27_QPG];
    for (int a = 0; a < Q27_QPG; a++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d);
    }
    if (cap) {
      const char16 v8 = vload16(lane, (const global char *)vp + kr * 256);
      const float s = vs[kr];
      for (int i = 0; i < 16; i++) kv[i] = (float)v8[i] * s;
    } else {
      const ushort16 vb = vload16(lane, vp + kr * 256);
      for (int i = 0; i < 16; i++) kv[i] = bf2f(vb[i]);
    }
    for (int a = 0; a < Q27_QPG; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < Q27_QPG; a++) {
    const long base = ((long)t * Q27_AQ + qh0 + a) * NS + sp;
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + lane * 16 + i] = o[a][i];
  }
}

// Merge of attn_decode_span slices: O bf16 [T, Q27_AQ, 256]. One work-group of 16 subgroups per (head, row), each
// subgroup takes every 16th slice, then the work-group adds the 16 partials.
KERNEL(256) void attn_merge_span(global const float *PM, global const float *PL, global const float *PO,
                                 global ushort *O, int p0, int NS, int span, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), g = get_sub_group_id(), lane = get_sub_group_local_id();
  const int used = (POS ? POS[t] : p0 + t) / span + 1;
  const long base = ((long)t * Q27_AQ + hh) * NS;
  local float lm[16], ll[16], lo[16][256];
  float m = -INFINITY;
  for (int s = g; s < used; s += 16) m = fmax(m, PM[base + s]);
  m = sub_group_reduce_max(m);
  if (lane == 0) lm[g] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  m = -INFINITY;
  for (int i = 0; i < 16; i++) m = fmax(m, lm[i]);
  float l = 0.0f, o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int s = g; s < used; s += 16) {
    const float w = exp(PM[base + s] - m);
    l += PL[base + s] * w;
    for (int i = 0; i < 16; i++) o[i] += PO[(base + s) * 256 + i * 16 + lane] * w;
  }
  if (lane == 0) ll[g] = l;
  for (int i = 0; i < 16; i++) lo[g][i * 16 + lane] = o[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int e = get_local_id(0);
  float sl = 0.0f, so = 0.0f;
  for (int i = 0; i < 16; i++) {
    sl += ll[i];
    so += lo[i][e];
  }
  O[((long)t * Q27_AQ + hh) * 256 + e] = f2bf(so / sl);
}

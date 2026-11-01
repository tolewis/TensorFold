// K1 body, included by k1.cl once a variant (k1.cl names the macros): decode attention over int8 pool pages on
// the XMX. Macros in: K1_FN (kernel name), K1_R (rows a group), K1_C (columns, K1_R x Q27_GQA), K1_P (DPAS passes),
// K1_T (keys a tile), K1_G (sub-groups that read K rows: K1_T / 16). Not a translation unit on its own.
//
// The pool is [pool, PAGE, AKV, 256] (Sage's allocPool): page id PT[k / Q27_PAGE], the K row of key k at
// page + (k % Q27_PAGE) * Q27_KVROW + hk * 256, then the page's fp32 scales (Q27_PAGE8 bytes a page).
// One work-group per (split, K/V head, row group). The columns of the DPAS are the group's (row, query head)
// pairs, so each K and V row is read once for the whole group. QK as K1_G sub-groups x 16 keys through 2D block
// reads and i8_i8_k32, the K scale on the int32 score, P in f16 in SLM with the V scale folded in, PV as 16
// sub-groups x 16 dimensions through transform reads and f16_f16_k16. Partials as attn_decode_span (PM, PL, PO),
// merged by attn_merge_span.
KERNEL(256) void K1_FN(global const ushort *Q, global const char *KC, global const char *VC,
                       global float *PM, global float *PL, global float *PO, int p0, int NS, int span,
                       float scale, global const ulong *MT, global const int *POS, int cap,
                       global const uint *PT, global const int *GRP) {
  local uint QI[K1_P * 64 * 16];                   // (pass, 32-dimension step, word, lane): Q in int8, column (row, head)
  local ushort P[K1_G * K1_C * 16];                // (16-key group, column, key): P in f16 with the V scale folded in
  local float QS[K1_C], MX[K1_P][16][16], SM[K1_P][16][16], AL[K1_C];
  const int sp = get_group_id(0), hk = get_group_id(1), g = get_group_id(2);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int t0 = GRP[2 * g], nr = GRP[2 * g + 1], npass = (nr * Q27_GQA + 15) / 16;
  if (MT) {                                        // the rows of a group share a stream
    KC = (global const char *)MT[t0 * 8];
    VC = (global const char *)MT[t0 * 8 + 1];
    PT = (global const uint *)MT[t0 * 8 + 2];
  }
  int lastr[K1_R], max_last = -1;
  for (int r = 0; r < K1_R; r++) {
    lastr[r] = r < nr ? (MT ? POS[t0 + r] : p0 + t0 + r) : -1;
    max_last = max(max_last, lastr[r]);
  }
  const int nkeys = max_last + 1, j0 = sp * span, j1 = min(j0 + span, nkeys);
  const int qh0 = hk * Q27_GQA;
  for (int c = sg; c < K1_C; c += 16) {            // column c: row c / Q27_GQA, head c % Q27_GQA; the lane's 16 dims
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
  float qsl[K1_P];                                 // the lane's column of each pass: p * 16 + lane
  int lastc[K1_P];
  for (int p = 0; p < K1_P; p++) {
    qsl[p] = QS[p * 16 + lane] * scale * M_LOG2E_F;
    lastc[p] = lastr[(p * 16 + lane) / Q27_GQA];
  }
  float8 o[K1_C / 8];                              // lane: dimension 16 sg + lane; o[c / 8]: column c % 8
  for (int q = 0; q < K1_C / 8; q++) o[q] = 0.0f;
  float m[K1_P], l[K1_P];
  for (int p = 0; p < K1_P; p++) {
    m[p] = -INFINITY;
    l[p] = 0.0f;
  }
  if (j0 < j1 && sg < K1_G) {                      // the first tile's K rows
    const int kb = j0 + sg * 16;
    if (kb < j1) {
      global const char *kp = KC + (long)PT[kb / Q27_PAGE] * Q27_PAGE8 + hk * 256;
      const int rows = min(Q27_PAGE, nkeys - (kb / Q27_PAGE) * Q27_PAGE);
      for (int ds = 0; ds < 8; ds++)
        intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)kp, 256, rows, Q27_KVROW, (int2)(ds * 16, kb % Q27_PAGE));
    }
  }
  for (int k0 = j0; k0 < j1; k0 += K1_T) {
    const int kb = k0 + sg * 16, kpg = kb / Q27_PAGE;
    const bool kin = sg < K1_G && kb < j1;
    const long kpb = kin ? (long)PT[kpg] * Q27_PAGE8 : 0;
    const int krows = kin ? min(Q27_PAGE, nkeys - kpg * Q27_PAGE) : 0;
    global const char *kpage = KC + kpb + hk * 256;
    { // the next tile's V rows: this sub-group's 16 keys of it
      const int vb = k0 + K1_T + sg * 16;
      if (sg < K1_G && vb < j1) {
        const int pg = vb / Q27_PAGE;
        global const char *vp = VC + (long)PT[pg] * Q27_PAGE8 + hk * 256;
        const int rows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
        for (int ds = 0; ds < 8; ds++)
          intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)vp, 256, rows, Q27_KVROW, (int2)(ds * 16, vb % Q27_PAGE));
      }
    }
    int8 si[K1_P][2];
    for (int p = 0; p < K1_P; p++) si[p][0] = si[p][1] = 0;
    if (kin) {
      __attribute__((opencl_unroll_hint))
      for (int ds = 0; ds < 8; ds++) {             // steps of 32 dimensions: 16 keys x 32 bytes, 2 A tiles of 8 keys
        ushort ka[16];
        intel_sub_group_2d_block_read_16b_16r16x1c((global void *)kpage, 256, krows, Q27_KVROW, (int2)(ds * 16, kb % Q27_PAGE), ka);
        short8 a[2];
        for (int b = 0; b < 2; b++)
          for (int r = 0; r < 8; r++) a[b][r] = (short)ka[b * 8 + r];
        for (int p = 0; p < npass; p++) {
          const int8 qb = as_int8(intel_sub_group_block_read8(QI + (p * 64 + ds * 8) * 16));
          si[p][0] = intel_sub_group_i8_i8_matrix_mad_k32(a[0], qb, si[p][0]);
          si[p][1] = intel_sub_group_i8_i8_matrix_mad_k32(a[1], qb, si[p][1]);
        }
      }
    }
    const int key = kb + lane;
    const bool kok = kin && key < j1;
    const float ksc = kok ? ((global const float *)(KC + kpb + (long)Q27_PAGE * Q27_KVROW))[(key % Q27_PAGE) * Q27_AKV + hk] : 0.0f;
    const float vsc = kok ? ((global const float *)(VC + kpb + (long)Q27_PAGE * Q27_KVROW))[(key % Q27_PAGE) * Q27_AKV + hk] : 0.0f;
    float s[K1_P][16], mx[K1_P], alpha[K1_P], mn[K1_P];
    for (int p = 0; p < K1_P; p++) {
      mx[p] = -INFINITY;
      if (p < npass) {
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < 16; kk++) {
          const float ks = sub_group_broadcast(ksc, kk);
          const float x = (kin && kb + kk <= lastc[p]) ? (float)si[p][kk / 8][kk % 8] * ks * qsl[p] : -INFINITY;
          s[p][kk] = x;
          mx[p] = fmax(mx[p], x);
        }
      }
      MX[p][sg][lane] = mx[p];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 0; p < K1_P; p++) {
      float tm = -INFINITY;
      for (int i = 0; i < 16; i++) tm = fmax(tm, MX[p][i][lane]);
      mn[p] = fmax(m[p], tm);
      alpha[p] = mn[p] == -INFINITY ? 1.0f : native_exp2(m[p] - mn[p]);
      float sum = 0.0f;
      if (p < npass && sg < K1_G) {
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < 16; kk++) {
          const float pr = mn[p] == -INFINITY ? 0.0f : native_exp2(s[p][kk] - mn[p]);
          sum += pr;
          P[(sg * K1_C + p * 16 + lane) * 16 + kk] = as_ushort(convert_half(pr * sub_group_broadcast(vsc, kk) * K1_PSCALE));
        }
      }
      SM[p][sg][lane] = sum;
      if (sg == 0) AL[p * 16 + lane] = alpha[p];
      m[p] = mn[p];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 0; p < K1_P; p++) {
      float ls = 0.0f;
      for (int i = 0; i < 16; i++) ls += SM[p][i][lane];
      l[p] = l[p] * alpha[p] + ls;
    }
    for (int q = 0; q < K1_C / 8; q++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) o[q][r] *= AL[q * 8 + r];
    { // the next tile's K rows: this sub-group's 16 keys of it
      const int nb = k0 + K1_T + sg * 16;
      if (sg < K1_G && nb < j1) {
        const int pg = nb / Q27_PAGE;
        global const char *np = KC + (long)PT[pg] * Q27_PAGE8 + hk * 256;
        const int rows = min(Q27_PAGE, nkeys - pg * Q27_PAGE);
        for (int ds = 0; ds < 8; ds++)
          intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void *)np, 256, rows, Q27_KVROW, (int2)(ds * 16, nb % Q27_PAGE));
      }
    }
    for (int c2 = 0; c2 < K1_T / 32; c2++) {       // PV, 32 keys a read: lane = dimension 16 sg + lane, word j = keys 4j..
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
        for (int q = 0; q < K1_C / 8; q++) {
          const short8 pa = as_short8(intel_sub_group_block_read_us8(P + (kg * K1_C + q * 8) * 16));
          o[q] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(vw), o[q]);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);                  // P, MX, SM and AL are reused by the next tile
  }
  if (sg == 0)
    for (int p = 0; p < npass; p++) {
      const int c = p * 16 + lane;
      if (c < Q27_GQA * nr) {
        const int row = c / Q27_GQA, head = c % Q27_GQA;
        const long base = ((long)(t0 + row) * Q27_AQ + qh0 + head) * NS + sp;
        PM[base] = m[p];
        PL[base] = l[p];
      }
    }
  for (int q = 0; q < K1_C / 8; q++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      const int c = q * 8 + r;
      if (c < Q27_GQA * nr) {
        const int row = c / Q27_GQA, head = c % Q27_GQA;
        PO[(((long)(t0 + row) * Q27_AQ + qh0 + head) * NS + sp) * 256 + sg * 16 + lane] = o[q][r] * (1.0f / K1_PSCALE);
      }
    }
}

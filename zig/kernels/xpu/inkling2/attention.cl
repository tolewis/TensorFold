// Inkling 2 Mini attention on Intel Xe2: input preparation, the relative-bias table, decode by key chunks and the
// ordered sink and gate merge. The formulas, the bf16 roundings and the layout of the chunk partials follow
// kernels/metal/inkling2/attention.metal; only the lane mapping differs (a sub-group is 16 lanes here).
#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

#define D 128       // head width
#define REL 16      // relative-bias taps a head
#define EXTENT 1024 // distances the bias table covers
#define GROUP 3     // query heads one K/V head serves (24 query heads, 8 K/V heads)
#define BLOCK 16    // keys a sub-group scores at once, one a lane

#define LOG2E 1.4426950408889634f

#define UNROLL __attribute__((opencl_unroll_hint))

// Planted faults for the kernel contract's controls, set with -DINK2_CTL=n at build time; 0 is the shipped kernel.
#ifndef INK2_CTL
#define INK2_CTL 0
#endif

typedef struct {
  int row0, n, p0, pad;
  long off; // elements into the K/V cache buffers
} InkSeg;

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even
static inline float bf_round(float f) { return bf2f(f2bf(f)); }

static inline float clip_factor(float ss, float threshold) { return fmin(threshold / (sqrt(ss) + 1e-6f), 1.0f); }

static inline float sum8(float8 x) {
  return ((((((x.s0 * x.s0 + x.s1 * x.s1) + x.s2 * x.s2) + x.s3 * x.s3) + x.s4 * x.s4) + x.s5 * x.s5) + x.s6 * x.s6) + x.s7 * x.s7;
}

static inline float8 bf2f8(ushort8 h) { return as_float8(convert_uint8(h) << 16); }

static inline float max8abs(float8 x) {
  return fmax(fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))),
              fmax(fmax(fabs(x.s4), fabs(x.s5)), fmax(fabs(x.s6), fabs(x.s7))));
}

// One int8 code: the value over the scale, rounded to nearest even, clamped to [-127, 127]; zero when the scale is zero.
static inline char q8(float v, float scale) {
  const float t = scale > 0.0f ? rint(v / scale) : 0.0f;
  return (char)convert_int_sat(clamp(t, -127.0f, 127.0f));
}

// One sub-group a (row, projection unit). Units 0 .. heads - 1 are query heads (q and the relative taps), then the K
// heads, then the V heads. Independent q, k, v and relative rows keep their first bf16 rounding.
KERNEL(16) void tf_ink2_prep(global const ushort *q_raw, global const ushort *k_raw, global const ushort *v_raw, global const ushort *rel_raw,
                             global const ushort *q_scale, global const ushort *rel_scale, global const int *rpos, global const int *rseg,
                             global const InkSeg *segs, global ushort *q_out, global ushort *rel_out, global ushort *kc, global ushort *vc,
                             uint heads, uint kv_heads, uint ring, uint is_global, float qk_clip, float rel_clip, float floor_pos,
                             float alpha, float eps) {
  const uint row = get_group_id(0), unit = get_group_id(1), lane = get_sub_group_local_id(), pos = (uint)rpos[row];
#if INK2_CTL == 2
  const float tau = 1.0f;
#else
  const float tau = is_global ? 1.0f + alpha * log(fmax((float)(pos + 1) / floor_pos, 1.0f)) : 1.0f;
#endif
  if (unit < heads) {
    const size_t rh = (size_t)row * heads + unit;
    const float8 x = bf2f8(vload8(0, q_raw + rh * D + lane * 8)) * bf2f8(vload8(0, q_scale + unit * D + lane * 8));
    const float f = clip_factor(sub_group_reduce_add(sum8(x)), qk_clip);
    ushort8 o;
    o.s0 = f2bf(x.s0 * f); o.s1 = f2bf(x.s1 * f); o.s2 = f2bf(x.s2 * f); o.s3 = f2bf(x.s3 * f);
    o.s4 = f2bf(x.s4 * f); o.s5 = f2bf(x.s5 * f); o.s6 = f2bf(x.s6 * f); o.s7 = f2bf(x.s7 * f);
    if (is_global) {
      const float8 t = bf2f8(o) * tau;
      o.s0 = f2bf(t.s0); o.s1 = f2bf(t.s1); o.s2 = f2bf(t.s2); o.s3 = f2bf(t.s3);
      o.s4 = f2bf(t.s4); o.s5 = f2bf(t.s5); o.s6 = f2bf(t.s6); o.s7 = f2bf(t.s7);
    }
    vstore8(o, 0, q_out + rh * D + lane * 8);
    const float r = bf2f(rel_raw[rh * REL + lane]) * bf2f(rel_scale[unit * REL + lane]);
    const float rf = clip_factor(sub_group_reduce_add(r * r), rel_clip);
    const ushort once = f2bf(r * rf);
    rel_out[rh * REL + lane] = is_global ? f2bf(bf2f(once) * tau) : once;
    return;
  }
  const bool key = unit < heads + kv_heads;
  const uint h = key ? unit - heads : unit - heads - kv_heads;
#if INK2_CTL == 1
  global const ushort *src = (key ? v_raw : k_raw) + ((size_t)row * kv_heads + h) * D;
#else
  global const ushort *src = (key ? k_raw : v_raw) + ((size_t)row * kv_heads + h) * D;
#endif
  const float8 x = bf2f8(vload8(0, src + lane * 8));
  const float ss = sub_group_reduce_add(sum8(x));
  const float scale = key ? clip_factor(ss, qk_clip) : rsqrt(ss / (float)D + eps);
  const InkSeg s = segs[rseg[row]];
  global ushort *dst = (key ? kc : vc) + s.off + ((size_t)h * ring + pos % ring) * D;
  const float8 y = x * scale;
  ushort8 o;
  o.s0 = f2bf(y.s0); o.s1 = f2bf(y.s1); o.s2 = f2bf(y.s2); o.s3 = f2bf(y.s3);
  o.s4 = f2bf(y.s4); o.s5 = f2bf(y.s5); o.s6 = f2bf(y.s6); o.s7 = f2bf(y.s7);
  vstore8(o, 0, dst + lane * 8);
}

// The int8 K/V write: the rows, the scalings and the first bf16 rounding of tf_ink2_prep, then one signed byte a value and
// one fp32 scale a (position, K/V head). K codes go to 16-position tiles, V codes row-major, scales head-major. The scale
// pointers take no segment offset: `s.off` is zero for every segment.
KERNEL(16) void tf_ink2_prep8(global const ushort *q_raw, global const ushort *k_raw, global const ushort *v_raw, global const ushort *rel_raw,
                              global const ushort *q_scale, global const ushort *rel_scale, global const int *rpos, global const int *rseg,
                              global const InkSeg *segs, global ushort *q_out, global ushort *rel_out, global char *kc, global float *ksc,
                              global char *vc, global float *vsc, uint heads, uint kv_heads, uint ring, uint is_global, float qk_clip,
                              float rel_clip, float floor_pos, float alpha, float eps) {
  const uint row = get_group_id(0), unit = get_group_id(1), lane = get_sub_group_local_id(), pos = (uint)rpos[row];
#if INK2_CTL == 2
  const float tau = 1.0f;
#else
  const float tau = is_global ? 1.0f + alpha * log(fmax((float)(pos + 1) / floor_pos, 1.0f)) : 1.0f;
#endif
  if (unit < heads) {
    const size_t rh = (size_t)row * heads + unit;
    const float8 x = bf2f8(vload8(0, q_raw + rh * D + lane * 8)) * bf2f8(vload8(0, q_scale + unit * D + lane * 8));
    const float f = clip_factor(sub_group_reduce_add(sum8(x)), qk_clip);
    ushort8 o;
    o.s0 = f2bf(x.s0 * f); o.s1 = f2bf(x.s1 * f); o.s2 = f2bf(x.s2 * f); o.s3 = f2bf(x.s3 * f);
    o.s4 = f2bf(x.s4 * f); o.s5 = f2bf(x.s5 * f); o.s6 = f2bf(x.s6 * f); o.s7 = f2bf(x.s7 * f);
    if (is_global) {
      const float8 t = bf2f8(o) * tau;
      o.s0 = f2bf(t.s0); o.s1 = f2bf(t.s1); o.s2 = f2bf(t.s2); o.s3 = f2bf(t.s3);
      o.s4 = f2bf(t.s4); o.s5 = f2bf(t.s5); o.s6 = f2bf(t.s6); o.s7 = f2bf(t.s7);
    }
    vstore8(o, 0, q_out + rh * D + lane * 8);
    const float r = bf2f(rel_raw[rh * REL + lane]) * bf2f(rel_scale[unit * REL + lane]);
    const float rf = clip_factor(sub_group_reduce_add(r * r), rel_clip);
    const ushort once = f2bf(r * rf);
    rel_out[rh * REL + lane] = is_global ? f2bf(bf2f(once) * tau) : once;
    return;
  }
  const bool key = unit < heads + kv_heads;
  const uint h = key ? unit - heads : unit - heads - kv_heads;
#if INK2_CTL == 1
  global const ushort *src = (key ? v_raw : k_raw) + ((size_t)row * kv_heads + h) * D;
#else
  global const ushort *src = (key ? k_raw : v_raw) + ((size_t)row * kv_heads + h) * D;
#endif
  const float8 x = bf2f8(vload8(0, src + lane * 8));
  const float ss = sub_group_reduce_add(sum8(x));
  const float scale = key ? clip_factor(ss, qk_clip) : rsqrt(ss / (float)D + eps);
  const InkSeg s = segs[rseg[row]];
  const float8 y = x * scale;
  ushort8 o;
  o.s0 = f2bf(y.s0); o.s1 = f2bf(y.s1); o.s2 = f2bf(y.s2); o.s3 = f2bf(y.s3);
  o.s4 = f2bf(y.s4); o.s5 = f2bf(y.s5); o.s6 = f2bf(y.s6); o.s7 = f2bf(y.s7);
  // the value the bf16 cache would hold: its largest magnitude over the row, then one scale and one code a value
  const float8 v = bf2f8(o);
  const float amax = sub_group_reduce_max(max8abs(v));
  const float qs = amax > 0.0f ? amax / 127.0f : 0.0f;
  const char8 c = (char8)(q8(v.s0, qs), q8(v.s1, qs), q8(v.s2, qs), q8(v.s3, qs),
                          q8(v.s4, qs), q8(v.s5, qs), q8(v.s6, qs), q8(v.s7, qs));
  const uint at = pos % ring;
  if (key) {
    global char *kt = kc + s.off + (size_t)h * ring * D + (size_t)(at / 16) * 2048;
    vstore4((char4)(c.s0, c.s1, c.s2, c.s3), 0, kt + (size_t)lane * 128 + (size_t)(at % 16) * 4);
    vstore4((char4)(c.s4, c.s5, c.s6, c.s7), 0, kt + (size_t)lane * 128 + 64 + (size_t)(at % 16) * 4);
    if (lane == 0) ksc[(size_t)h * ring + at] = qs;
  } else {
    global char *vt = vc + s.off + (size_t)h * ring * D + (size_t)at * D;
    vstore8(c, 0, vt + lane * 8);
    if (lane == 0) vsc[(size_t)h * ring + at] = qs;
  }
}

// The bf16 relative-bias table: one fp32 accumulation in projection tap order.
KERNEL(256) void tf_ink2_rel_table(global const ushort *rel, global const ushort *proj, global ushort *tab) {
  const uint rh = get_group_id(0), t = get_local_id(0);
  float r[REL];
  for (uint j = 0; j < REL; ++j) r[j] = bf2f(rel[rh * REL + j]);
  for (uint d = t; d < EXTENT; d += 256) {
    float acc = 0.0f;
    for (uint j = 0; j < REL; ++j) {
#if INK2_CTL == 3
      acc += r[j] * bf2f(proj[j * EXTENT + (d + 1) % EXTENT]);
#else
      acc += r[j] * bf2f(proj[j * EXTENT + d]);
#endif
    }
    tab[rh * EXTENT + d] = f2bf(acc);
  }
}

// One sub-group a (row, K/V head, key chunk) scores BLOCK keys at a time, a key a lane, for the GROUP query heads
// that share the K/V head, so K and V are read once. Every P is rounded to bf16 before its denominator add and its
// P.V product, relative to the running maximum after the block. The chunk's (max, sum, P.V) are the partials.
KERNEL(16) void tf_ink2_attn_decode(global const ushort *q, global const ushort *kc, global const ushort *vc, global const ushort *tab,
                                    global const InkSeg *segs, global float *part_o, global float2 *part_ml, global const int *rpos,
                                    global const int *rseg, uint heads, uint kv_heads, uint ring, uint window, uint chunk, uint c0,
                                    uint chunks, float scale) {
  const uint unit = get_group_id(0), row = unit / kv_heads, kvh = unit % kv_heads, lane = get_sub_group_local_id();
  const uint pos = (uint)rpos[row];
  const uint lo = window && pos + 1 > window ? pos + 1 - window : 0;
  const uint first = window ? lo / chunk : c0;
  const uint c = first + get_group_id(1);
  if (c > pos / chunk || (window && c < lo / chunk)) return;
  const uint from = max(lo, c * chunk), to = min(pos + 1, (c + 1) * chunk);
  const InkSeg s = segs[rseg[row]];
  global const ushort *kh = kc + s.off + (size_t)kvh * ring * D;
  global const ushort *vh = vc + s.off + (size_t)kvh * ring * D;
  const size_t rh0 = (size_t)row * heads + kvh * GROUP;
  local float qs[GROUP][D];
  UNROLL
  for (uint g = 0; g < GROUP; ++g)
    UNROLL
    for (uint j = 0; j < D / 16; ++j) qs[g][j * 16 + lane] = bf2f(q[(rh0 + g) * D + j * 16 + lane]);
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);
  float top[GROUP], total[GROUP], acc[GROUP][D / 16];
  UNROLL
  for (uint g = 0; g < GROUP; ++g) {
    top[g] = -INFINITY;
    total[g] = 0.0f;
    UNROLL
    for (uint j = 0; j < D / 16; ++j) acc[g][j] = 0.0f;
  }
  for (uint kp0 = from; kp0 < to; kp0 += BLOCK) {
    const uint n = min((uint)BLOCK, to - kp0);
    const bool live = lane < n;
    const uint kp = kp0 + (live ? lane : 0);
    // every row of the block is requested before any is used, so the latencies overlap; rows past the end read the last
    // key's row and enter as zero
    ushort8 vr[BLOCK];
    UNROLL
    for (uint k = 0; k < BLOCK; ++k) vr[k] = intel_sub_group_block_read_us8(vh + (size_t)((kp0 + min(k, n - 1)) % ring) * D);
    global const ushort *krow = kh + (size_t)(kp % ring) * D;
    float dot[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) dot[g] = 0.0f;
    UNROLL
    for (uint i = 0; i < D / 8; ++i) {
      const float8 k = bf2f8(vload8(0, krow + i * 8));
      UNROLL
      for (uint g = 0; g < GROUP; ++g) {
        const float8 w = vload8(0, &qs[g][i * 8]);
        float a = dot[g];
        a += k.s0 * w.s0; a += k.s1 * w.s1; a += k.s2 * w.s2; a += k.s3 * w.s3;
        a += k.s4 * w.s4; a += k.s5 * w.s5; a += k.s6 * w.s6; a += k.s7 * w.s7;
        dot[g] = a;
      }
    }
    const uint distance = pos - kp;
    float p[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) {
      const float bias = distance < EXTENT ? bf2f(tab[(rh0 + g) * EXTENT + distance]) : 0.0f;
#if INK2_CTL == 4
      const float score = live ? (dot[g] * scale + bias + (kp == from ? 128.0f : 0.0f)) * LOG2E : -INFINITY;
#else
      const float score = live ? (dot[g] * scale + bias) * LOG2E : -INFINITY;
#endif
      const float next = fmax(top[g], sub_group_reduce_max(score));
      const float keep = exp2(top[g] - next);
      p[g] = live ? bf_round(exp2(score - next)) : 0.0f;
      total[g] = total[g] * keep + sub_group_reduce_add(p[g]);
      top[g] = next;
      UNROLL
      for (uint j = 0; j < D / 16; ++j) acc[g][j] *= keep;
    }
    UNROLL
    for (uint k = 0; k < BLOCK; ++k) {
      const float8 v = k < n ? bf2f8(vr[k]) : (float8)(0.0f);
      UNROLL
      for (uint g = 0; g < GROUP; ++g) {
        const float w = sub_group_broadcast(p[g], k);
        acc[g][0] += w * v.s0; acc[g][1] += w * v.s1; acc[g][2] += w * v.s2; acc[g][3] += w * v.s3;
        acc[g][4] += w * v.s4; acc[g][5] += w * v.s5; acc[g][6] += w * v.s6; acc[g][7] += w * v.s7;
      }
    }
  }
  UNROLL
  for (uint g = 0; g < GROUP; ++g) {
    const size_t slot = (rh0 + g) * chunks + get_group_id(1);
    if (lane == 0) part_ml[slot] = (float2)(top[g], total[g]);
    UNROLL
    for (uint j = 0; j < D / 16; ++j) part_o[slot * D + j * 16 + lane] = acc[g][j];
  }
}

// The int8 decode read: the grid, the chunk partials and the merge of tf_ink2_attn_decode. The chunk is walked in 16-key
// tiles, lane `l` serves key `kp0 + l` of the tile and is live when it lies in [from, to). K codes come from the tile: eight
// 64-byte block reads a 32-dim slab, four slabs, all issued before the dot, so each lane holds its own key's 128 codes. The
// dot runs in fp32 in dim order and takes the key's scale after it. V codes are read a row a time with the row's scale
// broadcast. A dead key enters as an exact zero by a select, never by a multiply: a position never written can hold any
// bits, also an inf or a NaN scale.
KERNEL(16) void tf_ink2_attn_decode8(global const ushort *q, global const char *kc, global const float *ksc, global const char *vc,
                                     global const float *vsc, global const ushort *tab, global const InkSeg *segs, global float *part_o,
                                     global float2 *part_ml, global const int *rpos, global const int *rseg, uint heads, uint kv_heads,
                                     uint ring, uint window, uint chunk, uint c0, uint chunks, float scale) {
  const uint unit = get_group_id(0), row = unit / kv_heads, kvh = unit % kv_heads, lane = get_sub_group_local_id();
  const uint pos = (uint)rpos[row];
  const uint lo = window && pos + 1 > window ? pos + 1 - window : 0;
  const uint first = window ? lo / chunk : c0;
  const uint c = first + get_group_id(1);
  if (c > pos / chunk || (window && c < lo / chunk)) return;
  const uint from = max(lo, c * chunk), to = min(pos + 1, (c + 1) * chunk);
  const InkSeg s = segs[rseg[row]];
  global const char *kh = kc + s.off + (size_t)kvh * ring * D;
  global const char *vh = vc + s.off + (size_t)kvh * ring * D;
  global const float *ks = ksc + (size_t)kvh * ring;
  global const float *vs = vsc + (size_t)kvh * ring;
  const size_t rh0 = (size_t)row * heads + kvh * GROUP;
  local float qs[GROUP][D];
  UNROLL
  for (uint g = 0; g < GROUP; ++g)
    UNROLL
    for (uint j = 0; j < D / 16; ++j) qs[g][j * 16 + lane] = bf2f(q[(rh0 + g) * D + j * 16 + lane]);
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);
  float top[GROUP], total[GROUP], acc[GROUP][D / 16];
  UNROLL
  for (uint g = 0; g < GROUP; ++g) {
    top[g] = -INFINITY;
    total[g] = 0.0f;
    UNROLL
    for (uint j = 0; j < D / 16; ++j) acc[g][j] = 0.0f;
  }
  for (uint kp0 = from & ~15u; kp0 < to; kp0 += 16) {
    const uint kp = kp0 + lane;
    const bool live = kp >= from && kp < to;
    const uint at = kp % ring;
    // every row of the tile is requested before any is used, so the latencies overlap; a row past the end enters as zero
    uchar8 vr[16];
    UNROLL
    for (uint k = 0; k < 16; ++k) vr[k] = intel_sub_group_block_read_uc8(vh + (size_t)((kp0 + k) % ring) * D);
    // the tile's K codes: lane `l` reads dims 4g .. 4g + 3 of key kp0 + l from the 64-byte group g of each slab
    global const char *kt = kh + (size_t)((kp0 % ring) / 16) * 2048;
    uint kr[4][8];
    UNROLL
    for (uint j = 0; j < 4; ++j)
      UNROLL
      for (uint gi = 0; gi < 8; ++gi) kr[j][gi] = intel_sub_group_block_read_ui((global const uint *)(kt + (size_t)j * 512 + (size_t)gi * 64));
    float dot[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) dot[g] = 0.0f;
    UNROLL
    for (uint j = 0; j < 4; ++j)
      UNROLL
      for (uint gi = 0; gi < 8; ++gi) {
        const char4 cd = as_char4(kr[j][gi]);
        const uint d0 = 32 * j + 4 * gi;
        UNROLL
        for (uint g = 0; g < GROUP; ++g) {
          float a = dot[g];
          a += qs[g][d0 + 0] * (float)cd.s0; a += qs[g][d0 + 1] * (float)cd.s1;
          a += qs[g][d0 + 2] * (float)cd.s2; a += qs[g][d0 + 3] * (float)cd.s3;
          dot[g] = a;
        }
      }
    const float kscale = ks[at];
    const uint distance = pos - kp;
    float p[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) {
      const float bias = distance < EXTENT ? bf2f(tab[(rh0 + g) * EXTENT + distance]) : 0.0f;
#if INK2_CTL == 4
      const float score = live ? (dot[g] * kscale * scale + bias + (kp == from ? 128.0f : 0.0f)) * LOG2E : -INFINITY;
#else
      const float score = live ? (dot[g] * kscale * scale + bias) * LOG2E : -INFINITY;
#endif
      const float next = fmax(top[g], sub_group_reduce_max(score));
      const float keep = exp2(top[g] - next);
      p[g] = live ? bf_round(exp2(score - next)) : 0.0f;
      total[g] = total[g] * keep + sub_group_reduce_add(p[g]);
      top[g] = next;
      UNROLL
      for (uint j = 0; j < D / 16; ++j) acc[g][j] *= keep;
    }
    // this lane's own key's V scale, broadcast from lane k for the row it read
    const float vown = vs[at];
    UNROLL
    for (uint k = 0; k < 16; ++k) {
      const float vk = sub_group_broadcast(vown, k);
      const uchar8 cd = vr[k];
      const float8 v8 = (float8)((float)cd.s0, (float)cd.s1, (float)cd.s2, (float)cd.s3,
                                 (float)cd.s4, (float)cd.s5, (float)cd.s6, (float)cd.s7) * vk;
      const bool kl = kp0 + k >= from && kp0 + k < to;
      const float8 v = kl ? v8 : (float8)(0.0f);
      UNROLL
      for (uint g = 0; g < GROUP; ++g) {
        const float w = sub_group_broadcast(p[g], k);
        acc[g][0] += w * v.s0; acc[g][1] += w * v.s1; acc[g][2] += w * v.s2; acc[g][3] += w * v.s3;
        acc[g][4] += w * v.s4; acc[g][5] += w * v.s5; acc[g][6] += w * v.s6; acc[g][7] += w * v.s7;
      }
    }
  }
  UNROLL
  for (uint g = 0; g < GROUP; ++g) {
    const size_t slot = (rh0 + g) * chunks + get_group_id(1);
    if (lane == 0) part_ml[slot] = (float2)(top[g], total[g]);
    UNROLL
    for (uint j = 0; j < D / 16; ++j) part_o[slot * D + j * 16 + lane] = acc[g][j];
  }
}

// Folds the chunks and any sub-chunks in order, adds the sink logit, applies the gate and rounds once to bf16.
// Grid (rows * heads). Eight sub-groups, each over one eighth of the chunks and every channel, so each thread folds eight
// channels over eight chunks instead of one channel over sixty-four: the loads of the next channels run while the last
// fold waits. The eight partials are merged in sub-group order, then the sink and the gate as tf_ink2_merge does.
KERNEL(128) void tf_ink2_merge_split(global const float *part_o, global const float2 *part_ml, global const ushort *sink,
                                     global const ushort *gate, global const int *rpos, global ushort *out, uint heads,
                                     uint window, uint chunk, uint c0, uint chunks, uint subs) {
  const uint rh = get_group_id(0), t = get_local_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  const uint row = rh / heads, pos = (uint)rpos[row];
  const uint lo = window && pos + 1 > window ? pos + 1 - window : 0;
  const uint first = lo / chunk, last_c = pos / chunk, n = last_c + 1 - first;
  const uint per = (n + 7) / 8;
  const uint a = first + sg * per, b = min(first + (sg + 1) * per, last_c + 1);
  local float pmax[8][D], psum[8][D], po[8][D];
  float m8[8], l8[8], o8[8];
  for (uint i = 0; i < 8; ++i) { m8[i] = -INFINITY; l8[i] = 0.0f; o8[i] = 0.0f; }
  for (uint c = a; c < b; ++c) {
    float cm, cl, co8[8];
    if (subs > 1) {
      float sm = -INFINITY, sl = 0.0f;
      float so8[8];
      for (uint i = 0; i < 8; ++i) so8[i] = 0.0f;
      const uint lastu = min(subs - 1, (pos - c * chunk) / 1024);
      for (uint u = 0; u <= lastu; ++u) {
        const size_t at = (size_t)rh * chunks + (c - c0) * subs + u;
        const float2 ml = part_ml[at];
        const float mn = fmax(sm, ml.x);
        const float x = exp2(sm - mn), y = exp2(ml.x - mn);
        sl = sl * x + ml.y * y;
        for (uint i = 0; i < 8; ++i) so8[i] = so8[i] * x + part_o[at * D + lane * 8 + i] * y;
        sm = mn;
      }
      cm = sm; cl = sl;
      for (uint i = 0; i < 8; ++i) co8[i] = so8[i];
    } else {
      const size_t at = (size_t)rh * chunks + c - (window ? lo / chunk : c0);
      const float2 ml = part_ml[at];
      cm = ml.x; cl = ml.y;
      for (uint i = 0; i < 8; ++i) co8[i] = part_o[at * D + lane * 8 + i];
    }
    for (uint i = 0; i < 8; ++i) {
      const float mn = fmax(m8[i], cm);
      const float x = m8[i] == -INFINITY ? 0.0f : exp2(m8[i] - mn);
      const float y = cm == -INFINITY ? 0.0f : exp2(cm - mn);
      l8[i] = l8[i] * x + cl * y;
      o8[i] = o8[i] * x + co8[i] * y;
      m8[i] = mn;
    }
  }
  for (uint i = 0; i < 8; ++i) {
    pmax[sg][lane * 8 + i] = m8[i];
    psum[sg][lane * 8 + i] = l8[i];
    po[sg][lane * 8 + i] = o8[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // every thread owns one channel and reads the eight partials of that channel
  float mm = pmax[0][t], ll = psum[0][t], o = po[0][t];
  for (uint k = 1; k < 8; ++k) {
    const float mn = fmax(mm, pmax[k][t]);
    const float x = mm == -INFINITY ? 0.0f : exp2(mm - mn);
    const float y = pmax[k][t] == -INFINITY ? 0.0f : exp2(pmax[k][t] - mn);
    ll = ll * x + psum[k][t] * y;
    o = o * x + po[k][t] * y;
    mm = mn;
  }
#if INK2_CTL == 7
  const float sk = -INFINITY; // control: the sink dropped
#else
  const float sk = bf2f(sink[rh]) * LOG2E;
#endif
  const float mf = fmax(mm, sk), a0 = exp2(mm - mf);
  const float y = o * a0 / (ll * a0 + exp2(sk - mf));
  const float gv = bf2f(gate[(size_t)rh * D + t]);
  out[(size_t)rh * D + t] = f2bf(y * (1.0f / (1.0f + exp(-gv))));
}

KERNEL(128) void tf_ink2_merge(global const float *part_o, global const float2 *part_ml, global const ushort *sink, global const ushort *gate,
                               global const int *rpos, global ushort *out, uint heads, uint window, uint chunk, uint c0, uint chunks,
                               uint subs) {
  const uint rh = get_group_id(0), d = get_local_id(0), row = rh / heads, pos = (uint)rpos[row];
  const uint lo = window && pos + 1 > window ? pos + 1 - window : 0;
  float mm = -INFINITY, ll = 0.0f, o = 0.0f;
  for (uint c = lo / chunk; c <= pos / chunk; ++c) {
    float cm, cl, co;
    if (subs > 1) {
      float sm = -INFINITY, sl = 0.0f, so = 0.0f;
      const uint last = min(subs - 1, (pos - c * chunk) / 1024);
      for (uint u = 0; u <= last; ++u) {
        const size_t at = (size_t)rh * chunks + (c - c0) * subs + u;
        const float2 ml = part_ml[at];
        const float mn = fmax(sm, ml.x);
        const float x = exp2(sm - mn), y = exp2(ml.x - mn);
        sl = sl * x + ml.y * y;
        so = so * x + part_o[at * D + d] * y;
        sm = mn;
      }
      cm = sm; cl = sl; co = so;
    } else {
      const size_t at = (size_t)rh * chunks + c - (window ? lo / chunk : c0);
      const float2 ml = part_ml[at];
      cm = ml.x; cl = ml.y; co = part_o[at * D + d];
    }
    const float mn = fmax(mm, cm), x = exp2(mm - mn), y = exp2(cm - mn);
    ll = ll * x + cl * y;
    o = o * x + co * y;
    mm = mn;
  }
#if INK2_CTL == 5
  const float sk = -INFINITY;
#else
  const float sk = bf2f(sink[rh]) * LOG2E;
#endif
  const float mf = fmax(mm, sk), a0 = exp2(mm - mf);
  const float y = o * a0 / (ll * a0 + exp2(sk - mf));
  const float gv = bf2f(gate[(size_t)rh * D + d]);
  out[(size_t)rh * D + d] = f2bf(y * (1.0f / (1.0f + exp(-gv))));
}

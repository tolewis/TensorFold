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

// A query value of the decode kernels: lane `l` holds dims 16j + l of each of the GROUP heads in private registers, and the
// dot takes dim d from lane d % 16 by a broadcast at a fixed lane (p3-review R9: no local memory, no barrier). The int8
// kernel keeps its local copy: its compiled tile loop has no local load (the compiler holds the values in registers).
#define QV(qr, g, d) sub_group_broadcast((qr)[g][(d) / 16], (d) % 16)

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

// The sliding stage of an int8 cache on the prompt path: the codes and the scales of tf_ink2_prep8, written as one
// bf16(code * scale) a value to the bf16 staging ring of tf_ink2_pf_attn_window. The arguments and the query units are
// those of tf_ink2_prep, then the int8 cache ring. The value is the one tf_ink2_kv_dequant8 makes from the codes in the
// cache ring, so a key has one value in the stage whatever chunk wrote it. A row at position `ring_from` or later also
// writes its codes and scales to the cache ring (`cache_ring` positions, the layout of tf_ink2_prep8), so each row's
// codes are computed one time a visit. The window kernel reads the stage only, and tf_ink2_kv_dequant8 has read the
// history before this launch, so the ring write overwrites no key that the visit still reads. The cache ring takes no
// segment offset (the host refuses one for an int8 cache).
KERNEL(16) void tf_ink2_prep8q(global const ushort *q_raw, global const ushort *k_raw, global const ushort *v_raw, global const ushort *rel_raw,
                               global const ushort *q_scale, global const ushort *rel_scale, global const int *rpos, global const int *rseg,
                               global const InkSeg *segs, global ushort *q_out, global ushort *rel_out, global ushort *kc, global ushort *vc,
                               global char *rkc, global float *rksc, global char *rvc, global float *rvsc, uint heads, uint kv_heads,
                               uint ring, uint is_global, float qk_clip, float rel_clip, float floor_pos, float alpha, float eps,
                               uint cache_ring, uint ring_from) {
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
  global ushort *dst = (key ? kc : vc) + s.off + ((size_t)h * ring + pos % ring) * D;
  ushort8 w;
  w.s0 = f2bf((float)c.s0 * qs); w.s1 = f2bf((float)c.s1 * qs); w.s2 = f2bf((float)c.s2 * qs); w.s3 = f2bf((float)c.s3 * qs);
  w.s4 = f2bf((float)c.s4 * qs); w.s5 = f2bf((float)c.s5 * qs); w.s6 = f2bf((float)c.s6 * qs); w.s7 = f2bf((float)c.s7 * qs);
  vstore8(w, 0, dst + lane * 8);
  if (pos < ring_from) return;
  const uint at = pos % cache_ring;
  if (key) {
    global char *kt = rkc + (size_t)h * cache_ring * D + (size_t)(at / 16) * 2048;
    vstore4((char4)(c.s0, c.s1, c.s2, c.s3), 0, kt + (size_t)lane * 128 + (size_t)(at % 16) * 4);
    vstore4((char4)(c.s4, c.s5, c.s6, c.s7), 0, kt + (size_t)lane * 128 + 64 + (size_t)(at % 16) * 4);
    if (lane == 0) rksc[(size_t)h * cache_ring + at] = qs;
  } else {
    vstore8(c, 0, rvc + ((size_t)h * cache_ring + at) * D + lane * 8);
    if (lane == 0) rvsc[(size_t)h * cache_ring + at] = qs;
  }
}

// Positions from .. from + n - 1 of an int8 ring (the K tiles, V rows and scales of tf_ink2_prep8) to a bf16 ring of
// the tf_ink2_pf_kv_copy layout, one bf16(code * scale) a value. One sub-group a (position, K/V head), 8 dims a lane.
KERNEL(16) void tf_ink2_kv_dequant8(global const char *kc, global const float *ksc, global const char *vc, global const float *vsc,
                                    global ushort *kd, global ushort *vd, uint from, uint src_ring, uint dst_ring, uint n, uint kv_heads) {
  const uint i = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id();
  if (i >= n || h >= kv_heads) return;
  const uint p = from + i, at = p % src_ring;
  const float ks = ksc[(size_t)h * src_ring + at], vs = vsc[(size_t)h * src_ring + at];
  global const char *kt = kc + (size_t)h * src_ring * D + (size_t)(at / 16) * 2048 + (size_t)lane * 128 + (size_t)(at % 16) * 4;
  const char4 k0 = vload4(0, kt), k1 = vload4(0, kt + 64);
  const char8 v = vload8(0, vc + ((size_t)h * src_ring + at) * D + lane * 8);
  const size_t o = ((size_t)h * dst_ring + p % dst_ring) * D + lane * 8;
  ushort8 a, b;
  a.s0 = f2bf((float)k0.s0 * ks); a.s1 = f2bf((float)k0.s1 * ks); a.s2 = f2bf((float)k0.s2 * ks); a.s3 = f2bf((float)k0.s3 * ks);
  a.s4 = f2bf((float)k1.s0 * ks); a.s5 = f2bf((float)k1.s1 * ks); a.s6 = f2bf((float)k1.s2 * ks); a.s7 = f2bf((float)k1.s3 * ks);
  b.s0 = f2bf((float)v.s0 * vs); b.s1 = f2bf((float)v.s1 * vs); b.s2 = f2bf((float)v.s2 * vs); b.s3 = f2bf((float)v.s3 * vs);
  b.s4 = f2bf((float)v.s4 * vs); b.s5 = f2bf((float)v.s5 * vs); b.s6 = f2bf((float)v.s6 * vs); b.s7 = f2bf((float)v.s7 * vs);
  vstore8(a, 0, kd + o);
  vstore8(b, 0, vd + o);
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
  float qr[GROUP][D / 16];
  UNROLL
  for (uint g = 0; g < GROUP; ++g)
    UNROLL
    for (uint j = 0; j < D / 16; ++j) qr[g][j] = bf2f(q[(rh0 + g) * D + j * 16 + lane]);
  float top[GROUP], total[GROUP], acc[GROUP][D / 16];
  UNROLL
  for (uint g = 0; g < GROUP; ++g) {
    top[g] = -INFINITY;
    total[g] = 0.0f;
    UNROLL
    for (uint j = 0; j < D / 16; ++j) acc[g][j] = 0.0f;
  }
  // the block's first ring slot: one modulo a chunk, then add and wrap by a compare (p3-review R8). `from` is not a multiple
  // of 16 in a window's first chunk, so a block can cross the ring end: every slot wraps by its own compare.
  uint b0 = from % ring;
  for (uint kp0 = from; kp0 < to; kp0 += BLOCK) {
    const uint n = min((uint)BLOCK, to - kp0);
    const bool live = lane < n;
    const uint kp = kp0 + (live ? lane : 0);
    // every row of the block is requested before any is used, so the latencies overlap; rows past the end read the last
    // key's row and enter as zero
    ushort8 vr[BLOCK];
    UNROLL
    for (uint k = 0; k < BLOCK; ++k) {
      uint at = b0 + min(k, n - 1);
      at = at >= ring ? at - ring : at;
      vr[k] = intel_sub_group_block_read_us8(vh + (size_t)at * D);
    }
    uint own = b0 + (live ? lane : 0);
    own = own >= ring ? own - ring : own;
    global const ushort *krow = kh + (size_t)own * D;
    b0 += BLOCK;
    b0 = b0 >= ring ? b0 - ring : b0;
    float dot[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) dot[g] = 0.0f;
    UNROLL
    for (uint i = 0; i < D / 8; ++i) {
      const float8 k = bf2f8(vload8(0, krow + i * 8));
      UNROLL
      for (uint g = 0; g < GROUP; ++g) {
        float a = dot[g];
        a += k.s0 * QV(qr, g, i * 8 + 0); a += k.s1 * QV(qr, g, i * 8 + 1); a += k.s2 * QV(qr, g, i * 8 + 2); a += k.s3 * QV(qr, g, i * 8 + 3);
        a += k.s4 * QV(qr, g, i * 8 + 4); a += k.s5 * QV(qr, g, i * 8 + 5); a += k.s6 * QV(qr, g, i * 8 + 6); a += k.s7 * QV(qr, g, i * 8 + 7);
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
// tiles, lane `l` serves key `kp0 + l` of the tile and is live when it lies in [from, to). K codes come from the tile: four
// 32-dim slabs, one block read a slab, all issued before the dot, so each lane holds its own key's 128 codes. The dot runs
// in fp32 in dim order and takes the key's scale after it. V codes are read a row a time with the row's scale broadcast. A
// dead key enters as an exact zero by a select, never by a multiply: a position never written can hold any bits, also an
// inf or a NaN scale.
// kv8-fix Part 6: access-pattern variants and split forms of tf_ink2_attn_decode8, each behind a build switch that is off by
// default. With every switch at 0 the kernel below the #else is the shipped kernel, text unchanged. The variants keep the
// arithmetic and the order of every sum of the shipped kernel; only the order, the width and the count of the loads change.
//   INK2_ATT_AHEAD=n (A1): the loads of tile t + 1 .. t + n are sent before the arithmetic of tile t (n = 1 or 2).
//   INK2_ATT_FEW=1 (A2): V codes in 8 block reads of two rows (uc16) in place of 16 of one row, the K and V scales in one
//     block read each in place of a gather.
//   INK2_ATT_FEW=2 (A4): V codes in 4 block reads of four rows (ui8), the K read's layout: lane l holds dims 4l .. 4l + 3 and
//     64 + 4l .. 64 + 4l + 3 of each row. Each dim keeps its own sum in key order, so only the lane that holds it changes.
//   INK2_ATT_SPLIT (time-only forms, wrong output bits): 1 the loads only (K codes, V codes, scales), folded by XOR into
//     the partials so that no load is removed. 2 the K side only (K codes, K scale, the dot, the running top and sum, no V
//     load). 3 every load and the K side, no V arithmetic (the V codes and scale folded by XOR).
#ifndef INK2_ATT_AHEAD
#define INK2_ATT_AHEAD 0
#endif
#ifndef INK2_ATT_FEW
#define INK2_ATT_FEW 0
#endif
#ifndef INK2_ATT_SPLIT
#define INK2_ATT_SPLIT 0
#endif
#define INK2_ATT_V (INK2_ATT_SPLIT != 2)

#if INK2_ATT_AHEAD || INK2_ATT_FEW || INK2_ATT_SPLIT
typedef struct {
  uint kr[4][8];
#if INK2_ATT_FEW == 2
  uint vw[4][8]; // read j: rows 4j .. 4j + 3, word 2r + h: row 4j + r, dims 64h + 4l .. 64h + 4l + 3
#elif INK2_ATT_FEW == 1
  uchar16 vr[8]; // read j: row 2j in .s0 .. .s7, row 2j + 1 in .s8 .. .sf, lane l dims 16i + l
#else
  uchar8 vr[16];
#endif
  float ksc, vsc; // this lane's key's scales
} InkTile;

// The loads of the tile at ring slot t0, in the order of the shipped kernel: V rows, then K slabs, then the scales.
static inline void ink_tile_load(InkTile *t, global const char *kh, global const char *vh, global const float *ks,
                                 global const float *vs, uint t0) {
#if INK2_ATT_V
#if INK2_ATT_FEW == 2
  UNROLL
  for (uint j = 0; j < 4; ++j) {
    const uint8 x = intel_sub_group_block_read_ui8((global const uint *)(vh + (size_t)(t0 + 4 * j) * D));
    t->vw[j][0] = x.s0; t->vw[j][1] = x.s1; t->vw[j][2] = x.s2; t->vw[j][3] = x.s3;
    t->vw[j][4] = x.s4; t->vw[j][5] = x.s5; t->vw[j][6] = x.s6; t->vw[j][7] = x.s7;
  }
#elif INK2_ATT_FEW == 1
  UNROLL
  for (uint j = 0; j < 8; ++j) t->vr[j] = intel_sub_group_block_read_uc16((global const uchar *)(vh + (size_t)(t0 + 2 * j) * D));
#else
  UNROLL
  for (uint k = 0; k < 16; ++k) t->vr[k] = intel_sub_group_block_read_uc8(vh + (size_t)(t0 + k) * D);
#endif
#endif
  global const char *kt = kh + (size_t)(t0 / 16) * 2048;
  UNROLL
  for (uint j = 0; j < 4; ++j) {
    const uint8 x = intel_sub_group_block_read_ui8((global const uint *)(kt + (size_t)j * 512));
    t->kr[j][0] = x.s0; t->kr[j][1] = x.s1; t->kr[j][2] = x.s2; t->kr[j][3] = x.s3;
    t->kr[j][4] = x.s4; t->kr[j][5] = x.s5; t->kr[j][6] = x.s6; t->kr[j][7] = x.s7;
  }
#if INK2_ATT_FEW
  t->ksc = as_float(intel_sub_group_block_read((global const uint *)(ks + t0)));
#if INK2_ATT_V
  t->vsc = as_float(intel_sub_group_block_read((global const uint *)(vs + t0)));
#endif
#else
  const uint lane = get_sub_group_local_id();
  t->ksc = ks[t0 + lane];
#if INK2_ATT_V
  t->vsc = vs[t0 + lane];
#endif
#endif
}

// Code `i` of V row `k` of the tile for this lane, as the shipped kernel reads it (a signed code), dims of acc slot i.
#if INK2_ATT_FEW == 2
#define INK_VROW(T, k) (convert_float8((char8)(as_char4((T).vw[(k) / 4][2 * ((k) % 4)]), as_char4((T).vw[(k) / 4][2 * ((k) % 4) + 1]))))
#elif INK2_ATT_FEW == 1
#define INK_VROW(T, k) (convert_float8(as_char8((k) % 2 ? (T).vr[(k) / 2].s89abcdef : (T).vr[(k) / 2].s01234567)))
#else
#define INK_VROW(T, k) (convert_float8(as_char8((T).vr[k])))
#endif

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
#if INK2_ATT_SPLIT == 1 || INK2_ATT_SPLIT == 3
  uint fold = 0;
#endif
  const uint kfirst = from & ~15u, tiles = (to - kfirst + 15) / 16;
  // buf[0] holds the tile in work, buf[1 .. AHEAD] the next tiles, already requested. `tl` is the ring slot of the next
  // tile to request, `asked` the count requested.
  InkTile buf[INK2_ATT_AHEAD + 1];
  uint tl = kfirst % ring, at0 = tl, asked = 0;
  UNROLL
  for (uint i = 0; i < INK2_ATT_AHEAD; ++i)
    if (asked < tiles) {
      ink_tile_load(&buf[i], kh, vh, ks, vs, tl);
      tl += 16;
      if (tl == ring) tl = 0;
      ++asked;
    }
  for (uint ti = 0; ti < tiles; ++ti) {
    if (asked < tiles) {
      ink_tile_load(&buf[INK2_ATT_AHEAD], kh, vh, ks, vs, tl);
      tl += 16;
      if (tl == ring) tl = 0;
      ++asked;
    }
    const uint kp0 = kfirst + 16 * ti;
    const uint kp = kp0 + lane;
    const bool live = kp >= from && kp < to;
#if INK2_ATT_SPLIT == 1
    {
      uint x = as_uint(buf[0].ksc) ^ as_uint(buf[0].vsc);
      UNROLL
      for (uint j = 0; j < 4; ++j)
        UNROLL
        for (uint i = 0; i < 8; ++i) x ^= buf[0].kr[j][i];
#if INK2_ATT_FEW == 2
      UNROLL
      for (uint j = 0; j < 4; ++j)
        UNROLL
        for (uint i = 0; i < 8; ++i) x ^= buf[0].vw[j][i];
#elif INK2_ATT_FEW == 1
      UNROLL
      for (uint j = 0; j < 8; ++j) { const uint4 w = as_uint4(buf[0].vr[j]); x ^= w.s0 ^ w.s1 ^ w.s2 ^ w.s3; }
#else
      UNROLL
      for (uint k = 0; k < 16; ++k) { const uint2 w = as_uint2(buf[0].vr[k]); x ^= w.s0 ^ w.s1; }
#endif
      fold ^= x;
    }
#else
    float dot[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) dot[g] = 0.0f;
    UNROLL
    for (uint j = 0; j < 4; ++j)
      UNROLL
      for (uint gi = 0; gi < 8; ++gi) {
        const char4 cd = as_char4(buf[0].kr[j][gi]);
        const uint d0 = 32 * j + 4 * gi;
        UNROLL
        for (uint g = 0; g < GROUP; ++g) {
          float a = dot[g];
          a += qs[g][d0 + 0] * (float)cd.s0; a += qs[g][d0 + 1] * (float)cd.s1;
          a += qs[g][d0 + 2] * (float)cd.s2; a += qs[g][d0 + 3] * (float)cd.s3;
          dot[g] = a;
        }
      }
    const float kscale = buf[0].ksc;
    const uint distance = pos - kp;
    float p[GROUP];
    UNROLL
    for (uint g = 0; g < GROUP; ++g) {
      const float bias = distance < EXTENT ? bf2f(tab[(rh0 + g) * EXTENT + distance]) : 0.0f;
      const float score = live ? (dot[g] * kscale * scale + bias) * LOG2E : -INFINITY;
      const float next = fmax(top[g], sub_group_reduce_max(score));
      const float keep = exp2(top[g] - next);
      p[g] = live ? bf_round(exp2(score - next)) : 0.0f;
      total[g] = total[g] * keep + sub_group_reduce_add(p[g]);
      top[g] = next;
#if INK2_ATT_SPLIT == 0
      UNROLL
      for (uint j = 0; j < D / 16; ++j) acc[g][j] *= keep;
#endif
    }
#if INK2_ATT_SPLIT == 3
    {
      uint x = as_uint(buf[0].vsc);
#if INK2_ATT_FEW == 2
      UNROLL
      for (uint j = 0; j < 4; ++j)
        UNROLL
        for (uint i = 0; i < 8; ++i) x ^= buf[0].vw[j][i];
#elif INK2_ATT_FEW == 1
      UNROLL
      for (uint j = 0; j < 8; ++j) { const uint4 w = as_uint4(buf[0].vr[j]); x ^= w.s0 ^ w.s1 ^ w.s2 ^ w.s3; }
#else
      UNROLL
      for (uint k = 0; k < 16; ++k) { const uint2 w = as_uint2(buf[0].vr[k]); x ^= w.s0 ^ w.s1; }
#endif
      fold ^= x;
    }
#elif INK2_ATT_SPLIT == 0
    const float vown = buf[0].vsc;
    UNROLL
    for (uint k = 0; k < 16; ++k) {
      const float vk = sub_group_broadcast(vown, k);
      const float8 v8 = INK_VROW(buf[0], k) * vk;
      const bool kl = kp0 + k >= from && kp0 + k < to;
      const float8 v = kl ? v8 : (float8)(0.0f);
      UNROLL
      for (uint g = 0; g < GROUP; ++g) {
        const float w = sub_group_broadcast(p[g], k);
        acc[g][0] += w * v.s0; acc[g][1] += w * v.s1; acc[g][2] += w * v.s2; acc[g][3] += w * v.s3;
        acc[g][4] += w * v.s4; acc[g][5] += w * v.s5; acc[g][6] += w * v.s6; acc[g][7] += w * v.s7;
      }
    }
#endif
#endif
    UNROLL
    for (uint i = 0; i < INK2_ATT_AHEAD; ++i) buf[i] = buf[i + 1];
  }
#if INK2_ATT_SPLIT == 1 || INK2_ATT_SPLIT == 3
  acc[0][0] += as_float(fold & 0x007fffffu); // keeps every load: the partials depend on each loaded word
#endif
  UNROLL
  for (uint g = 0; g < GROUP; ++g) {
    const size_t slot = (rh0 + g) * chunks + get_group_id(1);
    if (lane == 0) part_ml[slot] = (float2)(top[g], total[g]);
#if INK2_ATT_FEW == 2
    // lane l holds dims 4l .. 4l + 3 in acc slots 0 .. 3 and 64 + 4l .. 64 + 4l + 3 in slots 4 .. 7
    UNROLL
    for (uint j = 0; j < 4; ++j) {
      part_o[slot * D + 4 * lane + j] = acc[g][j];
      part_o[slot * D + 64 + 4 * lane + j] = acc[g][4 + j];
    }
#else
    UNROLL
    for (uint j = 0; j < D / 16; ++j) part_o[slot * D + j * 16 + lane] = acc[g][j];
#endif
  }
}
#else
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
  // the tile's first ring slot: one modulo a chunk, then add and wrap by a compare. `ring` is a multiple of 16 and kp0 is
  // too, so a tile never crosses the ring end and slot `t0 + k` is `(kp0 + k) % ring` (p3-review R8)
  uint t0 = (from & ~15u) % ring;
  for (uint kp0 = from & ~15u; kp0 < to; kp0 += 16) {
    const uint kp = kp0 + lane;
    const bool live = kp >= from && kp < to;
    const uint at = t0 + lane;
    // every row of the tile is requested before any is used, so the latencies overlap; a row past the end enters as zero
    uchar8 vr[16];
    UNROLL
    for (uint k = 0; k < 16; ++k) vr[k] = intel_sub_group_block_read_uc8(vh + (size_t)(t0 + k) * D);
    // the tile's K codes: four 32-dim slabs, one block read a slab, all issued before the dot. Lane `l` gets dims
    // 32j .. 32j + 31 of its own key: word i of the read sits at byte 64i + 4l, which is dim group 8j + i at position l.
    global const char *kt = kh + (size_t)(t0 / 16) * 2048;
    t0 += 16;
    if (t0 == ring) t0 = 0;
    uint kr[4][8];
    UNROLL
    for (uint j = 0; j < 4; ++j) {
      const uint8 x = intel_sub_group_block_read_ui8((global const uint *)(kt + (size_t)j * 512));
      kr[j][0] = x.s0; kr[j][1] = x.s1; kr[j][2] = x.s2; kr[j][3] = x.s3;
      kr[j][4] = x.s4; kr[j][5] = x.s5; kr[j][6] = x.s6; kr[j][7] = x.s7;
    }
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
      // the block read returns unsigned bytes: the codes are signed, so they are converted as char, not as uchar
      const float8 v8 = convert_float8(as_char8(vr[k])) * vk;
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
#endif

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
  local float xs[128], ys[128], fin[2]; // the fold factors of the subs == 1 path (OpenCL wants them at kernel scope)
  float mm = -INFINITY, ll = 0.0f, o = 0.0f;
  if (subs == 1) {
    // p3-review R15: sub-group 0 makes the fold factors of each chunk once, x = exp2(m_before - m) and y = exp2(cm - m)
    // with the running maximum m, a chunk a lane, and the denominator ll in chunk order. Every channel then folds its own
    // partial in chunk order with those factors: the values and the order of the serial fold of every thread before.
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const uint last = pos / chunk, slot0 = window ? lo / chunk : c0;
    const size_t at0 = (size_t)rh * chunks;
    for (uint tile = lo / chunk; tile <= last; tile += 128) {
      const uint n = min(128u, last + 1 - tile);
      if (sg == 0) {
        for (uint u = 0; u < n; u += 16) {
          const uint k = u + lane;
          const bool live = k < n;
          const float2 ml = live ? part_ml[at0 + tile + k - slot0] : (float2)(-INFINITY, 0.0f);
          const float mn = fmax(mm, sub_group_scan_inclusive_max(ml.x));
          const float before = intel_sub_group_shuffle(mn, lane == 0 ? 0 : lane - 1);
          const float mb = lane == 0 ? mm : before;
          const float x = exp2(mb - mn), y = exp2(ml.x - mn);
          const float t = ml.y * y;
          if (live) { xs[k] = x; ys[k] = y; }
          const uint m = min(16u, n - u);
          for (uint j = 0; j < m; ++j) ll = ll * sub_group_broadcast(x, j) + sub_group_broadcast(t, j);
          mm = sub_group_broadcast(mn, m - 1);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      __attribute__((opencl_unroll_hint(8)))
      for (uint k = 0; k < n; ++k) o = o * xs[k] + part_o[(at0 + tile + k - slot0) * D + d] * ys[k];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (d == 0) { fin[0] = mm; fin[1] = ll; }
    barrier(CLK_LOCAL_MEM_FENCE);
    mm = fin[0];
    ll = fin[1];
  } else for (uint c = lo / chunk; c <= pos / chunk; ++c) {
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

// fused_bench.cl: the Part 1 benches of lane fused-visit (Recipe lanes/inkspeed/fused-visit/DESIGN.md section 8).
// Resident-kernel pieces on one B70: phase-change mechanisms (M-a, M-b, M-c, M-e), the native work-group barrier, an
// empty resident kernel for the launch cost, a Q8 matvec phase with today's Q8_DOT, a chain of dependent Q8 phases with
// an L2 prefetch between arrive and wait (M-d), and an L2 survival probe. No kernel here sets reqd_work_group_size: the
// host sets the group size, so one kernel serves every grid shape. The image builds at 256 GRF and at 128 GRF.
//
// The sync buffer (uint): word 0 the grid counter, word 16 the generation word, then one flag a work-group, 16 uints
// (one 64-byte line) apart from word 32. Round numbers grow from 1 and never reset inside a launch. The host zeroes the
// buffer before each launch. fault[0] is set when a spin reaches its cap, fault[1] counts failed visibility checks,
// fault[2] holds the round of the fault.

#include "dense_q8.cl" // Q8_DOT and today's tf_ink2_dense_q8_1, the same source and options as the dense_q8 image

#include "fused_sync.h"

// B2 and B3: `rounds` phase changes. With work_u4 > 0 each group first reads work_u4 uint4 (64 KB at 4,096) of its
// own window, a new window each round. The visibility check: the last work-item of the group (lane 0 at one sub-group)
// stores the round before it arrives and reads the next group's word after the wait.
FB_SGA kernel void fb_phase(global uint *sync, global uint *stamp, global uint *fault, global const uint4 *data,
                            global uint *sink, uint rounds, uint spin_cap, uint mech, uint work_u4) {
  local uint lf;
  const uint G = get_num_groups(0), g = get_group_id(0), L = get_local_size(0), lid = get_local_id(0);
  const uint checker = L == 16u ? 0u : L - 1u;
  const uint nwin = work_u4 == 0u ? 1u : (1u << 26) / work_u4; // the data buffer is 1 GiB, 2^26 uint4
  uint acc = 0u;
  fb_enter(mech, sync);
  fb_enter(mech, sync);
  fb_enter(mech, sync);
  fb_enter(mech, sync);
  fb_enter(mech, sync);
  fb_enter(mech, sync);
  fb_enter(mech, sync);
  for (uint r = 1u; r <= rounds; ++r) {
    if (work_u4 != 0u) {
      const ulong base = (ulong)((r * G + g) % nwin) * work_u4;
      __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
        const uint4 v = data[base + i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }
    if (mech != FB_NONE && lid == checker) stamp[g * FB_LINE] = r;
    fb_arrive(mech, sync, r);
    if (fb_wait(mech, sync, fault, r, spin_cap, &lf) != 0u) {
      return;
    }
    if (mech != FB_NONE && lid == checker) {
      const uint v = stamp[((g + 1u) % G) * FB_LINE];
      if (v < r) atomic_inc(fault + 1);
    }
  }
  if (acc == 0x9e3779b9u) sink[g] = acc;
}

// B1: `rounds` native barriers. mode 0 local fence, 1 local and global fence, 2 a device-scope acq_rel fence then a
// local barrier, 3 the device-scope fence alone. Two local buffers alternate, so one barrier a round is enough.
FB_SGA kernel void fb_wgbar(global uint *sink, uint rounds, uint mode) {
  local uint s[2048];
  const uint lid = get_local_id(0), L = get_local_size(0);
  uint acc = lid;
  for (uint r = 0u; r < rounds; ++r) {
    if (mode == 3u) {
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
      acc += r;
      continue;
    }
    local uint *b = s + (r & 1u) * 1024u;
    b[lid] = acc;
    if (mode == 0u) work_group_barrier(CLK_LOCAL_MEM_FENCE);
    else if (mode == 1u) work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    else {
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device);
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
    acc += b[(lid + 1u) % L];
  }
  if (acc == 0x9e3779b9u) sink[get_group_id(0)] = acc;
}

// B6 and B7: a resident kernel with no work, 64 KB of local memory and 5 arguments.
FB_SGA kernel void fb_empty(global uint *a, global uint *b, global uint *c, uint x, uint y) {
  local uint slm[16384];
  const uint lid = get_local_id(0);
  slm[lid] = x;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (slm[16383u - lid] == y) a[get_group_id(0)] = 1u;
}

// One column j of a Q8 phase: today's Q8_DOT (nine planes in flight at k = 2,304), the sub-group sum, one bf16 store.
static inline ushort fb_q8_col(global const ushort *x, global const uchar *codes, global const half *scales, uint j, uint n) {
  float acc[1];
  Q8_DOT(1, x, codes, scales, j, n, FB_ROW)
  return Q8_STORE(sub_group_reduce_add(acc[0]));
}

// B4: `reps` Q8 matvec phases of n columns, each on its own weight window (win_stride bytes apart, nwin windows),
  fb_enter(mech, sync);
// columns over the sub-groups of the grid in turn, one phase change (mech) after each phase.
  fb_enter(mech, sync);
FB_SGA kernel void fb_q8(global const uchar *w, global const ushort *x, global ushort *y, global uint *sync,
  fb_enter(mech, sync);
                         global uint *fault, uint n, uint reps, uint nwin, uint win_stride, uint spin_cap, uint mech) {
  fb_enter(mech, sync);
  local uint lf;
  fb_enter(mech, sync);
  const uint T = get_num_groups(0) * get_num_sub_groups();
  fb_enter(mech, sync);
  const uint gs = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
  fb_enter(mech, sync);
  for (uint r = 0u; r < reps; ++r) {
    global const uchar *codes = w + (ulong)(r % nwin) * win_stride;
    global const half *scales = (global const half *)(codes + (ulong)n * FB_ROW);
    for (uint j = gs; j < n; j += T) {
      const ushort v = fb_q8_col(x, codes, scales, j, n);
      if (get_sub_group_local_id() == 0u) y[j] = v;
    }
    fb_arrive(mech, sync, r + 1u);
    if (fb_wait(mech, sync, fault, r + 1u, spin_cap, &lf) != 0u) {
      return;
    }
  }
}

// B5: `passes` x `phases` dependent Q8 phases of n columns. Phase i reads the 2,304 values the phase before wrote
  fb_enter(mech, sync);
// (columns 0 to 2,303 of a phase write the next x, the others write y). Between arrive and wait every sub-group
// prefetches pf_lines 64-byte lines of its first columns of the next phase into L2 (M-d). A pass covers phases
  fb_enter(mech, sync);
// windows of win_stride bytes. Pass p starts at w + (p % 24) x pass_stride.
FB_SGA kernel void fb_stream(global const uchar *w, global ushort *xbuf, global ushort *y, global uint *sync,
  fb_enter(mech, sync);
                             global uint *fault, uint n, uint phases, uint passes, uint win_stride, uint spin_cap,
                             uint mech, uint pf_lines) {
  fb_enter(mech, sync);
  local uint lf;
  const uint T = get_num_groups(0) * get_num_sub_groups();
  fb_enter(mech, sync);
  const uint gs = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  fb_enter(mech, sync);
  const ulong pass_stride = (ulong)phases * win_stride;
  const uint total = phases * passes;
  fb_enter(mech, sync);
  for (uint i = 0u; i < total; ++i) {
    global const uchar *codes = w + (ulong)((i / phases) % 24u) * pass_stride + (ulong)(i % phases) * win_stride;
    global const half *scales = (global const half *)(codes + (ulong)n * FB_ROW);
    global const ushort *x = xbuf + (i & 1u) * FB_ROW;
    global ushort *xo = xbuf + ((i + 1u) & 1u) * FB_ROW;
    for (uint j = gs; j < n; j += T) {
      const ushort v = fb_q8_col(x, codes, scales, j, n);
      if (lane == 0u) {
        if (j < FB_ROW) xo[j] = v;
        else y[j] = v;
      }
    }
    if (i + 1u == total) break;
    fb_arrive(mech, sync, i + 1u);
    if (pf_lines != 0u) {
      const uint k = i + 1u;
      global const uchar *nc = w + (ulong)((k / phases) % 24u) * pass_stride + (ulong)(k % phases) * win_stride;
      for (uint l = lane; l < pf_lines; l += 16u) {
        const uint col = gs + (l / 36u) * T;
        if (col < n) prefetch(nc + (ulong)col * FB_ROW + (l % 36u) * 64u, 64);
      }
    }
    if (fb_wait(mech, sync, fault, i + 1u, spin_cap, &lf) != 0u) {
  fb_enter(FB_A, sync);
      return;
    }
  }
  fb_enter(FB_A, sync);
}

// B8: each round prefetches (pf_on) block A_r of 2,400,000 bytes into L2, waits, streams stream_bytes of region B,
  fb_enter(FB_A, sync);
// waits, reads A_r (read_on), waits. A_r lies in the first 256 MiB of w, region B in the next 768 MiB.
#define FB_PF_BLOCK 2400000u
#define FB_PF_STRIDE 2457600u
  fb_enter(FB_A, sync);
FB_SGA kernel void fb_l2keep(global const uchar *w, global uint *sink, global uint *sync, global uint *fault,
                             uint rounds, uint pf_on, uint read_on, uint stream_bytes, uint spin_cap) {
  local uint lf;
  fb_enter(FB_A, sync);
  const uint T = get_num_groups(0) * get_num_sub_groups();
  const uint gs = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  fb_enter(FB_A, sync);
  const uint b_stride = (stream_bytes + 4095u) & ~4095u;
  const uint nb = (768u << 20) / b_stride;
  uint acc = 0u;
  fb_enter(FB_A, sync);
  for (uint r = 0u; r < rounds; ++r) {
    global const uchar *a = w + (ulong)(r % 100u) * FB_PF_STRIDE;
    global const uint4 *bq = (global const uint4 *)(w + (256ul << 20) + (ulong)(r % nb) * b_stride);
    if (pf_on != 0u)
      for (uint l = gs * 16u + lane; l < FB_PF_BLOCK / 64u; l += T * 16u) prefetch(a + (ulong)l * 64u, 64);
    fb_arrive(FB_A, sync, 3u * r + 1u);
    if (fb_wait(FB_A, sync, fault, 3u * r + 1u, spin_cap, &lf) != 0u) return;
    __attribute__((opencl_unroll_hint(8))) for (uint i = gs * 16u + lane; i < stream_bytes / 16u; i += T * 16u) {
      const uint4 v = bq[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    fb_arrive(FB_A, sync, 3u * r + 2u);
    if (fb_wait(FB_A, sync, fault, 3u * r + 2u, spin_cap, &lf) != 0u) return;
    if (read_on != 0u) {
      global const uint4 *aq = (global const uint4 *)a;
      __attribute__((opencl_unroll_hint(8))) for (uint i = gs * 16u + lane; i < FB_PF_BLOCK / 16u; i += T * 16u) {
        const uint4 v = aq[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }
    fb_arrive(FB_A, sync, 3u * r + 3u);
    if (fb_wait(FB_A, sync, fault, 3u * r + 3u, spin_cap, &lf) != 0u) return;
  }
  if (acc == 0x9e3779b9u) sink[gs] = acc;
}

// ---------------------------------------------------------------- B9: the NVFP4 decode, today's against the new one
// The new decode (DESIGN section 4d, briefs/e1-nvfp4-expert-rate.md): a block computes A = bf16((t(1.0) * k1) * phi26)
// and B = bf16((t(1.5) * k1) * phi26) the reference way, then a value is bits(A or B) plus (e << 7), the nibble's sign
// on top. A block whose A or B exponent field lies outside 3 to 252 takes today's nv_frag (the guard).
#define INK2_ONLY 0 // the helpers alone, no expert kernel
#include "experts.cl"

#define FB_NV_T(c) as_float((uint)(c) << 22) // the e2m1 magnitude code under the FP32 2^-126 offset, as INK2_NIBBLE_T

static inline uint fb_nv_val(uint nib, uint a, uint b) {
  const uint m = nib & 7u;
  const uint base = ((m & 1u) != 0u && m != 1u) ? b : a;
  const uint mag = m == 0u ? (a & 0x8000u) : base + ((m >> 1) << 7) - 128u;
  return (mag ^ ((nib & 8u) << 12)) & 0xffffu;
}

static inline uint fb_nv_pair(uint w, uint i, uint a, uint b) {
  return fb_nv_val(w >> (4u * i), a, b) | (fb_nv_val(w >> (4u * i + 4u), a, b) << 16);
}

// The new decode of one 16-column block; *guard counts the blocks that took today's nv_frag.
static inline int8 fb_nv_frag_new(uint w0, uint w1, uint scale, float phi26, uint *guard) {
  const float k1 = tf_ink2_e4m3(scale) * 0x1p100f;
  const ushort2 ab = intel_convert_bfloat162_as_ushort2((float2)((FB_NV_T(2u) * k1) * phi26, (FB_NV_T(3u) * k1) * phi26));
  const uint a = ab.x, b = ab.y;
  const uint ea = (a >> 7) & 0xffu, eb = (b >> 7) & 0xffu;
  if (ea < 3u || ea > 252u || eb < 3u || eb > 252u) {
    *guard += 1u;
    return tf_ink2_nv_frag(w0, w1, scale, phi26);
  }
  int8 r;
  r.s0 = (int)fb_nv_pair(w0, 0u, a, b);
  r.s1 = (int)fb_nv_pair(w0, 2u, a, b);
  r.s2 = (int)fb_nv_pair(w0, 4u, a, b);
  r.s3 = (int)fb_nv_pair(w0, 6u, a, b);
  r.s4 = (int)fb_nv_pair(w1, 0u, a, b);
  r.s5 = (int)fb_nv_pair(w1, 2u, a, b);
  r.s6 = (int)fb_nv_pair(w1, 4u, a, b);
  r.s7 = (int)fb_nv_pair(w1, 6u, a, b);
  return r;
}

// Fills `n` uints with a hash of their index (the B9 weights and scales).
FB_SGA kernel void fb_fill_hash(global uint *w, uint n, uint seed) {
  for (uint i = get_global_id(0); i < n; i += get_global_size(0)) {
    uint x = i * 0x9e3779b9u + seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    w[i] = x;
  }
}

// Decode check: every block of `blocks` (8 code bytes each from w) against every scale code and each phi26 of `phis`.
// out[0] counts dwords that differ, out[1] blocks that took the guard.
FB_SGA kernel void fb_nv_check(global const uint2 *w, global const float *phis, uint nphi, uint blocks, global uint *out) {
  uint bad = 0u, guard = 0u;
  for (uint i = get_global_id(0); i < blocks; i += get_global_size(0)) {
    const uint2 v = w[i];
    for (uint s = 0u; s < 256u; s += 17u) {
      const uint scale = (s + i) & 255u;
      for (uint p = 0u; p < nphi; ++p) {
        const int8 o = tf_ink2_nv_frag(v.x, v.y, scale, phis[p]);
        const int8 n = fb_nv_frag_new(v.x, v.y, scale, phis[p], &guard);
        const int8 d = o ^ n;
        bad += (uint)(d.s0 != 0) + (uint)(d.s1 != 0) + (uint)(d.s2 != 0) + (uint)(d.s3 != 0) + (uint)(d.s4 != 0) +
               (uint)(d.s5 != 0) + (uint)(d.s6 != 0) + (uint)(d.s7 != 0);
      }
    }
  }
  bad = sub_group_reduce_add(bad);
  guard = sub_group_reduce_add(guard);
  fb_enter(FB_A, sync);
  if (get_sub_group_local_id() == 0u) {
    if (bad) atomic_add(out, bad);
    if (guard) atomic_add(out + 1, guard);
  }
  fb_enter(FB_A, sync);
}

// B9 phase: NVFP4 gate-up rows of k = 2,304 (1,152 code bytes and 144 scale bytes a row), tiles of 16 rows, one tile a
// triple of sub-groups over three slices of k (today's INK2_KS_NV_GU 3, today's nv_pass chunk loop at live 1), the three
  fb_enter(FB_A, sync);
// partial sums added in sub-group order and stored as bf16. 10 triples a group of 32 sub-groups, the last 2 sub-groups
// idle. Every triple runs ceil(tiles / triples) tiles (the last tile again past the end), so the group barriers match.
// `reps` phases over windows of win_stride bytes (codes, then scales at codes + rows * 1,152), M-a after each.
FB_SGA kernel void fb_nv_phase(global const uchar *w, global const ushort *x, global ushort *y, global uint *sync,
  fb_enter(FB_A, sync);
                               global uint *fault, uint rows, uint reps, uint nwin, uint win_stride, uint spin_cap,
                               uint decode, float phi26) {
  local float part[10 * 3 * 16];
  local uint lf;
  fb_enter(FB_A, sync);
  const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const uint tri = sg / 3u, ks = sg % 3u;
  const uint triples = get_num_groups(0) * 10u;
  const uint gt = get_group_id(0) * 10u + tri;
  fb_enter(FB_A, sync);
  const uint tiles = rows / 16u;
  const uint iters = (tiles + triples - 1u) / triples;
  const uint nb = 2304u / 16u;
  uint guard = 0u;
  fb_enter(FB_A, sync);
  for (uint r = 0u; r < reps; ++r) {
    global const uchar *codes = w + (ulong)(r % nwin) * win_stride;
    global const uchar *scales = codes + (ulong)rows * 1152u;
    for (uint it = 0u; it < iters; ++it) {
      const uint tile = min(gt + it * triples, tiles - 1u);
      float acc = 0.0f;
      if (tri < 10u) {
        const uint row = tile * 16u + lane;
        global const uchar *wr = codes + (ulong)row * 1152u;
        global const uchar *sr = scales + (ulong)row * nb;
        for (uint c = ks; c < nb / 8u; c += 3u) {
          const uint16 v = vload16(0, (global const uint *)(wr + 64u * c));
          const uint2 s2 = vload2(0, (global const uint *)(sr + 8u * c));
#pragma unroll
          for (int u = 0; u < 8; u++) {
            const uint sc = (s2[u >> 2] >> (8 * (u & 3))) & 255u;
            const int8 b = decode ? fb_nv_frag_new(v[2 * u], v[2 * u + 1], sc, phi26, &guard)
                                  : tf_ink2_nv_frag(v[2 * u], v[2 * u + 1], sc, phi26);
            acc = INK2_DOT((short)intel_sub_group_block_read_us(x + 128u * c + 16u * u), b, acc);
          }
        }
        part[(tri * 3u + ks) * 16u + lane] = acc;

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}
      }
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      if (tri < 10u && ks == 0u) {
        float t = part[(tri * 3u) * 16u + lane];
        t += part[(tri * 3u + 1u) * 16u + lane];

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}
        t += part[(tri * 3u + 2u) * 16u + lane];
        y[tile * 16u + lane] = intel_convert_bfloat16_as_ushort(t);
      }
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}
    fb_arrive(FB_A, sync, r + 1u);
    if (fb_wait(FB_A, sync, fault, r + 1u, spin_cap, &lf) != 0u) {
      return;
    }
  }

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}
  guard = sub_group_reduce_add(guard);
  if (lane == 0u && guard != 0u) atomic_add(fault + 3, guard);
}

// The tm0 rate: one work-item spins on its own clock (no memory) until `ticks` have passed. The host divides by the

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}
// event time of the launch.
FB_SGA kernel void fb_tsc(global ulong *o, uint ticks) {
  if (get_global_id(0) != 0u) return;
  const ulong t0 = __builtin_IB_read_cycle_counter();
  ulong t = t0;

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}
  uint n = 0u; // a bound in reads, so a clock that does not move cannot hang the launch
  while (t - t0 < (ulong)ticks && n < (1u << 24)) { t = __builtin_IB_read_cycle_counter(); ++n; }
  o[0] = t - t0;
  o[1] = n;
}

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}

// Review C1: the tm0 rate beside a full grid. Every work-item of every group reads its group's window of `work_u4`
// uint4 `passes` times. Lane 0 of group 0 stores its tm0 ticks from its start to its end; the host divides by the
// event time of the launch (all groups do the same reads, so group 0 runs about as long as the launch).
FB_SGA kernel void fb_tsc_grid(global const uint4 *data, global ulong *o, uint passes, uint work_u4) {
  const ulong t0 = __builtin_IB_read_cycle_counter();
  const uint g = get_group_id(0), lid = get_local_id(0), L = get_local_size(0);
  const ulong base = (ulong)g * work_u4;
  uint acc = 0u;
  for (uint p = 0u; p < passes; ++p) {
    __attribute__((opencl_unroll_hint(8))) for (uint i = lid; i < work_u4; i += L) {
      const uint4 v = data[base + i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc += p;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (g == 0u && lid == 0u) o[0] = __builtin_IB_read_cycle_counter() - t0;
  if (acc == 0x9e3779b9u) o[2 + g] = acc;
}

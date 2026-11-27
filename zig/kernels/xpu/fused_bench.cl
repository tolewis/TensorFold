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

#define FB_SGA __attribute__((intel_reqd_sub_group_size(16)))
#define FB_LINE 16u
#define FB_CNT 0u
#define FB_GEN 16u
#define FB_FLAGS 32u
#define FB_ROW 2304u // Q8 codes a column of k = 2,304

// Mechanism ids: 0 no wait, 1 M-a (counter, few arrivals), 2 M-b (flags and a coordinator), 3 M-c (flags, every
// leader polls), 4 M-e (B0 at one sub-group a work-group with a relaxed spin and one acquire fence).
#define FB_NONE 0u
#define FB_A 1u
#define FB_B 2u
#define FB_C 3u
#define FB_E 4u

typedef global atomic_uint fb_au;

static inline uint fb_ld(global uint *p) { return atomic_load_explicit((global fb_au *)p, memory_order_relaxed, memory_scope_device); }
static inline void fb_rel(void) { atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device); }
static inline void fb_acq(void) { atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device); }

// The arrive half of a phase change. Every work-item of the group calls it after its last store of the phase.
static inline void fb_arrive(uint mech, global uint *sync, uint round) {
  if (mech == FB_NONE) return;
  if (mech == FB_E) {
    if (get_local_id(0) == 0u)
      atomic_fetch_add_explicit((global fb_au *)(sync + FB_CNT), 1u, memory_order_acq_rel, memory_scope_device);
    return;
  }
  fb_rel();
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0u) {
    if (mech == FB_A)
      atomic_fetch_add_explicit((global fb_au *)(sync + FB_CNT), 1u, memory_order_relaxed, memory_scope_device);
    else
      atomic_store_explicit((global fb_au *)(sync + FB_FLAGS + get_group_id(0) * FB_LINE), round, memory_order_relaxed,
                            memory_scope_device);
  }
}

// The wait half. Sub-group 0 of the group spins (all 16 lanes, one address or one flag a lane), with a spin cap (R7).
// Returns 1 when the cap was reached. M-e returns without a native barrier (it runs at one sub-group a group).
static inline uint fb_wait(uint mech, global uint *sync, uint round, uint spin_cap, local uint *lf) {
  if (mech == FB_NONE) return 0u;
  const uint G = get_num_groups(0), g = get_group_id(0);
  uint f = 0u;
  if (get_sub_group_id() == 0u) {
    const uint lane = get_sub_group_local_id();
    uint spins = 0u;
    if (mech == FB_A || mech == FB_E) {
      const uint target = round * G;
      while (fb_ld(sync + FB_CNT) < target) {
        if (++spins > spin_cap) { f = 1u; break; }
      }
    } else if (mech == FB_C || g == 0u) {
      for (;;) {
        uint ok = 1u;
        for (uint b = 0u; b < G; b += 16u) {
          const uint i = b + lane;
          if (i < G) ok &= (uint)(fb_ld(sync + FB_FLAGS + i * FB_LINE) >= round);
        }
        if (sub_group_all(ok)) break;
        if (++spins > spin_cap) { f = 1u; break; }
      }
      if (mech == FB_B && f == 0u) {
        fb_acq();
        if (lane == 0u) atomic_store_explicit((global fb_au *)(sync + FB_GEN), round, memory_order_release, memory_scope_device);
      }
    } else {
      while (fb_ld(sync + FB_GEN) < round) {
        if (++spins > spin_cap) { f = 1u; break; }
      }
    }
    fb_acq();
    if (mech != FB_E && lane == 0u) *lf = f;
  }
  if (mech == FB_E) return f;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  return *lf;
}

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
    if (fb_wait(mech, sync, r, spin_cap, &lf) != 0u) {
      if (lid == 0u) { fault[0] = 1u; fault[2] = r; }
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
// columns over the sub-groups of the grid in turn, one phase change (mech) after each phase.
FB_SGA kernel void fb_q8(global const uchar *w, global const ushort *x, global ushort *y, global uint *sync,
                         global uint *fault, uint n, uint reps, uint nwin, uint win_stride, uint spin_cap, uint mech) {
  local uint lf;
  const uint T = get_num_groups(0) * get_num_sub_groups();
  const uint gs = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
  for (uint r = 0u; r < reps; ++r) {
    global const uchar *codes = w + (ulong)(r % nwin) * win_stride;
    global const half *scales = (global const half *)(codes + (ulong)n * FB_ROW);
    for (uint j = gs; j < n; j += T) {
      const ushort v = fb_q8_col(x, codes, scales, j, n);
      if (get_sub_group_local_id() == 0u) y[j] = v;
    }
    fb_arrive(mech, sync, r + 1u);
    if (fb_wait(mech, sync, r + 1u, spin_cap, &lf) != 0u) {
      if (get_local_id(0) == 0u) { fault[0] = 1u; fault[2] = r + 1u; }
      return;
    }
  }
}

// B5: `passes` x `phases` dependent Q8 phases of n columns. Phase i reads the 2,304 values the phase before wrote
// (columns 0 to 2,303 of a phase write the next x, the others write y). Between arrive and wait every sub-group
// prefetches pf_lines 64-byte lines of its first columns of the next phase into L2 (M-d). A pass covers phases
// windows of win_stride bytes. Pass p starts at w + (p % 24) x pass_stride.
FB_SGA kernel void fb_stream(global const uchar *w, global ushort *xbuf, global ushort *y, global uint *sync,
                             global uint *fault, uint n, uint phases, uint passes, uint win_stride, uint spin_cap,
                             uint mech, uint pf_lines) {
  local uint lf;
  const uint T = get_num_groups(0) * get_num_sub_groups();
  const uint gs = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const ulong pass_stride = (ulong)phases * win_stride;
  const uint total = phases * passes;
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
    if (fb_wait(mech, sync, i + 1u, spin_cap, &lf) != 0u) {
      if (get_local_id(0) == 0u) { fault[0] = 1u; fault[2] = i + 1u; }
      return;
    }
  }
}

// B8: each round prefetches (pf_on) block A_r of 2,400,000 bytes into L2, waits, streams stream_bytes of region B,
// waits, reads A_r (read_on), waits. A_r lies in the first 256 MiB of w, region B in the next 768 MiB.
#define FB_PF_BLOCK 2400000u
#define FB_PF_STRIDE 2457600u
FB_SGA kernel void fb_l2keep(global const uchar *w, global uint *sink, global uint *sync, global uint *fault,
                             uint rounds, uint pf_on, uint read_on, uint stream_bytes, uint spin_cap) {
  local uint lf;
  const uint T = get_num_groups(0) * get_num_sub_groups();
  const uint gs = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
  const uint lane = get_sub_group_local_id();
  const uint b_stride = (stream_bytes + 4095u) & ~4095u;
  const uint nb = (768u << 20) / b_stride;
  uint acc = 0u;
  for (uint r = 0u; r < rounds; ++r) {
    global const uchar *a = w + (ulong)(r % 100u) * FB_PF_STRIDE;
    global const uint4 *bq = (global const uint4 *)(w + (256ul << 20) + (ulong)(r % nb) * b_stride);
    if (pf_on != 0u)
      for (uint l = gs * 16u + lane; l < FB_PF_BLOCK / 64u; l += T * 16u) prefetch(a + (ulong)l * 64u, 64);
    fb_arrive(FB_A, sync, 3u * r + 1u);
    if (fb_wait(FB_A, sync, 3u * r + 1u, spin_cap, &lf) != 0u) { if (get_local_id(0) == 0u) fault[0] = 1u; return; }
    __attribute__((opencl_unroll_hint(8))) for (uint i = gs * 16u + lane; i < stream_bytes / 16u; i += T * 16u) {
      const uint4 v = bq[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    fb_arrive(FB_A, sync, 3u * r + 2u);
    if (fb_wait(FB_A, sync, 3u * r + 2u, spin_cap, &lf) != 0u) { if (get_local_id(0) == 0u) fault[0] = 1u; return; }
    if (read_on != 0u) {
      global const uint4 *aq = (global const uint4 *)a;
      __attribute__((opencl_unroll_hint(8))) for (uint i = gs * 16u + lane; i < FB_PF_BLOCK / 16u; i += T * 16u) {
        const uint4 v = aq[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }
    fb_arrive(FB_A, sync, 3u * r + 3u);
    if (fb_wait(FB_A, sync, 3u * r + 3u, spin_cap, &lf) != 0u) { if (get_local_id(0) == 0u) fault[0] = 1u; return; }
  }
  if (acc == 0x9e3779b9u) sink[gs] = acc;
}

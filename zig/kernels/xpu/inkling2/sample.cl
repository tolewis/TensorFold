// Inkling 2 Mini candidate selection for sampling on Intel Xe2 (OpenCL C).
//
// One work-group of 256 work-items takes one row of logits and returns the row's best `n` entries (n at most 1024) in
// the order the host sampler sorts them: the scaled logit v = logit * inv_t, larger first, ties to the lower token id.
// Everything here is comparison and integer arithmetic, so the list is exactly the one a host sort of the whole row gives.
// The host finishes the draw (the normalizer, nucleus and the race) from this list with its own libm.
//
// With bit 31 of n set (the row is drawn with top_k off) the kernel also writes the row's normalizer, the sum over every
// token of exp(v - m) with m the largest v, in the order the host's `mass` sums it: 1,024 strided lane sums (lane l takes
// ids l, l + 1,024, ... in id order), a 32-lane butterfly inside each group of 32 lanes, then the 32 groups in order. The
// exp is the device's, not the host libm's, so z can differ from the host's sum in the last bits.
//
// The cut is found by a radix select over the order-preserving key of v (four passes of eight bits). The entries above
// the cut and the first of the entries equal to it, in id order, are gathered by a chunked scan and then sorted in
// local memory with a bitonic network.

#pragma OPENCL FP_CONTRACT OFF

#define SMP_W 256
#define SMP_TOP 1024
#define SMP_PER 4 // ids each work-item takes in a row: one chunk is SMP_W * SMP_PER ids
#define SMP_LANES 1024 // the normalizer's lane sums
#define SMP_WANT_Z 0x80000000u

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// Larger value, larger key: the sign bit flips for a positive float, and every bit flips for a negative one.
static inline uint order_key(float v) {
  const uint b = as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// out[row * stride + j] = { bits of v, token id } for j below the row's n; cfg[2 * row] holds inv_t's bits, cfg[2 * row + 1] n
// (with SMP_WANT_Z set when zout[row] is wanted).
KERNEL(SMP_W) void tf_ink2_sample_top(global const float *logits, global const uint *cfg, global uint *out, global float *zout, uint vocab, uint stride) {
  local uint hist[256];
  local float lane_sum[SMP_LANES];
  local float group_sum[32];
  local ulong order[SMP_TOP];
  local uint cut_s, need_s, base_g, base_e, tot_g, tot_e;
  const uint t = get_local_id(0);
  const uint row = get_group_id(0);
  const float inv_t = as_float(cfg[2 * row]);
  const uint n = cfg[2 * row + 1] & ~SMP_WANT_Z;
  const bool want_z = (cfg[2 * row + 1] & SMP_WANT_Z) != 0;
  global const float *l = logits + (ulong)row * vocab;

  uint cut = 0;
  uint need = n;
  for (int p = 0; p < 4; p++) {
    const int shift = 24 - 8 * p;
    if (t < 256) hist[t] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = t; i < vocab; i += SMP_W) {
      const uint k = order_key(l[i] * inv_t);
      if (p == 0 || (k >> (shift + 8)) == (cut >> (shift + 8))) atomic_inc(&hist[(k >> shift) & 0xFFu]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      uint s = 0;
      uint d = 255;
      while (d > 0 && s + hist[d] < need) {
        s += hist[d];
        d--;
      }
      cut_s = cut | (d << shift);
      need_s = need - s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    cut = cut_s;
    need = need_s;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // cut is the n-th largest key; `need` of the entries equal to it are kept, the earliest ids first.
  const uint above = n - need;
  for (uint j = t; j < SMP_TOP; j += SMP_W) order[j] = 0;
  if (t == 0) {
    base_g = 0;
    base_e = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint c = 0; c < vocab; c += SMP_W * SMP_PER) {
    // each work-item owns SMP_PER consecutive ids, so the scan over work-items follows id order
    uint keys[SMP_PER];
    uint tg = 0;
    uint te = 0;
    for (uint j = 0; j < SMP_PER; j++) {
      const uint i = c + t * SMP_PER + j;
      keys[j] = i < vocab ? order_key(l[i] * inv_t) : 0u;
      tg += (i < vocab && keys[j] > cut) ? 1u : 0u;
      te += (i < vocab && keys[j] == cut) ? 1u : 0u;
    }
    const uint sg = work_group_scan_exclusive_add(tg);
    const uint se = work_group_scan_exclusive_add(te);
    if (t == SMP_W - 1) {
      tot_g = sg + tg;
      tot_e = se + te;
    }
    uint pg = base_g + sg;
    uint pe = base_e + se;
    for (uint j = 0; j < SMP_PER; j++) {
      const uint i = c + t * SMP_PER + j;
      if (i >= vocab) break;
      const ulong w = ((ulong)keys[j] << 32) | (ulong)(0xFFFFFFFFu - i);
      if (keys[j] > cut) {
        order[pg++] = w;
      } else if (keys[j] == cut) {
        if (pe < need) order[above + pe] = w;
        pe++;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      base_g += tot_g;
      base_e += tot_e;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Bitonic sort, larger first: a larger key wins, and for one key the larger 0xFFFFFFFF - id, which is the lower id.
  for (uint size = 2; size <= SMP_TOP; size <<= 1) {
    for (uint gap = size >> 1; gap > 0; gap >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint q = t; q < SMP_TOP / 2; q += SMP_W) {
        const uint a = 2 * q - (q & (gap - 1));
        const uint b = a + gap;
        const bool descending = (q & (size >> 1)) == 0;
        const ulong x = order[a];
        const ulong y = order[b];
        if (descending ? (x < y) : (x > y)) {
          order[a] = y;
          order[b] = x;
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (want_z) {
    const uint k0 = (uint)(order[0] >> 32);
    const float m = as_float((k0 & 0x80000000u) ? (k0 & 0x7FFFFFFFu) : ~k0);
    for (uint q = 0; q < SMP_LANES / SMP_W; q++) {
      const uint lane = t + q * SMP_W;
      float sum = 0.0f;
      for (uint i = lane; i < vocab; i += SMP_LANES) {
        const float v = l[i] * inv_t;
        if (v >= -INFINITY) sum += exp(v - m);
      }
      lane_sum[lane] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < 32) {
      float simd[32];
      for (uint j = 0; j < 32; j++) simd[j] = lane_sum[t * 32 + j];
      for (uint offset = 16; offset > 0; offset >>= 1) {
        float next[32];
        for (uint j = 0; j < 32; j++) next[j] = simd[j] + simd[j ^ offset];
        for (uint j = 0; j < 32; j++) simd[j] = next[j];
      }
      group_sum[t] = simd[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
      float z = 0.0f;
      for (uint g = 0; g < 32; g++) z += group_sum[g];
      zout[row] = z;
    }
  }
  for (uint j = t; j < n; j += SMP_W) {
    const ulong w = order[j];
    const uint k = (uint)(w >> 32);
    const uint bits = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    out[((ulong)row * stride + j) * 2] = bits;
    out[((ulong)row * stride + j) * 2 + 1] = 0xFFFFFFFFu - (uint)w;
  }
}

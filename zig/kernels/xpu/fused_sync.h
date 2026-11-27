// fused_sync.h: the phase-change mechanisms of lane fused-visit (Recipe lanes/inkspeed/fused-visit/DESIGN.md section 3),
// shared by fused_bench.cl and fused_small.cl. See fused_bench.cl for the sync buffer layout.
#ifndef FUSED_SYNC_H
#define FUSED_SYNC_H
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
#define FB_F 5u // M-f (CL 15:56Z): M-a with a lane-0 poll by a plain load, L1 and L3 uncached, no RMW

// IGC built-in: a plain LSC load with cache control. cacheOpt 1 is L1 uncached, L3 uncached; the vISA is
// `lsc_load.ugm.uc.uc` (checked with IGC_ShaderDumpEnable on inference1 ocloc, 2026-11-27).
uint __builtin_IB_lsc_load_global_uint(global uint *base, int immElemOff, int cacheOpt);

typedef global atomic_uint fb_au;

static inline uint fb_ld(global uint *p) { return atomic_load_explicit((global fb_au *)p, memory_order_relaxed, memory_scope_device); }
// A poll of one word. A relaxed atomic load compiles to an lsc_atomic_or at L2 (an RMW), so 16 lanes on one word make
// 16 RMWs on one line a poll. Run 1 of the bench polled with all 16 lanes and M-b reached the spin cap at G 32 and up.
// FB_POLL16 builds that form again.
static inline uint fb_poll1(global uint *p) {
#ifdef FB_POLL16
  return fb_ld(p);
#else
  uint v = 0u;
  if (get_sub_group_local_id() == 0u) v = fb_ld(p);
  return sub_group_broadcast(v, 0);
#endif
}
// The M-f poll: lane 0 loads with no RMW, the sub-group gets its value.
static inline uint fb_poll_plain(global uint *p) {
  uint v = 0u;
  if (get_sub_group_local_id() == 0u) v = __builtin_IB_lsc_load_global_uint(p, 0, 1);
  return sub_group_broadcast(v, 0);
}
static inline void fb_rel(void) { atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device); }
static inline void fb_acq(void) { atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device); }

// FB_FORCE in the mechanism argument forces a fault (brief section 6 rule 1): the last group stops its arrivals at
// round 3, so every other group reaches the spin cap or sees the fault word.
#define FB_FORCE 0x100u
#define FB_MECH(m) ((m) & 0xffu)

// The arrive half of a phase change. Every work-item of the group calls it after its last store of the phase.
static inline void fb_arrive(uint mech_arg, global uint *sync, uint round) {
  const uint mech = FB_MECH(mech_arg);
  if (mech == FB_NONE) return;
  const bool skip = (mech_arg & FB_FORCE) != 0u && get_group_id(0) == get_num_groups(0) - 1u && round >= 3u;
  if (mech == FB_E) {
    if (get_local_id(0) == 0u && !skip)
      atomic_fetch_add_explicit((global fb_au *)(sync + FB_CNT), 1u, memory_order_acq_rel, memory_scope_device);
    return;
  }
  fb_rel();
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0u && !skip) {
    if (mech == FB_A || mech == FB_F)
      atomic_fetch_add_explicit((global fb_au *)(sync + FB_CNT), 1u, memory_order_relaxed, memory_scope_device);
    else
      atomic_store_explicit((global fb_au *)(sync + FB_FLAGS + get_group_id(0) * FB_LINE), round, memory_order_relaxed,
                            memory_scope_device);
  }
}

// The fault word fault[0] (brief section 6 rule 1). A group that reaches the spin cap sets it. Every group reads it
// when it enters a wait and every 16 polls inside the spin, and leaves the launch when it is set.
static inline uint fb_fault_seen(global uint *fault) { return fb_poll1(fault); }

// The wait half. Lane 0 of sub-group 0 polls one word (M-a, M-e, M-b others), or sub-group 0 polls one flag a lane
// (M-c, M-b coordinator), with a spin cap (R7). Returns 1 on a fault (the cap here or the fault word set by another
// group). M-e returns without a native barrier (it runs at one sub-group a group).
static inline uint fb_wait(uint mech_arg, global uint *sync, global uint *fault, uint round, uint spin_cap, local uint *lf) {
  const uint mech = FB_MECH(mech_arg);
  if (mech == FB_NONE) return 0u;
  const uint G = get_num_groups(0), g = get_group_id(0);
  uint f = 0u;
  if (get_sub_group_id() == 0u) {
    const uint lane = get_sub_group_local_id();
    uint spins = 0u;
#define FB_SPIN_CHECK                                                    \
  if ((spins & 15u) == 0u && fb_fault_seen(fault) != 0u) { f = 2u; break; } \
  if (++spins > spin_cap) { f = 1u; break; }
    if (mech == FB_A || mech == FB_E) {
      const uint target = round * G;
      while (fb_poll1(sync + FB_CNT) < target) { FB_SPIN_CHECK }
    } else if (mech == FB_F) {
      // An uncached plain-load poll took about 6 us in the block 2 force run (20,000 polls, 121 ms), so M-f spins to
      // half the cap: a faulted launch ends inside 100 ms (brief section 6 rule 1).
      spin_cap /= 2u;
      const uint target = round * G;
      while (fb_poll_plain(sync + FB_CNT) < target) { FB_SPIN_CHECK }
    } else if (mech == FB_C || g == 0u) {
      for (;;) {
        uint ok = 1u;
        for (uint b = 0u; b < G; b += 16u) {
          const uint i = b + lane;
          if (i < G) ok &= (uint)(fb_ld(sync + FB_FLAGS + i * FB_LINE) >= round);
        }
        if (sub_group_all(ok)) break;
        FB_SPIN_CHECK
      }
      if (mech == FB_B && f == 0u) {
        fb_acq();
        if (lane == 0u) atomic_store_explicit((global fb_au *)(sync + FB_GEN), round, memory_order_release, memory_scope_device);
      }
    } else {
      while (fb_poll1(sync + FB_GEN) < round) { FB_SPIN_CHECK }
    }
#undef FB_SPIN_CHECK
    if (f == 0u && fb_fault_seen(fault) != 0u) f = 2u; // a fault set while this group was in its phase
    if (f == 1u && lane == 0u) {
      atomic_store_explicit((global fb_au *)(fault + 2), round, memory_order_relaxed, memory_scope_device);
      atomic_store_explicit((global fb_au *)fault, 1u, memory_order_relaxed, memory_scope_device);
    }
    fb_acq();
    if (mech != FB_E && lane == 0u) *lf = f;
  }
  if (mech == FB_E) return f;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  return *lf;
}

#endif

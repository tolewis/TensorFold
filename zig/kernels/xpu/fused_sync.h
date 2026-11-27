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
#define FB_ENTRY 32000u // the entry count (review C2): local id 0 of each group adds 1 when the group starts

// The fault buffer (uint words). 0 the fault word, 1 failed visibility checks, 2 the round of the first cap, 3 the B9
// guard count, then the record of the first group that reached the cap (review C2): 4 its group id, 5 the value of
// its last poll (counter, generation word, or the count of flags at the round), 6 the entry count at its cap, 7 and 8
// its tm0 ticks from the wait entry to the cap (low, high), 9 its polls. 10 the count of groups that reached the cap,
// 11 the count of groups that left on the fault word.
#define FB_F_ROUND 2u
#define FB_F_GROUP 4u
#define FB_F_SEEN 5u
#define FB_F_ENTERED 6u
#define FB_F_TICKS 7u
#define FB_F_POLLS 9u
#define FB_F_CAPPED 10u
#define FB_F_LEFT 11u
#define FB_ENTRY 32000u // the entry count (review C2): local id 0 of each group adds 1 when the group starts

// The fault buffer (uint words). 0 the fault word, 1 failed visibility checks, 2 the round of the first cap, 3 the B9
// guard count, then the record of the first group that reached the cap (review C2): 4 its group id, 5 the value of
// its last poll (counter, generation word, or the count of flags at the round), 6 the entry count at its cap, 7 and 8
// its tm0 ticks from the wait entry to the cap (low, high), 9 its polls. 10 the count of groups that reached the cap,
// 11 the count of groups that left on the fault word.
#define FB_F_ROUND 2u
#define FB_F_GROUP 4u
#define FB_F_SEEN 5u
#define FB_F_ENTERED 6u
#define FB_F_TICKS 7u
#define FB_F_POLLS 9u
// IGC built-in: the thread timestamp register tm0 (ISA `mov (2) ... tm0.0`, both dwords in one move;
// __builtin_IB_get_timestamp gave an undefined value). tm0 counts at the GT clock, so its rate moves with the clock
// (review C1). The host passes the cap in ticks for the lowest clock of the card (sysfs rpn_freq), so a cap ends a
// wait in that time or less at every clock (brief section 6 rule 1).
#define FB_ENTRY 32000u // the entry count (review C2): local id 0 of each group adds 1 when the group starts

// The fault buffer (uint words). 0 the fault word, 1 failed visibility checks, 2 the round of the first cap, 3 the B9
// guard count, then the record of the first group that reached the cap (review C2): 4 its group id, 5 the value of
// its last poll (counter, generation word, or the count of flags at the round), 6 the entry count at its cap, 7 and 8
// its tm0 ticks from the wait entry to the cap (low, high), 9 its polls. 10 the count of groups that reached the cap,
// 11 the count of groups that left on the fault word.
#define FB_F_ROUND 2u
#define FB_F_GROUP 4u
#define FB_F_SEEN 5u
#define FB_F_ENTERED 6u
#define FB_F_TICKS 7u
#define FB_F_POLLS 9u
// IGC built-in: the thread timestamp register tm0 (ISA `mov (2) ... tm0.0`, both dwords in one move;
// __builtin_IB_get_timestamp gave an undefined value). tm0 counts at the GT clock, so its rate moves with the clock
// (review C1). The host passes the cap in ticks for the lowest clock of the card (sysfs rpn_freq), so a cap ends a
// wait in that time or less at every clock (brief section 6 rule 1).

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

// The entry count (review C2): a fault record shows how many groups had started at the cap.
static inline void fb_enter(uint mech_arg, global uint *sync) {
  if (FB_MECH(mech_arg) != FB_NONE && get_local_id(0) == 0u)
    atomic_fetch_add_explicit((global fb_au *)(sync + FB_ENTRY), 1u, memory_order_relaxed, memory_scope_device);
}
// IGC built-in: the thread timestamp register tm0 (ISA `mov (2) ... tm0.0`, both dwords in one move;
// __builtin_IB_get_timestamp gave an undefined value). tm0 counts at the GT clock, so its rate moves with the clock
// (review C1). The host passes the cap in ticks for the lowest clock of the card (sysfs rpn_freq), so a cap ends a
// wait in that time or less at every clock (brief section 6 rule 1).
// (review C1). The host passes the cap in ticks for the lowest clock of the card (sysfs rpn_freq), so a cap ends a
// wait in that time or less at every clock (brief section 6 rule 1).
ulong __builtin_IB_read_cycle_counter(void);

typedef global atomic_uint fb_au;

static inline uint fb_ld(global uint *p) { return atomic_load_explicit((global fb_au *)p, memory_order_relaxed, memory_scope_device); }
// A poll of one word. A relaxed atomic load compiles to an lsc_atomic_or at L2 (an RMW), so 16 lanes on one word make
// 16 RMWs on one line a poll. Run 1 of the bench polled with all 16 lanes and M-b reached the spin cap at G 32 and up.
// FB_POLL16 builds that form again.
static inline uint fb_poll1(global uint *p) {
#ifdef FB_POLL16
  return fb_ld(p);

// The entry count (review C2): a fault record shows how many groups had started at the cap.
static inline void fb_enter(uint mech_arg, global uint *sync) {
  if (FB_MECH(mech_arg) != FB_NONE && get_local_id(0) == 0u)
    atomic_fetch_add_explicit((global fb_au *)(sync + FB_ENTRY), 1u, memory_order_relaxed, memory_scope_device);
}
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
    // spin_cap is the cap in tm0 ticks. The clock and the fault word are read every 16 polls.
    uint spins = 0u, seen = 0u;
    ulong dt = 0ul;
// FB_FORCE in the mechanism argument forces a fault (brief section 6 rule 1): the last group stops its arrivals at
// round 3, so every other group reaches the spin cap or sees the fault word.

    if (fb_fault_read(mech_arg, fault) != 0u) { f = 2u; break; }                        \
    dt = __builtin_IB_read_cycle_counter() - t0;                                        \
    if (dt > (ulong)spin_cap) { f = 1u; break; }                                        \
  if (FB_MECH(mech_arg) != FB_NONE && get_local_id(0) == 0u)
    atomic_fetch_add_explicit((global fb_au *)(sync + FB_ENTRY), 1u, memory_order_relaxed, memory_scope_device);
}
      while ((seen = fb_poll1(sync + FB_CNT)) < target) { FB_SPIN_CHECK }
#define FB_MECH(m) ((m) & 0xffu)

      while ((seen = fb_poll_plain(sync + FB_CNT)) < target) { FB_SPIN_CHECK }
static inline void fb_arrive(uint mech_arg, global uint *sync, uint round) {
  const uint mech = FB_MECH(mech_arg);
  if (mech == FB_NONE) return;
  const bool skip = (mech_arg & FB_FORCE) != 0u && get_group_id(0) == get_num_groups(0) - 1u && round >= 3u;
  if (mech == FB_E) {
    if (get_local_id(0) == 0u && !skip)
      atomic_fetch_add_explicit((global fb_au *)(sync + FB_CNT), 1u, memory_order_acq_rel, memory_scope_device);
    return;
  }
    // spin_cap is the cap in tm0 ticks. The clock and the fault word are read every 16 polls.
      if (f == 1u) { // the record: the count of flags at this round or past it
        uint n = 0u;
        for (uint b = 0u; b < G; b += 16u) {
          const uint i = b + lane;
          if (i < G) n += (uint)(fb_ld(sync + FB_FLAGS + i * FB_LINE) >= round);
        }
        seen = sub_group_reduce_add(n);
      }
    uint spins = 0u, seen = 0u;
    ulong dt = 0ul;
  if (get_local_id(0) == 0u && !skip) {
    if (mech == FB_A || mech == FB_F)
      atomic_fetch_add_explicit((global fb_au *)(sync + FB_CNT), 1u, memory_order_relaxed, memory_scope_device);
      while ((seen = fb_poll1(sync + FB_GEN)) < round) { FB_SPIN_CHECK }
    dt = __builtin_IB_read_cycle_counter() - t0;                                        \
    if (dt > (ulong)spin_cap) { f = 1u; break; }                                        \
                            memory_scope_device);
  }
}
      atomic_fetch_add_explicit((global fb_au *)(fault + FB_F_CAPPED), 1u, memory_order_relaxed, memory_scope_device);
      uint zero = 0u;
      if (atomic_compare_exchange_strong_explicit((global fb_au *)fault, &zero, 1u, memory_order_relaxed, memory_order_relaxed,
                                                  memory_scope_device)) {
        fault[FB_F_ROUND] = round;
        fault[FB_F_GROUP] = g;
        fault[FB_F_SEEN] = seen;
        fault[FB_F_ENTERED] = fb_ld(sync + FB_ENTRY);
        fault[FB_F_TICKS] = (uint)dt;
        fault[FB_F_TICKS + 1u] = (uint)(dt >> 32);
        fault[FB_F_POLLS] = spins;
      }
// when it enters a wait and every 16 polls inside the spin, and leaves the launch when it is set.
    if (f == 2u && lane == 0u)
      atomic_fetch_add_explicit((global fb_au *)(fault + FB_F_LEFT), 1u, memory_order_relaxed, memory_scope_device);
    if ((mech_arg & (FB_NO_ACQ | FB_ACQ_ALL)) == 0u) fb_acq();

// The wait half. Lane 0 of sub-group 0 polls one word (M-a, M-e, M-b others), or sub-group 0 polls one flag a lane
// (M-c, M-b coordinator), with a spin cap (R7). Returns 1 on a fault (the cap here or the fault word set by another
// group). M-e returns without a native barrier (it runs at one sub-group a group).
static inline uint fb_wait(uint mech_arg, global uint *sync, global uint *fault, uint round, uint spin_cap, local uint *lf) {
  const uint mech = FB_MECH(mech_arg);
  if (mech == FB_NONE) return 0u;
  const uint G = get_num_groups(0), g = get_group_id(0);
  uint f = 0u;
    // spin_cap is the cap in tm0 ticks. The clock and the fault word are read every 16 polls.
      if (f == 1u) { // the record: the count of flags at this round or past it
        uint n = 0u;
        for (uint b = 0u; b < G; b += 16u) {
          const uint i = b + lane;
          if (i < G) n += (uint)(fb_ld(sync + FB_FLAGS + i * FB_LINE) >= round);
        }
        seen = sub_group_reduce_add(n);
      }
    uint spins = 0u, seen = 0u;
    ulong dt = 0ul;
    // spin_cap is the cap in tm0 ticks (the host converts 50 ms). The clock and the fault word are read every 16 polls.
    uint spins = 0u;
    const ulong t0 = __builtin_IB_read_cycle_counter();
      while ((seen = fb_poll1(sync + FB_GEN)) < round) { FB_SPIN_CHECK }
    dt = __builtin_IB_read_cycle_counter() - t0;                                        \
    if (dt > (ulong)spin_cap) { f = 1u; break; }                                        \
    if (fb_fault_seen(fault) != 0u) { f = 2u; break; }                                  \
    if (__builtin_IB_read_cycle_counter() - t0 > (ulong)spin_cap) { f = 1u; break; }       \
  }
      atomic_fetch_add_explicit((global fb_au *)(fault + FB_F_CAPPED), 1u, memory_order_relaxed, memory_scope_device);
      uint zero = 0u;
      if (atomic_compare_exchange_strong_explicit((global fb_au *)fault, &zero, 1u, memory_order_relaxed, memory_order_relaxed,
                                                  memory_scope_device)) {
        fault[FB_F_ROUND] = round;
        fault[FB_F_GROUP] = g;
        fault[FB_F_SEEN] = seen;
        fault[FB_F_ENTERED] = fb_ld(sync + FB_ENTRY);
        fault[FB_F_TICKS] = (uint)dt;
        fault[FB_F_TICKS + 1u] = (uint)(dt >> 32);
        fault[FB_F_POLLS] = spins;
      }
      while (fb_poll1(sync + FB_CNT) < target) { FB_SPIN_CHECK }
    if (f == 2u && lane == 0u)
      atomic_fetch_add_explicit((global fb_au *)(fault + FB_F_LEFT), 1u, memory_order_relaxed, memory_scope_device);
    if ((mech_arg & (FB_NO_ACQ | FB_ACQ_ALL)) == 0u) fb_acq();
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
      if (f == 1u) { // the record: the count of flags at this round or past it
        uint n = 0u;
        for (uint b = 0u; b < G; b += 16u) {
          const uint i = b + lane;
          if (i < G) n += (uint)(fb_ld(sync + FB_FLAGS + i * FB_LINE) >= round);
        }
        seen = sub_group_reduce_add(n);
      }
        FB_SPIN_CHECK
      }
      if (mech == FB_B && f == 0u) {
        fb_acq();
        if (lane == 0u) atomic_store_explicit((global fb_au *)(sync + FB_GEN), round, memory_order_release, memory_scope_device);
      while ((seen = fb_poll1(sync + FB_GEN)) < round) { FB_SPIN_CHECK }
    } else {
      while (fb_poll1(sync + FB_GEN) < round) { FB_SPIN_CHECK }
    }
#undef FB_SPIN_CHECK
    if (f == 0u && fb_fault_seen(fault) != 0u) f = 2u; // a fault set while this group was in its phase
      atomic_fetch_add_explicit((global fb_au *)(fault + FB_F_CAPPED), 1u, memory_order_relaxed, memory_scope_device);
      uint zero = 0u;
      if (atomic_compare_exchange_strong_explicit((global fb_au *)fault, &zero, 1u, memory_order_relaxed, memory_order_relaxed,
                                                  memory_scope_device)) {
        fault[FB_F_ROUND] = round;
        fault[FB_F_GROUP] = g;
        fault[FB_F_SEEN] = seen;
        fault[FB_F_ENTERED] = fb_ld(sync + FB_ENTRY);
        fault[FB_F_TICKS] = (uint)dt;
        fault[FB_F_TICKS + 1u] = (uint)(dt >> 32);
        fault[FB_F_POLLS] = spins;
      }
      atomic_store_explicit((global fb_au *)(fault + 2), round, memory_order_relaxed, memory_scope_device);
    if (f == 2u && lane == 0u)
      atomic_fetch_add_explicit((global fb_au *)(fault + FB_F_LEFT), 1u, memory_order_relaxed, memory_scope_device);
    if ((mech_arg & (FB_NO_ACQ | FB_ACQ_ALL)) == 0u) fb_acq();
    }
    fb_acq();
    if (mech != FB_E && lane == 0u) *lf = f;
  }
  if (mech == FB_E) return f;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  return *lf;
}

#endif

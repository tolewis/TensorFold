__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void tf_barrier_bench(global atomic_uint *arrive, global atomic_uint *gen,
                               global uint *fault, global uint *sink,
                               uint barriers, uint spin_cap)
{
    int group = get_group_id(0);
    int lane = get_local_id(0);
    uint sum = 0u;
    for (uint round = 0u; round < barriers; round++) {
        uint fault_state = 0u;
        if (lane == 0) {
            uint g = atomic_load_explicit(gen, memory_order_acquire, memory_scope_device);
            uint old = atomic_fetch_add_explicit(arrive, 1u, memory_order_acq_rel, memory_scope_device);
            if (old == get_num_groups(0) - 1u) {
                atomic_store_explicit(arrive, 0u, memory_order_relaxed, memory_scope_device);
                atomic_store_explicit(gen, g + 1u, memory_order_release, memory_scope_device);
            } else {
                uint spins = 0u;
                while (atomic_load_explicit(gen, memory_order_acquire, memory_scope_device) == g) {
                    spins++;
                    if (spins > spin_cap) {
                        *fault = 1u;
                        fault_state = 1u;
                        break;
                    }
                }
            }
        }
        fault_state = sub_group_broadcast(fault_state, 0);
        if (fault_state != 0u) return;
        sum += round;
    }
    if (group == 0 && lane == 0) *sink = sum;
}

// A two-level barrier: one leaf counter a leaf of `leaf` work-groups, each leaf a cache line apart,
// then one root counter among the leaves. Every leaf frees its own groups after the root completes.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void tf_barrier_bench_tree(global atomic_uint *arrive, global atomic_uint *gen,
                                    global atomic_uint *root_arrive, global atomic_uint *root_gen,
                                    global uint *fault, global uint *sink,
                                    uint barriers, uint spin_cap, uint leaf)
{
    uint group = get_group_id(0);
    uint lane = get_local_id(0);
    uint groups = get_num_groups(0);
    uint li = group / leaf;
    uint leaf_start = li * leaf;
    uint leaf_size = min(leaf, groups - leaf_start);
    uint n_leaves = (groups + leaf - 1u) / leaf;
    global atomic_uint *la = arrive + li * 16u;
    global atomic_uint *lg = gen + li * 16u;
    uint sum = 0u;
    for (uint round = 0u; round < barriers; round++) {
        uint fault_state = 0u;
        if (lane == 0u) {
            uint g = atomic_load_explicit(lg, memory_order_acquire, memory_scope_device);
            uint old = atomic_fetch_add_explicit(la, 1u, memory_order_acq_rel, memory_scope_device);
            if (old != leaf_size - 1u) {
                uint spins = 0u;
                while (atomic_load_explicit(lg, memory_order_acquire, memory_scope_device) == g) {
                    spins++;
                    if (spins > spin_cap) { *fault = 1u; fault_state = 1u; break; }
                }
            } else {
                atomic_store_explicit(la, 0u, memory_order_relaxed, memory_scope_device);
                uint rg = atomic_load_explicit(root_gen, memory_order_acquire, memory_scope_device);
                uint rold = atomic_fetch_add_explicit(root_arrive, 1u, memory_order_acq_rel, memory_scope_device);
                if (rold != n_leaves - 1u) {
                    uint spins = 0u;
                    while (atomic_load_explicit(root_gen, memory_order_acquire, memory_scope_device) == rg) {
                        spins++;
                        if (spins > spin_cap) { *fault = 1u; fault_state = 1u; break; }
                    }
                } else {
                    atomic_store_explicit(root_arrive, 0u, memory_order_relaxed, memory_scope_device);
                    atomic_store_explicit(root_gen, rg + 1u, memory_order_release, memory_scope_device);
                }
                atomic_store_explicit(lg, g + 1u, memory_order_release, memory_scope_device);
            }
        }
        fault_state = sub_group_broadcast(fault_state, 0);
        if (fault_state != 0u) return;
        sum += round;
    }
    if (group == 0u && lane == 0u) *sink = sum;
}

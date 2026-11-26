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

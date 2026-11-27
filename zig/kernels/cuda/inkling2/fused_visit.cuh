// The fused visit's table (fused-cuda DESIGN section 5): one FusedVisit a visit in device memory, filled by
// zig/src/families/inkling2/fused.zig (extern structs, field for field). The argument structs of today's kernels
// (ops.cu ConvArgs and MixArgs, loop.cu LoopMixArgs and LoopGateArgs) live in device memory too, and the table holds
// their addresses, so every unit reads today's argument struct.
#pragma once
#include <cstdint>

struct FzProj {          // one MXFP8 qmmf call of one row: out[N] = bf16(scale * W x)
    const void* x;        // bf16 [K]
    const unsigned char* codes;
    const uint8_t* scales;
    void* out;            // bf16 [N]
    float* part;          // fp32 [SK][N] scratch
    float scale;
    int N, K, SK, npad, mode;  // mode: qmmf Mode (1 FP8, 2 MXFP8)
};

struct FzPrep {          // tf_ink2_prep8 (kernels.zig prep)
    const uint16_t *q_raw, *k_raw, *v_raw, *rel_raw, *q_scale, *rel_scale;
    const int *rpos, *rseg;
    const void* segs;
    uint16_t *q_out, *rel_out, *kc, *vc;
    float *ksc, *vsc;
    int heads, kv_heads, ring, global;
    float qk_clip, rel_clip, floor_pos, alpha, eps, pad;
};

struct FzAttn {          // tf_ink2_attn8_db and tf_ink2_merge (kernels.zig attend), tf_ink2_rel_table
    const uint16_t *q, *kc, *vc, *tab;
    const void* segs;
    const int2* tiles;
    float* part_o;
    float2* part_ml;
    const float *ksc, *vsc;
    const uint16_t *sink, *gate;
    const int* rpos;
    uint16_t* att;
    const uint16_t *relp, *rel_proj;
    uint16_t* rtab;
    int heads, kv_heads, ring, window, chunk, chunks, split, zgrid, slots_rh, subs;
    float scale;
    int db;  // 1: the prefetching tiles (tf_ink2_attn8_db), 0: tf_ink2_attn8 (TF_INKLING2_ATTN8_DB picks, as today)
};

struct FzMoe {           // router, top 8, plan, NVFP4 experts (cuda/experts.zig runSplit), combine
    const uint16_t *xm, *router;
    float* scores;
    const float* bias;
    int* pick;
    float* rw;
    int *members, *items, *counts;
    const uint4 *up, *down;
    const float *up_scale, *down_scale;
    uint16_t* act;
    float* y;
    uint16_t* moe;
    int E, K, T, H, width, pad;
    float route_scale, gate_cap, up_cap, pad2;
};

struct FzTaps {          // the k and v taps of a global visit: dense_rows_scaled_1 (32 outputs a side)
    const uint16_t* x[2];
    const uint16_t* w[2];
    const float* ones;
    float* out[2];
    int n, k;
};

struct FusedVisit {
    int global, gated, nq_attn, n_proj;   // n_proj: 1 (all) or 3 (qx, wk, wv)
    int copy_bytes, drop, probe, fix;      // the boundary copy (0: none); drop: a test's skipped arrival, probe: the
                                           // probed phase + 1 (0: none), fix: the review corrections on (fz_fix), fused.zig
    void* copy_dst;
    const void* copy_src;
    const void* mix_attn;   // ops MixArgs (plain) or loop LoopMixArgs (gated)
    const void* mix_mlp;
    const void* gate_attn;  // loop LoopGateArgs (gated), 3 gates
    const void* gate_mlp;   // 1 gate
    const void* conv[4];    // ops ConvArgs: k, v, attn, mlp
    FzProj proj[3];
    FzProj wo;
    FzTaps taps;
    FzPrep prep;
    FzAttn attn;
    FzMoe moe;
};

struct FusedState {
    unsigned* cnt;          // zeroed counters, 8 a phase (32 phases a visit at most)
    unsigned* tk;           // zeroed tickets, 1,024 a visit: 256 a qmmf call
    unsigned* fault;        // one word a visit
    float* part1;           // mode S partials
    float* part2;
    unsigned long long* times;  // 64 a visit: globaltimer after each phase change (block 0)
    unsigned long long* probe;  // the probed phase: start [G], arrival [G], wait exit [G] a block, then 32 unit stamps
};

// The review corrections a launch runs (FusedVisit fix), so that one run times a phase with and without each.
namespace fz_fix {
constexpr int kBusy = 1;     // A9: the global attention deals its busy units only
constexpr int kGateHalf = 2; // A11: gate down as half units, one rank row a warp
constexpr int kLoopRead = 4; // A12: a loop mix pass 2 unit reads each partial once
constexpr int kLastWait = 8; // A18: no wait after the last phase of a launch
constexpr int kMixBatch = 64; // X1: a mode S mix pass 2 issues its partial loads before its warp sums
constexpr int kGateUnroll = 128; // X2: gate down's rank row loop unrolled by 4 (each lane's sum order kept)
constexpr int kPlanCg = 32;  // A1: the expert units read the plan with __ldcg (16 is the host's own-visit clear, A17)
}

// Unit stamps of a probed phase (FusedState probe + 3 G), globaltimer ns
namespace fz_probe {
constexpr int kMix = 0;    // mix: 0 pass 2 unit 0 staged, 1 after its first wait, 2 coefficients, 3 after its second
                           // wait, 4 unit end, 5 pass 1 unit 0 arrival
constexpr int kGate = 8;   // gate down unit 0: 8 start, 9 norms and xs, 10 rank rows done
constexpr int kTop = 12;   // block 0: 12 start, 13 top 8 done, 14 plan done
constexpr int kProj = 16;  // block 0, unit j < 4: 16 + 4 j start, + 1 body done, + 2 ticket, + 3 tile sum done
}

// The waits of the fused visit (review A2, A3). Every wait spins on one counter with the same 200 ms limit (R8) and
// reads the visit's fault word in each spin. A wait that reaches the limit sets the fault word once:
// (code << 16) | (block + 1), code = ((group + 1) << 3) | sub. `group` is the counter group (8 counters) of the wait,
// unique in a launch. `sub` is the counter in the group: 0 for an all-of wait and for the first wait of a mix, 1 + q
// for the second wait of mix query q, 7 for the last wait of a mix. So no two waits of a launch have one code.
namespace fz_sync {
constexpr unsigned long long kLimitNs = 200000000ull;  // 200 ms a wait (R8)

__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned code(int group, int sub) {
    return (static_cast<unsigned>(group + 1) << 3) | static_cast<unsigned>(sub);
}
// Spin until *p >= target. False when the fault word is set, or at the limit (this wait's code goes in first).
__device__ __forceinline__ bool wait_ge(const unsigned* p, unsigned target, unsigned* fault, unsigned c) {
    const unsigned long long t0 = now();
    while (ld_acq(p) < target) {
        if (now() - t0 > kLimitNs || ld_acq(fault) != 0) {
            atomicCAS(fault, 0u, (c << 16) | (blockIdx.x + 1u));
            return false;
        }
    }
    return true;
}
}  // namespace fz_sync

// fused.zig asserts the same sizes
static_assert(sizeof(FzProj) == 64 && sizeof(FzPrep) == 160 && sizeof(FzAttn) == 184, "fused.zig layout");
static_assert(sizeof(FzMoe) == 168 && sizeof(FzTaps) == 64 && sizeof(FusedVisit) == 944, "fused.zig layout");
static_assert(sizeof(FusedState) == 56, "fused.zig layout");

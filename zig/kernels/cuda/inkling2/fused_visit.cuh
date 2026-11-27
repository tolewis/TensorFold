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
    int copy_bytes, pad0, pad1, pad2;      // the boundary copy (0: none)
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
};

// fused.zig asserts the same sizes
static_assert(sizeof(FzProj) == 64 && sizeof(FzPrep) == 160 && sizeof(FzAttn) == 184, "fused.zig layout");
static_assert(sizeof(FzMoe) == 168 && sizeof(FzTaps) == 64 && sizeof(FusedVisit) == 944, "fused.zig layout");
static_assert(sizeof(FusedState) == 48, "fused.zig layout");

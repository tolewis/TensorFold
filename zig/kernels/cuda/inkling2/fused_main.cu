// tf_ink2_fused_visit: one cooperative kernel runs every phase of a range of decode visits (fused-cuda DESIGN sections
// 5, 10 and 11). The glue flags. Each phase calls the unit of today's kernel in its own unit file. Between two phases
// every block waits (M4 counter, P = G). Each wait has a time limit: at the limit the block sets the visit's fault
// word, (phase << 16) | (block + 1), and leaves (R8). Block 0 stores the global timer after each phase change.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inkling2/fused_visit.cuh"

struct ConvArgs;
struct LoopGateArgs;
namespace fz_mix {
struct MixArgs;
struct LoopMixArgs;
}  // namespace fz_mix
namespace ink2 {
struct Seg;
}

extern "C" __device__ void fz_conv(uint3 b, const ConvArgs& a);
extern "C" __device__ void fz_rowdot_warp(int j, const uint16_t* x, const uint16_t* W, void* out, int J, int C,
                                          int out_bf16, int act);
extern "C" __device__ void fz_topk_warp(const float* scores, const float* bias, int* pick, float* wts, int E, int K,
                                        float scale);
extern "C" __device__ void fz_prep8(int y, const uint16_t* q_raw, const uint16_t* k_raw, const uint16_t* v_raw,
                                    const uint16_t* rel_raw, const uint16_t* q_scale, const uint16_t* rel_scale,
                                    const int* rpos, const int* rseg, const ink2::Seg* segs, uint16_t* q_out,
                                    uint16_t* rel_out, uint16_t* kc, uint16_t* vc, int heads, int kv_heads, int ring,
                                    int global, float qk_clip, float rel_clip, float floor_pos, float alpha, float eps,
                                    float* ksc, float* vsc);
extern "C" __device__ void fz_rel_table(int rh, const uint16_t* rel, const uint16_t* proj, uint16_t* tab);
extern "C" __device__ void fz_attn8(uint3 b, int zn_grid, const uint16_t* q, const uint16_t* kc, const uint16_t* vc,
                                    const uint16_t* tab, const ink2::Seg* segs, const int2* tiles, float* part_o,
                                    float2* part_ml, int heads, int kv_heads, int ring, int window, int chunk, int c0,
                                    int chunks, float scale, int split, const float* ksc, const float* vsc, int db,
                                    unsigned char* dyn);
extern "C" __device__ void fz_merge(int rh, const float* part_o, const float2* part_ml, const uint16_t* sink,
                                    const uint16_t* gate, const int* rpos, uint16_t* out, int heads, int window,
                                    int chunk, int c0, int chunks, int subs);
extern "C" __device__ void fz_qmmf_unit(const __nv_bfloat16* x, const unsigned char* w, const uint8_t* bs, float scale,
                                        void* out, float* part, int N, int K, int SK, int npad, int tile, int slc,
                                        int mode);
extern "C" __device__ void fz_qmmf_tile_sum(const float* part, __nv_bfloat16* out, int N, int SK, float scale,
                                            int tile);
extern "C" __device__ void fz_gate_down(const LoopGateArgs& a, int g, int z, float* xs, float* sh);
extern "C" __device__ void fz_gate_lin(const LoopGateArgs& a, int g, int z, float* dn);
extern "C" __device__ void fz_gate_mix(const LoopGateArgs& a, int g, float* sh);
extern "C" __device__ void fz_nvfp4_up(const __nv_bfloat16* X, int x_stride, int slots, const uint4* W,
                                       const float* scale, int KG, int NB, const int* items, const int* counts,
                                       const int* members, void* out, int N, float gate_cap, float up_cap, int skip);
extern "C" __device__ void fz_nvfp4_down(const __nv_bfloat16* X, int x_stride, int slots, const uint4* W,
                                         const float* scale, int KG, int NB, const int* items, const int* counts,
                                         const int* members, void* out, int N, float limit, int skip);
extern "C" __device__ void fz_combine(const float* y, const float* w, uint16_t* out, int slots, int dims, int d);
extern "C" __device__ void fz_dense_scaled_warp(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* scale,
                                                float* out, int n, int k, int col);
extern "C" __device__ void fz_plan(const int* picks, int P, int E, int T, int* members, int* items, int* counts);
extern "C" __device__ void fz_mix_s(const fz_mix::MixArgs& a, float* part1, float* part2, unsigned* cnt,
                                    unsigned* fault, unsigned char* stg);
extern "C" __device__ void fz_loop_s(const fz_mix::LoopMixArgs& a, float* part1, float* part2, unsigned* cnt,
                                     unsigned* fault, unsigned char* stg);

namespace {

constexpr unsigned long long kLimitNs = 200000000ull;  // 200 ms a wait (R8)

__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void red_rel_add(unsigned* p, unsigned v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

struct Ctx {
    FusedState st;
    unsigned* fault;
    unsigned long long* times;
    int ci;   // the next counter group (block-uniform)
    int ph;   // the phase number in the visit
};

// The all-of wait (M4, P = G). Returns false when this block or another block set the fault word.
__device__ bool all_wait(Ctx& c) {
    __shared__ int bad;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* cnt = c.st.cnt + 8 * c.ci;
        red_rel_add(cnt, 1u);
        const unsigned long long t0 = now();
        bool ok = true;
        while (ld_acq(cnt) < gridDim.x) {
            if (now() - t0 > kLimitNs || ld_acq(c.fault) != 0) {
                ok = false;
                break;
            }
        }
        if (!ok) atomicCAS(c.fault, 0u, (static_cast<unsigned>(c.ph + 1) << 16) | (blockIdx.x + 1u));
        bad = !ok;
        if (ok && blockIdx.x == 0 && c.ph < 63) c.times[c.ph + 1] = now();
    }
    __syncthreads();
    c.ci++;
    c.ph++;
    return !bad;
}

__device__ __forceinline__ bool fault_set(const Ctx& c) {
    __shared__ int f;
    __syncthreads();
    if (threadIdx.x == 0) f = ld_acq(c.fault) != 0;
    __syncthreads();
    return f;
}

// One phase of up to 3 MXFP8 projections of one row: units (tile, slice), the slice sum by a ticket a tile.
__device__ void proj_phase(const FzProj* p, int n, const Ctx& c, int call0) {
    __shared__ int last;
    int total = 0;
    for (int i = 0; i < n; ++i) total += ((p[i].N + 63) / 64) * p[i].SK;
    for (int u = blockIdx.x; u < total; u += gridDim.x) {
        int i = 0, base = 0;
        while (u - base >= ((p[i].N + 63) / 64) * p[i].SK) {
            base += ((p[i].N + 63) / 64) * p[i].SK;
            ++i;
        }
        const FzProj& q = p[i];
        const int tile = (u - base) / q.SK, slc = (u - base) % q.SK;
        fz_qmmf_unit(static_cast<const __nv_bfloat16*>(q.x), q.codes, q.scales, q.scale, q.out, q.part, q.N, q.K, q.SK,
                     q.npad, tile, slc, q.mode);
        if (q.SK > 1) {
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                last = atomicAdd(c.st.tk + 256 * (call0 + i) + tile, 1u) == static_cast<unsigned>(q.SK) - 1;
                if (last) __threadfence();
            }
            __syncthreads();
            if (last) fz_qmmf_tile_sum(q.part, static_cast<__nv_bfloat16*>(q.out), q.N, q.SK, q.scale, tile);
        }
        __syncthreads();
    }
}

// A conv of today's grid [ceil(C / 128)] as virtual blocks: block b runs blocks b, b + G, ...
__device__ void conv_phase(const void* a, int C, int first) {
    const int nb = (C + 127) / 128;
    for (int u = static_cast<int>(blockIdx.x) - first; u < nb; u += gridDim.x)
        if (u >= 0) fz_conv(make_uint3(u, 0, 0), *static_cast<const ConvArgs*>(a));
}

__device__ int conv_c(const void* a) { return reinterpret_cast<const int*>(static_cast<const char*>(a) + 88)[4]; }

// The gates of one input: gate down (8 z blocks a gate), gate lin (18 z blocks), gate mix (a block a gate).
__device__ bool gates_phase(const void* ga, int gates, Ctx& c, unsigned char* dyn) {
    const LoopGateArgs& g = *static_cast<const LoopGateArgs*>(ga);
    float* sh = reinterpret_cast<float*>(dyn);
    float* buf = sh + 32;
    for (int u = blockIdx.x; u < gates * 8; u += gridDim.x) fz_gate_down(g, u / 8, u % 8, buf, sh);
    if (!all_wait(c)) return false;
    for (int u = blockIdx.x; u < gates * 18; u += gridDim.x) fz_gate_lin(g, u / 18, u % 18, buf);
    if (!all_wait(c)) return false;
    for (int u = blockIdx.x; u < gates; u += gridDim.x) fz_gate_mix(g, u, sh);
    return all_wait(c);
}

__device__ bool mix_phase(const void* a, bool gated, Ctx& c, unsigned char* dyn) {
    unsigned* cnt = c.st.cnt + 8 * c.ci;
    if (gated)
        fz_loop_s(*static_cast<const fz_mix::LoopMixArgs*>(a), c.st.part1, c.st.part2, cnt, c.fault, dyn);
    else
        fz_mix_s(*static_cast<const fz_mix::MixArgs*>(a), c.st.part1, c.st.part2, cnt, c.fault, dyn);
    c.ci++;
    c.ph++;
    if (threadIdx.x == 0 && blockIdx.x == 0 && c.ph < 64) c.times[c.ph] = now();
    return !fault_set(c);
}

__device__ bool visit(const FusedVisit& V, Ctx& c, unsigned char* dyn) {
    if (threadIdx.x == 0 && blockIdx.x == 0) c.times[0] = now();
    // the attention input: the mix, or the loop mix and the 3 gates
    if (!mix_phase(V.mix_attn, V.gated, c, dyn)) return false;
    if (V.gated && !gates_phase(V.gate_attn, 3, c, dyn)) return false;
    // the input projections (one call, or qx, wk, wv)
    proj_phase(V.proj, V.n_proj, c, 0);
    if (!all_wait(c)) return false;
    if (V.global) {  // the k and v taps: a warp an output
        const int w = blockIdx.x * 4 + (threadIdx.x >> 5), W = gridDim.x * 4;
        for (int u = w; u < 2 * V.taps.n; u += W) {
            const int side = u / V.taps.n;
            fz_dense_scaled_warp(reinterpret_cast<const __nv_bfloat16*>(V.taps.x[side]),
                                 reinterpret_cast<const __nv_bfloat16*>(V.taps.w[side]), V.taps.ones, V.taps.out[side],
                                 V.taps.n, V.taps.k, u % V.taps.n);
        }
        if (!all_wait(c)) return false;
    }
    // the k and v convs
    const int ck = conv_c(V.conv[0]);
    conv_phase(V.conv[0], ck, 0);
    conv_phase(V.conv[1], ck, (ck + 127) / 128);
    if (!all_wait(c)) return false;
    // prep8: (24 q heads + 8 k + 8 v) warp units, one a block on warp 0
    const FzPrep& p = V.prep;
    for (int y = blockIdx.x; y < p.heads + 2 * p.kv_heads; y += gridDim.x)
        fz_prep8(y, p.q_raw, p.k_raw, p.v_raw, p.rel_raw, p.q_scale, p.rel_scale, p.rpos, p.rseg,
                 static_cast<const ink2::Seg*>(p.segs), p.q_out, p.rel_out, p.kc, p.vc, p.heads, p.kv_heads, p.ring,
                 p.global, p.qk_clip, p.rel_clip, p.floor_pos, p.alpha, p.eps, p.ksc, p.vsc);
    if (!all_wait(c)) return false;
    const FzAttn& a = V.attn;
    for (int rh = blockIdx.x; rh < a.heads; rh += gridDim.x) fz_rel_table(rh, a.relp, a.rel_proj, a.rtab);
    if (!all_wait(c)) return false;
    // attention: today's grid (1 tile, kv heads, zgrid) as virtual blocks
    for (int u = blockIdx.x; u < a.kv_heads * a.zgrid; u += gridDim.x)
        fz_attn8(make_uint3(0, u / a.zgrid, u % a.zgrid), a.zgrid, a.q, a.kc, a.vc, a.tab,
                 static_cast<const ink2::Seg*>(a.segs), a.tiles, a.part_o, a.part_ml, a.heads, a.kv_heads, a.ring,
                 a.window, a.chunk, 0, a.chunks, a.scale, a.split, a.ksc, a.vsc, a.db, dyn);
    if (!all_wait(c)) return false;
    for (int rh = blockIdx.x; rh < a.heads; rh += gridDim.x)
        fz_merge(rh, a.part_o, a.part_ml, a.sink, a.gate, a.rpos, a.att, a.heads, a.window, a.chunk, 0, a.slots_rh,
                 a.subs);
    if (!all_wait(c)) return false;
    proj_phase(&V.wo, 1, c, 3);
    if (!all_wait(c)) return false;
    // the attention conv into h32, and the boundary copy of hin into its block
    const int ch = conv_c(V.conv[2]);
    conv_phase(V.conv[2], ch, 0);
    if (V.copy_bytes)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V.copy_bytes / 16; i += gridDim.x * blockDim.x)
            static_cast<uint4*>(V.copy_dst)[i] = static_cast<const uint4*>(V.copy_src)[i];
    if (!all_wait(c)) return false;
    // the MLP input
    if (!mix_phase(V.mix_mlp, V.gated, c, dyn)) return false;
    if (V.gated && !gates_phase(V.gate_mlp, 1, c, dyn)) return false;
    // router: a warp a score, then top 8 and the plan on block 0
    const FzMoe& m = V.moe;
    {
        const int w = blockIdx.x * 4 + (threadIdx.x >> 5), W = gridDim.x * 4;
        for (int j = w; j < m.E; j += W) fz_rowdot_warp(j, m.xm, m.router, m.scores, m.E, m.H, 0, 1);
    }
    if (!all_wait(c)) return false;
    if (blockIdx.x == 0) {
        fz_topk_warp(m.scores, m.bias, m.pick, m.rw, m.E, m.K, m.route_scale);
        __syncthreads();
        __threadfence_block();
        fz_plan(m.pick, m.K, m.E, m.T, m.members, m.items, m.counts);
    }
    if (!all_wait(c)) return false;
    fz_nvfp4_up(reinterpret_cast<const __nv_bfloat16*>(m.xm), m.H, m.K, m.up, m.up_scale, m.H / 32, m.width / 32,
                m.items, m.counts, m.members, m.act, m.width, m.gate_cap, m.up_cap, -1);
    if (!all_wait(c)) return false;
    fz_nvfp4_down(reinterpret_cast<const __nv_bfloat16*>(m.act), m.width, 0, m.down, m.down_scale, m.width / 32,
                  m.H / 32, m.items, m.counts, m.members, m.y, m.H, 0.0f, -1);
    if (!all_wait(c)) return false;
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < m.H; d += gridDim.x * blockDim.x)
        fz_combine(m.y, m.rw, m.moe, m.K, m.H, d);
    if (!all_wait(c)) return false;
    conv_phase(V.conv[3], conv_c(V.conv[3]), 0);
    return all_wait(c);
}

}  // namespace

// Visits [first, first + count) of the table. Counters and tickets are zero at launch (the host clears them).
extern "C" __global__ void __launch_bounds__(128) tf_ink2_fused_visit(const FusedVisit* __restrict__ table, int first,
                                                                     int count, FusedState st) {
    extern __shared__ __align__(128) unsigned char dyn[];
    Ctx c{st, nullptr, nullptr, 0, 0};
    for (int v = first; v < first + count; ++v) {
        c.fault = st.fault + v;
        c.st.tk = st.tk + 1024 * v;  // 4 qmmf calls a visit, 256 tiles a call
        c.times = st.times + 64 * v;
        c.ph = 0;
        if (!visit(table[v], c, dyn)) return;
    }
}

// The fused visit's unit of nvfp4_experts.cu (the -O3 sm_120a flags of nvfp4_experts): the decode gate-up and down
// of tf_nvfp4_softcap_half and tf_nvfp4_down_half. half_body takes its units by the real grid (unit = block x 4 +
// warp, stride grid x 4). Each unit is independent and keeps today's code, so the bits do not depend on the grid.
// The decode rings in the dynamic shared memory of the fused kernel (offset 0, free during the expert phases): two
// static rings would add 59 KB of static shared memory beside the attention ring of 68,608 B.
extern __shared__ __align__(128) unsigned char fz_dyn_experts[];
// The plan is written in this launch: a coherent load with fz_fix::kPlanCg (review A1), else today's __ldg (the A/B of
// the check). The block's copy of the switch is set at the entry of each expert phase.
__shared__ int fz_plan_cg;
#define TF_PLAN_LD(p) (fz_plan_cg ? __ldcg(p) : __ldg(p))
#define TF_RING(name, ...) __VA_ARGS__& name = *reinterpret_cast<__VA_ARGS__*>(fz_dyn_experts)
#include "nvfp4_experts.cu"

extern "C" __device__ void fz_nvfp4_up(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                       const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                       const int* __restrict__ items, const int* __restrict__ counts,
                                       const int* __restrict__ members, void* __restrict__ out, int N, float gate_cap,
                                       float up_cap, int skip, int cg) {
    __syncthreads();
    if (threadIdx.x == 0) fz_plan_cg = cg;
    __syncthreads();
    tf_nvfp4_experts::half_body<2, 4, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N,
                                         gate_cap, skip, up_cap);
    __syncthreads();
}

extern "C" __device__ void fz_nvfp4_down(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                         const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                         const int* __restrict__ items, const int* __restrict__ counts,
                                         const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                         int skip, int cg) {
    __syncthreads();
    if (threadIdx.x == 0) fz_plan_cg = cg;
    __syncthreads();
    tf_nvfp4_experts::half_body<1, 0, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N, limit,
                                         skip, 0.f);
    __syncthreads();
}

// review A10: with two blocks a multiprocessor the dynamic shared memory is one K and V tile of tf_ink2_attn8
static_assert(sizeof(tf_nvfp4_experts::Ring<2, 4, uint2>) <= 34816 && sizeof(tf_nvfp4_experts::Ring<1, 4, uint2>) <= 34816,
              "fused.zig dyn2");

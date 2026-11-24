// The planted twins of prefill_experts.cl for the XPU kernel contract: each includes the real source with one fault planted
// and one kernel built, under its own symbol prefix. Gate 3 must turn every one red.

#define PF_PREFIX tf_ctl_pf_plan_
#define PF_ONLY 1
#define PF_PLANT_PLAN
#include "prefill_experts.cl"
#undef PF_PLANT_PLAN
#undef PF_ONLY
#undef PF_PREFIX

#define PF_PREFIX tf_ctl_pf_gather_
#define PF_ONLY 2
#define PF_PLANT_GATHER
#include "prefill_experts.cl"
#undef PF_PLANT_GATHER
#undef PF_ONLY
#undef PF_PREFIX

#define PF_PREFIX tf_ctl_pf_nvfp4_gu_
#define PF_ONLY 4
#define PF_PLANT_UP
#include "prefill_experts.cl"
#undef PF_PLANT_UP
#undef PF_ONLY
#undef PF_PREFIX

#define PF_PREFIX tf_ctl_pf_nvfp4_down_
#define PF_ONLY 8
#define PF_PLANT_PHI
#include "prefill_experts.cl"
#undef PF_PLANT_PHI
#undef PF_ONLY
#undef PF_PREFIX

#define PF_PREFIX tf_ctl_pf_mxfp8_gu_
#define PF_ONLY 16
#define PF_PLANT_UP
#include "prefill_experts.cl"
#undef PF_PLANT_UP
#undef PF_ONLY
#undef PF_PREFIX

#define PF_PREFIX tf_ctl_pf_mxfp8_down_
#define PF_ONLY 32
#define PF_PLANT_SCALE
#include "prefill_experts.cl"
#undef PF_PLANT_SCALE
#undef PF_ONLY
#undef PF_PREFIX

#define PF_PREFIX tf_ctl_pf_combine_
#define PF_ONLY 64
#define PF_PLANT_ORDER
#include "prefill_experts.cl"
#undef PF_PLANT_ORDER
#undef PF_ONLY
#undef PF_PREFIX

// The planted twins of experts.cl for the XPU kernel contract: each includes the real source with one fault planted
// and one kernel built, under its own symbol prefix. Gate 3 must turn every one red.

#define INK2_PREFIX tf_ctl_i2_nvfp4_gu_
#define INK2_ONLY 1
#define INK2_PLANT_UP
#include "experts.cl"
#undef INK2_PLANT_UP
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_nvfp4_down_
#define INK2_ONLY 2
#define INK2_PLANT_PHI
#include "experts.cl"
#undef INK2_PLANT_PHI
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_mxfp8_gu_
#define INK2_ONLY 4
#define INK2_PLANT_UP
#include "experts.cl"
#undef INK2_PLANT_UP
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_mxfp8_down_
#define INK2_ONLY 8
#define INK2_PLANT_SCALE
#include "experts.cl"
#undef INK2_PLANT_SCALE
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_nvfp4_quantize_
#define INK2_ONLY 16
#define INK2_PLANT_NIBBLE
#include "experts.cl"
#undef INK2_PLANT_NIBBLE
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_mxfp8_quantize_
#define INK2_ONLY 32
#define INK2_PLANT_MX_QUANT
#include "experts.cl"
#undef INK2_PLANT_MX_QUANT
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_combine_
#define INK2_ONLY 64
#define INK2_PLANT_ORDER
#include "experts.cl"
#undef INK2_PLANT_ORDER
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_plan_
#define INK2_ONLY 128
#define INK2_PLANT_PLAN
#include "experts.cl"
#undef INK2_PLANT_PLAN
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_nvfp4_gu_prompt_
#define INK2_ONLY 256
#define INK2_PLANT_UP
#include "experts.cl"
#undef INK2_PLANT_UP
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_nvfp4_down_prompt_
#define INK2_ONLY 512
#define INK2_PLANT_PHI
#include "experts.cl"
#undef INK2_PLANT_PHI
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_mxfp8_gu_prompt_
#define INK2_ONLY 1024
#define INK2_PLANT_UP
#include "experts.cl"
#undef INK2_PLANT_UP
#undef INK2_ONLY
#undef INK2_PREFIX

#define INK2_PREFIX tf_ctl_i2_mxfp8_down_prompt_
#define INK2_ONLY 2048
#define INK2_PLANT_SCALE
#include "experts.cl"
#undef INK2_PLANT_SCALE
#undef INK2_ONLY
#undef INK2_PREFIX


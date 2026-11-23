// The kernel contract's planted controls for Mini's dense projections and norm on XPU: dense.cl with one arithmetic
// mistake in each kernel, under the names tf_ctl_ink2_*. The build makes two images from this file.
//
// -DDENSE_CTL=1: every store truncates to bf16 instead of rounding to nearest even (gate 3 must catch it).
// -DDENSE_CTL=2: the matvec keeps its running sums in fp16 (gate 4 must catch it), and the norm drops its eps
//                (gate 3 must catch it on a row whose mean square is near 1e-6).

#define DENSE_NAME(x) tf_ctl_ink2_##x

#if DENSE_CTL == 1
#define DENSE_STORE(v) ((ushort)(as_uint(v) >> 16))
#define NORM_STORE(v) ((ushort)(as_uint(v) >> 16))
#elif DENSE_CTL == 2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define DENSE_ADD(acc, part) ((float)convert_half_rte((acc) + (part)))
#define NORM_EPS 0.0f
#else
#error "set -DDENSE_CTL=1 or 2"
#endif

#include "dense.cl"

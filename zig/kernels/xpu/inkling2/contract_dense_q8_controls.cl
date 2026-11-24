// The kernel contract's planted controls for Mini's Q8_0-weight dense and head kernels on XPU: dense_q8.cl with one
// arithmetic mistake in each kernel, under the names tf_ctl_ink2_*. The build makes three images from this file.
//
// -DQ8_CTL=1: every store truncates to bf16 instead of rounding to nearest even (gate 3 must catch it).
// -DQ8_CTL=2: the lane keeps its running sums in fp16 (gate 4 must catch it).
// -DQ8_CTL=3: every block of a plane uses the plane's first scale (gate 3 must catch it).

#define Q8_NAME(x) tf_ctl_ink2_##x

#if Q8_CTL == 1
#define Q8_STORE(v) ((ushort)(as_uint(v) >> 16))
#define Q8_CAP_ROUND(v) as_float(as_uint(v) & 0xffff0000u)
#elif Q8_CTL == 2
#define Q8_ACC(acc, s, part) ((float)convert_half_rte(fma((s), (part), (acc))))
#elif Q8_CTL == 3
#define Q8_SCALE(s, i) ((s)[0])
#else
#error "set -DQ8_CTL=1, 2 or 3"
#endif

#include "dense_q8.cl"

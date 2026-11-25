// The kernel contract's planted controls for Mini DFlash 2 on XPU: dflash2.cl with one arithmetic mistake in each of its
// six kernels, under the names tf_ctl_ink2_dflash2_*. Each mistake must turn gate 3 red at that kernel's shapes.

#define DF_NAME(x) tf_ctl_ink2_dflash2_##x

// conv: the dynamic part of the kernel is dropped, only the learned taps remain.
#define DF_CONV_KERNEL(base, dyn) (base)
// block_qk and context_kv: the upper half of the rotate-half swaps its two terms.
#define DF_QK_ROPE_HI(x, other, cs, sn) ((x) * (sn) + (other) * (cs))
#define DF_CTX_ROPE_HI(x, other, cs, sn) ((x) * (sn) + (other) * (cs))
// attention: a block row keeps all of the context instead of its window of 2,048 positions.
#define DF_FIRST(from, position, row) (from)
// merge: the chunks are weighted as if they all had the same maximum.
#define DF_MERGE_WEIGHT(m, mx) 1.0f
// topk: the kept values are halved (the order stays, the values are wrong).
#define DF_TOPK_VALUE(v) ((v) * 0.5f)

#include "dflash2.cl"

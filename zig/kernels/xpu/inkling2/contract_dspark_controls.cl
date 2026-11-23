// The kernel contract's planted controls for Mini DSpark on XPU: dspark.cl with one arithmetic mistake in each of its
// eleven kernels, under the names tf_ctl_ink2_dspark_*. Each mistake must turn gate 3 red at that kernel's shapes.

#define DSP_NAME(x) tf_ctl_ink2_dspark_##x

// init_rows: every row copies the mask embedding, so row zero loses the anchor.
#define DSP_INIT_SRC(row, anchor, mask) (mask)
// fuse_taps: the softmax weights stay in FP32 instead of rounding to BF16.
#define DSP_ALPHA(x) (x)
// block_qkv and context_kv: the upper half of the rotate-half swaps its two terms.
#define DSP_QKV_ROPE_HI(x, other, cs, sn) ((x) * (sn) + (other) * (cs))
#define DSP_CTX_ROPE_HI(x, other, cs, sn) ((x) * (sn) + (other) * (cs))
// attention: a sliding layer keeps all of the context instead of its last 1,024 tokens.
#define DSP_FIRST(from, local, committed) (from)
// residual_norm: the norm reads the rounded BF16 residual instead of the FP32 sum.
#define DSP_NORM_SRC(sigma) rbf(sigma)
// swiglu: the gate is not passed through its sigmoid.
#define DSP_SILU(gate) (gate)
// kappa: the y part reads the RNN projection's first columns instead of its y columns.
#define DSP_KAPPA_Y_COL 0
// recurrent: the kept state is not rounded to BF16 before it is added.
#define DSP_LEFT(gamma, previous) ((gamma) * (previous))
// base: the raw projection is softcapped as the target's logits are.
#define DSP_BASE_OUT(dot) (30.0f * tanh((dot) / 30.0f))
// logits: the folded bias keeps FP32 instead of rounding to BF16.
#define DSP_BIAS(beta, dot) ((beta) + (dot))

#include "dspark.cl"

// Qwen3.8-27B decode attention over int8 pool pages on the XMX (K1), ported from Sage's attn_decode_dpas
// (decode.cl) to the 27B shape: Q27_GQA (6) query heads of a K/V head, a row group of up to K1_ROWS rows of one
// stream (a verify window's rows, or one row), pages through the stream's page table. Two variants share k1_body.h:
// attn_k1_decode (8 rows a group, 256-key tiles) and attn_k1_decode16 (16 rows a group, 96 columns, 128-key
// tiles: a 16-row window reads each K/V block once). See k1_body.h for the layout and the argument list.
#pragma OPENCL FP_CONTRACT OFF
#include "shape.h"

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }

#if Q27_GQA == 6
// P is f16 with the V scale folded in: at 128k keys a p of 1e-5 times a V scale of 0.02 is 2e-7, below the f16
// normal range (6e-5), so the tail of the softmax loses its bits while l is summed in fp32. P is scaled by
// K1_PSCALE before the half conversion (p x vs at most about 0.5, so 2048 at most) and o divided once at the end.
#define K1_PSCALE 4096.0f

#define K1_FN attn_k1_decode
#define K1_R 8
#define K1_C (8 * Q27_GQA)
#define K1_P ((K1_C + 15) / 16)
#define K1_T 256
#define K1_G 16
#include "k1_body.h"
#undef K1_FN
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G

#define K1_FN attn_k1_decode16
#define K1_R 16
#define K1_C (16 * Q27_GQA)
#define K1_P ((K1_C + 15) / 16)
#define K1_T 128
#define K1_G 8
#include "k1_body.h"
#undef K1_FN
#undef K1_R
#undef K1_C
#undef K1_P
#undef K1_T
#undef K1_G
#endif // Q27_GQA == 6

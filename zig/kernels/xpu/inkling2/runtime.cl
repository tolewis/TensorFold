// Inkling 2 Mini runtime helpers on Intel Xe2 (OpenCL C), the port of kernels/metal/inkling2/runtime.metal.
// Layout rule for the contract's control generator: helpers come first, then one kernel at a time, each starting on
// the line that holds its `void tf_ink2_<name>(` and ending at the first `}` in column 0.

#pragma OPENCL FP_CONTRACT OFF

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) { return intel_convert_bfloat16_as_ushort(f); } // round to nearest even

// Every launch covers `count` values in groups of 256; `boundary` is an attention boundary or a tap channel.
KERNEL(256) void tf_ink2_copy_bf16(global const ushort *x, global ushort *y, uint count, uint boundary) {
  const uint i = get_global_id(0);
  if (i < count) y[i] = x[i];
}

KERNEL(256) void tf_ink2_copy_f32(global const float *x, global float *y, uint count, uint boundary) {
  const uint i = get_global_id(0);
  if (i < count) y[i] = x[i];
}

KERNEL(256) void tf_ink2_widen(global const ushort *x, global float *y, uint count, uint boundary) {
  const uint i = get_global_id(0);
  if (i < count) y[i] = bf2f(x[i]);
}

// boundary 0: t = p + x. boundary b in 1 to 4: t = x, and p is stored as completed block b of its row.
KERNEL(256) void tf_ink2_attention_partial(global const ushort *p, global const ushort *x, global ushort *blocks, global float *t,
                                           uint count, uint boundary) {
  const uint i = get_global_id(0);
  if (i >= count) return;
  const uint row = i / 2304, d = i % 2304;
  t[i] = boundary ? bf2f(x[i]) : bf2f(p[i]) + bf2f(x[i]);
  if (boundary) blocks[(ulong)row * 4 * 2304 + (boundary - 1) * 2304 + d] = p[i];
}

KERNEL(256) void tf_ink2_mlp_partial(global const float *t, global const ushort *x, global ushort *p, uint count, uint boundary) {
  const uint i = get_global_id(0);
  if (i < count) p[i] = f2bf(t[i] + bf2f(x[i]));
}

KERNEL(256) void tf_ink2_capture_tap(global const ushort *x, global ushort *y, uint count, uint boundary) {
  const uint i = get_global_id(0);
  if (i >= count) return;
  const uint row = i / 2304, d = i % 2304;
  y[(ulong)row * 4 * 2304 + boundary * 2304 + d] = x[i];
}

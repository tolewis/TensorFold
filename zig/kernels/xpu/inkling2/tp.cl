// The production add (glue_attn.cl tp_add, e5506ae) sums fp16. Inkling partials are bf16, so this twin
// uses the same fp32 sum and one rounding, with the bf16 conversion the family kernels already use.
static inline float bf_to_f(ushort v) { return as_float((uint)v << 16); }
static inline ushort f_to_bf(float v) { return intel_convert_bfloat16_as_ushort(v); }

__kernel void tp_add(__global ushort *out, __global const ushort *a, __global const ushort *b, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = f_to_bf(bf_to_f(a[i]) + bf_to_f(b[i]));
}

__kernel void tp_up(__global float *out, __global const ushort *a, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = bf_to_f(a[i]);
}

__kernel void tp_add_f32(__global float *out, __global const float *a, __global const float *b, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = a[i] + b[i];
}

__kernel void tp_down(__global ushort *out, __global const float *a, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = f_to_bf(a[i]);
}

// Zero the route weight when the expert is not in this rank's half. The expert kernels are not changed.
__kernel void tp_zero_unowned(__global int *ids, __global float *weights, int routes, int base, int count) {
  const int i = get_global_id(0);
  if (i >= routes) return;
  const int id = ids[i];
  if (id < base || id >= base + count) weights[i] = 0.0f;
}

// Keep plan items this rank holds and rewrite the expert id to the packed table. Items past the new count are unused.
__kernel void tp_filter_items(__global int *items, __global int *item_count, int base, int nlocal) {
  if (get_global_id(0) != 0) return;
  const int n = item_count[0];
  int kept = 0;
  for (int i = 0; i < n; i++) {
    const int e = items[3 * i];
    if (e < base || e >= base + nlocal) continue;
    items[3 * kept] = e - base;
    items[3 * kept + 1] = items[3 * i + 1];
    items[3 * kept + 2] = items[3 * i + 2];
    kept++;
  }
  item_count[0] = kept;
}

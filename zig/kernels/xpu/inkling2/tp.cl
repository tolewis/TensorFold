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

// One launch: promote the local bf16 row, add the peer's f32 row, store bf16. The up and down launches are gone.
__kernel void tp_add_f32(__global ushort *out, __global const ushort *local_bf, __global const float *peer_f32, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = f_to_bf(bf_to_f(local_bf[i]) + peer_f32[i]);
}

__kernel void tp_down(__global ushort *out, __global const float *a, int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = f_to_bf(a[i]);
}

// Compact each row to this rank's experts. The id buffer then holds only local ids, and counts[row] is how many.
// A dropped slot is local id 0 with weight 0, so a kernel that still walks eight slots never reads a foreign expert.
__kernel void tp_keep_local(__global int *ids, __global float *weights, __global int *counts, int rows, int base, int nlocal) {
  const int row = get_global_id(0);
  if (row >= rows) return;
  int kept = 0;
  int loc[8];
  float w[8];
  for (int j = 0; j < 8; ++j) {
    const int id = ids[row * 8 + j];
    if (id >= base && id < base + nlocal) {
      loc[kept] = id - base;
      w[kept] = weights[row * 8 + j];
      kept++;
    }
  }
  for (int j = 0; j < 8; ++j) {
    if (j < kept) {
      ids[row * 8 + j] = loc[j];
      weights[row * 8 + j] = w[j];
    } else {
      ids[row * 8 + j] = nlocal;
      weights[row * 8 + j] = 0.0f;
    }
  }
  counts[row] = kept;
}

// Write the indirect launch size for one row: X is the tile count, Y is this rank's kept routes.
__kernel void tp_set_launch(__global uint *gc, __global const int *counts, uint tiles) {
  gc[0] = tiles;
  const uint y = (uint)counts[0];
  gc[1] = y == 0 ? 1 : y;
  gc[2] = 1;
}

// A foreign route keeps its id and loses its weight. Combine then adds nothing and does not read the row.
__kernel void tp_zero_foreign_weight(__global const int *ids, __global float *weights, int routes, int base, int nlocal) {
  const int i = get_global_id(0);
  if (i >= routes) return;
  const int id = ids[i];
  if (id < base || id >= base + nlocal) weights[i] = 0.0f;
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

// EXL3 (ExLlamaV3 trellis) projections on Xe2: a port of zig/kernels/metal/core/exl3_projection.metal.
// y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh, H the 128x128 Sylvester Hadamard / sqrt(128).
// Model-agnostic: a launch covers many groups through a descriptor table. Codebook mul1 only; 2 to 8 bits.
// The format is docs/recipes/exl3.md; zig/src/core/exl3_format.zig is the scalar reference.
//
// Metal runs SIMD-groups of 32. Here a sub-group is 16 lanes (Xe2 native). The mapping:
// - Hadamard: Metal lane l holds indices l + 32 j (j < 4), five shuffle stages and two in the lane.
//   Here lane l holds l + 16 j (j < 8), four shuffle stages and three in the lane. The butterfly is the same.
// - Matvec: Metal decodes a 16x16 tile into threadgroup memory (8 values a lane) and then multiplies.
//   Here lane c decodes the 16 values of tile column c straight into registers. No local memory, no barrier.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16
#ifndef ROWS
#define ROWS 1
#endif
#define INV_SQRT128 0.08838834764831845f

// One group: offsets in bytes from the base pointer of the launch (Level Zero passes any device address
// as base and stores ptr - base, modulo 2^64). x is bf16 [rows][x_stride]; rot fp32 [rows][k] (fp16 lost 1e-3 on Sage down_proj);
// sums fp32 [rows][n]; out fp32 or bf16 [rows][out_stride].
typedef struct {
    ulong trellis, suh, svh, x, rot, sums, out;
    uint k, n, halves, rows;
    uint x_stride, out_stride, out_f32, pad;
} exl3_desc;

#define AT(T, off) ((__global T *)(base + (off)))

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// Lane l holds indices l + 16 j. Four cross-lane stages, then three in the lane; fixed FP32 order.
inline void hadamard128(float v[8], uint lane) {
    for (uint d = 1; d < SG; d *= 2) {
        for (uint j = 0; j < 8; j++) {
            const float peer = sub_group_shuffle_xor(v[j], d);
            v[j] = (lane & d) ? peer - v[j] : v[j] + peer;
        }
    }
    for (uint d = 1; d < 8; d *= 2) {
        for (uint j = 0; j < 8; j++) {
            if (j & d) continue;
            const float a = v[j], b = v[j + d];
            v[j] = a + b;
            v[j + d] = a - b;
        }
    }
}

// Two little-endian int16 words make one stream word, read from its most significant bit.
inline uint stream_word(__global const ushort *tile, uint i) { return (uint)tile[2 * i] | ((uint)tile[2 * i + 1] << 16); }

// The window of value p as (word, bit offset, next word): the 16 stream bits that end at E(p), wrapping.
inline uint state_map(uint halves, uint p) {
    const uint ring = 128 * halves;
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    const uint start = (end + ring - 16) % ring;
    const uint wi = start / 32, off = start % 32, next = wi + 1 == ring / 32 ? 0 : wi + 1;
    return wi | (off << 6) | (next << 11);
}

inline uint mapped_state(__global const ushort *tile, uint map) {
    const uint wi = map & 63u, off = (map >> 6) & 31u;
    const uint lo = stream_word(tile, wi);
    const uint hi = stream_word(tile, map >> 11);
    const uint carry = off ? hi >> (32 - off) : 0u;
    return ((lo << off) | carry) >> 16;
}

// mul1: x = s * 0x83DCD12D; fp16((1024 + the bytes of x) * fp16(0x1EEE) + fp16(0xC931)). The FP32 fma is exact;
// the one rounding is to fp16. IGC (driver 26.05) folds (float)convert_half_rte(f) to f, so the rounding is done
// on the bits: round to nearest even at 11 significant bits. This is the fp16 rounding because every mul1 value
// is a normal fp16 (0.0018 <= |v| <= 3.5). exl3_mul1_check proves all 65536 states.
inline float mul1_value(uint s) {
    const uint x = s * 0x83DCD12Du;
    const uint h = 1024u + (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
    const uint u = as_uint(fma((float)h, as_float(0x3BDDC000u), as_float(0xC1262000u)));
    return as_float((u + 0xFFFu + ((u >> 13) & 1u)) & 0xFFFFE000u);
}

// Lane c owns tile column c. Its 16 stream values are p = 32 (c % 8) + 8 m + 4 (c / 8) + i (m, i < 4),
// at tile row 2 m + (i & 1) + 8 (i >> 1). This is the inverse of exl3_format.tilePlacement.
inline uint lane_value(uint c, uint v) { return 32 * (c % 8) + 8 * (v / 4) + 4 * (c / 8) + (v % 4); }
inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

// Qualification only: the 256 states of each tile in stream order, and W_q fp16 [K, N] by placement.
// Grid (tiles_n, tiles_k), one sub-group a work-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_decode_tiles(__global const ushort *trellis, __global uint *states, __global half *wq, uint halves) {
    const uint tn = get_group_id(0), tk = get_group_id(1), c = get_sub_group_local_id();
    const uint tiles_n = get_num_groups(0), ti = tk * tiles_n + tn;
    __global const ushort *tile = trellis + (ulong)ti * 8 * halves;
    for (uint v = 0; v < 16; v++) {
        const uint p = lane_value(c, v);
        const uint s = mapped_state(tile, state_map(halves, p));
        states[(ulong)ti * 256 + p] = s;
        vstore_half(mul1_value(s), ((ulong)tk * 16 + lane_row(v)) * tiles_n * 16 + tn * 16 + c, wq);
    }
}

// Qualification only: every mul1 state's value as the matvec uses it (fp32), to compare with the float64 host table.
__kernel void exl3_mul1_check(__global float *out) {
    const uint i = get_global_id(0);
    out[i] = mul1_value(i);
}

// rot[row][k] = H_K (x * suh) for one 128 block. Grid (K_max / 128, ROWS_max, groups).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_in(__global const uchar *base, __global const exl3_desc *desc) {
    const exl3_desc d = desc[get_group_id(2)];
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    if (blk * 128 >= d.k || row >= d.rows) return;
    __global const ushort *x = AT(const ushort, d.x) + (ulong)row * d.x_stride;
    __global const half *suh = AT(const half, d.suh);
    float v[8];
    for (uint j = 0; j < 8; j++) {
        const uint i = blk * 128 + lane + SG * j;
        v[j] = bf(x[i]) * vload_half(i, suh);
    }
    hadamard128(v, lane);
    __global float *rot = AT(float, d.rot) + (ulong)row * d.k;
    for (uint j = 0; j < 8; j++) rot[blk * 128 + lane + SG * j] = v[j] * INV_SQRT128;
}

// sums[row][n] = rot[row] @ W_q[:, n] for one 16-column strip and up to ROWS rows (built per ROWS).
// Grid (N_max / 16 * SG, groups); one sub-group a strip. FP32 fma accumulation.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_matvec(__global const uchar *base, __global const exl3_desc *desc) {
    const exl3_desc d = desc[get_group_id(1)];
    const uint tn = get_group_id(0), c = get_sub_group_local_id();
    if (tn * 16 >= d.n) return;
    if (d.halves < 4 || d.halves > 16) return;  // the host refuses these; never decode garbage
    const uint tiles_n = d.n / 16, words = 8 * d.halves;
    uint maps[16];
    for (uint v = 0; v < 16; v++) maps[v] = state_map(d.halves, lane_value(c, v));
    __global const ushort *tile = AT(const ushort, d.trellis) + (ulong)tn * words;
    __global const float *rot = AT(const float, d.rot);
    const ulong tile_k = (ulong)tiles_n * words;
    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
    for (uint kb = 0; kb < d.k; kb += 16, tile += tile_k) {
        float xr[ROWS];
        for (uint r = 0; r < ROWS; r++) xr[r] = r < d.rows ? rot[(ulong)r * d.k + kb + c] : 0.0f;
        for (uint v = 0; v < 16; v++) {
            const float w = mul1_value(mapped_state(tile, maps[v]));
            const uint kr = lane_row(v);
            for (uint r = 0; r < ROWS; r++) acc[r] = fma(sub_group_broadcast(xr[r], kr), w, acc[r]);
        }
    }
    __global float *sums = AT(float, d.sums);
    for (uint r = 0; r < ROWS; r++)
        if (r < d.rows) sums[(ulong)r * d.n + tn * 16 + c] = acc[r];
}

// out[row][n] = (H_N sums) * svh, fp32 or bf16, for one 128 block. Grid (N_max / 128, ROWS_max, groups).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_out(__global uchar *base, __global const exl3_desc *desc) {
    const exl3_desc d = desc[get_group_id(2)];
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    if (blk * 128 >= d.n || row >= d.rows) return;
    __global const float *sums = AT(const float, d.sums) + (ulong)row * d.n;
    __global const half *svh = AT(const half, d.svh);
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = sums[blk * 128 + lane + SG * j];
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++) {
        const uint i = blk * 128 + lane + SG * j;
        const float y = v[j] * INV_SQRT128 * vload_half(i, svh);
        const ulong at = (ulong)row * d.out_stride + i;
        if (d.out_f32) AT(float, d.out)[at] = y;
        else AT(ushort, d.out)[at] = to_bf(y);
    }
}

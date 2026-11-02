// EXL3 rows-1 4-bit matvec, second decode form (lane p31-mv3). Behind XPU_EXL3_MV3, default off.
//
// Same descriptor table, same scratch contract, same rotate_in / rotate_out, same sums layout as exl3.cl, so the
// engine can swap only this kernel. Numerics are the base kernel's: the same window for each value, the same
// lane_row placement, the same mul1 rounding, the same hi/lo activation split and the same DPAS order.
//
// The one change is the trellis extraction. A lane's 16 windows are four groups of four, four bits apart, inside
// one 64-bit stream word pair. The base kernel does a 32-bit shl/shr/or funnel and an h window shift for every
// value. Here the h offset folds into one 64-bit shift a group (36 for h 0, 20 for h 1, both fit in the pair at
// index m), so a group costs one runtime 64-bit shr and three constant 64-bit shr, and each window is a 16-bit
// sub-register read. That is 17 shr a tile against 51 shl/shr/or in the base.
//
// The trellis is loaded as four aligned 64-bit words a lane (wbase is a multiple of 4 words, the ring is 32 words,
// so no pair straddles the ring end) instead of seven 32-bit words.
//
// Rows 1, BITS 4, NS 1, KU 1, no fused rotate. Any other shape stays on exl3_matvec.

#define SG 16
#ifndef BITS
#define BITS 4
#endif
#if BITS != 4
#error exl3_mv3 is BITS 4 only
#endif
#define RINGW (8 * BITS)                              // uint32 stream words a tile
#define OFF0 (BITS + 16)
#define SHB ((4 * BITS) % 32)
#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)

// The table row of exl3.cl (exl3_desc), byte for byte.
typedef struct {
    ulong trellis, suh, svh, x, rot, sums, out;
    uint k, n, halves, rows;
    uint x_stride, out_stride, out_type, x_f16;
} exl3_desc;

inline half mul1_half(uint s) {
    // 0x6400 is fp16 1024 and the sum is below 1024, so the dp4a result is already the fp16 bits of 1024 + sum.
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}

// Lane c owns tile column c. Its 16 stream values are p = 32 (c % 8) + 8 m + 4 (c / 8) + i (m, i < 4),
// at tile row 2 m + (i & 1) + 8 (i >> 1).
inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_matvec3(__global const uchar *base, __global const exl3_desc *desc, uint ks) {
    const exl3_desc d = desc[get_group_id(1)];
    const uint tn = get_global_id(0) / SG, c = get_sub_group_local_id();
    if (tn * 16 >= d.n || d.halves != 2 * BITS || d.rows != 1) return;
    const uint g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;  // a multiple of 4 words
    __global const uint *tile = (__global const uint *)((__global const ushort *)d.trellis + (ulong)tn * 16 * BITS);
    __global const float *rot = (__global const float *)d.rot;
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    const uint split = get_group_id(2), kspan = d.k / ks, k0 = split * kspan;
    tile += (ulong)(k0 / 16) * tile_k;
    float8 acc = 0.0f;
    // The DPAS A operand is eight halves a row block and only row 0 is real at rows 1, so the seven zero slots are
    // built once here and only slot 0 is written in the loop.
    short8 ah, al;
    for (uint r = 0; r < 8; r++) {
        ah[r] = 0;
        al[r] = 0;
    }
    // h 0 needs the window 16 bits later than h 1. Both fit in the 64-bit pair at index m, so one runtime shift
    // covers both and every window offset stays a constant.
    const uint sh = (OFF0 + 16) - SHB * h;
    for (uint kb = k0; kb < k0 + kspan; kb += 16, tile += tile_k) {
        ulong p[4];
        for (uint j = 0; j < 4; j++) p[j] = *(__global const ulong *)(tile + ((wbase + 2 * j) % RINGW));
        const float xv = rot[(ulong)kb + c];
        const half hv = convert_half_rte(xv);
        ah[0] = as_short(hv);
        al[0] = as_short(convert_half_rte(xv - (float)hv));
        int8 b;
        for (uint m = 0; m < 4; m++) {
            const ulong u = (m & 1) ? ((p[m >> 1] >> 32) | (p[(m >> 1) + 1] << 32)) : p[m >> 1];
            const ulong y = u >> sh;
            const half v0 = mul1_half((uint)(ushort)(y));
            const half v1 = mul1_half((uint)(ushort)(y >> 4));
            const half v2 = mul1_half((uint)(ushort)(y >> 8));
            const half v3 = mul1_half((uint)(ushort)(y >> 12));
            ((int *)&b)[m] = as_int((half2)(v0, v1));      // tile rows 2m, 2m + 1
            ((int *)&b)[m + 4] = as_int((half2)(v2, v3));  // tile rows 2m + 8, 2m + 9
        }
        acc = DPH(ah, b, acc);
        acc = DPH(al, b, acc);
    }
    __global float *sums = (__global float *)d.sums + (ulong)split * d.n;
    sums[(ulong)(tn * 16 + c)] = acc[0];
}

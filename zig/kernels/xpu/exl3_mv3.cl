// EXL3 rows-1 decode design study (lane p31-mvdesign). NOT wired into the engine: exl3.zig still loads
// exl3.cl. This file exists so ocloc can be asked what the instruction count of the proposed matvec is.
//
// It computes the same thing as exl3_matvec at ROWS 1, NS 1, KU 1, with three changes:
//
// 1. Grouped funnel extraction. A lane's 16 windows are 4 groups of 4 windows at 4-bit spacing inside one
//    64-bit stream word pair. The base kernel does a 32-bit shl/shr/or funnel and an h-window shift for every
//    value (51 instructions a tile in the b4 r1 n1 image). Here the h offset is folded into one 64-bit shift
//    per group (sh = OFF0 + 16 - SHB h) and the four windows come out as three 64-bit shr and one free
//    sub-register read (16 instructions a tile).
// 2. The rotated input is pre-split. The base kernel converts 8 activations to fp16 hi and lo every tile
//    (about 16 to 24 instructions). The rotate kernel writes the hi and lo planes in lane-major order
//    [blk][lane][j] (the GEMM path already stores two planes, GEMM_LO 1), so the matvec loads 4 dwords a plane
//    (2 block loads) instead of converting.
// 3. MV3_LUT 1 replaces the mul1 codebook step (mul by 0x83DCD12D, dp4a, fp16 fma = 3 instructions a value,
//    48 a tile) with one 16-bit table read from a 65536-entry fp16 table. MV3_LUT 0 keeps the dp4a form.
//
// BITS 4 only. At BITS 6 the word base of a group differs between h 0 and h 1, so that case stays on the
// per-value 32-bit funnel (and still gets 2 and 3).
//
// Numerics: the window, the lane_row placement, the mul1 rounding and the DPAS order are the base kernel's.
// The LUT must be built on the host with the same fp16 rounding (the base's exl3_mul1_check kernel prints the
// whole table for exactly this comparison).

#define SG 16
#ifndef BITS
#define BITS 4
#endif
#ifndef MV3_LUT
#define MV3_LUT 0
#endif
#define RINGW (8 * BITS)                              // uint32 stream words a tile
#define OFF0 (BITS + 16)
#define SHB ((4 * BITS) % 32)
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)
#define LOADW (USEW + 1)
#define INV_SQRT128 0.08838834764831845f
#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)

// trellis, hi and lo planes (lane-major fp16), sums, out, the mul1 table; then the shape.
typedef struct {
    ulong trellis, hi, lo, sums, out, lut;
    uint k, n, halves, rows;
    uint out_stride, out_type;
} mv3_desc;

inline half mul1_half(uint s) {
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}
inline half mul1_tab(uint s, __global const half *lut) { return lut[s]; }

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_matvec3(__global const mv3_desc *desc, uint ks) {
    const mv3_desc d = desc[get_group_id(1)];
    const uint tn = get_global_id(0) / SG, c = get_sub_group_local_id();
    if (tn * 16 >= d.n || d.halves != 2 * BITS) return;
    const uint g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    __global const uint *tile = (__global const uint *)((__global const ushort *)d.trellis + (ulong)tn * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;   // uint32 words between k tiles
    const uint split = get_group_id(2), kspan = d.k / ks, k0 = split * kspan;
    tile += (ulong)(k0 / 16) * tile_k;
    __global const ushort *hi = (__global const ushort *)d.hi;  // [blk][lane][j], 8 halves a lane
    __global const ushort *lo = (__global const ushort *)d.lo;
#if MV3_LUT
    __global const half *lut = (__global const half *)d.lut;
#endif
    float8 acc = 0.0f;
    // h = 0 needs the window 16 bits later than h = 1. Both fit in the 64-bit word pair at index m
    // (offsets 36..64 and 20..48), so one runtime shift covers both and every window offset stays constant.
    const uint sh = (OFF0 + 16) - SHB * h;
    for (uint kb = k0; kb < k0 + kspan; kb += 16, tile += tile_k) {
        uint raw[LOADW];
        for (uint j = 0; j < LOADW; j++) raw[j] = tile[(wbase + j) % RINGW];
        short8 ah, al;
        {
            const ulong at = ((ulong)(kb / 128) * SG + c) * 8;
            __global const uint *hp = (__global const uint *)(hi + at);
            __global const uint *lp = (__global const uint *)(lo + at);
            for (uint j = 0; j < 4; j++) {
                ((uint *)&ah)[j] = hp[j];
                ((uint *)&al)[j] = lp[j];
            }
        }
        int8 b;
        for (uint m = 0; m < 4; m++) {
            const ulong u = (ulong)raw[m] | ((ulong)raw[m + 1] << 32);
            const ulong y = u >> sh;
            const uint s0 = (uint)(ushort)(y);
            const uint s1 = (uint)(ushort)(y >> 4);
            const uint s2 = (uint)(ushort)(y >> 8);
            const uint s3 = (uint)(ushort)(y >> 12);
#if MV3_LUT
            const half v0 = mul1_tab(s0, lut), v1 = mul1_tab(s1, lut);
            const half v2 = mul1_tab(s2, lut), v3 = mul1_tab(s3, lut);
#else
            const half v0 = mul1_half(s0), v1 = mul1_half(s1);
            const half v2 = mul1_half(s2), v3 = mul1_half(s3);
#endif
            ((int *)&b)[m] = as_int((half2)(v0, v1));      // tile rows 2m, 2m + 1
            ((int *)&b)[m + 4] = as_int((half2)(v2, v3));  // tile rows 2m + 8, 2m + 9
        }
        acc = DPH(ah, b, acc);
        acc = DPH(al, b, acc);
    }
    __global float *sums = (__global float *)d.sums + (ulong)split * d.rows * d.n;
    if (d.rows > 0) sums[(ulong)(tn * 16 + c)] = acc[0];
}

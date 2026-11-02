// EXL3 rows-1 4-bit matvec, second decode form (lane p31-mv3). Behind XPU_EXL3_MV3, default off.
//
// Same descriptor table, same scratch contract, same rotate_in / rotate_out, same sums layout as exl3.cl, so the
// engine can swap only this kernel. Numerics are the base kernel's: the same window for each value, the same
// lane_row placement, the same mul1 rounding, the same hi/lo activation split and the same DPAS order. The
// extraction below was checked against exl3.cl's own funnel for all 16 lanes and all 16 values of random tiles
// (4800 lane cases, no mismatch), both h values.
//
// The one change is how a lane's 16 windows are taken. exl3.cl builds one 32-bit funnel per value (3
// instructions) plus a select for the h half. Here a group of four values is one 64-bit rotate of the two stream
// words it spans: the four windows sit at 4-bit steps of the rotated pair, so three shifts and one 32-bit read
// give all four. That is 4 groups of about 7 instructions against 16 values of 4.
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
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)
#define LOADW (USEW + 1)
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

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_matvec3(__global const uchar *base, __global const exl3_desc *desc, uint ks) {
    const exl3_desc d = desc[get_group_id(1)];
    const uint tn = get_global_id(0) / SG, c = get_sub_group_local_id();
    if (tn * 16 >= d.n || d.halves != 2 * BITS || d.rows != 1) return;
    const uint g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
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
    for (uint kb = k0; kb < k0 + kspan; kb += 16, tile += tile_k) {
        uint raw[LOADW];
        for (uint j = 0; j < LOADW; j++) raw[j] = tile[(wbase + j) % RINGW];
        // the h = 1 half reads the same words shifted 16 bits, as exl3.cl does
        uint wa[USEW];
        for (uint j = 0; j < USEW; j++) wa[j] = h ? (raw[j] << 16) | (raw[j + 1] >> 16) : raw[j];
        const float xv = rot[(ulong)kb + c];
        const half hv = convert_half_rte(xv);
        ah[0] = as_short(hv);
        al[0] = as_short(convert_half_rte(xv - (float)hv));
        half2 bv[8];
        for (uint m = 0; m < 4; m++) {
            const ulong x = (ulong)wa[m] | ((ulong)wa[m + 1] << 32);
            const ulong z = (x << 12) | (x >> 52);
            const half v0 = mul1_half((uint)(ushort)(z >> 8));  // tile row 2m
            const half v1 = mul1_half((uint)(ushort)(z >> 4));  // tile row 2m + 1
            const half v2 = mul1_half((uint)(ushort)z);         // tile row 2m + 8
            const half v3 = mul1_half((uint)(ushort)(wa[m + 1] >> 16)); // tile row 2m + 9
            bv[m] = (half2)(v0, v1);
            bv[m + 4] = (half2)(v2, v3);
        }
        const int8 b = as_int8(*(int8 *)&bv);
        acc = DPH(ah, b, acc);
        acc = DPH(al, b, acc);
    }
    __global float *sums = (__global float *)d.sums + (ulong)split * d.n;
    sums[(ulong)(tn * 16 + c)] = acc[0];
}

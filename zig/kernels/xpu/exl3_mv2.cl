// exl3_matvec for the 1 row window, lane p29-exl3bw (XPU_EXL3_MV2). The trellis tile of a sub-group is staged in
// local memory: each lane loads its WL words of the 128 to 192 byte tile once (16 8 byte loads a tile at 4 bits,
// 128 bytes in flight with no overlapping loads between lanes, against 96 4 byte loads in exl3_matvec), one
// sub-group barrier publishes them, and every strip decodes its window from local memory. A KU step loads
// KU x NS tiles before one barrier, so NS * KU * RINGW * 4 bytes are in flight per sub-group. The rotation of x
// is fused (rows 1): each 128 block of x is rotated once per sub-group, the 64 bytes of rotated x it would re-read
// per k tile at 544 strips drop out. Built per (BITS, NS, KU).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define SG 16
#ifndef BITS
#define BITS 4
#endif
#ifndef NS
#define NS 4
#endif
#ifndef KU
#define KU 2
#endif
#define INV_SQRT128 0.08838834764831845f

// The stream layout and mul1 codebook are the ones of exl3.cl (see it); the format is docs/recipes/exl3.md.
typedef struct {
    ulong trellis, suh, svh, x, rot, sums, out;
    uint k, n, halves, rows;
    uint x_stride, out_stride, out_type, x_f16;
} exl3_desc;

#ifdef EXL3_ABS
#define AT(T, off) ((__global T *)(off))
#else
#define AT(T, off) ((__global T *)(base + (off)))
#endif

inline float bf(ushort v) { return as_float((uint)v << 16); }

inline float load_x(__global const ushort *x, ulong i, uint f16) { return f16 ? vload_half(i, (__global const half *)x) : bf(x[i]); }

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

// mul1: x = s * 0x83DCD12D; fp16((1024 + the bytes of x) * fp16(0x1EEE) + fp16(0xC931)), one rounding.
inline half mul1_half(uint s) {
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}

inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

// The stream windows of exl3.cl, the same macros with the same values for 4 and 6 bits (8 BITS words a tile).
#define RINGW (8 * BITS)
#define OFF0 (BITS + 16)
#define SHW ((4 * BITS) / 32)
#define SHB ((4 * BITS) % 32)
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)
#define LOADW (USEW + SHW + 1)
// The words one lane stages of its tile: the ring divided over the 16 lanes, the last lane the odd remainder.
#define WL ((RINGW + SG - 1) / SG)

// The 16 values of tile column c as the DPAS B operand, from a RINGW word local window. The decode of exl3.cl
// (tile_decode) with its raw array read from local memory.
inline int8 tile_decode_lds(const uint raw[LOADW], uint c) {
    const uint h = c / 8;
    uint w[USEW];
    for (uint j = 0; j < USEW; j++) w[j] = h ? (SHB ? (raw[j + SHW] << SHB) | (raw[j + SHW + 1] >> (32 - SHB)) : raw[j + SHW]) : raw[j];
    half kv[16];
    for (uint v = 0; v < 16; v++) {
        const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
        kv[lane_row(v)] = mul1_half(off ? ((w[wi] << off) | (w[wi + 1] >> (32 - off))) >> 16 : w[wi] >> 16);
    }
    int8 b;
    for (uint j = 0; j < 8; j++) ((int *)&b)[j] = as_int((half2)(kv[2 * j], kv[2 * j + 1]));
    return b;
}

#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)

// sums[0] += rot[0] @ W for this sub-group's NS strips, one row. The rotation fuses from x (fuse) when the K span
// of each split is a whole number of 128 blocks, else rot is read. Grid (n / 16 / NS rounded to the work-group,
// groups, ks), work-group 8 sub-groups of SG lanes.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_mv2(__global const uchar *base, __global const exl3_desc *desc, uint ks, uint fuse) {
    const exl3_desc d = desc[get_group_id(1)];
    const uint tn0 = get_global_id(0) / SG * NS, c = get_sub_group_local_id();
    if (tn0 * 16 >= d.n || d.halves != 2 * BITS) return;
    const uint g = c % 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;  // words between k tiles
    const uint split = get_group_id(2), kspan = d.k / ks, k0 = split * kspan;
    const ulong tile0 = ((ulong)k0 / 16) * tile_k;
    const __global const half *suh = AT(const half, d.suh);
    const __global const ushort *x = AT(const ushort, d.x);
    const __global const float *rot = AT(const float, d.rot);
    __global float *sums = AT(float, d.sums) + (ulong)split * d.n;

    float8 acc[NS] = {0};
    float vr[8];
    __local uint ld[8 * NS * KU * RINGW];
    const uint ldp = (get_local_id(0) / SG) * NS * KU * RINGW;

    for (uint kb = k0; kb < k0 + kspan; kb += 16 * KU) {
        #pragma unroll
        for (uint u = 0; u < KU; u++) {
            const ulong tile = tile0 + (ulong)u * tile_k;
            #pragma unroll
            for (uint s = 0; s < NS; s++) {
                const uint *t = AT(const uint, d.trellis) + tile + (tn0 + s) * RINGW;
                #if BITS == 4
                const uint2 v = *((__global const uint2 *)(t + 2 * c));
                ld[ldp + (u * NS + s) * RINGW + 2 * c] = v.x;
                ld[ldp + (u * NS + s) * RINGW + 2 * c + 1] = v.y;
                #else
                #pragma unroll
                for (uint j = 0; j < WL; j++) ld[ldp + (u * NS + s) * RINGW + WL * c + j] = t[WL * c + j];
                #endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (uint u = 0; u < KU; u++) {
            const uint kbi = kb + 16 * u;
            if (fuse && (kbi & 127) == 0) {
                for (uint j = 0; j < 8; j++) {
                    const uint i = (kbi / 128) * 128 + c + SG * j;
                    vr[j] = load_x(x, i, d.x_f16) * vload_half(i, suh);
                }
                hadamard128(vr, c);
                for (uint j = 0; j < 8; j++) vr[j] *= INV_SQRT128;
            }
            #pragma unroll
            for (uint s = 0; s < NS; s++) {
                const float xr = fuse ? vr[(kbi / 16) & 7] : rot[kbi + c];
                uint raw[LOADW];
                #pragma unroll
                for (uint j = 0; j < LOADW; j++) raw[j] = ld[ldp + (u * NS + s) * RINGW + (wbase + j) % RINGW];
                const int8 b = tile_decode_lds(raw, c);
                short8 ah, al;
                for (uint r = 0; r < 8; r++) {
                    const float f = r < 1 ? xr : 0.0f;
                    const half hv = convert_half_rte(f);
                    ah[r] = as_short(hv);
                    al[r] = as_short(convert_half_rte(f - (float)hv));
                }
                acc[s] = DPH(ah, b, acc[s]);
                acc[s] = DPH(al, b, acc[s]);
            }
        }
    }

    #pragma unroll
    for (uint s = 0; s < NS; s++) {
        const uint col = (tn0 + s) * 16 + c;
        if (col < d.n) sums[col] = acc[s].x;
    }
}

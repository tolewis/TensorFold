// EXL3 prompt GEMM, second form (lane p26-gemm): an A tile staged in local memory for a K range, decoded B
// trellis tiles streamed past it, fp16 DPAS with fp32 accumulate. Prompt rows only (XPU_PROMPT_GEMM2=1).
//
// exl3_gemm (exl3.cl) reads the rotated A plane once for every work-group that covers a column block, so the A
// traffic rises with N / (16 * GEMM_S) and the kernel sits at about 32 percent of the fp16 peak. Here one
// work-group holds G2C columns and G2R rows: the 8 sub-groups split the rows, the A tile of the K step is staged
// into local memory by all 128 lanes once, and the B tiles of the G2C columns are decoded into local memory once
// and shared by the 8 sub-groups. G2C 128 columns a work-group halves the A reads of exl3_gemm and gives the
// sub-groups G2S strips each to reuse the A registers over.
//
// The descriptor is exl3.cl's exl3_desc (EXL3_ABS: offsets are device addresses). sums[row][n] fp32, as
// exl3_gemm writes it, then exl3_rotate_out_rows. Only the hi plane of the rotated input is read (GEMM_LO 0,
// the shipped prompt image); the host keeps this kernel off a GEMM_LO 1 build.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16

typedef struct {
    ulong trellis, suh, svh, x, rot, sums, out;
    uint k, n, halves, rows;
    uint x_stride, out_stride, out_type, x_f16;
} exl3_desc;

#define AT(T, off) ((__global T *)(off))

#ifndef BITS
#error "exl3_gemm2 needs -DBITS"
#endif
#define RINGW (8 * BITS)                              // uint32 stream words a tile
#define OFF0 (BITS + 16)                              // S(32 g) mod 32, for whole bits below 16
#define SHW ((4 * BITS) / 32)                         // the h = 1 shift: whole words
#define SHB ((4 * BITS) % 32)                         //                 and bits
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)  // aligned words the windows touch
#define LOADW (USEW + SHW + 1)
#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)

// mul1: x = s * 0x83DCD12D; fp16((1024 + the bytes of x) * fp16(0x1EEE) + fp16(0xC931)), one rounding.
// Same as exl3.cl (0xSero's exl3xpu method, MIT).
inline half mul1_half(uint s) {
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}

inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

// The lane's LOADW stream words of one tile, then its column c as a DPAS B operand (16 weights in k order).
inline void tile_load(__global const uint *t, uint c, uint raw[LOADW]) {
    const uint g = c % 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    for (uint j = 0; j < LOADW; j++) raw[j] = t[(wbase + j) % RINGW];
}
inline int8 tile_decode(const uint raw[LOADW], uint c) {
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
inline int8 tile_b(__global const uint *t, uint c) {
    uint raw[LOADW];
    tile_load(t, c, raw);
    return tile_decode(raw, c);
}

#ifndef GEMM2_M
#define GEMM2_M 24                        // rows a sub-group: GEMM2_M / 8 DPAS a k tile
#endif
#ifndef GEMM2_S
#define GEMM2_S 8                         // 16-column strips a sub-group, shared by its GEMM2_M rows
#endif
#ifndef GEMM2_KC
#define GEMM2_KC 2                        // k tiles a local-memory step (A and B together)
#endif
#define G2R (8 * GEMM2_M)                 // rows a work-group: 8 sub-groups of GEMM2_M rows
#define G2C (16 * GEMM2_S)                // columns a work-group
#define G2AK (16 * GEMM2_KC)              // k values staged in local memory a step
#define G2CH (G2AK / 8)                   // 8-half chunks a row of the A tile
#define G2JOBS (GEMM2_KC * GEMM2_S)       // tiles decoded a step: one a sub-group slot

// sums[row][n] = rot[row] @ W_q for G2C columns and G2R rows of a group (exl3_rotate_in_h before,
// exl3_rotate_out_rows after). Grid (n_max / G2C, rows_max / G2R, groups), work-group 128.
// Local memory: the A tile of the K step (G2R x G2AK fp16) and the decoded B tiles of the step (G2JOBS).
__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void exl3_gemm2(__global const uchar *base, __global const exl3_desc *desc, uint rows) {
    __local half as[G2R][G2AK];
    __local int8 bs[GEMM2_KC][GEMM2_S][SG];
    exl3_desc d = desc[get_group_id(2)];
    d.rows = min(d.rows, rows);
    const uint q = get_sub_group_id(), c = get_sub_group_local_id(), lid = get_local_id(0);
    const uint tn0 = get_group_id(0) * GEMM2_S, w0 = get_group_id(1) * G2R, m0 = q * GEMM2_M;
    if (tn0 * 16 >= d.n || w0 >= d.rows || d.halves != 2 * BITS) return;  // the whole work-group
    __global const uint *tiles = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    __global const half *hi = AT(const half, d.rot);
    const uint last = d.rows - 1;
    float8 acc[GEMM2_S][GEMM2_M / 8];
    for (uint s = 0; s < GEMM2_S; s++)
        for (uint rb = 0; rb < GEMM2_M / 8; rb++) acc[s][rb] = 0.0f;
    for (uint kc = 0; kc < d.k; kc += G2AK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        // A tile: every lane stages 16-byte chunks of its rows, clamped to the last stored row past the end.
        for (uint t = lid; t < G2R * G2CH; t += 128) {
            const uint lr = t / G2CH, k8 = (t % G2CH) * 8;
            const uint row = min(w0 + lr, last);
            const half8 v = vload8(0, hi + (ulong)row * d.k + kc + k8);
            vstore8(v, 0, (__local half *)&as[lr][k8]);
        }
        // B tiles: each of the 128 lanes decodes one column of one tile, as exl3_gemm does.
        for (uint job = q; job < G2JOBS; job += 8) {
            const uint t = job / GEMM2_S, s = job % GEMM2_S;
            bs[t][s][c] = tile_b(tiles + (ulong)(kc / 16 + t) * tile_k + s * RINGW, c);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t = 0; t < GEMM2_KC; t++) {
            const uint kl = 16 * t;  // k offset inside the staged A tile
            short8 ah[GEMM2_M / 8];
            for (uint rb = 0; rb < GEMM2_M / 8; rb++)
                for (uint r = 0; r < 8; r++) ah[rb][r] = as_short(as[m0 + rb * 8 + r][kl + c]);
            for (uint s = 0; s < GEMM2_S; s++) {
                const int8 b = bs[t][s][c];
                for (uint rb = 0; rb < GEMM2_M / 8; rb++) acc[s][rb] = DPH(ah[rb], b, acc[s][rb]);
            }
        }
    }
    __global float *sums = AT(float, d.sums) + tn0 * 16 + c;
    for (uint s = 0; s < GEMM2_S; s++)
        for (uint rb = 0; rb < GEMM2_M / 8; rb++)
            for (uint r = 0; r < 8; r++) {
                const uint row = w0 + m0 + rb * 8 + r;
                if (row < d.rows) sums[(ulong)row * d.n + s * 16] = acc[s][rb][r];
            }
}

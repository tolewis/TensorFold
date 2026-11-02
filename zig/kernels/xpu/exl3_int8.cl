// EXL3 prompt GEMM on int8 DPAS (W4A8 on decoded trellis tiles), prompt rows only, behind XPU_PROMPT_INT8
// (zig/src/xpu/exl3.zig, encodeRows). The rotated activation plane is quantized to int8 per row and 32-dimension
// group with an fp32 scale, the trellis tiles are decoded and quantized to int8 per column and 32-dimension group
// with an fp32 scale, the products accumulate in int32 and the store applies both scales.
//
// The operand layouts are the ones the other kernels on this device use: an A operand lane l holds row l % 8 and
// K dimensions (l / 8) * 16 .. + 15 (attn_prompt8, k1_body.h), a B operand lane c holds column c and K 0 .. 31
// (the trellis tile decode of exl3.cl), and an accumulator lane c holds column c and rows 0 .. 7.
//
// Scratch layout of a group (exl3.promptInt8ScratchBytes): A8 [rows][k] int8, SA [rows][k / 32] fp32, the rotated
// hi plane [rows][k] fp16 (d.rot), then sums [rows][n] fp32 (d.sums).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define SG 16
#ifndef BITS
#define BITS 4
#endif
#ifndef GEMM_M
#define GEMM_M 24                         // rows a sub-group: GEMM_M / 8 int8 DPAS row blocks
#endif
#ifndef GEMM_S
#define GEMM_S 4                          // 16-column strips a work-group (N divides by 16 * GEMM_S)
#endif
#ifndef GEMM_KC
#define GEMM_KC 8                         // 16-row k tiles a local-memory step: GEMM_KC / 2 int8 DPAS k32 steps
#endif
#define GEMM_WM (8 * GEMM_M)
#define K32 32                            // one int8 DPAS K depth is one activation scale group
#define ALIGN256(x) (((x) + 255ul) & ~255ul)

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

#define RINGW (8 * BITS)                              // uint32 stream words a tile
#define OFF0 (BITS + 16)                              // the window of value 32 g starts here
#define SHW ((4 * BITS) / 32)                         // the h = 1 shift: whole words
#define SHB ((4 * BITS) % 32)                         //                 and bits
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)  // aligned words the windows touch
#define LOADW (USEW + SHW + 1)

// mul1: x = s * 0x83DCD12D; fp16((1024 + the bytes of x) * fp16(0x1EEE) + fp16(0xC931)), one rounding (exl3.cl).
inline half mul1_half(uint s) {
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}
inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

// The lane's LOADW stream words of one tile, then the tile's 16 values of column c in k order (exl3.cl tile_decode).
inline void tile_load(__global const uint *t, uint c, uint raw[LOADW]) {
    const uint g = c % 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    for (uint j = 0; j < LOADW; j++) raw[j] = t[(wbase + j) % RINGW];
}
inline void tile_halves(__global const uint *t, uint c, half kv[16]) {
    uint raw[LOADW];
    tile_load(t, c, raw);
    const uint h = c / 8;
    uint w[USEW];
    for (uint j = 0; j < USEW; j++) w[j] = h ? (SHB ? (raw[j + SHW] << SHB) | (raw[j + SHW + 1] >> (32 - SHB)) : raw[j + SHW]) : raw[j];
    for (uint v = 0; v < 16; v++) {
        const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
        kv[lane_row(v)] = mul1_half(off ? ((w[wi] << off) | (w[wi + 1] >> (32 - off))) >> 16 : w[wi] >> 16);
    }
}

// A8[row][k] int8 = the rotated hi plane quantized, SA[row][k / 32] fp32 its scales. Grid (k / 32, rows, groups) of
// one sub-group: lane l quantizes dimensions kg * 32 + l and kg * 32 + 16 + l of one row.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_int8_quant(__global const uchar *base, __global const exl3_desc *desc, uint rows) {
    const exl3_desc d = desc[get_group_id(2)];
    const uint kg = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    const uint k32 = d.k / K32;
    if (kg >= k32 || row >= min(d.rows, rows)) return;   // one value a work-group: every lane takes the same branch
    // A8 and SA sit before the rotated plane: promptInt8ScratchBytes, with the plan's full row count
    const ulong a8_bytes = ALIGN256((ulong)d.k * d.rows), sa_bytes = ALIGN256((ulong)d.rows * k32 * 4);
    __global char *a8 = AT(char, d.rot - a8_bytes - sa_bytes) + (ulong)row * d.k + kg * K32;
    __global float *sa = AT(float, d.rot - sa_bytes) + (ulong)row * k32 + kg;
    __global const half *hi = AT(half, d.rot) + (ulong)row * d.k + kg * K32;
    const float v0 = (float)hi[lane], v1 = (float)hi[lane + SG];
    const float m = sub_group_reduce_max(fmax(fabs(v0), fabs(v1)));
    const float s = m > 0.0f ? m / 127.0f : 1.0f, inv = 1.0f / s;
    a8[lane] = (char)clamp(rint(v0 * inv), -127.0f, 127.0f);
    a8[lane + SG] = (char)clamp(rint(v1 * inv), -127.0f, 127.0f);
    if (lane == 0) sa[0] = s;
}

// sums[row][n] = A8[row] @ W_q for GEMM_S strips and GEMM_WM rows of a group (exl3_rotate_in_h and
// exl3_int8_quant before, exl3_rotate_out_rows after). The work-group decodes each pair of trellis tiles once into
// local memory as an int8 DPAS B operand with its column scale, and its 8 sub-groups (GEMM_M rows each) all use it.
// Grid (n / 16 / GEMM_S, rows / GEMM_WM, groups), work-group 128.
__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void exl3_int8_gemm(__global const uchar *base, __global const exl3_desc *desc, uint rows) {
    __local int8 bs[GEMM_KC / 2][GEMM_S][SG];    // 32 int8 a lane: 8 ints
    __local float ws[GEMM_KC / 2][GEMM_S][SG];   // that k32 step's scale of the lane's column
    exl3_desc d = desc[get_group_id(2)];
    const uint full = d.rows;
    d.rows = min(d.rows, rows);
    const uint q = get_sub_group_id(), c = get_sub_group_local_id();
    const uint tn0 = get_group_id(0) * GEMM_S, w0 = get_group_id(1) * GEMM_WM, m0 = w0 + q * GEMM_M;
    if (tn0 * 16 >= d.n || w0 >= d.rows || d.halves != 2 * BITS) return;  // the whole work-group
    const uint k32 = d.k / K32;
    const ulong a8_bytes = ALIGN256((ulong)d.k * full), sa_bytes = ALIGN256((ulong)full * k32 * 4);
    __global const char *a8 = AT(const char, d.rot - a8_bytes - sa_bytes);
    __global const float *sa = AT(const float, d.rot - sa_bytes);
    __global const uint *tiles = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    float8 acc[GEMM_S][GEMM_M / 8];
    for (uint s = 0; s < GEMM_S; s++)
        for (uint rb = 0; rb < GEMM_M / 8; rb++) acc[s][rb] = 0.0f;
    for (uint kc = 0; kc < d.k; kc += 16 * GEMM_KC) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint job = q; job < (GEMM_KC / 2) * GEMM_S; job += 8) {
            const uint t = job / GEMM_S, s = job % GEMM_S;
            __global const uint *tk = tiles + (ulong)(kc / 16 + 2 * t) * tile_k + s * RINGW;
            half kv[32];
            tile_halves(tk, c, kv);
            tile_halves(tk + tile_k, c, kv + 16);
            float m = 0.0f;
            for (uint i = 0; i < 32; i++) m = fmax(m, fabs((float)kv[i]));
            const float sc = m > 0.0f ? m / 127.0f : 1.0f, inv = 1.0f / sc;
            int8 bq;
            for (uint j = 0; j < 8; j++) {
                int packed = 0;
                for (uint e = 0; e < 4; e++) packed |= ((int)clamp(rint((float)kv[4 * j + e] * inv), -127.0f, 127.0f) & 0xff) << (8 * e);
                ((int *)&bq)[j] = packed;
            }
            bs[t][s][c] = bq;
            ws[t][s][c] = sc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t = 0; t < GEMM_KC / 2; t++) {
            const uint kg = kc / K32 + t;
            short8 a[GEMM_M / 8];  // 8 rows x 32 int8 of the k32 step: lane l row l % 8, dimensions (l / 8) * 16 ..
            for (uint rb = 0; rb < GEMM_M / 8; rb++)
                intel_sub_group_2d_block_read_16b_8r16x1c((global void *)a8, d.k, d.rows, d.k, (int2)(kg * 16, m0 + rb * 8),
                                                         (private ushort *)&a[rb]);
            for (uint s = 0; s < GEMM_S; s++) {
                const int8 b = bs[t][s][c];
                const float wsc = ws[t][s][c];
                int8 ai[GEMM_M / 8];
                for (uint rb = 0; rb < GEMM_M / 8; rb++) ai[rb] = (int8)(0);
                for (uint rb = 0; rb < GEMM_M / 8; rb++) ai[rb] = intel_sub_group_i8_i8_matrix_mad_k32(a[rb], b, ai[rb]);
                for (uint rb = 0; rb < GEMM_M / 8; rb++)
                    for (uint r = 0; r < 8; r++) {
                        const uint row = m0 + rb * 8 + r;
                        if (row < d.rows) acc[s][rb][r] += (float)ai[rb][r] * sa[(ulong)row * k32 + kg] * wsc;
                    }
            }
        }
    }
    __global float *sums = AT(float, d.sums) + tn0 * 16 + c;
    for (uint s = 0; s < GEMM_S; s++)
        for (uint rb = 0; rb < GEMM_M / 8; rb++)
            for (uint r = 0; r < 8; r++) {
                const uint row = m0 + rb * 8 + r;
                if (row < d.rows) sums[(ulong)row * d.n + s * 16] = acc[s][rb][r];
            }
}

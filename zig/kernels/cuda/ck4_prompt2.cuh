// The NVFP4 prompt GEMM on a 128 x 256 tile: 8 warps of 64 x 64, one barrier a step, weight words transposed in place.
#pragma once

namespace tf_ck4p {

constexpr int QM = 128, QN = 256, QSTAGES = 4;  // rows, columns, staged 64-input steps

// One staged 64-input step: codes with an m8 block's rows in a line for ldmatrix, four 64-column weight tiles by quad.
struct Step2 {
    uint4 a[QM / 8][2][8];
    uint32_t sa[QM];
    uint4 b[512];
    uint32_t sb[4][64];
};
static_assert(sizeof(Step2) == 13824, "ck.zig's prompt4w_shared");

// 16 bytes through L1: a 128-byte line of a row's codes serves four steps.
__device__ __forceinline__ void cp16_l1(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}

// A quad's word v from the dense packer's order to the mxf4nvf4 order: lane t's byte j is nibbles t, t + 4 of lane j.
__device__ __forceinline__ void quad_transpose(uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3) {
    const uint32_t m = 0x0F0F0F0Fu;
    const uint32_t l0 = w0 & m, l1 = w1 & m, l2 = w2 & m, l3 = w3 & m;
    const uint32_t h0 = (w0 >> 4) & m, h1 = (w1 >> 4) & m, h2 = (w2 >> 4) & m, h3 = (w3 >> 4) & m;
    // bytes 0, 1 of u: nibbles (0, 4), (2, 6) of the lane's word. Of v: nibbles (1, 5), (3, 7).
    const uint32_t u0 = l0 | (l0 >> 12), u1 = l1 | (l1 >> 12), u2 = l2 | (l2 >> 12), u3 = l3 | (l3 >> 12);
    const uint32_t v0 = h0 | (h0 >> 12), v1 = h1 | (h1 >> 12), v2 = h2 | (h2 >> 12), v3 = h3 | (h3 >> 12);
    const uint32_t ua = __byte_perm(u0, u1, 0x5140), ub = __byte_perm(u2, u3, 0x5140);
    const uint32_t va = __byte_perm(v0, v1, 0x5140), vb = __byte_perm(v2, v3, 0x5140);
    w0 = __byte_perm(ua, ub, 0x5410);
    w1 = __byte_perm(va, vb, 0x5410);
    w2 = __byte_perm(ua, ub, 0x7632);
    w3 = __byte_perm(va, vb, 0x7632);
}

__device__ __forceinline__ void prompt2(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs,
                                        const uint32_t* __restrict__ w, const uint8_t* __restrict__ ws, float scale,
                                        __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int group) {
    extern __shared__ __align__(128) unsigned char tf_ck4p2_raw[];
    Step2* st = reinterpret_cast<Step2*>(tf_ck4p2_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int NB = npad / 64, KG = K / 64;
    const int2 at = tile_of(blockIdx.x, M, N, QM, QN, group);
    const int m0 = at.x, n0 = at.y;
    // loader roles: row tid / 2 (16 bytes tid % 2) and its scales; a warp, two 512-byte runs of weight tile warp / 2
    const int arow = tid >> 1, half = tid & 1, bt = tid >> 6, bq = tid & 63;
    const int bo = ((tid & 32) << 1) + lane, bl = bt * 128 + bo;
    const int r = m0 + arow < M ? m0 + arow : M - 1;            // rows past M read the last, masked at the epilogue
    const int nbt = n0 / 64 + bt < NB ? n0 / 64 + bt : NB - 1;  // past npad: the last tile, unread
    const uint8_t* xa = codes + static_cast<size_t>(r) * (K / 2) + half * 16;
    const uint8_t* xsr = xs + static_cast<size_t>(r) * (K / 16);
    const uint4* wt = reinterpret_cast<const uint4*>(w) + static_cast<size_t>(nbt) * KG * 128 + bo;
    const uint4* wst = reinterpret_cast<const uint4*>(ws) + static_cast<size_t>(nbt) * KG * 16 + bq;
    // warps w and w + 4 transpose on opposite sides of the mma block: one's integer work runs under the other's mma
    const bool early = wm == 0;
    auto load = [&](int kg) {
        Step2& p = st[kg % QSTAGES];
        cp16_l1(&p.a[arow >> 3][half][arow & 7], xa + kg * 32);
        if (!half) cp4(&p.sa[arow], xsr + kg * 4);
        cp16(&p.b[bl], wt + static_cast<size_t>(kg) * 128);
        cp16(&p.b[bl + 32], wt + static_cast<size_t>(kg) * 128 + 32);
        if (bq < 16) cp16(reinterpret_cast<uint4*>(p.sb[bt]) + bq, wst + static_cast<size_t>(kg) * 16);
    };
    // a thread transposes in place a quad its own warp copied: the warp's wait covers the copy, the barrier the readers
    auto transpose = [&](int kg) {
        Step2& p = st[kg % QSTAGES];
        uint4 q0 = p.b[tid * 2], q1 = p.b[tid * 2 + 1];
        quad_transpose(q0.x, q0.z, q1.x, q1.z);
        quad_transpose(q0.y, q0.w, q1.y, q1.w);
        p.b[tid * 2] = q0;
        p.b[tid * 2 + 1] = q1;
    };
#pragma unroll
    for (int s = 0; s < QSTAGES - 1; ++s) {
        if (s < KG) load(s);
        commit();
    }
    wait<QSTAGES - 2>();
    transpose(0);
    __syncthreads();
    float acc[4][8][4] = {};
    for (int kg = 0; kg < KG; ++kg) {
        const Step2& p = st[kg % QSTAGES];
        const uint2* pb = reinterpret_cast<const uint2*>(p.b) + wn * 256 + lane;
        uint32_t a[4][4], sa[4], sb[8];
        uint2 b[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r0 = wm * 64 + i * 16;
            // ldmatrix: lane l gives row (l % 8) + 8 ((l / 8) % 2) of chunk l / 16
            ldmatrix4(a[i], &p.a[(r0 >> 3) + ((lane >> 3) & 1)][lane >> 4][lane & 7]);
            sa[i] = p.sa[r0 + g + (t & 1) * 8];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = pb[j * 32], sb[j] = p.sb[wn][j * 8 + g];
        // the step's operands are in registers: its copies for step kg + QSTAGES - 1 fill under the mma block
        if (kg + QSTAGES - 1 < KG) load(kg + QSTAGES - 1);
        commit();
        if (early) {
            wait<QSTAGES - 2>();
            if (kg + 1 < KG) transpose(kg + 1);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) mma_fp4(acc[i][j], a[i], b[j].x, b[j].y, sa[i], sb[j]);
        if (!early) {
            wait<QSTAGES - 2>();
            if (kg + 1 < KG) transpose(kg + 1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = m0 + wm * 64 + i * 16 + g + h * 8;
            if (row >= M) continue;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = n0 + wn * 64 + j * 8 + t * 2;
                const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else if (col < N) {
                    dst[0] = __float2bfloat16_rn(v0);
                }
            }
        }
}

}  // namespace tf_ck4p

// As tf_ck4_prompt, on ceil(M / 128) * ceil(N / 256) blocks of 256 threads with QSTAGES * sizeof(Step2) dynamic shared.
extern "C" __global__ void __launch_bounds__(256, 1)
    tf_ck4_prompt2(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, const uint32_t* __restrict__ w,
                   const uint8_t* __restrict__ ws, float scale, __nv_bfloat16* __restrict__ out, int M, int N, int K,
                   int npad, int group) {
    tf_ck4p::prompt2(codes, xs, w, ws, scale, out, M, N, K, npad, group);
}

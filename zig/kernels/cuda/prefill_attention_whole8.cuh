// Whole-window e4m3 row attention with each thread's raw key bytes staged by cp.async: one barrier an item, no
// load a warp waits on. hpattn8_whole_kernel's sums and folds on the same rows. Included inside the namespace.

__device__ __forceinline__ void cpz(void* dst, const void* src, int bytes) {
    const uint32_t d = tf_common::smem(dst);
    if (bytes == 8) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(d), "l"(src));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}

// Dynamic shared bytes: two widened items, the block's query rows, then 8 raw bytes and a scale a thread.
template <int D, int WARPS>
constexpr int whole2_shared = 2 * HALF * (D / 8) * 16 + WARPS * 16 * (D / 8) * 16 + 32 * WARPS * 12;

template <int D, int WARPS, int HPC>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn8_whole2_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                      const uint8_t* __restrict__ vc, const float* __restrict__ ks, const float* __restrict__ vs,
                      __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK, int G, float scale, int CAP) {
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr int NS = 2, TURNS = HALF * C / THREADS;  // widened slots, e4m3 words a thread stages an item
    static_assert(D % 64 == 0 && D > 128 && C == 32 && TURNS == 4 && K16 % 4 == 0 && NB % 8 == 0);
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = slots + NS * HALF * C;
    uint2* raw = reinterpret_cast<uint2*>(qs + WARPS * 16 * C) + threadIdx.x;       // this thread's e4m3 word in flight
    float* raw_s = reinterpret_cast<float*>(qs + WARPS * 16 * C) + 2 * THREADS + threadIdx.x;  // and its key's scale
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = (p0 + last) / BN + 1, items = 4 * tiles, limit = p0 + W;
    const int phase = warp < WARPS / 2 ? 0 : 2;  // half of the warps take their turns two steps after the others
    const uint8_t* kh = kc + static_cast<int64_t>(hk) * CAP * D + lane * 8;
    const uint8_t* vh = vc + static_cast<int64_t>(hk) * CAP * D + lane * 8;
    const float* ksh = ks + static_cast<int64_t>(hk) * CAP;
    const float* vsh = vs + static_cast<int64_t>(hk) * CAP;
    // turn c of an item: word `lane` of key warp + 8 c of its 32, as hpattn8_whole_kernel's threads take them
    auto issue = [&](int item, int c) {
        const int key = (item / 4) * BN + (item % 2) * HALF + warp + 8 * c;
        if (key < limit) {
            cpz(raw, (item % 4 < 2 ? kh : vh) + static_cast<int64_t>(key) * D, 8);
            cpz(raw_s, (item % 4 < 2 ? ksh : vsh) + key, 4);
        } else {
            *raw = make_uint2(0u, 0u);  // past the filled keys: zeros, as the staged slot reads them
            *raw_s = 0.0f;
        }
    };
    // the landed word into its slot: each e4m3 pair to fp32, times the key's scale, to bf16 (hpattn8's put)
    auto widen = [&](int item, int c) {
        const uint2 pre = *raw;
        const float s = *raw_s;
        uint4 w;
        const __nv_fp8x2_storage_t* e = reinterpret_cast<const __nv_fp8x2_storage_t*>(&pre);
        uint32_t* o = reinterpret_cast<uint32_t*>(&w);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 x = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(e[k], __NV_E4M3)));
            o[k] = pack(__fmul_rn(x.x, s), __fmul_rn(x.y, s));
        }
        slots[(item % NS) * HALF * C + sw<8 * C>(warp + 8 * c, lane)] = w;
    };
    // turn c of item it + 1 lands in the slot item it - 1 left, then the next turn's copy starts
    auto turn = [&](int it, int c) {
        if (it + 1 >= items) return;
        cp_wait<0>();
        widen(it + 1, c);
        if (c + 1 < TURNS) issue(it + 1, c + 1);
        else if (it + 2 < items) issue(it + 2, 0);
        commit();
    };
    for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
        const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
        const int hd = hk * G + hg * HPC + w % HPC;
        const __nv_bfloat16* src = q + (static_cast<int64_t>(min(row, W - 1)) * H + hd) * D + (i % C) * 8;
        cp16z(qs + sw<8 * C>(i / C, i % C), src, row < W);
    }
    issue(0, 0);
    commit();
#pragma unroll
    for (int c = 0; c < TURNS; ++c) turn(-1, c);  // item 0, and the first copy of item 1
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    // item it's slot: every thread's part of it is widened, and every warp is done with the other slot
    auto take = [&](int it) -> const uint4* {
        __syncthreads();
        return slots + (it % NS) * HALF * C;
    };
    auto scores = [&](float (&s)[4][4], const uint4* sl, int it) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            if (k % (K16 / 4) == phase * (K16 / 16)) turn(it, k / (K16 / 4));
            uint32_t a[4];
            ldmatrix4(a, qs + sw<8 * C>(warp * 16 + lane % 16, 2 * k + lane / 16));
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<8 * C>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], a, b[0], b[1]);
                    mma0(s[n + 1], a, b[2], b[3]);
                } else {
                    mma(s[n], a, b[0], b[1]);
                    mma(s[n + 1], a, b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](const uint32_t (&pa0)[4], const uint32_t (&pa1)[4], const uint4* sl, int it) {
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            if (i % (NB / 2) == phase * (NB / 16)) turn(it, i / (NB / 2));
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<8 * C>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa0, b[0], b[1]);
            mma(o[i + 1], pa0, b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            if (i % (NB / 2) == phase * (NB / 16)) turn(it, 2 + i / (NB / 2));
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<8 * C>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa1, b[0], b[1]);
            mma(o[i + 1], pa1, b[2], b[3]);
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it), it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1), it + 1);
        {  // the fold of the tile's 64 scores into the running maximum, sum and weighted values
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = c <= pos0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                sc[n][1] = c + 1 <= pos0 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                sc[n][2] = c <= pos1 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                sc[n][3] = c + 1 <= pos1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = act0 && c <= pos0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && c + 1 <= pos0 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && c <= pos1 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && c + 1 <= pos1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        values(pa[0], pa[1], take(it + 2), it + 2);
        values(pa[2], pa[3], take(it + 3), it + 3);
    }
    cp_wait<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < C; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + (static_cast<int64_t>(r0) * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + (static_cast<int64_t>(r1) * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

// Port of wojo's GLM 0110 (PR #124, spill v2: block pack/unpack through a bounded device pack buffer) and 0118 (PR
// #152, a failed or carried destination's rows reset): the one byte mover those features need that no existing kernel
// does. zig/src/families/dsv41/port_kernels.zig launches it.
//
// UNTESTED ON A GPU: written on a Mac without nvcc. zig/src/families/dsv41/port_main.zig (tf-dsv41-port) checks it
// against port_kernels.zig's CPU reference on a Spark; run it before anything relies on this kernel.
//
// Built like dsv41_ops.cu (torch's flags: no half/bf16 operators, --fmad=false --ftz=false): it only moves bytes, so no
// arithmetic flag changes a result. Raw device pointers (a mapped pinned host buffer works too: UVA), the caller's
// stream, no allocation, no host sync; capturable in a CUDA graph (the plane set is passed by value).
//
// What already exists and is used instead (docs/port/kit.md has the full map):
//   dsv41_shard.cu tf_ds_kv_fetch_kernel        the sparse fetch: picked compressed rows (codes + scales) by base + pick,
//                                               the renumbered pick list, and owner decode's mask (shard_ops.fetch)
//   dsv41_shard.cu tf_ds_select_rows_kernel     owner decode's return rows (shard_ops.select)
//   dsv41_shard.cu tf_ds_union_*                the prompt fetch's union: mark, scan, compact, renumber
//   torch_ops/movement.cu tf_gather_rows_kernel dst row i = src row idx[i] (ops.gatherRows)
//   dsv41_ops.cu tf_ds_scatter_rows_kernel      dst row idx[r] = src row r (ops.scatterRows)
//   dsv41_ops.cu tf_ds_copy_rows16_kernel       one strided 2-D copy (ops.copyRows)
//   cuMemcpy*Async / cuMemsetD8Async            spill_state.Packer's per-plane copies (16 a block) and clears
// The kernel below adds the one thing those lack: up to 32 planes x n blocks moved to or from one contiguous staging
// buffer (or reset to their fill bytes) in a single launch, in spill_state's block layout.
#include <cuda_runtime.h>
#include <stdint.h>

// Copies n bytes from s to d with the widest unit all three allow (16, 8, 4 or 1 bytes), lanes `lane` of `lanes`.
__device__ __forceinline__ void tf_ds_port_copy(uint8_t* d, const uint8_t* s, unsigned long long n,
                                                unsigned long long lane, unsigned long long lanes) {
    const unsigned long long a = (unsigned long long)(uintptr_t)d | (unsigned long long)(uintptr_t)s | n;
    if ((a & 15) == 0) {
        uint4* d4 = (uint4*)d;
        const uint4* s4 = (const uint4*)s;
        for (unsigned long long i = lane; i < n / 16; i += lanes) d4[i] = s4[i];
    } else if ((a & 7) == 0) {
        uint2* d2 = (uint2*)d;
        const uint2* s2 = (const uint2*)s;
        for (unsigned long long i = lane; i < n / 8; i += lanes) d2[i] = s2[i];
    } else if ((a & 3) == 0) {
        uint32_t* d1 = (uint32_t*)d;
        const uint32_t* s1 = (const uint32_t*)s;
        for (unsigned long long i = lane; i < n / 4; i += lanes) d1[i] = s1[i];
    } else {
        for (unsigned long long i = lane; i < n; i += lanes) d[i] = s[i];
    }
}

// Sets n bytes at d to `v`, lanes `lane` of `lanes` (16-byte words when d and n allow).
__device__ __forceinline__ void tf_ds_port_set(uint8_t* d, uint8_t v, unsigned long long n, unsigned long long lane,
                                               unsigned long long lanes) {
    if ((((unsigned long long)(uintptr_t)d | n) & 15) == 0) {
        const uint32_t w = 0x01010101u * v;
        const uint4 q = make_uint4(w, w, w, w);
        uint4* d4 = (uint4*)d;
        for (unsigned long long i = lane; i < n / 16; i += lanes) d4[i] = q;
    } else {
        for (unsigned long long i = lane; i < n; i += lanes) d[i] = v;
    }
}

// A plane's rows a staged block holds: rows row0 + b * rows .. + rows of the plane at ptr (rows ld bytes apart,
// row_bytes each) for block b of the call, at stage_off within each staged block (rows packed row_bytes apart).
struct TfDsPortPlane {
    unsigned long long ptr;
    unsigned long long ld;
    unsigned long long row_bytes;
    unsigned long long row0;
    unsigned long long rows;
    unsigned long long stage_off;
    unsigned int fill;
    unsigned int pad;
};
struct TfDsPortPlanes {
    TfDsPortPlane plane[32];
    unsigned int n;
    unsigned int pad;
};

// Planes to and from a contiguous staging buffer (0110's block staging; spill_state's block layout when the set is
// built by port_kernels.spillBlocks): mode 0 packs (plane rows -> staging + b * block_bytes + stage_off), mode 1 unpacks
// (the reverse), mode 2 fills the planes' rows with their fill byte (a failed restore's destination; staging unused).
// Grid (x, planes, n_blocks): a (plane, block) pair's bytes grid-strided over gridDim.x blocks. A packed plane
// (ld == row_bytes: every spill plane) is one contiguous span, moved in the widest unit its addresses and length allow;
// a strided one in 16-byte words when its address, ld, row_bytes, stage_off, block_bytes and staging allow, else bytes.
extern "C" __global__ void tf_ds_port_planes_kernel(TfDsPortPlanes set, uint8_t* staging, unsigned long long block_bytes,
                                                    unsigned int n_blocks, int mode) {
    const unsigned z = blockIdx.y;
    const unsigned b = blockIdx.z;
    if (z >= set.n || b >= n_blocks) return;
    const TfDsPortPlane p = set.plane[z];
    const unsigned long long first = p.row0 + (unsigned long long)b * p.rows;
    uint8_t* plane = (uint8_t*)p.ptr;
    const unsigned long long lanes = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long lane = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long span = p.rows * p.row_bytes;
    if (p.ld == p.row_bytes) {
        uint8_t* rows = plane + first * p.ld;
        if (mode == 2) {
            tf_ds_port_set(rows, (uint8_t)p.fill, span, lane, lanes);
            return;
        }
        uint8_t* stage = staging + (unsigned long long)b * block_bytes + p.stage_off;
        if (mode == 0) tf_ds_port_copy(stage, rows, span, lane, lanes);
        else tf_ds_port_copy(rows, stage, span, lane, lanes);
        return;
    }
    if (mode == 2) {
        for (unsigned long long e = lane; e < span; e += lanes)
            plane[(first + e / p.row_bytes) * p.ld + e % p.row_bytes] = (uint8_t)p.fill;
        return;
    }
    uint8_t* stage = staging + (unsigned long long)b * block_bytes + p.stage_off;
    const unsigned long long a =
        p.ptr | p.ld | p.row_bytes | p.stage_off | block_bytes | (unsigned long long)(uintptr_t)staging;
    if ((a & 15) == 0) {
        const unsigned long long w = p.row_bytes / 16;
        for (unsigned long long e = lane; e < p.rows * w; e += lanes) {
            uint4* pl = (uint4*)(plane + (first + e / w) * p.ld) + e % w;
            uint4* st = (uint4*)stage + e;
            if (mode == 0) *st = *pl;
            else *pl = *st;
        }
        return;
    }
    for (unsigned long long e = lane; e < span; e += lanes) {
        uint8_t* pl = plane + (first + e / p.row_bytes) * p.ld + e % p.row_bytes;
        uint8_t* st = stage + e;
        if (mode == 0) *st = *pl;
        else *pl = *st;
    }
}

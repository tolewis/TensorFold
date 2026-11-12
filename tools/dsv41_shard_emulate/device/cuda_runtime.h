// A syntax-only stand-in for the CUDA runtime header: clang's CUDA front end (no SDK, no NVPTX backend on macOS) checks
// dsv41_shard.cu's device code, the inline PTX included:
//   clang++ -x cuda -nocudainc -nocudalib --cuda-device-only --cuda-gpu-arch=sm_90 -fms-extensions \
//     -I tools/dsv41_shard_emulate/device -fsyntax-only zig/kernels/cuda/dsv41_shard.cu
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __attribute__((always_inline)) inline
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
typedef unsigned long long uint64_t; typedef unsigned uint32_t;
struct uint4 { unsigned x, y, z, w; };
struct __cuda_builtin_threadIdx_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_tid_x(); } };
struct __cuda_builtin_blockIdx_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; __declspec(property(get = __fetch_builtin_y)) unsigned y; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_ctaid_x(); } static __device__ unsigned __fetch_builtin_y() { return __nvvm_read_ptx_sreg_ctaid_y(); } };
struct __cuda_builtin_blockDim_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_ntid_x(); } };
struct __cuda_builtin_gridDim_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_nctaid_x(); } };
extern const __device__ __cuda_builtin_threadIdx_t threadIdx; extern const __device__ __cuda_builtin_blockIdx_t blockIdx; extern const __device__ __cuda_builtin_blockDim_t blockDim; extern const __device__ __cuda_builtin_gridDim_t gridDim;
__device__ unsigned atomicAdd(unsigned* p, unsigned v) { return __nvvm_atom_add_gen_i((int*)p, (int)v); }
__device__ uint4 __ldcv(const uint4* p) { return *p; }
typedef unsigned long size_t;
__device__ inline void __threadfence_system() { __nvvm_membar_sys(); }

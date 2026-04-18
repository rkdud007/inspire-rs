// Common CUDA device helpers for InsPIRe PIR GPU kernels
//
// Shared Barrett reduction, NTT table access, and error checking macros.
// Included by all kernel .cu files to eliminate code duplication.

#ifndef INSPIRE_CUDA_COMMON_H
#define INSPIRE_CUDA_COMMON_H

#include <cstdint>
#include <cstdio>

// ============================================================================
// Error checking
// ============================================================================

#define CHECK_CUDA_TAG(tag, call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[%s] CUDA error at %s:%d: %s\n", \
                tag, __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// ============================================================================
// Barrett reduction (u64)
// ============================================================================

// val mod modulus, given cr = floor(2^64 / modulus)
// Two naming variants for historical reasons:
//   barrett_u64_dev(val, cr, modulus) — used by hint/collapse kernels
//   barrett_u64(val, modulus, cr)     — used by gemv/packing_online kernels
//   barrett_reduce(val, modulus, cr)  — used by packing kernel

__device__ __forceinline__ uint64_t barrett_u64_dev(uint64_t val, uint64_t cr1, uint64_t modulus) {
    uint64_t tmp = __umul64hi(val, cr1);
    uint64_t res = val - tmp * modulus;
    return res >= modulus ? res - modulus : res;
}

__device__ __forceinline__ uint64_t barrett_u64(uint64_t val, uint64_t modulus, uint64_t cr) {
    unsigned long long hi;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(val), "l"(cr));
    uint64_t res = val - hi * modulus;
    return res >= modulus ? res - modulus : res;
}

__device__ __forceinline__ uint64_t barrett_reduce(uint64_t val, uint64_t modulus, uint64_t cr) {
    unsigned long long hi;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(val), "l"(cr));
    uint64_t remainder = val - hi * modulus;
    return remainder >= modulus ? remainder - modulus : remainder;
}

// ============================================================================
// Barrett reduction (u128)
// ============================================================================

// (zx + zy * 2^64) mod modulus, using cr0/cr1 = floor(2^128 / modulus) split
// Used by hint_kernel and collapse_kernel
__device__ __forceinline__ uint64_t barrett_u128_dev(
    uint64_t zx, uint64_t zy,
    uint64_t cr0, uint64_t cr1,
    uint64_t modulus
) {
    uint64_t carry = __umul64hi(zx, cr0);
    uint64_t tmp2x = zx * cr1;
    uint64_t tmp2y = __umul64hi(zx, cr1);
    uint64_t tmp1 = tmp2x + carry;
    uint64_t tmp3 = tmp2y + (uint64_t)(tmp1 < tmp2x);
    uint64_t t2x = zy * cr0;
    uint64_t t2y = __umul64hi(zy, cr0);
    uint64_t old_tmp1 = tmp1;
    tmp1 = old_tmp1 + t2x;
    carry = t2y + (uint64_t)(tmp1 < old_tmp1);
    tmp1 = zy * cr1 + tmp3 + carry;
    tmp3 = zx - tmp1 * modulus;
    tmp3 -= modulus * (uint64_t)(tmp3 >= modulus);
    return tmp3;
}

// SEAL-style Barrett u128 — used by gemv_kernel
__device__ __forceinline__ uint64_t barrett_u128(
    uint64_t val_lo, uint64_t val_hi,
    uint64_t modulus, uint64_t cr0, uint64_t cr1
) {
    uint64_t tmp1, carry;
    unsigned long long prody;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(prody) : "l"(val_lo), "l"(cr0));
    carry = prody;

    unsigned long long tmp2x, tmp2y;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(tmp2x) : "l"(val_lo), "l"(cr1));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(tmp2y) : "l"(val_lo), "l"(cr1));

    uint64_t sum = tmp2x + carry;
    uint64_t carry_out = (sum < tmp2x) ? 1ULL : 0ULL;
    tmp1 = sum;
    uint64_t tmp3 = tmp2y + carry_out;

    asm("mul.lo.u64 %0, %1, %2;" : "=l"(tmp2x) : "l"(val_hi), "l"(cr0));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(tmp2y) : "l"(val_hi), "l"(cr0));

    sum = tmp1 + tmp2x;
    carry_out = (sum < tmp1) ? 1ULL : 0ULL;
    tmp1 = sum;
    carry = tmp2y + carry_out;

    tmp1 = val_hi * cr1 + tmp3 + carry;
    uint64_t result = val_lo - tmp1 * modulus;
    result -= modulus * (result >= modulus);
    return result;
}

// ============================================================================
// NTT table access
// ============================================================================

// Table layout: ntt_tables[crt * 4 * POLY_LEN + table_id * POLY_LEN + idx]
#define TBL_FWD_ROOT  0
#define TBL_FWD_PRIME 1
#define TBL_INV_ROOT  2
#define TBL_INV_PRIME 3

#ifndef POLY_LEN
#define POLY_LEN 2048
#endif

__device__ __forceinline__ uint64_t tbl(
    const uint64_t* __restrict__ ntt_tables,
    int crt, int table_id, int idx
) {
    return ntt_tables[crt * 4 * POLY_LEN + table_id * POLY_LEN + idx];
}

// ============================================================================
// Stream-ordered async memory allocation (CUDA 11.2+)
// ============================================================================
// cudaMalloc/cudaFree cause implicit device-wide synchronization, stalling
// ALL streams. These async variants are stream-ordered: they only synchronize
// within the calling thread's stream (with --default-stream per-thread).
// This eliminates cross-thread stalls between fold and query pipelines.

#define GPU_MALLOC(ptr, size) cudaMallocAsync((ptr), (size), cudaStreamPerThread)
#define GPU_FREE(ptr) cudaFreeAsync((ptr), cudaStreamPerThread)
#define GPU_STREAM_SYNC() cudaStreamSynchronize(cudaStreamPerThread)

// ============================================================================
// High-priority query stream
// ============================================================================
// Query-path kernels (GEMV, packing_online, rotation) use this stream so the
// GPU scheduler preempts fold kernel thread blocks in favor of query work.
// Defined in encode_kernel.cu, initialized by gpu_init_memory_pool().
extern cudaStream_t g_query_stream;

// Pool-aware available GPU memory query.
// cudaMemGetInfo only reports physical free memory, ignoring memory held by
// the async allocator pool. This function adds pool's unused reserved memory
// for accurate batch size calculations with async allocation.
static inline size_t gpu_available_memory() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        size_t pool_reserved = 0, pool_used = 0;
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &pool_used);
        if (pool_reserved > pool_used) {
            free_mem += (pool_reserved - pool_used);
        }
    }

    return free_mem;
}

#endif // INSPIRE_CUDA_COMMON_H

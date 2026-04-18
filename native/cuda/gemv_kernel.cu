// GPU-accelerated GEMV kernel for InsPIRe PIR online first-pass
//
// Computes: c[j] = sum_k a[k] * b[j * db_rows + k]  mod q
//
// Where:
//   a[k] is CRT-packed u64: lower 32 bits = a_k mod q0, upper 32 bits = a_k mod q1
//   b[j * db_rows + k] is u16 (plaintext DB value)
//   The dot product is computed separately per CRT limb, then composed.
//
// The DB (b) is uploaded once and kept resident on GPU across queries.
// Only the query vector (a) and result (c) are transferred per query.

#include "common.h"

// CRT compose: given lo_coeff = x mod q0, hi_coeff = x mod q1
// Compute x mod (q0 * q1) using precomputed orthogonal idempotents:
//   mod0_inv_mod1 = q0 * inv(q0, q1)
//   mod1_inv_mod0 = q1 * inv(q1, q0)
//   result = (lo * mod1_inv_mod0 + hi * mod0_inv_mod1) mod q
__device__ __forceinline__ uint64_t crt_compose(
    uint64_t lo, uint64_t hi,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t modulus, uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    // Compute lo * mod1_inv_mod0 as u128
    unsigned long long prod1_lo, prod1_hi;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(prod1_lo) : "l"(lo), "l"(mod1_inv_mod0));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(prod1_hi) : "l"(lo), "l"(mod1_inv_mod0));

    // Compute hi * mod0_inv_mod1 as u128
    unsigned long long prod2_lo, prod2_hi;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(prod2_lo) : "l"(hi), "l"(mod0_inv_mod1));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(prod2_hi) : "l"(hi), "l"(mod0_inv_mod1));

    // Add the two u128 products
    uint64_t sum_lo = prod1_lo + prod2_lo;
    uint64_t sum_hi = prod1_hi + prod2_hi + (sum_lo < prod1_lo ? 1ULL : 0ULL);

    // Barrett reduce the u128 sum mod q
    return barrett_u128(sum_lo, sum_hi, modulus, barrett_cr0_mod, barrett_cr1_mod);
}


// Main GEMV kernel
// Each warp (32 threads) computes one output element c[j]
__global__ void gemv_kernel(
    uint64_t* __restrict__ c,          // [b_cols] output
    const uint64_t* __restrict__ a,    // [db_rows] query vector (CRT-packed)
    const uint16_t* __restrict__ b_t,  // [db_rows * b_cols] DB column-major
    int db_rows,
    int b_cols,
    uint64_t mod0, uint64_t mod1,      // CRT moduli
    uint64_t cr0, uint64_t cr1,        // Barrett constants for CRT moduli (floor(2^64/qi))
    uint64_t mod0_inv_mod1,            // q0 * inv(q0, q1) for CRT compose
    uint64_t mod1_inv_mod0,            // q1 * inv(q1, q0) for CRT compose
    uint64_t modulus,                  // q = q0 * q1
    uint64_t barrett_cr0_mod,          // Barrett cr0 for full modulus (floor(2^128/q) lo64)
    uint64_t barrett_cr1_mod           // Barrett cr1 for full modulus (floor(2^128/q) hi64)
) {
    int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x & 31;

    if (warp_global_id >= b_cols) return;

    int j = warp_global_id;
    const uint16_t* col_ptr = b_t + (size_t)j * db_rows;

    uint64_t sum_lo = 0;  // accumulate (a_lo * b) for CRT limb 0
    uint64_t sum_hi = 0;  // accumulate (a_hi * b) for CRT limb 1

    // Each lane processes a strided subset of the inner dimension
    for (int k = lane_id; k < db_rows; k += 32) {
        uint64_t a_val = a[k];
        uint64_t b_val = (uint64_t)col_ptr[k];

        uint32_t a_lo = (uint32_t)a_val;          // a mod q0 (lower 32 bits)
        uint32_t a_hi = (uint32_t)(a_val >> 32);   // a mod q1 (upper 32 bits)

        sum_lo += (uint64_t)a_lo * b_val;
        sum_hi += (uint64_t)a_hi * b_val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_lo += __shfl_down_sync(0xFFFFFFFF, sum_lo, offset);
        sum_hi += __shfl_down_sync(0xFFFFFFFF, sum_hi, offset);
    }

    if (lane_id == 0) {
        // Barrett reduce each CRT limb
        uint64_t lo_coeff = barrett_u64(sum_lo, mod0, cr0);
        uint64_t hi_coeff = barrett_u64(sum_hi, mod1, cr1);

        // CRT compose to get result mod q
        uint64_t result = crt_compose(
            lo_coeff, hi_coeff,
            mod0_inv_mod1, mod1_inv_mod0,
            modulus, barrett_cr0_mod, barrett_cr1_mod
        );

        c[j] = result;
    }
}


// Host: adopt an existing device DB pointer (no upload, no allocation)
extern "C" void gpu_gemv_set_device_db(
    uint16_t** d_db_out,        // output: device DB pointer (set to given ptr)
    const uint16_t* d_db_in,    // existing device pointer to adopt
    size_t db_size_bytes         // for logging only
) {
    *d_db_out = const_cast<uint16_t*>(d_db_in);
}

// Host: upload DB to GPU (called once during server creation)
extern "C" void gpu_gemv_upload_db(
    const uint16_t* h_db,   // host DB pointer
    uint16_t** d_db_out,    // output: device DB pointer
    size_t db_size_bytes    // db_rows * db_cols * sizeof(u16)
) {
    cudaError_t err;
    err = GPU_MALLOC(d_db_out, db_size_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU GEMV] cudaMalloc failed (%.1f GB): %s\n",
                db_size_bytes / 1e9, cudaGetErrorString(err));
        *d_db_out = nullptr;
        return;
    }
    err = cudaMemcpy(*d_db_out, h_db, db_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU GEMV] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        GPU_FREE(*d_db_out);
        *d_db_out = nullptr;
        return;
    }
}

// Host: free GPU DB
extern "C" void gpu_gemv_free_db(uint16_t* d_db) {
    if (d_db) GPU_FREE(d_db);
}

// Host: run GEMV query (uses high-priority query stream)
extern "C" void gpu_gemv_query(
    uint64_t* h_c,             // host output [b_cols]
    const uint64_t* h_a,       // host query [db_rows]
    const uint16_t* d_db,      // device DB (already uploaded)
    int db_rows,
    int b_cols,
    uint64_t mod0, uint64_t mod1,
    uint64_t cr0, uint64_t cr1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    cudaStream_t s = g_query_stream;

    // Upload query
    uint64_t* d_a = nullptr;
    size_t a_size = (size_t)db_rows * sizeof(uint64_t);
    cudaMallocAsync(&d_a, a_size, s);
    cudaMemcpyAsync(d_a, h_a, a_size, cudaMemcpyHostToDevice, s);

    // Allocate output
    uint64_t* d_c = nullptr;
    size_t c_size = (size_t)b_cols * sizeof(uint64_t);
    cudaMallocAsync(&d_c, c_size, s);

    // Launch kernel: one warp per output element
    // 256 threads/block = 8 warps/block
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (b_cols + warps_per_block - 1) / warps_per_block;

    gemv_kernel<<<blocks, threads_per_block, 0, s>>>(
        d_c, d_a, d_db,
        db_rows, b_cols,
        mod0, mod1, cr0, cr1,
        mod0_inv_mod1, mod1_inv_mod0,
        modulus, barrett_cr0_mod, barrett_cr1_mod
    );

    // Download result
    cudaMemcpyAsync(h_c, d_c, c_size, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFreeAsync(d_a, s);
    cudaFreeAsync(d_c, s);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU GEMV] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

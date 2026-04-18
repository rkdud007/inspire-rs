// GPU-accelerated hint computation for InsPIRe PIR
//
// Replaces the CPU multiply_with_db_ring inner loop.
// For each DB column, computes:
//   sum = Σ_{row=0..db_rows_poly-1} NTT(db_col_row) ⊙ query[row]
// Then inverse NTT + CRT compose to get coefficient-domain result.
//
// Architecture: one block per DB column, 256 threads per block.
// Forward/inverse NTT done in shared memory with __syncthreads per level.

#include "common.h"

#define CHECK_CUDA(call) CHECK_CUDA_TAG("GPU Hint", call)

#define HINT_THREADS 256
#define POLY_LEN_LOG2 11
#define NUM_BUTTERFLIES 1024
#define ELEMS_PER_THREAD (POLY_LEN / HINT_THREADS)       // 8
#define BFLIES_PER_THREAD (NUM_BUTTERFLIES / HINT_THREADS) // 4


// Main hint kernel: one block per DB column
__global__ void hint_multiply_kernel(
    const uint16_t* __restrict__ db,       // [db_cols * db_rows] column-major u16
    uint64_t* __restrict__ output,         // [db_cols * POLY_LEN] coefficient domain
    const uint64_t* __restrict__ query,    // [db_rows_poly * 2 * POLY_LEN] NTT domain
    const uint64_t* __restrict__ ntt_tables, // [2 * 4 * POLY_LEN]
    int db_rows,
    int db_rows_poly,
    int db_cols,
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    int col = blockIdx.x;
    if (col >= db_cols) return;
    int tid = threadIdx.x;

    extern __shared__ uint64_t ntt_buf[];  // [POLY_LEN]

    // Per-thread accumulators for 8 elements each, 2 CRT limbs
    uint64_t sum0[ELEMS_PER_THREAD];  // CRT0 accumulator
    uint64_t sum1[ELEMS_PER_THREAD];  // CRT1 accumulator
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        sum0[e] = 0;
        sum1[e] = 0;
    }

    const uint32_t mods[2] = {mod0, mod1};

    for (int row = 0; row < db_rows_poly; row++) {
        // Load DB values for this row into registers (u16 → u64)
        uint64_t db_vals[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int z = tid + e * HINT_THREADS;
            size_t db_idx = (size_t)col * db_rows + (size_t)row * POLY_LEN + z;
            db_vals[e] = (uint64_t)db[db_idx];
        }

        // Process each CRT limb
        for (int crt = 0; crt < 2; crt++) {
            uint32_t mod_val = mods[crt];
            uint32_t two_mod = 2 * mod_val;

            // Load DB values to shared memory (trivially < both moduli)
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                ntt_buf[tid + e * HINT_THREADS] = db_vals[e];
            }
            __syncthreads();

            // Forward NTT (POLY_LEN_LOG2 = 11 levels)
            for (int mm = 0; mm < POLY_LEN_LOG2; mm++) {
                int m = 1 << mm;
                int t = POLY_LEN >> (mm + 1);

                #pragma unroll
                for (int b = 0; b < BFLIES_PER_THREAD; b++) {
                    int bfly_id = tid + b * HINT_THREADS;
                    int group_idx = bfly_id / t;
                    int elem_idx = bfly_id % t;
                    int idx_a = group_idx * 2 * t + elem_idx;
                    int idx_b = idx_a + t;

                    uint64_t w = tbl(ntt_tables, crt, TBL_FWD_ROOT, m + group_idx);
                    uint64_t w_prime = tbl(ntt_tables, crt, TBL_FWD_PRIME, m + group_idx);

                    uint32_t x = (uint32_t)ntt_buf[idx_a];
                    uint32_t y = (uint32_t)ntt_buf[idx_b];

                    uint32_t curr_x = x - (two_mod * (uint32_t)(x >= two_mod));
                    uint64_t q_tmp = ((uint64_t)y * (uint64_t)(uint32_t)w_prime) >> 32;
                    uint64_t q_new = (uint64_t)(uint32_t)w * (uint64_t)y
                                     - q_tmp * (uint64_t)mod_val;

                    ntt_buf[idx_a] = (uint64_t)curr_x + q_new;
                    ntt_buf[idx_b] = (uint64_t)curr_x + ((uint64_t)two_mod - q_new);
                }
                __syncthreads();
            }

            // Final reduction after forward NTT
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * HINT_THREADS;
                uint64_t v = ntt_buf[z];
                v -= (uint64_t)two_mod * (uint64_t)(v >= (uint64_t)two_mod);
                v -= (uint64_t)mod_val * (uint64_t)(v >= (uint64_t)mod_val);
                ntt_buf[z] = v;
            }
            __syncthreads();

            // Pointwise multiply with query and accumulate
            uint64_t barrett_cr1 = (crt == 0) ? barrett_cr1_0 : barrett_cr1_1;
            uint64_t mod64 = (uint64_t)mod_val;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * HINT_THREADS;
                uint64_t ntt_val = ntt_buf[z];
                uint64_t q_val = query[(size_t)row * 2 * POLY_LEN + crt * POLY_LEN + z];
                uint64_t prod = ntt_val * q_val;
                // Barrett reduce the product
                prod = barrett_u64_dev(prod, barrett_cr1, mod64);
                if (crt == 0)
                    sum0[e] += prod;
                else
                    sum1[e] += prod;
            }
            __syncthreads();
        }
    }

    // Barrett reduce the final sums
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        sum0[e] = barrett_u64_dev(sum0[e], barrett_cr1_0, (uint64_t)mod0);
        sum1[e] = barrett_u64_dev(sum1[e], barrett_cr1_1, (uint64_t)mod1);
    }

    // ========== Inverse NTT for CRT0 ==========
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        ntt_buf[tid + e * HINT_THREADS] = sum0[e];
    }
    __syncthreads();

    for (int mm = POLY_LEN_LOG2 - 1; mm >= 0; mm--) {
        int h = 1 << mm;
        int t = POLY_LEN >> (mm + 1);

        #pragma unroll
        for (int b = 0; b < BFLIES_PER_THREAD; b++) {
            int bfly_id = tid + b * HINT_THREADS;
            int group_idx = bfly_id / t;
            int elem_idx = bfly_id % t;
            int idx_a = group_idx * 2 * t + elem_idx;
            int idx_b = idx_a + t;

            uint64_t w = tbl(ntt_tables, 0, TBL_INV_ROOT, h + group_idx);
            uint64_t w_prime = tbl(ntt_tables, 0, TBL_INV_PRIME, h + group_idx);

            uint64_t x = ntt_buf[idx_a];
            uint64_t y = ntt_buf[idx_b];
            uint64_t two_mod64 = 2ULL * (uint64_t)mod0;

            uint64_t t_tmp = two_mod64 - y + x;
            uint64_t curr_x = x + y - (two_mod64 * (uint64_t)((x << 1) >= t_tmp));
            uint64_t h_tmp = (t_tmp * (uint64_t)(uint32_t)w_prime) >> 32;

            uint64_t res_x = (curr_x + ((uint64_t)mod0 * (t_tmp & 1ULL))) >> 1;
            uint64_t res_y = (uint64_t)(uint32_t)w * t_tmp - h_tmp * (uint64_t)mod0;

            ntt_buf[idx_a] = res_x;
            ntt_buf[idx_b] = res_y;
        }
        __syncthreads();
    }

    // Read CRT0 inverse NTT results
    uint64_t val0[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        val0[e] = ntt_buf[tid + e * HINT_THREADS];
    }
    __syncthreads();

    // ========== Inverse NTT for CRT1 ==========
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        ntt_buf[tid + e * HINT_THREADS] = sum1[e];
    }
    __syncthreads();

    for (int mm = POLY_LEN_LOG2 - 1; mm >= 0; mm--) {
        int h = 1 << mm;
        int t = POLY_LEN >> (mm + 1);

        #pragma unroll
        for (int b = 0; b < BFLIES_PER_THREAD; b++) {
            int bfly_id = tid + b * HINT_THREADS;
            int group_idx = bfly_id / t;
            int elem_idx = bfly_id % t;
            int idx_a = group_idx * 2 * t + elem_idx;
            int idx_b = idx_a + t;

            uint64_t w = tbl(ntt_tables, 1, TBL_INV_ROOT, h + group_idx);
            uint64_t w_prime = tbl(ntt_tables, 1, TBL_INV_PRIME, h + group_idx);

            uint64_t x = ntt_buf[idx_a];
            uint64_t y = ntt_buf[idx_b];
            uint64_t two_mod64 = 2ULL * (uint64_t)mod1;

            uint64_t t_tmp = two_mod64 - y + x;
            uint64_t curr_x = x + y - (two_mod64 * (uint64_t)((x << 1) >= t_tmp));
            uint64_t h_tmp = (t_tmp * (uint64_t)(uint32_t)w_prime) >> 32;

            uint64_t res_x = (curr_x + ((uint64_t)mod1 * (t_tmp & 1ULL))) >> 1;
            uint64_t res_y = (uint64_t)(uint32_t)w * t_tmp - h_tmp * (uint64_t)mod1;

            ntt_buf[idx_a] = res_x;
            ntt_buf[idx_b] = res_y;
        }
        __syncthreads();
    }

    // Read CRT1 inverse NTT results
    uint64_t val1[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        val1[e] = ntt_buf[tid + e * HINT_THREADS];
    }

    // CRT compose and store output in TRANSPOSED layout:
    // output[z * db_cols + col] instead of output[col * POLY_LEN + z]
    // This matches the transpose_generic output format expected by the caller.
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int z = tid + e * HINT_THREADS;

        // val0 * mod1_inv_mod0 (u128)
        uint64_t p0_lo = val0[e] * mod1_inv_mod0;
        uint64_t p0_hi = __umul64hi(val0[e], mod1_inv_mod0);

        // val1 * mod0_inv_mod1 (u128)
        uint64_t p1_lo = val1[e] * mod0_inv_mod1;
        uint64_t p1_hi = __umul64hi(val1[e], mod0_inv_mod1);

        // sum = p0 + p1 (u128)
        uint64_t s_lo = p0_lo + p1_lo;
        uint64_t s_hi = p0_hi + p1_hi + (uint64_t)(s_lo < p0_lo);

        uint64_t result = barrett_u128_dev(s_lo, s_hi, barrett_cr0_mod, barrett_cr1_mod, modulus);
        output[(size_t)z * db_cols + col] = result;
    }
}


// ======================== Host-side functions ========================

static uint64_t* d_ntt_tables = nullptr;
static uint64_t* d_query = nullptr;
static uint64_t* d_hint_output = nullptr;
static int saved_db_cols = 0;

extern "C" void gpu_hint_upload_tables(
    const uint64_t* h_ntt_tables,  // [2 * 4 * POLY_LEN] flattened
    int poly_len
) {
    size_t size = 2 * 4 * poly_len * sizeof(uint64_t);
    if (d_ntt_tables) GPU_FREE(d_ntt_tables);
    CHECK_CUDA(GPU_MALLOC(&d_ntt_tables, size));
    CHECK_CUDA(cudaMemcpy(d_ntt_tables, h_ntt_tables, size, cudaMemcpyHostToDevice));
}

extern "C" void gpu_hint_upload_query_ffi(
    const uint64_t* h_query,  // [db_rows_poly * 2 * POLY_LEN]
    int db_rows_poly,
    int poly_len
) {
    size_t size = (size_t)db_rows_poly * 2 * poly_len * sizeof(uint64_t);
    if (d_query) GPU_FREE(d_query);
    CHECK_CUDA(GPU_MALLOC(&d_query, size));
    CHECK_CUDA(cudaMemcpy(d_query, h_query, size, cudaMemcpyHostToDevice));
}

extern "C" void gpu_hint_compute(
    const uint16_t* d_db,      // device pointer to encoded DB (u16, column-major)
    uint64_t* h_output,        // host pointer for output [db_cols * POLY_LEN]
    int db_rows,
    int db_rows_poly,
    int db_cols,
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    if (!d_ntt_tables || !d_query) {
        fprintf(stderr, "[GPU Hint] Tables or query not uploaded!\n");
        return;
    }

    // Allocate output buffer on GPU
    size_t out_size = (size_t)db_cols * POLY_LEN * sizeof(uint64_t);
    if (db_cols != saved_db_cols) {
        if (d_hint_output) GPU_FREE(d_hint_output);
        CHECK_CUDA(GPU_MALLOC(&d_hint_output, out_size));
        saved_db_cols = db_cols;
    }

    size_t shared_mem = POLY_LEN * sizeof(uint64_t);  // 16 KB

    hint_multiply_kernel<<<db_cols, HINT_THREADS, shared_mem>>>(
        d_db, d_hint_output, d_query, d_ntt_tables,
        db_rows, db_rows_poly, db_cols,
        mod0, mod1,
        barrett_cr1_0, barrett_cr1_1,
        mod0_inv_mod1, mod1_inv_mod0,
        modulus, barrett_cr0_mod, barrett_cr1_mod
    );
    CHECK_CUDA(GPU_STREAM_SYNC());

    // Copy result to host
    CHECK_CUDA(cudaMemcpy(h_output, d_hint_output, out_size, cudaMemcpyDeviceToHost));
}

// Variant that uploads DB from host (for standalone use)
extern "C" void gpu_hint_compute_from_host(
    const uint16_t* h_db,
    uint64_t* h_output,
    int db_rows,
    int db_rows_poly,
    int db_cols,
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    size_t db_size = (size_t)db_rows * db_cols * sizeof(uint16_t);

    uint16_t* d_db = nullptr;
    CHECK_CUDA(GPU_MALLOC(&d_db, db_size));
    CHECK_CUDA(cudaMemcpy(d_db, h_db, db_size, cudaMemcpyHostToDevice));

    gpu_hint_compute(
        d_db, h_output,
        db_rows, db_rows_poly, db_cols,
        mod0, mod1,
        barrett_cr1_0, barrett_cr1_1,
        mod0_inv_mod1, mod1_inv_mod0,
        modulus, barrett_cr0_mod, barrett_cr1_mod
    );

    GPU_FREE(d_db);
}

extern "C" void gpu_hint_free() {
    if (d_ntt_tables) { GPU_FREE(d_ntt_tables); d_ntt_tables = nullptr; }
    if (d_query) { GPU_FREE(d_query); d_query = nullptr; }
    if (d_hint_output) { GPU_FREE(d_hint_output); d_hint_output = nullptr; }
    saved_db_cols = 0;
}

// Free only query (keep hint output and NTT tables for prep_pack)
extern "C" void gpu_hint_free_query_only() {
    if (d_query) { GPU_FREE(d_query); d_query = nullptr; }
}

// Free hint output and NTT tables (after prep_pack consumes them)
extern "C" void gpu_hint_free_output_and_tables() {
    if (d_ntt_tables) { GPU_FREE(d_ntt_tables); d_ntt_tables = nullptr; }
    if (d_hint_output) { GPU_FREE(d_hint_output); d_hint_output = nullptr; }
    saved_db_cols = 0;
}

// Accessors for device pointers (used by prep_pack kernel)
extern "C" const uint64_t* gpu_hint_get_device_output() { return d_hint_output; }
extern "C" const uint64_t* gpu_hint_get_ntt_tables() { return d_ntt_tables; }
extern "C" int gpu_hint_get_db_cols() { return saved_db_cols; }

// Hint compute variant that skips D2H copy (keeps result on GPU)
extern "C" void gpu_hint_compute_no_d2h(
    const uint16_t* d_db,
    int db_rows,
    int db_rows_poly,
    int db_cols,
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    if (!d_ntt_tables || !d_query) {
        fprintf(stderr, "[GPU Hint] Tables or query not uploaded!\n");
        return;
    }

    size_t out_size = (size_t)db_cols * POLY_LEN * sizeof(uint64_t);
    if (db_cols != saved_db_cols) {
        if (d_hint_output) GPU_FREE(d_hint_output);
        CHECK_CUDA(GPU_MALLOC(&d_hint_output, out_size));
        saved_db_cols = db_cols;
    }

    size_t shared_mem = POLY_LEN * sizeof(uint64_t);

    hint_multiply_kernel<<<db_cols, HINT_THREADS, shared_mem>>>(
        d_db, d_hint_output, d_query, d_ntt_tables,
        db_rows, db_rows_poly, db_cols,
        mod0, mod1,
        barrett_cr1_0, barrett_cr1_1,
        mod0_inv_mod1, mod1_inv_mod0,
        modulus, barrett_cr0_mod, barrett_cr1_mod
    );
    CHECK_CUDA(GPU_STREAM_SYNC());
}

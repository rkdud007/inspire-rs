// GPU-accelerated prep_pack: negacyclic_perm + forward NTT on hint output
//
// Replaces CPU prep_pack_many_lwes by operating directly on device-resident
// hint output (d_hint_output) and NTT tables (d_ntt_tables) from hint_kernel.cu.
//
// For each output polynomial gc (0..db_cols):
//   1. Gather: poly[z] = d_hint_output[z * db_cols + gc]  (strided read)
//   2. negacyclic_perm(shift=0, modulus=65535)
//   3. Forward NTT × 2 CRT limbs (input < 65535 < both moduli, CRT decomp is trivial)
//   4. Write: d_prepacked[gc * coeff_count + crt * poly_len + z]
//
// Architecture: one block per output polynomial, 256 threads per block.

#include "common.h"

#define PP_THREADS 256
#define POLY_LEN_LOG2 11
#define NUM_BUTTERFLIES 1024
#define PP_ELEMS_PER_THREAD (POLY_LEN / PP_THREADS)       // 8
#define PP_BFLIES_PER_THREAD (NUM_BUTTERFLIES / PP_THREADS) // 4

__global__ void prep_pack_kernel(
    const uint64_t* __restrict__ d_hint_output,  // [poly_len * db_cols] transposed
    const uint64_t* __restrict__ d_ntt_tables,   // [2 * 4 * poly_len]
    uint64_t* __restrict__ d_prepacked,           // [db_cols * 2 * poly_len]
    int db_cols,
    uint32_t mod0, uint32_t mod1,
    uint64_t modulus  // CRT product modulus (mod0 * mod1)
) {
    int gc = blockIdx.x;  // global column index
    if (gc >= db_cols) return;
    int tid = threadIdx.x;

    extern __shared__ uint64_t ntt_buf[];  // [POLY_LEN]

    // Step 1: Gather from hint output (strided access, stride = db_cols)
    #pragma unroll
    for (int e = 0; e < PP_ELEMS_PER_THREAD; e++) {
        int z = tid + e * PP_THREADS;
        ntt_buf[z] = d_hint_output[(size_t)z * db_cols + gc];
    }
    __syncthreads();

    // Step 2: Apply negacyclic_perm(shift=0) — save to registers
    // shift=0: out[0] = a[0], out[i] = modulus - a[n-i] for i>0 (0 if a[n-i]==0)
    uint64_t nega_vals[PP_ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < PP_ELEMS_PER_THREAD; e++) {
        int z = tid + e * PP_THREADS;
        if (z == 0) {
            nega_vals[e] = ntt_buf[0];
        } else {
            uint64_t v = ntt_buf[POLY_LEN - z];
            nega_vals[e] = (v == 0) ? 0 : (modulus - v);
        }
    }
    __syncthreads();

    // Step 3: Forward NTT for each CRT limb
    // Hint output is mod (mod0*mod1) — CRT decompose by reducing mod each limb
    const uint32_t mods[2] = {mod0, mod1};

    for (int crt = 0; crt < 2; crt++) {
        uint32_t mod_val = mods[crt];
        uint32_t two_mod = 2 * mod_val;

        // Load negacyclic values to shared memory with CRT reduction
        #pragma unroll
        for (int e = 0; e < PP_ELEMS_PER_THREAD; e++) {
            ntt_buf[tid + e * PP_THREADS] = nega_vals[e] % (uint64_t)mod_val;
        }
        __syncthreads();

        // Forward NTT (identical to hint_kernel.cu forward NTT)
        for (int mm = 0; mm < POLY_LEN_LOG2; mm++) {
            int m = 1 << mm;
            int t = POLY_LEN >> (mm + 1);

            #pragma unroll
            for (int b = 0; b < PP_BFLIES_PER_THREAD; b++) {
                int bfly_id = tid + b * PP_THREADS;
                int group_idx = bfly_id / t;
                int elem_idx = bfly_id % t;
                int idx_a = group_idx * 2 * t + elem_idx;
                int idx_b = idx_a + t;

                uint64_t w = tbl(d_ntt_tables, crt, TBL_FWD_ROOT, m + group_idx);
                uint64_t w_prime = tbl(d_ntt_tables, crt, TBL_FWD_PRIME, m + group_idx);

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

        // Final reduction + write output
        // Output layout: d_prepacked[gc * coeff_count + crt * poly_len + z]
        // where coeff_count = 2 * poly_len
        #pragma unroll
        for (int e = 0; e < PP_ELEMS_PER_THREAD; e++) {
            int z = tid + e * PP_THREADS;
            uint64_t v = ntt_buf[z];
            v -= (uint64_t)two_mod * (uint64_t)(v >= (uint64_t)two_mod);
            v -= (uint64_t)mod_val * (uint64_t)(v >= (uint64_t)mod_val);
            d_prepacked[(size_t)gc * 2 * POLY_LEN + crt * POLY_LEN + z] = v;
        }
        __syncthreads();
    }
}


// ======================== Host-side functions ========================

static uint64_t* d_prepacked = nullptr;
static int pp_db_cols = 0;

extern "C" void gpu_prep_pack_compute(
    const uint64_t* d_hint_output,
    const uint64_t* d_ntt_tables,
    int db_cols,
    uint32_t mod0, uint32_t mod1,
    uint64_t modulus
) {
    size_t coeff_count = 2 * POLY_LEN;
    size_t out_size = (size_t)db_cols * coeff_count * sizeof(uint64_t);

    // Allocate output buffer
    if (db_cols != pp_db_cols) {
        if (d_prepacked) GPU_FREE(d_prepacked);
        cudaError_t err = GPU_MALLOC(&d_prepacked, out_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[GPU prep_pack] cudaMalloc failed (%.2f GB): %s\n",
                    out_size / (1024.0*1024.0*1024.0), cudaGetErrorString(err));
            d_prepacked = nullptr;
            return;
        }
        pp_db_cols = db_cols;
    }

    size_t shared_mem = POLY_LEN * sizeof(uint64_t);  // 16 KB

    prep_pack_kernel<<<db_cols, PP_THREADS, shared_mem>>>(
        d_hint_output, d_ntt_tables, d_prepacked,
        db_cols, mod0, mod1, modulus
    );
    cudaError_t err = GPU_STREAM_SYNC();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU prep_pack] kernel error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" const uint64_t* gpu_prep_pack_get_output() {
    return d_prepacked;
}

extern "C" void gpu_prep_pack_free() {
    if (d_prepacked) { GPU_FREE(d_prepacked); d_prepacked = nullptr; }
    pp_db_cols = 0;
}

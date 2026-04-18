// GPU-accelerated InspiRING collapse kernel
//
// Replaces the CPU automorphism + collapse loop in packing.rs.
// Two kernels:
//   1. automorphism_kernel: permutes r_flat → r_all for all columns
//   2. collapse_kernel: persistent fold (INTT → CRT compose → gadget → NTT → mul-acc)
//
// Architecture:
//   - Automorphism: grid kernel, embarrassingly parallel
//   - Collapse: one block per column, 256 threads, sequential fold within block

#include "common.h"

#define CHECK_CUDA(call) CHECK_CUDA_TAG("GPU Collapse", call)

#define POLY_LEN_LOG2 11
#define COLLAPSE_THREADS 256
#define NUM_BUTTERFLIES 1024
#define ELEMS_PER_THREAD (POLY_LEN / COLLAPSE_THREADS)       // 8
#define BFLIES_PER_THREAD (NUM_BUTTERFLIES / COLLAPSE_THREADS) // 4

// ============================================================================
// Shared-memory NTT helpers
// ============================================================================

// Forward NTT in shared memory. ntt_buf has POLY_LEN u64 entries.
// mod_val, two_mod are for this CRT limb.
__device__ void forward_ntt_shared(
    uint64_t* ntt_buf,
    const uint64_t* __restrict__ ntt_tables,
    int crt,
    uint32_t mod_val,
    int tid
) {
    uint32_t two_mod = 2 * mod_val;

    for (int mm = 0; mm < POLY_LEN_LOG2; mm++) {
        int m = 1 << mm;
        int t = POLY_LEN >> (mm + 1);

        #pragma unroll
        for (int b = 0; b < BFLIES_PER_THREAD; b++) {
            int bfly_id = tid + b * COLLAPSE_THREADS;
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

    // Final reduction
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int z = tid + e * COLLAPSE_THREADS;
        uint64_t v = ntt_buf[z];
        v -= (uint64_t)two_mod * (uint64_t)(v >= (uint64_t)two_mod);
        v -= (uint64_t)mod_val * (uint64_t)(v >= (uint64_t)mod_val);
        ntt_buf[z] = v;
    }
    __syncthreads();
}

// Inverse NTT in shared memory.
__device__ void inverse_ntt_shared(
    uint64_t* ntt_buf,
    const uint64_t* __restrict__ ntt_tables,
    int crt,
    uint32_t mod_val,
    int tid
) {
    for (int mm = POLY_LEN_LOG2 - 1; mm >= 0; mm--) {
        int h = 1 << mm;
        int t = POLY_LEN >> (mm + 1);

        #pragma unroll
        for (int b = 0; b < BFLIES_PER_THREAD; b++) {
            int bfly_id = tid + b * COLLAPSE_THREADS;
            int group_idx = bfly_id / t;
            int elem_idx = bfly_id % t;
            int idx_a = group_idx * 2 * t + elem_idx;
            int idx_b = idx_a + t;

            uint64_t w = tbl(ntt_tables, crt, TBL_INV_ROOT, h + group_idx);
            uint64_t w_prime = tbl(ntt_tables, crt, TBL_INV_PRIME, h + group_idx);

            uint64_t x = ntt_buf[idx_a];
            uint64_t y = ntt_buf[idx_b];
            uint64_t two_mod64 = 2ULL * (uint64_t)mod_val;

            uint64_t t_tmp = two_mod64 - y + x;
            uint64_t curr_x = x + y - (two_mod64 * (uint64_t)((x << 1) >= t_tmp));
            uint64_t h_tmp = (t_tmp * (uint64_t)(uint32_t)w_prime) >> 32;

            uint64_t res_x = (curr_x + ((uint64_t)mod_val * (t_tmp & 1ULL))) >> 1;
            uint64_t res_y = (uint64_t)(uint32_t)w * t_tmp - h_tmp * (uint64_t)mod_val;

            ntt_buf[idx_a] = res_x;
            ntt_buf[idx_b] = res_y;
        }
        __syncthreads();
    }
}

// ============================================================================
// Automorphism kernel
// ============================================================================
// Permutes r_flat[col][i][crt*2048+z] → r_all[col][i][crt*2048+table[z]]
// where table = tables[table_idx] and table_idx = (gen_pows[i]-1)/2
// For r_bar: table_idx = (2*poly_len - gen_pows[i] - 1)/2
//
// Grid: (num_cols, num_to_pack_half), Block: POLY_LEN/2
// Each thread handles z and z+poly_len/2 for both CRT limbs.

__global__ void automorphism_kernel(
    uint64_t* __restrict__ r_all,          // [num_cols * num_to_pack_half * coeff_count]
    uint64_t* __restrict__ r_bar_all,      // same layout
    const uint64_t* __restrict__ r_flat,       // [num_to_pack_half * coeff_count] (s_r_out)
    const uint64_t* __restrict__ r_bar_flat,   // [num_to_pack_half * coeff_count] (s_r_bar_out)
    const uint32_t* __restrict__ tables,   // [num_tables * poly_len]
    const uint32_t* __restrict__ gen_pows, // [num_to_pack_half]
    int num_to_pack_half,
    int poly_len,
    int crt_count
) {
    int col = blockIdx.x;
    int i = blockIdx.y;
    int half = poly_len / 2;
    int z0 = threadIdx.x;
    int z1 = z0 + half;

    if (i >= num_to_pack_half || z0 >= half) return;

    int coeff_count = crt_count * poly_len;
    uint32_t gp = gen_pows[i];
    uint32_t table_idx_r = (gp - 1) / 2;
    uint32_t table_idx_rbar = (2 * poly_len - gp - 1) / 2;

    size_t r_flat_base = (size_t)i * coeff_count;
    size_t out_base = ((size_t)col * num_to_pack_half + i) * coeff_count;

    for (int crt = 0; crt < crt_count; crt++) {
        size_t crt_off = (size_t)crt * poly_len;

        // r path: permute s_r_out with table[table_idx_r]
        size_t t_base_r = (size_t)table_idx_r * poly_len;
        uint32_t src0_r = tables[t_base_r + z0];
        uint32_t src1_r = tables[t_base_r + z1];
        r_all[out_base + crt_off + z0] = r_flat[r_flat_base + crt_off + src0_r];
        r_all[out_base + crt_off + z1] = r_flat[r_flat_base + crt_off + src1_r];

        // r_bar path: permute s_r_bar_out with table[table_idx_rbar]
        size_t t_base_rb = (size_t)table_idx_rbar * poly_len;
        uint32_t src0_rb = tables[t_base_rb + z0];
        uint32_t src1_rb = tables[t_base_rb + z1];
        r_bar_all[out_base + crt_off + z0] = r_bar_flat[r_flat_base + crt_off + src0_rb];
        r_bar_all[out_base + crt_off + z1] = r_bar_flat[r_flat_base + crt_off + src1_rb];
    }
}

// ============================================================================
// Collapse kernel (persistent, one block per column)
// ============================================================================
// Each block processes one column: 1023 reverse iterations of
//   INTT(r_all[i+1]) → CRT compose → gadget_invert → NTT → mul-acc → r_all[i]
// Also writes condensed bold_t output for the online kernel.
//
// Grid: num_cols blocks, COLLAPSE_THREADS threads, POLY_LEN*8 bytes shared.

__global__ void collapse_kernel(
    uint64_t* __restrict__ r_all,          // [num_cols * nphalf * coeff_count] in-place
    uint64_t* __restrict__ r_bar_all,      // same layout
    uint64_t* __restrict__ bold_t,         // [num_cols * nphalf_m1 * t_exp * poly_len] condensed
    uint64_t* __restrict__ bold_t_bar,     // same layout
    uint64_t* __restrict__ bold_t_hat,     // [num_cols * t_exp * poly_len] condensed
    uint64_t* __restrict__ a_hat,          // [num_cols * poly_len] coefficient domain output
    const uint64_t* __restrict__ w_all,    // [nphalf_m1 * t_exp * coeff_count]
    const uint64_t* __restrict__ w_bar_all,// same layout
    const uint64_t* __restrict__ v_mask,   // [t_exp * coeff_count]
    const uint64_t* __restrict__ ntt_tables,
    int num_cols,
    int nphalf,           // num_to_pack_half = 1024
    int t_exp,            // t_exp_left = 3
    int bits_per,         // gadget bit width = 19
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t crt_modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    int col = blockIdx.x;
    if (col >= num_cols) return;
    int tid = threadIdx.x;

    extern __shared__ uint64_t ntt_buf[];  // [POLY_LEN]

    int coeff_count = 2 * POLY_LEN;  // crt_count=2
    int nphalf_m1 = nphalf - 1;
    uint64_t gadget_mask = (1ULL << bits_per) - 1;

    size_t col_r_base = (size_t)col * nphalf * coeff_count;
    size_t col_bt_base = (size_t)col * nphalf_m1 * t_exp * POLY_LEN;
    size_t col_bh_base = (size_t)col * t_exp * POLY_LEN;

    // Main collapse loop: iterate from nphalf_m1-1 down to 0
    for (int iter = nphalf_m1 - 1; iter >= 0; iter--) {
        // Process both r and r_bar paths
        for (int path = 0; path < 2; path++) {
            uint64_t* r_data = (path == 0) ? r_all : r_bar_all;
            uint64_t* bt_out = (path == 0) ? bold_t : bold_t_bar;
            const uint64_t* w_data = (path == 0) ? w_all : w_bar_all;

            size_t src_base = col_r_base + (size_t)(iter + 1) * coeff_count;
            size_t dst_base = col_r_base + (size_t)iter * coeff_count;

            // ---- INTT CRT0 of r[iter+1] ----
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                ntt_buf[tid + e * COLLAPSE_THREADS] = r_data[src_base + tid + e * COLLAPSE_THREADS];
            }
            __syncthreads();

            inverse_ntt_shared(ntt_buf, ntt_tables, 0, mod0, tid);

            uint64_t coeff_crt0[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                coeff_crt0[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
            }
            __syncthreads();

            // ---- INTT CRT1 of r[iter+1] ----
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                ntt_buf[tid + e * COLLAPSE_THREADS] = r_data[src_base + POLY_LEN + tid + e * COLLAPSE_THREADS];
            }
            __syncthreads();

            inverse_ntt_shared(ntt_buf, ntt_tables, 1, mod1, tid);

            uint64_t coeff_crt1[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                coeff_crt1[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
            }
            __syncthreads();

            // ---- CRT compose: coeff domain values → single modulus ----
            uint64_t composed[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                uint64_t v0 = coeff_crt0[e];
                uint64_t v1 = coeff_crt1[e];

                // val0 * mod1_inv_mod0 (u128)
                uint64_t p0_lo = v0 * mod1_inv_mod0;
                uint64_t p0_hi = __umul64hi(v0, mod1_inv_mod0);
                // val1 * mod0_inv_mod1 (u128)
                uint64_t p1_lo = v1 * mod0_inv_mod1;
                uint64_t p1_hi = __umul64hi(v1, mod0_inv_mod1);
                // sum (u128)
                uint64_t s_lo = p0_lo + p1_lo;
                uint64_t s_hi = p0_hi + p1_hi + (uint64_t)(s_lo < p0_lo);

                composed[e] = barrett_u128_dev(s_lo, s_hi, barrett_cr0_mod, barrett_cr1_mod, crt_modulus);
            }

            // ---- Accumulator for r[iter] updates (NTT domain) ----
            uint64_t acc_crt0[ELEMS_PER_THREAD];
            uint64_t acc_crt1[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                acc_crt0[e] = 0;
                acc_crt1[e] = 0;
            }

            // ---- For each gadget piece k ----
            for (int k = 0; k < t_exp; k++) {
                // Gadget extract: piece = (composed[e] >> (k * bits_per)) & mask
                // Then NTT both CRT limbs (input is coeff domain, < crt_modulus)

                // ---- NTT CRT0 of gadget piece k ----
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    int z = tid + e * COLLAPSE_THREADS;
                    uint64_t val = composed[e];
                    int bit_offs = k * bits_per;
                    uint64_t piece = (bit_offs >= 64) ? 0 : ((val >> bit_offs) & gadget_mask);
                    ntt_buf[z] = piece;
                }
                __syncthreads();

                forward_ntt_shared(ntt_buf, ntt_tables, 0, mod0, tid);

                uint64_t ntt_crt0[ELEMS_PER_THREAD];
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    ntt_crt0[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
                }
                __syncthreads();

                // ---- NTT CRT1 of gadget piece k ----
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    int z = tid + e * COLLAPSE_THREADS;
                    uint64_t val = composed[e];
                    int bit_offs = k * bits_per;
                    uint64_t piece = (bit_offs >= 64) ? 0 : ((val >> bit_offs) & gadget_mask);
                    ntt_buf[z] = piece;
                }
                __syncthreads();

                forward_ntt_shared(ntt_buf, ntt_tables, 1, mod1, tid);

                // Read NTT CRT1 results from shared
                uint64_t ntt_crt1[ELEMS_PER_THREAD];
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    ntt_crt1[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
                }
                __syncthreads();

                // ---- Write condensed bold_t ----
                // bold_t[col][iter*t_exp+k][z] = (ntt_crt0 & 0xFFFFFFFF) | (ntt_crt1 << 32)
                size_t bt_offset = col_bt_base + (size_t)(iter * t_exp + k) * POLY_LEN;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    int z = tid + e * COLLAPSE_THREADS;
                    bt_out[bt_offset + z] = (ntt_crt0[e] & 0xFFFFFFFFULL) | (ntt_crt1[e] << 32);
                }

                // ---- Multiply-accumulate with w_all ----
                // w_all[iter][k] has coeff_count = 2*POLY_LEN values (CRT0 then CRT1)
                size_t w_base = ((size_t)iter * t_exp + k) * coeff_count;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    int z = tid + e * COLLAPSE_THREADS;
                    uint64_t w0 = w_data[w_base + z];
                    uint64_t w1 = w_data[w_base + POLY_LEN + z];
                    acc_crt0[e] += ntt_crt0[e] * w0;
                    acc_crt1[e] += ntt_crt1[e] * w1;
                }
            }

            // ---- Barrett reduce accumulators ----
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                acc_crt0[e] = barrett_u64_dev(acc_crt0[e], barrett_cr1_0, (uint64_t)mod0);
                acc_crt1[e] = barrett_u64_dev(acc_crt1[e], barrett_cr1_1, (uint64_t)mod1);
            }

            // ---- Add accumulator into r[iter] with Barrett reduction (fast_add_into) ----
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * COLLAPSE_THREADS;
                uint64_t v0 = r_data[dst_base + z] + acc_crt0[e];
                uint64_t v1 = r_data[dst_base + POLY_LEN + z] + acc_crt1[e];
                r_data[dst_base + z] = barrett_u64_dev(v0, barrett_cr1_0, (uint64_t)mod0);
                r_data[dst_base + POLY_LEN + z] = barrett_u64_dev(v1, barrett_cr1_1, (uint64_t)mod1);
            }

            // No __syncthreads needed between paths since each thread works independently
        } // end path loop
    } // end iter loop

    // ============================================================================
    // Final step: gadget_invert(INTT(r_bar_all[0])) → bold_t_hat,
    //             then v_mask × bold_t_hat → r_all[0]
    // ============================================================================
    {
        size_t rbar0_base = col_r_base; // r_bar_all, iter=0

        // INTT CRT0 of r_bar_all[0]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            ntt_buf[tid + e * COLLAPSE_THREADS] = r_bar_all[rbar0_base + tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();
        inverse_ntt_shared(ntt_buf, ntt_tables, 0, mod0, tid);

        uint64_t coeff_crt0[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            coeff_crt0[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();

        // INTT CRT1 of r_bar_all[0]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            ntt_buf[tid + e * COLLAPSE_THREADS] = r_bar_all[rbar0_base + POLY_LEN + tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();
        inverse_ntt_shared(ntt_buf, ntt_tables, 1, mod1, tid);

        uint64_t coeff_crt1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            coeff_crt1[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();

        // CRT compose
        uint64_t composed[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            uint64_t v0 = coeff_crt0[e];
            uint64_t v1 = coeff_crt1[e];
            uint64_t p0_lo = v0 * mod1_inv_mod0;
            uint64_t p0_hi = __umul64hi(v0, mod1_inv_mod0);
            uint64_t p1_lo = v1 * mod0_inv_mod1;
            uint64_t p1_hi = __umul64hi(v1, mod0_inv_mod1);
            uint64_t s_lo = p0_lo + p1_lo;
            uint64_t s_hi = p0_hi + p1_hi + (uint64_t)(s_lo < p0_lo);
            composed[e] = barrett_u128_dev(s_lo, s_hi, barrett_cr0_mod, barrett_cr1_mod, crt_modulus);
        }

        // Gadget + NTT + condense → bold_t_hat, accumulate with v_mask → r_all[0]
        uint64_t acc_crt0[ELEMS_PER_THREAD];
        uint64_t acc_crt1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            acc_crt0[e] = 0;
            acc_crt1[e] = 0;
        }

        for (int k = 0; k < t_exp; k++) {
            // NTT CRT0 of gadget piece
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * COLLAPSE_THREADS;
                int bit_offs = k * bits_per;
                uint64_t piece = (bit_offs >= 64) ? 0 : ((composed[e] >> bit_offs) & gadget_mask);
                ntt_buf[z] = piece;
            }
            __syncthreads();
            forward_ntt_shared(ntt_buf, ntt_tables, 0, mod0, tid);

            uint64_t ntt_crt0[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                ntt_crt0[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
            }
            __syncthreads();

            // NTT CRT1 of gadget piece
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * COLLAPSE_THREADS;
                int bit_offs = k * bits_per;
                uint64_t piece = (bit_offs >= 64) ? 0 : ((composed[e] >> bit_offs) & gadget_mask);
                ntt_buf[z] = piece;
            }
            __syncthreads();
            forward_ntt_shared(ntt_buf, ntt_tables, 1, mod1, tid);

            uint64_t ntt_crt1[ELEMS_PER_THREAD];
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                ntt_crt1[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
            }
            __syncthreads();

            // Write condensed bold_t_hat
            size_t bh_offset = col_bh_base + (size_t)k * POLY_LEN;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * COLLAPSE_THREADS;
                bold_t_hat[bh_offset + z] = (ntt_crt0[e] & 0xFFFFFFFFULL) | (ntt_crt1[e] << 32);
            }

            // Multiply-accumulate with v_mask
            size_t v_base = (size_t)k * coeff_count;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                int z = tid + e * COLLAPSE_THREADS;
                uint64_t vm0 = v_mask[v_base + z];
                uint64_t vm1 = v_mask[v_base + POLY_LEN + z];
                acc_crt0[e] += ntt_crt0[e] * vm0;
                acc_crt1[e] += ntt_crt1[e] * vm1;
            }
        }

        // Barrett reduce and add into r_all[0]
        size_t r0_base = col_r_base; // r_all col, index 0
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int z = tid + e * COLLAPSE_THREADS;
            acc_crt0[e] = barrett_u64_dev(acc_crt0[e], barrett_cr1_0, (uint64_t)mod0);
            acc_crt1[e] = barrett_u64_dev(acc_crt1[e], barrett_cr1_1, (uint64_t)mod1);
            uint64_t v0 = r_all[r0_base + z] + acc_crt0[e];
            uint64_t v1 = r_all[r0_base + POLY_LEN + z] + acc_crt1[e];
            r_all[r0_base + z] = barrett_u64_dev(v0, barrett_cr1_0, (uint64_t)mod0);
            r_all[r0_base + POLY_LEN + z] = barrett_u64_dev(v1, barrett_cr1_1, (uint64_t)mod1);
        }
    }

    // ============================================================================
    // Output a_hat: INTT(r_all[0]) → CRT compose → coefficient domain
    // ============================================================================
    {
        size_t r0_base = col_r_base;

        // Reduce r_all[0] first (accumulated many times without reduction)
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int z = tid + e * COLLAPSE_THREADS;
            r_all[r0_base + z] = barrett_u64_dev(r_all[r0_base + z], barrett_cr1_0, (uint64_t)mod0);
            r_all[r0_base + POLY_LEN + z] = barrett_u64_dev(r_all[r0_base + POLY_LEN + z], barrett_cr1_1, (uint64_t)mod1);
        }

        // INTT CRT0
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            ntt_buf[tid + e * COLLAPSE_THREADS] = r_all[r0_base + tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();
        inverse_ntt_shared(ntt_buf, ntt_tables, 0, mod0, tid);

        uint64_t val0[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            val0[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();

        // INTT CRT1
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            ntt_buf[tid + e * COLLAPSE_THREADS] = r_all[r0_base + POLY_LEN + tid + e * COLLAPSE_THREADS];
        }
        __syncthreads();
        inverse_ntt_shared(ntt_buf, ntt_tables, 1, mod1, tid);

        uint64_t val1[ELEMS_PER_THREAD];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            val1[e] = ntt_buf[tid + e * COLLAPSE_THREADS];
        }

        // CRT compose → a_hat
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int z = tid + e * COLLAPSE_THREADS;
            uint64_t v0 = val0[e];
            uint64_t v1 = val1[e];
            uint64_t p0_lo = v0 * mod1_inv_mod0;
            uint64_t p0_hi = __umul64hi(v0, mod1_inv_mod0);
            uint64_t p1_lo = v1 * mod0_inv_mod1;
            uint64_t p1_hi = __umul64hi(v1, mod0_inv_mod1);
            uint64_t s_lo = p0_lo + p1_lo;
            uint64_t s_hi = p0_hi + p1_hi + (uint64_t)(s_lo < p0_lo);
            a_hat[(size_t)col * POLY_LEN + z] = barrett_u128_dev(s_lo, s_hi, barrett_cr0_mod, barrett_cr1_mod, crt_modulus);
        }
    }
}


// ============================================================================
// Host-side functions
// ============================================================================

// Persistent device pointers
static uint64_t* d_r_all = nullptr;
static uint64_t* d_r_bar_all = nullptr;
static uint64_t* d_w_all = nullptr;
static uint64_t* d_w_bar_all = nullptr;
static uint64_t* d_v_mask = nullptr;
static uint64_t* d_bold_t = nullptr;
static uint64_t* d_bold_t_bar = nullptr;
static uint64_t* d_bold_t_hat = nullptr;
static uint64_t* d_a_hat = nullptr;
static uint64_t* d_ntt_tables_collapse = nullptr;
static uint32_t* d_automorph_tables = nullptr;
static uint32_t* d_automorph_gen_pows = nullptr;

// Saved dimensions
static int saved_num_cols = 0;
static int saved_nphalf = 0;
static int saved_t_exp = 0;

// Secondary bold_t buffers for fold
static uint64_t* d_bold_t_new = nullptr;
static uint64_t* d_bold_t_bar_new = nullptr;
static uint64_t* d_bold_t_hat_new = nullptr;
static uint64_t* d_a_hat_new = nullptr;

// Allocate r_all and r_bar_all buffers on device
extern "C" void gpu_collapse_alloc_r(
    int num_cols,
    int nphalf,
    int coeff_count
) {
    size_t r_size = (size_t)num_cols * nphalf * coeff_count * sizeof(uint64_t);

    if (d_r_all) GPU_FREE(d_r_all);
    if (d_r_bar_all) GPU_FREE(d_r_bar_all);

    CHECK_CUDA(GPU_MALLOC(&d_r_all, r_size));
    CHECK_CUDA(cudaMemset(d_r_all, 0, r_size));
    CHECK_CUDA(GPU_MALLOC(&d_r_bar_all, r_size));
    CHECK_CUDA(cudaMemset(d_r_bar_all, 0, r_size));
}

// Get device pointer to r_all slot for a given column (for automorphism write target)
extern "C" uint64_t* gpu_collapse_get_r_all_col(int col, int nphalf, int coeff_count) {
    return d_r_all + (size_t)col * nphalf * coeff_count;
}

extern "C" uint64_t* gpu_collapse_get_r_bar_all_col(int col, int nphalf, int coeff_count) {
    return d_r_bar_all + (size_t)col * nphalf * coeff_count;
}

// Run automorphism kernel for one column: copy from packing s_r_out → r_all[col]
extern "C" void gpu_collapse_automorph_column(
    const uint64_t* d_r_flat,      // device ptr: s_r_out from packing kernel
    const uint64_t* d_r_bar_flat,  // device ptr: s_r_bar_out
    int col,
    int nphalf,
    int coeff_count,
    int poly_len,
    int crt_count,
    const uint32_t* d_tables,      // from rotation kernel (already on device)
    const uint32_t* d_gen_pows,    // from rotation kernel (already on device)
    int num_tables
) {
    // Launch automorphism kernel for this single column
    // Grid: (1, nphalf), Block: poly_len/2
    dim3 grid(1, nphalf);
    dim3 block(poly_len / 2);

    uint64_t* r_all_col = d_r_all + (size_t)col * nphalf * coeff_count;
    uint64_t* r_bar_all_col = d_r_bar_all + (size_t)col * nphalf * coeff_count;

    automorphism_kernel<<<grid, block>>>(
        r_all_col, r_bar_all_col,
        d_r_flat, d_r_bar_flat, d_tables, d_gen_pows,
        nphalf, poly_len, crt_count
    );
    // Don't synchronize — let it overlap with CPU work
}

// Setup collapse: upload weights, NTT tables, allocate bold_t outputs
extern "C" void gpu_collapse_setup(
    const uint64_t* h_w_all,       // [nphalf_m1 * t_exp * coeff_count]
    const uint64_t* h_w_bar_all,   // same
    const uint64_t* h_v_mask,      // [t_exp * coeff_count]
    const uint64_t* h_ntt_tables,  // [2 * 4 * POLY_LEN]
    int num_cols,
    int nphalf,
    int t_exp,
    int coeff_count
) {
    saved_num_cols = num_cols;
    saved_nphalf = nphalf;
    saved_t_exp = t_exp;

    int nphalf_m1 = nphalf - 1;

    size_t w_size = (size_t)nphalf_m1 * t_exp * coeff_count * sizeof(uint64_t);
    size_t v_size = (size_t)t_exp * coeff_count * sizeof(uint64_t);
    size_t tbl_size = 2 * 4 * POLY_LEN * sizeof(uint64_t);
    size_t bt_size = (size_t)num_cols * nphalf_m1 * t_exp * POLY_LEN * sizeof(uint64_t);
    size_t bh_size = (size_t)num_cols * t_exp * POLY_LEN * sizeof(uint64_t);
    size_t ah_size = (size_t)num_cols * POLY_LEN * sizeof(uint64_t);

    // Upload weights
    if (d_w_all) GPU_FREE(d_w_all);
    if (d_w_bar_all) GPU_FREE(d_w_bar_all);
    if (d_v_mask) GPU_FREE(d_v_mask);
    CHECK_CUDA(GPU_MALLOC(&d_w_all, w_size));
    CHECK_CUDA(GPU_MALLOC(&d_w_bar_all, w_size));
    CHECK_CUDA(GPU_MALLOC(&d_v_mask, v_size));
    CHECK_CUDA(cudaMemcpy(d_w_all, h_w_all, w_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_bar_all, h_w_bar_all, w_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_mask, h_v_mask, v_size, cudaMemcpyHostToDevice));

    // Upload NTT tables
    if (d_ntt_tables_collapse) GPU_FREE(d_ntt_tables_collapse);
    CHECK_CUDA(GPU_MALLOC(&d_ntt_tables_collapse, tbl_size));
    CHECK_CUDA(cudaMemcpy(d_ntt_tables_collapse, h_ntt_tables, tbl_size, cudaMemcpyHostToDevice));

    // Allocate output buffers
    if (d_bold_t) GPU_FREE(d_bold_t);
    if (d_bold_t_bar) GPU_FREE(d_bold_t_bar);
    if (d_bold_t_hat) GPU_FREE(d_bold_t_hat);
    if (d_a_hat) GPU_FREE(d_a_hat);
    CHECK_CUDA(GPU_MALLOC(&d_bold_t, bt_size));
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_bar, bt_size));
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_hat, bh_size));
    CHECK_CUDA(GPU_MALLOC(&d_a_hat, ah_size));

}

// Run the collapse kernel
extern "C" void gpu_collapse_run(
    int num_cols,
    int nphalf,
    int t_exp,
    int bits_per,
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t crt_modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    if (!d_r_all || !d_w_all || !d_ntt_tables_collapse) {
        fprintf(stderr, "[GPU Collapse] Data not set up!\n");
        return;
    }

    // Ensure automorphism kernels have finished
    CHECK_CUDA(GPU_STREAM_SYNC());

    size_t shared_mem = POLY_LEN * sizeof(uint64_t);  // 16 KB

    collapse_kernel<<<num_cols, COLLAPSE_THREADS, shared_mem>>>(
        d_r_all, d_r_bar_all,
        d_bold_t, d_bold_t_bar, d_bold_t_hat, d_a_hat,
        d_w_all, d_w_bar_all, d_v_mask,
        d_ntt_tables_collapse,
        num_cols, nphalf, t_exp, bits_per,
        mod0, mod1,
        barrett_cr1_0, barrett_cr1_1,
        mod0_inv_mod1, mod1_inv_mod0,
        crt_modulus, barrett_cr0_mod, barrett_cr1_mod
    );

    CHECK_CUDA(GPU_STREAM_SYNC());
}

// Download a_hat to host
extern "C" void gpu_collapse_download_a_hat(
    uint64_t* h_a_hat,
    int num_cols
) {
    size_t size = (size_t)num_cols * POLY_LEN * sizeof(uint64_t);
    CHECK_CUDA(cudaMemcpy(h_a_hat, d_a_hat, size, cudaMemcpyDeviceToHost));
}

// Get device pointers for bold_t (for online kernel to adopt)
extern "C" uint64_t* gpu_collapse_get_bold_t() { return d_bold_t; }
extern "C" uint64_t* gpu_collapse_get_bold_t_bar() { return d_bold_t_bar; }
extern "C" uint64_t* gpu_collapse_get_bold_t_hat() { return d_bold_t_hat; }

// Free working buffers (r_all, w_all, etc.) but keep bold_t for online phase
extern "C" void gpu_collapse_free_working() {
    if (d_r_all) { GPU_FREE(d_r_all); d_r_all = nullptr; }
    if (d_r_bar_all) { GPU_FREE(d_r_bar_all); d_r_bar_all = nullptr; }
    if (d_w_all) { GPU_FREE(d_w_all); d_w_all = nullptr; }
    if (d_w_bar_all) { GPU_FREE(d_w_bar_all); d_w_bar_all = nullptr; }
    if (d_v_mask) { GPU_FREE(d_v_mask); d_v_mask = nullptr; }
    if (d_ntt_tables_collapse) { GPU_FREE(d_ntt_tables_collapse); d_ntt_tables_collapse = nullptr; }
    if (d_a_hat) { GPU_FREE(d_a_hat); d_a_hat = nullptr; }
    if (d_automorph_tables) { GPU_FREE(d_automorph_tables); d_automorph_tables = nullptr; }
    if (d_automorph_gen_pows) { GPU_FREE(d_automorph_gen_pows); d_automorph_gen_pows = nullptr; }
}

// Free everything including bold_t
extern "C" void gpu_collapse_free_all() {
    gpu_collapse_free_working();
    if (d_bold_t) { GPU_FREE(d_bold_t); d_bold_t = nullptr; }
    if (d_bold_t_bar) { GPU_FREE(d_bold_t_bar); d_bold_t_bar = nullptr; }
    if (d_bold_t_hat) { GPU_FREE(d_bold_t_hat); d_bold_t_hat = nullptr; }
}

// Accessors for r_all/r_bar_all device pointers (used by batched packing kernel)
extern "C" uint64_t* gpu_collapse_get_r_all() { return d_r_all; }
extern "C" uint64_t* gpu_collapse_get_r_bar_all() { return d_r_bar_all; }

// ============================================================================
// Batched in-place automorphism kernel
// ============================================================================
// Permutes each polynomial in r_all (or r_bar_all) using automorphism tables.
// Uses shared memory as temporary buffer for in-place permutation.
// Grid: num_cols * nphalf blocks, poly_len/2 threads per block.

__global__ void batched_automorphism_inplace_kernel(
    uint64_t* __restrict__ data,           // [num_cols * nphalf * coeff_count]
    const uint32_t* __restrict__ tables,   // [num_tables * poly_len]
    const uint32_t* __restrict__ gen_pows, // [nphalf]
    int nphalf, int poly_len, int crt_count,
    int is_rbar  // 0: table_idx = (gp-1)/2, 1: table_idx = (2*poly_len-gp-1)/2
) {
    extern __shared__ uint64_t abuf[];  // [poly_len]

    int block_id = blockIdx.x;
    int col = block_id / nphalf;
    int i = block_id % nphalf;
    int half = poly_len / 2;
    int z0 = threadIdx.x;
    int z1 = z0 + half;
    int coeff_count = crt_count * poly_len;

    if (z0 >= half) return;

    uint32_t gp = gen_pows[i];
    uint32_t table_idx = is_rbar ? (2 * poly_len - gp - 1) / 2 : (gp - 1) / 2;

    size_t base = ((size_t)col * nphalf + i) * coeff_count;
    size_t t_base = (size_t)table_idx * poly_len;

    for (int crt = 0; crt < crt_count; crt++) {
        size_t crt_off = (size_t)crt * poly_len;

        // Load polynomial into shared memory
        abuf[z0] = data[base + crt_off + z0];
        abuf[z1] = data[base + crt_off + z1];
        __syncthreads();

        // Write back permuted: output[z] = buf[table[z]]
        uint32_t src0 = tables[t_base + z0];
        uint32_t src1 = tables[t_base + z1];
        data[base + crt_off + z0] = abuf[src0];
        data[base + crt_off + z1] = abuf[src1];
        __syncthreads();
    }
}

// Host function: run batched in-place automorphism on r_all and r_bar_all
extern "C" void gpu_collapse_batched_automorphism(
    const uint32_t* d_tables,
    const uint32_t* d_gen_pows,
    int num_cols, int nphalf, int poly_len, int crt_count
) {
    if (!d_r_all || !d_r_bar_all) {
        fprintf(stderr, "[GPU Collapse] r_all not allocated for automorphism!\n");
        return;
    }

    int half = poly_len / 2;
    int blocks = num_cols * nphalf;
    int threads = half;
    size_t smem = poly_len * sizeof(uint64_t);  // 16 KB

    // r path
    batched_automorphism_inplace_kernel<<<blocks, threads, smem>>>(
        d_r_all, d_tables, d_gen_pows, nphalf, poly_len, crt_count, 0);
    // r_bar path
    batched_automorphism_inplace_kernel<<<blocks, threads, smem>>>(
        d_r_bar_all, d_tables, d_gen_pows, nphalf, poly_len, crt_count, 1);

    CHECK_CUDA(GPU_STREAM_SYNC());
}

// ============================================================================
// Fold support: secondary bold_t buffers
// ============================================================================

// Setup fold collapse: upload weights, NTT tables, allocate NEW bold_t outputs
// Reuses working buffers (r_all etc.) — caller must ensure no overlap
extern "C" void gpu_collapse_setup_fold(
    const uint64_t* h_w_all,
    const uint64_t* h_w_bar_all,
    const uint64_t* h_v_mask,
    const uint64_t* h_ntt_tables,
    int num_cols,
    int nphalf,
    int t_exp,
    int coeff_count
) {
    saved_num_cols = num_cols;
    saved_nphalf = nphalf;
    saved_t_exp = t_exp;

    int nphalf_m1 = nphalf - 1;

    size_t w_size = (size_t)nphalf_m1 * t_exp * coeff_count * sizeof(uint64_t);
    size_t v_size = (size_t)t_exp * coeff_count * sizeof(uint64_t);
    size_t tbl_size = 2 * 4 * POLY_LEN * sizeof(uint64_t);
    size_t bt_size = (size_t)num_cols * nphalf_m1 * t_exp * POLY_LEN * sizeof(uint64_t);
    size_t bh_size = (size_t)num_cols * t_exp * POLY_LEN * sizeof(uint64_t);
    size_t ah_size = (size_t)num_cols * POLY_LEN * sizeof(uint64_t);

    // Upload weights (reuse same statics — working buffers)
    if (d_w_all) GPU_FREE(d_w_all);
    if (d_w_bar_all) GPU_FREE(d_w_bar_all);
    if (d_v_mask) GPU_FREE(d_v_mask);
    CHECK_CUDA(GPU_MALLOC(&d_w_all, w_size));
    CHECK_CUDA(GPU_MALLOC(&d_w_bar_all, w_size));
    CHECK_CUDA(GPU_MALLOC(&d_v_mask, v_size));
    CHECK_CUDA(cudaMemcpy(d_w_all, h_w_all, w_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_bar_all, h_w_bar_all, w_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_mask, h_v_mask, v_size, cudaMemcpyHostToDevice));

    // Upload NTT tables
    if (d_ntt_tables_collapse) GPU_FREE(d_ntt_tables_collapse);
    CHECK_CUDA(GPU_MALLOC(&d_ntt_tables_collapse, tbl_size));
    CHECK_CUDA(cudaMemcpy(d_ntt_tables_collapse, h_ntt_tables, tbl_size, cudaMemcpyHostToDevice));

    // Allocate NEW output buffers (don't touch old d_bold_t etc.)
    if (d_bold_t_new) GPU_FREE(d_bold_t_new);
    if (d_bold_t_bar_new) GPU_FREE(d_bold_t_bar_new);
    if (d_bold_t_hat_new) GPU_FREE(d_bold_t_hat_new);
    if (d_a_hat_new) GPU_FREE(d_a_hat_new);
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_new, bt_size));
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_bar_new, bt_size));
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_hat_new, bh_size));
    CHECK_CUDA(GPU_MALLOC(&d_a_hat_new, ah_size));
}

// Run the collapse kernel writing to NEW bold_t buffers
extern "C" void gpu_collapse_run_fold(
    int num_cols,
    int nphalf,
    int t_exp,
    int bits_per,
    uint32_t mod0, uint32_t mod1,
    uint64_t barrett_cr1_0, uint64_t barrett_cr1_1,
    uint64_t mod0_inv_mod1, uint64_t mod1_inv_mod0,
    uint64_t crt_modulus,
    uint64_t barrett_cr0_mod, uint64_t barrett_cr1_mod
) {
    if (!d_r_all || !d_w_all || !d_ntt_tables_collapse) {
        fprintf(stderr, "[GPU Collapse Fold] Data not set up!\n");
        return;
    }

    CHECK_CUDA(GPU_STREAM_SYNC());

    size_t shared_mem = POLY_LEN * sizeof(uint64_t);

    collapse_kernel<<<num_cols, COLLAPSE_THREADS, shared_mem>>>(
        d_r_all, d_r_bar_all,
        d_bold_t_new, d_bold_t_bar_new, d_bold_t_hat_new, d_a_hat_new,
        d_w_all, d_w_bar_all, d_v_mask,
        d_ntt_tables_collapse,
        num_cols, nphalf, t_exp, bits_per,
        mod0, mod1,
        barrett_cr1_0, barrett_cr1_1,
        mod0_inv_mod1, mod1_inv_mod0,
        crt_modulus, barrett_cr0_mod, barrett_cr1_mod
    );

    CHECK_CUDA(GPU_STREAM_SYNC());
}

// Download a_hat from the NEW buffers
extern "C" void gpu_collapse_download_a_hat_new(
    uint64_t* h_a_hat,
    int num_cols
) {
    size_t size = (size_t)num_cols * POLY_LEN * sizeof(uint64_t);
    CHECK_CUDA(cudaMemcpy(h_a_hat, d_a_hat_new, size, cudaMemcpyDeviceToHost));
}

// Get device pointers for NEW bold_t (for online kernel swap)
extern "C" uint64_t* gpu_collapse_get_bold_t_new() { return d_bold_t_new; }
extern "C" uint64_t* gpu_collapse_get_bold_t_bar_new() { return d_bold_t_bar_new; }
extern "C" uint64_t* gpu_collapse_get_bold_t_hat_new() { return d_bold_t_hat_new; }

// Swap: free old primary bold_t, promote new to primary
extern "C" void gpu_collapse_swap_bold_t() {
    // Free old primary
    if (d_bold_t) GPU_FREE(d_bold_t);
    if (d_bold_t_bar) GPU_FREE(d_bold_t_bar);
    if (d_bold_t_hat) GPU_FREE(d_bold_t_hat);

    // Promote new to primary
    d_bold_t = d_bold_t_new;
    d_bold_t_bar = d_bold_t_bar_new;
    d_bold_t_hat = d_bold_t_hat_new;

    d_bold_t_new = nullptr;
    d_bold_t_bar_new = nullptr;
    d_bold_t_hat_new = nullptr;

    // Also free a_hat_new (its data was already downloaded to host)
    if (d_a_hat_new) { GPU_FREE(d_a_hat_new); d_a_hat_new = nullptr; }
}

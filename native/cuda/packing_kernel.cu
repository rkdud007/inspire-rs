// GPU-accelerated InspiRING packing precomputation kernel
//
// Computes the r_all and r_bar_all arrays from full_packing_with_preprocessing_offline.
// This is a batched indexed multiply-accumulate:
//   For each i in 0..num_to_pack_half, for each coeff k in 0..coeff_count:
//     r[i][k] = sum_{j=0..non_zeros} monomial_table[idx[i][j]][k] * a_ct[j][k]
//     r_bar[i][k] = sum_{j=0..non_zeros} monomial_table[idx_bar[i][j]][k] * a_ct[j][k]
//   Then: Barrett reduce, scalar multiply by mod_inv, and apply automorphism.
//
// The monomial_table has 2*poly_len entries (positive and negative monomials in NTT form).
// The index tables idx[i][j] and idx_bar[i][j] are precomputed on the host.

#include "common.h"

// Main kernel: batched indexed multiply-accumulate with Barrett reduction
// Each thread handles one (i, k) pair
// For each j in 0..non_zeros:
//   r[i*coeff_count+k] += monomial_table[idx[i*non_zeros+j]*coeff_count+k] * a_ct[j*coeff_count+k]
// With periodic Barrett reduction to prevent overflow
__global__ void packing_mul_acc_kernel(
    uint64_t* __restrict__ r_out,           // [num_to_pack_half, coeff_count]
    uint64_t* __restrict__ r_bar_out,       // [num_to_pack_half, coeff_count]
    const uint64_t* __restrict__ mono_table,// [2*poly_len, coeff_count]
    const uint64_t* __restrict__ a_ct,      // [non_zeros, coeff_count]
    const int* __restrict__ idx,            // [num_to_pack_half, non_zeros]
    const int* __restrict__ idx_bar,        // [num_to_pack_half, non_zeros]
    int num_to_pack_half,
    int non_zeros,
    int coeff_count,                        // = crt_count * poly_len
    int poly_len,
    int crt_count,
    const uint64_t* __restrict__ moduli,    // [crt_count]
    const uint64_t* __restrict__ barrett_crs // [crt_count]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_to_pack_half * coeff_count;
    if (tid >= total) return;

    int i = tid / coeff_count;
    int k = tid % coeff_count;

    // Determine which CRT limb this coefficient belongs to
    int crt_idx = k / poly_len;
    uint64_t mod_val = moduli[crt_idx];
    uint64_t cr = barrett_crs[crt_idx];

    // Reduction interval: reduce every N steps to prevent u64 overflow
    // Each multiply gives ~56 bits, accumulating N of them needs ~56 + log2(N) bits
    // With u64, we can safely accumulate ~256 products before reduction
    const int REDUCE_INTERVAL = 128;

    uint64_t acc_r = 0;
    uint64_t acc_r_bar = 0;

    const int* my_idx = idx + (size_t)i * non_zeros;
    const int* my_idx_bar = idx_bar + (size_t)i * non_zeros;

    for (int j = 0; j < non_zeros; j++) {
        int mono_idx = my_idx[j];
        int mono_idx_bar = my_idx_bar[j];

        uint64_t mono_val = mono_table[(size_t)mono_idx * coeff_count + k];
        uint64_t mono_val_bar = mono_table[(size_t)mono_idx_bar * coeff_count + k];
        uint64_t a_val = a_ct[(size_t)j * coeff_count + k];

        // Lazy multiply (no reduction yet)
        acc_r += mono_val * a_val;
        acc_r_bar += mono_val_bar * a_val;

        if ((j + 1) % REDUCE_INTERVAL == 0) {
            acc_r = barrett_reduce(acc_r, mod_val, cr);
            acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);
        }
    }

    // Final reduction
    acc_r = barrett_reduce(acc_r, mod_val, cr);
    acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);

    r_out[(size_t)i * coeff_count + k] = acc_r;
    r_bar_out[(size_t)i * coeff_count + k] = acc_r_bar;
}


// Scalar multiply kernel: r[i][k] = (r[i][k] * mod_inv[k]) mod q
__global__ void scalar_mul_kernel(
    uint64_t* __restrict__ data,            // [count, coeff_count]
    const uint64_t* __restrict__ mod_inv,   // [coeff_count]
    int total_elems,
    int coeff_count,
    int poly_len,
    int crt_count,
    const uint64_t* __restrict__ moduli,
    const uint64_t* __restrict__ barrett_crs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elems) return;

    int k = tid % coeff_count;
    int crt_idx = k / poly_len;
    uint64_t mod_val = moduli[crt_idx];
    uint64_t cr = barrett_crs[crt_idx];

    uint64_t val = data[tid];
    uint64_t inv = mod_inv[k];
    // Modular multiply using 128-bit intermediate
    unsigned __int128 prod = (unsigned __int128)val * inv;
    uint64_t result = (uint64_t)(prod % mod_val);
    data[tid] = result;
}


// ============================================================================
// Persistent shared data for amortized per-column calls
// ============================================================================
static uint64_t* s_mono_table = nullptr;
static int*      s_idx = nullptr;
static int*      s_idx_bar = nullptr;
static uint64_t* s_mod_inv = nullptr;
static uint64_t* s_moduli = nullptr;
static uint64_t* s_barrett_crs = nullptr;
// Pre-allocated per-column buffers (reused across calls)
static uint64_t* s_a_ct = nullptr;
static uint64_t* s_r_out = nullptr;
static uint64_t* s_r_bar_out = nullptr;
// Cached dimensions
static int s_num_to_pack_half = 0;
static int s_non_zeros = 0;
static int s_coeff_count = 0;
static int s_poly_len = 0;
static int s_crt_count = 0;

// Setup: upload shared data once and pre-allocate per-column buffers
extern "C" void gpu_packing_setup(
    const uint64_t* h_mono_table,
    const int* h_idx,
    const int* h_idx_bar,
    const uint64_t* h_mod_inv,
    int num_to_pack_half,
    int non_zeros,
    int poly_len,
    int crt_count,
    uint64_t mod0,
    uint64_t mod1,
    uint64_t barrett_cr0,
    uint64_t barrett_cr1
) {
    int coeff_count = crt_count * poly_len;
    int total_monos = 2 * poly_len;

    s_num_to_pack_half = num_to_pack_half;
    s_non_zeros = non_zeros;
    s_coeff_count = coeff_count;
    s_poly_len = poly_len;
    s_crt_count = crt_count;

    size_t mono_size = (size_t)total_monos * coeff_count * sizeof(uint64_t);
    size_t idx_size = (size_t)num_to_pack_half * non_zeros * sizeof(int);
    size_t inv_size = (size_t)coeff_count * sizeof(uint64_t);
    size_t act_size = (size_t)non_zeros * coeff_count * sizeof(uint64_t);
    size_t out_size = (size_t)num_to_pack_half * coeff_count * sizeof(uint64_t);

#define CHECK_CUDA_S(call) CHECK_CUDA_TAG("GPU Packing", call)

    // Free previous buffers if any (prevents leak when called per-fold)
    if (s_mono_table) GPU_FREE(s_mono_table);
    if (s_idx) GPU_FREE(s_idx);
    if (s_idx_bar) GPU_FREE(s_idx_bar);
    if (s_mod_inv) GPU_FREE(s_mod_inv);
    if (s_moduli) GPU_FREE(s_moduli);
    if (s_barrett_crs) GPU_FREE(s_barrett_crs);
    if (s_a_ct) GPU_FREE(s_a_ct);
    if (s_r_out) GPU_FREE(s_r_out);
    if (s_r_bar_out) GPU_FREE(s_r_bar_out);

    // Shared data
    CHECK_CUDA_S(GPU_MALLOC(&s_mono_table, mono_size));
    CHECK_CUDA_S(GPU_MALLOC(&s_idx, idx_size));
    CHECK_CUDA_S(GPU_MALLOC(&s_idx_bar, idx_size));
    CHECK_CUDA_S(GPU_MALLOC(&s_mod_inv, inv_size));
    CHECK_CUDA_S(GPU_MALLOC(&s_moduli, crt_count * sizeof(uint64_t)));
    CHECK_CUDA_S(GPU_MALLOC(&s_barrett_crs, crt_count * sizeof(uint64_t)));

    cudaMemcpy(s_mono_table, h_mono_table, mono_size, cudaMemcpyHostToDevice);
    cudaMemcpy(s_idx, h_idx, idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(s_idx_bar, h_idx_bar, idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(s_mod_inv, h_mod_inv, inv_size, cudaMemcpyHostToDevice);

    uint64_t h_moduli[2] = {mod0, mod1};
    uint64_t h_barrett_crs[2] = {barrett_cr0, barrett_cr1};
    cudaMemcpy(s_moduli, h_moduli, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(s_barrett_crs, h_barrett_crs, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Pre-allocate per-column buffers (reused)
    CHECK_CUDA_S(GPU_MALLOC(&s_a_ct, act_size));
    CHECK_CUDA_S(GPU_MALLOC(&s_r_out, out_size));
    CHECK_CUDA_S(GPU_MALLOC(&s_r_bar_out, out_size));

}

// Per-column call: only uploads a_ct and downloads results (no alloc/free, no shared re-upload)
extern "C" void gpu_packing_precomp_fast(
    const uint64_t* h_a_ct,
    uint64_t* h_r_out,
    uint64_t* h_r_bar_out
) {
    size_t act_size = (size_t)s_non_zeros * s_coeff_count * sizeof(uint64_t);
    size_t out_size = (size_t)s_num_to_pack_half * s_coeff_count * sizeof(uint64_t);

    // Upload per-column data
    cudaMemcpy(s_a_ct, h_a_ct, act_size, cudaMemcpyHostToDevice);

    // Launch multiply-accumulate kernel
    int total = s_num_to_pack_half * s_coeff_count;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    packing_mul_acc_kernel<<<blocks, threads>>>(
        s_r_out, s_r_bar_out, s_mono_table, s_a_ct,
        s_idx, s_idx_bar,
        s_num_to_pack_half, s_non_zeros, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );

    // Scalar multiply both outputs
    scalar_mul_kernel<<<blocks, threads>>>(
        s_r_out, s_mod_inv, total, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );
    scalar_mul_kernel<<<blocks, threads>>>(
        s_r_bar_out, s_mod_inv, total, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );
    GPU_STREAM_SYNC();

    // Download results
    cudaMemcpy(h_r_out, s_r_out, out_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_bar_out, s_r_bar_out, out_size, cudaMemcpyDeviceToHost);
}

// Per-column call with device-resident a_ct (skips H2D upload)
extern "C" void gpu_packing_precomp_fast_device(
    const uint64_t* d_a_ct_device,  // device pointer (from prep_pack output)
    uint64_t* h_r_out,
    uint64_t* h_r_bar_out
) {
    size_t out_size = (size_t)s_num_to_pack_half * s_coeff_count * sizeof(uint64_t);

    // Launch multiply-accumulate kernel directly with device pointer
    int total = s_num_to_pack_half * s_coeff_count;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    packing_mul_acc_kernel<<<blocks, threads>>>(
        s_r_out, s_r_bar_out, s_mono_table, d_a_ct_device,
        s_idx, s_idx_bar,
        s_num_to_pack_half, s_non_zeros, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );

    // Scalar multiply both outputs
    scalar_mul_kernel<<<blocks, threads>>>(
        s_r_out, s_mod_inv, total, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );
    scalar_mul_kernel<<<blocks, threads>>>(
        s_r_bar_out, s_mod_inv, total, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );
    GPU_STREAM_SYNC();

    // Download results
    cudaMemcpy(h_r_out, s_r_out, out_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_bar_out, s_r_bar_out, out_size, cudaMemcpyDeviceToHost);
}

// Fused packing kernel: u32 multiply + fused scalar_mul in a single kernel.
// Used by the no-d2h path for maximum efficiency (1 kernel launch instead of 3).
__global__ void packing_mul_acc_fused_kernel(
    uint64_t* __restrict__ r_out,
    uint64_t* __restrict__ r_bar_out,
    const uint64_t* __restrict__ mono_table,
    const uint64_t* __restrict__ a_ct,
    const int* __restrict__ idx,
    const int* __restrict__ idx_bar,
    const uint64_t* __restrict__ mod_inv,
    int num_to_pack_half,
    int non_zeros,
    int coeff_count,
    int poly_len,
    int crt_count,
    const uint64_t* __restrict__ moduli,
    const uint64_t* __restrict__ barrett_crs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_to_pack_half * coeff_count;
    if (tid >= total) return;

    int i = tid / coeff_count;
    int k = tid % coeff_count;

    int crt_idx = k / poly_len;
    uint64_t mod_val = moduli[crt_idx];
    uint64_t cr = barrett_crs[crt_idx];

    const int REDUCE_INTERVAL = 128;

    uint64_t acc_r = 0;
    uint64_t acc_r_bar = 0;

    const int* my_idx = idx + (size_t)i * non_zeros;
    const int* my_idx_bar = idx_bar + (size_t)i * non_zeros;

    for (int j = 0; j < non_zeros; j++) {
        int mono_idx = my_idx[j];
        int mono_idx_bar = my_idx_bar[j];

        // u32 narrowing: NTT values < q < 2^28
        uint32_t mono_val = (uint32_t)mono_table[(size_t)mono_idx * coeff_count + k];
        uint32_t mono_val_bar = (uint32_t)mono_table[(size_t)mono_idx_bar * coeff_count + k];
        uint32_t a_val = (uint32_t)a_ct[(size_t)j * coeff_count + k];

        // u32 x u32 -> u64 multiply-accumulate
        acc_r += (uint64_t)mono_val * (uint64_t)a_val;
        acc_r_bar += (uint64_t)mono_val_bar * (uint64_t)a_val;

        if ((j + 1) % REDUCE_INTERVAL == 0) {
            acc_r = barrett_reduce(acc_r, mod_val, cr);
            acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);
        }
    }

    // Final Barrett reduction
    acc_r = barrett_reduce(acc_r, mod_val, cr);
    acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);

    // Fused scalar multiply: val * mod_inv[k] mod q (Barrett u64, no u128 needed)
    uint32_t inv = (uint32_t)mod_inv[k];
    acc_r = barrett_reduce((uint64_t)(uint32_t)acc_r * (uint64_t)inv, mod_val, cr);
    acc_r_bar = barrett_reduce((uint64_t)(uint32_t)acc_r_bar * (uint64_t)inv, mod_val, cr);

    r_out[(size_t)i * coeff_count + k] = acc_r;
    r_bar_out[(size_t)i * coeff_count + k] = acc_r_bar;
}

// Per-column call with device-resident a_ct: runs kernel but skips D2H.
// Results stay in s_r_out/s_r_bar_out for the collapse automorphism kernel.
extern "C" void gpu_packing_precomp_fast_device_no_d2h(
    const uint64_t* d_a_ct_device
) {
    int total = s_num_to_pack_half * s_coeff_count;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    packing_mul_acc_kernel<<<blocks, threads>>>(
        s_r_out, s_r_bar_out, s_mono_table, d_a_ct_device,
        s_idx, s_idx_bar,
        s_num_to_pack_half, s_non_zeros, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );

    scalar_mul_kernel<<<blocks, threads>>>(
        s_r_out, s_mod_inv, total, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );
    scalar_mul_kernel<<<blocks, threads>>>(
        s_r_bar_out, s_mod_inv, total, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );
    GPU_STREAM_SYNC();
    // No D2H copy — results stay in s_r_out/s_r_bar_out
}

// Teardown: free all persistent data
extern "C" void gpu_packing_teardown() {
    if (s_mono_table) { GPU_FREE(s_mono_table); s_mono_table = nullptr; }
    if (s_idx) { GPU_FREE(s_idx); s_idx = nullptr; }
    if (s_idx_bar) { GPU_FREE(s_idx_bar); s_idx_bar = nullptr; }
    if (s_mod_inv) { GPU_FREE(s_mod_inv); s_mod_inv = nullptr; }
    if (s_moduli) { GPU_FREE(s_moduli); s_moduli = nullptr; }
    if (s_barrett_crs) { GPU_FREE(s_barrett_crs); s_barrett_crs = nullptr; }
    if (s_a_ct) { GPU_FREE(s_a_ct); s_a_ct = nullptr; }
    if (s_r_out) { GPU_FREE(s_r_out); s_r_out = nullptr; }
    if (s_r_bar_out) { GPU_FREE(s_r_bar_out); s_r_bar_out = nullptr; }
    s_num_to_pack_half = 0;
}

// Original host entry point (unchanged, for backward compat)
extern "C" void gpu_packing_precomp(
    const uint64_t* h_mono_table,
    const uint64_t* h_a_ct,
    const int* h_idx,
    const int* h_idx_bar,
    const uint64_t* h_mod_inv,
    uint64_t* h_r_out,
    uint64_t* h_r_bar_out,
    int num_to_pack_half,
    int non_zeros,
    int poly_len,
    int crt_count,
    uint64_t mod0,
    uint64_t mod1,
    uint64_t barrett_cr0,
    uint64_t barrett_cr1
) {
    int coeff_count = crt_count * poly_len;
    int total_monos = 2 * poly_len;

    uint64_t *d_mono_table, *d_a_ct, *d_r_out, *d_r_bar_out, *d_mod_inv;
    int *d_idx, *d_idx_bar;
    uint64_t *d_moduli, *d_barrett_crs;

    size_t mono_size = (size_t)total_monos * coeff_count * sizeof(uint64_t);
    size_t act_size = (size_t)non_zeros * coeff_count * sizeof(uint64_t);
    size_t out_size = (size_t)num_to_pack_half * coeff_count * sizeof(uint64_t);
    size_t idx_size = (size_t)num_to_pack_half * non_zeros * sizeof(int);
    size_t inv_size = (size_t)coeff_count * sizeof(uint64_t);

#define CHECK_CUDA(call) CHECK_CUDA_TAG("GPU Packing", call)

    CHECK_CUDA(GPU_MALLOC(&d_mono_table, mono_size));
    CHECK_CUDA(GPU_MALLOC(&d_a_ct, act_size));
    CHECK_CUDA(GPU_MALLOC(&d_r_out, out_size));
    CHECK_CUDA(GPU_MALLOC(&d_r_bar_out, out_size));
    CHECK_CUDA(GPU_MALLOC(&d_idx, idx_size));
    CHECK_CUDA(GPU_MALLOC(&d_idx_bar, idx_size));
    CHECK_CUDA(GPU_MALLOC(&d_mod_inv, inv_size));
    CHECK_CUDA(GPU_MALLOC(&d_moduli, crt_count * sizeof(uint64_t)));
    CHECK_CUDA(GPU_MALLOC(&d_barrett_crs, crt_count * sizeof(uint64_t)));

    cudaMemcpy(d_mono_table, h_mono_table, mono_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_ct, h_a_ct, act_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_bar, h_idx_bar, idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod_inv, h_mod_inv, inv_size, cudaMemcpyHostToDevice);

    uint64_t h_moduli[2] = {mod0, mod1};
    uint64_t h_barrett_crs[2] = {barrett_cr0, barrett_cr1};
    cudaMemcpy(d_moduli, h_moduli, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_barrett_crs, h_barrett_crs, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

    {
        int total = num_to_pack_half * coeff_count;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        packing_mul_acc_kernel<<<blocks, threads>>>(
            d_r_out, d_r_bar_out, d_mono_table, d_a_ct,
            d_idx, d_idx_bar,
            num_to_pack_half, non_zeros, coeff_count,
            poly_len, crt_count, d_moduli, d_barrett_crs
        );
        GPU_STREAM_SYNC();
    }

    {
        int total = num_to_pack_half * coeff_count;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        scalar_mul_kernel<<<blocks, threads>>>(
            d_r_out, d_mod_inv, total, coeff_count,
            poly_len, crt_count, d_moduli, d_barrett_crs
        );
        scalar_mul_kernel<<<blocks, threads>>>(
            d_r_bar_out, d_mod_inv, total, coeff_count,
            poly_len, crt_count, d_moduli, d_barrett_crs
        );
        GPU_STREAM_SYNC();
    }

    cudaMemcpy(h_r_out, d_r_out, out_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_bar_out, d_r_bar_out, out_size, cudaMemcpyDeviceToHost);

    GPU_FREE(d_mono_table);
    GPU_FREE(d_a_ct);
    GPU_FREE(d_r_out);
    GPU_FREE(d_r_bar_out);
    GPU_FREE(d_idx);
    GPU_FREE(d_idx_bar);
    GPU_FREE(d_mod_inv);
    GPU_FREE(d_moduli);
    GPU_FREE(d_barrett_crs);
}


// ============================================================================
// Batched version: processes all columns in a single GPU call
// ============================================================================

// Batched mul-acc kernel: each thread handles one (batch, i, k) triple
__global__ void packing_mul_acc_kernel_batched(
    uint64_t* __restrict__ r_out,           // [batch_count * num_to_pack_half * coeff_count]
    uint64_t* __restrict__ r_bar_out,       // [batch_count * num_to_pack_half * coeff_count]
    const uint64_t* __restrict__ mono_table,// [2*poly_len * coeff_count] (shared across batches)
    const uint64_t* __restrict__ a_ct,      // [batch_count * non_zeros * coeff_count]
    const int* __restrict__ idx,            // [num_to_pack_half * non_zeros] (shared across batches)
    const int* __restrict__ idx_bar,        // [num_to_pack_half * non_zeros] (shared)
    int batch_count,
    int num_to_pack_half,
    int non_zeros,
    int coeff_count,
    int poly_len,
    int crt_count,
    const uint64_t* __restrict__ moduli,
    const uint64_t* __restrict__ barrett_crs
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_per_batch = (long long)num_to_pack_half * coeff_count;
    long long total = (long long)batch_count * total_per_batch;
    if (tid >= total) return;

    int batch = (int)(tid / total_per_batch);
    int local_tid = (int)(tid % total_per_batch);
    int i = local_tid / coeff_count;
    int k = local_tid % coeff_count;

    int crt_idx = k / poly_len;
    uint64_t mod_val = moduli[crt_idx];
    uint64_t cr = barrett_crs[crt_idx];

    const int REDUCE_INTERVAL = 128;

    uint64_t acc_r = 0;
    uint64_t acc_r_bar = 0;

    const int* my_idx = idx + (size_t)i * non_zeros;
    const int* my_idx_bar = idx_bar + (size_t)i * non_zeros;
    // Per-batch a_ct offset
    const uint64_t* my_a_ct = a_ct + (size_t)batch * non_zeros * coeff_count;

    for (int j = 0; j < non_zeros; j++) {
        int mono_idx = my_idx[j];
        int mono_idx_bar = my_idx_bar[j];

        uint64_t mono_val = mono_table[(size_t)mono_idx * coeff_count + k];
        uint64_t mono_val_bar = mono_table[(size_t)mono_idx_bar * coeff_count + k];
        uint64_t a_val = my_a_ct[(size_t)j * coeff_count + k];

        acc_r += mono_val * a_val;
        acc_r_bar += mono_val_bar * a_val;

        if ((j + 1) % REDUCE_INTERVAL == 0) {
            acc_r = barrett_reduce(acc_r, mod_val, cr);
            acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);
        }
    }

    acc_r = barrett_reduce(acc_r, mod_val, cr);
    acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);

    size_t out_offset = (size_t)batch * total_per_batch + (size_t)i * coeff_count + k;
    r_out[out_offset] = acc_r;
    r_bar_out[out_offset] = acc_r_bar;
}

// Batched scalar multiply kernel
__global__ void scalar_mul_kernel_batched(
    uint64_t* __restrict__ data,            // [batch_count * num_to_pack_half * coeff_count]
    const uint64_t* __restrict__ mod_inv,   // [coeff_count] (shared)
    long long total_elems,
    int coeff_count,
    int poly_len,
    int crt_count,
    const uint64_t* __restrict__ moduli,
    const uint64_t* __restrict__ barrett_crs
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elems) return;

    int k = (int)(tid % coeff_count);
    int crt_idx = k / poly_len;
    uint64_t mod_val = moduli[crt_idx];

    uint64_t val = data[tid];
    uint64_t inv = mod_inv[k];
    unsigned __int128 prod = (unsigned __int128)val * inv;
    uint64_t result = (uint64_t)(prod % mod_val);
    data[tid] = result;
}

// Batched host entry point: processes all columns in one GPU call
extern "C" void gpu_packing_precomp_batched(
    const uint64_t* h_mono_table,   // [2*poly_len][coeff_count] (shared)
    const uint64_t* h_a_ct_batch,   // [batch_count * non_zeros * coeff_count]
    const int* h_idx,               // [num_to_pack_half * non_zeros] (shared)
    const int* h_idx_bar,           // [num_to_pack_half * non_zeros] (shared)
    const uint64_t* h_mod_inv,      // [coeff_count]
    uint64_t* h_r_out,              // [batch_count * num_to_pack_half * coeff_count]
    uint64_t* h_r_bar_out,          // [batch_count * num_to_pack_half * coeff_count]
    int batch_count,
    int num_to_pack_half,
    int non_zeros,
    int poly_len,
    int crt_count,
    uint64_t mod0,
    uint64_t mod1,
    uint64_t barrett_cr0,
    uint64_t barrett_cr1
) {
    int coeff_count = crt_count * poly_len;
    int total_monos = 2 * poly_len;

    // Shared data sizes (uploaded once)
    size_t mono_size = (size_t)total_monos * coeff_count * sizeof(uint64_t);
    size_t idx_size = (size_t)num_to_pack_half * non_zeros * sizeof(int);
    size_t inv_size = (size_t)coeff_count * sizeof(uint64_t);

    // Batched data sizes
    size_t act_batch_size = (size_t)batch_count * non_zeros * coeff_count * sizeof(uint64_t);
    size_t out_batch_size = (size_t)batch_count * num_to_pack_half * coeff_count * sizeof(uint64_t);

    uint64_t *d_mono_table, *d_a_ct, *d_r_out, *d_r_bar_out, *d_mod_inv;
    int *d_idx, *d_idx_bar;
    uint64_t *d_moduli, *d_barrett_crs;

#define CHECK_CUDA_B(call) CHECK_CUDA_TAG("GPU Packing Batched", call)

    CHECK_CUDA_B(GPU_MALLOC(&d_mono_table, mono_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_a_ct, act_batch_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_r_out, out_batch_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_r_bar_out, out_batch_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_idx, idx_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_idx_bar, idx_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_mod_inv, inv_size));
    CHECK_CUDA_B(GPU_MALLOC(&d_moduli, crt_count * sizeof(uint64_t)));
    CHECK_CUDA_B(GPU_MALLOC(&d_barrett_crs, crt_count * sizeof(uint64_t)));

    // Upload shared data once
    cudaMemcpy(d_mono_table, h_mono_table, mono_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_bar, h_idx_bar, idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod_inv, h_mod_inv, inv_size, cudaMemcpyHostToDevice);

    // Upload batched a_ct (single large transfer)
    cudaMemcpy(d_a_ct, h_a_ct_batch, act_batch_size, cudaMemcpyHostToDevice);

    uint64_t h_moduli[2] = {mod0, mod1};
    uint64_t h_barrett_crs[2] = {barrett_cr0, barrett_cr1};
    cudaMemcpy(d_moduli, h_moduli, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_barrett_crs, h_barrett_crs, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch batched multiply-accumulate kernel
    {
        long long total = (long long)batch_count * num_to_pack_half * coeff_count;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        packing_mul_acc_kernel_batched<<<blocks, threads>>>(
            d_r_out, d_r_bar_out, d_mono_table, d_a_ct,
            d_idx, d_idx_bar,
            batch_count, num_to_pack_half, non_zeros, coeff_count,
            poly_len, crt_count, d_moduli, d_barrett_crs
        );
        GPU_STREAM_SYNC();
    }

    // Launch batched scalar multiply kernel for both r and r_bar
    {
        long long total = (long long)batch_count * num_to_pack_half * coeff_count;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);

        scalar_mul_kernel_batched<<<blocks, threads>>>(
            d_r_out, d_mod_inv, total, coeff_count,
            poly_len, crt_count, d_moduli, d_barrett_crs
        );
        scalar_mul_kernel_batched<<<blocks, threads>>>(
            d_r_bar_out, d_mod_inv, total, coeff_count,
            poly_len, crt_count, d_moduli, d_barrett_crs
        );
        GPU_STREAM_SYNC();
    }

    // Download results (single large transfer)
    cudaMemcpy(h_r_out, d_r_out, out_batch_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r_bar_out, d_r_bar_out, out_batch_size, cudaMemcpyDeviceToHost);

    // Cleanup
    GPU_FREE(d_mono_table);
    GPU_FREE(d_a_ct);
    GPU_FREE(d_r_out);
    GPU_FREE(d_r_bar_out);
    GPU_FREE(d_idx);
    GPU_FREE(d_idx_bar);
    GPU_FREE(d_mod_inv);
    GPU_FREE(d_moduli);
    GPU_FREE(d_barrett_crs);

}

// Accessors for persistent per-column device buffers (used by collapse kernel)
extern "C" uint64_t* gpu_packing_get_s_r_out() { return s_r_out; }
extern "C" uint64_t* gpu_packing_get_s_r_bar_out() { return s_r_bar_out; }
extern "C" int gpu_packing_get_coeff_count() { return s_coeff_count; }
extern "C" int gpu_packing_get_nphalf() { return s_num_to_pack_half; }


// ============================================================================
// Batched fused packing kernel: u32 multiply + fused scalar_mul
// Writes directly to r_all/r_bar_all (no automorphism — done separately)
// ============================================================================

__global__ void batched_packing_u32_fused_kernel(
    uint64_t* __restrict__ r_all,           // [num_cols * nphalf * coeff_count]
    uint64_t* __restrict__ r_bar_all,       // same
    const uint64_t* __restrict__ mono_table,// [2*poly_len * coeff_count] (shared)
    const uint64_t* __restrict__ prepacked, // [num_cols * non_zeros * coeff_count]
    const int* __restrict__ idx,            // [nphalf * non_zeros] (shared)
    const int* __restrict__ idx_bar,        // same
    const uint64_t* __restrict__ mod_inv,   // [coeff_count]
    int num_cols,
    int nphalf,
    int non_zeros,
    int coeff_count,
    int poly_len,
    int crt_count,
    const uint64_t* __restrict__ moduli,
    const uint64_t* __restrict__ barrett_crs
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_per_col = (long long)nphalf * coeff_count;
    long long total = (long long)num_cols * total_per_col;
    if (tid >= total) return;

    int col = (int)(tid / total_per_col);
    int local_tid = (int)(tid % total_per_col);
    int i = local_tid / coeff_count;
    int k = local_tid % coeff_count;

    int crt_idx = k / poly_len;
    uint64_t mod_val = moduli[crt_idx];
    uint64_t cr = barrett_crs[crt_idx];

    // With u32 values (< 2^28), each product < 2^56.
    // 128 products accumulate to < 128 * 2^56 = 2^63, safely within u64.
    const int REDUCE_INTERVAL = 128;

    uint64_t acc_r = 0;
    uint64_t acc_r_bar = 0;

    const int* my_idx = idx + (size_t)i * non_zeros;
    const int* my_idx_bar = idx_bar + (size_t)i * non_zeros;
    const uint64_t* my_a_ct = prepacked + (size_t)col * non_zeros * coeff_count;

    for (int j = 0; j < non_zeros; j++) {
        int mono_idx_r = my_idx[j];
        int mono_idx_rb = my_idx_bar[j];

        // u32 narrowing: NTT-domain values are < q < 2^28
        uint32_t mono_r = (uint32_t)mono_table[(size_t)mono_idx_r * coeff_count + k];
        uint32_t mono_rb = (uint32_t)mono_table[(size_t)mono_idx_rb * coeff_count + k];
        uint32_t a_val = (uint32_t)my_a_ct[(size_t)j * coeff_count + k];

        // u32 x u32 -> u64 multiply-accumulate (mul.wide.u32, ~1 cycle vs ~4 for u64)
        acc_r += (uint64_t)mono_r * (uint64_t)a_val;
        acc_r_bar += (uint64_t)mono_rb * (uint64_t)a_val;

        if ((j + 1) % REDUCE_INTERVAL == 0) {
            acc_r = barrett_reduce(acc_r, mod_val, cr);
            acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);
        }
    }

    // Final Barrett reduction
    acc_r = barrett_reduce(acc_r, mod_val, cr);
    acc_r_bar = barrett_reduce(acc_r_bar, mod_val, cr);

    // Fused scalar multiply: val * mod_inv[k] mod q
    // Both operands < q < 2^28, product < 2^56, Barrett u64 safe
    uint32_t inv = (uint32_t)mod_inv[k];
    acc_r = barrett_reduce((uint64_t)(uint32_t)acc_r * (uint64_t)inv, mod_val, cr);
    acc_r_bar = barrett_reduce((uint64_t)(uint32_t)acc_r_bar * (uint64_t)inv, mod_val, cr);

    // Write to output (no automorphism — done by separate kernel)
    size_t out_idx = (size_t)col * total_per_col + (size_t)i * coeff_count + k;
    r_all[out_idx] = acc_r;
    r_bar_all[out_idx] = acc_r_bar;
}

// Host function: batched packing + scalar_mul for all columns using persistent shared data.
// Outputs to provided r_all/r_bar_all buffers (no automorphism).
extern "C" void gpu_packing_batched_fused(
    uint64_t* d_r_all,
    uint64_t* d_r_bar_all,
    const uint64_t* d_prepacked,
    int num_cols
) {
    if (!s_mono_table || !s_idx || !s_mod_inv) {
        fprintf(stderr, "[GPU packing batched fused] Shared data not set up!\n");
        return;
    }

    long long total = (long long)num_cols * s_num_to_pack_half * s_coeff_count;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    batched_packing_u32_fused_kernel<<<blocks, threads>>>(
        d_r_all, d_r_bar_all,
        s_mono_table, d_prepacked,
        s_idx, s_idx_bar, s_mod_inv,
        num_cols, s_num_to_pack_half, s_non_zeros, s_coeff_count,
        s_poly_len, s_crt_count, s_moduli, s_barrett_crs
    );

    GPU_STREAM_SYNC();
}

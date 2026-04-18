// GPU-accelerated online packing for InspiRING LWE-to-RLWE repacking
//
// Replaces the CPU inner-product loop in full_packing_with_preprocessing_online.
// For each of num_outputs RLWE outputs, computes:
//   sum[z] = Σ_i Σ_k y_all[i*t+k][z] * bold_t[j][i*t+k][z]
//          + Σ_i Σ_k y_bar_all[i*t+k][z] * bold_t_bar[j][i*t+k][z]
//          + Σ_k z_body[k][z] * bold_t_hat[j][k][z]
//
// All polynomials are CRT-condensed: lower 32 bits = mod q0, upper 32 bits = mod q1.
// Output is in NTT domain with CRT limbs separated: [0..poly_len) = mod q0, [poly_len..2*poly_len) = mod q1.

#include "common.h"

#define CHECK_CUDA(call) CHECK_CUDA_TAG("GPU Packing Online", call)

// Main packing online kernel
// Each thread computes one NTT coefficient for one RLWE output
__global__ void packing_online_kernel(
    uint64_t* __restrict__ output,           // [num_outputs * 2 * poly_len]
    const uint64_t* __restrict__ bold_t,     // [num_outputs * n_inner * poly_len]
    const uint64_t* __restrict__ bold_t_bar, // same layout
    const uint64_t* __restrict__ bold_t_hat, // [num_outputs * t_exp_left * poly_len]
    const uint64_t* __restrict__ y_all,      // [n_inner * poly_len]
    const uint64_t* __restrict__ y_bar_all,  // same layout
    const uint64_t* __restrict__ z_body,     // [t_exp_left * poly_len]
    int num_outputs,
    int nphalf_minus_1,   // num_to_pack_half - 1 = 1023
    int t_exp_left,       // 3
    int poly_len,         // 2048
    int addition_capacity,
    uint64_t mod0, uint64_t mod1,
    uint64_t cr0, uint64_t cr1
) {
    int output_id = blockIdx.x;
    int z = blockIdx.y * blockDim.x + threadIdx.x;

    if (output_id >= num_outputs || z >= poly_len) return;

    int n_inner = nphalf_minus_1 * t_exp_left;
    size_t bt_base = (size_t)output_id * n_inner * poly_len;
    size_t bh_base = (size_t)output_id * t_exp_left * poly_len;

    uint64_t sum_lo = 0, sum_hi = 0;
    int num_added = 0;

    for (int i = 0; i < nphalf_minus_1; i++) {
        // y_all × bold_t path
        for (int k = 0; k < t_exp_left; k++) {
            int inner_idx = i * t_exp_left + k;
            uint64_t y_val = y_all[(size_t)inner_idx * poly_len + z];
            uint64_t b_val = bold_t[bt_base + (size_t)inner_idx * poly_len + z];

            sum_lo += (uint64_t)(uint32_t)y_val * (uint32_t)b_val;
            sum_hi += (y_val >> 32) * (b_val >> 32);
        }
        num_added += t_exp_left;

        // y_bar_all × bold_t_bar path
        for (int k = 0; k < t_exp_left; k++) {
            int inner_idx = i * t_exp_left + k;
            uint64_t yb_val = y_bar_all[(size_t)inner_idx * poly_len + z];
            uint64_t bb_val = bold_t_bar[bt_base + (size_t)inner_idx * poly_len + z];

            sum_lo += (uint64_t)(uint32_t)yb_val * (uint32_t)bb_val;
            sum_hi += (yb_val >> 32) * (bb_val >> 32);
        }
        num_added += t_exp_left;

        if (num_added >= addition_capacity || i == nphalf_minus_1 - 1) {
            sum_lo = barrett_u64(sum_lo, mod0, cr0);
            sum_hi = barrett_u64(sum_hi, mod1, cr1);
            num_added = 0;
        }
    }

    // Final: z_body × bold_t_hat
    for (int k = 0; k < t_exp_left; k++) {
        uint64_t z_val = z_body[(size_t)k * poly_len + z];
        uint64_t bh_val = bold_t_hat[bh_base + (size_t)k * poly_len + z];

        sum_lo += (uint64_t)(uint32_t)z_val * (uint32_t)bh_val;
        sum_hi += (z_val >> 32) * (bh_val >> 32);
    }

    sum_lo = barrett_u64(sum_lo, mod0, cr0);
    sum_hi = barrett_u64(sum_hi, mod1, cr1);

    // Store in standard NTT layout: CRT0 at [0..poly_len), CRT1 at [poly_len..2*poly_len)
    size_t out_base = (size_t)output_id * 2 * poly_len;
    output[out_base + z] = sum_lo;
    output[out_base + poly_len + z] = sum_hi;
}


// Device pointers (persistent across queries)
static uint64_t* d_bold_t = nullptr;
static uint64_t* d_bold_t_bar = nullptr;
static uint64_t* d_bold_t_hat = nullptr;
static uint64_t* d_y_all = nullptr;
static uint64_t* d_y_bar_all = nullptr;
static uint64_t* d_z_body = nullptr;
static uint64_t* d_output = nullptr;

// Saved dimensions for key buffer sizing
static int saved_n_inner = 0;
static int saved_t_exp_left = 0;
static int saved_poly_len = 0;

// Host: upload precomp data (called once after offline phase)
extern "C" void gpu_packing_online_upload_precomp(
    const uint64_t* h_bold_t,
    const uint64_t* h_bold_t_bar,
    const uint64_t* h_bold_t_hat,
    int num_outputs,
    int n_inner,
    int t_exp_left,
    int poly_len
) {
    // Free any previous precomp data
    if (d_bold_t) GPU_FREE(d_bold_t);
    if (d_bold_t_bar) GPU_FREE(d_bold_t_bar);
    if (d_bold_t_hat) GPU_FREE(d_bold_t_hat);
    if (d_output) GPU_FREE(d_output);
    d_bold_t = d_bold_t_bar = d_bold_t_hat = d_output = nullptr;

    saved_n_inner = n_inner;
    saved_t_exp_left = t_exp_left;
    saved_poly_len = poly_len;

    size_t bt_size = (size_t)num_outputs * n_inner * poly_len * sizeof(uint64_t);
    size_t bh_size = (size_t)num_outputs * t_exp_left * poly_len * sizeof(uint64_t);
    size_t out_size = (size_t)num_outputs * 2 * poly_len * sizeof(uint64_t);

    CHECK_CUDA(GPU_MALLOC(&d_bold_t, bt_size));
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_bar, bt_size));
    CHECK_CUDA(GPU_MALLOC(&d_bold_t_hat, bh_size));
    CHECK_CUDA(GPU_MALLOC(&d_output, out_size));

    CHECK_CUDA(cudaMemcpy(d_bold_t, h_bold_t, bt_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bold_t_bar, h_bold_t_bar, bt_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bold_t_hat, h_bold_t_hat, bh_size, cudaMemcpyHostToDevice));

}

// Host: upload keys (called per trial after key expansion)
// Uses high-priority query stream.
extern "C" void gpu_packing_online_upload_keys(
    const uint64_t* h_y_all,
    const uint64_t* h_y_bar_all,
    const uint64_t* h_z_body,
    int n_inner,
    int t_exp_left,
    int poly_len
) {
    cudaStream_t s = g_query_stream;
    // Free previous key buffers if dimensions changed; otherwise reuse
    size_t y_size = (size_t)n_inner * poly_len * sizeof(uint64_t);
    size_t z_size = (size_t)t_exp_left * poly_len * sizeof(uint64_t);

    if (!d_y_all || n_inner != saved_n_inner || poly_len != saved_poly_len) {
        if (d_y_all) cudaFreeAsync(d_y_all, s);
        if (d_y_bar_all) cudaFreeAsync(d_y_bar_all, s);
        if (d_z_body) cudaFreeAsync(d_z_body, s);
        CHECK_CUDA(cudaMallocAsync(&d_y_all, y_size, s));
        CHECK_CUDA(cudaMallocAsync(&d_y_bar_all, y_size, s));
        CHECK_CUDA(cudaMallocAsync(&d_z_body, z_size, s));
    }

    CHECK_CUDA(cudaMemcpyAsync(d_y_all, h_y_all, y_size, cudaMemcpyHostToDevice, s));
    CHECK_CUDA(cudaMemcpyAsync(d_y_bar_all, h_y_bar_all, y_size, cudaMemcpyHostToDevice, s));
    CHECK_CUDA(cudaMemcpyAsync(d_z_body, h_z_body, z_size, cudaMemcpyHostToDevice, s));
}

// Ensure key buffers are allocated (for GPU rotation to write into)
// Uses high-priority query stream.
extern "C" void gpu_packing_online_ensure_key_buffers(
    int n_inner,
    int t_exp_left,
    int poly_len
) {
    cudaStream_t s = g_query_stream;
    size_t y_size = (size_t)n_inner * poly_len * sizeof(uint64_t);
    size_t z_size = (size_t)t_exp_left * poly_len * sizeof(uint64_t);

    if (!d_y_all || n_inner != saved_n_inner || poly_len != saved_poly_len) {
        if (d_y_all) cudaFreeAsync(d_y_all, s);
        if (d_y_bar_all) cudaFreeAsync(d_y_bar_all, s);
        if (d_z_body) cudaFreeAsync(d_z_body, s);
        CHECK_CUDA(cudaMallocAsync(&d_y_all, y_size, s));
        CHECK_CUDA(cudaMallocAsync(&d_y_bar_all, y_size, s));
        CHECK_CUDA(cudaMallocAsync(&d_z_body, z_size, s));
        saved_n_inner = n_inner;
        saved_poly_len = poly_len;
    }
}

// Get device pointers for y_all and y_bar_all (for rotation kernel to write into)
extern "C" uint64_t* gpu_packing_online_get_d_y_all() { return d_y_all; }
extern "C" uint64_t* gpu_packing_online_get_d_y_bar_all() { return d_y_bar_all; }

// Upload only z_body (used when rotation kernel handles y_all/y_bar_all)
// Uses high-priority query stream.
extern "C" void gpu_packing_online_upload_z_body(
    const uint64_t* h_z_body,
    int t_exp_left,
    int poly_len
) {
    cudaStream_t s = g_query_stream;
    size_t z_size = (size_t)t_exp_left * poly_len * sizeof(uint64_t);
    CHECK_CUDA(cudaMemcpyAsync(d_z_body, h_z_body, z_size, cudaMemcpyHostToDevice, s));
}

// Host: run the packing kernel (called per query)
// Uses high-priority query stream for preemptive scheduling over fold.
extern "C" void gpu_packing_online_compute(
    uint64_t* h_output,
    int num_outputs,
    int nphalf_minus_1,
    int t_exp_left,
    int poly_len,
    int addition_capacity,
    uint64_t mod0, uint64_t mod1,
    uint64_t cr0, uint64_t cr1
) {
    if (!d_bold_t || !d_y_all) {
        fprintf(stderr, "[GPU Packing Online] Data not uploaded!\n");
        return;
    }

    cudaStream_t s = g_query_stream;

    int threads_per_block = 256;
    dim3 grid(num_outputs, (poly_len + threads_per_block - 1) / threads_per_block);
    dim3 block(threads_per_block);

    packing_online_kernel<<<grid, block, 0, s>>>(
        d_output, d_bold_t, d_bold_t_bar, d_bold_t_hat,
        d_y_all, d_y_bar_all, d_z_body,
        num_outputs, nphalf_minus_1, t_exp_left, poly_len,
        addition_capacity, mod0, mod1, cr0, cr1
    );

    size_t out_size = (size_t)num_outputs * 2 * poly_len * sizeof(uint64_t);
    cudaMemcpyAsync(h_output, d_output, out_size, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Packing Online] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Host: adopt existing device pointers for bold_t (from collapse kernel)
// Skips the 24 GB H2D upload — bold_t data already on GPU.
// IMPORTANT: these pointers are "borrowed" — free is handled by collapse_free_all.
static bool adopted_precomp = false;

extern "C" void gpu_packing_online_adopt_precomp(
    uint64_t* d_bold_t_in,
    uint64_t* d_bold_t_bar_in,
    uint64_t* d_bold_t_hat_in,
    int num_outputs,
    int n_inner,
    int t_exp_left,
    int poly_len
) {
    // Free any previously uploaded (non-adopted) precomp
    if (!adopted_precomp) {
        if (d_bold_t) GPU_FREE(d_bold_t);
        if (d_bold_t_bar) GPU_FREE(d_bold_t_bar);
        if (d_bold_t_hat) GPU_FREE(d_bold_t_hat);
    }

    d_bold_t = d_bold_t_in;
    d_bold_t_bar = d_bold_t_bar_in;
    d_bold_t_hat = d_bold_t_hat_in;
    adopted_precomp = true;

    saved_n_inner = n_inner;
    saved_t_exp_left = t_exp_left;
    saved_poly_len = poly_len;

    // Allocate output buffer
    size_t out_size = (size_t)num_outputs * 2 * poly_len * sizeof(uint64_t);
    if (d_output) GPU_FREE(d_output);
    CHECK_CUDA(GPU_MALLOC(&d_output, out_size));

}

// Host: free all GPU memory
extern "C" void gpu_packing_online_free() {
    // Only free bold_t if we own it (not adopted from collapse kernel)
    if (!adopted_precomp) {
        if (d_bold_t) { GPU_FREE(d_bold_t); }
        if (d_bold_t_bar) { GPU_FREE(d_bold_t_bar); }
        if (d_bold_t_hat) { GPU_FREE(d_bold_t_hat); }
    }
    d_bold_t = nullptr;
    d_bold_t_bar = nullptr;
    d_bold_t_hat = nullptr;
    adopted_precomp = false;
    if (d_y_all) { GPU_FREE(d_y_all); d_y_all = nullptr; }
    if (d_y_bar_all) { GPU_FREE(d_y_bar_all); d_y_bar_all = nullptr; }
    if (d_z_body) { GPU_FREE(d_z_body); d_z_body = nullptr; }
    if (d_output) { GPU_FREE(d_output); d_output = nullptr; }
}

// Swap adopted precomp pointers (called after fold completes + cudaDeviceSynchronize)
// New bold_t pointers come from collapse_kernel.cu's _new buffers
extern "C" void gpu_packing_online_swap_precomp(
    uint64_t* new_bt,
    uint64_t* new_bt_bar,
    uint64_t* new_bt_hat,
    int num_outputs,
    int n_inner,
    int t_exp_left,
    int poly_len
) {
    // Old bold_t is freed by collapse_swap_bold_t — we just update pointers
    d_bold_t = new_bt;
    d_bold_t_bar = new_bt_bar;
    d_bold_t_hat = new_bt_hat;
    adopted_precomp = true;  // still borrowed from collapse

    saved_n_inner = n_inner;
    saved_t_exp_left = t_exp_left;
    saved_poly_len = poly_len;

    // Reallocate output buffer if num_outputs changed
    size_t out_size = (size_t)num_outputs * 2 * poly_len * sizeof(uint64_t);
    if (d_output) GPU_FREE(d_output);
    CHECK_CUDA(GPU_MALLOC(&d_output, out_size));
}

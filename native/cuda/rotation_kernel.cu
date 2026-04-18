// GPU-accelerated Galois automorphism rotations for InspiRING packing keys
//
// Replaces CPU generate_rotations_double: applies 1023 automorphisms to
// y_body_condensed to produce y_all_condensed and y_bar_all_condensed.
//
// Each automorphism is a permutation: out[z] = in[table[z]]
// For the "double" case: rotation i uses table[gen_pows[i]/2] for y_all
// and table[(2*poly_len - gen_pows[i] - 1)/2] for y_bar_all.
//
// Input: y_body_condensed (t_exp_left=3 condensed polys, each poly_len u64)
// Output: directly into d_y_all and d_y_bar_all (already allocated by packing kernel)

#include "common.h"

#define CHECK_CUDA(call) CHECK_CUDA_TAG("GPU Rotation", call)

// Rotation kernel: each thread handles two coefficient positions
// for one rotation i and one polynomial k.
// Grid: (num_rotations, t_exp_left), Block: (poly_len/2)
// Each thread processes z and z + poly_len/2.
__global__ void rotation_double_kernel(
    uint64_t* __restrict__ y_all,          // [num_rotations * t_exp_left * poly_len]
    uint64_t* __restrict__ y_bar_all,      // same layout
    const uint64_t* __restrict__ y_body,   // [t_exp_left * poly_len]
    const uint32_t* __restrict__ tables,   // [num_tables * poly_len]
    const uint32_t* __restrict__ gen_pows, // [num_rotations]
    int num_rotations,
    int t_exp_left,
    int poly_len
) {
    int rot_idx = blockIdx.x;           // which rotation (0..1022)
    int k = blockIdx.y;                 // which polynomial (0..t_exp_left-1)
    int half = poly_len / 2;
    int z0 = threadIdx.x;               // first element
    int z1 = z0 + half;                 // second element

    if (rot_idx >= num_rotations || k >= t_exp_left || z0 >= half) return;

    uint32_t t = gen_pows[rot_idx];
    uint32_t table_idx_1 = (t - 1) / 2;
    uint32_t table_idx_2 = (2 * poly_len - t - 1) / 2;

    size_t t1_base = (size_t)table_idx_1 * poly_len;
    size_t t2_base = (size_t)table_idx_2 * poly_len;
    size_t body_base = (size_t)k * poly_len;
    size_t out_base = ((size_t)rot_idx * t_exp_left + k) * poly_len;

    // Process z0
    uint32_t src1_z0 = tables[t1_base + z0];
    uint32_t src2_z0 = tables[t2_base + z0];
    y_all[out_base + z0] = y_body[body_base + src1_z0];
    y_bar_all[out_base + z0] = y_body[body_base + src2_z0];

    // Process z1
    uint32_t src1_z1 = tables[t1_base + z1];
    uint32_t src2_z1 = tables[t2_base + z1];
    y_all[out_base + z1] = y_body[body_base + src1_z1];
    y_bar_all[out_base + z1] = y_body[body_base + src2_z1];
}

// Device pointers
static uint32_t* d_tables = nullptr;
static uint32_t* d_gen_pows = nullptr;
static uint64_t* d_y_body = nullptr;

// Saved dimensions
static int saved_num_tables = 0;
static int saved_poly_len_rot = 0;

// Upload automorphism tables (called once at setup, ~16 MB)
extern "C" void gpu_rotation_upload_tables(
    const uint32_t* h_tables,
    const uint32_t* h_gen_pows,
    int num_tables,
    int num_rotations,
    int poly_len
) {
    if (d_tables) GPU_FREE(d_tables);
    if (d_gen_pows) GPU_FREE(d_gen_pows);
    d_tables = nullptr;
    d_gen_pows = nullptr;

    saved_num_tables = num_tables;
    saved_poly_len_rot = poly_len;

    size_t tables_size = (size_t)num_tables * poly_len * sizeof(uint32_t);
    size_t gen_pows_size = (size_t)num_rotations * sizeof(uint32_t);

    CHECK_CUDA(GPU_MALLOC(&d_tables, tables_size));
    CHECK_CUDA(GPU_MALLOC(&d_gen_pows, gen_pows_size));
    CHECK_CUDA(cudaMemcpy(d_tables, h_tables, tables_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gen_pows, h_gen_pows, gen_pows_size, cudaMemcpyHostToDevice));

}

// Run rotation kernel, writing directly into d_y_all and d_y_bar_all
// (the packing kernel's key buffers)
// Uses high-priority query stream.
extern "C" void gpu_rotation_expand_keys(
    uint64_t* d_y_all_out,       // device pointer (from packing kernel)
    uint64_t* d_y_bar_all_out,   // device pointer (from packing kernel)
    const uint64_t* h_y_body,    // host: y_body_condensed [t_exp_left * poly_len]
    int num_rotations,
    int t_exp_left,
    int poly_len
) {
    if (!d_tables || !d_gen_pows) {
        fprintf(stderr, "[GPU Rotation] Tables not uploaded!\n");
        return;
    }

    cudaStream_t s = g_query_stream;

    // Upload y_body (~48 KB)
    size_t body_size = (size_t)t_exp_left * poly_len * sizeof(uint64_t);
    if (!d_y_body) {
        CHECK_CUDA(cudaMallocAsync(&d_y_body, body_size, s));
    }
    CHECK_CUDA(cudaMemcpyAsync(d_y_body, h_y_body, body_size, cudaMemcpyHostToDevice, s));

    // Launch kernel: each thread handles 2 coefficients
    // Grid: (num_rotations=1023, t_exp_left=3), Block: (poly_len/2=1024)
    dim3 grid(num_rotations, t_exp_left);
    dim3 block(poly_len / 2);
    rotation_double_kernel<<<grid, block, 0, s>>>(
        d_y_all_out, d_y_bar_all_out, d_y_body,
        d_tables, d_gen_pows,
        num_rotations, t_exp_left, poly_len
    );

    cudaStreamSynchronize(s);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Rotation] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Free rotation-specific GPU memory
extern "C" void gpu_rotation_free() {
    if (d_tables) { GPU_FREE(d_tables); d_tables = nullptr; }
    if (d_gen_pows) { GPU_FREE(d_gen_pows); d_gen_pows = nullptr; }
    if (d_y_body) { GPU_FREE(d_y_body); d_y_body = nullptr; }
}

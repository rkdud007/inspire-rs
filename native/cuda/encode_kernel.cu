// GPU-accelerated DB encoding kernel for InsPIRe PIR
// Implements batch cooley_tukey_without_ntt over all db_rows
//
// The encoding transforms each column of the database using polynomial
// interpolation via Cooley-Tukey FFT over the ring Z_p[X]/(X^N+1).
// Since p=65535 is not NTT-friendly, we use the non-NTT variant where
// twiddle factors are monomial multiplications (coefficient rotations).
//
// Work buffers use u16 (values are mod 65535 < 2^16) to halve memory.
// Batching is over column groups (wp values) for correct tiled kernel indexing.

#include "common.h"

#define CHECK_CUDA(call) CHECK_CUDA_TAG("GPU Encode", call)

// Butterfly kernel with u16 work buffers.
// Arithmetic done in u32 registers, results stored as u16.
__global__ void butterfly_kernel(
    uint16_t* __restrict__ work,          // [batch_wu, interp_degree, gamma]
    const uint16_t* __restrict__ work_in, // same shape, read buffer (ping-pong)
    int batch_wu,
    int interp_degree,
    int gamma,                          // = poly_len = 2048
    int level,                          // butterfly level (0 = bottom, smallest butterflies)
    const int* __restrict__ exponents,  // precomputed twiddle exponents for this level
    uint32_t modulus
) {
    int wu = blockIdx.x;
    if (wu >= batch_wu) return;

    int group_size = 1 << (level + 1);
    int half = 1 << level;
    int num_groups = interp_degree / group_size;

    uint16_t* base_out = work + (size_t)wu * interp_degree * gamma;
    const uint16_t* base_in = work_in + (size_t)wu * interp_degree * gamma;

    for (int g = 0; g < num_groups; g++) {
        for (int e = 0; e < half; e++) {
            int even_idx = g * group_size + e;
            int odd_idx = even_idx + half;
            int exponent = exponents[e];

            const uint16_t* even_poly = base_in + (size_t)even_idx * gamma;
            const uint16_t* odd_poly = base_in + (size_t)odd_idx * gamma;
            uint16_t* out_even = base_out + (size_t)even_idx * gamma;
            uint16_t* out_odd = base_out + (size_t)odd_idx * gamma;

            int two_n = 2 * gamma;
            int mask = two_n - 1;  // two_n is power of 2
            for (int k = threadIdx.x; k < gamma; k += blockDim.x) {
                uint32_t twiddle_val;
                int src = (k - exponent + two_n) & mask;
                if (src < gamma) {
                    twiddle_val = (uint32_t)odd_poly[src];
                } else {
                    twiddle_val = modulus - (uint32_t)odd_poly[src - gamma];
                }

                uint32_t ev = (uint32_t)even_poly[k];

                // Mersenne reduction: modulus = 65535 = 2^16 - 1
                // For x < 2*65535 = 131070 < 2^17, (x & 0xFFFF) + (x >> 16) works
                uint32_t sum_add = ev + twiddle_val;
                uint32_t r_add = (sum_add & 0xFFFFu) + (sum_add >> 16);
                if (r_add >= 65535u) r_add -= 65535u;
                out_even[k] = (uint16_t)r_add;

                uint32_t sum_sub = ev + modulus - twiddle_val;
                uint32_t r_sub = (sum_sub & 0xFFFFu) + (sum_sub >> 16);
                if (r_sub >= 65535u) r_sub -= 65535u;
                out_odd[k] = (uint16_t)r_sub;
            }
            __syncthreads();
        }
    }
}


// Tiled scatter with shared-memory transpose for coalesced writes.
// Uses 32×32 tiles: reads from work buffer (coalesced by k),
// writes to column-major output (coalesced by row).
#define TILE_DIM 32

__global__ void scale_scatter_tiled_kernel(
    uint16_t* __restrict__ out_db,     // [db_cols * db_rows] column-major
    const uint16_t* __restrict__ work, // [batch_wu, interp_degree, gamma]
    int db_rows,
    int db_cols,
    int interp_degree,
    int gamma,
    int c,
    uint32_t num_inv,
    int col_offset,   // first DB column in this batch
    int batch_wp      // number of wp groups in this batch
) {
    __shared__ uint16_t tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids bank conflicts

    int interp_gamma = interp_degree * gamma;
    int batch_cols = batch_wp * interp_gamma;
    int tiles_per_col_dim = batch_cols / TILE_DIM;
    int tile_row = blockIdx.x / tiles_per_col_dim;
    int tile_col = blockIdx.x % tiles_per_col_dim;

    int row_start = tile_row * TILE_DIM;
    int db_col_start = col_offset + tile_col * TILE_DIM;

    // All 32 columns in this tile share the same (wp, j_prime) since TILE_DIM | gamma
    int wp = db_col_start / interp_gamma;
    int wp_offset = col_offset / interp_gamma;
    int local_wp = wp - wp_offset;
    int inner = db_col_start % interp_gamma;
    int j_prime = inner / gamma;
    int k_start = inner % gamma;

    // Load: read from work buffer (coalesced by k within each warp)
    // Scale by num_inv and store to shared memory
    for (int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += blockDim.x) {
        int r_local = idx / TILE_DIM;
        int c_local = idx % TILE_DIM;

        int row = row_start + r_local;
        int local_wu = row * batch_wp + local_wp;
        size_t work_idx = (size_t)local_wu * interp_gamma + (size_t)j_prime * gamma + (k_start + c_local);
        uint32_t val = (uint32_t)work[work_idx];

        // Mersenne reduction: scale by num_inv
        uint32_t product = val * num_inv;
        uint32_t result = (product & 0xFFFFu) + (product >> 16);
        result = (result & 0xFFFFu) + (result >> 16);
        if (result >= 65535u) result -= 65535u;

        tile[r_local][c_local] = (uint16_t)result;
    }
    __syncthreads();

    // Write: output to column-major DB (coalesced by row within each warp)
    for (int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += blockDim.x) {
        int c_local = idx / TILE_DIM;
        int r_local = idx % TILE_DIM;
        size_t out_idx = (size_t)(db_col_start + c_local) * db_rows + (row_start + r_local);
        out_db[out_idx] = tile[r_local][c_local];
    }
}


// Tiled gather with shared-memory transpose for coalesced reads from column-major DB.
// col_offset and batch_wp enable column-group batching: the kernel only processes
// columns [col_offset, col_offset + batch_wp * interp_degree * gamma).
__global__ void gather_tiled_kernel(
    uint16_t* __restrict__ work,       // [batch_wu, interp_degree, gamma]
    const uint16_t* __restrict__ db,   // [db_cols * db_rows] column-major
    const int* __restrict__ bitrev,
    int db_rows,
    int db_cols,
    int interp_degree,
    int gamma,
    int c,
    int col_offset,   // first DB column in this batch
    int batch_wp      // number of wp groups in this batch
) {
    __shared__ uint16_t tile[TILE_DIM][TILE_DIM + 1];

    int interp_gamma = interp_degree * gamma;
    int batch_cols = batch_wp * interp_gamma;
    int tiles_per_col_dim = batch_cols / TILE_DIM;
    int tile_row = blockIdx.x / tiles_per_col_dim;
    int tile_col = blockIdx.x % tiles_per_col_dim;

    int row_start = tile_row * TILE_DIM;
    int db_col_start = col_offset + tile_col * TILE_DIM;

    int wp = db_col_start / interp_gamma;
    int wp_offset = col_offset / interp_gamma;
    int local_wp = wp - wp_offset;
    int inner = db_col_start % interp_gamma;
    int j_prime = inner / gamma;
    int k_start = inner % gamma;
    int dest_j_prime = bitrev[j_prime];

    // Load from column-major DB: coalesced by row within each warp
    for (int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += blockDim.x) {
        int c_local = idx / TILE_DIM;
        int r_local = idx % TILE_DIM;
        size_t db_idx = (size_t)(db_col_start + c_local) * db_rows + (row_start + r_local);
        tile[r_local][c_local] = db[db_idx];
    }
    __syncthreads();

    // Write to work buffer: coalesced by k within each warp
    for (int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += blockDim.x) {
        int r_local = idx / TILE_DIM;
        int c_local = idx % TILE_DIM;
        int row = row_start + r_local;
        int local_wu = row * batch_wp + local_wp;
        size_t work_idx = (size_t)local_wu * interp_gamma + (size_t)dest_j_prime * gamma + (k_start + c_local);
        work[work_idx] = tile[r_local][c_local];
    }
}


// Host-side helper: compute bit-reversal permutation
void compute_bitrev(int* bitrev, int n) {
    int log_n = 0;
    int temp = n;
    while (temp > 1) { temp >>= 1; log_n++; }

    for (int i = 0; i < n; i++) {
        int rev = 0;
        int val = i;
        for (int b = 0; b < log_n; b++) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        bitrev[i] = rev;
    }
}

// Host-side helper: compute twiddle exponents for a given level
void compute_twiddle_exponents(int* exponents, int level, int gamma) {
    int group_size = 1 << (level + 1);
    int half = 1 << level;
    int two_n = 2 * gamma;

    for (int e = 0; e < half; e++) {
        exponents[e] = (two_n - two_n * e / group_size) % two_n;
    }
}


// Initialize CUDA memory pool: set release threshold to max so freed memory
// stays in the pool for reuse (no device sync from pool shrinking).
extern "C" void gpu_trim_memory_pool() {
    cudaStreamSynchronize(cudaStreamPerThread);
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    cudaMemPoolTrimTo(pool, 0);
}

extern "C" void gpu_init_memory_pool() {
    cudaMemPool_t pool;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU] Warning: could not get default memory pool: %s\n",
                cudaGetErrorString(err));
        return;
    }
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    // Create high-priority stream for query path once.
    if (g_query_stream == nullptr) {
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        cudaStreamCreateWithPriority(&g_query_stream, cudaStreamNonBlocking, greatest_priority);
    }
}

extern "C" uint64_t gpu_available_memory_bytes() {
    return static_cast<uint64_t>(gpu_available_memory());
}

// Main entry point called from Rust
// Uses u16 work buffers to halve memory vs u32, enabling single-batch at 32 GB.
extern "C" void gpu_encode_db(
    uint16_t* d_db_in,      // device pointer to input DB (u16, column-major)
    uint16_t* d_db_out,     // device pointer to output DB (u16, column-major)
    int db_rows,
    int db_cols,
    int interp_degree,
    int gamma,              // = poly_len
    int c,                  // = db_cols_prime / interp_degree
    uint32_t modulus,
    uint32_t num_inv
) {
    int total_work_units = db_rows * c;
    int log_interp = 0;
    { int temp = interp_degree; while (temp > 1) { temp >>= 1; log_interp++; } }

    // Per work unit: interp_degree * gamma * sizeof(uint16_t) bytes
    size_t per_wu_bytes = (size_t)interp_degree * gamma * sizeof(uint16_t);
    size_t total_work_size = (size_t)total_work_units * per_wu_bytes;

    // Query available GPU memory (pool-aware) to determine batch size
    size_t free_mem = gpu_available_memory();

    // Reserve some memory for other allocations
    size_t reserved = 512ULL * 1024 * 1024; // 512 MB
    size_t available = (free_mem > reserved) ? (free_mem - reserved) : 0;

    // Batch over column groups (wp values) so tiled kernels only touch batch's columns.
    // Each wp group has db_rows work units; each batch needs 2 work buffers.
    size_t per_wp_group_bytes = (size_t)db_rows * per_wu_bytes;
    int max_batch_wp = (int)(available / (2 * per_wp_group_bytes));
    if (max_batch_wp > c) max_batch_wp = c;
    if (max_batch_wp <= 0) {
        fprintf(stderr, "[GPU encode] ERROR: not enough GPU memory for even 1 wp group!\n");
        return;
    }

    int num_batches = (c + max_batch_wp - 1) / max_batch_wp;
    int batch_wu = max_batch_wp * db_rows;

    // Batching: num_batches batches of max_batch_wp wp groups

    // Compute bit-reversal permutation on host and copy to device
    int* h_bitrev = new int[interp_degree];
    compute_bitrev(h_bitrev, interp_degree);
    int* d_bitrev = nullptr;
    CHECK_CUDA(GPU_MALLOC(&d_bitrev, interp_degree * sizeof(int)));
    cudaMemcpy(d_bitrev, h_bitrev, interp_degree * sizeof(int), cudaMemcpyHostToDevice);

    // Precompute all twiddle exponents on device
    int** d_exponents_all = new int*[log_interp];
    for (int level = 0; level < log_interp; level++) {
        int half = 1 << level;
        int* h_exponents = new int[half];
        compute_twiddle_exponents(h_exponents, level, gamma);
        CHECK_CUDA(GPU_MALLOC(&d_exponents_all[level], half * sizeof(int)));
        cudaMemcpy(d_exponents_all[level], h_exponents, half * sizeof(int), cudaMemcpyHostToDevice);
        delete[] h_exponents;
    }

    // Allocate u16 work buffers
    size_t batch_work_size = (size_t)batch_wu * per_wu_bytes;
    uint16_t* d_work_a = nullptr;
    uint16_t* d_work_b = nullptr;
    CHECK_CUDA(GPU_MALLOC(&d_work_a, batch_work_size));
    CHECK_CUDA(GPU_MALLOC(&d_work_b, batch_work_size));

    // Process batches over column groups (wp values)
    for (int batch = 0; batch < num_batches; batch++) {
        int wp_offset = batch * max_batch_wp;
        int this_batch_wp = max_batch_wp;
        if (wp_offset + this_batch_wp > c) {
            this_batch_wp = c - wp_offset;
        }
        int this_batch_wu = this_batch_wp * db_rows;
        int col_offset = wp_offset * interp_degree * gamma;
        int batch_cols = this_batch_wp * interp_degree * gamma;

        int threads = 256;
        int tile_rows = db_rows / TILE_DIM;
        int tile_cols = batch_cols / TILE_DIM;
        int total_tiles = tile_rows * tile_cols;

        // Step 1: Tiled gather with bit-reversal (coalesced reads from column-major DB)
        gather_tiled_kernel<<<total_tiles, threads>>>(
            d_work_a, d_db_in, d_bitrev,
            db_rows, db_cols, interp_degree, gamma, c,
            col_offset, this_batch_wp
        );

        // Step 2: Butterfly levels (ping-pong between u16 buffers)
        uint16_t* src = d_work_a;
        uint16_t* dst = d_work_b;

        for (int level = 0; level < log_interp; level++) {
            butterfly_kernel<<<this_batch_wu, threads>>>(
                dst, src, this_batch_wu, interp_degree, gamma,
                level, d_exponents_all[level], modulus
            );

            uint16_t* tmp = src;
            src = dst;
            dst = tmp;
        }

        // Step 3: Tiled scatter with coalesced writes to column-major output
        scale_scatter_tiled_kernel<<<total_tiles, threads>>>(
            d_db_out, src,
            db_rows, db_cols, interp_degree, gamma, c, num_inv,
            col_offset, this_batch_wp
        );
    }
    CHECK_CUDA(GPU_STREAM_SYNC());

    // Cleanup
    GPU_FREE(d_work_a);
    GPU_FREE(d_work_b);
    GPU_FREE(d_bitrev);
    for (int level = 0; level < log_interp; level++) {
        GPU_FREE(d_exponents_all[level]);
    }
    delete[] d_exponents_all;
    delete[] h_bitrev;
}


// High-priority query stream (declared extern in common.h)
cudaStream_t g_query_stream = nullptr;

// Persistent encoded DB pointer (kept on GPU for hint computation)
static uint16_t* d_encoded_db = nullptr;
static size_t d_encoded_db_size = 0;

// Secondary encoded DB for fold (new data being built while old serves queries)
static uint16_t* d_encoded_db_new = nullptr;
static size_t d_encoded_db_new_size = 0;

// Convenience wrapper that handles host<->device transfers
// Keeps the encoded DB on GPU for subsequent hint computation
extern "C" void gpu_encode_db_from_host(
    const uint16_t* h_db_in,   // host pointer to input DB
    uint16_t* h_db_out,        // host pointer to output encoded DB
    int db_rows,
    int db_cols,
    int interp_degree,
    int gamma,
    int c,
    uint32_t modulus,
    uint32_t num_inv
) {
    size_t db_size = (size_t)db_rows * db_cols * sizeof(uint16_t);

    uint16_t* d_db_in = nullptr;
    uint16_t* d_db_out = nullptr;
    CHECK_CUDA(GPU_MALLOC(&d_db_in, db_size));
    CHECK_CUDA(GPU_MALLOC(&d_db_out, db_size));

    cudaMemcpy(d_db_in, h_db_in, db_size, cudaMemcpyHostToDevice);

    gpu_encode_db(d_db_in, d_db_out, db_rows, db_cols,
                  interp_degree, gamma, c, modulus, num_inv);

    // Encoded DB stays GPU-only — no D2H copy needed.
    // All consumers (hint, GEMV, prep_pack, InspiRING precomp) use the device pointer.

    GPU_FREE(d_db_in);

    // Keep d_db_out on GPU for hint computation
    if (d_encoded_db && d_encoded_db != d_db_out) GPU_FREE(d_encoded_db);
    d_encoded_db = d_db_out;
    d_encoded_db_size = db_size;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Encode] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Get the device pointer to the encoded DB (for hint kernel)
extern "C" const uint16_t* gpu_encode_get_device_db() {
    return d_encoded_db;
}

// Free the persistent encoded DB
extern "C" void gpu_encode_free_device_db() {
    if (d_encoded_db) { GPU_FREE(d_encoded_db); d_encoded_db = nullptr; }
    d_encoded_db_size = 0;
}

// Fold: encode a new DB into d_encoded_db_new without touching d_encoded_db
extern "C" void gpu_encode_db_from_host_fold(
    const uint16_t* h_db_in,
    int db_rows,
    int db_cols,
    int interp_degree,
    int gamma,
    int c,
    uint32_t modulus,
    uint32_t num_inv
) {
    size_t db_size = (size_t)db_rows * db_cols * sizeof(uint16_t);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    (void)free_mem; (void)total_mem;

    uint16_t* d_db_in = nullptr;
    uint16_t* d_db_out = nullptr;
    CHECK_CUDA(GPU_MALLOC(&d_db_in, db_size));
    CHECK_CUDA(GPU_MALLOC(&d_db_out, db_size));

    // Chunked H2D: break large copy into 256MB pieces so query-stream
    // copies can interleave between chunks (reduces query H2D stall).
    {
        const size_t chunk = 256ULL * 1024 * 1024; // 256 MB
        for (size_t off = 0; off < db_size; off += chunk) {
            size_t n = (off + chunk <= db_size) ? chunk : (db_size - off);
            cudaMemcpyAsync((uint8_t*)d_db_in + off, (const uint8_t*)h_db_in + off,
                            n, cudaMemcpyHostToDevice, cudaStreamPerThread);
        }
        cudaStreamSynchronize(cudaStreamPerThread);
    }

    gpu_encode_db(d_db_in, d_db_out, db_rows, db_cols,
                  interp_degree, gamma, c, modulus, num_inv);

    GPU_FREE(d_db_in);

    // Store in _new slot (old d_encoded_db untouched)
    if (d_encoded_db_new && d_encoded_db_new != d_db_out) GPU_FREE(d_encoded_db_new);
    d_encoded_db_new = d_db_out;
    d_encoded_db_new_size = db_size;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Encode Fold] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Get device pointer to the NEW encoded DB (for fold hint computation)
extern "C" const uint16_t* gpu_encode_get_device_db_fold() {
    return d_encoded_db_new;
}

// Swap: free old primary, promote new to primary
extern "C" void gpu_encode_swap_primary() {
    if (d_encoded_db && d_encoded_db != d_encoded_db_new) {
        GPU_FREE(d_encoded_db);
    }
    d_encoded_db = d_encoded_db_new;
    d_encoded_db_size = d_encoded_db_new_size;
    d_encoded_db_new = nullptr;
    d_encoded_db_new_size = 0;
}

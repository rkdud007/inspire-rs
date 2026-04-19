// Rust FFI wrapper for GPU-accelerated DB encoding

unsafe extern "C" {
    fn gpu_init_memory_pool();
    fn gpu_trim_memory_pool();
    fn gpu_available_memory_bytes() -> u64;
    fn gpu_encode_db_from_host(
        h_db_in: *const u16,
        h_db_out: *mut u16,
        db_rows: i32,
        db_cols: i32,
        interp_degree: i32,
        gamma: i32,
        c: i32,
        modulus: u32,
        num_inv: u32,
    );
    fn gpu_encode_get_device_db() -> *const u16;
    fn gpu_encode_free_device_db();
    fn gpu_encode_db_from_host_fold(
        h_db_in: *const u16,
        db_rows: i32,
        db_cols: i32,
        interp_degree: i32,
        gamma: i32,
        c: i32,
        modulus: u32,
        num_inv: u32,
    );
    fn gpu_encode_get_device_db_fold() -> *const u16;
    fn gpu_encode_swap_primary();
}

/// GPU-accelerated DB encoding via Cooley-Tukey interpolation.
///
/// Takes the raw database (u16, column-major) and produces the encoded
/// database with polynomial interpolation coefficients.
///
/// # Safety
/// db_in must have length db_rows * db_cols.
/// db_out must have length db_rows * db_cols.
pub fn gpu_encode_database(
    db_in: &[u16],
    db_out: &mut [u16],
    db_rows: usize,
    db_cols: usize,
    interp_degree: usize,
    gamma: usize,
    c: usize,
    modulus: u64,
    num_inv: u64,
) {
    assert_eq!(db_in.len(), db_rows * db_cols);
    assert_eq!(db_out.len(), db_rows * db_cols);
    assert!(interp_degree.is_power_of_two());
    assert!(modulus <= u32::MAX as u64);
    assert!(num_inv <= u32::MAX as u64);

    unsafe {
        gpu_encode_db_from_host(
            db_in.as_ptr(),
            db_out.as_mut_ptr(),
            db_rows as i32,
            db_cols as i32,
            interp_degree as i32,
            gamma as i32,
            c as i32,
            modulus as u32,
            num_inv as u32,
        );
    }
}

/// Get the device pointer to the encoded DB kept on GPU after encoding.
/// Returns null if no encoded DB is resident.
pub fn get_device_db() -> *const u16 {
    unsafe { gpu_encode_get_device_db() }
}

/// Free the encoded DB from GPU memory.
pub fn free_device_db() {
    unsafe {
        gpu_encode_free_device_db();
    }
}

/// GPU-accelerated DB encoding for fold — writes to secondary buffer.
pub fn gpu_encode_database_fold(
    db_in: &[u16],
    db_rows: usize,
    db_cols: usize,
    interp_degree: usize,
    gamma: usize,
    c: usize,
    modulus: u64,
    num_inv: u64,
) {
    assert_eq!(db_in.len(), db_rows * db_cols);
    assert!(interp_degree.is_power_of_two());
    unsafe {
        gpu_encode_db_from_host_fold(
            db_in.as_ptr(),
            db_rows as i32,
            db_cols as i32,
            interp_degree as i32,
            gamma as i32,
            c as i32,
            modulus as u32,
            num_inv as u32,
        );
    }
}

/// Get the device pointer to the NEW encoded DB (fold secondary buffer).
pub fn get_device_db_fold() -> *const u16 {
    unsafe { gpu_encode_get_device_db_fold() }
}

/// Swap: free old primary encoded DB, promote new to primary.
pub fn swap_encoded_db() {
    unsafe {
        gpu_encode_swap_primary();
    }
}

/// Initialize CUDA memory pool with max release threshold.
/// Must be called once at startup before any GPU allocation.
/// Prevents cudaFreeAsync from returning memory to OS (which causes device sync).
pub fn init_memory_pool() {
    unsafe {
        gpu_init_memory_pool();
    }
}

/// Trim memory pool to release freed GPU memory back to CUDA.
/// Call after large temporary allocations are freed (e.g., fold encode).
pub fn trim_memory_pool() {
    unsafe {
        gpu_trim_memory_pool();
    }
}

/// Return pool-aware available GPU memory in bytes.
pub fn available_memory_bytes() -> u64 {
    unsafe { gpu_available_memory_bytes() }
}

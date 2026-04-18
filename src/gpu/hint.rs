// Rust FFI wrapper for GPU-accelerated hint computation

use crate::params::Params;
use crate::poly::{PolyMatrix, PolyMatrixNTT};

unsafe extern "C" {
    fn gpu_hint_upload_tables(h_ntt_tables: *const u64, poly_len: i32);

    fn gpu_hint_upload_query_ffi(h_query: *const u64, db_rows_poly: i32, poly_len: i32);

    fn gpu_hint_compute(
        d_db: *const u16,
        h_output: *mut u64,
        db_rows: i32,
        db_rows_poly: i32,
        db_cols: i32,
        mod0: u32,
        mod1: u32,
        barrett_cr1_0: u64,
        barrett_cr1_1: u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        modulus: u64,
        barrett_cr0_mod: u64,
        barrett_cr1_mod: u64,
    );

    fn gpu_hint_compute_from_host(
        h_db: *const u16,
        h_output: *mut u64,
        db_rows: i32,
        db_rows_poly: i32,
        db_cols: i32,
        mod0: u32,
        mod1: u32,
        barrett_cr1_0: u64,
        barrett_cr1_1: u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        modulus: u64,
        barrett_cr0_mod: u64,
        barrett_cr1_mod: u64,
    );

    fn gpu_hint_free();
    fn gpu_hint_free_query_only();
    fn gpu_hint_free_output_and_tables();

    fn gpu_hint_get_device_output() -> *const u64;
    fn gpu_hint_get_ntt_tables() -> *const u64;
    fn gpu_hint_get_db_cols() -> i32;

    fn gpu_hint_compute_no_d2h(
        d_db: *const u16,
        db_rows: i32,
        db_rows_poly: i32,
        db_cols: i32,
        mod0: u32,
        mod1: u32,
        barrett_cr1_0: u64,
        barrett_cr1_1: u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        modulus: u64,
        barrett_cr0_mod: u64,
        barrett_cr1_mod: u64,
    );

    // From encode_kernel.cu: get device pointer to encoded DB
    fn gpu_encode_get_device_db() -> *const u16;
}

/// Upload NTT tables to GPU. Called once at setup.
pub fn gpu_hint_setup_tables(params: &Params) {
    let poly_len = params.poly_len;

    let mut flat_tables = Vec::with_capacity(2 * 4 * poly_len);
    for crt in 0..2 {
        for table_id in 0..4 {
            let table = &params.ntt_tables[crt][table_id];
            flat_tables.extend_from_slice(&table[..poly_len]);
        }
    }
    assert_eq!(flat_tables.len(), 2 * 4 * poly_len);

    unsafe {
        gpu_hint_upload_tables(flat_tables.as_ptr(), poly_len as i32);
    }
}

/// Upload query polynomials to GPU.
pub fn gpu_hint_setup_query(query: &[PolyMatrixNTT], params: &Params) {
    let poly_len = params.poly_len;
    let db_rows_poly = query.len();
    let crt_count = params.crt_count;

    let mut flat_query = Vec::with_capacity(db_rows_poly * crt_count * poly_len);
    for q in query {
        let data = q.get_poly(0, 0);
        flat_query.extend_from_slice(&data[..crt_count * poly_len]);
    }
    assert_eq!(flat_query.len(), db_rows_poly * crt_count * poly_len);

    unsafe {
        gpu_hint_upload_query_ffi(flat_query.as_ptr(), db_rows_poly as i32, poly_len as i32);
    }
}

/// Run GPU hint computation using device DB pointer from encode.
/// Falls back to host upload if device DB not available.
pub fn gpu_hint_compute_rs(
    db_u16: &[u16],
    db_rows: usize,
    db_cols: usize,
    db_rows_poly: usize,
    params: &Params,
) -> Vec<u64> {
    let poly_len = params.poly_len;
    let mut output = vec![0u64; db_cols * poly_len];

    let d_db = unsafe { gpu_encode_get_device_db() };

    if !d_db.is_null() {
        // Using encoded DB already on GPU (no upload needed)
        unsafe {
            gpu_hint_compute(
                d_db,
                output.as_mut_ptr(),
                db_rows as i32,
                db_rows_poly as i32,
                db_cols as i32,
                params.moduli[0] as u32,
                params.moduli[1] as u32,
                params.barrett_cr_1[0],
                params.barrett_cr_1[1],
                params.mod0_inv_mod1,
                params.mod1_inv_mod0,
                params.modulus,
                params.barrett_cr_0_modulus,
                params.barrett_cr_1_modulus,
            );
        }
        // Don't free device DB here — GEMV will reuse it
    } else {
        unsafe {
            gpu_hint_compute_from_host(
                db_u16.as_ptr(),
                output.as_mut_ptr(),
                db_rows as i32,
                db_rows_poly as i32,
                db_cols as i32,
                params.moduli[0] as u32,
                params.moduli[1] as u32,
                params.barrett_cr_1[0],
                params.barrett_cr_1[1],
                params.mod0_inv_mod1,
                params.mod1_inv_mod0,
                params.modulus,
                params.barrett_cr_0_modulus,
                params.barrett_cr_1_modulus,
            );
        }
    }

    // Output is already in transposed layout [poly_len × db_cols] from the GPU kernel
    output
}

/// Free GPU hint memory
pub fn gpu_hint_cleanup() {
    unsafe {
        gpu_hint_free();
    }
}

/// Free only the query buffer (keep hint output + NTT tables for prep_pack)
pub fn gpu_hint_cleanup_partial() {
    unsafe {
        gpu_hint_free_query_only();
    }
}

/// Free hint output and NTT tables (after prep_pack consumes them)
pub fn gpu_hint_cleanup_output() {
    unsafe {
        gpu_hint_free_output_and_tables();
    }
}

/// Get device pointer to hint output (stays on GPU)
pub fn gpu_hint_device_output() -> *const u64 {
    unsafe { gpu_hint_get_device_output() }
}

/// Get device pointer to NTT tables (stays on GPU)
pub fn gpu_hint_device_ntt_tables() -> *const u64 {
    unsafe { gpu_hint_get_ntt_tables() }
}

/// Get saved db_cols from hint kernel
pub fn gpu_hint_saved_db_cols() -> i32 {
    unsafe { gpu_hint_get_db_cols() }
}

/// Run GPU hint computation without D2H copy — result stays on device.
pub fn gpu_hint_compute_device_only(
    db_u16: &[u16],
    db_rows: usize,
    db_cols: usize,
    db_rows_poly: usize,
    params: &Params,
) {
    let d_db = unsafe { gpu_encode_get_device_db() };

    if !d_db.is_null() {
        // Using encoded DB on GPU (no-D2H mode)
        unsafe {
            gpu_hint_compute_no_d2h(
                d_db,
                db_rows as i32,
                db_rows_poly as i32,
                db_cols as i32,
                params.moduli[0] as u32,
                params.moduli[1] as u32,
                params.barrett_cr_1[0],
                params.barrett_cr_1[1],
                params.mod0_inv_mod1,
                params.mod1_inv_mod0,
                params.modulus,
                params.barrett_cr_0_modulus,
                params.barrett_cr_1_modulus,
            );
        }
    } else {
        // Fallback: compute with D2H (shouldn't normally happen)
        // Fallback to host path
        let _ = gpu_hint_compute_rs(db_u16, db_rows, db_cols, db_rows_poly, params);
    }
}

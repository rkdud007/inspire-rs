// Rust FFI wrapper for GPU prep_pack kernel
//
// Replaces CPU prep_pack_many_lwes by running negacyclic_perm + forward NTT
// on the GPU, directly consuming device-resident hint output.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static PREP_PACK_READY: AtomicBool = AtomicBool::new(false);
static PREP_PACK_DB_COLS: AtomicUsize = AtomicUsize::new(0);
static PREP_PACK_COEFF_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" {
    fn gpu_prep_pack_compute(
        d_hint_output: *const u64,
        d_ntt_tables: *const u64,
        db_cols: i32,
        mod0: u32,
        mod1: u32,
        modulus: u64,
    );

    fn gpu_prep_pack_get_output() -> *const u64;
    fn gpu_prep_pack_free();
}

/// Run the GPU prep_pack kernel. Hint output and NTT tables must still be on device.
pub fn gpu_prep_pack_run(
    d_hint_output: *const u64,
    d_ntt_tables: *const u64,
    db_cols: usize,
    mod0: u64,
    mod1: u64,
    modulus: u64,
    poly_len: usize,
    crt_count: usize,
) {
    let coeff_count = crt_count * poly_len;

    unsafe {
        gpu_prep_pack_compute(
            d_hint_output,
            d_ntt_tables,
            db_cols as i32,
            mod0 as u32,
            mod1 as u32,
            modulus,
        );
    }

    PREP_PACK_DB_COLS.store(db_cols, Ordering::Release);
    PREP_PACK_COEFF_COUNT.store(coeff_count, Ordering::Release);
    PREP_PACK_READY.store(true, Ordering::Release);
}

/// Check if GPU prep_pack output is available on device.
pub fn gpu_prep_pack_is_ready() -> bool {
    PREP_PACK_READY.load(Ordering::Acquire)
}

/// Get device pointer to the prepacked output buffer.
pub fn gpu_prep_pack_output_ptr() -> *const u64 {
    unsafe { gpu_prep_pack_get_output() }
}

/// Get device pointer for a specific column group's a_ct data.
/// Returns pointer to d_prepacked + col_group * gamma * coeff_count.
pub fn gpu_prep_pack_column_ptr(col_group: usize, gamma: usize) -> *const u64 {
    let coeff_count = PREP_PACK_COEFF_COUNT.load(Ordering::Acquire);
    let base = unsafe { gpu_prep_pack_get_output() };
    unsafe { base.add(col_group * gamma * coeff_count) }
}

/// Free device prep_pack output.
pub fn gpu_prep_pack_cleanup() {
    if PREP_PACK_READY.load(Ordering::Acquire) {
        unsafe {
            gpu_prep_pack_free();
        }
        PREP_PACK_READY.store(false, Ordering::Release);
        PREP_PACK_DB_COLS.store(0, Ordering::Release);
        PREP_PACK_COEFF_COUNT.store(0, Ordering::Release);
    }
}

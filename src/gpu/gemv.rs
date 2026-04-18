// Rust FFI wrapper for GPU-accelerated GEMV (online first-pass)

use std::sync::atomic::{AtomicU64, Ordering};

unsafe extern "C" {
    fn gpu_gemv_upload_db(h_db: *const u16, d_db_out: *mut *mut u16, db_size_bytes: usize);
    fn gpu_gemv_set_device_db(d_db_out: *mut *mut u16, d_db_in: *const u16, db_size_bytes: usize);
    fn gpu_gemv_free_db(d_db: *mut u16);
    fn gpu_gemv_query(
        h_c: *mut u64,
        h_a: *const u64,
        d_db: *const u16,
        db_rows: i32,
        b_cols: i32,
        mod0: u64,
        mod1: u64,
        cr0: u64,
        cr1: u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        modulus: u64,
        barrett_cr0_mod: u64,
        barrett_cr1_mod: u64,
    );
}

// Store the device DB pointer as a global (one DB resident on GPU at a time)
static GPU_DB_PTR: AtomicU64 = AtomicU64::new(0);
static GPU_DB_ROWS: AtomicU64 = AtomicU64::new(0);
static GPU_DB_COLS: AtomicU64 = AtomicU64::new(0);
// 0 = no DB, 1 = owned (we allocated), 2 = borrowed (from encode, don't free)
static GPU_DB_OWNED: AtomicU64 = AtomicU64::new(0);

/// Upload the encoded DB to GPU memory. Call once during server creation.
pub fn gpu_upload_db(db: &[u16], db_rows: usize, db_cols: usize) {
    assert_eq!(db.len(), db_rows * db_cols);

    // Free any previously uploaded DB (only if owned)
    free_if_owned();

    let db_size_bytes = db_rows * db_cols * std::mem::size_of::<u16>();
    let mut d_db: *mut u16 = std::ptr::null_mut();

    unsafe {
        gpu_gemv_upload_db(db.as_ptr(), &mut d_db, db_size_bytes);
    }

    if d_db.is_null() {
        eprintln!("[GPU GEMV] Failed to upload DB to GPU");
        return;
    }

    GPU_DB_PTR.store(d_db as u64, Ordering::SeqCst);
    GPU_DB_ROWS.store(db_rows as u64, Ordering::SeqCst);
    GPU_DB_COLS.store(db_cols as u64, Ordering::SeqCst);
    GPU_DB_OWNED.store(1, Ordering::SeqCst);
}

/// Adopt a device DB pointer from encode (no upload). Caller retains ownership.
pub fn gpu_set_device_db(d_db: *const u16, db_rows: usize, db_cols: usize) {
    // Free any previously uploaded DB (only if owned)
    free_if_owned();

    let db_size_bytes = db_rows * db_cols * std::mem::size_of::<u16>();
    let mut d_db_out: *mut u16 = std::ptr::null_mut();

    unsafe {
        gpu_gemv_set_device_db(&mut d_db_out, d_db, db_size_bytes);
    }

    GPU_DB_PTR.store(d_db_out as u64, Ordering::SeqCst);
    GPU_DB_ROWS.store(db_rows as u64, Ordering::SeqCst);
    GPU_DB_COLS.store(db_cols as u64, Ordering::SeqCst);
    GPU_DB_OWNED.store(2, Ordering::SeqCst); // borrowed
}

fn free_if_owned() {
    let old_ptr = GPU_DB_PTR.swap(0, Ordering::SeqCst);
    let owned = GPU_DB_OWNED.swap(0, Ordering::SeqCst);
    if old_ptr != 0 && owned == 1 {
        unsafe {
            gpu_gemv_free_db(old_ptr as *mut u16);
        }
    }
}

/// Run the GEMV on GPU: c = a^T * B mod q
/// Returns true if GPU was used, false if no DB is uploaded.
pub fn gpu_gemv(
    c: &mut [u64],
    a: &[u64],
    db_rows: usize,
    db_cols: usize,
    moduli: &[u64],      // [q0, q1]
    barrett_cr1: &[u64], // Barrett cr1 for each CRT modulus
    mod0_inv_mod1: u64,
    mod1_inv_mod0: u64,
    modulus: u64,             // q = q0 * q1
    barrett_cr0_modulus: u64, // Barrett cr0 for full modulus
    barrett_cr1_modulus: u64, // Barrett cr1 for full modulus
) -> bool {
    let d_db = GPU_DB_PTR.load(Ordering::SeqCst);
    if d_db == 0 {
        return false;
    }

    assert_eq!(a.len(), db_rows);
    assert_eq!(c.len(), db_cols);
    assert_eq!(GPU_DB_ROWS.load(Ordering::SeqCst) as usize, db_rows);
    assert_eq!(GPU_DB_COLS.load(Ordering::SeqCst) as usize, db_cols);

    // Barrett constants for CRT moduli: floor(2^64 / qi)
    let cr0 = barrett_cr1[0]; // NOTE: spiral-rs calls this "cr1" but it's floor(2^64/q)
    let cr1 = barrett_cr1[1];

    unsafe {
        gpu_gemv_query(
            c.as_mut_ptr(),
            a.as_ptr(),
            d_db as *const u16,
            db_rows as i32,
            db_cols as i32,
            moduli[0],
            moduli[1],
            cr0,
            cr1,
            mod0_inv_mod1,
            mod1_inv_mod0,
            modulus,
            barrett_cr0_modulus,
            barrett_cr1_modulus,
        );
    }

    true
}

/// Free GPU DB memory (only frees if we own the allocation)
pub fn gpu_free_db() {
    free_if_owned();
}

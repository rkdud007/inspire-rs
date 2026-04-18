// Rust FFI wrapper for GPU-accelerated InspiRING packing precomputation

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub static GPU_MUTEX: Mutex<()> = Mutex::new(());
static SETUP_DONE: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn gpu_packing_precomp(
        h_mono_table: *const u64,
        h_a_ct: *const u64,
        h_idx: *const i32,
        h_idx_bar: *const i32,
        h_mod_inv: *const u64,
        h_r_out: *mut u64,
        h_r_bar_out: *mut u64,
        num_to_pack_half: i32,
        non_zeros: i32,
        poly_len: i32,
        crt_count: i32,
        mod0: u64,
        mod1: u64,
        barrett_cr0: u64,
        barrett_cr1: u64,
    );

    fn gpu_packing_setup(
        h_mono_table: *const u64,
        h_idx: *const i32,
        h_idx_bar: *const i32,
        h_mod_inv: *const u64,
        num_to_pack_half: i32,
        non_zeros: i32,
        poly_len: i32,
        crt_count: i32,
        mod0: u64,
        mod1: u64,
        barrett_cr0: u64,
        barrett_cr1: u64,
    );

    fn gpu_packing_precomp_fast(h_a_ct: *const u64, h_r_out: *mut u64, h_r_bar_out: *mut u64);

    fn gpu_packing_precomp_fast_device(
        d_a_ct_device: *const u64,
        h_r_out: *mut u64,
        h_r_bar_out: *mut u64,
    );

    fn gpu_packing_precomp_fast_device_no_d2h(d_a_ct_device: *const u64);

    fn gpu_packing_teardown();

    fn gpu_packing_precomp_batched(
        h_mono_table: *const u64,
        h_a_ct_batch: *const u64,
        h_idx: *const i32,
        h_idx_bar: *const i32,
        h_mod_inv: *const u64,
        h_r_out: *mut u64,
        h_r_bar_out: *mut u64,
        batch_count: i32,
        num_to_pack_half: i32,
        non_zeros: i32,
        poly_len: i32,
        crt_count: i32,
        mod0: u64,
        mod1: u64,
        barrett_cr0: u64,
        barrett_cr1: u64,
    );

    fn gpu_packing_batched_fused(
        d_r_all: *mut u64,
        d_r_bar_all: *mut u64,
        d_prepacked: *const u64,
        num_cols: i32,
    );
}

/// Upload shared data to GPU once. Must be called before gpu_packing_precompute_fast.
pub fn gpu_packing_setup_shared(
    mono_table: &[u64],
    idx: &[i32],
    idx_bar: &[i32],
    mod_inv: &[u64],
    num_to_pack_half: usize,
    non_zeros: usize,
    poly_len: usize,
    crt_count: usize,
    moduli: &[u64],
) {
    let barrett_cr0 = if !moduli.is_empty() {
        ((1u128 << 64) / moduli[0] as u128) as u64
    } else {
        0
    };
    let barrett_cr1 = if moduli.len() > 1 {
        ((1u128 << 64) / moduli[1] as u128) as u64
    } else {
        0
    };

    unsafe {
        gpu_packing_setup(
            mono_table.as_ptr(),
            idx.as_ptr(),
            idx_bar.as_ptr(),
            mod_inv.as_ptr(),
            num_to_pack_half as i32,
            non_zeros as i32,
            poly_len as i32,
            crt_count as i32,
            moduli[0],
            if moduli.len() > 1 { moduli[1] } else { 0 },
            barrett_cr0,
            barrett_cr1,
        );
    }
    SETUP_DONE.store(true, Ordering::Release);
}

/// Free persistent GPU buffers.
pub fn gpu_packing_teardown_shared() {
    if SETUP_DONE.load(Ordering::Acquire) {
        unsafe {
            gpu_packing_teardown();
        }
        SETUP_DONE.store(false, Ordering::Release);
    }
}

pub fn is_packing_setup() -> bool {
    SETUP_DONE.load(Ordering::Acquire)
}

/// Fast per-column GPU call. Shared data must already be uploaded via setup.
/// Only uploads a_ct and downloads results. Caller must hold GPU_MUTEX.
pub fn gpu_packing_precompute_fast(
    a_ct: &[u64],
    num_to_pack_half: usize,
    coeff_count: usize,
) -> (Vec<u64>, Vec<u64>) {
    let out_size = num_to_pack_half * coeff_count;
    let mut r_out = vec![0u64; out_size];
    let mut r_bar_out = vec![0u64; out_size];

    unsafe {
        gpu_packing_precomp_fast(a_ct.as_ptr(), r_out.as_mut_ptr(), r_bar_out.as_mut_ptr());
    }

    (r_out, r_bar_out)
}

/// Fast per-column GPU call with device-resident a_ct (no H2D copy).
/// Shared data must already be uploaded via setup. Caller must hold GPU_MUTEX.
pub fn gpu_packing_precompute_fast_device(
    d_a_ct_device: *const u64,
    num_to_pack_half: usize,
    coeff_count: usize,
) -> (Vec<u64>, Vec<u64>) {
    let out_size = num_to_pack_half * coeff_count;
    let mut r_out = vec![0u64; out_size];
    let mut r_bar_out = vec![0u64; out_size];

    unsafe {
        gpu_packing_precomp_fast_device(d_a_ct_device, r_out.as_mut_ptr(), r_bar_out.as_mut_ptr());
    }

    (r_out, r_bar_out)
}

/// Fast per-column GPU call with device-resident a_ct, NO D2H copy.
/// Results stay in s_r_out/s_r_bar_out for collapse automorphism kernel.
/// Caller must hold GPU_MUTEX.
pub fn gpu_packing_precompute_fast_device_no_d2h(d_a_ct_device: *const u64) {
    unsafe {
        gpu_packing_precomp_fast_device_no_d2h(d_a_ct_device);
    }
}

/// GPU-accelerated packing precomputation (original per-column, alloc-each-time).
/// Uses a global mutex to serialize GPU access from concurrent rayon threads.
pub fn gpu_packing_precompute(
    mono_table: &[u64],
    a_ct: &[u64],
    idx: &[i32],
    idx_bar: &[i32],
    mod_inv: &[u64],
    num_to_pack_half: usize,
    non_zeros: usize,
    poly_len: usize,
    crt_count: usize,
    moduli: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let coeff_count = crt_count * poly_len;
    let out_size = num_to_pack_half * coeff_count;

    assert_eq!(mono_table.len(), 2 * poly_len * coeff_count);
    assert_eq!(a_ct.len(), non_zeros * coeff_count);
    assert_eq!(idx.len(), num_to_pack_half * non_zeros);
    assert_eq!(idx_bar.len(), num_to_pack_half * non_zeros);
    assert_eq!(mod_inv.len(), coeff_count);

    let mut r_out = vec![0u64; out_size];
    let mut r_bar_out = vec![0u64; out_size];

    let barrett_cr0 = if !moduli.is_empty() {
        ((1u128 << 64) / moduli[0] as u128) as u64
    } else {
        0
    };
    let barrett_cr1 = if moduli.len() > 1 {
        ((1u128 << 64) / moduli[1] as u128) as u64
    } else {
        0
    };

    let _guard = GPU_MUTEX.lock().unwrap();

    unsafe {
        gpu_packing_precomp(
            mono_table.as_ptr(),
            a_ct.as_ptr(),
            idx.as_ptr(),
            idx_bar.as_ptr(),
            mod_inv.as_ptr(),
            r_out.as_mut_ptr(),
            r_bar_out.as_mut_ptr(),
            num_to_pack_half as i32,
            non_zeros as i32,
            poly_len as i32,
            crt_count as i32,
            moduli[0],
            if moduli.len() > 1 { moduli[1] } else { 0 },
            barrett_cr0,
            barrett_cr1,
        );
    }

    (r_out, r_bar_out)
}

/// Batched GPU packing precomputation — all columns in a single GPU call.
pub fn gpu_packing_precompute_batched(
    mono_table: &[u64],
    a_ct_batch: &[u64],
    idx: &[i32],
    idx_bar: &[i32],
    mod_inv: &[u64],
    batch_count: usize,
    num_to_pack_half: usize,
    non_zeros: usize,
    poly_len: usize,
    crt_count: usize,
    moduli: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let coeff_count = crt_count * poly_len;
    let out_size = batch_count * num_to_pack_half * coeff_count;

    assert_eq!(mono_table.len(), 2 * poly_len * coeff_count);
    assert_eq!(a_ct_batch.len(), batch_count * non_zeros * coeff_count);
    assert_eq!(idx.len(), num_to_pack_half * non_zeros);
    assert_eq!(idx_bar.len(), num_to_pack_half * non_zeros);
    assert_eq!(mod_inv.len(), coeff_count);

    let mut r_out = vec![0u64; out_size];
    let mut r_bar_out = vec![0u64; out_size];

    let barrett_cr0 = if !moduli.is_empty() {
        ((1u128 << 64) / moduli[0] as u128) as u64
    } else {
        0
    };
    let barrett_cr1 = if moduli.len() > 1 {
        ((1u128 << 64) / moduli[1] as u128) as u64
    } else {
        0
    };

    unsafe {
        gpu_packing_precomp_batched(
            mono_table.as_ptr(),
            a_ct_batch.as_ptr(),
            idx.as_ptr(),
            idx_bar.as_ptr(),
            mod_inv.as_ptr(),
            r_out.as_mut_ptr(),
            r_bar_out.as_mut_ptr(),
            batch_count as i32,
            num_to_pack_half as i32,
            non_zeros as i32,
            poly_len as i32,
            crt_count as i32,
            moduli[0],
            if moduli.len() > 1 { moduli[1] } else { 0 },
            barrett_cr0,
            barrett_cr1,
        );
    }

    (r_out, r_bar_out)
}

/// Batched fused packing: all columns in one kernel with u32 multiply + fused scalar_mul.
/// Uses persistent shared data (mono_table, idx, mod_inv) already on device.
/// Writes directly to r_all/r_bar_all (no automorphism — done separately).
pub fn gpu_packing_batched_fused_call(
    d_r_all: *mut u64,
    d_r_bar_all: *mut u64,
    d_prepacked: *const u64,
    num_cols: usize,
) {
    unsafe {
        gpu_packing_batched_fused(d_r_all, d_r_bar_all, d_prepacked, num_cols as i32);
    }
}

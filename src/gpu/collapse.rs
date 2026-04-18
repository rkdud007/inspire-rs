// Rust FFI wrapper for GPU-accelerated InspiRING collapse kernel

use crate::packing::PackParams;
use crate::spiral::gadget::get_bits_per;
use crate::spiral::params::Params;
use crate::spiral::poly::{PolyMatrix, PolyMatrixNTT};
use std::sync::atomic::{AtomicBool, Ordering};

#[allow(dead_code)]
unsafe extern "C" {
    fn gpu_collapse_alloc_r(num_cols: i32, nphalf: i32, coeff_count: i32);

    fn gpu_collapse_automorph_column(
        d_r_flat: *const u64,
        d_r_bar_flat: *const u64,
        col: i32,
        nphalf: i32,
        coeff_count: i32,
        poly_len: i32,
        crt_count: i32,
        d_tables: *const u32,
        d_gen_pows: *const u32,
        num_tables: i32,
    );

    fn gpu_collapse_setup(
        h_w_all: *const u64,
        h_w_bar_all: *const u64,
        h_v_mask: *const u64,
        h_ntt_tables: *const u64,
        num_cols: i32,
        nphalf: i32,
        t_exp: i32,
        coeff_count: i32,
    );

    fn gpu_collapse_run(
        num_cols: i32,
        nphalf: i32,
        t_exp: i32,
        bits_per: i32,
        mod0: u32,
        mod1: u32,
        barrett_cr1_0: u64,
        barrett_cr1_1: u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        crt_modulus: u64,
        barrett_cr0_mod: u64,
        barrett_cr1_mod: u64,
    );

    fn gpu_collapse_download_a_hat(h_a_hat: *mut u64, num_cols: i32);

    fn gpu_collapse_get_bold_t() -> *mut u64;
    fn gpu_collapse_get_bold_t_bar() -> *mut u64;
    fn gpu_collapse_get_bold_t_hat() -> *mut u64;

    fn gpu_collapse_free_working();
    fn gpu_collapse_free_all();

    fn gpu_collapse_setup_fold(
        h_w_all: *const u64,
        h_w_bar_all: *const u64,
        h_v_mask: *const u64,
        h_ntt_tables: *const u64,
        num_cols: i32,
        nphalf: i32,
        t_exp: i32,
        coeff_count: i32,
    );
    fn gpu_collapse_run_fold(
        num_cols: i32,
        nphalf: i32,
        t_exp: i32,
        bits_per: i32,
        mod0: u32,
        mod1: u32,
        barrett_cr1_0: u64,
        barrett_cr1_1: u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        crt_modulus: u64,
        barrett_cr0_mod: u64,
        barrett_cr1_mod: u64,
    );
    fn gpu_collapse_download_a_hat_new(h_a_hat: *mut u64, num_cols: i32);
    fn gpu_collapse_get_bold_t_new() -> *mut u64;
    fn gpu_collapse_get_bold_t_bar_new() -> *mut u64;
    fn gpu_collapse_get_bold_t_hat_new() -> *mut u64;
    fn gpu_collapse_swap_bold_t();

    // From packing_kernel.cu: accessors for persistent buffers
    fn gpu_packing_get_s_r_out() -> *mut u64;
    fn gpu_packing_get_s_r_bar_out() -> *mut u64;
    fn gpu_packing_get_coeff_count() -> i32;
    fn gpu_packing_get_nphalf() -> i32;

    // From rotation_kernel.cu: reuse already-uploaded tables
    fn gpu_rotation_upload_tables(
        h_tables: *const u32,
        h_gen_pows: *const u32,
        num_tables: i32,
        num_rotations: i32,
        poly_len: i32,
    );

    // Accessors for r_all/r_bar_all device pointers
    fn gpu_collapse_get_r_all() -> *mut u64;
    fn gpu_collapse_get_r_bar_all() -> *mut u64;

    // Batched in-place automorphism
    fn gpu_collapse_batched_automorphism(
        d_tables: *const u32,
        d_gen_pows: *const u32,
        num_cols: i32,
        nphalf: i32,
        poly_len: i32,
        crt_count: i32,
    );

    // From packing_online_kernel.cu: adopt device pointers
    fn gpu_packing_online_adopt_precomp(
        d_bold_t: *mut u64,
        d_bold_t_bar: *mut u64,
        d_bold_t_hat: *mut u64,
        num_outputs: i32,
        n_inner: i32,
        t_exp_left: i32,
        poly_len: i32,
    );
}

static COLLAPSE_READY: AtomicBool = AtomicBool::new(false);
static BOLD_T_ON_DEVICE: AtomicBool = AtomicBool::new(false);

pub fn is_collapse_ready() -> bool {
    COLLAPSE_READY.load(Ordering::SeqCst)
}

pub fn is_bold_t_on_device() -> bool {
    BOLD_T_ON_DEVICE.load(Ordering::SeqCst)
}

/// Allocate r_all/r_bar_all device buffers for all columns.
pub fn gpu_collapse_alloc_r_all(num_cols: usize, nphalf: usize, coeff_count: usize) {
    unsafe {
        gpu_collapse_alloc_r(num_cols as i32, nphalf as i32, coeff_count as i32);
    }
}

/// Run automorphism for one column: copies packing s_r_out → r_all[col] with permutation.
/// Caller must hold GPU_MUTEX. d_tables/d_gen_pows must already be on device.
pub fn gpu_collapse_automorph_col(
    col: usize,
    nphalf: usize,
    coeff_count: usize,
    poly_len: usize,
    crt_count: usize,
    d_tables: *const u32,
    d_gen_pows: *const u32,
    num_tables: usize,
) {
    unsafe {
        let d_r_flat = gpu_packing_get_s_r_out();
        let d_r_bar_flat = gpu_packing_get_s_r_bar_out();
        gpu_collapse_automorph_column(
            d_r_flat,
            d_r_bar_flat,
            col as i32,
            nphalf as i32,
            coeff_count as i32,
            poly_len as i32,
            crt_count as i32,
            d_tables,
            d_gen_pows,
            num_tables as i32,
        );
    }
}

/// Upload automorphism tables to GPU and return device pointers.
/// Returns (d_tables, d_gen_pows, num_tables).
pub fn gpu_collapse_upload_automorph_tables(
    pack_params: &PackParams,
    params: &Params,
) -> (*const u32, *const u32, usize) {
    let poly_len = params.poly_len;
    let tables = &pack_params.tables;
    let gen_pows = &pack_params.gen_pows;
    let num_to_pack_half = poly_len / 2; // 1024

    let num_tables = tables.len();
    let mut flat_tables = Vec::with_capacity(num_tables * poly_len);
    for table in tables {
        for &idx in table {
            flat_tables.push(idx as u32);
        }
    }

    let gen_pows_u32: Vec<u32> = gen_pows[..num_to_pack_half]
        .iter()
        .map(|&x| x as u32)
        .collect();

    // Use the rotation kernel's upload function (it manages d_tables/d_gen_pows)
    unsafe {
        gpu_rotation_upload_tables(
            flat_tables.as_ptr(),
            gen_pows_u32.as_ptr(),
            num_tables as i32,
            num_to_pack_half as i32,
            poly_len as i32,
        );
    }

    // We need raw device pointers back — but rotation_kernel.cu stores them as statics.
    // We'll need to access them. Let's add accessors.
    // For now, we upload separately to collapse kernel's own buffers.
    // Actually, let's just upload to our own device buffers via cudaMalloc.
    // This is simpler and avoids cross-TU static access issues.

    // We'll allocate and upload in a helper CUDA function. For now, let's
    // use the rotation kernel's approach and add accessors.
    // Actually the collapse_automorph_column takes d_tables and d_gen_pows as params.
    // We need to get them from the rotation kernel. Let's add accessors.

    // HACK: For now, allocate separate copies.
    unsafe extern "C" {
        fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> i32;
        fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: i32) -> i32;
    }

    let tables_size = num_tables * poly_len * std::mem::size_of::<u32>();
    let gen_pows_size = num_to_pack_half * std::mem::size_of::<u32>();

    let mut d_tables_ptr: *mut u8 = std::ptr::null_mut();
    let mut d_gen_pows_ptr: *mut u8 = std::ptr::null_mut();

    unsafe {
        let r1 = cudaMalloc(&mut d_tables_ptr, tables_size);
        let r2 = cudaMalloc(&mut d_gen_pows_ptr, gen_pows_size);
        if r1 != 0 || r2 != 0 {
            panic!("[GPU Collapse] Failed to allocate automorph tables on GPU");
        }
        cudaMemcpy(
            d_tables_ptr,
            flat_tables.as_ptr() as *const u8,
            tables_size,
            1,
        ); // H2D=1
        cudaMemcpy(
            d_gen_pows_ptr,
            gen_pows_u32.as_ptr() as *const u8,
            gen_pows_size,
            1,
        );
    }

    (
        d_tables_ptr as *const u32,
        d_gen_pows_ptr as *const u32,
        num_tables,
    )
}

/// Setup collapse kernel: upload weights, NTT tables, allocate bold_t outputs.
pub fn gpu_collapse_setup_weights(
    w_all: &PolyMatrixNTT,
    w_bar_all: &PolyMatrixNTT,
    v_mask: &PolyMatrixNTT,
    params: &Params,
    num_cols: usize,
) {
    let poly_len = params.poly_len;
    let nphalf = poly_len / 2;
    let t_exp = params.t_exp_left;
    let coeff_count = params.crt_count * poly_len;
    let nphalf_m1 = nphalf - 1;

    // Flatten w_all: [nphalf_m1 * t_exp * coeff_count]
    let w_size = nphalf_m1 * t_exp * coeff_count;
    let mut w_flat = Vec::with_capacity(w_size);
    for i in 0..nphalf_m1 {
        for k in 0..t_exp {
            let poly = w_all.get_poly(i, k);
            w_flat.extend_from_slice(&poly[..coeff_count]);
        }
    }
    assert_eq!(w_flat.len(), w_size);

    let mut w_bar_flat = Vec::with_capacity(w_size);
    for i in 0..nphalf_m1 {
        for k in 0..t_exp {
            let poly = w_bar_all.get_poly(i, k);
            w_bar_flat.extend_from_slice(&poly[..coeff_count]);
        }
    }
    assert_eq!(w_bar_flat.len(), w_size);

    let v_size = t_exp * coeff_count;
    let mut v_flat = Vec::with_capacity(v_size);
    for k in 0..t_exp {
        let poly = v_mask.get_poly(0, k);
        v_flat.extend_from_slice(&poly[..coeff_count]);
    }
    assert_eq!(v_flat.len(), v_size);

    // NTT tables: [2 * 4 * poly_len]
    let mut flat_tables = Vec::with_capacity(2 * 4 * poly_len);
    for crt in 0..2 {
        for table_id in 0..4 {
            let table = &params.ntt_tables[crt][table_id];
            flat_tables.extend_from_slice(&table[..poly_len]);
        }
    }
    assert_eq!(flat_tables.len(), 2 * 4 * poly_len);

    unsafe {
        gpu_collapse_setup(
            w_flat.as_ptr(),
            w_bar_flat.as_ptr(),
            v_flat.as_ptr(),
            flat_tables.as_ptr(),
            num_cols as i32,
            nphalf as i32,
            t_exp as i32,
            coeff_count as i32,
        );
    }

    COLLAPSE_READY.store(true, Ordering::SeqCst);
}

/// Run the collapse kernel on all columns.
pub fn gpu_collapse_run_kernel(params: &Params, num_cols: usize) {
    let poly_len = params.poly_len;
    let nphalf = poly_len / 2;
    let t_exp = params.t_exp_left;
    let bits_per = get_bits_per(params, t_exp);

    unsafe {
        gpu_collapse_run(
            num_cols as i32,
            nphalf as i32,
            t_exp as i32,
            bits_per as i32,
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

/// Download a_hat from GPU. Returns flat Vec<u64> of [num_cols * poly_len].
pub fn gpu_collapse_download_a_hat_rs(num_cols: usize, poly_len: usize) -> Vec<u64> {
    let mut a_hat = vec![0u64; num_cols * poly_len];
    unsafe {
        gpu_collapse_download_a_hat(a_hat.as_mut_ptr(), num_cols as i32);
    }
    a_hat
}

/// Have the online packing kernel adopt our device-resident bold_t pointers.
pub fn gpu_collapse_adopt_bold_t_for_online(params: &Params, num_cols: usize) {
    let poly_len = params.poly_len;
    let nphalf = poly_len / 2;
    let nphalf_m1 = nphalf - 1;
    let t_exp = params.t_exp_left;
    let n_inner = nphalf_m1 * t_exp;

    unsafe {
        gpu_packing_online_adopt_precomp(
            gpu_collapse_get_bold_t(),
            gpu_collapse_get_bold_t_bar(),
            gpu_collapse_get_bold_t_hat(),
            num_cols as i32,
            n_inner as i32,
            t_exp as i32,
            poly_len as i32,
        );
    }

    BOLD_T_ON_DEVICE.store(true, Ordering::SeqCst);

    // Mark precomp as uploaded in the online packing module
    crate::gpu::packing_online::set_precomp_uploaded();
}

/// Free working buffers, keep bold_t for online.
pub fn gpu_collapse_cleanup_working() {
    unsafe {
        gpu_collapse_free_working();
    }
    COLLAPSE_READY.store(false, Ordering::SeqCst);
}

/// Free everything.
pub fn gpu_collapse_cleanup_all() {
    unsafe {
        gpu_collapse_free_all();
    }
    COLLAPSE_READY.store(false, Ordering::SeqCst);
    BOLD_T_ON_DEVICE.store(false, Ordering::SeqCst);
}

/// Get device pointers for r_all/r_bar_all (for batched packing output).
pub fn gpu_collapse_get_r_all_ptr() -> *mut u64 {
    unsafe { gpu_collapse_get_r_all() }
}

pub fn gpu_collapse_get_r_bar_all_ptr() -> *mut u64 {
    unsafe { gpu_collapse_get_r_bar_all() }
}

/// Run batched in-place automorphism on r_all and r_bar_all.
pub fn gpu_collapse_batched_automorphism_rs(
    d_tables: *const u32,
    d_gen_pows: *const u32,
    num_cols: usize,
    nphalf: usize,
    poly_len: usize,
    crt_count: usize,
) {
    unsafe {
        gpu_collapse_batched_automorphism(
            d_tables,
            d_gen_pows,
            num_cols as i32,
            nphalf as i32,
            poly_len as i32,
            crt_count as i32,
        );
    }
}

/// Setup fold collapse: upload weights to GPU, allocate NEW bold_t outputs.
pub fn gpu_collapse_setup_weights_fold(
    w_all: &PolyMatrixNTT,
    w_bar_all: &PolyMatrixNTT,
    v_mask: &PolyMatrixNTT,
    params: &Params,
    num_cols: usize,
) {
    let poly_len = params.poly_len;
    let nphalf = poly_len / 2;
    let t_exp = params.t_exp_left;
    let coeff_count = params.crt_count * poly_len;
    let nphalf_m1 = nphalf - 1;

    let w_size = nphalf_m1 * t_exp * coeff_count;
    let mut w_flat = Vec::with_capacity(w_size);
    for i in 0..nphalf_m1 {
        for k in 0..t_exp {
            let poly = w_all.get_poly(i, k);
            w_flat.extend_from_slice(&poly[..coeff_count]);
        }
    }

    let mut w_bar_flat = Vec::with_capacity(w_size);
    for i in 0..nphalf_m1 {
        for k in 0..t_exp {
            let poly = w_bar_all.get_poly(i, k);
            w_bar_flat.extend_from_slice(&poly[..coeff_count]);
        }
    }

    let v_size = t_exp * coeff_count;
    let mut v_flat = Vec::with_capacity(v_size);
    for k in 0..t_exp {
        let poly = v_mask.get_poly(0, k);
        v_flat.extend_from_slice(&poly[..coeff_count]);
    }

    let mut flat_tables = Vec::with_capacity(2 * 4 * poly_len);
    for crt in 0..2 {
        for table_id in 0..4 {
            let table = &params.ntt_tables[crt][table_id];
            flat_tables.extend_from_slice(&table[..poly_len]);
        }
    }

    unsafe {
        gpu_collapse_setup_fold(
            w_flat.as_ptr(),
            w_bar_flat.as_ptr(),
            v_flat.as_ptr(),
            flat_tables.as_ptr(),
            num_cols as i32,
            nphalf as i32,
            t_exp as i32,
            coeff_count as i32,
        );
    }
}

/// Run the fold collapse kernel (writes to NEW bold_t buffers).
pub fn gpu_collapse_run_kernel_fold(params: &Params, num_cols: usize) {
    let poly_len = params.poly_len;
    let nphalf = poly_len / 2;
    let t_exp = params.t_exp_left;
    let bits_per = get_bits_per(params, t_exp);

    unsafe {
        gpu_collapse_run_fold(
            num_cols as i32,
            nphalf as i32,
            t_exp as i32,
            bits_per as i32,
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

/// Download a_hat from the NEW (fold) buffers.
pub fn gpu_collapse_download_a_hat_new_rs(num_cols: usize, poly_len: usize) -> Vec<u64> {
    let mut a_hat = vec![0u64; num_cols * poly_len];
    unsafe {
        gpu_collapse_download_a_hat_new(a_hat.as_mut_ptr(), num_cols as i32);
    }
    a_hat
}

/// Swap: free old primary bold_t, promote new to primary.
pub fn gpu_collapse_swap_bold_t_rs() {
    unsafe {
        gpu_collapse_swap_bold_t();
    }
}

/// Have the online packing kernel adopt the NEW bold_t pointers (after swap).
pub fn gpu_collapse_adopt_bold_t_new_for_online(params: &Params, num_cols: usize) {
    let poly_len = params.poly_len;
    let nphalf = poly_len / 2;
    let nphalf_m1 = nphalf - 1;
    let t_exp = params.t_exp_left;
    let n_inner = nphalf_m1 * t_exp;

    unsafe {
        // After swap, d_bold_t/d_bold_t_bar/d_bold_t_hat point to the new data
        gpu_packing_online_adopt_precomp(
            gpu_collapse_get_bold_t(),
            gpu_collapse_get_bold_t_bar(),
            gpu_collapse_get_bold_t_hat(),
            num_cols as i32,
            n_inner as i32,
            t_exp as i32,
            poly_len as i32,
        );
    }

    BOLD_T_ON_DEVICE.store(true, Ordering::SeqCst);
    crate::gpu::packing_online::set_precomp_uploaded();
}

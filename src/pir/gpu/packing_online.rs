// Rust FFI wrapper for GPU-accelerated online packing

use crate::params::Params;
use crate::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::packing::{PackParams, PrecompInsPIR};
use crate::poly::add_raw;

#[allow(dead_code)]
unsafe extern "C" {
    fn gpu_packing_online_adopt_precomp(
        d_bold_t: *mut u64,
        d_bold_t_bar: *mut u64,
        d_bold_t_hat: *mut u64,
        num_outputs: i32,
        n_inner: i32,
        t_exp_left: i32,
        poly_len: i32,
    );

    fn gpu_rotation_upload_tables(
        h_tables: *const u32,
        h_gen_pows: *const u32,
        num_tables: i32,
        num_rotations: i32,
        poly_len: i32,
    );

    fn gpu_rotation_expand_keys(
        d_y_all_out: *mut u64,
        d_y_bar_all_out: *mut u64,
        h_y_body: *const u64,
        num_rotations: i32,
        t_exp_left: i32,
        poly_len: i32,
    );

    fn gpu_rotation_free();

    fn gpu_packing_online_ensure_key_buffers(n_inner: i32, t_exp_left: i32, poly_len: i32);

    fn gpu_packing_online_get_d_y_all() -> *mut u64;
    fn gpu_packing_online_get_d_y_bar_all() -> *mut u64;

    fn gpu_packing_online_upload_z_body(h_z_body: *const u64, t_exp_left: i32, poly_len: i32);

    fn gpu_packing_online_upload_precomp(
        h_bold_t: *const u64,
        h_bold_t_bar: *const u64,
        h_bold_t_hat: *const u64,
        num_outputs: i32,
        n_inner: i32,
        t_exp_left: i32,
        poly_len: i32,
    );

    fn gpu_packing_online_upload_keys(
        h_y_all: *const u64,
        h_y_bar_all: *const u64,
        h_z_body: *const u64,
        n_inner: i32,
        t_exp_left: i32,
        poly_len: i32,
    );

    fn gpu_packing_online_compute(
        h_output: *mut u64,
        num_outputs: i32,
        nphalf_minus_1: i32,
        t_exp_left: i32,
        poly_len: i32,
        addition_capacity: i32,
        mod0: u64,
        mod1: u64,
        cr0: u64,
        cr1: u64,
    );

    fn gpu_packing_online_free();

    fn gpu_packing_online_swap_precomp(
        new_bt: *mut u64,
        new_bt_bar: *mut u64,
        new_bt_hat: *mut u64,
        num_outputs: i32,
        n_inner: i32,
        t_exp_left: i32,
        poly_len: i32,
    );
}

static GPU_PRECOMP_UPLOADED: AtomicBool = AtomicBool::new(false);
static GPU_KEYS_UPLOADED: AtomicBool = AtomicBool::new(false);
static GPU_ROTATION_TABLES_UPLOADED: AtomicBool = AtomicBool::new(false);

/// Check if GPU packing precomp has been uploaded.
pub fn is_precomp_uploaded() -> bool {
    GPU_PRECOMP_UPLOADED.load(Ordering::SeqCst)
}

/// Check if rotation tables have been uploaded.
pub fn is_rotation_tables_uploaded() -> bool {
    GPU_ROTATION_TABLES_UPLOADED.load(Ordering::SeqCst)
}

/// Upload automorphism tables to GPU (called once at setup).
pub fn gpu_rotation_setup_tables(pack_params: &PackParams, params: &Params) {
    let poly_len = params.poly_len;
    let tables = &pack_params.tables;
    let gen_pows = &pack_params.gen_pows;
    let num_rotations = poly_len / 2 - 1; // 1023

    // Flatten tables to u32 (indices fit in u32)
    let num_tables = tables.len();
    let mut flat_tables = Vec::with_capacity(num_tables * poly_len);
    for table in tables {
        for &idx in table {
            flat_tables.push(idx as u32);
        }
    }

    // gen_pows as u32
    let gen_pows_u32: Vec<u32> = gen_pows[..num_rotations]
        .iter()
        .map(|&x| x as u32)
        .collect();

    unsafe {
        gpu_rotation_upload_tables(
            flat_tables.as_ptr(),
            gen_pows_u32.as_ptr(),
            num_tables as i32,
            num_rotations as i32,
            poly_len as i32,
        );
    }

    GPU_ROTATION_TABLES_UPLOADED.store(true, Ordering::SeqCst);
}

/// Run dummy packing_online kernel to warm GPU TLB/L2 for d_bold_t.
/// Eliminates ~80ms first-query overhead at 32GB during fold.
pub fn gpu_prewarm_online_kernels(params: &Params, num_outputs: usize) {
    let poly_len = params.poly_len;
    let t_exp_left = params.t_exp_left;
    let n_inner = (poly_len / 2 - 1) * t_exp_left;
    let nphalf_minus_1 = (poly_len >> 1) - 1;
    let addition_capacity = 1i32 << (64 - 2 * params.q2_bits as i32 - 1);
    let cr0 = params.barrett_cr_1[0];
    let cr1 = params.barrett_cr_1[1];

    // Ensure key buffers allocated
    unsafe {
        gpu_packing_online_ensure_key_buffers(n_inner as i32, t_exp_left as i32, poly_len as i32);
    }

    // Run packing_online kernel with garbage keys — warms TLB/L2 for d_bold_t
    let mut dummy = vec![0u64; num_outputs * 2 * poly_len];
    unsafe {
        gpu_packing_online_compute(
            dummy.as_mut_ptr(),
            num_outputs as i32,
            nphalf_minus_1 as i32,
            t_exp_left as i32,
            poly_len as i32,
            addition_capacity,
            params.moduli[0],
            params.moduli[1],
            cr0,
            cr1,
        );
    }
}

/// Expand keys on GPU: run rotation kernel + upload z_body.
/// Replaces both CPU expand() and gpu_packing_online_upload_keys_rs().
pub fn gpu_expand_and_set_keys(
    y_body_condensed: &PolyMatrixNTT,
    z_body_condensed: &PolyMatrixNTT,
    params: &Params,
) {
    assert!(
        GPU_ROTATION_TABLES_UPLOADED.load(Ordering::SeqCst),
        "Rotation tables not uploaded!"
    );

    let poly_len = params.poly_len;
    let t_exp_left = params.t_exp_left;
    let num_rotations = poly_len / 2 - 1;
    let n_inner = num_rotations * t_exp_left;

    // Ensure packing key buffers are allocated on GPU
    unsafe {
        gpu_packing_online_ensure_key_buffers(n_inner as i32, t_exp_left as i32, poly_len as i32);
    }

    // Extract y_body condensed tight
    let y_body_tight = extract_condensed_tight(y_body_condensed, poly_len);

    // Run rotation kernel — writes directly into d_y_all and d_y_bar_all
    unsafe {
        let d_y_all = gpu_packing_online_get_d_y_all();
        let d_y_bar_all = gpu_packing_online_get_d_y_bar_all();
        gpu_rotation_expand_keys(
            d_y_all,
            d_y_bar_all,
            y_body_tight.as_ptr(),
            num_rotations as i32,
            t_exp_left as i32,
            poly_len as i32,
        );
    }

    // Upload z_body separately (~48 KB)
    let z_body_tight = extract_condensed_tight(z_body_condensed, poly_len);
    unsafe {
        gpu_packing_online_upload_z_body(z_body_tight.as_ptr(), t_exp_left as i32, poly_len as i32);
    }

    GPU_KEYS_UPLOADED.store(true, Ordering::SeqCst);
}

/// Extract tightly-packed condensed polynomial data from a PolyMatrixNTT.
/// Condensed format: only first poly_len values of each polynomial slot are meaningful.
/// NTT storage has stride crt_count * poly_len per polynomial; we strip the zero CRT1 half.
fn extract_condensed_tight(mat: &PolyMatrixNTT, poly_len: usize) -> Vec<u64> {
    let num_polys = mat.rows * mat.cols;
    let mut tight = Vec::with_capacity(num_polys * poly_len);
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            let poly = mat.get_poly(i, j);
            tight.extend_from_slice(&poly[..poly_len]);
        }
    }
    tight
}

/// Upload precomp data to GPU. Called once after offline phase.
pub fn gpu_packing_online_setup_precomp(precomp_inspir_vec: &[PrecompInsPIR], params: &Params) {
    let poly_len = params.poly_len;
    let num_outputs = precomp_inspir_vec.len();
    let t_exp_left = params.t_exp_left;
    let num_to_pack_half = poly_len >> 1;
    let nphalf_minus_1 = num_to_pack_half - 1;
    let n_inner = nphalf_minus_1 * t_exp_left;

    let mut bold_t_all = Vec::with_capacity(num_outputs * n_inner * poly_len);
    let mut bold_t_bar_all = Vec::with_capacity(num_outputs * n_inner * poly_len);
    let mut bold_t_hat_all = Vec::with_capacity(num_outputs * t_exp_left * poly_len);

    for j in 0..num_outputs {
        let precomp = &precomp_inspir_vec[j];
        bold_t_all.extend_from_slice(&extract_condensed_tight(
            &precomp.bold_t_condensed,
            poly_len,
        ));
        bold_t_bar_all.extend_from_slice(&extract_condensed_tight(
            &precomp.bold_t_bar_condensed,
            poly_len,
        ));
        bold_t_hat_all.extend_from_slice(&extract_condensed_tight(
            &precomp.bold_t_hat_condensed,
            poly_len,
        ));
    }

    assert_eq!(bold_t_all.len(), num_outputs * n_inner * poly_len);
    assert_eq!(bold_t_bar_all.len(), num_outputs * n_inner * poly_len);
    assert_eq!(bold_t_hat_all.len(), num_outputs * t_exp_left * poly_len);

    unsafe {
        gpu_packing_online_upload_precomp(
            bold_t_all.as_ptr(),
            bold_t_bar_all.as_ptr(),
            bold_t_hat_all.as_ptr(),
            num_outputs as i32,
            n_inner as i32,
            t_exp_left as i32,
            poly_len as i32,
        );
    }

    GPU_PRECOMP_UPLOADED.store(true, Ordering::SeqCst);
}

/// Upload keys to GPU. Called per trial after key expansion.
pub fn gpu_packing_online_upload_keys_rs(
    y_all_condensed: &PolyMatrixNTT,
    y_bar_all_condensed: &PolyMatrixNTT,
    z_body_condensed: &PolyMatrixNTT,
    params: &Params,
) {
    let poly_len = params.poly_len;
    let t_exp_left = params.t_exp_left;
    let num_to_pack_half = poly_len >> 1;
    let nphalf_minus_1 = num_to_pack_half - 1;
    let n_inner = nphalf_minus_1 * t_exp_left;

    let y_all_tight = extract_condensed_tight(y_all_condensed, poly_len);
    let y_bar_all_tight = extract_condensed_tight(y_bar_all_condensed, poly_len);
    let z_body_tight = extract_condensed_tight(z_body_condensed, poly_len);

    assert_eq!(y_all_tight.len(), n_inner * poly_len);

    unsafe {
        gpu_packing_online_upload_keys(
            y_all_tight.as_ptr(),
            y_bar_all_tight.as_ptr(),
            z_body_tight.as_ptr(),
            n_inner as i32,
            t_exp_left as i32,
            poly_len as i32,
        );
    }

    GPU_KEYS_UPLOADED.store(true, Ordering::SeqCst);
}

/// Run GPU packing kernel and post-process results on CPU.
/// Returns Vec<PolyMatrixRaw> matching the output of pack_many_lwes_inspir.
pub fn gpu_packing_online_run<'a>(
    params: &'a Params,
    precomp_inspir_vec: &[PrecompInsPIR<'a>],
    b_values: &[u64],
    gamma: usize,
) -> Vec<PolyMatrixRaw<'a>> {
    if !GPU_PRECOMP_UPLOADED.load(Ordering::SeqCst) || !GPU_KEYS_UPLOADED.load(Ordering::SeqCst) {
        panic!("GPU packing data not uploaded! Call setup_precomp and upload_keys first.");
    }

    let poly_len = params.poly_len;
    let num_outputs = b_values.len() / gamma;
    let t_exp_left = params.t_exp_left;
    let num_to_pack_half = poly_len >> 1;
    let nphalf_minus_1 = num_to_pack_half - 1;
    let addition_capacity = 1i32 << (64 - 2 * params.q2_bits as i32 - 1);

    // Barrett constants for CRT moduli
    let cr0 = params.barrett_cr_1[0];
    let cr1 = params.barrett_cr_1[1];

    let mut gpu_output = vec![0u64; num_outputs * 2 * poly_len];
    unsafe {
        gpu_packing_online_compute(
            gpu_output.as_mut_ptr(),
            num_outputs as i32,
            nphalf_minus_1 as i32,
            t_exp_left as i32,
            poly_len as i32,
            addition_capacity,
            params.moduli[0],
            params.moduli[1],
            cr0,
            cr1,
        );
    }
    // Post-process on CPU with rayon: NTT inverse + CRT compose + add b_poly
    let group_size = poly_len / gamma;

    let result = (0..num_outputs)
        .into_par_iter()
        .map(|j| {
            // Create sum_poly from GPU output (NTT domain, uncondensed CRT)
            let mut sum_poly = PolyMatrixNTT::zero(params, 1, 1);
            let data = sum_poly.as_mut_slice();
            let base = j * 2 * poly_len;
            data[..poly_len].copy_from_slice(&gpu_output[base..base + poly_len]);
            data[poly_len..2 * poly_len]
                .copy_from_slice(&gpu_output[base + poly_len..base + 2 * poly_len]);

            // NTT inverse + CRT compose
            let sum_raw = sum_poly.raw();

            // Build b_poly from b_values
            let mut b_poly = PolyMatrixRaw::zero(params, 1, 1);
            let which_group = j / group_size;
            let within_group = j % group_size;
            for k in 0..gamma {
                let index = which_group * poly_len + within_group * gamma + k;
                if index < b_values.len() {
                    b_poly.get_poly_mut(0, 0)[k] = b_values[index];
                }
            }

            // Add b_poly + sum_raw
            let mut final_b_poly = PolyMatrixRaw::zero(params, 1, 1);
            add_raw(&mut final_b_poly, &b_poly, &sum_raw);

            // Form output: (a_hat, final_b_poly)
            let mut packed_raw = PolyMatrixRaw::zero(params, 2, 1);
            packed_raw.copy_into(&precomp_inspir_vec[j].a_hat, 0, 0);
            packed_raw.copy_into(&final_b_poly, 1, 0);
            packed_raw
        })
        .collect::<Vec<_>>();
    result
}

/// Mark precomp as uploaded (called after gpu_collapse adopts bold_t).
pub fn set_precomp_uploaded() {
    GPU_PRECOMP_UPLOADED.store(true, Ordering::SeqCst);
}

/// Swap precomp pointers after fold completes.
pub fn gpu_packing_online_swap_precomp_rs(
    new_bt: *mut u64,
    new_bt_bar: *mut u64,
    new_bt_hat: *mut u64,
    num_outputs: usize,
    n_inner: usize,
    t_exp_left: usize,
    poly_len: usize,
) {
    unsafe {
        gpu_packing_online_swap_precomp(
            new_bt,
            new_bt_bar,
            new_bt_hat,
            num_outputs as i32,
            n_inner as i32,
            t_exp_left as i32,
            poly_len as i32,
        );
    }
}

/// Free GPU packing memory
pub fn gpu_packing_online_cleanup() {
    GPU_PRECOMP_UPLOADED.store(false, Ordering::SeqCst);
    GPU_KEYS_UPLOADED.store(false, Ordering::SeqCst);
    unsafe {
        gpu_packing_online_free();
    }
}

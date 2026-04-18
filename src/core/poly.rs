#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;

use rand::RngExt;
use rand::distr::StandardUniform;
use rand_chacha::ChaCha20Rng;
use std::cell::RefCell;
use std::ops::{Add, Mul, Neg};
use std::sync::Arc;

use crate::{aligned_memory::*, arith::*, discrete_gaussian::*, ntt::*, params::*, util::*};

const SCRATCH_SPACE: usize = 8192;
thread_local!(static SCRATCH: RefCell<AlignedMemory64> = RefCell::new(AlignedMemory64::new(SCRATCH_SPACE)));

pub trait IntoSharedParams {
    fn into_shared_params(self) -> Arc<Params>;
}

impl IntoSharedParams for Arc<Params> {
    fn into_shared_params(self) -> Arc<Params> {
        self
    }
}

impl IntoSharedParams for &Arc<Params> {
    fn into_shared_params(self) -> Arc<Params> {
        Arc::clone(self)
    }
}

impl IntoSharedParams for &&Arc<Params> {
    fn into_shared_params(self) -> Arc<Params> {
        Arc::clone(self)
    }
}

impl IntoSharedParams for &Params {
    fn into_shared_params(self) -> Arc<Params> {
        Arc::new(self.clone())
    }
}

impl IntoSharedParams for &&Params {
    fn into_shared_params(self) -> Arc<Params> {
        Arc::new((**self).clone())
    }
}

impl IntoSharedParams for &mut Params {
    fn into_shared_params(self) -> Arc<Params> {
        Arc::new((*self).clone())
    }
}

impl IntoSharedParams for Params {
    fn into_shared_params(self) -> Arc<Params> {
        Arc::new(self)
    }
}

pub trait PolyMatrix {
    fn is_ntt(&self) -> bool;
    fn get_rows(&self) -> usize;
    fn get_cols(&self) -> usize;
    fn get_params(&self) -> &Params;
    fn num_words(&self) -> usize;
    fn as_slice(&self) -> &[u64];
    fn as_mut_slice(&mut self) -> &mut [u64];
    fn zero_out(&mut self) {
        for item in self.as_mut_slice() {
            *item = 0;
        }
    }
    fn get_poly(&self, row: usize, col: usize) -> &[u64] {
        let num_words = self.num_words();
        let start = (row * self.get_cols() + col) * num_words;
        // &self.as_slice()[start..start + num_words]
        unsafe { self.as_slice().get_unchecked(start..start + num_words) }
        // unsafe {
        //     let num_words = self.num_words();
        //     let ptr = self
        //         .as_slice()
        //         .as_ptr()
        //         .add((row * self.get_cols() + col) * num_words);
        //     std::slice::from_raw_parts(ptr, num_words)
        // }
    }
    fn get_poly_mut(&mut self, row: usize, col: usize) -> &mut [u64] {
        let num_words = self.num_words();
        let start = (row * self.get_cols() + col) * num_words;
        // &mut self.as_mut_slice()[start..start + num_words]
        unsafe {
            self.as_mut_slice()
                .get_unchecked_mut(start..start + num_words)
        }
        // unsafe {
        //     self.as_mut_slice().get_unchecked_mut(index)
        //     let num_words = self.num_words();
        //     let ptr = self
        //         .as_mut_slice()
        //         .as_mut_ptr()
        //         .add((row * self.get_cols() + col) * num_words);
        //     std::slice::from_raw_parts_mut(ptr, num_words)
        // }
    }
    fn copy_into(&mut self, p: &Self, target_row: usize, target_col: usize) {
        assert!(target_row < self.get_rows());
        assert!(target_col < self.get_cols());
        assert!(target_row + p.get_rows() <= self.get_rows());
        assert!(target_col + p.get_cols() <= self.get_cols());
        for r in 0..p.get_rows() {
            for c in 0..p.get_cols() {
                let pol_src = p.get_poly(r, c);
                let pol_dst = self.get_poly_mut(target_row + r, target_col + c);
                pol_dst.copy_from_slice(pol_src);
            }
        }
    }

    fn submatrix(&self, target_row: usize, target_col: usize, rows: usize, cols: usize) -> Self;
    fn pad_top(&self, pad_rows: usize) -> Self;
}

pub struct PolyMatrixRaw {
    pub params: Arc<Params>,
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedMemory64,
}

pub struct PolyMatrixNTT {
    pub params: Arc<Params>,
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedMemory64,
}

impl PolyMatrix for PolyMatrixRaw {
    fn is_ntt(&self) -> bool {
        false
    }
    fn get_rows(&self) -> usize {
        self.rows
    }
    fn get_cols(&self) -> usize {
        self.cols
    }
    fn get_params(&self) -> &Params {
        &self.params
    }
    fn as_slice(&self) -> &[u64] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u64] {
        self.data.as_mut_slice()
    }
    fn num_words(&self) -> usize {
        self.params.poly_len
    }
    fn pad_top(&self, pad_rows: usize) -> Self {
        let mut padded = Self::zero(self.params.clone(), self.rows + pad_rows, self.cols);
        padded.copy_into(&self, pad_rows, 0);
        padded
    }
    fn submatrix(&self, target_row: usize, target_col: usize, rows: usize, cols: usize) -> Self {
        let mut m = Self::zero(self.params.clone(), rows, cols);
        assert!(target_row < self.rows);
        assert!(target_col < self.cols);
        assert!(target_row + rows <= self.rows);
        assert!(target_col + cols <= self.cols);
        for r in 0..rows {
            for c in 0..cols {
                let pol_src = self.get_poly(target_row + r, target_col + c);
                let pol_dst = m.get_poly_mut(r, c);
                pol_dst.copy_from_slice(pol_src);
            }
        }
        m
    }
}

impl Clone for PolyMatrixRaw {
    fn clone(&self) -> Self {
        let mut data_clone = AlignedMemory64::new(self.data.len());
        data_clone
            .as_mut_slice()
            .copy_from_slice(self.data.as_slice());
        PolyMatrixRaw {
            params: self.params.clone(),
            rows: self.rows,
            cols: self.cols,
            data: data_clone,
        }
    }
}

impl PolyMatrixRaw {
    pub fn zero<P: IntoSharedParams>(params: P, rows: usize, cols: usize) -> PolyMatrixRaw {
        let params = params.into_shared_params();
        let num_coeffs = rows * cols * params.poly_len;
        let data = AlignedMemory64::new(num_coeffs);
        PolyMatrixRaw {
            params,
            rows,
            cols,
            data,
        }
    }

    pub fn random_rng<P: IntoSharedParams, T: rand::Rng>(
        params: P,
        rows: usize,
        cols: usize,
        rng: &mut T,
    ) -> Self {
        let params = params.into_shared_params();
        let poly_len = params.poly_len;
        let modulus = params.modulus;
        let mut iter = rng.sample_iter(StandardUniform);
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                for i in 0..poly_len {
                    let val: u64 = iter.next().unwrap();
                    out.get_poly_mut(r, c)[i] = val % modulus;
                }
            }
        }
        out
    }

    pub fn random<P: IntoSharedParams>(params: P, rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        Self::random_rng(params, rows, cols, &mut rng)
    }

    pub fn noise<P: IntoSharedParams>(
        params: P,
        rows: usize,
        cols: usize,
        dg: &DiscreteGaussian,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        dg.sample_matrix(&mut out, rng);
        out
    }

    pub fn fast_noise<P: IntoSharedParams>(
        params: P,
        rows: usize,
        cols: usize,
        dg: &DiscreteGaussian,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let params = params.into_shared_params();
        let modulus = params.modulus;
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        for r in 0..out.rows {
            for c in 0..out.cols {
                let poly = out.get_poly_mut(r, c);
                for z in 0..poly.len() {
                    poly[z] = dg.fast_sample(modulus, rng);
                }
            }
        }
        out
    }

    pub fn identity<P: IntoSharedParams>(params: P, rows: usize, cols: usize) -> PolyMatrixRaw {
        let params = params.into_shared_params();
        let num_coeffs = rows * cols * params.poly_len;
        let mut data = AlignedMemory::new(num_coeffs);
        for r in 0..rows {
            let c = r;
            let idx = r * cols * params.poly_len + c * params.poly_len;
            data[idx] = 1;
        }
        PolyMatrixRaw {
            params,
            rows,
            cols,
            data,
        }
    }

    pub fn single_value<P: IntoSharedParams>(params: P, value: u64) -> PolyMatrixRaw {
        let mut out = Self::zero(params, 1, 1);
        out.data[0] = value;
        out
    }

    pub fn ntt(&self) -> PolyMatrixNTT {
        to_ntt_alloc(&self)
    }

    pub fn reduce_mod(&mut self, modulus: u64) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                for z in 0..self.params.poly_len {
                    self.get_poly_mut(r, c)[z] %= modulus;
                }
            }
        }
    }

    pub fn apply_func<F: Fn(u64) -> u64>(&mut self, func: F) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                let pol_mut = self.get_poly_mut(r, c);
                for el in pol_mut {
                    *el = func(*el);
                }
            }
        }
    }

    pub fn to_vec(&self, modulus_bits: usize, num_coeffs: usize) -> Vec<u8> {
        let sz_bits = self.rows * self.cols * num_coeffs * modulus_bits;
        let sz_bytes = f64::ceil((sz_bits as f64) / 8f64) as usize + 32;
        let sz_bytes_roundup_16 = ((sz_bytes + 15) / 16) * 16;
        let mut data = vec![0u8; sz_bytes_roundup_16];
        let mut bit_offs = 0;
        for r in 0..self.rows {
            for c in 0..self.cols {
                for z in 0..num_coeffs {
                    let val = self.get_poly(r, c)[z];
                    write_arbitrary_bits(data.as_mut_slice(), val, bit_offs, modulus_bits);
                    // assert_eq!(
                    //     read_arbitrary_bits(data.as_slice(), bit_offs, modulus_bits),
                    //     val
                    // );

                    bit_offs += modulus_bits;
                }
                // round bit_offs down to nearest byte boundary
                bit_offs = (bit_offs / 8) * 8
            }
        }
        data
    }

}

impl PolyMatrix for PolyMatrixNTT {
    fn is_ntt(&self) -> bool {
        true
    }
    fn get_rows(&self) -> usize {
        self.rows
    }
    fn get_cols(&self) -> usize {
        self.cols
    }
    fn get_params(&self) -> &Params {
        &self.params
    }
    fn as_slice(&self) -> &[u64] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u64] {
        self.data.as_mut_slice()
    }
    fn num_words(&self) -> usize {
        self.params.poly_len * self.params.crt_count
    }
    fn pad_top(&self, pad_rows: usize) -> Self {
        let mut padded = Self::zero(self.params.clone(), self.rows + pad_rows, self.cols);
        padded.copy_into(&self, pad_rows, 0);
        padded
    }

    fn submatrix(&self, target_row: usize, target_col: usize, rows: usize, cols: usize) -> Self {
        let mut m = Self::zero(self.params.clone(), rows, cols);
        assert!(target_row < self.rows);
        assert!(target_col < self.cols);
        assert!(target_row + rows <= self.rows);
        assert!(target_col + cols <= self.cols);
        for r in 0..rows {
            for c in 0..cols {
                let pol_src = self.get_poly(target_row + r, target_col + c);
                let pol_dst = m.get_poly_mut(r, c);
                pol_dst.copy_from_slice(pol_src);
            }
        }
        m
    }
}

impl Clone for PolyMatrixNTT {
    fn clone(&self) -> Self {
        let mut data_clone = AlignedMemory64::new(self.data.len());
        data_clone
            .as_mut_slice()
            .copy_from_slice(self.data.as_slice());
        PolyMatrixNTT {
            params: self.params.clone(),
            rows: self.rows,
            cols: self.cols,
            data: data_clone,
        }
    }
}

impl PolyMatrixNTT {
    pub fn zero<P: IntoSharedParams>(params: P, rows: usize, cols: usize) -> PolyMatrixNTT {
        let params = params.into_shared_params();
        let num_coeffs = rows * cols * params.poly_len * params.crt_count;
        let data = AlignedMemory::new(num_coeffs);
        PolyMatrixNTT {
            params,
            rows,
            cols,
            data,
        }
    }

    pub fn random_rng<P: IntoSharedParams, T: rand::Rng>(
        params: P,
        rows: usize,
        cols: usize,
        rng: &mut T,
    ) -> Self {
        let params = params.into_shared_params();
        let crt_count = params.crt_count;
        let poly_len = params.poly_len;
        let moduli = params.moduli;
        let mut iter = rng.sample_iter(StandardUniform);
        let mut out = PolyMatrixNTT::zero(params, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                for i in 0..crt_count {
                    for j in 0..poly_len {
                        let idx = calc_index(&[i, j], &[crt_count, poly_len]);
                        let val: u64 = iter.next().unwrap();
                        out.get_poly_mut(r, c)[idx] = val % moduli[i];
                    }
                }
            }
        }
        out
    }

    pub fn random<P: IntoSharedParams>(params: P, rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        Self::random_rng(params, rows, cols, &mut rng)
    }

    pub fn raw(&self) -> PolyMatrixRaw {
        from_ntt_alloc(&self)
    }
}

pub fn shift_rows_by_one(inp: &PolyMatrixNTT) -> PolyMatrixNTT {
    if inp.rows == 1 {
        return inp.clone();
    }

    let all_but_last_row = inp.submatrix(0, 0, inp.rows - 1, inp.cols);
    let last_row = inp.submatrix(inp.rows - 1, 0, 1, inp.cols);
    let out = stack_ntt(&last_row, &all_but_last_row);
    out
}

pub fn multiply_poly(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = multiply_modular(params, a[idx], b[idx], c);
        }
    }
}

pub fn multiply_add_poly(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = multiply_add_modular(params, a[idx], b[idx], res[idx], c);
        }
    }
}

pub fn add_poly(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = add_modular(params, a[idx], b[idx], c);
        }
    }
}

pub fn add_poly_raw(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for i in 0..params.poly_len {
        res[i] = a[i] + b[i];
    }
}

pub fn add_poly_into(params: &Params, res: &mut [u64], a: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = add_modular(params, res[idx], a[idx], c);
        }
    }
}

pub fn sub_poly_into(params: &Params, res: &mut [u64], a: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = sub_modular(params, res[idx], a[idx], c);
        }
    }
}

pub fn invert_poly(params: &Params, res: &mut [u64], a: &[u64]) {
    for i in 0..params.poly_len {
        res[i] = params.modulus - a[i];
    }
}

pub fn invert_poly_ntt(params: &Params, res: &mut [u64], a: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = invert_modular(params, a[idx], c);
        }
    }
}

pub fn automorph_poly(params: &Params, res: &mut [u64], a: &[u64], t: usize) {
    let poly_len = params.poly_len;
    for i in 0..poly_len {
        let num = (i * t) / poly_len;
        let rem = (i * t) % poly_len;

        if num % 2 == 0 {
            res[rem] = a[i];
        } else {
            res[rem] = params.modulus - a[i];
        }
    }
}

pub fn automorph_poly_uncrtd(params: &Params, res: &mut [u64], a: &[u64], t: usize) {
    let poly_len = params.poly_len;
    for m in 0..params.crt_count {
        let a_chunk = &a[m * poly_len..(m + 1) * poly_len];
        let res_chunk = &mut res[m * poly_len..(m + 1) * poly_len];
        for i in 0..poly_len {
            let num = (i * t) / poly_len;
            let rem = (i * t) % poly_len;

            if num % 2 == 0 {
                res_chunk[rem] = a_chunk[i];
            } else {
                res_chunk[rem] = params.moduli[m] - a_chunk[i];
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiply_add_poly_avx(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in (0..params.poly_len).step_by(4) {
            unsafe {
                let p_x = &a[c * params.poly_len + i] as *const u64;
                let p_y = &b[c * params.poly_len + i] as *const u64;
                let p_z = &mut res[c * params.poly_len + i] as *mut u64;
                let x = _mm256_load_si256(p_x as *const __m256i);
                let y = _mm256_load_si256(p_y as *const __m256i);
                let z = _mm256_load_si256(p_z as *const __m256i);

                let product = _mm256_mul_epu32(x, y);
                let out = _mm256_add_epi64(z, product);

                _mm256_store_si256(p_z as *mut __m256i, out);
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiply_poly_avx(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in (0..params.poly_len).step_by(4) {
            unsafe {
                let p_x = &a[c * params.poly_len + i] as *const u64;
                let p_y = &b[c * params.poly_len + i] as *const u64;
                let p_z = &mut res[c * params.poly_len + i] as *mut u64;
                let x = _mm256_load_si256(p_x as *const __m256i);
                let y = _mm256_load_si256(p_y as *const __m256i);

                let product = _mm256_mul_epu32(x, y);

                _mm256_store_si256(p_z as *mut __m256i, product);
            }
        }
    }
}

pub fn modular_reduce(params: &Params, res: &mut [u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = barrett_coeff_u64(params, res[idx], c);
        }
    }
}

#[cfg(not(target_feature = "avx2"))]
pub fn multiply(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == b.cols);
    assert!(a.cols == b.rows);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..b.cols {
            for z in 0..params.poly_len * params.crt_count {
                res.get_poly_mut(i, j)[z] = 0;
            }
            for k in 0..a.cols {
                let res_poly = res.get_poly_mut(i, j);
                let pol1 = a.get_poly(i, k);
                let pol2 = b.get_poly(k, j);
                multiply_add_poly(params, res_poly, pol1, pol2);
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiply(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert_eq!(res.rows, a.rows);
    assert_eq!(res.cols, b.cols);
    assert_eq!(a.cols, b.rows);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..b.cols {
            for z in 0..params.poly_len * params.crt_count {
                res.get_poly_mut(i, j)[z] = 0;
            }
            let res_poly = res.get_poly_mut(i, j);
            for k in 0..a.cols {
                let pol1 = a.get_poly(i, k);
                let pol2 = b.get_poly(k, j);
                multiply_add_poly_avx(params, res_poly, pol1, pol2);
            }
            modular_reduce(params, res_poly);
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiply_no_reduce(
    res: &mut PolyMatrixNTT,
    a: &PolyMatrixNTT,
    b: &PolyMatrixNTT,
    start_inner_dim: usize,
) {
    assert_eq!(res.rows, a.rows);
    assert_eq!(res.cols, b.cols);
    assert_eq!(a.cols, b.rows);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..b.cols {
            let res_poly = res.get_poly_mut(i, j);
            for k in start_inner_dim..a.cols {
                let pol1 = a.get_poly(i, k);
                let pol2 = b.get_poly(k, j);
                multiply_add_poly_avx(params, res_poly, pol1, pol2);
            }
        }
    }
}

pub fn add(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);
    assert!(a.rows == b.rows);
    assert!(a.cols == b.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            let pol2 = b.get_poly(i, j);
            add_poly(params, res_poly, pol1, pol2);
        }
    }
}

pub fn add_raw(res: &mut PolyMatrixRaw, a: &PolyMatrixRaw, b: &PolyMatrixRaw) {
    assert_eq!(res.rows, a.rows);
    assert_eq!(res.cols, a.cols);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            let pol2 = b.get_poly(i, j);
            add_poly_raw(params, res_poly, pol1, pol2);
        }
    }
}

pub fn add_into(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..res.rows {
        for j in 0..res.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol2 = a.get_poly(i, j);
            add_poly_into(params, res_poly, pol2);
        }
    }
}

pub fn sub_into(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..res.rows {
        for j in 0..res.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol2 = a.get_poly(i, j);
            sub_poly_into(params, res_poly, pol2);
        }
    }
}

pub fn add_into_at(
    res: &mut PolyMatrixNTT,
    a: &PolyMatrixNTT,
    t_row: usize,
    t_col: usize,
) {
    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(t_row + i, t_col + j);
            let pol2 = a.get_poly(i, j);
            add_poly_into(params, res_poly, pol2);
        }
    }
}

pub fn invert(res: &mut PolyMatrixRaw, a: &PolyMatrixRaw) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            invert_poly(params, res_poly, pol1);
        }
    }
}

pub fn invert_ntt(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            invert_poly_ntt(params, res_poly, pol1);
        }
    }
}

pub fn automorph(res: &mut PolyMatrixRaw, a: &PolyMatrixRaw, t: usize) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            automorph_poly(params, res_poly, pol1, t);
        }
    }
}

pub fn automorph_alloc(a: &PolyMatrixRaw, t: usize) -> PolyMatrixRaw {
    let mut res = PolyMatrixRaw::zero(&a.params, a.rows, a.cols);
    automorph(&mut res, a, t);
    res
}

pub fn stack(a: &PolyMatrixRaw, b: &PolyMatrixRaw) -> PolyMatrixRaw {
    assert_eq!(a.cols, b.cols);
    let mut c = PolyMatrixRaw::zero(&a.params, a.rows + b.rows, a.cols);
    c.copy_into(a, 0, 0);
    c.copy_into(b, a.rows, 0);
    c
}

pub fn stack_ntt(a: &PolyMatrixNTT, b: &PolyMatrixNTT) -> PolyMatrixNTT {
    assert_eq!(a.cols, b.cols);
    let mut c = PolyMatrixNTT::zero(&a.params, a.rows + b.rows, a.cols);
    c.copy_into(a, 0, 0);
    c.copy_into(b, a.rows, 0);
    c
}

pub fn scalar_multiply(
    res: &mut PolyMatrixNTT,
    a: &PolyMatrixNTT,
    b: &PolyMatrixNTT,
) {
    assert_eq!(a.rows, 1);
    assert_eq!(a.cols, 1);

    let params = Arc::clone(&res.params);
    let params = params.as_ref();
    let pol2 = a.get_poly(0, 0);
    for i in 0..b.rows {
        for j in 0..b.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = b.get_poly(i, j);
            multiply_poly(params, res_poly, pol1, pol2);
        }
    }
}

pub fn scalar_multiply_alloc(a: &PolyMatrixNTT, b: &PolyMatrixNTT) -> PolyMatrixNTT {
    let mut res = PolyMatrixNTT::zero(&b.params, b.rows, b.cols);
    scalar_multiply(&mut res, a, b);
    res
}

pub fn single_poly<P: IntoSharedParams>(params: P, val: u64) -> PolyMatrixRaw {
    let mut res = PolyMatrixRaw::zero(params, 1, 1);
    res.get_poly_mut(0, 0)[0] = val;
    res
}

fn reduce_copy(params: &Params, out: &mut [u64], inp: &[u64]) {
    for n in 0..params.crt_count {
        for z in 0..params.poly_len {
            out[n * params.poly_len + z] = barrett_coeff_u64(params, inp[z], n);
        }
    }
}

pub fn to_ntt(a: &mut PolyMatrixNTT, b: &PolyMatrixRaw) {
    let params = Arc::clone(&a.params);
    let params = params.as_ref();
    for r in 0..a.rows {
        for c in 0..a.cols {
            let pol_src = b.get_poly(r, c);
            let pol_dst = a.get_poly_mut(r, c);
            reduce_copy(params, pol_dst, pol_src);
            ntt_forward(params, pol_dst);
        }
    }
}

pub fn to_ntt_no_reduce(a: &mut PolyMatrixNTT, b: &PolyMatrixRaw) {
    let params = Arc::clone(&a.params);
    let params = params.as_ref();
    for r in 0..a.rows {
        for c in 0..a.cols {
            let pol_src = b.get_poly(r, c);
            let pol_dst = a.get_poly_mut(r, c);
            for n in 0..params.crt_count {
                let idx = n * params.poly_len;
                pol_dst[idx..idx + params.poly_len].copy_from_slice(pol_src);
            }
            ntt_forward(params, pol_dst);
        }
    }
}

pub fn to_ntt_alloc(b: &PolyMatrixRaw) -> PolyMatrixNTT {
    let mut a = PolyMatrixNTT::zero(&b.params, b.rows, b.cols);
    to_ntt(&mut a, b);
    a
}

pub fn from_ntt(a: &mut PolyMatrixRaw, b: &PolyMatrixNTT) {
    let params = Arc::clone(&a.params);
    let params = params.as_ref();
    SCRATCH.with(|scratch_cell| {
        let scratch_vec = &mut *scratch_cell.borrow_mut();
        let scratch = scratch_vec.as_mut_slice();
        for r in 0..a.rows {
            for c in 0..a.cols {
                let pol_src = b.get_poly(r, c);
                let pol_dst = a.get_poly_mut(r, c);
                scratch[0..pol_src.len()].copy_from_slice(pol_src);
                ntt_inverse(params, scratch);
                for z in 0..params.poly_len {
                    pol_dst[z] = params.crt_compose(scratch, z);
                }
            }
        }
    });
}

pub fn from_ntt_scratch(
    a: &mut PolyMatrixRaw,
    scratch: &mut [u64],
    b: &PolyMatrixNTT,
) {
    assert_eq!(b.rows, 2);
    assert_eq!(b.cols, 1);

    let params = Arc::clone(&b.params);
    let params = params.as_ref();
    for r in 0..b.rows {
        for c in 0..b.cols {
            let pol_src = b.get_poly(r, c);
            scratch[0..pol_src.len()].copy_from_slice(pol_src);
            ntt_inverse(params, scratch);
            if r == 0 {
                let pol_dst = a.get_poly_mut(r, c);
                for z in 0..params.poly_len {
                    pol_dst[z] = params.crt_compose(scratch, z);
                }
            }
        }
    }
}

pub fn from_ntt_alloc(b: &PolyMatrixNTT) -> PolyMatrixRaw {
    let mut a = PolyMatrixRaw::zero(&b.params, b.rows, b.cols);
    from_ntt(&mut a, b);
    a
}

impl Neg for &PolyMatrixRaw {
    type Output = PolyMatrixRaw;

    fn neg(self) -> Self::Output {
        let mut out = PolyMatrixRaw::zero(&self.params, self.rows, self.cols);
        invert(&mut out, self);
        out
    }
}

impl Neg for &PolyMatrixNTT {
    type Output = PolyMatrixNTT;

    fn neg(self) -> Self::Output {
        let mut out = PolyMatrixNTT::zero(&self.params, self.rows, self.cols);
        invert_ntt(&mut out, self);
        out
    }
}

impl Mul for &PolyMatrixNTT {
    type Output = PolyMatrixNTT;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = PolyMatrixNTT::zero(&self.params, self.rows, rhs.cols);
        multiply(&mut out, self, rhs);
        out
    }
}

impl Add for &PolyMatrixNTT {
    type Output = PolyMatrixNTT;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = PolyMatrixNTT::zero(&self.params, self.rows, self.cols);
        add(&mut out, self, rhs);
        out
    }
}

impl Add for &PolyMatrixRaw {
    type Output = PolyMatrixRaw;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = PolyMatrixRaw::zero(&self.params, self.rows, self.cols);
        add_raw(&mut out, self, rhs);
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn get_params() -> Params {
        get_test_params()
    }

    fn assert_all_zero(a: &[u64]) {
        for i in a {
            assert_eq!(*i, 0);
        }
    }

    #[test]
    fn sets_all_zeros() {
        let params = Arc::new(get_params());
        let m1 = PolyMatrixNTT::zero(&params, 2, 1);
        assert_all_zero(m1.as_slice());
    }

    #[test]
    fn multiply_correctness() {
        let params = Arc::new(get_params());
        let m1 = PolyMatrixNTT::zero(&params, 2, 1);
        let m2 = PolyMatrixNTT::zero(&params, 3, 2);
        let m3 = &m2 * &m1;
        assert_all_zero(m3.as_slice());
    }

    #[test]
    fn full_multiply_correctness() {
        let params = Arc::new(get_params());
        let mut m1 = PolyMatrixRaw::zero(&params, 1, 1);
        let mut m2 = PolyMatrixRaw::zero(&params, 1, 1);
        m1.get_poly_mut(0, 0)[1] = 100;
        m2.get_poly_mut(0, 0)[1] = 7;
        let m1_ntt = to_ntt_alloc(&m1);
        let m2_ntt = to_ntt_alloc(&m2);
        let m3_ntt = &m1_ntt * &m2_ntt;
        let m3 = from_ntt_alloc(&m3_ntt);
        assert_eq!(m3.get_poly(0, 0)[2], 700);
    }

    fn get_alt_params() -> Params {
        Params::init(
            2048,
            &vec![180143985094819841u64],
            6.4,
            2,
            256,
            20,
            4,
            8,
            56,
            8,
            true,
            9,
            6,
            1,
            2048,
            0,
        )
    }

    #[test]
    fn alt_full_multiply_correctness() {
        let params = Arc::new(get_alt_params());
        let mut m1 = PolyMatrixRaw::zero(&params, 1, 1);
        let mut m2 = PolyMatrixRaw::zero(&params, 1, 1);
        m1.get_poly_mut(0, 0)[1] = 100;
        m2.get_poly_mut(0, 0)[1] = 7;
        let m1_ntt = to_ntt_alloc(&m1);
        let m2_ntt = to_ntt_alloc(&m2);
        let m3_ntt = &m1_ntt * &m2_ntt;
        let m3 = from_ntt_alloc(&m3_ntt);
        assert_eq!(m3.get_poly(0, 0)[2], 700);
    }

    #[test]
    fn to_vec_correctness() {
        let params = Arc::new(get_params());
        let mut m1 = PolyMatrixRaw::zero(&params, 1, 1);
        for i in 0..params.poly_len {
            m1.data[i] = 1;
        }
        let modulus_bits = 9;
        let v = m1.to_vec(modulus_bits, params.poly_len);
        for i in 0..v.len() {
            println!("{:?}", v[i]);
        }
        let mut bit_offs = 0;
        for i in 0..params.poly_len {
            let val = read_arbitrary_bits(v.as_slice(), bit_offs, modulus_bits);
            assert_eq!(m1.data[i], val);
            bit_offs += modulus_bits;
        }
    }
}

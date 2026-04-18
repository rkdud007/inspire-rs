use std::io;
use std::net::TcpStream;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::commons::{
    HandshakeParams, KeywordPirHandshake, MSG_KEYWORD_QUERY, MSG_KEYWORD_RESPONSE, RGSW_SEEDS,
    deserialize_keyword_query, recv_msg, send_msg, serialize_keyword_response,
};
use crate::dataset::Dataset;
use crate::gadget::gadget_invert;
use crate::gpu::gemv as cuda_gemv;
use crate::gpu::packing_online as cuda_packing_online;
use crate::kv::cuckoo::{CuckooParams, CuckooTable, DETERMINISTIC_SEED};
use crate::modulus_switch::ModulusSwitch;
use crate::packing::{PackParams, PrecompInsPIR};
use crate::params::Params;
use crate::pir::params::GetQPrime;
use crate::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw, multiply, to_ntt};

pub fn default_bucket_count(record_count: usize) -> usize {
    (record_count.max(512) * 2).next_power_of_two()
}

pub fn build_cuckoo_table(dataset: &Dataset, num_items: usize) -> CuckooTable {
    let params = CuckooParams::new(
        num_items,
        crate::PUBLIC_KEY_ID_LEN,
        dataset.public_key_len,
        0,
        DETERMINISTIC_SEED,
    );
    let mut table = CuckooTable::new(params);

    for record in &dataset.records {
        table.insert(&record.public_key_id, &record.public_key);
    }

    table
}

pub struct KeywordServer<'a> {
    table: CuckooTable,
    params: &'a Params,
    pack_params: PackParams<'a>,
    precomp_inspir_vec: Vec<PrecompInsPIR<'a>>,
    num_items: usize,
    entry_size: usize,
    dim0: usize,
    kem_name: String,
    db_rows: usize,
    db_cols: usize,
    gamma: usize,
    interpolate_degree: usize,
    c: usize,
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
}

impl<'a> KeywordServer<'a> {
    pub fn new(
        table: CuckooTable,
        params: &'a Params,
        precomp_inspir_vec: Vec<PrecompInsPIR<'a>>,
        num_items: usize,
        entry_size: usize,
        dim0: usize,
        kem_name: String,
        interpolate_degree: usize,
    ) -> Self {
        let gamma = params.poly_len;
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let db_cols_prime = db_cols / gamma;
        let c = db_cols_prime / interpolate_degree;
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        Self {
            table,
            params,
            pack_params: PackParams::new_fast(params, gamma),
            precomp_inspir_vec,
            num_items,
            entry_size,
            dim0,
            kem_name,
            db_rows,
            db_cols,
            gamma,
            interpolate_degree,
            c,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
        }
    }

    pub fn handshake(&self) -> HandshakeParams {
        let num_accounts = self
            .table
            .occupied
            .iter()
            .filter(|occupied| **occupied)
            .count()
            + self.table.stash.len();
        HandshakeParams {
            num_items: self.num_items,
            item_size_bits: self.entry_size * 8,
            dim0: self.dim0,
            keyword_pir: Some(KeywordPirHandshake {
                cuckoo_seed: DETERMINISTIC_SEED.to_vec(),
                num_hashes: self.table.params.num_hashes,
                key_size: self.table.params.key_size,
                value_size: self.table.params.value_size,
                entry_size: self.table.params.entry_size(),
                num_accounts,
                kem_name: self.kem_name.clone(),
            }),
        }
    }

    pub fn process_query(&self, stream: &mut TcpStream) -> io::Result<()> {
        let (msg_type, query_bytes) = recv_msg(stream)?;
        if msg_type != MSG_KEYWORD_QUERY {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unexpected message type: {msg_type}"),
            ));
        }

        let (packing_keys, per_query) =
            deserialize_keyword_query(self.params, &self.pack_params, query_bytes);
        cuda_packing_online::gpu_expand_and_set_keys(
            packing_keys.y_body_condensed.as_ref().unwrap(),
            packing_keys.z_body_condensed.as_ref().unwrap(),
            self.params,
        );

        let rgsw_fold_and_switch = |ct_gsw_body: &PolyMatrixNTT<'_>,
                                    packed: &[PolyMatrixRaw<'_>]|
         -> Vec<u8> {
            let mut ct_gsw = ct_gsw_body.pad_top(1);
            for i in 0..ct_gsw.cols {
                let a = PolyMatrixRaw::random_rng(
                    self.params,
                    1,
                    1,
                    &mut ChaCha20Rng::from_seed(RGSW_SEEDS[i]),
                );
                ct_gsw.copy_into(&(-&a).ntt(), 0, i);
            }

            let ell = ct_gsw.cols / 2;
            let rgsw_results: Vec<PolyMatrixRaw<'_>> = (0..self.c)
                .into_par_iter()
                .map(|which_poly| {
                    let mut ginv_c = PolyMatrixRaw::zero(self.params, 2 * ell, 1);
                    let mut ginv_c_ntt = PolyMatrixNTT::zero(self.params, 2 * ell, 1);
                    let mut sum = PolyMatrixRaw::zero(self.params, 2, 1);
                    for i in (0..self.interpolate_degree).rev() {
                        let mut prod = PolyMatrixNTT::zero(self.params, 2, 1);
                        gadget_invert(&mut ginv_c, &sum);
                        to_ntt(&mut ginv_c_ntt, &ginv_c);
                        multiply(&mut prod, &ct_gsw, &ginv_c_ntt);
                        sum = &prod.raw() + &packed[which_poly * self.interpolate_degree + i];
                        sum.reduce_mod(self.params.modulus);
                    }
                    sum
                })
                .collect();

            rgsw_results
                .iter()
                .map(|ct| ct.switch_and_keep(self.rlwe_q_prime_1, self.rlwe_q_prime_2, self.gamma))
                .collect::<Vec<_>>()
                .concat()
        };

        let gpu_first_pass = |packed_query_row: &crate::aligned_memory::AlignedMemory64| {
            let mut intermediate = crate::aligned_memory::AlignedMemory64::new(self.db_cols);
            cuda_gemv::gpu_gemv(
                intermediate.as_mut_slice(),
                packed_query_row.as_slice(),
                self.db_rows,
                self.db_cols,
                &self.params.moduli,
                &self.params.barrett_cr_1,
                self.params.mod0_inv_mod1,
                self.params.mod1_inv_mod0,
                self.params.modulus,
                self.params.barrett_cr_0_modulus,
                self.params.barrett_cr_1_modulus,
            );
            cuda_packing_online::gpu_packing_online_run(
                self.params,
                &self.precomp_inspir_vec,
                intermediate.as_slice(),
                self.gamma,
            )
        };

        let mut all_responses = Vec::with_capacity(per_query.len());
        if per_query.len() == 2 {
            let packed_0 = gpu_first_pass(&per_query[0].0);
            let (response_0, packed_1) = rayon::join(
                || rgsw_fold_and_switch(&per_query[0].1, &packed_0),
                || gpu_first_pass(&per_query[1].0),
            );
            let response_1 = rgsw_fold_and_switch(&per_query[1].1, &packed_1);
            all_responses.push(response_0);
            all_responses.push(response_1);
        } else {
            for (packed_query_row, ct_gsw_body) in &per_query {
                let packed = gpu_first_pass(packed_query_row);
                let response = rgsw_fold_and_switch(ct_gsw_body, &packed);
                all_responses.push(response);
            }
        }

        let stash_entries = self.table.stash.clone();
        let empty_sidecar: [(Vec<u8>, Vec<u8>); 0] = [];
        let response_data =
            serialize_keyword_response(&all_responses, &stash_entries, &empty_sidecar, 0);
        send_msg(stream, MSG_KEYWORD_RESPONSE, &response_data)
    }
}

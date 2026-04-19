use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io;
use std::marker::PhantomData;

use crate::aligned_memory::AlignedMemory64;
use crate::commons::{
    KeywordPirHandshake, KeywordQueryPayload, KeywordResponsePayload, PublicParams, RGSW_SEEDS,
    deserialize_keyword_query, params_rgswpir_given_input_size_and_dim0,
    serialize_keyword_response,
};
use crate::gadget::gadget_invert;
use crate::gpu::encode as cuda_encode;
use crate::gpu::gemv as cuda_gemv;
use crate::gpu::packing_online as cuda_packing_online;
use crate::kv::cuckoo::{CuckooParams, CuckooTable, DETERMINISTIC_SEED};
use crate::modulus_switch::ModulusSwitch;
use crate::number_theory::invert_uint_mod;
use crate::packing::{PackParams, PackingType, PrecompInsPIR};
use crate::pir::keyword::{KeywordRecord, KeywordServerConfig};
use crate::pir::measurement::Measurement;
use crate::pir::params::GetQPrime;
use crate::pir::scheme::ProtocolType;
use crate::pir::server::YServer;
use crate::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw, multiply, to_ntt};

pub fn default_bucket_count(record_count: usize) -> usize {
    (record_count.max(512) * 2).next_power_of_two()
}

pub fn build_cuckoo_table(
    records: &[KeywordRecord],
    num_items: usize,
    key_size: usize,
    value_size: usize,
) -> CuckooTable {
    let params = CuckooParams::new(num_items, key_size, value_size, 0, DETERMINISTIC_SEED);
    let mut table = CuckooTable::new(params);

    for record in records {
        table.insert(&record.key, &record.value);
    }

    table
}

pub struct KeywordServer<T: Sync> {
    table: CuckooTable,
    y_server: YServer<T>,
    precomp_inspir_vec: Vec<PrecompInsPIR>,
    num_items: usize,
    entry_size: usize,
    dim0: usize,
    db_rows: usize,
    db_cols: usize,
    gamma: usize,
    interpolate_degree: usize,
    c: usize,
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
}

#[derive(Clone, Debug)]
pub struct KeywordServerGpuMemoryEstimate {
    pub num_items: usize,
    pub item_size_bits: usize,
    pub dim0: usize,
    pub db_rows: usize,
    pub db_cols: usize,
    pub db_cols_prime: usize,
    pub instances: usize,
    pub poly_len: usize,
    pub crt_count: usize,
    pub t_exp_left: usize,
    pub db_bytes: u128,
    pub hint_bytes: u128,
    pub prep_pack_bytes: u128,
    pub r_buffers_bytes: u128,
    pub bold_t_bytes: u128,
    pub peak_bytes: u128,
}

#[derive(Clone, Debug)]
pub struct KeywordServerGpuMemoryCheck {
    pub estimate: KeywordServerGpuMemoryEstimate,
    pub available_bytes: u64,
}

struct KeywordServerSetupPlan {
    num_items: usize,
    entry_size: usize,
    item_size_bits: usize,
    dim0: usize,
    interpolate_degree: usize,
    params: crate::params::Params,
    db_rows: usize,
    db_cols: usize,
    db_cols_prime: usize,
    c: usize,
}

impl KeywordServerGpuMemoryEstimate {
    fn from_plan(plan: &KeywordServerSetupPlan) -> Self {
        let params = &plan.params;
        let db_rows = plan.db_rows as u128;
        let db_cols = plan.db_cols as u128;
        let db_cols_prime = plan.db_cols_prime as u128;
        let poly_len = params.poly_len as u128;
        let crt_count = params.crt_count as u128;
        let t_exp_left = params.t_exp_left as u128;
        let nphalf = (params.poly_len / 2) as u128;
        let coeff_count = crt_count * poly_len;

        let db_bytes = db_rows * db_cols * 2;
        let hint_bytes = db_cols * poly_len * 8;
        let prep_pack_bytes = db_cols * coeff_count * 8;
        let r_buffers_bytes = 2 * db_cols_prime * nphalf * coeff_count * 8;
        let bold_t_bytes = 2 * db_cols_prime * (nphalf - 1) * t_exp_left * poly_len * 8;
        let peak_bytes = db_bytes + prep_pack_bytes + r_buffers_bytes + bold_t_bytes;

        Self {
            num_items: plan.num_items,
            item_size_bits: plan.item_size_bits,
            dim0: plan.dim0,
            db_rows: plan.db_rows,
            db_cols: plan.db_cols,
            db_cols_prime: plan.db_cols_prime,
            instances: params.instances,
            poly_len: params.poly_len,
            crt_count: params.crt_count,
            t_exp_left: params.t_exp_left,
            db_bytes,
            hint_bytes,
            prep_pack_bytes,
            r_buffers_bytes,
            bold_t_bytes,
            peak_bytes,
        }
    }
}

impl KeywordServerSetupPlan {
    fn from_shape(
        record_count: usize,
        key_size: usize,
        value_size: usize,
        buckets: Option<usize>,
        dim0: Option<usize>,
    ) -> io::Result<Self> {
        if record_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "keyword record set must contain at least one record",
            ));
        }
        if key_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "keyword key size must be non-zero",
            ));
        }
        if value_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "keyword value size must be non-zero",
            ));
        }

        let num_items = buckets.unwrap_or_else(|| default_bucket_count(record_count));
        if num_items < record_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "bucket count {} must be >= record count {}",
                    num_items, record_count
                ),
            ));
        }

        let entry_size = key_size + value_size;
        let item_size_bits = entry_size * 8;
        if item_size_bits % 16 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "entry size must be a multiple of 16 bits",
            ));
        }

        let dim0 = dim0.unwrap_or_else(|| {
            let default_dim0 = if num_items >= 256_000_000 {
                32768
            } else {
                2048
            };
            let (_, _, (db_rows, _, _)) =
                params_rgswpir_given_input_size_and_dim0(num_items, item_size_bits, default_dim0);
            db_rows
        });

        let (params, interpolate_degree, _) =
            params_rgswpir_given_input_size_and_dim0(num_items, item_size_bits, dim0);

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let db_cols_prime = params.instances;
        let c = db_cols_prime / interpolate_degree;

        if num_items < db_rows {
            let min_records = db_rows / 2 + 1;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "bucket count {} is too small for the PIR scheme (needs at least {}); use at least {} records, or pass --buckets {}",
                    num_items, db_rows, min_records, db_rows
                ),
            ));
        }

        Ok(Self {
            num_items,
            entry_size,
            item_size_bits,
            dim0,
            interpolate_degree,
            params,
            db_rows,
            db_cols,
            db_cols_prime,
            c,
        })
    }

    fn from_records(records: &[KeywordRecord], config: &KeywordServerConfig) -> io::Result<Self> {
        if records.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "keyword record set must contain at least one record",
            ));
        }
        for (idx, record) in records.iter().enumerate() {
            if record.key.len() != config.key_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "record {} key has {} bytes, expected {}",
                        idx,
                        record.key.len(),
                        config.key_size
                    ),
                ));
            }
            if record.value.len() != config.value_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "record {} value has {} bytes, expected {}",
                        idx,
                        record.value.len(),
                        config.value_size
                    ),
                ));
            }
        }

        Self::from_shape(
            records.len(),
            config.key_size,
            config.value_size,
            config.buckets,
            config.dim0,
        )
    }
}

impl KeywordServer<u16> {
    pub fn estimate_setup_memory(
        records: &[KeywordRecord],
        config: &KeywordServerConfig,
    ) -> io::Result<KeywordServerGpuMemoryEstimate> {
        let plan = KeywordServerSetupPlan::from_records(records, config)?;
        Ok(KeywordServerGpuMemoryEstimate::from_plan(&plan))
    }

    pub fn estimate_setup_memory_for_shape(
        record_count: usize,
        key_size: usize,
        value_size: usize,
        buckets: Option<usize>,
        dim0: Option<usize>,
    ) -> io::Result<KeywordServerGpuMemoryEstimate> {
        let plan =
            KeywordServerSetupPlan::from_shape(record_count, key_size, value_size, buckets, dim0)?;
        Ok(KeywordServerGpuMemoryEstimate::from_plan(&plan))
    }

    pub fn check_setup_memory(
        records: &[KeywordRecord],
        config: &KeywordServerConfig,
    ) -> io::Result<KeywordServerGpuMemoryCheck> {
        let plan = KeywordServerSetupPlan::from_records(records, config)?;
        let estimate = KeywordServerGpuMemoryEstimate::from_plan(&plan);

        cuda_encode::init_memory_pool();
        let available_bytes = cuda_encode::available_memory_bytes();
        if available_bytes == 0 {
            return Err(io::Error::other(
                "failed to query available GPU memory from CUDA runtime",
            ));
        }
        if estimate.peak_bytes > available_bytes as u128 {
            return Err(io::Error::other(format!(
                "insufficient GPU memory for startup precomputation: requires {} bytes, only {} bytes available (db_rows={}, db_cols={}, instances={}, dim0={})",
                estimate.peak_bytes,
                available_bytes,
                estimate.db_rows,
                estimate.db_cols,
                estimate.instances,
                estimate.dim0,
            )));
        }

        Ok(KeywordServerGpuMemoryCheck {
            estimate,
            available_bytes,
        })
    }

    pub fn check_setup_memory_for_shape(
        record_count: usize,
        key_size: usize,
        value_size: usize,
        buckets: Option<usize>,
        dim0: Option<usize>,
    ) -> io::Result<KeywordServerGpuMemoryCheck> {
        let plan =
            KeywordServerSetupPlan::from_shape(record_count, key_size, value_size, buckets, dim0)?;
        let estimate = KeywordServerGpuMemoryEstimate::from_plan(&plan);

        cuda_encode::init_memory_pool();
        let available_bytes = cuda_encode::available_memory_bytes();
        if available_bytes == 0 {
            return Err(io::Error::other(
                "failed to query available GPU memory from CUDA runtime",
            ));
        }
        if estimate.peak_bytes > available_bytes as u128 {
            return Err(io::Error::other(format!(
                "insufficient GPU memory for startup precomputation: requires {} bytes, only {} bytes available (db_rows={}, db_cols={}, instances={}, dim0={})",
                estimate.peak_bytes,
                available_bytes,
                estimate.db_rows,
                estimate.db_cols,
                estimate.instances,
                estimate.dim0,
            )));
        }

        Ok(KeywordServerGpuMemoryCheck {
            estimate,
            available_bytes,
        })
    }

    pub fn setup(records: &[KeywordRecord], config: KeywordServerConfig) -> io::Result<Self> {
        let plan = KeywordServerSetupPlan::from_records(records, &config)?;
        let estimate = KeywordServerGpuMemoryEstimate::from_plan(&plan);

        cuda_encode::init_memory_pool();
        let available_bytes = cuda_encode::available_memory_bytes();
        if available_bytes == 0 {
            return Err(io::Error::other(
                "failed to query available GPU memory from CUDA runtime",
            ));
        }
        if estimate.peak_bytes > available_bytes as u128 {
            return Err(io::Error::other(format!(
                "insufficient GPU memory for startup precomputation: requires {} bytes, only {} bytes available (db_rows={}, db_cols={}, instances={}, dim0={})",
                estimate.peak_bytes,
                available_bytes,
                estimate.db_rows,
                estimate.db_cols,
                estimate.instances,
                estimate.dim0,
            )));
        }

        let num_items = plan.num_items;
        let entry_size = plan.entry_size;
        let item_size_bits = plan.item_size_bits;
        let dim0 = plan.dim0;
        let interpolate_degree = plan.interpolate_degree;
        let params = plan.params;
        let gamma = params.poly_len;
        let db_rows = plan.db_rows;
        let db_cols = plan.db_cols;
        let c = plan.c;
        let pt_modulus = params.pt_modulus;

        let per = num_items / db_rows;
        let item_size_elements = item_size_bits / 16;
        let table = build_cuckoo_table(records, num_items, config.key_size, config.value_size);
        let raw_db = table.to_raw_db(db_rows, db_cols, per, item_size_elements);

        let num_inv = invert_uint_mod(interpolate_degree as u64, pt_modulus).unwrap();
        let mut dummy_out = vec![0u16; raw_db.len().min(8)];
        let db_out_slice =
            unsafe { std::slice::from_raw_parts_mut(dummy_out.as_mut_ptr(), raw_db.len()) };
        cuda_encode::gpu_encode_database(
            &raw_db,
            db_out_slice,
            db_rows,
            db_cols,
            interpolate_degree,
            gamma,
            c,
            pt_modulus,
            num_inv,
        );
        drop(dummy_out);

        let d_db_ptr = cuda_encode::get_device_db();
        cuda_gemv::gpu_set_device_db(d_db_ptr, db_rows, db_cols);

        let packing_params_new = PackParams::new(&params, gamma);
        let half_packing_params = PackParams::new(&params, gamma >> 1);
        let mut packing_params_set = HashMap::new();
        let mut half_packing_params_set = HashMap::new();
        packing_params_set.insert(gamma, packing_params_new);
        half_packing_params_set.insert(gamma, half_packing_params);

        let smaller_params = params.clone();
        let db_buf_aligned = AlignedMemory64::new(1);
        let y_server: YServer<u16> = YServer {
            params: std::sync::Arc::new(params.clone()),
            packing_params_set,
            half_packing_params_set,
            smaller_params,
            db_buf_aligned,
            phantom: PhantomData,
            protocol_type: ProtocolType::SimplePIR,
            second_level_packing_mask: PackingType::InspiRING,
            second_level_packing_body: PackingType::NoPacking,
        };

        let mut measurement = Measurement::default();
        let offline_vals =
            y_server.perform_offline_precomputation_simplepir(gamma, Some(&mut measurement), false);
        let precomp_inspir_vec = offline_vals.precomp_inspir_vec;

        if !cuda_packing_online::is_rotation_tables_uploaded() {
            cuda_packing_online::gpu_rotation_setup_tables(
                &y_server.packing_params_set[&gamma],
                &params,
            );
        }

        Ok(KeywordServer::new(
            table,
            y_server,
            precomp_inspir_vec,
            num_items,
            entry_size,
            dim0,
            interpolate_degree,
        ))
    }
}

impl<T: Sync> KeywordServer<T> {
    pub fn new(
        table: CuckooTable,
        y_server: YServer<T>,
        precomp_inspir_vec: Vec<PrecompInsPIR>,
        num_items: usize,
        entry_size: usize,
        dim0: usize,
        interpolate_degree: usize,
    ) -> Self {
        let params = y_server.params.as_ref();
        let gamma = params.poly_len;
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let db_cols_prime = db_cols / gamma;
        let c = db_cols_prime / interpolate_degree;
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        Self {
            table,
            y_server,
            precomp_inspir_vec,
            num_items,
            entry_size,
            dim0,
            db_rows,
            db_cols,
            gamma,
            interpolate_degree,
            c,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
        }
    }

    fn params(&self) -> &crate::params::Params {
        self.y_server.params.as_ref()
    }

    pub fn public_params(&self) -> PublicParams {
        let num_records = self
            .table
            .occupied
            .iter()
            .filter(|occupied| **occupied)
            .count()
            + self.table.stash.len();
        PublicParams {
            num_items: self.num_items,
            item_size_bits: self.entry_size * 8,
            dim0: self.dim0,
            keyword_pir: Some(KeywordPirHandshake {
                cuckoo_seed: DETERMINISTIC_SEED.to_vec(),
                num_hashes: self.table.params.num_hashes,
                key_size: self.table.params.key_size,
                value_size: self.table.params.value_size,
                entry_size: self.table.params.entry_size(),
                num_records,
            }),
        }
    }

    pub fn respond(&self, query: &KeywordQueryPayload) -> KeywordResponsePayload {
        let params = self.params();
        cuda_packing_online::gpu_expand_and_set_keys(
            query.packing_keys.y_body_condensed.as_ref().unwrap(),
            query.packing_keys.z_body_condensed.as_ref().unwrap(),
            params,
        );
        let queries = &query.queries;

        let rgsw_fold_and_switch = |ct_gsw_body: &PolyMatrixNTT,
                                    packed: &[PolyMatrixRaw]|
         -> Vec<u8> {
            let mut ct_gsw = ct_gsw_body.pad_top(1);
            for i in 0..ct_gsw.cols {
                let a = PolyMatrixRaw::random_rng(
                    params,
                    1,
                    1,
                    &mut ChaCha20Rng::from_seed(RGSW_SEEDS[i]),
                );
                ct_gsw.copy_into(&(-&a).ntt(), 0, i);
            }

            let ell = ct_gsw.cols / 2;
            let rgsw_results: Vec<PolyMatrixRaw> = (0..self.c)
                .into_par_iter()
                .map(|which_poly| {
                    let mut ginv_c = PolyMatrixRaw::zero(params, 2 * ell, 1);
                    let mut ginv_c_ntt = PolyMatrixNTT::zero(params, 2 * ell, 1);
                    let mut sum = PolyMatrixRaw::zero(params, 2, 1);
                    for i in (0..self.interpolate_degree).rev() {
                        let mut prod = PolyMatrixNTT::zero(params, 2, 1);
                        gadget_invert(&mut ginv_c, &sum);
                        to_ntt(&mut ginv_c_ntt, &ginv_c);
                        multiply(&mut prod, &ct_gsw, &ginv_c_ntt);
                        sum = &prod.raw() + &packed[which_poly * self.interpolate_degree + i];
                        sum.reduce_mod(params.modulus);
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
                &params.moduli,
                &params.barrett_cr_1,
                params.mod0_inv_mod1,
                params.mod1_inv_mod0,
                params.modulus,
                params.barrett_cr_0_modulus,
                params.barrett_cr_1_modulus,
            );
            cuda_packing_online::gpu_packing_online_run(
                params,
                &self.precomp_inspir_vec,
                intermediate.as_slice(),
                self.gamma,
            )
        };

        let mut all_responses = Vec::with_capacity(queries.len());
        if queries.len() == 2 {
            let packed_0 = gpu_first_pass(&queries[0].packed_query_row);
            let (response_0, packed_1) = rayon::join(
                || rgsw_fold_and_switch(&queries[0].ct_gsw_body, &packed_0),
                || gpu_first_pass(&queries[1].packed_query_row),
            );
            let response_1 = rgsw_fold_and_switch(&queries[1].ct_gsw_body, &packed_1);
            all_responses.push(response_0);
            all_responses.push(response_1);
        } else {
            for query in queries.iter() {
                let packed = gpu_first_pass(&query.packed_query_row);
                let response = rgsw_fold_and_switch(&query.ct_gsw_body, &packed);
                all_responses.push(response);
            }
        }

        KeywordResponsePayload {
            responses: all_responses,
            stash_entries: self.table.stash.clone(),
        }
    }

    pub fn deserialize_query(&self, query_bytes: Vec<u8>) -> KeywordQueryPayload {
        let params = self.params();
        let packing_params = &self.y_server.packing_params_set[&self.gamma];
        deserialize_keyword_query(params, packing_params, query_bytes)
    }

    pub fn respond_to_serialized_query(&self, query_bytes: Vec<u8>) -> Vec<u8> {
        let query = self.deserialize_query(query_bytes);
        let response = self.respond(&query);
        serialize_keyword_response(&response)
    }
}

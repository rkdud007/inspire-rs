use crate::commons::{
    HandshakeParams, KeywordPirHandshake, KeywordQuery, KeywordResponsePayload, RGSW_SEEDS,
    serialize_keyword_query,
};
use crate::gadget::get_bits_per;
use crate::kv::cuckoo::{CuckooHash, u16_be_to_bytes};
use crate::modulus_switch::ModulusSwitch;
use crate::packing::{PackParams, PackingKeys, PackingType};
use crate::pir::client::{Client, YClient, decrypt_ct_reg_measured, pack_query};
use crate::pir::params::GetQPrime;
use crate::pir::scheme::{V_SEED, W_SEED};
use crate::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};
use crate::{KEM_ML_KEM_768, PUBLIC_KEY_ID_LEN, derive_public_key_id};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[derive(Clone)]
pub struct KeywordClientConfig {
    pub kem_name: String,
    pub packing_type: PackingType,
}

impl Default for KeywordClientConfig {
    fn default() -> Self {
        Self {
            kem_name: KEM_ML_KEM_768.to_string(),
            packing_type: PackingType::InspiRING,
        }
    }
}

pub struct KeywordClient {
    y_client: YClient,
    keyword: KeywordPirHandshake,
    kem_name: String,
    packing_type: PackingType,
    num_items: usize,
    interpolate_degree: usize,
    gamma: usize,
    c: usize,
    per: usize,
    item_size_elements: usize,
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
}

impl KeywordClient {
    pub fn from_handshake(
        handshake: &HandshakeParams,
        config: KeywordClientConfig,
    ) -> Result<Self, String> {
        let keyword = handshake
            .keyword_pir
            .clone()
            .ok_or_else(|| "server is not in keyword PIR mode".to_string())?;
        if keyword.key_size != PUBLIC_KEY_ID_LEN {
            return Err(format!(
                "expected {}-byte keyword ids, server uses {}",
                PUBLIC_KEY_ID_LEN, keyword.key_size
            ));
        }

        let (params, _, _) = crate::commons::params_rgswpir_given_input_size_and_dim0(
            handshake.num_items,
            handshake.item_size_bits,
            handshake.dim0,
        );
        let y_client = YClient::new(Client::init(&params));
        let params = y_client.params();
        let gamma = params.poly_len;
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let db_cols_prime = db_cols / gamma;
        let item_size_elements = handshake.item_size_bits / 16;
        if handshake.num_items < db_rows {
            return Err(format!(
                "server bucket count {} is too small for the PIR scheme (needs at least {}); the dataset must be regenerated with more records",
                handshake.num_items, db_rows
            ));
        }

        let (_, interpolate_degree, _) = crate::commons::params_rgswpir_given_input_size_and_dim0(
            handshake.num_items,
            handshake.item_size_bits,
            handshake.dim0,
        );
        let c = db_cols_prime / interpolate_degree;
        let kem_name = if !config.kem_name.is_empty() {
            config.kem_name
        } else if !keyword.kem_name.is_empty() {
            keyword.kem_name.clone()
        } else {
            KEM_ML_KEM_768.to_string()
        };

        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        Ok(Self {
            y_client,
            keyword,
            kem_name,
            packing_type: config.packing_type,
            num_items: handshake.num_items,
            interpolate_degree,
            gamma,
            c,
            per: handshake.num_items / db_rows,
            item_size_elements,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
        })
    }

    fn params(&self) -> &crate::params::Params {
        self.y_client.params()
    }

    pub fn kem_name(&self) -> &str {
        &self.kem_name
    }

    pub fn packing_params(&self) -> PackParams {
        PackParams::new_fast(self.params(), self.gamma)
    }

    pub fn positions(&self, query_id: &[u8; PUBLIC_KEY_ID_LEN]) -> Vec<usize> {
        let cuckoo_seed: [u8; 16] = self.keyword.cuckoo_seed.as_slice().try_into().unwrap();
        let hasher =
            CuckooHash::new_from_seed(cuckoo_seed, self.keyword.num_hashes, self.num_items);
        hasher.all_positions(query_id)
    }

    fn build_query_for_index(&self, which_item: usize) -> KeywordQuery {
        let target_row = which_item / self.per;
        let target_col = (which_item % self.per) * self.item_size_elements;
        let target_sub_col = (target_col % (self.interpolate_degree * self.gamma)) / self.gamma;

        let params = self.params();
        let mut ct_gsw_body = PolyMatrixNTT::zero(params, 1, 2 * params.t_gsw);
        let bits_per = get_bits_per(params, params.t_gsw);
        for j in 0..params.t_gsw {
            let mut sigma = PolyMatrixRaw::zero(params, 1, 1);
            let exponent = (2 * params.poly_len * target_sub_col / self.interpolate_degree)
                % (2 * params.poly_len);
            sigma.get_poly_mut(0, 0)[exponent % params.poly_len] = if exponent < params.poly_len {
                1u64 << (bits_per * j)
            } else {
                params.modulus - (1u64 << (bits_per * j))
            };
            let sigma_ntt = sigma.ntt();
            let ct = self.y_client.client().encrypt_matrix_reg(
                &sigma_ntt,
                &mut ChaCha20Rng::from_seed(rand::random::<[u8; 32]>()),
                &mut ChaCha20Rng::from_seed(RGSW_SEEDS[2 * j + 1]),
            );
            ct_gsw_body.copy_into(&ct.submatrix(1, 0, 1, 1), 0, 2 * j + 1);

            let prod = &self.y_client.client().get_sk_reg().ntt() * &sigma_ntt;
            let ct = self.y_client.client().encrypt_matrix_reg(
                &prod,
                &mut ChaCha20Rng::from_seed(rand::random::<[u8; 32]>()),
                &mut ChaCha20Rng::from_seed(RGSW_SEEDS[2 * j]),
            );
            ct_gsw_body.copy_into(&ct.submatrix(1, 0, 1, 1), 0, 2 * j);
        }

        let query_b_values = self.y_client.generate_query_b_values(
            crate::pir::scheme::SEED_0,
            params.db_dim_1,
            self.packing_type,
            target_row,
        );
        let packed_query_row = pack_query(params, &query_b_values);
        KeywordQuery {
            packed_query_row,
            ct_gsw_body,
        }
    }

    pub fn serialize_request(
        &self,
        packing_keys: &mut PackingKeys,
        positions: &[usize],
    ) -> Vec<u8> {
        let queries: Vec<KeywordQuery> = positions
            .iter()
            .map(|&bucket_idx| self.build_query_for_index(bucket_idx))
            .collect();
        serialize_keyword_query(self.params(), packing_keys, &queries)
    }

    pub fn build_request(&self, query_id: &[u8; PUBLIC_KEY_ID_LEN]) -> (Vec<usize>, Vec<u8>) {
        let positions = self.positions(query_id);
        let packing_params = self.packing_params();
        let sk_reg = self.y_client.client().get_sk_reg().clone();
        let mut packing_keys = PackingKeys::init_full(&packing_params, &sk_reg, W_SEED, V_SEED);
        let serialized = self.serialize_request(&mut packing_keys, &positions);
        (positions, serialized)
    }

    fn decrypt_response_values(&self, response_data: &[u8]) -> Vec<u64> {
        let chunk_size = response_data.len() / self.c;
        let sum_switched: Vec<Vec<u8>> = response_data
            .chunks_exact(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut results = Vec::new();
        let params = self.params();
        for which_poly in 0..self.c {
            let sum = PolyMatrixRaw::recover_how_many(
                params,
                self.rlwe_q_prime_1,
                self.rlwe_q_prime_2,
                self.gamma,
                sum_switched[which_poly].as_slice(),
            );
            results.push(sum);
        }

        results
            .iter()
            .flat_map(|ct| {
                decrypt_ct_reg_measured(self.y_client.client(), params, &ct.ntt(), params.poly_len)
                    .as_slice()[..self.gamma]
                    .to_vec()
            })
            .collect()
    }

    pub fn find_public_key(
        &self,
        query_id: &[u8; PUBLIC_KEY_ID_LEN],
        positions: &[usize],
        response: &KeywordResponsePayload,
    ) -> Option<Vec<u8>> {
        for (i, response_bytes) in response.responses.iter().enumerate() {
            let decrypted = self.decrypt_response_values(response_bytes);
            let bucket_idx = positions[i];
            let target_col = (bucket_idx % self.per) * self.item_size_elements;
            let db_poly = target_col / self.gamma;
            let result_poly = db_poly / self.interpolate_degree;
            let coeff_in_poly = target_col % self.gamma;
            let item_offset = result_poly * self.gamma + coeff_in_poly;
            let item_values = &decrypted[item_offset..item_offset + self.item_size_elements];
            let entry_bytes =
                u16_be_to_bytes(&item_values.iter().map(|&v| v as u16).collect::<Vec<_>>());

            if entry_bytes.len() >= self.keyword.entry_size
                && entry_bytes[..self.keyword.key_size] == *query_id
            {
                let public_key = entry_bytes
                    [self.keyword.key_size..self.keyword.key_size + self.keyword.value_size]
                    .to_vec();
                if derive_public_key_id(&public_key, &self.kem_name) == *query_id {
                    return Some(public_key);
                }
            }
        }

        for entry in &response.stash_entries {
            if entry.len() >= self.keyword.entry_size && entry[..self.keyword.key_size] == *query_id
            {
                let public_key = entry
                    [self.keyword.key_size..self.keyword.key_size + self.keyword.value_size]
                    .to_vec();
                if derive_public_key_id(&public_key, &self.kem_name) == *query_id {
                    return Some(public_key);
                }
            }
        }

        None
    }
}

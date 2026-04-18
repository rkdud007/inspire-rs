use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use inspire_rs::client::Client;
use inspire_rs::commons::params_rgswpir_given_input_size_and_dim0;
use inspire_rs::commons::{
    HandshakeParams, MSG_HANDSHAKE, MSG_KEYWORD_QUERY, MSG_KEYWORD_RESPONSE, RGSW_SEEDS,
    deserialize_keyword_response, recv_msg, send_msg, serialize_keyword_query,
};
use inspire_rs::gadget::get_bits_per;
use inspire_rs::kv::cuckoo::{CuckooHash, u16_be_to_bytes};
use inspire_rs::modulus_switch::ModulusSwitch;
use inspire_rs::packing::{PackParams, PackingKeys, PackingType};
use inspire_rs::params::Params;
use inspire_rs::pir::client::{YClient, decrypt_ct_reg_measured, pack_query};
use inspire_rs::pir::params::GetQPrime;
use inspire_rs::pir::scheme::{SEED_0, V_SEED, W_SEED};
use inspire_rs::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use inspire_rs::{PUBLIC_KEY_ID_LEN, derive_public_key_id};

#[derive(Parser, Debug)]
#[command(version, about = "Keyword PIR client for ML-KEM-768 public keys")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8082")]
    server: String,

    /// 32-byte public key id as hex.
    #[arg(long)]
    id: Option<String>,

    /// Public key bytes as hex. The client derives the id locally.
    #[arg(long)]
    public_key: Option<String>,

    /// Optional file containing raw public key bytes.
    #[arg(long)]
    public_key_file: Option<PathBuf>,
}

fn parse_hex(input: &str) -> Result<Vec<u8>, String> {
    hex::decode(input.trim_start_matches("0x")).map_err(|err| err.to_string())
}

fn resolve_query_id(args: &Args, kem_name: &str) -> Result<[u8; PUBLIC_KEY_ID_LEN], String> {
    if let Some(id_hex) = &args.id {
        let bytes = parse_hex(id_hex)?;
        if bytes.len() != PUBLIC_KEY_ID_LEN {
            return Err(format!(
                "expected {} bytes for --id, got {}",
                PUBLIC_KEY_ID_LEN,
                bytes.len()
            ));
        }
        let mut id = [0u8; PUBLIC_KEY_ID_LEN];
        id.copy_from_slice(&bytes);
        return Ok(id);
    }

    if let Some(public_key_hex) = &args.public_key {
        let public_key = parse_hex(public_key_hex)?;
        return Ok(derive_public_key_id(&public_key, kem_name));
    }

    if let Some(path) = &args.public_key_file {
        let public_key = std::fs::read(path).map_err(|err| err.to_string())?;
        return Ok(derive_public_key_id(&public_key, kem_name));
    }

    Err("provide either --id, --public-key, or --public-key-file".into())
}

fn generate_query_for_index<'a>(
    which_item: usize,
    params: &'a Params,
    y_client: &YClient<'a>,
    packing_type: PackingType,
    num_items: usize,
    item_size_bits: usize,
    interpolate_degree: usize,
    gamma: usize,
    db_rows: usize,
) -> (
    inspire_rs::aligned_memory::AlignedMemory64,
    PolyMatrixNTT<'a>,
) {
    let per = num_items / db_rows;
    let target_row = which_item / per;
    let target_col = (which_item % per) * (item_size_bits / 16);
    let target_sub_col = (target_col % (interpolate_degree * gamma)) / gamma;

    let mut ct_gsw_body = PolyMatrixNTT::zero(params, 1, 2 * params.t_gsw);
    let bits_per = get_bits_per(params, params.t_gsw);
    for j in 0..params.t_gsw {
        let mut sigma = PolyMatrixRaw::zero(params, 1, 1);
        let exponent =
            (2 * params.poly_len * target_sub_col / interpolate_degree) % (2 * params.poly_len);
        sigma.get_poly_mut(0, 0)[exponent % params.poly_len] = if exponent < params.poly_len {
            1u64 << (bits_per * j)
        } else {
            params.modulus - (1u64 << (bits_per * j))
        };
        let sigma_ntt = sigma.ntt();
        let ct = y_client.client().encrypt_matrix_reg(
            &sigma_ntt,
            &mut ChaCha20Rng::from_seed(rand::random::<[u8; 32]>()),
            &mut ChaCha20Rng::from_seed(RGSW_SEEDS[2 * j + 1]),
        );
        ct_gsw_body.copy_into(&ct.submatrix(1, 0, 1, 1), 0, 2 * j + 1);

        let prod = &y_client.client().get_sk_reg().ntt() * &sigma_ntt;
        let ct = &y_client.client().encrypt_matrix_reg(
            &prod,
            &mut ChaCha20Rng::from_seed(rand::random::<[u8; 32]>()),
            &mut ChaCha20Rng::from_seed(RGSW_SEEDS[2 * j]),
        );
        ct_gsw_body.copy_into(&ct.submatrix(1, 0, 1, 1), 0, 2 * j);
    }

    let query_b_values =
        y_client.generate_query_b_values(SEED_0, params.db_dim_1, packing_type, target_row);
    let packed_query_row = pack_query(params, &query_b_values);
    (packed_query_row, ct_gsw_body)
}

fn decrypt_response(
    y_client: &YClient<'_>,
    params: &Params,
    response_data: &[u8],
    c: usize,
    gamma: usize,
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
) -> Vec<u64> {
    let chunk_size = response_data.len() / c;
    let sum_switched: Vec<Vec<u8>> = response_data
        .chunks_exact(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    let mut results = Vec::new();
    for which_poly in 0..c {
        let sum = PolyMatrixRaw::recover_how_many(
            params,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
            gamma,
            sum_switched[which_poly].as_slice(),
        );
        results.push(sum);
    }

    results
        .iter()
        .flat_map(|ct| {
            decrypt_ct_reg_measured(y_client.client(), params, &ct.ntt(), params.poly_len)
                .as_slice()[..gamma]
                .to_vec()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut stream = TcpStream::connect(&args.server)?;
    stream.set_nodelay(true).ok();

    let (msg_type, handshake_bytes) = recv_msg(&mut stream)?;
    if msg_type != MSG_HANDSHAKE {
        return Err(format!("expected handshake, got message type {}", msg_type).into());
    }

    let handshake: HandshakeParams = serde_json::from_slice(&handshake_bytes)?;
    let keyword = handshake
        .keyword_pir
        .as_ref()
        .ok_or("server is not in keyword PIR mode")?;

    // Use kem_name from handshake so ID derivation matches how the dataset was built.
    let kem_name = if keyword.kem_name.is_empty() {
        inspire_rs::KEM_ML_KEM_768
    } else {
        keyword.kem_name.as_str()
    };
    let query_id = resolve_query_id(&args, kem_name)?;

    if keyword.key_size != PUBLIC_KEY_ID_LEN {
        return Err(format!(
            "expected {}-byte keyword ids, server uses {}",
            PUBLIC_KEY_ID_LEN, keyword.key_size
        )
        .into());
    }

    let num_items = handshake.num_items;
    let item_size_bits = handshake.item_size_bits;
    let dim0 = handshake.dim0;
    let (params, interpolate_degree, _) =
        params_rgswpir_given_input_size_and_dim0(num_items, item_size_bits, dim0);

    let gamma = params.poly_len;
    let packing_type = PackingType::InspiRING;
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let db_cols = params.instances * params.poly_len;
    let db_cols_prime = db_cols / gamma;
    let c = db_cols_prime / interpolate_degree;
    let item_size_elements = item_size_bits / 16;
    if num_items < db_rows {
        return Err(format!(
            "server bucket count {} is too small for the PIR scheme (needs at least {}); \
             the dataset must be regenerated with more records",
            num_items, db_rows
        )
        .into());
    }
    let per = num_items / db_rows;

    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();

    let cuckoo_seed: [u8; 16] = keyword.cuckoo_seed.as_slice().try_into()?;
    let hasher = CuckooHash::new_from_seed(cuckoo_seed, keyword.num_hashes, num_items);
    let positions = hasher.all_positions(&query_id);

    let mut client = Client::init(&params);
    let sk_reg_owned = client.get_sk_reg().clone();
    let packing_params = PackParams::new_fast(&params, gamma);
    let y_client = YClient::new(&mut client, &params);

    let t_total = Instant::now();
    let mut packing_keys = PackingKeys::init_full(&packing_params, &sk_reg_owned, W_SEED, V_SEED);

    let queries: Vec<_> = positions
        .iter()
        .map(|&bucket_idx| {
            generate_query_for_index(
                bucket_idx,
                &params,
                &y_client,
                packing_type,
                num_items,
                item_size_bits,
                interpolate_degree,
                gamma,
                db_rows,
            )
        })
        .collect();

    let serialized = serialize_keyword_query(&params, &mut packing_keys, &queries);
    send_msg(&mut stream, MSG_KEYWORD_QUERY, &serialized)?;

    let (msg_type, response_data) = recv_msg(&mut stream)?;
    if msg_type != MSG_KEYWORD_RESPONSE {
        return Err(format!("expected keyword response, got {}", msg_type).into());
    }

    let (responses, stash, _sidecar, _block_number) = deserialize_keyword_response(&response_data);

    let mut found_public_key = None;
    for (i, response_bytes) in responses.iter().enumerate() {
        let decrypted = decrypt_response(
            &y_client,
            &params,
            response_bytes,
            c,
            gamma,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
        );
        let bucket_idx = positions[i];
        let target_col = (bucket_idx % per) * item_size_elements;
        let db_poly = target_col / gamma;
        let result_poly = db_poly / interpolate_degree;
        let coeff_in_poly = target_col % gamma;
        let item_offset = result_poly * gamma + coeff_in_poly;
        let item_values = &decrypted[item_offset..item_offset + item_size_elements];
        let entry_bytes =
            u16_be_to_bytes(&item_values.iter().map(|&v| v as u16).collect::<Vec<_>>());

        if entry_bytes.len() >= keyword.entry_size && entry_bytes[..keyword.key_size] == query_id {
            let public_key =
                entry_bytes[keyword.key_size..keyword.key_size + keyword.value_size].to_vec();
            if derive_public_key_id(&public_key, kem_name) == query_id {
                found_public_key = Some(public_key);
                break;
            }
        }
    }

    if found_public_key.is_none() {
        for entry in &stash {
            if entry.len() >= keyword.entry_size && entry[..keyword.key_size] == query_id {
                let public_key =
                    entry[keyword.key_size..keyword.key_size + keyword.value_size].to_vec();
                if derive_public_key_id(&public_key, kem_name) == query_id {
                    found_public_key = Some(public_key);
                    break;
                }
            }
        }
    }

    println!("query id: {}", hex::encode(query_id));
    if let Some(public_key) = found_public_key {
        println!("public key: {}", hex::encode(public_key));
    } else {
        println!("not found");
    }
    println!("elapsed: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

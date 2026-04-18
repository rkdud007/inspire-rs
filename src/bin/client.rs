use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use inspire_rs::commons::params_rgswpir_given_input_size_and_dim0;
use inspire_rs::commons::{
    HandshakeParams, MSG_HANDSHAKE, MSG_KEYWORD_QUERY, MSG_KEYWORD_RESPONSE,
    deserialize_keyword_response, recv_msg, send_msg,
};
use inspire_rs::packing::{PackingKeys, PackingType};
use inspire_rs::pir::client::{Client, KeywordClient, KeywordClientConfig, YClient};
use inspire_rs::pir::scheme::{V_SEED, W_SEED};

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

    let (params, _, _) = params_rgswpir_given_input_size_and_dim0(
        handshake.num_items,
        handshake.item_size_bits,
        handshake.dim0,
    );

    let packing_type = PackingType::InspiRING;

    let client = Client::init(&params);
    let sk_reg_owned = client.get_sk_reg().clone();
    let y_client = YClient::new(&client, &params);
    let keyword_client = KeywordClient::from_handshake(
        &params,
        &y_client,
        &handshake,
        KeywordClientConfig {
            kem_name: kem_name.to_string(),
            packing_type,
        },
    )?;
    let positions = keyword_client.positions(&query_id);
    let packing_params = keyword_client.packing_params();

    let t_total = Instant::now();
    let mut packing_keys = PackingKeys::init_full(&packing_params, &sk_reg_owned, W_SEED, V_SEED);
    let serialized = keyword_client.serialize_request(&mut packing_keys, &positions);
    send_msg(&mut stream, MSG_KEYWORD_QUERY, &serialized)?;

    let (msg_type, response_data) = recv_msg(&mut stream)?;
    if msg_type != MSG_KEYWORD_RESPONSE {
        return Err(format!("expected keyword response, got {}", msg_type).into());
    }

    let (responses, stash, _sidecar, _block_number) = deserialize_keyword_response(&response_data);

    let found_public_key =
        keyword_client.find_public_key(&query_id, &positions, &responses, &stash);

    println!("query id: {}", hex::encode(query_id));
    if let Some(public_key) = found_public_key {
        println!("public key: {}", hex::encode(public_key));
    } else {
        println!("not found");
    }
    println!("elapsed: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

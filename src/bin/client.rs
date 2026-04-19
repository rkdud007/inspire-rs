use std::net::TcpStream;
use std::time::Instant;

use clap::Parser;
use inspire_rs::commons::{
    MSG_HANDSHAKE, MSG_KEYWORD_QUERY, MSG_KEYWORD_RESPONSE, PublicParams, recv_msg, send_msg,
};
use inspire_rs::packing::PackingType;
use inspire_rs::pir::client::{KeywordClient, KeywordClientConfig};

use inspire_rs::PUBLIC_KEY_ID_LEN;

#[derive(Parser, Debug)]
#[command(version, about = "Keyword PIR client for ML-KEM-768 public keys")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8082")]
    server: String,

    /// 32-byte public key id as hex.
    #[arg(long)]
    id: Option<String>,
}

fn parse_hex(input: &str) -> Result<Vec<u8>, String> {
    hex::decode(input.trim_start_matches("0x")).map_err(|err| err.to_string())
}

fn resolve_query_id(args: &Args) -> Result<[u8; PUBLIC_KEY_ID_LEN], String> {
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

    Err("provide --id".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut stream = TcpStream::connect(&args.server)?;
    stream.set_nodelay(true).ok();

    let (msg_type, handshake_bytes) = recv_msg(&mut stream)?;
    if msg_type != MSG_HANDSHAKE {
        return Err(format!("expected handshake, got message type {}", msg_type).into());
    }

    let public_params: PublicParams = serde_json::from_slice(&handshake_bytes)?;
    let keyword = public_params
        .keyword_pir
        .as_ref()
        .ok_or("server is not in keyword PIR mode")?;

    // Use kem_name from handshake so ID derivation matches how the dataset was built.
    let kem_name = if keyword.kem_name.is_empty() {
        inspire_rs::KEM_ML_KEM_768
    } else {
        keyword.kem_name.as_str()
    };
    let query_id = resolve_query_id(&args)?;

    if keyword.key_size != PUBLIC_KEY_ID_LEN {
        return Err(format!(
            "expected {}-byte keyword ids, server uses {}",
            PUBLIC_KEY_ID_LEN, keyword.key_size
        )
        .into());
    }

    let packing_type = PackingType::InspiRING;
    let keyword_client = KeywordClient::setup_from_public_params(
        &public_params,
        KeywordClientConfig {
            kem_name: kem_name.to_string(),
            packing_type,
        },
    )?;

    let t_total = Instant::now();
    let query = keyword_client.query(&query_id);
    let query_bytes = keyword_client.serialize_query(&query);
    send_msg(&mut stream, MSG_KEYWORD_QUERY, &query_bytes)?;

    let (msg_type, response_data) = recv_msg(&mut stream)?;
    if msg_type != MSG_KEYWORD_RESPONSE {
        return Err(format!("expected keyword response, got {}", msg_type).into());
    }

    let found_public_key =
        keyword_client.extract_from_serialized_response(&query_id, &query, &response_data);

    println!("query id: {}", hex::encode(query_id));
    if let Some(public_key) = found_public_key {
        println!("public key: {}", hex::encode(public_key));
    } else {
        println!("not found");
    }
    println!("elapsed: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

#![cfg(feature = "gpu")]

use std::env;
use std::fs;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use inspire_rs::commons::{
    MSG_HANDSHAKE, MSG_KEYWORD_QUERY, MSG_KEYWORD_RESPONSE, PublicParams, recv_msg, send_msg,
};
use inspire_rs::pir::client::{KeywordClient, KeywordClientConfig};
use inspire_rs::pir::server::KeywordServer;
use inspire_rs::{
    Dataset, DatasetGenerator, DatasetWriter, KemVariant, load_dataset, public_key_matches_id,
    setup_keyword_server,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!("inspire-rs-e2e-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn make_dataset(temp_dir: &Path) -> ([u8; 32], Vec<u8>, Dataset) {
    const RECORD_COUNT: usize = 10_000;
    const KEM: KemVariant = KemVariant::MlKem768;

    let generator = DatasetGenerator::new(KEM, RECORD_COUNT).with_seed(7);
    let dataset_path = temp_dir.join("keys.bin");
    let mut writer =
        DatasetWriter::create(&dataset_path, KEM.name(), RECORD_COUNT, KEM.public_key_len())
            .unwrap();
    generator.stream_to_writer(&mut writer, |_, _| {}).unwrap();
    writer.finish().unwrap();

    let dataset = load_dataset(&dataset_path).unwrap();
    let mut selection_rng = ChaCha20Rng::seed_from_u64(2026);
    let target_index = selection_rng.random_range(0..RECORD_COUNT);
    let expected_record = dataset.records[target_index].clone();

    (
        expected_record.public_key_id,
        expected_record.public_key,
        dataset,
    )
}

fn spawn_keyword_server(
    server: KeywordServer<u16>,
    listener: TcpListener,
) -> thread::JoinHandle<std::io::Result<()>> {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept()?;
        let public_params = serde_json::to_vec(&server.public_params()).unwrap();
        send_msg(&mut stream, MSG_HANDSHAKE, &public_params)?;
        let (msg_type, query_bytes) = recv_msg(&mut stream)?;
        if msg_type != MSG_KEYWORD_QUERY {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected message type: {msg_type}"),
            ));
        }
        let response_bytes = server.respond_to_serialized_query(query_bytes);
        send_msg(&mut stream, MSG_KEYWORD_RESPONSE, &response_bytes)
    })
}

#[test]
#[ignore = "requires --features gpu and a working CUDA toolchain/runtime"]
fn keyword_pir_roundtrip_tcp_mlkem() {
    let temp_dir = TempDirGuard::new();
    let (query_id, expected_public_key, dataset) = make_dataset(temp_dir.path());

    let server = setup_keyword_server(&dataset, None, None).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server_thread = spawn_keyword_server(server, listener);

    let mut stream = TcpStream::connect(addr).unwrap();
    let (msg_type, handshake_bytes) = recv_msg(&mut stream).unwrap();
    assert_eq!(msg_type, MSG_HANDSHAKE);

    let public_params: PublicParams = serde_json::from_slice(&handshake_bytes).unwrap();
    let client =
        KeywordClient::setup_from_public_params(&public_params, KeywordClientConfig::default())
            .unwrap();
    let query = client.query(&query_id).unwrap();
    let query_bytes = client.serialize_query(&query);
    send_msg(&mut stream, MSG_KEYWORD_QUERY, &query_bytes).unwrap();

    let (msg_type, response_bytes) = recv_msg(&mut stream).unwrap();
    assert_eq!(msg_type, MSG_KEYWORD_RESPONSE);

    let public_key = client
        .extract_from_serialized_response(&query_id, &query, &response_bytes)
        .expect("query should return the inserted public key");
    assert!(public_key_matches_id(
        &query_id,
        &public_key,
        &dataset.kem_name
    ));
    assert_eq!(public_key, expected_public_key);

    server_thread
        .join()
        .expect("server thread should not panic")
        .expect("server thread should complete successfully");
}

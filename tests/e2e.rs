#![cfg(feature = "gpu")]

use std::env;
use std::fs;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use inspire_rs::commons::{
    HandshakeParams, MSG_HANDSHAKE, MSG_KEYWORD_QUERY, MSG_KEYWORD_RESPONSE,
    deserialize_keyword_response, recv_msg, send_msg,
};
use inspire_rs::pir::client::{KeywordClient, KeywordClientConfig};
use inspire_rs::pir::server::KeywordServer;
use inspire_rs::{Dataset, KemVariant, generate_dataset, load_dataset, save_dataset};
use rand::{Rng, SeedableRng};
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
    let dataset = generate_dataset(KemVariant::MlKem768, RECORD_COUNT, 7).unwrap();
    let mut selection_rng = ChaCha20Rng::seed_from_u64(2026);
    let target_index = selection_rng.random_range(0..RECORD_COUNT);
    let expected_record = dataset.records[target_index].clone();

    let dataset_path = temp_dir.join("keys.bin");
    save_dataset(&dataset_path, &dataset).unwrap();
    let loaded = load_dataset(&dataset_path).unwrap();
    assert_eq!(loaded, dataset);

    (
        expected_record.public_key_id,
        expected_record.public_key,
        loaded,
    )
}

fn spawn_keyword_server(
    server: KeywordServer<u16>,
    listener: TcpListener,
) -> thread::JoinHandle<std::io::Result<()>> {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept()?;
        let handshake = serde_json::to_vec(&server.handshake()).unwrap();
        send_msg(&mut stream, MSG_HANDSHAKE, &handshake)?;
        server.process_query(&mut stream)
    })
}

#[test]
#[ignore = "requires --features gpu and a working CUDA toolchain/runtime"]
fn keyword_pir_roundtrip() {
    let temp_dir = TempDirGuard::new();
    let (query_id, expected_public_key, dataset) = make_dataset(temp_dir.path());

    let server = KeywordServer::from_dataset(&dataset, None, None).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server_thread = spawn_keyword_server(server, listener);

    let mut stream = TcpStream::connect(addr).unwrap();
    let (msg_type, handshake_bytes) = recv_msg(&mut stream).unwrap();
    assert_eq!(msg_type, MSG_HANDSHAKE);

    let handshake: HandshakeParams = serde_json::from_slice(&handshake_bytes).unwrap();
    let client = KeywordClient::from_handshake(&handshake, KeywordClientConfig::default()).unwrap();
    let (positions, request_bytes) = client.build_request(&query_id);
    send_msg(&mut stream, MSG_KEYWORD_QUERY, &request_bytes).unwrap();

    let (msg_type, response_bytes) = recv_msg(&mut stream).unwrap();
    assert_eq!(msg_type, MSG_KEYWORD_RESPONSE);

    let response = deserialize_keyword_response(&response_bytes);
    let public_key = client
        .find_public_key(&query_id, &positions, &response)
        .expect("query should return the inserted public key");
    assert_eq!(public_key, expected_public_key);

    server_thread
        .join()
        .expect("server thread should not panic")
        .expect("server thread should complete successfully");
}

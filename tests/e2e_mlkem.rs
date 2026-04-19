#![cfg(feature = "gpu")]

use inspire_rs::pir::client::{KeywordClient, KeywordClientConfig};
use inspire_rs::{DatasetGenerator, KemVariant, public_key_matches_id, setup_keyword_server};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;

#[test]
#[ignore = "requires --features gpu and a working CUDA toolchain/runtime"]
fn keyword_pir_roundtrip_mlkem() {
    const RECORD_COUNT: usize = 10_000;

    let dataset = DatasetGenerator::new(KemVariant::MlKem768, RECORD_COUNT)
        .with_seed(7)
        .generate_in_memory()
        .unwrap();
    let mut selection_rng = ChaCha20Rng::seed_from_u64(2026);
    let target_index = selection_rng.random_range(0..RECORD_COUNT);
    let expected_record = dataset.records[target_index].clone();

    let server = setup_keyword_server(&dataset, None, None).unwrap();
    let public_params = server.public_params();
    let client =
        KeywordClient::setup_from_public_params(&public_params, KeywordClientConfig::default())
            .unwrap();

    let query = client.query(&expected_record.public_key_id).unwrap();
    let response = server.respond(query.payload());
    let public_key = client
        .extract(&expected_record.public_key_id, &query, &response)
        .expect("query should return the inserted public key");

    assert!(public_key_matches_id(
        &expected_record.public_key_id,
        &public_key,
        &dataset.kem_name
    ));
    assert_eq!(public_key, expected_record.public_key);
}

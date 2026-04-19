use crate::dataset::{Dataset, DatasetRecord, derive_public_key_id};
use crate::pir::keyword::KeywordRecord;

#[cfg(feature = "gpu")]
use std::io;

#[cfg(feature = "gpu")]
use crate::pir::keyword::KeywordServerConfig;
#[cfg(feature = "gpu")]
use crate::pir::server::KeywordServer;

pub fn dataset_record_to_keyword_record(record: &DatasetRecord) -> KeywordRecord {
    KeywordRecord::new(record.public_key_id.to_vec(), record.public_key.clone())
}

pub fn dataset_to_keyword_records(dataset: &Dataset) -> Vec<KeywordRecord> {
    dataset
        .records
        .iter()
        .map(dataset_record_to_keyword_record)
        .collect()
}

pub fn public_key_matches_id(public_key_id: &[u8], public_key: &[u8], kem_name: &str) -> bool {
    derive_public_key_id(public_key, kem_name).as_slice() == public_key_id
}

#[cfg(feature = "gpu")]
pub fn setup_keyword_server(
    dataset: &Dataset,
    buckets: Option<usize>,
    dim0: Option<usize>,
) -> io::Result<KeywordServer<u16>> {
    let records = dataset_to_keyword_records(dataset);
    KeywordServer::setup(
        &records,
        KeywordServerConfig {
            key_size: crate::PUBLIC_KEY_ID_LEN,
            value_size: dataset.public_key_len,
            buckets,
            dim0,
        },
    )
}

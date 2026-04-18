pub mod core;
pub mod dataset;
pub mod pir;
pub mod protocol;

pub use core::{
    aligned_memory, arith, bits, discrete_gaussian, gadget, lwe, noise_analysis, noise_estimate,
    ntt, number_theory, params, poly, transpose, util,
};
pub use dataset::kem;
pub use pir::gpu;
pub use pir::keyword as kv;
pub use pir::{client::base as client, server::base as server};
pub use pir::{convolution, matmul, modulus_switch, packing};
pub use protocol::wire as commons;

pub use dataset::{
    DATASET_MAGIC, DEFAULT_DATASET_BATCH_SIZE, Dataset, DatasetRecord, DatasetWriter,
    KEM_ML_KEM_512, KEM_ML_KEM_768, KEM_ML_KEM_1024, PUBLIC_KEY_ID_DOMAIN, PUBLIC_KEY_ID_LEN,
    derive_public_key_id, generate_dataset, load_dataset, save_dataset,
};
pub use kem::KemVariant;

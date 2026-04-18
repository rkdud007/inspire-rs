pub mod aligned_memory;
pub mod arith;
pub mod bits;
pub mod client;
pub mod commons;
pub mod convolution;
pub mod dataset;
pub mod discrete_gaussian;
pub mod gadget;
pub mod gpu;
#[allow(dead_code)]
pub(crate) mod inspire_util;
pub mod kem;
pub mod kv;
pub mod lwe;
pub mod matmul;
pub mod modulus_switch;
pub mod noise_analysis;
pub mod noise_estimate;
pub mod ntt;
pub mod number_theory;
pub mod packing;
pub mod params;
pub mod pir;
pub mod poly;
pub mod server;
pub mod transpose;
pub mod util;

pub use dataset::{
    DATASET_MAGIC, Dataset, DatasetRecord, DatasetWriter, KEM_ML_KEM_512, KEM_ML_KEM_768,
    KEM_ML_KEM_1024, PUBLIC_KEY_ID_DOMAIN, PUBLIC_KEY_ID_LEN, derive_public_key_id, load_dataset,
    save_dataset,
};
pub use kem::KemVariant;

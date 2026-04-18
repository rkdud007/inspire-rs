use clap::ValueEnum;
use ml_kem::{
    MlKem512, MlKem768, MlKem1024, Seed,
    kem::{FromSeed, KeyExport},
};

use crate::{KEM_ML_KEM_512, KEM_ML_KEM_768, KEM_ML_KEM_1024};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum KemVariant {
    #[value(name = "ml-kem-512")]
    MlKem512,
    #[value(name = "ml-kem-768")]
    MlKem768,
    #[value(name = "ml-kem-1024")]
    MlKem1024,
}

impl KemVariant {
    pub fn name(self) -> &'static str {
        match self {
            Self::MlKem512 => KEM_ML_KEM_512,
            Self::MlKem768 => KEM_ML_KEM_768,
            Self::MlKem1024 => KEM_ML_KEM_1024,
        }
    }

    pub fn generate_public_key(self, seed: &Seed) -> Vec<u8> {
        match self {
            Self::MlKem512 => {
                let (_dk, ek) = MlKem512::from_seed(seed);
                ek.to_bytes().as_slice().to_vec()
            }
            Self::MlKem768 => {
                let (_dk, ek) = MlKem768::from_seed(seed);
                ek.to_bytes().as_slice().to_vec()
            }
            Self::MlKem1024 => {
                let (_dk, ek) = MlKem1024::from_seed(seed);
                ek.to_bytes().as_slice().to_vec()
            }
        }
    }
}

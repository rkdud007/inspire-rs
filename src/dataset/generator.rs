use std::collections::HashSet;
use std::io;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use super::dataset::{DEFAULT_DATASET_BATCH_SIZE, Dataset, DatasetRecord, derive_public_key_id};
use super::kem::KemVariant;

#[derive(Clone, Debug)]
pub struct DatasetGenerator {
    kem: KemVariant,
    count: usize,
    seed: u64,
    batch_size: usize,
}

impl DatasetGenerator {
    pub fn new(kem: KemVariant, count: usize) -> Self {
        Self {
            kem,
            count,
            seed: 1,
            batch_size: DEFAULT_DATASET_BATCH_SIZE,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn kem(&self) -> KemVariant {
        self.kem
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn generate(&self) -> io::Result<Dataset> {
        self.validate()?;

        let public_key_len = self.sample_public_key_len();
        let mut records = Vec::with_capacity(self.count);
        let mut seen_ids = HashSet::with_capacity(self.count);

        for chunk_start in (0..self.count).step_by(self.batch_size) {
            let chunk_end = (chunk_start + self.batch_size).min(self.count);
            let chunk = self.generate_chunk(chunk_start, chunk_end);

            for record in chunk {
                if !seen_ids.insert(record.public_key_id) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate public key id generated",
                    ));
                }
                debug_assert_eq!(record.public_key.len(), public_key_len);
                records.push(record);
            }
        }

        Ok(Dataset {
            kem_name: self.kem.name().to_string(),
            public_key_len,
            records,
        })
    }

    fn validate(&self) -> io::Result<()> {
        if self.count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "count must be greater than zero",
            ));
        }
        if self.batch_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "batch_size must be greater than zero",
            ));
        }
        Ok(())
    }

    fn sample_public_key_len(&self) -> usize {
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        let mut sample_seed = [0u8; SEED_BYTES];
        rng.fill_bytes(&mut sample_seed);
        self.kem.generate_public_key(&sample_seed).len()
    }

    fn generate_chunk(&self, chunk_start: usize, chunk_end: usize) -> Vec<DatasetRecord> {
        (chunk_start..chunk_end)
            .into_par_iter()
            .map(|idx| self.generate_record(idx))
            .collect()
    }

    fn generate_record(&self, idx: usize) -> DatasetRecord {
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        rng.set_word_pos(idx as u128 * SEED_WORDS);

        let mut record_seed = [0u8; SEED_BYTES];
        rng.fill_bytes(&mut record_seed);

        let public_key = self.kem.generate_public_key(&record_seed);
        let public_key_id = derive_public_key_id(&public_key, self.kem.name());

        DatasetRecord {
            public_key_id,
            public_key,
        }
    }
}

// Each ML-KEM seed is 64 bytes. ChaCha20 words are 4 bytes each, so each
// seed occupies 16 words in the stream. set_word_pos lets each worker seek
// independently while preserving deterministic sequential output.
const SEED_BYTES: usize = 64;
const CHACHA_WORD_BYTES: usize = 4;
const SEED_WORDS: u128 = (SEED_BYTES / CHACHA_WORD_BYTES) as u128;

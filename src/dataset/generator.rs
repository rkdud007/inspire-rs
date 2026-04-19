use std::collections::HashSet;
use std::io;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

use super::dataset::{
    DEFAULT_DATASET_BATCH_SIZE, Dataset, DatasetRecord, PUBLIC_KEY_ID_DOMAIN, PUBLIC_KEY_ID_LEN,
    derive_public_key_id,
};
use super::kem::KemVariant;
use super::writer::DatasetWriter;

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

    /// Generate all records into memory and return a `Dataset`.
    pub fn generate_in_memory(&self) -> io::Result<Dataset> {
        self.validate()?;

        let public_key_len = self.kem.public_key_len();
        let mut records = Vec::with_capacity(self.count);
        let mut seen_ids = HashSet::with_capacity(self.count);

        for chunk_start in (0..self.count).step_by(self.batch_size) {
            let chunk_end = (chunk_start + self.batch_size).min(self.count);
            for record in self.generate_chunk(chunk_start, chunk_end) {
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

    /// Stream generated records directly to a `DatasetWriter` batch by batch.
    /// Does not hold the full dataset in memory.
    /// Calls `progress(written, total)` after each batch is flushed.
    pub fn stream_to_writer(
        &self,
        writer: &mut DatasetWriter,
        mut progress: impl FnMut(usize, usize),
    ) -> io::Result<()> {
        self.validate()?;

        let pk_len = writer.public_key_len;
        let record_size = PUBLIC_KEY_ID_LEN + pk_len;
        let kem = self.kem;
        let kem_name = kem.name();

        for chunk_start in (0..self.count).step_by(self.batch_size) {
            let chunk_end = (chunk_start + self.batch_size).min(self.count);
            let chunk_len = chunk_end - chunk_start;

            // Pre-allocate flat buffer: [id(32) | pk(pk_len)] × chunk_len
            let mut buf = vec![0u8; chunk_len * record_size];

            buf.par_chunks_exact_mut(record_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let idx = chunk_start + i;
                    let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
                    rng.set_word_pos(idx as u128 * SEED_WORDS);
                    let mut seed = [0u8; SEED_BYTES];
                    rng.fill_bytes(&mut seed);

                    let (id_slot, pk_slot) = chunk.split_at_mut(PUBLIC_KEY_ID_LEN);
                    kem.generate_public_key_into(&seed, pk_slot);

                    let mut hasher = Sha3_256::new();
                    hasher.update(PUBLIC_KEY_ID_DOMAIN);
                    hasher.update(kem_name.as_bytes());
                    hasher.update(&*pk_slot);
                    id_slot.copy_from_slice(&hasher.finalize());
                });

            // Write the batch sequentially (DatasetWriter is not Sync)
            for chunk in buf.chunks_exact(record_size) {
                let id: [u8; PUBLIC_KEY_ID_LEN] = chunk[..PUBLIC_KEY_ID_LEN].try_into().unwrap();
                let record = DatasetRecord {
                    public_key_id: id,
                    public_key: chunk[PUBLIC_KEY_ID_LEN..].to_vec(),
                };
                writer.write_record(&record)?;
            }

            progress(chunk_end, self.count);
        }

        Ok(())
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

    fn generate_chunk(&self, chunk_start: usize, chunk_end: usize) -> Vec<DatasetRecord> {
        let pk_len = self.kem.public_key_len();
        (chunk_start..chunk_end)
            .into_par_iter()
            .map(|idx| self.generate_record(idx, pk_len))
            .collect()
    }

    fn generate_record(&self, idx: usize, pk_len: usize) -> DatasetRecord {
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
        rng.set_word_pos(idx as u128 * SEED_WORDS);

        let mut record_seed = [0u8; SEED_BYTES];
        rng.fill_bytes(&mut record_seed);

        let mut public_key = vec![0u8; pk_len];
        self.kem.generate_public_key_into(&record_seed, &mut public_key);
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

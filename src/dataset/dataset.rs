use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use sha3::{Digest, Sha3_256};

/// Binary format version 2: magic + kem_name + count + pk_len + records.
pub const DATASET_MAGIC: &[u8; 8] = b"PQKEYV2\0";
pub const PUBLIC_KEY_ID_LEN: usize = 32;
pub const PUBLIC_KEY_ID_DOMAIN: &[u8] = b"pq-key:v1:";
pub const DEFAULT_DATASET_BATCH_SIZE: usize = 100_000;

/// Canonical KEM name strings used in the dataset and ID derivation.
pub const KEM_ML_KEM_512: &str = "ml-kem-512";
pub const KEM_ML_KEM_768: &str = "ml-kem-768";
pub const KEM_ML_KEM_1024: &str = "ml-kem-1024";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetRecord {
    pub public_key_id: [u8; PUBLIC_KEY_ID_LEN],
    pub public_key: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    /// KEM name, e.g. `"ml-kem-768"`.
    pub kem_name: String,
    pub public_key_len: usize,
    pub records: Vec<DatasetRecord>,
}

/// Derive a 32-byte public-key ID.
///
/// The ID is `SHA3-256(domain || kem_name || public_key_bytes)` where
/// `domain = b"pq-key:v1:"`.
pub fn derive_public_key_id(public_key: &[u8], kem_name: &str) -> [u8; PUBLIC_KEY_ID_LEN] {
    let mut hasher = Sha3_256::new();
    hasher.update(PUBLIC_KEY_ID_DOMAIN);
    hasher.update(kem_name.as_bytes());
    hasher.update(public_key);

    let digest = hasher.finalize();
    let mut id = [0u8; PUBLIC_KEY_ID_LEN];
    id.copy_from_slice(&digest);
    id
}

pub fn save_dataset(path: &Path, dataset: &Dataset) -> io::Result<()> {
    if dataset.records.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dataset must contain at least one record",
        ));
    }
    if dataset.public_key_len > u16::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "public key length does not fit in u16",
        ));
    }
    let kem_bytes = dataset.kem_name.as_bytes();
    if kem_bytes.len() > u8::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "kem_name too long",
        ));
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(DATASET_MAGIC)?;
    writer.write_all(&[kem_bytes.len() as u8])?;
    writer.write_all(kem_bytes)?;
    writer.write_all(&(dataset.records.len() as u32).to_le_bytes())?;
    writer.write_all(&(dataset.public_key_len as u16).to_le_bytes())?;

    for record in &dataset.records {
        if record.public_key.len() != dataset.public_key_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dataset contains mixed public key lengths",
            ));
        }
        writer.write_all(&record.public_key_id)?;
        writer.write_all(&record.public_key)?;
    }

    writer.flush()
}

pub fn load_dataset(path: &Path) -> io::Result<Dataset> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != DATASET_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid dataset magic (expected PQKEYV2)",
        ));
    }

    let mut kem_len_buf = [0u8; 1];
    reader.read_exact(&mut kem_len_buf)?;
    let kem_len = kem_len_buf[0] as usize;
    let mut kem_buf = vec![0u8; kem_len];
    reader.read_exact(&mut kem_buf)?;
    let kem_name = String::from_utf8(kem_buf)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "kem_name is not valid UTF-8"))?;

    let mut count_bytes = [0u8; 4];
    reader.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;

    let mut pk_len_bytes = [0u8; 2];
    reader.read_exact(&mut pk_len_bytes)?;
    let public_key_len = u16::from_le_bytes(pk_len_bytes) as usize;
    if public_key_len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "dataset public key length must be non-zero",
        ));
    }

    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let mut public_key_id = [0u8; PUBLIC_KEY_ID_LEN];
        reader.read_exact(&mut public_key_id)?;

        let mut public_key = vec![0u8; public_key_len];
        reader.read_exact(&mut public_key)?;
        records.push(DatasetRecord {
            public_key_id,
            public_key,
        });
    }

    Ok(Dataset {
        kem_name,
        public_key_len,
        records,
    })
}

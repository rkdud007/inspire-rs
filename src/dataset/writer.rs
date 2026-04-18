use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use super::dataset::{DATASET_MAGIC, DatasetRecord};

/// Streaming writer that writes the header once then accepts records one at a time.
/// Avoids buffering the entire dataset in memory.
pub struct DatasetWriter {
    writer: BufWriter<File>,
    pub public_key_len: usize,
    written: usize,
    expected: usize,
}

impl DatasetWriter {
    pub fn create(
        path: &Path,
        kem_name: &str,
        count: usize,
        public_key_len: usize,
    ) -> io::Result<Self> {
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "count must be > 0",
            ));
        }
        if public_key_len == 0 || public_key_len > u16::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid public key length",
            ));
        }
        let kem_bytes = kem_name.as_bytes();
        if kem_bytes.len() > u8::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "kem_name too long",
            ));
        }

        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
        writer.write_all(DATASET_MAGIC)?;
        writer.write_all(&[kem_bytes.len() as u8])?;
        writer.write_all(kem_bytes)?;
        writer.write_all(&(count as u32).to_le_bytes())?;
        writer.write_all(&(public_key_len as u16).to_le_bytes())?;

        Ok(Self {
            writer,
            public_key_len,
            written: 0,
            expected: count,
        })
    }

    pub fn write_record(&mut self, record: &DatasetRecord) -> io::Result<()> {
        if record.public_key.len() != self.public_key_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "public key length mismatch",
            ));
        }
        self.writer.write_all(&record.public_key_id)?;
        self.writer.write_all(&record.public_key)?;
        self.written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        if self.written != self.expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} records, wrote {}", self.expected, self.written),
            ));
        }
        self.writer.flush()
    }
}

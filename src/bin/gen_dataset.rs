use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use ml_kem::Seed;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use inspire_rs::{DatasetRecord, DatasetWriter, KemVariant, derive_public_key_id};

#[derive(Parser, Debug)]
#[command(version, about = "Generate a static ML-KEM dataset for keyword PIR")]
struct Args {
    /// KEM variant to use.
    #[arg(long, default_value = "ml-kem-768")]
    kem: KemVariant,

    /// Number of public keys to generate.
    #[arg(long)]
    count: usize,

    /// Output dataset file.
    #[arg(long)]
    out: PathBuf,

    /// Deterministic RNG seed for reproducible datasets.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Print the first N key_ids (hex) after writing the dataset.
    /// Omit N to print all records.
    #[arg(long, num_args = 0..=1, default_missing_value = "18446744073709551615")]
    list: Option<usize>,

    /// Print the first N key_ids and public keys (hex) after writing the dataset.
    /// Omit N to print all records.
    #[arg(long, num_args = 0..=1, default_missing_value = "18446744073709551615")]
    list_keys: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.count == 0 {
        return Err("count must be greater than zero".into());
    }

    let kem_name = args.kem.name();

    // Each ML-KEM seed is 64 bytes. ChaCha20 words are 4 bytes each, so each
    // seed occupies 16 words in the stream. set_word_pos lets each thread seek
    // independently, producing the same output as the original sequential loop.
    const SEED_BYTES: usize = 64;
    const CHACHA_WORD_BYTES: usize = 4;
    const SEED_WORDS: u128 = (SEED_BYTES / CHACHA_WORD_BYTES) as u128;

    let log_interval = 1000;
    let start = std::time::Instant::now();
    let completed = Arc::new(AtomicUsize::new(0));

    // Derive public key length from a single sample key (no wasted allocation).
    let sample_key = {
        let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
        let mut seed = Seed::default();
        rng.fill_bytes(&mut seed);
        args.kem.generate_public_key(&seed)
    };
    let public_key_len = sample_key.len();

    // Probe disk throughput to pick a chunk size that keeps the disk busy for
    // ~1 second per chunk — long enough to amortise syscall overhead, short
    // enough that CPU threads don't stall waiting for I/O.
    let chunk_size = {
        const PROBE_BYTES: usize = 32 * 1024 * 1024; // 32 MB probe
        const TARGET_CHUNK_SECS: f64 = 1.0;
        const MIN_CHUNK: usize = 10_000;
        const MAX_CHUNK: usize = 500_000;

        let probe_path = args.out.with_extension("probe");
        let probe_buf = vec![0u8; PROBE_BYTES];
        let t0 = std::time::Instant::now();
        std::fs::write(&probe_path, &probe_buf).ok();
        let _ = std::fs::remove_file(&probe_path);
        let elapsed = t0.elapsed().as_secs_f64().max(1e-6);

        let disk_bytes_per_sec = PROBE_BYTES as f64 / elapsed;
        let record_bytes = public_key_len + inspire_rs::PUBLIC_KEY_ID_LEN;
        let chunk = (disk_bytes_per_sec * TARGET_CHUNK_SECS / record_bytes as f64) as usize;
        let chunk = chunk.clamp(MIN_CHUNK, MAX_CHUNK);
        eprintln!(
            "disk throughput ~{:.0} MB/s  →  chunk size {}",
            disk_bytes_per_sec / 1e6,
            chunk
        );
        chunk
    };

    let mut writer = DatasetWriter::create(&args.out, kem_name, args.count, public_key_len)?;
    let mut seen_ids = HashSet::with_capacity(args.count);
    let mut list_printed = 0usize;
    let list_n = args.list_keys.or(args.list).unwrap_or(0);

    for chunk_start in (0..args.count).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(args.count);

        let chunk: Vec<DatasetRecord> = (chunk_start..chunk_end)
            .into_par_iter()
            .map({
                let completed = Arc::clone(&completed);
                move |idx| {
                    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
                    rng.set_word_pos(idx as u128 * SEED_WORDS);
                    let mut seed = Seed::default();
                    rng.fill_bytes(&mut seed);

                    let public_key = args.kem.generate_public_key(&seed);
                    let public_key_id = derive_public_key_id(&public_key, kem_name);

                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % log_interval == 0 || done == args.count {
                        let elapsed = start.elapsed().as_secs_f64();
                        let rate = done as f64 / elapsed;
                        let remaining = (args.count - done) as f64 / rate;
                        eprintln!(
                            "[{}/{}] {:.0} keys/s  elapsed {:.1}s  eta {:.1}s",
                            done, args.count, rate, elapsed, remaining,
                        );
                    }

                    DatasetRecord {
                        public_key_id,
                        public_key,
                    }
                }
            })
            .collect();

        for (i, record) in chunk.iter().enumerate() {
            if !seen_ids.insert(record.public_key_id) {
                return Err(
                    format!("duplicate public key id at record {}", chunk_start + i).into(),
                );
            }
            writer.write_record(record)?;

            if list_printed < list_n {
                if args.list_keys.is_some() {
                    println!(
                        "key_id={} public_key={}",
                        hex::encode(record.public_key_id),
                        hex::encode(&record.public_key)
                    );
                } else {
                    println!("key_id={}", hex::encode(record.public_key_id));
                }
                list_printed += 1;
            }
        }
    }

    writer.finish()?;
    println!(
        "wrote {} {} records (public key len {} bytes) to {}",
        args.count,
        kem_name,
        public_key_len,
        args.out.display()
    );

    Ok(())
}

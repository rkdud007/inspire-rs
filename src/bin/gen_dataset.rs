use std::path::PathBuf;

use clap::Parser;
use rayon;

#[cfg(feature = "gpu")]
use inspire_rs::PUBLIC_KEY_ID_LEN;
use inspire_rs::dataset::writer::DatasetWriter;
#[cfg(feature = "gpu")]
use inspire_rs::pir::server::KeywordServer;
use inspire_rs::{DatasetGenerator, KemVariant, load_dataset};

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

    /// Optional batch size override for parallel generation.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Number of threads for parallel generation.
    /// Defaults to the number of CPUs available (respects cgroup quota).
    #[arg(long)]
    threads: Option<usize>,

    /// Target cuckoo bucket count for the server that will consume this dataset.
    #[arg(long)]
    buckets: Option<usize>,

    /// Target first-dimension size override for the server that will consume this dataset.
    #[arg(long)]
    dim0: Option<usize>,

    /// Print the first N key_ids (hex) after writing the dataset.
    /// Omit N to print all records.
    #[arg(long, num_args = 0..=1, default_missing_value = "18446744073709551615")]
    list: Option<usize>,

    /// Print the first N key_ids and public keys (hex) after writing the dataset.
    /// Omit N to print all records.
    #[arg(long, num_args = 0..=1, default_missing_value = "18446744073709551615")]
    list_keys: Option<usize>,
}

#[cfg(feature = "gpu")]
fn format_gib(bytes: u128) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let available_cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let num_threads = args.threads.unwrap_or(available_cpus);

    let pk_len = args.kem.public_key_len();
    #[cfg(feature = "gpu")]
    {
        let memory_check = KeywordServer::<u16>::check_setup_memory_for_shape(
            args.count,
            PUBLIC_KEY_ID_LEN,
            pk_len,
            args.buckets,
            args.dim0,
        )?;
        eprintln!(
            "server GPU preflight: require {} ({} bytes), available {} ({} bytes)",
            format_gib(memory_check.estimate.peak_bytes),
            memory_check.estimate.peak_bytes,
            format_gib(memory_check.available_bytes as u128),
            memory_check.available_bytes,
        );
        eprintln!(
            "target server layout: buckets {}, dim0 {}, db_rows {}, db_cols {}, instances {}",
            memory_check.estimate.num_items,
            memory_check.estimate.dim0,
            memory_check.estimate.db_rows,
            memory_check.estimate.db_cols,
            memory_check.estimate.instances,
        );
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    eprintln!("using {num_threads} threads ({available_cpus} CPUs available)");

    let mut generator = DatasetGenerator::new(args.kem, args.count).with_seed(args.seed);
    if let Some(batch_size) = args.batch_size {
        generator = generator.with_batch_size(batch_size);
    }

    let mut writer = DatasetWriter::create(&args.out, args.kem.name(), args.count, pk_len)?;

    generator.stream_to_writer(&mut writer, |done, total| {
        let pct = done * 100 / total;
        eprint!("\r  generating... {done}/{total} ({pct}%)   ");
    })?;
    writer.finish()?;
    eprintln!();

    println!(
        "wrote {} {} records (public key len {} bytes) to {}",
        args.count,
        args.kem.name(),
        pk_len,
        args.out.display()
    );

    // --list / --list-keys: reload just the first N records to print them
    let list_n = args.list_keys.or(args.list).unwrap_or(0);
    if list_n > 0 {
        let dataset = load_dataset(&args.out)?;
        for record in dataset.records.iter().take(list_n) {
            if args.list_keys.is_some() {
                println!(
                    "key_id={} public_key={}",
                    hex::encode(record.public_key_id),
                    hex::encode(&record.public_key)
                );
            } else {
                println!("key_id={}", hex::encode(record.public_key_id));
            }
        }
    }

    Ok(())
}

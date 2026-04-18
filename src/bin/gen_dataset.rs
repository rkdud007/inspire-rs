use std::path::PathBuf;

use clap::Parser;

use inspire_rs::{DatasetGenerator, KemVariant, save_dataset};

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

    let mut generator = DatasetGenerator::new(args.kem, args.count).with_seed(args.seed);
    if let Some(batch_size) = args.batch_size {
        generator = generator.with_batch_size(batch_size);
    }

    let dataset = generator.generate()?;
    save_dataset(&args.out, &dataset)?;

    let list_n = args
        .list_keys
        .or(args.list)
        .unwrap_or(0)
        .min(dataset.records.len());

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

    println!(
        "wrote {} {} records (public key len {} bytes) to {}",
        dataset.records.len(),
        dataset.kem_name,
        dataset.public_key_len,
        args.out.display()
    );

    Ok(())
}

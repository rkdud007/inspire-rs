# inspire-rs

`inspire-rs` is a Rust library and demo app for keyword PIR over a static database of ML-KEM public keys.

The library exposes:
- batched dataset generation
- a keyword-PIR server wrapper
- a keyword-PIR client wrapper
- transport-neutral request/response handling, with the demo binaries using TCP framing

## Library API

### Generate a dataset

```rust
use inspire_rs::{DatasetGenerator, KemVariant, save_dataset};

let dataset = DatasetGenerator::new(KemVariant::MlKem768, 10_000)
    .with_seed(7)
    .generate()?;
save_dataset("keys.bin".as_ref(), &dataset)?;
```

### Run a server and client

```rust
use inspire_rs::pir::client::{KeywordClient, KeywordClientConfig};
use inspire_rs::setup_keyword_server;

let server = setup_keyword_server(&dataset, None, None)?;
let public_params = server.public_params();

let client = KeywordClient::setup_from_public_params(&public_params, KeywordClientConfig::default())
    .map_err(std::io::Error::other)?;
let query = client.query(&query_id).map_err(std::io::Error::other)?;

let response = server.respond(query.payload());
let value = client.extract(
    &query_id,
    &query,
    &response,
);
```

## Demo Commands

### 1. Generate a dataset

Generate 10,000 deterministic `ml-kem-768` public keys:

```bash
cargo run --bin gen_dataset -- --count 10000 --seed 7 --out ./keys.bin
```

Print the first few IDs while generating:

```bash
cargo run --bin gen_dataset -- --count 10000 --seed 7 --out ./keys.bin --list 10
```

Print IDs and public keys:

```bash
cargo run --bin gen_dataset -- --count 10000 --seed 7 --out ./keys.bin --list-keys 10
```

### 2. Start the server

The server is GPU-only and requires the `gpu` feature plus a working CUDA toolchain/runtime.

```bash
cargo run --features gpu --bin server -- --dataset ./keys.bin --listen 127.0.0.1:8082
```

Optional overrides:

```bash
cargo run --features gpu --bin server -- \
  --dataset ./keys.bin \
  --listen 127.0.0.1:8082 \
  --buckets 262144 \
  --dim0 2048
```

### 3. Query from the client

Query by public-key ID:

```bash
cargo run --bin client -- --server 127.0.0.1:8082 --id <32-byte-hex-id>
```

## End-to-End Test

There are two ignored ML-KEM end-to-end tests:
- [tests/e2e_mlkem.rs](tests/e2e_mlkem.rs): typed in-memory request/response, no TCP or HTTP
- [tests/e2e_tcp_mlkem.rs](tests/e2e_tcp_mlkem.rs): TCP adapter roundtrip on top of the same library abstractions

Run the in-memory typed test with:

```bash
cargo test --features gpu --test e2e_mlkem -- --ignored --nocapture
```

Run the TCP adapter test with:

```bash
cargo test --features gpu --test e2e_tcp_mlkem -- --ignored --nocapture
```

## Notes

- `client` is CPU-runnable.
- `server` requires `--features gpu`.
- Dataset files use the `PQKEYV2` binary format.

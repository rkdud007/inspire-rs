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
use inspire_rs::pir::server::KeywordServer;

let server = KeywordServer::from_dataset(&dataset, None, None)?;
let handshake = server.handshake();

let client = KeywordClient::from_handshake(&handshake, KeywordClientConfig::default())
    .map_err(std::io::Error::other)?;
let request = client.build_request(&query_id);

// Send `request.payload()` over any transport you want.
let response_bytes = server.handle_serialized_request(request.payload().to_vec())?;

let public_key = client.find_public_key_in_serialized_response(
    &query_id,
    &request,
    &response_bytes,
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

Query by public key:

```bash
cargo run --bin client -- --server 127.0.0.1:8082 --public-key <hex-public-key>
```

Query by file:

```bash
cargo run --bin client -- --server 127.0.0.1:8082 --public-key-file ./public_key.bin
```

## End-to-End Test

The integration test in [tests/e2e.rs](tests/e2e.rs) does not spawn binaries. It:
- generates a 10,000-record dataset through the library
- saves and reloads it from a temp directory
- constructs `KeywordServer` and `KeywordClient` directly
- performs a real TCP handshake/query/response roundtrip

Run it with:

```bash
cargo test --features gpu --test e2e -- --ignored --nocapture
```

## Notes

- `client` is CPU-runnable.
- `server` requires `--features gpu`.
- Dataset files use the `PQKEYV2` binary format.

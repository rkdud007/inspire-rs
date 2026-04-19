# Performance Log

## 2026-04-19 07:26:29 UTC - 5M end-to-end run

### Command set

Dataset generation:

```bash
cargo run --features gpu --bin gen_dataset -- \
  --count 5000000 \
  --out /root/inspire-rs/dataset-5m.bin \
  --buckets 8388608 \
  --dim0 65536
```

Server startup:

```bash
cargo run --features gpu --bin server -- \
  --dataset /root/inspire-rs/dataset-5m.bin \
  --buckets 8388608 \
  --dim0 65536
```

Client query:

```bash
cargo run --features gpu --bin client -- \
  --server 127.0.0.1:8082 \
  --id 2e2a85f85dbba83c05312a8a3eb91842998ee5e142221b793b5b350f21ed59a6
```

### Dataset artifact

- Records: `5,000,000`
- KEM: `ml-kem-768`
- Public key length: `1184` bytes
- Entry size served by keyword PIR: `1216` bytes
- Output file: `/root/inspire-rs/dataset-5m.bin`
- Output size: `6080000025` bytes (`5.7G` as reported by `ls -lh`)
- Batch size: `100000`
- Threads used for generation: `22`

### Server shape

- Buckets: `8,388,608`
- `dim0`: `65,536`
- `db_rows`: `65,536`
- `db_cols`: `131,072`
- Instances: `64`

### GPU memory preflight

- Estimated startup requirement: `32,205,963,264` bytes (`29.99 GiB`)
- Available GPU memory at startup check: `84,465,090,560` bytes (`78.66 GiB`)

### Observed timings

- Dataset generation: completed successfully
- Server initialization: `264s`
  - Derived from init log timestamps: `83058s -> 83322s`
- Client end-to-end latency: `1521.8ms`
- Server-side query time: `1472.4ms`

### Notes

- The 5M run succeeded only after moving from the default shape to `--buckets 8388608 --dim0 65536`.
- The default 5M shape attempted earlier (`dim0=2048`) failed the new startup RAM gate with an estimated requirement of `669,207,494,656` bytes versus about `84,465,090,560` bytes available.
- The client query used a real key id extracted from the generated dataset and returned a matching public key record.

use rand::RngExt;
use rayon::prelude::*;
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};
use siphasher::sip::SipHasher;
use std::fs::File;
use std::hash::Hasher;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

/// Max entry size for stack-allocated entries (key 20 + value 40 + padding 4 = 64)
const MAX_ENTRY_SIZE: usize = 64;

/// Deterministic address from index: SHAKE-256(i as u64 LE) → first 20 bytes.
pub fn address_from_index(i: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(&(i as u64).to_le_bytes());
    let mut reader = hasher.finalize_xof();
    let mut buf = vec![0u8; 20];
    XofReader::read(&mut reader, &mut buf);
    buf
}

/// Trivial address: index encoded as 20-byte big-endian (for identity hash testing).
pub fn trivial_address_from_index(i: usize) -> Vec<u8> {
    let mut buf = vec![0u8; 20];
    buf[12..20].copy_from_slice(&(i as u64).to_be_bytes());
    buf
}

/// Initial value for account i: 32B balance = i as u128 BE padded, 8B nonce = 0.
pub fn initial_value(i: usize) -> Vec<u8> {
    let mut value = vec![0u8; 40];
    let balance_bytes = (i as u128).to_be_bytes(); // 16 bytes
    value[16..32].copy_from_slice(&balance_bytes);
    // nonce = 0 (already zeroed)
    value
}

/// Convert 64 bytes to 32 u16 big-endian elements.
pub fn bytes_to_u16_be(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks_exact(2)
        .map(|chunk| (chunk[0] as u16) << 8 | (chunk[1] as u16))
        .collect()
}

/// Convert u16 big-endian elements back to bytes.
pub fn u16_be_to_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &v in values {
        bytes.push((v >> 8) as u8);
        bytes.push(v as u8);
    }
    bytes
}

// ============================================================================
// Canary entry for freshness verification
// ============================================================================

/// Canary address: "InspirePIR" encoded in the last 10 bytes.
pub const CANARY_ADDRESS: [u8; 20] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x6e, 0x73, 0x70, 0x69, 0x72,
    0x65, 0x50, 0x49, 0x52,
];

/// Nonzero padding marks canary entries (real entries always have zero padding).
pub const CANARY_PADDING: [u8; 4] = [0xCA, 0xFE, 0xBA, 0xBE];

/// Build a canary value: balance = block_number, nonce = block_number.
pub fn canary_value(block_number: u64) -> Vec<u8> {
    let mut value = vec![0u8; 40];
    value[16..32].copy_from_slice(&(block_number as u128).to_be_bytes());
    value[32..40].copy_from_slice(&block_number.to_be_bytes());
    value
}

// ============================================================================
// Cuckoo hash parameters
// ============================================================================

pub const DETERMINISTIC_SEED: [u8; 16] = [
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
];

/// Trivial seed: when used, hash becomes identity (key bytes → bucket index directly).
/// account i → bucket i (hash 0), bucket i + num_buckets/2 (hash 1). Zero collisions.
pub const TRIVIAL_SEED: [u8; 16] = [0u8; 16];

#[derive(Clone, Debug)]
pub struct CuckooParams {
    pub num_buckets: usize,
    pub key_size: usize,      // 20
    pub value_size: usize,    // 40
    pub padding_size: usize,  // 4 (to reach 64B)
    pub num_hashes: usize,    // 2
    pub max_evictions: usize, // 500
    pub seed: [u8; 16],
}

impl CuckooParams {
    pub fn new(
        num_buckets: usize,
        key_size: usize,
        value_size: usize,
        padding_size: usize,
        seed: [u8; 16],
    ) -> Self {
        CuckooParams {
            num_buckets,
            key_size,
            value_size,
            padding_size,
            num_hashes: 2,
            max_evictions: 500,
            seed,
        }
    }

    /// Total entry size: key + value + padding
    pub fn entry_size(&self) -> usize {
        self.key_size + self.value_size + self.padding_size
    }
}

// ============================================================================
// SHAKE-256 based hash function
// ============================================================================

#[derive(Clone, Debug)]
pub struct CuckooHash {
    pub seed: [u8; 16],
    pub num_buckets: usize,
    pub num_hashes: usize,
}

impl CuckooHash {
    pub fn new(params: &CuckooParams) -> Self {
        CuckooHash {
            seed: params.seed,
            num_buckets: params.num_buckets,
            num_hashes: params.num_hashes,
        }
    }

    pub fn new_from_seed(seed: [u8; 16], num_hashes: usize, num_buckets: usize) -> Self {
        CuckooHash {
            seed,
            num_buckets,
            num_hashes,
        }
    }

    /// Hash a key to a bucket index.
    /// With TRIVIAL_SEED: identity hash (key → index directly, no collisions).
    /// Otherwise: SipHash-2-4 with keys derived from seed + hash_idx.
    pub fn hash(&self, hash_idx: u8, key: &[u8]) -> usize {
        if self.seed == TRIVIAL_SEED {
            // Identity: interpret last 8 bytes of key as u64 LE
            let mut buf = [0u8; 8];
            let start = if key.len() >= 8 { key.len() - 8 } else { 0 };
            buf[..key.len().min(8)].copy_from_slice(&key[start..]);
            let idx = u64::from_be_bytes(buf) as usize;
            let half = self.num_buckets / 2;
            match hash_idx {
                0 => idx % half,
                _ => half + (idx % half),
            }
        } else {
            let k0 = u64::from_le_bytes(self.seed[..8].try_into().unwrap()) ^ (hash_idx as u64);
            let k1 = u64::from_le_bytes(self.seed[8..16].try_into().unwrap())
                ^ ((hash_idx as u64) << 32);
            let mut hasher = SipHasher::new_with_keys(k0, k1);
            hasher.write(key);
            let val = hasher.finish();
            (val % self.num_buckets as u64) as usize
        }
    }

    /// Return all possible bucket positions for a key.
    pub fn all_positions(&self, key: &[u8]) -> Vec<usize> {
        (0..self.num_hashes as u8)
            .map(|i| self.hash(i, key))
            .collect()
    }

    /// Fast 2-hash version (avoids Vec allocation in hot path).
    #[inline]
    pub fn positions_2(&self, key: &[u8]) -> [usize; 2] {
        [self.hash(0, key), self.hash(1, key)]
    }
}

// ============================================================================
// Cuckoo table
// ============================================================================

pub struct CuckooTable {
    pub params: CuckooParams,
    pub hasher: CuckooHash,
    /// Flat buffer: bucket i is at data[i * entry_size .. (i+1) * entry_size]
    pub data: Vec<u8>,
    /// Which buckets are occupied
    pub occupied: Vec<bool>,
    pub stash: Vec<Vec<u8>>,
}

impl CuckooTable {
    pub fn new(params: CuckooParams) -> Self {
        let num_buckets = params.num_buckets;
        let entry_size = params.entry_size();
        let hasher = CuckooHash::new(&params);
        CuckooTable {
            params,
            hasher,
            data: vec![0u8; num_buckets * entry_size],
            occupied: vec![false; num_buckets],
            stash: Vec::new(),
        }
    }

    #[inline]
    fn bucket_slice(&self, pos: usize) -> &[u8] {
        let es = self.params.entry_size();
        &self.data[pos * es..(pos + 1) * es]
    }

    #[inline]
    fn bucket_slice_mut(&mut self, pos: usize) -> &mut [u8] {
        let es = self.params.entry_size();
        &mut self.data[pos * es..(pos + 1) * es]
    }

    #[inline]
    fn write_entry_to_bucket(&mut self, pos: usize, key: &[u8], value: &[u8]) {
        let ks = self.params.key_size;
        let vs = self.params.value_size;
        let slot = self.bucket_slice_mut(pos);
        slot[..ks].copy_from_slice(key);
        slot[ks..ks + vs].copy_from_slice(value);
        // padding bytes stay zero (initialized in new())
        self.occupied[pos] = true;
    }

    #[inline]
    fn write_raw_to_bucket(&mut self, pos: usize, entry: &[u8]) {
        self.bucket_slice_mut(pos).copy_from_slice(entry);
        self.occupied[pos] = true;
    }

    /// Insert a key-value pair. Returns list of (bucket_idx, entry_bytes) for modified buckets.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Vec<(usize, Vec<u8>)> {
        let ks = self.params.key_size;
        let es = self.params.entry_size();
        let mut modified = Vec::new();
        let mut rng = rand::rng();

        // Build entry on stack
        let mut entry = vec![0u8; es];
        entry[..ks].copy_from_slice(key);
        entry[ks..ks + value.len()].copy_from_slice(value);

        // Try each hash position for an empty bucket
        let positions = self.hasher.positions_2(key);
        for &pos in &positions {
            if !self.occupied[pos] {
                self.write_raw_to_bucket(pos, &entry);
                modified.push((pos, entry));
                return modified;
            }
        }

        // All positions occupied — evict
        for _ in 0..self.params.max_evictions {
            let hash_idx = rng.random_range(0..self.params.num_hashes) as u8;
            let pos = self.hasher.hash(hash_idx, &entry[..ks]);

            // Swap: read evicted, write entry
            let mut evicted = vec![0u8; es];
            evicted.copy_from_slice(self.bucket_slice(pos));
            self.write_raw_to_bucket(pos, &entry);
            modified.push((pos, entry));
            entry = evicted;

            // Try to place evicted entry
            let positions = self.hasher.positions_2(&entry[..ks]);
            for &p in &positions {
                if !self.occupied[p] {
                    self.write_raw_to_bucket(p, &entry);
                    modified.push((p, entry));
                    return modified;
                }
            }
        }

        // Stash overflow
        self.stash.push(entry);
        modified
    }

    /// Update existing key or insert new.
    pub fn upsert(&mut self, key: &[u8], value: &[u8]) -> Vec<(usize, Vec<u8>)> {
        let ks = self.params.key_size;
        let es = self.params.entry_size();
        // Check if key exists in buckets
        let positions = self.hasher.positions_2(key);
        for &pos in &positions {
            if self.occupied[pos] && &self.bucket_slice(pos)[..ks] == key {
                self.write_entry_to_bucket(pos, key, value);
                return vec![(pos, self.bucket_slice(pos).to_vec())];
            }
        }
        // Check stash
        for entry in &mut self.stash {
            if &entry[..ks] == key {
                let mut new_entry = vec![0u8; es];
                new_entry[..ks].copy_from_slice(key);
                new_entry[ks..ks + value.len()].copy_from_slice(value);
                *entry = new_entry;
                return vec![];
            }
        }
        // New key
        self.insert(key, value)
    }

    /// Delete a key. Returns the bucket index if found in main table.
    pub fn delete(&mut self, key: &[u8]) -> Option<usize> {
        let ks = self.params.key_size;
        let positions = self.hasher.positions_2(key);
        for &pos in &positions {
            if self.occupied[pos] && &self.bucket_slice(pos)[..ks] == key {
                self.occupied[pos] = false;
                // Zero out the bucket data
                let es = self.params.entry_size();
                self.data[pos * es..(pos + 1) * es].fill(0);
                return Some(pos);
            }
        }
        if let Some(idx) = self.stash.iter().position(|e| &e[..ks] == key) {
            self.stash.remove(idx);
        }
        None
    }

    /// Lookup a key. Returns value bytes (without key/padding) if found.
    pub fn lookup(&self, key: &[u8]) -> Option<&[u8]> {
        let ks = self.params.key_size;
        let vs = self.params.value_size;
        let positions = self.hasher.positions_2(key);
        for &pos in &positions {
            if self.occupied[pos] && &self.bucket_slice(pos)[..ks] == key {
                return Some(&self.bucket_slice(pos)[ks..ks + vs]);
            }
        }
        for entry in &self.stash {
            if &entry[..ks] == key {
                return Some(&entry[ks..ks + vs]);
            }
        }
        None
    }

    /// Bulk-build cuckoo table from deterministic accounts, with parallel address generation.
    pub fn build_accounts_parallel(&mut self, num_accounts: usize) {
        self.build_accounts_parallel_with(num_accounts, false);
    }

    /// Build entry directly into a stack buffer. Returns entry size.
    #[inline]
    fn make_entry_inline(buf: &mut [u8; MAX_ENTRY_SIZE], i: usize, ks: usize, trivial: bool) {
        // Zero the buffer
        *buf = [0u8; MAX_ENTRY_SIZE];
        // Write key (20 bytes)
        if trivial {
            buf[12..20].copy_from_slice(&(i as u64).to_be_bytes());
        } else {
            let mut hasher = Shake256::default();
            hasher.update(&(i as u64).to_le_bytes());
            let mut reader = hasher.finalize_xof();
            XofReader::read(&mut reader, &mut buf[..ks]);
        }
        // Write value: balance = i as u128 BE at bytes [ks+16..ks+32], nonce = 0
        let balance = (i as u128).to_be_bytes();
        buf[ks + 16..ks + 32].copy_from_slice(&balance);
        // padding stays zero
    }

    /// Fast bulk insert: no heap allocation in the hot path.
    #[inline]
    fn insert_fast(&mut self, entry: &[u8; MAX_ENTRY_SIZE], rng: &mut fastrand::Rng) {
        let ks = self.params.key_size;
        let es = self.params.entry_size();

        // Try each hash position for an empty bucket
        let positions = self.hasher.positions_2(&entry[..ks]);
        for &pos in &positions {
            if !self.occupied[pos] {
                self.bucket_slice_mut(pos)[..es].copy_from_slice(&entry[..es]);
                self.occupied[pos] = true;
                return;
            }
        }

        // All positions occupied — evict
        let mut current = *entry;
        for _ in 0..self.params.max_evictions {
            let hash_idx = (rng.u8(..)) % self.params.num_hashes as u8;
            let pos = self.hasher.hash(hash_idx, &current[..ks]);

            // Swap: read evicted into current, write current to bucket
            let slot = self.bucket_slice_mut(pos);
            let mut evicted = [0u8; MAX_ENTRY_SIZE];
            evicted[..es].copy_from_slice(&slot[..es]);
            slot[..es].copy_from_slice(&current[..es]);
            self.occupied[pos] = true;
            current = evicted;

            // Try to place evicted entry
            let positions = self.hasher.positions_2(&current[..ks]);
            for &p in &positions {
                if !self.occupied[p] {
                    self.bucket_slice_mut(p)[..es].copy_from_slice(&current[..es]);
                    self.occupied[p] = true;
                    return;
                }
            }
        }

        // Stash overflow
        self.stash.push(current[..es].to_vec());
    }

    pub fn build_accounts_parallel_with(&mut self, num_accounts: usize, trivial: bool) {
        let ks = self.params.key_size;
        let t0 = Instant::now();
        let mut rng = fastrand::Rng::new();

        // Generate all entries in parallel (flat buffer, no Vec<Vec>)
        let entries_per_chunk = 4_000_000;
        let num_chunks = (num_accounts + entries_per_chunk - 1) / entries_per_chunk;

        let mut last_pct_bucket: u32 = 0;
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * entries_per_chunk;
            let end = (start + entries_per_chunk).min(num_accounts);
            let count = end - start;

            let mut flat_entries = vec![[0u8; MAX_ENTRY_SIZE]; count];
            flat_entries
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, buf)| {
                    Self::make_entry_inline(buf, start + j, ks, trivial);
                });

            for entry in &flat_entries {
                self.insert_fast(entry, &mut rng);
            }

            let pct = end as f64 / num_accounts as f64 * 100.0;
            let pct_bucket = (pct / 10.0) as u32;
            if pct_bucket > last_pct_bucket || chunk_idx == num_chunks - 1 {
                last_pct_bucket = pct_bucket;
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = end as f64 / elapsed;
                let eta = (num_accounts - end) as f64 / rate;
                eprint!(
                    "\r  Inserting accounts... {:.0}% ({:.0}s remaining)    ",
                    pct, eta
                );
            }
        }
        eprintln!("\r  Inserting accounts... done.                      ");
    }

    /// Upsert an entry and set nonzero padding bytes (for canary entries).
    pub fn upsert_canary(
        &mut self,
        key: &[u8],
        value: &[u8],
        padding: &[u8],
    ) -> Vec<(usize, Vec<u8>)> {
        let ks = self.params.key_size;
        let vs = self.params.value_size;
        let ps = self.params.padding_size;
        let mut modified = self.upsert(key, value);

        // Find the canary's bucket and write padding
        let positions = self.hasher.positions_2(key);
        for &pos in &positions {
            if self.occupied[pos] && &self.bucket_slice(pos)[..ks] == key {
                let pad_start = ks + vs;
                self.bucket_slice_mut(pos)[pad_start..pad_start + ps]
                    .copy_from_slice(&padding[..ps]);
                // Fix returned entry to include the padding
                for (idx, entry) in &mut modified {
                    if *idx == pos {
                        entry[pad_start..pad_start + ps].copy_from_slice(&padding[..ps]);
                    }
                }
                break;
            }
        }
        modified
    }

    // ========================================================================
    // CSV loading (HuggingFace eth snapshot format)
    // ========================================================================

    /// Build cuckoo table from a HuggingFace-format CSV.
    /// Supports both column orders:
    ///   - `address,nonce,balance_wei`
    ///   - `balance_wei,address,nonce`
    /// First line should contain `block=NUMBER` metadata.
    /// Returns (block_number, num_accounts_inserted).
    pub fn build_from_csv(&mut self, path: &Path) -> io::Result<(u64, usize)> {
        let reader = BufReader::with_capacity(64 * 1024 * 1024, File::open(path)?);
        let ks = self.params.key_size;
        let t0 = Instant::now();
        let mut rng = fastrand::Rng::new();
        let mut block_number = 0u64;
        let mut num_accounts = 0usize;

        // Column indices (detected from header)
        let mut col_addr: usize = 0;
        let mut col_nonce: usize = 1;
        let mut col_balance: usize = 2;
        let mut header_parsed = false;

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Extract block number from metadata line
            if let Some(pos) = trimmed.find("block=") {
                let after = &trimmed[pos + 6..];
                let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = num_str.parse::<u64>() {
                    block_number = n;
                }
                continue;
            }

            // Detect column order from header line
            if !header_parsed && (trimmed.contains("balance") || trimmed.contains("address")) {
                let cols: Vec<&str> = trimmed.split(',').collect();
                for (i, col) in cols.iter().enumerate() {
                    let c = col.trim().to_lowercase();
                    if c == "address" {
                        col_addr = i;
                    } else if c == "nonce" {
                        col_nonce = i;
                    } else if c.contains("balance") {
                        col_balance = i;
                    }
                }
                header_parsed = true;
                continue;
            }

            // Parse data line
            let parts: Vec<&str> = trimmed.split(',').collect();
            let max_col = *[col_addr, col_nonce, col_balance].iter().max().unwrap();
            if parts.len() <= max_col {
                continue;
            }

            let address_str = parts[col_addr].trim();
            let nonce_str = parts[col_nonce].trim();
            let balance_str = parts[col_balance].trim();

            // Parse address → 20 bytes
            let addr_hex = address_str.strip_prefix("0x").unwrap_or(address_str);
            if addr_hex.len() != 40 {
                continue;
            }
            let address = match hex::decode(addr_hex) {
                Ok(a) => a,
                Err(_) => continue,
            };

            // Parse balance → u128 → 16 bytes BE
            let balance: u128 = balance_str.parse().unwrap_or(0);
            let balance_bytes = balance.to_be_bytes();

            // Parse nonce → u64
            let nonce: u64 = nonce_str.parse().unwrap_or(0);

            // Build entry: [20B address][16B zero][16B balance BE][8B nonce BE][4B pad]
            let mut entry = [0u8; MAX_ENTRY_SIZE];
            entry[..ks].copy_from_slice(&address);
            entry[ks + 16..ks + 32].copy_from_slice(&balance_bytes);
            entry[ks + 32..ks + 40].copy_from_slice(&nonce.to_be_bytes());

            self.insert_fast(&entry, &mut rng);
            num_accounts += 1;

            if num_accounts % 4_000_000 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = num_accounts as f64 / elapsed;
                eprint!(
                    "\r  Loading CSV... {}M accounts ({:.0}k/s)    ",
                    num_accounts / 1_000_000,
                    rate / 1000.0
                );
            }
        }

        eprintln!(
            "\r  CSV loaded: {} accounts from block #{} ({:.1}s)              ",
            num_accounts,
            block_number,
            t0.elapsed().as_secs_f64()
        );

        Ok((block_number, num_accounts))
    }

    /// Export all occupied entries to CSV format: `address,nonce,balance_wei`.
    /// First line is `# block=BLOCK_NUMBER`, second line is the header.
    pub fn export_csv(&self, path: &Path, block_number: u64) -> io::Result<usize> {
        use std::io::Write;
        let ks = self.params.key_size;
        let mut writer = std::io::BufWriter::with_capacity(64 * 1024 * 1024, File::create(path)?);
        writeln!(writer, "# block={}", block_number)?;
        writeln!(writer, "address,nonce,balance_wei")?;

        let t0 = Instant::now();
        let mut count = 0usize;
        for i in 0..self.params.num_buckets {
            if !self.occupied[i] {
                continue;
            }
            let entry = self.bucket_slice(i);
            let address = &entry[..ks];
            // value: [16B zero-pad][16B balance BE][8B nonce BE]
            let balance_bytes = &entry[ks + 16..ks + 32];
            let nonce_bytes = &entry[ks + 32..ks + 40];
            let balance = u128::from_be_bytes(balance_bytes.try_into().unwrap());
            let nonce = u64::from_be_bytes(nonce_bytes.try_into().unwrap());
            writeln!(writer, "0x{},{},{}", hex::encode(address), nonce, balance)?;
            count += 1;
            if count % 4_000_000 == 0 {
                eprint!("\r  Exporting CSV... {}M accounts    ", count / 1_000_000);
            }
        }
        writer.flush()?;
        eprintln!(
            "\r  CSV exported: {} accounts at block #{} ({:.1}s)              ",
            count,
            block_number,
            t0.elapsed().as_secs_f64()
        );
        Ok(count)
    }

    /// Convert cuckoo table to column-major raw_db Vec<u16> (parallel).
    /// Iterates by item-within-row (outer, parallel) × row (inner), so each
    /// task writes to item_size_elements consecutive columns (~2MB, fits L2).
    pub fn to_raw_db(
        &self,
        db_rows: usize,
        db_cols: usize,
        per: usize,
        item_size_elements: usize,
    ) -> Vec<u16> {
        let total = db_rows * db_cols;
        let es = self.params.entry_size();

        let mut col_major = vec![0u16; total];
        let cm_ptr = col_major.as_mut_ptr() as usize;
        let data_ptr = self.data.as_ptr() as usize;
        let occ_ptr = self.occupied.as_ptr() as usize;

        (0..per).into_par_iter().for_each(|item_in_row| {
            let base_col = item_in_row * item_size_elements;
            let dst = cm_ptr as *mut u16;
            let data = data_ptr as *const u8;
            let occ = occ_ptr as *const bool;

            for row in 0..db_rows {
                let bucket_idx = row * per + item_in_row;
                unsafe {
                    if bucket_idx >= self.params.num_buckets || !*occ.add(bucket_idx) {
                        continue;
                    }
                    let entry = data.add(bucket_idx * es);
                    for j in 0..item_size_elements {
                        let byte_off = j * 2;
                        let hi = *entry.add(byte_off) as u16;
                        let lo = *entry.add(byte_off + 1) as u16;
                        *dst.add((base_col + j) * db_rows + row) = (hi << 8) | lo;
                    }
                }
            }
        });

        col_major
    }

    /// Write a single bucket's entry into raw_db (column-major).
    pub fn write_bucket_to_raw_db(
        raw_db: &mut [u16],
        bucket_idx: usize,
        entry_bytes: &[u8],
        db_rows: usize,
        db_cols: usize,
        per: usize,
        item_size_elements: usize,
    ) {
        let u16_values = bytes_to_u16_be(entry_bytes);
        let row = bucket_idx / per;
        let base_col = (bucket_idx % per) * item_size_elements;
        for j in 0..item_size_elements {
            let col = base_col + j;
            if col < db_cols && row < db_rows {
                raw_db[col * db_rows + row] = u16_values[j];
            }
        }
    }

    /// Zero out a bucket in raw_db.
    pub fn zero_bucket_in_raw_db(
        raw_db: &mut [u16],
        bucket_idx: usize,
        db_rows: usize,
        db_cols: usize,
        per: usize,
        item_size_elements: usize,
    ) {
        let row = bucket_idx / per;
        let base_col = (bucket_idx % per) * item_size_elements;
        for j in 0..item_size_elements {
            let col = base_col + j;
            if col < db_cols && row < db_rows {
                raw_db[col * db_rows + row] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_from_index_deterministic() {
        let a1 = address_from_index(0);
        let a2 = address_from_index(0);
        assert_eq!(a1, a2);
        assert_eq!(a1.len(), 20);

        let a3 = address_from_index(1);
        assert_ne!(a1, a3);
    }

    #[test]
    fn test_bytes_u16_roundtrip() {
        let bytes: Vec<u8> = (0..64).collect();
        let u16s = bytes_to_u16_be(&bytes);
        assert_eq!(u16s.len(), 32);
        let back = u16_be_to_bytes(&u16s);
        assert_eq!(back, bytes);
    }

    #[test]
    fn test_cuckoo_basic() {
        let params = CuckooParams::new(200, 20, 40, 4, DETERMINISTIC_SEED);
        let mut table = CuckooTable::new(params);

        let key = address_from_index(0);
        let value = initial_value(0);
        table.insert(&key, &value);

        let result = table.lookup(&key);
        assert_eq!(result, Some(value.as_slice()));
    }

    #[test]
    fn test_cuckoo_many_inserts() {
        let n = 10_000;
        let params = CuckooParams::new(n * 2, 20, 40, 4, DETERMINISTIC_SEED);
        let mut table = CuckooTable::new(params);

        let mut keys = Vec::new();
        for i in 0..n {
            let key = address_from_index(i);
            let value = initial_value(i);
            table.insert(&key, &value);
            keys.push(key);
        }

        // Verify all lookups
        for (i, key) in keys.iter().enumerate() {
            let val = table.lookup(key).expect(&format!("key {} not found", i));
            let expected = initial_value(i);
            assert_eq!(val, expected.as_slice(), "mismatch at index {}", i);
        }

        assert!(
            table.stash.len() < 10,
            "stash too large: {}",
            table.stash.len()
        );
    }

    #[test]
    fn test_cuckoo_upsert() {
        let params = CuckooParams::new(200, 20, 40, 4, DETERMINISTIC_SEED);
        let mut table = CuckooTable::new(params);

        let key = address_from_index(0);
        let value1 = initial_value(0);
        table.insert(&key, &value1);
        assert_eq!(table.lookup(&key), Some(value1.as_slice()));

        let mut value2 = vec![0xFFu8; 40];
        value2[32..40].copy_from_slice(&42u64.to_be_bytes());
        table.upsert(&key, &value2);
        assert_eq!(table.lookup(&key), Some(value2.as_slice()));
    }

    #[test]
    fn test_cuckoo_delete() {
        let params = CuckooParams::new(200, 20, 40, 4, DETERMINISTIC_SEED);
        let mut table = CuckooTable::new(params);

        let key = address_from_index(0);
        let value = initial_value(0);
        table.insert(&key, &value);
        assert!(table.lookup(&key).is_some());

        table.delete(&key);
        assert!(table.lookup(&key).is_none());
    }

    #[test]
    fn test_cuckoo_to_raw_db() {
        let n = 100;
        let num_buckets = n * 2;
        let params = CuckooParams::new(num_buckets, 20, 40, 4, DETERMINISTIC_SEED);
        let mut table = CuckooTable::new(params);

        for i in 0..n {
            let key = address_from_index(i);
            let value = initial_value(i);
            table.insert(&key, &value);
        }
        let item_size_elements = 64 / 2; // 32 u16 per entry
        // Use simple layout: db_rows = num_buckets, per = 1
        let db_rows = num_buckets;
        let db_cols = item_size_elements;
        let per = num_buckets / db_rows;

        let raw_db = table.to_raw_db(db_rows, db_cols, per, item_size_elements);
        assert_eq!(raw_db.len(), db_rows * db_cols);

        // Verify a known entry can be found
        let key0 = address_from_index(0);
        let positions = table.hasher.all_positions(&key0);
        let ks = table.params.key_size;
        let mut found = false;
        for &pos in &positions {
            if table.occupied[pos] && &table.bucket_slice(pos)[..ks] == key0.as_slice() {
                // Read from raw_db
                let row = pos / per;
                let base_col = (pos % per) * item_size_elements;
                let mut u16s = Vec::new();
                for j in 0..item_size_elements {
                    u16s.push(raw_db[(base_col + j) * db_rows + row]);
                }
                let entry_bytes = u16_be_to_bytes(&u16s);
                assert_eq!(&entry_bytes[..20], key0.as_slice());
                found = true;
                break;
            }
        }
        assert!(found, "key0 not found in raw_db");
    }

    #[test]
    fn test_build_from_csv() {
        // Create a small test CSV (address,nonce,balance_wei format)
        let dir = std::env::temp_dir();
        let csv_path = dir.join("test_accounts.csv");
        {
            let mut f = File::create(&csv_path).unwrap();
            use std::io::Write;
            writeln!(f, "# block=24644657").unwrap();
            writeln!(f, "address,nonce,balance_wei").unwrap();
            writeln!(
                f,
                "0x00000000219ab540356cbb839cbe05303d7705fa,5,1000000000000000000"
            )
            .unwrap();
            writeln!(f, "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2,0,0").unwrap();
        }

        let params = CuckooParams::new(100, 20, 40, 4, DETERMINISTIC_SEED);
        let mut table = CuckooTable::new(params);
        let (block, n) = table.build_from_csv(&csv_path).unwrap();

        assert_eq!(block, 24644657);
        assert_eq!(n, 2);

        // Verify first address is present
        let addr = hex::decode("00000000219ab540356cbb839cbe05303d7705fa").unwrap();
        let val = table.lookup(&addr).expect("address not found");
        // Balance 1e18 = 0xDE0B6B3A7640000 → 16 bytes BE
        // val[16..32] = balance, val[32..40] = nonce=5
        let nonce = u64::from_be_bytes(val[32..40].try_into().unwrap());
        assert_eq!(nonce, 5);

        std::fs::remove_file(&csv_path).ok();
    }
}

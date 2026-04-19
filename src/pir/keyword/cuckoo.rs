use rand::RngExt;
use rayon::prelude::*;
use siphasher::sip::SipHasher;
use std::hash::Hasher;

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

    fn key_from_index(i: usize, key_size: usize) -> Vec<u8> {
        let mut key = vec![0u8; key_size];
        let idx = i.to_be_bytes();
        let start = key_size.saturating_sub(idx.len());
        key[start..start + idx.len()].copy_from_slice(&idx);
        key
    }

    fn value_from_index(i: usize, value_size: usize) -> Vec<u8> {
        let mut value = vec![0u8; value_size];
        let idx = (i as u128).to_be_bytes();
        let copy_len = idx.len().min(value_size);
        value[value_size - copy_len..].copy_from_slice(&idx[idx.len() - copy_len..]);
        value
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

        let key = key_from_index(0, 20);
        let value = value_from_index(0, 40);
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
            let key = key_from_index(i, 20);
            let value = value_from_index(i, 40);
            table.insert(&key, &value);
            keys.push(key);
        }

        // Verify all lookups
        for (i, key) in keys.iter().enumerate() {
            let val = table.lookup(key).expect(&format!("key {} not found", i));
            let expected = value_from_index(i, 40);
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

        let key = key_from_index(0, 20);
        let value1 = value_from_index(0, 40);
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

        let key = key_from_index(0, 20);
        let value = value_from_index(0, 40);
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
            let key = key_from_index(i, 20);
            let value = value_from_index(i, 40);
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
        let key0 = key_from_index(0, 20);
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
}

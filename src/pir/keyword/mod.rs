pub mod cuckoo;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeywordRecord {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl KeywordRecord {
    pub fn new<K: Into<Vec<u8>>, V: Into<Vec<u8>>>(key: K, value: V) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeywordServerConfig {
    pub key_size: usize,
    pub value_size: usize,
    pub buckets: Option<usize>,
    pub dim0: Option<usize>,
}

use crate::aligned_memory::AlignedMemory64;
use crate::arith::log2_ceil;
use crate::packing::*;
use crate::params::*;
use crate::pir::params::*;
use crate::poly::{PolyMatrix, PolyMatrixNTT};
use crate::util::{read_arbitrary_bits, write_arbitrary_bits};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::Duration;

// Wire protocol message types
pub const MSG_HANDSHAKE: u8 = 0x01;
pub const MSG_QUERY: u8 = 0x02;
pub const MSG_RESPONSE: u8 = 0x03;
pub const MSG_SIDECAR: u8 = 0x04;
pub const MSG_KEYWORD_QUERY: u8 = 0x05;
pub const MSG_KEYWORD_RESPONSE: u8 = 0x06;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct KeywordPirHandshake {
    pub cuckoo_seed: Vec<u8>, // 16 bytes
    pub num_hashes: usize,    // 2
    pub key_size: usize,      // 20
    pub value_size: usize,    // 40
    pub entry_size: usize,    // 64 (padded)
    pub num_accounts: usize,  // pre-expansion count
    #[serde(default)]
    pub kem_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct HandshakeParams {
    pub num_items: usize,
    pub item_size_bits: usize,
    pub dim0: usize,
    #[serde(default)]
    pub keyword_pir: Option<KeywordPirHandshake>,
}

/// Send a framed message: [8 bytes: payload length (BE u64)] [1 byte: msg type] [payload]
pub fn send_msg(stream: &mut TcpStream, msg_type: u8, payload: &[u8]) -> io::Result<()> {
    // Combine header + type + payload into a single write to avoid
    // multiple small TCP segments (especially with TCP_NODELAY).
    let len = payload.len() as u64;
    let mut buf = Vec::with_capacity(9 + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.push(msg_type);
    buf.extend_from_slice(payload);
    stream.write_all(&buf)?;
    stream.flush()
}

/// Maximum message size (64 MiB) — sanity check to prevent OOM on corrupted frames.
const MAX_MSG_SIZE: u64 = 64 << 20;

/// Receive a framed message. Returns (msg_type, payload).
/// The first 8-byte read uses the caller's timeout (may be short for event draining).
/// Once the length is read, the timeout is extended so the full payload can arrive.
pub fn recv_msg(stream: &mut TcpStream) -> io::Result<(u8, Vec<u8>)> {
    let mut len_bytes = [0u8; 8];
    stream.read_exact(&mut len_bytes)?;
    let len = u64::from_be_bytes(len_bytes);

    if len > MAX_MSG_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {} bytes", len),
        ));
    }
    let len = len as usize;

    // Once we've committed to reading a message, give plenty of time for the payload
    let prev_timeout = stream.read_timeout()?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;

    let mut type_byte = [0u8; 1];
    let result = stream.read_exact(&mut type_byte).and_then(|_| {
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload)?;
        Ok((type_byte[0], payload))
    });

    // Restore original timeout
    stream.set_read_timeout(prev_timeout)?;
    result
}

pub const RGSW_SEEDS: [[u8; 32]; 6] = [
    [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7,
    ],
    [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7,
    ],
    [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7,
    ],
    [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7,
    ],
    [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7,
    ],
    [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7,
    ],
];

// ============================================================================
// Compact bit-packing helpers
// ============================================================================

/// CRT bit widths for the two moduli.
fn crt_bits(params: &Params) -> (usize, usize) {
    (
        log2_ceil(params.moduli[0]) as usize,
        log2_ceil(params.moduli[1]) as usize,
    )
}

/// Pack a condensed CRT value (crt0 | crt1<<32) into crt0_bits+crt1_bits bits.
#[inline]
fn pack_condensed(val: u64, crt0_bits: usize) -> u64 {
    let crt0 = val & ((1u64 << crt0_bits) - 1);
    let crt1 = val >> 32;
    crt0 | (crt1 << crt0_bits)
}

/// Unpack crt0_bits+crt1_bits bits back to condensed format (crt0 | crt1<<32).
#[inline]
fn unpack_condensed(packed: u64, crt0_bits: usize, crt1_bits: usize) -> u64 {
    let crt0 = packed & ((1u64 << crt0_bits) - 1);
    let crt1 = (packed >> crt0_bits) & ((1u64 << crt1_bits) - 1);
    crt0 | (crt1 << 32)
}

/// Allocate a bit-packing buffer. Adds 16 bytes of padding for write_arbitrary_bits
/// which may do u128 writes at the end of the buffer.
fn bitpack_buf(num_values: usize, bits_per: usize) -> Vec<u8> {
    let total_bits = num_values * bits_per;
    let bytes = (total_bits + 63) / 64 * 8 + 16;
    vec![0u8; bytes]
}

/// Write condensed values (packing keys or query row) into bit-packed buffer.
/// Returns the new bit offset.
fn write_condensed_values(
    buf: &mut [u8],
    bit_offs: usize,
    values: &[u64],
    crt0_bits: usize,
    total_bits: usize,
) -> usize {
    let mut bo = bit_offs;
    for &val in values {
        write_arbitrary_bits(buf, pack_condensed(val, crt0_bits), bo, total_bits);
        bo += total_bits;
    }
    bo
}

/// Read condensed values from bit-packed buffer. Returns the new bit offset.
fn read_condensed_values(
    buf: &[u8],
    bit_offs: usize,
    out: &mut [u64],
    crt0_bits: usize,
    crt1_bits: usize,
    total_bits: usize,
) -> usize {
    let mut bo = bit_offs;
    for v in out.iter_mut() {
        *v = unpack_condensed(
            read_arbitrary_bits(buf, bo, total_bits),
            crt0_bits,
            crt1_bits,
        );
        bo += total_bits;
    }
    bo
}

/// Write raw CRT poly (first poly_len = CRT0, next poly_len = CRT1) into bit-packed buffer.
/// Returns the new bit offset.
fn write_raw_crt_poly(
    buf: &mut [u8],
    bit_offs: usize,
    poly: &[u64],
    poly_len: usize,
    crt0_bits: usize,
    total_bits: usize,
) -> usize {
    let mut bo = bit_offs;
    for z in 0..poly_len {
        let packed = (poly[z] & ((1u64 << crt0_bits) - 1)) | (poly[z + poly_len] << crt0_bits);
        write_arbitrary_bits(buf, packed, bo, total_bits);
        bo += total_bits;
    }
    bo
}

/// Read raw CRT poly from bit-packed buffer. Returns the new bit offset.
fn read_raw_crt_poly(
    buf: &[u8],
    bit_offs: usize,
    poly: &mut [u64],
    poly_len: usize,
    crt0_bits: usize,
    crt1_bits: usize,
    total_bits: usize,
) -> usize {
    let mut bo = bit_offs;
    for z in 0..poly_len {
        let packed = read_arbitrary_bits(buf, bo, total_bits);
        poly[z] = packed & ((1u64 << crt0_bits) - 1);
        poly[z + poly_len] = (packed >> crt0_bits) & ((1u64 << crt1_bits) - 1);
        bo += total_bits;
    }
    bo
}

// ============================================================================
// Index PIR serialization (compact bit-packed)
// ============================================================================

pub fn serialize_everything(
    params: &Params,
    packing_keys: &mut PackingKeys<'_>,
    packed_query_row: AlignedMemory64,
    ct_gsw_body: PolyMatrixNTT<'_>,
) -> Vec<u8> {
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let (crt0_bits, crt1_bits) = crt_bits(params);
    let total_bits = crt0_bits + crt1_bits;

    // Count values: keys + query_row are condensed; RGSW body has poly_len coefficients per poly
    let num_condensed = params.t_exp_left * params.poly_len * 2 + db_rows;
    let num_rgsw_coeffs = 2 * params.t_gsw * params.poly_len; // poly_len per poly (CRT0+CRT1 packed)
    let total_values = num_condensed + num_rgsw_coeffs;

    let mut buf = bitpack_buf(total_values, total_bits);
    let mut bo = 0;

    // Packing keys (y, z interleaved)
    for i in 0..params.t_exp_left {
        let y_poly = packing_keys
            .y_body_condensed
            .as_ref()
            .unwrap()
            .get_poly(0, i);
        let z_poly = packing_keys
            .z_body_condensed
            .as_ref()
            .unwrap()
            .get_poly(0, i);
        for k in 0..params.poly_len {
            write_arbitrary_bits(
                &mut buf,
                pack_condensed(y_poly[k], crt0_bits),
                bo,
                total_bits,
            );
            bo += total_bits;
            write_arbitrary_bits(
                &mut buf,
                pack_condensed(z_poly[k], crt0_bits),
                bo,
                total_bits,
            );
            bo += total_bits;
        }
    }

    // Query row (condensed)
    bo = write_condensed_values(
        &mut buf,
        bo,
        packed_query_row.as_slice(),
        crt0_bits,
        total_bits,
    );

    // RGSW body (raw CRT: 2*poly_len u64s per poly, pack CRT0+CRT1 into total_bits)
    for i in 0..2 * params.t_gsw {
        bo = write_raw_crt_poly(
            &mut buf,
            bo,
            ct_gsw_body.get_poly(0, i),
            params.poly_len,
            crt0_bits,
            total_bits,
        );
    }

    // Trim trailing padding
    let used_bytes = (bo + 7) / 8;
    buf.truncate(used_bytes);
    buf
}

pub fn deserialize_everything<'a>(
    params: &'a Params,
    packing_params: &'a PackParams,
    all_u8: Vec<u8>,
) -> (PackingKeys<'a>, AlignedMemory64, PolyMatrixNTT<'a>) {
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let (crt0_bits, crt1_bits) = crt_bits(params);
    let total_bits = crt0_bits + crt1_bits;

    // Pad buffer for read_arbitrary_bits u128 reads at boundary
    let mut buf = all_u8;
    buf.resize(buf.len() + 16, 0);
    let mut bo = 0;

    // Packing keys
    let mut y_body_condensed = PolyMatrixNTT::zero(params, 1, params.t_exp_left);
    let mut z_body_condensed = PolyMatrixNTT::zero(params, 1, params.t_exp_left);
    for i in 0..params.t_exp_left {
        for k in 0..params.poly_len {
            y_body_condensed.get_poly_mut(0, i)[k] = unpack_condensed(
                read_arbitrary_bits(&buf, bo, total_bits),
                crt0_bits,
                crt1_bits,
            );
            bo += total_bits;
            z_body_condensed.get_poly_mut(0, i)[k] = unpack_condensed(
                read_arbitrary_bits(&buf, bo, total_bits),
                crt0_bits,
                crt1_bits,
            );
            bo += total_bits;
        }
    }

    let packing_keys = PackingKeys {
        packing_type: PackingType::InspiRING,
        packing_params: Some(packing_params.clone()),
        full_key: true,
        y_body: Some(y_body_condensed.clone()),
        z_body: Some(z_body_condensed.clone()),
        y_body_condensed: Some(y_body_condensed),
        z_body_condensed: Some(z_body_condensed),
        expanded: false,
        y_all_condensed: None,
        y_bar_all_condensed: None,
        params: None,
        pack_pub_params_row_1s: vec![],
        fake_pack_pub_params: vec![],
    };

    // Query row
    let mut packed_query_row = AlignedMemory64::new(db_rows);
    bo = read_condensed_values(
        &buf,
        bo,
        packed_query_row.as_mut_slice(),
        crt0_bits,
        crt1_bits,
        total_bits,
    );

    // RGSW body
    let mut ct_gsw_body_raw = PolyMatrixNTT::zero(params, 1, 2 * params.t_gsw);
    for i in 0..2 * params.t_gsw {
        bo = read_raw_crt_poly(
            &buf,
            bo,
            ct_gsw_body_raw.get_poly_mut(0, i),
            params.poly_len,
            crt0_bits,
            crt1_bits,
            total_bits,
        );
    }
    let _ = bo;

    (packing_keys, packed_query_row, ct_gsw_body_raw)
}

/// Serialize sidecar entries for wire transmission.
/// Format: [u32 num_entries] [per entry: u64 index, u32 num_values, values as u16 LE...]
pub fn serialize_sidecar(entries: &[(usize, Vec<u16>)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (index, values) in entries {
        buf.extend_from_slice(&(*index as u64).to_le_bytes());
        buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for &v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Deserialize sidecar entries from wire format.
pub fn deserialize_sidecar(data: &[u8]) -> Vec<(usize, Vec<u16>)> {
    let mut entries = Vec::new();
    if data.len() < 4 {
        return entries;
    }
    let num_entries = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut offset = 4;
    for _ in 0..num_entries {
        if offset + 12 > data.len() {
            break;
        }
        let index = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let num_values = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut values = Vec::with_capacity(num_values);
        for _ in 0..num_values {
            if offset + 2 > data.len() {
                break;
            }
            values.push(u16::from_le_bytes(
                data[offset..offset + 2].try_into().unwrap(),
            ));
            offset += 2;
        }
        entries.push((index, values));
    }
    entries
}

fn max_interpolate_degree(
    modulus: f64,
    d0: f64,
    p: f64,
    t_exp: f64,
    t_rgsw: f64,
    poly_len: f64,
) -> usize {
    let sigma_x = 6.4 as f64;
    let modulus_len = modulus.log2();
    let z1 = (2.0 as f64).powi((modulus_len / t_exp).ceil() as i32);
    let z2 = (2.0 as f64).powi((modulus_len / t_rgsw).ceil() as i32);
    let term1_variance = d0 * p.powi(2) * sigma_x.powi(2);
    let term2_variance = t_exp * poly_len.powi(2) * z1.powi(2) * sigma_x.powi(2) / 4.0;
    let term3_variance = t_rgsw * poly_len * z2.powi(2) * sigma_x.powi(2) / 2.0;

    let total_log2_std_before_poly_eval = (term1_variance + term2_variance + term3_variance)
        .sqrt()
        .log2();

    let log2_std_upper_bound =
        (modulus / (2. * 2. * p)).log2() - (2.0 * 41.0 * (2 as f64).ln()).sqrt().log2();
    let max_log_interpolate_degree = 2. * (log2_std_upper_bound - total_log2_std_before_poly_eval);
    assert!(max_log_interpolate_degree >= 0.);
    let max_interpolate_degree = (2.0 as f64).powf(max_log_interpolate_degree).floor() as usize;
    std::cmp::min(max_interpolate_degree as usize, poly_len as usize)
}

pub fn params_rgswpir_given_input_size_and_dim0<'a>(
    input_num_items: usize,
    input_item_size_bits: usize,
    dim0: usize,
) -> (Params, usize, (usize, usize, usize)) {
    let poly_len_log2 = 11;
    let poly_len = 1 << poly_len_log2;
    let p = 65535;
    let log_p = 16;
    let q2_bits = 28; // modulus after packing
    let t_exp_left = 3;
    let modulus = (268369921u64 * 249561089u64) as f64;

    let max_interpolate_degree = max_interpolate_degree(
        modulus,
        dim0 as f64,
        p as f64,
        t_exp_left as f64,
        t_exp_left as f64,
        poly_len as f64,
    ) as usize;
    assert!(max_interpolate_degree > 1);

    let bits_per_poly = (poly_len * log_p) as f64;

    let mut factor = (bits_per_poly as f64 / input_item_size_bits as f64).floor() as usize;
    if factor == 0 {
        factor = 1;
    }
    let input_num_items = (input_num_items as f64 / factor as f64).ceil() as usize;
    let input_item_size_bits = input_item_size_bits * factor;

    let padded_item_size_num_bits =
        ((input_item_size_bits as f64) / bits_per_poly).ceil() as usize * bits_per_poly as usize;
    let padded_item_num_pts = (padded_item_size_num_bits as f64 / log_p as f64).ceil() as usize;

    let dim1_lower_bound =
        padded_item_num_pts * (input_num_items as f64 / dim0 as f64).ceil() as usize;
    let mut current_dim1 = padded_item_num_pts;
    let mut interpolate_degree = 1;
    while current_dim1 < dim1_lower_bound {
        interpolate_degree *= 2;
        current_dim1 *= 2;
        if 2 * interpolate_degree > max_interpolate_degree {
            break;
        }
    }

    let new_item_size_num_pts = round_up_to_multiple_of(dim1_lower_bound, current_dim1);

    let db_rows = if dim0 >= 8 { dim0 } else { 8 };

    let db_cols_poly = (new_item_size_num_pts as f64 / (poly_len as f64)).ceil() as usize;

    let nu_1 = db_rows.next_power_of_two().trailing_zeros() as usize - poly_len_log2;

    let mut params = internal_params_for(poly_len, nu_1, 0, p, q2_bits, t_exp_left, DEF_MODULI);
    params.instances = db_cols_poly;
    (
        params,
        interpolate_degree,
        (db_rows, 1, new_item_size_num_pts * log_p),
    )
}

fn pad_to_width(s: &str, width: usize) -> String {
    if s.len() >= width {
        String::from_utf8_lossy(&s.as_bytes()[..width]).into_owned()
    } else {
        format!("{s:<width$}")
    }
}

pub fn read_file_into_matrix(
    path: &Path,
    num_items: usize,
    item_size_bits: usize,
) -> io::Result<Vec<Vec<u16>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut matrix: Vec<Vec<u16>> = Vec::with_capacity(num_items);

    for line_result in reader.lines() {
        let line = line_result?;

        let line = pad_to_width(&line, item_size_bits / 8);

        let row: Vec<u16> = line
            .as_bytes()
            .chunks_exact(2)
            .map(|chunk| (chunk[0] as u16) << 8 | (chunk[1] as u16))
            .collect();

        matrix.push(row);
    }

    while matrix.len() < num_items {
        matrix.push(vec![0u16; item_size_bits / 16]);
    }

    Ok(matrix)
}

// ============================================================================
// Keyword PIR serialization
// ============================================================================

#[derive(Clone)]
pub struct KeywordQuery<'a> {
    pub packed_query_row: AlignedMemory64,
    pub ct_gsw_body: PolyMatrixNTT<'a>,
}

pub struct DecodedKeywordQuery<'a> {
    pub packing_keys: PackingKeys<'a>,
    pub queries: Vec<KeywordQuery<'a>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeywordResponseSidecarEntry {
    pub address: Vec<u8>,
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeywordResponsePayload {
    pub responses: Vec<Vec<u8>>,
    pub stash_entries: Vec<Vec<u8>>,
    pub sidecar_entries: Vec<KeywordResponseSidecarEntry>,
    pub block_number: u64,
}

/// Serialize a keyword query: shared packing keys + N per-hash queries.
/// Format: [u8 num_queries][bit-packed: keys | per-query (query_row + rgsw_body)]
pub fn serialize_keyword_query(
    params: &Params,
    packing_keys: &mut PackingKeys<'_>,
    queries: &[KeywordQuery<'_>],
) -> Vec<u8> {
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let (crt0_bits, _crt1_bits) = crt_bits(params);
    let total_bits = crt0_bits + _crt1_bits;
    let num_queries = queries.len();

    let num_keys = params.t_exp_left * params.poly_len * 2;
    let per_query = db_rows + 2 * params.t_gsw * params.poly_len;
    let total_values = num_keys + num_queries * per_query;

    // 1 byte for num_queries header, then bit-packed data
    let mut buf = vec![0u8; 1];
    buf[0] = num_queries as u8;
    buf.resize(1 + (total_values * total_bits + 63) / 64 * 8 + 16, 0);
    let base = 8; // bit offset: skip first byte (= 8 bits)
    let mut bo = base;

    // Shared packing keys (y, z interleaved)
    for i in 0..params.t_exp_left {
        let y_poly = packing_keys
            .y_body_condensed
            .as_ref()
            .unwrap()
            .get_poly(0, i);
        let z_poly = packing_keys
            .z_body_condensed
            .as_ref()
            .unwrap()
            .get_poly(0, i);
        for k in 0..params.poly_len {
            write_arbitrary_bits(
                &mut buf,
                pack_condensed(y_poly[k], crt0_bits),
                bo,
                total_bits,
            );
            bo += total_bits;
            write_arbitrary_bits(
                &mut buf,
                pack_condensed(z_poly[k], crt0_bits),
                bo,
                total_bits,
            );
            bo += total_bits;
        }
    }

    // Per-query data
    for query in queries {
        bo = write_condensed_values(
            &mut buf,
            bo,
            query.packed_query_row.as_slice(),
            crt0_bits,
            total_bits,
        );
        for i in 0..2 * params.t_gsw {
            bo = write_raw_crt_poly(
                &mut buf,
                bo,
                query.ct_gsw_body.get_poly(0, i),
                params.poly_len,
                crt0_bits,
                total_bits,
            );
        }
    }

    let used_bytes = (bo + 7) / 8;
    buf.truncate(used_bytes);
    buf
}

/// Deserialize a keyword query.
pub fn deserialize_keyword_query<'a>(
    params: &'a Params,
    packing_params: &'a PackParams,
    all_u8: Vec<u8>,
) -> DecodedKeywordQuery<'a> {
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let (crt0_bits, crt1_bits) = crt_bits(params);
    let total_bits = crt0_bits + crt1_bits;

    let num_queries = all_u8[0] as usize;

    // Pad for read_arbitrary_bits u128 boundary reads
    let mut buf = all_u8;
    buf.resize(buf.len() + 16, 0);
    let mut bo = 8; // skip header byte

    // Shared packing keys
    let mut y_body_condensed = PolyMatrixNTT::zero(params, 1, params.t_exp_left);
    let mut z_body_condensed = PolyMatrixNTT::zero(params, 1, params.t_exp_left);
    for i in 0..params.t_exp_left {
        for k in 0..params.poly_len {
            y_body_condensed.get_poly_mut(0, i)[k] = unpack_condensed(
                read_arbitrary_bits(&buf, bo, total_bits),
                crt0_bits,
                crt1_bits,
            );
            bo += total_bits;
            z_body_condensed.get_poly_mut(0, i)[k] = unpack_condensed(
                read_arbitrary_bits(&buf, bo, total_bits),
                crt0_bits,
                crt1_bits,
            );
            bo += total_bits;
        }
    }

    let packing_keys = PackingKeys {
        packing_type: PackingType::InspiRING,
        packing_params: Some(packing_params.clone()),
        full_key: true,
        y_body: Some(y_body_condensed.clone()),
        z_body: Some(z_body_condensed.clone()),
        y_body_condensed: Some(y_body_condensed),
        z_body_condensed: Some(z_body_condensed),
        expanded: false,
        y_all_condensed: None,
        y_bar_all_condensed: None,
        params: None,
        pack_pub_params_row_1s: vec![],
        fake_pack_pub_params: vec![],
    };

    // Per-query data
    let mut queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let mut packed_query_row = AlignedMemory64::new(db_rows);
        bo = read_condensed_values(
            &buf,
            bo,
            packed_query_row.as_mut_slice(),
            crt0_bits,
            crt1_bits,
            total_bits,
        );

        let mut ct_gsw_body = PolyMatrixNTT::zero(params, 1, 2 * params.t_gsw);
        for i in 0..2 * params.t_gsw {
            bo = read_raw_crt_poly(
                &buf,
                bo,
                ct_gsw_body.get_poly_mut(0, i),
                params.poly_len,
                crt0_bits,
                crt1_bits,
                total_bits,
            );
        }
        queries.push(KeywordQuery {
            packed_query_row,
            ct_gsw_body,
        });
    }

    DecodedKeywordQuery {
        packing_keys,
        queries,
    }
}

/// Serialize keyword response: PIR responses + stash + sidecar + block_number.
pub fn serialize_keyword_response(payload: &KeywordResponsePayload) -> Vec<u8> {
    let mut buf = Vec::new();

    // Responses
    buf.extend_from_slice(&(payload.responses.len() as u32).to_le_bytes());
    for resp in &payload.responses {
        buf.extend_from_slice(&(resp.len() as u32).to_le_bytes());
        buf.extend_from_slice(resp);
    }

    // Stash entries (each is entry_size bytes)
    buf.extend_from_slice(&(payload.stash_entries.len() as u32).to_le_bytes());
    for entry in &payload.stash_entries {
        buf.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        buf.extend_from_slice(entry);
    }

    // Sidecar: (address, value) pairs
    buf.extend_from_slice(&(payload.sidecar_entries.len() as u32).to_le_bytes());
    for entry in &payload.sidecar_entries {
        buf.extend_from_slice(&(entry.address.len() as u32).to_le_bytes());
        buf.extend_from_slice(&entry.address);
        buf.extend_from_slice(&(entry.value.len() as u32).to_le_bytes());
        buf.extend_from_slice(&entry.value);
    }

    // Block number
    buf.extend_from_slice(&payload.block_number.to_le_bytes());

    buf
}

/// Deserialize keyword response.
pub fn deserialize_keyword_response(data: &[u8]) -> KeywordResponsePayload {
    let mut offset = 0;

    let read_u32 = |off: &mut usize| -> u32 {
        let v = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
        *off += 4;
        v
    };
    let read_u64 = |off: &mut usize| -> u64 {
        let v = u64::from_le_bytes(data[*off..*off + 8].try_into().unwrap());
        *off += 8;
        v
    };

    // Responses
    let num_responses = read_u32(&mut offset) as usize;
    let mut responses = Vec::with_capacity(num_responses);
    for _ in 0..num_responses {
        let len = read_u32(&mut offset) as usize;
        responses.push(data[offset..offset + len].to_vec());
        offset += len;
    }

    // Stash
    let num_stash = read_u32(&mut offset) as usize;
    let mut stash = Vec::with_capacity(num_stash);
    for _ in 0..num_stash {
        let len = read_u32(&mut offset) as usize;
        stash.push(data[offset..offset + len].to_vec());
        offset += len;
    }

    // Sidecar
    let num_sidecar = read_u32(&mut offset) as usize;
    let mut sidecar_entries = Vec::with_capacity(num_sidecar);
    for _ in 0..num_sidecar {
        let addr_len = read_u32(&mut offset) as usize;
        let address = data[offset..offset + addr_len].to_vec();
        offset += addr_len;
        let val_len = read_u32(&mut offset) as usize;
        let value = data[offset..offset + val_len].to_vec();
        offset += val_len;
        sidecar_entries.push(KeywordResponseSidecarEntry { address, value });
    }

    // Block number
    let block_number = read_u64(&mut offset);

    KeywordResponsePayload {
        responses,
        stash_entries: stash,
        sidecar_entries,
        block_number,
    }
}

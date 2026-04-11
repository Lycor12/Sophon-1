//! u64 ternary packing — 32 ternary weights per u64.
//!
//! Encoding: 2 bits per weight, 32 weights per u64.
//!   00 = 0, 01 = +1, 10 = -1, 11 = reserved (treated as 0).
//!
//! Bit layout (LSB-first): bits [1:0] = weight 0, [3:2] = weight 1, ... [63:62] = weight 31.
//!
//! This is 4x denser than u8 packing (32 vs 4 per unit), enabling wider
//! SIMD loads: one u64 load fetches 32 weights for parallel accumulation.

const WEIGHTS_PER_U64: usize = 32;

/// Pack up to 32 ternary i8 values into a single u64.
///
/// Values beyond `vals.len()` are treated as 0.
/// Encoding: -1 -> 0b10, 0 -> 0b00, +1 -> 0b01.
///
/// # Examples
///
/// ```
/// use sophon_accel::pack64::pack_32_ternary;
///
/// // Pack 32 ternary values
/// let vals: Vec<i8> = vec![
///     1, -1, 0, 1, -1, 0, 1, -1,
///     0, 1, -1, 0, 1, -1, 0, 1,
///     -1, 0, 1, -1, 0, 1, -1, 0,
///     1, -1, 0, 1, -1, 0, 1, -1,
/// ];
/// let packed = pack_32_ternary(&vals);
/// assert_ne!(packed, 0); // Packed representation is non-zero
///
/// // Partial packing (only 5 values)
/// let partial = vec![1i8, -1, 0, 1, -1];
/// let packed_partial = pack_32_ternary(&partial);
/// // The remaining 27 values are treated as 0
/// ```
pub fn pack_32_ternary(vals: &[i8]) -> u64 {
    let n = vals.len().min(WEIGHTS_PER_U64);
    let mut packed: u64 = 0;
    for i in 0..n {
        let bits: u64 = match vals[i] {
            1 => 0b01,
            -1 => 0b10,
            _ => 0b00,
        };
        packed |= bits << (i * 2);
    }
    packed
}

/// Unpack a u64 into up to 32 ternary i8 values.
///
/// `count` specifies how many to extract (max 32).
///
/// # Examples
///
/// ```
/// use sophon_accel::pack64::{pack_32_ternary, unpack_32_ternary};
///
/// // Roundtrip: pack then unpack
/// let original: Vec<i8> = vec![1, -1, 0, 1, -1];
/// let packed = pack_32_ternary(&original);
/// let unpacked = unpack_32_ternary(packed, 5);
/// assert_eq!(original, unpacked);
///
/// // Unpack full 32 values
/// let vals: Vec<i8> = (0..32).map(|i| match i % 3 {
///     0 => -1,
///     1 => 0,
///     _ => 1,
/// }).collect();
/// let packed = pack_32_ternary(&vals);
/// let unpacked = unpack_32_ternary(packed, 32);
/// assert_eq!(vals, unpacked);
///
/// // Extract partial count
/// let partial = unpack_32_ternary(packed, 10);
/// assert_eq!(partial.len(), 10);
/// ```
pub fn unpack_32_ternary(packed: u64, count: usize) -> Vec<i8> {
    let n = count.min(WEIGHTS_PER_U64);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = (packed >> (i * 2)) & 0b11;
        let val = match bits {
            0b01 => 1i8,
            0b10 => -1i8,
            _ => 0i8,
        };
        out.push(val);
    }
    out
}

/// Pack a full slice of ternary i8 values into u64 words.
pub fn pack_all_u64(vals: &[i8]) -> Vec<u64> {
    let n_words = (vals.len() + WEIGHTS_PER_U64 - 1) / WEIGHTS_PER_U64;
    let mut out = Vec::with_capacity(n_words);
    for chunk in vals.chunks(WEIGHTS_PER_U64) {
        out.push(pack_32_ternary(chunk));
    }
    out
}

/// Unpack a slice of u64 words back to ternary i8 values.
pub fn unpack_all_u64(packed: &[u64], original_len: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(original_len);
    let mut remaining = original_len;
    for &word in packed {
        let count = remaining.min(WEIGHTS_PER_U64);
        out.extend_from_slice(&unpack_32_ternary(word, count));
        remaining = remaining.saturating_sub(WEIGHTS_PER_U64);
    }
    out.truncate(original_len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_32() {
        let vals: Vec<i8> = (0..32)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let packed = pack_32_ternary(&vals);
        let unpacked = unpack_32_ternary(packed, 32);
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_unpack_roundtrip_partial() {
        let vals = vec![1i8, -1, 0, 1, -1];
        let packed = pack_32_ternary(&vals);
        let unpacked = unpack_32_ternary(packed, 5);
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_all_unpack_all_roundtrip() {
        let vals: Vec<i8> = (0..100)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let packed = pack_all_u64(&vals);
        assert_eq!(packed.len(), 4); // ceil(100/32) = 4
        let unpacked = unpack_all_u64(&packed, 100);
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn all_zeros_packs_to_zero() {
        let vals = vec![0i8; 32];
        assert_eq!(pack_32_ternary(&vals), 0u64);
    }

    #[test]
    fn density_32_per_u64() {
        // 32 weights in 8 bytes = 4 weights per byte (same density as u8 packing)
        // but enables wider SIMD loads
        let vals: Vec<i8> = vec![1; 32];
        let packed = pack_all_u64(&vals);
        assert_eq!(packed.len(), 1);
        assert_eq!(std::mem::size_of_val(&packed[0]), 8);
    }

    #[test]
    fn empty_input() {
        let packed = pack_32_ternary(&[]);
        assert_eq!(packed, 0u64);
        let unpacked = unpack_32_ternary(packed, 0);
        assert!(unpacked.is_empty());
    }
}

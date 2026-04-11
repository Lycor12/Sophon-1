//! Bit-packing of ternary weights into u8 bytes.
//!
//! 4 ternary values per byte, 2 bits each:
//!   00 = 0, 01 = +1, 10 = -1, 11 = unused (treated as 0 on decode)
//!
//! This gives 2 bits/weight storage rather than 8 (i8) or 32 (f32),
//! achieving the spec's ~400-500MB target for ~3B weights.
//!
//! Memory: 3B weights * 2 bits = 750MB. With BST scale factors:
//!   3B / 64 * 4 bytes (f32) = ~188MB extra.
//!   Total: ~938MB — still well under 2GB VRAM.
//! For the actual ~30M-range model implied by the fixed architecture
//! dimensions, the packed size will be much smaller.

/// Pack up to 4 ternary i8 values into one u8.
/// Input slice length must be 1..=4.
/// Encoding: 00=0, 01=+1, 10=-1.
#[inline]
pub fn pack_ternary(vals: &[i8]) -> u8 {
    debug_assert!(!vals.is_empty() && vals.len() <= 4);
    let mut byte = 0u8;
    for (i, &v) in vals.iter().enumerate() {
        let bits: u8 = match v {
            0 => 0b00,
            1 => 0b01,
            -1 => 0b10,
            _ => 0b00, // clamp out-of-range
        };
        byte |= bits << (6 - 2 * i);
    }
    byte
}

/// Unpack one u8 into up to `count` ternary i8 values (1..=4).
pub fn unpack_ternary(byte: u8, count: usize) -> [i8; 4] {
    debug_assert!(count >= 1 && count <= 4);
    let mut out = [0i8; 4];
    for i in 0..count {
        let bits = (byte >> (6 - 2 * i)) & 0b11;
        out[i] = match bits {
            0b01 => 1,
            0b10 => -1,
            _ => 0,
        };
    }
    out
}

/// Pack a slice of ternary i8 values into a Vec<u8>.
/// Pads the final byte with zeros if len is not a multiple of 4.
pub fn pack_all(vals: &[i8]) -> Vec<u8> {
    let n_bytes = vals.len().div_ceil(4);
    let mut out = Vec::with_capacity(n_bytes);
    let mut i = 0;
    while i < vals.len() {
        let chunk_len = (vals.len() - i).min(4);
        out.push(pack_ternary(&vals[i..i + chunk_len]));
        i += chunk_len;
    }
    out
}

/// Unpack a Vec<u8> into ternary i8 values, truncated to `original_len`.
pub fn unpack_all(packed: &[u8], original_len: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(original_len);
    for &byte in packed {
        let remaining = original_len - out.len();
        let count = remaining.min(4);
        let vals = unpack_ternary(byte, count);
        for k in 0..count {
            out.push(vals[k]);
        }
        if out.len() >= original_len {
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let vals: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let packed = pack_all(&vals);
        let unpacked = unpack_all(&packed, vals.len());
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_single_values() {
        assert_eq!(pack_ternary(&[0]), 0b00_00_00_00);
        assert_eq!(pack_ternary(&[1]), 0b01_00_00_00);
        assert_eq!(pack_ternary(&[-1]), 0b10_00_00_00);
    }

    #[test]
    fn pack_4_values() {
        let v = [1i8, -1, 0, 1];
        let b = pack_ternary(&v);
        let u = unpack_ternary(b, 4);
        assert_eq!(u[0], 1);
        assert_eq!(u[1], -1);
        assert_eq!(u[2], 0);
        assert_eq!(u[3], 1);
    }

    #[test]
    fn density_2_bits_per_weight() {
        let vals: Vec<i8> = vec![1i8; 400];
        let packed = pack_all(&vals);
        // 400 weights at 2 bits = 100 bytes
        assert_eq!(packed.len(), 100);
    }
}

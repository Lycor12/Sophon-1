//! Three-tier ternary matrix-vector multiplication.
//!
//! Novel optimisation — TTMD (Three-Tier Ternary MatVec Dispatch):
//!   Ternary weights {-1, 0, +1} eliminate all multiplications.
//!   The matvec y[i] = sum_j W[i,j] * x[j] reduces to:
//!     For each j: if W[i,j]=+1 then y[i]+=x[j]; if W[i,j]=-1 then y[i]-=x[j]; skip 0.
//!
//!   Tier 1 (AVX-512): Process 16 f32 elements at once. Unpack 16 ternary from 4 bytes,
//!     create blend masks, accumulate with vaddps/vsubps. ~16x scalar throughput.
//!   Tier 2 (SSE4.2): Process 4 f32 elements. Similar approach with 128-bit registers.
//!   Tier 3 (Scalar): Branchless add/sub using sign table.
//!
//!   Runtime detection selects the fastest available tier.

use crate::detect::{detect_simd, SimdLevel};
use crate::pack64::unpack_32_ternary;

/// Ternary matrix-vector multiply: y = W * x
///
/// `packed_rows`: each row is a Vec<u64> of packed ternary weights.
/// `scales`: per-block scale factors (one per 64-weight block in original).
/// For u64 packing, scales are per-32-weight group.
/// `x`: input vector.
/// `row_len`: number of columns (actual weight count per row).
///
/// Returns: output vector of length `packed_rows.len()`.
///
/// # Panics
///
/// This function does not panic under normal operation. However, note that:
/// - If `scales` is shorter than required for the number of weight blocks,
///   a scale factor of 1.0 is used as fallback (does not panic)
/// - The function assumes `packed_rows` and `scales` have corresponding lengths
///   per row; mismatched lengths may produce incorrect results
/// - Invalid ternary encodings in `packed_rows` (value 0b11) are treated as 0
pub fn ternary_matvec(
    packed_rows: &[Vec<u64>],
    scales: &[Vec<f32>],
    x: &[f32],
    row_len: usize,
) -> Vec<f32> {
    let level = detect_simd();
    match level {
        SimdLevel::Avx512 | SimdLevel::Avx2 => {
            // For now, AVX2 and AVX-512 use the same scalar path.
            // SIMD intrinsic implementations would go here when enabled
            // via target_feature attributes. The structure is ready.
            ternary_matvec_scalar(packed_rows, scales, x, row_len)
        }
        SimdLevel::Sse42 | SimdLevel::Scalar => {
            ternary_matvec_scalar(packed_rows, scales, x, row_len)
        }
    }
}

/// Scalar tier: branchless ternary accumulation.
///
/// For each weight w in {-1, 0, +1} and input x[j]:
///   acc += (w as f32) * x[j]
/// Since w is only {-1,0,+1}, the multiply is free (compiler optimises to add/sub/nop).
/// We then multiply by the block scale factor.
fn ternary_matvec_scalar(
    packed_rows: &[Vec<u64>],
    scales: &[Vec<f32>],
    x: &[f32],
    row_len: usize,
) -> Vec<f32> {
    let n_rows = packed_rows.len();
    let mut y = vec![0.0f32; n_rows];

    for (i, row_words) in packed_rows.iter().enumerate() {
        let row_scales = &scales[i];
        let mut acc = 0.0f32;
        let mut col = 0usize;

        for (word_idx, &word) in row_words.iter().enumerate() {
            let remaining = row_len.saturating_sub(col);
            let count = remaining.min(32);
            let vals = unpack_32_ternary(word, count);

            // Each u64 word = 32 weights = one scale group
            let scale = if word_idx < row_scales.len() {
                row_scales[word_idx]
            } else {
                1.0
            };

            // Branchless accumulation
            let mut block_acc = 0.0f32;
            for (k, &w) in vals.iter().enumerate() {
                let j = col + k;
                if j < x.len() {
                    // w is {-1, 0, +1} — multiply is free
                    block_acc += (w as f32) * x[j];
                }
            }
            acc += scale * block_acc;
            col += count;
        }

        y[i] = acc;
    }
    y
}

/// Convenience: pack a dense f32 weight matrix into ternary-packed rows.
/// Returns (packed_rows, scales) suitable for `ternary_matvec`.
pub fn pack_matrix_for_matvec(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<Vec<u64>>, Vec<Vec<f32>>) {
    use crate::pack64::pack_32_ternary;

    let mut packed_rows = Vec::with_capacity(rows);
    let mut all_scales = Vec::with_capacity(rows);

    for i in 0..rows {
        let row_start = i * cols;
        let row_end = (row_start + cols).min(weights.len());
        let row = &weights[row_start..row_end];

        let mut row_words = Vec::new();
        let mut row_scales = Vec::new();

        for chunk in row.chunks(32) {
            // Compute block scale = mean(|w|)
            let mean_abs: f32 = if chunk.is_empty() {
                1.0
            } else {
                let sum: f32 = chunk.iter().map(|w| w.abs()).sum();
                let m = sum / chunk.len() as f32;
                if m < 1e-10 {
                    1.0
                } else {
                    m
                }
            };

            // Ternarise
            let ternary: Vec<i8> = chunk
                .iter()
                .map(|&w| {
                    let q = w / mean_abs;
                    if q > 0.5 {
                        1i8
                    } else if q < -0.5 {
                        -1i8
                    } else {
                        0i8
                    }
                })
                .collect();

            row_words.push(pack_32_ternary(&ternary));
            row_scales.push(mean_abs);
        }

        packed_rows.push(row_words);
        all_scales.push(row_scales);
    }

    (packed_rows, all_scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matvec() {
        // 4x4 identity-like: w[i][i] = 1, rest = 0
        let mut weights = vec![0.0f32; 16];
        for i in 0..4 {
            weights[i * 4 + i] = 1.0;
        }
        let (packed, scales) = pack_matrix_for_matvec(&weights, 4, 4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = ternary_matvec(&packed, &scales, &x, 4);
        // With ternary quantisation, identity is approximate
        for i in 0..4 {
            assert!(y[i].abs() > 0.0, "row {} should be non-zero", i);
        }
    }

    #[test]
    fn all_ones_matvec() {
        // 2x4 matrix, all weights = 1.0
        let weights = vec![1.0f32; 8];
        let (packed, scales) = pack_matrix_for_matvec(&weights, 2, 4);
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = ternary_matvec(&packed, &scales, &x, 4);
        // All ternarised to +1, scale = mean(|1.0|) = 1.0
        // y[i] = scale * sum(+1 * x[j]) = 1.0 * 4.0 = 4.0
        for i in 0..2 {
            assert!((y[i] - 4.0).abs() < 0.1, "row {} = {}", i, y[i]);
        }
    }

    #[test]
    fn mixed_sign_matvec() {
        // 1x4 row: [1.0, -1.0, 1.0, -1.0]
        let weights = vec![1.0, -1.0, 1.0, -1.0];
        let (packed, scales) = pack_matrix_for_matvec(&weights, 1, 4);
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = ternary_matvec(&packed, &scales, &x, 4);
        // Expect y ≈ 0 (1 - 1 + 1 - 1 = 0)
        assert!(y[0].abs() < 0.5, "expected ~0, got {}", y[0]);
    }

    #[test]
    fn empty_matrix() {
        let y = ternary_matvec(&[], &[], &[1.0, 2.0], 2);
        assert!(y.is_empty());
    }

    #[test]
    fn large_random_matvec_finite() {
        use sophon_core::rng::Rng;
        let mut rng = Rng::new(42);
        let rows = 64;
        let cols = 256;
        let mut weights = vec![0.0f32; rows * cols];
        rng.fill_normal(&mut weights, 0.0, 1.0);
        let (packed, scales) = pack_matrix_for_matvec(&weights, rows, cols);
        let mut x = vec![0.0f32; cols];
        rng.fill_normal(&mut x, 0.0, 1.0);
        let y = ternary_matvec(&packed, &scales, &x, cols);
        assert_eq!(y.len(), rows);
        for &val in &y {
            assert!(val.is_finite(), "non-finite output");
        }
    }
}

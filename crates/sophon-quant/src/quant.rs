//! Ternary quantisation: float -> {-1, 0, +1}.

/// Block size for Block-Scaled Ternary (BST).
pub const BLOCK_SIZE: usize = 64;

/// Quantise a single f32 value to ternary {-1, 0, +1}.
///
/// threshold = 0.5 (default): values with |v| < threshold * scale round to 0.
#[inline]
pub fn ternarize(v: f32, scale: f32, threshold: f32) -> i8 {
    if scale.abs() < 1e-12 {
        return 0;
    }
    let normalised = v / scale;
    if normalised > threshold {
        1
    } else if normalised < -threshold {
        -1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// TernaryBlock
// ---------------------------------------------------------------------------

/// A block of BLOCK_SIZE ternary weights with its scale factor.
#[derive(Clone, Debug)]
pub struct TernaryBlock {
    /// Quantised weights in {-1, 0, +1}. Stored as i8 for easy arithmetic.
    pub weights: [i8; BLOCK_SIZE],
    /// Per-block scale = mean(|w_float|) of the original block.
    pub scale: f32,
}

/// Quantise a slice of f32 weights (length exactly BLOCK_SIZE) into a TernaryBlock.
///
/// BST algorithm:
///   1. Compute scale = mean(|w[i]|)
///   2. quantise each w[i] with threshold = 0.5
pub fn ternarize_block(w: &[f32]) -> TernaryBlock {
    debug_assert_eq!(w.len(), BLOCK_SIZE);
    let scale: f32 = w.iter().map(|&v| v.abs()).sum::<f32>() / BLOCK_SIZE as f32;
    let mut weights = [0i8; BLOCK_SIZE];
    for (i, &v) in w.iter().enumerate() {
        weights[i] = ternarize(v, scale, 0.5);
    }
    TernaryBlock { weights, scale }
}

/// Dequantise a TernaryBlock back to f32.
pub fn dequantize_block(block: &TernaryBlock, out: &mut [f32]) {
    debug_assert_eq!(out.len(), BLOCK_SIZE);
    for i in 0..BLOCK_SIZE {
        out[i] = block.weights[i] as f32 * block.scale;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternarize_correct() {
        assert_eq!(ternarize(2.0, 1.0, 0.5), 1);
        assert_eq!(ternarize(-2.0, 1.0, 0.5), -1);
        assert_eq!(ternarize(0.3, 1.0, 0.5), 0);
    }

    #[test]
    fn block_scale_equals_mean_abs() {
        let w: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let block = ternarize_block(&w);
        let expected_scale: f32 = w.iter().map(|&v| v.abs()).sum::<f32>() / BLOCK_SIZE as f32;
        assert!((block.scale - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn dequantize_close_to_original() {
        // For weights with clear sign and magnitude > 0.5 * scale,
        // dequantize should be close to the original
        let w: Vec<f32> = (0..BLOCK_SIZE)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let block = ternarize_block(&w);
        let mut out = vec![0.0f32; BLOCK_SIZE];
        dequantize_block(&block, &mut out);
        // All should be ±scale (because all |w[i]| = 1.0 = mean = scale)
        for (i, (&orig, &recon)) in w.iter().zip(&out).enumerate() {
            assert_eq!(orig.signum(), recon.signum(), "sign mismatch at i={i}");
        }
    }

    #[test]
    fn all_zeros_produces_zero_scale() {
        let w = [0.0f32; BLOCK_SIZE];
        let block = ternarize_block(&w);
        assert_eq!(block.scale, 0.0);
        assert!(block.weights.iter().all(|&v| v == 0));
    }
}

//! SSM hidden state with O(1) memory growth.
//!
//! The hidden state h in R^N is stored in a single Vec<f32> that is
//! updated in-place at each token step. No history is accumulated —
//! this gives O(1) memory growth regardless of sequence length,
//! as required by spec §1.2 and §F (anti-scaling, bounded memory).
//!
//! # Novel technique: Recurrent State Normalization (RSN)
//!
//! To prevent state explosion during long sequences, we apply L2 norm
//! clamping to the hidden state after each update. This keeps the state
//! bounded while preserving the recurrent dynamics. The clamping is
//! applied only when the norm exceeds a threshold (default: 100.0),
//! scaling the state down proportionally.

use sophon_config::SSM_N;

// ---------------------------------------------------------------------------
// SsmState
// ---------------------------------------------------------------------------

/// Default L2 norm threshold for recurrent state normalization.
/// When the state norm exceeds this, it is scaled down proportionally.
pub const DEFAULT_NORM_THRESHOLD: f32 = 100.0;

/// In-place hidden state for one SSM layer.
#[derive(Clone, Debug)]
pub struct SsmState {
    /// h in R^N, updated in-place.
    pub h: Vec<f32>,
    /// L2 norm threshold for state normalization.
    pub norm_threshold: f32,
}

impl SsmState {
    /// Create a zero-initialised state with default norm threshold.
    pub fn new() -> Self {
        Self {
            h: vec![0.0f32; SSM_N],
            norm_threshold: DEFAULT_NORM_THRESHOLD,
        }
    }

    /// Create a state with custom norm threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            h: vec![0.0f32; SSM_N],
            norm_threshold: threshold,
        }
    }

    /// Reset state to zero (start of new sequence).
    #[inline]
    pub fn reset(&mut self) {
        for v in self.h.iter_mut() {
            *v = 0.0;
        }
    }

    /// Verify the state vector contains no NaN or Inf.
    pub fn is_valid(&self) -> bool {
        self.h.iter().all(|&v| v.is_finite())
    }

    /// Compute the L2 norm (Euclidean norm) of the state.
    pub fn l2_norm(&self) -> f32 {
        let sum_sq: f32 = self.h.iter().map(|&v| v * v).sum();
        sum_sq.sqrt()
    }

    /// Apply recurrent state normalization (RSN).
    ///
    /// If the L2 norm exceeds the threshold, scales the state down
    /// proportionally so that the new norm equals the threshold.
    /// This prevents state explosion during long sequences.
    #[inline]
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > self.norm_threshold && norm > 1e-12 {
            let scale = self.norm_threshold / norm;
            for v in self.h.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// Apply normalization with a custom threshold.
    #[inline]
    pub fn normalize_with_threshold(&mut self, threshold: f32) {
        let norm = self.l2_norm();
        if norm > threshold && norm > 1e-12 {
            let scale = threshold / norm;
            for v in self.h.iter_mut() {
                *v *= scale;
            }
        }
    }
}

impl Default for SsmState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_zero() {
        let s = SsmState::new();
        assert!(s.h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reset_zeros_state() {
        let mut s = SsmState::new();
        s.h[0] = 5.0;
        s.reset();
        assert!(s.h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn valid_after_init() {
        let s = SsmState::new();
        assert!(s.is_valid());
    }

    #[test]
    fn n_matches_constant() {
        let s = SsmState::new();
        assert_eq!(s.h.len(), SSM_N);
    }

    #[test]
    fn l2_norm_computed_correctly() {
        let mut s = SsmState::new();
        // Set state to [3, 4, 0, ...] which has norm 5
        s.h[0] = 3.0;
        s.h[1] = 4.0;
        assert!((s.l2_norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn zero_state_has_zero_norm() {
        let s = SsmState::new();
        assert_eq!(s.l2_norm(), 0.0);
    }

    #[test]
    fn normalization_preserves_direction() {
        let mut s = SsmState::with_threshold(10.0);
        // Set large values that exceed threshold
        for i in 0..SSM_N {
            s.h[i] = 100.0;
        }
        let before_norm = s.l2_norm();
        assert!(before_norm > s.norm_threshold);

        s.normalize();

        let after_norm = s.l2_norm();
        // After normalization, norm should equal threshold
        assert!((after_norm - s.norm_threshold).abs() < 1e-4);
        // Direction should be preserved (all values still equal)
        let first = s.h[0];
        assert!(s.h.iter().all(|&v| (v - first).abs() < 1e-5));
    }

    #[test]
    fn normalization_skips_small_states() {
        let mut s = SsmState::with_threshold(100.0);
        // Set small values
        s.h[0] = 1.0;
        s.h[1] = 2.0;
        let original: Vec<f32> = s.h.clone();

        s.normalize();

        // State should remain unchanged (below threshold)
        assert_eq!(s.h, original);
    }

    #[test]
    fn custom_threshold_works() {
        let mut s = SsmState::new();
        s.h[0] = 50.0;
        s.h[1] = 50.0;
        // Norm is ~70.71

        // Apply with high threshold - should not normalize
        s.normalize_with_threshold(100.0);
        assert!((s.h[0] - 50.0).abs() < 1e-5);

        // Reset and apply with low threshold
        s.h[0] = 50.0;
        s.h[1] = 50.0;
        s.normalize_with_threshold(10.0);
        // Should be scaled down
        assert!(s.h[0] < 50.0);
        assert!(s.h[1] < 50.0);
    }

    #[test]
    fn normalization_handles_edge_cases() {
        // Test with very small threshold (should still work)
        let mut s = SsmState::with_threshold(1e-10);
        s.h[0] = 1.0;
        s.normalize();
        // Should scale down to threshold
        assert!((s.l2_norm() - 1e-10).abs() < 1e-12);
    }
}

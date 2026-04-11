//! SSM hidden state with O(1) memory growth.
//!
//! The hidden state h in R^N is stored in a single Vec<f32> that is
//! updated in-place at each token step. No history is accumulated —
//! this gives O(1) memory growth regardless of sequence length,
//! as required by spec §1.2 and §F (anti-scaling, bounded memory).

use sophon_config::SSM_N;

// ---------------------------------------------------------------------------
// SsmState
// ---------------------------------------------------------------------------

/// In-place hidden state for one SSM layer.
#[derive(Clone, Debug)]
pub struct SsmState {
    /// h in R^N, updated in-place.
    pub h: Vec<f32>,
}

impl SsmState {
    /// Create a zero-initialised state.
    pub fn new() -> Self {
        Self {
            h: vec![0.0f32; SSM_N],
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
}

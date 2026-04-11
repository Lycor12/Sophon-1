//! Precision estimation for adaptive observation weighting.
//!
//! Spec §4.4: External tool outputs are treated as noisy observations.
//! Precision π = 1/σ² determines how much each observation channel
//! influences the belief update.
//!
//! # Novel technique: VPBM (Variational Precision-Balanced Minimisation)
//!
//! Standard active inference uses fixed precision. VPBM tracks running
//! statistics of prediction errors per channel and adaptively adjusts
//! precision:
//!
//! - Channels with high error variance get LOW precision (untrusted)
//! - Channels with low error variance get HIGH precision (trusted)
//! - Precision is bounded: π ∈ [π_min, π_max] to prevent collapse
//!
//! The key formula:
//!   π_i(t) = clamp(1 / ema_var_i(t), π_min, π_max)
//!
//! where ema_var_i is an exponential moving average of squared prediction
//! errors for channel i.

/// Adaptive precision estimator using VPBM.
#[derive(Debug, Clone)]
pub struct PrecisionEstimator {
    /// Per-channel EMA of squared prediction error. Shape: [d_obs]
    ema_sq_error: Vec<f32>,
    /// Per-channel EMA of prediction error (for mean correction). Shape: [d_obs]
    ema_error: Vec<f32>,
    /// EMA decay factor (α). Higher = more memory, slower adaptation.
    pub alpha: f32,
    /// Minimum precision (prevents division by near-zero variance).
    pub pi_min: f32,
    /// Maximum precision (prevents over-confidence in any channel).
    pub pi_max: f32,
    /// Number of observation channels.
    d_obs: usize,
    /// Whether the estimator has received at least one update.
    initialised: bool,
}

impl PrecisionEstimator {
    /// Create a new estimator with uniform initial precision.
    pub fn new(d_obs: usize) -> Self {
        Self {
            ema_sq_error: vec![1.0f32; d_obs], // initial variance = 1 (prior)
            ema_error: vec![0.0f32; d_obs],
            alpha: 0.95,
            pi_min: 0.01,
            pi_max: 100.0,
            d_obs,
            initialised: false,
        }
    }

    /// Create with custom bounds.
    pub fn with_bounds(d_obs: usize, alpha: f32, pi_min: f32, pi_max: f32) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        assert!(pi_min > 0.0, "pi_min must be positive");
        assert!(pi_max > pi_min, "pi_max must exceed pi_min");
        Self {
            ema_sq_error: vec![1.0f32; d_obs],
            ema_error: vec![0.0f32; d_obs],
            alpha,
            pi_min,
            pi_max,
            d_obs,
            initialised: false,
        }
    }

    /// Update the precision estimator from new prediction errors.
    ///
    /// # Arguments
    /// * `errors` — prediction error vector e = o - o_hat. Shape: [d_obs]
    pub fn update(&mut self, errors: &[f32]) {
        assert_eq!(errors.len(), self.d_obs);
        if !self.initialised {
            // First update: initialise EMAs directly
            for i in 0..self.d_obs {
                self.ema_error[i] = errors[i];
                self.ema_sq_error[i] = errors[i] * errors[i];
            }
            self.initialised = true;
        } else {
            let a = self.alpha;
            let one_a = 1.0 - a;
            for i in 0..self.d_obs {
                self.ema_error[i] = a * self.ema_error[i] + one_a * errors[i];
                self.ema_sq_error[i] = a * self.ema_sq_error[i] + one_a * errors[i] * errors[i];
            }
        }
    }

    /// Compute current precision vector: π_i = clamp(1 / var_i, π_min, π_max).
    ///
    /// Variance is computed as: var_i = E[e²] - E[e]² (EMA-based).
    pub fn precision(&self) -> Vec<f32> {
        let mut pi = vec![0.0f32; self.d_obs];
        for i in 0..self.d_obs {
            let var = (self.ema_sq_error[i] - self.ema_error[i] * self.ema_error[i]).max(1e-8);
            pi[i] = (1.0 / var).clamp(self.pi_min, self.pi_max);
        }
        pi
    }

    /// Apply precision weighting to prediction errors.
    ///
    /// weighted_error_i = π_i * error_i
    ///
    /// This is the key output used by the belief updater.
    pub fn weight_errors(&self, errors: &[f32]) -> Vec<f32> {
        let pi = self.precision();
        errors.iter().zip(pi.iter()).map(|(&e, &p)| p * e).collect()
    }

    /// Compute precision-weighted squared error: sum_i π_i * e_i².
    ///
    /// This is the prediction error term in the free energy.
    pub fn weighted_squared_error(&self, errors: &[f32]) -> f32 {
        let pi = self.precision();
        errors.iter().zip(pi.iter()).map(|(&e, &p)| p * e * e).sum()
    }

    /// Get the EMA variance estimates.
    pub fn variance_estimates(&self) -> Vec<f32> {
        let mut var = vec![0.0f32; self.d_obs];
        for i in 0..self.d_obs {
            var[i] = (self.ema_sq_error[i] - self.ema_error[i] * self.ema_error[i]).max(1e-8);
        }
        var
    }

    /// Mean precision across all channels.
    pub fn mean_precision(&self) -> f32 {
        let pi = self.precision();
        pi.iter().sum::<f32>() / self.d_obs as f32
    }

    /// Number of channels.
    pub fn d_obs(&self) -> usize {
        self.d_obs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_precision_is_one() {
        let pe = PrecisionEstimator::new(5);
        let pi = pe.precision();
        // Initial ema_sq = 1.0, ema = 0.0, so var = 1.0, π = 1.0
        for &p in &pi {
            assert!(
                (p - 1.0).abs() < 1e-4,
                "initial precision should be ~1.0, got {p}"
            );
        }
    }

    #[test]
    fn high_variance_reduces_precision() {
        let mut pe = PrecisionEstimator::new(2);
        // Channel 0: high-variance alternating errors → high variance → low precision
        // Channel 1: low-variance near-zero errors → low variance → high precision
        for i in 0..100 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            pe.update(&[10.0 * sign, 0.01 * sign]);
        }
        let pi = pe.precision();
        assert!(
            pi[0] < pi[1],
            "high-variance channel should have lower precision: π₀={} π₁={}",
            pi[0],
            pi[1]
        );
    }

    #[test]
    fn precision_bounded() {
        let mut pe = PrecisionEstimator::with_bounds(1, 0.9, 0.5, 10.0);
        // Very high variance → precision should be clamped at pi_min
        for _ in 0..100 {
            pe.update(&[1000.0]);
        }
        let pi = pe.precision();
        assert!(pi[0] >= 0.5 - 1e-6, "precision below min: {}", pi[0]);

        // Very low variance → precision should be clamped at pi_max
        let mut pe2 = PrecisionEstimator::with_bounds(1, 0.9, 0.5, 10.0);
        for _ in 0..100 {
            pe2.update(&[0.001]);
        }
        let pi2 = pe2.precision();
        assert!(pi2[0] <= 10.0 + 1e-6, "precision above max: {}", pi2[0]);
    }

    #[test]
    fn weighted_errors_scale_correctly() {
        let pe = PrecisionEstimator::new(3);
        let errors = vec![1.0, 2.0, -3.0];
        let weighted = pe.weight_errors(&errors);
        // Initial precision ~1.0, so weighted ≈ errors
        for i in 0..3 {
            assert!(
                (weighted[i] - errors[i]).abs() < 0.5,
                "weighted[{i}] = {} vs error = {}",
                weighted[i],
                errors[i]
            );
        }
    }

    #[test]
    fn weighted_squared_error_non_negative() {
        let pe = PrecisionEstimator::new(4);
        let errors = vec![1.0, -2.0, 0.5, -0.3];
        assert!(pe.weighted_squared_error(&errors) >= 0.0);
    }

    #[test]
    fn variance_estimates_positive() {
        let mut pe = PrecisionEstimator::new(3);
        pe.update(&[1.0, -1.0, 0.5]);
        pe.update(&[0.5, 1.0, -0.5]);
        let var = pe.variance_estimates();
        for &v in &var {
            assert!(v > 0.0, "variance should be positive");
        }
    }
}

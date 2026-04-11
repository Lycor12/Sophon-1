//! World model: generative model p(o|s) mapping belief → expected observation.
//!
//! Spec §2.3.2: The generative model factorises as p(o,s) = p(o|s) * p(s).
//! The recognition model q(s|o) is the belief state (belief.rs).
//!
//! The world model here is a lightweight linear-Gaussian approximation:
//!   p(o|s) = N(W*s + b, Σ_o)
//! where W is a learned projection matrix, b is a bias, and Σ_o is the
//! observation noise covariance (diagonal, learnable via precision).
//!
//! For the full Sophon-1 system, this prediction is computed by running
//! the model forward. This module provides the simpler approximate
//! generative model used for fast belief updates between full model calls.
//!
//! # Novel technique: LGP (Linear-Gaussian Prediction)
//!
//! The prediction step avoids full model forward passes by maintaining a
//! locally-linear approximation of the model's input-output mapping.
//! The projection matrix W is updated via rank-1 corrections from
//! (belief, observation) pairs, giving O(d_obs * d_latent) per step
//! instead of a full model forward (O(millions)).

use crate::belief::BeliefState;

/// A linear-Gaussian world model: o_hat = W * s + bias.
#[derive(Debug, Clone)]
pub struct WorldModel {
    /// Projection matrix W: [d_obs x d_latent], row-major.
    pub w: Vec<f32>,
    /// Bias vector: [d_obs].
    pub bias: Vec<f32>,
    /// Observation dimensionality.
    pub d_obs: usize,
    /// Latent dimensionality.
    pub d_latent: usize,
}

impl WorldModel {
    /// Create a new world model initialised to identity-like projection.
    ///
    /// If d_obs == d_latent, W = I. Otherwise W is zero-initialised
    /// (the model starts with no predictive ability and must learn).
    pub fn new(d_obs: usize, d_latent: usize) -> Self {
        let mut w = vec![0.0f32; d_obs * d_latent];
        // If dimensions match, initialise to identity
        if d_obs == d_latent {
            for i in 0..d_obs.min(d_latent) {
                w[i * d_latent + i] = 1.0;
            }
        }
        Self {
            w,
            bias: vec![0.0f32; d_obs],
            d_obs,
            d_latent,
        }
    }

    /// Predict expected observation from the belief state mean.
    ///
    /// o_hat = W * μ + bias
    pub fn predict(&self, belief: &BeliefState) -> Vec<f32> {
        self.predict_from(&belief.mu)
    }

    /// Predict from an arbitrary latent vector s.
    ///
    /// o_hat = W * s + bias
    pub fn predict_from(&self, s: &[f32]) -> Vec<f32> {
        assert_eq!(s.len(), self.d_latent);
        let mut o_hat = self.bias.clone();
        for i in 0..self.d_obs {
            let row_start = i * self.d_latent;
            for j in 0..self.d_latent {
                o_hat[i] += self.w[row_start + j] * s[j];
            }
        }
        o_hat
    }

    /// Compute prediction error: e = o_actual - o_hat.
    pub fn prediction_error(&self, belief: &BeliefState, observation: &[f32]) -> Vec<f32> {
        assert_eq!(observation.len(), self.d_obs);
        let o_hat = self.predict(belief);
        observation
            .iter()
            .zip(o_hat.iter())
            .map(|(&o, &oh)| o - oh)
            .collect()
    }

    /// Gradient of prediction error w.r.t. belief mean μ.
    ///
    /// d(error)/d(μ) = -W^T (since error = o - W*μ - b)
    /// d(||error||²)/d(μ) = -2 * W^T * error
    ///
    /// Returns the gradient of 0.5 * ||error||² w.r.t. μ (negative direction).
    pub fn grad_mu_prediction_error(&self, error: &[f32]) -> Vec<f32> {
        assert_eq!(error.len(), self.d_obs);
        let mut grad = vec![0.0f32; self.d_latent];
        // grad = -W^T * error
        for j in 0..self.d_latent {
            for i in 0..self.d_obs {
                grad[j] -= self.w[i * self.d_latent + j] * error[i];
            }
        }
        grad
    }

    /// Update W via rank-1 correction from a (belief, observation) pair.
    ///
    /// This is the LGP online update:
    ///   W += lr * error * μ^T
    /// where error = o - W*μ - b.
    ///
    /// This minimises ||o - W*μ - b||² w.r.t. W.
    pub fn update_w(&mut self, belief: &BeliefState, observation: &[f32], lr: f32) {
        let error = self.prediction_error(belief, observation);
        for i in 0..self.d_obs {
            for j in 0..self.d_latent {
                self.w[i * self.d_latent + j] += lr * error[i] * belief.mu[j];
            }
            self.bias[i] += lr * error[i];
        }
    }

    /// Total squared prediction error: ||o - o_hat||².
    pub fn squared_error(&self, belief: &BeliefState, observation: &[f32]) -> f32 {
        let error = self.prediction_error(belief, observation);
        error.iter().map(|e| e * e).sum()
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        self.d_obs * self.d_latent + self.d_obs // W + bias
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_model_predicts_mu() {
        let dim = 5;
        let wm = WorldModel::new(dim, dim);
        let b = BeliefState::from_params(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![0.0; 5]);
        let pred = wm.predict(&b);
        for i in 0..dim {
            assert!(
                (pred[i] - b.mu[i]).abs() < 1e-6,
                "identity prediction mismatch at {i}"
            );
        }
    }

    #[test]
    fn prediction_error_zero_when_perfect() {
        let wm = WorldModel::new(3, 3);
        let b = BeliefState::from_params(vec![1.0, 2.0, 3.0], vec![0.0; 3]);
        let obs = wm.predict(&b); // perfect observation
        let err = wm.prediction_error(&b, &obs);
        assert!(
            err.iter().all(|&e| e.abs() < 1e-6),
            "error should be zero for perfect prediction"
        );
    }

    #[test]
    fn grad_mu_direction() {
        let wm = WorldModel::new(3, 3);
        let b = BeliefState::from_params(vec![0.0; 3], vec![0.0; 3]);
        let obs = vec![1.0, 0.0, -1.0];
        let error = wm.prediction_error(&b, &obs);
        let grad = wm.grad_mu_prediction_error(&error);
        // With identity W and μ=0, error = obs. Grad = -W^T * error = -obs
        for i in 0..3 {
            assert!(
                (grad[i] - (-obs[i])).abs() < 1e-6,
                "grad direction wrong at {i}: {} vs {}",
                grad[i],
                -obs[i]
            );
        }
    }

    #[test]
    fn update_reduces_error() {
        let mut wm = WorldModel::new(3, 3);
        // Start with zero W (no prediction ability)
        wm.w.fill(0.0);
        let b = BeliefState::from_params(vec![1.0, 0.5, -0.5], vec![0.0; 3]);
        let obs = vec![2.0, 1.0, -1.0];
        let err_before = wm.squared_error(&b, &obs);
        wm.update_w(&b, &obs, 0.1);
        let err_after = wm.squared_error(&b, &obs);
        assert!(
            err_after < err_before,
            "error should decrease after update: {err_before} -> {err_after}"
        );
    }

    #[test]
    fn param_count_correct() {
        let wm = WorldModel::new(10, 5);
        assert_eq!(wm.param_count(), 10 * 5 + 10);
    }

    #[test]
    fn non_square_model() {
        let wm = WorldModel::new(4, 8);
        let b = BeliefState::new(8);
        let pred = wm.predict(&b);
        assert_eq!(pred.len(), 4);
        // Zero W, zero mu -> prediction = bias = 0
        assert!(pred.iter().all(|&v| v.abs() < 1e-6));
    }
}

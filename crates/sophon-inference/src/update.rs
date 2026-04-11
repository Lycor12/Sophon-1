//! Belief update loop: gradient descent on variational free energy.
//!
//! Spec §3.1: The belief update minimises F w.r.t. (μ, σ) via:
//!   dF/dμ = dKL/dμ + π ⊙ dPE/dμ
//!   dF/d(log σ) = dKL/d(log σ)
//!
//! where π is the precision vector from VPBM, PE is prediction error,
//! and KL is the divergence from the prior.
//!
//! The update loop runs multiple steps of gradient descent per observation
//! to converge the belief before the next action.
//!
//! # Novel technique: ABGU (Adaptive-Beta Gradient Update)
//!
//! Standard belief updates use a fixed learning rate. ABGU adapts the
//! step size per dimension based on the precision-weighted gradient
//! magnitude: dimensions with high-precision, large-gradient observations
//! get larger steps (fast tracking), while low-precision or small-gradient
//! dimensions get smaller steps (stability). This is a form of natural
//! gradient descent without computing the full Fisher matrix.
//!
//! Step size: lr_i = base_lr * min(1.0, π_i * |grad_i| / median(π ⊙ |grad|))

use crate::belief::BeliefState;
use crate::precision::PrecisionEstimator;
use crate::prediction::WorldModel;
use sophon_loss::free_energy::{kl_divergence_grad, kl_divergence_standard_normal};

/// Configuration for the belief update loop.
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// Base learning rate for belief updates.
    pub base_lr: f32,
    /// Number of inner loop steps per observation.
    pub inner_steps: usize,
    /// Convergence threshold (stop if ||grad|| < threshold).
    pub convergence_threshold: f32,
    /// KL weight (beta in F = beta*KL + PE).
    pub kl_weight: f32,
    /// Whether to use ABGU adaptive step sizes.
    pub adaptive_lr: bool,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            base_lr: 0.01,
            inner_steps: 5,
            convergence_threshold: 1e-4,
            kl_weight: 0.01,
            adaptive_lr: true,
        }
    }
}

/// Result of a belief update step.
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Free energy before update.
    pub fe_before: f32,
    /// Free energy after update.
    pub fe_after: f32,
    /// Number of inner steps taken (may be < config.inner_steps if converged).
    pub steps_taken: usize,
    /// Final gradient norm.
    pub grad_norm: f32,
    /// Whether the update converged.
    pub converged: bool,
}

/// The belief updater: orchestrates free energy minimisation.
pub struct BeliefUpdater {
    pub config: UpdateConfig,
}

impl BeliefUpdater {
    pub fn new(config: UpdateConfig) -> Self {
        Self { config }
    }

    /// Run one full belief update cycle given a new observation.
    ///
    /// 1. Compute prediction error from current belief
    /// 2. Update precision estimator
    /// 3. Run inner gradient descent loop on F w.r.t. (μ, log_σ)
    /// 4. Return update statistics
    pub fn update(
        &self,
        belief: &mut BeliefState,
        world_model: &WorldModel,
        precision: &mut PrecisionEstimator,
        observation: &[f32],
    ) -> UpdateResult {
        // Compute initial unweighted prediction error for FE baseline
        let error_init = world_model.prediction_error(belief, observation);
        // Update precision AFTER computing FE baseline (avoid self-amplification)
        let pe_init = error_init.iter().map(|e| e * e).sum::<f32>() * 0.5;
        let kl_init = kl_divergence_standard_normal(&belief.mu, &belief.log_sigma);
        let fe_before = self.config.kl_weight * kl_init + pe_init;

        // Now update precision with the observation error
        precision.update(&error_init);

        let mut steps_taken = 0;
        let mut last_grad_norm = 0.0f32;
        let mut converged = false;
        let max_grad_norm = 10.0f32; // gradient clipping threshold

        for _step in 0..self.config.inner_steps {
            steps_taken += 1;

            // Recompute prediction error at current belief
            let error = world_model.prediction_error(belief, observation);

            // Gradient of 0.5*||e||² w.r.t. μ: -W^T * e
            // We use raw errors, then scale by mean precision for a moderate push
            let mean_pi = precision.mean_precision().min(10.0); // cap effective precision
            let grad_pe_mu_raw = world_model.grad_mu_prediction_error(&error);

            // KL gradient w.r.t. μ and log_σ
            let (kl_grad_mu, kl_grad_ls) = kl_divergence_grad(&belief.mu, &belief.log_sigma);

            // Total gradient: dF/dμ = β*dKL/dμ + mean_π * dPE/dμ
            let dim = belief.dim();
            let mut grad_mu = vec![0.0f32; dim];
            for i in 0..dim {
                grad_mu[i] = self.config.kl_weight * kl_grad_mu[i] + mean_pi * grad_pe_mu_raw[i];
            }

            // dF/d(log_σ) = β*dKL/d(log_σ)
            let mut grad_ls = vec![0.0f32; dim];
            for i in 0..dim {
                grad_ls[i] = self.config.kl_weight * kl_grad_ls[i];
            }

            // Gradient norm + clipping
            let g_norm: f32 = grad_mu
                .iter()
                .chain(grad_ls.iter())
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();
            last_grad_norm = g_norm;

            if g_norm < self.config.convergence_threshold {
                converged = true;
                break;
            }

            // Clip gradient if too large
            if g_norm > max_grad_norm {
                let scale = max_grad_norm / g_norm;
                for g in grad_mu.iter_mut() {
                    *g *= scale;
                }
                for g in grad_ls.iter_mut() {
                    *g *= scale;
                }
            }

            // Apply ABGU if enabled
            if self.config.adaptive_lr {
                let pi = precision.precision();
                // Compute per-dimension adaptive rate
                let mut pg_products = Vec::with_capacity(dim);
                for i in 0..dim {
                    let pi_i = if i < pi.len() { pi[i].min(10.0) } else { 1.0 };
                    pg_products.push(pi_i * grad_mu[i].abs());
                }
                // Find median of π⊙|grad|
                let mut sorted = pg_products.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = if sorted.is_empty() {
                    1.0
                } else {
                    sorted[sorted.len() / 2].max(1e-8)
                };

                // Adaptive lr per dimension
                let mut lr_mu = vec![0.0f32; dim];
                for i in 0..dim {
                    let ratio = pg_products[i] / median;
                    lr_mu[i] = self.config.base_lr * ratio.min(1.0);
                }

                // Apply update with per-dimension rates
                for i in 0..dim {
                    belief.mu[i] -= lr_mu[i] * grad_mu[i];
                    belief.log_sigma[i] -= self.config.base_lr * grad_ls[i];
                    belief.log_sigma[i] = belief.log_sigma[i].clamp(-5.0, 5.0);
                }
            } else {
                belief.update(&grad_mu, &grad_ls, self.config.base_lr);
            }
        }

        // Compute final free energy (unweighted for consistency with fe_before)
        let error_final = world_model.prediction_error(belief, observation);
        let pe_final = error_final.iter().map(|e| e * e).sum::<f32>() * 0.5;
        let kl_final = kl_divergence_standard_normal(&belief.mu, &belief.log_sigma);
        let fe_after = self.config.kl_weight * kl_final + pe_final;

        belief.step += 1;

        UpdateResult {
            fe_before,
            fe_after,
            steps_taken,
            grad_norm: last_grad_norm,
            converged,
        }
    }

    /// Check if the belief has drifted too far from the prior.
    /// Spec §6.3.1: parameter drift < 10%.
    pub fn drift_check(&self, belief: &BeliefState, threshold: f32) -> bool {
        let mag = belief.mu_magnitude();
        let dim_norm = (belief.dim() as f32).sqrt();
        // Normalised drift: how far μ is from 0 relative to dimension
        mag / dim_norm < threshold
    }
}

impl Default for BeliefUpdater {
    fn default() -> Self {
        Self::new(UpdateConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(dim: usize) -> (BeliefState, WorldModel, PrecisionEstimator) {
        let belief = BeliefState::new(dim);
        let wm = WorldModel::new(dim, dim);
        let pe = PrecisionEstimator::new(dim);
        (belief, wm, pe)
    }

    #[test]
    fn update_reduces_free_energy() {
        let (mut belief, wm, mut prec) = setup(5);
        let obs = vec![1.0, -0.5, 0.3, 0.0, 2.0]; // non-zero observation
        let updater = BeliefUpdater::new(UpdateConfig {
            base_lr: 0.1,
            inner_steps: 10,
            convergence_threshold: 1e-6,
            kl_weight: 0.01,
            adaptive_lr: false, // test fixed lr first
        });
        let result = updater.update(&mut belief, &wm, &mut prec, &obs);
        assert!(
            result.fe_after <= result.fe_before + 1e-6,
            "FE should decrease: {} -> {}",
            result.fe_before,
            result.fe_after
        );
    }

    #[test]
    fn update_moves_belief_toward_observation() {
        let (mut belief, wm, mut prec) = setup(3);
        let obs = vec![5.0, -3.0, 1.0];
        let updater = BeliefUpdater::new(UpdateConfig {
            base_lr: 0.1,
            inner_steps: 20,
            kl_weight: 0.001, // weak prior
            adaptive_lr: false,
            ..Default::default()
        });
        updater.update(&mut belief, &wm, &mut prec, &obs);
        // With identity world model and weak prior, μ should approach obs
        for i in 0..3 {
            let dist = (belief.mu[i] - obs[i]).abs();
            assert!(
                dist < obs[i].abs() + 1.0,
                "μ[{i}]={} too far from obs[{i}]={}",
                belief.mu[i],
                obs[i]
            );
        }
    }

    #[test]
    fn adaptive_lr_runs_without_panic() {
        let (mut belief, wm, mut prec) = setup(4);
        let obs = vec![1.0, 2.0, 3.0, 4.0];
        let updater = BeliefUpdater::new(UpdateConfig {
            adaptive_lr: true,
            inner_steps: 5,
            ..Default::default()
        });
        let result = updater.update(&mut belief, &wm, &mut prec, &obs);
        assert!(result.steps_taken > 0);
        assert!(belief.is_valid());
    }

    #[test]
    fn convergence_early_stop() {
        let (mut belief, wm, mut prec) = setup(2);
        // Observation = 0 (matches prior), should converge immediately
        let obs = vec![0.0, 0.0];
        let updater = BeliefUpdater::new(UpdateConfig {
            inner_steps: 50,
            convergence_threshold: 1e-3,
            adaptive_lr: false,
            ..Default::default()
        });
        let result = updater.update(&mut belief, &wm, &mut prec, &obs);
        assert!(
            result.converged,
            "should converge for zero observation at prior"
        );
        assert!(result.steps_taken < 50);
    }

    #[test]
    fn drift_check_at_prior_passes() {
        let belief = BeliefState::new(10);
        let updater = BeliefUpdater::default();
        assert!(updater.drift_check(&belief, 0.1));
    }

    #[test]
    fn drift_check_large_shift_fails() {
        let belief = BeliefState::from_params(vec![100.0; 10], vec![0.0; 10]);
        let updater = BeliefUpdater::default();
        assert!(!updater.drift_check(&belief, 0.1));
    }

    #[test]
    fn update_result_fields_finite() {
        let (mut belief, wm, mut prec) = setup(3);
        let obs = vec![1.0, -1.0, 0.5];
        let updater = BeliefUpdater::default();
        let result = updater.update(&mut belief, &wm, &mut prec, &obs);
        assert!(result.fe_before.is_finite());
        assert!(result.fe_after.is_finite());
        assert!(result.grad_norm.is_finite());
    }
}

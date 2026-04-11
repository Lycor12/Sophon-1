//! Self-improvement loop: hypothesis → simulate → verify → update.
//!
//! Spec §3.2: The system improves by:
//! 1. Generating hypotheses from the current belief q(s)
//! 2. Mentally simulating outcomes using the world model
//! 3. Verifying predictions against actual observations or formal proofs
//! 4. Updating the belief and model parameters based on verification
//!
//! This module implements the orchestration of this cycle.
//!
//! # Novel technique: HVSC (Hypothesis-Verified Selective Consolidation)
//!
//! Not all hypotheses are equally valuable. HVSC scores hypotheses by their
//! expected information gain (EIG): how much the hypothesis would reduce
//! uncertainty if verified. High-EIG hypotheses are prioritised for
//! verification, and only verified hypotheses update the permanent
//! knowledge base. This prevents the system from wasting verification
//! resources on trivially true or trivially false predictions.

use crate::belief::BeliefState;
use crate::prediction::WorldModel;
use sophon_core::rng::Rng;

/// A hypothesis generated from the belief state.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Sampled latent state from q(s).
    pub latent: Vec<f32>,
    /// Predicted observation from the world model.
    pub prediction: Vec<f32>,
    /// Expected information gain (EIG) score.
    pub eig: f32,
    /// Whether this hypothesis has been verified.
    pub verified: bool,
    /// Actual observation (filled in after verification).
    pub actual: Option<Vec<f32>>,
    /// Prediction error (filled in after verification).
    pub error: Option<f32>,
}

/// Configuration for the self-improvement loop.
#[derive(Debug, Clone)]
pub struct ImprovementConfig {
    /// Number of hypotheses to generate per cycle.
    pub n_hypotheses: usize,
    /// Top-K hypotheses to verify (ranked by EIG).
    pub top_k: usize,
    /// Temperature for hypothesis sampling (scales σ).
    pub temperature: f32,
    /// Minimum EIG threshold to bother verifying.
    pub eig_threshold: f32,
    /// Maximum consecutive failed hypotheses before resetting belief.
    pub max_failures: usize,
}

impl Default for ImprovementConfig {
    fn default() -> Self {
        Self {
            n_hypotheses: 10,
            top_k: 3,
            temperature: 1.0,
            eig_threshold: 0.01,
            max_failures: 20,
        }
    }
}

/// Result of one self-improvement cycle.
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// Hypotheses generated (sorted by EIG, descending).
    pub hypotheses: Vec<Hypothesis>,
    /// Number verified.
    pub n_verified: usize,
    /// Number that passed verification.
    pub n_passed: usize,
    /// Mean prediction error of verified hypotheses.
    pub mean_error: f32,
    /// Whether the cycle resulted in a belief update.
    pub belief_updated: bool,
}

/// The self-improvement loop orchestrator.
pub struct SelfImprovementLoop {
    pub config: ImprovementConfig,
    /// Consecutive failure counter.
    failure_count: usize,
    /// Total cycles run.
    pub total_cycles: u64,
    /// Total hypotheses verified across all cycles.
    pub total_verified: u64,
    /// Total hypotheses that passed verification.
    pub total_passed: u64,
}

impl SelfImprovementLoop {
    pub fn new(config: ImprovementConfig) -> Self {
        Self {
            config,
            failure_count: 0,
            total_cycles: 0,
            total_verified: 0,
            total_passed: 0,
        }
    }

    /// Generate hypotheses from the current belief.
    ///
    /// Each hypothesis is a sample from q(s) passed through the world model
    /// to produce an expected observation, scored by expected information gain.
    pub fn generate_hypotheses(
        &self,
        belief: &BeliefState,
        world_model: &WorldModel,
        rng: &mut Rng,
    ) -> Vec<Hypothesis> {
        let mut hypotheses = Vec::with_capacity(self.config.n_hypotheses);

        for _ in 0..self.config.n_hypotheses {
            // Sample from q(s) with temperature scaling
            let mut latent = vec![0.0f32; belief.dim()];
            for i in 0..belief.dim() {
                let eps = rng.next_normal(0.0, 1.0);
                let sigma = belief.log_sigma[i].exp() * self.config.temperature;
                latent[i] = belief.mu[i] + sigma * eps;
            }

            // Predict observation
            let prediction = world_model.predict_from(&latent);

            // Compute EIG: expected information gain
            // Approximation: EIG ≈ ||prediction - world_model(μ)||²
            // (how different is this prediction from the MAP prediction?)
            let map_pred = world_model.predict(belief);
            let eig: f32 = prediction
                .iter()
                .zip(map_pred.iter())
                .map(|(&p, &m)| (p - m) * (p - m))
                .sum();

            hypotheses.push(Hypothesis {
                latent,
                prediction,
                eig,
                verified: false,
                actual: None,
                error: None,
            });
        }

        // Sort by EIG descending
        hypotheses.sort_by(|a, b| {
            b.eig
                .partial_cmp(&a.eig)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        hypotheses
    }

    /// Select top-K hypotheses above EIG threshold for verification.
    pub fn select_for_verification(&self, hypotheses: &[Hypothesis]) -> Vec<usize> {
        hypotheses
            .iter()
            .enumerate()
            .filter(|(_, h)| h.eig >= self.config.eig_threshold)
            .take(self.config.top_k)
            .map(|(i, _)| i)
            .collect()
    }

    /// Verify a hypothesis against an actual observation.
    ///
    /// Returns the squared prediction error.
    pub fn verify_hypothesis(
        &self,
        hypothesis: &mut Hypothesis,
        actual_observation: &[f32],
    ) -> f32 {
        let error: f32 = hypothesis
            .prediction
            .iter()
            .zip(actual_observation.iter())
            .map(|(&p, &a)| (p - a) * (p - a))
            .sum();
        hypothesis.verified = true;
        hypothesis.actual = Some(actual_observation.to_vec());
        hypothesis.error = Some(error);
        error
    }

    /// Run a complete improvement cycle.
    ///
    /// This is the core loop:
    /// 1. Generate N hypotheses from current belief
    /// 2. Select top-K by EIG
    /// 3. Verify selected hypotheses (caller provides observations)
    /// 4. Update belief based on verification results
    ///
    /// The `observer` callback is called for each selected hypothesis to
    /// obtain the actual observation. In a full system, this would involve
    /// executing actions and observing results. Here it's a callback for
    /// testability.
    pub fn run_cycle<F>(
        &mut self,
        belief: &mut BeliefState,
        world_model: &mut WorldModel,
        rng: &mut Rng,
        observer: F,
    ) -> CycleResult
    where
        F: Fn(&[f32]) -> Vec<f32>, // maps predicted obs → actual obs
    {
        let mut hypotheses = self.generate_hypotheses(belief, world_model, rng);
        let selected = self.select_for_verification(&hypotheses);

        let mut n_verified = 0;
        let mut n_passed = 0;
        let mut total_error = 0.0f32;
        let verification_threshold = 1.0; // MSE threshold for "pass"

        for &idx in &selected {
            let actual = observer(&hypotheses[idx].prediction);
            let error = self.verify_hypothesis(&mut hypotheses[idx], &actual);
            n_verified += 1;
            total_error += error;

            let mse = error / hypotheses[idx].prediction.len() as f32;
            if mse < verification_threshold {
                n_passed += 1;
                // Update world model from verified (prediction, actual) pair
                // Use the sampled latent as if it were the true state
                let temp_belief = BeliefState::from_params(
                    hypotheses[idx].latent.clone(),
                    belief.log_sigma.clone(),
                );
                world_model.update_w(&temp_belief, &actual, 0.01);
            }
        }

        let mean_error = if n_verified > 0 {
            total_error / n_verified as f32
        } else {
            0.0
        };

        let belief_updated = n_passed > 0;
        if belief_updated {
            self.failure_count = 0;
        } else if n_verified > 0 {
            self.failure_count += 1;
        }

        // If too many failures, contract uncertainty (belief becomes more diffuse)
        if self.failure_count >= self.config.max_failures {
            // Increase uncertainty to explore more
            for ls in &mut belief.log_sigma {
                *ls = (*ls + 0.1).min(5.0);
            }
            self.failure_count = 0;
        }

        self.total_cycles += 1;
        self.total_verified += n_verified as u64;
        self.total_passed += n_passed as u64;

        CycleResult {
            hypotheses,
            n_verified,
            n_passed,
            mean_error,
            belief_updated,
        }
    }

    /// Verification success rate across all cycles.
    pub fn success_rate(&self) -> f32 {
        if self.total_verified == 0 {
            0.0
        } else {
            self.total_passed as f32 / self.total_verified as f32
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.failure_count = 0;
        self.total_cycles = 0;
        self.total_verified = 0;
        self.total_passed = 0;
    }
}

impl Default for SelfImprovementLoop {
    fn default() -> Self {
        Self::new(ImprovementConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_hypotheses_count() {
        let belief = BeliefState::new(5);
        let wm = WorldModel::new(5, 5);
        let mut rng = Rng::new(42);
        let config = ImprovementConfig {
            n_hypotheses: 8,
            ..Default::default()
        };
        let sil = SelfImprovementLoop::new(config);
        let hyps = sil.generate_hypotheses(&belief, &wm, &mut rng);
        assert_eq!(hyps.len(), 8);
    }

    #[test]
    fn hypotheses_sorted_by_eig_descending() {
        let belief = BeliefState::from_params(vec![1.0; 3], vec![0.5; 3]);
        let wm = WorldModel::new(3, 3);
        let mut rng = Rng::new(123);
        let sil = SelfImprovementLoop::default();
        let hyps = sil.generate_hypotheses(&belief, &wm, &mut rng);
        for i in 1..hyps.len() {
            assert!(
                hyps[i - 1].eig >= hyps[i].eig,
                "hypotheses not sorted: {} < {}",
                hyps[i - 1].eig,
                hyps[i].eig
            );
        }
    }

    #[test]
    fn select_respects_top_k_and_threshold() {
        let config = ImprovementConfig {
            n_hypotheses: 10,
            top_k: 3,
            eig_threshold: 0.5,
            ..Default::default()
        };
        let sil = SelfImprovementLoop::new(config);

        let hyps: Vec<Hypothesis> = (0..10)
            .map(|i| Hypothesis {
                latent: vec![0.0],
                prediction: vec![0.0],
                eig: i as f32 * 0.2, // 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, ...
                verified: false,
                actual: None,
                error: None,
            })
            .rev() // sorted descending
            .collect();

        let selected = sil.select_for_verification(&hyps);
        assert!(selected.len() <= 3);
        // All selected should have EIG >= 0.5
        for &idx in &selected {
            assert!(hyps[idx].eig >= 0.5);
        }
    }

    #[test]
    fn verify_hypothesis_fills_fields() {
        let sil = SelfImprovementLoop::default();
        let mut h = Hypothesis {
            latent: vec![1.0],
            prediction: vec![2.0],
            eig: 1.0,
            verified: false,
            actual: None,
            error: None,
        };
        let err = sil.verify_hypothesis(&mut h, &[1.5]);
        assert!(h.verified);
        assert!(h.actual.is_some());
        assert!(h.error.is_some());
        assert!((err - 0.25).abs() < 1e-6); // (2.0 - 1.5)^2
    }

    #[test]
    fn run_cycle_identity_world() {
        let mut belief = BeliefState::new(3);
        let mut wm = WorldModel::new(3, 3);
        let mut rng = Rng::new(77);
        let mut sil = SelfImprovementLoop::new(ImprovementConfig {
            n_hypotheses: 5,
            top_k: 2,
            eig_threshold: 0.0, // accept all
            temperature: 0.5,
            ..Default::default()
        });

        // Identity observer: actual = prediction (perfect world model)
        let result = sil.run_cycle(&mut belief, &mut wm, &mut rng, |pred| pred.to_vec());

        assert!(result.n_verified > 0);
        assert_eq!(result.n_verified, result.n_passed); // all should pass
        assert!(result.mean_error < 1e-6);
        assert_eq!(sil.total_cycles, 1);
    }

    #[test]
    fn success_rate_tracking() {
        let mut sil = SelfImprovementLoop::default();
        sil.total_verified = 10;
        sil.total_passed = 7;
        assert!((sil.success_rate() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn failure_count_increases_uncertainty() {
        let mut belief = BeliefState::new(2);
        let mut wm = WorldModel::new(2, 2);
        let mut rng = Rng::new(99);
        let mut sil = SelfImprovementLoop::new(ImprovementConfig {
            max_failures: 2,
            n_hypotheses: 3,
            top_k: 1,
            eig_threshold: 0.0,
            ..Default::default()
        });

        let sigma_before = belief.log_sigma.clone();

        // Observer that always returns very wrong results → failures
        for _ in 0..3 {
            sil.run_cycle(&mut belief, &mut wm, &mut rng, |_pred| {
                vec![1000.0, -1000.0]
            });
        }

        // After max_failures exceeded, log_sigma should have increased
        let sigma_increased = belief
            .log_sigma
            .iter()
            .zip(sigma_before.iter())
            .any(|(&after, &before)| after > before);
        assert!(
            sigma_increased,
            "uncertainty should increase after repeated failures"
        );
    }

    #[test]
    fn reset_clears_counters() {
        let mut sil = SelfImprovementLoop::default();
        sil.total_cycles = 100;
        sil.total_verified = 50;
        sil.total_passed = 30;
        sil.failure_count = 5;
        sil.reset();
        assert_eq!(sil.total_cycles, 0);
        assert_eq!(sil.total_verified, 0);
        assert_eq!(sil.total_passed, 0);
    }
}

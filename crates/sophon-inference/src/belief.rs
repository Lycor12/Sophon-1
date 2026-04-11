//! Belief state: the Gaussian variational posterior q(s) = N(μ, σ²I).
//!
//! The belief state is the system's internal representation of what it
//! believes the true latent state to be. It is parameterised by:
//! - μ (mu): the mean vector — best estimate of the latent state
//! - log_σ (log_sigma): log standard deviation — uncertainty estimate
//!
//! Log-space parameterisation ensures σ > 0 without constrained optimisation.
//!
//! Spec §2.3: b_t = P(s_t | o_{1:t}) approximated by q(s_t) = N(μ_t, σ_t²I)

use sophon_core::rng::Rng;

/// The variational posterior q(s) = N(μ, diag(σ²)).
#[derive(Debug, Clone)]
pub struct BeliefState {
    /// Mean of the variational posterior. Shape: [dim]
    pub mu: Vec<f32>,
    /// Log standard deviation. Shape: [dim]
    /// σ = exp(log_sigma), σ² = exp(2 * log_sigma)
    pub log_sigma: Vec<f32>,
    /// Dimensionality of the latent state.
    dim: usize,
    /// Step counter for temporal tracking.
    pub step: u64,
}

impl BeliefState {
    /// Create a new belief state initialised at the prior: N(0, I).
    pub fn new(dim: usize) -> Self {
        Self {
            mu: vec![0.0f32; dim],
            log_sigma: vec![0.0f32; dim], // σ = 1
            dim,
            step: 0,
        }
    }

    /// Create with specific initial values.
    pub fn from_params(mu: Vec<f32>, log_sigma: Vec<f32>) -> Self {
        assert_eq!(mu.len(), log_sigma.len(), "mu and log_sigma must match");
        let dim = mu.len();
        Self {
            mu,
            log_sigma,
            dim,
            step: 0,
        }
    }

    /// Dimensionality of the belief state.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Standard deviation vector: σ_i = exp(log_sigma_i).
    pub fn sigma(&self) -> Vec<f32> {
        self.log_sigma.iter().map(|&ls| ls.exp()).collect()
    }

    /// Variance vector: σ²_i = exp(2 * log_sigma_i).
    pub fn variance(&self) -> Vec<f32> {
        self.log_sigma.iter().map(|&ls| (2.0 * ls).exp()).collect()
    }

    /// Entropy of the Gaussian: H = 0.5 * dim * (1 + log(2π)) + sum(log_sigma).
    pub fn entropy(&self) -> f32 {
        let log_2pi = (2.0 * std::f32::consts::PI).ln();
        let sum_log_sigma: f32 = self.log_sigma.iter().sum();
        0.5 * self.dim as f32 * (1.0 + log_2pi) + sum_log_sigma
    }

    /// Sample from the posterior using the reparameterisation trick:
    /// s = μ + σ ⊙ ε, where ε ~ N(0, I)
    ///
    /// This allows gradients to flow through the sampling operation.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f32> {
        let mut s = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let eps = rng.next_normal(0.0, 1.0);
            let sigma = self.log_sigma[i].exp();
            s[i] = self.mu[i] + sigma * eps;
        }
        s
    }

    /// Update the belief state parameters.
    /// grad_mu and grad_log_sigma are gradients of the free energy.
    /// The update applies gradient DESCENT (subtract gradient * lr).
    pub fn update(&mut self, grad_mu: &[f32], grad_log_sigma: &[f32], lr: f32) {
        assert_eq!(grad_mu.len(), self.dim);
        assert_eq!(grad_log_sigma.len(), self.dim);
        for i in 0..self.dim {
            self.mu[i] -= lr * grad_mu[i];
            self.log_sigma[i] -= lr * grad_log_sigma[i];
            // Clamp log_sigma to prevent numerical issues
            // σ ∈ [exp(-5), exp(5)] ≈ [0.007, 148.4]
            self.log_sigma[i] = self.log_sigma[i].clamp(-5.0, 5.0);
        }
        self.step += 1;
    }

    /// Reset to the prior N(0, I).
    pub fn reset(&mut self) {
        self.mu.fill(0.0);
        self.log_sigma.fill(0.0);
        self.step = 0;
    }

    /// Check if the belief state has valid (finite) parameters.
    pub fn is_valid(&self) -> bool {
        self.mu.iter().all(|v| v.is_finite()) && self.log_sigma.iter().all(|v| v.is_finite())
    }

    /// L2 distance of μ from origin (magnitude of belief shift from prior).
    pub fn mu_magnitude(&self) -> f32 {
        self.mu.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Mean uncertainty: average σ across all dimensions.
    pub fn mean_uncertainty(&self) -> f32 {
        let sigma = self.sigma();
        sigma.iter().sum::<f32>() / self.dim as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_at_prior() {
        let b = BeliefState::new(10);
        assert_eq!(b.dim(), 10);
        assert_eq!(b.step, 0);
        assert!(b.mu.iter().all(|&v| v == 0.0));
        assert!(b.log_sigma.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sigma_from_log_sigma() {
        let b = BeliefState::from_params(vec![0.0; 3], vec![0.0, 1.0, -1.0]);
        let s = b.sigma();
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[1] - 1.0f32.exp()).abs() < 1e-4);
        assert!((s[2] - (-1.0f32).exp()).abs() < 1e-4);
    }

    #[test]
    fn entropy_at_prior() {
        let b = BeliefState::new(1);
        let h = b.entropy();
        // H(N(0,1)) = 0.5 * (1 + log(2π)) ≈ 1.4189
        let expected = 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln());
        assert!((h - expected).abs() < 1e-4, "entropy: {h} vs {expected}");
    }

    #[test]
    fn sample_reparameterisation() {
        let b = BeliefState::new(100);
        let mut rng = Rng::new(42);
        let s = b.sample(&mut rng);
        assert_eq!(s.len(), 100);
        // At prior N(0,1), samples should be roughly zero-mean
        let mean: f32 = s.iter().sum::<f32>() / s.len() as f32;
        assert!(mean.abs() < 0.5, "sample mean {mean} too far from 0");
    }

    #[test]
    fn update_moves_mu() {
        let mut b = BeliefState::new(3);
        let grad_mu = vec![1.0, -1.0, 0.5];
        let grad_ls = vec![0.0, 0.0, 0.0];
        b.update(&grad_mu, &grad_ls, 0.1);
        assert!((b.mu[0] - (-0.1)).abs() < 1e-6);
        assert!((b.mu[1] - 0.1).abs() < 1e-6);
        assert!((b.mu[2] - (-0.05)).abs() < 1e-6);
        assert_eq!(b.step, 1);
    }

    #[test]
    fn log_sigma_clamped() {
        let mut b = BeliefState::from_params(vec![0.0], vec![4.5]);
        // Large positive gradient should push log_sigma down, but let's push it up
        b.update(&[0.0], &[-10.0], 1.0); // log_sigma += 10 → 14.5, clamped to 5
        assert_eq!(b.log_sigma[0], 5.0);
    }

    #[test]
    fn reset_returns_to_prior() {
        let mut b = BeliefState::from_params(vec![1.0, 2.0], vec![0.5, -0.5]);
        b.step = 100;
        b.reset();
        assert!(b.mu.iter().all(|&v| v == 0.0));
        assert!(b.log_sigma.iter().all(|&v| v == 0.0));
        assert_eq!(b.step, 0);
    }

    #[test]
    fn validity_check() {
        let b = BeliefState::new(5);
        assert!(b.is_valid());

        let bad = BeliefState::from_params(vec![f32::NAN], vec![0.0]);
        assert!(!bad.is_valid());
    }

    #[test]
    fn mu_magnitude_and_uncertainty() {
        let b = BeliefState::from_params(vec![3.0, 4.0], vec![0.0, 0.0]);
        assert!((b.mu_magnitude() - 5.0).abs() < 1e-6);
        assert!((b.mean_uncertainty() - 1.0).abs() < 1e-6);
    }
}

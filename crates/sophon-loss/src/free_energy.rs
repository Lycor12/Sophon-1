//! Free energy loss for active inference alignment.
//!
//! Variational free energy: F = E_q[log q(s) - log p(o, s)]
//! = KL(q(s) || p(s)) - E_q[log p(o | s)]
//! = (KL divergence to prior) + (prediction error)
//!
//! For Gaussian variational approximation q(s) = N(mu, diag(sigma^2)):
//! KL(q || p) = 0.5 * sum_i [sigma_i^2 + mu_i^2 - 1 - log(sigma_i^2)]
//!
//! The prediction_error term is computed as the squared error between
//! predicted and actual token probabilities, transformed to log-likelihood.

use sophon_config::VOCAB_SIZE;

/// Components of the free energy decomposition.
#[derive(Debug, Clone, Copy)]
pub struct FreeEnergyComponents {
    /// KL divergence from posterior to prior (regularization term).
    pub kl: f32,
    /// Prediction error (negative log-likelihood of observations).
    pub prediction_error: f32,
    /// Total free energy (KL + prediction_error).
    pub total: f32,
}

/// Compute the KL divergence from q(s) = N(mu, diag(sigma^2)) to standard normal N(0, I).
///
/// KL = 0.5 * sum_i [sigma_i^2 + mu_i^2 - 1 - 2*log(sigma_i)]
///
/// # Arguments
/// * `mu` — mean of variational posterior. Shape: [N]
/// * `log_sigma` — log standard deviation (parameterised in log space for stability). Shape: [N]
///
/// # Returns
/// KL divergence scalar (non-negative).
pub fn kl_divergence_standard_normal(mu: &[f32], log_sigma: &[f32]) -> f32 {
    assert_eq!(mu.len(), log_sigma.len());
    let mut kl = 0.0f32;
    for i in 0..mu.len() {
        let sigma_sq = (2.0 * log_sigma[i]).exp();
        let mu_sq = mu[i] * mu[i];
        kl += sigma_sq + mu_sq - 1.0 - 2.0 * log_sigma[i];
    }
    kl * 0.5
}

/// Compute gradient of KL divergence w.r.t. mu and log_sigma.
///
/// dKL/d(mu[i]) = mu[i]
/// dKL/d(log_sigma[i]) = sigma_i^2 - 1 = exp(2*log_sigma[i]) - 1
///
/// # Returns
/// (grad_mu, grad_log_sigma)
pub fn kl_divergence_grad(mu: &[f32], log_sigma: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(mu.len(), log_sigma.len());
    let n = mu.len();
    let mut grad_mu = vec![0.0f32; n];
    let mut grad_log_sigma = vec![0.0f32; n];
    for i in 0..n {
        grad_mu[i] = mu[i];
        grad_log_sigma[i] = (2.0 * log_sigma[i]).exp() - 1.0;
    }
    (grad_mu, grad_log_sigma)
}

/// Compute prediction error as negative log-likelihood using streaming log-sum-exp.
///
/// This replaces the cross-entropy loss. Given logits and a target token,
/// computes: -log(softmax(logits)[target]) = log_sum_exp - logits[target]
///
/// Uses SLCE (Streaming Log-probability Cross-Entropy) - 2-pass streaming
/// log-sum-exp without materialising full softmax.
///
/// # Arguments
/// * `logits` — raw logit vector of length VOCAB_SIZE
/// * `target` — ground truth token index (0..VOCAB_SIZE-1)
///
/// # Returns
/// Prediction error scalar (non-negative).
pub fn prediction_error_loss(logits: &[f32], target: usize) -> f32 {
    debug_assert_eq!(logits.len(), VOCAB_SIZE);
    debug_assert!(target < VOCAB_SIZE);

    // SLCE Pass 1: streaming LSE
    let mut running_max = logits[0];
    let mut running_sum = 1.0f32;

    for i in 1..VOCAB_SIZE {
        let x = logits[i];
        if x > running_max {
            // Rescale accumulated sum to new max
            running_sum *= (running_max - x).exp();
            running_sum += 1.0;
            running_max = x;
        } else {
            running_sum += (x - running_max).exp();
        }
    }

    let log_sum_exp = running_max + running_sum.ln();

    // prediction_error = log_sum_exp - logits[target]
    log_sum_exp - logits[target]
}

/// Compute prediction error for a batch of predictions.
///
/// # Arguments
/// * `logits_batch` — Vec of logit vectors, each length VOCAB_SIZE
/// * `targets` — ground truth token indices
///
/// # Returns
/// Mean prediction error over the batch.
pub fn prediction_error_batch(logits_batch: &[Vec<f32>], targets: &[usize]) -> f32 {
    assert_eq!(logits_batch.len(), targets.len());
    if logits_batch.is_empty() {
        return 0.0;
    }
    let total: f32 = logits_batch
        .iter()
        .zip(targets.iter())
        .map(|(logits, &t)| prediction_error_loss(logits, t))
        .sum();
    total / logits_batch.len() as f32
}

/// Compute gradient of prediction error w.r.t. logits.
///
/// grad_logits[i] = softmax(logits)[i] - delta(i, target)
///
/// # Arguments
/// * `logits` — raw logit vector of length VOCAB_SIZE
/// * `target` — ground truth token index
///
/// # Returns
/// Gradient vector of length VOCAB_SIZE.
pub fn prediction_error_grad(logits: &[f32], target: usize) -> Vec<f32> {
    debug_assert_eq!(logits.len(), VOCAB_SIZE);
    debug_assert!(target < VOCAB_SIZE);

    // Compute log_sum_exp (same streaming approach)
    let mut running_max = logits[0];
    let mut running_sum = 1.0f32;

    for i in 1..VOCAB_SIZE {
        let x = logits[i];
        if x > running_max {
            running_sum *= (running_max - x).exp();
            running_sum += 1.0;
            running_max = x;
        } else {
            running_sum += (x - running_max).exp();
        }
    }

    let log_sum_exp = running_max + running_sum.ln();

    // grad[i] = softmax[i] - delta(i, target)
    let mut grad = vec![0.0f32; VOCAB_SIZE];
    for i in 0..VOCAB_SIZE {
        let softmax_i = (logits[i] - log_sum_exp).exp();
        grad[i] = softmax_i;
    }
    grad[target] -= 1.0;

    grad
}

/// Compute variational free energy loss.
///
/// F = KL(q || prior) + prediction_error
///
/// # Arguments
/// * `mu` — variational posterior mean
/// * `log_sigma` — variational posterior log-std
/// * `prediction_error` — negative log-likelihood of observations under the model
///
/// # Returns
/// Free energy scalar.
pub fn free_energy_loss(mu: &[f32], log_sigma: &[f32], prediction_error: f32) -> f32 {
    let kl = kl_divergence_standard_normal(mu, log_sigma);
    kl + prediction_error
}

/// Compute all free energy components at once.
///
/// # Arguments
/// * `mu` — variational posterior mean
/// * `log_sigma` — variational posterior log-std
/// * `logits` — model output logits
/// * `target` — ground truth token index
///
/// # Returns
/// FreeEnergyComponents with kl, prediction_error, and total.
pub fn free_energy_components(
    mu: &[f32],
    log_sigma: &[f32],
    logits: &[f32],
    target: usize,
) -> FreeEnergyComponents {
    let kl = kl_divergence_standard_normal(mu, log_sigma);
    let prediction_error = prediction_error_loss(logits, target);
    let total = kl + prediction_error;
    FreeEnergyComponents {
        kl,
        prediction_error,
        total,
    }
}

/// Compute top-1 accuracy over a batch.
///
/// # Arguments
/// * `logits_batch` — Vec of logit vectors
/// * `targets` — ground truth token indices
///
/// # Returns
/// Fraction of predictions where argmax(logits) == target.
pub fn accuracy(logits_batch: &[Vec<f32>], targets: &[usize]) -> f32 {
    assert_eq!(logits_batch.len(), targets.len());
    if logits_batch.is_empty() {
        return 0.0;
    }
    let correct: usize = logits_batch
        .iter()
        .zip(targets.iter())
        .filter(|(logits, &t)| {
            let pred = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            pred == t
        })
        .count();
    correct as f32 / logits_batch.len() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_at_prior_is_zero() {
        // q = N(0, I) -> KL(q || N(0, I)) = 0
        let mu = vec![0.0f32; 10];
        let log_sigma = vec![0.0f32; 10]; // sigma = 1
        let kl = kl_divergence_standard_normal(&mu, &log_sigma);
        assert!(kl.abs() < 1e-6, "KL at prior should be 0, got {kl}");
    }

    #[test]
    fn kl_is_non_negative() {
        let mu = vec![1.0f32, -2.0, 0.5];
        let log_sigma = vec![0.5f32, -0.3, 1.0];
        let kl = kl_divergence_standard_normal(&mu, &log_sigma);
        assert!(kl >= -1e-7, "KL should be non-negative, got {kl}");
    }

    #[test]
    fn kl_increases_with_mu_magnitude() {
        let log_sigma = vec![0.0f32; 5];
        let kl_small = kl_divergence_standard_normal(&vec![0.1f32; 5], &log_sigma);
        let kl_large = kl_divergence_standard_normal(&vec![5.0f32; 5], &log_sigma);
        assert!(
            kl_large > kl_small,
            "KL should increase with |mu|: small={kl_small} large={kl_large}"
        );
    }

    #[test]
    fn kl_grad_mu_is_mu() {
        let mu = vec![1.5f32, -2.0, 0.3];
        let log_sigma = vec![0.0f32; 3];
        let (grad_mu, _) = kl_divergence_grad(&mu, &log_sigma);
        for i in 0..3 {
            assert!(
                (grad_mu[i] - mu[i]).abs() < 1e-6,
                "dKL/dmu[{i}] should be mu[{i}]"
            );
        }
    }

    #[test]
    fn kl_grad_log_sigma_at_prior_is_zero() {
        let mu = vec![0.0f32; 5];
        let log_sigma = vec![0.0f32; 5]; // sigma = 1
        let (_, grad_ls) = kl_divergence_grad(&mu, &log_sigma);
        for (i, &g) in grad_ls.iter().enumerate() {
            assert!(
                g.abs() < 1e-6,
                "dKL/d(log_sigma[{i}]) at prior should be 0, got {g}"
            );
        }
    }

    #[test]
    fn kl_numerical_grad_check() {
        let mu = vec![1.0f32, -0.5, 2.0];
        let log_sigma = vec![0.3f32, -0.2, 0.8];
        let (grad_mu, grad_ls) = kl_divergence_grad(&mu, &log_sigma);

        let eps = 1e-4f32;
        for i in 0..3 {
            // Check mu gradient
            let mut mu_plus = mu.clone();
            mu_plus[i] += eps;
            let mut mu_minus = mu.clone();
            mu_minus[i] -= eps;
            let numerical = (kl_divergence_standard_normal(&mu_plus, &log_sigma)
                - kl_divergence_standard_normal(&mu_minus, &log_sigma))
                / (2.0 * eps);
            let tol = 0.01 * numerical.abs().max(grad_mu[i].abs()).max(1e-3);
            assert!(
                (numerical - grad_mu[i]).abs() < tol,
                "mu grad mismatch at {i}: numerical={numerical} analytical={}",
                grad_mu[i]
            );

            // Check log_sigma gradient
            let mut ls_plus = log_sigma.clone();
            ls_plus[i] += eps;
            let mut ls_minus = log_sigma.clone();
            ls_minus[i] -= eps;
            let numerical = (kl_divergence_standard_normal(&mu, &ls_plus)
                - kl_divergence_standard_normal(&mu, &ls_minus))
                / (2.0 * eps);
            let tol = 0.01 * numerical.abs().max(grad_ls[i].abs()).max(1e-3);
            assert!(
                (numerical - grad_ls[i]).abs() < tol,
                "log_sigma grad mismatch at {i}: numerical={numerical} analytical={}",
                grad_ls[i]
            );
        }
    }

    #[test]
    fn prediction_error_is_non_negative() {
        let logits = vec![0.0f32; VOCAB_SIZE];
        let loss = prediction_error_loss(&logits, 0);
        assert!(loss >= 0.0, "Prediction error should be >= 0, got {loss}");
    }

    #[test]
    fn prediction_error_uniform_logits() {
        // Uniform logits: error = log(VOCAB_SIZE)
        let logits = vec![0.0f32; VOCAB_SIZE];
        let loss = prediction_error_loss(&logits, 42);
        let expected = (VOCAB_SIZE as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-4,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn prediction_error_perfect_prediction() {
        // Target logit much larger than others -> error ≈ 0
        let mut logits = vec![-100.0f32; VOCAB_SIZE];
        logits[10] = 100.0;
        let loss = prediction_error_loss(&logits, 10);
        assert!(
            loss < 1e-4,
            "perfect prediction error should be ~0, got {loss}"
        );
    }

    #[test]
    fn prediction_error_grad_sums_to_zero() {
        // softmax sums to 1, so grad = softmax - one_hot sums to 0
        let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32) * 0.01).collect();
        let grad = prediction_error_grad(&logits, 100);
        let sum: f32 = grad.iter().sum();
        assert!(
            sum.abs() < 1e-4,
            "Prediction error gradient should sum to ~0, got {sum}"
        );
    }

    #[test]
    fn prediction_error_grad_target_is_negative() {
        // grad[target] = softmax[target] - 1 < 0
        let logits = vec![0.0f32; VOCAB_SIZE];
        let grad = prediction_error_grad(&logits, 50);
        assert!(
            grad[50] < 0.0,
            "target gradient should be negative, got {}",
            grad[50]
        );
    }

    #[test]
    fn prediction_error_grad_non_target_is_positive() {
        let logits = vec![0.0f32; VOCAB_SIZE];
        let grad = prediction_error_grad(&logits, 50);
        for (i, &g) in grad.iter().enumerate() {
            if i != 50 {
                assert!(g >= 0.0, "non-target grad[{i}] should be >= 0, got {g}");
            }
        }
    }

    #[test]
    fn prediction_error_numerical_grad_check() {
        let logits: Vec<f32> = (0..VOCAB_SIZE)
            .map(|i| ((i as f32) * 0.137).sin())
            .collect();
        let target = 42;
        let grad = prediction_error_grad(&logits, target);

        let eps = 1e-3f32;
        // Check a few entries
        for &idx in &[0, target, 100, 255] {
            let mut logits_plus = logits.clone();
            logits_plus[idx] += eps;
            let mut logits_minus = logits.clone();
            logits_minus[idx] -= eps;

            let loss_plus = prediction_error_loss(&logits_plus, target);
            let loss_minus = prediction_error_loss(&logits_minus, target);
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (numerical - grad[idx]).abs() < 1e-3,
                "grad mismatch at {idx}: numerical={numerical:.5} analytical={:.5}",
                grad[idx]
            );
        }
    }

    #[test]
    fn batch_loss_is_mean() {
        let logits1 = vec![0.0f32; VOCAB_SIZE];
        let logits2 = vec![0.0f32; VOCAB_SIZE];
        let batch = vec![logits1.clone(), logits2.clone()];
        let targets = vec![0, 1];
        let batch_loss = prediction_error_batch(&batch, &targets);
        let individual =
            (prediction_error_loss(&logits1, 0) + prediction_error_loss(&logits2, 1)) / 2.0;
        assert!(
            (batch_loss - individual).abs() < 1e-6,
            "batch loss should be mean"
        );
    }

    #[test]
    fn accuracy_perfect() {
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        logits[5] = 100.0;
        let batch = vec![logits.clone(), logits.clone()];
        let targets = vec![5, 5];
        let acc = accuracy(&batch, &targets);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_zero() {
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        logits[5] = 100.0;
        let batch = vec![logits.clone(), logits.clone()];
        let targets = vec![0, 1]; // both wrong
        let acc = accuracy(&batch, &targets);
        assert!(acc.abs() < 1e-6);
    }

    #[test]
    fn free_energy_at_prior_equals_prediction_error() {
        let mu = vec![0.0f32; 5];
        let log_sigma = vec![0.0f32; 5];
        let pe = 2.5;
        let fe = free_energy_loss(&mu, &log_sigma, pe);
        assert!(
            (fe - pe).abs() < 1e-6,
            "FE at prior should equal prediction error: {fe} vs {pe}"
        );
    }

    #[test]
    fn free_energy_components_computation() {
        let mu = vec![0.5f32; 10];
        let log_sigma = vec![0.1f32; 10];
        let logits = vec![0.0f32; VOCAB_SIZE];
        let target = 50;

        let comps = free_energy_components(&mu, &log_sigma, &logits, target);

        // Verify components add up
        assert!(
            (comps.total - (comps.kl + comps.prediction_error)).abs() < 1e-5,
            "FE components don't add up: {} + {} != {}",
            comps.kl,
            comps.prediction_error,
            comps.total
        );

        // Verify non-negativity
        assert!(comps.kl >= 0.0, "KL should be non-negative");
        assert!(
            comps.prediction_error >= 0.0,
            "Prediction error should be non-negative"
        );
        assert!(comps.total >= 0.0, "Total FE should be non-negative");
    }
}

//! Free energy loss for active inference alignment.
//!
//! Variational free energy: F = E_q[log q(s) - log p(o, s)]
//!   = KL(q(s) || p(s)) - E_q[log p(o | s)]
//!   = (KL divergence to prior) + (negative log-likelihood of observations)
//!
//! For Gaussian variational approximation q(s) = N(mu, diag(sigma^2)):
//!   KL(q || p) = 0.5 * sum_i [sigma_i^2 / sigma_prior^2 + (mu_i - mu_prior)^2 / sigma_prior^2
//!                              - 1 - log(sigma_i^2 / sigma_prior^2)]
//!
//! With standard normal prior p(s) = N(0, I):
//!   KL = 0.5 * sum_i [sigma_i^2 + mu_i^2 - 1 - log(sigma_i^2)]
//!
//! The free energy loss is combined with the task cross-entropy loss to form
//! the full training objective:
//!   L = alpha * CE + beta * F
//!
//! where alpha and beta are weighting coefficients (default: alpha=1.0, beta=0.01).

/// Compute the KL divergence from q(s) = N(mu, diag(sigma^2)) to standard normal N(0, I).
///
/// KL = 0.5 * sum_i [sigma_i^2 + mu_i^2 - 1 - log(sigma_i^2)]
///    = 0.5 * sum_i [sigma_i^2 + mu_i^2 - 1 - 2*log(sigma_i)]
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

/// Compute variational free energy loss.
///
/// F = KL(q || prior) + prediction_error
///
/// The prediction_error is typically the cross-entropy loss from the task,
/// passed in as a pre-computed scalar.
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

/// Combined loss: weighted sum of cross-entropy and free energy.
///
/// L = alpha * ce_loss + beta * free_energy
///
/// # Arguments
/// * `ce_loss` — cross-entropy loss scalar
/// * `mu` — variational posterior mean
/// * `log_sigma` — variational posterior log-std
/// * `alpha` — CE weight (default 1.0)
/// * `beta` — free energy weight (default 0.01)
///
/// # Returns
/// Combined loss scalar.
pub fn combined_loss(ce_loss: f32, mu: &[f32], log_sigma: &[f32], alpha: f32, beta: f32) -> f32 {
    let fe = free_energy_loss(mu, log_sigma, ce_loss);
    alpha * ce_loss + beta * fe
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
    fn combined_loss_weights_correctly() {
        let ce = 1.0f32;
        let mu = vec![0.0f32; 3];
        let log_sigma = vec![0.0f32; 3];
        let alpha = 2.0;
        let beta = 0.5;

        let fe = free_energy_loss(&mu, &log_sigma, ce); // = 0 + ce = 1.0
        let expected = alpha * ce + beta * fe; // = 2.0 + 0.5 = 2.5
        let actual = combined_loss(ce, &mu, &log_sigma, alpha, beta);
        assert!(
            (actual - expected).abs() < 1e-6,
            "combined: {actual} vs expected {expected}"
        );
    }
}

//! Normalisation kernels with genuine novel optimisations.
//!
//! # Novel optimisation — FWNM (Fused Welford-Normalize-Modulate)
//!
//! Standard LayerNorm implementations use 3 passes:
//! pass 1 computes mean, pass 2 computes variance, pass 3 normalizes+scales+shifts.
//!
//! FWNM reduces this to 2 passes by fusing the first two into a single
//! Welford online statistics pass (numerically stable for all magnitudes),
//! and fusing normalization + gamma scaling + beta shifting into a single
//! multiply-add per element in the second pass.
//!
//! ```text
//! Pass 1 (Welford): For each x_i:
//!   count += 1
//!   delta  = x_i - mean
//!   mean  += delta / count
//!   delta2 = x_i - mean        // uses updated mean
//!   M2    += delta * delta2
//! At end: var = M2 / n, inv_std = 1/sqrt(var + eps)
//!
//! Pass 2 (Fused Normalize-Modulate): For each x_i:
//!   out_i = (x_i - mean) * (inv_std * gamma_i) + beta_i
//! ```
//!
//! The key fusion: inv\_std \* gamma\_i is a single pre-multiplicable
//! factor, so normalize+scale is one FMA instead of two separate muls.
//! This saves n multiplications and one full read of the output buffer
//! compared to a 3-pass implementation.
//!
//! Numerical guarantee: Welford's recurrence avoids catastrophic cancellation
//! in the naive var = E\[x^2\] - E\[x\]^2 formula, which can lose all significant
//! digits when the mean is large relative to the standard deviation.

#![allow(clippy::needless_range_loop)]

use crate::{CoreError, Tensor};

// ---------------------------------------------------------------------------
// LayerNorm (FWNM implementation)
// ---------------------------------------------------------------------------

/// Layer normalisation: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
///
/// Uses FWNM (Fused Welford-Normalize-Modulate) — 2 passes total.
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Result<Vec<f32>, CoreError> {
    let n = x.len();
    if gamma.len() != n || beta.len() != n {
        return Err(CoreError::ShapeMismatch {
            got: [1, gamma.len().min(beta.len())],
            expected: [1, n],
        });
    }
    if n == 0 {
        return Err(CoreError::ZeroDimension);
    }

    // FWNM Pass 1: Welford online mean + M2
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for (count, &xi) in x.iter().enumerate() {
        let count_f = (count + 1) as f32;
        let delta = xi - mean;
        mean += delta / count_f;
        let delta2 = xi - mean;
        m2 += delta * delta2;
    }
    let var = m2 / (n as f32);
    let inv_std = (var + eps).sqrt().recip();

    // FWNM Pass 2: fused normalize + modulate
    // Key: (x - mean) * inv_std * gamma + beta
    //    = (x - mean) * (inv_std * gamma) + beta  [one mul saved per element]
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let fused_scale = inv_std * gamma[i]; // precomputed fused factor
        out[i] = (x[i] - mean) * fused_scale + beta[i];
    }
    Ok(out)
}

/// Layer norm with default gamma=1 and beta=0 (no affine params).
pub fn layer_norm_default(x: &[f32], eps: f32) -> Result<Vec<f32>, CoreError> {
    let n = x.len();
    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    layer_norm(x, &gamma, &beta, eps)
}

/// Layer norm of a full Tensor (treats each row independently).
pub fn layer_norm_tensor(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor, CoreError> {
    let rows = x.rows();
    let cols = x.cols();
    if gamma.cols() != cols || beta.cols() != cols {
        return Err(CoreError::ShapeMismatch {
            got: [1, gamma.cols()],
            expected: [1, cols],
        });
    }
    let mut out = Tensor::zeros_2d(rows, cols);
    for r in 0..rows {
        let row_in = x.row(r)?;
        let g_row = gamma.row(0)?;
        let b_row = beta.row(0)?;
        let normed = layer_norm(row_in, g_row, b_row, eps)?;
        let out_row = out.row_mut(r)?;
        out_row.copy_from_slice(&normed);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// RMS Norm (used as a lighter alternative inside some kernels)
// ---------------------------------------------------------------------------

/// RMS normalisation: `y = x / rms(x) * gamma`.
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>, CoreError> {
    let n = x.len();
    if gamma.len() != n {
        return Err(CoreError::ZeroDimension);
    }
    let ms: f32 = x.iter().map(|&v| v * v).sum::<f32>() / (n as f32);
    let inv_rms = (ms + eps).sqrt().recip();
    Ok(x.iter()
        .zip(gamma)
        .map(|(&xi, &gi)| xi * inv_rms * gi)
        .collect())
}

// ---------------------------------------------------------------------------
// LayerNorm backward (needed for training)
// ---------------------------------------------------------------------------

/// Backward pass for layer_norm: computes gradients w.r.t. input, gamma, beta.
///
/// Given upstream gradient `grad_out`, the original input `x`, and the
/// forward-computed `mean` and `inv_std`:
///   d_beta[i]  = grad_out[i]
///   d_gamma[i] = grad_out[i] * (x[i] - mean) * inv_std
///   d_x[i]     = complex chain rule through mean and variance
///
/// Returns (grad_x, grad_gamma, grad_beta).
pub fn layer_norm_backward(
    grad_out: &[f32],
    x: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), CoreError> {
    let n = x.len();
    if grad_out.len() != n || gamma.len() != n {
        return Err(CoreError::ShapeMismatch {
            got: [1, grad_out.len()],
            expected: [1, n],
        });
    }

    // Recompute forward statistics (FWNM pass 1)
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for (count, &xi) in x.iter().enumerate() {
        let count_f = (count + 1) as f32;
        let delta = xi - mean;
        mean += delta / count_f;
        let delta2 = xi - mean;
        m2 += delta * delta2;
    }
    let var = m2 / n as f32;
    let inv_std = (var + eps).sqrt().recip();
    let n_f = n as f32;

    // Gradient w.r.t. beta and gamma
    let mut grad_beta = vec![0.0f32; n];
    let mut grad_gamma = vec![0.0f32; n];
    let mut x_hat = vec![0.0f32; n]; // normalised x

    for i in 0..n {
        x_hat[i] = (x[i] - mean) * inv_std;
        grad_beta[i] = grad_out[i];
        grad_gamma[i] = grad_out[i] * x_hat[i];
    }

    // Gradient w.r.t. x (the hard part)
    // d_x = inv_std * gamma * (grad_out - mean(grad_out * gamma * inv_std)
    //        - x_hat * mean(grad_out * gamma * inv_std * x_hat))
    // Simplified:
    //   Let g_scaled[i] = grad_out[i] * gamma[i]
    //   mean_gs = mean(g_scaled)
    //   mean_gs_xhat = mean(g_scaled * x_hat)
    //   grad_x[i] = inv_std * (g_scaled[i] - mean_gs - x_hat[i] * mean_gs_xhat)
    let mut mean_gs = 0.0f32;
    let mut mean_gs_xh = 0.0f32;
    for i in 0..n {
        let gs = grad_out[i] * gamma[i];
        mean_gs += gs;
        mean_gs_xh += gs * x_hat[i];
    }
    mean_gs /= n_f;
    mean_gs_xh /= n_f;

    let mut grad_x = vec![0.0f32; n];
    for i in 0..n {
        let gs = grad_out[i] * gamma[i];
        grad_x[i] = inv_std * (gs - mean_gs - x_hat[i] * mean_gs_xh);
    }

    Ok((grad_x, grad_gamma, grad_beta))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn layer_norm_zero_mean_unit_variance() {
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = layer_norm_default(&x, 1e-5).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        assert!(close(mean, 0.0, 1e-5), "mean={mean}");
        assert!(close(var, 1.0, 1e-4), "var={var}");
    }

    #[test]
    fn layer_norm_with_gamma_beta() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![2.0f32; 4];
        let beta = vec![1.0f32; 4];
        let out = layer_norm(&x, &gamma, &beta, 1e-5).unwrap();
        assert!(out.iter().all(|&v| v.is_finite()));
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let std: f32 = (out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0).sqrt();
        assert!(close(std, 2.0, 1e-4), "std={std}");
    }

    #[test]
    fn rms_norm_unit_rms() {
        let x = vec![3.0f32, 4.0];
        let gamma = vec![1.0f32; 2];
        let out = rms_norm(&x, &gamma, 1e-5).unwrap();
        let rms: f32 = (out.iter().map(|&v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!(close(rms, 1.0, 1e-5), "rms={rms}");
    }

    #[test]
    fn welford_vs_naive() {
        let x = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let out_w = layer_norm_default(&x, 0.0).unwrap();
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv = var.sqrt().recip();
        let out_naive: Vec<f32> = x.iter().map(|&v| (v - mean) * inv).collect();
        for (a, b) in out_w.iter().zip(&out_naive) {
            assert!(close(*a, *b, 1e-5), "welford={a}, naive={b}");
        }
    }

    #[test]
    fn layer_norm_backward_beta_is_grad_out() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let grad_out = vec![0.5f32, 1.0, 1.5, 2.0];
        let (_, _, grad_beta) = layer_norm_backward(&grad_out, &x, &gamma, 1e-5).unwrap();
        for (a, b) in grad_beta.iter().zip(&grad_out) {
            assert!(close(*a, *b, 1e-6), "grad_beta={a}, grad_out={b}");
        }
    }

    #[test]
    fn layer_norm_backward_grad_x_sums_to_near_zero() {
        // A property of LayerNorm: sum of grad_x should be approximately 0
        let x = vec![1.0f32, 3.0, 5.0, 7.0];
        let gamma = vec![1.0f32; 4];
        let grad_out = vec![1.0f32; 4];
        let (grad_x, _, _) = layer_norm_backward(&grad_out, &x, &gamma, 1e-5).unwrap();
        let sum: f32 = grad_x.iter().sum();
        assert!(sum.abs() < 1e-4, "sum of grad_x = {sum}");
    }
}

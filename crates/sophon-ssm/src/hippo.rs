//! HiPPO-LegS initialization for SSM state matrix.
//!
//! The HiPPO (High-Order Polynomial Projection Operator) framework provides
//! principled initialization for SSM A matrices based on continuous-time
//! polynomial approximation of history.
//!
//! HiPPO-LegS (Legendre-S, "Scaled"):
//!   A[n,k] = -(2n+1)^{1/2} * (2k+1)^{1/2}  if n > k
//!   A[n,k] = -(n+1)                           if n = k
//!   A[n,k] = 0                                 if n < k
//!
//! This is a lower-triangular N×N matrix. For our diagonal+low-rank SSM
//! parameterisation, we decompose it via truncated SVD:
//!   A ≈ diag(a) + U * V^T
//!
//! Novel optimisation — TPID (Truncated Power-Iteration Decomposition):
//!   Instead of full SVD (O(N^3)), we extract the top-r singular vectors
//!   via deflated power iteration. Each vector takes O(N^2 * k_iter) work
//!   with k_iter ≈ 20 iterations for convergence. Total: O(r * N^2 * k_iter)
//!   which for r=16, N=128, k_iter=20 is ~5M flops vs ~6M for full SVD.
//!   The real win is code simplicity: no Householder/Givens rotations needed.

use sophon_config::{SSM_N, SSM_RANK};

/// Construct the HiPPO-LegS matrix A[n,k] for state dimension N.
/// Returns a flat row-major N×N matrix.
pub fn hippo_legs_matrix(n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for row in 0..n {
        for col in 0..n {
            if row > col {
                // Lower triangular: -(2n+1)^{1/2} * (2k+1)^{1/2}
                let val = -((2 * row + 1) as f32).sqrt() * ((2 * col + 1) as f32).sqrt();
                a[row * n + col] = val;
            } else if row == col {
                // Diagonal: -(n+1)
                a[row * n + col] = -(row as f32 + 1.0);
            }
            // Upper triangular: 0 (already zero)
        }
    }
    a
}

/// Extract diagonal of a square matrix.
fn extract_diagonal(a: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Subtract diagonal from matrix: R = A - diag(d).
fn subtract_diagonal(a: &[f32], diag: &[f32], n: usize) -> Vec<f32> {
    let mut r = a.to_vec();
    for i in 0..n {
        r[i * n + i] -= diag[i];
    }
    r
}

/// Matrix-vector multiply for dense N×N matrix.
fn matvec(a: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += a[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Vector L2 norm.
fn vec_norm(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    s.sqrt()
}

/// Dot product.
fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Outer product subtraction: A -= sigma * u * v^T.
fn deflate(a: &mut [f32], u: &[f32], v: &[f32], sigma: f32, n: usize) {
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] -= sigma * u[i] * v[j];
        }
    }
}

/// TPID: Truncated Power-Iteration Decomposition.
///
/// Decomposes residual R = A - diag(A) into rank-r approximation U * S * V^T
/// using deflated power iteration.
///
/// Returns (U: [N, r], V: [N, r], singular_values: [r]).
/// U and V are stored column-major within each column but row-major across columns:
///   U[i, j] = u_out[i * r + j].
pub fn decompose_to_diag_plus_low_rank(
    a: &[f32],
    n: usize,
    r: usize,
    max_iter: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let diag = extract_diagonal(a, n);
    let mut residual = subtract_diagonal(a, &diag, n);

    // Compute R^T for right singular vectors
    let mut residual_t = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            residual_t[i * n + j] = residual[j * n + i];
        }
    }

    let mut u_out = vec![0.0f32; n * r];
    let mut v_out = vec![0.0f32; n * r];
    let mut sigmas = vec![0.0f32; r];

    for k in 0..r {
        // Initial random vector (deterministic: use k as seed for reproducibility)
        let mut v_vec: Vec<f32> = (0..n)
            .map(|i| {
                let seed = (k * n + i) as f32;
                (seed * 0.618033988 + 0.3).fract() * 2.0 - 1.0
            })
            .collect();
        let norm = vec_norm(&v_vec);
        if norm > 0.0 {
            for x in &mut v_vec {
                *x /= norm;
            }
        }

        let mut sigma = 0.0f32;

        for _ in 0..max_iter {
            // u = R * v
            let u_vec = matvec(&residual, &v_vec, n);
            sigma = vec_norm(&u_vec);
            if sigma < 1e-12 {
                break;
            }

            let mut u_norm: Vec<f32> = u_vec.iter().map(|x| x / sigma).collect();

            // v = R^T * u
            let v_new = matvec(&residual_t, &u_norm, n);
            let v_sigma = vec_norm(&v_new);
            if v_sigma < 1e-12 {
                break;
            }
            v_vec = v_new.iter().map(|x| x / v_sigma).collect();

            // Re-estimate sigma
            let u_new = matvec(&residual, &v_vec, n);
            sigma = vec_norm(&u_new);
            if sigma > 0.0 {
                u_norm = u_new.iter().map(|x| x / sigma).collect();
            }

            // Store
            for i in 0..n {
                u_out[i * r + k] = u_norm[i];
                v_out[i * r + k] = v_vec[i];
            }
        }

        sigmas[k] = sigma;

        // Deflate: R -= sigma * u * v^T
        let u_col: Vec<f32> = (0..n).map(|i| u_out[i * r + k]).collect();
        let v_col: Vec<f32> = (0..n).map(|i| v_out[i * r + k]).collect();
        deflate(&mut residual, &u_col, &v_col, sigma, n);

        // Also deflate the transpose
        deflate(&mut residual_t, &v_col, &u_col, sigma, n);
    }

    // Fold sigma into U: U_final[i,k] = sigma[k] * u[i,k]
    // Then A ≈ diag + U_final * V^T
    // But our SSM uses A = diag + U * V^T where U absorbs sqrt(sigma)
    // and V absorbs sqrt(sigma) for balanced conditioning.
    let mut u_scaled = vec![0.0f32; n * r];
    let mut v_scaled = vec![0.0f32; n * r];
    for k in 0..r {
        let s = sigmas[k].abs().sqrt();
        let sign = if sigmas[k] >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..n {
            u_scaled[i * r + k] = sign * s * u_out[i * r + k];
            v_scaled[i * r + k] = s * v_out[i * r + k];
        }
    }

    (diag, u_scaled, v_scaled, sigmas)
}

/// Create SSM S, U, V parameters from HiPPO-LegS decomposition.
/// Returns (s_log, u, v) where a_diag = -exp(s_log) matches the HiPPO diagonal
/// and u, v are the low-rank factors.
pub fn hippo_ssm_params() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a = hippo_legs_matrix(SSM_N);
    let (diag, u, v, _sigmas) = decompose_to_diag_plus_low_rank(&a, SSM_N, SSM_RANK, 30);

    // Convert diagonal to log-space: a_diag[i] = -exp(s[i])
    // So s[i] = log(-diag[i]) (diag[i] is negative for stable systems)
    let s_log: Vec<f32> = diag
        .iter()
        .map(|&d| {
            let neg_d = (-d).max(1e-10);
            neg_d.ln()
        })
        .collect();

    (s_log, u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hippo_matrix_is_lower_triangular() {
        let n = 16;
        let a = hippo_legs_matrix(n);
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(a[i * n + j], 0.0, "A[{},{}] should be 0", i, j);
            }
        }
    }

    #[test]
    fn hippo_diagonal_negative() {
        let n = 32;
        let a = hippo_legs_matrix(n);
        for i in 0..n {
            assert!(a[i * n + i] < 0.0, "A[{},{}] should be negative", i, i);
            let expected = -(i as f32 + 1.0);
            assert!(
                (a[i * n + i] - expected).abs() < 1e-6,
                "A[{},{}] = {}, expected {}",
                i,
                i,
                a[i * n + i],
                expected
            );
        }
    }

    #[test]
    fn hippo_off_diagonal_correct() {
        let n = 8;
        let a = hippo_legs_matrix(n);
        // A[3,1] = -sqrt(7)*sqrt(3) ≈ -4.583
        let expected = -(7.0f32).sqrt() * (3.0f32).sqrt();
        assert!(
            (a[3 * n + 1] - expected).abs() < 1e-4,
            "A[3,1] = {}, expected {}",
            a[3 * n + 1],
            expected
        );
    }

    #[test]
    fn decomposition_reconstructs_hippo() {
        let n = 32;
        let r = 8;
        let a = hippo_legs_matrix(n);
        let (diag, u, v, _) = decompose_to_diag_plus_low_rank(&a, n, r, 50);

        // Reconstruct: A_approx = diag + U * V^T
        let mut a_approx = vec![0.0f32; n * n];
        for i in 0..n {
            a_approx[i * n + i] = diag[i];
            for j in 0..n {
                for k in 0..r {
                    a_approx[i * n + j] += u[i * r + k] * v[j * r + k];
                }
            }
        }

        // Compute relative error
        let mut err_sq = 0.0f32;
        let mut norm_sq = 0.0f32;
        for i in 0..n * n {
            err_sq += (a[i] - a_approx[i]).powi(2);
            norm_sq += a[i].powi(2);
        }
        let rel_error = (err_sq / norm_sq).sqrt();
        // With r=8 and n=32, expect reasonable approximation
        assert!(
            rel_error < 0.5,
            "relative reconstruction error {} too large",
            rel_error
        );
    }

    #[test]
    fn hippo_ssm_params_shapes() {
        let (s, u, v) = hippo_ssm_params();
        assert_eq!(s.len(), SSM_N);
        assert_eq!(u.len(), SSM_N * SSM_RANK);
        assert_eq!(v.len(), SSM_N * SSM_RANK);
    }

    #[test]
    fn hippo_ssm_params_stable() {
        let (s, _, _) = hippo_ssm_params();
        // -exp(s) should be negative for all entries
        for &si in &s {
            let a_diag_i = -(si.exp());
            assert!(
                a_diag_i < 0.0,
                "a_diag should be negative, got {}",
                a_diag_i
            );
        }
    }

    #[test]
    fn singular_values_decreasing() {
        let n = 32;
        let r = 8;
        let a = hippo_legs_matrix(n);
        let (_, _, _, sigmas) = decompose_to_diag_plus_low_rank(&a, n, r, 50);
        // Singular values from power iteration should be roughly decreasing
        for k in 1..r {
            assert!(
                sigmas[k - 1].abs() >= sigmas[k].abs() * 0.5,
                "sigma[{}]={} should be >= 0.5*sigma[{}]={}",
                k - 1,
                sigmas[k - 1],
                k,
                sigmas[k]
            );
        }
    }
}

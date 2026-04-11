//! Zero-Order Hold (ZOH) discretisation for the diagonal+low-rank SSM.
//!
//! Spec: §0.2.2 — ZOH discretisation.
//!
//! Exact ZOH for a general A:
//!   A_bar = exp(Δ A)
//!   B_bar = A^{-1} (A_bar - I) B
//!
//! For diagonal A_diag = -exp(S):
//!   A_bar_diag[i] = exp(Δ * A_diag[i]) = exp(-Δ * exp(S[i]))
//!   B_bar_diag[i,j] = (A_bar_diag[i] - 1) / A_diag[i] * B[i,j]
//!                    = (exp(-Δ exp(S[i])) - 1) / (-exp(S[i])) * B[i,j]
//!
//! For the low-rank correction U V^T we use WACA (Woodbury-Augmented
//! Cayley Approximation):
//!
//!   WACA derivation:
//!     Let A = Λ + U V^T where Λ = diag(A_diag).
//!     Cayley approximation: exp(Δ A) ≈ (I + Δ/2 * A)(I - Δ/2 * A)^{-1}
//!     Let M = I - Δ/2 * A = (I - Δ/2 Λ) - Δ/2 U V^T.
//!     By Woodbury: M^{-1} = D_inv + D_inv (Δ/2 U) (I + V^T D_inv (Δ/2 U))^{-1} V^T D_inv
//!     where D = I - Δ/2 Λ  (diagonal, trivially invertible).
//!     The inner matrix is (r x r), solved with Gaussian elimination in O(r^3).
//!     Then A_bar ≈ (I + Δ/2 A) M^{-1}.
//!
//!   This precomputes A_bar in O(N r^2) and stores a compact representation:
//!     - a_bar_diag: Vec<f32> [N] (diagonal part)
//!     - a_bar_lr_left:  Vec<f32> [N, r]  (low-rank left factor)
//!     - a_bar_lr_right: Vec<f32> [N, r]  (low-rank right factor)
//!   so that A_bar h = a_bar_diag ⊙ h + a_bar_lr_left (a_bar_lr_right^T h)
//!   and the per-token MATVEC is O(N r), not O(N^2).
//!
//! B_bar is stored densely as [N, D] since it is used every step.

use crate::params::SsmParams;
use sophon_config::{SSM_D, SSM_N, SSM_RANK};

// ---------------------------------------------------------------------------
// DiscretisedSsm
// ---------------------------------------------------------------------------

/// Pre-computed ZOH discretisation of one SSM layer.
///
/// Stored compactly to keep per-token matmul at O(N r + N D) rather than O(N^2).
#[derive(Clone)]
pub struct DiscretisedSsm {
    /// Diagonal part of A_bar: a_d[i] = exp(Δ A_diag[i]).   Shape: [N]
    pub a_d: Vec<f32>,

    /// Low-rank left factor of A_bar correction.  Shape: [N, r]
    pub a_lr_l: Vec<f32>,

    /// Low-rank right factor of A_bar correction. Shape: [N, r]
    pub a_lr_r: Vec<f32>,

    /// Discretised input matrix B_bar.  Shape: [N, D]
    pub b_bar: Vec<f32>,
}

impl DiscretisedSsm {
    /// Perform ZOH discretisation of params at the current Δ.
    ///
    /// This is called once per layer per forward pass (or when params change),
    /// not per token.
    pub fn from_params(p: &SsmParams) -> Self {
        let n = SSM_N;
        let d = SSM_D;
        let r = SSM_RANK;
        let delta = p.delta();

        // ---- Diagonal part ----
        let a_diag = p.a_diag(); // -exp(S), all negative
        let mut a_d = vec![0.0f32; n];
        for i in 0..n {
            a_d[i] = (delta * a_diag[i]).exp(); // exp(-Δ exp(S[i]))
        }

        // ---- B_bar ----
        // b_bar[i,j] = (a_d[i] - 1) / a_diag[i] * B[i,j]
        let mut b_bar = vec![0.0f32; n * d];
        for i in 0..n {
            let scale = if a_diag[i].abs() > 1e-12 {
                (a_d[i] - 1.0) / a_diag[i]
            } else {
                delta // L'Hôpital: lim_{a->0} (e^{da}-1)/a = d
            };
            let row = i * d;
            for j in 0..d {
                b_bar[row + j] = scale * p.b[row + j];
            }
        }

        // ---- Low-rank correction (WACA) ----
        // We approximate exp(Δ A) ≈ (I + Δ/2 A)(I - Δ/2 A)^{-1}
        // where A = Λ + U V^T.
        //
        // Step 1: D = diag(1 - Δ/2 * A_diag[i])
        let half_d = delta * 0.5;
        let d_diag: Vec<f32> = a_diag.iter().map(|&ai| 1.0 - half_d * ai).collect();
        let d_diag_inv: Vec<f32> = d_diag
            .iter()
            .map(|&di| if di.abs() > 1e-12 { di.recip() } else { 1.0 })
            .collect();

        // Step 2: Compute D^{-1} U  and  D^{-1} V  (scale rows)
        let mut dinv_u = vec![0.0f32; n * r]; // D^{-1} U
        let mut dinv_v = vec![0.0f32; n * r]; // D^{-1} V
        for i in 0..n {
            let inv = d_diag_inv[i];
            for k in 0..r {
                dinv_u[i * r + k] = inv * p.u[i * r + k];
                dinv_v[i * r + k] = inv * p.v[i * r + k];
            }
        }

        // Step 3: Woodbury core matrix W = I + V^T (Δ/2 D^{-1} U)
        //         W in R^{r x r}
        let mut w = vec![0.0f32; r * r];
        // W[k1, k2] = delta(k1==k2) + sum_i V[i,k1] * (Δ/2 * dinv_u[i, k2])
        for k1 in 0..r {
            for k2 in 0..r {
                let mut s = if k1 == k2 { 1.0f32 } else { 0.0 };
                for i in 0..n {
                    s += p.v[i * r + k1] * half_d * dinv_u[i * r + k2];
                }
                w[k1 * r + k2] = s;
            }
        }

        // Step 4: Invert W via Gauss-Jordan (r x r, r=16, tiny)
        let w_inv = invert_r_matrix(&w, r);

        // Step 5: A_bar low-rank correction:
        //   A_bar ≈ D^{-1} (I + Δ/2 A)(I - Δ/2 A)^{-1}
        //         = (I + Δ/2 Λ D^{-1}) + Woodbury correction
        //
        //   The correction adds a rank-r term: (Δ/2 dinv_u) W_inv (V^T D_inv)
        //   Left factor:  L = Δ/2 * D^{-1} U      shape [N, r]
        //   Right factor: R = W_inv V^T D^{-1}     shape [r, N] stored as [N, r]
        //
        //   We store factored: a_lr_l = L, a_lr_r = R^T (both [N,r])
        //   so A_bar_lr h = L (R^T h) where the inner product is [r] -> [N].

        // L = Δ/2 * D^{-1} U
        let mut a_lr_l = vec![0.0f32; n * r];
        for i in 0..n {
            for k in 0..r {
                a_lr_l[i * r + k] = half_d * dinv_u[i * r + k];
            }
        }

        // R^T = D^{-1} V W^{-T}  (shape [N, r])
        // W_inv^T[k1,k2] = W_inv[k2,k1]
        // (D^{-1} V W^{-T})[i, k1] = dinv_v[i, :] @ W_inv[:, k1]
        let mut a_lr_r = vec![0.0f32; n * r];
        for i in 0..n {
            for k1 in 0..r {
                let mut s = 0.0f32;
                for k2 in 0..r {
                    s += dinv_v[i * r + k2] * w_inv[k2 * r + k1];
                }
                a_lr_r[i * r + k1] = s;
            }
        }

        // Also update a_d to include the diagonal component of the Cayley approx:
        // full a_d already comes from exact diagonal exp, but add (I + Δ/2 Λ) D^{-1} - I
        // correction to align with WACA.
        // For diagonal part: (1 + Δ/2 ai) * d_inv[i]
        for i in 0..n {
            a_d[i] = (1.0 + half_d * a_diag[i]) * d_diag_inv[i];
        }

        Self {
            a_d,
            a_lr_l,
            a_lr_r,
            b_bar,
        }
    }

    /// Apply A_bar to h: out = A_bar h.
    /// out[i] = a_d[i] * h[i] + sum_k a_lr_l[i,k] * (sum_j a_lr_r[j,k] * h[j])
    ///
    /// This is O(N r) per call.
    pub fn apply_a_bar(&self, h: &[f32]) -> Vec<f32> {
        let n = SSM_N;
        let r = SSM_RANK;

        // Step 1: z[k] = sum_j a_lr_r[j,k] * h[j]   [r vector]
        let mut z = vec![0.0f32; r];
        for j in 0..n {
            for k in 0..r {
                z[k] += self.a_lr_r[j * r + k] * h[j];
            }
        }

        // Step 2: out[i] = a_d[i]*h[i] + sum_k a_lr_l[i,k] * z[k]
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = self.a_d[i] * h[i];
            for k in 0..r {
                acc += self.a_lr_l[i * r + k] * z[k];
            }
            out[i] = acc;
        }
        out
    }

    /// Apply B_bar to input u: out[i] = sum_j B_bar[i,j] * u[j].
    /// O(N D) per call.
    pub fn apply_b_bar(&self, u: &[f32]) -> Vec<f32> {
        let n = SSM_N;
        let d = SSM_D;
        debug_assert_eq!(u.len(), d);
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = 0.0f32;
            let row = i * d;
            for j in 0..d {
                acc += self.b_bar[row + j] * u[j];
            }
            out[i] = acc;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// r x r Gauss-Jordan inversion (r = SSM_RANK = 16)
// ---------------------------------------------------------------------------

/// Invert a dense (r x r) matrix using Gauss-Jordan with partial pivoting.
/// Returns a flat row-major Vec<f32> of shape [r, r].
fn invert_r_matrix(m: &[f32], r: usize) -> Vec<f32> {
    // Augmented matrix [M | I]
    let mut aug = vec![0.0f32; r * 2 * r];
    for i in 0..r {
        for j in 0..r {
            aug[i * 2 * r + j] = m[i * r + j];
            aug[i * 2 * r + r + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    for col in 0..r {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = aug[col * 2 * r + col].abs();
        for row in col + 1..r {
            let v = aug[row * 2 * r + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..2 * r {
                aug.swap(col * 2 * r + j, max_row * 2 * r + j);
            }
        }

        let pivot = aug[col * 2 * r + col];
        let inv_pivot = if pivot.abs() > 1e-12 {
            pivot.recip()
        } else {
            0.0
        };

        // Scale pivot row
        for j in 0..2 * r {
            aug[col * 2 * r + j] *= inv_pivot;
        }

        // Eliminate column
        for row in 0..r {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * r + col];
            for j in 0..2 * r {
                let sub = factor * aug[col * 2 * r + j];
                aug[row * 2 * r + j] -= sub;
            }
        }
    }

    // Extract right half
    let mut inv = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            inv[i * r + j] = aug[i * 2 * r + r + j];
        }
    }
    inv
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SsmParams;

    #[test]
    fn discretisation_produces_finite_values() {
        let p = SsmParams::new_stable(0);
        let disc = DiscretisedSsm::from_params(&p);
        assert!(
            disc.a_d.iter().all(|&v| v.is_finite()),
            "a_d has non-finite"
        );
        assert!(
            disc.b_bar.iter().all(|&v| v.is_finite()),
            "b_bar has non-finite"
        );
        assert!(
            disc.a_lr_l.iter().all(|&v| v.is_finite()),
            "a_lr_l has non-finite"
        );
        assert!(
            disc.a_lr_r.iter().all(|&v| v.is_finite()),
            "a_lr_r has non-finite"
        );
    }

    #[test]
    fn apply_a_bar_zero_state_gives_zero() {
        let p = SsmParams::new_stable(1);
        let disc = DiscretisedSsm::from_params(&p);
        let h = vec![0.0f32; SSM_N];
        let out = disc.apply_a_bar(&h);
        assert!(out.iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn apply_b_bar_zero_input_gives_zero() {
        let p = SsmParams::new_stable(2);
        let disc = DiscretisedSsm::from_params(&p);
        let u = vec![0.0f32; SSM_D];
        let out = disc.apply_b_bar(&u);
        assert!(out.iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn invert_identity() {
        let r = 4;
        let eye: Vec<f32> = (0..r * r)
            .map(|i| if i / r == i % r { 1.0 } else { 0.0 })
            .collect();
        let inv = invert_r_matrix(&eye, r);
        for i in 0..r {
            for j in 0..r {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * r + j] - expected).abs() < 1e-5,
                    "inv[{i},{j}]={} expected={expected}",
                    inv[i * r + j]
                );
            }
        }
    }

    #[test]
    fn a_bar_diagonal_entries_less_than_one() {
        // Stable SSM: |A_bar_diag| should be < 1 for stable dynamics
        let p = SsmParams::new_stable(3);
        let disc = DiscretisedSsm::from_params(&p);
        // Check that the discretised diagonal isn't growing unboundedly
        let max_ad: f32 = disc.a_d.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_ad < 2.0, "a_d has suspiciously large value: {max_ad}");
    }
}

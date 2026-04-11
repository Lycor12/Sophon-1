//! SSM parameter tensors.
//!
//! All parameters are stored as Vec<f32>.
//! Initialisation follows spec §1.2.4 stability constraints:
//!   - S initialised to 0 => A_diag = -exp(0) = -1 (stable)
//!   - U, V initialised small (Kaiming) => low-rank term is small perturbation
//!   - B initialised Kaiming
//!   - C initialised Kaiming
//!   - D initialised to 0 (no skip connection by default)
//!   - Delta initialised to log(0.1) => Δ = 0.1 (small step, high fidelity)

use sophon_config::{SSM_D, SSM_N, SSM_P, SSM_RANK};
use sophon_core::rng::Rng;

// ---------------------------------------------------------------------------
// SsmParams
// ---------------------------------------------------------------------------

/// All learnable parameters for one SSM layer.
#[derive(Clone)]
pub struct SsmParams {
    // Diagonal log-magnitude: A_diag = -exp(S)
    pub s: Vec<f32>, // shape: [N]

    // Low-rank factors: A_lr = U V^T
    pub u: Vec<f32>, // shape: [N, r]
    pub v: Vec<f32>, // shape: [N, r]

    // Input matrix
    pub b: Vec<f32>, // shape: [N, D]

    // Output matrix
    pub c: Vec<f32>, // shape: [P, N]

    // Skip/feedthrough matrix (usually small)
    pub d: Vec<f32>, // shape: [P, D]

    // Log step size (scalar, learnable)
    pub log_delta: f32,

    // Optional: delta projection weights for selective scan.
    // When Some, delta is input-dependent: Δ(u) = softplus(w_delta · u + b_delta).
    pub delta_proj_w: Option<Vec<f32>>, // shape: [D] if present
    pub delta_proj_b: Option<f32>,
}

impl SsmParams {
    /// Create parameters with stable initialisation.
    pub fn new_stable(seed: u64) -> Self {
        let mut rng = Rng::new(seed);

        let n = SSM_N;
        let d = SSM_D;
        let p = SSM_P;
        let r = SSM_RANK;

        // S = 0 => A_diag = -1 everywhere (maximally stable)
        let s = vec![0.0f32; n];

        // U, V: small Kaiming init with fan = r
        let mut u = vec![0.0f32; n * r];
        let mut v = vec![0.0f32; n * r];
        rng.fill_kaiming_uniform(&mut u, r);
        rng.fill_kaiming_uniform(&mut v, r);
        // Scale down U,V to make low-rank term a small perturbation
        for x in u.iter_mut() {
            *x *= 0.1;
        }
        for x in v.iter_mut() {
            *x *= 0.1;
        }

        // B: Kaiming uniform with fan = D
        let mut b = vec![0.0f32; n * d];
        rng.fill_kaiming_uniform(&mut b, d);

        // C: Kaiming uniform with fan = N
        let mut c = vec![0.0f32; p * n];
        rng.fill_kaiming_uniform(&mut c, n);

        // D: zero initialisation (no skip by default)
        let d_mat = vec![0.0f32; p * d];

        // log_delta = log(0.1) => Δ = 0.1
        let log_delta = 0.1f32.ln();

        Self {
            s,
            u,
            v,
            b: b,
            c: c,
            d: d_mat,
            log_delta,
            delta_proj_w: None,
            delta_proj_b: None,
        }
    }

    // ------------------------------------------------------------------
    // Accessors with shape metadata
    // ------------------------------------------------------------------

    #[inline]
    pub fn n(&self) -> usize {
        SSM_N
    }
    #[inline]
    pub fn d(&self) -> usize {
        SSM_D
    }
    #[inline]
    pub fn p(&self) -> usize {
        SSM_P
    }
    #[inline]
    pub fn r(&self) -> usize {
        SSM_RANK
    }

    /// Compute A_diag[i] = -exp(S[i]).
    /// Returns a Vec<f32> of length N.
    pub fn a_diag(&self) -> Vec<f32> {
        self.s.iter().map(|&si| -si.exp()).collect()
    }

    /// Delta = exp(log_delta). Clamped to [1e-4, 1.0] for numerical safety.
    #[inline]
    pub fn delta(&self) -> f32 {
        self.log_delta.exp().clamp(1e-4, 1.0)
    }

    /// Low-rank contribution: (U V^T)[i, j] = sum_k U[i,k] * V[j,k].
    /// Returns a flat row-major Vec<f32> of shape [N, N].
    /// NOTE: O(N^2 r) — only called during WACA discretisation precompute,
    /// not the hot per-token step.
    pub fn a_low_rank_dense(&self) -> Vec<f32> {
        let n = SSM_N;
        let r = SSM_RANK;
        let mut out = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += self.u[i * r + k] * self.v[j * r + k];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        let base = self.s.len()
            + self.u.len()
            + self.v.len()
            + self.b.len()
            + self.c.len()
            + self.d.len()
            + 1;
        let delta_proj = self.delta_proj_w.as_ref().map_or(0, |w| w.len() + 1);
        base + delta_proj
    }

    /// Create parameters from HiPPO-LegS initialization.
    ///
    /// Uses the HiPPO Legendre-S matrix decomposed into diagonal + low-rank form
    /// for principled long-range memory initialization.
    pub fn new_hippo(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = SSM_N;
        let d = SSM_D;
        let p = SSM_P;

        // Get HiPPO-derived S, U, V
        let (s, u, v) = crate::hippo::hippo_ssm_params();

        // B, C, D same as new_stable
        let mut b = vec![0.0f32; n * d];
        rng.fill_kaiming_uniform(&mut b, d);

        let mut c = vec![0.0f32; p * n];
        rng.fill_kaiming_uniform(&mut c, n);

        let d_mat = vec![0.0f32; p * d];

        // Use a slightly larger delta for HiPPO (0.01 is typical for Legendre)
        let log_delta = 0.01f32.ln();

        Self {
            s,
            u,
            v,
            b,
            c,
            d: d_mat,
            log_delta,
            delta_proj_w: None,
            delta_proj_b: None,
        }
    }

    /// Enable selective scan by adding delta projection parameters.
    pub fn enable_selective(&mut self, seed: u64) {
        let mut rng = Rng::new(seed);
        let mut w = vec![0.0f32; SSM_D];
        rng.fill_normal(&mut w, 0.0, 0.01);
        self.delta_proj_w = Some(w);
        self.delta_proj_b = Some(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_init_a_diag_negative() {
        let p = SsmParams::new_stable(0);
        let ad = p.a_diag();
        // All diagonal entries must be negative (stability)
        assert!(ad.iter().all(|&v| v < 0.0), "A_diag has non-negative entry");
    }

    #[test]
    fn delta_in_range() {
        let p = SsmParams::new_stable(42);
        let delta = p.delta();
        assert!(delta > 0.0 && delta <= 1.0, "delta={delta}");
    }

    #[test]
    fn param_count_positive() {
        let p = SsmParams::new_stable(0);
        assert!(p.param_count() > 0);
    }

    #[test]
    fn sizes_match_constants() {
        let p = SsmParams::new_stable(1);
        assert_eq!(p.s.len(), SSM_N);
        assert_eq!(p.u.len(), SSM_N * SSM_RANK);
        assert_eq!(p.v.len(), SSM_N * SSM_RANK);
        assert_eq!(p.b.len(), SSM_N * SSM_D);
        assert_eq!(p.c.len(), SSM_P * SSM_N);
        assert_eq!(p.d.len(), SSM_P * SSM_D);
    }
}

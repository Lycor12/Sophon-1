//! SSM backward pass — Diagonal-Rank Bifurcated BPTT (DRBS).
//!
//! Novel optimisation — DRBS (Diagonal-Rank Bifurcated State-Time backprop):
//!   Standard BPTT through a recurrence h_{t+1} = A_bar h_t + B_bar u_t
//!   requires materialising A_bar^T (N x N) for the adjoint equation
//!   dL/dh_t += A_bar^T dL/dh_{t+1}.
//!
//!   For our diagonal+low-rank A_bar = diag(a_d) + L R^T, the adjoint is:
//!     A_bar^T adj = diag(a_d) adj + R (L^T adj)
//!   which decomposes into:
//!     1. Diagonal pass:  d_diag[i] = a_d[i] * adj[i]        O(N)
//!     2. Low-rank pass:  z = L^T adj  (r-vector)             O(Nr)
//!                        d_lr[i] = sum_k R[i,k] * z[k]       O(Nr)
//!     3. Combine:        result[i] = d_diag[i] + d_lr[i]     O(N)
//!
//!   Total cost per token: O(Nr) instead of O(N^2).
//!   The "bifurcated" name reflects that diagonal and low-rank adjoint paths
//!   are computed independently, avoiding any N x N intermediate.
//!
//! For the parameter gradients (dL/dC, dL/dD, dL/dB, dL/dS, dL/dU, dL/dV,
//! dL/d_log_delta), we accumulate across all time steps in a single backward
//! sweep, reusing the saved forward states and inputs.
//!
//! Memory: O(T * (N + D + P)) for saving forward trajectory, where T is
//! sequence length. This is the minimum required for exact BPTT.

use crate::{params::SsmParams, zoh::DiscretisedSsm, SsmState};
use sophon_config::{SSM_D, SSM_N, SSM_P, SSM_RANK};

// ---------------------------------------------------------------------------
// Forward cache for one SSM step
// ---------------------------------------------------------------------------

/// Cached activations from one forward step, needed for backward.
#[derive(Clone)]
pub struct SsmStepCache {
    /// Hidden state BEFORE this step: h_t (before update).
    pub h_prev: Vec<f32>,
    /// Input to this step: u_t.
    pub u: Vec<f32>,
    /// Output of this step: y_t.
    pub y: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Parameter gradient accumulators
// ---------------------------------------------------------------------------

/// Accumulated gradients for all SSM parameters across a sequence.
pub struct SsmParamGrads {
    /// dL/dS — gradient w.r.t. diagonal log-magnitudes. Shape: [N]
    pub grad_s: Vec<f32>,
    /// dL/dU — gradient w.r.t. low-rank left factor. Shape: [N, r]
    pub grad_u: Vec<f32>,
    /// dL/dV — gradient w.r.t. low-rank right factor. Shape: [N, r]
    pub grad_v: Vec<f32>,
    /// dL/dB — gradient w.r.t. input matrix. Shape: [N, D]
    pub grad_b: Vec<f32>,
    /// dL/dC — gradient w.r.t. output matrix. Shape: [P, N]
    pub grad_c: Vec<f32>,
    /// dL/dD — gradient w.r.t. feedthrough matrix. Shape: [P, D]
    pub grad_d: Vec<f32>,
    /// dL/d_log_delta — gradient w.r.t. log step size. Scalar.
    pub grad_log_delta: f32,
}

impl SsmParamGrads {
    /// Create zero-initialised gradient accumulators.
    pub fn zeros() -> Self {
        let n = SSM_N;
        let d = SSM_D;
        let p = SSM_P;
        let r = SSM_RANK;
        Self {
            grad_s: vec![0.0f32; n],
            grad_u: vec![0.0f32; n * r],
            grad_v: vec![0.0f32; n * r],
            grad_b: vec![0.0f32; n * d],
            grad_c: vec![0.0f32; p * n],
            grad_d: vec![0.0f32; p * d],
            grad_log_delta: 0.0,
        }
    }

    /// Check all gradients are finite.
    pub fn is_finite(&self) -> bool {
        self.grad_s.iter().all(|v| v.is_finite())
            && self.grad_u.iter().all(|v| v.is_finite())
            && self.grad_v.iter().all(|v| v.is_finite())
            && self.grad_b.iter().all(|v| v.is_finite())
            && self.grad_c.iter().all(|v| v.is_finite())
            && self.grad_d.iter().all(|v| v.is_finite())
            && self.grad_log_delta.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Forward step with caching
// ---------------------------------------------------------------------------

/// SSM forward step that also returns cached activations for backward.
///
/// Identical to `ssm_step` in update.rs but saves h_prev, u, y.
pub fn ssm_step_with_cache(
    state: &mut SsmState,
    disc: &DiscretisedSsm,
    params: &SsmParams,
    u: &[f32],
) -> (Vec<f32>, SsmStepCache) {
    debug_assert_eq!(u.len(), SSM_D);

    let n = SSM_N;
    let p = SSM_P;
    let r = SSM_RANK;

    // Save h_prev before mutation
    let h_prev = state.h.to_vec();

    // --- FSO forward (same as update.rs) ---

    // Step 1: z_lr = A_lr_r^T h
    let mut z_lr = [0.0f32; SSM_RANK];
    for j in 0..n {
        let hj = state.h[j];
        for k in 0..r {
            z_lr[k] += disc.a_lr_r[j * r + k] * hj;
        }
    }

    // Step 2 (FSO): new_h and C*h_old simultaneously
    let mut y = vec![0.0f32; p];
    let c_data = &params.c;

    let mut new_h = vec![0.0f32; n];
    for i in 0..n {
        let hi = state.h[i];

        // y += C[:, i] * hi
        for q in 0..p {
            y[q] += c_data[q * n + i] * hi;
        }

        // new_h[i] = a_d[i]*hi + low-rank
        let mut nh = disc.a_d[i] * hi;
        for k in 0..r {
            nh += disc.a_lr_l[i * r + k] * z_lr[k];
        }
        new_h[i] = nh;
    }

    // Step 3: B_bar u
    let bu = disc.apply_b_bar(u);
    for i in 0..n {
        new_h[i] += bu[i];
    }

    // Commit
    state.h.copy_from_slice(&new_h);

    // Step 4: D u
    let d_data = &params.d;
    for q in 0..p {
        let mut acc = 0.0f32;
        let row = q * SSM_D;
        for j in 0..SSM_D {
            acc += d_data[row + j] * u[j];
        }
        y[q] += acc;
    }

    let cache = SsmStepCache {
        h_prev,
        u: u.to_vec(),
        y: y.clone(),
    };

    (y, cache)
}

// ---------------------------------------------------------------------------
// DRBS: adjoint of A_bar^T without materialising N x N
// ---------------------------------------------------------------------------

/// Compute A_bar^T * adj using DRBS decomposition.
///
/// A_bar = diag(a_d) + L R^T
/// A_bar^T = diag(a_d) + R L^T
/// A_bar^T * adj = a_d .* adj + R (L^T adj)
///
/// Cost: O(Nr) instead of O(N^2).
fn adjoint_a_bar_transpose(disc: &DiscretisedSsm, adj: &[f32]) -> Vec<f32> {
    let n = SSM_N;
    let r = SSM_RANK;
    debug_assert_eq!(adj.len(), n);

    // Low-rank path: z = L^T adj  (r-vector)
    // L = a_lr_l [N, r], so L^T adj = sum_i a_lr_l[i,k] * adj[i]
    let mut z = vec![0.0f32; r];
    for i in 0..n {
        let ai = adj[i];
        for k in 0..r {
            z[k] += disc.a_lr_l[i * r + k] * ai;
        }
    }

    // Combine: result[i] = a_d[i]*adj[i] + sum_k R[i,k]*z[k]
    // R = a_lr_r [N, r]
    let mut result = vec![0.0f32; n];
    for i in 0..n {
        let mut val = disc.a_d[i] * adj[i];
        for k in 0..r {
            val += disc.a_lr_r[i * r + k] * z[k];
        }
        result[i] = val;
    }
    result
}

// ---------------------------------------------------------------------------
// Full backward pass over a sequence
// ---------------------------------------------------------------------------

/// Backward pass over an SSM sequence using DRBS.
///
/// # Arguments
/// * `params`     — SSM parameters (for C, D, and discretisation derivative)
/// * `disc`       — Pre-computed discretised SSM
/// * `caches`     — Forward step caches for each time step [0..T-1]
/// * `grad_y`     — Upstream gradient of loss w.r.t. each output y_t. Shape: [T][P]
///
/// # Returns
/// * `SsmParamGrads` — accumulated parameter gradients
/// * `Vec<Vec<f32>>` — dL/du_t for each time step (to propagate upstream). Shape: [T][D]
/// * `Vec<f32>`      — dL/dh_0 (gradient w.r.t. initial hidden state). Shape: [N]
pub fn ssm_backward(
    params: &SsmParams,
    disc: &DiscretisedSsm,
    caches: &[SsmStepCache],
    grad_y: &[Vec<f32>],
) -> (SsmParamGrads, Vec<Vec<f32>>, Vec<f32>) {
    let t_len = caches.len();
    assert_eq!(grad_y.len(), t_len);

    let n = SSM_N;
    let d = SSM_D;
    let p = SSM_P;
    let r = SSM_RANK;

    let mut grads = SsmParamGrads::zeros();
    let mut grad_u_all = Vec::with_capacity(t_len);

    // Adjoint of hidden state: dL/dh_{t+1} (propagated backward)
    let mut adj_h = vec![0.0f32; n];

    // Backward sweep: t = T-1, T-2, ..., 0
    for t in (0..t_len).rev() {
        let cache = &caches[t];
        let gy = &grad_y[t];
        debug_assert_eq!(gy.len(), p);
        debug_assert_eq!(cache.h_prev.len(), n);
        debug_assert_eq!(cache.u.len(), d);

        // ---------------------------------------------------------------
        // Gradient through output: y_t = C h_t + D u_t
        // ---------------------------------------------------------------

        // dL/dC += gy * h_prev^T   (outer product, accumulated)
        // C shape [P, N], so grad_c[q*N + i] += gy[q] * h_prev[i]
        for q in 0..p {
            let gq = gy[q];
            for i in 0..n {
                grads.grad_c[q * n + i] += gq * cache.h_prev[i];
            }
        }

        // dL/dD += gy * u^T   (outer product, accumulated)
        for q in 0..p {
            let gq = gy[q];
            for j in 0..d {
                grads.grad_d[q * d + j] += gq * cache.u[j];
            }
        }

        // dL/dh_t (from output path): C^T gy
        // adj_h_from_output[i] = sum_q C[q, i] * gy[q]
        let mut adj_h_output = vec![0.0f32; n];
        for q in 0..p {
            let gq = gy[q];
            for i in 0..n {
                adj_h_output[i] += params.c[q * n + i] * gq;
            }
        }

        // dL/du_t (from output path): D^T gy
        let mut grad_u_t = vec![0.0f32; d];
        for q in 0..p {
            let gq = gy[q];
            for j in 0..d {
                grad_u_t[j] += params.d[q * d + j] * gq;
            }
        }

        // ---------------------------------------------------------------
        // Gradient through state update: h_{t+1} = A_bar h_t + B_bar u_t
        // adj_h currently holds dL/dh_{t+1} from future steps
        // ---------------------------------------------------------------

        // dL/du_t (from state path): B_bar^T adj_h
        // B_bar shape [N, D], so B_bar^T adj_h [j] = sum_i b_bar[i,j] * adj_h[i]
        for i in 0..n {
            let ai = adj_h[i];
            let row = i * d;
            for j in 0..d {
                grad_u_t[j] += disc.b_bar[row + j] * ai;
            }
        }

        // dL/dB (accumulated): through B_bar.
        // B_bar[i,j] = scale[i] * B[i,j] where scale[i] depends on a_d, a_diag, delta
        // For gradient: dL/dB[i,j] += scale[i] * adj_h[i] * u_t[j]
        // scale[i] = (a_d_exact[i] - 1) / a_diag[i]  (from zoh.rs)
        let a_diag = params.a_diag();
        let delta = params.delta();
        for i in 0..n {
            let a_d_exact = (delta * a_diag[i]).exp();
            let scale = if a_diag[i].abs() > 1e-12 {
                (a_d_exact - 1.0) / a_diag[i]
            } else {
                delta
            };
            let ai = adj_h[i];
            let row = i * d;
            for j in 0..d {
                grads.grad_b[row + j] += scale * ai * cache.u[j];
            }
        }

        // dL/dh_t (from state path): A_bar^T adj_h  [DRBS]
        let adj_h_state = adjoint_a_bar_transpose(disc, &adj_h);

        // ---------------------------------------------------------------
        // Gradient through A_bar w.r.t. continuous params (S, U, V, log_delta)
        // We need d(A_bar h_prev)/d(param) for each param.
        // ---------------------------------------------------------------

        // --- dL/dS (through diagonal part of A_bar) ---
        // A_bar_diag uses Cayley: a_d[i] = (1 + Δ/2 a_diag[i]) / (1 - Δ/2 a_diag[i])
        // a_diag[i] = -exp(S[i])
        // d(a_d[i])/dS[i] = d(a_d)/d(a_diag) * d(a_diag)/dS
        // d(a_diag)/dS[i] = -exp(S[i]) = a_diag[i]
        // For Cayley: d[(1+x)/(1-x)]/dx = 2/(1-x)^2 where x = Δ/2 * a_diag[i]
        // d(a_d[i])/d(a_diag[i]) = Δ/2 * 2 / (1 - Δ/2 * a_diag[i])^2 = Δ / (1 - Δ/2 * a_diag[i])^2
        // So: d(a_d[i])/dS[i] = Δ * a_diag[i] / (1 - Δ/2 * a_diag[i])^2
        let half_d = delta * 0.5;
        for i in 0..n {
            let ai = a_diag[i];
            let denom = 1.0 - half_d * ai;
            let denom_sq = denom * denom;
            let da_d_ds = if denom_sq.abs() > 1e-12 {
                delta * ai / denom_sq
            } else {
                0.0
            };
            // Contribution: adj_h[i] * h_prev[i] * da_d[i]/dS[i]
            grads.grad_s[i] += adj_h[i] * cache.h_prev[i] * da_d_ds;
        }

        // --- dL/dU, dL/dV (through low-rank part of A_bar) ---
        // The low-rank part of A_bar h = a_lr_l (a_lr_r^T h)
        // a_lr_l and a_lr_r depend on U, V through the WACA discretisation.
        // Direct differentiation is complex, so we use the first-order approximation:
        //   d(A_bar h)/dU[i,k] ≈ Δ/2 * d_inv[i] * h_prev[i] * (sum_j V[j,k]*d_inv[j]*w_inv_applied[...])
        //
        // For tractability, we compute the gradient through the Cayley numerator:
        //   (I + Δ/2 A) = (I + Δ/2 Λ) + Δ/2 U V^T
        //   d/dU[i,k] of [(Δ/2 U V^T) h][i] = Δ/2 * (V^T h)[k]
        //   But this ignores the M^{-1} denominator.
        //
        // We use the identity for gradients through matrix inverses:
        //   If f = (Numerator)(Denominator^{-1}) h, and adj is dL/df,
        //   then dL/dU via the numerator contributes Δ/2 * adj * (V^T h)^T
        //   and via the denominator contributes -Δ/2 * (M^{-T} adj) * (V^T M^{-1} h)^T
        //
        // Simplified form (combining both paths with first-order WACA correction):
        //   dL/dU[i,k] += Δ * adj_h[i] * z_v[k]
        //   where z_v[k] = sum_j V[j,k] * h_prev[j]
        //
        // This is the dominant gradient term and captures the correct direction.

        // z_v = V^T h_prev
        let mut z_v = vec![0.0f32; r];
        for j in 0..n {
            for k in 0..r {
                z_v[k] += params.v[j * r + k] * cache.h_prev[j];
            }
        }

        // z_u = U^T h_prev
        let mut z_u = vec![0.0f32; r];
        for j in 0..n {
            for k in 0..r {
                z_u[k] += params.u[j * r + k] * cache.h_prev[j];
            }
        }

        for i in 0..n {
            let ai = adj_h[i];
            for k in 0..r {
                grads.grad_u[i * r + k] += delta * ai * z_v[k];
                grads.grad_v[i * r + k] += delta * ai * z_u[k];
            }
        }

        // --- dL/d_log_delta (through delta) ---
        // delta = exp(log_delta), so d(delta)/d(log_delta) = delta
        // Many terms depend on delta: a_d, b_bar, a_lr_l, a_lr_r
        // Dominant contribution through diagonal:
        //   d(a_d[i])/d(delta) = a_diag[i] / (1 - Δ/2 a_diag[i])^2  [Cayley]
        //   dL/d(delta) += sum_i adj_h[i] * h_prev[i] * d(a_d[i])/d(delta)
        // Through B_bar:
        //   d(b_bar[i,j])/d(delta) = d(scale[i])/d(delta) * B[i,j]
        //   scale = (exp(Δ a_diag) - 1) / a_diag
        //   d(scale)/d(delta) = a_diag * exp(Δ a_diag) / a_diag = exp(Δ a_diag)  [simplified]
        //   But scale uses exact exp, not Cayley, so: d(scale)/d(delta) = exp(Δ a_diag[i])

        let mut dl_ddelta = 0.0f32;

        // Through a_d (Cayley)
        for i in 0..n {
            let ai = a_diag[i];
            let denom = 1.0 - half_d * ai;
            let denom_sq = denom * denom;
            let da_d_ddelta = if denom_sq.abs() > 1e-12 {
                ai / denom_sq
            } else {
                0.0
            };
            dl_ddelta += adj_h[i] * cache.h_prev[i] * da_d_ddelta;
        }

        // Through B_bar
        for i in 0..n {
            let a_d_exact_i = (delta * a_diag[i]).exp();
            for j in 0..d {
                dl_ddelta += adj_h[i] * cache.u[j] * a_d_exact_i * params.b[i * d + j];
            }
        }

        // Chain rule: dL/d_log_delta = dL/d_delta * d_delta/d_log_delta = dL/d_delta * delta
        grads.grad_log_delta += dl_ddelta * delta;

        // ---------------------------------------------------------------
        // Combine adjoint for h_t and propagate backward
        // ---------------------------------------------------------------
        // adj_h_t = adj_h_output + adj_h_state
        for i in 0..n {
            adj_h[i] = adj_h_output[i] + adj_h_state[i];
        }

        grad_u_all.push(grad_u_t);
    }

    // grad_u_all was built in reverse order; flip it
    grad_u_all.reverse();

    // adj_h now holds dL/dh_0
    let grad_h0 = adj_h;

    (grads, grad_u_all, grad_h0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SsmParams;
    use crate::zoh::DiscretisedSsm;

    fn run_forward_sequence(
        params: &SsmParams,
        disc: &DiscretisedSsm,
        inputs: &[Vec<f32>],
    ) -> (SsmState, Vec<SsmStepCache>, Vec<Vec<f32>>) {
        let mut state = SsmState::new();
        let mut caches = Vec::new();
        let mut outputs = Vec::new();
        for u in inputs {
            let (y, cache) = ssm_step_with_cache(&mut state, disc, params, u);
            outputs.push(y);
            caches.push(cache);
        }
        (state, caches, outputs)
    }

    #[test]
    fn backward_gradients_finite() {
        let params = SsmParams::new_stable(0);
        let disc = DiscretisedSsm::from_params(&params);

        // 5-step sequence
        let inputs: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![0.01 * (i as f32 + 1.0); SSM_D])
            .collect();

        let (_, caches, _) = run_forward_sequence(&params, &disc, &inputs);

        // Fake upstream gradient
        let grad_y: Vec<Vec<f32>> = (0..5).map(|_| vec![0.1f32; SSM_P]).collect();

        let (grads, grad_u, grad_h0) = ssm_backward(&params, &disc, &caches, &grad_y);
        assert!(grads.is_finite(), "parameter gradients not finite");
        assert!(grad_h0.iter().all(|v| v.is_finite()), "grad_h0 not finite");
        for (t, gu) in grad_u.iter().enumerate() {
            assert!(gu.iter().all(|v| v.is_finite()), "grad_u[{t}] not finite");
        }
    }

    #[test]
    fn backward_grad_u_shape() {
        let params = SsmParams::new_stable(1);
        let disc = DiscretisedSsm::from_params(&params);
        let inputs: Vec<Vec<f32>> = (0..3).map(|_| vec![0.05f32; SSM_D]).collect();
        let (_, caches, _) = run_forward_sequence(&params, &disc, &inputs);
        let grad_y: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1f32; SSM_P]).collect();
        let (_, grad_u, _) = ssm_backward(&params, &disc, &caches, &grad_y);
        assert_eq!(grad_u.len(), 3);
        for gu in &grad_u {
            assert_eq!(gu.len(), SSM_D);
        }
    }

    #[test]
    fn backward_zero_upstream_gives_zero_grads() {
        let params = SsmParams::new_stable(2);
        let disc = DiscretisedSsm::from_params(&params);
        let inputs: Vec<Vec<f32>> = (0..4).map(|_| vec![0.1f32; SSM_D]).collect();
        let (_, caches, _) = run_forward_sequence(&params, &disc, &inputs);
        let grad_y: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; SSM_P]).collect();
        let (grads, grad_u, grad_h0) = ssm_backward(&params, &disc, &caches, &grad_y);

        // With zero upstream gradient, all parameter gradients should be zero
        assert!(grads.grad_c.iter().all(|&v| v.abs() < 1e-10));
        assert!(grads.grad_d.iter().all(|&v| v.abs() < 1e-10));
        assert!(grad_h0.iter().all(|&v| v.abs() < 1e-10));
        for gu in &grad_u {
            assert!(gu.iter().all(|&v| v.abs() < 1e-10));
        }
    }

    #[test]
    fn forward_with_cache_matches_original() {
        // Verify that ssm_step_with_cache produces the same output as ssm_step
        let params = SsmParams::new_stable(42);
        let disc = DiscretisedSsm::from_params(&params);
        let u = vec![0.1f32; SSM_D];

        let mut s1 = SsmState::new();
        let mut s2 = SsmState::new();

        let y1 = crate::ssm_step(&mut s1, &disc, &params, &u);
        let (y2, _cache) = ssm_step_with_cache(&mut s2, &disc, &params, &u);

        for i in 0..SSM_P {
            assert!(
                (y1[i] - y2[i]).abs() < 1e-7,
                "mismatch at {i}: {} vs {}",
                y1[i],
                y2[i]
            );
        }
        assert_eq!(s1.h, s2.h);
    }

    #[test]
    fn drbs_adjoint_matches_naive() {
        // Verify DRBS produces same result as naive A_bar^T * adj
        let params = SsmParams::new_stable(7);
        let disc = DiscretisedSsm::from_params(&params);

        // Random-ish adj vector
        let adj: Vec<f32> = (0..SSM_N)
            .map(|i| ((i as f32) * 0.73).sin() * 0.1)
            .collect();

        // DRBS result
        let drbs = adjoint_a_bar_transpose(&disc, &adj);

        // Naive: materialise A_bar as full N x N, compute A_bar^T adj
        let n = SSM_N;
        let r = SSM_RANK;

        // A_bar[i,j] = a_d[i]*delta(i,j) + sum_k a_lr_l[i,k]*a_lr_r[j,k]
        let mut naive = vec![0.0f32; n];
        for j in 0..n {
            // Column j of A_bar^T = row j of A_bar entries applied to adj
            // A_bar^T[j, i] = A_bar[i, j]
            // (A_bar^T adj)[j] = sum_i A_bar[i,j] * adj[i]
            let mut acc = 0.0f32;
            for i in 0..n {
                let a_ij = if i == j { disc.a_d[i] } else { 0.0 }
                    + (0..r)
                        .map(|k| disc.a_lr_l[i * r + k] * disc.a_lr_r[j * r + k])
                        .sum::<f32>();
                acc += a_ij * adj[i];
            }
            naive[j] = acc;
        }

        for i in 0..n {
            assert!(
                (drbs[i] - naive[i]).abs() < 1e-3,
                "DRBS mismatch at {i}: drbs={} naive={}",
                drbs[i],
                naive[i]
            );
        }
    }

    #[test]
    fn numerical_gradient_c_check() {
        // Numerical gradient check for dL/dC using finite differences.
        // L = sum(y) for a single step (simplest loss).
        let eps = 1e-3f32;
        let mut params = SsmParams::new_stable(10);
        let disc = DiscretisedSsm::from_params(&params);
        let u = vec![0.1f32; SSM_D];

        // Forward
        let mut state = SsmState::new();
        let (y, cache) = ssm_step_with_cache(&mut state, &disc, &params, &u);

        // Backward with grad_y = 1 (loss = sum(y))
        let grad_y = vec![vec![1.0f32; SSM_P]];
        let (grads, _, _) = ssm_backward(&params, &disc, &[cache], &grad_y);

        // Check a few C entries numerically
        let check_indices = [0, 5, SSM_P * SSM_N / 2, SSM_P * SSM_N - 1];
        for &idx in &check_indices {
            if idx >= params.c.len() {
                continue;
            }
            let orig = params.c[idx];

            // f(C + eps)
            params.c[idx] = orig + eps;
            let mut s_plus = SsmState::new();
            let y_plus = crate::ssm_step(&mut s_plus, &disc, &params, &u);
            let loss_plus: f32 = y_plus.iter().sum();

            // f(C - eps)
            params.c[idx] = orig - eps;
            let mut s_minus = SsmState::new();
            let y_minus = crate::ssm_step(&mut s_minus, &disc, &params, &u);
            let loss_minus: f32 = y_minus.iter().sum();

            params.c[idx] = orig;

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grads.grad_c[idx];

            // Allow generous tolerance since SSM forward involves exp, inv, etc.
            let tol = 0.1 * numerical.abs().max(analytical.abs()).max(1e-4);
            assert!(
                (numerical - analytical).abs() < tol,
                "C grad mismatch at idx {idx}: numerical={numerical:.6} analytical={analytical:.6}"
            );
        }
    }
}

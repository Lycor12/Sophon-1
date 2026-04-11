//! TSM-SGD optimizer implementation.
//!
//! See lib.rs for the full description of the four novel mechanisms.

use crate::param_group::{ParamGroup, ParamKind};

// ---------------------------------------------------------------------------
// Momentum buffer for one parameter group
// ---------------------------------------------------------------------------

/// Momentum state for one parameter vector.
pub struct MomentumState {
    /// Velocity buffer (same shape as params).
    pub velocity: Vec<f32>,
    /// Step count for this group.
    pub step: u64,
}

impl MomentumState {
    pub fn new(size: usize) -> Self {
        Self {
            velocity: vec![0.0f32; size],
            step: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TsmSgd
// ---------------------------------------------------------------------------

/// TSM-SGD optimizer with dual-rate momentum and parameter-aware mechanisms.
pub struct TsmSgd {
    /// Global learning rate multiplier (scales per-group LR).
    pub global_lr: f32,
    /// Gradient clipping threshold (max L2 norm per group).
    pub grad_clip: f32,
}

impl TsmSgd {
    /// Create a new TSM-SGD optimizer.
    pub fn new(global_lr: f32, grad_clip: f32) -> Self {
        Self {
            global_lr,
            grad_clip,
        }
    }

    /// Perform one optimiser step for a parameter group.
    ///
    /// # Arguments
    /// * `params`   — mutable parameter slice
    /// * `grad`     — gradient slice (same length as params)
    /// * `group`    — parameter group configuration
    /// * `momentum` — momentum state (updated in-place)
    /// * `knot_spans` — optional knot span info for coefficient coupling.
    ///                  If provided and `group.kind == KanCoeff`, modulates grad.
    pub fn step(
        &self,
        params: &mut [f32],
        grad: &[f32],
        group: &ParamGroup,
        momentum: &mut MomentumState,
        knot_spans: Option<&[f32]>,
    ) {
        let n = params.len();
        debug_assert_eq!(grad.len(), n);
        debug_assert_eq!(momentum.velocity.len(), n);

        let lr = self.global_lr * group.lr;
        let beta = group.momentum();

        // --- Step 1: prepare effective gradient ---
        let mut g = grad.to_vec();

        // Weight decay: g += wd * params
        if group.weight_decay > 0.0 {
            for i in 0..n {
                g[i] += group.weight_decay * params[i];
            }
        }

        // Mechanism 3: Knot-aware coefficient coupling
        if group.kind == ParamKind::KanCoeff {
            if let Some(spans) = knot_spans {
                apply_knot_coupling(&mut g, spans);
            }
        }

        // Mechanism 2: Ternary gradient projection
        if group.ternary_project {
            apply_ternary_projection(&mut g, params, group.ternary_threshold);
        }

        // Mechanism 4: Spectral norm clipping for UV factors
        if group.spectral_clip {
            apply_spectral_clip(&mut g, group.spectral_max_norm);
        }

        // Global gradient clipping (L2 norm)
        if self.grad_clip > 0.0 {
            let norm_sq: f32 = g.iter().map(|&v| v * v).sum();
            let norm = norm_sq.sqrt();
            if norm > self.grad_clip {
                let scale = self.grad_clip / norm;
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // --- Step 2: momentum update ---
        // v_t = β * v_{t-1} + g
        // params -= lr * v_t
        for i in 0..n {
            momentum.velocity[i] = beta * momentum.velocity[i] + g[i];
            params[i] -= lr * momentum.velocity[i];
        }

        momentum.step += 1;
    }
}

// ---------------------------------------------------------------------------
// Mechanism 2: Ternary gradient projection
// ---------------------------------------------------------------------------

/// Project gradient away from ternary decision boundaries.
///
/// Near quantisation thresholds (|w| ≈ threshold), the gradient component
/// that would push w across the boundary is damped. This prevents training
/// oscillation at ternary boundaries.
///
/// The damping factor is: 1 - exp(-distance_to_boundary^2 / (2 * sigma^2))
/// where sigma = threshold * 0.3 (30% of the threshold width).
fn apply_ternary_projection(grad: &mut [f32], params: &[f32], threshold: f32) {
    let sigma = threshold * 0.3;
    let inv_2sigma_sq = if sigma > 1e-12 {
        0.5 / (sigma * sigma)
    } else {
        0.0
    };

    for i in 0..grad.len() {
        let w = params[i];
        // Distance to nearest ternary boundary
        // Boundaries at: -threshold, +threshold (between 0 and ±1 regions)
        let dist_neg = (w.abs() - threshold).abs();
        let dist = dist_neg;

        // Damping: close to boundary -> small factor -> gradient is damped
        let factor = 1.0 - (-dist * dist * inv_2sigma_sq).exp();
        grad[i] *= factor;
    }
}

// ---------------------------------------------------------------------------
// Mechanism 3: Knot-aware coefficient coupling
// ---------------------------------------------------------------------------

/// Modulate KAN coefficient gradients by local knot density.
///
/// Dense knot regions (small spans) get slower updates.
/// Factor: min(1.0, local_span / median_span)
/// Small span -> small factor -> slower update in dense regions.
///
/// `knot_spans` should contain the span width for each coefficient group.
/// If it's shorter than `grad`, it wraps cyclically.
fn apply_knot_coupling(grad: &mut [f32], knot_spans: &[f32]) {
    if knot_spans.is_empty() {
        return;
    }

    // Compute median span
    let mut sorted = knot_spans.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };

    if median < 1e-12 {
        return;
    }

    // Modulate gradients: small span -> small factor -> slower update
    for i in 0..grad.len() {
        let span = knot_spans[i % knot_spans.len()];
        if span > 1e-12 {
            let factor = (span / median).min(1.0);
            grad[i] *= factor;
        }
    }
}

// ---------------------------------------------------------------------------
// Mechanism 4: Spectral norm clipping
// ---------------------------------------------------------------------------

/// Clip gradient vector by its L2 norm (proxy for spectral norm of rank-1 update).
///
/// For UV low-rank factors, the gradient update dU represents a rank-1
/// perturbation to U. The spectral norm of dU is bounded by ||dU||_F
/// (Frobenius norm), which equals the L2 norm of the flattened gradient.
/// We clip ||dU||_F <= max_norm.
fn apply_spectral_clip(grad: &mut [f32], max_norm: f32) {
    let norm_sq: f32 = grad.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for v in grad.iter_mut() {
            *v *= scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_group::ParamKind;

    #[test]
    fn basic_sgd_step_moves_params() {
        let opt = TsmSgd::new(1.0, 0.0);
        let group = ParamGroup::default_for(ParamKind::KanCoeff);
        let mut params = vec![1.0f32; 10];
        let grad = vec![0.1f32; 10];
        let mut mom = MomentumState::new(10);

        let before = params.clone();
        opt.step(&mut params, &grad, &group, &mut mom, None);

        // Params should have moved
        assert!(params
            .iter()
            .zip(before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8));
        // Should have moved in negative gradient direction
        for i in 0..10 {
            assert!(params[i] < before[i], "param[{i}] should decrease");
        }
    }

    #[test]
    fn momentum_accumulates() {
        let opt = TsmSgd::new(1.0, 0.0);
        let group = ParamGroup::default_for(ParamKind::SsmDiag);
        let mut params = vec![0.0f32; 5];
        let grad = vec![1.0f32; 5];
        let mut mom = MomentumState::new(5);

        // Step 1
        opt.step(&mut params, &grad, &group, &mut mom, None);
        let after_1 = params.clone();

        // Step 2: momentum should cause a larger move
        opt.step(&mut params, &grad, &group, &mut mom, None);
        let move_1 = (after_1[0] - 0.0).abs();
        let move_2 = (params[0] - after_1[0]).abs();
        assert!(
            move_2 > move_1,
            "momentum should increase: move_1={move_1} move_2={move_2}"
        );
    }

    #[test]
    fn dual_rate_different_velocities() {
        let opt = TsmSgd::new(1.0, 0.0);
        let kan_group = ParamGroup::default_for(ParamKind::KanCoeff);
        let ssm_group = ParamGroup::default_for(ParamKind::SsmDiag);

        let mut p_kan = vec![0.0f32; 5];
        let mut p_ssm = vec![0.0f32; 5];
        let grad = vec![1.0f32; 5];
        let mut m_kan = MomentumState::new(5);
        let mut m_ssm = MomentumState::new(5);

        // Run 3 steps
        for _ in 0..3 {
            opt.step(&mut p_kan, &grad, &kan_group, &mut m_kan, None);
            opt.step(&mut p_ssm, &grad, &ssm_group, &mut m_ssm, None);
        }

        // Velocities should differ due to different beta
        assert!(
            (m_kan.velocity[0] - m_ssm.velocity[0]).abs() > 0.01,
            "dual-rate should produce different velocities"
        );
    }

    #[test]
    fn ternary_projection_damps_near_boundary() {
        let mut grad_near = vec![1.0f32; 3];
        let params_near = vec![0.5f32; 3]; // exactly at threshold
        apply_ternary_projection(&mut grad_near, &params_near, 0.5);

        let mut grad_far = vec![1.0f32; 3];
        let params_far = vec![0.0f32; 3]; // far from threshold
        apply_ternary_projection(&mut grad_far, &params_far, 0.5);

        // Gradient near boundary should be more damped
        assert!(
            grad_near[0].abs() < grad_far[0].abs(),
            "near-boundary gradient should be smaller: near={} far={}",
            grad_near[0],
            grad_far[0]
        );
    }

    #[test]
    fn knot_coupling_damps_dense_regions() {
        let mut grad = vec![1.0f32; 4];
        // Spans: [0.1, 0.5, 0.5, 0.5] — first coefficient is in dense region
        let spans = vec![0.1f32, 0.5, 0.5, 0.5];
        apply_knot_coupling(&mut grad, &spans);

        // First coefficient should be damped (small span -> slow update)
        // Others should be near 1.0 (at or above median)
        assert!(
            grad[0] < grad[1],
            "dense region should be damped: {} vs {}",
            grad[0],
            grad[1]
        );
    }

    #[test]
    fn spectral_clip_limits_norm() {
        let mut grad = vec![3.0f32; 100];
        let norm_before: f32 = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm_before > 1.0);

        apply_spectral_clip(&mut grad, 1.0);
        let norm_after: f32 = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_after - 1.0).abs() < 1e-5,
            "norm should be clipped to 1.0, got {norm_after}"
        );
    }

    #[test]
    fn weight_decay_increases_gradient() {
        let opt = TsmSgd::new(1.0, 0.0);
        let mut group = ParamGroup::default_for(ParamKind::KanCoeff);
        group.weight_decay = 0.1;

        let mut params = vec![2.0f32; 5];
        let grad = vec![0.0f32; 5]; // zero base gradient
        let mut mom = MomentumState::new(5);

        opt.step(&mut params, &grad, &group, &mut mom, None);

        // Weight decay should push params toward zero even with zero gradient
        for &p in &params {
            assert!(p < 2.0, "weight decay should shrink params");
        }
    }

    #[test]
    fn grad_clip_limits_update() {
        let opt = TsmSgd::new(1.0, 1.0); // clip at norm 1.0
        let group = ParamGroup::default_for(ParamKind::SsmMatrix);

        let mut params = vec![0.0f32; 100];
        let grad = vec![10.0f32; 100]; // large gradient
        let mut mom = MomentumState::new(100);

        opt.step(&mut params, &grad, &group, &mut mom, None);

        // The clipped gradient norm should be ≤ 1.0, so velocity ≤ 1.0
        let vel_norm: f32 = mom.velocity.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            vel_norm <= 1.0 + 1e-5,
            "velocity norm should be clipped: {vel_norm}"
        );
    }

    #[test]
    fn step_count_increments() {
        let opt = TsmSgd::new(1.0, 0.0);
        let group = ParamGroup::default_for(ParamKind::Embedding);
        let mut params = vec![0.0f32; 3];
        let grad = vec![0.1f32; 3];
        let mut mom = MomentumState::new(3);

        assert_eq!(mom.step, 0);
        opt.step(&mut params, &grad, &group, &mut mom, None);
        assert_eq!(mom.step, 1);
        opt.step(&mut params, &grad, &group, &mut mom, None);
        assert_eq!(mom.step, 2);
    }
}

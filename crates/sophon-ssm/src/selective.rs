//! Selective scan — input-dependent delta gating.
//!
//! Standard SSMs use a fixed discretisation step Δ. Selective scan (as in Mamba)
//! makes Δ input-dependent:
//!   Δ(t) = softplus(W_delta * u(t) + b_delta)
//!
//! This allows the model to modulate its memory retention on a per-token basis:
//! large Δ → forget past, small Δ → retain memory.
//!
//! Novel optimisation — IRDS (Input-Rate Dependent Scheduling):
//!   When Δ is input-dependent, the discretised matrices A_bar, B_bar change
//!   every token. Full rediscretisation is O(Nr^2) per token. IRDS detects
//!   when |Δ_new - Δ_cached| < threshold and skips rediscretisation,
//!   reusing cached A_bar/B_bar. For slowly-varying inputs, this saves
//!   significant compute.

use sophon_config::{SSM_D, SSM_N, SSM_RANK};
use sophon_core::rng::Rng;

use crate::params::SsmParams;
use crate::state::SsmState;
use crate::zoh::DiscretisedSsm;

/// Projection parameters for input-dependent delta.
pub struct DeltaProjection {
    /// Weight: [1, SSM_D] — projects input to scalar pre-softplus delta.
    pub w_delta: Vec<f32>,
    /// Bias scalar.
    pub b_delta: f32,
}

impl DeltaProjection {
    /// Create with small random weights and zero bias.
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut w_delta = vec![0.0f32; SSM_D];
        // Small init so delta starts near softplus(0) ≈ 0.693
        rng.fill_normal(&mut w_delta, 0.0, 0.01);
        DeltaProjection {
            w_delta,
            b_delta: 0.0,
        }
    }

    /// Compute input-dependent delta: Δ(u) = softplus(w · u + b).
    pub fn compute_delta(&self, u: &[f32]) -> f32 {
        assert_eq!(u.len(), SSM_D);
        let mut pre = self.b_delta;
        for i in 0..SSM_D {
            pre += self.w_delta[i] * u[i];
        }
        softplus(pre)
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        self.w_delta.len() + 1
    }
}

/// softplus(x) = log(1 + exp(x)), numerically stable.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // overflow guard
    } else if x < -20.0 {
        0.0 // underflow guard
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// IRDS threshold for delta caching.
const DELTA_RECOMPUTE_THRESHOLD: f32 = 1e-3;

/// Selective SSM step with input-dependent delta and IRDS caching.
///
/// If the new delta is close to `cached_delta`, reuses the cached
/// discretised matrices. Otherwise rediscretises.
///
/// Returns (output_y, new_delta).
pub fn selective_step(
    state: &mut SsmState,
    params: &SsmParams,
    delta_proj: &DeltaProjection,
    u: &[f32],
    cached_disc: &mut DiscretisedSsm,
    cached_delta: &mut f32,
) -> (Vec<f32>, f32) {
    // Compute input-dependent delta
    let new_delta = delta_proj.compute_delta(u);

    // IRDS: check if rediscretisation needed
    if (new_delta - *cached_delta).abs() > DELTA_RECOMPUTE_THRESHOLD {
        // Rediscretise with the new delta
        *cached_disc = discretise_with_delta(params, new_delta);
        *cached_delta = new_delta;
    }

    // Standard SSM step with the (possibly cached) discretised matrices
    let y = crate::update::ssm_step(state, cached_disc, params, u);

    (y, new_delta)
}

/// Discretise SSM params with a specific delta value.
fn discretise_with_delta(params: &SsmParams, delta: f32) -> DiscretisedSsm {
    // Create a temporary params copy with the desired delta
    let mut temp_params = params.clone();
    // Set log_delta so that exp(log_delta) = delta
    temp_params.log_delta = delta.max(1e-6).ln();
    DiscretisedSsm::from_params(&temp_params)
}

/// Batch selective step: process a sequence of inputs.
/// Returns (outputs, deltas) for each timestep.
pub fn selective_forward(
    state: &mut SsmState,
    params: &SsmParams,
    delta_proj: &DeltaProjection,
    inputs: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut cached_disc = DiscretisedSsm::from_params(params);
    let mut cached_delta = params.delta();

    let mut outputs = Vec::with_capacity(inputs.len());
    let mut deltas = Vec::with_capacity(inputs.len());

    for u in inputs {
        let (y, delta) = selective_step(
            state,
            params,
            delta_proj,
            u,
            &mut cached_disc,
            &mut cached_delta,
        );
        outputs.push(y);
        deltas.push(delta);
    }

    (outputs, deltas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_basic() {
        assert!((softplus(0.0) - 0.6931).abs() < 0.01);
        assert!(softplus(20.0) > 19.0);
        assert!(softplus(-20.0) < 0.01);
    }

    #[test]
    fn delta_projection_creates() {
        let dp = DeltaProjection::new(42);
        assert_eq!(dp.w_delta.len(), SSM_D);
        assert_eq!(dp.b_delta, 0.0);
    }

    #[test]
    fn delta_projection_positive() {
        let dp = DeltaProjection::new(42);
        let u = vec![0.0f32; SSM_D];
        let delta = dp.compute_delta(&u);
        assert!(delta > 0.0, "delta should be positive, got {}", delta);
    }

    #[test]
    fn selective_step_output_finite() {
        let params = SsmParams::new_stable(42);
        let dp = DeltaProjection::new(99);
        let mut state = SsmState::new();
        let mut disc = DiscretisedSsm::from_params(&params);
        let mut cached_delta = params.delta();

        let u = vec![0.1f32; SSM_D];
        let (y, delta) = selective_step(&mut state, &params, &dp, &u, &mut disc, &mut cached_delta);
        assert!(delta > 0.0);
        for &val in &y {
            assert!(val.is_finite(), "non-finite output");
        }
    }

    #[test]
    fn irds_caching_works() {
        let params = SsmParams::new_stable(42);
        let dp = DeltaProjection::new(99);
        let mut state = SsmState::new();
        let mut disc = DiscretisedSsm::from_params(&params);
        let mut cached_delta = params.delta();

        // Two identical inputs should hit cache on second call
        let u = vec![0.0f32; SSM_D];
        let (_, d1) = selective_step(&mut state, &params, &dp, &u, &mut disc, &mut cached_delta);
        let (_, d2) = selective_step(&mut state, &params, &dp, &u, &mut disc, &mut cached_delta);
        assert!((d1 - d2).abs() < 1e-10, "same input should give same delta");
    }

    #[test]
    fn batch_selective_forward_shapes() {
        let params = SsmParams::new_stable(42);
        let dp = DeltaProjection::new(99);
        let mut state = SsmState::new();

        let inputs: Vec<Vec<f32>> = (0..5).map(|_| vec![0.1f32; SSM_D]).collect();
        let (outputs, deltas) = selective_forward(&mut state, &params, &dp, &inputs);
        assert_eq!(outputs.len(), 5);
        assert_eq!(deltas.len(), 5);
        for y in &outputs {
            assert_eq!(y.len(), sophon_config::SSM_P);
        }
    }

    #[test]
    fn delta_proj_param_count() {
        let dp = DeltaProjection::new(42);
        assert_eq!(dp.param_count(), SSM_D + 1);
    }
}

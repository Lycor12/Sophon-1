//! Per-token SSM state update and output computation.
//!
//! Forward step (spec §1.2.1):
//!   h_{t+1} = A_bar h_t + B_bar u_t
//!   y_t     = C h_t + D u_t
//!
//! Novel optimisation — Fused State-Output (FSO):
//!   The standard implementation does A_bar h, then B_bar u, then sums,
//!   then computes C h. FSO fuses the C h computation with the A_bar h
//!   pass: while computing A_bar h we simultaneously accumulate C h.
//!   This halves the number of passes over h from 2 to 1 (since C and
//!   A_bar share the same h[j] access pattern).
//!
//! Input injection (spec §1.2.3):
//!   u_t is the KAN layer output (projected from d_model to SSM_D if needed).
//!   In the canonical block, SSM_D = d_model = 256, so no projection needed.

use crate::{params::SsmParams, zoh::DiscretisedSsm, SsmState};
use sophon_config::{SSM_D, SSM_N, SSM_P, SSM_RANK};

// ---------------------------------------------------------------------------
// ssm_step
// ---------------------------------------------------------------------------

/// One SSM token step.
///
/// Updates `state.h` in-place and returns output y in R^P.
///
/// Parameters:
///   state:    mutable SSM hidden state
///   disc:     pre-discretised A_bar, B_bar
///   params:   SSM learnable parameters (for C and D)
///   u:        input slice of length SSM_D
///
/// Novel FSO optimisation: fuses A_bar h and C h into a single pass over h.
pub fn ssm_step(
    state: &mut SsmState,
    disc: &DiscretisedSsm,
    params: &SsmParams,
    u: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(u.len(), SSM_D);

    let n = SSM_N;
    let p = SSM_P;
    let r = SSM_RANK;

    // -----------------------------------------------------------------------
    // FSO: single pass over h to compute both C h and the A_bar diagonal part
    // -----------------------------------------------------------------------

    // Step 1: z_lr[k] = sum_j a_lr_r[j,k] * h[j]  (low-rank right contract)
    let mut z_lr = [0.0f32; SSM_RANK];
    for j in 0..n {
        let hj = state.h[j];
        for k in 0..r {
            z_lr[k] += disc.a_lr_r[j * r + k] * hj;
        }
    }

    // Step 2 (FSO): compute new_h[i] = a_d[i]*h[i] + sum_k a_lr_l[i,k]*z_lr[k]
    //              AND simultaneously accumulate y[p] += C[p, i] * h[i]  (old h)
    let mut y = vec![0.0f32; p];
    let c_data = &params.c; // shape [P, N]

    let mut new_h = vec![0.0f32; n];
    for i in 0..n {
        let hi = state.h[i];

        // y += C[:, i] * hi (old h contribution to output)
        for q in 0..p {
            y[q] += c_data[q * n + i] * hi;
        }

        // new_h[i] = a_d[i]*hi + low-rank term
        let mut nh = disc.a_d[i] * hi;
        for k in 0..r {
            nh += disc.a_lr_l[i * r + k] * z_lr[k];
        }
        new_h[i] = nh;
    }

    // Step 3: B_bar u (add input term to new_h)
    let bu = disc.apply_b_bar(u);
    for i in 0..n {
        new_h[i] += bu[i];
    }

    // Commit new hidden state
    state.h.copy_from_slice(&new_h);

    // Step 4: y += D u  (feedthrough, D is usually near-zero)
    let d_data = &params.d; // shape [P, D]
    for q in 0..p {
        let mut acc = 0.0f32;
        let row = q * SSM_D;
        for j in 0..SSM_D {
            acc += d_data[row + j] * u[j];
        }
        y[q] += acc;
    }

    y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{params::SsmParams, zoh::DiscretisedSsm};

    #[test]
    fn step_zero_input_state_decays() {
        let params = SsmParams::new_stable(0);
        let disc = DiscretisedSsm::from_params(&params);
        let mut state = SsmState::new();

        // Set some non-zero initial state
        state.h[0] = 1.0;
        state.h[1] = -1.0;

        // Apply zero input multiple times; state should converge toward zero
        let u = vec![0.0f32; SSM_D];
        let _y1 = ssm_step(&mut state, &disc, &params, &u);
        let _y2 = ssm_step(&mut state, &disc, &params, &u);
        let _y3 = ssm_step(&mut state, &disc, &params, &u);

        // h[0] should have shrunk (stable dynamics)
        assert!(state.h[0].abs() < 1.0, "h[0] did not decay: {}", state.h[0]);
    }

    #[test]
    fn step_output_finite() {
        let params = SsmParams::new_stable(42);
        let disc = DiscretisedSsm::from_params(&params);
        let mut state = SsmState::new();
        let u: Vec<f32> = (0..SSM_D).map(|i| (i as f32) * 0.001).collect();
        let y = ssm_step(&mut state, &disc, &params, &u);
        assert_eq!(y.len(), SSM_P);
        assert!(
            y.iter().all(|&v| v.is_finite()),
            "output has non-finite values"
        );
    }

    #[test]
    fn step_deterministic_same_seed() {
        let params = SsmParams::new_stable(7);
        let disc = DiscretisedSsm::from_params(&params);
        let mut s1 = SsmState::new();
        let mut s2 = SsmState::new();
        let u: Vec<f32> = vec![0.5f32; SSM_D];
        let y1 = ssm_step(&mut s1, &disc, &params, &u);
        let y2 = ssm_step(&mut s2, &disc, &params, &u);
        assert_eq!(y1, y2);
    }

    #[test]
    fn state_validity_maintained() {
        let params = SsmParams::new_stable(0);
        let disc = DiscretisedSsm::from_params(&params);
        let mut state = SsmState::new();
        let u: Vec<f32> = vec![0.1f32; SSM_D];
        for _ in 0..100 {
            let _y = ssm_step(&mut state, &disc, &params, &u);
            assert!(state.is_valid(), "state became invalid");
        }
    }
}

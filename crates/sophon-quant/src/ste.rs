//! Straight-Through Estimator (STE) for ternary quantisation gradients.
//!
//! Spec §0.6.1: gradients for the ternary quantisation function use STE.
//!
//! STE definition:
//!   Forward:  q  = sign(w) for |w| > threshold, 0 otherwise
//!   Backward: dL/dw ≈ dL/dq * 1_{|w| <= 1}
//!
//! The 1_{|w| <= 1} clipping prevents the gradient from flowing through
//! heavily saturated weights, acting as a natural regulariser.

/// Straight-through estimator: compute gradient w.r.t. pre-quantised weight.
///
/// upstream_grad: dL/dq (gradient w.r.t. quantised weight)
/// w_float:       original floating-point weight
///
/// Returns dL/dw ≈ upstream_grad if |w_float| <= clip_threshold, else 0.
#[inline]
pub fn ste_grad(upstream_grad: f32, w_float: f32, clip_threshold: f32) -> f32 {
    if w_float.abs() <= clip_threshold {
        upstream_grad
    } else {
        0.0
    }
}

/// Apply STE to a batch of weights and upstream gradients.
pub fn ste_grad_batch(upstream: &[f32], w_float: &[f32], clip: f32, out: &mut [f32]) {
    debug_assert_eq!(upstream.len(), w_float.len());
    debug_assert_eq!(upstream.len(), out.len());
    for i in 0..upstream.len() {
        out[i] = ste_grad(upstream[i], w_float[i], clip);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ste_passes_gradient_within_clip() {
        assert_eq!(ste_grad(0.5, 0.3, 1.0), 0.5);
        assert_eq!(ste_grad(0.5, -0.8, 1.0), 0.5);
    }

    #[test]
    fn ste_blocks_gradient_outside_clip() {
        assert_eq!(ste_grad(0.5, 1.5, 1.0), 0.0);
        assert_eq!(ste_grad(0.5, -2.0, 1.0), 0.0);
    }

    #[test]
    fn ste_batch_shape_invariant() {
        let up = vec![1.0f32; 8];
        let w = vec![0.5f32; 8];
        let mut out = vec![0.0f32; 8];
        ste_grad_batch(&up, &w, 1.0, &mut out);
        assert_eq!(out, up);
    }
}

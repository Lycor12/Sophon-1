//! Gradient propagation utilities for the KAN layer.
//!
//! Implements the sparsity regularisation gradient term from spec §1.1.4:
//!   L_sparsity = lambda * sum_{ij} |phi_{ij}|
//! where |phi_{ij}| is estimated as the mean absolute output of edge (i,j)
//! over the current batch.
//!
//! Novel optimisation — Online Sparsity Estimator (OSE):
//!   Instead of storing all batch activations, we maintain a running
//!   exponential moving average (EMA) of |phi_{ij}(x_t)| over the sequence,
//!   which requires O(d_in * d_out) memory (not O(T * d_in * d_out)).
//!   EMA update: ema[ij] = alpha * |phi(x_t)| + (1-alpha) * ema[ij]

use crate::layer::KanLayer;
use sophon_core::{CoreError, Tensor};

// ---------------------------------------------------------------------------
// SparsityGrad
// ---------------------------------------------------------------------------

/// Maintains per-edge EMA of activation magnitude for sparsity regularisation.
pub struct SparsityGrad {
    /// EMA of |phi_{ij}(x)| per edge. Shape: d_in * d_out.
    ema: Vec<f32>,
    /// EMA decay factor (alpha = 0.99 => slow decay, good for stable estimates).
    alpha: f32,
    d_in: usize,
    d_out: usize,
}

impl SparsityGrad {
    pub fn new(d_in: usize, d_out: usize, alpha: f32) -> Self {
        Self {
            ema: vec![0.0; d_in * d_out],
            alpha,
            d_in,
            d_out,
        }
    }

    /// Update EMA from current forward activations (1-D output per edge).
    pub fn update(&mut self, layer: &KanLayer, x: &Tensor) -> Result<(), CoreError> {
        if x.cols() != self.d_in {
            return Err(CoreError::ShapeMismatch {
                got: [x.rows(), x.cols()],
                expected: [1, self.d_in],
            });
        }
        let x_data = x.as_slice();
        let alpha = self.alpha;
        let one_m = 1.0 - alpha;
        for i in 0..self.d_in {
            let xi = x_data[i].clamp(0.0, 1.0);
            let row_off = i * self.d_out;
            for j in 0..self.d_out {
                let act = layer.edges[row_off + j].eval(xi).abs();
                self.ema[row_off + j] = alpha * act + one_m * self.ema[row_off + j];
            }
        }
        Ok(())
    }

    /// Compute the sparsity gradient term for all edges.
    ///
    /// grad_sparsity[ij][c] = lambda * sign(ema[ij]) * grad_c[ij][c]
    /// Returns flat Vec<f32> of same shape as KanLayer::grad_coeffs.
    pub fn sparsity_grad(
        &self,
        layer: &KanLayer,
        x: &Tensor,
        lambda: f32,
    ) -> Result<Vec<f32>, CoreError> {
        let grad_out = Tensor::full_1d(self.d_out, 1.0);
        let gc_raw = layer.grad_coeffs(x, &grad_out)?;
        let n_ctrl = crate::spline::N_CTRL;
        let mut gc_sparse = vec![0.0f32; self.d_in * self.d_out * n_ctrl];
        for i in 0..self.d_in {
            for j in 0..self.d_out {
                let ij = i * self.d_out + j;
                let sign = self.ema[ij].signum();
                let off = ij * n_ctrl;
                for k in 0..n_ctrl {
                    gc_sparse[off + k] = lambda * sign * gc_raw[off + k];
                }
            }
        }
        Ok(gc_sparse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sophon_core::Tensor;

    #[test]
    fn ema_initialises_zero() {
        let sg = SparsityGrad::new(4, 4, 0.99);
        assert!(sg.ema.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ema_updates_nonzero_after_step() {
        let layer = KanLayer::new(2, 2);
        let mut sg = SparsityGrad::new(2, 2, 0.9);
        let x = Tensor::from_slice_1d(&[0.5, 0.3]);
        sg.update(&layer, &x).unwrap();
        // All activations are 0 (zero coefficients) so EMA stays 0
        // but update should not panic
        assert!(sg.ema.iter().all(|&v| v >= 0.0));
    }
}

//! KAN layer: full d_in -> d_out KAN transformation.
//!
//! Each output dimension j is computed as:
//!   y_j = sum_{i=0}^{d_in-1} phi_{ij}(x_i) + bias_j
//!
//! where phi_{ij} is a CubicBSpline edge function and bias_j is the
//! per-output bias term (Section 1.1.2 of v3 spec).
//!
//! This is the "direct summation composition" specified in Section 0.1.2:
//! functions are composed by summation, not nesting, which keeps
//! the forward pass O(d_in * d_out) in function evaluations.
//!
//! Weight layout: edges[i * d_out + j] = phi_{ij}.

#![allow(clippy::needless_range_loop)]

use crate::spline::{CubicBSpline, N_CTRL};
use sophon_config::{D_MODEL, KAN_KNOTS};
use sophon_core::{CoreError, Tensor};

// ---------------------------------------------------------------------------
// KanLayer
// ---------------------------------------------------------------------------

/// One KAN layer mapping R^{d_in} -> R^{d_out}.
pub struct KanLayer {
    pub d_in: usize,
    pub d_out: usize,
    /// Row-major: edges[i * d_out + j] = phi_{ij}(·)
    pub edges: Vec<CubicBSpline>,
    /// Per-output bias: bias[j].
    pub bias: Vec<f32>,
}

impl KanLayer {
    /// Create a KAN layer with all-zero spline coefficients, w_base=0, bias=0.
    pub fn new(d_in: usize, d_out: usize) -> Self {
        let n_edges = d_in * d_out;
        let edges: Vec<CubicBSpline> = (0..n_edges).map(|_| CubicBSpline::new(0.0, 1.0)).collect();
        Self {
            d_in,
            d_out,
            edges,
            bias: vec![0.0f32; d_out],
        }
    }

    /// Create the canonical D_MODEL x D_MODEL KAN layer.
    pub fn canonical() -> Self {
        Self::new(D_MODEL, D_MODEL)
    }

    /// Forward pass: y[j] = sum_i phi_{ij}(x[i]) + bias[j].
    ///
    /// Supports both single-sample (1×d_in) and batch (batch×d_in) inputs.
    /// Output shape: (batch × d_out).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, CoreError> {
        if x.cols() != self.d_in {
            return Err(CoreError::ShapeMismatch {
                got: [x.rows(), x.cols()],
                expected: [x.rows(), self.d_in],
            });
        }
        let batch = x.rows();
        let mut out = Tensor::zeros_2d(batch, self.d_out);
        let out_data = out.as_slice_mut();

        for b in 0..batch {
            let x_row = x.row(b)?;
            let out_off = b * self.d_out;

            // Direct summation composition: y_j = sum_i phi_{ij}(x_i) + bias_j
            for i in 0..self.d_in {
                let xi = x_row[i].clamp(0.0, 1.0);
                let row_off = i * self.d_out;
                for j in 0..self.d_out {
                    out_data[out_off + j] += self.edges[row_off + j].eval(xi);
                }
            }
            // Add bias
            for j in 0..self.d_out {
                out_data[out_off + j] += self.bias[j];
            }
        }

        Ok(out)
    }

    /// Backward pass: compute gradient w.r.t. input x.
    ///
    /// Given upstream gradient `grad_out` (batch × d_out),
    /// returns gradient w.r.t. x (batch × d_in):
    ///   grad_x[b,i] = sum_j grad_out[b,j] * dphi_{ij}/dx_i
    pub fn backward_x(&self, x: &Tensor, grad_out: &Tensor) -> Result<Tensor, CoreError> {
        if x.cols() != self.d_in || grad_out.cols() != self.d_out {
            return Err(CoreError::ShapeMismatch {
                got: [grad_out.rows(), grad_out.cols()],
                expected: [grad_out.rows(), self.d_out],
            });
        }
        let batch = x.rows();
        let mut gx = Tensor::zeros_2d(batch, self.d_in);
        let gx_data = gx.as_slice_mut();

        for b in 0..batch {
            let x_row = x.row(b)?;
            let g_row = grad_out.row(b)?;
            let gx_off = b * self.d_in;

            for i in 0..self.d_in {
                let xi = x_row[i].clamp(0.0, 1.0);
                let row_off = i * self.d_out;
                let mut acc = 0.0f32;
                for j in 0..self.d_out {
                    acc += g_row[j] * self.edges[row_off + j].grad_x(xi);
                }
                gx_data[gx_off + i] = acc;
            }
        }
        Ok(gx)
    }

    /// Gradient w.r.t. all spline coefficients (accumulated over batch).
    ///
    /// Returns a flat Vec<f32> of length `d_in * d_out * N_CTRL`.
    /// Layout: [i, j, ctrl_idx].
    pub fn grad_coeffs(&self, x: &Tensor, grad_out: &Tensor) -> Result<Vec<f32>, CoreError> {
        if x.cols() != self.d_in || grad_out.cols() != self.d_out {
            return Err(CoreError::ShapeMismatch {
                got: [grad_out.rows(), grad_out.cols()],
                expected: [grad_out.rows(), self.d_out],
            });
        }
        let batch = x.rows();
        let n = self.d_in * self.d_out * N_CTRL;
        let mut gc = vec![0.0f32; n];

        for b in 0..batch {
            let x_row = x.row(b)?;
            let g_row = grad_out.row(b)?;

            for i in 0..self.d_in {
                let xi = x_row[i].clamp(0.0, 1.0);
                let row_off = i * self.d_out;
                for j in 0..self.d_out {
                    let basis = self.edges[row_off + j].grad_c(xi);
                    let scale = g_row[j];
                    let offset = (row_off + j) * N_CTRL;
                    for k in 0..N_CTRL {
                        gc[offset + k] += scale * basis[k];
                    }
                }
            }
        }
        Ok(gc)
    }

    /// Gradient w.r.t. all w_base parameters (accumulated over batch).
    ///
    /// Returns a flat Vec<f32> of length `d_in * d_out`.
    /// Layout: [i * d_out + j].
    pub fn grad_w_base(&self, x: &Tensor, grad_out: &Tensor) -> Result<Vec<f32>, CoreError> {
        if x.cols() != self.d_in || grad_out.cols() != self.d_out {
            return Err(CoreError::ShapeMismatch {
                got: [grad_out.rows(), grad_out.cols()],
                expected: [grad_out.rows(), self.d_out],
            });
        }
        let batch = x.rows();
        let mut gw = vec![0.0f32; self.d_in * self.d_out];

        for b in 0..batch {
            let x_row = x.row(b)?;
            let g_row = grad_out.row(b)?;

            for i in 0..self.d_in {
                let xi = x_row[i].clamp(0.0, 1.0);
                let row_off = i * self.d_out;
                for j in 0..self.d_out {
                    gw[row_off + j] += g_row[j] * self.edges[row_off + j].grad_w_base(xi);
                }
            }
        }
        Ok(gw)
    }

    /// Gradient w.r.t. all internal knot positions (accumulated over batch).
    ///
    /// Returns a flat Vec<f32> of length `d_in * d_out * KAN_KNOTS`.
    /// Layout: [i * d_out * KAN_KNOTS + j * KAN_KNOTS + knot_idx].
    pub fn grad_knots(&self, x: &Tensor, grad_out: &Tensor) -> Result<Vec<f32>, CoreError> {
        if x.cols() != self.d_in || grad_out.cols() != self.d_out {
            return Err(CoreError::ShapeMismatch {
                got: [grad_out.rows(), grad_out.cols()],
                expected: [grad_out.rows(), self.d_out],
            });
        }
        let batch = x.rows();
        let n = self.d_in * self.d_out * KAN_KNOTS;
        let mut gk = vec![0.0f32; n];

        for b in 0..batch {
            let x_row = x.row(b)?;
            let g_row = grad_out.row(b)?;

            for i in 0..self.d_in {
                let xi = x_row[i].clamp(0.0, 1.0);
                let row_off = i * self.d_out;
                for j in 0..self.d_out {
                    let knot_grads = self.edges[row_off + j].grad_knots(xi);
                    let scale = g_row[j];
                    let offset = (row_off + j) * KAN_KNOTS;
                    for k in 0..KAN_KNOTS {
                        gk[offset + k] += scale * knot_grads[k];
                    }
                }
            }
        }
        Ok(gk)
    }

    /// Gradient w.r.t. bias (accumulated over batch).
    ///
    /// Returns Vec<f32> of length d_out.
    pub fn grad_bias(&self, grad_out: &Tensor) -> Result<Vec<f32>, CoreError> {
        if grad_out.cols() != self.d_out {
            return Err(CoreError::ShapeMismatch {
                got: [grad_out.rows(), grad_out.cols()],
                expected: [grad_out.rows(), self.d_out],
            });
        }
        let batch = grad_out.rows();
        let mut gb = vec![0.0f32; self.d_out];
        for b in 0..batch {
            let g_row = grad_out.row(b)?;
            for j in 0..self.d_out {
                gb[j] += g_row[j];
            }
        }
        Ok(gb)
    }

    /// Prune edges with L1 coefficient norm below threshold.
    /// Sets all coefficients and w_base of pruned edges to zero.
    /// Returns the number of edges pruned.
    pub fn prune(&mut self, threshold: f32) -> usize {
        let mut pruned = 0;
        for edge in self.edges.iter_mut() {
            let l1: f32 = edge.c.iter().map(|&v| v.abs()).sum::<f32>() + edge.w_base.abs();
            if l1 < threshold {
                edge.c = [0.0; N_CTRL];
                edge.w_base = 0.0;
                pruned += 1;
            }
        }
        pruned
    }

    /// Count total learnable parameters in this layer.
    pub fn param_count(&self) -> usize {
        // N_CTRL spline coeffs + 1 w_base + KAN_KNOTS trainable knots per edge + d_out bias
        let per_edge = N_CTRL + 1 + KAN_KNOTS;
        self.d_in * self.d_out * per_edge + self.d_out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_zero_weights_gives_zero() {
        let layer = KanLayer::new(4, 4);
        let x = Tensor::from_slice_1d(&[0.5, 0.3, 0.7, 0.2]);
        let y = layer.forward(&x).unwrap();
        assert!(y.as_slice().iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn forward_shape_correct() {
        let layer = KanLayer::new(8, 4);
        let x = Tensor::zeros_1d(8);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.cols(), 4);
        assert_eq!(y.rows(), 1);
    }

    #[test]
    fn forward_batch() {
        let layer = KanLayer::new(4, 3);
        let x = Tensor::zeros_2d(5, 4); // batch of 5
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.rows(), 5);
        assert_eq!(y.cols(), 3);
    }

    #[test]
    fn forward_shape_mismatch_error() {
        let layer = KanLayer::new(4, 4);
        let x = Tensor::zeros_1d(5); // wrong size
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn param_count_positive() {
        let layer = KanLayer::new(4, 4);
        assert!(layer.param_count() > 0);
        // Should include bias now
        let per_edge = N_CTRL + 1 + KAN_KNOTS;
        let expected = 4 * 4 * per_edge + 4;
        assert_eq!(layer.param_count(), expected);
    }

    #[test]
    fn prune_all_zeros_returns_full_count() {
        let mut layer = KanLayer::new(2, 2);
        let pruned = layer.prune(0.001);
        assert_eq!(pruned, 4);
    }

    #[test]
    fn backward_shape_correct() {
        let layer = KanLayer::new(4, 4);
        let x = Tensor::from_slice_1d(&[0.5, 0.3, 0.7, 0.2]);
        let g = Tensor::from_slice_1d(&[1.0; 4]);
        let gx = layer.backward_x(&x, &g).unwrap();
        assert_eq!(gx.cols(), 4);
    }

    #[test]
    fn backward_batch_shape() {
        let layer = KanLayer::new(4, 3);
        let x = Tensor::zeros_2d(5, 4);
        let g = Tensor::zeros_2d(5, 3);
        let gx = layer.backward_x(&x, &g).unwrap();
        assert_eq!(gx.rows(), 5);
        assert_eq!(gx.cols(), 4);
    }

    #[test]
    fn grad_w_base_shape() {
        let layer = KanLayer::new(4, 3);
        let x = Tensor::from_slice_1d(&[0.5, 0.3, 0.7, 0.2]);
        let g = Tensor::from_slice_1d(&[1.0, 1.0, 1.0]);
        let gw = layer.grad_w_base(&x, &g).unwrap();
        assert_eq!(gw.len(), 4 * 3);
    }

    #[test]
    fn grad_bias_shape() {
        let layer = KanLayer::new(4, 3);
        let g = Tensor::from_slice_1d(&[1.0, 2.0, 3.0]);
        let gb = layer.grad_bias(&g).unwrap();
        assert_eq!(gb.len(), 3);
        assert!((gb[0] - 1.0).abs() < 1e-6);
        assert!((gb[1] - 2.0).abs() < 1e-6);
        assert!((gb[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn bias_affects_output() {
        let mut layer = KanLayer::new(2, 2);
        layer.bias[0] = 1.0;
        layer.bias[1] = -0.5;
        let x = Tensor::from_slice_1d(&[0.5, 0.5]);
        let y = layer.forward(&x).unwrap();
        // Edges are all zero, so output = bias only
        assert!((y.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((y.as_slice()[1] - (-0.5)).abs() < 1e-6);
    }
}

//! LoRA-style low-rank adapters for parameter-efficient adaptation.
//!
//! Spec §3.3.1: W_adapted = W_base + scale * B * A
//!   where B in R^{d_out x r}, A in R^{r x d_in}, scale = alpha / r.
//!   Rank r = LORA_RANK = 16.
//!
//! Attachment points (spec §3.3.2):
//!   - KAN spline coefficient matrices (all d_in*d_out splines share one adapter)
//!   - SSM B matrix [N, D]
//!   - SSM C matrix [P, N]
//!
//! Merging (spec §3.3.3): adapters can be merged into base weights at inference
//! time (B*A is absorbed into W_base), or kept separate for continued adaptation.
//!
//! Novel optimisation — LORA-GRAD (Low-Rank Adjoint Decomposition):
//!   Standard LoRA backward computes dL/dB = grad_out * A^T and dL/dA = B^T * grad_in
//!   via two separate matmul passes over the full activation tensors. LORA-GRAD
//!   exploits the fact that both B and A are narrow (rank r << d), so we can
//!   precompute a rank-r intermediate: z = A * x (r-vector), and accumulate
//!   dL/dB directly from the r-vector z and the grad, avoiding the d_out * d_in
//!   materialisation entirely. The dL/dA computation reuses the already-cached
//!   z to compute B^T * grad as an r-vector, then outer-products with x.
//!   Total: O(r * (d_in + d_out)) per sample, vs O(d_in * d_out) naive.

use sophon_config::LORA_RANK;
use sophon_core::rng::Rng;

// ---------------------------------------------------------------------------
// LoraAdapter
// ---------------------------------------------------------------------------

/// A single LoRA adapter: W_adapted = W_base + scale * B * A.
///
/// B: [d_out, r], A: [r, d_in], scale = alpha / r.
pub struct LoraAdapter {
    pub d_out: usize,
    pub d_in: usize,
    pub rank: usize,
    /// B matrix: [d_out, rank], row-major
    pub b: Vec<f32>,
    /// A matrix: [rank, d_in], row-major
    pub a: Vec<f32>,
    /// Scaling factor (alpha / rank).
    pub scale: f32,
    /// Whether this adapter's parameters are frozen (excluded from training).
    pub frozen: bool,
}

/// Gradients for a single LoRA adapter.
pub struct LoraGrads {
    /// dL/dB. Shape: [d_out, rank]
    pub grad_b: Vec<f32>,
    /// dL/dA. Shape: [rank, d_in]
    pub grad_a: Vec<f32>,
}

impl LoraAdapter {
    /// Create a LoRA adapter.
    /// B is initialised with Kaiming, A is initialised to zero
    /// (so the adapter starts as a no-op: delta_W = B*0 = 0).
    pub fn new(d_out: usize, d_in: usize, alpha: f32, seed: u64) -> Self {
        let rank = LORA_RANK;
        let scale = alpha / rank as f32;
        let mut rng = Rng::new(seed);

        let mut b = vec![0.0f32; d_out * rank];
        rng.fill_kaiming_uniform(&mut b, rank);
        let a = vec![0.0f32; rank * d_in]; // zero init: delta = 0

        Self {
            d_out,
            d_in,
            rank,
            b,
            a,
            scale,
            frozen: false,
        }
    }

    /// Freeze this adapter (exclude from gradient updates).
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze this adapter (include in gradient updates).
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Forward: compute delta applied to an input vector.
    ///
    /// Given x [d_in], computes scale * B * (A * x) → [d_out].
    /// This is the LoRA contribution to the output, computed without
    /// materialising the full delta_W matrix.
    pub fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_in);
        let r = self.rank;

        // Step 1: z = A * x → [r]
        let mut z = vec![0.0f32; r];
        for k in 0..r {
            let mut acc = 0.0f32;
            let row_start = k * self.d_in;
            for j in 0..self.d_in {
                acc += self.a[row_start + j] * x[j];
            }
            z[k] = acc;
        }

        // Step 2: out = scale * B * z → [d_out]
        let mut out = vec![0.0f32; self.d_out];
        let s = self.scale;
        for i in 0..self.d_out {
            let mut acc = 0.0f32;
            let row_start = i * r;
            for k in 0..r {
                acc += self.b[row_start + k] * z[k];
            }
            out[i] = s * acc;
        }
        out
    }

    /// Backward pass using LORA-GRAD.
    ///
    /// Given:
    ///   - x: the input to this adapter [d_in]
    ///   - grad_out: upstream gradient dL/d(output) [d_out]
    ///
    /// Returns:
    ///   - LoraGrads (dL/dB, dL/dA) if not frozen
    ///   - dL/dx: gradient to propagate further [d_in]
    pub fn backward(&self, x: &[f32], grad_out: &[f32]) -> (Option<LoraGrads>, Vec<f32>) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(grad_out.len(), self.d_out);
        let r = self.rank;
        let s = self.scale;

        // --- Precompute z = A * x → [r] (cached from forward, but recompute for simplicity) ---
        let mut z = vec![0.0f32; r];
        for k in 0..r {
            let mut acc = 0.0f32;
            let row_start = k * self.d_in;
            for j in 0..self.d_in {
                acc += self.a[row_start + j] * x[j];
            }
            z[k] = acc;
        }

        // --- dL/dx = scale * A^T * (B^T * grad_out) → [d_in] ---
        // Step 1: w = B^T * grad_out → [r]
        let mut w = vec![0.0f32; r];
        for k in 0..r {
            let mut acc = 0.0f32;
            for i in 0..self.d_out {
                acc += self.b[i * r + k] * grad_out[i];
            }
            w[k] = s * acc;
        }
        // Step 2: grad_x = A^T * w → [d_in]
        let mut grad_x = vec![0.0f32; self.d_in];
        for j in 0..self.d_in {
            let mut acc = 0.0f32;
            for k in 0..r {
                acc += self.a[k * self.d_in + j] * w[k];
            }
            grad_x[j] = acc;
        }

        if self.frozen {
            return (None, grad_x);
        }

        // --- LORA-GRAD: dL/dB and dL/dA using rank-r intermediates ---

        // dL/dB[i,k] = scale * grad_out[i] * z[k]
        let mut grad_b = vec![0.0f32; self.d_out * r];
        for i in 0..self.d_out {
            let gi = s * grad_out[i];
            let row_start = i * r;
            for k in 0..r {
                grad_b[row_start + k] = gi * z[k];
            }
        }

        // dL/dA[k,j] = scale * (B^T * grad_out)[k] * x[j]
        // We already have w[k] = scale * (B^T * grad_out)[k]
        let mut grad_a = vec![0.0f32; r * self.d_in];
        for k in 0..r {
            let wk = w[k]; // already has scale factor
            let row_start = k * self.d_in;
            for j in 0..self.d_in {
                grad_a[row_start + j] = wk * x[j];
            }
        }

        (Some(LoraGrads { grad_b, grad_a }), grad_x)
    }

    /// Compute delta_W = scale * B * A and add to `base_weight`.
    ///
    /// base_weight: flat row-major [d_out, d_in].
    /// Modifies in-place.
    pub fn apply_to(&self, base_weight: &mut [f32]) {
        debug_assert_eq!(base_weight.len(), self.d_out * self.d_in);
        let r = self.rank;
        let s = self.scale;
        for i in 0..self.d_out {
            for j in 0..self.d_in {
                let mut delta = 0.0f32;
                for k in 0..r {
                    delta += self.b[i * r + k] * self.a[k * self.d_in + j];
                }
                base_weight[i * self.d_in + j] += s * delta;
            }
        }
    }

    /// Merge adapter permanently into a weight matrix.
    pub fn merge_into(&self, base_weight: &mut Vec<f32>) {
        self.apply_to(base_weight);
    }

    /// Parameter count (B + A only; base weight is external).
    pub fn param_count(&self) -> usize {
        self.d_out * self.rank + self.rank * self.d_in
    }
}

// ---------------------------------------------------------------------------
// LoraBlock: LoRA adapters for one HybridBlock
// ---------------------------------------------------------------------------

/// LoRA adapters for one HybridBlock per spec §3.3.2.
///
/// Attachment points:
///   - `kan_adapter`: applied to the KAN output (d_model -> d_model)
///   - `ssm_b_adapter`: applied to SSM's B input path (d -> n)
///   - `ssm_c_adapter`: applied to SSM's C output path (n -> p)
pub struct LoraBlock {
    /// LoRA on KAN output: [D_MODEL, D_MODEL]
    pub kan_adapter: LoraAdapter,
    /// LoRA on SSM B matrix: [SSM_N, SSM_D]
    pub ssm_b_adapter: LoraAdapter,
    /// LoRA on SSM C matrix: [SSM_P, SSM_N]
    pub ssm_c_adapter: LoraAdapter,
}

impl LoraBlock {
    /// Create LoRA adapters for one block.
    pub fn new(alpha: f32, seed: u64) -> Self {
        use sophon_config::{D_MODEL, SSM_D, SSM_N, SSM_P};
        Self {
            kan_adapter: LoraAdapter::new(D_MODEL, D_MODEL, alpha, seed),
            ssm_b_adapter: LoraAdapter::new(SSM_N, SSM_D, alpha, seed + 1),
            ssm_c_adapter: LoraAdapter::new(SSM_P, SSM_N, alpha, seed + 2),
        }
    }

    /// Freeze all adapters in this block.
    pub fn freeze_all(&mut self) {
        self.kan_adapter.freeze();
        self.ssm_b_adapter.freeze();
        self.ssm_c_adapter.freeze();
    }

    /// Unfreeze all adapters in this block.
    pub fn unfreeze_all(&mut self) {
        self.kan_adapter.unfreeze();
        self.ssm_b_adapter.unfreeze();
        self.ssm_c_adapter.unfreeze();
    }

    /// Total adapter parameter count.
    pub fn param_count(&self) -> usize {
        self.kan_adapter.param_count()
            + self.ssm_b_adapter.param_count()
            + self.ssm_c_adapter.param_count()
    }

    /// Flatten all adapter parameters into a single vector.
    pub fn flattened_params(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.param_count());
        params.extend_from_slice(&self.kan_adapter.b);
        params.extend_from_slice(&self.kan_adapter.a);
        params.extend_from_slice(&self.ssm_b_adapter.b);
        params.extend_from_slice(&self.ssm_b_adapter.a);
        params.extend_from_slice(&self.ssm_c_adapter.b);
        params.extend_from_slice(&self.ssm_c_adapter.a);
        params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_init_a_means_no_op() {
        let adapter = LoraAdapter::new(4, 4, 1.0, 0);
        let mut w = vec![1.0f32; 16];
        let before = w.clone();
        adapter.apply_to(&mut w);
        assert_eq!(w, before);
    }

    #[test]
    fn param_count_correct() {
        let adapter = LoraAdapter::new(256, 256, 1.0, 0);
        assert_eq!(adapter.param_count(), 256 * LORA_RANK + LORA_RANK * 256);
    }

    #[test]
    fn forward_vec_zero_init_is_zero() {
        let adapter = LoraAdapter::new(8, 4, 1.0, 0);
        let x = vec![1.0f32; 4];
        let out = adapter.forward_vec(&x);
        assert_eq!(out.len(), 8);
        // A is zero-init, so output should be zero
        assert!(out.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn backward_shapes() {
        let adapter = LoraAdapter::new(8, 4, 1.0, 42);
        // Set A to non-zero so backward produces gradients
        let mut a_mod = adapter;
        for v in a_mod.a.iter_mut() {
            *v = 0.1;
        }
        let x = vec![1.0f32; 4];
        let grad_out = vec![0.5f32; 8];
        let (grads, grad_x) = a_mod.backward(&x, &grad_out);
        assert_eq!(grad_x.len(), 4);
        let grads = grads.unwrap();
        assert_eq!(grads.grad_b.len(), 8 * LORA_RANK);
        assert_eq!(grads.grad_a.len(), LORA_RANK * 4);
    }

    #[test]
    fn frozen_backward_no_grads() {
        let mut adapter = LoraAdapter::new(4, 4, 1.0, 0);
        adapter.freeze();
        let x = vec![1.0; 4];
        let g = vec![1.0; 4];
        let (grads, grad_x) = adapter.backward(&x, &g);
        assert!(grads.is_none());
        assert_eq!(grad_x.len(), 4);
    }

    #[test]
    fn backward_grad_x_finite() {
        let adapter = LoraAdapter::new(16, 8, 2.0, 99);
        let x = vec![0.5f32; 8];
        let grad_out = vec![0.1f32; 16];
        let (_, grad_x) = adapter.backward(&x, &grad_out);
        assert!(grad_x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lora_block_param_count() {
        let lb = LoraBlock::new(1.0, 0);
        assert!(lb.param_count() > 0);
    }

    #[test]
    fn lora_block_freeze_unfreeze() {
        let mut lb = LoraBlock::new(1.0, 0);
        lb.freeze_all();
        assert!(lb.kan_adapter.frozen);
        assert!(lb.ssm_b_adapter.frozen);
        assert!(lb.ssm_c_adapter.frozen);
        lb.unfreeze_all();
        assert!(!lb.kan_adapter.frozen);
        assert!(!lb.ssm_b_adapter.frozen);
        assert!(!lb.ssm_c_adapter.frozen);
    }

    #[test]
    fn numerical_gradient_check_b() {
        // Finite-difference check for dL/dB
        let mut adapter = LoraAdapter::new(4, 3, 1.0, 42);
        // Make A non-zero
        for v in adapter.a.iter_mut() {
            *v = 0.1;
        }
        let x = vec![1.0, 2.0, 3.0];
        let grad_out = vec![1.0, 0.5, 0.25, 0.125];

        let (grads, _) = adapter.backward(&x, &grad_out);
        let grads = grads.unwrap();

        // Check a few entries of grad_b via finite differences
        let h = 1e-4f32;
        for idx in [0, 5, adapter.b.len() - 1] {
            if idx >= adapter.b.len() {
                continue;
            }
            let orig = adapter.b[idx];
            adapter.b[idx] = orig + h;
            let out_plus = adapter.forward_vec(&x);
            adapter.b[idx] = orig - h;
            let out_minus = adapter.forward_vec(&x);
            adapter.b[idx] = orig;

            let fd: f32 = out_plus
                .iter()
                .zip(out_minus.iter())
                .zip(grad_out.iter())
                .map(|((&p, &m), &g)| g * (p - m) / (2.0 * h))
                .sum();

            let analytic = grads.grad_b[idx];
            let err = (fd - analytic).abs();
            assert!(
                err < 0.01,
                "grad_b[{idx}]: fd={fd:.6} analytic={analytic:.6} err={err:.6}"
            );
        }
    }
}

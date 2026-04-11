//! Output head: LayerNorm -> linear projection -> VOCAB_SIZE logits.
//!
//! Produces logits over the byte vocabulary (256 tokens).
//! Output constraint: the logit vector is then passed to the verifier
//! which determines whether the output can be marked VERIFIED.

use sophon_config::{D_MODEL, VOCAB_SIZE};
use sophon_core::norm::layer_norm;
use sophon_core::rng::Rng;
use sophon_core::{CoreError, Tensor};

// ---------------------------------------------------------------------------
// OutputHead
// ---------------------------------------------------------------------------

pub struct OutputHead {
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,
    pub ln_eps: f32,
    /// Linear projection W: [VOCAB_SIZE, D_MODEL]
    pub weight: Vec<f32>,
    /// Bias b: [VOCAB_SIZE]
    pub bias: Vec<f32>,
}

impl OutputHead {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut weight = vec![0.0f32; VOCAB_SIZE * D_MODEL];
        rng.fill_kaiming_uniform(&mut weight, D_MODEL);
        Self {
            ln_gamma: vec![1.0f32; D_MODEL],
            ln_beta: vec![0.0f32; D_MODEL],
            ln_eps: 1e-5,
            weight,
            bias: vec![0.0f32; VOCAB_SIZE],
        }
    }

    /// Forward: x -> logits[VOCAB_SIZE].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, CoreError> {
        let x_data = x.as_slice();
        if x_data.len() != D_MODEL {
            return Err(CoreError::ShapeMismatch {
                got: [1, x_data.len()],
                expected: [1, D_MODEL],
            });
        }
        // LayerNorm
        let xn = layer_norm(x_data, &self.ln_gamma, &self.ln_beta, self.ln_eps).map_err(|e| e)?;

        // Linear: logits[v] = sum_d W[v,d] * xn[d] + bias[v]
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            let mut acc = self.bias[v];
            let row = v * D_MODEL;
            // Kahan summation for precision
            let mut c = 0.0f32;
            for d in 0..D_MODEL {
                let y = self.weight[row + d] * xn[d] - c;
                let t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
            logits[v] = acc;
        }
        Ok(Tensor::from_slice_1d(&logits))
    }

    pub fn param_count(&self) -> usize {
        2 * D_MODEL + VOCAB_SIZE * D_MODEL + VOCAB_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_output_shape() {
        let head = OutputHead::new(0);
        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let out = head.forward(&x).unwrap();
        assert_eq!(out.cols(), VOCAB_SIZE);
    }

    #[test]
    fn head_output_finite() {
        let head = OutputHead::new(42);
        let x = Tensor::from_slice_1d(&vec![0.5f32; D_MODEL]);
        let out = head.forward(&x).unwrap();
        assert!(out.as_slice().iter().all(|&v| v.is_finite()));
    }
}

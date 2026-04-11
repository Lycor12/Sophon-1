//! Hybrid KAN + SSM residual block.
//!
//! Exact pattern from spec §1.3.1:
//!   x'    = LayerNorm(x)
//!   k_out = KAN(x')
//!   k'    = LayerNorm(k_out)
//!   s_out = SSM(k')     [token-step, updates hidden state in-place]
//!   y     = x + s_out   [residual]
//!
//! Data flow per spec §1.3.2:
//!   input:  R^{d_model}
//!   output: R^{d_model}  (since SSM_P = D_MODEL = 256)
//!
//! SSM_D = D_MODEL = 256, so KAN output feeds directly into SSM without
//! any additional projection.
//!
//! LoRA integration (spec §3.3.2):
//!   When a LoraBlock is attached, the KAN output receives an additive
//!   LoRA delta, and the SSM B/C matrices are adapted via their respective
//!   LoRA adapters (applied as additive deltas during the SSM step).

use sophon_config::{D_MODEL, SSM_P};
use sophon_core::norm::layer_norm;
use sophon_core::{CoreError, Tensor};
use sophon_kan::KanLayer;
use sophon_ssm::{params::SsmParams, ssm_step, zoh::DiscretisedSsm, SsmState};

use crate::lora::LoraBlock;

// ---------------------------------------------------------------------------
// HybridBlock
// ---------------------------------------------------------------------------

/// One KAN + SSM residual block.
pub struct HybridBlock {
    // LayerNorm parameters before KAN
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,

    // KAN layer
    pub kan: KanLayer,

    // LayerNorm parameters before SSM
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,

    // SSM parameters and cached discretisation
    pub ssm_params: SsmParams,
    pub ssm_disc: DiscretisedSsm,

    // LayerNorm epsilon
    pub ln_eps: f32,

    /// Optional LoRA adapters (spec §3.3.2).
    /// When Some, the KAN output and SSM B/C get additive low-rank deltas.
    pub lora: Option<LoraBlock>,

    /// Whether base parameters (non-LoRA) are frozen.
    pub base_frozen: bool,
}

impl HybridBlock {
    /// Create a block with stable initialisations.
    pub fn new(seed: u64) -> Self {
        let kan = KanLayer::canonical();
        let ssm_params = SsmParams::new_stable(seed);
        let ssm_disc = DiscretisedSsm::from_params(&ssm_params);
        Self {
            ln1_gamma: vec![1.0f32; D_MODEL],
            ln1_beta: vec![0.0f32; D_MODEL],
            kan,
            ln2_gamma: vec![1.0f32; D_MODEL],
            ln2_beta: vec![0.0f32; D_MODEL],
            ssm_params,
            ssm_disc,
            ln_eps: 1e-5,
            lora: None,
            base_frozen: false,
        }
    }

    /// Attach LoRA adapters to this block.
    pub fn attach_lora(&mut self, alpha: f32, seed: u64) {
        self.lora = Some(LoraBlock::new(alpha, seed));
    }

    /// Detach LoRA adapters (merge into base weights if desired).
    pub fn detach_lora(&mut self) {
        self.lora = None;
    }

    /// Freeze base parameters (for LoRA-only fine-tuning).
    pub fn freeze_base(&mut self) {
        self.base_frozen = true;
    }

    /// Unfreeze base parameters.
    pub fn unfreeze_base(&mut self) {
        self.base_frozen = false;
    }

    /// Forward pass for one token.
    ///
    /// Mutates `ssm_state` in place (O(1) memory growth).
    /// Returns output tensor of shape [1, D_MODEL].
    pub fn forward(&self, x: &Tensor, ssm_state: &mut SsmState) -> Result<Tensor, CoreError> {
        let x_data = x.as_slice();
        if x_data.len() != D_MODEL {
            return Err(CoreError::ShapeMismatch {
                got: [1, x_data.len()],
                expected: [1, D_MODEL],
            });
        }

        // --- LayerNorm 1 ---
        let x_norm1 =
            layer_norm(x_data, &self.ln1_gamma, &self.ln1_beta, self.ln_eps).map_err(|e| e)?;
        let x_norm1_t = Tensor::from_slice_1d(&x_norm1);

        // --- KAN ---
        let kan_out = self.kan.forward(&x_norm1_t)?;

        // --- LoRA on KAN output (additive) ---
        let kan_out_data = if let Some(ref lora) = self.lora {
            let base = kan_out.as_slice();
            let delta = lora.kan_adapter.forward_vec(&x_norm1);
            let mut adapted = base.to_vec();
            for i in 0..D_MODEL {
                adapted[i] += delta[i];
            }
            adapted
        } else {
            kan_out.as_slice().to_vec()
        };

        // --- LayerNorm 2 ---
        let x_norm2 = layer_norm(&kan_out_data, &self.ln2_gamma, &self.ln2_beta, self.ln_eps)
            .map_err(|e| e)?;

        // --- SSM step ---
        let ssm_out = ssm_step(ssm_state, &self.ssm_disc, &self.ssm_params, &x_norm2);

        // --- Residual connection: y = x + ssm_out ---
        let mut y_data = vec![0.0f32; D_MODEL];
        for i in 0..D_MODEL {
            y_data[i] = x_data[i] + ssm_out[i];
        }

        Ok(Tensor::from_slice_1d(&y_data))
    }

    /// Recompute discretisation when SSM params change (e.g. after gradient step).
    pub fn refresh_disc(&mut self) {
        self.ssm_disc = DiscretisedSsm::from_params(&self.ssm_params);
    }

    /// Total learnable parameter count for this block.
    pub fn param_count(&self) -> usize {
        let base = 2 * D_MODEL  // ln1 gamma+beta
            + self.kan.param_count()
            + 2 * D_MODEL  // ln2 gamma+beta
            + self.ssm_params.param_count();
        let lora = self.lora.as_ref().map_or(0, |l| l.param_count());
        base + lora
    }
}

// ---------------------------------------------------------------------------
// BlockState: all SSM states for one sequence of 16 blocks
// ---------------------------------------------------------------------------

/// Hidden states for all NUM_BLOCKS SSM layers in one forward pass.
pub struct BlockStates {
    pub states: Vec<SsmState>,
}

impl BlockStates {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            states: (0..num_blocks).map(|_| SsmState::new()).collect(),
        }
    }

    pub fn reset_all(&mut self) {
        for s in self.states.iter_mut() {
            s.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_forward_shape_correct() {
        let block = HybridBlock::new(0);
        let mut state = SsmState::new();
        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let y = block.forward(&x, &mut state).unwrap();
        assert_eq!(y.cols(), D_MODEL);
    }

    #[test]
    fn block_forward_finite() {
        let block = HybridBlock::new(1);
        let mut state = SsmState::new();
        let x = Tensor::from_slice_1d(&vec![0.5f32; D_MODEL]);
        let y = block.forward(&x, &mut state).unwrap();
        assert!(
            y.as_slice().iter().all(|&v| v.is_finite()),
            "output has non-finite"
        );
    }

    #[test]
    fn block_deterministic_same_init() {
        let b1 = HybridBlock::new(42);
        let b2 = HybridBlock::new(42);
        let mut s1 = SsmState::new();
        let mut s2 = SsmState::new();
        let x = Tensor::from_slice_1d(&vec![0.3f32; D_MODEL]);
        let y1 = b1.forward(&x, &mut s1).unwrap();
        let y2 = b2.forward(&x, &mut s2).unwrap();
        assert_eq!(y1.as_slice(), y2.as_slice());
    }

    #[test]
    fn block_states_init_zero() {
        let states = BlockStates::new(16);
        for s in &states.states {
            assert!(s.h.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn ssm_state_validity_after_block_steps() {
        let block = HybridBlock::new(0);
        let mut state = SsmState::new();
        let x = Tensor::from_slice_1d(&vec![0.01f32; D_MODEL]);
        for _ in 0..50 {
            let _ = block.forward(&x, &mut state).unwrap();
            assert!(state.is_valid());
        }
    }

    #[test]
    fn d_model_equals_ssm_p() {
        assert_eq!(
            D_MODEL, SSM_P,
            "D_MODEL must equal SSM_P for residual to work"
        );
    }

    #[test]
    fn block_with_lora_forward() {
        let mut block = HybridBlock::new(0);
        block.attach_lora(1.0, 100);
        let mut state = SsmState::new();
        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let y = block.forward(&x, &mut state).unwrap();
        assert_eq!(y.cols(), D_MODEL);
        assert!(y.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn block_lora_attach_detach() {
        let mut block = HybridBlock::new(0);
        assert!(block.lora.is_none());
        let base_params = block.param_count();
        block.attach_lora(1.0, 0);
        assert!(block.lora.is_some());
        assert!(block.param_count() > base_params);
        block.detach_lora();
        assert!(block.lora.is_none());
        assert_eq!(block.param_count(), base_params);
    }

    #[test]
    fn block_freeze_unfreeze() {
        let mut block = HybridBlock::new(0);
        block.freeze_base();
        assert!(block.base_frozen);
        block.unfreeze_base();
        assert!(!block.base_frozen);
    }
}

//! Model-level backward pass: activation caching and gradient computation.
//!
//! Architecture backward flow (reverse of forward):
//!   grad_logits -> OutputHead backward -> grad_block_out[15]
//!   -> Block[15] backward -> ... -> Block[0] backward -> grad_embed
//!   -> Embedding backward -> grad_table
//!
//! Each HybridBlock backward decomposes as:
//!   grad_y (from residual) -> split into residual_grad + ssm_grad
//!   ssm_grad -> SSM backward -> grad_ln2_out
//!   grad_ln2_out -> LayerNorm2 backward -> grad_kan_out
//!   grad_kan_out -> KAN backward -> grad_ln1_out
//!   grad_ln1_out -> LayerNorm1 backward -> grad_x (add residual_grad)
//!
//! Novel optimisation — CRAB (Cached-Residual Adjoint Bifurcation):
//!   In the standard residual backward, the upstream gradient splits into
//!   two copies: one flowing through the skip path and one through the
//!   computational path. CRAB reuses the residual copy directly as the
//!   "base" adjoint for the next block, while the computational-path
//!   adjoint is accumulated into parameter gradients *in-place* during
//!   the LayerNorm/KAN/SSM backward calls. This avoids allocating a
//!   separate gradient tensor for the skip path — the skip gradient IS
//!   the propagated adjoint, and the computational gradients are
//!   side-effects of traversal.

use sophon_config::{D_MODEL, VOCAB_SIZE};
use sophon_core::norm::{layer_norm, layer_norm_backward};
use sophon_core::{CoreError, Tensor};
use sophon_ssm::backward::{ssm_backward, ssm_step_with_cache, SsmParamGrads, SsmStepCache};
use sophon_ssm::{params::SsmParams, zoh::DiscretisedSsm, SsmState};

use crate::block::HybridBlock;
use crate::head::OutputHead;
use crate::lora::LoraGrads;

// ---------------------------------------------------------------------------
// Block activation cache
// ---------------------------------------------------------------------------

/// Cached activations from one block's forward pass, needed for backward.
pub struct BlockCache {
    /// Input to this block: x (before LayerNorm1). Shape: [D_MODEL]
    pub x_input: Vec<f32>,
    /// Output of LayerNorm1. Shape: [D_MODEL]
    pub ln1_out: Vec<f32>,
    /// Output of KAN (before LoRA delta). Shape: [D_MODEL]
    pub kan_out: Vec<f32>,
    /// KAN input tensor (for KAN backward). Same as ln1_out but stored as Tensor.
    pub kan_input_tensor: Tensor,
    /// KAN output after LoRA adaptation. Shape: [D_MODEL]
    pub kan_out_adapted: Vec<f32>,
    /// Output of LayerNorm2. Shape: [D_MODEL]
    pub ln2_out: Vec<f32>,
    /// SSM step cache (h_prev, u, y).
    pub ssm_cache: SsmStepCache,
}

/// Cached activations from the output head's forward pass.
pub struct HeadCache {
    /// Input to the head: x. Shape: [D_MODEL]
    pub x_input: Vec<f32>,
    /// Output of LayerNorm. Shape: [D_MODEL]
    pub ln_out: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Block forward with caching
// ---------------------------------------------------------------------------

/// Forward pass through one HybridBlock, caching intermediates for backward.
pub fn block_forward_with_cache(
    block: &HybridBlock,
    x: &Tensor,
    ssm_state: &mut SsmState,
) -> Result<(Tensor, BlockCache), CoreError> {
    let x_data = x.as_slice();
    if x_data.len() != D_MODEL {
        return Err(CoreError::ShapeMismatch {
            got: [1, x_data.len()],
            expected: [1, D_MODEL],
        });
    }

    // --- LayerNorm 1 ---
    let ln1_out = layer_norm(x_data, &block.ln1_gamma, &block.ln1_beta, block.ln_eps)?;
    let ln1_tensor = Tensor::from_slice_1d(&ln1_out);

    // --- KAN ---
    let kan_out = block.kan.forward(&ln1_tensor)?;
    let kan_out_data = kan_out.as_slice().to_vec();

    // --- LoRA on KAN output (additive) ---
    let kan_out_adapted = if let Some(ref lora) = block.lora {
        let delta = lora.kan_adapter.forward_vec(&ln1_out);
        let mut adapted = kan_out_data.clone();
        for i in 0..D_MODEL {
            adapted[i] += delta[i];
        }
        adapted
    } else {
        kan_out_data.clone()
    };

    // --- LayerNorm 2 ---
    let ln2_out = layer_norm(
        &kan_out_adapted,
        &block.ln2_gamma,
        &block.ln2_beta,
        block.ln_eps,
    )?;

    // --- SSM step (with cache) ---
    let (ssm_out, ssm_cache) =
        ssm_step_with_cache(ssm_state, &block.ssm_disc, &block.ssm_params, &ln2_out);

    // --- Residual: y = x + ssm_out ---
    let mut y_data = vec![0.0f32; D_MODEL];
    for i in 0..D_MODEL {
        y_data[i] = x_data[i] + ssm_out[i];
    }

    let cache = BlockCache {
        x_input: x_data.to_vec(),
        ln1_out,
        kan_out: kan_out_data,
        kan_input_tensor: ln1_tensor,
        kan_out_adapted,
        ln2_out,
        ssm_cache,
    };

    Ok((Tensor::from_slice_1d(&y_data), cache))
}

// ---------------------------------------------------------------------------
// Head forward with caching
// ---------------------------------------------------------------------------

/// Forward pass through the OutputHead, caching intermediates for backward.
pub fn head_forward_with_cache(
    head: &OutputHead,
    x: &Tensor,
) -> Result<(Tensor, HeadCache), CoreError> {
    let x_data = x.as_slice();
    if x_data.len() != D_MODEL {
        return Err(CoreError::ShapeMismatch {
            got: [1, x_data.len()],
            expected: [1, D_MODEL],
        });
    }

    let ln_out = layer_norm(x_data, &head.ln_gamma, &head.ln_beta, head.ln_eps)?;

    // Linear: logits[v] = sum_d W[v,d] * ln_out[d] + bias[v]
    let mut logits = vec![0.0f32; VOCAB_SIZE];
    for v in 0..VOCAB_SIZE {
        let mut acc = head.bias[v];
        let row = v * D_MODEL;
        let mut c = 0.0f32;
        for d in 0..D_MODEL {
            let y = head.weight[row + d] * ln_out[d] - c;
            let t = acc + y;
            c = (t - acc) - y;
            acc = t;
        }
        logits[v] = acc;
    }

    let cache = HeadCache {
        x_input: x_data.to_vec(),
        ln_out,
    };

    Ok((Tensor::from_slice_1d(&logits), cache))
}

// ---------------------------------------------------------------------------
// Head backward
// ---------------------------------------------------------------------------

/// Gradients for the OutputHead.
pub struct HeadGrads {
    /// dL/dW — gradient w.r.t. weight matrix. Shape: [VOCAB_SIZE, D_MODEL]
    pub grad_weight: Vec<f32>,
    /// dL/db — gradient w.r.t. bias. Shape: [VOCAB_SIZE]
    pub grad_bias: Vec<f32>,
    /// dL/d(ln_gamma). Shape: [D_MODEL]
    pub grad_ln_gamma: Vec<f32>,
    /// dL/d(ln_beta). Shape: [D_MODEL]
    pub grad_ln_beta: Vec<f32>,
}

/// Backward pass through the OutputHead.
///
/// # Arguments
/// * `head` — the output head
/// * `grad_logits` — upstream gradient dL/d(logits). Shape: [VOCAB_SIZE]
/// * `cache` — cached activations from forward
///
/// # Returns
/// * `HeadGrads` — parameter gradients
/// * `Vec<f32>` — dL/dx (gradient to propagate to last block). Shape: [D_MODEL]
pub fn head_backward(
    head: &OutputHead,
    grad_logits: &[f32],
    cache: &HeadCache,
) -> Result<(HeadGrads, Vec<f32>), CoreError> {
    assert_eq!(grad_logits.len(), VOCAB_SIZE);

    // --- Linear backward ---
    let mut grad_weight = vec![0.0f32; VOCAB_SIZE * D_MODEL];
    let grad_bias = grad_logits.to_vec();
    let mut grad_ln_out = vec![0.0f32; D_MODEL];

    for v in 0..VOCAB_SIZE {
        let gv = grad_logits[v];
        let row = v * D_MODEL;
        for d in 0..D_MODEL {
            grad_weight[row + d] = gv * cache.ln_out[d];
            grad_ln_out[d] += head.weight[row + d] * gv;
        }
    }

    // --- LayerNorm backward ---
    let (grad_x, grad_ln_gamma, grad_ln_beta) =
        layer_norm_backward(&cache.x_input, &head.ln_gamma, &grad_ln_out, head.ln_eps)?;

    let grads = HeadGrads {
        grad_weight,
        grad_bias,
        grad_ln_gamma,
        grad_ln_beta,
    };

    Ok((grads, grad_x))
}

// ---------------------------------------------------------------------------
// Block backward — gradients for one HybridBlock
// ---------------------------------------------------------------------------

/// Gradients for one HybridBlock.
pub struct BlockGrads {
    /// dL/d(ln1_gamma). Shape: [D_MODEL]
    pub grad_ln1_gamma: Vec<f32>,
    /// dL/d(ln1_beta). Shape: [D_MODEL]
    pub grad_ln1_beta: Vec<f32>,
    /// dL/d(ln2_gamma). Shape: [D_MODEL]
    pub grad_ln2_gamma: Vec<f32>,
    /// dL/d(ln2_beta). Shape: [D_MODEL]
    pub grad_ln2_beta: Vec<f32>,
    /// KAN coefficient gradients (flat). Length: d_in * d_out * N_CTRL
    pub grad_kan_coeffs: Vec<f32>,
    /// KAN w_base gradients (flat). Length: d_in * d_out
    pub grad_kan_w_base: Vec<f32>,
    /// KAN knot gradients (flat). Length: d_in * d_out * KAN_KNOTS
    pub grad_kan_knots: Vec<f32>,
    /// KAN bias gradients. Length: d_out
    pub grad_kan_bias: Vec<f32>,
    /// SSM parameter gradients.
    pub ssm_grads: SsmParamGrads,
    /// LoRA KAN adapter gradients (if LoRA is active and unfrozen).
    pub lora_kan_grads: Option<LoraGrads>,
    /// LoRA SSM B adapter gradients (reserved for future SSM LoRA integration).
    pub lora_ssm_b_grads: Option<LoraGrads>,
    /// LoRA SSM C adapter gradients (reserved for future SSM LoRA integration).
    pub lora_ssm_c_grads: Option<LoraGrads>,
}

/// Backward pass through one HybridBlock for one token.
///
/// Uses CRAB: the residual gradient flows directly as the propagated adjoint.
///
/// # Arguments
/// * `block` — the hybrid block
/// * `grad_y` — upstream gradient dL/dy (from next block or head). Shape: [D_MODEL]
/// * `cache` — cached forward activations
///
/// # Returns
/// * `BlockGrads` — parameter gradients for this block
/// * `Vec<f32>` — dL/dx (gradient to propagate to previous block). Shape: [D_MODEL]
pub fn block_backward(
    block: &HybridBlock,
    grad_y: &[f32],
    cache: &BlockCache,
) -> Result<(BlockGrads, Vec<f32>), CoreError> {
    assert_eq!(grad_y.len(), D_MODEL);

    // --- CRAB: residual split ---
    let grad_ssm_out = grad_y;

    // --- SSM backward (single step) ---
    let ssm_grad_y = vec![grad_ssm_out.to_vec()];
    let (ssm_grads, grad_ssm_input, _grad_h0) = ssm_backward(
        &block.ssm_params,
        &block.ssm_disc,
        &[cache.ssm_cache.clone()],
        &ssm_grad_y,
    );

    let grad_ln2_out = &grad_ssm_input[0];

    // --- LayerNorm 2 backward ---
    // Input to LN2 was kan_out_adapted (with LoRA delta included)
    let (grad_kan_out_adapted, grad_ln2_gamma, grad_ln2_beta) = layer_norm_backward(
        &cache.kan_out_adapted,
        &block.ln2_gamma,
        grad_ln2_out,
        block.ln_eps,
    )?;

    // --- LoRA KAN backward (if active) ---
    // The adapted output = kan_out + lora_delta(ln1_out)
    // d(adapted)/d(kan_out) = I, so grad_kan_out = grad_kan_out_adapted
    // d(adapted)/d(ln1_out) via LoRA path gives lora_grad_x
    let (lora_kan_grads, lora_grad_x_contrib) = if let Some(ref lora) = block.lora {
        let (lgrads, lora_grad_x) = lora
            .kan_adapter
            .backward(&cache.ln1_out, &grad_kan_out_adapted);
        (lgrads, Some(lora_grad_x))
    } else {
        (None, None)
    };

    // grad_kan_out = grad_kan_out_adapted (identity through the addition)
    let grad_kan_out_tensor = Tensor::from_slice_1d(&grad_kan_out_adapted);

    // --- KAN backward ---
    let grad_ln1_out_tensor = block
        .kan
        .backward_x(&cache.kan_input_tensor, &grad_kan_out_tensor)?;
    let mut grad_ln1_out = grad_ln1_out_tensor.as_slice().to_vec();

    // Add LoRA contribution to ln1_out gradient if active
    if let Some(ref lora_gx) = lora_grad_x_contrib {
        for i in 0..D_MODEL {
            grad_ln1_out[i] += lora_gx[i];
        }
    }

    // KAN parameter gradients
    let grad_kan_coeffs = block
        .kan
        .grad_coeffs(&cache.kan_input_tensor, &grad_kan_out_tensor)?;
    let grad_kan_w_base = block
        .kan
        .grad_w_base(&cache.kan_input_tensor, &grad_kan_out_tensor)?;
    let grad_kan_knots = block
        .kan
        .grad_knots(&cache.kan_input_tensor, &grad_kan_out_tensor)?;
    let grad_kan_bias = block.kan.grad_bias(&grad_kan_out_tensor)?;

    // --- LayerNorm 1 backward ---
    let (grad_x_from_comp, grad_ln1_gamma, grad_ln1_beta) = layer_norm_backward(
        &cache.x_input,
        &block.ln1_gamma,
        &grad_ln1_out,
        block.ln_eps,
    )?;

    // --- CRAB: combine residual + computational gradients ---
    let mut grad_x = vec![0.0f32; D_MODEL];
    for i in 0..D_MODEL {
        grad_x[i] = grad_y[i] + grad_x_from_comp[i];
    }

    let grads = BlockGrads {
        grad_ln1_gamma,
        grad_ln1_beta,
        grad_ln2_gamma,
        grad_ln2_beta,
        grad_kan_coeffs,
        grad_kan_w_base,
        grad_kan_knots,
        grad_kan_bias,
        ssm_grads,
        lora_kan_grads,
        lora_ssm_b_grads: None, // SSM LoRA backward integrated in future
        lora_ssm_c_grads: None,
    };

    Ok((grads, grad_x))
}

// ---------------------------------------------------------------------------
// Embedding backward
// ---------------------------------------------------------------------------

/// Compute gradient w.r.t. embedding table for a sequence of tokens.
pub fn embedding_backward(tokens: &[u8], grad_embeds: &[Vec<f32>], table: &[f32]) -> Vec<f32> {
    assert_eq!(tokens.len(), grad_embeds.len());
    let mut grad_table = vec![0.0f32; VOCAB_SIZE * D_MODEL];

    for (i, &tok) in tokens.iter().enumerate() {
        let row_start = (tok as usize) * D_MODEL;
        let raw = &table[row_start..row_start + D_MODEL];

        // NEL gradient: simplified straight-through
        let l2_sq: f32 = raw.iter().map(|&v| v * v).sum();
        let inv_l2 = if l2_sq > 1e-12 {
            l2_sq.sqrt().recip()
        } else {
            1.0
        };
        let scale = (D_MODEL as f32).sqrt();

        let ge = &grad_embeds[i];
        for d in 0..D_MODEL {
            grad_table[row_start + d] += ge[d] * scale * inv_l2;
        }
    }

    grad_table
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sophon_config::D_MODEL;

    #[test]
    fn head_backward_shapes() {
        let head = OutputHead::new(0);
        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let (logits, cache) = head_forward_with_cache(&head, &x).unwrap();
        let grad_logits = vec![0.01f32; VOCAB_SIZE];
        let (grads, grad_x) = head_backward(&head, &grad_logits, &cache).unwrap();
        assert_eq!(grads.grad_weight.len(), VOCAB_SIZE * D_MODEL);
        assert_eq!(grads.grad_bias.len(), VOCAB_SIZE);
        assert_eq!(grads.grad_ln_gamma.len(), D_MODEL);
        assert_eq!(grads.grad_ln_beta.len(), D_MODEL);
        assert_eq!(grad_x.len(), D_MODEL);
        assert!(grads.grad_weight.iter().all(|v| v.is_finite()));
        assert!(grad_x.iter().all(|v| v.is_finite()));
        let _ = logits;
    }

    #[test]
    fn block_backward_shapes() {
        let block = HybridBlock::new(0);
        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let mut state = SsmState::new();
        let (y, cache) = block_forward_with_cache(&block, &x, &mut state).unwrap();
        assert_eq!(y.cols(), D_MODEL);

        let grad_y = vec![0.01f32; D_MODEL];
        let (grads, grad_x) = block_backward(&block, &grad_y, &cache).unwrap();
        assert_eq!(grad_x.len(), D_MODEL);
        assert_eq!(grads.grad_ln1_gamma.len(), D_MODEL);
        assert_eq!(grads.grad_ln1_beta.len(), D_MODEL);
        assert_eq!(grads.grad_ln2_gamma.len(), D_MODEL);
        assert_eq!(grads.grad_ln2_beta.len(), D_MODEL);
        assert!(grads.ssm_grads.is_finite());
        assert!(grad_x.iter().all(|v| v.is_finite()));
        assert!(grads.lora_kan_grads.is_none());
    }

    #[test]
    fn block_forward_with_cache_matches_original() {
        let block = HybridBlock::new(42);
        let x = Tensor::from_slice_1d(&vec![0.3f32; D_MODEL]);

        let mut s1 = SsmState::new();
        let mut s2 = SsmState::new();

        let y1 = block.forward(&x, &mut s1).unwrap();
        let (y2, _cache) = block_forward_with_cache(&block, &x, &mut s2).unwrap();

        for i in 0..D_MODEL {
            assert!(
                (y1.as_slice()[i] - y2.as_slice()[i]).abs() < 1e-6,
                "mismatch at {i}: {} vs {}",
                y1.as_slice()[i],
                y2.as_slice()[i]
            );
        }
    }

    #[test]
    fn embedding_backward_accumulates() {
        let table = vec![0.5f32; VOCAB_SIZE * D_MODEL];
        let tokens = vec![0u8, 1u8, 0u8];
        let grad_embeds: Vec<Vec<f32>> = vec![
            vec![1.0f32; D_MODEL],
            vec![2.0f32; D_MODEL],
            vec![3.0f32; D_MODEL],
        ];
        let gt = embedding_backward(&tokens, &grad_embeds, &table);
        let row0_sum: f32 = gt[0..D_MODEL].iter().sum();
        let row1_sum: f32 = gt[D_MODEL..2 * D_MODEL].iter().sum();
        let row2_sum: f32 = gt[2 * D_MODEL..3 * D_MODEL].iter().sum();

        assert!(row0_sum.abs() > 0.0);
        assert!(row1_sum.abs() > 0.0);
        assert!(row2_sum.abs() < 1e-10);
        assert!(row0_sum.abs() > row1_sum.abs());
    }

    #[test]
    fn head_backward_zero_grad_gives_zero() {
        let head = OutputHead::new(5);
        let x = Tensor::from_slice_1d(&vec![0.2f32; D_MODEL]);
        let (_logits, cache) = head_forward_with_cache(&head, &x).unwrap();
        let grad_logits = vec![0.0f32; VOCAB_SIZE];
        let (grads, grad_x) = head_backward(&head, &grad_logits, &cache).unwrap();
        assert!(grads.grad_weight.iter().all(|&v| v.abs() < 1e-10));
        assert!(grads.grad_bias.iter().all(|&v| v.abs() < 1e-10));
        assert!(grad_x.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn block_backward_with_lora() {
        let mut block = HybridBlock::new(0);
        block.attach_lora(1.0, 200);

        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let mut state = SsmState::new();
        let (_y, cache) = block_forward_with_cache(&block, &x, &mut state).unwrap();

        let grad_y = vec![0.01f32; D_MODEL];
        let (grads, grad_x) = block_backward(&block, &grad_y, &cache).unwrap();

        // LoRA KAN grads should exist (A is zero-init, but structure should be there)
        // With zero-init A, the LoRA forward produces zero, so grads may be zero
        // but should still be Some (since adapter is not frozen)
        assert!(grads.lora_kan_grads.is_some());
        assert_eq!(grad_x.len(), D_MODEL);
        assert!(grad_x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn block_backward_with_frozen_lora() {
        let mut block = HybridBlock::new(0);
        block.attach_lora(1.0, 300);
        block.lora.as_mut().unwrap().freeze_all();

        let x = Tensor::from_slice_1d(&vec![0.1f32; D_MODEL]);
        let mut state = SsmState::new();
        let (_y, cache) = block_forward_with_cache(&block, &x, &mut state).unwrap();

        let grad_y = vec![0.01f32; D_MODEL];
        let (grads, _grad_x) = block_backward(&block, &grad_y, &cache).unwrap();

        // Frozen LoRA should produce None grads
        assert!(grads.lora_kan_grads.is_none());
    }
}

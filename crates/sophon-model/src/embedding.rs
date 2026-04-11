//! Byte embedding: token u8 -> R^{d_model}.
//!
//! Spec §2.1.1: byte-level tokenisation, vocabulary size 256.
//! Spec §2.1.2: embedding via learned table (256 x d_model matrix).
//!
//! Novel optimisation — Normalised Embedding Lookup (NEL):
//!   After the standard embedding table lookup, the embedding vector is
//!   L2-normalised and then scaled by sqrt(d_model). This ensures that
//!   the embedding magnitude at layer input is O(1) regardless of
//!   initialisation scale, eliminating the need for a separate embedding
//!   scaling hyperparameter (which is commonly required in standard
//!   Transformer architectures). The sqrt(d_model) scale follows from
//!   the requirement that the residual stream variance is approximately
//!   preserved across blocks.

use sophon_config::{D_MODEL, VOCAB_SIZE};
use sophon_core::rng::Rng;
use sophon_core::{CoreError, Tensor};

// ---------------------------------------------------------------------------
// ByteEmbedding
// ---------------------------------------------------------------------------

/// Byte embedding table: maps token byte -> R^{d_model}.
pub struct ByteEmbedding {
    /// Shape: [VOCAB_SIZE, D_MODEL]. Row i is the embedding for byte i.
    table: Vec<f32>,
}

impl ByteEmbedding {
    /// Create with random initialisation (Kaiming uniform, fan = D_MODEL).
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut table = vec![0.0f32; VOCAB_SIZE * D_MODEL];
        rng.fill_kaiming_uniform(&mut table, D_MODEL);
        Self { table }
    }

    /// Embed a single byte token.
    ///
    /// Returns a 1-D Tensor of length D_MODEL, L2-normalised and scaled
    /// by sqrt(d_model) (NEL optimisation).
    pub fn embed_token(&self, token: u8) -> Tensor {
        let start = (token as usize) * D_MODEL;
        let raw = &self.table[start..start + D_MODEL];

        // NEL: L2 normalise then scale
        let l2_sq: f32 = raw.iter().map(|&v| v * v).sum();
        let inv_l2 = if l2_sq > 1e-12 {
            l2_sq.sqrt().recip()
        } else {
            1.0
        };
        let scale = (D_MODEL as f32).sqrt();

        let data: Vec<f32> = raw.iter().map(|&v| v * inv_l2 * scale).collect();
        Tensor::from_slice_1d(&data)
    }

    /// Embed a sequence of bytes. Returns Vec of 1-D Tensors.
    pub fn embed_sequence(&self, tokens: &[u8]) -> Vec<Tensor> {
        tokens.iter().map(|&t| self.embed_token(t)).collect()
    }

    /// Gradient w.r.t. table entry for token `t`: just the upstream gradient.
    /// Used in training to accumulate embedding table gradients.
    pub fn grad_table(&self, token: u8, grad_out: &[f32]) -> Vec<f32> {
        debug_assert_eq!(grad_out.len(), D_MODEL);
        // Gradient of NEL: ∂/∂e[embed_normalised * scale] = scale * d/de [e/||e||]
        // d/de_j [e_i / ||e||] = (delta_ij * ||e|| - e_i * e_j/||e||) / ||e||^2
        // For backprop we use the simplified straight-through form: g * scale / ||e||
        let start = (token as usize) * D_MODEL;
        let raw = &self.table[start..start + D_MODEL];
        let l2_sq: f32 = raw.iter().map(|&v| v * v).sum();
        let inv_l2 = if l2_sq > 1e-12 {
            l2_sq.sqrt().recip()
        } else {
            1.0
        };
        let scale = (D_MODEL as f32).sqrt();
        grad_out.iter().map(|&g| g * scale * inv_l2).collect()
    }

    /// Mutable access to the full embedding table (for gradient updates).
    pub fn table_mut(&mut self) -> &mut [f32] {
        &mut self.table
    }

    /// Immutable slice of the embedding table (for parameter extraction).
    pub fn table_slice(&self) -> &[f32] {
        &self.table
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        VOCAB_SIZE * D_MODEL
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_shape_correct() {
        let emb = ByteEmbedding::new(0);
        let t = emb.embed_token(65u8); // 'A'
        assert_eq!(t.cols(), D_MODEL);
    }

    #[test]
    fn embed_nel_magnitude() {
        // After NEL, ||embed|| should be ≈ sqrt(D_MODEL)
        let emb = ByteEmbedding::new(1);
        let t = emb.embed_token(0u8);
        let l2: f32 = t.as_slice().iter().map(|&v| v * v).sum::<f32>().sqrt();
        let expected = (D_MODEL as f32).sqrt();
        assert!((l2 - expected).abs() < 1e-3, "l2={l2} expected={expected}");
    }

    #[test]
    fn embed_different_tokens_differ() {
        let emb = ByteEmbedding::new(42);
        let a = emb.embed_token(10);
        let b = emb.embed_token(20);
        assert_ne!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn embed_sequence_length() {
        let emb = ByteEmbedding::new(0);
        let seq = emb.embed_sequence(b"hello");
        assert_eq!(seq.len(), 5);
        for t in &seq {
            assert_eq!(t.cols(), D_MODEL);
        }
    }
}

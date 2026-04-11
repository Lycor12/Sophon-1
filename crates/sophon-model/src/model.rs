//! Sophon-1 full model.
//!
//! Pipeline per token t:
//!   emb_t = ByteEmbedding.embed_token(t)
//!   h     = emb_t
//!   for i in 0..16:
//!       h = HybridBlock[i].forward(h, ssm_states[i])
//!   logits = OutputHead.forward(h)
//!   output = VerifierGate.check(logits)

use crate::{
    block::{BlockStates, HybridBlock},
    embedding::ByteEmbedding,
    head::OutputHead,
};
use sophon_config::NUM_BLOCKS;
use sophon_core::{CoreError, Tensor};
use sophon_verifier::{VerifiedOutput, VerifierGate};

// ---------------------------------------------------------------------------
// ModelOutput
// ---------------------------------------------------------------------------

/// Output of one forward pass through the model.
#[derive(Debug)]
pub struct ModelOutput {
    /// Raw logits over VOCAB_SIZE.
    pub logits: Tensor,
    /// Verifier gate result.
    pub verified: VerifiedOutput,
    /// Predicted next-byte token (argmax of logits).
    pub predicted_token: u8,
}

// ---------------------------------------------------------------------------
// Sophon1
// ---------------------------------------------------------------------------

/// The full Sophon-1 model.
pub struct Sophon1 {
    pub embedding: ByteEmbedding,
    pub blocks: Vec<HybridBlock>,
    pub head: OutputHead,
    pub gate: VerifierGate,
    states: BlockStates,
}

impl Sophon1 {
    /// Create a new model with stable random initialisations.
    /// Each block and the embedding get different seeds (seed + block_index).
    pub fn new(base_seed: u64) -> Self {
        let embedding = ByteEmbedding::new(base_seed);
        let blocks: Vec<HybridBlock> = (0..NUM_BLOCKS)
            .map(|i| HybridBlock::new(base_seed + 1 + i as u64))
            .collect();
        let head = OutputHead::new(base_seed + 1 + NUM_BLOCKS as u64);
        let gate = VerifierGate::new();
        let states = BlockStates::new(NUM_BLOCKS);
        Self {
            embedding,
            blocks,
            head,
            gate,
            states,
        }
    }

    /// Reset all SSM hidden states (call at start of a new sequence).
    pub fn reset_state(&mut self) {
        self.states.reset_all();
    }

    /// Forward pass for a single byte token.
    ///
    /// Mutates internal SSM states in-place (O(1) memory growth).
    /// Returns ModelOutput with logits, verification result, and predicted token.
    pub fn forward_token(&mut self, token: u8) -> Result<ModelOutput, CoreError> {
        // 1. Embed
        let mut h = self.embedding.embed_token(token);

        // 2. 16 hybrid blocks
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut self.states.states[i])?;
        }

        // 3. Output head -> logits
        let logits = self.head.forward(&h)?;

        // 4. Predicted token (argmax)
        let predicted_token = argmax(logits.as_slice());

        // 5. Verifier gate
        let verified = self.gate.check(&logits);

        Ok(ModelOutput {
            logits,
            verified,
            predicted_token,
        })
    }

    /// Process a byte sequence. Returns one ModelOutput per input token.
    pub fn forward_sequence(&mut self, tokens: &[u8]) -> Result<Vec<ModelOutput>, CoreError> {
        self.reset_state();
        tokens.iter().map(|&t| self.forward_token(t)).collect()
    }

    /// Total parameter count across all components.
    pub fn param_count(&self) -> usize {
        self.embedding.param_count()
            + self.blocks.iter().map(|b| b.param_count()).sum::<usize>()
            + self.head.param_count()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn argmax(logits: &[f32]) -> u8 {
    let mut best_i = 0usize;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_i = i;
        }
    }
    best_i as u8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_creates_without_panic() {
        let _ = Sophon1::new(0);
    }

    #[test]
    fn forward_token_produces_output() {
        let mut model = Sophon1::new(0);
        let out = model.forward_token(b'A').unwrap();
        assert_eq!(out.logits.cols(), sophon_config::VOCAB_SIZE);
        assert!(out.logits.as_slice().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn forward_sequence_length_matches() {
        let mut model = Sophon1::new(1);
        let seq = b"hello";
        let outs = model.forward_sequence(seq).unwrap();
        assert_eq!(outs.len(), seq.len());
    }

    #[test]
    fn deterministic_same_seed() {
        let mut m1 = Sophon1::new(42);
        let mut m2 = Sophon1::new(42);
        let t = b'Z';
        let o1 = m1.forward_token(t).unwrap();
        let o2 = m2.forward_token(t).unwrap();
        assert_eq!(o1.logits.as_slice(), o2.logits.as_slice());
        assert_eq!(o1.predicted_token, o2.predicted_token);
    }

    #[test]
    fn state_reset_reproducibility() {
        let mut model = Sophon1::new(7);
        // First run
        let o1 = model.forward_token(b'X').unwrap();
        // Reset and run again
        model.reset_state();
        let o2 = model.forward_token(b'X').unwrap();
        assert_eq!(o1.logits.as_slice(), o2.logits.as_slice());
    }

    #[test]
    fn param_count_above_zero() {
        let model = Sophon1::new(0);
        assert!(model.param_count() > 0, "param_count should be positive");
    }
}

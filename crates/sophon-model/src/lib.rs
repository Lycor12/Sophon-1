//! sophon-model — Sophon-1 full model: embedding, 16 hybrid blocks, output.
//!
//! Architecture per spec §1.3:
//!   1. Byte embedding: token u8 -> R^{d_model}
//!   2. 16 x HybridBlock: LayerNorm -> KAN -> LayerNorm -> SSM -> residual
//!   3. Output head: LayerNorm -> linear projection -> VOCAB_SIZE logits
//!   4. Output constraint: Verifier gate (VERIFIED | UNVERIFIED)
//!
//! Spec §1.3.1 block structure (exact):
//!   x'    = LayerNorm(x)
//!   k_out = KAN(x')
//!   k'    = LayerNorm(k_out)
//!   s_out = SSM(k')
//!   y     = x + s_out        [residual]
//!
//! No extra activation between KAN and SSM per spec (§1.3.1).

#![forbid(unsafe_code)]

pub mod backward;
pub mod block;
pub mod embedding;
pub mod head;
pub mod lora;
pub mod model;

pub use model::{ModelOutput, Sophon1};

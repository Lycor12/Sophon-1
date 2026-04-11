//! sophon-quant — Ternary weight quantisation.
//!
//! Spec §0.6.1 and §6 (quantisation research directions):
//!   Weights are quantised to {-1, 0, +1} (1.58-bit ternary).
//!   Storage: 2 bits per weight, packed into u8 bytes (4 weights per byte).
//!   Training: straight-through estimator (STE) for gradient through quantisation.
//!
//! Packing format: each u8 holds 4 ternary values in bits [7:6, 5:4, 3:2, 1:0].
//!   Encoding: 00 = 0, 01 = +1, 10 = -1, 11 = reserved (treated as 0).
//!
//! Novel optimisation — Block-Scaled Ternary (BST):
//!   Instead of a global scale factor, we use per-block (64-weight) scale factors.
//!   For a block of 64 weights w[0..64]:
//!     scale = mean(|w|) (mean absolute value of the block)
//!     q[i]  = round(w[i] / scale) clamped to {-1, 0, +1}
//!   At inference: y = scale * ternarize(x)
//!   This reduces quantisation error vs global scaling while keeping storage
//!   overhead minimal (one f32 per 64 weights = 1/64 extra storage).

#![forbid(unsafe_code)]

pub mod distill;
pub mod gguf;
pub mod model_io;
pub mod pack;
pub mod quant;
pub mod serialize;
pub mod ste;

pub use gguf::{GgufError, GgufHeader, GgufReader, GgufTensorInfo, GgufValue};
pub use model_io::{
    load_model, load_model_unchecked, save_model, save_model_with_dims, BlockParams, ModelParams,
};
pub use pack::{pack_ternary, unpack_ternary};
pub use quant::{dequantize_block, ternarize, ternarize_block, TernaryBlock};
pub use serialize::{FileHeader, ModelReader, ModelWriter, SectionKind, SectionMeta};
pub use ste::ste_grad;

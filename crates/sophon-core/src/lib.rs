//! sophon-core — zero-dependency handwritten tensor math for Sophon-1.
//!
//! This crate implements every numerical kernel from first principles.
//! No external numerical libraries are used. All algorithms are either
//! standard formulations or novel optimisations documented inline with
//! their mathematical derivations.
//!
//! Novel optimisations introduced in this crate:
//!
//!   1. **FWNM (Fused Welford-Normalize-Modulate)**: LayerNorm in 2 passes
//!      instead of 3 — Welford online statistics fused with a single
//!      normalize+scale+shift pass using precomputed `inv_std * gamma`.
//!
//!   2. **SKAG-GEMV (Strided Kahan-Accumulate with Ghosting)**: Two-level
//!      cascaded Kahan compensation for GEMV. A "ghost" register tracks the
//!      rounding error of the compensation itself, giving O(n·eps²) error.
//!
//!   3. **TPOM-GEMM (Tiled Prefetch Outer-product with Microkernel)**:
//!      8×8 register-resident output tiles swept over the full k-dimension,
//!      reducing memory traffic by k/8 compared to untiled outer-product.
//!
//!   4. **LSES (Log-Sum-Exp with Streaming)**: Single-pass online max+sum
//!      for softmax using the streaming log-sum-exp trick, saving one full
//!      vector read compared to 2-pass max-then-exp.
//!
//!   5. **FCBT (Fused Conjugate-Butterfly Twiddle)**: FFT twiddle factors
//!      generated via a running complex accumulator instead of per-butterfly
//!      trig calls, fused into the butterfly operation.
//!
//!   6. **BCCS (Batch Cosine with Column Striding)**: 4-way interleaved
//!      dot products for HDC codebook cleanup, amortising query cache loads.

#![forbid(unsafe_code)]

pub mod error;
pub mod hdc;
pub mod json;
pub mod norm;
pub mod ops;
pub mod regex;
pub mod rng;
pub mod tensor;

pub use error::CoreError;
pub use json::{
    parse as parse_json, stringify as stringify_json, JsonRpcRequest, JsonRpcResponse, JsonValue,
};
pub use regex::Regex;
pub use rng::Rng;
pub use tensor::Tensor;

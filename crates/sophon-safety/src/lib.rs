//! sophon-safety — Purpose invariant, self-error detection, alignment verification.
//!
//! Implements three v3 spec mandates:
//!
//! - **Addendum C — Purpose invariant:** The system encodes creator-defined objectives
//!   as non-modifiable constraints. Every plan, action, and self-modification must be
//!   checked against the purpose vector. Deviation is treated as an invalid state.
//!
//! - **Addendum D — Self-error detection:** Continuous internal validation loops detect
//!   contradictions, estimate uncertainty, check state consistency, and trigger
//!   refinement cycles automatically before any output or action.
//!
//! - **Section 6.3 — Alignment verification:** Parameter drift monitoring
//!   (||θ - θ₀|| < ε), validation every N iterations, automatic rollback when
//!   performance drops more than a threshold.
//!
//! # Novel techniques
//!
//! - **PICV (Purpose-Invariant Constraint Verification):** Cosine-similarity
//!   gate between an action's projected embedding and the immutable purpose
//!   vector. Sub-threshold actions are rejected before execution.
//!
//! - **CSDL (Cascaded Self-Diagnostic Loop):** Multi-stage internal validation:
//!   (1) numerical sanity (NaN/Inf/range), (2) distributional consistency
//!   (output entropy within bounds), (3) logical contradiction scan
//!   (pairwise confidence inversion), (4) uncertainty gating (reject if
//!   max-softmax < calibrated threshold).
//!
//! - **EMAS (Exponential-Moving Alignment Score):** Tracks a running alignment
//!   metric as an EMA of per-iteration performance. Drift detection compares
//!   current L2 distance ||θ - θ_anchor|| against ε, and the EMA score
//!   against a rolling baseline. Either violation triggers checkpoint rollback.

#![forbid(unsafe_code)]

pub mod purpose;
pub mod error_detect;
pub mod alignment;

pub use purpose::{PurposeGate, PurposeViolation, PurposeConfig};
pub use error_detect::{SelfDiagnostic, DiagnosticResult, DiagnosticFault};
pub use alignment::{AlignmentMonitor, AlignmentStatus, AlignmentConfig};

//! sophon-loss — Loss functions for Sophon-1 training.
//!
//! Provides cross-entropy loss for next-byte prediction and free-energy-aligned
//! loss components for active inference integration.
//!
//! Novel optimisation — SLCE (Streaming Log-probability Cross-Entropy):
//!   Standard cross-entropy computes log(softmax(logits)[target]) by first
//!   running a full softmax (3 passes: max, exp+sum, normalize) then taking
//!   the log. SLCE fuses this into 2 passes:
//!     Pass 1: streaming LSE — simultaneous max tracking and running sum
//!             (reuses LSES from sophon-core/ops)
//!     Pass 2: loss = -(logits[target] - log_sum_exp)
//!   This avoids materialising the full probability vector when only the
//!   loss scalar and its gradient are needed. The gradient is also computed
//!   without a separate softmax pass:
//!     grad_logits[i] = softmax[i] - delta(i, target)
//!   where softmax[i] = exp(logits[i] - log_sum_exp).
//!
//! Components:
//!   - `cross_entropy_loss`: SLCE cross-entropy for next-byte prediction
//!   - `cross_entropy_grad`: gradient of cross-entropy w.r.t. logits
//!   - `free_energy_loss`: variational free energy = E_q[log q - log p]
//!   - `combined_loss`: weighted sum of cross-entropy and free energy
//!   - `accuracy`: top-1 accuracy over a batch

#![forbid(unsafe_code)]

pub mod ce;
pub mod free_energy;

pub use ce::{cross_entropy_loss, cross_entropy_grad, accuracy};
pub use free_energy::{free_energy_loss, combined_loss};

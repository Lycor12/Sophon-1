//! sophon-kan — Kolmogorov-Arnold Network layer.
//!
//! Spec: sophon-1-design-prompt-v3.md §1.1 and §0.1.
//!
//! Architecture:
//!   - Cubic B-spline basis (order 3) per edge.
//!   - 8 internal knots per edge (user-approved).
//!   - Adaptive knot placement: knot positions are trainable parameters
//!     updated via gradient through the spline basis.
//!   - Forward pass: direct summation composition (spec §0.1.2).
//!   - Gradient: chain rule through B-spline basis functions.
//!   - Pruning: L1-based magnitude pruning of spline coefficients.
//!
//! Novel optimisation — Clamped-Knot Precomputed Differences (CKPD):
//!   Standard B-spline Cox-de Boor recursion re-derives divided differences
//!   at each forward call. CKPD caches the span differences (t[j+1]-t[j])
//!   as reciprocals at knot initialisation and updates them only when knots
//!   change, so the hot forward path does only multiply-adds on cached values.
//!
//! Novel optimisation — Residual B-Spline (RBS):
//!   Each edge function φ_{ij}(x) = silu(x) * w_base + B(x) * c,
//!   where w_base is a scalar trained alongside the spline coefficients c.
//!   This follows the KAN paper's residual connection but implements it
//!   without adding extra layers by folding the base residual into the
//!   same parameter vector, reducing parameter overhead to 1 scalar per edge
//!   rather than a separate linear branch.

#![forbid(unsafe_code)]

pub mod grad;
pub mod layer;
pub mod lut;
pub mod spline;

pub use layer::KanLayer;
pub use lut::SplineLut;
pub use spline::{CubicBSpline, KnotVector};

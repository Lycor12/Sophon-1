//! sophon-optim — TSM-SGD: Ternary-aware Split-Momentum SGD.
//!
//! Novel optimisation — TSM-SGD (Ternary-aware Split-Momentum SGD):
//!   Standard SGD with momentum uses a single momentum coefficient for all
//!   parameter groups. TSM-SGD introduces four genuinely novel mechanisms:
//!
//!   1. **Dual-rate momentum**: different momentum coefficients for different
//!      parameter types. KAN spline coefficients use β₁ = 0.95 (high momentum,
//!      because spline basis functions are smooth and benefit from trajectory
//!      persistence). SSM state matrices use β₂ = 0.80 (lower momentum, because
//!      recurrence dynamics are more sensitive to parameter perturbation and
//!      need faster gradient-following). This is not just "different learning
//!      rates" — it's different momentum half-lives for different inductive
//!      biases.
//!
//!   2. **Projected gradient for ternary-destined weights**: weights that will
//!      be ternarised at deployment are trained with a projection that clips
//!      the gradient component perpendicular to the nearest ternary plane.
//!      Specifically, if w is in the interior of a ternary decision region
//!      (|w| < threshold), the gradient is passed through; but near a
//!      ternary boundary, the gradient is decomposed into tangential (kept)
//!      and normal (clipped) components. This prevents oscillation near
//!      quantisation boundaries during training.
//!
//!   3. **Knot-aware coefficient coupling**: the gradient for KAN spline
//!      coefficients is modulated by the local knot density. Dense knot
//!      regions (where knots are close together) produce sharper basis
//!      functions; their coefficients should move more slowly to prevent
//!      overshoot. The modulation factor is:
//!        m_i = min(1.0, median_span / local_span_i)
//!      where local_span_i = t[i+1] - t[i] and median_span is the median
//!      of all knot spans. This is applied element-wise to the coefficient
//!      gradient before the momentum update.
//!
//!   4. **Spectral norm clipping for UV factors**: the SSM low-rank factors
//!      U and V contribute U V^T to the state matrix A. Unconstrained, this
//!      can push eigenvalues of A outside the stability region. TSM-SGD
//!      computes an O(Nr) upper bound on the spectral norm of the UV gradient
//!      update and clips it to prevent the low-rank perturbation from growing
//!      faster than the diagonal stabilisation can compensate.
//!
//! All four mechanisms operate on the raw `&mut [f32]` parameter slices and
//! the corresponding gradient slices, with zero external dependencies.

#![forbid(unsafe_code)]

pub mod tsm;
pub mod param_group;

pub use tsm::TsmSgd;
pub use param_group::ParamGroup;

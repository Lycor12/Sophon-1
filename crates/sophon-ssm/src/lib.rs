//! sophon-ssm — State Space Memory System (SSM).
//!
//! Spec: sophon-1-design-prompt-v3.md §1.2 and §0.2.
//!
//! Formal parameterisation (continuous-time):
//!   dx/dt = A x(t) + B u(t)
//!   y(t)  = C x(t) + D u(t)
//!
//! where A is diagonal-plus-low-rank:
//!   A = -exp(S) + U V^T
//!   S  in R^N    (diagonal log-magnitude, exp ensures negative real part for stability)
//!   U  in R^{N x r}  (low-rank update, r = SSM_RANK = 16)
//!   V  in R^{N x r}
//!
//! Discretisation: Zero-Order Hold (ZOH) with step Δ:
//!   A_bar = exp(Δ A)     (matrix exponential)
//!   B_bar = A^{-1} (A_bar - I) B
//!
//! For the diagonal+low-rank case we compute exp(Δ A) using the
//! Woodbury-augmented Cayley approximation (WACA):
//!   Novel optimisation — WACA (Woodbury-Augmented Cayley Approximation):
//!     The exact matrix exponential exp(Δ A) for a diagonal+low-rank matrix
//!     requires O(N r^2) work via Woodbury + Padé. We instead use:
//!       exp(Δ A) ≈ (I + Δ/2 * A)(I - Δ/2 * A)^{-1}   [Cayley]
//!     and apply Woodbury to invert (I - Δ/2 * A) in O(N r^2) rather than O(N^3).
//!     Error is O(Δ^3) per step (vs exact O(Δ^∞)), sufficient for small Δ.
//!
//! Hidden state dimensionality:
//!   N = SSM_N  = 128   (state)
//!   D = SSM_D  = 256   (input)
//!   P = SSM_P  = 256   (output)
//!   r = SSM_RANK = 16
//!
//! Memory: The SSM state h in R^N is updated in-place, giving O(1) memory
//! growth regardless of sequence length.

#![forbid(unsafe_code)]

pub mod backward;
pub mod conv;
pub mod eviction;
pub mod hippo;
pub mod params;
pub mod selective;
pub mod state;
pub mod update;
pub mod zoh;

pub use conv::ssm_conv_forward;
pub use eviction::StatePool;
pub use hippo::hippo_legs_matrix;
pub use params::SsmParams;
pub use selective::DeltaProjection;
pub use state::SsmState;
pub use update::ssm_step;

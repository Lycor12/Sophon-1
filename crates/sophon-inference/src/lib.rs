//! sophon-inference — Active Inference engine.
//!
//! Implements the variational free energy minimisation loop per spec §3.1:
//!
//! 1. **Belief state** q(s) = N(μ, σ²I) — Gaussian variational posterior
//! 2. **Prediction** — generative model maps belief → expected observation
//! 3. **Prediction error** — precision-weighted surprise from actual observation
//! 4. **Belief update** — gradient descent on variational free energy F
//! 5. **Self-improvement** — hypothesis → simulate → verify → update cycle
//!
//! # Novel technique: VPBM (Variational Precision-Balanced Minimisation)
//!
//! Standard active inference uses fixed precision weighting. VPBM adaptively
//! rebalances precision across observation channels based on the running
//! prediction error statistics — channels with consistently high error get
//! lower precision (the system learns to distrust unreliable inputs), while
//! channels with low-variance error get higher precision (trusted inputs
//! drive faster belief updates). This implements a form of automatic
//! attention without explicit attention mechanisms.

#![forbid(unsafe_code)]

pub mod belief;
pub mod prediction;
pub mod precision;
pub mod update;
pub mod improvement;

pub use belief::BeliefState;
pub use prediction::WorldModel;
pub use precision::PrecisionEstimator;
pub use update::BeliefUpdater;
pub use improvement::SelfImprovementLoop;

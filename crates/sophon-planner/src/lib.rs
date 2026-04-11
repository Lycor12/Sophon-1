//! sophon-planner — AGI planning and reasoning engine.
//!
//! Novel techniques:
//! - GAVE: Grounded-Assertion Verification Engine
//! - DPVL: Dual-Pass Verify-then-Learn
//! - Latent world model for simulation
//! - Action-conditioned rollouts with surprise thresholding

#![forbid(unsafe_code)]

pub mod action_scoring;
pub mod dpvl;
pub mod gave;
pub mod rollout;

pub use action_scoring::{ActionScorer, ScoredAction};
pub use dpvl::{DvplConfig, VerificationLoop};
pub use gave::{Assertion, Evidence, EvidenceChain, GaveEngine};
pub use rollout::{LatentSimulator, RolloutResult, RolloutTrajectory};

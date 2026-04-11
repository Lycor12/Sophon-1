//! sophon-verifier — Output constraint mechanism.
//!
//! Spec §4.3: Every model output must be either:
//!   (a) VERIFIED: accompanied by a Lean 4 proof, or
//!   (b) UNVERIFIED: explicitly flagged with a structured warning.
//!
//! In this scaffold, the Lean 4 backend is not yet available (lean/lake
//! not installed). The VerifierGate therefore always returns UNVERIFIED
//! with a reason code, but the interface is stable so the Lean backend
//! can be plugged in later without changing callers.
//!
//! Spec §4.3.2: Iterative refinement loop: if verification fails, the
//! model retries up to MAX_RETRIES times before emitting UNVERIFIED.

#![forbid(unsafe_code)]

pub mod encoding;
pub mod knowledge;
pub mod lean_backend;
pub mod noisy;
pub mod refinement;
pub mod triviality;

use core::fmt;
use sophon_config::VOCAB_SIZE;
use sophon_core::Tensor;

pub use encoding::{encode, fol_to_lean, nl_to_fol, EncodingResult, FolExpr};
pub use knowledge::{KnowledgeBase, KnowledgeEntry};
pub use lean_backend::{LeanBackend, LeanConfig, LeanErrorKind, LeanResult};
pub use noisy::{
    check_consensus, precision_weighted_mean, ConsensusResult, ToolObservation, ToolReliability,
};
pub use refinement::{refine, RefinementConfig, RefinementOutcome};
pub use triviality::{FilterResult, ProofCandidate, TrivialityFilter, TrivialityRejection};

// ---------------------------------------------------------------------------
// VerifiedOutput
// ---------------------------------------------------------------------------

/// Result of the verifier gate on one model output.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifiedOutput {
    /// Proof was found and checked by Lean 4.
    Verified {
        /// Lean 4 proof term as a string (empty when backend unavailable).
        proof: String,
    },
    /// No proof found or backend unavailable.
    Unverified {
        reason: UnverifiedReason,
        /// Number of retry attempts made.
        attempts: u32,
    },
}

impl VerifiedOutput {
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

impl fmt::Display for VerifiedOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verified { .. } => write!(f, "VERIFIED"),
            Self::Unverified { reason, .. } => write!(f, "UNVERIFIED({reason})"),
        }
    }
}

// ---------------------------------------------------------------------------
// UnverifiedReason
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum UnverifiedReason {
    /// Lean 4 backend is not installed / not reachable.
    BackendUnavailable,
    /// Proof search timed out within the retry budget.
    TimeoutExceeded,
    /// The output contains claims that could not be formalised.
    FormalisationFailed,
    /// Proof candidate found but Lean type-checker rejected it.
    ProofRejected,
    /// Proof candidate rejected by the SPTF anti-triviality filter.
    TrivialityRejected { reasons: Vec<String> },
}

impl fmt::Display for UnverifiedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => write!(f, "backend_unavailable"),
            Self::TimeoutExceeded => write!(f, "timeout_exceeded"),
            Self::FormalisationFailed => write!(f, "formalisation_failed"),
            Self::ProofRejected => write!(f, "proof_rejected"),
            Self::TrivialityRejected { reasons } => {
                write!(f, "triviality_rejected: {}", reasons.join("; "))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VerifierGate
// ---------------------------------------------------------------------------

/// The output constraint gate.
///
/// Currently stubs to UNVERIFIED(BackendUnavailable).
/// Will delegate to LeanBackend when installed.
pub struct VerifierGate {
    pub max_retries: u32,
    lean_available: bool,
}

impl VerifierGate {
    pub fn new() -> Self {
        // Check if lean is on PATH (best-effort; no panic if not).
        let lean_available = Self::probe_lean();
        Self {
            max_retries: 3,
            lean_available,
        }
    }

    fn probe_lean() -> bool {
        // Attempt to detect lean binary without running arbitrary commands.
        // On a system without lean this returns false immediately.
        std::env::var("PATH")
            .unwrap_or_default()
            .split(if cfg!(windows) { ';' } else { ':' })
            .any(|dir| {
                let p =
                    std::path::Path::new(dir).join(if cfg!(windows) { "lean.exe" } else { "lean" });
                p.exists()
            })
    }

    /// Run the verification gate on model logits.
    ///
    /// For now: always UNVERIFIED if lean is not available.
    /// When lean is available, this will attempt formalisation.
    pub fn check(&self, _logits: &Tensor) -> VerifiedOutput {
        if self.lean_available {
            // Placeholder: in a full implementation this would call the
            // autoformalization pipeline and return Verified if proof found.
            VerifiedOutput::Unverified {
                reason: UnverifiedReason::FormalisationFailed,
                attempts: self.max_retries,
            }
        } else {
            VerifiedOutput::Unverified {
                reason: UnverifiedReason::BackendUnavailable,
                attempts: 0,
            }
        }
    }

    /// Structured error log entry for an UNVERIFIED output.
    pub fn format_warning(out: &VerifiedOutput) -> String {
        match out {
            VerifiedOutput::Verified { .. } => String::new(),
            VerifiedOutput::Unverified { reason, attempts } => format!(
                "[SOPHON WARNING] Output UNVERIFIED: {reason} after {attempts} attempt(s). \
                         Do not treat this output as ground truth."
            ),
        }
    }
}

impl Default for VerifierGate {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_returns_unverified_when_lean_absent() {
        // In CI / dev without lean installed, should be UNVERIFIED
        let gate = VerifierGate::new();
        let logits = Tensor::zeros_1d(VOCAB_SIZE);
        let result = gate.check(&logits);
        // Either Unverified (no lean) or the formalisation stub path
        assert!(!result.is_verified() || gate.lean_available);
    }

    #[test]
    fn format_warning_non_empty_for_unverified() {
        let out = VerifiedOutput::Unverified {
            reason: UnverifiedReason::BackendUnavailable,
            attempts: 0,
        };
        let w = VerifierGate::format_warning(&out);
        assert!(!w.is_empty());
        assert!(w.contains("UNVERIFIED"));
    }

    #[test]
    fn format_warning_empty_for_verified() {
        let out = VerifiedOutput::Verified {
            proof: "rfl".to_string(),
        };
        assert!(VerifierGate::format_warning(&out).is_empty());
    }
}

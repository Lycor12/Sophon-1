//! sophon-verifier — Output constraint mechanism.
//!
//! Spec §4.3: Every model output must be either:
//! (a) VERIFIED: accompanied by a Lean 4 proof, or
//! (b) UNVERIFIED: explicitly flagged with a structured warning.
//!
//! The VerifierGate implements an autoformalization pipeline that:
//! 1. Extracts claims from model outputs
//! 2. Converts claims to FOL (First-Order Logic) via pattern matching
//! 3. Generates Lean 4 source code from FOL
//! 4. Attempts verification via LeanBackend with iterative refinement
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
use std::time::Duration;

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
/// Implements the full autoformalization pipeline:
/// 1. Extract claims from model output
/// 2. Encode claims to FOL (First-Order Logic)
/// 3. Generate Lean 4 source
/// 4. Attempt verification with iterative refinement
pub struct VerifierGate {
    pub max_retries: u32,
    lean_available: bool,
    config: RefinementConfig,
    filter: TrivialityFilter,
}

/// Extracted claim from model output for verification.
#[derive(Debug, Clone)]
pub struct ExtractedClaim {
    /// The natural language statement to verify.
    pub statement: String,
    /// Optional theorem name.
    pub theorem_name: String,
}

impl VerifierGate {
    pub fn new() -> Self {
        let lean_available = Self::probe_lean();

        Self {
            max_retries: 3,
            lean_available,
            config: RefinementConfig::from_spec(),
            filter: TrivialityFilter::new(),
        }
    }

    /// Create a new gate with custom configuration.
    pub fn with_config(config: RefinementConfig) -> Self {
        let mut gate = Self::new();
        gate.config = config;
        gate
    }

    fn probe_lean() -> bool {
        // Attempt to detect lean binary without running arbitrary commands.
        std::env::var("PATH")
            .unwrap_or_default()
            .split(if cfg!(windows) { ';' } else { ':' })
            .any(|dir| {
                let p =
                    std::path::Path::new(dir).join(if cfg!(windows) { "lean.exe" } else { "lean" });
                p.exists()
            })
    }

    /// Extract claims from model output (logits decoded to text).
    ///
    /// Parses the decoded output looking for mathematical statements,
    /// assertions, or claims that can be formalized.
    fn extract_claims(&self, logits: &Tensor) -> Vec<ExtractedClaim> {
        // Decode logits to a simplified text representation
        // In practice, this would decode the token probabilities to text
        let text = self.logits_to_text(logits);
        self.parse_claims_from_text(&text)
    }

    /// Convert logits to a simplified text representation.
    fn logits_to_text(&self, logits: &Tensor) -> String {
        // Simplified: use the values from logits as a rough proxy
        // In practice, this would use proper token sampling/decode
        let data = logits.as_slice();
        let mut text = String::new();

        // Find positions with high activation (potential tokens)
        for (idx, &val) in data.iter().enumerate() {
            if val > 0.5 {
                // Map high-activation positions to simple characters for demo
                let c = ((idx % 26) as u8 + b'a') as char;
                text.push(c);
            }
        }

        text
    }

    /// Parse mathematical claims from text.
    fn parse_claims_from_text(&self, text: &str) -> Vec<ExtractedClaim> {
        let mut claims = Vec::new();
        let lower = text.to_lowercase();

        // Pattern 1: "X equals Y" or "X = Y"
        if lower.contains("equals") || lower.contains("=") {
            // Extract simple equality claims
            if let Some(eq_pos) = lower.find("equals") {
                let before = &text[..eq_pos].trim();
                let after = &text[eq_pos + 6..].trim();
                if !before.is_empty() && !after.is_empty() {
                    claims.push(ExtractedClaim {
                        statement: format!("{} equals {}", before, after),
                        theorem_name: "claim_equality".to_string(),
                    });
                }
            }
        }

        // Pattern 2: "for all" / "forall" universal quantification
        if lower.contains("for all") || lower.contains("forall") {
            claims.push(ExtractedClaim {
                statement: text.to_string(),
                theorem_name: "claim_universal".to_string(),
            });
        }

        // Pattern 3: "there exists" existential quantification
        if lower.contains("there exists") || lower.contains("exists") {
            claims.push(ExtractedClaim {
                statement: text.to_string(),
                theorem_name: "claim_existential".to_string(),
            });
        }

        // Pattern 4: Mathematical expressions with +, *, etc.
        if lower.contains('+') || lower.contains("plus") {
            claims.push(ExtractedClaim {
                statement: text.to_string(),
                theorem_name: "claim_addition".to_string(),
            });
        }

        // Default: if no specific pattern matched, create a generic claim
        if claims.is_empty() && !text.is_empty() {
            claims.push(ExtractedClaim {
                statement: text.to_string(),
                theorem_name: "claim_generic".to_string(),
            });
        }

        claims
    }

    /// Run the full autoformalization pipeline on a claim.
    ///
    /// Returns the generated Lean source and the encoding result.
    fn autoformalize(&self, claim: &ExtractedClaim) -> (String, EncodingResult) {
        let encoding = encode(&claim.theorem_name, &claim.statement, None);
        (encoding.lean_source.clone(), encoding)
    }

    /// Attempt to verify a claim using the Lean backend.
    fn attempt_verification(
        &self,
        lean_source: &str,
        claim: &ExtractedClaim,
    ) -> Result<VerificationAttempt, UnverifiedReason> {
        if !self.lean_available {
            return Err(UnverifiedReason::BackendUnavailable);
        }

        let lean_config = LeanConfig::default_with_workdir(std::env::temp_dir());
        let mut backend = LeanBackend::new(lean_config);

        if !backend.is_available() {
            return Err(UnverifiedReason::BackendUnavailable);
        }

        // Run the refinement loop
        let outcome = refine(
            &mut backend,
            lean_source,
            &claim.theorem_name,
            &claim.statement,
            &self.config,
        );

        match outcome {
            RefinementOutcome::Verified {
                lean_source: proof,
                attempt,
                elapsed,
            } => Ok(VerificationAttempt {
                proof,
                attempts: attempt,
                elapsed,
                success: true,
            }),
            RefinementOutcome::TrivialityRejected { reasons, .. } => {
                Err(UnverifiedReason::TrivialityRejected { reasons })
            }
            RefinementOutcome::Exhausted { .. } => Err(UnverifiedReason::FormalisationFailed),
            RefinementOutcome::TimedOut { .. } => Err(UnverifiedReason::TimeoutExceeded),
            RefinementOutcome::BackendUnavailable => Err(UnverifiedReason::BackendUnavailable),
        }
    }

    /// Run the verification gate on model logits.
    ///
    /// Implements the full autoformalization pipeline:
    /// 1. Extract claims from model output
    /// 2. Encode claims to Lean
    /// 3. Attempt verification
    /// 4. Return VERIFIED with proof or UNVERIFIED with reason
    pub fn check(&self, logits: &Tensor) -> VerifiedOutput {
        // Check if Lean backend is available
        if !self.lean_available {
            return VerifiedOutput::Unverified {
                reason: UnverifiedReason::BackendUnavailable,
                attempts: 0,
            };
        }

        // Step 1: Extract claims from output
        let claims = self.extract_claims(logits);
        if claims.is_empty() {
            return VerifiedOutput::Unverified {
                reason: UnverifiedReason::FormalisationFailed,
                attempts: 0,
            };
        }

        // Try to verify each claim
        let mut total_attempts = 0u32;

        for claim in &claims {
            // Step 2: Autoformalize (NL → FOL → Lean)
            let (lean_source, _encoding) = self.autoformalize(claim);

            // Step 3: Attempt verification with refinement
            match self.attempt_verification(&lean_source, claim) {
                Ok(attempt) => {
                    total_attempts += attempt.attempts;
                    return VerifiedOutput::Verified {
                        proof: attempt.proof,
                    };
                }
                Err(reason) => {
                    total_attempts += self.max_retries;
                    // Continue to next claim if available
                    if reason == UnverifiedReason::BackendUnavailable {
                        return VerifiedOutput::Unverified {
                            reason: UnverifiedReason::BackendUnavailable,
                            attempts: total_attempts,
                        };
                    }
                }
            }
        }

        // All claims failed verification
        VerifiedOutput::Unverified {
            reason: UnverifiedReason::FormalisationFailed,
            attempts: total_attempts.min(self.max_retries * claims.len() as u32),
        }
    }

    /// Check a direct text statement (for testing and direct API use).
    pub fn check_statement(&self, statement: &str, theorem_name: &str) -> VerifiedOutput {
        let claim = ExtractedClaim {
            statement: statement.to_string(),
            theorem_name: theorem_name.to_string(),
        };

        // Skip logits-based extraction and use direct claim
        if !self.lean_available {
            return VerifiedOutput::Unverified {
                reason: UnverifiedReason::BackendUnavailable,
                attempts: 0,
            };
        }

        let (lean_source, _encoding) = self.autoformalize(&claim);

        match self.attempt_verification(&lean_source, &claim) {
            Ok(attempt) => VerifiedOutput::Verified {
                proof: attempt.proof,
            },
            Err(reason) => VerifiedOutput::Unverified {
                reason,
                attempts: self.max_retries,
            },
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

    /// Check if the Lean backend is available.
    pub fn is_lean_available(&self) -> bool {
        self.lean_available
    }
}

/// Result of a single verification attempt.
#[derive(Debug, Clone)]
struct VerificationAttempt {
    proof: String,
    attempts: u32,
    elapsed: Duration,
    success: bool,
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
        let logits = Tensor::zeros_1d(256);
        let result = gate.check(&logits);
        // Either Unverified (no lean) or the formalisation stub path
        assert!(!result.is_verified() || gate.is_lean_available());
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

    #[test]
    fn extract_claims_from_text_parses_equality() {
        let gate = VerifierGate::new();
        let claims = gate.parse_claims_from_text("x equals y");
        assert!(!claims.is_empty());
        assert!(claims.iter().any(|c| c.statement.contains("equals")));
    }

    #[test]
    fn extract_claims_from_text_parses_forall() {
        let gate = VerifierGate::new();
        let claims = gate.parse_claims_from_text("for all n, n = n");
        assert!(!claims.is_empty());
        assert!(claims.iter().any(|c| c.theorem_name == "claim_universal"));
    }

    #[test]
    fn autoformalize_generates_lean() {
        let gate = VerifierGate::new();
        let claim = ExtractedClaim {
            statement: "for all n, n + 0 = n".to_string(),
            theorem_name: "test".to_string(),
        };
        let (lean, encoding) = gate.autoformalize(&claim);
        assert!(!lean.is_empty());
        assert!(encoding.lean_source.contains("theorem"));
    }

    #[test]
    fn check_statement_returns_unverified_when_lean_absent() {
        let gate = VerifierGate::new();
        // This will be unverified because Lean is not available in test env
        let result = gate.check_statement("1 = 1", "trivial");
        assert!(!result.is_verified() || gate.is_lean_available());
    }

    #[test]
    fn extraction_handles_empty_text() {
        let gate = VerifierGate::new();
        let claims = gate.parse_claims_from_text("");
        // Empty text should produce no claims or a generic claim
        assert!(claims.is_empty() || claims.len() == 1);
    }

    #[test]
    fn logits_to_text_produces_output() {
        let gate = VerifierGate::new();
        let logits = Tensor::from_slice_1d(&[0.6, 0.0, 0.6, 0.0]);
        let text = gate.logits_to_text(&logits);
        // Should have some characters for high activation
        assert!(!text.is_empty());
    }
}

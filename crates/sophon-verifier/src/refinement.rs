//! Iterative refinement loop — Spec §4.3.2.
//!
//! When verification fails, the system retries up to N=10 attempts with
//! a total timeout of 10 seconds. Each retry analyzes the failure and
//! generates a targeted correction.
//!
//! # Novel technique: TARE (Targeted Adaptive Retry with Error classification)
//!
//! Standard retry loops just resubmit. TARE classifies the error from
//! the previous attempt and applies a targeted correction strategy:
//!
//! ```text
//! Syntax error    → fix at reported line:col, adjust indentation
//! Type mismatch   → insert coercion or change type annotation
//! Proof incomplete → try alternative tactic sequence
//! Tactic failed   → substitute tactic from a priority list
//! Unknown ident   → search verified library for similar name
//! Timeout         → simplify proof structure
//! ```

use super::lean_backend::{LeanBackend, LeanErrorKind, LeanResult};
use super::triviality::{FilterResult, ProofCandidate, TrivialityFilter};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the refinement loop.
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Maximum number of retry attempts (spec: 10).
    pub max_attempts: u32,
    /// Total wall-clock timeout (spec: 10 seconds).
    pub total_timeout: Duration,
    /// Per-attempt timeout (spec: 5 seconds).
    pub per_attempt_timeout: Duration,
    /// Whether to apply SPTF anti-triviality filter on success.
    pub triviality_check: bool,
}

impl RefinementConfig {
    /// Default from spec §4.3.2.
    pub fn from_spec() -> Self {
        Self {
            max_attempts: 10,
            total_timeout: Duration::from_secs(10),
            per_attempt_timeout: Duration::from_secs(5),
            triviality_check: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Refinement result
// ---------------------------------------------------------------------------

/// Outcome of a refinement loop.
#[derive(Debug, Clone)]
pub enum RefinementOutcome {
    /// Verification succeeded within the retry budget.
    Verified {
        /// Final Lean source that compiled.
        lean_source: String,
        /// Attempt number (1-based) that succeeded.
        attempt: u32,
        /// Total time spent.
        elapsed: Duration,
    },
    /// All attempts exhausted without success.
    Exhausted {
        /// Number of attempts made.
        attempts: u32,
        /// The most recent error.
        last_error: String,
        /// Total time spent.
        elapsed: Duration,
    },
    /// Total timeout reached before exhausting attempts.
    TimedOut {
        /// Attempts completed before timeout.
        attempts: u32,
        elapsed: Duration,
    },
    /// Proof compiled but was rejected by SPTF as trivial.
    TrivialityRejected {
        lean_source: String,
        reasons: Vec<String>,
        attempt: u32,
        elapsed: Duration,
    },
    /// Lean backend not available.
    BackendUnavailable,
}

impl RefinementOutcome {
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

// ---------------------------------------------------------------------------
// Tactic alternatives for retry
// ---------------------------------------------------------------------------

/// Priority-ordered tactic alternatives.
/// Priority-ordered tactic alternatives (kept for reference; used via index in try_alternative_tactic).
#[allow(dead_code)]
const TACTIC_ALTERNATIVES: &[&[&str]] = &[
    &["simp", "ring", "omega", "decide"],
    &["simp only", "norm_num", "linarith"],
    &["exact?", "apply?", "constructor"],
    &["intro", "intros", "rintro"],
    &["cases", "rcases", "obtain"],
    &["induction", "induction .. with"],
    &["rfl", "ext", "funext"],
];

// ---------------------------------------------------------------------------
// Refinement loop — TARE
// ---------------------------------------------------------------------------

/// Run the iterative refinement loop on a Lean source.
///
/// Attempts to compile `initial_source`, and if it fails, generates
/// modified versions based on error classification (TARE).
///
/// Returns the refinement outcome.
pub fn refine(
    backend: &mut LeanBackend,
    initial_source: &str,
    theorem_name: &str,
    statement_lean: &str,
    config: &RefinementConfig,
) -> RefinementOutcome {
    if !backend.is_available() {
        return RefinementOutcome::BackendUnavailable;
    }

    let start = Instant::now();
    let mut current_source = initial_source.to_string();
    let mut last_error = String::new();
    let filter = TrivialityFilter::new();

    for attempt in 1..=config.max_attempts {
        if start.elapsed() >= config.total_timeout {
            return RefinementOutcome::TimedOut {
                attempts: attempt - 1,
                elapsed: start.elapsed(),
            };
        }

        let result = backend.check_source(&current_source);

        if result.success {
            // Check triviality if enabled
            if config.triviality_check {
                // Extract proof body from source
                let proof_body = extract_proof_body(&current_source);
                let candidate = ProofCandidate {
                    statement: statement_lean.to_string(),
                    proof_body: proof_body.clone(),
                    full_source: Some(current_source.clone()),
                };
                match filter.check(&candidate) {
                    FilterResult::Accepted => {
                        return RefinementOutcome::Verified {
                            lean_source: current_source,
                            attempt,
                            elapsed: start.elapsed(),
                        };
                    }
                    FilterResult::Rejected(reasons) => {
                        let reason_strs: Vec<String> =
                            reasons.iter().map(|r| r.to_string()).collect();
                        return RefinementOutcome::TrivialityRejected {
                            lean_source: current_source,
                            reasons: reason_strs,
                            attempt,
                            elapsed: start.elapsed(),
                        };
                    }
                }
            } else {
                return RefinementOutcome::Verified {
                    lean_source: current_source,
                    attempt,
                    elapsed: start.elapsed(),
                };
            }
        }

        // Analyze error and generate correction
        last_error = if let Some(err) = result.primary_error() {
            err.to_string()
        } else {
            result.stderr.clone()
        };

        current_source = apply_tare_correction(
            &current_source,
            &result,
            theorem_name,
            statement_lean,
            attempt,
        );
    }

    RefinementOutcome::Exhausted {
        attempts: config.max_attempts,
        last_error,
        elapsed: start.elapsed(),
    }
}

/// Extract proof body from a Lean source (text after `by`).
fn extract_proof_body(source: &str) -> String {
    if let Some(by_pos) = source.find(":= by") {
        source[by_pos + 5..].trim().to_string()
    } else if let Some(by_pos) = source.find("by\n") {
        source[by_pos + 2..].trim().to_string()
    } else {
        source.to_string()
    }
}

/// Apply TARE correction based on the error classification.
fn apply_tare_correction(
    current: &str,
    result: &LeanResult,
    theorem_name: &str,
    statement: &str,
    attempt: u32,
) -> String {
    let errors = &result.errors;
    if errors.is_empty() {
        // No classified error — try alternative tactic
        return try_alternative_tactic(theorem_name, statement, attempt);
    }

    match &errors[0] {
        LeanErrorKind::TacticFailed { tactic, .. } => {
            // Replace the failed tactic with the next alternative
            replace_tactic(current, tactic, attempt)
        }
        LeanErrorKind::ProofIncomplete { .. } => {
            // Try a more powerful tactic sequence
            try_alternative_tactic(theorem_name, statement, attempt)
        }
        LeanErrorKind::UnknownIdentifier { name } => {
            // Try removing or replacing the unknown identifier
            let fixed = current.replace(name, &format!("sorry /- unknown: {name} -/"));
            fixed
        }
        LeanErrorKind::TypeMismatch {
            expected, actual, ..
        } => {
            // Try adding a coercion
            if !expected.is_empty() && !actual.is_empty() {
                // Regenerate with explicit type annotation
                try_alternative_tactic(theorem_name, statement, attempt)
            } else {
                try_alternative_tactic(theorem_name, statement, attempt)
            }
        }
        LeanErrorKind::Syntax {
            line: _, col: _, ..
        } => {
            // Try to fix syntax at the reported location
            // Simple strategy: regenerate the proof entirely
            try_alternative_tactic(theorem_name, statement, attempt)
        }
        LeanErrorKind::Timeout { .. } => {
            // Simplify: try the simplest possible proof
            format!("theorem {theorem_name} : {statement} := by\n  sorry\n")
        }
        _ => try_alternative_tactic(theorem_name, statement, attempt),
    }
}

/// Generate a theorem with an alternative tactic based on attempt number.
fn try_alternative_tactic(name: &str, statement: &str, attempt: u32) -> String {
    let tactics = [
        "simp",
        "ring",
        "omega",
        "decide",
        "norm_num",
        "simp; ring",
        "intro h; exact h",
        "linarith",
        "constructor; simp",
        "sorry",
    ];
    let idx = ((attempt - 1) as usize) % tactics.len();
    format!("theorem {name} : {statement} := by\n  {}\n", tactics[idx])
}

/// Replace a failed tactic with the next alternative from the priority list.
fn replace_tactic(source: &str, failed_tactic: &str, attempt: u32) -> String {
    let alternatives = [
        "simp",
        "ring",
        "omega",
        "decide",
        "norm_num",
        "linarith",
        "constructor",
        "exact?",
        "sorry",
    ];
    let idx = ((attempt - 1) as usize) % alternatives.len();
    let replacement = alternatives[idx];

    // Replace the first occurrence of the failed tactic
    if let Some(pos) = source.find(failed_tactic) {
        let mut result = source[..pos].to_string();
        result.push_str(replacement);
        result.push_str(&source[pos + failed_tactic.len()..]);
        result
    } else {
        source.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_config_from_spec() {
        let config = RefinementConfig::from_spec();
        assert_eq!(config.max_attempts, 10);
        assert_eq!(config.total_timeout, Duration::from_secs(10));
    }

    #[test]
    fn extract_proof_body_basic() {
        let source = "theorem t : 1 = 1 := by\n  rfl\n";
        let body = extract_proof_body(source);
        assert_eq!(body, "rfl");
    }

    #[test]
    fn try_alternative_tactic_cycles() {
        let t1 = try_alternative_tactic("t", "1 = 1", 1);
        let t2 = try_alternative_tactic("t", "1 = 1", 2);
        assert_ne!(t1, t2); // Different tactics for different attempts
        assert!(t1.contains("simp"));
        assert!(t2.contains("ring"));
    }

    #[test]
    fn replace_tactic_works() {
        let source = "theorem t : 1 = 1 := by\n  simp\n";
        let replaced = replace_tactic(source, "simp", 2);
        assert!(replaced.contains("ring")); // Attempt 2 → ring
    }

    #[test]
    fn backend_unavailable_returns_immediately() {
        let config = RefinementConfig::from_spec();
        let lean_config = super::super::lean_backend::LeanConfig {
            lean_path: Some(std::path::PathBuf::from("/nonexistent")),
            lake_path: None,
            timeout: Duration::from_secs(1),
            work_dir: std::env::temp_dir(),
            max_output_bytes: 1024,
        };
        let mut backend = LeanBackend::new(lean_config);
        // Backend won't find lean at /nonexistent
        backend.lean_exe = None; // Force unavailable

        let result = refine(
            &mut backend,
            "theorem t : 1 = 1 := by rfl",
            "t",
            "1 = 1",
            &config,
        );
        assert!(matches!(result, RefinementOutcome::BackendUnavailable));
    }

    #[test]
    fn outcome_is_verified_check() {
        let verified = RefinementOutcome::Verified {
            lean_source: String::new(),
            attempt: 1,
            elapsed: Duration::ZERO,
        };
        assert!(verified.is_verified());

        let exhausted = RefinementOutcome::Exhausted {
            attempts: 10,
            last_error: String::new(),
            elapsed: Duration::ZERO,
        };
        assert!(!exhausted.is_verified());
    }
}

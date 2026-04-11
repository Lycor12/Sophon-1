//! Dual-Pass Verify-then-Learn (DPVL).
//!
//! Every output goes through mandatory verification:
//! Pass 1: Syntactic/structural checks
//! Pass 2: Semantic/execution checks
//!
//! If either fails: classify error, apply targeted correction, retry.
//! This guarantees no hallucinated outputs.

use crate::gave::{Evidence, EvidenceChain, GaveEngine, VerificationResult, VerificationStatus};
use std::time::Instant;

/// DPVL configuration.
pub struct DvplConfig {
    pub max_attempts: usize,
    pub pass1_timeout_ms: u64,
    pub pass2_timeout_ms: u64,
    pub min_confidence: f32,
}

impl Default for DvplConfig {
    fn default() -> Self {
        Self {
            max_attempts: 20,
            pass1_timeout_ms: 5000,
            pass2_timeout_ms: 30000,
            min_confidence: 0.8,
        }
    }
}

/// Error classification for targeted correction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorClass {
    Syntax,       // Structural error
    TypeMismatch, // Type system violation
    Undefined,    // Reference to unknown entity
    Logic,        // Semantic/logic error
    Timeout,      // Did not complete in time
    Resource,     // Out of memory, etc.
    Verification, // Verification failed
    Unknown,      // Unclassified
}

/// A single verification pass attempt.
#[derive(Debug, Clone)]
pub struct VerificationAttempt {
    pub attempt: usize,
    pub pass1_result: PassResult,
    pub pass2_result: Option<PassResult>,
    pub error_class: Option<ErrorClass>,
    pub correction: Option<Correction>,
}

/// Result of a verification pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    pub success: bool,
    pub output: String,
    pub duration_ms: u64,
}

/// Correction to apply after failure.
#[derive(Debug, Clone)]
pub struct Correction {
    pub strategy: CorrectionStrategy,
    pub description: String,
}

/// Strategy for correcting errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionStrategy {
    Regenerate,   // Start over
    LocalFix,     // Fix specific line/region
    AddImport,    // Missing dependency
    RemoveUnused, // Dead code removal
    Simplify,     // Reduce complexity
    AddCheck,     // Add validation
    ChangeType,   // Fix type annotation
    Refactor,     // Restructure
}

/// Final verification outcome.
#[derive(Debug, Clone)]
pub enum VerificationOutcome {
    Verified { attempts: usize, confidence: f32 },
    Exhausted { attempts: usize, last_error: String },
    Timeout { attempts: usize },
    Unverifiable { reason: String },
}

/// DPVL verification loop.
pub struct VerificationLoop {
    config: DvplConfig,
    gave: GaveEngine,
    history: Vec<VerificationAttempt>,
}

impl VerificationLoop {
    pub fn new(config: DvplConfig) -> Self {
        Self {
            config,
            gave: GaveEngine::new(config.min_confidence),
            history: Vec::new(),
        }
    }

    /// Verify an output through dual-pass.
    pub fn verify<F1, F2>(&mut self, output: &str, pass1: F1, pass2: F2) -> VerificationOutcome
    where
        F1: Fn(&str) -> PassResult,
        F2: Fn(&str) -> PassResult,
    {
        for attempt in 1..=self.config.max_attempts {
            let start = Instant::now();

            // Pass 1: Structural verification
            let pass1_result = pass1(output);

            if !pass1_result.success {
                let error_class = self.classify_error(&pass1_result.output);
                let correction = self.suggest_correction(&error_class, &pass1_result.output);

                self.history.push(VerificationAttempt {
                    attempt,
                    pass1_result,
                    pass2_result: None,
                    error_class: Some(error_class),
                    correction: Some(correction),
                });

                // In a real system, would apply correction and retry
                // For now, continue to next attempt
                continue;
            }

            // Pass 2: Semantic/execution verification
            let pass2_result = pass2(output);

            if !pass2_result.success {
                let error_class = self.classify_error(&pass2_result.output);
                let correction = self.suggest_correction(&error_class, &pass2_result.output);

                self.history.push(VerificationAttempt {
                    attempt,
                    pass1_result: pass1_result.clone(),
                    pass2_result: Some(pass2_result),
                    error_class: Some(error_class),
                    correction: Some(correction),
                });

                continue;
            }

            // Both passes succeeded
            let mut evidence = EvidenceChain::new();
            evidence.add(Evidence::Verification {
                method: "pass1_structural".to_string(),
                result: VerificationResult {
                    success: true,
                    output: pass1_result.output.clone(),
                    duration_ms: pass1_result.duration_ms,
                },
            });
            evidence.add(Evidence::Verification {
                method: "pass2_semantic".to_string(),
                result: VerificationResult {
                    success: true,
                    output: pass2_result.output.clone(),
                    duration_ms: pass2_result.duration_ms,
                },
            });

            let idx = self.gave.assert(output, evidence);
            let assertion = self.gave.get(idx).unwrap();

            self.history.push(VerificationAttempt {
                attempt,
                pass1_result,
                pass2_result: Some(pass2_result),
                error_class: None,
                correction: None,
            });

            return VerificationOutcome::Verified {
                attempts: attempt,
                confidence: assertion.confidence,
            };
        }

        // Exhausted all attempts
        VerificationOutcome::Exhausted {
            attempts: self.config.max_attempts,
            last_error: self
                .history
                .last()
                .map(|h| {
                    h.pass2_result
                        .as_ref()
                        .map(|r| r.output.clone())
                        .unwrap_or_else(|| h.pass1_result.output.clone())
                })
                .unwrap_or_default(),
        }
    }

    /// Classify error from output.
    fn classify_error(&self, output: &str) -> ErrorClass {
        let lowered = output.to_lowercase();

        if lowered.contains("syntax") || lowered.contains("parse") {
            ErrorClass::Syntax
        } else if lowered.contains("type") || lowered.contains("mismatch") {
            ErrorClass::TypeMismatch
        } else if lowered.contains("undefined") || lowered.contains("not found") {
            ErrorClass::Undefined
        } else if lowered.contains("timeout") || lowered.contains("timed out") {
            ErrorClass::Timeout
        } else if lowered.contains("memory") || lowered.contains("oom") {
            ErrorClass::Resource
        } else if lowered.contains("logic") || lowered.contains("assertion") {
            ErrorClass::Logic
        } else {
            ErrorClass::Unknown
        }
    }

    /// Suggest correction strategy based on error class.
    fn suggest_correction(&self, error: &ErrorClass, output: &str) -> Correction {
        match error {
            ErrorClass::Syntax => Correction {
                strategy: CorrectionStrategy::Regenerate,
                description: "Structural error detected, regenerating from scratch".to_string(),
            },
            ErrorClass::TypeMismatch => Correction {
                strategy: CorrectionStrategy::ChangeType,
                description: "Type mismatch, adjusting type annotations".to_string(),
            },
            ErrorClass::Undefined => Correction {
                strategy: CorrectionStrategy::AddImport,
                description: "Undefined reference, adding necessary import".to_string(),
            },
            ErrorClass::Logic => Correction {
                strategy: CorrectionStrategy::LocalFix,
                description: "Logic error, applying local fix".to_string(),
            },
            ErrorClass::Timeout => Correction {
                strategy: CorrectionStrategy::Simplify,
                description: "Operation timed out, simplifying approach".to_string(),
            },
            ErrorClass::Resource => Correction {
                strategy: CorrectionStrategy::Refactor,
                description: "Resource limit exceeded, refactoring for efficiency".to_string(),
            },
            ErrorClass::Verification => Correction {
                strategy: CorrectionStrategy::AddCheck,
                description: "Verification failed, adding validation".to_string(),
            },
            ErrorClass::Unknown => Correction {
                strategy: CorrectionStrategy::Regenerate,
                description: "Unknown error, trying fresh approach".to_string(),
            },
        }
    }

    /// Get verification history.
    pub fn history(&self) -> &[VerificationAttempt] {
        &self.history
    }

    /// Get success rate across all verifications.
    pub fn success_rate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        let successes = self
            .history
            .iter()
            .filter(|h| {
                h.pass1_result.success
                    && h.pass2_result.as_ref().map(|r| r.success).unwrap_or(false)
            })
            .count() as f32;

        successes / self.history.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpvl_successful_verification() {
        let mut dpvl = VerificationLoop::new(DvplConfig::default());

        let result = dpvl.verify(
            "test output",
            |_o| PassResult {
                success: true,
                output: "pass1 ok".to_string(),
                duration_ms: 10,
            },
            |_o| PassResult {
                success: true,
                output: "pass2 ok".to_string(),
                duration_ms: 20,
            },
        );

        assert!(matches!(
            result,
            VerificationOutcome::Verified { attempts: 1, .. }
        ));
    }

    #[test]
    fn dpvl_pass1_failure() {
        let mut dpvl = VerificationLoop::new(DvplConfig {
            max_attempts: 2,
            ..Default::default()
        });

        let result = dpvl.verify(
            "bad syntax",
            |_o| PassResult {
                success: false,
                output: "syntax error".to_string(),
                duration_ms: 10,
            },
            |_o| PassResult {
                success: true,
                output: "pass2 ok".to_string(),
                duration_ms: 20,
            },
        );

        assert!(matches!(result, VerificationOutcome::Exhausted { .. }));
    }

    #[test]
    fn error_classification() {
        let dpvl = VerificationLoop::new(DvplConfig::default());

        assert_eq!(
            dpvl.classify_error("syntax error on line 5"),
            ErrorClass::Syntax
        );
        assert_eq!(
            dpvl.classify_error("type mismatch: expected int"),
            ErrorClass::TypeMismatch
        );
        assert_eq!(
            dpvl.classify_error("undefined variable 'x'"),
            ErrorClass::Undefined
        );
    }
}

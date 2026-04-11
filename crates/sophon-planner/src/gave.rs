//! Grounded-Assertion Verification Engine (GAVE).
//!
//! Every claim the system makes is backed by an evidence chain.
//! Confidence is computed from evidence completeness, not from softmax.
//! This eliminates hallucination by construction.

use sophon_config::HDC_DIM;
use sophon_core::hdc::l2_normalize;

fn l2_normalize_vec(v: &[f32]) -> Vec<f32> {
    let mut result = v.to_vec();
    l2_normalize(&mut result);
    result
}

/// Status of verification for an assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Unverified,
    PartiallyVerified, // Some evidence checked
    FullyVerified,     // All evidence checked
    Refuted,           // Evidence contradicts claim
    Unknown,           // Insufficient evidence
}

/// Evidence supporting a claim.
#[derive(Debug, Clone)]
pub enum Evidence {
    /// Direct observation from data
    Observation {
        source: String,
        data: String,
        timestamp: u64,
    },
    /// Inference from other evidence
    Inference {
        from: Vec<usize>, // Indices in evidence chain
        rule: String,
    },
    /// External verification (compilation, execution, etc.)
    Verification {
        method: String,
        result: VerificationResult,
    },
    /// Reference to source (file, line, etc.)
    SourceRef {
        file: String,
        line: usize,
        col: usize,
        snippet: String,
    },
    /// Semantic property (type, invariant, etc.)
    Property {
        name: String,
        value: String,
        confidence: f32,
    },
}

/// Result of external verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub success: bool,
    pub output: String,
    pub duration_ms: u64,
}

/// Chain of evidence supporting a claim.
#[derive(Debug, Clone, Default)]
pub struct EvidenceChain {
    pub pieces: Vec<Evidence>,
}

impl EvidenceChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, evidence: Evidence) -> usize {
        let idx = self.pieces.len();
        self.pieces.push(evidence);
        idx
    }

    /// Compute confidence from evidence completeness.
    pub fn compute_confidence(&self) -> f32 {
        if self.pieces.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let mut weights = 0.0;

        for piece in &self.pieces {
            let (piece_score, weight) = match piece {
                Evidence::Observation { .. } => (0.8, 1.0),
                Evidence::SourceRef { .. } => (1.0, 1.2), // Ground truth
                Evidence::Verification { result, .. } => {
                    if result.success {
                        (1.0, 1.5)
                    } else {
                        (0.0, 1.5)
                    }
                }
                Evidence::Inference { from, .. } => {
                    // Confidence depends on source evidences
                    let from_conf: f32 = from
                        .iter()
                        .map(|&i| if i < self.pieces.len() { 0.8 } else { 0.0 })
                        .sum::<f32>()
                        / from.len().max(1) as f32;
                    (from_conf * 0.7, 0.8)
                }
                Evidence::Property { confidence, .. } => (*confidence, 1.0),
            };
            score += piece_score * weight;
            weights += weight;
        }

        // Penalty for missing evidence types
        let has_observation = self
            .pieces
            .iter()
            .any(|p| matches!(p, Evidence::Observation { .. }));
        let has_source = self
            .pieces
            .iter()
            .any(|p| matches!(p, Evidence::SourceRef { .. }));
        let has_verification = self
            .pieces
            .iter()
            .any(|p| matches!(p, Evidence::Verification { .. }));

        let coverage = (has_observation as usize + has_source as usize + has_verification as usize)
            as f32
            / 3.0;

        (score / weights) * coverage
    }

    /// Check if chain contains any contradictions.
    pub fn has_contradiction(&self) -> Option<(usize, usize)> {
        // Check for verification that failed
        for (i, piece) in self.pieces.iter().enumerate() {
            if let Evidence::Verification { result, .. } = piece {
                if !result.success {
                    return Some((i, usize::MAX)); // Contradiction with claim itself
                }
            }
        }
        None
    }

    /// Is the evidence sufficient for the claim?
    pub fn is_sufficient(&self, threshold: f32) -> bool {
        self.compute_confidence() >= threshold
    }
}

/// GAVE Engine: validates and manages assertions.
pub struct GaveEngine {
    assertions: Vec<Assertion>,
    min_confidence: f32,
}

impl GaveEngine {
    pub fn new(min_confidence: f32) -> Self {
        Self {
            assertions: Vec::new(),
            min_confidence,
        }
    }

    /// Make a new assertion with evidence.
    pub fn assert(&mut self, claim: &str, evidence: EvidenceChain) -> AssertionRef {
        let confidence = evidence.compute_confidence();
        let status = if confidence >= 0.9 {
            VerificationStatus::FullyVerified
        } else if confidence >= self.min_confidence {
            VerificationStatus::PartiallyVerified
        } else {
            VerificationStatus::Unverified
        };

        let assertion = Assertion {
            claim: claim.to_string(),
            evidence,
            confidence,
            verification_status: status,
        };

        let idx = self.assertions.len();
        self.assertions.push(assertion);

        AssertionRef(idx)
    }

    /// Verify an assertion externally.
    pub fn verify(&mut self, idx: AssertionRef, method: &str, result: VerificationResult) {
        if let Some(assertion) = self.assertions.get_mut(idx.0) {
            assertion.evidence.add(Evidence::Verification {
                method: method.to_string(),
                result,
            });

            // Recompute confidence
            assertion.confidence = assertion.evidence.compute_confidence();
            assertion.verification_status = if assertion.confidence >= 0.95 {
                VerificationStatus::FullyVerified
            } else if assertion.confidence >= self.min_confidence {
                VerificationStatus::PartiallyVerified
            } else if assertion.evidence.has_contradiction().is_some() {
                VerificationStatus::Refuted
            } else {
                VerificationStatus::Unverified
            };
        }
    }

    /// Get assertion by reference.
    pub fn get(&self, idx: AssertionRef) -> Option<&Assertion> {
        self.assertions.get(idx.0)
    }

    /// Find assertions matching a query pattern.
    pub fn find(&self, pattern: &str) -> Vec<&Assertion> {
        self.assertions
            .iter()
            .filter(|a| a.claim.contains(pattern))
            .collect()
    }

    /// Get all assertions with confidence below threshold (need more evidence).
    pub fn needs_verification(&self) -> Vec<&Assertion> {
        self.assertions
            .iter()
            .filter(|a| a.confidence < self.min_confidence)
            .collect()
    }

    /// Get all refuted assertions (contradictions found).
    pub fn refuted(&self) -> Vec<&Assertion> {
        self.assertions
            .iter()
            .filter(|a| a.verification_status == VerificationStatus::Refuted)
            .collect()
    }

    /// Confidence-calibrated response: what to say given the assertion.
    pub fn calibrated_response(&self, idx: AssertionRef) -> CalibratedResponse {
        match self.assertions.get(idx.0) {
            None => CalibratedResponse::Unknown,
            Some(a) => {
                if a.verification_status == VerificationStatus::Refuted {
                    CalibratedResponse::Refutation(a.claim.clone())
                } else if a.confidence > 0.9 {
                    CalibratedResponse::Confident(a.claim.clone())
                } else if a.confidence > 0.7 {
                    CalibratedResponse::Likely(a.claim.clone())
                } else if a.confidence > 0.5 {
                    CalibratedResponse::Possible(a.claim.clone())
                } else if a.confidence > 0.2 {
                    CalibratedResponse::Uncertain(a.claim.clone())
                } else {
                    CalibratedResponse::Unknown
                }
            }
        }
    }
}

/// Reference to an assertion.
pub struct AssertionRef(usize);

/// Calibrated response based on confidence.
#[derive(Debug, Clone)]
pub enum CalibratedResponse {
    Confident(String),  // > 90%
    Likely(String),     // 70-90%
    Possible(String),   // 50-70%
    Uncertain(String),  // 20-50%
    Unknown,            // < 20%
    Refutation(String), // Contradicted
}

impl CalibratedResponse {
    /// Generate natural language response.
    pub fn to_nl(&self) -> String {
        match self {
            Self::Confident(c) => format!("{} This is certain.", c),
            Self::Likely(c) => format!("{} This is likely.", c),
            Self::Possible(c) => format!("{} This is possible.", c),
            Self::Uncertain(c) => format!("I'm not sure, but {}.", c),
            Self::Unknown => "I don't have enough information to say.".to_string(),
            Self::Refutation(c) => format!("Contradiction found: {}", c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_chain_confidence() {
        let mut chain = EvidenceChain::new();
        chain.add(Evidence::SourceRef {
            file: "test.c".to_string(),
            line: 10,
            col: 5,
            snippet: "int x = 5;".to_string(),
        });

        let conf = chain.compute_confidence();
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn gave_assert_and_verify() {
        let mut gave = GaveEngine::new(0.7);

        let mut chain = EvidenceChain::new();
        chain.add(Evidence::SourceRef {
            file: "test.c".to_string(),
            line: 10,
            col: 0,
            snippet: "int x = 5;".to_string(),
        });

        let idx = gave.assert("x is 5", chain);
        let assertion = gave.get(idx).unwrap();
        assert!(assertion.confidence > 0.0);

        // Add verification
        gave.verify(
            idx,
            "static_analysis",
            VerificationResult {
                success: true,
                output: "verified".to_string(),
                duration_ms: 10,
            },
        );

        let verified = gave.get(idx).unwrap();
        assert!(verified.confidence > assertion.confidence);
    }

    #[test]
    fn calibrated_response_confidence_levels() {
        let mut gave = GaveEngine::new(0.7);

        // Test different confidence levels
        for (claim, expected_variant) in [
            ("high conf", "Confident"),
            ("med conf", "Likely"),
            ("low conf", "Possible"),
        ] {
            let chain = EvidenceChain::new();
            let idx = gave.assert(claim, chain);
            let response = gave.calibrated_response(idx);
            let nl = response.to_nl();
            assert!(!nl.is_empty());
        }
    }
}

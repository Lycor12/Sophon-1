//! Self-error detection — Addendum D.
//!
//! Continuous internal validation loops that run before any output or action.
//! Detects contradictions, estimates uncertainty, checks state consistency,
//! and triggers refinement cycles automatically.
//!
//! # Novel technique: CSDL (Cascaded Self-Diagnostic Loop)
//!
//! Rather than a single validation check, CSDL cascades four diagnostic stages
//! in sequence, each gating the next. This ensures cheap checks catch obvious
//! failures early, while more expensive checks (contradiction, uncertainty)
//! only run on outputs that pass basic sanity:
//!
//! ```text
//! Stage 1: Numerical sanity (NaN, Inf, out-of-range)
//! Stage 2: Distributional consistency (entropy bounds)
//! Stage 3: Contradiction detection (pairwise confidence inversion)
//! Stage 4: Uncertainty gating (max-softmax calibration)
//! ```
//!
//! If any stage fails, the cascade halts and reports the fault without
//! running more expensive downstream checks.

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the self-diagnostic system.
#[derive(Debug, Clone)]
pub struct DiagnosticConfig {
    /// Minimum allowed entropy for a logit distribution (bits).
    /// Below this, the model is suspiciously confident.
    pub min_entropy: f32,
    /// Maximum allowed entropy (bits). Above this, the model is guessing.
    pub max_entropy: f32,
    /// Minimum max-softmax probability to consider output "confident enough".
    pub confidence_threshold: f32,
    /// Number of top predictions to check for contradictions.
    pub contradiction_window: usize,
    /// Maximum absolute value for any logit (numerical sanity).
    pub max_logit_abs: f32,
    /// If true, a failed diagnostic triggers automatic refinement marking.
    pub auto_refine: bool,
}

impl DiagnosticConfig {
    /// Default configuration for byte-level model (VOCAB_SIZE=256).
    pub fn default_byte_model() -> Self {
        Self {
            min_entropy: 0.1,           // Suspiciously low: model "knows" too much
            max_entropy: 7.5,           // log2(256) ≈ 8.0; close to uniform
            confidence_threshold: 0.15, // At least 15% on top prediction
            contradiction_window: 5,
            max_logit_abs: 100.0,
            auto_refine: true,
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticFault
// ---------------------------------------------------------------------------

/// A specific fault detected by the diagnostic pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticFault {
    /// Stage 1: NaN or Inf detected in logits.
    NumericalNaN { position: usize },
    /// Stage 1: Logit value exceeds absolute bound.
    NumericalOverflow { position: usize, value: f32 },
    /// Stage 2: Output entropy below minimum (over-confident).
    EntropyTooLow { entropy: f32, min: f32 },
    /// Stage 2: Output entropy above maximum (near-random).
    EntropyTooHigh { entropy: f32, max: f32 },
    /// Stage 3: Contradiction — token ranked high in consecutive steps
    /// with inverted confidence (was top-k, then bottom-k, or vice versa).
    ContradictionDetected {
        token: usize,
        rank_before: usize,
        rank_after: usize,
    },
    /// Stage 4: Max softmax probability below threshold.
    UncertainOutput { max_prob: f32, threshold: f32 },
    /// Input validation: empty logits.
    EmptyInput,
}

impl core::fmt::Display for DiagnosticFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NumericalNaN { position } => {
                write!(f, "DIAG: NaN/Inf at position {position}")
            }
            Self::NumericalOverflow { position, value } => {
                write!(f, "DIAG: overflow at position {position}, value={value}")
            }
            Self::EntropyTooLow { entropy, min } => {
                write!(f, "DIAG: entropy {entropy:.3} < min {min:.3}")
            }
            Self::EntropyTooHigh { entropy, max } => {
                write!(f, "DIAG: entropy {entropy:.3} > max {max:.3}")
            }
            Self::ContradictionDetected {
                token,
                rank_before,
                rank_after,
            } => {
                write!(
                    f,
                    "DIAG: contradiction on token {token}: rank {rank_before} -> {rank_after}"
                )
            }
            Self::UncertainOutput {
                max_prob,
                threshold,
            } => {
                write!(
                    f,
                    "DIAG: uncertain output, max_prob={max_prob:.4} < threshold={threshold:.4}"
                )
            }
            Self::EmptyInput => write!(f, "DIAG: empty input"),
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticResult
// ---------------------------------------------------------------------------

/// Result of a full diagnostic cascade run.
#[derive(Debug, Clone)]
pub struct DiagnosticResult {
    /// Whether all stages passed.
    pub passed: bool,
    /// Faults found (empty if passed).
    pub faults: Vec<DiagnosticFault>,
    /// The stage at which the cascade halted (0 if all passed, 1-4 otherwise).
    pub halted_at_stage: u8,
    /// Computed entropy of the output distribution (if stage 1 passed).
    pub entropy: Option<f32>,
    /// Max softmax probability (if stages 1-2 passed).
    pub max_confidence: Option<f32>,
    /// Whether automatic refinement was triggered.
    pub refinement_triggered: bool,
}

impl DiagnosticResult {
    fn pass(entropy: f32, max_conf: f32) -> Self {
        Self {
            passed: true,
            faults: Vec::new(),
            halted_at_stage: 0,
            entropy: Some(entropy),
            max_confidence: Some(max_conf),
            refinement_triggered: false,
        }
    }

    fn fail(stage: u8, faults: Vec<DiagnosticFault>, auto_refine: bool) -> Self {
        Self {
            passed: false,
            faults,
            halted_at_stage: stage,
            entropy: None,
            max_confidence: None,
            refinement_triggered: auto_refine,
        }
    }

    fn fail_with_entropy(
        stage: u8,
        faults: Vec<DiagnosticFault>,
        entropy: f32,
        auto_refine: bool,
    ) -> Self {
        Self {
            passed: false,
            faults,
            halted_at_stage: stage,
            entropy: Some(entropy),
            max_confidence: None,
            refinement_triggered: auto_refine,
        }
    }
}

// ---------------------------------------------------------------------------
// SelfDiagnostic — CSDL
// ---------------------------------------------------------------------------

/// The cascaded self-diagnostic system.
///
/// Maintains a history of recent output distributions for contradiction
/// detection across sequential outputs.
pub struct SelfDiagnostic {
    config: DiagnosticConfig,
    /// Recent top-K rankings: Vec<(token_id, rank)> for each recent step.
    recent_rankings: Vec<Vec<(usize, usize)>>,
    /// Maximum history length.
    max_history: usize,
    /// Total checks performed.
    total_checks: u64,
    /// Total faults detected.
    total_faults: u64,
}

impl SelfDiagnostic {
    pub fn new(config: DiagnosticConfig) -> Self {
        Self {
            max_history: config.contradiction_window,
            config,
            recent_rankings: Vec::new(),
            total_checks: 0,
            total_faults: 0,
        }
    }

    /// Run the full CSDL cascade on a logit vector.
    ///
    /// This is the main entry point. Call before every output or action.
    pub fn check(&mut self, logits: &[f32]) -> DiagnosticResult {
        self.total_checks += 1;

        if logits.is_empty() {
            self.total_faults += 1;
            return DiagnosticResult::fail(
                1,
                vec![DiagnosticFault::EmptyInput],
                self.config.auto_refine,
            );
        }

        // ---- Stage 1: Numerical sanity ----
        if let Some(faults) = self.stage_numerical(logits) {
            self.total_faults += 1;
            return DiagnosticResult::fail(1, faults, self.config.auto_refine);
        }

        // ---- Stage 2: Distributional consistency ----
        let (entropy, softmax) = self.compute_entropy_and_softmax(logits);
        if let Some(faults) = self.stage_distributional(entropy) {
            self.total_faults += 1;
            return DiagnosticResult::fail_with_entropy(
                2,
                faults,
                entropy,
                self.config.auto_refine,
            );
        }

        // ---- Stage 3: Contradiction detection ----
        let current_ranking = self.extract_ranking(&softmax);
        if let Some(faults) = self.stage_contradiction(&current_ranking) {
            self.total_faults += 1;
            self.push_ranking(current_ranking);
            return DiagnosticResult::fail_with_entropy(
                3,
                faults,
                entropy,
                self.config.auto_refine,
            );
        }
        self.push_ranking(current_ranking);

        // ---- Stage 4: Uncertainty gating ----
        let max_prob = softmax.iter().cloned().fold(0.0f32, f32::max);
        if let Some(faults) = self.stage_uncertainty(max_prob) {
            self.total_faults += 1;
            return DiagnosticResult {
                passed: false,
                faults,
                halted_at_stage: 4,
                entropy: Some(entropy),
                max_confidence: Some(max_prob),
                refinement_triggered: self.config.auto_refine,
            };
        }

        DiagnosticResult::pass(entropy, max_prob)
    }

    /// Stage 1: Check for NaN, Inf, and out-of-range values.
    fn stage_numerical(&self, logits: &[f32]) -> Option<Vec<DiagnosticFault>> {
        let mut faults = Vec::new();
        for (i, &v) in logits.iter().enumerate() {
            if !v.is_finite() {
                faults.push(DiagnosticFault::NumericalNaN { position: i });
            } else if v.abs() > self.config.max_logit_abs {
                faults.push(DiagnosticFault::NumericalOverflow {
                    position: i,
                    value: v,
                });
            }
        }
        if faults.is_empty() {
            None
        } else {
            Some(faults)
        }
    }

    /// Compute entropy (in bits) and softmax of a logit vector.
    fn compute_entropy_and_softmax(&self, logits: &[f32]) -> (f32, Vec<f32>) {
        // Numerically stable: subtract max
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut softmax = vec![0.0f32; logits.len()];
        let mut sum_exp = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            let e = (v - max_val).exp();
            softmax[i] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        let mut entropy = 0.0f32;
        for p in softmax.iter_mut() {
            *p *= inv_sum;
            if *p > 1e-12 {
                entropy -= *p * p.log2();
            }
        }
        (entropy, softmax)
    }

    /// Stage 2: Check entropy bounds.
    fn stage_distributional(&self, entropy: f32) -> Option<Vec<DiagnosticFault>> {
        if entropy < self.config.min_entropy {
            Some(vec![DiagnosticFault::EntropyTooLow {
                entropy,
                min: self.config.min_entropy,
            }])
        } else if entropy > self.config.max_entropy {
            Some(vec![DiagnosticFault::EntropyTooHigh {
                entropy,
                max: self.config.max_entropy,
            }])
        } else {
            None
        }
    }

    /// Extract top-K ranking from softmax probabilities.
    fn extract_ranking(&self, softmax: &[f32]) -> Vec<(usize, usize)> {
        let k = self.config.contradiction_window.min(softmax.len());
        // Collect (index, probability) and sort by probability descending.
        let mut indexed: Vec<(usize, f32)> = softmax.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Return (token_id, rank) for top-K tokens
        indexed
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, &(tok, _))| (tok, rank))
            .collect()
    }

    /// Stage 3: Detect contradictions by comparing current ranking to recent history.
    ///
    /// A contradiction is when a token was in the top-K at a recent step but is
    /// now ranked far outside, or vice versa. This indicates the model's confidence
    /// on that token has dramatically inverted.
    fn stage_contradiction(&self, current: &[(usize, usize)]) -> Option<Vec<DiagnosticFault>> {
        if self.recent_rankings.is_empty() {
            return None;
        }

        let mut faults = Vec::new();
        let k = self.config.contradiction_window;

        // Check against the most recent ranking only (one-step contradiction)
        let prev = &self.recent_rankings[self.recent_rankings.len() - 1];
        let current_tokens: std::collections::HashMap<usize, usize> =
            current.iter().cloned().collect();
        let _prev_tokens: std::collections::HashMap<usize, usize> = prev.iter().cloned().collect();

        for (&(tok, prev_rank), _) in prev.iter().zip(0..k) {
            if let Some(&curr_rank) = current_tokens.get(&tok) {
                // Token exists in both rankings — check for dramatic swap
                let rank_diff = if curr_rank > prev_rank {
                    curr_rank - prev_rank
                } else {
                    prev_rank - curr_rank
                };
                // If rank changed by more than k, it's a contradiction
                if rank_diff >= k {
                    faults.push(DiagnosticFault::ContradictionDetected {
                        token: tok,
                        rank_before: prev_rank,
                        rank_after: curr_rank,
                    });
                }
            }
        }

        // Also check: tokens that were in top-k before but not at all in current top-k
        // This counts as a rank change from prev_rank to k (at minimum)
        for &(tok, prev_rank) in prev.iter() {
            if !current_tokens.contains_key(&tok) && prev_rank < k.saturating_sub(1) {
                // Was ranked high, now not even in top-K
                // This is suspicious but only flag if the window is large enough
                if k >= 3 {
                    faults.push(DiagnosticFault::ContradictionDetected {
                        token: tok,
                        rank_before: prev_rank,
                        rank_after: k, // At least k
                    });
                }
            }
        }

        if faults.is_empty() {
            None
        } else {
            Some(faults)
        }
    }

    /// Stage 4: Check max softmax probability against threshold.
    fn stage_uncertainty(&self, max_prob: f32) -> Option<Vec<DiagnosticFault>> {
        if max_prob < self.config.confidence_threshold {
            Some(vec![DiagnosticFault::UncertainOutput {
                max_prob,
                threshold: self.config.confidence_threshold,
            }])
        } else {
            None
        }
    }

    /// Push a ranking into history, maintaining bounded size.
    fn push_ranking(&mut self, ranking: Vec<(usize, usize)>) {
        self.recent_rankings.push(ranking);
        if self.recent_rankings.len() > self.max_history {
            self.recent_rankings.remove(0);
        }
    }

    /// Clear history (e.g. at start of new sequence).
    pub fn reset_history(&mut self) {
        self.recent_rankings.clear();
    }

    /// Fault rate: fraction of checks that detected faults.
    pub fn fault_rate(&self) -> f32 {
        if self.total_checks == 0 {
            0.0
        } else {
            self.total_faults as f32 / self.total_checks as f32
        }
    }

    /// Total diagnostics run.
    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }

    /// Total faults detected.
    pub fn total_faults(&self) -> u64 {
        self.total_faults
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diagnostic() -> SelfDiagnostic {
        SelfDiagnostic::new(DiagnosticConfig::default_byte_model())
    }

    /// Helper: create logits that softmax to roughly uniform.
    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    /// Helper: create logits with one dominant value.
    fn peaked_logits(n: usize, peak_idx: usize, peak_val: f32) -> Vec<f32> {
        let mut logits = vec![0.0f32; n];
        logits[peak_idx] = peak_val;
        logits
    }

    #[test]
    fn stage1_nan_detected() {
        let mut diag = make_diagnostic();
        let mut logits = vec![1.0; 256];
        logits[42] = f32::NAN;
        let result = diag.check(&logits);
        assert!(!result.passed);
        assert_eq!(result.halted_at_stage, 1);
        assert!(matches!(
            result.faults[0],
            DiagnosticFault::NumericalNaN { position: 42 }
        ));
    }

    #[test]
    fn stage1_overflow_detected() {
        let mut diag = make_diagnostic();
        let mut logits = vec![1.0; 256];
        logits[10] = 200.0;
        let result = diag.check(&logits);
        assert!(!result.passed);
        assert_eq!(result.halted_at_stage, 1);
        assert!(matches!(
            result.faults[0],
            DiagnosticFault::NumericalOverflow { position: 10, .. }
        ));
    }

    #[test]
    fn stage2_entropy_too_high_for_uniform() {
        let mut diag = make_diagnostic();
        let logits = uniform_logits(256);
        let result = diag.check(&logits);
        // Uniform over 256 gives entropy = 8.0 bits, max is 7.5
        assert!(!result.passed);
        assert_eq!(result.halted_at_stage, 2);
        assert!(matches!(
            result.faults[0],
            DiagnosticFault::EntropyTooHigh { .. }
        ));
    }

    #[test]
    fn stage2_entropy_too_low_for_spike() {
        let mut diag = make_diagnostic();
        // Very peaked distribution
        let logits = peaked_logits(256, 0, 50.0);
        let result = diag.check(&logits);
        assert!(!result.passed);
        assert_eq!(result.halted_at_stage, 2);
        assert!(matches!(
            result.faults[0],
            DiagnosticFault::EntropyTooLow { .. }
        ));
    }

    #[test]
    fn reasonable_distribution_passes() {
        let mut diag = make_diagnostic();
        // Moderately peaked logits — entropy ~ 4-6 bits
        let mut logits = vec![0.0f32; 256];
        for i in 0..256 {
            logits[i] = -((i as f32) / 30.0);
        }
        logits[0] = 3.0; // Stronger peak to ensure max_prob > 0.15
        let result = diag.check(&logits);
        assert!(result.passed, "Faults: {:?}", result.faults);
        assert!(result.entropy.unwrap() > 0.1);
        assert!(result.max_confidence.unwrap() > 0.15);
    }

    #[test]
    fn stage4_uncertainty_detected() {
        let mut diag = SelfDiagnostic::new(DiagnosticConfig {
            confidence_threshold: 0.5, // High threshold
            max_entropy: 8.0,          // Allow uniform
            min_entropy: 0.0,          // Allow any
            ..DiagnosticConfig::default_byte_model()
        });
        // Slightly peaked but not 50% confident
        let mut logits = vec![0.0f32; 256];
        logits[0] = 1.0;
        let result = diag.check(&logits);
        assert!(!result.passed);
        assert_eq!(result.halted_at_stage, 4);
        assert!(matches!(
            result.faults[0],
            DiagnosticFault::UncertainOutput { .. }
        ));
    }

    #[test]
    fn empty_input_detected() {
        let mut diag = make_diagnostic();
        let result = diag.check(&[]);
        assert!(!result.passed);
        assert_eq!(result.halted_at_stage, 1);
        assert!(matches!(result.faults[0], DiagnosticFault::EmptyInput));
    }

    #[test]
    fn fault_rate_tracks() {
        let mut diag = make_diagnostic();

        // One failing check (uniform)
        let _ = diag.check(&uniform_logits(256));
        // One passing check (reasonable, well-peaked)
        let mut logits = vec![0.0f32; 256];
        for i in 0..256 {
            logits[i] = -((i as f32) / 30.0);
        }
        logits[0] = 3.0;
        let _ = diag.check(&logits);

        assert_eq!(diag.total_checks(), 2);
        assert_eq!(diag.total_faults(), 1);
        assert!((diag.fault_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn reset_clears_history() {
        let mut diag = make_diagnostic();
        let mut logits = vec![0.0f32; 256];
        for i in 0..256 {
            logits[i] = -((i as f32) / 50.0);
        }
        logits[0] = 2.0;
        let _ = diag.check(&logits);
        assert!(!diag.recent_rankings.is_empty());
        diag.reset_history();
        assert!(diag.recent_rankings.is_empty());
    }

    #[test]
    fn contradiction_detection_works() {
        let mut diag = SelfDiagnostic::new(DiagnosticConfig {
            min_entropy: 0.0,
            max_entropy: 8.0,
            confidence_threshold: 0.0, // Disable uncertainty gate
            contradiction_window: 5,
            max_logit_abs: 100.0,
            auto_refine: true,
        });

        // Step 1: token 0 is top-ranked
        let mut logits1 = vec![0.0f32; 10];
        logits1[0] = 5.0; // token 0 will be rank 0
        logits1[1] = 4.0;
        logits1[2] = 3.0;
        logits1[3] = 2.0;
        logits1[4] = 1.0;
        let r1 = diag.check(&logits1);
        assert!(r1.passed, "Step 1 should pass: {:?}", r1.faults);

        // Step 2: token 0 drops to bottom, token 9 rises
        let mut logits2 = vec![0.0f32; 10];
        logits2[0] = -5.0; // token 0 now last
        logits2[9] = 5.0;
        logits2[8] = 4.0;
        logits2[7] = 3.0;
        logits2[6] = 2.0;
        logits2[5] = 1.0;
        let r2 = diag.check(&logits2);
        // Token 0 was rank 0, now not in top-5 → contradiction
        assert!(!r2.passed, "Step 2 should detect contradiction");
        assert_eq!(r2.halted_at_stage, 3);
    }
}

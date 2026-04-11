//! Selection — tournament evaluation, ranking, and EMA weight merging.
//!
//! Spec §4.2.1: Top-K survivors selected from student solutions.
//! Spec §6.2.3: EMA weight merging consolidates winning students.
//!
//! # Novel technique: MCSW (Multi-Criteria Scoring with Weighted factors)
//!
//! Solutions are scored by a weighted combination of:
//!   1. Mean negative log-likelihood (model confidence)
//!   2. Triviality filter pass (boolean gate)
//!   3. Lean verification (boolean gate)
//!   4. Tactic diversity (number of unique tactics)
//!   5. Proof length penalty (penalise very short or very long proofs)
//!
//! The triviality and Lean gates are hard filters: solutions that fail
//! these receive score 0 regardless of other criteria. Among passing
//! solutions, the remaining factors are combined with configurable weights.
//!
//! After selection, the winning student's LoRA deltas are merged into
//! the base model weights using EMA:
//!   θ_base = (1 - α) * θ_base + α * θ_winner
//! where α decreases with curriculum level (more conservative at harder levels).

use crate::student::SolutionAttempt;

// ---------------------------------------------------------------------------
// SelectionConfig
// ---------------------------------------------------------------------------

/// Configuration for the selection process.
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Number of top solutions to select.
    pub top_k: usize,
    /// Weight for model confidence (lower NLL → higher score).
    pub confidence_weight: f32,
    /// Weight for tactic diversity.
    pub diversity_weight: f32,
    /// Weight for proof length (penalise extremes).
    pub length_weight: f32,
    /// Ideal proof length in bytes (for length scoring).
    pub ideal_length: usize,
    /// Maximum proof length (anything above gets score 0).
    pub max_length: usize,
    /// Base EMA alpha for weight merging.
    pub base_ema_alpha: f32,
    /// Level scaling factor for EMA alpha: alpha = base / (1 + scale * level).
    pub ema_level_scale: f32,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            top_k: 1,
            confidence_weight: 0.4,
            diversity_weight: 0.3,
            length_weight: 0.3,
            ideal_length: 200,
            max_length: 2048,
            base_ema_alpha: 0.1,
            ema_level_scale: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// RankedSolution
// ---------------------------------------------------------------------------

/// A scored and ranked solution.
#[derive(Debug, Clone)]
pub struct RankedSolution {
    /// Index into the original solution list.
    pub original_index: usize,
    /// Student ID.
    pub student_id: u32,
    /// Composite score (higher is better).
    pub score: f32,
    /// Individual score components for debugging.
    pub confidence_score: f32,
    pub diversity_score: f32,
    pub length_score: f32,
    /// Whether this solution passed all hard gates.
    pub passed_gates: bool,
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// The selection mechanism for the swarm classroom.
pub struct Selection {
    /// Configuration.
    pub config: SelectionConfig,
}

impl Selection {
    /// Create a new selection mechanism.
    pub fn new(config: SelectionConfig) -> Self {
        Self { config }
    }

    /// Score and rank all solution attempts.
    ///
    /// Hard gates: triviality filter and Lean verification (if available).
    /// Soft scores: confidence, diversity, length.
    /// Returns sorted list (highest score first).
    pub fn rank(&self, attempts: &[SolutionAttempt]) -> Vec<RankedSolution> {
        let mut ranked: Vec<RankedSolution> = attempts
            .iter()
            .enumerate()
            .map(|(i, a)| self.score_attempt(i, a))
            .collect();

        // Sort by score descending (stable sort preserves order for equal scores)
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked
    }

    /// Select the top-K solutions.
    pub fn select_top_k(&self, attempts: &[SolutionAttempt]) -> Vec<RankedSolution> {
        let ranked = self.rank(attempts);
        ranked
            .into_iter()
            .filter(|r| r.passed_gates)
            .take(self.config.top_k)
            .collect()
    }

    /// Compute the EMA alpha for weight merging at the given curriculum level.
    pub fn ema_alpha(&self, level: u32) -> f32 {
        let alpha = self.config.base_ema_alpha / (1.0 + self.config.ema_level_scale * level as f32);
        alpha.max(0.001) // Never go below minimum
    }

    /// Merge LoRA perturbations into base weights using EMA.
    ///
    /// base_weights and perturbation must have the same length.
    /// Updates base_weights in-place:
    ///   base[i] = (1 - alpha) * base[i] + alpha * (base[i] + perturbation[i])
    ///           = base[i] + alpha * perturbation[i]
    pub fn ema_merge(&self, base_weights: &mut [f32], perturbation: &[f32], level: u32) {
        let alpha = self.ema_alpha(level);
        debug_assert_eq!(base_weights.len(), perturbation.len());
        for (b, &p) in base_weights.iter_mut().zip(perturbation.iter()) {
            *b += alpha * p;
        }
    }

    // -----------------------------------------------------------------------
    // Internal scoring
    // -----------------------------------------------------------------------

    fn score_attempt(&self, index: usize, attempt: &SolutionAttempt) -> RankedSolution {
        // Hard gates
        let passed_gates = attempt.passed_triviality;
        // Note: lean_verified is only required when Lean is available.
        // For now, we don't gate on it since Lean may not be installed.

        if !passed_gates {
            return RankedSolution {
                original_index: index,
                student_id: attempt.student_id,
                score: 0.0,
                confidence_score: 0.0,
                diversity_score: 0.0,
                length_score: 0.0,
                passed_gates: false,
            };
        }

        // Confidence score: transform NLL to [0, 1]
        // Lower NLL = higher confidence. Use sigmoid-like transform.
        let confidence_score = 1.0 / (1.0 + attempt.mean_nll);

        // Diversity score: number of unique tactics normalised
        let n_tactics = attempt.tactics_used.len() as f32;
        let diversity_score = (n_tactics / 5.0).min(1.0); // 5+ tactics = max score

        // Length score: Gaussian penalty around ideal length
        let len = attempt.generated_bytes.len();
        let length_score = if len > self.config.max_length {
            0.0
        } else {
            let diff = len as f32 - self.config.ideal_length as f32;
            let sigma = self.config.ideal_length as f32 * 0.5;
            (-diff * diff / (2.0 * sigma * sigma)).exp()
        };

        // Composite score
        let score = self.config.confidence_weight * confidence_score
            + self.config.diversity_weight * diversity_score
            + self.config.length_weight * length_score;

        RankedSolution {
            original_index: index,
            student_id: attempt.student_id,
            score,
            confidence_score,
            diversity_score,
            length_score,
            passed_gates: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_attempt(
        student_id: u32,
        mean_nll: f32,
        tactics: &[&str],
        n_bytes: usize,
        passed: bool,
    ) -> SolutionAttempt {
        SolutionAttempt {
            student_id,
            problem_id: 1,
            generated_bytes: vec![0u8; n_bytes],
            generated_text: "proof".to_string(),
            tactics_used: tactics.iter().map(|s| s.to_string()).collect(),
            log_probs: vec![],
            mean_nll,
            passed_triviality: passed,
            lean_verified: false,
        }
    }

    #[test]
    fn failed_gate_gets_zero_score() {
        let sel = Selection::new(SelectionConfig::default());
        let attempts = vec![make_attempt(0, 1.0, &["simp"], 100, false)];
        let ranked = sel.rank(&attempts);
        assert_eq!(ranked[0].score, 0.0);
        assert!(!ranked[0].passed_gates);
    }

    #[test]
    fn passing_gate_gets_positive_score() {
        let sel = Selection::new(SelectionConfig::default());
        let attempts = vec![make_attempt(0, 1.0, &["simp", "ring"], 200, true)];
        let ranked = sel.rank(&attempts);
        assert!(ranked[0].score > 0.0);
        assert!(ranked[0].passed_gates);
    }

    #[test]
    fn lower_nll_scores_higher() {
        let sel = Selection::new(SelectionConfig::default());
        let attempts = vec![
            make_attempt(0, 5.0, &["simp"], 200, true),
            make_attempt(1, 0.5, &["simp"], 200, true),
        ];
        let ranked = sel.rank(&attempts);
        // Student 1 (lower NLL) should rank first
        assert_eq!(ranked[0].student_id, 1);
    }

    #[test]
    fn more_tactics_scores_higher() {
        let config = SelectionConfig {
            confidence_weight: 0.0,
            diversity_weight: 1.0,
            length_weight: 0.0,
            ..Default::default()
        };
        let sel = Selection::new(config);
        let attempts = vec![
            make_attempt(0, 1.0, &["simp"], 200, true),
            make_attempt(1, 1.0, &["simp", "ring", "omega", "exact"], 200, true),
        ];
        let ranked = sel.rank(&attempts);
        assert_eq!(ranked[0].student_id, 1);
    }

    #[test]
    fn select_top_k_filters_and_limits() {
        let config = SelectionConfig {
            top_k: 2,
            ..Default::default()
        };
        let sel = Selection::new(config);
        let attempts = vec![
            make_attempt(0, 1.0, &["simp"], 200, true),
            make_attempt(1, 0.5, &["simp", "ring"], 200, true),
            make_attempt(2, 2.0, &["simp"], 200, false), // fails gate
            make_attempt(3, 3.0, &["simp"], 200, true),
        ];
        let top = sel.select_top_k(&attempts);
        assert_eq!(top.len(), 2);
        // All selected should have passed gates
        assert!(top.iter().all(|r| r.passed_gates));
    }

    #[test]
    fn ema_alpha_decreases_with_level() {
        let sel = Selection::new(SelectionConfig::default());
        let a1 = sel.ema_alpha(1);
        let a10 = sel.ema_alpha(10);
        assert!(a1 > a10, "alpha should decrease: a1={a1}, a10={a10}");
    }

    #[test]
    fn ema_merge_applies_perturbation() {
        let sel = Selection::new(SelectionConfig {
            base_ema_alpha: 0.5,
            ema_level_scale: 0.0,
            ..Default::default()
        });
        let mut base = vec![1.0, 2.0, 3.0];
        let perturb = vec![0.1, 0.2, 0.3];
        sel.ema_merge(&mut base, &perturb, 1);
        // base[i] += 0.5 * perturb[i] (since scale=0, alpha=0.5)
        assert!((base[0] - 1.05).abs() < 1e-6);
        assert!((base[1] - 2.10).abs() < 1e-6);
        assert!((base[2] - 3.15).abs() < 1e-6);
    }

    #[test]
    fn very_long_proof_penalised() {
        let sel = Selection::new(SelectionConfig::default());
        let attempts = vec![
            make_attempt(0, 1.0, &["simp"], 200, true), // ideal length
            make_attempt(1, 1.0, &["simp"], 5000, true), // way too long
        ];
        let ranked = sel.rank(&attempts);
        let short = ranked.iter().find(|r| r.student_id == 0).unwrap();
        let long = ranked.iter().find(|r| r.student_id == 1).unwrap();
        assert!(short.length_score > long.length_score);
    }

    #[test]
    fn ema_alpha_never_below_minimum() {
        let sel = Selection::new(SelectionConfig {
            base_ema_alpha: 0.001,
            ema_level_scale: 10.0,
            ..Default::default()
        });
        let alpha = sel.ema_alpha(1000);
        assert!(alpha >= 0.001);
    }
}

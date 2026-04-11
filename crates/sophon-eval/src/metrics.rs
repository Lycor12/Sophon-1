//! AGI-relevant metrics beyond traditional ML.
//!
//! Tracks capabilities that matter for AGI:
//! - Calibration (predicted confidence vs actual accuracy)
//! - Verification rate (can it check its own work?)
//! - Recovery from errors
//! - Generalization across domains

/// Type of metric being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Accuracy,
    Calibration,
    Verification,
    Generalization,
    Efficiency,
    Robustness,
    Creativity,
    Consistency,
}

/// Complete AGI metrics.
#[derive(Debug, Clone, Default)]
pub struct AgiMetrics {
    pub accuracy: f32,
    pub calibration: CalibrationMetrics,
    pub verification_rate: f32,
    pub generalization: GeneralizationMetrics,
    pub efficiency: EfficiencyMetrics,
    pub robustness: RobustnessMetrics,
    pub creativity: CreativityMetrics,
    pub consistency: ConsistencyMetrics,
}

/// Calibration: how well does predicted confidence match actual accuracy?
#[derive(Debug, Clone, Default)]
pub struct CalibrationMetrics {
    pub expected_accuracy: f32,
    pub actual_accuracy: f32,
    pub calibration_error: f32, // |expected - actual|
}

/// Generalization: performance on unseen domains/tasks.
#[derive(Debug, Clone, Default)]
pub struct GeneralizationMetrics {
    pub seen_domain_score: f32,
    pub unseen_domain_score: f32,
    pub transfer_gap: f32, // seen - unseen
}

/// Efficiency: resource usage per task.
#[derive(Debug, Clone, Default)]
pub struct EfficiencyMetrics {
    pub tokens_per_task: f32,
    pub time_per_task_ms: f32,
    pub memory_per_task_mb: f32,
}

/// Robustness: performance under perturbation.
#[derive(Debug, Clone, Default)]
pub struct RobustnessMetrics {
    pub clean_score: f32,
    pub perturbed_score: f32,
    pub degradation: f32, // clean - perturbed
}

/// Creativity: generation of novel solutions.
#[derive(Debug, Clone, Default)]
pub struct CreativityMetrics {
    pub novelty_score: f32,
    pub diversity_score: f32,
    pub usefulness_score: f32,
}

/// Consistency: same answer for same question.
#[derive(Debug, Clone, Default)]
pub struct ConsistencyMetrics {
    pub consistency_rate: f32,
    pub num_trials: usize,
    pub disagreements: usize,
}

/// Latency statistics.
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub mean_ms: f32,
    pub max_ms: f32,
}

impl CalibrationMetrics {
    pub fn compute(expected: f32, actual: f32) -> Self {
        Self {
            expected_accuracy: expected,
            actual_accuracy: actual,
            calibration_error: (expected - actual).abs(),
        }
    }

    /// Is the model well-calibrated?
    pub fn is_calibrated(&self, threshold: f32) -> bool {
        self.calibration_error <= threshold
    }
}

impl GeneralizationMetrics {
    pub fn compute(seen: f32, unseen: f32) -> Self {
        Self {
            seen_domain_score: seen,
            unseen_domain_score: unseen,
            transfer_gap: seen - unseen,
        }
    }

    /// Is the model generalizing well?
    pub fn is_generalizing(&self, threshold: f32) -> bool {
        self.transfer_gap <= threshold
    }
}

impl RobustnessMetrics {
    pub fn compute(clean: f32, perturbed: f32) -> Self {
        Self {
            clean_score: clean,
            perturbed_score: perturbed,
            degradation: clean - perturbed,
        }
    }

    /// Is the model robust?
    pub fn is_robust(&self, threshold: f32) -> bool {
        self.degradation <= threshold
    }
}

impl CreativityMetrics {
    pub fn overall(&self) -> f32 {
        (self.novelty_score + self.diversity_score + self.usefulness_score) / 3.0
    }
}

impl ConsistencyMetrics {
    pub fn from_trials(answers: &[String]) -> Self {
        let num_trials = answers.len();
        if num_trials == 0 {
            return Self::default();
        }

        let first = &answers[0];
        let disagreements = answers.iter().skip(1).filter(|a| *a != first).count();
        let consistency_rate = (num_trials - disagreements) as f32 / num_trials as f32;

        Self {
            consistency_rate,
            num_trials,
            disagreements,
        }
    }
}

impl LatencyStats {
    pub fn compute(latencies: &[f32]) -> Self {
        if latencies.is_empty() {
            return Self::default();
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;
        let max = *sorted.last().unwrap();

        let p50_idx = (sorted.len() as f32 * 0.50) as usize;
        let p95_idx = (sorted.len() as f32 * 0.95) as usize;
        let p99_idx = (sorted.len() as f32 * 0.99) as usize;

        Self {
            p50_ms: sorted[p50_idx.min(sorted.len() - 1)],
            p95_ms: sorted[p95_idx.min(sorted.len() - 1)],
            p99_ms: sorted[p99_idx.min(sorted.len() - 1)],
            mean_ms: mean,
            max_ms: max,
        }
    }
}

impl AgiMetrics {
    /// Overall AGI score: weighted combination.
    pub fn overall(&self) -> f32 {
        0.20 * self.accuracy
            + 0.15 * (1.0 - self.calibration.calibration_error)
            + 0.15 * self.verification_rate
            + 0.15 * self.generalization.unseen_domain_score
            + 0.10 * (1.0 - self.robustness.degradation)
            + 0.10 * self.creativity.overall()
            + 0.10 * self.consistency.consistency_rate
            + 0.05 * (1.0 / (1.0 + self.efficiency.time_per_task_ms / 1000.0)) // Efficiency bonus
    }

    /// Format as report.
    pub fn report(&self) -> String {
        format!(
            "AGI Metrics Report\n\
            ==================\n\
            Overall Score: {:.1}%\n\
            - Accuracy: {:.1}%\n\
            - Calibration: {:.1}% (error: {:.1}%)\n\
            - Verification Rate: {:.1}%\n\
            - Generalization (unseen): {:.1}%\n\
            - Robustness: {:.1}% (degradation: {:.1}%)\n\
            - Creativity: {:.1}%\n\
            - Consistency: {:.1}%\n\
            - Efficiency: {:.0}ms/task\n",
            self.overall() * 100.0,
            self.accuracy * 100.0,
            (1.0 - self.calibration.calibration_error) * 100.0,
            self.calibration.calibration_error * 100.0,
            self.verification_rate * 100.0,
            self.generalization.unseen_domain_score * 100.0,
            (1.0 - self.robustness.degradation) * 100.0,
            self.robustness.degradation * 100.0,
            self.creativity.overall() * 100.0,
            self.consistency.consistency_rate * 100.0,
            self.efficiency.time_per_task_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_computation() {
        let cal = CalibrationMetrics::compute(0.9, 0.85);
        assert_eq!(cal.calibration_error, 0.05);
        assert!(cal.is_calibrated(0.1));
        assert!(!cal.is_calibrated(0.01));
    }

    #[test]
    fn generalization_computation() {
        let gen = GeneralizationMetrics::compute(0.9, 0.7);
        assert_eq!(gen.transfer_gap, 0.2);
    }

    #[test]
    fn robustness_computation() {
        let rob = RobustnessMetrics::compute(0.95, 0.85);
        assert_eq!(rob.degradation, 0.1);
    }

    #[test]
    fn consistency_from_trials() {
        let answers = vec!["A".to_string(), "A".to_string(), "B".to_string()];
        let cons = ConsistencyMetrics::from_trials(&answers);
        assert_eq!(cons.consistency_rate, 2.0 / 3.0);
        assert_eq!(cons.disagreements, 1);
    }

    #[test]
    fn latency_stats_computation() {
        let latencies = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = LatencyStats::compute(&latencies);
        assert_eq!(stats.p50_ms, 30.0);
        assert!(stats.mean_ms > 20.0 && stats.mean_ms < 40.0);
    }

    #[test]
    fn agi_overall_score() {
        let metrics = AgiMetrics {
            accuracy: 0.9,
            calibration: CalibrationMetrics::compute(0.9, 0.85),
            verification_rate: 0.8,
            generalization: GeneralizationMetrics::compute(0.9, 0.8),
            efficiency: EfficiencyMetrics {
                tokens_per_task: 100.0,
                time_per_task_ms: 500.0,
                memory_per_task_mb: 10.0,
            },
            robustness: RobustnessMetrics::compute(0.9, 0.85),
            creativity: CreativityMetrics {
                novelty_score: 0.7,
                diversity_score: 0.6,
                usefulness_score: 0.8,
            },
            consistency: ConsistencyMetrics {
                consistency_rate: 0.95,
                num_trials: 10,
                disagreements: 0,
            },
        };

        let score = metrics.overall();
        assert!(score > 0.0 && score <= 1.0);
    }
}

//! Alignment verification — Section 6.3.
//!
//! Monitors parameter drift from a frozen anchor checkpoint, tracks
//! performance on core tasks via an EMA, and triggers automatic rollback
//! when either constraint is violated.
//!
//! # Novel technique: EMAS (Exponential-Moving Alignment Score)
//!
//! Standard alignment monitoring compares against a fixed baseline. EMAS
//! uses a dual-track system:
//!
//! 1. **Parameter drift**: L2 distance ||θ - θ₀|| / ||θ₀|| normalised
//!    against the anchor checkpoint. Spec requires < ε = 0.1 (10%).
//!
//! 2. **Performance tracking**: EMA of per-iteration scores on a core
//!    validation set. If the EMA drops below (1 - δ) × peak_ema, a
//!    rollback is triggered. Spec requires δ = 0.05 (5% drop).
//!
//! The combination ensures both that the parameters haven't drifted too
//! far structurally AND that task performance hasn't degraded. Either
//! violation alone triggers intervention.
//!
//! ```text
//! Every `check_interval` iterations:
//!   drift = ||θ - θ_anchor|| / ||θ_anchor||
//!   if drift > epsilon: → rollback
//!   if ema_score < (1 - delta) * peak_ema: → rollback
//!   else: → continue
//! ```

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the alignment monitor.
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Maximum allowed relative parameter drift (spec: 0.1 = 10%).
    pub epsilon: f32,
    /// Maximum allowed relative performance drop (spec: 0.05 = 5%).
    pub delta: f32,
    /// How often to check (spec: every 1000 iterations).
    pub check_interval: u64,
    /// EMA decay factor for performance tracking.
    pub ema_decay: f32,
    /// Minimum number of checks before rollback is armed (warm-up period).
    pub warmup_checks: u64,
}

impl AlignmentConfig {
    /// Default from spec §6.3.1.
    pub fn from_spec() -> Self {
        Self {
            epsilon: 0.1,
            delta: 0.05,
            check_interval: 1000,
            ema_decay: 0.99,
            warmup_checks: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// AlignmentStatus
// ---------------------------------------------------------------------------

/// Result of an alignment check.
#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentStatus {
    /// Everything within bounds.
    Healthy {
        drift: f32,
        ema_score: f32,
        peak_score: f32,
    },
    /// Parameter drift exceeds epsilon — rollback recommended.
    DriftViolation { drift: f32, epsilon: f32 },
    /// Performance degraded by more than delta — rollback recommended.
    PerformanceDrop {
        ema_score: f32,
        peak_score: f32,
        delta: f32,
    },
    /// Both violations simultaneously.
    DualViolation {
        drift: f32,
        epsilon: f32,
        ema_score: f32,
        peak_score: f32,
        delta: f32,
    },
    /// Still in warm-up — no decision yet.
    WarmingUp { checks_remaining: u64 },
    /// Not time to check yet (between intervals).
    NotDue,
}

impl AlignmentStatus {
    /// Whether a rollback is recommended.
    pub fn needs_rollback(&self) -> bool {
        matches!(
            self,
            Self::DriftViolation { .. } | Self::PerformanceDrop { .. } | Self::DualViolation { .. }
        )
    }

    /// Whether the check was actually performed (vs skipped or warming up).
    pub fn was_checked(&self) -> bool {
        !matches!(self, Self::NotDue | Self::WarmingUp { .. })
    }
}

impl core::fmt::Display for AlignmentStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Healthy {
                drift,
                ema_score,
                peak_score,
            } => {
                write!(
                    f,
                    "ALIGNED: drift={drift:.4}, ema={ema_score:.4}, peak={peak_score:.4}"
                )
            }
            Self::DriftViolation { drift, epsilon } => {
                write!(f, "DRIFT VIOLATION: {drift:.4} > ε={epsilon:.4}")
            }
            Self::PerformanceDrop {
                ema_score,
                peak_score,
                delta,
            } => {
                write!(
                    f,
                    "PERF DROP: ema={ema_score:.4} < (1-δ={delta:.4})*peak={peak_score:.4}"
                )
            }
            Self::DualViolation {
                drift,
                epsilon,
                ema_score,
                peak_score,
                delta,
            } => {
                write!(f, "DUAL VIOLATION: drift={drift:.4}>ε={epsilon:.4} AND ema={ema_score:.4}<(1-δ={delta:.4})*peak={peak_score:.4}")
            }
            Self::WarmingUp { checks_remaining } => {
                write!(f, "WARMING UP: {checks_remaining} checks remaining")
            }
            Self::NotDue => write!(f, "NOT DUE"),
        }
    }
}

// ---------------------------------------------------------------------------
// AlignmentMonitor — EMAS
// ---------------------------------------------------------------------------

/// Monitors parameter drift and performance alignment.
///
/// Stores a frozen copy of the anchor parameters and tracks performance
/// via an EMA score. Checks are performed every `check_interval` iterations.
pub struct AlignmentMonitor {
    config: AlignmentConfig,
    /// Frozen anchor parameter snapshot (flattened).
    anchor_params: Vec<f32>,
    /// L2 norm of anchor (precomputed for normalisation).
    anchor_norm: f32,
    /// Current EMA of performance score.
    ema_score: f32,
    /// Peak EMA score observed.
    peak_score: f32,
    /// Current iteration counter.
    iteration: u64,
    /// Number of alignment checks performed.
    checks_performed: u64,
    /// Number of violations detected.
    violations: u64,
    /// Whether the EMA has been initialised.
    ema_initialised: bool,
}

impl AlignmentMonitor {
    /// Create a new monitor anchored to the given parameter snapshot.
    ///
    /// The anchor is cloned and frozen — the monitor holds its own copy
    /// that cannot be modified through any public interface.
    pub fn new(anchor_params: &[f32], config: AlignmentConfig) -> Self {
        let anchor_norm = anchor_params.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self {
            config,
            anchor_params: anchor_params.to_vec(),
            anchor_norm: anchor_norm.max(1e-12), // Prevent division by zero
            ema_score: 0.0,
            peak_score: 0.0,
            iteration: 0,
            checks_performed: 0,
            violations: 0,
            ema_initialised: false,
        }
    }

    /// Report a performance score for the current iteration.
    ///
    /// This should be called every iteration with a scalar metric
    /// (e.g. accuracy on a validation batch, or 1 - loss).
    pub fn report_score(&mut self, score: f32) {
        if !self.ema_initialised {
            self.ema_score = score;
            self.peak_score = score;
            self.ema_initialised = true;
        } else {
            self.ema_score =
                self.config.ema_decay * self.ema_score + (1.0 - self.config.ema_decay) * score;
            if self.ema_score > self.peak_score {
                self.peak_score = self.ema_score;
            }
        }
    }

    /// Advance the iteration counter and check alignment if due.
    ///
    /// Returns the alignment status. Callers should check `needs_rollback()`.
    pub fn step(&mut self, current_params: &[f32]) -> AlignmentStatus {
        self.iteration += 1;

        // Only check at intervals
        if self.iteration % self.config.check_interval != 0 {
            return AlignmentStatus::NotDue;
        }

        self.checks_performed += 1;

        // Warm-up period
        if self.checks_performed <= self.config.warmup_checks {
            return AlignmentStatus::WarmingUp {
                checks_remaining: self.config.warmup_checks - self.checks_performed + 1,
            };
        }

        // Compute parameter drift: ||θ - θ₀|| / ||θ₀||
        let drift = self.compute_drift(current_params);

        let drift_violated = drift > self.config.epsilon;
        let perf_violated =
            self.peak_score > 0.0 && self.ema_score < (1.0 - self.config.delta) * self.peak_score;

        if drift_violated && perf_violated {
            self.violations += 1;
            AlignmentStatus::DualViolation {
                drift,
                epsilon: self.config.epsilon,
                ema_score: self.ema_score,
                peak_score: self.peak_score,
                delta: self.config.delta,
            }
        } else if drift_violated {
            self.violations += 1;
            AlignmentStatus::DriftViolation {
                drift,
                epsilon: self.config.epsilon,
            }
        } else if perf_violated {
            self.violations += 1;
            AlignmentStatus::PerformanceDrop {
                ema_score: self.ema_score,
                peak_score: self.peak_score,
                delta: self.config.delta,
            }
        } else {
            AlignmentStatus::Healthy {
                drift,
                ema_score: self.ema_score,
                peak_score: self.peak_score,
            }
        }
    }

    /// Compute relative parameter drift.
    fn compute_drift(&self, current: &[f32]) -> f32 {
        let len = self.anchor_params.len().min(current.len());
        let mut diff_sq = 0.0f32;
        for i in 0..len {
            let d = current[i] - self.anchor_params[i];
            diff_sq += d * d;
        }
        // If current has more params (shouldn't happen), count them as drift
        for i in len..current.len() {
            diff_sq += current[i] * current[i];
        }
        diff_sq.sqrt() / self.anchor_norm
    }

    /// Manually check drift without advancing iteration (for debugging).
    pub fn check_drift(&self, current_params: &[f32]) -> f32 {
        self.compute_drift(current_params)
    }

    /// Reset the anchor to new parameters (after a successful consolidation).
    ///
    /// This is the ONLY way to update the anchor. It should only be called
    /// after explicit human approval or a verified self-improvement cycle.
    pub fn reset_anchor(&mut self, new_anchor: &[f32]) {
        self.anchor_params = new_anchor.to_vec();
        self.anchor_norm = new_anchor
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        // Reset peak to current EMA so the new anchor starts clean
        self.peak_score = self.ema_score;
    }

    /// Current EMA score.
    pub fn ema_score(&self) -> f32 {
        self.ema_score
    }

    /// Peak EMA score observed.
    pub fn peak_score(&self) -> f32 {
        self.peak_score
    }

    /// Current iteration.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Number of alignment checks performed.
    pub fn checks_performed(&self) -> u64 {
        self.checks_performed
    }

    /// Number of violations detected.
    pub fn violations(&self) -> u64 {
        self.violations
    }

    /// Violation rate (fraction of checks that violated).
    pub fn violation_rate(&self) -> f32 {
        if self.checks_performed == 0 {
            0.0
        } else {
            self.violations as f32 / self.checks_performed as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    }

    fn config_fast() -> AlignmentConfig {
        AlignmentConfig {
            epsilon: 0.1,
            delta: 0.05,
            check_interval: 1, // Check every iteration for testing
            ema_decay: 0.9,
            warmup_checks: 2,
        }
    }

    #[test]
    fn no_drift_is_healthy() {
        let a = anchor();
        let config = config_fast();
        let mut monitor = AlignmentMonitor::new(&a, config);

        // Report good scores
        for _ in 0..5 {
            monitor.report_score(0.95);
            let status = monitor.step(&a); // Same params = zero drift
            if status.was_checked() && !matches!(status, AlignmentStatus::WarmingUp { .. }) {
                assert!(
                    matches!(status, AlignmentStatus::Healthy { .. }),
                    "Expected healthy, got: {status}"
                );
                assert!(!status.needs_rollback());
            }
        }
    }

    #[test]
    fn large_drift_triggers_violation() {
        let a = anchor();
        let config = config_fast();
        let mut monitor = AlignmentMonitor::new(&a, config);

        // Warm up
        for _ in 0..3 {
            monitor.report_score(0.95);
            monitor.step(&a);
        }

        // Now drift the parameters significantly
        let drifted: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        monitor.report_score(0.95);
        let status = monitor.step(&drifted);
        assert!(status.needs_rollback(), "Expected rollback, got: {status}");
        assert!(matches!(
            status,
            AlignmentStatus::DriftViolation { .. } | AlignmentStatus::DualViolation { .. }
        ));
    }

    #[test]
    fn performance_drop_triggers_violation() {
        let a = anchor();
        let config = config_fast();
        let mut monitor = AlignmentMonitor::new(&a, config);

        // Establish high baseline
        for _ in 0..10 {
            monitor.report_score(0.95);
            monitor.step(&a);
        }

        // Now performance crashes
        for _ in 0..20 {
            monitor.report_score(0.50);
            let status = monitor.step(&a);
            if matches!(status, AlignmentStatus::PerformanceDrop { .. }) {
                // Found the violation
                assert!(status.needs_rollback());
                return;
            }
        }
        panic!("Expected performance drop to be detected");
    }

    #[test]
    fn warmup_period_respected() {
        let a = anchor();
        let config = config_fast();
        let mut monitor = AlignmentMonitor::new(&a, config);

        monitor.report_score(0.95);
        let s1 = monitor.step(&a);
        assert!(matches!(s1, AlignmentStatus::WarmingUp { .. }));

        monitor.report_score(0.95);
        let s2 = monitor.step(&a);
        assert!(matches!(s2, AlignmentStatus::WarmingUp { .. }));

        // Third check is after warm-up
        monitor.report_score(0.95);
        let s3 = monitor.step(&a);
        assert!(matches!(s3, AlignmentStatus::Healthy { .. }));
    }

    #[test]
    fn reset_anchor_updates_reference() {
        let a = anchor();
        let config = config_fast();
        let mut monitor = AlignmentMonitor::new(&a, config);

        let drifted: Vec<f32> = a.iter().map(|x| x * 1.5).collect();
        let drift = monitor.check_drift(&drifted);
        assert!(drift > 0.1);

        // Reset anchor to drifted point
        monitor.reset_anchor(&drifted);
        let drift_after = monitor.check_drift(&drifted);
        assert!(drift_after < 1e-6);
    }

    #[test]
    fn check_interval_skips_non_due() {
        let a = anchor();
        let config = AlignmentConfig {
            check_interval: 10,
            warmup_checks: 0, // No warmup for this test
            ..config_fast()
        };
        let mut monitor = AlignmentMonitor::new(&a, config);

        for i in 1..=15 {
            monitor.report_score(0.95);
            let status = monitor.step(&a);
            if i % 10 == 0 {
                assert!(status.was_checked(), "Should check at iteration {i}");
            } else {
                assert_eq!(status, AlignmentStatus::NotDue);
            }
        }
    }

    #[test]
    fn violation_counting() {
        let a = anchor();
        let config = config_fast();
        let mut monitor = AlignmentMonitor::new(&a, config);

        // Warm up
        for _ in 0..3 {
            monitor.report_score(0.95);
            monitor.step(&a);
        }

        assert_eq!(monitor.violations(), 0);

        // Trigger drift violation
        let drifted: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        monitor.report_score(0.95);
        monitor.step(&drifted);

        assert_eq!(monitor.violations(), 1);
        assert!(monitor.violation_rate() > 0.0);
    }

    #[test]
    fn zero_anchor_norm_handled() {
        let zero = vec![0.0f32; 5];
        let config = config_fast();
        let monitor = AlignmentMonitor::new(&zero, config);
        // anchor_norm should be clamped to 1e-12
        let drift = monitor.check_drift(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(drift.is_finite());
    }
}

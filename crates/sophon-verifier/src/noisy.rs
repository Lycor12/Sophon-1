//! Tool-assisted truth acquisition — Spec §4.4.
//!
//! External tool outputs are treated as noisy observations with precision
//! weighting. Cross-validation and consensus mechanisms prevent integration
//! of unreliable data.
//!
//! # Novel technique: PWNA (Precision-Weighted Noisy Acquisition)
//!
//! Each tool has a tracked reliability score based on historical accuracy.
//! New observations are weighted by `π_tool = 1 / var(errors)` — tools
//! that have been consistently accurate get high precision (their output
//! strongly influences belief), while tools with high error variance get
//! low precision. This mirrors the active inference precision mechanism
//! in sophon-inference but applied to external tool observations.

// ---------------------------------------------------------------------------
// Tool observation
// ---------------------------------------------------------------------------

/// A single observation from an external tool.
#[derive(Debug, Clone)]
pub struct ToolObservation {
    /// Which tool produced this observation.
    pub tool_id: String,
    /// The observation value (e.g. a fact, a number, a string).
    pub value: String,
    /// Self-reported confidence (0.0 to 1.0), if the tool provides one.
    pub self_confidence: Option<f32>,
    /// Timestamp of the observation.
    pub timestamp: u64,
}

/// Result of a consensus check.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusResult {
    /// k of n tools agree — consensus reached.
    Agreed {
        value: String,
        agreement_count: usize,
        total_tools: usize,
    },
    /// No consensus — tools disagree.
    Disagreed {
        unique_values: Vec<String>,
        total_tools: usize,
    },
    /// Not enough tools available for a consensus.
    InsufficientTools { available: usize, required: usize },
}

impl ConsensusResult {
    pub fn is_agreed(&self) -> bool {
        matches!(self, Self::Agreed { .. })
    }
}

// ---------------------------------------------------------------------------
// Tool reliability tracker — PWNA
// ---------------------------------------------------------------------------

/// Tracks reliability of individual tools.
pub struct ToolReliability {
    /// Per-tool error history: (tool_id → list of squared errors).
    error_history: std::collections::HashMap<String, Vec<f32>>,
    /// Per-tool running error count.
    observation_count: std::collections::HashMap<String, u64>,
    /// Maximum history length per tool.
    max_history: usize,
    /// Minimum observations before precision is meaningful.
    min_observations: usize,
    /// Default precision for new/unknown tools.
    default_precision: f32,
    /// Maximum precision cap.
    max_precision: f32,
    /// Minimum precision floor.
    min_precision: f32,
}

impl ToolReliability {
    /// Create a new reliability tracker.
    pub fn new() -> Self {
        Self {
            error_history: std::collections::HashMap::new(),
            observation_count: std::collections::HashMap::new(),
            max_history: 100,
            min_observations: 5,
            default_precision: 1.0,
            max_precision: 100.0,
            min_precision: 0.01,
        }
    }

    /// Record a tool's observation error (squared error from ground truth).
    pub fn record_error(&mut self, tool_id: &str, squared_error: f32) {
        let history = self.error_history.entry(tool_id.to_string()).or_default();
        history.push(squared_error);
        if history.len() > self.max_history {
            history.remove(0);
        }
        *self
            .observation_count
            .entry(tool_id.to_string())
            .or_insert(0) += 1;
    }

    /// Record a correct observation (zero error).
    pub fn record_correct(&mut self, tool_id: &str) {
        self.record_error(tool_id, 0.0);
    }

    /// Get the precision (1/variance) for a tool.
    pub fn precision(&self, tool_id: &str) -> f32 {
        let history = match self.error_history.get(tool_id) {
            Some(h) if h.len() >= self.min_observations => h,
            _ => return self.default_precision,
        };

        let n = history.len() as f32;
        let mean = history.iter().sum::<f32>() / n;
        let variance = history.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / n;

        // precision = 1 / (variance + epsilon)
        let precision = 1.0 / (variance + 1e-6);
        precision.clamp(self.min_precision, self.max_precision)
    }

    /// Weight an observation by its tool's precision.
    pub fn weighted_value(&self, tool_id: &str, raw_value: f32) -> f32 {
        let pi = self.precision(tool_id);
        pi * raw_value
    }

    /// Get the number of observations recorded for a tool.
    pub fn observations_for(&self, tool_id: &str) -> u64 {
        self.observation_count.get(tool_id).copied().unwrap_or(0)
    }

    /// Whether a tool has enough history for meaningful precision.
    pub fn is_calibrated(&self, tool_id: &str) -> bool {
        self.error_history
            .get(tool_id)
            .map(|h| h.len() >= self.min_observations)
            .unwrap_or(false)
    }
}

impl Default for ToolReliability {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Consensus mechanism
// ---------------------------------------------------------------------------

/// Check consensus among multiple tool observations.
///
/// Spec §4.4.2: k of n tools must agree for critical facts.
pub fn check_consensus(observations: &[ToolObservation], min_agreement: usize) -> ConsensusResult {
    if observations.len() < min_agreement {
        return ConsensusResult::InsufficientTools {
            available: observations.len(),
            required: min_agreement,
        };
    }

    // Count occurrences of each value
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for obs in observations {
        *counts.entry(&obs.value).or_insert(0) += 1;
    }

    // Find the most common value
    if let Some((&value, &count)) = counts.iter().max_by_key(|&(_, &c)| c) {
        if count >= min_agreement {
            return ConsensusResult::Agreed {
                value: value.to_string(),
                agreement_count: count,
                total_tools: observations.len(),
            };
        }
    }

    ConsensusResult::Disagreed {
        unique_values: counts.keys().map(|s| s.to_string()).collect(),
        total_tools: observations.len(),
    }
}

/// Precision-weighted integration of multiple numeric observations.
///
/// Combines observations weighted by each tool's reliability:
/// result = Σ(π_i * v_i) / Σ(π_i)
pub fn precision_weighted_mean(
    observations: &[(String, f32)], // (tool_id, value)
    reliability: &ToolReliability,
) -> Option<f32> {
    if observations.is_empty() {
        return None;
    }

    let mut weighted_sum = 0.0f32;
    let mut precision_sum = 0.0f32;

    for (tool_id, value) in observations {
        let pi = reliability.precision(tool_id);
        weighted_sum += pi * value;
        precision_sum += pi;
    }

    if precision_sum > 0.0 {
        Some(weighted_sum / precision_sum)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tool_gets_default_precision() {
        let tracker = ToolReliability::new();
        let p = tracker.precision("unknown_tool");
        assert!((p - 1.0).abs() < 0.01);
    }

    #[test]
    fn accurate_tool_gets_high_precision() {
        let mut tracker = ToolReliability::new();
        for _ in 0..10 {
            tracker.record_correct("good_tool");
        }
        let p = tracker.precision("good_tool");
        assert!(p > 1.0, "Precision should be high: {p}");
    }

    #[test]
    fn inaccurate_tool_gets_low_precision() {
        let mut tracker = ToolReliability::new();
        // Large and variable errors
        for i in 0..10 {
            tracker.record_error("bad_tool", (i as f32) * 10.0);
        }
        let p = tracker.precision("bad_tool");
        // Should be lower than default
        assert!(p < 1.0, "Precision should be low for inaccurate tool: {p}");
    }

    #[test]
    fn consensus_agreement() {
        let obs = vec![
            ToolObservation {
                tool_id: "a".into(),
                value: "42".into(),
                self_confidence: None,
                timestamp: 0,
            },
            ToolObservation {
                tool_id: "b".into(),
                value: "42".into(),
                self_confidence: None,
                timestamp: 0,
            },
            ToolObservation {
                tool_id: "c".into(),
                value: "43".into(),
                self_confidence: None,
                timestamp: 0,
            },
        ];
        let result = check_consensus(&obs, 2);
        assert!(result.is_agreed());
        if let ConsensusResult::Agreed {
            value,
            agreement_count,
            ..
        } = result
        {
            assert_eq!(value, "42");
            assert_eq!(agreement_count, 2);
        }
    }

    #[test]
    fn consensus_disagreement() {
        let obs = vec![
            ToolObservation {
                tool_id: "a".into(),
                value: "1".into(),
                self_confidence: None,
                timestamp: 0,
            },
            ToolObservation {
                tool_id: "b".into(),
                value: "2".into(),
                self_confidence: None,
                timestamp: 0,
            },
            ToolObservation {
                tool_id: "c".into(),
                value: "3".into(),
                self_confidence: None,
                timestamp: 0,
            },
        ];
        let result = check_consensus(&obs, 2);
        assert!(!result.is_agreed());
    }

    #[test]
    fn consensus_insufficient() {
        let obs = vec![ToolObservation {
            tool_id: "a".into(),
            value: "42".into(),
            self_confidence: None,
            timestamp: 0,
        }];
        let result = check_consensus(&obs, 3);
        assert!(matches!(result, ConsensusResult::InsufficientTools { .. }));
    }

    #[test]
    fn precision_weighted_mean_basic() {
        let tracker = ToolReliability::new(); // All tools get default precision = 1.0
        let obs = vec![("a".to_string(), 10.0), ("b".to_string(), 20.0)];
        let mean = precision_weighted_mean(&obs, &tracker).unwrap();
        assert!((mean - 15.0).abs() < 0.01);
    }

    #[test]
    fn precision_weighted_mean_empty() {
        let tracker = ToolReliability::new();
        assert!(precision_weighted_mean(&[], &tracker).is_none());
    }

    #[test]
    fn calibration_check() {
        let mut tracker = ToolReliability::new();
        assert!(!tracker.is_calibrated("tool"));
        for _ in 0..5 {
            tracker.record_correct("tool");
        }
        assert!(tracker.is_calibrated("tool"));
    }
}

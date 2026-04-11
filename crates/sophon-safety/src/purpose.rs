//! Purpose invariant enforcement — Addendum C.
//!
//! The system encodes creator-defined objectives as a normalised purpose vector
//! in a latent space. Every proposed action or self-modification is projected
//! into the same space and compared via cosine similarity. If the similarity
//! falls below a threshold, the action is rejected as a purpose violation.
//!
//! # Novel technique: PICV (Purpose-Invariant Constraint Verification)
//!
//! Standard alignment checks rely on post-hoc reward signals. PICV operates
//! pre-execution: the purpose vector is frozen at initialisation and never
//! modified. Each candidate action is mapped through a lightweight linear
//! projection into purpose-space, and the cosine gate decides accept/reject
//! in O(d) time. The purpose vector itself is stored in a separate immutable
//! buffer — no gradient path can reach it.
//!
//! ```text
//! action_embed = W_proj * action_repr
//! sim = cosine(action_embed, purpose_vec)
//! if sim < threshold: REJECT
//! else: ACCEPT
//! ```

use sophon_core::error::CoreError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the purpose invariant gate.
#[derive(Debug, Clone)]
pub struct PurposeConfig {
    /// Dimensionality of the purpose vector.
    pub dim: usize,
    /// Cosine similarity threshold. Actions below this are rejected.
    /// Spec says deviation = invalid state, so this must be conservative.
    pub threshold: f32,
    /// Maximum number of purpose objectives (multi-objective support).
    pub max_objectives: usize,
    /// Penalty weight for near-boundary actions (soft warning zone).
    /// Actions with similarity in [threshold, threshold + soft_margin]
    /// are accepted but flagged.
    pub soft_margin: f32,
}

impl PurposeConfig {
    /// Conservative defaults per spec: high threshold, narrow margin.
    pub fn default_for(dim: usize) -> Self {
        Self {
            dim,
            threshold: 0.7,
            max_objectives: 4,
            soft_margin: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// PurposeViolation
// ---------------------------------------------------------------------------

/// Describes why an action was rejected or flagged by the purpose gate.
#[derive(Debug, Clone, PartialEq)]
pub enum PurposeViolation {
    /// Cosine similarity below hard threshold — action blocked.
    HardReject {
        similarity: f32,
        threshold: f32,
        objective_index: usize,
    },
    /// Similarity in the soft-margin zone — action allowed but flagged.
    SoftWarning {
        similarity: f32,
        threshold: f32,
        objective_index: usize,
    },
    /// The action vector contains NaN or Inf — cannot evaluate.
    NumericalFailure,
    /// Dimensionality mismatch between action embedding and purpose vector.
    DimensionMismatch { expected: usize, got: usize },
}

impl core::fmt::Display for PurposeViolation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HardReject {
                similarity,
                threshold,
                objective_index,
            } => {
                write!(f, "PURPOSE VIOLATION: action similarity {similarity:.4} < threshold {threshold:.4} on objective {objective_index}")
            }
            Self::SoftWarning {
                similarity,
                threshold,
                objective_index,
            } => {
                write!(f, "PURPOSE WARNING: action similarity {similarity:.4} near threshold {threshold:.4} on objective {objective_index}")
            }
            Self::NumericalFailure => {
                write!(f, "PURPOSE ERROR: action vector contains NaN/Inf")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "PURPOSE ERROR: expected dim {expected}, got {got}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PurposeGate — PICV
// ---------------------------------------------------------------------------

/// The purpose invariant gate.
///
/// Stores an immutable set of purpose vectors. No method modifies them after
/// construction — this is the "non-modifiable constraint" from Addendum C.
pub struct PurposeGate {
    /// Frozen purpose vectors (one per objective). Each is L2-normalised.
    purpose_vectors: Vec<Vec<f32>>,
    /// Linear projection matrix (flattened row-major, dim × input_dim).
    /// Maps action representations to purpose-space.
    projection: Vec<f32>,
    /// Input dimensionality (pre-projection).
    input_dim: usize,
    /// Purpose-space dimensionality.
    purpose_dim: usize,
    /// Configuration (immutable after construction).
    config: PurposeConfig,
    /// Running count of rejections for monitoring.
    rejection_count: u64,
    /// Running count of evaluations.
    eval_count: u64,
}

impl PurposeGate {
    /// Create a new purpose gate with the given purpose vectors and projection.
    ///
    /// # Arguments
    /// - `purpose_vectors`: One normalised vector per creator-defined objective.
    /// - `projection`: Flattened [purpose_dim, input_dim] matrix.
    /// - `input_dim`: Dimensionality of action representations.
    /// - `config`: Gate configuration.
    ///
    /// Each purpose vector is L2-normalised on construction and thereafter immutable.
    pub fn new(
        mut purpose_vectors: Vec<Vec<f32>>,
        projection: Vec<f32>,
        input_dim: usize,
        config: PurposeConfig,
    ) -> Result<Self, CoreError> {
        if purpose_vectors.is_empty() {
            return Err(CoreError::ZeroDimension);
        }
        if purpose_vectors.len() > config.max_objectives {
            return Err(CoreError::ShapeMismatch {
                expected: [config.max_objectives, 0],
                got: [purpose_vectors.len(), 0],
            });
        }
        for pv in purpose_vectors.iter_mut() {
            if pv.len() != config.dim {
                return Err(CoreError::ShapeMismatch {
                    expected: [config.dim, 0],
                    got: [pv.len(), 0],
                });
            }
            // L2-normalise
            let norm = pv.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-12 {
                return Err(CoreError::NumericalInstability {
                    op: "purpose_vector_normalise",
                });
            }
            for x in pv.iter_mut() {
                *x /= norm;
            }
        }
        let expected_proj_len = config.dim * input_dim;
        if projection.len() != expected_proj_len {
            return Err(CoreError::ShapeMismatch {
                expected: [config.dim, input_dim],
                got: [projection.len(), 1],
            });
        }
        Ok(Self {
            purpose_vectors,
            projection,
            input_dim,
            purpose_dim: config.dim,
            config,
            rejection_count: 0,
            eval_count: 0,
        })
    }

    /// Create a simple identity-projection gate where input_dim == purpose_dim.
    ///
    /// Useful when the action embedding is already in purpose-space.
    pub fn identity(
        purpose_vectors: Vec<Vec<f32>>,
        config: PurposeConfig,
    ) -> Result<Self, CoreError> {
        let dim = config.dim;
        // Identity matrix flattened
        let mut proj = vec![0.0f32; dim * dim];
        for i in 0..dim {
            proj[i * dim + i] = 1.0;
        }
        Self::new(purpose_vectors, proj, dim, config)
    }

    /// Project an action representation into purpose-space.
    fn project(&self, action_repr: &[f32]) -> Result<Vec<f32>, PurposeViolation> {
        if action_repr.len() != self.input_dim {
            return Err(PurposeViolation::DimensionMismatch {
                expected: self.input_dim,
                got: action_repr.len(),
            });
        }
        // Check for NaN/Inf
        for &v in action_repr {
            if !v.is_finite() {
                return Err(PurposeViolation::NumericalFailure);
            }
        }
        // Matrix-vector multiply: result[i] = sum_j projection[i*input_dim + j] * action_repr[j]
        let mut result = vec![0.0f32; self.purpose_dim];
        for i in 0..self.purpose_dim {
            let row_start = i * self.input_dim;
            let mut sum = 0.0f32;
            for j in 0..self.input_dim {
                sum += self.projection[row_start + j] * action_repr[j];
            }
            result[i] = sum;
        }
        Ok(result)
    }

    /// Cosine similarity between two vectors.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        dot / denom
    }

    /// Evaluate an action representation against all purpose objectives.
    ///
    /// Returns `Ok(similarities)` if all objectives pass the hard threshold.
    /// Returns `Err(violation)` on the first hard rejection.
    /// Soft warnings are collected but do not cause rejection.
    pub fn evaluate(
        &mut self,
        action_repr: &[f32],
    ) -> Result<(Vec<f32>, Vec<PurposeViolation>), PurposeViolation> {
        self.eval_count += 1;

        let projected = self.project(action_repr)?;
        let mut similarities = Vec::with_capacity(self.purpose_vectors.len());
        let mut warnings = Vec::new();

        for (idx, pv) in self.purpose_vectors.iter().enumerate() {
            let sim = Self::cosine(&projected, pv);
            similarities.push(sim);

            if sim < self.config.threshold {
                self.rejection_count += 1;
                return Err(PurposeViolation::HardReject {
                    similarity: sim,
                    threshold: self.config.threshold,
                    objective_index: idx,
                });
            } else if sim < self.config.threshold + self.config.soft_margin {
                warnings.push(PurposeViolation::SoftWarning {
                    similarity: sim,
                    threshold: self.config.threshold,
                    objective_index: idx,
                });
            }
        }

        Ok((similarities, warnings))
    }

    /// Quick boolean check: does this action pass the purpose gate?
    pub fn is_allowed(&mut self, action_repr: &[f32]) -> bool {
        self.evaluate(action_repr).is_ok()
    }

    /// Fraction of evaluations that were rejected.
    pub fn rejection_rate(&self) -> f32 {
        if self.eval_count == 0 {
            return 0.0;
        }
        self.rejection_count as f32 / self.eval_count as f32
    }

    /// Total evaluations performed.
    pub fn eval_count(&self) -> u64 {
        self.eval_count
    }

    /// Total rejections.
    pub fn rejection_count(&self) -> u64 {
        self.rejection_count
    }

    /// Number of purpose objectives.
    pub fn n_objectives(&self) -> usize {
        self.purpose_vectors.len()
    }

    /// Purpose-space dimensionality.
    pub fn purpose_dim(&self) -> usize {
        self.purpose_dim
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gate() -> PurposeGate {
        let config = PurposeConfig::default_for(4);
        // Purpose: "do math" direction
        let purpose = vec![vec![1.0, 0.0, 0.0, 0.0]];
        PurposeGate::identity(purpose, config).unwrap()
    }

    #[test]
    fn aligned_action_passes() {
        let mut gate = make_gate();
        // Action strongly aligned with purpose
        let action = vec![0.9, 0.1, 0.0, 0.0];
        let result = gate.evaluate(&action);
        assert!(result.is_ok());
        let (sims, _warnings) = result.unwrap();
        assert!(sims[0] > 0.9);
    }

    #[test]
    fn orthogonal_action_rejected() {
        let mut gate = make_gate();
        // Action orthogonal to purpose
        let action = vec![0.0, 1.0, 0.0, 0.0];
        let result = gate.evaluate(&action);
        assert!(result.is_err());
        match result.unwrap_err() {
            PurposeViolation::HardReject { similarity, .. } => {
                assert!(similarity.abs() < 0.01);
            }
            other => panic!("Expected HardReject, got {other}"),
        }
    }

    #[test]
    fn soft_margin_warns_but_allows() {
        let config = PurposeConfig {
            dim: 4,
            threshold: 0.7,
            max_objectives: 4,
            soft_margin: 0.2,
        };
        let purpose = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let mut gate = PurposeGate::identity(purpose, config).unwrap();

        // Action with cosine ~0.75 (between 0.7 and 0.9)
        let action = vec![0.75, 0.66, 0.0, 0.0];
        let result = gate.evaluate(&action);
        assert!(result.is_ok());
        let (_sims, warnings) = result.unwrap();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn nan_action_rejected() {
        let mut gate = make_gate();
        let action = vec![f32::NAN, 0.0, 0.0, 0.0];
        let result = gate.evaluate(&action);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PurposeViolation::NumericalFailure
        ));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut gate = make_gate();
        let action = vec![1.0, 0.0]; // Wrong dim
        let result = gate.evaluate(&action);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PurposeViolation::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn rejection_rate_tracks() {
        let mut gate = make_gate();
        let aligned = vec![1.0, 0.0, 0.0, 0.0];
        let orthogonal = vec![0.0, 1.0, 0.0, 0.0];

        let _ = gate.evaluate(&aligned);
        let _ = gate.evaluate(&orthogonal);
        let _ = gate.evaluate(&aligned);

        assert_eq!(gate.eval_count(), 3);
        assert_eq!(gate.rejection_count(), 1);
        assert!((gate.rejection_rate() - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn multi_objective_all_must_pass() {
        let config = PurposeConfig::default_for(4);
        let purposes = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Math
            vec![0.0, 1.0, 0.0, 0.0], // Code
        ];
        let mut gate = PurposeGate::identity(purposes, config).unwrap();

        // Action aligned with both (diagonal)
        let action = vec![1.0, 1.0, 0.0, 0.0];
        let result = gate.evaluate(&action);
        assert!(result.is_ok());

        // Action aligned with only first objective
        let action2 = vec![1.0, 0.0, 0.0, 0.0];
        let result2 = gate.evaluate(&action2);
        assert!(result2.is_err());
    }

    #[test]
    fn empty_purpose_rejected() {
        let config = PurposeConfig::default_for(4);
        let result = PurposeGate::identity(vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn zero_vector_purpose_rejected() {
        let config = PurposeConfig::default_for(4);
        let result = PurposeGate::identity(vec![vec![0.0, 0.0, 0.0, 0.0]], config);
        assert!(result.is_err());
    }

    #[test]
    fn purpose_is_immutable_after_construction() {
        // Verify that the gate has no &mut methods that modify purpose vectors.
        // (Structural test: the fact that purpose_vectors is private and
        //  no method takes &mut self to modify it is the invariant.)
        let gate = make_gate();
        assert_eq!(gate.n_objectives(), 1);
        assert_eq!(gate.purpose_dim(), 4);
    }
}

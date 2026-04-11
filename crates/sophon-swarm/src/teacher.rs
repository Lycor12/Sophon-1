//! Teacher agent — problem generation and curriculum control.
//!
//! Spec §4.2.1, §6.2.1: The teacher agent:
//!   1. Selects problems from the knowledge base at the current level
//!   2. Generates problem variants (perturbations of known theorems)
//!   3. Tracks student success rates to decide curriculum advancement
//!   4. Provides building blocks (proven lemmas) as context
//!
//! # Novel technique: ACLG (Adaptive Curriculum Level Gating)
//!
//! Standard curriculum learning advances when accuracy exceeds a threshold.
//! ACLG additionally monitors the *diversity* of successful solutions: if
//! students all converge to the same trivial solution strategy, the level
//! is NOT advanced even with high accuracy. Diversity is measured by the
//! Jaccard distance between the tactic sets of successful proofs.

use sophon_core::rng::Rng;
use sophon_verifier::KnowledgeBase;

// ---------------------------------------------------------------------------
// Problem
// ---------------------------------------------------------------------------

/// A problem for students to attempt.
#[derive(Debug, Clone)]
pub struct Problem {
    /// Unique problem ID within this session.
    pub id: u64,
    /// Natural language statement of the problem.
    pub statement: String,
    /// Curriculum level this problem is at.
    pub level: u32,
    /// Topic tags for classification.
    pub tags: Vec<String>,
    /// Building blocks (lean source snippets) available as hints.
    pub building_blocks: Vec<String>,
    /// Optional: expected difficulty score (0.0 = trivial, 1.0 = very hard).
    pub difficulty: f32,
}

// ---------------------------------------------------------------------------
// CurriculumConfig
// ---------------------------------------------------------------------------

/// Configuration for curriculum progression.
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    /// Minimum success rate to advance to next level.
    pub advance_threshold: f32,
    /// Minimum number of problems at a level before considering advancement.
    pub min_problems_per_level: usize,
    /// Minimum diversity score (Jaccard) among successful solutions.
    pub min_diversity: f32,
    /// Maximum curriculum level.
    pub max_level: u32,
    /// Number of variant problems to generate per base problem.
    pub variants_per_problem: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            advance_threshold: 0.7,
            min_problems_per_level: 10,
            min_diversity: 0.3,
            max_level: 20,
            variants_per_problem: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Teacher
// ---------------------------------------------------------------------------

/// The teacher agent — generates problems and controls curriculum.
pub struct Teacher {
    /// Current curriculum level.
    pub level: u32,
    /// Configuration.
    pub config: CurriculumConfig,
    /// Problem counter (for unique IDs).
    next_problem_id: u64,
    /// Success history at current level: (problem_id, succeeded, tactic_set).
    level_history: Vec<(u64, bool, Vec<String>)>,
    /// RNG for problem generation.
    rng: Rng,
}

impl Teacher {
    /// Create a new teacher at level 1.
    pub fn new(config: CurriculumConfig, seed: u64) -> Self {
        Self {
            level: 1,
            config,
            next_problem_id: 1,
            level_history: Vec::new(),
            rng: Rng::new(seed),
        }
    }

    /// Generate a batch of problems at the current level.
    ///
    /// Uses the knowledge base to find building blocks and generate variants.
    pub fn generate_problems(&mut self, kb: &KnowledgeBase, count: usize) -> Vec<Problem> {
        let blocks = kb.building_blocks(self.level.saturating_sub(1));
        let block_sources: Vec<String> = blocks.iter().map(|e| e.lean_source.clone()).collect();

        // Get entries at current level for variant generation
        let current_entries = kb.at_level(self.level);

        let mut problems = Vec::with_capacity(count);

        for _ in 0..count {
            let id = self.next_problem_id;
            self.next_problem_id += 1;

            // Generate problem statement
            let (statement, tags, difficulty) = if !current_entries.is_empty() {
                // Variant of existing problem: perturb statement
                let idx = (self.rng.next_u64() as usize) % current_entries.len();
                let base = &current_entries[idx];
                let variant = self
                    .generate_variant(base.nl_statement.as_deref().unwrap_or(&base.theorem_name));
                let tags = base.tags.clone();
                let diff = 0.3 + (self.level as f32) * 0.05;
                (variant, tags, diff.min(1.0))
            } else {
                // No existing entries at this level: generate from scratch
                let (stmt, tags) = self.generate_base_problem();
                let diff = 0.2 + (self.level as f32) * 0.05;
                (stmt, tags, diff.min(1.0))
            };

            // Select relevant building blocks
            let n_blocks = (block_sources.len()).min(5);
            let selected_blocks: Vec<String> = if n_blocks > 0 {
                let start = (self.rng.next_u64() as usize) % block_sources.len().max(1);
                block_sources
                    .iter()
                    .cycle()
                    .skip(start)
                    .take(n_blocks)
                    .cloned()
                    .collect()
            } else {
                Vec::new()
            };

            problems.push(Problem {
                id,
                statement,
                level: self.level,
                tags,
                building_blocks: selected_blocks,
                difficulty,
            });
        }

        problems
    }

    /// Record the result of a student's attempt on a problem.
    pub fn record_result(&mut self, problem_id: u64, succeeded: bool, tactics_used: Vec<String>) {
        self.level_history
            .push((problem_id, succeeded, tactics_used));
    }

    /// Check whether the teacher should advance to the next level.
    ///
    /// Returns true if:
    ///   1. Enough problems attempted at current level
    ///   2. Success rate exceeds threshold
    ///   3. Solution diversity exceeds minimum (ACLG)
    ///   4. Current level < max level
    pub fn should_advance(&self) -> bool {
        if self.level >= self.config.max_level {
            return false;
        }
        if self.level_history.len() < self.config.min_problems_per_level {
            return false;
        }

        // Success rate
        let successes = self.level_history.iter().filter(|(_, s, _)| *s).count();
        let rate = successes as f32 / self.level_history.len() as f32;
        if rate < self.config.advance_threshold {
            return false;
        }

        // ACLG: diversity check
        let diversity = self.compute_diversity();
        if diversity < self.config.min_diversity {
            return false;
        }

        true
    }

    /// Advance to the next curriculum level. Clears level history.
    pub fn advance(&mut self) -> u32 {
        self.level += 1;
        self.level_history.clear();
        self.level
    }

    /// Current success rate at this level.
    pub fn success_rate(&self) -> f32 {
        if self.level_history.is_empty() {
            return 0.0;
        }
        let successes = self.level_history.iter().filter(|(_, s, _)| *s).count();
        successes as f32 / self.level_history.len() as f32
    }

    /// Number of problems attempted at current level.
    pub fn problems_attempted(&self) -> usize {
        self.level_history.len()
    }

    /// Compute solution diversity via mean pairwise Jaccard distance.
    fn compute_diversity(&self) -> f32 {
        let successful: Vec<&Vec<String>> = self
            .level_history
            .iter()
            .filter(|(_, s, _)| *s)
            .map(|(_, _, t)| t)
            .collect();

        if successful.len() < 2 {
            return 1.0; // Single solution: no diversity check needed
        }

        let mut total_jaccard = 0.0f32;
        let mut pairs = 0u32;

        for i in 0..successful.len() {
            for j in (i + 1)..successful.len() {
                total_jaccard += jaccard_distance(successful[i], successful[j]);
                pairs += 1;
            }
        }

        if pairs == 0 {
            return 1.0;
        }

        total_jaccard / pairs as f32
    }

    /// Generate a variant of a problem statement.
    fn generate_variant(&mut self, base: &str) -> String {
        // Deterministic perturbation strategies:
        let variant_type = self.rng.next_u64() % 4;
        match variant_type {
            0 => format!("Prove the generalisation: for all n, {base}"),
            1 => format!("Show that the converse holds: {base}"),
            2 => format!("Prove by induction: {base}"),
            3 => format!("Prove without using simp: {base}"),
            _ => base.to_string(),
        }
    }

    /// Generate a base problem from the problem template library.
    fn generate_base_problem(&mut self) -> (String, Vec<String>) {
        // Level-appropriate problem templates
        let templates: &[(&str, &[&str])] = match self.level {
            1 => &[
                (
                    "Prove that 0 + n = n for all natural numbers n",
                    &["arithmetic", "induction"],
                ),
                (
                    "Prove that n + 0 = n for all natural numbers n",
                    &["arithmetic"],
                ),
                ("Prove that True implies True", &["logic"]),
                ("Prove that for all propositions P, P implies P", &["logic"]),
                ("Prove that 1 + 1 = 2", &["arithmetic"]),
            ],
            2 => &[
                (
                    "Prove that addition of natural numbers is commutative",
                    &["arithmetic", "induction"],
                ),
                (
                    "Prove that addition of natural numbers is associative",
                    &["arithmetic", "induction"],
                ),
                ("Prove that for all n, n * 0 = 0", &["arithmetic"]),
                ("Prove that P and Q implies Q and P", &["logic"]),
                ("Prove de Morgan's law for conjunction", &["logic"]),
            ],
            3 => &[
                (
                    "Prove that multiplication distributes over addition",
                    &["algebra", "induction"],
                ),
                (
                    "Prove that n * 1 = n for all natural numbers",
                    &["arithmetic"],
                ),
                (
                    "Prove the contrapositive: if not Q then not P, given P implies Q",
                    &["logic"],
                ),
                ("Prove that the empty list has length 0", &["lists"]),
                (
                    "Prove that reversing a singleton list returns the same list",
                    &["lists"],
                ),
            ],
            _ => &[
                (
                    &"Prove a non-trivial theorem at the current difficulty level",
                    &["general"],
                ),
                (
                    &"Prove a theorem combining arithmetic and logic",
                    &["arithmetic", "logic"],
                ),
                (&"Prove a structural induction theorem", &["induction"]),
            ],
        };

        let idx = (self.rng.next_u64() as usize) % templates.len();
        let (stmt, tags) = templates[idx];
        (
            stmt.to_string(),
            tags.iter().map(|s| s.to_string()).collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Jaccard distance between two sets represented as string slices.
/// Returns 1.0 - |A ∩ B| / |A ∪ B|. Range [0, 1].
fn jaccard_distance(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    1.0 - (intersection as f32 / union as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_creates_at_level_1() {
        let teacher = Teacher::new(CurriculumConfig::default(), 42);
        assert_eq!(teacher.level, 1);
        assert_eq!(teacher.problems_attempted(), 0);
    }

    #[test]
    fn generate_problems_produces_correct_count() {
        let mut teacher = Teacher::new(CurriculumConfig::default(), 42);
        let kb = KnowledgeBase::new();
        let problems = teacher.generate_problems(&kb, 5);
        assert_eq!(problems.len(), 5);
        for p in &problems {
            assert_eq!(p.level, 1);
            assert!(!p.statement.is_empty());
        }
    }

    #[test]
    fn unique_problem_ids() {
        let mut teacher = Teacher::new(CurriculumConfig::default(), 42);
        let kb = KnowledgeBase::new();
        let p1 = teacher.generate_problems(&kb, 3);
        let p2 = teacher.generate_problems(&kb, 3);
        let all_ids: Vec<u64> = p1.iter().chain(p2.iter()).map(|p| p.id).collect();
        let unique: std::collections::HashSet<u64> = all_ids.iter().cloned().collect();
        assert_eq!(all_ids.len(), unique.len());
    }

    #[test]
    fn no_advance_before_min_problems() {
        let mut teacher = Teacher::new(CurriculumConfig::default(), 42);
        teacher.record_result(1, true, vec!["simp".into()]);
        assert!(!teacher.should_advance());
    }

    #[test]
    fn advance_when_threshold_met() {
        let config = CurriculumConfig {
            min_problems_per_level: 3,
            advance_threshold: 0.6,
            min_diversity: 0.0, // disable diversity check for this test
            ..Default::default()
        };
        let mut teacher = Teacher::new(config, 42);
        teacher.record_result(1, true, vec!["simp".into(), "ring".into()]);
        teacher.record_result(2, true, vec!["omega".into()]);
        teacher.record_result(3, true, vec!["simp".into()]);
        assert!(teacher.should_advance());
        let new_level = teacher.advance();
        assert_eq!(new_level, 2);
        assert_eq!(teacher.problems_attempted(), 0); // cleared
    }

    #[test]
    fn no_advance_low_success_rate() {
        let config = CurriculumConfig {
            min_problems_per_level: 3,
            advance_threshold: 0.7,
            min_diversity: 0.0,
            ..Default::default()
        };
        let mut teacher = Teacher::new(config, 42);
        teacher.record_result(1, true, vec!["simp".into()]);
        teacher.record_result(2, false, vec![]);
        teacher.record_result(3, false, vec![]);
        assert!(!teacher.should_advance()); // 33% < 70%
    }

    #[test]
    fn aclg_blocks_low_diversity() {
        let config = CurriculumConfig {
            min_problems_per_level: 3,
            advance_threshold: 0.5,
            min_diversity: 0.5, // require high diversity
            ..Default::default()
        };
        let mut teacher = Teacher::new(config, 42);
        // All successful with identical tactics
        teacher.record_result(1, true, vec!["simp".into()]);
        teacher.record_result(2, true, vec!["simp".into()]);
        teacher.record_result(3, true, vec!["simp".into()]);
        // High success rate but zero diversity → should NOT advance
        assert!(!teacher.should_advance());
    }

    #[test]
    fn jaccard_distance_identical_is_zero() {
        let a = vec!["simp".into(), "ring".into()];
        let b = vec!["simp".into(), "ring".into()];
        assert!(jaccard_distance(&a, &b) < 1e-6);
    }

    #[test]
    fn jaccard_distance_disjoint_is_one() {
        let a = vec!["simp".into()];
        let b = vec!["ring".into()];
        assert!((jaccard_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn success_rate_computation() {
        let mut teacher = Teacher::new(CurriculumConfig::default(), 0);
        teacher.record_result(1, true, vec![]);
        teacher.record_result(2, false, vec![]);
        teacher.record_result(3, true, vec![]);
        assert!((teacher.success_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn max_level_prevents_advance() {
        let config = CurriculumConfig {
            min_problems_per_level: 1,
            advance_threshold: 0.0,
            min_diversity: 0.0,
            max_level: 1,
            ..Default::default()
        };
        let mut teacher = Teacher::new(config, 42);
        teacher.record_result(1, true, vec!["x".into()]);
        assert!(!teacher.should_advance());
    }
}

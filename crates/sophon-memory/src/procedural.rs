//! Procedural memory — Learned action patterns and skills.
#![forbid(unsafe_code)]

use sophon_config::HDC_DIM;

/// A learned skill/action pattern.
#[derive(Debug, Clone)]
pub struct ActionPattern {
    pub name: String,
    pub preconditions: Vec<String>,
    pub effects: Vec<String>,
    pub success_rate: f32,
    pub avg_cost: f32,
    pub context_hv: Vec<f32>,
}

impl ActionPattern {
    /// Compute expected utility: success_rate - avg_cost.
    pub fn expected_utility(&self) -> f32 {
        self.success_rate - self.avg_cost
    }

    /// Match against current context.
    pub fn context_match(&self, context: &[f32]) -> f32 {
        cosine_similarity(context, &self.context_hv)
    }
}

/// Skill with refinement history.
#[derive(Debug, Clone)]
pub struct Skill {
    pub pattern: ActionPattern,
    pub attempts: usize,
    pub successes: usize,
    pub refinements: Vec<SkillRefinement>,
}

impl Skill {
    pub fn success_rate(&self) -> f32 {
        if self.attempts == 0 {
            0.5
        } else {
            self.successes as f32 / self.attempts as f32
        }
    }

    /// Update after attempt.
    pub fn record_attempt(&mut self, success: bool, cost: f32) {
        self.attempts += 1;
        if success {
            self.successes += 1;
        }
        self.pattern.avg_cost =
            (self.pattern.avg_cost * (self.attempts - 1) as f32 + cost) / self.attempts as f32;
        self.pattern.success_rate = self.success_rate();
    }

    /// Add refinement (improved version).
    pub fn add_refinement(&mut self, refinement: SkillRefinement) {
        self.refinements.push(refinement);
    }
}

/// Record of skill improvement.
#[derive(Debug, Clone)]
pub struct SkillRefinement {
    pub timestamp: u64,
    pub change_description: String,
    pub previous_success_rate: f32,
    pub new_success_rate: f32,
}

/// Procedural memory store.
pub struct ProceduralMemory {
    skills: Vec<Skill>,
    by_name: std::collections::HashMap<String, usize>,
}

impl ProceduralMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            skills: Vec::with_capacity(capacity),
            by_name: std::collections::HashMap::new(),
        }
    }

    /// Learn a new skill.
    pub fn learn(&mut self, pattern: ActionPattern) {
        let idx = self.skills.len();
        self.skills.push(Skill {
            pattern,
            attempts: 0,
            successes: 0,
            refinements: Vec::new(),
        });
        self.by_name
            .insert(self.skills[idx].pattern.name.clone(), idx);
    }

    /// Retrieve skill by name.
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.by_name.get(name).map(|&idx| &self.skills[idx])
    }

    /// Get mutable skill.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Skill> {
        self.by_name
            .get(name)
            .copied()
            .map(|idx| &mut self.skills[idx])
    }

    /// Find matching skills by context.
    pub fn find_matching(&self, context: &[f32], k: usize) -> Vec<&ActionPattern> {
        let mut scored: Vec<_> = self
            .skills
            .iter()
            .map(|s| (&s.pattern, s.pattern.context_match(context)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(k).map(|(p, _)| p).collect()
    }

    /// Find best skill for current context.
    pub fn best_for_context(&self, context: &[f32]) -> Option<&ActionPattern> {
        self.find_matching(context, 1).into_iter().next()
    }

    /// Get all skills.
    pub fn all_skills(&self) -> &[Skill] {
        &self.skills
    }

    /// Get skills sorted by expected utility.
    pub fn by_utility(&self) -> Vec<&Skill> {
        let mut sorted: Vec<_> = self.skills.iter().collect();
        sorted.sort_by(|a, b| {
            b.pattern
                .expected_utility()
                .partial_cmp(&a.pattern.expected_utility())
                .unwrap()
        });
        sorted
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_and_retrieve_skill() {
        let mut mem = ProceduralMemory::new(10);
        let pattern = ActionPattern {
            name: "sort".to_string(),
            preconditions: vec!["list exists".to_string()],
            effects: vec!["list sorted".to_string()],
            success_rate: 0.9,
            avg_cost: 0.3,
            context_hv: vec![1.0; HDC_DIM],
        };

        mem.learn(pattern);
        assert!(mem.get("sort").is_some());
    }

    #[test]
    fn skill_success_rate_update() {
        let mut mem = ProceduralMemory::new(10);
        let pattern = ActionPattern {
            name: "test".to_string(),
            preconditions: vec![],
            effects: vec![],
            success_rate: 0.5,
            avg_cost: 0.5,
            context_hv: vec![1.0; HDC_DIM],
        };

        mem.learn(pattern);

        let skill = mem.get_mut("test").unwrap();
        skill.record_attempt(true, 0.2);
        skill.record_attempt(false, 0.3);

        assert!(skill.success_rate() > 0.4 && skill.success_rate() < 0.6);
    }
}

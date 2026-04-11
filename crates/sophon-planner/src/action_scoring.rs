//! Action scoring using approved multi-factor formula.
//!
//! Score(a) = α·ExpectedTaskProgress + β·RetrievalGain + γ·VerificationLikelihood
//!           - δ·Surprise - ε·HomeostasisCost - ζ·PurposeRisk

use crate::rollout::{Action, LatentSimulator, LatentState, MemoryContext};

/// Weights for action scoring.
#[derive(Debug, Clone, Copy)]
pub struct ScoreWeights {
    pub alpha: f32,   // Task progress
    pub beta: f32,    // Retrieval gain
    pub gamma: f32,   // Verification likelihood
    pub delta: f32,   // Surprise penalty
    pub epsilon: f32, // Homeostasis cost
    pub zeta: f32,    // Purpose risk
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            alpha: 0.35,
            beta: 0.20,
            gamma: 0.25,
            delta: 0.10,
            epsilon: 0.07,
            zeta: 0.03,
        }
    }
}

/// A scored action candidate.
#[derive(Debug, Clone)]
pub struct ScoredAction {
    pub action: Action,
    pub score: f32,
    pub components: ScoreComponents,
}

impl ScoredAction {
    pub fn expected_utility(&self) -> f32 {
        self.score
    }
}

/// Individual score components.
#[derive(Debug, Clone, Copy)]
pub struct ScoreComponents {
    pub task_progress: f32,
    pub retrieval_gain: f32,
    pub verification_likelihood: f32,
    pub surprise: f32,
    pub homeostasis_cost: f32,
    pub purpose_risk: f32,
}

impl ScoreComponents {
    pub fn compute_total(&self, weights: &ScoreWeights) -> f32 {
        weights.alpha * self.task_progress
            + weights.beta * self.retrieval_gain
            + weights.gamma * self.verification_likelihood
            - weights.delta * self.surprise
            - weights.epsilon * self.homeostasis_cost
            - weights.zeta * self.purpose_risk
    }
}

/// Action scorer.
pub struct ActionScorer {
    weights: ScoreWeights,
    simulator: LatentSimulator,
}

impl ActionScorer {
    pub fn new(weights: ScoreWeights, simulator: LatentSimulator) -> Self {
        Self { weights, simulator }
    }

    /// Score a single action.
    pub fn score(
        &self,
        action: &Action,
        current: &LatentState,
        memory: &MemoryContext,
        homeostasis: &sophon_memory::HomeostasisState,
    ) -> ScoredAction {
        let components = self.compute_components(action, current, memory, homeostasis);
        let score = components.compute_total(&self.weights);

        ScoredAction {
            action: action.clone(),
            score,
            components,
        }
    }

    /// Score multiple candidates and return sorted.
    pub fn score_candidates(
        &self,
        candidates: &[Action],
        current: &LatentState,
        memory: &MemoryContext,
        homeostasis: &sophon_memory::HomeostasisState,
    ) -> Vec<ScoredAction> {
        let mut scored: Vec<_> = candidates
            .iter()
            .map(|a| self.score(a, current, memory, homeostasis))
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored
    }

    /// Select best action.
    pub fn select_best(
        &self,
        candidates: &[Action],
        current: &LatentState,
        memory: &MemoryContext,
        homeostasis: &sophon_memory::HomeostasisState,
    ) -> Option<ScoredAction> {
        self.score_candidates(candidates, current, memory, homeostasis)
            .into_iter()
            .next()
    }

    /// Compute individual score components.
    fn compute_components(
        &self,
        action: &Action,
        current: &LatentState,
        memory: &MemoryContext,
        homeostasis: &sophon_memory::HomeostasisState,
    ) -> ScoreComponents {
        // Simulate single-step rollout
        let rollout = self.simulator.rollout(current, &[action.clone()], memory);

        ScoreComponents {
            task_progress: self.estimate_task_progress(action, &rollout),
            retrieval_gain: self.estimate_retrieval_gain(action, memory),
            verification_likelihood: self.estimate_verification_likelihood(action),
            surprise: rollout.total_surprise,
            homeostasis_cost: self.estimate_homeostasis_cost(action, homeostasis),
            purpose_risk: self.estimate_purpose_risk(action),
        }
    }

    fn estimate_task_progress(
        &self,
        action: &Action,
        rollout: &crate::rollout::RolloutResult,
    ) -> f32 {
        match action {
            Action::Read { .. } => 0.3 - rollout.total_cost * 0.1,
            Action::Write { .. } => 0.5 - rollout.total_cost * 0.1,
            Action::Execute { .. } => 0.7 - rollout.total_cost * 0.2,
            Action::Plan { .. } => 0.4,
            Action::Learn { .. } => 0.2,
            Action::Verify { .. } => 0.6,
            Action::Noop => 0.0,
            Action::Custom { .. } => 0.5,
        }
    }

    fn estimate_retrieval_gain(&self, action: &Action, memory: &MemoryContext) -> f32 {
        match action {
            Action::Read { .. } => 0.8,
            Action::Plan { .. } => 0.6,
            Action::Learn { .. } => 0.4,
            _ => {
                let base = if memory.active_facts.is_empty() {
                    0.5
                } else {
                    0.3
                };
                base + memory.recent_episodes.len() as f32 * 0.05
            }
        }
    }

    fn estimate_verification_likelihood(&self, action: &Action) -> f32 {
        match action {
            Action::Verify { .. } => 0.95,
            Action::Read { .. } => 0.8,
            Action::Write { .. } => 0.7,
            Action::Execute { .. } => 0.6,
            _ => 0.5,
        }
    }

    fn estimate_homeostasis_cost(
        &self,
        action: &Action,
        homeostasis: &sophon_memory::HomeostasisState,
    ) -> f32 {
        let base_cost = homeostasis.homeostasis_cost();

        let action_cost = match action {
            Action::Execute { .. } => 0.3,
            Action::Plan { subgoals } => subgoals.len() as f32 * 0.05,
            Action::Learn { .. } => 0.4,
            _ => 0.1,
        };

        (base_cost + action_cost).min(1.0)
    }

    fn estimate_purpose_risk(&self, action: &Action) -> f32 {
        match action {
            Action::Write { .. } => 0.3,   // Risk of overwriting
            Action::Execute { .. } => 0.4, // Risk of side effects
            Action::Noop => 0.0,
            _ => 0.1,
        }
    }

    /// Adjust weights based on context.
    pub fn adjust_weights(&mut self, adjustment: ScoreWeights) {
        self.weights.alpha = (self.weights.alpha + adjustment.alpha).clamp(0.0, 1.0);
        self.weights.beta = (self.weights.beta + adjustment.beta).clamp(0.0, 1.0);
        self.weights.gamma = (self.weights.gamma + adjustment.gamma).clamp(0.0, 1.0);
        self.weights.delta = (self.weights.delta + adjustment.delta).clamp(0.0, 1.0);
        self.weights.epsilon = (self.weights.epsilon + adjustment.epsilon).clamp(0.0, 1.0);
        self.weights.zeta = (self.weights.zeta + adjustment.zeta).clamp(0.0, 1.0);
    }

    /// Get current weights.
    pub fn weights(&self) -> &ScoreWeights {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sophon_config::D_MODEL;
    use sophon_memory::HomeostasisState;

    #[test]
    fn action_scoring() {
        let weights = ScoreWeights::default();
        let simulator = LatentSimulator::new(D_MODEL);
        let scorer = ActionScorer::new(weights, simulator);

        let action = Action::Read {
            target: "test".to_string(),
        };
        let current = LatentState::new(D_MODEL);
        let memory = MemoryContext {
            active_facts: vec![],
            recent_episodes: vec![],
        };
        let homeostasis = HomeostasisState {
            cpu_load: 0.5,
            memory_used: 0.3,
            io_pressure: 0.2,
            cache_miss_rate: 0.1,
            prediction_error: 0.1,
            timestamp: 1,
        };

        let scored = scorer.score(&action, &current, &memory, &homeostasis);
        assert!(scored.score > -1.0 && scored.score < 1.0);
    }

    #[test]
    fn score_candidate_selection() {
        let weights = ScoreWeights::default();
        let simulator = LatentSimulator::new(D_MODEL);
        let scorer = ActionScorer::new(weights, simulator);

        let candidates = vec![
            Action::Noop,
            Action::Read {
                target: "a".to_string(),
            },
            Action::Write {
                target: "b".to_string(),
                content: "c".to_string(),
            },
        ];

        let current = LatentState::new(D_MODEL);
        let memory = MemoryContext {
            active_facts: vec![],
            recent_episodes: vec![],
        };
        let homeostasis = HomeostasisState {
            cpu_load: 0.5,
            memory_used: 0.3,
            io_pressure: 0.2,
            cache_miss_rate: 0.1,
            prediction_error: 0.1,
            timestamp: 1,
        };

        let sorted = scorer.score_candidates(&candidates, &current, &memory, &homeostasis);
        assert_eq!(sorted.len(), 3);

        // Verify sorted by score
        for i in 1..sorted.len() {
            assert!(sorted[i - 1].score >= sorted[i].score);
        }
    }
}

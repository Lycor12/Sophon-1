//! Latent world model for mental simulation.
//!
//! Implements action-conditioned latent dynamics:
//! z_{t+1} = f(z_t, a_t, m_t)
//!
//! Enables counterfactual reasoning and surprise computation.

use sophon_config::{D_MODEL, HDC_DIM};
use sophon_core::hdc::l2_normalize;

/// Latent state representation.
#[derive(Debug, Clone)]
pub struct LatentState {
    pub z: Vec<f32>, // D_MODEL
    pub time_step: usize,
}

impl LatentState {
    pub fn new(d_model: usize) -> Self {
        Self {
            z: vec![0.0; d_model],
            time_step: 0,
        }
    }

    pub fn from_observation(obs: &[f32]) -> Self {
        let mut z = obs.to_vec();
        if z.len() < D_MODEL {
            z.resize(D_MODEL, 0.0);
        } else if z.len() > D_MODEL {
            z.truncate(D_MODEL);
        }
        Self { z, time_step: 0 }
    }

    /// Encode as HDC for retrieval.
    pub fn to_hdc(&self) -> Vec<f32> {
        // Simple projection to HDC_DIM
        let mut hv = vec![0.0f32; HDC_DIM];
        for (i, &val) in self.z.iter().enumerate() {
            let idx = i % HDC_DIM;
            hv[idx] += val;
        }
        l2_normalize(&mut hv);
        hv
    }
}

/// Action representation.
#[derive(Debug, Clone)]
pub enum Action {
    Noop,
    Read { target: String },
    Write { target: String, content: String },
    Execute { command: String },
    Plan { subgoals: Vec<String> },
    Learn { skill: String },
    Verify { claim: String },
    Custom { name: String, params: Vec<f32> },
}

impl Action {
    /// Encode action as latent vector.
    pub fn encode(&self) -> Vec<f32> {
        let mut a = vec![0.0f32; D_MODEL];

        match self {
            Action::Noop => a[0] = 1.0,
            Action::Read { .. } => a[1] = 1.0,
            Action::Write { .. } => a[2] = 1.0,
            Action::Execute { .. } => a[3] = 1.0,
            Action::Plan { .. } => a[4] = 1.0,
            Action::Learn { .. } => a[5] = 1.0,
            Action::Verify { .. } => a[6] = 1.0,
            Action::Custom { params, .. } => {
                for (i, &p) in params.iter().enumerate().take(D_MODEL) {
                    a[i] = p;
                }
            }
        }

        a
    }

    /// Get action name.
    pub fn name(&self) -> &str {
        match self {
            Action::Noop => "noop",
            Action::Read { .. } => "read",
            Action::Write { .. } => "write",
            Action::Execute { .. } => "execute",
            Action::Plan { .. } => "plan",
            Action::Learn { .. } => "learn",
            Action::Verify { .. } => "verify",
            Action::Custom { name, .. } => name,
        }
    }
}

/// Memory context (working memory state).
#[derive(Debug, Clone)]
pub struct MemoryContext {
    pub active_facts: Vec<String>,
    pub recent_episodes: Vec<usize>, // indices
}

/// Predicted outcome.
#[derive(Debug, Clone)]
pub struct PredictedOutcome {
    pub observation: Vec<f32>,
    pub cost: f32,
    pub probability: f32,
}

/// Latent simulator: action-conditioned state transition.
pub struct LatentSimulator {
    // Learned transition weights (simplified)
    w_state: Vec<Vec<f32>>,
    w_action: Vec<Vec<f32>>,
    w_memory: Vec<Vec<f32>>,
}

impl LatentSimulator {
    pub fn new(d_model: usize) -> Self {
        Self {
            w_state: vec![vec![0.0; d_model]; d_model],
            w_action: vec![vec![0.0; d_model]; d_model],
            w_memory: vec![vec![0.0; d_model]; d_model],
        }
    }

    /// Transition: z' = f(z, a, m)
    pub fn transition(
        &self,
        state: &LatentState,
        action: &Action,
        memory: &MemoryContext,
    ) -> LatentState {
        let a_enc = action.encode();
        let m_enc = self.encode_memory(memory);

        let mut next_z = vec![0.0; state.z.len()];

        for i in 0..state.z.len() {
            let state_contrib: f32 = state
                .z
                .iter()
                .enumerate()
                .map(|(j, &v)| v * self.w_state[i][j])
                .sum();
            let action_contrib: f32 = a_enc
                .iter()
                .enumerate()
                .map(|(j, &v)| v * self.w_action[i][j])
                .sum();
            let memory_contrib: f32 = m_enc
                .iter()
                .enumerate()
                .map(|(j, &v)| v * self.w_memory[i][j])
                .sum();

            next_z[i] = (state_contrib + action_contrib + memory_contrib).tanh();
        }

        LatentState {
            z: next_z,
            time_step: state.time_step + 1,
        }
    }

    /// Predict observation from latent state.
    pub fn predict_observation(&self, state: &LatentState) -> PredictedOutcome {
        // Simplified: direct mapping from latent to observation
        let obs = state.z.clone();
        let cost = self.estimate_cost(state);

        PredictedOutcome {
            observation: obs,
            cost,
            probability: 0.8, // Confidence
        }
    }

    /// Estimate cost of being in state.
    fn estimate_cost(&self, state: &LatentState) -> f32 {
        // Cost = ||z|| (deviation from zero)
        state.z.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }

    /// Compute surprise: mismatch between prediction and actual.
    pub fn compute_surprise(&self, predicted: &PredictedOutcome, actual: &[f32]) -> f32 {
        if predicted.observation.len() != actual.len() {
            return 1.0; // Maximum surprise
        }

        let mse: f32 = predicted
            .observation
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f32>()
            / actual.len() as f32;

        mse.sqrt()
    }

    /// Multi-step rollout.
    pub fn rollout(
        &self,
        initial: &LatentState,
        actions: &[Action],
        memory: &MemoryContext,
    ) -> RolloutResult {
        let mut trajectory = Vec::new();
        let mut current = initial.clone();
        let mut total_cost = 0.0;
        let mut total_surprise = 0.0;

        for (i, action) in actions.iter().enumerate() {
            let predicted = self.predict_observation(&current);
            current = self.transition(&current, action, memory);
            total_cost += predicted.cost;

            trajectory.push(RolloutStep {
                time_step: i,
                action: action.clone(),
                predicted: predicted.clone(),
                next_state: current.clone(),
            });
        }

        RolloutResult {
            trajectory,
            total_cost,
            total_surprise,
            final_state: current,
        }
    }

    /// Check if action is safe (no high-cost outcomes).
    pub fn is_safe(
        &self,
        initial: &LatentState,
        action: &Action,
        memory: &MemoryContext,
        threshold: f32,
    ) -> bool {
        let result = self.rollout(initial, &[action.clone()], memory);
        result.total_cost < threshold
    }

    /// Find best action from candidates.
    pub fn best_action(
        &self,
        state: &LatentState,
        candidates: &[Action],
        memory: &MemoryContext,
    ) -> Option<Action> {
        candidates
            .iter()
            .min_by(|a, b| {
                let cost_a = self.rollout(state, &[(*a).clone()], memory).total_cost;
                let cost_b = self.rollout(state, &[(*b).clone()], memory).total_cost;
                cost_a.partial_cmp(&cost_b).unwrap()
            })
            .cloned()
    }

    fn encode_memory(&self, memory: &MemoryContext) -> Vec<f32> {
        let mut enc = vec![0.0f32; D_MODEL];
        // Simplified: use number of active facts
        enc[0] = memory.active_facts.len() as f32 / 10.0;
        enc[1] = memory.recent_episodes.len() as f32 / 10.0;
        enc
    }

    /// Update from experience (online learning).
    pub fn update(
        &mut self,
        state: &LatentState,
        action: &Action,
        next: &LatentState,
        learning_rate: f32,
    ) {
        let a_enc = action.encode();

        for i in 0..state.z.len() {
            let error = next.z[i] - state.z[i].tanh();
            for j in 0..state.z.len() {
                self.w_state[i][j] += learning_rate * error * state.z[j];
            }
            for j in 0..a_enc.len() {
                self.w_action[i][j] += learning_rate * error * a_enc[j];
            }
        }
    }
}

/// Single step in rollout.
#[derive(Debug, Clone)]
pub struct RolloutStep {
    pub time_step: usize,
    pub action: Action,
    pub predicted: PredictedOutcome,
    pub next_state: LatentState,
}

/// Complete rollout result.
#[derive(Debug, Clone)]
pub struct RolloutResult {
    pub trajectory: Vec<RolloutStep>,
    pub total_cost: f32,
    pub total_surprise: f32,
    pub final_state: LatentState,
}

/// Full trajectory for storage.
pub struct RolloutTrajectory {
    pub initial: LatentState,
    pub actions: Vec<Action>,
    pub result: RolloutResult,
    pub actual: Option<Vec<f32>>, // If executed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_state_creation() {
        let state = LatentState::new(D_MODEL);
        assert_eq!(state.z.len(), D_MODEL);
    }

    #[test]
    fn action_encoding() {
        let read = Action::Read {
            target: "file".to_string(),
        };
        let write = Action::Write {
            target: "file".to_string(),
            content: "data".to_string(),
        };

        let r_enc = read.encode();
        let w_enc = write.encode();

        assert_ne!(r_enc[1], 0.0);
        assert_ne!(w_enc[2], 0.0);
    }

    #[test]
    fn simulator_transition() {
        let sim = LatentSimulator::new(D_MODEL);
        let state = LatentState::new(D_MODEL);
        let action = Action::Noop;
        let memory = MemoryContext {
            active_facts: vec![],
            recent_episodes: vec![],
        };

        let next = sim.transition(&state, &action, &memory);
        assert_eq!(next.time_step, 1);
    }

    #[test]
    fn rollout_computation() {
        let sim = LatentSimulator::new(D_MODEL);
        let initial = LatentState::new(D_MODEL);
        let actions = vec![Action::Noop, Action::Noop];
        let memory = MemoryContext {
            active_facts: vec![],
            recent_episodes: vec![],
        };

        let result = sim.rollout(&initial, &actions, &memory);
        assert_eq!(result.trajectory.len(), 2);
    }

    #[test]
    fn surprise_computation() {
        let sim = LatentSimulator::new(D_MODEL);
        let predicted = PredictedOutcome {
            observation: vec![1.0; 10],
            cost: 0.5,
            probability: 0.8,
        };
        let actual = vec![1.1; 10];

        let surprise = sim.compute_surprise(&predicted, &actual);
        assert!(surprise > 0.0 && surprise < 1.0);
    }
}

//! sophon-memory — Persistent multi-modal memory for AGI.
#![forbid(unsafe_code)]

use sophon_config::HDC_DIM;
use sophon_core::hdc::{bind, bundle, circular_conv, l2_normalize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Helper: normalize a slice and return the normalized vector.
fn l2_normalize_vec(v: &[f32]) -> Vec<f32> {
    let mut result = v.to_vec();
    l2_normalize(&mut result);
    result
}

pub mod episodic;
pub mod interoceptive;
pub mod procedural;
pub mod semantic;
pub mod working;

pub use episodic::{Episode, EpisodicMemory, TemporalBinding};
pub use interoceptive::{HomeostasisState, InteroceptiveMemory, ResourceMetric};
pub use procedural::{ActionPattern, ProceduralMemory, Skill};
pub use semantic::{Fact, QueryResult, SemanticMemory, Triple};
pub use working::{WorkingEntry, WorkingMemory};

/// Unified memory system combining all memory types.
pub struct UnifiedMemory {
    pub episodic: EpisodicMemory,
    pub semantic: SemanticMemory,
    pub procedural: ProceduralMemory,
    pub working: WorkingMemory,
    pub interoceptive: InteroceptiveMemory,
}

impl UnifiedMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            episodic: EpisodicMemory::new(capacity),
            semantic: SemanticMemory::new(capacity * 2),
            procedural: ProceduralMemory::new(capacity),
            working: WorkingMemory::new(32), // Small active context
            interoceptive: InteroceptiveMemory::new(),
        }
    }

    /// Record a complete experience across all memory systems.
    pub fn record_experience(
        &mut self,
        perception: &[f32],
        action: Option<&str>,
        outcome: &[f32],
        homeostasis: &HomeostasisState,
    ) {
        let timestamp = current_timestamp();

        // Episodic: store the sequence
        let episode = Episode {
            timestamp,
            perception_hv: l2_normalize_vec(perception),
            action: action.map(|s| s.to_string()),
            outcome_hv: l2_normalize_vec(outcome),
            surprise: homeostasis.prediction_error,
        };
        self.episodic.record(episode);

        // Semantic: extract facts
        if let Some(act) = action {
            let fact = Fact::new(
                &format!("action:{}", act),
                "produces",
                &format!("{:?}", outcome),
            );
            self.semantic.store(fact);
        }

        // Interoceptive: record resource state
        self.interoceptive.record_state(homeostasis.clone());

        // Working: keep in active context
        self.working.push(WorkingEntry {
            content_hv: l2_normalize_vec(perception),
            timestamp,
            access_count: 1,
        });
    }

    /// Query all memory systems with a unified HDC query.
    pub fn query(&self, query_hv: &[f32]) -> UnifiedQueryResult {
        UnifiedQueryResult {
            episodic: self.episodic.retrieve_similar(query_hv, 5),
            semantic: self.semantic.query_by_content(query_hv, 5),
            procedural: self
                .procedural
                .find_matching(query_hv, 5)
                .into_iter()
                .cloned()
                .collect(),
            interoceptive: self.interoceptive.current_state(),
        }
    }

    /// Get current self-model (interoceptive + recent episodic).
    pub fn self_model(&self) -> SelfModel {
        SelfModel {
            current_state: self.interoceptive.current_state(),
            recent_history: self.episodic.recent(10),
            active_goals: self.working.get_goals(),
        }
    }
}

pub struct UnifiedQueryResult {
    pub episodic: Vec<Episode>,
    pub semantic: Vec<QueryResult>,
    pub procedural: Vec<ActionPattern>,
    pub interoceptive: Option<HomeostasisState>,
}

pub struct SelfModel {
    pub current_state: Option<HomeostasisState>,
    pub recent_history: Vec<Episode>,
    pub active_goals: Vec<String>,
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_memory_creation() {
        let memory = UnifiedMemory::new(100);
        assert_eq!(memory.working.capacity(), 32);
    }

    #[test]
    fn record_and_query() {
        let mut memory = UnifiedMemory::new(50);

        let perception = vec![1.0; HDC_DIM];
        let outcome = vec![0.5; HDC_DIM];
        let homeostasis = HomeostasisState {
            cpu_load: 0.5,
            memory_used: 0.3,
            io_pressure: 0.2,
            cache_miss_rate: 0.1,
            prediction_error: 0.1,
            timestamp: 0,
        };

        memory.record_experience(&perception, Some("test_action"), &outcome, &homeostasis);

        let query = memory.query(&perception);
        assert!(!query.episodic.is_empty());
    }
}

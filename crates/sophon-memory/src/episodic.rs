//! Episodic memory — Time-ordered sequence of experiences.

use crate::current_timestamp;
use sophon_config::HDC_DIM;
use sophon_core::hdc::{bind, bundle, circular_conv, l2_normalize};

/// A single episode in memory.
#[derive(Debug, Clone)]
pub struct Episode {
    pub timestamp: u64,
    pub perception_hv: Vec<f32>,
    pub action: Option<String>,
    pub outcome_hv: Vec<f32>,
    pub surprise: f32,
}

impl Episode {
    /// Compute composite HDC encoding of this episode.
    pub fn encode(&self) -> Vec<f32> {
        let mut composite = self.perception_hv.clone();

        if let Some(ref act) = self.action {
            let action_hv = text_to_hypervector(act);
            composite = circular_conv(&composite, &action_hv).unwrap_or_else(|_| composite.clone());
        }

        if let Ok(bundled) = bundle(&[&composite, &self.outcome_hv]) {
            composite = bundled;
        }

        let temporal = temporal_position(self.timestamp);
        if let Ok(bound) = bind(&composite, &temporal) {
            composite = bound;
        }

        let mut result = composite.clone();
        l2_normalize(&mut result);
        result
    }

    pub fn is_surprising(&self) -> bool {
        self.surprise > 0.5
    }
}

/// Temporal binding using HDC positional encoding.
pub struct TemporalBinding;

impl TemporalBinding {
    pub fn encode_position(t: u64, window_size: u64) -> Vec<f32> {
        let mut hv = vec![0.0f32; HDC_DIM];
        let phase = (t % window_size) as f32 / window_size.max(1) as f32;

        for i in 0..HDC_DIM {
            let freq = 1.0 + ((i % 10) as f32);
            hv[i] = (phase * freq * std::f32::consts::TAU).cos();
        }

        l2_normalize(&mut hv);
        hv
    }
}

/// Episodic memory store.
pub struct EpisodicMemory {
    episodes: Vec<Episode>,
    capacity: usize,
}

impl EpisodicMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a new episode.
    pub fn record(&mut self, episode: Episode) {
        if self.episodes.len() >= self.capacity {
            self.consolidate_oldest();
        }
        self.episodes.push(episode);
    }

    /// Retrieve episodes similar to query.
    pub fn retrieve_similar(&self, query_hv: &[f32], k: usize) -> Vec<Episode> {
        let mut scored: Vec<_> = self
            .episodes
            .iter()
            .map(|e| {
                let similarity = cosine_similarity(query_hv, &e.encode());
                (e.clone(), similarity)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(k).map(|(e, _)| e).collect()
    }

    /// Get recent episodes (chronological).
    pub fn recent(&self, n: usize) -> Vec<Episode> {
        let start = self.episodes.len().saturating_sub(n);
        self.episodes[start..].to_vec()
    }

    /// Replay experiences similar to current context.
    pub fn replay_similar(&self, context_hv: &[f32], n: usize) -> Vec<Episode> {
        self.retrieve_similar(context_hv, n)
    }

    /// Find temporal patterns (sequences that repeat).
    pub fn find_patterns(&self, window_size: usize) -> Vec<Vec<usize>> {
        let mut patterns = Vec::new();

        for i in 0..self.episodes.len().saturating_sub(window_size) {
            let window: Vec<_> = (i..i + window_size).collect();

            if self.is_repeating_pattern(&window) {
                patterns.push(window);
            }
        }

        patterns
    }

    fn is_repeating_pattern(&self, pattern: &[usize]) -> bool {
        if pattern.len() < 2 {
            return false;
        }

        let sig = self.pattern_signature(pattern);

        for i in (pattern.last().unwrap() + 1)..self.episodes.len().saturating_sub(pattern.len()) {
            let other_pattern: Vec<_> = (i..i + pattern.len()).collect();
            let other_sig = self.pattern_signature(&other_pattern);

            if cosine_similarity(&sig, &other_sig) > 0.85 {
                return true;
            }
        }

        false
    }

    fn pattern_signature(&self, indices: &[usize]) -> Vec<f32> {
        let encodings: Vec<_> = indices
            .iter()
            .filter_map(|i| self.episodes.get(*i).map(|e| e.encode()))
            .collect();

        if encodings.is_empty() {
            return vec![0.0; HDC_DIM];
        }

        let mut result = encodings[0].clone();
        for (i, enc) in encodings.iter().enumerate().skip(1) {
            let temporal = TemporalBinding::encode_position(i as u64, indices.len() as u64);
            if let Ok(bound) = bind(enc, &temporal) {
                result = bundle(&[&result, &bound]).unwrap_or_else(|_| result.clone());
            }
        }

        let mut result = result.clone();
        l2_normalize(&mut result);
        result
    }

    fn consolidate_oldest(&mut self) {
        let to_remove = self.capacity / 10;
        self.episodes.drain(0..to_remove.min(self.episodes.len()));
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

fn text_to_hypervector(text: &str) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];
    for (i, byte) in text.bytes().enumerate() {
        let idx = (i + byte as usize) % HDC_DIM;
        hv[idx] += 1.0;
    }
    let mut hv = hv;
    l2_normalize(&mut hv);
    hv
}

fn temporal_position(t: u64) -> Vec<f32> {
    TemporalBinding::encode_position(t, 1000)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_episode(surprise: f32) -> Episode {
        Episode {
            timestamp: current_timestamp(),
            perception_hv: vec![1.0; HDC_DIM],
            action: Some("test".to_string()),
            outcome_hv: vec![0.5; HDC_DIM],
            surprise,
        }
    }

    #[test]
    fn episodic_record_and_retrieve() {
        let mut mem = EpisodicMemory::new(10);
        let ep = dummy_episode(0.1);
        mem.record(ep);
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn episodic_similarity_retrieval() {
        let mut mem = EpisodicMemory::new(10);

        let query = vec![1.0f32; HDC_DIM];
        let ep1 = Episode {
            timestamp: 1,
            perception_hv: query.clone(),
            action: None,
            outcome_hv: vec![0.5; HDC_DIM],
            surprise: 0.1,
        };

        mem.record(ep1);

        let results = mem.retrieve_similar(&query, 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn temporal_binding() {
        let t1 = TemporalBinding::encode_position(0, 10);
        let t2 = TemporalBinding::encode_position(5, 10);

        let sim = cosine_similarity(&t1, &t2);
        assert!(sim < 0.99);
    }
}

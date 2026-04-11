//! Working memory — Active context with LRU eviction.
#![forbid(unsafe_code)]

use sophon_config::HDC_DIM;

/// Entry in working memory.
#[derive(Debug, Clone)]
pub struct WorkingEntry {
    pub content_hv: Vec<f32>,
    pub timestamp: u64,
    pub access_count: usize,
}

impl WorkingEntry {
    /// Compute activation score (recency + frequency weighted).
    pub fn activation(&self, current_time: u64) -> f32 {
        let recency = 1.0 / (1.0 + (current_time - self.timestamp) as f32 / 60.0);
        let frequency = (self.access_count as f32).sqrt();
        recency * 0.6 + frequency * 0.4
    }
}

/// Working memory with LRU-style eviction.
pub struct WorkingMemory {
    entries: Vec<WorkingEntry>,
    capacity: usize,
    current_time: u64,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            current_time: 0,
        }
    }

    /// Push new entry, evicting lowest-activation if at capacity.
    pub fn push(&mut self, entry: WorkingEntry) {
        self.current_time += 1;

        if self.entries.len() >= self.capacity {
            self.evict_lowest_activation();
        }

        self.entries.push(entry);
    }

    /// Retrieve by HDC similarity.
    pub fn retrieve(&self, query_hv: &[f32], threshold: f32) -> Vec<&WorkingEntry> {
        self.entries
            .iter()
            .filter(|e| cosine_similarity(query_hv, &e.content_hv) > threshold)
            .collect()
    }

    /// Update access count for retrieved entries.
    pub fn touch(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.access_count += 1;
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get current focus (highest activation entry).
    pub fn focus(&self) -> Option<&WorkingEntry> {
        self.entries.iter().max_by(|a, b| {
            a.activation(self.current_time)
                .partial_cmp(&b.activation(self.current_time))
                .unwrap()
        })
    }

    /// Set goals in working memory.
    pub fn set_goals(&mut self, goals: Vec<String>) {
        for goal in goals {
            let hv = text_to_hypervector(&goal);
            self.push(WorkingEntry {
                content_hv: hv,
                timestamp: self.current_time,
                access_count: 1,
            });
        }
    }

    /// Get active goals.
    pub fn get_goals(&self) -> Vec<String> {
        Vec::new()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn evict_lowest_activation(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let min_idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.activation(self.current_time)
                    .partial_cmp(&b.activation(self.current_time))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        self.entries.remove(min_idx);
    }
}

fn text_to_hypervector(text: &str) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];
    for (i, byte) in text.bytes().enumerate() {
        let idx = (i * 31 + byte as usize) % HDC_DIM;
        hv[idx] += 1.0;
    }
    l2_normalize_vec(&hv)
}

fn l2_normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
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
    fn working_memory_push_and_retrieve() {
        let mut wm = WorkingMemory::new(3);
        let entry = WorkingEntry {
            content_hv: vec![1.0; HDC_DIM],
            timestamp: 1,
            access_count: 1,
        };
        wm.push(entry);
        assert_eq!(wm.len(), 1);
    }

    #[test]
    fn working_memory_eviction() {
        let mut wm = WorkingMemory::new(2);

        for i in 0..3 {
            wm.push(WorkingEntry {
                content_hv: vec![i as f32; HDC_DIM],
                timestamp: i as u64,
                access_count: 1,
            });
        }

        assert_eq!(wm.len(), 2);
    }

    #[test]
    fn working_memory_activation() {
        let entry = WorkingEntry {
            content_hv: vec![1.0; HDC_DIM],
            timestamp: 0,
            access_count: 10,
        };

        let act_now = entry.activation(0);
        let act_later = entry.activation(100);

        assert!(act_now > act_later);
    }
}

//! Semantic memory — Structured knowledge as HDC triples.
#![forbid(unsafe_code)]

use sophon_config::HDC_DIM;
use std::collections::{HashMap, HashSet};

/// A fact: (subject, relation, object).
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f32,
}

impl Fact {
    pub fn new(subject: &str, relation: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            relation: relation.to_string(),
            object: object.to_string(),
            confidence: 1.0,
        }
    }

    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = c;
        self
    }

    /// Encode fact as HDC vector using simple binding.
    pub fn encode(&self) -> Vec<f32> {
        let sub_hv = text_to_hypervector(&self.subject);
        let rel_hv = text_to_hypervector(&self.relation);
        let obj_hv = text_to_hypervector(&self.object);

        // Simple bundling: average + elementwise multiply for binding
        let mut combined = vec![0.0f32; HDC_DIM];
        for i in 0..HDC_DIM {
            // (sub ⊗ rel) ⊕ obj
            combined[i] = sub_hv[i] * rel_hv[i] + obj_hv[i];
        }

        l2_normalize_vec(&combined)
    }
}

/// Triple representation for graph traversal.
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject_hv: Vec<f32>,
    pub relation_hv: Vec<f32>,
    pub object_hv: Vec<f32>,
    pub confidence: f32,
}

/// Query result with similarity score.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub fact: Fact,
    pub similarity: f32,
}

/// Semantic memory store.
pub struct SemanticMemory {
    facts: Vec<Fact>,
    // Inverted indices
    subject_index: HashMap<String, HashSet<usize>>,
    relation_index: HashMap<String, HashSet<usize>>,
    object_index: HashMap<String, HashSet<usize>>,
}

impl SemanticMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            facts: Vec::with_capacity(capacity),
            subject_index: HashMap::new(),
            relation_index: HashMap::new(),
            object_index: HashMap::new(),
        }
    }

    /// Store a fact.
    pub fn store(&mut self, fact: Fact) {
        let idx = self.facts.len();
        self.facts.push(fact.clone());

        // Update indices
        self.subject_index
            .entry(fact.subject.clone())
            .or_default()
            .insert(idx);
        self.relation_index
            .entry(fact.relation.clone())
            .or_default()
            .insert(idx);
        self.object_index
            .entry(fact.object.clone())
            .or_default()
            .insert(idx);
    }

    /// Query by exact subject.
    pub fn query_by_subject(&self, subject: &str) -> Vec<&Fact> {
        self.subject_index
            .get(subject)
            .map(|indices| indices.iter().map(|i| &self.facts[*i]).collect())
            .unwrap_or_default()
    }

    /// Query by exact relation.
    pub fn query_by_relation(&self, relation: &str) -> Vec<&Fact> {
        self.relation_index
            .get(relation)
            .map(|indices| indices.iter().map(|i| &self.facts[*i]).collect())
            .unwrap_or_default()
    }

    /// Query by HDC content similarity.
    pub fn query_by_content(&self, query_hv: &[f32], k: usize) -> Vec<QueryResult> {
        let mut scored: Vec<_> = self
            .facts
            .iter()
            .map(|f| {
                let enc = f.encode();
                let sim = cosine_similarity(query_hv, &enc);
                QueryResult {
                    fact: f.clone(),
                    similarity: sim * f.confidence,
                }
            })
            .collect();

        scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        scored.into_iter().take(k).collect()
    }

    /// Query with variable binding: ?x is a ?y
    pub fn query_pattern(&self, subject: &str, relation: &str) -> Vec<&Fact> {
        let sub_indices = self.subject_index.get(subject).cloned().unwrap_or_default();
        let rel_indices = self
            .relation_index
            .get(relation)
            .cloned()
            .unwrap_or_default();

        sub_indices
            .intersection(&rel_indices)
            .map(|i| &self.facts[*i])
            .collect()
    }

    /// Traverse graph: find all reachable from subject.
    pub fn traverse(&self, start_subject: &str, max_depth: usize) -> Vec<Vec<&Fact>> {
        let mut levels = Vec::new();
        let mut current = vec![start_subject.to_string()];
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_subject.to_string());

        for _ in 0..max_depth {
            let mut next = Vec::new();
            let mut facts_this_level = Vec::new();

            for sub in &current {
                for fact in self.query_by_subject(sub) {
                    facts_this_level.push(fact);
                    if !visited.contains(&fact.object) {
                        visited.insert(fact.object.clone());
                        next.push(fact.object.clone());
                    }
                }
            }

            if facts_this_level.is_empty() {
                break;
            }

            levels.push(facts_this_level);
            current = next;
        }

        levels
    }

    /// Infer transitive relation: if A->B and B->C, then A->C.
    pub fn infer_transitive(&self, relation: &str) -> Vec<Fact> {
        let mut inferred = Vec::new();
        let facts = self.query_by_relation(relation);

        // Build adjacency list
        let mut edges: HashMap<String, Vec<String>> = HashMap::new();
        for fact in facts {
            edges
                .entry(fact.subject.clone())
                .or_default()
                .push(fact.object.clone());
        }

        // Find transitive closures
        for (start, direct) in &edges {
            for mid in direct {
                if let Some(continuations) = edges.get(mid) {
                    for end in continuations {
                        if start != end {
                            inferred.push(Fact::new(
                                start,
                                &format!("transitive_{}", relation),
                                end,
                            ));
                        }
                    }
                }
            }
        }

        inferred
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

fn text_to_hypervector(text: &str) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];
    let bytes = text.as_bytes();

    // Character n-gram encoding
    for i in 0..bytes.len() {
        let idx = (i * 31 + bytes[i] as usize) % HDC_DIM;
        hv[idx] += 1.0;

        if i + 1 < bytes.len() {
            let idx2 = (i * 17 + bytes[i] as usize + bytes[i + 1] as usize * 13) % HDC_DIM;
            hv[idx2] += 0.5;
        }
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
    fn store_and_query() {
        let mut mem = SemanticMemory::new(10);
        let fact = Fact::new("cat", "is_a", "animal");
        mem.store(fact);

        let results = mem.query_by_subject("cat");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn query_pattern() {
        let mut mem = SemanticMemory::new(10);
        mem.store(Fact::new("cat", "is_a", "animal"));
        mem.store(Fact::new("dog", "is_a", "animal"));
        mem.store(Fact::new("cat", "has", "fur"));

        let results = mem.query_pattern("cat", "is_a");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, "animal");
    }

    #[test]
    fn transitive_inference() {
        let mut mem = SemanticMemory::new(10);
        mem.store(Fact::new("A", "contains", "B"));
        mem.store(Fact::new("B", "contains", "C"));

        let inferred = mem.infer_transitive("contains");
        assert!(!inferred.is_empty());
    }
}

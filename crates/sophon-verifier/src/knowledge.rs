//! Verified knowledge base — append-only library of Lean equations.
//!
//! Spec §4.2.6, §6.2.2: All verified Lean equations are stored in an
//! append-only knowledge base. New equations can only be added after
//! both Lean compilation AND human verification (during training) or
//! SPTF filter (during inference).
//!
//! The knowledge base serves as:
//! - Building blocks for harder problems (curriculum progression)
//! - Context for the translation swarm
//! - Ground truth for alignment verification
//!
//! # Novel technique: VKBS (Verified Knowledge Base with Structural indexing)
//!
//! Entries are indexed by:
//! 1. Curriculum level (difficulty tier)
//! 2. Topic tags (e.g. "algebra", "logic", "analysis")
//! 3. Dependency graph (which entries depend on which)
//!
//! This allows the teacher agent to efficiently query for building blocks
//! at the appropriate difficulty level.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Knowledge entry
// ---------------------------------------------------------------------------

/// A single verified equation in the knowledge base.
#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    /// Unique identifier (monotonically increasing).
    pub id: u64,
    /// Lean 4 source code of the verified theorem/definition.
    pub lean_source: String,
    /// Original natural language statement (if available).
    pub nl_statement: Option<String>,
    /// Theorem name in Lean.
    pub theorem_name: String,
    /// Curriculum level at which this was verified.
    pub level: u32,
    /// Topic tags for indexing.
    pub tags: Vec<String>,
    /// IDs of entries this one depends on (imports/uses).
    pub dependencies: Vec<u64>,
    /// Whether this was human-verified (training) or auto-verified (inference).
    pub human_verified: bool,
    /// Timestamp (as iteration number during training, or wall-clock).
    pub added_at: u64,
}

// ---------------------------------------------------------------------------
// Knowledge base
// ---------------------------------------------------------------------------

/// Append-only verified knowledge base.
///
/// Entries cannot be modified or removed after insertion. This is a
/// structural invariant enforced by the API — no method exposes
/// mutable references to existing entries.
pub struct KnowledgeBase {
    /// All entries in insertion order.
    entries: Vec<KnowledgeEntry>,
    /// Next ID to assign.
    next_id: u64,
    /// Index by theorem name → entry ID.
    name_index: HashMap<String, u64>,
    /// Index by level → entry IDs.
    level_index: HashMap<u32, Vec<u64>>,
    /// Index by tag → entry IDs.
    tag_index: HashMap<String, Vec<u64>>,
}

impl KnowledgeBase {
    /// Create an empty knowledge base.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            name_index: HashMap::new(),
            level_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Add a verified entry to the knowledge base.
    ///
    /// Returns the assigned entry ID. The entry is immutable after insertion.
    pub fn add(
        &mut self,
        lean_source: String,
        theorem_name: String,
        nl_statement: Option<String>,
        level: u32,
        tags: Vec<String>,
        dependencies: Vec<u64>,
        human_verified: bool,
        timestamp: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        // Update indices
        self.name_index.insert(theorem_name.clone(), id);
        self.level_index.entry(level).or_default().push(id);
        for tag in &tags {
            self.tag_index.entry(tag.clone()).or_default().push(id);
        }

        self.entries.push(KnowledgeEntry {
            id,
            lean_source,
            nl_statement,
            theorem_name,
            level,
            tags,
            dependencies,
            human_verified,
            added_at: timestamp,
        });

        id
    }

    /// Look up an entry by ID.
    pub fn get(&self, id: u64) -> Option<&KnowledgeEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Look up an entry by theorem name.
    pub fn get_by_name(&self, name: &str) -> Option<&KnowledgeEntry> {
        self.name_index.get(name).and_then(|&id| self.get(id))
    }

    /// Get all entries at a given curriculum level.
    pub fn at_level(&self, level: u32) -> Vec<&KnowledgeEntry> {
        self.level_index
            .get(&level)
            .map(|ids| ids.iter().filter_map(|&id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all entries with a given tag.
    pub fn with_tag(&self, tag: &str) -> Vec<&KnowledgeEntry> {
        self.tag_index
            .get(tag)
            .map(|ids| ids.iter().filter_map(|&id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get entries at or below a given level (building blocks for harder problems).
    pub fn building_blocks(&self, max_level: u32) -> Vec<&KnowledgeEntry> {
        self.entries
            .iter()
            .filter(|e| e.level <= max_level)
            .collect()
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the knowledge base is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries in insertion order.
    pub fn entries(&self) -> &[KnowledgeEntry] {
        &self.entries
    }

    /// Number of human-verified entries.
    pub fn human_verified_count(&self) -> usize {
        self.entries.iter().filter(|e| e.human_verified).count()
    }

    /// Number of distinct curriculum levels.
    pub fn n_levels(&self) -> usize {
        self.level_index.len()
    }

    /// All distinct tags.
    pub fn all_tags(&self) -> Vec<&str> {
        self.tag_index.keys().map(|s| s.as_str()).collect()
    }

    /// Generate a combined Lean import file for all entries up to a level.
    ///
    /// This produces a single Lean file that can be imported to get access
    /// to all verified theorems as building blocks.
    pub fn generate_import_file(&self, max_level: u32) -> String {
        let mut output = String::from("-- Auto-generated by sophon-verifier knowledge base\n\n");
        for entry in &self.entries {
            if entry.level <= max_level {
                output.push_str(&entry.lean_source);
                output.push('\n');
            }
        }
        output
    }

    /// Check if an entry's dependencies are all satisfied.
    pub fn dependencies_satisfied(&self, deps: &[u64]) -> bool {
        deps.iter().all(|&id| self.get(id).is_some())
    }

    /// Serialize to a simple text format (for persistence).
    pub fn serialize(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("SOPHON_KB_V1\n"));
        output.push_str(&format!("entries: {}\n", self.entries.len()));
        for entry in &self.entries {
            output.push_str(&format!(
                "---\nid: {}\nname: {}\nlevel: {}\nhuman: {}\ntimestamp: {}\ntags: {}\ndeps: {}\n",
                entry.id,
                entry.theorem_name,
                entry.level,
                entry.human_verified,
                entry.added_at,
                entry.tags.join(","),
                entry
                    .dependencies
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ));
            if let Some(nl) = &entry.nl_statement {
                output.push_str(&format!("nl: {nl}\n"));
            }
            output.push_str("lean:\n");
            output.push_str(&entry.lean_source);
            output.push_str("\n---END---\n");
        }
        output
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_retrieve() {
        let mut kb = KnowledgeBase::new();
        let id = kb.add(
            "theorem t : 1 = 1 := rfl".to_string(),
            "t".to_string(),
            Some("one equals one".to_string()),
            1,
            vec!["arithmetic".to_string()],
            vec![],
            true,
            0,
        );
        assert_eq!(id, 1);
        assert_eq!(kb.len(), 1);
        assert!(kb.get(id).is_some());
        assert_eq!(kb.get(id).unwrap().theorem_name, "t");
    }

    #[test]
    fn get_by_name() {
        let mut kb = KnowledgeBase::new();
        kb.add(
            "theorem add_zero : ∀ n, n + 0 = n := by simp".to_string(),
            "add_zero".to_string(),
            None,
            1,
            vec![],
            vec![],
            true,
            0,
        );
        let entry = kb.get_by_name("add_zero");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().level, 1);
    }

    #[test]
    fn level_query() {
        let mut kb = KnowledgeBase::new();
        kb.add(
            "th l1".into(),
            "l1".into(),
            None,
            1,
            vec![],
            vec![],
            true,
            0,
        );
        kb.add(
            "th l2".into(),
            "l2".into(),
            None,
            2,
            vec![],
            vec![],
            true,
            1,
        );
        kb.add(
            "th l1b".into(),
            "l1b".into(),
            None,
            1,
            vec![],
            vec![],
            true,
            2,
        );

        assert_eq!(kb.at_level(1).len(), 2);
        assert_eq!(kb.at_level(2).len(), 1);
        assert_eq!(kb.at_level(3).len(), 0);
    }

    #[test]
    fn tag_query() {
        let mut kb = KnowledgeBase::new();
        kb.add(
            "th".into(),
            "t1".into(),
            None,
            1,
            vec!["algebra".into()],
            vec![],
            true,
            0,
        );
        kb.add(
            "th".into(),
            "t2".into(),
            None,
            1,
            vec!["algebra".into(), "logic".into()],
            vec![],
            true,
            1,
        );

        assert_eq!(kb.with_tag("algebra").len(), 2);
        assert_eq!(kb.with_tag("logic").len(), 1);
        assert_eq!(kb.with_tag("geometry").len(), 0);
    }

    #[test]
    fn building_blocks_filters() {
        let mut kb = KnowledgeBase::new();
        kb.add("th".into(), "t1".into(), None, 1, vec![], vec![], true, 0);
        kb.add("th".into(), "t2".into(), None, 3, vec![], vec![], true, 1);
        kb.add("th".into(), "t3".into(), None, 2, vec![], vec![], true, 2);

        assert_eq!(kb.building_blocks(2).len(), 2); // level 1 and 2
        assert_eq!(kb.building_blocks(5).len(), 3); // all
    }

    #[test]
    fn append_only_monotonic_ids() {
        let mut kb = KnowledgeBase::new();
        let id1 = kb.add("th".into(), "t1".into(), None, 1, vec![], vec![], true, 0);
        let id2 = kb.add("th".into(), "t2".into(), None, 1, vec![], vec![], true, 1);
        let id3 = kb.add("th".into(), "t3".into(), None, 1, vec![], vec![], true, 2);
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    #[test]
    fn dependencies_check() {
        let mut kb = KnowledgeBase::new();
        let id1 = kb.add("th".into(), "base".into(), None, 1, vec![], vec![], true, 0);
        let _id2 = kb.add(
            "th".into(),
            "derived".into(),
            None,
            2,
            vec![],
            vec![id1],
            true,
            1,
        );

        assert!(kb.dependencies_satisfied(&[id1]));
        assert!(!kb.dependencies_satisfied(&[999]));
    }

    #[test]
    fn import_file_generation() {
        let mut kb = KnowledgeBase::new();
        kb.add(
            "theorem t1 : True := trivial\n".into(),
            "t1".into(),
            None,
            1,
            vec![],
            vec![],
            true,
            0,
        );
        kb.add(
            "theorem t2 : 1 = 1 := rfl\n".into(),
            "t2".into(),
            None,
            2,
            vec![],
            vec![],
            true,
            1,
        );
        kb.add(
            "theorem t3 : 2 = 2 := rfl\n".into(),
            "t3".into(),
            None,
            3,
            vec![],
            vec![],
            true,
            2,
        );

        let import = kb.generate_import_file(2);
        assert!(import.contains("t1"));
        assert!(import.contains("t2"));
        assert!(!import.contains("t3")); // Level 3 excluded
    }

    #[test]
    fn human_verified_count() {
        let mut kb = KnowledgeBase::new();
        kb.add("th".into(), "t1".into(), None, 1, vec![], vec![], true, 0);
        kb.add("th".into(), "t2".into(), None, 1, vec![], vec![], false, 1);
        assert_eq!(kb.human_verified_count(), 1);
    }

    #[test]
    fn serialization_roundtrip_format() {
        let mut kb = KnowledgeBase::new();
        kb.add(
            "theorem t : 1 = 1 := rfl".into(),
            "t".into(),
            Some("one equals one".into()),
            1,
            vec!["math".into()],
            vec![],
            true,
            42,
        );
        let serialized = kb.serialize();
        assert!(serialized.contains("SOPHON_KB_V1"));
        assert!(serialized.contains("entries: 1"));
        assert!(serialized.contains("name: t"));
        assert!(serialized.contains("nl: one equals one"));
    }
}

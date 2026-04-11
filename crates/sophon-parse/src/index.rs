//! Hierarchical codebase index — MSCA Level 4.
//!
//! Persistent index for retrieval of code entities at all scales.

use crate::ast::{AstRoot, FunctionDecl, StructDecl};
use crate::encode::{encode_ast_node, encode_graph_entity};
use std::collections::{HashMap, HashSet};

/// Entity type in codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Function,
    Struct,
    Enum,
    Typedef,
    Global,
    Macro,
}

/// Metadata for indexed entity.
#[derive(Debug, Clone)]
pub struct EntityMetadata {
    pub name: String,
    pub entity_type: EntityType,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub hv: Vec<f32>, // HDC encoding
    pub dependencies: Vec<String>,
    pub dependents: Vec<String>,
}

/// Query result from index.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub entity: EntityMetadata,
    pub similarity: f32,
}

/// Hierarchical chunked code index.
pub struct CodebaseIndex {
    entities: Vec<EntityMetadata>,
    by_name: HashMap<String, usize>,
    by_type: HashMap<EntityType, Vec<usize>>,
    by_file: HashMap<String, Vec<usize>>,
    chunks: Vec<CodeChunk>, // Hierarchical chunks
}

/// Chunk of code at a specific granularity.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub level: ChunkLevel,
    pub entities: Vec<usize>, // Indices into entities
    pub hv: Vec<f32>,
    pub file_path: String,
}

/// Level of code chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChunkLevel {
    File,     // Whole file
    Module,   // Logical module
    Class,    // Class-level
    Function, // Function-level
    Block,    // Block-level
}

impl CodebaseIndex {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            by_name: HashMap::new(),
            by_type: HashMap::new(),
            by_file: HashMap::new(),
            chunks: Vec::new(),
        }
    }

    /// Index a parsed file.
    pub fn index_file(&mut self, path: &str, ast: &AstRoot) {
        // Index functions
        for func in &ast.functions {
            self.index_function(path, func);
        }

        // Index structs
        for struct_decl in &ast.structs {
            self.index_struct(path, struct_decl);
        }

        // Index globals
        for global in &ast.globals {
            self.index_global(path, global);
        }

        // Build hierarchical chunks
        self.build_chunks(path, ast);
    }

    fn index_function(&mut self, path: &str, func: &FunctionDecl) {
        let hv = encode_graph_entity(&func.name, "function");
        let idx = self.entities.len();

        let meta = EntityMetadata {
            name: func.name.clone(),
            entity_type: EntityType::Function,
            file_path: path.to_string(),
            line_start: 0,
            line_end: 0,
            hv,
            dependencies: extract_function_deps(func),
            dependents: Vec::new(),
        };

        self.entities.push(meta);
        self.by_name.insert(func.name.clone(), idx);
        self.by_type
            .entry(EntityType::Function)
            .or_default()
            .push(idx);
        self.by_file.entry(path.to_string()).or_default().push(idx);
    }

    fn index_struct(&mut self, path: &str, struct_decl: &StructDecl) {
        if let Some(ref name) = struct_decl.name {
            let hv = encode_graph_entity(name, "struct");
            let idx = self.entities.len();

            let meta = EntityMetadata {
                name: name.clone(),
                entity_type: EntityType::Struct,
                file_path: path.to_string(),
                line_start: 0,
                line_end: 0,
                hv,
                dependencies: Vec::new(),
                dependents: Vec::new(),
            };

            self.entities.push(meta);
            self.by_name.insert(name.clone(), idx);
            self.by_type
                .entry(EntityType::Struct)
                .or_default()
                .push(idx);
            self.by_file.entry(path.to_string()).or_default().push(idx);
        }
    }

    fn index_global(&mut self, path: &str, global: &VarDecl) {
        let hv = encode_graph_entity(&global.name, "global");
        let idx = self.entities.len();

        let meta = EntityMetadata {
            name: global.name.clone(),
            entity_type: EntityType::Global,
            file_path: path.to_string(),
            line_start: 0,
            line_end: 0,
            hv,
            dependencies: Vec::new(),
            dependents: Vec::new(),
        };

        self.entities.push(meta);
        self.by_name.insert(global.name.clone(), idx);
        self.by_type
            .entry(EntityType::Global)
            .or_default()
            .push(idx);
        self.by_file.entry(path.to_string()).or_default().push(idx);
    }

    fn build_chunks(&mut self, path: &str, _ast: &AstRoot) {
        // Build file-level chunk
        let file_entities: Vec<_> = self.by_file.get(path).cloned().unwrap_or_default();

        let mut file_hv = vec![0.0f32; sophon_config::HDC_DIM];
        for &idx in &file_entities {
            for (i, &v) in self.entities[idx].hv.iter().enumerate() {
                file_hv[i] += v;
            }
        }
        file_hv = l2_normalize_vec(&file_hv);

        self.chunks.push(CodeChunk {
            level: ChunkLevel::File,
            entities: file_entities,
            hv: file_hv,
            file_path: path.to_string(),
        });
    }

    /// Query by name.
    pub fn query_by_name(&self, name: &str) -> Option<&EntityMetadata> {
        self.by_name.get(name).map(|&idx| &self.entities[idx])
    }

    /// Query by type.
    pub fn query_by_type(&self, entity_type: EntityType) -> Vec<&EntityMetadata> {
        self.by_type
            .get(&entity_type)
            .map(|indices| indices.iter().map(|&i| &self.entities[i]).collect())
            .unwrap_or_default()
    }

    /// Semantic query by HDC.
    pub fn query_semantic(&self, query_hv: &[f32], k: usize) -> Vec<QueryResult> {
        let mut scored: Vec<_> = self
            .entities
            .iter()
            .map(|e| {
                let sim = cosine_similarity(query_hv, &e.hv);
                (e, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
            .into_iter()
            .take(k)
            .map(|(e, sim)| QueryResult {
                entity: e.clone(),
                similarity: sim,
            })
            .collect()
    }

    /// Hierarchical retrieval: find relevant chunks.
    pub fn retrieve_chunks(
        &self,
        query_hv: &[f32],
        level: ChunkLevel,
        k: usize,
    ) -> Vec<&CodeChunk> {
        let mut scored: Vec<_> = self
            .chunks
            .iter()
            .filter(|c| c.level == level)
            .map(|c| {
                let sim = cosine_similarity(query_hv, &c.hv);
                (c, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(k).map(|(c, _)| c).collect()
    }

    /// Get dependencies of an entity.
    pub fn dependencies(&self, name: &str) -> Vec<&EntityMetadata> {
        self.query_by_name(name)
            .map(|e| {
                e.dependencies
                    .iter()
                    .filter_map(|dep| self.query_by_name(dep))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get dependents of an entity.
    pub fn dependents(&self, name: &str) -> Vec<&EntityMetadata> {
        self.query_by_name(name)
            .map(|e| {
                e.dependents
                    .iter()
                    .filter_map(|dep| self.query_by_name(dep))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

fn extract_function_deps(func: &FunctionDecl) -> Vec<String> {
    // Simplified: would analyze function body for calls
    func.params.iter().filter_map(|p| p.name.clone()).collect()
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
    fn index_creation() {
        let index = CodebaseIndex::new();
        assert!(index.is_empty());
    }

    #[test]
    fn index_file() {
        let mut index = CodebaseIndex::new();
        let ast = AstRoot::default();
        index.index_file("test.c", &ast);
        assert!(index.is_empty()); // No entities in empty AST
    }
}

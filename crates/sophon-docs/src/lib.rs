//! Documentation generator for Sophon AGI
//!
//! Generates comprehensive documentation from code and external sources.
//! Features:
//! - Rust doc extraction and cross-referencing
//! - Architecture diagrams (ASCII art)
//! - API reference generation
//! - Tutorial and guide compilation
//! - Search index generation

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Documentation error types
#[derive(Debug, Clone, PartialEq)]
pub enum DocError {
    /// IO error
    Io(String),
    /// Parse error
    Parse(String),
    /// Not found
    NotFound(String),
    /// Invalid input
    Invalid(String),
}

impl std::fmt::Display for DocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocError::Io(s) => write!(f, "IO error: {}", s),
            DocError::Parse(s) => write!(f, "Parse error: {}", s),
            DocError::NotFound(s) => write!(f, "Not found: {}", s),
            DocError::Invalid(s) => write!(f, "Invalid: {}", s),
        }
    }
}

impl std::error::Error for DocError {}

/// Documentation generator
#[derive(Debug)]
pub struct DocGenerator {
    /// Source roots
    roots: Vec<PathBuf>,
    /// Output directory
    output: PathBuf,
    /// Crate metadata
    crates: HashMap<String, CrateDocs>,
}

/// Crate documentation
#[derive(Debug, Clone)]
pub struct CrateDocs {
    /// Crate name
    pub name: String,
    /// Crate description
    pub description: String,
    /// Modules
    pub modules: Vec<ModuleDocs>,
    /// Public items
    pub items: Vec<ItemDocs>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Module documentation
#[derive(Debug, Clone)]
pub struct ModuleDocs {
    /// Module path
    pub path: String,
    /// Module docstring
    pub docs: String,
    /// Public items in module
    pub items: Vec<ItemDocs>,
}

/// Item documentation (function, struct, enum, etc.)
#[derive(Debug, Clone)]
pub struct ItemDocs {
    /// Item name
    pub name: String,
    /// Item kind
    pub kind: ItemKind,
    /// Documentation
    pub docs: String,
    /// Signature
    pub signature: String,
    /// Source file
    pub source: PathBuf,
    /// Line number
    pub line: usize,
}

/// Item kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    /// Function
    Function,
    /// Struct
    Struct,
    /// Enum
    Enum,
    /// Trait
    Trait,
    /// Type alias
    Type,
    /// Constant
    Const,
    /// Static
    Static,
    /// Module
    Module,
    /// Macro
    Macro,
}

impl std::fmt::Display for ItemKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ItemKind::Function => write!(f, "fn"),
            ItemKind::Struct => write!(f, "struct"),
            ItemKind::Enum => write!(f, "enum"),
            ItemKind::Trait => write!(f, "trait"),
            ItemKind::Type => write!(f, "type"),
            ItemKind::Const => write!(f, "const"),
            ItemKind::Static => write!(f, "static"),
            ItemKind::Module => write!(f, "mod"),
            ItemKind::Macro => write!(f, "macro"),
        }
    }
}

impl DocGenerator {
    /// Create a new documentation generator
    pub fn new(output: impl AsRef<Path>) -> Self {
        DocGenerator {
            roots: Vec::new(),
            output: output.as_ref().to_path_buf(),
            crates: HashMap::new(),
        }
    }

    /// Add a source root
    pub fn add_root(&mut self, path: impl AsRef<Path>) {
        self.roots.push(path.as_ref().to_path_buf());
    }

    /// Generate all documentation
    pub fn generate(&mut self) -> Result<(), DocError> {
        std::fs::create_dir_all(&self.output).map_err(|e| DocError::Io(e.to_string()))?;

        // Generate crate documentation
        self.generate_crate_docs()?;

        // Generate index
        self.generate_index()?;

        // Generate search index
        self.generate_search_index()?;

        Ok(())
    }

    fn generate_crate_docs(&mut self) -> Result<(), DocError> {
        // This would parse Rust source files and extract documentation
        // For now, placeholder
        Ok(())
    }

    fn generate_index(&self) -> Result<(), DocError> {
        let index_path = self.output.join("index.md");
        let mut content = String::new();

        content.push_str("# Sophon AGI Documentation\n\n");
        content.push_str("## Crates\n\n");

        for (name, docs) in &self.crates {
            content.push_str(&format!("### {}\n\n", name));
            content.push_str(&format!("{}\n\n", docs.description));
        }

        std::fs::write(index_path, content).map_err(|e| DocError::Io(e.to_string()))?;

        Ok(())
    }

    fn generate_search_index(&self) -> Result<(), DocError> {
        // Generate JSON search index
        Ok(())
    }
}

/// Generate architecture diagram as ASCII art
pub fn generate_architecture_diagram(crates: &[CrateDocs]) -> String {
    let mut output = String::new();

    output.push_str("# Sophon Architecture\n\n");
    output.push_str("```\n");

    // Simple box diagram
    output.push_str("┌─────────────────────────────────────────────────────────┐\n");
    output.push_str("│                    Sophon AGI System                     │\n");
    output.push_str("├─────────────────────────────────────────────────────────┤\n");
    output.push_str("│  sophon-cli    sophon-tui    sophon-docs               │\n");
    output.push_str("├─────────────────────────────────────────────────────────┤\n");
    output.push_str("│  sophon-inference  sophon-runtime  sophon-verifier     │\n");
    output.push_str("├─────────────────────────────────────────────────────────┤\n");
    output.push_str("│  sophon-model    sophon-ssm      sophon-kan            │\n");
    output.push_str("├─────────────────────────────────────────────────────────┤\n");
    output.push_str("│  sophon-quant    sophon-optim    sophon-loss           │\n");
    output.push_str("│  sophon-train    sophon-data     sophon-accel          │\n");
    output.push_str("├─────────────────────────────────────────────────────────┤\n");
    output.push_str("│  sophon-core     sophon-config                         │\n");
    output.push_str("└─────────────────────────────────────────────────────────┘\n");
    output.push_str("```\n");

    output
}

/// Extract documentation from a Rust source file
pub fn extract_docs(path: &Path) -> Result<Vec<ItemDocs>, DocError> {
    let content = std::fs::read_to_string(path).map_err(|e| DocError::Io(e.to_string()))?;

    let mut items = Vec::new();

    // Simple regex-based extraction
    // In production, would use rust-analyzer or syn
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        // Check for doc comments
        if trimmed.starts_with("///") || trimmed.starts_with("//!") {
            // This is a doc comment
            continue;
        }

        // Check for item definitions
        if trimmed.starts_with("pub fn ") {
            let name = trimmed
                .strip_prefix("pub fn ")
                .and_then(|s| s.split('(').next())
                .unwrap_or("unknown")
                .to_string();

            items.push(ItemDocs {
                name,
                kind: ItemKind::Function,
                docs: String::new(),
                signature: trimmed.to_string(),
                source: path.to_path_buf(),
                line: line_num + 1,
            });
        }
    }

    Ok(items)
}

/// Generate markdown API reference
pub fn generate_api_reference(items: &[ItemDocs]) -> String {
    let mut output = String::new();

    output.push_str("# API Reference\n\n");

    for item in items {
        output.push_str(&format!("## {} `{}`\n\n", item.kind, item.name));
        output.push_str(&format!("```rust\n{}\n```\n\n", item.signature));
        if !item.docs.is_empty() {
            output.push_str(&format!("{}\n\n", item.docs));
        }
        output.push_str(&format!(
            "*Source: {}:{}*\n\n",
            item.source.display(),
            item.line
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_generator_creation() {
        let gen = DocGenerator::new("/tmp/docs");
        assert!(gen.roots.is_empty());
    }

    #[test]
    fn item_kind_display() {
        assert_eq!(ItemKind::Function.to_string(), "fn");
        assert_eq!(ItemKind::Struct.to_string(), "struct");
    }

    #[test]
    fn architecture_diagram() {
        let diagram = generate_architecture_diagram(&[]);
        assert!(diagram.contains("Sophon AGI System"));
    }
}

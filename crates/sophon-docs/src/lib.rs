//! Documentation generator for Sophon AGI
//!
//! Generates beautiful HTML documentation from code and external sources.
//! Features:
//! - Rust doc extraction and cross-referencing
//! - Beautiful HTML output with syntax highlighting
//! - Search functionality
//! - Responsive design with dark/light mode
//! - Architecture diagrams
//! - API reference generation

#![forbid(unsafe_code)]

pub mod html;

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

    /// Generate all documentation (HTML output only)
    pub fn generate(&mut self) -> Result<(), DocError> {
        std::fs::create_dir_all(&self.output).map_err(|e| DocError::Io(e.to_string()))?;

        // Generate crate documentation
        self.generate_crate_docs()?;

        // Generate index
        self.generate_index()?;

        // Generate search index
        self.generate_search_index()?;

        // Generate beautiful HTML documentation
        let mut html_gen = html::HtmlGenerator::new(&self.output);
        html_gen.generate(&self.crates)?;

        Ok(())
    }

    /// Generate HTML documentation only
    pub fn generate_html(&mut self) -> Result<(), DocError> {
        // Generate crate documentation first
        self.generate_crate_docs()?;

        // Generate HTML
        let mut html_gen = html::HtmlGenerator::new(&self.output);
        html_gen.generate(&self.crates)?;

        Ok(())
    }

    /// Write individual crate and module documentation files
    fn write_crate_files(&self) -> Result<(), DocError> {
        use std::fs;

        for (crate_name, crate_docs) in &self.crates {
            // Create crate directory
            let crate_dir = self.output.join(crate_name);
            fs::create_dir_all(&crate_dir).map_err(|e| DocError::Io(e.to_string()))?;

            // Write crate overview
            let crate_path = crate_dir.join("index.md");
            let mut crate_content = String::new();
            crate_content.push_str(&format!("# Crate: {}\n\n", crate_name));
            crate_content.push_str(&format!("{}\n\n", crate_docs.description));

            // List modules
            if !crate_docs.modules.is_empty() {
                crate_content.push_str("## Modules\n\n");
                for module in &crate_docs.modules {
                    crate_content.push_str(&format!("- [{}]({}.md)\n", module.path, module.path));
                }
                crate_content.push('\n');
            }

            // List public items
            if !crate_docs.items.is_empty() {
                crate_content.push_str("## Public Items\n\n");
                for item in &crate_docs.items {
                    crate_content.push_str(&format!("### {} `{}`\n\n", item.kind, item.name));
                    if !item.docs.is_empty() {
                        crate_content.push_str(&item.docs);
                        crate_content.push_str("\n\n");
                    }
                    crate_content.push_str(&format!(
                        "**Source:** {}:{}\n\n",
                        item.source.display(),
                        item.line
                    ));
                }
            }

            fs::write(&crate_path, crate_content).map_err(|e| DocError::Io(e.to_string()))?;

            // Write individual module files
            for module in &crate_docs.modules {
                let module_path = crate_dir.join(format!("{}.md", module.path));
                let mut module_content = String::new();
                module_content.push_str(&format!("# Module: {}\n\n", module.path));

                if !module.docs.is_empty() {
                    module_content.push_str(&module.docs);
                    module_content.push_str("\n\n");
                }

                // List items in module
                if !module.items.is_empty() {
                    module_content.push_str("## Items\n\n");
                    for item in &module.items {
                        module_content.push_str(&format!("### {} `{}`\n\n", item.kind, item.name));
                        if !item.docs.is_empty() {
                            module_content.push_str(&item.docs);
                            module_content.push_str("\n\n");
                        }
                        module_content
                            .push_str(&format!("**Signature:** `{}`\n\n", item.signature));
                        module_content.push_str(&format!(
                            "**Source:** {}:{}\n\n",
                            item.source.display(),
                            item.line
                        ));
                    }
                }

                fs::write(&module_path, module_content).map_err(|e| DocError::Io(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn generate_crate_docs(&mut self) -> Result<(), DocError> {
        // Extract documentation from Rust source files
        use std::fs;
        use std::io::Read;

        for root in &self.roots {
            let crate_name = root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let mut crate_docs = CrateDocs {
                name: crate_name.clone(),
                description: format!("Documentation for {}", crate_name),
                modules: Vec::new(),
                items: Vec::new(),
                dependencies: Vec::new(),
            };

            // Scan for .rs files
            if let Ok(entries) = fs::read_dir(root.join("src")) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                        let mut file =
                            fs::File::open(&path).map_err(|e| DocError::Io(e.to_string()))?;
                        let mut content = String::new();
                        file.read_to_string(&mut content)
                            .map_err(|e| DocError::Io(e.to_string()))?;

                        // Extract doc comments and public items
                        let module_name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string();

                        let (docs, items) = Self::extract_docs_from_source(&content, &path);

                        crate_docs.modules.push(ModuleDocs {
                            path: module_name,
                            docs,
                            items,
                        });
                    }
                }
            }

            self.crates.insert(crate_name, crate_docs);
        }

        Ok(())
    }

    /// Extract documentation from Rust source content
    fn extract_docs_from_source(content: &str, source_path: &Path) -> (String, Vec<ItemDocs>) {
        let mut docs = String::new();
        let mut items = Vec::new();
        let mut current_doc = String::new();
        let mut line_num = 0;

        for line in content.lines() {
            line_num += 1;
            let trimmed = line.trim();

            if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                let doc_line = if trimmed.starts_with("/// ") {
                    &trimmed[4..]
                } else if trimmed.starts_with("///") {
                    &trimmed[3..]
                } else if trimmed.starts_with("//! ") {
                    &trimmed[4..]
                } else {
                    &trimmed[3..]
                };
                current_doc.push_str(doc_line);
                current_doc.push('\n');
            } else if trimmed.starts_with("pub fn ") || trimmed.starts_with("pub fn\t") {
                let name = trimmed
                    .split_whitespace()
                    .nth(2)
                    .and_then(|n| n.split('(').next())
                    .unwrap_or("unknown")
                    .to_string();
                items.push(ItemDocs {
                    name,
                    kind: ItemKind::Function,
                    docs: current_doc.clone().trim().to_string(),
                    signature: trimmed.to_string(),
                    source: source_path.to_path_buf(),
                    line: line_num,
                });
                current_doc.clear();
            } else if trimmed.starts_with("pub struct ") || trimmed.starts_with("pub struct\t") {
                let name = trimmed
                    .split_whitespace()
                    .nth(2)
                    .and_then(|n| n.split('{').next())
                    .and_then(|n| n.split('(').next())
                    .unwrap_or("unknown")
                    .to_string();
                items.push(ItemDocs {
                    name,
                    kind: ItemKind::Struct,
                    docs: current_doc.clone().trim().to_string(),
                    signature: trimmed.to_string(),
                    source: source_path.to_path_buf(),
                    line: line_num,
                });
                current_doc.clear();
            } else if trimmed.starts_with("pub enum ") || trimmed.starts_with("pub enum\t") {
                let name = trimmed
                    .split_whitespace()
                    .nth(2)
                    .and_then(|n| n.split('{').next())
                    .unwrap_or("unknown")
                    .to_string();
                items.push(ItemDocs {
                    name,
                    kind: ItemKind::Enum,
                    docs: current_doc.clone().trim().to_string(),
                    signature: trimmed.to_string(),
                    source: source_path.to_path_buf(),
                    line: line_num,
                });
                current_doc.clear();
            } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                // End of doc block
                if !current_doc.is_empty() && docs.is_empty() {
                    docs = current_doc.clone().trim().to_string();
                }
                current_doc.clear();
            }
        }

        (docs, items)
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

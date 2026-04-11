//! sophon-parse — Multi-Scale Code Abstraction (MSCA) for deep code comprehension.
//!
//! This crate implements the foundation for analyzing massive codebases (e.g., FFmpeg's
//! 1.2M lines) through a multi-level representation hierarchy:
//!
//! - Level 0: Raw bytes (existing)
//! - Level 1: Lexical tokens (sophon-parse/lexer)
//! - Level 2: AST nodes (sophon-parse/ast)
//! - Level 3: Semantic graphs (sophon-parse/graph)
//! - Level 4: Architectural patterns (sophon-parse/index)
//!
//! Each level is encoded into HDC hypervectors using the existing hdc module,
//! enabling content-addressable retrieval of code patterns at any scale.
//!
//! Novel techniques implemented:
//! - MSCA: Multi-Scale Code Abstraction — zoom from architecture to line
//! - HCRP: Hierarchical Chunked Retrieval with Persistence — codebase indexing
//! - ZPD: Zero-external-dependency Parsing — handwritten recursive descent

#![forbid(unsafe_code)]

pub mod ast;
pub mod encode;
pub mod graph;
pub mod index;
pub mod lexer;

pub use ast::{parse_c_file, parse_c_fragment, AstNode, Parser};
pub use encode::{encode_ast_node, encode_graph_entity, encode_token, MscaLevel};
pub use graph::{CallGraph, DataFlowGraph, TypeGraph};
pub use index::{CodebaseIndex, EntityMetadata, EntityType, QueryResult};
pub use lexer::{Lexer, Token, TokenKind};

/// Parse a C source file and build its full MSCA representation.
pub fn parse_c_msca(source: &str) -> Result<MscaRepresentation, ParseError> {
    let tokens = lexer::lex_c(source)?;
    let ast = ast::parse_tokens(&tokens)?;
    let graphs = graph::extract_graphs(&ast)?;
    let encoding = encode::encode_ast_to_hdc(&ast)?;

    Ok(MscaRepresentation {
        tokens,
        ast,
        graphs,
        encoding,
    })
}

/// Multi-scale representation of a single source file.
pub struct MscaRepresentation {
    pub tokens: Vec<Token>,
    pub ast: ast::AstRoot,
    pub graphs: graph::GraphBundle,
    pub encoding: encode::MscaEncoding,
}

/// Unified parse error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    LexError { line: usize, col: usize, msg: String },
    ParseError { line: usize, col: usize, msg: String },
    EncodingError(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::LexError { line, col, msg } => {
                write!(f, "Lexer error at {}:{}: {}", line, col, msg)
            }
            ParseError::ParseError { line, col, msg } => {
                write!(f, "Parse error at {}:{}: {}", line, col, msg)
            }
            ParseError::EncodingError(msg) => {
                write!(f, "Encoding error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_function() {
        let source = r#"
            int add(int a, int b) {
                return a + b;
            }
        "#;

        let result = parse_c_msca(source);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let msca = result.unwrap();
        assert!(!msca.tokens.is_empty());
        assert!(!msca.ast.functions.is_empty());
    }

    #[test]
    fn parse_with_preprocessor() {
        let source = r#"
            #include <stdio.h>
            #define MAX 100
            
            int main(void) {
                printf("Hello\n");
                return 0;
            }
        "#;

        let result = parse_c_msca(source);
        assert!(result.is_ok(), "Failed to parse with preprocessor: {:?}", result.err());
    }

    #[test]
    fn lex_error_reporting() {
        // Invalid escape sequence
        let source = r#"char s = '\xgg';"#;
        let result = lexer::lex_c(source);
        assert!(result.is_err());
    }
}

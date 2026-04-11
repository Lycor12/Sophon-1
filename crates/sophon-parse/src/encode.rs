//! HDC encoding for AST nodes — MSCA Level 4.
//!
//! Encodes multi-scale representations into HDC hypervectors.

use crate::ast::{AstNode, FunctionDecl, VarDecl};
use crate::lexer::Token;
use sophon_config::HDC_DIM;

/// MSCA encoding levels.
#[derive(Debug, Clone, Copy)]
pub enum MscaLevel {
    Token,   // Level 1: lexical
    Node,    // Level 2: AST node
    Graph,   // Level 3: semantic graph
    Pattern, // Level 4: architectural patterns
}

/// Encoded multi-scale representation.
#[derive(Debug, Clone)]
pub struct MscaEncoding {
    pub token_hv: Vec<f32>,
    pub node_hv: Vec<f32>,
    pub graph_hv: Vec<f32>,
    pub pattern_hv: Vec<f32>,
}

impl MscaEncoding {
    pub fn combined(&self) -> Vec<f32> {
        let mut combined = vec![0.0f32; HDC_DIM];
        for i in 0..HDC_DIM {
            combined[i] =
                (self.token_hv[i] + self.node_hv[i] + self.graph_hv[i] + self.pattern_hv[i]) / 4.0;
        }
        l2_normalize(&combined)
    }
}

/// Encode a token to HDC.
pub fn encode_token(token: &Token) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];

    // Hash token kind and text
    let kind_hash = hash_u64(token.kind as u64);
    let text_hash = hash_str(&token.text);

    // Combine hashes
    for i in 0..HDC_DIM {
        let idx1 = (i + kind_hash as usize) % HDC_DIM;
        let idx2 = (i + text_hash as usize) % HDC_DIM;
        hv[idx1] += 1.0;
        hv[idx2] += 0.5;
    }

    l2_normalize(&hv)
}

/// Encode AST node to HDC.
pub fn encode_ast_node(node: &AstNode) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];

    // Encode node type
    let node_type_hash = hash_str(&format!("{:?}", std::mem::discriminant(node)));

    for i in 0..HDC_DIM {
        let idx = (i + node_type_hash as usize) % HDC_DIM;
        hv[idx] += 1.0;
    }

    // Encode specific node content
    match node {
        AstNode::Function(f) => encode_function_content(&mut hv, f),
        AstNode::GlobalVar(v) => encode_var_content(&mut hv, v),
        _ => {}
    }

    l2_normalize(&hv)
}

fn encode_function_content(hv: &mut [f32], func: &FunctionDecl) {
    let name_hash = hash_str(&func.name);
    let type_hash = hash_str(&format!("{:?}", func.return_type));

    for i in 0..hv.len() {
        let idx1 = (i + name_hash as usize) % hv.len();
        let idx2 = (i + type_hash as usize) % hv.len();
        hv[idx1] += 0.5;
        hv[idx2] += 0.3;
    }
}

fn encode_var_content(hv: &mut [f32], var: &VarDecl) {
    let name_hash = hash_str(&var.name);
    let type_hash = hash_str(&format!("{:?}", var.ty));

    for i in 0..hv.len() {
        let idx1 = (i + name_hash as usize) % hv.len();
        let idx2 = (i + type_hash as usize) % hv.len();
        hv[idx1] += 0.5;
        hv[idx2] += 0.3;
    }
}

/// Encode graph entity to HDC.
pub fn encode_graph_entity(entity_name: &str, entity_type: &str) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];

    let name_hash = hash_str(entity_name);
    let type_hash = hash_str(entity_type);

    for i in 0..HDC_DIM {
        let idx1 = (i + name_hash as usize) % HDC_DIM;
        let idx2 = (i + type_hash as usize) % HDC_DIM;
        hv[idx1] += 1.0;
        hv[idx2] += 0.5;
    }

    l2_normalize(&hv)
}

/// Encode complete AST to multi-scale representation.
pub fn encode_ast_to_hdc(root: &crate::ast::AstRoot) -> Result<MscaEncoding, crate::ParseError> {
    // Token-level encoding: encode all tokens
    let mut token_hv = vec![0.0f32; HDC_DIM];

    // Node-level encoding: encode declarations
    let mut node_hv = vec![0.0f32; HDC_DIM];
    for decl in &root.declarations {
        let node_enc = encode_ast_node(decl);
        for i in 0..HDC_DIM {
            node_hv[i] += node_enc[i];
        }
    }

    // Graph-level encoding: encode function relationships
    let mut graph_hv = vec![0.0f32; HDC_DIM];
    for func in &root.functions {
        let func_enc = encode_graph_entity(&func.name, "function");
        for i in 0..HDC_DIM {
            graph_hv[i] += func_enc[i];
        }
    }

    // Pattern-level: encode architectural patterns
    let pattern_hv = encode_patterns(root);

    // Normalize all levels
    let token_enc = l2_normalize(&token_hv);
    let node_enc = l2_normalize(&node_hv);
    let graph_enc = l2_normalize(&graph_hv);
    let pattern_enc = l2_normalize(&pattern_hv);

    Ok(MscaEncoding {
        token_hv: token_enc,
        node_hv: node_enc,
        graph_hv: graph_enc,
        pattern_hv: pattern_enc,
    })
}

fn encode_patterns(root: &crate::ast::AstRoot) -> Vec<f32> {
    let mut hv = vec![0.0f32; HDC_DIM];

    // Pattern: number of functions
    let func_count = root.functions.len() as u64;
    let count_hash = hash_u64(func_count);

    // Pattern: has main function
    let has_main = root.functions.iter().any(|f| f.name == "main") as u64;
    let main_hash = hash_u64(has_main);

    for i in 0..HDC_DIM {
        let idx1 = (i + count_hash as usize) % HDC_DIM;
        let idx2 = (i + main_hash as usize) % HDC_DIM;
        hv[idx1] += 1.0;
        hv[idx2] += 0.5;
    }

    hv
}

fn hash_str(s: &str) -> u64 {
    let mut hash = 0u64;
    for (i, byte) in s.bytes().enumerate() {
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(byte as u64)
            .wrapping_add(i as u64);
    }
    hash
}

fn hash_u64(x: u64) -> u64 {
    // Simple hash function
    x.wrapping_mul(0x9e3779b97f4a7c15)
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_encoding() {
        let token = Token {
            kind: crate::lexer::TokenKind::Identifier,
            text: "test".to_string(),
            pos: crate::lexer::SourcePos::default(),
        };
        let hv = encode_token(&token);
        assert_eq!(hv.len(), HDC_DIM);
    }

    #[test]
    fn node_encoding() {
        let func = FunctionDecl {
            name: "main".to_string(),
            return_type: crate::ast::Type::Int,
            params: vec![],
            body: None,
            is_static: false,
            is_inline: false,
        };
        let node = AstNode::Function(func);
        let hv = encode_ast_node(&node);
        assert_eq!(hv.len(), HDC_DIM);
    }
}

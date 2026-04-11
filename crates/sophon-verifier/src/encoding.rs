//! NL → FOL → Lean encoding pipeline — Spec §4.2.4.
//!
//! Translates natural language statements into first-order logic (FOL)
//! intermediate representation, then generates Lean 4 syntax.
//!
//! # Novel technique: STFL (Staged Token-to-Formal-Logic)
//!
//! Rather than a monolithic NL→Lean neural translator, STFL decomposes
//! the translation into three deterministic stages with well-defined
//! intermediate representations. Each stage can be independently verified
//! and corrected:
//!
//! ```text
//! Stage 1: NL → Token tree (deterministic parse)
//! Stage 2: Token tree → FOL IR (pattern-matched translation)
//! Stage 3: FOL IR → Lean source (template generation)
//! ```
//!
//! Stages 1-2 use rule-based patterns for known mathematical constructs.
//! Unknown constructs produce an explicit `Unknown` node in the FOL IR,
//! signaling to the swarm that neural translation is needed.

// ---------------------------------------------------------------------------
// FOL intermediate representation
// ---------------------------------------------------------------------------

/// A first-order logic formula.
#[derive(Debug, Clone, PartialEq)]
pub enum FolExpr {
    /// Universal quantification: ∀ var : ty, body
    ForAll {
        var: String,
        ty: String,
        body: Box<FolExpr>,
    },
    /// Existential quantification: ∃ var : ty, body
    Exists {
        var: String,
        ty: String,
        body: Box<FolExpr>,
    },
    /// Implication: lhs → rhs
    Implies {
        lhs: Box<FolExpr>,
        rhs: Box<FolExpr>,
    },
    /// Conjunction: lhs ∧ rhs
    And {
        lhs: Box<FolExpr>,
        rhs: Box<FolExpr>,
    },
    /// Disjunction: lhs ∨ rhs
    Or {
        lhs: Box<FolExpr>,
        rhs: Box<FolExpr>,
    },
    /// Negation: ¬ inner
    Not { inner: Box<FolExpr> },
    /// Equality: lhs = rhs
    Eq {
        lhs: Box<FolExpr>,
        rhs: Box<FolExpr>,
    },
    /// Predicate application: P(args...)
    Pred { name: String, args: Vec<FolExpr> },
    /// Function application: f(args...)
    Func { name: String, args: Vec<FolExpr> },
    /// Variable reference
    Var(String),
    /// Numeric literal
    Num(i64),
    /// Boolean literal
    Bool(bool),
    /// Addition
    Add {
        lhs: Box<FolExpr>,
        rhs: Box<FolExpr>,
    },
    /// Multiplication
    Mul {
        lhs: Box<FolExpr>,
        rhs: Box<FolExpr>,
    },
    /// Unknown / needs neural translation
    Unknown { original: String },
}

impl FolExpr {
    /// Whether this expression contains any Unknown nodes.
    pub fn has_unknown(&self) -> bool {
        match self {
            Self::Unknown { .. } => true,
            Self::ForAll { body, .. } | Self::Exists { body, .. } | Self::Not { inner: body } => {
                body.has_unknown()
            }
            Self::Implies { lhs, rhs }
            | Self::And { lhs, rhs }
            | Self::Or { lhs, rhs }
            | Self::Eq { lhs, rhs }
            | Self::Add { lhs, rhs }
            | Self::Mul { lhs, rhs } => lhs.has_unknown() || rhs.has_unknown(),
            Self::Pred { args, .. } | Self::Func { args, .. } => {
                args.iter().any(|a| a.has_unknown())
            }
            Self::Var(_) | Self::Num(_) | Self::Bool(_) => false,
        }
    }
}

// ---------------------------------------------------------------------------
// FOL → Lean code generation
// ---------------------------------------------------------------------------

/// Generate Lean 4 syntax from a FOL expression.
pub fn fol_to_lean(expr: &FolExpr) -> String {
    match expr {
        FolExpr::ForAll { var, ty, body } => {
            format!("∀ ({var} : {ty}), {}", fol_to_lean(body))
        }
        FolExpr::Exists { var, ty, body } => {
            format!("∃ ({var} : {ty}), {}", fol_to_lean(body))
        }
        FolExpr::Implies { lhs, rhs } => {
            format!("({} → {})", fol_to_lean(lhs), fol_to_lean(rhs))
        }
        FolExpr::And { lhs, rhs } => {
            format!("({} ∧ {})", fol_to_lean(lhs), fol_to_lean(rhs))
        }
        FolExpr::Or { lhs, rhs } => {
            format!("({} ∨ {})", fol_to_lean(lhs), fol_to_lean(rhs))
        }
        FolExpr::Not { inner } => {
            format!("¬{}", fol_to_lean(inner))
        }
        FolExpr::Eq { lhs, rhs } => {
            format!("({} = {})", fol_to_lean(lhs), fol_to_lean(rhs))
        }
        FolExpr::Pred { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let arg_strs: Vec<String> = args.iter().map(fol_to_lean).collect();
                format!("({} {})", name, arg_strs.join(" "))
            }
        }
        FolExpr::Func { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let arg_strs: Vec<String> = args.iter().map(fol_to_lean).collect();
                format!("({} {})", name, arg_strs.join(" "))
            }
        }
        FolExpr::Var(name) => name.clone(),
        FolExpr::Num(n) => n.to_string(),
        FolExpr::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        FolExpr::Add { lhs, rhs } => {
            format!("({} + {})", fol_to_lean(lhs), fol_to_lean(rhs))
        }
        FolExpr::Mul { lhs, rhs } => {
            format!("({} * {})", fol_to_lean(lhs), fol_to_lean(rhs))
        }
        FolExpr::Unknown { original } => {
            format!("sorry /- UNKNOWN: {original} -/")
        }
    }
}

/// Generate a complete Lean 4 theorem statement with proof placeholder.
pub fn generate_lean_theorem(name: &str, statement: &FolExpr, proof_body: Option<&str>) -> String {
    let stmt = fol_to_lean(statement);
    let proof = proof_body.unwrap_or("sorry");
    format!("theorem {name} : {stmt} := by\n  {proof}\n")
}

/// Generate a Lean 4 definition.
pub fn generate_lean_def(name: &str, ty: &str, body: &str) -> String {
    format!("def {name} : {ty} := {body}\n")
}

/// Generate a Lean 4 file header with imports.
pub fn generate_lean_header(imports: &[&str]) -> String {
    let mut header = String::new();
    for imp in imports {
        header.push_str(&format!("import {imp}\n"));
    }
    if !imports.is_empty() {
        header.push('\n');
    }
    header
}

// ---------------------------------------------------------------------------
// NL → FOL pattern matching (rule-based Stage 2)
// ---------------------------------------------------------------------------

/// Attempt to parse a simple natural language mathematical statement into FOL.
///
/// This is a rule-based parser for common mathematical patterns:
/// - "for all X, P(X)" → ForAll
/// - "there exists X such that P(X)" → Exists
/// - "if P then Q" → Implies
/// - "X = Y" → Eq
/// - "X + Y = Z" → Eq(Add, ...)
///
/// Returns `FolExpr::Unknown` for anything it cannot parse.
pub fn nl_to_fol(sentence: &str) -> FolExpr {
    let s = sentence.trim().to_lowercase();

    // Pattern: "for all X, ..."
    if s.starts_with("for all ") || s.starts_with("for every ") {
        let rest = if s.starts_with("for all ") {
            &s[8..]
        } else {
            &s[10..]
        };
        if let Some(comma_pos) = rest.find(',') {
            let var = rest[..comma_pos].trim().to_string();
            let body_str = rest[comma_pos + 1..].trim();
            let body = nl_to_fol(body_str);
            return FolExpr::ForAll {
                var,
                ty: "ℕ".to_string(), // Default type
                body: Box::new(body),
            };
        }
    }

    // Pattern: "there exists X such that ..."
    if s.starts_with("there exists ") {
        let rest = &s[13..];
        let sep = rest.find("such that").or_else(|| rest.find(","));
        if let Some(pos) = sep {
            let var = rest[..pos].trim().to_string();
            let body_start = if rest[pos..].starts_with("such that") {
                pos + 9
            } else {
                pos + 1
            };
            let body = nl_to_fol(rest[body_start..].trim());
            return FolExpr::Exists {
                var,
                ty: "ℕ".to_string(),
                body: Box::new(body),
            };
        }
    }

    // Pattern: "if P then Q"
    if s.starts_with("if ") {
        if let Some(then_pos) = s.find(" then ") {
            let lhs = nl_to_fol(&s[3..then_pos]);
            let rhs = nl_to_fol(&s[then_pos + 6..]);
            return FolExpr::Implies {
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
    }

    // Logical connectives bind more loosely than equality,
    // so check them first to avoid " = " matching inside "P and Q".

    // Pattern: "P and Q"
    if let Some(and_pos) = s.find(" and ") {
        let lhs = nl_to_fol(&s[..and_pos]);
        let rhs = nl_to_fol(&s[and_pos + 5..]);
        return FolExpr::And {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    // Pattern: "P or Q"
    if let Some(or_pos) = s.find(" or ") {
        let lhs = nl_to_fol(&s[..or_pos]);
        let rhs = nl_to_fol(&s[or_pos + 4..]);
        return FolExpr::Or {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    // Pattern: "not P"
    if s.starts_with("not ") {
        return FolExpr::Not {
            inner: Box::new(nl_to_fol(&s[4..])),
        };
    }

    // Pattern: "X = Y" (equality)
    if let Some(eq_pos) = s.find(" = ") {
        let lhs = parse_math_expr(&s[..eq_pos]);
        let rhs = parse_math_expr(&s[eq_pos + 3..]);
        return FolExpr::Eq {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    // Try as math expression
    let expr = parse_math_expr(&s);
    if !matches!(expr, FolExpr::Unknown { .. }) {
        return expr;
    }

    // Fallback: unknown
    FolExpr::Unknown {
        original: sentence.to_string(),
    }
}

/// Parse a simple mathematical expression (variables, numbers, +, *).
fn parse_math_expr(s: &str) -> FolExpr {
    let s = s.trim();

    // Check for addition
    if let Some(pos) = s.rfind('+') {
        if pos > 0 && pos < s.len() - 1 {
            let lhs = parse_math_expr(&s[..pos]);
            let rhs = parse_math_expr(&s[pos + 1..]);
            if !matches!(lhs, FolExpr::Unknown { .. }) && !matches!(rhs, FolExpr::Unknown { .. }) {
                return FolExpr::Add {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
            }
        }
    }

    // Check for multiplication
    if let Some(pos) = s.rfind('*') {
        if pos > 0 && pos < s.len() - 1 {
            let lhs = parse_math_expr(&s[..pos]);
            let rhs = parse_math_expr(&s[pos + 1..]);
            if !matches!(lhs, FolExpr::Unknown { .. }) && !matches!(rhs, FolExpr::Unknown { .. }) {
                return FolExpr::Mul {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                };
            }
        }
    }

    // Try as number
    if let Ok(n) = s.parse::<i64>() {
        return FolExpr::Num(n);
    }

    // Try as variable (single word, alphanumeric)
    let s_trimmed = s.trim();
    if !s_trimmed.is_empty() && s_trimmed.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return FolExpr::Var(s_trimmed.to_string());
    }

    FolExpr::Unknown {
        original: s.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Encoding result
// ---------------------------------------------------------------------------

/// Result of encoding a natural language statement.
#[derive(Debug, Clone)]
pub struct EncodingResult {
    /// Original natural language input.
    pub original: String,
    /// FOL intermediate representation.
    pub fol: FolExpr,
    /// Generated Lean source code.
    pub lean_source: String,
    /// Whether the encoding is fully determined (no Unknown nodes).
    pub fully_resolved: bool,
    /// Number of unknown sub-expressions.
    pub unknown_count: usize,
}

/// Full encoding pipeline: NL → FOL → Lean.
pub fn encode(name: &str, nl_statement: &str, proof_body: Option<&str>) -> EncodingResult {
    let fol = nl_to_fol(nl_statement);
    let lean_source = generate_lean_theorem(name, &fol, proof_body);
    let has_unknown = fol.has_unknown();
    let unknown_count = count_unknowns(&fol);

    EncodingResult {
        original: nl_statement.to_string(),
        fol,
        lean_source,
        fully_resolved: !has_unknown,
        unknown_count,
    }
}

fn count_unknowns(expr: &FolExpr) -> usize {
    match expr {
        FolExpr::Unknown { .. } => 1,
        FolExpr::ForAll { body, .. }
        | FolExpr::Exists { body, .. }
        | FolExpr::Not { inner: body } => count_unknowns(body),
        FolExpr::Implies { lhs, rhs }
        | FolExpr::And { lhs, rhs }
        | FolExpr::Or { lhs, rhs }
        | FolExpr::Eq { lhs, rhs }
        | FolExpr::Add { lhs, rhs }
        | FolExpr::Mul { lhs, rhs } => count_unknowns(lhs) + count_unknowns(rhs),
        FolExpr::Pred { args, .. } | FolExpr::Func { args, .. } => {
            args.iter().map(count_unknowns).sum()
        }
        FolExpr::Var(_) | FolExpr::Num(_) | FolExpr::Bool(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forall_pattern() {
        let fol = nl_to_fol("for all n, n + 0 = n");
        assert!(matches!(fol, FolExpr::ForAll { .. }));
        assert!(!fol.has_unknown());
    }

    #[test]
    fn exists_pattern() {
        let fol = nl_to_fol("there exists x such that x + 1 = 2");
        assert!(matches!(fol, FolExpr::Exists { .. }));
        assert!(!fol.has_unknown());
    }

    #[test]
    fn implication_pattern() {
        let fol = nl_to_fol("if n = 0 then n + 1 = 1");
        assert!(matches!(fol, FolExpr::Implies { .. }));
    }

    #[test]
    fn equality_pattern() {
        let fol = nl_to_fol("1 + 1 = 2");
        assert!(matches!(fol, FolExpr::Eq { .. }));
        assert!(!fol.has_unknown());
    }

    #[test]
    fn unknown_falls_through() {
        let fol = nl_to_fol("the quick brown fox jumps over the lazy dog");
        assert!(fol.has_unknown());
    }

    #[test]
    fn fol_to_lean_forall() {
        let fol = FolExpr::ForAll {
            var: "n".to_string(),
            ty: "ℕ".to_string(),
            body: Box::new(FolExpr::Eq {
                lhs: Box::new(FolExpr::Add {
                    lhs: Box::new(FolExpr::Var("n".to_string())),
                    rhs: Box::new(FolExpr::Num(0)),
                }),
                rhs: Box::new(FolExpr::Var("n".to_string())),
            }),
        };
        let lean = fol_to_lean(&fol);
        assert!(lean.contains("∀"));
        assert!(lean.contains("n : ℕ"));
    }

    #[test]
    fn generate_theorem_with_proof() {
        let stmt = FolExpr::Eq {
            lhs: Box::new(FolExpr::Num(1)),
            rhs: Box::new(FolExpr::Num(1)),
        };
        let code = generate_lean_theorem("one_eq_one", &stmt, Some("rfl"));
        assert!(code.contains("theorem one_eq_one"));
        assert!(code.contains("rfl"));
    }

    #[test]
    fn encode_pipeline_end_to_end() {
        let result = encode("add_zero", "for all n, n + 0 = n", Some("simp"));
        assert!(result.fully_resolved);
        assert_eq!(result.unknown_count, 0);
        assert!(result.lean_source.contains("theorem add_zero"));
        assert!(result.lean_source.contains("simp"));
    }

    #[test]
    fn encode_unknown_flagged() {
        let result = encode("test", "some incomprehensible gibberish", None);
        assert!(!result.fully_resolved);
        assert!(result.unknown_count > 0);
        assert!(result.lean_source.contains("sorry"));
    }

    #[test]
    fn and_or_patterns() {
        let fol = nl_to_fol("n = 0 and m = 1");
        assert!(matches!(fol, FolExpr::And { .. }));

        let fol2 = nl_to_fol("n = 0 or m = 1");
        assert!(matches!(fol2, FolExpr::Or { .. }));
    }

    #[test]
    fn negation_pattern() {
        let fol = nl_to_fol("not n = 0");
        assert!(matches!(fol, FolExpr::Not { .. }));
    }

    #[test]
    fn lean_header_generation() {
        let header = generate_lean_header(&["Mathlib.Tactic.Ring", "Init"]);
        assert!(header.contains("import Mathlib.Tactic.Ring"));
        assert!(header.contains("import Init"));
    }
}

//! SPTF — Structural Proof Triviality Filter
//!
//! Prevents Goodhart gaming in the swarm classroom by rejecting proofs
//! that are syntactically trivial, structurally shallow, or semantically
//! vacuous. Every proof candidate must pass ALL gates before entering
//! the verified knowledge base.
//!
//! # Gates
//!
//! 1. **Length gate**: minimum token count (rejects ultra-short proofs)
//! 2. **Tactic blacklist**: rejects proofs consisting solely of trivial tactics
//! 3. **Quantifier gate**: theorem statements must contain at least one
//!    universal (∀) or existential (∃) quantifier
//! 4. **Depth gate**: minimum nesting depth of proof structure
//! 5. **Novelty gate**: rejects known-trivial patterns (tautologies, `True`,
//!    self-equalities)
//! 6. **Sorry gate**: unconditionally rejects any proof containing `sorry`
//!
//! # Novel technique: SPTF (Structural Proof Triviality Filter)
//!
//! Unlike simple length thresholds, SPTF performs multi-axis structural
//! analysis. The key insight is that trivial proofs cluster in a low-
//! dimensional subspace of (length, depth, quantifier_count, tactic_diversity)
//! — checking all axes simultaneously catches gaming attempts that would
//! pass any single gate.

use core::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Minimum number of whitespace-separated tokens in the proof body.
pub const MIN_PROOF_TOKENS: usize = 5;

/// Minimum number of whitespace-separated tokens in the theorem statement.
pub const MIN_STATEMENT_TOKENS: usize = 4;

/// Minimum nesting depth (counted by matching delimiters: parens, braces, brackets).
pub const MIN_NESTING_DEPTH: usize = 1;

/// Minimum number of distinct tactics required when proof has multiple steps.
pub const MIN_TACTIC_DIVERSITY: usize = 2;

// ---------------------------------------------------------------------------
// Trivial tactic set
// ---------------------------------------------------------------------------

/// Tactics that, alone, constitute a trivial proof.
const TRIVIAL_TACTICS: &[&str] = &[
    "rfl",
    "trivial",
    "simp",
    "decide",
    "norm_num",
    "ring",
    "omega",
    "tauto",
    "aesop",
    "exact?",
    "assumption",
    "contradiction",
];

/// Patterns in theorem statements that indicate vacuous truth.
const VACUOUS_PATTERNS: &[&str] = &[
    "True",
    "true",
    "0 = 0",
    "1 = 1",
    "_ = _",
    "∀ _, True",
    "∃ _, True",
];

/// Forbidden keywords that unconditionally reject a proof.
const FORBIDDEN_KEYWORDS: &[&str] = &[
    "sorry",
    "admit",
    "axiom",         // user-defined axioms bypass the type checker
    "native_decide", // opaque oracle
];

// ---------------------------------------------------------------------------
// Rejection reason
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TrivialityRejection {
    /// Proof body too short.
    ProofTooShort { tokens: usize, min: usize },
    /// Theorem statement too short.
    StatementTooShort { tokens: usize, min: usize },
    /// Proof consists solely of trivial tactics.
    TrivialTacticsOnly { tactics: Vec<String> },
    /// No quantifiers in theorem statement.
    NoQuantifiers,
    /// Nesting depth too shallow.
    TooShallow { depth: usize, min: usize },
    /// Matched a known vacuous pattern.
    VacuousPattern { pattern: String },
    /// Contains a forbidden keyword (sorry, admit, etc.).
    ForbiddenKeyword { keyword: String },
    /// Insufficient tactic diversity (proof uses only one tactic repeatedly).
    LowTacticDiversity { unique: usize, min: usize },
    /// Statement is a self-equality (x = x pattern).
    SelfEquality,
}

impl fmt::Display for TrivialityRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProofTooShort { tokens, min } => {
                write!(f, "proof too short: {tokens} tokens (min {min})")
            }
            Self::StatementTooShort { tokens, min } => {
                write!(f, "statement too short: {tokens} tokens (min {min})")
            }
            Self::TrivialTacticsOnly { tactics } => {
                write!(f, "trivial tactics only: {}", tactics.join(", "))
            }
            Self::NoQuantifiers => write!(f, "no quantifiers in statement"),
            Self::TooShallow { depth, min } => {
                write!(f, "nesting depth {depth} < minimum {min}")
            }
            Self::VacuousPattern { pattern } => {
                write!(f, "vacuous pattern detected: {pattern}")
            }
            Self::ForbiddenKeyword { keyword } => {
                write!(f, "forbidden keyword: {keyword}")
            }
            Self::LowTacticDiversity { unique, min } => {
                write!(f, "tactic diversity {unique} < minimum {min}")
            }
            Self::SelfEquality => write!(f, "self-equality (x = x)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Filter result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum FilterResult {
    /// Proof passed all triviality gates.
    Accepted,
    /// Proof was rejected by one or more gates.
    Rejected(Vec<TrivialityRejection>),
}

impl FilterResult {
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }

    pub fn rejections(&self) -> &[TrivialityRejection] {
        match self {
            Self::Accepted => &[],
            Self::Rejected(r) => r,
        }
    }
}

// ---------------------------------------------------------------------------
// ProofCandidate
// ---------------------------------------------------------------------------

/// A proof candidate to be screened before entering the knowledge base.
#[derive(Debug, Clone)]
pub struct ProofCandidate {
    /// The theorem statement (e.g., "theorem foo : ∀ n : Nat, n + 0 = n")
    pub statement: String,
    /// The proof body (e.g., "by induction n with ...")
    pub proof_body: String,
    /// Optional: the full Lean source (statement + proof combined)
    pub full_source: Option<String>,
}

impl ProofCandidate {
    pub fn new(statement: &str, proof_body: &str) -> Self {
        Self {
            statement: statement.to_string(),
            proof_body: proof_body.to_string(),
            full_source: None,
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.full_source = Some(source.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// TrivialityFilter (SPTF)
// ---------------------------------------------------------------------------

/// The Structural Proof Triviality Filter.
///
/// Runs all gates and collects ALL rejections (not just the first).
/// This allows the swarm to understand exactly why a proof was rejected
/// and generate targeted improvements.
pub struct TrivialityFilter {
    pub min_proof_tokens: usize,
    pub min_statement_tokens: usize,
    pub min_nesting_depth: usize,
    pub min_tactic_diversity: usize,
    pub require_quantifier: bool,
}

impl TrivialityFilter {
    pub fn new() -> Self {
        Self {
            min_proof_tokens: MIN_PROOF_TOKENS,
            min_statement_tokens: MIN_STATEMENT_TOKENS,
            min_nesting_depth: MIN_NESTING_DEPTH,
            min_tactic_diversity: MIN_TACTIC_DIVERSITY,
            require_quantifier: true,
        }
    }

    /// Strict mode: higher thresholds for production use.
    pub fn strict() -> Self {
        Self {
            min_proof_tokens: 10,
            min_statement_tokens: 6,
            min_nesting_depth: 2,
            min_tactic_diversity: 3,
            require_quantifier: true,
        }
    }

    /// Run all gates on a proof candidate.
    pub fn check(&self, candidate: &ProofCandidate) -> FilterResult {
        let mut rejections = Vec::new();

        // Determine the source to scan for forbidden keywords
        let full = candidate
            .full_source
            .as_deref()
            .unwrap_or(&candidate.proof_body);

        // Gate 0: Forbidden keywords (unconditional, checked first)
        self.check_forbidden(full, &mut rejections);

        // Gate 1: Proof length
        self.check_proof_length(&candidate.proof_body, &mut rejections);

        // Gate 2: Statement length
        self.check_statement_length(&candidate.statement, &mut rejections);

        // Gate 3: Trivial-tactics-only
        self.check_trivial_tactics(&candidate.proof_body, &mut rejections);

        // Gate 4: Quantifier presence
        self.check_quantifiers(&candidate.statement, &mut rejections);

        // Gate 5: Nesting depth
        self.check_nesting_depth(&candidate.proof_body, &mut rejections);

        // Gate 6: Vacuous patterns
        self.check_vacuous(&candidate.statement, &mut rejections);

        // Gate 7: Self-equality
        self.check_self_equality(&candidate.statement, &mut rejections);

        // Gate 8: Tactic diversity
        self.check_tactic_diversity(&candidate.proof_body, &mut rejections);

        if rejections.is_empty() {
            FilterResult::Accepted
        } else {
            FilterResult::Rejected(rejections)
        }
    }

    // -- Individual gates --

    fn check_forbidden(&self, source: &str, rejections: &mut Vec<TrivialityRejection>) {
        for &kw in FORBIDDEN_KEYWORDS {
            // Match as whole word: check that characters before/after are not alphanumeric
            let mut start = 0;
            while let Some(pos) = source[start..].find(kw) {
                let abs_pos = start + pos;
                let before_ok =
                    abs_pos == 0 || !source.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
                let after_pos = abs_pos + kw.len();
                let after_ok = after_pos >= source.len()
                    || !source.as_bytes()[after_pos].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    rejections.push(TrivialityRejection::ForbiddenKeyword {
                        keyword: kw.to_string(),
                    });
                    break; // one rejection per keyword is enough
                }
                start = abs_pos + kw.len();
            }
        }
    }

    fn check_proof_length(&self, proof: &str, rejections: &mut Vec<TrivialityRejection>) {
        let tokens = token_count(proof);
        if tokens < self.min_proof_tokens {
            rejections.push(TrivialityRejection::ProofTooShort {
                tokens,
                min: self.min_proof_tokens,
            });
        }
    }

    fn check_statement_length(&self, stmt: &str, rejections: &mut Vec<TrivialityRejection>) {
        let tokens = token_count(stmt);
        if tokens < self.min_statement_tokens {
            rejections.push(TrivialityRejection::StatementTooShort {
                tokens,
                min: self.min_statement_tokens,
            });
        }
    }

    fn check_trivial_tactics(&self, proof: &str, rejections: &mut Vec<TrivialityRejection>) {
        let tactics = extract_tactics(proof);
        if tactics.is_empty() {
            return; // can't determine — let other gates catch it
        }
        let all_trivial = tactics
            .iter()
            .all(|t| TRIVIAL_TACTICS.contains(&t.as_str()));
        if all_trivial {
            rejections.push(TrivialityRejection::TrivialTacticsOnly { tactics });
        }
    }

    fn check_quantifiers(&self, stmt: &str, rejections: &mut Vec<TrivialityRejection>) {
        if !self.require_quantifier {
            return;
        }
        let has_forall = stmt.contains('∀')
            || stmt.contains("forall")
            || stmt.contains("\\forall")
            || stmt.contains("∀");
        let has_exists = stmt.contains('∃') || stmt.contains("exists") || stmt.contains("\\exists");
        // Also accept Lean 4 fun/lambda binders as implicit universal quantification
        let has_binder = stmt.contains("fun ")
            || stmt.contains("(n :")
            || stmt.contains("(x :")
            || stmt.contains("(m :");
        if !has_forall && !has_exists && !has_binder {
            rejections.push(TrivialityRejection::NoQuantifiers);
        }
    }

    fn check_nesting_depth(&self, proof: &str, rejections: &mut Vec<TrivialityRejection>) {
        let depth = nesting_depth(proof);
        if depth < self.min_nesting_depth {
            rejections.push(TrivialityRejection::TooShallow {
                depth,
                min: self.min_nesting_depth,
            });
        }
    }

    fn check_vacuous(&self, stmt: &str, rejections: &mut Vec<TrivialityRejection>) {
        let normalized = stmt.replace(char::is_whitespace, " ");
        for &pat in VACUOUS_PATTERNS {
            if normalized.contains(pat) {
                rejections.push(TrivialityRejection::VacuousPattern {
                    pattern: pat.to_string(),
                });
                return; // one is enough
            }
        }
    }

    fn check_self_equality(&self, stmt: &str, rejections: &mut Vec<TrivialityRejection>) {
        // Detect patterns like "x = x", "a = a", "foo = foo"
        // Compare the FULL normalised expression on each side of " = ".
        // Only triggers when lhs_expr and rhs_expr are identical strings.
        // This avoids false positives on "n + m = m + n" (commuted, not equal).
        if let Some(eq_pos) = stmt.find(" = ") {
            // LHS: everything after the last colon (type annotation) or comma
            let lhs_raw = stmt[..eq_pos].trim();
            let lhs_expr = if let Some(colon_pos) = lhs_raw.rfind(',') {
                lhs_raw[colon_pos + 1..].trim()
            } else if let Some(colon_pos) = lhs_raw.rfind(':') {
                lhs_raw[colon_pos + 1..].trim()
            } else {
                lhs_raw
            };
            // RHS: everything after " = " until end of statement
            let rhs_expr = stmt[eq_pos + 3..].trim();
            // Normalize whitespace for comparison
            let lhs_norm: String = lhs_expr.split_whitespace().collect::<Vec<_>>().join(" ");
            let rhs_norm: String = rhs_expr.split_whitespace().collect::<Vec<_>>().join(" ");
            if !lhs_norm.is_empty() && lhs_norm == rhs_norm {
                rejections.push(TrivialityRejection::SelfEquality);
            }
        }
    }

    fn check_tactic_diversity(&self, proof: &str, rejections: &mut Vec<TrivialityRejection>) {
        let tactics = extract_tactics(proof);
        if tactics.len() < 2 {
            return; // single-tactic proofs are caught by other gates
        }
        let mut unique = tactics.clone();
        unique.sort();
        unique.dedup();
        if unique.len() < self.min_tactic_diversity {
            rejections.push(TrivialityRejection::LowTacticDiversity {
                unique: unique.len(),
                min: self.min_tactic_diversity,
            });
        }
    }
}

impl Default for TrivialityFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Count whitespace-separated tokens.
fn token_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Compute maximum nesting depth from matched delimiters.
fn nesting_depth(s: &str) -> usize {
    let mut max_depth = 0usize;
    let mut current = 0usize;
    for c in s.chars() {
        match c {
            '(' | '{' | '[' | '⟨' => {
                current += 1;
                if current > max_depth {
                    max_depth = current;
                }
            }
            ')' | '}' | ']' | '⟩' => {
                current = current.saturating_sub(1);
            }
            _ => {}
        }
    }
    max_depth
}

/// Extract tactic names from a Lean 4 proof body.
///
/// Heuristic: after `by`, each line's first word (ignoring structural
/// keywords) is treated as a tactic invocation. For match-arm lines
/// (`| pattern => tactic`), the first word after `=>` is extracted.
fn extract_tactics(proof: &str) -> Vec<String> {
    let mut tactics = Vec::new();
    let body = if let Some(pos) = proof.find("by") {
        &proof[pos + 2..]
    } else if let Some(pos) = proof.find(":=") {
        &proof[pos + 2..]
    } else {
        proof
    };

    for line in body.lines() {
        let trimmed = line.trim();
        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with("/-") {
            continue;
        }
        // Handle match arms: `| pattern => tactic args`
        let cleaned = if let Some(arrow_pos) = trimmed.find("=>") {
            trimmed[arrow_pos + 2..].trim()
        } else {
            // Strip leading bullet points and semicolons
            trimmed
                .trim_start_matches('·')
                .trim_start_matches('|')
                .trim_start_matches("<;>")
                .trim()
        };
        // Handle semicolon-separated tactic chains: `tac1; tac2; tac3`
        for segment in cleaned.split(';') {
            let seg = segment.trim();
            if seg.is_empty() {
                continue;
            }
            if let Some(tactic) = seg.split_whitespace().next() {
                // Skip structural keywords and braces
                if !matches!(
                    tactic,
                    "case" | "next" | "where" | "with" | "do" | "|" | "·" | "{" | "}"
                ) {
                    tactics.push(tactic.to_string());
                }
            }
        }
    }
    tactics
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn filter() -> TrivialityFilter {
        TrivialityFilter::new()
    }

    #[test]
    fn rejects_sorry() {
        let c = ProofCandidate::new("theorem foo : ∀ n : Nat, n + 0 = n", "by sorry");
        let r = filter().check(&c);
        assert!(!r.is_accepted());
        assert!(r.rejections().iter().any(
            |r| matches!(r, TrivialityRejection::ForbiddenKeyword { keyword } if keyword == "sorry")
        ));
    }

    #[test]
    fn rejects_admit() {
        let c = ProofCandidate::new("theorem bar : ∀ x : Nat, x = x", "by { admit }");
        let r = filter().check(&c);
        assert!(!r.is_accepted());
        assert!(r.rejections().iter().any(
            |r| matches!(r, TrivialityRejection::ForbiddenKeyword { keyword } if keyword == "admit")
        ));
    }

    #[test]
    fn rejects_trivial_rfl() {
        let c = ProofCandidate::new("theorem t : 1 = 1", "by rfl");
        let r = filter().check(&c);
        assert!(!r.is_accepted());
        // Should catch: vacuous (1 = 1), trivial tactic (rfl), short proof, no quantifier
    }

    #[test]
    fn rejects_no_quantifiers() {
        let f = TrivialityFilter {
            min_proof_tokens: 1,
            min_statement_tokens: 1,
            min_nesting_depth: 0,
            min_tactic_diversity: 1,
            require_quantifier: true,
        };
        let c = ProofCandidate::new(
            "theorem t : Nat",
            "by { exact 42; ring; omega; intro h; cases h }",
        );
        let r = f.check(&c);
        assert!(!r.is_accepted());
        assert!(r
            .rejections()
            .iter()
            .any(|r| matches!(r, TrivialityRejection::NoQuantifiers)));
    }

    #[test]
    fn accepts_nontrivial_proof() {
        let c = ProofCandidate::new(
            "theorem add_comm : ∀ (n m : Nat), n + m = m + n",
            "by {\n  induction n with\n  | zero => simp [Nat.zero_add, Nat.add_zero]\n  | succ k ih => {\n    rw [Nat.succ_add, Nat.add_succ]\n    exact congrArg Nat.succ ih\n  }\n}",
        );
        let r = filter().check(&c);
        assert!(
            r.is_accepted(),
            "Expected accepted, got: {:?}",
            r.rejections()
        );
    }

    #[test]
    fn rejects_self_equality_in_statement() {
        let f = TrivialityFilter {
            min_proof_tokens: 1,
            min_statement_tokens: 1,
            min_nesting_depth: 0,
            min_tactic_diversity: 1,
            require_quantifier: false,
        };
        let c = ProofCandidate::new(
            "theorem t : foo = foo",
            "by { induction foo with\n  | a => rfl\n  | b => simp\n}",
        );
        let r = f.check(&c);
        assert!(!r.is_accepted());
        assert!(r
            .rejections()
            .iter()
            .any(|r| matches!(r, TrivialityRejection::SelfEquality)));
    }

    #[test]
    fn rejects_proof_too_short() {
        let c = ProofCandidate::new("theorem t : ∀ n : Nat, n = n", "rfl");
        let r = filter().check(&c);
        assert!(!r.is_accepted());
        assert!(r
            .rejections()
            .iter()
            .any(|r| matches!(r, TrivialityRejection::ProofTooShort { .. })));
    }

    #[test]
    fn strict_mode_higher_thresholds() {
        let f = TrivialityFilter::strict();
        assert_eq!(f.min_proof_tokens, 10);
        assert_eq!(f.min_statement_tokens, 6);
        assert_eq!(f.min_nesting_depth, 2);
        assert_eq!(f.min_tactic_diversity, 3);
    }

    #[test]
    fn nesting_depth_calculation() {
        assert_eq!(nesting_depth(""), 0);
        assert_eq!(nesting_depth("abc"), 0);
        assert_eq!(nesting_depth("(a)"), 1);
        assert_eq!(nesting_depth("((a))"), 2);
        assert_eq!(nesting_depth("{(a [b])}"), 3);
        assert_eq!(nesting_depth("(a) (b)"), 1);
    }

    #[test]
    fn extract_tactics_basic() {
        let tactics =
            extract_tactics("by\n  induction n with\n  | zero => simp\n  | succ k ih => rw [h]");
        assert!(tactics.contains(&"induction".to_string()));
        assert!(tactics.contains(&"simp".to_string()));
        assert!(tactics.contains(&"rw".to_string()));
    }

    #[test]
    fn tactic_diversity_rejection() {
        let f = TrivialityFilter {
            min_proof_tokens: 1,
            min_statement_tokens: 1,
            min_nesting_depth: 0,
            min_tactic_diversity: 2,
            require_quantifier: false,
        };
        let c = ProofCandidate::new("theorem t : ∀ n, P n", "by\n  simp\n  simp\n  simp");
        let r = f.check(&c);
        assert!(!r.is_accepted());
        assert!(r
            .rejections()
            .iter()
            .any(|r| matches!(r, TrivialityRejection::LowTacticDiversity { .. })));
    }

    #[test]
    fn vacuous_true_rejected() {
        let f = TrivialityFilter {
            min_proof_tokens: 1,
            min_statement_tokens: 1,
            min_nesting_depth: 0,
            min_tactic_diversity: 1,
            require_quantifier: false,
        };
        let c = ProofCandidate::new("theorem t : True", "by { exact True.intro; done }");
        let r = f.check(&c);
        assert!(!r.is_accepted());
        assert!(r
            .rejections()
            .iter()
            .any(|r| matches!(r, TrivialityRejection::VacuousPattern { .. })));
    }

    #[test]
    fn display_formats_all_rejections() {
        let r = TrivialityRejection::ProofTooShort { tokens: 2, min: 5 };
        let s = format!("{r}");
        assert!(s.contains("2 tokens"));

        let r = TrivialityRejection::ForbiddenKeyword {
            keyword: "sorry".to_string(),
        };
        assert!(format!("{r}").contains("sorry"));
    }

    #[test]
    fn filter_result_api() {
        let accepted = FilterResult::Accepted;
        assert!(accepted.is_accepted());
        assert!(accepted.rejections().is_empty());

        let rejected = FilterResult::Rejected(vec![TrivialityRejection::NoQuantifiers]);
        assert!(!rejected.is_accepted());
        assert_eq!(rejected.rejections().len(), 1);
    }

    #[test]
    fn forbidden_keyword_whole_word_only() {
        // "sorry" inside "sorrynotsorry" should NOT trigger
        let f = TrivialityFilter {
            min_proof_tokens: 1,
            min_statement_tokens: 1,
            min_nesting_depth: 0,
            min_tactic_diversity: 1,
            require_quantifier: false,
        };
        let c = ProofCandidate::new(
            "theorem t : ∀ n, P n",
            "by { intro n; exact sorrynotsorry n; apply foobar; cases h; ring_nf }",
        );
        let r = f.check(&c);
        // Should NOT have ForbiddenKeyword for "sorry"
        let has_sorry = r.rejections().iter().any(|r| {
            matches!(r, TrivialityRejection::ForbiddenKeyword { keyword } if keyword == "sorry")
        });
        assert!(
            !has_sorry,
            "should not match 'sorry' inside 'sorrynotsorry'"
        );
    }
}

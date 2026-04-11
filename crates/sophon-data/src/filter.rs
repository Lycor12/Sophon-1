//! Quality filter — corpus filtering for code/math/science.
//!
//! Filters documents by:
//!   1. Minimum/maximum byte length
//!   2. UTF-8 validity ratio
//!   3. Alphanumeric ratio (rejects binary / gibberish)
//!   4. Entropy bounds (rejects both too-random and too-repetitive)
//!   5. Code detection heuristics (presence of braces, semicolons, keywords)
//!   6. Math detection heuristics (presence of operators, greek, LaTeX)
//!   7. Deduplication via content hash
//!
//! # Novel technique: AEHF (Adaptive Entropy-Histogram Filtering)
//!
//! Instead of fixed entropy bounds, AEHF adjusts the acceptable entropy
//! range based on the running histogram of byte-entropy values seen so far.
//! Documents are accepted if their entropy falls within [mean - 2σ, mean + 2σ]
//! of the historical distribution. This prevents the filter from being
//! too aggressive early (when the distribution is unknown) and adapts to
//! the characteristics of each corpus.

use crate::corpus::Document;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// FilterConfig
// ---------------------------------------------------------------------------

/// Configuration for quality filtering.
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Minimum document length in bytes.
    pub min_length: usize,
    /// Maximum document length in bytes.
    pub max_length: usize,
    /// Minimum fraction of bytes that are valid UTF-8.
    pub min_utf8_ratio: f32,
    /// Minimum fraction of printable ASCII characters.
    pub min_printable_ratio: f32,
    /// Minimum byte entropy (rejects very repetitive).
    pub min_entropy: f32,
    /// Maximum byte entropy (rejects random noise).
    pub max_entropy: f32,
    /// Whether to use adaptive entropy filtering (AEHF).
    pub adaptive_entropy: bool,
    /// Enable deduplication via content hash.
    pub deduplicate: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_length: 64,
            max_length: 1_000_000, // 1MB
            min_utf8_ratio: 0.9,
            min_printable_ratio: 0.7,
            min_entropy: 1.0,
            max_entropy: 7.5,
            adaptive_entropy: true,
            deduplicate: true,
        }
    }
}

// ---------------------------------------------------------------------------
// FilterStats
// ---------------------------------------------------------------------------

/// Statistics from the filtering process.
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    /// Total documents seen.
    pub total: usize,
    /// Documents that passed all filters.
    pub passed: usize,
    /// Rejected by length filter.
    pub rejected_length: usize,
    /// Rejected by UTF-8 validity filter.
    pub rejected_utf8: usize,
    /// Rejected by printable ratio filter.
    pub rejected_printable: usize,
    /// Rejected by entropy filter.
    pub rejected_entropy: usize,
    /// Rejected by deduplication.
    pub rejected_duplicate: usize,
}

impl FilterStats {
    /// Pass rate as a fraction.
    pub fn pass_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f32 / self.total as f32
    }

    /// Total rejected.
    pub fn rejected(&self) -> usize {
        self.total - self.passed
    }
}

// ---------------------------------------------------------------------------
// QualityFilter
// ---------------------------------------------------------------------------

/// Corpus quality filter with AEHF.
pub struct QualityFilter {
    /// Configuration.
    pub config: FilterConfig,
    /// Running statistics.
    pub stats: FilterStats,
    /// Seen content hashes (for deduplication).
    seen_hashes: HashSet<u64>,
    /// AEHF state: running sum and sum-of-squares of entropy.
    entropy_sum: f64,
    entropy_sum_sq: f64,
    entropy_count: usize,
}

impl QualityFilter {
    /// Create a new filter.
    pub fn new(config: FilterConfig) -> Self {
        Self {
            config,
            stats: FilterStats::default(),
            seen_hashes: HashSet::new(),
            entropy_sum: 0.0,
            entropy_sum_sq: 0.0,
            entropy_count: 0,
        }
    }

    /// Check if a document passes all quality filters.
    pub fn check(&mut self, doc: &Document) -> bool {
        self.stats.total += 1;

        // Length filter
        if doc.len() < self.config.min_length || doc.len() > self.config.max_length {
            self.stats.rejected_length += 1;
            return false;
        }

        // UTF-8 validity filter
        let utf8_ratio = utf8_valid_ratio(&doc.bytes);
        if utf8_ratio < self.config.min_utf8_ratio {
            self.stats.rejected_utf8 += 1;
            return false;
        }

        // Printable character ratio
        let printable_ratio = printable_ratio(&doc.bytes);
        if printable_ratio < self.config.min_printable_ratio {
            self.stats.rejected_printable += 1;
            return false;
        }

        // Entropy filter (AEHF or fixed)
        let entropy = doc.byte_entropy();
        let (lo, hi) = if self.config.adaptive_entropy && self.entropy_count >= 10 {
            let mean = (self.entropy_sum / self.entropy_count as f64) as f32;
            let var = (self.entropy_sum_sq / self.entropy_count as f64
                - (self.entropy_sum / self.entropy_count as f64).powi(2))
                as f32;
            let std = var.max(0.0).sqrt();
            (
                (mean - 2.0 * std).max(self.config.min_entropy),
                (mean + 2.0 * std).min(self.config.max_entropy),
            )
        } else {
            (self.config.min_entropy, self.config.max_entropy)
        };

        if entropy < lo || entropy > hi {
            self.stats.rejected_entropy += 1;
            // Still update AEHF stats even for rejected documents
            self.entropy_sum += entropy as f64;
            self.entropy_sum_sq += (entropy as f64) * (entropy as f64);
            self.entropy_count += 1;
            return false;
        }

        // Update AEHF stats
        self.entropy_sum += entropy as f64;
        self.entropy_sum_sq += (entropy as f64) * (entropy as f64);
        self.entropy_count += 1;

        // Deduplication
        if self.config.deduplicate {
            let hash = content_hash(&doc.bytes);
            if !self.seen_hashes.insert(hash) {
                self.stats.rejected_duplicate += 1;
                return false;
            }
        }

        self.stats.passed += 1;
        true
    }

    /// Filter a batch of documents, returning only those that pass.
    pub fn filter_batch(&mut self, docs: &[Document]) -> Vec<Document> {
        docs.iter().filter(|d| self.check(d)).cloned().collect()
    }

    /// Reset filter state (clears dedup set and stats).
    pub fn reset(&mut self) {
        self.stats = FilterStats::default();
        self.seen_hashes.clear();
        self.entropy_sum = 0.0;
        self.entropy_sum_sq = 0.0;
        self.entropy_count = 0;
    }

    /// Current adaptive entropy bounds.
    pub fn adaptive_bounds(&self) -> (f32, f32) {
        if self.entropy_count < 10 {
            return (self.config.min_entropy, self.config.max_entropy);
        }
        let mean = (self.entropy_sum / self.entropy_count as f64) as f32;
        let var = (self.entropy_sum_sq / self.entropy_count as f64
            - (self.entropy_sum / self.entropy_count as f64).powi(2)) as f32;
        let std = var.max(0.0).sqrt();
        (
            (mean - 2.0 * std).max(self.config.min_entropy),
            (mean + 2.0 * std).min(self.config.max_entropy),
        )
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Fraction of bytes that form valid UTF-8 sequences.
fn utf8_valid_ratio(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 1.0;
    }
    let text = String::from_utf8_lossy(bytes);
    let valid_bytes = text.len().min(bytes.len());
    // Count non-replacement characters
    let replacements = text.chars().filter(|&c| c == '\u{FFFD}').count();
    let valid_chars = text.chars().count() - replacements;
    if text.chars().count() == 0 {
        return 0.0;
    }
    valid_chars as f32 / text.chars().count() as f32
}

/// Fraction of bytes that are printable ASCII or common whitespace.
fn printable_ratio(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 1.0;
    }
    let printable = bytes
        .iter()
        .filter(|&&b| (b >= 0x20 && b <= 0x7E) || b == b'\n' || b == b'\r' || b == b'\t')
        .count();
    printable as f32 / bytes.len() as f32
}

/// Simple content hash (FNV-1a 64-bit).
fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Check if content appears to be source code.
pub fn is_likely_code(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(bytes);
    let lower = text.to_lowercase();

    let code_indicators = [
        "{", "}", ";", "fn ", "def ", "class ", "import ", "include ", "return ", "if (", "for (",
        "while (", "pub fn", "let ", "const ",
    ];

    let count = code_indicators
        .iter()
        .filter(|&&ind| lower.contains(ind))
        .count();

    count >= 3
}

/// Check if content appears to be mathematical.
pub fn is_likely_math(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(bytes);

    let math_indicators = [
        "theorem", "proof", "lemma", "∀", "∃", "→", "⟨", "⟩", "\\frac", "\\int", "\\sum", "\\prod",
        "equation", "=", "+", "*", "^",
    ];

    let count = math_indicators
        .iter()
        .filter(|&&ind| text.contains(ind))
        .count();

    count >= 4
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(content: &str) -> Document {
        Document {
            bytes: content.as_bytes().to_vec(),
            source: "test".into(),
            index: 0,
        }
    }

    fn make_doc_bytes(bytes: Vec<u8>) -> Document {
        Document {
            bytes,
            source: "test".into(),
            index: 0,
        }
    }

    #[test]
    fn passes_normal_text() {
        let config = FilterConfig {
            min_length: 10,
            max_length: 1000,
            deduplicate: false,
            adaptive_entropy: false,
            ..Default::default()
        };
        let mut filter = QualityFilter::new(config);
        let doc = make_doc(
            "This is a normal English sentence with enough characters to pass the length filter.",
        );
        assert!(filter.check(&doc));
        assert_eq!(filter.stats.passed, 1);
    }

    #[test]
    fn rejects_too_short() {
        let mut filter = QualityFilter::new(FilterConfig::default());
        let doc = make_doc("hi");
        assert!(!filter.check(&doc));
        assert_eq!(filter.stats.rejected_length, 1);
    }

    #[test]
    fn rejects_too_long() {
        let config = FilterConfig {
            max_length: 100,
            ..Default::default()
        };
        let mut filter = QualityFilter::new(config);
        let doc = make_doc(&"a".repeat(200));
        assert!(!filter.check(&doc));
        assert_eq!(filter.stats.rejected_length, 1);
    }

    #[test]
    fn rejects_duplicate() {
        let config = FilterConfig {
            min_length: 10,
            deduplicate: true,
            adaptive_entropy: false,
            ..Default::default()
        };
        let mut filter = QualityFilter::new(config);
        let doc = make_doc("This is enough text to pass all the other filters easily.");
        assert!(filter.check(&doc));
        let doc2 = make_doc("This is enough text to pass all the other filters easily.");
        assert!(!filter.check(&doc2));
        assert_eq!(filter.stats.rejected_duplicate, 1);
    }

    #[test]
    fn rejects_repetitive_content() {
        let config = FilterConfig {
            min_length: 10,
            min_entropy: 1.0,
            adaptive_entropy: false,
            ..Default::default()
        };
        let mut filter = QualityFilter::new(config);
        let doc = make_doc(&"aaaa".repeat(100)); // very low entropy
        assert!(!filter.check(&doc));
        assert_eq!(filter.stats.rejected_entropy, 1);
    }

    #[test]
    fn pass_rate_computation() {
        let stats = FilterStats {
            total: 10,
            passed: 7,
            ..Default::default()
        };
        assert!((stats.pass_rate() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn code_detection() {
        let code = b"fn main() {\n    let x = 5;\n    if (x > 0) {\n        return x;\n    }\n}";
        assert!(is_likely_code(code));

        let not_code = b"The cat sat on the mat and looked out the window.";
        assert!(!is_likely_code(not_code));
    }

    #[test]
    fn math_detection() {
        let math = b"theorem add_comm : forall n m, n + m = m + n := by\nproof\n  induction\nqed";
        assert!(is_likely_math(math));

        let not_math = b"The weather today is sunny with a chance of rain.";
        assert!(!is_likely_math(not_math));
    }

    #[test]
    fn content_hash_deterministic() {
        let a = content_hash(b"hello world");
        let b = content_hash(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn content_hash_different_for_different_input() {
        let a = content_hash(b"hello");
        let b = content_hash(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn filter_batch_returns_passing_docs() {
        let config = FilterConfig {
            min_length: 10,
            max_length: 1000,
            deduplicate: false,
            adaptive_entropy: false,
            ..Default::default()
        };
        let mut filter = QualityFilter::new(config);
        let docs = vec![
            make_doc("Short"), // too short
            make_doc("This is long enough to pass the minimum length filter easily."),
            make_doc("Also long enough to pass all the minimum length requirements here."),
        ];
        let passed = filter.filter_batch(&docs);
        assert_eq!(passed.len(), 2);
    }
}

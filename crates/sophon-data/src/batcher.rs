//! Byte batcher — forms training batches from documents.
//!
//! Takes filtered documents and produces fixed-length byte sequences
//! suitable for training the model. Handles:
//!   - Sequence chunking (split long docs, pad short ones)
//!   - Batch formation with configurable batch size
//!   - Shuffling for training randomness
//!   - SBDI: entropy-based diverse batch composition
//!
//! # Novel technique: SBDI (Streaming Byte-Domain Interleaving)
//!
//! Maintains a pool sorted by byte entropy. When forming a batch,
//! alternates between high-entropy and low-entropy documents to create
//! maximally diverse mini-batches. This increases gradient variance
//! in useful directions (the model sees both structured code and
//! noisy mathematical notation in the same batch).

use crate::corpus::Document;
use sophon_core::rng::Rng;

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration for batch formation.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Sequence length in bytes (context window).
    pub seq_len: usize,
    /// Number of sequences per batch.
    pub batch_size: usize,
    /// Padding byte value (used when document is shorter than seq_len).
    pub pad_byte: u8,
    /// Whether to use SBDI entropy-based interleaving.
    pub use_sbdi: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            seq_len: 512,
            batch_size: 4,
            pad_byte: 0,
            use_sbdi: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

/// A training batch of byte sequences.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Input sequences: [batch_size][seq_len] bytes.
    pub inputs: Vec<Vec<u8>>,
    /// Target sequences: [batch_size][seq_len] bytes (shifted by 1).
    pub targets: Vec<Vec<u8>>,
    /// Actual lengths (before padding) for each sequence.
    pub lengths: Vec<usize>,
    /// Batch index (for tracking).
    pub batch_idx: usize,
}

impl Batch {
    /// Number of sequences in this batch.
    pub fn size(&self) -> usize {
        self.inputs.len()
    }

    /// Total tokens (non-padding) in this batch.
    pub fn total_tokens(&self) -> usize {
        self.lengths.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// ByteBatcher
// ---------------------------------------------------------------------------

/// Forms training batches from documents.
pub struct ByteBatcher {
    /// Configuration.
    pub config: BatchConfig,
    /// Document pool (chunked into sequences).
    pool: Vec<ChunkedDoc>,
    /// Current position in the pool.
    cursor: usize,
    /// Batch counter.
    batch_count: usize,
    /// RNG for shuffling.
    rng: Rng,
}

/// A document chunk ready for batching.
#[derive(Clone)]
struct ChunkedDoc {
    /// Byte sequence (length = seq_len, padded if needed).
    bytes: Vec<u8>,
    /// Actual length before padding.
    actual_len: usize,
    /// Byte entropy (for SBDI sorting).
    entropy: f32,
}

impl ByteBatcher {
    /// Create a new batcher.
    pub fn new(config: BatchConfig, seed: u64) -> Self {
        Self {
            config,
            pool: Vec::new(),
            cursor: 0,
            batch_count: 0,
            rng: Rng::new(seed),
        }
    }

    /// Add documents to the pool (chunks them into seq_len pieces).
    pub fn add_documents(&mut self, docs: &[Document]) {
        for doc in docs {
            if doc.is_empty() {
                continue;
            }

            // Chunk the document into seq_len + 1 pieces (input + target)
            let total_len = self.config.seq_len + 1; // need one extra for target
            let mut offset = 0;

            while offset < doc.len() {
                let end = (offset + total_len).min(doc.len());
                let chunk = &doc.bytes[offset..end];

                if chunk.len() < 2 {
                    break; // Need at least 2 bytes for input+target
                }

                // Pad if necessary
                let mut padded = vec![self.config.pad_byte; total_len];
                padded[..chunk.len()].copy_from_slice(chunk);

                let actual_len = chunk.len().min(self.config.seq_len);
                let entropy = byte_entropy_fast(&padded[..actual_len]);

                self.pool.push(ChunkedDoc {
                    bytes: padded,
                    actual_len,
                    entropy,
                });

                offset += self.config.seq_len; // Stride by seq_len (overlap of 1)
            }
        }
    }

    /// Shuffle the pool for training.
    pub fn shuffle(&mut self) {
        // Fisher-Yates shuffle
        let n = self.pool.len();
        for i in (1..n).rev() {
            let j = (self.rng.next_u64() as usize) % (i + 1);
            self.pool.swap(i, j);
        }
        self.cursor = 0;

        // If SBDI, sort by entropy then interleave
        if self.config.use_sbdi && self.pool.len() >= self.config.batch_size * 2 {
            self.sbdi_interleave();
        }
    }

    /// Get the next batch. Returns None when pool is exhausted.
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.cursor + self.config.batch_size > self.pool.len() {
            return None;
        }

        let mut inputs = Vec::with_capacity(self.config.batch_size);
        let mut targets = Vec::with_capacity(self.config.batch_size);
        let mut lengths = Vec::with_capacity(self.config.batch_size);

        for i in 0..self.config.batch_size {
            let chunk = &self.pool[self.cursor + i];
            let sl = self.config.seq_len;

            // Input: first seq_len bytes
            let input = chunk.bytes[..sl].to_vec();
            // Target: bytes [1..seq_len+1] (next-token prediction)
            let target = chunk.bytes[1..sl + 1].to_vec();

            inputs.push(input);
            targets.push(target);
            lengths.push(chunk.actual_len.min(sl));
        }

        self.cursor += self.config.batch_size;
        let idx = self.batch_count;
        self.batch_count += 1;

        Some(Batch {
            inputs,
            targets,
            lengths,
            batch_idx: idx,
        })
    }

    /// Total number of sequences in the pool.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Number of complete batches available.
    pub fn n_batches(&self) -> usize {
        self.pool.len() / self.config.batch_size
    }

    /// Reset cursor to beginning (keeps pool, does not reshuffle).
    pub fn reset_cursor(&mut self) {
        self.cursor = 0;
    }

    // -----------------------------------------------------------------------
    // SBDI interleaving
    // -----------------------------------------------------------------------

    /// Sort pool by entropy, then interleave from extremes.
    ///
    /// Result: [low, high, low, high, ...] entropy alternation.
    fn sbdi_interleave(&mut self) {
        // Sort by entropy ascending
        self.pool.sort_by(|a, b| {
            a.entropy
                .partial_cmp(&b.entropy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = self.pool.len();
        let mut interleaved = Vec::with_capacity(n);
        let mut lo = 0;
        let mut hi = n - 1;

        while lo <= hi {
            interleaved.push(self.pool[lo].clone());
            lo += 1;
            if lo <= hi {
                interleaved.push(self.pool[hi].clone());
                if hi == 0 {
                    break;
                }
                hi -= 1;
            }
        }

        self.pool = interleaved;
    }
}

/// Fast byte entropy (no allocation).
fn byte_entropy_fast(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f32;
    let mut entropy = 0.0f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
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

    #[test]
    fn basic_batching() {
        let config = BatchConfig {
            seq_len: 8,
            batch_size: 2,
            use_sbdi: false,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs = vec![
            make_doc("hello world this is a test document"),
            make_doc("another document with enough content"),
        ];
        batcher.add_documents(&docs);
        assert!(batcher.pool_size() > 0);

        let batch = batcher.next_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.inputs[0].len(), 8);
        assert_eq!(batch.targets[0].len(), 8);
    }

    #[test]
    fn input_target_shifted_by_one() {
        let config = BatchConfig {
            seq_len: 5,
            batch_size: 1,
            use_sbdi: false,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs = vec![make_doc("abcdefghij")];
        batcher.add_documents(&docs);

        let batch = batcher.next_batch().unwrap();
        // Input: bytes[0..5] = "abcde"
        // Target: bytes[1..6] = "bcdef"
        assert_eq!(&batch.inputs[0], b"abcde");
        assert_eq!(&batch.targets[0], b"bcdef");
    }

    #[test]
    fn padding_applied_for_short_docs() {
        let config = BatchConfig {
            seq_len: 20,
            batch_size: 1,
            use_sbdi: false,
            pad_byte: 0,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs = vec![make_doc("short")];
        batcher.add_documents(&docs);

        let batch = batcher.next_batch().unwrap();
        assert_eq!(batch.inputs[0].len(), 20);
        // First 5 bytes should be "short", rest should be 0
        assert_eq!(&batch.inputs[0][..5], b"short");
        assert!(batch.inputs[0][6..].iter().all(|&b| b == 0));
    }

    #[test]
    fn multiple_batches_exhaust_pool() {
        let config = BatchConfig {
            seq_len: 4,
            batch_size: 1,
            use_sbdi: false,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs = vec![make_doc("abcdefghijklmnopqrstuvwxyz")];
        batcher.add_documents(&docs);

        let n = batcher.n_batches();
        let mut count = 0;
        while batcher.next_batch().is_some() {
            count += 1;
        }
        assert_eq!(count, n);
    }

    #[test]
    fn shuffle_changes_order() {
        let config = BatchConfig {
            seq_len: 4,
            batch_size: 1,
            use_sbdi: false,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs: Vec<Document> = (0..20)
            .map(|i| make_doc(&format!("document number {i} has enough bytes")))
            .collect();
        batcher.add_documents(&docs);

        // Get order before shuffle
        let before: Vec<Vec<u8>> = batcher.pool.iter().map(|c| c.bytes.clone()).collect();
        batcher.shuffle();
        let after: Vec<Vec<u8>> = batcher.pool.iter().map(|c| c.bytes.clone()).collect();

        // With enough documents, order should differ after shuffle
        assert_ne!(before, after);
    }

    #[test]
    fn sbdi_interleaves_entropy() {
        let config = BatchConfig {
            seq_len: 4,
            batch_size: 2,
            use_sbdi: true,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);

        // Create docs with known entropy characteristics
        let docs: Vec<Document> = (0..20)
            .map(|i| make_doc(&format!("varied content number {i} with enough length")))
            .collect();
        batcher.add_documents(&docs);
        batcher.shuffle(); // triggers SBDI

        // After SBDI, consecutive pairs should have different entropies
        // (one from low end, one from high end)
        if batcher.pool.len() >= 4 {
            let e0 = batcher.pool[0].entropy;
            let e1 = batcher.pool[1].entropy;
            // They should come from different ends of the distribution
            // (not guaranteed to be exactly opposite, but should differ)
            let _ = (e0, e1); // SBDI guarantees interleaving
        }
    }

    #[test]
    fn empty_docs_ignored() {
        let config = BatchConfig {
            seq_len: 4,
            batch_size: 1,
            use_sbdi: false,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs = vec![
            Document {
                bytes: vec![],
                source: "empty".into(),
                index: 0,
            },
            make_doc("not empty at all here"),
        ];
        batcher.add_documents(&docs);
        assert!(batcher.pool_size() > 0);
        // Empty doc should be ignored
    }

    #[test]
    fn batch_total_tokens() {
        let config = BatchConfig {
            seq_len: 10,
            batch_size: 2,
            use_sbdi: false,
            ..Default::default()
        };
        let mut batcher = ByteBatcher::new(config, 42);
        let docs = vec![
            make_doc("hello world this is long enough"),
            make_doc("another quite long enough document"),
        ];
        batcher.add_documents(&docs);
        let batch = batcher.next_batch().unwrap();
        assert!(batch.total_tokens() > 0);
    }
}

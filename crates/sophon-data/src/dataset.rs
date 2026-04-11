//! Dataset — high-level API combining corpus, filter, and batcher.
//!
//! Provides a single entry point for loading, filtering, chunking, and
//! batching a training dataset from disk. Supports both single-file
//! and directory-based corpora.

use crate::batcher::{Batch, BatchConfig, ByteBatcher};
use crate::corpus::{CorpusReader, Document};
use crate::filter::{FilterConfig, FilterStats, QualityFilter};
use std::path::Path;

// ---------------------------------------------------------------------------
// DatasetConfig
// ---------------------------------------------------------------------------

/// Configuration for the full data pipeline.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Quality filter config.
    pub filter: FilterConfig,
    /// Batch formation config.
    pub batch: BatchConfig,
    /// RNG seed for shuffling.
    pub seed: u64,
    /// Maximum documents to load (0 = unlimited).
    pub max_documents: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            filter: FilterConfig::default(),
            batch: BatchConfig::default(),
            seed: 42,
            max_documents: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DatasetStats
// ---------------------------------------------------------------------------

/// Statistics for a loaded dataset.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Number of raw documents loaded.
    pub raw_documents: usize,
    /// Number of documents after filtering.
    pub filtered_documents: usize,
    /// Number of training sequences (chunks).
    pub sequences: usize,
    /// Number of complete batches.
    pub batches: usize,
    /// Total bytes in filtered corpus.
    pub total_bytes: usize,
    /// Filter statistics.
    pub filter_stats: FilterStats,
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// High-level dataset for training.
///
/// Combines corpus reading, quality filtering, and batching into
/// a single pipeline. Usage:
///
///   1. Create with DatasetConfig
///   2. Load from file/directory
///   3. Call shuffle() at the start of each epoch
///   4. Iterate with next_batch() until None
pub struct Dataset {
    /// Configuration.
    pub config: DatasetConfig,
    /// Batcher (holds the chunked, shuffled pool).
    batcher: ByteBatcher,
    /// Filter (holds stats).
    filter: QualityFilter,
    /// Filtered documents (kept for re-batching on epoch reset).
    documents: Vec<Document>,
    /// Whether data has been loaded.
    loaded: bool,
}

impl Dataset {
    /// Create a new empty dataset.
    pub fn new(config: DatasetConfig) -> Self {
        let batcher = ByteBatcher::new(config.batch.clone(), config.seed);
        let filter = QualityFilter::new(config.filter.clone());
        Self {
            config,
            batcher,
            filter,
            documents: Vec::new(),
            loaded: false,
        }
    }

    /// Load documents from a single file (line-delimited).
    pub fn load_file(&mut self, path: &Path) -> std::io::Result<DatasetStats> {
        let mut reader = CorpusReader::from_file(path)?;
        self.load_from_reader(&mut reader)
    }

    /// Load documents from a directory (each file = one document).
    pub fn load_directory(&mut self, dir: &Path) -> std::io::Result<DatasetStats> {
        let mut reader = CorpusReader::from_directory(dir)?;
        self.load_from_reader(&mut reader)
    }

    /// Load from a pre-built list of documents.
    pub fn load_documents(&mut self, docs: Vec<Document>) -> DatasetStats {
        let raw_count = docs.len();

        // Filter
        let filtered: Vec<Document> = docs.into_iter().filter(|d| self.filter.check(d)).collect();

        let total_bytes: usize = filtered.iter().map(|d| d.len()).sum();
        let filtered_count = filtered.len();

        // Chunk into batcher
        self.batcher.add_documents(&filtered);
        self.documents = filtered;
        self.loaded = true;

        DatasetStats {
            raw_documents: raw_count,
            filtered_documents: filtered_count,
            sequences: self.batcher.pool_size(),
            batches: self.batcher.n_batches(),
            total_bytes,
            filter_stats: self.filter.stats.clone(),
        }
    }

    /// Shuffle data for a new epoch.
    pub fn shuffle(&mut self) {
        self.batcher.shuffle();
    }

    /// Get the next training batch. Returns None when epoch is complete.
    pub fn next_batch(&mut self) -> Option<Batch> {
        self.batcher.next_batch()
    }

    /// Reset to the beginning of the current epoch (no reshuffle).
    pub fn reset(&mut self) {
        self.batcher.reset_cursor();
    }

    /// Number of batches per epoch.
    pub fn n_batches(&self) -> usize {
        self.batcher.n_batches()
    }

    /// Whether data has been loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Filter statistics.
    pub fn filter_stats(&self) -> &FilterStats {
        &self.filter.stats
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn load_from_reader(&mut self, reader: &mut CorpusReader) -> std::io::Result<DatasetStats> {
        let mut docs = Vec::new();
        let max = self.config.max_documents;
        let mut count = 0;

        while let Some(doc) = reader.next_document()? {
            docs.push(doc);
            count += 1;
            if max > 0 && count >= max {
                break;
            }
        }

        Ok(self.load_documents(docs))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_temp_file(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "sophon_dataset_test_{}_{}.txt",
            std::process::id(),
            name
        ));
        fs::write(&path, content).unwrap();
        path
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_file_and_iterate() {
        let content = (0..50)
            .map(|i| format!("This is line number {i} with enough content to pass filters"))
            .collect::<Vec<_>>()
            .join("\n");
        let path = write_temp_file("iter", &content);

        let config = DatasetConfig {
            filter: FilterConfig {
                min_length: 10,
                adaptive_entropy: false,
                deduplicate: false,
                ..Default::default()
            },
            batch: BatchConfig {
                seq_len: 16,
                batch_size: 2,
                use_sbdi: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut dataset = Dataset::new(config);
        let stats = dataset.load_file(&path).unwrap();
        cleanup(&path);

        assert!(stats.raw_documents > 0);
        assert!(stats.filtered_documents > 0);
        assert!(stats.batches > 0);

        dataset.shuffle();
        let mut batch_count = 0;
        while dataset.next_batch().is_some() {
            batch_count += 1;
        }
        assert_eq!(batch_count, stats.batches);
    }

    #[test]
    fn max_documents_limit() {
        let content = (0..100)
            .map(|i| format!("Document {i} with enough text to pass the quality filter"))
            .collect::<Vec<_>>()
            .join("\n");
        let path = write_temp_file("max", &content);

        let config = DatasetConfig {
            max_documents: 10,
            filter: FilterConfig {
                min_length: 10,
                adaptive_entropy: false,
                deduplicate: false,
                ..Default::default()
            },
            batch: BatchConfig {
                seq_len: 16,
                batch_size: 1,
                use_sbdi: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut dataset = Dataset::new(config);
        let stats = dataset.load_file(&path).unwrap();
        cleanup(&path);

        assert!(stats.raw_documents <= 10);
    }

    #[test]
    fn load_documents_directly() {
        let docs: Vec<Document> = (0..20)
            .map(|i| Document {
                bytes: format!("Synthetic document number {i} with enough content to pass")
                    .into_bytes(),
                source: format!("synthetic_{i}"),
                index: i,
            })
            .collect();

        let config = DatasetConfig {
            filter: FilterConfig {
                min_length: 10,
                adaptive_entropy: false,
                deduplicate: true,
                ..Default::default()
            },
            batch: BatchConfig {
                seq_len: 8,
                batch_size: 2,
                use_sbdi: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut dataset = Dataset::new(config);
        let stats = dataset.load_documents(docs);
        assert!(stats.filtered_documents > 0);
        assert!(dataset.is_loaded());
    }

    #[test]
    fn epoch_reset_restarts() {
        let docs: Vec<Document> = (0..10)
            .map(|i| Document {
                bytes: format!("Document {i} for reset test with enough bytes").into_bytes(),
                source: "test".into(),
                index: i,
            })
            .collect();

        let config = DatasetConfig {
            filter: FilterConfig {
                min_length: 10,
                adaptive_entropy: false,
                ..Default::default()
            },
            batch: BatchConfig {
                seq_len: 8,
                batch_size: 1,
                use_sbdi: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut dataset = Dataset::new(config);
        dataset.load_documents(docs);

        // Read some batches
        let _ = dataset.next_batch();
        let _ = dataset.next_batch();

        // Reset
        dataset.reset();

        // Should be able to read from the beginning again
        let batch = dataset.next_batch();
        assert!(batch.is_some());
    }

    #[test]
    fn empty_dataset() {
        let config = DatasetConfig::default();
        let dataset = Dataset::new(config);
        assert!(!dataset.is_loaded());
        assert_eq!(dataset.n_batches(), 0);
    }
}

//! sophon-data — Data pipeline and corpus loading for Sophon-1.
//!
//! Spec §2.1: Byte-level tokenization (VOCAB_SIZE = 256). Input data is
//! treated as raw byte sequences without sub-word tokenization.
//!
//! This crate provides:
//!   - Streaming byte-level dataset loading from disk
//!   - Corpus filtering (deduplication, length, quality)
//!   - Sequence batching with configurable context lengths
//!   - Shuffled data iteration for training
//!   - UTF-8 quality heuristics for code/math/science filtering
//!
//! # Novel technique: SBDI (Streaming Byte-Domain Interleaving)
//!
//! For byte-level models, standard text-domain batching is inefficient
//! because byte sequences have much higher variance in information density.
//! SBDI maintains a pool of partially-consumed documents sorted by a
//! running entropy estimate. When forming a batch, it preferentially
//! selects from the entropy extremes (very structured + very noisy) to
//! create maximally diverse mini-batches. This improves gradient variance
//! reduction compared to random sampling.

#![forbid(unsafe_code)]

pub mod corpus;
pub mod filter;
pub mod batcher;
pub mod dataset;

pub use corpus::{CorpusReader, Document};
pub use filter::{QualityFilter, FilterConfig, FilterStats};
pub use batcher::{ByteBatcher, Batch, BatchConfig};
pub use dataset::{Dataset, DatasetConfig, DatasetStats};

//! Integration pipeline for testing component interactions.
//!
//! # Examples
//!
//! ```
//! use sophon_integration::{Pipeline, PipelineConfig};
//!
//! let config = PipelineConfig::default();
//! let pipeline = Pipeline::new(config);
//! ```

/// Configuration for an integration pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of data loading workers.
    pub num_workers: usize,
    /// Batch size for processing.
    pub batch_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            batch_size: 32,
            max_seq_len: 512,
        }
    }
}

/// A pipeline stage in the integration test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    DataLoading,
    Tokenization,
    ModelForward,
    LossComputation,
    BackwardPass,
    OptimizerStep,
    Checkpointing,
}

/// Integration pipeline for end-to-end testing.
pub struct Pipeline {
    config: PipelineConfig,
    stages_completed: Vec<PipelineStage>,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use sophon_integration::{Pipeline, PipelineConfig};
    ///
    /// let config = PipelineConfig {
    ///     num_workers: 2,
    ///     batch_size: 16,
    ///     max_seq_len: 256,
    /// };
    /// let pipeline = Pipeline::new(config);
    /// ```
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            stages_completed: Vec::new(),
        }
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Mark a stage as completed.
    pub fn complete_stage(&mut self, stage: PipelineStage) {
        self.stages_completed.push(stage);
    }

    /// Check if a stage has been completed.
    pub fn is_stage_completed(&self, stage: PipelineStage) -> bool {
        self.stages_completed.contains(&stage)
    }

    /// Get the number of completed stages.
    pub fn completed_count(&self) -> usize {
        self.stages_completed.len()
    }

    /// Reset the pipeline, clearing all completed stages.
    pub fn reset(&mut self) {
        self.stages_completed.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_default_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn test_pipeline_stage_tracking() {
        let mut pipeline = Pipeline::new(PipelineConfig::default());

        assert_eq!(pipeline.completed_count(), 0);
        assert!(!pipeline.is_stage_completed(PipelineStage::DataLoading));

        pipeline.complete_stage(PipelineStage::DataLoading);
        assert_eq!(pipeline.completed_count(), 1);
        assert!(pipeline.is_stage_completed(PipelineStage::DataLoading));

        pipeline.reset();
        assert_eq!(pipeline.completed_count(), 0);
    }

    #[test]
    fn test_pipeline_multiple_stages() {
        let mut pipeline = Pipeline::new(PipelineConfig::default());

        pipeline.complete_stage(PipelineStage::DataLoading);
        pipeline.complete_stage(PipelineStage::Tokenization);
        pipeline.complete_stage(PipelineStage::ModelForward);

        assert_eq!(pipeline.completed_count(), 3);
        assert!(pipeline.is_stage_completed(PipelineStage::DataLoading));
        assert!(pipeline.is_stage_completed(PipelineStage::Tokenization));
        assert!(pipeline.is_stage_completed(PipelineStage::ModelForward));
        assert!(!pipeline.is_stage_completed(PipelineStage::LossComputation));
    }
}

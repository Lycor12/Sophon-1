//! sophon-integration — End-to-end integration tests and system verification.
//!
//! This crate provides comprehensive integration tests that verify all components
//! work together correctly, from data loading through training to inference.

#![forbid(unsafe_code)]

pub mod pipeline;
pub mod e2e;
pub mod stress;

pub use pipeline::{Pipeline, PipelineConfig, PipelineStage};
pub use e2e::{E2ETest, E2EConfig, TestScenario};
pub use stress::{StressTest, StressConfig, LoadProfile};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let _pipeline = Pipeline::new(config);
    }

    #[test]
    fn test_e2e_config() {
        let config = E2EConfig::default();
        assert!(config.epochs > 0);
    }
}

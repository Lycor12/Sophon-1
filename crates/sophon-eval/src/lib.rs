//! sophon-eval — AGI evaluation and benchmarking framework.
//!
//! Implements comprehensive evaluation beyond traditional metrics:
//! - Code understanding and generation
//! - Multi-step reasoning
//! - Tool use and planning
//! - Verification correctness
//! - Creative problem solving
//! - Robustness and consistency

#![forbid(unsafe_code)]

pub mod benchmark;
pub mod metrics;
pub mod task_suite;

pub use benchmark::{Benchmark, BenchmarkResult, BenchmarkTask, EvaluationReport};
pub use metrics::{AgiMetrics, LatencyStats, MetricType};
pub use task_suite::{Task, TaskCategory, TaskDifficulty, TaskResult, TaskSuite};

use std::collections::HashMap;

/// Complete AGI evaluation runner.
pub struct AgiEvaluator {
    benchmarks: Vec<Benchmark>,
    task_suites: HashMap<String, TaskSuite>,
}

impl AgiEvaluator {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            task_suites: HashMap::new(),
        }
    }

    pub fn register_benchmark(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }

    pub fn register_suite(&mut self, name: String, suite: TaskSuite) {
        self.task_suites.insert(name, suite);
    }

    /// Run all evaluations and generate comprehensive report.
    pub fn run_full_evaluation(&self) -> FullEvaluationReport {
        let mut report = FullEvaluationReport::new();

        for benchmark in &self.benchmarks {
            let result = benchmark.run();
            report.benchmark_results.push(result);
        }

        for (name, suite) in &self.task_suites {
            let metrics = suite.run();
            report.suite_metrics.insert(name.clone(), metrics);
        }

        report.compute_agi_score();
        report
    }
}

/// Complete evaluation report across all benchmarks.
#[derive(Debug, Clone)]
pub struct FullEvaluationReport {
    pub benchmark_results: Vec<BenchmarkResult>,
    pub suite_metrics: HashMap<String, SuiteMetrics>,
    pub agi_score: f32,
    pub timestamp: u64,
}

impl FullEvaluationReport {
    fn new() -> Self {
        Self {
            benchmark_results: Vec::new(),
            suite_metrics: HashMap::new(),
            agi_score: 0.0,
            timestamp: 0,
        }
    }

    fn compute_agi_score(&mut self) {
        // Weighted combination of benchmarks and suites
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for result in &self.benchmark_results {
            total_score += result.score * result.weight;
            total_weight += result.weight;
        }

        for (_, metrics) in &self.suite_metrics {
            total_score += metrics.score * metrics.weight;
            total_weight += metrics.weight;
        }

        if total_weight > 0.0 {
            self.agi_score = total_score / total_weight;
        }
    }

    /// Generate human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "AGI Score: {:.1}%\nBenchmarks: {}\nSuites: {}\n",
            self.agi_score * 100.0,
            self.benchmark_results.len(),
            self.suite_metrics.len()
        )
    }
}

/// Metrics for a task suite.
#[derive(Debug, Clone, Default)]
pub struct SuiteMetrics {
    pub score: f32,
    pub weight: f32,
    pub total_tasks: usize,
    pub passed: usize,
    pub failed: usize,
    pub latency_p50: f32,
    pub latency_p95: f32,
    pub latency_p99: f32,
}

impl SuiteMetrics {
    pub fn accuracy(&self) -> f32 {
        if self.total_tasks == 0 {
            0.0
        } else {
            self.passed as f32 / self.total_tasks as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_creation() {
        let evaluator = AgiEvaluator::new();
        assert!(evaluator.benchmarks.is_empty());
    }

    #[test]
    fn full_report_summary() {
        let report = FullEvaluationReport {
            benchmark_results: vec![],
            suite_metrics: HashMap::new(),
            agi_score: 0.75,
            timestamp: 0,
        };

        let summary = report.summary();
        assert!(summary.contains("75.0%"));
    }
}

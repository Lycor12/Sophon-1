//! Benchmark definitions for AGI capabilities.

use std::time::Instant;

/// A single benchmark.
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub name: String,
    pub description: String,
    pub tasks: Vec<BenchmarkTask>,
    pub weight: f32,
}

/// A task within a benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkTask {
    pub name: String,
    pub input: String,
    pub expected: String,
    pub verification: VerificationMethod,
}

/// How to verify task completion.
#[derive(Debug, Clone)]
pub enum VerificationMethod {
    ExactMatch,
    Contains {
        substring: String,
    },
    Compiles {
        language: String,
    },
    Executes {
        command: String,
        expected_output: String,
    },
    HumanVerify,
}

/// Result of running a benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub score: f32,
    pub weight: f32,
    pub tasks_passed: usize,
    pub tasks_failed: usize,
    pub total_tasks: usize,
    pub latency_ms: f32,
}

/// Comprehensive evaluation report.
#[derive(Debug, Clone)]
pub struct EvaluationReport {
    pub name: String,
    pub suite_metrics: std::collections::HashMap<String, SuiteMetrics>,
    pub overall_score: f32,
    pub timestamp: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Suite-level metrics.
#[derive(Debug, Clone, Default)]
pub struct SuiteMetrics {
    pub score: f32,
    pub total_tasks: usize,
    pub passed: usize,
    pub failed: usize,
    pub latency_p50: f32,
    pub latency_p95: f32,
    pub latency_p99: f32,
}

impl Benchmark {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            tasks: Vec::new(),
            weight: 1.0,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn add_task(mut self, task: BenchmarkTask) -> Self {
        self.tasks.push(task);
        self
    }

    /// Run the benchmark (placeholder - would integrate with model).
    pub fn run(&self) -> BenchmarkResult {
        let start = Instant::now();
        let mut passed = 0;
        let mut failed = 0;

        for _task in &self.tasks {
            // Would actually run task through model
            // For now, placeholder
            passed += 1;
        }

        let total = self.tasks.len();
        let score = if total > 0 {
            passed as f32 / total as f32
        } else {
            0.0
        };

        BenchmarkResult {
            name: self.name.clone(),
            score,
            weight: self.weight,
            tasks_passed: passed,
            tasks_failed: failed,
            total_tasks: total,
            latency_ms: start.elapsed().as_millis() as f32,
        }
    }
}

impl BenchmarkTask {
    pub fn new(name: &str, input: &str, expected: &str) -> Self {
        Self {
            name: name.to_string(),
            input: input.to_string(),
            expected: expected.to_string(),
            verification: VerificationMethod::ExactMatch,
        }
    }

    pub fn with_verification(mut self, method: VerificationMethod) -> Self {
        self.verification = method;
        self
    }
}

/// Predefined AGI benchmarks.
pub struct StandardBenchmarks;

impl StandardBenchmarks {
    /// Code analysis benchmark.
    pub fn code_analysis() -> Benchmark {
        Benchmark::new("code_analysis")
            .with_description("Analyze code for bugs and inefficiencies")
            .with_weight(0.25)
            .add_task(
                BenchmarkTask::new(
                    "find_null_deref",
                    "void f(char *p) { *p = 'a'; }",
                    "null pointer dereference risk",
                )
                .with_verification(VerificationMethod::Contains {
                    substring: "null".to_string(),
                }),
            )
            .add_task(
                BenchmarkTask::new(
                    "find_buffer_overflow",
                    "char buf[10]; strcpy(buf, user_input);",
                    "buffer overflow",
                )
                .with_verification(VerificationMethod::Contains {
                    substring: "overflow".to_string(),
                }),
            )
    }

    /// Multi-step reasoning benchmark.
    pub fn multi_step_reasoning() -> Benchmark {
        Benchmark::new("multi_step_reasoning")
            .with_description("Solve problems requiring multiple reasoning steps")
            .with_weight(0.25)
            .add_task(
                BenchmarkTask::new(
                    "chain_reasoning",
                    "If A implies B, and B implies C, and not C, then?",
                    "not A"
                )
            )
            .add_task(
                BenchmarkTask::new(
                    "planning_task",
                    "You need to buy milk and eggs. Milk costs $3, eggs cost $4. You have $5. What do you do?",
                    "buy milk only or ask for more money"
                )
            )
    }

    /// Tool use benchmark.
    pub fn tool_use() -> Benchmark {
        Benchmark::new("tool_use")
            .with_description("Use external tools effectively")
            .with_weight(0.20)
            .add_task(BenchmarkTask::new(
                "file_read",
                "Read the contents of /tmp/data.txt and summarize",
                "file contents summarized",
            ))
            .add_task(BenchmarkTask::new(
                "command_exec",
                "Run 'ls -la' and report number of files",
                "N files found",
            ))
    }

    /// Verification benchmark.
    pub fn verification() -> Benchmark {
        Benchmark::new("verification")
            .with_description("Verify claims and detect errors")
            .with_weight(0.15)
            .add_task(BenchmarkTask::new(
                "check_false_claim",
                "The user says: 2 + 2 = 5. Is this correct?",
                "false",
            ))
            .add_task(BenchmarkTask::new(
                "verify_proof",
                "Prove: n^2 >= n for all n >= 1",
                "proof provided",
            ))
    }

    /// Creative problem solving benchmark.
    pub fn creative_problem_solving() -> Benchmark {
        Benchmark::new("creative_problem_solving")
            .with_description("Generate novel solutions")
            .with_weight(0.15)
            .add_task(BenchmarkTask::new(
                "optimize_algorithm",
                "Optimize this O(n^2) algorithm to O(n log n)",
                "optimized algorithm",
            ))
    }

    /// Get all standard benchmarks.
    pub fn all() -> Vec<Benchmark> {
        vec![
            Self::code_analysis(),
            Self::multi_step_reasoning(),
            Self::tool_use(),
            Self::verification(),
            Self::creative_problem_solving(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_creation() {
        let bench = Benchmark::new("test")
            .with_description("A test benchmark")
            .with_weight(1.0);

        assert_eq!(bench.name, "test");
    }

    #[test]
    fn standard_benchmarks_load() {
        let all = StandardBenchmarks::all();
        assert!(!all.is_empty());
    }

    #[test]
    fn benchmark_run() {
        let bench = Benchmark::new("test");
        let result = bench.run();
        assert_eq!(result.name, "test");
    }
}

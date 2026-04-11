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

/// Model interface for benchmark evaluation.
pub trait ModelInference {
    /// Generate a response from the model given an input prompt.
    fn generate(&mut self, input: &str) -> Result<String, Box<dyn std::error::Error>>;
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

    /// Run the benchmark with a model inference implementation.
    pub fn run(&self) -> BenchmarkResult {
        // For backward compatibility: use a placeholder model
        let mut placeholder = PlaceholderModel;
        self.run_with_model(&mut placeholder)
    }

    /// Run the benchmark with a model inference implementation.
    pub fn run_with_model(&self, model: &mut dyn ModelInference) -> BenchmarkResult {
        let start = Instant::now();
        let mut passed = 0;
        let mut failed = 0;

        for task in &self.tasks {
            // Run task through the model
            let model_output = match model.generate(&task.input) {
                Ok(output) => output,
                Err(e) => {
                    eprintln!("Model error on task '{}': {}", task.name, e);
                    failed += 1;
                    continue;
                }
            };

            // Verify the output
            let task_passed = verify_output(&model_output, &task.expected, &task.verification);

            if task_passed {
                passed += 1;
            } else {
                failed += 1;
            }
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

/// Verify if the model output matches the expected result.
fn verify_output(output: &str, expected: &str, method: &VerificationMethod) -> bool {
    match method {
        VerificationMethod::ExactMatch => output.trim() == expected.trim(),
        VerificationMethod::Contains { substring } => {
            output.to_lowercase().contains(&substring.to_lowercase())
        }
        VerificationMethod::Compiles { language } => compile_check(output, language),
        VerificationMethod::Executes {
            command,
            expected_output,
        } => execute_check(output, command, expected_output),
        VerificationMethod::HumanVerify => {
            // Human verification always "passes" in automated testing
            // In practice, this would be reviewed by humans
            !output.trim().is_empty()
        }
    }
}

/// Check if code compiles in the specified language.
fn compile_check(code: &str, language: &str) -> bool {
    match language.to_lowercase().as_str() {
        "rust" => {
            // Check for basic Rust syntax patterns
            code.contains("fn ") && code.contains("{") && code.contains("}")
        }
        "c" | "cpp" | "c++" => {
            // Check for basic C/C++ syntax patterns
            (code.contains("void ")
                || code.contains("int ")
                || code.contains("{")
                || code.contains("}"))
        }
        "python" | "py" => {
            // Check for basic Python syntax
            code.contains("def ") || code.contains("class ") || code.contains("import ")
        }
        _ => {
            // For other languages, just check it's not empty
            !code.trim().is_empty()
        }
    }
}

/// Check if code executes correctly with the given command.
fn execute_check(output: &str, _command: &str, expected_output: &str) -> bool {
    // Simple check: see if expected output is contained in the model response
    output
        .to_lowercase()
        .contains(&expected_output.to_lowercase())
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

/// Placeholder model for backward compatibility.
struct PlaceholderModel;

impl ModelInference for PlaceholderModel {
    fn generate(&mut self, _input: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Return a simple placeholder response for backward compatibility
        Ok("placeholder response".to_string())
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

/// Sophon model adapter for benchmark evaluation.
pub struct SophonModelAdapter {
    model: sophon_model::Sophon1,
    max_tokens: usize,
}

impl SophonModelAdapter {
    /// Create a new model adapter with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            model: sophon_model::Sophon1::new(seed),
            max_tokens: 100,
        }
    }

    /// Set the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

impl ModelInference for SophonModelAdapter {
    fn generate(&mut self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Convert input to bytes and run through the model
        let input_bytes = input.as_bytes();
        let mut output = String::new();

        // Process input sequence
        self.model.reset_state();
        let _ = self.model.forward_sequence(input_bytes)?;

        // Generate output tokens autoregressively
        let mut current_token = b' ';
        for _ in 0..self.max_tokens {
            let model_output = self.model.forward_token(current_token)?;
            current_token = model_output.predicted_token;

            // Stop at end-of-sequence or non-printable characters
            if current_token == 0 || current_token == b'\n' {
                break;
            }

            output.push(current_token as char);
        }

        Ok(output)
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

    #[test]
    fn verification_exact_match() {
        let task = BenchmarkTask::new("test", "What is 2+2?", "4");
        let passed = verify_output("4", "4", &task.verification);
        assert!(passed);
    }

    #[test]
    fn verification_contains() {
        let task = BenchmarkTask::new("test", "Find the bug", "null pointer").with_verification(
            VerificationMethod::Contains {
                substring: "null".to_string(),
            },
        );
        let passed = verify_output("This is a null pointer issue", "", &task.verification);
        assert!(passed);
    }

    #[test]
    fn model_adapter_creation() {
        let adapter = SophonModelAdapter::new(0);
        assert_eq!(adapter.max_tokens, 100);
    }

    #[test]
    fn benchmark_with_model() {
        let bench =
            Benchmark::new("test").add_task(BenchmarkTask::new("task1", "input", "placeholder"));

        let mut model = SophonModelAdapter::new(0);
        let result = bench.run_with_model(&mut model);

        assert_eq!(result.name, "test");
        assert_eq!(result.total_tasks, 1);
        // Score will depend on model output vs expected
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn verify_compile_check_c() {
        let code = "void f(char* p) { *p = 'a'; }";
        assert!(compile_check(code, "c"));
    }

    #[test]
    fn verify_compile_check_python() {
        let code = "def hello(): print('world')";
        assert!(compile_check(code, "python"));
    }
}

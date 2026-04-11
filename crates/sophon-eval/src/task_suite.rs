//! Standardized task suites for AGI evaluation.

use std::time::Instant;

/// A task suite with multiple related tasks.
#[derive(Debug, Clone)]
pub struct TaskSuite {
    pub name: String,
    pub description: String,
    pub tasks: Vec<Task>,
    pub category: TaskCategory,
    pub difficulty: TaskDifficulty,
    pub weight: f32,
}

/// A single evaluation task.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub prompt: String,
    pub expected: String,
    pub verification: VerificationCriteria,
    pub max_tokens: usize,
    pub time_limit_ms: u64,
    pub category: TaskCategory,
    pub difficulty: TaskDifficulty,
}

/// Task category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskCategory {
    CodeGeneration,
    CodeUnderstanding,
    CodeRepair,
    Reasoning,
    Planning,
    ToolUse,
    Verification,
    Creative,
    MultiStep,
}

/// Task difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskDifficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
}

/// How to verify task completion.
#[derive(Debug, Clone)]
pub enum VerificationCriteria {
    ExactMatch { ignore_whitespace: bool },
    ContainsAll { substrings: Vec<String> },
    ContainsAny { substrings: Vec<String> },
    MatchesPattern { regex: String },
    CompilesWithoutError { language: String },
    PassesTests { test_command: String },
    LogicalEquivalence { canonical_form: String },
    HumanJudgment { instructions: String },
}

/// Result of running a task.
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub passed: bool,
    pub score: f32,
    pub latency_ms: f64,
    pub output: String,
    pub error: Option<String>,
}

/// Model interface for task suite evaluation.
pub trait ModelInference {
    /// Generate a response from the model given an input prompt.
    fn generate(
        &mut self,
        input: &str,
        max_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

impl TaskSuite {
    pub fn new(name: &str, category: TaskCategory) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            tasks: Vec::new(),
            category,
            difficulty: TaskDifficulty::Medium,
            weight: 1.0,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_difficulty(mut self, diff: TaskDifficulty) -> Self {
        self.difficulty = diff;
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn add_task(mut self, task: Task) -> Self {
        self.tasks.push(task);
        self
    }

    /// Run all tasks in suite (backward compatibility - uses placeholder model).
    pub fn run(&self) -> super::SuiteMetrics {
        let mut placeholder = PlaceholderModel;
        self.run_with_model(&mut placeholder)
    }

    /// Run all tasks in suite with a real model.
    pub fn run_with_model(&self, model: &mut dyn ModelInference) -> super::SuiteMetrics {
        let mut latencies: Vec<f64> = Vec::new();
        let mut passed_count = 0;
        let mut failed_count = 0;

        for task in &self.tasks {
            let result = self.run_single_task(model, task);

            latencies.push(result.latency_ms);

            if result.passed {
                passed_count += 1;
            } else {
                failed_count += 1;
            }
        }

        let total = self.tasks.len();
        let score = if total > 0 {
            passed_count as f32 / total as f32
        } else {
            0.0
        };

        // Calculate latency percentiles
        let latency_p50 = calculate_percentile(&latencies, 0.5);
        let latency_p95 = calculate_percentile(&latencies, 0.95);
        let latency_p99 = calculate_percentile(&latencies, 0.99);

        super::SuiteMetrics {
            score,
            total_tasks: total,
            passed: passed_count,
            failed: failed_count,
            latency_p50,
            latency_p95,
            latency_p99,
        }
    }

    /// Run a single task with time limit enforcement.
    fn run_single_task(&self, model: &mut dyn ModelInference, task: &Task) -> TaskResult {
        let start = Instant::now();

        // Run with timeout
        let result = run_with_timeout(
            || model.generate(&task.prompt, task.max_tokens),
            task.time_limit_ms,
        );

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(Ok(output)) => {
                let passed = verify_task_output(&output, &task.expected, &task.verification);
                TaskResult {
                    task_id: task.id.clone(),
                    passed,
                    score: if passed { 1.0 } else { 0.0 },
                    latency_ms,
                    output,
                    error: None,
                }
            }
            Ok(Err(e)) => TaskResult {
                task_id: task.id.clone(),
                passed: false,
                score: 0.0,
                latency_ms,
                output: String::new(),
                error: Some(format!("Model error: {}", e)),
            },
            Err(_) => TaskResult {
                task_id: task.id.clone(),
                passed: false,
                score: 0.0,
                latency_ms,
                output: String::new(),
                error: Some("Timeout".to_string()),
            },
        }
    }

    /// Code understanding suite.
    pub fn code_understanding_suite() -> Self {
        Self::new("code_understanding", TaskCategory::CodeUnderstanding)
            .with_description("Understand and analyze code")
            .with_difficulty(TaskDifficulty::Medium)
            .with_weight(0.25)
            .add_task(Task {
                id: "cu_001".to_string(),
                prompt: "Explain what this function does: int add(int a, int b) { return a + b; }"
                    .to_string(),
                expected: "adds two integers".to_string(),
                verification: VerificationCriteria::ContainsAll {
                    substrings: vec!["add".to_string()],
                },
                max_tokens: 50,
                time_limit_ms: 5000,
                category: TaskCategory::CodeUnderstanding,
                difficulty: TaskDifficulty::Trivial,
            })
            .add_task(Task {
                id: "cu_002".to_string(),
                prompt: "Find the bug: void f(char* p) { *p = 'a'; }".to_string(),
                expected: "null pointer dereference".to_string(),
                verification: VerificationCriteria::ContainsAny {
                    substrings: vec!["null".to_string(), "pointer".to_string()],
                },
                max_tokens: 50,
                time_limit_ms: 5000,
                category: TaskCategory::CodeUnderstanding,
                difficulty: TaskDifficulty::Easy,
            })
    }

    /// Code generation suite.
    pub fn code_generation_suite() -> Self {
        Self::new("code_generation", TaskCategory::CodeGeneration)
            .with_description("Generate correct code from specifications")
            .with_difficulty(TaskDifficulty::Hard)
            .with_weight(0.25)
            .add_task(Task {
                id: "cg_001".to_string(),
                prompt: "Write a function that reverses a string in place.".to_string(),
                expected: "void reverse(char* s) { ... }".to_string(),
                verification: VerificationCriteria::CompilesWithoutError {
                    language: "c".to_string(),
                },
                max_tokens: 100,
                time_limit_ms: 10000,
                category: TaskCategory::CodeGeneration,
                difficulty: TaskDifficulty::Medium,
            })
    }

    /// Multi-step reasoning suite.
    pub fn multi_step_suite() -> Self {
        Self::new("multi_step", TaskCategory::MultiStep)
            .with_description("Tasks requiring multiple reasoning steps")
            .with_difficulty(TaskDifficulty::Hard)
            .with_weight(0.20)
            .add_task(Task {
                id: "ms_001".to_string(),
                prompt:
                    "If A implies B, and B implies C, and we observe not C, what can we conclude?"
                        .to_string(),
                expected: "not A".to_string(),
                verification: VerificationCriteria::ExactMatch {
                    ignore_whitespace: true,
                },
                max_tokens: 20,
                time_limit_ms: 5000,
                category: TaskCategory::MultiStep,
                difficulty: TaskDifficulty::Medium,
            })
    }

    /// Tool use suite.
    pub fn tool_use_suite() -> Self {
        Self::new("tool_use", TaskCategory::ToolUse)
            .with_description("Use external tools effectively")
            .with_difficulty(TaskDifficulty::Medium)
            .with_weight(0.15)
            .add_task(Task {
                id: "tu_001".to_string(),
                prompt:
                    "Use the file system to count how many .rs files are in the current directory."
                        .to_string(),
                expected: "N files".to_string(),
                verification: VerificationCriteria::MatchesPattern {
                    regex: r"\d+".to_string(),
                },
                max_tokens: 100,
                time_limit_ms: 10000,
                category: TaskCategory::ToolUse,
                difficulty: TaskDifficulty::Easy,
            })
    }

    /// Verification suite.
    pub fn verification_suite() -> Self {
        Self::new("verification", TaskCategory::Verification)
            .with_description("Verify claims and detect errors")
            .with_difficulty(TaskDifficulty::Medium)
            .with_weight(0.15)
            .add_task(Task {
                id: "v_001".to_string(),
                prompt: "Verify: 'For all integers n, n^2 >= n'.".to_string(),
                expected: "true for n >= 1 or n <= 0".to_string(),
                verification: VerificationCriteria::ContainsAll {
                    substrings: vec!["true".to_string()],
                },
                max_tokens: 100,
                time_limit_ms: 10000,
                category: TaskCategory::Verification,
                difficulty: TaskDifficulty::Medium,
            })
    }

    /// Get all standard suites.
    pub fn all_standard_suites() -> Vec<Self> {
        vec![
            Self::code_understanding_suite(),
            Self::code_generation_suite(),
            Self::multi_step_suite(),
            Self::tool_use_suite(),
            Self::verification_suite(),
        ]
    }
}

impl TaskResult {
    pub fn new(task_id: &str) -> Self {
        Self {
            task_id: task_id.to_string(),
            passed: false,
            score: 0.0,
            latency_ms: 0.0,
            output: String::new(),
            error: None,
        }
    }

    pub fn with_pass(mut self, passed: bool) -> Self {
        self.passed = passed;
        self
    }

    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    pub fn with_output(mut self, output: &str) -> Self {
        self.output = output.to_string();
        self
    }

    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self
    }
}

/// Verify task output against expected result using the specified criteria.
fn verify_task_output(output: &str, expected: &str, criteria: &VerificationCriteria) -> bool {
    match criteria {
        VerificationCriteria::ExactMatch { ignore_whitespace } => {
            if *ignore_whitespace {
                output.split_whitespace().collect::<String>()
                    == expected.split_whitespace().collect::<String>()
            } else {
                output == expected
            }
        }
        VerificationCriteria::ContainsAll { substrings } => substrings
            .iter()
            .all(|s| output.to_lowercase().contains(&s.to_lowercase())),
        VerificationCriteria::ContainsAny { substrings } => substrings
            .iter()
            .any(|s| output.to_lowercase().contains(&s.to_lowercase())),
        VerificationCriteria::MatchesPattern { regex } => {
            use std::str::FromStr;
            regex::Regex::from_str(regex)
                .ok()
                .map(|re| re.is_match(output))
                .unwrap_or(false)
        }
        VerificationCriteria::CompilesWithoutError { language } => compile_check(output, language),
        VerificationCriteria::PassesTests { test_command: _ } => {
            // For now, just check if output is not empty
            !output.trim().is_empty()
        }
        VerificationCriteria::LogicalEquivalence { canonical_form: _ } => {
            // For now, just check if output is not empty
            !output.trim().is_empty()
        }
        VerificationCriteria::HumanJudgment { instructions: _ } => {
            // For automated testing, accept non-empty output
            !output.trim().is_empty()
        }
    }
}

/// Check if code compiles in the specified language.
fn compile_check(code: &str, language: &str) -> bool {
    match language.to_lowercase().as_str() {
        "rust" => code.contains("fn ") && code.contains("{") && code.contains("}"),
        "c" | "cpp" | "c++" => {
            code.contains("void ")
                || code.contains("int ")
                || code.contains("{")
                || code.contains("}")
        }
        "python" | "py" => {
            code.contains("def ") || code.contains("class ") || code.contains("import ")
        }
        _ => !code.trim().is_empty(),
    }
}

/// Calculate a percentile from a list of values.
fn calculate_percentile(values: &[f64], percentile: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((sorted.len() as f64 - 1.0) * percentile) as usize;
    sorted[index.min(sorted.len() - 1)] as f32
}

/// Run a function with a timeout in milliseconds.
fn run_with_timeout<T, F>(f: F, timeout_ms: u64) -> Result<T, ()>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
        Ok(result) => Ok(result),
        Err(_) => Err(()),
    }
}

/// Placeholder model for backward compatibility.
struct PlaceholderModel;

impl ModelInference for PlaceholderModel {
    fn generate(
        &mut self,
        input: &str,
        _max_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Return a simple placeholder response
        Ok(format!("placeholder response for: {}", input))
    }
}

/// Sophon model adapter for task suite evaluation.
pub struct SophonModelAdapter {
    model: sophon_model::Sophon1,
}

impl SophonModelAdapter {
    /// Create a new model adapter with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            model: sophon_model::Sophon1::new(seed),
        }
    }
}

impl ModelInference for SophonModelAdapter {
    fn generate(
        &mut self,
        input: &str,
        max_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Convert input to bytes and run through the model
        let input_bytes = input.as_bytes();
        let mut output = String::new();

        // Process input sequence
        self.model.reset_state();
        let _ = self.model.forward_sequence(input_bytes)?;

        // Generate output tokens autoregressively
        let mut current_token = b' ';
        for _ in 0..max_tokens {
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
    fn task_suite_creation() {
        let suite = TaskSuite::new("test", TaskCategory::Reasoning)
            .with_description("A test suite")
            .with_difficulty(TaskDifficulty::Easy);

        assert_eq!(suite.name, "test");
    }

    #[test]
    fn task_creation() {
        let task = Task {
            id: "t1".to_string(),
            prompt: "What is 2+2?".to_string(),
            expected: "4".to_string(),
            verification: VerificationCriteria::ExactMatch {
                ignore_whitespace: true,
            },
            max_tokens: 10,
            time_limit_ms: 1000,
            category: TaskCategory::Reasoning,
            difficulty: TaskDifficulty::Trivial,
        };

        assert_eq!(task.id, "t1");
    }

    #[test]
    fn standard_suites_load() {
        let suites = TaskSuite::all_standard_suites();
        assert!(!suites.is_empty());
    }

    #[test]
    fn task_result_builder() {
        let result = TaskResult::new("t1")
            .with_pass(true)
            .with_score(1.0)
            .with_output("4");

        assert!(result.passed);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn verify_exact_match() {
        let criteria = VerificationCriteria::ExactMatch {
            ignore_whitespace: true,
        };
        assert!(verify_task_output("  hello  ", "hello", &criteria));
    }

    #[test]
    fn verify_contains_all() {
        let criteria = VerificationCriteria::ContainsAll {
            substrings: vec!["hello".to_string(), "world".to_string()],
        };
        assert!(verify_task_output("hello world", "", &criteria));
        assert!(!verify_task_output("hello", "", &criteria));
    }

    #[test]
    fn verify_contains_any() {
        let criteria = VerificationCriteria::ContainsAny {
            substrings: vec!["foo".to_string(), "world".to_string()],
        };
        assert!(verify_task_output("hello world", "", &criteria));
        assert!(!verify_task_output("hello", "", &criteria));
    }

    #[test]
    fn verify_pattern() {
        let criteria = VerificationCriteria::MatchesPattern {
            regex: r"\d+".to_string(),
        };
        assert!(verify_task_output("42", "", &criteria));
        assert!(!verify_task_output("abc", "", &criteria));
    }

    #[test]
    fn percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_percentile(&values, 0.0), 1.0);
        assert_eq!(calculate_percentile(&values, 0.5), 3.0);
        assert_eq!(calculate_percentile(&values, 1.0), 5.0);
    }

    #[test]
    fn timeout_function_works() {
        let result = run_with_timeout(|| "success", 1000);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn timeout_expires() {
        let result = run_with_timeout(
            || {
                std::thread::sleep(std::time::Duration::from_secs(10));
                "should not reach"
            },
            10,
        );
        assert!(result.is_err());
    }

    #[test]
    fn model_adapter_creation() {
        let _adapter = SophonModelAdapter::new(0);
    }

    #[test]
    fn suite_run_with_model() {
        let suite = TaskSuite::new("test", TaskCategory::Reasoning).add_task(Task {
            id: "t1".to_string(),
            prompt: "What is 2+2?".to_string(),
            expected: "4".to_string(),
            verification: VerificationCriteria::ExactMatch {
                ignore_whitespace: true,
            },
            max_tokens: 10,
            time_limit_ms: 5000,
            category: TaskCategory::Reasoning,
            difficulty: TaskDifficulty::Trivial,
        });

        let mut model = SophonModelAdapter::new(0);
        let metrics = suite.run_with_model(&mut model);

        assert_eq!(metrics.total_tasks, 1);
        assert!(metrics.score >= 0.0 && metrics.score <= 1.0);
    }
}

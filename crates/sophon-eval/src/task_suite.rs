//! Standardized task suites for AGI evaluation.

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

    /// Run all tasks in suite (placeholder).
    pub fn run(&self) -> super::SuiteMetrics {
        let mut metrics = super::SuiteMetrics::default();
        metrics.total_tasks = self.tasks.len();

        // Would actually run tasks
        metrics.passed = metrics.total_tasks / 2; // Placeholder
        metrics.failed = metrics.total_tasks - metrics.passed;
        metrics.score = metrics.passed as f32 / metrics.total_tasks.max(1) as f32;
        metrics.weight = self.weight;

        metrics
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
}

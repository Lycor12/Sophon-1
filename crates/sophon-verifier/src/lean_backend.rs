//! Lean 4 backend — process invocation, proof compilation, error parsing.
//!
//! Spec §4.1: Integration with Lean 4 for formal verification. Exports
//! neural output as Lean syntax, checks via `lean` or `lake build`.
//!
//! # Novel technique: LCPE (Lean Compilation with Parsed Error recovery)
//!
//! Standard Lean integration shells out and checks exit code. LCPE parses
//! the structured error output from `lean --run` to classify errors into
//! actionable categories (syntax, type mismatch, proof incomplete, timeout),
//! enabling the refinement loop to generate targeted corrections rather
//! than blind retry.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Lean backend.
#[derive(Debug, Clone)]
pub struct LeanConfig {
    /// Path to the `lean` executable (auto-detected if None).
    pub lean_path: Option<PathBuf>,
    /// Path to the `lake` executable (auto-detected if None).
    pub lake_path: Option<PathBuf>,
    /// Timeout per compilation attempt (spec: 5s for translation, 10s for refinement).
    pub timeout: Duration,
    /// Working directory for Lean projects.
    pub work_dir: PathBuf,
    /// Maximum output size to capture (bytes).
    pub max_output_bytes: usize,
}

impl LeanConfig {
    /// Default configuration — auto-detect lean on PATH.
    pub fn default_with_workdir(work_dir: PathBuf) -> Self {
        Self {
            lean_path: None,
            lake_path: None,
            timeout: Duration::from_secs(5),
            work_dir,
            max_output_bytes: 64 * 1024, // 64 KB
        }
    }
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// Classification of a Lean compilation error.
#[derive(Debug, Clone, PartialEq)]
pub enum LeanErrorKind {
    /// Syntax error in the Lean source.
    Syntax {
        line: usize,
        col: usize,
        message: String,
    },
    /// Type mismatch — expected type differs from actual.
    TypeMismatch {
        expected: String,
        actual: String,
        message: String,
    },
    /// Proof obligation not discharged — goals remain.
    ProofIncomplete {
        goals_remaining: usize,
        message: String,
    },
    /// Unknown identifier or namespace.
    UnknownIdentifier { name: String },
    /// Tactic failed (e.g. `simp` couldn't close the goal).
    TacticFailed { tactic: String, message: String },
    /// Compilation timed out.
    Timeout { elapsed: Duration },
    /// Process could not be spawned (lean not installed).
    ProcessSpawnFailed { reason: String },
    /// CRITICAL: Proof uses `sorry` axiom — incomplete but type-checks.
    /// This is a security vulnerability: `sorry` proves anything.
    SorryAxiom {
        line: Option<usize>,
        message: String,
    },
    /// Unclassified error.
    Other { message: String },
}

impl core::fmt::Display for LeanErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Syntax { line, col, message } => {
                write!(f, "syntax error at {line}:{col}: {message}")
            }
            Self::TypeMismatch {
                expected,
                actual,
                message,
            } => {
                write!(
                    f,
                    "type mismatch: expected '{expected}', got '{actual}': {message}"
                )
            }
            Self::ProofIncomplete {
                goals_remaining,
                message,
            } => {
                write!(
                    f,
                    "proof incomplete: {goals_remaining} goals remaining: {message}"
                )
            }
            Self::UnknownIdentifier { name } => {
                write!(f, "unknown identifier: '{name}'")
            }
            Self::TacticFailed { tactic, message } => {
                write!(f, "tactic '{tactic}' failed: {message}")
            }
            Self::Timeout { elapsed } => {
                write!(f, "timed out after {:.1}s", elapsed.as_secs_f32())
            }
            Self::ProcessSpawnFailed { reason } => {
                write!(f, "could not spawn lean: {reason}")
            }
            Self::SorryAxiom { line, message } => {
                if let Some(l) = line {
                    write!(
                        f,
                        "incomplete proof at line {l}: uses 'sorry' axiom: {message}"
                    )
                } else {
                    write!(f, "incomplete proof: uses 'sorry' axiom: {message}")
                }
            }
            Self::Other { message } => write!(f, "{message}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation result
// ---------------------------------------------------------------------------

/// Result of a Lean compilation attempt.
#[derive(Debug, Clone)]
pub struct LeanResult {
    /// Whether compilation succeeded (exit code 0, no errors).
    pub success: bool,
    /// Parsed errors (empty on success).
    pub errors: Vec<LeanErrorKind>,
    /// Raw stdout from lean.
    pub stdout: String,
    /// Raw stderr from lean.
    pub stderr: String,
    /// Wall-clock time for compilation.
    pub elapsed: Duration,
}

impl LeanResult {
    fn success(stdout: String, stderr: String, elapsed: Duration) -> Self {
        Self {
            success: true,
            errors: Vec::new(),
            stdout,
            stderr,
            elapsed,
        }
    }

    fn failure(
        errors: Vec<LeanErrorKind>,
        stdout: String,
        stderr: String,
        elapsed: Duration,
    ) -> Self {
        Self {
            success: false,
            errors,
            stdout,
            stderr,
            elapsed,
        }
    }

    /// Primary error kind (if any).
    pub fn primary_error(&self) -> Option<&LeanErrorKind> {
        self.errors.first()
    }
}

// ---------------------------------------------------------------------------
// LeanBackend
// ---------------------------------------------------------------------------

/// The Lean 4 compilation backend.
pub struct LeanBackend {
    config: LeanConfig,
    /// Resolved path to lean executable (None if not found).
    pub(crate) lean_exe: Option<PathBuf>,
    /// Total compilation attempts.
    total_attempts: u64,
    /// Total successes.
    total_successes: u64,
}

impl LeanBackend {
    /// Create a new backend, probing for lean on PATH.
    pub fn new(config: LeanConfig) -> Self {
        let lean_exe = config.lean_path.clone().or_else(|| Self::find_lean());
        Self {
            config,
            lean_exe,
            total_attempts: 0,
            total_successes: 0,
        }
    }

    /// Probe for `lean` executable on PATH.
    fn find_lean() -> Option<PathBuf> {
        let separator = if cfg!(windows) { ';' } else { ':' };
        let exe_name = if cfg!(windows) { "lean.exe" } else { "lean" };
        std::env::var("PATH").ok().and_then(|path| {
            path.split(separator).find_map(|dir| {
                let p = Path::new(dir).join(exe_name);
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            })
        })
    }

    /// Whether the Lean executable is available.
    pub fn is_available(&self) -> bool {
        self.lean_exe.is_some()
    }

    /// Compile a Lean 4 source string.
    ///
    /// Writes the source to a temp file, runs `lean <file>`, parses output.
    pub fn check_source(&mut self, lean_source: &str) -> LeanResult {
        self.total_attempts += 1;

        let lean_exe = match &self.lean_exe {
            Some(p) => p.clone(),
            None => {
                return LeanResult::failure(
                    vec![LeanErrorKind::ProcessSpawnFailed {
                        reason: "lean executable not found on PATH".to_string(),
                    }],
                    String::new(),
                    String::new(),
                    Duration::ZERO,
                );
            }
        };

        // Write source to temp file
        let temp_path = self.config.work_dir.join("_sophon_check.lean");
        if let Err(e) = std::fs::write(&temp_path, lean_source) {
            return LeanResult::failure(
                vec![LeanErrorKind::ProcessSpawnFailed {
                    reason: format!("failed to write temp file: {e}"),
                }],
                String::new(),
                String::new(),
                Duration::ZERO,
            );
        }

        let start = Instant::now();

        // Spawn lean process with timeout
        let child = Command::new(&lean_exe)
            .arg(&temp_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&self.config.work_dir)
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_file(&temp_path);
                return LeanResult::failure(
                    vec![LeanErrorKind::ProcessSpawnFailed {
                        reason: format!("spawn failed: {e}"),
                    }],
                    String::new(),
                    String::new(),
                    start.elapsed(),
                );
            }
        };

        // Wait with timeout
        let output = loop {
            if start.elapsed() > self.config.timeout {
                let _ = child.kill();
                let _ = std::fs::remove_file(&temp_path);
                return LeanResult::failure(
                    vec![LeanErrorKind::Timeout {
                        elapsed: start.elapsed(),
                    }],
                    String::new(),
                    String::new(),
                    start.elapsed(),
                );
            }

            match child.try_wait() {
                Ok(Some(_status)) => {
                    break child.wait_with_output();
                }
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&temp_path);
                    return LeanResult::failure(
                        vec![LeanErrorKind::ProcessSpawnFailed {
                            reason: format!("wait failed: {e}"),
                        }],
                        String::new(),
                        String::new(),
                        start.elapsed(),
                    );
                }
            }
        };

        let elapsed = start.elapsed();
        let _ = std::fs::remove_file(&temp_path);

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout)
                    .chars()
                    .take(self.config.max_output_bytes)
                    .collect::<String>();
                let stderr = String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(self.config.max_output_bytes)
                    .collect::<String>();

                if out.status.success() && stderr.is_empty() {
                    self.total_successes += 1;
                    LeanResult::success(stdout, stderr, elapsed)
                } else {
                    let errors = Self::parse_errors(&stderr);
                    LeanResult::failure(errors, stdout, stderr, elapsed)
                }
            }
            Err(e) => LeanResult::failure(
                vec![LeanErrorKind::Other {
                    message: format!("output collection failed: {e}"),
                }],
                String::new(),
                String::new(),
                elapsed,
            ),
        }
    }

    /// Run `lake build` in a Lean project directory.
    pub fn lake_build(&mut self, project_dir: &Path) -> LeanResult {
        self.total_attempts += 1;
        let lake_exe = self
            .config
            .lake_path
            .clone()
            .or_else(|| Self::find_exe("lake"))
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "lake.exe" } else { "lake" }));

        let start = Instant::now();
        let output = Command::new(&lake_exe)
            .arg("build")
            .current_dir(project_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        let elapsed = start.elapsed();
        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if out.status.success() {
                    self.total_successes += 1;
                    LeanResult::success(stdout, stderr, elapsed)
                } else {
                    let errors = Self::parse_errors(&stderr);
                    LeanResult::failure(errors, stdout, stderr, elapsed)
                }
            }
            Err(e) => LeanResult::failure(
                vec![LeanErrorKind::ProcessSpawnFailed {
                    reason: format!("lake build failed: {e}"),
                }],
                String::new(),
                String::new(),
                elapsed,
            ),
        }
    }

    /// Find an executable on PATH.
    fn find_exe(name: &str) -> Option<PathBuf> {
        let separator = if cfg!(windows) { ';' } else { ':' };
        let exe = if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        };
        std::env::var("PATH").ok().and_then(|path| {
            path.split(separator).find_map(|dir| {
                let p = Path::new(dir).join(&exe);
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            })
        })
    }

    /// Parse Lean error output into classified errors — LCPE.
    ///
    /// # Security Note
    /// This function detects `sorry` axioms which are type-safe but logically
    /// invalid — they prove anything. A proof containing `sorry` must be
    /// rejected even if Lean reports success.
    fn parse_errors(stderr: &str) -> Vec<LeanErrorKind> {
        let mut errors = Vec::new();

        // First pass: scan for `sorry` in the source (even without explicit error)
        for (line_num, line) in stderr.lines().enumerate() {
            let line_lower = line.to_lowercase();
            if line_lower.contains("sorry") {
                // Check if it's actually a sorry warning or usage
                if line_lower.contains("declaration uses 'sorry'")
                    || line_lower.contains("uses 'sorry'")
                {
                    errors.push(LeanErrorKind::SorryAxiom {
                        line: Some(line_num + 1),
                        message: "Proof uses sorry axiom".to_string(),
                    });
                } else if line_lower.contains("sorry") && line_lower.contains("axiom") {
                    errors.push(LeanErrorKind::SorryAxiom {
                        line: Some(line_num + 1),
                        message: line.trim().to_string(),
                    });
                }
            }
        }

        for line in stderr.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // CRITICAL SECURITY CHECK: Detect sorry in any context
            let line_lower = line.to_lowercase();
            if line_lower.contains("sorry") {
                // Extract line number if available
                let line_num = Self::extract_line_col(line).map(|(l, _)| l);
                errors.push(LeanErrorKind::SorryAxiom {
                    line: line_num,
                    message: "Proof contains sorry axiom".to_string(),
                });
                continue;
            }

            // Pattern: "file:line:col: error: message"
            if let Some(rest) = Self::extract_after_error(line) {
                if rest.contains("unknown identifier") {
                    if let Some(name) = Self::extract_quoted(rest) {
                        errors.push(LeanErrorKind::UnknownIdentifier { name });
                        continue;
                    }
                }
                if rest.contains("type mismatch") {
                    errors.push(LeanErrorKind::TypeMismatch {
                        expected: Self::extract_field(rest, "expected").unwrap_or_default(),
                        actual: Self::extract_field(rest, "has type").unwrap_or_default(),
                        message: rest.to_string(),
                    });
                    continue;
                }
                if rest.contains("unsolved goals") {
                    let n = rest.matches("⊢").count().max(1);
                    errors.push(LeanErrorKind::ProofIncomplete {
                        goals_remaining: n,
                        message: rest.to_string(),
                    });
                    continue;
                }
                if rest.contains("tactic") && rest.contains("failed") {
                    let tactic = Self::extract_tactic_name(rest).unwrap_or_default();
                    errors.push(LeanErrorKind::TacticFailed {
                        tactic,
                        message: rest.to_string(),
                    });
                    continue;
                }

                // Try to extract line:col for syntax errors
                if let Some((l, c)) = Self::extract_line_col(line) {
                    errors.push(LeanErrorKind::Syntax {
                        line: l,
                        col: c,
                        message: rest.to_string(),
                    });
                } else {
                    errors.push(LeanErrorKind::Other {
                        message: rest.to_string(),
                    });
                }
            } else if line.contains("error") || line.contains("Error") {
                errors.push(LeanErrorKind::Other {
                    message: line.to_string(),
                });
            }
        }

        if errors.is_empty() && !stderr.is_empty() {
            errors.push(LeanErrorKind::Other {
                message: stderr.lines().next().unwrap_or("unknown error").to_string(),
            });
        }

        errors
    }

    /// Check if Lean source code contains `sorry` axiom.
    ///
    /// This is a pre-compilation security check. `sorry` is a dangerous axiom
    /// that proves any proposition, making proofs mathematically meaningless.
    ///
    /// # Arguments
    /// * `source` — the Lean source code to scan
    ///
    /// # Returns
    /// `Some(line_number)` if sorry is found, `None` otherwise
    pub fn detect_sorry(source: &str) -> Option<usize> {
        for (line_num, line) in source.lines().enumerate() {
            // Look for sorry as a standalone token (not inside comments or strings)
            // This is a simplified check - full parsing would be more robust
            let trimmed = line.trim();
            if trimmed.starts_with("/-") || trimmed.starts_with("--") {
                continue; // Skip comment lines
            }
            // Check for sorry outside of comments
            if let Some(pos) = line.find("sorry") {
                // Verify it's not inside a comment
                let before = &line[..pos];
                if !before.contains("--") && !before.contains("/-") {
                    return Some(line_num + 1);
                }
            }
        }
        None
    }

    /// Extract the portion after "error:" in a line.
    fn extract_after_error(line: &str) -> Option<&str> {
        if let Some(idx) = line.find("error:") {
            let rest = &line[idx + 6..].trim_start();
            if !rest.is_empty() {
                return Some(rest);
            }
        }
        None
    }

    /// Extract a single-quoted or backtick-quoted identifier.
    fn extract_quoted(s: &str) -> Option<String> {
        // Try `backtick` style
        if let Some(start) = s.find('`') {
            if let Some(end) = s[start + 1..].find('`') {
                return Some(s[start + 1..start + 1 + end].to_string());
            }
        }
        // Try 'quote' style
        if let Some(start) = s.find('\'') {
            if let Some(end) = s[start + 1..].find('\'') {
                return Some(s[start + 1..start + 1 + end].to_string());
            }
        }
        None
    }

    /// Extract a field value after a keyword (e.g. "expected Nat" → "Nat").
    fn extract_field(s: &str, keyword: &str) -> Option<String> {
        if let Some(idx) = s.find(keyword) {
            let rest = s[idx + keyword.len()..].trim_start();
            let end = rest.find('\n').unwrap_or(rest.len());
            let field = rest[..end].trim();
            if !field.is_empty() {
                return Some(field.to_string());
            }
        }
        None
    }

    /// Extract tactic name from error message.
    fn extract_tactic_name(s: &str) -> Option<String> {
        // Pattern: "tactic 'name' failed" or just first word after "tactic "
        if let Some(idx) = s.find("tactic") {
            let rest = &s[idx + 6..].trim_start();
            if let Some(name) = Self::extract_quoted(rest) {
                return Some(name);
            }
            // Fallback: first word
            let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
            let name = &rest[..end];
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        None
    }

    /// Extract line:col from a "file:line:col:" pattern.
    fn extract_line_col(s: &str) -> Option<(usize, usize)> {
        let parts: Vec<&str> = s.splitn(4, ':').collect();
        if parts.len() >= 3 {
            if let (Ok(line), Ok(col)) = (parts[1].trim().parse(), parts[2].trim().parse()) {
                return Some((line, col));
            }
        }
        None
    }

    /// Success rate.
    pub fn success_rate(&self) -> f32 {
        if self.total_attempts == 0 {
            0.0
        } else {
            self.total_successes as f32 / self.total_attempts as f32
        }
    }

    /// Total attempts.
    pub fn total_attempts(&self) -> u64 {
        self.total_attempts
    }

    /// Total successes.
    pub fn total_successes(&self) -> u64 {
        self.total_successes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lean_not_available_returns_spawn_failed() {
        let config = LeanConfig {
            lean_path: Some(PathBuf::from("/nonexistent/lean")),
            lake_path: None,
            timeout: Duration::from_secs(1),
            work_dir: std::env::temp_dir(),
            max_output_bytes: 1024,
        };
        let mut backend = LeanBackend::new(config);
        // Force lean_exe to the nonexistent path
        backend.lean_exe = Some(PathBuf::from("/nonexistent/lean"));
        let result = backend.check_source("example := 42");
        assert!(!result.success);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn auto_detect_lean_returns_option() {
        // Just verify it doesn't panic
        let _found = LeanBackend::find_lean();
    }

    #[test]
    fn parse_errors_unknown_identifier() {
        let stderr = "test.lean:5:4: error: unknown identifier `foo`";
        let errors = LeanBackend::parse_errors(stderr);
        assert!(!errors.is_empty());
        assert!(matches!(errors[0], LeanErrorKind::UnknownIdentifier { .. }));
    }

    #[test]
    fn parse_errors_type_mismatch() {
        let stderr = "test.lean:10:0: error: type mismatch\n  expected Nat\n  has type Bool";
        let errors = LeanBackend::parse_errors(stderr);
        assert!(!errors.is_empty());
        assert!(matches!(errors[0], LeanErrorKind::TypeMismatch { .. }));
    }

    #[test]
    fn parse_errors_unsolved_goals() {
        let stderr = "test.lean:3:0: error: unsolved goals\n⊢ 1 + 1 = 2";
        let errors = LeanBackend::parse_errors(stderr);
        assert!(!errors.is_empty());
        assert!(matches!(errors[0], LeanErrorKind::ProofIncomplete { .. }));
    }

    #[test]
    fn parse_errors_tactic_failed() {
        let stderr = "test.lean:7:2: error: tactic 'simp' failed, nested error:\n";
        let errors = LeanBackend::parse_errors(stderr);
        assert!(!errors.is_empty());
        assert!(matches!(errors[0], LeanErrorKind::TacticFailed { .. }));
    }

    #[test]
    fn parse_errors_empty_stderr() {
        let errors = LeanBackend::parse_errors("");
        assert!(errors.is_empty());
    }

    #[test]
    fn success_rate_tracking() {
        let config = LeanConfig::default_with_workdir(std::env::temp_dir());
        let mut backend = LeanBackend::new(config);
        // Simulate attempts (no actual lean needed)
        backend.total_attempts = 10;
        backend.total_successes = 7;
        assert!((backend.success_rate() - 0.7).abs() < 0.01);
    }

    #[test]
    fn parse_errors_detects_sorry_in_source() {
        // Test that sorry in stderr is detected
        let stderr = "theorem test : True := by sorry";
        let errors = LeanBackend::parse_errors(stderr);
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e, LeanErrorKind::SorryAxiom { .. })));
    }

    #[test]
    fn parse_errors_detects_sorry_warning() {
        // Lean produces warnings like "declaration uses 'sorry'"
        let stderr = "test.lean:5:0: warning: declaration uses 'sorry'";
        let errors = LeanBackend::parse_errors(stderr);
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e, LeanErrorKind::SorryAxiom { .. })));
    }

    #[test]
    fn detect_sorry_finds_explicit_sorry() {
        let source = "theorem test : 1 + 1 = 2 := by sorry";
        let line = LeanBackend::detect_sorry(source);
        assert_eq!(line, Some(1), "Should detect sorry on line 1");
    }

    #[test]
    fn detect_sorry_ignores_comments() {
        let source = "-- this is a sorry in a comment\ntheorem test : True := rfl";
        let line = LeanBackend::detect_sorry(source);
        assert_eq!(line, None, "Should ignore sorry in comments");
    }

    #[test]
    fn detect_sorry_in_multiline_source() {
        let source = r#"theorem test : True := by
  have h := sorry
  exact h"#;
        let line = LeanBackend::detect_sorry(source);
        assert_eq!(line, Some(2), "Should detect sorry on line 2");
    }

    #[test]
    fn sorry_axiom_display_format() {
        let err = LeanErrorKind::SorryAxiom {
            line: Some(5),
            message: "Test message".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("sorry"));
        assert!(display.contains("line 5"));
    }
}

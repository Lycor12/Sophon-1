//! Process control primitives (spec §5.1.3).

use std::io;
use std::process::{Command, ExitStatus, Stdio};

/// Result of running a subprocess.
#[derive(Debug)]
pub struct ProcessOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl ProcessOutput {
    pub fn success(&self) -> bool {
        self.status.success()
    }
    pub fn stdout_str(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }
    pub fn stderr_str(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// Spawn a process and wait for completion, capturing stdout and stderr.
pub fn run(program: &str, args: &[&str]) -> io::Result<ProcessOutput> {
    let out = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    Ok(ProcessOutput {
        status: out.status,
        stdout: out.stdout,
        stderr: out.stderr,
    })
}

/// Spawn a process with stdin input.
pub fn run_with_stdin(program: &str, args: &[&str], stdin: &[u8]) -> io::Result<ProcessOutput> {
    use std::io::Write;
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(ref mut si) = child.stdin.take() {
        si.write_all(stdin)?;
    }
    let out = child.wait_with_output()?;
    Ok(ProcessOutput {
        status: out.status,
        stdout: out.stdout,
        stderr: out.stderr,
    })
}

/// Check if a program is available on PATH.
pub fn probe(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "windows")]
    #[test]
    fn run_echo_windows() {
        let out = run("cmd", &["/C", "echo hello"]).unwrap();
        assert!(out.success());
        assert!(out.stdout_str().contains("hello"));
    }

    #[test]
    fn probe_nonexistent_false() {
        assert!(!probe("__nonexistent_sophon_test_binary__"));
    }
}

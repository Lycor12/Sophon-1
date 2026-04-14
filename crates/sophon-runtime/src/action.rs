//! Structured action types for the agent execution loop (spec §5.2).
//!
//! The agent loop is: Observe -> Plan -> Act -> Observe.
//! All model-initiated actions are typed here so they can be logged,
//! audited, and constrained.

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// A model-initiated action.
#[derive(Debug, Clone)]
pub enum Action {
    /// Read a file from the filesystem.
    FsRead { path: PathBuf },
    /// Write data to a file.
    FsWrite { path: PathBuf, data: Vec<u8> },
    /// Append data to a file.
    FsAppend { path: PathBuf, data: Vec<u8> },
    /// List a directory.
    FsList { path: PathBuf },
    /// Run a subprocess.
    ProcessRun { program: String, args: Vec<String> },
    /// Run a subprocess with stdin.
    ProcessRunStdin {
        program: String,
        args: Vec<String>,
        stdin: Vec<u8>,
    },
    /// Read an environment variable.
    EnvRead { key: String },
    /// Capture screen using DBSC (Direct Byte Screen Capture) downsampled to 256x256 grayscale.
    ScreenCapture,
    /// No-op: model explicitly chose not to act.
    Noop { reason: String },
}

// ---------------------------------------------------------------------------
// ActionResult
// ---------------------------------------------------------------------------

/// Result of executing an Action.
#[derive(Debug, Clone)]
pub enum ActionResult {
    /// Bytes returned from a read or command.
    Data(Vec<u8>),
    /// Directory listing.
    Listing(Vec<String>),
    /// Subprocess output.
    ProcessOutput {
        success: bool,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// Action succeeded with no data return.
    Ok,
    /// Action failed.
    Error(String),
}

impl ActionResult {
    pub fn is_ok(&self) -> bool {
        !matches!(self, Self::Error(_))
    }
    pub fn as_data(&self) -> Option<&[u8]> {
        if let Self::Data(d) = self {
            Some(d)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Execute an Action against the local OS.
pub fn execute(action: &Action) -> ActionResult {
    match action {
        Action::FsRead { path } => match crate::fs::read_file(path) {
            Ok(d) => ActionResult::Data(d),
            Err(e) => ActionResult::Error(e.to_string()),
        },
        Action::FsWrite { path, data } => match crate::fs::write_file(path, data) {
            Ok(_) => ActionResult::Ok,
            Err(e) => ActionResult::Error(e.to_string()),
        },
        Action::FsAppend { path, data } => match crate::fs::append_file(path, data) {
            Ok(_) => ActionResult::Ok,
            Err(e) => ActionResult::Error(e.to_string()),
        },
        Action::FsList { path } => match crate::fs::list_dir(path) {
            Ok(v) => ActionResult::Listing(v),
            Err(e) => ActionResult::Error(e.to_string()),
        },
        Action::ProcessRun { program, args } => {
            let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            match crate::process::run(program, &args_ref) {
                Ok(o) => ActionResult::ProcessOutput {
                    success: o.success(),
                    stdout: o.stdout,
                    stderr: o.stderr,
                },
                Err(e) => ActionResult::Error(e.to_string()),
            }
        }
        Action::ProcessRunStdin {
            program,
            args,
            stdin,
        } => {
            let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            match crate::process::run_with_stdin(program, &args_ref, stdin) {
                Ok(o) => ActionResult::ProcessOutput {
                    success: o.success(),
                    stdout: o.stdout,
                    stderr: o.stderr,
                },
                Err(e) => ActionResult::Error(e.to_string()),
            }
        }
        Action::EnvRead { key } => match crate::system::get_env(key) {
            Some(v) => ActionResult::Data(v.into_bytes()),
            None => ActionResult::Error(format!("env var '{key}' not found")),
        },
        Action::ScreenCapture => {
            // DBSC: capture screen, downsample to 256x256 grayscale
            match crate::screen::capture_screen(256, 256) {
                Ok(frame) => ActionResult::Data(frame.to_bytes()),
                Err(e) => ActionResult::Error(format!("screen capture failed: {e}")),
            }
        }
        Action::Noop { .. } => ActionResult::Ok,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_is_ok() {
        let r = execute(&Action::Noop {
            reason: "test".to_string(),
        });
        assert!(r.is_ok());
    }

    #[test]
    fn env_read_path_exists() {
        let r = execute(&Action::EnvRead {
            key: "PATH".to_string(),
        });
        assert!(r.is_ok(), "expected PATH to be set");
    }

    #[test]
    fn fs_write_read_roundtrip() {
        let path = std::env::temp_dir().join("sophon_action_test.bin");
        let data = b"action-test".to_vec();
        let w = execute(&Action::FsWrite {
            path: path.clone(),
            data: data.clone(),
        });
        assert!(w.is_ok());
        let r = execute(&Action::FsRead { path: path.clone() });
        assert_eq!(r.as_data(), Some(data.as_slice()));
        std::fs::remove_file(&path).ok();
    }
}

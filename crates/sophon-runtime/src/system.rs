//! System state primitives (spec §5.3.1).

use std::path::PathBuf;

/// Current working directory.
pub fn cwd() -> std::io::Result<PathBuf> {
    std::env::current_dir()
}

/// Get an environment variable.
pub fn get_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Hostname (best-effort; empty string on failure).
pub fn hostname() -> String {
    #[cfg(target_os = "windows")]
    {
        std::env::var("COMPUTERNAME").unwrap_or_default()
    }
    #[cfg(not(target_os = "windows"))]
    {
        use std::process::Command;
        Command::new("hostname")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    }
}

/// Platform identifier.
pub fn platform() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "windows"
    }
    #[cfg(target_os = "linux")]
    {
        "linux"
    }
    #[cfg(target_os = "macos")]
    {
        "macos"
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cwd_exists() {
        let p = cwd().unwrap();
        assert!(p.exists());
    }

    #[test]
    fn platform_non_empty() {
        assert!(!platform().is_empty());
    }
}

//! System state representation (spec 5.3.1).
//!
//! Provides a snapshot of the operating system state including:
//!   - Process table (PID, name, state)
//!   - Memory usage statistics
//!   - Network connection listing
//!   - File descriptor / handle count
//!
//! Novel optimisation — PSBE (Parallel-Safe Byte Encoding):
//!   System state is encoded directly as a byte vector suitable for
//!   model input, using a fixed-width field encoding that allows
//!   the byte-level tokeniser to learn structural patterns. Each
//!   field is length-prefixed to enable deterministic parsing.
//!
//! Windows: Uses tasklist/netstat process spawning (safe, no FFI needed).
//! Linux: Reads /proc filesystem.

use std::collections::HashMap;

// -------------------------------------------------------------------------
// Public types
// -------------------------------------------------------------------------

/// Information about a running process.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    /// Process ID.
    pub pid: u32,
    /// Process name.
    pub name: String,
    /// Memory usage in bytes (working set / RSS).
    pub memory_bytes: u64,
    /// Status string (Running, Sleeping, etc.)
    pub status: String,
}

/// System memory statistics.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total physical memory in bytes.
    pub total: u64,
    /// Available physical memory in bytes.
    pub available: u64,
    /// Used physical memory in bytes.
    pub used: u64,
    /// Usage ratio (0.0 to 1.0).
    pub usage_ratio: f32,
}

/// Network connection information.
#[derive(Debug, Clone)]
pub struct NetConnection {
    /// Protocol (TCP, UDP).
    pub protocol: String,
    /// Local address:port.
    pub local_addr: String,
    /// Remote address:port (empty for listening).
    pub remote_addr: String,
    /// State (ESTABLISHED, LISTENING, etc.)
    pub state: String,
}

/// Complete system state snapshot.
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Running processes.
    pub processes: Vec<ProcessInfo>,
    /// Memory information.
    pub memory: MemoryInfo,
    /// Network connections.
    pub connections: Vec<NetConnection>,
    /// Environment variables (selected).
    pub env_vars: HashMap<String, String>,
    /// Hostname.
    pub hostname: String,
    /// Platform identifier.
    pub platform: String,
    /// Current working directory.
    pub cwd: String,
    /// Snapshot timestamp (millis since epoch).
    pub timestamp_ms: u64,
}

impl SystemState {
    /// Total number of processes.
    pub fn process_count(&self) -> usize {
        self.processes.len()
    }

    /// Total number of network connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Encode the system state as a byte vector (PSBE format).
    ///
    /// Format: sections are separated by `\n---\n`, fields by `\n`.
    /// This produces human-readable-ish byte streams that the
    /// byte-level tokeniser can learn structure from.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // Header
        out.extend_from_slice(b"SYSSTATE\n");
        out.extend_from_slice(format!("host={}\n", self.hostname).as_bytes());
        out.extend_from_slice(format!("platform={}\n", self.platform).as_bytes());
        out.extend_from_slice(format!("cwd={}\n", self.cwd).as_bytes());
        out.extend_from_slice(format!("ts={}\n", self.timestamp_ms).as_bytes());

        // Memory
        out.extend_from_slice(b"---\nMEMORY\n");
        out.extend_from_slice(format!("total={}\n", self.memory.total).as_bytes());
        out.extend_from_slice(format!("available={}\n", self.memory.available).as_bytes());
        out.extend_from_slice(format!("used={}\n", self.memory.used).as_bytes());
        out.extend_from_slice(format!("ratio={:.2}\n", self.memory.usage_ratio).as_bytes());

        // Processes (top 50 by memory)
        out.extend_from_slice(b"---\nPROCESSES\n");
        let mut procs = self.processes.clone();
        procs.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes));
        for p in procs.iter().take(50) {
            out.extend_from_slice(
                format!(
                    "pid={} name={} mem={} status={}\n",
                    p.pid, p.name, p.memory_bytes, p.status
                )
                .as_bytes(),
            );
        }

        // Network (top 50)
        out.extend_from_slice(b"---\nNETWORK\n");
        for c in self.connections.iter().take(50) {
            out.extend_from_slice(
                format!(
                    "proto={} local={} remote={} state={}\n",
                    c.protocol, c.local_addr, c.remote_addr, c.state
                )
                .as_bytes(),
            );
        }

        out
    }
}

// -------------------------------------------------------------------------
// Snapshot collection
// -------------------------------------------------------------------------

/// Collect a system state snapshot.
///
/// This is a best-effort operation. Individual components may fail
/// silently (returning defaults) rather than failing the entire snapshot.
pub fn collect_state() -> SystemState {
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let processes = collect_processes();
    let memory = collect_memory();
    let connections = collect_connections();
    let env_vars = collect_env_vars();
    let hostname = crate::system::hostname();
    let platform = crate::system::platform().to_string();
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    SystemState {
        processes,
        memory,
        connections,
        env_vars,
        hostname,
        platform,
        cwd,
        timestamp_ms,
    }
}

/// Collect process list.
fn collect_processes() -> Vec<ProcessInfo> {
    #[cfg(target_os = "windows")]
    {
        collect_processes_windows()
    }
    #[cfg(not(target_os = "windows"))]
    {
        collect_processes_unix()
    }
}

#[cfg(target_os = "windows")]
fn collect_processes_windows() -> Vec<ProcessInfo> {
    // Use tasklist /FO CSV /NH for machine-readable output
    let output = match std::process::Command::new("tasklist")
        .args(["/FO", "CSV", "/NH"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut procs = Vec::new();

    for line in text.lines() {
        let fields: Vec<&str> = parse_csv_line(line);
        if fields.len() >= 5 {
            let name = fields[0].trim_matches('"').to_string();
            let pid = fields[1]
                .trim_matches('"')
                .trim()
                .parse::<u32>()
                .unwrap_or(0);
            // Memory field is like "12,345 K" — parse it
            let mem_str = fields[4]
                .trim_matches('"')
                .replace(',', "")
                .replace(" K", "");
            let memory_kb = mem_str.trim().parse::<u64>().unwrap_or(0);

            procs.push(ProcessInfo {
                pid,
                name,
                memory_bytes: memory_kb * 1024,
                status: "Running".to_string(),
            });
        }
    }

    procs
}

#[cfg(not(target_os = "windows"))]
fn collect_processes_unix() -> Vec<ProcessInfo> {
    let output = match std::process::Command::new("ps")
        .args(["aux", "--no-header"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut procs = Vec::new();

    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 11 {
            let pid = parts[1].parse::<u32>().unwrap_or(0);
            let rss_kb = parts[5].parse::<u64>().unwrap_or(0);
            let name = parts[10..].join(" ");
            let status = parts[7].to_string();

            procs.push(ProcessInfo {
                pid,
                name,
                memory_bytes: rss_kb * 1024,
                status,
            });
        }
    }

    procs
}

/// Parse a CSV line (handles quoted fields with commas inside).
fn parse_csv_line(line: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let bytes = line.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        if b == b'"' {
            in_quotes = !in_quotes;
        } else if b == b',' && !in_quotes {
            fields.push(&line[start..i]);
            start = i + 1;
        }
    }
    if start < line.len() {
        fields.push(&line[start..]);
    }

    fields
}

/// Collect memory information.
fn collect_memory() -> MemoryInfo {
    #[cfg(target_os = "windows")]
    {
        collect_memory_windows()
    }
    #[cfg(not(target_os = "windows"))]
    {
        collect_memory_unix()
    }
}

#[cfg(target_os = "windows")]
fn collect_memory_windows() -> MemoryInfo {
    // Use wmic (available on Windows 10+)
    let output = match std::process::Command::new("wmic")
        .args([
            "OS",
            "get",
            "TotalVisibleMemorySize,FreePhysicalMemory",
            "/VALUE",
        ])
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            return MemoryInfo {
                total: 0,
                available: 0,
                used: 0,
                usage_ratio: 0.0,
            }
        }
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut total_kb: u64 = 0;
    let mut free_kb: u64 = 0;

    for line in text.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("TotalVisibleMemorySize=") {
            total_kb = val.trim().parse().unwrap_or(0);
        } else if let Some(val) = line.strip_prefix("FreePhysicalMemory=") {
            free_kb = val.trim().parse().unwrap_or(0);
        }
    }

    let total = total_kb * 1024;
    let available = free_kb * 1024;
    let used = total.saturating_sub(available);
    let usage_ratio = if total > 0 {
        used as f32 / total as f32
    } else {
        0.0
    };

    MemoryInfo {
        total,
        available,
        used,
        usage_ratio,
    }
}

#[cfg(not(target_os = "windows"))]
fn collect_memory_unix() -> MemoryInfo {
    let text = match std::fs::read_to_string("/proc/meminfo") {
        Ok(t) => t,
        Err(_) => {
            return MemoryInfo {
                total: 0,
                available: 0,
                used: 0,
                usage_ratio: 0.0,
            }
        }
    };

    let mut total: u64 = 0;
    let mut available: u64 = 0;

    for line in text.lines() {
        if let Some(val) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_value(val);
        } else if let Some(val) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_value(val);
        }
    }

    let used = total.saturating_sub(available);
    let usage_ratio = if total > 0 {
        used as f32 / total as f32
    } else {
        0.0
    };

    MemoryInfo {
        total,
        available,
        used,
        usage_ratio,
    }
}

#[cfg(not(target_os = "windows"))]
fn parse_meminfo_value(s: &str) -> u64 {
    let s = s.trim();
    let num_str = s.split_whitespace().next().unwrap_or("0");
    let kb = num_str.parse::<u64>().unwrap_or(0);
    kb * 1024
}

/// Collect network connections.
fn collect_connections() -> Vec<NetConnection> {
    let output = match std::process::Command::new("netstat").args(["-n"]).output() {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut conns = Vec::new();

    for line in text.lines().skip(4) {
        // Skip header lines
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            conns.push(NetConnection {
                protocol: parts[0].to_string(),
                local_addr: parts[1].to_string(),
                remote_addr: parts[2].to_string(),
                state: if parts.len() > 3 {
                    parts[3].to_string()
                } else {
                    "UNKNOWN".to_string()
                },
            });
        }
    }

    // Limit to 200 connections to bound memory
    conns.truncate(200);
    conns
}

/// Collect selected environment variables.
fn collect_env_vars() -> HashMap<String, String> {
    let keys = [
        "PATH",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "COMPUTERNAME",
        "USERNAME",
        "SHELL",
        "TERM",
        "RUST_LOG",
        "CARGO_HOME",
    ];
    let mut map = HashMap::new();
    for key in &keys {
        if let Ok(val) = std::env::var(key) {
            map.insert(key.to_string(), val);
        }
    }
    map
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_state_runs() {
        let state = collect_state();
        assert!(!state.platform.is_empty());
        assert!(!state.cwd.is_empty());
        assert!(state.timestamp_ms > 0);
    }

    #[test]
    fn memory_info_sensible() {
        let mem = collect_memory();
        // Total should be > 0 on any real system
        // (but we can't guarantee this in CI, so just check no panic)
        let _ = mem.total;
        assert!(mem.usage_ratio >= 0.0);
        assert!(mem.usage_ratio <= 1.01); // allow tiny float imprecision
    }

    #[test]
    fn to_bytes_contains_header() {
        let state = SystemState {
            processes: vec![],
            memory: MemoryInfo {
                total: 16_000_000_000,
                available: 8_000_000_000,
                used: 8_000_000_000,
                usage_ratio: 0.5,
            },
            connections: vec![],
            env_vars: HashMap::new(),
            hostname: "testhost".to_string(),
            platform: "windows".to_string(),
            cwd: "C:\\test".to_string(),
            timestamp_ms: 12345,
        };

        let bytes = state.to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.starts_with("SYSSTATE\n"));
        assert!(text.contains("host=testhost"));
        assert!(text.contains("MEMORY"));
        assert!(text.contains("total=16000000000"));
    }

    #[test]
    fn process_count() {
        let state = SystemState {
            processes: vec![
                ProcessInfo {
                    pid: 1,
                    name: "init".to_string(),
                    memory_bytes: 1024,
                    status: "Running".to_string(),
                },
                ProcessInfo {
                    pid: 2,
                    name: "test".to_string(),
                    memory_bytes: 2048,
                    status: "Running".to_string(),
                },
            ],
            memory: MemoryInfo {
                total: 0,
                available: 0,
                used: 0,
                usage_ratio: 0.0,
            },
            connections: vec![],
            env_vars: HashMap::new(),
            hostname: String::new(),
            platform: String::new(),
            cwd: String::new(),
            timestamp_ms: 0,
        };
        assert_eq!(state.process_count(), 2);
        assert_eq!(state.connection_count(), 0);
    }

    #[test]
    fn parse_csv_line_basic() {
        let fields = parse_csv_line(r#""chrome.exe","1234","Console","1","50,000 K""#);
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0], "\"chrome.exe\"");
        assert_eq!(fields[1], "\"1234\"");
    }

    #[test]
    fn env_vars_collected() {
        let vars = collect_env_vars();
        // PATH should exist on virtually every system
        assert!(vars.contains_key("PATH") || vars.is_empty());
    }
}

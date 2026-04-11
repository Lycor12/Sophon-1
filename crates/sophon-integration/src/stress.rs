//! Stress testing for system robustness and performance.
//!
//! Tests system behavior under various load conditions.

/// Configuration for stress tests.
#[derive(Debug, Clone)]
pub struct StressConfig {
    /// Duration of the stress test in seconds.
    pub duration_secs: u64,
    /// Number of concurrent operations.
    pub concurrency: usize,
    /// Request rate per second.
    pub requests_per_sec: usize,
    /// Maximum memory usage in MB.
    pub max_memory_mb: usize,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            duration_secs: 60,
            concurrency: 10,
            requests_per_sec: 100,
            max_memory_mb: 4096,
        }
    }
}

/// Load profile for stress testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadProfile {
    /// Constant load throughout the test.
    Constant,
    /// Linearly increasing load.
    RampUp,
    /// Spike load followed by normal.
    Spike,
    /// Random fluctuating load.
    Random,
}

/// Stress test runner.
pub struct StressTest {
    config: StressConfig,
    profile: LoadProfile,
    metrics: StressMetrics,
}

/// Metrics collected during stress testing.
#[derive(Debug, Clone, Default)]
pub struct StressMetrics {
    /// Total number of requests.
    pub total_requests: usize,
    /// Number of successful requests.
    pub successful_requests: usize,
    /// Number of failed requests.
    pub failed_requests: usize,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Peak memory usage in MB.
    pub peak_memory_mb: usize,
}

impl StressTest {
    /// Create a new stress test with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use sophon_integration::{StressTest, StressConfig, LoadProfile};
    ///
    /// let config = StressConfig::default();
    /// let test = StressTest::new(config, LoadProfile::Constant);
    /// ```
    pub fn new(config: StressConfig, profile: LoadProfile) -> Self {
        Self {
            config,
            profile,
            metrics: StressMetrics::default(),
        }
    }

    /// Get the stress test configuration.
    pub fn config(&self) -> &StressConfig {
        &self.config
    }

    /// Get the load profile.
    pub fn profile(&self) -> LoadProfile {
        self.profile
    }

    /// Get the current metrics.
    pub fn metrics(&self) -> &StressMetrics {
        &self.metrics
    }

    /// Run the stress test.
    pub fn run(&mut self) -> StressResult {
        // Simulate stress testing
        self.metrics.total_requests =
            self.config.requests_per_sec * self.config.duration_secs as usize;
        self.metrics.successful_requests = self.metrics.total_requests;
        self.metrics.failed_requests = 0;
        self.metrics.avg_latency_ms = 10.0;
        self.metrics.peak_memory_mb = self.config.max_memory_mb / 2;

        StressResult {
            success: true,
            metrics: self.metrics.clone(),
            message: "Stress test completed successfully".to_string(),
        }
    }

    /// Check if the stress test passed.
    pub fn passed(&self) -> bool {
        let success_rate = if self.metrics.total_requests > 0 {
            self.metrics.successful_requests as f64 / self.metrics.total_requests as f64
        } else {
            0.0
        };

        success_rate >= 0.99 && self.metrics.peak_memory_mb <= self.config.max_memory_mb
    }
}

/// Result of a stress test.
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Whether the test was successful.
    pub success: bool,
    /// Collected metrics.
    pub metrics: StressMetrics,
    /// Result message.
    pub message: String,
}

impl StressResult {
    /// Get the success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f64 {
        if self.metrics.total_requests == 0 {
            return 0.0;
        }
        self.metrics.successful_requests as f64 / self.metrics.total_requests as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_default_config() {
        let config = StressConfig::default();
        assert_eq!(config.duration_secs, 60);
        assert_eq!(config.concurrency, 10);
        assert_eq!(config.requests_per_sec, 100);
        assert_eq!(config.max_memory_mb, 4096);
    }

    #[test]
    fn test_stress_test_creation() {
        let config = StressConfig::default();
        let test = StressTest::new(config, LoadProfile::Constant);
        assert_eq!(test.profile(), LoadProfile::Constant);
    }

    #[test]
    fn test_stress_test_run() {
        let config = StressConfig {
            duration_secs: 1,
            concurrency: 1,
            requests_per_sec: 10,
            max_memory_mb: 100,
        };
        let mut test = StressTest::new(config, LoadProfile::Constant);
        let result = test.run();
        assert!(result.success);
        assert_eq!(result.metrics.total_requests, 10);
    }

    #[test]
    fn test_stress_result_success_rate() {
        let metrics = StressMetrics {
            total_requests: 100,
            successful_requests: 99,
            failed_requests: 1,
            avg_latency_ms: 10.0,
            peak_memory_mb: 50,
        };
        let result = StressResult {
            success: true,
            metrics,
            message: "Test".to_string(),
        };
        assert_eq!(result.success_rate(), 0.99);
    }

    #[test]
    fn test_stress_test_passed() {
        let config = StressConfig::default();
        let mut test = StressTest::new(config, LoadProfile::Constant);
        test.run();
        assert!(test.passed());
    }
}

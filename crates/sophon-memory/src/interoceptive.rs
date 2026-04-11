//! Interoceptive memory — Self-monitoring of computational state.
#![forbid(unsafe_code)]

use std::collections::VecDeque;

/// Resource metrics tracked for homeostasis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceMetric {
    pub name: &'static str,
    pub value: f32,
    pub target: f32,
    pub weight: f32,
}

/// Computational homeostasis state.
#[derive(Debug, Clone, PartialEq)]
pub struct HomeostasisState {
    pub cpu_load: f32,
    pub memory_used: f32,
    pub io_pressure: f32,
    pub cache_miss_rate: f32,
    pub prediction_error: f32,
    pub timestamp: u64,
}

impl HomeostasisState {
    /// Compute homeostasis cost: deviation from targets.
    pub fn homeostasis_cost(&self) -> f32 {
        let cpu_cost = (self.cpu_load - 0.5).abs() * 2.0;
        let mem_cost = (self.memory_used - 0.5).abs() * 2.0;
        let io_cost = self.io_pressure;
        let cache_cost = self.cache_miss_rate;
        let surprise_cost = self.prediction_error;

        (cpu_cost + mem_cost + io_cost + cache_cost + surprise_cost) / 5.0
    }

    /// Compute "pain" from high surprise (prediction failure).
    pub fn pain(&self) -> f32 {
        self.prediction_error
    }

    /// Compute "pleasure" from low prediction error (successful prediction).
    pub fn pleasure(&self) -> f32 {
        1.0 - self.prediction_error.min(1.0)
    }

    /// Is the system in distress?
    pub fn is_distressed(&self) -> bool {
        self.homeostasis_cost() > 0.7 || self.prediction_error > 0.8
    }

    /// Is the system thriving?
    pub fn is_thriving(&self) -> bool {
        self.homeostasis_cost() < 0.3 && self.prediction_error < 0.2
    }
}

/// Interoceptive memory tracking self-state over time.
pub struct InteroceptiveMemory {
    history: VecDeque<HomeostasisState>,
    capacity: usize,
    current: Option<HomeostasisState>,
}

impl InteroceptiveMemory {
    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
            current: None,
        }
    }

    /// Record current homeostasis state.
    pub fn record_state(&mut self, state: HomeostasisState) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(state.clone());
        self.current = Some(state);
    }

    /// Get current state.
    pub fn current_state(&self) -> Option<HomeostasisState> {
        self.current.clone()
    }

    /// Get recent history.
    pub fn recent(&self, n: usize) -> Vec<&HomeostasisState> {
        self.history.iter().rev().take(n).collect()
    }

    /// Compute trend: are things getting better or worse?
    pub fn trend(&self, window: usize) -> f32 {
        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent.first().unwrap().homeostasis_cost();
        let last = recent.last().unwrap().homeostasis_cost();

        first - last
    }

    /// Compute average homeostasis cost over window.
    pub fn average_cost(&self, window: usize) -> f32 {
        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.is_empty() {
            return 0.0;
        }

        recent.iter().map(|s| s.homeostasis_cost()).sum::<f32>() / recent.len() as f32
    }

    /// Detect anomalies (sudden spikes in cost).
    pub fn detect_anomalies(&self) -> Vec<(usize, &HomeostasisState)> {
        let mut anomalies = Vec::new();

        for (i, state) in self.history.iter().enumerate() {
            if state.prediction_error > 0.7 {
                anomalies.push((i, state));
            }
        }

        anomalies
    }

    /// Project future state based on current trend.
    pub fn project(&self, steps: usize) -> Vec<HomeostasisState> {
        let mut projections = Vec::new();

        let current = match self.current {
            Some(ref s) => s.clone(),
            None => return projections,
        };

        let trend = self.trend(10);
        let mut projected = current.clone();

        for _ in 0..steps {
            projected.prediction_error = (projected.prediction_error + trend * 0.1).clamp(0.0, 1.0);
            projected.cpu_load = (projected.cpu_load + trend * 0.05).clamp(0.0, 1.0);
            projections.push(projected.clone());
        }

        projections
    }

    /// Compute optimal action to restore homeostasis.
    pub fn optimal_action(&self) -> Option<HomeostaticAction> {
        let current = self.current?;

        if current.is_thriving() {
            return Some(HomeostaticAction::Maintain);
        }

        let costs = [
            ("cpu", (current.cpu_load - 0.5).abs()),
            ("memory", (current.memory_used - 0.5).abs()),
            ("io", current.io_pressure),
            ("cache", current.cache_miss_rate),
        ];

        let worst = costs.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;

        match worst.0 {
            "cpu" => {
                if current.cpu_load > 0.7 {
                    Some(HomeostaticAction::ReduceComputation)
                } else {
                    Some(HomeostaticAction::IncreaseUtilization)
                }
            }
            "memory" => {
                if current.memory_used > 0.8 {
                    Some(HomeostaticAction::FreeMemory)
                } else {
                    Some(HomeostaticAction::Maintain)
                }
            }
            "io" => Some(HomeostaticAction::BatchIO),
            "cache" => Some(HomeostaticAction::OptimizeDataLayout),
            _ => Some(HomeostaticAction::Maintain),
        }
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

/// Homeostatic actions to restore balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomeostaticAction {
    Maintain,
    ReduceComputation,
    IncreaseUtilization,
    FreeMemory,
    BatchIO,
    OptimizeDataLayout,
}

impl HomeostaticAction {
    /// Convert to natural language recommendation.
    pub fn to_recommendation(&self) -> &'static str {
        match self {
            Self::Maintain => "System is balanced. Continue current operation.",
            Self::ReduceComputation => "CPU load is high. Consider deferring non-critical tasks.",
            Self::IncreaseUtilization => "CPU is underutilized. Could process more in parallel.",
            Self::FreeMemory => "Memory pressure detected. Consider releasing cached data.",
            Self::BatchIO => "IO pressure high. Consider batching operations.",
            Self::OptimizeDataLayout => {
                "Cache misses high. Consider optimizing data access patterns."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_state(surprise: f32) -> HomeostasisState {
        HomeostasisState {
            cpu_load: 0.5,
            memory_used: 0.5,
            io_pressure: 0.2,
            cache_miss_rate: 0.1,
            prediction_error: surprise,
            timestamp: 1,
        }
    }

    #[test]
    fn homeostasis_cost() {
        let state = dummy_state(0.1);
        let cost = state.homeostasis_cost();
        assert!(cost > 0.0 && cost < 1.0);
    }

    #[test]
    fn pain_and_pleasure() {
        let distressed = dummy_state(0.9);
        let thriving = dummy_state(0.1);

        assert!(distressed.pain() > thriving.pain());
        assert!(thriving.pleasure() > distressed.pleasure());
    }

    #[test]
    fn interoceptive_record_and_retrieve() {
        let mut mem = InteroceptiveMemory::new();
        let state = dummy_state(0.3);
        mem.record_state(state.clone());

        assert_eq!(mem.current_state(), Some(state));
    }

    #[test]
    fn trend_detection() {
        let mut mem = InteroceptiveMemory::with_capacity(10);

        for i in 0..5 {
            mem.record_state(HomeostasisState {
                cpu_load: 0.3 + i as f32 * 0.1,
                memory_used: 0.5,
                io_pressure: 0.2,
                cache_miss_rate: 0.1,
                prediction_error: 0.1,
                timestamp: i as u64,
            });
        }

        let trend = mem.trend(5);
        assert!(trend < 0.0);
    }

    #[test]
    fn optimal_action() {
        let mut mem = InteroceptiveMemory::new();

        mem.record_state(HomeostasisState {
            cpu_load: 0.9,
            memory_used: 0.5,
            io_pressure: 0.2,
            cache_miss_rate: 0.1,
            prediction_error: 0.1,
            timestamp: 1,
        });

        let action = mem.optimal_action();
        assert_eq!(action, Some(HomeostaticAction::ReduceComputation));
    }
}

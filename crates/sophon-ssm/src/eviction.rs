//! LRU state pool with memory budget enforcement.
//!
//! When running multiple sequences or agent contexts, SSM states accumulate.
//! The StatePool manages a bounded set of SsmState instances keyed by
//! sequence/context ID, evicting least-recently-used entries when the
//! memory budget is exceeded.
//!
//! Memory accounting:
//!   Each SsmState = SSM_N * 4 bytes (f32) = 128 * 4 = 512 bytes.
//!   Per block: 512 bytes * NUM_BLOCKS = 8192 bytes.
//!   2GB budget / 8192 bytes per full-model state ≈ 262144 concurrent contexts.
//!   In practice, we reserve a fraction of the budget for states.

use sophon_config::{MAX_VRAM_BYTES, NUM_BLOCKS, SSM_N};

use crate::state::SsmState;

/// Bytes per state per block.
const BYTES_PER_STATE: usize = SSM_N * std::mem::size_of::<f32>();

/// Bytes per full model state (all blocks).
const BYTES_PER_CONTEXT: usize = BYTES_PER_STATE * NUM_BLOCKS;

/// Default fraction of VRAM budget reserved for state pool.
const STATE_BUDGET_FRACTION: f32 = 0.1;

/// A pooled set of SSM states keyed by context ID.
pub struct StatePool {
    entries: Vec<PoolEntry>,
    max_entries: usize,
    access_counter: u64,
}

struct PoolEntry {
    context_id: u64,
    states: Vec<SsmState>, // One per block
    last_access: u64,
}

impl StatePool {
    /// Create a state pool with automatic capacity from the VRAM budget.
    pub fn new() -> Self {
        let budget = (MAX_VRAM_BYTES as f32 * STATE_BUDGET_FRACTION) as usize;
        let max_entries = budget / BYTES_PER_CONTEXT;
        Self::with_capacity(max_entries.max(1))
    }

    /// Create a state pool with explicit max capacity.
    pub fn with_capacity(max_entries: usize) -> Self {
        StatePool {
            entries: Vec::new(),
            max_entries: max_entries.max(1),
            access_counter: 0,
        }
    }

    /// Get or create states for a context. Returns a mutable reference.
    /// If the pool is full, evicts the least-recently-used entry.
    pub fn get_or_create(&mut self, context_id: u64) -> &mut Vec<SsmState> {
        self.access_counter += 1;
        let counter = self.access_counter;

        // Search for existing entry
        if let Some(idx) = self.entries.iter().position(|e| e.context_id == context_id) {
            self.entries[idx].last_access = counter;
            return &mut self.entries[idx].states;
        }

        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        // Create new entry
        let states: Vec<SsmState> = (0..NUM_BLOCKS).map(|_| SsmState::new()).collect();
        self.entries.push(PoolEntry {
            context_id,
            states,
            last_access: counter,
        });

        let last = self.entries.len() - 1;
        &mut self.entries[last].states
    }

    /// Check if a context exists in the pool.
    pub fn contains(&self, context_id: u64) -> bool {
        self.entries.iter().any(|e| e.context_id == context_id)
    }

    /// Remove a specific context.
    pub fn remove(&mut self, context_id: u64) -> bool {
        if let Some(idx) = self.entries.iter().position(|e| e.context_id == context_id) {
            self.entries.swap_remove(idx);
            true
        } else {
            false
        }
    }

    /// Reset all states for a context.
    pub fn reset_context(&mut self, context_id: u64) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.context_id == context_id) {
            for state in &mut entry.states {
                state.reset();
            }
        }
    }

    /// Current number of cached contexts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_entries
    }

    /// Total memory used by all cached states in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.entries.len() * BYTES_PER_CONTEXT
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let mut lru_idx = 0;
        let mut lru_time = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.last_access < lru_time {
                lru_time = entry.last_access;
                lru_idx = i;
            }
        }
        self.entries.swap_remove(lru_idx);
    }
}

impl Default for StatePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_creates_empty() {
        let pool = StatePool::with_capacity(10);
        assert!(pool.is_empty());
        assert_eq!(pool.capacity(), 10);
    }

    #[test]
    fn get_or_create_returns_states() {
        let mut pool = StatePool::with_capacity(10);
        let states = pool.get_or_create(1);
        assert_eq!(states.len(), NUM_BLOCKS);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn get_existing_context() {
        let mut pool = StatePool::with_capacity(10);
        pool.get_or_create(42);
        assert!(pool.contains(42));
        pool.get_or_create(42); // Should reuse, not create new
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn eviction_at_capacity() {
        let mut pool = StatePool::with_capacity(3);
        pool.get_or_create(1);
        pool.get_or_create(2);
        pool.get_or_create(3);
        assert_eq!(pool.len(), 3);

        // This should evict context 1 (LRU)
        pool.get_or_create(4);
        assert_eq!(pool.len(), 3);
        assert!(!pool.contains(1));
        assert!(pool.contains(4));
    }

    #[test]
    fn lru_updates_on_access() {
        let mut pool = StatePool::with_capacity(3);
        pool.get_or_create(1);
        pool.get_or_create(2);
        pool.get_or_create(3);

        // Access 1 again to make it MRU
        pool.get_or_create(1);

        // Now 2 should be LRU and get evicted
        pool.get_or_create(4);
        assert!(pool.contains(1));
        assert!(!pool.contains(2));
        assert!(pool.contains(3));
        assert!(pool.contains(4));
    }

    #[test]
    fn remove_context() {
        let mut pool = StatePool::with_capacity(10);
        pool.get_or_create(1);
        assert!(pool.remove(1));
        assert!(!pool.contains(1));
        assert!(!pool.remove(1)); // Already removed
    }

    #[test]
    fn reset_context_zeros_states() {
        let mut pool = StatePool::with_capacity(10);
        {
            let states = pool.get_or_create(1);
            // Modify a state
            states[0].h[0] = 42.0;
        }
        pool.reset_context(1);
        let states = pool.get_or_create(1);
        assert_eq!(states[0].h[0], 0.0);
    }

    #[test]
    fn memory_accounting() {
        let mut pool = StatePool::with_capacity(10);
        assert_eq!(pool.memory_bytes(), 0);
        pool.get_or_create(1);
        assert_eq!(pool.memory_bytes(), BYTES_PER_CONTEXT);
        pool.get_or_create(2);
        assert_eq!(pool.memory_bytes(), 2 * BYTES_PER_CONTEXT);
    }

    #[test]
    fn default_pool_has_reasonable_capacity() {
        let pool = StatePool::new();
        // Budget = 10% of 2GB = ~200MB, each context = 8192 bytes
        // Expected ≈ 25000 entries
        assert!(
            pool.capacity() > 100,
            "capacity {} too small",
            pool.capacity()
        );
    }
}

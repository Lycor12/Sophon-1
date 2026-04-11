//! GALC — Gradient-Aware Lazy Checkpointing.
//!
//! Decides per-block whether to retain full activation caches or mark for
//! recomputation during backward. For Sophon-1's 16-block architecture with
//! D_MODEL=256, the total cache size per token is approximately:
//!   16 blocks * (5 * 256 * 4 bytes + SSM cache) ≈ ~140 KB
//! which fits comfortably in the 2 GB budget even for moderate sequences.
//!
//! Therefore, the initial implementation uses full caching (no recompute).
//! The GALC infrastructure is provided for future use when sequence lengths
//! or model sizes grow.

use sophon_config::{D_MODEL, NUM_BLOCKS};

/// Checkpointing strategy for the backward pass.
#[derive(Clone, Debug)]
pub enum CheckpointStrategy {
    /// Keep all activation caches (fastest, most memory).
    FullCache,
    /// Recompute blocks whose index is in the set (saves memory at compute cost).
    Selective(Vec<bool>),
    /// Recompute every N-th block.
    Periodic(usize),
}

impl CheckpointStrategy {
    /// Default strategy: full caching.
    pub fn default() -> Self {
        Self::FullCache
    }

    /// Should block `i` be recomputed during backward?
    pub fn should_recompute(&self, block_idx: usize) -> bool {
        match self {
            Self::FullCache => false,
            Self::Selective(flags) => {
                if block_idx < flags.len() {
                    flags[block_idx]
                } else {
                    false
                }
            }
            Self::Periodic(period) => {
                if *period == 0 {
                    false
                } else {
                    block_idx % period != 0
                }
            }
        }
    }

    /// Estimate memory savings in bytes for a given sequence length.
    pub fn estimated_savings(&self, seq_len: usize) -> usize {
        let cache_per_block = 5 * D_MODEL * 4; // 5 vectors * D_MODEL * sizeof(f32)
        let recompute_count = (0..NUM_BLOCKS)
            .filter(|&i| self.should_recompute(i))
            .count();
        recompute_count * cache_per_block * seq_len
    }
}

// ---------------------------------------------------------------------------
// GALC adaptive strategy builder
// ---------------------------------------------------------------------------

/// Build a GALC strategy from gradient norms observed during forward.
///
/// Blocks with small gradient contribution are candidates for recomputation.
/// `grad_norms`: per-block gradient L2 norm from a previous pass.
/// `memory_budget_bytes`: maximum activation memory allowed.
/// `seq_len`: current sequence length.
pub fn galc_build_strategy(
    grad_norms: &[f32],
    memory_budget_bytes: usize,
    seq_len: usize,
) -> CheckpointStrategy {
    let cache_per_block = 5 * D_MODEL * 4 * seq_len;
    let total_cache = NUM_BLOCKS * cache_per_block;

    if total_cache <= memory_budget_bytes {
        return CheckpointStrategy::FullCache;
    }

    // Sort blocks by gradient norm (ascending — low grad = recompute candidate)
    let mut indexed: Vec<(usize, f32)> = grad_norms.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut flags = vec![false; NUM_BLOCKS];
    let mut current_mem = total_cache;

    for (block_idx, _norm) in &indexed {
        if current_mem <= memory_budget_bytes {
            break;
        }
        flags[*block_idx] = true;
        current_mem -= cache_per_block;
    }

    CheckpointStrategy::Selective(flags)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cache_never_recomputes() {
        let strat = CheckpointStrategy::FullCache;
        for i in 0..NUM_BLOCKS {
            assert!(!strat.should_recompute(i));
        }
    }

    #[test]
    fn periodic_recomputes_correctly() {
        let strat = CheckpointStrategy::Periodic(4);
        // Blocks 0, 4, 8, 12 are NOT recomputed (they are checkpoints)
        // All others are recomputed
        assert!(!strat.should_recompute(0));
        assert!(strat.should_recompute(1));
        assert!(strat.should_recompute(2));
        assert!(strat.should_recompute(3));
        assert!(!strat.should_recompute(4));
    }

    #[test]
    fn selective_from_flags() {
        let flags = vec![false, true, false, true];
        let strat = CheckpointStrategy::Selective(flags);
        assert!(!strat.should_recompute(0));
        assert!(strat.should_recompute(1));
        assert!(!strat.should_recompute(2));
        assert!(strat.should_recompute(3));
        assert!(!strat.should_recompute(5)); // out of range -> false
    }

    #[test]
    fn galc_full_cache_when_budget_sufficient() {
        let norms = vec![1.0f32; NUM_BLOCKS];
        let budget = NUM_BLOCKS * 5 * D_MODEL * 4 * 100; // 100 tokens, plenty
        let strat = galc_build_strategy(&norms, budget, 100);
        match strat {
            CheckpointStrategy::FullCache => {} // expected
            _ => panic!("should be FullCache when budget is sufficient"),
        }
    }

    #[test]
    fn galc_selective_when_budget_tight() {
        let norms: Vec<f32> = (0..NUM_BLOCKS).map(|i| i as f32).collect();
        let budget = 5 * D_MODEL * 4 * 10; // only ~1 block's worth for 10 tokens
        let strat = galc_build_strategy(&norms, budget, 10);
        match strat {
            CheckpointStrategy::Selective(flags) => {
                // Low-gradient blocks should be marked for recompute
                let recompute_count = flags.iter().filter(|&&f| f).count();
                assert!(recompute_count > 0, "should recompute some blocks");
            }
            _ => panic!("should be Selective when budget is tight"),
        }
    }

    #[test]
    fn estimated_savings_full_cache_is_zero() {
        let strat = CheckpointStrategy::FullCache;
        assert_eq!(strat.estimated_savings(100), 0);
    }
}

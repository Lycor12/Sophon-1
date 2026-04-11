//! Training state: momentum buffers and step counters for all parameter groups.

use sophon_config::{D_MODEL, KAN_KNOTS, NUM_BLOCKS, VOCAB_SIZE};
use sophon_kan::spline::N_CTRL;
use sophon_optim::param_group::{ParamGroup, ParamKind};
use sophon_optim::tsm::MomentumState;

// ---------------------------------------------------------------------------
// Per-block momentum state
// ---------------------------------------------------------------------------

/// Momentum buffers for all parameters in one HybridBlock.
pub struct BlockMomentum {
    // LayerNorm 1
    pub ln1_gamma: MomentumState,
    pub ln1_beta: MomentumState,
    // KAN
    pub kan_coeffs: MomentumState,
    pub kan_w_base: MomentumState,
    pub kan_knots: MomentumState,
    pub kan_bias: MomentumState,
    // LayerNorm 2
    pub ln2_gamma: MomentumState,
    pub ln2_beta: MomentumState,
    // SSM
    pub ssm_s: MomentumState,
    pub ssm_u: MomentumState,
    pub ssm_v: MomentumState,
    pub ssm_b: MomentumState,
    pub ssm_c: MomentumState,
    pub ssm_d: MomentumState,
    pub ssm_log_delta: MomentumState,
}

impl BlockMomentum {
    pub fn new() -> Self {
        use sophon_config::{SSM_D, SSM_N, SSM_P, SSM_RANK};
        Self {
            ln1_gamma: MomentumState::new(D_MODEL),
            ln1_beta: MomentumState::new(D_MODEL),
            kan_coeffs: MomentumState::new(D_MODEL * D_MODEL * N_CTRL),
            kan_w_base: MomentumState::new(D_MODEL * D_MODEL),
            kan_knots: MomentumState::new(D_MODEL * D_MODEL * KAN_KNOTS),
            kan_bias: MomentumState::new(D_MODEL),
            ln2_gamma: MomentumState::new(D_MODEL),
            ln2_beta: MomentumState::new(D_MODEL),
            ssm_s: MomentumState::new(SSM_N),
            ssm_u: MomentumState::new(SSM_N * SSM_RANK),
            ssm_v: MomentumState::new(SSM_N * SSM_RANK),
            ssm_b: MomentumState::new(SSM_N * SSM_D),
            ssm_c: MomentumState::new(SSM_P * SSM_N),
            ssm_d: MomentumState::new(SSM_P * SSM_D),
            ssm_log_delta: MomentumState::new(1),
        }
    }
}

/// Momentum buffers for the output head.
pub struct HeadMomentum {
    pub weight: MomentumState,
    pub bias: MomentumState,
    pub ln_gamma: MomentumState,
    pub ln_beta: MomentumState,
}

impl HeadMomentum {
    pub fn new() -> Self {
        Self {
            weight: MomentumState::new(VOCAB_SIZE * D_MODEL),
            bias: MomentumState::new(VOCAB_SIZE),
            ln_gamma: MomentumState::new(D_MODEL),
            ln_beta: MomentumState::new(D_MODEL),
        }
    }
}

// ---------------------------------------------------------------------------
// Full training state
// ---------------------------------------------------------------------------

/// Complete training state for Sophon-1.
pub struct TrainState {
    /// Per-block momentum buffers.
    pub block_momentum: Vec<BlockMomentum>,
    /// Output head momentum buffers.
    pub head_momentum: HeadMomentum,
    /// Embedding table momentum buffer.
    pub embedding_momentum: MomentumState,
    /// Parameter groups (one per parameter kind).
    pub groups: TrainGroups,
    /// Global step counter.
    pub global_step: u64,
    /// Running exponential moving average of loss (for monitoring).
    pub ema_loss: f32,
    /// EMA decay factor.
    pub ema_decay: f32,
}

/// Pre-built parameter group configurations.
pub struct TrainGroups {
    pub kan_coeffs: ParamGroup,
    pub kan_w_base: ParamGroup,
    pub kan_knots: ParamGroup,
    pub kan_bias: ParamGroup,
    pub ssm_diag: ParamGroup,
    pub ssm_low_rank: ParamGroup,
    pub ssm_matrix: ParamGroup,
    pub ssm_feedthrough: ParamGroup,
    pub ssm_delta: ParamGroup,
    pub layer_norm: ParamGroup,
    pub head_weight: ParamGroup,
    pub head_bias: ParamGroup,
    pub embedding: ParamGroup,
}

impl TrainGroups {
    pub fn default() -> Self {
        Self {
            kan_coeffs: ParamGroup::default_for(ParamKind::KanCoeff),
            kan_w_base: ParamGroup::default_for(ParamKind::KanWBase),
            kan_knots: ParamGroup::default_for(ParamKind::KanKnot),
            kan_bias: ParamGroup::default_for(ParamKind::KanBias),
            ssm_diag: ParamGroup::default_for(ParamKind::SsmDiag),
            ssm_low_rank: ParamGroup::default_for(ParamKind::SsmLowRank),
            ssm_matrix: ParamGroup::default_for(ParamKind::SsmMatrix),
            ssm_feedthrough: ParamGroup::default_for(ParamKind::SsmFeedthrough),
            ssm_delta: ParamGroup::default_for(ParamKind::SsmDelta),
            layer_norm: ParamGroup::default_for(ParamKind::LayerNorm),
            head_weight: ParamGroup::default_for(ParamKind::HeadWeight),
            head_bias: ParamGroup::default_for(ParamKind::HeadBias),
            embedding: ParamGroup::default_for(ParamKind::Embedding),
        }
    }
}

impl TrainState {
    /// Create training state for a model.
    pub fn new() -> Self {
        let block_momentum: Vec<BlockMomentum> =
            (0..NUM_BLOCKS).map(|_| BlockMomentum::new()).collect();
        Self {
            block_momentum,
            head_momentum: HeadMomentum::new(),
            embedding_momentum: MomentumState::new(VOCAB_SIZE * D_MODEL),
            groups: TrainGroups::default(),
            global_step: 0,
            ema_loss: 0.0,
            ema_decay: 0.99,
        }
    }

    /// Update the EMA loss tracker.
    pub fn update_ema_loss(&mut self, loss: f32) {
        if self.global_step == 0 {
            self.ema_loss = loss;
        } else {
            self.ema_loss = self.ema_decay * self.ema_loss + (1.0 - self.ema_decay) * loss;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_state_creates() {
        let ts = TrainState::new();
        assert_eq!(ts.block_momentum.len(), NUM_BLOCKS);
        assert_eq!(ts.global_step, 0);
    }

    #[test]
    fn ema_loss_initialises_correctly() {
        let mut ts = TrainState::new();
        ts.update_ema_loss(5.0);
        assert!((ts.ema_loss - 5.0).abs() < 1e-6);

        ts.global_step = 1;
        ts.update_ema_loss(3.0);
        let expected = 0.99 * 5.0 + 0.01 * 3.0;
        assert!(
            (ts.ema_loss - expected).abs() < 1e-4,
            "ema={} expected={}",
            ts.ema_loss,
            expected
        );
    }
}

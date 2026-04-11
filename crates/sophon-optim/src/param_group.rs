//! Parameter group configuration for TSM-SGD.
//!
//! Each parameter group specifies the momentum coefficient, learning rate,
//! and optional modifiers (ternary projection, knot coupling, spectral clip).

/// Parameter group type — determines which TSM-SGD mechanisms apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamKind {
    /// KAN spline coefficients: high momentum (β₁), knot-aware coupling.
    KanCoeff,
    /// KAN w_base (residual scale): high momentum (β₁).
    KanWBase,
    /// KAN knot positions: high momentum (β₁), small LR.
    KanKnot,
    /// KAN bias: high momentum (β₁).
    KanBias,
    /// SSM diagonal log-magnitudes S: low momentum (β₂).
    SsmDiag,
    /// SSM low-rank factors U, V: low momentum (β₂), spectral norm clip.
    SsmLowRank,
    /// SSM input/output matrices B, C: low momentum (β₂).
    SsmMatrix,
    /// SSM feedthrough D: low momentum (β₂).
    SsmFeedthrough,
    /// SSM log_delta: low momentum (β₂), small LR.
    SsmDelta,
    /// LayerNorm gamma/beta: high momentum (β₁).
    LayerNorm,
    /// Output head weight: high momentum (β₁), ternary projection.
    HeadWeight,
    /// Output head bias: high momentum (β₁).
    HeadBias,
    /// Embedding table: high momentum (β₁).
    Embedding,
}

/// A group of parameters sharing the same optimiser configuration.
pub struct ParamGroup {
    /// Which kind of parameters this group contains.
    pub kind: ParamKind,
    /// Learning rate for this group.
    pub lr: f32,
    /// Weight decay (L2 regularisation coefficient).
    pub weight_decay: f32,
    /// Whether to apply ternary gradient projection.
    pub ternary_project: bool,
    /// Ternary projection threshold (distance from quantisation boundary).
    pub ternary_threshold: f32,
    /// Whether to apply spectral norm clipping on UV updates.
    pub spectral_clip: bool,
    /// Maximum spectral norm for UV gradient (only if spectral_clip is true).
    pub spectral_max_norm: f32,
}

impl ParamGroup {
    /// Default configuration for a given parameter kind.
    pub fn default_for(kind: ParamKind) -> Self {
        let (lr, wd, ternary, spectral) = match kind {
            ParamKind::KanCoeff => (1e-3, 1e-5, false, false),
            ParamKind::KanWBase => (1e-3, 1e-5, false, false),
            ParamKind::KanKnot => (1e-4, 0.0, false, false), // small LR for knots
            ParamKind::KanBias => (1e-3, 0.0, false, false),
            ParamKind::SsmDiag => (1e-3, 1e-5, false, false),
            ParamKind::SsmLowRank => (1e-3, 1e-4, false, true), // spectral clip
            ParamKind::SsmMatrix => (1e-3, 1e-5, false, false),
            ParamKind::SsmFeedthrough => (1e-3, 0.0, false, false),
            ParamKind::SsmDelta => (1e-4, 0.0, false, false), // small LR
            ParamKind::LayerNorm => (1e-3, 0.0, false, false),
            ParamKind::HeadWeight => (1e-3, 1e-5, true, false), // ternary project
            ParamKind::HeadBias => (1e-3, 0.0, false, false),
            ParamKind::Embedding => (1e-3, 1e-5, false, false),
        };
        Self {
            kind,
            lr,
            weight_decay: wd,
            ternary_project: ternary,
            ternary_threshold: 0.5,
            spectral_clip: spectral,
            spectral_max_norm: 1.0,
        }
    }

    /// Momentum coefficient for this group (dual-rate).
    pub fn momentum(&self) -> f32 {
        match self.kind {
            // KAN-family: β₁ = 0.95 (high momentum)
            ParamKind::KanCoeff
            | ParamKind::KanWBase
            | ParamKind::KanKnot
            | ParamKind::KanBias
            | ParamKind::LayerNorm
            | ParamKind::HeadWeight
            | ParamKind::HeadBias
            | ParamKind::Embedding => 0.95,

            // SSM-family: β₂ = 0.80 (lower momentum)
            ParamKind::SsmDiag
            | ParamKind::SsmLowRank
            | ParamKind::SsmMatrix
            | ParamKind::SsmFeedthrough
            | ParamKind::SsmDelta => 0.80,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kan_momentum_is_095() {
        let pg = ParamGroup::default_for(ParamKind::KanCoeff);
        assert!((pg.momentum() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn ssm_momentum_is_080() {
        let pg = ParamGroup::default_for(ParamKind::SsmDiag);
        assert!((pg.momentum() - 0.80).abs() < 1e-6);
    }

    #[test]
    fn head_weight_has_ternary_project() {
        let pg = ParamGroup::default_for(ParamKind::HeadWeight);
        assert!(pg.ternary_project);
    }

    #[test]
    fn ssm_low_rank_has_spectral_clip() {
        let pg = ParamGroup::default_for(ParamKind::SsmLowRank);
        assert!(pg.spectral_clip);
    }

    #[test]
    fn knot_lr_smaller_than_coeff_lr() {
        let knot = ParamGroup::default_for(ParamKind::KanKnot);
        let coeff = ParamGroup::default_for(ParamKind::KanCoeff);
        assert!(knot.lr < coeff.lr);
    }
}

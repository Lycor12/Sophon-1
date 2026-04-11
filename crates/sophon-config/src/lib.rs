//! sophon-config — authoritative locked constants for Sophon-1.
//!
//! All values below are directly specified by sophon-1-design-prompt-v3.md
//! or resolved by explicit human approval. Nothing here may be changed
//! without a corresponding update to the design document and a new user
//! approval entry in .sisyphus/drafts/.
//!
//! ASSUMPTIONS (per v3 §10.2 requirement to document all):
//!   - KAN knots per edge = 8  (user-approved 2026-04-11)
//!   - Interleaved blocks  = 16 (user-approved 2026-04-11)
//!   - SSM low-rank r      = 16 = N/8 (user-approved 2026-04-11)
//!   - ~3B parameter ceiling treated as upper bound, not target

#![forbid(unsafe_code)]

// ---------------------------------------------------------------------------
// Tokenisation
// ---------------------------------------------------------------------------

/// Byte-level vocabulary size. Exactly 256; no BPE or subword merges.
pub const VOCAB_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Embedding / model width
// ---------------------------------------------------------------------------

/// Model embedding dimension d_model.
pub const D_MODEL: usize = 256;

// ---------------------------------------------------------------------------
// KAN constants
// ---------------------------------------------------------------------------

/// Number of cubic B-spline knots per KAN edge (internal knots).
/// Approved value: 8.
pub const KAN_KNOTS: usize = 8;

/// Spline order for cubic B-splines. Must be 3.
pub const KAN_ORDER: usize = 3;

/// Total knot vector length per edge including clamped endpoints.
/// For cubic B-splines clamped at both ends:
///   len = KAN_KNOTS + 2*(KAN_ORDER+1) = 8 + 8 = 16
pub const KAN_KNOT_VEC_LEN: usize = KAN_KNOTS + 2 * (KAN_ORDER + 1);

// ---------------------------------------------------------------------------
// SSM constants
// ---------------------------------------------------------------------------

/// SSM hidden state dimension N.
pub const SSM_N: usize = 128;

/// SSM input projection dimension D.
pub const SSM_D: usize = 256;

/// SSM output projection dimension P.
pub const SSM_P: usize = 256;

/// SSM low-rank term r = N/8.
/// Parameterisation: A = -exp(S) + U V^T, U in R^{N x r}, V in R^{N x r}.
pub const SSM_RANK: usize = SSM_N / 8; // = 16

// ---------------------------------------------------------------------------
// Hybrid block stack
// ---------------------------------------------------------------------------

/// Number of interleaved KAN + SSM residual blocks.
/// Approved value: 16.
pub const NUM_BLOCKS: usize = 16;

// ---------------------------------------------------------------------------
// Hyperdimensional computing
// ---------------------------------------------------------------------------

/// HDC hypervector dimension for compositional binding.
pub const HDC_DIM: usize = 2048;

// ---------------------------------------------------------------------------
// LoRA adapters
// ---------------------------------------------------------------------------

/// Rank of LoRA-style low-rank adapters.
pub const LORA_RANK: usize = 16;

// ---------------------------------------------------------------------------
// Ternary quantisation
// ---------------------------------------------------------------------------

/// Ternary weight alphabet: {-1, 0, +1}.
/// Encoded as i8 values for arithmetic; packed to 2-bit fields for storage.
pub const TERNARY_NEG: i8 = -1;
pub const TERNARY_ZER: i8 = 0;
pub const TERNARY_POS: i8 = 1;

/// Bits per weight in packed ternary storage.
/// True information content = log2(3) ≈ 1.585 bits; we use 2-bit fields.
pub const TERNARY_BITS_PER_WEIGHT: usize = 2;

// ---------------------------------------------------------------------------
// Runtime / memory constraints
// ---------------------------------------------------------------------------

/// Hard VRAM ceiling at inference time (bytes).
pub const MAX_VRAM_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Target packed model size lower bound (bytes).
pub const MODEL_SIZE_LOWER: usize = 400 * 1024 * 1024; // 400 MiB

/// Target packed model size upper bound (bytes).
pub const MODEL_SIZE_UPPER: usize = 500 * 1024 * 1024; // 500 MiB

// ---------------------------------------------------------------------------
// Compile-time sanity assertions
// ---------------------------------------------------------------------------

const _: () = {
    assert!(SSM_RANK == SSM_N / 8, "SSM_RANK must equal N/8");
    assert!(KAN_ORDER == 3, "KAN must use cubic (order-3) B-splines");
    assert!(D_MODEL == SSM_D, "d_model must equal SSM_D");
    assert!(D_MODEL == SSM_P, "d_model must equal SSM_P");
    assert!(LORA_RANK == 16, "LoRA rank must be 16 per spec");
    assert!(HDC_DIM == 2048, "HDC hypervector dimension must be 2048");
    assert!(
        NUM_BLOCKS == 16,
        "Block count must be 16 per approved value"
    );
    assert!(KAN_KNOTS == 8, "KAN knots must be 8 per approved value");
    assert!(KAN_KNOT_VEC_LEN == KAN_KNOTS + 2 * (KAN_ORDER + 1));
};

// ---------------------------------------------------------------------------
// ModelConfig: single struct that captures every approved hyperparameter.
// ---------------------------------------------------------------------------

/// Immutable model configuration. All fields must match the constants above.
/// Provided as a runtime-accessible value for inspection and logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_blocks: usize,
    pub kan_knots: usize,
    pub kan_order: usize,
    pub ssm_n: usize,
    pub ssm_d: usize,
    pub ssm_p: usize,
    pub ssm_rank: usize,
    pub hdc_dim: usize,
    pub lora_rank: usize,
}

impl ModelConfig {
    /// The single canonical configuration derived from all approved constants.
    pub const fn canonical() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            d_model: D_MODEL,
            num_blocks: NUM_BLOCKS,
            kan_knots: KAN_KNOTS,
            kan_order: KAN_ORDER,
            ssm_n: SSM_N,
            ssm_d: SSM_D,
            ssm_p: SSM_P,
            ssm_rank: SSM_RANK,
            hdc_dim: HDC_DIM,
            lora_rank: LORA_RANK,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_config_matches_constants() {
        let cfg = ModelConfig::canonical();
        assert_eq!(cfg.vocab_size, VOCAB_SIZE);
        assert_eq!(cfg.d_model, D_MODEL);
        assert_eq!(cfg.num_blocks, NUM_BLOCKS);
        assert_eq!(cfg.kan_knots, KAN_KNOTS);
        assert_eq!(cfg.kan_order, KAN_ORDER);
        assert_eq!(cfg.ssm_n, SSM_N);
        assert_eq!(cfg.ssm_rank, SSM_RANK);
        assert_eq!(cfg.hdc_dim, HDC_DIM);
        assert_eq!(cfg.lora_rank, LORA_RANK);
    }

    #[test]
    fn ssm_rank_is_n_over_8() {
        assert_eq!(SSM_RANK * 8, SSM_N);
    }

    #[test]
    fn kan_knot_vec_len_correct() {
        assert_eq!(KAN_KNOT_VEC_LEN, 16);
    }
}

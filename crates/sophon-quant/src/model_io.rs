//! Save and load a full Sophon-1 model to/from the `.sophon` binary format.
//!
//! This module bridges `sophon-model` types with the `serialize` module.
//!
//! Weight categorization for quantization:
//! - **Ternary quantized** (large matrices, most of the parameter budget):
//!   - Embedding table (VOCAB_SIZE × D_MODEL)
//!   - KAN spline coefficients (per edge, 12 coeffs each)
//!   - SSM B matrix (N × D) per block
//!   - SSM C matrix (P × N) per block
//!   - SSM D matrix (P × D) per block
//!   - Head weight (VOCAB_SIZE × D_MODEL)
//!
//! - **F32 raw** (small, precision-sensitive parameters):
//!   - LayerNorm gamma/beta (2 × D_MODEL per block + head)
//!   - KAN w_base (1 per edge)
//!   - KAN bias (D_MODEL per layer)
//!   - KAN knot positions (8 per edge, needs precision for spline eval)
//!   - SSM S vector (N per block, log-space diagonal)
//!   - SSM U, V matrices (N × r per block, low-rank factors)
//!   - SSM log_delta (1 scalar per block)
//!   - Head bias (VOCAB_SIZE)

use crate::quant::BLOCK_SIZE;
use crate::serialize::{ModelReader, ModelWriter};
use sophon_config::{D_MODEL, KAN_KNOTS, NUM_BLOCKS, SSM_D, SSM_N, SSM_P, SSM_RANK, VOCAB_SIZE};

use std::io::{self, Read, Seek, Write};

/// Section naming convention:
///   "embed"                    — embedding table
///   "block.{i}.ln1_gamma"     — LN1 gamma for block i
///   "block.{i}.ln1_beta"      — LN1 beta for block i
///   "block.{i}.kan_coeffs"    — all KAN spline coefficients, flattened
///   "block.{i}.kan_w_base"    — all w_base values
///   "block.{i}.kan_knots"     — all knot positions
///   "block.{i}.kan_bias"      — KAN bias vector
///   "block.{i}.ssm_s"         — SSM diagonal log-magnitudes
///   "block.{i}.ssm_u"         — SSM low-rank U
///   "block.{i}.ssm_v"         — SSM low-rank V
///   "block.{i}.ssm_b"         — SSM input matrix
///   "block.{i}.ssm_c"         — SSM output matrix
///   "block.{i}.ssm_d"         — SSM feedthrough matrix
///   "block.{i}.ssm_log_delta" — SSM log step size
///   "head.ln_gamma"           — output head LN gamma
///   "head.ln_beta"            — output head LN beta
///   "head.weight"             — output head linear weight
///   "head.bias"               — output head linear bias
use crate::serialize::SectionKind;

/// Parameters extracted from a Sophon-1 model, ready for serialization.
/// This is a flat representation that doesn't depend on the model struct.
pub struct ModelParams {
    pub embed_table: Vec<f32>,
    pub blocks: Vec<BlockParams>,
    pub head_ln_gamma: Vec<f32>,
    pub head_ln_beta: Vec<f32>,
    pub head_weight: Vec<f32>,
    pub head_bias: Vec<f32>,
}

/// Parameters for one hybrid block.
pub struct BlockParams {
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub kan_coeffs: Vec<f32>, // all edges, flattened [d_in * d_out * N_CTRL]
    pub kan_w_base: Vec<f32>, // [d_in * d_out]
    pub kan_knots: Vec<f32>,  // [d_in * d_out * KAN_KNOTS] (internal knots only)
    pub kan_bias: Vec<f32>,   // [d_out]
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    pub ssm_s: Vec<f32>,
    pub ssm_u: Vec<f32>,
    pub ssm_v: Vec<f32>,
    pub ssm_b: Vec<f32>,
    pub ssm_c: Vec<f32>,
    pub ssm_d: Vec<f32>,
    pub ssm_log_delta: f32,
}

/// Write a complete model to the `.sophon` format using compile-time architecture constants.
///
/// `quantize`: if true, large matrices are ternary-quantized.
///             if false, everything is stored as f32.
pub fn save_model<W: Write + Seek>(
    params: &ModelParams,
    quantize: bool,
    w: &mut W,
) -> io::Result<()> {
    save_model_with_dims(
        params,
        quantize,
        w,
        D_MODEL as u32,
        VOCAB_SIZE as u32,
        SSM_N as u32,
        SSM_D as u32,
        SSM_P as u32,
        SSM_RANK as u32,
    )
}

/// Write a complete model with explicit dimension parameters.
/// Used when dimensions differ from compile-time constants (e.g. tests).
pub fn save_model_with_dims<W: Write + Seek>(
    params: &ModelParams,
    quantize: bool,
    w: &mut W,
    d_model: u32,
    vocab: u32,
    ssm_n: u32,
    ssm_d: u32,
    ssm_p: u32,
    ssm_r: u32,
) -> io::Result<()> {
    let n_ctrl = (KAN_KNOTS + 3 + 1) as u32; // 12
    let n_blocks = params.blocks.len() as u32;

    let mut writer = ModelWriter::new(
        d_model,
        n_blocks,
        vocab,
        ssm_n,
        ssm_d,
        ssm_p,
        ssm_r,
        KAN_KNOTS as u32,
        n_ctrl,
    );

    // Embedding (quantizable)
    if quantize {
        writer.add_ternary_section("embed", &params.embed_table);
    } else {
        writer.add_f32_section("embed", &params.embed_table);
    }

    // Blocks
    for (i, bp) in params.blocks.iter().enumerate() {
        let prefix = format!("block.{i}");

        // LayerNorm — always f32
        writer.add_f32_section(&format!("{prefix}.ln1_gamma"), &bp.ln1_gamma);
        writer.add_f32_section(&format!("{prefix}.ln1_beta"), &bp.ln1_beta);

        // KAN coefficients (quantizable), w_base/knots/bias (f32)
        if quantize {
            writer.add_ternary_section(&format!("{prefix}.kan_coeffs"), &bp.kan_coeffs);
        } else {
            writer.add_f32_section(&format!("{prefix}.kan_coeffs"), &bp.kan_coeffs);
        }
        writer.add_f32_section(&format!("{prefix}.kan_w_base"), &bp.kan_w_base);
        writer.add_f32_section(&format!("{prefix}.kan_knots"), &bp.kan_knots);
        writer.add_f32_section(&format!("{prefix}.kan_bias"), &bp.kan_bias);

        // LayerNorm 2 — always f32
        writer.add_f32_section(&format!("{prefix}.ln2_gamma"), &bp.ln2_gamma);
        writer.add_f32_section(&format!("{prefix}.ln2_beta"), &bp.ln2_beta);

        // SSM params: S, U, V are f32 (precision-sensitive); B, C, D quantizable
        writer.add_f32_section(&format!("{prefix}.ssm_s"), &bp.ssm_s);
        writer.add_f32_section(&format!("{prefix}.ssm_u"), &bp.ssm_u);
        writer.add_f32_section(&format!("{prefix}.ssm_v"), &bp.ssm_v);

        if quantize {
            writer.add_ternary_section(&format!("{prefix}.ssm_b"), &bp.ssm_b);
            writer.add_ternary_section(&format!("{prefix}.ssm_c"), &bp.ssm_c);
            writer.add_ternary_section(&format!("{prefix}.ssm_d"), &bp.ssm_d);
        } else {
            writer.add_f32_section(&format!("{prefix}.ssm_b"), &bp.ssm_b);
            writer.add_f32_section(&format!("{prefix}.ssm_c"), &bp.ssm_c);
            writer.add_f32_section(&format!("{prefix}.ssm_d"), &bp.ssm_d);
        }

        writer.add_f32_section(&format!("{prefix}.ssm_log_delta"), &[bp.ssm_log_delta]);
    }

    // Head
    writer.add_f32_section("head.ln_gamma", &params.head_ln_gamma);
    writer.add_f32_section("head.ln_beta", &params.head_ln_beta);
    if quantize {
        writer.add_ternary_section("head.weight", &params.head_weight);
    } else {
        writer.add_f32_section("head.weight", &params.head_weight);
    }
    writer.add_f32_section("head.bias", &params.head_bias);

    writer.write_to(w)
}

/// Load model parameters from a `.sophon` file.
///
/// Validates that the file's architecture constants match the compile-time
/// Sophon-1 config. Use `load_model_unchecked` for test/foreign files.
pub fn load_model<R: Read + Seek>(r: &mut R) -> io::Result<ModelParams> {
    let reader = ModelReader::read_header(r)?;

    // Validate architecture
    let h = &reader.header;
    if h.d_model != D_MODEL as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("d_model mismatch: file={}, expected={}", h.d_model, D_MODEL),
        ));
    }
    if h.n_blocks != NUM_BLOCKS as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "n_blocks mismatch: file={}, expected={}",
                h.n_blocks, NUM_BLOCKS
            ),
        ));
    }

    load_model_inner(&reader, r)
}

/// Load model parameters without checking compile-time architecture constants.
/// Used for tests with smaller model dimensions and for foreign weight files.
pub fn load_model_unchecked<R: Read + Seek>(r: &mut R) -> io::Result<ModelParams> {
    let reader = ModelReader::read_header(r)?;
    load_model_inner(&reader, r)
}

fn load_model_inner<R: Read + Seek>(reader: &ModelReader, r: &mut R) -> io::Result<ModelParams> {
    let n_blocks = reader.header.n_blocks as usize;

    let embed_table = reader.read_section(r, "embed")?;

    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let prefix = format!("block.{i}");
        let bp = BlockParams {
            ln1_gamma: reader.read_section(r, &format!("{prefix}.ln1_gamma"))?,
            ln1_beta: reader.read_section(r, &format!("{prefix}.ln1_beta"))?,
            kan_coeffs: reader.read_section(r, &format!("{prefix}.kan_coeffs"))?,
            kan_w_base: reader.read_section(r, &format!("{prefix}.kan_w_base"))?,
            kan_knots: reader.read_section(r, &format!("{prefix}.kan_knots"))?,
            kan_bias: reader.read_section(r, &format!("{prefix}.kan_bias"))?,
            ln2_gamma: reader.read_section(r, &format!("{prefix}.ln2_gamma"))?,
            ln2_beta: reader.read_section(r, &format!("{prefix}.ln2_beta"))?,
            ssm_s: reader.read_section(r, &format!("{prefix}.ssm_s"))?,
            ssm_u: reader.read_section(r, &format!("{prefix}.ssm_u"))?,
            ssm_v: reader.read_section(r, &format!("{prefix}.ssm_v"))?,
            ssm_b: reader.read_section(r, &format!("{prefix}.ssm_b"))?,
            ssm_c: reader.read_section(r, &format!("{prefix}.ssm_c"))?,
            ssm_d: reader.read_section(r, &format!("{prefix}.ssm_d"))?,
            ssm_log_delta: {
                let v = reader.read_section(r, &format!("{prefix}.ssm_log_delta"))?;
                if v.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "empty log_delta",
                    ));
                }
                v[0]
            },
        };
        blocks.push(bp);
    }

    let head_ln_gamma = reader.read_section(r, "head.ln_gamma")?;
    let head_ln_beta = reader.read_section(r, "head.ln_beta")?;
    let head_weight = reader.read_section(r, "head.weight")?;
    let head_bias = reader.read_section(r, "head.bias")?;

    Ok(ModelParams {
        embed_table,
        blocks,
        head_ln_gamma,
        head_ln_beta,
        head_weight,
        head_bias,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Create minimal valid model params for testing.
    ///
    /// Uses real architecture sizes but only 2 blocks (not 16) to keep
    /// memory and time reasonable in debug-mode tests.
    const TEST_N_BLOCKS: usize = 2;

    fn dummy_params() -> ModelParams {
        // Use smaller edge count to avoid 256*256=65K edges * 12 coeffs * 16 blocks
        // which causes multi-GB allocations in debug mode.
        // For testing the serialization format, 4x4 edges is sufficient.
        let test_d = 4;
        let n_edges = test_d * test_d;
        let n_ctrl = KAN_KNOTS + 3 + 1; // 12
        let test_vocab = 8;
        let test_ssm_n = 4;
        let test_ssm_d = 4;
        let test_ssm_p = 4;
        let test_ssm_r = 2;

        let blocks: Vec<BlockParams> = (0..TEST_N_BLOCKS)
            .map(|_| BlockParams {
                ln1_gamma: vec![1.0; test_d],
                ln1_beta: vec![0.0; test_d],
                kan_coeffs: vec![0.01; n_edges * n_ctrl],
                kan_w_base: vec![0.0; n_edges],
                kan_knots: vec![0.5; n_edges * KAN_KNOTS],
                kan_bias: vec![0.0; test_d],
                ln2_gamma: vec![1.0; test_d],
                ln2_beta: vec![0.0; test_d],
                ssm_s: vec![0.0; test_ssm_n],
                ssm_u: vec![0.01; test_ssm_n * test_ssm_r],
                ssm_v: vec![0.01; test_ssm_n * test_ssm_r],
                ssm_b: vec![0.01; test_ssm_n * test_ssm_d],
                ssm_c: vec![0.01; test_ssm_p * test_ssm_n],
                ssm_d: vec![0.0; test_ssm_p * test_ssm_d],
                ssm_log_delta: 0.1f32.ln(),
            })
            .collect();

        ModelParams {
            embed_table: vec![0.1; test_vocab * test_d],
            blocks,
            head_ln_gamma: vec![1.0; test_d],
            head_ln_beta: vec![0.0; test_d],
            head_weight: vec![0.01; test_vocab * test_d],
            head_bias: vec![0.0; test_vocab],
        }
    }

    fn test_writer() -> ModelWriter {
        ModelWriter::new(4, TEST_N_BLOCKS as u32, 8, 4, 4, 4, 2, KAN_KNOTS as u32, 12)
    }

    fn save_test(params: &ModelParams, quantize: bool) -> Cursor<Vec<u8>> {
        let mut buf = Cursor::new(Vec::new());
        save_model_with_dims(params, quantize, &mut buf, 4, 8, 4, 4, 4, 2).unwrap();
        buf
    }

    #[test]
    fn roundtrip_f32_model() {
        let params = dummy_params();
        let mut buf = save_test(&params, false);

        buf.seek(std::io::SeekFrom::Start(0)).unwrap();
        let loaded = load_model_unchecked(&mut buf).unwrap();

        assert_eq!(loaded.embed_table.len(), params.embed_table.len());
        assert_eq!(loaded.blocks.len(), TEST_N_BLOCKS);
        assert_eq!(loaded.head_weight.len(), params.head_weight.len());

        // Exact f32 roundtrip
        for (a, b) in params.embed_table.iter().zip(&loaded.embed_table) {
            assert!((a - b).abs() < 1e-7);
        }
        for (a, b) in params.head_ln_gamma.iter().zip(&loaded.head_ln_gamma) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn roundtrip_quantized_model() {
        let params = dummy_params();
        let mut buf = save_test(&params, true);

        let file_size = buf.get_ref().len();
        // Quantized file should be significantly smaller
        let buf_f32 = save_test(&params, false);
        let f32_size = buf_f32.get_ref().len();
        assert!(
            file_size < f32_size,
            "quantized={file_size} should be < f32={f32_size}"
        );

        buf.seek(std::io::SeekFrom::Start(0)).unwrap();
        let loaded = load_model_unchecked(&mut buf).unwrap();
        assert_eq!(loaded.blocks.len(), TEST_N_BLOCKS);
        assert_eq!(loaded.embed_table.len(), 8 * 4); // test_vocab * test_d

        // f32 sections should be exact
        for (a, b) in params.blocks[0]
            .ln1_gamma
            .iter()
            .zip(&loaded.blocks[0].ln1_gamma)
        {
            assert!((a - b).abs() < 1e-7);
        }
        assert!((params.blocks[0].ssm_log_delta - loaded.blocks[0].ssm_log_delta).abs() < 1e-7);
    }

    #[test]
    fn quantized_smaller_than_f32() {
        let params = dummy_params();
        let buf_q = save_test(&params, true);
        let buf_f32 = save_test(&params, false);
        assert!(
            buf_q.get_ref().len() < buf_f32.get_ref().len(),
            "quantized={} should be < f32={}",
            buf_q.get_ref().len(),
            buf_f32.get_ref().len()
        );
    }
}

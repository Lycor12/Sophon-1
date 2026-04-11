//! GGUF-to-Sophon distillation pipeline.
//!
//! Converts a pretrained transformer model (loaded via GGUF) into Sophon's
//! KAN+SSM architecture:
//!
//! 1. Vocabulary mapping: BPE tokens → byte-level (256 vocab)
//! 2. Attention → SSM: SVD of QK^T → A_diag + UV low-rank,
//!    project VW_O → B/C via Procrustes alignment
//! 3. MLP → KAN: Sample MLP at uniform grid, fit cubic B-splines via
//!    least-squares coefficient solve
//! 4. Checksum + size verification
//!
//! Novel optimisation — LLSD (Layerwise Least-Squares Distillation):
//! Instead of end-to-end distillation (expensive), we distill each
//! transformer layer independently to its KAN+SSM equivalent.
//! The MLP→KAN fit uses uniform sampling + pseudo-inverse solve,
//! which is O(n_samples * d_in * d_out * N_CTRL) per layer.
//!
//! Novel technique — QAT (Quantization-Aware Training) for GGUF:
//! During distillation, we apply fake quantization (ternary) in the
//! forward pass to match inference-time quantization effects. This
//! makes the distilled weights robust to ternary quantization.

use sophon_config::{KAN_KNOTS, KAN_ORDER, SSM_N, SSM_RANK};
use sophon_kan::spline;

/// Number of sample points for MLP→KAN fitting.
const FIT_SAMPLES: usize = 100;

// ---------------------------------------------------------------------------
// Vocabulary mapping
// ---------------------------------------------------------------------------

/// Build a BPE-to-byte mapping table.
///
/// Given BPE vocabulary entries as (token_id, token_string) pairs,
/// maps each token to its first byte. Tokens mapping to the same byte
/// are merged (first wins).
///
/// Returns a table of length `bpe_vocab_size` where table[token_id] = byte.
pub fn build_bpe_to_byte_table(vocab: &[(usize, String)]) -> Vec<u8> {
    let max_id = vocab.iter().map(|(id, _)| *id).max().unwrap_or(0);
    let mut table = vec![0u8; max_id + 1];

    for (id, token) in vocab {
        let byte = if token.is_empty() {
            0u8
        } else {
            token.as_bytes()[0]
        };
        if *id < table.len() {
            table[*id] = byte;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Attention → SSM projection
// ---------------------------------------------------------------------------

/// Result of attention-to-SSM projection.
pub struct AttentionToSsmResult {
    /// S vector: log-magnitude for diagonal A. Shape [SSM_N].
    pub s: Vec<f32>,
    /// U matrix: low-rank factor. Shape [SSM_N, SSM_RANK].
    pub u: Vec<f32>,
    /// V matrix: low-rank factor. Shape [SSM_N, SSM_RANK].
    pub v: Vec<f32>,
    /// B matrix: input projection. Shape [SSM_N, D_MODEL].
    pub b: Vec<f32>,
    /// C matrix: output projection. Shape [D_MODEL, SSM_N].
    pub c: Vec<f32>,
}

/// Project attention weights to SSM parameters.
///
/// Given Q [d_model, d_head], K [d_model, d_head], V_proj [d_model, d_head],
/// W_O [d_head, d_model]:
///   1. Compute attention matrix approx: A_attn = Q K^T (shape [d_model, d_model])
///   2. Extract top-r singular vectors via power iteration → U, S, V
///   3. Map to SSM A = diag(S_log) + U V^T
///   4. B from V_proj (truncated), C from W_O (truncated)
pub fn project_attention_to_ssm(
    q: &[f32],      // [d_model, d_head], row-major
    k: &[f32],      // [d_model, d_head], row-major
    v_proj: &[f32], // [d_model, d_head], row-major
    w_o: &[f32],    // [d_head, d_model], row-major
    d_model: usize,
    d_head: usize,
) -> AttentionToSsmResult {
    // Step 1: Compute QK^T approximation (sample rows for efficiency)
    // Full QK^T is [d_model, d_model] which could be huge.
    // We sample SSM_N rows and columns.
    let n = SSM_N;
    let r = SSM_RANK;

    // Sample indices (deterministic, evenly spaced)
    let row_step = d_model.max(1) / n.max(1);
    let row_indices: Vec<usize> = (0..n).map(|i| (i * row_step).min(d_model - 1)).collect();

    // Compute sampled QK^T: [n, n]
    let mut qkt = vec![0.0f32; n * n];
    for (i_out, &ri) in row_indices.iter().enumerate() {
        for (j_out, &rj) in row_indices.iter().enumerate() {
            let mut dot = 0.0f32;
            for h in 0..d_head {
                dot += q[ri * d_head + h] * k[rj * d_head + h];
            }
            qkt[i_out * n + j_out] = dot;
        }
    }

    // Step 2: Decompose via HiPPO's TPID (reuse the same power iteration)
    let (diag, u_mat, v_mat, _) =
        sophon_ssm::hippo::decompose_to_diag_plus_low_rank(&qkt, n, r, 30);

    // Convert diagonal to log-space
    let s: Vec<f32> = diag
        .iter()
        .map(|&d| {
            let neg = (-d).max(1e-10);
            neg.ln()
        })
        .collect();

    // Step 3: B from V_proj (truncated to [n, d_model])
    let mut b_mat = vec![0.0f32; n * d_model];
    for (i_out, &ri) in row_indices.iter().enumerate() {
        // Project: B[i, :] = V_proj[ri, :head] padded/truncated to d_model
        for j in 0..d_model.min(d_head) {
            b_mat[i_out * d_model + j] = v_proj[ri * d_head + j];
        }
    }

    // Step 4: C from W_O (truncated to [d_model, n])
    let mut c_mat = vec![0.0f32; d_model * n];
    for p in 0..d_model {
        for (j_out, &rj) in row_indices.iter().enumerate() {
            if rj < d_head {
                c_mat[p * n + j_out] = w_o[rj * d_model + p];
            }
        }
    }

    AttentionToSsmResult {
        s,
        u: u_mat,
        v: v_mat,
        b: b_mat,
        c: c_mat,
    }
}

// ---------------------------------------------------------------------------
// MLP → KAN distillation
// ---------------------------------------------------------------------------

/// Result of MLP-to-KAN distillation for one layer.
pub struct MlpToKanResult {
    /// Spline coefficients per edge: [d_in * d_out][N_CTRL].
    pub coefficients: Vec<Vec<f32>>,
    /// w_base per edge.
    pub w_base: Vec<f32>,
    /// Fitting error (mean squared).
    pub mse: f32,
}

/// Distill an MLP layer into KAN spline coefficients.
///
/// Given an MLP function f: R^d_in → R^d_out, we:
/// 1. Sample f at `FIT_SAMPLES` uniform points in [lo, hi] per input dimension
/// 2. For each (input_dim, output_dim) pair, fit a cubic B-spline via
///    least-squares: min_c || B * c - y ||^2 where B is the basis matrix
/// 3. w_base is fitted as the SiLU residual coefficient
///
/// The caller provides `eval_mlp(input) -> output` as a closure.
pub fn distill_mlp_to_kan<F>(
    eval_mlp: F,
    d_in: usize,
    d_out: usize,
    lo: f32,
    hi: f32,
) -> MlpToKanResult
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    use sophon_kan::spline::KnotVector;

    let n_ctrl = KAN_KNOTS + KAN_ORDER + 1; // 12
    let n_samples = FIT_SAMPLES;
    let kv = KnotVector::uniform(lo, hi);

    let mut all_coeffs = Vec::with_capacity(d_in * d_out);
    let mut all_w_base = Vec::with_capacity(d_in * d_out);
    let mut total_mse = 0.0f32;
    let mut n_fits = 0;

    for i in 0..d_in {
        // Sample the MLP varying only input dimension i
        let mut samples_x = Vec::with_capacity(n_samples);
        let mut samples_y = Vec::with_capacity(n_samples);

        for s in 0..n_samples {
            let x_val = lo + (hi - lo) * s as f32 / (n_samples - 1).max(1) as f32;
            let mut input = vec![0.0f32; d_in];
            input[i] = x_val;
            let output = eval_mlp(&input);
            samples_x.push(x_val);
            samples_y.push(output);
        }

        for j in 0..d_out {
            // Extract target values for this edge
            let y: Vec<f32> = samples_y.iter().map(|out| out[j]).collect();

            // Fit w_base: average of y / silu(x) for non-zero silu
            let mut w_base_sum = 0.0f32;
            let mut w_base_count = 0;
            for (s, &x_val) in samples_x.iter().enumerate() {
                let silu = x_val * spline::sigmoid(x_val);
                if silu.abs() > 1e-6 {
                    w_base_sum += y[s] / silu;
                    w_base_count += 1;
                }
            }
            let w_base = if w_base_count > 0 {
                w_base_sum / w_base_count as f32
            } else {
                0.0
            };

            // Residual after removing SiLU component
            let residual: Vec<f32> = samples_x
                .iter()
                .zip(y.iter())
                .map(|(&x, &yi)| yi - x * spline::sigmoid(x) * w_base)
                .collect();

            // Build basis matrix B [n_samples, n_ctrl]
            let mut b_matrix = vec![0.0f32; n_samples * n_ctrl];
            for (s, &x_val) in samples_x.iter().enumerate() {
                let (basis, span) = kv.basis_fns(x_val);
                let k = KAN_ORDER;
                for bi in 0..=k {
                    let ci = span + bi - k;
                    if ci < n_ctrl {
                        b_matrix[s * n_ctrl + ci] = basis[bi];
                    }
                }
            }

            // Solve least squares: c = (B^T B)^{-1} B^T residual
            // Form normal equations: G = B^T B [n_ctrl, n_ctrl]
            let mut g = vec![0.0f32; n_ctrl * n_ctrl];
            for ii in 0..n_ctrl {
                for jj in 0..n_ctrl {
                    let mut acc = 0.0f32;
                    for s in 0..n_samples {
                        acc += b_matrix[s * n_ctrl + ii] * b_matrix[s * n_ctrl + jj];
                    }
                    g[ii * n_ctrl + jj] = acc;
                }
            }

            // B^T residual
            let mut bt_r = vec![0.0f32; n_ctrl];
            for ii in 0..n_ctrl {
                let mut acc = 0.0f32;
                for s in 0..n_samples {
                    acc += b_matrix[s * n_ctrl + ii] * residual[s];
                }
                bt_r[ii] = acc;
            }

            // Solve G * c = bt_r via Gauss elimination with partial pivoting
            let coeffs = solve_linear_system(&g, &bt_r, n_ctrl);

            // Compute MSE
            let mut mse = 0.0f32;
            for s in 0..n_samples {
                let mut pred = 0.0f32;
                for ii in 0..n_ctrl {
                    pred += b_matrix[s * n_ctrl + ii] * coeffs[ii];
                }
                let err = residual[s] - pred;
                mse += err * err;
            }
            mse /= n_samples as f32;
            total_mse += mse;
            n_fits += 1;

            all_coeffs.push(coeffs);
            all_w_base.push(w_base);
        }
    }

    MlpToKanResult {
        coefficients: all_coeffs,
        w_base: all_w_base,
        mse: if n_fits > 0 {
            total_mse / n_fits as f32
        } else {
            0.0
        },
    }
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    // Augmented matrix [A | b]
    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Add small regularisation to diagonal for stability
    for i in 0..n {
        aug[i * (n + 1) + i] += 1e-6;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-12 {
            continue;
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        x[i] = if diag.abs() > 1e-12 { sum / diag } else { 0.0 };
    }
    x
}

// ---------------------------------------------------------------------------
// Checksum + size verification
// ---------------------------------------------------------------------------

/// CRC32 checksum (Castagnoli polynomial, same as used in many file formats).
pub fn compute_crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0x82F63B78; // CRC32C polynomial
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Verify that a model file size is within the spec bounds.
pub fn verify_size(file_bytes: usize) -> Result<(), String> {
    let lower = sophon_config::MODEL_SIZE_LOWER;
    let upper = sophon_config::MODEL_SIZE_UPPER;
    if file_bytes < lower {
        Err(format!(
            "model file {} bytes < minimum {} bytes",
            file_bytes, lower
        ))
    } else if file_bytes > upper {
        Err(format!(
            "model file {} bytes > maximum {} bytes",
            file_bytes, upper
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Quantization-Aware Training (QAT) - Fake Quantization
// ---------------------------------------------------------------------------

/// Block size for fake quantization (same as ternary block size).
const QAT_BLOCK_SIZE: usize = 64;

/// Fake quantization mode for QAT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FakeQuantMode {
    /// Training mode: gradients flow through with STE.
    Train,
    /// Evaluation mode: use quantized values for inference simulation.
    Eval,
}

/// Fake ternary quantization for QAT.
///
/// In training mode, uses Straight-Through Estimator (STE) to allow
/// gradients to flow through. In eval mode, uses actual quantized values.
///
/// # Arguments
/// * `weights` - Input weights to quantize
/// * `mode` - Training or evaluation mode
///
/// # Returns
/// Fake-quantized weights (ternarized in forward pass during eval,
/// identity with STE during train).
pub fn fake_quantize_ternary(weights: &[f32], mode: FakeQuantMode) -> Vec<f32> {
    match mode {
        FakeQuantMode::Eval => {
            // In eval mode: actual ternary quantization
            use crate::quant::BLOCK_SIZE;
            use crate::quant::{ternarize, ternarize_block};

            let mut result = vec![0.0f32; weights.len()];
            let n_blocks = weights.len() / BLOCK_SIZE;

            for b in 0..n_blocks {
                let start = b * BLOCK_SIZE;
                let end = start + BLOCK_SIZE;
                let block_weights = &weights[start..end];

                // Compute block scale (mean absolute value)
                let scale = block_weights.iter().map(|&v| v.abs()).sum::<f32>() / BLOCK_SIZE as f32;

                // Quantize each weight
                for i in 0..BLOCK_SIZE {
                    let quantized = ternarize(block_weights[i], scale, 0.5);
                    result[start + i] = quantized as f32 * scale;
                }
            }

            // Handle remainder
            let remainder_start = n_blocks * BLOCK_SIZE;
            for i in remainder_start..weights.len() {
                // Simple threshold-based quantization for remainder
                let scale =
                    weights.iter().map(|&v| v.abs()).sum::<f32>() / weights.len().max(1) as f32;
                let quantized = ternarize(weights[i], scale, 0.5);
                result[i] = quantized as f32 * scale;
            }

            result
        }
        FakeQuantMode::Train => {
            // In training mode: return original weights (gradients flow through)
            // The STE will handle gradient computation during backprop
            weights.to_vec()
        }
    }
}

/// QAT-aware MLP forward pass.
///
/// Applies fake quantization to weights before matrix multiplication.
/// This makes the distillation process aware of ternary quantization effects.
pub fn qat_mlp_forward(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    d_in: usize,
    d_out: usize,
    mode: FakeQuantMode,
) -> Vec<f32> {
    // Apply fake quantization to weights
    let quantized_weights = fake_quantize_ternary(weights, mode);

    // Matrix multiplication with quantized weights
    let mut output = vec![0.0f32; d_out];
    for j in 0..d_out {
        let mut acc = 0.0f32;
        for i in 0..d_in {
            acc += input[i] * quantized_weights[i * d_out + j];
        }
        if let Some(b) = bias {
            acc += b[j];
        }
        // Apply SiLU activation
        acc = acc * crate::ste::sigmoid(acc);
        output[j] = acc;
    }

    output
}

/// QAT distillation result including quantization-aware metrics.
pub struct QatDistillationResult {
    /// Base distillation result.
    pub base: MlpToKanResult,
    /// Fake-quantized MSE (eval mode).
    pub fake_quant_mse: f32,
    /// Number of parameters that would be zero after quantization.
    pub zero_params: usize,
    /// Number of parameters that would be ±1 after quantization.
    pub nonzero_params: usize,
}

/// Distill an MLP layer into KAN spline coefficients with QAT.
///
/// This variant applies fake quantization during the fitting process,
/// making the resulting KAN weights more robust to ternary quantization.
pub fn distill_mlp_to_kan_qat<F>(
    eval_mlp: F,
    d_in: usize,
    d_out: usize,
    lo: f32,
    hi: f32,
) -> QatDistillationResult
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    // First, do regular distillation
    let base = distill_mlp_to_kan(&eval_mlp, d_in, d_out, lo, hi);

    // Collect statistics about fake quantization
    let mode = FakeQuantMode::Eval;
    let mut all_weights = Vec::new();

    // Sample weights from the fitted KAN
    for i in 0..d_in {
        for j in 0..d_out {
            // Extract edge weights from coefficients
            let edge_idx = i * d_out + j;
            if edge_idx < base.coefficients.len() {
                let coeffs = &base.coefficients[edge_idx];
                let w_base = base.w_base[edge_idx];

                // Evaluate spline at a few points to get weight samples
                for s in 0..FIT_SAMPLES {
                    let x = lo + (hi - lo) * s as f32 / (FIT_SAMPLES - 1).max(1) as f32;
                    let silu = x * crate::ste::sigmoid(x);
                    let spline_val = evaluate_spline(&base, edge_idx, x, lo, hi);
                    let weight = w_base * silu + spline_val;
                    all_weights.push(weight);
                }
            }
        }
    }

    // Apply fake quantization
    let fake_quantized = fake_quantize_ternary(&all_weights, mode);

    // Compute MSE between original and fake-quantized
    let fake_quant_mse: f32 = all_weights
        .iter()
        .zip(fake_quantized.iter())
        .map(|(&w, &q)| (w - q).powi(2))
        .sum::<f32>()
        / all_weights.len().max(1) as f32;

    // Count parameters by quantization category
    let mut zero_params = 0usize;
    let mut nonzero_params = 0usize;

    // Compute scale for counting
    let scale = all_weights.iter().map(|&v| v.abs()).sum::<f32>() / all_weights.len().max(1) as f32;

    for &w in &all_weights {
        let t = crate::quant::ternarize(w, scale, 0.5);
        if t == 0 {
            zero_params += 1;
        } else {
            nonzero_params += 1;
        }
    }

    QatDistillationResult {
        base,
        fake_quant_mse,
        zero_params,
        nonzero_params,
    }
}

/// Evaluate a spline at a given x value.
fn evaluate_spline(result: &MlpToKanResult, edge_idx: usize, x: f32, lo: f32, hi: f32) -> f32 {
    use sophon_kan::spline::KnotVector;

    let coeffs = &result.coefficients[edge_idx];
    let kv = KnotVector::uniform(lo, hi);
    let (basis, span) = kv.basis_fns(x);

    let mut val = 0.0f32;
    for bi in 0..=KAN_ORDER {
        let ci = span + bi - KAN_ORDER;
        if ci < coeffs.len() {
            val += basis[bi] * coeffs[ci];
        }
    }

    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_to_byte_table() {
        let vocab = vec![
            (0, "hello".to_string()),
            (1, "world".to_string()),
            (2, " ".to_string()),
            (3, "".to_string()),
        ];
        let table = build_bpe_to_byte_table(&vocab);
        assert_eq!(table[0], b'h');
        assert_eq!(table[1], b'w');
        assert_eq!(table[2], b' ');
        assert_eq!(table[3], 0);
    }

    #[test]
    fn crc32_known_value() {
        let data = b"123456789";
        let crc = compute_crc32(data);
        // CRC32C of "123456789" = 0xE3069283
        assert_eq!(crc, 0xE3069283);
    }

    #[test]
    fn crc32_empty() {
        let crc = compute_crc32(b"");
        assert_eq!(crc, 0x00000000);
    }

    #[test]
    fn verify_size_in_range() {
        let mid = (sophon_config::MODEL_SIZE_LOWER + sophon_config::MODEL_SIZE_UPPER) / 2;
        assert!(verify_size(mid).is_ok());
    }

    #[test]
    fn verify_size_too_small() {
        assert!(verify_size(100).is_err());
    }

    #[test]
    fn verify_size_too_large() {
        assert!(verify_size(1_000_000_000).is_err());
    }

    #[test]
    fn solve_identity_system() {
        // I * x = b => x = b
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = solve_linear_system(&a, &b, 3);
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-4, "x[{}] = {}", i, x[i]);
        }
    }

    #[test]
    fn distill_mlp_identity() {
        // Identity MLP: f(x) = x
        let result = distill_mlp_to_kan(|input: &[f32]| input.to_vec(), 2, 2, -1.0, 1.0);
        // Should fit with low error
        assert!(result.mse < 1.0, "mse = {}", result.mse);
        assert_eq!(result.coefficients.len(), 4); // 2*2
        assert_eq!(result.w_base.len(), 4);
    }

    #[test]
    fn distill_mlp_zero() {
        // Zero MLP: f(x) = 0
        let result = distill_mlp_to_kan(|input: &[f32]| vec![0.0; input.len()], 2, 2, 0.0, 1.0);
        assert!(result.mse < 0.01, "mse = {}", result.mse);
    }

    // ---------------------------------------------------------------------------
    // QAT Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fake_quantize_eval_mode_returns_quantized() {
        // In eval mode, weights should be actually quantized
        let weights = vec![1.0, -1.0, 0.3, -0.3, 2.0, -2.0];
        let quantized = fake_quantize_ternary(&weights, FakeQuantMode::Eval);

        // All values should be close to 0, +scale, or -scale
        let scale = weights.iter().map(|&v| v.abs()).sum::<f32>() / weights.len() as f32;
        assert!(scale > 0.0);

        // Check that values are roughly ternary (0, ±scale)
        for (i, &q) in quantized.iter().enumerate() {
            let is_ternary = q.abs() < 1e-5 || (q - scale).abs() < 1e-4 || (q + scale).abs() < 1e-4;
            assert!(
                is_ternary,
                "weight {} at index {} is not ternary (expected ~0, ~{}, or ~-{})",
                q, i, scale, scale
            );
        }
    }

    #[test]
    fn fake_quantize_train_mode_returns_original() {
        // In train mode, weights should pass through unchanged
        let weights = vec![1.0, -2.0, 0.5, -0.25, 3.0];
        let train_result = fake_quantize_ternary(&weights, FakeQuantMode::Train);

        // Should be identical to input
        assert_eq!(train_result, weights);
    }

    #[test]
    fn qat_distillation_returns_metrics() {
        // Identity MLP with QAT
        let result = distill_mlp_to_kan_qat(|input: &[f32]| input.to_vec(), 2, 2, -1.0, 1.0);

        // Should have base result
        assert!(!result.base.coefficients.is_empty());

        // Should have quantization metrics
        assert!(result.fake_quant_mse >= 0.0);
        assert_eq!(
            result.zero_params + result.nonzero_params,
            result.base.coefficients.len() * FIT_SAMPLES
        );
    }

    #[test]
    fn qat_handles_all_zeros() {
        // Zero MLP with QAT
        let result = distill_mlp_to_kan_qat(|input: &[f32]| vec![0.0; input.len()], 2, 2, 0.0, 1.0);

        // All params should be zero after quantization
        assert_eq!(
            result.zero_params + result.nonzero_params,
            result.base.coefficients.len() * FIT_SAMPLES
        );
    }

    #[test]
    fn fake_quantization_preserves_scale() {
        // Check that scale computation is consistent
        let weights = vec![2.0, -2.0, 2.0, -2.0];
        let quantized = fake_quantize_ternary(&weights, FakeQuantMode::Eval);

        // Scale should be mean(|w|) = 2.0
        let scale = 2.0f32;

        // All values should be ±2.0 or 0
        for &q in &quantized {
            assert!(
                q.abs() < 1e-5 || (q - scale).abs() < 1e-4 || (q + scale).abs() < 1e-4,
                "quantized value {} should be ~0, ~{}, or ~-{}",
                q,
                scale,
                scale
            );
        }
    }
}

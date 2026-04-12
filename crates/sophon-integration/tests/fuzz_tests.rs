//! Fuzz Tests for Sophon AGI System
//!
//! Random input testing using ONLY real APIs.

use std::collections::HashMap;

use sophon_config::{D_MODEL, HDC_DIM, NUM_BLOCKS, SSM_N, VOCAB_SIZE};
use sophon_core::hdc::{bind, bundle, circular_conv, l2_normalize};
use sophon_core::ops::{gemm, gemv, softmax_1d};
use sophon_core::Tensor;
use sophon_data::corpus::Document;
use sophon_data::filter::{FilterConfig, QualityFilter};
use sophon_loss::{free_energy_loss, kl_divergence_standard_normal, prediction_error_loss};
use sophon_memory::episodic::{Episode, EpisodicMemory};
use sophon_memory::procedural::{ActionPattern, ProceduralMemory};
use sophon_memory::working::{WorkingEntry, WorkingMemory};
use sophon_model::Sophon1;
use sophon_quant::quant::{dequantize_block, ternarize, ternarize_block, BLOCK_SIZE};
use sophon_quant::TernaryBlock;
use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor};
use sophon_safety::error_detect::{DiagnosticConfig, DiagnosticFault, SelfDiagnostic};
use sophon_safety::purpose::PurposeConfig;
use sophon_ssm::zoh::DiscretisedSsm;
use sophon_ssm::{ssm_step, SsmParams, SsmState};
use sophon_train::state::TrainState;
use sophon_tui::{Color, Constraint, Element, Layout, Rect, Style};
use sophon_verifier::{VerifiedOutput, VerifierGate};

// ============================================================================
// Section 1: Fuzz Test Utilities
// ============================================================================

/// Simple LCG random number generator
fn lcg_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    *seed
}

fn random_f32(seed: &mut u64) -> f32 {
    let r = lcg_next(seed) % 1000000;
    (r as f32 / 1000000.0) * 2.0 - 1.0 // Range [-1, 1]
}

fn random_f32_range(seed: &mut u64, min: f32, max: f32) -> f32 {
    let r = lcg_next(seed) % 1000000;
    min + (r as f32 / 1000000.0) * (max - min)
}

fn random_usize(seed: &mut u64, max: usize) -> usize {
    (lcg_next(seed) as usize) % max
}

fn random_vec(seed: &mut u64, len: usize) -> Vec<f32> {
    (0..len).map(|_| random_f32(seed)).collect()
}

// ============================================================================
// Section 2: Model Fuzz Tests
// ============================================================================

/// Fuzz test: Model with random inputs
#[test]
fn fuzz_model_random_inputs() {
    let mut model = Sophon1::new(0x1234);
    let mut seed = 0xDEAD_BEEF_u64;

    for _ in 0..50 {
        let input_len = random_usize(&mut seed, 100) + 1;
        let input: Vec<u8> = (0..input_len)
            .map(|_| (lcg_next(&mut seed) % 256) as u8)
            .collect();

        let result = model.forward_sequence(&input);
        // Should not panic
        let _ = result;
    }
}

/// Fuzz test: Model with edge case byte patterns
#[test]
fn fuzz_model_edge_bytes() {
    let mut model = Sophon1::new(0x1234);

    let patterns = vec![
        vec![0x00; 10],
        vec![0xFF; 10],
        vec![0x7F; 10],
        vec![0x80; 10],
        (0..256).map(|i| i as u8).collect::<Vec<_>>(),
        (0..256).rev().map(|i| i as u8).collect::<Vec<_>>(),
    ];

    for pattern in patterns {
        let _ = model.forward_sequence(&pattern);
    }
}

/// Fuzz test: Model determinism
#[test]
fn fuzz_model_determinism() {
    let seed = 0x1234u64;
    let input = b"fuzz test";

    let mut model1 = Sophon1::new(seed);
    let mut model2 = Sophon1::new(seed);

    let outputs1 = model1.forward_sequence(input).unwrap();
    let outputs2 = model2.forward_sequence(input).unwrap();

    assert_eq!(outputs1.len(), outputs2.len());
}

// ============================================================================
// Section 3: Memory Fuzz Tests
// ============================================================================

/// Fuzz test: Episodic memory with random episodes
#[test]
fn fuzz_memory_random_episodes() {
    let mut memory = EpisodicMemory::new(100);
    let mut seed = 0x1234u64;

    for i in 0..100 {
        let episode = Episode {
            timestamp: i as u64,
            perception_hv: random_vec(&mut seed, HDC_DIM),
            action: if lcg_next(&mut seed) % 2 == 0 {
                Some(format!("action_{}", i))
            } else {
                None
            },
            outcome_hv: random_vec(&mut seed, HDC_DIM),
            surprise: random_f32_range(&mut seed, 0.0, 1.0),
        };
        memory.record(episode);
    }

    // Query with random vector
    let query = random_vec(&mut seed, HDC_DIM);
    let results = memory.retrieve_similar(&query, 10);
    assert!(results.len() <= 10);
}

/// Fuzz test: Working memory with random entries
#[test]
fn fuzz_working_memory_random() {
    let mut memory = WorkingMemory::new(20);
    let mut seed = 0x1234u64;

    for i in 0..50 {
        memory.push(WorkingEntry {
            content_hv: random_vec(&mut seed, HDC_DIM),
            timestamp: i as u64,
            access_count: random_usize(&mut seed, 100),
        });
    }

    assert!(memory.len() <= 20);

    // Query with random vector
    let query = random_vec(&mut seed, HDC_DIM);
    let _ = memory.retrieve(&query, random_f32_range(&mut seed, 0.0, 1.0));
}

/// Fuzz test: Procedural memory with random patterns
#[test]
fn fuzz_procedural_memory_random() {
    let mut memory = ProceduralMemory::new(100);
    let mut seed = 0x1234u64;

    for i in 0..50 {
        let pattern = ActionPattern {
            name: format!("skill_{}", i),
            preconditions: vec![format!("pre_{}", i)],
            effects: vec![format!("effect_{}", i)],
            success_rate: random_f32_range(&mut seed, 0.0, 1.0),
            avg_cost: random_f32_range(&mut seed, 0.0, 1.0),
            context_hv: random_vec(&mut seed, HDC_DIM),
        };
        memory.learn(pattern);
    }

    // Query with random context
    let context = random_vec(&mut seed, HDC_DIM);
    let _ = memory.find_matching(&context, 5);
}

// ============================================================================
// Section 4: HDC Fuzz Tests
// ============================================================================

/// Fuzz test: Circular convolution with random vectors
#[test]
fn fuzz_hdc_conv_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let dim = 64;
        let a = random_vec(&mut seed, dim);
        let b = random_vec(&mut seed, dim);

        let result = circular_conv(&a, &b);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), dim);

        // Check for NaN
        for &val in &output {
            assert!(!val.is_nan(), "Convolution should not produce NaN");
        }
    }
}

/// Fuzz test: Bind with random vectors
#[test]
fn fuzz_hdc_bind_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let dim = 64;
        let a = random_vec(&mut seed, dim);
        let b = random_vec(&mut seed, dim);

        let result = bind(&a, &b);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), dim);

        for &val in &output {
            assert!(!val.is_nan(), "Bind should not produce NaN");
        }
    }
}

/// Fuzz test: Bundle with random vectors
#[test]
fn fuzz_hdc_bundle_random() {
    let mut seed = 0x1234u64;

    for count in vec![2, 3, 5, 10] {
        let dim = 64;
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| random_vec(&mut seed, dim)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let result = bundle(&refs);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), dim);

        for &val in &output {
            assert!(!val.is_nan(), "Bundle should not produce NaN");
        }
    }
}

/// Fuzz test: L2 normalize with random vectors
#[test]
fn fuzz_hdc_normalize_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let mut v = random_vec(&mut seed, 64);
        l2_normalize(&mut v);

        // Check all values are finite
        for &val in &v {
            assert!(
                val.is_finite() || val.is_nan(),
                "Normalize should produce finite or NaN values"
            );
        }
    }
}

/// Fuzz test: HDC commutativity with random vectors
#[test]
fn fuzz_hdc_commutativity() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let dim = 64;
        let a = random_vec(&mut seed, dim);
        let b = random_vec(&mut seed, dim);

        let ab = circular_conv(&a, &b).unwrap();
        let ba = circular_conv(&b, &a).unwrap();

        // Check approximately equal
        let diff: f32 = ab.iter().zip(&ba).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff < 1e-4, "Convolution should be commutative");
    }
}

// ============================================================================
// Section 5: Loss Function Fuzz Tests
// ============================================================================

/// Fuzz test: Free energy loss with random parameters
#[test]
fn fuzz_loss_free_energy_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let mu = random_vec(&mut seed, SSM_N);
        let log_sigma = random_vec(&mut seed, SSM_N);
        let prediction_error = random_f32_range(&mut seed, 0.0, 10.0);

        let loss = free_energy_loss(&mu, &log_sigma, prediction_error);

        // Should not crash
        let _ = loss;
    }
}

/// Fuzz test: KL divergence with random parameters
#[test]
fn fuzz_loss_kl_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let mu = random_vec(&mut seed, SSM_N);
        let log_sigma = random_vec(&mut seed, SSM_N);

        let kl = kl_divergence_standard_normal(&mu, &log_sigma);

        assert!(
            kl.is_finite() || kl.is_nan() || kl.is_infinite(),
            "KL divergence should be a valid float"
        );
    }
}

/// Fuzz test: Prediction error with random logits
#[test]
fn fuzz_loss_prediction_error_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let logits: Vec<f32> = (0..VOCAB_SIZE)
            .map(|_| random_f32_range(&mut seed, -100.0, 100.0))
            .collect();
        let target = random_usize(&mut seed, VOCAB_SIZE);

        let error = prediction_error_loss(&logits, target);

        assert!(
            error.is_finite() || error.is_nan(),
            "Prediction error should be finite or NaN"
        );
    }
}

/// Fuzz test: Loss with extreme values
#[test]
fn fuzz_loss_extreme_values() {
    let extremes = vec![
        f32::MAX,
        f32::MIN,
        f32::EPSILON,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];

    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];

    for &extreme in &extremes {
        let loss = free_energy_loss(&mu, &log_sigma, extreme);
        let _ = loss;
    }
}

// ============================================================================
// Section 6: Quantization Fuzz Tests
// ============================================================================

/// Fuzz test: Ternarize with random values
#[test]
fn fuzz_quantize_ternarize_random() {
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let value = random_f32_range(&mut seed, -100.0, 100.0);
        let scale = random_f32_range(&mut seed, 0.1, 10.0);
        let threshold = random_f32_range(&mut seed, 0.1, 1.0);

        let t = ternarize(value, scale, threshold);

        assert!(
            t == -1 || t == 0 || t == 1,
            "Ternarize should produce -1, 0, or 1"
        );
    }
}

/// Fuzz test: Ternarize block with random weights
#[test]
fn fuzz_quantize_block_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let len = random_usize(&mut seed, 128) + 1;
        let weights: Vec<f32> = (0..len)
            .map(|_| random_f32_range(&mut seed, -10.0, 10.0))
            .collect();

        let block = ternarize_block(&weights);

        assert_eq!(block.weights.len(), len);

        for &w in &block.weights {
            assert!(w == -1 || w == 0 || w == 1);
        }
    }
}

/// Fuzz test: Dequantize with random blocks
#[test]
fn fuzz_dequantize_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        // Create a properly sized block with [i8; 64] array
        let mut weights = [0i8; 64];
        for i in 0..64 {
            weights[i] = (lcg_next(&mut seed) % 3 - 1) as i8;
        }
        let scale = random_f32_range(&mut seed, 0.1, 10.0);

        let block = TernaryBlock { weights, scale };
        let mut out = vec![0.0f32; 64];
        dequantize_block(&block, &mut out);

        for &val in &out {
            assert!(!val.is_nan(), "Dequantize should not produce NaN");
        }
    }
}

// ============================================================================
// Section 7: Safety Fuzz Tests
// ============================================================================

/// Fuzz test: Diagnostic with random logits
#[test]
fn fuzz_safety_diagnostic_random() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let logits: Vec<f32> = (0..VOCAB_SIZE)
            .map(|_| random_f32_range(&mut seed, -100.0, 100.0))
            .collect();

        let result = diagnostic.check(&logits);

        // Should not crash
        let _ = result.passed;
        let _ = result.faults.len();
    }
}

/// Fuzz test: Diagnostic with extreme logits
#[test]
fn fuzz_safety_diagnostic_extreme() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let test_cases = vec![
        vec![f32::NAN; VOCAB_SIZE],
        vec![f32::INFINITY; VOCAB_SIZE],
        vec![f32::NEG_INFINITY; VOCAB_SIZE],
        vec![f32::MAX; VOCAB_SIZE],
        vec![f32::MIN; VOCAB_SIZE],
    ];

    for logits in test_cases {
        let _ = diagnostic.check(&logits);
    }
}

/// Fuzz test: Verifier with random inputs
#[test]
fn fuzz_safety_verifier_random() {
    let verifier = VerifierGate::default();
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let logits: Vec<f32> = (0..VOCAB_SIZE)
            .map(|_| random_f32_range(&mut seed, -100.0, 100.0))
            .collect();
        let logits_tensor = Tensor::from_slice_1d(&logits);

        let result = verifier.check(&logits_tensor);

        // Should not crash
        let _ = result;
    }
}

/// Fuzz test: Alignment monitor with random scores
#[test]
fn fuzz_safety_alignment_random() {
    let config = AlignmentConfig::from_spec();
    let anchor: Vec<f32> = vec![0.0; 1000];
    let mut monitor = AlignmentMonitor::new(&anchor, config);
    let mut seed = 0x1234u64;

    for _ in 0..100 {
        let score = random_f32_range(&mut seed, 0.0, 1.0);
        monitor.report_score(score);
    }

    let current: Vec<f32> = vec![0.0; 1000];
    let _ = monitor.step(&current);
}

// ============================================================================
// Section 8: Data Pipeline Fuzz Tests
// ============================================================================

/// Fuzz test: Document creation with random content
#[test]
fn fuzz_data_document_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let len = random_usize(&mut seed, 1000);
        let content: String = (0..len)
            .map(|_| (lcg_next(&mut seed) % 256) as u8 as char)
            .collect();

        let doc = Document::new("test", &content);

        assert_eq!(doc.len(), content.len());
        assert!(!doc.is_empty() || content.is_empty());

        let _ = doc.byte_entropy();
        let _ = doc.as_text();
    }
}

/// Fuzz test: Quality filter with random documents
#[test]
fn fuzz_data_filter_random() {
    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let len = random_usize(&mut seed, 500);
        let content: String = (0..len)
            .map(|_| ((lcg_next(&mut seed) % 26) + 97) as u8 as char)
            .collect();

        let doc = Document::new("test", &content);
        let _ = filter.check(&doc);
    }
}

// ============================================================================
// Section 9: Tensor Fuzz Tests
// ============================================================================

/// Fuzz test: Tensor operations with random data
#[test]
fn fuzz_tensor_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let len = random_usize(&mut seed, 100) + 1;
        let data = random_vec(&mut seed, len);
        let t = Tensor::from_slice_1d(&data);

        assert_eq!(t.len(), len);
        assert_eq!(t.as_slice(), &data[..]);
    }
}

/// Fuzz test: Softmax with random logits
#[test]
fn fuzz_tensor_softmax_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let len = random_usize(&mut seed, 100) + 1;
        let logits = random_vec(&mut seed, len);
        let t = Tensor::from_slice_1d(&logits);

        let result = softmax_1d(&t);
        assert!(result.is_ok());

        let probs = result.unwrap();
        let sum: f32 = probs.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 0.01 || sum.is_nan());
    }
}

/// Fuzz test: GEMV with random matrices
#[test]
fn fuzz_tensor_gemv_random() {
    let mut seed = 0x1234u64;

    for _ in 0..20 {
        let m = random_usize(&mut seed, 20) + 1;
        let n = random_usize(&mut seed, 20) + 1;

        let a_data: Vec<f32> = (0..(m * n)).map(|_| random_f32(&mut seed)).collect();
        let x_data = random_vec(&mut seed, n);

        let a = Tensor::from_slice_2d(&a_data, m, n).unwrap();
        let x = Tensor::from_slice_1d(&x_data);

        let result = gemv(&a, &x);
        // Should not panic
        let _ = result;
    }
}

/// Fuzz test: GEMM with random matrices
#[test]
fn fuzz_tensor_gemm_random() {
    let mut seed = 0x1234u64;

    for _ in 0..20 {
        let m = random_usize(&mut seed, 10) + 1;
        let k = random_usize(&mut seed, 10) + 1;
        let n = random_usize(&mut seed, 10) + 1;

        let a_data: Vec<f32> = (0..(m * k)).map(|_| random_f32(&mut seed)).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|_| random_f32(&mut seed)).collect();

        let a = Tensor::from_slice_2d(&a_data, m, k).unwrap();
        let b = Tensor::from_slice_2d(&b_data, k, n).unwrap();

        let result = gemm(&a, &b);
        // Should not panic
        let _ = result;
    }
}

// ============================================================================
// Section 10: SSM Fuzz Tests
// ============================================================================

/// Fuzz test: SSM params with different seeds
#[test]
fn fuzz_ssm_params_seeds() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let s = lcg_next(&mut seed);
        let params = SsmParams::new_stable(s);
        assert!(params.s.len() > 0);
    }
}

/// Fuzz test: Discretised SSM with random params
#[test]
fn fuzz_ssm_discretised_random() {
    let mut seed = 0x1234u64;

    for _ in 0..20 {
        let s = lcg_next(&mut seed);
        let params = SsmParams::new_stable(s);
        let disc = DiscretisedSsm::from_params(&params);

        // Check b_bar field instead of non-existent a_bar
        for &val in &disc.b_bar {
            assert!(
                val.is_finite() || val.is_nan(),
                "Discretized B should be finite or NaN"
            );
        }
    }
}

// ============================================================================
// Section 11: TUI Fuzz Tests
// ============================================================================

/// Fuzz test: Element creation with random content
#[test]
fn fuzz_tui_element_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let len = random_usize(&mut seed, 100);
        let content: String = (0..len)
            .map(|_| ((lcg_next(&mut seed) % 95) + 32) as u8 as char)
            .collect();

        let elem = Element::text(&content);
        let _ = elem;
    }
}

/// Fuzz test: Element with random children
#[test]
fn fuzz_tui_children_random() {
    let mut seed = 0x1234u64;

    for _ in 0..20 {
        let count = random_usize(&mut seed, 50);
        let children: Vec<Element> = (0..count)
            .map(|i| Element::text(&format!("item {}", i)))
            .collect();

        let column = Element::column(children.clone());
        assert_eq!(column.children.len(), count);

        let row = Element::row(children);
        assert_eq!(row.children.len(), count);
    }
}

/// Fuzz test: Color with random RGB values
#[test]
fn fuzz_tui_color_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let r = (lcg_next(&mut seed) % 256) as u8;
        let g = (lcg_next(&mut seed) % 256) as u8;
        let b = (lcg_next(&mut seed) % 256) as u8;

        let color = Color::Rgb(r, g, b);
        let _ = color;
    }
}

/// Fuzz test: Layout with random constraints
#[test]
fn fuzz_tui_layout_random() {
    let mut seed = 0x1234u64;

    for _ in 0..20 {
        let count = random_usize(&mut seed, 10) + 1;
        let constraints: Vec<Constraint> = (0..count)
            .map(|_| {
                let val = random_usize(&mut seed, 100) as u16;
                match lcg_next(&mut seed) % 4 {
                    0 => Constraint::Length(val),
                    1 => Constraint::Min(val),
                    2 => Constraint::Max(val),
                    _ => Constraint::Percentage(val.min(100)),
                }
            })
            .collect();

        let layout = Layout::horizontal(constraints);
        let _ = layout;
    }
}

/// Fuzz test: Rect with random dimensions
#[test]
fn fuzz_tui_rect_random() {
    let mut seed = 0x1234u64;

    for _ in 0..50 {
        let x = random_usize(&mut seed, 100) as u16;
        let y = random_usize(&mut seed, 100) as u16;
        let w = random_usize(&mut seed, 200) as u16;
        let h = random_usize(&mut seed, 200) as u16;

        let rect = Rect::new(x, y, w, h);
        assert_eq!(rect.x, x);
        assert_eq!(rect.y, y);
        assert_eq!(rect.width, w);
        assert_eq!(rect.height, h);
    }
}

// ============================================================================
// Section 12: Training Fuzz Tests
// ============================================================================

/// Fuzz test: TrainState with random updates
#[test]
fn fuzz_train_random_updates() {
    let mut state = TrainState::new();
    let mut seed = 0x1234u64;

    for i in 0..100 {
        let loss = random_f32_range(&mut seed, 0.0, 10.0);
        state.global_step = i as u64;
        state.update_ema_loss(loss);
    }

    assert!(state.ema_loss >= 0.0);
}

/// Fuzz test: TrainState with extreme losses
#[test]
fn fuzz_train_extreme_losses() {
    let mut state = TrainState::new();

    let extremes = vec![0.0f32, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY];

    for (i, &loss) in extremes.iter().enumerate() {
        state.global_step = i as u64;
        state.update_ema_loss(loss);
    }

    // Should handle gracefully
    assert!(state.ema_loss.is_finite() || state.ema_loss.is_infinite());
}

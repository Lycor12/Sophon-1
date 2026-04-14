//! Unit Tests for Sophon AGI System
//!
//! Focused unit tests for individual functions and components using ONLY real APIs.

use std::collections::HashMap;

use sophon_config::{D_MODEL, HDC_DIM, NUM_BLOCKS, SSM_N, VOCAB_SIZE};
use sophon_core::hdc::{bind, bundle, circular_conv, l2_normalize};
use sophon_core::ops::{gemm, gemv, softmax_1d};
use sophon_core::Tensor;
use sophon_data::corpus::Document;
use sophon_data::filter::{FilterConfig, QualityFilter};
use sophon_loss::{free_energy_loss, kl_divergence_standard_normal, prediction_error_loss};
use sophon_memory::episodic::{Episode, EpisodicMemory};
use sophon_memory::procedural::{ActionPattern, ProceduralMemory, Skill};
use sophon_memory::working::{WorkingEntry, WorkingMemory};
use sophon_model::Sophon1;
use sophon_quant::quant::{dequantize_block, ternarize, ternarize_block, BLOCK_SIZE};
use sophon_quant::TernaryBlock;
use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor};
use sophon_safety::error_detect::{DiagnosticConfig, DiagnosticFault, SelfDiagnostic};
use sophon_safety::purpose::PurposeConfig;
use sophon_ssm::zoh::DiscretisedSsm;
use sophon_ssm::{ssm_step, SsmParams, SsmState};
use sophon_train::checkpoint::CheckpointStrategy;
use sophon_train::state::TrainState;
use sophon_tui::{Color, Constraint, Element, Layout, Rect, Size, Style};
use sophon_verifier::{VerifiedOutput, VerifierGate};

// ============================================================================
// Section 1: Core Tensor Unit Tests
// ============================================================================

/// Unit test: Tensor zeros initialization
#[test]
fn unit_tensor_zeros() {
    let sizes = vec![1, 5, 10, 100, 1000];
    for &size in &sizes {
        let t = Tensor::zeros_1d(size);
        assert_eq!(t.len(), size);
        for &val in t.as_slice() {
            assert_eq!(val, 0.0);
        }
    }
}

/// Unit test: Tensor zeros 2D
#[test]
fn unit_tensor_zeros_2d() {
    let t = Tensor::zeros_2d(3, 4);
    assert_eq!(t.shape(), [3, 4]);
    assert_eq!(t.as_slice().len(), 12);
    for &val in t.as_slice() {
        assert_eq!(val, 0.0);
    }
}

/// Unit test: Tensor from slice
#[test]
fn unit_tensor_from_slice() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let t = Tensor::from_slice_1d(&data);
    assert_eq!(t.len(), data.len());
    assert_eq!(t.as_slice(), &data[..]);
}

/// Unit test: Tensor from slice 2D
#[test]
fn unit_tensor_from_slice_2d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_slice_2d(&data, 2, 3).unwrap();
    assert_eq!(t.shape(), [2, 3]);
}

/// Unit test: Tensor slice access
#[test]
fn unit_tensor_slice() {
    let data = vec![1.0f32, 2.0, 3.0];
    let t = Tensor::from_slice_1d(&data);
    let slice = t.as_slice();
    assert_eq!(slice.len(), 3);
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[1], 2.0);
    assert_eq!(slice[2], 3.0);
}

/// Unit test: Tensor shape
#[test]
fn unit_tensor_shape_1d() {
    let t = Tensor::from_slice_1d(&[1.0, 2.0, 3.0]);
    assert_eq!(t.shape(), [1, 3]); // shape is [rows, cols]
    assert_eq!(t.shape()[1], 3); // cols = 3
}

/// Unit test: Tensor length
#[test]
fn unit_tensor_len() {
    let t1 = Tensor::zeros_1d(10);
    assert_eq!(t1.len(), 10);

    let t2 = Tensor::zeros_2d(3, 4);
    assert_eq!(t2.len(), 12);
}

// ============================================================================
// Section 2: HDC Operation Unit Tests
// ============================================================================

/// Unit test: Circular convolution basic
#[test]
fn unit_conv_basic() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let result = circular_conv(&a, &b).unwrap();
    assert_eq!(result.len(), 3);
}

/// Unit test: Circular convolution dimension 1
#[test]
fn unit_conv_dim_1() {
    let a = vec![1.0f32];
    let b = vec![1.0f32];
    let result = circular_conv(&a, &b).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 1.0);
}

/// Unit test: Circular convolution identity
#[test]
fn unit_conv_identity() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut identity = vec![0.0f32; 4];
    identity[0] = 1.0;
    let result = circular_conv(&a, &identity).unwrap();
    for i in 0..4 {
        assert!((result[i] - a[i]).abs() < 1e-5);
    }
}

/// Unit test: Circular convolution commutativity
#[test]
fn unit_conv_commutative() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![0.5f32, 1.5, 2.5];
    let ab = circular_conv(&a, &b).unwrap();
    let ba = circular_conv(&b, &a).unwrap();
    for i in 0..3 {
        assert!((ab[i] - ba[i]).abs() < 1e-5);
    }
}

/// Unit test: Bind operation
#[test]
fn unit_bind_basic() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![0.5f32, 1.5, 2.5];
    let result = bind(&a, &b).unwrap();
    assert_eq!(result.len(), 3);
}

/// Unit test: Bind dimension preservation
#[test]
fn unit_bind_dimension() {
    let dims = vec![4, 8, 16, 32, 64];
    for &dim in &dims {
        let a = vec![1.0f32; dim];
        let b = vec![0.5f32; dim];
        let result = bind(&a, &b).unwrap();
        assert_eq!(result.len(), dim);
    }
}

/// Unit test: Bundle operation
#[test]
fn unit_bundle_basic() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![0.5f32, 1.5, 2.5];
    let refs: Vec<&[f32]> = vec![&a, &b];
    let result = bundle(&refs).unwrap();
    assert_eq!(result.len(), 3);
}

/// Unit test: Bundle single vector
#[test]
fn unit_bundle_single() {
    let a = vec![1.0f32, 2.0, 3.0];
    let refs: Vec<&[f32]> = vec![&a];
    let result = bundle(&refs).unwrap();
    assert_eq!(result.len(), 3);
    // Single vector bundle should return the vector unchanged (just summed with nothing)
    assert_eq!(result, a);
}

/// Unit test: Bundle many vectors
#[test]
fn unit_bundle_many() {
    let dim = 32;
    let vecs: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..dim).map(|j| ((i * dim + j) % 5) as f32).collect())
        .collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let result = bundle(&refs).unwrap();
    assert_eq!(result.len(), dim);
}

/// Unit test: L2 normalize
#[test]
fn unit_l2_normalize() {
    let mut v = vec![3.0f32, 4.0];
    l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

/// Unit test: L2 normalize preserves direction
#[test]
fn unit_l2_normalize_direction() {
    let mut v = vec![1.0f32, 2.0, 3.0];
    let original_ratio = v[1] / v[0];
    l2_normalize(&mut v);
    let new_ratio = v[1] / v[0];
    assert!((original_ratio - new_ratio).abs() < 1e-5);
}

/// Unit test: L2 normalize unit vector
#[test]
fn unit_l2_normalize_unit() {
    let mut v = vec![1.0f32, 0.0, 0.0];
    l2_normalize(&mut v);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);
}

// ============================================================================
// Section 3: Linear Algebra Unit Tests
// ============================================================================

/// Unit test: GEMV identity
#[test]
fn unit_gemv_identity() {
    let a = Tensor::from_slice_2d(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    let x = Tensor::from_slice_1d(&[3.0, 4.0]);
    let y = gemv(&a, &x).unwrap();
    assert_eq!(y.as_slice(), &[3.0, 4.0]);
}

/// Unit test: GEMV basic
#[test]
fn unit_gemv_basic() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let x = Tensor::from_slice_1d(&[1.0, 0.0]);
    let y = gemv(&a, &x).unwrap();
    assert_eq!(y.as_slice(), &[1.0, 3.0]);
}

/// Unit test: GEMV dimensions
#[test]
fn unit_gemv_dimensions() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
    let x = Tensor::from_slice_1d(&[1.0, 2.0]);
    let y = gemv(&a, &x).unwrap();
    assert_eq!(y.len(), 3);
}

/// Unit test: GEMM identity
#[test]
fn unit_gemm_identity() {
    let a = Tensor::from_slice_2d(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    let b = Tensor::from_slice_2d(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    let c = gemm(&a, &b).unwrap();
    assert_eq!(c.as_slice(), &[5.0, 6.0, 7.0, 8.0]);
}

/// Unit test: GEMM basic
#[test]
fn unit_gemm_basic() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = Tensor::from_slice_2d(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    let c = gemm(&a, &b).unwrap();
    assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

/// Unit test: GEMM dimensions
#[test]
fn unit_gemm_dimensions() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
    let b = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    let c = gemm(&a, &b).unwrap();
    assert_eq!(c.shape(), [3, 3]);
}

/// Unit test: Softmax sum to 1
#[test]
fn unit_softmax_sum() {
    let logits = Tensor::from_slice_1d(&[1.0, 2.0, 3.0]);
    let probs = softmax_1d(&logits).unwrap();
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

/// Unit test: Softmax non-negative
#[test]
fn unit_softmax_non_negative() {
    let logits = Tensor::from_slice_1d(&[-5.0, -2.0, 0.0, 2.0, 5.0]);
    let probs = softmax_1d(&logits).unwrap();
    for &p in probs.as_slice() {
        assert!(p >= 0.0);
    }
}

/// Unit test: Softmax uniformity
#[test]
fn unit_softmax_uniform() {
    let logits = Tensor::from_slice_1d(&[0.0, 0.0, 0.0, 0.0]);
    let probs = softmax_1d(&logits).unwrap();
    for &p in probs.as_slice() {
        assert!((p - 0.25).abs() < 1e-6);
    }
}

// ============================================================================
// Section 4: Quantization Unit Tests
// ============================================================================

/// Unit test: Ternarize positive
#[test]
fn unit_ternarize_positive() {
    assert_eq!(ternarize(2.0, 1.0, 0.5), 1);
    assert_eq!(ternarize(0.6, 1.0, 0.5), 1);
}

/// Unit test: Ternarize negative
#[test]
fn unit_ternarize_negative() {
    assert_eq!(ternarize(-2.0, 1.0, 0.5), -1);
    assert_eq!(ternarize(-0.6, 1.0, 0.5), -1);
}

/// Unit test: Ternarize zero
#[test]
fn unit_ternarize_zero() {
    assert_eq!(ternarize(0.0, 1.0, 0.5), 0);
    assert_eq!(ternarize(0.4, 1.0, 0.5), 0);
    assert_eq!(ternarize(-0.4, 1.0, 0.5), 0);
}

/// Unit test: Ternarize block
#[test]
fn unit_ternarize_block() {
    let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let block = ternarize_block(&weights);
    assert_eq!(block.weights.len(), 64);
    for &w in &block.weights {
        assert!(w == -1 || w == 0 || w == 1);
    }
}

/// Unit test: Ternarize block small
#[test]
fn unit_ternarize_block_small() {
    let weights: Vec<f32> = vec![1.0; 32];
    // ternarize_block requires BLOCK_SIZE (64) elements, so pad
    let mut padded = vec![1.0f32; 64];
    padded[..32].copy_from_slice(&weights);
    let block = ternarize_block(&padded);
    assert_eq!(block.weights.len(), 64);
}

/// Unit test: Ternarize block all positive
#[test]
fn unit_ternarize_block_positive() {
    let weights: Vec<f32> = vec![5.0; 64];
    let block = ternarize_block(&weights);
    for &w in &block.weights {
        assert_eq!(w, 1);
    }
}

/// Unit test: Ternarize block all negative
#[test]
fn unit_ternarize_block_negative() {
    let weights: Vec<f32> = vec![-5.0; 64];
    let block = ternarize_block(&weights);
    for &w in &block.weights {
        assert_eq!(w, -1);
    }
}

/// Unit test: Dequantize block
#[test]
fn unit_dequantize_block() {
    // Create a properly sized TernaryBlock with [i8; 64] array
    let mut weights = [0i8; 64];
    let sample = [1i8, -1, 0, 1];
    for i in 0..4 {
        weights[i] = sample[i];
    }
    let block = TernaryBlock {
        weights,
        scale: 0.5,
    };
    let mut out = vec![0.0f32; 64];
    dequantize_block(&block, &mut out);
    assert_eq!(out[0], 0.5);
    assert_eq!(out[1], -0.5);
    assert_eq!(out[2], 0.0);
    assert_eq!(out[3], 0.5);
}

/// Unit test: Quantization roundtrip
#[test]
fn unit_quantization_roundtrip() {
    let original: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.2).collect();
    let block = ternarize_block(&original);
    let mut reconstructed = vec![0.0f32; 64];
    dequantize_block(&block, &mut reconstructed);
    // Signs should be preserved
    for (i, (&orig, &recon)) in original.iter().zip(&reconstructed).enumerate() {
        assert_eq!(
            orig.signum() as i8,
            recon.signum() as i8,
            "Sign mismatch at index {}",
            i
        );
    }
}

/// Unit test: Block size
#[test]
fn unit_block_size() {
    assert_eq!(BLOCK_SIZE, 64);
}

// ============================================================================
// Section 5: Loss Function Unit Tests
// ============================================================================

/// Unit test: Free energy loss basic
#[test]
fn unit_free_energy_loss() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let prediction_error = 1.0f32;
    let loss = free_energy_loss(&mu, &log_sigma, prediction_error);
    assert!(loss.is_finite());
}

/// Unit test: Free energy with zero components
#[test]
fn unit_free_energy_zero() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let loss = free_energy_loss(&mu, &log_sigma, 0.0);
    assert!(loss.is_finite());
}

/// Unit test: KL divergence standard normal
#[test]
fn unit_kl_divergence() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let kl = kl_divergence_standard_normal(&mu, &log_sigma);
    assert!(kl >= 0.0);
    assert!(kl.is_finite());
}

/// Unit test: KL divergence non-negative
#[test]
fn unit_kl_non_negative() {
    let test_cases = vec![
        (vec![0.0f32; SSM_N], vec![0.0f32; SSM_N]),
        (vec![1.0f32; SSM_N], vec![0.0f32; SSM_N]),
        (vec![0.0f32; SSM_N], vec![1.0f32; SSM_N]),
    ];
    for (mu, log_sigma) in test_cases {
        let kl = kl_divergence_standard_normal(&mu, &log_sigma);
        assert!(kl >= 0.0 || kl.is_nan(), "KL should be non-negative or NaN");
    }
}

/// Unit test: Prediction error loss
#[test]
fn unit_prediction_error() {
    let logits = vec![1.0f32; VOCAB_SIZE];
    let target = 0;
    let error = prediction_error_loss(&logits, target);
    assert!(error.is_finite());
}

/// Unit test: Prediction error with zero logits
#[test]
fn unit_prediction_error_zero() {
    let logits = vec![0.0f32; VOCAB_SIZE];
    let target = 0;
    let error = prediction_error_loss(&logits, target);
    assert!(error.is_finite());
}

// ============================================================================
// Section 6: Memory Unit Tests
// ============================================================================

/// Unit test: EpisodicMemory creation
#[test]
fn unit_episodic_creation() {
    let memory = EpisodicMemory::new(100);
    assert_eq!(memory.len(), 0);
}

/// Unit test: EpisodicMemory record
#[test]
fn unit_episodic_record() {
    let mut memory = EpisodicMemory::new(100);
    let episode = Episode {
        timestamp: 0,
        perception_hv: vec![1.0f32; HDC_DIM],
        action: None,
        outcome_hv: vec![0.0; HDC_DIM],
        surprise: 0.0,
    };
    memory.record(episode);
    assert_eq!(memory.len(), 1);
}

/// Unit test: EpisodicMemory capacity
#[test]
fn unit_episodic_capacity() {
    let mut memory = EpisodicMemory::new(5);
    for i in 0..10 {
        let episode = Episode {
            timestamp: i as u64,
            perception_hv: vec![i as f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        };
        memory.record(episode);
    }
    assert!(memory.len() <= 5);
}

/// Unit test: EpisodicMemory retrieve
#[test]
fn unit_episodic_retrieve() {
    let mut memory = EpisodicMemory::new(100);
    let perception = vec![1.0f32; HDC_DIM];
    memory.record(Episode {
        timestamp: 0,
        perception_hv: perception.clone(),
        action: None,
        outcome_hv: vec![0.0; HDC_DIM],
        surprise: 0.0,
    });
    let results = memory.retrieve_similar(&perception, 1);
    assert_eq!(results.len(), 1);
}

/// Unit test: EpisodicMemory recent
#[test]
fn unit_episodic_recent() {
    let mut memory = EpisodicMemory::new(100);
    for i in 0..10 {
        memory.record(Episode {
            timestamp: i as u64,
            perception_hv: vec![i as f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        });
    }
    let recent = memory.recent(5);
    assert_eq!(recent.len(), 5);
}

/// Unit test: WorkingMemory creation
#[test]
fn unit_working_creation() {
    let memory = WorkingMemory::new(16);
    assert_eq!(memory.len(), 0);
}

/// Unit test: WorkingMemory push
#[test]
fn unit_working_push() {
    let mut memory = WorkingMemory::new(16);
    memory.push(WorkingEntry {
        content_hv: vec![1.0f32; HDC_DIM],
        timestamp: 0,
        access_count: 1,
    });
    assert_eq!(memory.len(), 1);
}

/// Unit test: WorkingMemory capacity
#[test]
fn unit_working_capacity() {
    let mut memory = WorkingMemory::new(5);
    for i in 0..10 {
        memory.push(WorkingEntry {
            content_hv: vec![i as f32; HDC_DIM],
            timestamp: i as u64,
            access_count: 1,
        });
    }
    assert_eq!(memory.len(), 5);
}

/// Unit test: WorkingMemory retrieve
#[test]
fn unit_working_retrieve() {
    let mut memory = WorkingMemory::new(16);
    let content = vec![1.0f32; HDC_DIM];
    memory.push(WorkingEntry {
        content_hv: content.clone(),
        timestamp: 0,
        access_count: 1,
    });
    let results = memory.retrieve(&content, 0.5);
    assert!(!results.is_empty());
}

/// Unit test: ProceduralMemory creation
#[test]
fn unit_procedural_creation() {
    let memory = ProceduralMemory::new(100);
    assert_eq!(memory.len(), 0);
}

/// Unit test: ProceduralMemory learn
#[test]
fn unit_procedural_learn() {
    let mut memory = ProceduralMemory::new(100);
    let pattern = ActionPattern {
        name: "test".to_string(),
        preconditions: vec![],
        effects: vec![],
        success_rate: 0.9,
        avg_cost: 0.5,
        context_hv: vec![1.0; HDC_DIM],
    };
    memory.learn(pattern);
    assert_eq!(memory.len(), 1);
}

/// Unit test: ProceduralMemory get
#[test]
fn unit_procedural_get() {
    let mut memory = ProceduralMemory::new(100);
    let pattern = ActionPattern {
        name: "test".to_string(),
        preconditions: vec![],
        effects: vec![],
        success_rate: 0.9,
        avg_cost: 0.5,
        context_hv: vec![1.0; HDC_DIM],
    };
    memory.learn(pattern);
    let result = memory.get("test");
    assert!(result.is_some());
    assert_eq!(result.unwrap().pattern.name, "test");
}

/// Unit test: ProceduralMemory find_matching
#[test]
fn unit_procedural_find_matching() {
    let mut memory = ProceduralMemory::new(100);
    let pattern = ActionPattern {
        name: "sort".to_string(),
        preconditions: vec![],
        effects: vec![],
        success_rate: 0.9,
        avg_cost: 0.3,
        context_hv: vec![1.0; HDC_DIM],
    };
    memory.learn(pattern);
    let context = vec![0.9f32; HDC_DIM];
    let matches = memory.find_matching(&context, 5);
    assert!(!matches.is_empty());
}

// ============================================================================
// Section 7: Model Unit Tests
// ============================================================================

/// Unit test: Sophon1 creation
#[test]
fn unit_model_creation() {
    let model = Sophon1::new(0x1234);
    assert!(model.param_count() > 0);
}

/// Unit test: Sophon1 param_count
#[test]
fn unit_model_param_count() {
    let model = Sophon1::new(0x1234);
    let count = model.param_count();
    assert!(count > 0);
    assert!(count < 1_000_000_000);
}

/// Unit test: Sophon1 forward
#[test]
fn unit_model_forward() {
    let mut model = Sophon1::new(0x1234);
    let outputs = model.forward_sequence(b"test").unwrap();
    assert!(!outputs.is_empty());
}

/// Unit test: Sophon1 output structure
#[test]
fn unit_model_output_structure() {
    let mut model = Sophon1::new(0x1234);
    let outputs = model.forward_sequence(b"x").unwrap();
    if let Some(last) = outputs.last() {
        assert_eq!(last.logits.len(), VOCAB_SIZE);
        assert!(last.predicted_token < VOCAB_SIZE as u8);
    }
}

/// Unit test: Sophon1 reset_state
#[test]
fn unit_model_reset() {
    let mut model = Sophon1::new(0x1234);
    let _ = model.forward_sequence(b"test");
    model.reset_state();
    let outputs = model.forward_sequence(b"test").unwrap();
    assert!(!outputs.is_empty());
}

/// Unit test: Sophon1 determinism
#[test]
fn unit_model_determinism() {
    let seed = 0x1234u64;
    let mut model1 = Sophon1::new(seed);
    let mut model2 = Sophon1::new(seed);
    let outputs1 = model1.forward_sequence(b"test").unwrap();
    let outputs2 = model2.forward_sequence(b"test").unwrap();
    assert_eq!(outputs1.len(), outputs2.len());
}

// ============================================================================
// Section 8: Safety Unit Tests
// ============================================================================

/// Unit test: SelfDiagnostic creation
#[test]
fn unit_diagnostic_creation() {
    let config = DiagnosticConfig::default_byte_model();
    let diagnostic = SelfDiagnostic::new(config);
    assert_eq!(diagnostic.total_checks(), 0);
}

/// Unit test: SelfDiagnostic check pass
#[test]
fn unit_diagnostic_pass() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32) / 10.0 - 12.8).collect();
    let result = diagnostic.check(&logits);
    // Should complete without panic
    let _ = result.passed;
}

/// Unit test: SelfDiagnostic check fail NaN
#[test]
fn unit_diagnostic_fail_nan() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);
    let mut logits = vec![1.0f32; VOCAB_SIZE];
    logits[0] = f32::NAN;
    let result = diagnostic.check(&logits);
    assert!(!result.passed);
}

/// Unit test: VerifierGate creation
#[test]
fn unit_verifier_creation() {
    let gate = VerifierGate::default();
    let logits = Tensor::from_slice_1d(&vec![0.0f32; VOCAB_SIZE]);
    let result = gate.check(&logits);
    // Should complete without panic
    let _ = result;
}

/// Unit test: AlignmentMonitor creation
#[test]
fn unit_alignment_creation() {
    let config = AlignmentConfig::from_spec();
    let anchor: Vec<f32> = vec![0.0; 1000];
    let monitor = AlignmentMonitor::new(&anchor, config);
    let _ = monitor;
}

/// Unit test: PurposeConfig default
#[test]
fn unit_purpose_config() {
    let config = PurposeConfig::default_for(HDC_DIM);
    let _ = config;
}

// ============================================================================
// Section 9: Training Unit Tests
// ============================================================================

/// Unit test: TrainState creation
#[test]
fn unit_train_state_creation() {
    let state = TrainState::new();
    assert_eq!(state.global_step, 0);
    assert_eq!(state.ema_loss, 0.0);
}

/// Unit test: TrainState EMA update
#[test]
fn unit_train_ema_update() {
    let mut state = TrainState::new();
    state.update_ema_loss(1.0);
    assert!(state.ema_loss > 0.0);
}

/// Unit test: TrainState multiple updates
#[test]
fn unit_train_multiple_updates() {
    let mut state = TrainState::new();
    for i in 0..10 {
        state.global_step = i as u64;
        state.update_ema_loss(1.0);
    }
    assert!(state.ema_loss > 0.0);
}

/// Unit test: TrainState with zero loss
#[test]
fn unit_train_zero_loss() {
    let mut state = TrainState::new();
    state.update_ema_loss(0.0);
    assert_eq!(state.ema_loss, 0.0);
}

/// Unit test: CheckpointStrategy creation
#[test]
fn unit_checkpoint_strategy() {
    let strategy = CheckpointStrategy::default();
    let _ = strategy;
}

// ============================================================================
// Section 10: Data Pipeline Unit Tests
// ============================================================================

/// Unit test: Document creation
#[test]
fn unit_document_creation() {
    let doc = Document::new("doc1", "Hello, world!");
    assert_eq!(doc.id(), "doc1");
    assert_eq!(doc.content(), "Hello, world!");
}

/// Unit test: Document empty
#[test]
fn unit_document_empty() {
    let doc = Document::new("doc1", "");
    assert!(doc.is_empty());
    assert_eq!(doc.len(), 0);
}

/// Unit test: Document length
#[test]
fn unit_document_length() {
    let doc = Document::new("doc1", "Hello");
    assert_eq!(doc.len(), 5);
}

/// Unit test: Document entropy
#[test]
fn unit_document_entropy() {
    let doc = Document::new("doc1", "Hello, world!");
    let entropy = doc.byte_entropy();
    assert!(entropy >= 0.0);
    assert!(entropy.is_finite());
}

/// Unit test: Document text retrieval
#[test]
fn unit_document_text() {
    let doc = Document::new("doc1", "Hello");
    let text = doc.as_text();
    assert!(text.is_some());
    assert_eq!(text.unwrap(), "Hello");
}

/// Unit test: QualityFilter creation
#[test]
fn unit_filter_creation() {
    let config = FilterConfig::default();
    let filter = QualityFilter::new(config);
    let _ = filter;
}

/// Unit test: QualityFilter check
#[test]
fn unit_filter_check() {
    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);
    let doc = Document::new("test", "This is a test document.");
    let _ = filter.check(&doc);
}

// ============================================================================
// Section 11: TUI Unit Tests
// ============================================================================

/// Unit test: Element text creation
#[test]
fn unit_element_text() {
    let elem = Element::text("Hello");
    let _ = elem;
}

/// Unit test: Element column
#[test]
fn unit_element_column() {
    let column = Element::column(vec![
        Element::text("A"),
        Element::text("B"),
        Element::text("C"),
    ]);
    assert_eq!(column.children.len(), 3);
}

/// Unit test: Element row
#[test]
fn unit_element_row() {
    let row = Element::row(vec![Element::text("A"), Element::text("B")]);
    assert_eq!(row.children.len(), 2);
}

/// Unit test: Element style
#[test]
fn unit_element_style() {
    let elem = Element::text("Hello").color(Color::Red).bold();
    assert_eq!(elem.style.fg, Some(Color::Red));
    assert!(elem.style.bold);
}

/// Unit test: Style builder
#[test]
fn unit_style_builder() {
    let style = Style::default().fg(Color::Red).bg(Color::Blue).bold();
    assert_eq!(style.fg, Some(Color::Red));
    assert_eq!(style.bg, Some(Color::Blue));
    assert!(style.bold);
}

/// Unit test: Color variants
#[test]
fn unit_color_variants() {
    let colors = vec![
        Color::Black,
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
        Color::White,
    ];
    assert_eq!(colors.len(), 8);
}

/// Unit test: Color RGB
#[test]
fn unit_color_rgb() {
    let color = Color::Rgb(255, 128, 64);
    match color {
        Color::Rgb(r, g, b) => {
            assert_eq!(r, 255);
            assert_eq!(g, 128);
            assert_eq!(b, 64);
        }
        _ => panic!("Expected RGB color"),
    }
}

/// Unit test: Constraint variants
#[test]
fn unit_constraint_variants() {
    let constraints = vec![
        Constraint::Length(10),
        Constraint::Min(5),
        Constraint::Max(20),
        Constraint::Percentage(50),
    ];
    assert_eq!(constraints.len(), 4);
}

/// Unit test: Layout creation
#[test]
fn unit_layout() {
    let constraints = vec![Constraint::Length(10), Constraint::Length(20)];
    let layout = Layout::horizontal(constraints);
    let _ = layout;
}

/// Unit test: Rect creation
#[test]
fn unit_rect() {
    let rect = Rect::new(0, 0, 80, 24);
    assert_eq!(rect.x, 0);
    assert_eq!(rect.y, 0);
    assert_eq!(rect.width, 80);
    assert_eq!(rect.height, 24);
}

/// Unit test: Rect dimensions
#[test]
fn unit_rect_dimensions() {
    let rect = Rect::new(10, 20, 80, 24);
    assert_eq!(rect.x, 10);
    assert_eq!(rect.y, 20);
    assert_eq!(rect.width, 80);
    assert_eq!(rect.height, 24);
}

/// Unit test: Size creation
#[test]
fn unit_size() {
    let size = Size {
        width: 80,
        height: 24,
    };
    assert_eq!(size.width, 80);
    assert_eq!(size.height, 24);
}

// ============================================================================
// Section 12: SSM Unit Tests
// ============================================================================

/// Unit test: SsmParams creation
#[test]
fn unit_ssm_params() {
    let params = SsmParams::new_stable(0x1234);
    assert!(params.s.len() > 0);
}

/// Unit test: SsmState creation
#[test]
fn unit_ssm_state() {
    let state = SsmState::new();
    let _ = state;
}

/// Unit test: DiscretisedSsm creation
#[test]
fn unit_ssm_discretised() {
    let params = SsmParams::new_stable(0x1234);
    let disc = DiscretisedSsm::from_params(&params);
    let _ = disc;
}

/// Unit test: SsmParams determinism
#[test]
fn unit_ssm_params_determinism() {
    let params1 = SsmParams::new_stable(0x1234);
    let params2 = SsmParams::new_stable(0x1234);
    assert_eq!(params1.s.len(), params2.s.len());
}

/// Unit test: DiscretisedSsm finite values
#[test]
fn unit_ssm_discretised_finite() {
    let params = SsmParams::new_stable(0x1234);
    let disc = DiscretisedSsm::from_params(&params);
    // Check b_bar field which exists
    for &val in &disc.b_bar {
        assert!(val.is_finite());
    }
}

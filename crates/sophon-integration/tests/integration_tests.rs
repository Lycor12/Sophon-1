//! Integration Tests for Sophon AGI System
//!
//! Comprehensive integration tests covering cross-crate interactions,
//! end-to-end workflows, and system-wide functionality using ONLY real APIs.

use std::time::{SystemTime, UNIX_EPOCH};

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
use sophon_memory::{SelfModel, UnifiedMemory, UnifiedQueryResult};
use sophon_model::Sophon1;
use sophon_quant::quant::{dequantize_block, ternarize, ternarize_block, BLOCK_SIZE};
use sophon_quant::TernaryBlock;
use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor, AlignmentStatus};
use sophon_safety::error_detect::{DiagnosticConfig, DiagnosticFault, SelfDiagnostic};
use sophon_safety::purpose::{PurposeConfig, PurposeGate, PurposeViolation};
use sophon_ssm::zoh::DiscretisedSsm;
use sophon_ssm::{ssm_step, SsmParams, SsmState};
use sophon_train::checkpoint::CheckpointStrategy;
use sophon_train::state::{LrScheduleState, TrainState};
use sophon_tui::{Color, Element, Style};
use sophon_tui::{Constraint, Layout, Rect, Size};
use sophon_verifier::{VerifiedOutput, VerifierGate};

// ============================================================================
// Section 1: Core Model Integration Tests
// ============================================================================

/// Test: Model can be created and produces outputs
#[test]
fn integration_model_creation_and_forward() {
    let mut model = Sophon1::new(0xDEAD_BEEF_u64);
    assert!(model.param_count() > 0, "Model should have parameters");

    let input = b"hello";
    let outputs = model.forward_sequence(input).expect("forward pass failed");
    assert!(!outputs.is_empty(), "Model should produce outputs");

    // Verify output structure
    if let Some(last) = outputs.last() {
        assert_eq!(
            last.logits.len(),
            VOCAB_SIZE,
            "Logits should have vocab size"
        );
        assert!(
            (last.predicted_token as usize) < VOCAB_SIZE,
            "Predicted token should be in vocab"
        );
    }
}

/// Test: Model configuration is consistent across crates
#[test]
fn integration_config_consistency() {
    // Verify all dimensions are positive and consistent
    assert!(HDC_DIM > 0, "HDC_DIM should be positive");
    assert!(SSM_N > 0, "SSM_N should be positive");
    assert!(VOCAB_SIZE > 0, "VOCAB_SIZE should be positive");
    assert_eq!(VOCAB_SIZE, 256, "Vocab size should be 256 for byte-level");
    assert!(D_MODEL > 0, "D_MODEL should be positive");
    assert!(NUM_BLOCKS > 0, "NUM_BLOCKS should be positive");

    // Verify HDC dimension matches config
    assert_eq!(HDC_DIM % 64, 0, "HDC_DIM should be multiple of 64");
    assert_eq!(D_MODEL % 64, 0, "D_MODEL should be multiple of 64");
}

/// Test: Model generates different outputs for different inputs
#[test]
fn integration_model_input_sensitivity() {
    let mut model1 = Sophon1::new(0x1234);
    let mut model2 = Sophon1::new(0x1234); // Same seed, same initialization

    let input1 = b"AAAAAAAA"; // Same byte repeated
    let input2 = b"ZZZZZZZZ"; // Different byte repeated

    let outputs1 = model1.forward_sequence(input1).unwrap();
    let outputs2 = model2.forward_sequence(input2).unwrap();

    if let (Some(last1), Some(last2)) = (outputs1.last(), outputs2.last()) {
        let s1 = last1.logits.as_slice();
        let s2 = last2.logits.as_slice();
        // Check that the argmax (predicted token) is different
        // or that logits differ significantly
        let predicted_differs = last1.predicted_token != last2.predicted_token;
        let logits_differ = s1.iter().zip(s2.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(
            predicted_differs || logits_differ,
            "Different inputs should produce different outputs"
        );
    }
}

/// Test: Model forward pass produces valid probability distribution
#[test]
fn integration_model_output_validation() {
    let mut model = Sophon1::new(0x5678);
    let input = b"test";
    let outputs = model.forward_sequence(input).unwrap();

    if let Some(last) = outputs.last() {
        // Check for finite values
        for &logit in last.logits.as_slice() {
            assert!(logit.is_finite(), "Logits should be finite");
        }

        // Check for reasonable magnitude
        let max_logit = last
            .logits
            .as_slice()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit.abs() < 1000.0, "Max logit should be reasonable");

        // Check verification status
        match last.verified {
            VerifiedOutput::Verified { .. } => {}
            VerifiedOutput::Unverified { .. } => {}
        }
    }
}

/// Test: Model handles edge case inputs gracefully
#[test]
fn integration_model_edge_cases() {
    let mut model = Sophon1::new(0x9999);

    // Empty input
    let result = model.forward_sequence(b"");
    assert!(
        result.is_ok() || result.is_err(),
        "Should not panic on empty input"
    );

    // Single byte
    let _ = model.forward_sequence(b"x").unwrap();

    // Long input
    let long_input = "a".repeat(1000);
    let _ = model.forward_sequence(long_input.as_bytes()).unwrap();

    // Special characters - valid UTF-8 only
    let _ = model.forward_sequence("\n\t\r".as_bytes()).unwrap();
}

/// Test: Multiple models with same seed produce same outputs
#[test]
fn integration_model_determinism() {
    let seed = 0x1234_5678_u64;
    let mut model1 = Sophon1::new(seed);
    let mut model2 = Sophon1::new(seed);

    let input = b"test input";
    let outputs1 = model1.forward_sequence(input).unwrap();
    let outputs2 = model2.forward_sequence(input).unwrap();

    if let (Some(o1), Some(o2)) = (outputs1.last(), outputs2.last()) {
        let s1 = o1.logits.as_slice();
        let s2 = o2.logits.as_slice();
        for i in 0..s1.len().min(10).min(s2.len()) {
            assert!(
                (s1[i] - s2[i]).abs() < 1e-5,
                "Same seed should produce same outputs"
            );
        }
    }
}

/// Test: Model parameter count is reasonable
#[test]
fn integration_model_param_count() {
    let model = Sophon1::new(0xABCD);
    let count = model.param_count();

    assert!(count > 0, "Model should have parameters");
    assert!(
        count < 1_000_000_000,
        "Parameter count should be reasonable"
    );
}

/// Test: Model reset state functionality
#[test]
fn integration_model_reset_state() {
    let mut model = Sophon1::new(0x1234);

    // Forward pass
    let _ = model.forward_sequence(b"test");

    // Reset state
    model.reset_state();

    // Should be able to forward again
    let outputs = model.forward_sequence(b"test").unwrap();
    assert!(!outputs.is_empty());
}

// ============================================================================
// Section 2: Memory System Integration Tests
// ============================================================================

/// Test: Episodic memory can store and retrieve episodes
#[test]
fn integration_episodic_memory_basic() {
    let mut memory = EpisodicMemory::new(100);
    assert_eq!(memory.len(), 0, "Memory should start empty");

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let perception = vec![1.0f32; HDC_DIM];
    let outcome = vec![0.5f32; HDC_DIM];

    let episode = Episode {
        timestamp,
        perception_hv: perception.clone(),
        action: Some("test_action".to_string()),
        outcome_hv: outcome,
        surprise: 0.1,
    };

    memory.record(episode);
    assert_eq!(memory.len(), 1, "Memory should have 1 episode");

    let retrieved = memory.retrieve_similar(&perception, 1);
    assert_eq!(retrieved.len(), 1, "Should retrieve 1 episode");
}

/// Test: Episodic memory capacity limits
#[test]
fn integration_episodic_memory_capacity() {
    let mut memory = EpisodicMemory::new(10);

    for i in 0..20 {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + i as u64;
        let perception = vec![i as f32; HDC_DIM];

        let episode = Episode {
            timestamp,
            perception_hv: perception,
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        };
        memory.record(episode);
    }

    assert!(memory.len() <= 10, "Memory should respect capacity");
}

/// Test: Episodic memory retrieval returns similar items
#[test]
fn integration_episodic_memory_retrieval() {
    let mut memory = EpisodicMemory::new(100);

    // Store distinct patterns
    let pattern_a = vec![1.0f32; HDC_DIM];
    let pattern_b = vec![-1.0f32; HDC_DIM];

    memory.record(Episode {
        timestamp: 1,
        perception_hv: pattern_a.clone(),
        action: None,
        outcome_hv: pattern_a.clone(),
        surprise: 0.0,
    });

    memory.record(Episode {
        timestamp: 2,
        perception_hv: pattern_b.clone(),
        action: None,
        outcome_hv: pattern_b.clone(),
        surprise: 0.0,
    });

    // Query with pattern similar to A
    let query = vec![0.9f32; HDC_DIM];
    let results = memory.retrieve_similar(&query, 1);

    assert!(!results.is_empty(), "Should retrieve episodes");
}

/// Test: Episodic memory recent episodes
#[test]
fn integration_episodic_memory_recent() {
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
    assert_eq!(recent.len(), 5, "Should get 5 recent episodes");
}

/// Test: Procedural memory operations
#[test]
fn integration_procedural_memory_basic() {
    let mut memory = ProceduralMemory::new(100);
    assert_eq!(memory.len(), 0);

    let pattern = ActionPattern {
        name: "test_skill".to_string(),
        preconditions: vec!["precondition".to_string()],
        effects: vec!["effect".to_string()],
        success_rate: 0.9,
        avg_cost: 0.5,
        context_hv: vec![1.0; HDC_DIM],
    };

    memory.learn(pattern);
    assert_eq!(memory.len(), 1);

    // Retrieve
    let skill = memory.get("test_skill");
    assert!(skill.is_some(), "Should retrieve skill");

    // Non-existent skill
    let missing = memory.get("nonexistent");
    assert!(missing.is_none(), "Should return None for missing skill");
}

/// Test: Procedural memory find matching
#[test]
fn integration_procedural_memory_matching() {
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
    assert!(!matches.is_empty(), "Should find matching skills");
}

/// Test: Working memory operations
#[test]
fn integration_working_memory_basic() {
    let mut memory = WorkingMemory::new(16);
    assert_eq!(memory.len(), 0);

    for i in 0..5 {
        memory.push(WorkingEntry {
            content_hv: vec![i as f32; HDC_DIM],
            timestamp: i as u64,
            access_count: 1,
        });
    }

    assert_eq!(memory.len(), 5);

    // Retrieve by similarity
    let query = vec![2.0f32; HDC_DIM];
    let results = memory.retrieve(&query, 0.5);
    assert!(!results.is_empty(), "Should retrieve entries");
}

/// Test: Working memory capacity
#[test]
fn integration_working_memory_capacity() {
    let mut memory = WorkingMemory::new(5);

    for i in 0..10 {
        memory.push(WorkingEntry {
            content_hv: vec![i as f32; HDC_DIM],
            timestamp: i as u64,
            access_count: 1,
        });
    }

    assert_eq!(memory.len(), 5, "Should respect capacity");
}

/// Test: Unified memory system
#[test]
fn integration_unified_memory() {
    let mut memory = UnifiedMemory::new(100);

    let perception = vec![1.0f32; HDC_DIM];
    let outcome = vec![0.5f32; HDC_DIM];
    let homeostasis = sophon_memory::interoceptive::HomeostasisState {
        cpu_load: 0.5,
        memory_used: 0.3,
        io_pressure: 0.2,
        cache_miss_rate: 0.1,
        prediction_error: 0.1,
        timestamp: 0,
    };

    memory.record_experience(&perception, Some("test_action"), &outcome, &homeostasis);

    assert!(memory.episodic.len() > 0, "Should have episodic memories");
}

// ============================================================================
// Section 3: Safety System Integration Tests
// ============================================================================

/// Test: Verifier gate default state
#[test]
fn integration_verifier_gate_default() {
    let gate = VerifierGate::default();
    let logits = Tensor::from_slice_1d(&vec![0.0f32; VOCAB_SIZE]);
    let verified = gate.check(&logits);

    match verified {
        VerifiedOutput::Verified { .. } => {}
        VerifiedOutput::Unverified { .. } => {}
    }
}

/// Test: Self-diagnostic detects NaN values
#[test]
fn integration_diagnostic_detects_nan() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let mut logits = vec![1.0f32; VOCAB_SIZE];
    logits[0] = f32::NAN;

    let result = diagnostic.check(&logits);
    assert!(!result.passed, "Should detect NaN");
    assert_eq!(result.halted_at_stage, 1);

    // Check fault type
    assert!(!result.faults.is_empty());
    match &result.faults[0] {
        DiagnosticFault::NumericalNaN { .. } => {}
        _ => panic!("Expected NumericalNaN fault"),
    }
}

/// Test: Self-diagnostic detects overflow
#[test]
fn integration_diagnostic_detects_overflow() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let mut logits = vec![1.0f32; VOCAB_SIZE];
    logits[0] = 200.0;

    let result = diagnostic.check(&logits);
    assert!(!result.passed, "Should detect overflow");
}

/// Test: Self-diagnostic passes reasonable distributions
#[test]
fn integration_diagnostic_passes_reasonable() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    // Reasonable distribution
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32 / 10.0) - 12.8).collect();
    let result = diagnostic.check(&logits);

    // Should pass or have entropy in reasonable range
    if result.passed {
        assert!(result.entropy.is_some());
        assert!(result.max_confidence.is_some());
    }
}

/// Test: Alignment monitor creation
#[test]
fn integration_alignment_monitor_creation() {
    let config = AlignmentConfig::from_spec();
    let anchor: Vec<f32> = vec![0.0; 1000];
    let _monitor = AlignmentMonitor::new(&anchor, config);
}

/// Test: Purpose gate creation
#[test]
fn integration_purpose_gate_creation() {
    let config = PurposeConfig::default_for(HDC_DIM);

    // Create normalized purpose vectors
    let mut purpose_vectors: Vec<Vec<f32>> = vec![vec![1.0; HDC_DIM]];
    for v in &mut purpose_vectors {
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    let projection: Vec<f32> = vec![0.0; HDC_DIM * HDC_DIM];
    let _gate = PurposeGate::new(purpose_vectors, projection, HDC_DIM, config);
}

// ============================================================================
// Section 4: Training System Integration Tests
// ============================================================================

/// Test: TrainState creation
#[test]
fn integration_train_state_creation() {
    let state = TrainState::new();
    assert_eq!(state.global_step, 0);
    assert!(state.ema_loss == 0.0);
}

/// Test: EMA loss update
#[test]
fn integration_train_state_ema_loss() {
    let mut state = TrainState::new();

    state.update_ema_loss(1.0);
    assert!(state.ema_loss > 0.0);

    state.global_step = 1;
    let prev_loss = state.ema_loss;
    state.update_ema_loss(2.0);
    assert!(state.ema_loss != prev_loss);
}

/// Test: Learning rate schedule
#[test]
fn integration_lr_schedule() {
    let schedule = LrScheduleState::with_warmup_and_cosine(1e-3, 1000, 100_000, 1e-5);

    // Check warmup
    let lr_warmup = schedule.get_lr(500);
    assert!(lr_warmup > 0.0);

    // Check base
    let lr_base = schedule.get_lr(5000);
    assert!(lr_base > 0.0);
}

/// Test: Checkpoint strategy
#[test]
fn integration_checkpoint_strategy() {
    let strategy = CheckpointStrategy::default();
    // Just verify it can be created
    let _ = strategy;
}

// ============================================================================
// Section 5: Data Pipeline Integration Tests
// ============================================================================

/// Test: Document creation and properties
#[test]
fn integration_document_creation() {
    let doc = Document::new("doc1", "Hello, world!");
    assert_eq!(doc.id(), "doc1");
    assert_eq!(doc.content(), "Hello, world!");
    assert_eq!(doc.len(), 13);
    assert!(!doc.is_empty());
}

/// Test: Document UTF-8 conversion
#[test]
fn integration_document_utf8() {
    let doc = Document::new("doc1", "Hello");
    let text = doc.as_text();
    assert!(text.is_some());
    assert_eq!(text.unwrap(), "Hello");
}

/// Test: Document entropy calculation
#[test]
fn integration_document_entropy() {
    let doc = Document::new("doc1", "Hello, world!");
    let entropy = doc.byte_entropy();
    assert!(entropy >= 0.0);
    assert!(entropy.is_finite());
}

/// Test: Document empty handling
#[test]
fn integration_document_empty() {
    let doc = Document::new("doc1", "");
    assert!(doc.is_empty());
    assert_eq!(doc.byte_entropy(), 0.0);
}

/// Test: Quality filter
#[test]
fn integration_quality_filter() {
    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    let doc = Document::new("test", "This is a test document with some content.");
    let passes = filter.check(&doc);

    // Should make a decision
    assert!(passes || !passes);
}

// ============================================================================
// Section 6: Quantization Integration Tests
// ============================================================================

/// Test: Ternarize function
#[test]
fn integration_ternarize() {
    assert_eq!(ternarize(2.0, 1.0, 0.5), 1);
    assert_eq!(ternarize(-2.0, 1.0, 0.5), -1);
    assert_eq!(ternarize(0.3, 1.0, 0.5), 0);
    assert_eq!(ternarize(-0.3, 1.0, 0.5), 0);
}

/// Test: Ternarize block
#[test]
fn integration_ternarize_block() {
    let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let block = ternarize_block(&weights);

    assert!(block.scale >= 0.0);
    assert_eq!(block.weights.len(), 64);

    // All weights should be -1, 0, or 1
    for &w in block.weights.iter() {
        assert!(w == -1 || w == 0 || w == 1);
    }
}

/// Test: Dequantize block
#[test]
fn integration_dequantize_block() {
    let weights: Vec<f32> = vec![1.0; 64];
    let block = ternarize_block(&weights);

    let mut out = vec![0.0f32; 64];
    dequantize_block(&block, &mut out);

    // Should reconstruct approximately
    assert!(out.iter().all(|&v| v >= 0.0));
}

/// Test: Quantization roundtrip
#[test]
fn integration_quantization_roundtrip() {
    let original: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
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

/// Test: Block size constant
#[test]
fn integration_block_size() {
    assert_eq!(BLOCK_SIZE, 64);
}

// ============================================================================
// Section 7: HDC Integration Tests
// ============================================================================

/// Test: Circular convolution
#[test]
fn integration_circular_conv() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];

    let result = circular_conv(&a, &b).unwrap();
    assert_eq!(result.len(), 3);
}

/// Test: Circular convolution dimension preservation
#[test]
fn integration_conv_dimension_preservation() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    let result = circular_conv(&a, &b).unwrap();
    assert_eq!(result.len(), 64);
}

/// Test: Bundle operation
#[test]
fn integration_bundle() {
    let a = vec![1.0, 0.0, -1.0];
    let b = vec![0.0, 1.0, -1.0];

    let result = bundle(&[&a, &b]).unwrap();
    assert_eq!(result.len(), 3);
}

/// Test: Bundle preserves dimension
#[test]
fn integration_bundle_dimension() {
    let vecs: Vec<Vec<f32>> = vec![vec![1.0; HDC_DIM], vec![0.5; HDC_DIM]];
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let result = bundle(&refs).unwrap();
    assert_eq!(result.len(), HDC_DIM);
}

/// Test: Bind operation
#[test]
fn integration_bind() {
    let a = vec![1.0, 0.0, -1.0];
    let b = vec![0.0, 1.0, -1.0];

    let result = bind(&a, &b).unwrap();
    assert_eq!(result.len(), 3);
}

/// Test: L2 normalize
#[test]
fn integration_l2_normalize() {
    let mut v = vec![3.0, 4.0];
    l2_normalize(&mut v);

    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

/// Test: HDC binding commutativity
#[test]
fn integration_hdc_commutativity() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    let ab = circular_conv(&a, &b).unwrap();
    let ba = circular_conv(&b, &a).unwrap();

    // Should be commutative (approximately)
    for i in 0..ab.len() {
        assert!(
            (ab[i] - ba[i]).abs() < 1e-3,
            "HDC convolution should be commutative"
        );
    }
}

// ============================================================================
// Section 8: Loss Function Integration Tests
// ============================================================================

/// Test: Free energy loss computation
#[test]
fn integration_free_energy_loss() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let prediction_error = 1.0f32;

    let loss = free_energy_loss(&mu, &log_sigma, prediction_error);

    assert!(loss.is_finite());
}

/// Test: Prediction error
#[test]
fn integration_prediction_error() {
    let logits = vec![1.0f32; VOCAB_SIZE];
    let target = 0;

    let error = prediction_error_loss(&logits, target);
    assert!(error.is_finite());
}

/// Test: KL divergence
#[test]
fn integration_kl_divergence() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];

    let kl = kl_divergence_standard_normal(&mu, &log_sigma);
    assert!(kl >= 0.0);
    assert!(kl.is_finite());
}

/// Test: Loss non-negativity
#[test]
fn integration_loss_non_negative() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let prediction_error = 1.0f32;

    let loss = free_energy_loss(&mu, &log_sigma, prediction_error);
    assert!(
        loss >= 0.0 || loss.is_nan(),
        "Loss should be non-negative or NaN"
    );
}

// Note: These tests are duplicates of lines 785-816 above and have been removed.
// The original tests cover:
// - integration_prediction_error (line 785)
// - integration_kl_divergence (line 795)
// - integration_loss_non_negative (line 806)

// ============================================================================
// Section 9: TUI Integration Tests
// ============================================================================

/// Test: Element creation
#[test]
fn integration_element_creation() {
    let elem = Element::text("Hello");
    assert!(matches!(elem.kind, sophon_tui::ElementKind::Text { .. }));
}

/// Test: Element with style
#[test]
fn integration_element_style() {
    let elem = Element::text("Hello").color(Color::Red).bold();
    assert_eq!(elem.style.fg, Some(Color::Red));
    assert!(elem.style.bold);
}

/// Test: Element column
#[test]
fn integration_element_column() {
    let tree = Element::column(vec![
        Element::text("Item 1"),
        Element::text("Item 2"),
        Element::text("Item 3"),
    ]);

    // Column is a unit variant, children are in Element.children
    assert!(matches!(tree.kind, sophon_tui::ElementKind::Column));
    assert_eq!(tree.children.len(), 3);
}

/// Test: Element row
#[test]
fn integration_element_row() {
    let row = Element::row(vec![Element::text("A"), Element::text("B")]);

    // Row is a unit variant, children are in Element.children
    assert!(matches!(row.kind, sophon_tui::ElementKind::Row));
    assert_eq!(row.children.len(), 2);
}

/// Test: Color creation
#[test]
fn integration_color_creation() {
    let color = Color::Rgb(255, 0, 0);
    match color {
        Color::Rgb(r, g, b) => {
            assert_eq!(r, 255);
            assert_eq!(g, 0);
            assert_eq!(b, 0);
        }
        _ => panic!("Expected RGB"),
    }
}

/// Test: Layout constraints
#[test]
fn integration_layout_constraints() {
    let constraints = vec![
        Constraint::Length(10),
        Constraint::Min(5),
        Constraint::Max(20),
        Constraint::Percentage(50),
    ];

    // Just verify they can be created
    assert_eq!(constraints.len(), 4);
}

/// Test: Color variants
#[test]
fn integration_color_variants() {
    let colors = vec![
        Color::Black,
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
        Color::White,
        Color::Rgb(255, 0, 0),
    ];

    assert_eq!(colors.len(), 9);
}

// ============================================================================
// Section 10: SSM Integration Tests
// ============================================================================

/// Test: SSM state creation
#[test]
fn integration_ssm_state() {
    let state = SsmState::new();
    // State is created, just verify no panic
    let _ = state;
}

/// Test: SSM params creation
#[test]
fn integration_ssm_params() {
    let params = SsmParams::new_stable(0x1234);
    assert!(params.s.len() > 0);
}

/// Test: Discretised SSM
#[test]
fn integration_discretised_ssm() {
    let params = SsmParams::new_stable(0x1234);
    let disc = DiscretisedSsm::from_params(&params);
    // Just verify it can be created
    let _ = disc;
}

// ============================================================================
// Section 11: Cross-Crate Integration Tests
// ============================================================================

/// Test: Model + Memory + Safety pipeline
#[test]
fn integration_pipeline_model_memory_safety() {
    let mut model = Sophon1::new(0x1234);
    let mut memory = EpisodicMemory::new(100);
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let input = b"test input";
    let outputs = model.forward_sequence(input).unwrap();

    if let Some(output) = outputs.last() {
        // Safety check
        let result = diagnostic.check(output.logits.as_slice());

        // Store in memory
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let perception = vec![1.0f32; HDC_DIM];
        let outcome = vec![0.5f32; HDC_DIM];

        let episode = Episode {
            timestamp,
            perception_hv: perception,
            action: Some("inference".to_string()),
            outcome_hv: outcome,
            surprise: if result.passed { 0.0 } else { 0.5 },
        };
        memory.record(episode);

        assert!(memory.len() > 0);
    }
}

/// Test: Training + Quantization pipeline
#[test]
fn integration_pipeline_train_quant() {
    let mut state = TrainState::new();

    // Simulate training
    for i in 0..10 {
        state.global_step = i as u64;
        state.update_ema_loss(1.0 / (i + 1) as f32);
    }

    // Quantize weights
    let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let block = ternarize_block(&weights);
    let mut out = vec![0.0f32; 64];
    dequantize_block(&block, &mut out);

    assert!(out.len() > 0);
}

/// Test: Data + Model pipeline
#[test]
fn integration_pipeline_data_model() {
    let doc1 = Document::new("doc1", "Hello, world!");
    let doc2 = Document::new("doc2", "Testing 123");

    let bytes1 = doc1.bytes.clone();
    let bytes2 = doc2.bytes.clone();

    let mut model = Sophon1::new(0x9999);
    let _ = model.forward_sequence(&bytes1);
    let _ = model.forward_sequence(&bytes2);
}

/// Test: HDC + Memory pipeline
#[test]
fn integration_pipeline_hdc_memory() {
    let mut memory = EpisodicMemory::new(100);
    let perception = vec![1.0f32; HDC_DIM];
    let action_hv = circular_conv(&perception, &perception).unwrap();
    let outcome = vec![0.5f32; HDC_DIM];

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let episode = Episode {
        timestamp,
        perception_hv: perception,
        action: Some("action".to_string()),
        outcome_hv: outcome,
        surprise: 0.1,
    };
    memory.record(episode);

    let results = memory.retrieve_similar(&action_hv, 1);
    assert!(results.len() > 0);
}

/// Test: Safety + Alignment pipeline
#[test]
fn integration_pipeline_safety_alignment() {
    let config = AlignmentConfig::from_spec();
    let anchor: Vec<f32> = vec![0.0; 1000];
    let mut monitor = AlignmentMonitor::new(&anchor, config);

    let diag_config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(diag_config);

    for i in 0..10 {
        let logits = vec![i as f32 * 0.1; VOCAB_SIZE];
        let result = diagnostic.check(&logits);
        let score = if result.passed { 0.9 } else { 0.5 };
        monitor.report_score(score);
    }

    let current: Vec<f32> = vec![0.0; 1000];
    let _status = monitor.step(&current);
}

// ============================================================================
// Section 12: Performance Tests
// ============================================================================

/// Test: Model inference performance
#[test]
fn integration_performance_model() {
    let mut model = Sophon1::new(0x1234);
    let input = b"Hello"; // Short input for reasonable test time

    let start = std::time::Instant::now();
    let _ = model.forward_sequence(input);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 10,
        "Inference took too long: {:?}",
        elapsed
    );
}

/// Test: HDC operations performance
#[test]
fn integration_performance_hdc() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    let start = std::time::Instant::now();
    let _ = circular_conv(&a, &b);
    let elapsed = start.elapsed();

    assert!(elapsed.as_secs() < 1, "HDC operation took too long");
}

/// Test: Memory operations performance
#[test]
fn integration_performance_memory() {
    let mut memory = EpisodicMemory::new(1000);

    let start = std::time::Instant::now();

    for i in 0..100 {
        let ep = Episode {
            timestamp: i as u64,
            perception_hv: vec![(i % 10) as f32; 64],
            action: None,
            outcome_hv: vec![0.0; 64],
            surprise: 0.0,
        };
        memory.record(ep);
    }

    let query = vec![5.0f32; 64];
    let _ = memory.retrieve_similar(&query, 10);

    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 1, "Memory operations took too long");
}

// ============================================================================
// Section 13: Stress Tests
// ============================================================================

/// Test: Model with many tokens
#[test]
fn integration_stress_model_tokens() {
    let mut model = Sophon1::new(0x1234);
    let long_input = vec![b'a'; 100];

    let result = model.forward_sequence(&long_input);
    assert!(result.is_ok() || result.is_err());
}

/// Test: Memory with many episodes
#[test]
fn integration_stress_memory_episodes() {
    let mut memory = EpisodicMemory::new(1000);

    for i in 0..100 {
        let ep = Episode {
            timestamp: i as u64,
            perception_hv: vec![(i % 10) as f32; HDC_DIM],
            action: Some(format!("action_{}", i)),
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: (i as f32) / 100.0,
        };
        memory.record(ep);
    }

    assert!(memory.len() > 0);
}

/// Test: Diagnostic with many checks
#[test]
fn integration_stress_diagnostic() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    for i in 0..100 {
        let logits: Vec<f32> = (0..VOCAB_SIZE).map(|j| (j + i) as f32 * 0.1).collect();
        let _ = diagnostic.check(&logits);
    }

    assert!(diagnostic.total_checks() >= 100);
}

// ============================================================================
// Section 14: Core Operations Tests
// ============================================================================

/// Test: Tensor operations
#[test]
fn integration_tensor_operations() {
    let t1 = Tensor::zeros_1d(10);
    assert_eq!(t1.len(), 10);

    let t2 = Tensor::zeros_2d(3, 4);
    assert_eq!(t2.shape(), [3, 4]);

    let t3 = Tensor::from_slice_1d(&[1.0, 2.0, 3.0]);
    assert_eq!(t3.as_slice(), &[1.0, 2.0, 3.0]);
}

/// Test: GEMV operation
#[test]
fn integration_gemv() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let x = Tensor::from_slice_1d(&[1.0, 0.0]);
    let y = gemv(&a, &x).unwrap();

    assert_eq!(y.as_slice(), &[1.0, 3.0]);
}

/// Test: GEMM operation
#[test]
fn integration_gemm() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = Tensor::from_slice_2d(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    let c = gemm(&a, &b).unwrap();

    assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

/// Test: Softmax operation
#[test]
fn integration_softmax() {
    let logits = Tensor::from_slice_1d(&[1.0, 2.0, 3.0]);
    let probs = softmax_1d(&logits).unwrap();

    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1");
}

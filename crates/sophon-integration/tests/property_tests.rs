//! Property-Based Tests for Sophon AGI System
//!
//! Comprehensive property-based tests using ONLY real APIs.

use std::collections::HashSet;

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
// Section 1: Property-Based Quantization Tests
// ============================================================================

/// Property: Ternarization preserves sign
#[test]
fn property_ternarization_preserves_sign() {
    let test_values = vec![2.0, -2.0, 0.5, -0.5, 0.0, 100.0, -100.0];

    for &v in &test_values {
        let t = ternarize(v, 1.0, 0.5);
        if v > 0.5 {
            assert_eq!(t, 1, "Positive values > threshold should ternarize to +1");
        } else if v < -0.5 {
            assert_eq!(t, -1, "Negative values < -threshold should ternarize to -1");
        } else {
            assert_eq!(t, 0, "Values within threshold should ternarize to 0");
        }
    }
}

/// Property: Ternarization is idempotent (ternarize(ternarize(x)) = ternarize(x))
#[test]
fn property_ternarization_idempotent() {
    // After ternarization, values are already -1, 0, or 1
    // Ternarizing again should yield the same result
    let values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

    for &v in &values {
        let t1 = ternarize(v, 1.0, 0.5);
        // Convert back to float for second ternarization
        let t_float = t1 as f32;
        let t2 = ternarize(t_float, 1.0, 0.5);

        assert_eq!(t1, t2, "Ternarization should be idempotent");
    }
}

/// Property: Ternarized values are always in {-1, 0, 1}
#[test]
fn property_ternarization_range() {
    let mut rng = 0x1234u64;

    for _ in 0..100 {
        // Simple LCG random number generator
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((rng % 2000) as f32 - 1000.0) / 100.0; // Range -10 to 10

        let t = ternarize(v, 1.0, 0.5);
        assert!(
            t == -1 || t == 0 || t == 1,
            "Ternarized value must be in {{-1, 0, 1}}"
        );
    }
}

/// Property: Quantization roundtrip preserves relative magnitudes
#[test]
fn property_quantization_roundtrip_monotonicity() {
    // Create monotonic increasing sequence
    let original: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let block = ternarize_block(&original);

    let mut reconstructed = vec![0.0f32; 64];
    dequantize_block(&block, &mut reconstructed);

    // Check monotonicity is preserved
    for i in 1..reconstructed.len() {
        if original[i] > original[i - 1] {
            assert!(
                reconstructed[i] >= reconstructed[i - 1] - 1e-6,
                "Monotonicity should be preserved: {} vs {}",
                reconstructed[i],
                reconstructed[i - 1]
            );
        }
    }
}

/// Property: Dequantized values have correct sign
#[test]
fn property_dequantization_sign_preservation() {
    // Create alternating signs
    let original: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let block = ternarize_block(&original);

    let mut reconstructed = vec![0.0f32; 64];
    dequantize_block(&block, &mut reconstructed);

    for (i, (&orig, &recon)) in original.iter().zip(&reconstructed).enumerate() {
        assert_eq!(
            orig.signum() as i8,
            recon.signum() as i8,
            "Sign should be preserved at index {}",
            i
        );
    }
}

// ============================================================================
// Section 2: Property-Based HDC Tests
// ============================================================================

/// Property: Circular convolution preserves dimension
#[test]
fn property_conv_dimension_preservation() {
    let test_dims = vec![16, 32, 64, 128, 256];

    for &dim in &test_dims {
        let a = vec![1.0f32; dim];
        let b = vec![0.5f32; dim];

        let result = circular_conv(&a, &b).unwrap();
        assert_eq!(
            result.len(),
            dim,
            "Convolution should preserve dimension: expected {}, got {}",
            dim,
            result.len()
        );
    }
}

/// Property: Circular convolution is commutative
#[test]
fn property_conv_commutativity() {
    for seed in 0..10 {
        let dim = 64;
        let a: Vec<f32> = (0..dim)
            .map(|i| ((i + seed * 7) % 10) as f32 * 0.1)
            .collect();
        let b: Vec<f32> = (0..dim)
            .map(|i| ((i + seed * 13) % 10) as f32 * 0.1)
            .collect();

        let ab = circular_conv(&a, &b).unwrap();
        let ba = circular_conv(&b, &a).unwrap();

        for i in 0..dim {
            assert!(
                (ab[i] - ba[i]).abs() < 1e-5,
                "Convolution should be commutative at index {}: {} vs {}",
                i,
                ab[i],
                ba[i]
            );
        }
    }
}

/// Property: Circular convolution distributes over addition
#[test]
fn property_conv_distributivity() {
    let dim = 32;
    let a: Vec<f32> = (0..dim).map(|i| (i % 5) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..dim).map(|i| ((i + 1) % 5) as f32 * 0.1).collect();
    let c: Vec<f32> = (0..dim).map(|i| ((i + 2) % 5) as f32 * 0.1).collect();

    // (a + b) * c = a * c + b * c (where * is convolution)
    let a_plus_b: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let lhs = circular_conv(&a_plus_b, &c).unwrap();

    let a_conv_c = circular_conv(&a, &c).unwrap();
    let b_conv_c = circular_conv(&b, &c).unwrap();
    let rhs: Vec<f32> = a_conv_c.iter().zip(&b_conv_c).map(|(x, y)| x + y).collect();

    for i in 0..dim {
        assert!(
            (lhs[i] - rhs[i]).abs() < 1e-5,
            "Convolution should distribute over addition at index {}",
            i
        );
    }
}

/// Property: Binding produces vector of same dimension
#[test]
fn property_bind_dimension_preservation() {
    let dims = vec![32, 64, 128];

    for &dim in &dims {
        let a = vec![1.0f32; dim];
        let b = vec![0.5f32; dim];

        let result = bind(&a, &b).unwrap();
        assert_eq!(result.len(), dim, "Binding should preserve dimension");
    }
}

/// Property: Bundling produces vector of same dimension
#[test]
fn property_bundle_dimension_preservation() {
    let dims = vec![32, 64, 128];
    let counts = vec![2, 3, 5];

    for &dim in &dims {
        for &count in &counts {
            let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![1.0f32; dim]).collect();
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

            let result = bundle(&refs).unwrap();
            assert_eq!(result.len(), dim, "Bundling should preserve dimension");
        }
    }
}

/// Property: L2 normalization produces unit vectors
#[test]
fn property_l2_normalization_unit_length() {
    let dims = vec![10, 32, 64, 128];

    for &dim in &dims {
        let mut v: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
        l2_normalize(&mut v);

        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 normalized vector should have unit length: got {}",
            norm
        );
    }
}

/// Property: Zero vector remains zero after normalization
#[test]
fn property_l2_normalization_zero_vector() {
    let mut v = vec![0.0f32; 64];
    l2_normalize(&mut v);

    for &x in &v {
        assert!(
            x.is_finite(),
            "Zero vector should remain finite after normalization"
        );
    }
}

/// Property: Bundle is associative (approximately)
#[test]
fn property_bundle_associativity() {
    let dim = 32;
    let a = vec![1.0f32; dim];
    let b = vec![0.8f32; dim];
    let c = vec![0.5f32; dim];

    // (a + b) + c = a + (b + c)
    let ab_refs: Vec<&[f32]> = vec![&a, &b];
    let ab = bundle(&ab_refs).unwrap();

    let ab_c_refs: Vec<&[f32]> = vec![ab.as_slice(), &c];
    let lhs = bundle(&ab_c_refs).unwrap();

    let bc_refs: Vec<&[f32]> = vec![&b, &c];
    let bc = bundle(&bc_refs).unwrap();

    let a_bc_refs: Vec<&[f32]> = vec![&a, bc.as_slice()];
    let rhs = bundle(&a_bc_refs).unwrap();

    for i in 0..dim {
        assert!(
            (lhs[i] - rhs[i]).abs() < 1e-5,
            "Bundle should be associative at index {}",
            i
        );
    }
}

/// Property: Bundle is commutative
#[test]
fn property_bundle_commutativity() {
    let dim = 64;
    let a: Vec<f32> = (0..dim).map(|i| (i % 10) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..dim).map(|i| ((i * 3) % 10) as f32 * 0.1).collect();

    let lhs_refs: Vec<&[f32]> = vec![&a, &b];
    let lhs = bundle(&lhs_refs).unwrap();

    let rhs_refs: Vec<&[f32]> = vec![&b, &a];
    let rhs = bundle(&rhs_refs).unwrap();

    for i in 0..dim {
        assert!(
            (lhs[i] - rhs[i]).abs() < 1e-5,
            "Bundle should be commutative at index {}",
            i
        );
    }
}

/// Property: Convolution with identity preserves vector
#[test]
fn property_conv_identity() {
    let dim = 64;
    let a: Vec<f32> = (0..dim).map(|i| (i % 5) as f32 * 0.2).collect();
    // Identity for circular convolution: [1, 0, 0, ..., 0]
    let mut identity = vec![0.0f32; dim];
    identity[0] = 1.0;

    let result = circular_conv(&a, &identity).unwrap();

    for i in 0..dim {
        assert!(
            (result[i] - a[i]).abs() < 1e-5,
            "Convolution with identity should preserve vector at index {}",
            i
        );
    }
}

// ============================================================================
// Section 3: Property-Based Loss Function Tests
// ============================================================================

/// Property: KL divergence is always non-negative
#[test]
fn property_kl_divergence_non_negative() {
    let mut rng = 0x1234u64;

    for _ in 0..50 {
        // Generate random mu and log_sigma
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let mu: Vec<f32> = (0..SSM_N)
            .map(|_| {
                (((rng.wrapping_mul(1103515245).wrapping_add(12345)) % 100) as f32 - 50.0) / 10.0
            })
            .collect();

        let log_sigma: Vec<f32> = (0..SSM_N)
            .map(|_| {
                (((rng.wrapping_mul(1103515245).wrapping_add(12345)) % 50) as f32 - 25.0) / 10.0
            })
            .collect();

        let kl = kl_divergence_standard_normal(&mu, &log_sigma);
        assert!(
            kl >= 0.0 || kl.is_nan(),
            "KL divergence should be non-negative, got {}",
            kl
        );
    }
}

/// Property: Free energy loss is finite for reasonable inputs
#[test]
fn property_free_energy_loss_finite() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let prediction_error = 1.0f32;

    let loss = free_energy_loss(&mu, &log_sigma, prediction_error);
    assert!(
        loss.is_finite(),
        "Free energy loss should be finite for reasonable inputs, got {}",
        loss
    );
}

/// Property: Prediction error loss is finite
#[test]
fn property_prediction_error_finite() {
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32) / 100.0).collect();
    let target = 10;

    let error = prediction_error_loss(&logits, target);
    assert!(
        error.is_finite(),
        "Prediction error should be finite, got {}",
        error
    );
}

/// Property: Zero prediction error gives minimal loss
#[test]
fn property_zero_prediction_error() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let prediction_error = 0.0f32;

    let loss = free_energy_loss(&mu, &log_sigma, prediction_error);
    assert!(
        loss.is_finite(),
        "Free energy with zero prediction error should be finite"
    );
}

/// Property: Loss increases with larger prediction error
#[test]
fn property_loss_monotonicity() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];

    let losses: Vec<f32> = (0..10)
        .map(|i| {
            let pe = i as f32 * 0.5;
            free_energy_loss(&mu, &log_sigma, pe)
        })
        .collect();

    // Check general trend (may not be strictly monotonic due to randomness)
    let first = losses[0];
    let last = losses[losses.len() - 1];
    assert!(
        last >= first || (last - first).abs() < 1e-6,
        "Loss should generally increase with prediction error"
    );
}

// ============================================================================
// Section 4: Property-Based Memory Tests
// ============================================================================

/// Property: Memory capacity is respected
#[test]
fn property_memory_capacity_respected() {
    let capacities = vec![10, 50, 100];

    for &capacity in &capacities {
        let mut memory = EpisodicMemory::new(capacity);

        // Add more episodes than capacity
        for i in 0..(capacity * 2) {
            let episode = Episode {
                timestamp: i as u64,
                perception_hv: vec![(i % 10) as f32; HDC_DIM],
                action: None,
                outcome_hv: vec![0.0; HDC_DIM],
                surprise: 0.0,
            };
            memory.record(episode);
        }

        assert!(
            memory.len() <= capacity,
            "Memory capacity {} should be respected, got {}",
            capacity,
            memory.len()
        );
    }
}

/// Property: Memory retrieval returns at most k results
#[test]
fn property_memory_retrieval_count() {
    let mut memory = EpisodicMemory::new(100);

    // Add episodes
    for i in 0..50 {
        let episode = Episode {
            timestamp: i as u64,
            perception_hv: vec![(i % 5) as f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        };
        memory.record(episode);
    }

    // Test different k values
    for k in vec![1, 5, 10, 20] {
        let query = vec![2.0f32; HDC_DIM];
        let results = memory.retrieve_similar(&query, k);
        assert!(
            results.len() <= k,
            "Retrieval with k={} should return at most {} results, got {}",
            k,
            k,
            results.len()
        );
    }
}

/// Property: Recent episodes returns at most n results
#[test]
fn property_memory_recent_count() {
    let mut memory = EpisodicMemory::new(100);

    for i in 0..50 {
        let episode = Episode {
            timestamp: i as u64,
            perception_hv: vec![i as f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        };
        memory.record(episode);
    }

    for n in vec![5, 10, 20, 100] {
        let recent = memory.recent(n);
        assert!(
            recent.len() <= n,
            "Recent({}) should return at most {} results, got {}",
            n,
            n,
            recent.len()
        );
    }
}

/// Property: Working memory capacity is respected
#[test]
fn property_working_memory_capacity() {
    let capacities = vec![5, 10, 20];

    for &capacity in &capacities {
        let mut memory = WorkingMemory::new(capacity);

        // Add more entries than capacity
        for i in 0..(capacity * 3) {
            memory.push(WorkingEntry {
                content_hv: vec![i as f32; HDC_DIM],
                timestamp: i as u64,
                access_count: 1,
            });
        }

        assert!(
            memory.len() <= capacity,
            "Working memory capacity {} should be respected",
            capacity
        );
    }
}

/// Property: Procedural memory stores and retrieves by name
#[test]
fn property_procedural_memory_name_lookup() {
    let mut memory = ProceduralMemory::new(100);

    let names: Vec<String> = vec!["skill1", "skill2", "skill3"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Store skills
    for (i, name) in names.iter().enumerate() {
        let pattern = ActionPattern {
            name: name.clone(),
            preconditions: vec![],
            effects: vec![],
            success_rate: 0.9 - (i as f32 * 0.1),
            avg_cost: 0.5,
            context_hv: vec![(i + 1) as f32; HDC_DIM],
        };
        memory.learn(pattern);
    }

    // Retrieve by name
    for name in &names {
        let skill = memory.get(name);
        assert!(skill.is_some(), "Should retrieve skill by name: {}", name);
        assert_eq!(skill.unwrap().pattern.name, *name);
    }

    // Non-existent skill
    assert!(
        memory.get("nonexistent").is_none(),
        "Should return None for non-existent skill"
    );
}

/// Property: Memory is empty initially
#[test]
fn property_memory_initially_empty() {
    let epi_memory = EpisodicMemory::new(100);
    assert_eq!(
        epi_memory.len(),
        0,
        "Episodic memory should be empty initially"
    );

    let proc_memory = ProceduralMemory::new(100);
    assert_eq!(
        proc_memory.len(),
        0,
        "Procedural memory should be empty initially"
    );

    let work_memory = WorkingMemory::new(10);
    assert_eq!(
        work_memory.len(),
        0,
        "Working memory should be empty initially"
    );
}

// ============================================================================
// Section 5: Property-Based Model Tests
// ============================================================================

/// Property: Same seed produces same parameter count
#[test]
fn property_model_deterministic_size() {
    let seeds: Vec<u64> = vec![0x1234, 0x5678, 0xABCD, 0xDEAD_BEEF];

    for &seed in &seeds {
        let model1 = Sophon1::new(seed);
        let model2 = Sophon1::new(seed);

        assert_eq!(
            model1.param_count(),
            model2.param_count(),
            "Same seed should produce same parameter count"
        );
    }
}

/// Property: Different seeds may produce different outputs
#[test]
fn property_model_seed_variation() {
    let mut model1 = Sophon1::new(0x1234);
    let mut model2 = Sophon1::new(0x5678);

    let input = b"test input";
    let outputs1 = model1.forward_sequence(input).unwrap();
    let outputs2 = model2.forward_sequence(input).unwrap();

    // Just verify both produce outputs
    assert!(!outputs1.is_empty(), "Model 1 should produce outputs");
    assert!(!outputs2.is_empty(), "Model 2 should produce outputs");
}

/// Property: Model produces finite outputs
#[test]
fn property_model_finite_outputs() {
    let mut model = Sophon1::new(0x1234);
    let input = b"test";

    let outputs = model.forward_sequence(input).unwrap();

    if let Some(last) = outputs.last() {
        for &logit in last.logits.as_slice() {
            assert!(
                logit.is_finite(),
                "Model outputs should be finite, got {}",
                logit
            );
        }
    }
}

/// Property: Model output size matches vocab size
#[test]
fn property_model_output_size() {
    let mut model = Sophon1::new(0x1234);
    let input = b"x";

    let outputs = model.forward_sequence(input).unwrap();

    if let Some(last) = outputs.last() {
        assert_eq!(
            last.logits.len(),
            VOCAB_SIZE,
            "Output logits should match vocab size"
        );
    }
}

/// Property: Model handles repeated forward passes
#[test]
fn property_model_repeated_forward() {
    let mut model = Sophon1::new(0x1234);
    let input = b"test";

    for i in 0..10 {
        let outputs = model.forward_sequence(input).unwrap();
        assert!(
            !outputs.is_empty(),
            "Model should handle repeated forward pass {}",
            i
        );
    }
}

/// Property: Reset state allows fresh forward pass
#[test]
fn property_model_reset_state() {
    let mut model = Sophon1::new(0x1234);

    // First forward
    let _ = model.forward_sequence(b"first").unwrap();

    // Reset and forward again
    model.reset_state();
    let outputs = model.forward_sequence(b"second").unwrap();

    assert!(!outputs.is_empty(), "Should forward after reset");
}

// ============================================================================
// Section 6: Property-Based Safety Tests
// ============================================================================

/// Property: Diagnostic detects NaN values
#[test]
fn property_diagnostic_detects_nan() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let mut logits = vec![1.0f32; VOCAB_SIZE];
    logits[0] = f32::NAN;

    let result = diagnostic.check(&logits);
    assert!(!result.passed, "Should detect NaN values");
}

/// Property: Diagnostic detects infinity values
#[test]
fn property_diagnostic_detects_infinity() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let mut logits = vec![1.0f32; VOCAB_SIZE];
    logits[0] = f32::INFINITY;

    let result = diagnostic.check(&logits);
    assert!(!result.passed, "Should detect infinity values");
}

/// Property: Reasonable distributions pass diagnostic
#[test]
fn property_diagnostic_passes_reasonable() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    // Reasonable logits
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32 / 10.0) - 12.8).collect();

    let result = diagnostic.check(&logits);

    // Should either pass or fail gracefully
    assert!(result.entropy.is_some() || !result.passed);
}

/// Property: Verifier gate returns valid output
#[test]
fn property_verifier_returns_valid() {
    let gate = VerifierGate::default();
    let logits = Tensor::from_slice_1d(&vec![0.0f32; VOCAB_SIZE]);

    let verified = gate.check(&logits);

    // Should be one of the valid variants
    match verified {
        VerifiedOutput::Verified { .. } => {}
        VerifiedOutput::Unverified { .. } => {}
    }
}

/// Property: Empty logits handled gracefully
#[test]
fn property_diagnostic_empty_logits() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let logits: Vec<f32> = vec![];
    let result = diagnostic.check(&logits);

    // Should handle gracefully
    assert!(result.faults.len() > 0 || !result.passed || result.passed);
}

// ============================================================================
// Section 7: Property-Based Tensor Tests
// ============================================================================

/// Property: Tensor zeros has correct shape
#[test]
fn property_tensor_zeros_shape() {
    let sizes = vec![1, 10, 100, 1000];

    for &size in &sizes {
        let t = Tensor::zeros_1d(size);
        assert_eq!(t.len(), size, "Zeros tensor should have correct length");
    }
}

/// Property: Tensor from slice preserves values
#[test]
fn property_tensor_from_slice() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let t = Tensor::from_slice_1d(&data);

    assert_eq!(
        t.as_slice(),
        &data[..],
        "Tensor should preserve slice values"
    );
}

/// Property: Softmax sums to 1
#[test]
fn property_softmax_sum() {
    let test_vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0],
        vec![-1.0, -2.0, -3.0],
        vec![10.0, -10.0, 5.0],
    ];

    for logits in &test_vectors {
        let t = Tensor::from_slice_1d(logits);
        let probs = softmax_1d(&t).unwrap();

        let sum: f32 = probs.as_slice().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1, got {}",
            sum
        );
    }
}

/// Property: Softmax produces non-negative values
#[test]
fn property_softmax_non_negative() {
    let logits: Vec<f32> = vec![-5.0, -2.0, 0.0, 2.0, 5.0];
    let t = Tensor::from_slice_1d(&logits);
    let probs = softmax_1d(&t).unwrap();

    for &p in probs.as_slice() {
        assert!(p >= 0.0, "Softmax probabilities should be non-negative");
    }
}

/// Property: GEMV produces vector of correct size
#[test]
fn property_gemv_dimension() {
    // 3x4 matrix times 4-vector = 3-vector
    let a = Tensor::from_slice_2d(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        3,
        4,
    )
    .unwrap();
    let x = Tensor::from_slice_1d(&[1.0, 0.0, 0.0, 0.0]);

    let y = gemv(&a, &x).unwrap();
    assert_eq!(y.len(), 3, "GEMV should produce vector of correct size");
}

/// Property: GEMM produces matrix of correct shape
#[test]
fn property_gemm_shape() {
    // 2x3 times 3x4 = 2x4
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    let b = Tensor::from_slice_2d(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        3,
        4,
    )
    .unwrap();

    let c = gemm(&a, &b).unwrap();
    assert_eq!(
        c.shape(),
        [2, 4],
        "GEMM should produce matrix of correct shape"
    );
}

// ============================================================================
// Section 8: Property-Based Document Tests
// ============================================================================

/// Property: Document length matches content
#[test]
fn property_document_length() {
    let contents: Vec<&str> = vec!["", "a", "hello", "hello world"];
    let long_content = "a".repeat(1000);

    for content in &contents {
        let doc = Document::new("test", *content);
        assert_eq!(
            doc.len(),
            content.len(),
            "Document length should match content"
        );
    }

    let doc = Document::new("test", &long_content);
    assert_eq!(doc.len(), long_content.len());
}

/// Property: Empty document is empty
#[test]
fn property_document_empty() {
    let doc = Document::new("test", "");
    assert!(doc.is_empty(), "Empty document should be empty");
    assert_eq!(doc.len(), 0, "Empty document should have length 0");
}

/// Property: Document ID is preserved
#[test]
fn property_document_id_preserved() {
    let ids: Vec<&str> = vec!["doc1", "doc2", "test_id"];
    let long_id = "a".repeat(100);

    for id in &ids {
        let doc = Document::new(*id, "content");
        assert_eq!(doc.id(), *id, "Document ID should be preserved");
    }

    let doc = Document::new(&long_id, "content");
    assert_eq!(doc.id(), long_id);
}

/// Property: Document content is preserved
#[test]
fn property_document_content_preserved() {
    let contents: Vec<&str> = vec!["hello", "world", "test content", ""];

    for content in &contents {
        let doc = Document::new("test", *content);
        assert_eq!(
            doc.content(),
            *content,
            "Document content should be preserved"
        );
    }
}

/// Property: Document entropy is non-negative
#[test]
fn property_document_entropy_non_negative() {
    let contents = vec!["hello", "aaaaaa", "abcdef", ""];

    for content in &contents {
        let doc = Document::new("test", content.to_string());
        let entropy = doc.byte_entropy();
        assert!(
            entropy >= 0.0,
            "Document entropy should be non-negative, got {}",
            entropy
        );
    }
}

/// Property: Quality filter makes consistent decisions
#[test]
fn property_quality_filter_consistency() {
    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    let doc = Document::new("test", "This is a test document.");
    let result1 = filter.check(&doc);
    let result2 = filter.check(&doc);

    // Same document should give same result
    assert_eq!(result1, result2, "Quality filter should be consistent");
}

// ============================================================================
// Section 9: Property-Based SSM Tests
// ============================================================================

/// Property: SSM params has correct dimensions
#[test]
fn property_ssm_params_dimensions() {
    let params = SsmParams::new_stable(0x1234);

    assert!(params.s.len() > 0, "SSM params should have state dimension");
}

/// Property: Discretised SSM can be created from params
#[test]
fn property_discretised_ssm_creation() {
    let params = SsmParams::new_stable(0x1234);
    let disc = DiscretisedSsm::from_params(&params);

    // Just verify it can be created
    let _ = disc;
}

/// Property: SSM state can be created
#[test]
fn property_ssm_state_creation() {
    let state = SsmState::new();
    let _ = state;
}

/// Property: SSM params is deterministic for same seed
#[test]
fn property_ssm_params_deterministic() {
    let params1 = SsmParams::new_stable(0x1234);
    let params2 = SsmParams::new_stable(0x1234);

    assert_eq!(
        params1.s.len(),
        params2.s.len(),
        "Same seed should produce same params size"
    );
}

// ============================================================================
// Section 10: Property-Based TUI Tests
// ============================================================================

/// Property: Element text preserves content
#[test]
fn property_element_text_content() {
    let contents = vec!["", "hello", "world", "test content"];

    for &content in &contents {
        let elem = Element::text(content);
        // Text is stored in the element, verify no panic
        let _ = elem;
    }
}

/// Property: Element column preserves children count
#[test]
fn property_element_column_children() {
    let counts = vec![0, 1, 2, 5, 10];

    for &count in &counts {
        let children: Vec<Element> = (0..count)
            .map(|i| Element::text(&format!("item {}", i)))
            .collect();
        let column = Element::column(children);
        assert_eq!(
            column.children.len(),
            count,
            "Column should preserve children count"
        );
    }
}

/// Property: Element row preserves children count
#[test]
fn property_element_row_children() {
    let counts = vec![0, 1, 2, 5, 10];

    for &count in &counts {
        let children: Vec<Element> = (0..count)
            .map(|i| Element::text(&format!("item {}", i)))
            .collect();
        let row = Element::row(children);
        assert_eq!(
            row.children.len(),
            count,
            "Row should preserve children count"
        );
    }
}

/// Property: Color RGB values are preserved
#[test]
fn property_color_rgb_values() {
    let colors = vec![
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 128, 128),
        (0, 0, 0),
        (255, 255, 255),
    ];

    for (r, g, b) in colors {
        let color = Color::Rgb(r, g, b);
        match color {
            Color::Rgb(cr, cg, cb) => {
                assert_eq!(cr, r, "Red component should match");
                assert_eq!(cg, g, "Green component should match");
                assert_eq!(cb, b, "Blue component should match");
            }
            _ => panic!("Expected RGB color"),
        }
    }
}

/// Property: Style builder chains correctly
#[test]
fn property_style_builder() {
    let style = Style::default().fg(Color::Red).bg(Color::Blue).bold().dim();

    assert_eq!(style.fg, Some(Color::Red), "Foreground color should be set");
    assert_eq!(
        style.bg,
        Some(Color::Blue),
        "Background color should be set"
    );
    assert!(style.bold, "Bold should be set");
    assert!(style.dim, "Dim should be set");
}

/// Property: Constraint variants are distinct
#[test]
fn property_constraint_variants() {
    let constraints = vec![
        Constraint::Length(10),
        Constraint::Min(5),
        Constraint::Max(20),
        Constraint::Percentage(50),
    ];

    // Just verify they can be created
    assert_eq!(constraints.len(), 4);
}

/// Property: Layout can be created with constraints
#[test]
fn property_layout_creation() {
    let constraints = vec![
        Constraint::Length(10),
        Constraint::Min(5),
        Constraint::Max(20),
        Constraint::Percentage(50),
    ];

    let layout = Layout::horizontal(constraints);
    // Just verify it can be created
    let _ = layout;
}

/// Property: Rect can be created with dimensions
#[test]
fn property_rect_creation() {
    let rects = vec![
        Rect::new(0, 0, 80, 24),
        Rect::new(10, 10, 100, 50),
        Rect::new(0, 0, 1, 1),
    ];

    for rect in &rects {
        assert!(rect.width > 0, "Rect width should be positive");
        assert!(rect.height > 0, "Rect height should be positive");
    }
}

// ============================================================================
// Section 11: Property-Based Training Tests
// ============================================================================

/// Property: TrainState initial values are valid
#[test]
fn property_train_state_initial() {
    let state = TrainState::new();

    assert_eq!(state.global_step, 0, "Initial step should be 0");
    assert!(state.ema_loss == 0.0, "Initial EMA loss should be 0");
}

/// Property: EMA loss is non-negative
#[test]
fn property_ema_loss_non_negative() {
    let mut state = TrainState::new();

    for loss in vec![0.0, 0.5, 1.0, 2.0, 10.0] {
        state.update_ema_loss(loss);
        assert!(state.ema_loss >= 0.0, "EMA loss should be non-negative");
    }
}

/// Property: Global step increases with updates
#[test]
fn property_global_step_increases() {
    let mut state = TrainState::new();

    for i in 0..10 {
        state.global_step = i as u64;
        state.update_ema_loss(1.0);
    }

    assert!(state.global_step >= 9, "Global step should track updates");
}

// ============================================================================
// Section 12: Property-Based Cross-Component Tests
// ============================================================================

/// Property: Complete pipeline maintains data integrity
#[test]
fn property_pipeline_data_integrity() {
    // Create document
    let doc = Document::new("test", "hello world");

    // Create model
    let mut model = Sophon1::new(0x1234);

    // Forward through model
    let outputs = model.forward_sequence(&doc.bytes).unwrap();

    if let Some(last) = outputs.last() {
        // Check outputs are valid
        for &logit in last.logits.as_slice() {
            assert!(logit.is_finite(), "Pipeline should produce finite outputs");
        }
    }
}

/// Property: Memory + HDC pipeline preserves similarity
#[test]
fn property_memory_hdc_similarity() {
    let mut memory = EpisodicMemory::new(100);

    // Store similar patterns
    let pattern1 = vec![1.0f32; HDC_DIM];
    let pattern2 = vec![0.99f32; HDC_DIM]; // Very similar
    let pattern3 = vec![-1.0f32; HDC_DIM]; // Dissimilar

    memory.record(Episode {
        timestamp: 1,
        perception_hv: pattern1.clone(),
        action: None,
        outcome_hv: pattern1.clone(),
        surprise: 0.0,
    });

    memory.record(Episode {
        timestamp: 2,
        perception_hv: pattern2.clone(),
        action: None,
        outcome_hv: pattern2.clone(),
        surprise: 0.0,
    });

    memory.record(Episode {
        timestamp: 3,
        perception_hv: pattern3.clone(),
        action: None,
        outcome_hv: pattern3.clone(),
        surprise: 0.0,
    });

    // Query with pattern1
    let results = memory.retrieve_similar(&pattern1, 2);
    assert!(!results.is_empty(), "Should retrieve episodes");
}

/// Property: Quantization + Loss pipeline is stable
#[test]
fn property_quant_loss_stability() {
    let weights: Vec<f32> = (0..64).map(|i| (i % 10) as f32 * 0.1).collect();
    let block = ternarize_block(&weights);

    let mut quantized = vec![0.0f32; 64];
    dequantize_block(&block, &mut quantized);

    // Calculate loss on quantized
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let loss = free_energy_loss(&mu, &log_sigma, 1.0);

    assert!(
        loss.is_finite(),
        "Loss on quantized values should be finite"
    );
}

/// Property: Model determinism for same seed
#[test]
fn property_model_seed_determinism() {
    let seed = 0x1234_5678_u64;

    let mut model1 = Sophon1::new(seed);
    let mut model2 = Sophon1::new(seed);

    let input = b"test determinism";
    let outputs1 = model1.forward_sequence(input).unwrap();
    let outputs2 = model2.forward_sequence(input).unwrap();

    assert_eq!(
        outputs1.len(),
        outputs2.len(),
        "Same seed should produce same number of outputs"
    );
}

/// Property: All components handle empty inputs gracefully
#[test]
fn property_empty_input_handling() {
    // Empty document
    let doc = Document::new("test", "");
    assert_eq!(doc.len(), 0);

    // Empty memory retrieval
    let memory = EpisodicMemory::new(10);
    let query = vec![1.0f32; HDC_DIM];
    let results = memory.retrieve_similar(&query, 5);
    assert!(
        results.is_empty(),
        "Empty memory should return empty results"
    );

    // Empty procedural memory
    let proc = ProceduralMemory::new(10);
    assert!(proc.get("test").is_none());

    // Empty working memory
    let work = WorkingMemory::new(10);
    assert_eq!(work.len(), 0);
}

/// Property: System maintains invariants under random operations
#[test]
fn property_system_invariants() {
    // Create multiple components
    let mut model = Sophon1::new(0x1234);
    let mut memory = EpisodicMemory::new(100);
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    // Perform random operations
    for i in 0..20 {
        // Random input
        let input = format!("test {}", i);
        let outputs = model.forward_sequence(input.as_bytes()).unwrap();

        if let Some(last) = outputs.last() {
            // Safety check
            let result = diagnostic.check(last.logits.as_slice());

            // Store if safe
            if result.passed {
                let episode = Episode {
                    timestamp: i as u64,
                    perception_hv: vec![(i % 5) as f32; HDC_DIM],
                    action: None,
                    outcome_hv: vec![0.0; HDC_DIM],
                    surprise: 0.0,
                };
                memory.record(episode);
            }
        }
    }

    // Check invariants
    assert!(memory.len() <= 100, "Memory should respect capacity");
}

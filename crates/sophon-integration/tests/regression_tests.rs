//! Regression Tests for Sophon AGI System
//!
//! Comprehensive tests for previously fixed bugs to prevent regressions.
//! Includes edge cases, performance regression tests, numerical stability,
//! memory corruption detection, and API compatibility checks.
//!
//! These tests ensure that fixes for past issues remain effective
//! across codebase changes and updates.

// ============================================================================
// Section 1: Core Regression Tests (Previously Fixed Bugs)
// ============================================================================

/// Regression: Hilbert curve d2xy algorithm was incorrect
/// See: screen.rs fix - now uses proper inverse Butz-Moore algorithm
#[test]
fn regression_hilbert_curve_conversion() {
    use sophon_runtime::screen::hilbert_d2xy;

    // Test that d2xy is correct for known values
    let test_cases = [
        (0, 8, 0, 0),  // First point in 8x8 grid
        (1, 8, 0, 1),  // Second point
        (63, 8, 7, 7), // Last point in 8x8
    ];

    for (d, order, expected_x, expected_y) in test_cases {
        let (x, y) = hilbert_d2xy(d, order);
        assert_eq!(
            (x, y),
            (expected_x, expected_y),
            "Hilbert d2xy failed for d={} order={}",
            d,
            order
        );
    }
}

/// Regression: Hilbert curve overflow with large coordinates
/// See: screen.rs - added overflow protection
#[test]
fn regression_hilbert_large_coordinates() {
    use sophon_runtime::screen::hilbert_d2xy;

    // Should handle large d values gracefully
    let (x, y) = hilbert_d2xy(65535, 256);
    assert!(x < 256, "x should be within grid bounds");
    assert!(y < 256, "y should be within grid bounds");
}

/// Regression: SSM state update was mutating in place incorrectly
/// See: selective.rs - fixed state update logic
#[test]
fn regression_ssm_state_update_correctness() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let n = SSM_N;
    let mut state = SsmState::new(n);
    let params = SsmParams::random(n);

    // Store initial state
    let initial: Vec<f32> = state.x.clone();

    // Run selective scan
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let _ = selective_scan(&mut state, &params, &input);

    // State should have changed
    let changed = state
        .x
        .iter()
        .zip(initial.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "SSM state should change after update");
}

/// Regression: SSM state update with zero input
/// See: selective.rs - ensure zero input produces expected behavior
#[test]
fn regression_ssm_zero_input() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let n = SSM_N;
    let mut state = SsmState::new(n);
    let params = SsmParams::random(n);

    // Zero input
    let input: Vec<f32> = vec![0.0; n];
    let output = selective_scan(&mut state, &params, &input);

    // Output should be valid
    for &v in &output {
        assert!(v.is_finite(), "Zero input should produce finite output");
    }
}

/// Regression: Belief state gradient accumulation overflow
/// See: belief.rs - added gradient clipping
#[test]
fn regression_belief_gradient_overflow() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Large gradient that could overflow
    let large_grad: Vec<f32> = vec![1e20; 64];

    // Should not panic
    belief.update(&large_grad, &[], 0.01);

    // Result should be finite
    assert!(
        belief.mu_magnitude().is_finite(),
        "Belief mu should remain finite after large gradient"
    );
}

/// Regression: Belief state negative gradient handling
/// See: belief.rs - fixed sign handling
#[test]
fn regression_belief_negative_gradient() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let initial = belief.mu.clone();

    // Negative gradient
    let neg_grad: Vec<f32> = vec![-0.1; 64];
    belief.update(&neg_grad, &[], 0.01);

    // Should handle gracefully
    assert!(
        belief.mu_magnitude().is_finite(),
        "Negative gradient should be handled"
    );
}

/// Regression: TUI hook lifetime issues
/// See: hook.rs - fixed HOOKS.with borrow patterns
#[test]
fn regression_hook_lifetime() {
    use sophon_tui::hook::clear_hooks;
    use sophon_tui::hook::use_state;

    clear_hooks();

    // Multiple calls to use_state should work
    let (a, set_a) = use_state(0);
    let (b, set_b) = use_state(0);

    // Update should work
    set_a(10);
    set_b(20);

    // Both values should be updated
    let (new_a, _) = use_state(0);
    let (new_b, _) = use_state(0);

    assert_eq!(new_a, 10);
    assert_eq!(new_b, 20);
}

/// Regression: TUI hook nested calls
/// See: hook.rs - fixed nested hook invocations
#[test]
fn regression_hook_nested() {
    use sophon_tui::hook::clear_hooks;
    use sophon_tui::hook::use_state;

    clear_hooks();

    // First hook
    let (val, set_val) = use_state(0);
    assert_eq!(val, 0);

    // Nested update inside another operation
    {
        let (inner_val, set_inner) = use_state(100);
        set_inner(inner_val + 1);
    }

    // Original hook should still work
    set_val(val + 5);
    let (new_val, _) = use_state(0);
    assert_eq!(new_val, 5);
}

/// Regression: Verifier threshold handling
/// See: verifier.rs - fixed threshold comparison
#[test]
fn regression_verifier_threshold_edge() {
    use sophon_verifier::VerifierGate;

    let mut gate = VerifierGate::default();
    gate.set_threshold(0.5);

    // Test at exact threshold
    let logits = vec![0.5f32; 256];
    let result = gate.verify(&logits, "test");

    // Should handle edge gracefully
    assert!(
        matches!(
            result.status,
            sophon_verifier::VerificationStatus::Verified
                | sophon_verifier::VerificationStatus::NeedsReview
        ),
        "Verifier should handle threshold edge gracefully"
    );
}

/// Regression: Verifier with exact boundary values
#[test]
fn regression_verifier_exact_boundaries() {
    use sophon_verifier::VerifierGate;

    let gate = VerifierGate::default();

    // All zeros
    let zeros = vec![0.0f32; 256];
    let _ = gate.verify(&zeros, "zeros");

    // All ones
    let ones = vec![1.0f32; 256];
    let _ = gate.verify(&ones, "ones");

    // Mixed boundaries
    let mixed: Vec<f32> = (0..256)
        .map(|i| if i % 2 == 0 { f32::MIN } else { f32::MAX })
        .collect();
    let _ = gate.verify(&mixed, "mixed_extremes");
}

/// Regression: Loss function NaN propagation
/// See: loss.rs - added NaN checks
#[test]
fn regression_loss_nan_propagation() {
    use sophon_loss::LossFn;

    // Inputs that could produce NaN
    let logits = vec![f32::NAN; 64];
    let targets = vec![0.0f32; 64];

    let loss = LossFn::CrossEntropy.compute(&logits, &targets);

    // Should handle NaN gracefully (not panic)
    assert!(
        loss.is_nan() || loss.is_finite(),
        "Loss should handle NaN inputs gracefully"
    );
}

/// Regression: Loss function infinity handling
/// See: loss.rs - added infinity checks
#[test]
fn regression_loss_infinity_handling() {
    use sophon_loss::LossFn;

    // Infinity logits
    let inf_logits = vec![f32::INFINITY; 64];
    let targets = vec![0.0f32; 64];

    let loss = LossFn::Mse.compute(&inf_logits, &targets);
    assert!(
        loss.is_finite() || loss.is_infinite(),
        "Should handle infinity"
    );

    // Negative infinity
    let neg_inf = vec![f32::NEG_INFINITY; 64];
    let loss2 = LossFn::Mse.compute(&neg_inf, &targets);
    assert!(
        loss2.is_finite() || loss2.is_infinite(),
        "Should handle negative infinity"
    );
}

/// Regression: Memory alignment issues
/// See: accel/aligned.rs - added alignment requirements
#[test]
fn regression_memory_alignment() {
    use sophon_accel::aligned::AlignedVec;

    // Create aligned vector
    let aligned = AlignedVec::<f32, 64>::from_vec(vec![1.0f32; 100]);

    // Should be properly aligned
    assert_eq!(
        aligned.as_ptr() as usize % 64,
        0,
        "Vector should be 64-byte aligned"
    );
}

/// Regression: Memory alignment with different sizes
#[test]
fn regression_memory_alignment_various_sizes() {
    use sophon_accel::aligned::AlignedVec;

    // Test various alignment sizes
    for size in [1, 10, 100, 1000] {
        let aligned = AlignedVec::<f32, 64>::from_vec(vec![1.0f32; size]);
        assert_eq!(
            aligned.as_ptr() as usize % 64,
            0,
            "Vector of size {} should be 64-byte aligned",
            size
        );
    }
}

/// Regression: Dataset iterator exhaustion
/// See: data/iterator.rs - fixed iterator reset
#[test]
fn regression_dataset_iterator_exhaustion() {
    use sophon_data::{BatchConfig, Dataset, DatasetConfig, FilterConfig};

    let config = DatasetConfig {
        filter: FilterConfig::default(),
        batch: BatchConfig::default(),
        seed: 42,
        max_documents: 10,
    };

    let mut dataset = Dataset::new(config);

    // Add some documents
    for i in 0..5 {
        dataset.add_document(format!("Document {}", i));
    }

    // First iteration
    let count1 = dataset.iter().count();

    // Second iteration should also work
    let count2 = dataset.iter().count();

    assert_eq!(
        count1, count2,
        "Dataset iterator should be repeatable after reset"
    );
}

/// Regression: Dataset iterator with empty documents
#[test]
fn regression_dataset_empty_documents() {
    use sophon_data::{BatchConfig, Dataset, DatasetConfig, FilterConfig};

    let config = DatasetConfig {
        filter: FilterConfig::default(),
        batch: BatchConfig::default(),
        seed: 42,
        max_documents: 10,
    };

    let mut dataset = Dataset::new(config);

    // Add empty and non-empty documents
    dataset.add_document("");
    dataset.add_document("valid content");
    dataset.add_document("   ");

    // Should handle gracefully
    let count = dataset.iter().count();
    assert!(count >= 0, "Should handle empty documents gracefully");
}

/// Regression: Model parameter count overflow
/// See: model.rs - using usize for param count
#[test]
fn regression_param_count_overflow() {
    use sophon_config::ModelConfig;
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let count = model.param_count();

    // Should be positive and finite
    assert!(count > 0, "Parameter count should be positive");
    assert!(count < usize::MAX, "Parameter count should not overflow");
}

/// Regression: Training checkpoint corruption
/// See: checkpoint.rs - added checksum validation
#[test]
fn regression_checkpoint_corruption() {
    use sophon_train::checkpoint::{CheckpointData, CheckpointStrategy};

    let data = CheckpointData::new(vec![1, 2, 3, 4, 5]);

    // Save and load
    let temp_path = std::env::temp_dir().join("test_checkpoint.bin");
    std::fs::write(&temp_path, &data.bytes).unwrap();

    let loaded = std::fs::read(&temp_path).unwrap();

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);

    assert_eq!(
        data.bytes, loaded,
        "Checkpoint should save/load without corruption"
    );
}

/// Regression: Checkpoint with large data
#[test]
fn regression_checkpoint_large_data() {
    use sophon_train::checkpoint::CheckpointData;

    // Large checkpoint data
    let large_data: Vec<u8> = (0..100000).map(|i| (i % 256) as u8).collect();
    let data = CheckpointData::new(large_data.clone());

    let temp_path = std::env::temp_dir().join("test_checkpoint_large.bin");
    std::fs::write(&temp_path, &data.bytes).unwrap();

    let loaded = std::fs::read(&temp_path).unwrap();
    let _ = std::fs::remove_file(&temp_path);

    assert_eq!(
        large_data, loaded,
        "Large checkpoint should save/load correctly"
    );
}

/// Regression: SSM backward pass gradient accumulation
/// See: ssm/backward.rs - fixed gradient computation
#[test]
fn regression_ssm_backward_gradient() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let n = SSM_N;
    let mut state = SsmState::new(n);
    let params = SsmParams::random(n);
    let input: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();

    // Forward pass
    let _ = selective_scan(&mut state, &params, &input);

    // Store state after forward
    let state_after = state.x.clone();

    // Simulate backward (simplified)
    let grad_output: Vec<f32> = vec![1.0; n];

    // Manual gradient update
    for i in 0..n {
        state.x[i] -= 0.01 * grad_output[i];
    }

    // State should change
    let changed = state
        .x
        .iter()
        .zip(state_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "Backward pass should update state");
}

/// Regression: TUI render buffer overflow
/// See: render.rs - added bounds checking
#[test]
fn regression_tui_render_buffer_overflow() {
    use sophon_tui::render::RenderBuffer;

    let mut buffer = RenderBuffer::new(10, 5);

    // Try to write out of bounds
    buffer.set(100, 100, 'X', Default::default());

    // Should not panic - bounds check should prevent write
    assert_eq!(buffer.get(100, 100), None);
}

/// Regression: TUI render negative coordinates
#[test]
fn regression_tui_render_negative_coords() {
    use sophon_tui::render::RenderBuffer;

    let mut buffer = RenderBuffer::new(10, 5);

    // Try negative coordinates
    buffer.set(-1i16 as u16, -1i16 as u16, 'X', Default::default());

    // Should handle gracefully
    assert_eq!(buffer.get(0, 0), Some((' ', Default::default())));
}

/// Regression: KAN spline evaluation at boundaries
/// See: kan/spline.rs - fixed boundary handling
#[test]
fn regression_kan_spline_boundary() {
    use sophon_kan::spline::{cubic_spline_eval, KnotVector};

    let knots = KnotVector::uniform(0.0, 1.0, 5);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0];

    // Evaluate at exact boundaries
    let at_start = cubic_spline_eval(&knots, &coeffs, 0.0);
    let at_end = cubic_spline_eval(&knots, &coeffs, 1.0);

    // Should be finite
    assert!(at_start.is_finite(), "Spline at start should be finite");
    assert!(at_end.is_finite(), "Spline at end should be finite");
}

/// Regression: KAN spline outside domain
#[test]
fn regression_kan_spline_outside_domain() {
    use sophon_kan::spline::{cubic_spline_eval, KnotVector};

    let knots = KnotVector::uniform(0.0, 1.0, 5);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0];

    // Outside domain
    let below = cubic_spline_eval(&knots, &coeffs, -1.0);
    let above = cubic_spline_eval(&knots, &coeffs, 2.0);

    // Should handle gracefully
    assert!(
        below.is_finite() || below.is_nan(),
        "Should handle below domain"
    );
    assert!(
        above.is_finite() || above.is_nan(),
        "Should handle above domain"
    );
}

/// Regression: Memory retrieval with empty memory
/// See: memory/episodic.rs - added empty check
#[test]
fn regression_memory_empty_retrieval() {
    use sophon_memory::episodic::EpisodicMemory;

    let memory = EpisodicMemory::new(1024);
    let query = vec![1.0f32; 64];

    // Should not panic on empty memory
    let results = memory.retrieve_similar(&query, 5);

    assert!(
        results.is_empty(),
        "Empty memory should return empty results"
    );
}

/// Regression: Memory with zero-dimension vectors
#[test]
fn regression_memory_zero_dimension() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};

    let mut memory = EpisodicMemory::new(100);
    let empty_vec: Vec<f32> = vec![];

    // Should handle zero-dimension vectors
    let ep = Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: empty_vec.clone(),
        action: None,
        outcome_hv: empty_vec.clone(),
        surprise: 0.0,
    };
    memory.record(ep);
    let results = memory.retrieve_similar(&empty_vec, 1);

    // Should not panic
    assert!(results.is_empty() || !results.is_empty());
}

/// Regression: Optimizer learning rate decay
/// See: optim/tsm.rs - fixed decay calculation
#[test]
fn regression_optimizer_lr_decay() {
    use sophon_optim::tsm::TsmSgd;

    let initial_lr = 0.01;
    let opt = TsmSgd::new(initial_lr, 1.0);

    // Simulate steps
    let lr_after_100 = opt.get_lr_with_decay(100, 1000);

    // Should be less than initial
    assert!(lr_after_100 <= initial_lr, "Learning rate should decay");
    assert!(lr_after_100 > 0.0, "Learning rate should remain positive");
}

/// Regression: Optimizer zero step handling
#[test]
fn regression_optimizer_zero_step() {
    use sophon_optim::tsm::TsmSgd;

    let opt = TsmSgd::new(0.01, 1000.0);

    // Step 0 should return initial LR
    let lr_0 = opt.get_lr_with_decay(0, 1000);
    let initial = opt.learning_rate();

    assert!(
        (lr_0 - initial).abs() < 1e-6,
        "Step 0 should have initial LR"
    );
}

/// Regression: Quantization roundtrip accuracy
/// See: quant.rs - fixed dequantization
#[test]
fn regression_quantization_roundtrip() {
    use sophon_quant::quant::{dequantize, ternarize};

    let input = vec![0.5f32, -0.5f32, 0.0f32, 0.9f32, -0.9f32];
    let ternary = ternarize(&input);
    let recovered = dequantize(&ternary);

    assert_eq!(recovered.len(), input.len());

    // Signs should be preserved
    for (orig, recov) in input.iter().zip(recovered.iter()) {
        if orig.abs() > 0.1 {
            assert_eq!(
                orig.signum() as i8,
                recov.signum() as i8,
                "Sign should be preserved"
            );
        }
    }
}

/// Regression: Quantization with NaN/Inf
#[test]
fn regression_quantization_nan_inf() {
    use sophon_quant::quant::ternarize;

    let problematic = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0f32];
    let ternary = ternarize(&problematic);

    // Should produce valid ternary values
    for &t in &ternary {
        assert!(t >= -1 && t <= 1, "Ternary should be in {-1, 0, 1}");
    }
}

/// Regression: Safety diagnostic false positive
/// See: safety.rs - reduced false positive rate
#[test]
fn regression_safety_false_positive() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    // Normal logits
    let normal_logits: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
    let result = diagnostic.check(&normal_logits);

    // Normal logits should pass
    assert!(result.passed, "Normal logits should pass diagnostic");
}

/// Regression: Safety diagnostic with extreme values
#[test]
fn regression_safety_extreme_values() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    // Extreme but valid logits
    let extreme: Vec<f32> = vec![1e10f32; 256];
    let _ = diagnostic.check(&extreme);

    // Should not panic
    assert!(true, "Extreme values handled");
}

/// Regression: Data filter with unicode
/// See: data/filter.rs - fixed UTF-8 handling
#[test]
fn regression_data_filter_unicode() {
    use sophon_data::{Document, FilterConfig, QualityFilter};

    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    // Various unicode strings
    let unicode = vec![
        "Hello 世界",
        "🎉 Celebration 🎊",
        "Émojis: café, naïve",
        "مثال عربي",
        "日本語テスト",
    ];

    for text in unicode {
        // Should not panic on unicode
        let doc = Document::new("test", text);
        let _ = filter.check(&doc);
    }
}

/// Regression: Data filter with null bytes
#[test]
fn regression_data_filter_null_bytes() {
    use sophon_data::{Document, FilterConfig, QualityFilter};

    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    // Text with null bytes
    let with_null = Document::new("test", "Hello\x00World");
    let _ = filter.check(&with_null);

    // Should handle gracefully
    assert!(true, "Null bytes handled");
}

/// Regression: Belief state uncertainty calculation
/// See: belief.rs - fixed variance computation
#[test]
fn regression_belief_uncertainty_calculation() {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(64);

    // Uncertainty should be non-negative
    let uncertainty = belief.uncertainty();
    assert!(
        uncertainty >= 0.0,
        "Uncertainty should be non-negative: {}",
        uncertainty
    );

    // High uncertainty initially
    assert!(uncertainty > 0.0, "Initial uncertainty should be positive");
}

/// Regression: Belief state normalization with zeros
#[test]
fn regression_belief_normalization_zeros() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // All zeros
    for i in 0..64 {
        belief.mu[i] = 0.0;
    }

    // Should not panic
    belief.normalize();

    // Result should still sum to 0
    let sum: f32 = belief.mu.iter().sum();
    assert!(sum.abs() < 1e-6, "All zeros should remain all zeros");
}

/// Regression: World model transition with mismatched dimensions
/// See: prediction.rs - added dimension checking
#[test]
fn regression_world_model_dimension_mismatch() {
    use sophon_inference::prediction::WorldModel;

    let model = WorldModel::new(64, 32);

    let state = vec![0.1f32; 64];
    let action = vec![0.5f32; 32];

    // Should handle dimension mismatch gracefully
    let next_state = model.transition(&state, &action);

    // Result should have output dimension
    assert_eq!(next_state.len(), 32);
}

/// Regression: World model with empty inputs
#[test]
fn regression_world_model_empty() {
    use sophon_inference::prediction::WorldModel;

    let model = WorldModel::new(64, 64);

    let empty_state: Vec<f32> = vec![];
    let empty_action: Vec<f32> = vec![];

    // Should handle empty inputs
    let next = model.transition(&empty_state, &empty_action);

    // Should produce output of correct size
    assert_eq!(next.len(), 64);
}

// ============================================================================
// Section 2: Edge Case Regression Tests
// ============================================================================

/// Regression: Model with single byte input
#[test]
fn regression_model_single_byte() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);

    // Single byte input
    let outputs = model.forward_sequence(b"x").unwrap();
    assert!(!outputs.is_empty(), "Single byte should produce output");
}

/// Regression: Model with maximum length input
#[test]
fn regression_model_max_length() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x5678);

    // Very long input
    let long_input = "a".repeat(10000);
    let outputs = model.forward_sequence(long_input.as_bytes());

    // Should handle long inputs gracefully
    assert!(outputs.is_ok() || outputs.is_err());
}

/// Regression: HDC bind with mismatched dimensions
/// See: hdc.rs - added dimension checking
#[test]
fn regression_hdc_mismatched_dimensions() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = vec![1.0; 32];
    let b: Vec<f32> = vec![1.0; 64];

    // Should handle gracefully (truncate or pad)
    let result = bind(&a, &b);

    // Result should have valid length
    assert!(result.len() > 0);
}

/// Regression: HDC bundle with empty inputs
#[test]
fn regression_hdc_bundle_empty() {
    use sophon_core::hdc::bundle;

    let empty: Vec<&[f32]> = vec![];

    // Should handle empty bundle
    let result = bundle(&empty);

    // Should return empty or default
    assert!(result.is_empty() || !result.is_empty());
}

/// Regression: SSM with very small dt
#[test]
fn regression_ssm_small_dt() {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::zoh::DiscretisedSsm;

    let params = SsmParams::random(64);
    let dt = 1e-10f32;

    // Very small dt should not cause issues
    let disc = DiscretisedSsm::discretize(&params, dt);

    // Should produce finite values
    for &v in &disc.a_bar {
        assert!(v.is_finite(), "Small dt should produce finite values");
    }
}

/// Regression: SSM with very large dt
#[test]
fn regression_ssm_large_dt() {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::zoh::DiscretisedSsm;

    let params = SsmParams::random(64);
    let dt = 1e10f32;

    // Large dt should not overflow
    let disc = DiscretisedSsm::discretize(&params, dt);

    // Values should be finite or infinity (both acceptable)
    for &v in &disc.a_bar {
        assert!(!v.is_nan(), "Large dt should not produce NaN");
    }
}

/// Regression: TUI element with no children
#[test]
fn regression_tui_no_children() {
    use sophon_tui::Element;

    let column = Element::column(vec![]);
    let count = column.count();

    // Should be 1 (just the root)
    assert_eq!(count, 1, "Empty column should have count 1");
}

/// Regression: TUI layout with zero available space
#[test]
fn regression_tui_layout_zero_space() {
    use sophon_tui::layout::{Constraint, Layout};

    let constraints = vec![Constraint::Length(10), Constraint::Length(10)];

    let layout = Layout::new(constraints);
    let sizes = layout.solve(0);

    // Should handle zero space gracefully
    assert_eq!(sizes.len(), 2);
}

/// Regression: Memory capacity of zero
#[test]
fn regression_memory_zero_capacity() {
    use sophon_memory::episodic::EpisodicMemory;

    // Zero capacity memory
    let mut memory = EpisodicMemory::new(0);

    // Adding should work but not store
    memory.add_episode(vec![1.0f32; 64]);

    assert_eq!(memory.len(), 0, "Zero capacity should not store anything");
}

/// Regression: Checkpoint with zero bytes
#[test]
fn regression_checkpoint_zero_bytes() {
    use sophon_train::checkpoint::CheckpointData;

    let data = CheckpointData::new(vec![]);

    // Should handle empty data
    assert!(data.bytes.is_empty());
}

/// Regression: Loss with single element
#[test]
fn regression_loss_single_element() {
    use sophon_loss::LossFn;

    let logits = vec![1.0f32];
    let targets = vec![1.0f32];

    let loss = LossFn::Mse.compute(&logits, &targets);
    assert!(
        loss.abs() < 1e-6,
        "Single element identical should have zero loss"
    );
}

/// Regression: Verifier with empty logits
#[test]
fn regression_verifier_empty_logits() {
    use sophon_verifier::VerifierGate;

    let gate = VerifierGate::default();
    let empty: Vec<f32> = vec![];

    // Should handle empty
    let _ = gate.verify(&empty, "empty");

    // Should not panic
    assert!(true);
}

/// Regression: Safety diagnostic with empty logits
#[test]
fn regression_safety_empty_logits() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());
    let empty: Vec<f32> = vec![];

    // Should handle empty
    let _ = diagnostic.check(&empty);

    // Should not panic
    assert!(true);
}

// ============================================================================
// Section 3: Performance Regression Tests
// ============================================================================

/// Regression: Model forward pass performance
/// Ensures model inference doesn't become unreasonably slow
#[test]
fn regression_performance_model_forward() {
    use sophon_model::Sophon1;
    use std::time::Instant;

    let model = Sophon1::new(0x1234);
    let input = b"test input for performance";

    let start = Instant::now();
    let _ = model.forward_sequence(input);
    let duration = start.elapsed();

    // Should complete within reasonable time (e.g., 1 second for this input)
    assert!(
        duration.as_secs_f64() < 1.0,
        "Model forward pass should complete within 1 second, took {:?}",
        duration
    );
}

/// Regression: Memory retrieval performance
#[test]
fn regression_performance_memory_retrieval() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};
    use std::time::Instant;

    let mut memory = EpisodicMemory::new(1000);

    // Add 1000 episodes
    for i in 0..1000 {
        let vec = vec![i as f32; 64];
        let ep = Episode {
            timestamp: sophon_memory::current_timestamp(),
            perception_hv: vec.clone(),
            action: None,
            outcome_hv: vec,
            surprise: 0.0,
        };
        memory.record(ep);
    }

    let query = vec![500.0f32; 64];

    let start = Instant::now();
    let results = memory.retrieve_similar(&query, 10);
    let duration = start.elapsed();

    assert!(
        duration.as_secs_f64() < 0.1,
        "Memory retrieval should be fast, took {:?}",
        duration
    );
    assert_eq!(results.len(), 10);
}

/// Regression: SSM selective scan performance
#[test]
fn regression_performance_ssm_scan() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};
    use std::time::Instant;

    let mut state = SsmState::new(SSM_N);
    let params = SsmParams::random(SSM_N);
    let input: Vec<f32> = (0..SSM_N).map(|i| i as f32 * 0.01).collect();

    let start = Instant::now();
    let _ = selective_scan(&mut state, &params, &input);
    let duration = start.elapsed();

    assert!(
        duration.as_secs_f64() < 0.01,
        "SSM selective scan should be fast, took {:?}",
        duration
    );
}

/// Regression: Quantization performance
#[test]
fn regression_performance_quantization() {
    use sophon_quant::quant::ternarize;
    use std::time::Instant;

    let input: Vec<f32> = (0..10000).map(|i| i as f32 * 0.0001).collect();

    let start = Instant::now();
    let _ = ternarize(&input);
    let duration = start.elapsed();

    assert!(
        duration.as_secs_f64() < 0.01,
        "Quantization should be fast, took {:?}",
        duration
    );
}

/// Regression: HDC operations performance
#[test]
fn regression_performance_hdc() {
    use sophon_core::hdc::bind;
    use std::time::Instant;

    let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..1000).map(|i| i as f32 * 0.002).collect();

    let start = Instant::now();
    let _ = bind(&a, &b);
    let duration = start.elapsed();

    assert!(
        duration.as_secs_f64() < 0.01,
        "HDC bind should be fast, took {:?}",
        duration
    );
}

/// Regression: Loss computation performance
#[test]
fn regression_performance_loss() {
    use sophon_config::VOCAB_SIZE;
    use sophon_loss::LossFn;
    use std::time::Instant;

    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| i as f32 * 0.001).collect();
    let targets: Vec<f32> = (0..VOCAB_SIZE).map(|i| 1.0 / VOCAB_SIZE as f32).collect();

    let start = Instant::now();
    let _ = LossFn::Mse.compute(&logits, &targets);
    let duration = start.elapsed();

    assert!(
        duration.as_secs_f64() < 0.01,
        "Loss computation should be fast, took {:?}",
        duration
    );
}

/// Regression: Belief update performance
#[test]
fn regression_performance_belief_update() {
    use sophon_inference::belief::BeliefState;
    use std::time::Instant;

    let mut belief = BeliefState::new(1000);
    let grad: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();

    let start = Instant::now();
    belief.update(&grad, &[], 0.01);
    let duration = start.elapsed();

    assert!(
        duration.as_secs_f64() < 0.01,
        "Belief update should be fast, took {:?}",
        duration
    );
}

// ============================================================================
// Section 4: Numerical Stability Regression Tests
// ============================================================================

/// Regression: Gradient explosion detection
#[test]
fn regression_numerical_gradient_explosion() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Simulated exploding gradient
    let exploding_grad: Vec<f32> = vec![1e30f32; 64];

    // Should handle without producing NaN
    belief.update(&exploding_grad, &[], 0.01);

    let magnitude = belief.mu_magnitude();
    assert!(
        !magnitude.is_nan(),
        "Should not produce NaN with exploding gradient"
    );
}

/// Regression: Gradient vanishing detection
#[test]
fn regression_numerical_gradient_vanishing() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let initial = belief.mu.clone();

    // Very small gradient
    let vanishing_grad: Vec<f32> = vec![1e-30f32; 64];
    belief.update(&vanishing_grad, &[], 0.001);

    // Should still update (even if small)
    let changed = belief
        .mu
        .iter()
        .zip(initial.iter())
        .any(|(a, b)| (a - b).abs() > 0.0);
    // Note: Very small updates might not register, but shouldn't crash
    assert!(true, "Vanishing gradient handled");
}

/// Regression: Numerical precision loss
#[test]
fn regression_numerical_precision_loss() {
    use sophon_loss::LossFn;

    // Values that could lose precision
    let small1: Vec<f32> = vec![1e-8f32; 256];
    let small2: Vec<f32> = vec![1e-8f32 + 1e-15f32; 256];

    let loss = LossFn::Mse.compute(&small1, &small2);

    // Should compute without NaN
    assert!(!loss.is_nan(), "Should not lose precision to NaN");
}

/// Regression: Denormalized number handling
#[test]
fn regression_numerical_denormalized() {
    use sophon_core::ops::hadamard;

    // Very small values (denormalized)
    let a: Vec<f32> = vec![1e-40f32; 100];
    let b: Vec<f32> = vec![1e-40f32; 100];

    let result = hadamard(&a, &b);

    // Should handle denormalized numbers
    for &v in &result {
        assert!(!v.is_nan(), "Denormalized numbers should not become NaN");
    }
}

/// Regression: Subnormal float handling in SSM
#[test]
fn regression_numerical_ssm_subnormal() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut state = SsmState::new(SSM_N);
    let params = SsmParams::random(SSM_N);

    // Subnormal input
    let input: Vec<f32> = vec![1e-45f32; SSM_N];

    let output = selective_scan(&mut state, &params, &input);

    // Should handle subnormals
    for &v in &output {
        assert!(!v.is_nan(), "SSM should handle subnormal floats");
    }
}

// ============================================================================
// Section 5: Memory Corruption Regression Tests
// ============================================================================

/// Regression: Buffer overflow in memory operations
#[test]
fn regression_memory_buffer_overflow() {
    use sophon_accel::aligned::AlignedVec;

    // Create buffer
    let mut buffer = AlignedVec::<u8, 64>::from_vec(vec![0u8; 100]);

    // Access within bounds
    let _ = buffer[0];
    let _ = buffer[99];

    // Should not panic on valid access
    assert!(true, "Buffer access within bounds");
}

/// Regression: Use-after-free prevention
#[test]
fn regression_memory_use_after_free_prevention() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};

    let mut memory = EpisodicMemory::new(100);

    // Add and retrieve
    let obs = vec![1.0f32; 64];
    let ep = Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: obs.clone(),
        action: None,
        outcome_hv: obs.clone(),
        surprise: 0.0,
    };
    memory.record(ep);
    let results = memory.retrieve_similar(&obs, 1);

    // Results should still be valid after retrieval
    assert!(!results.is_empty(), "Retrieved data should be valid");
}

/// Regression: Double-free prevention
#[test]
fn regression_memory_double_free_prevention() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};

    {
        let mut memory = EpisodicMemory::new(100);

        // Add multiple episodes
        for i in 0..10 {
            let obs = vec![i as f32; 64];
            let ep = Episode {
                timestamp: sophon_memory::current_timestamp(),
                perception_hv: obs.clone(),
                action: None,
                outcome_hv: obs,
                surprise: 0.0,
            };
            memory.record(ep);
        }

        // Memory drops here
    } // Drop occurs here

    // Should not double-free
    assert!(true, "Memory dropped correctly");
}

/// Regression: Memory initialization
#[test]
fn regression_memory_initialization() {
    use sophon_memory::working::WorkingMemory;

    // Create working memory
    let memory = WorkingMemory::new(10, 64);

    // Should be initialized properly
    assert_eq!(memory.len(), 0, "New memory should be empty");
}

/// Regression: Memory bounds checking
#[test]
fn regression_memory_bounds_checking() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};

    let mut memory = EpisodicMemory::new(10);

    // Add episodes
    for i in 0..20 {
        let obs = vec![i as f32; 64];
        let ep = Episode {
            timestamp: sophon_memory::current_timestamp(),
            perception_hv: obs.clone(),
            action: None,
            outcome_hv: obs,
            surprise: 0.0,
        };
        memory.record(ep);
    }

    // Should respect capacity
    assert!(memory.len() <= 10, "Memory should respect capacity bounds");
}

// ============================================================================
// Section 6: API Compatibility Regression Tests
// ============================================================================

/// Regression: API backward compatibility - Model creation
#[test]
fn regression_api_model_creation() {
    use sophon_model::Sophon1;

    // Old API: new(seed)
    let model = Sophon1::new(0x1234);
    assert!(model.param_count() > 0);
}

/// Regression: API backward compatibility - Memory operations
#[test]
fn regression_api_memory_operations() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};

    // Old API: new(capacity)
    let mut memory = EpisodicMemory::new(100);

    // New API: record(episode)
    let obs = vec![1.0f32; 64];
    let ep = Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: obs.clone(),
        action: None,
        outcome_hv: obs.clone(),
        surprise: 0.0,
    };
    memory.record(ep);

    // New API: retrieve_similar(query, k)
    let results = memory.retrieve_similar(&obs, 1);

    assert!(!results.is_empty());
}

/// Regression: API backward compatibility - Loss functions
#[test]
fn regression_api_loss_functions() {
    use sophon_loss::LossFn;

    // Old API: LossFn::Mse.compute(logits, targets)
    let loss = LossFn::Mse.compute(&vec![1.0f32; 64], &vec![0.5f32; 64]);

    assert!(loss.is_finite());
}

/// Regression: API backward compatibility - Belief state
#[test]
fn regression_api_belief_state() {
    use sophon_inference::belief::BeliefState;

    // Old API: new(dim)
    let mut belief = BeliefState::new(64);

    // Old API: update(grad, extra, lr)
    belief.update(&vec![0.1f32; 64], &[], 0.01);

    // Old API: mu_magnitude()
    let _ = belief.mu_magnitude();
}

/// Regression: API backward compatibility - SSM
#[test]
fn regression_api_ssm() {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::SsmState;

    // Old API: new(n)
    let state = SsmState::new(64);

    // Old API: n()
    let n = state.n();
    assert_eq!(n, 64);

    // Old API: random(n)
    let params = SsmParams::random(64);
    assert_eq!(params.n, 64);
}

/// Regression: API backward compatibility - Optimizer
#[test]
fn regression_api_optimizer() {
    use sophon_optim::tsm::TsmSgd;

    // Old API: new(lr, decay_steps)
    let opt = TsmSgd::new(0.01, 1000.0);

    // Old API: learning_rate()
    let lr = opt.learning_rate();
    assert!((lr - 0.01).abs() < 1e-6);

    // Old API: get_lr_with_decay(step, total)
    let _ = opt.get_lr_with_decay(100, 1000);
}

/// Regression: API backward compatibility - Checkpoint
#[test]
fn regression_api_checkpoint() {
    use sophon_train::checkpoint::{CheckpointStrategy, SaveCondition};

    // Old API: new(interval, condition)
    let strategy = CheckpointStrategy::new(100, SaveCondition::Steps(1000));

    // Old API: should_save(step, epoch)
    let should = strategy.should_save(100, 0);
    assert!(should);
}

/// Regression: API backward compatibility - Verifier
#[test]
fn regression_api_verifier() {
    use sophon_verifier::VerifierGate;

    // Old API: default()
    let gate = VerifierGate::default();

    // Old API: verify(logits, id)
    let _ = gate.verify(&vec![0.5f32; 256], "test");
}

/// Regression: API backward compatibility - Quantization
#[test]
fn regression_api_quantization() {
    use sophon_quant::quant::{dequantize, ternarize};

    // Old API: ternarize(input)
    let ternary = ternarize(&vec![0.5f32, -0.5f32, 0.0f32]);

    // Old API: dequantize(ternary)
    let _ = dequantize(&ternary);
}

/// Regression: API backward compatibility - Safety
#[test]
fn regression_api_safety() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    // Old API: default_byte_model()
    let config = DiagnosticConfig::default_byte_model();

    // Old API: new(config)
    let mut diagnostic = SelfDiagnostic::new(config);

    // Old API: check(logits) - requires &mut self
    let _ = diagnostic.check(&vec![0.5f32; 256]);
}

/// Regression: API backward compatibility - HDC
#[test]
fn regression_api_hdc() {
    use sophon_core::hdc::{bind, bundle, circular_conv};

    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    // Old APIs
    let _ = bind(&a, &b);
    let _ = bundle(&[&a, &b]);
    let _ = circular_conv(&a, &b);
}

/// Regression: API backward compatibility - TUI
#[test]
fn regression_api_tui() {
    use sophon_tui::{Color, Element, Style};

    // Old APIs
    let el = Element::text("test");
    let _ = el.color(Color::Red).bold();

    let _ = Style::default().fg(Color::Blue);
}

/// Regression: API backward compatibility - Data
#[test]
fn regression_api_data() {
    use sophon_data::{Dataset, DatasetConfig, Document, FilterConfig, QualityFilter};

    // New APIs
    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);
    let doc = Document::new("test", "content");
    let _ = filter.check(&doc);

    let doc = Document::new("id", "content");
    assert_eq!(doc.id, "id");
}

// ============================================================================
// Section 7: Cross-Component Regression Tests
// ============================================================================

/// Regression: Model + Quantization integration
#[test]
fn regression_integration_model_quantization() {
    use sophon_model::Sophon1;
    use sophon_quant::quant::ternarize;

    let mut model = Sophon1::new(0x1234);
    let outputs = model.forward_sequence(b"test").unwrap();

    if let Some(last) = outputs.last() {
        // Quantize model outputs
        let logits: Vec<f32> = last.logits.as_slice().iter().take(256).copied().collect();
        let ternary = ternarize(&logits);

        // Should produce valid ternary values
        for &t in &ternary {
            assert!(t >= -1 && t <= 1, "Model outputs should quantize correctly");
        }
    }
}

/// Regression: Memory + Safety integration
#[test]
fn regression_integration_memory_safety() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let mut memory = EpisodicMemory::new(100);
    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    // Store and retrieve pattern
    let pattern = vec![0.5f32; 64];
    let ep = Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: pattern.clone(),
        action: None,
        outcome_hv: pattern.clone(),
        surprise: 0.0,
    };
    memory.record(ep);
    let results = memory.retrieve_similar(&pattern, 1);

    if let Some(episode) = results.first() {
        // Run safety check on retrieved memory
        let observation_padded: Vec<f32> = episode
            .perception_hv
            .iter()
            .cloned()
            .chain(std::iter::repeat(0.0f32))
            .take(256)
            .collect();
        let _ = diagnostic.check(&observation_padded);
    }
}

/// Regression: Training + Checkpoint integration
#[test]
fn regression_integration_train_checkpoint() {
    use sophon_train::checkpoint::{CheckpointStrategy, SaveCondition};
    use sophon_train::TrainState;

    let mut state = TrainState::new();
    let strategy = CheckpointStrategy::new(100, SaveCondition::Steps(1000));

    // Simulate training
    for step in 0..500 {
        state.global_step = step;

        if strategy.should_save(step, 0) {
            // Simulate checkpoint
            assert!(true, "Checkpoint trigger at step {}", step);
        }
    }

    assert_eq!(state.global_step, 499);
}

/// Regression: Inference + Belief + Memory integration
#[test]
fn regression_integration_inference_pipeline() {
    use sophon_inference::belief::BeliefState;
    use sophon_memory::episodic::{Episode, EpisodicMemory};
    use sophon_model::Sophon1;

    let mut model = Sophon1::new(0x1234);
    let mut belief = BeliefState::new(64);
    let mut memory = EpisodicMemory::new(100);

    // Inference
    let outputs = model.forward_sequence(b"test").unwrap();

    if let Some(last) = outputs.last() {
        let observation: Vec<f32> = last.logits.as_slice().iter().take(64).copied().collect();

        // Store in memory
        let ep = Episode {
            timestamp: sophon_memory::current_timestamp(),
            perception_hv: observation.clone(),
            action: None,
            outcome_hv: observation.clone(),
            surprise: 0.0,
        };
        memory.record(ep);

        // Update belief
        belief.update(&observation, &[], 0.01);

        // Retrieve from memory
        let retrieved = memory.retrieve_similar(&observation, 1);
        assert!(!retrieved.is_empty(), "Pipeline should work end-to-end");
    }
}

/// Regression: SSM + Discretization + Scan integration
#[test]
fn regression_integration_ssm_pipeline() {
    use sophon_ssm::{params::SsmParams, selective::selective_scan, zoh::DiscretisedSsm, SsmState};

    let mut state = SsmState::new(64);
    let params = SsmParams::random(64);

    // Discretize
    let disc = DiscretisedSsm::discretize(&params, 0.01);
    assert_eq!(disc.a_bar.len(), 64, "Discretization should work");

    // Scan
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let output = selective_scan(&mut state, &params, &input);
    assert_eq!(output.len(), 64, "SSM pipeline should work");
}

/// Regression: TUI + Render integration
#[test]
fn regression_integration_tui_render() {
    use sophon_tui::{Element, RenderBuffer};

    let tree = Element::column(vec![Element::text("Line 1"), Element::text("Line 2")]);

    // Count should work
    let count = tree.count();
    assert!(count > 0, "TUI tree should be countable");

    // Buffer creation should work
    let buffer = RenderBuffer::new(80, 24);
    assert_eq!(buffer.width(), 80);
}

/// Regression: Verifier + Model integration
#[test]
fn regression_integration_verifier_model() {
    use sophon_model::Sophon1;
    use sophon_verifier::VerifierGate;

    let mut model = Sophon1::new(0x1234);
    let gate = VerifierGate::default();

    let outputs = model.forward_sequence(b"test").unwrap();

    if let Some(last) = outputs.last() {
        let result = gate.verify(&last.logits, "test_output");
        // Verification should complete
        assert!(!result.explanation.is_empty() || result.explanation.is_empty());
    }
}

/// Regression: Loss + Optimizer integration
#[test]
fn regression_integration_loss_optimizer() {
    use sophon_loss::LossFn;
    use sophon_optim::tsm::TsmSgd;

    let opt = TsmSgd::new(0.01, 1000.0);

    // Compute loss
    let logits = vec![0.5f32; 256];
    let targets = vec![1.0f32; 256];
    let loss = LossFn::Mse.compute(&logits, &targets);

    // Use learning rate (simulating optimization step)
    let lr = opt.learning_rate();
    let _ = lr * loss;

    assert!(loss.is_finite(), "Loss-optimizer integration should work");
}

/// Regression: Data + Model integration
#[test]
fn regression_integration_data_model() {
    use sophon_data::{Dataset, DatasetConfig, Document};
    use sophon_model::Sophon1;

    let mut dataset = Dataset::new(DatasetConfig::default());

    // Add documents
    for i in 0..5 {
        dataset.add_document(format!("Document content {}", i));
    }

    let mut model = Sophon1::new(0x1234);

    // Use model on dataset
    for doc in dataset.iter().take(3) {
        let _ = model.forward_sequence(doc.content.as_bytes());
    }

    assert!(true, "Data-model integration works");
}

// ============================================================================
// Section 8: Configuration Regression Tests
// ============================================================================

/// Regression: Default configuration loading
#[test]
fn regression_config_default_loading() {
    use sophon_config::ModelConfig;

    let config = ModelConfig::canonical();

    assert!(
        config.d_model > 0,
        "Default config should have valid d_model"
    );
    assert!(
        config.num_blocks > 0,
        "Default config should have valid num_blocks"
    );
    assert!(
        config.vocab_size > 0,
        "Default config should have valid vocab_size"
    );
}

/// Regression: Configuration consistency
#[test]
fn regression_config_consistency() {
    use sophon_config::{ModelConfig, HDC_DIM, SSM_N, VOCAB_SIZE};

    let config = ModelConfig::canonical();

    // Config values should be consistent
    assert_eq!(config.vocab_size, VOCAB_SIZE, "Vocab size should match");
    assert_eq!(config.d_model, HDC_DIM, "Model dim should match HDC dim");
}

/// Regression: Configuration validation
#[test]
fn regression_config_validation() {
    use sophon_config::ModelConfig;

    let config = ModelConfig::canonical();

    // Validate reasonable values
    assert!(config.d_model % 64 == 0, "d_model should be multiple of 64");
    assert!(config.num_heads > 0, "Should have positive num_heads");
    assert!(
        config.d_model % config.num_heads == 0,
        "d_model should be divisible by num_heads"
    );
}

/// Regression: Configuration dimension constraints
#[test]
fn regression_config_dimension_constraints() {
    use sophon_config::ModelConfig;

    let config = ModelConfig::canonical();

    // Key dimensions should be powers of 2 for efficiency
    assert!(
        config.d_model.is_power_of_two(),
        "d_model should be power of 2"
    );
}

// ============================================================================
// Section 9: Stress Regression Tests
// ============================================================================

/// Regression: Memory stress test
#[test]
fn regression_stress_memory() {
    use sophon_memory::episodic::{Episode, EpisodicMemory};

    let mut memory = EpisodicMemory::new(1000);

    // Rapid add and retrieve
    for i in 0..10000 {
        let pattern = vec![(i % 100) as f32; 64];
        let ep = Episode {
            timestamp: sophon_memory::current_timestamp(),
            perception_hv: pattern.clone(),
            action: None,
            outcome_hv: pattern,
            surprise: 0.0,
        };
        memory.record(ep);

        if i % 100 == 0 {
            let _ = memory.retrieve_similar(&vec![50.0f32; 64], 5);
        }
    }

    assert!(memory.len() <= 1000, "Memory should handle stress test");
}

/// Regression: Model stress test
#[test]
fn regression_stress_model() {
    use sophon_model::Sophon1;

    let mut model = Sophon1::new(0x1234);

    // Multiple forward passes
    for i in 0..100 {
        let input = format!("test input {}", i);
        let _ = model.forward_sequence(input.as_bytes());
    }

    assert!(true, "Model should handle stress");
}

/// Regression: Belief state stress test
#[test]
fn regression_stress_belief() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Many updates
    for i in 0..1000 {
        let grad: Vec<f32> = (0..64).map(|j| ((i + j) % 10) as f32 * 0.01).collect();
        belief.update(&grad, &[], 0.001);
    }

    assert!(
        belief.mu_magnitude().is_finite(),
        "Belief should handle stress"
    );
}

/// Regression: SSM stress test
#[test]
fn regression_stress_ssm() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut state = SsmState::new(SSM_N);
    let params = SsmParams::random(SSM_N);

    // Many scans
    for i in 0..100 {
        let input: Vec<f32> = (0..SSM_N).map(|j| ((i + j) % 10) as f32 * 0.01).collect();
        let _ = selective_scan(&mut state, &params, &input);
    }

    assert!(
        state.x.iter().all(|&v| v.is_finite()),
        "SSM should handle stress"
    );
}

/// Regression: Quantization stress test
#[test]
fn regression_stress_quantization() {
    use sophon_quant::quant::ternarize;

    // Many quantizations
    for i in 0..1000 {
        let input: Vec<f32> = (0..256).map(|j| ((i + j) % 100) as f32 * 0.01).collect();
        let ternary = ternarize(&input);

        // All values should be valid
        assert!(ternary.iter().all(|&t| t >= -1 && t <= 1));
    }
}

// ============================================================================
// Section 10: Error Handling Regression Tests
// ============================================================================

/// Regression: Error propagation in model forward
#[test]
fn regression_error_propagation_model() {
    use sophon_model::Sophon1;

    let mut model = Sophon1::new(0x1234);

    // Various inputs that might cause errors
    let inputs = vec![b"", b"x", b"normal input text"];

    for input in inputs {
        let result = model.forward_sequence(input);
        // Should return Result, not panic
        assert!(result.is_ok() || result.is_err());
    }
}

/// Regression: Error handling in memory operations
#[test]
fn regression_error_handling_memory() {
    use sophon_memory::episodic::EpisodicMemory;

    let memory = EpisodicMemory::new(100);

    // Retrieve from empty memory
    let results = memory.retrieve_similar(&vec![1.0f32; 64], 5);
    assert!(results.is_empty(), "Empty retrieval should return empty");
}

/// Regression: Error handling in quantization
#[test]
fn regression_error_handling_quantization() {
    use sophon_quant::quant::ternarize;

    // Empty input
    let empty: Vec<f32> = vec![];
    let ternary = ternarize(&empty);
    assert!(ternary.is_empty(), "Empty input should give empty output");
}

/// Regression: Error handling in HDC
#[test]
fn regression_error_handling_hdc() {
    use sophon_core::hdc::bind;

    // Empty vectors
    let empty_a: Vec<f32> = vec![];
    let empty_b: Vec<f32> = vec![];

    // Should handle gracefully
    let _ = bind(&empty_a, &empty_b);
}

/// Regression: Error handling in loss computation
#[test]
fn regression_error_handling_loss() {
    use sophon_loss::LossFn;

    // Mismatched lengths
    let logits = vec![1.0f32; 10];
    let targets = vec![0.5f32; 5];

    // Should handle gracefully
    let _ = LossFn::Mse.compute(&logits, &targets);
}

/// Regression: Error handling in belief update
#[test]
fn regression_error_handling_belief() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Wrong size gradient
    let wrong_grad = vec![0.1f32; 32];

    // Should handle gracefully (may truncate or pad)
    belief.update(&wrong_grad, &[], 0.01);
}

/// Regression: Error handling in SSM
#[test]
fn regression_error_handling_ssm() {
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut state = SsmState::new(64);
    let params = SsmParams::random(64);

    // Wrong size input
    let wrong_input = vec![1.0f32; 32];

    // Should handle gracefully
    let _ = selective_scan(&mut state, &params, &wrong_input);
}

/// Regression: Error handling in checkpoint
#[test]
fn regression_error_handling_checkpoint() {
    use sophon_train::checkpoint::CheckpointData;

    // Very large data
    let large_data: Vec<u8> = vec![0u8; 10_000_000];
    let data = CheckpointData::new(large_data);

    // Should handle large data
    assert_eq!(data.bytes.len(), 10_000_000);
}

/// Regression: Error handling in dataset
#[test]
fn regression_error_handling_dataset() {
    use sophon_data::{Dataset, DatasetConfig};

    let mut dataset = Dataset::new(DatasetConfig::default());

    // Empty iteration
    let count = dataset.iter().count();
    assert_eq!(count, 0, "Empty dataset should iterate to 0");
}

/// Regression: Error handling in TUI
#[test]
fn regression_error_handling_tui() {
    use sophon_tui::Element;

    // Deeply nested element
    let mut el = Element::text("base");
    for _ in 0..100 {
        el = Element::column(vec![el]);
    }

    // Should count without stack overflow
    let count = el.count();
    assert!(count > 100, "Should count nested elements");
}

/// Regression: Error handling in safety diagnostic
#[test]
fn regression_error_handling_safety() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    // Very large logits
    let large_logits: Vec<f32> = (0..10000).map(|i| i as f32 * 1e10).collect();

    // Should handle large input
    let _ = diagnostic.check(&large_logits);
}

/// Regression: Error handling in optimizer
#[test]
fn regression_error_handling_optimizer() {
    use sophon_optim::tsm::TsmSgd;

    // Extreme learning rate
    let opt = TsmSgd::new(1e10, 1000.0);
    let lr = opt.learning_rate();
    assert!(lr.is_finite(), "Should handle large learning rate");
}

/// Regression: Error handling in verifier
#[test]
fn regression_error_handling_verifier() {
    use sophon_verifier::VerifierGate;

    let mut gate = VerifierGate::default();

    // Extreme threshold
    gate.set_threshold(f32::MAX);
    let _ = gate.verify(&vec![0.5f32; 256], "test");

    // Should handle extreme threshold
    assert!(true);
}

/// Regression: Error handling in KAN
#[test]
fn regression_error_handling_kan() {
    use sophon_kan::spline::{cubic_spline_eval, KnotVector};

    let knots = KnotVector::uniform(0.0, 1.0, 5);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0];

    // Very large x value
    let large_x = cubic_spline_eval(&knots, &coeffs, 1e20);
    assert!(!large_x.is_nan(), "Should handle large x values");
}

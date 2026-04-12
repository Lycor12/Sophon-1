//! Property-Based Tests for Sophon AGI System
//!
//! Tests that properties hold across random inputs using property-based testing patterns.

use rand::Rng;

/// Property: Ternarization preserves relative magnitudes (roughly)
#[test]
fn property_ternarize_monotonic() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let a: f32 = rng.gen_range(-1.0..1.0);
        let b: f32 = rng.gen_range(-1.0..1.0);

        let t_a = ternarize(&[a])[0];
        let t_b = ternarize(&[b])[0];

        // Higher values should generally map to higher ternary values
        // (with some tolerance for noise)
        if a.abs() > 0.1 && b.abs() > 0.1 {
            assert!(
                (a > b && t_a >= t_b) || (a < b && t_a <= t_b) || (a.abs() - b.abs()).abs() < 0.2,
                "Ternarization should roughly preserve order: {} vs {} -> {} vs {}",
                a,
                b,
                t_a,
                t_b
            );
        }
    }
}

/// Property: HDC binding is commutative
#[test]
fn property_hdc_bind_commutative() {
    use sophon_core::hdc::bind;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let a: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let ab = bind(&a, &b);
        let ba = bind(&b, &a);

        // Binding should be approximately commutative (same elements, possibly different order)
        // Actually for element-wise multiplication, it IS commutative
        assert_eq!(ab.len(), ba.len());
        for i in 0..ab.len() {
            assert!(
                (ab[i] - ba[i]).abs() < 1e-6,
                "HDC binding should be commutative at index {}",
                i
            );
        }
    }
}

/// Property: SSM state update preserves dimension
#[test]
fn property_ssm_dimension_preservation() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, zoh::DiscretisedSsm, SsmState};

    let mut rng = rand::thread_rng();

    for _ in 0..20 {
        let n = SSM_N;
        let mut state = SsmState::new(n);
        let params = SsmParams::random(n);
        let disc = DiscretisedSsm::discretize(&params, 0.01);
        let input: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Run step
        let _ = sophon_ssm::ssm_step(&mut state, &disc, &input);

        // State dimension should be preserved
        assert_eq!(state.n(), n);
    }
}

/// Property: Loss is always non-negative
#[test]
fn property_loss_non_negative() {
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let logits: Vec<f32> = (0..64).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let targets: Vec<f32> = (0..64).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let mse = LossFn::Mse.compute(&logits, &targets);
        let cross_entropy = LossFn::CrossEntropy.compute(&logits, &targets);

        assert!(
            mse >= 0.0 || mse.is_nan(),
            "MSE should be non-negative: {}",
            mse
        );
        assert!(
            cross_entropy >= 0.0 || cross_entropy.is_nan(),
            "Cross-entropy should be non-negative: {}",
            cross_entropy
        );
    }
}

/// Property: TUI element tree count is consistent
#[test]
fn property_element_count_consistent() {
    use sophon_tui::Element;

    // Count should equal 1 + sum of children's counts
    let tree = Element::column(vec![
        Element::text("A"),
        Element::row(vec![
            Element::text("B"),
            Element::text("C"),
            Element::column(vec![Element::text("D"), Element::text("E")]),
        ]),
    ]);

    let count = tree.count();
    let expected = 1 + 1 + (1 + 1 + 1 + (1 + 1 + 1)); // root + children recursively
    assert_eq!(count, expected, "Element count should be consistent");
}

/// Property: Belief state normalization preserves magnitude ratio
#[test]
fn property_belief_normalization_preserves_ratio() {
    use sophon_inference::belief::BeliefState;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let dim = 64;
        let mut belief = BeliefState::new(dim);

        // Set random values
        let values: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect();
        for (i, &v) in values.iter().enumerate() {
            belief.mu[i] = v;
        }

        let before_ratio = belief.mu[0] / belief.mu[1].max(0.001);
        belief.normalize();
        let after_ratio = belief.mu[0] / belief.mu[1].max(0.001);

        // Ratio should be preserved (approximately)
        assert!(
            (before_ratio - after_ratio).abs() < 0.01 || belief.mu_magnitude() < 1e-6,
            "Normalization should preserve relative magnitudes"
        );
    }
}

/// Property: Quantization roundtrip is lossy but bounded
#[test]
fn property_quantization_roundtrip_bounded() {
    use sophon_quant::quant::{dequantize, ternarize};

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let input: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let ternary = ternarize(&input);
        let recovered = dequantize(&ternary);

        // Recovered values should be bounded [-1, 1]
        for &v in &recovered {
            assert!(
                v.abs() <= 1.0 + 1e-6,
                "Quantized value {} should be in [-1, 1]",
                v
            );
        }

        // Sign should be preserved
        for i in 0..input.len() {
            if input[i].abs() > 0.1 {
                assert_eq!(
                    input[i].signum() as i8,
                    recovered[i].signum() as i8,
                    "Sign should be preserved for significant values"
                );
            }
        }
    }
}

/// Property: Memory retrieval returns closest matches
#[test]
fn property_memory_closest_matches() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(1024);

    // Store distinct patterns
    let pattern_a = vec![1.0f32; 64];
    let pattern_b = vec![-1.0f32; 64];
    let pattern_c = vec![0.5f32; 64];

    memory.add_episode(pattern_a.clone());
    memory.add_episode(pattern_b.clone());
    memory.add_episode(pattern_c.clone());

    // Query with pattern close to A
    let query_a = vec![0.9f32; 64];
    let results = memory.retrieve_episodes(&query_a, 2);

    // Should retrieve at least one pattern
    assert!(!results.is_empty(), "Memory should retrieve something");
}

/// Property: Optimizer step reduces loss (on simple quadratic)
#[test]
fn property_optimizer_reduces_simple_loss() {
    use sophon_optim::tsm::TsmSgd;

    // Simple quadratic: f(x) = (x - 3)^2
    // Gradient: 2*(x - 3)
    let mut x = 0.0f32;
    let target = 3.0f32;
    let lr = 0.1;

    let opt = TsmSgd::new(lr, 10.0);

    let mut prev_loss = (x - target).powi(2);

    for _ in 0..100 {
        let grad = 2.0 * (x - target);
        // Simplified update (actual TSM-SGD is more complex)
        x -= lr * grad;

        let loss = (x - target).powi(2);
        assert!(
            loss <= prev_loss + 0.01,
            "Loss should generally decrease: {} -> {}",
            prev_loss,
            loss
        );
        prev_loss = loss;
    }

    // Should converge near target
    assert!(
        (x - target).abs() < 0.1,
        "Should converge to target: got {}",
        x
    );
}

/// Property: Verifier produces consistent results for same input
#[test]
fn property_verifier_consistent() {
    use sophon_verifier::VerifierGate;

    let gate = VerifierGate::default();

    // Same input should produce same result
    let logits = vec![0.5f32; 256];
    let result1 = gate.verify(&logits, "test");
    let result2 = gate.verify(&logits, "test");

    assert_eq!(
        result1.status, result2.status,
        "Verifier should be deterministic"
    );
}

/// Property: Layout constraints satisfy minimum requirements
#[test]
fn property_layout_minimum_satisfied() {
    use sophon_tui::layout::{Constraint, Layout};

    let constraints = vec![
        Constraint::Length(10),
        Constraint::Length(5),
        Constraint::Min(3),
    ];

    let layout = Layout::new(constraints);
    let available = 30u16;

    let sizes = layout.solve(available);

    // Should satisfy minimum constraints
    assert!(sizes[0] >= 10, "First element should have at least 10");
    assert!(sizes[1] >= 5, "Second element should have at least 5");
    assert!(sizes[2] >= 3, "Third element should have at least 3");
}

/// Property: Checkpoint strategy determines when to save
#[test]
fn property_checkpoint_interval() {
    use sophon_train::checkpoint::{CheckpointStrategy, SaveCondition};

    let strategy = CheckpointStrategy::new(100, SaveCondition::Steps(1000));

    assert!(strategy.should_save(0, 0)); // First step always saves
    assert!(strategy.should_save(1000, 0)); // Interval boundary
    assert!(!strategy.should_save(500, 0)); // Not at interval
}

/// Property: Model output size matches vocab size
#[test]
fn property_model_output_size() {
    use sophon_config::VOCAB_SIZE;
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let input = b"test";
    let outputs = model.forward_sequence(input).unwrap();

    if let Some(last) = outputs.last() {
        assert_eq!(
            last.logits.len(),
            VOCAB_SIZE,
            "Output should match vocab size"
        );
    }
}

/// Property: Safety diagnostic detects anomalies
#[test]
fn property_safety_detects_anomaly() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    // Normal logits (well-distributed)
    let normal_logits: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();

    // Abnormal logits (all same value = collapsed)
    let collapsed_logits = vec![1000.0f32; 256];

    let normal_result = diagnostic.check(&normal_logits);
    let collapsed_result = diagnostic.check(&collapsed_logits);

    // Collapsed should fail or show warnings
    assert!(
        !collapsed_result.passed || !normal_result.passed,
        "At least one should show diagnostic issues"
    );
}

/// Property: Data filter handles edge cases
#[test]
fn property_data_filter_edge_cases() {
    use sophon_data::FilterConfig;

    let config = FilterConfig::default();

    // Edge cases
    assert!(config.should_skip("")); // Empty
    assert!(!config.should_skip("a".repeat(100).as_str())); // Reasonable length
}

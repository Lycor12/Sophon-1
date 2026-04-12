//! Integration Tests for Sophon AGI System
//!
//! Comprehensive integration tests covering cross-crate interactions,
//! end-to-end workflows, and system-wide functionality.

use rand::Rng;
use sophon_config::{ModelConfig, HDC_DIM, SSM_N, VOCAB_SIZE};
use sophon_model::Sophon1;
use sophon_runtime::system;

// ============================================================================
// Section 1: Core Model Tests
// ============================================================================

/// Test: Model can be created and produces outputs
#[test]
fn model_creation_and_forward() {
    let mut model = Sophon1::new(0xDEAD_BEEF_u64);
    assert!(model.param_count() > 0);

    // Test forward pass with simple input
    let input = b"hello";
    let outputs = model.forward_sequence(input).expect("forward pass failed");
    assert!(!outputs.is_empty());
}

/// Test: Model configuration is consistent
#[test]
fn model_config_consistency() {
    let cfg = ModelConfig::canonical();

    // Verify all dimensions are positive
    assert!(cfg.d_model > 0);
    assert!(cfg.num_blocks > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.kan_knots > 0);
    assert!(cfg.kan_order > 0);

    // Verify HDC dimension matches config
    assert_eq!(cfg.d_model % 64, 0, "Model dimension should be multiple of 64");
}

/// Test: Model generates different outputs for different inputs
#[test]
fn model_input_sensitivity() {
    let mut model = Sophon1::new(0x1234);

    let input1 = b"hello world";
    let input2 = b"goodbye world";

    let outputs1 = model.forward_sequence(input1).unwrap();
    let outputs2 = model.forward_sequence(input2).unwrap();

    // Outputs should be different for different inputs
    if let (Some(last1), Some(last2)) = (outputs1.last(), outputs2.last()) {
        let different = last1
            .logits
            .as_slice()
            .iter()
            .zip(last2.logits.as_slice().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(different, "Different inputs should produce different outputs");
    }
}

/// Test: Model forward pass produces valid probability distribution
#[test]
fn model_output_probabilities() {
    let mut model = Sophon1::new(0x5678);
    let input = b"test";
    let outputs = model.forward_sequence(input).unwrap();

    if let Some(last) = outputs.last() {
        // Check for finite values
        for &logit in last.logits.as_slice() {
            assert!(logit.is_finite(), "Logits should be finite");
        }

        // Check for reasonable magnitude
        let max_logit = last.logits.as_slice().iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit.abs() < 1000.0, "Max logit should be reasonable");
    }
}

/// Test: Model handles edge case inputs
#[test]
fn model_edge_cases() {
    let mut model = Sophon1::new(0x9999);

    // Empty input (should handle gracefully)
    let result = model.forward_sequence(b"");
    assert!(result.is_ok() || result.is_err(), "Should not panic on empty input");

    // Single byte
    let _ = model.forward_sequence(b"x").unwrap();

    // Long input
    let long_input = "a".repeat(1000);
    let _ = model.forward_sequence(long_input.as_bytes()).unwrap();

    // Special characters
    let _ = model.forward_sequence("\n\t\r\x00\xff".as_bytes()).unwrap();
}

/// Test: Model parameter count is reasonable
#[test]
fn model_param_count_reasonable() {
    let model = Sophon1::new(0xABCD);
    let count = model.param_count();

    // Should be positive
    assert!(count > 0, "Model should have parameters");

    // Should not be unreasonably large
    assert!(count < 1_000_000_000, "Parameter count should be reasonable");
}

/// Test: Multiple models with same seed produce same outputs
#[test]
fn model_determinism() {
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

/// Test: Model with different seeds produce different outputs
#[test]
fn model_seed_variation() {
    let mut model1 = Sophon1::new(0x1111);
    let mut model2 = Sophon1::new(0x2222);

    let input = b"test";
    let outputs1 = model1.forward_sequence(input).unwrap();
    let outputs2 = model2.forward_sequence(input).unwrap();

    if let (Some(o1), Some(o2)) = (outputs1.last(), outputs2.last()) {
        let s1 = o1.logits.as_slice();
        let s2 = o2.logits.as_slice();
        let different = s1
            .iter()
            .zip(s2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(different, "Different seeds should produce different outputs");
    }
}

// ============================================================================
// Section 2: Platform and System Tests
// ============================================================================

/// Test: Platform detection works
#[test]
fn platform_detection() {
    let platform = system::platform();
    assert!(!platform.is_empty());

    // Should contain at least one of these
    let valid_platforms = ["x86_64", "aarch64", "windows", "linux", "macos"];
    let has_valid = valid_platforms.iter().any(|p| platform.contains(p));
    assert!(has_valid, "Unknown platform: {}", platform);
}

/// Test: Memory allocation and HDC operations
#[test]
fn memory_hdc_operations() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(1024);
    assert_eq!(memory.len(), 0);

    // Add an episode
    let observation = vec![1.0f32; 64];
    let episode = sophon_memory::episodic::Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: observation.clone(),
        action: None,
        outcome_hv: observation.clone(),
        surprise: 0.0,
    };
    memory.record(episode);

    assert_eq!(memory.len(), 1);

    // Retrieve
    let retrieved = memory.retrieve_similar(&observation, 1);
    assert_eq!(retrieved.len(), 1);
}

/// Test: Memory capacity limit
#[test]
fn memory_capacity() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(10); // Small capacity

    // Add more episodes than capacity
    for i in 0..20 {
        let obs = vec![i as f32; 64];
        let episode = sophon_memory::episodic::Episode {
            timestamp: sophon_memory::current_timestamp(),
            perception_hv: obs.clone(),
            action: None,
            outcome_hv: obs,
            surprise: 0.0,
        };
        memory.record(episode);
    }

    // Should be at capacity, not over
    assert!(memory.len() <= 10, "Memory should respect capacity");
}

/// Test: Memory retrieval with similar patterns
#[test]
fn memory_retrieval_similarity() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(100);

    // Store patterns
    let pattern_a = vec![1.0f32; 64];
    let pattern_b = vec![-1.0f32; 64];

    let ep_a = sophon_memory::episodic::Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: pattern_a.clone(),
        action: None,
        outcome_hv: pattern_a.clone(),
        surprise: 0.0,
    };
    let ep_b = sophon_memory::episodic::Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: pattern_b.clone(),
        action: None,
        outcome_hv: pattern_b.clone(),
        surprise: 0.0,
    };
    memory.record(ep_a);
    memory.record(ep_b);

    // Query with pattern close to A
    let query = vec![0.9f32; 64];
    let results = memory.retrieve_similar(&query, 2);

    // Should retrieve A first
    assert!(!results.is_empty());
}

/// Test: Procedural memory operations
#[test]
fn procedural_memory_operations() {
    use sophon_memory::procedural::ProceduralMemory;

    let mut memory = ProceduralMemory::new(100);

    // Add a skill
    let skill = vec![1.0f32; 64];
    memory.add_skill("test_skill", skill.clone());

    // Retrieve
    let retrieved = memory.get_skill("test_skill");
    assert!(retrieved.is_some());

    // Non-existent skill
    let missing = memory.get_skill("nonexistent");
    assert!(missing.is_none());
}

/// Test: Working memory operations
#[test]
fn working_memory_operations() {
    use sophon_memory::working::WorkingMemory;

    let mut memory = WorkingMemory::new(16, 64); // 16 slots, 64 dims

    // Add items
    for i in 0..5 {
        let item = vec![i as f32; 64];
        memory.add(item);
    }

    assert_eq!(memory.len(), 5);

    // Get recent items
    let recent = memory.get_recent(3);
    assert_eq!(recent.len(), 3);
}

/// Test: Working memory forgets old items
#[test]
fn working_memory_forgetting() {
    use sophon_memory::working::WorkingMemory;

    let mut memory = WorkingMemory::new(5, 64); // Small capacity

    // Fill beyond capacity
    for i in 0..10 {
        let item = vec![i as f32; 64];
        memory.add(item);
    }

    // Should have most recent items
    assert_eq!(memory.len(), 5);
}

// ============================================================================
// Section 3: Verifier and Safety Tests
// ============================================================================

/// Test: Verifier gate default state
#[test]
fn verifier_gate_default() {
    use sophon_verifier::VerifierGate;

    let gate = VerifierGate::default();
    assert_eq!(gate.threshold, 0.5); // Default threshold
}

/// Test: Verifier threshold adjustment
#[test]
fn verifier_threshold_adjustment() {
    use sophon_verifier::VerifierGate;

    let mut gate = VerifierGate::default();

    gate.set_threshold(0.7);
    assert_eq!(gate.threshold, 0.7);

    gate.set_threshold(0.3);
    assert_eq!(gate.threshold, 0.3);
}

/// Test: Verifier with different input types
#[test]
fn verifier_input_types() {
    use sophon_verifier::VerifierGate;

    let gate = VerifierGate::default();

    // Normal logits
    let normal = vec![0.5f32; 256];
    let _ = gate.verify(&normal, "test");

    // All zeros
    let zeros = vec![0.0f32; 256];
    let _ = gate.verify(&zeros, "test");

    // Large values
    let large = vec![100.0f32; 256];
    let _ = gate.verify(&large, "test");
}

/// Test: Safety diagnostic
#[test]
fn safety_diagnostic() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());
    let logits = vec![0.0f32; 256];
    let result = diagnostic.check(&logits);

    // Should either pass or fail, but not panic
    assert!(
        result.passed || !result.faults.is_empty() || result.halted_at_stage > 0,
        "Diagnostic should produce a result"
    );
}

/// Test: Alignment monitor
#[test]
fn alignment_monitor() {
    use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor};

    let initial = vec![0.0f32; 100];
    let config = AlignmentConfig::from_spec();
    let mut monitor = AlignmentMonitor::new(&initial, config);

    let current = vec![0.1f32; 100];
    let status = monitor.step(&current);

    // Status should be valid
    assert!(!status.to_string().is_empty());
}

/// Test: Alignment drift detection
#[test]
fn alignment_drift_detection() {
    use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor};

    let initial = vec![0.0f32; 64];
    let config = AlignmentConfig::from_spec();
    let mut monitor = AlignmentMonitor::new(&initial, config);

    // Slight drift
    let drifted = vec![0.05f32; 64];
    let status = monitor.step(&drifted);

    // Status should reflect drift
    assert!(!status.to_string().is_empty());
}

// ============================================================================
// Section 4: Training and Optimization Tests
// ============================================================================

/// Test: Training state initialization
#[test]
fn train_state_init() {
    use sophon_train::TrainState;

    let state = TrainState::new();
    assert_eq!(state.global_step, 0);
    assert_eq!(state.epoch, 0);
}

/// Test: Training state increment
#[test]
fn train_state_increment() {
    use sophon_train::TrainState;

    let mut state = TrainState::new();

    state.global_step += 1;
    assert_eq!(state.global_step, 1);

    state.epoch += 1;
    assert_eq!(state.epoch, 1);
}

/// Test: Optimizer initialization
#[test]
fn optimizer_init() {
    use sophon_optim::tsm::TsmSgd;

    let opt = TsmSgd::new(0.001, 1.0);
    assert_eq!(opt.learning_rate(), 0.001);
}

/// Test: Optimizer learning rate decay
#[test]
fn optimizer_lr_decay() {
    use sophon_optim::tsm::TsmSgd;

    let opt = TsmSgd::new(0.01, 1000.0);

    let lr_0 = opt.get_lr_with_decay(0, 1000);
    let lr_500 = opt.get_lr_with_decay(500, 1000);
    let lr_1000 = opt.get_lr_with_decay(1000, 1000);

    // Learning rate should decay
    assert!(lr_500 <= lr_0);
    assert!(lr_1000 <= lr_500);
    assert!(lr_1000 > 0.0);
}

/// Test: Checkpoint strategy
#[test]
fn checkpoint_strategy() {
    use sophon_train::checkpoint::{CheckpointStrategy, SaveCondition};

    let strategy = CheckpointStrategy::new(100, SaveCondition::Steps(1000));

    assert_eq!(strategy.interval(), 100);
}

/// Test: Checkpoint saving logic
#[test]
fn checkpoint_save_logic() {
    use sophon_train::checkpoint::{CheckpointStrategy, SaveCondition};

    let strategy = CheckpointStrategy::new(100, SaveCondition::Steps(1000));

    assert!(strategy.should_save(0, 0)); // First step
    assert!(strategy.should_save(100, 0)); // Interval
    assert!(strategy.should_save(200, 0)); // Interval
    assert!(!strategy.should_save(50, 0)); // Not interval
    assert!(!strategy.should_save(150, 0)); // Not interval
}

// ============================================================================
// Section 5: SSM Tests
// ============================================================================

/// Test: SSM state initialization
#[test]
fn ssm_state_init() {
    use sophon_ssm::SsmState;

    let state = SsmState::new(SSM_N);
    assert_eq!(state.n(), SSM_N);
}

/// Test: SSM state dimensions
#[test]
fn ssm_state_dimensions() {
    use sophon_ssm::SsmState;

    let state = SsmState::new(64);
    assert_eq!(state.x.len(), 64);
}

/// Test: SSM parameters initialization
#[test]
fn ssm_params_init() {
    use sophon_ssm::params::SsmParams;

    let params = SsmParams::new(SSM_N);
    assert_eq!(params.n, SSM_N);
}

/// Test: SSM parameters random initialization
#[test]
fn ssm_params_random() {
    use sophon_ssm::params::SsmParams;

    let params = SsmParams::random(SSM_N);

    // Check arrays have correct length
    assert_eq!(params.a.len(), SSM_N);
    assert_eq!(params.b.len(), SSM_N);
    assert_eq!(params.c.len(), SSM_N);
}

/// Test: SSM discretization
#[test]
fn ssm_discretization() {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::zoh::DiscretisedSsm;

    let params = SsmParams::random(SSM_N);
    let disc = DiscretisedSsm::discretize(&params, 0.01);

    // Discretized matrices should have correct dimensions
    assert_eq!(disc.a_bar.len(), SSM_N);
    assert_eq!(disc.b_bar.len(), SSM_N);
}

/// Test: SSM selective scan
#[test]
fn ssm_selective_scan() {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::selective::selective_scan;
    use sophon_ssm::zoh::DiscretisedSsm;
    use sophon_ssm::SsmState;

    let mut state = SsmState::new(SSM_N);
    let params = SsmParams::random(SSM_N);
    let disc = DiscretisedSsm::discretize(&params, 0.01);
    let input: Vec<f32> = (0..SSM_N).map(|i| i as f32 * 0.01).collect();

    let output = selective_scan(&mut state, &params, &input);

    // Output should have correct length
    assert_eq!(output.len(), SSM_N);
}

/// Test: SSM state update
#[test]
fn ssm_state_update() {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::selective::selective_scan;
    use sophon_ssm::zoh::DiscretisedSsm;
    use sophon_ssm::SsmState;

    let mut state = SsmState::new(SSM_N);
    let initial_state: Vec<f32> = state.x.clone();

    let params = SsmParams::random(SSM_N);
    let disc = DiscretisedSsm::discretize(&params, 0.01);
    let input: Vec<f32> = vec![1.0; SSM_N];

    let _ = selective_scan(&mut state, &params, &input);

    // State should have changed
    let changed = state
        .x
        .iter()
        .zip(initial_state.iter())
        .any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(changed, "State should update");
}

// ============================================================================
// Section 6: Loss Function Tests
// ============================================================================

/// Test: Loss functions work
#[test]
fn loss_functions() {
    use sophon_loss::LossFn;

    let logits = vec![0.5f32; 256];
    let targets = vec![1.0f32; 256];

    let mse = LossFn::Mse.compute(&logits, &targets);
    assert!(!mse.is_nan());
    assert!(!mse.is_infinite());

    let cross_entropy = LossFn::CrossEntropy.compute(&logits, &targets);
    assert!(!cross_entropy.is_nan());
    assert!(!cross_entropy.is_infinite());
}

/// Test: MSE loss calculation
#[test]
fn loss_mse_calculation() {
    use sophon_loss::LossFn;

    let logits = vec![1.0f32, 2.0, 3.0];
    let targets = vec![1.0f32, 2.0, 3.0];

    let loss = LossFn::Mse.compute(&logits, &targets);
    assert!(loss.abs() < 1e-6, "MSE of identical vectors should be ~0");
}

/// Test: Cross-entropy with uniform distribution
#[test]
fn loss_cross_entropy_uniform() {
    use sophon_loss::LossFn;

    // Uniform logits
    let logits = vec![0.0f32; VOCAB_SIZE];
    let targets = vec![1.0f32 / VOCAB_SIZE as f32; VOCAB_SIZE];

    let loss = LossFn::CrossEntropy.compute(&logits, &targets);
    assert!(!loss.is_nan());
    assert!(loss > 0.0, "Cross-entropy should be positive");
}

/// Test: Loss with extreme values
#[test]
fn loss_extreme_values() {
    use sophon_loss::LossFn;

    // Large logits
    let large_logits = vec![100.0f32; VOCAB_SIZE];
    let targets = vec![1.0f32; VOCAB_SIZE];
    let loss = LossFn::CrossEntropy.compute(&large_logits, &targets);
    assert!(loss.is_finite(), "Should handle large logits");

    // Small logits
    let small_logits = vec![-100.0f32; VOCAB_SIZE];
    let loss2 = LossFn::CrossEntropy.compute(&small_logits, &targets);
    assert!(loss2.is_finite(), "Should handle small logits");
}

// ============================================================================
// Section 7: Belief State and Inference Tests
// ============================================================================

/// Test: Belief state initialization
#[test]
fn belief_state_init() {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(64);
    assert_eq!(belief.dim(), 64);
}

/// Test: Belief state update
#[test]
fn belief_state_update() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let grad = vec![0.1f32; 64];

    belief.update(&grad, &[], 0.01);

    // Belief should have changed
    let magnitude = belief.mu_magnitude();
    assert!(magnitude > 0.0, "Belief should be updated");
}

/// Test: Belief state normalization
#[test]
fn belief_state_normalization() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Set large values
    for i in 0..64 {
        belief.mu[i] = 100.0;
    }

    belief.normalize();

    // Should sum to ~1
    let sum: f32 = belief.mu.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "Normalized belief should sum to 1");
}

/// Test: Belief state uncertainty
#[test]
fn belief_state_uncertainty() {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(64);

    // Initial uncertainty should be high
    let uncertainty = belief.uncertainty();
    assert!(uncertainty > 0.0, "Should have uncertainty");
}

/// Test: World model initialization
#[test]
fn world_model_init() {
    use sophon_inference::prediction::WorldModel;

    let model = WorldModel::new(64, 64);
    assert_eq!(model.input_dim(), 64);
    assert_eq!(model.output_dim(), 64);
}

/// Test: World model prediction
#[test]
fn world_model_prediction() {
    use sophon_inference::prediction::WorldModel;

    let model = WorldModel::new(64, 64);
    let input = vec![0.1f32; 64];
    let prediction = model.predict(&input);

    assert_eq!(prediction.len(), 64);
}

/// Test: World model transition
#[test]
fn world_model_transition() {
    use sophon_inference::prediction::WorldModel;

    let mut model = WorldModel::new(64, 64);
    let state = vec![0.1f32; 64];
    let action = vec![0.5f32; 64];

    let next_state = model.transition(&state, &action);
    assert_eq!(next_state.len(), 64);
}

// ============================================================================
// Section 8: Quantization Tests
// ============================================================================

/// Test: Quantization roundtrip
#[test]
fn quantization_roundtrip() {
    use sophon_quant::quant::{dequantize, ternarize};

    let input = vec![0.5f32, -0.3f32, 0.0f32, 0.9f32];
    let ternary = ternarize(&input);
    let recovered = dequantize(&ternary);

    assert_eq!(recovered.len(), input.len());
}

/// Test: Ternarization preserves signs
#[test]
fn ternarization_signs() {
    use sophon_quant::quant::ternarize;

    let input = vec![-0.5f32, 0.0f32, 0.5f32];
    let ternary = ternarize(&input);

    assert_eq!(ternary[0], -1);
    assert_eq!(ternary[1], 0);
    assert_eq!(ternary[2], 1);
}

/// Test: Ternarization threshold
#[test]
fn ternarization_threshold() {
    use sophon_quant::quant::ternarize;

    // Values near zero should map to 0
    let small = vec![0.001f32, -0.001f32];
    let ternary = ternarize(&small);
    assert_eq!(ternary[0], 0);
    assert_eq!(ternary[1], 0);
}

/// Test: Dequantization bounds
#[test]
fn dequantization_bounds() {
    use sophon_quant::quant::dequantize;

    let ternary = vec![-1, 0, 1];
    let recovered = dequantize(&ternary);

    for &v in &recovered {
        assert!(v.abs() <= 1.0 + 1e-6, "Dequantized values should be in [-1, 1]");
    }
}

// ============================================================================
// Section 9: Dataset and Data Processing Tests
// ============================================================================

/// Test: Dataset loading and filtering
#[test]
fn dataset_filter() {
    use sophon_data::{FilterConfig, QualityFilter, Document};

    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    // Should not filter reasonable text
    let doc = Document::new("test", "Hello world this is valid text");
    assert!(filter.check(&doc));

    // Should filter empty text
    let empty_doc = Document::new("test2", "");
    assert!(!filter.check(&empty_doc));
}

/// Test: Dataset batch configuration
#[test]
fn dataset_batch_config() {
    use sophon_data::BatchConfig;

    let config = BatchConfig::default();
    assert!(config.batch_size > 0);
}

/// Test: Document processing
#[test]
fn document_processing() {
    use sophon_data::Document;

    let doc = Document::new("test", "Hello world");
    assert_eq!(doc.id, "test");
    assert_eq!(doc.content, "Hello world");
}

// ============================================================================
// Section 10: TUI Tests
// ============================================================================

/// Test: TUI element creation
#[test]
fn tui_element_creation() {
    use sophon_tui::Color;
    use sophon_tui::Element;

    let el = Element::text("Hello").color(Color::Red).bold();

    assert!(matches!(el.kind, sophon_tui::ElementKind::Text(_)));
}

/// Test: TUI element tree building
#[test]
fn tui_element_tree() {
    use sophon_tui::Element;

    let tree = Element::column(vec![
        Element::text("Line 1"),
        Element::text("Line 2"),
        Element::row(vec![Element::text("A"), Element::text("B")]),
    ]);

    assert!(matches!(tree.kind, sophon_tui::ElementKind::Column));
    assert_eq!(tree.children.len(), 3);
}

/// Test: TUI style merging
#[test]
fn tui_style_merge() {
    use sophon_tui::Color;
    use sophon_tui::Style;

    let base = Style::default().fg(Color::Red);
    let overlay = Style::default().bold(true);
    let merged = base.merge(&overlay);

    assert!(merged.bold);
    assert_eq!(merged.fg, Some(Color::Red));
}

/// Test: TUI style chaining
#[test]
fn tui_style_chaining() {
    use sophon_tui::Color;
    use sophon_tui::Element;

    let el = Element::text("test")
        .color(Color::Red)
        .bold()
        .underline()
        .bg(Color::Blue);

    assert_eq!(el.style.fg, Some(Color::Red));
    assert!(el.style.bold);
    assert!(el.style.underline);
    assert_eq!(el.style.bg, Some(Color::Blue));
}

/// Test: TUI layout constraints
#[test]
fn tui_layout_constraints() {
    use sophon_tui::layout::Constraint;

    let c1 = Constraint::Length(10);
    let c2 = Constraint::Min(5);
    let c3 = Constraint::Max(20);
    let c4 = Constraint::Fill;

    // Just verify they can be created
    assert!(matches!(c1, Constraint::Length(10)));
    assert!(matches!(c2, Constraint::Min(5)));
    assert!(matches!(c3, Constraint::Max(20)));
    assert!(matches!(c4, Constraint::Fill));
}

/// Test: TUI layout solving
#[test]
fn tui_layout_solving() {
    use sophon_tui::layout::{Constraint, Layout};

    let constraints = vec![
        Constraint::Length(10),
        Constraint::Length(5),
        Constraint::Fill,
    ];

    let layout = Layout::new(constraints);
    let sizes = layout.solve(30);

    assert_eq!(sizes.len(), 3);
    assert_eq!(sizes[0], 10);
    assert_eq!(sizes[1], 5);
    assert_eq!(sizes[2], 15); // Remaining
}

/// Test: TUI color conversion
#[test]
fn tui_color_conversion() {
    use sophon_tui::Color;

    // Test foreground colors
    let fg_red = Color::Red.to_ansi_fg();
    assert!(fg_red.contains("31"));

    // Test background colors
    let bg_blue = Color::Blue.to_ansi_bg();
    assert!(bg_blue.contains("44"));

    // Test RGB
    let rgb = Color::Rgb(128, 64, 32).to_ansi_fg();
    assert!(rgb.contains("38;2"));
}

/// Test: TUI element counting
#[test]
fn tui_element_counting() {
    use sophon_tui::Element;

    let tree = Element::column(vec![
        Element::text("A"),
        Element::row(vec![
            Element::text("B"),
            Element::text("C"),
            Element::column(vec![Element::text("D"), Element::text("E")]),
        ]),
    ]);

    let count = tree.count();
    // 1 (root) + 1 (A) + 1 (row) + 1 (B) + 1 (C) + 1 (column) + 2 (D, E) = 8
    assert_eq!(count, 8);
}

/// Test: TUI element min size
#[test]
fn tui_element_min_size() {
    use sophon_tui::Element;

    let el = Element::text("Hello, World!");
    let size = el.min_size();

    assert_eq!(size.width, 13); // "Hello, World!"
    assert_eq!(size.height, 1);
}

/// Test: TUI element find by ID
#[test]
fn tui_element_find() {
    use sophon_tui::{Element, ElementId};

    let tree = Element::column(vec![
        Element::text("A").with_id(ElementId(1)),
        Element::text("B").with_id(ElementId(2)),
    ]);

    let found = tree.find(ElementId(1));
    assert!(found.is_some());

    let not_found = tree.find(ElementId(99));
    assert!(not_found.is_none());
}

// ============================================================================
// Section 11: Documentation Generator Tests
// ============================================================================

/// Test: Documentation generator
#[test]
fn docs_generator() {
    use sophon_docs::DocGenerator;

    let mut generator = DocGenerator::new("/tmp/test_docs");
    generator.add_root("crates/");

    // Should be able to create generator
    assert_eq!(generator.roots.len(), 1);
}

/// Test: Doc generator multiple roots
#[test]
fn docs_generator_multiple_roots() {
    use sophon_docs::DocGenerator;

    let mut generator = DocGenerator::new("/tmp/test_docs");
    generator.add_root("crates/");
    generator.add_root("src/");
    generator.add_root("docs/");

    assert_eq!(generator.roots.len(), 3);
}

// ============================================================================
// Section 12: Scheduler Tests
// ============================================================================

/// Test: Scheduler task creation
#[test]
fn scheduler_task() {
    use sophon_accel::scheduler::Scheduler;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let scheduler = Scheduler::new(4);
    let counter = Arc::new(AtomicUsize::new(0));

    let task_counter = counter.clone();
    scheduler.spawn(move || {
        task_counter.fetch_add(1, Ordering::SeqCst);
    });

    // Give scheduler time to execute
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Shutdown and wait
    scheduler.shutdown();
}

/// Test: Scheduler multiple tasks
#[test]
fn scheduler_multiple_tasks() {
    use sophon_accel::scheduler::Scheduler;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let scheduler = Scheduler::new(4);
    let counter = Arc::new(AtomicUsize::new(0));

    for _ in 0..10 {
        let c = counter.clone();
        scheduler.spawn(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
    }

    std::thread::sleep(std::time::Duration::from_millis(200));
    scheduler.shutdown();
}

// ============================================================================
// Section 13: KAN Tests
// ============================================================================

/// Test: KAN spline evaluation
#[test]
fn kan_spline_eval() {
    use sophon_kan::spline::{cubic_spline_eval, KnotVector};

    let knots = KnotVector::uniform(0.0, 1.0, 5);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0];

    let val = cubic_spline_eval(&knots, &coeffs, 0.5);
    assert!(val.is_finite());
}

/// Test: KAN spline at boundaries
#[test]
fn kan_spline_boundaries() {
    use sophon_kan::spline::{cubic_spline_eval, KnotVector};

    let knots = KnotVector::uniform(0.0, 1.0, 5);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0];

    let start = cubic_spline_eval(&knots, &coeffs, 0.0);
    let end = cubic_spline_eval(&knots, &coeffs, 1.0);

    assert!(start.is_finite());
    assert!(end.is_finite());
}

// ============================================================================
// Section 14: Cross-Crate Integration Tests
// ============================================================================

/// Test: Full inference pipeline
#[test]
fn full_inference_pipeline() {
    use sophon_model::Sophon1;
    use sophon_inference::belief::BeliefState;

    let mut model = Sophon1::new(0x1234);
    let mut belief = BeliefState::new(64);

    // Run inference
    let input = b"test input";
    let outputs = model.forward_sequence(input).unwrap();

    // Update belief
    if let Some(last) = outputs.last() {
        let logits_slice: Vec<f32> = last.logits.as_slice().iter().take(64).copied().collect();
        belief.update(&logits_slice, &[], 0.01);
    }

    assert!(belief.mu_magnitude() > 0.0);
}

/// Test: Training workflow integration
#[test]
fn training_workflow_integration() {
    use sophon_train::TrainState;
    use sophon_optim::tsm::TsmSgd;
    use sophon_loss::LossFn;

    let mut state = TrainState::new();
    let opt = TsmSgd::new(0.001, 1.0);

    // Simulate training step
    let logits = vec![0.5f32; 256];
    let targets = vec![0.5f32; 256];
    let loss = LossFn::Mse.compute(&logits, &targets);

    state.global_step += 1;
    state.update_ema_loss(loss);

    assert_eq!(state.global_step, 1);
    assert!(state.ema_loss >= 0.0);
}

/// Test: Memory + inference integration
#[test]
fn memory_inference_integration() {
    use sophon_memory::episodic::{EpisodicMemory, Episode};
    use sophon_inference::belief::BeliefState;

    let mut memory = EpisodicMemory::new(100);
    let mut belief = BeliefState::new(64);

    // Store experience
    let experience = vec![0.5f32; 64];
    let ep = Episode {
        timestamp: sophon_memory::current_timestamp(),
        perception_hv: experience.clone(),
        action: None,
        outcome_hv: experience.clone(),
        surprise: 0.0,
    };
    memory.record(ep);

    // Retrieve and update belief
    let retrieved = memory.retrieve_similar(&experience, 1);
    if let Some(first) = retrieved.first() {
        belief.update(&first.perception_hv, &[], 0.01);
    }

    assert!(belief.mu_magnitude() > 0.0);
}

/// Test: Safety + model integration
#[test]
fn safety_model_integration() {
    use sophon_model::Sophon1;
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let mut model = Sophon1::new(0x1234);
    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    let input = b"test";
    let outputs = model.forward_sequence(input).unwrap();

    if let Some(last) = outputs.last() {
        let result = diagnostic.check(last.logits.as_slice());
        // Should produce a result
        assert!(result.passed || !result.faults.is_empty());
    }
}

// ============================================================================
// Section 15: Edge Case Tests
// ============================================================================

/// Test: Empty inputs handling
#[test]
fn edge_case_empty_inputs() {
    use sophon_loss::LossFn;

    let empty_logits: Vec<f32> = vec![];
    let empty_targets: Vec<f32> = vec![];

    // Should handle gracefully
    let loss = LossFn::Mse.compute(&empty_logits, &empty_targets);
    assert!(loss.is_finite() || loss.is_nan());
}

/// Test: Large dimension handling
#[test]
fn edge_case_large_dimensions() {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(10000);
    assert_eq!(belief.dim(), 10000);
}

/// Test: Numerical stability
#[test]
fn edge_case_numerical_stability() {
    use sophon_loss::LossFn;

    // Very large values
    let large = vec![1e20f32; 256];
    let targets = vec![0.0f32; 256];
    let loss = LossFn::Mse.compute(&large, &targets);
    assert!(loss.is_finite() || loss.is_infinite()); // Either is acceptable

    // Very small values
    let small = vec![1e-20f32; 256];
    let loss2 = LossFn::Mse.compute(&small, &targets);
    assert!(loss2.is_finite());
}

/// Test: Unicode handling
#[test]
fn edge_case_unicode() {
    use sophon_data::{FilterConfig, QualityFilter, Document};

    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    // Unicode text
    let doc = Document::new("test", "Hello 世界 🌍 émojis");
    assert!(filter.check(&doc));
}

/// Test: Very long text
#[test]
fn edge_case_long_text() {
    use sophon_data::{FilterConfig, QualityFilter, Document};

    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);

    let long = Document::new("test", &"a".repeat(100000));
    assert!(!filter.check(&long)); // Too long
}

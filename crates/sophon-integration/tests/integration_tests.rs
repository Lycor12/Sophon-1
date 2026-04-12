//! Integration Tests for Sophon AGI System
//!
//! Tests the interaction between multiple crates and full workflows.

use sophon_config::ModelConfig;
use sophon_model::Sophon1;
use sophon_runtime::system;

/// Test: Model can be created and produces outputs
#[test]
fn model_creation_and_forward() {
    let model = Sophon1::new(0xDEAD_BEEF_u64);
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
}

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
    memory.add_episode(observation.clone());

    assert_eq!(memory.len(), 1);

    // Retrieve
    let retrieved = memory.retrieve_episodes(&observation, 1);
    assert_eq!(retrieved.len(), 1);
}

/// Test: Verifier gate default state
#[test]
fn verifier_gate_default() {
    use sophon_verifier::VerifierGate;

    let gate = VerifierGate::default();
    assert_eq!(gate.threshold, 0.5); // Default threshold
}

/// Test: Training state initialization
#[test]
fn train_state_init() {
    use sophon_train::TrainState;

    let state = TrainState::new();
    assert_eq!(state.global_step, 0);
    assert_eq!(state.epoch, 0);
}

/// Test: SSM state initialization
#[test]
fn ssm_state_init() {
    use sophon_config::SSM_N;
    use sophon_ssm::SsmState;

    let state = SsmState::new(SSM_N);
    assert_eq!(state.n(), SSM_N);
}

/// Test: Loss functions work
#[test]
fn loss_functions() {
    use sophon_loss::LossFn;

    let logits = vec![0.5f32; 256];
    let targets = vec![1.0f32; 256];

    let cross_entropy = LossFn::CrossEntropy.compute(&logits, &targets);
    assert!(!cross_entropy.is_nan());
    assert!(!cross_entropy.is_infinite());
}

/// Test: Optimizer initialization
#[test]
fn optimizer_init() {
    use sophon_optim::tsm::TsmSgd;

    let opt = TsmSgd::new(0.001, 1.0);
    assert_eq!(opt.learning_rate(), 0.001);
}

/// Test: Belief state initialization
#[test]
fn belief_state_init() {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(64);
    assert_eq!(belief.dim(), 64);
}

/// Test: World model initialization
#[test]
fn world_model_init() {
    use sophon_inference::prediction::WorldModel;

    let model = WorldModel::new(64, 64);
    assert_eq!(model.input_dim(), 64);
    assert_eq!(model.output_dim(), 64);
}

/// Test: Safety diagnostic
#[test]
fn safety_diagnostic() {
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());
    let logits = vec![0.0f32; 256];
    let result = diagnostic.check(&logits);

    // Should either pass or fail, but not panic
    assert!(
        result.passed || !result.faults.is_empty() || result.halted_at_stage.is_some(),
        "Diagnostic should produce a result"
    );
}

/// Test: Alignment monitor
#[test]
fn alignment_monitor() {
    use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor};

    let initial = vec![0.0f32; 100];
    let config = AlignmentConfig::from_spec();
    let monitor = AlignmentMonitor::new(&initial, config);

    let current = vec![0.1f32; 100];
    let status = monitor.step(&current);

    // Status should be valid
    assert!(!status.to_string().is_empty());
}

/// Test: Dataset loading and filtering
#[test]
fn dataset_filter() {
    use sophon_data::FilterConfig;

    let config = FilterConfig::default();

    // Should not filter reasonable text
    let text = "Hello world this is valid text";
    assert!(!config.should_skip(text));

    // Should filter empty text
    assert!(config.should_skip(""));
}

/// Test: Checkpoint strategy
#[test]
fn checkpoint_strategy() {
    use sophon_train::checkpoint::{CheckpointStrategy, SaveCondition};

    let strategy = CheckpointStrategy::new(100, SaveCondition::Steps(1000));

    assert_eq!(strategy.interval(), 100);
}

/// Test: Quantization
#[test]
fn quantization_roundtrip() {
    use sophon_quant::quant::{dequantize, ternarize};

    let input = vec![0.5f32, -0.3f32, 0.0f32, 0.9f32];
    let ternary = ternarize(&input);
    let recovered = dequantize(&ternary);

    assert_eq!(recovered.len(), input.len());
}

/// Test: TUI element creation
#[test]
fn tui_element_creation() {
    use sophon_tui::Color;
    use sophon_tui::Element;

    let el = Element::text("Hello").color(Color::Red).bold();

    assert!(matches!(el.kind, sophon_tui::ElementKind::Text(_)));
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

/// Test: Documentation generation
#[test]
fn docs_generator() {
    use sophon_docs::DocGenerator;

    let mut generator = DocGenerator::new("/tmp/test_docs");
    generator.add_root("crates/");

    // Should be able to create generator
    assert_eq!(generator.roots.len(), 1);
}

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

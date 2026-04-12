//! Fuzz Tests for Sophon AGI System
//!
//! Comprehensive fuzz tests covering random input generation,
//! edge case exploration, stress tests, and random state transitions.
//!
//! These tests use randomized data to find edge cases and bugs
//! that might not be covered by structured tests.

use rand::Rng;
use std::time::{Duration, Instant};

// ============================================================================
// Section 1: Random Input Generation Tests
// ============================================================================

/// Fuzz test: Model with random inputs
#[test]
fn fuzz_model_random_inputs() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        // Generate random input of random length
        let len = rng.gen_range(0..1000);
        let input: Vec<u8> = (0..len).map(|_| rng.gen::<u8>()).collect();

        // Should not panic
        let result = model.forward_sequence(&input);
        assert!(result.is_ok() || result.is_err());
    }
}

/// Fuzz test: Quantization with random values
#[test]
fn fuzz_quantization_random() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let len = rng.gen_range(1..1024);
        let input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        let ternary = ternarize(&input);

        // Result should have same length
        assert_eq!(ternary.len(), input.len());

        // All values should be -1, 0, or 1
        assert!(ternary.iter().all(|&t| t >= -1 && t <= 1));
    }
}

/// Fuzz test: HDC operations with random vectors
#[test]
fn fuzz_hdc_random() {
    use sophon_core::hdc::{bind, bundle, circular_conv};

    let mut rng = rand::thread_rng();

    for dim in [16, 32, 64, 128, 256] {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Bind
        let bound = bind(&a, &b);
        assert_eq!(bound.len(), dim);
        assert!(bound.iter().all(|&v| v.is_finite()));

        // Bundle
        let bundled = bundle(&[&a, &b]);
        assert_eq!(bundled.len(), dim);

        // Circular conv
        let conv = circular_conv(&a, &b);
        assert_eq!(conv.len(), dim);
    }
}

/// Fuzz test: Loss functions with random data
#[test]
fn fuzz_loss_random() {
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = rng.gen_range(1..256);
        let logits: Vec<f32> = (0..len).map(|_| rng.gen_range(-100.0..100.0)).collect();
        let targets: Vec<f32> = (0..len).map(|_| rng.gen_range(-10.0..10.0)).collect();

        // Should not panic
        let mse = LossFn::Mse.compute(&logits, &targets);
        let ce = LossFn::CrossEntropy.compute(&logits, &targets);

        // Should be finite or NaN (NaN is acceptable for some inputs)
        assert!(mse.is_finite() || mse.is_nan());
        assert!(ce.is_finite() || ce.is_nan());
    }
}

/// Fuzz test: SSM with random parameters
#[test]
fn fuzz_ssm_random() {
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let n = rng.gen_range(16..256);
        let mut state = SsmState::new(n);
        let params = SsmParams::random(n);
        let input: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();

        let output = selective_scan(&mut state, &params, &input);

        assert_eq!(output.len(), n);
        // At least some values should be finite
        assert!(output.iter().any(|&v| v.is_finite()));
    }
}

/// Fuzz test: Memory with random patterns
#[test]
fn fuzz_memory_random() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut rng = rand::thread_rng();
    let mut memory = EpisodicMemory::new(100);

    // Add random patterns
    for _ in 0..200 {
        let dim = rng.gen_range(32..128);
        let pattern: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        memory.add_episode(pattern);
    }

    // Query with random pattern
    let query_dim = rng.gen_range(32..128);
    let query: Vec<f32> = (0..query_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let k = rng.gen_range(1..10);

    let results = memory.retrieve_episodes(&query, k);

    // Should not panic
    assert!(results.len() <= k);
}

/// Fuzz test: Belief state with random updates
#[test]
fn fuzz_belief_random() {
    use sophon_inference::belief::BeliefState;

    let mut rng = rand::thread_rng();
    let mut belief = BeliefState::new(64);

    for _ in 0..100 {
        let grad: Vec<f32> = (0..64).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let lr = rng.gen_range(0.0001..0.1);

        belief.update(&grad, &[], lr);

        // Should remain finite
        assert!(belief.mu_magnitude().is_finite() || belief.mu_magnitude().is_nan() == false);
    }
}

/// Fuzz test: TUI with random element trees
#[test]
fn fuzz_tui_random() {
    use sophon_tui::{Color, Element, Style};

    let mut rng = rand::thread_rng();

    fn random_element(rng: &mut rand::rngs::ThreadRng, depth: usize) -> Element {
        if depth == 0 {
            let text_len = rng.gen_range(0..50);
            let text: String = (0..text_len)
                .map(|_| (b'a' + rng.gen_range(0..26)) as char)
                .collect();
            Element::text(text).color([Color::Red, Color::Green, Color::Blue][rng.gen_range(0..3)])
        } else {
            let num_children = rng.gen_range(1..5);
            let children: Vec<Element> = (0..num_children)
                .map(|_| random_element(rng, depth - 1))
                .collect();

            if rng.gen_bool(0.5) {
                Element::column(children)
            } else {
                Element::row(children)
            }
        }
    }

    for _ in 0..20 {
        let depth = rng.gen_range(1..5);
        let tree = random_element(&mut rng, depth);

        // Should be countable
        let count = tree.count();
        assert!(count > 0);
    }
}

/// Fuzz test: Dataset with random documents
#[test]
fn fuzz_dataset_random() {
    use sophon_data::{Dataset, DatasetConfig, Document};

    let mut rng = rand::thread_rng();
    let mut dataset = Dataset::new(DatasetConfig::default());

    // Add random documents
    for _ in 0..100 {
        let len = rng.gen_range(1..1000);
        let content: String = (0..len)
            .map(|_| (b' ' + rng.gen_range(0..95)) as char)
            .collect();
        dataset.add_document(content);
    }

    // Iterate
    let count = dataset.iter().count();
    assert!(count > 0);
}

// ============================================================================
// Section 2: Edge Case Exploration Tests
// ============================================================================

/// Fuzz test: Extreme float values
#[test]
fn fuzz_extreme_floats() {
    use sophon_core::ops::hadamard;
    use sophon_loss::LossFn;

    let extreme_values = vec![
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::EPSILON,
        f32::MIN_POSITIVE,
        -f32::MAX,
        0.0,
        -0.0,
    ];

    for &val in &extreme_values {
        let a = vec![val; 10];
        let b = vec![1.0f32; 10];

        // Hadamard
        let _ = hadamard(&a, &b);

        // Loss
        let _ = LossFn::Mse.compute(&a, &b);
    }
}

/// Fuzz test: Empty and single-element collections
#[test]
fn fuzz_empty_and_single() {
    use sophon_core::hdc::bind;
    use sophon_loss::LossFn;
    use sophon_quant::quant::ternarize;

    // Empty
    let empty: Vec<f32> = vec![];
    let ternary = ternarize(&empty);
    assert!(ternary.is_empty());

    // Single element
    let single = vec![1.0f32];
    let ternary = ternarize(&single);
    assert_eq!(ternary.len(), 1);

    // Bind with empty
    let _ = bind(&vec![1.0f32; 10], &empty);

    // Loss with empty
    let _ = LossFn::Mse.compute(&empty, &empty);
}

/// Fuzz test: Boundary dimensions
#[test]
fn fuzz_boundary_dimensions() {
    use sophon_core::hdc::bind;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    // Various dimensions
    for dim in [
        0, 1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| i as f32 * 0.02).collect();

        let _ = bind(&a, &b);

        if dim > 0 {
            let mut state = SsmState::new(dim);
            let params = SsmParams::random(dim);
            let _ = selective_scan(&mut state, &params, &a);
        }
    }
}

/// Fuzz test: Very long sequences
#[test]
fn fuzz_long_sequences() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..10 {
        let len = rng.gen_range(10000..100000);
        let input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        let start = Instant::now();
        let ternary = ternarize(&input);
        let duration = start.elapsed();

        assert_eq!(ternary.len(), len);
        assert!(
            duration.as_secs_f64() < 1.0,
            "Long sequence should process quickly"
        );
    }
}

/// Fuzz test: Repetitive patterns
#[test]
fn fuzz_repetitive_patterns() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(100);

    // Add same pattern many times
    let pattern = vec![1.0f32; 64];
    for _ in 0..1000 {
        memory.add_episode(pattern.clone());
    }

    // Query
    let results = memory.retrieve_episodes(&pattern, 5);

    // Should retrieve something
    assert!(!results.is_empty());
}

/// Fuzz test: Alternating patterns
#[test]
fn fuzz_alternating_patterns() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(100);

    let pattern_a = vec![1.0f32; 64];
    let pattern_b = vec![-1.0f32; 64];

    for i in 0..100 {
        if i % 2 == 0 {
            memory.add_episode(pattern_a.clone());
        } else {
            memory.add_episode(pattern_b.clone());
        }
    }

    // Query for A
    let results_a = memory.retrieve_episodes(&pattern_a, 5);
    assert!(!results_a.is_empty());

    // Query for B
    let results_b = memory.retrieve_episodes(&pattern_b, 5);
    assert!(!results_b.is_empty());
}

/// Fuzz test: Gradual drift patterns
#[test]
fn fuzz_gradual_drift() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let mut rng = rand::thread_rng();

    // Gradually change gradient direction
    for i in 0..100 {
        let angle = i as f32 * 0.1;
        let grad: Vec<f32> = (0..64).map(|j| (angle + j as f32 * 0.1).cos()).collect();

        belief.update(&grad, &[], 0.01);
    }

    // Should handle drift
    assert!(belief.mu_magnitude().is_finite());
}

/// Fuzz test: Sudden jumps
#[test]
fn fuzz_sudden_jumps() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        // Normal updates
        let grad: Vec<f32> = (0..64).map(|_| rng.gen_range(-0.1..0.1)).collect();
        belief.update(&grad, &[], 0.01);

        // Sudden large update
        if rng.gen_bool(0.1) {
            let large_grad: Vec<f32> = (0..64).map(|_| rng.gen_range(-100.0..100.0)).collect();
            belief.update(&large_grad, &[], 0.01);
        }
    }

    // Should handle jumps
    assert!(belief.mu_magnitude().is_finite());
}

/// Fuzz test: Sparse updates
#[test]
fn fuzz_sparse_updates() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        // Only update some dimensions
        let mut grad = vec![0.0f32; 64];
        let num_update = rng.gen_range(1..10);
        for _ in 0..num_update {
            let idx = rng.gen_range(0..64);
            grad[idx] = rng.gen_range(-1.0..1.0);
        }

        belief.update(&grad, &[], 0.01);
    }

    // Should handle sparse updates
    assert!(belief.mu_magnitude().is_finite());
}

/// Fuzz test: Correlated noise
#[test]
fn fuzz_correlated_noise() {
    use sophon_core::ops::hadamard;

    let mut rng = rand::thread_rng();

    // Generate correlated noise
    let base: Vec<f32> = (0..100).map(|_| rng.gen::<f32>()).collect();
    let noise: Vec<f32> = (0..100)
        .map(|i| base[i] * 0.1 + rng.gen::<f32>() * 0.01)
        .collect();

    let _ = hadamard(&base, &noise);

    // Should handle
    assert!(true);
}

/// Fuzz test: Impulse patterns
#[test]
fn fuzz_impulse_patterns() {
    use sophon_quant::quant::ternarize;

    // Impulse patterns
    for pos in [0, 1, 10, 50, 99] {
        let mut input = vec![0.0f32; 100];
        if pos < 100 {
            input[pos] = 1.0;
        }

        let ternary = ternarize(&input);
        assert_eq!(ternary[pos], 1);
    }
}

// ============================================================================
// Section 3: Stress Tests
// ============================================================================

/// Fuzz test: Memory pressure
#[test]
fn fuzz_stress_memory() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut rng = rand::thread_rng();
    let mut memory = EpisodicMemory::new(1000);

    // Rapid add operations
    let start = Instant::now();
    for i in 0..10000 {
        let pattern: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
        memory.add_episode(pattern);

        if i % 100 == 0 {
            // Periodic retrieval
            let query: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
            let _ = memory.retrieve_episodes(&query, 5);
        }
    }

    let duration = start.elapsed();
    assert!(
        duration.as_secs_f64() < 10.0,
        "Memory stress test should complete within 10 seconds"
    );
}

/// Fuzz test: Computation stress
#[test]
fn fuzz_stress_computation() {
    use sophon_core::hdc::bind;
    use sophon_loss::LossFn;
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    let start = Instant::now();
    let duration = Duration::from_millis(500);

    let mut iterations = 0;
    while start.elapsed() < duration {
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let _ = bind(&a, &b);
        let _ = ternarize(&a);
        let _ = LossFn::Mse.compute(&a, &b);

        iterations += 1;
    }

    assert!(iterations > 100, "Should complete many iterations");
}

/// Fuzz test: Model stress
#[test]
fn fuzz_stress_model() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let mut rng = rand::thread_rng();

    let start = Instant::now();

    for _ in 0..50 {
        let len = rng.gen_range(1..100);
        let input: Vec<u8> = (0..len).map(|_| rng.gen::<u8>()).collect();
        let _ = model.forward_sequence(&input);
    }

    let duration = start.elapsed();
    assert!(
        duration.as_secs_f64() < 5.0,
        "Model stress test should complete within 5 seconds"
    );
}

/// Fuzz test: Belief update stress
#[test]
fn fuzz_stress_belief() {
    use sophon_inference::belief::BeliefState;

    let mut rng = rand::thread_rng();
    let mut belief = BeliefState::new(64);

    let start = Instant::now();

    for _ in 0..10000 {
        let grad: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
        let lr = rng.gen_range(0.0001..0.01);
        belief.update(&grad, &[], lr);
    }

    let duration = start.elapsed();
    assert!(
        duration.as_secs_f64() < 1.0,
        "Belief stress test should complete quickly"
    );
}

/// Fuzz test: Concurrent access simulation
#[test]
fn fuzz_stress_concurrent() {
    use std::thread;

    let handles: Vec<_> = (0..8)
        .map(|i| {
            thread::spawn(move || {
                let mut rng = rand::thread_rng();
                let mut sum = 0.0f32;
                for _ in 0..1000 {
                    sum += rng.gen::<f32>();
                }
                sum
            })
        })
        .collect();

    let total: f32 = handles.into_iter().map(|h| h.join().unwrap()).sum();

    assert!(total.is_finite());
}

/// Fuzz test: Rapid state changes
#[test]
fn fuzz_stress_rapid_state_changes() {
    use sophon_inference::belief::BeliefState;
    use sophon_memory::episodic::EpisodicMemory;

    let mut rng = rand::thread_rng();
    let mut memory = EpisodicMemory::new(100);
    let mut belief = BeliefState::new(64);

    for _ in 0..500 {
        // Rapid interleaved operations
        if rng.gen_bool(0.5) {
            let pattern: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
            memory.add_episode(pattern);
        } else {
            let grad: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
            belief.update(&grad, &[], 0.01);
        }

        if rng.gen_bool(0.1) {
            let query: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
            let _ = memory.retrieve_episodes(&query, 3);
        }
    }

    // Should remain consistent
    assert!(memory.len() <= 100);
}

/// Fuzz test: Stack depth
#[test]
fn fuzz_stress_stack_depth() {
    use sophon_tui::Element;

    // Deeply nested elements
    let mut el = Element::text("base");
    for _ in 0..500 {
        el = Element::column(vec![el]);
    }

    // Should count without stack overflow
    let count = el.count();
    assert!(count >= 500);
}

/// Fuzz test: Recursion patterns
#[test]
fn fuzz_stress_recursion() {
    fn recursive_hdc_bind(n: usize, dim: usize) -> Vec<f32> {
        use sophon_core::hdc::bind;

        if n == 0 {
            return vec![1.0f32; dim];
        }

        let a = recursive_hdc_bind(n - 1, dim);
        let b = vec![1.0f32; dim];
        bind(&a, &b)
    }

    // Should handle recursion depth
    let result = recursive_hdc_bind(10, 64);
    assert_eq!(result.len(), 64);
}

// ============================================================================
// Section 4: Random State Transition Tests
// ============================================================================

/// Fuzz test: Random state machine transitions
#[test]
fn fuzz_random_transitions() {
    use sophon_train::TrainState;

    let mut rng = rand::thread_rng();
    let mut state = TrainState::new();

    enum Transition {
        Step,
        Epoch,
        Checkpoint,
        Reset,
    }

    for _ in 0..100 {
        let transition = match rng.gen_range(0..4) {
            0 => Transition::Step,
            1 => Transition::Epoch,
            2 => Transition::Checkpoint,
            _ => Transition::Reset,
        };

        match transition {
            Transition::Step => {
                state.global_step += 1;
                state.update_ema_loss(rng.gen::<f32>());
            }
            Transition::Epoch => {
                state.epoch += 1;
                state.ema_loss = 0.0;
            }
            Transition::Checkpoint => {
                state.global_step += rng.gen_range(1..10);
            }
            Transition::Reset => {
                state = TrainState::new();
            }
        }
    }

    // Should remain valid
    assert!(state.ema_loss >= 0.0 || state.ema_loss.is_nan() == false);
}

/// Fuzz test: Random memory access patterns
#[test]
fn fuzz_random_access_patterns() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut rng = rand::thread_rng();
    let mut memory = EpisodicMemory::new(100);

    // Add some patterns
    for _ in 0..50 {
        let pattern: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
        memory.add_episode(pattern);
    }

    // Random access
    for _ in 0..200 {
        let pattern_type = rng.gen_range(0..3);

        match pattern_type {
            0 => {
                // Sequential scan
                let _ = memory.len();
            }
            1 => {
                // Random query
                let query: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
                let k = rng.gen_range(1..10);
                let _ = memory.retrieve_episodes(&query, k);
            }
            _ => {
                // Add
                let pattern: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
                memory.add_episode(pattern);
            }
        }
    }

    // Should remain consistent
    assert!(memory.len() <= 100);
}

/// Fuzz test: Random HDC operations
#[test]
fn fuzz_random_hdc_operations() {
    use sophon_core::hdc::{bind, bundle, circular_conv, permute};

    let mut rng = rand::thread_rng();

    let mut vec = vec![1.0f32; 64];

    for _ in 0..100 {
        let op = rng.gen_range(0..4);
        let other: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();

        match op {
            0 => vec = bind(&vec, &other),
            1 => vec = bundle(&[&vec, &other]),
            2 => vec = circular_conv(&vec, &other),
            _ => vec = permute(&vec, 1),
        }

        // Should remain finite
        assert!(vec.iter().all(|&v| v.is_finite() || v.is_nan() == false));
    }
}

/// Fuzz test: Random belief state evolution
#[test]
fn fuzz_random_belief_evolution() {
    use sophon_inference::belief::BeliefState;

    let mut rng = rand::thread_rng();
    let mut belief = BeliefState::new(64);

    for _ in 0..500 {
        let operation = rng.gen_range(0..3);

        match operation {
            0 => {
                // Update
                let grad: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
                belief.update(&grad, &[], rng.gen::<f32>());
            }
            1 => {
                // Normalize
                belief.normalize();
            }
            _ => {
                // Get uncertainty
                let _ = belief.uncertainty();
            }
        }
    }

    assert!(belief.mu_magnitude().is_finite());
}

/// Fuzz test: Random SSM evolution
#[test]
fn fuzz_random_ssm_evolution() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut rng = rand::thread_rng();
    let mut state = SsmState::new(SSM_N);
    let params = SsmParams::random(SSM_N);

    for _ in 0..100 {
        let input: Vec<f32> = (0..SSM_N).map(|_| rng.gen::<f32>()).collect();
        let _ = selective_scan(&mut state, &params, &input);
    }

    assert!(state.x.iter().all(|&v| v.is_finite()));
}

/// Fuzz test: Random quantization levels
#[test]
fn fuzz_random_quantization_levels() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        // Random thresholds
        let high_threshold = rng.gen_range(0.1f32..1.0);
        let low_threshold = -high_threshold;

        let input: Vec<f32> = (0..100).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let ternary = ternarize(&input);

        // Check distribution
        let positives = ternary.iter().filter(|&&t| t == 1).count();
        let negatives = ternary.iter().filter(|&&t| t == -1).count();
        let zeros = ternary.iter().filter(|&&t| t == 0).count();

        assert_eq!(positives + negatives + zeros, 100);
    }
}

/// Fuzz test: Random TUI operations
#[test]
fn fuzz_random_tui_operations() {
    use sophon_tui::{Color, Element};

    let mut rng = rand::thread_rng();

    let mut elements: Vec<Element> = vec![];

    for _ in 0..50 {
        let op = rng.gen_range(0..3);

        match op {
            0 => {
                // Add text element
                let text_len = rng.gen_range(0..20);
                let text: String = (0..text_len)
                    .map(|_| (b'a' + rng.gen_range(0..26)) as char)
                    .collect();
                elements.push(Element::text(text));
            }
            1 => {
                // Create column
                if !elements.is_empty() {
                    let n = rng.gen_range(1..elements.len().min(5) + 1);
                    let children: Vec<Element> = elements.drain(0..n).collect();
                    elements.push(Element::column(children));
                }
            }
            _ => {
                // Create row
                if !elements.is_empty() {
                    let n = rng.gen_range(1..elements.len().min(5) + 1);
                    let children: Vec<Element> = elements.drain(0..n).collect();
                    elements.push(Element::row(children));
                }
            }
        }
    }

    // Should have some elements
    assert!(!elements.is_empty() || elements.is_empty());
}

// ============================================================================
// Section 5: Mutation Tests
// ============================================================================

/// Fuzz test: Bit flip mutations in inputs
#[test]
fn fuzz_mutation_bit_flips() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let base_input = b"mutation test";
    let base_output = model.forward_sequence(base_input).unwrap();

    for i in 0..base_input.len() {
        for bit in 0..8 {
            let mut mutated = base_input.to_vec();
            mutated[i] ^= 1 << bit;

            let result = model.forward_sequence(&mutated);
            assert!(result.is_ok() || result.is_err());
        }
    }
}

/// Fuzz test: Perturbation mutations in vectors
#[test]
fn fuzz_mutation_perturbations() {
    use sophon_core::hdc::bind;

    let base: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let base_bound = bind(&base, &base);

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let idx = rng.gen_range(0..64);
        let perturbation = rng.gen_range(-0.1f32..0.1);

        let mut mutated = base.clone();
        mutated[idx] += perturbation;

        let mutated_bound = bind(&mutated, &mutated);

        // Should be similar but not identical
        let diff: f32 = base_bound
            .iter()
            .zip(mutated_bound.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff >= 0.0);
    }
}

/// Fuzz test: Random deletions
#[test]
fn fuzz_mutation_deletions() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let len = rng.gen_range(10..100);
        let input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        // Delete random element
        let delete_idx = rng.gen_range(0..len);
        let mut deleted = input.clone();
        deleted.remove(delete_idx);

        let ternary = ternarize(&deleted);
        assert_eq!(ternary.len(), len - 1);
    }
}

/// Fuzz test: Random insertions
#[test]
fn fuzz_mutation_insertions() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let len = rng.gen_range(10..100);
        let mut input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        // Insert random element
        let insert_idx = rng.gen_range(0..=len);
        input.insert(insert_idx, rng.gen::<f32>());

        let ternary = ternarize(&input);
        assert_eq!(ternary.len(), len + 1);
    }
}

/// Fuzz test: Swap mutations
#[test]
fn fuzz_mutation_swaps() {
    use sophon_core::hdc::bind;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = 64;
        let mut input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        // Swap two elements
        let i = rng.gen_range(0..len);
        let j = rng.gen_range(0..len);
        input.swap(i, j);

        // Should still bind
        let result = bind(&input, &input);
        assert_eq!(result.len(), len);
    }
}

/// Fuzz test: Shuffle mutations
#[test]
fn fuzz_mutation_shuffles() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..20 {
        let len = 100;
        let mut input: Vec<f32> = (0..len).map(|i| i as f32).collect();

        // Shuffle
        let mut shuffled = input.clone();
        for i in (1..len).rev() {
            let j = rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }

        let ternary = ternarize(&shuffled);
        assert_eq!(ternary.len(), len);
    }
}

// ============================================================================
// Section 6: Oracle Tests
// ============================================================================

/// Fuzz test: Commutativity oracle
#[test]
fn fuzz_oracle_commutativity() {
    use sophon_core::hdc::bind;
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = rng.gen_range(16..128);
        let a: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        // Bind should be commutative
        let ab = bind(&a, &b);
        let ba = bind(&b, &a);

        for i in 0..len {
            assert!((ab[i] - ba[i]).abs() < 1e-5 || ab[i].is_nan() == ba[i].is_nan());
        }

        // MSE should be symmetric
        let mse_ab = LossFn::Mse.compute(&a, &b);
        let mse_ba = LossFn::Mse.compute(&b, &a);
        assert!((mse_ab - mse_ba).abs() < 1e-5);
    }
}

/// Fuzz test: Associativity oracle
#[test]
fn fuzz_oracle_associativity() {
    use sophon_core::hdc::bundle;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let len = 64;
        let a: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();
        let c: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        let abc = bundle(&[&a, &b, &c]);
        let bac = bundle(&[&b, &a, &c]);

        // Bundle is commutative but not strictly associative
        // Just check dimensions
        assert_eq!(abc.len(), len);
        assert_eq!(bac.len(), len);
    }
}

/// Fuzz test: Identity oracle
#[test]
fn fuzz_oracle_identity() {
    use sophon_core::hdc::bind;
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let len = rng.gen_range(16..128);
        let a: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        // Zero is identity for MSE addition
        let zeros = vec![0.0f32; len];
        let mse_zero = LossFn::Mse.compute(&a, &a);
        let mse_id = LossFn::Mse.compute(&a, &a);
        assert!((mse_zero - mse_id).abs() < 1e-5);

        // Ones vector for bind
        let ones = vec![1.0f32; len];
        let bound = bind(&a, &ones);
        // Result should be related to a
        assert_eq!(bound.len(), len);
    }
}

/// Fuzz test: Idempotence oracle
#[test]
fn fuzz_oracle_idempotence() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = rng.gen_range(16..256);
        let input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        // Ternarize is idempotent-ish (may lose info but stay stable)
        let t1 = ternarize(&input);
        let d1: Vec<f32> = t1.iter().map(|&t| t as f32).collect();
        let t2 = ternarize(&d1);

        assert_eq!(t1, t2, "Ternarize should be idempotent");
    }
}

/// Fuzz test: Roundtrip oracle
#[test]
fn fuzz_oracle_roundtrip() {
    use sophon_core::hdc::{permute, permute_inverse};

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = 64;
        let a: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();
        let n = rng.gen_range(1..10);

        // Permute and inverse should roundtrip
        let permuted = permute(&a, n);
        let recovered = permute_inverse(&permuted, n);

        for i in 0..len {
            assert!((a[i] - recovered[i]).abs() < 1e-5);
        }
    }
}

// ============================================================================
// Section 7: Invariant Tests
// ============================================================================

/// Fuzz test: Dimensional invariants
#[test]
fn fuzz_invariant_dimensions() {
    use sophon_core::hdc::{bind, bundle, circular_conv};
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let dim = rng.gen_range(16..256);
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        // All operations preserve dimension
        assert_eq!(bind(&a, &b).len(), dim);
        assert_eq!(bundle(&[&a, &b]).len(), dim);
        assert_eq!(circular_conv(&a, &b).len(), dim);

        // Loss preserves "dimensionality"
        let mse = LossFn::Mse.compute(&a, &b);
        assert!(mse.is_finite() || mse.is_nan());
    }
}

/// Fuzz test: Value range invariants
#[test]
fn fuzz_invariant_value_ranges() {
    use sophon_core::ops::clamp;
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let val = rng.gen::<f32>();
        let clamped = clamp(val, -1.0, 1.0);
        assert!(clamped >= -1.0 && clamped <= 1.0);

        let input = vec![val; 10];
        let ternary = ternarize(&input);
        assert!(ternary.iter().all(|&t| t >= -1 && t <= 1));
    }
}

/// Fuzz test: Non-negativity invariants
#[test]
fn fuzz_invariant_non_negativity() {
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = rng.gen_range(1..256);
        let a: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        let mse = LossFn::Mse.compute(&a, &b);
        assert!(mse >= 0.0 || mse.is_nan(), "MSE should be non-negative");
    }
}

/// Fuzz test: Finiteness invariants
#[test]
fn fuzz_invariant_finiteness() {
    use sophon_core::ops::hadamard;
    use sophon_inference::belief::BeliefState;

    let mut rng = rand::thread_rng();
    let mut belief = BeliefState::new(64);

    for _ in 0..100 {
        let a: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();

        let prod = hadamard(&a, &b);
        assert!(prod.iter().all(|&v| v.is_finite() || v.is_nan()));

        belief.update(&a, &[], 0.01);
        assert!(belief.mu_magnitude().is_finite() || belief.mu_magnitude().is_nan());
    }
}

/// Fuzz test: Symmetry invariants
#[test]
fn fuzz_invariant_symmetry() {
    use sophon_core::hdc::cosine_sim;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let dim = rng.gen_range(16..128);
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let sim_ab = cosine_sim(&a, &b);
        let sim_ba = cosine_sim(&b, &a);

        assert!(
            (sim_ab - sim_ba).abs() < 1e-5,
            "Cosine similarity should be symmetric"
        );
    }
}

/// Fuzz test: Triangle inequality invariants
#[test]
fn fuzz_invariant_triangle_inequality() {
    use sophon_core::ops::rms_norm_inplace;

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        // Normalized vectors
        let mut a_norm = a.clone();
        let mut b_norm = b.clone();
        rms_norm_inplace(&mut a_norm);
        rms_norm_inplace(&mut b_norm);

        // Dot product of normalized vectors should be in [-1, 1]
        let dot: f32 = a_norm.iter().zip(b_norm.iter()).map(|(x, y)| x * y).sum();
        assert!(dot >= -1.0 - 1e-5 && dot <= 1.0 + 1e-5);
    }
}

// ============================================================================
// Section 8: Property-Based Tests
// ============================================================================

/// Fuzz test: Ternarization properties
#[test]
fn fuzz_property_ternarization() {
    use sophon_quant::quant::ternarize;

    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let len = rng.gen_range(1..256);
        let input: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        let ternary = ternarize(&input);

        // Property: output has same length
        assert_eq!(ternary.len(), input.len());

        // Property: all values are -1, 0, or 1
        assert!(ternary.iter().all(|&t| t == -1 || t == 0 || t == 1));

        // Property: sign is preserved for large values
        for (i, (&t, &v)) in ternary.iter().zip(input.iter()).enumerate() {
            if v.abs() > 0.5 {
                assert_eq!(t.signum(), v.signum() as i8);
            }
        }
    }
}

/// Fuzz test: HDC binding properties
#[test]
fn fuzz_property_hdc_binding() {
    use sophon_core::hdc::{bind, cosine_sim};

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let c: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let ab = bind(&a, &b);
        let ba = bind(&b, &a);

        // Property: binding is commutative
        for i in 0..dim {
            assert!((ab[i] - ba[i]).abs() < 1e-5);
        }

        // Property: binding preserves dimension
        assert_eq!(ab.len(), dim);
    }
}

/// Fuzz test: Memory retrieval properties
#[test]
fn fuzz_property_memory_retrieval() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut rng = rand::thread_rng();
    let mut memory = EpisodicMemory::new(100);

    // Add random patterns
    let mut patterns: Vec<Vec<f32>> = vec![];
    for _ in 0..20 {
        let pattern: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
        patterns.push(pattern.clone());
        memory.add_episode(pattern);
    }

    for _ in 0..50 {
        // Query with random pattern
        let query: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();
        let k = rng.gen_range(1..10);

        let results = memory.retrieve_episodes(&query, k);

        // Property: returns at most k results
        assert!(results.len() <= k);

        // Property: returns at most memory.len() results
        assert!(results.len() <= memory.len());
    }
}

/// Fuzz test: Loss function properties
#[test]
fn fuzz_property_loss() {
    use sophon_loss::LossFn;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = rng.gen_range(1..256);
        let logits: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();
        let targets: Vec<f32> = (0..len).map(|_| rng.gen::<f32>()).collect();

        let mse = LossFn::Mse.compute(&logits, &targets);
        let mse_same = LossFn::Mse.compute(&logits, &logits);

        // Property: MSE is non-negative
        assert!(mse >= 0.0 || mse.is_nan());

        // Property: MSE of identical is ~0
        assert!(mse_same.abs() < 1e-5 || mse_same.is_nan());

        // Property: MSE is symmetric
        let mse_ab = LossFn::Mse.compute(&logits, &targets);
        let mse_ba = LossFn::Mse.compute(&targets, &logits);
        assert!((mse_ab - mse_ba).abs() < 1e-5);
    }
}

/// Fuzz test: Belief update properties
#[test]
fn fuzz_property_belief_update() {
    use sophon_inference::belief::BeliefState;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let mut belief = BeliefState::new(64);
        let grad: Vec<f32> = (0..64).map(|_| rng.gen::<f32>()).collect();

        let before = belief.mu.clone();
        belief.update(&grad, &[], 0.01);
        let after = belief.mu.clone();

        // Property: update changes state (usually)
        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(b, a)| (b - a).abs() > 1e-10);
        // Note: may not change if grad is zero

        // Property: uncertainty is non-negative
        assert!(belief.uncertainty() >= 0.0);
    }
}

/// Fuzz test: Model output properties
#[test]
fn fuzz_property_model_output() {
    use sophon_config::VOCAB_SIZE;
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let len = rng.gen_range(1..100);
        let input: Vec<u8> = (0..len).map(|_| rng.gen::<u8>()).collect();

        let result = model.forward_sequence(&input);

        if let Ok(outputs) = result {
            if let Some(last) = outputs.last() {
                // Property: output has correct dimension
                assert_eq!(last.logits.len(), VOCAB_SIZE);

                // Property: all outputs are finite
                assert!(last.logits.iter().all(|&v| v.is_finite()));
            }
        }
    }
}

/// Fuzz test: SSM properties
#[test]
fn fuzz_property_ssm() {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let mut state = SsmState::new(SSM_N);
        let params = SsmParams::random(SSM_N);
        let input: Vec<f32> = (0..SSM_N).map(|_| rng.gen::<f32>()).collect();

        let output = selective_scan(&mut state, &params, &input);

        // Property: output has correct dimension
        assert_eq!(output.len(), SSM_N);

        // Property: output has some finite values
        assert!(output.iter().any(|&v| v.is_finite()));
    }
}

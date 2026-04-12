//! Benchmark Tests for Sophon AGI System
//!
//! Comprehensive benchmarks covering performance benchmarks for key operations,
//! memory benchmarks, and comparison benchmarks.
//!
//! These tests measure and track performance characteristics of the system.

#![feature(test)]

extern crate test;

use std::time::{Duration, Instant};
use test::Bencher;

// ============================================================================
// Section 1: Model Inference Benchmarks
// ============================================================================

/// Benchmark: Model forward pass - short input
#[bench]
fn bench_model_forward_short(b: &mut Bencher) {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let input = b"short";

    b.iter(|| {
        let _ = model.forward_sequence(input);
    });
}

/// Benchmark: Model forward pass - medium input
#[bench]
fn bench_model_forward_medium(b: &mut Bencher) {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let input = b"This is a medium length input for benchmarking";

    b.iter(|| {
        let _ = model.forward_sequence(input);
    });
}

/// Benchmark: Model forward pass - long input
#[bench]
fn bench_model_forward_long(b: &mut Bencher) {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let input = "This is a much longer input string for benchmarking purposes. ".repeat(10);

    b.iter(|| {
        let _ = model.forward_sequence(input.as_bytes());
    });
}

/// Benchmark: Model with different seeds
#[bench]
fn bench_model_different_seeds(b: &mut Bencher) {
    use sophon_model::Sophon1;

    let mut seed = 0u64;
    let input = b"test input";

    b.iter(|| {
        seed += 1;
        let model = Sophon1::new(seed);
        let _ = model.forward_sequence(input);
    });
}

// ============================================================================
// Section 2: Quantization Benchmarks
// ============================================================================

/// Benchmark: Ternarization - small vector
#[bench]
fn bench_ternarize_small(b: &mut Bencher) {
    use sophon_quant::quant::ternarize;

    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        let _ = ternarize(&input);
    });
}

/// Benchmark: Ternarization - medium vector
#[bench]
fn bench_ternarize_medium(b: &mut Bencher) {
    use sophon_quant::quant::ternarize;

    let input: Vec<f32> = (0..512).map(|i| i as f32 * 0.001).collect();

    b.iter(|| {
        let _ = ternarize(&input);
    });
}

/// Benchmark: Ternarization - large vector
#[bench]
fn bench_ternarize_large(b: &mut Bencher) {
    use sophon_quant::quant::ternarize;

    let input: Vec<f32> = (0..4096).map(|i| i as f32 * 0.0001).collect();

    b.iter(|| {
        let _ = ternarize(&input);
    });
}

/// Benchmark: Dequantization
#[bench]
fn bench_dequantize(b: &mut Bencher) {
    use sophon_quant::quant::dequantize;

    let input: Vec<i8> = (0..1000).map(|i| ((i % 3) as i8) - 1).collect();

    b.iter(|| {
        let _ = dequantize(&input);
    });
}

/// Benchmark: Quantization roundtrip
#[bench]
fn bench_quantization_roundtrip(b: &mut Bencher) {
    use sophon_quant::quant::{dequantize, ternarize};

    let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();

    b.iter(|| {
        let ternary = ternarize(&input);
        let _ = dequantize(&ternary);
    });
}

// ============================================================================
// Section 3: HDC Operation Benchmarks
// ============================================================================

/// Benchmark: HDC bind operation
#[bench]
fn bench_hdc_bind(b: &mut Bencher) {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();

    b.iter(|| {
        let _ = bind(&a, &b);
    });
}

/// Benchmark: HDC bundle operation - 2 vectors
#[bench]
fn bench_hdc_bundle_2(b: &mut Bencher) {
    use sophon_core::hdc::bundle;

    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (i + 100) as f32 * 0.01).collect();

    b.iter(|| {
        let _ = bundle(&[&a, &b]);
    });
}

/// Benchmark: HDC bundle operation - 5 vectors
#[bench]
fn bench_hdc_bundle_5(b: &mut Bencher) {
    use sophon_core::hdc::bundle;

    let vecs: Vec<Vec<f32>> = (0..5)
        .map(|j| (0..256).map(|i| (i + j * 100) as f32 * 0.01).collect())
        .collect();

    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    b.iter(|| {
        let _ = bundle(&refs);
    });
}

/// Benchmark: HDC circular convolution
#[bench]
fn bench_hdc_circular_conv(b: &mut Bencher) {
    use sophon_core::hdc::circular_conv;

    let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();

    b.iter(|| {
        let _ = circular_conv(&a, &b);
    });
}

/// Benchmark: HDC permutation
#[bench]
fn bench_hdc_permute(b: &mut Bencher) {
    use sophon_core::hdc::permute;

    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        let _ = permute(&a, 5);
    });
}

/// Benchmark: HDC similarity computation
#[bench]
fn bench_hdc_similarity(b: &mut Bencher) {
    use sophon_core::hdc::cosine_sim;

    let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();

    b.iter(|| {
        let _ = cosine_sim(&a, &b);
    });
}

// ============================================================================
// Section 4: Loss Function Benchmarks
// ============================================================================

/// Benchmark: MSE loss computation
#[bench]
fn bench_loss_mse(b: &mut Bencher) {
    use sophon_loss::LossFn;

    let logits: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let targets: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.01).collect();

    b.iter(|| {
        let _ = LossFn::Mse.compute(&logits, &targets);
    });
}

/// Benchmark: Cross-entropy loss computation
#[bench]
fn bench_loss_cross_entropy(b: &mut Bencher) {
    use sophon_config::VOCAB_SIZE;
    use sophon_loss::LossFn;

    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| i as f32 * 0.001).collect();
    let targets: Vec<f32> = (0..VOCAB_SIZE).map(|_| 1.0 / VOCAB_SIZE as f32).collect();

    b.iter(|| {
        let _ = LossFn::CrossEntropy.compute(&logits, &targets);
    });
}

/// Benchmark: Loss computation batch
#[bench]
fn bench_loss_batch(b: &mut Bencher) {
    use sophon_loss::LossFn;

    let batch_size = 32;
    let dim = 256;

    let logits_batch: Vec<Vec<f32>> = (0..batch_size)
        .map(|j| (0..dim).map(|i| (i + j) as f32 * 0.01).collect())
        .collect();

    let targets: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        for logits in &logits_batch {
            let _ = LossFn::Mse.compute(logits, &targets);
        }
    });
}

// ============================================================================
// Section 5: SSM Benchmarks
// ============================================================================

/// Benchmark: SSM selective scan
#[bench]
fn bench_ssm_selective_scan(b: &mut Bencher) {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    let mut state = SsmState::new(SSM_N);
    let params = SsmParams::random(SSM_N);
    let input: Vec<f32> = (0..SSM_N).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        let _ = selective_scan(&mut state, &params, &input);
    });
}

/// Benchmark: SSM discretization
#[bench]
fn bench_ssm_discretization(b: &mut Bencher) {
    use sophon_ssm::params::SsmParams;
    use sophon_ssm::zoh::DiscretisedSsm;

    let params = SsmParams::random(64);

    b.iter(|| {
        let _ = DiscretisedSsm::discretize(&params, 0.01);
    });
}

/// Benchmark: SSM state update
#[bench]
fn bench_ssm_state_update(b: &mut Bencher) {
    use sophon_config::SSM_N;
    use sophon_ssm::{params::SsmParams, selective::selective_scan, SsmState};

    b.iter(|| {
        let mut state = SsmState::new(SSM_N);
        let params = SsmParams::random(SSM_N);
        let input: Vec<f32> = (0..SSM_N).map(|i| i as f32 * 0.01).collect();
        let _ = selective_scan(&mut state, &params, &input);
    });
}

// ============================================================================
// Section 6: Memory Benchmarks
// ============================================================================

/// Benchmark: Memory add episode
#[bench]
fn bench_memory_add(b: &mut Bencher) {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(1000);
    let pattern: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        memory.add_episode(pattern.clone());
    });
}

/// Benchmark: Memory retrieval
#[bench]
fn bench_memory_retrieval(b: &mut Bencher) {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(100);
    let query: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    // Pre-populate
    for _ in 0..100 {
        let pattern: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        memory.add_episode(pattern);
    }

    b.iter(|| {
        let _ = memory.retrieve_episodes(&query, 10);
    });
}

/// Benchmark: Memory retrieval with large capacity
#[bench]
fn bench_memory_retrieval_large(b: &mut Bencher) {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(1000);
    let query: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    // Pre-populate with many patterns
    for i in 0..1000 {
        let pattern: Vec<f32> = (0..64).map(|j| ((i + j) % 100) as f32 * 0.01).collect();
        memory.add_episode(pattern);
    }

    b.iter(|| {
        let _ = memory.retrieve_episodes(&query, 10);
    });
}

/// Benchmark: Working memory operations
#[bench]
fn bench_working_memory(b: &mut Bencher) {
    use sophon_memory::working::WorkingMemory;

    let mut memory = WorkingMemory::new(100, 64);
    let item: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        memory.add(item.clone());
        let _ = memory.get_recent(10);
    });
}

// ============================================================================
// Section 7: Belief State Benchmarks
// ============================================================================

/// Benchmark: Belief state update
#[bench]
fn bench_belief_update(b: &mut Bencher) {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);
    let grad: Vec<f32> = (0..64).map(|i| i as f32 * 0.001).collect();

    b.iter(|| {
        belief.update(&grad, &[], 0.01);
    });
}

/// Benchmark: Belief normalization
#[bench]
fn bench_belief_normalize(b: &mut Bencher) {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Set some values
    for i in 0..64 {
        belief.mu[i] = i as f32 * 0.1;
    }

    b.iter(|| {
        belief.normalize();
    });
}

/// Benchmark: Belief uncertainty computation
#[bench]
fn bench_belief_uncertainty(b: &mut Bencher) {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(64);

    b.iter(|| {
        let _ = belief.uncertainty();
    });
}

/// Benchmark: Belief magnitude computation
#[bench]
fn bench_belief_magnitude(b: &mut Bencher) {
    use sophon_inference::belief::BeliefState;

    let belief = BeliefState::new(64);

    b.iter(|| {
        let _ = belief.mu_magnitude();
    });
}

// ============================================================================
// Section 8: Core Operations Benchmarks
// ============================================================================

/// Benchmark: Hadamard product
#[bench]
fn bench_ops_hadamard(b: &mut Bencher) {
    use sophon_core::ops::hadamard;

    let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..1024).map(|i| (1023 - i) as f32 * 0.001).collect();

    b.iter(|| {
        let _ = hadamard(&a, &b);
    });
}

/// Benchmark: RMS normalization
#[bench]
fn bench_ops_rms_norm(b: &mut Bencher) {
    use sophon_core::ops::rms_norm_inplace;

    let mut data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();

    b.iter(|| {
        rms_norm_inplace(&mut data);
    });
}

/// Benchmark: Softmax
#[bench]
fn bench_ops_softmax(b: &mut Bencher) {
    use sophon_core::ops::softmax;

    let logits: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();

    b.iter(|| {
        let _ = softmax(&logits);
    });
}

/// Benchmark: Argmax
#[bench]
fn bench_ops_argmax(b: &mut Bencher) {
    use sophon_core::ops::argmax;

    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();

    b.iter(|| {
        let _ = argmax(&data);
    });
}

/// Benchmark: Dot product
#[bench]
fn bench_ops_dot_product(b: &mut Bencher) {
    let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..1024).map(|i| (1023 - i) as f32 * 0.001).collect();

    b.iter(|| {
        let _: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    });
}

// ============================================================================
// Section 9: Comparison Benchmarks
// ============================================================================

/// Benchmark: Compare HDC bind vs circular conv
#[bench]
fn bench_compare_bind_vs_conv(b: &mut Bencher) {
    use sophon_core::hdc::{bind, circular_conv};

    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (i + 100) as f32 * 0.01).collect();

    b.iter(|| {
        let _bind_result = bind(&a, &b);
        let _conv_result = circular_conv(&a, &b);
    });
}

/// Benchmark: Compare different loss functions
#[bench]
fn bench_compare_loss_functions(b: &mut Bencher) {
    use sophon_loss::LossFn;

    let logits: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let targets: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.01).collect();

    b.iter(|| {
        let _ = LossFn::Mse.compute(&logits, &targets);
        let _ = LossFn::CrossEntropy.compute(&logits, &targets);
    });
}

/// Benchmark: Compare quantization strategies
#[bench]
fn bench_compare_quantization(b: &mut Bencher) {
    use sophon_quant::quant::ternarize;

    let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();

    b.iter(|| {
        // Full ternarization
        let _ = ternarize(&input);
    });
}

/// Benchmark: Compare different dimensions
#[bench]
fn bench_compare_dimensions(b: &mut Bencher) {
    use sophon_core::hdc::bind;

    let dims = vec![64, 128, 256, 512];
    let inputs: Vec<(Vec<f32>, Vec<f32>)> = dims
        .iter()
        .map(|&d| {
            let a: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();
            let b: Vec<f32> = (0..d).map(|i| (i + 50) as f32 * 0.01).collect();
            (a, b)
        })
        .collect();

    b.iter(|| {
        for (a, b) in &inputs {
            let _ = bind(&a, &b);
        }
    });
}

// ============================================================================
// Section 10: Memory Usage Benchmarks
// ============================================================================

/// Benchmark: Memory allocation patterns
#[bench]
fn bench_memory_allocation(b: &mut Bencher) {
    b.iter(|| {
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let _: f32 = data.iter().sum();
    });
}

/// Benchmark: Repeated allocation
#[bench]
fn bench_memory_repeated_allocation(b: &mut Bencher) {
    b.iter(|| {
        for _ in 0..100 {
            let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
            let _ = data.len();
        }
    });
}

/// Benchmark: Vector cloning
#[bench]
fn bench_memory_clone(b: &mut Bencher) {
    let data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();

    b.iter(|| {
        let _ = data.clone();
    });
}

/// Benchmark: Aligned vector allocation
#[bench]
fn bench_memory_aligned_allocation(b: &mut Bencher) {
    use sophon_accel::aligned::AlignedVec;

    b.iter(|| {
        let _ = AlignedVec::<f32, 64>::from_vec(vec![1.0f32; 1000]);
    });
}

// ============================================================================
// Section 11: Throughput Benchmarks
// ============================================================================

/// Benchmark: Model throughput
#[test]
fn bench_throughput_model() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let input = b"test";

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = model.forward_sequence(input);
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("Model throughput: {:.2} inferences/sec", throughput);
    assert!(throughput > 0.0);
}

/// Benchmark: Quantization throughput
#[test]
fn bench_throughput_quantization() {
    use sophon_quant::quant::ternarize;

    let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();

    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = ternarize(&input);
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("Quantization throughput: {:.2} ops/sec", throughput);
    assert!(throughput > 0.0);
}

/// Benchmark: HDC operation throughput
#[test]
fn bench_throughput_hdc() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (i + 100) as f32 * 0.01).collect();

    let iterations = 10000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = bind(&a, &b);
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("HDC bind throughput: {:.2} ops/sec", throughput);
    assert!(throughput > 0.0);
}

/// Benchmark: Loss computation throughput
#[test]
fn bench_throughput_loss() {
    use sophon_loss::LossFn;

    let logits: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let targets: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.01).collect();

    let iterations = 10000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = LossFn::Mse.compute(&logits, &targets);
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("Loss computation throughput: {:.2} ops/sec", throughput);
    assert!(throughput > 0.0);
}

/// Benchmark: Memory operation throughput
#[test]
fn bench_throughput_memory() {
    use sophon_memory::episodic::EpisodicMemory;

    let mut memory = EpisodicMemory::new(1000);
    let query: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    // Pre-populate
    for _ in 0..500 {
        let pattern: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        memory.add_episode(pattern);
    }

    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = memory.retrieve_episodes(&query, 10);
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("Memory retrieval throughput: {:.2} ops/sec", throughput);
    assert!(throughput > 0.0);
}

// ============================================================================
// Section 12: Latency Benchmarks
// ============================================================================

/// Benchmark: Model latency
#[test]
fn bench_latency_model() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let inputs = vec![
        b"a",
        b"short",
        b"medium length input",
        b"this is a longer input for testing latency",
    ];

    for input in &inputs {
        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = model.forward_sequence(input);
        }

        let duration = start.elapsed();
        let latency_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;

        println!(
            "Model latency ({} bytes): {:.3} ms",
            input.len(),
            latency_ms
        );
    }
}

/// Benchmark: HDC operation latency
#[test]
fn bench_latency_hdc() {
    use sophon_core::hdc::bind;

    let dims = vec![64, 128, 256, 512, 1024];

    for &dim in &dims {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i + 50) as f32 * 0.01).collect();

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = bind(&a, &b);
        }

        let duration = start.elapsed();
        let latency_us = duration.as_secs_f64() * 1_000_000.0 / iterations as f64;

        println!("HDC bind latency (dim={}): {:.3} us", dim, latency_us);
    }
}

/// Benchmark: Memory operation latency
#[test]
fn bench_latency_memory() {
    use sophon_memory::episodic::EpisodicMemory;

    let capacities = vec![10, 100, 500, 1000];

    for &capacity in &capacities {
        let mut memory = EpisodicMemory::new(capacity);
        let query: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

        // Pre-populate
        for _ in 0..capacity {
            let pattern: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
            memory.add_episode(pattern);
        }

        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = memory.retrieve_episodes(&query, 10);
        }

        let duration = start.elapsed();
        let latency_us = duration.as_secs_f64() * 1_000_000.0 / iterations as f64;

        println!(
            "Memory retrieval latency (capacity={}): {:.3} us",
            capacity, latency_us
        );
    }
}

/// Benchmark: End-to-end latency
#[test]
fn bench_latency_end_to_end() {
    use sophon_inference::belief::BeliefState;
    use sophon_loss::LossFn;
    use sophon_memory::episodic::EpisodicMemory;
    use sophon_model::Sophon1;
    use sophon_quant::quant::ternarize;

    let model = Sophon1::new(0x1234);
    let mut memory = EpisodicMemory::new(100);
    let mut belief = BeliefState::new(64);

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        // Inference
        let input = b"test";
        let outputs = model.forward_sequence(input).unwrap();

        if let Some(last) = outputs.last() {
            // Quantize
            let obs: Vec<f32> = last.logits.iter().take(256).copied().collect();
            let _ = ternarize(&obs);

            // Memory
            let slice: Vec<f32> = last.logits.iter().take(64).copied().collect();
            memory.add_episode(slice.clone());
            let _ = memory.retrieve_episodes(&slice, 3);

            // Belief
            belief.update(&slice, &[], 0.01);

            // Loss
            let targets: Vec<f32> = (0..last.logits.len()).map(|_| 0.01f32).collect();
            let _ = LossFn::Mse.compute(&last.logits, &targets);
        }
    }

    let duration = start.elapsed();
    let latency_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;

    println!("End-to-end latency: {:.3} ms", latency_ms);
}

// ============================================================================
// Section 13: Scalability Benchmarks
// ============================================================================

/// Benchmark: Scale with input size
#[test]
fn bench_scale_input_size() {
    use sophon_model::Sophon1;

    let model = Sophon1::new(0x1234);
    let sizes = vec![10, 50, 100, 500, 1000];

    for &size in &sizes {
        let input = "x".repeat(size);

        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = model.forward_sequence(input.as_bytes());
        }

        let duration = start.elapsed();
        let avg_time = duration.as_secs_f64() / iterations as f64;

        println!("Input size {}: {:.3} sec avg", size, avg_time);
    }
}

/// Benchmark: Scale with dimension
#[test]
fn bench_scale_dimension() {
    use sophon_core::hdc::bind;

    let dims = vec![16, 32, 64, 128, 256, 512, 1024];

    for &dim in &dims {
        let a: Vec<f32> = (0..dim).map(|_| 1.0f32).collect();
        let b: Vec<f32> = (0..dim).map(|_| 1.0f32).collect();

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = bind(&a, &b);
        }

        let duration = start.elapsed();
        let throughput = iterations as f64 / duration.as_secs_f64();

        println!("Dimension {}: {:.0} ops/sec", dim, throughput);
    }
}

/// Benchmark: Scale with batch size
#[test]
fn bench_scale_batch_size() {
    use sophon_loss::LossFn;

    let batch_sizes = vec![1, 10, 50, 100, 500];

    for &batch_size in &batch_sizes {
        let logits_batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| (0..256).map(|_| rand::random::<f32>()).collect())
            .collect();
        let targets: Vec<f32> = (0..256).map(|_| 0.01f32).collect();

        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            for logits in &logits_batch {
                let _ = LossFn::Mse.compute(logits, &targets);
            }
        }

        let duration = start.elapsed();
        let throughput = (iterations * batch_size) as f64 / duration.as_secs_f64();

        println!("Batch size {}: {:.0} items/sec", batch_size, throughput);
    }
}

/// Benchmark: Scale with memory capacity
#[test]
fn bench_scale_memory_capacity() {
    use sophon_memory::episodic::EpisodicMemory;

    let capacities = vec![10, 100, 1000, 5000];

    for &capacity in &capacities {
        let mut memory = EpisodicMemory::new(capacity);
        let query: Vec<f32> = (0..64).map(|_| 1.0f32).collect();

        // Pre-populate
        for _ in 0..capacity {
            let pattern: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
            memory.add_episode(pattern);
        }

        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = memory.retrieve_episodes(&query, 10);
        }

        let duration = start.elapsed();
        let latency_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;

        println!(
            "Memory capacity {}: {:.3} ms avg latency",
            capacity, latency_ms
        );
    }
}

// ============================================================================
// Section 14: Stress Benchmarks
// ============================================================================

/// Benchmark: Sustained load
#[test]
fn bench_stress_sustained() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (i + 100) as f32 * 0.01).collect();

    let duration = Duration::from_secs(1);
    let start = Instant::now();
    let mut iterations = 0u64;

    while start.elapsed() < duration {
        let _ = bind(&a, &b);
        iterations += 1;
    }

    let total_duration = start.elapsed();
    let throughput = iterations as f64 / total_duration.as_secs_f64();

    println!("Sustained load: {} ops/sec", throughput);
    assert!(iterations > 1000);
}

/// Benchmark: Burst load
#[test]
fn bench_stress_burst() {
    use sophon_loss::LossFn;

    let logits: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let targets: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.01).collect();

    let iterations = 100000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = LossFn::Mse.compute(&logits, &targets);
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("Burst load: {:.0} ops/sec", throughput);
}

/// Benchmark: Gradual load increase
#[test]
fn bench_stress_gradual_increase() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (i + 100) as f32 * 0.01).collect();

    for load_multiplier in 1..=10 {
        let iterations = 1000 * load_multiplier;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = bind(&a, &b);
        }

        let duration = start.elapsed();
        let throughput = iterations as f64 / duration.as_secs_f64();

        println!("Load x{}: {:.0} ops/sec", load_multiplier, throughput);
    }
}

/// Benchmark: Recovery after stress
#[test]
fn bench_stress_recovery() {
    use sophon_inference::belief::BeliefState;

    let mut belief = BeliefState::new(64);

    // Stress
    for _ in 0..10000 {
        let grad: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        belief.update(&grad, &[], 0.01);
    }

    let after_stress = belief.mu_magnitude();

    // Recovery period
    for _ in 0..100 {
        let grad: Vec<f32> = (0..64).map(|_| 0.01f32).collect();
        belief.update(&grad, &[], 0.001);
    }

    let after_recovery = belief.mu_magnitude();

    println!("After stress: {:.6}", after_stress);
    println!("After recovery: {:.6}", after_recovery);

    assert!(after_recovery.is_finite());
}

// ============================================================================
// Section 15: Baseline Benchmarks
// ============================================================================

/// Benchmark: Empty loop baseline
#[bench]
fn bench_baseline_empty(b: &mut Bencher) {
    b.iter(|| {
        // Empty operation
    });
}

/// Benchmark: Vector allocation baseline
#[bench]
fn bench_baseline_vec_allocation(b: &mut Bencher) {
    b.iter(|| {
        let _: Vec<f32> = Vec::with_capacity(1000);
    });
}

/// Benchmark: Simple arithmetic baseline
#[bench]
fn bench_baseline_arithmetic(b: &mut Bencher) {
    let mut sum = 0.0f32;
    b.iter(|| {
        for i in 0..1000 {
            sum += i as f32 * 0.001;
        }
    });
}

/// Benchmark: Memory copy baseline
#[bench]
fn bench_baseline_memcopy(b: &mut Bencher) {
    let src: Vec<f32> = (0..1000).map(|i| i as f32).collect();

    b.iter(|| {
        let _ = src.clone();
    });
}

/// Benchmark: Random number generation baseline
#[bench]
fn bench_baseline_random(b: &mut Bencher) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    b.iter(|| {
        let _: f32 = rng.gen();
    });
}

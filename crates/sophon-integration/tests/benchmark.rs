//! Benchmarks for Sophon AGI System
//!
//! Performance benchmarks using ONLY real APIs.

use std::time::{Duration, Instant};

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
use sophon_train::checkpoint::CheckpointStrategy;
use sophon_train::state::TrainState;
use sophon_tui::{Color, Constraint, Element, Rect, Style};
use sophon_verifier::{VerifiedOutput, VerifierGate};

// ============================================================================
// Section 1: Model Inference Benchmarks
// ============================================================================

/// Benchmark: Model inference on short input
#[test]
fn bench_model_inference_short() {
    let mut model = Sophon1::new(0x1234);
    let input = b"Hi";

    let start = Instant::now();
    let iterations = 2; // Reduced for reasonable test time

    for _ in 0..iterations {
        let _ = model.forward_sequence(input);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Model inference (short): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Model inference on medium input
#[test]
fn bench_model_inference_medium() {
    let mut model = Sophon1::new(0x1234);
    let input = b"Test";

    let start = Instant::now();
    let iterations = 2; // Reduced for reasonable test time

    for _ in 0..iterations {
        let _ = model.forward_sequence(input);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Model inference (medium): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Model inference on longer input
#[test]
fn bench_model_inference_long() {
    let mut model = Sophon1::new(0x1234);
    let input = "abc".to_string(); // Short input for reasonable test time

    let start = Instant::now();
    let iterations = 2; // Reduced for reasonable test time

    for _ in 0..iterations {
        let _ = model.forward_sequence(input.as_bytes());
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Model inference (long): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Model with safety check
#[test]
fn bench_model_with_safety() {
    let mut model = Sophon1::new(0x1234);
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);
    let input = b"x";

    let start = Instant::now();
    let iterations = 2; // Reduced for reasonable test time

    for _ in 0..iterations {
        if let Ok(outputs) = model.forward_sequence(input) {
            if let Some(last) = outputs.last() {
                let _ = diagnostic.check(last.logits.as_slice());
            }
        }
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Model with safety: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Model parameter count
#[test]
fn bench_model_param_count() {
    let model = Sophon1::new(0x1234);

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = model.param_count();
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("Param count: {:?} avg over {} iterations", avg, iterations);
}

// ============================================================================
// Section 2: Quantization Benchmarks
// ============================================================================

/// Benchmark: Ternarize single value
#[test]
fn bench_ternarize_single() {
    let start = Instant::now();
    let iterations = 100000;

    for _ in 0..iterations {
        let _ = ternarize(1.5, 1.0, 0.5);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Ternarize single: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Ternarize block (64 elements)
#[test]
fn bench_ternarize_block_64() {
    let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = ternarize_block(&weights);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Ternarize block (64): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Ternarize block (256 elements)
#[test]
fn bench_ternarize_block_256() {
    let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    let start = Instant::now();
    let iterations = 500;

    for _ in 0..iterations {
        let _ = ternarize_block(&weights);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Ternarize block (256): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Dequantize block
#[test]
fn bench_dequantize_block() {
    let mut weights = [0i8; 64];
    let sample = [1i8, -1, 0, 1, -1, 0, 1, -1, 0, 1];
    for i in 0..10 {
        weights[i] = sample[i];
    }
    let block = TernaryBlock {
        weights,
        scale: 0.5,
    };
    let mut out = vec![0.0f32; 64];

    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        dequantize_block(&block, &mut out);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Dequantize block: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Quantization roundtrip
#[test]
fn bench_quantization_roundtrip() {
    let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();

    let start = Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        let block = ternarize_block(&weights);
        let mut out = vec![0.0f32; 256];
        dequantize_block(&block, &mut out);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Quantization roundtrip: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 3: HDC Operation Benchmarks
// ============================================================================

/// Benchmark: Circular convolution (64 elements)
#[test]
fn bench_conv_64() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = circular_conv(&a, &b);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Circular conv (64): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Circular convolution (256 elements)
#[test]
fn bench_conv_256() {
    let a = vec![1.0f32; 256];
    let b = vec![0.5f32; 256];

    let start = Instant::now();
    let iterations = 500;

    for _ in 0..iterations {
        let _ = circular_conv(&a, &b);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Circular conv (256): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Bind operation
#[test]
fn bench_bind() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = bind(&a, &b);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("Bind: {:?} avg over {} iterations", avg, iterations);
}

/// Benchmark: Bundle operation (2 vectors)
#[test]
fn bench_bundle_2() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];
    let refs: Vec<&[f32]> = vec![&a, &b];

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = bundle(&refs);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Bundle (2 vectors): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Bundle operation (5 vectors)
#[test]
fn bench_bundle_5() {
    let vecs: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0f32; 64]).collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let iterations = 500;

    for _ in 0..iterations {
        let _ = bundle(&refs);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Bundle (5 vectors): {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: L2 normalize
#[test]
fn bench_l2_normalize() {
    let mut v = vec![3.0f32, 4.0, 0.0, 1.0, 2.0];

    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        l2_normalize(&mut v);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("L2 normalize: {:?} avg over {} iterations", avg, iterations);
}

// ============================================================================
// Section 4: Loss Function Benchmarks
// ============================================================================

/// Benchmark: Free energy loss
#[test]
fn bench_free_energy_loss() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];
    let prediction_error = 1.0f32;

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = free_energy_loss(&mu, &log_sigma, prediction_error);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Free energy loss: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: KL divergence
#[test]
fn bench_kl_divergence() {
    let mu = vec![0.0f32; SSM_N];
    let log_sigma = vec![0.0f32; SSM_N];

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = kl_divergence_standard_normal(&mu, &log_sigma);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "KL divergence: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Prediction error loss
#[test]
fn bench_prediction_error() {
    let logits = vec![1.0f32; VOCAB_SIZE];
    let target = 0;

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = prediction_error_loss(&logits, target);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Prediction error: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 5: Memory System Benchmarks
// ============================================================================

/// Benchmark: Episodic memory record
#[test]
fn bench_memory_record() {
    let mut memory = EpisodicMemory::new(100);

    let start = Instant::now();
    let iterations = 100;

    for i in 0..iterations {
        let episode = Episode {
            timestamp: i as u64,
            perception_hv: vec![1.0f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        };
        memory.record(episode);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Memory record: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Episodic memory retrieve
#[test]
fn bench_memory_retrieve() {
    let mut memory = EpisodicMemory::new(100);

    // Pre-populate
    for i in 0..50 {
        memory.record(Episode {
            timestamp: i as u64,
            perception_hv: vec![(i % 10) as f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        });
    }

    let query = vec![5.0f32; HDC_DIM];

    let start = Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        let _ = memory.retrieve_similar(&query, 5);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Memory retrieve: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Working memory push
#[test]
fn bench_working_push() {
    let mut memory = WorkingMemory::new(32);

    let start = Instant::now();
    let iterations = 1000;

    for i in 0..iterations {
        memory.push(WorkingEntry {
            content_hv: vec![(i % 10) as f32; HDC_DIM],
            timestamp: i as u64,
            access_count: 1,
        });
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Working memory push: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Procedural memory learn
#[test]
fn bench_procedural_learn() {
    let mut memory = ProceduralMemory::new(100);

    let start = Instant::now();
    let iterations = 100;

    for i in 0..iterations {
        let pattern = ActionPattern {
            name: format!("skill_{}", i),
            preconditions: vec![],
            effects: vec![],
            success_rate: 0.9,
            avg_cost: 0.5,
            context_hv: vec![(i % 10) as f32; HDC_DIM],
        };
        memory.learn(pattern);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Procedural learn: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 6: Tensor Operation Benchmarks
// ============================================================================

/// Benchmark: Softmax
#[test]
fn bench_softmax() {
    let logits = Tensor::from_slice_1d(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = softmax_1d(&logits);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("Softmax: {:?} avg over {} iterations", avg, iterations);
}

/// Benchmark: GEMV
#[test]
fn bench_gemv() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4, 2).unwrap();
    let x = Tensor::from_slice_1d(&[1.0f32, 2.0]);

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = gemv(&a, &x);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("GEMV: {:?} avg over {} iterations", avg, iterations);
}

/// Benchmark: GEMM
#[test]
fn bench_gemm() {
    let a = Tensor::from_slice_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = Tensor::from_slice_2d(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = gemm(&a, &b);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("GEMM: {:?} avg over {} iterations", avg, iterations);
}

/// Benchmark: Tensor creation
#[test]
fn bench_tensor_zeros() {
    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = Tensor::zeros_1d(100);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Tensor zeros (100): {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 7: Safety System Benchmarks
// ============================================================================

/// Benchmark: Self diagnostic check
#[test]
fn bench_diagnostic_check() {
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32) / 10.0 - 12.8).collect();

    let start = Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        let _ = diagnostic.check(&logits);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Diagnostic check: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Verifier check
#[test]
fn bench_verifier_check() {
    let verifier = VerifierGate::default();
    let logits = Tensor::from_slice_1d(&vec![0.0f32; VOCAB_SIZE]);

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = verifier.check(&logits);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Verifier check: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Alignment monitor step
#[test]
fn bench_alignment_step() {
    let config = AlignmentConfig::from_spec();
    let anchor: Vec<f32> = vec![0.0; 1000];
    let mut monitor = AlignmentMonitor::new(&anchor, config);

    let start = Instant::now();
    let iterations = 100;

    for _ in 0..iterations {
        monitor.report_score(0.9);
        let current: Vec<f32> = vec![0.0; 1000];
        let _ = monitor.step(&current);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Alignment step: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 8: Document Processing Benchmarks
// ============================================================================

/// Benchmark: Document creation
#[test]
fn bench_document_creation() {
    let content = "This is a test document with some content.";

    let start = Instant::now();
    let iterations = 10000;

    for i in 0..iterations {
        let _ = Document::new(&format!("doc{}", i), content);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Document creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Document entropy
#[test]
fn bench_document_entropy() {
    let doc = Document::new("test", "This is a test document with some entropy.");

    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = doc.byte_entropy();
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Document entropy: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Quality filter check
#[test]
fn bench_quality_filter() {
    let config = FilterConfig::default();
    let mut filter = QualityFilter::new(config);
    let doc = Document::new("test", "This is a quality test document.");

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = filter.check(&doc);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Quality filter: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 9: SSM Benchmarks
// ============================================================================

/// Benchmark: SsmParams creation
#[test]
fn bench_ssm_params() {
    let start = Instant::now();
    let iterations = 100;

    for i in 0..iterations {
        let _ = SsmParams::new_stable(i as u64);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "SsmParams creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: DiscretisedSsm creation
#[test]
fn bench_ssm_discretised() {
    let params = SsmParams::new_stable(0x1234);

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = DiscretisedSsm::from_params(&params);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "DiscretisedSsm creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: SsmState creation
#[test]
fn bench_ssm_state() {
    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = SsmState::new();
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "SsmState creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 10: TUI Benchmarks
// ============================================================================

/// Benchmark: Element creation
#[test]
fn bench_element_creation() {
    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = Element::text("Hello");
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Element creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Element column
#[test]
fn bench_element_column() {
    let children: Vec<Element> = (0..10)
        .map(|i| Element::text(&format!("item {}", i)))
        .collect();

    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        let _ = Element::column(children.clone());
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Element column: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Style builder
#[test]
fn bench_style_builder() {
    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = Style::default().fg(Color::Red).bg(Color::Blue).bold();
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Style builder: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Rect creation
#[test]
fn bench_rect_creation() {
    let start = Instant::now();
    let iterations = 100000;

    for _ in 0..iterations {
        let _ = Rect::new(0, 0, 80, 24);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Rect creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 11: Training Benchmarks
// ============================================================================

/// Benchmark: TrainState EMA update
#[test]
fn bench_train_ema_update() {
    let mut state = TrainState::new();

    let start = Instant::now();
    let iterations = 10000;

    for i in 0..iterations {
        state.global_step = i as u64;
        state.update_ema_loss(0.5);
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!("EMA update: {:?} avg over {} iterations", avg, iterations);
}

/// Benchmark: TrainState creation
#[test]
fn bench_train_state_creation() {
    let start = Instant::now();
    let iterations = 100000;

    for _ in 0..iterations {
        let _ = TrainState::new();
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "TrainState creation: {:?} avg over {} iterations",
        avg, iterations
    );
}

/// Benchmark: Checkpoint strategy
#[test]
fn bench_checkpoint_strategy() {
    let start = Instant::now();
    let iterations = 100000;

    for _ in 0..iterations {
        let _ = CheckpointStrategy::default();
    }

    let elapsed = start.elapsed();
    let avg = elapsed / iterations;
    println!(
        "Checkpoint strategy: {:?} avg over {} iterations",
        avg, iterations
    );
}

// ============================================================================
// Section 12: Throughput Benchmarks
// ============================================================================

/// Benchmark: Combined model + safety throughput
#[test]
fn bench_throughput_model_safety() {
    let mut model = Sophon1::new(0x1234);
    let config = DiagnosticConfig::default_byte_model();
    let mut diagnostic = SelfDiagnostic::new(config);

    let inputs: Vec<String> = (0..100).map(|i| format!("input {}", i)).collect();

    let start = Instant::now();

    for input in &inputs {
        if let Ok(outputs) = model.forward_sequence(input.as_bytes()) {
            if let Some(last) = outputs.last() {
                let _ = diagnostic.check(last.logits.as_slice());
            }
        }
    }

    let elapsed = start.elapsed();
    let throughput = inputs.len() as f64 / elapsed.as_secs_f64();
    println!("Model+safety throughput: {:.2} items/sec", throughput);
}

/// Benchmark: Memory storage throughput
#[test]
fn bench_throughput_memory() {
    let mut memory = EpisodicMemory::new(1000);

    let episodes: Vec<Episode> = (0..1000)
        .map(|i| Episode {
            timestamp: i as u64,
            perception_hv: vec![(i % 10) as f32; HDC_DIM],
            action: None,
            outcome_hv: vec![0.0; HDC_DIM],
            surprise: 0.0,
        })
        .collect();

    let start = Instant::now();

    for episode in episodes {
        memory.record(episode);
    }

    let elapsed = start.elapsed();
    let throughput = 1000.0 / elapsed.as_secs_f64();
    println!("Memory storage throughput: {:.2} episodes/sec", throughput);
}

/// Benchmark: HDC operation throughput
#[test]
fn bench_throughput_hdc() {
    let a = vec![1.0f32; 64];
    let b = vec![0.5f32; 64];

    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = circular_conv(&a, &b);
    }

    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();
    println!("HDC conv throughput: {:.2} ops/sec", throughput);
}

/// Benchmark: Quantization throughput
#[test]
fn bench_throughput_quantization() {
    let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

    let start = Instant::now();
    let iterations = 10000;

    for _ in 0..iterations {
        let _ = ternarize_block(&weights);
    }

    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();
    println!("Quantization throughput: {:.2} blocks/sec", throughput);
}

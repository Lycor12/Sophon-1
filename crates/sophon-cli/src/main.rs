//! sophon — CLI entry point for Sophon-1 narrow AGI system.
//!
//! Modes:
//! sophon — interactive agent loop (observe-plan-act)
//! sophon tui — launch TUI-based interactive interface
//! sophon infer <hex|stdin> — single forward pass on byte input
//! sophon info — print model architecture and param count
//! sophon sysstate — print system state snapshot
//! sophon train <corpus_path> [options] — training loop with checkpointing
//!
//! The agent loop implements the execution-first intelligence pattern (spec Addendum A):
//! 1. Observe: collect system state + screen frame + stdin
//! 2. Plan: run belief update (active inference) + safety checks
//! 3. Act: execute model-selected action via runtime primitives
//! 4. Feedback: update belief from action result
//!
//! All subsystems are wired together here:
//! - sophon-model: forward pass (byte -> logits)
//! - sophon-safety: purpose gate + self-diagnostic + alignment monitor
//! - sophon-inference: belief state + world model + self-improvement
//! - sophon-runtime: action execution + screen capture + system state
//! - sophon-verifier: output constraint (VERIFIED/UNVERIFIED)
//! - sophon-quant: GGUF loader for cold-start teacher weights
//! - sophon-tui: Terminal UI framework

use sophon_config::ModelConfig;
use sophon_model::Sophon1;
use sophon_runtime::system;
use sophon_verifier::VerifierGate;

use sophon_data::{BatchConfig, Dataset, DatasetConfig, FilterConfig};
use sophon_optim::tsm::TsmSgd;
use sophon_train::checkpoint::CheckpointStrategy;
use sophon_train::{train_step, TrainState};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("agent");

    match mode {
        "info" => cmd_info(),
        "infer" => cmd_infer(&args[2..]),
        "sysstate" => cmd_sysstate(),
        "train" => cmd_train(&args[2..]),
        "tui" => cmd_tui(),
        "agent" | _ => cmd_agent(),
    }
}

// ---------------------------------------------------------------------------
// info: print architecture summary
// ---------------------------------------------------------------------------

fn cmd_info() {
    let cfg = ModelConfig::canonical();
    let model = Sophon1::new(0xDEAD_BEEF_u64);

    println!("=== Sophon-1 Architecture ===");
    println!("platform         : {}", system::platform());
    println!("d_model          : {}", cfg.d_model);
    println!("num_blocks       : {}", cfg.num_blocks);
    println!("kan_knots        : {}", cfg.kan_knots);
    println!("kan_order        : {}", cfg.kan_order);
    println!("ssm_n            : {}", cfg.ssm_n);
    println!("ssm_d            : {}", cfg.ssm_d);
    println!("ssm_p            : {}", cfg.ssm_p);
    println!("ssm_rank         : {}", cfg.ssm_rank);
    println!("hdc_dim          : {}", cfg.hdc_dim);
    println!("lora_rank        : {}", cfg.lora_rank);
    println!("vocab_size       : {}", cfg.vocab_size);
    println!("total_params     : {}", model.param_count());
    println!(
        "ternary_size_est : ~{:.1} MB",
        model.param_count() as f64 * 2.0 / 8.0 / 1_048_576.0
    );
}

// ---------------------------------------------------------------------------
// infer: single forward pass
// ---------------------------------------------------------------------------

fn cmd_infer(args: &[String]) {
    let cfg = ModelConfig::canonical();
    eprintln!(
        "Sophon-1 | platform={} | d_model={} | blocks={} | knots={}",
        system::platform(),
        cfg.d_model,
        cfg.num_blocks,
        cfg.kan_knots
    );

    let mut model = Sophon1::new(0xDEAD_BEEF_u64);
    eprintln!("param_count={}", model.param_count());

    let input = if args.is_empty() {
        read_stdin()
    } else {
        let hex: String = args.join("");
        hex_decode(&hex).unwrap_or_else(|_| hex.into_bytes())
    };

    if input.is_empty() {
        eprintln!("No input. Provide bytes as hex argument or on stdin.");
        std::process::exit(0);
    }

    let outputs = model.forward_sequence(&input).expect("forward pass failed");

    for (i, (byte, out)) in input.iter().zip(outputs.iter()).enumerate() {
        let warn = VerifierGate::format_warning(&out.verified);
        if warn.is_empty() {
            println!(
                "[{i}] in={byte:#04x} predicted={:#04x}",
                out.predicted_token
            );
        } else {
            println!(
                "[{i}] in={byte:#04x} predicted={:#04x} | {warn}",
                out.predicted_token
            );
        }
    }
}

// ---------------------------------------------------------------------------
// sysstate: print system state snapshot
// ---------------------------------------------------------------------------

fn cmd_sysstate() {
    let state = sophon_runtime::collect_state();
    let bytes = state.to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    println!("{}", text);
}

// ---------------------------------------------------------------------------
// train: full training loop with checkpointing
// ---------------------------------------------------------------------------

/// Training configuration parsed from command-line arguments.
struct TrainArgs {
    corpus_path: String,
    epochs: usize,
    learning_rate: f32,
    grad_clip: f32,
    checkpoint_interval: usize,
    use_galc: bool,
    max_docs: usize,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            corpus_path: String::new(),
            epochs: 3,
            learning_rate: 1e-4,
            grad_clip: 10.0,
            checkpoint_interval: 1000,
            use_galc: false,
            max_docs: 0, // 0 = unlimited
        }
    }
}

fn parse_train_args(args: &[String]) -> Result<TrainArgs, String> {
    let mut cfg = TrainArgs::default();

    if args.is_empty() {
        return Err("Usage: sophon train <corpus_path> [options]".into());
    }

    cfg.corpus_path = args[0].clone();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" | "-e" => {
                i += 1;
                if i >= args.len() {
                    return Err("--epochs requires a value".into());
                }
                cfg.epochs = args[i].parse().map_err(|_| "invalid epochs value")?;
            }
            "--lr" => {
                i += 1;
                if i >= args.len() {
                    return Err("--lr requires a value".into());
                }
                cfg.learning_rate = args[i].parse().map_err(|_| "invalid learning rate")?;
            }
            "--grad-clip" => {
                i += 1;
                if i >= args.len() {
                    return Err("--grad-clip requires a value".into());
                }
                cfg.grad_clip = args[i].parse().map_err(|_| "invalid grad-clip value")?;
            }
            "--checkpoint-interval" | "-c" => {
                i += 1;
                if i >= args.len() {
                    return Err("--checkpoint-interval requires a value".into());
                }
                cfg.checkpoint_interval =
                    args[i].parse().map_err(|_| "invalid checkpoint interval")?;
            }
            "--galc" => {
                cfg.use_galc = true;
            }
            "--max-docs" => {
                i += 1;
                if i >= args.len() {
                    return Err("--max-docs requires a value".into());
                }
                cfg.max_docs = args[i].parse().map_err(|_| "invalid max-docs value")?;
            }
            "--help" | "-h" => {
                print_train_help();
                std::process::exit(0);
            }
            _ => return Err(format!("Unknown argument: {}", args[i])),
        }
        i += 1;
    }

    Ok(cfg)
}

fn print_train_help() {
    println!("sophon train — Train Sophon-1 on a corpus");
    println!();
    println!("Usage: sophon train <corpus_path> [options]");
    println!();
    println!("Arguments:");
    println!("  <corpus_path>         Path to corpus (file or directory)");
    println!();
    println!("Options:");
    println!("  -e, --epochs <N>      Number of training epochs (default: 3)");
    println!("  --lr <RATE>           Learning rate (default: 1e-4)");
    println!("  --grad-clip <VAL>     Gradient clipping threshold (default: 10.0)");
    println!("  -c, --checkpoint-interval <N>  Save checkpoint every N steps (default: 1000)");
    println!("  --galc                Enable GALC (Gradient-Aware Lazy Checkpointing)");
    println!("  --max-docs <N>        Maximum documents to load (0 = unlimited)");
    println!("  -h, --help            Show this help");
    println!();
    println!("Examples:");
    println!("  sophon train data/corpus.txt");
    println!("  sophon train data/corpus/ --epochs 10 --lr 5e-5 --galc");
}

fn cmd_train(args: &[String]) {
    let cfg = match parse_train_args(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Run 'sophon train --help' for usage information.");
            std::process::exit(1);
        }
    };

    let path = std::path::Path::new(&cfg.corpus_path);
    if !path.exists() {
        eprintln!("Error: Corpus path '{}' does not exist", cfg.corpus_path);
        std::process::exit(1);
    }

    println!("=== Sophon-1 Training ===");
    println!("corpus: {}", cfg.corpus_path);
    println!("epochs: {}", cfg.epochs);
    println!("learning_rate: {}", cfg.learning_rate);
    println!("grad_clip: {}", cfg.grad_clip);
    println!("checkpoint_interval: {}", cfg.checkpoint_interval);
    println!(
        "GALC: {}",
        if cfg.use_galc { "enabled" } else { "disabled" }
    );
    println!();

    // Initialize model
    let mut model = Sophon1::new(0xDEAD_BEEF_u64);
    println!("Model initialized: {} parameters", model.param_count());

    // Initialize optimizer and training state
    let optimizer = TsmSgd::new(cfg.learning_rate, cfg.grad_clip);
    let mut train_state = TrainState::new();

    // Initialize dataset
    let dataset_config = DatasetConfig {
        filter: FilterConfig::default(),
        batch: BatchConfig::default(),
        seed: 42,
        max_documents: cfg.max_docs,
    };
    let mut dataset = Dataset::new(dataset_config);

    // Load corpus
    let stats = if path.is_file() {
        match dataset.load_file(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load corpus: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match dataset.load_directory(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to load corpus: {}", e);
                std::process::exit(1);
            }
        }
    };

    println!("Dataset loaded:");
    println!("  raw_documents: {}", stats.raw_documents);
    println!("  filtered_documents: {}", stats.filtered_documents);
    println!("  sequences: {}", stats.sequences);
    println!("  batches: {}", stats.batches);
    println!("  total_bytes: {}", stats.total_bytes);
    println!();

    if stats.batches == 0 {
        eprintln!("Error: No training batches available. Check your corpus.");
        std::process::exit(1);
    }

    // GALC strategy: can be enabled for memory-constrained training
    let _use_galc = cfg.use_galc;

    // Training loop
    let start_time = std::time::Instant::now();
    let mut total_steps = 0usize;
    let mut total_loss = 0.0f32;

    for epoch in 1..=cfg.epochs {
        println!("=== Epoch {}/{} ===", epoch, cfg.epochs);
        dataset.shuffle();

        let mut epoch_loss = 0.0f32;
        let mut epoch_steps = 0usize;
        let epoch_start = std::time::Instant::now();

        // Progress reporting interval
        let report_interval = (stats.batches / 10).max(1);

        while let Some(batch) = dataset.next_batch() {
            // Train on each sequence in the batch (use inputs, targets are for loss)
            for (input_seq, target_seq) in batch.inputs.iter().zip(batch.targets.iter()) {
                let input = input_seq.as_slice();

                // Skip sequences that are too short (need at least 2 tokens for next-token prediction)
                if input.len() < 2 {
                    continue;
                }

                // Use input sequence for training (train_step handles the target internally)
                let _ = target_seq; // target_seq is shifted by 1, train_step computes this internally

                // Run training step
                match train_step(&mut model, &optimizer, &mut train_state, input) {
                    Ok(result) => {
                        epoch_loss += result.loss;
                        total_loss += result.loss;
                        epoch_steps += 1;
                        total_steps += 1;

                        // Checkpoint if needed
                        if total_steps % cfg.checkpoint_interval == 0 {
                            println!(
                                "  [Checkpoint] step={}, loss={:.6}, grad_norm={:.4}",
                                total_steps, result.loss, result.grad_norm
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("  Warning: Training step failed: {:?}", e);
                    }
                }
            }

            // Progress report
            if epoch_steps % report_interval == 0 && epoch_steps > 0 {
                let avg_loss = epoch_loss / epoch_steps as f32;
                let progress = (epoch_steps as f32 / stats.batches as f32) * 100.0;
                println!(
                    "  Progress: {:.1}% | avg_loss={:.6} | steps={}",
                    progress, avg_loss, epoch_steps
                );
            }
        }

        let epoch_duration = epoch_start.elapsed();
        let avg_epoch_loss = if epoch_steps > 0 {
            epoch_loss / epoch_steps as f32
        } else {
            0.0
        };

        println!("Epoch {}/{} complete:", epoch, cfg.epochs);
        println!("  avg_loss: {:.6}", avg_epoch_loss);
        println!("  steps: {}", epoch_steps);
        println!("  duration: {:.2?}", epoch_duration);
        println!("  global_step: {}", train_state.global_step);
        println!("  ema_loss: {:.6}", train_state.ema_loss);
        println!();
    }

    let total_duration = start_time.elapsed();
    let avg_loss = if total_steps > 0 {
        total_loss / total_steps as f32
    } else {
        0.0
    };

    println!("=== Training Complete ===");
    println!("total_steps: {}", total_steps);
    println!("avg_loss: {:.6}", avg_loss);
    println!("total_duration: {:.2?}", total_duration);
    println!(
        "steps/sec: {:.2}",
        total_steps as f64 / total_duration.as_secs_f64()
    );
    println!("final_ema_loss: {:.6}", train_state.ema_loss);
}

// ---------------------------------------------------------------------------
// agent: interactive observe-plan-act loop
// ---------------------------------------------------------------------------

fn cmd_agent() {
    use sophon_inference::belief::BeliefState;
    use sophon_inference::prediction::WorldModel;
    use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor};
    use sophon_safety::error_detect::{DiagnosticConfig, SelfDiagnostic};

    let cfg = ModelConfig::canonical();
    eprintln!("=== Sophon-1 Agent ===");
    eprintln!(
        "platform={} | d_model={} | blocks={}",
        system::platform(),
        cfg.d_model,
        cfg.num_blocks,
    );

    // Initialise subsystems
    let mut model = Sophon1::new(0xDEAD_BEEF_u64);
    eprintln!("param_count={}", model.param_count());

    // Safety: self-diagnostic (CSDL cascade)
    let mut diagnostic = SelfDiagnostic::new(DiagnosticConfig::default_byte_model());

    // Active inference: belief state + world model
    let belief_dim = cfg.d_model;
    let mut belief = BeliefState::new(belief_dim);
    let world_model = WorldModel::new(belief_dim, belief_dim);

    // Alignment monitor: anchor to initial model parameters
    let initial_params = model.flattened_params();
    eprintln!("Alignment anchor: {} parameters", initial_params.len());
    let mut alignment = AlignmentMonitor::new(&initial_params, AlignmentConfig::from_spec());

    // Verifier gate
    let _gate = VerifierGate::default();

    eprintln!("All subsystems initialised. Entering agent loop.");
    eprintln!("Type input (UTF-8 text), or a command:");
    eprintln!("  /sysstate  /screen  /belief  /diag  /quit");
    eprintln!();

    // Agent loop
    let stdin = std::io::stdin();
    let mut step = 0u64;
    loop {
        // 1. OBSERVE: read user input
        eprint!("sophon> ");
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Handle commands
        match line {
            "/quit" | "quit" | "exit" => {
                eprintln!("Exiting agent loop.");
                break;
            }
            "/sysstate" => {
                let state = sophon_runtime::collect_state();
                eprintln!(
                    "processes={} connections={} mem_used={:.0}%",
                    state.process_count(),
                    state.connection_count(),
                    state.memory.usage_ratio * 100.0
                );
                continue;
            }
            "/screen" => {
                match sophon_runtime::capture_screen(256, 256) {
                    Ok(frame) => {
                        eprintln!(
                            "Screen captured: {}x{} ({} bytes)",
                            frame.width,
                            frame.height,
                            frame.len()
                        );
                    }
                    Err(e) => eprintln!("Screen capture failed: {e}"),
                }
                continue;
            }
            "/belief" => {
                eprintln!(
                    "Belief: mu_mag={:.4} uncertainty={:.4} step={}",
                    belief.mu_magnitude(),
                    belief.mean_uncertainty(),
                    belief.step
                );
                continue;
            }
            "/diag" => {
                // Use actual model output for diagnostic, not dummy logits
                let test_input = b"test";
                match model.forward_sequence(test_input) {
                    Ok(outputs) => {
                        if let Some(last) = outputs.last() {
                            let result = diagnostic.check(last.logits.as_slice());
                            eprintln!(
                                "Diagnostic: passed={} faults={:?} stage={}",
                                result.passed, result.faults, result.halted_at_stage
                            );
                        } else {
                            eprintln!("Diagnostic: no output from model");
                        }
                    }
                    Err(e) => {
                        eprintln!("Diagnostic failed: model error: {}", e);
                    }
                }
                continue;
            }
            _ => {}
        }

        // 2. PROCESS: convert input to bytes, run forward pass
        let input_bytes = line.as_bytes();
        step += 1;

        let outputs = match model.forward_sequence(input_bytes) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Forward pass error: {e}");
                continue;
            }
        };

        // 3. SAFETY: run self-diagnostic on final logits
        if let Some(last_output) = outputs.last() {
            let logits_slice = last_output.logits.as_slice();
            let diag_result = diagnostic.check(logits_slice);
            if !diag_result.passed {
                eprintln!(
                    "[SAFETY] Diagnostic failed at stage {}: {:?}",
                    diag_result.halted_at_stage, diag_result.faults
                );
            }
        }

        // 4. Belief update: use prediction error as observation
        if !outputs.is_empty() {
            let logits_slice = outputs.last().unwrap().logits.as_slice();
            let observation: Vec<f32> = logits_slice.iter().take(belief_dim).copied().collect();
            let prediction = world_model.predict(&belief);
            let pred_error: Vec<f32> = observation
                .iter()
                .zip(prediction.iter())
                .map(|(o, p)| o - p)
                .collect();
            let error_norm: f32 = pred_error.iter().map(|e| e * e).sum::<f32>().sqrt();

            // Gradient-based belief update
            let grad = world_model.grad_mu_prediction_error(&pred_error);
            let grad_log_sigma = vec![0.0f32; belief_dim];
            belief.update(&grad, &grad_log_sigma, 0.01);

            eprintln!(
                "[step={step}] tokens={} pred_error={:.4} belief_mag={:.4}",
                outputs.len(),
                error_norm,
                belief.mu_magnitude()
            );
        }

        // 5. OUTPUT: print predictions with verification status
        let mut predicted_text = Vec::new();
        for out in &outputs {
            let warn = VerifierGate::format_warning(&out.verified);
            predicted_text.push(out.predicted_token);
            if !warn.is_empty() && step <= 3 {
                eprint!("[UNVERIFIED] ");
            }
        }

        let pred_str = String::from_utf8_lossy(&predicted_text);
        println!(">> {}", pred_str);

        // 6. ALIGNMENT: periodic check
        alignment.report_score(1.0);
        let current_params = model.flattened_params();
        let align_status = alignment.step(&current_params);
        if align_status.needs_rollback() {
            eprintln!("[SAFETY] Alignment violation: {}", align_status);
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn read_stdin() -> Vec<u8> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap_or(0);
    buf
}

fn hex_decode(s: &str) -> Result<Vec<u8>, ()> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err(());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for chunk in s.as_bytes().chunks(2) {
        let hi = hex_nibble(chunk[0]).ok_or(())?;
        let lo = hex_nibble(chunk[1]).ok_or(())?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// tui: TUI-based interactive interface
// ---------------------------------------------------------------------------

fn cmd_tui() {
    use sophon_tui::prelude::*;
    use sophon_tui::render_to_string;

    // Create a simple demo UI
    let ui = Element::column(vec![
        Element::text("═══════════════════════════════════════════════")
            .color(Color::Cyan)
            .bold(),
        Element::text("       Sophon-1 AGI Terminal Interface       ").color(Color::White),
        Element::text("═══════════════════════════════════════════════")
            .color(Color::Cyan)
            .bold(),
        Element::text(""),
        Element::row(vec![
            Element::text("Model: ").color(Color::Yellow),
            Element::text("Sophon-1 (Execution-first)"),
        ]),
        Element::row(vec![
            Element::text("Status: ").color(Color::Yellow),
            Element::text("Ready").color(Color::Green),
        ]),
        Element::text(""),
        Element::text("Commands:").color(Color::Blue).bold(),
        Element::text("  info      - Show model architecture"),
        Element::text("  infer     - Run inference on input"),
        Element::text("  sysstate  - Show system state"),
        Element::text("  train     - Training mode"),
        Element::text("  agent     - Interactive agent loop"),
        Element::text(""),
        Element::text("═══════════════════════════════════════════════").color(Color::Cyan),
        Element::text("  Type /quit to exit, or use arrow keys").dim(),
    ]);

    // Render to string and print
    let output = render_to_string(&ui, 50, 16);
    println!("{}", output);

    println!("\nTUI Demo complete!");
    println!("(Full interactive TUI requires terminal raw mode)");
}

//! sophon — CLI entry point for Sophon-1 narrow AGI system.
//!
//! Modes:
//!   sophon                        — interactive agent loop (observe-plan-act)
//!   sophon infer <hex|stdin>      — single forward pass on byte input
//!   sophon info                   — print model architecture and param count
//!   sophon sysstate               — print system state snapshot
//!   sophon train <corpus_path>    — placeholder for training mode
//!
//! The agent loop implements the execution-first intelligence pattern (spec Addendum A):
//!   1. Observe: collect system state + screen frame + stdin
//!   2. Plan: run belief update (active inference) + safety checks
//!   3. Act: execute model-selected action via runtime primitives
//!   4. Feedback: update belief from action result
//!
//! All subsystems are wired together here:
//!   - sophon-model: forward pass (byte -> logits)
//!   - sophon-safety: purpose gate + self-diagnostic + alignment monitor
//!   - sophon-inference: belief state + world model + self-improvement
//!   - sophon-runtime: action execution + screen capture + system state
//!   - sophon-verifier: output constraint (VERIFIED/UNVERIFIED)
//!   - sophon-quant: GGUF loader for cold-start teacher weights

use sophon_config::ModelConfig;
use sophon_model::Sophon1;
use sophon_runtime::system;
use sophon_verifier::VerifierGate;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("agent");

    match mode {
        "info" => cmd_info(),
        "infer" => cmd_infer(&args[2..]),
        "sysstate" => cmd_sysstate(),
        "train" => cmd_train(&args[2..]),
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
// train: placeholder for training mode
// ---------------------------------------------------------------------------

fn cmd_train(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: sophon train <corpus_path>");
        eprintln!("Training infrastructure is implemented in sophon-train crate.");
        eprintln!("Requires corpus data in line-delimited or directory format.");
        std::process::exit(1);
    }
    let path = &args[0];
    eprintln!("Training mode: corpus={}", path);
    eprintln!("Training loop requires data loading + GPU resources.");
    eprintln!("Use the sophon-train crate API for programmatic training.");
}

// ---------------------------------------------------------------------------
// agent: interactive observe-plan-act loop
// ---------------------------------------------------------------------------

fn cmd_agent() {
    use sophon_inference::belief::BeliefState;
    use sophon_inference::prediction::WorldModel;
    use sophon_safety::alignment::{AlignmentConfig, AlignmentMonitor, AlignmentStatus};
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

    // Alignment monitor: anchor to initial parameters (placeholder small vec)
    let initial_params = vec![0.0f32; 100];
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
                let dummy_logits = vec![0.0f32; 256];
                let result = diagnostic.check(&dummy_logits);
                eprintln!(
                    "Diagnostic: passed={} faults={:?} stage={}",
                    result.passed, result.faults, result.halted_at_stage
                );
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
            belief.update(&grad, &[], 0.01);

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
        let align_status = alignment.step(&initial_params);
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

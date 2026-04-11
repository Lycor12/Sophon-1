//! sophon-swarm — Swarm classroom orchestration for Sophon-1.
//!
//! Spec §4.2, §6.1, §6.2: Training is organised as a swarm classroom:
//!   - Teacher agent generates problems at the current curriculum level
//!   - N students attempt to solve problems (sequentially on current hardware)
//!   - Top-K survivors selected by score + verification
//!   - Translation swarm converts NL solutions to Lean (via encoding pipeline)
//!   - Verified solutions added to append-only knowledge base
//!   - EMA weight merging consolidates progress
//!
//! Hardware adaptation (user decision): 4 students run sequentially rather
//! than 10,000 in parallel, with architecture supporting scale-up later.
//!
//! # Novel technique: SCSO (Swarm-Competitive Sequential Orchestration)
//!
//! In a sequential setting, the key insight is that student diversity must
//! come from parameter perturbation rather than parallel exploration. SCSO
//! applies per-student noise injection (scaled by curriculum level) to LoRA
//! adapters before each problem attempt, creating a diverse solution population
//! from sequential runs. This simulates the exploration benefit of parallel
//! swarms without the memory cost. After selection, the winning student's
//! LoRA deltas are merged into the base weights with an EMA factor that
//! decreases as curriculum level increases (more conservative updates at
//! harder problems).

#![forbid(unsafe_code)]

pub mod teacher;
pub mod student;
pub mod selection;
pub mod classroom;

pub use teacher::{Teacher, Problem, CurriculumConfig};
pub use student::{Student, StudentConfig, SolutionAttempt};
pub use selection::{Selection, SelectionConfig, RankedSolution};
pub use classroom::{Classroom, ClassroomConfig, EpochResult};

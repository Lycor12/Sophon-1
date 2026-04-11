//! sophon-train — End-to-end training loop for Sophon-1.
//!
//! Wires together:
//!   - Model forward-with-cache (embedding → 16 blocks → head)
//!   - Cross-entropy loss + gradient
//!   - Full backward pass (head → blocks → embedding)
//!   - TSM-SGD parameter updates with per-group configuration
//!
//! Novel optimisation — GALC (Gradient-Aware Lazy Checkpointing):
//!   Rather than checkpointing at fixed block intervals (standard gradient
//!   checkpointing), GALC monitors the gradient norm flowing through the
//!   residual stream. If the accumulated gradient norm exceeds a threshold
//!   (indicating high-loss regions), subsequent blocks are recomputed
//!   from cached inputs rather than stored activations. In low-gradient
//!   regions, full caches are retained. This adapts the memory/compute
//!   tradeoff to the loss landscape: more recomputation where it matters
//!   (high-loss), less where gradients are small. For the initial training
//!   phase, GALC defaults to full caching (no recompute) since the model
//!   is small enough to fit all 16 block caches in memory.

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod checkpoint_io;
pub mod state;
pub mod step;

pub use checkpoint_io::{load_checkpoint, save_checkpoint, CheckpointError};
pub use state::TrainState;
pub use step::{train_step, TrainStepResult};

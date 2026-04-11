//! sophon-runtime — Windows-native OS interaction layer.
//!
//! Spec §5.1 and §5.2: execution-first intelligence, direct system interaction.
//! Spec §B: No API dependency. All OS primitives are native, offline.
//!
//! Primitives provided:
//!   - Filesystem: read, write, append, list, stat, delete
//!   - Process:    spawn, wait, kill, stdout/stderr capture
//!   - System:     env vars, working directory, hostname
//!   - Screen:     DBSC screen capture + BADS downsampling + VLA input (§5.2)
//!   - SysState:   Process table, memory, network snapshot (§5.3.1)
//!   - Action:     structured action type for the agent loop
//!
//! The interfaces are defined here as Rust traits and concrete types.
//! Windows-specific implementations use handwritten Win32 FFI (no external crate);
//! POSIX implementations follow the same interfaces.
//!
//! `unsafe` is allowed only in screen.rs for Win32 FFI (CreateDCA, BitBlt,
//! GetDIBits, SendInput). All other modules remain safe Rust.

pub mod action;
pub mod fs;
pub mod process;
pub mod screen;
pub mod sysstate;
pub mod system;

pub use action::{Action, ActionResult};
pub use screen::{capture_screen, execute_input, InputAction, ScreenError, ScreenFrame};
pub use sysstate::{collect_state, SystemState};

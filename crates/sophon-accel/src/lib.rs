//! sophon-accel — Hardware acceleration and unsafe infrastructure.
//!
//! This is the ONLY crate in the Sophon workspace that permits unsafe code.
//! All unsafe operations are isolated here:
//!   - SIMD intrinsics (AVX-512, AVX2, SSE4.2)
//!   - Memory-mapped file I/O (Win32 CreateFileMapping / Unix mmap)
//!   - Cache-aligned allocations
//!   - Work-stealing thread pool
//!   - u64 ternary weight packing (32 weights per u64)
//!
//! All other crates maintain `#![forbid(unsafe_code)]`.

pub mod aligned;
pub mod detect;
pub mod mmap;
pub mod pack64;
pub mod scheduler;
pub mod simd;

pub use aligned::{AlignedBlock, AlignedVec};
pub use detect::{detect_simd, SimdLevel};
pub use mmap::MappedFile;
pub use pack64::{pack_32_ternary, unpack_32_ternary};
pub use scheduler::ThreadPool;
pub use simd::ternary_matvec;

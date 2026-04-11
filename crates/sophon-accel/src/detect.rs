//! Runtime SIMD capability detection.
//!
//! Three-tier dispatch: AVX-512 > AVX2 > SSE4.2 > Scalar.
//! Detection result is cached in an atomic for zero-cost subsequent queries.

use std::sync::atomic::{AtomicU8, Ordering};

/// SIMD capability levels, ordered from most capable to least.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SimdLevel {
    /// No SIMD support - scalar fallback using standard Rust operations.
    /// This is the baseline implementation that works on all platforms.
    Scalar = 0,

    /// SSE 4.2 support - 128-bit SIMD instructions.
    /// Provides basic vectorized operations for x86/x86_64 processors.
    Sse42 = 1,

    /// AVX2 support - 256-bit SIMD instructions.
    /// Enables wider vector operations and fused multiply-add instructions.
    Avx2 = 2,

    /// AVX-512 support - 512-bit SIMD instructions.
    /// Maximum throughput with widest vector registers and additional mask registers.
    /// Requires both AVX-512F and AVX-512BW feature flags.
    Avx512 = 3,
}

// 0xFF = not yet detected
static CACHED_LEVEL: AtomicU8 = AtomicU8::new(0xFF);

/// Detect the highest available SIMD level on this CPU.
/// Result is cached after first call.
pub fn detect_simd() -> SimdLevel {
    let cached = CACHED_LEVEL.load(Ordering::Relaxed);
    if cached != 0xFF {
        return match cached {
            3 => SimdLevel::Avx512,
            2 => SimdLevel::Avx2,
            1 => SimdLevel::Sse42,
            _ => SimdLevel::Scalar,
        };
    }

    let level = detect_impl();
    CACHED_LEVEL.store(level as u8, Ordering::Relaxed);
    level
}

/// Reset cached detection (for testing).
pub fn reset_detection() {
    CACHED_LEVEL.store(0xFF, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
fn detect_impl() -> SimdLevel {
    // Use CPUID to check feature flags.
    // We avoid std::is_x86_feature_detected! because it may not cover AVX-512
    // sub-features we need (like VNNI). Instead we use raw CPUID.

    // Safety: cpuid is always available on x86_64.
    let (_, ebx7, ecx7, _) = unsafe { cpuid(7, 0) };
    let (_, _, ecx1, _) = unsafe { cpuid(1, 0) };

    // AVX-512F = EBX7 bit 16, AVX-512BW = EBX7 bit 30
    let avx512f = (ebx7 >> 16) & 1 == 1;
    let avx512bw = (ebx7 >> 30) & 1 == 1;

    // AVX2 = EBX7 bit 5
    let avx2 = (ebx7 >> 5) & 1 == 1;

    // SSE4.2 = ECX1 bit 20
    let sse42 = (ecx1 >> 20) & 1 == 1;

    // Also check OS XSAVE support (ECX1 bit 27) and verify XCR0 bits
    let osxsave = (ecx1 >> 27) & 1 == 1;

    if osxsave {
        let xcr0 = unsafe { xgetbv(0) };
        let ymm_enabled = (xcr0 & 0x06) == 0x06; // XMM + YMM
        let zmm_enabled = (xcr0 & 0xE0) == 0xE0; // opmask + ZMM_Hi256 + Hi16_ZMM

        if avx512f && avx512bw && zmm_enabled {
            return SimdLevel::Avx512;
        }
        if avx2 && ymm_enabled {
            return SimdLevel::Avx2;
        }
    }

    if sse42 {
        return SimdLevel::Sse42;
    }

    SimdLevel::Scalar
}

/// Execute the CPUID instruction to query processor capabilities.
///
/// # Safety
///
/// This function uses inline assembly to execute the CPUID instruction.
/// It is only safe to call on x86_64 architectures where CPUID is supported.
///
/// The caller must ensure:
/// - The code runs on x86_64 architecture
/// - The processor supports the CPUID instruction (all modern x86_64 CPUs do)
/// - The leaf and sub_leaf values are valid for the processor
///
/// CPUID may modify the EBX/RBX register, which is used by LLVM's internal
/// stack canary on some platforms. This implementation preserves RBX.
#[cfg(target_arch = "x86_64")]
unsafe fn cpuid(leaf: u32, sub_leaf: u32) -> (u32, u32, u32, u32) {
    let (eax, ebx, ecx, edx): (u32, u32, u32, u32);
    unsafe {
        std::arch::asm!(
            "mov {tmp_rbx:r}, rbx",
            "cpuid",
            "xchg {tmp_rbx:r}, rbx",
            tmp_rbx = out(reg) ebx,
            inout("eax") leaf => eax,
            inout("ecx") sub_leaf => ecx,
            out("edx") edx,
        );
    }
    (eax, ebx, ecx, edx)
}

/// Execute the XGETBV instruction to read extended control register values.
///
/// # Safety
///
/// This function uses inline assembly to execute the XGETBV instruction.
/// It is only safe to call on x86_64 architectures where XGETBV is supported.
///
/// The caller must ensure:
/// - The code runs on x86_64 architecture
/// - The OS supports extended state management (indicated by OSXSAVE CPUID bit)
/// - The xcr parameter specifies a valid extended control register (typically 0 for XCR0)
///
/// XGETBV is used to verify that the OS has enabled AVX/AVX-512 registers
/// for use by applications.
#[cfg(target_arch = "x86_64")]
unsafe fn xgetbv(xcr: u32) -> u64 {
    let (eax, edx): (u32, u32);
    unsafe {
        std::arch::asm!(
            "xgetbv",
            inout("ecx") xcr => _,
            out("eax") eax,
            out("edx") edx,
        );
    }
    ((edx as u64) << 32) | (eax as u64)
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_impl() -> SimdLevel {
    SimdLevel::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_deterministic() {
        reset_detection();
        let a = detect_simd();
        let b = detect_simd();
        assert_eq!(a, b);
    }

    #[test]
    fn level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 > SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 > SimdLevel::Scalar);
    }

    #[test]
    fn cached_after_first_call() {
        reset_detection();
        let _ = detect_simd();
        let cached = CACHED_LEVEL.load(std::sync::atomic::Ordering::Relaxed);
        assert_ne!(cached, 0xFF);
    }

    #[test]
    fn detect_returns_valid_level() {
        reset_detection();
        let level = detect_simd();
        assert!(matches!(
            level,
            SimdLevel::Scalar | SimdLevel::Sse42 | SimdLevel::Avx2 | SimdLevel::Avx512
        ));
    }
}

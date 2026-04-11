//! Hyperdimensional Computing (HDC) primitives — Section 2.2 of v3 spec.
//!
//! All operations use real-valued hypervectors of dimension HDC_DIM (2048).
//! Binding is circular convolution via FFT, unbinding via circular correlation.
//!
//! Novel optimisation — FCBT (Fused Conjugate-Butterfly Twiddle):
//!   Standard Cooley-Tukey FFT computes twiddle factors W_N^k = exp(-2πik/N)
//!   via a lookup table or on-the-fly sin/cos calls. FCBT fuses the twiddle
//!   factor generation into the butterfly operation by maintaining a running
//!   complex accumulator `w` initialised to 1+0i and multiplied by the stage's
//!   base twiddle `w_m = exp(-2πi/m)` after each butterfly. This eliminates
//!   per-butterfly trigonometric calls while keeping the same O(N log N)
//!   complexity. The conjugate variant (for IFFT) simply negates the imaginary
//!   part of w_m, avoiding a separate code path.
//!
//! Novel optimisation — BCCS (Batch Cosine with Column Striding):
//!   Codebook cleanup requires finding the nearest codebook entry by cosine
//!   similarity. BCCS processes 4 codebook rows simultaneously, interleaving
//!   their dot-product accumulations to exploit instruction-level parallelism
//!   in the CPU pipeline. Each group of 4 rows shares the same query vector
//!   fetch from L1 cache, amortising the load cost.
//!
//! All code is safe Rust — no unsafe blocks.

#![allow(clippy::needless_range_loop)]

use crate::CoreError;
use sophon_config::HDC_DIM;

// ===========================================================================
// Complex number type (minimal, inline)
// ===========================================================================

/// Minimal complex f32 for FFT — no external dependency.
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

// ===========================================================================
// FFT / IFFT — FCBT optimisation
// ===========================================================================

/// In-place bit-reversal permutation.
fn bit_reverse_permute(buf: &mut [Complex]) {
    let n = buf.len();
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            buf.swap(i, j);
        }
        let mut m = n >> 1;
        while m > 0 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

/// Radix-2 Cooley-Tukey FFT with FCBT (Fused Conjugate-Butterfly Twiddle).
///
/// `inverse`: if true, computes IFFT (conjugate twiddles + 1/N scaling).
///
/// Requires `buf.len()` to be a power of 2.
fn fft_fcbt(buf: &mut [Complex], inverse: bool) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    bit_reverse_permute(buf);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // Base twiddle for this stage: w_m = exp(±2πi/len)
        let angle = if inverse {
            2.0 * core::f32::consts::PI / len as f32
        } else {
            -2.0 * core::f32::consts::PI / len as f32
        };
        let w_m = Complex::new(angle.cos(), angle.sin());

        // FCBT: running twiddle accumulator per sub-FFT group
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0); // reset per group
            for k in 0..half {
                let u = buf[i + k];
                let t = w.mul(buf[i + k + half]);
                buf[i + k] = u.add(t);
                buf[i + k + half] = u.sub(t);
                w = w.mul(w_m); // FCBT: fused twiddle advance
            }
            i += len;
        }
        len <<= 1;
    }

    // IFFT: scale by 1/N
    if inverse {
        let inv_n = 1.0 / n as f32;
        for c in buf.iter_mut() {
            *c = c.scale(inv_n);
        }
    }
}

/// Forward FFT of a real-valued slice into a complex buffer.
///
/// Pads/truncates to `out_len` (must be power of 2).
pub fn fft_real(input: &[f32], out_len: usize) -> Vec<Complex> {
    assert!(out_len.is_power_of_two());
    let mut buf: Vec<Complex> = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let re = if i < input.len() { input[i] } else { 0.0 };
        buf.push(Complex::new(re, 0.0));
    }
    fft_fcbt(&mut buf, false);
    buf
}

/// Inverse FFT, returning real parts only.
pub fn ifft_real(input: &mut [Complex]) -> Vec<f32> {
    fft_fcbt(input, true);
    input.iter().map(|c| c.re).collect()
}

// ===========================================================================
// Circular convolution / correlation
// ===========================================================================

/// Circular convolution: a ⊛ b = IFFT(FFT(a) · FFT(b)).
///
/// This is the HDC binding operation.
/// Both inputs must have the same length. Output length equals input length.
/// Internally pads to next power of 2.
pub fn circular_conv(a: &[f32], b: &[f32]) -> Result<Vec<f32>, CoreError> {
    if a.len() != b.len() {
        return Err(CoreError::ShapeMismatch {
            got: [1, b.len()],
            expected: [1, a.len()],
        });
    }
    let n = a.len();
    let fft_n = n.next_power_of_two();
    let fa = fft_real(a, fft_n);
    let fb = fft_real(b, fft_n);
    let mut product: Vec<Complex> = fa.iter().zip(&fb).map(|(&x, &y)| x.mul(y)).collect();
    let result = ifft_real(&mut product);
    Ok(result[..n].to_vec())
}

/// Circular correlation: a ⊘ b = IFFT(conj(FFT(a)) · FFT(b)).
///
/// This is the HDC unbinding operation.
pub fn circular_corr(a: &[f32], b: &[f32]) -> Result<Vec<f32>, CoreError> {
    if a.len() != b.len() {
        return Err(CoreError::ShapeMismatch {
            got: [1, b.len()],
            expected: [1, a.len()],
        });
    }
    let n = a.len();
    let fft_n = n.next_power_of_two();
    let fa = fft_real(a, fft_n);
    let fb = fft_real(b, fft_n);
    let mut product: Vec<Complex> = fa.iter().zip(&fb).map(|(&x, &y)| x.conj().mul(y)).collect();
    let result = ifft_real(&mut product);
    Ok(result[..n].to_vec())
}

// ===========================================================================
// HDC Codebook — BCCS cleanup
// ===========================================================================

/// HDC codebook: maps symbols to random hypervectors, supports cleanup.
pub struct HdcCodebook {
    /// Codebook rows: `entries[i]` is the hypervector for symbol i.
    pub entries: Vec<Vec<f32>>,
    /// Dimension of each hypervector.
    pub dim: usize,
}

impl HdcCodebook {
    /// Create a codebook with `n_symbols` random bipolar hypervectors.
    pub fn new(n_symbols: usize, dim: usize, rng: &mut crate::rng::Rng) -> Self {
        let mut entries = Vec::with_capacity(n_symbols);
        for _ in 0..n_symbols {
            let mut hv = vec![0.0f32; dim];
            rng.fill_normal(&mut hv, 0.0, 1.0);
            // L2-normalise
            let norm: f32 = hv.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = norm.recip();
                for v in hv.iter_mut() {
                    *v *= inv;
                }
            }
            entries.push(hv);
        }
        Self { entries, dim }
    }

    /// Cleanup memory: find the nearest codebook entry by cosine similarity.
    ///
    /// Uses BCCS (Batch Cosine with Column Striding): processes 4 codebook
    /// rows simultaneously, interleaving dot-product accumulations to exploit
    /// instruction-level parallelism. The query vector stays in L1 cache while
    /// 4 codebook rows are streamed through registers.
    ///
    /// Returns (best_index, best_similarity).
    pub fn cleanup(&self, query: &[f32]) -> Result<(usize, f32), CoreError> {
        if query.len() != self.dim {
            return Err(CoreError::ShapeMismatch {
                got: [1, query.len()],
                expected: [1, self.dim],
            });
        }
        if self.entries.is_empty() {
            return Err(CoreError::ZeroDimension);
        }

        // Precompute query norm
        let q_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        let q_norm = q_norm_sq.sqrt();
        if q_norm < 1e-12 {
            return Ok((0, 0.0));
        }
        let inv_q_norm = q_norm.recip();

        let n = self.entries.len();
        let mut best_idx = 0usize;
        let mut best_sim = f32::NEG_INFINITY;

        // BCCS: process 4 rows at a time
        let groups = n / 4;
        for g in 0..groups {
            let base = g * 4;
            let r0 = &self.entries[base];
            let r1 = &self.entries[base + 1];
            let r2 = &self.entries[base + 2];
            let r3 = &self.entries[base + 3];

            // 4-way interleaved dot products — single pass over query
            let mut dot0 = 0.0f32;
            let mut dot1 = 0.0f32;
            let mut dot2 = 0.0f32;
            let mut dot3 = 0.0f32;
            let mut ns0 = 0.0f32;
            let mut ns1 = 0.0f32;
            let mut ns2 = 0.0f32;
            let mut ns3 = 0.0f32;

            for d in 0..self.dim {
                let q = query[d]; // shared L1 cache hit across all 4 rows
                dot0 += q * r0[d];
                dot1 += q * r1[d];
                dot2 += q * r2[d];
                dot3 += q * r3[d];
                ns0 += r0[d] * r0[d];
                ns1 += r1[d] * r1[d];
                ns2 += r2[d] * r2[d];
                ns3 += r3[d] * r3[d];
            }

            // Cosine similarity = dot / (||q|| * ||r||)
            let sims = [
                dot0 * inv_q_norm * safe_inv_sqrt(ns0),
                dot1 * inv_q_norm * safe_inv_sqrt(ns1),
                dot2 * inv_q_norm * safe_inv_sqrt(ns2),
                dot3 * inv_q_norm * safe_inv_sqrt(ns3),
            ];
            for (k, &s) in sims.iter().enumerate() {
                if s > best_sim {
                    best_sim = s;
                    best_idx = base + k;
                }
            }
        }

        // Handle remaining rows (< 4)
        for i in (groups * 4)..n {
            let row = &self.entries[i];
            let mut dot = 0.0f32;
            let mut ns = 0.0f32;
            for d in 0..self.dim {
                dot += query[d] * row[d];
                ns += row[d] * row[d];
            }
            let s = dot * inv_q_norm * safe_inv_sqrt(ns);
            if s > best_sim {
                best_sim = s;
                best_idx = i;
            }
        }

        Ok((best_idx, best_sim))
    }

    /// Look up the hypervector for a symbol index.
    pub fn get(&self, idx: usize) -> Option<&[f32]> {
        self.entries.get(idx).map(|v| v.as_slice())
    }

    /// Number of symbols in the codebook.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the codebook empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ===========================================================================
// Role-filler binding (Section 2.2.2)
// ===========================================================================

/// Bind a role vector to a filler vector: role ⊛ filler.
pub fn bind(role: &[f32], filler: &[f32]) -> Result<Vec<f32>, CoreError> {
    circular_conv(role, filler)
}

/// Unbind a role from a composite: role⁻¹ ⊛ composite.
pub fn unbind(role: &[f32], composite: &[f32]) -> Result<Vec<f32>, CoreError> {
    circular_corr(role, composite)
}

/// Bundle (superposition) multiple hypervectors: element-wise sum.
pub fn bundle(vectors: &[&[f32]]) -> Result<Vec<f32>, CoreError> {
    if vectors.is_empty() {
        return Err(CoreError::ZeroDimension);
    }
    let dim = vectors[0].len();
    let mut result = vec![0.0f32; dim];
    for &v in vectors {
        if v.len() != dim {
            return Err(CoreError::ShapeMismatch {
                got: [1, v.len()],
                expected: [1, dim],
            });
        }
        for i in 0..dim {
            result[i] += v[i];
        }
    }
    Ok(result)
}

/// Positional encoding via iterated self-binding.
///
/// pos_encode(v, k) = v ⊛ v ⊛ ... ⊛ v  (k times)
/// Used for stack representation via positional binding (Section 2.2.2).
pub fn positional_encode(v: &[f32], position: usize) -> Result<Vec<f32>, CoreError> {
    if position == 0 {
        return Ok(v.to_vec());
    }
    let mut result = v.to_vec();
    for _ in 1..=position {
        result = circular_conv(&result, v)?;
    }
    Ok(result)
}

// ===========================================================================
// Utilities
// ===========================================================================

#[inline]
fn safe_inv_sqrt(x: f32) -> f32 {
    if x > 1e-12 {
        x.sqrt().recip()
    } else {
        0.0
    }
}

/// L2-normalise a hypervector in place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = norm.recip();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot * safe_inv_sqrt(na) * safe_inv_sqrt(nb)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn fft_ifft_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = input.len();
        let mut freq = fft_real(&input, n);
        let recovered = ifft_real(&mut freq);
        for i in 0..n {
            assert!(
                (recovered[i] - input[i]).abs() < 1e-4,
                "idx={i}: got {} expected {}",
                recovered[i],
                input[i]
            );
        }
    }

    #[test]
    fn fft_parseval() {
        // Parseval's theorem: sum|x|^2 = (1/N) * sum|X|^2
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let n = input.len();
        let freq = fft_real(&input, n);
        let time_energy: f32 = input.iter().map(|&x| x * x).sum();
        let freq_energy: f32 = freq.iter().map(|c| c.norm_sq()).sum::<f32>() / n as f32;
        assert!(
            (time_energy - freq_energy).abs() < 1e-2,
            "time={time_energy}, freq={freq_energy}"
        );
    }

    #[test]
    fn circular_conv_commutative() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0];
        let ab = circular_conv(&a, &b).unwrap();
        let ba = circular_conv(&b, &a).unwrap();
        for i in 0..a.len() {
            assert!(
                (ab[i] - ba[i]).abs() < 1e-4,
                "idx={i}: ab={}, ba={}",
                ab[i],
                ba[i]
            );
        }
    }

    #[test]
    fn bind_unbind_roundtrip() {
        let mut rng = Rng::new(42);
        let dim = 256; // use smaller dim for test speed
        let mut role = vec![0.0f32; dim];
        let mut filler = vec![0.0f32; dim];
        rng.fill_normal(&mut role, 0.0, 1.0);
        rng.fill_normal(&mut filler, 0.0, 1.0);
        l2_normalize(&mut role);
        l2_normalize(&mut filler);

        let bound = bind(&role, &filler).unwrap();
        let recovered = unbind(&role, &bound).unwrap();

        // Cosine similarity between recovered and original filler should be high
        let sim = cosine_similarity(&recovered, &filler);
        assert!(sim > 0.5, "bind-unbind cosine similarity too low: {sim}");
    }

    #[test]
    fn codebook_cleanup_finds_exact() {
        let mut rng = Rng::new(123);
        let dim = 64;
        let cb = HdcCodebook::new(16, dim, &mut rng);

        // Query with exact codebook entry should return that entry
        let query = cb.entries[7].clone();
        let (idx, sim) = cb.cleanup(&query).unwrap();
        assert_eq!(idx, 7, "expected index 7, got {idx}");
        assert!(sim > 0.99, "similarity too low: {sim}");
    }

    #[test]
    fn bundle_preserves_components() {
        let mut rng = Rng::new(99);
        let dim = 256;
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];
        rng.fill_normal(&mut a, 0.0, 1.0);
        rng.fill_normal(&mut b, 0.0, 1.0);
        l2_normalize(&mut a);
        l2_normalize(&mut b);

        let bundled = bundle(&[&a, &b]).unwrap();
        let sim_a = cosine_similarity(&bundled, &a);
        let sim_b = cosine_similarity(&bundled, &b);
        // Both components should have positive similarity with the bundle
        assert!(sim_a > 0.3, "sim_a too low: {sim_a}");
        assert!(sim_b > 0.3, "sim_b too low: {sim_b}");
    }

    #[test]
    fn positional_encode_identity_at_zero() {
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let enc = positional_encode(&v, 0).unwrap();
        assert_eq!(enc, v);
    }

    #[test]
    fn cosine_similarity_self_is_one() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity: {sim}");
    }
}

//! Deterministic pseudo-random number generation for Sophon-1.
//!
//! Uses a 64-bit xoshiro256** PRNG seeded from a fixed u64 seed.
//! No external rand crate is used. The xoshiro256** algorithm is a
//! well-studied, high-quality PRNG with 256 bits of state and a
//! period of 2^256 - 1.
//!
//! Reference: Blackman & Vigna 2019, "Scrambled Linear Pseudorandom
//! Number Generators".
//!
//! # Examples
//!
//! ```
//! use sophon_core::Rng;
//!
//! // Create RNG with fixed seed
//! let mut rng = Rng::new(42);
//!
//! // Generate random values
//! let u = rng.next_u64();
//! let f = rng.next_f32();
//! assert!(f >= 0.0 && f < 1.0);
//!
//! // Normal distribution
//! let n = rng.next_normal(0.0, 1.0);
//!
//! // Fill buffers
//! let mut buf = [0.0f32; 100];
//! rng.fill_uniform(&mut buf);
//! ```

// ---------------------------------------------------------------------------
// xoshiro256**
// ---------------------------------------------------------------------------

/// Deterministic 64-bit PRNG. All Sophon-1 stochastic operations that need
/// reproducibility must use this generator with a fixed seed.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create from a 64-bit seed. State is initialised by splitmix64.
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *si = z ^ (z >> 31);
        }
        Self { s }
    }

    /// Generate the next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use top 24 bits for mantissa precision.
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Normal sample via Box-Muller transform.
    pub fn next_normal(&mut self, mean: f32, std: f32) -> f32 {
        let u1 = self.next_f32().max(1e-38); // avoid log(0)
        let u2 = self.next_f32();
        let mag = std * (-2.0 * u1.ln()).sqrt();
        let phi = core::f32::consts::TAU * u2;
        mean + mag * phi.cos()
    }

    /// Fill a slice with standard normal samples.
    pub fn fill_normal(&mut self, buf: &mut [f32], mean: f32, std: f32) {
        for x in buf.iter_mut() {
            *x = self.next_normal(mean, std);
        }
    }

    /// Fill a slice with uniform [0, 1) samples.
    pub fn fill_uniform(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.next_f32();
        }
    }

    /// Kaiming uniform initialisation for a weight matrix of fan-in `fan`.
    /// Range: [-bound, +bound] where bound = sqrt(1 / fan).
    pub fn fill_kaiming_uniform(&mut self, buf: &mut [f32], fan: usize) {
        let bound = (1.0 / fan as f32).sqrt();
        for x in buf.iter_mut() {
            *x = self.next_f32() * 2.0 * bound - bound;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        let vals_1: Vec<u64> = (0..10).map(|_| r1.next_u64()).collect();
        let vals_2: Vec<u64> = (0..10).map(|_| r2.next_u64()).collect();
        assert_ne!(vals_1, vals_2);
    }

    #[test]
    fn normal_samples_reasonable() {
        let mut r = Rng::new(123);
        let samples: Vec<f32> = (0..1024).map(|_| r.next_normal(0.0, 1.0)).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.15, "mean={mean}");
        assert!((var - 1.0).abs() < 0.2, "var={var}");
    }

    #[test]
    fn f32_in_range() {
        let mut r = Rng::new(0);
        for _ in 0..10000 {
            let v = r.next_f32();
            assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
        }
    }
}

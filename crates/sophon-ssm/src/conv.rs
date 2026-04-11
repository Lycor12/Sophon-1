//! FFT convolution mode for batch-parallel SSM inference.
//!
//! For a fixed-Δ SSM, the recurrence h(t) = A_bar h(t-1) + B_bar u(t)
//! with output y(t) = C h(t) + D u(t) can be computed as a convolution:
//!   K[t] = C A_bar^t B_bar   (SSM kernel)
//!   y = K * u + D u           (causal convolution)
//!
//! Using FFT, this runs in O(T log T) instead of O(T N) for the sequential
//! scan, with much better parallelism.
//!
//! Novel optimisation — HKSF (Hybrid Kernel-Scan Fusion):
//!   For short sequences (T < threshold), sequential scan is faster.
//!   For long sequences, precompute kernel K via A_bar powers,
//!   FFT-convolve with input, then add feedthrough.
//!   The threshold is determined by T * N vs T * log2(T).

use sophon_config::{SSM_D, SSM_N, SSM_P};
use sophon_core::hdc::{fft_real, ifft_real, Complex};

use crate::params::SsmParams;
use crate::zoh::DiscretisedSsm;

/// Threshold: use conv mode when T > HKSF_THRESHOLD.
const HKSF_THRESHOLD: usize = 64;

/// Compute the SSM kernel K[t] = C * A_bar^t * B_bar for t = 0..T-1.
///
/// K has shape [T, P, D] stored as T vectors of length P*D (flattened).
/// For efficiency we compute per output dimension p and input dimension d.
///
/// Returns: Vec of T entries, each a Vec<f32> of length P (one input dim at a time).
fn compute_kernel(
    disc: &DiscretisedSsm,
    params: &SsmParams,
    seq_len: usize,
    input_dim: usize,
) -> Vec<Vec<f32>> {
    // K[t] for a single input dimension d:
    //   K[t][p] = sum_n C[p,n] * (A_bar^t * B_bar[:,d])[n]
    //
    // Let s[n] = B_bar[n, d] (the d-th column of B_bar).
    // At each t: s_new = A_bar * s, K[t][p] = sum_n C[p,n] * s[n].

    let c = &params.c; // [P, N]

    // Extract column d of B_bar
    let mut s = vec![0.0f32; SSM_N];
    for n in 0..SSM_N {
        s[n] = disc.b_bar[n * SSM_D + input_dim];
    }

    let mut kernel = Vec::with_capacity(seq_len);

    for _t in 0..seq_len {
        // K[t][p] = C * s
        let mut k_t = vec![0.0f32; SSM_P];
        for p in 0..SSM_P {
            let mut acc = 0.0f32;
            for n in 0..SSM_N {
                acc += c[p * SSM_N + n] * s[n];
            }
            k_t[p] = acc;
        }
        kernel.push(k_t);

        // s = A_bar * s (apply discretised state transition)
        s = disc.apply_a_bar(&s);
    }

    kernel
}

/// 1D causal convolution of kernel k[0..T] with input u[0..T].
/// Result[t] = sum_{s=0}^{t} k[s] * u[t-s].
///
/// Uses FFT for O(T log T) when T is large enough.
fn causal_conv_1d(k: &[f32], u: &[f32]) -> Vec<f32> {
    let t = k.len();
    assert_eq!(u.len(), t);

    if t == 0 {
        return vec![];
    }

    // Pad to next power of 2 for FFT, at least 2*T to avoid circular aliasing
    let fft_len = (2 * t).next_power_of_two();

    let mut k_padded = vec![0.0f32; fft_len];
    k_padded[..t].copy_from_slice(k);

    let mut u_padded = vec![0.0f32; fft_len];
    u_padded[..t].copy_from_slice(u);

    // FFT both
    let k_freq = fft_real(&k_padded, fft_len);
    let u_freq = fft_real(&u_padded, fft_len);

    // Pointwise multiply in frequency domain
    let mut y_freq: Vec<Complex> = Vec::with_capacity(k_freq.len());
    for i in 0..k_freq.len() {
        y_freq.push(k_freq[i].mul(u_freq[i]));
    }

    // IFFT
    let y_padded = ifft_real(&mut y_freq);

    // Take first T elements (causal part)
    y_padded[..t].to_vec()
}

/// Batch-parallel SSM forward using convolution mode.
///
/// Computes y[t] = sum_d K_d * u_d[t] + D * u[t] for all timesteps.
///
/// `inputs`: sequence of T input vectors, each of dimension D.
/// Returns: sequence of T output vectors, each of dimension P.
pub fn ssm_conv_forward(
    params: &SsmParams,
    disc: &DiscretisedSsm,
    inputs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let seq_len = inputs.len();
    if seq_len == 0 {
        return vec![];
    }

    // For short sequences, fall back to sequential scan
    if seq_len <= HKSF_THRESHOLD {
        return ssm_sequential_forward(params, disc, inputs);
    }

    // Initialise output
    let mut outputs: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![0.0f32; SSM_P]).collect();

    // For each input dimension d, compute kernel and convolve
    for d in 0..SSM_D {
        let kernel = compute_kernel(disc, params, seq_len, d);

        // Extract u[t][d] for all t
        let u_d: Vec<f32> = inputs.iter().map(|u| u[d]).collect();

        // Extract kernel per output dimension p
        for p in 0..SSM_P {
            let k_p: Vec<f32> = kernel.iter().map(|k_t| k_t[p]).collect();
            let conv_result = causal_conv_1d(&k_p, &u_d);
            for t in 0..seq_len {
                outputs[t][p] += conv_result[t];
            }
        }
    }

    // Add feedthrough: y[t] += D * u[t]
    for t in 0..seq_len {
        for p in 0..SSM_P {
            for d in 0..SSM_D {
                outputs[t][p] += params.d[p * SSM_D + d] * inputs[t][d];
            }
        }
    }

    outputs
}

/// Sequential SSM forward (for short sequences or when conv would be slower).
fn ssm_sequential_forward(
    params: &SsmParams,
    disc: &DiscretisedSsm,
    inputs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut state = crate::state::SsmState::new();
    let mut outputs = Vec::with_capacity(inputs.len());
    for u in inputs {
        let y = crate::update::ssm_step(&mut state, disc, params, u);
        outputs.push(y);
    }
    outputs
}

/// HKSF decision: should we use convolution mode?
pub fn should_use_conv(seq_len: usize) -> bool {
    seq_len > HKSF_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_conv_impulse_response() {
        // Convolving with a delta should return the kernel
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let u = vec![1.0, 0.0, 0.0, 0.0]; // impulse
        let y = causal_conv_1d(&k, &u);
        for i in 0..4 {
            assert!(
                (y[i] - k[i]).abs() < 0.01,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                k[i]
            );
        }
    }

    #[test]
    fn causal_conv_commutative_for_same_length() {
        let a = vec![1.0, 0.5, 0.25, 0.125];
        let b = vec![1.0, -1.0, 0.5, -0.5];
        let y1 = causal_conv_1d(&a, &b);
        let y2 = causal_conv_1d(&b, &a);
        for i in 0..4 {
            assert!(
                (y1[i] - y2[i]).abs() < 0.01,
                "y1[{}]={}, y2[{}]={}",
                i,
                y1[i],
                i,
                y2[i]
            );
        }
    }

    #[test]
    fn sequential_forward_shapes() {
        let params = SsmParams::new_stable(42);
        let disc = DiscretisedSsm::from_params(&params);
        let inputs: Vec<Vec<f32>> = (0..10).map(|_| vec![0.1f32; SSM_D]).collect();
        let outputs = ssm_sequential_forward(&params, &disc, &inputs);
        assert_eq!(outputs.len(), 10);
        for y in &outputs {
            assert_eq!(y.len(), SSM_P);
        }
    }

    #[test]
    fn conv_forward_shapes() {
        let params = SsmParams::new_stable(42);
        let disc = DiscretisedSsm::from_params(&params);
        // Use > HKSF_THRESHOLD to trigger conv path
        let inputs: Vec<Vec<f32>> = (0..128).map(|_| vec![0.01f32; SSM_D]).collect();
        let outputs = ssm_conv_forward(&params, &disc, &inputs);
        assert_eq!(outputs.len(), 128);
        for y in &outputs {
            assert_eq!(y.len(), SSM_P);
        }
    }

    #[test]
    fn conv_forward_finite() {
        let params = SsmParams::new_stable(42);
        let disc = DiscretisedSsm::from_params(&params);
        let inputs: Vec<Vec<f32>> = (0..128).map(|_| vec![0.01f32; SSM_D]).collect();
        let outputs = ssm_conv_forward(&params, &disc, &inputs);
        for (t, y) in outputs.iter().enumerate() {
            for &val in y {
                assert!(val.is_finite(), "non-finite output at t={}", t);
            }
        }
    }

    #[test]
    fn hksf_threshold() {
        assert!(!should_use_conv(10));
        assert!(!should_use_conv(64));
        assert!(should_use_conv(65));
        assert!(should_use_conv(1024));
    }

    #[test]
    fn empty_inputs() {
        let params = SsmParams::new_stable(42);
        let disc = DiscretisedSsm::from_params(&params);
        let outputs = ssm_conv_forward(&params, &disc, &[]);
        assert!(outputs.is_empty());
    }
}

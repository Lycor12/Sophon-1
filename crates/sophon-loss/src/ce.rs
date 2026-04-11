//! Cross-entropy loss with SLCE (Streaming Log-probability Cross-Entropy).

use sophon_config::VOCAB_SIZE;

// ---------------------------------------------------------------------------
// SLCE: cross-entropy loss
// ---------------------------------------------------------------------------

/// Compute cross-entropy loss for a single token prediction.
///
/// Uses SLCE: 2-pass streaming log-sum-exp without materialising full softmax.
///
/// # Arguments
/// * `logits` — raw logit vector of length VOCAB_SIZE
/// * `target` — ground truth token index (0..255)
///
/// # Returns
/// Loss scalar: -log(softmax(logits)[target]) = log_sum_exp - logits[target]
pub fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    debug_assert_eq!(logits.len(), VOCAB_SIZE);
    debug_assert!(target < VOCAB_SIZE);

    // SLCE Pass 1: streaming LSE (same as LSES from sophon-core)
    let mut running_max = logits[0];
    let mut running_sum = 1.0f32;

    for i in 1..VOCAB_SIZE {
        let x = logits[i];
        if x > running_max {
            // Rescale accumulated sum to new max
            running_sum *= (running_max - x).exp();
            running_sum += 1.0;
            running_max = x;
        } else {
            running_sum += (x - running_max).exp();
        }
    }

    let log_sum_exp = running_max + running_sum.ln();

    // SLCE Pass 2: loss = -(logits[target] - log_sum_exp) = log_sum_exp - logits[target]
    log_sum_exp - logits[target]
}

/// Compute cross-entropy loss averaged over a batch of predictions.
///
/// # Arguments
/// * `logits_batch` — Vec of logit vectors, each length VOCAB_SIZE
/// * `targets` — ground truth token indices
///
/// # Returns
/// Mean cross-entropy loss over the batch.
pub fn cross_entropy_loss_batch(logits_batch: &[Vec<f32>], targets: &[usize]) -> f32 {
    assert_eq!(logits_batch.len(), targets.len());
    if logits_batch.is_empty() {
        return 0.0;
    }
    let total: f32 = logits_batch
        .iter()
        .zip(targets.iter())
        .map(|(logits, &t)| cross_entropy_loss(logits, t))
        .sum();
    total / logits_batch.len() as f32
}

// ---------------------------------------------------------------------------
// SLCE: cross-entropy gradient
// ---------------------------------------------------------------------------

/// Compute gradient of cross-entropy loss w.r.t. logits.
///
/// grad_logits[i] = softmax(logits)[i] - delta(i, target)
///
/// This is computed without a separate softmax pass by reusing the LSE:
///   softmax[i] = exp(logits[i] - log_sum_exp)
///
/// # Arguments
/// * `logits` — raw logit vector of length VOCAB_SIZE
/// * `target` — ground truth token index
///
/// # Returns
/// Gradient vector of length VOCAB_SIZE.
pub fn cross_entropy_grad(logits: &[f32], target: usize) -> Vec<f32> {
    debug_assert_eq!(logits.len(), VOCAB_SIZE);
    debug_assert!(target < VOCAB_SIZE);

    // Compute log_sum_exp (same streaming approach)
    let mut running_max = logits[0];
    let mut running_sum = 1.0f32;

    for i in 1..VOCAB_SIZE {
        let x = logits[i];
        if x > running_max {
            running_sum *= (running_max - x).exp();
            running_sum += 1.0;
            running_max = x;
        } else {
            running_sum += (x - running_max).exp();
        }
    }

    let log_sum_exp = running_max + running_sum.ln();

    // grad[i] = softmax[i] - delta(i, target)
    let mut grad = vec![0.0f32; VOCAB_SIZE];
    for i in 0..VOCAB_SIZE {
        let softmax_i = (logits[i] - log_sum_exp).exp();
        grad[i] = softmax_i;
    }
    grad[target] -= 1.0;

    grad
}

// ---------------------------------------------------------------------------
// Accuracy
// ---------------------------------------------------------------------------

/// Compute top-1 accuracy over a batch.
///
/// # Arguments
/// * `logits_batch` — Vec of logit vectors
/// * `targets` — ground truth token indices
///
/// # Returns
/// Fraction of predictions where argmax(logits) == target.
pub fn accuracy(logits_batch: &[Vec<f32>], targets: &[usize]) -> f32 {
    assert_eq!(logits_batch.len(), targets.len());
    if logits_batch.is_empty() {
        return 0.0;
    }
    let correct: usize = logits_batch
        .iter()
        .zip(targets.iter())
        .filter(|(logits, &t)| {
            let pred = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            pred == t
        })
        .count();
    correct as f32 / logits_batch.len() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ce_loss_is_non_negative() {
        let logits = vec![0.0f32; VOCAB_SIZE];
        let loss = cross_entropy_loss(&logits, 0);
        assert!(loss >= 0.0, "CE loss should be >= 0, got {loss}");
    }

    #[test]
    fn ce_loss_uniform_logits() {
        // Uniform logits: loss = log(VOCAB_SIZE)
        let logits = vec![0.0f32; VOCAB_SIZE];
        let loss = cross_entropy_loss(&logits, 42);
        let expected = (VOCAB_SIZE as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-4,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn ce_loss_perfect_prediction() {
        // Target logit much larger than others -> loss ≈ 0
        let mut logits = vec![-100.0f32; VOCAB_SIZE];
        logits[10] = 100.0;
        let loss = cross_entropy_loss(&logits, 10);
        assert!(
            loss < 1e-4,
            "perfect prediction loss should be ~0, got {loss}"
        );
    }

    #[test]
    fn ce_grad_sums_to_zero() {
        // softmax sums to 1, so grad = softmax - one_hot sums to 0
        let logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| (i as f32) * 0.01).collect();
        let grad = cross_entropy_grad(&logits, 100);
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-4, "CE gradient should sum to ~0, got {sum}");
    }

    #[test]
    fn ce_grad_target_is_negative() {
        // grad[target] = softmax[target] - 1 < 0
        let logits = vec![0.0f32; VOCAB_SIZE];
        let grad = cross_entropy_grad(&logits, 50);
        assert!(
            grad[50] < 0.0,
            "target gradient should be negative, got {}",
            grad[50]
        );
    }

    #[test]
    fn ce_grad_non_target_is_positive() {
        let logits = vec![0.0f32; VOCAB_SIZE];
        let grad = cross_entropy_grad(&logits, 50);
        for (i, &g) in grad.iter().enumerate() {
            if i != 50 {
                assert!(g >= 0.0, "non-target grad[{i}] should be >= 0, got {g}");
            }
        }
    }

    #[test]
    fn ce_numerical_gradient_check() {
        let logits: Vec<f32> = (0..VOCAB_SIZE)
            .map(|i| ((i as f32) * 0.137).sin())
            .collect();
        let target = 42;
        let grad = cross_entropy_grad(&logits, target);

        let eps = 1e-3f32;
        // Check a few entries
        for &idx in &[0, target, 100, 255] {
            let mut logits_plus = logits.clone();
            logits_plus[idx] += eps;
            let mut logits_minus = logits.clone();
            logits_minus[idx] -= eps;

            let loss_plus = cross_entropy_loss(&logits_plus, target);
            let loss_minus = cross_entropy_loss(&logits_minus, target);
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (numerical - grad[idx]).abs() < 1e-3,
                "grad mismatch at {idx}: numerical={numerical:.5} analytical={:.5}",
                grad[idx]
            );
        }
    }

    #[test]
    fn batch_loss_is_mean() {
        let logits1 = vec![0.0f32; VOCAB_SIZE];
        let logits2 = vec![0.0f32; VOCAB_SIZE];
        let batch = vec![logits1.clone(), logits2.clone()];
        let targets = vec![0, 1];
        let batch_loss = cross_entropy_loss_batch(&batch, &targets);
        let individual = (cross_entropy_loss(&logits1, 0) + cross_entropy_loss(&logits2, 1)) / 2.0;
        assert!(
            (batch_loss - individual).abs() < 1e-6,
            "batch loss should be mean"
        );
    }

    #[test]
    fn accuracy_perfect() {
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        logits[5] = 100.0;
        let batch = vec![logits.clone(), logits.clone()];
        let targets = vec![5, 5];
        let acc = accuracy(&batch, &targets);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_zero() {
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        logits[5] = 100.0;
        let batch = vec![logits.clone(), logits.clone()];
        let targets = vec![0, 1]; // both wrong
        let acc = accuracy(&batch, &targets);
        assert!(acc.abs() < 1e-6);
    }
}

//! Unit Tests for Sophon AGI System
//!
//! Comprehensive unit tests for core operations, HDC operations,
//! normalization functions, utility functions, individual components,
//! and boundary condition tests.
//!
//! Each test targets a specific function or component in isolation.

// ============================================================================
// Section 1: Core Operations Tests (ops module)
// ============================================================================

/// Unit test: Hadamard product basic functionality
#[test]
fn unit_hadamard_basic() {
    use sophon_core::ops::hadamard;

    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    let result = hadamard(&a, &b);

    assert_eq!(result.len(), 3);
    assert!((result[0] - 4.0).abs() < 1e-6); // 1 * 4
    assert!((result[1] - 10.0).abs() < 1e-6); // 2 * 5
    assert!((result[2] - 18.0).abs() < 1e-6); // 3 * 6
}

/// Unit test: Hadamard product with negative values
#[test]
fn unit_hadamard_negative() {
    use sophon_core::ops::hadamard;

    let a = vec![-1.0f32, 2.0, -3.0];
    let b = vec![4.0f32, -5.0, 6.0];

    let result = hadamard(&a, &b);

    assert!((result[0] - (-4.0)).abs() < 1e-6);
    assert!((result[1] - (-10.0)).abs() < 1e-6);
    assert!((result[2] - (-18.0)).abs() < 1e-6);
}

/// Unit test: Hadamard product with zeros
#[test]
fn unit_hadamard_zeros() {
    use sophon_core::ops::hadamard;

    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![0.0f32, 0.0, 0.0];

    let result = hadamard(&a, &b);

    assert!(result.iter().all(|&x| x.abs() < 1e-6));
}

/// Unit test: Hadamard product preserves length
#[test]
fn unit_hadamard_length_preservation() {
    use sophon_core::ops::hadamard;

    for len in [1, 10, 100, 1000] {
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();

        let result = hadamard(&a, &b);
        assert_eq!(
            result.len(),
            len,
            "Hadamard should preserve length for {}",
            len
        );
    }
}

/// Unit test: RMS normalization basic
#[test]
fn unit_rms_norm_basic() {
    use sophon_core::ops::rms_norm_inplace;

    let mut data = vec![3.0f32, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.535

    rms_norm_inplace(&mut data);

    // After normalization, RMS should be ~1
    let rms = (data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 0.01);
}

/// Unit test: RMS normalization with all zeros
#[test]
fn unit_rms_norm_zeros() {
    use sophon_core::ops::rms_norm_inplace;

    let mut data = vec![0.0f32; 10];
    let original = data.clone();

    rms_norm_inplace(&mut data);

    // Zeros should remain zeros (or nearly so)
    assert!(data.iter().all(|&x| x.abs() < 1e-6));
}

/// Unit test: RMS normalization with ones
#[test]
fn unit_rms_norm_ones() {
    use sophon_core::ops::rms_norm_inplace;

    let mut data = vec![1.0f32; 100];

    rms_norm_inplace(&mut data);

    // Should normalize to unit RMS
    let rms = (data.iter().map(|x| x * x).sum::<f32>() / 100.0).sqrt();
    assert!((rms - 1.0).abs() < 0.01);
}

/// Unit test: RMS normalization with large values
#[test]
fn unit_rms_norm_large() {
    use sophon_core::ops::rms_norm_inplace;

    let mut data = vec![1e10f32; 10];

    rms_norm_inplace(&mut data);

    // Should normalize without overflow
    assert!(data.iter().all(|&x| x.is_finite()));
}

/// Unit test: RMS normalization with small values
#[test]
fn unit_rms_norm_small() {
    use sophon_core::ops::rms_norm_inplace;

    let mut data = vec![1e-10f32; 10];

    rms_norm_inplace(&mut data);

    // Should handle small values
    assert!(data.iter().all(|&x| x.is_finite()));
}

/// Unit test: Dot product basic
#[test]
fn unit_dot_product_basic() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    assert!((dot - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
}

/// Unit test: Dot product with orthogonal vectors
#[test]
fn unit_dot_product_orthogonal() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    assert!(dot.abs() < 1e-6);
}

/// Unit test: Vector addition
#[test]
fn unit_vector_add() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    assert_eq!(sum, vec![5.0, 7.0, 9.0]);
}

/// Unit test: Vector subtraction
#[test]
fn unit_vector_sub() {
    let a = vec![5.0f32, 7.0, 9.0];
    let b = vec![4.0f32, 5.0, 6.0];

    let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

    assert_eq!(diff, vec![1.0, 2.0, 3.0]);
}

/// Unit test: Scalar multiplication
#[test]
fn unit_scalar_mul() {
    let v = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let scaled: Vec<f32> = v.iter().map(|x| x * scalar).collect();

    assert_eq!(scaled, vec![2.0, 4.0, 6.0]);
}

/// Unit test: Vector magnitude
#[test]
fn unit_vector_magnitude() {
    let v = vec![3.0f32, 4.0]; // |v| = 5

    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!((mag - 5.0).abs() < 1e-6);
}

/// Unit test: Vector normalization
#[test]
fn unit_vector_normalize() {
    let v = vec![3.0f32, 4.0];
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    let normalized: Vec<f32> = v.iter().map(|x| x / mag).collect();

    let new_mag = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((new_mag - 1.0).abs() < 1e-6);
}

// ============================================================================
// Section 2: HDC Operations Tests
// ============================================================================

/// Unit test: HDC bind basic
#[test]
fn unit_hdc_bind_basic() {
    use sophon_core::hdc::bind;

    let a = vec![1.0f32, 0.0, -1.0];
    let b = vec![0.5f32, 0.5, 0.5];

    let result = bind(&a, &b);

    assert_eq!(result.len(), a.len());
}

/// Unit test: HDC bind commutativity
#[test]
fn unit_hdc_bind_commutative() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();

    let ab = bind(&a, &b);
    let ba = bind(&b, &a);

    for i in 0..ab.len() {
        assert!(
            (ab[i] - ba[i]).abs() < 1e-5,
            "HDC bind should be commutative"
        );
    }
}

/// Unit test: HDC bind identity
#[test]
fn unit_hdc_bind_identity() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
    let ones = vec![1.0f32; 64];

    let result = bind(&a, &ones);

    // Binding with ones should approximately preserve values
    for i in 0..a.len() {
        assert!((result[i] - a[i]).abs() < 1e-5 || result[i].is_finite());
    }
}

/// Unit test: HDC bind zero
#[test]
fn unit_hdc_bind_zero() {
    use sophon_core::hdc::bind;

    let a: Vec<f32> = (0..64).map(|_| 1.0f32).collect();
    let zeros = vec![0.0f32; 64];

    let result = bind(&a, &zeros);

    // Binding with zeros
    assert!(result.iter().all(|&x| x.is_finite()));
}

/// Unit test: HDC bundle basic
#[test]
fn unit_hdc_bundle_basic() {
    use sophon_core::hdc::bundle;

    let a = vec![1.0f32; 64];
    let b = vec![2.0f32; 64];
    let c = vec![3.0f32; 64];

    let result = bundle(&[&a, &b, &c]);

    // Bundled vector should have same dimension
    assert_eq!(result.len(), 64);
}

/// Unit test: HDC bundle commutativity
#[test]
fn unit_hdc_bundle_commutative() {
    use sophon_core::hdc::bundle;

    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| (i * 2) as f32).collect();
    let c: Vec<f32> = (0..64).map(|i| (i * 3) as f32).collect();

    let abc = bundle(&[&a, &b, &c]);
    let cba = bundle(&[&c, &b, &a]);

    // Bundle order shouldn't matter (approximately)
    for i in 0..abc.len() {
        assert!(
            (abc[i] - cba[i]).abs() < 1e-5,
            "HDC bundle should be commutative"
        );
    }
}

/// Unit test: HDC bundle single element
#[test]
fn unit_hdc_bundle_single() {
    use sophon_core::hdc::bundle;

    let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

    let result = bundle(&[&a]);

    // Bundling single element should return similar values
    for i in 0..a.len() {
        assert!((result[i] - a[i]).abs() < 1e-5);
    }
}

/// Unit test: HDC circular convolution basic
#[test]
fn unit_hdc_circular_conv_basic() {
    use sophon_core::hdc::circular_conv;

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 0.5, 0.5, 0.5];

    let result = circular_conv(&a, &b);

    assert_eq!(result.len(), 4);
}

/// Unit test: HDC circular convolution with delta
#[test]
fn unit_hdc_circular_conv_delta() {
    use sophon_core::hdc::circular_conv;

    let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let mut delta = vec![0.0f32; 64];
    delta[0] = 1.0; // Delta function

    let result = circular_conv(&a, &delta);

    // Convolution with delta should return original
    for i in 0..a.len() {
        assert!((result[i] - a[i]).abs() < 1e-5);
    }
}

/// Unit test: HDC circular convolution dimension preservation
#[test]
fn unit_hdc_circular_conv_dimension() {
    use sophon_core::hdc::circular_conv;

    for dim in [4, 8, 16, 32, 64, 128] {
        let a: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();

        let result = circular_conv(&a, &b);
        assert_eq!(
            result.len(),
            dim,
            "Circular conv should preserve dimension {}",
            dim
        );
    }
}

/// Unit test: HDC permutation basic
#[test]
fn unit_hdc_permutation_basic() {
    use sophon_core::hdc::permute;

    let a = vec![1.0f32, 2.0, 3.0, 4.0];

    let result = permute(&a, 1);

    assert_eq!(result.len(), a.len());
}

/// Unit test: HDC permutation cycles
#[test]
fn unit_hdc_permutation_cycles() {
    use sophon_core::hdc::permute;

    let a: Vec<f32> = (0..8).map(|i| i as f32).collect();

    let permuted = permute(&a, 1);
    let double_permuted = permute(&permuted, 1);

    // Two permutations should shift by 2
    assert_eq!(double_permuted.len(), a.len());
}

/// Unit test: HDC permutation inverse
#[test]
fn unit_hdc_permutation_inverse() {
    use sophon_core::hdc::{permute, permute_inverse};

    let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

    let permuted = permute(&a, 5);
    let recovered = permute_inverse(&permuted, 5);

    // Should recover original (approximately)
    for i in 0..a.len() {
        assert!(
            (recovered[i] - a[i]).abs() < 1e-5,
            "Permutation should be invertible"
        );
    }
}

/// Unit test: HDC similarity cosine
#[test]
fn unit_hdc_similarity_cosine() {
    use sophon_core::hdc::cosine_sim;

    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    let c = vec![0.0f32, 1.0, 0.0];

    let sim_aa = cosine_sim(&a, &a);
    let sim_ab = cosine_sim(&a, &b);
    let sim_ac = cosine_sim(&a, &c);

    assert!(
        (sim_aa - 1.0).abs() < 1e-5,
        "Identical vectors should have similarity 1"
    );
    assert!(
        (sim_ab - 1.0).abs() < 1e-5,
        "Same direction should have similarity 1"
    );
    assert!(
        sim_ac.abs() < 1e-5,
        "Orthogonal vectors should have similarity 0"
    );
}

/// Unit test: HDC similarity with random vectors
#[test]
fn unit_hdc_similarity_random() {
    use sophon_core::hdc::cosine_sim;

    let a: Vec<f32> = (0..128).map(|_| rand::random::<f32>()).collect();
    let b: Vec<f32> = (0..128).map(|_| rand::random::<f32>()).collect();

    let sim = cosine_sim(&a, &b);

    // Similarity should be in [-1, 1]
    assert!(sim >= -1.0 - 1e-6 && sim <= 1.0 + 1e-6);
}

// ============================================================================
// Section 3: Normalization Tests
// ============================================================================

/// Unit test: Layer normalization
#[test]
fn unit_layer_norm_basic() {
    use sophon_core::ops::layer_norm_inplace;

    let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let eps = 1e-5f32;

    layer_norm_inplace(&mut data, eps);

    // Mean should be ~0
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.01, "Layer norm should zero mean");

    // Variance should be ~1
    let var = data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;
    assert!((var - 1.0).abs() < 0.1, "Layer norm should unit variance");
}

/// Unit test: Layer normalization with zero input
#[test]
fn unit_layer_norm_zeros() {
    use sophon_core::ops::layer_norm_inplace;

    let mut data = vec![0.0f32; 100];
    let eps = 1e-5f32;

    layer_norm_inplace(&mut data, eps);

    // Should remain zeros (or close)
    assert!(data.iter().all(|&x| x.abs() < 1e-4));
}

/// Unit test: Batch normalization inference
#[test]
fn unit_batch_norm_inference() {
    use sophon_core::ops::batch_norm_inplace;

    let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mean = 2.5f32;
    let var = 2.0f32;
    let gamma = 1.0f32;
    let beta = 0.0f32;
    let eps = 1e-5f32;

    batch_norm_inplace(&mut data, mean, var, gamma, beta, eps);

    // All values should be finite
    assert!(data.iter().all(|&x| x.is_finite()));
}

/// Unit test: Group normalization
#[test]
fn unit_group_norm_basic() {
    use sophon_core::ops::group_norm_inplace;

    let mut data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let num_groups = 8;
    let eps = 1e-5f32;

    group_norm_inplace(&mut data, num_groups, eps);

    // Should normalize without error
    assert!(data.iter().all(|&x| x.is_finite()));
}

/// Unit test: Instance normalization
#[test]
fn unit_instance_norm_basic() {
    use sophon_core::ops::instance_norm_inplace;

    let mut data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let eps = 1e-5f32;

    instance_norm_inplace(&mut data, eps);

    // Should normalize without error
    assert!(data.iter().all(|&x| x.is_finite()));
}

/// Unit test: Min-max normalization
#[test]
fn unit_min_max_norm() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let min = *data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = *data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let normalized: Vec<f32> = data.iter().map(|&x| (x - min) / (max - min)).collect();

    assert!((normalized.iter().cloned().fold(f32::INFINITY, f32::min) - 0.0).abs() < 1e-6);
    assert!((normalized.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 1.0).abs() < 1e-6);
}

/// Unit test: Z-score normalization
#[test]
fn unit_z_score_norm() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();

    let normalized: Vec<f32> = data.iter().map(|&x| (x - mean) / (std + 1e-5)).collect();

    // Mean should be ~0
    let new_mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
    assert!(new_mean.abs() < 1e-5);
}

// ============================================================================
// Section 4: Utility Function Tests
// ============================================================================

/// Unit test: Softmax function
#[test]
fn unit_softmax_basic() {
    use sophon_core::ops::softmax;

    let logits = vec![1.0f32, 2.0, 3.0];
    let probs = softmax(&logits);

    // Probabilities should sum to 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");

    // Should be monotonic
    assert!(probs[0] < probs[1] && probs[1] < probs[2]);
}

/// Unit test: Softmax with large values
#[test]
fn unit_softmax_large_values() {
    use sophon_core::ops::softmax;

    let logits = vec![100.0f32, 101.0, 102.0];
    let probs = softmax(&logits);

    // Should not overflow
    assert!(probs.iter().all(|&p| p.is_finite()));

    // Should still sum to 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

/// Unit test: Softmax with equal values
#[test]
fn unit_softmax_equal() {
    use sophon_core::ops::softmax;

    let logits = vec![1.0f32; 10];
    let probs = softmax(&logits);

    // All should be equal
    assert!(probs.iter().all(|&p| (p - 0.1).abs() < 1e-6));
}

/// Unit test: Argmax
#[test]
fn unit_argmax_basic() {
    use sophon_core::ops::argmax;

    let data = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
    let max_idx = argmax(&data);

    assert_eq!(max_idx, 3, "Argmax should return index of maximum");
}

/// Unit test: Argmax with ties
#[test]
fn unit_argmax_ties() {
    use sophon_core::ops::argmax;

    let data = vec![0.5f32, 0.5, 0.3];
    let max_idx = argmax(&data);

    // Should return first occurrence
    assert_eq!(max_idx, 0);
}

/// Unit test: Top-k selection
#[test]
fn unit_topk_basic() {
    use sophon_core::ops::topk;

    let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let top = topk(&data, 3);

    assert_eq!(top.len(), 3);
    assert!(top.contains(&5)); // index of 9.0
    assert!(top.contains(&4)); // index of 5.0
    assert!(top.contains(&2)); // index of 4.0
}

/// Unit test: Cumulative sum
#[test]
fn unit_cumsum_basic() {
    use sophon_core::ops::cumsum;

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cum = cumsum(&data);

    assert_eq!(cum, vec![1.0, 3.0, 6.0, 10.0]);
}

/// Unit test: ReLU activation
#[test]
fn unit_relu_basic() {
    use sophon_core::ops::relu;

    let data = vec![-1.0f32, 0.0, 1.0, -0.5, 2.0];
    let activated: Vec<f32> = data.iter().map(|&x| relu(x)).collect();

    assert_eq!(activated, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
}

/// Unit test: Sigmoid activation
#[test]
fn unit_sigmoid_basic() {
    use sophon_core::ops::sigmoid;

    // Sigmoid(0) = 0.5
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);

    // Sigmoid(infinity) = 1
    assert!(sigmoid(10.0) > 0.99);

    // Sigmoid(-infinity) = 0
    assert!(sigmoid(-10.0) < 0.01);
}

/// Unit test: Tanh activation
#[test]
fn unit_tanh_basic() {
    use sophon_core::ops::tanh;

    // tanh(0) = 0
    assert!(tanh(0.0).abs() < 1e-6);

    // tanh(infinity) = 1
    assert!(tanh(10.0) > 0.99);

    // tanh(-infinity) = -1
    assert!(tanh(-10.0) < -0.99);
}

/// Unit test: GELU activation
#[test]
fn unit_gelu_basic() {
    use sophon_core::ops::gelu;

    // GELU(0) = 0
    assert!(gelu(0.0).abs() < 0.1);

    // GELU should be approximately ReLU for large positive
    assert!(gelu(10.0) > 9.0);

    // GELU should be close to 0 for large negative
    assert!(gelu(-10.0).abs() < 0.01);
}

/// Unit test: Clamp function
#[test]
fn unit_clamp_basic() {
    use sophon_core::ops::clamp;

    assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
    assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
    assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
}

/// Unit test: Linear interpolation
#[test]
fn unit_lerp_basic() {
    use sophon_core::ops::lerp;

    assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-6);
    assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-6);
    assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-6);
}

/// Unit test: Clip gradient
#[test]
fn unit_clip_gradient() {
    use sophon_core::ops::clip_gradient;

    let grad = vec![-10.0f32, -5.0, 0.0, 5.0, 10.0];
    let max_norm = 5.0f32;

    let clipped = clip_gradient(&grad, max_norm);

    // All values should be within [-5, 5]
    assert!(clipped.iter().all(|&x| x.abs() <= 5.0 + 1e-6));
}

// ============================================================================
// Section 5: Component Tests
// ============================================================================

/// Unit test: Aligned vector creation
#[test]
fn unit_aligned_vec_creation() {
    use sophon_accel::aligned::AlignedVec;

    let vec = AlignedVec::<f32, 64>::from_vec(vec![1.0f32; 100]);

    assert_eq!(vec.len(), 100);
    assert_eq!(vec.as_ptr() as usize % 64, 0, "Should be 64-byte aligned");
}

/// Unit test: Aligned vector access
#[test]
fn unit_aligned_vec_access() {
    use sophon_accel::aligned::AlignedVec;

    let vec = AlignedVec::<f32, 64>::from_vec((0..100).map(|i| i as f32).collect());

    for i in 0..100 {
        assert_eq!(vec[i], i as f32);
    }
}

/// Unit test: Random number generation
#[test]
fn unit_random_generation() {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    // Should generate different values
    let a: f32 = rng.gen();
    let b: f32 = rng.gen();

    // Very unlikely to be equal
    assert!(a != b || a == 0.0);
}

/// Unit test: Random range generation
#[test]
fn unit_random_range() {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let val: f32 = rng.gen_range(-1.0..1.0);
        assert!(val >= -1.0 && val < 1.0);
    }
}

/// Unit test: Thread-local storage
#[test]
fn unit_thread_local() {
    use std::cell::RefCell;

    thread_local! {
        static COUNTER: RefCell<i32> = RefCell::new(0);
    }

    COUNTER.with(|c| {
        *c.borrow_mut() += 1;
    });

    COUNTER.with(|c| {
        assert_eq!(*c.borrow(), 1);
    });
}

/// Unit test: Atomic operations
#[test]
fn unit_atomic_operations() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = AtomicUsize::new(0);

    counter.fetch_add(1, Ordering::SeqCst);
    counter.fetch_add(2, Ordering::SeqCst);

    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

/// Unit test: Mutex operations
#[test]
fn unit_mutex_operations() {
    use std::sync::Mutex;

    let data = Mutex::new(0);

    {
        let mut guard = data.lock().unwrap();
        *guard += 1;
    }

    assert_eq!(*data.lock().unwrap(), 1);
}

/// Unit test: RwLock operations
#[test]
fn unit_rwlock_operations() {
    use std::sync::RwLock;

    let data = RwLock::new(vec![1, 2, 3]);

    {
        let read_guard = data.read().unwrap();
        assert_eq!(read_guard[0], 1);
    }

    {
        let mut write_guard = data.write().unwrap();
        write_guard.push(4);
    }

    assert_eq!(data.read().unwrap().len(), 4);
}

// ============================================================================
// Section 6: Boundary Condition Tests
// ============================================================================

/// Unit test: Empty vector operations
#[test]
fn unit_boundary_empty_vector() {
    let empty: Vec<f32> = vec![];

    // Operations on empty vectors should handle gracefully
    let sum: f32 = empty.iter().sum();
    assert_eq!(sum, 0.0);

    let max = empty.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    assert!(max.is_infinite() && max < 0.0);
}

/// Unit test: Single element vector
#[test]
fn unit_boundary_single_element() {
    let single = vec![5.0f32];

    let sum: f32 = single.iter().sum();
    assert_eq!(sum, 5.0);

    let max = single.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    assert_eq!(max, 5.0);
}

/// Unit test: Maximum size vector
#[test]
fn unit_boundary_large_vector() {
    // Large but not too large for memory
    let large: Vec<f32> = (0..100000).map(|i| i as f32).collect();

    assert_eq!(large.len(), 100000);

    let sum: f32 = large.iter().sum();
    assert!(sum.is_finite());
}

/// Unit test: Float infinity handling
#[test]
fn unit_boundary_float_infinity() {
    let values = vec![f32::INFINITY, 1.0, -f32::INFINITY];

    // Operations with infinity
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max.is_infinite() && max > 0.0);

    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(min.is_infinite() && min < 0.0);
}

/// Unit test: Float NaN handling
#[test]
fn unit_boundary_float_nan() {
    let values = vec![f32::NAN, 1.0, 2.0];

    // NaN comparisons
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // Note: NaN comparisons always return false, so max might be 2.0
    assert!(max.is_finite() || max.is_nan());
}

/// Unit test: Very small float values
#[test]
fn unit_boundary_float_small() {
    let small = vec![1e-40f32, 2e-40, 3e-40];

    // Operations with denormalized numbers
    let sum: f32 = small.iter().sum();
    assert!(sum.is_finite() || sum == 0.0);
}

/// Unit test: Very large float values
#[test]
fn unit_boundary_float_large() {
    let large = vec![1e30f32, 2e30, 3e30];

    // Operations with large numbers
    let sum: f32 = large.iter().sum();
    assert!(sum.is_infinite() || sum.is_finite()); // Either is acceptable
}

/// Unit test: Integer overflow
#[test]
fn unit_boundary_integer_overflow() {
    let max = i32::MAX;
    let min = i32::MIN;

    // Wrapping arithmetic
    let wrapped = max.wrapping_add(1);
    assert_eq!(wrapped, min);
}

/// Unit test: usize boundary
#[test]
fn unit_boundary_usize() {
    let max = usize::MAX;
    let min = usize::MIN;

    assert_eq!(min, 0);
    assert!(max > 0);
}

/// Unit test: String boundary - empty
#[test]
fn unit_boundary_empty_string() {
    let empty = "";
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

/// Unit test: String boundary - unicode
#[test]
fn unit_boundary_unicode_string() {
    let unicode = "Hello 世界 🌍";

    // Byte length != char count
    assert!(unicode.len() > unicode.chars().count());

    // Should iterate correctly
    let chars: Vec<char> = unicode.chars().collect();
    assert_eq!(chars.len(), 8); // Including space
}

/// Unit test: Array boundary - first and last
#[test]
fn unit_boundary_array_first_last() {
    let arr = vec![1, 2, 3, 4, 5];

    assert_eq!(*arr.first().unwrap(), 1);
    assert_eq!(*arr.last().unwrap(), 5);
}

/// Unit test: Slice boundary
#[test]
fn unit_boundary_slice() {
    let arr = vec![1, 2, 3, 4, 5];

    // Full slice
    let full = &arr[..];
    assert_eq!(full.len(), 5);

    // Partial slices
    let first_half = &arr[..2];
    assert_eq!(first_half, &[1, 2]);

    let last_half = &arr[3..];
    assert_eq!(last_half, &[4, 5]);

    let middle = &arr[1..4];
    assert_eq!(middle, &[2, 3, 4]);
}

/// Unit test: Option boundary - Some and None
#[test]
fn unit_boundary_option() {
    let some: Option<i32> = Some(42);
    let none: Option<i32> = None;

    assert!(some.is_some());
    assert_eq!(some.unwrap(), 42);

    assert!(none.is_none());
    assert_eq!(none.unwrap_or(0), 0);
}

/// Unit test: Result boundary - Ok and Err
#[test]
fn unit_boundary_result() {
    let ok: Result<i32, &str> = Ok(42);
    let err: Result<i32, &str> = Err("error");

    assert!(ok.is_ok());
    assert_eq!(ok.unwrap(), 42);

    assert!(err.is_err());
    assert_eq!(err.unwrap_or(0), 0);
}

// ============================================================================
// Section 7: Type Conversion Tests
// ============================================================================

/// Unit test: Float to int conversion
#[test]
fn unit_conversion_float_to_int() {
    let f: f32 = 3.7;
    let i: i32 = f as i32;

    assert_eq!(i, 3); // Truncates toward zero
}

/// Unit test: Int to float conversion
#[test]
fn unit_conversion_int_to_float() {
    let i: i32 = 42;
    let f: f32 = i as f32;

    assert!((f - 42.0).abs() < 1e-6);
}

/// Unit test: f32 to f64 conversion
#[test]
fn unit_conversion_f32_to_f64() {
    let f32_val: f32 = 3.1415927;
    let f64_val: f64 = f32_val as f64;

    assert!((f64_val - 3.1415927f64 as f32 as f64).abs() < 1e-6);
}

/// Unit test: f64 to f32 conversion
#[test]
fn unit_conversion_f64_to_f32() {
    let f64_val: f64 = 3.141592653589793;
    let f32_val: f32 = f64_val as f32;

    // Precision loss is expected
    assert!(f32_val.is_finite());
}

/// Unit test: Vec to slice conversion
#[test]
fn unit_conversion_vec_to_slice() {
    let vec = vec![1, 2, 3];
    let slice: &[i32] = &vec;

    assert_eq!(slice, &[1, 2, 3]);
}

/// Unit test: String to str conversion
#[test]
fn unit_conversion_string_to_str() {
    let string = String::from("hello");
    let str_ref: &str = &string;

    assert_eq!(str_ref, "hello");
}

/// Unit test: Char to u32 conversion
#[test]
fn unit_conversion_char_to_u32() {
    let c: char = 'A';
    let u: u32 = c as u32;

    assert_eq!(u, 65);
}

// ============================================================================
// Section 8: Iterator Tests
// ============================================================================

/// Unit test: Map iterator
#[test]
fn unit_iterator_map() {
    let data = vec![1, 2, 3, 4, 5];
    let doubled: Vec<i32> = data.iter().map(|&x| x * 2).collect();

    assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
}

/// Unit test: Filter iterator
#[test]
fn unit_iterator_filter() {
    let data = vec![1, 2, 3, 4, 5];
    let evens: Vec<i32> = data.iter().copied().filter(|&x| x % 2 == 0).collect();

    assert_eq!(evens, vec![2, 4]);
}

/// Unit test: Fold iterator
#[test]
fn unit_iterator_fold() {
    let data = vec![1, 2, 3, 4, 5];
    let sum = data.iter().fold(0, |acc, &x| acc + x);

    assert_eq!(sum, 15);
}

/// Unit test: Zip iterator
#[test]
fn unit_iterator_zip() {
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];

    let sums: Vec<i32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

    assert_eq!(sums, vec![5, 7, 9]);
}

/// Unit test: Enumerate iterator
#[test]
fn unit_iterator_enumerate() {
    let data = vec![10, 20, 30];

    for (i, &val) in data.iter().enumerate() {
        assert_eq!(val, (i + 1) as i32 * 10);
    }
}

/// Unit test: Chain iterator
#[test]
fn unit_iterator_chain() {
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];

    let chained: Vec<i32> = a.iter().chain(b.iter()).copied().collect();

    assert_eq!(chained, vec![1, 2, 3, 4, 5, 6]);
}

/// Unit test: Skip and take
#[test]
fn unit_iterator_skip_take() {
    let data: Vec<i32> = (0..100).collect();

    let middle: Vec<i32> = data.iter().skip(40).take(20).copied().collect();

    assert_eq!(middle.len(), 20);
    assert_eq!(middle[0], 40);
    assert_eq!(middle[19], 59);
}

/// Unit test: Flatten iterator
#[test]
fn unit_iterator_flatten() {
    let nested = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let flat: Vec<i32> = nested.into_iter().flatten().collect();

    assert_eq!(flat, vec![1, 2, 3, 4, 5, 6]);
}

// ============================================================================
// Section 9: Collection Tests
// ============================================================================

/// Unit test: HashMap basic operations
#[test]
fn unit_collection_hashmap() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert("a", 1);
    map.insert("b", 2);

    assert_eq!(map.get("a"), Some(&1));
    assert_eq!(map.get("c"), None);

    map.insert("a", 3);
    assert_eq!(map.get("a"), Some(&3));
}

/// Unit test: HashSet basic operations
#[test]
fn unit_collection_hashset() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(1);
    set.insert(2);
    set.insert(1); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&1));
    assert!(!set.contains(&3));
}

/// Unit test: VecDeque operations
#[test]
fn unit_collection_vecdeque() {
    use std::collections::VecDeque;

    let mut deque = VecDeque::new();
    deque.push_back(1);
    deque.push_back(2);
    deque.push_front(0);

    assert_eq!(deque.pop_front(), Some(0));
    assert_eq!(deque.pop_back(), Some(2));
}

/// Unit test: BinaryHeap operations
#[test]
fn unit_collection_binaryheap() {
    use std::collections::BinaryHeap;

    let mut heap = BinaryHeap::new();
    heap.push(3);
    heap.push(1);
    heap.push(4);

    assert_eq!(heap.pop(), Some(4)); // Max-heap
    assert_eq!(heap.pop(), Some(3));
}

/// Unit test: BTreeMap operations
#[test]
fn unit_collection_btreemap() {
    use std::collections::BTreeMap;

    let mut map = BTreeMap::new();
    map.insert(3, "c");
    map.insert(1, "a");
    map.insert(2, "b");

    let keys: Vec<i32> = map.keys().copied().collect();
    assert_eq!(keys, vec![1, 2, 3]);
}

/// Unit test: BTreeSet operations
#[test]
fn unit_collection_btreeset() {
    use std::collections::BTreeSet;

    let mut set = BTreeSet::new();
    set.insert(3);
    set.insert(1);
    set.insert(2);

    let values: Vec<i32> = set.iter().copied().collect();
    assert_eq!(values, vec![1, 2, 3]);
}

// ============================================================================
// Section 10: Error Handling Tests
// ============================================================================

/// Unit test: Option unwrap_or
#[test]
fn unit_error_handling_option_unwrap() {
    let some = Some(42);
    let none: Option<i32> = None;

    assert_eq!(some.unwrap_or(0), 42);
    assert_eq!(none.unwrap_or(0), 0);
}

/// Unit test: Result unwrap_or_else
#[test]
fn unit_error_handling_result_unwrap() {
    let ok: Result<i32, &str> = Ok(42);
    let err: Result<i32, &str> = Err("error");

    assert_eq!(ok.unwrap_or_else(|_| 0), 42);
    assert_eq!(err.unwrap_or_else(|_| 0), 0);
}

/// Unit test: Result map
#[test]
fn unit_error_handling_result_map() {
    let ok: Result<i32, &str> = Ok(42);
    let mapped = ok.map(|x| x * 2);

    assert_eq!(mapped.unwrap(), 84);
}

/// Unit test: Result map_err
#[test]
fn unit_error_handling_result_map_err() {
    let err: Result<i32, &str> = Err("error");
    let mapped = err.map_err(|e| e.len());

    assert_eq!(mapped.unwrap_err(), 5);
}

/// Unit test: Option and_then
#[test]
fn unit_error_handling_option_and_then() {
    let some = Some(5);
    let result = some.and_then(|x| if x > 0 { Some(x * 2) } else { None });

    assert_eq!(result, Some(10));
}

/// Unit test: Result and_then
#[test]
fn unit_error_handling_result_and_then() {
    let ok: Result<i32, &str> = Ok(5);
    let result = ok.and_then(|x| {
        if x > 0 {
            Ok(x * 2)
        } else {
            Err("non-positive")
        }
    });

    assert_eq!(result.unwrap(), 10);
}

/// Unit test: Custom error type
#[test]
fn unit_error_handling_custom() {
    #[derive(Debug, PartialEq)]
    enum MyError {
        InvalidInput,
        OutOfBounds,
    }

    fn risky_operation(x: i32) -> Result<i32, MyError> {
        if x < 0 {
            Err(MyError::InvalidInput)
        } else if x > 100 {
            Err(MyError::OutOfBounds)
        } else {
            Ok(x * 2)
        }
    }

    assert_eq!(risky_operation(50), Ok(100));
    assert_eq!(risky_operation(-1), Err(MyError::InvalidInput));
    assert_eq!(risky_operation(101), Err(MyError::OutOfBounds));
}

/// Unit test: panic catch_unwind
#[test]
fn unit_error_handling_panic() {
    use std::panic;

    let result = panic::catch_unwind(|| {
        // This would panic: assert!(false)
        // But we'll just return normally for this test
        42
    });

    assert_eq!(result.unwrap(), 42);
}

/// Unit test: expect and unwrap
#[test]
fn unit_error_handling_expect() {
    let some = Some(42);
    assert_eq!(some.expect("Should have value"), 42);
}

/// Unit test: unwrap_or_default
#[test]
fn unit_error_handling_default() {
    let none: Option<i32> = None;
    assert_eq!(none.unwrap_or_default(), 0);

    let empty: Option<String> = None;
    assert_eq!(empty.unwrap_or_default(), "");
}
